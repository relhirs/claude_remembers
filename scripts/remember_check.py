#!/usr/bin/env python3
"""
remember_check.py - The Memory Check: is search finding the right memories?

Runs by itself in the background every couple of weeks, once there is enough
history. It builds a set of questions from YOUR OWN past sessions, asks them
the ways memory search can work, and reports in plain words how often each
way brought back the right conversation. You don't write questions, run
anything, or review anything. The result shows up once at the start of a later
session, and you can read it any time with `python3 scripts/remember_check.py`.

How the questions are made: each session's notes list the decisions made in it.
One decision per session is turned into the kind of question you might ask
weeks later ("where did we move the API key?"), reworded so it does not just
repeat the note. That uses one small Claude Haiku call through the same API key
the session summaries already use. With no key it falls back to plain templates
("What did we decide about ...?"), which reuse the note's words and are easier,
so the report says which style was used. A few questions about things you never
discussed check that search stays quiet when it has nothing.

Honest limits: nobody reviews these questions. Two things keep them fair: a
decision is only used if its own session really contains the answer (checked
by meaning against the session's messages), and a second session that recorded
a near-identical decision also counts as correct. On this project's history the
result landed within a few points of a hand-checked set. With only a couple of
dozen questions, one question still moves a score by ~4 points, so the
comparison between modes is the useful part. Hand-checked numbers come from
remember_eval.py with a golden set.

Usage:
    python3 scripts/remember_check.py            # show the latest result (runs one if none yet)
    python3 scripts/remember_check.py --now      # run a fresh check now
    python3 scripts/remember_check.py --auto     # what startup uses: run only if due, quietly
    python3 scripts/remember_check.py --json
Turn it off with `memory_check: enabled: false` in config.yaml.
"""

import argparse
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

from config_utils import resolve_db_path
from memory_check_state import (FAILURE_BACKOFF_HOURS, acquire_run_lock, lock_active, lock_path, log,  # noqa: F401
                                log_path, post_notice, read_state, release_run_lock, state_path,
                                take_notice, write_state)   # a few are re-exported for callers and tests
from metrics_utils import ensure_metrics_schema, utc_now

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIN_SESSIONS = 12            # sessions with recorded decisions needed for a meaningful check
MIN_DAYS_BETWEEN = 14
MIN_NEW_SESSIONS = 10
MAX_QUESTIONS = 30
MIN_BULLET_CHARS = 40
# A decision only makes a fair question if its own session really contains the
# answer. Measured on this project's history: decisions whose best-matching
# message in their session scored under ~0.45 (cosine) were missed by every
# search method, because the answer was never said in the conversation text.
SUPPORT_MIN = 0.45
# Two sessions that recorded essentially the same decision both answer it.
DUPLICATE_DECISION_SIM = 0.7
ALWAYS_RAM_MB = 610
# Questions about things nobody would have discussed in a coding-assistant
# history. If a user's memory really does contain one, that question is
# simply noisy; the report counts across all of them.
NO_ANSWER_QUESTIONS = [
    "what is the recipe for the sourdough starter",
    "what time is the dentist appointment",
    "which airline did we pick for the honeymoon flight",
    "what did the landlord say about the lease renewal",
    "how many laps was the marathon training plan",
    "which paint colour did we choose for the nursery",
]


def check_enabled(config):
    cfg = (config or {}).get("memory_check")
    if isinstance(cfg, dict) and cfg.get("enabled") is False:
        return False
    sem = (config or {}).get("semantic_search") or {}
    return sem.get("enabled") is not False


# ---------------------------------------------------------------------------
# Building the questions
# ---------------------------------------------------------------------------

def _session_decisions(notes):
    """The usable decision bullets in one session's notes, markdown stripped."""
    match = re.search(r"## Decisions\n(.*?)(?:\n## |\Z)", notes or "", re.S)
    if not match:
        return []
    found = []
    for line in match.group(1).splitlines():
        text = line.strip()
        if not text.startswith("- "):
            continue
        text = re.sub(r"[*`]", "", text[2:]).strip()
        if len(text) >= MIN_BULLET_CHARS and not text.lower().startswith(("none", "no decisions", "no changes")):
            found.append(text)
    return found


def _sessions_with_notes(conn):
    return conn.execute(
        "SELECT session_id, project, started_at, notes FROM sys_sessions "
        "WHERE notes IS NOT NULL AND TRIM(notes) != '' AND notes NOT LIKE 'AUTO-GENERATED FALLBACK%' "
        "ORDER BY started_at").fetchall()


def decision_bullets(conn):
    """One usable decision per session that has real (not fallback) notes, oldest first."""
    out = []
    for session_id, project, started_at, notes in _sessions_with_notes(conn):
        found = _session_decisions(notes)
        if found:                          # one per session keeps the questions spread out
            out.append({"session": session_id, "project": project, "started": started_at, "text": found[0]})
    return out


def all_decisions(conn):
    """Every usable decision in every session, as (session_id, text)."""
    return [(sid, text) for sid, _, _, notes in _sessions_with_notes(conn) for text in _session_decisions(notes)]


def _unit_rows(blobs):
    import numpy as np
    matrix = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def answer_support(conn, vector, session_id):
    """Best cosine between a decision's vector and any embedded message of its session."""
    rows = conn.execute("SELECT te.embedding FROM transcript_embeddings te JOIN transcripts t "
                        "ON t.id = te.transcript_id WHERE t.session_id = ?", (session_id,)).fetchall()
    if not rows:
        return 0.0
    return float((_unit_rows([r[0] for r in rows]) @ vector).max())


def vet_decisions(conn, config, bullets):
    """Keep only decisions whose session really contains the answer, and attach
    any other sessions that recorded a near-identical decision (they are also
    correct). Returns (kept, dropped_count, vetted). With no embedding model
    (or an unreadable embedding) it returns the decisions unchanged and
    vetted=False, so the report can say the fairness filter didn't run."""
    try:
        import numpy as np
        from semantic_search import get_embed_model
        model = get_embed_model(config)
        pool = all_decisions(conn)
        pool_vectors = np.asarray(model.encode([t for _, t in pool], normalize_embeddings=True))
    except Exception:
        return bullets, 0, False
    try:
        kept, dropped = [], 0
        for b in bullets:
            vector = np.asarray(model.encode(b["text"], normalize_embeddings=True))
            if answer_support(conn, vector, b["session"]) < SUPPORT_MIN:
                dropped += 1
                continue
            sims = pool_vectors @ vector
            twins = sorted({sid for (sid, _), sim in zip(pool, sims)
                            if sid != b["session"] and sim >= DUPLICATE_DECISION_SIM})
            kept.append({**b, "also_answers": twins})
        return kept, dropped, True
    except Exception:
        return bullets, 0, False


def pick_evenly(items, n):
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def template_question(text):
    words = text.split()
    return f"What did we decide about {' '.join(words[:10]).rstrip('.,;:')}?"


# Each decision is asked two different ways: one unlucky wording can move a
# score by several points, two averages the luck out.
QUESTIONS_PER_DECISION = 2
QUESTION_PROMPT = (
    "You help test a memory system. Below are numbered notes about decisions a person made "
    "while working with an AI assistant. For each, write TWO short questions that person might "
    "ask weeks later when they half-remember it, like messages typed quickly. People who half-"
    "remember a decision still remember its distinctive terms (a name, a file, a setting, a "
    "number), so keep one or two of those, but phrase the rest in your own words instead of "
    "copying the note. Make the two questions phrased differently from each other. "
    "Reply with ONLY a JSON array with one entry per note, in the same order, each entry being "
    "an array of the two question strings."
)


def llm_questions(config, texts):
    """Reword notes into natural questions with one Haiku call. Returns
    (questions, reason): a list the same length as `texts`, each entry a list of
    question strings, with reason "ok"; or (None, why) when templates must be
    used, so the report can name the real cause instead of guessing."""
    try:
        import anthropic
    except ImportError:
        return None, "the anthropic package is not installed"
    try:
        sum_cfg = (config.get("summarization") or {})
        api_key = sum_cfg.get("api_key") or None
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        numbered = "\n".join(f"{i}. {t[:300]}" for i, t in enumerate(texts, 1))
        response = client.messages.create(
            model=sum_cfg.get("model", "claude-haiku-4-5"), max_tokens=4096,
            system=QUESTION_PROMPT, messages=[{"role": "user", "content": numbered}])
    except Exception as e:
        return None, f"the API call failed ({type(e).__name__})"
    if getattr(response, "stop_reason", None) == "max_tokens":
        return None, "the reply was cut off"
    try:
        text = "".join(b.text for b in response.content if b.type == "text")
        found = json.loads(text[text.index("["): text.rindex("]") + 1])
    except ValueError:
        return None, "the reply was not valid JSON"
    if not isinstance(found, list) or len(found) != len(texts):
        return None, "the reply did not have one entry per decision"
    cleaned = []
    for entry in found:
        entry = [entry] if isinstance(entry, str) else entry
        good = [q.strip() for q in entry if isinstance(q, str) and 8 <= len(q.strip()) <= 300]
        if not good:
            return None, "the reply had an empty or malformed question"
        cleaned.append(good[:QUESTIONS_PER_DECISION])
    return cleaned, "ok"


def build_questions(conn, config, n=MAX_QUESTIONS, use_llm=True):
    """(questions for remember_eval, info). Decisions that are not actually
    answerable from their own session are dropped first, so a miss means search
    failed, not that the question was unfair. `info["style"]` is "reworded" or
    "templated"; the rest counts what was filtered."""
    candidates = decision_bullets(conn)
    vetted, dropped, vetting_ran = vet_decisions(conn, config, candidates)
    chosen = pick_evenly(vetted, n)
    worded, why = llm_questions(config, [c["text"] for c in chosen]) if (use_llm and chosen) else (None, "not attempted")
    questions = []
    for i, c in enumerate(chosen):
        phrasings = worded[i] if worded else [template_question(c["text"])]
        for j, text in enumerate(phrasings):
            questions.append({"id": f"d{i + 1:02d}{'abcdefgh'[j]}", "kind": "answerable",
                              "expect": [c["session"], *c.get("also_answers", [])], "question": text})
    for i, q in enumerate(NO_ANSWER_QUESTIONS, 1):
        questions.append({"id": f"n{i:02d}", "kind": "no_answer", "expect": [], "question": q})
    info = {"style": "reworded" if worded else "templated", "wording_note": why,
            "vetting_ran": vetting_ran, "decisions_used": len(chosen),
            "candidates": len(candidates),
            "dropped_unanswerable": dropped,
            "with_twin_sessions": sum(1 for c in chosen if c.get("also_answers"))}
    return questions, info


# ---------------------------------------------------------------------------
# When to run
# ---------------------------------------------------------------------------

def sessions_with_decisions(conn):
    return len(decision_bullets(conn))


def is_due(conn, root, config, now=None):
    """(due?, reason). Cheap: the time and lock checks come first, and the one
    query is a plain count unless it could matter. No model, no network."""
    if not check_enabled(config):
        return False, "disabled"
    now = now or time.time()
    if lock_active(root, now):
        return False, "already running"
    state = read_state(root)
    if state.get("last_failed_at") and now - state["last_failed_at"] < FAILURE_BACKOFF_HOURS * 3600:
        return False, "failed recently, waiting before retrying"
    last = state.get("last_run_at")
    if last is not None and now - last < MIN_DAYS_BETWEEN * 86400:
        return False, "ran recently"
    # Cheap upper bound first: sessions whose notes have a Decisions section at all.
    try:
        rough = conn.execute("SELECT COUNT(*) FROM sys_sessions WHERE notes LIKE '%## Decisions%'").fetchone()[0]
    except sqlite3.Error:
        rough = MIN_SESSIONS
    if rough < MIN_SESSIONS:
        return False, f"not enough history yet ({rough} of {MIN_SESSIONS} sessions with recorded decisions)"
    count = sessions_with_decisions(conn)
    if count < MIN_SESSIONS:
        return False, f"not enough history yet ({count} of {MIN_SESSIONS} sessions with recorded decisions)"
    if last is None:
        return True, "first check"
    if count - state.get("last_session_count", 0) < MIN_NEW_SESSIONS:
        return False, "not enough new history since the last check"
    return True, "due"


def launch_background(root=ROOT, python=None):
    """Start `--auto` detached so a session start never waits on it. Never raises."""
    try:
        cmd = [python or sys.executable, str(pathlib.Path(__file__).resolve()), "--auto"]
        quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: outlive the hook
            subprocess.Popen(cmd, cwd=str(root), creationflags=0x00000008 | 0x00000200, **quiet)
        else:
            subprocess.Popen(cmd, cwd=str(root), start_new_session=True, **quiet)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------

def _tenths(x):
    return f"{round(x * 10)} out of 10"


def recommend(result, current_mode):
    """Plain-language advice from the numbers. Returns {"advice", "headline", "current_mode"}."""
    m = result["methods"]
    base, on_miss, always = m["keyword"], m["on_miss"], m["always"]
    cur = "always" if current_mode == "always" else "on_miss"
    if result.get("question_style") == "templated":
        # Template questions reuse the notes' own words, so every method looks
        # great and the gaps vanish. Don't pretend that tells us anything.
        why = (result.get("question_info") or {}).get("wording_note") or "it was unavailable"
        return {"advice": "unreliable", "current_mode": cur,
                "headline": (f"This run used quick template questions because Claude Haiku couldn't word them ({why}). "
                             "Those are easier than real life, so it can't tell whether changing settings would help. "
                             "If this keeps happening, make sure an Anthropic API key is available (ANTHROPIC_API_KEY, "
                             "or summarization.api_key in config.yaml) and run `python3 scripts/remember_check.py --now`.")}
    gain_always = always["recall_at_5"] - base["recall_at_5"]
    extra = always["recall_at_5"] - on_miss["recall_at_5"]
    if gain_always <= 0.05:
        advice = "keep"
        headline = (f"Meaning-based search adds very little on your history (keyword search alone already finds "
                    f"{_tenths(base['recall_at_5'])}), so there is nothing to gain from changing your settings.")
    elif extra >= 0.10 and cur == "on_miss":
        advice = "consider_always"
        headline = (f"Running meaning search on every prompt would find noticeably more of your past work "
                    f"({_tenths(always['recall_at_5'])} instead of {_tenths(on_miss['recall_at_5'])}). "
                    f"It costs about {ALWAYS_RAM_MB} MB of memory while you work. Set `mode: always` in config.yaml if that is fine.")
    elif extra < 0.05 and cur == "always":
        advice = "consider_on_miss"
        headline = (f"'always' finds about the same as 'on_miss' on your history ({_tenths(always['recall_at_5'])} vs "
                    f"{_tenths(on_miss['recall_at_5'])}). You could set `mode: on_miss` and stop using {ALWAYS_RAM_MB} MB.")
    else:
        advice = "keep"
        headline = (f"Your current setting ('{cur}') is a good fit: it finds {_tenths(m[cur]['recall_at_5'])} "
                    "of your past work.")
    return {"advice": advice, "headline": headline, "current_mode": cur}


def render_report(result, config_mode="on_miss"):
    m, n = result["methods"], result["counts"]
    rec = recommend(result, config_mode)
    info = result.get("question_info") or {}
    when = (result.get("created_at") or "")[:10]
    style = result.get("question_style", "reworded")
    junk_q = n["no_answer"]
    used = info.get("decisions_used")
    if used and used != n["answerable"]:
        how = (f" ({used} decisions, each asked two ways)" if n["answerable"] == 2 * used
               else f" ({used} decisions, asked in one or two ways each)")
    else:
        how = ""
    L = ["=" * 66, f"  MEMORY CHECK  ({when})", "=" * 66,
         f"I asked {n['answerable']} questions about your own past work{how} and checked whether search",
         "brought back the right conversation. " + (
             "The questions were reworded so they don't just repeat your notes." if style == "reworded" else
             "The questions are simple templates that reuse your notes' wording, so scores run high."),
         "", "HOW OFTEN IT FOUND THE RIGHT CONVERSATION"]
    labels = (("keyword", "Keyword search alone"),
              ("on_miss", "Meaning search only when keyword is unsure (on_miss)"),
              ("always", "Meaning search on every prompt (always)"))
    for key, label in labels:
        mark = "   <- your setting" if key == rec["current_mode"] else ""
        L.append(f"  {label:<58} {_tenths(m[key]['recall_at_5'])}{mark}")
    L += ["", "WHEN YOU ASK ABOUT SOMETHING YOU NEVER DISCUSSED",
          f"  Search brought back unrelated memories on {round(m['keyword']['false_injection_rate'] * junk_q)} of {junk_q} "
          f"questions (keyword alone) and {round(m['always']['false_injection_rate'] * junk_q)} of {junk_q} (with meaning search).",
          "", "WHAT THIS MEANS", f"  {rec['headline']}",
          "  Nothing is broken, and nothing needs to be done unless you want to change a setting.",
          "", f"(Technical: recall@5 keyword {m['keyword']['recall_at_5']}, on_miss {m['on_miss']['recall_at_5']}, "
          f"always {m['always']['recall_at_5']}; meaning search loads the model on {round((m['on_miss'].get('model_load_rate') or 0) * 100)}% "
          "of these questions, which are phrased about the past, so everyday prompts trigger it less.)",
          f"These questions are generated automatically ({n['answerable']} of them), so one question moves a "
          f"score by about {round(100 / max(n['answerable'], 1))} points"
          + ("; the same decision asked twice counts twice, so the real sample is smaller. " if used and used != n["answerable"] else "; ")
          + "trust the comparison more than the exact numbers."]
    if info.get("vetting_ran") is False:
        L.append("(The fairness filter that removes questions whose answer was never actually said couldn't run this "
                 "time, so a few misses may be unfair questions rather than search failures.)")
    elif info.get("dropped_unanswerable"):
        L.append(f"({info['dropped_unanswerable']} decisions were left out because their own session never actually "
                 "said the answer, so a miss here means search failed, not that the question was unfair.)")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def run_check(config, root=ROOT, label="memory-check"):
    """Build questions from history, score them, save the result and the state.
    Returns the result dict, or {"skipped": reason}.

    The run lock is taken first (before the slow question-building), so two
    sessions can't both do the work, and a failure is logged and backed off
    instead of being retried on every session start."""
    if not acquire_run_lock(root):
        return {"skipped": "another check is already running"}
    try:
        from remember_eval import run_questions
        db_path = resolve_db_path(config)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            questions, info = build_questions(conn, config)
            count = sessions_with_decisions(conn)
        finally:
            conn.close()
        # Count decisions, not questions: each decision is asked twice.
        if info["decisions_used"] < MIN_SESSIONS:
            state = read_state(root)       # wait the normal interval instead of retrying every start
            state.update({"last_run_at": time.time(), "last_session_count": count})
            write_state(root, state)
            return {"skipped": f"only {info['decisions_used']} usable decisions so far"}
        result = run_questions(config, [], questions, source="memory_check")
        result["question_style"] = info["style"]
        result["question_info"] = info
        mode = ((config.get("semantic_search") or {}).get("mode") or "on_miss")
        rec = recommend(result, str(mode).strip().lower())
        result["recommendation"] = rec
        wconn = sqlite3.connect(db_path, timeout=10.0)
        try:
            ensure_metrics_schema(wconn)
            wconn.execute("INSERT INTO metrics_snapshots (created_at, kind, label, git_commit, params, results) "
                          "VALUES (?, 'memory_check', ?, ?, ?, ?)",
                          (result["created_at"], label, result.get("git_commit"),
                           json.dumps({"answerable": result["counts"]["answerable"], **info}),
                           json.dumps(result)))
            wconn.commit()
        finally:
            wconn.close()
        state = read_state(root)
        state.update({"last_run_at": time.time(), "last_session_count": count})
        for stale in ("last_failed_at", "last_error", "notice", "notified", "running_since"):
            state.pop(stale, None)         # last two are leftovers from an earlier design
        write_state(root, state)
        post_notice(root, "Memory Check finished. " + rec["headline"]
                    + " (Full report: python3 scripts/remember_check.py)")
        return result
    except Exception as e:
        import traceback
        log(root, f"check failed: {e!r}\n{traceback.format_exc()}")
        state = read_state(root)
        state.update({"last_failed_at": time.time(), "last_error": repr(e)[:300]})
        write_state(root, state)
        raise
    finally:
        release_run_lock(root)


def latest_saved(config):
    conn = sqlite3.connect(f"file:{resolve_db_path(config)}?mode=ro", uri=True, timeout=5.0)
    try:
        row = conn.execute("SELECT results FROM metrics_snapshots WHERE kind = 'memory_check' "
                           "ORDER BY id DESC LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def main():
    parser = argparse.ArgumentParser(description="Is memory search finding the right memories?")
    parser.add_argument("--now", action="store_true", help="run a fresh check now")
    parser.add_argument("--auto", action="store_true", help="run only if due (what startup uses)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = resolve_db_path(config)
    if not os.path.exists(db_path):
        print(f"FATAL: database not found at {db_path}")
        sys.exit(2)
    mode = str((config.get("semantic_search") or {}).get("mode") or "on_miss").strip().lower()

    if args.auto:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            due, _ = is_due(conn, ROOT, config)
        finally:
            conn.close()
        if due:
            try:
                run_check(config, ROOT, label="auto")
            except Exception:
                pass                        # run_check already logged it to logs/memory_check.log
        return

    result = None if args.now else latest_saved(config)
    if result is None:
        if not check_enabled(config):
            print("The Memory Check is turned off in config.yaml.")
            return
        print("Running the Memory Check (about 15 seconds)...", file=sys.stderr)
        result = run_check(config, ROOT, label="manual")
        if "skipped" in result:
            print(f"The Memory Check didn't run: {result['skipped']}.")
            if "decisions" in result["skipped"]:
                print(f"It needs about {MIN_SESSIONS} past sessions with recorded decisions.")
            return
    print(json.dumps(result, indent=2) if args.json else render_report(result, mode))


if __name__ == "__main__":
    main()
