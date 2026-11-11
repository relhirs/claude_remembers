#!/usr/bin/env python3
"""
remember_eval.py - Score memory search against a golden set of questions.

A golden set is a list of questions whose answer you already know lives in a
specific past session (plus some questions with NO answer in memory). This
script asks each question four ways and reports, in plain language, how often
the right session shows up and how often it injects junk when it shouldn't:

    keyword          keyword search only, with its weak-match gate
    keyword_ungated  the same search without the gate (what ran before it)
    meaning          meaning-based (embedding) search only, cosine cutoff 0.35
    on_miss          keyword, plus meaning search only when the gate says
                     keyword can't be trusted (config: mode: on_miss)
    always           keyword plus meaning search on every prompt, merged by
                     Reciprocal Rank Fusion (config: mode: always)

Read-only: it never writes to the transcript tables. The only write is saving
the finished result to metrics_snapshots (--no-save skips that).

The golden set lives in eval/golden_set.yaml (personal, gitignored):

    exclude_sessions: [<session ids that quote the questions, e.g. today's>]
    questions:
      - id: q01
        kind: answerable           # or no_answer
        question: "where is the anthropic key stored now"
        expect: [<session ids that hold the answer; any of them counts>]

Known limits, printed in the report: a question is "right" if ANY message from
an expected session comes back (session-level, not exact-message); the search
is run with no current-project boost; and a rival session that also answers a
question counts against it unless you add that session to `expect`.

Usage:
    python3 scripts/remember_eval.py
    python3 scripts/remember_eval.py --golden eval/golden_candidates.yaml
    python3 scripts/remember_eval.py --label before-gate --json
"""

import argparse
import importlib.util
import json
import os
import pathlib
import sqlite3
import statistics
import sys
import time

import yaml

from config_utils import is_machine_generated, resolve_db_path
from metrics_utils import ensure_metrics_schema, utc_now
from remember_baseline import git_stamp, load_hook

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.yaml"
SEMANTIC_CUTOFF = 0.35
RRF_K = 60
CANDIDATES = 30            # meaning-search rows considered before the cutoff
METHODS = ("keyword", "keyword_ungated", "meaning", "on_miss", "always")
METHOD_LABELS = {
    "keyword": "Keyword search only",
    "keyword_ungated": "Keyword, without the weak-match gate",
    "meaning": "Meaning-based search only",
    "on_miss": "Meaning search only when keyword looks unreliable (mode: on_miss)",
    "always": "Keyword + meaning on every prompt (mode: always)",
}


def load_golden(path):
    data = yaml.safe_load(pathlib.Path(path).read_text())
    if isinstance(data, list):
        data = {"questions": data}
    return data.get("exclude_sessions") or [], data["questions"]


def _rank_sessions(rows):
    """Rows (best first) -> distinct sessions in order of first appearance."""
    seen, out = set(), []
    for _, session in rows:
        if session and session not in seen:
            seen.add(session)
            out.append(session)
    return out


def _rrf(*ranked_lists):
    """Reciprocal Rank Fusion: each list votes 1/(k + rank) for a row; best total wins."""
    score = {}
    for lst in ranked_lists:
        for rank, (row_id, session) in enumerate(lst, 1):
            entry = score.setdefault(row_id, [0.0, session])
            entry[0] += 1.0 / (RRF_K + rank)
    return [(rid, s) for rid, (_, s) in sorted(score.items(), key=lambda kv: -kv[1][0])]


def _clean(question, rows):
    """Drop rows that merely quote the question (the questions themselves get
    written into transcripts when they are discussed)."""
    q = question.lower().strip()
    return [r for r in rows if q not in (r["content"] or "").lower()]


def run_question(hook, semantic, conn, config, db_path, item, exclude, cold_start_ms=0.0):
    """Ask one question every way, through the hook's own functions so the
    eval measures what the live hook would do. `cold_start_ms` is charged to
    on_miss whenever its gate loads the model."""
    question = item["question"]
    quoted = question.lower().strip()
    exclude_one = exclude[0] if exclude else None
    session_of = {}

    def keyword_rows(keywords, gated, info=None):
        if not keywords:
            return [], 0.0
        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
        started = time.perf_counter()
        raw = hook._keyword_search(conn, fts_query, None, exclude_session=exclude_one,
                                   gated=gated, info=info)
        ms = (time.perf_counter() - started) * 1000
        rows = []
        for row in raw:                      # (project, timestamp, content, rank, id)
            found_session = conn.execute("SELECT session_id FROM transcripts WHERE id = ?",
                                         (row[4],)).fetchone()
            session = found_session[0] if found_session else None
            if session in exclude or quoted in (row[2] or "").lower():
                continue
            session_of[row[4]] = session
            rows.append(row)
        return rows, ms

    keywords = hook.build_search_keywords(question)
    kw_info = {}
    kw_rows, kw_ms = keyword_rows(keywords, gated=True, info=kw_info)
    ug_rows, ug_ms = keyword_rows(keywords, gated=False)

    started = time.perf_counter()
    found = semantic.semantic_search(conn, config, question, project=None, limit=CANDIDATES)
    sem_ms = (time.perf_counter() - started) * 1000
    usable = []
    for r in found:
        session_of[r["id"]] = r["session_id"]
        if r["session_id"] not in exclude and quoted not in (r["content"] or "").lower():
            usable.append(r)
    sem_rows = hook.semantic_rows(usable, None, exclude_one)
    best_similarity = round(usable[0]["similarity"], 3) if usable else None

    always_rows = hook.merge_results(kw_rows, sem_rows)
    fires, reason = hook.should_use_semantic(question, keywords, kw_rows, kw_info, conn)
    on_miss_rows = always_rows if fires else kw_rows

    def pack(rows, ms, **extra):
        return {"sessions": _rank_sessions([(r[4], session_of.get(r[4])) for r in rows]),
                "injected": len(rows), "ms": ms, **extra}

    return {
        "keyword": pack(kw_rows, kw_ms, keywords=keywords),
        "keyword_ungated": pack(ug_rows, ug_ms, keywords=keywords),
        "meaning": pack(sem_rows[:hook.MAX_RESULTS], sem_ms, best_similarity=best_similarity),
        "on_miss": pack(on_miss_rows, kw_ms + (sem_ms + cold_start_ms if fires else 0.0),
                        fired=fires, reason=reason),
        "always": pack(always_rows, kw_ms + sem_ms),
    }


def score(per_question, questions):
    """Aggregate per-method metrics. Answerable: recall@k, MRR. No-answer: false injection."""
    out = {}
    answerable = [q for q in questions if q["kind"] == "answerable"]
    no_answer = [q for q in questions if q["kind"] == "no_answer"]
    for method in METHODS:
        hits = {1: 0, 3: 0, 5: 0}
        rr = []
        misses = []
        for q in answerable:
            sessions = per_question[q["id"]][method]["sessions"]
            expect = set(q["expect"])
            first = next((i for i, s in enumerate(sessions, 1) if s in expect), None)
            rr.append(1.0 / first if first else 0.0)
            for k in hits:
                hits[k] += bool(first and first <= k)
            if first is None or first > 5:
                misses.append(q["id"])
        false_inj = sum(per_question[q["id"]][method]["injected"] > 0 for q in no_answer)
        empty_answerable = sum(per_question[q["id"]][method]["injected"] == 0 for q in answerable)
        times = [per_question[q["id"]][method]["ms"] for q in questions]
        fired = [per_question[q["id"]][method]["fired"] for q in questions
                 if "fired" in per_question[q["id"]][method]]
        n = len(answerable) or 1
        out[method] = {
            "recall_at_1": round(hits[1] / n, 3), "recall_at_3": round(hits[3] / n, 3),
            "recall_at_5": round(hits[5] / n, 3), "mrr": round(sum(rr) / n, 3),
            "false_injection_rate": round(false_inj / len(no_answer), 3) if no_answer else None,
            "answerable_with_nothing_returned": empty_answerable,
            "latency_ms_median": round(statistics.median(times), 1),
            "latency_ms_slowest": round(max(times), 1),
            "missed_ids": misses,
        }
        if fired:
            out[method]["model_load_rate"] = round(sum(fired) / len(fired), 3)
    return out


def run_eval(config, golden_path):
    exclude, questions = load_golden(golden_path)
    return run_questions(config, exclude, questions, str(golden_path))


def run_questions(config, exclude, questions, source):
    """Score a list of questions every way. Used with a hand-checked golden
    set by this script and with auto-generated questions by remember_check.py."""
    db_path = resolve_db_path(config)
    hook = load_hook()
    spec = importlib.util.spec_from_file_location("semantic_search", ROOT / "scripts" / "semantic_search.py")
    semantic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        started = time.perf_counter()
        semantic.get_embed_model(config)              # load once; reported separately
        model_load_s = time.perf_counter() - started
        per_question = {q["id"]: run_question(hook, semantic, conn, config, db_path, q, exclude,
                                              cold_start_ms=model_load_s * 1000)
                        for q in questions}
    finally:
        conn.close()
    return {
        "created_at": utc_now(), "git_commit": git_stamp(),
        "params": {"golden": source, "semantic_cutoff": SEMANTIC_CUTOFF,
                   "rrf_k": RRF_K, "recency_scale_days": hook.RECENCY_SCALE_DAYS,
                   "fts_rank_floor": hook.FTS_RANK_FLOOR, "max_results": hook.MAX_RESULTS},
        "counts": {"answerable": sum(q["kind"] == "answerable" for q in questions),
                   "no_answer": sum(q["kind"] == "no_answer" for q in questions)},
        "model_load_s": round(model_load_s, 2),
        "methods": score(per_question, questions),
        "per_question": {qid: {m: {k: v for k, v in r.items() if k != "keywords"}
                               for m, r in res.items()} for qid, res in per_question.items()},
        "keywords_used": {qid: {"keyword": res["keyword"]["keywords"],
                                "keyword_ungated": res["keyword_ungated"]["keywords"]}
                          for qid, res in per_question.items()},
        "questions": {q["id"]: q["question"] for q in questions},
        "caveats": [
            "A question counts as found if ANY message from an expected session comes back.",
            "No current-project boost is applied, so every project is treated the same.",
            "A rival session that also answers a question counts against the method unless "
            "it is listed in 'expect'.",
            "Every question here is phrased about the past, which trips the on_miss gate more "
            "often than everyday prompts do; remember_baseline.py shows the rate on your real prompts.",
            "Meaning search runs here with the model already loaded; a cold start adds "
            f"about {round(model_load_s, 1)} s per process.",
        ],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x):
    return "n/a" if x is None else f"{round(x * 100)}%"


def render_report(r):
    m, c = r["methods"], r["counts"]
    na, nn = c["answerable"], c["no_answer"]
    L = ["=" * 66, "  MEMORY CHECK: does search find the right memory?", "=" * 66,
         f"I asked {na} questions whose answer is in memory and {nn} whose answer is not,",
         "five different ways. Higher is better, except for the junk rate.", ""]
    for method in METHODS:
        s = m[method]
        found5 = round(s["recall_at_5"] * na)
        found1 = round(s["recall_at_1"] * na)
        L += [METHOD_LABELS[method].upper(),
              f"  Right memory among the top 5:   {found5} of {na}  ({_pct(s['recall_at_5'])})",
              f"  Right memory ranked first:      {found1} of {na}  ({_pct(s['recall_at_1'])})",
              f"  Injected junk when memory had no answer:  "
              f"{round((s['false_injection_rate'] or 0) * nn)} of {nn}  ({_pct(s['false_injection_rate'])})",
              f"  Speed: {s['latency_ms_median']} ms typical, {s['latency_ms_slowest']} ms slowest"
              + (f"; loads the model on {_pct(s['model_load_rate'])} of questions" if "model_load_rate" in s else ""),
              f"  (recall@1 {s['recall_at_1']}, @3 {s['recall_at_3']}, @5 {s['recall_at_5']}; MRR {s['mrr']})", ""]
    base, mean, on_miss, always = m["keyword"], m["meaning"], m["on_miss"], m["always"]
    L.append("VERDICT")
    L.append(f"  Keyword search alone finds the right memory {_pct(base['recall_at_5'])} of the time "
             f"and injects junk on {_pct(base['false_injection_rate'])} of unanswerable questions.")
    full_gain = always["recall_at_5"] - base["recall_at_5"]
    if full_gain < -0.02:
        L.append(f"  Adding meaning search LOWERS it to {_pct(always['recall_at_5'])}; leave it off.")
    elif full_gain <= 0.02:
        L.append("  Adding meaning search changes the hit rate by under 2 points here, so it is not worth its cost.")
    else:
        got = on_miss["recall_at_5"] - base["recall_at_5"]
        L.append(f"  'always' raises that to {_pct(always['recall_at_5'])} (+{round(full_gain * 100)} points) "
                 f"and answers in about {always['latency_ms_median']} ms, but keeps a helper process running "
                 "(about 610 MB of memory).")
        L.append(f"  'on_miss' reaches {_pct(on_miss['recall_at_5'])} (+{round(got * 100)} points, "
                 f"{round(100 * got / full_gain) if full_gain else 0}% of that gain) and loads the model on only "
                 f"{_pct(on_miss.get('model_load_rate'))} of questions, with no helper process, "
                 f"but each of those takes about {round(r['model_load_s'], 1)} s.")
        L.append("  Pick 'always' if you can spare the memory and want the best results; "
                 "'on_miss' if you cannot, or want nothing running in the background.")
    if mean["false_injection_rate"] is not None and base["false_injection_rate"] is not None:
        if mean["false_injection_rate"] < base["false_injection_rate"]:
            L.append(f"  Meaning search is better at staying quiet when there is no answer "
                     f"({_pct(mean['false_injection_rate'])} junk vs {_pct(base['false_injection_rate'])}).")
    ungated = m["keyword_ungated"]
    if ungated["false_injection_rate"] is not None and base["false_injection_rate"] is not None:
        lost = round((ungated["recall_at_5"] - base["recall_at_5"]) * na)
        L.append(f"  The weak-match gate cuts junk from {_pct(ungated['false_injection_rate'])} to "
                 f"{_pct(base['false_injection_rate'])} of unanswerable questions"
                 f"{f' and loses {lost} found memories' if lost > 0 else ' without losing a found memory'}.")
    L += ["", f"Small sample ({na + nn} questions): a one-question change moves a score "
          f"by about {round(100 / na)} points.", "", "LIMITS"] + [f"  - {t}" for t in r["caveats"]]
    return "\n".join(L)


def save_result(config, result, label):
    conn = sqlite3.connect(resolve_db_path(config), timeout=10.0)
    try:
        ensure_metrics_schema(conn)
        conn.execute("INSERT INTO metrics_snapshots (created_at, kind, label, git_commit, params, results) "
                     "VALUES (?, 'eval', ?, ?, ?, ?)",
                     (result["created_at"], label, result["git_commit"],
                      json.dumps(result["params"]), json.dumps(result)))
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Score memory search against a golden set")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="path to the golden-set YAML")
    parser.add_argument("--label", default="eval", help="name for the saved snapshot")
    parser.add_argument("--no-save", action="store_true", help="print only; don't save")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    if not os.path.exists(resolve_db_path(config)):
        print(f"FATAL: database not found at {resolve_db_path(config)}")
        sys.exit(2)
    if not os.path.exists(args.golden):
        print(f"No golden set at {args.golden}. See the usage notes at the top of this script.")
        sys.exit(2)

    result = run_eval(config, args.golden)
    if not args.no_save:
        save_result(config, result, args.label)
    print(json.dumps(result, indent=2) if args.json else render_report(result))


if __name__ == "__main__":
    main()
