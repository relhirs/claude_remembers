#!/usr/bin/env python3
"""
remember_eval.py - Score memory search against a golden set of questions.

A golden set is a list of questions whose answer you already know lives in a
specific past session (plus some questions with NO answer in memory). This
script asks each question each of the ways below and reports, in plain language, how often
the right session shows up and how often it injects junk when it shouldn't:

    keyword          keyword search only, with its weak-match gate
    keyword_ungated  the same search without the gate (what ran before it)
    meaning          meaning-based (embedding) search only, cosine cutoff 0.35
    on_miss          keyword, plus meaning search only when the gate says
                     keyword can't be trusted (config: mode: on_miss)
    always           keyword plus meaning search on every prompt, merged by
                     Reciprocal Rank Fusion (config: mode: always)
    mcp              the MCP tool search_transcripts, exactly as Claude calls it
                     when it looks memory up by hand (mcp/server.py, unchanged;
                     its rows are captured, not re-implemented)

Read-only: it never writes to the transcript tables. The only write is saving
the finished result to metrics_snapshots (--no-save skips that).

The golden set lives in eval/golden_set.yaml (personal, gitignored); a public,
synthetic one lives in examples/eval/golden_set.yaml:

    as_of: "2026-06-01T00:00:00Z"  # optional: fixes "now" for keyword recency
    exclude_sessions: [<session ids that quote the questions, e.g. today's>]
    questions:
      - id: q01
        kind: answerable           # or no_answer
        question: "where is the api key stored now"
        expect: [<session ids that hold the answer; any of them counts>]
        tags: [recency]            # optional: scores are also broken down by tag
        superseded: [<older session with an out-of-date answer>]  # optional,
                                   # never counts as a hit; checks newer-first
        project: tdp               # optional: used by remember_baseline.py
                                   # --questions, not here (no project boost)

Known limits, printed in the report: a question is "right" if ANY message from
an expected session comes back (session-level, not exact-message); the search
is run with no current-project boost; and a rival session that also answers a
question counts against it unless you add that session to `expect`.

Usage:
    python3 scripts/remember_eval.py
    python3 scripts/remember_eval.py --golden eval/golden_candidates.yaml
    python3 scripts/remember_eval.py --label before-gate --json
    python3 scripts/remember_eval.py --keyword-only   # no model: keyword, keyword_ungated, mcp
"""

import argparse
import contextlib
import importlib.util
import json
import os
import pathlib
import sqlite3
import statistics
import sys
import time
import types

import yaml

from config_utils import is_machine_generated, resolve_db_path
from metrics_utils import ensure_metrics_schema, utc_now
from remember_baseline import git_stamp
import retrieval

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.yaml"
METHODS = ("keyword", "keyword_ungated", "meaning", "on_miss", "always")
KEYWORD_METHODS = ("keyword", "keyword_ungated")      # need no model
SKIPPED_NOTE = "skipped (too short or no usable keywords)"
ALL_METHODS = METHODS + ("mcp",)
METHOD_LABELS = {
    "keyword": "Keyword search only",
    "keyword_ungated": "Keyword, without the weak-match gate",
    "meaning": "Meaning-based search only",
    "on_miss": "Meaning search only when keyword looks unreliable (mode: on_miss)",
    "always": "Keyword + meaning on every prompt (mode: always)",
    "mcp": "MCP search_transcripts (Claude looking memory up by hand)",
}


def load_golden(path):
    data = yaml.safe_load(pathlib.Path(path).read_text())
    if isinstance(data, list):
        data = {"questions": data}
    return data.get("exclude_sessions") or [], data["questions"]


def golden_as_of(path):
    """The golden set's fixed 'now' (ISO string), or None to use the clock."""
    data = yaml.safe_load(pathlib.Path(path).read_text())
    return data.get("as_of") if isinstance(data, dict) else None


def _ro_connect(db_path):
    """Read-only connection; the file: URI form also works with Windows paths."""
    return sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)


def load_mcp_server(db_path):
    """Import mcp/server.py by path and point it at `db_path`, read-only.

    The module itself is not changed: only its connection factory is swapped
    for a read-only one, so the eval measures exactly what the tool does.
    """
    spec = importlib.util.spec_from_file_location("remember_mcp_server", ROOT / "mcp" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    def read_only_db():
        conn = _ro_connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    server.get_db = read_only_db
    server.DB_PATH = db_path
    return server


@contextlib.contextmanager
def _read_only_fuzzy(server):
    """While active, the typo-correction step the MCP tool falls back to when
    nothing matches cannot write. Left alone, fuzzy_search opens the database
    read-write and runs CREATE VIRTUAL TABLE IF NOT EXISTS for its vocabulary
    table. Here its sqlite3 connections are read-only and that table is made in
    the connection's temporary schema instead (temp tables never touch the
    file). The correction logic itself is the real one, unchanged, and
    mcp/server.py and fuzzy_search.py are not edited: only names inside the
    already-loaded fuzzy_search module are swapped, then put back."""
    fuzzy = sys.modules.get(getattr(server.fuzzy_correct, "__module__", None))
    if fuzzy is None:
        yield
        return
    saved = (fuzzy.sqlite3, fuzzy._ensure_vocab_table)

    def ensure_temp_vocab(conn):
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.transcripts_fts_vocab "
                     "USING fts5vocab(main, transcripts_fts, row)")

    fuzzy.sqlite3 = types.SimpleNamespace(connect=lambda path, *a, **k: _ro_connect(path), Connection=sqlite3.Connection)
    fuzzy._ensure_vocab_table = ensure_temp_vocab
    try:
        yield
    finally:
        fuzzy.sqlite3, fuzzy._ensure_vocab_table = saved


def mcp_search(server, question):
    """Call the real search_transcripts and capture the rows it formats."""
    captured = []
    original = server._format_results

    def capture(rows, header=""):
        captured.extend(rows)
        return original(rows, header)

    server._format_results = capture
    try:
        started = time.perf_counter()
        with _read_only_fuzzy(server):
            server.search_transcripts(question)
        ms = (time.perf_counter() - started) * 1000
    finally:
        server._format_results = original
    return captured, ms


def _rank_sessions(rows):
    """Rows (best first) -> distinct sessions in order of first appearance."""
    seen, out = set(), []
    for _, session in rows:
        if session and session not in seen:
            seen.add(session)
            out.append(session)
    return out


def _clean(question, rows):
    """Drop rows that merely quote the question (the questions themselves get
    written into transcripts when they are discussed)."""
    q = question.lower().strip()
    return [r for r in rows if q not in (r["content"] or "").lower()]


def run_question(semantic, conn, config, db_path, item, exclude, cold_start_ms=0.0,
                 as_of=None, server=None, keyword_only=False):
    """Ask one question every way with ONE retrieve() call (the function the live
    hook runs), then read each method's rows from what it did. `cold_start_ms`
    is charged to on_miss whenever its gate loads the model. `as_of` fixes
    "now" for the keyword search's recency weighting (None = the clock, as
    live). `server` (from load_mcp_server) adds the mcp method; `keyword_only`
    skips every method that needs the embedding model."""
    question = item["question"]
    quoted = question.lower().strip()
    exclude_one = exclude[0] if exclude else None
    session_of = {}

    def leak_filter(kind, row):
        """Drop rows from excluded sessions and rows that quote the question
        (the questions themselves get written into transcripts when discussed)."""
        if kind == "keyword":                # (project, timestamp, content, rank, id)
            found_session = conn.execute("SELECT session_id FROM transcripts WHERE id = ?",
                                         (row[4],)).fetchone()
            session = found_session[0] if found_session else None
            if session in exclude or quoted in (row[2] or "").lower():
                return False
            session_of[row[4]] = session
            return True
        session_of[row["id"]] = row["session_id"]           # meaning search result (dict)
        return row["session_id"] not in exclude and quoted not in (row["content"] or "").lower()

    def pack(rows, ms, **extra):
        return {"sessions": _rank_sessions([(r[4], session_of.get(r[4])) for r in rows]),
                "ids": [r[4] for r in rows],
                "machine_rows": sum(1 for r in rows if is_machine_generated(r[2])),
                "injected": len(rows), "ms": ms, **extra}

    # mode "always" with no helper function embeds in-process, so meaning,
    # always and on_miss can all be read from one pass.
    trace = {}
    result = retrieval.retrieve(conn, config, question, project=None, exclude_session=exclude_one,
                                as_of=as_of, mode="off" if keyword_only else "always",
                                semantic_fn=semantic.semantic_search, row_filter=leak_filter, trace=trace)
    keywords = trace["keywords"]["kept"]
    if trace["skip"]:                        # nothing to search: every method returns nothing
        kw_rows, ug_rows, kw_ms = [], [], 0.0
    else:
        kw_rows, ug_rows, kw_ms = trace["keyword_rows"], trace["ungated_rows"], trace["keyword"]["ms"]

    out = {
        "keyword": pack(kw_rows, kw_ms, keywords=keywords),
        "keyword_ungated": pack(ug_rows, kw_ms, keywords=keywords),
    }
    if server is not None:
        # Same leak filters as above (excluded sessions, rows quoting the
        # question). No machine-row filter: the MCP tool has none, and that
        # difference is part of what this measures.
        found, mcp_ms = mcp_search(server, question)
        mcp_rows = [r for r in found
                    if r["session_id"] not in exclude and quoted not in (r["preview"] or "").lower()]
        out["mcp"] = {"sessions": _rank_sessions([(None, r["session_id"]) for r in mcp_rows]),
                      "ids": [f"{r['session_id']}@{r['timestamp']}" for r in mcp_rows],
                      "machine_rows": sum(1 for r in mcp_rows if is_machine_generated(r["preview"])),
                      "injected": len(mcp_rows), "ms": mcp_ms}
    if trace["skip"]:
        for method in KEYWORD_METHODS:
            out[method]["skipped"] = SKIPPED_NOTE
    if trace["skip"]:
        for method in KEYWORD_METHODS:
            out[method]["skipped"] = SKIPPED_NOTE
    if keyword_only:
        return out

    if trace["skip"]:
        # retrieve() stopped before any search (too short, or only stop words), so
        # it recorded no meaning-search step. Every hook method returns nothing.
        out.update({
            "meaning": pack([], 0.0, best_similarity=None),
            "on_miss": pack([], 0.0, fired=False, reason=trace["skip"]),
            "always": pack([], 0.0),
        })
        for method in METHODS:
            out[method]["skipped"] = SKIPPED_NOTE
        return out

    sem = trace["semantic"]
    sem_ms = sem["ms"] or 0.0
    sem_rows = trace.get("semantic_rows", [])
    first = sem["usable_first_similarity"]
    best_similarity = round(first, 3) if first is not None else None

    always_rows = result.rows
    fires, reason = trace["gate"]["fires"], trace["gate"]["reason"]
    on_miss_rows = always_rows if fires else kw_rows

    out.update({
        "meaning": pack(sem_rows[:retrieval.MAX_RESULTS], sem_ms, best_similarity=best_similarity),
        "on_miss": pack(on_miss_rows, kw_ms + (sem_ms + cold_start_ms if fires else 0.0),
                        fired=fires, reason=reason),
        "always": pack(always_rows, kw_ms + sem_ms),
    })
    return out


def _first_rank(sessions, wanted):
    return next((i for i, s in enumerate(sessions, 1) if s in wanted), None)


def score(per_question, questions):
    """Aggregate per-method metrics. Answerable: recall@k, MRR. No-answer: false
    injection. All questions: how often a machine-generated row came back.
    Optional extras when the golden set has them: per-tag recall and, for
    questions with a `superseded` session, whether the newer answer ranked first."""
    out = {}
    answerable = [q for q in questions if q["kind"] == "answerable"]
    no_answer = [q for q in questions if q["kind"] == "no_answer"]
    first_q = next(iter(per_question.values()), {})
    methods = [m for m in ALL_METHODS if m in first_q]
    for method in methods:
        hits = {1: 0, 3: 0, 5: 0}
        rr = []
        misses = []
        firsts = {}
        for q in answerable:
            sessions = per_question[q["id"]][method]["sessions"]
            expect = set(q["expect"])
            first = _first_rank(sessions, expect)
            firsts[q["id"]] = first
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
        if all("machine_rows" in per_question[q["id"]][method] for q in questions):
            out[method]["machine_rows_returned"] = sum(
                per_question[q["id"]][method]["machine_rows"] > 0 for q in questions)
        tags = sorted({t for q in answerable for t in (q.get("tags") or [])})
        if tags:
            out[method]["by_tag"] = {}
            for tag in tags:
                tagged = [q for q in answerable if tag in (q.get("tags") or [])]
                out[method]["by_tag"][tag] = {
                    "n": len(tagged),
                    "recall_at_1": round(sum(bool(firsts[q["id"]] and firsts[q["id"]] <= 1)
                                             for q in tagged) / len(tagged), 3),
                    "recall_at_5": round(sum(bool(firsts[q["id"]] and firsts[q["id"]] <= 5)
                                             for q in tagged) / len(tagged), 3)}
        dated = [q for q in answerable if q.get("superseded")]
        if dated:
            newer = older = 0
            for q in dated:
                sessions = per_question[q["id"]][method]["sessions"]
                new_at, old_at = firsts[q["id"]], _first_rank(sessions, set(q["superseded"]))
                if new_at and (old_at is None or new_at < old_at):
                    newer += 1
                elif old_at:
                    older += 1
            out[method]["recency"] = {"n": len(dated), "newer_ranked_first": newer,
                                      "older_ranked_first": older,
                                      "neither_found": len(dated) - newer - older}
    return out


def run_eval(config, golden_path, include_mcp=False, keyword_only=False):
    exclude, questions = load_golden(golden_path)
    return run_questions(config, exclude, questions, str(golden_path), as_of=golden_as_of(golden_path),
                         include_mcp=include_mcp, keyword_only=keyword_only)


def run_questions(config, exclude, questions, source, as_of=None, include_mcp=False, keyword_only=False):
    """Score a list of questions every way. Used with a hand-checked golden
    set by this script and with auto-generated questions by remember_check.py.
    The defaults (no mcp, clock time, all model methods) are what
    remember_check.py relies on."""
    db_path = resolve_db_path(config)
    spec = importlib.util.spec_from_file_location("semantic_search", ROOT / "scripts" / "semantic_search.py")
    semantic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic)
    conn = _ro_connect(db_path)
    try:
        model_load_s = None
        if not keyword_only:
            started = time.perf_counter()
            semantic.get_embed_model(config)              # load once; reported separately
            model_load_s = time.perf_counter() - started
        # After the timing above: server.py imports sentence_transformers at
        # module level, and loading it first would hide that import's ~3 s
        # from the cold-start cost charged to on_miss.
        server = load_mcp_server(db_path) if include_mcp else None
        per_question = {q["id"]: run_question(semantic, conn, config, db_path, q, exclude,
                                              cold_start_ms=(model_load_s or 0.0) * 1000, as_of=as_of,
                                              server=server, keyword_only=keyword_only)
                        for q in questions}
    finally:
        conn.close()
    methods = score(per_question, questions)
    skipped_ids = [qid for qid, res in per_question.items() if "skipped" in res["keyword"]]
    return {
        "created_at": utc_now(), "git_commit": git_stamp(),
        "params": {"golden": source, "as_of": as_of, "keyword_only": keyword_only,
                   "methods_run": list(methods),
                   "semantic_cutoff": retrieval.SEMANTIC_CUTOFF,
                   "rrf_k": retrieval.RRF_K, "recency_scale_days": retrieval.RECENCY_SCALE_DAYS,
                   "fts_rank_floor": retrieval.FTS_RANK_FLOOR, "max_results": retrieval.MAX_RESULTS},
        "counts": {"answerable": sum(q["kind"] == "answerable" for q in questions),
                   "no_answer": sum(q["kind"] == "no_answer" for q in questions),
                   **({"skipped": len(skipped_ids)} if skipped_ids else {})},
        **({"skipped_ids": skipped_ids} if skipped_ids else {}),
        "model_load_s": round(model_load_s, 2) if model_load_s is not None else None,
        "methods": methods,
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
        ] + ([
            "Meaning search runs here with the model already loaded; a cold start adds "
            f"about {round(model_load_s, 1)} s per process."] if model_load_s is not None else [
            "Keyword-only run: meaning, on_miss and always were not run (they need the embedding model)."]
        ) + ([
            f"{len(skipped_ids)} question(s) were skipped by the hook rule (under {retrieval.SHORT_PROMPT_CHARS} "
            "characters, or only stop words): the hook methods returned nothing for them, so they count as "
            "misses when answerable and as correctly quiet when not. The mcp tool is not subject to that rule."]
            if skipped_ids else []) + ([
            "mcp is the search_transcripts tool as Claude calls it: up to 20 rows, no weak-match gate, "
            "no recency weighting and no machine-message filter; its 'injected' count is rows returned."]
            if include_mcp else []) + ([
            f"Keyword recency is measured from a fixed time ({as_of}), not the clock."] if as_of else []),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x):
    return "n/a" if x is None else f"{round(x * 100)}%"


def render_report(r):
    m, c = r["methods"], r["counts"]
    na, nn = c["answerable"], c["no_answer"]
    ways = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(len(m), str(len(m)))
    L = ["=" * 66, "  MEMORY CHECK: does search find the right memory?", "=" * 66,
         f"I asked {na} questions whose answer is in memory and {nn} whose answer is not,",
         f"{ways} different ways. Higher is better, except for the junk rate.", ""]
    if c.get("skipped"):
        L += [f"NOTE: {c['skipped']} of those questions were skipped (too short or no usable keywords):",
              "search returned nothing for them, so they count as misses (answerable) or as correctly",
              "quiet (no answer) in the scores below. Skipped: " + ", ".join(r.get("skipped_ids", [])), ""]
    for method in ALL_METHODS:
        if method not in m:
            continue
        s = m[method]
        found5 = round(s["recall_at_5"] * na)
        found1 = round(s["recall_at_1"] * na)
        L += [METHOD_LABELS[method].upper(),
              f"  Right memory among the top 5:   {found5} of {na}  ({_pct(s['recall_at_5'])})",
              f"  Right memory ranked first:      {found1} of {na}  ({_pct(s['recall_at_1'])})",
              f"  Injected junk when memory had no answer:  "
              f"{round((s['false_injection_rate'] or 0) * nn)} of {nn}  ({_pct(s['false_injection_rate'])})"]
        if "machine_rows_returned" in s:
            L.append(f"  Returned a machine-generated message (subagent hand-back, command output) on:  "
                     f"{s['machine_rows_returned']} of {na + nn} questions")
        if "recency" in s:
            rc = s["recency"]
            L.append(f"  Old vs updated answer: newer ranked first on {rc['newer_ranked_first']} of {rc['n']}, "
                     f"older first on {rc['older_ranked_first']}, neither found on {rc['neither_found']}")
        if "by_tag" in s:
            L.append("  By question type (top 5 / first):  " + ";  ".join(
                f"{tag} {round(t['recall_at_5'] * t['n'])}/{t['n']} / {round(t['recall_at_1'] * t['n'])}/{t['n']}"
                for tag, t in s["by_tag"].items()))
        L += [f"  Speed: {s['latency_ms_median']} ms typical, {s['latency_ms_slowest']} ms slowest"
              + (f"; loads the model on {_pct(s['model_load_rate'])} of questions" if "model_load_rate" in s else ""),
              f"  (recall@1 {s['recall_at_1']}, @3 {s['recall_at_3']}, @5 {s['recall_at_5']}; MRR {s['mrr']})", ""]
    base = m["keyword"]
    L.append("VERDICT")
    L.append(f"  Keyword search alone finds the right memory {_pct(base['recall_at_5'])} of the time "
             f"and injects junk on {_pct(base['false_injection_rate'])} of unanswerable questions.")
    if "mcp" in m:
        mc = m["mcp"]
        L.append(f"  The MCP lookup tool finds it {_pct(mc['recall_at_5'])} of the time in its top 5 and returns "
                 f"something on {_pct(mc['false_injection_rate'])} of unanswerable questions.")
    if "always" not in m:
        L += ["  Meaning-based methods were not run (--keyword-only)."]
        L += ["", f"Small sample ({na + nn} questions): a one-question change moves a score "
              f"by about {round(100 / na)} points.", "", "LIMITS"] + [f"  - {t}" for t in r["caveats"]]
        return "\n".join(L)
    mean, on_miss, always = m["meaning"], m["on_miss"], m["always"]
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
    parser.add_argument("--keyword-only", action="store_true",
                        help="skip the embedding model: run keyword, keyword_ungated and mcp only")
    parser.add_argument("--no-mcp", action="store_true", help="don't score the MCP search_transcripts tool")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    if not os.path.exists(resolve_db_path(config)):
        print(f"FATAL: database not found at {resolve_db_path(config)}")
        sys.exit(2)
    if not os.path.exists(args.golden):
        print(f"No golden set at {args.golden}. See the usage notes at the top of this script.")
        sys.exit(2)

    result = run_eval(config, args.golden, include_mcp=not args.no_mcp, keyword_only=args.keyword_only)
    if not args.no_save:
        save_result(config, result, args.label)
    print(json.dumps(result, indent=2) if args.json else render_report(result))


if __name__ == "__main__":
    main()
