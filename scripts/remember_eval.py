#!/usr/bin/env python3
"""
remember_eval.py - Score memory search against a golden set of questions.

A golden set is a list of questions whose answer you already know lives in a
specific past session (plus some questions with NO answer in memory). This
script asks each question four ways and reports, in plain language, how often
the right session shows up and how often it injects junk when it shouldn't:

    keyword          the search the live hook runs (with its weak-match gate)
    keyword_ungated  the same search without the gate (what ran before it)
    meaning          meaning-based (embedding) search, cosine cutoff 0.35
    both             keyword + meaning, merged by Reciprocal Rank Fusion

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
METHODS = ("keyword", "keyword_ungated", "meaning", "both")
METHOD_LABELS = {
    "keyword": "Keyword search (what runs today)",
    "keyword_ungated": "Keyword, without the weak-match gate",
    "meaning": "Meaning-based search",
    "both": "Keyword + meaning together",
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


def run_question(hook, semantic, conn, config, db_path, item, exclude):
    question = item["question"]
    exclude_one = exclude[0] if exclude else None

    def keyword_rows(keywords, gated):
        if not keywords:
            return [], 0.0
        fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
        started = time.perf_counter()
        raw = hook._keyword_search(conn, fts_query, None, exclude_session=exclude_one, gated=gated)
        ms = (time.perf_counter() - started) * 1000
        rows = []
        for row in raw:                      # (project, timestamp, content, rank, id)
            session = conn.execute("SELECT session_id FROM transcripts WHERE id = ?",
                                   (row[4],)).fetchone()
            rows.append({"id": row[4], "session": session[0] if session else None,
                         "content": row[2]})
        rows = [r for r in _clean(question, rows) if r["session"] not in exclude]
        return rows, ms

    keywords = hook.build_search_keywords(question)
    kw_rows, kw_ms = keyword_rows(keywords, gated=True)
    ug_rows, ug_ms = keyword_rows(keywords, gated=False)

    started = time.perf_counter()
    found = semantic.semantic_search(conn, config, question, limit=CANDIDATES)
    sem_ms = (time.perf_counter() - started) * 1000
    sem_all = [{"id": r["id"], "session": r["session_id"], "content": r["content"],
                "similarity": r["similarity"]} for r in found
               if r["session_id"] not in exclude and not is_machine_generated(r["content"])]
    sem_all = _clean(question, sem_all)
    sem_rows = [r for r in sem_all if r["similarity"] >= SEMANTIC_CUTOFF]

    pairs = lambda rows: [(r["id"], r["session"]) for r in rows]
    both = _rrf(pairs(kw_rows), pairs(sem_rows))
    results = {
        "keyword": {"sessions": _rank_sessions(pairs(kw_rows)), "injected": len(kw_rows),
                    "ms": kw_ms, "keywords": keywords},
        "keyword_ungated": {"sessions": _rank_sessions(pairs(ug_rows)), "injected": len(ug_rows),
                            "ms": ug_ms, "keywords": keywords},
        "meaning": {"sessions": _rank_sessions(pairs(sem_rows)), "injected": len(sem_rows),
                    "ms": sem_ms,
                    "best_similarity": round(sem_all[0]["similarity"], 3) if sem_all else None},
        "both": {"sessions": _rank_sessions(both), "injected": len(both),
                 "ms": kw_ms + sem_ms},
    }
    return results


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
    return out


def run_eval(config, golden_path):
    db_path = resolve_db_path(config)
    exclude, questions = load_golden(golden_path)
    hook = load_hook()
    spec = importlib.util.spec_from_file_location("semantic_search", ROOT / "scripts" / "semantic_search.py")
    semantic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        started = time.perf_counter()
        semantic.get_embed_model(config)              # load once; reported separately
        model_load_s = time.perf_counter() - started
        per_question = {q["id"]: run_question(hook, semantic, conn, config, db_path, q, exclude)
                        for q in questions}
    finally:
        conn.close()
    return {
        "created_at": utc_now(), "git_commit": git_stamp(),
        "params": {"golden": str(golden_path), "semantic_cutoff": SEMANTIC_CUTOFF,
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
         "four different ways. Higher is better, except for the junk rate.", ""]
    for method in METHODS:
        s = m[method]
        found5 = round(s["recall_at_5"] * na)
        found1 = round(s["recall_at_1"] * na)
        L += [METHOD_LABELS[method].upper(),
              f"  Right memory among the top 5:   {found5} of {na}  ({_pct(s['recall_at_5'])})",
              f"  Right memory ranked first:      {found1} of {na}  ({_pct(s['recall_at_1'])})",
              f"  Injected junk when memory had no answer:  "
              f"{round((s['false_injection_rate'] or 0) * nn)} of {nn}  ({_pct(s['false_injection_rate'])})",
              f"  Speed: {s['latency_ms_median']} ms typical, {s['latency_ms_slowest']} ms slowest",
              f"  (recall@1 {s['recall_at_1']}, @3 {s['recall_at_3']}, @5 {s['recall_at_5']}; MRR {s['mrr']})", ""]
    base, mean, both = m["keyword"], m["meaning"], m["both"]
    L.append("VERDICT")
    L.append(f"  Keyword search finds the right memory {_pct(base['recall_at_5'])} of the time "
             f"and injects junk on {_pct(base['false_injection_rate'])} of unanswerable questions.")
    gain = both["recall_at_5"] - base["recall_at_5"]
    if gain > 0.02:
        L.append(f"  Adding meaning search raises that to {_pct(both['recall_at_5'])} "
                 f"(+{round(gain * 100)} points) at a cost of {both['latency_ms_median']} ms "
                 "typical per search once the model is loaded.")
    elif gain < -0.02:
        L.append(f"  Adding meaning search LOWERS it to {_pct(both['recall_at_5'])}; do not turn it on as-is.")
    else:
        L.append("  Adding meaning search changes the hit rate by under 2 points here.")
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
