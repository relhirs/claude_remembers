#!/usr/bin/env python3
"""
remember_cap_study.py - How many memories should the hook inject?

A developer tool (Phase 4, Part A). It MEASURES; it changes no retrieval
behavior. The hook, retrieve() and every threshold are untouched. The question:
is a fixed cap (today: at most 5) right, or should the count depend on how good
each match is?

For every golden question it makes three retrieve() calls (plus read-only
counts), then replays a fixed list of "inject which rows?" policies offline over
the pool of candidates those calls found:

    G1  remember_eval.run_question(strict=True): the eval's own call (no project)
    G2  retrieve(config {}, mode off, golden project, no leak filter): the
        baseline replay's call (synthetic only: the private set has no as_of)
    P   retrieve(mode always, detail=True, the project the study was given,
        the eval's leak filter): the call the policies are built from

Before any policy number is written, three checks must pass (a mismatch stops
the run and prints question ids only):

    G1  the offline k=5 cap-then-filter equals run_question's keyword, on_miss
        and always ids, and the saved eval.json's
    G2  the offline k=5 keyword rows equal the saved replay_questions.jsonl
    G3  the offline k=5 merge equals the rows the real retrieve() returned in P

Plan interpretations (fixed before the first run, listed in the report):

  * Filter-then-cap: policies are scored with the leak filter (excluded
    sessions, rows that quote the question; an eval artifact the live hook never
    sees) applied BEFORE any cap, so a quoting row cannot use up a slot. Cap-
    then-filter (today's order) is kept only for the match checks and for the
    "k=5 as it runs today" row of the report.
  * Gates still see 5: whether a question's weak-match gate / on_miss gate
    fires is taken from the real P call, never recomputed from a policy's list.
    A policy therefore only decides WHICH ROWS ARE INJECTED. A diagnostic counts
    how many questions would flip a gate if the gates saw the policy's list.
  * Fixed caps cap both the keyword list and the merged list at k (what changing
    MAX_RESULTS would do). Meaning rows are not capped before the merge, as today.
  * Score rules act on each list BEFORE the merge, because the merge keeps only
    ranks. Keyword score = recency-adjusted BM25 (negative; ratio to the best,
    in (0, 1]); meaning score = cosine. Boundaries are inclusive with a 1e-9
    tolerance, so "exactly rho" and "exactly best - delta" are kept.
  * Largest gap (S5/S6): neither list is sorted (project rows come before
    global rows; meaning rows are ordered by cosine plus a project boost), so
    the rule SORTS by score to find the cut, then keeps the rows, in their
    original order, whose score is at least as strong as the last row above the
    cut. Keyword gap = the smallest adj_i / adj_(i-1) in sorted order (the
    largest relative drop); meaning gap = the largest absolute cosine drop. A
    list of 1 stays. A 2-row list always becomes 1 row. Equal gaps cut at the
    first one.
  * Token budget (B1-B3): rows are added in the no-cap order until the next one
    would exceed N tokens, measured on the real format_memories output (header
    included); landing exactly on N is included; the first row is always injected.
  * The pool is bounded by the search limits: at most 3 project + 7 global
    keyword rows and 30 meaning candidates.

Known limitations:
  * LIMIT 3 (project) and LIMIT 7 (global) bind on 45 of 46 private questions,
    and the 30-candidate meaning limit on 12, so gains from deeper caps (F5, F6,
    T1, B2, B3) are lower bounds: "no cap" means at most 3 + 7 + 30 rows.
    No limit was changed or measured beyond this.
  * It measures whether the right memory is injected and how much junk comes
    with it, not whether Claude answers better.
  * The private golden set is small (38 answerable): one question is about 5 points.

Output (--out DIR): cap_study.json and cap_study.txt. Only ids and numbers are
written (no memory text, no question text, no keywords); an allow-list check
runs over the JSON before it is saved.

Usage (normally run by examples/eval/run_eval.py --cap-study, inside its sandbox):
    python3 scripts/remember_cap_study.py --golden G.yaml --out DIR
        [--default-project oth] [--as-of ISO] [--role select|illustration]
        [--replay-questions F.jsonl] [--ref-eval saved.json] [--same-run-eval run.json]
"""

import argparse
import contextlib
import importlib.util
import json
import pathlib
import re
import sqlite3
import statistics
import sys
import time

import yaml

from config_utils import MACHINE_GENERATED_PREFIXES, resolve_db_path
import remember_eval as ev
import retrieval

ROOT = pathlib.Path(__file__).resolve().parent.parent
BIG = 10 ** 6                     # "no cap": merge_results compares len(...) >= MAX_RESULTS, so never None
EPS = 1e-9
BUCKETS = ("1", "2", "3", "4", "5", "6-10", "11+", "none")

# ---------------------------------------------------------------------------
# Policies (fixed before any number was seen)
# ---------------------------------------------------------------------------

POLICIES = (
    [{"id": f"F{i}", "kind": "fixed", "k": k, "rank": 0,
      "label": f"Inject at most {k}, gates still see 5"} for i, k in enumerate((1, 2, 3, 5, 10, 15), 1)]
    + [{"id": "S1", "kind": "score", "rho": 0.5, "delta": 0.10, "C": 5, "rank": 3,
        "label": "Score margin loose (rho 0.5, delta 0.10), ceiling 5"},
       {"id": "S2", "kind": "score", "rho": 0.5, "delta": 0.10, "C": 10, "rank": 3,
        "label": "Score margin loose (rho 0.5, delta 0.10), ceiling 10"},
       {"id": "S3", "kind": "score", "rho": 0.75, "delta": 0.05, "C": 5, "rank": 3,
        "label": "Score margin tight (rho 0.75, delta 0.05), ceiling 5"},
       {"id": "S4", "kind": "score", "rho": 0.75, "delta": 0.05, "C": 10, "rank": 3,
        "label": "Score margin tight (rho 0.75, delta 0.05), ceiling 10"},
       {"id": "S5", "kind": "gap", "C": 5, "rank": 3, "label": "Largest gap, ceiling 5"},
       {"id": "S6", "kind": "gap", "C": 10, "rank": 3, "label": "Largest gap, ceiling 10"},
       {"id": "T1", "kind": "t1", "rank": 1, "label": "No cap (bounded only by the search limits)"}]
    + [{"id": f"B{i}", "kind": "budget", "tokens": n, "rank": 2, "label": f"Token budget {n}"}
       for i, n in enumerate((250, 500, 1000), 1)]
)
NOISE_NOTE = "one question is about 5 points of 19; a 1-question difference is noise"
BASELINE_ID = "F4"                # inject at most 5, filter-then-cap


@contextlib.contextmanager
def max_results(n):
    """retrieval.MAX_RESULTS = n while active, then restored (merge_results and
    _keyword_candidates read it at call time)."""
    old = retrieval.MAX_RESULTS
    retrieval.MAX_RESULTS = n
    try:
        yield
    finally:
        retrieval.MAX_RESULTS = old


# ---------------------------------------------------------------------------
# Pure selection logic (no database, no model): tested by test_cap_study_selector
# ---------------------------------------------------------------------------

def keyword_survivors(project_rows, global_rows):
    """Project rows then global rows, minus duplicates (same project and first
    50 characters) and rows weaker than the floor, in the live order and with no
    cap. A row seen once is a duplicate even if it was then dropped by the floor.
    Returns (rows, {"duplicate": n, "below_floor": n})."""
    seen, out, dropped = set(), [], {"duplicate": 0, "below_floor": 0}
    for r in list(project_rows) + list(global_rows):
        key = (r[0], (r[2] or "")[:50])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        if r[3] is not None and r[3] > retrieval.FTS_RANK_FLOOR:
            dropped["below_floor"] += 1
            continue
        out.append(r)
    return out, dropped


def cap_then_filter(survivors, k, keep, weak_fn):
    """Today's order: cap at k, the weak-match gate sees that capped list (and
    empties it when weak), then the leak filter. Returns (rows, weak)."""
    capped = list(survivors)[:k]
    weak = bool(capped) and bool(weak_fn(capped))
    return ([] if weak else [r for r in capped if keep(r)]), weak


def filter_then_cap(survivors, k, keep):
    """The study's scoring order: leak-filter first, then cap."""
    return [r for r in survivors if keep(r)][:k]


def merge(keyword_rows, sem_rows, cap):
    """The real merge_results (RRF, keyword rows first on ties) with the cap set."""
    with max_results(cap):
        return retrieval.merge_results(keyword_rows, sem_rows)


def margin_keyword(rows, rho, adj):
    """Keep a keyword row if adj / best_adj >= rho (both negative, ratio in (0, 1])."""
    if not rows:
        return []
    best = min(adj(r) for r in rows)
    return [r for r in rows if best == 0 or adj(r) / best >= rho - EPS]


def margin_meaning(rows, delta, cos):
    """Keep a meaning row if cosine >= best cosine - delta."""
    if not rows:
        return []
    best = max(cos[r[4]] for r in rows)
    return [r for r in rows if cos[r[4]] >= best - delta - EPS]


def gap_keyword(rows, adj):
    """Largest relative drop: sort by adj (strongest first), cut before the pair
    with the smallest adj_i / adj_(i-1), keep rows (original order) at least as
    strong as the last row above the cut."""
    if len(rows) <= 1:
        return list(rows)
    s = sorted(adj(r) for r in rows)                      # most negative (best) first
    ratios = [(s[i] / s[i - 1] if s[i - 1] else 1.0) for i in range(1, len(s))]
    cut = ratios.index(min(ratios))                       # first of equal gaps; s[:cut + 1] stays
    keep_up_to = s[cut]
    return [r for r in rows if adj(r) <= keep_up_to + EPS]


def gap_meaning(rows, cos):
    """Largest absolute cosine drop: sort descending, cut before the largest
    drop, keep rows (original order) with cosine at least the last one above it."""
    if len(rows) <= 1:
        return list(rows)
    s = sorted((cos[r[4]] for r in rows), reverse=True)
    drops = [s[i - 1] - s[i] for i in range(1, len(s))]
    cut = drops.index(max(drops))
    floor_cos = s[cut]
    return [r for r in rows if cos[r[4]] >= floor_cos - EPS]


def tokens_of(rows):
    """Tokens the real injected block would take (0 when nothing is injected)."""
    return len(retrieval.format_memories(rows)) // retrieval.CHARS_PER_TOKEN if rows else 0


def budget_trim(rows, n):
    """Add rows in order until the next would go over n tokens; the first row
    is always injected; landing exactly on n is included."""
    out = []
    for r in rows:
        if out and tokens_of(out + [r]) > n:
            break
        out.append(r)
    return out


def policy_rows(p, mode, L, M, cos, adj, weak, fires):
    """The rows policy `p` injects for one question.
    L  leak-filtered keyword survivors (the keyword list before any cap)
    M  meaning rows (cutoff, leak filter and order already applied)
    cos  {row id: cosine} for M;  adj  row -> recency-adjusted BM25
    weak, fires  today's gate decisions from the real call (never recomputed)
    mode  "on_miss" | "always" | "keyword" """
    kw = [] if weak else list(L)
    kind = p["kind"]
    if kind == "fixed":
        kw_p, sem_p, cap = kw[:p["k"]], M, p["k"]
    elif kind == "score":
        kw_p, sem_p, cap = margin_keyword(kw, p["rho"], adj), margin_meaning(M, p["delta"], cos), p["C"]
    elif kind == "gap":
        kw_p, sem_p, cap = gap_keyword(kw, adj), gap_meaning(M, cos), p["C"]
    else:                                                  # t1 and budget start from the no-cap list
        kw_p, sem_p, cap = kw, M, BIG
    if mode == "keyword" or (mode == "on_miss" and not fires):
        rows = kw_p[:cap]
    else:
        rows = merge(kw_p, sem_p, cap)
    if kind == "budget":
        rows = budget_trim(rows, p["tokens"])
    return rows


def first_hit_bucket(pos):
    if pos is None:
        return "none"
    return str(pos) if pos <= 5 else ("6-10" if pos <= 10 else "11+")


def judge(rows, expect, session_of):
    """How one question's injected rows score. found = any row from an expected
    session; junk = rows from other sessions; neutral = extra rows from an
    expected session after the first."""
    sessions = [session_of.get(r[4]) for r in rows]
    hits = [i for i, s in enumerate(sessions, 1) if s in expect]
    return {"n": len(rows), "tokens": tokens_of(rows), "found": bool(hits),
            "first_pos": hits[0] if hits else None,
            "junk": sum(1 for s in sessions if s not in expect),
            "neutral": max(len(hits) - 1, 0)}


def aggregate(judged):
    """Per-policy numbers for a list of judged answerable questions."""
    n = len(judged)
    hist = {b: 0 for b in BUCKETS}
    for j in judged:
        hist[first_hit_bucket(j["first_pos"])] += 1
    toks = [j["tokens"] for j in judged]
    return {"questions": n, "found": sum(j["found"] for j in judged), "first_hit": hist,
            "junk_total": sum(j["junk"] for j in judged),
            "junk_per_question": round(sum(j["junk"] for j in judged) / n, 3) if n else 0.0,
            "neutral_total": sum(j["neutral"] for j in judged),
            "tokens_mean": round(sum(toks) / n, 1) if n else 0.0, "tokens_max": max(toks, default=0),
            "rows_max": max((j["n"] for j in judged), default=0)}


def harm(judged):
    """No-answer harm: rows and tokens injected where memory has no answer."""
    return {"questions": len(judged), "rows_total": sum(j["n"] for j in judged),
            "rows_max": max((j["n"] for j in judged), default=0),
            "tokens_total": sum(j["tokens"] for j in judged),
            "tokens_max": max((j["tokens"] for j in judged), default=0),
            "false_injection": sum(1 for j in judged if j["n"] > 0)}


SIMPLICITY = {"fixed": 0, "t1": 1, "budget": 2, "score": 3, "gap": 3}


def select_policy(candidates, results, lost_on, half="tuning"):
    """Choose a policy on ONE half (the tuning half), per section A6.
    candidates  list of policy dicts to consider
    results     {policy id: {"tuning": agg, "no_answer": harm, ...}}
    lost_on     {policy id: answerable questions lost vs the baseline on the tuning half}
    Eligible = lost at most 1. Then: found within 1 of the best eligible count
    counts as tied -> fewest junk per question -> least no-answer harm (rows,
    then tokens) -> fewest tokens -> simplest. Returns the policy dict or None."""
    eligible = [p for p in candidates if lost_on[p["id"]] <= 1]
    if not eligible:
        return None
    best_found = max(results[p["id"]][half]["found"] for p in eligible)
    tied = [p for p in eligible if results[p["id"]][half]["found"] >= best_found - 1]

    def key(p):
        r = results[p["id"]]
        return (r[half]["junk_per_question"], r["no_answer"]["rows_total"], r["no_answer"]["tokens_total"],
                r[half]["tokens_mean"], SIMPLICITY[p["kind"]], p["id"])
    return min(tied, key=key)


# ---------------------------------------------------------------------------
# Read-only counts: how many keyword rows matched before LIMIT
# ---------------------------------------------------------------------------

def _keyword_where(project, as_of, exclude_session):
    """The WHERE pieces of retrieval._keyword_candidates, copied (it has no seam
    to count without a LIMIT). The self-check below proves the copy returns the
    same rows as the real query."""
    time_filter = "AND t.timestamp < ?" if as_of else ""
    time_args = (as_of,) if as_of else ()
    session_filter = "AND t.session_id != ?" if exclude_session else ""
    session_args = (exclude_session,) if exclude_session else ()
    machine_filter = "AND NOT (" + " OR ".join(
        f"LOWER(LTRIM(SUBSTR(t.content, 1, 80))) LIKE '{prefix}%'"
        for prefix in MACHINE_GENERATED_PREFIXES) + ")"
    proj = "AND t.project = ?" if project else ""
    proj_args = (project,) if project else ()
    where = f"transcripts_fts MATCH ? {proj} {session_filter} {machine_filter} {time_filter}"
    args = lambda fts_query: (fts_query, *proj_args, *session_args, *time_args)
    return where, args, time_args


def count_before_limit(conn, fts_query, project, as_of, exclude_session, expect_rows, limit):
    """Rows the keyword query matches before LIMIT, plus a self-check that the
    copied WHERE with the real ORDER BY and LIMIT returns the real rows' ids in
    order. Raises AssertionError on any mismatch."""
    where, args, time_args = _keyword_where(project, as_of, exclude_session)
    base = "FROM transcripts_fts fts JOIN transcripts t ON t.rowid = fts.rowid"
    total = conn.execute(f"SELECT COUNT(*) {base} WHERE {where}", args(fts_query)).fetchone()[0]
    age_ref = as_of or "now"
    ids = [r[0] for r in conn.execute(
        f"SELECT t.id {base} WHERE {where} "
        f"ORDER BY fts.rank / (1.0 + MAX(0, julianday(?) - julianday(t.timestamp)) / ?) LIMIT {limit}",
        (*args(fts_query), age_ref, retrieval.RECENCY_SCALE_DAYS)).fetchall()]
    assert ids == [r[4] for r in expect_rows], "the copied keyword WHERE does not return the real rows"
    assert min(total, limit) == len(expect_rows), "matched-before-LIMIT count disagrees with the rows returned"
    return total


# ---------------------------------------------------------------------------
# Data capture
# ---------------------------------------------------------------------------

class StudyStop(Exception):
    """Carries only question ids and field names (safe to print)."""


def _ids(rows):
    return [r[4] for r in rows]


def capture_question(conn, config, semantic, q, exclude, project, as_of, g1_as_of, replay):
    """Run the three retrieve() calls for one question and capture its pool."""
    question, qid = q["question"], q["id"]
    exclude_one = exclude[0] if exclude else None
    session_of = {}
    leak = ev.make_leak_filter(conn, exclude, question, session_of)
    keep = lambda r: leak("keyword", r)
    weak_fn = lambda rows: retrieval._strength_gate(conn, rows)["weak"]
    keywords = retrieval.build_search_keywords(question)
    skipped = len(question) < retrieval.SHORT_PROMPT_CHARS or not keywords
    rec = {"id": qid, "kind": q["kind"], "expect": set(q.get("expect") or []), "skipped": skipped,
           "session_of": session_of, "checks": {}}

    # --- G1: the eval's own call, strict --------------------------------
    g1_trace = {}
    g1 = ev.run_question(semantic, conn, config, None, q, exclude, as_of=g1_as_of, strict=True,
                         keyword_only=False, trace=g1_trace)
    if skipped:
        rec.update(S=[], L=[], M=[], cos={}, weak=False, fires=False, kw_ctf=[], gate_reason=None,
                   sem_stats={"scored": 0, "above_cutoff": 0, "kept": 0, "tail_above_cutoff": False},
                   kw_stats={}, checks={"G1": g1["keyword"]["ids"] == [] and g1["on_miss"]["ids"] == []
                                       and g1["always"]["ids"] == [], "G2": True, "G3": True},
                   g1_ids={m: g1[m]["ids"] for m in ("keyword", "on_miss", "always")})
        return rec
    fts_query = " OR ".join(f'"{kw}"' for kw in keywords)

    def pool(proj, excl, when):
        diag = {}
        with max_results(BIG):
            surv = retrieval._keyword_candidates(conn, fts_query, proj, as_of=when, exclude_session=excl, diag=diag)
        mine, dropped = keyword_survivors(diag["project_rows"], diag["global_rows"])
        assert _ids(mine) == _ids(surv), "keyword_survivors disagrees with the real _keyword_candidates"
        return surv, diag, dropped

    # offline G1 (project None, eval's as_of, eval's leak): k=5 cap-then-filter
    S1, diag1, _ = pool(None, exclude_one, g1_as_of)
    kw1, weak1 = cap_then_filter(S1, 5, keep, weak_fn)
    M1 = g1_trace.get("semantic_rows", [])
    kw_info1 = {"weak": True} if weak1 else {}
    fires1, _ = retrieval.should_use_semantic(question, keywords, kw1, kw_info1, conn)
    off1 = {"keyword": _ids(kw1), "on_miss": _ids(merge(kw1, M1, 5)) if fires1 else _ids(kw1),
            "always": _ids(merge(kw1, M1, 5))}
    rec["g1_ids"] = {m: g1[m]["ids"] for m in ("keyword", "on_miss", "always")}
    rec["checks"]["G1"] = all(off1[m] == rec["g1_ids"][m] for m in off1)

    # --- G2: the replay's call (synthetic only) ----------------------------
    rec["checks"]["G2"] = None
    if replay is not None and q.get("project") and q["id"] in replay:
        S2, _, _ = pool(q["project"], None, as_of)
        C2 = S2[:5]
        weak2 = bool(C2) and weak_fn(C2)
        rec["checks"]["G2"] = ([] if weak2 else _ids(C2)) == replay[q["id"]]

    # --- P: the primary call ------------------------------------------------
    trace = {}
    result = retrieval.retrieve(conn, config, question, project=project, exclude_session=exclude_one, as_of=as_of,
                                mode="always", semantic_fn=semantic.semantic_search, row_filter=leak, detail=True,
                                trace=trace)
    sem, gate = trace["semantic"], trace["gate"]
    if sem["error"]:
        raise StudyStop(f"question {qid}: error in semantic (primary call)")
    if gate["error"]:
        raise StudyStop(f"question {qid}: error in gate (primary call)")
    if gate["fires"] is None:
        raise StudyStop(f"question {qid}: gate never decided (primary call)")
    S, diag, dropped = pool(project, exclude_one, as_of)
    kw_ctf, weak = cap_then_filter(S, 5, keep, weak_fn)
    M = trace.get("semantic_rows", [])
    rec["checks"]["G3"] = (weak == trace["keyword"]["strength_gate"]["weak"]
                           and _ids(kw_ctf) == _ids(trace["keyword_rows"])
                           and _ids(merge(kw_ctf, M, 5)) == _ids(result.rows))
    L = [r for r in S if keep(r)]
    cands = sem["candidates"]
    sim = {c["id"]: c["similarity"] for c in cands}
    n_proj = count_before_limit(conn, fts_query, project, as_of, exclude_one, diag["project_rows"], 3) if project else 0
    n_glob = count_before_limit(conn, fts_query, None, as_of, exclude_one, diag["global_rows"], 7)
    ref_now = retrieval._parse_ts(as_of) or retrieval.datetime.now(retrieval.timezone.utc)
    rec.update(S=S, L=L, M=M, cos={r[4]: sim[r[4]] for r in M}, weak=weak, fires=gate["fires"], kw_ctf=kw_ctf,
               gate_reason=gate["reason"], ref_now=ref_now,
               kw_stats={"project_matched": n_proj, "global_matched": n_glob,
                         "project_returned": len(diag["project_rows"]), "global_returned": len(diag["global_rows"]),
                         "duplicates": dropped["duplicate"], "below_floor": dropped["below_floor"],
                         "survivors": len(S), "leak_removed": len(S) - len(L), "after_leak": len(L),
                         "project_cut_by_limit": n_proj > 3, "global_cut_by_limit": n_glob > 7},
               sem_stats={"scored": len(cands), "above_cutoff": sum(c["status"] != "below_cutoff" for c in cands),
                          "kept": len(M), "dropped": len(cands) - len(M),
                          "tail_above_cutoff": bool(len(cands) >= retrieval.SEMANTIC_CANDIDATES
                                                    and cands[-1]["similarity"] > retrieval.SEMANTIC_CUTOFF)})
    rec["keywords"] = keywords
    rec["question"] = question                       # only used in memory (gate diagnostic); never written
    rec["g2_replay_ids"] = replay.get(qid) if replay else None
    return rec


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policies(conn, recs):
    """policy id -> qid -> {"on_miss": judged, "keyword_ids", "always_ids", gate_flip}."""
    out = {p["id"]: {} for p in POLICIES}
    out["F4-ctf"] = {}
    for rec in recs:
        adj = lambda r, now=rec.get("ref_now"): retrieval.row_view(r, now)["adjusted_rank"]
        for p in POLICIES:
            res = {}
            for mode in ("on_miss", "always", "keyword"):
                rows = [] if rec["skipped"] else policy_rows(p, mode, rec["L"], rec["M"], rec["cos"], adj,
                                                             rec["weak"], rec["fires"])
                res[mode] = judge(rows, rec["expect"], rec["session_of"])
                res[mode]["ids"] = _ids(rows)
            res["flip"] = None if rec["skipped"] else gate_flip(conn, rec, p, adj)
            out[p["id"]][rec["id"]] = res
        # (a) k=5 exactly as it runs today: cap-then-filter, then the same merge
        res = {}
        for mode in ("on_miss", "always", "keyword"):
            if rec["skipped"]:
                rows = []
            elif mode == "keyword" or (mode == "on_miss" and not rec["fires"]):
                rows = rec["kw_ctf"]
            else:
                rows = merge(rec["kw_ctf"], rec["M"], 5)
            res[mode] = judge(rows, rec["expect"], rec["session_of"])
            res[mode]["ids"] = _ids(rows)
        res["flip"] = False
        out["F4-ctf"][rec["id"]] = res
    return out


def gate_flip(conn, rec, p, adj):
    """Would this question's weak-match or on_miss gate decide differently if the
    gates saw the policy's keyword list instead of today's first 5? Diagnostic
    only: the study never lets a policy move a gate. The policy's own list is
    built with today's weak decision NOT applied (it would empty the very list
    the gate is being asked about)."""
    if p["kind"] == "fixed":                       # what a plain MAX_RESULTS=k would do
        C = rec["S"][:p["k"]]
        weak = bool(C) and retrieval._strength_gate(conn, C)["weak"]
        rows = [] if weak else [r for r in C if ev_keep(rec, r)]
    else:                                          # the list the policy injects
        rows = policy_rows(p, "keyword", rec["L"], rec["M"], rec["cos"], adj, False, True)
        weak = bool(rows) and retrieval._strength_gate(conn, rows)["weak"]
        if weak:                                   # a weak list is emptied before the on_miss gate sees it
            rows = []
    fires, _ = retrieval.should_use_semantic(rec["question"], rec["keywords"], rows,
                                             {"weak": True} if weak else {}, conn)
    return bool(weak != rec["weak"] or fires != rec["fires"])


def ev_keep(rec, row):
    """Did this survivor pass the leak filter? (rec["L"] is the filtered list.)"""
    return row[4] in {x[4] for x in rec["L"]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

ALLOWED_KEYS = frozenset("""
role dataset questions answerable no_answer skipped as_of default_project git_commit rows_hidden_by_as_of
checks G1 G2 G3 passed failed_ids compared_to same_run saved saved_matches saved_mismatch_tolerated skipped_check
keyword_self_check pool minmedmax min median max over5 over10 over15 more_than_5 more_than_10 more_than_15
project_matched global_matched project_returned global_returned duplicates below_floor survivors leak_removed after_leak
scored above_cutoff kept dropped union rows_limited_by_keyword_limit rows_limited_by_meaning_limit
project_cut_by_limit global_cut_by_limit tail_above_cutoff
first_expected_row where keyword_list meaning_list rrf_list pool_rows id bm25 adjusted_rank cosine rrf_position source tokens
baselines k5_today k5_filter_then_cap questions_that_differ gained_ids lost_ids
policies label kind found first_hit junk_total junk_per_question neutral_total tokens_mean tokens_max rows_max
overall tuning confirm no_answer rows_total tokens_total false_injection false_injection_constant
lost_vs_baseline gained_vs_baseline lost_ids gained_ids gate_flips gate_flip_ids per_question
n found_flag first_pos junk neutral ids keyword_ids always_ids on_miss
selection tuning_half confirm_half eligible_ids chosen confirmed reason lost_both_halves winner_is_score_rule
best_fixed_cap beats_best_fixed_cap_on_both_halves worst_case
tuning_ids confirm_ids
""".split())


def check_allowlist(obj, extra_keys=(), extra_strings=()):
    """Raise ValueError if the JSON-to-be has a key that is not allow-listed or a
    string value that looks like text (only ids, labels and numbers may leave)."""
    keys = ALLOWED_KEYS | set(map(str, extra_keys))
    strings = set(extra_strings)

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k) not in keys:
                    raise ValueError(f"key not allow-listed: {path}/{k}")
                walk(v, f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, path)
        elif isinstance(o, str):
            if o not in strings and not re.fullmatch(r"[A-Za-z0-9_\-.:+]{0,40}", o):
                raise ValueError(f"text value not allowed at {path}")
    walk(obj, "")


def _mmm(values):
    values = list(values)
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else \
        {"min": 0, "median": 0, "max": 0}


def build_pool_section(recs):
    live = [r for r in recs if not r["skipped"]]
    kw = lambda k: _mmm(r["kw_stats"][k] for r in live)
    sm = lambda k: _mmm(r["sem_stats"][k] for r in live)
    union = [len({x[4] for x in r["L"]} | {x[4] for x in r["M"]}) for r in live]
    pool = {"project_matched": kw("project_matched"), "global_matched": kw("global_matched"),
            "project_returned": kw("project_returned"), "global_returned": kw("global_returned"),
            "duplicates": kw("duplicates"), "below_floor": kw("below_floor"), "survivors": kw("survivors"),
            "leak_removed": kw("leak_removed"), "after_leak": kw("after_leak"),
            "scored": sm("scored"), "above_cutoff": sm("above_cutoff"), "kept": sm("kept"),
            "dropped": sm("dropped"), "union": _mmm(union),
            "more_than_5": sum(u > 5 for u in union), "more_than_10": sum(u > 10 for u in union),
            "more_than_15": sum(u > 15 for u in union),
            "project_cut_by_limit": sum(r["kw_stats"]["project_cut_by_limit"] for r in live),
            "global_cut_by_limit": sum(r["kw_stats"]["global_cut_by_limit"] for r in live),
            "tail_above_cutoff": sum(r["sem_stats"]["tail_above_cutoff"] for r in live)}
    return pool


def build_pool_rows(recs):
    """Per question: every candidate row in the no-cap merge order, with its numbers."""
    out, where = {}, {"keyword_list": [], "meaning_list": [], "rrf_list": []}
    for r in recs:
        if r["skipped"] and r["kind"] == "answerable":      # nothing was searched: counts as "none"
            for name in where:
                where[name].append(None)
        if r["skipped"] or r["kind"] != "answerable":
            if not r["skipped"]:
                out[r["id"]] = []
            continue
        merged = merge(r["L"], r["M"], BIG)
        view = {v["id"]: v for v in retrieval.merge_view(r["L"], r["M"], merged)}
        rows = []
        for pos, row in enumerate(merged, 1):
            v = retrieval.row_view(row, r["ref_now"])
            rows.append({"id": row[4], "bm25": v["rank"], "adjusted_rank": v["adjusted_rank"],
                         "cosine": r["cos"].get(row[4]), "rrf_position": pos, "source": view[row[4]]["source"],
                         "tokens": (len(retrieval.format_memories([row])) - len(retrieval.format_memories([])))
                                       // retrieval.CHARS_PER_TOKEN})
        out[r["id"]] = rows
        for name, lst in (("keyword_list", r["L"]), ("meaning_list", r["M"]), ("rrf_list", merged)):
            pos = next((i for i, x in enumerate(lst, 1) if r["session_of"].get(x[4]) in r["expect"]), None)
            where[name].append(pos)
    first = {name: {b: sum(first_hit_bucket(p) == b for p in positions) for b in BUCKETS}
             for name, positions in where.items()}
    return out, first


def study(conn, config, semantic, golden_path, args):
    exclude, questions = ev.load_golden(golden_path)
    golden_as = ev.golden_as_of(golden_path)
    as_of = golden_as or args.as_of
    if not as_of:
        raise SystemExit("the study needs a fixed time: add as_of to the golden set or pass --as-of")
    g1_as_of = golden_as                          # what the eval itself uses (None = the clock)
    replay = None
    if args.replay_questions and pathlib.Path(args.replay_questions).exists():
        replay = {}
        for line in pathlib.Path(args.replay_questions).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                replay[row["id"]] = row["result_ids"]
    recs, stops = [], []
    for q in questions:
        project = q.get("project") or args.default_project
        try:
            recs.append(capture_question(conn, config, semantic, q, exclude, project, as_of, g1_as_of, replay))
        except ev.EvalError as e:
            stops.append(str(e))
        except StudyStop as e:
            stops.append(str(e))
    if stops:
        raise StudyStop("; ".join(stops))
    hidden = conn.execute("SELECT COUNT(*) FROM transcripts WHERE timestamp >= ?", (as_of,)).fetchone()[0]

    # ---- match checks (before any policy number is written) ----------------
    live = [r for r in recs if not r["skipped"]]
    failed = {g: [r["id"] for r in recs if r["checks"].get(g) is False] for g in ("G1", "G2", "G3")}
    refs = {}
    for label, path in (("saved", args.ref_eval), ("same_run", args.same_run_eval)):
        if path and pathlib.Path(path).exists():
            per = json.loads(pathlib.Path(path).read_text())["per_question"]
            refs[label] = [r["id"] for r in recs if r["id"] in per and any(
                per[r["id"]][m]["ids"] != r["g1_ids"][m] for m in ("keyword", "on_miss", "always"))]
    tolerate_saved = g1_as_of is None             # the clock moved between the saved run and this one
    g1_ref_ok = all(not ids for label, ids in refs.items() if label == "same_run" or not tolerate_saved)
    checks = {"G1": {"passed": not failed["G1"] and g1_ref_ok, "failed_ids": failed["G1"],
                     "compared_to": sorted(refs),
                     "saved_matches": (not refs["saved"]) if "saved" in refs else None,
                     "saved_mismatch_tolerated": bool(tolerate_saved and refs.get("saved"))},
              "G2": {"passed": not failed["G2"], "failed_ids": failed["G2"],
                     "skipped_check": replay is None},
              "G3": {"passed": not failed["G3"], "failed_ids": failed["G3"]},
              "rows_hidden_by_as_of": hidden, "keyword_self_check": "passed"}
    bad = [g for g in ("G1", "G2", "G3") if not checks[g]["passed"]]
    if bad or hidden:
        raise StudyStop("match checks failed: " + ", ".join(
            f"{g} on {checks[g]['failed_ids'] or sorted(refs)}" for g in bad)
            + (f"; {hidden} rows are newer than as_of" if hidden else ""))

    # ---- policies --------------------------------------------------------
    res = evaluate_policies(conn, recs)
    answerable = [r for r in recs if r["kind"] == "answerable"]
    no_ans = [r for r in recs if r["kind"] == "no_answer"]
    tuning = answerable[0::2]                      # 1st, 3rd, 5th ... answerable (golden-file order)
    confirm = answerable[1::2]
    select = args.role == "select"
    ids_of = lambda rs: [r["id"] for r in rs]

    def block(pid, rs, key="on_miss"):
        return aggregate([res[pid][r["id"]][key] for r in rs])

    base = res[BASELINE_ID]
    found = lambda pid, r, mode="on_miss": res[pid][r["id"]][mode]["found"]
    pol_out, summary = {}, {}
    for p in POLICIES + [{"id": "F4-ctf", "label": "Inject at most 5 as it runs today (cap, then filter)", "kind": "fixed"}]:
        pid = p["id"]
        entry = {"label": p["label"], "kind": p["kind"], "overall": block(pid, answerable),
                 "no_answer": harm([res[pid][r["id"]]["on_miss"] for r in no_ans])}
        lost = [r["id"] for r in answerable if found(BASELINE_ID, r) and not found(pid, r)]
        gained = [r["id"] for r in answerable if not found(BASELINE_ID, r) and found(pid, r)]
        entry.update({"lost_vs_baseline": len(lost), "gained_vs_baseline": len(gained),
                      "lost_ids": lost, "gained_ids": gained})
        flips = [r["id"] for r in recs if res[pid][r["id"]].get("flip")]
        entry.update({"gate_flips": len(flips), "gate_flip_ids": flips})
        if select:
            entry["tuning"], entry["confirm"] = block(pid, tuning), block(pid, confirm)
        entry["per_question"] = {r["id"]: {"found_flag": res[pid][r["id"]]["on_miss"]["found"],
                                           "first_pos": res[pid][r["id"]]["on_miss"]["first_pos"],
                                           "n": res[pid][r["id"]]["on_miss"]["n"],
                                           "tokens": res[pid][r["id"]]["on_miss"]["tokens"],
                                           "junk": res[pid][r["id"]]["on_miss"]["junk"],
                                           "neutral": res[pid][r["id"]]["on_miss"]["neutral"],
                                           "ids": res[pid][r["id"]]["on_miss"]["ids"],
                                           "keyword_ids": res[pid][r["id"]]["keyword"]["ids"],
                                           "always_ids": res[pid][r["id"]]["always"]["ids"]} for r in recs}
        pol_out[pid] = entry

    fi = {pid: e["no_answer"]["false_injection"] for pid, e in pol_out.items()}
    constant = len(set(fi.values())) == 1
    for e in pol_out.values():
        e["false_injection_constant"] = constant
    differ = [r["id"] for r in recs if res["F4-ctf"][r["id"]]["on_miss"]["ids"] != res[BASELINE_ID][r["id"]]["on_miss"]["ids"]]
    baselines = {"k5_today": pol_out["F4-ctf"]["overall"], "k5_filter_then_cap": pol_out[BASELINE_ID]["overall"],
                 "questions_that_differ": differ,
                 "gained_ids": [r["id"] for r in answerable if not found("F4-ctf", r) and found(BASELINE_ID, r)],
                 "lost_ids": [r["id"] for r in answerable if found("F4-ctf", r) and not found(BASELINE_ID, r)]}

    selection = None
    if select:
        lost_on = {p["id"]: sum(1 for r in tuning if found(BASELINE_ID, r) and not found(p["id"], r))
                   for p in POLICIES}
        views = {p["id"]: {"tuning": pol_out[p["id"]]["tuning"], "no_answer": pol_out[p["id"]]["no_answer"]}
                 for p in POLICIES}
        chosen = select_policy(POLICIES, views, lost_on)
        selection = {"tuning_half": {"questions": len(tuning)}, "confirm_half": {"questions": len(confirm)},
                     "tuning_ids": ids_of(tuning), "confirm_ids": ids_of(confirm),
                     "eligible_ids": [p["id"] for p in POLICIES if lost_on[p["id"]] <= 1],
                     "chosen": chosen["id"] if chosen else None}
        if chosen:
            cid = chosen["id"]
            lost_both = pol_out[cid]["lost_vs_baseline"]
            confirm_junk = pol_out[cid]["confirm"]["junk_per_question"]
            base_junk = pol_out[BASELINE_ID]["confirm"]["junk_per_question"]
            selection.update({"lost_both_halves": lost_both,
                              "confirmed": bool(lost_both <= 1 and confirm_junk <= base_junk),
                              "reason": "ok" if lost_both <= 1 and confirm_junk <= base_junk else
                              ("lost_too_many" if lost_both > 1 else "more_junk_on_confirm_half")})
            fixed = [p for p in POLICIES if p["kind"] == "fixed"]
            best_fixed = select_policy(fixed, views, lost_on)
            selection["winner_is_score_rule"] = chosen["kind"] in ("score", "gap")
            if best_fixed:
                bf = best_fixed["id"]
                selection["best_fixed_cap"] = bf
                beats = all(pol_out[cid][h]["junk_per_question"] < pol_out[bf][h]["junk_per_question"]
                            or pol_out[cid][h]["tokens_mean"] < pol_out[bf][h]["tokens_mean"]
                            for h in ("tuning", "confirm"))
                selection["beats_best_fixed_cap_on_both_halves"] = bool(beats)

    dataset = {"role": args.role, "questions": len(recs), "answerable": len(answerable), "no_answer": len(no_ans),
               "skipped": sum(r["skipped"] for r in recs), "as_of": as_of, "default_project": args.default_project}
    pool_rows, first_expected = build_pool_rows(recs)
    out = {"dataset": dataset, "checks": checks, "pool": build_pool_section(recs),
           "first_expected_row": {"where": first_expected}, "pool_rows": pool_rows,
           "baselines": baselines, "policies": pol_out}
    if selection:
        out["selection"] = selection
    worst = {pid: {"rows_max": e["overall"]["rows_max"], "tokens_max": e["overall"]["tokens_max"]}
             for pid, e in pol_out.items()}
    out["worst_case"] = worst
    return out, recs


def render_text(out):
    d, pol, b = out["dataset"], out["policies"], out["baselines"]
    project = d["default_project"] or "each question's own"
    L = [f"CAP STUDY  ({d['role']}; {d['answerable']} answerable + {d['no_answer']} no-answer questions; "
         f"fixed time {d['as_of']}; project {project})", "",
         "Checks before any number counted: G1 " + ("pass" if out["checks"]["G1"]["passed"] else "FAIL")
         + ", G2 " + ("skipped (no replay file)" if out["checks"]["G2"]["skipped_check"] else
                      ("pass" if out["checks"]["G2"]["passed"] else "FAIL"))
         + ", G3 " + ("pass" if out["checks"]["G3"]["passed"] else "FAIL")
         + f"; rows newer than the fixed time: {out['checks']['rows_hidden_by_as_of']}.", ""]
    p = out["pool"]
    L += ["POOL BEFORE ANY CAP (min / median / max across questions)"]
    for k, label in (("project_matched", "project keyword rows matched before LIMIT 3"),
                     ("global_matched", "global keyword rows matched before LIMIT 7"),
                     ("survivors", "keyword rows after duplicates and floor"),
                     ("after_leak", "keyword rows after the leak filter"),
                     ("scored", "meaning rows scored"), ("above_cutoff", "meaning rows above 0.35"),
                     ("kept", "meaning rows kept"), ("union", "candidate rows in total")):
        v = p[k]
        L.append(f"  {label:<46} {v['min']} / {v['median']} / {v['max']}")
    L += [f"  questions with more than 5 / 10 / 15 candidates: {p['more_than_5']} / {p['more_than_10']} / {p['more_than_15']}",
          f"  cut off by LIMIT 3: {p['project_cut_by_limit']}, by LIMIT 7: {p['global_cut_by_limit']}, "
          f"by the 30-candidate meaning limit (30th still above 0.35): {p['tail_above_cutoff']} questions",
          "  (caps of 10 or 15 can only help when more candidates exist than the cap; the pool tops out at "
          f"{p['union']['max']} here)", ""]
    L += ["WHERE THE FIRST EXPECTED ROW SITS (no cap)  1 / 2 / 3 / 4 / 5 / 6-10 / 11+ / none"]
    for name, hist in out["first_expected_row"]["where"].items():
        L.append(f"  {name:<14} " + " / ".join(str(hist[x]) for x in BUCKETS))
    L += ["", "k=5 TWICE: as it runs today (cap, then filter) vs the baseline every policy is compared to (filter, then cap)",
          f"  today: found {b['k5_today']['found']} of {b['k5_today']['questions']}, junk {b['k5_today']['junk_total']}; "
          f"baseline: found {b['k5_filter_then_cap']['found']}, junk {b['k5_filter_then_cap']['junk_total']}; "
          f"questions that differ: {len(b['questions_that_differ'])} {b['questions_that_differ']}", ""]
    sel = "tuning" in next(iter(pol.values()))
    L += ["POLICIES (injected rows only; gates still see 5)",
          f"  {'id':<6}{'found':>6}{'lost':>6}{'gain':>6}{'junk/q':>8}{'tok mean':>10}{'tok max':>9}{'rows max':>9}"
          f"{'NA rows':>9}{'NA tok':>8}{'NA inj':>8}{'gate flips':>11}"]
    for pid, e in pol.items():
        o = e["overall"]
        L.append(f"  {pid:<6}{o['found']:>6}{e['lost_vs_baseline']:>6}{e['gained_vs_baseline']:>6}"
                 f"{o['junk_per_question']:>8}{o['tokens_mean']:>10}{o['tokens_max']:>9}{o['rows_max']:>9}"
                 f"{e['no_answer']['rows_total']:>9}{e['no_answer']['tokens_total']:>8}{e['no_answer']['false_injection']:>8}"
                 f"{e['gate_flips']:>11}")
    same = next(iter(pol.values()))["false_injection_constant"]
    nn = d["no_answer"]
    L += ["", (f"False injection ({next(iter(pol.values()))['no_answer']['false_injection']} of {nn} no-answer questions) "
               "is the same for every policy, as expected: every cap keeps at least 1 row, the score rules keep the best "
               "row, the budget always injects the first row, and all policies share the same filtered list and the same "
               "gates. It can only change through a gate flip (last column), which the study never applies."
               if same else "WARNING: false injection differs between policies, which should not happen by construction "
               "(see the NA inj column); investigate before trusting any policy number.")]
    if "selection" in out:
        s = out["selection"]
        L += ["", "SELECTION (tuning half only, then confirmed on both halves)",
              f"  tuning half: {s['tuning_half']['questions']} questions; confirm half: {s['confirm_half']['questions']}",
              f"  eligible on the tuning half (lost at most 1): {s['eligible_ids']}",
              f"  chosen: {s['chosen']}; lost on both halves: {s.get('lost_both_halves')}; "
              f"confirmed: {s.get('confirmed')} ({s.get('reason')})",
              f"  winner is a score rule: {s.get('winner_is_score_rule')}; best fixed cap: {s.get('best_fixed_cap')}; "
              f"score rule beats it on both halves: {s.get('beats_best_fixed_cap_on_both_halves')}",
              f"  {NOISE_NOTE}"]
    else:
        L += ["", "(illustration only: not split into halves and not used to choose a policy)"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="How many memories should the hook inject? (measurement only)")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--default-project")
    ap.add_argument("--as-of")
    ap.add_argument("--role", choices=("select", "illustration"), default="illustration")
    ap.add_argument("--replay-questions")
    ap.add_argument("--ref-eval")
    ap.add_argument("--same-run-eval")
    args = ap.parse_args()
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = resolve_db_path(config)
    spec = importlib.util.spec_from_file_location("semantic_search", ROOT / "scripts" / "semantic_search.py")
    semantic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic)
    semantic.get_embed_model(config)
    conn = ev._ro_connect(db_path)
    try:
        started = time.perf_counter()
        try:
            out, recs = study(conn, config, semantic, args.golden, args)
        except (StudyStop, ev.EvalError) as e:           # question ids and field names only
            print(f"STOPPED: {e}", file=sys.stderr)
            sys.exit(4)
        except AssertionError as e:
            print(f"STOPPED: self-check failed: {e}", file=sys.stderr)
            sys.exit(5)
        qids = [r["id"] for r in recs]
        check_allowlist(out, extra_keys=qids + [p["id"] for p in POLICIES] + ["F4-ctf"] + list(BUCKETS),
                        extra_strings=qids + [p["label"] for p in POLICIES] + ["Inject at most 5 as it runs today (cap, then filter)"]
                        + [args.default_project or "", args.as_of or "", out["dataset"]["as_of"]])
        outdir = pathlib.Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "cap_study.json").write_text(json.dumps(out, indent=2) + "\n")
        (outdir / "cap_study.txt").write_text(render_text(out))
        print(json.dumps({"ok": True, "seconds": round(time.perf_counter() - started, 1)}))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
