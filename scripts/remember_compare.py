#!/usr/bin/env python3
"""
remember_compare.py - Paired comparisons: the same questions asked two ways.

A developer tool (Phase 4, Part B). It MEASURES; it changes no retrieval
behavior. For each golden question it asks the question twice and puts the two
results side by side, so a change can be judged question by question.

Comparison available now:

    project-bias   the hook searches a project first and boosts its meaning rows.
                   Side "on" passes project=P to retrieve(); side "off" passes
                   project=None. Nothing else differs.

    mcp-machine-filter   (Phase 4, B2a) the MCP search_transcripts tool hides
                   harness-written messages (task notices, system reminders ...) in
                   its SQL, before the 20-row LIMIT, so up to 20 real rows still come
                   back. With --ref REF, side "today" is mcp/server.py as it was at
                   git ref REF and side "filtered" is the working tree's, both run on
                   the same database. --expect FILE (a saved earlier result of this
                   comparison, e.g. the simulation that was approved) must match the
                   new numbers exactly, or the run stops and prints only the keys
                   that differ. Also replays the real recorded calls (--recorded):
                   counts only, per call number, never the query.

    mcp-gate       (Phase 4, B2b, REJECTED: measurement only, nothing ships) would the
                   hook's weak-match bar help the MCP tool? With --simulate the loaded
                   server's _run_fts_query also returns fts.rank; when the best raw rank is above retrieval._confidence_floor
                   (the hook's bar), the simulated tool returns no rows (the real
                   change would return a one-line "no strong match" note). Applies to
                   the normal path and the typo fallback, never to explicit FTS
                   syntax. Two sides: the tool as it ships, and the tool plus the gate.
                   Caveat: MCP picks words differently from the hook (digits allowed,
                   up to 10 words, repeats kept), so its scores are not on exactly
                   the hook's scale.

Call shape for project-bias (the same as the cap study's primary call and the eval's run_question):
one retrieve() per side, mode "always", the eval's leak filter, a fixed as_of.
"on_miss" (your config's mode) is read from the gate decision: the merged rows if
the gate fired, otherwise the keyword rows. Keyword-only and "always" are saved too.

P is the project passed with --default-project (every question), or, when a
question has its own `project:`, that one.

Only ids and numbers are written (no memory text, no question text, no keywords);
an allow-list check runs over the JSON before it is saved.

Usage (normally run by examples/eval/run_eval.py --compare NAME, inside its sandbox):
    python3 scripts/remember_compare.py project-bias --golden G.yaml --out DIR
        [--default-project oth] [--as-of ISO] [--check-against cap_study.json]

--check-against: a cap_study.json from the same dataset. The "on" side must equal
its policies["F4-ctf"]["per_question"] (found flag, first position, injected ids)
for every question; a mismatch stops the run and prints only question ids.
"""

import argparse
import shutil
import subprocess
import tempfile
import importlib.util
import json
import math
import pathlib
import sys
import time

import yaml

from config_utils import MACHINE_GENERATED_PREFIXES, is_machine_generated, machine_filter_sql, resolve_db_path
import remember_eval as ev
import remember_cap_study as cap
import retrieval

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPARISONS = ("project-bias", "mcp-machine-filter", "mcp-gate")
RECORDED_TOOL = "mcp__remember-server__search_transcripts"
MODES = ("on_miss", "keyword", "always")
SIDES = ("on", "off")
EXTRA_KEYS = """
comparison sides on off dataset project_for_each as_of default_project git_commit checks cross_check passed failed_ids
checked_against paired found_both found_neither only_on only_off sign_test p_value n_only_on n_only_off plain
per_question n found_flag first_pos junk neutral tokens ids keyword_ids always_ids gate_fires gate_flips gate_flip_ids
gate_starts_with_bias_off gate_stops_with_bias_off no_answer injected_on injected_off injected_differs
overall tokens_mean junk_total questions found pool keyword_rows_mean modes
""".split()


class CompareStop(Exception):
    """Carries only question ids and field names (safe to print)."""


def sign_test(only_on, only_off):
    """Two-sided exact sign test on the questions where exactly one side found
    the answer. Returns (p, plain-words sentence)."""
    n = only_on + only_off
    if n == 0:
        return 1.0, "No question changed, so there is nothing to test."
    k = min(only_on, only_off)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    lean = ("bias helps" if only_on > only_off else "bias hurts" if only_off > only_on else "no lean")
    verdict = ("a clear direction" if p < 0.05 else
               "a direction, but too few questions to rule out luck" if only_on != only_off else "an even split")
    return round(p, 4), (f"{only_on} vs {only_off} ({lean}) is {verdict} (p = {p:.2f}).")


def run_side(conn, config, semantic, q, exclude, project, as_of):
    """One retrieve() call; returns the rows for each mode plus the gate decision."""
    question = q["question"]
    session_of = {}
    leak = ev.make_leak_filter(conn, exclude, question, session_of)
    exclude_one = exclude[0] if exclude else None
    keywords = retrieval.build_search_keywords(question)
    if len(question) < retrieval.SHORT_PROMPT_CHARS or not keywords:
        return {"skipped": True, "session_of": session_of,
                "rows": {m: [] for m in MODES}, "fires": False, "kw_pool": 0}
    trace = {}
    result = retrieval.retrieve(conn, config, question, project=project, exclude_session=exclude_one, as_of=as_of,
                                mode="always", semantic_fn=semantic.semantic_search, row_filter=leak, detail=True,
                                trace=trace)
    sem, gate = trace["semantic"], trace["gate"]
    if sem["error"]:
        raise CompareStop(f"question {q['id']}: error in semantic")
    if gate["error"]:
        raise CompareStop(f"question {q['id']}: error in gate")
    if gate["fires"] is None:
        raise CompareStop(f"question {q['id']}: gate never decided")
    kw = trace["keyword_rows"]
    rows = {"keyword": kw, "always": result.rows, "on_miss": result.rows if gate["fires"] else kw}
    return {"skipped": False, "session_of": session_of, "rows": rows, "fires": bool(gate["fires"]),
            "kw_pool": len(trace["keyword"].get("project_candidates", [])) + len(trace["keyword"].get("global_candidates", []))}


def judged(side, expect):
    out = {}
    for m in MODES:
        j = cap.judge(side["rows"][m], expect, side["session_of"])
        j["ids"] = [r[4] for r in side["rows"][m]]
        out[m] = j
    return out


def compare_project_bias(conn, config, semantic, golden_path, default_project, as_of_arg):
    exclude, questions = ev.load_golden(golden_path)
    as_of = ev.golden_as_of(golden_path) or as_of_arg
    if not as_of:
        raise SystemExit("the comparison needs a fixed time: add as_of to the golden set or pass --as-of")
    recs = []
    for q in questions:
        project = q.get("project") or default_project
        if not project:
            raise SystemExit("project-bias needs a project: pass --default-project or give each question a project")
        expect = set(q.get("expect") or [])
        sides = {"on": run_side(conn, config, semantic, q, exclude, project, as_of),
                 "off": run_side(conn, config, semantic, q, exclude, None, as_of)}
        recs.append({"id": q["id"], "kind": q["kind"], "expect": expect, "sides": sides,
                     "j": {s: judged(sides[s], expect) for s in SIDES}})
    return recs, as_of


def cross_check(recs, ref_path):
    """The 'on' side must equal the cap study's k=5-as-it-runs-today per question."""
    ref = json.loads(pathlib.Path(ref_path).read_text())["policies"]["F4-ctf"]["per_question"]
    bad = []
    for r in recs:
        want = ref.get(r["id"])
        got = r["j"]["on"]["on_miss"]
        if want is None or (want["found_flag"], want["first_pos"], want["ids"]) != (got["found"], got["first_pos"], got["ids"]):
            bad.append(r["id"])
    return bad


def build_output(recs, as_of, default_project, checked_against, git_commit):
    answerable = [r for r in recs if r["kind"] == "answerable"]
    no_ans = [r for r in recs if r["kind"] == "no_answer"]
    found = lambda r, s, m="on_miss": r["j"][s][m]["found"]
    out = {"comparison": "project-bias",
           "dataset": {"questions": len(recs), "answerable": len(answerable), "no_answer": len(no_ans),
                       "as_of": as_of, "default_project": default_project, "git_commit": git_commit},
           "cross_check": {"passed": True, "checked_against": checked_against}}
    modes = {}
    for m in MODES:
        both = [r["id"] for r in answerable if found(r, "on", m) and found(r, "off", m)]
        neither = [r["id"] for r in answerable if not found(r, "on", m) and not found(r, "off", m)]
        only_on = [r["id"] for r in answerable if found(r, "on", m) and not found(r, "off", m)]
        only_off = [r["id"] for r in answerable if not found(r, "on", m) and found(r, "off", m)]
        p, plain = sign_test(len(only_on), len(only_off))
        modes[m] = {"paired": {"found_both": both, "found_neither": neither,
                               "only_on": only_on, "only_off": only_off},
                    "sign_test": {"n_only_on": len(only_on), "n_only_off": len(only_off), "p_value": p, "plain": plain},
                    "overall": {s: cap.aggregate([r["j"][s][m] for r in answerable]) for s in SIDES},
                    "no_answer": {s: cap.harm([r["j"][s][m] for r in no_ans]) for s in SIDES}}
    out["modes"] = modes
    flips = {"starts_with_bias_off": [r["id"] for r in recs if r["sides"]["off"]["fires"] and not r["sides"]["on"]["fires"]],
             "stops_with_bias_off": [r["id"] for r in recs if r["sides"]["on"]["fires"] and not r["sides"]["off"]["fires"]]}
    out["gate_flips"] = flips
    out["pool"] = {s: {"keyword_rows_mean": round(sum(r["sides"][s]["kw_pool"] for r in recs) / len(recs), 2)}
                   for s in SIDES}
    out["per_question"] = {r["id"]: {s: {"found_flag": r["j"][s]["on_miss"]["found"],
                                         "first_pos": r["j"][s]["on_miss"]["first_pos"],
                                         "n": r["j"][s]["on_miss"]["n"], "tokens": r["j"][s]["on_miss"]["tokens"],
                                         "junk": r["j"][s]["on_miss"]["junk"], "ids": r["j"][s]["on_miss"]["ids"],
                                         "keyword_ids": r["j"][s]["keyword"]["ids"],
                                         "always_ids": r["j"][s]["always"]["ids"],
                                         "gate_fires": r["sides"][s]["fires"]} for s in SIDES} for r in recs}
    return out


def render_text(out):
    d, m = out["dataset"], out["modes"]["on_miss"]
    own_project = "(each question's own)"
    lines = [f"Project bias on vs off - {d['questions']} questions ({d['answerable']} answerable, {d['no_answer']} no-answer), "
             f"as_of {d['as_of']}, project {d['default_project'] or own_project}",
             f"Cross-check against {out['cross_check']['checked_against'] or 'nothing (not requested)'}: passed", ""]
    for mode in MODES:
        mm = out["modes"][mode]
        pr = mm["paired"]
        lines += [f"[{mode}]  found by both {len(pr['found_both'])}, by neither {len(pr['found_neither'])}, "
                  f"only with bias {len(pr['only_on'])} {pr['only_on']}, only without bias {len(pr['only_off'])} {pr['only_off']}",
                  f"    {mm['sign_test']['plain']}"]
        for s in SIDES:
            o = mm["overall"][s]
            lines.append(f"    bias {s:3}: found {o['found']}/{o['questions']}, junk {o['junk_total']}, "
                         f"tokens mean {o['tokens_mean']}, first hit {o['first_hit']}")
        for s in SIDES:
            lines.append(f"    bias {s:3} no-answer: {mm['no_answer'][s]}")
        lines.append("")
    f = out["gate_flips"]
    lines += [f"Meaning search starts only with bias off: {len(f['starts_with_bias_off'])} {f['starts_with_bias_off']}",
              f"Meaning search stops with bias off: {len(f['stops_with_bias_off'])} {f['stops_with_bias_off']}",
              f"Keyword pool (rows before cap, mean): on {out['pool']['on']['keyword_rows_mean']}, "
              f"off {out['pool']['off']['keyword_rows_mean']} (bias off: 3+7 becomes 7)", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# mcp-machine-filter
# ---------------------------------------------------------------------------

def hook_sql_machine(text):
    """What the hook's SQL filter does: first 80 characters, spaces stripped (only
    spaces), lowercased, starts with a marker."""
    return (text or "")[:80].lstrip(" ").lower().startswith(MACHINE_GENERATED_PREFIXES)


def simulate_rank(server):
    """Make _run_fts_query also select fts.rank, with the same machine filter the
    tool ships. Returns an undo function. Rows keep their usual columns plus `rank`."""
    original = server._run_fts_query
    flt = machine_filter_sql("t.content")

    def ranked(conn, fts_query, project, limit, recency_bias):
        order = ("fts.rank * (1.0 / (1.0 + julianday('now') - julianday(t.timestamp)))" if recency_bias
                 else "fts.rank")
        where = "transcripts_fts MATCH ?" + (" AND t.project = ?" if project else "")
        args = (fts_query, project, limit) if project else (fts_query, limit)
        return conn.execute(
            f"""SELECT t.session_id, t.project, t.timestamp, t.type,
                      substr(t.content, 1, 600) as preview, fts.rank AS rank
               FROM transcripts_fts fts
               JOIN transcripts t ON t.rowid = fts.rowid
               WHERE {where} {flt}
               ORDER BY {order}
               LIMIT ?""", args).fetchall()

    server._run_fts_query = ranked
    return lambda: setattr(server, "_run_fts_query", original)


GATE_SIDES = {"today": False, "gate": True}        # side -> does the simulated gate apply


def mcp_gate_call(server, floor, query, project, side):
    """One simulated call. Returns (rows the tool would show, gated?, best rank, explicit FTS syntax?)."""
    gate = GATE_SIDES[side]
    syntax = server._is_fts5_syntax(query)
    undo = simulate_rank(server)
    try:
        rows = mcp_rows(server, query, project)
    finally:
        undo()
    best = min((r["rank"] for r in rows), default=None)
    gated = bool(gate and not syntax and floor is not None and best is not None and best > floor)
    return ([] if gated else rows), gated, best, syntax


def compare_mcp_gate(conn, db_path, golden_path, with_recorded):
    server = ev.load_mcp_server(db_path)
    floor = retrieval._confidence_floor(conn)
    exclude, questions = ev.load_golden(golden_path)
    per_q = {}
    for q in questions:
        quoted = q["question"].lower().strip()
        expect = set(q.get("expect") or [])
        entry = {"kind": q["kind"]}
        for side in GATE_SIDES:
            rows, gated, best, _ = mcp_gate_call(server, floor, q["question"], None, side)
            kept = [r for r in rows if r["session_id"] not in exclude and quoted not in (r["preview"] or "").lower()]
            sessions = [r["session_id"] for r in kept]
            entry[side] = {"n": len(kept), "gated": gated,
                           "first_pos": next((i for i, sid in enumerate(sessions, 1) if sid in expect), None),
                           **machine_counts(kept)}
            if side == "today":
                entry["best_rank"] = None if best is None else round(best, 2)
        per_q[q["id"]] = entry
    recorded = []
    if with_recorded:
        for query, project in recorded_calls(conn):
            entry = {}
            for side in GATE_SIDES:
                rows, gated, best, syntax = mcp_gate_call(server, floor, query, project, side)
                entry[side] = {"n": len(rows), "gated": gated}
                if side == "today":
                    entry["fts_syntax"] = syntax
                    entry["best_rank"] = None if best is None else round(best, 2)
            recorded.append(entry)
    return per_q, recorded, None if floor is None else round(floor, 3)


def build_gate_output(per_q, recorded, floor, git_commit):
    answerable = {k: v for k, v in per_q.items() if v["kind"] == "answerable"}
    no_ans = {k: v for k, v in per_q.items() if v["kind"] != "answerable"}
    in_top = lambda e, n: e["first_pos"] is not None and e["first_pos"] <= n
    summary = {"questions": len(per_q), "answerable": len(answerable), "no_answer": len(no_ans), "floor": floor}
    flips = {}
    for side in GATE_SIDES:
        summary[f"found_top5_{side}"] = sum(in_top(v[side], 5) for v in answerable.values())
        summary[f"found_in_all_rows_{side}"] = sum(in_top(v[side], 10 ** 6) for v in answerable.values())
        summary[f"no_answer_returning_rows_{side}"] = sum(1 for v in no_ans.values() if v[side]["n"] > 0)
        summary[f"gated_{side}"] = sum(1 for v in per_q.values() if v[side]["gated"])
        if side != "today":
            flips[f"lost_top5_{side}"] = [k for k, v in answerable.items() if in_top(v["today"], 5) and not in_top(v[side], 5)]
            flips[f"lost_all_rows_{side}"] = [k for k, v in answerable.items()
                                              if in_top(v["today"], 10 ** 6) and not in_top(v[side], 10 ** 6)]
            flips[f"gained_top5_{side}"] = [k for k, v in answerable.items() if not in_top(v["today"], 5) and in_top(v[side], 5)]
            flips[f"gated_answerable_{side}"] = [k for k, v in answerable.items() if v[side]["gated"]]
            flips[f"gated_no_answer_{side}"] = [k for k, v in no_ans.items() if v[side]["gated"]]
    ranks = lambda d: sorted(v["best_rank"] for v in d.values() if v["best_rank"] is not None)
    summary["best_rank_answerable"] = ranks(answerable)
    summary["best_rank_no_answer"] = ranks(no_ans)
    out = {"comparison": "mcp-gate", "dataset": {"git_commit": git_commit, "mode": "simulation"},
           "summary": summary, "flips": flips, "per_question": per_q}
    if recorded:
        out["recorded"] = {"calls": len(recorded),
                           "fts_syntax_calls": sum(1 for r in recorded if r["fts_syntax"]),
                           "returning_rows_today": sum(1 for r in recorded if r["today"]["n"] > 0),
                           "gated": [f"r{i:02d}" for i, r in enumerate(recorded, 1) if r["gate"]["gated"]],
                           "per_call": {f"r{i:02d}": r for i, r in enumerate(recorded, 1)}}
    return out


def render_gate_text(out):
    s, f = out["summary"], out["flips"]
    lines = [f"MCP weak-match gate (simulation) - {s['questions']} questions ({s['answerable']} answerable, "
             f"{s['no_answer']} no-answer); hook bar (floor) = {s['floor']}", ""]
    for side, name in (("today", "tool as shipped"), ("gate", "tool + gate")):
        lines.append(f"{name:22}: answers in top 5 {s[f'found_top5_{side}']}, in all rows {s[f'found_in_all_rows_{side}']}; "
                     f"no-answer questions returning rows {s[f'no_answer_returning_rows_{side}']} of {s['no_answer']}; "
                     f"questions gated {s[f'gated_{side}']}")
    for side in ("gate",):
        lines += [f"[{side}] answerable gated: {f[f'gated_answerable_{side}']}  lost top 5: {f[f'lost_top5_{side}']}  "
                  f"lost in all rows: {f[f'lost_all_rows_{side}']}  gained top 5: {f[f'gained_top5_{side}']}",
                  f"[{side}] no-answer gated: {f[f'gated_no_answer_{side}']}"]
    lines += [f"Best rank per answerable question (more negative = stronger): {s['best_rank_answerable']}",
              f"Best rank per no-answer question: {s['best_rank_no_answer']}", ""]
    if "recorded" in out:
        r = out["recorded"]
        lines += [f"Recorded real calls: {r['calls']} ({r['fts_syntax_calls']} explicit FTS syntax, never gated); "
                  f"returning rows today {r['returning_rows_today']}",
                  f"  would get 'no strong match': {len(r['gated'])} {r['gated']}"]
        for cid, c in r["per_call"].items():
            lines.append(f"  {cid}: rows today {c['today']['n']}, best rank {c['best_rank']}"
                         f"{', FTS syntax' if c['fts_syntax'] else ''}, gated: {'yes' if c['gate']['gated'] else 'no'}")
    return "\n".join(lines) + "\n"


def mcp_rows(server, query, project=None):
    """The rows search_transcripts formats for this call (read-only, like the eval)."""
    captured = []
    original = server._format_results

    def capture(rows, header=""):
        captured.extend(rows)
        return original(rows, header)

    server._format_results = capture
    try:
        with ev._read_only_fuzzy(server):
            server.search_transcripts(query, project)
    finally:
        server._format_results = original
    return captured


def machine_counts(rows):
    previews = [r["preview"] for r in rows]
    return {"machine_python": sum(is_machine_generated(t) for t in previews),
            "machine_hook_sql": sum(hook_sql_machine(t) for t in previews)}


def recorded_calls(conn):
    """(query, project) for every real tool_use of search_transcripts, oldest first.
    Text stays in memory; only counts leave."""
    calls = []
    for raw, in conn.execute("SELECT raw_json FROM transcripts WHERE raw_json LIKE ? AND raw_json LIKE '%\"tool_use\"%' "
                             "ORDER BY timestamp, id", (f"%{RECORDED_TOOL}%",)):
        try:
            content = (json.loads(raw).get("message") or {}).get("content")
        except (ValueError, AttributeError):
            continue
        for b in content if isinstance(content, list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == RECORDED_TOOL:
                inp = b.get("input") or {}
                if inp.get("query"):
                    calls.append((inp["query"], inp.get("project")))
    return calls


def load_ref_server(db_path, ref, repo):
    """mcp/server.py as it was at git ref `ref`, run in a temporary folder that holds
    that ref's mcp/ and scripts/ plus this folder's config.yaml, pointed at db_path.
    Load the working-tree server FIRST: both import the package `scripts`, and the
    first one loaded wins, so the newer scripts/ (which has machine_filter_sql) must
    be the one that is cached."""
    tmp = tempfile.mkdtemp(prefix="remember_ref_server_")
    arch = subprocess.run(["git", "archive", ref, "mcp", "scripts"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", tmp], input=arch.stdout, check=True)
    shutil.copy(ROOT / "config.yaml", pathlib.Path(tmp) / "config.yaml")
    return ev.load_mcp_server(db_path, server_path=pathlib.Path(tmp) / "mcp" / "server.py")


def compare_mcp_machine_filter(conn, db_path, golden_path, with_recorded, ref, repo):
    now_server = ev.load_mcp_server(db_path)
    servers = {"filtered": now_server, "today": load_ref_server(db_path, ref, repo)}
    exclude, questions = ev.load_golden(golden_path)
    per_q = {}
    for q in questions:
        quoted = q["question"].lower().strip()
        expect = set(q.get("expect") or [])
        entry = {"kind": q["kind"]}
        for side in ("today", "filtered"):
            rows = mcp_rows(servers[side], q["question"])
            rows = [r for r in rows if r["session_id"] not in exclude and quoted not in (r["preview"] or "").lower()]
            sessions = [r["session_id"] for r in rows]
            entry[side] = {"n": len(rows), **machine_counts(rows),
                           "first_pos": next((i for i, sid in enumerate(sessions, 1) if sid in expect), None)}
        per_q[q["id"]] = entry
    recorded = []
    if with_recorded:
        for query, project in recorded_calls(conn):
            entry = {}
            for side in ("today", "filtered"):
                rows = mcp_rows(servers[side], query, project)
                entry[side] = {"n": len(rows), **machine_counts(rows)}
            recorded.append(entry)
    return per_q, recorded


def differing_keys(a, b, path=""):
    """Dotted keys where two result trees differ (never values: they may be private)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += differing_keys(a.get(k), b.get(k), f"{path}.{k}" if path else str(k)) if k in a and k in b else [f"{path}.{k}" if path else str(k)]
        return out
    return [] if a == b else [path]


def build_mcp_output(per_q, recorded, git_commit):
    answerable = {k: v for k, v in per_q.items() if v["kind"] == "answerable"}
    in_top = lambda e, n: e["first_pos"] is not None and e["first_pos"] <= n
    summary = {"questions": len(per_q), "answerable": len(answerable)}
    for d in ("machine_python", "machine_hook_sql"):
        summary[f"with_machine_rows_today_{d}"] = sum(1 for v in per_q.values() if v["today"][d] > 0)
        summary[f"with_machine_rows_filtered_{d}"] = sum(1 for v in per_q.values() if v["filtered"][d] > 0)
    for side in ("today", "filtered"):
        summary[f"found_top5_{side}"] = sum(in_top(v[side], 5) for v in answerable.values())
        summary[f"found_in_all_rows_{side}"] = sum(in_top(v[side], 10 ** 6) for v in answerable.values())
        summary[f"rows_total_{side}"] = sum(v[side]["n"] for v in per_q.values())
    flips = {}
    for label, n in (("top5", 5), ("all_rows", 10 ** 6)):
        flips[f"lost_{label}"] = [k for k, v in answerable.items() if in_top(v["today"], n) and not in_top(v["filtered"], n)]
        flips[f"gained_{label}"] = [k for k, v in answerable.items() if not in_top(v["today"], n) and in_top(v["filtered"], n)]
    flips["first_position_changed"] = [k for k, v in answerable.items()
                                       if v["today"]["first_pos"] != v["filtered"]["first_pos"]]
    out = {"comparison": "mcp-machine-filter", "dataset": {"git_commit": git_commit, "mode": "ref-vs-working-tree"},
           "summary": summary, "flips": flips, "per_question": per_q}
    if recorded:
        out["recorded"] = {
            "calls": len(recorded),
            "with_machine_rows_today": sum(1 for r in recorded if r["today"]["machine_python"] > 0),
            "with_machine_rows_filtered": sum(1 for r in recorded if r["filtered"]["machine_python"] > 0),
            "rows_today": sum(r["today"]["n"] for r in recorded),
            "rows_filtered": sum(r["filtered"]["n"] for r in recorded),
            "machine_rows_today": sum(r["today"]["machine_python"] for r in recorded),
            "machine_rows_filtered": sum(r["filtered"]["machine_python"] for r in recorded),
            "per_call": {f"r{i:02d}": r for i, r in enumerate(recorded, 1)}}
    return out


def render_mcp_text(out):
    s, f = out["summary"], out["flips"]
    lines = [f"MCP machine filter (ref vs working tree) - {s['questions']} questions ({s['answerable']} answerable)", ""]
    for d, name in (("machine_python", "Python rule (all whitespace, 60 chars)"),
                    ("machine_hook_sql", "hook SQL rule (spaces only, 80 chars)")):
        lines.append(f"Questions with at least one machine row, {name}: today {s[f'with_machine_rows_today_{d}']}, "
                     f"filtered {s[f'with_machine_rows_filtered_{d}']}")
    for side in ("today", "filtered"):
        lines.append(f"{side:9}: found in top 5 {s[f'found_top5_{side}']}, found in all returned rows "
                     f"{s[f'found_in_all_rows_{side}']}, rows returned {s[f'rows_total_{side}']}")
    lines += [f"Lost in top 5: {f['lost_top5']}  gained in top 5: {f['gained_top5']}",
              f"Lost in all rows: {f['lost_all_rows']}  gained in all rows: {f['gained_all_rows']}",
              f"First position changed: {len(f['first_position_changed'])} {f['first_position_changed']}", ""]
    if "recorded" in out:
        r = out["recorded"]
        lines += [f"Recorded real calls: {r['calls']}; calls with machine rows today {r['with_machine_rows_today']}, "
                  f"filtered {r['with_machine_rows_filtered']}",
                  f"  rows: today {r['rows_today']} ({r['machine_rows_today']} machine), filtered {r['rows_filtered']} "
                  f"({r['machine_rows_filtered']} machine)"]
        for cid, c in r["per_call"].items():
            lines.append(f"  {cid}: today {c['today']['n']} rows / {c['today']['machine_python']} machine; "
                         f"filtered {c['filtered']['n']} rows / {c['filtered']['machine_python']} machine")
    return "\n".join(lines) + "\n"


def _all_keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(str(k))
            _all_keys(v, acc)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _all_keys(v, acc)
    return acc


def main():
    ap = argparse.ArgumentParser(description="Paired comparisons (measurement only)")
    ap.add_argument("comparison", choices=COMPARISONS)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--default-project")
    ap.add_argument("--as-of")
    ap.add_argument("--check-against")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--simulate", action="store_true", help="mcp-gate: wrap the loaded server (measurement only)")
    ap.add_argument("--repo", help="mcp-machine-filter: the git repository holding --ref (default: this folder)")
    ap.add_argument("--ref", help="mcp-machine-filter: git ref of the tool before the change")
    ap.add_argument("--expect", help="mcp-machine-filter: a saved result the new numbers must equal exactly")
    ap.add_argument("--recorded", action="store_true", help="mcp-machine-filter: replay the recorded real calls")
    args = ap.parse_args()
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = resolve_db_path(config)
    if args.comparison == "mcp-gate":
        if not args.simulate:
            raise SystemExit("mcp-gate needs --simulate (the --ref proof comes with the real change)")
        conn = ev._ro_connect(db_path)
        try:
            started = time.perf_counter()
            per_q, recorded, floor = compare_mcp_gate(conn, db_path, args.golden, args.recorded)
            out = build_gate_output(per_q, recorded, floor, args.git_commit)
            cap.check_allowlist(out, extra_keys=_all_keys(out, set()),
                                extra_strings=list(per_q) + ["mcp-gate", "simulation", args.git_commit])
            outdir = pathlib.Path(args.out)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "compare_mcp-gate.json").write_text(json.dumps(out, indent=2) + "\n")
            (outdir / "compare_mcp-gate.txt").write_text(render_gate_text(out))
            print(json.dumps({"ok": True, "seconds": round(time.perf_counter() - started, 1)}))
        finally:
            conn.close()
        return
    if args.comparison == "mcp-machine-filter":
        if not args.ref:
            raise SystemExit("mcp-machine-filter needs --ref GIT_REF (the tool as it was before the change)")
        conn = ev._ro_connect(db_path)
        try:
            started = time.perf_counter()
            per_q, recorded = compare_mcp_machine_filter(conn, db_path, args.golden, args.recorded, args.ref, args.repo or ROOT)
            out = build_mcp_output(per_q, recorded, args.git_commit)
            if args.expect:
                want = json.loads(pathlib.Path(args.expect).read_text())
                bad = [k for k in differing_keys({x: out[x] for x in out if x != "dataset"},
                                                 {x: want[x] for x in want if x != "dataset"})]
                if bad:
                    print(f"STOPPED: the new numbers differ from {pathlib.Path(args.expect).name} at: {bad[:40]}",
                          file=sys.stderr)
                    sys.exit(4)
            cap.check_allowlist(out, extra_keys=_all_keys(out, set()),
                                extra_strings=list(per_q) + ["mcp-machine-filter", "ref-vs-working-tree", args.git_commit])
            outdir = pathlib.Path(args.out)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "compare_mcp-machine-filter.json").write_text(json.dumps(out, indent=2) + "\n")
            (outdir / "compare_mcp-machine-filter.txt").write_text(render_mcp_text(out))
            print(json.dumps({"ok": True, "seconds": round(time.perf_counter() - started, 1)}))
        finally:
            conn.close()
        return
    spec = importlib.util.spec_from_file_location("semantic_search", ROOT / "scripts" / "semantic_search.py")
    semantic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic)
    semantic.get_embed_model(config)
    conn = ev._ro_connect(db_path)
    try:
        started = time.perf_counter()
        try:
            recs, as_of = compare_project_bias(conn, config, semantic, args.golden, args.default_project, args.as_of)
            if args.check_against:
                bad = cross_check(recs, args.check_against)
                if bad:
                    raise CompareStop(f"bias-on side differs from the cap study on questions {bad}")
        except (CompareStop, ev.EvalError) as e:         # ids and field names only
            print(f"STOPPED: {e}", file=sys.stderr)
            sys.exit(4)
        out = build_output(recs, as_of, args.default_project, "cap_study.json" if args.check_against else None,
                           args.git_commit)
        qids = [r["id"] for r in recs]
        cap.check_allowlist(out, extra_keys=qids + EXTRA_KEYS + ["started_with_bias_off"] + list(cap.BUCKETS)
                            + ["starts_with_bias_off", "stops_with_bias_off"] + list(MODES),
                            extra_strings=qids + ["project-bias", "cap_study.json", args.default_project or "",
                                                  as_of, args.git_commit]
                            + [mm["sign_test"]["plain"] for mm in out["modes"].values()])
        outdir = pathlib.Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"compare_{args.comparison}.json").write_text(json.dumps(out, indent=2) + "\n")
        (outdir / f"compare_{args.comparison}.txt").write_text(render_text(out))
        print(json.dumps({"ok": True, "seconds": round(time.perf_counter() - started, 1)}))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
