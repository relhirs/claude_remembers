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

Call shape (the same as the cap study's primary call and the eval's run_question):
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
import importlib.util
import json
import math
import pathlib
import sys
import time

import yaml

from config_utils import resolve_db_path
import remember_eval as ev
import remember_cap_study as cap
import retrieval

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPARISONS = ("project-bias",)
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
    lines = [f"Project bias on vs off - {d['questions']} questions ({d['answerable']} answerable, {d['no_answer']} no-answer), "
             f"as_of {d['as_of']}, project {d['default_project'] or '(each question\'s own)'}",
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


def main():
    ap = argparse.ArgumentParser(description="Paired comparisons (measurement only)")
    ap.add_argument("comparison", choices=COMPARISONS)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--default-project")
    ap.add_argument("--as-of")
    ap.add_argument("--check-against")
    ap.add_argument("--git-commit", default="")
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
