#!/usr/bin/env python3
"""
hook_equivalence.py - Prove two versions of hooks/user-prompt-submit.py behave identically.

Not part of CI (it needs a database and, for the synthetic set, the embedding
model to build one). Run it by hand around any change to the hook:

    .venv/bin/python tests/hook_equivalence.py --ref 2d1805e                 # synthetic set
    .venv/bin/python tests/hook_equivalence.py --ref 2d1805e --dataset private \\
        --db eval/snapshots/private-2026-11-23.db --golden eval/golden_set.yaml \\
        --config-from config.yaml --limit 300

The OLD hook (and its scripts/) come from git ref `--ref`; the NEW one is the
working tree. Each variant gets its own sandbox folder, its own fresh copy of
the database and its own logs/, so neither run can affect the other (the
post-/clear mark and the retrieval_events rows are written per copy). The same
scenarios are replayed through hook._run() in both, with meaning search and the
warm helper replaced by deterministic stand-ins (no model, no helper process,
and the real helper is never started).

Per scenario it compares: the exact bytes the hook prints (as a sha256),
the lines it logs (timestamps stripped, prompt text hashed), the telemetry dict
`ev` (timings stripped, INCLUDING which keys are present), the
retrieval_events rows it writes (ts/timings/id stripped), how often meaning
search and the helper were called, and whether the /clear mark was set.
Message text is only ever compared as hashes, so a private run prints no
memory content; a mismatch names the scenario and the field only.

Every scenario is LABELED when it is created: folder_kind "mapped" (some mapping
entry matches its folder) or "unmapped" (none does). Nothing reads the code under
test to decide a label. The OLD hook's own project_for_cwd() then confirms every
label before any comparison; a contradicted label stops the run.

Three runs make the proof for a change in how unmapped folders rank results
(Phase 4, B1): the old hook, the new hook, and "old hook without bias" (the old
hook with its retrieve() wrapped so project="oth" becomes None, on unmapped
scenarios only). It passes when
  1. every mapped scenario (including every folder mapped to "oth") is identical
     to the old hook on every field;
  2. every unmapped scenario is identical to "old hook without bias";
  3. clear:unmapped-pending is also identical to the plain old hook (post-/clear
     is untouched);
  4. on a private run, at least one unmapped scenario really changed.
Without a label-aware change, use a ref whose hook behaves the same and the
unmapped scenarios simply match the old hook too.

Exit code 0 = identical (as above), 1 = something differs.
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
SELF = pathlib.Path(__file__).resolve()

TIMING_KEYS = {"ts", "keyword_ms", "semantic_ms", "latency_ms", "id"}
FUZZY_FLOAT_LISTS = {"result_ages_days"}      # ages are measured against the clock
AGE_TOLERANCE_DAYS = 0.11

SYNTHETIC_PROJECTS = {"tidepool": "tdp", "ledgerline": "lgl", "quillstack": "qst"}
MODES = ("off", "on_miss", "always_up", "always_down")


def _sha(text):
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# Runner: executes inside one sandbox, imports THAT sandbox's hook
# ---------------------------------------------------------------------------

def _stub_semantic(kind):
    def stub(conn, config, prompt, project=None, limit=10, query_vec=None):
        stub.calls.append({"limit": limit, "query_vec": None if query_vec is None else list(query_vec)})
        if kind == "raise":
            raise RuntimeError("stub meaning search failure")
        rows = conn.execute(
            "SELECT t.project, t.timestamp, t.content, t.id, t.session_id FROM transcript_embeddings te "
            "JOIN transcripts t ON t.id = te.transcript_id ORDER BY t.id LIMIT 600").fetchall()
        out = []
        for proj, ts, content, rid, sid in rows:
            sim = 0.15 + ((rid * 37 + len(prompt) * 11) % 70) / 100.0
            out.append({"similarity": sim, "project": proj, "timestamp": ts, "content": content,
                        "id": rid, "session_id": sid})
        out.sort(key=lambda r: (-r["similarity"], r["id"]))
        return out[:limit]
    stub.calls = []
    return stub


def _ensure_clear_columns(conn):
    have = {r[1] for r in conn.execute("PRAGMA table_info(sys_sessions)")}
    for col, ddl in (("end_reason", "end_reason TEXT"), ("clear_recovery_shown", "clear_recovery_shown INTEGER DEFAULT 0")):
        if col not in have:
            conn.execute(f"ALTER TABLE sys_sessions ADD COLUMN {ddl}")


def _clean_event(row):
    out = {}
    for k, v in row.items():
        if k in TIMING_KEYS:
            continue
        if k in FUZZY_FLOAT_LISTS and isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        out[k] = v
    return out


def run_scenarios(root, scenarios, strip_oth_bias=False):
    import contextlib
    import importlib.util
    import io
    import time

    root = pathlib.Path(root)
    base_config = json.loads((root / "base_config.json").read_text())
    spec = importlib.util.spec_from_file_location("hook_under_test", root / "hooks" / "user-prompt-submit.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    db_path = base_config["storage"]["local_db_path"]
    log_path = root / "logs" / "retrieval.log"
    helper = {"started": 0}
    real_retrieve = getattr(hook, "retrieve", None)

    def retrieve_without_oth_bias(*a, **k):
        # the stand-in: the old hook's search with its catch-all project favored by nobody
        if k.get("project") == "oth":
            k["project"] = None
        return real_retrieve(*a, **k)
    hook._start_helper = lambda r, c: helper.__setitem__("started", helper["started"] + 1)

    results = []
    for sc in scenarios:
        cfg = json.loads(json.dumps(base_config))
        cfg.update(sc.get("config_extra") or {})
        sem = cfg.setdefault("semantic_search", {})
        mode = sc["mode"]
        sem["enabled"] = mode != "off"
        sem["mode"] = "always" if mode.startswith("always") else "on_miss"
        (root / "config.yaml").write_text(yaml.safe_dump(cfg))

        stub = None
        if sc.get("semantic") == "missing":
            hook.semantic_search = None
        else:
            stub = _stub_semantic("raise" if sc.get("semantic") == "raise" else "ok")
            hook.semantic_search = stub
        hook._helper_vector = (lambda r, t: [0.1, 0.2]) if mode == "always_up" else (lambda r, t: None)
        helper["started"] = 0
        os.environ["CWD"] = sc["cwd"]
        if real_retrieve is not None:
            hook.retrieve = (retrieve_without_oth_bias
                             if strip_oth_bias and sc.get("folder_kind") == "unmapped" else real_retrieve)
        # what THIS sandbox's hook says about the folder, for the label check
        norm = hook.normalize_for_match
        probe = {"project": hook.project_for_cwd(cfg, sc["cwd"]),
                 "entry_match": any(norm(f) in norm(sc["cwd"]) for f in (cfg.get("jsonl_project_mapping") or {}))}

        conn = sqlite3.connect(db_path)
        pending = sc.get("pending_clear")
        if pending:
            _ensure_clear_columns(conn)
            conn.execute("DELETE FROM sys_sessions WHERE session_id = ?", (pending["session"],))
            conn.execute(
                "INSERT INTO sys_sessions (session_id, project, ended_at, message_count, notes, end_reason, "
                "clear_recovery_shown) VALUES (?, ?, '2026-01-01T00:00:00', 7, ?, 'clear', 0)",
                (pending["session"], pending["project"], "Topic: equivalence fixture summary"))
        conn.commit()
        try:
            max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM retrieval_events").fetchone()[0]
        except sqlite3.Error:
            max_id = 0
        conn.close()
        log_before = log_path.stat().st_size if log_path.exists() else 0

        raw = sc["raw"] if "raw" in sc else json.dumps({"session_id": sc["session_id"], "prompt": sc["prompt"]})
        ev = {"ts": "T", "source": "live"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hook._run(raw, ev)
            ev_view = {k: v for k, v in ev.items() if not k.startswith("_")}
            hook._finish_event(ev, time.perf_counter())
        stdout = buf.getvalue()

        log_lines = []
        if log_path.exists():
            with open(log_path, "rb") as f:
                f.seek(log_before)
                text = f.read().decode("utf-8", "replace")
            for line in text.splitlines():
                parts = line.split(" ", 4)
                # ts [session] outcome n=N rest
                log_lines.append({"session": parts[1] if len(parts) > 1 else None,
                                  "outcome": parts[2] if len(parts) > 2 else None,
                                  "n": parts[3] if len(parts) > 3 else None,
                                  "rest_sha": _sha(parts[4] if len(parts) > 4 else "")})

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            events = [_clean_event(dict(r)) for r in
                      conn.execute("SELECT * FROM retrieval_events WHERE id > ? ORDER BY id", (max_id,))]
        except sqlite3.Error:
            events = None
        shown = None
        if pending:
            shown = conn.execute("SELECT clear_recovery_shown FROM sys_sessions WHERE session_id = ?",
                                 (pending["session"],)).fetchone()[0]
        conn.close()

        ev_clean = {k: v for k, v in ev_view.items() if k not in TIMING_KEYS}
        results.append({
            "name": sc["name"],
            "stdout_sha": _sha(stdout), "stdout_len": len(stdout),
            "stdout_kind": ("empty" if stdout.strip() == "{}" else "injected" if stdout.strip() else "blank"),
            "log": log_lines,
            "ev": ev_clean,
            "ev_keys": sorted(ev_clean),
            "events": events,
            "semantic_calls": None if stub is None else stub.calls,
            "helper_started": helper["started"],
            "clear_shown": shown,
            "_probe": probe,
        })
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_variant(base, label, source, db_src, base_config):
    """A sandbox root holding the hook + scripts from `source` ('worktree' or a git ref)."""
    root = pathlib.Path(base) / label
    root.mkdir(parents=True)
    if source == "worktree":
        (root / "hooks").mkdir()
        (root / "scripts").mkdir()
        shutil.copy(REPO / "hooks" / "user-prompt-submit.py", root / "hooks")
        for f in (REPO / "scripts").glob("*.py"):
            shutil.copy(f, root / "scripts")
    else:
        arch = subprocess.run(["git", "archive", source, "hooks/user-prompt-submit.py", "scripts"],
                              cwd=REPO, capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(root)], input=arch.stdout, check=True)
    db_copy = root / "claude_remembers.db"
    src = sqlite3.connect(pathlib.Path(db_src).resolve().as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(db_copy)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    cfg = json.loads(json.dumps(base_config))
    cfg["storage"] = {"mode": "synced", "local_db_path": str(db_copy), "root_path": root.as_posix()}
    (root / "base_config.json").write_text(json.dumps(cfg))
    return root


UNMAPPED_CWD = "/work/unmapped-folder"


def folder_for(project, mapping):
    """(folder, label, prefix the folder was built from). A mapping entry for the
    project gives a "mapped" folder; the fallback folder is "unmapped". The label
    is set here, from how the folder was made, never from the hook's answer."""
    for folder, prefix in mapping.items():
        if prefix == project:
            return f"/work/{folder}", "mapped", prefix
    return UNMAPPED_CWD, "unmapped", None


def cwd_for(project, mapping):
    return folder_for(project, mapping)[0]


def _labeled(sc, project, mapping):
    cwd, kind, prefix = folder_for(project, mapping)
    sc.update({"cwd": cwd, "folder_kind": kind, "expected_prefix": prefix})
    return sc


def make_scenarios(db_path, mapping, golden_path, limit):
    scenarios = []
    conn = sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        projects = [r[0] for r in conn.execute("SELECT project FROM transcripts GROUP BY project ORDER BY COUNT(*) DESC")]
        main_project = next((p for p in projects if p in mapping.values()), projects[0])

        if golden_path:
            data = yaml.safe_load(pathlib.Path(golden_path).read_text())
            questions = data["questions"] if isinstance(data, dict) else data
            for q in questions:
                project = q.get("project") or main_project
                for mode in MODES:
                    scenarios.append(_labeled({"name": f"q:{q['id']}:{mode}", "prompt": q["question"],
                                               "session_id": "equiv-session", "mode": mode}, project, mapping))
                # the same question from a folder no mapping entry matches
                for mode in MODES:
                    scenarios.append({"name": f"qu:{q['id']}:{mode}", "prompt": q["question"],
                                      "session_id": "equiv-session", "cwd": UNMAPPED_CWD, "mode": mode,
                                      "folder_kind": "unmapped", "expected_prefix": None})
                # and from a folder deliberately mapped to "oth" (only if the config has one)
                if "oth" in mapping.values():
                    scenarios.append(_labeled({"name": f"mapped-oth:{q['id']}:on_miss", "prompt": q["question"],
                                               "session_id": "equiv-session", "mode": "on_miss"}, "oth", mapping))

        rows = conn.execute(
            "SELECT session_id, project, content FROM transcripts WHERE type = 'user' AND role = 'user' "
            "AND is_subagent = 0 AND LENGTH(TRIM(content)) > 0 ORDER BY timestamp").fetchall()
        stride = max(1, len(rows) // limit) if limit else 1
        for i, (sid, project, content) in enumerate(rows[::stride]):
            scenarios.append(_labeled({"name": f"p:{i}", "prompt": content[:4000], "session_id": sid,
                                       "mode": MODES[i % 4]}, project, mapping))
        sample = next((c for _, _, c in rows if len(c.strip()) >= 15), "what did we decide about the plan")
        long_prompt = sample[:300]
    finally:
        conn.close()

    cwd, cwd_kind, cwd_prefix = folder_for(main_project, mapping)
    edge = [
        ("edge:empty-prompt", {"prompt": ""}),
        ("edge:short-prompt", {"prompt": "ok thanks"}),
        ("edge:all-stop-words", {"prompt": "what is the and for with this that"}),
        ("edge:nonsense", {"prompt": "zzqplex wibblefrotz quuxborg nothing matches here"}),
        ("edge:invalid-json", {"raw": "{not json"}),
        ("edge:missing-prompt-key", {"raw": json.dumps({"session_id": "equiv-session"})}),
        ("edge:non-string-prompt", {"raw": json.dumps({"session_id": "equiv-session", "prompt": 42})}),
        ("edge:no-session-id", {"raw": json.dumps({"prompt": long_prompt})}),
        ("edge:memory-phrasing", {"prompt": "what did we decide about " + long_prompt[:120]}),
    ]
    for name, extra in edge:
        for mode in MODES:
            sc = {"name": f"{name}:{mode}", "session_id": "equiv-session", "cwd": cwd, "mode": mode,
                  "prompt": long_prompt, "folder_kind": cwd_kind, "expected_prefix": cwd_prefix}
            sc.update(extra)
            scenarios.append(sc)
    base = {"folder_kind": cwd_kind, "expected_prefix": cwd_prefix}
    scenarios += [{**base, **sc} for sc in [
        {"name": "edge:semantic-missing:on_miss", "prompt": "zzqplex wibblefrotz quuxborg nothing matches here",
         "session_id": "equiv-session", "cwd": cwd, "mode": "on_miss", "semantic": "missing"},
        {"name": "edge:semantic-missing:always", "prompt": long_prompt, "session_id": "equiv-session",
         "cwd": cwd, "mode": "always_up", "semantic": "missing"},
        {"name": "edge:semantic-raises:on_miss", "prompt": "zzqplex wibblefrotz quuxborg nothing matches here",
         "session_id": "equiv-session", "cwd": cwd, "mode": "on_miss", "semantic": "raise"},
        {"name": "edge:semantic-raises:always", "prompt": long_prompt, "session_id": "equiv-session",
         "cwd": cwd, "mode": "always_up", "semantic": "raise"},
        {"name": "edge:metrics-off", "prompt": long_prompt, "session_id": "equiv-session", "cwd": cwd,
         "mode": "on_miss", "config_extra": {"metrics": {"enabled": False}}},
        {"name": "edge:unmapped-folder", "prompt": long_prompt, "session_id": "equiv-session",
         "cwd": "/nowhere/special", "mode": "on_miss", "folder_kind": "unmapped", "expected_prefix": None},
    ]]
    pending = {"session": "equiv-old-clear", "project": main_project}
    scenarios += [{**base, **sc} for sc in [
        {"name": "clear:first-prompt", "prompt": long_prompt, "session_id": "equiv-new-session", "cwd": cwd,
         "mode": "on_miss", "pending_clear": pending},
        {"name": "clear:second-prompt", "prompt": long_prompt, "session_id": "equiv-new-session", "cwd": cwd,
         "mode": "on_miss"},
        {"name": "clear:short-prompt", "prompt": "continue", "session_id": "equiv-new-session-2", "cwd": cwd,
         "mode": "on_miss", "pending_clear": pending},
    ]]
    # an unmapped folder with a pending /clear under "oth": post-/clear must be untouched
    scenarios.append({"name": "clear:unmapped-pending", "prompt": long_prompt, "session_id": "equiv-new-session-3",
                      "cwd": UNMAPPED_CWD, "mode": "on_miss", "folder_kind": "unmapped", "expected_prefix": None,
                      "pending_clear": {"session": "equiv-old-clear-unmapped", "project": "oth"}})
    return scenarios


def build_synthetic_db(base):
    """The public synthetic database, built the way run_eval.py builds it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_eval_mod", REPO / "examples" / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)
    root = run_eval.make_sandbox(base)
    db = str(root / "claude_remembers.db")
    run_eval.write_config(root, {"storage": {"mode": "local", "root_path": root.as_posix()},
                                 "jsonl_project_mapping": SYNTHETIC_PROJECTS,
                                 "semantic_search": {"enabled": True, "model": run_eval.MODEL}})
    run_eval.build_schema(root, db)
    run_eval.ingest_sessions(root)
    return db


def run_variant(root, scenarios, out_dir, strip_oth_bias=False):
    sc_file = pathlib.Path(out_dir) / f"{root.name}-scenarios.json"
    out_file = pathlib.Path(out_dir) / f"{root.name}-results.json"
    sc_file.write_text(json.dumps(scenarios))
    res = subprocess.run([sys.executable, str(SELF), "--runner", str(root), str(sc_file), str(out_file)]
                         + (["--strip-oth-bias"] if strip_oth_bias else []),
                         capture_output=True, text=True, timeout=3600, cwd=str(root))
    if res.returncode != 0:
        raise SystemExit(f"runner failed for {root.name}:\n{res.stderr[-3000:]}")
    return json.loads(out_file.read_text())


def _close(a, b):
    """Equal, except lists of ages may differ by clock drift."""
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a and \
            all(isinstance(x, (int, float, type(None))) for x in a + b):
        return all((x is None and y is None) or (x is not None and y is not None and abs(x - y) <= AGE_TOLERANCE_DAYS)
                   for x, y in zip(a, b))
    return a == b


def compare(old, new):
    diffs = []
    if len(old) != len(new):
        return [("<all>", "scenario count", len(old), len(new))]
    for o, n in zip(old, new):
        for key in sorted(set(o) | set(n)):       # a field only the new run has is a difference too
            if key.startswith("_"):               # label probes, not behavior
                continue
            if key not in o or key not in n:
                diffs.append((o["name"], f"field presence: {key}", key in o, key in n))
            elif key == "ev":
                for k in sorted(set(o["ev"]) | set(n["ev"])):
                    if k not in o["ev"] or k not in n["ev"]:
                        diffs.append((o["name"], f"ev key presence: {k}", k in o["ev"], k in n["ev"]))
                    elif not _close(o["ev"][k], n["ev"][k]):
                        diffs.append((o["name"], f"ev[{k}]", o["ev"][k], n["ev"][k]))
            elif key == "events":
                oe, ne = o["events"], n["events"]
                # `oe and ...` would skip the length check when the OLD hook wrote
                # no rows (an empty list is falsy), hiding rows only the new hook writes.
                if (oe is None) != (ne is None) or (oe is not None and len(oe) != len(ne)):
                    diffs.append((o["name"], "events (row count)", None if oe is None else len(oe),
                                  None if ne is None else len(ne)))
                elif oe:
                    for ro, rn in zip(oe, ne):
                        for k in sorted(set(ro) | set(rn)):
                            if not _close(ro.get(k), rn.get(k)):
                                diffs.append((o["name"], f"event.{k}", ro.get(k), rn.get(k)))
            elif o[key] != n[key]:
                diffs.append((o["name"], key, o[key] if key != "log" else len(o[key]),
                              n[key] if key != "log" else len(n[key])))
    return diffs


def check_labels(scenarios, old_results):
    """Names of scenarios whose label the OLD hook's own project_for_cwd() contradicts.
    unmapped: no entry matches and the project is "oth". mapped: an entry matches and
    the project is the prefix the folder was built from."""
    probes = {r["name"]: r["_probe"] for r in old_results}
    bad = []
    for sc in scenarios:
        probe = probes[sc["name"]]
        if sc["folder_kind"] == "unmapped":
            ok = probe["project"] == "oth" and not probe["entry_match"]
        else:
            ok = probe["entry_match"] and probe["project"] == sc["expected_prefix"]
        if not ok:
            bad.append(sc["name"])
    return bad


def _fields(diffs):
    out = {}
    for _name, field, _a, _b in diffs:
        out[field] = out.get(field, 0) + 1
    return out


def judge(scenarios, old, new, standin, private):
    """The pass rule (see the module docstring). Returns a report; report["ok"] is the verdict."""
    O, N, S = ({r["name"]: r for r in rs} for rs in (old, new, standin))
    rep = {"mapped_total": 0, "mapped_differ": {}, "unmapped_total": 0, "unmapped_changed": {},
           "unmapped_not_standin": {}, "clear_unmapped_differs": None}
    for sc in scenarios:
        name = sc["name"]
        vs_old = compare([O[name]], [N[name]])
        if sc["folder_kind"] == "mapped":
            rep["mapped_total"] += 1
            if vs_old:
                rep["mapped_differ"][name] = _fields(vs_old)
            continue
        rep["unmapped_total"] += 1
        vs_standin = compare([S[name]], [N[name]])
        if vs_standin:
            rep["unmapped_not_standin"][name] = _fields(vs_standin)
        if vs_old:
            rep["unmapped_changed"][name] = _fields(vs_old)
        if name == "clear:unmapped-pending":
            rep["clear_unmapped_differs"] = bool(vs_old)
    rep["private_exercised"] = bool(rep["unmapped_changed"])
    rep["ok"] = (not rep["mapped_differ"] and not rep["unmapped_not_standin"]
                 and not rep["clear_unmapped_differs"]
                 and (rep["private_exercised"] or not private))
    return rep


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == "--runner":
        results = run_scenarios(sys.argv[2], json.loads(pathlib.Path(sys.argv[3]).read_text()),
                                strip_oth_bias="--strip-oth-bias" in sys.argv[5:])
        pathlib.Path(sys.argv[4]).write_text(json.dumps(results))
        return 0

    p = argparse.ArgumentParser(description="Prove two versions of the hook behave identically")
    p.add_argument("--ref", default="HEAD", help="git ref of the OLD hook + scripts (default HEAD)")
    p.add_argument("--dataset", choices=["synthetic", "private"], default="synthetic")
    p.add_argument("--db", help="database to replay against (required for --dataset private)")
    p.add_argument("--golden", help="golden set whose questions are replayed")
    p.add_argument("--config-from", help="config.yaml to take jsonl_project_mapping from")
    p.add_argument("--limit", type=int, default=300, help="max stored prompts to replay (spread evenly)")
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()

    base = tempfile.mkdtemp(prefix="hook_equiv_")
    try:
        snapshot_sha = None
        if args.dataset == "synthetic":
            db = args.db or build_synthetic_db(os.path.join(base, "synthetic_build"))
            mapping = SYNTHETIC_PROJECTS
            golden = args.golden or str(REPO / "examples" / "eval" / "golden_set.yaml")
            base_config = {"jsonl_project_mapping": mapping, "semantic_search": {"model": "all-MiniLM-L6-v2"}}
        else:
            if not args.db:
                raise SystemExit("--dataset private needs --db")
            db = args.db
            cfg = yaml.safe_load(pathlib.Path(args.config_from or REPO / "config.yaml").read_text()) or {}
            mapping = cfg.get("jsonl_project_mapping") or {}
            golden = args.golden
            base_config = {"jsonl_project_mapping": mapping, "semantic_search": {"model": "all-MiniLM-L6-v2"}}
            snapshot_sha = hashlib.sha256(pathlib.Path(db).read_bytes()).hexdigest()

        scenarios = make_scenarios(db, mapping, golden, args.limit)
        print(f"{len(scenarios)} scenarios; old = {args.ref}, new = working tree", file=sys.stderr)
        old_root = build_variant(base, "old", args.ref, db, base_config)
        new_root = build_variant(base, "new", "worktree", db, base_config)
        standin_root = build_variant(base, "oldnobias", args.ref, db, base_config)
        old = run_variant(old_root, scenarios, base)
        # label check first: no comparison number is written if a label is wrong
        bad_labels = check_labels(scenarios, old)
        if bad_labels:
            print(f"LABEL CHECK FAILED: the old hook's project_for_cwd() contradicts the label of "
                  f"{len(bad_labels)} scenarios: {bad_labels[:40]}")
            return 1
        n_un = sum(1 for sc in scenarios if sc["folder_kind"] == "unmapped")
        print(f"label check passed: {len(scenarios) - n_un} mapped and {n_un} unmapped scenarios agree "
              "with the old hook's project_for_cwd()")
        new = run_variant(new_root, scenarios, base)
        standin = run_variant(standin_root, scenarios, base, strip_oth_bias=True)

        if snapshot_sha:
            after = hashlib.sha256(pathlib.Path(db).read_bytes()).hexdigest()
            print(f"source database unchanged: {after == snapshot_sha}", file=sys.stderr)
            if after != snapshot_sha:
                return 1

        diffs = []                                  # replaced by the labeled rules below
        kinds = {}
        for r in new:
            kinds[r["stdout_kind"]] = kinds.get(r["stdout_kind"], 0) + 1
        outcomes = {}
        for r in new:
            outcomes[r["ev"].get("outcome")] = outcomes.get(r["ev"].get("outcome"), 0) + 1
        print(f"scenarios: {len(old)}  stdout kinds: {kinds}")
        print(f"outcomes seen: {dict(sorted(outcomes.items(), key=lambda kv: str(kv[0])))}")
        rep = judge(scenarios, old, new, standin, private=args.dataset == "private")
        print(f"mapped scenarios: {rep['mapped_total']}, differing from the old hook: {len(rep['mapped_differ'])}")
        print(f"unmapped scenarios: {rep['unmapped_total']}, changed vs the old hook: {len(rep['unmapped_changed'])}, "
              f"unchanged: {rep['unmapped_total'] - len(rep['unmapped_changed'])}")
        for name, fields in list(rep["unmapped_changed"].items())[:60]:
            print(f"  changed: {name} :: fields {dict(sorted(fields.items()))}")
        if len(rep["unmapped_changed"]) > 60:
            print(f"  ... and {len(rep['unmapped_changed']) - 60} more changed unmapped scenarios")
        n_oth = sqlite3.connect(pathlib.Path(db).resolve().as_uri() + "?mode=ro", uri=True).execute(
            "SELECT COUNT(*) FROM transcripts WHERE project = 'oth'").fetchone()[0]
        if n_oth == 0:
            print("this dataset has no `oth` rows, so it does not exercise the change")
        print(f"clear:unmapped-pending identical to the plain old hook: {rep['clear_unmapped_differs'] is False}")
        if rep["mapped_differ"]:
            print("FAIL: mapped scenarios must be identical to the old hook:")
            for name, fields in list(rep["mapped_differ"].items())[:40]:
                print(f"  {name} :: fields {dict(sorted(fields.items()))}")
        if rep["unmapped_not_standin"]:
            print("FAIL: unmapped scenarios must equal 'old hook without bias':")
            for name, fields in list(rep["unmapped_not_standin"].items())[:40]:
                print(f"  {name} :: fields {dict(sorted(fields.items()))}")
        if args.dataset == "private" and not rep["private_exercised"]:
            print("FAIL: no unmapped scenario changed, so this private run did not exercise the change")
        if not rep["ok"]:
            return 1
        print("PASS: mapped scenarios identical to the old hook; unmapped scenarios identical to the old hook "
              "without bias (stdout bytes, log lines, ev incl. key presence, event rows, meaning-search and "
              "helper calls, /clear mark).")
        return 0
    finally:
        if args.keep:
            print(f"kept {base}", file=sys.stderr)
        else:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
