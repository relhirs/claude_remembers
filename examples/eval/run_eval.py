#!/usr/bin/env python3
"""
run_eval.py - Reproduce the retrieval numbers on the public synthetic dataset.

    python3 examples/eval/run_eval.py

From a clean state, this:
  1. builds a throwaway copy of the project in a temp folder (hook, MCP
     server and scripts/, copied unchanged) with its own config.yaml, so your
     real config, database and logs/ are never opened;
  2. creates the real schema (the DDL constants in scripts/remember-setup.py,
     loaded without running setup) and ingests examples/eval/sessions/*.jsonl
     with the real ingest code: the FTS index is filled by the real trigger
     and each message is embedded with the real model, as live capture does;
  3. runs scripts/remember_eval.py on examples/eval/golden_set.yaml (every
     method, plus the MCP search_transcripts tool) and
     scripts/remember_baseline.py twice: on the 26 golden questions (each from
     its own project, at the golden set's fixed as_of) and on the stored
     session prompts (each at its own timestamp);
  4. writes everything, stamped with the git commit and date, to
     examples/eval/results/<date>-<commit>/.

Options:
  --keyword-only   no embedding model (what CI runs): keyword, keyword_ungated
                   and mcp are scored; meaning, on_miss and always are not.
  --db PATH        score an existing database instead of building one. It is
                   only read; its sha256 is checked before and after.
  --golden PATH    golden set to use (default: examples/eval/golden_set.yaml).
  --config-from P  take every setting except storage from this config.yaml
                   (default for --db runs: none; synthetic runs use a fixed one).
  --out DIR        where to write results.
  --keep           keep the temp sandbox and print where it is.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = REPO / "examples" / "eval"
SESSIONS = HERE / "sessions"
DEFAULT_GOLDEN = HERE / "golden_set.yaml"
# Files whose code decides what retrieval returns. The stamp records whether
# they match the commit, so a result can be tied to exact retrieval code.
RETRIEVAL_FILES = ["hooks/user-prompt-submit.py", "mcp/server.py", "scripts/semantic_search.py",
                   "scripts/config_utils.py", "scripts/fuzzy_search.py"]
MODEL = "all-MiniLM-L6-v2"
SYNTHETIC_PROJECTS = {"tidepool": "tdp", "ledgerline": "lgl", "quillstack": "qst"}


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)


def stamp():
    commit = _git("rev-parse", "--short", "HEAD").stdout.strip() or None
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        # Tracked files only: an untracked scratch file says nothing about the code.
        "uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no").stdout.strip()),
        "retrieval_code_matches_commit": _git("diff", "--quiet", "HEAD", "--", *RETRIEVAL_FILES).returncode == 0,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "platform": sys.platform,
    }


def _rel(path):
    """Repo-relative path for anything written to results (never an absolute path)."""
    try:
        return pathlib.Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return pathlib.Path(path).name


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sandbox(base):
    root = pathlib.Path(base) / "root"
    for d in ("hooks", "mcp", "scripts"):
        (root / d).mkdir(parents=True)
    shutil.copy(REPO / "hooks" / "user-prompt-submit.py", root / "hooks")
    shutil.copy(REPO / "mcp" / "server.py", root / "mcp")
    for f in (REPO / "scripts").glob("*.py"):
        shutil.copy(f, root / "scripts")
    return root


def write_config(root, config):
    with open(root / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def build_schema(root, db_path):
    """The real schema, from remember-setup.py's DDL constants. Only the
    constants are used; none of the setup functions run."""
    sys.path.insert(0, str(root / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("remember_setup", root / "scripts" / "remember-setup.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        conn = sqlite3.connect(db_path)
        try:
            for ddl in (setup.DDL_STATEMENTS + setup.DDL_INDEXES + [setup.DDL_FTS, setup.DDL_FTS_VOCAB]
                        + setup.DDL_TRIGGERS):
                conn.execute(ddl)
            setup.ensure_metrics_schema(conn)
            conn.commit()
        finally:
            conn.close()
    finally:
        sys.path.remove(str(root / "scripts"))


INGEST = """
import json, sys
sys.path.insert(0, sys.argv[1] + "/scripts")
import ingest_jsonl
out = []
for path in sys.argv[2:]:
    r = ingest_jsonl.ingest(path, root_path=sys.argv[1])
    out.append([path, r["records_imported"], r["exit_code"]])
print(json.dumps(out))
"""


def ingest_sessions(root):
    """All session files through ingest_jsonl.ingest() (what its CLI calls),
    in one process so the embedding model loads once."""
    files = sorted(str(p) for p in SESSIONS.glob("*.jsonl"))
    res = subprocess.run([sys.executable, "-c", INGEST, str(root), *files],
                         capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise SystemExit(f"ingest failed:\n{res.stderr[-2000:]}")
    results = json.loads(res.stdout.strip().splitlines()[-1])
    bad = [r for r in results if r[2] != 0]
    if bad:
        raise SystemExit(f"ingest returned errors for: {[pathlib.Path(b[0]).name for b in bad]}")
    return sum(r[1] for r in results), len(results)


def db_facts(db_path):
    conn = sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        one = lambda sql: conn.execute(sql).fetchone()[0]
        return {
            "sessions": one("SELECT COUNT(DISTINCT session_id) FROM transcripts"),
            "messages": one("SELECT COUNT(*) FROM transcripts"),
            "messages_50_chars_or_more": one("SELECT COUNT(*) FROM transcripts WHERE LENGTH(TRIM(content)) >= 50"),
            "embedded": one("SELECT COUNT(*) FROM transcript_embeddings"),
            "fts_rows": one("SELECT COUNT(*) FROM transcripts_fts_docsize"),
            "projects": dict(conn.execute("SELECT project, COUNT(*) FROM transcripts GROUP BY project").fetchall()),
        }
    finally:
        conn.close()


def run_script(root, name, *args):
    res = subprocess.run([sys.executable, str(root / "scripts" / name), *args], cwd=root,
                         capture_output=True, text=True, timeout=3600)
    if res.returncode != 0:
        raise SystemExit(f"{name} failed:\n{res.stderr[-3000:]}\n{res.stdout[-1000:]}")
    return json.loads(res.stdout)


def render(root, eval_result):
    sys.path.insert(0, str(root / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("sandbox_remember_eval", root / "scripts" / "remember_eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.render_report(eval_result)
    finally:
        sys.path.remove(str(root / "scripts"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--keyword-only", action="store_true")
    p.add_argument("--db")
    p.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    p.add_argument("--config-from")
    p.add_argument("--out")
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()

    st = stamp()
    golden = pathlib.Path(args.golden).resolve()
    out = pathlib.Path(args.out).resolve() if args.out else (
        HERE / "results" / f"{st['date'][:10]}-{st['commit'] or 'nocommit'}{'-kw' if args.keyword_only else ''}")
    out.mkdir(parents=True, exist_ok=True)

    base = tempfile.mkdtemp(prefix="remember_eval_sandbox_")
    try:
        root = make_sandbox(base)
        shutil.copy(golden, root / "golden_set.yaml")
        if args.db:
            db_path = str(pathlib.Path(args.db).resolve())
            config = {}
            if args.config_from:
                config = yaml.safe_load(pathlib.Path(args.config_from).read_text()) or {}
            # Only storage changes: point at the given file, keep the sandbox as root.
            config["storage"] = {"mode": "synced", "local_db_path": db_path, "root_path": root.as_posix()}
            write_config(root, config)
            db_hash_before = _sha256(db_path)
            facts_note = {"database": _rel(db_path), "sha256_before": db_hash_before}
        else:
            db_path = str(root / "claude_remembers.db")
            config = {"storage": {"mode": "local", "root_path": root.as_posix()},
                      "jsonl_project_mapping": SYNTHETIC_PROJECTS,
                      "semantic_search": {"enabled": not args.keyword_only, "model": MODEL}}
            write_config(root, config)
            build_schema(root, db_path)
            print("Ingesting synthetic sessions...", file=sys.stderr)
            imported, n_files = ingest_sessions(root)
            facts_note = {"database": "built from examples/eval/sessions", "files": n_files,
                          "records_imported": imported}
        facts = db_facts(db_path)
        sem = config.get("semantic_search") or {}
        run_info = {
            **st, "keyword_only": args.keyword_only, "golden": _rel(golden), **facts_note,
            "database_facts": facts,
            "config_used": {"semantic_search.enabled": sem.get("enabled"),
                            "semantic_search.model": sem.get("model"),
                            "semantic_search.mode": sem.get("mode", "(not set: on_miss)")},
        }

        print("Running remember_eval.py...", file=sys.stderr)
        ev_args = ["--golden", str(root / "golden_set.yaml"), "--no-save", "--json"]
        ev = run_script(root, "remember_eval.py", *ev_args, *(["--keyword-only"] if args.keyword_only else []))
        ev["params"]["golden"] = _rel(golden)
        ev["git_commit"] = f"{st['commit']}{'+uncommitted' if st['uncommitted_changes'] else ''}"

        print("Replaying the hook (fixed time)...", file=sys.stderr)
        replays = {}
        golden_data = yaml.safe_load(golden.read_text())
        if isinstance(golden_data, dict) and golden_data.get("as_of"):
            replays["questions"] = run_script(root, "remember_baseline.py", "--no-save", "--json",
                                              "--questions", str(root / "golden_set.yaml"),
                                              "--outputs", str(out / "replay_questions.jsonl"))
        replays["prompts"] = run_script(root, "remember_baseline.py", "--no-save", "--json",
                                        "--outputs", str(out / "replay_prompts.jsonl"))
        for r in replays.values():
            r["git_commit"] = ev["git_commit"]

        if args.db:
            run_info["sha256_after"] = _sha256(db_path)
            run_info["database_unchanged"] = run_info["sha256_after"] == db_hash_before
            if not run_info["database_unchanged"]:
                print("WARNING: the database file changed during the run.", file=sys.stderr)

        (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
        (out / "eval.json").write_text(json.dumps(ev, indent=2) + "\n")
        (out / "eval.txt").write_text(render(root, ev) + "\n")
        (out / "replay.json").write_text(json.dumps(replays, indent=2) + "\n")
        print(f"Results written to {_rel(out)}/", file=sys.stderr)
        if args.keep:
            print(f"Sandbox kept at {root}", file=sys.stderr)
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
