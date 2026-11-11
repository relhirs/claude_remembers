#!/usr/bin/env python3
"""
CI test suite for claude_remembers.

Verifies cross-platform compatibility:
1. All Python scripts compile
2. All dependencies import
3. SQLite database can be created with full schema
4. FTS5 full-text search works
5. Config file parses correctly

Does NOT require interactive input or a running Claude Code session.
"""

import importlib
import json
import os
import py_compile
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def test_python_compilation():
    """Compile every .py file in scripts/, hooks/, mcp/, tests/."""
    print("\n--- Python Compilation ---")
    dirs = ["scripts", "hooks", "mcp"]
    for d in dirs:
        folder = ROOT / d
        if not folder.exists():
            check(f"{d}/ exists", False, "directory not found")
            continue
        py_files = sorted(folder.glob("*.py"))
        check(f"{d}/ has Python files", len(py_files) > 0, f"found {len(py_files)}")
        for f in py_files:
            try:
                py_compile.compile(str(f), doraise=True)
                check(f"compile {d}/{f.name}", True)
            except py_compile.PyCompileError as e:
                check(f"compile {d}/{f.name}", False, str(e))


def test_dependencies():
    """Verify critical imports work."""
    print("\n--- Dependency Imports ---")
    modules = [
        ("yaml", "PyYAML"),
        ("sqlite3", "sqlite3 (stdlib)"),
        ("json", "json (stdlib)"),
        ("pathlib", "pathlib (stdlib)"),
        ("hashlib", "hashlib (stdlib)"),
    ]
    for mod, label in modules:
        try:
            importlib.import_module(mod)
            check(f"import {label}", True)
        except ImportError:
            check(f"import {label}", False, "not installed")

    # Optional but recommended
    for mod, label in [("sentence_transformers", "sentence-transformers"), ("numpy", "numpy")]:
        try:
            importlib.import_module(mod)
            check(f"import {label} (optional)", True)
        except ImportError:
            check(f"import {label} (optional)", True, "skipped (optional)")


def test_database_creation():
    """Create a fresh database with the full schema and verify all tables."""
    print("\n--- Database Creation ---")

    # Read DDL from remember-setup.py source to stay in sync
    setup_path = ROOT / "scripts" / "remember-setup.py"
    check("remember-setup.py exists", setup_path.exists())

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test-remember.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create all tables using the same DDL as remember-setup.py
        ddl_statements = [
            """CREATE TABLE IF NOT EXISTS sys_sessions (
                session_id TEXT PRIMARY KEY, project TEXT, started_at TEXT,
                ended_at TEXT, cwd TEXT, claude_version TEXT, model TEXT,
                source TEXT, message_count INTEGER DEFAULT 0, created_at TEXT,
                quality_score INTEGER DEFAULT NULL, quality_tags TEXT DEFAULT NULL,
                notes TEXT, tags TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS sys_ingest_log (
                file_path TEXT PRIMARY KEY, file_size INTEGER, file_type TEXT,
                records_imported INTEGER, ingested_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS project_registry (
                folder_name TEXT PRIMARY KEY, prefix TEXT UNIQUE, label TEXT,
                registered_at TEXT, summary TEXT, summary_updated_at TEXT,
                status TEXT DEFAULT 'active', health TEXT DEFAULT 'green'
            )""",
            """CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
                project TEXT, uuid TEXT UNIQUE, parent_uuid TEXT, type TEXT,
                subtype TEXT, role TEXT, content TEXT, model TEXT,
                timestamp TEXT, token_input INTEGER, token_output INTEGER,
                stop_reason TEXT, is_subagent INTEGER DEFAULT 0,
                source_file TEXT, raw_json TEXT, created_at TEXT, source TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS tool_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
                project TEXT, tool_use_id TEXT, content TEXT,
                source_file TEXT, created_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS remember_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT,
                key TEXT, value TEXT, source TEXT, source_session TEXT,
                confidence TEXT, created_at TEXT, updated_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS remember_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT,
                preference TEXT, source TEXT, created_at TEXT, updated_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT,
                decision_number INTEGER, session_id TEXT, description TEXT,
                rationale TEXT, created_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT,
                category TEXT, key TEXT, value TEXT, source TEXT,
                session_id TEXT, created_at TEXT, updated_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS transcript_embeddings (
                transcript_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL,
                model TEXT DEFAULT 'all-MiniLM-L6-v2', created_at TEXT
            )""",
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_transcripts_project ON transcripts(project)",
            "CREATE INDEX IF NOT EXISTS idx_transcripts_timestamp ON transcripts(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_transcripts_type ON transcripts(type)",
            "CREATE INDEX IF NOT EXISTS idx_transcripts_uuid ON transcripts(uuid)",
        ]

        fts = """CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
            content, content=transcripts, content_rowid=id
        )"""

        # Create tables
        for ddl in ddl_statements:
            try:
                cursor.execute(ddl)
            except Exception as e:
                check(f"DDL execution", False, str(e))
                conn.close()
                return

        check("10 tables created", True)

        # Create indexes
        for idx in indexes:
            cursor.execute(idx)
        check("5 indexes created", True)

        # Create FTS5
        try:
            cursor.execute(fts)
            check("FTS5 virtual table created", True)
        except Exception as e:
            check("FTS5 virtual table created", False, str(e))

        conn.commit()

        # Verify all tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cursor.fetchall()]
        expected = [
            "remember_facts", "remember_preferences", "decisions", "facts",
            "project_registry", "sys_ingest_log", "sys_sessions",
            "tool_results", "transcript_embeddings", "transcripts",
            "transcripts_fts",
        ]
        for t in expected:
            check(f"table {t} exists", t in tables)

        # Test FTS5 insert and search
        cursor.execute("""INSERT INTO transcripts
            (session_id, project, uuid, role, content, timestamp, source)
            VALUES ('test-session', 'test', 'uuid-001', 'user',
                    'How do I configure the database schema?',
                    '2026-01-01T00:00:00Z', 'test')""")
        cursor.execute("""INSERT INTO transcripts_fts(rowid, content)
            VALUES (last_insert_rowid(), 'How do I configure the database schema?')""")
        conn.commit()

        cursor.execute("SELECT content FROM transcripts_fts WHERE transcripts_fts MATCH 'database'")
        results = cursor.fetchall()
        check("FTS5 search works", len(results) == 1)

        # Test insert into project_registry
        cursor.execute("""INSERT INTO project_registry (folder_name, prefix, label, registered_at)
            VALUES ('test-project', 'tp', 'Test Project', '2026-01-01T00:00:00Z')""")
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM project_registry")
        check("project_registry insert works", cursor.fetchone()[0] == 1)

        conn.close()

        # Verify database file was created and has size
        check("database file created", db_path.exists())
        check("database has data", db_path.stat().st_size > 0)


def test_ingest_fixtures():
    """Run the real ingestion engine against verification/fixtures/*.jsonl
    in an isolated temp DB, to actually exercise dedup, malformed-line
    handling, and subagent detection (not just confirm files compile)."""
    print("\n--- Ingest Fixtures ---")

    fixtures_dir = ROOT / "verification" / "fixtures"
    check("verification/fixtures/ exists", fixtures_dir.exists())
    if not fixtures_dir.exists():
        return

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import ingest_jsonl
    except Exception as e:
        check("import ingest_jsonl", False, str(e))
        return

    import logging
    logger = logging.getLogger("ci_test_ingest")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, project TEXT,
                uuid TEXT UNIQUE, parent_uuid TEXT, type TEXT, subtype TEXT, role TEXT,
                content TEXT, model TEXT, timestamp TEXT, token_input INTEGER,
                token_output INTEGER, stop_reason TEXT, is_subagent INTEGER DEFAULT 0,
                source_file TEXT, raw_json TEXT, created_at TEXT, source TEXT);
            CREATE TABLE sys_sessions (
                session_id TEXT PRIMARY KEY, project TEXT, started_at TEXT, ended_at TEXT,
                cwd TEXT, claude_version TEXT, model TEXT, source TEXT,
                message_count INTEGER DEFAULT 0, created_at TEXT, notes TEXT, tags TEXT);
            CREATE TABLE tool_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, project TEXT,
                tool_use_id TEXT, content TEXT, source_file TEXT, created_at TEXT);
        """)

        # config=None here (no --semantic flag) means embed_message() short-circuits
        # on SEMANTIC_AVAILABLE/enabled checks - fine, this test is about parsing,
        # not embedding. Pass a disabled-semantic config explicitly to be sure.
        fake_config = {"semantic_search": {"enabled": False}}

        def run(fname, file_type="jsonl"):
            path = fixtures_dir / fname
            return ingest_jsonl.ingest_jsonl_file(
                conn, str(path), "oth", file_type, logger, config=fake_config
            )

        imported, skipped = run("valid_session.jsonl")
        check("valid_session.jsonl: imports storable lines", imported == 7,
              f"got imported={imported}")
        check("valid_session.jsonl: skips snapshot/progress lines", skipped == 3,
              f"got skipped={skipped}")

        imported, skipped = run("valid_session_with_tools.jsonl")
        check("valid_session_with_tools.jsonl: imports all lines", imported == 6,
              f"got imported={imported}")

        imported, skipped = run("duplicate_uuids.jsonl")
        check("duplicate_uuids.jsonl: dedups by uuid", imported == 2,
              f"got imported={imported} (expected 2 - duplicates must be ignored, not errored)")

        imported, skipped = run("malformed.jsonl")
        check("malformed.jsonl: imports only the valid line", imported == 1,
              f"got imported={imported}")
        check("malformed.jsonl: skips bad-JSON/missing-uuid/missing-type lines", skipped == 4,
              f"got skipped={skipped}")

        imported, skipped = run("empty.jsonl")
        check("empty.jsonl: no crash, nothing imported", imported == 0 and skipped == 0,
              f"got imported={imported} skipped={skipped}")

        imported, skipped = run("subagent_session.jsonl", file_type="subagent")
        check("subagent_session.jsonl: imports all lines", imported == 5,
              f"got imported={imported}")
        subagent_flag = conn.execute(
            "SELECT DISTINCT is_subagent FROM transcripts WHERE session_id = 'test-session-subagent'"
        ).fetchall()
        check("subagent_session.jsonl: rows flagged is_subagent=1",
              subagent_flag == [(1,)], f"got {subagent_flag}")

        conn.close()


def test_config_parsing():
    """Verify config.yaml.example parses correctly."""
    print("\n--- Config Parsing ---")
    config_example = ROOT / "config.yaml.example"
    check("config.yaml.example exists", config_example.exists())

    if config_example.exists():
        try:
            import yaml
            with open(config_example) as f:
                cfg = yaml.safe_load(f)
            check("config.yaml.example parses", cfg is not None)
            check("config has 'projects' key", "projects" in cfg)
            check("config has 'database' key", "database" in cfg)
        except Exception as e:
            check("config.yaml.example parses", False, str(e))


def test_user_prompt_submit_hook_payload():
    """Feed hooks/user-prompt-submit.py a real-shaped Claude Code
    UserPromptSubmit payload (session_id/prompt/cwd/hook_event_name) and
    verify it actually performs retrieval instead of silently no-op'ing.

    Regression test for the bug where the hook read input_data["user_prompt"]
    - a field Claude Code never sends - instead of "prompt", so prompt_text
    was always empty and retrieval silently never ran. A payload-shape
    mismatch like that doesn't raise; it just returns "{}" every time, which
    is why 95 passing tests never caught it. This test fails loudly on that
    same mismatch by asserting a seeded, keyword-matching prompt produces a
    non-empty "Relevant Memories" result.
    """
    print("\n--- user-prompt-submit.py Hook Payload ---")

    hook_src = ROOT / "hooks" / "user-prompt-submit.py"
    config_utils_src = ROOT / "scripts" / "config_utils.py"
    check("hooks/user-prompt-submit.py exists", hook_src.exists())
    check("scripts/config_utils.py exists", config_utils_src.exists())

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_root = Path(tmpdir)
        (fake_root / "hooks").mkdir()
        (fake_root / "scripts").mkdir()
        shutil.copy(hook_src, fake_root / "hooks" / "user-prompt-submit.py")
        shutil.copy(config_utils_src, fake_root / "scripts" / "config_utils.py")

        db_path = fake_root / "claude_remembers.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
            project TEXT, uuid TEXT UNIQUE, role TEXT, content TEXT,
            timestamp TEXT, source TEXT
        )""")
        conn.execute("""CREATE VIRTUAL TABLE transcripts_fts USING fts5(
            content, content=transcripts, content_rowid=id
        )""")
        # Several filler rows alongside the matching one - with a single-row
        # corpus, FTS5's bm25 rank stays too close to 0 (no idf signal) to
        # clear FTS_RANK_FLOOR, which would make this test fail for a reason
        # unrelated to the bug it's checking for.
        seed_contents = [
            "The frobnicator rollout needs a staging flag before launch",
            "Completely unrelated discussion about apples and oranges",
            "Another unrelated note about hiking trails in the mountains",
            "Random text about coffee brewing methods and equipment",
            "Notes on gardening tips for spring planting season",
        ]
        for i, content in enumerate(seed_contents):
            conn.execute("""INSERT INTO transcripts
                (session_id, project, uuid, role, content, timestamp, source)
                VALUES (?, 'oth', ?, 'assistant', ?, '2026-01-01T00:00:00Z', 'test')""",
                (f"seed-session-{i}", f"seed-uuid-{i}", content))
            conn.execute("""INSERT INTO transcripts_fts(rowid, content)
                VALUES (last_insert_rowid(), ?)""", (content,))
        conn.commit()
        conn.close()

        # Use as_posix() (forward slashes), not str(fake_root) - on Windows,
        # str() gives backslashes, which YAML's double-quoted scalar syntax
        # treats as escape characters ("\U" isn't a valid escape), breaking
        # yaml.safe_load() inside the hook. Forward slashes work fine for
        # file I/O on Windows and sidestep the escaping issue entirely.
        config_yaml = f"""
storage:
  mode: local
  root_path: "{fake_root.as_posix()}"
jsonl_project_mapping: {{}}
"""
        (fake_root / "config.yaml").write_text(config_yaml)

        payload = {
            "session_id": "test-session-id",
            "transcript_path": str(fake_root / "transcript.jsonl"),
            "cwd": str(fake_root),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "what's the status of the frobnicator staging rollout",
        }

        result = subprocess.run(
            [sys.executable, str(fake_root / "hooks" / "user-prompt-submit.py")],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
        )

        check("hook exits cleanly", result.returncode == 0, result.stderr[:200])

        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            output = None
        check("hook stdout is valid JSON", output is not None, result.stdout[:200])

        context = (
            output.get("hookSpecificOutput", {}).get("additionalContext", "")
            if output else ""
        )
        check(
            "hook used the 'prompt' field and found the seeded match",
            "frobnicator" in context,
            f"got additionalContext={context!r}",
        )

        log_path = fake_root / "logs" / "retrieval.log"
        check("retrieval.log was written", log_path.exists())
        if log_path.exists():
            check(
                "retrieval.log recorded a keyword_hit, not skipped_no_keywords",
                "keyword_hit" in log_path.read_text(),
                log_path.read_text()[:200],
            )


def test_metrics_layer():
    """Measurement layer building blocks: schema, tool/usage extraction,
    upsert ordering, and the rule that no command text is ever stored."""
    print("\n--- Metrics Layer ---")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import metrics_utils as mu
    except Exception as e:
        check("import metrics_utils", False, str(e))
        return

    conn = sqlite3.connect(":memory:")
    mu.ensure_metrics_schema(conn)
    mu.ensure_metrics_schema(conn)  # second call must be a no-op, not an error
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("retrieval_events", "tool_events", "api_usage", "metrics_snapshots"):
        check(f"metrics table {t} created (idempotent)", t in tables)

    secret = "--token sk-ant-FAKE-SECRET-VALUE"
    assistant = {"type": "assistant", "message": {
        "id": "msg_1", "model": "m",
        "usage": {"input_tokens": 5, "cache_creation_input_tokens": None,
                  "cache_read_input_tokens": 100, "output_tokens": 7},
        "content": [
            {"type": "tool_use", "id": "tu_bash", "name": "Bash",
             "input": {"command": f"cd /a/b && FOO=1 pytest -q {secret}"}},
            {"type": "tool_use", "id": "tu_read", "name": "Read",
             "input": {"file_path": "/a/b/c.py"}},
            {"type": "tool_use", "id": "tu_mem", "name": "mcp__remember-server__search_transcripts",
             "input": {"query": "private words"}},
        ]}}
    result = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "tu_bash", "is_error": True,
         "content": [{"type": "text", "text": "boom"}]},
        {"type": "tool_result", "tool_use_id": "tu_read", "content": "x" * 40},
    ]}}
    # The result is written BEFORE its tool call here, to prove either order works.
    mu.write_capture_metrics(conn, result, "s1", "oth", "2026-01-01T00:00:02Z")
    mu.write_capture_metrics(conn, assistant, "s1", "oth", "2026-01-01T00:00:01Z", transcript_id=9)
    rows = {r[0]: r for r in conn.execute(
        "SELECT tool_use_id, tool_name, target, is_memory_tool, ok, result_chars, ts, "
        "transcript_id FROM tool_events")}
    check("one row per tool call", len(rows) == 3, f"got {len(rows)}")
    check("Bash target is the executable only (cd/env prefix skipped)",
          rows["tu_bash"][2] == "pytest", f"got {rows['tu_bash'][2]!r}")
    check("failed tool result recorded", rows["tu_bash"][4] == 0 and rows["tu_bash"][5] == 4,
          f"got ok={rows['tu_bash'][4]} chars={rows['tu_bash'][5]}")
    check("Read target is the file path, success and size recorded",
          rows["tu_read"][2] == "/a/b/c.py" and rows["tu_read"][4] == 1 and rows["tu_read"][5] == 40)
    check("memory tool call flagged", rows["tu_mem"][3] == 1)
    check("tool-call fields win over the result stub",
          rows["tu_bash"][6] == "2026-01-01T00:00:01Z" and rows["tu_bash"][7] == 9)
    dump = " ".join(str(v) for r in conn.execute("SELECT * FROM tool_events") for v in r)
    check("no command text or secret is ever stored",
          secret not in dump and "pytest -q" not in dump and "FOO=1" not in dump
          and "private words" not in dump)

    # One API response is logged as several rows with the same usage line.
    repeat = json.loads(json.dumps(assistant))
    repeat["message"]["usage"]["output_tokens"] = 12
    mu.write_capture_metrics(conn, repeat, "s1", "oth", "2026-01-01T00:00:03Z")
    mu.write_capture_metrics(conn, assistant, "s1", "oth", "2026-01-01T00:00:04Z")  # smaller, later
    usage = conn.execute("SELECT COUNT(*), MAX(output_tokens), MAX(cache_read_tokens), "
                         "MAX(cache_creation_tokens) FROM api_usage").fetchone()
    check("api_usage dedupes by message id, keeps the largest, None counts as 0",
          usage == (1, 12, 100, 0), f"got {usage}")
    conn.close()

    check("machine-generated prompts are recognized",
          mu.is_system_generated_prompt('<agent-message from="a1"> [Subagent hand-back]')
          and mu.is_system_generated_prompt("  <command-name>/clear</command-name>"))
    check("a normal prompt is not flagged",
          not mu.is_system_generated_prompt("what did we decide about the database schema"))
    check("metrics.enabled defaults on, and can be turned off",
          mu.metrics_enabled({}) and mu.metrics_enabled(None)
          and not mu.metrics_enabled({"metrics": {"enabled": False}}))


def test_metrics_capture_path():
    """Run the real ingest engine on the tool-call fixture and confirm the
    metrics tables fill in (and stay empty when metrics are switched off)."""
    print("\n--- Metrics Capture Path ---")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import ingest_jsonl
    except Exception as e:
        check("import ingest_jsonl", False, str(e))
        return

    import logging
    logger = logging.getLogger("ci_test_metrics_capture")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    fixture = ROOT / "verification" / "fixtures" / "valid_session_with_tools.jsonl"

    def fresh_db(tmpdir):
        conn = sqlite3.connect(str(Path(tmpdir) / "test.db"))
        conn.executescript("""
            CREATE TABLE transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, project TEXT,
                uuid TEXT UNIQUE, parent_uuid TEXT, type TEXT, subtype TEXT, role TEXT,
                content TEXT, model TEXT, timestamp TEXT, token_input INTEGER,
                token_output INTEGER, stop_reason TEXT, is_subagent INTEGER DEFAULT 0,
                source_file TEXT, raw_json TEXT, created_at TEXT, source TEXT);
            CREATE TABLE sys_sessions (
                session_id TEXT PRIMARY KEY, project TEXT, started_at TEXT, ended_at TEXT,
                cwd TEXT, claude_version TEXT, model TEXT, source TEXT,
                message_count INTEGER DEFAULT 0, created_at TEXT, notes TEXT, tags TEXT);
        """)
        return conn

    with tempfile.TemporaryDirectory() as tmpdir:
        conn = fresh_db(tmpdir)
        cfg = {"semantic_search": {"enabled": False}}
        imported, _ = ingest_jsonl.ingest_jsonl_file(conn, str(fixture), "oth", "jsonl", logger, config=cfg)
        events = {r[0]: r for r in conn.execute(
            "SELECT tool_use_id, tool_name, target, ok FROM tool_events")}
        check("metrics tables are created on first capture", len(events) == 2, f"got {events}")
        check("fixture Read call captured with its file path and success",
              events.get("toolu_test_tools_001") == ("toolu_test_tools_001", "Read", "/home/testuser/config.yaml", 1),
              f"got {events.get('toolu_test_tools_001')}")
        check("fixture Bash call captured as the executable name only",
              events.get("toolu_test_tools_002", (0, 0, 0))[2] == "ls")
        usage = conn.execute("SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens) FROM api_usage").fetchone()
        check("one api_usage row per API response, null cache fields as 0",
              usage == (3, 3500, 330), f"got {usage}")
        ingest_jsonl.ingest_jsonl_file(conn, str(fixture), "oth", "jsonl", logger, config=cfg)
        check("re-ingesting the same file adds nothing",
              conn.execute("SELECT COUNT(*) FROM tool_events").fetchone()[0] == 2
              and conn.execute("SELECT COUNT(*) FROM api_usage").fetchone()[0] == 3)
        conn.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        conn = fresh_db(tmpdir)
        cfg = {"semantic_search": {"enabled": False}, "metrics": {"enabled": False}}
        imported, _ = ingest_jsonl.ingest_jsonl_file(conn, str(fixture), "oth", "jsonl", logger, config=cfg)
        has_table = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'tool_events'").fetchone()[0]
        check("metrics.enabled: false records nothing, but messages are still stored",
              has_table == 0 and imported == 6, f"tables={has_table} imported={imported}")
        conn.close()


def _make_hook_sandbox(tmpdir, with_metrics=True, config_extra=""):
    """A throwaway copy of the hook with a seeded database, like the payload
    test above, but with explicit timestamps and optional metrics module."""
    root = Path(tmpdir)
    (root / "hooks").mkdir()
    (root / "scripts").mkdir()
    shutil.copy(ROOT / "hooks" / "user-prompt-submit.py", root / "hooks" / "user-prompt-submit.py")
    shutil.copy(ROOT / "scripts" / "config_utils.py", root / "scripts" / "config_utils.py")
    if with_metrics:
        shutil.copy(ROOT / "scripts" / "metrics_utils.py", root / "scripts" / "metrics_utils.py")

    db_path = root / "claude_remembers.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, project TEXT,
        uuid TEXT UNIQUE, role TEXT, content TEXT, timestamp TEXT, source TEXT)""")
    conn.execute("""CREATE VIRTUAL TABLE transcripts_fts USING fts5(
        content, content=transcripts, content_rowid=id)""")
    seeds = [
        ("2026-01-01T00:00:00Z", "The frobnicator staging rollout needs a feature flag before launch"),
        ("2026-03-01T00:00:00Z", "The frobnicator staging rollout checklist was updated after review"),
    ] + [("2026-01-02T00:00:00Z", f"Unrelated filler note number {i} about hiking, coffee and gardens")
         for i in range(10)]
    for i, (ts, content) in enumerate(seeds):
        conn.execute("""INSERT INTO transcripts (session_id, project, uuid, role, content, timestamp, source)
            VALUES (?, 'oth', ?, 'assistant', ?, ?, 'test')""", (f"seed-{i}", f"seed-uuid-{i}", content, ts))
        conn.execute("INSERT INTO transcripts_fts(rowid, content) VALUES (last_insert_rowid(), ?)", (content,))
    conn.commit()
    conn.close()

    (root / "config.yaml").write_text(
        f'storage:\n  mode: local\n  root_path: "{root.as_posix()}"\njsonl_project_mapping: {{}}\n{config_extra}')
    return root, db_path


def _run_hook(root, prompt, session_id="telemetry-session"):
    payload = {"session_id": session_id, "cwd": str(root), "hook_event_name": "UserPromptSubmit",
               "prompt": prompt}
    result = subprocess.run([sys.executable, str(root / "hooks" / "user-prompt-submit.py")],
                            input=json.dumps(payload), capture_output=True, text=True, timeout=30)
    return result


def test_hook_telemetry():
    """The hook records exactly one retrieval_events row per prompt, on every
    path, without changing what it injects or breaking when metrics are off
    or the metrics module is missing."""
    print("\n--- Hook Telemetry ---")
    prompt = "what is the status of the frobnicator staging rollout checklist"

    with tempfile.TemporaryDirectory() as tmpdir:
        root, db_path = _make_hook_sandbox(tmpdir)
        hit = _run_hook(root, prompt)
        short = _run_hook(root, "ok thanks")
        stop = _run_hook(root, "what is the and for with this that")
        system = _run_hook(root, '<agent-message from="a1"> [Subagent hand-back] frobnicator staging rollout report')
        check("hook still injects memories with the metrics module present",
              "frobnicator" in json.loads(hit.stdout).get("hookSpecificOutput", {}).get("additionalContext", ""),
              hit.stdout[:200])
        check("short and no-keyword prompts still return {}",
              short.stdout.strip() == "{}" and stop.stdout.strip() == "{}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        events = conn.execute("SELECT * FROM retrieval_events ORDER BY id").fetchall()
        check("exactly one event per prompt, table created on first use", len(events) == 4,
              f"got {[e['outcome'] for e in events]}")
        outcomes = [e["outcome"] for e in events]
        check("outcomes: hit / skipped_short / skipped_no_keywords / hit",
              outcomes == ["keyword_hit", "skipped_short", "skipped_no_keywords", "keyword_hit"], outcomes)
        e = events[0]
        ids = json.loads(e["result_ids"])
        check("hit event records result ids, ages, project and size",
              e["n_results"] >= 1 and len(ids) == e["n_results"] and all(isinstance(i, int) for i in ids)
              and len(json.loads(e["result_ages_days"])) == e["n_results"]
              and e["project"] == "oth" and e["injected_chars"] > 0
              and e["injected_tokens_est"] == e["injected_chars"] // 4
              and e["mode"] == "fallback_v2" and e["source"] == "live",
              dict(e))
        check("hit event records match strength, coverage and timing",
              e["top_rank"] is not None and e["top_rank"] < 0 and 0 < e["top_coverage"] <= 1
              and e["keyword_ms"] is not None and e["latency_ms"] >= e["keyword_ms"], dict(e))
        check("prompt kind distinguishes typed from machine-generated prompts",
              events[0]["prompt_kind"] == "user" and events[3]["prompt_kind"] == "system_generated")
        check("no prompt text is stored in the event row",
              "frobnicator" not in " ".join(str(v) for v in dict(e).values()))
        conn.close()
        check("retrieval.log line format is unchanged",
              "keyword_hit" in (root / "logs" / "retrieval.log").read_text())

    with tempfile.TemporaryDirectory() as tmpdir:
        root, db_path = _make_hook_sandbox(tmpdir, config_extra="metrics:\n  enabled: false\n")
        res = _run_hook(root, prompt)
        conn = sqlite3.connect(str(db_path))
        has_table = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'retrieval_events'").fetchone()[0]
        conn.close()
        check("metrics.enabled: false records nothing but retrieval still works",
              has_table == 0 and "frobnicator" in res.stdout, f"table={has_table}")

    with tempfile.TemporaryDirectory() as tmpdir:
        root, db_path = _make_hook_sandbox(tmpdir, with_metrics=False)
        res = _run_hook(root, prompt)
        check("hook works with the metrics module missing entirely",
              res.returncode == 0 and "frobnicator" in res.stdout, res.stderr[:200])


def test_hook_internals():
    """In-process checks of the hook's helpers: replay support (as_of),
    result ids, and the semantic-fallback telemetry path with a stub model."""
    print("\n--- Hook Internals ---")
    import contextlib
    import importlib.util
    import io

    with tempfile.TemporaryDirectory() as tmpdir:
        root, db_path = _make_hook_sandbox(tmpdir)
        spec = importlib.util.spec_from_file_location("ups_hook_under_test", root / "hooks" / "user-prompt-submit.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        hook.fuzzy_correct = None  # keep keywords predictable

        check("extract_keywords drops stop-words and short words",
              hook.extract_keywords("What is the frobnicator rollout?") == ["frobnicator", "rollout"])

        conn = sqlite3.connect(str(db_path))
        query = '"frobnicator" OR "staging" OR "rollout"'
        now_rows = hook._keyword_search(conn, query, "oth")
        old_rows = hook._keyword_search(conn, query, "oth", as_of="2026-02-01T00:00:00Z")
        check("search rows carry the transcript id as a 5th field",
              len(now_rows) >= 2 and all(len(r) == 5 and isinstance(r[4], int) for r in now_rows))
        check("as_of hides anything stored after it (replay support)",
              len(old_rows) == 1 and old_rows[0][1].startswith("2026-01-01"),
              f"got {[(r[1], r[4]) for r in old_rows]}")
        check("without as_of both matching messages are found", len(now_rows) == 2,
              f"got {len(now_rows)}")
        check("format_memories renders 5-field rows",
              hook.format_memories(now_rows).startswith("## Relevant Memories"))
        check("coverage is the share of keywords present",
              hook._coverage(["frobnicator", "zzz"], "the frobnicator is here") == 0.5)
        conn.close()

        def run_with_stub(results):
            hook.semantic_search = lambda conn, config, prompt, project=None, limit=5: results
            ev = {}
            buf = io.StringIO()
            raw = json.dumps({"session_id": "stub-session", "prompt": "zzqplex wibblefrotz quuxborg nothing matches"})
            with contextlib.redirect_stdout(buf):
                hook._run(raw, ev)
                hook._finish_event(ev, 0.0)
            return ev, buf.getvalue()

        ev, out = run_with_stub([{"similarity": 0.8, "project": "oth", "timestamp": "2026-01-01T00:00:00Z",
                                  "content": "stubbed memory", "id": 3}])
        check("semantic fallback hit is injected and recorded with its id and similarity",
              ev.get("outcome") == "semantic_hit" and "stubbed memory" in out, f"{ev} {out[:100]}")
        ev, out = run_with_stub([{"similarity": 0.2, "project": "oth", "timestamp": "2026-01-01T00:00:00Z",
                                  "content": "too weak", "id": 4}])
        check("a below-cutoff semantic result injects nothing but records the similarity",
              out.strip() == "{}" and ev.get("outcome") == "no_match", f"{ev} {out[:100]}")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT outcome, top_similarity, result_ids FROM retrieval_events "
                            "WHERE session_id = 'stub-session' ORDER BY id").fetchall()
        conn.close()
        check("both semantic paths were written to retrieval_events",
              rows == [("semantic_hit", 0.8, "[3]"), ("no_match", 0.2, None)], f"got {rows}")

        ev, out = run_with_stub([
            {"similarity": 0.9, "project": "oth", "timestamp": "2026-01-01T00:00:00Z",
             "content": "said earlier this session", "id": 5, "session_id": "stub-session"},
            {"similarity": 0.8, "project": "oth", "timestamp": "2026-01-01T00:00:00Z",
             "content": "<bash-input>ls</bash-input>", "id": 6, "session_id": "other"},
            {"similarity": 0.7, "project": "oth", "timestamp": "2026-01-01T00:00:00Z",
             "content": "a real older memory", "id": 7, "session_id": "other"}])
        check("semantic results from the current session or machine-generated rows are skipped",
              "a real older memory" in out and "said earlier this session" not in out
              and "bash-input" not in out, out[:200])


def test_profile_has_no_repeats():
    """CLAUDE.md is loaded by Claude Code itself, so memory must never inject a
    second copy: legacy imported rows are skipped, repeated rows collapse, and
    the setup script no longer creates them."""
    print("\n--- Profile Has No Repeats ---")
    sys.path.insert(0, str(ROOT / "scripts"))
    import profile_utils

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE remember_facts (id INTEGER PRIMARY KEY, category TEXT, key TEXT,
            value TEXT, confidence TEXT, project TEXT);
        CREATE TABLE remember_preferences (id INTEGER PRIMARY KEY, category TEXT,
            preference TEXT, project TEXT);
        INSERT INTO remember_facts (category, key, value, confidence, project)
            VALUES ('identity', 'role', 'Student', 'confirmed', NULL);
    """)
    rows = [
        ("imported-claude-md", "This machine has a cross-project memory system", None),
        ("imported-claude-md", "This machine has a cross-project memory system", None),
        ("working-style", "Prefers short answers", None),
        ("working-style", "Prefers short answers", None),      # exact repeat
        ("working-style", "Prefers short answers", "proj1"),   # same text, different scope: kept
        ("working-style", "Likes examples", None),
    ]
    conn.executemany("INSERT INTO remember_preferences (category, preference, project) VALUES (?, ?, ?)", rows)

    facts, prefs = profile_utils.fetch_merged_profile(conn, "proj1")
    texts = [(p["preference"], p["project"]) for p in prefs]
    check("legacy imported CLAUDE.md rows are not injected",
          not any("cross-project memory" in t for t, _ in texts), f"got {texts}")
    check("exact repeats collapse to one, a different scope is kept",
          sorted(texts, key=str) == sorted([("Prefers short answers", None),
                                            ("Prefers short answers", "proj1"),
                                            ("Likes examples", None)], key=str), f"got {texts}")
    markdown = profile_utils.format_profile_markdown(facts, prefs, heading="## Profile")
    lines = [l for l in markdown.splitlines() if l.startswith("- ")]
    check("rendered profile has no duplicate lines", len(lines) == len(set(lines)), markdown)
    check("real facts and preferences still appear",
          "role: Student" in markdown and "Likes examples" in markdown)

    setup_src = (ROOT / "scripts" / "remember-setup.py").read_text(encoding="utf-8")
    check("setup no longer copies ~/.claude/CLAUDE.md into the database",
          "INSERT INTO remember_preferences" not in setup_src and "claude_md_content" not in setup_src)


def test_retrieval_quality_fixes():
    """Keyword repeats, result ordering, and which rows are allowed back as
    memories: the three fixes behind the 'best match only covered 29% of the
    keywords' finding."""
    print("\n--- Retrieval Quality Fixes ---")
    import importlib.util

    with tempfile.TemporaryDirectory() as tmpdir:
        root, db_path = _make_hook_sandbox(tmpdir)
        spec = importlib.util.spec_from_file_location("ups_hook_quality", root / "hooks" / "user-prompt-submit.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        hook.fuzzy_correct = None

        # --- repeated keywords must not use up the 8 slots ---
        check("a repeated word is searched once",
              hook.extract_keywords("token context token lever context") == ["token", "context", "lever"])
        many = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        distinct = hook.extract_keywords(f"alpha alpha bravo bravo {many}")
        check("repeats don't consume the 8-keyword cap", distinct == many.split()[:8], f"got {distinct}")
        hook.fuzzy_correct = lambda kws, path: (["worth" if k.startswith("w") else k for k in kws], {})
        check("typo correction mapping two words to one is collapsed again",
              hook.build_search_keywords("wlorpt wibble frobnicator", str(db_path)) == ["worth", "frobnicator"])
        hook.fuzzy_correct = None

        # --- ordering direction, current-session and machine-row exclusion ---
        conn = sqlite3.connect(":memory:")
        conn.execute("""CREATE TABLE transcripts (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
            project TEXT, uuid TEXT UNIQUE, role TEXT, content TEXT, timestamp TEXT, source TEXT)""")
        conn.execute("CREATE VIRTUAL TABLE transcripts_fts USING fts5(content, content=transcripts, content_rowid=id)")
        # Equal length and equal relevance; only the dates differ. Distinct
        # openings, because the hook collapses results whose first 50
        # characters match.
        text = "{} the frobnicator staging rollout needs a feature flag before launch"
        rows = [("old-session", "2025-01-01T00:00:00Z", text.format("alpha")),
                ("new-session", "2026-03-01T00:00:00Z", text.format("bravo")),
                ("current-session", "2026-03-02T00:00:00Z", text.format("delta")),
                ("bot-session", "2026-03-03T00:00:00Z", "<bash-input>frobnicator staging rollout flag</bash-input>")]
        rows += [(f"filler-{i}", "2026-01-02T00:00:00Z", f"unrelated filler note {i} about hiking and coffee") for i in range(24)]
        for i, (sess, ts, content) in enumerate(rows):
            conn.execute("""INSERT INTO transcripts (session_id, project, uuid, role, content, timestamp, source)
                VALUES (?, 'oth', ?, 'assistant', ?, ?, 'test')""", (sess, f"u{i}", content, ts))
            conn.execute("INSERT INTO transcripts_fts(rowid, content) VALUES (last_insert_rowid(), ?)", (content,))
        query = '"frobnicator" OR "staging" OR "rollout"'

        found = hook._keyword_search(conn, query, "oth")
        sessions = [r[4] for r in found]
        order = [conn.execute("SELECT session_id FROM transcripts WHERE id = ?", (i,)).fetchone()[0] for i in sessions]
        check("with identical text, the newer memory outranks the older one",
              order.index("new-session") < order.index("old-session"), f"order was {order}")
        check("machine-generated rows are never returned as memories",
              "bot-session" not in order, f"order was {order}")
        excluded = hook._keyword_search(conn, query, "oth", exclude_session="current-session")
        excluded_sessions = [conn.execute("SELECT session_id FROM transcripts WHERE id = ?", (r[4],)).fetchone()[0] for r in excluded]
        check("the current session is excluded when asked",
              "current-session" in order and "current-session" not in excluded_sessions
              and "new-session" in excluded_sessions, f"{order} vs {excluded_sessions}")
        replay = hook._keyword_search(conn, query, "oth", as_of="2026-02-01T00:00:00Z", exclude_session="current-session")
        check("replay (as_of) combines with the filters",
              [r[1][:4] for r in replay] == ["2025"], f"got {[r[1] for r in replay]}")
        conn.close()

        check("config_utils and the metrics layer share one machine-generated definition",
              all(hook.is_machine_generated(p + " x") for p in hook.MACHINE_GENERATED_PREFIXES)
              and not hook.is_machine_generated("what did we decide"))


def test_session_start_is_slim():
    """The session-start hook reports how many sessions lack notes as one
    accurate line, not a capped list of unusable ID fragments."""
    print("\n--- Session Start Is Slim ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "hooks").mkdir()
        (root / "scripts").mkdir()
        shutil.copy(ROOT / "hooks" / "session-start.py", root / "hooks" / "session-start.py")
        for name in ("config_utils.py", "profile_utils.py"):
            shutil.copy(ROOT / "scripts" / name, root / "scripts" / name)

        conn = sqlite3.connect(str(root / "claude_remembers.db"))
        conn.executescript("""
            CREATE TABLE sys_sessions (session_id TEXT PRIMARY KEY, project TEXT, started_at TEXT,
                ended_at TEXT, notes TEXT, source TEXT, message_count INTEGER);
            CREATE TABLE project_registry (prefix TEXT, summary TEXT);
            CREATE TABLE remember_facts (category TEXT, key TEXT, value TEXT, confidence TEXT, project TEXT);
            CREATE TABLE remember_preferences (category TEXT, preference TEXT, project TEXT);
            CREATE TABLE transcripts (session_id TEXT, content TEXT, timestamp TEXT, role TEXT);
        """)
        for i in range(30):   # more than the old LIMIT 20, to prove the count is real
            conn.execute("INSERT INTO sys_sessions (session_id, project, started_at, notes) VALUES (?, 'oth', ?, '')",
                         (f"aaaaaaaaaaaa-{i:04d}-session", f"2026-10-{i % 28 + 1:02d}T00:00:00Z"))
        conn.execute("INSERT INTO sys_sessions (session_id, project, started_at, notes) "
                     "VALUES ('with-notes', 'oth', '2026-11-01T00:00:00Z', 'Topic: something useful')")
        conn.commit()
        conn.close()
        (root / "config.yaml").write_text(
            f'storage:\n  mode: local\n  root_path: "{root.as_posix()}"\njsonl_project_mapping: {{}}\n')

        result = subprocess.run([sys.executable, str(root / "hooks" / "session-start.py")], input="{}",
                                capture_output=True, text=True, timeout=60, env={**os.environ, "CWD": str(root)})
        check("session-start hook exits cleanly with valid JSON", result.returncode == 0, result.stderr[:200])
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        check("it reports the real number of sessions without notes (not capped at 20)",
              "30 past session(s) have no notes yet" in context, context[:300])
        check("it no longer lists session ID fragments",
              "aaaaaaaaaaaa" not in context and "Consider writing notes" not in context)
        check("the rest of the injection still works", "Session Start Checklist" in context
              and "Topic: something useful" in context)


def test_file_structure():
    """Verify expected directories and key files exist."""
    print("\n--- File Structure ---")
    expected_dirs = ["scripts", "hooks", "mcp", "verification"]
    for d in expected_dirs:
        check(f"{d}/ exists", (ROOT / d).exists())

    expected_files = [
        "README.md", "GO_DEEPER.md",
        "requirements.txt", "config.yaml.example",
        "mcp/server.py", "hooks/session-start.py",
        "hooks/session-end.py", "hooks/stop.py",
        "hooks/user-prompt-submit.py",
        "scripts/metrics_utils.py", "scripts/backfill_metrics.py",
        "scripts/remember_baseline.py",
    ]
    for f in expected_files:
        check(f"{f} exists", (ROOT / f).exists())


if __name__ == "__main__":
    print("=" * 60)
    print("  claude_remembers CI Test Suite")
    print(f"  Python {sys.version}")
    print(f"  Platform: {sys.platform}")
    print("=" * 60)

    test_file_structure()
    test_python_compilation()
    test_dependencies()
    test_database_creation()
    test_ingest_fixtures()
    test_config_parsing()
    test_user_prompt_submit_hook_payload()
    test_metrics_layer()
    test_metrics_capture_path()
    test_hook_telemetry()
    test_hook_internals()
    test_profile_has_no_repeats()
    test_retrieval_quality_fixes()
    test_session_start_is_slim()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
