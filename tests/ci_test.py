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
              and e["mode"] == "on_miss_v3" and e["source"] == "live",
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
            hook.semantic_search = lambda conn, config, prompt, project=None, limit=5, query_vec=None: results
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

        # --- repeated keywords must not use up the 8 slots ---
        check("a repeated word is searched once",
              hook.extract_keywords("token context token lever context") == ["token", "context", "lever"])
        many = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        distinct = hook.extract_keywords(f"alpha alpha bravo bravo {many}")
        check("repeats don't consume the 8-keyword cap", distinct == many.split()[:8], f"got {distinct}")
        check("there is no typo correction: unknown words are searched as typed",
              hook.build_search_keywords("kubernetes frobnicator", str(db_path)) == ["kubernetes", "frobnicator"]
              and not hasattr(hook, "fuzzy_correct"))

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


def test_live_metrics_report():
    """remember_metrics.py: windowing, project filter, carry cost, read-only."""
    print("\n--- Live Metrics Report ---")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import metrics_utils as mu
        import remember_metrics as rm
    except Exception as e:
        check("import remember_metrics", False, str(e))
        return
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "m.db")
        conn = sqlite3.connect(db)
        conn.executescript("CREATE TABLE transcripts (id INTEGER PRIMARY KEY, content TEXT, raw_json TEXT);"
                           "CREATE TABLE transcript_embeddings (transcript_id INTEGER);"
                           "CREATE TABLE sys_sessions (session_id TEXT, notes TEXT);")
        mu.ensure_metrics_schema(conn)
        ev = ("INSERT INTO retrieval_events (ts, source, session_id, project, outcome, n_results, "
              "top_coverage, latency_ms, result_ages_days, same_project_n, injected_tokens_est) "
              "VALUES (?, 'live', 's1', ?, ?, ?, 0.4, 10, '[2, 100]', 1, 100)")
        conn.execute(ev, ("2026-11-10T10:00:00Z", "fin", "keyword_hit", 2))
        conn.execute(ev, ("2026-11-10T10:05:00Z", "oth", "keyword_hit", 2))
        conn.execute(ev, ("2026-11-10T10:06:00Z", "fin", "no_results", 0))
        conn.execute("UPDATE retrieval_events SET injected_tokens_est = 0 WHERE n_results = 0")
        conn.execute(ev, ("2020-01-01T00:00:00Z", "fin", "keyword_hit", 2))  # outside window
        for i, ts in enumerate(("2026-11-10T10:01:00Z", "2026-11-10T10:07:00Z", "2026-11-10T10:08:00Z")):
            conn.execute("INSERT INTO api_usage (message_id, session_id, ts, input_tokens) "
                         "VALUES (?, 's1', ?, 1000)", (f"m{i}", ts))
        conn.commit()
        conn.close()

        ro = rm._connect_readonly(db)
        r = rm.retrieval_facts(ro, "2026-11-01T00:00:00Z", None)
        check("window excludes old events", r["events"] == 3)
        check("injection rate counts only events with results", r["injection_rate"] == round(2 / 3, 3))
        check("median age read from the JSON list", r["median_age_days"] == 51.0)
        check("project filter", rm.retrieval_facts(ro, "2026-11-01T00:00:00Z", "oth")["events"] == 1)
        t = rm.token_cost_facts(ro, "2026-11-01T00:00:00Z", None, r)
        # 100 tokens at 10:00 -> 3 replies follow; 100 at 10:05 -> 2 replies follow
        check("carry cost = tokens x later replies", t["carry_tokens_reread"] == 100 * 3 + 100 * 2)
        try:
            ro.execute("INSERT INTO tool_events (tool_use_id) VALUES ('x')")
            check("report connection is read-only", False)
        except sqlite3.OperationalError:
            check("report connection is read-only", True)
        ro.close()
        check("empty window renders a message, not a crash",
              "No prompts" in rm.render_text({"window_days": 1, "project": None,
                                              "retrieval": {"events": 0}, "tokens": {},
                                              "search_anyway": {}, "pipeline": {}}))


def test_eval_scoring():
    """remember_eval.py: session ranking, RRF merge, leak guard, metric math."""
    print("\n--- Retrieval Eval ---")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import remember_eval as ev
    except Exception as e:
        check("import remember_eval", False, str(e))
        return
    check("rows collapse to distinct sessions in order",
          ev._rank_sessions([(1, "a"), (2, "b"), (3, "a"), (4, None)]) == ["a", "b"])
    fused = ev._rrf([(1, "a"), (2, "b")], [(2, "b"), (3, "c")])
    check("RRF puts the row both lists agree on first", fused[0] == (2, "b"))
    kept = ev._clean("Where is the key", [{"content": "so where is the key stored"},
                                          {"content": "in zshrc"}])
    check("leak guard drops rows that quote the question", [r["content"] for r in kept] == ["in zshrc"])

    questions = [{"id": "q1", "kind": "answerable", "expect": ["s1"]},
                 {"id": "q2", "kind": "answerable", "expect": ["s2", "s9"]},
                 {"id": "n1", "kind": "no_answer", "expect": []}]
    blank = {"injected": 0, "sessions": [], "ms": 1.0}
    per = {"q1": {m: {"sessions": ["x", "s1"], "injected": 2, "ms": 1.0} for m in ev.METHODS},
           "q2": {m: {"sessions": ["x", "y", "z", "w", "v", "s9"], "injected": 6, "ms": 1.0}
                  for m in ev.METHODS},
           "n1": {m: ({"sessions": ["x"], "injected": 1, "ms": 1.0} if m == "keyword" else dict(blank))
                  for m in ev.METHODS}}
    m = ev.score(per, questions)
    check("recall@1 / @5 count first-rank and top-5 hits", m["keyword"]["recall_at_1"] == 0.0
          and m["keyword"]["recall_at_5"] == 0.5)
    check("MRR averages 1/rank (a hit at rank 6 scores as found, not as top-5)",
          m["keyword"]["mrr"] == round((0.5 + 1 / 6) / 2, 3))
    check("missed question is listed", m["keyword"]["missed_ids"] == ["q2"])
    check("false injection counts only no-answer questions that injected",
          m["keyword"]["false_injection_rate"] == 1.0 and m["meaning"]["false_injection_rate"] == 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "g.yaml")
        with open(path, "w") as f:
            f.write("exclude_sessions: [abc]\nquestions:\n  - {id: q1, kind: answerable, question: hi, expect: [s1]}\n")
        ex, qs = ev.load_golden(path)
        check("golden set loads with its excluded sessions", ex == ["abc"] and len(qs) == 1)


def test_weak_match_gate():
    """Keyword results are dropped when even the best one is weak, and the bar
    scales with database size so a small memory is not silenced."""
    print("\n--- Weak-Match Gate ---")
    import importlib.util, math
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_hook_sandbox(tmpdir)
        spec = importlib.util.spec_from_file_location("ups_hook_gate", root / "hooks" / "user-prompt-submit.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)

        def build(n_filler):
            conn = sqlite3.connect(":memory:")
            conn.execute("""CREATE TABLE transcripts (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
                project TEXT, uuid TEXT UNIQUE, role TEXT, content TEXT, timestamp TEXT, source TEXT)""")
            conn.execute("CREATE VIRTUAL TABLE transcripts_fts USING fts5(content, content=transcripts, content_rowid=id)")
            rows = [("target", "2026-03-01T00:00:00Z", "decided the quokka migration needs a zebracode rollback plan")]
            rows += [(f"f{i}", "2026-03-01T00:00:00Z",
                      f"note {i} about the common rollout plan{' with a marmot' if i % 20 == 0 else ''} and routine meeting")
                     for i in range(n_filler)]
            for i, (sid, ts, content) in enumerate(rows):
                conn.execute("INSERT INTO transcripts (session_id, project, uuid, role, content, timestamp) "
                             "VALUES (?, 'oth', ?, 'user', ?, ?)", (sid, f"u{i}", content, ts))
            conn.execute("INSERT INTO transcripts_fts(transcripts_fts) VALUES('rebuild')")
            conn.commit()
            return conn

        conn = build(300)
        strong = hook._keyword_search(conn, '"quokka" OR "zebracode"', "oth")
        check("a strong match (two rare words) is injected", len(strong) == 1)
        info = {}
        weak = hook._keyword_search(conn, '"marmot"', "oth", info=info)
        check("only a moderately common word matching: nothing injected, flagged weak",
              weak == [] and info.get("weak") is True)
        ungated = hook._keyword_search(conn, '"marmot"', "oth", gated=False)
        check("gated=False still returns the weak rows (for the eval)", len(ungated) >= 1)

        floors = {n: hook._confidence_floor(build(n)) for n in (30, 300)}
        check("the bar rises with database size", floors[300] < floors[30] < 0)
        check("the bar is 1.05 x ln(messages)",
              abs(floors[300] - (-hook.MIN_STRENGTH_PER_LOG_DOC * math.log(301))) < 1e-9)
        small = build(20)
        check("a small database is not silenced by the bar",
              len(hook._keyword_search(small, '"quokka" OR "zebracode"', "oth")) == 1)
        check("with no docsize table there is no gate (fail open)",
              hook._confidence_floor(sqlite3.connect(":memory:")) is None)


def test_semantic_modes():
    """Phase 5b: the on_miss gate, meaning-search row handling, the merge, mode
    parsing, and how the hook behaves in each mode."""
    print("\n--- Semantic Modes ---")
    import contextlib
    import importlib.util
    import io

    with tempfile.TemporaryDirectory() as tmpdir:
        root, db_path = _make_hook_sandbox(tmpdir)
        spec = importlib.util.spec_from_file_location("ups_hook_modes", root / "hooks" / "user-prompt-submit.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)

        # --- mode parsing ---
        check("a config with no mode setting means on_miss", hook.semantic_mode({}) == "on_miss"
              and hook.semantic_mode({"semantic_search": {"enabled": True}}) == "on_miss")
        check("mode: always is recognised, case and spaces ignored",
              hook.semantic_mode({"semantic_search": {"mode": " Always "}}) == "always")
        check("an unrecognised mode falls back to on_miss",
              hook.semantic_mode({"semantic_search": {"mode": "sometimes"}}) == "on_miss")
        check("enabled: false switches meaning search off in every mode",
              hook.semantic_mode({"semantic_search": {"enabled": False, "mode": "always"}}) == "off")

        # --- the four gate signals ---
        conn = sqlite3.connect(str(db_path))
        strong = [("oth", "2026-01-01", "frobnicator staging rollout flag", -30.0, 1)]
        kws = ["frobnicator", "staging"]
        check("no keyword result: gate fires", hook.should_use_semantic("some prompt here", kws, [], {}, conn) == (True, "no_keyword_result"))
        check("weak keyword result: gate fires (score strength)",
              hook.should_use_semantic("some prompt here", kws, [], {"weak": True}, conn) == (True, "weak_keyword"))
        check("a strong, well-covered keyword result: gate stays shut",
              hook.should_use_semantic("tell me about the rollout", kws, strong, {}, conn) == (False, "keyword_confident"))
        six = ["alpha", "bravo", "charlie", "delta", "echo", "frobnicator"]
        check("matching 1 of 6 keywords is weak coverage: gate fires",
              hook.should_use_semantic("some prompt here", six, strong, {}, conn) == (True, "low_coverage"))
        check("a question about the past fires the gate even with a good keyword hit",
              hook.should_use_semantic("what did we decide about the rollout", kws, strong, {}, conn) == (True, "memory_phrasing"))
        check("single words like 'earlier' or 'before' are not memory phrasing",
              not hook.MEMORY_PHRASING.search("as I said earlier, fix this before launch"))
        conn.close()

        # --- meaning-search rows ---
        def res(sim, project, content, rid, session="other"):
            return {"similarity": sim, "project": project, "timestamp": "2026-01-01T00:00:00Z",
                    "content": content, "id": rid, "session_id": session}
        rows = hook.semantic_rows([res(0.50, "b2", "plain other-project memory", 1),
                                   res(0.47, "oth", "current project memory", 2),
                                   res(0.34, "oth", "under the cutoff", 3),
                                   res(0.60, "oth", "from this very session", 4, session="now")], "oth", "now")
        check("current-project boost reorders results that cleared the cutoff",
              [r[4] for r in rows] == [2, 1], f"got {[r[4] for r in rows]}")
        check("the cutoff uses the raw similarity: the boost can't rescue a weak result",
              all(r[4] != 3 for r in rows))
        check("the current session is dropped", all(r[4] != 4 for r in rows))

        # --- merging ---
        a = ("oth", "2026-01-01", "only keyword found this one", -9.0, 10)
        b = ("oth", "2026-01-01", "both searches found this one", -8.0, 11)
        c = ("oth", "2026-01-01", "only meaning found this one", None, 12)
        merged = hook.merge_results([a, b], [b, c])
        check("a memory both searches agree on rises to the top", merged[0][4] == 11)
        check("merging keeps every distinct memory, capped at the injection limit",
              {r[4] for r in merged} == {10, 11, 12} and len(hook.merge_results([a] * 9, [])) == 1)

        # --- the hook in each mode (stubbed model and helper) ---
        calls = {"semantic": [], "started": 0}

        def stub_semantic(conn, config, prompt, project=None, limit=5, query_vec=None):
            calls["semantic"].append(query_vec)
            return [res(0.8, "oth", "a meaning-search memory", 99)]

        def run(mode, prompt, helper_vec="down", gate=None):
            calls["semantic"].clear()
            calls["started"] = 0
            hook.semantic_search = stub_semantic
            hook.semantic_mode = lambda config: mode
            hook._helper_vector = lambda r, t: None if helper_vec == "down" else helper_vec
            hook._start_helper = lambda r, c: calls.__setitem__("started", calls["started"] + 1)
            if gate is not None:
                hook.should_use_semantic = gate
            ev, buf = {}, io.StringIO()
            raw = json.dumps({"session_id": f"mode-{mode}", "prompt": prompt})
            with contextlib.redirect_stdout(buf):
                hook._run(raw, ev)
            return ev, buf.getvalue()

        ev, out = run("always", "zzqplex wibblefrotz quuxborg nothing matches", helper_vec=[0.1, 0.2])
        check("always: the helper's vector is handed to the meaning search",
              calls["semantic"] == [[0.1, 0.2]] and "a meaning-search memory" in out
              and ev["semantic_reason"] == "always" and ev["mode"] == "always_v3", f"{ev} {calls}")
        ev, out = run("always", "zzqplex wibblefrotz quuxborg nothing matches", helper_vec="down")
        check("always, helper down: the prompt stays keyword-only, never waits, and the helper is restarted",
              calls["semantic"] == [] and calls["started"] == 1 and ev["semantic_reason"] == "helper_unavailable"
              and out.strip() == "{}", f"{ev} {calls}")
        ev, out = run("always", "frobnicator staging rollout feature flag launch", helper_vec=[0.1])
        check("keyword and meaning results together are reported as a hybrid hit",
              ev["outcome"] == "hybrid_hit" and "a meaning-search memory" in out and "frobnicator" in out, f"{ev} {out[:200]}")

        real_gate = hook.should_use_semantic
        ev, out = run("on_miss", "frobnicator staging rollout feature flag launch",
                      gate=lambda *a: (False, "keyword_confident"))
        check("on_miss, gate shut: the model is never loaded", calls["semantic"] == []
              and ev["semantic_reason"] == "keyword_confident")
        ev, out = run("on_miss", "zzqplex wibblefrotz quuxborg nothing matches",
                      gate=lambda *a: (True, "weak_keyword"))
        check("on_miss, gate fires: the model loads in-process (no precomputed vector)",
              calls["semantic"] == [None] and ev["semantic_reason"] == "weak_keyword"
              and "a meaning-search memory" in out)
        hook.should_use_semantic = real_gate
        ev, out = run("off", "zzqplex wibblefrotz quuxborg nothing matches")
        check("off: no meaning search at all", calls["semantic"] == [] and calls["started"] == 0)

    # --- telemetry column on a database from before this change ---
    import metrics_utils as mu
    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.path.join(tmpdir, "old.db")
        c = sqlite3.connect(old)
        c.execute("""CREATE TABLE retrieval_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'live', session_id TEXT, project TEXT, mode TEXT, outcome TEXT NOT NULL,
            prompt_kind TEXT, prompt_chars INTEGER, n_keywords INTEGER, n_results INTEGER, top_rank REAL,
            top_similarity REAL, top_coverage REAL, keyword_ms REAL, semantic_ms REAL, latency_ms REAL,
            result_ids TEXT, result_projects TEXT, result_ages_days TEXT, same_project_n INTEGER,
            injected_chars INTEGER, injected_tokens_est INTEGER)""")
        c.commit()
        c.close()
        ok = mu.record_retrieval_event(old, {"outcome": "weak_keyword", "semantic_reason": "weak_keyword"})
        c = sqlite3.connect(old)
        got = c.execute("SELECT semantic_reason FROM retrieval_events").fetchall()
        c.close()
        check("an older retrieval_events table is upgraded on first write, nothing lost",
              ok and got == [("weak_keyword",)], f"{ok} {got}")


def test_semantic_helper():
    """The always-mode helper: protocol, token check, idle exit, and the
    start-up lock that stops two sessions launching two helpers."""
    print("\n--- Semantic Helper ---")
    import importlib.util
    import socket
    import threading
    import time
    spec = importlib.util.spec_from_file_location("semantic_helper_under_test", ROOT / "scripts" / "semantic_helper.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    with tempfile.TemporaryDirectory() as tmpdir:
        ready = threading.Event()
        thread = threading.Thread(target=helper.serve, args=(tmpdir, lambda text: [1.0, 2.0, float(len(text))]),
                                  kwargs={"idle_min": 0.05, "ready": ready}, daemon=True)
        thread.start()
        check("helper announces itself once ready", ready.wait(5) and helper.read_info(tmpdir) is not None)
        check("a request returns the embedding of the text", helper.query(tmpdir, "hello") == [1.0, 2.0, 5.0])
        check("ping and ensure_running see it as running",
              helper.ping(tmpdir) and helper.ensure_running(tmpdir) == "running")
        check("it only listens on localhost", helper.HOST == "127.0.0.1")
        if os.name != "nt":
            check("the token file is private to the user",
                  (os.stat(helper.info_path(tmpdir)).st_mode & 0o077) == 0)

        info = helper.read_info(tmpdir)
        with socket.create_connection(("127.0.0.1", info["port"]), timeout=2) as s:
            s.sendall((json.dumps({"token": "wrong", "text": "x"}) + "\n").encode())
            reply = s.recv(4096)
        check("a request with the wrong token is refused", b"bad token" in reply and b"vector" not in reply)
        with socket.create_connection(("127.0.0.1", info["port"]), timeout=2) as s:
            s.sendall(b"not json\n")
            s.recv(4096)
        check("garbage input doesn't kill it", helper.query(tmpdir, "still alive") is not None)

        thread.join(timeout=12)
        check("it exits by itself after the idle period and removes its info file",
              not thread.is_alive() and helper.read_info(tmpdir) is None)
        started = time.perf_counter()
        check("asking a helper that is gone returns None quickly, not an error",
              helper.query(tmpdir, "anyone?") is None and time.perf_counter() - started < 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        launched = []
        real_popen = helper.subprocess.Popen
        helper.subprocess.Popen = lambda cmd, **kw: launched.append(cmd)
        try:
            check("no helper and no lock: one is launched", helper.ensure_running(tmpdir) == "started"
                  and len(launched) == 1 and "semantic_helper.py" in launched[0][1])
            check("a second session starting at the same moment launches nothing",
                  helper.ensure_running(tmpdir) == "started" and len(launched) == 1)
            old = time.time() - helper.SPAWN_LOCK_STALE_S - 5
            os.utime(helper._lock_path(tmpdir), (old, old))
            check("a lock left by a start that died is cleared and a new start is allowed",
                  helper.ensure_running(tmpdir) == "started" and len(launched) == 2)
        finally:
            helper.subprocess.Popen = real_popen


def test_memory_check():
    """The automatic Memory Check: question source, when it is due, the plain
    advice it gives, and that its note is shown only once."""
    print("\n--- Memory Check ---")
    import time
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import remember_check as rc
    except Exception as e:
        check("import remember_check", False, str(e))
        return

    def sessions_db(n_good):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE sys_sessions (session_id TEXT, project TEXT, started_at TEXT, notes TEXT)")
        for i in range(n_good):
            notes = (f"Topic: t{i}\n\n## Decisions\n- Use option {i} for the thing because it is simpler to run\n"
                     "- A second decision that should not be picked\n\n## Done\n- stuff\n")
            conn.execute("INSERT INTO sys_sessions VALUES (?, 'oth', ?, ?)", (f"s{i}", f"2026-10-{i + 1:02d}", notes))
        conn.execute("INSERT INTO sys_sessions VALUES ('fb', 'oth', '2026-10-30', 'AUTO-GENERATED FALLBACK note')")
        conn.execute("INSERT INTO sys_sessions VALUES ('none', 'oth', '2026-10-31', '## Decisions\n- None locked in yet; user is still evaluating options.\n')")
        conn.execute("INSERT INTO sys_sessions VALUES ('short', 'oth', '2026-10-31', '## Decisions\n- tiny\n')")
        conn.execute("INSERT INTO sys_sessions VALUES ('nodec', 'oth', '2026-10-31', '## Done\n- only done items here, no decisions section\n')")
        return conn

    conn = sessions_db(15)
    bullets = rc.decision_bullets(conn)
    check("one usable decision per session; fallback, 'none', too-short and no-decision sessions are skipped",
          len(bullets) == 15 and all(b["session"].startswith("s") for b in bullets)
          and all("second decision" not in b["text"] for b in bullets))
    check("markdown marks are stripped from decisions", all("*" not in b["text"] and "`" not in b["text"] for b in bullets))
    picked = rc.pick_evenly(list(range(100)), 10)
    check("sessions are sampled evenly across the whole history", len(picked) == 10 and picked[0] == 0 and picked[-1] >= 85)
    check("a template question is built from the decision text",
          rc.template_question("Use storage.mode: local instead of synced for this machine today") .startswith("What did we decide about Use storage.mode"))

    real_llm, real_vet = rc.llm_questions, rc.vet_decisions
    rc.vet_decisions = lambda conn, config, bullets: (bullets, 0, True)     # vetting has its own test below
    rc.llm_questions = lambda config, texts: (None, "no key")
    qs, info = rc.build_questions(conn, {})
    style = info["style"]
    check("with no API the questions fall back to templates and say why", style == "templated"
          and info["wording_note"] == "no key"
          and sum(q["kind"] == "answerable" for q in qs) == 15
          and sum(q["kind"] == "no_answer" for q in qs) == len(rc.NO_ANSWER_QUESTIONS))
    check("each question points at the session its decision came from",
          all(q["expect"] and q["expect"][0].startswith("s") for q in qs if q["kind"] == "answerable"))
    rc.llm_questions = lambda config, texts: ([[f"how did we handle thing number {i}?", f"what about number {i} again?"]
                                               for i, _ in enumerate(texts)], "ok")
    qs, info = rc.build_questions(conn, {})
    style = info["style"]
    answerable = [q for q in qs if q["kind"] == "answerable"]
    check("reworded questions are used when the API answers", style == "reworded"
          and qs[0]["question"] == "how did we handle thing number 0?")
    check("each decision is asked two ways, both pointing at the same session",
          len(answerable) == 30 and answerable[0]["expect"] == answerable[1]["expect"]
          and answerable[0]["id"] != answerable[1]["id"] and info["decisions_used"] == 15)
    rc.llm_questions, rc.vet_decisions = real_llm, real_vet

    # --- only fair questions: the decision must really be in its own session ---
    import types
    import numpy as np
    vocab = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]

    def vec(text):
        v = np.array([1.0 if w in text.lower() else 0.0 for w in vocab], dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            return np.stack([vec(t) for t in text]) if isinstance(text, list) else vec(text)

    fake = types.ModuleType("semantic_search")
    fake.get_embed_model = lambda config: FakeModel()
    real_module = sys.modules.get("semantic_search")
    sys.modules["semantic_search"] = fake
    try:
        v = sqlite3.connect(":memory:")
        v.execute("CREATE TABLE sys_sessions (session_id TEXT, project TEXT, started_at TEXT, notes TEXT)")
        v.execute("CREATE TABLE transcripts (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT)")
        v.execute("CREATE TABLE transcript_embeddings (transcript_id INTEGER, embedding BLOB)")

        def add(sid, decision, said):
            v.execute("INSERT INTO sys_sessions VALUES (?, 'oth', '2026-10-01', ?)",
                      (sid, f"## Decisions\n- {decision} and more words to be long enough here\n"))
            cur = v.execute("INSERT INTO transcripts (session_id, content) VALUES (?, ?)", (sid, said))
            v.execute("INSERT INTO transcript_embeddings VALUES (?, ?)", (cur.lastrowid, vec(said).tobytes()))
        add("said_it", "alpha bravo charlie delta", "we picked alpha bravo charlie delta today")
        add("never_said", "echo foxtrot golf hotel", "talked about india juliet only")
        add("twin", "alpha bravo charlie delta", "alpha bravo charlie delta again")
        kept, dropped, ran = rc.vet_decisions(v, {}, rc.decision_bullets(v))
        by_session = {b["session"]: b for b in kept}
        check("a decision its own session never actually said is dropped", "never_said" not in by_session and dropped == 1 and ran is True)
        check("a decision backed by its session's text is kept", "said_it" in by_session)
        check("a session that recorded the same decision is also accepted as correct",
              by_session["said_it"]["also_answers"] == ["twin"])
        v.execute("DROP TABLE transcript_embeddings")
        unvetted = rc.vet_decisions(v, {}, rc.decision_bullets(v))
        check("with no embeddings to judge by, vetting leaves the decisions alone (no crash) and says it didn't run",
              unvetted[1] == 0 and unvetted[2] is False and len(unvetted[0]) == 3)
    finally:
        if real_module is not None:
            sys.modules["semantic_search"] = real_module
        else:
            sys.modules.pop("semantic_search", None)

    # --- the Haiku call says WHY it fell back ---
    class FakeResp:
        def __init__(self, text, stop="end_turn"):
            self.content = [types.SimpleNamespace(type="text", text=text)]
            self.stop_reason = stop

    def with_api(handler):
        mod = types.ModuleType("anthropic")

        class Client:
            def __init__(self, **kw):
                pass
            messages = types.SimpleNamespace(create=lambda **kw: handler(kw))
        mod.Anthropic = Client
        saved = sys.modules.get("anthropic", "absent")
        sys.modules["anthropic"] = mod
        return saved

    def restore_api(saved):
        if saved == "absent":
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = saved

    two = ["alpha one", "beta two"]
    for label, handler, expect in (
            ("a good reply", lambda kw: FakeResp('[["what is alpha?", "tell me alpha"], ["what is beta?"]]'), "ok"),
            ("a reply that was cut off", lambda kw: FakeResp('[["what is al', "max_tokens"), "cut off"),
            ("a reply that isn't JSON", lambda kw: FakeResp("sorry, here you go"), "not valid JSON"),
            ("a reply with the wrong number of entries", lambda kw: FakeResp('[["what is alpha?"]]'), "one entry per decision"),
            ("an API error", lambda kw: (_ for _ in ()).throw(RuntimeError("boom")), "API call failed (RuntimeError)")):
        saved = with_api(handler)
        try:
            got, why = rc.llm_questions({}, two)
        finally:
            restore_api(saved)
        check(f"Haiku call, {label}: reason is '{expect}'", expect in why and ((got is not None) == (expect == "ok")), f"{got} {why}")
    saved = with_api(lambda kw: FakeResp('[["what is alpha?", "tell me alpha"], ["what is beta?"]]'))
    try:
        got, why = rc.llm_questions({}, two)
    finally:
        restore_api(saved)
    check("one valid question for a decision is accepted (two are not required)", got == [["what is alpha?", "tell me alpha"], ["what is beta?"]])
    sys.modules["anthropic"] = None            # makes `import anthropic` raise ImportError
    try:
        got, why = rc.llm_questions({}, two)
    finally:
        sys.modules.pop("anthropic", None)
    check("a missing anthropic package is named as the reason", got is None and "not installed" in why)

    # --- when it is due ---
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {}
        small = sessions_db(5)
        check("not enough history: not due, and says why", rc.is_due(small, tmpdir, cfg)[0] is False
              and "not enough history" in rc.is_due(small, tmpdir, cfg)[1])
        check("first time with enough history: due", rc.is_due(conn, tmpdir, cfg) == (True, "first check"))
        check("turned off in config: never due", rc.is_due(conn, tmpdir, {"memory_check": {"enabled": False}})[0] is False)
        now = time.time()
        rc.write_state(tmpdir, {"last_run_at": now - 3 * 86400, "last_session_count": 5})
        check("ran 3 days ago: not due", rc.is_due(conn, tmpdir, cfg, now=now) == (False, "ran recently"))
        rc.write_state(tmpdir, {"last_run_at": now - 20 * 86400, "last_session_count": 12})
        check("20 days ago but only 3 new sessions: not due",
              rc.is_due(conn, tmpdir, cfg, now=now)[0] is False)
        rc.write_state(tmpdir, {"last_run_at": now - 20 * 86400, "last_session_count": 3})
        check("20 days ago and 12 new sessions: due", rc.is_due(conn, tmpdir, cfg, now=now) == (True, "due"))
        check("a check already running blocks another (atomic lock)", rc.acquire_run_lock(tmpdir)
              and not rc.acquire_run_lock(tmpdir) and rc.is_due(conn, tmpdir, cfg, now=now) == (False, "already running"))
        old = now - 3600
        os.utime(rc.lock_path(tmpdir), (old, old))
        check("a lock left by a crash expires instead of blocking forever", not rc.lock_active(tmpdir, now)
              and rc.is_due(conn, tmpdir, cfg, now=now)[0] is True)
        check("an expired lock can be taken over", rc.acquire_run_lock(tmpdir, now))
        rc.release_run_lock(tmpdir)
        check("releasing the lock lets the next check start", rc.acquire_run_lock(tmpdir))
        rc.release_run_lock(tmpdir)
        rc.write_state(tmpdir, {"last_failed_at": now - 3600})
        check("a check that failed an hour ago waits instead of retrying on every start",
              rc.is_due(conn, tmpdir, cfg, now=now)[0] is False and "failed recently" in rc.is_due(conn, tmpdir, cfg, now=now)[1])
        rc.write_state(tmpdir, {"last_failed_at": now - 25 * 3600})
        check("after the back-off it tries again", rc.is_due(conn, tmpdir, cfg, now=now)[0] is True)
        os.remove(rc.state_path(tmpdir))

        launched = []
        real_popen = rc.subprocess.Popen
        rc.subprocess.Popen = lambda cmd, **kw: launched.append(cmd)
        try:
            check("startup launches the check in the background with --auto",
                  rc.launch_background(tmpdir) and launched and launched[0][-1] == "--auto")
        finally:
            rc.subprocess.Popen = real_popen

        # --- one-time notice ---
        rc.post_notice(tmpdir, "Memory Check finished. All good.")
        first, second = rc.take_notice(tmpdir), rc.take_notice(tmpdir)
        check("the note is handed out once, then never again", first == "Memory Check finished. All good." and second is None)
        rc.post_notice(tmpdir, "first note")
        rc.post_notice(tmpdir, "second note replaces an unread one")
        check("a newer note replaces an unread one, and is still shown exactly once",
              rc.take_notice(tmpdir) == "second note replaces an unread one" and rc.take_notice(tmpdir) is None)

        # --- run_check: lock first, failures logged and backed off, gate counts decisions ---
        dbfile = os.path.join(tmpdir, "check.db")
        db = sqlite3.connect(dbfile)
        db.execute("CREATE TABLE sys_sessions (session_id TEXT, project TEXT, started_at TEXT, notes TEXT)")
        db.commit()
        db.close()
        real_resolve, real_build = rc.resolve_db_path, rc.build_questions
        rc.resolve_db_path = lambda c: dbfile
        try:
            rc.build_questions = lambda conn, config: (_ for _ in ()).throw(RuntimeError("model would not load"))
            try:
                rc.run_check({}, root=tmpdir)
                raised = False
            except RuntimeError:
                raised = True
            st = rc.read_state(tmpdir)
            check("a crashed check re-raises, is logged, is backed off, and releases its lock",
                  raised and "last_failed_at" in st and "model would not load" in st["last_error"]
                  and "model would not load" in open(rc.log_path(tmpdir)).read() and not rc.lock_active(tmpdir))
            check("...so the next startup does not launch it again", rc.is_due(conn, tmpdir, cfg)[0] is False)
            rc.write_state(tmpdir, {})
            rc.build_questions = lambda conn, config: ([], {"decisions_used": 6, "style": "reworded"})
            out = rc.run_check({}, root=tmpdir)
            check("only 6 decisions (12 questions) is not enough history: skipped, and waits its interval",
                  "skipped" in out and "6 usable" in out["skipped"] and "last_run_at" in rc.read_state(tmpdir))
            rc.acquire_run_lock(tmpdir)
            check("a second check while one is running does nothing",
                  rc.run_check({}, root=tmpdir) == {"skipped": "another check is already running"})
            rc.release_run_lock(tmpdir)
        finally:
            rc.resolve_db_path, rc.build_questions = real_resolve, real_build

    # --- advice and wording ---
    def result(kw, om, al, junk_kw=0.0, junk_al=0.25):
        def m(r, junk):
            return {"recall_at_5": r, "false_injection_rate": junk, "model_load_rate": 0.4}
        return {"methods": {"keyword": m(kw, junk_kw), "on_miss": m(om, junk_kw), "always": m(al, junk_al)},
                "counts": {"answerable": 20, "no_answer": 4}, "created_at": "2026-11-11T00:00:00Z",
                "question_style": "reworded"}
    check("meaning search adds almost nothing: advice is to keep things as they are",
          rc.recommend(result(0.7, 0.72, 0.74), "on_miss")["advice"] == "keep")
    check("always finds 10+ points more than on_miss: suggest always",
          rc.recommend(result(0.6, 0.65, 0.85), "on_miss")["advice"] == "consider_always")
    check("on always but it finds no more than on_miss: suggest giving the memory back",
          rc.recommend(result(0.6, 0.8, 0.82), "always")["advice"] == "consider_on_miss")
    easy = result(0.8, 0.9, 0.9)
    easy["question_style"] = "templated"
    easy["question_info"] = {"wording_note": "the reply was cut off"}
    rec = rc.recommend(easy, "on_miss")
    check("template questions are too easy to judge by: no verdict, the real reason, and how to fix it",
          rec["advice"] == "unreliable" and "the reply was cut off" in rec["headline"] and "--now" in rec["headline"])
    mixed = result(0.6, 0.65, 0.85)
    mixed["counts"]["answerable"] = 15
    mixed["question_info"] = {"decisions_used": 10, "vetting_ran": False}
    text = rc.render_report(mixed)
    check("the report doesn't claim 'each asked two ways' when some were asked once", "one or two ways" in text)
    check("the report says when the fairness filter couldn't run", "fairness filter" in text and "couldn't run" in text)
    report = rc.render_report(result(0.6, 0.65, 0.85))
    check("the report is plain: scores out of 10, a verdict, and reassurance",
          "out of 10" in report and "WHAT THIS MEANS" in report and "Nothing is broken" in report
          and "<- your setting" in report)


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
    test_live_metrics_report()
    test_eval_scoring()
    test_weak_match_gate()
    test_semantic_modes()
    test_semantic_helper()
    test_memory_check()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
