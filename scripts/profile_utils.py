#!/usr/bin/env python3
"""
profile_utils.py - Shared remember_facts/remember_preferences merge logic.

Used by both mcp/server.py's get_profile() tool and hooks/session-start.py's
automatic context injection, so the merge algorithm lives in exactly one
place. Callers must set conn.row_factory = sqlite3.Row before using these.
"""


def fetch_merged_profile(conn, project=None):
    """Return (facts, pref_rows): global remember_facts/remember_preferences
    rows, merged with `project`-scoped rows when `project` is given.

    Override semantics: for remember_facts, a project-scoped row replaces
    a global row sharing the same (category, key). remember_preferences
    has no key column (the preference text itself IS the content), so
    there's nothing to override on - rows from both scopes are simply
    unioned, and the caller (format_profile_markdown) labels each by scope.
    """
    if project:
        fact_rows = conn.execute(
            """SELECT category, key, value, confidence, project
               FROM remember_facts
               WHERE project IS NULL OR project = ?
               ORDER BY category, key""",
            (project,),
        ).fetchall()
        pref_rows = conn.execute(
            """SELECT category, preference, project
               FROM remember_preferences
               WHERE project IS NULL OR project = ?
               ORDER BY category""",
            (project,),
        ).fetchall()
    else:
        fact_rows = conn.execute(
            "SELECT category, key, value, confidence, project FROM remember_facts "
            "WHERE project IS NULL ORDER BY category, key"
        ).fetchall()
        pref_rows = conn.execute(
            "SELECT category, preference, project FROM remember_preferences "
            "WHERE project IS NULL ORDER BY category"
        ).fetchall()

    merged = {}
    for row in fact_rows:
        key = (row["category"], row["key"])
        if key not in merged or row["project"] is not None:
            merged[key] = row
    facts = sorted(merged.values(), key=lambda r: (r["category"] or "", r["key"] or ""))
    return facts, pref_rows


def format_profile_markdown(facts, pref_rows, heading="## Memory Profile"):
    """Render facts/pref_rows (as returned by fetch_merged_profile) to
    markdown. Returns None if both are empty. Rows with a non-NULL
    `project` are labeled as a project-specific override/addition."""
    if not facts and not pref_rows:
        return None

    lines = [heading, ""]

    if facts:
        lines.append("### Facts")
        current_cat = None
        for row in facts:
            if row["category"] != current_cat:
                current_cat = row["category"]
                lines.append(f"\n**{current_cat}**")
            conf = f" ({row['confidence']})" if row["confidence"] else ""
            scope = " [project override]" if row["project"] else ""
            lines.append(f"- {row['key']}: {row['value']}{conf}{scope}")

    if pref_rows:
        lines.append("\n### Preferences")
        current_cat = None
        for row in pref_rows:
            if row["category"] != current_cat:
                current_cat = row["category"]
                lines.append(f"\n**{current_cat}**")
            scope = f" [project:{row['project']}]" if row["project"] else ""
            lines.append(f"- {row['preference']}{scope}")

    return "\n".join(lines)
