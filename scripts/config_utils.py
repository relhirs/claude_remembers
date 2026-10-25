#!/usr/bin/env python3
"""
config_utils.py - Shared config helpers for claude_remembers.

storage.mode decides where the SQLite database lives:
  "synced" -> storage.local_db_path (kept outside the synced folder,
              since SQLite corrupts under concurrent sync writes)
  "local"  -> storage.root_path/claude_remembers.db

Every script that opens the database must resolve the path through
resolve_db_path() instead of reading storage.local_db_path directly -
that key is only present (and only required) in "synced" mode.
"""

import os


def resolve_db_path(config):
    """Return the expanded, absolute path to the claude_remembers SQLite DB."""
    storage = config["storage"]
    if storage.get("mode") == "synced":
        db_path = storage["local_db_path"]
    else:
        db_path = os.path.join(storage["root_path"], "claude_remembers.db")
    return os.path.expanduser(db_path)
