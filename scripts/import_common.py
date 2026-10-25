#!/usr/bin/env python3
"""
import_common.py - Shared helpers for import_claude_ai.py / import_chatgpt.py /
import_gemini.py.

Centralizes what used to be copy-pasted (with drift) across all three bulk
importers: color printing, keyword-based tag suggestion, building a short
"about" blurb for a conversation, and resolving a human-typed folder name to
a project (creating it on the fly if it doesn't exist yet).

None of this calls out to any model or embedding - the import scripts
deliberately don't make API calls at scan time. "about" text is either an
export's own pre-existing summary (claude.ai sometimes has one) or cheap
string extraction from the first message; tags are plain substring matching.
"""

import importlib.util
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def green(text):
    return f"\033[92m{text}\033[0m"

def yellow(text):
    return f"\033[93m{text}\033[0m"

def red(text):
    return f"\033[91m{text}\033[0m"


# Merged union of the tag keyword dicts that used to live separately in each
# import script (plus remember_tag_review.py's superset where it didn't overlap).
TAG_KEYWORDS = {
    "book-editing": ["edit", "polish", "chapter", "manuscript", "copyedit", "proofread", "rewrite", "revision"],
    "memoir": ["memoir", "johnny", "goods", "maffia", "mob", "harlem", "gangster", "three fingers", "pleasant avenue"],
    "dialogue": ["dialogue", "dialog", "conversation", "speech", "quotes"],
    "job-search": ["resume", "job", "interview", "career", "hiring", "salary", "recruiter", "linkedin"],
    "finance": ["money", "bank", "invest", "stock", "crypto", "budget", "tax", "financial"],
    "coding": ["python", "script", "code", "programming", "debug", "function", "api", "github"],
    "ai-tools": ["chatgpt", "claude", "gpt", "openai", "anthropic", "gemini", "ai model", "llm", "prompt"],
    "tech-setup": ["linux", "fedora", "install", "setup", "config", "terminal", "laptop", "asus", "ubuntu", "samba"],
    "family": ["mom", "mother", "father", "wife", "daughter", "son", "family", "sister", "brother", "grandpa"],
    "health": ["doctor", "medical", "health", "therapy", "leg", "injury", "exercise", "weight"],
    "legal": ["lawyer", "legal", "court", "contract", "lawsuit", "attorney", "will", "estate"],
    "home": ["house", "home", "repair", "plumbing", "electric", "hvac", "mortgage", "renovation"],
    "auto": ["car", "toyota", "highlander", "vehicle", "mechanic", "tire", "oil"],
    "travel": ["travel", "flight", "hotel", "trip", "vacation", "passport", "airport"],
    "music": ["music", "song", "guitar", "band", "album", "spotify"],
    "business": ["business", "company", "startup", "entrepreneur", "marketing", "client", "consultant"],
    "writing": ["writing", "write", "story", "narrative", "voice", "author", "publisher"],
    "research": ["research", "compare", "analysis", "review", "best", "recommend", "options"],
    "remember-project": ["brain", "mcp", "hook", "session", "memory", "database"],
}


def suggest_tags(search_text):
    """Auto-suggest up to 3 tags via plain substring matching against
    TAG_KEYWORDS. `search_text` should already be the lowercased text to
    search (title + a bit of content) - callers build that themselves since
    each import source's message shape is different."""
    if not search_text:
        return ""
    matched = [tag for tag, kws in TAG_KEYWORDS.items() if any(kw in search_text for kw in kws)]
    return ", ".join(matched[:3]) if matched else ""


def _clean_text(text):
    text = re.sub(r'^\*\*.*?\*\*\s*', '', text.strip())  # strip a leading **Heading** like claude.ai's summary
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _truncate(text, max_len):
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len // 2:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


def build_about(native_summary, first_user_text, title, max_len=220):
    """Build a short, descriptive 'what was this about' blurb, with no model
    calls. Prefers an export's own pre-existing summary (only claude.ai
    conversations have this); falls back to the first user message; falls
    back to the title if both are empty."""
    text = ""
    if native_summary:
        text = _clean_text(native_summary)
    if not text and first_user_text:
        text = _clean_text(first_user_text)
    if not text:
        text = title or ""
    return _truncate(text, max_len) if text else "(no content)"


def load_add_project_module():
    """Dynamically load add-project.py (hyphenated filename, not a valid
    module name for a plain import) to reuse create_project()/generate_prefix()
    instead of duplicating project-creation logic here."""
    add_project_path = SCRIPT_DIR / "add-project.py"
    spec = importlib.util.spec_from_file_location("add_project", add_project_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_or_create_project(config, config_path, folder_name, add_project_module=None):
    """Resolve a human-typed folder name to a project prefix, creating the
    project on the fly if it doesn't exist yet. Returns (prefix, label, created: bool).
    """
    existing = next((p for p in config.get("projects", []) if p["folder_name"] == folder_name), None)
    if existing:
        return existing["prefix"], existing["label"], False

    add_project = add_project_module or load_add_project_module()
    existing_prefixes = {p["prefix"] for p in config.get("projects", [])}
    prefix = add_project.generate_prefix(folder_name, existing_prefixes)
    label = folder_name.replace("-", " ").title()
    created = add_project.create_project(config, config_path, folder_name, prefix, label)
    return created["prefix"], created["label"], True
