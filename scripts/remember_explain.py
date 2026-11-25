#!/usr/bin/env python3
"""remember_explain.py - show every step of memory retrieval for one question.

Used by the /remember-explain slash command (the question arrives on stdin) and
by the MCP tool explain_retrieval, so both print identical text. It runs
retrieve() (scripts/retrieval.py), the same function the live hook runs, on a
read-only connection. It writes nothing to the memory database: no
retrieval_events row, no "/clear recovery shown" mark. It writes no retrieval.log
line and never starts the warm helper. It may query a helper that is already
running and, if meaning search runs, load the embedding model (the same cost as
the real hook). The one thing it can write: if that model fails to load, the
loader appends an error line to logs/semantic_search.log.

All formatting lives here; retrieve() and the hook know nothing about it.
"""

import argparse
import glob
import importlib.util
import os
import pathlib
import re
import sqlite3
import sys
import textwrap

try:                                    # as a script (scripts/ on sys.path) ...
    import retrieval
    from config_utils import resolve_db_path
except ImportError:                     # ... or imported by the MCP server as scripts.X
    from scripts import retrieval
    from scripts.config_utils import resolve_db_path

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOOK_PATH = ROOT / "hooks" / "user-prompt-submit.py"
MINUS = "−"
MODES = ("off", "on_miss", "always")


class ExplainError(Exception):
    """A problem the user can act on; the message says what happened and what to do."""


# ---------------------------------------------------------------------------
# Glossary: every technical term, in one place. A term is explained the first
# time it appears in the output and is just its name after that.
# ---------------------------------------------------------------------------

GLOSSARY = {
    "bm25": ("BM25 score", "how strongly the words match; more negative = stronger match"),
    "fts5": ("FTS5", "SQLite's full-text search"),
    "strength_check": ("strength check", "the best keyword result must be strong enough, or none are used"),
    "per_result_min": ("minimum for any single result", "weaker results are dropped one by one"),
    "recency": ("adjusted for age", "older memories need a stronger match to rank as high"),
    "coverage": ("coverage", "how many of your search words a result contains"),
    "memory_phrasing": ("memory phrasing", "phrases like \"what did we decide\""),
    "setting": ("meaning-search setting", "off, on_miss or always"),
    "on_miss": ("on_miss", "meaning search runs only when keyword results look unreliable"),
    "always": ("always", "meaning search runs on every question, using a helper that stays running"),
    "similarity": ("similarity", "how close in meaning, 0 to 1; higher = closer"),
    "cutoff": ("cutoff", "the bar a result must clear"),
    "rrf": ("combined by rank", "a memory both searches found moves up (Reciprocal Rank Fusion)"),
    "session": ("conversation", "one Claude Code session"),
    "machine": ("machine message", "text the tools wrote, not you or Claude"),
    "token": ("tokens", "roughly 3/4 of a word each: Claude's reading budget"),
}

# One table for every threshold, so the words always match the code's real comparison
# (strict vs inclusive). Each entry cites the comparison in scripts/retrieval.py.
THRESHOLD_WORDING = {
    "per_result": "{t} or stronger",      # _keyword_candidates drops a row if rank > FTS_RANK_FLOOR
    "strength": "{t} or stronger",        # weak if best > floor (_strength_gate)
    "cutoff": "above {t}",                # semantic_rows keeps similarity > SEMANTIC_CUTOFF
    "coverage": "at least {t}",           # should_use_semantic fires if max coverage < GATE_MIN_COVERAGE
    "length": "at least {t}",             # skipped if len(prompt) < SHORT_PROMPT_CHARS
}

KEYWORD_DROP_WORDS = {"too_short": "under 3 letters", "stop_word": "too common to help",
                      "repeat": "repeat", "over_cap": f"over the {retrieval.MAX_KEYWORDS}-word limit"}


class Glossary:
    def __init__(self):
        self.used = set()

    def __call__(self, key):
        name, gloss = GLOSSARY[key]
        if key in self.used:
            return name
        self.used.add(key)
        return f"{name} ({gloss})"


def num(x, d):
    return f"{x:.{d}f}".replace("-", MINUS)


def pair(value, threshold, d):
    """Value and threshold with the same number of decimals; one more decimal
    each when they would otherwise print the same although they differ."""
    dd = d
    while dd < 6 and value != threshold and f"{value:.{dd}f}" == f"{threshold:.{dd}f}":
        dd += 1
    return num(value, dd), num(threshold, dd)


def bm25(x):
    return num(x, 2)


def rule(kind, threshold, d):
    return THRESHOLD_WORDING[kind].format(t=num(threshold, d))


def short_snippet(text, n=80):
    text = (text or "").replace("\n", " ").strip()
    return f'"{text[:n]}{"…" if len(text) > n else ""}"'


def table(headers, rows, indent=6, width=100, right=()):
    """A plain-text table that lines up in a monospace block (a terminal or a chat
    code block). Columns are separated by ' | '. If the table is too wide, the
    widest columns shrink and their cells wrap onto extra lines, so no line passes
    `width`. Columns listed in `right` (by position) are right-aligned numbers."""
    cols = len(headers)
    cells = [[str(c) for c in r] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]
    avail = width - indent - 3 * (cols - 1)
    while sum(w) > avail:
        i = max(range(cols), key=lambda k: w[k])
        if w[i] <= max(len(headers[i]), 8):
            break
        w[i] -= 1

    def line(parts):
        out = [(t.rjust(w[i]) if i in right else t.ljust(w[i])) for i, t in enumerate(parts)]
        return " " * indent + " | ".join(out).rstrip()

    lines = [line(headers), " " * indent + "-+-".join("-" * n for n in w)]
    for r in cells:
        wrapped = [textwrap.wrap(t, w[i], break_long_words=True, break_on_hyphens=False) or [""]
                   for i, t in enumerate(r)]
        for k in range(max(len(x) for x in wrapped)):
            lines.append(line([x[k] if k < len(x) else "" for x in wrapped]))
    return lines


def _day(view):
    return (view.get("timestamp") or "")[:10]


def top_n(items, n=5):
    return items[:n], max(0, len(items) - n)


def wrap(text, indent=2, width=96):
    return textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _count_by(items, key):
    out = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out


def _gate_sentence(gate, trace, g):
    """Why meaning search would or wouldn't run in on_miss mode, in plain words."""
    reason, cov = gate.get("reason"), gate.get("best_coverage")
    n_kw = gate.get("n_keywords") or 0
    if reason == "weak_keyword":
        return "keyword results were too weak"
    if reason == "no_keyword_result":
        return "keyword search found nothing"
    if reason == "low_coverage":
        found = round((cov or 0) * n_kw)
        return (f"the best result had only {found} of {n_kw} search words ({num(cov, 2)}), "
                f"needs {rule('coverage', gate['min_coverage'], 2)}")
    if reason == "memory_phrasing":
        return f"your question sounds like it is about the past (\"{gate.get('phrase')}\")"
    if reason == "keyword_confident":
        found = round((cov or 0) * n_kw)
        return (f"keyword results look trustworthy: the best result has {found} of {n_kw} search words "
                f"({num(cov, 2)}, needs {rule('coverage', gate['min_coverage'], 2)}) and no memory phrasing")
    return "the check could not be completed"


def _plural_gloss(g, key):
    """The glossary term in the plural, with its explanation after it the first time:
    'machine messages (text the tools wrote, not you or Claude)'."""
    name, sep, rest = g(key).partition(" (")
    return name + "s" + (sep + rest if sep else "")


def THISRULE(t):
    return THRESHOLD_WORDING["strength"].format(t=t)


def _why_meaning_ran(trace, g):
    sem, gate, mode = trace.get("semantic") or {}, trace.get("gate") or {}, trace["inputs"]["mode"]
    if mode == "always":
        return f"your setting is {g('always')}"
    return _gate_sentence(gate, trace, g) if gate.get("reason") else "the keyword results looked unreliable"


def summarize(trace, ctx, g):
    inp, sem, kw = trace["inputs"], trace["semantic"] if "semantic" in trace else {}, trace.get("keyword", {})
    outcome = trace.get("outcome")
    rows = trace.get("rows", [])
    n = len(rows)
    strength = (kw.get("strength_gate") or {})
    best, floor = strength.get("best"), strength.get("floor")
    lines = []
    if ctx.get("clear_pending"):
        lines.append(wrap("Right after a /clear, the hook shows your last session's summary plus a short "
                          "keyword-only list of related messages, with no meaning search; below is what the "
                          "normal search would find.", 2))
        lines.append("")
    if outcome == "skipped_short":
        text = (f"Nothing would be searched: the question is {inp['prompt_chars']} characters and the hook "
                f"skips anything under {retrieval.SHORT_PROMPT_CHARS}.")
    elif outcome == "skipped_no_keywords":
        text = "Nothing would be searched: every word is too common or too short to search on."
    elif outcome == "no_match_semantic_unavailable":
        text = ("No memories would be added. Keyword search found nothing, and meaning search is not "
                "available on this machine, so there was no second chance.")
    elif outcome == "keyword_hit":
        why = ""
        if trace.get("helper_unavailable"):
            why = ("The meaning-search helper isn't running, so this question stays keyword-only "
                   "(the hook would start the helper for the next question).")
        elif inp["mode"] == "off":
            why = "Meaning search is switched off in your settings."
        elif sem.get("ran"):
            why = "Meaning search also ran but added nothing new."
        elif sem.get("unavailable"):
            why = "Meaning search is not available on this machine."
        else:
            why = "Keyword results looked trustworthy, so meaning search was skipped."
        bm = ""
        if best is not None and floor is not None:
            bp, fp = pair(best, floor, 2)
            bm = f" Keyword search found strong matches (best {bp}, needs {THISRULE(fp)})."
        text = f"{n} memor{'y' if n == 1 else 'ies'} would be added, all from keyword search.{bm} {why}"
    elif outcome == "hybrid_hit":
        by = _count_by(trace.get("merge", []), "source")
        text = (f"{n} memories would be added: {by.get('keyword', 0)} only from keyword search, "
                f"{by.get('meaning', 0)} only from meaning search and {by.get('both', 0)} found by both; "
                f"the two lists are {g('rrf')}. Meaning search ran because {_why_meaning_ran(trace, g)}.")
    elif outcome == "semantic_hit":
        text = (f"{n} memor{'y' if n == 1 else 'ies'} would be added, all from meaning search. Keyword search "
                f"found nothing usable, so meaning search ran and found {n} close enough.")
    else:   # no_match, weak_keyword left standing
        if trace.get("helper_unavailable"):
            text = ("No memories would be added. The meaning-search helper isn't running, so this question "
                    "stays keyword-only, and keyword search found nothing usable.")
        elif sem.get("ran"):
            top, cut = sem.get("top_similarity"), retrieval.SEMANTIC_CUTOFF
            if strength.get("weak") and best is not None:
                bp, fp = pair(best, floor, 2)
                lead = (f"Keyword search found only weak matches (best {bp}, needs {THISRULE(fp)}), "
                        "so meaning search ran, but ")
            else:
                lead = "Keyword search found nothing usable, so meaning search ran, but "
            if top is None:
                text = f"No memories would be added. {lead}it found nothing at all."
            else:
                tp, cp = pair(top, cut, 3)
                text = (f"No memories would be added. {lead}the closest match in meaning was {tp} and it "
                        f"needs to be {THRESHOLD_WORDING['cutoff'].format(t=cp)}.")
        elif inp["mode"] == "off":
            text = "No memories would be added. Keyword search found nothing usable, and meaning search is switched off."
        else:
            text = "No memories would be added. Keyword search found nothing usable."
    lines.append(wrap(text, 2))
    proj = ctx.get("project_label")
    sess = inp.get("exclude_session")
    src = "forced for this explanation" if inp.get("mode_source") == "given" else "from your config"
    lines.append("")
    lines += table(["Setting", "Value"], [
        ("Project", f"{inp['project']}" + (f" ({proj})" if proj else "") if inp["project"] else "none (folder not mapped)"),
        ("Conversation left out",
         f"{sess[:8]}… ({g('session')} that Claude already has)" if sess else "none given"),
        (g('setting')[0].upper() + g('setting')[1:], f"{inp['mode']}, {src}"),
        ("As of", inp['as_of'] or 'now'),
    ], indent=2)
    if ctx.get("project_note"):
        lines.append(wrap(ctx["project_note"], 2))
    return lines


def wrap_output(text, width=100):
    """Wrap any line longer than `width` with a hanging indent, so the output
    reads the same in a terminal, a chat window or a copied snippet."""
    out = []
    for line in text.splitlines():
        if len(line) <= width:
            out.append(line)
            continue
        lead = len(line) - len(line.lstrip())
        m = re.match(r"\s*\d+\. ", line)
        hang = len(m.group(0)) if m else lead + 4
        out.append(textwrap.fill(line, width=width, subsequent_indent=" " * hang,
                                 break_long_words=False, break_on_hyphens=False))
    return "\n".join(out)


def render(trace, ctx=None):
    """The full explanation as text, wrapped at 100 columns. `ctx` carries things
    the trace doesn't: project_label, project_note, clear_pending (session id),
    helper_note."""
    return wrap_output(_render(trace, ctx or {}))


def _render(trace, ctx):
    g = Glossary()
    out = ["WHAT HAPPENED"]
    out += summarize(trace, ctx, g)
    out += ["", "STEP BY STEP  (the numbered lines alone tell the story)"]

    inp = trace["inputs"]
    kw = trace.get("keyword") or {}
    sem = trace.get("semantic") or {}
    skip = trace.get("skip")
    steps = []

    def step(n, headline, *details):
        block = [f" {n}. {headline}"]
        for d in details:
            if d is None:
                continue
            block.extend(d if isinstance(d, list) else [d])
        steps.append("\n".join(block))

    skipped_because = ("the question is too short" if skip == "skipped_short"
                       else "there are no usable search words" if skip == "skipped_no_keywords" else None)

    # 1. long enough
    chars = inp["prompt_chars"]
    ok = skip != "skipped_short"
    need = rule("length", retrieval.SHORT_PROMPT_CHARS, 0)
    step(1, f"Long enough to search? {'Yes' if ok else 'No'}: {chars} characters, needs {need}.")

    # 2. words
    kws = trace["keywords"]
    dropped = {}
    for w, why in kws["dropped"]:
        dropped.setdefault(why, []).append(w)
    set_aside = " · ".join(f"{', '.join(ws)} ({KEYWORD_DROP_WORDS[why]})" for why, ws in dropped.items())
    if skip == "skipped_short":
        step(2, f"Search words: skipped, because {skipped_because}.")
    else:
        kept_txt = ", ".join(kws["kept"]) or "none"
        q = kws.get("fts_query")
        step(2, f"Picked {len(kws['kept'])} search word{'s' if len(kws['kept']) != 1 else ''}"
                f"{', set ' + str(sum(len(v) for v in dropped.values())) + ' aside' if dropped else ''}.",
             f"      kept: {kept_txt}",
             f"      set aside: {set_aside}" if set_aside else None,
             f"      query sent to {g('fts5')}: {q}" if q else None)

    # 3. keyword search
    if skipped_because:
        for n_, label in ((3, "Keyword search"), (4, "Strength check"), (5, "Should meaning search run?"),
                          (6, "Meaning search")):
            step(n_, f"{label}: skipped, because {skipped_because}.")
        step(7, "What was left out, and why: nothing was searched.")
        step(8, "Final list: empty.")
        step(9, "Would add nothing to Claude's context (0 tokens).")
        return "\n".join(out + steps + [""])

    cands = kw.get("candidates", [])
    raw_found = len(kw.get("project_candidates", [])) + len(kw.get("global_candidates", []))
    detail = "dropped" in kw
    shown, more = top_n(cands)
    head = (f"Keyword search found {raw_found} possible match{'es' if raw_found != 1 else ''} and kept {len(cands)}."
            if detail else f"Keyword search kept {len(cands)} result{'s' if len(cands) != 1 else ''}.")
    body = []
    if cands:
        body.append(textwrap.fill(f"Ranked by {g('bm25')}, {g('recency')}:", width=100,
                                  initial_indent="      ", subsequent_indent="        "))
        body += table(["#", "Memory", "Project", "Date", "BM25", "Adjusted", "Days old", "id"],
                      [(i, short_snippet(v["snippet"], 70), v.get("project") or "", _day(v), bm25(v["rank"]),
                        "" if v.get("adjusted_rank") is None else bm25(v["adjusted_rank"]),
                        "" if v.get("age_days") is None else f"{v['age_days']:g}", v.get("id"))
                       for i, v in enumerate(shown, 1)], right={0, 4, 5, 6, 7})
        body.append("      (Adjusted = the BM25 score after the age adjustment; it decides the order.)")
        if more:
            body.append(f"      … and {more} more, all weaker")
    step(3, head, body if body else None)

    # 4. strength check
    sg = kw.get("strength_gate") or {}
    if not sg.get("applied"):
        step(4, "Strength check: skipped, there were no keyword matches to check.")
    elif sg.get("floor") is None:
        step(4, "Strength check: skipped, the search index has no size statistics.")
    else:
        bp, fp = pair(sg["best"], sg["floor"], 2)
        verdict = "failed, the best match wasn't strong enough" if sg["weak"] else "passed, the best match was strong enough"
        step(4, f"{g('strength_check')[0].upper() + g('strength_check')[1:]}: {verdict}.",
             f"      best {bp}; the best result must reach {THISRULE(fp)} "
             f"(this bar rises as memory grows: {retrieval.MIN_STRENGTH_PER_LOG_DOC} × ln({sg['n_docs']} messages))",
             f"      {g('per_result_min')}: {rule('per_result', retrieval.FTS_RANK_FLOOR, 1)}")

    # 5. should meaning search run
    gate = trace.get("gate") or {}
    mode = inp["mode"]
    gate_text = _gate_sentence(gate, trace, g) if gate.get("reason") else None
    if sem.get("unavailable"):
        step(5, "Should meaning search run? No: it is not available on this machine.")
    elif mode == "off":
        step(5, f"Should meaning search run? No: your {g('setting')} is off.",
             f"      (for reference, in on_miss mode the answer would be {'yes' if gate.get('fires') else 'no'}: {gate_text})"
             if gate_text else None)
    elif mode == "always":
        if trace.get("helper_unavailable"):
            step(5, f"Should meaning search run? Not this time: mode {g('always')} needs the helper, and it isn't running.",
                 "      the hook would stay keyword-only and start the helper for next time; explain did not start it")
        else:
            step(5, f"Should meaning search run? Yes: mode {g('always')} runs it every time.",
                 f"      (for reference, in on_miss mode the answer would be {'yes' if gate.get('fires') else 'no'}: {gate_text})"
                 if gate_text else None)
    else:
        detail5 = None
        if gate.get("error"):
            detail5 = f"      the check itself failed ({gate['error']})"
        step(5, f"Should meaning search run? {'Yes' if gate.get('fires') else 'No'}: {gate_text or 'the check could not be completed'}.",
             textwrap.fill(f"Mode {g('on_miss')}. It checks {g('coverage')} and {g('memory_phrasing')}.",
                           width=100, initial_indent="      ", subsequent_indent="        ") if gate_text else None,
             detail5)

    # 6. meaning search
    if not sem.get("ran"):
        why = ("it is not available" if sem.get("unavailable") else "your setting is off" if mode == "off"
               else "the helper isn't running" if trace.get("helper_unavailable") else "keyword results looked trustworthy"
               if gate.get("fires") is False else "it did not run")
        step(6, f"Meaning search: skipped, because {why}.")
    else:
        cands6 = sem.get("candidates", [])
        kept6 = [c for c in cands6 if c["status"] == "kept"]
        top = sem.get("top_similarity")
        if kept6:
            head6 = f"Meaning search found {len(cands6)} candidates and kept {len(kept6)}."
        elif top is not None:
            head6 = "Meaning search found nothing close enough."
        else:
            head6 = "Meaning search found nothing."
        body6 = []
        if cands6:
            body6.append(textwrap.fill(
                f"Ranked by {g('similarity')}. The {g('cutoff')}: "
                f"{THRESHOLD_WORDING['cutoff'].format(t=num(retrieval.SEMANTIC_CUTOFF, 3))}.",
                width=100, initial_indent="      ", subsequent_indent="        "))
            shown6, more6 = top_n(cands6)
            status_words = {"kept": "kept", "below_cutoff": "left out: below the cutoff",
                            "current_session": "left out: this conversation",
                            "machine_message": "left out: machine message"}
            body6 += table(["#", "Memory", "Project", "Date", "Similarity", "id", "Result"],
                           [(i, short_snippet(c["snippet"], 70), c.get("project") or "", _day(c),
                             "" if c.get("similarity") is None else num(c["similarity"], 3), c.get("id"),
                             status_words.get(c["status"], c["status"]))
                            for i, c in enumerate(shown6, 1)], right={0, 4, 5})
            if more6:
                body6.append(f"      … and {more6} more, all further")
        if sem.get("error"):
            body6.append(f"      meaning search failed ({sem['error']}); the hook ignores that and goes on without it")
        step(6, head6, body6 if body6 else None)

    # 7. what was left out: counts go in the headline so the headlines alone tell the story
    parts, notes = [], []
    if kw:
        names = {"duplicate": "duplicates (same project and opening words)",
                 "below_floor": f"too weak for any single result ({rule('per_result', retrieval.FTS_RANK_FLOOR, 1)} needed)",
                 "over_cap": f"over the {retrieval.MAX_RESULTS}-result limit"}
        for why, count in _count_by(kw.get("dropped", []), "reason").items():
            parts.append((count, names[why]))
        if sg.get("weak"):
            parts.append((len(cands), "set aside by the strength check"))
        sqlf = kw.get("sql_filtered") or {}
        for key in ("current_session", "machine_message", "after_as_of"):
            if sqlf.get(key):                    # the glossary term is used up only when it is shown
                label = {"current_session": lambda: f"from this {g('session')}",
                         "machine_message": lambda: _plural_gloss(g, "machine"),
                         "after_as_of": lambda: "newer than the as-of time"}[key]()
                notes.append((sqlf[key], label))
    names_m = {"below_cutoff": "below the meaning cutoff", "current_session": "from this conversation",
               "machine_message": "machine messages"}
    for why, count in _count_by([c for c in sem.get("candidates", []) if c["status"] != "kept"], "status").items():
        parts.append((count, f"meaning-search results {names_m[why]}"))
    total = sum(c for c, _ in parts)
    head7 = ("What was left out, and why: nothing." if not parts else
             f"What was left out, and why: {total} in all: " + " · ".join(f"{c} {t}" for c, t in parts) + ".")
    detail7 = None
    if notes:
        detail7 = ["      Also counted just for this explanation (the real search removes these before ranking):"]
        detail7 += table(["Matching messages", "Which"], [(c, t) for c, t in notes], right={0})
    body7 = table(["Left out", "Reason"], [(c, t) for c, t in parts], right={0}) if parts else None
    why5 = None
    if any(d["reason"] == "over_cap" for d in kw.get("dropped", [])):
        why5 = textwrap.fill(
            f"Why {retrieval.MAX_RESULTS}? It is a fixed limit (MAX_RESULTS in scripts/retrieval.py), set when the "
            "hook was written and not tuned against the eval. What it does: whatever is added to Claude's context "
            f"on every prompt stays small ({trace.get('injected_chars', 0)} characters, about "
            f"{trace.get('tokens_est', 0)} tokens for this question). The results over the limit are not errors: they "
            "come after the first 5 in the search's order (matches from your current project first, then the rest by score).", width=100, initial_indent="      ", subsequent_indent="      ")
    step(7, head7, body7, why5, detail7)

    # 8. final list
    rows = trace.get("rows", [])
    merge = {m["id"]: m for m in trace.get("merge", [])}
    sim_by_id = {c["id"]: c["similarity"] for c in sem.get("candidates", [])}
    if not rows:
        step(8, "Final list: empty.")
    else:
        shown8, more8 = top_n(rows)
        srcs = {"keyword": "keyword search", "meaning": "meaning search", "both": "both searches"}
        body8 = table(["#", "Found by", "id", "Memory"],
                      [(i, srcs.get(merge.get(v["id"], {}).get("source"), "keyword search"), v["id"],
                        short_snippet(v["snippet"], 70)) for i, v in enumerate(shown8, 1)], right={0, 2})
        if more8:
            body8.append(f"      … and {more8} more")
        both = any(m.get("source") == "both" for m in merge.values()) or (
            any(m.get("source") == "meaning" for m in merge.values()) and any(m.get("source") == "keyword" for m in merge.values()))
        step(8, f"Final list: {len(rows)} memor{'y' if len(rows) == 1 else 'ies'}"
                f"{', ' + g('rrf') if both else ''}.", body8)

    # 9. tokens
    if rows:
        step(9, f"Would add {len(rows)} memor{'y' if len(rows) == 1 else 'ies'} to Claude's context: "
                f"{trace['injected_chars']} characters, about {trace['tokens_est']} {g('token')}.")
    else:
        step(9, f"Would add nothing to Claude's context: 0 {g('token')}.")

    out += steps
    if ctx.get("helper_note"):
        out += ["", wrap(ctx["helper_note"], 2)]
    return "\n".join(out + [""])


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

_HOOK = None


def _share_package_modules():
    """The hook imports its helpers by bare name (retrieval, config_utils, ...) after
    putting scripts/ on sys.path. When this file runs inside the MCP server it was
    imported as scripts.remember_explain, so those names would load a SECOND copy of
    each module (and a second embedding-model cache). Point the bare names at the
    copies already loaded, so there is one of each. A no-op when run as a script."""
    pkg = __package__
    if not pkg:
        return
    for name in ("config_utils", "retrieval", "semantic_search", "metrics_utils"):
        mod = sys.modules.get(f"{pkg}.{name}")
        if mod is None:
            try:
                mod = importlib.import_module(f"{pkg}.{name}")
            except Exception:           # optional modules (metrics_utils) may be absent; the hook copes
                continue
        sys.modules.setdefault(name, mod)


def _load_hook():
    """The hook module, loaded once per process. Loading runs its top-level code,
    which adds scripts/ to sys.path; doing that on every call made a long-running
    MCP server's sys.path grow by one entry per explain_retrieval call."""
    global _HOOK
    if _HOOK is None:
        _share_package_modules()
        saved_path = list(sys.path)
        try:
            spec = importlib.util.spec_from_file_location("ups_hook_for_explain", HOOK_PATH)
            hook = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hook)
        except BaseException:
            sys.path[:] = saved_path     # a failed load must not leave scripts/ behind
            raise
        _HOOK = hook
    return _HOOK


def detect_session_id(cwd):
    """The newest Claude Code session file for this folder (same rule as
    write_session_notes.auto_detect_session_id), or None."""
    encoded = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    files = glob.glob(os.path.join(os.path.expanduser("~"), ".claude", "projects", encoded, "*.jsonl"))
    if not files:
        return None
    return os.path.splitext(os.path.basename(max(files, key=os.path.getmtime)))[0]


def explain(question, **kwargs):
    """The explanation for `question` as text (see run_explain for the options).
    Raises ExplainError with a plain message when it can't run. Writes nothing to
    the memory database (see the module docstring for the one log line it can write)."""
    trace, ctx = run_explain(question, **kwargs)
    return render(trace, ctx)


def run_explain(question, *, project=None, mode=None, as_of=None, exclude_session=None, root=None,
                cwd=None, semantic_fn=retrieval._DEFAULT, helper_fn=None):
    """Run retrieve() for `question` and return (trace, ctx) for render().
    `semantic_fn` and `helper_fn` (text -> query vector or None) are for tests;
    by default meaning search is the real one and the helper is the real warm helper."""
    question = (question or "").strip()
    if not question:
        raise ExplainError("No question given. Usage: /remember-explain <your question>")
    if mode is not None and mode not in MODES:
        raise ExplainError(f"Unknown mode '{mode}'. Use one of: off, on_miss, always.")
    root = pathlib.Path(root) if root else ROOT
    config_path = root / "config.yaml"
    if not config_path.exists():
        raise ExplainError(f"config.yaml not found at {config_path}. Run scripts/remember-setup.py to create it.")
    config = yaml.safe_load(config_path.read_text()) or {}
    try:
        db_path = resolve_db_path(config)
    except Exception as e:
        raise ExplainError(f"The database path in {config_path} is not usable ({type(e).__name__}: {e}).")
    if not os.path.exists(db_path):
        raise ExplainError(f"Memory database not found at {db_path}. Run scripts/remember-setup.py to create it.")

    hook = _load_hook()
    cwd = cwd or os.environ.get("CWD") or os.getcwd()
    ctx = {}
    hook_bias = hook.mapped_project_for_cwd(config, cwd)      # what the hook favors here (None = nobody)
    if project is None:
        project = hook_bias
        if hook_bias is None:
            ctx["project_note"] = ("This folder isn't mapped to a project, so no project is favored. "
                                   "/clear recovery still checks 'oth'.")
    else:
        if project != hook_bias:
            ctx["project_note"] = (f"You chose project '{project}', so it is favored. "
                                   f"The hook here would favor {repr(hook_bias) if hook_bias else 'none'}.")
    # The /clear check follows the folder, not --project: the hook looks under the folder's
    # mapped project, else 'oth', whatever project is forced for ranking.
    clear_project = hook_bias or "oth"
    if exclude_session is None:
        exclude_session = detect_session_id(cwd)

    conn = sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        try:
            row = (conn.execute("SELECT label FROM project_registry WHERE prefix = ?", (project,)).fetchone()
                   if project else None)
            ctx["project_label"] = row[0] if row and row[0] else None
        except sqlite3.Error:
            pass
        trace = {}
        try:
            retrieval.retrieve(conn, config, question, project=project, exclude_session=exclude_session,
                               as_of=as_of, mode=mode,
                               query_vec_fn=helper_fn or (lambda text: hook._helper_vector(str(root), text)),
                               semantic_fn=semantic_fn, detail=True, trace=trace)
        except sqlite3.Error as e:
            raise ExplainError(f"The search failed ({type(e).__name__}: {e}). Try again, or run scripts/remember_health.py.")
        if exclude_session:
            try:
                pending = hook._find_pending_clear(conn, clear_project, exclude_session)
                if pending:
                    ctx["clear_pending"] = pending[0]
            except sqlite3.Error:
                pass
    finally:
        conn.close()
    if trace.get("helper_unavailable"):
        ctx["helper_note"] = ("The helper isn't answering, so the hook would stay keyword-only for this question "
                              "and start the helper for the next one. Explain did not start it.")
    return trace, ctx


def main(argv=None):
    # Quiet the model loader's progress bars; they would end up in the output.
    for var, val in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("TQDM_DISABLE", "1"),
                     ("TRANSFORMERS_VERBOSITY", "error")):
        os.environ.setdefault(var, val)
    p = argparse.ArgumentParser(description="Explain every step of memory retrieval for one question (question on stdin)")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--as-of")
    p.add_argument("--project")
    p.add_argument("--session", help="conversation to skip (default: the newest for this folder)")
    p.add_argument("words", nargs="*", help="the question, if not given on stdin")
    args = p.parse_args(argv)
    question = " ".join(args.words) if args.words else (sys.stdin.read() if not sys.stdin.isatty() else "")
    try:
        text = explain(question, project=args.project, mode=args.mode, as_of=args.as_of,
                       exclude_session=args.session or None)
    except ExplainError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
