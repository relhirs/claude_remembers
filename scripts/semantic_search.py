#!/usr/bin/env python3
"""
semantic_search.py - Shared meaning-based (embedding) search for claude_remembers.

Used by two callers with very different cost profiles:
  - mcp/server.py's search_semantic tool: a long-lived process, so the
    embedding model's ~4-5s load cost is paid once per Claude Code session.
  - hooks/user-prompt-submit.py: a fresh process per message. It only calls
    this as a FALLBACK when keyword (FTS5) search comes up weak/empty, by
    design - that's the one place per-message search can be slow, and only
    on messages that already failed the fast path.

Factored out so both callers share one implementation instead of two
copies drifting apart.

Errors here (model load failure, corrupt embedding row) are logged to
logs/semantic_search.log and degrade to an empty result list rather than
raising, since this is a fallback path - callers shouldn't crash just
because the fallback itself broke.
"""

import os
from datetime import datetime, timezone

_embed_model = None
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "semantic_search.log")


def _log(message):
    """Append a timestamped line to logs/semantic_search.log. Never raises."""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(_LOG_PATH, "a") as f:
            f.write(f"{ts} {message}\n")
    except Exception:
        pass


def get_embed_model(config):
    """Lazily load the SentenceTransformer model. Cached per-process."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from sentence_transformers import SentenceTransformer
    sem_config = config.get("semantic_search", {})
    model_name = sem_config.get("model", "all-MiniLM-L6-v2")
    # The model is normally already on disk, so look there first: it skips the
    # Hugging Face network request (and its "unauthenticated" warning) on every
    # cold start. Only a first-time download needs the network.
    try:
        try:
            _embed_model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _embed_model = SentenceTransformer(model_name)
    except Exception as e:
        _log(f"ERROR model load failed (model={model_name!r}): {e!r}")
        raise
    return _embed_model


def embed_query(config, text):
    """Embed one piece of text. Returns a float32 numpy vector."""
    import numpy as np
    return np.array(get_embed_model(config).encode(text), dtype=np.float32)


def semantic_search(conn, config, query, project=None, limit=10, query_vec=None):
    """Cosine-similarity search over transcript_embeddings.

    Returns a list of dicts sorted by similarity descending:
    {"similarity": float, "project": str, "timestamp": str, "content": str, "id": int, "session_id": str}
    `query_vec` skips loading the model: the always-on helper embeds the text
    ahead of time and only the scoring happens here.
    Returns [] (and logs) on any failure instead of raising, so a broken
    embedding model or a corrupt row degrades to "no semantic results"
    rather than crashing whichever hook or MCP call invoked this fallback.
    """
    import numpy as np

    if query_vec is None:
        try:
            model = get_embed_model(config)
        except Exception:
            return []  # get_embed_model already logged the specific cause

        try:
            query_vec = np.array(model.encode(query), dtype=np.float32)
        except Exception as e:
            _log(f"ERROR encoding query failed: {e!r}")
            return []

    try:
        if project:
            rows = conn.execute(
                """SELECT te.embedding, t.project, t.timestamp, t.content, t.id, t.session_id
                   FROM transcript_embeddings te
                   JOIN transcripts t ON t.id = te.transcript_id
                   WHERE t.project = ?""",
                (project,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT te.embedding, t.project, t.timestamp, t.content, t.id, t.session_id
                   FROM transcript_embeddings te
                   JOIN transcripts t ON t.id = te.transcript_id"""
            ).fetchall()
    except Exception as e:
        _log(f"ERROR embedding query failed: {e!r}")
        return []

    if not rows:
        return []

    scored = []
    skipped = 0
    for emb_blob, proj, ts, content, row_id, session_id in rows:
        try:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            dot = np.dot(query_vec, emb)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(emb)
            sim = float(dot / norm) if norm > 0 else 0.0
            scored.append((sim, proj, ts, content, row_id, session_id))
        except Exception:
            skipped += 1
            continue

    if skipped:
        _log(f"WARN skipped {skipped}/{len(rows)} rows with unreadable embeddings")

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"similarity": sim, "project": proj, "timestamp": ts, "content": content,
         "id": row_id, "session_id": session_id}
        for sim, proj, ts, content, row_id, session_id in scored[:limit]
    ]
