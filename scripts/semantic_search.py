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
"""

_embed_model = None


def get_embed_model(config):
    """Lazily load the SentenceTransformer model. Cached per-process."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from sentence_transformers import SentenceTransformer
    sem_config = config.get("semantic_search", {})
    model_name = sem_config.get("model", "all-MiniLM-L6-v2")
    _embed_model = SentenceTransformer(model_name)
    return _embed_model


def semantic_search(conn, config, query, project=None, limit=10):
    """Cosine-similarity search over transcript_embeddings.

    Returns a list of dicts sorted by similarity descending:
    {"similarity": float, "project": str, "timestamp": str, "content": str}
    """
    import numpy as np

    model = get_embed_model(config)
    query_vec = np.array(model.encode(query), dtype=np.float32)

    if project:
        rows = conn.execute(
            """SELECT te.embedding, t.project, t.timestamp, t.content
               FROM transcript_embeddings te
               JOIN transcripts t ON t.id = te.transcript_id
               WHERE t.project = ?""",
            (project,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT te.embedding, t.project, t.timestamp, t.content
               FROM transcript_embeddings te
               JOIN transcripts t ON t.id = te.transcript_id"""
        ).fetchall()

    if not rows:
        return []

    scored = []
    for emb_blob, proj, ts, content in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        dot = np.dot(query_vec, emb)
        norm = np.linalg.norm(query_vec) * np.linalg.norm(emb)
        sim = float(dot / norm) if norm > 0 else 0.0
        scored.append((sim, proj, ts, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"similarity": sim, "project": proj, "timestamp": ts, "content": content}
        for sim, proj, ts, content in scored[:limit]
    ]
