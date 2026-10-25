#!/usr/bin/env python3
"""
project_classifier.py - Suggest a project for new content via embedding similarity.

Builds a per-project "centroid" (mean of every embedded message's vector
under that project) from transcript_embeddings, then compares a new
conversation's own mean embedding against each centroid with the same
cosine-similarity math as semantic_search.py.

Suggestion-only: callers must present this as a recommendation and let a
human confirm or override - never write a project assignment from this
alone.
"""

import numpy as np

from semantic_search import get_embed_model


def compute_project_centroids(conn):
    """Return {project_prefix: mean_embedding} for every project with at
    least one embedded message. Projects with zero embedded messages are
    omitted - there's nothing to compare against yet (cold start)."""
    rows = conn.execute(
        """SELECT t.project, te.embedding
           FROM transcript_embeddings te
           JOIN transcripts t ON t.id = te.transcript_id
           WHERE t.project IS NOT NULL"""
    ).fetchall()

    vectors_by_project = {}
    for project, emb_blob in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        vectors_by_project.setdefault(project, []).append(vec)

    return {
        project: np.mean(vecs, axis=0)
        for project, vecs in vectors_by_project.items()
    }


def suggest_project(config, conn, text_samples, centroids=None):
    """Rank existing projects by semantic similarity to text_samples.

    text_samples: list of strings representative of the new conversation
    (e.g. title + a handful of user messages).

    centroids: optional precomputed result of compute_project_centroids(),
    for callers classifying many conversations in a loop - avoids
    rescanning transcript_embeddings on every call.

    Returns a list of (prefix, score) sorted by score descending, or []
    if no project has any embedded content yet. score is a 0-1 cosine
    similarity, not a probability - treat it as a ranking signal only.
    """
    samples = [s for s in text_samples if s and s.strip()]
    if not samples:
        return []

    if centroids is None:
        centroids = compute_project_centroids(conn)
    if not centroids:
        return []

    model = get_embed_model(config)
    sample_vecs = np.array(model.encode(samples), dtype=np.float32)
    query_vec = np.mean(sample_vecs, axis=0)
    query_norm = np.linalg.norm(query_vec)

    scored = []
    for project, centroid in centroids.items():
        norm = query_norm * np.linalg.norm(centroid)
        sim = float(np.dot(query_vec, centroid) / norm) if norm > 0 else 0.0
        scored.append((project, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
