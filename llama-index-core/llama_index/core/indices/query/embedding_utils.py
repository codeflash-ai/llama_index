"""Embedding utils for queries."""
import heapq
import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from llama_index.core.base.embeddings.base import similarity as default_similarity_fn
from llama_index.core.vector_stores.types import VectorStoreQueryMode


def get_top_k_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_fn: Optional[Callable[..., float]] = None,
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List] = None,
    similarity_cutoff: Optional[float] = None,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query."""
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))

    similarity_fn = similarity_fn or default_similarity_fn

    if not embeddings:
        return [], []

    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)

    # Vectorized batch similarity calculation if possible
    try:
        # If function matches the standard, do a batch similarity.
        if similarity_fn is default_similarity_fn:
            # Default similarity: cosine, dot, or euclidean
            # Let's check for cosine for fastest vectorized
            sim = np.dot(embeddings_np, query_embedding_np)
            norm = np.linalg.norm(query_embedding_np) * np.linalg.norm(
                embeddings_np, axis=1
            )
            similarities = sim / norm
        else:
            similarities = np.array(
                [similarity_fn(query_embedding_np, emb) for emb in embeddings_np]
            )
    except Exception:
        # Fallback: original per-embedding loop
        similarities = np.array(
            [similarity_fn(query_embedding_np, emb) for emb in embeddings_np]
        )

    similarity_heap: List[Tuple[float, Any]] = []
    for i, similarity in enumerate(similarities):
        if similarity_cutoff is None or similarity > similarity_cutoff:
            heapq.heappush(similarity_heap, (similarity, embedding_ids[i]))
            if similarity_top_k and len(similarity_heap) > similarity_top_k:
                heapq.heappop(similarity_heap)
    result_tups = sorted(similarity_heap, key=lambda x: x[0], reverse=True)

    result_similarities = [s for s, _ in result_tups]
    result_ids = [n for _, n in result_tups]

    return result_similarities, result_ids


def get_top_k_embeddings_learner(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List] = None,
    query_mode: VectorStoreQueryMode = VectorStoreQueryMode.SVM,
) -> Tuple[List[float], List]:
    """Get top embeddings by fitting a learner against query.

    Inspired by Karpathy's SVM demo:
    https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb

    Can fit SVM, linear regression, and more.

    """
    try:
        from sklearn import linear_model, svm
    except ImportError:
        raise ImportError("Please install scikit-learn to use this feature.")

    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
    query_embedding_np = np.array(query_embedding)
    embeddings_np = np.array(embeddings)
    # create dataset
    dataset_len = len(embeddings) + 1
    dataset = np.concatenate([query_embedding_np[None, ...], embeddings_np])
    y = np.zeros(dataset_len)
    y[0] = 1

    if query_mode == VectorStoreQueryMode.SVM:
        # train our SVM
        # TODO: make params configurable
        clf = svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
    elif query_mode == VectorStoreQueryMode.LINEAR_REGRESSION:
        clf = linear_model.LinearRegression()
    elif query_mode == VectorStoreQueryMode.LOGISTIC_REGRESSION:
        clf = linear_model.LogisticRegression(class_weight="balanced")
    else:
        raise ValueError(f"Unknown query mode: {query_mode}")

    clf.fit(dataset, y)  # train

    # infer on whatever data you wish, e.g. the original data
    similarities = clf.decision_function(dataset[1:])
    sorted_ix = np.argsort(-similarities)
    top_sorted_ix = (
        sorted_ix[:similarity_top_k] if similarity_top_k is not None else sorted_ix
    )

    result_similarities = similarities[top_sorted_ix]
    result_ids = [embedding_ids[ix] for ix in top_sorted_ix]

    return result_similarities, result_ids


def get_top_k_mmr_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_fn: Optional[Callable[..., float]] = None,
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List] = None,
    similarity_cutoff: Optional[float] = None,
    mmr_threshold: Optional[float] = None,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query,
    discount by their similarity to previous results.

    A mmr_threshold of 0 will strongly avoid similarity to previous results.
    A mmr_threshold of 1 will check similarity the query and ignore previous results.

    """
    threshold = mmr_threshold if mmr_threshold is not None else 0.5
    similarity_fn = similarity_fn or default_similarity_fn

    if embedding_ids is None or not embedding_ids:
        embedding_ids = list(range(len(embeddings)))
    if not embeddings:
        return [], []
    full_embed_map = dict(zip(embedding_ids, range(len(embedding_ids))))
    embed_map = full_embed_map.copy()
    embed_similarity = {}
    score: float = -math.inf
    high_score_id = None

    # Vectorize similarities if using default similarity and possible
    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)
    use_vectorized = False
    if similarity_fn is default_similarity_fn:
        # Try vectorized cosine similarity (see base.py default logic)
        try:
            sim_values = np.dot(embeddings_np, query_embedding_np)
            norm = np.linalg.norm(query_embedding_np) * np.linalg.norm(
                embeddings_np, axis=1
            )
            sim_values = sim_values / norm
            use_vectorized = True
        except Exception:
            use_vectorized = False

    for i, emb_id in enumerate(embedding_ids):
        if use_vectorized:
            similarity = sim_values[i]
        else:
            similarity = similarity_fn(query_embedding, embeddings[i])
        embed_similarity[emb_id] = similarity
        candidate_score = similarity * threshold
        if candidate_score > score:
            high_score_id = emb_id
            score = candidate_score

    results: List[Tuple[Any, Any]] = []

    embedding_length = len(embeddings or [])
    similarity_top_k_count = similarity_top_k or embedding_length
    while (
        len(results) < min(similarity_top_k_count, embedding_length)
        and high_score_id is not None
    ):
        # Calculate the similarity score the for the leading one.
        results.append((score, high_score_id))

        # Reset so a new high scoring result can be found
        del embed_map[high_score_id]
        recent_embedding_id = high_score_id
        score = -math.inf

        # Iterate through results to find high score
        high_score_id = None

        for embed_id in embed_map:
            idx = full_embed_map[embed_id]
            recent_idx = full_embed_map[recent_embedding_id]
            if use_vectorized:
                overlap_with_recent = np.dot(
                    embeddings_np[idx], embeddings_np[recent_idx]
                ) / (
                    np.linalg.norm(embeddings_np[idx])
                    * np.linalg.norm(embeddings_np[recent_idx])
                )
            else:
                overlap_with_recent = similarity_fn(
                    embeddings[idx], embeddings[recent_idx]
                )
            candidate_score = threshold * embed_similarity[embed_id] - (
                (1 - threshold) * overlap_with_recent
            )
            if candidate_score > score:
                score = candidate_score
                high_score_id = embed_id

    result_similarities = [s for s, _ in results]
    result_ids = [n for _, n in results]

    return result_similarities, result_ids
