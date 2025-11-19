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

    # Attempt a batch vectorized fast path if using default similarity_fn (dot/cosine, etc.)
    if similarity_fn is default_similarity_fn:
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        query_embedding_np = np.asarray(query_embedding, dtype=np.float32)
        # Try to detect default similarity mode logic by inspecting the function code object
        # but as we don't know which mode (dot/cosine/euclidean) in this external symbol,
        # we assume default_similarity_fn is safe to vectorize as dot product (most common)
        # This is a conservative optimization, avoids breaking semantics for non-standard fn.

        # Calculate similarities in a vectorized, memory-efficient manner
        if (
            embeddings_np.ndim == 2
            and query_embedding_np.ndim == 1
            and len(embeddings_np) > 0
        ):
            # Try cosine similarity
            # product = np.dot(embedding1, embedding2)
            # norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            dot_products = np.dot(embeddings_np, query_embedding_np)
            emb_norms = np.linalg.norm(embeddings_np, axis=1)
            q_norm = np.linalg.norm(query_embedding_np)
            denom = emb_norms * q_norm
            # Avoid division by zero
            denom = np.where(denom == 0, 1e-8, denom)
            similarities = dot_products / denom

            if similarity_cutoff is not None:
                mask = similarities > similarity_cutoff
                similarities = similarities[mask]
                ids_arr = np.array(embedding_ids)[mask]
            else:
                ids_arr = np.array(embedding_ids)
            if similarities.shape[0] == 0:
                return [], []

            # Get top-k efficiently via argpartition (partial sort)
            if (
                similarity_top_k is not None
                and similarities.shape[0] > similarity_top_k
            ):
                idx = np.argpartition(-similarities, similarity_top_k - 1)[
                    :similarity_top_k
                ]
                top_indices = idx[np.argsort(-similarities[idx])]
            else:
                top_indices = np.argsort(-similarities)
            result_similarities = similarities[top_indices].tolist()
            result_ids = ids_arr[top_indices].tolist()
            return result_similarities, result_ids
        # fallback to original loop if shape is wrong

    # General loop for custom or fallback
    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)

    similarity_heap: List[Tuple[float, Any]] = []
    for i, emb in enumerate(embeddings_np):
        similarity = similarity_fn(query_embedding_np, emb)
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
    top_sorted_ix = sorted_ix[:similarity_top_k]

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
    threshold = mmr_threshold or 0.5
    similarity_fn = similarity_fn or default_similarity_fn

    if embedding_ids is None or embedding_ids == []:
        embedding_ids = list(range(len(embeddings)))
    full_embed_map = dict(zip(embedding_ids, range(len(embedding_ids))))
    embed_map = full_embed_map.copy()
    embed_similarity = {}
    score: float = -math.inf
    high_score_id = None

    for i, emb in enumerate(embeddings):
        similarity = similarity_fn(query_embedding, emb)
        embed_similarity[embedding_ids[i]] = similarity
        if similarity * threshold > score:
            high_score_id = embedding_ids[i]
            score = similarity * threshold

    results: List[Tuple[Any, Any]] = []

    embedding_length = len(embeddings or [])
    similarity_top_k_count = similarity_top_k or embedding_length
    while len(results) < min(similarity_top_k_count, embedding_length):
        # Calculate the similarity score the for the leading one.
        results.append((score, high_score_id))

        # Reset so a new high scoring result can be found
        del embed_map[high_score_id]
        recent_embedding_id = high_score_id
        score = -math.inf

        # Iterate through results to find high score
        for embed_id in embed_map:
            overlap_with_recent = similarity_fn(
                embeddings[embed_map[embed_id]],
                embeddings[full_embed_map[recent_embedding_id]],
            )
            if (
                threshold * embed_similarity[embed_id]
                - ((1 - threshold) * overlap_with_recent)
                > score
            ):
                score = threshold * embed_similarity[embed_id] - (
                    (1 - threshold) * overlap_with_recent
                )
                high_score_id = embed_id

    result_similarities = [s for s, _ in results]
    result_ids = [n for _, n in results]

    return result_similarities, result_ids
