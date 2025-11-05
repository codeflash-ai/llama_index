"""Notebook utils."""

from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.evaluation.retrieval.base import RetrievalEvalResult

DEFAULT_METRIC_KEYS = ["hit_rate", "mrr"]


def get_retrieval_results_df(
    names: List[str],
    results_arr: List[List[RetrievalEvalResult]],
    metric_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Display retrieval results."""
    metric_keys = metric_keys or DEFAULT_METRIC_KEYS

    avg_metrics_dict = defaultdict(list)
    for name, eval_results in zip(names, results_arr):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)
        results_df = pd.DataFrame(metric_dicts)

        for metric_key in metric_keys:
            if metric_key not in results_df.columns:
                raise ValueError(f"Metric key {metric_key} not in results_df")
            avg_metrics_dict[metric_key].append(results_df[metric_key].mean())

    return pd.DataFrame({"retrievers": names, **avg_metrics_dict})


def get_eval_results_df(
    names: List[str], results_arr: List[EvaluationResult], metric: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Organizes EvaluationResults into a deep dataframe and computes the mean
    score.

    result:
        result_df: pd.DataFrame representing all the evaluation results
        mean_df: pd.DataFrame of average scores groupby names
    """
    if len(names) != len(results_arr):
        raise ValueError("names and results_arr must have same length.")

    # Extract columns in a single pass using list comprehensions for faster performance
    qs = [res.query for res in results_arr]
    ss = [res.score for res in results_arr]
    fs = [res.feedback for res in results_arr]
    rs = [res.response for res in results_arr]
    cs = [res.contexts for res in results_arr]

    # Construct DataFrame directly
    data = {
        "rag": names,
        "query": qs,
        "answer": rs,
        "contexts": cs,
        "scores": ss,
        "feedbacks": fs,
    }
    deep_df = pd.DataFrame(data, copy=False)

    # Groupby and mean can be done faster via a Series for a single column, get in dict then make DataFrame and transpose
    mean_series = deep_df.groupby("rag", sort=False)["scores"].mean()
    mean_df = pd.DataFrame(mean_series).T

    if metric:
        mean_df.index = [f"mean_{metric}_score"]

    return deep_df, mean_df
