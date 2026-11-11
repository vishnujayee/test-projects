"""Metrics. Part of the evaluator, frozen at the end of Phase 1.

recall@k : agreement with exact float32 search (ground truth, hard rule 4).
nDCG@k   : agreement with human relevance labels (qrels), position-discounted.
           Linear gain (gain = relevance grade) and log2(rank+1) discount, which is what
           trec_eval / pytrec_eval use, so numbers are comparable to BEIR papers.
           Only queries with at least one relevant document count.

Identical ids: in some BEIR datasets (ArguAna) most queries are themselves documents in the
corpus, with the same id. Retrieving "yourself" is meaningless, and BEIR removes such hits
(ignore_identical_ids=True). We do the same, so callers should pass at least k+1 results.
"""

import math

import numpy as np

from trivec.quantizers.base import IntArray

Qrels = dict[str, dict[str, int]]


def recall_per_query(found: IntArray, truth: IntArray) -> list[float]:
    """For each query, |found ∩ truth| / k, where k = truth.shape[1]."""
    if found.shape[0] != truth.shape[0]:
        raise ValueError("found and truth must have one row per query")
    k = truth.shape[1]
    return [
        len(set(f[:k].tolist()) & set(t.tolist())) / k for f, t in zip(found, truth, strict=True)
    ]


def recall_at_k(found: IntArray, truth: IntArray) -> float:
    """Mean of recall_per_query."""
    per_query = recall_per_query(found, truth)
    return float(np.mean(per_query)) if per_query else 0.0


def dcg(grades: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_per_query(
    ranked: IntArray,
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: Qrels,
    k: int,
    *,
    ignore_identical_ids: bool = True,
) -> list[float | None]:
    """One value per query; None for a query with no relevant document (it does not count)."""
    if ranked.shape[0] != len(query_ids):
        raise ValueError("ranked must have one row per query id")
    scores: list[float | None] = []
    for row, qid in zip(ranked, query_ids, strict=True):
        rels = qrels.get(qid, {})
        ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
        if not ideal:
            scores.append(None)
            continue
        doc_ids = [corpus_ids[int(i)] for i in row]
        if ignore_identical_ids:
            doc_ids = [d for d in doc_ids if d != qid]
        got = [rels.get(d, 0) for d in doc_ids[:k]]
        scores.append(dcg(got) / dcg(ideal))
    return scores


def ndcg_at_k(
    ranked: IntArray,
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: Qrels,
    k: int,
    *,
    ignore_identical_ids: bool = True,
) -> float:
    """Mean of ndcg_per_query over the queries that count. ranked holds row indices, best first."""
    per_query = ndcg_per_query(
        ranked, query_ids, corpus_ids, qrels, k, ignore_identical_ids=ignore_identical_ids
    )
    counted = [s for s in per_query if s is not None]
    return float(np.mean(counted)) if counted else 0.0
