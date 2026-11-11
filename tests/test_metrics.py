"""Metrics must be right BEFORE they are frozen. Hand-worked examples."""

import math

import numpy as np
import pytest

from trivec.eval import dcg, ndcg_at_k, recall_at_k


def test_recall_hand_example() -> None:
    truth = np.array([[1, 2, 3, 4], [10, 11, 12, 13]])
    found = np.array([[4, 3, 9, 9], [10, 11, 12, 13]])  # 2 of 4, then 4 of 4
    assert recall_at_k(found, truth) == pytest.approx((0.5 + 1.0) / 2)


def test_recall_ignores_extra_columns_beyond_k() -> None:
    truth = np.array([[1, 2]])
    found = np.array([[1, 7, 2]])  # third column is beyond k=2 and must not count
    assert recall_at_k(found, truth) == pytest.approx(0.5)


def test_recall_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError):
        recall_at_k(np.array([[1]]), np.array([[1], [2]]))


def test_dcg_hand_example() -> None:
    # grades [1, 0, 2]: 1/log2(2) + 0/log2(3) + 2/log2(4) = 1 + 0 + 1 = 2
    assert dcg([1, 0, 2]) == pytest.approx(2.0)


def test_ndcg_hand_example() -> None:
    corpus_ids = ["a", "b", "x", "y"]
    qrels = {"q1": {"a": 2, "b": 1}}
    # ranking b, x, a  ->  DCG = 1/1 + 0 + 2/log2(4) = 2.0
    # ideal   a, b     -> IDCG = 2/1 + 1/log2(3) = 2.6309
    ranked = np.array([[1, 2, 0]])
    expected = 2.0 / (2.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(ranked, ["q1"], corpus_ids, qrels, k=3) == pytest.approx(expected)
    assert expected == pytest.approx(0.7602, abs=1e-4)


def test_ndcg_perfect_and_zero() -> None:
    corpus_ids = ["a", "b", "c"]
    qrels = {"q": {"a": 1}}
    assert ndcg_at_k(np.array([[0, 1, 2]]), ["q"], corpus_ids, qrels, k=3) == 1.0
    assert ndcg_at_k(np.array([[1, 2, 0]]), ["q"], corpus_ids, qrels, k=2) == 0.0


def test_ndcg_ignores_the_query_itself_like_beir() -> None:
    """ArguAna: a query is also a document with the same id. It must not take a result slot."""
    corpus_ids = ["q", "good", "bad"]
    qrels = {"q": {"good": 1}}
    ranked = np.array([[0, 1, 2]])  # the query's own document comes first, as it would in search
    assert ndcg_at_k(ranked, ["q"], corpus_ids, qrels, k=1) == 1.0
    assert ndcg_at_k(ranked, ["q"], corpus_ids, qrels, k=1, ignore_identical_ids=False) == 0.0


def test_ndcg_skips_queries_without_relevant_docs() -> None:
    corpus_ids = ["a", "b"]
    qrels = {"q_rel": {"a": 1}, "q_none": {"b": 0}}
    ranked = np.array([[0, 1], [0, 1]])
    # q_none has no positive grade -> excluded; q_rel is perfect -> 1.0
    assert ndcg_at_k(ranked, ["q_rel", "q_none"], corpus_ids, qrels, k=2) == 1.0
    assert ndcg_at_k(ranked[:1], ["unknown"], corpus_ids, qrels, k=2) == 0.0
