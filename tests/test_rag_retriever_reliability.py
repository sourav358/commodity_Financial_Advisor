
import math
import pytest

from commodity_finance_assistance.rag.retriever import RAGRetriever


def test_cosine_similarity_identical_vectors():
    assert RAGRetriever.cosine_similarity(
        [1.0, 2.0],
        [1.0, 2.0],
    ) == pytest.approx(1.0)


def test_cosine_similarity_opposite_vectors():
    assert RAGRetriever.cosine_similarity(
        [1.0, 2.0],
        [-1.0, -2.0],
    ) == pytest.approx(-1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert RAGRetriever.cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_cosine_similarity_handles_large_finite_values():
    score = RAGRetriever.cosine_similarity(
        [1e308, 1e308],
        [1e308, 1e308],
    )

    assert math.isfinite(score)
    assert score == pytest.approx(1.0)


def test_vector_rejects_boolean_values():
    with pytest.raises(ValueError):
        RAGRetriever._validate_vector([1.0, True])


def test_vector_rejects_zero_vector():
    with pytest.raises(ValueError):
        RAGRetriever._validate_vector([0.0, 0.0])


def test_vector_rejects_non_finite_values():
    with pytest.raises(ValueError):
        RAGRetriever._validate_vector([1.0, float("inf")])


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        RAGRetriever.cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_cosine_similarity_rejects_nan():
    with pytest.raises(ValueError):
        RAGRetriever.cosine_similarity(
            [1.0, float("nan")],
            [1.0, 2.0],
        )
