
"""Evaluation utilities for measuring RAG retrieval quality."""

from dataclasses import dataclass
from pathlib import PurePosixPath
from statistics import mean
from typing import Any, Sequence


@dataclass(frozen=True)
class RetrievalTestCase:
    """A question and the document sources expected to answer it."""

    question: str
    relevant_sources: frozenset[str]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question cannot be empty")

        if not self.relevant_sources:
            raise ValueError("relevant_sources cannot be empty")


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Evaluation metrics for one retrieval question."""

    question: str
    precision_at_k: float
    precision_at_retrieved: float
    recall_at_k: float
    reciprocal_rank: float
    hit_rate_at_k: float
    retrieved_sources: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregated evaluation results across multiple questions."""

    total_questions: int
    k: int
    precision_at_k: float
    precision_at_retrieved: float
    recall_at_k: float
    mrr: float
    hit_rate_at_k: float
    case_results: tuple[RetrievalCaseResult, ...]


def normalize_source(source: str) -> str:
    """Normalize source names for consistent document matching.

    Supports Windows paths, Unix paths, and document names with or
    without supported file extensions.
    """
    normalized = str(source).replace("\\", "/").strip()

    if not normalized:
        return ""

    filename = PurePosixPath(normalized).name.casefold()

    for extension in (".md", ".txt", ".pdf"):
        if filename.endswith(extension):
            filename = filename[: -len(extension)]
            break

    return filename


def evaluate_single_case(
    results: Sequence[dict[str, Any]],
    test_case: RetrievalTestCase,
    k: int,
) -> RetrievalCaseResult:
    """Calculate retrieval metrics for a single evaluation question."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    expected = {
        normalize_source(source)
        for source in test_case.relevant_sources
    }

    retrieved = [
        str(result.get("source", ""))
        for result in results[:k]
    ]

    normalized_retrieved = [
        normalize_source(source)
        for source in retrieved
    ]

    relevant_flags = [
        source in expected
        for source in normalized_retrieved
    ]

    relevant_count = sum(relevant_flags)
    retrieved_count = len(retrieved)

    # Standard Precision@K: relevant results divided by requested K.
    precision = relevant_count / k

    # Precision among the results actually returned.
    precision_at_retrieved = (
        relevant_count / retrieved_count
        if retrieved_count > 0
        else 0.0
    )

    # Recall@K counts each relevant document only once.
    retrieved_relevant = {
        source
        for source, is_relevant in zip(
            normalized_retrieved,
            relevant_flags,
        )
        if is_relevant
    }

    recall = (
        len(retrieved_relevant) / len(expected)
        if expected
        else 0.0
    )

    # Reciprocal rank of the first relevant result.
    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    hit_rate = 1.0 if relevant_count > 0 else 0.0

    return RetrievalCaseResult(
        question=test_case.question,
        precision_at_k=precision,
        precision_at_retrieved=precision_at_retrieved,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        hit_rate_at_k=hit_rate,
        retrieved_sources=tuple(retrieved),
    )


class RAGEvaluator:
    """Evaluate a RAG query service against a retrieval test dataset."""

    def __init__(
        self,
        query_service: Any,
        k: int = 5,
        min_score: float | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")

        self.query_service = query_service
        self.k = k
        self.min_score = min_score

    def evaluate(
        self,
        test_cases: Sequence[RetrievalTestCase],
    ) -> RetrievalEvaluationReport:
        """Run retrieval evaluation and aggregate the metrics."""
        if not test_cases:
            raise ValueError("test_cases cannot be empty")

        case_results: list[RetrievalCaseResult] = []

        for test_case in test_cases:
            results = self.query_service.query(
                test_case.question,
                top_k=self.k,
                min_score=self.min_score,
            )

            case_result = evaluate_single_case(
                results=results,
                test_case=test_case,
                k=self.k,
            )

            case_results.append(case_result)

        return RetrievalEvaluationReport(
            total_questions=len(case_results),
            k=self.k,
            precision_at_k=mean(
                result.precision_at_k for result in case_results
            ),
            precision_at_retrieved=mean(
                result.precision_at_retrieved
                for result in case_results
            ),
            recall_at_k=mean(
                result.recall_at_k for result in case_results
            ),
            mrr=mean(
                result.reciprocal_rank for result in case_results
            ),
            hit_rate_at_k=mean(
                result.hit_rate_at_k for result in case_results
            ),
            case_results=tuple(case_results),
        )
