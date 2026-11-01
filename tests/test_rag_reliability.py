from unittest.mock import Mock

import pytest

from commodity_finance_assistance.llm.service import LLMService


CONVERSATION_ID = "rag-reliability-test"


@pytest.fixture
def query_service():
    return Mock()


@pytest.fixture
def formatter():
    instance = Mock()
    instance.format_context.return_value = (
        "Commodity prices are influenced by supply and demand."
    )
    return instance


@pytest.fixture
def service(query_service, formatter):
    return LLMService(
        rag_query_service=query_service,
        rag_context_formatter=formatter,
    )


def test_successful_retrieval_returns_context_and_available_status(
    service,
    query_service,
    formatter,
):
    query_service.query.return_value = [
        {
            "content": "Commodity prices are influenced by supply and demand.",
            "source": "commodity_market_basics.md",
        }
    ]

    context, status = service._retrieve_rag_context_with_status(
        "What influences commodity prices?",
        CONVERSATION_ID,
    )

    assert status == "available"
    assert "supply and demand" in context

    query_service.query.assert_called_once_with(
        "What influences commodity prices?",
        top_k=5,
        min_score=None,
    )
    formatter.format_context.assert_called_once()


def test_empty_retrieval_returns_no_results_status(
    service,
    query_service,
):
    query_service.query.return_value = []

    context, status = service._retrieve_rag_context_with_status(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context is None
    assert status == "no_results"


def test_embedding_or_database_error_returns_failed_status(
    service,
    query_service,
):
    query_service.query.side_effect = RuntimeError(
        "Simulated retrieval failure"
    )

    context, status = service._retrieve_rag_context_with_status(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context is None
    assert status == "failed"


def test_malformed_query_result_returns_failed_status(
    service,
    query_service,
):
    query_service.query.return_value = {
        "content": "This is not a list."
    }

    context, status = service._retrieve_rag_context_with_status(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context is None
    assert status == "failed"


def test_formatter_error_returns_failed_status(
    service,
    query_service,
    formatter,
):
    query_service.query.return_value = [
        {"content": "Some content", "source": "example.md"}
    ]
    formatter.format_context.side_effect = ValueError(
        "Simulated formatting failure"
    )

    context, status = service._retrieve_rag_context_with_status(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context is None
    assert status == "failed"


def test_empty_formatted_context_returns_empty_context_status(
    service,
    query_service,
    formatter,
):
    query_service.query.return_value = [
        {"content": "Some content", "source": "example.md"}
    ]
    formatter.format_context.return_value = "   "

    context, status = service._retrieve_rag_context_with_status(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context is None
    assert status == "empty_context"


def test_disabled_rag_returns_disabled_status(formatter):
    service = LLMService(
        rag_query_service=None,
        rag_context_formatter=formatter,
    )

    context, status = service._retrieve_rag_context_with_status(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context is None
    assert status == "disabled"
    formatter.format_context.assert_not_called()


def test_legacy_retrieval_method_still_returns_context(
    service,
    query_service,
):
    query_service.query.return_value = [
        {"content": "Some content", "source": "example.md"}
    ]

    context = service._retrieve_rag_context(
        "Explain commodity markets",
        CONVERSATION_ID,
    )

    assert context == (
        "Commodity prices are influenced by supply and demand."
    )