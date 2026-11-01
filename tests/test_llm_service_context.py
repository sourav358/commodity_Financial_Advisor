
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

import commodity_finance_assistance.llm.service as service_module
from commodity_finance_assistance.llm.service import LLMService


def make_tool_definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool: {name}",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


class FakeMCPClient:
    def __init__(self):
        self.definitions = [
            make_tool_definition("get_mcx_quote"),
            make_tool_definition("calculate_mcx_contract_value"),
            make_tool_definition("request_human_approval"),
        ]
        self.execute_tool = MagicMock()

    async def get_openai_tool_definitions(self):
        return self.definitions


@pytest.fixture
def setup_service(monkeypatch):
    memory = MagicMock()
    memory.get_messages.return_value = []

    llm_client = MagicMock()
    llm_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Test response.",
                    tool_calls=None,
                )
            )
        ]
    )

    monkeypatch.setattr(
        service_module,
        "get_model_config",
        lambda role: {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    )

    monkeypatch.setattr(
        service_module,
        "load_config",
        lambda: {
            "memory": {
                "conversation": {
                    "max_messages": 20,
                }
            }
        },
    )

    fake_mcp = FakeMCPClient()

    @asynccontextmanager
    async def fake_connect_mcp_server(server_name="commodity"):
        assert server_name == "commodity"
        yield fake_mcp

    monkeypatch.setattr(
        service_module,
        "connect_mcp_server",
        fake_connect_mcp_server,
    )

    service = LLMService(
        client=llm_client,
        memory=memory,
    )

    return SimpleNamespace(
        service=service,
        memory=memory,
        client=llm_client,
        mcp=fake_mcp,
    )


def test_trader_mode_instruction_reaches_llm(setup_service):
    service = setup_service.service
    client = setup_service.client

    conversation_id = f"trader-test-{uuid4()}"

    trader_instruction = (
        "TRADER MODE: Focus on commodity quotes, "
        "contract details, expiry, exposure and risk. "
        "Never invent prices or lot sizes."
    )

    answer = service.reply(
        conversation_id,
        "Explain the risk exposure of a gold position.",
        mode_instruction=trader_instruction,
    )

    assert answer == "Test response."

    messages = client.chat.completions.create.call_args.kwargs[
        "messages"
    ]

    system_messages = [
        message["content"]
        for message in messages
        if message["role"] == "system"
    ]

    assert any(
        trader_instruction in message
        for message in system_messages
    )


def test_rag_context_is_injected_into_llm_prompt(
    setup_service,
    monkeypatch,
):
    memory = setup_service.memory
    client = setup_service.client

    rag_query = Mock()
    rag_query.query.return_value = [
        {
            "content": "Gold futures contract reference information.",
            "source": "gold_contract_guide.pdf",
        }
    ]

    formatter = Mock()
    formatter.format_context.return_value = (
        "SOURCE: gold_contract_guide.pdf\n"
        "Gold futures contract reference information."
    )

    service = LLMService(
        client=client,
        memory=memory,
        rag_query_service=rag_query,
        rag_context_formatter=formatter,
        rag_top_k=3,
        rag_min_score=0.25,
    )

    conversation_id = f"rag-test-{uuid4()}"

    answer = service.reply(
        conversation_id,
        "Explain gold futures contract specifications.",
    )

    assert answer == "Test response."

    rag_query.query.assert_called_once_with(
        "Explain gold futures contract specifications.",
        top_k=3,
        min_score=0.25,
    )

    formatter.format_context.assert_called_once()

    messages = client.chat.completions.create.call_args.kwargs[
        "messages"
    ]

    rag_messages = [
        message["content"]
        for message in messages
        if message["role"] == "system"
        and "Retrieved reference context follows" in message["content"]
    ]

    assert len(rag_messages) == 1
    assert "gold_contract_guide.pdf" in rag_messages[0]
    assert "Gold futures contract reference information." in rag_messages[0]


def test_skill_selection_limits_tools_exposed_to_llm(setup_service):
    service = setup_service.service
    client = setup_service.client

    conversation_id = f"skill-test-{uuid4()}"

    answer = service.reply(
        conversation_id,
        "What is the current GOLD quote?",
    )

    assert answer == "Test response."

    request = client.chat.completions.create.call_args.kwargs

    exposed_tools = {
        item["function"]["name"]
        for item in request["tools"]
    }

    assert exposed_tools == {"get_mcx_quote"}

    assert "calculate_mcx_contract_value" not in exposed_tools
    assert "request_human_approval" not in exposed_tools


def test_general_question_does_not_expose_mcp_tools(setup_service):
    service = setup_service.service
    client = setup_service.client

    conversation_id = f"general-test-{uuid4()}"

    answer = service.reply(
        conversation_id,
        "Explain the basic concept of financial diversification.",
    )

    assert answer == "Test response."

    request = client.chat.completions.create.call_args.kwargs

    assert "tools" not in request
    assert "tool_choice" not in request
