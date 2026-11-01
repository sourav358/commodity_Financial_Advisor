
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from commodity_finance_assistance.mcp.client import MCPToolClient


def make_tool(
    name="get_mcx_quote",
    description="Get an MCX commodity quote",
    input_schema=None,
):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema or {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
            },
            "required": ["symbol"],
        },
    )


def make_result(
    *,
    structured_content=None,
    content=None,
    is_error=False,
):
    return SimpleNamespace(
        structuredContent=structured_content,
        content=content or [],
        isError=is_error,
    )


@pytest.fixture
def session():
    mock = AsyncMock()
    mock.list_tools.return_value = SimpleNamespace(
        tools=[
            make_tool(),
            make_tool(
                name="calculate_mcx_contract_value",
                description="Calculate MCX contract value",
            ),
        ]
    )
    return mock


@pytest.fixture
def client(session):
    return MCPToolClient(session)


@pytest.mark.asyncio
async def test_discover_tools_returns_server_tools(client, session):
    tools = await client.discover_tools()

    assert len(tools) == 2
    assert {tool.name for tool in tools} == {
        "get_mcx_quote",
        "calculate_mcx_contract_value",
    }

    session.list_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_discover_tools_uses_cache(client, session):
    first = await client.discover_tools()
    second = await client.discover_tools()

    assert first is second
    session.list_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_calls_session_initialize(session):
    client = MCPToolClient(session)

    await client.initialize()

    session.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_definitions_are_generated_correctly(client):
    definitions = await client.get_openai_tool_definitions()

    assert len(definitions) == 2

    first = definitions[0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "get_mcx_quote"
    assert first["function"]["description"] == "Get an MCX commodity quote"
    assert first["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_execute_tool_rejects_non_dictionary_arguments(
    client,
    session,
):
    result = await client.execute_tool(
        "get_mcx_quote",
        ["GOLD"],
    )

    parsed = json.loads(result)

    assert parsed["error"] == "Tool arguments must be an object."
    session.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_tool_rejects_unknown_tool(client, session):
    result = await client.execute_tool(
        "delete_trade",
        {},
    )

    parsed = json.loads(result)

    assert parsed["error"] == (
        "Tool is not available on this MCP server."
    )
    session.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_tool_returns_structured_content(client, session):
    expected = {
        "symbol": "GOLD",
        "price": 75000,
        "currency": "INR",
    }

    session.call_tool.return_value = make_result(
        structured_content=expected,
    )

    result = await client.execute_tool(
        "get_mcx_quote",
        {"symbol": "GOLD"},
    )

    assert json.loads(result) == expected
    session.call_tool.assert_awaited_once_with(
        "get_mcx_quote",
        arguments={"symbol": "GOLD"},
    )


@pytest.mark.asyncio
async def test_execute_tool_returns_text_content(client, session):
    session.call_tool.return_value = make_result(
        content=[
            SimpleNamespace(text="GOLD quote: 75000"),
            SimpleNamespace(text="Currency: INR"),
            SimpleNamespace(image="ignored"),
        ]
    )

    result = await client.execute_tool(
        "get_mcx_quote",
        {"symbol": "GOLD"},
    )

    assert result == "GOLD quote: 75000\nCurrency: INR"


@pytest.mark.asyncio
async def test_execute_tool_returns_fallback_when_no_content(
    client,
    session,
):
    session.call_tool.return_value = make_result()

    result = await client.execute_tool(
        "get_mcx_quote",
        {"symbol": "GOLD"},
    )

    assert json.loads(result) == {
        "result": "Tool executed without text output."
    }


@pytest.mark.asyncio
async def test_execute_tool_handles_server_exception(client, session):
    session.call_tool.side_effect = RuntimeError("Internal server detail")

    result = await client.execute_tool(
        "get_mcx_quote",
        {"symbol": "GOLD"},
    )

    parsed = json.loads(result)

    assert parsed["error"] == "MCP tool execution failed."
    assert parsed["error_type"] == "MCPToolExecutionError"
    assert parsed["retryable"] is True

    # Internal exception details must not leak into the response.
    assert "Internal server detail" not in result


@pytest.mark.asyncio
async def test_execute_tool_returns_error_when_server_reports_error(
    client,
    session,
):
    session.call_tool.return_value = make_result(
        content=[
            SimpleNamespace(text="Invalid commodity symbol")
        ],
        is_error=True,
    )

    result = await client.execute_tool(
        "get_mcx_quote",
        {"symbol": "GOLD"},
    )

    parsed = json.loads(result)

    assert parsed == {
        "error": "MCP tool reported an execution error.",
        "error_type": "MCPToolReportedError",
        "retryable": False,
    }

    assert "Invalid commodity symbol" not in result
