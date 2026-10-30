
import pytest

from commodity_finance_assistance.mcp.client import (
    connect_mcp_server,
)


@pytest.mark.asyncio
async def test_mcp_server_discovers_tools():
    async with connect_mcp_server("commodity") as client:
        tools = await client.discover_tools()

        tool_names = {tool.name for tool in tools}

        assert "get_mcx_quote" in tool_names
        assert "calculate_mcx_contract_value" in tool_names


@pytest.mark.asyncio
async def test_mcp_tools_convert_to_openai_definitions():
    async with connect_mcp_server("commodity") as client:
        definitions = await client.get_openai_tool_definitions()

        names = {
            item["function"]["name"]
            for item in definitions
        }

        assert "get_mcx_quote" in names
        assert "calculate_mcx_contract_value" in names
