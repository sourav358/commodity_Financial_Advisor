
import pytest

from commodity_finance_assistance.mcp.client import (
    connect_mcp_server,
)


@pytest.mark.asyncio
async def test_mcp_quote_tool_execution():
    async with connect_mcp_server("commodity") as client:
        result = await client.execute_tool(
            name="get_mcx_quote",
            arguments={"commodity": "gold"},
        )

        assert isinstance(result, str)
        assert result.strip()

        # The tool must return a result or a controlled error.
        assert "Traceback" not in result


@pytest.mark.asyncio
async def test_mcp_calculation_tool_execution():
    async with connect_mcp_server("commodity") as client:
        result = await client.execute_tool(
            name="calculate_mcx_contract_value",
            arguments={"commodity": "gold"},
        )

        assert isinstance(result, str)
        assert result.strip()
        assert "Traceback" not in result
