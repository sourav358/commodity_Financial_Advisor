
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from commodity_finance_assistance.tools.agent_tools import (
    execute_agent_tool,
)


logger = logging.getLogger(__name__)

mcp = FastMCP("Commodity Finance MCP Server")


@mcp.tool()
def get_mcx_quote(commodity: str) -> str:
    """
    Retrieve the latest imported end-of-day MCX futures quote.

    This is historical end-of-day data, not a live market quote.
    """
    return execute_agent_tool(
        name="get_mcx_quote",
        arguments={"commodity": commodity},
    )


@mcp.tool()
def calculate_mcx_contract_value(commodity: str) -> str:
    """
    Calculate MCX futures notional contract value.

    Calculation is permitted only when contract quotation and
    trading-unit metadata have been verified.
    """
    return execute_agent_tool(
        name="calculate_mcx_contract_value",
        arguments={"commodity": commodity},
    )


def main() -> None:
    """Start the MCP server using standard input/output transport."""
    logger.info("Starting Commodity Finance MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
