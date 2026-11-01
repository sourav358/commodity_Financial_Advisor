
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from commodity_finance_assistance.config_loader import load_config


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MCPToolClient:
    """Client for discovering and executing tools on an MCP server."""

    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self._tool_cache: list[Any] | None = None

    async def initialize(self) -> None:
        """Initialize the MCP session."""
        await self.session.initialize()
        logger.info("MCP session initialized")

    async def discover_tools(self) -> list[Any]:
        """Discover tools exposed by the connected MCP server."""
        if self._tool_cache is None:
            result = await self.session.list_tools()
            self._tool_cache = result.tools

            logger.info(
                "MCP tools discovered | count=%s | tools=%s",
                len(self._tool_cache),
                [tool.name for tool in self._tool_cache],
            )

        return self._tool_cache

    async def get_openai_tool_definitions(self) -> list[dict[str, Any]]:
        """Convert discovered MCP tools into OpenAI function definitions."""
        tools = await self.discover_tools()

        definitions = []

        for tool in tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })

        return definitions

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a discovered MCP tool."""
        if not isinstance(arguments, dict):
            return json.dumps({
                "error": "Tool arguments must be an object."
            })

        available_tools = await self.discover_tools()
        allowed_names = {tool.name for tool in available_tools}

        if name not in allowed_names:
            return json.dumps({
                "error": "Tool is not available on this MCP server."
            })

        try:
            result = await self.session.call_tool(
                name,
                arguments=arguments,
            )

            if getattr(result, "isError", False):
                logger.warning(
                    "MCP tool returned an error | tool=%s",
                    name,
                )

                return json.dumps({
                    "error": "MCP tool reported an execution error.",
                    "error_type": "MCPToolReportedError",
                    "retryable": False,
                })
            structured_content = getattr(
                result,
                "structuredContent",
                None,
            )

            if structured_content is not None:
                return json.dumps(structured_content, default=str)

            text_parts = [
                item.text
                for item in result.content
                if hasattr(item, "text")
            ]

            if text_parts:
                return "\n".join(text_parts)

            return json.dumps({
                "result": "Tool executed without text output."
            })

        except Exception:
            logger.exception(
                "MCP tool execution failed | tool=%s",
                name,
            )

            return json.dumps({
                "error": "MCP tool execution failed.",
                "error_type": "MCPToolExecutionError",
                "retryable": True,
            })


@asynccontextmanager
async def connect_mcp_server(
    server_name: str = "commodity",
) -> AsyncIterator[MCPToolClient]:
    """Launch and maintain an MCP server session."""

    config = load_config()

    mcp_config = config.get("mcp", {})

    if not mcp_config.get("enabled", False):
        raise RuntimeError("MCP integration is disabled in config.yml")

    servers = mcp_config.get("servers", {})
    server_config = servers.get(server_name)

    if not isinstance(server_config, dict):
        raise ValueError(
            f"MCP server configuration not found: {server_name}"
        )

    transport = server_config.get("transport", "stdio")

    if transport != "stdio":
        raise ValueError(
            f"Unsupported MCP transport: {transport}"
        )

    command = server_config.get("command")
    args = server_config.get("args", [])

    if not isinstance(command, str) or not command.strip():
        raise ValueError("MCP server command must be configured.")

    if not isinstance(args, list) or not all(
        isinstance(item, str) for item in args
    ):
        raise ValueError("MCP server args must be a list of strings.")

    parameters = StdioServerParameters(
        command=command,
        args=args,
        cwd=str(PROJECT_ROOT),
    )

    logger.info(
        "Connecting to MCP server | server=%s | command=%s",
        server_name,
        command,
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            client = MCPToolClient(session)
            await client.initialize()

            try:
                yield client
            finally:
                logger.info(
                    "MCP session closed | server=%s",
                    server_name,
                )
