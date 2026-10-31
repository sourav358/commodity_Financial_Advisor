from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from mcp.server.fastmcp import FastMCP

from commodity_finance_assistance.guardrails.engine import (
    APPROVAL_TOOL,
    validate_tool_call,
)
from commodity_finance_assistance.hitl import build_hitl_service_from_config
from commodity_finance_assistance.tools.agent_tools import execute_agent_tool


logger = logging.getLogger(__name__)

mcp = FastMCP("Commodity Finance MCP Server")


@mcp.tool()
def get_mcx_quote(commodity: str) -> str:
    """Retrieve the latest imported end-of-day MCX futures quote.

    This is historical end-of-day data, not a live market quote.
    """
    return execute_agent_tool(
        name="get_mcx_quote",
        arguments={"commodity": commodity},
    )


@mcp.tool()
def calculate_mcx_contract_value(commodity: str) -> str:
    """Calculate MCX futures notional contract value.

    Calculation is permitted only when contract quotation and
    trading-unit metadata have been verified.
    """
    return execute_agent_tool(
        name="calculate_mcx_contract_value",
        arguments={"commodity": commodity},
    )


def _create_approval_request(
    action_type: str,
    reason: str,
    payload_json: str,
    idempotency_key: str | None = None,
) -> str:
    """Create a pending approval request without executing any action."""

    arguments: dict[str, Any] = {
        "action_type": action_type,
        "reason": reason,
        "payload_json": payload_json,
    }

    if idempotency_key is not None:
        arguments["idempotency_key"] = idempotency_key

    # Validate again at the server boundary.
    # Do not rely exclusively on LLMService validation.
    validate_tool_call(
        name=APPROVAL_TOOL,
        arguments=arguments,
        allowed_tools={APPROVAL_TOOL},
    )

    payload = json.loads(payload_json)

    hitl = build_hitl_service_from_config()

    approval_arguments = {
        "action_type": action_type,
        "requester_id": "agent",
        "reason": reason,
        "payload": payload,
    }

    if idempotency_key is not None:
        approval_arguments["idempotency_key"] = idempotency_key

    request = hitl.request_approval(**approval_arguments)
    logger.info(
        "HITL approval request processed | request_id=%s | "
        "action_type=%s | idempotency_key_present=%s",
        request.request_id,
        request.action_type,
        idempotency_key is not None,
    )

    return json.dumps(
        {
            "request_id": request.request_id,
            "action_type": request.action_type,
            "status": request.status.value,
            "expires_at": request.expires_at,
            "idempotency_key": getattr(request, "idempotency_key", None),
            "message": (
                "Approval request created or retrieved and is pending "
                "human review. No trade or external action has been executed."
            ),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def request_human_approval(
    action_type: str,
    reason: str,
    payload_json: str,
    idempotency_key: str | None = None,
) -> str:
    """Submit a proposed action for human review.

    payload_json must be a JSON-encoded object.

    Allowed action types:
    - risk_limit_change
    - portfolio_rebalance
    - trade_execution_request

    idempotency_key is optional. When supplied, retries with the same
    key and identical request content return the existing approval
    request instead of creating a duplicate.

    Reusing a key with different request content is rejected.

    This tool only creates or retrieves an approval record.
    It does not execute, place, amend, or cancel a trade.
    """

    try:
        return _create_approval_request(
            action_type=action_type,
            reason=reason,
            payload_json=payload_json,
            idempotency_key=idempotency_key,
        )

    except (ValueError, OSError, sqlite3.Error) as exc:
        logger.warning(
            "HITL request rejected | error_type=%s",
            type(exc).__name__,
        )

        return json.dumps(
            {
                "error": "The approval request could not be created.",
                "error_type": type(exc).__name__,
            },
            ensure_ascii=False,
        )

    except Exception:
        logger.exception(
            "Unexpected HITL approval request failure"
        )

        return json.dumps(
            {
                "error": "An unexpected internal error occurred.",
                "error_type": "InternalApprovalError",
            },
            ensure_ascii=False,
        )


def main() -> None:
    """Start the MCP server using standard input/output transport."""

    logger.info("Starting Commodity Finance MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()