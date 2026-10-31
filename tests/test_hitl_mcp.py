import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from commodity_finance_assistance.guardrails.engine import GuardrailViolation
from commodity_finance_assistance.mcp import server
from commodity_finance_assistance.hitl.models import ApprovalStatus


def test_mcp_tool_creates_pending_request_without_executing_action():
    fake_request = SimpleNamespace(
        request_id="request-123",
        action_type="trade_execution_request",
        status=ApprovalStatus.PENDING,
        expires_at="2030-01-01T00:00:00+00:00",
    )
    fake_hitl = Mock()
    fake_hitl.request_approval.return_value = fake_request

    with patch.object(server, "build_hitl_service_from_config", return_value=fake_hitl):
        result = json.loads(
            server._create_approval_request(
                action_type="trade_execution_request",
                reason="Human review requested for proposed trade.",
                payload_json='{"commodity":"GOLD","side":"BUY","quantity":1}',
            )
        )

    assert result["request_id"] == "request-123"
    assert result["status"] == "PENDING"
    assert "No trade or external action has been executed." in result["message"]
    fake_hitl.request_approval.assert_called_once_with(
        action_type="trade_execution_request",
        requester_id="agent",
        reason="Human review requested for proposed trade.",
        payload={"commodity": "GOLD", "side": "BUY", "quantity": 1},
    )


def test_mcp_tool_rejects_invalid_action_before_creating_request():
    with patch.object(server, "build_hitl_service_from_config") as build:
        with pytest.raises(GuardrailViolation):
            server._create_approval_request(
                action_type="execute_arbitrary_code",
                reason="Not permitted.",
                payload_json='{"command":"anything"}',
            )
    build.assert_not_called()
def test_mcp_tool_passes_idempotency_key_to_service():
    fake_request = SimpleNamespace(
        request_id="request-idempotent",
        action_type="trade_execution_request",
        status=ApprovalStatus.PENDING,
        expires_at="2030-01-01T00:00:00+00:00",
    )

    fake_hitl = Mock()
    fake_hitl.request_approval.return_value = fake_request

    with patch.object(
        server,
        "build_hitl_service_from_config",
        return_value=fake_hitl,
    ):
        result = json.loads(
            server._create_approval_request(
                action_type="trade_execution_request",
                reason="Human review requested.",
                payload_json='{"commodity":"GOLD","quantity":1}',
                idempotency_key="draft-key-123",
            )
        )

    assert result["request_id"] == "request-idempotent"

    fake_hitl.request_approval.assert_called_once_with(
        action_type="trade_execution_request",
        requester_id="agent",
        reason="Human review requested.",
        payload={"commodity": "GOLD", "quantity": 1},
        idempotency_key="draft-key-123",
    )