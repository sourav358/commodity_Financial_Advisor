"""End-to-end tests for LLMService's deterministic HITL approval workflow.

Copy this file to tests/test_llm_approval_workflow.py in the project.
Run: poetry run pytest tests/test_llm_approval_workflow.py -q
"""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from commodity_finance_assistance.llm.service import LLMService


CONVERSATION_ID = "hitl-workflow-test-conversation"

VALID_TRADE_DETAILS = """\
Commodity: GOLD
Instrument: MCX Gold Futures
Direction: BUY
Quantity: 2
Unit: lots
Price: INR 142000
Trade date: 2026-11-01
Delivery: November 2026
Exchange: MCX
Counterparty: ABC Metals
Legal entity: Example Trading Pvt Ltd
Book: COMMODITIES
Portfolio: INDIA
"""


class FakeApprovalDraftStore:
    """In-memory stand-in for ApprovalDraftStore; keeps tests isolated."""

    def __init__(self):
        self.drafts = {}
        self.key_counter = 0

    def _new_key(self):
        self.key_counter += 1
        return f"test-idempotency-key-{self.key_counter}"

    def create(self, conversation_id, action_type):
        self.drafts[conversation_id] = {
            "conversation_id": conversation_id,
            "action_type": action_type,
            "status": "collecting",
            "fields": {},
            "idempotency_key": self._new_key(),
        }

    def get(self, conversation_id):
        draft = self.drafts.get(conversation_id)
        return deepcopy(draft) if draft is not None else None

    def update(self, conversation_id, fields, status):
        self.drafts[conversation_id]["fields"] = deepcopy(fields)
        self.drafts[conversation_id]["status"] = status

    def set_status(self, conversation_id, status):
        self.drafts[conversation_id]["status"] = status

    def reset(self, conversation_id, action_type):
        self.create(conversation_id, action_type)


class FakeMCPClient:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "request_id": "approval-request-123",
            "status": "PENDING",
        }
        self.error = error
        self.calls = []

    async def execute_tool(self, *, name, arguments):
        self.calls.append((name, deepcopy(arguments)))
        if self.error:
            raise self.error
        return json.dumps(self.response)


class FakeMCPContext:
    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture

def workflow():
    drafts = FakeApprovalDraftStore()
    memory = MagicMock()
    llm_client = MagicMock()
    with patch(
        "commodity_finance_assistance.llm.service.ApprovalDraftStore",
        return_value=drafts,
    ):
        service = LLMService(client=llm_client, memory=memory)
    return service, drafts, memory, llm_client


def send_details(service, details=VALID_TRADE_DETAILS):
    service.reply(
        CONVERSATION_ID,
        "Please prepare a trade execution request for human approval.",
    )
    return service.reply(CONVERSATION_ID, details)


def test_approval_request_starts_draft_and_asks_for_missing_fields(workflow):
    service, drafts, _, llm_client = workflow

    answer = service.reply(
        CONVERSATION_ID,
        "Please prepare a trade execution request for human approval.",
    )

    draft = drafts.get(CONVERSATION_ID)
    assert draft["status"] == "collecting"
    assert "Still required" in answer
    assert "Commodity" in answer
    llm_client.chat.completions.create.assert_not_called()


def test_complete_trade_details_require_explicit_confirmation(workflow):
    service, drafts, _, llm_client = workflow

    answer = send_details(service)

    assert drafts.get(CONVERSATION_ID)["status"] == "awaiting_confirmation"
    assert "Reply CONFIRM" in answer
    assert "No trade has been executed" in answer
    llm_client.chat.completions.create.assert_not_called()


def test_confirmation_before_all_required_fields_does_not_submit(workflow):
    service, drafts, _, _ = workflow
    service.reply(CONVERSATION_ID, "Please submit this trade for human approval.")

    answer = service.reply(CONVERSATION_ID, "CONFIRM")

    assert drafts.get(CONVERSATION_ID)["status"] == "collecting"
    assert "not ready" in answer


def test_invalid_trade_details_are_rejected_before_confirmation(workflow):
    service, drafts, _, _ = workflow
    invalid = VALID_TRADE_DETAILS.replace("Direction: BUY", "Direction: HOLD")

    answer = send_details(service, invalid)

    assert drafts.get(CONVERSATION_ID)["status"] == "collecting"
    assert "Direction must be BUY or SELL" in answer


def test_cancellation_discards_draft_without_mcp_submission(workflow, monkeypatch):
    service, drafts, _, _ = workflow
    service.reply(CONVERSATION_ID, "Please request human approval for a trade.")
    client = FakeMCPClient()
    monkeypatch.setattr(
        "commodity_finance_assistance.mcp.client.connect_mcp_server",
        lambda server_name: FakeMCPContext(client),
    )

    answer = service.reply(CONVERSATION_ID, "CANCEL")

    assert drafts.get(CONVERSATION_ID)["status"] == "cancelled"
    assert "cancelled" in answer.lower()
    assert client.calls == []


def test_confirm_submits_once_and_marks_request_pending(workflow, monkeypatch):
    service, drafts, _, _ = workflow
    proposal = send_details(service)
    assert "Reply CONFIRM" in proposal
    client = FakeMCPClient()
    monkeypatch.setattr(
        "commodity_finance_assistance.mcp.client.connect_mcp_server",
        lambda server_name: FakeMCPContext(client),
    )

    answer = service.reply(CONVERSATION_ID, "CONFIRM")

    assert drafts.get(CONVERSATION_ID)["status"] == "submitted"
    assert len(client.calls) == 1
    tool_name, arguments = client.calls[0]
    assert tool_name == "request_human_approval"
    payload = json.loads(arguments["payload_json"])
    assert payload["commodity"] == "GOLD"
    assert payload["direction"] == "BUY"
    assert arguments["idempotency_key"] == drafts.get(CONVERSATION_ID)["idempotency_key"]
    assert "approval-request-123" in answer
    assert "No trade has been executed" in answer

    duplicate_answer = service.reply(CONVERSATION_ID, "CONFIRM")
    assert len(client.calls) == 1
    assert "already has a submitted approval request" in duplicate_answer


def test_mcp_timeout_marks_submission_uncertain_and_retry_reuses_key(
    workflow, monkeypatch
):
    service, drafts, _, _ = workflow
    send_details(service)
    failed_client = FakeMCPClient(error=TimeoutError("simulated timeout"))
    monkeypatch.setattr(
        "commodity_finance_assistance.mcp.client.connect_mcp_server",
        lambda server_name: FakeMCPContext(failed_client),
    )

    uncertain_answer = service.reply(CONVERSATION_ID, "CONFIRM")
    assert drafts.get(CONVERSATION_ID)["status"] == "submission_uncertain"
    assert "outcome could not be confirmed" in uncertain_answer
    original_key = drafts.get(CONVERSATION_ID)["idempotency_key"]

    retry_client = FakeMCPClient()
    monkeypatch.setattr(
        "commodity_finance_assistance.mcp.client.connect_mcp_server",
        lambda server_name: FakeMCPContext(retry_client),
    )
    success_answer = service.reply(CONVERSATION_ID, "RETRY SUBMISSION")

    assert drafts.get(CONVERSATION_ID)["status"] == "submitted"
    assert retry_client.calls[0][1]["idempotency_key"] == original_key
    assert "No trade has been executed" in success_answer


def test_mcp_error_response_returns_draft_to_confirmation_state(
    workflow, monkeypatch
):
    service, drafts, _, _ = workflow
    send_details(service)
    client = FakeMCPClient(response={"error": "review service unavailable"})
    monkeypatch.setattr(
        "commodity_finance_assistance.mcp.client.connect_mcp_server",
        lambda server_name: FakeMCPContext(client),
    )

    answer = service.reply(CONVERSATION_ID, "CONFIRM")

    assert drafts.get(CONVERSATION_ID)["status"] == "awaiting_confirmation"
    assert "approval request was not created" in answer
