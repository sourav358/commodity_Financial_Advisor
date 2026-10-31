
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import commodity_finance_assistance.hitl.store as store_module

from commodity_finance_assistance.hitl.models import (
    ApprovalExpiredError,
    ApprovalPayloadMismatchError,
    ApprovalPermissionError,
    ApprovalStatus,
    InvalidApprovalTransitionError,
)
from commodity_finance_assistance.hitl.service import HITLService
from commodity_finance_assistance.hitl.store import SQLiteApprovalStore


@pytest.fixture
def hitl_service(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "hitl_lifecycle.db")
    return HITLService(store)


def create_request(service, *, ttl_minutes=None):
    kwargs = {
        "action_type": "trade_execution_request",
        "requester_id": "trading-agent",
        "reason": "Human review required before trade execution.",
        "payload": {
            "commodity": "GOLD",
            "direction": "BUY",
            "quantity": 1,
        },
    }

    if ttl_minutes is not None:
        kwargs["ttl_minutes"] = ttl_minutes

    return service.request_approval(**kwargs)


def valid_payload():
    return {
        "commodity": "GOLD",
        "direction": "BUY",
        "quantity": 1,
    }


def test_expired_request_cannot_be_approved(hitl_service):
    request = create_request(hitl_service, ttl_minutes=1)

    future = datetime.now(timezone.utc) + timedelta(minutes=2)

    with patch.object(store_module, "_utc_now", return_value=future):
        with pytest.raises(ApprovalExpiredError):
            hitl_service.approve(
                request.request_id,
                approver_id="human-reviewer",
            )

    stored = hitl_service.get_request(request.request_id)

    assert stored.status is ApprovalStatus.EXPIRED
    assert len(hitl_service.list_pending()) == 0

    events = hitl_service.audit_events(request.request_id)
    assert [event["event_type"] for event in events] == [
        "CREATED",
        "EXPIRED",
    ]


def test_expired_request_cannot_be_consumed(hitl_service):
    request = create_request(hitl_service, ttl_minutes=1)

    future = datetime.now(timezone.utc) + timedelta(minutes=2)

    with patch.object(store_module, "_utc_now", return_value=future):
        with pytest.raises(ApprovalExpiredError):
            hitl_service.consume_approval(
                request.request_id,
                executor_id="execution-agent",
                expected_action_type="trade_execution_request",
                expected_payload=valid_payload(),
            )

    stored = hitl_service.get_request(request.request_id)
    assert stored.status is ApprovalStatus.EXPIRED

    events = hitl_service.audit_events(request.request_id)
    assert [event["event_type"] for event in events] == [
        "CREATED",
        "EXPIRED",
    ]


def test_rejected_request_cannot_be_approved_or_consumed(hitl_service):
    request = create_request(hitl_service)

    rejected = hitl_service.reject(
        request.request_id,
        approver_id="human-reviewer",
        reason="Risk threshold exceeded.",
    )

    assert rejected.status is ApprovalStatus.REJECTED

    with pytest.raises(InvalidApprovalTransitionError):
        hitl_service.approve(
            request.request_id,
            approver_id="another-reviewer",
        )

    with pytest.raises(InvalidApprovalTransitionError):
        hitl_service.consume_approval(
            request.request_id,
            executor_id="execution-agent",
            expected_action_type="trade_execution_request",
            expected_payload=valid_payload(),
        )

    assert hitl_service.get_request(
        request.request_id
    ).status is ApprovalStatus.REJECTED


def test_approved_request_cannot_receive_another_decision(hitl_service):
    request = create_request(hitl_service)

    hitl_service.approve(
        request.request_id,
        approver_id="human-reviewer",
        reason="Approved after risk review.",
    )

    with pytest.raises(InvalidApprovalTransitionError):
        hitl_service.approve(
            request.request_id,
            approver_id="another-reviewer",
        )

    with pytest.raises(InvalidApprovalTransitionError):
        hitl_service.reject(
            request.request_id,
            approver_id="another-reviewer",
            reason="Attempted second decision.",
        )

    events = hitl_service.audit_events(request.request_id)

    assert [event["event_type"] for event in events] == [
        "CREATED",
        "APPROVED",
    ]


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_requester_cannot_approve_or_reject_own_request(
    hitl_service,
    decision,
):
    request = create_request(hitl_service)

    with pytest.raises(ApprovalPermissionError):
        if decision == "approve":
            hitl_service.approve(
                request.request_id,
                approver_id="trading-agent",
            )
        else:
            hitl_service.reject(
                request.request_id,
                approver_id="trading-agent",
                reason="Self-approval attempt.",
            )

    assert hitl_service.get_request(
        request.request_id
    ).status is ApprovalStatus.PENDING

    events = hitl_service.audit_events(request.request_id)
    assert [event["event_type"] for event in events] == ["CREATED"]


def test_pending_request_cannot_be_consumed(hitl_service):
    request = create_request(hitl_service)

    with pytest.raises(InvalidApprovalTransitionError):
        hitl_service.consume_approval(
            request.request_id,
            executor_id="execution-agent",
            expected_action_type="trade_execution_request",
            expected_payload=valid_payload(),
        )

    assert hitl_service.get_request(
        request.request_id
    ).status is ApprovalStatus.PENDING


def test_approved_request_can_be_consumed_only_once(hitl_service):
    request = create_request(hitl_service)

    hitl_service.approve(
        request.request_id,
        approver_id="human-reviewer",
        reason="Approved.",
    )

    consumed = hitl_service.consume_approval(
        request.request_id,
        executor_id="execution-agent",
        expected_action_type="trade_execution_request",
        expected_payload=valid_payload(),
    )

    assert consumed.status is ApprovalStatus.CONSUMED
    assert consumed.consumed_at is not None

    with pytest.raises(InvalidApprovalTransitionError):
        hitl_service.consume_approval(
            request.request_id,
            executor_id="execution-agent",
            expected_action_type="trade_execution_request",
            expected_payload=valid_payload(),
        )

    assert hitl_service.get_request(
        request.request_id
    ).status is ApprovalStatus.CONSUMED


def test_consumption_rejects_modified_payload(hitl_service):
    request = create_request(hitl_service)

    hitl_service.approve(
        request.request_id,
        approver_id="human-reviewer",
        reason="Approved.",
    )

    modified_payload = {
        "commodity": "GOLD",
        "direction": "SELL",
        "quantity": 10,
    }

    with pytest.raises(ApprovalPayloadMismatchError):
        hitl_service.consume_approval(
            request.request_id,
            executor_id="execution-agent",
            expected_action_type="trade_execution_request",
            expected_payload=modified_payload,
        )

    # Failed payload validation must not consume the approval.
    assert hitl_service.get_request(
        request.request_id
    ).status is ApprovalStatus.APPROVED


def test_consumption_rejects_wrong_action_type(hitl_service):
    request = create_request(hitl_service)

    hitl_service.approve(
        request.request_id,
        approver_id="human-reviewer",
    )

    with pytest.raises(ValueError, match="action type"):
        hitl_service.consume_approval(
            request.request_id,
            executor_id="execution-agent",
            expected_action_type="risk_limit_change",
            expected_payload=valid_payload(),
        )

    assert hitl_service.get_request(
        request.request_id
    ).status is ApprovalStatus.APPROVED


def test_complete_lifecycle_has_exact_audit_history(hitl_service):
    request = create_request(hitl_service)

    hitl_service.approve(
        request.request_id,
        approver_id="human-reviewer",
        reason="Risk checks passed.",
    )

    hitl_service.consume_approval(
        request.request_id,
        executor_id="execution-agent",
        expected_action_type="trade_execution_request",
        expected_payload=valid_payload(),
    )

    events = hitl_service.audit_events(request.request_id)

    assert [event["event_type"] for event in events] == [
        "CREATED",
        "APPROVED",
        "CONSUMED",
    ]

    assert len(events) == 3

    assert events[0]["actor_id"] == "trading-agent"
    assert events[1]["actor_id"] == "human-reviewer"
    assert events[2]["actor_id"] == "execution-agent"
