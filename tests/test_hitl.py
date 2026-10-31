from datetime import datetime, timedelta, timezone

import pytest

from commodity_finance_assistance.hitl import (
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalPayloadMismatchError,
    ApprovalPermissionError,
    ApprovalStatus,
    HITLService,
    InvalidApprovalTransitionError,
    SQLiteApprovalStore,
)


@pytest.fixture
def hitl(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "hitl_test.db")
    return HITLService(store), store


def create_request(service, **overrides):
    values = {
        "action_type": "risk_limit_change",
        "requester_id": "agent-01",
        "reason": "Proposed change exceeds the configured risk threshold.",
        "payload": {"commodity": "GOLD", "new_limit": 250000},
    }
    values.update(overrides)
    return service.request_approval(**values)


def test_create_request_is_pending_and_persisted(hitl):
    service, store = hitl
    request = create_request(service)
    loaded = store.get(request.request_id)
    assert loaded.status is ApprovalStatus.PENDING
    assert loaded.payload == {"commodity": "GOLD", "new_limit": 250000}
    assert len(store.events(request.request_id)) == 1


def test_requester_cannot_approve_own_request(hitl):
    service, _ = hitl
    request = create_request(service)
    with pytest.raises(ApprovalPermissionError):
        service.approve(request.request_id, "agent-01")


def test_human_can_approve_request(hitl):
    service, _ = hitl
    request = create_request(service)
    approved = service.approve(request.request_id, "risk-manager-01", "Reviewed and accepted.")
    assert approved.status is ApprovalStatus.APPROVED
    assert approved.approver_id == "risk-manager-01"
    assert approved.decision_reason == "Reviewed and accepted."


def test_rejected_request_cannot_be_consumed(hitl):
    service, _ = hitl
    request = create_request(service)
    rejected = service.reject(request.request_id, "risk-manager-01", "Outside policy.")
    assert rejected.status is ApprovalStatus.REJECTED
    with pytest.raises(InvalidApprovalTransitionError):
        service.consume_approval(
            request.request_id,
            "executor-01",
            expected_action_type="risk_limit_change",
            expected_payload=request.payload,
        )


def test_approved_request_can_be_consumed_once(hitl):
    service, _ = hitl
    request = create_request(service)
    service.approve(request.request_id, "risk-manager-01")
    consumed = service.consume_approval(
        request.request_id,
        "executor-01",
        expected_action_type=request.action_type,
        expected_payload=request.payload,
    )
    assert consumed.status is ApprovalStatus.CONSUMED
    with pytest.raises(InvalidApprovalTransitionError):
        service.consume_approval(
            request.request_id,
            "executor-01",
            expected_action_type=request.action_type,
            expected_payload=request.payload,
        )


def test_payload_cannot_change_after_approval(hitl):
    service, _ = hitl
    request = create_request(service)
    service.approve(request.request_id, "risk-manager-01")
    with pytest.raises(ApprovalPayloadMismatchError):
        service.consume_approval(
            request.request_id,
            "executor-01",
            expected_action_type=request.action_type,
            expected_payload={"commodity": "GOLD", "new_limit": 999999},
        )


def test_action_type_must_match(hitl):
    service, _ = hitl
    request = create_request(service)
    service.approve(request.request_id, "risk-manager-01")
    with pytest.raises(ValueError, match="action type"):
        service.consume_approval(
            request.request_id,
            "executor-01",
            expected_action_type="trade_execution",
            expected_payload=request.payload,
        )


def test_expired_request_cannot_be_approved(hitl):
    service, store = hitl
    request = create_request(service, ttl_minutes=1)
    # Move expiry into the past to test expiry deterministically.
    with store._connect() as connection:
        connection.execute(
            "UPDATE hitl_approval_requests SET expires_at = ? WHERE request_id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), request.request_id),
        )
    with pytest.raises(ApprovalExpiredError):
        service.approve(request.request_id, "risk-manager-01")
    assert store.get(request.request_id).status is ApprovalStatus.EXPIRED


def test_expired_approved_request_cannot_be_consumed(hitl):
    service, store = hitl
    request = create_request(service)
    service.approve(request.request_id, "risk-manager-01")
    with store._connect() as connection:
        connection.execute(
            "UPDATE hitl_approval_requests SET expires_at = ? WHERE request_id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), request.request_id),
        )
    with pytest.raises(ApprovalExpiredError):
        service.consume_approval(
            request.request_id,
            "executor-01",
            expected_action_type=request.action_type,
            expected_payload=request.payload,
        )
    assert store.get(request.request_id).status is ApprovalStatus.EXPIRED


def test_unknown_request_raises(hitl):
    service, _ = hitl
    with pytest.raises(ApprovalNotFoundError):
        service.get_request("missing-request")


def test_invalid_payload_and_ttl_are_rejected(hitl):
    service, _ = hitl
    with pytest.raises(ValueError, match="JSON-serializable"):
        create_request(service, payload={"bad": object()})
    with pytest.raises(ValueError, match="ttl_minutes"):
        create_request(service, ttl_minutes=0)


def test_audit_log_records_decisions(hitl):
    service, _ = hitl
    request = create_request(service)
    service.approve(request.request_id, "risk-manager-01", "Approved after review")
    events = service.audit_events(request.request_id)
    assert [event["event_type"] for event in events] == ["CREATED", "APPROVED"]
    assert events[-1]["actor_id"] == "risk-manager-01"
def test_repeated_idempotency_key_returns_original_request(hitl):
    service, store = hitl
    key = "test-idempotency-key-001"

    first = create_request(service, idempotency_key=key)
    second = create_request(service, idempotency_key=key)

    assert first.request_id == second.request_id
    assert second.idempotency_key == key
    assert len(store.events(first.request_id)) == 1


def test_idempotency_key_rejects_different_payload(hitl):
    service, _ = hitl
    key = "test-idempotency-key-002"

    create_request(service, idempotency_key=key)

    with pytest.raises(ApprovalPayloadMismatchError):
        create_request(
            service,
            idempotency_key=key,
            payload={"commodity": "SILVER", "new_limit": 500000},
        )


def test_idempotency_key_rejects_different_action(hitl):
    service, _ = hitl
    key = "test-idempotency-key-003"

    create_request(service, idempotency_key=key)

    with pytest.raises(ApprovalPayloadMismatchError):
        create_request(
            service,
            idempotency_key=key,
            action_type="portfolio_rebalance",
        )


def test_different_idempotency_keys_create_separate_requests(hitl):
    service, _ = hitl

    first = create_request(service, idempotency_key="request-key-A")
    second = create_request(service, idempotency_key="request-key-B")

    assert first.request_id != second.request_id