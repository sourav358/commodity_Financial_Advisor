import pytest

from commodity_finance_assistance.hitl.models import (
    ApprovalPayloadMismatchError,
)
from commodity_finance_assistance.hitl.service import HITLService
from commodity_finance_assistance.hitl.store import SQLiteApprovalStore
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier


@pytest.fixture
def hitl_service(tmp_path):
    store = SQLiteApprovalStore(
        tmp_path / "hitl_idempotency.db"
    )
    return HITLService(store)


def submit_request(
    service,
    *,
    key="draft-key-123",
    reason="Human review requested.",
    payload=None,
    action_type="trade_execution_request",
    requester_id="agent",
):
    if payload is None:
        payload = {
            "commodity": "GOLD",
            "direction": "BUY",
            "quantity": 1,
        }

    return service.request_approval(
        action_type=action_type,
        requester_id=requester_id,
        reason=reason,
        payload=payload,
        idempotency_key=key,
    )


def test_identical_retry_returns_original_request(hitl_service):
    first = submit_request(hitl_service)

    # Simulate an uncertain response: the first request was saved,
    # but the caller did not receive its response.
    retry = submit_request(hitl_service)

    assert first.request_id == retry.request_id
    assert retry.status == first.status
    assert retry.idempotency_key == "draft-key-123"

    pending = hitl_service.list_pending()

    assert len(pending) == 1


def test_identical_retry_creates_only_one_audit_event(hitl_service):
    first = submit_request(hitl_service)

    retry = submit_request(hitl_service)

    assert first.request_id == retry.request_id

    events = hitl_service.audit_events(first.request_id)

    created_events = [
        event for event in events
        if event["event_type"] == "CREATED"
    ]

    assert len(created_events) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": "A different approval reason."},
        {
            "payload": {
                "commodity": "SILVER",
                "direction": "BUY",
                "quantity": 1,
            }
        },
        {"action_type": "risk_limit_change"},
        {"requester_id": "different-agent"},
    ],
)
def test_same_key_with_different_content_is_rejected(
    hitl_service,
    changes,
):
    submit_request(hitl_service)

    with pytest.raises(ApprovalPayloadMismatchError):
        submit_request(hitl_service, **changes)

    assert len(hitl_service.list_pending()) == 1


def test_different_keys_create_separate_requests(hitl_service):
    first = submit_request(
        hitl_service,
        key="draft-key-001",
    )

    second = submit_request(
        hitl_service,
        key="draft-key-002",
    )

    assert first.request_id != second.request_id
    assert len(hitl_service.list_pending()) == 2


def test_requests_without_idempotency_keys_remain_supported(
    hitl_service,
):
    first = submit_request(
        hitl_service,
        key=None,
    )

    second = submit_request(
        hitl_service,
        key=None,
    )

    assert first.request_id != second.request_id
    assert len(hitl_service.list_pending()) == 2


def test_retry_after_approval_returns_existing_approved_request(
    hitl_service,
):
    first = submit_request(hitl_service)

    hitl_service.approve(
        first.request_id,
        approver_id="human-reviewer",
        reason="Approved after review.",
    )

    retry = submit_request(hitl_service)

    assert retry.request_id == first.request_id
    assert retry.status.value == "APPROVED"

    assert len(hitl_service.list_pending()) == 0
def test_concurrent_identical_submissions_create_only_one_request(
    hitl_service,
):
    workers = 8
    barrier = Barrier(workers)

    def submit_concurrently():
        barrier.wait(timeout=10)
        return submit_request(
            hitl_service,
            key="concurrent-draft-key",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(submit_concurrently)
            for _ in range(workers)
        ]
        results = [future.result(timeout=30) for future in futures]

    # Every submission must resolve to the same approval request.
    request_ids = {request.request_id for request in results}
    assert len(request_ids) == 1

    original_request = results[0]

    # Only one pending approval should exist.
    pending_requests = hitl_service.list_pending()
    assert len(pending_requests) == 1
    assert pending_requests[0].request_id == original_request.request_id

    # Only one CREATED audit event should exist.
    events = hitl_service.audit_events(original_request.request_id)
    created_events = [
        event for event in events
        if event["event_type"] == "CREATED"
    ]
    assert len(created_events) == 1
def test_concurrent_conflicting_submissions_reject_one_request(
    hitl_service,
):
    workers = 2
    barrier = Barrier(workers)

    payloads = [
        {
            "commodity": "GOLD",
            "direction": "BUY",
            "quantity": 1,
        },
        {
            "commodity": "SILVER",
            "direction": "SELL",
            "quantity": 5,
        },
    ]

    def submit_concurrently(payload):
        barrier.wait(timeout=10)

        try:
            result = submit_request(
                hitl_service,
                key="shared-conflicting-key",
                payload=payload,
            )
            return ("success", result)

        except ApprovalPayloadMismatchError as exc:
            return ("conflict", exc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(submit_concurrently, payload)
            for payload in payloads
        ]
        results = [future.result(timeout=30) for future in futures]

    successful = [
        result for status, result in results
        if status == "success"
    ]
    conflicts = [
        result for status, result in results
        if status == "conflict"
    ]

    assert len(successful) == 1
    assert len(conflicts) == 1

    pending = hitl_service.list_pending()
    assert len(pending) == 1

    stored_request = pending[0]
    assert stored_request.idempotency_key == "shared-conflicting-key"

    events = hitl_service.audit_events(stored_request.request_id)
    created_events = [
        event for event in events
        if event["event_type"] == "CREATED"
    ]

    assert len(created_events) == 1