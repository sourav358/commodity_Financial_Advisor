
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from commodity_finance_assistance.hitl import (
    ApprovalStatus,
    HITLService,
    InvalidApprovalTransitionError,
    SQLiteApprovalStore,
)


@pytest.fixture
def hitl(tmp_path):
    store = SQLiteApprovalStore(tmp_path / "hitl_concurrency.db")
    return HITLService(store), store


def create_request(service, **overrides):
    values = {
        "action_type": "risk_limit_change",
        "requester_id": "agent-01",
        "reason": "Proposed change exceeds configured risk threshold.",
        "payload": {
            "commodity": "GOLD",
            "new_limit": 250000,
        },
    }
    values.update(overrides)
    return service.request_approval(**values)


def capture_result(function):
    try:
        return function()
    except Exception as exc:
        return exc


def test_concurrent_approval_vs_rejection(hitl):
    service, store = hitl
    request = create_request(service)

    barrier = Barrier(2)

    def approve():
        barrier.wait()
        return capture_result(
            lambda: service.approve(
                request.request_id,
                "risk-manager-01",
            )
        )

    def reject():
        barrier.wait()
        return capture_result(
            lambda: service.reject(
                request.request_id,
                "risk-manager-02",
                "Rejected during concurrent review.",
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(approve),
            executor.submit(reject),
        ]
        results = [future.result() for future in futures]

    successes = [
        result for result in results
        if not isinstance(result, Exception)
    ]
    failures = [
        result for result in results
        if isinstance(result, Exception)
    ]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(
        failures[0],
        InvalidApprovalTransitionError,
    )

    final_request = store.get(request.request_id)

    assert final_request.status in {
        ApprovalStatus.APPROVED,
        ApprovalStatus.REJECTED,
    }

    events = store.events(request.request_id)
    decision_events = [
        event for event in events
        if event["event_type"] in {"APPROVED", "REJECTED"}
    ]

    assert len(decision_events) == 1


def test_concurrent_consumption_succeeds_only_once(hitl):
    service, store = hitl
    request = create_request(service)

    service.approve(
        request.request_id,
        "risk-manager-01",
    )

    barrier = Barrier(2)

    def consume():
        barrier.wait()
        return capture_result(
            lambda: service.consume_approval(
                request.request_id,
                "executor-01",
                expected_action_type=request.action_type,
                expected_payload=request.payload,
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(consume),
            executor.submit(consume),
        ]
        results = [future.result() for future in futures]

    successes = [
        result for result in results
        if not isinstance(result, Exception)
    ]
    failures = [
        result for result in results
        if isinstance(result, Exception)
    ]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(
        failures[0],
        InvalidApprovalTransitionError,
    )

    final_request = store.get(request.request_id)

    assert final_request.status is ApprovalStatus.CONSUMED

    events = store.events(request.request_id)
    consumed_events = [
        event for event in events
        if event["event_type"] == "CONSUMED"
    ]

    assert len(consumed_events) == 1


def test_concurrent_idempotent_submissions_create_one_request(hitl):
    service, store = hitl

    worker_count = 8
    barrier = Barrier(worker_count)
    idempotency_key = "concurrent-risk-request-001"

    def submit():
        barrier.wait()
        return service.request_approval(
            action_type="risk_limit_change",
            requester_id="agent-01",
            reason="Proposed change exceeds configured risk threshold.",
            payload={
                "commodity": "GOLD",
                "new_limit": 250000,
            },
            idempotency_key=idempotency_key,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(submit)
            for _ in range(worker_count)
        ]
        results = [future.result() for future in futures]

    request_ids = {
        result.request_id for result in results
    }

    assert len(request_ids) == 1

    request_id = results[0].request_id
    stored_request = store.get(request_id)

    assert stored_request.status is ApprovalStatus.PENDING
    assert stored_request.idempotency_key == idempotency_key

    events = store.events(request_id)

    assert len(events) == 1
    assert events[0]["event_type"] == "CREATED"

    with store._connect() as connection:
        request_count = connection.execute(
            "SELECT COUNT(*) FROM hitl_approval_requests"
        ).fetchone()[0]

    assert request_count == 1
