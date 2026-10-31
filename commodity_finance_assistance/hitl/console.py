from __future__ import annotations

import json
import sqlite3

from commodity_finance_assistance.hitl import (
    ApprovalError,
    build_hitl_service_from_config,
)


def handle_approvals() -> None:
    """List pending approval requests in the local CLI."""
    try:
        hitl = build_hitl_service_from_config()
        requests = hitl.list_pending(limit=50)
        if not requests:
            print("No pending approval requests.")
            return

        print("\nPending approval requests:")
        for request in requests:
            print("-" * 72)
            print(f"Request ID : {request.request_id}")
            print(f"Action     : {request.action_type}")
            print(f"Requester  : {request.requester_id}")
            print(f"Reason     : {request.reason}")
            print(f"Created    : {request.created_at}")
            print(f"Expires    : {request.expires_at}")
            print(
                "Payload    : "
                + json.dumps(request.payload, ensure_ascii=False, sort_keys=True)
            )
        print("-" * 72)
        print("Reviewer IDs are audit labels in this local prototype, not authentication.")
    except (ApprovalError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"Approval listing failed: {type(exc).__name__}: {exc}")


def handle_approve(
    request_id: str,
    approver_id: str,
    reason: str = "Approved via local CLI.",
) -> None:
    """Approve a pending request."""
    try:
        hitl = build_hitl_service_from_config()
        request = hitl.approve(
            request_id=request_id,
            approver_id=approver_id,
            reason=reason,
        )
        print(
            f"Approval recorded: {request.request_id} "
            f"({request.status.value})."
        )
        print("This records authorization only; it does not execute the action.")
    except (ApprovalError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"Approval failed: {type(exc).__name__}: {exc}")


def handle_reject(
    request_id: str,
    approver_id: str,
    reason: str,
) -> None:
    """Reject a pending request."""
    try:
        hitl = build_hitl_service_from_config()
        request = hitl.reject(
            request_id=request_id,
            approver_id=approver_id,
            reason=reason,
        )
        print(f"Request rejected: {request.request_id} ({request.status.value}).")
    except (ApprovalError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"Rejection failed: {type(exc).__name__}: {exc}")
