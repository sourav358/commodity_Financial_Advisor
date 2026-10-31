from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    action_type: str
    requester_id: str
    reason: str
    payload: dict[str, Any]
    payload_hash: str
    status: ApprovalStatus
    created_at: str
    expires_at: str
    approver_id: str | None = None
    decision_reason: str | None = None
    decided_at: str | None = None
    consumed_at: str | None = None
    idempotency_key: str | None = None


class ApprovalError(Exception):
    """Base exception for approval workflow errors."""


class ApprovalNotFoundError(ApprovalError):
    pass


class InvalidApprovalTransitionError(ApprovalError):
    pass


class ApprovalExpiredError(ApprovalError):
    pass


class ApprovalPermissionError(ApprovalError):
    pass


class ApprovalPayloadMismatchError(ApprovalError):
    pass
