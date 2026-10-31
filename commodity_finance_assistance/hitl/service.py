from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import ApprovalRequest, ApprovalStatus
from .store import SQLiteApprovalStore, _payload_hash


_IDEMPOTENCY_KEY_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class HITLService:
    """Application service for creating, reviewing, and consuming approvals."""

    def __init__(
        self,
        store: SQLiteApprovalStore,
        *,
        default_ttl_minutes: int = 30,
        require_separation_of_duties: bool = True,
    ) -> None:
        if (
            isinstance(default_ttl_minutes, bool)
            or not isinstance(default_ttl_minutes, int)
            or not 1 <= default_ttl_minutes <= 10080
        ):
            raise ValueError(
                "default_ttl_minutes must be between 1 and 10080"
            )

        self.store = store
        self.default_ttl_minutes = default_ttl_minutes
        self.require_separation_of_duties = require_separation_of_duties

    @staticmethod
    def _required_text(
        value: str,
        field_name: str,
        max_length: int = 500,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )

        value = value.strip()

        if len(value) > max_length:
            raise ValueError(
                f"{field_name} cannot exceed {max_length} characters"
            )

        return value

    @staticmethod
    def _validate_idempotency_key(
        value: str | None,
    ) -> str | None:
        """Validate an optional key used for retry-safe submissions."""

        if value is None:
            return None

        if (
            not isinstance(value, str)
            or _IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(
                "idempotency_key must be 1-128 characters and contain only "
                "letters, numbers, '.', '_', ':', or '-'"
            )

        return value

    def request_approval(
        self,
        *,
        action_type: str,
        requester_id: str,
        reason: str,
        payload: dict[str, Any],
        ttl_minutes: int | None = None,
        idempotency_key: str | None = None,
    ) -> ApprovalRequest:
        """Create an approval request with idempotent retry support."""

        action_type = self._required_text(
            action_type, "action_type", 100
        )
        requester_id = self._required_text(
            requester_id, "requester_id", 150
        )
        reason = self._required_text(reason, "reason", 2000)

        idempotency_key = self._validate_idempotency_key(
            idempotency_key
        )

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        # Validate JSON serialization before persisting the request.
        try:
            json.dumps(
                payload,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "payload must contain JSON-serializable values"
            ) from exc

        ttl = (
            self.default_ttl_minutes
            if ttl_minutes is None
            else ttl_minutes
        )

        if (
            isinstance(ttl, bool)
            or not isinstance(ttl, int)
            or not 1 <= ttl <= 10080
        ):
            raise ValueError(
                "ttl_minutes must be between 1 and 10080"
            )

        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            action_type=action_type,
            requester_id=requester_id,
            reason=reason,
            payload=payload,
            payload_hash=_payload_hash(payload),
            idempotency_key=idempotency_key,
            status=ApprovalStatus.PENDING,
            created_at=now.isoformat(),
            expires_at=(
                now + timedelta(minutes=ttl)
            ).isoformat(),
        )

        # SQLiteApprovalStore.create() must enforce idempotency
        # transactionally and reject key reuse with different content.
        return self.store.create(request)

    def approve(
        self,
        request_id: str,
        approver_id: str,
        reason: str | None = None,
    ) -> ApprovalRequest:
        approver_id = self._required_text(
            approver_id, "approver_id", 150
        )

        return self.store.approve(
            request_id,
            approver_id,
            reason,
            forbid_requester=self.require_separation_of_duties,
        )

    def reject(
        self,
        request_id: str,
        approver_id: str,
        reason: str,
    ) -> ApprovalRequest:
        approver_id = self._required_text(
            approver_id, "approver_id", 150
        )
        reason = self._required_text(
            reason, "reason", 2000
        )

        return self.store.reject(
            request_id,
            approver_id,
            reason,
            forbid_requester=self.require_separation_of_duties,
        )

    def get_request(
        self,
        request_id: str,
    ) -> ApprovalRequest:
        return self.store.get(request_id)

    def list_pending(
        self,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        return self.store.list_pending(limit)

    def audit_events(
        self,
        request_id: str,
    ) -> list[dict[str, Any]]:
        return self.store.events(request_id)

    def consume_approval(
        self,
        request_id: str,
        executor_id: str,
        *,
        expected_action_type: str,
        expected_payload: dict[str, Any],
    ) -> ApprovalRequest:
        """Atomically consume an approval once.

        The action type and payload must match the approved request.

        This method returns authorization evidence only.
        It does not execute a trade or external action.
        """

        executor_id = self._required_text(
            executor_id, "executor_id", 150
        )

        expected_action_type = self._required_text(
            expected_action_type,
            "expected_action_type",
            100,
        )

        if not isinstance(expected_payload, dict):
            raise ValueError(
                "expected_payload must be a dictionary"
            )

        try:
            json.dumps(
                expected_payload,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected_payload must contain JSON-serializable values"
            ) from exc

        request = self.store.get(request_id)

        if request.action_type != expected_action_type:
            raise ValueError(
                "Approval action type does not match the requested action"
            )

        return self.store.consume(
            request_id,
            executor_id,
            expected_payload=expected_payload,
        )


def build_hitl_service_from_config() -> HITLService:
    """Build the approval service using the configured SQLite database."""

    from pathlib import Path

    from commodity_finance_assistance.config_loader import (
        PROJECT_ROOT,
        load_config,
    )

    config = load_config()

    database_config = config.get("database", {})
    database_url = database_config.get("url")

    if (
        not isinstance(database_url, str)
        or not database_url.startswith("sqlite:///")
    ):
        raise ValueError(
            "HITL currently requires a SQLite database URL in config.yml"
        )

    raw_path = database_url[len("sqlite:///"):]
    db_path = Path(raw_path)

    if not db_path.is_absolute():
        db_path = Path(PROJECT_ROOT) / db_path

    return HITLService(
        SQLiteApprovalStore(db_path)
    )