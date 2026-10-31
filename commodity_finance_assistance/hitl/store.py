from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import replace

from .models import (
    ApprovalNotFoundError,
    ApprovalPayloadMismatchError,
    ApprovalPermissionError,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalExpiredError,
    InvalidApprovalTransitionError,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class SQLiteApprovalStore:
    """SQLite persistence for approval requests and immutable decision history."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hitl_approval_requests (
                    request_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    requester_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    idempotency_key TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED', 'CONSUMED')
                    ),
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    approver_id TEXT,
                    decision_reason TEXT,
                    decided_at TEXT,
                    consumed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_hitl_status_created
                    ON hitl_approval_requests(status, created_at);

                CREATE TABLE IF NOT EXISTS hitl_approval_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    reason TEXT,
                    event_at TEXT NOT NULL,
                    FOREIGN KEY(request_id)
                        REFERENCES hitl_approval_requests(request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_hitl_events_request
                    ON hitl_approval_events(request_id, event_at);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(hitl_approval_requests)"
                ).fetchall()
            }

            if "idempotency_key" not in columns:
                connection.execute(
                    "ALTER TABLE hitl_approval_requests "
                    "ADD COLUMN idempotency_key TEXT"
                )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_hitl_approval_idempotency
                ON hitl_approval_requests(idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )

    @staticmethod
    def _to_model(row: sqlite3.Row) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=row["request_id"],
            action_type=row["action_type"],
            requester_id=row["requester_id"],
            reason=row["reason"],
            payload=json.loads(row["payload_json"]),
            payload_hash=row["payload_hash"],
            status=ApprovalStatus(row["status"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            approver_id=row["approver_id"],
            decision_reason=row["decision_reason"],
            decided_at=row["decided_at"],
            consumed_at=row["consumed_at"],
            idempotency_key=row["idempotency_key"],
        )

    def create(
        self,
        request: ApprovalRequest,
        idempotency_key: str | None = None,
    ) -> ApprovalRequest:
        """Create an approval exactly once for a given idempotency key."""
        key = (
            idempotency_key
            if idempotency_key is not None
            else request.idempotency_key
        )

        if key is not None:
            if not isinstance(key, str) or not key.strip() or len(key) > 128:
                raise ValueError("Invalid idempotency key.")
            key = key.strip()

        connection = self._connect()

        try:
            # Serialize competing submissions before checking/inserting.
            connection.execute("BEGIN IMMEDIATE")

            if key is not None:
                existing = connection.execute(
                    """
                    SELECT *
                    FROM hitl_approval_requests
                    WHERE idempotency_key = ?
                    """,
                    (key,),
                ).fetchone()

                if existing is not None:
                    same_request = (
                        existing["action_type"] == request.action_type
                        and existing["requester_id"] == request.requester_id
                        and existing["reason"] == request.reason
                        and existing["payload_hash"] == request.payload_hash
                    )

                    if not same_request:
                        raise ApprovalPayloadMismatchError(
                            "This idempotency key was already used "
                            "for a different approval request."
                        )

                    connection.commit()
                    return self._to_model(existing)

            connection.execute(
                """
                INSERT INTO hitl_approval_requests (
                    request_id,
                    action_type,
                    requester_id,
                    reason,
                    payload_json,
                    payload_hash,
                    status,
                    created_at,
                    expires_at,
                    idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.action_type,
                    request.requester_id,
                    request.reason,
                    _canonical_json(request.payload),
                    request.payload_hash,
                    request.status.value,
                    request.created_at,
                    request.expires_at,
                    key,
                ),
            )

            connection.execute(
                """
                INSERT INTO hitl_approval_events (
                    request_id,
                    event_type,
                    actor_id,
                    reason,
                    event_at
                )
                VALUES (?, 'CREATED', ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.requester_id,
                    request.reason,
                    request.created_at,
                ),
            )

            connection.commit()

            return replace(request, idempotency_key=key)

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()
         

    def get(self, request_id: str) -> ApprovalRequest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM hitl_approval_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            raise ApprovalNotFoundError(f"Approval request not found: {request_id}")
        return self._to_model(row)

    def list_pending(self, limit: int = 100) -> list[ApprovalRequest]:
        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM hitl_approval_requests
                   WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._to_model(row) for row in rows]

    def events(self, request_id: str) -> list[dict[str, Any]]:
        # Confirm the request exists before returning its audit history.
        self.get(request_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT event_type, actor_id, reason, event_at
                   FROM hitl_approval_events WHERE request_id = ?
                   ORDER BY event_id ASC""",
                (request_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _transition(
        self,
        request_id: str,
        *,
        target: ApprovalStatus,
        actor_id: str,
        reason: str | None = None,
        expected_payload: dict[str, Any] | None = None,
        forbid_requester: bool = False,
    ) -> ApprovalRequest:
        now = _utc_now()
        expired = False
        error: Exception | None = None
        result: ApprovalRequest | None = None

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM hitl_approval_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise ApprovalNotFoundError(f"Approval request not found: {request_id}")

            current = self._to_model(row)
            if forbid_requester and actor_id == current.requester_id:
                raise ApprovalPermissionError("The requester cannot approve or reject their own request.")

            if current.status is not ApprovalStatus.PENDING and target in {
                ApprovalStatus.APPROVED, ApprovalStatus.REJECTED
            }:
                raise InvalidApprovalTransitionError(
                    f"Cannot change a {current.status.value} request to {target.value}."
                )
            if target is ApprovalStatus.CONSUMED and current.status is not ApprovalStatus.APPROVED:
                raise InvalidApprovalTransitionError("Only approved requests can be consumed.")

            expiry = datetime.fromisoformat(current.expires_at)
            if expiry <= now:
                connection.execute(
                    "UPDATE hitl_approval_requests SET status = 'EXPIRED' WHERE request_id = ?",
                    (request_id,),
                )
                connection.execute(
                    """INSERT INTO hitl_approval_events
                       (request_id, event_type, actor_id, reason, event_at)
                       VALUES (?, 'EXPIRED', ?, 'Approval request expired', ?)""",
                    (request_id, actor_id, now.isoformat()),
                )
                expired = True
            else:
                if expected_payload is not None and _payload_hash(expected_payload) != current.payload_hash:
                    raise ApprovalPayloadMismatchError(
                        "The action payload differs from the approved payload."
                    )

                if target is ApprovalStatus.APPROVED:
                    connection.execute(
                        """UPDATE hitl_approval_requests
                           SET status = 'APPROVED', approver_id = ?,
                               decision_reason = ?, decided_at = ?
                           WHERE request_id = ? AND status = 'PENDING'""",
                        (actor_id, reason, now.isoformat(), request_id),
                    )
                    event_type = "APPROVED"
                elif target is ApprovalStatus.REJECTED:
                    connection.execute(
                        """UPDATE hitl_approval_requests
                           SET status = 'REJECTED', approver_id = ?,
                               decision_reason = ?, decided_at = ?
                           WHERE request_id = ? AND status = 'PENDING'""",
                        (actor_id, reason, now.isoformat(), request_id),
                    )
                    event_type = "REJECTED"
                elif target is ApprovalStatus.CONSUMED:
                    connection.execute(
                        """UPDATE hitl_approval_requests
                           SET status = 'CONSUMED', consumed_at = ?
                           WHERE request_id = ? AND status = 'APPROVED'""",
                        (now.isoformat(), request_id),
                    )
                    event_type = "CONSUMED"
                else:
                    raise InvalidApprovalTransitionError(f"Unsupported target state: {target}")

                connection.execute(
                    """INSERT INTO hitl_approval_events
                       (request_id, event_type, actor_id, reason, event_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (request_id, event_type, actor_id, reason, now.isoformat()),
                )
                updated = connection.execute(
                    "SELECT * FROM hitl_approval_requests WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                result = self._to_model(updated)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        if expired:
            raise ApprovalExpiredError("Approval request has expired.")
        assert result is not None
        return result

    def approve(
        self, request_id: str, approver_id: str, reason: str | None = None, *,
        forbid_requester: bool = True,
    ) -> ApprovalRequest:
        return self._transition(
            request_id,
            target=ApprovalStatus.APPROVED,
            actor_id=approver_id,
            reason=reason,
            forbid_requester=forbid_requester,
        )

    def reject(
        self, request_id: str, approver_id: str, reason: str, *,
        forbid_requester: bool = True,
    ) -> ApprovalRequest:
        return self._transition(
            request_id,
            target=ApprovalStatus.REJECTED,
            actor_id=approver_id,
            reason=reason,
            forbid_requester=forbid_requester,
        )

    def consume(
        self,
        request_id: str,
        executor_id: str,
        expected_payload: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        return self._transition(
            request_id,
            target=ApprovalStatus.CONSUMED,
            actor_id=executor_id,
            expected_payload=expected_payload,
        )
