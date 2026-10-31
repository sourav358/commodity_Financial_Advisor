
from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any
import uuid
from commodity_finance_assistance.memory.database import get_connection


TRADE_FIELDS = {
    "commodity": ("commodity",),
    "instrument": ("instrument",),
    "direction": ("direction", "side"),
    "quantity": ("quantity",),
    "unit": ("unit",),
    "price": ("price", "trade price"),
    "trade_date": ("trade date", "trade_date", "date"),
    "delivery": ("delivery", "delivery month"),
    "exchange": ("exchange",),
    "counterparty": ("counterparty",),
    "legal_entity": ("legal entity", "legal_entity", "entity"),
    "book": ("book",),
    "portfolio": ("portfolio",),
}

REQUIRED_FIELDS = tuple(TRADE_FIELDS)

FIELD_LABELS = {
    "commodity": "Commodity",
    "instrument": "Instrument",
    "direction": "Direction",
    "quantity": "Quantity",
    "unit": "Unit",
    "price": "Price",
    "trade_date": "Trade date",
    "delivery": "Delivery",
    "exchange": "Exchange",
    "counterparty": "Counterparty",
    "legal_entity": "Legal entity",
    "book": "Book",
    "portfolio": "Portfolio",
}


class ApprovalDraftStore:
    """Persist one active approval draft per conversation."""

    def __init__(self) -> None:
        self.initialize()

    @staticmethod
    def initialize() -> None:
        with get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hitl_approval_drafts (
                    conversation_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(hitl_approval_drafts)"
                ).fetchall()
            }

            if "idempotency_key" not in columns:
                connection.execute(
                    "ALTER TABLE hitl_approval_drafts "
                    "ADD COLUMN idempotency_key TEXT"
                )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_hitl_drafts_idempotency
                ON hitl_approval_drafts(idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    conversation_id,
                    action_type,
                    status,
                    fields_json,
                    idempotency_key
                FROM hitl_approval_drafts
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "conversation_id": row["conversation_id"],
            "action_type": row["action_type"],
            "status": row["status"],
            "fields": json.loads(row["fields_json"]),
            "idempotency_key": row["idempotency_key"],
        }
    def create(self, conversation_id: str, action_type: str) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO hitl_approval_drafts (
                    conversation_id,
                    action_type,
                    status,
                    fields_json,
                    idempotency_key
                )
                VALUES (?, ?, 'collecting', '{}', ?)
                """,
                (
                    conversation_id,
                    action_type,
                    str(uuid.uuid4()),
                ),
            )

    def update(
        self,
        conversation_id: str,
        fields: dict[str, Any],
        status: str,
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE hitl_approval_drafts
                SET fields_json = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (
                    json.dumps(fields, ensure_ascii=False),
                    status,
                    conversation_id,
                ),
            )

    def set_status(self, conversation_id: str, status: str) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE hitl_approval_drafts
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (status, conversation_id),
            )
    def reset(self, conversation_id: str, action_type: str) -> None:
        """Start a new draft after cancellation with a fresh idempotency key."""
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE hitl_approval_drafts
                SET action_type = ?,
                    status = 'collecting',
                    fields_json = '{}',
                    idempotency_key = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (
                    action_type,
                    str(uuid.uuid4()),
                    conversation_id,
                ),
            )


def parse_trade_fields(message: str) -> dict[str, str]:
    """Extract explicitly labelled trade fields from a user message."""
    aliases = {
        alias.casefold(): field
        for field, names in TRADE_FIELDS.items()
        for alias in names
    }

    extracted: dict[str, str] = {}

    for line in message.splitlines():
        match = re.match(r"^\s*([A-Za-z_ ]+)\s*:\s*(.*?)\s*$", line)

        if not match:
            continue

        label = match.group(1).strip().casefold()
        value = match.group(2).strip()

        field = aliases.get(label)

        if field and value:
            extracted[field] = value

    return extracted


def validate_trade_fields(fields: dict[str, Any]) -> None:
    """Validate required trade fields before requesting human approval."""
    missing = [
        field for field in REQUIRED_FIELDS
        if not isinstance(fields.get(field), str)
        or not fields[field].strip()
    ]

    if missing:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(FIELD_LABELS[field] for field in missing)
        )

    direction = fields["direction"].strip().upper()
    if direction not in {"BUY", "SELL"}:
        raise ValueError("Direction must be BUY or SELL.")

    try:
        quantity = Decimal(fields["quantity"])
        if not quantity.is_finite() or quantity <= 0:
            raise ValueError
    except (InvalidOperation, ValueError):
        raise ValueError("Quantity must be a positive number.")

    try:
        price_match = re.search(r"\d+(?:\.\d+)?", fields["price"])
        if not price_match:
            raise ValueError
        price = Decimal(price_match.group())
        if not price.is_finite() or price <= 0:
            raise ValueError
    except (InvalidOperation, ValueError):
        raise ValueError("Price must contain a positive numeric value.")

    try:
        date.fromisoformat(fields["trade_date"])
    except ValueError:
        raise ValueError("Trade date must use YYYY-MM-DD format.")

    for field in REQUIRED_FIELDS:
        if len(fields[field]) > 500:
            raise ValueError(
                f"{FIELD_LABELS[field]} exceeds the 500-character limit."
            )


def missing_trade_fields(fields: dict[str, Any]) -> list[str]:
    return [
        FIELD_LABELS[field]
        for field in REQUIRED_FIELDS
        if not isinstance(fields.get(field), str)
        or not fields[field].strip()
    ]
