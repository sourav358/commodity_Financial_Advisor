
from datetime import date

import pytest
from uuid import uuid4

from commodity_finance_assistance.hitl.drafts import (
    ApprovalDraftStore,
    missing_trade_fields,
    parse_trade_fields,
    validate_trade_fields,
)


@pytest.fixture
def store():
    return ApprovalDraftStore()


def complete_trade():
    return {
        "commodity": "GOLD",
        "instrument": "MCX Gold Futures",
        "direction": "BUY",
        "quantity": "1",
        "unit": "lot",
        "price": "142000 INR",
        "trade_date": date.today().isoformat(),
        "delivery": "December 2026",
        "exchange": "MCX",
        "counterparty": "Test Counterparty",
        "legal_entity": "Test Trading Entity",
        "book": "TEST_BOOK",
        "portfolio": "TEST_PORTFOLIO",
    }


def test_draft_persists_by_conversation(store):
    conversation_id = str(uuid4())

    store.create(conversation_id, "trade_execution_request")

    draft = store.get(conversation_id)

    assert draft is not None
    assert draft["status"] == "collecting"
    assert draft["fields"] == {}


def test_drafts_are_isolated_by_conversation(store):
    conversation_1 = str(uuid4())
    conversation_2 = str(uuid4())

    store.create(conversation_1, "trade_execution_request")
    store.create(conversation_2, "trade_execution_request")

    store.update(
        conversation_1,
        {"commodity": "GOLD"},
        "collecting",
    )

    first = store.get(conversation_1)
    second = store.get(conversation_2)

    assert first["fields"] == {"commodity": "GOLD"}
    assert second["fields"] == {}

def test_parse_trade_fields():
    message = """Commodity: GOLD
Direction: BUY
Quantity: 1
Price: 142000 INR"""

    fields = parse_trade_fields(message)

    assert fields == {
        "commodity": "GOLD",
        "direction": "BUY",
        "quantity": "1",
        "price": "142000 INR",
    }


def test_missing_fields_are_reported():
    missing = missing_trade_fields({"commodity": "GOLD"})

    assert "Commodity" not in missing
    assert "Instrument" in missing
    assert "Direction" in missing


def test_complete_trade_passes_validation():
    validate_trade_fields(complete_trade())


def test_invalid_direction_is_rejected():
    trade = complete_trade()
    trade["direction"] = "HOLD"

    with pytest.raises(ValueError, match="Direction"):
        validate_trade_fields(trade)


def test_invalid_quantity_is_rejected():
    trade = complete_trade()
    trade["quantity"] = "-1"

    with pytest.raises(ValueError, match="Quantity"):
        validate_trade_fields(trade)


def test_invalid_trade_date_is_rejected():
    trade = complete_trade()
    trade["trade_date"] = "31-10-2026"

    with pytest.raises(ValueError, match="Trade date"):
        validate_trade_fields(trade)
def test_draft_has_persistent_idempotency_key(store):
    conversation_id = str(uuid4())

    store.create(conversation_id, "trade_execution_request")

    first = store.get(conversation_id)
    second = store.get(conversation_id)

    assert first["idempotency_key"]
    assert first["idempotency_key"] == second["idempotency_key"]


def test_reset_draft_generates_new_idempotency_key(store):
    conversation_id = str(uuid4())

    store.create(conversation_id, "trade_execution_request")
    first = store.get(conversation_id)

    store.update(
        conversation_id,
        {"commodity": "GOLD"},
        "cancelled",
    )
    store.reset(conversation_id, "trade_execution_request")

    second = store.get(conversation_id)

    assert second["status"] == "collecting"
    assert second["fields"] == {}
    assert second["idempotency_key"] != first["idempotency_key"]