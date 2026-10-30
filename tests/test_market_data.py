
from types import SimpleNamespace

import pytest

from commodity_finance_assistance.tools.market_data import (
    MarketDataTool,
    UnavailableMarketDataProvider,
)


def test_unavailable_provider_returns_no_quote():
    tool = MarketDataTool()

    assert tool.get_quote("crude oil") is None


def test_empty_commodity_is_rejected():
    tool = MarketDataTool()

    with pytest.raises(ValueError, match="cannot be empty"):
        tool.get_quote("   ")


def test_quote_is_formatted_with_source_and_timestamp():
    tool = MarketDataTool()
    quote = SimpleNamespace(
        commodity="Crude Oil",
        price="75.00",
        currency="USD",
        unit="barrel",
        observed_at=SimpleNamespace(
            isoformat=lambda: "2026-10-21T10:00:00+00:00"
        ),
        source="Example Market Data Provider",
    )

    result = tool.format_quote(quote)

    assert "Crude Oil" in result
    assert "75.00 USD/barrel" in result
    assert "Example Market Data Provider" in result
    assert "2026-10-21T10:00:00+00:00" in result
