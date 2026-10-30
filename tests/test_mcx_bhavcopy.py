
from datetime import date
from decimal import Decimal

import pytest

from commodity_finance_assistance.tools.mcx_bhavcopy import (
    MCXBhavcopyProvider,
)


@pytest.fixture
def bhavcopy(tmp_path):
    path = tmp_path / "bhavcopy.csv"
    path.write_text(
        "Date,Instrument,Commodity,Symbol,Expiry Date,Price Unit,Close\n"
        "2026-10-20,FUTCOM,GOLD,GOLD,2026-11-05,10 GRAMS,125000\n"
        "2026-10-20,FUTCOM,GOLD,GOLD,2026-12-05,10 GRAMS,126000\n"
        "2026-10-20,OPTCOM,GOLD,GOLD,2026-11-05,10 GRAMS,1000\n"
        "2026-10-20,FUTCOM,CRUDEOIL,CRUDEOIL,2026-11-19,BBL,5200\n"
        "2026-10-20,FUTCOM,SILVER,SILVER,2026-11-05,KG,150000\n",
        encoding="utf-8",
    )
    return path


def test_selects_nearest_gold_futures_expiry(bhavcopy):
    quote = MCXBhavcopyProvider(bhavcopy).get_quote("gold")

    assert quote.commodity == "Gold"
    assert quote.expiry_date == date(2026, 11, 5)
    assert quote.price == Decimal("125000")
    assert quote.currency == "INR"
    assert quote.unit == "10 GRAMS"
    assert quote.observed_at.date() == date(2026, 10, 20)


def test_selects_crude_oil(bhavcopy):
    quote = MCXBhavcopyProvider(bhavcopy).get_quote("crude oil")

    assert quote.commodity == "Crude Oil"
    assert quote.price == Decimal("5200")
    assert quote.unit == "BBL"


def test_does_not_select_options(bhavcopy):
    # The fixture has both a gold futures row and a gold options row.
    quote = MCXBhavcopyProvider(bhavcopy).get_quote("gold")
    assert quote.price == Decimal("125000")


def test_unknown_commodity_raises_value_error(bhavcopy):
    with pytest.raises(ValueError, match="Unsupported MCX commodity"):
        MCXBhavcopyProvider(bhavcopy).get_quote("wheat")


def test_missing_file_has_clear_error(tmp_path):
    provider = MCXBhavcopyProvider(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError, match="MCX bhavcopy CSV not found"):
        provider.get_quote("gold")


def test_format_quote_identifies_mcx_and_expiry(bhavcopy):
    quote = MCXBhavcopyProvider(bhavcopy).get_quote("gold")
    formatted = MCXBhavcopyProvider.format_quote(quote)

    assert "Market: MCX futures" in formatted
    assert "Expiry: 2026-11-05" in formatted
    assert "Data date: 2026-10-20" in formatted
