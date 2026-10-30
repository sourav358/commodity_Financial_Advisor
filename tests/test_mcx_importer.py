
import sqlite3

import pytest

from commodity_finance_assistance.tools.mcx_importer import (
    get_latest_quote,
    import_bhavcopy,
)


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def bhavcopy(tmp_path):
    path = tmp_path / "mcx_bhavcopy.csv"
    path.write_text(
        "Date,Instrument Name,Symbol,Expiry Date,Option Type,"
        "Strike Price,Open,High,Low,Close,Previous Close,"
        "Volume(Lots),Volume(In 000's),Value(Lacs),Open Interest(Lots)\n"
        "23 Oct 2026,FUTCOM,GOLD,30OCT2026,,,,,,125000,,100,20,30\n"
        "23 Oct 2026,FUTCOM,GOLD,27NOV2026,,,,,,126000,,80,18,28\n"
        "23 Oct 2026,FUTCOM,CRUDEOIL,17NOV2026,,,,,,96.80,,300,40,50\n"
        "23 Oct 2026,OPTFUT,GOLD,30OCT2026,CE,125000,,,,1000,,1,2,3\n"
        "23 Oct 2026,FUTCOM,SILVER,05NOV2026,,,,,,0,,1,1,1\n"

    )
    return path


def test_imports_futures_and_skips_options_and_invalid_prices(
    bhavcopy, connection
):
    result = import_bhavcopy(bhavcopy, connection)

    assert result.rows_read == 5
    assert result.rows_imported == 3
    assert result.rows_skipped == 2

    rows = connection.execute(
        "SELECT symbol, close_price FROM mcx_eod_quotes ORDER BY symbol, expiry_date"
    ).fetchall()

    assert rows == [
        ("CRUDEOIL", "96.80"),
        ("GOLD", "125000"),
        ("GOLD", "126000"),
    ]


def test_reimport_updates_existing_rows_without_duplicates(
    bhavcopy, connection
):
    import_bhavcopy(bhavcopy, connection)
    import_bhavcopy(bhavcopy, connection)

    count = connection.execute(
        "SELECT COUNT(*) FROM mcx_eod_quotes"
    ).fetchone()[0]

    assert count == 3


def test_latest_quote_returns_nearest_unexpired_gold_contract(
    bhavcopy, connection
):
    import_bhavcopy(bhavcopy, connection)

    quote = get_latest_quote(connection, "gold")

    assert quote is not None
    assert quote.commodity == "Gold"
    assert quote.contract == "GOLD"
    assert str(quote.price) == "125000"
    assert quote.expiry_date.isoformat() == "2026-10-30"
    assert quote.observed_at.date().isoformat() == "2026-10-23"


def test_latest_quote_returns_none_when_no_matching_data(
    bhavcopy, connection
):
    import_bhavcopy(bhavcopy, connection)

    assert get_latest_quote(connection, "natural gas") is None


def test_missing_file_raises_clear_error(tmp_path, connection):
    with pytest.raises(FileNotFoundError, match="Bhavcopy file not found"):
        import_bhavcopy(tmp_path / "missing.csv", connection)


def test_empty_commodity_is_rejected(connection):
    with pytest.raises(ValueError, match="cannot be empty"):
        get_latest_quote(connection, " ")
