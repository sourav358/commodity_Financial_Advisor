
import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from commodity_finance_assistance.tools.mcx_contract_metadata import (
    load_contract_metadata,
    is_contract_metadata_verified,
)
from datetime import timezone

from commodity_finance_assistance.tools.mcx_bhavcopy import MCXFutureQuote

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS mcx_eod_quotes (
    trade_date TEXT NOT NULL,
    instrument TEXT NOT NULL,
    symbol TEXT NOT NULL,
    commodity TEXT NOT NULL,
    expiry_date TEXT NOT NULL,
    close_price TEXT NOT NULL,
    volume_lots TEXT,
    open_interest_lots TEXT,
    source_file TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, instrument, symbol, expiry_date)
)
"""

DISPLAY_NAMES = {
    "GOLD": "Gold",
    "GOLDM": "Gold Mini",
    "GOLDGUINEA": "Gold Guinea",
    "GOLDPETAL": "Gold Petal",
    "GOLDTEN": "Gold Ten",
    "SILVER": "Silver",
    "SILVERM": "Silver Mini",
    "SILVERMIC": "Silver Micro",
    "SILVER100": "Silver 100",
    "CRUDEOIL": "Crude Oil",
    "CRUDEOILM": "Crude Oil Mini",
    "NATURALGAS": "Natural Gas",
    "NATGASMINI": "Natural Gas Mini",
    "COPPER": "Copper",
    "COPPERM": "Copper Mini",
    "ALUMINIUM": "Aluminium",
    "ALUMINI": "Aluminium Mini",
    "ZINC": "Zinc",
    "ZINCMINI": "Zinc Mini",
    "LEAD": "Lead",
    "LEADMINI": "Lead Mini",
    "NICKEL": "Nickel",
    "MENTHAOIL": "Mentha Oil",
    "COTTON": "Cotton",
    "RUBBER": "Rubber",
    "CARDAMOM": "Cardamom",
    "COTTONOIL": "Cottonseed Oil",
    "KAPAS": "Kapas",
    "SUNOIL": "Sunflower Oil",
    "STEELREBAR": "Steel Rebar",
}

ALIASES = {
    "gold": ("GOLD", "GOLDM", "GOLDGUINEA", "GOLDPETAL", "GOLDTEN"),
    "silver": ("SILVER", "SILVERM", "SILVERMIC", "SILVER100"),
    "crude oil": ("CRUDEOIL", "CRUDEOILM"),
    "oil": ("CRUDEOIL", "CRUDEOILM"),
    "natural gas": ("NATURALGAS", "NATGASMINI"),
    "copper": ("COPPER", "COPPERM"),
    "aluminium": ("ALUMINIUM", "ALUMINI"),
    "aluminum": ("ALUMINIUM", "ALUMINI"),
    "zinc": ("ZINC", "ZINCMINI"),
    "lead": ("LEAD", "LEADMINI"),
    "nickel": ("NICKEL",),
    "mentha oil": ("MENTHAOIL",),
    "cotton": ("COTTON",),
    "rubber": ("RUBBER",),
    "cardamom": ("CARDAMOM",),
    "kapas": ("KAPAS",),
}


def normalize_header(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def parse_trade_date(value):
    value = str(value).strip().upper()

    for fmt in (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%b-%Y",
        "%d %b %Y",
        "%d-%B-%Y",
        "%d%b%Y",
        "%d%B%Y",
    ):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue

    raise ValueError(f"Unsupported MCX date: {value!r}")


def parse_price(value):
    cleaned = str(value).strip().replace(",", "")
    if not cleaned or cleaned in {".", "-", "NA", "N/A"}:
        raise ValueError("Missing closing price.")

    try:
        price = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid closing price: {value!r}") from exc

    if not price.is_finite() or price <= 0:
        raise ValueError(f"Closing price must be positive: {value!r}")

    return str(price)


@dataclass(frozen=True)
class ImportResult:
    rows_read: int
    rows_imported: int
    rows_skipped: int


def import_bhavcopy(csv_path, connection):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bhavcopy file not found: {path}")

    connection.execute(CREATE_TABLE)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Bhavcopy CSV has no header row.")

        columns = {normalize_header(name): name for name in reader.fieldnames}

        def find_column(*names, required=True):
            for name in names:
                key = normalize_header(name)
                if key in columns:
                    return columns[key]
            if required:
                raise ValueError(
                    "Bhavcopy is missing a required column: "
                    + ", ".join(names)
                )
            return None

        date_col = find_column("Date", "Trade Date", "Trading Date")
        instrument_col = find_column(
            "Instrument", "Instrument Type", "Instrument Name"
        )
        symbol_col = find_column("Symbol", "Commodity")
        expiry_col = find_column("Expiry Date", "Expiry")
        close_col = find_column("Close", "Closing Price", "Close Price")
        volume_col = find_column(
            "Volume(Lots)", "Volume (Lots)", "Vol (Lots)", required=False
        )
        oi_col = find_column(
            "Open Interest(Lots)", "Open Interest (Lots)",
            "OI (Lots)", required=False
        )

        rows_read = 0
        rows_imported = 0
        rows_skipped = 0

        sql = """
        INSERT INTO mcx_eod_quotes (
            trade_date, instrument, symbol, commodity, expiry_date,
            close_price, volume_lots, open_interest_lots, source_file
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (trade_date, instrument, symbol, expiry_date)
        DO UPDATE SET
            commodity = excluded.commodity,
            close_price = excluded.close_price,
            volume_lots = excluded.volume_lots,
            open_interest_lots = excluded.open_interest_lots,
            source_file = excluded.source_file,
            imported_at = CURRENT_TIMESTAMP
        """

        # One transaction: a database failure rolls back the entire import.
        with connection:
            for row in reader:
                rows_read += 1

                # Import commodity futures only; exclude options and indices.
                instrument = str(row.get(instrument_col, "")).strip().upper()
                if instrument != "FUTCOM":
                    rows_skipped += 1
                    continue

                symbol = str(row.get(symbol_col, "")).strip().upper()

                try:
                    trade_date = parse_trade_date(row[date_col])
                    expiry_date = parse_trade_date(row[expiry_col])
                    close_price = parse_price(row[close_col])

                    if not symbol:
                        raise ValueError("Missing commodity symbol.")
                except (ValueError, TypeError, KeyError):
                    rows_skipped += 1
                    continue

                volume = (
                    str(row.get(volume_col, "") or "").strip() or None
                    if volume_col else None
                )
                open_interest = (
                    str(row.get(oi_col, "") or "").strip() or None
                    if oi_col else None
                )

                connection.execute(
                    sql,
                    (
                        trade_date,
                        instrument,
                        symbol,
                        DISPLAY_NAMES.get(symbol, symbol.title()),
                        expiry_date,
                        close_price,
                        volume,
                        open_interest,
                        path.name,
                    ),
                )
                rows_imported += 1

    return ImportResult(rows_read, rows_imported, rows_skipped)




def get_latest_quote(connection, commodity):
    requested = commodity.strip().lower()
    if not requested:
        raise ValueError("Commodity name cannot be empty.")

    symbols = ALIASES.get(requested)
    if symbols is None:
        raise ValueError(
            "Unsupported MCX commodity. Try gold, silver, crude oil, "
            "natural gas, copper, aluminium, zinc, lead, nickel, "
            "mentha oil, cotton, rubber, cardamom or kapas."
        )

    # Try symbols in preference order.
    row = None

    for symbol in symbols:
        row = connection.execute(
            """
            SELECT trade_date, symbol, commodity, expiry_date,
                   close_price, source_file
            FROM mcx_eod_quotes
            WHERE symbol = ?
              AND expiry_date >= trade_date
              AND close_price IS NOT NULL
            ORDER BY trade_date DESC, expiry_date ASC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()

        if row is not None:
            break

    if row is None:
        return None

    trade_date, symbol, name, expiry, price, source_file = row

    observed_at = datetime.strptime(
        trade_date, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    metadata = load_contract_metadata(symbol)
    verified = is_contract_metadata_verified(metadata)

    if verified:
        display_unit = metadata["price_unit"]
    else:
        display_unit = "contract price unit (UNVERIFIED)"

    return MCXFutureQuote(
        commodity=name,
        contract=symbol,
        expiry_date=datetime.strptime(expiry, "%Y-%m-%d").date(),
        price=Decimal(str(price)),
        observed_at=observed_at,
        unit=display_unit,
        source=f"MCX official bhavcopy — end of day ({source_file})",
        unit_status="VERIFIED" if verified else "UNVERIFIED",
    )
