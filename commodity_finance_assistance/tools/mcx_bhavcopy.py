
import csv
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


@dataclass(frozen=True)
class MCXFutureQuote:
    commodity: str
    contract: str
    expiry_date: date
    price: Decimal
    observed_at: datetime
    currency: str = "INR"
    unit_status: str = "UNVERIFIED"
    unit: str = "contract price unit"
    source: str = "MCX official bhavcopy — end of day"


class MCXBhavcopyProvider:
    ALIASES = {
        "gold": ("GOLD", "GOLDM", "GOLDGUINEA", "GOLDPETAL"),
        "silver": ("SILVER", "SILVERM", "SILVERMIC"),
        "crude oil": ("CRUDEOIL", "CRUDEOILM"),
        "oil": ("CRUDEOIL", "CRUDEOILM"),
        "natural gas": ("NATURALGAS", "NATGASMINI"),
        "copper": ("COPPER", "COPPERM"),
        "aluminium": ("ALUMINIUM", "ALUMINI"),
        "aluminum": ("ALUMINIUM", "ALUMINI"),
        "zinc": ("ZINC", "ZINCMINI"),
        "lead": ("LEAD", "LEADMINI"),
        "nickel": ("NICKEL", "NICKELM"),
        "mentha oil": ("MENTHAOIL",),
        "cotton": ("COTTON",),
        "rubber": ("RUBBER",),
        "brent": (),
    }

    DISPLAY_NAMES = {
        "GOLD": "Gold",
        "GOLDM": "Gold Mini",
        "GOLDGUINEA": "Gold Guinea",
        "GOLDPETAL": "Gold Petal",
        "SILVER": "Silver",
        "SILVERM": "Silver Mini",
        "SILVERMIC": "Silver Micro",
        "CRUDEOIL": "Crude Oil",
        "CRUDEOILM": "Crude Oil Mini",
        "NATURALGAS": "Natural Gas",
        "NATGASMINI": "Natural Gas Mini",
        "COPPER": "Copper",
        "COPPERM": "Copper Mini",
        "ALUMINIUM": "Aluminium",
        "ALUMINI": "Aluminium",
        "ZINC": "Zinc",
        "ZINCMINI": "Zinc Mini",
        "LEAD": "Lead",
        "LEADMINI": "Lead Mini",
        "NICKEL": "Nickel",
        "NICKELM": "Nickel Mini",
        "MENTHAOIL": "Mentha Oil",
        "COTTON": "Cotton",
        "RUBBER": "Rubber",
    }

    def __init__(self, csv_path=None):
        configured_path = (
            csv_path
            or os.getenv("MCX_BHAVCOPY_PATH")
            or "data/mcx_bhavcopy.csv"
        )
        self.csv_path = Path(configured_path)

    @staticmethod
    def _key(value):
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())

    
    @staticmethod
    def _parse_date(value):
        value = str(value).strip().upper()

        for fmt in (
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d %b %Y",
            "%d-%B-%Y",
            "%d%b%Y",       # MCX format: 30OCT2026
            "%d%B%Y",       # Full month name without separators
        ):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue

        raise ValueError(
            f"Unsupported date format in MCX CSV: {value!r}"
        )


    @staticmethod
    def _decimal(value):
        cleaned = str(value).strip().replace(",", "")
        if not cleaned or cleaned in {".", "-", "NA", "N/A"}:
            raise ValueError("Missing closing price in MCX CSV.")
        try:
            result = Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid price in MCX CSV: {value!r}") from exc
        if not result.is_finite() or result <= 0:
            raise ValueError(f"Invalid positive price required: {value!r}")
        return result

    def get_quote(self, commodity):
        requested = commodity.strip().lower()
        if not requested:
            raise ValueError("Please provide a commodity name.")

        aliases = self.ALIASES.get(requested)
        if aliases is None:
            raise ValueError(
                "Unsupported MCX commodity. Try gold, silver, crude oil, "
                "natural gas, copper, aluminium, zinc, lead, nickel, "
                "mentha oil, cotton or rubber."
            )
        if not aliases:
            raise ValueError(
                f"{requested.title()} is not configured as an MCX commodity."
            )

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"MCX bhavcopy CSV not found: {self.csv_path}. "
                "Download the official MCX bhavcopy as CSV and save it there, "
                "or set MCX_BHAVCOPY_PATH to its location."
            )

        with self.csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("MCX CSV has no header row.")

            columns = {
                self._key(name): name for name in reader.fieldnames
            }

            def column(*names):
                for name in names:
                    if self._key(name) in columns:
                        return columns[self._key(name)]
                return None

            date_col = column("Date", "Trade Date", "Trading Date")
            expiry_col = column("Expiry Date", "Expiry")
            commodity_col = column("Commodity", "Symbol", "Contract")
            instrument_col = column("Instrument", "Instrument Type","Instrument Name")
            close_col = column("Close", "Closing Price", "Close Price")
            unit_col = column("Price Unit", "Unit", "Measure")

            required = {
                "Date": date_col,
                "Expiry Date": expiry_col,
                "Commodity": commodity_col,
                "Instrument": instrument_col,
                "Close": close_col,
            }
            missing = [name for name, col in required.items() if col is None]
            if missing:
                raise ValueError(
                    "MCX CSV is missing required columns: "
                    + ", ".join(missing)
                )

            matches = []
            alias_keys = {self._key(alias) for alias in aliases}

            for row in reader:
                instrument = self._key(row.get(instrument_col, ""))
                if "FUT" not in instrument:
                    continue

                symbol = self._key(row.get(commodity_col, ""))
                if symbol not in alias_keys:
                    continue

                try:
                    trade_date = self._parse_date(row[date_col])
                    expiry = self._parse_date(row[expiry_col])
                except (ValueError, TypeError):
                    continue

                matches.append((trade_date, expiry, symbol, row, unit_col))

        if not matches:
            raise ValueError(
                f"No futures rows for {requested!r} were found in "
                f"{self.csv_path.name}. Check the selected commodity and "
                "ensure the file contains futures rows."
            )

        latest_date = max(item[0] for item in matches)
        latest_rows = [item for item in matches if item[0] == latest_date]

        active_rows = [item for item in latest_rows if item[1] >= latest_date]
        if not active_rows:
            raise ValueError(
                f"No unexpired {requested} futures contract was found "
                f"for the latest data date ({latest_date})."
            )

        # Prefer the nearest expiry; aliases are ordered by contract preference.
        alias_priority = {self._key(alias): i for i, alias in enumerate(aliases)}
        trade_date, expiry, symbol, row, unit_col = min(
            active_rows,
            key=lambda item: (
                item[1],
                alias_priority.get(item[2], len(alias_priority)),
            ),
        )

        price = self._decimal(row[close_col])
        unit = (
            str(row.get(unit_col, "")).strip()
            if unit_col else ""
        ) or "contract price unit"

        contract_col = column("Contract", "Symbol")
        contract = (
            str(row.get(contract_col, "")).strip()
            if contract_col else ""
        ) or symbol

        observed_at = datetime.combine(
            trade_date, datetime.min.time(), tzinfo=timezone.utc
        )

        return MCXFutureQuote(
            commodity=self.DISPLAY_NAMES.get(symbol, requested.title()),
            contract=contract,
            expiry_date=expiry,
            price=price,
            observed_at=observed_at,
            unit=unit,
        )

    
        
    @staticmethod
    def format_quote(quote):
        return (
            f"Market: MCX futures\n"
            f"Commodity: {quote.commodity}\n"
            f"Contract: {quote.contract}\n"
            f"Expiry: {quote.expiry_date.isoformat()}\n"
            f"Closing price: {quote.price} {quote.currency}\n"
            f"Quotation unit: {quote.unit}\n"
            f"Data date: {quote.observed_at.date().isoformat()}\n"
            f"Source: {quote.source}\n"
            f"Unit status: {quote.unit_status}\n"
            f"WARNING: Verify the quotation unit before financial calculations."
        )
