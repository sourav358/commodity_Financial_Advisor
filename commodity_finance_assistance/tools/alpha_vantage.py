import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from commodity_finance_assistance.domain.market.models import CommodityQuote


class AlphaVantageProvider:
    BASE_URL = "https://www.alphavantage.co/query"

    METALS = {
        "gold": ("XAU", "Gold"),
        "xau": ("XAU", "Gold"),
        "silver": ("XAG", "Silver"),
        "xag": ("XAG", "Silver"),
    }

    OIL = {
        "crude oil": ("WTI", "WTI Crude Oil"),
        "oil": ("WTI", "WTI Crude Oil"),
        "wti": ("WTI", "WTI Crude Oil"),
        "brent": ("BRENT", "Brent Crude Oil"),
    }

    def __init__(self, api_key=None, timeout=10):
        load_dotenv()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError(
                "ALPHA_VANTAGE_API_KEY is not configured in .env"
            )

    def _request(self, params):
        params["apikey"] = self.api_key
        url = f"{self.BASE_URL}?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": "CommodityFinanceAssistance/0.1"},
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exc:
            raise RuntimeError(
                f"Alpha Vantage returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Could not reach Alpha Vantage"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Alpha Vantage returned invalid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Alpha Vantage returned an unexpected response.")

        if any(
            key in payload
            for key in ("Error Message", "Information", "Note")
        ):
            raise RuntimeError(
                "Alpha Vantage rejected the request or returned an API "
                "limit/access message. Check your API key and quota."
            )

        return payload

    def get_quote(self, commodity):
        normalized = commodity.strip().lower()

        if not normalized:
            raise ValueError("Please provide a commodity name.")

        if normalized in self.METALS:
            symbol, name = self.METALS[normalized]
            payload = self._request({
                "function": "GOLD_SILVER_SPOT",
                "symbol": symbol,
            })

            try:
                nominal = str(payload["nominal"]).upper()
                price = Decimal(str(payload["price"]))
                timestamp_text = str(payload["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Alpha Vantage metal response is missing expected fields."
                ) from exc

            if not nominal.startswith(symbol) or len(nominal) <= 3:
                raise RuntimeError(
                    f"Unexpected metal quote symbol: {nominal}"
                )

            currency = nominal[3:]

            try:
                observed_at = datetime.strptime(
                    timestamp_text, "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError as exc:
                raise RuntimeError(
                    "Alpha Vantage returned an invalid metal timestamp."
                ) from exc

            unit = "troy ounce"
            source = "Alpha Vantage — spot reference"

        elif normalized in self.OIL:
            function, name = self.OIL[normalized]
            payload = self._request({
                "function": function,
                "interval": "daily",
            })

            try:
                observations = payload["data"]
                valid_rows = [
                    row for row in observations
                    if row.get("value") not in (None, ".", "")
                ]
                latest = max(
                    valid_rows,
                    key=lambda row: row["date"],
                )
                price = Decimal(str(latest["value"]))
                observed_at = datetime.strptime(
                    latest["date"], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Alpha Vantage oil response is missing expected daily "
                    "price data."
                ) from exc

            currency = "USD"
            unit = "barrel"
            source = f"Alpha Vantage — {function} daily benchmark"

        else:
            raise ValueError(
                "Unsupported commodity. Supported: gold, silver, crude oil, "
                "oil, WTI, brent."
            )

        if not price.is_finite() or price <= 0:
            raise RuntimeError("Alpha Vantage returned an invalid price.")

        return CommodityQuote(
            commodity=name,
            price=price,
            currency=currency,
            unit=unit,
            observed_at=observed_at,
            source=source,
        )
