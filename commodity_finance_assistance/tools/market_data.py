
from typing import Protocol

from commodity_finance_assistance.domain.market.models import CommodityQuote


class MarketDataProvider(Protocol):
    def get_quote(self, commodity: str) -> CommodityQuote | None:
        ...


class UnavailableMarketDataProvider:
    def get_quote(self, commodity: str) -> None:
        return None


class MarketDataTool:
    def __init__(self, provider=None):
        # No implicit network calls: use an unavailable provider unless
        # a concrete provider is explicitly supplied.
        self.provider = provider or UnavailableMarketDataProvider()

    def get_quote(self, commodity: str) -> CommodityQuote | None:
        normalized = commodity.strip().lower()

        if not normalized:
            raise ValueError("Commodity name cannot be empty.")

        return self.provider.get_quote(normalized)

    @staticmethod
    def format_quote(quote: CommodityQuote) -> str:
        return (
            f"Commodity: {quote.commodity}\n"
            f"Price: {quote.price} {quote.currency}/{quote.unit}\n"
            f"Observed at: {quote.observed_at.isoformat()}\n"
            f"Source: {quote.source}"
        )
