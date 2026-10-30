
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal


@dataclass(frozen=True)
class CommodityQuote:
    """A commodity price observation with provenance."""

    commodity: str
    price: Decimal
    currency: str
    unit: str
    observed_at: datetime
    source: str

    def __post_init__(self) -> None:
        if not self.commodity.strip():
            raise ValueError("Commodity name cannot be empty")

        if not self.price.is_finite() or self.price <= 0:
            raise ValueError("Price must be a positive, finite Decimal")

        if not self.currency.strip():
            raise ValueError("Currency cannot be empty")

        if not self.unit.strip():
            raise ValueError("Unit cannot be empty")

        if not self.source.strip():
            raise ValueError("Quote source cannot be empty")

        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
