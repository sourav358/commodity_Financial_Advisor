
from pathlib import Path

import yaml
from decimal import Decimal, InvalidOperation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "config" / "mcx_contracts.yml"


def load_contract_metadata(
    symbol: str,
    path: Path = DEFAULT_METADATA_PATH,
) -> dict:
    """Load metadata for a contract symbol."""
    with path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file) or {}

    contracts = document.get("contracts", {})
    metadata = contracts.get(symbol.strip().upper())

    if metadata is None:
        return {
            "verified": False,
            "price_unit": "",
            "trading_unit": "",
            "source": "",
            "effective_date": "",
        }

    return metadata



def is_contract_metadata_verified(metadata: dict) -> bool:
    """Verify metadata completeness before permitting calculations."""
    if metadata.get("verified") is not True:
        return False

    required_text = (
        "price_unit",
        "trading_unit",
        "source",
        "effective_date",
    )

    if any(
        not isinstance(metadata.get(key), str)
        or not metadata[key].strip()
        for key in required_text
    ):
        return False

    required_positive_numbers = (
        "quoted_quantity",
        "trading_quantity",
    )

    for key in required_positive_numbers:
        try:
            value = Decimal(str(metadata[key]))
        except (KeyError, InvalidOperation, TypeError, ValueError):
            return False

        if not value.is_finite() or value <= 0:
            return False

    if (
        not isinstance(metadata.get("quoted_quantity_unit"), str)
        or not metadata["quoted_quantity_unit"].strip()
        or not isinstance(metadata.get("trading_quantity_unit"), str)
        or not metadata["trading_quantity_unit"].strip()
    ):
        return False

    return True

    """Return True only when required metadata is complete."""
    if metadata.get("verified") is not True:
        return False

    required_text = (
        "price_unit",
        "trading_unit",
        "source",
        "effective_date",
    )

    if any(
        not isinstance(metadata.get(key), str)
        or not metadata[key].strip()
        for key in required_text
    ):
        return False

    return True
