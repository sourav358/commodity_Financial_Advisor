
from decimal import Decimal, InvalidOperation

from commodity_finance_assistance.tools.mcx_contract_metadata import (
    load_contract_metadata,
    is_contract_metadata_verified,
)


def calculate_contract_value(
    symbol: str,
    quoted_price,
    metadata_path=None,
) -> Decimal:
    """Calculate notional value only for verified, compatible units."""
    metadata = (
        load_contract_metadata(symbol, metadata_path)
        if metadata_path is not None
        else load_contract_metadata(symbol)
    )

    if not is_contract_metadata_verified(metadata):
        raise ValueError(
            f"Contract {symbol} has unverified metadata. "
            "Contract-value calculation is blocked."
        )

    if metadata.get("quoted_quantity_unit") != metadata.get(
        "trading_quantity_unit"
    ):
        raise ValueError(
            "Quoted quantity and trading quantity must use "
            "the same unit before calculating contract value."
        )

    try:
        price = Decimal(str(quoted_price))
        quoted_quantity = Decimal(str(metadata["quoted_quantity"]))
        trading_quantity = Decimal(str(metadata["trading_quantity"]))
    except (InvalidOperation, KeyError, TypeError) as exc:
        raise ValueError("Invalid price or quantity metadata.") from exc

    if (
        not price.is_finite()
        or not quoted_quantity.is_finite()
        or not trading_quantity.is_finite()
        or price < 0
        or quoted_quantity <= 0
        or trading_quantity <= 0
    ):
        raise ValueError("Price and quantities must be valid and non-negative.")

    return price * trading_quantity / quoted_quantity
