
from decimal import Decimal

import pytest

from commodity_finance_assistance.tools.mcx_calculations import (
    calculate_contract_value,
)


def test_unverified_contract_blocks_calculation(tmp_path):
    config = tmp_path / "mcx_contracts.yml"
    config.write_text(
        """
contracts:
  GOLD:
    price_unit: INR per 10 grams
    trading_unit: 1 kilogram
    quoted_quantity: 10
    quoted_quantity_unit: grams
    trading_quantity: 1000
    trading_quantity_unit: grams
    verified: false
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unverified"):
        calculate_contract_value(
            "GOLD", "142024.00", metadata_path=config
        )


def test_verified_contract_value_calculation(tmp_path):
    config = tmp_path / "mcx_contracts.yml"
    config.write_text(
        """
contracts:
  GOLD:
    price_unit: INR per 10 grams
    trading_unit: 1 kilogram
    quoted_quantity: 10
    quoted_quantity_unit: grams
    trading_quantity: 1000
    trading_quantity_unit: grams
    verified: true
    source: MCX contract specification
    effective_date: "2026-10-01"
        """,
        encoding="utf-8",
    )

    value = calculate_contract_value(
        "GOLD", "142024.00", metadata_path=config
    )

    assert value == Decimal("14202400.00")


def test_incompatible_units_block_calculation(tmp_path):
    config = tmp_path / "mcx_contracts.yml"
    config.write_text(
        """
contracts:
  GOLD:
    price_unit: INR per 10 grams
    trading_unit: 1 kilogram
    quoted_quantity: 10
    quoted_quantity_unit: grams
    trading_quantity: 1000
    trading_quantity_unit: kilograms
    verified: true
    source: MCX contract specification
    effective_date: "2026-10-01"
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same unit"):
        calculate_contract_value(
            "GOLD", "142024.00", metadata_path=config
        )
