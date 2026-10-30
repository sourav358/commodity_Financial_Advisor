
import json
import sqlite3
import logging

from commodity_finance_assistance.memory.database import get_connection
from commodity_finance_assistance.tools.mcx_bhavcopy import (
    MCXBhavcopyProvider,
)
from commodity_finance_assistance.tools.mcx_calculations import (
    calculate_contract_value,
)
from commodity_finance_assistance.tools.mcx_contract_metadata import (
    load_contract_metadata,
    is_contract_metadata_verified,
)
from commodity_finance_assistance.tools.mcx_importer import get_latest_quote


logger = logging.getLogger(__name__)


AGENT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_mcx_quote",
            "description": (
                "Retrieve the latest imported end-of-day MCX futures quote "
                "for a supported commodity. This is not a live quote."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": (
                            "Commodity name, for example gold, silver, "
                            "crude oil, or natural gas."
                        ),
                    }
                },
                "required": ["commodity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_mcx_contract_value",
            "description": (
                "Calculate notional contract value using the latest "
                "imported MCX quote. Refuses to calculate unless the "
                "contract metadata and units are verified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Supported MCX commodity name.",
                    }
                },
                "required": ["commodity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def execute_agent_tool(name: str, arguments: dict) -> str:
    """Execute only explicitly allowlisted, validated tool requests."""
    if not isinstance(arguments, dict):
        return json.dumps({"error": "Tool arguments must be an object."})

    commodity = arguments.get("commodity")
    if not isinstance(commodity, str) or not commodity.strip():
        return json.dumps({"error": "A non-empty commodity is required."})

    if name not in {
        "get_mcx_quote",
        "calculate_mcx_contract_value",
    }:
        return json.dumps({"error": "Tool is not allowed."})

    try:
        with get_connection() as connection:
            quote = get_latest_quote(connection, commodity)

        if quote is None:
            return json.dumps({
                "error": f"No eligible imported MCX quote found for {commodity}."
            })

        if name == "get_mcx_quote":
            return MCXBhavcopyProvider.format_quote(quote)

        metadata = load_contract_metadata(quote.contract)
        if not is_contract_metadata_verified(metadata):
            return json.dumps({
                "error": (
                    f"Calculation blocked for {quote.contract}: "
                    "contract quotation and trading-unit metadata "
                    "is not verified."
                ),
                "contract": quote.contract,
                "unit_status": "UNVERIFIED",
                "quote_date": quote.observed_at.date().isoformat(),
            })

        value = calculate_contract_value(
            quote.contract,
            quote.price,
        )

        return json.dumps({
            "commodity": quote.commodity,
            "contract": quote.contract,
            "closing_price": str(quote.price),
            "quotation_unit": metadata["price_unit"],
            "trading_unit": metadata["trading_unit"],
            "contract_value": str(value),
            "currency": quote.currency,
            "data_date": quote.observed_at.date().isoformat(),
            "source": quote.source,
            "unit_status": "VERIFIED",
            "note": "Notional value; not margin or profit/loss.",
        })

    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        logger.warning(
            "Agent tool failed | tool=%s | error_type=%s",
            name,
            type(exc).__name__,
        )
        return json.dumps({
            "error": "The tool could not complete the request.",
            "error_type": type(exc).__name__,
            "retryable": isinstance(exc, sqlite3.Error),
        })

    except Exception:
        logger.exception(
            "Unexpected agent tool failure | tool=%s",
            name,
        )
        return json.dumps({
            "error": "An unexpected internal error occurred while executing the tool.",
            "error_type": "InternalToolError",
            "retryable": False,
        })