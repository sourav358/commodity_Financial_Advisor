import json
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from datetime import datetime

from commodity_finance_assistance.tools.agent_tools import (
    AGENT_TOOL_DEFINITIONS,
    execute_agent_tool,
)


MODULE = "commodity_finance_assistance.tools.agent_tools"


class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.quote = SimpleNamespace(
            contract="GOLD",
            price=Decimal("142024.00"),
            observed_at=datetime(2026, 10, 23),
        )

    def configure_connection(self, mock_get_connection):
        context = MagicMock()
        connection = object()
        context.__enter__.return_value = connection
        mock_get_connection.return_value = context
        return connection

    def test_tool_definitions_are_strict(self):
        names = {
            item["function"]["name"]
            for item in AGENT_TOOL_DEFINITIONS
        }

        self.assertEqual(
            names,
            {
                "get_mcx_quote",
                "calculate_mcx_contract_value",
            },
        )

        for item in AGENT_TOOL_DEFINITIONS:
            function = item["function"]
            schema = function["parameters"]

            self.assertTrue(function.get("strict"))
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["required"], ["commodity"])

    @patch(f"{MODULE}.MCXBhavcopyProvider.format_quote")
    @patch(f"{MODULE}.get_latest_quote")
    @patch(f"{MODULE}.get_connection")
    def test_quote_tool_returns_formatted_quote(
        self,
        mock_get_connection,
        mock_get_latest_quote,
        mock_format_quote,
    ):
        connection = self.configure_connection(mock_get_connection)
        mock_get_latest_quote.return_value = self.quote
        mock_format_quote.return_value = "Formatted MCX GOLD quote"

        result = execute_agent_tool(
            "get_mcx_quote",
            {"commodity": "gold"},
        )

        self.assertEqual(result, "Formatted MCX GOLD quote")
        mock_get_latest_quote.assert_called_once_with(
            connection, "gold"
        )
        mock_format_quote.assert_called_once_with(self.quote)

    @patch(f"{MODULE}.calculate_contract_value")
    @patch(
        f"{MODULE}.is_contract_metadata_verified",
        return_value=False,
    )
    @patch(
        f"{MODULE}.load_contract_metadata",
        return_value={"verified": False},
    )
    @patch(f"{MODULE}.get_latest_quote")
    @patch(f"{MODULE}.get_connection")
    def test_unverified_metadata_blocks_calculation(
        self,
        mock_get_connection,
        mock_get_latest_quote,
        mock_load_metadata,
        mock_is_verified,
        mock_calculate,
    ):
        self.configure_connection(mock_get_connection)
        mock_get_latest_quote.return_value = self.quote

        result = execute_agent_tool(
            "calculate_mcx_contract_value",
            {"commodity": "gold"},
        )

        self.assertIn(
            "verif",
            result.lower(),
            "The result should explain the metadata verification issue.",
        )
        mock_calculate.assert_not_called()

    @patch(f"{MODULE}.get_latest_quote", return_value=None)
    @patch(f"{MODULE}.get_connection")
    def test_missing_quote_returns_error(
        self,
        mock_get_connection,
        mock_get_latest_quote,
    ):
        self.configure_connection(mock_get_connection)

        result = execute_agent_tool(
            "get_mcx_quote",
            {"commodity": "unknown-commodity"},
        )

        self.assertTrue(result.strip())
        self.assertRegex(
            result.lower(),
            r"error|not found|no quote|unavailable",
        )

    @patch(f"{MODULE}.get_connection")
    def test_empty_commodity_is_rejected(self, mock_get_connection):
        result = execute_agent_tool(
            "get_mcx_quote",
            {"commodity": "   "},
        )

        self.assertTrue(result.strip())
        mock_get_connection.assert_not_called()

    @patch(f"{MODULE}.get_connection")
    def test_unknown_tool_is_rejected(self, mock_get_connection):
        result = execute_agent_tool(
            "delete_everything",
            {"commodity": "gold"},
        )

        self.assertTrue(result.strip())
        self.assertRegex(
            result.lower(),
            r"unknown|unsupported|not allowed|invalid tool",
        )
        mock_get_connection.assert_not_called()
    @patch(
        "commodity_finance_assistance.tools.agent_tools.get_connection",
        side_effect=Exception("sensitive internal detail"),
    )
    def test_unexpected_error_returns_safe_response(self, mock_get_connection):
        result = execute_agent_tool(
            "get_mcx_quote",
            {"commodity": "gold"},
        )

        payload = json.loads(result)

        self.assertEqual(
            payload["error_type"],
            "InternalToolError",
        )
        self.assertNotIn(
            "sensitive internal detail",
            result,
        )
        self.assertFalse(payload["retryable"])

    @patch(
        "commodity_finance_assistance.tools.agent_tools.get_connection",
        side_effect=__import__("sqlite3").OperationalError(
            "sensitive database detail"
        ),
    )
    def test_database_error_returns_controlled_response(
        self, mock_get_connection
    ):
        result = execute_agent_tool(
            "get_mcx_quote",
            {"commodity": "gold"},
        )

        payload = json.loads(result)

        self.assertEqual(
            payload["error"],
            "The tool could not complete the request.",
        )
        self.assertEqual(
            payload["error_type"],
            "OperationalError",
        )
        self.assertTrue(payload["retryable"])
        self.assertNotIn(
            "sensitive database detail",
            result,
        )


if __name__ == "__main__":
    unittest.main()