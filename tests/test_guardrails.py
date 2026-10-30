
import pytest

from commodity_finance_assistance.guardrails.engine import (
    GuardrailViolation,
    validate_assistant_output,
    validate_tool_call,
    validate_user_input,
)


ALL_TOOLS = frozenset({
    "get_mcx_quote",
    "calculate_mcx_contract_value",
})


def test_valid_user_input():
    assert validate_user_input("  Explain commodity futures.  ") == (
        "Explain commodity futures."
    )


@pytest.mark.parametrize(
    "message",
    ["", "   ", None, 123],
)
def test_invalid_user_input(message):
    with pytest.raises(GuardrailViolation):
        validate_user_input(message)


def test_user_input_length_limit():
    with pytest.raises(GuardrailViolation):
        validate_user_input("a" * 8001)


def test_user_input_rejects_null_character():
    with pytest.raises(GuardrailViolation):
        validate_user_input("gold\x00quote")


def test_valid_quote_tool_call():
    validate_tool_call(
        "get_mcx_quote",
        {"commodity": "gold"},
        ALL_TOOLS,
    )


def test_valid_calculation_tool_call():
    validate_tool_call(
        "calculate_mcx_contract_value",
        {"commodity": "gold"},
        ALL_TOOLS,
    )


def test_unknown_tool_is_rejected():
    with pytest.raises(GuardrailViolation):
        validate_tool_call(
            "place_trade",
            {"commodity": "gold"},
            ALL_TOOLS,
        )


def test_tool_not_allowed_by_selected_skill():
    with pytest.raises(GuardrailViolation):
        validate_tool_call(
            "calculate_mcx_contract_value",
            {"commodity": "gold"},
            frozenset({"get_mcx_quote"}),
        )


def test_extra_tool_arguments_are_rejected():
    with pytest.raises(GuardrailViolation):
        validate_tool_call(
            "get_mcx_quote",
            {
                "commodity": "gold",
                "execute_trade": True,
            },
            ALL_TOOLS,
        )


def test_invalid_commodity_is_rejected():
    with pytest.raises(GuardrailViolation):
        validate_tool_call(
            "get_mcx_quote",
            {"commodity": "gold; DROP TABLE users"},
            ALL_TOOLS,
        )


def test_valid_assistant_output():
    answer = "The imported gold quote is end-of-day data."
    assert validate_assistant_output(answer) == answer


def test_empty_assistant_output_is_rejected():
    with pytest.raises(GuardrailViolation):
        validate_assistant_output("  ")


def test_secret_in_output_is_rejected():
    with pytest.raises(GuardrailViolation):
        validate_assistant_output(
            "Your key is sk-123456789012345678901234."
        )


def test_unsupported_trade_execution_claim_is_rejected():
    with pytest.raises(GuardrailViolation):
        validate_assistant_output(
            "Your trade has been successfully placed."
        )


def test_explanatory_trade_response_is_allowed():
    answer = "I cannot place or execute trades."
    assert validate_assistant_output(answer) == answer
