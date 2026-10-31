import json

import pytest

from commodity_finance_assistance.guardrails.engine import (
    APPROVAL_TOOL,
    GuardrailViolation,
    validate_tool_call,
)
from commodity_finance_assistance.skills.registry import (
    APPROVAL_TOOL as SKILL_APPROVAL_TOOL,
    SkillRegistry,
)


def valid_approval_arguments():
    return {
        "action_type": "risk_limit_change",
        "reason": "A human must review the proposed risk limit change.",
        "payload_json": json.dumps({
            "commodity": "GOLD",
            "current_limit": 100000,
            "proposed_limit": 125000,
        }),
    }


def test_approval_tool_is_exposed_only_for_explicit_approval_request():
    registry = SkillRegistry()

    selection = registry.select("Please submit this trade for approval.")
    assert "approval_workflow" in selection.names
    assert SKILL_APPROVAL_TOOL in selection.allowed_tools

    general = registry.select("Explain what human approval means in trading.")
    assert SKILL_APPROVAL_TOOL not in general.allowed_tools


def test_valid_approval_tool_call_passes_guardrails():
    validate_tool_call(
        APPROVAL_TOOL,
        valid_approval_arguments(),
        {APPROVAL_TOOL},
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"action_type": "unknown", "reason": "Review", "payload": {"x": 1}},
        {"action_type": "risk_limit_change", "reason": "", "payload": {"x": 1}},
        {"action_type": "risk_limit_change", "reason": "Review", "payload_json": '{}'},
        {"action_type": "risk_limit_change", "reason": "Review", "payload_json": '{"x": object}'},
        {"action_type": "risk_limit_change", "reason": "Review", "payload_json": '{"x": NaN}'},
        {"action_type": "risk_limit_change", "reason": "Review", "payload_json": json.dumps({"x": "a" * 9000})},
        {"action_type": "risk_limit_change", "reason": "Review", "payload_json": '{"x": 1}', "requester_id": "spoofed"},
    ],
)
def test_invalid_approval_arguments_are_rejected(arguments):
    with pytest.raises(GuardrailViolation):
        validate_tool_call(APPROVAL_TOOL, arguments, {APPROVAL_TOOL})


def test_approval_tool_is_rejected_when_skill_does_not_allow_it():
    with pytest.raises(GuardrailViolation):
        validate_tool_call(
            APPROVAL_TOOL,
            valid_approval_arguments(),
            {"get_mcx_quote"},
        )


def test_approval_tool_is_in_global_allowlist():
    assert APPROVAL_TOOL in {
        "get_mcx_quote",
        "calculate_mcx_contract_value",
        "request_human_approval",
    }
def test_valid_idempotency_key_passes_guardrails():
    arguments = valid_approval_arguments()
    arguments["idempotency_key"] = "draft-key-123"

    validate_tool_call(
        APPROVAL_TOOL,
        arguments,
        {APPROVAL_TOOL},
    )


@pytest.mark.parametrize(
    "key",
    [
        "",
        "contains spaces",
        "invalid/key",
        "x" * 129,
    ],
)
def test_invalid_idempotency_key_is_rejected(key):
    arguments = valid_approval_arguments()
    arguments["idempotency_key"] = key

    with pytest.raises(GuardrailViolation):
        validate_tool_call(
            APPROVAL_TOOL,
            arguments,
            {APPROVAL_TOOL},
        )