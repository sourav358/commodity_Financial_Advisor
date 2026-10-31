from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Collection


MAX_USER_INPUT_LENGTH = 8_000
MAX_ASSISTANT_OUTPUT_LENGTH = 12_000
MAX_COMMODITY_LENGTH = 80
MAX_APPROVAL_REASON_LENGTH = 2_000
MAX_APPROVAL_PAYLOAD_BYTES = 8_192

QUOTE_TOOL = "get_mcx_quote"
CALCULATION_TOOL = "calculate_mcx_contract_value"
APPROVAL_TOOL = "request_human_approval"

ALLOWED_TOOL_NAMES = {
    QUOTE_TOOL,
    CALCULATION_TOOL,
    APPROVAL_TOOL,
}

ALLOWED_APPROVAL_ACTION_TYPES = frozenset({
    "risk_limit_change",
    "portfolio_rebalance",
    "trade_execution_request",
})

COMMODITY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}$")

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bOPENAI_API_KEY\s*[:=]\s*\S+", re.IGNORECASE),
)

UNSUPPORTED_ACTION_PATTERNS = (
    re.compile(
        r"\b(?:your|the)\s+(?:trade|order)\s+"
        r"(?:has been|was|is now)\s+"
        r"(?:successfully\s+)?"
        r"(?:placed|executed|submitted|cancelled|completed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:successfully|success)\s+"
        r"(?:placed|executed|submitted|cancelled|completed)\s+"
        r"(?:your\s+)?(?:trade|order)\b",
        re.IGNORECASE,
    ),
)


class GuardrailViolation(ValueError):
    """Raised when an input or action violates application policy."""


def validate_user_input(message: str) -> str:
    """Validate and normalize a user message."""
    if not isinstance(message, str):
        raise GuardrailViolation("Message must be text.")
    message = message.strip()
    if not message:
        raise GuardrailViolation("Message cannot be empty.")
    if len(message) > MAX_USER_INPUT_LENGTH:
        raise GuardrailViolation(
            f"Message exceeds the {MAX_USER_INPUT_LENGTH}-character limit."
        )
    for character in message:
        if character == "\x00":
            raise GuardrailViolation("Message contains an invalid null character.")
        if (
            unicodedata.category(character) == "Cc"
            and character not in "\n\r\t"
        ):
            raise GuardrailViolation(
                "Message contains unsupported control characters."
            )
    return message


def _validate_approval_arguments(arguments: dict[str, Any]) -> None:
    required_keys = {"action_type", "reason", "payload_json"}
    optional_keys = {"idempotency_key"}

    supplied_keys = set(arguments)

    if not required_keys.issubset(supplied_keys):
        raise GuardrailViolation(
            "Approval tool requires action_type, reason, and payload_json."
        )

    if supplied_keys - required_keys - optional_keys:
        raise GuardrailViolation(
            "Approval tool received unsupported arguments."
        )

    action_type = arguments["action_type"]

    if (
        not isinstance(action_type, str)
        or action_type not in ALLOWED_APPROVAL_ACTION_TYPES
    ):
        raise GuardrailViolation("Approval action type is not permitted.")

    reason = arguments["reason"]

    if not isinstance(reason, str) or not reason.strip():
        raise GuardrailViolation("Approval reason must be non-empty text.")

    if len(reason.strip()) > MAX_APPROVAL_REASON_LENGTH:
        raise GuardrailViolation("Approval reason is too long.")

    if "idempotency_key" in arguments:
        key = arguments["idempotency_key"]

        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", key)
        ):
            raise GuardrailViolation("Invalid approval idempotency key.")

    payload_json = arguments["payload_json"]

    if not isinstance(payload_json, str) or not payload_json.strip():
        raise GuardrailViolation(
            "Approval payload_json must be non-empty JSON text."
        )

    try:
        payload = json.loads(payload_json)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise GuardrailViolation(
            "Approval payload_json must contain valid JSON."
        ) from exc

    if not isinstance(payload, dict) or not payload:
        raise GuardrailViolation(
            "Approval payload must decode to a non-empty object."
        )

    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise GuardrailViolation(
            "Approval payload must contain valid JSON values."
        ) from exc

    if len(serialized.encode("utf-8")) > MAX_APPROVAL_PAYLOAD_BYTES:
        raise GuardrailViolation("Approval payload exceeds the 8 KB limit.")

def validate_tool_call(
    name: str,
    arguments: dict[str, Any],
    allowed_tools: Collection[str],
) -> None:
    """Enforce global allowlisting, skill permissions, and strict arguments."""
    if name not in ALLOWED_TOOL_NAMES:
        raise GuardrailViolation("Tool is not permitted by application policy.")
    if name not in allowed_tools:
        raise GuardrailViolation("Tool is not permitted by the selected skill.")
    if not isinstance(arguments, dict):
        raise GuardrailViolation("Tool arguments must be an object.")

    if name == APPROVAL_TOOL:
        _validate_approval_arguments(arguments)
        return

    if set(arguments) != {"commodity"}:
        raise GuardrailViolation(
            "Tool must receive exactly one commodity argument."
        )

    commodity = arguments["commodity"]
    if not isinstance(commodity, str):
        raise GuardrailViolation("Commodity must be a string.")
    commodity = commodity.strip()
    if not commodity:
        raise GuardrailViolation("Commodity cannot be empty.")
    if len(commodity) > MAX_COMMODITY_LENGTH:
        raise GuardrailViolation("Commodity name is too long.")
    if not COMMODITY_PATTERN.fullmatch(commodity):
        raise GuardrailViolation("Commodity contains invalid characters.")


def validate_assistant_output(output: str) -> str:
    """Validate output before it is saved or returned to the user."""
    if not isinstance(output, str) or not output.strip():
        raise GuardrailViolation("Assistant returned an empty response.")
    output = output.strip()
    if len(output) > MAX_ASSISTANT_OUTPUT_LENGTH:
        raise GuardrailViolation(
            "Assistant response exceeded the output length limit."
        )
    if any(pattern.search(output) for pattern in SECRET_PATTERNS):
        raise GuardrailViolation(
            "Assistant response was blocked by the secret disclosure guardrail."
        )
    if any(pattern.search(output) for pattern in UNSUPPORTED_ACTION_PATTERNS):
        raise GuardrailViolation(
            "Assistant response claimed an unsupported financial action."
        )
    return output
