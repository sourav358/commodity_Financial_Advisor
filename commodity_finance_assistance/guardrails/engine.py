
from __future__ import annotations

import re
import unicodedata
from typing import Any, Collection



MAX_USER_INPUT_LENGTH = 8_000
MAX_ASSISTANT_OUTPUT_LENGTH = 12_000
MAX_COMMODITY_LENGTH = 80

ALLOWED_TOOL_NAMES = {
    "get_mcx_quote",
    "calculate_mcx_contract_value",
}

COMMODITY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}$"
)

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"\bOPENAI_API_KEY\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
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
            raise GuardrailViolation(
                "Message contains an invalid null character."
            )

        if (
            unicodedata.category(character) == "Cc"
            and character not in "\n\r\t"
        ):
            raise GuardrailViolation(
                "Message contains unsupported control characters."
            )

    return message


def validate_tool_call(
    name: str,
    arguments: dict[str, Any],
    allowed_tools: Collection[str],
) -> None:
    """Enforce tool allowlisting and strict argument validation."""
    if name not in ALLOWED_TOOL_NAMES:
        raise GuardrailViolation("Tool is not permitted by application policy.")

    if name not in allowed_tools:
        raise GuardrailViolation(
            "Tool is not permitted by the selected skill."
        )

    if not isinstance(arguments, dict):
        raise GuardrailViolation("Tool arguments must be an object.")

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

    if any(
        pattern.search(output)
        for pattern in UNSUPPORTED_ACTION_PATTERNS
    ):
        raise GuardrailViolation(
            "Assistant response claimed an unsupported financial action."
        )

    return output