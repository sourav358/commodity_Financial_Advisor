from typing import Any

from commodity_finance_assistance.config_loader import (
    get_model_config,
    load_config,
)
from commodity_finance_assistance.llm.client import get_openai_client
from commodity_finance_assistance.memory.local_memory import LocalMemory


class LLMService:
    """Coordinate model calls with persistent conversation memory."""

    def __init__(
        self,
        client: Any = None,
        memory: LocalMemory | None = None,
    ) -> None:
        self._client = client
        self.memory = memory or LocalMemory()

    def start_conversation(self, title: str = "") -> str:
        """Create a conversation and return its ID."""
        return self.memory.create_conversation(title)

    def reply(
        self,
        conversation_id: str,
        user_message: str,
        model_role: str = "chat",
    ) -> str:
        """Send a message to the configured model and save the reply."""
        if not user_message.strip():
            raise ValueError("User message cannot be empty")

        model_config = get_model_config(model_role)

        provider = model_config.get("provider", "openai")
        if provider != "openai":
            raise ValueError(f"Unsupported provider: {provider}")

        config = load_config()
        memory_limit = config.get("memory", {}).get(
            "conversation", {}
        ).get("max_messages", 20)

        self.memory.add_message(
            conversation_id, "user", user_message
        )

        history = self.memory.get_messages(
            conversation_id, limit=memory_limit
        )

        system_prompt = (
            "You are a finance research assistant. "
            "Distinguish verified data from assumptions. "
            "Do not invent market prices or present uncertain "
            "information as fact."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *history,
        ]

        if self._client is None:
            self._client = get_openai_client()

        response = self._client.chat.completions.create(
            model=model_config["model"],
            messages=messages,
            temperature=model_config.get("temperature", 0.2),
            max_tokens=model_config.get("max_tokens", 2000),
        )

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise RuntimeError("The model returned an empty response")

        self.memory.add_message(
            conversation_id, "assistant", answer
        )

        return answer
 
