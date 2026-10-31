
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from commodity_finance_assistance.llm.service import LLMService


class TestLLMService(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.memory = MagicMock()
        self.memory.get_messages.return_value = [
            {"role": "user", "content": "Analyze gold"}
        ]

        self.client.chat.completions.create.return_value = (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Gold analysis response"
                        )
                    )
                ]
            )
        )

        self.service = LLMService(
            client=self.client,
            memory=self.memory,
        )

    @patch(
        "commodity_finance_assistance.llm.service.get_model_config"
    )
    @patch(
        "commodity_finance_assistance.llm.service.load_config"
    )
    def test_reply_uses_configured_model_and_saves_answer(
        self, mock_load_config, mock_get_model_config
    ):
        mock_get_model_config.return_value = {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
        mock_load_config.return_value = {
            "memory": {"conversation": {"max_messages": 20}}
        }
        conversation_id = f"llm-test-{uuid4()}"

        answer = self.service.reply(conversation_id, "Analyze gold")

        self.assertEqual(answer, "Gold analysis response")
        self.client.chat.completions.create.assert_called_once()

        call = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(call["model"], "gpt-4.1-mini")
        self.assertEqual(call["temperature"], 0.2)

        self.memory.add_message.assert_any_call(
            conversation_id, "user", "Analyze gold"
        )
        self.memory.add_message.assert_any_call(
            conversation_id, "assistant", "Gold analysis response"
        )

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.reply("conversation-1", "   ")

        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
 
