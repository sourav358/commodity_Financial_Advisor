
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from commodity_finance_assistance.memory.database import (
    initialize_database,
)
from commodity_finance_assistance.memory.local_memory import LocalMemory


class TestLocalMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        self.path_patcher = patch(
            "commodity_finance_assistance.memory.database.get_database_path",
            return_value=self.db_path,
        )
        self.path_patcher.start()

        initialize_database()
        self.memory = LocalMemory()
        self.conversation_id = self.memory.create_conversation(
            "Test conversation"
        )

    def tearDown(self):
        self.path_patcher.stop()
        self.temp_dir.cleanup()

    def test_message_persists_across_connections(self):
        self.memory.add_message(
            self.conversation_id, "user", "Analyze gold prices"
        )

        messages = self.memory.get_messages(self.conversation_id)

        self.assertEqual(
            messages,
            [{"role": "user", "content": "Analyze gold prices"}],
        )

    def test_message_limit_returns_latest_messages_in_order(self):
        for content in ["first", "second", "third"]:
            self.memory.add_message(
                self.conversation_id, "user", content
            )

        messages = self.memory.get_messages(
            self.conversation_id, limit=2
        )

        self.assertEqual(
            [message["content"] for message in messages],
            ["second", "third"],
        )

    def test_invalid_role_is_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.add_message(
                self.conversation_id, "unknown", "Invalid message"
            )

    def test_unknown_conversation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.add_message(
                "missing-conversation", "user", "Hello"
            )

    def test_non_positive_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.get_messages(self.conversation_id, limit=0)


if __name__ == "__main__":
    unittest.main()
 
