"""Tests for the Telegram Chat ID Finder workflow."""

import pytest
from orcheo.graph import END, START
from orcheo.graph.state import State
from tests.conftest import load_workflow_module


workflow = load_workflow_module("news_desk/telegram_chat_id_finder")


def _state_with_updates(updates: list) -> State:
    return State(
        {
            "results": {
                "fetch_updates": {"json": {"ok": True, "result": updates}},
            }
        }
    )


class TestFormatTelegramChatIdNode:
    """Tests for the getUpdates response formatter."""

    async def test_raises_when_telegram_response_is_not_ok(self) -> None:
        """A non-dict HTTP result is treated as an empty, failing payload."""
        node = workflow.FormatTelegramChatIdNode(name="get_chat_id")
        state = State({"results": {"fetch_updates": "not-a-dict"}})

        with pytest.raises(ValueError, match="Telegram API returned an error"):
            await node.run(state, {})

    async def test_raises_when_json_payload_is_not_a_dict(self) -> None:
        """A non-dict ``json`` field on an otherwise dict HTTP result fails too."""
        node = workflow.FormatTelegramChatIdNode(name="get_chat_id")
        state = State({"results": {"fetch_updates": {"json": "oops"}}})

        with pytest.raises(ValueError, match="Telegram API returned an error"):
            await node.run(state, {})

    async def test_reports_no_match_when_updates_is_not_a_list(self) -> None:
        """A malformed (non-list) ``result`` field yields the not-found reply."""
        node = workflow.FormatTelegramChatIdNode(name="get_chat_id")
        state = State(
            {
                "results": {
                    "fetch_updates": {"json": {"ok": True, "result": "oops"}},
                }
            }
        )

        result = await node.run(state, {})

        assert "couldn't find" in result["assistant_message"]
        assert result["results"]["get_chat_id"]["chat_id"] is None
        assert result["results"]["get_chat_id"]["update_count"] == 0

    async def test_skips_non_dict_and_unmatched_updates_before_finding_chat(
        self,
    ) -> None:
        """Non-dict entries and mismatched chat types are skipped in order.

        Updates are scanned newest-first (``reversed``), so the desired match
        is placed at index 0 and everything scanned before it (indexes 1-2)
        must fail to match: a chat-bearing update whose type doesn't match
        (``my_chat_member``/group), a recognised key with no usable ``chat``
        dict, and a non-dict update entirely.
        """
        updates = [
            {
                "message": {
                    "chat": {
                        "type": "private",
                        "id": 444,
                        "first_name": "Ada",
                        "last_name": "Lovelace",
                        "username": "ada",
                    }
                }
            },
            {"my_chat_member": {"chat": {"type": "group", "id": 222}}},
            {"edited_message": {"not_chat": "x"}},
            "not-a-dict",
        ]
        node = workflow.FormatTelegramChatIdNode(
            name="get_chat_id", chat_type="private"
        )
        state = _state_with_updates(updates)

        result = await node.run(state, {})

        payload = result["results"]["get_chat_id"]
        assert payload["chat_id"] == 444
        assert payload["first_name"] == "Ada"
        assert "👤 Name: Ada Lovelace" in result["assistant_message"]
        assert "🔗 Username: @ada" in result["assistant_message"]

    async def test_prefers_title_over_first_and_last_name(self) -> None:
        """A group/channel title is used as the display name when present."""
        updates = [
            {
                "message": {
                    "chat": {
                        "type": "private",
                        "id": 5,
                        "title": "  Support Room  ",
                        "first_name": "Ignored",
                    }
                }
            }
        ]
        node = workflow.FormatTelegramChatIdNode(name="get_chat_id")
        state = _state_with_updates(updates)

        result = await node.run(state, {})

        assert "👤 Name: Support Room" in result["assistant_message"]

    async def test_falls_back_to_username_when_no_name_fields_present(self) -> None:
        """The username is used as the display name absent title/first/last."""
        updates = [
            {"message": {"chat": {"type": "private", "id": 6, "username": "solo"}}}
        ]
        node = workflow.FormatTelegramChatIdNode(name="get_chat_id")
        state = _state_with_updates(updates)

        result = await node.run(state, {})

        assert "👤 Name: @solo" in result["assistant_message"]
        assert "🔗 Username: @solo" in result["assistant_message"]

    async def test_omits_name_and_username_lines_when_entirely_anonymous(
        self,
    ) -> None:
        """No name/username fields means both display lines are omitted."""
        updates = [{"message": {"chat": {"type": "private", "id": 7}}}]
        node = workflow.FormatTelegramChatIdNode(name="get_chat_id")
        state = _state_with_updates(updates)

        result = await node.run(state, {})

        assert "👤 Name" not in result["assistant_message"]
        assert "🔗 Username" not in result["assistant_message"]
        assert "✅ Found your Telegram chat ID!" in result["assistant_message"]


async def test_orcheo_workflow_builds_the_lookup_pipeline() -> None:
    """The workflow loads the token, calls getUpdates, then formats the reply."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "load_telegram_token",
        "fetch_updates",
        "get_chat_id",
    }
    assert graph.edges == {
        (START, "load_telegram_token"),
        ("load_telegram_token", "fetch_updates"),
        ("fetch_updates", "get_chat_id"),
        ("get_chat_id", END),
    }
