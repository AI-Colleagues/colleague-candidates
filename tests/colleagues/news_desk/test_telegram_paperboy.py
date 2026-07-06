"""Tests for the Telegram Paperboy workflow."""

from orcheo.graph import END, START
from orcheo.graph.state import State
from tests.conftest import load_workflow_module


workflow = load_workflow_module("news_desk/telegram_paperboy")


class TestDetectTriggerNode:
    """Tests for the inbound-vs-scheduled trigger detector."""

    async def test_detects_listener_payload(self) -> None:
        """A ``listener`` key in inputs marks the run as inbound."""
        node = workflow.DetectTriggerNode(name="detect_trigger")
        state = State({"inputs": {"listener": {"chat_id": 1}}})

        result = await node.run(state, {})

        assert result == {"is_listener": True}

    async def test_detects_platform_payload(self) -> None:
        """A ``platform`` key in inputs also marks the run as inbound."""
        node = workflow.DetectTriggerNode(name="detect_trigger")
        state = State({"inputs": {"platform": "telegram"}})

        result = await node.run(state, {})

        assert result["is_listener"] is True

    async def test_scheduled_run_has_no_listener_or_platform(self) -> None:
        """Absent listener/platform keys mean a scheduled (cron) run."""
        node = workflow.DetectTriggerNode(name="detect_trigger")
        state = State({"inputs": {}})

        result = await node.run(state, {})

        assert result["is_listener"] is False

    async def test_non_dict_inputs_are_treated_as_scheduled(self) -> None:
        """Non-dict ``inputs`` is treated as a scheduled run."""
        node = workflow.DetectTriggerNode(name="detect_trigger")
        state = State({"inputs": "oops"})

        result = await node.run(state, {})

        assert result["is_listener"] is False


class TestFormatDigestNode:
    """Tests for the RSS digest formatter."""

    async def test_reports_no_updates_when_results_missing(self) -> None:
        """A missing ``node_results`` dict yields the no-updates message."""
        node = workflow.FormatDigestNode(name="format_digest")
        state = State({})

        result = await node.run(state, {})

        assert result["content"] == "Today's RSS News:\n\nNo news updates today."
        assert result["ids"] == []
        assert result["has_items"] is False

    async def test_reports_no_updates_when_escaped_result_is_not_a_list(self) -> None:
        """A non-list ``result`` field also yields the no-updates message."""
        node = workflow.FormatDigestNode(name="format_digest")
        state = State(
            {"node_results": {"escape_titles": {"result": "oops"}}},
        )

        result = await node.run(state, {})

        assert result["has_items"] is False

    async def test_non_dict_results_and_escape_titles_are_tolerated(self) -> None:
        """A non-dict ``node_results`` or ``escape_titles`` value is treated as empty."""
        node = workflow.FormatDigestNode(name="format_digest")

        state_bad_results = State({"node_results": "oops"})
        result = await node.run(state_bad_results, {})
        assert result["has_items"] is False

        state_bad_escape_titles = State({"node_results": {"escape_titles": "oops"}})
        result = await node.run(state_bad_escape_titles, {})
        assert result["has_items"] is False

    async def test_builds_digest_lines_with_and_without_links(self) -> None:
        """Items with a link render as anchors; items without render as text."""
        node = workflow.FormatDigestNode(name="format_digest")
        state = State(
            {
                "node_results": {
                    "escape_titles": {
                        "result": [
                            "not-a-dict",
                            {"_id": "a1", "title": "Has A Link", "link": "https://x"},
                            {"_id": "a2", "link": ""},
                            {"title": "No Id Here"},
                        ]
                    }
                }
            }
        )

        result = await node.run(state, {})

        assert result["ids"] == ["a1", "a2"]
        assert result["has_items"] is True
        assert '<a href="https://x">Has A Link</a>' in result["content"]
        assert "- No Title" in result["content"]
        assert "- No Id Here" in result["content"]


class TestResolveTargetChatNode:
    """Tests for resolving which chat receives the digest."""

    async def test_uses_reply_target_chat_id_when_present(self) -> None:
        """An inbound reply target takes priority over the default chat."""
        node = workflow.ResolveTargetChatNode(
            name="resolve_target", default_chat_id="default-chat"
        )
        state = State(
            {
                "node_results": {
                    "telegram_listener": {
                        "reply_target": {"chat_id": 999},
                    }
                }
            }
        )

        result = await node.run(state, {})

        assert result["chat_id"] == "999"

    async def test_falls_back_to_listener_chat_id(self) -> None:
        """Without a reply target, the listener's own chat_id is used."""
        node = workflow.ResolveTargetChatNode(
            name="resolve_target", default_chat_id="default-chat"
        )
        state = State(
            {"node_results": {"telegram_listener": {"chat_id": 555}}},
        )

        result = await node.run(state, {})

        assert result["chat_id"] == "555"

    async def test_falls_back_to_default_chat_id(self) -> None:
        """Absent any listener context, the configured default chat is used."""
        node = workflow.ResolveTargetChatNode(
            name="resolve_target", default_chat_id="default-chat"
        )
        state = State({"node_results": {}})

        result = await node.run(state, {})

        assert result["chat_id"] == "default-chat"

    async def test_non_dict_results_and_listener_are_tolerated(self) -> None:
        """A non-dict ``node_results`` or ``telegram_listener`` value is empty."""
        node = workflow.ResolveTargetChatNode(
            name="resolve_target", default_chat_id="default-chat"
        )

        state_bad_results = State({"node_results": "oops"})
        result = await node.run(state_bad_results, {})
        assert result["chat_id"] == "default-chat"

        state_bad_listener = State({"node_results": {"telegram_listener": "oops"}})
        result = await node.run(state_bad_listener, {})
        assert result["chat_id"] == "default-chat"


async def test_orcheo_workflow_builds_the_digest_pipeline() -> None:
    """The workflow branches on trigger type and gates delivery on unread items."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "detect_trigger",
        "cron_trigger",
        "telegram_listener",
        "find_unread",
        "escape_titles",
        "format_digest",
        "resolve_target",
        "send_news",
        "mark_read",
    }
    assert graph.edges == {
        (START, "detect_trigger"),
        ("cron_trigger", "find_unread"),
        ("find_unread", "escape_titles"),
        ("escape_titles", "format_digest"),
        ("resolve_target", "send_news"),
        ("send_news", "mark_read"),
        ("mark_read", END),
    }
    assert {"detect_trigger", "telegram_listener", "format_digest"} <= set(
        graph.branches.keys()
    )
