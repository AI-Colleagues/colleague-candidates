# /// orcheo
# name = "Telegram Paperboy"
# handle = "telegram-paperboy"
# description = "Deliver scheduled RSS news digests through Telegram."
# version = "0.1.0"
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# avatar = "avatar-01"
# subtitle = "Scheduled delivery"
# ///

"""News Desk - Telegram Paperboy workflow.

Sends the latest *unread* RSS news items through Telegram and marks the
delivered items as read. Two entry points share the same digest flow:

- Scheduled: a cron trigger (daily at 9:00 AM Amsterdam time by default)
  broadcasts the digest to the configured ``telegram_chat_id``.
- On demand: a managed Telegram bot listener replies to whoever messages
  the bot with the next batch of unread news.

Configurable inputs (config.json):
- cron_expression (cron schedule, Europe/Amsterdam timezone)
- rss_database (MongoDB database name)
- rss_collection (collection for RSS feed items)
- telegram_chat_id (chat that receives the scheduled broadcast)

Orcheo vault secrets required:
- telegram_token: Telegram bot token
- mdb_connection_string: MongoDB connection string
"""

from orcheo.graph import END, StateGraph
from orcheo.graph.state import State
from orcheo.nodes import CodeNode
from orcheo.nodes.connectors.telegram import (
    MessageTelegramNode,
    TelegramBotListenerNode,
)
from orcheo.nodes.data import HtmlTextTransformNode
from orcheo.nodes.storage.mongodb import MongoDBFindNode, MongoDBUpdateManyNode
from orcheo.nodes.triggers import CronTriggerNode


class DetectTriggerNode(CodeNode):
    """Detect whether the run was started by an inbound listener event."""

    async def run(self, state, config):
        """Return whether a listener payload is present in inputs."""
        inputs = state.get("inputs", {})
        is_listener = bool(
            isinstance(inputs, dict)
            and (inputs.get("listener") or inputs.get("platform"))
        )
        return {"is_listener": is_listener}


class FormatDigestNode(CodeNode):
    """Format the latest unread RSS news items into a digest message."""

    async def run(self, state, config):
        """Return the digest content string and the delivered item IDs."""
        results = state.get("node_results", {})
        if not isinstance(results, dict):
            results = {}
        html_result = results.get("escape_titles", {})
        if not isinstance(html_result, dict):
            html_result = {}
        data = html_result.get("result")
        if isinstance(data, list):
            items = data
        else:
            items = []

        lines = []
        ids = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("_id")
            if item_id is not None:
                ids.append(item_id)
            title = item.get("title")
            if not title:
                title = "No Title"
            url = item.get("link", "")
            if url:
                lines.append('- <a href="' + str(url) + '">' + str(title) + "</a>")
            else:
                lines.append("- " + str(title))

        content = "\n".join(lines) if lines else "No news updates today."
        return {
            "content": "Today's RSS News:\n\n" + content,
            "ids": ids,
            "has_items": bool(ids),
        }


class ResolveTargetChatNode(CodeNode):
    """Pick the chat that receives the digest.

    Inbound messages are answered in the originating chat; scheduled runs
    fall back to the configured broadcast chat.
    """

    default_chat_id: str = "{{config.configurable.telegram_chat_id}}"

    async def run(self, state, config):
        """Return the chat ID from the inbound Telegram listener event."""
        results = state.get("node_results", {})
        if not isinstance(results, dict):
            results = {}
        listener = results.get("telegram_listener", {})
        if not isinstance(listener, dict):
            listener = {}
        reply_target = listener.get("reply_target", {})
        chat_id = (
            reply_target.get("chat_id") if isinstance(reply_target, dict) else None
        ) or listener.get("chat_id")
        if chat_id:
            target = str(chat_id)
        else:
            target = self.default_chat_id
        return {"chat_id": target}


async def orcheo_workflow() -> StateGraph:
    """Build the Telegram Paperboy workflow."""
    graph = StateGraph(State)

    # --- Trigger detection ---
    graph.add_node("detect_trigger", DetectTriggerNode(name="detect_trigger"))
    graph.add_node(
        "cron_trigger",
        CronTriggerNode(
            name="cron_trigger",
            expression="{{config.configurable.cron_expression}}",
            timezone="Europe/Amsterdam",
        ),
    )
    graph.add_node(
        "telegram_listener",
        TelegramBotListenerNode(
            name="telegram_listener",
            token="[[telegram_token]]",
            allowed_updates=["message"],
            allowed_chat_types=["private"],
            poll_timeout_seconds=30,
            bot_identity_key="telegram:primary",
        ),
    )

    # --- Fetch unread items ---
    graph.add_node(
        "find_unread",
        MongoDBFindNode(
            name="find_unread",
            database="{{config.configurable.rss_database}}",
            collection="{{config.configurable.rss_collection}}",
            filter={"read": False},
            sort={"isoDate": -1},
            limit=20,
        ),
    )

    graph.add_node(
        "escape_titles",
        HtmlTextTransformNode(
            name="escape_titles",
            input_data="{{find_unread.data}}",
            operations=["unescape", "normalize_nbsp", "escape"],
            fields=["title"],
        ),
    )

    # --- Format digest ---
    graph.add_node(
        "format_digest",
        FormatDigestNode(name="format_digest"),
    )

    # --- Resolve the target chat ---
    graph.add_node(
        "resolve_target",
        ResolveTargetChatNode(name="resolve_target"),
    )

    # --- Deliver to the resolved chat ---
    graph.add_node(
        "send_news",
        MessageTelegramNode(
            name="send_news",
            token="[[telegram_token]]",
            chat_id="{{resolve_target.chat_id}}",
            message="{{format_digest.content}}",
            parse_mode="HTML",
        ),
    )

    # --- Mark delivered items as read ---
    graph.add_node(
        "mark_read",
        MongoDBUpdateManyNode(
            name="mark_read",
            database="{{config.configurable.rss_database}}",
            collection="{{config.configurable.rss_collection}}",
            filter={"_id": {"$in": "{{format_digest.ids}}"}},
            update={"$set": {"read": True}},
        ),
    )

    # --- Edges ---
    graph.set_entry_point("detect_trigger")

    # Route inbound messages to the listener, scheduled runs to the cron trigger.
    graph.add_conditional_edges(
        "detect_trigger",
        {
            "path": "node_results.detect_trigger.is_listener",
            "mapping": {
                "true": "telegram_listener",
                "false": "cron_trigger",
            },
        },
    )

    graph.add_edge("cron_trigger", "find_unread")

    # Only build a digest for inbound updates that carry a message.
    graph.add_conditional_edges(
        "telegram_listener",
        {
            "path": "node_results.telegram_listener.should_process",
            "mapping": {
                "true": "find_unread",
                "false": END,
            },
        },
    )

    graph.add_edge("find_unread", "escape_titles")
    graph.add_edge("escape_titles", "format_digest")

    # Only send (and mark read) when there are unread items to deliver.
    graph.add_conditional_edges(
        "format_digest",
        {
            "path": "node_results.format_digest.has_items",
            "mapping": {
                "true": "resolve_target",
                "false": END,
            },
        },
    )

    graph.add_edge("resolve_target", "send_news")
    graph.add_edge("send_news", "mark_read")
    graph.add_edge("mark_read", END)

    return graph
