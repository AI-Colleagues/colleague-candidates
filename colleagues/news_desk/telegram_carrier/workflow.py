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

import html
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from orcheo.edges import Condition, IfElseEdge
from orcheo.graph.state import State
from orcheo.nodes.base import TaskNode
from orcheo.nodes.connectors.telegram import MessageTelegram, TelegramBotListenerNode
from orcheo.nodes.storage.mongodb import MongoDBFindNode, MongoDBUpdateManyNode
from orcheo.nodes.triggers import CronTriggerNode


class DetectTriggerNode(TaskNode):
    """Detect whether the run was started by an inbound listener event."""

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Return whether a listener payload is present in inputs."""
        inputs = state.get("inputs", {})
        is_listener = bool(
            isinstance(inputs, dict)
            and (inputs.get("listener") or inputs.get("platform"))
        )
        return {"is_listener": is_listener}


class FormatDigestNode(TaskNode):
    """Format the latest unread RSS news items into a digest message."""

    @staticmethod
    def decode_title(text: str | None) -> str:
        """Decode HTML entities and escape title text for Telegram HTML."""
        if not text:
            return "No Title"
        decoded = html.unescape(text).replace("\xa0", " ")
        return html.escape(decoded)

    @staticmethod
    def read_items(state: State) -> list[dict[str, Any]]:
        """Extract news items from the find_unread node results."""
        results = state.get("results", {})
        if not isinstance(results, dict):
            return []
        find_result = results.get("find_unread", {})
        if not isinstance(find_result, dict):
            return []
        data = find_result.get("data")
        if isinstance(data, list):
            return data
        return []

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Return the digest content string and the delivered item IDs."""
        items = self.read_items(state)

        lines = []
        for item in items:
            title = self.decode_title(item.get("title"))
            url = item.get("link", "")
            if url:
                lines.append(f'- <a href="{url}">{title}</a>')
            else:
                lines.append(f"- {title}")

        content = "\n".join(lines) if lines else "No news updates today."
        return {
            "content": f"Today's RSS News:\n\n{content}",
            "ids": [item.get("_id") for item in items if item.get("_id") is not None],
        }


class ResolveTargetChatNode(TaskNode):
    """Pick the chat that receives the digest.

    Inbound messages are answered in the originating chat; scheduled runs
    fall back to the configured broadcast chat.
    """

    default_chat_id: str = "{{config.configurable.telegram_chat_id}}"

    @staticmethod
    def listener_chat_id(state: State) -> str | None:
        """Return the chat ID from the inbound Telegram listener event."""
        results = state.get("results", {})
        if not isinstance(results, dict):
            return None
        listener = results.get("telegram_listener", {})
        if not isinstance(listener, dict):
            return None
        chat_id = listener.get("chat_id")
        return str(chat_id) if chat_id else None

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Return the resolved Telegram chat ID for delivery."""
        return {"chat_id": self.listener_chat_id(state) or self.default_chat_id}


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
        MessageTelegram(
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
    trigger_router = IfElseEdge(
        name="trigger_router",
        conditions=[
            Condition(left="{{detect_trigger.is_listener}}", operator="is_truthy"),
        ],
    )
    graph.add_conditional_edges(
        "detect_trigger",
        trigger_router,
        {
            "true": "telegram_listener",
            "false": "cron_trigger",
        },
    )

    graph.add_edge("cron_trigger", "find_unread")

    # Only build a digest for inbound updates that carry a message.
    inbound_router = IfElseEdge(
        name="inbound_router",
        conditions=[
            Condition(
                left="{{telegram_listener.should_process}}", operator="is_truthy"
            ),
        ],
    )
    graph.add_conditional_edges(
        "telegram_listener",
        inbound_router,
        {
            "true": "find_unread",
            "false": END,
        },
    )

    graph.add_edge("find_unread", "format_digest")

    # Only send (and mark read) when there are unread items to deliver.
    deliver_router = IfElseEdge(
        name="deliver_router",
        conditions=[
            Condition(left="{{format_digest.ids}}", operator="is_truthy"),
        ],
    )
    graph.add_conditional_edges(
        "format_digest",
        deliver_router,
        {
            "true": "resolve_target",
            "false": END,
        },
    )

    graph.add_edge("resolve_target", "send_news")
    graph.add_edge("send_news", "mark_read")
    graph.add_edge("mark_read", END)

    return graph
