# /// orcheo
# name = "Telegram News Carrier"
# handle = "telegram-news-carrier"
# description = "Deliver scheduled RSS news digests through Telegram."
# version = "0.1.0"
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# avatar = "avatar-01"
# subtitle = "Scheduled delivery"
# ///

"""News Desk - Telegram News Carrier workflow.

Cron-triggered workflow that sends the latest *unread* RSS news items to a
single configured Telegram chat on a configurable schedule (daily at
9:00 AM Amsterdam time by default) and marks the delivered items as read.

Configurable inputs (config.json):
- cron_expression (cron schedule, Europe/Amsterdam timezone)
- rss_database (MongoDB database name)
- rss_collection (collection for RSS feed items)
- telegram_chat_id (Telegram chat that receives the digest)

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
from orcheo.nodes.connectors.telegram import MessageTelegram
from orcheo.nodes.storage.mongodb import MongoDBFindNode, MongoDBUpdateManyNode
from orcheo.nodes.triggers import CronTriggerNode


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


async def orcheo_workflow() -> StateGraph:
    """Build the Telegram News Carrier workflow."""
    graph = StateGraph(State)

    # --- Trigger ---
    graph.add_node(
        "cron_trigger",
        CronTriggerNode(
            name="cron_trigger",
            expression="{{config.configurable.cron_expression}}",
            timezone="Europe/Amsterdam",
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

    # --- Deliver to the configured chat ---
    graph.add_node(
        "send_news",
        MessageTelegram(
            name="send_news",
            chat_id="{{config.configurable.telegram_chat_id}}",
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
    graph.set_entry_point("cron_trigger")
    graph.add_edge("cron_trigger", "find_unread")
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
            "true": "send_news",
            "false": END,
        },
    )

    graph.add_edge("send_news", "mark_read")
    graph.add_edge("mark_read", END)

    return graph
