# /// orcheo
# name = "Telegram News Carrier"
# handle = "telegram-news-carrier"
# description = "Deliver scheduled RSS news digests through Telegram."
# version = "0.1.0"
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# subtitle = "Scheduled delivery"
# ///

"""News Desk - Telegram News Carrier workflow.

Cron-triggered workflow that sends the latest RSS news items to all
active subscribers via Telegram daily at 9:00 AM Amsterdam time.

Configurable inputs (config.json):
- rss_database (MongoDB database name)
- subscribers_collection (collection for subscriber profiles)
- rss_collection (collection for RSS feed items)

Orcheo vault secrets required:
- telegram_token: Telegram bot token
- mdb_connection_string: MongoDB connection string
"""

import html
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from orcheo.edges import Condition, IfElse
from orcheo.graph.state import State
from orcheo.nodes.base import TaskNode
from orcheo.nodes.connectors.telegram import MessageTelegram
from orcheo.nodes.logic import ForLoopNode
from orcheo.nodes.storage import GraphStoreAppendMessageNode
from orcheo.nodes.storage.mongodb import MongoDBFindNode
from orcheo.nodes.triggers import CronTriggerNode


class FormatDigestNode(TaskNode):
    """Format the latest RSS news items into a digest message."""

    @staticmethod
    def decode_title(text: str | None) -> str:
        """Decode HTML entities and escape title text for Telegram HTML."""
        if not text:
            return "No Title"
        decoded = html.unescape(text).replace("\xa0", " ")
        return html.escape(decoded)

    @staticmethod
    def read_items(state: State) -> list[dict[str, Any]]:
        """Extract news items from the find_latest node results."""
        results = state.get("results", {})
        if not isinstance(results, dict):
            return []
        find_result = results.get("find_latest", {})
        if not isinstance(find_result, dict):
            return []
        data = find_result.get("data")
        if isinstance(data, list):
            return data
        return []

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Return the digest content string."""
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
        return {"content": f"Today's RSS News:\n\n{content}"}


async def orcheo_workflow() -> StateGraph:
    """Build the Telegram News Carrier workflow."""
    graph = StateGraph(State)

    # --- Trigger ---
    graph.add_node(
        "cron_trigger",
        CronTriggerNode(
            name="cron_trigger",
            expression="0 9 * * *",
            timezone="Europe/Amsterdam",
        ),
    )

    # --- Fetch data ---
    graph.add_node(
        "find_latest",
        MongoDBFindNode(
            name="find_latest",
            database="{{config.configurable.rss_database}}",
            collection="{{config.configurable.rss_collection}}",
            sort={"isoDate": -1},
            limit=20,
        ),
    )
    graph.add_node(
        "find_active_subscribers",
        MongoDBFindNode(
            name="find_active_subscribers",
            database="{{config.configurable.rss_database}}",
            collection="{{config.configurable.subscribers_collection}}",
            filter={"status": "active"},
        ),
    )

    # --- Format digest ---
    graph.add_node(
        "format_digest",
        FormatDigestNode(name="format_digest"),
    )

    # --- ForLoop over subscribers ---
    graph.add_node(
        "for_each_subscriber",
        ForLoopNode(
            name="for_each_subscriber",
            items="{{find_active_subscribers.data}}",
        ),
    )
    graph.add_node(
        "send_news",
        MessageTelegram(
            name="send_news",
            chat_id="{{for_each_subscriber.current_item.chat_id}}",
            message="{{format_digest.content}}",
            parse_mode="HTML",
        ),
    )
    graph.add_node(
        "persist_digest_history",
        GraphStoreAppendMessageNode(
            name="persist_digest_history",
            key="telegram:{{for_each_subscriber.current_item.chat_id}}",
            content="{{format_digest.content}}",
        ),
    )

    # --- Edges ---
    graph.set_entry_point("cron_trigger")
    graph.add_edge("cron_trigger", "find_latest")
    graph.add_edge("find_latest", "find_active_subscribers")
    graph.add_edge("find_active_subscribers", "format_digest")
    graph.add_edge("format_digest", "for_each_subscriber")

    # ForLoop routes: body or done
    loop_router = IfElse(
        name="for_each_subscriber",
        conditions=[
            Condition(
                left="{{for_each_subscriber.done}}",
                operator="is_falsy",
            ),
        ],
    )
    graph.add_conditional_edges(
        "for_each_subscriber",
        loop_router,
        {
            "true": "send_news",
            "false": END,
        },
    )

    graph.add_edge("send_news", "persist_digest_history")

    # After persisting history, loop back to for_each_subscriber
    graph.add_edge("persist_digest_history", "for_each_subscriber")

    return graph
