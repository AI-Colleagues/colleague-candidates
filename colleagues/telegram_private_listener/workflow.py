# /// orcheo
# name = "Telegram Private Listener"
# handle = "telegram-private-listener"
# description = "Receives Telegram bot messages and replies with AgentNode."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# emoji = "✈️"
# subtitle = "Telegram Listener"
# notes = "Seeded from Telegram Private Listener template."
# [metadata]
# template_version = "1.0.0"
# min_orcheo_version = "0.1.0"
# validated_provider_api = "telegram-bot-api"
# validation_date = "2026-03-11"
# owner = "Shaojie Jiang"
# acceptance_criteria = [
#   "Imports into Canvas without manual edits.",
#   "Runs TelegramBotListenerNode -> AgentNode -> MessageTelegramNode end to end.",
#   "Documents required credentials and provider/API compatibility.",
# ]
# revalidation_triggers = [
#   "Telegram Bot API major version change",
#   "MessageTelegramNode contract change",
#   "Listener runtime contract change",
# ]
# reply_node_contracts = ["MessageTelegramNode@1"]
# ///

from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode
from orcheo.nodes.connectors.telegram import (
    MessageTelegramNode,
    TelegramBotListenerNode,
)


async def orcheo_workflow() -> StateGraph:
    """Build a private Telegram listener workflow."""
    graph = StateGraph(State)

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
    graph.add_node(
        "agent_reply",
        AgentNode(
            name="agent_reply",
            ai_model="{{config.configurable.ai_model}}",
            system_prompt="{{config.configurable.system_prompt}}",
            model_kwargs={"api_key": "[[openai_api_key]]"},
            use_graph_chat_history=True,
        ),
    )
    graph.add_node(
        "send_telegram",
        MessageTelegramNode(
            name="send_telegram",
            token="[[telegram_token]]",
            chat_id="{{results.telegram_listener.reply_target.chat_id}}",
        ),
    )

    graph.add_edge(START, "telegram_listener")
    graph.add_edge("telegram_listener", "agent_reply")
    graph.add_edge("agent_reply", "send_telegram")
    graph.add_edge("send_telegram", END)
    return graph
