# /// orcheo
# name = "QQ Private Listener"
# handle = "qq-private-listener"
# description = "Receives QQ messages, generates an agent reply, and sends it back."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# avatar = "avatar-10"
# subtitle = "QQ Listener"
# notes = "Seeded from QQ Private Listener template."
# [metadata]
# template_version = "1.0.0"
# min_orcheo_version = "0.1.0"
# validated_provider_api = "qq-bot-api-v2"
# validation_date = "2026-03-11"
# owner = "Shaojie Jiang"
# acceptance_criteria = [
#   "Imports into Canvas without manual edits.",
#   "Runs QQBotListenerNode -> AgentNode -> MessageQQNode end to end.",
#   "Documents required credentials and provider/API compatibility.",
# ]
# revalidation_triggers = [
#   "QQ Bot API v2 major version change",
#   "MessageQQNode contract change",
#   "Listener runtime contract change",
# ]
# reply_node_contracts = ["MessageQQNode@1"]
# ///

from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode
from orcheo.nodes.connectors.qq import MessageQQNode, QQBotListenerNode


async def orcheo_workflow() -> StateGraph:
    """Build a private QQ listener workflow."""
    graph = StateGraph(State)

    graph.add_node(
        "qq_listener",
        QQBotListenerNode(
            name="qq_listener",
            app_id="[[qq_app_id]]",
            client_secret="[[qq_client_secret]]",
            allowed_events=[
                "C2C_MESSAGE_CREATE",
                "GROUP_AT_MESSAGE_CREATE",
                "AT_MESSAGE_CREATE",
            ],
            allowed_scene_types=["c2c", "group", "channel"],
            bot_identity_key="qq:primary",
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
        "send_qq",
        MessageQQNode(
            name="send_qq",
            app_id="[[qq_app_id]]",
            client_secret="[[qq_client_secret]]",
            openid="{{results.qq_listener.reply_target.openid}}",
            group_openid="{{results.qq_listener.reply_target.group_openid}}",
            channel_id="{{results.qq_listener.reply_target.channel_id}}",
            guild_id="{{results.qq_listener.reply_target.guild_id}}",
            msg_id="{{results.qq_listener.reply_target.msg_id}}",
        ),
    )

    graph.add_edge(START, "qq_listener")
    graph.add_edge("qq_listener", "agent_reply")
    graph.add_edge("agent_reply", "send_qq")
    graph.add_edge("send_qq", END)
    return graph
