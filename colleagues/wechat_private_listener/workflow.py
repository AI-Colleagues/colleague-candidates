# /// orcheo
# name = "WeChat Private Listener"
# handle = "wechat-private-listener"
# description = "Receives WeChat messages via plugin and replies with AgentNode."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# avatar = "avatar-21"
# subtitle = "WeChat Listener"
# notes = "Seeded from the WeChat private listener plugin template."
# [metadata]
# template_version = "1.0.0"
# min_orcheo_version = "0.1.0"
# validated_provider_api = "openclaw-wechat-plugin-2026-03-22"
# validation_date = "2026-03-22"
# owner = "Shaojie Jiang"
# required_plugins = ["orcheo-plugin-wechat-listener"]
# acceptance_criteria = [
#   "Imports into Canvas once the WeChat listener plugin is installed.",
#   "Compiles a valid WeChat listener subscription from the plugin-backed workflow.",
#   "Runs the listener, AgentNode, reply extractor, and WechatReplyNode end to end.",
# ]
# revalidation_triggers = [
#   "WeChat listener plugin contract change",
#   "WechatReplyNode or AgentReplyExtractorNode contract change",
#   "OpenClaw WeChat API contract change",
#   "Canvas template import contract change",
# ]
# reply_node_contracts = ["WechatReplyNode@1"]
# ///

from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode, AgentReplyExtractorNode
from orcheo_plugin_wechat_listener import WechatListenerPluginNode, WechatReplyNode


async def orcheo_workflow() -> StateGraph:
    """Build a private WeChat listener workflow backed by the plugin."""
    graph = StateGraph(State)

    graph.add_node(
        "wechat_listener",
        WechatListenerPluginNode(
            name="wechat_listener",
            account_id="[[wechat_account_id]]",
            bot_token="[[wechat_bot_token]]",
            base_url="[[wechat_base_url]]",
            bot_identity_key="wechat:primary",
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
        "extract_reply",
        AgentReplyExtractorNode(name="extract_reply"),
    )
    graph.add_node(
        "send_wechat",
        WechatReplyNode(
            name="send_wechat",
            account_id="[[wechat_account_id]]",
            bot_token="[[wechat_bot_token]]",
            base_url="[[wechat_base_url]]",
            message="{{results.extract_reply.agent_reply}}",
            reply_target="{{results.wechat_listener.reply_target}}",
            raw_event="{{results.wechat_listener.raw_event}}",
        ),
    )

    graph.add_edge(START, "wechat_listener")
    graph.add_edge("wechat_listener", "agent_reply")
    graph.add_edge("agent_reply", "extract_reply")
    graph.add_edge("extract_reply", "send_wechat")
    graph.add_edge("send_wechat", END)
    return graph
