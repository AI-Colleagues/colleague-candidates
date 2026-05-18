# /// orcheo
# name = "Medical Reminder Assistant"
# handle = "message-handler"
# description = "Handles WeChat Work patient chats for reminder sign-up and reporting."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# emoji = "💊"
# subtitle = "AI Patient Support"
# ///

"""WeChat Medical Reminder - Message Handler workflow.

Webhook-triggered workflow that handles user registration, deregistration,
and status reporting via an AI agent with MongoDB tools.

Configurable inputs (config.json):
- reminder_database (MongoDB database name)
- registered_users_collection (collection for user profiles)
- user_records_collection (collection for status reports)
- timezone_offset_hours (UTC offset for date context, default 8 for Asia/Shanghai)

Orcheo vault secrets required:
- wecom_app_secret_medical_reminder: WeCom app secret for access token
- wecom_token: WeCom callback token for signature validation
- wecom_encoding_aes_key: AES key for callback decryption
- mdb_connection_string: MongoDB connection string
- openai_api_key: OpenAI API key for the LLM agent
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from orcheo.edges import Condition, IfElse
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode, AgentReplyExtractorNode
from orcheo.nodes.base import TaskNode
from orcheo.nodes.connectors.wecom import (
    WeComAccessTokenNode,
    WeComCustomerServiceSendNode,
    WeComCustomerServiceSyncNode,
    WeComEventsParserNode,
)
from orcheo.nodes.mongodb import MongoDBFindNode
from orcheo.nodes.triggers import WebhookTriggerNode


AGENT_INSTRUCTIONS = """\
工具：
- mongodb_find：查询文档（支持 filter/sort/limit）
- mongodb_update_one：更新或插入文档（支持 filter/update/options）

你负责以下任务：

1. 用户注册：
   - 从用户消息中提取提醒项目列表和手机号
   - 先用 mongodb_find 检查 registered_users 中是否已存在该 \
external_userid + open_kf_id 组合
   - 如不存在，用 mongodb_update_one (upsert: true) 创建记录，包含：
     external_userid, external_username, open_kf_id, phone_number, reminder_items,
     status: "active", registered_at, updated_at
   - 如已存在且 status 为 active，提示用户已注册
   - 确认注册并列出已保存的提醒项目

2. 用户注销：
   - 检测用户注销意图时，先回复确认请求（如"确定要取消提醒服务吗？请回复'确认'"）
   - 收到确认后，用 mongodb_update_one 将 status 设为 "inactive"，更新 updated_at
   - 确认注销完成

3. 状态报告：
   - 只有注册状态为 active 的用户才能报告状态
   - 如果用户注册状态为 inactive 或未注册，拒绝记录并提示用户需要先注册
   - 用户报告每日健康状态时，提取各提醒项目的完成情况
   - 用 mongodb_update_one (upsert: true) 在 user_records 中创建记录：
     filter: {"external_userid": "<id>", "open_kf_id": "<id>", \
"record_date": "<YYYY-MM-DD>"}
     update: {"$set": {raw_text, items_status, recorded_at, external_userid, \
open_kf_id, record_date}}
   - 确认已记录并总结状态

4. 其他消息：
   - 如用户消息不属于以上类别，友好地介绍你的功能并引导用户

指南：
- 所有回复使用中文纯文本，不要输出 JSON 或 Markdown
- 对破坏性操作（注销）必须先确认
- 手机号格式验证：中国大陆手机号（11位，1开头）
- 提醒项目应以简洁短语存储
- 在合适的场景下使用用户的 external_username 称呼用户，如问候、确认操作等
- 所有 mongodb 查询和更新必须同时使用 external_userid 和 open_kf_id 作为筛选条件，\
确保不同客服账号的数据互相独立"""


class PrepareAgentContextNode(TaskNode):
    """Build the agent system prompt with all dynamic values resolved."""

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Assemble a system prompt with user profile and MongoDB config."""
        wecom_data = state.get("results", {}).get("wecom_cs_sync", {})
        configurable = config.get("configurable", {})
        offset_hours = configurable.get("timezone_offset_hours", 8)
        now = datetime.now(timezone(timedelta(hours=offset_hours)))

        # Extract user profile from the lookup_user query result.
        lookup_data = state.get("results", {}).get("lookup_user", {}).get("data", [])
        user_profile = lookup_data[0] if lookup_data else None

        user_status_line = "- 用户注册状态: 未注册\n"
        if user_profile:
            status = user_profile.get("status", "unknown")
            user_status_line = (
                f"- 用户注册状态: {status}\n"
                f"- 提醒项目: {user_profile.get('reminder_items', [])}\n"
            )

        system_prompt = (
            "你是一个微信健康提醒助理。"
            "你通过自然语言对话帮助用户管理健康提醒注册和每日状态记录。\n\n"
            "上下文（以下为实际值，请在工具调用中直接使用）：\n"
            "- external_userid: "
            f"<value>{wecom_data.get('external_userid', '')}</value>\n"
            "- external_username: "
            f"<value>{wecom_data.get('external_username', '')}</value>\n"
            f"- open_kf_id: <value>{wecom_data.get('open_kf_id', '')}</value>\n"
            f"- 当前日期: {now.date().isoformat()}\n"
            f"- 当前时间: {now.isoformat()}\n" + user_status_line + "\nMongoDB 配置：\n"
            f"- database: {configurable.get('reminder_database')}\n"
            "- registered_users collection: "
            f"{configurable.get('registered_users_collection')}\n"
            "- user_records collection: "
            f"{configurable.get('user_records_collection')}\n\n" + AGENT_INSTRUCTIONS
        )
        return {"system_prompt": system_prompt}


async def orcheo_workflow() -> StateGraph:
    """Build the Message Handler workflow."""
    graph = StateGraph(State)

    # --- Trigger ---
    graph.add_node(
        "webhook_trigger",
        WebhookTriggerNode(
            name="webhook_trigger",
            allowed_methods=["GET", "POST"],
        ),
    )

    # --- WeCom ingress ---
    graph.add_node(
        "wecom_events_parser",
        WeComEventsParserNode(
            name="wecom_events_parser",
        ),
    )
    graph.add_node(
        "get_cs_access_token",
        WeComAccessTokenNode(
            name="get_cs_access_token",
            app_secret="[[wecom_app_secret_medical_reminder]]",
        ),
    )
    graph.add_node(
        "wecom_cs_sync",
        WeComCustomerServiceSyncNode(name="wecom_cs_sync"),
    )

    # --- Lookup user registration ---
    graph.add_node(
        "lookup_user",
        MongoDBFindNode(
            name="lookup_user",
            database="{{config.configurable.reminder_database}}",
            collection="{{config.configurable.registered_users_collection}}",
            filter={
                "external_userid": "{{wecom_cs_sync.external_userid}}",
                "open_kf_id": "{{wecom_cs_sync.open_kf_id}}",
            },
            limit=1,
        ),
    )

    # --- Prepare agent context ---
    graph.add_node(
        "prepare_agent_context",
        PrepareAgentContextNode(name="prepare_agent_context"),
    )

    # --- Agent ---
    graph.add_node(
        "agent",
        AgentNode(
            name="agent",
            ai_model="openai:gpt-4o-mini",
            model_kwargs={"api_key": "[[openai_api_key]]"},
            system_prompt="{{prepare_agent_context.system_prompt}}",
            predefined_tools=["mongodb_find", "mongodb_update_one"],
            use_graph_chat_history=True,
            history_key_candidates=[
                "wecom_cs:{{results.wecom_cs_sync.open_kf_id}}:{{results.wecom_cs_sync.external_userid}}",
            ],
        ),
    )

    # --- Reply path ---
    graph.add_node(
        "extract_agent_reply",
        AgentReplyExtractorNode(
            name="extract_agent_reply",
            fallback_message="抱歉，处理您的请求时遇到了问题，请稍后再试。",
        ),
    )
    graph.add_node(
        "send_cs_reply",
        WeComCustomerServiceSendNode(
            name="send_cs_reply",
            message="{{extract_agent_reply.agent_reply}}",
        ),
    )

    # --- Edges ---
    graph.set_entry_point("webhook_trigger")
    graph.add_edge("webhook_trigger", "wecom_events_parser")

    # After parser: route based on immediate_response or should_process
    parser_router = IfElse(
        name="parser_router",
        conditions=[
            Condition(
                left="{{wecom_events_parser.immediate_response}}",
                operator="is_truthy",
            ),
        ],
    )
    graph.add_conditional_edges(
        "wecom_events_parser",
        parser_router,
        {
            "true": END,
            "false": "get_cs_access_token",
        },
    )

    graph.add_edge("get_cs_access_token", "wecom_cs_sync")

    # After sync: only process if should_process is truthy
    sync_router = IfElse(
        name="sync_router",
        conditions=[
            Condition(
                left="{{wecom_cs_sync.should_process}}",
                operator="is_truthy",
            ),
        ],
    )
    graph.add_conditional_edges(
        "wecom_cs_sync",
        sync_router,
        {
            "true": "lookup_user",
            "false": END,
        },
    )

    graph.add_edge("lookup_user", "prepare_agent_context")
    graph.add_edge("prepare_agent_context", "agent")
    graph.add_edge("agent", "extract_agent_reply")
    graph.add_edge("extract_agent_reply", "send_cs_reply")
    graph.add_edge("send_cs_reply", END)

    return graph
