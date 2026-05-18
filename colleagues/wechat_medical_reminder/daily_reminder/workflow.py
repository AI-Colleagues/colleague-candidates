# /// orcheo
# name = "Daily Health Reminder"
# handle = "daily-reminder"
# description = "Sends daily health reminders to registered WeChat Work users."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# emoji = "⏰"
# subtitle = "AI Health Reminders"
# ///

"""WeChat Medical Reminder - Daily Reminder workflow.

Cron-triggered workflow that sends personalised health reminders to all
active registered users daily at 9:00 AM Asia/Shanghai.

Configurable inputs (config.json):
- reminder_database (MongoDB database name)
- registered_users_collection (collection for user profiles)

Orcheo vault secrets required:
- wecom_app_secret_medical_reminder: WeCom app secret for access token
- wecom_corp_id: WeCom corp ID
- mdb_connection_string: MongoDB connection string
"""

from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from orcheo.edges import Condition, IfElse
from orcheo.graph.state import State
from orcheo.nodes.base import TaskNode
from orcheo.nodes.connectors.wecom import (
    WeComAccessTokenNode,
    WeComCustomerServiceSendNode,
)
from orcheo.nodes.logic import ForLoopNode
from orcheo.nodes.mongodb import MongoDBFindNode
from orcheo.nodes.storage import GraphStoreAppendMessageNode
from orcheo.nodes.triggers import CronTriggerNode


class PrepareMessageNode(TaskNode):
    """Prepare the personalised reminder message for the current user.

    Reads the current user from ForLoopNode output each iteration.
    """

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Return the personalised message and user identifiers."""
        results = state.get("results", {})
        user = results.get("for_each_user", {}).get("current_item", {})
        if not isinstance(user, dict):
            return {"message": "", "external_userid": "", "open_kf_id": ""}

        username = user.get("external_username", "")
        items = user.get("reminder_items", [])
        if isinstance(items, list) and items:
            items_text = "\n".join(f"- {item}" for item in items)
        else:
            items_text = "- 您的健康任务"

        greeting = f"{username}，早上好！" if username else "早上好！"
        message = (
            f"{greeting}今日健康提醒，请完成以下任务：\n"
            f"{items_text}\n"
            f"完成后请回复您的状态。"
        )
        return {
            "message": message,
            "external_userid": user.get("external_userid", ""),
            "open_kf_id": user.get("open_kf_id", ""),
        }


async def orcheo_workflow() -> StateGraph:
    """Build the Daily Reminder workflow."""
    graph = StateGraph(State)

    # --- Trigger ---
    graph.add_node(
        "cron_trigger",
        CronTriggerNode(
            name="cron_trigger",
            expression="0 9 * * *",
            timezone="Asia/Shanghai",
        ),
    )

    # --- Fetch data ---
    graph.add_node(
        "get_access_token",
        WeComAccessTokenNode(
            name="get_access_token",
            app_secret="[[wecom_app_secret_medical_reminder]]",
        ),
    )
    # NOTE: Fetches all active users at once. For large deployments (1000+
    # users), consider adding pagination with limit/skip.
    graph.add_node(
        "find_active_users",
        MongoDBFindNode(
            name="find_active_users",
            database="{{config.configurable.reminder_database}}",
            collection="{{config.configurable.registered_users_collection}}",
            filter={
                "status": "active",
                "external_userid": {"$exists": True, "$ne": ""},
                "open_kf_id": {"$exists": True, "$ne": ""},
            },
        ),
    )

    # --- ForLoop over users ---
    graph.add_node(
        "for_each_user",
        ForLoopNode(
            name="for_each_user",
            items="{{find_active_users.data}}",
        ),
    )
    graph.add_node(
        "prepare_message",
        PrepareMessageNode(name="prepare_message"),
    )
    graph.add_node(
        "send_reminder",
        WeComCustomerServiceSendNode(
            name="send_reminder",
            open_kf_id="{{prepare_message.open_kf_id}}",
            external_userid="{{prepare_message.external_userid}}",
            message="{{prepare_message.message}}",
            raise_on_error=False,
        ),
    )
    graph.add_node(
        "persist_reminder_history",
        GraphStoreAppendMessageNode(
            name="persist_reminder_history",
            key="wecom_cs:{{prepare_message.open_kf_id}}:{{prepare_message.external_userid}}",
            content="{{prepare_message.message}}",
        ),
    )

    # --- Edges ---
    graph.set_entry_point("cron_trigger")
    graph.add_edge("cron_trigger", "get_access_token")
    graph.add_edge("get_access_token", "find_active_users")
    graph.add_edge("find_active_users", "for_each_user")

    # ForLoop routes: body or done
    loop_router = IfElse(
        name="for_each_user_router",
        conditions=[
            Condition(
                left="{{for_each_user.done}}",
                operator="is_falsy",
            ),
        ],
    )
    graph.add_conditional_edges(
        "for_each_user",
        loop_router,
        {
            "true": "prepare_message",
            "false": END,
        },
    )

    graph.add_edge("prepare_message", "send_reminder")
    graph.add_edge("send_reminder", "persist_reminder_history")

    # After persisting history, loop back to for_each_user
    graph.add_edge("persist_reminder_history", "for_each_user")

    return graph
