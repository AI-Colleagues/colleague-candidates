# /// orcheo
# name = "Telegram Chat ID Finder"
# handle = "telegram-chat-id-finder"
# description = "Send a message to the bot on Telegram. Then send any message to ChatKit to get a templated reply with your Telegram chat ID."  # noqa: E501
# version = "0.1.0"
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# avatar = "avatar-03"
# subtitle = "Chat ID discovery"
# ///

"""News Desk - Telegram Chat ID Finder workflow.

Chat-triggered helper: send *any* message through ChatKit and the workflow
calls the Telegram Bot API ``getUpdates`` endpoint and replies with a templated
message containing the most recent matching chat ID. Use it to discover the
``telegram_chat_id`` value needed by the Telegram Paperboy colleague:
message your bot first, then send any message here to read the chat ID back.

Configurable inputs (config.json):
- chat_type (chat type to look for; defaults to "private")

Orcheo vault secrets required:
- telegram_token: Telegram bot token
"""

from orcheo.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes import CodeNode, HttpRequestNode
from orcheo.nodes.logic import SetVariableNode


class FormatTelegramChatIdNode(CodeNode):
    """Format the latest matching Telegram chat from the getUpdates response."""

    chat_type: str = "private"

    async def run(self, state, config):  # noqa: C901, PLR0912
        """Return the ChatKit reply and result metadata for the latest chat."""
        http_result = state.get("node_results", {}).get("fetch_updates", {})
        payload = http_result.get("json") if isinstance(http_result, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        if not payload.get("ok"):
            raise ValueError("Telegram API returned an error: " + str(payload))

        updates = payload.get("result", [])
        if not isinstance(updates, list):
            updates = []

        # Updates are ordered oldest-first, so scan from the end to find the
        # most recent chat of the requested type.
        chat = None
        for update in reversed(updates):
            if not isinstance(update, dict):
                continue
            for key in (
                "message",
                "edited_message",
                "channel_post",
                "edited_channel_post",
                "my_chat_member",
                "chat_member",
            ):
                update_payload = update.get(key)
                if isinstance(update_payload, dict):
                    candidate = update_payload.get("chat")
                    if isinstance(candidate, dict):
                        chat = candidate
                        break
            if chat is not None and chat.get("type") == self.chat_type:
                break
            chat = None

        if chat is not None:
            lines = [
                "✅ Found your Telegram chat ID!",
                "",
                "🆔 Chat ID: `" + str(chat.get("id")) + "`",
                "💬 Type: " + str(chat.get("type")),
            ]

            name = None
            title = chat.get("title")
            if isinstance(title, str) and title.strip():
                name = title.strip()

            first = chat.get("first_name")
            last = chat.get("last_name")
            name_parts = []
            if isinstance(first, str) and first.strip():
                name_parts.append(first.strip())
            if isinstance(last, str) and last.strip():
                name_parts.append(last.strip())
            if name is None and name_parts:
                name = " ".join(name_parts)

            username = chat.get("username")
            if name is None and isinstance(username, str) and username.strip():
                name = "@" + username.strip()

            if name is not None:
                lines.append("👤 Name: " + name)

            if isinstance(username, str) and username.strip():
                lines.append("🔗 Username: @" + username.strip())

            paperboy_line = (
                "Use this value as `telegram_chat_id` in the Telegram Paperboy "
                "colleague."
            )
            lines += ["", paperboy_line]
            assistant_message = "\n".join(lines)
            return {
                "assistant_message": assistant_message,
                "chat_id": chat.get("id"),
                "chat_type": chat.get("type"),
                "username": chat.get("username"),
                "first_name": chat.get("first_name"),
                "title": chat.get("title"),
                "update_count": len(updates),
            }

        assistant_message = (
            "🔍 I couldn't find a recent "
            + str(self.chat_type)
            + " chat for your bot.\n\n"
            + "Please message your bot directly first, then send me any message "
            + "here and I'll look again."
        )
        return {
            "assistant_message": assistant_message,
            "chat_id": None,
            "chat_type": self.chat_type,
            "update_count": len(updates),
        }


async def orcheo_workflow() -> StateGraph:
    """Build the Telegram Chat ID Finder workflow.

    Any chat message sent through ChatKit triggers the lookup; the message
    content itself is ignored and a templated reply is returned.
    """
    graph = StateGraph(State)

    graph.add_node(
        "load_telegram_token",
        SetVariableNode(
            name="load_telegram_token",
            variables={"telegram_token": "[[telegram_token]]"},
        ),
    )
    graph.add_node(
        "fetch_updates",
        HttpRequestNode(
            name="fetch_updates",
            method="GET",
            url=(
                "https://api.telegram.org/bot"
                "{{node_results.load_telegram_token.telegram_token}}/getUpdates"
            ),
            timeout=30.0,
            raise_for_status=True,
        ),
    )
    graph.add_node(
        "get_chat_id",
        FormatTelegramChatIdNode(
            name="get_chat_id",
            chat_type="{{config.configurable.chat_type}}",
        ),
    )

    graph.add_edge(START, "load_telegram_token")
    graph.add_edge("load_telegram_token", "fetch_updates")
    graph.add_edge("fetch_updates", "get_chat_id")
    graph.add_edge("get_chat_id", END)

    return graph
