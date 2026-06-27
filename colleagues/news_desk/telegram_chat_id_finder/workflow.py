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
``telegram_chat_id`` value needed by the Telegram News Carrier colleague:
message your bot first, then send any message here to read the chat ID back.

Configurable inputs (config.json):
- chat_type (chat type to look for; defaults to "private")

Orcheo vault secrets required:
- telegram_token: Telegram bot token
"""

from typing import Any
import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.base import TaskNode


class GetTelegramChatIdNode(TaskNode):
    """Fetch Telegram updates and reply with the latest matching chat ID.

    The ``token`` field defaults to the ``[[telegram_token]]`` vault
    placeholder, which Orcheo resolves to the real bot token before ``run``
    executes. The resolved token is used to build the ``getUpdates`` URL.

    The reply is returned as ``assistant_message`` so ChatKit renders it as the
    bot's chat response, regardless of what message the user sent.
    """

    token: str = "[[telegram_token]]"
    chat_type: str = "private"
    timeout: float = 30.0

    @staticmethod
    def extract_chat(update: dict[str, Any]) -> dict[str, Any] | None:
        """Return the chat object from any message-bearing update field."""
        for key in (
            "message",
            "edited_message",
            "channel_post",
            "edited_channel_post",
            "my_chat_member",
            "chat_member",
        ):
            payload = update.get(key)
            if isinstance(payload, dict):
                chat = payload.get("chat")
                if isinstance(chat, dict):
                    return chat
        return None

    @staticmethod
    def format_chat_name(chat: dict[str, Any]) -> str | None:
        """Build a human-readable name for the chat, if any field is present."""
        title = chat.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

        first = chat.get("first_name")
        last = chat.get("last_name")
        name_parts = [
            part.strip()
            for part in (first, last)
            if isinstance(part, str) and part.strip()
        ]
        if name_parts:
            return " ".join(name_parts)

        username = chat.get("username")
        if isinstance(username, str) and username.strip():
            return f"@{username.strip()}"

        return None

    def _found_message(self, chat: dict[str, Any]) -> str:
        """Render the templated reply for a discovered chat."""
        lines = [
            "✅ Found your Telegram chat ID!",
            "",
            f"🆔 Chat ID: `{chat.get('id')}`",
            f"💬 Type: {chat.get('type')}",
        ]

        name = self.format_chat_name(chat)
        if name is not None:
            lines.append(f"👤 Name: {name}")

        username = chat.get("username")
        if isinstance(username, str) and username.strip():
            lines.append(f"🔗 Username: @{username.strip()}")

        lines += [
            "",
            "Use this value as `telegram_chat_id` in the "
            "Telegram News Carrier colleague.",
        ]
        return "\n".join(lines)

    def _not_found_message(self) -> str:
        """Render the templated reply when no matching chat is found."""
        return (
            f"🔍 I couldn't find a recent {self.chat_type} chat for your bot.\n\n"
            "Please message your bot directly first, then send me any message "
            "here and I'll look again."
        )

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Call getUpdates and reply with the latest chat ID of the wanted type."""
        del state, config
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=self.timeout)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"Telegram getUpdates request failed: {exc!s}"
            raise ValueError(msg) from exc

        payload = response.json()
        if not payload.get("ok"):
            msg = f"Telegram API returned an error: {payload}"
            raise ValueError(msg)

        updates = payload.get("result", [])
        if not isinstance(updates, list):
            updates = []

        # Updates are ordered oldest-first, so scan from the end to find the
        # most recent chat of the requested type.
        for update in reversed(updates):
            if not isinstance(update, dict):
                continue
            chat = self.extract_chat(update)
            if chat is not None and chat.get("type") == self.chat_type:
                return {
                    "assistant_message": self._found_message(chat),
                    "chat_id": chat.get("id"),
                    "chat_type": chat.get("type"),
                    "username": chat.get("username"),
                    "first_name": chat.get("first_name"),
                    "title": chat.get("title"),
                    "update_count": len(updates),
                }

        return {
            "assistant_message": self._not_found_message(),
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
        "get_chat_id",
        GetTelegramChatIdNode(
            name="get_chat_id",
            chat_type="{{config.configurable.chat_type}}",
        ),
    )

    graph.add_edge(START, "get_chat_id")
    graph.add_edge("get_chat_id", END)

    return graph
