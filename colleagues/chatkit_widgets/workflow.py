# /// orcheo
# name = "ChatKit Widgets Agent"
# handle = "chatkit-widgets"
# description = "An agent using MCP ChatKit widget tools to render interactive UI."
# entrypoint = "build_graph"
# emoji = "🧑‍🎨"
# subtitle = "Widget UI"
# notes = "Seeded from ChatKit Widgets template (`chatkit_widgets.py`)."
# ///

"""Orcheo graph example that runs an agent over MCP ChatKit widgets."""

from collections.abc import Mapping, Sequence
from typing import Any
from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode


DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_WIDGETS_DIR = "/app/examples/chatkit_widgets/widgets"
DEFAULT_MESSAGE = "Generate a shopping list with the following items: apples, bananas, bread, milk, eggs, cheese, butter, and tomato."  # noqa: E501


def messages_from_state(state_view: Any) -> list[Any]:
    """Return LangChain messages carried in the workflow state, if any."""
    if not isinstance(state_view, Mapping):
        return []
    messages = state_view.get("_messages") or state_view.get("messages") or []
    return messages if isinstance(messages, list) else []


def text_from_content(content: Any) -> str | None:
    """Extract text from ToolMessage content payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content, bytes | bytearray | str
    ):
        for entry in content:
            if isinstance(entry, Mapping):
                text_value = entry.get("text")
                if isinstance(text_value, str):
                    return text_value
            text_attr = getattr(entry, "text", None)
            if isinstance(text_attr, str):
                return text_attr
    return None


def widget_payload_from_tool_message(message: Any) -> Any | None:
    """Return a widget payload parsed from a ToolMessage."""
    artifact = getattr(message, "artifact", None)
    content = getattr(message, "content", None)
    if isinstance(message, Mapping):
        artifact = message.get("artifact")
        content = message.get("content")

    if isinstance(artifact, Mapping):
        structured = artifact.get("structured_content")
        if structured is not None:
            return structured

    text_value = text_from_content(content)
    if not text_value:
        return None
    return text_value.strip() or None


def build_graph(
    model: str = DEFAULT_MODEL,
    widgets_dir: str = DEFAULT_WIDGETS_DIR,
) -> StateGraph:
    """Return a graph that routes all work through the ChatKit agent node."""
    mcp_servers = {
        "mcp-chatkit-widget": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-chatkit-widget", "--widgets-dir", widgets_dir],
        }
    }
    agent_node = AgentNode(
        name="agent",
        ai_model=model,
        model_kwargs={"api_key": "[[openai_api_key]]"},
        system_prompt="You are a helpful assistant that can use widget tools to interact with the user.",  # noqa: E501
        mcp_servers=mcp_servers,
    )

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph
