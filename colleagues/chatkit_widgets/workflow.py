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

from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode


DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_WIDGETS_DIR = "/app/examples/chatkit_widgets/widgets"


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
