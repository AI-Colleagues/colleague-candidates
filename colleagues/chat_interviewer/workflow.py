# /// orcheo
# name = "Chat Interviewer"
# handle = "chat-interviewer"
# description = "An agent for conducting chat interviews."
# version = "0.1.0"
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# avatar = "avatar-08"
# subtitle = "ChatKit Widget Agent"
# ///

"""Orcheo graph example that runs an agent over native ChatKit widgets."""

from orcheo.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode


async def orcheo_workflow() -> StateGraph:
    """Return a graph that routes all work through the ChatKit agent node."""
    agent_node = AgentNode(
        name="agent",
        ai_model="{{config.configurable.ai_model}}",
        model_kwargs={"api_key": "[[openai_api_key]]"},
        system_prompt="{{config.configurable.system_prompt}}",
        use_chatkit_widget_tools=True,
    )

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph
