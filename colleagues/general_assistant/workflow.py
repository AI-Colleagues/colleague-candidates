# /// orcheo
# name = "General Assistant"
# handle = "general-assistant"
# description = "General-purpose AI assistant answering questions from model knowledge. No access to tools or extra information."  # noqa: E501
# version = "0.1.0"
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# avatar = "avatar-20"
# subtitle = "AI Assistant"
# ///

from langgraph.graph import StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode


async def orcheo_workflow() -> StateGraph:
    """Build a Python agent workflow with a configurable model."""
    graph = StateGraph(State)
    agent = AgentNode(
        name="ai_agent",
        ai_model="{{config.configurable.ai_model}}",
        system_prompt="{{config.configurable.system_prompt}}",
        model_kwargs={"api_key": "[[openai_api_key]]"},
    )
    graph.add_node("ai_agent", agent)
    graph.set_entry_point("ai_agent")
    graph.set_finish_point("ai_agent")
    return graph
