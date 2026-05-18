# /// orcheo
# name = "Simple Agent"
# handle = "simple-agent"
# description = "A simple AI agent workflow with a configurable model picker."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# emoji = "🤖"
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
        system_prompt="You are a helpful assistant for workflow demos.",
        model_kwargs={"api_key": "[[openai_api_key]]"},
    )
    graph.add_node("ai_agent", agent)
    graph.set_entry_point("ai_agent")
    graph.set_finish_point("ai_agent")
    return graph
