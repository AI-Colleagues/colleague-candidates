"""Tests for the Chat Interviewer workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("chat_interviewer")


async def test_orcheo_workflow_builds_single_agent_graph() -> None:
    """The workflow routes all turns through the ChatKit agent node."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {"agent"}
    assert graph.edges == {(START, "agent"), ("agent", END)}
