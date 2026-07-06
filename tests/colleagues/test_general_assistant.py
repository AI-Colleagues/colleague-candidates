"""Tests for the General Assistant workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("general_assistant")


async def test_orcheo_workflow_builds_single_agent_graph() -> None:
    """The workflow routes all turns through a single agent node."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {"ai_agent"}
    assert graph.edges == {(START, "ai_agent"), ("ai_agent", END)}
