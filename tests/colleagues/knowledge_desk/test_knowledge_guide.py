"""Tests for the Knowledge Guide workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("knowledge_desk/knowledge_guide")


def test_build_hybrid_search_tool_graph_wires_the_search_pipeline() -> None:
    """The hybrid search subgraph chains embedding, search, and formatting."""
    graph = workflow.build_hybrid_search_tool_graph()

    assert set(graph.nodes.keys()) == {
        "query_embedding",
        "hybrid_search",
        "adapt_results",
        "format_results",
    }
    assert graph.edges == {
        (START, "query_embedding"),
        ("query_embedding", "hybrid_search"),
        ("hybrid_search", "adapt_results"),
        ("adapt_results", "format_results"),
        ("format_results", END),
    }


async def test_orcheo_workflow_builds_single_agent_graph_with_tool() -> None:
    """The workflow exposes the hybrid search subgraph as an agent tool."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {"agent"}
    assert graph.edges == {(START, "agent"), ("agent", END)}

    agent_node = graph.nodes["agent"].runnable.afunc
    (tool,) = agent_node.workflow_tools
    assert tool.name == "mongodb_hybrid_search"
    assert tool.args_schema is workflow.HybridSearchInput
