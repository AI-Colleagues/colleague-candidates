"""Tests for the Index Steward workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("knowledge_desk/index_steward")


async def test_orcheo_workflow_builds_index_pipeline() -> None:
    """The workflow ensures the text index before the vector index."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {"ensure_text_index", "ensure_vector_index"}
    assert graph.edges == {
        (START, "ensure_text_index"),
        ("ensure_text_index", "ensure_vector_index"),
        ("ensure_vector_index", END),
    }
