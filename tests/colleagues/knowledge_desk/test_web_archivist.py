"""Tests for the Web Archivist workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("knowledge_desk/web_archivist")


async def test_orcheo_workflow_builds_scrape_and_upload_pipeline() -> None:
    """The workflow scrapes, chunks, embeds, then uploads to MongoDB."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "web_loader",
        "chunking",
        "chunk_embedding",
        "mongodb_upload",
    }
    assert graph.edges == {
        (START, "web_loader"),
        ("web_loader", "chunking"),
        ("chunking", "chunk_embedding"),
        ("chunk_embedding", "mongodb_upload"),
        ("mongodb_upload", END),
    }
