"""Tests for the Theme Finder workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("qualitative_research_desk/theme_finder")


async def test_orcheo_workflow_builds_the_router_and_pipelines() -> None:
    """The router agent dispatches to the codebook pipeline or export/respond."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "load_attachments",
        "validate_files",
        "router_agent",
        "extract_ai_message",
        "review_codebook",
        "export_codebook",
        "generate_codebook",
    }
    assert graph.edges == {
        (START, "load_attachments"),
        ("load_attachments", "validate_files"),
        ("generate_codebook", "review_codebook"),
        ("review_codebook", "router_agent"),
        ("export_codebook", END),
        ("extract_ai_message", END),
    }
    assert {"validate_files", "router_agent"} <= set(graph.branches.keys())


async def test_generate_codebook_subgraph_wires_the_open_coding_loop() -> None:
    """The nested subgraph loops open coding before consolidating a codebook."""
    graph = await workflow.orcheo_workflow()

    subgraph = graph.nodes["generate_codebook"].runnable.builder

    assert set(subgraph.nodes.keys()) == {
        "ingest",
        "open_coder_prepare",
        "open_coder",
        "open_coder_finalize",
        "codebook_consolidator_prepare",
        "codebook_consolidator",
        "codebook_consolidator_finalize",
        "codebook_output",
    }
    assert subgraph.edges == {
        (START, "ingest"),
        ("open_coder", "open_coder_finalize"),
        ("codebook_consolidator", "codebook_consolidator_finalize"),
        ("codebook_consolidator_finalize", "codebook_output"),
        ("codebook_output", END),
    }
    assert {
        "ingest",
        "open_coder_prepare",
        "open_coder_finalize",
        "codebook_consolidator_prepare",
    } <= set(subgraph.branches.keys())
