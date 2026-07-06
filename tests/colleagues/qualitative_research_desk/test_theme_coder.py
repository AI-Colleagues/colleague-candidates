"""Tests for the Theme Coder workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("qualitative_research_desk/theme_coder")


async def test_orcheo_workflow_builds_the_router_and_pipelines() -> None:
    """The router agent dispatches to the recoding pipeline or export/respond."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "load_attachments",
        "validate_files",
        "router_agent",
        "extract_ai_message",
        "export_coded_data",
        "recode_data",
        "review_coded_data",
    }
    assert graph.edges == {
        (START, "load_attachments"),
        ("load_attachments", "validate_files"),
        ("recode_data", "review_coded_data"),
        ("review_coded_data", "router_agent"),
        ("export_coded_data", END),
        ("extract_ai_message", END),
    }
    assert {"validate_files", "router_agent"} <= set(graph.branches.keys())


async def test_recode_data_subgraph_wires_the_recoding_loop() -> None:
    """The nested subgraph runs data quality checks then recodes in a loop."""
    graph = await workflow.orcheo_workflow()

    subgraph = graph.nodes["recode_data"].runnable.builder

    assert set(subgraph.nodes.keys()) == {
        "ingest",
        "data_quality",
        "recoder_prepare",
        "recoder",
        "recoder_finalize",
        "recode_output",
    }
    assert subgraph.edges == {
        (START, "ingest"),
        ("data_quality", "recoder_prepare"),
        ("recoder", "recoder_finalize"),
        ("recode_output", END),
    }
    assert {"ingest", "recoder_prepare", "recoder_finalize"} <= set(
        subgraph.branches.keys()
    )
