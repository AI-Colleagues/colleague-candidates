"""Tests for the Theme Reporter workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("qualitative_research_desk/theme_reporter")


async def test_orcheo_workflow_builds_the_router_and_pipeline() -> None:
    """The router agent dispatches to the report pipeline or export/respond."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "load_attachments",
        "validate_files",
        "router_agent",
        "extract_ai_message",
        "export_report",
        "generate_report",
        "review_report",
    }
    assert graph.edges == {
        (START, "load_attachments"),
        ("load_attachments", "validate_files"),
        ("generate_report", "review_report"),
        ("review_report", "router_agent"),
        ("export_report", END),
        ("extract_ai_message", END),
    }
    assert {"validate_files", "router_agent"} <= set(graph.branches.keys())


async def test_generate_report_subgraph_wires_the_synthesis_pipeline() -> None:
    """The nested subgraph selects quotes, generates insights, then reports."""
    graph = await workflow.orcheo_workflow()

    subgraph = graph.nodes["generate_report"].runnable.builder

    assert set(subgraph.nodes.keys()) == {
        "ingest",
        "quote_selector_prepare",
        "quote_selector",
        "quote_selector_finalize",
        "insight_generator_prepare",
        "insight_generator",
        "insight_generator_finalize",
        "insight_critic",
        "recommendation_generator",
        "report_output",
    }
    assert subgraph.edges == {
        (START, "ingest"),
        ("quote_selector", "quote_selector_finalize"),
        ("quote_selector_finalize", "insight_generator_prepare"),
        ("insight_generator", "insight_generator_finalize"),
        ("insight_generator_finalize", "insight_critic"),
        ("insight_critic", "recommendation_generator"),
        ("recommendation_generator", "report_output"),
        ("report_output", END),
    }
    assert {"ingest", "quote_selector_prepare", "insight_generator_prepare"} <= set(
        subgraph.branches.keys()
    )
