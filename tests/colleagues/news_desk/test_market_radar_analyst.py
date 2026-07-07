"""Tests for the Market Radar Analyst workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("news_desk/market_radar_analyst")


async def test_orcheo_workflow_builds_the_radar_pipeline() -> None:
    """The workflow triggers on a schedule, processes unread items, and reports."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "cron_trigger",
        "find_unread",
        "segment_units",
        "open_coder_prepare",
        "open_coder",
        "open_coder_finalize",
        "codebook_consolidator_prepare",
        "codebook_consolidator",
        "codebook_consolidator_finalize",
        "recoder_prepare",
        "recoder",
        "recoder_finalize",
        "quantify",
        "quote_selector_prepare",
        "quote_selector",
        "quote_selector_finalize",
        "insight_generator_prepare",
        "insight_generator",
        "insight_generator_finalize",
        "insight_critic",
        "recommendation_generator",
        "compose_report",
        "send_report",
        "mark_read",
    }
    assert graph.edges == {
        (START, "cron_trigger"),
        ("cron_trigger", "find_unread"),
        ("find_unread", "segment_units"),
        ("open_coder", "open_coder_finalize"),
        ("codebook_consolidator", "codebook_consolidator_finalize"),
        ("codebook_consolidator_finalize", "recoder_prepare"),
        ("recoder", "recoder_finalize"),
        ("quote_selector", "quote_selector_finalize"),
        ("quote_selector_finalize", "insight_generator_prepare"),
        ("insight_generator", "insight_generator_finalize"),
        ("insight_generator_finalize", "insight_critic"),
        ("insight_critic", "recommendation_generator"),
        ("recommendation_generator", "compose_report"),
        ("compose_report", "send_report"),
        ("mark_read", END),
    }
    assert {
        "segment_units",
        "open_coder_prepare",
        "open_coder_finalize",
        "codebook_consolidator_prepare",
        "recoder_prepare",
        "recoder_finalize",
        "quantify",
        "quote_selector_prepare",
        "insight_generator_prepare",
        "send_report",
    } <= set(graph.branches.keys())
