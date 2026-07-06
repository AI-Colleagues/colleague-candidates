"""Tests for the Feed Curator workflow graph assembly."""

from orcheo.graph import END, START
from tests.conftest import load_workflow_module


workflow = load_workflow_module("news_desk/feed_curator")


async def test_orcheo_workflow_builds_cron_to_storage_pipeline() -> None:
    """The workflow triggers on a cron schedule, fetches, then stores RSS items."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {"cron_trigger", "fetch_rss", "store_rss"}
    assert graph.edges == {
        (START, "cron_trigger"),
        ("cron_trigger", "fetch_rss"),
        ("fetch_rss", "store_rss"),
        ("store_rss", END),
    }
