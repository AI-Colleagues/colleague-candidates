# /// orcheo
# name = "Feed Curator"
# handle = "feed-curator"
# description = "Pull curated RSS feeds and store new items in MongoDB."
# version = "0.1.0"
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# avatar = "avatar-06"
# subtitle = "RSS ingestion and curation"
# ///

"""Feed Curator: pull curated RSS feeds and store new items in MongoDB.

Fetches RSS feeds from a curated list every 30 minutes,
adds a ``read: false`` flag to each entry, and inserts them into MongoDB.

RSS source URLs are maintained in the companion ``config.json``
and uploaded via ``--config-file config.json``.
"""

from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.connectors.rss import RSSNode
from orcheo.nodes.mongodb import MongoDBUpsertManyNode
from orcheo.nodes.triggers import CronTriggerNode


async def orcheo_workflow() -> StateGraph:
    """Build and return the Feed Curator workflow."""
    graph = StateGraph(State)

    graph.add_node(
        "cron_trigger",
        CronTriggerNode(
            name="cron_trigger",
            expression="*/30 * * * *",
        ),
    )

    graph.add_node(
        "fetch_rss",
        RSSNode(
            name="fetch_rss",
            sources="{{config.configurable.rss_sources}}",
        ),
    )

    graph.add_node(
        "store_rss",
        MongoDBUpsertManyNode(
            name="store_rss",
            database="{{config.configurable.mongodb_database}}",
            collection="{{config.configurable.mongodb_collection}}",
            source_result_key="fetch_rss",
            filter_fields=["link"],
            exclude_fields=["read"],
            set_on_insert={"read": False},
        ),
    )

    graph.add_edge(START, "cron_trigger")
    graph.add_edge("cron_trigger", "fetch_rss")
    graph.add_edge("fetch_rss", "store_rss")
    graph.add_edge("store_rss", END)

    return graph
