# /// orcheo
# name = "Index Steward"
# handle = "index-steward"
# description = "Ensure MongoDB Atlas text and vector search indexes exist."
# version = "0.1.0"
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# avatar = "avatar-14"
# subtitle = "Search Infrastructure"
# ///

"""Index Steward workflow for MongoDB Atlas Search.

Configurable inputs (config.json):
- database: MongoDB database name
- collection: MongoDB collection name
- fields: Atlas Search field mappings for the text index
- dimensions: Vector embedding dimensions
- vector_path: Document field containing vector embeddings
"""

from orcheo.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.storage.mongodb import (
    MongoDBEnsureSearchIndexNode,
    MongoDBEnsureVectorIndexNode,
)


async def orcheo_workflow() -> StateGraph:
    """Build a workflow to ensure text and vector search indexes exist."""
    text_index = MongoDBEnsureSearchIndexNode(
        name="ensure_text_index",
        database="{{config.configurable.database}}",
        collection="{{config.configurable.collection}}",
        definition={
            "mappings": {
                "dynamic": False,
                # Nested template values are resolved recursively by Orcheo.
                "fields": "{{config.configurable.fields}}",
            }
        },
        mode="ensure_or_update",
    )

    vector_index = MongoDBEnsureVectorIndexNode(
        name="ensure_vector_index",
        database="{{config.configurable.database}}",
        collection="{{config.configurable.collection}}",
        dimensions="{{config.configurable.dimensions}}",
        similarity="cosine",
        path="{{config.configurable.vector_path}}",
        mode="ensure_or_update",
    )

    graph = StateGraph(State)
    graph.add_node("ensure_text_index", text_index)
    graph.add_node("ensure_vector_index", vector_index)
    graph.add_edge(START, "ensure_text_index")
    graph.add_edge("ensure_text_index", "ensure_vector_index")
    graph.add_edge("ensure_vector_index", END)
    return graph
