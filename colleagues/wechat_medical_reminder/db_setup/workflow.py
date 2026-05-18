# /// orcheo
# name = "Reminder DB Setup"
# handle = "db-setup"
# description = "Creates the MongoDB collections and indexes for the reminder system."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# emoji = "🗄️"
# subtitle = "AI Setup Utility"
# ///

"""WeChat Medical Reminder - DB Setup workflow.

Admin workflow to create the required MongoDB database, collections, and
indexes for the WeChat Medical Reminder system. Running this workflow is
idempotent — re-running it is safe because ``create_index`` is a no-op
when the index already exists.

Configurable inputs (config.json):
- reminder_database (MongoDB database name)
- registered_users_collection (collection for user profiles)
- user_records_collection (collection for status reports)

Orcheo vault secrets required:
- mdb_connection_string: MongoDB connection string
"""

from langgraph.graph import END, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.mongodb import MongoDBNode


async def orcheo_workflow() -> StateGraph:
    """Build the DB Setup workflow."""
    graph = StateGraph(State)

    # --- Create registered_users collection + compound index ---
    graph.add_node(
        "create_registered_users_index",
        MongoDBNode(
            name="create_registered_users_index",
            database="{{config.configurable.reminder_database}}",
            collection=("{{config.configurable.registered_users_collection}}"),
            operation="create_index",
            query={"external_userid": 1, "open_kf_id": 1},
            options={"name": "idx_userid_kfid", "unique": True},
        ),
    )

    # --- Create user_records collection + compound index ---
    graph.add_node(
        "create_user_records_index",
        MongoDBNode(
            name="create_user_records_index",
            database="{{config.configurable.reminder_database}}",
            collection="{{config.configurable.user_records_collection}}",
            operation="create_index",
            query={"external_userid": 1, "open_kf_id": 1, "record_date": 1},
            options={"name": "idx_userid_kfid_date", "unique": True},
        ),
    )

    # --- Edges ---
    graph.set_entry_point("create_registered_users_index")
    graph.add_edge("create_registered_users_index", "create_user_records_index")
    graph.add_edge("create_user_records_index", END)

    return graph
