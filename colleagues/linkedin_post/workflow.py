# /// orcheo
# name = "LinkedIn Publisher"
# handle = "linkedin-post"
# description = "Publishes posts to LinkedIn using vault-stored credentials."
# config = "./config.json"
# entrypoint = "orcheo_workflow"
# avatar = "avatar-06"
# subtitle = "AI Social Media"
# ///

"""LinkedIn posting workflow backed by Orcheo vault credentials."""

from langgraph.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.connectors.linkedin import LinkedInPostNode


async def orcheo_workflow() -> StateGraph:
    """Build the LinkedIn posting workflow."""
    graph = StateGraph(State)
    graph.add_node(
        "post_linkedin",
        LinkedInPostNode(
            name="post_linkedin",
        ),
    )
    graph.add_edge(START, "post_linkedin")
    graph.add_edge("post_linkedin", END)
    return graph
