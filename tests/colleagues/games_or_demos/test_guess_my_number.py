"""Tests for the Guess My Number workflow."""

import pytest
from orcheo.graph import END, START
from orcheo.graph.state import State
from tests.conftest import load_workflow_module


workflow = load_workflow_module("games_or_demos/guess_my_number")


async def test_orcheo_workflow_builds_the_game_loop_graph() -> None:
    """The workflow wires the RNG, agent, and human-input loop nodes."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "random_number",
        "agent",
        "prepare_human",
        "human_input",
        "extract_ai_message",
    }
    assert graph.edges == {
        (START, "random_number"),
        ("random_number", "agent"),
        ("prepare_human", "human_input"),
        ("human_input", "agent"),
        ("extract_ai_message", END),
    }
    assert "agent" in graph.branches


class TestRNGNode:
    """Tests for the secret-number generator node."""

    async def test_generates_number_within_configured_range(self) -> None:
        """The generated number stays within the configured min/max bounds."""
        node = workflow.RNGNode(name="random_number", minimum=5, maximum=10)
        state = State({"inputs": {"message": "hello"}})
        config = {"configurable": {"thread_id": "thread-1"}}

        result = await node.run(state, config)

        assert 5 <= result["number_to_guess"] <= 10
        assert result["range"] == {"min": 5, "max": 10}

    async def test_same_thread_and_inputs_produce_deterministic_number(self) -> None:
        """The same thread id and inputs seed the same secret number."""
        node = workflow.RNGNode(name="random_number", minimum=0, maximum=100)
        state = State({"inputs": {"message": "hello"}})
        config = {"configurable": {"thread_id": "thread-1"}}

        first = await node.run(state, config)
        second = await node.run(state, config)

        assert first == second

    async def test_accepts_string_typed_bounds(self) -> None:
        """Template-resolved string bounds are coerced to integers."""
        node = workflow.RNGNode(name="random_number", minimum="0", maximum="5")
        state = State({"inputs": {}})

        result = await node.run(state, {})

        assert result["range"] == {"min": 0, "max": 5}

    async def test_raises_when_minimum_exceeds_maximum(self) -> None:
        """An inverted range is rejected."""
        node = workflow.RNGNode(name="random_number", minimum=10, maximum=1)
        state = State({"inputs": {}})

        with pytest.raises(ValueError, match="Minimum number must be"):
            await node.run(state, {})


class TestPrepareHumanPromptNode:
    """Tests for the human-input interrupt payload builder."""

    async def test_builds_interrupt_payload_from_assistant_message(self) -> None:
        """The prepared prompt echoes the agent's assistant message and range."""
        node = workflow.PrepareHumanPromptNode(
            name="prepare_human",
            interrupt_kind="guess_my_number",
            minimum=1,
            maximum=50,
        )
        state = State(
            {
                "structured_response": {
                    "branch": "human",
                    "assistant_message": "What's your guess?",
                }
            }
        )

        result = await node.run(state, {})

        assert result == {
            "prompt": "What's your guess?",
            "kind": "guess_my_number",
            "expected": {"type": "integer", "minimum": 1, "maximum": 50},
        }
