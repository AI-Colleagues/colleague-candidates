# /// orcheo
# name = "Guess My Number"
# handle = "guess-my-number"
# description = "Human-in-the-loop number guessing with LangGraph interrupts."
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# avatar = "avatar-02"
# subtitle = "Interactive guessing game"
# ///

"""Guess My Number workflow to demonstrate using HumanInputNode."""

from orcheo.graph import END, START, RunnableConfig, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode
from orcheo.nodes.data import CodeNode
from orcheo.nodes.logic import ExtractAIMessageNode, HumanInputNode
from orcheo.schema import Any, BaseModel, Field, Literal


class AgentDecision(BaseModel):
    """Structured agent decision for routing and human-facing text."""

    branch: Literal["human", "finish"] = Field(  # noqa: F821
        description="Next branch selected by the agent."
    )
    assistant_message: str = Field(description="Message to show to the human.")
    parsed_guess: int | None = Field(
        default=None,
        description="Human guess parsed as an integer, when available.",
    )


class RNGNode(CodeNode):
    """Generate and persist the secret integer as a CodeNode state update."""

    minimum: int | str = 0
    maximum: int | str = 100

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Return the existing secret or generate a new one."""
        configurable = config.get("configurable", {})
        seed_text = (
            str(configurable.get("thread_id", "")) + "|" + repr(state.get("inputs", {}))
        )
        seed = 2166136261
        for character in seed_text:
            seed = (seed ^ ord(character)) * 16777619
            seed = seed % 4294967296

        minimum = int(self.minimum)
        maximum = int(self.maximum)
        if minimum > maximum:
            msg = "Minimum number must be less than or equal to maximum number."
            raise ValueError(msg)

        span = maximum - minimum + 1
        number_to_guess = minimum + (seed % span)
        result = {
            "number_to_guess": number_to_guess,
            "range": {"min": minimum, "max": maximum},
        }

        return result


class PrepareHumanPromptNode(CodeNode):
    """Prepare the interrupt payload for the next human turn."""

    interrupt_kind: str = "guess_my_number"
    minimum: int | str = 0
    maximum: int | str = 100

    async def run(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Store the next human prompt and expected response schema."""
        prompt = state.get("structured_response").get("assistant_message")
        minimum = int(self.minimum)
        maximum = int(self.maximum)
        return {
            "prompt": prompt,
            "kind": self.interrupt_kind,
            "expected": {
                "type": "integer",
                "minimum": minimum,
                "maximum": maximum,
            },
        }


async def orcheo_workflow() -> StateGraph:
    """Build the Guess My Number workflow graph."""
    graph = StateGraph(State)

    graph.add_node(
        "random_number",
        RNGNode(
            name="random_number",
            minimum="{{config.configurable.minimum}}",
            maximum="{{config.configurable.maximum}}",
        ),
    )
    graph.add_node(
        "agent",
        AgentNode(
            name="agent",
            ai_model="{{config.configurable.ai_model}}",
            model_kwargs={"api_key": "[[openai_api_key]]"},
            system_prompt=(
                "You are the game master for Guess My Number.\n\n"
                "The secret integer is {{results.random_number.number_to_guess}}. "
                "Never reveal it unless the human guesses it exactly.\n"
                "The valid range is {{results.random_number.range.min}} to "
                "{{results.random_number.range.max}} inclusive.\n"
                "Use the conversation messages to identify the latest human guess. "
                "Ignore assistant prompts when deciding whether a guess was made.\n\n"
                "Return a structured AgentDecision. Put the human-facing text in "
                "assistant_message. Choose branch='human' when the game should "
                "continue "
                "and branch='finish' only when the latest valid integer guess equals "
                "the secret number.\n\n"
                "Rules:\n"
                "- If there is no latest human guess yet, ask for the first guess.\n"
                "- If the latest message does not contain one integer in the valid "
                "range, "
                "ask for one whole-number guess in that range.\n"
                "- If the guess is lower than the secret, say it is too low and ask "
                "again.\n"
                "- If the guess is higher than the secret, say it is too high and ask "
                "again.\n"
                "- If the guess equals the secret, congratulate the human.\n"
                "- Keep the message short and direct."
            ),
            response_format=AgentDecision,
            use_graph_chat_history=False,
            max_messages=20,
        ),
    )
    graph.add_node(
        "prepare_human",
        PrepareHumanPromptNode(
            name="prepare_human",
            fallback_prompt="What is your guess?",
            interrupt_kind="guess_my_number",
            minimum="{{config.configurable.minimum}}",
            maximum="{{config.configurable.maximum}}",
        ),
    )
    graph.add_node(
        "human_input",
        HumanInputNode(
            name="human_input",
            prompt="{{results.prepare_human.prompt}}",
            kind="{{results.prepare_human.kind}}",
            expected="{{results.prepare_human.expected}}",
        ),
    )
    graph.add_node(
        "extract_ai_message",
        ExtractAIMessageNode(
            name="extract_ai_message",
            fallback_message="Game finished.",
        ),
    )

    graph.add_edge(START, "random_number")
    graph.add_edge("random_number", "agent")
    graph.add_conditional_edges(
        "agent",
        {
            "path": "structured_response.branch",
            "mapping": {"finish": "extract_ai_message"},
            "default": "prepare_human",
        },
    )
    graph.add_edge("prepare_human", "human_input")
    graph.add_edge("human_input", "agent")
    graph.add_edge("extract_ai_message", END)

    return graph
