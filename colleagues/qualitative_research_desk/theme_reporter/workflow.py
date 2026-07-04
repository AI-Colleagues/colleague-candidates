# /// orcheo
# name = "Theme Reporter"
# handle = "theme-reporter"
# description = "Synthesise theme insights and render a research report."
# version = "1.0.0"
# entrypoint = "orcheo_workflow"
# config = "./config.json"
# avatar = "avatar-14"
# subtitle = "Theme synthesis and reporting"
#
# [[updates]]
# version = "1.0.0"
# summary = "Renames Insight Reporter to Theme Reporter."
# migration = "Re-onboard under new handle; old handle was `insight-reporter`."
# ///

"""Theme Reporter: turn coded data into an evidence-grounded report."""

from orcheo.graph import END, START, StateGraph
from orcheo.graph.state import State
from orcheo.nodes.ai import AgentNode, LLMNode
from orcheo.nodes.logic import HumanInputNode
from orcheo.nodes.logic.routing import ExtractAIMessageNode
from orcheo.nodes.qualitative import (
    CodedDataIngestNode,
    ExportReportNode,
    InsightCriticNode,
    InsightGenerationResponse,
    LLMStageFinalizeNode,
    LLMStagePrepareNode,
    LoadAttachmentsNode,
    QuoteSelectionResponse,
    RecommendationGeneratorNode,
    ReportOutputNode,
    ValidateFilesNode,
)
from orcheo.schema import BaseModel, Field, Literal


class RoutingDecision(BaseModel):
    """Structured router output for the Theme Reporter entry agent."""

    branch: Literal["generate_report", "export_report", "respond"] = "respond"  # noqa: F821
    assistant_message: str | None = Field(
        default=None,
        description="Reply to show the user when branch is respond.",
    )
    research_objective: str | None = Field(
        default=None,
        description=(
            "Research objective extracted from the user's message when routing "
            "to generate_report."
        ),
    )


async def orcheo_workflow() -> StateGraph:
    """Build the Theme Reporter workflow graph."""
    generate_report = StateGraph(State)

    generate_report.add_node(
        "ingest",
        CodedDataIngestNode(
            name="ingest",
            source_payload="{{results.validate_files.source_payload}}",
            approved_codebook="{{results.validate_files.approved_codebook}}",
            units_field="units",
            assignments_field="code_assignments",
            approved_codebook_field="approved_codebook",
            quantification_field="quantification",
            cooccurrence_field="cooccurrence",
            segment_breakdowns_field="segment_breakdowns",
            segment_comparisons_field="segment_comparisons",
        ),
    )
    generate_report.add_node(
        "quote_selector_prepare",
        LLMStagePrepareNode(
            name="quote_selector_prepare",
            stage="quote_selector",
            research_objective="{{structured_response.research_objective}}",
            units="{{results.ingest.units}}",
            code_assignments="{{results.ingest.code_assignments}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            quantification="{{results.ingest.quantification}}",
            quote_selector_system_prompt_template=(
                "You are selecting representative verbatim quotes for a research "
                "report. Research objective:\n{objective}\n\n"
                "Return concise quotes bound to existing theme_id and unit_id "
                "values only."
            ),
        ),
    )
    generate_report.add_node(
        "quote_selector",
        LLMNode(
            name="quote_selector",
            ai_model="{{config.configurable.ai_model}}",
            model_kwargs={"api_key": "[[openai_api_key]]"},
            system_prompt="{{results.quote_selector_prepare.system_prompt}}",
            input_text="{{results.quote_selector_prepare.input_text}}",
            response_format=QuoteSelectionResponse,
        ),
    )
    generate_report.add_node(
        "quote_selector_finalize",
        LLMStageFinalizeNode(
            name="quote_selector_finalize",
            stage="quote_selector",
            units="{{results.ingest.units}}",
            code_assignments="{{results.ingest.code_assignments}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            selected_quotes_field="selected_quotes",
            response_schema=QuoteSelectionResponse,
        ),
    )
    generate_report.add_node(
        "insight_generator_prepare",
        LLMStagePrepareNode(
            name="insight_generator_prepare",
            stage="insight_generator",
            research_objective="{{structured_response.research_objective}}",
            units="{{results.ingest.units}}",
            code_assignments="{{results.ingest.code_assignments}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            quantification="{{results.ingest.quantification}}",
            selected_quotes="{{results.quote_selector_finalize.selected_quotes}}",
            insight_generator_system_prompt_template=(
                "You are synthesising evidence-grounded research insights. "
                "Research objective:\n{objective}\n\n"
                "Use only supplied codebook, quantification, assignments, and "
                "quotes. Each insight must include at least one supporting "
                "code_id and unit_id."
            ),
        ),
    )
    generate_report.add_node(
        "insight_generator",
        LLMNode(
            name="insight_generator",
            ai_model="{{config.configurable.ai_model}}",
            model_kwargs={"api_key": "[[openai_api_key]]"},
            system_prompt="{{results.insight_generator_prepare.system_prompt}}",
            input_text="{{results.insight_generator_prepare.input_text}}",
            response_format=InsightGenerationResponse,
        ),
    )
    generate_report.add_node(
        "insight_generator_finalize",
        LLMStageFinalizeNode(
            name="insight_generator_finalize",
            stage="insight_generator",
            units="{{results.ingest.units}}",
            code_assignments="{{results.ingest.code_assignments}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            quantification="{{results.ingest.quantification}}",
            candidate_insights_field="candidate_insights",
            response_schema=InsightGenerationResponse,
        ),
    )
    generate_report.add_node(
        "insight_critic",
        InsightCriticNode(
            name="insight_critic",
            units="{{results.ingest.units}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            code_assignments="{{results.ingest.code_assignments}}",
            segment_comparisons="{{results.ingest.segment_comparisons}}",
            candidate_insights=(
                "{{results.insight_generator_finalize.candidate_insights}}"
            ),
            candidate_insights_field="candidate_insights",
        ),
    )
    generate_report.add_node(
        "recommendation_generator",
        RecommendationGeneratorNode(
            name="recommendation_generator",
            candidate_insights="{{results.insight_critic.candidate_insights}}",
            candidate_insights_field="candidate_insights",
            recommendations_field="recommendations",
            approved_insight_ids_field="approved_insight_ids",
        ),
    )
    generate_report.add_node(
        "report_output",
        ReportOutputNode(
            name="report_output",
            research_objective="{{structured_response.research_objective}}",
            source_payload="{{results.validate_files.source_payload}}",
            units="{{results.ingest.units}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            code_assignments="{{results.ingest.code_assignments}}",
            quantification="{{results.ingest.quantification}}",
            cooccurrence="{{results.ingest.cooccurrence}}",
            segment_breakdowns="{{results.ingest.segment_breakdowns}}",
            segment_comparisons="{{results.ingest.segment_comparisons}}",
            selected_quotes="{{results.quote_selector_finalize.selected_quotes}}",
            candidate_insights="{{results.recommendation_generator.candidate_insights}}",
            recommendations="{{results.recommendation_generator.recommendations}}",
            approved_insight_ids=(
                "{{results.recommendation_generator.approved_insight_ids}}"
            ),
            ingest_node_name="ingest",
        ),
    )

    generate_report.add_edge(START, "ingest")
    generate_report.add_conditional_edges(
        "ingest",
        {
            "path": "results.ingest.halt",
            "mapping": {"true": "report_output", "false": "quote_selector_prepare"},
        },
    )
    generate_report.add_conditional_edges(
        "quote_selector_prepare",
        {
            "path": "results.quote_selector_prepare.skip_llm",
            "mapping": {
                "true": "quote_selector_finalize",
                "false": "quote_selector",
            },
        },
    )
    generate_report.add_edge("quote_selector", "quote_selector_finalize")
    generate_report.add_edge("quote_selector_finalize", "insight_generator_prepare")
    generate_report.add_conditional_edges(
        "insight_generator_prepare",
        {
            "path": "results.insight_generator_prepare.skip_llm",
            "mapping": {
                "true": "insight_generator_finalize",
                "false": "insight_generator",
            },
        },
    )
    generate_report.add_edge("insight_generator", "insight_generator_finalize")
    generate_report.add_edge("insight_generator_finalize", "insight_critic")
    generate_report.add_edge("insight_critic", "recommendation_generator")
    generate_report.add_edge("recommendation_generator", "report_output")
    generate_report.add_edge("report_output", END)

    graph = StateGraph(State)

    graph.add_node("load_attachments", LoadAttachmentsNode(name="load_attachments"))
    graph.add_node(
        "validate_files",
        ValidateFilesNode(
            name="validate_files",
            data_field="source_payload",
            codebook_field="approved_codebook",
            data_kind="coded",
            require_codebook=False,
        ),
    )
    graph.add_node(
        "router_agent",
        AgentNode(
            name="router_agent",
            ai_model="{{config.configurable.ai_model}}",
            model_kwargs={"api_key": "[[openai_api_key]]"},
            system_prompt=(
                "You are the Theme Reporter, an AI qualitative research "
                "assistant.\n\n"
                "File validation is performed automatically before you run. "
                "Use only the programmed validation facts below to decide whether "
                "the required coded data is available. Do not infer file validity "
                "from chat history or raw file text.\n"
                "Validation ok: {{results.validate_files.ok}}\n"
                "Validation summary: {{results.validate_files.assistant_message}}\n"
                "Coded data file: {{results.validate_files.data_file}}\n"
                "Optional codebook file: {{results.validate_files.codebook_file}}\n"
                "Validation errors: {{results.validate_files.errors}}\n\n"
                "Previous research objective, if any:\n"
                "{{results.report_output.research_objective}}\n\n"
                "Your job each turn is to decide ONE next action and return it as "
                "a structured RoutingDecision. You do not run the pipelines "
                "yourself; the graph executes the branch you choose. Treat the "
                "latest user message as the current request, including replies "
                "collected after the report review prompt. The expected input is "
                "the `coded_data.csv` exported by the Theme Coder; a "
                "codebook CSV is optional because the codebook can be "
                "reconstructed from the coded data.\n\n"
                "Set `branch` to one of:\n"
                "- `generate_report` - run the full pipeline and return the "
                "complete Markdown report plus a download link. Requires valid "
                "uploaded coded data and an available research objective.\n"
                "- `export_report` - regenerate the report download link. Route "
                "here only if the user asks for the file again later.\n"
                "- `respond` - reply to the user directly in `assistant_message` "
                "instead of running a pipeline.\n\n"
                "Rules:\n"
                "- If validation fails, do not route to a pipeline. Reply with "
                "the validation errors or summary.\n"
                "- If a research objective is missing, set `branch` to `respond` "
                "and ask for it.\n"
                "- If the user asks to generate, redo, rerun, or revise the "
                "report, route to `generate_report` when a research objective is "
                "available. Reuse the previous research objective if the user "
                "does not provide a new one.\n"
                "- When routing to `generate_report`, extract the user's "
                "research objective into `research_objective`. If reusing the "
                "previous objective, return that exact objective. Do not invent "
                "one.\n"
                "- When the user asks for the report file again, route to "
                "`export_report`.\n"
                "- Only route to a pipeline branch when the user's intent is "
                "clear. Keep direct replies short and action-oriented.\n\n"
                "When validation fails, the correct shape is:\n"
                '{"branch":"respond",'
                '"assistant_message":"Please correct the uploaded files: '
                '<brief errors>"}'
            ),
            response_format=RoutingDecision,
            use_graph_chat_history=False,
            max_messages=50,
        ),
    )
    graph.add_node(
        "extract_ai_message",
        ExtractAIMessageNode(
            name="extract_ai_message",
            fallback_message=(
                "Upload the `coded_data.csv` exported by the Theme Coder "
                "and share your research objective, then I will "
                "synthesise an evidence-grounded insight report."
            ),
        ),
    )
    graph.add_node(
        "export_report",
        ExportReportNode(
            name="export_report",
            research_objective="{{results.report_output.research_objective}}",
            source_payload="{{results.validate_files.source_payload}}",
            units="{{results.ingest.units}}",
            approved_codebook="{{results.ingest.approved_codebook}}",
            code_assignments="{{results.ingest.code_assignments}}",
            quantification="{{results.ingest.quantification}}",
            cooccurrence="{{results.ingest.cooccurrence}}",
            segment_breakdowns="{{results.ingest.segment_breakdowns}}",
            segment_comparisons="{{results.ingest.segment_comparisons}}",
            selected_quotes="{{results.quote_selector_finalize.selected_quotes}}",
            candidate_insights="{{results.recommendation_generator.candidate_insights}}",
            recommendations="{{results.recommendation_generator.recommendations}}",
            approved_insight_ids=(
                "{{results.recommendation_generator.approved_insight_ids}}"
            ),
        ),
    )
    graph.add_node("generate_report", generate_report.compile())
    graph.add_node(
        "review_report",
        HumanInputNode(
            name="review_report",
            prompt="{{assistant_message}}",
            expected={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "Revision request, rerun request, approval, or export "
                            "request"
                        ),
                    }
                },
                "required": ["message"],
            },
        ),
    )

    graph.add_edge(START, "load_attachments")
    graph.add_edge("load_attachments", "validate_files")
    graph.add_conditional_edges(
        "validate_files",
        {
            "path": "results.validate_files.ok",
            "mapping": {"true": "router_agent", "false": END},
        },
    )
    graph.add_conditional_edges(
        "router_agent",
        {
            "path": "structured_response.branch",
            "mapping": {
                "generate_report": "generate_report",
                "export_report": "export_report",
            },
            "default": "extract_ai_message",
        },
    )
    graph.add_edge("generate_report", "review_report")
    graph.add_edge("review_report", "router_agent")
    graph.add_edge("export_report", END)
    graph.add_edge("extract_ai_message", END)

    return graph
