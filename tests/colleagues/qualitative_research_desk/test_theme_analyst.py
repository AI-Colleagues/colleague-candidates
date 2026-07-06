"""Tests for the Theme Analyst workflow."""

from orcheo.graph import END, START
from orcheo.graph.state import State
from tests.conftest import load_workflow_module


workflow = load_workflow_module("qualitative_research_desk/theme_analyst")


async def test_orcheo_workflow_builds_the_router_and_three_pipelines() -> None:
    """The router agent dispatches across codebook, recode, and report work."""
    graph = await workflow.orcheo_workflow()

    assert set(graph.nodes.keys()) == {
        "load_attachments",
        "validate_codebook_files",
        "validate_recode_files",
        "validate_report_files",
        "resolve_inputs",
        "router_agent",
        "extract_ai_message",
        "export_codebook",
        "export_coded_data",
        "export_report",
        "generate_codebook",
        "recode_data",
        "generate_report",
        "review_codebook",
        "review_coded_data",
        "review_report",
    }
    assert graph.edges == {
        (START, "load_attachments"),
        ("load_attachments", "validate_codebook_files"),
        ("validate_codebook_files", "validate_recode_files"),
        ("validate_recode_files", "validate_report_files"),
        ("validate_report_files", "resolve_inputs"),
        ("resolve_inputs", "router_agent"),
        ("generate_codebook", "review_codebook"),
        ("review_codebook", "resolve_inputs"),
        ("recode_data", "review_coded_data"),
        ("review_coded_data", "resolve_inputs"),
        ("generate_report", "review_report"),
        ("review_report", "resolve_inputs"),
        ("export_codebook", END),
        ("export_coded_data", END),
        ("export_report", END),
        ("extract_ai_message", END),
    }
    assert "router_agent" in graph.branches


async def test_generate_codebook_subgraph_wires_the_open_coding_loop() -> None:
    """The nested codebook subgraph loops open coding then consolidates."""
    graph = await workflow.orcheo_workflow()

    subgraph = graph.nodes["generate_codebook"].runnable.builder

    assert set(subgraph.nodes.keys()) == {
        "codebook_ingest",
        "open_coder_prepare",
        "open_coder",
        "open_coder_finalize",
        "codebook_consolidator_prepare",
        "codebook_consolidator",
        "codebook_consolidator_finalize",
        "codebook_output",
    }
    assert (START, "codebook_ingest") in subgraph.edges
    assert ("codebook_output", END) in subgraph.edges


async def test_recode_data_subgraph_wires_the_recoding_loop() -> None:
    """The nested recode subgraph runs data quality checks then recodes."""
    graph = await workflow.orcheo_workflow()

    subgraph = graph.nodes["recode_data"].runnable.builder

    assert set(subgraph.nodes.keys()) == {
        "recode_ingest",
        "data_quality",
        "recoder_prepare",
        "recoder",
        "recoder_finalize",
        "recode_output",
    }
    assert (START, "recode_ingest") in subgraph.edges
    assert ("recode_output", END) in subgraph.edges


async def test_generate_report_subgraph_wires_the_synthesis_pipeline() -> None:
    """The nested report subgraph selects quotes and synthesises insights."""
    graph = await workflow.orcheo_workflow()

    subgraph = graph.nodes["generate_report"].runnable.builder

    assert set(subgraph.nodes.keys()) == {
        "report_ingest",
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
    assert (START, "report_ingest") in subgraph.edges
    assert ("report_output", END) in subgraph.edges


class TestResolveMergedInputsNode:
    """Tests for merging uploaded and chained pipeline inputs."""

    async def test_nothing_provided_yields_all_flags_false(self) -> None:
        """With no prior node results, every readiness flag is false."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State({})

        result = await node.run(state, {})

        assert result["draft_codebook"] is None
        assert result["codebook_gen_ready"] is False
        assert result["recode_ready"] is False
        assert result["report_ready"] is False
        assert result["report_uploaded_ready"] is False
        assert result["report_chained_ready"] is False
        assert result["has_draft_codebook"] is False
        assert result["has_recoded_data"] is False
        assert result["codebook_gen_status"] == (
            "not ready: no valid files were uploaded in this conversation"
        )
        assert result["recode_status"].startswith("not ready:")
        assert result["report_status"].startswith("not ready:")
        assert result["previous_objective"] == "(none)"

    async def test_non_dict_node_results_is_tolerated(self) -> None:
        """A missing or non-dict ``node_results`` value falls back to empty."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State({"node_results": "oops"})

        result = await node.run(state, {})

        assert result["codebook_gen_ready"] is False

    async def test_codebook_generation_ready_from_uploaded_raw_data(self) -> None:
        """Uploaded raw data makes codebook generation ready."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_codebook_files": {"source_payload": "raw,data\n1,2"},
                }
            }
        )

        result = await node.run(state, {})

        assert result["codebook_gen_ready"] is True

    async def test_recode_falls_back_to_generated_codebook_and_payload(self) -> None:
        """Recoding reuses the generated codebook/payload when not re-uploaded."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_codebook_files": {"source_payload": "raw-data"},
                    "codebook_consolidator_finalize": {
                        "draft_codebook": {"themes": ["a"]},
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["has_draft_codebook"] is True
        assert result["recode_source_payload"] == "raw-data"
        assert result["recode_codebook"] == {"themes": ["a"]}
        assert result["recode_ready"] is True

    async def test_recode_prefers_explicit_upload_over_fallback(self) -> None:
        """An explicitly uploaded recode payload/codebook wins over fallback."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_codebook_files": {"source_payload": "raw-data"},
                    "codebook_consolidator_finalize": {
                        "draft_codebook": {"themes": ["a"]},
                    },
                    "validate_recode_files": {
                        "source_payload": "own-data",
                        "approved_codebook": {"themes": ["b"]},
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["recode_source_payload"] == "own-data"
        assert result["recode_codebook"] == {"themes": ["b"]}

    async def test_report_uses_uploaded_coded_data_when_not_chained(self) -> None:
        """Report can use directly uploaded, validated coded data."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_report_files": {
                        "ok": True,
                        "source_payload": "coded-data",
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["report_uploaded_ready"] is True
        assert result["report_chained_ready"] is False
        assert result["report_ready"] is True
        assert result["report_source_payload"] == "coded-data"
        assert result["report_status"] == (
            "ready: the uploaded coded-data file will be used"
        )

    async def test_report_prefers_freshly_chained_results_over_stale_upload(
        self,
    ) -> None:
        """Freshly recoded results in-thread win over a stale uploaded export."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_report_files": {
                        "ok": True,
                        "source_payload": "stale-export",
                    },
                    "validate_recode_files": {
                        "approved_codebook": {"themes": ["a"]},
                    },
                    "data_quality": {"units": [{"id": 1}]},
                    "recoder_finalize": {"assignments": [{"unit_id": 1}]},
                }
            }
        )

        result = await node.run(state, {})

        assert result["report_chained_ready"] is True
        assert result["report_ready"] is True
        assert result["has_recoded_data"] is True
        # Chained results are used directly; no upload fallback is populated.
        assert result["report_source_payload"] is None
        assert result["report_status"] == (
            "ready: recoded results produced in this conversation will be used"
        )

    async def test_report_codebook_prefers_directly_validated_codebook(self) -> None:
        """A codebook validated directly for reporting skips both fallbacks."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_report_files": {
                        "ok": True,
                        "approved_codebook": {"themes": ["direct"]},
                    },
                    "validate_recode_files": {
                        "approved_codebook": {"themes": ["from-recode"]},
                    },
                    "codebook_consolidator_finalize": {
                        "draft_codebook": {"themes": ["draft"]},
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["report_codebook"] == {"themes": ["direct"]}

    async def test_report_falls_back_to_recode_payload_when_not_chained(
        self,
    ) -> None:
        """Without validated or chained report data, recode's payload is reused."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_recode_files": {"source_payload": "recode-data"},
                }
            }
        )

        result = await node.run(state, {})

        assert result["report_chained_ready"] is False
        assert result["report_uploaded_ready"] is False
        assert result["report_source_payload"] == "recode-data"

    async def test_status_lines_surface_validation_errors(self) -> None:
        """Not-ready statuses relay the validator error strings."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "validate_report_files": {
                        "ok": False,
                        "errors": ["No valid data file found.", "bad.csv: unreadable"],
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["report_status"] == (
            "not ready: No valid data file found.; bad.csv: unreadable"
        )

    async def test_previous_objective_prefers_report_then_prepare_nodes(self) -> None:
        """The previous objective is recovered from earlier pipeline runs."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "open_coder_prepare": {"objective": "understand churn"},
                    "quote_selector_prepare": {"objective": "(not provided)"},
                }
            }
        )

        result = await node.run(state, {})

        assert result["previous_objective"] == "understand churn"

    async def test_previous_objective_prefers_freshly_recoded_objective(
        self,
    ) -> None:
        """A newer codebook-stage objective wins over a stale report objective."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "report_output": {"research_objective": "objective A"},
                    "quote_selector_prepare": {"objective": "objective A"},
                    "open_coder_prepare": {"objective": "objective B"},
                    "resolve_inputs": {
                        "_report_objective_seen": "objective A",
                        "_codebook_objective_seen": "objective A",
                        "_previous_objective_source": "report",
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["previous_objective"] == "objective B"
        assert result["_previous_objective_source"] == "codebook"

    async def test_previous_objective_stays_stable_when_neither_stage_changed(
        self,
    ) -> None:
        """An unrelated pipeline run (e.g. recoding) does not flip the objective."""
        node = workflow.ResolveMergedInputsNode(name="resolve_inputs")
        state = State(
            {
                "node_results": {
                    "report_output": {"research_objective": "objective A"},
                    "quote_selector_prepare": {"objective": "objective A"},
                    "open_coder_prepare": {"objective": "objective B"},
                    "resolve_inputs": {
                        "_report_objective_seen": "objective A",
                        "_codebook_objective_seen": "objective B",
                        "_previous_objective_source": "codebook",
                    },
                }
            }
        )

        result = await node.run(state, {})

        assert result["previous_objective"] == "objective B"
        assert result["_previous_objective_source"] == "codebook"
