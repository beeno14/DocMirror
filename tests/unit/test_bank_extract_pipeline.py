# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared bank statement extract pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import docmirror.plugins.bank_statement.extract_pipeline as extract_pipeline
from docmirror.plugins.bank_statement import community_plugin as community_module
from docmirror.plugins.bank_statement.canonical import build_style_meta, records_from_raw_transactions
from docmirror.plugins.bank_statement.canonical_quality import audit_row_accounting
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extract_pipeline import (
    _apply_source_reported_transaction_count,
    _bank_spe_ltro_warnings,
    _physical_logical_row_mismatch_warning,
    enrich_identity_fields,
    is_authoritative_issuer_row_count,
    run_bank_statement_extract,
)
from docmirror.plugins.bank_statement.extraction_dispatch import BankExtractionRoute
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence
from tests.unit.test_pipe_text_table_builder import _synthetic_boc_text


def _spe_with_explicit_table_counts(**overrides):
    spe = {
        "table_extraction": "full",
        "table_extraction_skipped_reason": "no_tabular_signal",
        "physical_table_count": 0,
        "native_table_candidate_count": 0,
        "logical_table_count": None,
        "table_reconstruction_gate": {
            "applicable": False,
            "candidate_count": 0,
            "physical_table_count": 0,
            "passed": True,
        },
    }
    spe.update(overrides)
    return spe


def test_pipe_reconstruction_is_not_called_a_fallback_when_spe_proves_no_table_route():
    assert _bank_spe_ltro_warnings(_spe_with_explicit_table_counts(), "pipe_text") == []


def test_pipe_reconstruction_still_warns_when_spe_has_a_physical_table_candidate():
    spe = _spe_with_explicit_table_counts(
        physical_table_count=1,
        table_extraction_skipped_reason=None,
        table_reconstruction_gate={
            "applicable": True,
            "candidate_count": 1,
            "physical_table_count": 1,
            "passed": True,
        },
    )
    assert "spe:mirror_table_extraction_full_used_ltro_fallback" in _bank_spe_ltro_warnings(
        spe, "pipe_text"
    )


def test_incomplete_spe_candidate_census_does_not_suppress_fallback_warning():
    spe = _spe_with_explicit_table_counts()
    spe.pop("logical_table_count")
    assert "spe:mirror_table_extraction_full_used_ltro_fallback" in _bank_spe_ltro_warnings(
        spe, "pipe_text"
    )


def test_non_pipe_reconstruction_never_gets_pipe_fallback_warning():
    assert _bank_spe_ltro_warnings(_spe_with_explicit_table_counts(), "native_wide_table") == []


def test_enrich_identity_fields_from_header():
    text = _synthetic_boc_text()
    fields = enrich_identity_fields({}, text)
    assert fields["account_holder"]["normalized_value"] == "南京创沃电气设备有限公司"


def test_run_bank_statement_extract_pipe_text():
    plugin = BankStatementCommunityPlugin()
    text = _synthetic_boc_text()
    result = run_bank_statement_extract(None, text, plugin)
    assert result.style_meta.reconstruction_source == "pipe_text"
    assert result.style_meta.extracted_rows >= 1
    assert "account_holder" in result.identity_fields


def test_lazy_final_gate_failure_forces_one_fresh_eager_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment_calls: list[bool] = []
    audit_calls = 0

    def record(day: int) -> dict:
        return {
            "normalized": {
                "date": f"2024-01-{day:02d}",
                "amount": float(day),
                "direction": "income",
            },
            "raw": {"date": f"2024-01-{day:02d}", "amount": str(day)},
            "source": {
                "source_page": 1,
                "page_range": [1, 1],
                "table_id": "table:1",
                "source_row_index": 0,
            },
        }

    def deployment(_builder, _parse_result, _full_text, _plugin, *, adaptive):
        deployment_calls.append(adaptive)
        ctx = StyleContext(
            tables=[],
            full_text="交易总笔数：1",
            institution=None,
            page_count=1,
            reconstruction=ReconstructionMeta(source="canonical_table"),
        )
        detection = StyleDetectionResult(
            primary_style="grid_standard",
            confidence=0.95,
            parser_chain=["grid_standard"],
        )
        diagnostic = {
            "selected_candidate": "parser:grid_standard",
            "deployment_mode": "lazy_primary" if adaptive else "eager_forced",
            "completion_state": "proven" if adaptive else "unknown",
            "completion_reason": "provisional" if adaptive else "adaptive_deployment_disabled",
            "attempted_strategies": ["parser:grid_standard"],
            "skipped_strategies": ["native_wide_table"] if adaptive else [],
        }
        meta = SimpleNamespace(
            tables_parsed=1,
            tables_skipped=0,
            candidate_diagnostics=[diagnostic],
        )
        return ctx, detection, [record(1 if adaptive else 2)], {}, meta

    def audit(*_args, **_kwargs):
        nonlocal audit_calls
        audit_calls += 1
        return ["bank_invariant_failed:provisional"] if audit_calls == 1 else []

    monkeypatch.setattr(extract_pipeline, "resolve_bank_extraction_route", lambda _result: BankExtractionRoute.DIGITAL)
    monkeypatch.setattr(extract_pipeline, "_run_strategy_deployment", deployment)
    monkeypatch.setattr(extract_pipeline, "page_texts_from_parse_result", lambda _result: [])
    monkeypatch.setattr(extract_pipeline, "page_texts_with_business_headers", lambda _result, texts: texts)
    monkeypatch.setattr(
        extract_pipeline,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence(1, "header_total", 0.95),
    )
    monkeypatch.setattr(extract_pipeline, "physical_transaction_row_estimate", lambda _result: 0)
    monkeypatch.setattr(extract_pipeline, "audit_bank_statement_invariants", audit)

    result = run_bank_statement_extract(None, "", BankStatementCommunityPlugin())

    assert deployment_calls == [True, False]
    assert audit_calls == 2
    assert result.records[0]["normalized"]["date"] == "2024-01-02"
    assert "bank_invariant_failed:provisional" not in result.warnings
    assert result.candidate_diagnostics[0]["deployment_mode"] == "eager_final_gate_fallback"
    assert result.candidate_diagnostics[0]["completion_reason"] == "final_invariant_audit_failed"
    assert result.candidate_diagnostics[0]["prior_lazy_attempt"]["deployment_mode"] == "lazy_primary"


def test_strategy_deployment_builds_fresh_contexts_and_preserves_eager_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[StyleContext] = []
    registry_modes: list[bool] = []

    def context_builder(_parse_result, full_text):
        ctx = StyleContext(
            tables=[],
            full_text=full_text,
            institution=None,
            page_count=1,
            reconstruction=ReconstructionMeta(source="canonical_table"),
        )
        contexts.append(ctx)
        return ctx

    class Registry:
        def __init__(self, *, adaptive):
            registry_modes.append(adaptive)

    class Orchestrator:
        def __init__(self, _registry):
            pass

        def run(self, _detection, _ctx, _plugin):
            return [], {}, SimpleNamespace(candidate_diagnostics=[])

    monkeypatch.setattr(extract_pipeline, "BankStyleParserRegistry", Registry)
    monkeypatch.setattr(extract_pipeline, "BankLedgerOrchestrator", Orchestrator)

    adaptive = extract_pipeline._run_strategy_deployment(
        context_builder,
        None,
        "statement",
        BankStatementCommunityPlugin(),
        adaptive=True,
    )
    eager = extract_pipeline._run_strategy_deployment(
        context_builder,
        None,
        "statement",
        BankStatementCommunityPlugin(),
        adaptive=False,
    )

    assert registry_modes == [True, False]
    assert adaptive[0] is contexts[0]
    assert eager[0] is contexts[1]
    assert adaptive[0] is not eager[0]


def test_internal_source_page_survives_canonical_record_materialization():
    records = records_from_raw_transactions(
        [
            {
                "date": "2023-01-01",
                "amount": "10.00",
                "direction": "income",
                "_source_page": "2",
                "_source_table_id": "pt_2_0",
                "_source_row_index": "4",
            }
        ],
        normalize_fn=lambda raw: {
            "date": raw["date"],
            "amount": 10.0,
            "direction": raw["direction"],
        },
        style_id="grid_standard",
    )

    assert records[0]["source"] == {
        "source_page": 2,
        "page_range": [2, 2],
        "table_id": "pt_2_0",
        "source_row_index": 4,
    }


def test_source_raw_replaces_internal_recovery_header_at_canonical_boundary():
    records = records_from_raw_transactions(
        [
            {
                "序号": "1",
                "交易日期": "20211025",
                "交易流水号": "",
                "收入金额": "100.00",
                "余额": "100.00",
                "对方账号": "1234567890",
                "对方户名": "甲公司",
                "摘要": "银联入账",
                "_source_raw": {
                    "序号": "1",
                    "摘要": "银联入账",
                    "币别": "人民币元",
                    "钞汇": "钞",
                    "交易日期": "20211025",
                    "交易金额": "100.00",
                    "账户余额": "100.00",
                    "对方账号与户名": "1234567890/甲公司",
                },
            }
        ],
        normalize_fn=lambda raw: {
            "date": "2021-10-25",
            "amount": 100.0,
            "direction": "income",
        },
        style_id="grid_standard",
    )

    assert list(records[0]["raw"]) == [
        "序号",
        "摘要",
        "币别",
        "钞汇",
        "交易日期",
        "交易金额",
        "账户余额",
        "对方账号与户名",
    ]
    assert "交易流水号" not in records[0]["raw"]


def test_row_accounting_rejects_canonical_dataset_mismatch() -> None:
    warnings = audit_row_accounting(parsed_rows=3, canonical_rows=2, emitted_rows=3)

    assert warnings == ["BANK_CANONICAL_EMITTED_ROW_MISMATCH:canonical=2:emitted=3"]


def test_complete_evidence_recovery_supersedes_sparse_physical_row_estimate() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="canonical_evidence_table",
        canonical_extracted=199,
        expected_primary_rows=199,
    )

    reconstruction = ReconstructionMeta(
        source="canonical_evidence_table",
        expected_primary_rows=199,
        expected_evidence_source="header_total",
        expected_evidence_confidence=0.94,
    )

    assert _physical_logical_row_mismatch_warning(4, style_meta, reconstruction) == ""


def test_candidate_count_alone_does_not_supersede_physical_row_estimate() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="canonical_evidence_table",
        canonical_extracted=199,
        expected_primary_rows=199,
    )
    reconstruction = ReconstructionMeta(
        source="canonical_evidence_table",
        expected_primary_rows=199,
    )

    assert _physical_logical_row_mismatch_warning(4, style_meta, reconstruction) == (
        "BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical=4:canonical=199"
    )


def test_candidate_sequence_does_not_suppress_sparse_physical_row_mismatch() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="canonical_table",
        canonical_extracted=475,
        expected_primary_rows=475,
    )
    reconstruction = ReconstructionMeta(
        source="canonical_table",
        expected_primary_rows=475,
        expected_evidence_source="continuous_source_sequence",
        expected_evidence_confidence=0.80,
    )

    assert _physical_logical_row_mismatch_warning(457, style_meta, reconstruction) == (
        "BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical=457:canonical=475"
    )


def test_row_plane_source_census_does_not_suppress_sparse_physical_row_mismatch() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="native_wide_table",
        canonical_extracted=130,
        expected_primary_rows=130,
    )
    reconstruction = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=130,
        expected_evidence_source="native_page_signed_ledger_census",
        expected_evidence_confidence=0.99,
    )

    assert _physical_logical_row_mismatch_warning(1, style_meta, reconstruction) == (
        "BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical=1:canonical=130"
    )


def test_page_footer_count_supersedes_sparse_physical_row_estimate() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="native_wide_table",
        canonical_extracted=130,
        expected_primary_rows=130,
    )
    reconstruction = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=130,
        expected_evidence_source="page_footer",
        expected_evidence_confidence=0.90,
    )

    assert _physical_logical_row_mismatch_warning(1, style_meta, reconstruction) == ""


@pytest.mark.parametrize(
    "evidence",
    [
        19,
        RowCountEvidence(19, "unknown", 0.99),
        RowCountEvidence(19, "candidate_rows", 0.99),
        RowCountEvidence(19, "page_transaction_anchors", 0.99),
        RowCountEvidence(19, "physical_rows", 0.99),
        RowCountEvidence(19, "positioned_date_anchors", 0.99),
        RowCountEvidence(19, "positioned_record_blocks", 0.99),
    ],
)
def test_nonissuer_count_cannot_populate_source_reported_identity_total(evidence: object) -> None:
    identity_fields: dict[str, dict] = {}

    _apply_source_reported_transaction_count(identity_fields, evidence)

    assert "total_transactions" not in identity_fields
    assert not is_authoritative_issuer_row_count(evidence)


@pytest.mark.parametrize(
    "source,confidence",
    [
        ("split_footer", 0.98),
        ("header_total", 0.94),
        ("statement_header_totals", 0.97),
        ("cumulative_footer_total", 0.99),
        ("page_footer", 0.90),
    ],
)
def test_issuer_count_populates_source_reported_identity_total(source: str, confidence: float) -> None:
    identity_fields: dict[str, dict] = {}
    evidence = RowCountEvidence(19, source, confidence)

    _apply_source_reported_transaction_count(identity_fields, evidence)

    assert identity_fields["total_transactions"]["normalized_value"] == "19"
    assert identity_fields["total_transactions"]["source"] == f"row_count_evidence.{source}"
    assert is_authoritative_issuer_row_count(evidence)


def _identity_only_parse_result(*, key_values: list[object] | None = None, text_atoms: list[dict] | None = None):
    return SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=1,
                key_values=list(key_values or []),
                tables=[],
                texts=[],
                width=600,
                height=850,
            )
        ],
        logical_tables=[],
        full_text="",
        raw_text="",
        parser_info=None,
        evidence_plane=(
            SimpleNamespace(evidence={"text_atoms": list(text_atoms or []), "vector_atoms": []})
            if text_atoms is not None
            else None
        ),
        entities=SimpleNamespace(document_type="bank_statement", domain_specific={}, metadata={}),
        provenance=None,
    )


def test_pipeline_drops_kv_total_without_authoritative_issuer_evidence() -> None:
    parse_result = _identity_only_parse_result(
        key_values=[SimpleNamespace(key="\u603b\u7b14\u6570", value="2", bbox=None, evidence_ids=[])]
    )

    result = run_bank_statement_extract(parse_result, "", BankStatementCommunityPlugin())

    assert "total_transactions" not in result.identity_fields


def test_pipeline_drops_right_neighbor_atom_total_without_authoritative_issuer_evidence() -> None:
    parse_result = _identity_only_parse_result(
        text_atoms=[
            {
                "id": "count_label",
                "page_id": "page:0001",
                "text": "\u6c47\u603b\u4ea4\u6613\u7b14\u6570",
                "bbox": [10.0, 220.0, 80.0, 228.0],
            },
            {
                "id": "count_value",
                "page_id": "page:0001",
                "text": "2\u7b14",
                "bbox": [110.0, 225.0, 140.0, 233.0],
            },
        ]
    )

    result = run_bank_statement_extract(parse_result, "", BankStatementCommunityPlugin())

    assert "total_transactions" not in result.identity_fields


def test_community_projection_drops_preexisting_total_without_authoritative_evidence(monkeypatch) -> None:
    from docmirror.plugins.bank_statement.canonical import StyleMeta
    from docmirror.plugins.bank_statement.extract_pipeline import BankExtractResult
    from docmirror.plugins.bank_statement.extraction_dispatch import BankExtractionRoute

    plugin = BankStatementCommunityPlugin()
    parse_result = _identity_only_parse_result()
    synthetic_result = BankExtractResult(
        ctx=SimpleNamespace(full_text="", reconstruction=ReconstructionMeta(source="none")),
        detection=SimpleNamespace(),
        records=[],
        identity_fields={"total_transactions": {"normalized_value": "2"}},
        style_meta=StyleMeta(style_id="grid_standard", style_confidence=1.0),
        warnings=[],
        extraction_route=BankExtractionRoute.DIGITAL,
    )
    monkeypatch.setattr(community_module, "run_bank_statement_extract", lambda *_args: synthetic_result)

    projection = plugin.derive(parse_result, "")

    assert "total_transactions" not in projection.domain_facts
    assert "source_reported_transaction_count" not in projection.domain_facts


def test_community_projection_reapplies_authoritative_issuer_total(monkeypatch) -> None:
    from docmirror.plugins.bank_statement.canonical import StyleMeta
    from docmirror.plugins.bank_statement.extract_pipeline import BankExtractResult
    from docmirror.plugins.bank_statement.extraction_dispatch import BankExtractionRoute

    plugin = BankStatementCommunityPlugin()
    parse_result = _identity_only_parse_result()
    synthetic_result = BankExtractResult(
        ctx=SimpleNamespace(full_text="", reconstruction=ReconstructionMeta(source="none")),
        detection=SimpleNamespace(),
        records=[],
        identity_fields={"total_transactions": {"normalized_value": "2"}},
        style_meta=StyleMeta(style_id="grid_standard", style_confidence=1.0),
        warnings=[],
        extraction_route=BankExtractionRoute.DIGITAL,
    )
    issuer_evidence = RowCountEvidence(19, "header_total", 0.94, page=1, evidence_ids=("count:1",))
    monkeypatch.setattr(community_module, "run_bank_statement_extract", lambda *_args: synthetic_result)
    monkeypatch.setattr(community_module, "resolve_row_count_evidence", lambda *_args, **_kwargs: issuer_evidence)

    projection = plugin.derive(parse_result, "")

    assert projection.domain_facts["total_transactions"] == "19"
    assert projection.domain_facts["source_reported_transaction_count"] == 19


def test_incomplete_evidence_recovery_keeps_physical_row_mismatch_warning() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="canonical_evidence_table",
        canonical_extracted=198,
        expected_primary_rows=199,
    )

    reconstruction = ReconstructionMeta(
        source="canonical_evidence_table",
        expected_primary_rows=199,
        expected_evidence_source="header_total",
        expected_evidence_confidence=0.94,
    )

    assert _physical_logical_row_mismatch_warning(4, style_meta, reconstruction) == (
        "BANK_PHYSICAL_LOGICAL_ROW_MISMATCH:physical=4:canonical=198"
    )


def test_projection_row_accounting_checks_the_emitted_dataset(monkeypatch) -> None:
    plugin = BankStatementCommunityPlugin()
    original_sanitize = community_module._sanitize_bank_records

    def drop_last_record(records):
        return original_sanitize(records)[:-1]

    monkeypatch.setattr(community_module, "_sanitize_bank_records", drop_last_record)
    text = _synthetic_boc_text()
    parse_result = SimpleNamespace(
        entities=SimpleNamespace(document_type="bank_statement", domain_specific={}),
        pages=[],
        logical_tables=[],
        full_text=text,
        raw_text=text,
        parser_info=None,
        evidence_plane=None,
    )

    projection = plugin.derive(parse_result, text)

    assert any(warning.startswith("BANK_CANONICAL_EMITTED_ROW_MISMATCH:") for warning in projection.warnings)
    assert projection.confidence == 0.35


def test_positioned_record_blocks_do_not_certify_public_completeness() -> None:
    meta = build_style_meta(
        SimpleNamespace(
            primary_style="grid_standard",
            confidence=1.0,
            parser_chain=[],
            institution_hint=None,
            secondary_styles=[],
            institution_authority="",
        ),
        reconstruction=ReconstructionMeta(source="positioned_record_block", expected_primary_rows=19),
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"


def test_source_reported_count_overrides_positioned_candidate_count() -> None:
    meta = build_style_meta(
        SimpleNamespace(
            primary_style="grid_standard",
            confidence=1.0,
            parser_chain=[],
            institution_hint=None,
            secondary_styles=[],
            institution_authority="",
        ),
        reconstruction=ReconstructionMeta(source="positioned_record_block", expected_primary_rows=19),
        source_reported_count=RowCountEvidence(21, "split_footer", 0.98),
    )

    assert meta.expected_primary_rows == 21


def test_native_wide_table_does_not_publish_positioned_date_count() -> None:
    records = [
        {
            "normalized": {
                "date": f"2025-01-{index:02d}",
                "amount": 1.0,
                "direction": "income",
            }
        }
        for index in range(1, 9)
    ]
    meta = build_style_meta(
        SimpleNamespace(
            primary_style="grid_standard",
            confidence=1.0,
            parser_chain=[],
            institution_hint=None,
            secondary_styles=[],
            institution_authority="",
        ),
        reconstruction=ReconstructionMeta(
            source="native_wide_table",
            expected_primary_rows=8,
            expected_evidence_source="positioned_date_anchors",
            expected_evidence_confidence=0.95,
        ),
        parse_result=SimpleNamespace(),
        records=records,
        record_count=8,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.canonical_extracted == 8
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"
