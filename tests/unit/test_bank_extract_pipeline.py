# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared bank statement extract pipeline."""

from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.bank_statement import community_plugin as community_module
from docmirror.plugins.bank_statement.canonical import build_style_meta, records_from_raw_transactions
from docmirror.plugins.bank_statement.canonical_quality import audit_row_accounting
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.extract_pipeline import (
    _physical_logical_row_mismatch_warning,
    enrich_identity_fields,
    run_bank_statement_extract,
)
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from tests.unit.test_pipe_text_table_builder import _synthetic_boc_text


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

    assert _physical_logical_row_mismatch_warning(4, style_meta) == ""


def test_incomplete_evidence_recovery_keeps_physical_row_mismatch_warning() -> None:
    style_meta = SimpleNamespace(
        reconstruction_source="canonical_evidence_table",
        canonical_extracted=198,
        expected_primary_rows=199,
    )

    assert _physical_logical_row_mismatch_warning(4, style_meta) == (
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


def test_positioned_record_blocks_override_collapsed_canonical_row_estimate() -> None:
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

    assert meta.expected_primary_rows == 19


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
        source_reported_count=21,
    )

    assert meta.expected_primary_rows == 21


def test_native_wide_table_uses_independent_positioned_date_count(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.canonical_quality as canonical_quality

    monkeypatch.setattr(canonical_quality, "canonical_expected_from_parse_result", lambda _parse_result: 1)
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

    assert meta.expected_primary_rows == 8
    assert meta.canonical_extracted == 8
    assert meta.coverage_ratio == 1.0
