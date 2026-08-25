# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin coverage denominator uses Mirror LTQG SSOT."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    LogicalTable,
    PageContent,
    ParseResult,
    ParserInfo,
    RowType,
    TableBlock,
    TableRow,
)
from docmirror.plugins.bank_statement.canonical import build_style_meta
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import SCANNED_POLICY
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import _expected_rows
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence


def test_bank_plugin_production_has_no_private_fixture_identity_switches() -> None:
    """Production rules may describe source formats, never private fixture identity."""
    plugin_root = Path(__file__).parents[2] / "docmirror" / "plugins" / "bank_statement"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(plugin_root.rglob("*.py"))
    )
    lowered = source.casefold()

    assert "fixtures-private" not in lowered
    assert "bank_cashflow" not in lowered
    assert re.search(r"case[_-]?\d+", lowered) is None
    assert re.search(r"\b[0-9a-f]{32,64}\b", lowered) is None
    # Account, card, statement, and transaction identifiers are document facts.
    # Format regexes are allowed, but literal identities are not.
    assert re.search(r"(?<!\\d)(?<!\{)\b\d{16,31}\b(?!\})", source) is None


def _detection() -> StyleDetectionResult:
    return StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.9,
        parser_chain=["grid_standard"],
        institution_hint="ccb",
    )


def _parse_result_with_ltqg(expected: int = 47) -> ParseResult:
    headers = ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"]
    rows = [
        TableRow(
            cells=[CellValue(text="2024-01-01"), CellValue(text="x"), CellValue(text="1.00")],
            row_type=RowType.DATA,
        )
        for _ in range(expected)
    ]
    return ParseResult(
        logical_tables=[
            LogicalTable(
                headers=headers,
                rows=rows,
                row_count=expected,
                quality_passed=True,
                data_row_estimate=expected,
            )
        ],
        parser_info=ParserInfo(
            structure={
                "ltqg_enabled": True,
                "ltqg_expected_data_rows": expected,
            }
        ),
    )


def test_build_style_meta_keeps_mirror_expected_rows_internal():
    pr = _parse_result_with_ltqg(47)
    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=127),
        record_count=40,
        parse_result=pr,
    )
    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"


def test_build_style_meta_prefers_independent_source_count_over_weak_mirror_estimate():
    pr = _parse_result_with_ltqg(12)
    records = [
        {
            "normalized": {
                "date": "2025-07-01",
                "direction": "income",
                "amount": 1.0,
                "balance": 1.0,
            }
        }
        for _ in range(11)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=12),
        record_count=11,
        parse_result=pr,
        records=records,
        source_reported_count=RowCountEvidence(11, "header_total", 0.94),
    )

    assert meta.expected_primary_rows == 11
    assert meta.canonical_expected == 11
    assert meta.coverage_ratio == 1.0
    assert meta.extract_status == "success"


def test_build_style_meta_does_not_treat_candidate_sequence_as_document_count():
    pr = _parse_result_with_ltqg(20)
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(19)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="native_wide_table",
            expected_primary_rows=19,
            expected_evidence_source="continuous_source_sequence",
            expected_evidence_confidence=0.80,
        ),
        record_count=19,
        parse_result=pr,
        records=records,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.extract_status == "degraded"


def test_build_style_meta_does_not_treat_high_confidence_row_plane_census_as_document_count():
    pr = _parse_result_with_ltqg(20)
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(19)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="native_wide_table",
            expected_primary_rows=19,
            expected_evidence_source="native_page_datetime_census",
            expected_evidence_confidence=0.99,
        ),
        record_count=19,
        parse_result=pr,
        records=records,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.extract_status == "degraded"


def test_build_style_meta_rejects_row_plane_signal_misrouted_as_source_reported_count():
    pr = _parse_result_with_ltqg(20)
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(19)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(source="native_wide_table", expected_primary_rows=19),
        record_count=19,
        parse_result=pr,
        records=records,
        source_reported_count=RowCountEvidence(19, "native_page_datetime_census", 0.99),
    )

    assert meta.expected_primary_rows == 0
    assert meta.extract_status == "degraded"


def test_build_style_meta_rejects_unproven_scalar_count() -> None:
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(7)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=7),
        record_count=7,
        records=records,
        source_reported_count=7,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.extract_status == "degraded"


def test_row_plane_signal_does_not_gain_authority_from_recovery_source_name():
    pr = _parse_result_with_ltqg(20)
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(19)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="canonical_evidence_table",
            expected_primary_rows=19,
            expected_evidence_source="native_page_datetime_census",
            expected_evidence_confidence=0.99,
        ),
        record_count=19,
        parse_result=pr,
        records=records,
    )

    assert meta.expected_primary_rows == 0


def test_build_style_meta_does_not_replace_conflicting_issuer_total_with_short_sequence():
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(12)
    ]
    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="native_wide_table",
            expected_primary_rows=12,
            expected_evidence_source="continuous_source_sequence",
            expected_evidence_confidence=0.99,
        ),
        record_count=12,
        records=records,
        source_reported_count=RowCountEvidence(475, "split_footer", 0.98),
    )

    assert meta.expected_primary_rows == 475
    assert meta.extract_status in {"low_coverage", "degraded"}
    assert meta.coverage_ratio == 12 / 475


def test_build_style_meta_does_not_publish_heuristic_anchor_count():
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(549)
    ]
    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="native_wide_table",
            expected_primary_rows=549,
            expected_evidence_source="continuous_source_sequence",
            expected_evidence_confidence=0.80,
        ),
        record_count=549,
        records=records,
        source_reported_count=RowCountEvidence(475, "page_transaction_anchors", 0.93),
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"


def test_build_style_meta_does_not_subtract_stitches_from_issuer_transaction_total():
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(13)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="canonical_physical_tables",
            expected_primary_rows=13,
            stitched_continuation_rows=2,
            expected_evidence_source="statement_header_totals",
            expected_evidence_confidence=0.97,
        ),
        record_count=13,
        records=records,
        source_reported_count=RowCountEvidence(13, "statement_header_totals", 0.97),
    )

    assert meta.expected_primary_rows == 13
    assert meta.canonical_expected == 13
    assert meta.extract_status == "success"


def test_build_style_meta_still_removes_stitched_fragments_from_unproven_row_estimate():
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}}
        for _ in range(11)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="canonical_physical_tables",
            expected_primary_rows=13,
            stitched_continuation_rows=2,
            expected_evidence_source="physical_rows",
            expected_evidence_confidence=0.55,
        ),
        record_count=11,
        records=records,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.extract_status == "degraded"


def test_build_style_meta_does_not_publish_complete_candidate_over_sparse_mirror_estimate():
    pr = _parse_result_with_ltqg(4)

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(source="canonical_evidence_table", expected_primary_rows=199),
        record_count=199,
        parse_result=pr,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"


def test_matching_physical_prefix_does_not_certify_document_completeness() -> None:
    physical_rows = [
        TableRow(
            cells=[
                CellValue(text=f"2025-01-0{index}"),
                CellValue(text=f"{index}.00"),
                CellValue(text=f"{10 + index}.00"),
            ],
            row_type=RowType.DATA,
        )
        for index in (1, 2)
    ]
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        headers=["交易日期", "交易金额", "余额"],
                        rows=physical_rows,
                    )
                ],
            )
        ]
    )
    records = [
        {"normalized": {"date": f"2025-01-0{index}", "direction": "income", "amount": float(index)}}
        for index in (1, 2)
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source="canonical_physical_tables",
            expected_primary_rows=2,
            expected_evidence_source="physical_rows",
            expected_evidence_confidence=0.99,
        ),
        record_count=2,
        records=records,
        parse_result=parse_result,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"


@pytest.mark.parametrize(
    ("reconstruction_source", "evidence_source"),
    [
        ("canonical_physical_tables", "physical_rows"),
        ("native_wide_table", "positioned_date_anchors"),
        ("positioned_record_block", "positioned_record_blocks"),
        ("canonical_evidence_table", "page_transaction_anchors"),
    ],
)
def test_row_plane_reconstruction_never_certifies_public_completeness(
    reconstruction_source: str,
    evidence_source: str,
) -> None:
    records = [
        {"normalized": {"date": "2025-01-01", "direction": "income", "amount": 1.0}},
        {"normalized": {"date": "2025-01-02", "direction": "expense", "amount": 2.0}},
    ]

    meta = build_style_meta(
        _detection(),
        reconstruction=ReconstructionMeta(
            source=reconstruction_source,
            expected_primary_rows=2,
            expected_evidence_source=evidence_source,
            expected_evidence_confidence=0.99,
        ),
        record_count=2,
        records=records,
    )

    assert meta.expected_primary_rows == 0
    assert meta.canonical_expected == 0
    assert meta.coverage_ratio == 0.0
    assert meta.extract_status == "degraded"


def test_style_registry_expected_rows_from_parse_result():
    pr = _parse_result_with_ltqg(47)
    ctx = StyleContext(
        tables=[[["交易日期", "摘要"], ["2024-01-01", "x"]]],
        full_text="",
        institution="ccb",
        page_count=1,
        parse_result=pr,
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=127),
    )
    expected = _expected_rows(ctx)
    assert expected == 47


def test_style_registry_expected_rows_prefers_cached_ocr_recovery_over_weak_mirror_count():
    pr = _parse_result_with_ltqg(2)
    pr.entities.domain_specific["_bank_ocr_implicit_recovery"] = {
        "status": "ready",
        "row_count": 128,
        "tables": [[["交易日期", "收/支"], *[["2024-01-01", "收入"] for _ in range(128)]]],
    }
    ctx = StyleContext(
        tables=[[["交易日期", "摘要"], ["2024-01-01", "x"]]],
        full_text="",
        institution="ccb",
        page_count=1,
        parse_result=pr,
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=2),
        extraction_route=SCANNED_POLICY.route,
        extraction_policy=SCANNED_POLICY,
    )

    assert _expected_rows(ctx) == 128
