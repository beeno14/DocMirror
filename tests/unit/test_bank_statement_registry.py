# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.

"""Bank statement template registry and canonical style integration."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

import docmirror.plugins.bank_statement.style_registry as style_registry
from docmirror.models.sealed import seal_parse_result
from docmirror.output.mirror_projector import project_mirror
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import (
    DIGITAL_POLICY,
    SCANNED_POLICY,
    BankExtractionPolicy,
    BankExtractionRoute,
)
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import (
    BankStyleParserRegistry,
    BankTableCandidate,
    _candidate_expected_rows,
    _candidate_sequence_continuity,
    _select_candidate,
)
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence, _recover_cross_page_wide_tables


def _candidate_for_authority_test(
    candidate_id: str,
    rows: int,
    evidence: RowCountEvidence,
) -> BankTableCandidate:
    records = [{"normalized": {"date": "2025-01-01", "amount": 1.0, "direction": "income"}}] * rows
    return BankTableCandidate(
        candidate_id=candidate_id,
        records=records,
        source="native_wide_table",
        canonical_rows=rows,
        directional_rows=rows,
        source_page_rows=rows,
        expected_rows=evidence,
        balance_chain_score=0.5,
        field_completeness=1.0,
        score=1.0,
        canonical_coverage=1.0,
        source_page_coverage=1.0,
        extraction_confidence=0.9,
        source_column_width=5.0,
        sequence_continuity=1.0,
    )


def _complete_primary_candidate(rows: int = 2) -> BankTableCandidate:
    raw_records = [
        {
            "date": f"2025-01-{index + 1:02d}",
            "amount": f"{index + 1}.00",
            "direction": "income",
            "summary": f"row {index + 1}",
            "_source": {
                "source_page": 1,
                "page_range": [1, 1],
                "table_id": "table:1",
                "source_row_index": index,
                "source_cell_refs": [
                    {"page": 1, "table_id": "table:1", "row": index, "raw_row": index + 1, "col": col}
                    for col in range(4)
                ],
            },
        }
        for index in range(rows)
    ]

    def normalize(raw: dict[str, object]) -> dict[str, object]:
        return {
            "date": raw["date"],
            "amount": float(str(raw["amount"])),
            "direction": raw["direction"],
            "summary": raw["summary"],
        }

    return BankTableCandidate(
        candidate_id="parser:grid_standard",
        records=raw_records,
        source="canonical_table",
        canonical_rows=rows,
        directional_rows=rows,
        source_page_rows=rows,
        expected_rows=RowCountEvidence(rows, "header_total", 0.95),
        balance_chain_score=1.0,
        field_completeness=1.0,
        score=1.0,
        normalize_fn=normalize,
        canonical_coverage=1.0,
        source_page_coverage=1.0,
        extraction_confidence=0.9,
        source_column_width=4.0,
        sequence_continuity=1.0,
    )


def _single_scope_parse_result(
    headers: list[str] | None = None,
    rows: list[list[str]] | None = None,
) -> SimpleNamespace:
    headers = headers or ["date", "amount", "direction", "summary"]
    rows = rows or [
        ["2025-01-01", "1.00", "income", "row 1"],
        ["2025-01-02", "2.00", "income", "row 2"],
    ]
    raw_rows = [headers, *rows]
    table_atoms: list[dict[str, object]] = []
    cell_evidence_ids: list[list[list[str]]] = []
    for row_index, row in enumerate(raw_rows):
        evidence_row: list[list[str]] = []
        for col_index, value in enumerate(row):
            if not value:
                evidence_row.append([])
                continue
            atom_id = f"table:r{row_index}:c{col_index}"
            table_atoms.append(
                {
                    "id": atom_id,
                    "page_id": "page:0001",
                    "text": value,
                    "bbox": [
                        col_index * 100.0 + 5.0,
                        row_index * 20.0 + 102.0,
                        (col_index + 1) * 100.0 - 5.0,
                        row_index * 20.0 + 118.0,
                    ],
                    "source_kind": "pdf_native",
                }
            )
            evidence_row.append([atom_id])
        cell_evidence_ids.append(evidence_row)
    table = SimpleNamespace(
        table_id="table:1",
        headers=headers,
        rows=[SimpleNamespace(cells=[SimpleNamespace(text=value) for value in row]) for row in rows],
        metadata={"raw_rows": raw_rows},
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        tables=[table],
        texts=[],
        key_values=[],
    )
    return SimpleNamespace(
        pages=[page],
        logical_tables=[SimpleNamespace(quality_passed=True)],
        evidence_plane=SimpleNamespace(
            evidence={
                "text_atoms": [
                    {
                        "id": "scope:account",
                        "page_id": "page:0001",
                        "text": "Account Number: 1234567890",
                        "bbox": [10.0, 10.0, 200.0, 20.0],
                        "source_kind": "parse_result_text",
                    },
                    *table_atoms,
                ],
                "indexes": {
                    "table_candidates": [
                        {
                            "candidate_id": "table:1",
                            "page_id": "page:0001",
                            "page_number": 1,
                            "table_index": 0,
                            "bbox": [0.0, 100.0, len(headers) * 100.0, len(raw_rows) * 20.0 + 100.0],
                            "rows": raw_rows,
                            "geometry": {"cell_evidence_ids": cell_evidence_ids},
                        }
                    ]
                },
            }
        ),
        parser_info=None,
        full_text="",
        raw_text="",
    )


@pytest.mark.parametrize(
    "source",
    [
        "complete_page_local_sequences",
        "ccb_primary_source_sequence",
        "cmb_primary_source_rows",
        "native_page_datetime_census",
        "native_page_signed_ledger_census",
        "ocr_page_ordinal_census",
    ],
)
def test_short_row_plane_signal_cannot_beat_fuller_candidate(source: str) -> None:
    short = _candidate_for_authority_test("recovery:short", 4, RowCountEvidence(4, source, 0.99))
    fuller = _candidate_for_authority_test("parser:full", 5, RowCountEvidence(5, "candidate_rows", 0.55))

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is fuller


@pytest.mark.parametrize("fuller_route", ["ocr_implicit_table", "positioned_record_block"])
def test_fuller_candidate_guard_is_route_agnostic(fuller_route: str) -> None:
    short = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            2,
            RowCountEvidence(2, "positioned_date_anchors", 0.80),
        ),
        sequence_continuity=1.0,
    )
    fuller = replace(
        _candidate_for_authority_test(
            fuller_route,
            3,
            RowCountEvidence(3, "candidate_rows", 0.55),
        ),
        sequence_continuity=0.95,
    )

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is fuller


def test_fuller_candidate_guard_preserves_materially_richer_shorter_candidate() -> None:
    short = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            2,
            RowCountEvidence(2, "positioned_date_anchors", 0.80),
        ),
        sequence_continuity=1.0,
        source_column_width=8.0,
    )
    fuller = replace(
        _candidate_for_authority_test(
            "ocr_implicit_table",
            3,
            RowCountEvidence(3, "candidate_rows", 0.55),
        ),
        field_completeness=0.70,
        sequence_continuity=0.95,
        source_column_width=6.0,
    )

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is short


def _candidate_records_on_pages(rows: int, pages: list[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(rows):
        page = pages[index % len(pages)]
        records.append(
            {
                "normalized": {"date": "2025-01-01", "amount": 1.0, "direction": "income"},
                "_source": {
                    "source_page": page,
                    "page_range": [page, page],
                    "bbox": [10.0, float(index * 10), 100.0, float(index * 10 + 8)],
                },
            }
        )
    return records


def test_fuller_candidate_can_use_strict_source_page_superset_over_local_balance_score() -> None:
    short = replace(
        _candidate_for_authority_test(
            "native_wide_table:0",
            4,
            RowCountEvidence(4, "native_wide_rows", 0.70),
        ),
        records=_candidate_records_on_pages(4, [2, 3]),
        balance_chain_score=0.98,
        sequence_continuity=1.0,
        score=0.98,
    )
    fuller = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            10,
            RowCountEvidence(10, "positioned_date_anchors", 0.80),
        ),
        records=_candidate_records_on_pages(10, [1, 2, 3, 4, 5]),
        balance_chain_score=0.50,
        sequence_continuity=0.0,
        score=0.88,
    )

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is fuller


def test_complete_three_page_two_scope_candidate_beats_nonempty_partial_plane() -> None:
    def transaction(page: int, scope: str, sequence: int) -> dict[str, object]:
        return {
            "normalized": {
                "date": f"2025-0{page}-01",
                "amount": float(sequence),
                "direction": "income",
                "sequence_no": str(sequence),
                "statement_header_id": scope,
            },
            "_source": {
                "source_page": page,
                "page_range": [page, page],
                "table_id": f"physical:page:{page}",
                "source_row_index": sequence,
                "bbox": [10.0, sequence * 20.0, 100.0, sequence * 20.0 + 8.0],
            },
        }

    partial_records = [
        transaction(1, "statement_header:scope-a", 1),
        transaction(2, "statement_header:scope-a", 2),
    ]
    complete_records = [
        *partial_records,
        transaction(3, "statement_header:scope-b", 3),
        transaction(3, "statement_header:scope-b", 4),
    ]
    partial = replace(
        _candidate_for_authority_test(
            "native_wide_table:partial",
            2,
            RowCountEvidence(2, "native_wide_rows", 0.70),
        ),
        records=partial_records,
        balance_chain_score=0.98,
        sequence_continuity=1.0,
        score=0.98,
    )
    complete = replace(
        _candidate_for_authority_test(
            "evidence_atom:complete",
            4,
            RowCountEvidence(4, "positioned_date_anchors", 0.80),
        ),
        records=complete_records,
        balance_chain_score=0.50,
        sequence_continuity=0.95,
        score=0.88,
    )

    selected, diagnostics = _select_candidate([partial, complete])

    assert selected is complete
    assert diagnostics["selected_candidate"] == "evidence_atom:complete"
    assert diagnostics["candidate_counts"] == {
        "native_wide_table:partial": 2,
        "evidence_atom:complete": 4,
    }
    assert [
        (
            record["_source"]["source_page"],
            record["normalized"]["statement_header_id"],
            record["normalized"]["sequence_no"],
        )
        for record in selected.records
    ] == [
        (1, "statement_header:scope-a", "1"),
        (2, "statement_header:scope-a", "2"),
        (3, "statement_header:scope-b", "3"),
        (3, "statement_header:scope-b", "4"),
    ]


def test_fuller_candidate_with_same_source_pages_still_needs_semantic_parity() -> None:
    short = replace(
        _candidate_for_authority_test(
            "native_wide_table:0",
            4,
            RowCountEvidence(4, "native_wide_rows", 0.70),
        ),
        records=_candidate_records_on_pages(4, [1, 2]),
        balance_chain_score=0.98,
        sequence_continuity=1.0,
        score=0.98,
    )
    noisier = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            8,
            RowCountEvidence(8, "positioned_date_anchors", 0.80),
        ),
        records=_candidate_records_on_pages(8, [1, 2]),
        balance_chain_score=0.50,
        sequence_continuity=0.0,
        score=0.88,
    )

    selected, _diagnostics = _select_candidate([short, noisier])

    assert selected is short


def _sequenced_source_row(sequence: int, page: int, y: float) -> tuple[dict[str, str], dict[str, object]]:
    return (
        {"sequence_no": str(sequence)},
        {
            "_source": {
                "source_page": page,
                "page_range": [page, page],
                "bbox": [10.0, y, 100.0, y + 8.0],
            }
        },
    )


def test_sequence_continuity_accepts_only_forward_source_order_resets() -> None:
    rows = [
        _sequenced_source_row(1, 1, 100.0),
        _sequenced_source_row(2, 1, 120.0),
        _sequenced_source_row(1, 2, 80.0),
        _sequenced_source_row(2, 2, 100.0),
        _sequenced_source_row(3, 2, 120.0),
    ]

    assert _candidate_sequence_continuity(
        [normalized for normalized, _transaction in rows],
        [transaction for _normalized, transaction in rows],
    ) == 1.0


def test_sequence_continuity_rejects_a_duplicate_plane_page_rewind() -> None:
    rows = [
        _sequenced_source_row(1, 1, 100.0),
        _sequenced_source_row(2, 2, 100.0),
        _sequenced_source_row(1, 1, 101.0),
        _sequenced_source_row(2, 2, 101.0),
    ]

    score = _candidate_sequence_continuity(
        [normalized for normalized, _transaction in rows],
        [transaction for _normalized, transaction in rows],
    )

    assert score == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "source",
    [
        "complete_page_local_sequences",
        "ccb_primary_source_sequence",
        "cmb_primary_source_rows",
        "native_page_datetime_census",
        "native_page_signed_ledger_census",
        "ocr_page_ordinal_census",
    ],
)
def test_row_plane_signal_is_clamped_below_public_count_authority(source: str) -> None:
    evidence = _candidate_expected_rows(RowCountEvidence(4, source, 0.99))

    assert evidence == RowCountEvidence(4, source, 0.80)


def test_candidate_count_is_not_replaced_by_separate_row_plane_signal() -> None:
    evidence = _candidate_expected_rows(
        RowCountEvidence(4, "native_page_datetime_census", 0.99),
        count=5,
        source="native_wide_rows",
        confidence=0.70,
    )

    assert evidence == RowCountEvidence(5, "native_wide_rows", 0.70)


@pytest.mark.parametrize(
    "source",
    [
        "positioned_record_blocks",
        "physical_rows",
        "positioned_date_anchors",
        "page_transaction_anchors",
    ],
)
def test_candidate_fallback_row_plane_source_is_clamped_below_authority(source: str) -> None:
    evidence = _candidate_expected_rows(
        RowCountEvidence(0, "candidate_rows", 0.55),
        count=7,
        source=source,
        confidence=0.99,
    )

    assert evidence == RowCountEvidence(7, source, 0.80)


@pytest.mark.parametrize(
    "source,confidence",
    [
        ("split_footer", 0.98),
        ("header_total", 0.94),
        ("cumulative_footer_total", 0.99),
        ("page_footer", 0.90),
    ],
)
def test_issuer_count_remains_authoritative_for_candidate_selection(source: str, confidence: float) -> None:
    complete = _candidate_for_authority_test("recovery:complete", 4, RowCountEvidence(4, source, confidence))
    extra = _candidate_for_authority_test("parser:extra", 5, RowCountEvidence(4, source, confidence))

    selected, _diagnostics = _select_candidate([complete, extra])

    assert selected is complete


def test_competing_complete_planes_select_one_without_deduping_legitimate_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = ["交易日期", "交易金额", "余额", "交易流水号"]
    repeated_row = ["2025-01-01", "+1.00", "101.00", "same-reference"]
    first_plane = [headers, repeated_row.copy(), repeated_row.copy()]
    second_plane = [headers, repeated_row.copy(), repeated_row.copy()]
    plane_sources = {
        id(first_plane): ("native:plane:first", 0.0),
        id(second_plane): ("native:plane:second", 0.4),
    }

    monkeypatch.setattr(
        style_registry,
        "recover_wide_bank_tables",
        lambda *_args: [first_plane, second_plane],
    )
    monkeypatch.setattr(
        style_registry,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence(2, "split_footer", 0.98),
    )

    def run_parser(_parser_id, parser_ctx, _plugin):
        [table] = parser_ctx.tables
        table_id, drift = plane_sources[id(table)]
        records = []
        for row_index, row in enumerate(table[1:]):
            y0 = 100.0 + row_index * 20.0 + drift
            records.append(
                {
                    "交易日期": row[0],
                    "交易金额": row[1],
                    "余额": row[2],
                    "交易流水号": row[3],
                    "_source": {
                        "source_page": 1,
                        "page_range": [1, 1],
                        "table_id": table_id,
                        "source_row_index": row_index,
                        "bbox": [10.0, y0, 200.0, y0 + 10.0],
                    },
                }
            )

        return records, lambda raw: {
            "date": raw["交易日期"],
            "amount": abs(float(raw["交易金额"])),
            "direction": "income",
            "balance": float(raw["余额"]),
            "reference": raw["交易流水号"],
        }

    monkeypatch.setattr(style_registry, "_run_parser", run_parser)
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.DIGITAL,
        allowed_parser_ids=frozenset(),
        allow_native_wide_tables=True,
    )
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=None,
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance", "reference"],
        _normalize=lambda raw: raw,
    )

    candidates = style_registry._collect_table_candidates(
        StyleDetectionResult(primary_style="grid_standard", parser_chain=[]),
        ctx,
        plugin,
    )
    native = [candidate for candidate in candidates if candidate.candidate_id.startswith("native_wide_table")]
    selected, diagnostics = _select_candidate(native)

    assert [candidate.candidate_id for candidate in native] == [
        "native_wide_table:0",
        "native_wide_table:1",
    ]
    assert selected is native[0]
    assert diagnostics["selected_candidate"] == "native_wide_table:0"
    assert selected.expected_rows == RowCountEvidence(2, "split_footer", 0.98)
    assert len(selected.records) == 2
    assert {record["_source"]["table_id"] for record in selected.records} == {"native:plane:first"}
    assert selected.records[0]["_source"]["bbox"] != selected.records[1]["_source"]["bbox"]
    assert selected.normalize_fn(selected.records[0]) == selected.normalize_fn(selected.records[1])


def test_primary_collector_invokes_only_first_allowed_detected_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.DIGITAL,
        allowed_parser_ids=frozenset({"signed_amount", "grid_standard"}),
        allow_semantic_text=True,
        allow_physical_tables=True,
        allow_positioned_records=True,
        allow_evidence_atoms=True,
        allow_native_wide_tables=True,
    )
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        prefer_context_tables=True,
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    detection = StyleDetectionResult(
        primary_style="signed_amount",
        confidence=0.95,
        parser_chain=["kv_identity", "compact_merged", "signed_amount", "grid_standard"],
    )
    calls: list[str] = []

    def run_parser(parser_id, _ctx, _plugin):
        calls.append(parser_id)
        candidate = _complete_primary_candidate(1)
        return candidate.records, candidate.normalize_fn

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a reconstruction provider ran during primary preflight")

    monkeypatch.setattr(style_registry, "_run_parser", run_parser)
    for name in (
        "_semantic_text_table_candidates",
        "collect_physical_tables_from_parse_result",
        "recover_positioned_record_block_bank_tables",
        "recover_evidence_atom_bank_tables",
        "recover_wide_bank_tables",
        "recover_ocr_implicit_ledger_tables",
    ):
        monkeypatch.setattr(style_registry, name, forbidden)

    candidates = style_registry._collect_primary_table_candidates(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert calls == ["signed_amount"]
    assert [candidate.candidate_id for candidate in candidates] == ["parser:signed_amount"]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            DIGITAL_POLICY,
            [
                "parser:signed_amount",
                "parser:grid_standard",
                "parser:compact_merged",
                "semantic_text",
                "physical_table",
                "positioned_record_block",
                "evidence_atom",
                "native_wide_table",
            ],
        ),
        (
            SCANNED_POLICY,
            [
                "parser:signed_amount",
                "parser:borderless_ocr",
                "parser:grid_standard",
                "evidence_atom",
                "ocr_implicit_table",
            ],
        ),
    ],
    ids=["digital", "scanned"],
)
def test_eligible_strategy_graph_has_deterministic_parser_and_provider_order(
    policy: BankExtractionPolicy,
    expected: list[str],
) -> None:
    detection = StyleDetectionResult(
        primary_style="signed_amount",
        confidence=0.95,
        parser_chain=["kv_identity", "signed_amount", "signed_amount", "borderless_ocr"],
    )
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        extraction_route=policy.route,
        extraction_policy=policy,
    )

    strategies = style_registry._eligible_strategy_ids(detection, ctx)

    assert strategies == expected
    assert len(strategies) == len(set(strategies))


def test_proven_primary_skips_every_unused_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _complete_primary_candidate()
    page_scope = """
    账户名称：测试企业 账号：1234567890123456 起止日期：2025-01-01 - 2025-12-31
    交易日期 交易金额 余额 对方账号 摘要
    交易总金额：3.00 借方累计金额：0.00 贷方累计金额：3.00 总笔数: 2
    """
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text=page_scope,
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    parser_calls: list[str] = []

    def run_parser(parser_id, _ctx, _plugin):
        parser_calls.append(parser_id)
        return candidate.records, candidate.normalize_fn

    monkeypatch.setattr(style_registry, "_run_parser", run_parser)
    monkeypatch.setattr(style_registry, "page_texts_from_parse_result", lambda _result: [(1, page_scope)])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the eager collector ran after completion was proven")

    monkeypatch.setattr(style_registry, "_collect_table_candidates", forbidden)

    registry = BankStyleParserRegistry()
    records, _identity = registry.run_parser_chain(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 2
    assert registry.last_selection_diagnostics["selected_candidate"] == "parser:grid_standard"
    assert registry.last_selection_diagnostics["deployment_mode"] == "lazy_primary"
    assert registry.last_selection_diagnostics["completion_state"] == "proven"
    assert registry.last_selection_diagnostics["attempted_strategies"] == ["parser:grid_standard"]
    assert registry.last_selection_diagnostics["skipped_strategies"]
    assert parser_calls == ["grid_standard"]


def test_primary_core_fields_cannot_hide_unconserved_source_business_columns() -> None:
    candidate = _complete_primary_candidate()
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary", "reference"],
                ["2025-01-01", "1.00", "income", "row 1", "ref-1"],
                ["2025-01-02", "2.00", "income", "row 2", "ref-2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_prove_complete_against_richer_sealed_physical_schema() -> None:
    candidate = _complete_primary_candidate()
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(
            headers=["date", "amount", "direction", "summary", "reference"],
            rows=[
                ["2025-01-01", "1.00", "income", "row 1", "ref-1"],
                ["2025-01-02", "2.00", "income", "row 2", "ref-2"],
            ],
        ),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_prove_complete_when_source_cell_value_changed() -> None:
    candidate = _complete_primary_candidate()
    candidate.records[1]["summary"] = "altered row"
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


@pytest.mark.parametrize("changed_field", ["text", "cleaned"])
def test_primary_cannot_prove_complete_when_physical_cell_plane_conflicts(
    changed_field: str,
) -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    cell = parse_result.pages[0].tables[0].rows[0].cells[3]
    setattr(cell, changed_field, "different sealed value")
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_richer_sealed_evidence_table() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    indexed = parse_result.evidence_plane.evidence["indexes"]["table_candidates"][0]
    indexed["rows"] = [
        [*row, value]
        for row, value in zip(indexed["rows"], ["reference", "ref-1", "ref-2"], strict=True)
    ]
    indexed["geometry"]["cell_evidence_ids"] = [
        [*row, []] for row in indexed["geometry"]["cell_evidence_ids"]
    ]
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_unowned_positioned_business_cell() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    parse_result.evidence_plane.evidence["text_atoms"].append(
        {
            "id": "extra:reference:1",
            "page_id": "page:0001",
            "text": "ref-1",
            "bbox": [405.0, 122.0, 455.0, 138.0],
            "source_kind": "pdf_native",
        }
    )
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_prove_complete_when_sealed_header_atoms_conflict() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    header_atom = next(
        atom
        for atom in parse_result.evidence_plane.evidence["text_atoms"]
        if atom["id"] == "table:r0:c3"
    )
    header_atom["text"] = "summary/reference"
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_second_positioned_ledger_without_time_column() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    for index, text in enumerate(
        ["日期时间", "交易金额", "20250101000000", "1.00", "20250102000000", "2.00"]
    ):
        parse_result.evidence_plane.evidence["text_atoms"].append(
            {
                "id": f"second-ledger:{index}",
                "page_id": "page:0001",
                "text": text,
                "bbox": [10.0 + index * 50.0, 200.0, 50.0 + index * 50.0, 215.0],
                "source_kind": "pdf_native",
            }
        )
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_outside_band_transaction_role_fact() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    parse_result.evidence_plane.evidence["text_atoms"].append(
        {
            "id": "wrapped:counterparty-account",
            "page_id": "page:0001",
            "text": "对方账号: 622200001111",
            "bbox": [10.0, 200.0, 180.0, 215.0],
            "source_kind": "pdf_native",
        }
    )
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_richer_positioned_page_text_blocks() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    parse_result.pages[0].width = 600
    parse_result.pages[0].height = 850
    parse_result.pages[0].texts = [
        SimpleNamespace(
            content="序号\n摘要\n交易日期\n交易金额\n账户余额\n对方账号与户名",
            bbox=[20.0, 200.0, 60.0, 215.0],
            evidence_ids=[],
        ),
        SimpleNamespace(
            content="1\n银联入账\n20250101\n1.00\n1.00\n6222020202020001/甲公司",
            bbox=[20.0, 220.0, 60.0, 235.0],
            evidence_ids=[],
        ),
        SimpleNamespace(
            content="2\n转账支取\n20250102\n-2.00\n-1.00\n6222020202020002/乙公司",
            bbox=[20.0, 240.0, 60.0, 255.0],
            evidence_ids=[],
        ),
    ]
    recovery = style_registry.recover_positioned_record_block_bank_tables(parse_result)
    assert recovery.expected_rows == 2
    assert len(recovery.tables[0][0]) > 4

    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_bound_positioned_records_without_account_field() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    parse_result.pages[0].width = 600
    parse_result.pages[0].height = 850
    blocks = []
    for sequence in range(1, 4):
        content = f"{sequence}\n转账收入\n2025-01-0{sequence}\n1.00\n{sequence}.00"
        atom_id = f"positioned-record:{sequence}"
        bbox = [20.0, 200.0 + sequence * 20.0, 60.0, 215.0 + sequence * 20.0]
        parse_result.evidence_plane.evidence["text_atoms"].append(
            {
                "id": atom_id,
                "page_id": "page:0001",
                "text": content,
                "bbox": bbox,
                "source_kind": "pdf_native",
            }
        )
        blocks.append(SimpleNamespace(content=content, bbox=bbox, evidence_ids=[atom_id]))
    parse_result.pages[0].texts = blocks
    recovery = style_registry.recover_positioned_record_block_bank_tables(parse_result)
    assert recovery.expected_rows == 3

    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_richer_semantic_text_table() -> None:
    candidate = _complete_primary_candidate()
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text=(
            "| date | amount | direction | summary | reference |\n"
            "| 2025-01-01 | 1.00 | income | row 1 | ref-1 |\n"
            "| 2025-01-02 | 2.00 | income | row 2 | ref-2 |"
        ),
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_hide_richer_stacked_semantic_text_table() -> None:
    headers = ["交易日期", "交易时间", "摘要", "交易金额", "余额", "对方户名"]
    rows = [
        ["2025-01-01", "10:00:00", "summary1", "+1.00", "100.00", "Alice"],
        ["2025-01-02", "11:00:00", "summary2", "+2.00", "102.00", "Bob"],
    ]
    records = []
    for row_index, row in enumerate(rows):
        records.append(
            {
                **dict(zip(headers, row, strict=True)),
                "_source": {
                    "source_page": 1,
                    "page_range": [1, 1],
                    "table_id": "table:1",
                    "source_row_index": row_index,
                    "source_cell_refs": [
                        {
                            "page": 1,
                            "table_id": "table:1",
                            "row": row_index,
                            "raw_row": row_index + 1,
                            "col": col_index,
                        }
                        for col_index in range(len(headers))
                    ],
                },
            }
        )
    candidate = replace(
        _complete_primary_candidate(),
        records=records,
        source_column_width=float(len(headers)),
    )
    full_text = "\n".join(
        [
            "交易明细",
            "对方户名",
            "备注",
            "余额",
            "收入/支出金额",
            "Alice",
            "ref-1",
            "100.00",
            "/",
            "2025-01-01",
            "10:00:00 +1.00 summary1",
            "Bob",
            "ref-2",
            "102.00",
            "/",
            "2025-01-02",
            "11:00:00 +2.00 summary2",
        ]
    )
    semantic = style_registry._semantic_text_table_candidates(full_text)
    assert len(semantic) == 1
    assert len(semantic[0][0]) == 7
    assert [row[-1] for row in semantic[0][1:]] == ["ref-1", "ref-2"]

    ctx = StyleContext(
        tables=[[headers, *rows]],
        full_text=full_text,
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(headers=headers, rows=rows),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_source_value_proof_preserves_internal_whitespace() -> None:
    candidate = _complete_primary_candidate()
    candidate.records[0]["summary"] = "row1"
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=_single_scope_parse_result(),
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_primary_cannot_prove_complete_when_physical_header_is_unbound() -> None:
    candidate = _complete_primary_candidate()
    parse_result = _single_scope_parse_result()
    parse_result.pages[0].tables[0].headers = ["x-date", "x-amount", "x-direction", "x-summary"]
    ctx = StyleContext(
        tables=[
            [
                ["date", "amount", "direction", "summary"],
                ["2025-01-01", "1.00", "income", "row 1"],
                ["2025-01-02", "2.00", "income", "row 2"],
            ]
        ],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    proof = style_registry._prove_primary_candidate_complete(candidate, detection, ctx)

    assert proof.state == "unknown"
    assert proof.reason == "canonical_source_columns_not_conserved"


def test_statement_scope_count_requires_explicit_context_fact() -> None:
    parse_result = _single_scope_parse_result()
    parse_result.evidence_plane = None

    assert style_registry.statement_scope_count(parse_result) == 0


def test_unknown_primary_reruns_exact_eager_collector_and_matches_forced_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def run_parser(parser_id, _ctx, _plugin):
        calls.append(parser_id)
        candidate = _complete_primary_candidate(1)
        return candidate.records, candidate.normalize_fn

    monkeypatch.setattr(style_registry, "_run_parser", run_parser)
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )

    def make_ctx() -> StyleContext:
        return StyleContext(
            tables=[],
            full_text="",
            institution=None,
            page_count=1,
            reconstruction=ReconstructionMeta(source="canonical_table"),
            prefer_context_tables=True,
        )

    adaptive = BankStyleParserRegistry()
    adaptive_records, _ = adaptive.run_parser_chain(
        detection,
        make_ctx(),
        BankStatementCommunityPlugin(),
    )
    adaptive_calls = list(calls)
    calls.clear()
    eager = BankStyleParserRegistry(adaptive=False)
    eager_records, _ = eager.run_parser_chain(
        detection,
        make_ctx(),
        BankStatementCommunityPlugin(),
    )

    assert adaptive_records == eager_records
    assert adaptive.last_selection_diagnostics["selected_candidate"] == eager.last_selection_diagnostics[
        "selected_candidate"
    ]
    assert adaptive.last_selection_diagnostics["deployment_mode"] == "eager_fallback"
    assert adaptive.last_selection_diagnostics["completion_state"] == "unknown"
    assert adaptive.last_selection_diagnostics["candidate_counts"] == eager.last_selection_diagnostics[
        "candidate_counts"
    ]
    assert adaptive_calls == ["grid_standard", "grid_standard", "signed_amount", "compact_merged"]
    assert calls == ["grid_standard", "signed_amount", "compact_merged"]


@pytest.mark.parametrize(
    ("primary_state", "expected_reason"),
    [
        ("empty", "primary_parser_returned_no_candidate"),
        ("partial", "primary_rows_do_not_match_issuer_count"),
    ],
)
def test_empty_or_partial_primary_selects_one_complete_whole_document_alternate(
    monkeypatch: pytest.MonkeyPatch,
    primary_state: str,
    expected_reason: str,
) -> None:
    def candidate(
        candidate_id: str,
        rows: list[tuple[int, str, int]],
        *,
        expected_source: str,
        expected_confidence: float,
        balance_chain_score: float,
        sequence_continuity: float,
        score: float,
    ) -> BankTableCandidate:
        raw_records = [
            {
                "date": f"2025-0{page}-01",
                "amount": f"{sequence}.00",
                "direction": "income",
                "balance": f"{100 + sequence}.00",
                "sequence_no": str(sequence),
                "statement_header_id": scope,
                "_source": {
                    "source_page": page,
                    "page_range": [page, page],
                    "table_id": f"{candidate_id}:plane",
                    "source_row_index": row_index,
                    "bbox": [10.0, row_index * 20.0, 200.0, row_index * 20.0 + 8.0],
                },
            }
            for row_index, (page, scope, sequence) in enumerate(rows)
        ]

        def normalize(raw: dict[str, object]) -> dict[str, object]:
            return {
                "date": raw["date"],
                "amount": float(str(raw["amount"])),
                "direction": raw["direction"],
                "balance": float(str(raw["balance"])),
                "sequence_no": raw["sequence_no"],
                "statement_header_id": raw["statement_header_id"],
            }

        return replace(
            _candidate_for_authority_test(
                candidate_id,
                len(rows),
                RowCountEvidence(len(rows), expected_source, expected_confidence),
            ),
            records=raw_records,
            normalize_fn=normalize,
            balance_chain_score=balance_chain_score,
            sequence_continuity=sequence_continuity,
            score=score,
        )

    partial = candidate(
        "parser:grid_standard",
        [
            (1, "statement_header:scope-a", 1),
            (2, "statement_header:scope-a", 2),
        ],
        expected_source="native_wide_rows",
        expected_confidence=0.70,
        balance_chain_score=0.98,
        sequence_continuity=1.0,
        score=0.98,
    )
    complete = candidate(
        "evidence_atom:complete",
        [
            (1, "statement_header:scope-a", 1),
            (2, "statement_header:scope-a", 2),
            (3, "statement_header:scope-b", 3),
            (3, "statement_header:scope-b", 4),
        ],
        expected_source="positioned_date_anchors",
        expected_confidence=0.80,
        balance_chain_score=0.50,
        sequence_continuity=0.95,
        score=0.88,
    )
    collector_calls: list[str] = []

    def collect_primary(*_args, **_kwargs):
        collector_calls.append("primary")
        return [] if primary_state == "empty" else [partial]

    def collect_eager(*_args, **_kwargs):
        collector_calls.append("eager")
        return [partial, complete]

    def prove(primary_candidate, _detection, _ctx):
        reason = (
            "primary_parser_returned_no_candidate"
            if primary_candidate is None
            else "primary_rows_do_not_match_issuer_count"
        )
        return style_registry.CandidateCompletionProof("unknown", reason)

    monkeypatch.setattr(style_registry, "_collect_primary_table_candidates", collect_primary)
    monkeypatch.setattr(style_registry, "_collect_table_candidates", collect_eager)
    monkeypatch.setattr(style_registry, "_prove_primary_candidate_complete", prove)

    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=3,
        reconstruction=ReconstructionMeta(source="canonical_table"),
    )
    registry = BankStyleParserRegistry()

    records, _identity = registry.run_parser_chain(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert collector_calls == ["primary", "eager"]
    assert registry.last_selection_diagnostics["deployment_mode"] == "eager_fallback"
    assert registry.last_selection_diagnostics["completion_state"] == "unknown"
    assert registry.last_selection_diagnostics["completion_reason"] == expected_reason
    assert registry.last_selection_diagnostics["attempted_strategies"] == [
        "parser:grid_standard",
        "parser:signed_amount",
        "parser:compact_merged",
        "semantic_text",
        "physical_table",
        "positioned_record_block",
        "evidence_atom",
        "native_wide_table",
    ]
    assert registry.last_selection_diagnostics["skipped_strategies"] == []
    assert "prior_lazy_attempt" not in registry.last_selection_diagnostics
    assert registry.last_selection_diagnostics["selected_candidate"] == "evidence_atom:complete"
    assert registry.last_selection_diagnostics["candidate_counts"] == {
        "parser:grid_standard": 2,
        "evidence_atom:complete": 4,
    }
    assert [
        (
            record["source"]["source_page"],
            record["normalized"]["statement_header_id"],
            record["normalized"]["sequence_no"],
            record["source"]["table_id"],
        )
        for record in records
    ] == [
        (1, "statement_header:scope-a", "1", "evidence_atom:complete:plane"),
        (2, "statement_header:scope-a", "2", "evidence_atom:complete:plane"),
        (3, "statement_header:scope-b", "3", "evidence_atom:complete:plane"),
        (3, "statement_header:scope-b", "4", "evidence_atom:complete:plane"),
    ]


@pytest.mark.parametrize(
    "policy,expected_candidate_ids",
    [
        (
            DIGITAL_POLICY,
            {
                "parser:grid_standard",
                "parser:signed_amount",
                "parser:compact_merged",
                "semantic_text:0",
                "physical_table",
                "positioned_record_block",
                "evidence_atom",
                "native_wide_table:0",
            },
        ),
        (
            SCANNED_POLICY,
            {
                "parser:grid_standard",
                "parser:signed_amount",
                "parser:borderless_ocr",
                "evidence_atom",
                "ocr_implicit_table",
            },
        ),
    ],
)
def test_eager_collector_retains_all_existing_strategy_nodes(
    monkeypatch: pytest.MonkeyPatch,
    policy: BankExtractionPolicy,
    expected_candidate_ids: set[str],
) -> None:
    detection = StyleDetectionResult(
        primary_style="grid_standard",
        confidence=0.95,
        parser_chain=["grid_standard"],
    )
    table = [["date", "amount", "direction"], ["2025-01-01", "1.00", "income"]]
    candidate = _complete_primary_candidate(1)
    positioned = SimpleNamespace(tables=[table], row_sources=[], expected_rows=1)

    monkeypatch.setattr(
        style_registry,
        "_run_parser",
        lambda *_args, **_kwargs: (candidate.records, candidate.normalize_fn),
    )
    monkeypatch.setattr(style_registry, "page_texts_from_parse_result", lambda _result: [])
    monkeypatch.setattr(
        style_registry,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence.empty(),
    )
    monkeypatch.setattr(style_registry, "recovered_native_datetime_row_evidence", lambda *_args, **_kwargs: (0, "", 0.0))
    monkeypatch.setattr(style_registry, "_semantic_text_table_candidates", lambda _text: [table])
    monkeypatch.setattr(style_registry, "collect_physical_tables_from_parse_result", lambda _result: [table])
    monkeypatch.setattr(style_registry, "collect_physical_table_row_sources_from_parse_result", lambda _result: [])
    monkeypatch.setattr(style_registry, "physical_transaction_row_estimate", lambda _result: 1)
    monkeypatch.setattr(
        style_registry,
        "recover_positioned_record_block_bank_tables",
        lambda *_args, **_kwargs: positioned,
    )
    monkeypatch.setattr(style_registry, "recover_evidence_atom_bank_tables", lambda *_args, **_kwargs: [table])
    monkeypatch.setattr(
        style_registry,
        "_evidence_atom_expected_rows",
        lambda *_args, **_kwargs: RowCountEvidence(1, "candidate_rows", 0.55),
    )
    monkeypatch.setattr(style_registry, "recovered_evidence_atom_row_sources", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(style_registry, "recover_wide_bank_tables", lambda *_args, **_kwargs: [table])
    monkeypatch.setattr(style_registry, "recover_ocr_implicit_ledger_tables", lambda *_args, **_kwargs: [table])
    monkeypatch.setattr(style_registry, "recovered_ocr_implicit_row_evidence", lambda _result: (0, "", 0.0))

    ctx = StyleContext(
        tables=[table],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=SimpleNamespace(pages=[], logical_tables=[]),
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    candidates = style_registry._collect_table_candidates(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert {item.candidate_id for item in candidates} == expected_candidate_ids


def test_builtin_templates_registered():
    registry_module = pytest.importorskip(
        "docmirror_enterprise.plugins.bank_statement.configs.registry",
        reason="enterprise bank-statement templates are not available in OSS CI",
    )

    registry_module.reset_registry()
    reg = registry_module.ensure_builtin_templates()
    ids = {t["template_id"] for t in reg.list_templates()}
    assert "generic" in ids
    assert "icbc_personal_v2022" in ids
    assert reg.template_count >= 3


def test_style_registry_extracts_three_transactions_from_clean_table():
    ctx = StyleContext(
        tables=[
            [
                ["交易日期", "摘要", "收入", "支出", "余额"],
                ["2024-01-01", "工资入账", "5000.00", "0.00", "8000.00"],
                ["2024-01-02", "转账支出", "0.00", "200.00", "7800.00"],
                ["2024-01-03", "消费", "0.00", "50.00", "7750.00"],
            ]
        ],
        full_text="中国工商银行\n个人客户交易明细\n户名：张三",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    assert detection.primary_style == "split_debit_credit"
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3


def test_preferred_context_tables_exclude_document_acquisition_planes(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = StyleContext(
        tables=[
            [
                ["交易日期", "摘要", "收入", "支出", "余额"],
                ["2024-01-02", "continued", "", "2.00", "7.00"],
                ["2024-01-03", "continued", "3.00", "", "10.00"],
            ]
        ],
        full_text="document-wide text must stay outside this local continuation batch",
        institution=None,
        page_count=14,
        prefer_context_tables=True,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("document-wide acquisition ran inside a context-tables-only batch")

    for name in (
        "recovered_native_datetime_row_evidence",
        "recover_evidence_atom_bank_tables",
        "_semantic_text_table_candidates",
        "collect_physical_tables_from_parse_result",
        "recover_positioned_record_block_bank_tables",
        "recover_wide_bank_tables",
        "recover_ocr_implicit_ledger_tables",
    ):
        monkeypatch.setattr(style_registry, name, forbidden)

    candidates = style_registry._collect_table_candidates(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert candidates
    assert all(candidate.candidate_id.startswith("parser:") for candidate in candidates)
    assert max(len(candidate.records) for candidate in candidates) == 2


def _physical_candidate_parse_result(*, blank_body_geometry: bool = False) -> SimpleNamespace:
    headers = ["交易日期", "摘要", "收入", "支出", "余额"]
    values = ["2024-01-02", "工资", "100.00", "", "100.00"]
    header_boxes = [[float(index * 60), 1.0, float((index + 1) * 60), 10.0] for index in range(5)]
    body_boxes = (
        [None] * 5
        if blank_body_geometry
        else [
            [10.0, 20.0, 55.0, 32.0],
            [55.0, 20.0, 115.0, 32.0],
            [115.0, 20.0, 175.0, 32.0],
            [175.0, 20.0, 235.0, 32.0],
            [235.0, 20.0, 300.0, 32.0],
        ]
    )
    body_evidence = [[], [], [], [], []] if blank_body_geometry else [
        ["ev:body:date"],
        ["ev:body:summary"],
        ["ev:body:income"],
        [],
        ["ev:body:balance"],
    ]
    cells = [
        SimpleNamespace(
            text=value,
            cleaned=None,
            bbox=None,
            evidence_ids=[],
            source_cell_refs=[],
        )
        for value in values
    ]
    row = SimpleNamespace(
        cells=cells,
        source_page=3,
        source_physical_id="pt_3_7",
        source_row_index=4,
        source_cell_refs=[],
    )
    table = SimpleNamespace(
        headers=headers,
        rows=[row],
        table_id="pt_3_7",
        bbox=[0.0, 0.0, 999.0, 999.0],
        metadata={
            "raw_rows": [headers, values],
            "geometry": {
                "cell_bboxes": [header_boxes, body_boxes],
                "cell_evidence_ids": [
                    [[f"ev:header:{index}"] for index in range(5)],
                    body_evidence,
                ],
            },
        },
    )
    page = SimpleNamespace(page_number=3, source_page_number=3, tables=[table], texts=[])
    return SimpleNamespace(pages=[page], logical_tables=[], full_text="")


def _physical_candidate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    blank_body_geometry: bool = False,
) -> style_registry.BankTableCandidate:
    parse_result = _physical_candidate_parse_result(blank_body_geometry=blank_body_geometry)
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.DIGITAL,
        allowed_parser_ids=frozenset(),
        allow_physical_tables=True,
    )
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    monkeypatch.setattr(
        style_registry,
        "recovered_native_datetime_row_evidence",
        lambda *_args, **_kwargs: (0, "", 0.0),
    )
    monkeypatch.setattr(style_registry, "physical_transaction_row_estimate", lambda _result: 1)

    candidates = style_registry._collect_table_candidates(
        StyleDetectionResult(primary_style="grid_standard", parser_chain=[]),
        ctx,
        BankStatementCommunityPlugin(),
    )

    return next(candidate for candidate in candidates if candidate.candidate_id == "physical_table")


def test_physical_candidate_attaches_exact_body_row_geometry_and_never_header_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _physical_candidate(monkeypatch)

    assert len(candidate.records) == 1
    source = candidate.records[0]["_source"]
    assert source["source"] == "canonical_physical_table"
    assert source["source_page"] == 3
    assert source["page_range"] == [3, 3]
    assert source["table_id"] == "pt_3_7"
    assert source["source_row_index"] == 4
    assert source["bbox"] == [10.0, 20.0, 300.0, 32.0]
    assert source["evidence_ids"] == [
        "ev:body:date",
        "ev:body:summary",
        "ev:body:income",
        "ev:body:balance",
    ]
    assert len(source["source_cell_refs"]) == 5
    assert {ref["raw_row"] for ref in source["source_cell_refs"]} == {1}
    assert {ref["row"] for ref in source["source_cell_refs"]} == {4}
    assert not any(value.startswith("ev:header:") for value in source["evidence_ids"])
    assert source["bbox"] != [0.0, 0.0, 999.0, 999.0]


def test_physical_candidate_blank_row_geometry_keeps_identity_without_borrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _physical_candidate(monkeypatch, blank_body_geometry=True)

    assert len(candidate.records) == 1
    source = candidate.records[0]["_source"]
    assert source["source"] == "canonical_physical_table"
    assert source["source_page"] == 3
    assert source["page_range"] == [3, 3]
    assert source["table_id"] == "pt_3_7"
    assert source["source_row_index"] == 4
    assert "bbox" not in source
    assert "evidence_ids" not in source
    assert "source_cell_refs" not in source


def test_style_registry_extracts_wide_debit_credit_table_and_skips_footer():
    ctx = StyleContext(
        tables=[
            [
                [
                    "序号",
                    "会计日期",
                    "交易日期",
                    "交易名称",
                    "借方发生额",
                    "贷方发生额",
                    "余额",
                    "对方账号",
                    "对方户名",
                    "摘要",
                ],
                [
                    "1",
                    "20251114",
                    "20251114",
                    "来账",
                    "",
                    "120,000.00",
                    "139,038.63",
                    "011101421000 9630",
                    "重庆正大华日软 件有限公司",
                    "往来款",
                ],
                [
                    "2",
                    "20251114",
                    "20251114",
                    "代付",
                    "97,462.92",
                    "",
                    "41,575.71",
                    "641106012890 900100012499",
                    "应付代收业务款 项",
                    "代发工资",
                ],
                ["当前账单借方发生数： 1", "当前账单贷方发生数：1", "", "", "", "", "", "", "", ""],
            ]
        ],
        full_text="交通银行\n当前账单借方发生数：1 当前账单贷方发生数：1 本月累计借方发生额：97,462.92 本月累计贷方发生额：120,000.00",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)

    assert len(records) == 2
    assert [r["normalized"]["direction"] for r in records] == ["income", "expense"]
    assert records[0]["normalized"]["counter_account"] == "0111014210009630"
    assert records[0]["normalized"]["counter_party"] == "重庆正大华日软件有限公司"


def test_cross_page_native_income_expense_table_inherits_header():
    page_tables = [
        [
            ["序 号", "交易日期", "交易时 间", "支出金额", "收入金额", "余额", "对方账号", "对方户名"],
            [
                "1",
                "2023-12- 28",
                "15:28:5 3",
                "",
                "2,800.00",
                "2,932.04",
                "24020034091 00018033",
                "贵阳世钟 汽车配件 有限公司",
            ],
            ["2", "2023-12- 27", "14:06:0 2", "7.00", "", "132.04", "60220903", "网上银行 结算手续 费收入"],
        ],
        [
            [
                "3",
                "2023-12- 27",
                "14:06:0 2",
                "10,500.00",
                "",
                "139.04",
                "32050161716 000000050",
                "无锡市融 达汽车零 部件有限 公司",
            ],
            ["4", "2023-12- 27", "13:58:5 9", "", "10,500.00", "10,639.04", "62284810431 55907917", "张淑红"],
        ],
    ]
    tables = _recover_cross_page_wide_tables(page_tables)

    assert len(tables) == 1
    assert len(tables[0]) == 5
    assert tables[0][1][1] == "2023-12-28"
    assert tables[0][1][2] == "15:28:53"

    ctx = StyleContext(
        tables=tables,
        full_text="收入总金额：13300.00 收入总笔数：2 支出总金额：10507.00 支出总笔数：2",
        institution=None,
        page_count=2,
    )
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(BankStyleDetector().detect(ctx), ctx, plugin)
    norms = [record["normalized"] for record in records]

    assert len(records) == 4
    assert sum(1 for norm in norms if norm["direction"] == "income") == 2
    assert sum(1 for norm in norms if norm["direction"] == "expense") == 2
    assert norms[0]["counter_party"] == "贵阳世钟汽车配件有限公司"


def test_removed_detector_is_not_registered():
    pytest.importorskip("docmirror_enterprise", reason="enterprise package is not available in OSS CI")

    with pytest.raises(ImportError):
        from docmirror_enterprise.plugins.bank_statement.detectors.template_detector import (  # noqa: F401
            BankStatementDetector,
        )


@pytest.mark.skipif(
    not os.environ.get("DOCMIRROR_RUN_SYNTHETIC_TESTS"),
    reason="Synthetic PDF OCR test requires DOCMIRROR_RUN_SYNTHETIC_TESTS=1",
)
@pytest.mark.asyncio
async def test_bank_synthetic_extracts_transactions():
    from docmirror.plugins.bank_statement.context import collect_tables_from_parse_result
    from scripts.generate_synthetic_golden_pdfs import ensure_bank_synthetic
    from tests.golden.test_golden_matrix_benchmark import _parse_case

    pdf = ensure_bank_synthetic()
    pr = await _parse_case(pdf)
    assert pr.entities.document_type == "bank_statement"
    assert len(pr.extractor_full_text or pr.full_text) > 50

    ctx = StyleContext(
        tables=collect_tables_from_parse_result(pr),
        full_text=pr.full_text or "",
        institution=None,
        page_count=len(pr.pages or []),
        parse_result=pr,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3

    api = project_mirror(seal_parse_result(pr))
    doc = api
    assert len(doc.get("pages") or []) >= 1
