# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bank Ledger Orchestrator (BLO) unit tests."""

from __future__ import annotations

from docmirror.models.entities.parse_result import (
    CellValue,
    LogicalTable,
    PageContent,
    ParseResult,
    TableBlock,
    TableRow,
)
from docmirror.plugins.bank_statement.blo import (
    BankLedgerOrchestrator,
    _attach_quarantine_sources,
    _merge_quarantine_continuation_matrices,
    _quarantine_continuation_sources,
    logical_table_to_matrices,
)
from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import SCANNED_POLICY
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry


def test_logical_table_to_matrices_includes_headers():
    lt = LogicalTable(
        headers=["交易日期", "摘要", "余额"],
        rows=[
            TableRow(cells=[CellValue(text="2024-01-01"), CellValue(text="x"), CellValue(text="1")]),
        ],
        row_count=1,
        logical_id="lt_0",
    )
    matrices = logical_table_to_matrices(lt)
    assert matrices[0][0] == ["交易日期", "摘要", "余额"]
    assert len(matrices[0]) == 2


def test_dedupe_transaction_rows():
    records = [
        {
            "row_index": 1,
            "normalized": {"date": "2024-01-01", "amount": 1.0, "balance": 2.0, "counter_party": "a"},
            "source": {"source_page": 1, "page_range": [1, 1], "table_id": "table:1", "source_row_index": 1},
        },
        {
            "row_index": 2,
            "normalized": {"date": "2024-01-01", "amount": 1.0, "balance": 2.0, "counter_party": "a"},
            "source": {"source_page": 1, "page_range": [1, 1], "table_id": "table:1", "source_row_index": 1},
        },
    ]
    out = dedupe_transaction_rows(records)
    assert len(out) == 1
    assert out[0]["row_index"] == 1


def test_dedupe_preserves_identical_unsourced_business_rows() -> None:
    records = [
        {"row_index": 1, "normalized": {"date": "2024-01-01", "amount": 1.0, "balance": 2.0}},
        {"row_index": 2, "normalized": {"date": "2024-01-01", "amount": 1.0, "balance": 2.0}},
    ]

    out = dedupe_transaction_rows(records)

    assert len(out) == 2


def test_dedupe_keeps_identical_business_rows_from_different_source_pages():
    records = [
        {
            "row_index": 1,
            "normalized": {
                "date": "2024-01-01",
                "direction": "expense",
                "amount": 1.0,
                "balance": 2.0,
                "counter_party": "a",
            },
            "source": {"source_page": 1, "page_range": [1, 1], "source_row_index": 8},
        },
        {
            "row_index": 2,
            "normalized": {
                "date": "2024-01-01",
                "direction": "expense",
                "amount": 1.0,
                "balance": 2.0,
                "counter_party": "a",
            },
            "source": {"source_page": 2, "page_range": [2, 2], "source_row_index": 8},
        },
    ]

    out = dedupe_transaction_rows(records)

    assert len(out) == 2
    assert [record["source"]["source_page"] for record in out] == [1, 2]


def test_dedupe_preserves_repeated_same_page_transactions_at_distinct_bboxes() -> None:
    def record(*, amount: float, balance: float, top: float) -> dict:
        return {
            "normalized": {
                "date": "2024-01-01",
                "direction": "expense" if amount < 0 else "income",
                "amount": abs(amount),
                "balance": balance,
                "summary": "batch transfer",
            },
            "source": {"source_page": 1, "page_range": [1, 1], "bbox": [36.0, top, 456.0, top + 9.75]},
        }

    records = [
        record(amount=-100.0, balance=900.0, top=100.0),
        record(amount=100.0, balance=1000.0, top=125.0),
        record(amount=-100.0, balance=900.0, top=150.0),
        record(amount=100.0, balance=1000.0, top=175.0),
        record(amount=-100.0, balance=900.0, top=200.0),
        record(amount=100.0, balance=1000.0, top=225.0),
    ]

    out = dedupe_transaction_rows(records)

    assert len(out) == 6
    assert [record["source"]["bbox"][1] for record in out] == [100.0, 125.0, 150.0, 175.0, 200.0, 225.0]


def test_dedupe_collapses_the_same_bbox_backed_source_row() -> None:
    record = {
        "normalized": {
            "date": "2024-01-01",
            "direction": "expense",
            "amount": 100.0,
            "balance": 900.0,
            "summary": "fee",
        },
        "source": {"source_page": 1, "page_range": [1, 1], "bbox": [36.0, 100.0, 456.0, 109.75]},
    }

    out = dedupe_transaction_rows([record, dict(record)])

    assert len(out) == 1


def test_dedupe_does_not_trust_page_scope_or_repeated_sequence_as_row_identity() -> None:
    records = [
        {
            "normalized": {
                "sequence_no": "1",
                "date": "2024-01-01",
                "direction": "expense",
                "amount": 1.0,
                "balance": 9.0,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
        {
            "normalized": {
                "sequence_no": "1",
                "date": "2024-01-01",
                "direction": "expense",
                "amount": 1.0,
                "balance": 9.0,
            },
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
    ]

    assert len(dedupe_transaction_rows(records)) == 2


def test_blo_skips_failed_ltqg_table():
    good = LogicalTable(
        headers=["交易日期", "摘要", "收入", "支出", "余额"],
        rows=[
            TableRow(
                cells=[
                    CellValue(text="2024-01-01"),
                    CellValue(text="x"),
                    CellValue(text="0"),
                    CellValue(text="1"),
                    CellValue(text="9"),
                ]
            ),
        ],
        row_count=1,
        logical_id="lt_good",
        quality_passed=True,
    )
    bad = LogicalTable(
        headers=["", "", ""],
        rows=[TableRow(cells=[CellValue(text="?"), CellValue(text="?"), CellValue(text="?")])],
        row_count=20,
        logical_id="lt_bad",
        quality_passed=False,
        quality_skip_reason="fragment_table",
    )
    ctx = StyleContext(
        tables=[],
        full_text="中国工商银行",
        institution=None,
        page_count=2,
        parse_result=ParseResult(logical_tables=[good, bad]),
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _, meta = BankLedgerOrchestrator(BankStyleParserRegistry()).run(detection, ctx, plugin)
    assert meta.tables_skipped == 1
    assert meta.tables_parsed == 1
    assert len(records) >= 1


def _quarantine_continuation_tables() -> tuple[LogicalTable, LogicalTable]:
    good = LogicalTable(
        headers=["交易日期", "摘要", "收入", "支出", "余额"],
        rows=[
            TableRow(
                cells=[
                    CellValue(text="2024-01-01"),
                    CellValue(text="start"),
                    CellValue(text=""),
                    CellValue(text="1"),
                    CellValue(text="9"),
                ],
                source_page=13,
            )
        ],
        logical_id="lt_main",
        quality_passed=True,
        source_pages=[1, *range(2, 14)],
        page_span=(1, 13),
    )
    continuation = LogicalTable(
        # The table composer promoted the first real row to ``headers``.
        headers=["20240102\n12:34:56", "continued", "", "2", "7"],
        rows=[
            TableRow(
                cells=[
                    CellValue(text="2024-01-03"),
                    CellValue(text="continued"),
                    CellValue(text="3"),
                    CellValue(text=""),
                    CellValue(text="10"),
                ],
                source_page=14,
            )
        ],
        logical_id="lt_quarantined_tail",
        quality_passed=False,
        quality_skip_reason="merge_quarantine",
        source_physical_ids=["pt_14_0"],
        source_pages=[14],
        page_span=(14, 14),
    )
    return good, continuation


def test_blo_recovers_adjacent_transaction_shaped_merge_quarantine_tail() -> None:
    good, continuation = _quarantine_continuation_tables()
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=14,
        parse_result=ParseResult(logical_tables=[good, continuation]),
    )

    records, _, meta = BankLedgerOrchestrator(BankStyleParserRegistry()).run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert [record["normalized"]["date"] for record in records] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert meta.tables_parsed == 2
    assert meta.tables_skipped == 0
    assert records[1]["source"]["source_page"] == 14
    assert records[1]["source"]["page_range"] == [14, 14]
    assert records[2]["source"]["source_page"] == 14


def test_quarantine_tail_rebinds_exact_physical_cell_provenance() -> None:
    good, continuation = _quarantine_continuation_tables()
    recovered = _merge_quarantine_continuation_matrices(good, continuation)
    physical = TableBlock(
        table_id="pt_14_0",
        page=14,
        headers=list(continuation.headers),
        rows=[
            TableRow(
                cells=[CellValue(text=cell.text) for cell in continuation.rows[0].cells],
                source_page=14,
                source_physical_id="pt_14_0",
                source_row_index=1,
            )
        ],
        metadata={
            "raw_rows": [list(continuation.headers), continuation.rows[0].cell_texts],
            "geometry": {
                "cell_bboxes": [
                    [[float(col), 10.0, float(col + 1), 20.0] for col in range(5)],
                    [[float(col), 20.0, float(col + 1), 30.0] for col in range(5)],
                ],
                "cell_evidence_ids": [
                    [[f"ev:14:0:{col}"] for col in range(5)],
                    [[f"ev:14:1:{col}"] for col in range(5)],
                ],
            },
        },
    )
    parse_result = ParseResult(
        pages=[PageContent(page_number=14, tables=[physical])],
        logical_tables=[good, continuation],
    )

    sources = _quarantine_continuation_sources(parse_result, good, continuation, recovered)

    assert len(sources) == 2
    assert sources[0]["source"] == {
        "source": "canonical_physical_table",
        "source_page": 14,
        "page_id": "page:0014",
        "page_range": [14, 14],
        "table_id": "pt_14_0",
        "source_row_index": 0,
        "bbox": [0.0, 10.0, 5.0, 20.0],
        "evidence_ids": [f"ev:14:0:{col}" for col in range(5)],
        "source_cell_refs": [
            {
                "source": "canonical_physical_table",
                "page": 14,
                "table_id": "pt_14_0",
                "row": 0,
                "raw_row": 0,
                "col": col,
            }
            for col in range(5)
            if continuation.headers[col]
        ],
    }
    assert sources[1]["source"]["source_row_index"] == 1
    assert sources[1]["source"]["bbox"] == [0.0, 20.0, 5.0, 30.0]
    assert sources[1]["source_raw"] == dict(zip(good.headers, continuation.rows[0].cell_texts, strict=True))


def test_quarantine_source_binding_rejects_filtered_or_reordered_rows() -> None:
    sources = [
        {
            "source": {"source_page": 14, "source_row_index": 0},
            "source_raw": {"交易日期": "2024-01-02", "支出": "2"},
        },
        {
            "source": {"source_page": 14, "source_row_index": 1},
            "source_raw": {"交易日期": "2024-01-03", "收入": "3"},
        },
    ]
    records = [
        {"normalized": {"date": "2024-01-03", "direction": "income", "amount": 3.0}},
        {"normalized": {"date": "2024-01-02", "direction": "expense", "amount": 2.0}},
    ]

    assert _attach_quarantine_sources(records[:1], sources) == []
    assert _attach_quarantine_sources(records, sources) == []


def test_blo_quarantine_recovery_prefers_the_guarded_context_matrix() -> None:
    good, continuation = _quarantine_continuation_tables()
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=14,
        parse_result=ParseResult(logical_tables=[good, continuation]),
    )
    seen: list[bool] = []

    class ContextAwareRegistry:
        last_selection_diagnostics: dict = {}

        def run_parser_chain(self, _detection, sub_ctx, _plugin):
            seen.append(sub_ctx.prefer_context_tables)
            days = (2, 3) if sub_ctx.prefer_context_tables else (1,)
            return [
                {
                    "normalized": {
                        "date": f"2024-01-0{day}",
                        "direction": "expense",
                        "amount": float(day),
                        "balance": float(10 - day),
                    },
                    "source": {"source_page": 14 if day > 1 else 13, "source_row_index": day},
                }
                for day in days
            ], {}

    records, _, _ = BankLedgerOrchestrator(ContextAwareRegistry()).run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert seen == [False, True]
    assert [record["normalized"]["date"] for record in records] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]


def test_merge_quarantine_recovery_stays_fail_closed() -> None:
    good, continuation = _quarantine_continuation_tables()

    nontransaction_header = continuation.model_copy(update={"headers": ["footer", "", "", "", ""]})
    width_mismatch = continuation.model_copy(
        update={"headers": ["2024-01-02", "continued", "2", "7"]}
    )
    page_gap = continuation.model_copy(update={"source_pages": [15], "page_span": (15, 15)})
    page_rewind = continuation.model_copy(update={"source_pages": [13], "page_span": (13, 13)})

    assert _merge_quarantine_continuation_matrices(good, nontransaction_header) == []
    assert _merge_quarantine_continuation_matrices(good, width_mismatch) == []
    assert _merge_quarantine_continuation_matrices(good, page_gap) == []
    assert _merge_quarantine_continuation_matrices(good, page_rewind) == []


def test_blo_preserves_extraction_policy_for_each_logical_table():
    tables = [
        LogicalTable(
            headers=["浜ゆ槗鏃ユ湡", "鎽樿", "鏀跺叆", "鏀嚭", "浣欓"],
            rows=[
                TableRow(
                    cells=[
                        CellValue(text=f"2024-01-0{index}"),
                        CellValue(text="x"),
                        CellValue(text="1"),
                        CellValue(text="0"),
                        CellValue(text=str(index)),
                    ]
                )
            ],
            logical_id=f"lt_{index}",
            quality_passed=True,
        )
        for index in (1, 2)
    ]
    parse_result = ParseResult(logical_tables=tables)
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=2,
        parse_result=parse_result,
        extraction_route=SCANNED_POLICY.route,
        extraction_policy=SCANNED_POLICY,
    )
    seen = []

    class RecordingRegistry:
        last_selection_diagnostics = {}

        def run_parser_chain(self, _detection, sub_ctx, _plugin):
            seen.append((sub_ctx.extraction_route, sub_ctx.extraction_policy))
            return [], {}

    BankLedgerOrchestrator(RecordingRegistry()).run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert seen == [
        (SCANNED_POLICY.route, SCANNED_POLICY),
        (SCANNED_POLICY.route, SCANNED_POLICY),
    ]


def test_sparse_logical_table_does_not_suppress_fuller_document_context() -> None:
    logical = LogicalTable(
        headers=["交易日期", "收入", "支出", "余额"],
        rows=[
            TableRow(
                cells=[
                    CellValue(text="2024-01-01"),
                    CellValue(text="1"),
                    CellValue(text=""),
                    CellValue(text="1"),
                ]
            )
        ],
        row_count=1,
        logical_id="lt_sparse",
        quality_passed=True,
    )
    ctx = StyleContext(
        tables=[
            [["交易日期", "收入", "支出", "余额"], ["2024-01-01", "1", "", "1"]],
            [["交易日期", "收入", "支出", "余额"], ["2024-01-02", "2", "", "3"]],
            [["交易日期", "收入", "支出", "余额"], ["2024-01-03", "3", "", "6"]],
        ],
        full_text="",
        institution=None,
        page_count=3,
        parse_result=ParseResult(logical_tables=[logical]),
    )
    detection = BankStyleDetector().detect(ctx)

    records, _, _ = BankLedgerOrchestrator(BankStyleParserRegistry()).run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 3
    assert [record["normalized"]["date"] for record in records] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]


def _blo_record(day: int, *, canonical: bool = True) -> dict:
    normalized = {"date": f"2024-01-{day:02d}", "direction": "income", "amount": float(day)}
    if not canonical:
        normalized.pop("direction")
    return {"normalized": normalized, "source": {"source_page": day, "source_row_index": day}}


def _blo_logical_table(logical_id: str) -> LogicalTable:
    return LogicalTable(
        headers=["date", "amount"],
        rows=[TableRow(cells=[CellValue(text="2024-01-01"), CellValue(text="1")])],
        logical_id=logical_id,
        quality_passed=True,
    )


def test_rejected_document_trial_does_not_replace_logical_reconstruction() -> None:
    logical = _blo_logical_table("lt_preferred")
    ctx = StyleContext(
        tables=[[['date', 'amount'], ['2024-01-01', '1']]],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=ParseResult(logical_tables=[logical]),
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=2),
    )

    class Registry:
        last_selection_diagnostics = {}
        calls = 0

        def run_parser_chain(self, _detection, sub_ctx, _plugin):
            self.calls += 1
            if self.calls == 1:
                sub_ctx.reconstruction = ReconstructionMeta(source="canonical_table", expected_primary_rows=2)
                self.last_selection_diagnostics = {"selected_candidate": "logical"}
                return [_blo_record(1), _blo_record(2)], {}
            sub_ctx.reconstruction = ReconstructionMeta(source="native_wide_table", expected_primary_rows=99)
            self.last_selection_diagnostics = {"selected_candidate": "document"}
            return [_blo_record(1)], {}

    records, _, meta = BankLedgerOrchestrator(Registry()).run(
        BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin()
    )

    assert len(records) == 2
    assert ctx.reconstruction.source == "canonical_table"
    assert ctx.reconstruction.expected_primary_rows == 2
    assert meta.candidate_diagnostics[0]["selected_candidate"] == "logical"


def test_larger_document_result_with_fewer_canonical_rows_is_not_preferred() -> None:
    logical = _blo_logical_table("lt_canonical")
    ctx = StyleContext(
        tables=[[['date', 'amount'], ['2024-01-01', '1']]],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=ParseResult(logical_tables=[logical]),
    )

    class Registry:
        last_selection_diagnostics = {}
        calls = 0

        def run_parser_chain(self, _detection, _sub_ctx, _plugin):
            self.calls += 1
            if self.calls == 1:
                return [_blo_record(1), _blo_record(2)], {}
            return [_blo_record(1), _blo_record(2, canonical=False), _blo_record(3, canonical=False)], {}

    records, _, _ = BankLedgerOrchestrator(Registry()).run(
        BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin()
    )

    assert len(records) == 2
    assert all(record["normalized"].get("direction") == "income" for record in records)


def test_document_context_runs_once_when_all_logical_results_are_empty() -> None:
    logical_tables = [_blo_logical_table("lt_1"), _blo_logical_table("lt_2")]
    ctx = StyleContext(
        tables=[[['date', 'amount'], ['2024-01-03', '3']]],
        full_text="",
        institution=None,
        page_count=2,
        parse_result=ParseResult(logical_tables=logical_tables),
    )

    class Registry:
        last_selection_diagnostics = {}
        calls = 0

        def run_parser_chain(self, _detection, _sub_ctx, _plugin):
            self.calls += 1
            return ([], {}) if self.calls <= 2 else ([_blo_record(3)], {})

    registry = Registry()
    records, _, _ = BankLedgerOrchestrator(registry).run(
        BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin()
    )

    assert registry.calls == 3
    assert len(records) == 1


def test_candidate_sequence_document_result_does_not_beat_fuller_logical_row() -> None:
    logical_records = [_blo_record(day) for day in range(1, 6)]
    document_records = [_blo_record(day) for day in range(1, 5)]
    exact = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=4,
        expected_evidence_source="continuous_source_sequence",
        expected_evidence_confidence=0.80,
    )

    assert not BankLedgerOrchestrator._prefer_document_result(
        logical_records,
        document_records,
        logical_reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=5),
        document_reconstruction=exact,
    )


def test_candidate_sequence_logical_result_loses_to_fuller_document_row() -> None:
    logical_records = [_blo_record(day) for day in range(1, 5)]
    document_records = [_blo_record(day) for day in range(1, 6)]
    exact = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=4,
        expected_evidence_source="continuous_source_sequence",
        expected_evidence_confidence=0.80,
    )

    assert BankLedgerOrchestrator._prefer_document_result(
        logical_records,
        document_records,
        logical_reconstruction=exact,
        document_reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=5),
    )


def test_row_plane_census_document_result_does_not_beat_fuller_logical_row() -> None:
    logical_records = [_blo_record(day) for day in range(1, 6)]
    document_records = [_blo_record(day) for day in range(1, 5)]
    bounded_source_census = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=4,
        expected_evidence_source="native_page_signed_ledger_census",
        expected_evidence_confidence=0.99,
    )

    assert not BankLedgerOrchestrator._prefer_document_result(
        logical_records,
        document_records,
        logical_reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=5),
        document_reconstruction=bounded_source_census,
    )


def test_cumulative_footer_document_result_beats_one_extra_unproven_logical_row() -> None:
    logical_records = [_blo_record(day) for day in range(1, 6)]
    document_records = [_blo_record(day) for day in range(1, 5)]
    issuer_total = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=4,
        expected_evidence_source="cumulative_footer_total",
        expected_evidence_confidence=0.99,
    )

    assert BankLedgerOrchestrator._prefer_document_result(
        logical_records,
        document_records,
        logical_reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=5),
        document_reconstruction=issuer_total,
    )


def test_issuer_footer_complete_logical_result_beats_truncated_exact_sequence_document() -> None:
    logical_records = [_blo_record(day) for day in range(1, 14)]
    document_records = [_blo_record(day) for day in range(1, 13)]
    issuer_total = ReconstructionMeta(
        source="canonical_table",
        expected_primary_rows=13,
        expected_evidence_source="split_footer",
        expected_evidence_confidence=0.98,
    )
    sequence = ReconstructionMeta(
        source="native_wide_table",
        expected_primary_rows=12,
        expected_evidence_source="continuous_source_sequence",
        expected_evidence_confidence=0.99,
    )

    assert not BankLedgerOrchestrator._prefer_document_result(
        logical_records,
        document_records,
        logical_reconstruction=issuer_total,
        document_reconstruction=sequence,
    )
