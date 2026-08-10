# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical evidence atom split debit/credit bank ledger recovery tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import CellValue, TableBlock, TableRow
from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.evidence_atom_table_recovery import (
    _column_aggregate_source_raw,
    _infer_positioned_block_directions,
    _is_geometry_footer_text,
    _join_geometry_atoms,
    _positioned_block_counter_account,
    _repair_geometry_rows,
    _sort_positioned_block_records,
    recover_evidence_atom_bank_tables,
    recover_positioned_record_block_bank_tables,
    recovered_evidence_atom_expected_row_count,
    recovered_evidence_atom_expected_row_evidence,
    recovered_evidence_atom_row_sources,
)
from docmirror.plugins.bank_statement.style_registry import (
    BankTableCandidate,
    _candidate_from_batch,
    _select_candidate,
)
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence

pytestmark = pytest.mark.unit


def _atom(atom_id: str, text: str, x0: float, y0: float, x1: float | None = None) -> dict:
    return {
        "id": atom_id,
        "page_id": "page:0001",
        "text": text,
        "bbox": [x0, y0, x1 if x1 is not None else x0 + 20.0, y0 + 8.0],
    }


def _result(atoms: list[dict], vector_atoms: list[dict] | None = None) -> SimpleNamespace:
    page_numbers = sorted({int(str(atom.get("page_id") or "page:0001").rsplit(":", 1)[-1]) for atom in atoms})
    return SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence={
                "text_atoms": atoms,
                "vector_atoms": list(vector_atoms or []),
            }
        ),
        pages=[SimpleNamespace(page_number=page, width=600, height=850, tables=[], texts=[]) for page in page_numbers],
        logical_tables=[],
        entities=SimpleNamespace(domain_specific={}),
    )


def _rotated_90(atom: dict, *, page_id: str) -> dict:
    x0, y0, x1, y1 = atom["bbox"]
    return {
        **atom,
        "page_id": page_id,
        "bbox": [600.0 - y1, x0, 600.0 - y0, x1],
    }


def _candidate(
    candidate_id: str,
    *,
    canonical_coverage: float = 1.0,
    source_page_coverage: float = 1.0,
    field_completeness: float = 1.0,
    balance_chain_score: float = 1.0,
    score: float = 0.9,
    rows: int = 3,
    expected_rows: RowCountEvidence | None = None,
    source_column_width: float = 0.0,
    extraction_confidence: float = 0.0,
    sequence_continuity: float = 0.0,
) -> BankTableCandidate:
    return BankTableCandidate(
        candidate_id=candidate_id,
        records=[{} for _ in range(rows)],
        source=candidate_id,
        canonical_rows=rows,
        directional_rows=rows,
        source_page_rows=round(rows * source_page_coverage),
        expected_rows=expected_rows,
        balance_chain_score=balance_chain_score,
        field_completeness=field_completeness,
        score=score,
        canonical_coverage=canonical_coverage,
        source_page_coverage=source_page_coverage,
        source_column_width=source_column_width,
        extraction_confidence=extraction_confidence,
        sequence_continuity=sequence_continuity,
    )


def test_candidate_selection_rejects_larger_ocr_result_without_page_provenance():
    selected, diagnostics = _select_candidate(
        [
            _candidate("physical_table", score=0.88, rows=4),
            _candidate("ocr_implicit_table", source_page_coverage=0.0, score=0.80, rows=6),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "physical_table"
    assert diagnostics["selected_candidate"] == "physical_table"


def test_candidate_selection_prefers_continuous_source_sequence_over_noisy_extra_rows():
    selected, diagnostics = _select_candidate(
        [
            _candidate("evidence_atom", rows=199, sequence_continuity=1.0, score=0.88),
            _candidate("ocr_implicit_table", rows=211, sequence_continuity=0.0, score=0.92),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "evidence_atom"
    assert diagnostics["selected_candidate"] == "evidence_atom"


def test_candidate_scoring_rejects_column_aggregated_dates_copied_as_rows() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "amount_cny", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
    )
    collapsed_date = "2025-09-2100:07:462025-10-2716:36:242025-10-2814:44:10"
    collapsed = [{"交易日期": collapsed_date, "金额": "0.04", "方向": "收入", "余额": "306.09"} for _ in range(3)]

    candidate = _candidate_from_batch(
        candidate_id="evidence_atom",
        transactions=collapsed,
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="canonical_evidence_table",
        expected_rows=RowCountEvidence(count=3, source="positioned_date_anchors", confidence=0.80),
        extraction_confidence=0.90,
    )

    assert candidate.canonical_rows == 0
    assert candidate.canonical_coverage == 0.0


def test_candidate_scoring_preserves_distinct_rows_with_same_business_values() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "amount_cny", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
    )
    repeated_business_rows = [
        {"交易日期": "2025-09-21", "金额": "0.04", "方向": "收入", "余额": "306.09"},
        {"交易日期": "2025-09-21", "金额": "0.04", "方向": "收入", "余额": "306.09"},
    ]

    candidate = _candidate_from_batch(
        candidate_id="native_wide_table",
        transactions=repeated_business_rows,
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="native_wide_table",
        expected_rows=RowCountEvidence(count=2, source="native_wide_rows", confidence=0.70),
        extraction_confidence=0.85,
    )

    assert candidate.canonical_rows == 2
    assert candidate.canonical_coverage == 1.0


def _normalized_candidate_row(raw: dict) -> dict:
    direction = "income" if raw["方向"] == "收入" else "expense"
    return {
        "date": raw["交易日期"][:10],
        "amount": float(raw["金额"]),
        "amount_cny": float(raw["金额"]),
        "direction": direction,
        "balance": float(raw["余额"]),
    }


def test_geometry_direction_repair_uses_summary_and_cross_page_balance() -> None:
    columns = {"direction": 0, "amount": 1, "balance": 2, "summary": 3}
    first_page = [["转账", "198.87", "144.74", "出账网联"]]
    second_page = [["转账", "3,000.00", "36,914.33", "微信转账"]]

    previous_balance = _repair_geometry_rows(first_page, columns)
    _repair_geometry_rows(second_page, columns, previous_balance=39_914.33)

    assert previous_balance == 144.74
    assert first_page[0][0] == "支出"
    assert second_page[0][0] == "支出"


def test_geometry_direction_repair_corrects_explicit_direction_when_balance_uniquely_disagrees() -> None:
    columns = {"direction": 0, "amount": 1, "balance": 2, "summary": 3}
    rows = [["收入", "2.00", "3,641.74", "短信收费"]]

    _repair_geometry_rows(rows, columns, previous_balance=3_643.74)

    assert rows[0][0] == "支出"


def test_geometry_atom_join_preserves_visual_account_order_across_font_baselines() -> None:
    atoms = [
        _atom("prefix", "00000000000000", 340.0, 160.2, 380.0),
        _atom("suffix", "864", 380.0, 160.0, 400.0),
    ]

    assert _join_geometry_atoms(atoms, line_tolerance=1.5) == "00000000000000864"


def test_geometry_footer_recognizes_issuer_important_notice() -> None:
    assert _is_geometry_footer_text("重要提示：请仔细核对账户余额，客服电话：95588") is True


def test_geometry_recovery_prefers_sequence_spine_and_repairs_glued_cells() -> None:
    atoms = [
        _atom("hs", "序号", 20.0, 80.0, 40.0),
        _atom("hd", "记账日期", 55.0, 80.0, 100.0),
        _atom("ha", "交易金额", 145.0, 80.0, 200.0),
        _atom("hb", "账户余额", 225.0, 80.0, 280.0),
        _atom("hm", "摘要描述", 300.0, 80.0, 370.0),
        _atom("hp", "对方户名", 430.0, 80.0, 500.0),
        _atom("s1", "1", 25.0, 110.0, 32.0),
        _atom("d1", "2025-01-02", 55.0, 110.0, 100.0),
        _atom("a1", "-10.00", 160.0, 110.0, 195.0),
        _atom("b1", "90.00税费社保", 230.0, 110.0, 330.0),
        _atom("p1", "待报解预算收入", 430.0, 110.0, 500.0),
        _atom("s2", "2", 25.0, 130.0, 32.0),
        _atom("d2", "2025-01--03", 55.0, 130.0, 105.0),
        _atom("a2", "5.00", 165.0, 130.0, 195.0),
        _atom("b2", "85.00电子银行转账", 230.0, 130.0, 335.0),
        _atom("p2", "测试公司", 430.0, 130.0, 480.0),
        _atom("s3", "3", 25.0, 150.0, 32.0),
        _atom("d3", "2025-01-04", 55.0, 150.0, 100.0),
        _atom("a3", "10.00", 165.0, 150.0, 195.0),
        _atom("b3", "75.00", 230.0, 150.0, 275.0),
        _atom("m3", "0网银路行互联", 300.0, 150.0, 370.0),
        _atom("p3", "另一公司", 430.0, 150.0, 480.0),
        _atom("footer-in", "收入总额:0.00", 25.0, 175.0, 120.0),
        _atom("footer-out", "支出总额:25.00", 230.0, 175.0, 330.0),
        _atom("print-date", "打印日期:2025-02-01", 55.0, 190.0, 180.0),
    ]
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert recovered_evidence_atom_expected_row_count(parse_result) == 3
    assert recovered_evidence_atom_expected_row_evidence(parse_result) == (3, "page_transaction_anchors", 0.97)
    assert len(tables) == 1
    assert len(tables[0]) == 4
    assert tables[0][1] == ["1", "2025-01-02", "-10.00", "90.00", "税费社保", "待报解预算收入"]
    assert tables[0][2] == ["2", "2025-01-03", "-5.00", "85.00", "电子银行转账", "测试公司"]
    assert tables[0][3] == ["3", "2025-01-04", "-10.00", "75.00", "网银跨行互联", "另一公司"]
    assert all(source["row_anchor_type"] == "sequence" for source in sources)
    assert any(source.get("reconstruction_repairs") for source in sources)


def test_candidate_selection_rejects_balance_chain_weaker_near_tie():
    selected, _diagnostics = _select_candidate(
        [
            _candidate("evidence_atom", score=0.91, balance_chain_score=1.0),
            _candidate("fallback", score=0.87, balance_chain_score=0.0),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "evidence_atom"


def test_candidate_selection_preserves_richer_equal_quality_source_columns():
    selected, diagnostics = _select_candidate(
        [
            _candidate("legacy_primary", source_column_width=8.0),
            _candidate(
                "evidence_atom",
                expected_rows=RowCountEvidence(count=3, source="positioned_date_anchors", confidence=0.95),
                extraction_confidence=0.95,
                source_column_width=7.0,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "legacy_primary"
    assert diagnostics["selected_candidate"] == "legacy_primary"


def test_candidate_derived_count_cannot_replace_full_native_candidate():
    selected, _diagnostics = _select_candidate(
        [
            _candidate("legacy_primary", rows=4),
            _candidate(
                "evidence_atom",
                rows=3,
                expected_rows=RowCountEvidence(count=3, source="positioned_date_anchors", confidence=0.95),
                extraction_confidence=0.95,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "legacy_primary"


def test_candidate_selection_penalizes_rows_above_independent_total():
    evidence = RowCountEvidence(count=10, source="header_total", confidence=0.94)
    selected, _diagnostics = _select_candidate(
        [
            _candidate("exact", rows=10, expected_rows=evidence),
            _candidate("over_extracted", rows=12, expected_rows=evidence),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "exact"


def test_recovers_complete_split_debit_credit_rows():
    atoms = [
        _atom("hs", "序号", 25.0, 80.0, 41.0),
        _atom("hd", "交易日期", 52.0, 80.0, 85.0),
        _atom("hr", "交易流水号", 108.0, 80.0, 149.0),
        _atom("hm", "支出（元）收入（元）账户余额（元）", 165.0, 80.0, 312.0),
        _atom("ha", "对方账号", 339.0, 80.0, 372.0),
        _atom("hp", "对方户名", 428.0, 80.0, 461.0),
        _atom("hn", "对方行号", 505.0, 80.0, 538.0),
        _atom("hbn", "对方行名", 563.0, 80.0, 596.0),
        _atom("hc", "交易渠道", 613.0, 80.0, 646.0),
        _atom("hpu", "用途", 698.0, 80.0, 715.0),
        _atom("hsu", "摘要", 782.0, 80.0, 799.0),
        _atom("s1", "1", 30.0, 110.0, 34.0),
        _atom("d1", "20260102", 49.0, 110.0, 88.0),
        _atom("r1", "REF001", 108.0, 110.0, 145.0),
        _atom("e1", "12.34", 181.0, 110.0, 205.2),
        _atom("b1", "100.00", 281.0, 110.0, 313.1),
        _atom("a1", "1234567890", 340.0, 110.0, 390.0),
        _atom("p1", "甲公司", 428.0, 110.0, 460.0),
        _atom("bn1", "BANK001", 505.0, 110.0, 535.0),
        _atom("bname1", "测试银行", 563.0, 110.0, 600.0),
        _atom("channel1", "网银", 613.0, 110.0, 635.0),
        _atom("purpose1", "货款", 698.0, 110.0, 720.0),
        _atom("summary1", "转账", 782.0, 110.0, 804.0),
        _atom("s2", "2", 30.0, 140.0, 34.0),
        _atom("d2", "20260103", 49.0, 140.0, 88.0),
        _atom("r2", "REF002", 108.0, 140.0, 145.0),
        _atom("i2", "20.00", 224.0, 140.0, 248.3),
        _atom("b2", "120.00", 281.0, 140.0, 313.1),
        _atom("s3", "3", 30.0, 170.0, 34.0),
        _atom("d3", "20260104", 49.0, 170.0, 88.0),
        _atom("r3", "REF003", 108.0, 170.0, 145.0),
        _atom("e3", "1.00", 181.0, 170.0, 205.2),
        _atom("b3", "119.00", 281.0, 170.0, 313.1),
        _atom("s4", "4", 30.0, 200.0, 34.0),
        _atom("d4", "20260105", 49.0, 200.0, 88.0),
        _atom("r4", "REF004", 108.0, 200.0, 145.0),
        _atom("i4", "1.00", 224.0, 200.0, 248.3),
        _atom("b4", "120.00", 281.0, 200.0, 313.1),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][0] == [
        "序号",
        "交易日期",
        "交易流水号",
        "支出金额",
        "收入金额",
        "余额",
        "对方账号",
        "对方户名",
        "对方行号",
        "对方行名",
        "交易渠道",
        "用途",
        "摘要",
    ]
    assert tables[0][1] == [
        "1",
        "20260102",
        "REF001",
        "12.34",
        "",
        "100.00",
        "1234567890",
        "甲公司",
        "BANK001",
        "测试银行",
        "网银",
        "货款",
        "转账",
    ]
    assert tables[0][2][:6] == ["2", "20260103", "REF002", "", "20.00", "120.00"]
    assert len(tables[0]) == 5


def test_rejects_layout_without_complete_issuer_headers():
    atoms = [
        _atom("hd", "交易日期", 52.0, 80.0),
        _atom("d1", "20260102", 49.0, 110.0),
        _atom("a1", "12.34", 181.0, 110.0, 205.2),
    ]

    assert recover_evidence_atom_bank_tables(_result(atoms)) == []


def test_recovers_rotated_column_major_record_blocks_with_single_page_sources():
    atoms = [
        _atom(
            "header",
            "\u5e8f\u53f7\n\u6458\u8981\n\u5e01\u522b\n\u949e\u6c47\n\u4ea4\u6613\u65e5\u671f\n"
            "\u4ea4\u6613\u91d1\u989d\n\u8d26\u6237\u4f59\u989d\n\u4ea4\u6613\u5730\u70b9/\u9644\u8a00\n"
            "\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
            20.0,
            20.0,
            32.0,
        ),
        _atom(
            "row-1",
            "1\n\u94f6\u8054\u5165\u8d26\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n100.00\n100.00\n"
            "\u5546\u6237\n1234567890/\u7532\u516c\u53f8",
            50.0,
            20.0,
            62.0,
        ),
        _atom(
            "row-2",
            "2\n\u8f6c\u8d26\u652f\u53d6\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n-20.00\n80.00\n"
            "\u5546\u6237\n1234567891/\u4e59\u516c\u53f8",
            70.0,
            20.0,
            82.0,
        ),
        _atom(
            "row-3",
            "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n10.00\n90.00\n"
            "\u5546\u6237\n6217002120017593862/\u4e19\u516c\u53f8",
            90.0,
            20.0,
            102.0,
        ),
    ]

    recovery = recover_positioned_record_block_bank_tables(_result(atoms))

    assert recovery.expected_rows == 3
    assert len(recovery.tables) == 1
    assert len(recovery.tables[0]) == 4
    assert recovery.tables[0][1][3:6] == ["", "100.00", "100.00"]
    assert recovery.tables[0][2][3:6] == ["20.00", "", "80.00"]
    assert recovery.tables[0][3][3:6] == ["", "10.00", "90.00"]
    assert [source["source_page"] for source in recovery.row_sources] == [1, 1, 1]
    assert all(source["page_range"] == [1, 1] for source in recovery.row_sources)
    assert recovery.row_sources[0]["source_raw"] == {
        "序号": "1",
        "摘要": "银联入账",
        "币别": "人民币元",
        "钞汇": "钞",
        "交易日期": "20211025",
        "交易金额": "100.00",
        "账户余额": "100.00",
        "交易地点/附言": "商户",
        "对方账号与户名": "1234567890/甲公司",
    }


def test_positioned_recovery_keeps_short_continuation_page():
    atoms: list[dict] = []
    for sequence, page_number in [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2)]:
        atom = _atom(
            f"row-{sequence}",
            f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
            f"6222020202020{sequence:03d}/甲公司",
            20.0,
            20.0 + (sequence - 1) * 20.0,
            60.0,
        )
        atom["page_id"] = f"page:{page_number:04d}"
        atoms.append(atom)

    recovery = recover_positioned_record_block_bank_tables(_result(atoms))

    assert recovery.expected_rows == 5
    assert sum(len(table) - 1 for table in recovery.tables) == 5
    assert [source["source_page"] for source in recovery.row_sources] == [1, 1, 1, 2, 2]


def test_positioned_page_text_uses_actual_source_page_number():
    blocks = [
        SimpleNamespace(
            content=(
                f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
                f"6222020202020{sequence:03d}/甲公司"
            ),
            bbox=[20.0, 20.0 + sequence * 20.0, 60.0, 30.0 + sequence * 20.0],
            evidence_ids=[],
        )
        for sequence in range(1, 4)
    ]
    result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={}),
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=5,
                width=600,
                height=850,
                tables=[],
                texts=blocks,
            )
        ],
        logical_tables=[],
        entities=SimpleNamespace(domain_specific={}),
    )

    recovery = recover_positioned_record_block_bank_tables(result)

    assert [source["source_page"] for source in recovery.row_sources] == [5, 5, 5]
    assert all(source["page_range"] == [5, 5] for source in recovery.row_sources)


def test_geometry_orientation_uses_transaction_columns_instead_of_header_dates():
    atoms = [
        _atom("h-seq", "序号", 20.0, 100.0, 40.0),
        _atom("h-date", "交易日期", 60.0, 100.0, 92.0),
        _atom("h-time", "交易时间", 110.0, 100.0, 142.0),
        _atom("h-type", "交易类型", 160.0, 100.0, 192.0),
        _atom("h-direction", "借贷", 210.0, 100.0, 226.0),
        _atom("h-amount", "交易金额", 250.0, 100.0, 282.0),
        _atom("h-balance", "余额", 320.0, 100.0, 336.0),
        _atom("h-account", "对方账号", 370.0, 100.0, 402.0),
        _atom("h-party", "对方户名", 430.0, 100.0, 462.0),
        _atom("h-channel", "交易地点", 490.0, 100.0, 522.0),
        _atom("h-summary", "摘要", 550.0, 100.0, 566.0),
    ]
    for index in range(6):
        atoms.append(_atom(f"query-date-{index}", f"2024-01-{index + 1:02d}", 300.0, 20.0 + index * 10.0, 340.0))
    atoms.extend(
        [
            _atom("r1-seq", "1", 25.0, 130.0, 29.0),
            _atom("r1-date", "2022-08-05", 60.0, 130.0, 100.0),
            _atom("r1-time", "14:05:18", 110.0, 130.0, 142.0),
            _atom("r1-type", "跨行汇款", 160.0, 130.0, 192.0),
            _atom("r1-direction", "贷 Cr", 210.0, 130.0, 230.0),
            _atom("r1-amount", "40.00", 250.0, 130.0, 272.0),
            _atom("r1-balance", "41.06", 320.0, 130.0, 342.0),
            _atom("r1-account", "6214857212810271", 370.0, 130.0, 426.0),
            _atom("r1-party", "周深", 440.0, 130.0, 456.0),
            _atom("r1-channel", "网上银行", 490.0, 130.0, 522.0),
            _atom("r1-summary", "转账", 550.0, 130.0, 566.0),
            _atom("r2-seq", "2", 25.0, 150.0, 29.0),
            _atom("r2-date", "2022-08-06", 60.0, 150.0, 100.0),
            _atom("r2-time", "16:14:05", 110.0, 150.0, 142.0),
            _atom("r2-type", "网上支付", 160.0, 150.0, 192.0),
            _atom("r2-direction", "借 Dr", 210.0, 150.0, 230.0),
            _atom("r2-amount", "37.98", 250.0, 150.0, 272.0),
            _atom("r2-balance", "3.08", 320.0, 150.0, 338.0),
            _atom("r2-account", "301440373999502", 370.0, 150.0, 426.0),
            _atom("r2-party", "江苏欧飞电子商务有限公司", 430.0, 150.0, 482.0),
            _atom("r2-channel", "网上支付", 490.0, 150.0, 522.0),
            _atom("r2-summary", "消费", 550.0, 150.0, 566.0),
        ]
    )
    result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(result)

    assert recovered_evidence_atom_expected_row_count(result) == 2
    assert len(tables) == 1
    assert tables[0][1][0:7] == ["1", "2022-08-05", "14:05:18", "跨行汇款", "贷 Cr", "40.00", "41.06"]
    assert tables[0][2][0:7] == ["2", "2022-08-06", "16:14:05", "网上支付", "借 Dr", "37.98", "3.08"]
    assert [source["source_page"] for source in recovered_evidence_atom_row_sources(result)] == [1, 1]


def test_positioned_recovery_uses_evidence_atoms_for_pages_without_text_blocks():
    page_one_blocks = [
        SimpleNamespace(
            content=(
                f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
                f"6222020202020{sequence:03d}/甲公司"
            ),
            bbox=[20.0, 20.0 + sequence * 20.0, 60.0, 30.0 + sequence * 20.0],
            evidence_ids=[],
        )
        for sequence in range(1, 4)
    ]
    page_two_atoms = []
    for sequence in range(4, 6):
        atom = _atom(
            f"row-{sequence}",
            f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
            f"6222020202020{sequence:03d}/甲公司",
            20.0,
            20.0 + sequence * 20.0,
            60.0,
        )
        atom["page_id"] = "page:0002"
        page_two_atoms.append(atom)
    result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={"text_atoms": page_two_atoms}),
        pages=[
            SimpleNamespace(page_number=1, width=600, height=850, tables=[], texts=page_one_blocks),
            SimpleNamespace(page_number=2, width=600, height=850, tables=[], texts=[]),
        ],
        logical_tables=[],
        entities=SimpleNamespace(domain_specific={}),
    )

    recovery = recover_positioned_record_block_bank_tables(result)

    assert recovery.expected_rows == 5
    assert [source["source_page"] for source in recovery.row_sources] == [1, 1, 1, 2, 2]


def test_recovers_column_aggregate_table_from_positioned_record_spines():
    atoms = [
        _atom(
            "header",
            "\u5e8f\u53f7\n\u6458\u8981\n\u5e01\u522b\n\u949e\u6c47\n\u4ea4\u6613\u65e5\u671f",
            20.0,
            20.0,
            32.0,
        ),
        _atom("row-1", "1\n\u94f6\u8054\u5165\u8d26\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025", 50.0, 20.0, 62.0),
        _atom("row-2", "2\n\u8f6c\u8d26\u652f\u53d6\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025", 70.0, 20.0, 82.0),
        _atom(
            "row-3",
            "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025",
            90.0,
            20.0,
            102.0,
        ),
    ]
    result = _result(atoms)
    result.pages[0].tables = [
        SimpleNamespace(
            table_id="aggregate",
            confidence=1.0,
            row_count=1,
            headers=[
                "\u5e8f\u53f7",
                "\u6458\u8981",
                "\u4ea4\u6613\u65e5\u671f",
                "\u4ea4\u6613\u91d1\u989d",
                "\u8d26\u6237\u4f59\u989d",
                "\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
            ],
            rows=[
                SimpleNamespace(
                    source_page=1,
                    cells=[
                        SimpleNamespace(text=value)
                        for value in [
                            "1\n2\n3",
                            "\u94f6\u8054\u5165\u8d26\n\u8f6c\u8d26\u652f\u53d6\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0",
                            "20211025\n20211025\n20211025",
                            "100.00\n-20.00\n10.00",
                            "100.00\n80.00\n90.00",
                            "1234567890/\u7532\u516c\u53f8\n1234567891/\u4e59\u516c\u53f8\n6217002120017593862/\u4e19\u516c\u53f8",
                        ]
                    ],
                )
            ],
        )
    ]

    recovery = recover_positioned_record_block_bank_tables(result)

    assert recovery.expected_rows == 3
    assert recovery.tables[0][1][3:6] == ["", "100.00", "100.00"]
    assert recovery.tables[0][2][3:6] == ["20.00", "", "80.00"]
    assert recovery.tables[0][3][3:6] == ["", "10.00", "90.00"]
    assert all(source["page_range"] == [1, 1] for source in recovery.row_sources)
    assert recovery.row_sources[0]["source_raw"] == {
        "序号": "1",
        "摘要": "银联入账",
        "交易日期": "20211025",
        "交易金额": "100.00",
        "账户余额": "100.00",
        "对方账号与户名": "1234567890/甲公司",
    }


def test_column_aggregate_source_raw_recovers_counterparty_after_blank_cell():
    row_atoms = [
        _atom("date", "20220128", 0.0, 10.0, 40.0),
        _atom("amount", "-4,515.48", 50.0, 10.0, 85.0),
        _atom("balance", "1,172.52", 95.0, 10.0, 130.0),
        _atom("counterparty", "6214921500056813/黄说英", 160.0, 10.0, 270.0),
    ]
    raw = _column_aggregate_source_raw(
        None,
        "page:0001",
        [{"atom": _atom("spine", "1\n摘要\n20220128", 0.0, 10.0, 20.0)}],
        0,
        ["交易日期", "交易金额", "账户余额", "对方账号与户名"],
        {"date": ["20220128"], "amount": ["-4,515.48"], "balance": ["1,172.52"]},
        row_atoms=row_atoms,
        column_axis=0,
    )

    assert raw["对方账号与户名"] == "6214921500056813/黄说英"


def test_positioned_record_direction_uses_continuous_previous_page_balance():
    previous = {"sequence_no": 285, "amount": "3021.00", "balance": "25733.42"}
    records = [
        {
            "sequence_no": 286,
            "amount": "3826.07",
            "balance": "29559.49",
            "direction": "",
        }
    ]

    _infer_positioned_block_directions(records, preceding_record=previous)

    assert records[0]["direction"] == "income"


def test_positioned_record_direction_rejects_non_contiguous_balance_pair():
    records = [
        {"sequence_no": 1, "amount": "10.00", "balance": "100.00", "direction": ""},
        {"sequence_no": 3, "amount": "10.00", "balance": "110.00", "direction": ""},
    ]

    _infer_positioned_block_directions(records)

    assert records[1]["direction"] == ""


def test_positioned_counter_account_prefers_account_joined_to_party_name():
    text = "\n".join(
        [
            "1",
            "转账",
            "20250101",
            "100.00",
            "900.00",
            "123456789012",
            "6222020202020202/甲公司",
        ]
    )

    assert _positioned_block_counter_account(text) == "6222020202020202"


def test_positioned_counter_account_preserves_masked_source_value():
    text = "\n".join(["1", "转账", "20250101", "-10.00", "90.00", "6230****6516/甲公司"])

    assert _positioned_block_counter_account(text) == "6230****6516"


def test_positioned_record_sort_preserves_descending_source_order():
    records = [
        {"sequence_no": 3, "atom": _atom("r3", "3", 10.0, 10.0)},
        {"sequence_no": 2, "atom": _atom("r2", "2", 10.0, 30.0)},
        {"sequence_no": 1, "atom": _atom("r1", "1", 10.0, 50.0)},
    ]

    _sort_positioned_block_records(records)

    assert [record["sequence_no"] for record in records] == [3, 2, 1]


def test_registry_selects_column_aggregate_recovery_when_physical_table_is_collapsed():
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
    from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry

    atoms = [
        _atom("header", "\u5e8f\u53f7\n\u6458\u8981\n\u4ea4\u6613\u65e5\u671f", 20.0, 20.0, 32.0),
        _atom("row-1", "1\n\u94f6\u8054\u5165\u8d26\n20211025", 50.0, 20.0, 62.0),
        _atom("row-2", "2\n\u8f6c\u8d26\u652f\u53d6\n20211025", 70.0, 20.0, 82.0),
        _atom("row-3", "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n20211025", 90.0, 20.0, 102.0),
    ]
    result = _result(atoms)
    result.pages[0].tables = [
        TableBlock(
            table_id="aggregate-registry",
            confidence=1.0,
            headers=[
                "\u5e8f\u53f7",
                "\u6458\u8981",
                "\u4ea4\u6613\u65e5\u671f",
                "\u4ea4\u6613\u91d1\u989d",
                "\u8d26\u6237\u4f59\u989d",
            ],
            rows=[
                TableRow(
                    source_page=1,
                    cells=[
                        CellValue(text=value)
                        for value in [
                            "1\n2\n3",
                            "\u94f6\u8054\u5165\u8d26\n\u8f6c\u8d26\u652f\u53d6\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0",
                            "20211025\n20211025\n20211025",
                            "100.00\n-20.00\n10.00",
                            "100.00\n80.00\n90.00",
                        ]
                    ],
                )
            ],
        )
    ]
    ctx = StyleContext(
        tables=[],
        full_text="\u4e2a\u4eba\u8d26\u6237\u4ea4\u6613\u660e\u7ec6",
        institution=None,
        page_count=1,
        parse_result=result,
    )
    registry = BankStyleParserRegistry()

    records, _identity = registry.run(BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin())

    assert len(records) == 3
    assert registry.last_selection_diagnostics["selected_candidate"] == "positioned_record_block"
    assert all(record["source"]["page_range"] == [1, 1] for record in records)


def test_registry_prefers_positioned_record_blocks_over_collapsed_physical_row():
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
    from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry

    atoms = [
        _atom(
            "header",
            "\u5e8f\u53f7\n\u6458\u8981\n\u5e01\u522b\n\u949e\u6c47\n\u4ea4\u6613\u65e5\u671f\n"
            "\u4ea4\u6613\u91d1\u989d\n\u8d26\u6237\u4f59\u989d\n\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
            20.0,
            20.0,
            32.0,
        ),
        _atom(
            "row-1",
            "1\n\u94f6\u8054\u5165\u8d26\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n100.00\n100.00\n"
            "\u5546\u6237\n1234567890/\u7532\u516c\u53f8",
            50.0,
            20.0,
            62.0,
        ),
        _atom(
            "row-2",
            "2\n\u8f6c\u8d26\u652f\u53d6\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n-20.00\n80.00\n"
            "\u5546\u6237\n1234567891/\u4e59\u516c\u53f8",
            70.0,
            20.0,
            82.0,
        ),
        _atom(
            "row-3",
            "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n10.00\n90.00\n"
            "\u5546\u6237\n1234567892/\u4e19\u516c\u53f8",
            90.0,
            20.0,
            102.0,
        ),
    ]
    parse_result = _result(atoms)
    collapsed = [
        [
            "\u5e8f\u53f7",
            "\u6458\u8981",
            "\u5e01\u522b",
            "\u949e\u6c47",
            "\u4ea4\u6613\u65e5\u671f",
            "\u4ea4\u6613\u91d1\u989d",
            "\u8d26\u6237\u4f59\u989d",
            "\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
        ],
        [
            "123",
            "\u94f6\u8054\u5165\u8d26\u8f6c\u8d26\u652f\u53d6\u652f\u4ed8\u673a\u6784\u63d0\u73b0",
            "\u4eba\u6c11\u5e01\u5143" * 3,
            "\u949e" * 3,
            "20211025" * 3,
            "100.00-20.0010.00",
            "100.0080.0090.00",
            "1234567890/\u7532\u516c\u53f8",
        ],
    ]
    ctx = StyleContext(
        tables=[collapsed],
        full_text="\u4e2a\u4eba\u8d26\u6237\u4ea4\u6613\u660e\u7ec6",
        institution=None,
        page_count=1,
        parse_result=parse_result,
    )
    registry = BankStyleParserRegistry()
    records, _ = registry.run(BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin())

    assert len(records) == 3
    assert registry.last_selection_diagnostics["selected_candidate"] == "positioned_record_block"
    assert all(record["source"]["page_range"] == [1, 1] for record in records)


def test_recovers_borderless_date_anchored_split_columns():
    atoms = [
        _atom("ht", "交易时间", 22.0, 80.0, 50.0),
        _atom("hs", "摘要", 61.0, 80.0, 75.0),
        _atom("hd", "借方发生额", 311.0, 80.0, 346.0),
        _atom("hc", "贷方发生额", 385.0, 80.0, 420.0),
        _atom("hb", "账户余额流水号", 466.0, 80.0, 519.0),
        _atom("d1", "2025/01/03", 22.0, 110.0, 57.0),
        _atom("t1", "16:18:35", 22.0, 119.0, 50.0),
        _atom("s1", "个人所得税", 61.0, 110.0, 100.0),
        _atom("e1", "15.00", 329.0, 110.0, 346.0),
        _atom("c1", "0.00", 403.0, 110.0, 420.0),
        _atom("b1", "363,693.02", 459.0, 110.0, 494.0),
        _atom("r1", "5542025010300824", 466.0, 119.0, 530.0),
        _atom("d2", "2025/01/07", 22.0, 140.0, 57.0),
        _atom("s2", "社保费", 61.0, 140.0, 90.0),
        _atom("e2", "113.16", 329.0, 140.0, 346.0),
        _atom("c2", "0.00", 403.0, 140.0, 420.0),
        _atom("b2", "363,579.86", 459.0, 140.0, 494.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][0] == ["交易时间", "摘要", "借方发生额", "贷方发生额", "账户余额流水号"]
    assert len(tables[0]) == 3
    assert tables[0][1][0] == "2025/01/0316:18:35"
    assert tables[0][1][4].startswith("363,693.02")


def test_recovers_parenthetical_split_columns_and_glued_balance_summary_header():
    atoms = [
        _atom("hd", "交易日期", 17.0, 106.0, 53.0),
        _atom("hdebit", "借方(出账)", 107.0, 106.0, 152.0),
        _atom("hcredit", "贷方(入账)", 193.0, 106.0, 238.0),
        _atom("hbalance_summary", "余额摘要", 305.0, 106.0, 344.0),
        _atom("hparty", "收(付)方名称", 394.0, 106.0, 449.0),
        _atom("haccount", "收(付)方账号", 463.0, 106.0, 517.0),
        _atom("htype", "交易类型", 531.0, 106.0, 568.0),
        _atom("d1", "2025-01-02", 17.0, 124.0, 62.0),
        _atom("debit1", "25.00", 129.0, 129.0, 152.0),
        _atom("balance1", "5,000,888.02", 269.0, 129.0, 324.0),
        _atom("summary1", "服务费", 326.0, 124.0, 370.0),
        _atom("party1", "测试有限公司", 394.0, 124.0, 449.0),
        _atom("account1", "123917394110001", 463.0, 124.0, 527.0),
        _atom("type1", "对公转账", 531.0, 124.0, 568.0),
        _atom("d2", "2025-01-03", 17.0, 156.0, 62.0),
        _atom("credit2", "200.00", 210.0, 161.0, 238.0),
        _atom("balance2", "5,001,088.02", 269.0, 161.0, 324.0),
        _atom("summary2", "往来款", 326.0, 156.0, 370.0),
        _atom("party2", "第二有限公司", 394.0, 156.0, 449.0),
        _atom("account2", "123917394110002", 463.0, 156.0, 527.0),
        _atom("type2", "提回收款", 531.0, 156.0, 568.0),
    ]
    vector_atoms = [
        {"id": f"rule-{index}", "page_id": "page:0001", "bbox": [17.0, y, 568.0, y]}
        for index, y in enumerate((116.0, 149.0, 182.0), start=1)
    ]
    parse_result = _result(atoms, vector_atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert tables[0][0] == [
        "交易日期",
        "借方(出账)",
        "贷方(入账)",
        "余额",
        "摘要",
        "收(付)方名称",
        "收(付)方账号",
        "交易类型",
    ]
    assert tables[0][1] == [
        "2025-01-02",
        "25.00",
        "",
        "5,000,888.02",
        "服务费",
        "测试有限公司",
        "123917394110001",
        "对公转账",
    ]
    assert tables[0][2][2:6] == ["200.00", "5,001,088.02", "往来款", "第二有限公司"]
    assert recovered_evidence_atom_expected_row_count(parse_result) == 2
    assert [source["source_page"] for source in sources] == [1, 1]


def test_recovers_borderless_embedded_direction_amount_rows():
    atoms = [
        _atom("hd", "交易日期", 36.0, 80.0, 72.0),
        _atom("hbd", "记账日期", 87.0, 80.0, 123.0),
        _atom("hs", "摘要", 138.0, 80.0, 156.0),
        _atom("ha", "支/收交易金额", 178.0, 80.0, 238.0),
        _atom("hb", "账户余额", 246.0, 80.0, 282.0),
        _atom("d1", "2023-10-02", 36.0, 110.0, 81.0),
        _atom("bd1", "2023-10-02", 87.0, 110.0, 132.0),
        _atom("s1", "跨行代付收", 138.0, 110.0, 187.0),
        _atom("a1", "23,903.69", 202.0, 110.0, 241.0),
        _atom("b1", "23,903.69", 246.0, 110.0, 286.0),
        _atom("d2", "2023-10-07", 36.0, 140.0, 81.0),
        _atom("bd2", "2023-10-07", 87.0, 140.0, 132.0),
        _atom("s2", "跨行代付收", 138.0, 140.0, 187.0),
        _atom("a2", "13,610.09", 202.0, 140.0, 241.0),
        _atom("b2", "13,610.09", 246.0, 140.0, 286.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][1][2] == "跨行代付收"
    assert tables[0][1][3] == "23,903.69"


def test_recovers_staggered_multilingual_borderless_header():
    atoms = [
        _atom("hd", "交易日期", 75.0, 145.7, 107.0),
        _atom("ht", "交易时间", 120.0, 145.7, 152.0),
        _atom("hy", "交易类型", 165.0, 145.7, 197.0),
        _atom("hf", "借贷", 220.0, 145.7, 236.0),
        _atom("ha", "交易金额", 258.0, 145.7, 290.0),
        _atom("hb", "余额", 337.0, 145.7, 353.0),
        _atom("hp", "交易地点", 583.0, 145.7, 615.0),
        _atom("hm", "摘要", 667.0, 145.7, 683.0),
        _atom("hs", "序号", 51.0, 150.2, 67.0),
        _atom("hc", "对方账号", 411.0, 150.2, 443.0),
        _atom("hn", "对方户名", 496.0, 150.2, 528.0),
        _atom("s1", "1", 56.5, 164.6, 60.5),
        _atom("d1", "2022-08-05", 75.0, 164.6, 115.0),
        _atom("t1", "14:05:18", 120.0, 164.6, 152.0),
        _atom("y1", "跨行汇款", 165.0, 164.6, 197.0),
        _atom("f1", "贷", 220.0, 164.6, 228.0),
        _atom("a1", "40.00", 253.0, 164.6, 273.0),
        _atom("b1", "41.06", 332.0, 164.6, 352.0),
        _atom("c1", "6214857212810271", 412.0, 164.6, 476.0),
        _atom("n1", "周深", 497.0, 164.6, 513.0),
        _atom("p1", "网上银行", 581.0, 164.6, 613.0),
        _atom("m1", "转账", 667.0, 164.6, 683.0),
        _atom("s2", "2", 56.5, 181.1, 60.5),
        _atom("d2", "2022-08-06", 75.0, 181.1, 115.0),
        _atom("t2", "16:14:05", 120.0, 181.1, 152.0),
        _atom("y2", "网上支付", 165.0, 181.1, 197.0),
        _atom("f2", "借", 220.0, 181.1, 228.0),
        _atom("a2", "37.98", 253.0, 181.1, 273.0),
        _atom("b2", "3.08", 332.0, 181.1, 348.0),
        _atom("c2", "301440373999502", 412.0, 181.1, 472.0),
        _atom("n2", "测试公司", 497.0, 181.1, 529.0),
        _atom("done", "打印完毕", 412.0, 195.6, 444.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][1][:7] == [
        "1",
        "2022-08-05",
        "14:05:18",
        "跨行汇款",
        "贷",
        "40.00",
        "41.06",
    ]
    assert tables[0][2][0:2] == ["2", "2022-08-06"]


def test_recovers_mixed_page_orientations_and_preserves_source_geometry():
    def page_atoms(page_id: str, sequence: str, date: str, amount: str, balance: str, y: float) -> list[dict]:
        values = [
            ("序号", 20.0, 50.0),
            ("交易日期", 60.0, 110.0),
            ("收入/支出", 120.0, 180.0),
            ("交易金额", 190.0, 250.0),
            ("账户余额", 260.0, 320.0),
            ("对方账号", 330.0, 410.0),
            ("对方户名", 420.0, 500.0),
            ("摘要", 510.0, 560.0),
        ]
        atoms = [
            {
                "id": f"{page_id}:h:{index}",
                "page_id": page_id,
                "text": text,
                "bbox": [x0, 80.0, x1, 90.0],
            }
            for index, (text, x0, x1) in enumerate(values)
        ]
        row = [
            (sequence, 20.0, 50.0),
            (date, 60.0, 110.0),
            ("支出", 120.0, 180.0),
            (amount, 190.0, 250.0),
            (balance, 260.0, 320.0),
            ("1000050001", 330.0, 410.0),
            ("测试对手", 420.0, 500.0),
            ("转账", 510.0, 560.0),
        ]
        atoms.extend(
            {
                "id": f"{page_id}:r:{index}",
                "page_id": page_id,
                "text": text,
                "bbox": [x0, y, x1, y + 10.0],
            }
            for index, (text, x0, x1) in enumerate(row)
        )
        atoms.append(
            {
                "id": f"{page_id}:footer",
                "page_id": page_id,
                "text": "本页合计及打印信息",
                "bbox": [20.0, y + 25.0, 300.0, y + 35.0],
            }
        )
        return atoms

    page_one = page_atoms("page:0001", "1", "2023-01-01", "10.00", "90.00", 110.0)
    page_two_horizontal = page_atoms("page:0002", "2", "2023-01-02", "20.00", "70.00", 110.0)
    page_two = [_rotated_90(atom, page_id="page:0002") for atom in page_two_horizontal]
    parse_result = _result([*page_one, *page_two])

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert sum(len(table) - 1 for table in tables) == 2
    assert recovered_evidence_atom_expected_row_count(parse_result) == 2
    assert [source["source_page"] for source in sources] == [1, 2]
    assert all(source.get("bbox") and source.get("evidence_ids") for source in sources)
    assert sources[1]["bbox"][0] > 400.0
    assert all("本页合计" not in "".join(row) for table in tables for row in table[1:])


def test_uses_positioned_page_text_when_evidence_atoms_are_not_promoted():
    atoms = [
        _atom("h1", "序号", 20.0, 80.0, 50.0),
        _atom("h2", "交易日期", 60.0, 80.0, 110.0),
        _atom("h3", "收入/支出", 120.0, 80.0, 180.0),
        _atom("h4", "交易金额", 190.0, 80.0, 250.0),
        _atom("h5", "账户余额", 260.0, 80.0, 320.0),
        _atom("r1", "1", 20.0, 110.0, 50.0),
        _atom("r2", "2023-01-01", 60.0, 110.0, 110.0),
        _atom("r3", "支出", 120.0, 110.0, 180.0),
        _atom("r4", "10.00", 190.0, 110.0, 250.0),
        _atom("r5", "90.00", 260.0, 110.0, 320.0),
    ]
    page = SimpleNamespace(
        page_number=1,
        width=600,
        height=850,
        texts=[
            SimpleNamespace(
                content=atom["text"],
                bbox=atom["bbox"],
                evidence_ids=[atom["id"]],
            )
            for atom in atoms
        ],
    )
    parse_result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={}),
        pages=[page],
        entities=SimpleNamespace(domain_specific={}),
    )

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert sum(len(table) - 1 for table in tables) == 1
    assert recovered_evidence_atom_expected_row_count(parse_result) == 1
    assert sources[0]["source_page"] == 1
    assert "r2" in sources[0]["evidence_ids"]


def test_dedupe_uses_bank_reference_before_lossy_business_fields():
    base = {"normalized": {"date": "2026-01-02", "amount": 100.0, "balance": 200.0, "counter_party": "甲"}}
    records = [
        {**base, "raw": {"交易流水号": "REF001"}},
        {**base, "raw": {"交易流水号": "REF002"}},
        {**base, "raw": {"交易流水号": "REF002"}},
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 2


def test_dedupe_preserves_distinct_rows_that_share_a_bank_reference():
    records = [
        {
            "normalized": {
                "date": "2026-01-02",
                "direction": "expense",
                "amount": 100.0,
                "balance": 900.0,
                "counter_party": "甲",
                "summary": "付款",
            },
            "raw": {"交易流水号": "REF001"},
        },
        {
            "normalized": {
                "date": "2026-01-02",
                "direction": "expense",
                "amount": 0.9,
                "balance": 899.1,
                "counter_party": "",
                "summary": "手续费",
            },
            "raw": {"交易流水号": "REF001"},
        },
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 2


def test_dedupe_keeps_same_business_fields_when_sequence_differs():
    base = {"date": "2026-01-02", "amount": 100.0, "balance": 200.0, "counter_party": "same"}
    records = [
        {"normalized": {**base, "sequence_no": "491"}, "raw": {}},
        {"normalized": {**base, "sequence_no": "638"}, "raw": {}},
        {"normalized": {**base, "sequence_no": "638"}, "raw": {}},
    ]

    deduped = dedupe_transaction_rows(records)

    assert [record["normalized"]["sequence_no"] for record in deduped] == ["491", "638"]


def test_recovers_bank_header_title_and_total_row_count_from_evidence_atoms():
    atoms = [
        _atom("title", "测试银行账户交易明细表", 200.0, 10.0, 400.0),
        _atom("bank_label", "开户行", 10.0, 20.0, 50.0),
        _atom("bank_value", "浦发银行重庆分行营业部", 80.0, 18.0, 220.0),
        _atom("print", "打印日期：2026-07-18", 10.0, 30.0, 150.0),
        _atom("period", "交易时段：2026-01-01 至 2026-06-30", 10.0, 45.0, 260.0),
        _atom("holder", "户名：测试用户", 10.0, 60.0, 100.0),
        _atom("account", "账号：1234567890", 110.0, 60.0, 230.0),
        _atom("currency", "币种：人民币", 240.0, 60.0, 320.0),
        _atom("total_label", "汇总交易笔数", 10.0, 220.0, 80.0),
        _atom("total_value", "38笔", 110.0, 225.0, 140.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["statement_title"]["normalized_value"] == "测试银行账户交易明细表"
    assert fields["print_date"]["normalized_value"] == "2026-07-18"
    assert fields["query_period"]["normalized_value"] == "2026-01-01 至 2026-06-30"
    assert fields["total_transactions"]["normalized_value"] == "38"
    assert fields["account_number"]["normalized_value"] == "1234567890"
    assert fields["bank_name"]["normalized_value"] == "浦发银行重庆分行营业部"


def test_evidence_identity_ignores_native_atoms_rejected_by_ocr_fallback():
    atoms = [
        _atom("holder", "户名：上上上上上上", 10.0, 60.0, 120.0),
        _atom("account", "账号：1234567890", 130.0, 60.0, 240.0),
    ]
    for atom in atoms:
        atom["source_kind"] = "pdf_native"
    result = _result(atoms)
    result.parser_info = SimpleNamespace(options={"native_text_ocr_fallback_pages": [1]})

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(result)

    assert fields == {}


def test_evidence_identity_stays_within_selected_source_pages():
    atoms = [
        _atom("selected_holder", "户名：测试科技有限公司 账号：1234567890", 10.0, 60.0, 240.0),
        {
            **_atom("unselected_holder", "户名：错误公司 验证码：", 10.0, 60.0, 240.0),
            "page_id": "page:0002",
        },
    ]
    result = _result(atoms)
    result.evidence_plane.pages = [
        SimpleNamespace(page_id="page:0001", page_number=1),
        SimpleNamespace(page_id="page:0002", page_number=2),
    ]
    result.parser_info = SimpleNamespace(options={"selected_source_pages": [1]})

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(result)

    assert fields["account_holder"]["normalized_value"] == "测试科技有限公司"


def test_evidence_identity_stops_holder_before_account_card_label():
    atoms = [
        _atom("holder", "户名：吴文坤", 10.0, 60.0, 90.0),
        _atom("account_label", "账号/卡号：", 100.0, 60.0, 165.0),
        _atom("account", "6230361108033553943", 175.0, 60.0, 310.0),
        _atom("currency", "币种：人民币", 320.0, 60.0, 400.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_holder"]["normalized_value"] == "吴文坤"
    assert fields["account_number"]["normalized_value"] == "6230361108033553943"


def test_evidence_identity_ignores_account_reference_below_transaction_header() -> None:
    atoms = [
        _atom("hd", "交易日期", 17.0, 106.0, 53.0),
        _atom("hdebit", "借方(出账)", 107.0, 106.0, 152.0),
        _atom("hcredit", "贷方(入账)", 193.0, 106.0, 238.0),
        _atom("hbalance_summary", "余额摘要", 305.0, 106.0, 344.0),
        _atom("hparty", "收(付)方名称", 394.0, 106.0, 449.0),
        _atom("haccount", "收(付)方账号", 463.0, 106.0, 517.0),
        _atom("htype", "交易类型", 531.0, 106.0, 568.0),
        _atom("date", "2025-03-21", 17.0, 130.0, 62.0),
        _atom("summary", "收息，结息账号:999019305110001", 326.0, 130.0, 450.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert "account_number" not in fields


def test_evidence_identity_recovers_split_header_values_and_directional_totals():
    atoms = [
        _atom("title", "交通银行某分行明细对账单", 180.0, 20.0, 390.0),
        _atom("account_label", "账号：", 20.0, 50.0, 55.0),
        _atom("account", "641301106013000859983", 60.0, 50.0, 180.0),
        _atom("holder_label", "户名：", 220.0, 50.0, 255.0),
        _atom("holder", "测试软件有限公司银川分公司", 260.0, 50.0, 430.0),
        _atom("year_label", "年份：", 20.0, 65.0, 55.0),
        _atom("year", "2025", 60.0, 65.0, 90.0),
        _atom("month_label", "月份：", 120.0, 65.0, 155.0),
        _atom("month", "07", 160.0, 65.0, 180.0),
        _atom("currency_label", "币种：", 220.0, 65.0, 255.0),
        _atom("currency", "人民币", 260.0, 65.0, 300.0),
        _atom("carry", "承前", 20.0, 100.0, 50.0),
        _atom("debit_label", "当前账单借方发生数：", 20.0, 220.0, 145.0),
        _atom("debit", "10", 150.0, 220.0, 170.0),
        _atom("credit", "当前账单贷方发生数：1", 300.0, 220.0, 450.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_holder"]["normalized_value"] == "测试软件有限公司银川分公司"
    assert fields["account_number"]["normalized_value"] == "641301106013000859983"
    assert fields["currency"]["raw_value"] == "人民币"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["query_period"]["normalized_value"] == "2025-07-01 至 2025-07-31"
    assert fields["total_transactions"]["normalized_value"] == "11"


def test_evidence_identity_pairs_parallel_label_value_columns_by_geometry():
    atoms = [
        _atom("account_label", "银行账号：", 20.0, 50.0, 80.0),
        _atom("account", "120023710020000001988", 90.0, 50.0, 230.0),
        _atom("currency_label", "币种：", 430.0, 50.0, 470.0),
        _atom("currency", "人民币", 480.0, 50.0, 530.0),
        _atom("holder_label", "账户名称：", 20.0, 70.0, 80.0),
        _atom("holder", "测试信用管理有限公司", 90.0, 70.0, 250.0),
        _atom("deposit_label", "存款种类：", 430.0, 70.0, 490.0),
        _atom("deposit", "单位活期存款", 500.0, 70.0, 580.0),
        _atom("print_bank_label", "打印机构：", 350.0, 780.0, 420.0),
        _atom("print_bank", "富滇银行", 430.0, 780.0, 490.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_number"]["normalized_value"] == "120023710020000001988"
    assert fields["account_holder"]["normalized_value"] == "测试信用管理有限公司"
    assert fields["currency"]["raw_value"] == "人民币"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["bank_name"]["normalized_value"] == "富滇银行"


def test_evidence_identity_supports_hyphenated_account_and_chinese_date_range():
    atoms = [
        _atom("title", "账户明细", 250.0, 20.0, 340.0),
        _atom("account", "账号:31-080201040015288", 20.0, 45.0, 170.0),
        _atom("identity", "户名:测试农业科技有限公司币种:人民币", 190.0, 45.0, 430.0),
        _atom("period_label", "起止日期:", 450.0, 45.0, 510.0),
        _atom("period_start", "2025年11月01日", 520.0, 45.0, 610.0),
        _atom("period_sep", "-", 615.0, 45.0, 620.0),
        _atom("period_end", "2025年12月31日", 625.0, 45.0, 715.0),
        _atom("income_count", "总收入笔数", 20.0, 220.0, 90.0),
        _atom("income_value", "2", 95.0, 220.0, 105.0),
        _atom("expense_count", "总支出笔数", 220.0, 220.0, 290.0),
        _atom("expense_value", "2", 295.0, 220.0, 305.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_number"]["normalized_value"] == "31-080201040015288"
    assert fields["account_holder"]["normalized_value"] == "测试农业科技有限公司"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["query_period"]["normalized_value"] == "2025-11-01 至 2025-12-31"
    assert fields["total_transactions"]["normalized_value"] == "4"


def test_evidence_identity_normalizes_compatibility_currency_without_changing_raw_value():
    atoms = [_atom("currency", "币种：⼈⺠币", 20.0, 45.0, 120.0)]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["currency"]["raw_value"] == "⼈⺠币"
    assert fields["currency"]["normalized_value"] == "CNY"


def test_geometry_recovery_keeps_wrapped_cells_with_preceding_date_and_stops_at_footer():
    atoms = [
        _atom("h1", "交易时间", 20.0, 80.0, 80.0),
        _atom("h2", "收入金额", 100.0, 80.0, 160.0),
        _atom("h3", "支出金额", 180.0, 80.0, 240.0),
        _atom("h4", "账户余额", 260.0, 80.0, 320.0),
        _atom("h5", "对方账号", 340.0, 80.0, 400.0),
        _atom("h6", "对方户名", 420.0, 80.0, 480.0),
        _atom("h7", "对方开户行", 500.0, 80.0, 575.0),
        _atom("h8", "摘要", 600.0, 80.0, 640.0),
        _atom("d1", "2025-11-24", 20.0, 110.0, 80.0),
        _atom("t1", "00:46:17", 20.0, 120.0, 70.0),
        _atom("e1", "100.00", 180.0, 110.0, 235.0),
        _atom("b1", "110.97", 260.0, 110.0, 315.0),
        _atom("a1", "31080243CNYFC0445", 340.0, 110.0, 400.0),
        _atom("p1", "测试银行股份有限公司", 420.0, 110.0, 480.0),
        _atom("bn1a", "测试银行股份有限公司", 500.0, 110.0, 575.0),
        _atom("bn1b", "重庆九龙坡二", 500.0, 125.0, 560.0),
        _atom("bn1c", "郎支行", 500.0, 145.0, 535.0),
        _atom("s1", "批量扣费", 600.0, 110.0, 640.0),
        _atom("d2", "2025-12-21", 20.0, 160.0, 80.0),
        _atom("t2", "01:10:43", 20.0, 170.0, 70.0),
        _atom("i2", "0.03", 100.0, 160.0, 150.0),
        _atom("b2", "111.00", 260.0, 160.0, 315.0),
        _atom("bn2", "第二银行", 500.0, 152.0, 550.0),
        _atom("s2", "批量结息", 600.0, 160.0, 640.0),
        _atom("income_count", "总收入笔数", 20.0, 205.0, 90.0),
        _atom("income_value", "1", 100.0, 205.0, 110.0),
        _atom("income_total", "总收入金额", 180.0, 205.0, 250.0),
        _atom("income_amount", "0.03", 260.0, 205.0, 300.0),
        _atom("print_date", "2026/02/24", 20.0, 390.0, 90.0),
        _atom("page", "第1页/共1页", 280.0, 400.0, 350.0),
    ]

    vector_atoms = [
        {
            "id": f"rule-{index}",
            "page_id": "page:0001",
            "bbox": [0.0, y, 650.0, y],
        }
        for index, y in enumerate((95.0, 150.0, 200.0), start=1)
    ]
    parse_result = _result(atoms, vector_atoms)
    tables = recover_evidence_atom_bank_tables(parse_result)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][1][6] == "测试银行股份有限公司重庆九龙坡二郎支行"
    assert tables[0][2][5:7] == ["", "第二银行"]
    assert all("总收入" not in "".join(row) for row in tables[0][1:])
    assert all("第1页" not in "".join(row) for row in tables[0][1:])
    assert recovered_evidence_atom_expected_row_count(parse_result) == 2


def test_geometry_recovery_assigns_leading_wrapped_cells_to_nearest_date_anchor():
    atoms = [
        _atom("h0", "序号", 10.0, 80.0, 30.0),
        _atom("h1", "交易日期", 40.0, 80.0, 90.0),
        _atom("h2", "交易时间", 100.0, 80.0, 150.0),
        _atom("h3", "交易类型", 160.0, 80.0, 210.0),
        _atom("h4", "借贷", 220.0, 80.0, 250.0),
        _atom("h5", "交易金额", 260.0, 80.0, 320.0),
        _atom("h6", "余额", 330.0, 80.0, 370.0),
        _atom("h7", "对方账号", 380.0, 80.0, 440.0),
        _atom("h8", "对方户名", 450.0, 80.0, 510.0),
        _atom("h9", "摘要", 520.0, 80.0, 560.0),
        _atom("s1", "1", 10.0, 110.0, 20.0),
        _atom("d1", "2022-08-05", 40.0, 110.0, 90.0),
        _atom("t1", "14:05:18", 100.0, 110.0, 150.0),
        _atom("type1", "跨行汇款", 160.0, 110.0, 210.0),
        _atom("dir1", "贷 Cr", 220.0, 110.0, 250.0),
        _atom("a1", "40.00", 260.0, 110.0, 320.0),
        _atom("b1", "41.06", 330.0, 110.0, 370.0),
        _atom("cp1", "周深", 450.0, 110.0, 510.0),
        _atom("s2", "2", 10.0, 140.0, 20.0),
        _atom("d2", "2022-08-06", 40.0, 140.0, 90.0),
        _atom("t2", "16:14:05", 100.0, 140.0, 150.0),
        _atom("type2", "网上支付", 160.0, 140.0, 210.0),
        _atom("dir2", "借 Dr", 220.0, 140.0, 250.0),
        _atom("a2", "37.98", 260.0, 140.0, 320.0),
        _atom("b2", "3.08", 330.0, 140.0, 370.0),
        _atom("cp2a", "江苏欧飞电子商务有限", 450.0, 126.0, 510.0),
        _atom("cp2b", "公司", 450.0, 145.0, 480.0),
        _atom("summary2", "有限公司", 520.0, 145.0, 560.0),
        _atom("footer", "打印完毕", 380.0, 165.0, 440.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][1][4] == "贷 Cr"
    assert tables[0][1][8] == "周深"
    assert tables[0][2][4] == "借 Dr"
    assert tables[0][2][8] == "江苏欧飞电子商务有限公司"


def test_evidence_identity_stops_at_branch_and_ignores_transaction_loan_account():
    atoms = [
        _atom("title", "对公客户账户明细", 200.0, 10.0, 400.0),
        _atom("holder", "客户名称：重庆正大华日软件有限公司", 10.0, 40.0, 220.0),
        _atom("branch", "开户机构：510601", 230.0, 40.0, 340.0),
        _atom("account", "账    号：5106010120010001125", 10.0, 60.0, 230.0),
        _atom("loan", "贷款账号：5101010179730017689", 10.0, 160.0, 230.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_holder"]["normalized_value"] == "重庆正大华日软件有限公司"
    assert fields["account_number"]["normalized_value"] == "5106010120010001125"
