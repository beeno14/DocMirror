# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Logical Table Reconstruction Orchestrator (LTRO)."""

from __future__ import annotations

import pytest

from docmirror.plugins.bank_statement import ltro as ltro_module
from docmirror.plugins.bank_statement.ltro import reconstruct_tables
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence
from tests.unit.test_pipe_text_table_builder import BOC_ROW1, _synthetic_boc_text

SAMPLE_OCR = """
交易明细清单
客户账号：6236030100000354601 客户姓名：于鑫日
交易日期 交易金额月收/支 账户余额 摘要
20220402支出 3.00 1070.13 POS消费
"""


def test_canonical_table_short_circuit():
    mirror = [[["交易日期", "交易金额", "余额"], ["2024-01-01", "1.00", "9.00"]]]
    tables, meta = reconstruct_tables(mirror, "ignored")
    assert meta.source == "canonical_table"
    assert tables == mirror
    assert meta.expected_primary_rows == 1


def test_canonical_header_and_print_footer_do_not_short_circuit():
    mirror = [
        [
            ["对方账户", "借/贷方发生额", "序号", "余额", "传票号", "对方户名", "摘要"],
            ["已打印次数", "1", "打印时间", "2024-01-01", "", "", ""],
        ]
    ]
    tables, meta = reconstruct_tables(mirror, "")
    assert tables == []
    assert meta.source == "none"
    assert meta.expected_primary_rows == 0


def test_canonical_timestamp_header_is_a_valid_date_anchor():
    mirror = [
        [
            ["交易时间", "借方发生额", "贷方发生额", "账户余额流水号"],
            ["2025/01/0316:18:35", "15.00", "0.00", "363,693.0255420250100824712870"],
        ]
    ]
    tables, meta = reconstruct_tables(mirror, "")
    assert tables == mirror
    assert meta.source == "canonical_table"


def test_canonical_split_amount_headers_with_units_are_usable() -> None:
    mirror = [
        [
            [
                "序号",
                "交易日期",
                "支出（元）",
                "收入（元）",
                "账户余额（元）",
                "对方账号",
                "对方户名",
            ],
            [
                "1",
                "2025-01-24\n16:38:19",
                "200000.00",
                "",
                "2369231.13",
                "830100788013000002\n20",
                "重庆中链农科技有限公司",
            ],
        ]
    ]

    tables, meta = reconstruct_tables(mirror, "")

    assert tables == mirror
    assert meta.source == "canonical_table"
    assert meta.expected_primary_rows == 1


def test_canonical_stacked_debit_credit_headers_are_usable() -> None:
    mirror = [
        [
            [
                "交易日期\nTransaction Date",
                "交易流水号\nTeller's Serial Number",
                "发生额\nTransaction Amount",
                "",
                "账户余额\nAccount Balance",
                "交易对手信息\nCounterparty Information",
                "",
                "摘要代码\nAbstract Code",
                "备注\nDescription",
            ],
            ["", "", "借方\nDebit", "贷方\nCredit", "", "对手机构", "对手名称", "", ""],
            ["2025/01/02", "0001", "50.00", "", "100.00", "浦发银行", "甲公司", "S1", "付款"],
            ["2025/01/03", "0002", "", "75.00", "175.00", "浦发银行", "乙公司", "S2", "收款"],
        ]
    ]

    tables, meta = reconstruct_tables(mirror, "")

    assert tables == mirror
    assert meta.source == "canonical_table"
    assert meta.expected_primary_rows == 2
    assert meta.expected_evidence_source == ""
    assert meta.expected_evidence_confidence == 0.0


def test_richer_pipe_table_wins_over_sparse_canonical_table():
    mirror = [[["交易日期", "交易金额", "余额"], ["2024-01-01", "1.00", "9.00"]]]
    row2 = BOC_ROW1.replace("| 1  |", "| 2  |").replace("43627150", "43627151")
    row3 = BOC_ROW1.replace("| 1  |", "| 3  |").replace("43627150", "43627152")
    text = _synthetic_boc_text([row2, row3])
    tables, meta = reconstruct_tables(mirror, text)
    assert meta.source == "pipe_text"
    assert meta.expected_primary_rows == 3
    assert len(tables[0]) == 4


def test_canonical_noise_table_falls_through_to_spaced_ocr():
    mirror = [
        [
            ["3.21:80.14", "", ""],
            ["14 交易日期 交易类型 收入/支出 交易金额 账户余额", "", ""],
            ["合计", "100.00", "200.00"],
        ]
    ]
    tables, meta = reconstruct_tables(mirror, SAMPLE_OCR, route="scanned")
    assert meta.source == "spaced_ocr"
    assert len(tables[0]) >= 2


def test_mirror_table_expected_uses_mirror_ssot_not_raw_max():
    from docmirror.models.entities.parse_result import (
        CellValue,
        LogicalTable,
        ParseResult,
        ParserInfo,
        RowType,
        TableRow,
    )

    headers = ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"]
    rows = [
        TableRow(
            cells=[CellValue(text="2024-01-01"), CellValue(text="x"), CellValue(text="1.00")],
            row_type=RowType.DATA,
        )
        for _ in range(47)
    ]
    pr = ParseResult(
        logical_tables=[
            LogicalTable(
                headers=headers,
                rows=rows,
                row_count=47,
                quality_passed=True,
                data_row_estimate=47,
            )
        ],
        parser_info=ParserInfo(
            structure={
                "ltqg_enabled": True,
                "ltqg_expected_data_rows": 47,
            }
        ),
    )
    mirror = [[headers] + [[c.text for c in row.cells] for row in rows]]
    tables, meta = reconstruct_tables(
        mirror,
        "",
        parse_result=pr,
        structure_spe=pr.parser_info.structure,
    )
    assert tables == mirror
    assert meta.expected_primary_rows == 47
    assert meta.expected_primary_rows < 127


def test_unbounded_count_label_does_not_override_stale_mirror_expected_rows() -> None:
    from docmirror.models.entities.parse_result import ParseResult, ParserInfo

    mirror = [
        [
            ["交易日期", "支出（元）", "收入（元）", "账户余额（元）"],
            ["2025-01-01", "10.00", "", "90.00"],
            ["2025-01-02", "", "20.00", "110.00"],
        ]
    ]
    parse_result = ParseResult(
        parser_info=ParserInfo(
            structure={
                "ltqg_enabled": True,
                "ltqg_expected_data_rows": 1,
            }
        )
    )

    _, meta = reconstruct_tables(
        mirror,
        "账户名称：测试企业\n交易日期 交易金额 账户余额\n总条数：2",
        parse_result=parse_result,
        structure_spe=parse_result.parser_info.structure,
    )

    assert meta.source == "canonical_table"
    assert meta.expected_primary_rows == 1
    assert meta.expected_evidence_source == ""


@pytest.mark.parametrize(
    "evidence",
    [
        RowCountEvidence(9, "unknown", 0.99),
        RowCountEvidence(9, "candidate_rows", 0.99),
        RowCountEvidence(9, "page_transaction_anchors", 0.99),
        RowCountEvidence(9, "physical_rows", 0.99),
        RowCountEvidence(9, "positioned_date_anchors", 0.99),
        RowCountEvidence(9, "positioned_record_blocks", 0.99),
    ],
)
def test_canonical_reconstruction_rejects_nonissuer_row_count_evidence(
    monkeypatch: pytest.MonkeyPatch,
    evidence: RowCountEvidence,
) -> None:
    mirror = [[["Date", "Amount", "Balance"], ["2025-01-01", "1.00", "9.00"]]]
    monkeypatch.setattr(ltro_module, "_resolve_source_row_count_evidence", lambda *_args, **_kwargs: evidence)

    _, meta = reconstruct_tables(mirror, "")

    assert meta.expected_primary_rows == 1
    assert meta.expected_evidence_source == ""
    assert meta.expected_evidence_confidence == 0.0


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
def test_canonical_reconstruction_accepts_issuer_row_count_evidence(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    confidence: float,
) -> None:
    mirror = [[["Date", "Amount", "Balance"], ["2025-01-01", "1.00", "9.00"]]]
    evidence = RowCountEvidence(9, source, confidence)
    monkeypatch.setattr(ltro_module, "_resolve_source_row_count_evidence", lambda *_args, **_kwargs: evidence)

    _, meta = reconstruct_tables(mirror, "")

    assert meta.expected_primary_rows == 9
    assert meta.expected_evidence_source == source
    assert meta.expected_evidence_confidence == confidence


def test_mirror_table_raw_max_without_parse_result():
    mirror = [
        [["交易日期", "交易金额", "余额"], ["2024-01-01", "1.00", "9.00"]],
        [["x", "y"]] + [["bad"] for _ in range(10)],
    ]
    _, meta = reconstruct_tables(mirror, "")
    assert meta.expected_primary_rows == 10


def test_pipe_before_spaced_ocr():
    text = _synthetic_boc_text()
    tables, meta = reconstruct_tables([], text, page_count=1)
    assert meta.source == "pipe_text"
    assert len(tables[0]) >= 2


def test_pipe_fail_no_spaced_fallback():
    text = _synthetic_boc_text().split(BOC_ROW1)[0]
    tables, meta = reconstruct_tables([], text)
    assert tables == []
    assert meta.pipe_header_detected is True
    assert meta.pipe_parse_failed is True
    assert meta.source == "none"


def test_spaced_ocr_when_no_pipe():
    tables, meta = reconstruct_tables([], SAMPLE_OCR, route="scanned")
    assert meta.source == "spaced_ocr"
    assert len(tables[0]) >= 2
