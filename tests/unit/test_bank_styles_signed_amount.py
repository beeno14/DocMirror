# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for signed_amount bank statement style."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    LogicalTable,
    PageContent,
    ParseResult,
    RowProvenance,
    TableRow,
)
from docmirror.plugins.bank_statement.canonical import ensure_canonical_normalized
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extract_pipeline import run_bank_statement_extract
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.signed_amount import (
    parse_signed_amount,
    table_has_signed_amount_cells,
)

SIGNED_TABLE = [
    [
        ["交易日期", "摘要", "交易金额", "余额"],
        ["2024-01-01", "工资入账", "+5000.00", "5000.00"],
        ["2024-01-02", "消费", "-200.00", "4800.00"],
        ["2024-01-03", "转账", "-50.00", "4750.00"],
    ]
]


def test_parse_signed_amount_income_and_expense():
    amount, direction = parse_signed_amount("+5000.00")
    assert amount == 5000.0
    assert direction == "income"

    amount, direction = parse_signed_amount("-200.00")
    assert amount == 200.0
    assert direction == "expense"


def test_table_has_signed_amount_cells():
    assert table_has_signed_amount_cells(SIGNED_TABLE) is True
    split_table = [
        [
            ["交易日期", "摘要", "收入", "支出", "余额"],
            ["2024-01-01", "工资", "5000.00", "0.00", "5000.00"],
        ]
    ]
    assert table_has_signed_amount_cells(split_table) is False


def test_signed_amount_style_accepts_unsigned_credits_when_debits_keep_minus_sign():
    table = [
        [
            ["交易日期", "摘要", "发生额", "余额"],
            ["2024-01-01\n10:30:00", "消费", "-200.00", "4800.00"],
            ["2024-01-02", "结息", "5.27", "4805.27"],
        ]
    ]

    assert table_has_signed_amount_cells(table) is True

    ctx = StyleContext(tables=table, full_text="某银行交易明细", institution=None, page_count=1)
    records, _ = BankStyleParserRegistry().run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert [record["normalized"]["direction"] for record in records] == ["expense", "income"]
    assert records[0]["normalized"]["timestamp"] == "2024-01-01T10:30:00"


def test_canonical_normalization_does_not_invent_zero_amount() -> None:
    normalized = ensure_canonical_normalized({"date": "2024-01-01"}, ["date", "amount", "amount_cny"])

    assert normalized["amount"] is None
    assert normalized["amount_cny"] is None


@pytest.mark.parametrize(
    ("source_amount", "expected_direction"),
    [("+0.00", "income"), ("-0.00", "expense")],
)
def test_explicit_signed_zero_amount_is_preserved(
    source_amount: str,
    expected_direction: str,
) -> None:
    parsed_amount, direction = parse_signed_amount(source_amount)

    assert parsed_amount == 0.0
    assert direction == expected_direction


def test_detector_signed_amount_style():
    ctx = StyleContext(
        tables=SIGNED_TABLE,
        full_text="某银行交易明细",
        institution=None,
        page_count=1,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style == "signed_amount"


def test_registry_signed_amount_records():
    ctx = StyleContext(
        tables=SIGNED_TABLE,
        full_text="某银行交易明细",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) == 3
    assert records[0]["normalized"]["direction"] == "income"
    assert records[0]["normalized"]["amount"] == pytest.approx(5000.0)
    assert records[1]["normalized"]["direction"] == "expense"
    assert records[1]["normalized"]["amount"] == pytest.approx(200.0)


@pytest.mark.parametrize("document_type", ["bank_statement", "bank_reconciliation"])
def test_compatibility_headers_share_multi_page_signed_ledger_rules(document_type: str) -> None:
    """Compatibility glyphs and ledger semantics are generic across both bank document types."""
    headers = ["交易⽇期", "交易⾦额", "账户余额", "对⼿信息", "摘要"]
    values = [
        ["20220808", "+100.00", "1100.00", "甲公司", "转入"],
        ["20220808", "-20.00", "1080.00", "乙公司", "转出"],
        ["20220808", "-20.00", "1080.00", "乙公司", "转出"],
        ["20220809", "5.00", "1085.00", "", "结息"],
    ]
    pages = [1, 1, 2, 2]
    rows = [
        TableRow(
            cells=[CellValue(text=value) for value in row],
            source_page=page,
            source_physical_id=f"pt_{page}_0",
            source_row_index=index,
        )
        for index, (row, page) in enumerate(zip(values, pages))
    ]
    provenance = [
        RowProvenance(
            source_page=page,
            source_table_id=f"pt_{page}_0",
            source_row_index=index,
        )
        for index, page in enumerate(pages)
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1), PageContent(page_number=2)],
        entities=DocumentEntities(document_type=document_type),
        logical_tables=[
            LogicalTable(
                table_id="lt_compat",
                headers=headers,
                rows=rows,
                row_count=len(rows),
                data_row_estimate=len(rows),
                source_pages=[1, 2],
                source_physical_ids=["pt_1_0", "pt_2_0"],
                page_span=(1, 2),
                provenance=provenance,
                quality_passed=True,
            )
        ],
    )

    result = run_bank_statement_extract(
        parse_result,
        "中国农业银行账户交易明细\n交易⽇期 交易⾦额 账户余额 对⼿信息 摘要",
        BankStatementCommunityPlugin(),
    )

    assert len(result.records) == 4
    assert [record["normalized"]["direction"] for record in result.records] == [
        "income",
        "expense",
        "expense",
        "income",
    ]
    assert [record["source"]["source_page"] for record in result.records] == [1, 1, 2, 2]
    assert all(record["source"]["page_range"][0] == record["source"]["page_range"][1] for record in result.records)
    # The logical table proves what this candidate emitted, not that the source
    # document has no omitted terminal rows. Completeness remains unknown until
    # an issuer-provided denominator is independently established.
    assert result.style_meta.expected_primary_rows == 0
    assert result.style_meta.canonical_extracted == 4
    assert result.style_meta.extract_status == "degraded"
