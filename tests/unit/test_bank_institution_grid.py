# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Institution column maps + grid_standard / split_debit_credit integration."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    LogicalTable,
    ParseResult,
    RowProvenance,
    TableRow,
)
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.grid_standard import normalize_split_debit_credit
from docmirror.plugins.bank_statement.wide_table_recovery import _select_wide_bank_table, is_wide_bank_header


def test_match_institution_ccb():
    variant = match_institution("中国建设银行账户明细信息")
    assert variant is not None
    assert variant.id == "ccb"
    assert variant.column_map.get("交易日期") == "交易时间"


def test_normalize_table_headers_ccb_alias():
    variant = match_institution("中国建设银行")
    tables = [[["交易日期", "摘要", "余额"], ["2024-01-01", "转账", "100.00"]]]
    normalized = normalize_table_headers(tables, variant=variant)
    assert normalized[0][0][0] == "交易时间"


def test_split_debit_credit_style_detection():
    ctx = StyleContext(
        tables=[[
            ["交易日期", "摘要", "收入", "支出", "余额"],
            ["2024-01-01", "工资入账", "5000.00", "0.00", "8000.00"],
        ]],
        full_text="中国工商银行 个人客户交易明细",
        institution=None,
        page_count=1,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style == "split_debit_credit"


def test_style_registry_icbc_split_columns():
    ctx = StyleContext(
        tables=[[
            ["交易日期", "摘要", "收入", "支出", "余额"],
            ["2024-01-01", "工资入账", "5000.00", "0.00", "8000.00"],
            ["2024-01-02", "转账支出", "0.00", "200.00", "7800.00"],
            ["2024-01-03", "消费", "0.00", "50.00", "7750.00"],
        ]],
        full_text="中国工商银行\n个人客户交易明细\n户名：张三",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _identity = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3
    directions = {r["normalized"].get("direction") for r in records}
    assert "income" in directions
    assert "expense" in directions


def test_normalize_split_debit_credit_direct():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2024-01-02",
            "摘要": "转账支出",
            "收入": "0.00",
            "支出": "200.00",
            "余额": "7800.00",
        },
        plugin,
    )
    assert norm is not None
    assert norm["amount"] == 200.0
    assert norm["direction"] == "expense"


def test_normalize_merged_balance_and_timestamp_split_columns():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易时间": "2025/01/0316:18:35",
            "摘要": "个人所得税",
            "借方发生额": "15.00",
            "贷方发生额": "0.00",
            "账户余额流水号": "363,693.0255420250100824712870",
        },
        plugin,
    )
    assert norm is not None
    assert norm["amount"] == 15.0
    assert norm["direction"] == "expense"
    assert norm["balance"] == 363693.02


def test_normalize_direction_embedded_after_amount():
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2023-10-02",
            "摘要": "跨行代付",
            "支/收交易金额": "23,903.69付收",
            "账户余额": "23,903.69",
        },
        plugin,
    )
    assert norm is not None
    assert norm["amount"] == 23903.69
    assert norm["direction"] == "income"
    assert norm["balance"] == 23903.69


@pytest.mark.parametrize(
    ("raw_direction", "expected"),
    [("贷Cr", "income"), ("借Dr", "expense")],
)
def test_normalize_bilingual_debit_credit_flag(raw_direction, expected):
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2022-08-05",
            "借贷": raw_direction,
            "交易金额": "40.00",
            "余额": "41.06",
        },
        plugin,
    )

    assert norm is not None
    assert norm["direction"] == expected


@pytest.mark.parametrize(
    ("raw_direction", "expected"),
    [
        ("转入", "income"),
        ("转出", "expense"),
        ("转\n入", "income"),
        ("转 出", "expense"),
        ("收入", "income"),
        ("支出", "expense"),
    ],
)
def test_normalize_transaction_category_direction(raw_direction, expected):
    plugin = BankStatementCommunityPlugin()
    norm = normalize_split_debit_credit(
        {
            "交易日期": "2023-08-29",
            "交易类别": raw_direction,
            "交易金额": "228.00",
            "账户余额": "372.38",
        },
        plugin,
    )

    assert norm is not None
    assert norm["amount"] == 228.0
    assert norm["direction"] == expected


def test_canonical_logical_grid_preserves_generic_row_provenance_and_raw_columns():
    headers = ["交易日期", "交易金额", "交易类别", "账户余额", "对方账号", "对方户名", "备注", "交易机构"]
    raw_rows = [
        ["20230829", "228.00", "转出", "372.38", "243300133", "扫二维码付款", "财付通支\n付", "101001"],
        ["20230828", "1600.00", "转入", "1972.38", "6230000000000000000", "测试对手方", "转账", "101001"],
    ]
    rows = []
    provenance = []
    for row_index, (page_number, values) in enumerate(zip((1, 2), raw_rows, strict=True)):
        cells = [
            CellValue(
                text=value,
                evidence_ids=[f"ev:{page_number:04d}:{row_index:06d}:{col_index:02d}"],
                source_cell_refs=[
                    {
                        "page": page_number,
                        "table_id": f"pt_{page_number}_0",
                        "row": row_index,
                        "col": col_index,
                    }
                ],
            )
            for col_index, value in enumerate(values)
        ]
        rows.append(
            TableRow(
                cells=cells,
                source_page=page_number,
                source_physical_id=f"pt_{page_number}_0",
                source_row_index=row_index,
            )
        )
        provenance.append(
            RowProvenance(
                source_page=page_number,
                source_table_id=f"pt_{page_number}_0",
                source_row_index=row_index,
            )
        )

    logical_table = LogicalTable(
        table_id="lt_transactions",
        headers=headers,
        rows=rows,
        source_physical_ids=["pt_1_0", "pt_2_0"],
        source_pages=[1, 2],
        page_span=(1, 2),
        row_count=2,
        provenance=provenance,
    )
    parse_result = ParseResult(logical_tables=[logical_table])
    ctx = StyleContext(
        tables=[[headers, *raw_rows]],
        full_text="银行账户交易明细",
        institution=None,
        page_count=2,
        parse_result=parse_result,
        reconstruction=ReconstructionMeta(source="canonical_table", expected_primary_rows=2),
    )
    detection = BankStyleDetector().detect(ctx)
    records, _identity = BankStyleParserRegistry().run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert len(records) == 2
    assert [record["normalized"]["direction"] for record in records] == ["expense", "income"]
    assert records[0]["normalized"]["summary"] == "财付通支付"
    assert records[0]["raw"]["备注"] == "财付通支\n付"
    assert records[0]["raw"]["交易机构"] == "101001"
    assert "交易机构" not in records[0]["normalized"]
    assert "交易机构" not in records[0]["canonical_raw"]
    assert [record["source"]["source_page"] for record in records] == [1, 2]
    assert [record["source"]["table_id"] for record in records] == ["pt_1_0", "pt_2_0"]
    assert all(record["source"]["source_cell_refs"] for record in records)
    assert all(record["source"]["evidence_ids"] for record in records)


def test_wide_table_accepts_date_anchored_rows_without_sequence_column():
    table = [
        ["交易时间", "摘要", "借方发生额", "贷方发生额", "账户余额流水号"],
        ["2025/01/0316:18:35", "个人所得税", "15.00", "0.00", "363,693.0255420250100824712870"],
        ["2025/01/0710:41:35", "社保费", "113.16", "0.00", "363,579.8650400202507000001657"],
    ]
    assert is_wide_bank_header(table[0]) is True
    assert len(_select_wide_bank_table(table)) == 3


def test_wide_table_accepts_direction_embedded_amount_header():
    table = [
        ["交易日期", "记账日期", "摘要", "支/收交易金额", "账户余额"],
        ["2023-10-02", "2023-10-02", "跨行代付", "23,903.69付收", "23,903.69"],
        ["2023-10-07", "2023-10-07", "跨行代付", "13,610.09付收", "13,610.09"],
    ]
    assert is_wide_bank_header(table[0]) is True
    assert len(_select_wide_bank_table(table)) == 3
