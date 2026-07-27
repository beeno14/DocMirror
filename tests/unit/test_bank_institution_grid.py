# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Institution column maps + grid_standard / split_debit_credit integration."""

from __future__ import annotations

import pytest

from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
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
