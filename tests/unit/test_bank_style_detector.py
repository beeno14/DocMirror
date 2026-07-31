# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BankStyleDetector."""

from __future__ import annotations

from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector


def _yinzuo_table():
    return [
        ["银座银行交易明细", "", ""],
        ["账号:651204680300015", "第1 / 1页", ""],
        ["日期支出收入余额", "对方账户对方户名", "摘要/附言"],
        ["2025-09-210.04306.09", "", "结息"],
        ["00:07:46", "", ""],
    ]


def _grid_table():
    return [
        ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"],
        ["2024-01-01", "工资", "0.00", "5000.00", "8000.00"],
    ]


def test_detector_yinzuo_compact_merged():
    ctx = StyleContext(
        tables=[_yinzuo_table()],
        full_text="银座银行交易明细",
        institution="银座银行",
        page_count=1,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style == "compact_merged_ledger"
    assert "compact_merged" in result.parser_chain
    assert result.confidence >= 0.55


def test_detector_grid_standard():
    ctx = StyleContext(
        tables=[_grid_table()],
        full_text="中国建设银行账户明细",
        institution="中国建设银行",
        page_count=3,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style in ("grid_standard", "split_debit_credit")
    assert result.primary_style != "compact_merged_ledger"


def test_detector_institution_hint():
    ctx = StyleContext(
        tables=[_yinzuo_table()],
        full_text="银座银行",
        institution=None,
        page_count=1,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.institution_hint == "银座银行"


def test_detector_ignores_income_word_in_transaction_body_for_signed_amount_style():
    table = [
        ["交易日期", "对方户名", "交易摘要", "发生额", "余额"],
        ["2024-03-11 22:27:09", "北京微播视界科技有限公司", "汇款", "-45.00", "15214.50"],
        ["2024-03-12 12:05:32", "短信提醒服务费收入", "收费", "5.27", "15219.77"],
    ]
    ctx = StyleContext(
        tables=[table],
        full_text="湖北农商银行个人交易流水",
        institution="湖北农商银行",
        page_count=1,
    )

    result = BankStyleDetector().detect(ctx)

    assert result.primary_style == "signed_amount"
    assert result.parser_chain == ["signed_amount"]
