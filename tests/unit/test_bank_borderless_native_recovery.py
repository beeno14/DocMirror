# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for source-column-driven native borderless ledgers."""

from __future__ import annotations

from typing import Any

from docmirror.plugins.bank_statement.community_plugin import (
    BankStatementCommunityPlugin,
    _clean_counterparty_text,
    _raw_markdown_cell,
)
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.styles import grid_standard
from docmirror.plugins.bank_statement.wide_table_recovery import (
    _dedupe_tables,
    _recover_borderless_native_page,
    _recover_cross_page_wide_tables,
    resolve_row_count_evidence,
)


class _Page:
    height = 200.0

    def __init__(self, words: list[dict[str, Any]]) -> None:
        self._words = words

    def extract_words(self, **_: Any) -> list[dict[str, Any]]:
        return list(self._words)


def _word(text: str, x0: float, top: float, width: float = 32.0) -> dict[str, Any]:
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "bottom": top + 6.0}


def test_borderless_recovery_preserves_dynamic_source_columns() -> None:
    starts = [10.0, 80.0, 140.0, 210.0, 280.0, 350.0, 430.0]
    headers = ["交易日期", "交易时间", "摘要", "交易金额", "余额", "商户名称", "备注"]
    row_one = ["20240101", "083000", "消费", "-10", "90", "商户甲", "早餐"]
    row_two = ["20240102", "090000", "入账", "+20.5", "110.5", "商户乙", "退款"]
    words = [
        *[_word(value, starts[index], 10.0) for index, value in enumerate(headers)],
        *[_word(value, starts[index], 30.0) for index, value in enumerate(row_one)],
        *[_word(value, starts[index], 50.0) for index, value in enumerate(row_two)],
    ]

    table = _recover_borderless_native_page(_Page(words), 3)

    assert table[0][:-2] == headers
    assert len(table[0][:-2]) == 7
    assert table[1][:-2] == row_one
    assert table[2][:-2] == row_two
    assert table[1][-2] == "3"


def test_borderless_recovery_skips_bilingual_header_continuation_before_first_row() -> None:
    starts = [10.0, 80.0, 140.0, 210.0, 280.0, 350.0]
    headers = ["交易日期", "币种", "交易金额", "余额", "交易摘要", "对手信息"]
    words = [*[_word(value, starts[index], 10.0) for index, value in enumerate(headers)]]
    words.extend(
        [
            _word("Date", starts[0], 20.0),
            _word("Currency", starts[1], 20.0),
            _word("Transaction", starts[2], 20.0),
            _word("Amount", starts[2], 26.0),
            _word("Balance", starts[3], 20.0),
            _word("Transaction", starts[4], 20.0),
            _word("Type", starts[4], 26.0),
            _word("Counter", starts[5], 20.0),
            _word("Party", starts[5], 26.0),
            _word("测试有限公司", starts[5], 35.0, 60.0),
            _word("2024-01-01", starts[0], 40.0, 52.0),
            _word("CNY", starts[1], 40.0),
            _word("-10.00", starts[2], 40.0),
            _word("90.00", starts[3], 40.0),
            _word("付款", starts[4], 40.0),
            _word("123456789012345", starts[5], 40.0, 70.0),
        ]
    )

    table = _recover_borderless_native_page(_Page(words), 1)

    assert [row[0] for row in table[1:]] == ["2024-01-01"]
    assert table[1][5] == "测试有限公司\n123456789012345"


def test_borderless_recovery_accepts_unsigned_amount_when_direction_is_explicit() -> None:
    starts = [10.0, 45.0, 120.0, 185.0, 250.0, 325.0, 405.0]
    headers = ["序号", "交易日期", "收入/支出", "交易金额", "账户余额", "对方户名", "摘要"]
    row_one = ["1", "2025-06-27", "支出", "198.87", "144.74", "测试商户", "消费"]
    row_two = ["2", "2025-07-06", "收入", "25,000.00", "25,144.74", "测试用户", "转入"]
    words = [
        *[_word(value, starts[index], 10.0, 28.0) for index, value in enumerate(headers)],
        *[_word(value, starts[index], 30.0, 28.0) for index, value in enumerate(row_one)],
        *[_word(value, starts[index], 50.0, 28.0) for index, value in enumerate(row_two)],
    ]

    table = _recover_borderless_native_page(_Page(words), 1)
    count = resolve_row_count_evidence(
        "",
        page_texts=[
            (
                1,
                "序号 交易日期 收入/支出 交易金额 账户余额 对方户名 摘要\n"
                "1 2025-06-27 支出 198.87 144.74 测试商户 消费\n"
                "2 2025-07-06 收入 25,000.00 25,144.74 测试用户 转入",
            )
        ],
    )

    assert len(table) == 3
    assert table[1][:-2] == row_one
    assert table[2][:-2] == row_two
    assert count.count == 2
    assert count.source == "page_transaction_anchors"


def test_negative_expense_reversal_uses_absolute_amount_and_balance_direction() -> None:
    plugin = BankStatementCommunityPlugin()
    previous_raw = {
        "序号": "74",
        "交易日期": "2025-09-19",
        "收入/支出": "支出",
        "交易金额": "2,496.00",
        "账户余额": "3,883.31",
    }
    reversal_raw = {
        "序号": "75",
        "交易日期": "2025-09-19",
        "收入/支出": "支出",
        "交易金额": "-2,496.00",
        "账户余额": "6,379.31",
    }
    records = [
        {"raw": previous_raw, "normalized": grid_standard.normalize_record(previous_raw, plugin)},
        {"raw": reversal_raw, "normalized": grid_standard.normalize_record(reversal_raw, plugin)},
    ]

    grid_standard.refine_missing_directions_from_balance_chain(records)

    assert records[1]["normalized"]["amount"] == 2496.0
    assert records[1]["normalized"]["direction"] == "income"


def test_unsigned_rows_use_source_summary_semantics_and_forward_balance_chain() -> None:
    plugin = BankStatementCommunityPlugin()
    raw_rows = [
        {
            "序号": "1",
            "交易日期": "20250407",
            "发生额": "1000",
            "账户余额": "1000.00",
            "摘要描述": "小额跨行转入",
            "备注": "附言:往来结算款。",
        },
        {
            "序号": "5",
            "交易日期": "20250910",
            "发生额": "100",
            "账户余额": "999.27",
            "摘要描述": "小额跨行转入",
        },
        {
            "序号": "6",
            "交易日期": "20250921",
            "发生额": "0.19",
            "账户余额": "999.46",
            "摘要描述": "入息",
        },
        {
            "序号": "7",
            "交易日期": "20251221",
            "发生额": "0.2",
            "账户余额": "999.66",
            "摘要描述": "入息",
        },
    ]
    records = [{"raw": raw, "normalized": grid_standard.normalize_record(raw, plugin)} for raw in raw_rows]

    grid_standard.refine_missing_directions_from_balance_chain(records)

    assert records[0]["normalized"]["direction"] == "income"
    assert records[2]["normalized"]["direction"] == "income"


def test_balance_chain_overrides_misleading_payment_word_in_summary() -> None:
    plugin = BankStatementCommunityPlugin()
    raw_rows = [
        {
            "序号": "29",
            "交易日期": "2023-06-28",
            "交易金额": "-504,025.00",
            "账户余额": "8,033.18",
            "摘要描述": "偿还贷款本金和利息",
        },
        {
            "序号": "1",
            "交易日期": "2023-07-03",
            "交易金额": "600,000.00",
            "账户余额": "608,033.18",
            "摘要描述": "普通汇兑支付信息服务费",
        },
    ]
    records = [{"raw": raw, "normalized": grid_standard.normalize_record(raw, plugin)} for raw in raw_rows]

    grid_standard.refine_missing_directions_from_balance_chain(records)

    assert records[1]["normalized"]["direction"] == "income"


def test_explicit_summary_precedes_remark_and_wrapped_account_is_compacted() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "序号": "2",
        "交易日期": "20250616",
        "发生额": "-100",
        "账户余额": "900.00",
        "摘要描述": "单位他行非同客户转账",
        "备注": "用途:往来款。",
        "对方账号": "98010202900778158\n3",
        "对方户名": "测试公司",
    }

    normalized = grid_standard.normalize_record(raw, plugin)
    canonical_raw = plugin._canonical_raw_values(raw, normalized)

    assert normalized["summary"] == "单位他行非同客户转账"
    assert normalized["purpose"] == "用途:往来款。"
    assert normalized["counter_account"] == "980102029007781583"
    assert canonical_raw["summary"] == "单位他行非同客户转账"
    assert canonical_raw["purpose"] == "用途:往来款。"


def test_borderless_source_fields_normalize_without_losing_raw_columns() -> None:
    starts = [10.0, 70.0, 125.0, 185.0, 245.0, 305.0, 365.0, 425.0, 485.0]
    headers = ["交易日期", "交易时间", "交易摘要", "交易金额", "本次余额", "对手信息", "日志号", "交易渠道", "交易附言"]
    values = [
        "20230622",
        "195925",
        "微信支付",
        "-500.00",
        "10705.87",
        "243300133",
        "457272650",
        "电子商务",
        "二维码付款",
    ]
    words = [
        *[_word(value, starts[index], 10.0, 28.0) for index, value in enumerate(headers)],
        *[_word(value, starts[index], 30.0, 28.0) for index, value in enumerate(values)],
    ]
    table = _recover_borderless_native_page(_Page(words), 1)
    plugin = BankStatementCommunityPlugin()
    ctx = StyleContext(
        tables=[table],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=None,
        prefer_context_tables=True,
    )

    raw = grid_standard.extract_transactions(ctx, plugin)[0]
    normalized = grid_standard.normalize_record(raw, plugin)

    assert list(raw)[:9] == headers
    assert raw["交易金额"] == "-500.00"
    assert raw["交易时间"] == "195925"
    assert normalized["timestamp"] == "2023-06-22T19:59:25"
    assert normalized["sequence_no"] == "457272650"
    assert normalized["counter_account"] == "243300133"
    assert normalized["counter_party"] == ""
    assert normalized["channel"] == "电子商务"
    assert normalized["purpose"] == "二维码付款"
    assert raw["_source"]["source_page"] == 1
    assert len(raw["_source"]["bbox"]) == 4
    assert raw["_source"]["source_refs"][0]["bbox"] == raw["_source"]["bbox"]


def test_separate_compact_transaction_times_use_date_column_context() -> None:
    plugin = BankStatementCommunityPlugin()
    cases = (
        ("20240208", "140229", "2024-02-08T14:02:29"),
        ("20240223", "140631", "2024-02-23T14:06:31"),
        ("20240208", "120101", "2024-02-08T12:01:01"),
    )

    for date_value, time_value, expected in cases:
        raw = {
            "交易日期": date_value,
            "交易时间": time_value,
            "交易金额": "-1.00",
            "余额": "99.00",
        }

        normalized = grid_standard.normalize_record(raw, plugin)

        assert normalized["date"] == expected[:10]
        assert normalized["timestamp"] == expected
        assert raw["交易日期"] == date_value
        assert raw["交易时间"] == time_value


def test_repeated_cross_page_headers_preserve_source_page_order() -> None:
    header = [
        "交易日期",
        "交易时间",
        "交易摘要",
        "交易金额",
        "本次余额",
        "对手信息",
        "日志号",
        "交易渠道",
        "交易附言",
        "_source_page",
        "_source_bbox",
    ]
    page_one = [
        header,
        ["20230622", "195925", "微信支付", "-500.00", "10705.87", "243300133", "1", "电子商务", "", "1", "1,1,2,2"],
    ]
    page_two = [
        header,
        ["20230623", "080743", "微信支付", "-615.00", "283.29", "243300133", "2", "电子商务", "", "2", "1,1,2,2"],
    ]

    recovered = _recover_cross_page_wide_tables([page_one, page_two])
    deduped = _dedupe_tables([*recovered, page_one, page_two])

    assert len(deduped) == 1
    assert [row[-2] for row in deduped[0][1:]] == ["1", "2"]
    assert [row[0] for row in deduped[0][1:]] == ["20230622", "20230623"]


def test_borderless_recovery_supports_iso_dates_split_amounts_and_wrapped_cells() -> None:
    starts = [10.0, 80.0, 150.0, 220.0, 280.0, 350.0, 430.0]
    headers = ["日期", "支出", "收入", "余额", "对方账户", "对方户名", "摘要/附言"]
    words = [*[_word(value, starts[index], 10.0) for index, value in enumerate(headers)]]
    words.extend(
        [
            _word("2025-09-21", starts[0], 30.0, 50.0),
            _word("00:07:46", starts[0], 38.0, 40.0),
            _word("0.04", starts[2], 30.0),
            _word("306.09", starts[3], 30.0),
            _word("结息", starts[6], 30.0),
            _word("2025-10-27", starts[0], 55.0, 50.0),
            _word("16:36:24", starts[0], 63.0, 40.0),
            _word("3765000.00", starts[2], 55.0, 50.0),
            _word("3765306.09", starts[3], 55.0, 50.0),
            _word("01041560012000235", starts[4], 55.0, 80.0),
            _word("重庆正大能科科", starts[5], 55.0, 70.0),
            _word("技有限公司", starts[5], 63.0, 50.0),
            _word("电汇/服务费", starts[6], 55.0, 60.0),
            _word("2025-10-28", starts[0], 80.0, 50.0),
            _word("14:44:10", starts[0], 88.0, 40.0),
            _word("3000000.00", starts[1], 80.0, 50.0),
            _word("765306.09", starts[3], 80.0, 45.0),
            _word("100102029005622957", starts[4], 80.0, 90.0),
            _word("重庆数宜信信用", starts[5], 80.0, 70.0),
            _word("管理有限公司", starts[5], 88.0, 60.0),
            _word("电汇/服务费", starts[6], 80.0, 60.0),
        ]
    )

    table = _recover_borderless_native_page(_Page(words), 1)
    plugin = BankStatementCommunityPlugin()
    ctx = StyleContext(
        tables=[table],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=None,
        prefer_context_tables=True,
    )
    raw_records = grid_standard.extract_transactions(ctx, plugin)
    normalized = [grid_standard.normalize_record(row, plugin) for row in raw_records]

    assert table[0][:-2] == headers
    assert len(raw_records) == 3
    assert normalized[0]["timestamp"] == "2025-09-21T00:07:46"
    assert normalized[0]["direction"] == "income"
    assert normalized[0]["balance"] == 306.09
    assert normalized[1]["counter_account"] == "01041560012000235"
    assert normalized[1]["counter_party"] == "重庆正大能科科技有限公司"
    assert normalized[1]["summary"] == "电汇/服务费"
    assert normalized[2]["direction"] == "expense"
    assert normalized[2]["amount"] == 3000000.0
    assert normalized[2]["balance"] == 765306.09
    assert raw_records[0]["_source"]["source_page"] == 1
    assert raw_records[0]["_source"]["bbox"]


def test_source_counterparty_cell_splits_trailing_account() -> None:
    plugin = BankStatementCommunityPlugin()
    raw = {
        "交易日期": "2024-01-01",
        "交易金额": "-10.00",
        "余额": "90.00",
        "交易摘要": "付款",
        "对手信息": "测试有限公司 123456789012345",
    }

    normalized = grid_standard.normalize_record(raw, plugin)

    assert normalized["counter_party"] == "测试有限公司"
    assert normalized["counter_account"] == "123456789012345"


def test_wrapped_counterparty_parentheses_do_not_create_a_synthetic_space() -> None:
    assert grid_standard._clean_wrapped_text("测试软件\n(集团)有限公司") == "测试软件(集团)有限公司"
    assert _clean_counterparty_text("测试软件\n(集团)有限公司") == "测试软件(集团)有限公司"
    assert _raw_markdown_cell("测试软件\n(集团)有限公司") == "测试软件(集团)有限公司"
