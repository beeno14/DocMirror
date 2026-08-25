# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for source-column-driven native borderless ledgers."""

from __future__ import annotations

import json
from typing import Any

from docmirror.plugins.bank_statement.canonical import (
    _canonical_raw_input_for_source_repair,
    records_from_raw_transactions,
)
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
    _recover_borderless_native_page_with_header,
    _recover_cross_page_wide_tables,
    _repair_native_summary_signed_money_spill,
    resolve_row_count_evidence,
)


class _Page:
    height = 200.0

    def __init__(
        self,
        words: list[dict[str, Any]],
        *,
        edges: list[dict[str, Any]] | None = None,
        flow_words: list[dict[str, Any]] | None = None,
    ) -> None:
        self._words = words
        self._flow_words = flow_words
        self.edges = list(edges or [])

    def extract_words(self, **kwargs: Any) -> list[dict[str, Any]]:
        if kwargs.get("use_text_flow") and self._flow_words is not None:
            return list(self._flow_words)
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

    assert table[0][:-4] == headers
    assert len(table[0][:-4]) == 7
    assert table[1][:-4] == row_one
    assert table[2][:-4] == row_two
    assert table[1][-2] == "3"


def test_flow_order_owns_wrapped_words_until_next_date_anchor() -> None:
    starts = [10.0, 100.0, 180.0, 260.0, 340.0, 430.0, 520.0]
    headers = ["交易日期", "摘要", "交易金额", "余额", "交易地点", "对方户名", "对方账户/对方银行"]
    header_words = [_word(value, starts[index], 10.0) for index, value in enumerate(headers)]
    row_one = [
        _word("2022-01-01", 10.0, 30.0),
        _word("支出", 100.0, 30.0),
        _word("-10.00", 180.0, 30.0),
        _word("90.00", 260.0, 30.0),
        _word("一支行", 340.0, 34.0),
        _word("甲公司", 430.0, 36.0),
        _word("11111111甲银行", 520.0, 38.0),
    ]
    row_two = [
        _word("2022-01-02", 10.0, 60.0),
        _word("收入", 100.0, 60.0),
        _word("+20.00", 180.0, 60.0),
        _word("110.00", 260.0, 60.0),
        _word("二支行", 340.0, 64.0),
        _word("乙公司", 430.0, 66.0),
        _word("22222222乙银行", 520.0, 68.0),
    ]
    geometry_words = [*header_words, *sorted([*row_one, *row_two], key=lambda word: (word["top"], word["x0"]))]
    # Content-stream order owns the wrapped prefix of row two after its date
    # anchor even though those words render above that anchor geometrically.
    flow_words = [*header_words, *row_one, row_two[3], *row_two[:3], *row_two[4:]]

    table = _recover_borderless_native_page(_Page(geometry_words, flow_words=flow_words), 1)

    assert table[1][4:7] == ["一支行", "甲公司", "11111111甲银行"]
    assert table[2][4:7] == ["二支行", "乙公司", "22222222乙银行"]


def test_flow_order_stops_before_footer_prefix_serialized_ahead_of_print_time() -> None:
    starts = [10.0, 100.0, 180.0, 260.0, 340.0, 430.0, 520.0]
    headers = ["交易日期", "摘要", "交易金额", "余额", "交易地点", "对方户名", "对方账户/对方银行"]
    header_words = [_word(value, starts[index], 10.0) for index, value in enumerate(headers)]
    row_one = [
        _word("2022-01-01", 10.0, 30.0),
        _word("支出", 100.0, 30.0),
        _word("-10.00", 180.0, 30.0),
        _word("90.00", 260.0, 30.0),
        _word("一支行", 340.0, 34.0),
        _word("甲公司", 430.0, 36.0),
        _word("11111111甲银行", 520.0, 38.0),
    ]
    row_two = [
        _word("二支行", 340.0, 44.0),
        _word("乙公司", 430.0, 45.0),
        _word("22222222乙银行", 520.0, 46.0),
        _word("2022-01-02", 10.0, 60.0),
        _word("收入", 100.0, 60.0),
        _word("+20.00", 180.0, 60.0),
        _word("110.00", 260.0, 60.0),
    ]
    footer = [
        _word("某银行", 520.0, 130.0),
        _word("2026/02/24", 10.0, 132.0),
        _word("打印时间：", 100.0, 134.0),
    ]
    geometry_words = [
        *header_words,
        *sorted([*row_one, *row_two, *footer], key=lambda word: (word["top"], word["x0"])),
    ]
    flow_words = [*header_words, *row_one, *row_two, *footer]

    table = _recover_borderless_native_page(_Page(geometry_words, flow_words=flow_words), 1)

    assert [row[0] for row in table[1:]] == ["2022-01-01", "2022-01-02"]
    assert not any("2026/02/24" in cell or "打印时间" in cell for row in table[1:] for cell in row)


def test_flow_order_excludes_pre_header_furniture_from_last_row() -> None:
    starts = [10.0, 100.0, 180.0, 260.0, 340.0]
    headers = ["交易日期", "摘要", "交易金额", "余额", "对方户名"]
    header_words = [_word(value, starts[index], 10.0) for index, value in enumerate(headers)]
    row_one = ["2022-01-01", "支出", "-10.00", "90.00", "甲公司"]
    row_two = ["2022-01-02", "收入", "+20.00", "110.00", "乙公司"]
    row_one_words = [_word(value, starts[index], 30.0) for index, value in enumerate(row_one)]
    row_two_words = [_word(value, starts[index], 50.0) for index, value in enumerate(row_two)]
    print_date = _word("打印日期：2023-12-22", starts[0], 1.0, 100.0)
    geometry_words = [print_date, *header_words, *row_one_words, *row_two_words]
    flow_words = [*header_words, *row_one_words, *row_two_words, print_date]

    table = _recover_borderless_native_page(_Page(geometry_words, flow_words=flow_words), 1)

    assert [row[0] for row in table[1:]] == ["2022-01-01", "2022-01-02"]


def test_flow_order_falls_back_when_anchor_count_disagrees() -> None:
    starts = [10.0, 100.0, 180.0, 260.0, 340.0]
    headers = ["交易日期", "摘要", "交易金额", "余额", "对方户名"]
    row_one = ["20240101", "消费", "-10", "90", "甲公司"]
    row_two = ["20240102", "入账", "+20", "110", "乙公司"]
    words = [
        *[_word(value, starts[index], 10.0) for index, value in enumerate(headers)],
        *[_word(value, starts[index], 30.0) for index, value in enumerate(row_one)],
        *[_word(value, starts[index], 50.0) for index, value in enumerate(row_two)],
    ]
    flow_words = [word for word in words if word["text"] != "20240102"]

    table = _recover_borderless_native_page(_Page(words, flow_words=flow_words), 1)

    assert [row[0] for row in table[1:]] == ["20240101", "20240102"]


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


def test_header_band_keeps_slightly_shifted_adjacent_field() -> None:
    starts = [10.0, 80.0, 150.0, 220.0, 290.0, 370.0]
    headers = ["交易日期", "交易金额", "账户余额", "对手信息", "交易备注", "渠道"]
    values = ["20240101", "-10.00", "90.00", "测试公司", "往来款", "网银"]
    words = [
        *[
            _word(value, starts[index], 10.8 if value == "交易备注" else 10.0)
            for index, value in enumerate(headers)
        ],
        *[_word(value, starts[index], 30.0) for index, value in enumerate(values)],
    ]

    table = _recover_borderless_native_page(_Page(words), 1)

    assert table[0][:-4] == headers
    assert table[1][:-4] == values


def test_touching_balance_and_serial_headers_stay_distinct() -> None:
    starts = [22.0, 61.1, 200.13, 234.88, 311.45, 385.30, 466.16, 498.16, 545.95, 711.04]
    headers = [
        "交易时间",
        "摘要",
        "凭证类型",
        "凭证号码",
        "借方发生额",
        "贷方发生额",
        "账户余额",
        "流水号",
        "对方户名/账号",
        "对方行名",
    ]
    values = [
        "2025/01/0316:18:35",
        "个人所得税",
        "电子凭证",
        "1",
        "15.00",
        "0.00",
        "363,693.02",
        "554202501030\n08247128705",
        "暂收款/190700000003371002",
        "国家金库江门市中心支库",
    ]
    words = [
        *[
            _word(value, starts[index], 10.0, 28.0 if value == "账户余额" else 26.0)
            for index, value in enumerate(headers)
        ]
    ]
    for index, value in enumerate(values):
        if index == 0:
            words.extend(
                [
                    _word("2025/01/03", starts[index], 30.0, 52.0),
                    _word("16:18:35", starts[index], 36.0, 40.0),
                ]
            )
        elif index == 7:
            words.extend(
                [
                    _word("554202501030", starts[index], 30.0, 60.0),
                    _word("08247128705", starts[index], 36.0, 60.0),
                ]
            )
        else:
            words.append(_word(value, starts[index], 30.0, 60.0))

    table = _recover_borderless_native_page(_Page(words), 1)

    assert table[0][:-4] == headers
    assert table[1][6] == "363,693.02"
    assert table[1][7] == "554202501030\n08247128705"


def test_flow_order_does_not_prepend_bottom_footer_artifacts_to_first_row() -> None:
    starts = [22.0, 80.0, 180.0, 260.0, 340.0]
    headers = ["交易时间", "摘要", "借方发生额", "贷方发生额", "账户余额"]
    header_words = [_word(value, starts[index], 10.0) for index, value in enumerate(headers)]
    row = ["2025/01/03", "个人所得税", "15.00", "0.00", "363,693.02"]
    row_words = [_word(value, starts[index], 30.0) for index, value in enumerate(row)]
    row_words.append(_word("16:18:35", starts[0], 36.0, 40.0))
    footer_words = [
        _word("_" * 80, 20.0, 160.0, 500.0),
        _word("打印渠道:网银", 20.0, 170.0, 100.0),
        _word("合计笔数:1", 200.0, 170.0, 100.0),
    ]
    geometry_words = [*header_words, *row_words, *footer_words]
    flow_words = [*footer_words, *header_words, *row_words]

    table = _recover_borderless_native_page(_Page(geometry_words, flow_words=flow_words), 1)

    assert len(table) == 2
    assert table[1][0] == "2025/01/03\n16:18:35"
    assert "_" not in "".join(table[1])


def test_flow_order_owns_pre_date_voucher_prefix_and_keeps_time_in_transaction_field() -> None:
    starts = [22.0, 56.13, 97.08, 173.87, 344.07, 416.60, 448.60, 482.72, 533.91, 568.04, 696.02]
    headers = [
        "凭证类型",
        "凭证号码",
        "交易时间",
        "摘要",
        "交易金额",
        "账户余额",
        "现转标志",
        "交易渠道",
        "交易机构",
        "对方户名/账号",
        "对方行名",
    ]
    header_widths = [30.0, 36.0, 36.0, 28.0, 28.0, 28.0, 28.0, 36.0, 28.0, 70.0, 60.0]
    header_words = [
        _word(value, starts[index], 10.0, header_widths[index])
        for index, value in enumerate(headers)
    ]
    row_one_prefix = [
        _word("卡", starts[0], 30.0, 10.0),
        _word("6226223880", starts[1], 30.0, 38.0),
        _word("005635", starts[1], 37.0, 24.0),
    ]
    row_one_body = [
        _word("2023/01/22", starts[2], 30.0, 36.0),
        _word("05:30:27", 135.58, 30.0, 28.0),
        _word("信用卡自扣还款", starts[3], 30.0, 70.0),
        _word("-2,061.15", starts[4], 30.0, 42.0),
        _word("0.00", starts[5], 30.0, 24.0),
        _word("转账", starts[6], 30.0, 18.0),
        _word("其它外围", starts[7], 30.0, 40.0),
        _word("6600", starts[8], 30.0, 20.0),
        _word("6226230006293805", starts[9], 30.0, 72.0),
    ]
    row_two_prefix = [
        _word("卡", starts[0], 52.5, 10.0),
        _word("6226223803", starts[1], 52.5, 38.0),
        _word("076168", starts[1], 59.5, 24.0),
    ]
    row_two_body = [
        _word("2023/01/25", starts[2], 52.5, 36.0),
        _word("07:31:45", 135.58, 52.5, 28.0),
        _word("银联入账-广州合利宝支付科技有限公司", starts[3], 52.5, 140.0),
        _word("3,421.13", starts[4], 52.5, 42.0),
        _word("3,421.13", starts[5], 52.5, 42.0),
        _word("转账", starts[6], 52.5, 18.0),
        _word("银联", starts[7], 52.5, 24.0),
        _word("9003", starts[8], 52.5, 20.0),
        _word("广州合利宝支付科技有限公司备付金", starts[9], 52.5, 120.0),
        _word("/991581002602", starts[9], 59.5, 60.0),
        _word("中国银联股份有限公司", starts[10], 52.5, 90.0),
    ]
    geometry_words = [
        *header_words,
        *sorted(
            [*row_one_prefix, *row_one_body, *row_two_prefix, *row_two_body],
            key=lambda word: (word["top"], word["x0"]),
        ),
    ]
    flow_words = [*header_words, *row_one_prefix, *row_one_body, *row_two_prefix, *row_two_body]

    table = _recover_borderless_native_page(_Page(geometry_words, flow_words=flow_words), 1)

    assert table[0][:-4] == headers
    assert table[1][:-4] == [
        "卡",
        "6226223880\n005635",
        "2023/01/2205:30:27",
        "信用卡自扣还款",
        "-2,061.15",
        "0.00",
        "转账",
        "其它外围",
        "6600",
        "6226230006293805",
        "",
    ]
    assert table[2][:-4] == [
        "卡",
        "6226223803\n076168",
        "2023/01/2507:31:45",
        "银联入账-广州合利宝支付科技有限公司",
        "3,421.13",
        "3,421.13",
        "转账",
        "银联",
        "9003",
        "广州合利宝支付科技有限公司备付金\n/991581002602",
        "中国银联股份有限公司",
    ]
    first = grid_standard.normalize_record(dict(zip(table[0], table[1])), BankStatementCommunityPlugin())
    assert first["timestamp"] == "2023-01-22T05:30:27"
    assert first["voucher_number"] == "6226223880005635"
    assert first["cash_remittance"] == "转账"
    assert first["channel"] == "其它外围"
    assert first["counter_account"] == "6226230006293805"


def test_native_recovery_excludes_colored_stamp_overlay_and_preserves_bojs_time_summary() -> None:
    starts = [27.826, 59.0, 104.0, 197.0, 274.0, 315.0, 365.0, 427.0, 469.0, 524.0]
    headers = [
        "序号",
        "交易日期",
        "交易时间",
        "摘要",
        "凭证种类",
        "借方发生额",
        "贷方发生额",
        "余额",
        "对方账户",
        "对方户名",
    ]
    header_words = [
        _word(value, starts[index], 10.0, 20.0 if index == 0 else 28.0)
        for index, value in enumerate(headers)
    ]
    row_42 = [
        _word("42", starts[0], 30.0, 8.0),
        _word("2022-07-13", starts[1], 30.0, 36.0),
        _word("16:18:49", 107.0, 30.0, 26.0),
        _word("7.13号电费预存1500元", 143.79, 27.0, 76.0),
        _word("7.13号电费预", 221.22, 27.0, 45.0),
        _word("存1500元", 189.61, 37.0, 31.0),
        _word("1,500.00", starts[5], 32.0, 32.0),
        _word("651.11", starts[7], 32.0, 24.0),
        _word("7065018800007", starts[8], 27.0, 50.0),
        _word("9350", starts[8], 37.0, 16.0),
        _word("镇江大学科技园", starts[9], 27.0, 58.0),
        _word("发展有限公司", starts[9], 37.0, 48.0),
    ]
    stamp = _word("CPKYG0GJD514", 169.1, 30.2, 62.0)
    stamp.update(
        {
            "fontname": "Helvetica",
            "non_stroking_color": (1.0, 0.0, 0.0),
            "stroking_color": (0.0,),
        }
    )
    for word in [*header_words, *row_42]:
        word.update(
            {
                "fontname": "STSong-Light",
                "non_stroking_color": (0.0, 0.0, 0.0),
                "stroking_color": (0.0,),
            }
        )
    row_43 = [
        _word("43", starts[0], 55.0, 8.0),
        _word("2022-07-15", starts[1], 55.0, 36.0),
        _word("11:54:58", 107.0, 55.0, 26.0),
        _word("报销", starts[3], 55.0, 20.0),
        _word("298.00", starts[5], 55.0, 28.0),
        _word("353.11", starts[7], 55.0, 24.0),
    ]
    for word in row_43:
        word.update(
            {
                "fontname": "STSong-Light",
                "non_stroking_color": (0.0, 0.0, 0.0),
                "stroking_color": (0.0,),
            }
        )
    geometry_words = [
        *header_words,
        *sorted([*row_42, stamp, *row_43], key=lambda word: (word["top"], word["x0"])),
    ]
    flow_words = [*header_words, *row_42, stamp, *row_43]

    table = _recover_borderless_native_page(_Page(geometry_words, flow_words=flow_words), 2)

    assert len(table) == 3
    assert table[0][:-4] == headers
    assert table[1][2] == "16:18:49"
    assert table[1][3] == "7.13号电费预存1500元"
    assert "CPKYG0GJD514" not in "".join(table[1])
    normalized = grid_standard.normalize_record(dict(zip(table[0], table[1])), BankStatementCommunityPlugin())
    assert normalized["timestamp"] == "2022-07-13T16:18:49"
    assert normalized["summary"] == "7.13号电费预存1500元"


def test_compound_counterparty_tolerance_does_not_swallow_note_code() -> None:
    starts = [10.0, 90.0, 170.0, 240.0, 390.0, 480.0]
    headers = ["交易日期", "交易金额", "账户余额", "交易对手信息", "交易备注", "渠道"]
    words = [*[_word(value, starts[index], 10.0) for index, value in enumerate(headers)]]
    words.extend(
        [
            _word("20220105", starts[0], 30.0),
            _word("-500000.00", starts[1], 30.0),
            _word("591664.74", starts[2], 30.0),
            _word("九江钟瓴募尚创业投资合伙企业(有限合", starts[3] - 3.0, 30.0, 145.0),
            _word("伙)", starts[3], 36.0),
            _word("190251710708", starts[3], 42.0, 60.0),
            _word("中国银行股份有限公司九江市八里湖新区支行", starts[3], 48.0, 145.0),
            _word("632259888745", starts[3], 54.0, 60.0),
            _word("108", starts[4] - 1.0, 30.0, 14.0),
            _word("汇入投资款", starts[4] + 16.0, 30.0, 60.0),
            _word("907", starts[5], 30.0),
        ]
    )

    table = _recover_borderless_native_page(_Page(words), 1)
    raw = dict(zip(table[0], table[1]))
    normalized = grid_standard.normalize_record(raw, BankStatementCommunityPlugin())

    assert raw["交易备注"] == "108汇入投资款"
    assert normalized["counter_party"] == "九江钟瓴募尚创业投资合伙企业(有限合伙)"
    assert normalized["counter_account"] == "190251710708"
    assert normalized["counter_bank_code"] == "632259888745"


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
    assert table[1][:-4] == row_one
    assert table[2][:-4] == row_two
    assert count.count == 2
    assert count.source == "page_transaction_anchors"


def test_native_summary_signed_money_spill_is_bounded_and_idempotent() -> None:
    headers = ["交易日期", "摘要", "交易金额", "账户余额"]
    variants = (
        ("Apple-19.00", "Apple", "-19.00"),
        ("_Apple-19.00", "_Apple", "-19.00"),
        ("音乐+1,219.00", "音乐", "+1,219.00"),
    )
    for fused, prefix, money in variants:
        cells = ["2023-04-08", "AppStore_Music", fused, "549.28"]

        manifest = _repair_native_summary_signed_money_spill(cells, headers, amount_columns=[2])

        assert manifest is not None
        assert cells == ["2023-04-08", f"AppStore_Music{prefix}", money, "549.28"]
        assert manifest["source_amount"] == fused
        assert manifest["summary_prefix"] == prefix
        assert _repair_native_summary_signed_money_spill(cells, headers, amount_columns=[2]) is None
        assert cells[1] == f"AppStore_Music{prefix}"


def test_native_summary_signed_money_spill_controls_fail_closed() -> None:
    adjacent_headers = ["交易日期", "摘要", "交易金额", "账户余额"]
    for value in ("USD-19.00", "_USD-19.00", "Apple19.00", "Apple-refund", "$Apple-19.00"):
        cells = ["2023-04-08", "purchase", value, "549.28"]
        original = list(cells)

        assert _repair_native_summary_signed_money_spill(cells, adjacent_headers, amount_columns=[2]) is None
        assert cells == original

    misaligned_headers = ["交易日期", "摘要", "账户余额", "交易金额"]
    misaligned = ["2023-04-08", "purchase", "549.28", "Apple-19.00"]
    assert _repair_native_summary_signed_money_spill(
        misaligned,
        misaligned_headers,
        amount_columns=[3],
    ) is None
    assert misaligned[-1] == "Apple-19.00"

    ambiguous_headers = ["摘要", "交易金额", "备注", "发生额", "余额"]
    ambiguous = ["first", "Apple-1.00", "second", "Music-2.00", "90.00"]
    assert _repair_native_summary_signed_money_spill(
        ambiguous,
        ambiguous_headers,
        amount_columns=[1, 3],
    ) is None
    assert ambiguous == ["first", "Apple-1.00", "second", "Music-2.00", "90.00"]


def _native_repaired_table(
    *,
    source_summary: str = "AppStore_Music",
) -> tuple[list[list[str]], list[str], list[str], dict[str, str]]:
    headers = ["交易日期", "摘要", "交易金额", "账户余额"]
    source_cells = ["2023-04-08", source_summary, "_Apple-19.00", "549.28"]
    source_raw = dict(zip(headers, source_cells, strict=True))
    working_cells = list(source_cells)
    manifest = _repair_native_summary_signed_money_spill(working_cells, headers, amount_columns=[2])
    assert manifest is not None
    table = [
        [*headers, "_source_raw_json", "_source_repair_json", "_source_page", "_source_bbox"],
        [
            *working_cells,
            json.dumps(source_raw, ensure_ascii=False, separators=(",", ":")),
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            "1",
            "10.000,20.000,300.000,30.000",
        ],
    ]
    return table, headers, working_cells, manifest


def test_cross_page_composition_preserves_source_json_and_synchronizes_only_working_values() -> None:
    table, headers, _working_cells, _manifest = _native_repaired_table(source_summary="AppStore\nMusic")
    source_json = table[1][-4]

    composed = _recover_cross_page_wide_tables([table])
    transactions = grid_standard._extract_internal_source_grid_records(composed)
    plugin = BankStatementCommunityPlugin()
    records = records_from_raw_transactions(
        transactions,
        normalize_fn=lambda raw: grid_standard.normalize_record(raw, plugin),
        style_id="grid_standard",
        canonical_raw_fn=plugin._canonical_raw_values,
    )

    assert len(composed) == 1
    assert composed[0][1][-4] == source_json
    synchronized = json.loads(composed[0][1][-3])
    assert synchronized["source_summary"] == "AppStore\nMusic"
    assert synchronized["source_amount"] == "_Apple-19.00"
    assert synchronized["working_summary"] == "AppStore Music_Apple"
    assert synchronized["working_amount"] == "-19.00"
    assert records[0]["raw"][headers[1]] == "AppStore\nMusic"
    assert records[0]["raw"][headers[2]] == "_Apple-19.00"
    assert records[0]["normalized"]["summary"] == "AppStore Music_Apple"
    assert records[0]["canonical_raw"]["summary"] == "AppStore Music_Apple"
    assert records[0]["canonical_raw"]["amount"] == "-19.00"


def test_cross_page_repair_validates_cleaned_wrapped_nonrepair_fields_as_a_full_map() -> None:
    headers = ["凭证号码", "交易日期", "摘要", "交易金额", "账户余额", "交易对方"]
    source_cells = [
        "PZ\n001",
        "2023-04-08",
        "AppStore\nMusic",
        "_Apple-19.00",
        "549.28",
        "Example\nMerchant",
    ]
    source_raw = dict(zip(headers, source_cells, strict=True))
    working_cells = list(source_cells)
    manifest = _repair_native_summary_signed_money_spill(working_cells, headers, amount_columns=[3])
    assert manifest is not None
    table = [
        [*headers, "_source_raw_json", "_source_repair_json", "_source_page", "_source_bbox"],
        [
            *working_cells,
            json.dumps(source_raw, ensure_ascii=False, separators=(",", ":")),
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            "1",
            "10.000,20.000,300.000,30.000",
        ],
    ]

    composed = _recover_cross_page_wide_tables([table])
    transactions = grid_standard._extract_internal_source_grid_records(composed)
    plugin = BankStatementCommunityPlugin()
    records = records_from_raw_transactions(
        transactions,
        normalize_fn=lambda raw: grid_standard.normalize_record(raw, plugin),
        style_id="grid_standard",
        canonical_raw_fn=plugin._canonical_raw_values,
    )

    assert len(transactions) == 1
    assert {header: transactions[0][header] for header in headers} == {
        "凭证号码": "PZ 001",
        "交易日期": "2023-04-08",
        "摘要": "AppStore Music_Apple",
        "交易金额": "-19.00",
        "账户余额": "549.28",
        "交易对方": "Example Merchant",
    }
    assert transactions[0]["_source_raw"] == source_raw
    assert records[0]["raw"] == source_raw
    assert records[0]["canonical_raw"]["summary"] == "AppStore Music_Apple"
    assert records[0]["canonical_raw"]["amount"] == "-19.00"
    assert records[0]["normalized"]["amount"] == 19.0
    assert records[0]["normalized"]["direction"] == "expense"

    forged_working = {header: transactions[0][header] for header in headers}
    forged_working["交易对方"] = "Forged Merchant"
    assert (
        _canonical_raw_input_for_source_repair(
            transactions[0],
            source_public=source_raw,
            working_public=forged_working,
        )
        is source_raw
    )

    tampered_composed = [[list(row) for row in composed[0]]]
    tampered_composed[0][1][headers.index("交易对方")] = "Forged Merchant"
    tampered_transactions = grid_standard._extract_internal_source_grid_records(tampered_composed)
    assert len(tampered_transactions) == 1
    assert "_source_raw" not in tampered_transactions[0]
    assert "_source_repair_manifest" not in tampered_transactions[0]
    assert "_canonical_raw_from_working" not in tampered_transactions[0]


def test_cross_page_composition_does_not_synchronize_forged_repair_manifest() -> None:
    table, _headers, _working_cells, _manifest = _native_repaired_table(source_summary="AppStore\nMusic")
    forged = json.loads(table[1][-3])
    forged["working_summary"] = "forged\nApple"
    forged_json = json.dumps(forged, ensure_ascii=False, separators=(",", ":"))
    table[1][-3] = forged_json

    composed = _recover_cross_page_wide_tables([table])
    transactions = grid_standard._extract_internal_source_grid_records(composed)

    assert composed[0][1][-3] == forged_json
    assert "_source_raw" not in transactions[0]
    assert "_source_repair_manifest" not in transactions[0]
    assert "_canonical_raw_from_working" not in transactions[0]


def test_native_source_repair_preserves_raw_and_uses_working_canonical_values() -> None:
    table, headers, working_cells, _manifest = _native_repaired_table()
    plugin = BankStatementCommunityPlugin()

    transactions = grid_standard._extract_internal_source_grid_records([table])
    records = records_from_raw_transactions(
        transactions,
        normalize_fn=lambda raw: grid_standard.normalize_record(raw, plugin),
        style_id="grid_standard",
        canonical_raw_fn=plugin._canonical_raw_values,
    )

    assert len(transactions) == 1
    assert transactions[0][headers[1]] == working_cells[1] == "AppStore_Music_Apple"
    assert transactions[0][headers[2]] == working_cells[2] == "-19.00"
    assert transactions[0]["_source_raw"][headers[1]] == "AppStore_Music"
    assert transactions[0]["_source_raw"][headers[2]] == "_Apple-19.00"
    assert transactions[0]["_canonical_raw_from_working"] is True
    assert records[0]["raw"][headers[1]] == "AppStore_Music"
    assert records[0]["raw"][headers[2]] == "_Apple-19.00"
    assert records[0]["normalized"]["summary"] == "AppStore_Music_Apple"
    assert records[0]["normalized"]["amount"] == 19.0
    assert records[0]["normalized"]["direction"] == "expense"
    assert records[0]["canonical_raw"]["summary"] == "AppStore_Music_Apple"
    assert records[0]["canonical_raw"]["amount"] == "-19.00"


def test_native_source_repair_metadata_rejects_malformed_mismatched_and_colliding_inputs() -> None:
    table, _headers, _working_cells, _manifest = _native_repaired_table()
    raw_headers = list(table[0])
    baseline = dict(zip(raw_headers, table[1], strict=True))
    malformed_inputs = []

    malformed = dict(baseline)
    malformed["_source_raw_json"] = "{not-json"
    malformed_inputs.append((malformed, raw_headers))

    mismatched = dict(baseline)
    mismatched_source = json.loads(mismatched["_source_raw_json"])
    mismatched_source.pop("账户余额")
    mismatched["_source_raw_json"] = json.dumps(mismatched_source, ensure_ascii=False)
    malformed_inputs.append((mismatched, raw_headers))

    collision = dict(baseline)
    collision_headers = [*raw_headers, "_source_raw_json"]
    malformed_inputs.append((collision, collision_headers))

    for transaction, headers in malformed_inputs:
        grid_standard._decode_native_source_repair(transaction, headers)
        assert "_source_raw" not in transaction
        assert "_source_repair_manifest" not in transaction
        assert "_canonical_raw_from_working" not in transaction
        assert "_source_raw_json" not in transaction
        assert "_source_repair_json" not in transaction


def test_canonical_working_raw_gate_rejects_forged_or_broadened_repairs() -> None:
    table, headers, working_cells, manifest = _native_repaired_table()
    source_public = json.loads(table[1][-4])
    working_public = dict(zip(headers, working_cells, strict=True))

    forged = {"_canonical_raw_from_working": True, "_source_raw": source_public}
    assert (
        _canonical_raw_input_for_source_repair(
            forged,
            source_public=source_public,
            working_public=working_public,
        )
        is source_public
    )

    broadened_working = dict(working_public)
    broadened_working["账户余额"] = "548.28"
    broadened = {
        "_canonical_raw_from_working": True,
        "_source_raw": source_public,
        "_source_repair_manifest": manifest,
    }
    assert (
        _canonical_raw_input_for_source_repair(
            broadened,
            source_public=source_public,
            working_public=broadened_working,
        )
        is source_public
    )

    mismatched_source = dict(source_public)
    mismatched_source.pop("账户余额")
    mismatched = {
        "_canonical_raw_from_working": True,
        "_source_raw": mismatched_source,
        "_source_repair_manifest": manifest,
    }
    assert (
        _canonical_raw_input_for_source_repair(
            mismatched,
            source_public=mismatched_source,
            working_public=working_public,
        )
        is mismatched_source
    )


def test_blank_native_source_repair_metadata_leaves_ordinary_row_unchanged() -> None:
    headers = ["交易日期", "摘要", "交易金额", "账户余额"]
    values = ["2023-04-08", "purchase", "-19.00", "549.28"]
    table = [
        [*headers, "_source_raw_json", "_source_repair_json", "_source_page", "_source_bbox"],
        [*values, "", "", "1", "10.000,20.000,300.000,30.000"],
    ]

    transactions = grid_standard._extract_internal_source_grid_records([table])

    assert len(transactions) == 1
    assert {header: transactions[0][header] for header in headers} == dict(zip(headers, values, strict=True))
    assert "_source_raw" not in transactions[0]
    assert "_canonical_raw_from_working" not in transactions[0]


def test_repeated_header_uses_local_coordinates_without_changing_source_labels() -> None:
    headers = ["交易日期", "记账日期", "摘要", "支/收", "交易金额", "账户余额", "对方户名"]
    inherited_starts = [10.0, 70.0, 130.0, 185.0, 215.0, 270.0, 350.0]
    local_starts = [10.0, 72.0, 134.0, 205.0, 230.0, 272.0, 350.0]
    values = ["2022-07-15", "2022-07-15", "快捷支付", "支", "-2,783.90", "4,209.11", "测试公司"]
    words = [
        *[_word(value, local_starts[index], 10.0, 24.0) for index, value in enumerate(headers)],
        *[_word(value, local_starts[index], 30.0, 24.0) for index, value in enumerate(values)],
    ]

    table, carried = _recover_borderless_native_page_with_header(
        _Page(words),
        2,
        inherited_header=(headers, inherited_starts),
    )

    assert table[0][:-4] == headers
    assert table[1][:-4] == values
    assert carried == (headers, inherited_starts)


def test_native_vertical_rules_separate_sequence_and_summary_cells() -> None:
    headers = ["序号", "摘要", "币别", "交易日期", "交易金额", "账户余额", "对方账号与户名"]
    starts = [34.0, 100.0, 184.0, 272.0, 338.0, 414.0, 679.0]
    values = ["1", "往来款", "人民币元", "20240102", "80,000.00", "102,214.76", "35001/*****有限公司"]
    value_starts = [35.0, 58.0, 175.0, 274.0, 357.0, 429.0, 617.0]
    boundaries = [32.0, 55.0, 164.0, 263.0, 321.0, 395.0, 472.0, 813.0]
    edges = [
        {
            "orientation": "v",
            "object_type": "line",
            "x0": boundary,
            "x1": boundary,
            "top": 5.0 + repetition * 20.0,
            "bottom": 40.0 + repetition * 20.0,
        }
        for boundary in boundaries
        for repetition in range(2)
    ]
    words = [
        *[_word(value, starts[index], 10.0, 24.0) for index, value in enumerate(headers)],
        *[_word(value, value_starts[index], 30.0, 24.0) for index, value in enumerate(values)],
    ]

    table = _recover_borderless_native_page(_Page(words, edges=edges), 1)

    assert table[1][:-4] == values


def test_sequence_anchor_keeps_shifted_trailing_text_in_distinct_fields() -> None:
    headers = ["序号", "记账日期", "交易金额", "账户余额", "摘要描述", "对方户名"]
    starts = [36.0, 69.0, 149.0, 223.0, 326.0, 475.0]
    values = ["1", "2023-09-07", "-20,000.00", "66,428.91", "城商行-个人实时贷记往账", "储成"]
    value_starts = [44.0, 69.0, 149.0, 227.0, 271.0, 432.0]
    words = [
        *[_word(value, starts[index], 10.0, 24.0) for index, value in enumerate(headers)],
        *[_word(value, value_starts[index], 30.0, 24.0) for index, value in enumerate(values)],
    ]

    table = _recover_borderless_native_page(_Page(words), 1)

    assert table[1][:-4] == values


def test_count_evidence_rejects_unbounded_cib_total_label_and_conflicts() -> None:
    cib = resolve_row_count_evidence(
        "",
        page_texts=[(10, "账号：6222000000000000\n交易日期 交易金额 账户余额\n交易总笔额：122")],
    )
    conflicting = resolve_row_count_evidence(
        "",
        page_texts=[(1, "交易总笔数：20"), (2, "交易总笔数：30")],
    )

    assert cib.count == 0
    assert cib.source == "none"
    assert conflicting.count == 0
    assert conflicting.source == "none"


def test_negative_expense_reversal_preserves_explicit_source_direction() -> None:
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
    assert records[1]["normalized"]["direction"] == "expense"


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
    assert canonical_raw["note"] == "用途:往来款。"


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
    assert normalized["remittance_note"] == "二维码付款"
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

    assert table[0][:-4] == headers
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


def test_transaction_amount_alias_and_distinct_description_fields_are_preserved() -> None:
    plugin = BankStatementCommunityPlugin()
    normalized = grid_standard.normalize_record(
        {
            "序 号": "63",
            "交易日期": "2024-03-31",
            "交易发生金额": "-8.00",
            "账户余额": "92.00",
            "摘要": "服务费",
            "交易描述": "网上银行收费",
        },
        plugin,
    )

    assert normalized["sequence_no"] == "63"
    assert normalized["amount"] == 8.0
    assert normalized["direction"] == "expense"
    assert normalized["summary"] == "服务费"
    assert normalized["transaction_name"] == "网上银行收费"
