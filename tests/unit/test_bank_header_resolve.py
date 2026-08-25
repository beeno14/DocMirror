# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for unified bank header resolution."""

from __future__ import annotations

from docmirror.plugins.bank_statement.community_plugin import BANK_COLUMN_REGISTRY
from docmirror.plugins.bank_statement.header_resolve import (
    align_bank_ledger_row,
    detect_headers,
    normalize_header_cell,
    registry_strict_header_match_count,
)

OCR_HEADERS = ["值日", "交易说明", "发生金额", "账面余领"]
CLEAN_HEADERS = ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"]


def test_normalize_header_cell_maps_ocr_variants():
    assert normalize_header_cell("值日") == "交易日期"
    assert normalize_header_cell("账面余领") == "余额"


def test_registry_strict_fails_on_ocr_aliases():
    table = [[OCR_HEADERS]]
    assert registry_strict_header_match_count(table, BANK_COLUMN_REGISTRY) < 3


def test_detect_headers_succeeds_on_ocr_aliases():
    table = [[OCR_HEADERS]]
    header = detect_headers(table, BANK_COLUMN_REGISTRY, prefer_strict=True)
    assert header is not None
    assert len(header.col_map) >= 3


def test_explicit_debit_credit_flag_wins_over_transaction_type_alias():
    headers = ["序号", "交易日期", "交易时间", "交易类型", "借贷", "交易金额", "余额"]

    header = detect_headers([[headers]], BANK_COLUMN_REGISTRY, prefer_strict=True)

    assert header is not None
    assert header.col_map["direction"] == 4


def test_transaction_row_is_not_merged_as_a_debit_credit_subheader() -> None:
    headers = ["交易日期", "记账日期", "摘要", "支/收", "交易金额", "账户余额", "对方户名", "对方账户/对方银行"]
    first_transaction = [
        "2025-04-01",
        "2025-04-01",
        "服务费",
        "支出",
        "300",
        "399",
        "测试收入服务有限公司",
        "123456789测试银行",
    ]

    header = detect_headers([[headers, first_transaction]], BANK_COLUMN_REGISTRY, prefer_strict=True)

    assert header is not None
    assert header.raw_headers == headers
    assert header.row_index == 0
    assert header.col_map["direction"] == 3
    assert header.col_map["counter_account"] == 7


def test_true_debit_credit_subheader_is_still_merged() -> None:
    parent = ["交易日期", "摘要", "交易金额", "交易金额", "余额"]
    child = ["", "", "借方发生额", "贷方发生额", ""]

    header = detect_headers([[parent, child]], BANK_COLUMN_REGISTRY, prefer_strict=True)

    assert header is not None
    assert "借方发生额" in header.raw_headers[2]
    assert "贷方发生额" in header.raw_headers[3]


def test_compact_date_row_detection():
    from docmirror.plugins.bank_statement.row_extract import row_has_transaction_data

    row = ["20220505", "152713", "代付", "+800.00", "805.72", "财付通", "电子商务", ""]
    assert row_has_transaction_data(row) is True


def test_compact_datetime_row_detection_validates_date_and_time() -> None:
    from docmirror.plugins.bank_statement.row_extract import row_has_transaction_data

    assert row_has_transaction_data(["20220911112748", "微信支付", "-12.13", "13181.12"]) is True
    assert row_has_transaction_data(["20221311112748", "微信支付", "-12.13", "13181.12"]) is False
    assert row_has_transaction_data(["20220911252748", "微信支付", "-12.13", "13181.12"]) is False


def test_unicode_header_normalization():
    assert normalize_header_cell("交易⽇期") == "交易日期"
    assert normalize_header_cell("本次余额") == "余额"

    table = [[CLEAN_HEADERS]]
    header = detect_headers(table, BANK_COLUMN_REGISTRY, prefer_strict=True)
    assert header is not None
    assert header.mode == "strict"
    assert len(header.col_map) >= 3


def test_align_bank_ledger_row_restores_omitted_optional_time_cell():
    headers = ["交易日期", "交易时间", "交易摘要", "交易金额", "本次余额", "对手信息", "交易渠道", "附言"]
    shifted = ["20220904", "短信费", "-2.00", "102.47", "345601940050307", "短信费", "", ""]

    assert align_bank_ledger_row(headers, shifted) == [
        "20220904",
        "",
        "短信费",
        "-2.00",
        "102.47",
        "345601940050307",
        "短信费",
        "",
    ]
