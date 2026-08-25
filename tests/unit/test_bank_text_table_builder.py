# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservation tests for native/OCR text-ledger reconstruction."""

from __future__ import annotations

from docmirror.plugins.bank_statement.text_table_builder import (
    build_tables_from_spaced_ocr_text,
    build_tables_from_stacked_bank_text,
)


def test_spaced_text_builder_preserves_repeated_ordered_transactions() -> None:
    text = "\n".join(
        [
            "银行流水 账号 1234567890123456 账户余额",
            "20240101支出100.00 900.00 fee",
            "20240101收入100.00 1000.00 reversal",
            "20240101支出100.00 900.00 fee",
        ]
    )

    tables = build_tables_from_spaced_ocr_text(text)

    assert len(tables) == 1
    assert [row[2:] for row in tables[0][1:]] == [
        ["-100.00", "900.00"],
        ["+100.00", "1000.00"],
        ["-100.00", "900.00"],
    ]


def test_stacked_text_builder_preserves_repeated_ordered_transactions() -> None:
    text = "\n".join(
        [
            "银行流水",
            "账号 1234567890123456",
            "交易日期",
            "账户余额",
            "甲方",
            "首笔",
            "900.00",
            "/",
            "2024-01-01",
            "10:00:00",
            "转账",
            "-100.00",
            "乙方",
            "冲正",
            "1000.00",
            "/",
            "2024-01-01",
            "10:01:00",
            "冲正",
            "+100.00",
            "甲方",
            "再次扣款",
            "900.00",
            "/",
            "2024-01-01",
            "10:02:00",
            "转账",
            "-100.00",
        ]
    )

    tables = build_tables_from_stacked_bank_text(text)

    assert len(tables) == 1
    assert [(row[0], row[1], row[3], row[4]) for row in tables[0][1:]] == [
        ("2024-01-01", "10:00:00", "-100.00", "900.00"),
        ("2024-01-01", "10:01:00", "+100.00", "1000.00"),
        ("2024-01-01", "10:02:00", "-100.00", "900.00"),
    ]
