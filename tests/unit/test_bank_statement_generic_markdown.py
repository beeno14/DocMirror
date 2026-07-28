# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generic bank statement Markdown and record cleanup."""

from __future__ import annotations

from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import (
    _render_bank_statement_content_markdown,
    _sanitize_bank_records,
)


def test_bank_statement_dedupe_keeps_repeated_page_sequences() -> None:
    records = [
        {
            "normalized": {"sequence_no": "1", "date": "2023-03-09", "amount": 1, "balance": 9},
            "raw": {},
            "source": {"table_id": "bank_table_p1", "page_range": [1, 1]},
        },
        {
            "normalized": {"sequence_no": "1", "date": "2023-04-01", "amount": 2, "balance": 11},
            "raw": {},
            "source": {"table_id": "bank_table_p2", "page_range": [2, 2]},
        },
        {
            "normalized": {"sequence_no": "1", "date": "2023-04-01", "amount": 2, "balance": 11},
            "raw": {},
            "source": {"table_id": "bank_table_p2", "page_range": [2, 2]},
        },
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 2


def test_bank_statement_content_markdown_is_record_complete_and_generic() -> None:
    records = [
        {
            "raw": {
                "序号": "1",
                "摘要": "往来款",
                "交易日期": "20240102",
                "交易金额": "80,000.00",
                "余额": "102,214.76",
                "对方户名": "35001677107*****5957/顺***融竹木有限公司",
            },
            "normalized": {"direction": "income"},
            "source": {"source_page": 1, "page_range": [1, 1]},
        },
        {
            "raw": {
                "序号": "549",
                "摘要": "电子汇入生成时间:2025-02-0710:29:08",
                "交易日期": "20241231",
                "交易金额": "40,000.00",
                "余额": "168,166.41",
                "对方户名": "6230522020*****8471/陈*明总页数：29",
            },
            "normalized": {"direction": "income"},
            "source": {"source_page": 29, "page_range": [29, 29]},
        },
    ]

    markdown = _render_bank_statement_content_markdown(
        records,
        {"account_holder": "郑云华", "account_number": "6227001863030091717", "currency": "CNY"},
        {"start": "2024-01-02", "end": "2024-12-31"},
    )

    business_rows = [
        line for line in markdown.splitlines() if line.startswith("| ") and line.split("|", 3)[1].strip().isdigit()
    ]
    assert len(business_rows) == 2
    assert markdown.count("docmirror:page") == 2
    assert "郑云华" in markdown
    assert "生成时间" not in markdown
    assert "总页数" not in markdown
    assert "35001677107*****5957/顺***融竹木有限公司" in markdown


def test_bank_statement_record_sanitizer_removes_footer_and_money_noise() -> None:
    records = [
        {
            "raw": {
                "摘要": "电子汇入生成时间:2025-02-0710:29:08",
                "余额": "168,166.41***",
                "对方户名": "6230522020*****8471/陈*明总页数：29",
            },
            "canonical_raw": {
                "summary": "电子汇入生成时间:2025-02-0710:29:08",
                "balance": "168,166.41***",
                "counter_party": "6230522020*****8471/陈*明总页数：29",
            },
            "normalized": {
                "summary": "电子汇入生成时间:2025-02-0710:29:08",
                "balance": "168,166.41***",
                "counter_party": "6230522020*****8471/陈*明总页数：29",
            },
        }
    ]

    sanitized = _sanitize_bank_records(records)

    assert sanitized[0]["raw"]["摘要"] == "电子汇入"
    assert sanitized[0]["raw"]["余额"] == "168,166.41"
    assert sanitized[0]["raw"]["对方户名"] == "6230522020*****8471/陈*明"
    assert sanitized[0]["canonical_raw"]["balance"] == "168,166.41"
    assert sanitized[0]["normalized"]["counter_party"] == "6230522020*****8471/陈*明"
