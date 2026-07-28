# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ping An monthly statement recovery."""

from __future__ import annotations

import pytest

from docmirror.models.entities.parse_result import DocumentEntities, ParseResult
from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import (
    BankStatementCommunityPlugin,
    _render_bank_statement_content_markdown,
    _sanitize_bank_records,
)
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.pingan_monthly import (
    _extract_identity_from_tokens,
    _extract_page_rows,
)


def _token(text: str, x0: float, y0: float, x1: float, y1: float, idx: int, conf: float = 0.99) -> dict:
    return {
        "id": f"t{idx}",
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "confidence": conf,
        "page": 1,
    }


def _pingan_tokens() -> list[dict]:
    tokens = [
        _token("平安银行", 150, 37, 277, 68, 1),
        _token("客户存款月结单", 335, 69, 456, 92, 2),
        _token("客户行：平安银行上海西南支行", 49, 98, 274, 120, 3),
        _token("户名：上海炫酷广告有限公司", 431, 99, 641, 119, 4),
        _token("账号：11005350836201", 47, 121, 212, 141, 5),
        _token("币种:RMB", 431, 121, 509, 142, 6),
        _token("序号", 47, 155, 86, 178, 7),
        _token("日期", 119, 156, 158, 178, 8),
        _token("借/贷方发生额", 208, 157, 320, 177, 9),
        _token("余额", 369, 156, 410, 178, 10),
        _token("对方户名", 527, 155, 599, 179, 11),
        _token("对方账户", 806, 155, 877, 180, 12),
        _token("传票号", 1000, 156, 1055, 178, 13),
        _token("摘要", 1082, 156, 1120, 178, 14),
        _token("20230309", 117, 191, 184, 208, 15),
        _token("-4,819.00", 208, 190, 273, 210, 16),
        _token("401,143.31", 367, 190, 439, 210, 17),
        _token("杨光", 532, 189, 569, 211, 18),
        _token("6226192013864418", 809, 191, 921, 208, 19),
        _token("备用金", 1085, 190, 1133, 211, 20),
        _token("2", 46, 211, 62, 231, 21),
        _token("20230309", 117, 211, 185, 231, 22),
        _token("-13,300.00", 208, 211, 278, 231, 23),
        _token("387,843.31", 366, 211, 439, 231, 24),
        _token("谢林华", 532, 211, 582, 232, 25),
        _token("6226192011784154", 809, 213, 920, 230, 26),
        _token("备用金", 1085, 211, 1133, 233, 27),
        _token("3", 45, 232, 61, 252, 28),
        _token("20230310", 116, 232, 186, 252, 29),
        _token("+1,000.00", 208, 232, 278, 252, 30),
        _token("388,843.31", 366, 232, 439, 253, 31),
        _token("王恒英", 532, 232, 582, 253, 32),
        _token("6230880020022554141", 809, 233, 920, 251, 33),
        _token("往来款", 1085, 232, 1133, 253, 34),
    ]
    return tokens


def test_pingan_monthly_extracts_rows_identity_and_source() -> None:
    tokens = _pingan_tokens()

    identity, evidence_ids = _extract_identity_from_tokens(tokens)
    rows = _extract_page_rows(tokens, page_no=1)

    assert identity["bank_name"] == "平安银行"
    assert identity["bank_branch"] == "平安银行上海西南支行"
    assert identity["account_holder"] == "上海炫酷广告有限公司"
    assert identity["account_number"] == "11005350836201"
    assert identity["currency"] == "RMB"
    assert evidence_ids

    assert len(rows) == 3
    assert rows[0]["交易日期"] == "20230309"
    assert rows[0]["收/支"] == "支出"
    assert rows[0]["交易金额"] == "4819.00"
    assert rows[0]["余额"] == "401143.31"
    assert rows[0]["对方户名"] == "杨光"
    assert rows[0]["对方账号"] == "6226192013864418"
    assert rows[2]["收/支"] == "收入"
    assert rows[2]["交易金额"] == "1000.00"
    assert rows[2]["余额"] == "388843.31"
    assert rows[0]["_source"]["page_range"] == [1, 1]
    assert rows[0]["_source"]["bbox"]
    assert rows[0]["_source"]["confidence"] > 0
    assert rows[0]["_source"]["evidence_ids"]


def test_pingan_monthly_registry_uses_cached_ocr_rows() -> None:
    rows = _extract_page_rows(_pingan_tokens(), page_no=1)
    parse_result = ParseResult(
        entities=DocumentEntities(
            document_type="bank_statement",
            domain_specific={
                "_bank_pingan_monthly_ocr": {
                    "status": "ready",
                    "source": "pingan_monthly_ocr",
                    "identity": {
                        "bank_name": "平安银行",
                        "bank_branch": "平安银行上海西南支行",
                        "account_holder": "上海炫酷广告有限公司",
                        "account_number": "11005350836201",
                        "currency": "RMB",
                    },
                    "identity_evidence_ids": ["t1", "t4", "t5"],
                    "transactions": rows,
                    "page_counts": {1: 3},
                    "warnings": [],
                }
            },
        )
    )
    ctx = StyleContext(
        tables=[],
        full_text="客户存款月结单 结单号 借/贷方发生额 承前余额",
        institution="平安银行",
        page_count=1,
        parse_result=parse_result,
    )

    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _identity = BankStyleParserRegistry().run(detection, ctx, plugin)

    assert detection.primary_style == "pingan_monthly_statement"
    assert len(records) == 3
    assert records[0]["normalized"]["direction"] == "expense"
    assert records[0]["normalized"]["amount"] == pytest.approx(4819.0)
    assert records[2]["normalized"]["direction"] == "income"
    assert records[2]["normalized"]["balance"] == pytest.approx(388843.31)
    assert records[0]["source"]["page_range"] == [1, 1]


def test_pingan_monthly_dedupe_keeps_repeated_page_sequences() -> None:
    records = [
        {
            "normalized": {"sequence_no": "1", "date": "2023-03-09", "amount": 1, "balance": 9},
            "raw": {},
            "source": {"table_id": "pingan_monthly_ocr_p1", "page_range": [1, 1]},
        },
        {
            "normalized": {"sequence_no": "1", "date": "2023-04-01", "amount": 2, "balance": 11},
            "raw": {},
            "source": {"table_id": "pingan_monthly_ocr_p2", "page_range": [2, 2]},
        },
        {
            "normalized": {"sequence_no": "1", "date": "2023-04-01", "amount": 2, "balance": 11},
            "raw": {},
            "source": {"table_id": "pingan_monthly_ocr_p2", "page_range": [2, 2]},
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
