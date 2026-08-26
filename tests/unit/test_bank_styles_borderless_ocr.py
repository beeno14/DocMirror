# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for borderless_ocr bank statement style."""

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.models.entities.parse_result import ExtractionMethod, ParserInfo
from docmirror.plugins.bank_statement.community_plugin import BANK_COLUMN_REGISTRY, BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import (
    SCANNED_POLICY,
    BankExtractionPolicy,
    BankExtractionRoute,
)
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.borderless_ocr import (
    detect_headers_relaxed,
    is_ocr_dominant,
    strict_header_match_count,
    table_is_borderless_ocr,
)

OCR_BORDERLESS_TABLE = [[
    ["个人客户交易明细", "", "", ""],
    ["账号", "6217001234567890", "", ""],
    ["值日", "交易说明", "发生金额", "账面余领"],
    ["2024-01-01", "工资入账", "5000.00", "8000.00"],
    ["2024-01-02", "转账支出", "200.00", "7800.00"],
    ["2024-01-03", "消费", "50.00", "7750.00"],
]]

CLEAN_GRID_TABLE = [[
    ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"],
    ["2024-01-01", "工资", "0.00", "5000.00", "8000.00"],
]]


class _ParseResultStub:
    def __init__(self, extraction_method: ExtractionMethod):
        self.parser_info = ParserInfo(extraction_method=extraction_method)
        self.logical_tables = []
        self.pages = []


def test_strict_header_match_fails_on_ocr_aliases():
    assert strict_header_match_count(OCR_BORDERLESS_TABLE, BANK_COLUMN_REGISTRY) < 3


def test_relaxed_header_detection_finds_columns():
    idx, headers, col_map = detect_headers_relaxed(OCR_BORDERLESS_TABLE, BANK_COLUMN_REGISTRY)
    assert idx == 2
    assert len(col_map) >= 2
    assert "date" in col_map
    assert headers


def test_table_is_borderless_ocr_shape():
    ctx = StyleContext(
        tables=OCR_BORDERLESS_TABLE,
        full_text="个人客户交易明细 中国工商银行",
        institution=None,
        page_count=1,
    )
    assert table_is_borderless_ocr(ctx) is True


def test_table_is_borderless_ocr_rejects_clean_grid():
    ctx = StyleContext(
        tables=CLEAN_GRID_TABLE,
        full_text="中国建设银行账户明细",
        institution="中国建设银行",
        page_count=1,
    )
    assert table_is_borderless_ocr(ctx) is False


def test_is_ocr_dominant_from_parse_result():
    ctx = StyleContext(
        tables=OCR_BORDERLESS_TABLE,
        full_text="",
        institution=None,
        page_count=1,
        parse_result=_ParseResultStub(ExtractionMethod.OCR),
    )
    assert is_ocr_dominant(ctx) is True


def test_detector_borderless_ocr_style():
    ctx = StyleContext(
        tables=OCR_BORDERLESS_TABLE,
        full_text="个人客户交易明细",
        institution=None,
        page_count=1,
        extraction_route=BankExtractionRoute.SCANNED,
        extraction_policy=SCANNED_POLICY,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style == "borderless_ocr"
    assert "borderless_ocr" in result.parser_chain


def test_registry_borderless_ocr_records():
    ctx = StyleContext(
        tables=OCR_BORDERLESS_TABLE,
        full_text="个人客户交易明细",
        institution=None,
        page_count=1,
        extraction_route=BankExtractionRoute.SCANNED,
        extraction_policy=SCANNED_POLICY,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3
    assert records[0]["normalized"].get("date") == "2024-01-01"
    assert records[0]["normalized"].get("amount") == pytest.approx(5000.0)


_BORDERLESS_SOURCE_HEADERS = [
    "值日",
    "收/支",
    "交易说明",
    "发生金额",
    "账面余领",
    "_source_page",
    "_source_table_id",
    "_source_row_index",
]
_BORDERLESS_BUSINESS_ROWS = [
    ["2024-01-01", "收人", "工资入账", "5000.00", "8000.00"],
    ["2024-01-02", "支山", "转账支出", "200.00", "7800.00"],
    ["2024-01-03", "支山", "消费", "50.00", "7750.00"],
    ["2024-01-04", "收人", "退款", "25.00", "7775.00"],
]


def _two_page_borderless_source_tables() -> list[list[list[str]]]:
    return [
        [
            list(_BORDERLESS_SOURCE_HEADERS),
            [*_BORDERLESS_BUSINESS_ROWS[0], "1", "ocr:p1", "0"],
            [*_BORDERLESS_BUSINESS_ROWS[1], "1", "ocr:p1", "1"],
        ],
        [
            list(_BORDERLESS_SOURCE_HEADERS),
            [*_BORDERLESS_BUSINESS_ROWS[2], "2", "ocr:p2", "0"],
            [*_BORDERLESS_BUSINESS_ROWS[3], "2", "ocr:p2", "1"],
        ],
    ]


def _run_isolated_borderless_strategy(tables: list[list[list[str]]]) -> list[dict]:
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.SCANNED,
        allowed_parser_ids=frozenset({"borderless_ocr"}),
    )
    ctx = StyleContext(
        tables=tables,
        full_text="个人客户交易明细 第1/2页 第2/2页",
        institution=None,
        page_count=2,
        parse_result=_ParseResultStub(ExtractionMethod.OCR),
        prefer_context_tables=True,
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    detection = StyleDetectionResult(
        primary_style="borderless_ocr",
        confidence=1.0,
        parser_chain=["borderless_ocr"],
    )
    records, _identity = BankStyleParserRegistry(adaptive=False).run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )
    return records


def test_borderless_strategy_conserves_two_page_raw_canonical_normalized_and_source_lineage() -> None:
    records = _run_isolated_borderless_strategy(_two_page_borderless_source_tables())

    assert len(records) == 4
    for record, source_values in zip(records, _BORDERLESS_BUSINESS_ROWS, strict=True):
        assert record["raw"] == dict(
            zip(_BORDERLESS_SOURCE_HEADERS[:5], source_values, strict=True)
        )
        assert record["canonical_raw"] == {
            "date": source_values[0],
            "direction": source_values[1],
            "summary": source_values[2],
            "amount": source_values[3],
            "balance": source_values[4],
        }

    assert [
        {
            key: record["normalized"][key]
            for key in ("date", "direction", "summary", "amount", "balance")
        }
        for record in records
    ] == [
        {
            "date": "2024-01-01",
            "direction": "income",
            "summary": "工资入账",
            "amount": 5000.0,
            "balance": 8000.0,
        },
        {
            "date": "2024-01-02",
            "direction": "expense",
            "summary": "转账支出",
            "amount": 200.0,
            "balance": 7800.0,
        },
        {
            "date": "2024-01-03",
            "direction": "expense",
            "summary": "消费",
            "amount": 50.0,
            "balance": 7750.0,
        },
        {
            "date": "2024-01-04",
            "direction": "income",
            "summary": "退款",
            "amount": 25.0,
            "balance": 7775.0,
        },
    ]
    assert [record["source"] for record in records] == [
        {
            "source_page": 1,
            "page_range": [1, 1],
            "table_id": "ocr:p1",
            "source_row_index": 0,
        },
        {
            "source_page": 1,
            "page_range": [1, 1],
            "table_id": "ocr:p1",
            "source_row_index": 1,
        },
        {
            "source_page": 2,
            "page_range": [2, 2],
            "table_id": "ocr:p2",
            "source_row_index": 0,
        },
        {
            "source_page": 2,
            "page_range": [2, 2],
            "table_id": "ocr:p2",
            "source_row_index": 1,
        },
    ]


def test_borderless_strategy_rejects_whole_plane_when_page_two_loses_required_amount_role() -> None:
    tables = deepcopy(_two_page_borderless_source_tables())
    tables[1][0][3] = "客户编号"

    # Dates, directions, balances, and source row identities still prove that
    # page two is transaction-like. The strategy may neither keep page one as a
    # prefix nor emit page-two records from a partially mapped column plane.
    assert _run_isolated_borderless_strategy(tables) == []
