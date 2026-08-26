# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for compact bank ledger parsing (bank layout)."""

from __future__ import annotations

from copy import deepcopy

from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import (
    BankExtractionPolicy,
    BankExtractionRoute,
)
from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles.compact_merged import (
    extract_compact_ledger_transactions,
    normalize_compact_transaction,
    parse_compact_ledger_cell,
    parse_counterparty_cell,
    resolve_amount_fields,
)


def test_parse_compact_ledger_cell_three_amounts():
    parsed = parse_compact_ledger_cell("2025-10-273765000.003765306.09")
    resolved = resolve_amount_fields(parsed, summary="电汇/服务费")
    assert resolved["date"] == "2025-10-27"
    assert resolved["expense"] == 3765000.00
    assert resolved["amount"] == 3765000.00
    assert resolved["direction"] == "expense"
    assert resolved["balance"] == 3765306.09


def test_parse_compact_ledger_cell_interest_two_amounts():
    parsed = parse_compact_ledger_cell("2025-09-210.04306.09")
    resolved = resolve_amount_fields(parsed, summary="结息")
    assert resolved["date"] == "2025-09-21"
    assert resolved["amount"] == 0.04
    assert resolved["direction"] == "income"
    assert resolved["balance"] == 306.09


def test_parse_counterparty_cell():
    account, name = parse_counterparty_cell("01041560012000235重庆正大能科科")
    assert account == "01041560012000235"
    assert name == "重庆正大能科科"


def test_extract_and_normalize_with_continuation_rows():
    table = [
        ["银座银行交易明细", "", ""],
        ["账号:651204680300015", "第1 / 1页", ""],
        ["日期支出收入余额", "对方账户对方户名", "摘要/附言"],
        ["2025-09-210.04306.09", "", "结息"],
        ["00:07:46", "", ""],
        ["2025-10-273765000.003765306.09", "01041560012000235重庆正大能科科", "电汇/服务费"],
        ["16:36:24", "技有限公司", ""],
    ]
    raws = extract_compact_ledger_transactions(table)
    assert len(raws) == 2

    first = normalize_compact_transaction(raws[0])
    assert first["date"] == "2025-09-21"
    assert first["timestamp"] == "2025-09-21 00:07:46"
    assert first["amount"] == 0.04
    assert first["summary"] == "结息"

    second = normalize_compact_transaction(raws[1])
    assert second["date"] == "2025-10-27"
    assert second["timestamp"] == "2025-10-27 16:36:24"
    assert second["amount"] == 3765000.00
    assert second["counter_account"] == "01041560012000235"
    assert "重庆正大能科" in second["counter_party"]
    assert "技有限公司" in second["counter_party"]


_COMPACT_SOURCE_HEADERS = [
    "日期支出收入余额",
    "对方账户对方户名",
    "摘要/附言",
    "_source_page",
    "_source_table_id",
    "_source_row_index",
]


def _two_page_compact_source_tables() -> list[list[list[str]]]:
    """Two physical compact-ledger pages with row-local lineage columns."""
    return [
        [
            list(_COMPACT_SOURCE_HEADERS),
            ["2025-09-210.04306.09", "", "结息", "1", "compact:p1", "0"],
            ["00:07:46", "", "", "1", "compact:p1", "1"],
        ],
        [
            list(_COMPACT_SOURCE_HEADERS),
            [
                "2025-10-273765000.003765306.09",
                "01041560012000235重庆正大能科",
                "电汇/服务费",
                "2",
                "compact:p2",
                "0",
            ],
            ["16:36:24", "技有限公司", "", "2", "compact:p2", "1"],
        ],
    ]


def _run_isolated_compact_strategy(tables: list[list[list[str]]]) -> list[dict]:
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.DIGITAL,
        allowed_parser_ids=frozenset({"compact_merged"}),
    )
    ctx = StyleContext(
        tables=tables,
        full_text="银座银行交易明细 第1/2页 第2/2页",
        institution=None,
        page_count=2,
        prefer_context_tables=True,
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    detection = StyleDetectionResult(
        primary_style="compact_merged_ledger",
        confidence=1.0,
        parser_chain=["compact_merged"],
    )
    records, _identity = BankStyleParserRegistry(adaptive=False).run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )
    return records


def test_compact_strategy_conserves_two_page_raw_canonical_normalized_and_source_lineage() -> None:
    records = _run_isolated_compact_strategy(_two_page_compact_source_tables())

    assert [record["raw"] for record in records] == [
        {
            "日期支出收入余额": "2025-09-210.04306.09",
            "对方账户对方户名": "",
            "摘要/附言": "结息",
        },
        {
            "日期支出收入余额": "2025-10-273765000.003765306.09",
            "对方账户对方户名": "01041560012000235重庆正大能科技有限公司",
            "摘要/附言": "电汇/服务费",
        },
    ]
    assert [
        {
            key: record["canonical_raw"][key]
            for key in ("date", "amount", "balance", "summary")
        }
        for record in records
    ] == [
        {"date": "2025-09-21", "amount": "0.04", "balance": "306.09", "summary": "结息"},
        {
            "date": "2025-10-27",
            "amount": "3765000.00",
            "balance": "3765306.09",
            "summary": "电汇/服务费",
        },
    ]
    assert records[1]["canonical_raw"]["counter_account"] == "01041560012000235"
    assert records[1]["canonical_raw"]["counter_party"] == "重庆正大能科技有限公司"
    assert [
        {
            key: record["normalized"][key]
            for key in (
                "date",
                "timestamp",
                "amount",
                "balance",
                "direction",
                "counter_account",
                "counter_party",
                "summary",
            )
        }
        for record in records
    ] == [
        {
            "date": "2025-09-21",
            "timestamp": "2025-09-21 00:07:46",
            "amount": 0.04,
            "balance": 306.09,
            "direction": "income",
            "counter_account": "",
            "counter_party": "",
            "summary": "结息",
        },
        {
            "date": "2025-10-27",
            "timestamp": "2025-10-27 16:36:24",
            "amount": 3_765_000.0,
            "balance": 3_765_306.09,
            "direction": "expense",
            "counter_account": "01041560012000235",
            "counter_party": "重庆正大能科技有限公司",
            "summary": "电汇/服务费",
        },
    ]
    assert [record["source"] for record in records] == [
        {
            "source_page": 1,
            "page_range": [1, 1],
            "table_id": "compact:p1",
            "source_row_index": 0,
        },
        {
            "source_page": 2,
            "page_range": [2, 2],
            "table_id": "compact:p2",
            "source_row_index": 0,
        },
    ]


def test_compact_strategy_rejects_whole_two_page_plane_when_page_two_row_is_unproved() -> None:
    tables = deepcopy(_two_page_compact_source_tables())
    # The second page remains transaction-shaped and source-addressable, but its
    # date grammar no longer proves a compact row. Keeping page one alone would
    # silently turn a document plane into a prefix.
    tables[1][1][0] = "2025/10/273765000.003765306.09"

    assert _run_isolated_compact_strategy(tables) == []
