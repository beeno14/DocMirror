# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_summary_datasets,
)
from docmirror.plugins.credit_report.enterprise_native.pipeline import (
    run_enterprise_pipeline,
)


def _table(table_id: str, rows: list[list[str]]) -> TableBlock:
    return TableBlock(table_id=table_id, metadata={"raw_rows": rows})


def test_displayed_credit_group_stops_at_printed_count_and_excludes_liability_group() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    _table(
                        "displayed-credit",
                        [
                            ["贴现", "", "共2笔", "", ""],
                            ["授信机构", "业务种类", "五级分类", "账户数", "余额"],
                            ["甲银行", "贴现", "正常", "1", "10"],
                            ["乙银行", "贴现", "正常", "1", "20"],
                            # Same-shaped rows from a neighbouring canonical
                            # group must not leak past the printed count.
                            ["丙银行", "银行承兑汇票", "正常", "1", "30"],
                            ["丁银行", "信用证", "正常", "1", "40"],
                        ],
                    ),
                ],
            ),
            PageContent(
                page_number=2,
                texts=[TextBlock(content="相关还款责任")],
                tables=[
                    _table(
                        "repayment-responsibility",
                        [
                            ["贴现", "", "共1笔", "", ""],
                            ["债权机构", "业务种类", "五级分类", "账户数", "余额"],
                            ["戊银行", "贴现", "正常", "1", "50"],
                        ],
                    ),
                ],
            )
        ]
    )

    rows = extract_enterprise_summary_datasets(result)[
        "enterprise_displayed_credit_summary"
    ]

    assert [(row["institution"], row["source_reported_amount"]) for row in rows] == [
        ("甲银行", 10),
        ("乙银行", 20),
    ]


def test_subset_pipeline_extracts_exchange_rate_from_report_notes_section() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）")],
            ),
            PageContent(
                page_number=2,
                texts=[
                    TextBlock(
                        content=(
                            "报告说明\n"
                            "1．示例说明。\n"
                            "汇率（美元折人民币）：6.83 有效期：2015-11"
                        )
                    )
                ],
            ),
            PageContent(
                page_number=3,
                texts=[TextBlock(content="身份标识")],
            ),
        ],
        confidence=1.0,
    )

    semantic = run_enterprise_pipeline(result).semantic_document
    rates = semantic.datasets["enterprise_exchange_rates"]

    assert len(rates) == 1
    assert rates[0]["normalized"]["exchange_rate_usd_cny"] == "6.83"
    assert rates[0]["normalized"]["exchange_rate_effective_period"] == "2015-11"


def test_subset_pipeline_keeps_cover_metadata_out_of_identity() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content=(
                            "企业信用报告（自主查询版）\n"
                            "NO.2025072809594736281877\n"
                            "企业名称：甲公司\n"
                            "中征码：1234567890123456\n"
                            "统一社会信用代码：91330100TEST000001\n"
                            "查询机构：甲银行\n"
                            "报告时间：2025-07-28T09:59:47"
                        )
                    )
                ],
            ),
            PageContent(page_number=2, texts=[TextBlock(content="报告说明")]),
            PageContent(
                page_number=3,
                texts=[TextBlock(content="身份标识")],
                tables=[
                    _table(
                        "identity",
                        [
                            ["企业名称", "甲公司"],
                            ["中征码", "1234567890123456"],
                            ["统一社会信用代码", "91330100TEST000001"],
                        ],
                    )
                ],
            ),
        ],
        confidence=1.0,
    )

    semantic = run_enterprise_pipeline(result).semantic_document
    metadata = semantic.datasets["enterprise_report_metadata"][0]["normalized"]
    identity = semantic.datasets["enterprise_report_identity"][0]["normalized"]

    assert metadata == {
        "query_institution": "甲银行",
        "report_edition": "independent_query",
        "report_number": "2025072809594736281877",
        "report_time": "2025-07-28T09:59:47",
        "sequence": 1,
        "source_page": 1,
    }
    assert semantic.facts["report_number"] == metadata["report_number"]
    assert semantic.facts["query_institution"] == metadata["query_institution"]
    assert semantic.facts["report_time"] == metadata["report_time"]
    assert not ({"report_number", "query_institution", "report_time", "report_edition"} & set(identity))


def test_subset_pipeline_routes_identity_dispute_notice_to_its_business_dataset() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n身份标识")],
            ),
            PageContent(
                page_number=2,
                texts=[TextBlock(content="该企业提出了3笔异议且正在处理中。\n信息概要")],
            ),
        ],
        confidence=1.0,
    )

    semantic = run_enterprise_pipeline(result).semantic_document
    dispute = semantic.datasets["enterprise_dispute_overview"][0]["normalized"]

    assert dispute["in_progress_dispute_count"] == 3
    assert dispute["dispute_status"] == "in_progress"
