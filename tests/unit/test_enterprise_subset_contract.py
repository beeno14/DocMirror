# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.enterprise_native.ir import (
    build_canonical_enterprise_document,
)
from docmirror.plugins.credit_report.enterprise_native.subset_contract import (
    CANONICAL_ENTERPRISE_SECTIONS,
    build_canonical_enterprise_subset,
)


def _table(table_id: str, rows: list[list[str]]) -> TableBlock:
    return TableBlock(table_id=table_id, metadata={"raw_rows": rows})


def test_closed_world_subset_partitions_same_text_block_and_skips_absent_sections() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content=(
                            "企业信用报告（自主查询版）\n"
                            "企业名称：甲公司\n"
                            "报告说明\n"
                            "本报告所展示的基本信息来自多个来源。\n"
                            "身份标识"
                        )
                    )
                ],
                tables=[_table("identity", [["企业名称", "甲公司"]])],
            ),
            PageContent(
                page_number=2,
                texts=[TextBlock(content="信息概要")],
                tables=[_table("overview", [["首次有信贷交易的年份"], ["2024"]])],
            ),
        ],
        confidence=1.0,
    )

    subset = build_canonical_enterprise_subset(build_canonical_enterprise_document(result))

    assert len(subset.sections) == len(CANONICAL_ENTERPRISE_SECTIONS)
    assert subset.section("report_metadata").heading_detected
    assert "企业名称：甲公司" in subset.section("report_metadata").text
    assert subset.section("report_notes").heading_detected
    assert "本报告所展示的基本信息" in subset.section("report_notes").text
    assert not subset.section("basic_information").heading_detected
    assert subset.section("identity").table_ids == ("identity",)
    assert subset.section("information_overview").table_ids == ("overview",)
    assert not subset.section("public_records").present


def test_section_view_keeps_cross_page_fragments_and_excludes_neighbours() -> None:
    result = SimpleNamespace(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n信贷记录明细")],
                tables=[
                    _table(
                        "credit_a",
                        [["账户编号", "授信机构", "业务种类", "开立日期", "到期日", "币种", "借款金额"]],
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[_table("credit_b", [["A123456789012", "甲银行", "贷款", "2024-01-01"]])],
            ),
            PageContent(
                page_number=3,
                texts=[TextBlock(content="公共记录明细")],
                tables=[_table("public", [["许可部门", "许可类型"], ["甲局", "许可"]])],
            ),
        ],
        confidence=1.0,
    )

    document = build_canonical_enterprise_document(result)
    subset = build_canonical_enterprise_subset(document)
    credit = subset.view("credit_details")
    public = subset.view("public_records")

    assert [table_id for _page, table_id, _rows in credit.table_rows] == [
        "credit_a",
        "credit_b",
    ]
    assert [fragment.index for fragment in credit.continuation_fragments] == [0, 1]
    assert [table_id for _page, table_id, _rows in public.table_rows] == ["public"]
    assert "credit_a" not in public.table_headings


def test_section_view_preserves_source_geometry_for_connected_heading_reconstruction() -> None:
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
                        content="附件1：信用记录补充信息\n中长期借款的历史表现",
                        bbox=[36.0, 40.0, 220.0, 80.0],
                    ),
                    TextBlock(
                        content="1.已结清账户编号：ACCOUNT000001",
                        bbox=[43.0, 100.0, 206.0, 132.0],
                    ),
                    TextBlock(
                        content="授信机构：甲银行",
                        bbox=[214.0, 100.0, 372.0, 121.0],
                    ),
                    TextBlock(
                        content="业务种类：流动资金贷款",
                        bbox=[385.0, 100.0, 501.0, 111.0],
                    ),
                ],
            ),
        ],
        confidence=1.0,
    )

    subset = build_canonical_enterprise_subset(build_canonical_enterprise_document(result))
    attachment = subset.view("attachment")
    segments = attachment.components[0].segments

    heading = next(segment for segment in segments if "账户编号" in str(segment.get("text") or ""))
    institution = next(segment for segment in segments if "授信机构" in str(segment.get("text") or ""))
    assert heading["bbox"] == [43.0, 100.0, 206.0, 132.0]
    assert institution["bbox"] == [214.0, 100.0, 372.0, 121.0]
