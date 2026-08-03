# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_brief_native.extraction import (
    _pair_record_chunks,
    _personal_header_datasets,
    _postpaid_records,
)
from docmirror.plugins.credit_report.personal_brief_native.schema import (
    personal_brief_data_dictionary,
)


def test_personal_brief_metadata_emits_marital_status() -> None:
    blocks = [
        (
            1,
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X 已婚",
        ),
        (1, "信贷记录"),
    ]

    _identities, metadata, _amount_policy = _personal_header_datasets(
        SimpleNamespace(),
        blocks,
    )

    assert metadata[0]["marital_status"] == "married"
    assert (
        personal_brief_data_dictionary()["datasets"]["personal_report_metadata"]
        ["columns"]["marital_status"]["type"]
        == "enum"
    )


def test_personal_brief_metadata_preserves_extended_marital_status() -> None:
    blocks = [
        (
            1,
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X 婚姻状况：丧偶",
        ),
        (1, "信贷记录"),
    ]

    _identities, metadata, _amount_policy = _personal_header_datasets(SimpleNamespace(), blocks)

    assert metadata[0]["marital_status"] == "widowed"
    assert metadata[0]["marital_status_raw"] == "丧偶"
    assert personal_brief_data_dictionary()["enums"]["marital_status"]["widowed"] == "丧偶"


def test_postpaid_record_can_continue_on_the_next_page() -> None:
    blocks = [
        (1, "非信贷交易记录 后付费记录"),
        (1, "机构名称：中国电信 业务类型：固定电话 记账年月：2026年6月"),
        (1, "第1页/共2页"),
        (2, "业务开通日期：2020年1月1日 当前缴费状态：正常 当前欠费金额：0"),
        (2, "公共记录"),
    ]

    records = _postpaid_records(blocks)

    assert len(records) == 1
    assert records[0]["service_start_date"] == "2020-01-01"
    assert records[0]["current_arrears_amount"] == 0
    assert [ref["page"] for ref in records[0]["source_refs"]] == [1, 2]


def test_public_record_chunks_pair_by_identity_and_page_not_input_ordinal() -> None:
    anchors = [
        (3, "立案法院：甲法院 案号：A-1"),
        (4, "立案法院：乙法院 案号：B-2"),
    ]
    chunks = [
        (4, "立案法院：乙法院 案号：B-2 诉讼标的：乙事项"),
        (3, "立案法院：甲法院 案号：A-1 诉讼标的：甲事项"),
    ]

    paired = _pair_record_chunks(anchors, chunks, identity_labels=("立案法院", "案号"))

    assert [text for _page, text in paired] == [chunks[1][1], chunks[0][1]]
