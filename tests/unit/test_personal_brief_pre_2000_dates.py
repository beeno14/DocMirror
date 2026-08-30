# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_brief_native.date_rules import (
    normalize_personal_brief_date,
    normalize_personal_brief_month,
)
from docmirror.plugins.credit_report.personal_brief_native.extraction import (
    _asset_and_compensation_records,
    _postpaid_records,
)
from docmirror.plugins.credit_report.personal_brief_native.pipeline import (
    run_personal_brief_pipeline,
)
from docmirror.plugins.credit_report.personal_brief_native.projector import (
    derive_personal_brief_projection,
)
from tests.unit.test_personal_brief_canonical_pipeline import _result

HEADER = (
    "个人信用报告 报告编号：2026071900012345678901 "
    "报告时间：2026-07-19 09:08:07 姓名：张三 "
    "证件类型：身份证 证件号码：11010519491231002X"
)


def test_date_decoder_preserves_an_explicit_pre_2000_century() -> None:
    assert normalize_personal_brief_date("1998年1月2日") == "1998-01-02"
    assert normalize_personal_brief_date("1998/1/2") == "1998-01-02"
    assert normalize_personal_brief_month("1999年12月") == "1999-12"
    assert normalize_personal_brief_date("98年1月2日") == ""


def test_pre_2000_account_dates_survive_the_complete_public_pipeline() -> None:
    narrative = (
        "1998年12月31日中国银行股份有限公司北京分行发放的100,000元（人民币）"
        "个人住房贷款1999年12月31日到期。截至1999年11月，余额为0，"
        "当前无逾期。1999年12月已结清。"
    )
    result = _result(HEADER, "信贷记录", "贷款", narrative)
    semantic = run_personal_brief_pipeline(result).semantic_document
    source_account = semantic.datasets["credit_accounts"][0]
    projection = derive_personal_brief_projection(
        SimpleNamespace(projector_id="test", domain_name="credit_report"),
        result,
    )
    public_account = projection.datasets["credit_accounts"][0]["normalized"]

    source_expected = {
        "open_date": "1998-12-31",
        "due_date": "1999-12-31",
        "contract_maturity_date": "1999-12-31",
        "information_as_of": "1999-11",
        "close_date": "1999-12",
        "termination_event_date": "1999-12",
    }
    public_expected = {
        **{key: value for key, value in source_expected.items() if key != "information_as_of"},
        "snapshot_date": "1999-11",
    }
    assert {field: source_account[field] for field in source_expected} == source_expected
    assert {field: public_account[field] for field in public_expected} == public_expected
    assert source_account["business_type"] == "个人住房贷款"
    assert semantic.dataset_completeness["credit_accounts"]["verified"] is True


def test_pre_2000_optional_credit_dates_are_not_dropped() -> None:
    text = (
        "资产处置信息"
        "1.1997年1月2日，示例资产管理公司接收债权，金额为50,000。"
        "截至1999年3月4日，余额为20,000，最近一次还款日期为1998年2月3日"
        "垫款信息"
        "1.1996年4月5日以来示例担保公司累计代偿金额10,000。1999年6月已结清。"
        "信用卡"
    )
    parse_result = _result(HEADER, text)

    assets, compensations = _asset_and_compensation_records(parse_result, text)

    assert {
        "disposition_date": assets[0]["disposition_date"],
        "snapshot_date": assets[0]["snapshot_date"],
        "last_repayment_date": assets[0]["last_repayment_date"],
    } == {
        "disposition_date": "1997-01-02",
        "snapshot_date": "1999-03-04",
        "last_repayment_date": "1998-02-03",
    }
    assert compensations[0]["compensation_start_date"] == "1996-04-05"
    assert compensations[0]["settlement_date"] == "1999-06"


def test_pre_2000_liability_and_inquiry_dates_reach_typed_records() -> None:
    liability = (
        "1.1998年1月2日，为李四（证件类型：身份证，证件号码："
        "110105194912310011）在示例商业银行办理的个人经营性贷款承担相关还款责任，"
        "责任人类型为保证人，相关还款责任金额100,000（保证合同编号：A1）。"
        "截至1999年2月，个人经营性贷款余额50,000。"
    )
    semantic = run_personal_brief_pipeline(
        _result(
            HEADER,
            "信贷记录",
            "相关还款责任信息",
            liability,
            "非信贷交易记录",
            "查询记录",
            "机构查询记录明细",
            "1 1999年12月31日 示例商业银行 贷款审批",
            "个人查询记录明细",
        )
    ).semantic_document

    liability_record = semantic.datasets["repayment_liability_records"][0]
    inquiry_record = semantic.datasets["inquiry_records"][0]
    assert liability_record["liability_date"] == "1998-01-02"
    assert liability_record["snapshot_date"] == "1999-02"
    assert inquiry_record["inquiry_date"] == "1999-12-31"


def test_pre_2000_postpaid_and_statement_dates_are_preserved() -> None:
    postpaid = _postpaid_records(
        [
            (1, "非信贷交易记录 后付费记录"),
            (
                1,
                "机构名称：示例电信公司 业务类型：固定电话 记账年月：1999年12月 "
                "业务开通日期：1998年1月2日 当前缴费状态：正常 当前欠费金额：0",
            ),
        ]
    )[0]
    semantic = run_personal_brief_pipeline(
        _result(
            HEADER,
            "机构说明",
            "1.说明内容：本人声明该账户已有异议。添加日期：1999年1月2日",
        )
    ).semantic_document

    assert postpaid["billing_month"] == "1999-12"
    assert postpaid["service_start_date"] == "1998-01-02"
    assert semantic.datasets["institution_statement_records"][0]["added_date"] == ("1999-01-02")
