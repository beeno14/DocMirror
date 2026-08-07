# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.enterprise_native.canonical_records import (
    ACCOUNT_ANNOTATION_DATASET,
    CREDIT_DETAIL_GROUP_DATASET,
    GROUPED_RESPONSIBILITY_DATASET,
    RECOVERED_BUSINESS_DATASET,
    extract_account_annotations,
    extract_canonical_enterprise_record_families,
    extract_credit_detail_groups,
    extract_grouped_repayment_responsibility_details,
    extract_recovered_business_accounts,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    build_canonical_enterprise_document,
)


def _document(*pages: PageContent):
    return build_canonical_enterprise_document(SimpleNamespace(pages=list(pages), confidence=1.0))


def test_canonical_record_families_reject_non_ir_input() -> None:
    with pytest.raises(TypeError, match="CanonicalEnterpriseDocumentIR"):
        extract_canonical_enterprise_record_families(SimpleNamespace())  # type: ignore[arg-type]


def test_credit_detail_group_headings_retain_reported_counts_without_duplicates() -> None:
    document = _document(
        PageContent(
            page_number=6,
            texts=[TextBlock(content="信贷记录明细\n被追偿业务共1笔\n未结清信贷\n中长期借款共2笔")],
            tables=[
                TableBlock(
                    table_id="long-loans",
                    metadata={"raw_rows": [["中长期借款", "", "共2笔"]]},
                )
            ],
        ),
        PageContent(
            page_number=7,
            texts=[TextBlock(content="贴现共3笔\n授信信息共1笔\n已结清信贷\n贴现共4笔")],
        ),
        PageContent(
            page_number=8,
            texts=[
                TextBlock(
                    content=(
                        "相关还款责任\n除贴现外的其他业务共5笔\n"
                        "贴现共6笔\n为担保交易承担的相关还款责任共7笔"
                    )
                )
            ],
        ),
    )

    records = extract_credit_detail_groups(document)

    assert [
        (row["group_phase"], row["business_category"], row["reported_record_count"])
        for row in records
    ] == [
        ("recovered", "被追偿业务", 1),
        ("active", "中长期借款", 2),
        ("active", "贴现", 3),
        ("active", "授信信息", 1),
        ("settled", "贴现", 4),
        ("repayment_responsibility", "除贴现外的其他业务", 5),
        ("repayment_responsibility", "贴现", 6),
        ("repayment_responsibility", "为担保交易承担的相关还款责任", 7),
    ]
    active_long = records[1]
    assert len(active_long["source_refs"]) == 2
    assert records[6]["represented_dataset"] == GROUPED_RESPONSIBILITY_DATASET
    assert CREDIT_DETAIL_GROUP_DATASET in extract_canonical_enterprise_record_families(document)


def test_credit_detail_group_conflicting_counts_remain_structured() -> None:
    document = _document(
        PageContent(
            page_number=6,
            texts=[TextBlock(content="未结清信贷\n中长期借款共2笔\n中长期借款共3笔")],
        )
    )

    records = extract_credit_detail_groups(document)

    assert len(records) == 1
    assert records[0]["reported_record_count"] == 2
    assert records[0]["reported_record_count_status"] == "conflict"
    assert records[0]["reported_record_count_conflicts"] == [2, 3]


def test_recovered_business_card_retains_both_rows_across_page_split() -> None:
    document = _document(
        PageContent(
            page_number=6,
            texts=[TextBlock(content="信贷记录明细\n被追偿业务共1笔")],
            tables=[
                TableBlock(
                    table_id="recovery-primary",
                    metadata={
                        "raw_rows": [
                            [
                                "账户编号",
                                "债权机构",
                                "业务种类",
                                "接收日期",
                                "币种",
                                "借款金额",
                                "余额",
                                "关闭日期",
                                "信息报告日期",
                            ],
                            [
                                "",
                                "五级分类",
                                "最近一次还款日期",
                                "最近一次还款总额",
                                "最近一次还款形式",
                                "历史表现",
                                "初始债权人名称",
                                "原债权种类",
                                "",
                            ],
                            [
                                "N101W5810H0001123",
                                "华融资产管理有限公司",
                                "资产处置",
                                "2014-01-01",
                                "人民币",
                                "50",
                                "38",
                                "--",
                                "2015-04-16",
                            ],
                        ]
                    },
                )
            ],
        ),
        PageContent(
            page_number=7,
            tables=[
                TableBlock(
                    table_id="recovery-secondary-continuation",
                    metadata={
                        "raw_rows": [
                            [
                                "",
                                "可疑",
                                "2015-04-15",
                                "2",
                                "正常还款",
                                "见附件",
                                "中国银行股份有限公司北京分行",
                                "贸易融资",
                                "",
                            ]
                        ]
                    },
                )
            ],
        ),
        PageContent(
            page_number=8,
            texts=[TextBlock(content="未结清信贷\n中长期借款共1笔")],
            tables=[
                TableBlock(
                    table_id="ordinary-credit-account",
                    metadata={
                        "raw_rows": [
                            [
                                "账户编号",
                                "授信机构",
                                "业务种类",
                                "开立日期",
                                "到期日",
                                "币种",
                                "借款金额",
                                "发放形式",
                            ],
                            [
                                "N101W5810H9999999",
                                "宁波银行股份有限公司北京分行",
                                "固定资产贷款",
                                "2014-01-01",
                                "2017-12-31",
                                "人民币",
                                "80",
                                "新增",
                            ],
                        ]
                    },
                )
            ],
        ),
    )

    records = extract_recovered_business_accounts(document)

    assert len(records) == 1
    record = records[0]
    assert record["account_identifier"] == "N101W5810H0001123"
    assert record["creditor_institution"] == "华融资产管理有限公司"
    assert record["business_type"] == "资产处置"
    assert record["receive_date"] == "2014-01-01"
    assert record["loan_amount"] == 50
    assert record["balance"] == 38
    assert record["close_date"] is None
    assert record["close_date_status"] == "not_reported"
    assert record["status"] == "active"
    assert record["snapshot_date"] == "2015-04-16"
    assert record["five_tier_class"] == "可疑"
    assert record["last_repayment_date"] == "2015-04-15"
    assert record["last_repayment_amount"] == 2
    assert record["repayment_method"] == "正常还款"
    assert record["history_status"] == "见附件"
    assert record["original_creditor_name"] == "中国银行股份有限公司北京分行"
    assert record["original_debt_type"] == "贸易融资"
    assert record["source_page"] == 6
    assert record["source_page_end"] == 7
    assert len(record["source_refs"]) == 2


def test_account_statements_and_dispute_are_individually_linked() -> None:
    document = _document(
        PageContent(
            page_number=6,
            texts=[
                TextBlock(
                    content=(
                        "①对于账户编号为“N101W5810H0001123”的业务，"
                        "华融资产管理有限公司于2014 年8 月12 日做出说明："
                        "该企业委托XX公司偿还贷款，因XX公司不按时还款导致出现多次逾期；"
                    )
                ),
                TextBlock(
                    content=(
                        "信息主体于2015 年9 月12 日提出声明：本企业委托XX公司偿还贷款，"
                        "因XX公司不按时还款导致出现多次逾期；该业务处于异议处理期。"
                    )
                ),
                TextBlock(content="未结清信贷"),
            ],
        )
    )

    records = extract_account_annotations(document)

    assert [record["annotation_type"] for record in records] == [
        "data_provider_statement",
        "subject_statement",
        "dispute_processing",
    ]
    assert {record["account_identifier"] for record in records} == {"N101W5810H0001123"}
    provider, subject, dispute = records
    assert provider["issuer"] == "华融资产管理有限公司"
    assert provider["annotation_date"] == "2014-08-12"
    assert provider["annotation_content"].endswith("导致出现多次逾期")
    assert subject["issuer"] == "信息主体"
    assert subject["annotation_date"] == "2015-09-12"
    assert subject["annotation_content"].startswith("本企业委托XX公司")
    assert dispute["annotation_date"] is None
    assert dispute["annotation_date_status"] == "not_applicable"
    assert dispute["annotation_content"] == "该业务处于异议处理期"
    assert dispute["dispute_status"] == "in_progress"


def test_grouped_responsibility_rows_cover_all_canonical_groups_and_continuation() -> None:
    borrowing_headers = [
        "责任类型",
        "保证合同编号",
        "还款责任金额",
        "授信机构",
        "业务种类",
        "五级分类",
        "账户数",
        "借款金额",
        "余额",
        "逾期总额",
        "逾期本金",
    ]
    guarantee_headers = [
        "责任类型",
        "保证合同编号",
        "还款责任金额",
        "授信机构",
        "业务种类",
        "五级分类",
        "账户数",
        "担保金额",
        "余额",
    ]
    document = _document(
        PageContent(
            page_number=8,
            texts=[TextBlock(content="其他借贷交易共2笔")],
            tables=[
                TableBlock(
                    table_id="borrowing-group",
                    metadata={
                        "raw_rows": [
                            borrowing_headers,
                            [
                                "其他",
                                "--",
                                "8.20",
                                "中国银行股份有限公司",
                                "流动资金贷款",
                                "正常",
                                "2",
                                "8.20",
                                "5",
                                "0",
                                "0",
                            ],
                        ]
                    },
                )
            ],
        ),
        PageContent(
            page_number=9,
            texts=[TextBlock(content="贴现共3笔")],
            tables=[
                TableBlock(
                    table_id="discount-group",
                    metadata={
                        "raw_rows": [
                            borrowing_headers,
                            [
                                "其他",
                                "--",
                                "15",
                                "中国银行股份有限公司",
                                "有追索权的银行承兑汇票贴现",
                                "次级",
                                "1",
                                "15",
                                "10",
                                "3",
                                "3",
                            ],
                            [
                                "保证人/反担保人",
                                "N101W1100H0051284",
                                "50",
                                "中国工商银行股份有限公司",
                                "有追索权的银行承兑汇票贴现",
                                "正常",
                                "2",
                                "50",
                                "30",
                                "0",
                                "0",
                            ],
                        ]
                    },
                )
            ],
        ),
        PageContent(
            page_number=10,
            texts=[TextBlock(content="为担保交易承担的相关还款责任共10笔")],
            tables=[
                TableBlock(
                    table_id="guarantee-group-start",
                    metadata={
                        "raw_rows": [
                            guarantee_headers,
                            [
                                "个人信贷",
                                "--",
                                "--",
                                "中国民生银行",
                                "贷款担保",
                                "正常",
                                "1",
                                "10",
                                "2",
                            ],
                        ]
                    },
                )
            ],
        ),
        PageContent(
            page_number=11,
            tables=[
                TableBlock(
                    table_id="guarantee-group-continuation",
                    metadata={
                        "raw_rows": [
                            [
                                "交易共同还款人/共同债务人",
                                "",
                                "",
                                "股份有限公司",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ],
                            [
                                "保证人/反担保人",
                                "N101W1100H0051284",
                                "100",
                                "中国民生银行股份有限公司",
                                "融资类银行保函",
                                "正常",
                                "3",
                                "10",
                                "8",
                            ],
                            [
                                "保证人/反担保人",
                                "N101W1100H065189",
                                "200",
                                "中国银行股份有限公司",
                                "银行承兑汇票",
                                "正常",
                                "4",
                                "50",
                                "38",
                            ],
                            [
                                "其他",
                                "--",
                                "10",
                                "中国银行股份有限公司",
                                "信用证",
                                "正常",
                                "2",
                                "15",
                                "6",
                            ],
                        ]
                    },
                )
            ],
        ),
    )

    records = extract_grouped_repayment_responsibility_details(document)

    assert len(records) == 7
    assert [record["transaction_group"] for record in records] == [
        "borrowing",
        "discount",
        "discount",
        "guarantee",
        "guarantee",
        "guarantee",
        "guarantee",
    ]
    assert records[0]["source_group_account_count"] == 2
    assert sum(record["account_count"] for record in records[1:3]) == 3
    assert records[1]["contract_number"] is None
    assert records[1]["contract_number_status"] == "not_reported"
    assert records[1]["loan_amount"] == 15
    first_guarantee = records[3]
    assert first_guarantee["responsibility_type"] == ("个人信贷交易共同还款人/共同债务人")
    assert first_guarantee["institution"] == "中国民生银行股份有限公司"
    assert first_guarantee["responsibility_amount"] is None
    assert first_guarantee["responsibility_amount_status"] == "not_reported"
    assert first_guarantee["guarantee_amount"] == 10
    assert first_guarantee["guarantee_amount_status"] == "reported"
    assert first_guarantee["loan_amount_status"] == "not_applicable"
    assert first_guarantee["overdue_total_status"] == "not_applicable"
    assert first_guarantee["source_page"] == 10
    assert first_guarantee["source_page_end"] == 11
    assert sum(record["account_count"] for record in records[3:]) == 10

    datasets = extract_canonical_enterprise_record_families(document)
    assert tuple(datasets) == (
        RECOVERED_BUSINESS_DATASET,
        ACCOUNT_ANNOTATION_DATASET,
        GROUPED_RESPONSIBILITY_DATASET,
        CREDIT_DETAIL_GROUP_DATASET,
    )
    assert datasets[GROUPED_RESPONSIBILITY_DATASET] == records
