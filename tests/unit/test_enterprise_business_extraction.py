# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_non_credit_history_datasets,
    extract_enterprise_report_default_amount_unit,
)
from docmirror.plugins.credit_report.enterprise_native.pipeline import run_enterprise_pipeline


def _table(table_id: str, rows: list[list[str]]) -> TableBlock:
    return TableBlock(table_id=table_id, metadata={"raw_rows": rows})


def _result(pages: list[PageContent]) -> SimpleNamespace:
    return SimpleNamespace(pages=pages, confidence=1.0)


def _normalized_rows(datasets: dict[str, list[dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [row["normalized"] for row in datasets[name]]


def test_personnel_continuation_inherits_trailing_table_metadata() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
                tables=[
                    _table(
                        "personnel-start",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事长", "甲", "身份证", "110101198001010011"],
                            ["董事", "乙", "身份证", "110101198001010022"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[
                    _table(
                        "personnel-continuation",
                        [
                            ["监事", "丙", "身份证", "110101198001010033"],
                            ["经理", "丁", "身份证", "110101198001010044"],
                            [
                                "信息来源机构：中国银行股份有限公司北京分行 更新日期：2014-8-12",
                                "",
                                "",
                                "",
                            ],
                        ],
                    ),
                    _table(
                        "relationships",
                        [
                            ["类型", "名称", "身份标识类型", "身份标识号码"],
                            ["实际控制人", "戊", "身份证", "110101198001010055"],
                            ["信息来源机构：乙银行 更新日期：2014-9-13", "", "", ""],
                        ],
                    ),
                ],
            ),
        ]
    )

    datasets = run_enterprise_pipeline(result).semantic_document.datasets
    personnel = _normalized_rows(datasets, "enterprise_key_personnel")
    relationships = _normalized_rows(datasets, "enterprise_relationships")

    assert [row["name"] for row in personnel] == ["甲", "乙", "丙", "丁"]
    assert {row["source_institution"] for row in personnel} == {
        "中国银行股份有限公司北京分行"
    }
    assert {row["update_date"] for row in personnel} == {"2014-08-12"}
    assert relationships[0]["source_institution"] == "乙银行"
    assert relationships[0]["update_date"] == "2014-09-13"


def test_personnel_without_footer_does_not_borrow_relationship_metadata() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
                tables=[
                    _table(
                        "personnel",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事长", "甲", "身份证", "110101198001010011"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[
                    _table(
                        "relationships",
                        [
                            ["类型", "名称", "身份标识类型", "身份标识号码"],
                            ["实际控制人", "乙", "身份证", "110101198001010022"],
                            ["信息来源机构：乙银行 更新日期：2014-9-13", "", "", ""],
                        ],
                    )
                ],
            ),
        ]
    )

    datasets = run_enterprise_pipeline(result).semantic_document.datasets
    personnel = _normalized_rows(datasets, "enterprise_key_personnel")[0]
    relationship = _normalized_rows(datasets, "enterprise_relationships")[0]

    assert "source_institution" not in personnel
    assert "update_date" not in personnel
    assert relationship["source_institution"] == "乙银行"
    assert relationship["update_date"] == "2014-09-13"


def test_repeated_personnel_header_accepts_an_adjacent_metadata_only_footer() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
                tables=[
                    _table(
                        "personnel-start",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事长", "甲", "身份证", "110101198001010011"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[
                    _table(
                        "personnel-repeated-header",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事", "乙", "身份证", "110101198001010022"],
                        ],
                    ),
                    _table(
                        "personnel-footer",
                        [
                            [
                                "信息来源机构：中国银行股份有限公司北京分行 更新日期：2014-8-12",
                                "",
                                "",
                                "",
                            ]
                        ],
                    ),
                ],
            ),
        ]
    )

    datasets = run_enterprise_pipeline(result).semantic_document.datasets
    personnel = _normalized_rows(datasets, "enterprise_key_personnel")

    assert [row["name"] for row in personnel] == ["甲", "乙"]
    assert {row["source_institution"] for row in personnel} == {
        "中国银行股份有限公司北京分行"
    }
    assert {row["update_date"] for row in personnel} == {"2014-08-12"}


def test_explicit_personnel_header_preserves_uncommon_identity_type() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
                tables=[
                    _table(
                        "personnel",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事", "甲", "外国人永久居留身份证", "A12345"],
                            [
                                "信息来源机构：中国银行 更新日期：2014-8-12",
                                "",
                                "",
                                "",
                            ],
                        ],
                    )
                ],
            )
        ]
    )

    personnel = _normalized_rows(
        run_enterprise_pipeline(result).semantic_document.datasets,
        "enterprise_key_personnel",
    )

    assert personnel == [
        {
            "sequence": 1,
            "role": "董事",
            "name": "甲",
            "identity_type": "外国人永久居留身份证",
            "identity_number": "A12345",
            "source_institution": "中国银行",
            "update_date": "2014-08-12",
        }
    ]


def test_headerless_personnel_continuation_preserves_uncommon_identity_type() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
                tables=[
                    _table(
                        "personnel-start",
                        [
                            ["职位", "姓名", "身份标识类型", "证件号码"],
                            ["董事长", "甲", "身份证", "110101198001010011"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[
                    _table(
                        "personnel-continuation",
                        [
                            ["董事", "乙", "外国人永久居留身份证", "A12345"],
                            [
                                "信息来源机构：中国银行 更新日期：2014-8-12",
                                "",
                                "",
                                "",
                            ],
                        ],
                    )
                ],
            ),
        ]
    )

    personnel = _normalized_rows(
        run_enterprise_pipeline(result).semantic_document.datasets,
        "enterprise_key_personnel",
    )

    assert [row["name"] for row in personnel] == ["甲", "乙"]
    assert personnel[1]["identity_type"] == "外国人永久居留身份证"
    assert personnel[1]["identity_number"] == "A12345"
    assert {row["source_institution"] for row in personnel} == {"中国银行"}
    assert {row["update_date"] for row in personnel} == {"2014-08-12"}


def test_non_credit_money_uses_report_default_and_explicit_table_override() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content=(
                            "企业信用报告（自主查询版）\n报告说明\n"
                            "12. 如无特别说明，本报告中的金额类数据项单位均为万元。"
                        )
                    )
                ],
            ),
            PageContent(
                page_number=2,
                texts=[TextBlock(content="非信贷记录明细")],
                tables=[
                    _table(
                        "utility-current",
                        [
                            [
                                "公用事业单位名称",
                                "业务类型",
                                "缴费状态",
                                "累计欠费金额",
                                "统计年月",
                                "查看过去24个月缴费情况",
                            ],
                            ["中国移动", "电信", "欠缴费用", "0.30", "2015-06", "见附件"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=3,
                texts=[TextBlock(content="公共记录明细\n住房公积金缴费记录")],
                tables=[
                    _table(
                        "housing-current",
                        [
                            [
                                "统计年月",
                                "初缴年月",
                                "职工人数",
                                "缴费基数（元）",
                                "最近一次缴费日期",
                                "缴至年月",
                                "缴费状态",
                                "累计欠费金额（元）",
                                "过去缴费情况状态",
                                "过去缴费情况月数",
                            ],
                            [
                                "2015-06",
                                "2010-01",
                                "10",
                                "15000",
                                "2015-06-30",
                                "2015-06",
                                "正常缴费",
                                "0",
                                "见附件",
                                "24",
                            ],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=4,
                texts=[TextBlock(content="附件\n公用事业历史缴费记录明细")],
                tables=[
                    _table(
                        "utility-history",
                        [
                            ["统计年月", "缴费状态", "本月应缴金额", "本月实缴金额", "累计欠费金额"],
                            ["2015-06", "欠缴费用", "0.10", "0", "0.30"],
                        ],
                    )
                ],
            ),
            PageContent(
                page_number=5,
                texts=[TextBlock(content="住房公积金历史缴费记录明细")],
                tables=[
                    _table(
                        "housing-history",
                        [
                            [
                                "统计年月",
                                "缴费状态",
                                "本月应缴金额（元）",
                                "本月实缴金额（元）",
                                "累计欠费金额（元）",
                            ],
                            ["2015-06", "正常缴费", "15000", "15000", "0"],
                        ],
                    )
                ],
            ),
        ]
    )

    datasets = run_enterprise_pipeline(result).semantic_document.datasets
    utility = _normalized_rows(datasets, "enterprise_public_utility_payment_records")[0]
    housing = _normalized_rows(datasets, "enterprise_public_housing_fund_payment_records")[0]
    utility_history = _normalized_rows(datasets, "enterprise_utility_payment_history")[0]
    housing_history = _normalized_rows(datasets, "enterprise_housing_fund_history")[0]

    assert (utility["cumulative_arrears"], utility["currency"], utility["amount_unit"]) == (
        "0.3",
        "CNY",
        "CNY_10K",
    )
    assert (housing["contribution_base"], housing["currency"], housing["amount_unit"]) == (
        "15000",
        "CNY",
        "CNY_1",
    )
    assert (utility_history["amount_due"], utility_history["amount_unit"]) == (
        "0.1",
        "CNY_10K",
    )
    assert (housing_history["amount_due"], housing_history["amount_unit"]) == (
        "15000",
        "CNY_1",
    )


def test_non_credit_money_does_not_invent_a_unit_without_source_evidence() -> None:
    result = _result(
        [
            PageContent(
                page_number=1,
                texts=[TextBlock(content="公用事业历史缴费记录明细")],
                tables=[
                    _table(
                        "utility-history",
                        [
                            ["统计年月", "缴费状态", "本月应缴金额", "本月实缴金额", "累计欠费金额"],
                            ["2015-06", "欠缴费用", "0.10", "0", "0.30"],
                        ],
                    )
                ],
            )
        ]
    )

    rows = extract_enterprise_non_credit_history_datasets(result)
    row = rows["enterprise_utility_payment_history"][0]

    assert extract_enterprise_report_default_amount_unit(result) == ""
    assert "amount_unit" not in row
