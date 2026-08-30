# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import PageContent, TableBlock, TextBlock
from docmirror.plugins.credit_report.enterprise_native import ir as ir_module
from docmirror.plugins.credit_report.enterprise_native.extraction_validation import (
    build_enterprise_extraction_report,
)
from docmirror.plugins.credit_report.enterprise_native.header_visual_recovery import (
    EnterpriseHeaderVisualRecovery,
    RecoveredEnterpriseHeaderField,
    _report_number_from_ocr_text,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    build_canonical_enterprise_document,
)
from docmirror.plugins.credit_report.enterprise_native.pipeline import run_enterprise_pipeline


def _table(table_id: str, rows: list[list[str]]) -> TableBlock:
    return TableBlock(table_id=table_id, metadata={"raw_rows": rows})


def _result(*pages: PageContent) -> SimpleNamespace:
    return SimpleNamespace(pages=list(pages), confidence=1.0)


def test_u1_report_number_candidate_requires_header_prefix_or_long_bounded_value() -> None:
    assert _report_number_from_ocr_text(
        "NO.2015110110051525534123",
        allow_bare=False,
    ) == "2015110110051525534123"
    assert _report_number_from_ocr_text(
        "2015110110051525534123",
        allow_bare=True,
    ) == "2015110110051525534123"
    assert _report_number_from_ocr_text(
        "91110102183797313",
        allow_bare=True,
    ) == ""


def test_u1_bounded_visual_report_number_enters_the_canonical_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered = EnterpriseHeaderVisualRecovery(
        fields=(
            RecoveredEnterpriseHeaderField(
                field_name="report_number",
                value="2015110110051525534123",
                source_page=1,
                bbox=(368.0, 146.0, 535.0, 167.0),
                confidence=0.99,
                source_text="NO.2015110110051525534123",
            ),
        )
    )
    monkeypatch.setattr(
        ir_module,
        "recover_enterprise_header_visual_fields",
        lambda _result, *, existing_text: recovered,
    )
    result = _result(
        PageContent(
            page_number=1,
            texts=[
                TextBlock(
                    content=(
                        "企业信用报告（自主查询版）\n"
                        "查询机构：中国某某银行北京分行\n"
                        "报告时间：2015-11-01T10:05:15"
                    )
                )
            ],
        )
    )

    artifacts = run_enterprise_pipeline(result)
    metadata = artifacts.semantic_document.datasets["enterprise_report_metadata"][0][
        "normalized"
    ]

    assert metadata["report_number"] == "2015110110051525534123"
    assert "报告编号：2015110110051525534123" in artifacts.document_ir.full_text
    assert any(
        unit.source_view == "bounded_header_visual_recovery"
        and unit.key == "报告编号"
        and unit.value == "2015110110051525534123"
        for unit in artifacts.document_ir.source_units
    )
    assert not [
        failure
        for failure in artifacts.semantic_document.extraction_report["failures"]
        if failure.get("field") == "report_number"
    ]


def test_u2_split_utility_history_header_and_value_are_bound_deterministically() -> None:
    result = _result(
        PageContent(
            page_number=1,
            texts=[TextBlock(content="企业信用报告（自主查询版）\n公共记录明细")],
            tables=[
                _table(
                    "utility",
                    [
                        [
                            "公用事业单位名称",
                            "业务类型",
                            "账户编号",
                            "缴费状态",
                            "累计欠费金额（元）",
                            "统计年月",
                            "查看过去24个月缴费情况",
                        ],
                        ["某供电公司", "电费", "U001", "正常", "0.30", "2015-09", "见附件"],
                    ],
                )
            ],
        )
    )

    record = run_enterprise_pipeline(result).semantic_document.datasets[
        "enterprise_public_utility_payment_records"
    ][0]

    assert record["normalized"]["history_period_months"] == 24
    assert record["normalized"]["history_status"] == "见附件"
    assert record["field_info"]["history_period_months"]["basis"] == (
        "bounded_header_value_pointer"
    )


def test_u3_relationship_type_has_stable_code_and_source_label() -> None:
    result = _result(
        PageContent(
            page_number=1,
            texts=[TextBlock(content="企业信用报告（自主查询版）\n基本信息")],
            tables=[
                _table(
                    "relationships",
                    [
                        ["类型", "名称", "身份标识类型", "身份标识号码"],
                        ["集团母公司", "甲集团", "中征码", "1000000000000001"],
                    ],
                )
            ],
        )
    )

    relationship = run_enterprise_pipeline(result).semantic_document.datasets[
        "enterprise_relationships"
    ][0]["normalized"]

    assert relationship["relationship_type"] == "group_parent_company"
    assert relationship["relationship_type_label"] == "集团母公司"


def test_u4_warns_when_positive_report_metadata_is_not_conserved() -> None:
    document = build_canonical_enterprise_document(
        _result(
            PageContent(
                page_number=1,
                texts=[
                    TextBlock(
                        content=(
                            "企业信用报告（自主查询版）\n"
                            "NO.2015110110051525534123\n"
                            "查询机构：中国某某银行北京分行\n"
                            "报告时间：2015-11-01T10:05:15"
                        )
                    )
                ],
            )
        )
    )
    report = build_enterprise_extraction_report(
        document,
        {},
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary={"datasets": {}, "enums": {}},
    ).to_payload()

    report_number = next(
        failure
        for failure in report["failures"]
        if failure.get("field") == "report_number"
    )
    assert report_number["code"] == "CANONICAL_POSITIVE_FIELD_NOT_CONSERVED"
    assert report_number["severity"] == "warning"
    assert report_number["path"].endswith("/report_number")
    assert report_number["evidence"]["source_value"] == "2015110110051525534123"


def test_u4_warns_when_split_history_business_fields_are_not_conserved() -> None:
    document = build_canonical_enterprise_document(
        _result(
            PageContent(
                page_number=1,
                tables=[
                    _table(
                        "utility",
                        [
                            ["公用事业单位名称", "查看过去24个月缴费情况"],
                            ["某供电公司", "见附件"],
                        ],
                    )
                ],
            )
        )
    )
    datasets = {
        "enterprise_public_utility_payment_records": [
            {
                "public_record_id": "utility:1",
                "normalized": {"utility_provider": "某供电公司"},
                "source_page": 1,
                "source_table_id": "utility",
                "source_refs": [
                    {"page": 1, "table_id": "utility", "row": 1}
                ],
            }
        ]
    }
    report = build_enterprise_extraction_report(
        document,
        datasets,
        continuation_audit=(),
        dataset_completeness={},
        data_dictionary={"datasets": {}, "enums": {}},
    ).to_payload()

    failures = {
        failure.get("field"): failure
        for failure in report["failures"]
        if failure["code"] == "CANONICAL_POSITIVE_FIELD_NOT_CONSERVED"
    }
    assert set(failures) == {"history_period_months", "history_status"}
    assert all(failure["severity"] == "warning" for failure in failures.values())
