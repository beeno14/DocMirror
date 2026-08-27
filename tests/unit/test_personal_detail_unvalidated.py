# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction-only contract tests. All source cells/pages below are synthetic.

No PDF, saved customer output, OCR engine, image, network service or private
fixture is used. Report-like layouts exercise the same entry/serialization
boundaries that production uses, not only pre-populated business dictionaries.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    KeyValuePair,
    PageContent,
    ParseResult,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.credit_report.community_plugin import CreditReportPlugin, _CreditReportCommunityBundle
from docmirror.plugins.credit_report.personal_detail_scanned.unvalidated import (
    CONTROL_DATASETS,
    VALIDATION_FIELDS,
    omit_validation,
)


def _table(rows: list[list[Any]], title: str = "", *, top: float = 100, table_id: str = "table") -> TableBlock:
    return TableBlock(
        table_id=table_id,
        caption=title or None,
        rows=[TableRow(cells=[CellValue(text=str(value), confidence=0.01) for value in row]) for row in rows],
        bbox=[0, top, 1000, top + 50],
        confidence=0.01,
    )


def _report(
    *tables: TableBlock, texts: list[TextBlock] | None = None, domain: dict[str, Any] | None = None
) -> ParseResult:
    return ParseResult(
        entities=DocumentEntities(document_type="credit_report", domain_specific=domain or {}),
        pages=[PageContent(page_number=1, page_mode="scanned", tables=list(tables), texts=texts or [])],
        raw_text="个人信用报告（本人版）",
    )


def _derive(result: ParseResult) -> ProjectionData:
    return CreditReportPlugin(unvalidated=True).derive(result, result.raw_text)


def _rows(projection: ProjectionData, dataset: str) -> list[dict[str, Any]]:
    return [row["normalized"] for row in projection.datasets.get(dataset, [])]


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("checked pipeline/repair must not run in extraction-only mode")


def test_hook_default_and_false_delegate_to_unchanged_checked_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    from docmirror.plugins.credit_report import projection

    expected = ProjectionData(projector_id="credit_report", domain_facts={"sentinel": "checked"})
    calls = []
    monkeypatch.setattr(projection, "derive_credit_report_projection", lambda *args: calls.append(args) or expected)
    result = _report()
    assert CreditReportPlugin().derive(result, result.raw_text) is expected
    assert CreditReportPlugin(unvalidated=False).derive(result, result.raw_text) is expected
    assert len(calls) == 2


@pytest.mark.parametrize("value", ["on", "off", "false", 0, 1, None, {}])
def test_hook_requires_a_real_boolean(value: Any) -> None:
    with pytest.raises(TypeError, match="boolean"):
        CreditReportPlugin(unvalidated=value)


@pytest.mark.parametrize(
    ("text", "module_name", "function"),
    [
        ("企业信用报告", "enterprise_native.projector", "derive_enterprise_projection"),
        ("个人信用报告 信贷记录", "personal_brief_native.projector", "derive_personal_brief_projection"),
    ],
)
def test_hook_does_not_change_other_report_variants(
    monkeypatch: pytest.MonkeyPatch, text: str, module_name: str, function: str
) -> None:
    from importlib import import_module

    expected = ProjectionData(projector_id="credit_report")
    module = import_module(f"docmirror.plugins.credit_report.{module_name}")
    monkeypatch.setattr(module, function, lambda *args: expected)
    assert CreditReportPlugin(unvalidated=True).derive(_report(), text) is expected


def test_unvalidated_bypasses_quality_repair_and_checked_output_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    from docmirror.plugins.credit_report import community_plugin, projection
    from docmirror.plugins.credit_report.personal_detail_scanned import context, schema

    monkeypatch.setattr(projection, "derive_credit_report_projection", _fail)
    monkeypatch.setattr(context, "build_personal_detail_extraction_context", _fail)
    monkeypatch.setattr(schema, "project_personal_detail_datasets", _fail)
    monkeypatch.setattr(schema, "_canonical_quality_gate", _fail)
    monkeypatch.setattr(community_plugin, "_compact_personal_detail_public_projection", _fail)
    monkeypatch.setattr(community_plugin, "_apply_personal_detail_dataset_status", _fail)
    result = _report(
        _table(
            [
                ["账户标识", "管理机构", "贷款金额", "账户状态"],
                ["bad-id", "测试银行", "6O00", "不可识别状态"],
            ],
            "非循环贷账户",
        )
    )
    bundle = CreditReportPlugin(unvalidated=True).project_bundle(seal_parse_result(result))
    assert bundle is not None
    row = next(dataset for dataset in bundle.json_payload()["datasets"] if dataset["name"] == "credit_accounts")[
        "rows"
    ][0]
    assert row["normalized"]["account_identifier"] == "bad-id"
    assert row["normalized"]["loan_amount"] == "6O00"
    assert row["normalized"]["account_state"] == "不可识别状态"


def test_report_header_and_query_retain_malformed_id_and_low_confidence() -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["报告编号", "报告时间"],
                    ["R-OCR-?", "2025年02月31日 09:02:03"],
                    ["被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"],
                    ["测试甲", "居民身份证", "broken-id", "测试银行", "贷款审批"],
                ]
            )
        )
    )
    metadata = _rows(projection, "report_metadata")
    assert len(metadata) == 1
    assert metadata[0]["primary_id_number"] == "broken-id"
    assert metadata[0]["report_time"] == "2025-02-31T09:02:03"
    query = _rows(projection, "report_query")[0]
    assert query["query_institution"] == "测试银行"
    assert query["query_reason"] == "贷款审批"
    assert "query_institution" not in metadata[0]


@pytest.mark.parametrize("number", ["0", "1,200.00", "1O0", "-50", "--", ""])
def test_scalar_decode_never_discards_observed_money(number: str) -> None:
    projection = _derive(_report(_table([["账户标识", "余额"], ["A-1", number]], "贷记卡账户")))
    row = projection.datasets["credit_accounts"][0]
    assert row["canonical_raw"]["balance"] == number
    assert row["normalized"]["balance"] == number.replace(",", "")


def test_unknown_column_and_missing_cell_do_not_shift_later_values() -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["序号", "查询日期", "查询机构", "查询原因", "额外印刷列"],
                    ["1", "2025-01-04", "", "贷款审批", "保留原文"],
                    ["2", "坏日期", "测试乙行"],
                ],
                "机构查询记录明细",
            )
        )
    )
    records = projection.datasets["inquiries"]
    assert len(records) == 2
    assert records[0]["normalized"]["institution"] == ""
    assert records[0]["normalized"]["reason"] == "贷款审批"
    assert records[0]["raw"]["额外印刷列"] == "保留原文"
    assert records[1]["normalized"]["inquiry_date"] == "坏日期"
    assert records[1]["normalized"]["reason"] == ""


def test_repeated_identical_rows_keep_distinct_source_records() -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["查询日期", "查询机构", "查询原因"],
                    ["2025-01-01", "测试银行", "贷款审批"],
                    ["2025-01-01", "测试银行", "贷款审批"],
                ],
                "查询记录",
            )
        )
    )
    records = projection.datasets["inquiries"]
    assert len(records) == 2
    assert len({row["record_id"] for row in records}) == 2


@pytest.mark.parametrize(
    ("title", "code"),
    [
        ("非循环贷账户", "D1"),
        ("循环贷账户（一）", "R1"),
        ("循环贷账户（二）", "R2"),
        ("贷记卡账户", "R3"),
        ("准贷记卡账户", "R4"),
    ],
)
def test_existing_account_family_mapping_is_reused(title: str, code: str) -> None:
    projection = _derive(_report(_table([["账户标识", "管理机构"], ["A-1", "测试银行"]], title)))
    assert _rows(projection, "credit_accounts")[0]["pboc_account_type_code"] == code


@pytest.mark.parametrize("status", ["N", "*", "0", "/", "#", "??", "O", ""])
def test_monthly_grid_preserves_every_observed_status_without_neighbor_fill(status: str) -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["账户标识", "管理机构"],
                    ["A-1", "测试银行"],
                    ["年份", "1月", "2月", "3月"],
                    ["2025", "N", status, "*"],
                    ["逾期金额", "0", "6O0", ""],
                ],
                "非循环贷账户",
            )
        )
    )
    accounts = projection.datasets["credit_accounts"]
    monthly = _rows(projection, "credit_account_monthly_performance")
    assert [row["performance_month"] for row in monthly] == ["2025-01", "2025-02", "2025-03"]
    assert monthly[1]["status_code"] == status
    assert monthly[1]["status_amount"] == "6O0"
    assert {row["account_id"] for row in monthly} == {accounts[0]["record_id"]}


def test_two_account_cards_keep_their_own_months_and_do_not_infer_missing_columns() -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["账户标识", "管理机构"],
                    ["A-1", "测试甲行"],
                    ["年份", "3", "7"],
                    ["2024", "N", "*"],
                    ["账户标识", "管理机构"],
                    ["A-2", "测试乙行"],
                    ["年份", "3", "7"],
                    ["2025", "??", "N"],
                ],
                "贷记卡账户",
            )
        )
    )
    accounts = projection.datasets["credit_accounts"]
    monthly = _rows(projection, "credit_account_monthly_performance")
    assert len(accounts) == 2
    assert [row["account_id"] for row in monthly] == [accounts[0]["record_id"]] * 2 + [accounts[1]["record_id"]] * 2
    assert {row["performance_month"] for row in monthly} == {"2024-03", "2024-07", "2025-03", "2025-07"}


def test_orphan_monthly_grid_is_not_rejected_or_given_a_fabricated_account() -> None:
    projection = _derive(_report(_table([["年份", "1", "2"], ["2025", "N", "O"]], "还款记录")))
    monthly = _rows(projection, "credit_account_monthly_performance")
    assert len(monthly) == 2
    assert not any("account_id" in row for row in monthly)
    assert "credit_accounts" not in projection.datasets


def test_profile_and_optional_sections_are_extracted_from_report_like_tables() -> None:
    projection = _derive(
        _report(
            _table(
                [["性别", "出生日期", "婚姻状况", "学历"], ["男", "bad-date", "未婚", "本科"]], "个人基本信息", top=10
            ),
            _table([["姓名", "证件号码", "联系电话"], ["测试配偶", "bad-id", "13O-invalid"]], "配偶信息", top=100),
            _table(
                [["手机号码", "信息更新日期"], ["13O-invalid", "2025.02.03"], ["13800000000", "2025.02.04"]],
                "手机号码历史",
                top=200,
            ),
            _table([["居住地址", "住宅电话", "居住状况"], ["测试市测试路1号", "--", "自置"]], "居住信息", top=300),
            _table([["工作单位", "单位性质", "职务"], ["测试公司", "民营", "员工"]], "职业信息", top=400),
        )
    )
    assert _rows(projection, "subject_profile")[0]["birth_date"] == "bad-date"
    assert _rows(projection, "subject_spouse")[0]["document_number"] == "bad-id"
    assert len(_rows(projection, "subject_mobile_phones")) == 2
    assert _rows(projection, "subject_residences")[0]["residential_phone"] == "--"
    assert _rows(projection, "subject_employment")[0]["employer"] == "测试公司"


def test_agreement_liability_postpaid_public_summary_and_notes() -> None:
    projection = _derive(
        _report(
            _table(
                [["授信协议标识", "管理机构", "授信额度", "已用额度"], ["AG?", "测试银行", "10,000", "2O00"]],
                "授信协议信息",
                top=10,
            ),
            _table(
                [
                    ["保证合同编号", "主业务借款人", "责任人类型", "还款责任金额", "还款状态"],
                    ["G?", "测试借款人", "保证人", "3O00", "??"],
                ],
                "相关还款责任信息",
                top=100,
            ),
            _table(
                [
                    ["管理机构", "业务类型", "当前缴费状态", "当前欠费金额"],
                    ["测试运营商", "移动电话", "欠费", "5O"],
                    ["年份", "1", "2"],
                    ["2025", "N", "?"],
                ],
                "后付费记录",
                top=200,
            ),
            _table(
                [["主管税务机关", "欠税总额", "欠税统计日期"], ["测试税务局", "8O0", "2025-02-03"]], "欠税记录", top=300
            ),
            _table([["业务类型", "账户数", "余额"], ["贷款", "2", "1O00"]], "信息概要", top=400),
            _table([["这是一条完全合成的本人声明。"]], "本人声明", top=500),
        )
    )
    assert _rows(projection, "credit_agreements")[0]["account_identifier"] == "AG?"
    assert _rows(projection, "credit_agreements")[0]["used_limit"] == "2O00"
    assert _rows(projection, "repayment_responsibilities")[0]["repayment_status_code"] == "??"
    assert _rows(projection, "postpaid_accounts")[0]["payment_status"] == "欠费"
    assert [row["status_code"] for row in _rows(projection, "postpaid_monthly_performance")] == ["N", "?"]
    assert _rows(projection, "tax_arrears_records")[0]["arrears_amount"] == "8O0"
    assert len(_rows(projection, "credit_business_overview")) == 2
    assert _rows(projection, "annotation_statements")[0]["text"] == "这是一条完全合成的本人声明。"


def test_initial_ocr_cells_use_geometry_and_keep_an_empty_middle_slot() -> None:
    lines = [
        {"text": "个人基本信息", "bbox": [0, 5, 120, 15]},
        {"text": "性别", "bbox": [0, 30, 50, 40]},
        {"text": "出生日期", "bbox": [100, 30, 150, 40]},
        {"text": "婚姻状况", "bbox": [200, 31, 250, 41]},
        {"text": "男", "bbox": [0, 60, 50, 70]},
        {"text": "未婚", "bbox": [200, 61, 250, 71], "confidence": 0.001},
    ]
    projection = _derive(
        _report(domain={"_page_evidence_bundles": [{"page": 1, "local_structure_evidence": {"lines": lines}}]})
    )
    row = _rows(projection, "subject_profile")[0]
    assert row["gender"] == "男"
    assert row["birth_date"] == ""
    assert row["marital_status"] == "未婚"


def test_ocr_text_without_geometry_and_inline_key_values() -> None:
    result = _report(texts=[TextBlock(content="被查询者姓名：测试甲  被查询者证件号码：bad-id\n报告编号：R-1")])
    metadata = _rows(_derive(result), "report_metadata")[0]
    assert metadata["subject_name"] == "测试甲"
    assert metadata["primary_id_number"] == "bad-id"
    assert metadata["report_number"] == "R-1"


def test_page_key_values_and_numeric_zero_in_raw_matrix_are_preserved() -> None:
    result = _report()
    result.pages[0].key_values = [
        KeyValuePair(key="报告编号", value="R-1"),
        KeyValuePair(key="被查询者姓名", value="测试甲"),
    ]
    assert _rows(_derive(result), "report_metadata")[0]["subject_name"] == "测试甲"
    table = _table([], "贷记卡账户")
    table.metadata["raw_rows"] = [["账户标识", "余额"], ["A-1", 0]]
    assert _rows(_derive(_report(table)), "credit_accounts")[0]["balance"] == "0"


def test_mode_notice_only_in_logs_and_json_uses_the_normal_envelope(caplog: pytest.LogCaptureFixture) -> None:
    result = _report(_table([["账户标识", "管理机构", "账户状态"], ["A-1", "测试银行", "正常"]], "贷记卡账户"))
    sealed = seal_parse_result(result)
    before = sealed.integrity_fingerprint
    with caplog.at_level(logging.INFO):
        bundle = CreditReportPlugin(unvalidated=True).project_bundle(sealed)
    assert bundle is not None
    assert "unvalidated=on" in caplog.text
    assert "A-1" not in caplog.text and "测试银行" not in caplog.text
    payload = bundle.json_payload()
    semantic = bundle.semantic_payload()
    assert payload["schema"]["name"] == "docmirror.community"
    assert set(payload) == {"schema", "document", "sections", "datasets", "reading", "files"}
    assert "unvalidated" not in json.dumps(payload) and "unvalidated" not in json.dumps(semantic)
    for data in (payload, semantic):
        assert "warnings" not in data and "diagnostics" not in data
        assert not ({dataset["name"] for dataset in data["datasets"]} & CONTROL_DATASETS)
        for dataset in data["datasets"]:
            assert "completeness" not in dataset and "status" not in dataset
            assert not ({column["key"] for column in dataset["columns"]} & VALIDATION_FIELDS)
            assert all(
                {"record_id", "normalized", "raw", "canonical_raw", "source"} <= row.keys() for row in dataset["rows"]
            )
    assert sealed.integrity_fingerprint == before and sealed.verify_integrity()


def test_cleanup_is_copy_only_and_keeps_business_status_and_administrative_review() -> None:
    source = {
        "datasets": [
            {
                "name": "administrative_penalty_records",
                "status": "complete",
                "completeness": {"verified": True},
                "columns": [{"key": "confidence"}, {"key": "administrative_review_result"}],
                "rows": [
                    {
                        "review": {"status": "requires_review"},
                        "confidence": 0.1,
                        "normalized": {
                            "status": "active",
                            "payment_status": "欠费",
                            "case_status": "未结案",
                            "administrative_review_result": "维持",
                            "mapping_status": "unmapped",
                        },
                    }
                ],
            },
            {"name": "dataset_status", "rows": []},
        ],
        "warnings": [{"message": "validation"}],
        "files": {"dataset_audit_csv": "_audit.csv", "content_md": "content.md"},
    }
    original = deepcopy(source)
    result = omit_validation(source)
    assert source == original
    assert len(result["datasets"]) == 1
    dataset = result["datasets"][0]
    assert dataset["columns"] == [{"key": "administrative_review_result"}]
    assert dataset["rows"][0]["normalized"] == {
        "status": "active",
        "payment_status": "欠费",
        "case_status": "未结案",
        "administrative_review_result": "维持",
    }
    assert result["files"] == {"content_md": "content.md"}


def test_supplied_semantic_cannot_reintroduce_validation_and_is_not_mutated() -> None:
    bundle = CreditReportPlugin(unvalidated=True).project_bundle(
        seal_parse_result(_report(_table([["账户标识", "余额"], ["A-1", "0"]], "贷记卡账户")))
    )
    assert bundle is not None
    semantic = bundle.semantic_payload()
    semantic["warnings"] = [{"message": "injected assessment"}]
    semantic["datasets"][0]["completeness"] = {"verified": True}
    before = deepcopy(semantic)
    payload = bundle.json_payload(semantic)
    assert semantic == before
    assert "warnings" not in payload
    assert all("completeness" not in dataset for dataset in payload["datasets"])


def test_mode_is_instance_local_and_does_not_mutate_sources_or_dictionaries() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import personal_detail_data_dictionary

    result = _report(_table([["账户标识", "管理机构"], ["A-1", "测试银行"]], "贷记卡账户"))
    before = result.model_dump()
    dictionary = personal_detail_data_dictionary()
    first = _derive(result)
    second = _derive(result)
    assert first == second
    assert before == result.model_dump()
    assert dictionary == personal_detail_data_dictionary()
    assert CreditReportPlugin().unvalidated is False
    with pytest.raises(AttributeError):
        CreditReportPlugin(unvalidated=True).unvalidated = False


def test_stale_checked_collections_are_not_mixed_into_fresh_output() -> None:
    result = _report(
        _table([["账户标识", "管理机构"], ["A-1", "测试银行"]], "贷记卡账户"),
        domain={
            "inquiries": [{"institution": "stale checked value"}],
            "dataset_status": [{"presence_status": "failed"}],
            "credit_extraction_audit": "old",
        },
    )
    bundle = CreditReportPlugin(unvalidated=True).project_bundle(seal_parse_result(result))
    assert bundle is not None
    payload = bundle.json_payload()
    assert {dataset["name"] for dataset in payload["datasets"]} == {"credit_accounts"}
    assert "stale checked value" not in json.dumps(payload)


def test_default_bundle_still_publishes_validation_fields() -> None:
    # The same serializer class remains checked unless the hook sets its private
    # per-bundle switch. This also guards against a global cleaning side effect.
    bundle = _CreditReportCommunityBundle(
        schema={"domain": "personal_credit_report_detailed"},
        document={},
        sections=[],
        datasets=[],
        files={},
        warnings=[{"code": "EXAMPLE", "level": "warning", "message": "checked"}],
        result=_report(),
        source_fingerprint="synthetic",
    )
    assert "warnings" in bundle.json_payload()


@pytest.mark.parametrize(
    ("label", "field_name", "raw", "expected"),
    [
        ("借款金额", "loan_amount", "2O00", "2O00"),
        ("账户授信额度", "credit_limit", "20,000", "20000"),
        ("账户币种", "account_currency", "人民币", "CNY"),
        ("币种", "account_currency", "美元", "USD"),
        ("币种", "account_currency", "坏币种", "坏币种"),
        ("业务种类", "business_type", "个人消费贷款", "个人消费贷款"),
        ("剩余还款期数", "remaining_periods", "12", 12),
        ("剩余还款期数", "remaining_periods", "1O", "1O"),
        ("逾期31—60天未还本金", "overdue_principal_31_60", "50", "50"),
        ("未出单的大额专项分期余额", "unbilled_installment_balance", "1,000", "1000"),
    ],
)
def test_native_printed_account_labels_remain_supported(label: str, field_name: str, raw: str, expected: Any) -> None:
    projection = _derive(_report(_table([["账户标识", label], ["A-1", raw]], "非循环贷账户")))
    assert _rows(projection, "credit_accounts")[0][field_name] == expected
    assert projection.datasets["credit_accounts"][0]["canonical_raw"][field_name] == raw


def test_month_grid_continuation_keeps_owner_until_a_new_business_section() -> None:
    result = _report(_table([["账户标识", "管理机构"], ["A-1", "测试银行"]], "非循环贷账户"))
    result.pages.extend(
        [
            PageContent(
                page_number=2,
                page_mode="scanned",
                tables=[
                    _table([["年份", "1", "2"], ["2025", "N", "*"]], "还款记录"),
                ],
            ),
            PageContent(
                page_number=3,
                page_mode="scanned",
                tables=[
                    _table(
                        [["查询日期", "查询机构", "查询原因"], ["2025-01-01", "测试银行", "贷款审批"]],
                        "查询记录",
                        top=10,
                    ),
                    _table([["年份", "1", "2"], ["2024", "N", "*"]], "还款记录", top=100),
                ],
            ),
        ]
    )
    projection = _derive(result)
    account_id = projection.datasets["credit_accounts"][0]["record_id"]
    monthly = _rows(projection, "credit_account_monthly_performance")
    assert len(monthly) == 4
    assert [row.get("account_id") for row in monthly] == [account_id, account_id, None, None]
    assert [row["source"]["page_range"] for row in projection.datasets["credit_account_monthly_performance"]] == [
        [2, 2]
    ] * 2 + [[3, 3]] * 2


def test_account_numeric_fields_are_not_mistaken_for_a_month_header() -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["账户标识", "余额", "还款期数", "当前逾期期数"],
                    ["A-1", "1000", "2", "3"],
                ],
                "非循环贷账户",
            )
        )
    )
    assert _rows(projection, "credit_accounts")[0]["repayment_periods"] == 2
    assert "credit_account_monthly_performance" not in projection.datasets


def test_unbound_source_is_logged_without_asserting_an_empty_complete_dataset(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        projection = _derive(_report(_table([["尚未绑定的表头"], ["未识别的业务值"]], "报告说明")))
    assert not projection.datasets
    assert "no label/column binding" in caplog.text
    assert "未识别的业务值" not in caplog.text


def test_empty_source_and_text_only_source_do_not_need_a_document_file() -> None:
    assert not _derive(_report()).datasets
    result = ParseResult(
        entities=DocumentEntities(document_type="credit_report"),
        raw_text="个人基本信息\n性别：男  出生日期：1990-01-02",
    )
    assert _rows(_derive(result), "subject_profile")[0]["gender"] == "男"


def test_sealed_input_requirement_is_not_weakened_by_the_hook() -> None:
    with pytest.raises(TypeError, match="SealedParseResult"):
        CreditReportPlugin(unvalidated=True).project_bundle(_report())


def test_stacked_profile_and_phone_tables_keep_their_own_fields() -> None:
    projection = _derive(
        _report(
            _table(
                [
                    ["性别", "出生日期", "婚姻状况"],
                    ["女", "1990-01-02", "已婚"],
                    ["手机号码", "信息更新日期"],
                    ["13800000000", "2025-01-01"],
                    ["居住地址", "住宅电话"],
                    ["测试市测试路1号", "010-00000000"],
                ],
                "个人基本信息",
            )
        )
    )
    assert _rows(projection, "subject_profile")[0]["birth_date"] == "1990-01-02"
    assert _rows(projection, "subject_mobile_phones")[0]["mobile_phone"] == "13800000000"
    assert _rows(projection, "subject_residences")[0]["address"] == "测试市测试路1号"


def test_report_metadata_split_between_text_and_table_is_one_card() -> None:
    projection = _derive(
        _report(
            _table([["被查询者姓名", "被查询者证件号码", "查询机构"], ["测试甲", "bad-id", "测试银行"]]),
            texts=[TextBlock(content="报告编号：R-1\n报告时间：2025-01-01 12:00:00", bbox=[0, 0, 500, 20])],
        )
    )
    metadata = _rows(projection, "report_metadata")
    assert len(metadata) == 1
    assert metadata[0]["report_number"] == "R-1"
    assert metadata[0]["primary_id_number"] == "bad-id"
    assert len(_rows(projection, "report_query")) == 1


@pytest.mark.parametrize("header", ["| |1月|2月|", "||1月|2月|", "\t1月\t2月"])
def test_text_calendar_preserves_empty_year_header_slot(header: str) -> None:
    delimiter = "\t" if "\t" in header else "|"
    status = delimiter.join(["2025", "N", "O"])
    result = _report(texts=[TextBlock(content=f"还款记录\n{header}\n{status}")])
    assert [row["performance_month"] for row in _rows(_derive(result), "credit_account_monthly_performance")] == [
        "2025-01",
        "2025-02",
    ]


def test_identity_profile_uses_printed_id_labels_without_id_validation() -> None:
    projection = _derive(
        _report(_table([["姓名", "证件类型", "证件号码"], ["测试甲", "居民身份证", "bad-id"]], "身份信息"))
    )
    assert _rows(projection, "subject_profile")[0]["primary_id_number"] == "bad-id"


@pytest.mark.parametrize("bbox", [[None, 0, 20, 10], [0, float("nan"), 20, 10], [0, "bad", 20, 10]])
def test_bad_ocr_geometry_falls_back_to_text_without_rejecting_fields(bbox: list[Any]) -> None:
    projection = _derive(
        _report(
            domain={
                "_page_evidence_bundles": [
                    {
                        "page": 1,
                        "local_structure_evidence": {
                            "lines": [{"text": "被查询者姓名：测试甲  被查询者证件号码：bad-id", "bbox": bbox}],
                        },
                    }
                ]
            }
        )
    )
    assert _rows(projection, "report_metadata")[0]["primary_id_number"] == "bad-id"


def test_extremely_long_integer_observation_does_not_crash_extraction() -> None:
    value = "9" * 5000
    projection = _derive(_report(_table([["账户标识", "还款期数"], ["A-1", value]], "非循环贷账户")))
    assert _rows(projection, "credit_accounts")[0]["repayment_periods"] == value


def test_headerless_inquiry_continuation_uses_preceding_columns() -> None:
    result = _report(
        _table(
            [
                ["序号", "查询日期", "查询机构", "查询原因"],
                ["1", "2025-01-01", "测试甲行", "贷款审批"],
            ],
            "机构查询记录明细",
        )
    )
    result.pages.append(
        PageContent(
            page_number=2,
            page_mode="scanned",
            tables=[
                _table([["2", "bad-date", "", "信用卡审批"], ["3", "2025-01-03", "测试乙行", "贷后管理"]]),
            ],
        )
    )
    projection = _derive(result)
    inquiries = _rows(projection, "inquiries")
    assert len(inquiries) == 3
    assert inquiries[1]["inquiry_date"] == "bad-date"
    assert inquiries[1]["institution"] == ""
    assert inquiries[1]["reason"] == "信用卡审批"
    assert inquiries[2]["institution"] == "测试乙行"
    assert projection.datasets["inquiries"][1]["source"]["page_range"] == [2, 2]


def test_header_carry_stops_at_a_different_section() -> None:
    projection = _derive(
        _report(
            _table([["查询日期", "查询机构", "查询原因"], ["2025-01-01", "测试银行", "贷款审批"]], "查询记录", top=10),
            _table([["2025-01-02", "不是查询记录", "不是查询原因"]], "报告说明", top=100),
        )
    )
    assert len(_rows(projection, "inquiries")) == 1
