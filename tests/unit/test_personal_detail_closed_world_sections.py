from __future__ import annotations

import json
from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    bboxes = [
        [[column * 100, row * 20, (column + 1) * 100, (row + 1) * 20] for column in range(len(cells))]
        for row, cells in enumerate(rows)
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[0, 0, 1200, max(20, len(rows) * 20)],
        confidence=0.96,
        metadata={"raw_rows": rows, "source_cell_bboxes": bboxes},
    )


def _result(*tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=list(tables))]
    )


def _issues(result: SimpleNamespace) -> list[dict]:
    return list(getattr(result, "_personal_detail_extraction_issues", []) or [])


def test_public_rows_keep_physical_columns_and_report_missing_cell() -> None:
    result = _result(
        _table(
            "tax",
            [
                ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                ["1", "某市税务局", "", "2024-01-31"],
                ["2", "某区税务局", "500", "2024-02-29"],
            ],
        )
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 2
    first = json.loads(records[0]["content"])
    second = json.loads(records[1]["content"])
    assert "arrears_amount" not in first
    assert first["statistics_date"] == "2024-01-31"
    assert second["arrears_amount"] == 500
    assert second["statistics_date"] == "2024-02-29"
    assert any(
        issue.get("issue_code") == "candidate_b_public_record_cell_unresolved"
        and issue.get("field_name") == "arrears_amount"
        for issue in _issues(result)
    )


def test_public_two_part_civil_record_joins_by_printed_sequence() -> None:
    result = _result(
        _table(
            "civil",
            [
                ["编号", "立案法院", "案由", "立案日期", "结案方式"],
                ["1", "某法院", "合同纠纷", "2023-01-02", "判决"],
                ["编号", "判决/调解结果", "判决/调解生效日期", "诉讼标的", "诉讼标的金额"],
                ["1", "被告偿还借款", "2023-05-06", "借款", "10000"],
            ],
        )
    )

    records = native_extraction._extract_public_records(result)

    assert len(records) == 1
    content = json.loads(records[0]["content"])
    assert content["filing_court"] == "某法院"
    assert content["judgment_result"] == "被告偿还借款"
    assert content["claim_amount"] == 10000
    assert records[0]["source_refs_by_field"]["claim_amount"][0]["geometry_scope"] == "cell"


def test_postpaid_card_does_not_shift_after_blank_value() -> None:
    result = _result(
        _table(
            "postpaid",
            [
                ["机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"],
                ["某通信公司", "移动电话", "2020-01-02", "正常", "", "2024-06"],
            ],
        )
    )

    records = native_extraction._extract_postpaid_records(result)

    assert len(records) == 1
    assert records[0]["billing_month"] == "2024-06"
    assert "current_arrears_amount" not in records[0]
    assert records[0]["source_refs_by_field"]["billing_month"][0]["canonical_column"] == 5
    assert any(
        issue.get("target_record_id") == records[0]["postpaid_record_id"]
        and issue.get("field_name") == "current_arrears_amount"
        for issue in _issues(result)
    )


def test_postpaid_months_use_exact_header_columns_and_retain_bad_cell_as_issue() -> None:
    result = _result(
        _table(
            "postpaid-history",
            [
                ["机构名称", "业务类型", "记账年月"],
                ["某通信公司", "移动电话", "2024-06"],
                ["缴费记录", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                ["2024", "N", "N", "Q", "N", "N", "N", "N", "N", "N", "N", "N", "N"],
            ],
        )
    )

    records = native_extraction._extract_postpaid_payment_history(result)

    assert len(records) == 12
    march = next(record for record in records if record["month"] == 3)
    april = next(record for record in records if record["month"] == 4)
    assert "status" not in march
    assert april["status"] == "N"
    assert april["source_refs"][0]["canonical_column"] == 4
    assert any(
        issue.get("target_record_id") == march["postpaid_payment_history_id"]
        and issue.get("field_name") == "status"
        for issue in _issues(result)
    )


def test_note_table_keeps_blank_text_separate_from_date() -> None:
    result = _result(
        _table(
            "notes",
            [
                ["编号", "标注内容", "添加日期"],
                ["1", "", "2024-01-02"],
            ],
        )
    )

    annotations, statements = native_extraction._extract_personal_notes(result)

    assert statements == []
    assert len(annotations) == 1
    assert "text" not in annotations[0]
    assert annotations[0]["added_date"] == "2024-01-02"
    assert any(
        issue.get("target_dataset") == "annotation_statements"
        and issue.get("target_record_id") == f"annotation_statement:{annotations[0]['id']}"
        and issue.get("field_name") == "text"
        for issue in _issues(result)
    )


def test_recovery_card_preserves_later_slots_when_amount_is_blank() -> None:
    result = _result(
        _table(
            "recovery",
            [
                [
                    "管理机构",
                    "业务种类",
                    "债权接收日期",
                    "原债权人",
                    "原债务业务种类",
                    "债权金额",
                    "债权转移时的还款状态",
                    "账户状态",
                    "余额",
                    "最近一次还款日期",
                    "账户关闭日期",
                ],
                [
                    "某资产公司",
                    "资产处置",
                    "2022-01-02",
                    "某银行",
                    "个人贷款",
                    "",
                    "逾期",
                    "结清",
                    "0",
                    "2023-01-02",
                    "2023-02-03",
                ],
            ],
        )
    )

    records = native_extraction._extract_recovery_records(result)

    assert len(records) == 1
    assert "debt_amount" not in records[0]
    assert records[0]["account_status"] == "结清"
    assert records[0]["balance"] == 0
    assert any(
        issue.get("target_record_id") == records[0]["recovery_record_id"]
        and issue.get("field_name") == "debt_amount"
        for issue in _issues(result)
    )


def test_schema_withholds_unknown_public_type_without_generic_extension() -> None:
    projected = project_personal_detail_datasets(
        {
            "public_records": [
                {
                    "record_id": "public:unknown",
                    "public_record_id": "public:unknown",
                    "record_type": "future_public_type",
                    "content": '{"field_a":"alpha"}',
                }
            ]
        }
    )

    assert "pboc_extension_fields" not in projected
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row["issue_code"] == "canonical_public_record_type_unresolved"
    )
    assert issue["target_record_id"] == "public:unknown"
    assert "target_dataset" not in issue
    assert any(
        row.get("evidence_kind") == "observed" and row.get("string_value") == "alpha"
        for row in projected["extraction_issue_evidence"]
    )


def test_schema_removes_noncatalog_public_scalar_and_reports_it() -> None:
    projected = project_personal_detail_datasets(
        {
            "tax_arrears_records": [
                {
                    "record_id": "tax:1",
                    "tax_arrears_id": "tax:1",
                    "tax_authority": "某税务局",
                    "arrears_amount": 100,
                    "unmapped_content": "多个字段被错误拼接",
                }
            ]
        }
    )

    record = projected["tax_arrears_records"][0]
    assert "unmapped_content" not in record
    assert "unmapped_content" not in record.get("normalized", {})
    assert any(
        row["issue_code"] == "canonical_field_outside_closed_catalog"
        and row["field_name"] == "unmapped_content"
        for row in projected["extraction_issues"]
    )


def test_schema_withholds_known_event_extra_scalar_instead_of_extension() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {"record_id": "account:1", "account_id": "account:1", "account_type": "credit_card"}
            ],
            "personal_detail_account_events": [
                {
                    "record_id": "event:1",
                    "account_event_id": "event:1",
                    "account_id": "account:1",
                    "event_type": "special_event",
                    "details": "已知事件",
                    "future_scalar": "不应发布",
                }
            ],
        }
    )

    event = projected["credit_account_special_events"][0]
    assert event["details"] == "已知事件"
    assert "future_scalar" not in event
    assert "pboc_extension_fields" not in projected
    assert any(
        row["issue_code"] == "canonical_field_outside_closed_catalog"
        and row["field_name"] == "future_scalar"
        for row in projected["extraction_issues"]
    )


def test_schema_postpaid_month_requires_canonical_parent() -> None:
    projected = project_personal_detail_datasets(
        {
            "postpaid_payment_history": [
                {
                    "record_id": "month:1",
                    "postpaid_payment_history_id": "month:1",
                    "postpaid_record_id": "postpaid:missing",
                    "institution": "某通信公司",
                    "business_type": "移动电话",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                }
            ]
        }
    )

    month = projected["postpaid_monthly_performance"][0]
    assert month["postpaid_record_id"] is None
    assert any(
        row["issue_code"] == "unresolved_postpaid_parent_identity"
        and row["target_record_id"] == "month:1"
        for row in projected["extraction_issues"]
    )


def test_schema_postpaid_month_inherits_complete_parent_identity_silently() -> None:
    projected = project_personal_detail_datasets(
        {
            "postpaid_records": [
                {
                    "record_id": "postpaid:1",
                    "postpaid_record_id": "postpaid:1",
                    "institution": "某通信公司",
                    "business_type": "移动电话",
                    "billing_month": "2024-01",
                }
            ],
            "postpaid_payment_history": [
                {
                    "record_id": "month:1",
                    "postpaid_payment_history_id": "month:1",
                    "postpaid_record_id": "postpaid:1",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                }
            ],
        }
    )

    month = projected["postpaid_monthly_performance"][0]
    assert month["institution"] == "某通信公司"
    assert month["business_type"] == "移动电话"
    assert "extraction_issues" not in projected


def test_schema_postpaid_month_reports_incomplete_but_linked_parent_identity() -> None:
    projected = project_personal_detail_datasets(
        {
            "postpaid_records": [
                {
                    "record_id": "postpaid:partial",
                    "postpaid_record_id": "postpaid:partial",
                    "institution": "某通信公司",
                    "business_type": None,
                    "billing_month": "2024-01",
                }
            ],
            "postpaid_payment_history": [
                {
                    "record_id": "month:partial",
                    "postpaid_payment_history_id": "month:partial",
                    "postpaid_record_id": "postpaid:partial",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                }
            ],
        }
    )

    month = projected["postpaid_monthly_performance"][0]
    assert month["postpaid_record_id"] == "postpaid:partial"
    assert any(
        row["issue_code"] == "postpaid_parent_identity_incomplete"
        and row["target_record_id"] == "month:partial"
        for row in projected["extraction_issues"]
    )
