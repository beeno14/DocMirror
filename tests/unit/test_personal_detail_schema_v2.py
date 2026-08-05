# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PBOC-native personal detailed credit-report v2 tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.schemas.registry import (
    get_projection_schema,
    validate_projection_payload,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PBOC_DATASET_ORDER,
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)


def _record(record_id: str, **values: Any) -> dict[str, Any]:
    return {"record_id": record_id, **values}


def test_v2_projection_corrects_business_semantics_and_retains_unmapped_values() -> None:
    source = {
        "personal_report_metadata": [
            _record(
                "metadata:1",
                personal_report_metadata_id="metadata:1",
                report_number="2025052510051518624525",
                report_time="2025-05-25T10:05:15+08:00",
                query_institution="中国人民银行征信中心",
                query_reason="本人查询",
            )
        ],
        "personal_profile": [
            _record("profile:1", personal_profile_id="profile:1", birth_date="1981-08-15")
        ],
        "personal_detail_field_observations": [
            _record(
                "field:1",
                field_observation_id="field:1",
                dataset_name="personal_profile",
                business_record_id="profile:1",
                field_name="work_phone",
                observation_status="not_observed",
            )
        ],
        "credit_accounts": [
            _record(
                "account:loan",
                account_id="account:loan",
                account_type="non_revolving_loan",
            ),
            _record(
                "account:quasi",
                account_id="account:quasi",
                account_type="quasi_credit_card",
            ),
        ],
        "repayment_records": [
            _record(
                "monthly:loan",
                repayment_id="monthly:loan",
                account_id="account:loan",
                year=2025,
                month=5,
                status="1",
                overdue_amount=100,
            ),
            _record(
                "monthly:quasi",
                repayment_id="monthly:quasi",
                account_id="account:quasi",
                year="2025",
                month="05",
                status=1,
                overdue_amount="1,000.00",
            ),
        ],
        "repayment_liability_records": [
            _record(
                "responsibility:person",
                liability_id="responsibility:person",
                related_party_id_type="身份证",
                related_party_id_number="110101198108151111",
                overdue_months_or_repayment_status="2",
            ),
            _record(
                "responsibility:organization",
                liability_id="responsibility:organization",
                related_party_id_type="统一社会信用代码",
                related_party_id_number="91110000123456789X",
                overdue_months_or_repayment_status="N",
            ),
        ],
        "personal_housing_fund_records": [
            _record(
                "housing:1",
                personal_housing_fund_id="housing:1",
                first_contribution_month="2020-01",
                personal_contribution_ratio="12%",
                employer_contribution_ratio="12%",
            )
        ],
        "administrative_penalty_records": [
            _record(
                "penalty:1",
                administrative_penalty_id="penalty:1",
                effective_date="2021-08-31",
                end_date="2024/7/15",
            )
        ],
        "statements": [_record("statement:1", statement_id="statement:1", text="本人声明")],
        "personal_detail_credit_summary_metrics": [
            _record(
                "metric:1",
                credit_summary_metric_id="metric:1",
                summary_record_id="summary:1",
                row_index=1,
                column_index=1,
                metric_code="account_count",
                value_type="integer",
                integer_value=2,
            )
        ],
        "personal_detail_summary_cells": [
            _record(
                "cell:1",
                summary_cell_id="cell:1",
                summary_record_id="summary:1",
                row_index=1,
                column_index=1,
                column_label="账户数",
                value="2",
            ),
            _record(
                "cell:2",
                summary_cell_id="cell:2",
                summary_record_id="summary:1",
                row_index=1,
                column_index=2,
                column_label="未映射指标",
                value="业务值",
            ),
        ],
    }

    projected = project_personal_detail_datasets(source)

    assert set(projected) <= set(PBOC_DATASET_ORDER)
    assert "repayment_records" not in projected
    accounts = {row["account_id"]: row for row in projected["credit_accounts"]}
    assert accounts["account:loan"]["pboc_account_type_code"] == "D1"
    assert accounts["account:quasi"]["pboc_account_type_code"] == "R4"
    monthly = {
        row["account_id"]: row
        for row in projected["credit_account_monthly_performance"]
    }
    assert monthly["account:loan"]["performance_month"] == "2025-05"
    assert monthly["account:loan"]["status_amount"] == "100"
    assert monthly["account:loan"]["status_amount_semantics"] == "delinquent_amount"
    assert monthly["account:quasi"]["performance_month"] == "2025-05"
    assert monthly["account:quasi"]["status_code"] == "1"
    assert monthly["account:quasi"]["status_amount"] == "1000"
    assert monthly["account:quasi"]["status_amount_semantics"] == "overdraft_balance"
    assert "overdue_amount" not in monthly["account:quasi"]
    responsibilities = {
        row["related_party_category"]: row
        for row in projected["repayment_responsibilities"]
    }
    assert responsibilities["person"]["overdue_months"] == 2
    assert responsibilities["organization"]["repayment_status_code"] == "N"
    assert all(
        "overdue_months_or_repayment_status" not in row
        for row in responsibilities.values()
    )
    housing = projected["housing_fund_records"][0]
    assert housing["personal_contribution_ratio_percent"] == 12
    assert housing["employer_contribution_ratio_percent"] == 12
    penalty = projected["administrative_penalty_records"][0]
    assert penalty["effective_month"] == "2021-08"
    assert penalty["end_month"] == "2024-07"
    assert projected["annotation_statements"][0]["annotation_statement_group_id"]
    assert projected["pboc_extension_fields"][0]["value"] == "业务值"


def test_v2_routes_account_events_and_builds_schema_native_dataset_status() -> None:
    source = {
        "personal_detail_account_events": [
            _record(
                "event:latest",
                account_event_id="event:latest",
                account_id="account:1",
                event_type="latest_repayment",
            ),
            _record(
                "event:special",
                account_event_id="event:special",
                account_id="account:1",
                event_type="special_event_note",
                details="sample event",
            ),
            _record(
                "event:transaction",
                account_event_id="event:transaction",
                account_id="account:1",
                event_type="special_transaction",
                transaction_type="debt restructuring",
            ),
            _record(
                "event:unknown",
                account_event_id="event:unknown",
                account_id="account:1",
                event_type="future_event_type",
                details="future extension",
            ),
        ],
        "personal_detail_dataset_status": [
            _record(
                "status:events",
                dataset_status_id="status:events",
                dataset_name="personal_detail_account_events",
                applicability="applicable",
                presence_status="observed_nonempty",
                observed_row_count=4,
                reason="records_projected",
            ),
            _record(
                "status:statements",
                dataset_status_id="status:statements",
                dataset_name="statements",
                applicability="applicable",
                presence_status="explicitly_empty",
                observed_row_count=0,
                reason="source_explicitly_empty",
            ),
            _record(
                "status:annotations",
                dataset_status_id="status:annotations",
                dataset_name="annotations",
                applicability="applicable",
                presence_status="explicitly_empty",
                observed_row_count=0,
                reason="source_explicitly_empty",
            ),
        ],
    }

    projected = project_personal_detail_datasets(source)

    assert len(projected["credit_account_latest_repayments"]) == 1
    assert len(projected["credit_account_special_events"]) == 1
    assert len(projected["credit_account_special_transactions"]) == 1
    extensions = projected["pboc_extension_fields"]
    assert any(
        row["source_dataset"] == "personal_detail_account_events"
        and row["source_record_id"] == "event:unknown"
        for row in extensions
    )

    statuses = {row["dataset_name"]: row for row in projected["dataset_status"]}
    assert set(statuses) == set(PBOC_DATASET_ORDER) - {
        "dataset_status",
        "field_observations",
        "extraction_issues",
        "credit_account_latest_repayments",
        "credit_account_special_events",
        "credit_account_special_transactions",
        "annotation_statements",
        "annotation_statement_groups",
        "pboc_extension_fields",
    }
    assert len({row["dataset_status_record_id"] for row in statuses.values()}) == len(statuses)
    assert all(row["presence_status"] == "not_observed" for row in statuses.values())
    assert statuses["credit_card_large_installments"]["presence_status"] == "not_observed"
    assert "source_dataset_name" not in statuses["fraud_warnings"]
    assert not any(row["dataset_name"].startswith("personal_detail_") for row in statuses.values())


def test_v2_preserves_invalid_month_source_without_emitting_invalid_month() -> None:
    projected = project_personal_detail_datasets(
        {
            "administrative_penalty_records": [
                _record(
                    "penalty:invalid",
                    administrative_penalty_id="penalty:invalid",
                    effective_date="2021-19-01",
                )
            ],
            "postpaid_payment_history": [
                _record(
                    "postpaid:1",
                    postpaid_payment_history_id="postpaid:1",
                    year="2024",
                    month="7",
                    status=1,
                )
            ],
        }
    )

    penalty = projected["administrative_penalty_records"][0]
    assert "effective_month" not in penalty
    assert penalty["source_effective_date"] == "2021-19-01"
    postpaid = projected["postpaid_monthly_performance"][0]
    assert postpaid["performance_month"] == "2024-07"
    assert postpaid["status_code"] == "1"


def test_v2_preserves_sparse_uncertainty_as_typed_control_datasets() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_field_observations": [
                _record(
                    "field:1",
                    field_observation_id="field:1",
                    dataset_name="personal_profile",
                    business_record_id="personal_profile:primary",
                    field_name="work_phone",
                    observation_status="not_observed",
                ),
                _record(
                    "field:2",
                    field_observation_id="field:2",
                    dataset_name="datasets",
                    business_record_id="unresolved_record",
                    field_name="balance",
                    observation_status="ambiguous",
                ),
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:1",
                    extraction_issue_id="issue:1",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="open",
                    target_dataset="credit_lines",
                    field_name="facility_limit",
                )
            ],
        }
    )

    observation, unresolved = projected["field_observations"]
    issue = projected["extraction_issues"][0]
    assert observation["dataset_name"] == "subject_profile"
    assert observation["observation_status"] == "not_observed"
    assert unresolved["dataset_name"] == "unknown"
    assert unresolved["source_dataset_name"] == "datasets"
    assert issue["target_dataset"] == "credit_agreements"
    assert "pboc_extension_fields" not in projected


def test_v2_dictionary_covers_pboc_catalog_and_logical_types() -> None:
    dictionary = personal_detail_data_dictionary()

    assert dictionary["version"] == "2.0.0"
    assert set(PBOC_DATASET_ORDER) <= set(dictionary["datasets"])
    assert dictionary["logical_types"]["Month"]["json_type"] == "string"
    assert dictionary["logical_types"]["Long"]["json_type"] == "string"
    assert dictionary["datasets"]["housing_fund_records"]["columns"][
        "personal_contribution_ratio_percent"
    ]["logical_type"] == "Short"


def _canonical_dataset(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"ds_{name}",
        "name": name,
        "label": name,
        "type": name,
        "section_id": "sec_personal_basic",
        "csv": f"001_datasets/{name}.csv",
        "row_count": 1,
        "grain": f"one row per {name}",
        "primary_key": "record_id",
        "schema_version": "1.0",
        "storage_role": "canonical",
        "record_path": "rows",
        "status": "complete",
        "columns": [],
        "completeness": {
            "expected_row_count": 1,
            "emitted_row_count": 1,
            "omitted_row_count": 0,
            "verified": True,
            "basis": "unit_test",
        },
        "rows": [
            {
                "record_id": str(next(value for key, value in row.items() if key.endswith("_id"))),
                "normalized": row,
                "canonical_raw": dict(row),
                "raw": dict(row),
                "source": {},
            }
        ],
    }


def test_v2_schema_is_the_only_registered_detailed_personal_contract() -> None:
    assert get_projection_schema("personal_credit_report_detailed").version == "2.0.0"
    assert get_projection_schema("personal_credit_report_detailed_v2") is None
    payload = {
        "schema": {
            "name": "docmirror.community",
            "version": "3.0.0",
            "edition": "community",
            "domain": "personal_credit_report_detailed",
            "support_level": "beta",
        },
        "document": {
            "id": "document:1",
            "type": "personal_credit_report_detailed",
            "title": "个人信用报告",
            "page_count": 1,
            "language": ["zh-CN"],
            "source_file": {
                "name": "report.pdf",
                "mime_type": "application/pdf",
                "sha256": f"sha256:{'0' * 64}",
            },
            "units": {},
            "domain_schema": {
                "id": "personal_credit_report_detailed",
                "version": "2.0.0",
                "contract_uri": (
                    "https://valuemapglobal.github.io/DocMirror/schemas/"
                    "personal_credit_report_detailed.schema.json"
                ),
                "compatibility": "canonical-v2; detailed-report-only",
            },
        },
        "sections": [],
        "datasets": [
            _canonical_dataset(
                "report_metadata",
                {
                    "report_metadata_id": "metadata:1",
                    "report_number": "2025052510051518624525",
                    "report_time": "2025-05-25T10:05:15+08:00",
                },
            ),
            _canonical_dataset(
                "report_query",
                {
                    "report_query_id": "query:1",
                    "report_number": "2025052510051518624525",
                    "query_institution": "中国人民银行征信中心",
                    "query_reason": "本人查询",
                },
            ),
            _canonical_dataset(
                "subject_profile",
                {"subject_profile_id": "profile:1", "birth_date": "1981-08-15"},
            ),
        ],
        "reading": {},
        "files": {},
        "warnings": [],
    }

    validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert validation.valid, validation.errors


def test_v2_projection_builds_a_valid_community_bundle(tmp_path: Path) -> None:
    source = {
        "personal_report_metadata": [
            _record(
                "metadata:1",
                personal_report_metadata_id="metadata:1",
                report_number="2025052510051518624525",
                report_time="2025-05-25T10:05:15+08:00",
                query_institution="中国人民银行征信中心",
                query_reason="本人查询",
            )
        ],
        "personal_profile": [
            _record("profile:1", personal_profile_id="profile:1", birth_date="1981-08-15")
        ],
        "personal_detail_field_observations": [
            _record(
                "field:1",
                field_observation_id="field:1",
                dataset_name="personal_profile",
                business_record_id="profile:1",
                field_name="work_phone",
                observation_status="not_observed",
            )
        ],
        "credit_accounts": [
            _record(
                "account:1",
                account_id="account:1",
                sequence=1,
                account_type="quasi_credit_card",
            )
        ],
        "repayment_records": [
            _record(
                "monthly:1",
                repayment_id="monthly:1",
                account_id="account:1",
                year=2025,
                month=5,
                status="1",
                overdue_amount=1000,
            )
        ],
        "personal_detail_dataset_status": [
            _record(
                "status:metadata",
                dataset_status_id="status:metadata",
                dataset_name="personal_report_metadata",
                presence_status="observed_nonempty",
                observed_row_count=1,
            ),
            _record(
                "status:accounts",
                dataset_status_id="status:accounts",
                dataset_name="credit_accounts",
                presence_status="observed_nonempty",
                observed_row_count=1,
            ),
            _record(
                "status:repayments",
                dataset_status_id="status:repayments",
                dataset_name="repayment_records",
                presence_status="observed_nonempty",
                observed_row_count=1,
            ),
        ],
    }
    semantic = personal_detail_semantic_extensions()
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            "data_dictionary": personal_detail_data_dictionary(),
        },
        "semantic": semantic,
        "datasets": project_personal_detail_datasets(source),
        "sections": [
            {
                "id": "sec_personal_basic",
                "title": "个人基本信息",
                "type": "basic_information",
                "page_start": 1,
                "page_end": 1,
            }
        ],
    }
    result = ParseResult(
        entities=DocumentEntities(document_type="personal_credit_report_detailed"),
        pages=[PageContent(page_number=1)],
    )
    source_pdf = tmp_path / "report.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    payload = project_community_bundle(
        seal_parse_result(result),
        file_path=str(source_pdf),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    ).json_payload()

    assert payload["document"]["domain_schema"]["version"] == "2.0.0"
    assert validate_projection_payload("community", payload).valid
    v2_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert v2_validation.valid, v2_validation.errors
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    observation = datasets["field_observations"]["rows"][0]["normalized"]
    assert observation["dataset_name"] == "subject_profile"
    assert observation["source_dataset_name"] is None
    statuses = [row["normalized"] for row in datasets["dataset_status"]["rows"]]
    assert all(
        row["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for row in statuses
    )
    assert datasets["dataset_status"]["row_count"] == len(PBOC_DATASET_ORDER) - 9
