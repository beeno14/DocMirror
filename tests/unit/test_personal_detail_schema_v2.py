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
from docmirror.plugins.credit_report.personal_detail_scanned.schema_v2 import (
    PBOC_DATASET_ORDER,
    PBOC_SCHEMA_VERSION_ENV,
    personal_detail_v2_data_dictionary,
    personal_detail_v2_enabled,
    personal_detail_v2_semantic_extensions,
    project_personal_detail_v2_datasets,
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
                year=2025,
                month=5,
                status="1",
                overdue_amount=1000,
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
                effective_date="2021-08",
                end_date="2024-07",
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

    projected = project_personal_detail_v2_datasets(source)

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
    assert monthly["account:loan"]["status_amount_semantics"] == "delinquent_amount"
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


def test_v2_dictionary_covers_pboc_catalog_and_logical_types() -> None:
    dictionary = personal_detail_v2_data_dictionary()

    assert dictionary["version"] == "2.0.0"
    assert set(PBOC_DATASET_ORDER) <= set(dictionary["datasets"])
    assert dictionary["logical_types"]["Month"]["json_type"] == "string"
    assert dictionary["logical_types"]["Long"]["json_type"] == "string"
    assert dictionary["datasets"]["housing_fund_records"]["columns"][
        "personal_contribution_ratio_percent"
    ]["logical_type"] == "Short"


def test_v2_is_opt_in_during_shadow_rollout(monkeypatch: Any) -> None:
    monkeypatch.delenv(PBOC_SCHEMA_VERSION_ENV, raising=False)
    assert personal_detail_v2_enabled() is False
    monkeypatch.setenv(PBOC_SCHEMA_VERSION_ENV, "2.0.0")
    assert personal_detail_v2_enabled() is True


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


def test_v2_schema_is_registered_alongside_frozen_v1() -> None:
    assert get_projection_schema("personal_credit_report_detailed").version == "1.2.0"
    assert get_projection_schema("personal_credit_report_detailed_v2").version == "2.0.0"
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
                    "personal_credit_report_detailed_v2.schema.json"
                ),
                "compatibility": "breaking-from-1.2; detailed-report-only",
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

    validation = validate_projection_payload("personal_credit_report_detailed_v2", payload)
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
    }
    semantic = personal_detail_v2_semantic_extensions()
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            "data_dictionary": personal_detail_v2_data_dictionary(),
        },
        "semantic": semantic,
        "datasets": project_personal_detail_v2_datasets(source),
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
    v2_validation = validate_projection_payload("personal_credit_report_detailed_v2", payload)
    assert v2_validation.valid, v2_validation.errors
