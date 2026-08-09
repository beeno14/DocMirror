# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Community-boundary tests for printed source-absence sentinels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.community_plugin import _CreditReportCommunityBundle
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)


def _record(record_id: str, **values: Any) -> dict[str, Any]:
    return {"record_id": record_id, **values}


def _issue(
    issue_id: str,
    *,
    dataset: str,
    record_id: str,
    field_name: str,
    observed_value: Any,
    issue_code: str = "candidate_b_exact_slot_value_invalid",
) -> dict[str, Any]:
    return _record(
        issue_id,
        extraction_issue_id=issue_id,
        category="ocr_cell_level_error",
        issue_code=issue_code,
        severity="warning",
        status="requires_review",
        parser_stage="candidate_b_test_fixture",
        target_dataset=dataset,
        target_record_id=record_id,
        field_name=field_name,
        observed_value=observed_value,
        reason_codes=["canonical_field_slot", "normalized_value_withheld"],
        message="Synthetic field-level fixture.",
    )


def _community_payload(tmp_path: Path, source: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    datasets = project_personal_detail_datasets(source)
    semantic = personal_detail_semantic_extensions()
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            "report_subtype": "personal_detail",
            "content_mode": "scanned_ocr",
            "data_dictionary": personal_detail_data_dictionary(),
            **{f"personal_detail_v2_expected_{name}_count": len(rows) for name, rows in datasets.items()},
        },
        "semantic": semantic,
        "datasets": datasets,
        "sections": [],
    }
    parse_result = ParseResult(
        entities=DocumentEntities(document_type="personal_credit_report_detailed"),
        pages=[PageContent(page_number=1)],
    )
    source_pdf = tmp_path / "source-absence.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    projected = project_community_bundle(
        seal_parse_result(parse_result),
        file_path=str(source_pdf),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    )
    return _CreditReportCommunityBundle(
        schema=projected.schema,
        document=projected.document,
        sections=projected.sections,
        datasets=projected.datasets,
        files=projected.files,
        warnings=projected.warnings,
        result=projected.result,
        source_fingerprint=projected.source_fingerprint,
        parse_result_schema=projected.parse_result_schema,
        classification=projected.classification,
        domain=projected.domain,
        diagnostics=projected.diagnostics,
        content_markdown_override=projected.content_markdown_override,
    ).json_payload()


def test_community_treats_printed_dashes_as_silent_source_absence(
    tmp_path: Path,
) -> None:
    absent_targets = {
        ("subject_profile", "profile:1", "degree"),
        ("subject_profile", "profile:1", "household_address"),
        ("subject_spouse", "spouse:1", "name"),
        ("subject_spouse", "spouse:1", "document_type"),
        ("subject_spouse", "spouse:1", "document_number"),
        ("subject_spouse", "spouse:1", "employer"),
        ("subject_spouse", "spouse:1", "phone"),
        ("subject_residences", "residence:1", "residential_phone"),
        ("credit_accounts", "account:absence", "repayment_method"),
        (
            "repayment_responsibilities",
            "liability:1",
            "overdue_months",
        ),
    }
    issue_codes = {
        ("subject_profile", "profile:1", "degree"): ("candidate_b_profile_contract_unresolved"),
        ("subject_profile", "profile:1", "household_address"): ("candidate_b_profile_contract_unresolved"),
        ("subject_spouse", "spouse:1", "document_type"): ("pboc_cell_contract_unresolved"),
        (
            "credit_accounts",
            "account:absence",
            "repayment_method",
        ): "candidate_b_account_cluster_field_unresolved",
        (
            "repayment_responsibilities",
            "liability:1",
            "overdue_months",
        ): "candidate_b_repayment_responsibility_field_invalid",
    }
    source: dict[str, list[dict[str, Any]]] = {
        "personal_profile": [
            _record(
                "profile:1",
                personal_profile_id="profile:1",
                degree="--",
                household_address="—",
                _source_absent_fields=["degree", "household_address"],
            )
        ],
        "spouse_records": [
            _record(
                "spouse:1",
                spouse_record_id="spouse:1",
                name="--",
                document_type="-",
                document_number="―",
                employer="---",
                phone="–",
                data_provider="示例银行",
                _source_absent_fields=[
                    "name",
                    "document_type",
                    "document_number",
                    "employer",
                    "phone",
                ],
            )
        ],
        "residence_records": [
            _record(
                "residence:1",
                residence_record_id="residence:1",
                residential_phone="－",
                _source_absent_fields=["residential_phone"],
            )
        ],
        "credit_accounts": [
            _record(
                "account:absence",
                account_id="account:absence",
                sequence=1,
                account_type="non_revolving_loan",
                guarantee_type="信用/免担保",
                repayment_method="--",
                _source_absent_fields=["repayment_method"],
            ),
            _record(
                "account:blank",
                account_id="account:blank",
                sequence=2,
                account_type="non_revolving_loan",
                repayment_method="",
            ),
        ],
        "repayment_records": [
            _record(
                "monthly:slash",
                repayment_id="monthly:slash",
                account_id="account:absence",
                year=2025,
                month=1,
                status="/",
            )
        ],
        "repayment_liability_records": [
            _record(
                "liability:1",
                liability_id="liability:1",
                sequence=1,
                overdue_months="--",
                _source_absent_fields=["overdue_months"],
            )
        ],
        "personal_detail_extraction_issues": [
            *[
                _issue(
                    f"issue:absence:{index}",
                    dataset=dataset,
                    record_id=record_id,
                    field_name=field_name,
                    observed_value=["--"],
                    issue_code=issue_codes.get(
                        (dataset, record_id, field_name),
                        "candidate_b_exact_slot_value_invalid",
                    ),
                )
                for index, (dataset, record_id, field_name) in enumerate(sorted(absent_targets), start=1)
            ],
            _issue(
                "issue:blank",
                dataset="credit_accounts",
                record_id="account:blank",
                field_name="repayment_method",
                observed_value=[""],
            ),
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    rows = {
        name: {row["record_id"]: row["normalized"] for row in dataset["rows"]} for name, dataset in datasets.items()
    }

    assert rows["subject_profile"]["profile:1"]["degree"] is None
    assert rows["subject_profile"]["profile:1"]["household_address"] is None
    spouse = rows["subject_spouse"]["spouse:1"]
    assert all(
        spouse[field_name] is None for field_name in ("name", "document_type", "document_number", "employer", "phone")
    )
    assert rows["subject_residences"]["residence:1"]["residential_phone"] is None
    account = rows["credit_accounts"]["account:absence"]
    assert account["repayment_method"] is None
    assert account["guarantee_type"] == "信用/免担保"
    assert rows["repayment_responsibilities"]["liability:1"]["overdue_months"] is None
    assert next(iter(rows["credit_account_monthly_performance"].values()))["status_code"] == "/"

    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]
    issue_targets = {
        (
            str(issue.get("target_dataset") or ""),
            str(issue.get("target_record_id") or ""),
            str(issue.get("field_name") or ""),
        )
        for issue in issues
    }
    assert not absent_targets & issue_targets
    assert (
        "credit_accounts",
        "account:blank",
        "repayment_method",
    ) in issue_targets

    observations = [row["normalized"] for row in datasets["field_observations"]["rows"]]
    observation_targets = {
        (
            str(observation.get("dataset_name") or ""),
            str(observation.get("business_record_id") or ""),
            str(observation.get("field_name") or ""),
        )
        for observation in observations
    }
    assert not absent_targets & observation_targets
    assert (
        "credit_accounts",
        "account:blank",
        "repayment_method",
    ) in observation_targets
