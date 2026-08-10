# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Community-boundary tests for printed source-absence sentinels."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.community_plugin import _CreditReportCommunityBundle
from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    _withhold_independent_plane_conflicts,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
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
        ("credit_accounts", "account:absence", "due_date"),
        ("credit_accounts", "account:absence", "shared_credit_limit"),
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
                due_date="-",
                repayment_method="--",
                shared_credit_limit="--",
                _source_absent_fields=[
                    "due_date",
                    "repayment_method",
                    "shared_credit_limit",
                ],
            ),
            _record(
                "account:blank",
                account_id="account:blank",
                sequence=2,
                account_type="non_revolving_loan",
                due_date="",
                repayment_method="",
                shared_credit_limit="",
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
            _issue(
                "issue:blank:due-date",
                dataset="credit_accounts",
                record_id="account:blank",
                field_name="due_date",
                observed_value=[""],
            ),
            _issue(
                "issue:blank:shared-limit",
                dataset="credit_accounts",
                record_id="account:blank",
                field_name="shared_credit_limit",
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
    assert account["due_date"] is None
    assert account["repayment_method"] is None
    assert account["shared_credit_limit"] is None
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
    assert (
        "credit_accounts",
        "account:blank",
        "due_date",
    ) in issue_targets
    assert (
        "credit_accounts",
        "account:blank",
        "shared_credit_limit",
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


def test_account_dash_absence_never_suppresses_substantive_competing_evidence() -> None:
    account_id = "account:competing-source-absence"
    competing_values = {
        "due_date": "2025-01-01",
        "repayment_method": "等额本息",
        "shared_credit_limit": "5000",
    }
    source = {
        "credit_accounts": [
            _record(
                account_id,
                account_id=account_id,
                sequence=1,
                account_type="non_revolving_loan",
                due_date=None,
                repayment_method=None,
                shared_credit_limit=None,
                canonical_raw={field_name: "--" for field_name in competing_values},
                _source_absent_fields=list(competing_values),
            )
        ],
        "personal_detail_extraction_issues": [
            _issue(
                f"issue:competing-source-absence:{field_name}",
                dataset="credit_accounts",
                record_id=account_id,
                field_name=field_name,
                observed_value=value,
                issue_code="candidate_b_account_cluster_field_unresolved",
            )
            for field_name, value in competing_values.items()
        ],
    }

    projected = project_personal_detail_datasets(source)

    account = projected["credit_accounts"][0]
    assert all(account[field_name] is None for field_name in competing_values)
    issues = projected["extraction_issues"]
    assert {
        (issue["target_record_id"], issue["field_name"], issue["observed_value"])
        for issue in issues
    } >= {
        (account_id, field_name, value)
        for field_name, value in competing_values.items()
    }
    observations = projected["field_observations"]
    assert {
        (observation["business_record_id"], observation["field_name"])
        for observation in observations
    } >= {(account_id, field_name) for field_name in competing_values}


def test_later_account_dash_preserves_prior_unresolved_raw_observation() -> None:
    account_id = "account:merged-source-absence"
    account = _record(
        account_id,
        account_id=account_id,
        sequence=1,
        account_type="non_revolving_loan",
        due_date=None,
        canonical_raw={"due_date": "2025-01-01"},
        _unresolved_fields=["due_date"],
    )

    native_extraction._mark_source_absent(account, "due_date", "--")

    assert account["canonical_raw"]["due_date"] == ["2025-01-01", "--"]
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [account],
            "personal_detail_extraction_issues": [
                _issue(
                    "issue:merged-source-absence:due-date",
                    dataset="credit_accounts",
                    record_id=account_id,
                    field_name="due_date",
                    observed_value="2025-01-01",
                    issue_code="candidate_b_account_cluster_field_unresolved",
                )
            ],
        }
    )

    assert projected["credit_accounts"][0]["due_date"] is None
    assert any(
        issue["target_record_id"] == account_id
        and issue["field_name"] == "due_date"
        and issue["observed_value"] == "2025-01-01"
        for issue in projected["extraction_issues"]
    )


def test_employment_and_agreement_absence_requires_direct_exact_dash_proof(
    tmp_path: Path,
) -> None:
    employment_cases = {
        "employment:clean": {
            "professional_title": "中级",
        },
        "employment:dash": {
            "professional_title": None,
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": "--"},
        },
        "employment:blank": {
            "professional_title": None,
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": ""},
        },
        "employment:stars": {
            "professional_title": None,
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": "**"},
        },
        "employment:mixed": {
            "professional_title": None,
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": ["--", "高级"]},
        },
        "employment:conflict": {
            "professional_title": None,
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": "--"},
        },
        "employment:no-ledger": {
            "professional_title": None,
            "canonical_raw": {"professional_title": "--"},
        },
        "employment:normalized-dash": {
            "professional_title": "--",
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": "--"},
        },
        "employment:collapsed": {
            "professional_title": None,
            "_source_absent_fields": ["professional_title"],
            "canonical_raw": {"professional_title": "--"},
        },
    }
    employment_issue_observations: dict[str, Any] = {
        "employment:dash": {
            "sequence": 2,
            "physical_cells": ["2", "示例单位", "--", "示例地址"],
        },
        "employment:blank": [""],
        "employment:stars": ["**"],
        "employment:mixed": ["--"],
        "employment:conflict": ["--", "高级"],
        "employment:no-ledger": ["--"],
        "employment:normalized-dash": ["--"],
        "employment:collapsed": {
            "raw_cluster": "商业、服务业人员 -",
            "unconsumed_residue": "-",
        },
    }

    agreement_cases = {
        "agreement:dash": {
            "total_limit": None,
            "_source_absent_fields": ["total_limit"],
            "canonical_raw": {"total_limit": "—"},
        },
        "agreement:blank": {
            "total_limit": None,
            "_source_absent_fields": ["total_limit"],
            "canonical_raw": {"total_limit": ""},
        },
        "agreement:stars": {
            "total_limit": None,
            "_source_absent_fields": ["total_limit"],
            "canonical_raw": {"total_limit": "**"},
        },
        "agreement:mixed": {
            "total_limit": None,
            "_source_absent_fields": ["total_limit"],
            "canonical_raw": {"total_limit": ["--", "5000"]},
        },
        "agreement:conflict": {
            "total_limit": None,
            "_source_absent_fields": ["total_limit"],
            "canonical_raw": {"total_limit": "--"},
        },
        "agreement:no-ledger": {
            "total_limit": None,
            "canonical_raw": {"total_limit": "--"},
        },
        "agreement:normalized-dash": {
            "total_limit": "--",
            "_source_absent_fields": ["total_limit"],
            "canonical_raw": {"total_limit": "--"},
        },
    }
    agreement_issue_observations: dict[str, Any] = {
        "agreement:dash": ["—"],
        "agreement:blank": [""],
        "agreement:stars": ["**"],
        "agreement:mixed": ["--"],
        "agreement:conflict": ["--", "5000"],
        "agreement:no-ledger": ["--"],
        "agreement:normalized-dash": ["--"],
    }

    source: dict[str, list[dict[str, Any]]] = {
        "employment_records": [
            _record(
                record_id,
                employment_record_id=record_id,
                sequence=index,
                **values,
            )
            for index, (record_id, values) in enumerate(
                employment_cases.items(), start=1
            )
        ],
        "credit_lines": [
            _record(
                record_id,
                credit_line_id=record_id,
                account_identifier=f"T10151210H0001ABC{index:05d}",
                sequence=index,
                **values,
            )
            for index, (record_id, values) in enumerate(
                agreement_cases.items(), start=1
            )
        ],
        "personal_detail_extraction_issues": [
            *[
                _issue(
                    f"issue:{record_id}",
                    dataset="employment_records",
                    record_id=record_id,
                    field_name="professional_title",
                    observed_value=observed_value,
                    issue_code=(
                        "candidate_b_employment_cluster_field_unresolved"
                        if record_id == "employment:collapsed"
                        else "candidate_b_employment_canonical_cell_unresolved"
                    ),
                )
                for record_id, observed_value in employment_issue_observations.items()
            ],
            *[
                _issue(
                    f"issue:{record_id}",
                    dataset="credit_lines",
                    record_id=record_id,
                    field_name="total_limit",
                    observed_value=observed_value,
                    issue_code="candidate_b_credit_agreement_required_field_unresolved",
                )
                for record_id, observed_value in agreement_issue_observations.items()
            ],
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    employment = {
        row["record_id"]: row for row in datasets["subject_employment"]["rows"]
    }
    agreements = {
        row["record_id"]: row for row in datasets["credit_agreements"]["rows"]
    }
    issue_targets = {
        (
            row["normalized"].get("target_dataset"),
            row["normalized"].get("target_record_id"),
            row["normalized"].get("field_name"),
        )
        for row in datasets["extraction_issues"]["rows"]
    }
    observation_targets = {
        (
            row["normalized"].get("dataset_name"),
            row["normalized"].get("business_record_id"),
            row["normalized"].get("field_name"),
        )
        for row in datasets["field_observations"]["rows"]
    }

    employment_dash_target = (
        "subject_employment",
        "employment:dash",
        "professional_title",
    )
    agreement_dash_target = (
        "credit_agreements",
        "agreement:dash",
        "facility_limit",
    )
    assert employment["employment:clean"]["normalized"]["professional_title"] == "中级"
    assert employment["employment:dash"]["normalized"]["professional_title"] is None
    assert agreements["agreement:dash"]["normalized"]["facility_limit"] is None
    assert employment["employment:dash"].get("canonical_raw", {}) == {}
    assert agreements["agreement:dash"].get("canonical_raw", {}) == {}
    assert employment_dash_target not in issue_targets
    assert employment_dash_target not in observation_targets
    assert agreement_dash_target not in issue_targets
    assert agreement_dash_target not in observation_targets

    expected_employment_failures = set(employment_issue_observations) - {
        "employment:dash"
    }
    assert {
        record_id
        for dataset_name, record_id, field_name in issue_targets
        if dataset_name == "subject_employment"
        and field_name == "professional_title"
        and record_id in employment_issue_observations
    } == expected_employment_failures
    expected_agreement_failures = set(agreement_issue_observations) - {
        "agreement:dash"
    }
    assert {
        record_id
        for dataset_name, record_id, field_name in issue_targets
        if dataset_name == "credit_agreements"
        and field_name == "facility_limit"
        and record_id in agreement_issue_observations
    } == expected_agreement_failures


def _spouse_absence_status() -> dict[str, Any]:
    return _record(
        "status:spouse",
        dataset_status_id="status:spouse",
        dataset_name="spouse_records",
        applicability="applicable",
        presence_status="partial",
        observed_row_count=1,
        reason="source_partially_observed",
    )


def _spouse_absence_record(
    *,
    provider: str | None = "广州广汽租赁有限公司",
    absent_fields: list[str] | None = None,
) -> dict[str, Any]:
    fields = ["name", "document_type", "document_number", "employer", "phone"]
    return _record(
        "spouse:absence",
        spouse_record_id="spouse:absence",
        name="--",
        document_type="--",
        document_number="--",
        employer="--",
        phone="--",
        data_provider=provider,
        _source_absent_fields=fields if absent_fields is None else absent_fields,
    )


def test_spouse_direct_dash_absence_clears_stale_partial_status(
    tmp_path: Path,
) -> None:
    payload = _community_payload(
        tmp_path,
        {
            "spouse_records": [_spouse_absence_record()],
            "personal_detail_dataset_status": [_spouse_absence_status()],
        },
    )
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}

    spouse = datasets["subject_spouse"]
    assert spouse["completeness"]["verified"] is True
    values = spouse["rows"][0]["normalized"]
    assert values["data_provider"] == "广州广汽租赁有限公司"
    assert all(
        values[field_name] is None
        for field_name in ("name", "document_type", "document_number", "employer", "phone")
    )
    assert not any(
        row["normalized"]["dataset_name"] == "subject_spouse"
        for row in datasets.get("dataset_status", {}).get("rows", [])
    )


def test_spouse_status_stays_partial_without_exact_absence_proof(
    tmp_path: Path,
) -> None:
    all_fields = ["name", "document_type", "document_number", "employer", "phone"]
    cases = (
        {
            "spouse_records": [
                _record(
                    "spouse:blank",
                    spouse_record_id="spouse:blank",
                    name=None,
                    document_type=None,
                    document_number=None,
                    employer=None,
                    phone=None,
                    data_provider="广州广汽租赁有限公司",
                )
            ],
            "personal_detail_dataset_status": [_spouse_absence_status()],
        },
        {
            "spouse_records": [
                _spouse_absence_record(absent_fields=all_fields[:-1])
            ],
            "personal_detail_dataset_status": [_spouse_absence_status()],
        },
        {
            "spouse_records": [_spouse_absence_record(provider=None)],
            "personal_detail_dataset_status": [_spouse_absence_status()],
        },
        {
            "spouse_records": [_spouse_absence_record()],
            "personal_detail_dataset_status": [_spouse_absence_status()],
            "personal_detail_extraction_issues": [
                _issue(
                    "issue:spouse:unreadable",
                    dataset="subject_spouse",
                    record_id="spouse:absence",
                    field_name="name",
                    observed_value="unreadable",
                    issue_code="candidate_b_spouse_row_unresolved",
                )
            ],
        },
    )

    for source in cases:
        payload = _community_payload(tmp_path, source)
        datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
        status = next(
            row["normalized"]
            for row in datasets["dataset_status"]["rows"]
            if row["normalized"]["dataset_name"] == "subject_spouse"
        )
        assert status["presence_status"] == "partial"
        assert status["reason"] == "source_partially_observed"
        assert datasets["subject_spouse"]["completeness"]["verified"] is False


def test_community_reports_every_emitted_monthly_row_with_missing_amount(
    tmp_path: Path,
) -> None:
    source = {
        "credit_accounts": [
            _record(
                "account:1",
                account_id="account:1",
                sequence=1,
                account_type="credit_card",
            )
        ],
        "repayment_records": [
            _record(
                "monthly:missing",
                repayment_id="monthly:missing",
                account_id="account:1",
                year=2025,
                month=1,
                status="N",
                overdue_amount=None,
            ),
            _record(
                "monthly:zero",
                repayment_id="monthly:zero",
                account_id="account:1",
                year=2025,
                month=2,
                status="/",
                overdue_amount=0,
            ),
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    monthly = {
        row["record_id"]: row["normalized"]
        for row in datasets.get("credit_account_monthly_performance", {}).get(
            "rows", []
        )
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]

    assert monthly["monthly:missing"]["status_code"] == "N"
    assert monthly["monthly:missing"]["status_amount"] is None
    assert monthly["monthly:zero"]["status_code"] == "/"
    assert monthly["monthly:zero"]["status_amount"] == "0"
    amount_issues = [
        issue
        for issue in issues
        if issue.get("target_dataset") == "credit_account_monthly_performance"
        and issue.get("field_name") == "status_amount"
    ]
    assert len(amount_issues) == 1
    assert amount_issues[0]["target_record_id"] == "monthly:missing"
    assert amount_issues[0]["issue_code"] == "candidate_b_monthly_status_amount_unresolved"


def test_community_quarantines_nonzero_amount_for_zero_overdue_status(
    tmp_path: Path,
) -> None:
    source = {
        "credit_accounts": [
            _record(
                "account:1",
                account_id="account:1",
                sequence=1,
                account_type="credit_card",
            )
        ],
        "repayment_records": [
            _record(
                "monthly:n-conflict",
                repayment_id="monthly:n-conflict",
                account_id="account:1",
                year=2025,
                month=1,
                status="N",
                overdue_amount="10",
                status_amount_semantics="delinquent_amount",
            ),
            _record(
                "monthly:star-conflict",
                repayment_id="monthly:star-conflict",
                account_id="account:1",
                year=2025,
                month=2,
                status="*",
                overdue_amount="20",
                status_amount_semantics="delinquent_amount",
            ),
            _record(
                "monthly:clean",
                repayment_id="monthly:clean",
                account_id="account:1",
                year=2025,
                month=3,
                status="N",
                overdue_amount="0",
            ),
            _record(
                "monthly:numeric",
                repayment_id="monthly:numeric",
                account_id="account:1",
                year=2025,
                month=4,
                status="1",
                overdue_amount="4691",
            ),
            _record(
                "monthly:missing",
                repayment_id="monthly:missing",
                account_id="account:1",
                year=2025,
                month=5,
                status="N",
                overdue_amount=None,
            ),
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    monthly = {
        row["record_id"]: row
        for row in datasets["credit_account_monthly_performance"]["rows"]
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]

    for record_id, status, observed in (
        ("monthly:n-conflict", "N", "10"),
        ("monthly:star-conflict", "*", "20"),
    ):
        wrapper = monthly[record_id]
        assert wrapper["normalized"]["status_code"] == status
        assert wrapper["normalized"]["status_amount"] is None
        assert wrapper["normalized"]["status_amount_semantics"] is None
        assert wrapper["canonical_raw"]["status_amount"] == observed

    assert monthly["monthly:clean"]["normalized"]["status_amount"] == "0"
    assert monthly["monthly:numeric"]["normalized"]["status_code"] == "1"
    assert monthly["monthly:numeric"]["normalized"]["status_amount"] == "4691"
    conflict_issues = [
        issue
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_monthly_zero_status_amount_conflict"
    ]
    assert {issue["target_record_id"] for issue in conflict_issues} == {
        "monthly:n-conflict",
        "monthly:star-conflict",
    }
    assert all(issue["field_name"] == "status_amount" for issue in conflict_issues)
    assert not any(
        issue.get("issue_code") == "candidate_b_monthly_status_amount_unresolved"
        and issue.get("target_record_id")
        in {"monthly:n-conflict", "monthly:star-conflict"}
        for issue in issues
    )
    missing = next(
        issue
        for issue in issues
        if issue.get("target_record_id") == "monthly:missing"
        and issue.get("field_name") == "status_amount"
    )
    assert missing["issue_code"] == "candidate_b_monthly_status_amount_unresolved"


def test_community_withholds_normal_status_in_exact_terminal_month(
    tmp_path: Path,
) -> None:
    source = {
        "credit_accounts": [
            _record(
                "account:settled",
                account_id="account:settled",
                sequence=1,
                account_type="non_revolving_loan",
                account_lifecycle_state="settled",
                close_date="2025-02-18",
            ),
            _record(
                "account:closed",
                account_id="account:closed",
                sequence=2,
                account_type="credit_card",
                account_lifecycle_state="closed",
                close_date="2025-02-28",
            ),
        ],
        "repayment_records": [
            _record(
                "monthly:before-close",
                repayment_id="monthly:before-close",
                account_id="account:settled",
                year=2025,
                month=1,
                status="N",
                overdue_amount="0",
            ),
            _record(
                "monthly:terminal-conflict",
                repayment_id="monthly:terminal-conflict",
                account_id="account:settled",
                year=2025,
                month=2,
                status="N",
                overdue_amount="0",
            ),
            _record(
                "monthly:terminal-c",
                repayment_id="monthly:terminal-c",
                account_id="account:closed",
                year=2025,
                month=2,
                status="C",
                overdue_amount="0",
            ),
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    monthly = {
        row["record_id"]: row
        for row in datasets["credit_account_monthly_performance"]["rows"]
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]

    assert set(monthly) == {"monthly:before-close", "monthly:terminal-c"}
    assert monthly["monthly:before-close"]["normalized"]["status_code"] == "N"
    assert monthly["monthly:terminal-c"]["normalized"]["status_code"] == "C"
    assert all("review" not in row for row in monthly.values())
    conflict = next(
        issue
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_monthly_terminal_status_conflict"
    )
    assert conflict["target_record_id"] == "monthly:terminal-conflict"
    assert conflict["field_name"] == "status_code"
    assert conflict["observed_value"] == "N"
    assert not any(
        issue.get("target_record_id")
        in {"monthly:before-close", "monthly:terminal-c"}
        for issue in issues
    )


def test_community_never_publishes_cross_plane_or_terminal_status_conflicts(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    page = SimpleNamespace(page_number=8, source_page_number=8)
    table = SimpleNamespace(
        table_id="pt_8_2",
        bbox=[10.0, 300.0, 590.0, 400.0],
        metadata={
            "raw_rows": [
                ["物 账户状态 爱", "? 账户关闭日期"],
                ["家 结清", "Q 2020.02.21"],
            ]
        },
    )
    account = _record(
        "credit_account:non_revolving_loan:12",
        account_id="credit_account:non_revolving_loan:12",
        sequence=12,
        account_type="non_revolving_loan",
        canonical_raw={},
    )
    native_extraction._apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )
    assert account["account_lifecycle_state"] == "settled"
    assert account["close_date"] == "2020-02-21"

    native = {
        "repayment_records": [
            _record(
                "grid:conflict:2020-01",
                repayment_id="grid:conflict:2020-01",
                grid_id="grid:conflict",
                account_id=account["account_id"],
                year=2020,
                month=1,
                status="C",
                overdue_amount="0",
                source_refs_by_field={
                    "status": [
                        {
                            "page": 8,
                            "logical_page": 8,
                            "grid_id": "grid:conflict",
                            "row": 9,
                            "col": 1,
                            "field_name": "status",
                            "geometry_scope": "cell",
                            "bbox": [10.0, 10.0, 20.0, 20.0],
                        }
                    ]
                },
            )
        ]
    }
    corrected_conflict = _record(
        "grid:conflict:2020-01",
        repayment_id="grid:conflict:2020-01",
        grid_id="grid:conflict",
        account_id=account["account_id"],
        year=2020,
        month=1,
        status="N",
        overdue_amount="0",
        source_refs_by_field={
            "status": [
                {
                    "page": 8,
                    "logical_page": 8,
                    "grid_id": "grid:conflict",
                    "row": 2,
                    "col": 1,
                    "field_name": "status",
                    "geometry_scope": "cell",
                    "bbox": [10.0, 30.0, 20.0, 40.0],
                }
            ]
        },
    )
    terminal_conflict = _record(
        "grid:terminal:2020-02",
        repayment_id="grid:terminal:2020-02",
        grid_id="grid:terminal",
        account_id=account["account_id"],
        year=2020,
        month=2,
        status="N",
        overdue_amount="0",
    )
    corrected = {"repayment_records": [corrected_conflict, terminal_conflict]}
    _withhold_independent_plane_conflicts(context, native, corrected)

    payload = _community_payload(
        tmp_path,
        {
            "credit_accounts": [account],
            **corrected,
            "personal_detail_extraction_issues": list(
                context._personal_detail_extraction_issues
            ),
        },
    )
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    monthly = {
        row["record_id"]: row["normalized"]
        for row in datasets.get("credit_account_monthly_performance", {}).get(
            "rows", []
        )
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]

    assert "grid:conflict:2020-01" not in monthly
    assert "grid:terminal:2020-02" not in monthly
    cross_plane = next(
        issue
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_independent_plane_repayment_status_conflict"
    )
    terminal = next(
        issue
        for issue in issues
        if issue.get("issue_code") == "candidate_b_monthly_terminal_status_conflict"
    )
    assert cross_plane["target_record_id"] == "grid:conflict:2020-01"
    assert cross_plane["field_name"] == "status_code"
    assert terminal["target_record_id"] == "grid:terminal:2020-02"
    assert terminal["field_name"] == "status_code"


def test_community_clears_only_stale_successful_monthly_review_metadata(
    tmp_path: Path,
) -> None:
    source = {
        "credit_accounts": [
            _record(
                "account:1",
                account_id="account:1",
                sequence=1,
                account_type="credit_card",
                account_lifecycle_state="open",
            )
        ],
        "repayment_records": [
            _record(
                "monthly:clean",
                repayment_id="monthly:clean",
                account_id="account:1",
                year=2025,
                month=1,
                status="N",
                overdue_amount="0",
                extraction_status="review",
                recognition_source="static_glyph_shape_validation",
                audit={"reason": "field_specific_zero_status_shape_validation"},
                canonical_raw={"status": "N", "extraction_status": "review"},
            ),
            _record(
                "monthly:field-issue",
                repayment_id="monthly:field-issue",
                account_id="account:1",
                year=2025,
                month=2,
                status="N",
                overdue_amount="0",
                extraction_status="review",
                recognition_source="static_glyph_shape_validation",
                audit={"reason": "field_specific_zero_status_shape_validation"},
            ),
            _record(
                "monthly:record-issue",
                repayment_id="monthly:record-issue",
                account_id="account:1",
                year=2025,
                month=3,
                status="C",
                overdue_amount="0",
                extraction_status="review",
            ),
        ],
        "personal_detail_extraction_issues": [
            _issue(
                "issue:monthly-field",
                dataset="repayment_records",
                record_id="monthly:field-issue",
                field_name="overdue_amount",
                observed_value=["unreadable"],
                issue_code="pboc_cell_contract_unresolved",
            ),
            _record(
                "issue:monthly-record",
                extraction_issue_id="issue:monthly-record",
                category="schema_incompleteness",
                issue_code="candidate_b_monthly_row_uncertain",
                severity="warning",
                status="requires_review",
                parser_stage="candidate_b_test_fixture",
                target_dataset="repayment_records",
                target_record_id="monthly:record-issue",
                message="Synthetic record-level fixture.",
            ),
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    monthly = {
        row["record_id"]: row
        for row in datasets["credit_account_monthly_performance"]["rows"]
    }

    clean = monthly["monthly:clean"]
    assert clean["normalized"]["status_code"] == "N"
    assert clean["normalized"]["status_amount"] == "0"
    assert "review" not in clean
    assert "confidence" not in clean
    assert clean["canonical_raw"] == {}
    assert clean["raw"] == {}
    assert monthly["monthly:field-issue"]["review"]["status"] == "requires_review"
    assert monthly["monthly:record-issue"]["review"]["status"] == "requires_review"


def test_community_collapses_only_explicitly_absent_spouse_scalar(
    tmp_path: Path,
) -> None:
    spouse_fields = ("name", "document_type", "document_number", "employer", "phone")
    source = {
        "spouse_records": [
            _record(
                "spouse:absent",
                spouse_record_id="spouse:absent",
                name=None,
                document_type=None,
                document_number=None,
                employer=None,
                phone=None,
                data_provider="Example Bank",
                _source_absent_fields=["name"],
                canonical_raw={"name": "--"},
            ),
            _record(
                "spouse:blank",
                spouse_record_id="spouse:blank",
                name="",
                document_type=None,
                document_number=None,
                employer=None,
                phone=None,
                data_provider="Example Bank",
            ),
        ],
        "personal_detail_extraction_issues": [
            *[
                _issue(
                    f"issue:spouse-absent:{index}",
                    dataset="subject_spouse",
                    record_id="spouse:absent",
                    field_name=field_name,
                    observed_value=[""],
                    issue_code="pboc_cell_contract_unresolved",
                )
                for index, field_name in enumerate(spouse_fields, start=1)
            ],
            *[
                _issue(
                    f"issue:spouse-blank:{index}",
                    dataset="subject_spouse",
                    record_id="spouse:blank",
                    field_name=field_name,
                    observed_value=[""],
                    issue_code="pboc_cell_contract_unresolved",
                )
                for index, field_name in enumerate(spouse_fields, start=1)
            ],
        ],
    }

    payload = _community_payload(tmp_path, source)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    spouses = {
        row["record_id"]: row["normalized"]
        for row in datasets["subject_spouse"]["rows"]
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]
    issue_targets = {
        (
            issue.get("target_record_id"),
            issue.get("field_name"),
        )
        for issue in issues
        if issue.get("target_dataset") == "subject_spouse"
    }

    assert all(spouses["spouse:absent"][field_name] is None for field_name in spouse_fields)
    assert spouses["spouse:absent"]["data_provider"] == "Example Bank"
    assert not any(record_id == "spouse:absent" for record_id, _field in issue_targets)
    assert ("spouse:blank", "name") in issue_targets


def test_community_keeps_canonical_employment_occupation_but_reports_near_match(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    corrected = overlay.correct_business_candidates(
        {
            "employment_records": [
                _record(
                    "employment:canonical",
                    employment_record_id="employment:canonical",
                    occupation="商业、服务业人员",
                ),
                _record(
                    "employment:polluted",
                    employment_record_id="employment:polluted",
                    occupation="商业、服务业人员X",
                ),
            ]
        },
        stage="candidate_b_final_validation",
    )
    context.ocr_correction_audit = overlay.audit
    corrected["personal_detail_extraction_issues"] = collect_extraction_issues(context)

    payload = _community_payload(tmp_path, corrected)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    employment = {
        row["record_id"]: row["normalized"]
        for row in datasets["subject_employment"]["rows"]
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]

    assert employment["employment:canonical"]["occupation"] == "商业、服务业人员"
    assert employment["employment:polluted"]["occupation"] is None
    assert not any(
        issue.get("target_record_id") == "employment:canonical"
        and issue.get("field_name") == "occupation"
        for issue in issues
    )
    assert any(
        issue.get("target_dataset") == "subject_employment"
        and issue.get("target_record_id") == "employment:polluted"
        and issue.get("field_name") == "occupation"
        for issue in issues
    )


def test_community_identity_number_contract_is_document_type_aware(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    corrected = overlay.correct_business_candidates(
        {
            "identity_documents": [
                _record(
                    "identity:passport",
                    document_type="护照",
                    document_number="E12345678",
                ),
                _record(
                    "identity:bad-passport",
                    document_type="护照",
                    document_number="E12?345",
                ),
                _record(
                    "identity:type-mismatch",
                    document_type="护照",
                    document_number="11010519491231002X",
                ),
            ]
        },
        stage="candidate_b_final_validation",
    )
    context.ocr_correction_audit = overlay.audit
    corrected["personal_detail_extraction_issues"] = collect_extraction_issues(
        context
    )

    payload = _community_payload(tmp_path, corrected)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    identity = {
        row["record_id"]: row["normalized"]
        for row in datasets["subject_identity_documents"]["rows"]
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]

    assert identity["identity:passport"]["document_number"] == "E12345678"
    assert identity["identity:bad-passport"]["document_number"] is None
    assert identity["identity:type-mismatch"]["document_number"] is None
    assert not any(
        issue.get("target_record_id") == "identity:passport"
        and issue.get("field_name") == "document_number"
        for issue in issues
    )
    assert {
        issue.get("target_record_id")
        for issue in issues
        if issue.get("target_dataset") == "subject_identity_documents"
        and issue.get("field_name") == "document_number"
    } == {"identity:bad-passport", "identity:type-mismatch"}


def test_community_summary_uncertainty_targets_only_emitted_business_rows(
    tmp_path: Path,
) -> None:
    source_cell = {
        "record_id": "summary-cell:polluted",
        "summary_cell_id": "summary-cell:polluted",
        "summary_record_id": "summary:1",
        "summary_type": "非循环贷账户信息汇总",
        "title": "非循环贷账户信息汇总",
        "row_index": 1,
        "column_index": 1,
        "column_label": "账户数",
        "value": "9户",
    }
    prepared = prepare_personal_detail_source_collections(
        {"facts": {}, "datasets": {"personal_detail_summary_cells": [source_cell]}}
    )["datasets"]
    prepared.setdefault("personal_detail_extraction_issues", []).append(
        _issue(
            "issue:summary-empty-anchor",
            dataset="personal_detail_summary_records",
            record_id="summary:empty-anchor",
            field_name="value",
            observed_value="unusable table anchor",
            issue_code="candidate_b_summary_anchor_without_usable_rows",
        )
    )

    payload = _community_payload(tmp_path, prepared)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    overview_ids = {
        row["record_id"] for row in datasets["credit_business_overview"]["rows"]
    }
    issues = [row["normalized"] for row in datasets["extraction_issues"]["rows"]]
    scalar_issue = next(
        issue
        for issue in issues
        if issue["issue_code"] == "candidate_b_summary_scalar_unresolved"
    )
    anchor_issue = next(
        issue
        for issue in issues
        if issue["issue_code"] == "candidate_b_summary_anchor_without_usable_rows"
    )

    assert scalar_issue["target_record_id"] in overview_ids
    assert scalar_issue["target_record_id"] == "credit_summary_metric:summary-cell:polluted"
    assert scalar_issue["field_name"] == "numeric_value"
    assert anchor_issue.get("target_record_id") is None
    row_ids_by_dataset = {
        name: {row["record_id"] for row in dataset["rows"]}
        for name, dataset in datasets.items()
    }
    for issue in issues:
        target_dataset = issue.get("target_dataset")
        target_record_id = issue.get("target_record_id")
        if target_dataset in row_ids_by_dataset and target_record_id:
            assert target_record_id in row_ids_by_dataset[target_dataset]
            field_name = issue.get("field_name")
            if field_name:
                assert field_name in personal_detail_data_dictionary()["datasets"][
                    target_dataset
                ]["columns"]

    issue_ids = {issue["extraction_issue_id"] for issue in issues}
    evidence = datasets["extraction_issue_evidence"]["rows"]
    assert evidence
    assert {
        row["normalized"]["extraction_issue_id"] for row in evidence
    } <= issue_ids
