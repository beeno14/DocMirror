# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.community_plugin import CreditReportPlugin
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PBOC_DATASET_ORDER,
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    PERSONAL_PROFILE_FIELDS,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURE = Path("tests/fixtures-private/credit_report/Scanned Personal Detailed/叶永燕征信.pdf")


def test_ye_yongyan_personal_detail_schema_contract() -> None:
    """Regress the plugin contract without asserting OCR extraction accuracy."""
    if not _FIXTURE.exists():
        pytest.skip("叶永燕 personal detailed report fixture is unavailable")

    audit_dir = os.environ.get("DOCMIRROR_PERSONAL_DETAIL_AUDIT_DIR")
    cached_payload = Path(audit_dir or "") / f"{_FIXTURE.stem}.community.json" if audit_dir else None
    if cached_payload is not None and cached_payload.exists():
        payload = json.loads(cached_payload.read_text(encoding="utf-8"))
        semantic = {
            "domain": {
                "data_dictionary": personal_detail_data_dictionary(),
                "extensions": personal_detail_semantic_extensions(),
            }
        }
    else:
        sealed = asyncio.run(
            perceive_document(
                _FIXTURE,
                PerceiveOptions(
                    policy=normalize_parse_policy(
                        enhance_mode="standard",
                        doc_type_hint="credit_report:force",
                    )
                ),
            )
        )
        bundle = CreditReportPlugin().project_bundle(sealed, file_path=str(_FIXTURE))
        semantic = bundle.semantic_payload()
        payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}

    assert payload["document"]["type"] == "personal_credit_report_detailed"
    assert validate_projection_payload("community", payload).valid
    domain_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    status_counts = {
        row["normalized"]["dataset_name"]: row["normalized"]["observed_row_count"]
        for row in datasets["dataset_status"]["rows"]
    }
    assert domain_validation.valid, {
        "errors": domain_validation.errors,
        "credit_agreements": (
            datasets.get("credit_agreements", {}).get("row_count", 0),
            status_counts.get("credit_agreements"),
        ),
        "monthly_performance": (
            datasets.get("credit_account_monthly_performance", {}).get("row_count", 0),
            status_counts.get("credit_account_monthly_performance"),
        ),
    }
    dictionary = semantic["domain"]["data_dictionary"]
    assert dictionary["schema_id"] == "personal_credit_report_detailed"
    assert dictionary["version"] == "2.0.0"

    profile = datasets["subject_profile"]["rows"][0]["normalized"]
    assert profile["subject_name"]
    assert profile["primary_id_number"]

    observations = [row["normalized"] for row in datasets["field_observations"]["rows"]]
    profile_observations = [row for row in observations if row["dataset_name"] == "subject_profile"]
    assert {row["field_name"] for row in profile_observations} <= set(PERSONAL_PROFILE_FIELDS)
    assert all(
        row["observation_status"] in {"ambiguous", "unreadable", "not_observed", "inferred"}
        for row in observations
        if row["dataset_name"] != "subject_profile"
    )
    assert {row["observation_status"] for row in observations} <= {
        "ocr_corrected",
        "inferred",
        "ambiguous",
        "unreadable",
        "not_observed",
    }
    assert all(
        row.get("reason")
        for row in observations
        if row["observation_status"] == "not_observed"
    )
    summary_metrics = datasets["credit_business_overview"]
    assert summary_metrics["row_count"] > 0
    assert all(
        row["normalized"]["reporting_status"] in {"reported", "not_reported", "unknown"}
        for row in summary_metrics["rows"]
    )

    statuses = {
        row["normalized"]["dataset_name"]: row["normalized"]
        for row in datasets["dataset_status"]["rows"]
    }
    assert set(statuses) <= set(PBOC_DATASET_ORDER) - {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
    assert all(
        row["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for row in statuses.values()
    )
    assert statuses["credit_accounts"]["presence_status"] == "partial"
    assert statuses["credit_accounts"]["observed_row_count"] == datasets["credit_accounts"]["row_count"]
    # Source-grounded structure: these counts protect the continuation repair
    # and the positional profile-table extraction from schema-only regressions.
    account_count = datasets["credit_accounts"]["row_count"]
    extraction_issues = [
        row["normalized"] for row in datasets["extraction_issues"]["rows"]
    ]
    issue_evidence = [
        row["normalized"]
        for row in datasets.get("extraction_issue_evidence", {}).get("rows", [])
    ]
    if account_count != 42:
        assert account_count < 42
        account_gap = next(
            issue
            for issue in extraction_issues
            if issue.get("issue_code") == "candidate_b_account_sequence_gap"
        )
        missing_sequences = [
            row
            for row in issue_evidence
            if row.get("extraction_issue_id") == account_gap["extraction_issue_id"]
            and row.get("evidence_kind") == "candidate"
            and str(row.get("evidence_path") or "").startswith("missing_category_sequences[")
        ]
        assert account_count + len(missing_sequences) >= 42
        assert any(
            issue.get("issue_code") == "monthly_linkage_collision_from_account_gap"
            for issue in extraction_issues
        )
    def assert_exact_or_reported(dataset_name: str, expected_count: int) -> None:
        observed_count = datasets.get(dataset_name, {}).get("row_count", 0)
        if observed_count == expected_count:
            return
        assert observed_count < expected_count
        assert statuses[dataset_name]["presence_status"] in {"partial", "extraction_failed", "unknown"}
        assert any(issue.get("target_dataset") == dataset_name for issue in extraction_issues)

    assert_exact_or_reported("subject_residences", 5)
    assert_exact_or_reported("subject_employment", 5)
    assert_exact_or_reported("subject_mobile_phones", 1)
    assert_exact_or_reported("subject_spouse", 1)
    agreements = [row["normalized"] for row in datasets["credit_agreements"]["rows"]]
    assert len(agreements) == 16
    agreement_sequences = [row["sequence"] for row in agreements if row.get("sequence") is not None]
    assert len(agreement_sequences) == len(set(agreement_sequences))
    assert all(1 <= sequence <= 16 for sequence in agreement_sequences)
    assert sum(
        issue.get("target_dataset") == "credit_agreements"
        and issue.get("field_name") == "sequence"
        and issue.get("issue_code") == "candidate_b_credit_agreement_sequence_unresolved"
        for issue in extraction_issues
    ) >= len(agreements) - len(agreement_sequences)
    assert all(
        not any(key.startswith(("observed__", "candidate__", "reason__")) for key in issue)
        for issue in extraction_issues
    )
    assert all(
        evidence["extraction_issue_id"]
        in {issue["extraction_issue_id"] for issue in extraction_issues}
        for evidence in issue_evidence
    )

    account_ids = {row["normalized"]["account_id"] for row in datasets["credit_accounts"]["rows"]}
    repayment_rows = [
        row["normalized"] for row in datasets["credit_account_monthly_performance"]["rows"]
    ]
    assert repayment_rows
    assert all(row.get("account_id") in account_ids for row in repayment_rows)

    contract = semantic["domain"]["extensions"]["personal_detail_contract"]
    assert contract["absence_requires_explicit_source_evidence"] is True
    assert contract["empty_dataset_means_absent"] is False
    assert set(contract["canonical_public_record_datasets"]) <= set(dictionary["datasets"])
