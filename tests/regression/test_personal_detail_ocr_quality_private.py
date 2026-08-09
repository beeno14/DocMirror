# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.community_plugin import CreditReportPlugin
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    PBOC_DATASET_ORDER,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]

_FIXTURE_DIR = Path(
    os.environ.get(
        "DOCMIRROR_PERSONAL_DETAIL_FIXTURE_DIR",
        "tests/fixtures-private/credit_report/Scanned Personal Detailed",
    )
)
_FIXTURES = sorted(_FIXTURE_DIR.glob("*.pdf"))
_EXPECTED_SCHEMA_INPUT_COUNTS = {
    "余泽熙7.15征信.pdf": (27, 641),
    "杨松林个人征信24.7.29.pdf": (38, 615),
    "叶永燕征信.pdf": (42, 884),
    "征信.pdf": (43, 408),
    # Source-page audit confirms 45 business accounts.  In addition to the
    # three responsibility-table false cards removed from the former 48-row
    # oracle, logical page 12 proves that D10053310... is the sole type-R2
    # account: the former 46-row result emitted its table again as an R1 row.
    # A source-grid audit counts all 40 printed repayment grids and their
    # bounded date ranges: 944 printed month positions.  The former 801 oracle
    # omitted valid grids and could make a silent population loss look healthy.
    "林岚挺征信.pdf": (45, 944),
    "洪晓鑫征信报告2025.11.05.pdf": (8, 176),
    "王根镇征信.pdf": (61, 757),
}
_EXPECTED_AGREEMENT_COUNTS = {
    "叶永燕征信.pdf": 16,
    "林岚挺征信.pdf": 11,
    "余泽熙7.15征信.pdf": 8,
    "杨松林个人征信24.7.29.pdf": 7,
    "洪晓鑫征信报告2025.11.05.pdf": 7,
}
_EXPECTED_INQUIRY_COUNTS = {
    "叶永燕征信.pdf": 112,
    "林岚挺征信.pdf": 90,
    "余泽熙7.15征信.pdf": 26,
    "杨松林个人征信24.7.29.pdf": 117,
    "洪晓鑫征信报告2025.11.05.pdf": 20,
}


def _perceive(fixture: Path):
    return asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda path: path.name)
def test_personal_detail_ocr_correction_invariants(
    fixture: Path,
) -> None:
    """Exercise source correction and the schema boundary on every private report."""
    sealed = _perceive(fixture)
    result = sealed.to_read_view()
    context = build_personal_detail_extraction_context(result)
    domain_specific = result.entities.domain_specific
    raw_bundles = domain_specific.get("_page_evidence_bundles") or []
    raw_snapshot = deepcopy(raw_bundles)
    topology_audit = context.page_topology_audit()

    corrected_pages = context.corrected_evidence_pages()
    business = context.scanned_business(result.full_text or "")
    native_business = context.native_business(result.full_text or "")
    repayment_records = context.corrected_repayment_records()
    audit = context.ocr_correction_audit()
    ocr_bundles = [
        bundle
        for bundle in raw_bundles
        if isinstance(bundle, dict)
        and isinstance(bundle.get("local_structure_evidence"), dict)
        and bundle["local_structure_evidence"].get("lines")
    ]

    # Persist the JSON audit artifact before diagnostic assertions so a
    # topology/test-harness failure cannot discard an otherwise usable output.
    bundle = _project_personal_detail_bundle(sealed, fixture)
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    audit_dir = os.environ.get("DOCMIRROR_PERSONAL_DETAIL_AUDIT_DIR")
    if audit_dir:
        destination = Path(audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{fixture.stem}.community.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (destination / f"{fixture.stem}.semantic.json").write_text(
            json.dumps(semantic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    assert raw_bundles == raw_snapshot
    assert topology_audit["valid"] is True
    assert topology_audit["logical_page_count"] == len(result.pages)
    assert all(len(logical_pages) >= 1 for logical_pages in topology_audit["logical_pages_by_source"].values())
    assert sorted(context.reading_order_by_logical.values()) == list(
        range(1, len(context.reading_order_by_logical) + 1)
    )
    assert context.entity_context.content_conserved is True
    static_pages = [page for page in corrected_pages if page.get("plugin_static_subpage")]
    static_sources = {int(page.get("source_page") or 0) for page in static_pages}
    if static_pages:
        raw_counts = Counter(
            int(
                bundle.get("source_page_number")
                or (bundle.get("local_structure_evidence") or {}).get("source_page")
                or 0
            )
            for bundle in ocr_bundles
        )
        corrected_counts = Counter(int(page.get("source_page") or 0) for page in corrected_pages)
        for source in static_sources:
            corrected_segments = []
            for page in corrected_pages:
                if int(page.get("source_page") or 0) != source:
                    continue
                if page.get("plugin_static_subpage"):
                    corrected_segments.append(int(page.get("segment_index") or 0))
                    continue
                geometry = context.page_topology.geometry(int(page.get("page") or 0))
                if geometry is not None and geometry.segment_index in {0, 1}:
                    corrected_segments.append(int(geometry.segment_index))
            assert sorted(corrected_segments) == [0, 1]
        assert all(corrected_counts[source] == 2 and raw_counts[source] == 1 for source in static_sources)
        assert all(
            corrected_counts[source] == count for source, count in raw_counts.items() if source not in static_sources
        )
    else:
        assert len(corrected_pages) == len(ocr_bundles)
    assert audit["ocr_started_by_correction_overlay"] is False
    repair_decisions = audit["business_repair"]["page_decisions"]
    assert audit["business_repair"]["field_triggered_ocr_requests"] == sum(
        int(decision.get("ocr_invocations") or 0) for decision in repair_decisions
    )

    assert all(int(decision.get("ocr_invocations") or 0) <= 1 for decision in repair_decisions)
    repaired_pages = [
        int(decision["logical_page"])
        for decision in repair_decisions
        if int(decision.get("ocr_invocations") or 0) == 1
    ]
    assert len(repaired_pages) == len(set(repaired_pages))
    assert all(
        decision["original"] != decision["corrected"]
        and decision["action"] in {"applied", "suggested"}
        and decision["role"]
        for decision in audit["decisions"]
    )

    accounts = business.get("credit_accounts") or []
    expected_counts = _EXPECTED_SCHEMA_INPUT_COUNTS.get(fixture.name)
    if expected_counts is not None:
        expected_accounts, expected_repayments = expected_counts
        if len(accounts) != expected_accounts:
            # A withheld/suppressed record is acceptable only when the final
            # community JSON exposes enough structured account-level issues to
            # cover the complete source-backed shortfall. Silent omissions are
            # never accepted as a count tolerance.
            assert len(accounts) < expected_accounts
            account_issues = [
                issue
                for issue in collect_extraction_issues(context)
                if issue.get("target_dataset") == "credit_accounts"
            ]
            suppressed = {
                str(issue.get("extraction_issue_id") or issue.get("target_record_id") or "")
                for issue in account_issues
                if issue.get("issue_code") == "candidate_b_unmatched_account_table_suppressed"
            }
            sequence_gap = max(
                (
                    len((issue.get("candidate_value") or {}).get("missing_category_sequences") or ())
                    for issue in account_issues
                    if issue.get("issue_code") == "candidate_b_account_sequence_gap"
                ),
                default=0,
            )
            assert len(accounts) + max(len(suppressed), sequence_gap) >= expected_accounts
    # Candidate B owns the final account/month relation. Re-running the retired
    # shared linker against sealed pre-repair grids would compare two different
    # evidence planes and can discard valid corrected-grid rows.
    linked_repayments = list(business.get("repayment_records") or ())
    source_issues = collect_extraction_issues(context)
    if expected_counts is not None:
        # Candidate-B deliberately removed typed cell-level OCR.  The former
        # cell-crop row count is retained only as a coverage regression guard,
        # not as a completeness oracle: a difference is reportable only when
        # canonical schema/source structure independently demonstrates a gap.
        canonical_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "canonical_monthly_reconstruction_incomplete"
        ]
        population_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "monthly_population_incomplete_from_account_gap"
        ]
        linkage_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "monthly_linkage_collision_from_account_gap"
        ]
        status_gaps = [
            issue
            for issue in collect_extraction_issues(context)
            if issue.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        ]
        if len(linked_repayments) < int(expected_repayments * 0.90):
            assert canonical_gaps or population_gaps or linkage_gaps or status_gaps
        for canonical_gap in canonical_gaps:
            canonical_count = canonical_gap["observed_value"]["canonical_row_count"]
            assert canonical_count >= len(linked_repayments)
            if canonical_count > len(linked_repayments):
                assert population_gaps or linkage_gaps or status_gaps
            assert canonical_gap["candidate_value"]["structural_expected_row_count"] > len(linked_repayments)
            assert canonical_gap["candidate_value"]["missing_month_count"] > 0
        for population_gap in population_gaps:
            assert population_gap["observed_value"]["canonical_grid_row_count"] >= len(linked_repayments)
            missing_sequences = population_gap["candidate_value"]["missing_account_category_sequences"]
            unresolved_printed_ordinals = any(
                int((issue.get("candidate_value") or {}).get("unresolved_printed_ordinal_count") or 0) > 0
                for issue in collect_extraction_issues(context)
                if issue.get("issue_code") == "candidate_b_account_sequence_gap"
            )
            assert any(missing_sequences.values()) or unresolved_printed_ordinals
        for linkage_gap in linkage_gaps:
            assert linkage_gap["observed_value"]["final_linked_row_count"] == len(linked_repayments)
            assert linkage_gap["candidate_value"]["pre_deduplication_row_count"] > len(linked_repayments)
    unresolved_monthly_ids = {
        str(issue.get("target_record_id") or "")
        for issue in source_issues
        if issue.get("issue_code") == "candidate_b_monthly_grid_owner_unresolved"
    }
    unresolved_monthly_grid_ids = {
        target.rsplit(":", 1)[0] for target in unresolved_monthly_ids if ":" in target
    }
    for record in linked_repayments:
        if record.get("account_id"):
            continue
        repayment_id = str(record.get("repayment_id") or record.get("record_id") or "")
        assert record.get("extraction_status") == "review"
        assert repayment_id in unresolved_monthly_ids or str(record.get("grid_id") or "") in unresolved_monthly_grid_ids
    account_ids = [account.get("account_id") for account in accounts]
    assert len(account_ids) == len(set(account_ids))
    logical_pages = set(context.source_page_by_logical)
    assert all(
        int(ref.get("logical_page") or ref.get("page") or 0) in logical_pages
        for account in accounts
        for ref in account.get("source_refs") or []
        if isinstance(ref, dict)
    )
    assert all(
        int(ref.get("page") or ref.get("logical_page") or 0) in logical_pages
        for record in repayment_records
        for ref in record.get("source_cell_refs") or []
        if isinstance(ref, dict) and (ref.get("page") or ref.get("logical_page"))
    )
    assert isinstance(native_business.get("credit_accounts") or [], list)

    inquiries = business.get("inquiry_records") or []
    for inquiry_type in {row.get("inquiry_type") for row in inquiries}:
        sequences = [row.get("sequence") for row in inquiries if row.get("inquiry_type") == inquiry_type]
        # OCR-visible gaps remain explicit evidence of a missed source row, but
        # duplicate/backward row numbers must not leak into reconstruction.
        assert sequences == sorted(set(sequences))

    assert validate_projection_payload("community", payload).valid
    v2_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert v2_validation.valid, v2_validation.errors
    assert payload["document"]["domain_schema"]["version"] == "2.0.0"
    v2_datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    for dataset in v2_datasets.values():
        assert dataset["row_count"] == len(dataset.get("rows") or [])
        record_ids = [str(row.get("record_id") or "") for row in dataset.get("rows") or []]
        assert all(record_ids)
        assert len(record_ids) == len(set(record_ids))
    assert v2_datasets["credit_accounts"]["row_count"] == len(accounts)
    canonical_monthly_statuses = {
        "*", "/", "#", "N", "1", "2", "3", "4", "5", "6", "7",
        "A", "B", "C", "D", "G", "M", "Z",
    }
    typed_linked_repayments = [
        record
        for record in linked_repayments
        if record.get("account_id")
        and str(record.get("status_code") or record.get("status") or "").strip().upper()
        in canonical_monthly_statuses
    ]
    assert (
        v2_datasets["credit_account_monthly_performance"]["row_count"]
        == len(typed_linked_repayments)
    )
    v2_statuses = {
        row["normalized"]["dataset_name"]: row["normalized"] for row in v2_datasets["dataset_status"]["rows"]
    }
    assert set(v2_statuses) <= set(PBOC_DATASET_ORDER) - {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
    assert all(
        status["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for status in v2_statuses.values()
    )
    assert all(
        status["observed_row_count"] == v2_datasets.get(name, {}).get("row_count", 0)
        for name, status in v2_statuses.items()
    )
    account_rows = [row["normalized"] for row in v2_datasets["credit_accounts"]["rows"]]
    monthly_record_rows = v2_datasets["credit_account_monthly_performance"]["rows"]
    monthly_rows = [row["normalized"] for row in monthly_record_rows]
    assert len(
        {
            (str(row.get("grid_id") or ""), str(row.get("performance_month") or ""))
            for row in monthly_rows
        }
    ) == len(monthly_rows)
    assert all("raw_detail_lines" not in row and "raw_detail_text" not in row for row in account_rows)
    account_id_set = {item["account_id"] for item in account_rows}
    assert all(
        not row.get("account_identifier")
        or str(row["account_identifier"]).replace("-", "").isalnum()
        and str(row["account_identifier"]).isascii()
        for row in monthly_rows
    )
    assert all(
        row.get("status_code") in {
            "*", "/", "#", "N", "1", "2", "3", "4", "5", "6", "7",
            "A", "B", "C", "D", "G", "M", "Z",
        }
        for row in monthly_rows
    )
    issue_rows = [row["normalized"] for row in v2_datasets["extraction_issues"]["rows"]]
    assert all(
        not row.get("target_dataset") or row["target_dataset"] in set(PBOC_DATASET_ORDER)
        for row in issue_rows
    )
    assert all(row.get("target_dataset") != "unknown" for row in issue_rows)
    final_unresolved_monthly_ids = {
        str(row.get("target_record_id") or "")
        for row in issue_rows
        if row.get("target_dataset") == "credit_account_monthly_performance"
        and row.get("issue_code") == "candidate_b_monthly_grid_owner_unresolved"
    }
    final_unresolved_monthly_grid_ids = {
        target.rsplit(":", 1)[0]
        for target in final_unresolved_monthly_ids
        if ":" in target
    }
    for wrapper, row in zip(monthly_record_rows, monthly_rows, strict=True):
        if row.get("account_id"):
            assert row["account_id"] in account_id_set
            continue
        monthly_id = str(wrapper.get("record_id") or row.get("monthly_performance_id") or "")
        assert row.get("extraction_status") == "review"
        assert (
            monthly_id in final_unresolved_monthly_ids
            or str(row.get("grid_id") or "") in final_unresolved_monthly_grid_ids
        )
    missing_identifier_ids = {
        str(account.get("account_id") or "")
        for account in account_rows
        if not account.get("account_identifier")
    }
    explicitly_reported_identifier_ids = {
        str(row.get("target_record_id") or "")
        for row in issue_rows
        if row.get("target_dataset") == "credit_accounts"
        and (
            row.get("field_name") == "account_identifier"
            or row.get("issue_code") == "candidate_b_account_table_missing"
        )
    }
    assert missing_identifier_ids <= explicitly_reported_identifier_ids
    forbidden_business_metadata = {
        "audit",
        "amount_bbox",
        "bbox",
        "raw_status",
        "recognition_source",
        "status_bbox",
    }
    for dataset_name, dataset in v2_datasets.items():
        if dataset_name in {
            "field_observations",
            "extraction_issues",
            "extraction_issue_evidence",
            "pboc_extension_fields",
            "dataset_status",
        }:
            continue
        for wrapper in dataset.get("rows", []):
            for value in (wrapper.get("normalized") or {}).values():
                assert not isinstance(value, (dict, list, tuple, set))
                if isinstance(value, str) and value[:1] in "[{" and value[-1:] in "]}":
                    try:
                        decoded = json.loads(value)
                    except (TypeError, ValueError):
                        decoded = None
                    assert not isinstance(decoded, (dict, list))
        assert all(
            not (forbidden_business_metadata & set(row.get(pool_name) or {}))
            for row in dataset.get("rows", [])
            for pool_name in ("normalized", "canonical_raw", "raw")
        )
    assert all(
        row.get("normalized", {}).get("status_code") in canonical_monthly_statuses
        for row in v2_datasets["credit_account_monthly_performance"]["rows"]
    )
    for dataset_name, dataset in v2_datasets.items():
        if dataset_name in {
            "field_observations",
            "extraction_issues",
            "extraction_issue_evidence",
            "pboc_extension_fields",
            "dataset_status",
        }:
            continue
        for wrapper in dataset.get("rows", []):
            values = wrapper.get("normalized") or {}
            has_review = values.get("extraction_status") == "review" or isinstance(
                wrapper.get("review"), dict
            )
            if not has_review:
                assert "confidence" not in wrapper
    months_by_grid: dict[str, list[str]] = {}
    for row in monthly_rows:
        grid_key = str(row.get("grid_id") or row.get("account_id") or "unresolved")
        months_by_grid.setdefault(grid_key, []).append(str(row.get("performance_month") or ""))
        if str(row.get("status_code") or "") in {"1", "2", "3", "4", "5", "6", "7"}:
            try:
                amount = Decimal(str(row.get("status_amount") or ""))
            except InvalidOperation:
                amount = Decimal(0)
            assert amount > 0
    assert all(months == sorted(months) for months in months_by_grid.values())
    assert not any(
        row.get("field_name") == "housing_fund_record_id"
        for row in issue_rows
    )
    unresolved_source_months = len(linked_repayments) - len(typed_linked_repayments)
    final_status_grid_issues = [
        row
        for row in issue_rows
        if row.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        and row.get("target_dataset") == "credit_account_monthly_performance"
    ]
    if unresolved_source_months:
        assert final_status_grid_issues
        assert v2_statuses["credit_account_monthly_performance"]["presence_status"] == "partial"
    assert not any(
        row.get("issue_code") == "pboc_cell_contract_unresolved"
        and row.get("target_dataset") == "credit_account_monthly_performance"
        and row.get("field_name") in {"status", "status_code"}
        for row in issue_rows
    )
    for row in v2_datasets.get("pboc_extension_fields", {}).get("rows", []):
        values = row.get("normalized") or {}
        if values.get("source_dataset") != "personal_detail_summary_cells":
            continue
        value = values.get("value")
        assert not isinstance(value, (dict, list))
        assert not (
            isinstance(value, str)
            and len(value) > 1
            and value[0] in "[{"
            and value[-1] in "]}"
        )
    assert all(not any(key.startswith(("observed__", "candidate__", "reason__")) for key in row) for row in issue_rows)
    evidence_rows = [
        row["normalized"] for row in v2_datasets.get("extraction_issue_evidence", {}).get("rows", [])
    ]
    issue_ids = {row["extraction_issue_id"] for row in issue_rows}
    assert evidence_rows
    assert all(row["extraction_issue_id"] in issue_ids for row in evidence_rows)
    assert all(
        set(row)
        <= {
            "extraction_issue_evidence_id",
            "extraction_issue_id",
            "evidence_kind",
            "evidence_path",
            "value_type",
            "string_value",
            "integer_value",
            "number_value",
            "boolean_value",
        }
        for row in evidence_rows
    )
    emitted_ids_by_dataset = {
        name: {
            str(value)
            for wrapper in dataset.get("rows", [])
            for value in (
                wrapper.get("record_id"),
                *((wrapper.get("normalized") or {}).get(key) for key in (wrapper.get("normalized") or {}) if key.endswith("_id")),
            )
            if value not in (None, "")
        }
        for name, dataset in v2_datasets.items()
    }
    reasons_by_issue: dict[str, set[str]] = {}
    for row in evidence_rows:
        if row.get("evidence_kind") == "reason" and row.get("string_value"):
            reasons_by_issue.setdefault(str(row["extraction_issue_id"]), set()).add(str(row["string_value"]))
    status_grid_issue_ids = {
        str(row["extraction_issue_id"])
        for row in final_status_grid_issues
    }
    reported_withheld_months = sum(
        int(row.get("integer_value") or 0)
        for row in evidence_rows
        if str(row.get("extraction_issue_id") or "") in status_grid_issue_ids
        and row.get("evidence_kind") == "observed"
        and row.get("evidence_path") == "withheld_month_count"
    )
    assert reported_withheld_months == unresolved_source_months
    for issue_id in status_grid_issue_ids:
        issue_evidence = [
            row
            for row in evidence_rows
            if str(row.get("extraction_issue_id") or "") == issue_id
            and row.get("evidence_kind") == "observed"
        ]
        withheld_count = next(
            int(row.get("integer_value") or 0)
            for row in issue_evidence
            if row.get("evidence_path") == "withheld_month_count"
        )
        withheld_months = sorted(
            str(row.get("string_value") or "")
            for row in issue_evidence
            if re.fullmatch(r"withheld_months\[\d+\]", str(row.get("evidence_path") or ""))
        )
        assert len(withheld_months) == withheld_count
        assert len(withheld_months) == len(set(withheld_months))
    non_emission_markers = (
        "withheld",
        "suppressed",
        "not_invented",
        "not_emitted",
        "unresolved",
        "record_not_silently_dropped",
        "silent_drop_prevented",
    )
    for row in issue_rows:
        dataset_name = str(row.get("target_dataset") or "")
        target_record_id = str(row.get("target_record_id") or "")
        if not dataset_name or not target_record_id:
            continue
        if target_record_id in emitted_ids_by_dataset.get(dataset_name, set()):
            continue
        assert any(
            marker in reason
            for reason in reasons_by_issue.get(str(row["extraction_issue_id"]), set())
            for marker in non_emission_markers
        )

    control_datasets = {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
    dash_only = re.compile(r"[-‐‑‒–—―－﹘﹣]+")
    assert not any(
        isinstance(value, str) and dash_only.fullmatch(value.strip())
        for dataset_name, dataset in v2_datasets.items()
        if dataset_name not in control_datasets
        for wrapper in dataset.get("rows", [])
        for value in (wrapper.get("normalized") or {}).values()
    )

    if expected_counts == (45, 944):
        amount_issue_targets = {
            str(row.get("target_record_id") or "")
            for row in issue_rows
            if row.get("target_dataset") == "credit_account_monthly_performance"
            and row.get("field_name") == "status_amount"
        }
        assert all(
            row.get("status_amount") not in (None, "")
            or str(wrapper.get("record_id") or row.get("monthly_performance_id") or "")
            in amount_issue_targets
            for wrapper, row in zip(monthly_record_rows, monthly_rows, strict=True)
        )

        overview_rows = [
            wrapper.get("normalized") or {}
            for wrapper in v2_datasets["credit_business_overview"]["rows"]
        ]
        for row in overview_rows:
            if row.get("metric_code") != "account_count" or row.get("numeric_value") in (None, ""):
                continue
            assert Decimal(str(row["numeric_value"])) <= Decimal(len(account_rows))

        def has_field_issue(dataset_name: str, record_id: str, field_name: str) -> bool:
            return any(
                row.get("target_dataset") == dataset_name
                and str(row.get("target_record_id") or "") == record_id
                and row.get("field_name") == field_name
                for row in issue_rows
            )

        account_12_february = [
            (wrapper, row)
            for wrapper, row in zip(monthly_record_rows, monthly_rows, strict=True)
            if row.get("grid_id") == "mg_p8_repayment_1"
            and row.get("performance_month") == "2020-02"
        ]
        if account_12_february:
            wrapper, row = account_12_february[0]
            assert row.get("status_code") == "C" or has_field_issue(
                "credit_account_monthly_performance",
                str(wrapper.get("record_id") or ""),
                "status_code",
            )
        else:
            unresolved_grid_issue_ids = {
                str(row.get("extraction_issue_id") or "")
                for row in final_status_grid_issues
            }
            grid_evidence = [
                row
                for row in evidence_rows
                if str(row.get("extraction_issue_id") or "") in unresolved_grid_issue_ids
            ]
            assert any(
                row.get("string_value") == "mg_p8_repayment_1"
                for row in grid_evidence
            ) and any(
                row.get("string_value") == "2020-02"
                and re.fullmatch(r"withheld_months\[\d+\]", str(row.get("evidence_path") or ""))
                for row in grid_evidence
            )

        residence_wrapper = next(
            wrapper
            for wrapper in v2_datasets["subject_residences"]["rows"]
            if (wrapper.get("normalized") or {}).get("sequence") == 5
        )
        residence = residence_wrapper.get("normalized") or {}
        assert "卢滨路" in str(residence.get("address") or "") or has_field_issue(
            "subject_residences",
            str(residence_wrapper.get("record_id") or ""),
            "address",
        )

        account_22_wrapper = next(
            wrapper
            for wrapper in v2_datasets["credit_accounts"]["rows"]
            if (wrapper.get("normalized") or {}).get("account_id")
            == "credit_account:non_revolving_loan:22"
        )
        account_22 = account_22_wrapper.get("normalized") or {}
        assert "蚂蚁商诚" in str(account_22.get("management_institution") or "") or has_field_issue(
            "credit_accounts",
            str(account_22_wrapper.get("record_id") or ""),
            "management_institution",
        )

        inquiry_1_wrapper = next(
            wrapper
            for wrapper in v2_datasets["inquiries"]["rows"]
            if (wrapper.get("normalized") or {}).get("query_channel") == "institution"
            and (wrapper.get("normalized") or {}).get("sequence") == 1
        )
        inquiry_1 = inquiry_1_wrapper.get("normalized") or {}
        assert "中国建设银行股份有限公司北京市分行" in str(
            inquiry_1.get("institution") or ""
        ) or has_field_issue(
            "inquiries",
            str(inquiry_1_wrapper.get("record_id") or ""),
            "institution",
        )

    agreement_rows = [row["normalized"] for row in v2_datasets["credit_agreements"]["rows"]]
    agreement_ids = {
        str(row.get("credit_agreement_id") or "")
        for row in agreement_rows
        if row.get("credit_agreement_id")
    }
    assert all(
        not row.get("target_record_id") or str(row["target_record_id"]) in agreement_ids
        for row in issue_rows
        if row.get("target_dataset") == "credit_agreements"
    )
    for required_field in ("institution", "facility_type", "effective_date"):
        missing_required_ids = {
            str(row.get("credit_agreement_id") or "")
            for row in agreement_rows
            if row.get(required_field) in (None, "")
        }
        explicitly_reported_required_ids = {
            str(row.get("target_record_id") or "")
            for row in issue_rows
            if row.get("target_dataset") == "credit_agreements"
            and row.get("field_name") == required_field
        }
        assert missing_required_ids <= explicitly_reported_required_ids
    expected_agreements = _EXPECTED_AGREEMENT_COUNTS.get(fixture.name)
    if expected_agreements is not None:
        assert len(agreement_rows) <= expected_agreements
        assert len({row.get("account_identifier") for row in agreement_rows}) == len(agreement_rows)
        printed_sequences = [row["sequence"] for row in agreement_rows if row.get("sequence") is not None]
        assert len(printed_sequences) == len(set(printed_sequences))
        assert all(1 <= sequence <= expected_agreements for sequence in printed_sequences)
        unresolved_sequences = len(agreement_rows) - len(printed_sequences)
        reported_sequences = sum(
            row.get("target_dataset") == "credit_agreements"
            and row.get("field_name") == "sequence"
            and row.get("issue_code") == "candidate_b_credit_agreement_sequence_unresolved"
            for row in issue_rows
        )
        assert reported_sequences >= unresolved_sequences
        if len(agreement_rows) < expected_agreements:
            assert v2_statuses["credit_agreements"]["presence_status"] == "partial"
            assert any(
                row.get("target_dataset") == "credit_agreements"
                and row.get("issue_code")
                in {
                    "source_sequence_or_count_gap",
                    "candidate_b_credit_agreement_population_gap",
                }
                for row in issue_rows
            )
    if fixture.name == "余泽熙7.15征信.pdf":
        assert [row.get("sequence") for row in agreement_rows] == list(range(1, 9))
        agreement_two = next(row for row in agreement_rows if row.get("sequence") == 2)
        assert agreement_two["account_identifier"] == (
            "B10711000H0001100000111111112446567900000"
        )
        assert agreement_two["institution"] == "中国光大银行股份有限公司"
        assert agreement_two["facility_type"] == "信用卡共享额度"
        assert agreement_two["effective_date"] == "2019-12-01"
        assert agreement_two["validity_type"] == "perpetual"
        assert agreement_two["facility_limit"] == "0"
        assert agreement_two["used_limit"] == "0"
        assert agreement_two["currency"] == "CNY"
        assert not any(
            row.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
            for row in issue_rows
        )
    expected_inquiries = _EXPECTED_INQUIRY_COUNTS.get(fixture.name)
    if expected_inquiries is not None and v2_datasets["inquiries"]["row_count"] != expected_inquiries:
        assert v2_datasets["inquiries"]["row_count"] < expected_inquiries
        assert v2_statuses["inquiries"]["presence_status"] == "partial"
        assert any(
            row.get("target_dataset") == "inquiries"
            and row.get("issue_code") in {"canonical_inquiry_sequence_gap", "source_sequence_or_count_gap"}
            for row in issue_rows
        )

    if fixture.name.startswith("叶永燕"):
        institutional = [row for row in inquiries if row.get("inquiry_type") == "institution"]
        personal = [row for row in inquiries if row.get("inquiry_type") == "personal"]

        # Population shortfalls are governed by the structured checks above;
        # this block pins the quality of values that were safely emitted.
        assert len(institutional) >= 94
        if len(institutional) < 96:
            gap = next(
                issue
                for issue in collect_extraction_issues(context)
                if issue.get("issue_code") == "canonical_inquiry_sequence_gap"
                and (issue.get("observed_value") or {}).get("inquiry_type") == "institution"
            )
            assert gap["candidate_value"]["missing_sequences"]
            assert "dataset_incomplete" in gap["reason_codes"]
        assert len(personal) == 16
        if len(institutional) == 96:
            assert [row["sequence"] for row in institutional] == list(range(1, 97))
        assert [row["sequence"] for row in personal] == list(range(1, 17))
        assert audit["applied_count"] >= 70


def _project_personal_detail_bundle(sealed, fixture: Path):
    """Project this plugin while an unrelated enterprise-only semantic contract is global."""
    return CreditReportPlugin().project_bundle(sealed, file_path=str(fixture))
