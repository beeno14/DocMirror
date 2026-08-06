# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)


def _bad_count(record_id: str, bbox: list[float]) -> dict[str, object]:
    return {
        "record_id": record_id,
        "column_label": "月份数",
        "value": "无法识别",
        "source_refs": [
            {
                "logical_page": 1,
                "bbox": bbox,
                "geometry_scope": "cell",
            }
        ],
    }


def test_business_uncertainty_reuses_complete_page_evidence_without_ocr() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {"personal_detail_summary_cells": [_bad_count("cell:1", [10, 10, 40, 30])]}
    plan = coordinator.plan(payload, canonical_audit={"unresolved_pages": []})
    calls = 0

    def page_ocr_loader(_pages, *, reason):
        nonlocal calls
        calls += 1
        return []

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[
            {
                "page": 1,
                "lines": [{"text": "2", "confidence": 0.98, "bbox": [10, 10, 40, 30]}],
            }
        ],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == 0
    assert plan.requires_second_pass is True
    assert plan.page_decisions == [
        {
            "logical_page": 1,
            "mode": "existing_complete_page_evidence",
            "ocr_invocations": 0,
            "target_count": 1,
        }
    ]

    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(plan.page_evidence.values(), affected_pages=plan.affected_pages)
    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")
    assert corrected["personal_detail_summary_cells"][0]["value"] == "2"


def test_business_uncertainties_are_grouped_into_one_page_ocr_request() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {
        "personal_detail_summary_cells": [
            _bad_count("cell:1", [10, 10, 40, 30]),
            _bad_count("cell:2", [50, 10, 80, 30]),
        ]
    }
    plan = coordinator.plan(payload, canonical_audit={"unresolved_pages": []})
    calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(pages, *, reason):
        calls.append((set(pages), reason))
        return [{"page": 1, "lines": [{"text": "2", "confidence": 0.98, "bbox": [10, 10, 40, 30]}]}]

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "lines": []}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [({1}, "business_field_evidence_insufficient")]
    assert len(plan.page_decisions) == 1
    assert plan.page_decisions[0]["target_count"] == 2
    assert plan.audit()["field_triggered_ocr_requests"] == 1


def test_overlapping_but_role_invalid_text_still_requests_one_page_ocr() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {"personal_detail_summary_cells": [_bad_count("cell:1", [10, 10, 40, 30])]}
    plan = coordinator.plan(payload, canonical_audit={"unresolved_pages": []})
    calls: list[set[int]] = []

    def page_ocr_loader(pages, *, reason):
        assert reason == "business_field_evidence_insufficient"
        calls.append(set(pages))
        return [{"page": 1, "lines": [{"text": "3", "bbox": [10, 10, 40, 30]}]}]

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "lines": [{"text": "月份数", "bbox": [10, 10, 40, 30]}]}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [{1}]
    assert plan.page_decisions[0]["ocr_invocations"] == 1


def test_unresolved_business_template_can_trigger_repair_but_topology_cannot() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    plan = coordinator.plan({}, canonical_audit={"unresolved_pages": [3]})
    calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(pages, *, reason):
        calls.append((set(pages), reason))
        return []

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 3, "lines": [{"text": "unknown", "bbox": [1, 1, 10, 10]}]}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [({3}, "business_schema_template_unresolved")]
    assert plan.audit()["topology_ocr_requests"] == 0
    assert plan.audit()["field_triggered_ocr_requests"] == 1


def test_structurally_missing_business_record_enters_the_page_repair_plan() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    plan = coordinator.plan(
        {},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=[
            {
                "issue_code": "candidate_b_unmatched_account_table_suppressed",
                "target_dataset": "credit_accounts",
                "target_record_id": "credit_account:credit_card:4",
                "source_refs": [{"logical_page": 20, "bbox": [10, 20, 300, 400]}],
                "reason_codes": ["printed_anchor_missing"],
            }
        ],
    )

    assert plan.affected_pages == (20,)
    assert plan.uncertainties[0].dataset_name == "credit_accounts"
    assert "candidate_b_unmatched_account_table_suppressed" in plan.uncertainties[0].reason_codes
