# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
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
                "source": "native_detail_table_cell",
                "logical_page": 1,
                "source_page": 1,
                "bbox": bbox,
                "evidence_ids": [f"native:{record_id}"],
                "geometry_scope": "cell",
            }
        ],
    }


def test_existing_complete_page_evidence_cannot_materially_correct_its_own_value() -> None:
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
                "source_page": 1,
                "lines": [
                    {
                        "text": "2",
                        "confidence": 0.98,
                        "bbox": [10, 10, 40, 30],
                        "evidence_ids": ["repair:cell:1"],
                    }
                ],
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
    cell = corrected["personal_detail_summary_cells"][0]
    assert cell["value"] is None
    assert cell["canonical_raw"]["value"] == "无法识别"
    assert overlay.audit()["applied_count"] == 0


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


def test_second_source_pass_invalidates_discovery_account_anchor_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repaired_page = {
        "page": 5,
        "lines": [
            {
                "text": "\u8d26\u62376",
                "bbox": [53.0, 229.0, 73.5, 240.0],
                "evidence_ids": ["ocr:sp0003:lp0005:0044"],
            }
        ],
    }
    plan = SimpleNamespace(
        affected_pages=(5,),
        page_evidence={5: repaired_page},
        requires_second_pass=True,
    )

    class _Coordinator:
        def __init__(self, _parse_result: object) -> None:
            pass

        def plan(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return plan

        def resolve_page_evidence(
            self, candidate: SimpleNamespace, **_kwargs: object
        ) -> SimpleNamespace:
            return candidate

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.business_repair."
        "BusinessUncertaintyRepairCoordinator",
        _Coordinator,
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context.parse_result = SimpleNamespace()
    context._personal_detail_extraction_issues = []
    context._initial_personal_detail_extraction_issues = []
    context._candidate_b_account_anchor_skeleton_cache = [
        {"account_id": "credit_account:non_revolving_loan:5"}
    ]
    context._cache = {"account_collections": ([{"account_id": "stale"}], [], [])}
    context._canonical_layout_projection_cache = object()
    context._pboc_layout_profile_cache = object()
    context._canonical_entity_context_ready = True
    context._source_evidence_pages = lambda: []
    context.full_page_ocr_evidence = lambda *_args, **_kwargs: []
    context.canonical_layout_audit = lambda: {"unresolved_pages": []}

    assert context.prepare_candidate_b_business_repair({}) is True
    assert "_candidate_b_account_anchor_skeleton_cache" not in context.__dict__
    assert context._candidate_b_pre_repair_account_anchor_inventory == (
        {"account_id": "credit_account:non_revolving_loan:5"},
    )
    assert context._cache == {}
    assert context._canonical_layout_projection_cache is None
    assert context._pboc_layout_profile_cache is None
    assert context._business_repair_evidence_by_page == {5: repaired_page}


def test_numeric_monthly_status_without_amount_is_withheld_and_explicitly_reported() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "repayment_records": [
            {
                "repayment_id": "repayment:2024-01",
                "year": 2024,
                "month": 1,
                "status": "2",
                "overdue_amount": None,
                "source_refs": [{"logical_page": 4, "geometry_scope": "table"}],
                "source_cell_refs": [
                    {
                        "logical_page": 4,
                        "bbox": [10, 20, 30, 40],
                        "geometry_scope": "cell",
                        "field_name": "status",
                    },
                    {
                        "logical_page": 4,
                        "bbox": [10, 40, 30, 60],
                        "geometry_scope": "cell",
                        "field_name": "overdue_amount",
                    },
                ],
            }
        ]
    }

    plan = BusinessUncertaintyRepairCoordinator(SimpleNamespace()).plan(
        payload,
        canonical_audit={"unresolved_pages": []},
    )
    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")
    anomalies = overlay.audit()["cell_anomalies"]

    assert corrected["repayment_records"][0]["status"] == "unknown"
    assert corrected["repayment_records"][0]["canonical_raw"]["status"] == "2"
    assert corrected["repayment_records"][0]["overdue_amount"] is None
    status_issue = next(
        anomaly
        for anomaly in anomalies
        if anomaly["field_name"] == "status_code"
        and "monthly_status_amount_unresolved" in anomaly["reason_codes"]
    )
    assert status_issue["value"] == "2"
    assert status_issue["normalized_value_withheld"] is True
    assert status_issue["source_refs"][0]["field_name"] == "status"
    amount_issue = next(
        anomaly
        for anomaly in anomalies
        if anomaly["field_name"] == "overdue_amount"
        and "monthly_status_amount_unresolved" in anomaly["reason_codes"]
    )
    assert amount_issue["source_refs"][0]["field_name"] == "overdue_amount"
    amount_uncertainty = next(item for item in plan.uncertainties if item.field_name == "overdue_amount")
    assert amount_uncertainty.source_refs[0]["bbox"] == [10, 40, 30, 60]
