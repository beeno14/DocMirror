# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Business-shaped contracts for Candidate B extraction planning.

These cases deliberately sit between generic planner unit tests and the
section-specific production replays.  They model complete PBOC report
inventories (including multi-page account and inquiry fragments) and verify
that the deployed strategy selects the unchanged extractors needed for each
business document shape.  No OCR or PDF access is involved.

Field names in the repair matrix are documentary provenance: Candidate B
repairs evidence by owning section/stage, while the existing extractor keeps
field-level correction and withholding inside that scoped replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    ACCOUNT_SECTION,
    AGREEMENT_SECTION,
    CANDIDATE_B_STAGE_REGISTRY,
    INQUIRY_SECTION,
    LIABILITY_SECTION,
    MIXED_SECTION_ENVELOPE,
    POSTPAID_SECTION,
    PUBLIC_SECTION,
    REPORT_EXPLANATION_SECTION,
    REPORT_HEADER_SECTION,
    SECTION_TO_CANONICAL_DATASETS,
    SUMMARY_SECTION,
    build_candidate_b_section_census,
    plan_candidate_b_initial_extraction,
    resolve_candidate_b_repair_scope,
    stage_names_for_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    EagerExtractionStrategy,
    MaterializationMode,
    SectionState,
)


_BUSINESS_SECTION_FRAGMENTS: dict[str, tuple[int, ...]] = {
    REPORT_HEADER_SECTION: (1,),
    SUMMARY_SECTION: (2,),
    # A typical dense account inventory spans more than one logical page.
    ACCOUNT_SECTION: (3, 4),
    AGREEMENT_SECTION: (5,),
    LIABILITY_SECTION: (6,),
    POSTPAID_SECTION: (7,),
    PUBLIC_SECTION: (8,),
    # Inquiry headers and their headerless continuation are one fragment.
    INQUIRY_SECTION: (9, 10),
    REPORT_EXPLANATION_SECTION: (11,),
}

_MANDATORY_STAGES = (
    "header",
    "residence",
    "employment",
    "source_rows",
    "profile_details",
    "profile",
)


def _registration(
    logical_page: int,
    section: str,
    *,
    fragment_pages: tuple[int, ...],
) -> dict[str, object]:
    return {
        "logical_page": logical_page,
        "source_page": (logical_page + 1) // 2,
        "status": "registered",
        "template_id": section,
        "basis": "printed_heading_and_table_signature",
        "affected_source_datasets": sorted(
            SECTION_TO_CANONICAL_DATASETS[section]
        ),
        "printed_page": logical_page,
        "printed_total": max(
            page
            for pages in _BUSINESS_SECTION_FRAGMENTS.values()
            for page in pages
        ),
        "fragment_logical_pages": list(fragment_pages),
    }


def _business_audit(
    observed_sections: tuple[str, ...],
) -> dict[str, object]:
    registrations: list[dict[str, object]] = []
    fragment_groups: list[dict[str, object]] = []
    for section in observed_sections:
        fragment_pages = _BUSINESS_SECTION_FRAGMENTS[section]
        registrations.extend(
            _registration(
                page,
                section,
                fragment_pages=fragment_pages,
            )
            for page in fragment_pages
        )
        fragment_groups.append(
            {
                "template_id": section,
                "fragment_logical_pages": list(fragment_pages),
                "canonical_page": fragment_pages[0],
                "coverage_status": "full",
                "coverage_ratio": 1.0,
            }
        )

    logical_pages = sorted(
        int(registration["logical_page"])
        for registration in registrations
    )
    return {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": logical_pages,
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": [],
        "registrations": registrations,
        "fragment_groups": fragment_groups,
    }


@dataclass(frozen=True)
class _InitialBusinessCase:
    name: str
    sections: tuple[str, ...]
    expected_stages: tuple[str, ...]


_INITIAL_BUSINESS_CASES = (
    _InitialBusinessCase(
        name="identity_only_report",
        sections=(REPORT_HEADER_SECTION,),
        expected_stages=_MANDATORY_STAGES,
    ),
    _InitialBusinessCase(
        name="account_history_report",
        sections=(REPORT_HEADER_SECTION, ACCOUNT_SECTION),
        expected_stages=(
            "account_inventory",
            "monthly_repayments",
            "overdue",
            *_MANDATORY_STAGES[:1],
            "recovery",
            *_MANDATORY_STAGES[1:],
        ),
    ),
    _InitialBusinessCase(
        name="agreement_without_account_pages",
        sections=(REPORT_HEADER_SECTION, AGREEMENT_SECTION),
        # Agreement reconciliation consumes the account inventory contract,
        # even when the report has no account-detail section of its own.
        expected_stages=(
            "account_inventory",
            "credit_agreements",
            *_MANDATORY_STAGES,
        ),
    ),
    _InitialBusinessCase(
        name="inquiry_with_headerless_continuation",
        sections=(REPORT_HEADER_SECTION, INQUIRY_SECTION),
        expected_stages=(
            "inquiries",
            "notes",
            *_MANDATORY_STAGES,
        ),
    ),
    _InitialBusinessCase(
        name="full_personal_detailed_report",
        sections=tuple(_BUSINESS_SECTION_FRAGMENTS),
        expected_stages=CANDIDATE_B_STAGE_REGISTRY.ordered(),
    ),
)


@pytest.mark.parametrize(
    "case",
    _INITIAL_BUSINESS_CASES,
    ids=lambda case: case.name,
)
def test_business_document_shape_selects_only_required_strategy_stages(
    case: _InitialBusinessCase,
) -> None:
    census, plan = plan_candidate_b_initial_extraction(
        _business_audit(case.sections)
    )

    assert census.complete is True
    assert plan.mode is MaterializationMode.LAZY
    assert plan.ordered_stage_names == case.expected_stages
    assert set(plan.ordered_stage_names).isdisjoint(plan.skipped_stage_names)
    assert (
        set(plan.ordered_stage_names) | set(plan.skipped_stage_names)
        == set(CANDIDATE_B_STAGE_REGISTRY.names)
    )

    observed = set(case.sections)
    for section in _BUSINESS_SECTION_FRAGMENTS:
        expected_state = (
            SectionState.OBSERVED
            if section in observed
            else SectionState.ABSENT_PROVEN
        )
        assert census.census.state_for(section) is expected_state


def test_every_registered_business_output_has_one_deployed_strategy_owner() -> None:
    """Prevent a new extractor output from silently bypassing lazy planning."""

    internal_outputs = {"status_glyph_observations"}
    observed_stage_outputs: set[tuple[str, str]] = set()
    for stage in CANDIDATE_B_STAGE_REGISTRY.stages:
        for output_name in stage.output_names:
            if output_name in internal_outputs:
                continue
            owners = stage_names_for_datasets([output_name])
            assert stage.name in owners, (stage.name, output_name, owners)
            observed_stage_outputs.add((stage.name, output_name))

    assert observed_stage_outputs
    assert {
        stage.name for stage in CANDIDATE_B_STAGE_REGISTRY.stages
    } == {
        stage_name for stage_name, _output_name in observed_stage_outputs
    }


def test_full_business_document_has_identical_lazy_and_legacy_eager_stage_order() -> None:
    """A dense report changes orchestration mode, never extractor semantics."""

    audit = _business_audit(tuple(_BUSINESS_SECTION_FRAGMENTS))
    _lazy_census, lazy_plan = plan_candidate_b_initial_extraction(audit)
    _eager_census, eager_plan = plan_candidate_b_initial_extraction(
        audit,
        strategy=EagerExtractionStrategy(),
    )

    assert lazy_plan.mode is MaterializationMode.LAZY
    assert eager_plan.mode is MaterializationMode.EAGER
    assert (
        lazy_plan.ordered_stage_names
        == eager_plan.ordered_stage_names
        == CANDIDATE_B_STAGE_REGISTRY.ordered()
    )
    assert lazy_plan.requested_stage_names == eager_plan.requested_stage_names
    assert lazy_plan.skipped_stage_names == eager_plan.skipped_stage_names == ()


@dataclass(frozen=True)
class _RepairBusinessCase:
    name: str
    section: str
    page: int
    dataset: str
    field_name: str
    expected_dirty_stages: tuple[str, ...]
    expected_expanded_pages: tuple[int, ...]
    required_reused_stages: tuple[str, ...] = ()


_REPAIR_BUSINESS_CASES = (
    _RepairBusinessCase(
        name="monthly_amount_cell",
        section=ACCOUNT_SECTION,
        page=3,
        dataset="repayment_records",
        field_name="overdue_amount",
        expected_dirty_stages=(
            "account_inventory",
            "monthly_repayments",
            "credit_agreements",
            "overdue",
            "recovery",
            "source_rows",
        ),
        expected_expanded_pages=(3, 4),
    ),
    _RepairBusinessCase(
        name="agreement_institution_cell",
        section=AGREEMENT_SECTION,
        page=5,
        dataset="credit_lines",
        field_name="institution",
        expected_dirty_stages=("credit_agreements", "source_rows"),
        expected_expanded_pages=(5,),
        required_reused_stages=("account_inventory",),
    ),
    _RepairBusinessCase(
        name="liability_related_party_cell",
        section=LIABILITY_SECTION,
        page=6,
        dataset="repayment_liability_records",
        field_name="related_party_name",
        expected_dirty_stages=("liabilities", "source_rows"),
        expected_expanded_pages=(6,),
    ),
    _RepairBusinessCase(
        name="postpaid_history_cell",
        section=POSTPAID_SECTION,
        page=7,
        dataset="postpaid_payment_history",
        field_name="status",
        expected_dirty_stages=(
            "postpaid_records",
            "postpaid_history",
            "source_rows",
        ),
        expected_expanded_pages=(7,),
    ),
    _RepairBusinessCase(
        name="public_record_cell",
        section=PUBLIC_SECTION,
        page=8,
        dataset="public_records",
        field_name="record_content",
        expected_dirty_stages=("public", "source_rows"),
        expected_expanded_pages=(8,),
    ),
    _RepairBusinessCase(
        name="inquiry_institution_cell",
        section=INQUIRY_SECTION,
        page=9,
        dataset="inquiry_records",
        field_name="institution",
        expected_dirty_stages=("inquiries", "notes", "source_rows"),
        expected_expanded_pages=(9, 10),
    ),
    _RepairBusinessCase(
        name="summary_metric_cell",
        section=SUMMARY_SECTION,
        page=2,
        dataset="personal_detail_summary_cells",
        field_name="value",
        expected_dirty_stages=("summary", "source_rows"),
        expected_expanded_pages=(2,),
    ),
    _RepairBusinessCase(
        name="identity_document_cell",
        section=REPORT_HEADER_SECTION,
        page=1,
        dataset="identity_document_records",
        field_name="document_number",
        expected_dirty_stages=(
            "header",
            "residence",
            "employment",
            "source_rows",
            "profile_details",
            "profile",
        ),
        expected_expanded_pages=(1,),
    ),
    _RepairBusinessCase(
        name="report_explanation_line",
        section=REPORT_EXPLANATION_SECTION,
        page=11,
        dataset="report_notes",
        field_name="text",
        expected_dirty_stages=("source_rows",),
        expected_expanded_pages=(11,),
    ),
)


@pytest.mark.parametrize(
    "case",
    _REPAIR_BUSINESS_CASES,
    ids=lambda case: case.name,
)
def test_field_uncertainty_repairs_only_its_business_section_and_dependents(
    case: _RepairBusinessCase,
) -> None:
    audit = _business_audit(tuple(_BUSINESS_SECTION_FRAGMENTS))
    discovery = build_candidate_b_section_census(audit)
    repaired = build_candidate_b_section_census(audit)

    scope = resolve_candidate_b_repair_scope(
        discovery,
        repaired,
        affected_pages=[case.page],
        repair_dataset_names=[case.dataset],
    )
    plan = scope.plan(
        available_stage_names=CANDIDATE_B_STAGE_REGISTRY.names
    )

    assert case.field_name
    assert scope.fallback_reason == ""
    assert scope.dirty_sections == (case.section,)
    assert scope.expanded_pages == case.expected_expanded_pages
    assert scope.dirty_stage_names == case.expected_dirty_stages
    assert plan.mode is MaterializationMode.LAZY
    assert plan.ordered_stage_names == case.expected_dirty_stages
    assert set(case.required_reused_stages).issubset(plan.reused_stage_names)
    assert (
        set(plan.ordered_stage_names) | set(plan.reused_stage_names)
        == set(CANDIDATE_B_STAGE_REGISTRY.names)
    )


def test_liability_field_repair_on_mixed_page_reuses_account_strategies() -> None:
    """The repaired field's table owner, not its physical page, sets scope.

    Real PBOC logical pages can contain the tail of one account card followed
    by a repayment-liability card.  Re-OCR of the liability ``business_type``
    cell must not make the account inventory, its months, agreements, overdue
    derivation, or recovery dirty merely because both tables share page 6.

    This is intentionally a desired-architecture contract: frozen production
    code currently widens a repair to every section on the affected page.
    """

    audit = _business_audit(tuple(_BUSINESS_SECTION_FRAGMENTS))
    mixed_registration = next(
        registration
        for registration in audit["registrations"]
        if registration["logical_page"] == 6
    )
    mixed_registration.update(
        {
            "template_id": MIXED_SECTION_ENVELOPE,
            "affected_source_datasets": sorted(
                SECTION_TO_CANONICAL_DATASETS[ACCOUNT_SECTION]
                | SECTION_TO_CANONICAL_DATASETS[LIABILITY_SECTION]
            ),
            "section_table_owners": {
                "pt_6_account_tail": {
                    "template_id": ACCOUNT_SECTION,
                    "ownership_basis": "printed_account_anchor_continuation",
                },
                "pt_6_liability_1": {
                    "template_id": LIABILITY_SECTION,
                    "ownership_basis": "printed_liability_anchor",
                },
            },
        }
    )
    mixed_group = next(
        group
        for group in audit["fragment_groups"]
        if group["fragment_logical_pages"] == [6]
    )
    mixed_group["template_id"] = MIXED_SECTION_ENVELOPE

    discovery = build_candidate_b_section_census(audit)
    repaired = build_candidate_b_section_census(audit)
    assert discovery.complete is True
    assert repaired.complete is True

    scope = resolve_candidate_b_repair_scope(
        discovery,
        repaired,
        affected_pages=[6],
        repair_dataset_names=["repayment_liability_records"],
    )
    plan = scope.plan(
        available_stage_names=CANDIDATE_B_STAGE_REGISTRY.names
    )

    assert scope.fallback_reason == ""
    assert scope.dirty_sections == (LIABILITY_SECTION,)
    assert scope.dirty_stage_names == ("liabilities", "source_rows")
    assert plan.mode is MaterializationMode.LAZY
    assert plan.ordered_stage_names == ("liabilities", "source_rows")
    assert {
        "account_inventory",
        "monthly_repayments",
        "credit_agreements",
        "overdue",
        "recovery",
    } <= set(plan.reused_stage_names)


@pytest.mark.parametrize(
    ("mutation", "reason_prefix"),
    (
        (
            lambda audit: audit["corrected_evidence_conservation"].update(
                {"valid": False}
            ),
            "corrected_evidence_conservation_invalid",
        ),
        (
            lambda audit: audit.update({"unresolved_pages": [4]}),
            "canonical_unresolved_pages:4",
        ),
        (
            lambda audit: audit["fragment_groups"][2].update(
                {"coverage_status": "partial"}
            ),
            "canonical_fragment_coverage_incomplete",
        ),
        (
            lambda audit: audit["registrations"][3].update(
                {"affected_source_datasets": ["unknown_business_dataset"]}
            ),
            "canonical_dataset_unknown:unknown_business_dataset",
        ),
    ),
    ids=(
        "broken_conservation",
        "unresolved_account_page",
        "partial_fragment",
        "unknown_dataset",
    ),
)
def test_business_document_strategy_falls_back_before_using_ambiguous_census(
    mutation,
    reason_prefix: str,
) -> None:
    audit = _business_audit(tuple(_BUSINESS_SECTION_FRAGMENTS))
    mutation(audit)

    census, plan = plan_candidate_b_initial_extraction(audit)

    assert census.complete is False
    assert census.fallback_reason.startswith(reason_prefix)
    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.ordered_stage_names == CANDIDATE_B_STAGE_REGISTRY.ordered()
    assert plan.skipped_stage_names == ()
