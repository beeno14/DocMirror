# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    ACCOUNT_SECTION,
    BLANK_SECTION,
    CANDIDATE_B_STAGE_REGISTRY,
    INQUIRY_SECTION,
    LIABILITY_SECTION,
    MIXED_SECTION_ENVELOPE,
    REPORT_HEADER_SECTION,
    SECTION_TO_CANONICAL_DATASETS,
    build_candidate_b_section_census,
    candidate_b_repair_scope,
    plan_candidate_b_initial_extraction,
    resolve_candidate_b_repair_scope,
    stage_names_for_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    MaterializationMode,
    SectionState,
)


def _registration(
    logical_page: int,
    section: str,
    *,
    basis: str = "printed_heading_and_table_signature",
) -> dict[str, object]:
    return {
        "logical_page": logical_page,
        "source_page": logical_page,
        "status": "registered",
        "template_id": section,
        "basis": basis,
        "affected_source_datasets": sorted(
            SECTION_TO_CANONICAL_DATASETS[section]
        ),
        "printed_page": logical_page,
        "printed_total": 9,
    }


def _blank_registration(logical_page: int) -> dict[str, object]:
    return {
        "logical_page": logical_page,
        "source_page": logical_page,
        "status": "blank",
        "template_id": BLANK_SECTION,
        "basis": "blank_page_proven",
    }


def _fragment_group(
    section: str,
    logical_pages: list[int],
    *,
    canonical_page: int | None = None,
) -> dict[str, object]:
    return {
        "template_id": section,
        "fragment_logical_pages": list(logical_pages),
        "canonical_page": canonical_page or logical_pages[0],
        "coverage_status": "full",
        "coverage_ratio": 1.0,
    }


def _complete_audit(
    *,
    account_fragment: tuple[int, ...] = (2,),
    include_inquiries: bool = True,
    include_blank: bool = False,
) -> dict[str, object]:
    registrations = [_registration(1, REPORT_HEADER_SECTION)]
    registrations.extend(
        _registration(page, ACCOUNT_SECTION) for page in account_fragment
    )
    fragment_groups = [
        _fragment_group(REPORT_HEADER_SECTION, [1]),
        _fragment_group(ACCOUNT_SECTION, list(account_fragment)),
    ]
    next_page = max(account_fragment) + 1
    if include_inquiries:
        registrations.append(_registration(next_page, INQUIRY_SECTION))
        fragment_groups.append(
            _fragment_group(INQUIRY_SECTION, [next_page])
        )
        next_page += 1
    if include_blank:
        registrations.append(_blank_registration(next_page))

    logical_pages = sorted(
        int(registration["logical_page"]) for registration in registrations
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


def _registration_for_page(
    audit: dict[str, object],
    logical_page: int,
) -> dict[str, object]:
    registrations = audit["registrations"]
    assert isinstance(registrations, list)
    return next(
        registration
        for registration in registrations
        if registration["logical_page"] == logical_page
    )


def test_stage_registry_preserves_legacy_candidate_b_callback_order():
    assert CANDIDATE_B_STAGE_REGISTRY.ordered() == (
        "account_inventory",
        "monthly_repayments",
        "credit_agreements",
        "liabilities",
        "overdue",
        "inquiries",
        "public",
        "notes",
        "summary",
        "header",
        "recovery",
        "postpaid_records",
        "postpaid_history",
        "residence",
        "employment",
        "source_rows",
        "profile_details",
        "profile",
    )


def test_complete_census_selects_observed_sections_lazily():
    audit = _complete_audit()

    census, plan = plan_candidate_b_initial_extraction(audit)

    assert census.complete is True
    assert census.fallback_reason == ""
    assert census.census.state_for(REPORT_HEADER_SECTION) is SectionState.OBSERVED
    assert census.census.state_for(ACCOUNT_SECTION) is SectionState.OBSERVED
    assert census.census.state_for(INQUIRY_SECTION) is SectionState.OBSERVED
    assert plan.mode is MaterializationMode.LAZY
    assert plan.ordered_stage_names == (
        "account_inventory",
        "monthly_repayments",
        "overdue",
        "inquiries",
        "notes",
        "header",
        "recovery",
        "residence",
        "employment",
        "source_rows",
        "profile_details",
        "profile",
    )
    assert plan.audit().to_dict()["fallback_reason"] is None


def test_proven_absent_optional_sections_do_not_materialize():
    census, plan = plan_candidate_b_initial_extraction(
        _complete_audit(include_inquiries=False, include_blank=True)
    )

    assert census.complete is True
    assert census.census.state_for(INQUIRY_SECTION) is SectionState.ABSENT_PROVEN
    assert {
        "credit_agreements",
        "liabilities",
        "inquiries",
        "public",
        "notes",
        "summary",
        "postpaid_records",
        "postpaid_history",
    }.issubset(plan.skipped_stage_names)
    assert {
        "account_inventory",
        "monthly_repayments",
        "overdue",
        "recovery",
    }.issubset(plan.ordered_stage_names)


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    (
        (
            lambda audit: audit.__setitem__("unresolved_pages", [2]),
            "canonical_unresolved_pages:2",
        ),
        (
            lambda audit: _registration_for_page(audit, 2).__setitem__(
                "affected_source_datasets",
                ["unknown_source_dataset"],
            ),
            "canonical_dataset_unknown:unknown_source_dataset",
        ),
        (
            lambda audit: audit.pop("fragment_groups"),
            "canonical_fragment_groups_missing",
        ),
    ),
)
def test_unresolved_unknown_or_partial_audit_falls_back_eagerly(
    mutate,
    expected_reason: str,
):
    audit = _complete_audit()
    mutate(audit)

    census, plan = plan_candidate_b_initial_extraction(audit)

    assert census.complete is False
    assert census.fallback_reason == expected_reason
    assert all(
        state is SectionState.UNRESOLVED
        for _section, state in census.census.states
    )
    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.ordered_stage_names == CANDIDATE_B_STAGE_REGISTRY.ordered()
    assert plan.skipped_stage_names == ()
    assert plan.audit().to_dict()["fallback_reason"] == expected_reason


def test_page_owner_fingerprint_is_semantic_stable_and_auditable():
    audit = _complete_audit()
    with_non_owner_noise = deepcopy(audit)
    with_non_owner_noise["diagnostic_text"] = "OCR text is not ownership"

    first = build_candidate_b_section_census(audit)
    second = build_candidate_b_section_census(with_non_owner_noise)

    assert first.complete and second.complete
    assert first.page_ownership == second.page_ownership
    account_owner = first.fingerprint_by_page[2]
    assert account_owner.sections == (ACCOUNT_SECTION,)
    assert account_owner.fragment_logical_pages == (2,)
    assert len(account_owner.digest) == 64
    assert account_owner.to_dict()["digest"] == account_owner.digest

    changed = deepcopy(audit)
    _registration_for_page(changed, 2)["basis"] = "repaired_heading"
    changed_census = build_candidate_b_section_census(changed)
    assert changed_census.complete
    assert changed_census.fingerprint_by_page[2].digest != account_owner.digest


def test_explicit_blank_fragment_group_is_accepted_without_materialization():
    audit = _complete_audit(
        include_inquiries=False,
        include_blank=True,
    )
    audit["fragment_groups"].append(
        {
            "template_id": BLANK_SECTION,
            "fragment_logical_pages": [3],
            "canonical_page": 3,
        }
    )

    census, plan = plan_candidate_b_initial_extraction(audit)

    assert census.complete is True
    assert census.fingerprint_by_page[3].sections == ()
    assert plan.mode is MaterializationMode.LAZY


def test_owner_contract_and_fragment_contract_changes_invalidate_lazy_repair():
    audit = _complete_audit(include_inquiries=False)
    account_registration = _registration_for_page(audit, 2)
    account_registration["template_id"] = MIXED_SECTION_ENVELOPE
    account_registration["affected_source_datasets"] = sorted(
        SECTION_TO_CANONICAL_DATASETS[ACCOUNT_SECTION]
        | SECTION_TO_CANONICAL_DATASETS[LIABILITY_SECTION]
    )
    account_registration["section_table_owners"] = {
        "account_table": {
            "template_id": ACCOUNT_SECTION,
            "binding": "account_header_lattice",
        },
        "liability_table": {
            "template_id": LIABILITY_SECTION,
            "binding": "liability_header_lattice",
        },
    }
    account_group = audit["fragment_groups"][1]
    account_group["template_id"] = MIXED_SECTION_ENVELOPE
    account_group["section_table_owners"] = deepcopy(
        account_registration["section_table_owners"]
    )

    discovery = build_candidate_b_section_census(audit)
    assert discovery.complete

    owner_changed_audit = deepcopy(audit)
    owner_changed = _registration_for_page(owner_changed_audit, 2)
    owner_changed["section_table_owners"]["account_table"][
        "binding"
    ] = "repaired_account_header_lattice"
    owner_changed_census = build_candidate_b_section_census(
        owner_changed_audit
    )
    assert owner_changed_census.complete
    assert (
        owner_changed_census.fingerprint_by_page[2].table_owner_contract_digests
        != discovery.fingerprint_by_page[2].table_owner_contract_digests
    )
    owner_scope = resolve_candidate_b_repair_scope(
        discovery,
        owner_changed_census,
        affected_pages=[2],
        repair_dataset_names=["credit_accounts"],
    )
    assert owner_scope.eager_fallback_required is True
    assert owner_scope.ownership_changed_pages == (2,)

    fragment_changed_audit = deepcopy(audit)
    fragment_changed_audit["fragment_groups"][1]["coverage_ratio"] = 0.999
    fragment_changed_census = build_candidate_b_section_census(
        fragment_changed_audit
    )
    assert fragment_changed_census.complete
    assert (
        fragment_changed_census.fingerprint_by_page[2].fragment_contract_digest
        != discovery.fingerprint_by_page[2].fragment_contract_digest
    )
    fragment_scope = resolve_candidate_b_repair_scope(
        discovery,
        fragment_changed_census,
        affected_pages=[2],
        repair_dataset_names=["credit_accounts"],
    )
    assert fragment_scope.eager_fallback_required is True
    assert fragment_scope.ownership_changed_pages == (2,)


def test_dataset_mapping_accepts_canonical_alias_and_derived_names_fail_closed():
    assert stage_names_for_datasets(["credit_accounts"]) == (
        "account_inventory",
    )
    assert stage_names_for_datasets(["recovery_records"]) == ("recovery",)
    assert stage_names_for_datasets(["personal_profile"]) == ("profile",)
    assert stage_names_for_datasets(["credit_summary"]) == (
        "account_inventory",
        "liabilities",
        "inquiries",
    )
    assert stage_names_for_datasets(["report_query"]) == ("header",)
    assert stage_names_for_datasets(["annotation_statements"]) == ("notes",)
    assert stage_names_for_datasets(["housing_fund_records"]) == ("public",)
    assert stage_names_for_datasets(["credit_account_monthly_performance"]) == (
        "monthly_repayments",
    )
    assert stage_names_for_datasets(["postpaid_monthly_performance"]) == (
        "postpaid_history",
    )
    assert stage_names_for_datasets(
        ["personal_detail_credit_summary_metrics"]
    ) == ("account_inventory", "liabilities", "inquiries")
    assert stage_names_for_datasets(["credit_account_special_events"]) == (
        "account_inventory",
    )

    with pytest.raises(ValueError, match="unknown Candidate B dataset"):
        stage_names_for_datasets(["unknown_dataset"])
    with pytest.raises(ValueError, match="iterable of names"):
        stage_names_for_datasets("credit_accounts")


def test_repair_scope_expands_fragment_and_recomputes_section_dependents():
    audit = _complete_audit(account_fragment=(2, 3))
    discovery = build_candidate_b_section_census(audit)
    repaired = build_candidate_b_section_census(deepcopy(audit))

    scope = resolve_candidate_b_repair_scope(
        discovery,
        repaired,
        affected_pages=[2],
        repair_dataset_names=["repayment_records"],
    )

    assert scope.eager_fallback_required is False
    assert scope.affected_pages == (2,)
    assert scope.expanded_pages == (2, 3)
    assert scope.dirty_sections == (ACCOUNT_SECTION,)
    assert scope.dirty_stage_names == (
        "account_inventory",
        "monthly_repayments",
        "credit_agreements",
        "overdue",
        "recovery",
        "source_rows",
    )
    assert scope.audit()["fallback_reason"] is None

    _census, initial_plan = plan_candidate_b_initial_extraction(audit)
    repair_plan = scope.plan(
        available_stage_names=initial_plan.ordered_stage_names
    )
    assert repair_plan.mode is MaterializationMode.LAZY
    assert repair_plan.ordered_stage_names == scope.dirty_stage_names
    assert "inquiries" in repair_plan.reused_stage_names
    assert "header" in repair_plan.reused_stage_names


def test_business_repair_adapter_uses_only_page_scoped_uncertainty_datasets():
    audit = _complete_audit()
    scope = candidate_b_repair_scope(
        {
            "affected_pages": [2],
            "uncertainties": [
                {
                    "logical_pages": [2],
                    "dataset_name": "repayment_records",
                },
                {
                    # An unscoped first-pass issue did not receive repaired
                    # evidence and must not widen the incremental stage plan.
                    "logical_pages": [],
                    "dataset_name": "unknown_unrepaired_dataset",
                },
            ],
        },
        audit,
        deepcopy(audit),
    )

    assert scope.eager_fallback_required is False
    assert scope.affected_pages == (2,)
    assert "monthly_repayments" in scope.dirty_stage_names
    assert "inquiries" not in scope.dirty_stage_names


def test_repair_falls_back_when_affected_owner_fingerprint_changes():
    discovery_audit = _complete_audit()
    repaired_audit = deepcopy(discovery_audit)
    _registration_for_page(repaired_audit, 2)["basis"] = "repaired_heading"
    discovery = build_candidate_b_section_census(discovery_audit)
    repaired = build_candidate_b_section_census(repaired_audit)

    scope = resolve_candidate_b_repair_scope(
        discovery,
        repaired,
        affected_pages=[2],
        repair_dataset_names=["credit_accounts"],
    )

    assert scope.eager_fallback_required is True
    assert scope.ownership_changed_pages == (2,)
    assert scope.fallback_reason == "repair_affected_page_ownership_changed:2"
    plan = scope.plan(
        available_stage_names=CANDIDATE_B_STAGE_REGISTRY.names
    )
    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.ordered_stage_names == CANDIDATE_B_STAGE_REGISTRY.ordered()
    assert plan.audit().to_dict()["fallback_reason"] == scope.fallback_reason


@pytest.mark.parametrize(
    ("dataset_name", "reason"),
    (
        (
            "credit_accounts",
            "repair_dataset_page_ownership_mismatch:credit_accounts:credit_account_detail",
        ),
        ("unknown_dataset", "repair_dataset_unknown:unknown_dataset"),
    ),
)
def test_repair_dataset_owner_mapping_fails_closed(
    dataset_name: str,
    reason: str,
):
    audit = _complete_audit()
    census = build_candidate_b_section_census(audit)
    inquiry_page = 3

    scope = resolve_candidate_b_repair_scope(
        census,
        census,
        affected_pages=[inquiry_page],
        repair_dataset_names=[dataset_name],
    )

    assert scope.eager_fallback_required is True
    assert scope.fallback_reason == reason
