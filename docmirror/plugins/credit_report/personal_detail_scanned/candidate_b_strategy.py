# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete extraction deployment metadata for Candidate B.

This module translates the canonical-layout audit into the generic planning
types from :mod:`extraction_strategy`.  It deliberately does not import or run
any extractor.  ``CandidateBPipeline`` remains the owner of extractor call
order, payload assembly, correction, and the document-wide release gates.

The adapter is conservative by construction.  A lazy census is complete only
when every conserved logical page has one unambiguous registered/blank owner,
every registered fragment has full canonical coverage, and every advertised
canonical dataset is known.  Any weaker audit produces an explicit reason that
the generic lazy strategy turns into an eager fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    ExtractionPlan,
    ExtractionRequest,
    ExtractionStage,
    ExtractionStrategy,
    LazyExtractionStrategy,
    SectionCensus,
    SectionState,
    StageRegistry,
)

REPORT_HEADER_SECTION = "report_header_and_identity"
SUMMARY_SECTION = "information_summary"
ACCOUNT_SECTION = "credit_account_detail"
LIABILITY_SECTION = "repayment_responsibility"
AGREEMENT_SECTION = "credit_agreement"
POSTPAID_SECTION = "postpaid_detail"
PUBLIC_SECTION = "public_information"
INQUIRY_SECTION = "annotations_and_inquiries"
REPORT_EXPLANATION_SECTION = "report_explanation"
MIXED_SECTION_ENVELOPE = "mixed_pboc_sections"
BLANK_SECTION = "blank_fragment"


_SECTION_DATASETS: dict[str, frozenset[str]] = {
    REPORT_HEADER_SECTION: frozenset(
        {
            "personal_report_metadata",
            "personal_profile",
            "identity_document_records",
            "mobile_phone_records",
            "spouse_records",
            "residence_records",
            "employment_records",
        }
    ),
    SUMMARY_SECTION: frozenset(
        {
            "personal_detail_summary_records",
            "personal_detail_summary_cells",
        }
    ),
    ACCOUNT_SECTION: frozenset(
        {
            "credit_accounts",
            "repayment_records",
            "personal_detail_account_events",
            "recovery_account_details",
        }
    ),
    LIABILITY_SECTION: frozenset({"repayment_liability_records"}),
    AGREEMENT_SECTION: frozenset({"credit_lines"}),
    POSTPAID_SECTION: frozenset(
        {"postpaid_accounts", "postpaid_payment_history"}
    ),
    PUBLIC_SECTION: frozenset({"public_records"}),
    INQUIRY_SECTION: frozenset(
        {"statements", "annotations", "inquiry_records"}
    ),
    REPORT_EXPLANATION_SECTION: frozenset({"report_notes"}),
}

SECTION_TO_CANONICAL_DATASETS: Mapping[str, frozenset[str]] = MappingProxyType(
    _SECTION_DATASETS
)

_dataset_sections = {
    dataset_name: section_name
    for section_name, dataset_names in _SECTION_DATASETS.items()
    for dataset_name in dataset_names
}
CANONICAL_DATASET_TO_SECTION: Mapping[str, str] = MappingProxyType(
    _dataset_sections
)
# Stable integration alias.  The longer canonical name remains available to
# make the provenance of the mapping explicit in audits and tests.
DATASET_TO_SECTION: Mapping[str, str] = CANONICAL_DATASET_TO_SECTION


# Declaration order exactly mirrors Candidate B's current source-pass callback
# order.  Keeping the callbacks as fine-grained stages avoids changing their
# issue-ledger/cache side effects when the pipeline starts using the registry.
# The registry describes orchestration only; callback ownership intentionally
# remains in candidate_b.py.
CANDIDATE_B_STAGE_REGISTRY = StageRegistry(
    (
        ExtractionStage(
            "account_inventory",
            section=ACCOUNT_SECTION,
            optional=True,
            output_names=("credit_accounts", "personal_detail_account_events"),
        ),
        ExtractionStage(
            "monthly_repayments",
            dependencies=frozenset({"account_inventory"}),
            section=ACCOUNT_SECTION,
            optional=True,
            output_names=("repayment_records", "status_glyph_observations"),
        ),
        ExtractionStage(
            "credit_agreements",
            # Existing agreement reconciliation reads account_collections().
            dependencies=frozenset({"account_inventory"}),
            section=AGREEMENT_SECTION,
            optional=True,
            output_names=("credit_lines",),
        ),
        ExtractionStage(
            "liabilities",
            section=LIABILITY_SECTION,
            optional=True,
            output_names=("repayment_liability_records",),
        ),
        ExtractionStage(
            "overdue",
            dependencies=frozenset(
                {"account_inventory", "monthly_repayments"}
            ),
            section=ACCOUNT_SECTION,
            optional=True,
            output_names=("overdue_records",),
        ),
        ExtractionStage(
            "inquiries",
            section=INQUIRY_SECTION,
            optional=True,
            output_names=("inquiry_records",),
        ),
        ExtractionStage(
            "public",
            section=PUBLIC_SECTION,
            optional=True,
            output_names=("public_records",),
        ),
        ExtractionStage(
            "notes",
            section=INQUIRY_SECTION,
            optional=True,
            output_names=("annotations", "statements"),
        ),
        ExtractionStage(
            "summary",
            section=SUMMARY_SECTION,
            optional=True,
            output_names=(
                "personal_detail_summary_records",
                "personal_detail_summary_cells",
            ),
        ),
        ExtractionStage(
            "header",
            section=REPORT_HEADER_SECTION,
            optional=False,
            output_names=(
                "personal_report_metadata",
                "identity_documents",
            ),
        ),
        ExtractionStage(
            "recovery",
            section=ACCOUNT_SECTION,
            optional=True,
            output_names=("recovery_records",),
        ),
        ExtractionStage(
            "postpaid_records",
            section=POSTPAID_SECTION,
            optional=True,
            output_names=("postpaid_records",),
        ),
        ExtractionStage(
            "postpaid_history",
            section=POSTPAID_SECTION,
            optional=True,
            output_names=("postpaid_payment_history",),
        ),
        ExtractionStage(
            "residence",
            section=REPORT_HEADER_SECTION,
            optional=False,
            output_names=("residence_records",),
        ),
        ExtractionStage(
            "employment",
            section=REPORT_HEADER_SECTION,
            optional=False,
            output_names=("employment_records",),
        ),
        ExtractionStage(
            "source_rows",
            optional=False,
            output_names=("personal_detail_source_rows",),
        ),
        ExtractionStage(
            "profile_details",
            section=REPORT_HEADER_SECTION,
            optional=False,
            output_names=("mobile_phone_records", "spouse_records"),
        ),
        ExtractionStage(
            "profile",
            section=REPORT_HEADER_SECTION,
            optional=False,
            output_names=("subject_profile",),
        ),
    )
)


_CANONICAL_DATASET_STAGES: dict[str, frozenset[str]] = {
    "personal_report_metadata": frozenset({"header"}),
    "personal_profile": frozenset({"profile"}),
    "identity_document_records": frozenset({"header"}),
    "mobile_phone_records": frozenset({"profile_details"}),
    "spouse_records": frozenset({"profile_details"}),
    "residence_records": frozenset({"residence"}),
    "employment_records": frozenset({"employment"}),
    "personal_detail_summary_records": frozenset({"summary"}),
    "personal_detail_summary_cells": frozenset({"summary"}),
    "credit_accounts": frozenset({"account_inventory"}),
    "personal_detail_account_events": frozenset({"account_inventory"}),
    "repayment_records": frozenset({"monthly_repayments"}),
    "recovery_account_details": frozenset({"recovery"}),
    "repayment_liability_records": frozenset({"liabilities"}),
    "credit_lines": frozenset({"credit_agreements"}),
    "postpaid_accounts": frozenset({"postpaid_records"}),
    "postpaid_payment_history": frozenset({"postpaid_history"}),
    "public_records": frozenset({"public"}),
    "statements": frozenset({"notes"}),
    "annotations": frozenset({"notes"}),
    "inquiry_records": frozenset({"inquiries"}),
    # Candidate B has no independent report-notes extractor.  Source rows are
    # the existing lossless output that observes this canonical section.
    "report_notes": frozenset({"source_rows"}),
}

CANONICAL_DATASET_TO_STAGE_NAMES: Mapping[str, frozenset[str]] = (
    MappingProxyType(_CANONICAL_DATASET_STAGES)
)


_REPAIR_DATASET_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "identity_documents": "identity_document_records",
        "recovery_records": "recovery_account_details",
        "postpaid_records": "postpaid_accounts",
        "credit_agreements": "credit_lines",
        "repayment_responsibilities": "repayment_liability_records",
        "inquiries": "inquiry_records",
        "subject_profile": "personal_profile",
        "subject_residences": "residence_records",
        "subject_employment": "employment_records",
        "subject_mobile_phones": "mobile_phone_records",
        "subject_spouse": "spouse_records",
        "report_metadata": "personal_report_metadata",
        "report_query": "personal_report_metadata",
        "annotation_statements": "annotations",
        "housing_fund_records": "public_records",
        "credit_account_monthly_performance": "repayment_records",
        "postpaid_monthly_performance": "postpaid_payment_history",
        "personal_detail_credit_summary_metrics": "credit_summary",
        "credit_account_special_transactions": "personal_detail_account_events",
        "credit_card_large_installments": "personal_detail_account_events",
        "credit_account_latest_repayments": "personal_detail_account_events",
        "credit_account_special_events": "personal_detail_account_events",
    }
)

_DERIVED_REPAIR_STAGES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "overdue_records": frozenset({"overdue"}),
        "credit_summary": frozenset(
            {
                "account_inventory",
                "liabilities",
                "inquiries",
            }
        ),
        "personal_detail_source_rows": frozenset({"source_rows"}),
    }
)


def stage_names_for_datasets(dataset_names: Iterable[str]) -> tuple[str, ...]:
    """Return the directly owning Candidate B stages for dataset names.

    Canonical source names, current projected aliases, and the small set of
    derived Candidate B payload names are accepted.  Unknown or empty names
    raise :class:`ValueError`; silently ignoring one would make a repair scope
    unsound.  Dependency/dependent expansion remains the planner's job.
    """

    if isinstance(dataset_names, (str, bytes, bytearray)):
        raise ValueError("dataset_names must be an iterable of names")
    stage_names: set[str] = set()
    for raw_dataset_name in dataset_names:
        dataset_name = str(raw_dataset_name or "").strip()
        if not dataset_name:
            raise ValueError("dataset_names cannot contain an empty name")
        canonical_name = _REPAIR_DATASET_ALIASES.get(
            dataset_name,
            dataset_name,
        )
        owners = CANONICAL_DATASET_TO_STAGE_NAMES.get(canonical_name)
        if owners is None:
            owners = _DERIVED_REPAIR_STAGES.get(
                canonical_name,
                _DERIVED_REPAIR_STAGES.get(dataset_name),
            )
        if owners is None:
            raise ValueError(f"unknown Candidate B dataset: {dataset_name}")
        stage_names.update(owners)
    return CANDIDATE_B_STAGE_REGISTRY.ordered(stage_names)


def _ordered_names(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _strict_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _strict_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (list, tuple)
    ):
        return None
    return value


def _strict_names(value: Any) -> tuple[str, ...] | None:
    sequence = _strict_sequence(value)
    if sequence is None:
        return None
    names: list[str] = []
    for raw_name in sequence:
        name = str(raw_name or "").strip()
        if not name or name in names:
            return None
        names.append(name)
    return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class PageOwnershipFingerprint:
    """Stable semantic ownership of one conserved logical page.

    OCR text and extracted values are intentionally absent.  The digest changes
    only when source ownership, section/table ownership, printed identity, or
    canonical fragment membership changes.
    """

    logical_page: int
    source_page: int
    template_id: str
    registration_basis: str
    sections: tuple[str, ...]
    canonical_datasets: tuple[str, ...]
    stage_names: tuple[str, ...]
    table_owners: tuple[tuple[str, str], ...]
    table_owner_contract_digests: tuple[tuple[str, str], ...]
    fragment_logical_pages: tuple[int, ...]
    canonical_page: int
    fragment_contract_digest: str
    printed_identity: tuple[int, int] | None
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-safe audit row."""

        return {
            "logical_page": self.logical_page,
            "source_page": self.source_page,
            "template_id": self.template_id,
            "registration_basis": self.registration_basis,
            "sections": list(self.sections),
            "canonical_datasets": list(self.canonical_datasets),
            "stage_names": list(self.stage_names),
            "table_owners": [
                {"table_id": table_id, "template_id": template_id}
                for table_id, template_id in self.table_owners
            ],
            "table_owner_contract_digests": dict(
                self.table_owner_contract_digests
            ),
            "fragment_logical_pages": list(self.fragment_logical_pages),
            "canonical_page": self.canonical_page,
            "fragment_contract_digest": self.fragment_contract_digest,
            "printed_identity": (
                list(self.printed_identity)
                if self.printed_identity is not None
                else None
            ),
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class CandidateBSectionCensus:
    """Candidate B section census plus per-page ownership fingerprints."""

    census: SectionCensus
    page_ownership: tuple[PageOwnershipFingerprint, ...] = ()
    fallback_reason: str = ""

    @property
    def complete(self) -> bool:
        return self.census.complete and not self.fallback_reason

    @property
    def fingerprint_by_page(self) -> dict[int, PageOwnershipFingerprint]:
        return {row.logical_page: row for row in self.page_ownership}

    def initial_request(
        self,
        *,
        requested_stage_names: Iterable[str] = (),
    ) -> ExtractionRequest:
        """Build the generic initial request with an explicit fallback cause."""

        return ExtractionRequest.initial(
            self.census,
            requested_stage_names=requested_stage_names,
            force_eager_reason=self.fallback_reason,
        )

    def audit(self) -> dict[str, Any]:
        """Serialize the census and ownership evidence for extraction audit."""

        return {
            "complete": self.complete,
            "fallback_reason": self.fallback_reason or None,
            "sections": {
                section: state.value for section, state in self.census.states
            },
            "page_ownership": [row.to_dict() for row in self.page_ownership],
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`audit` for generic serialization call sites."""

        return self.audit()


@dataclass(frozen=True, slots=True)
class CandidateBRepairScope:
    """Resolved incremental repair scope or explicit eager-fallback decision."""

    census: SectionCensus
    affected_pages: tuple[int, ...]
    expanded_pages: tuple[int, ...]
    dirty_sections: tuple[str, ...]
    dirty_stage_names: tuple[str, ...]
    ownership_changed_pages: tuple[int, ...] = ()
    fallback_reason: str = ""

    @property
    def eager_fallback_required(self) -> bool:
        return bool(self.fallback_reason)

    def extraction_request(
        self,
        *,
        available_stage_names: Iterable[str],
        requested_stage_names: Iterable[str] = (),
    ) -> ExtractionRequest:
        """Build a generic repair request consumable by a strategy planner."""

        return ExtractionRequest.repair(
            self.census,
            available_stage_names=available_stage_names,
            dirty_stage_names=self.dirty_stage_names,
            dirty_section_names=(
                section_name
                for section_name in self.dirty_sections
                if section_name in CANDIDATE_B_STAGE_REGISTRY.sections
            ),
            requested_stage_names=requested_stage_names,
            dependency_closure_known=not self.eager_fallback_required,
            force_eager_reason=self.fallback_reason,
        )

    def plan(
        self,
        *,
        available_stage_names: Iterable[str],
        strategy: ExtractionStrategy | None = None,
        requested_stage_names: Iterable[str] = (),
    ) -> ExtractionPlan:
        """Resolve the repair scope through the generic Candidate B registry."""

        planner = strategy or LazyExtractionStrategy()
        return planner.plan(
            CANDIDATE_B_STAGE_REGISTRY,
            self.extraction_request(
                available_stage_names=available_stage_names,
                requested_stage_names=requested_stage_names,
            ),
        )

    def audit(self) -> dict[str, Any]:
        """Serialize the resolved scope and its fail-safe decision."""

        return {
            "affected_pages": list(self.affected_pages),
            "expanded_pages": list(self.expanded_pages),
            "dirty_sections": list(self.dirty_sections),
            "dirty_stages": list(self.dirty_stage_names),
            "ownership_changed_pages": list(self.ownership_changed_pages),
            "eager_fallback_required": self.eager_fallback_required,
            "fallback_reason": self.fallback_reason or None,
            "section_census": {
                "complete": self.census.complete,
                "incomplete_reason": self.census.incomplete_reason or None,
                "sections": {
                    section: state.value for section, state in self.census.states
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`audit` for generic serialization call sites."""

        return self.audit()


def _incomplete_census(reason: str) -> CandidateBSectionCensus:
    states = {
        section_name: SectionState.UNRESOLVED
        for section_name in _SECTION_DATASETS
    }
    return CandidateBSectionCensus(
        census=SectionCensus.from_mapping(
            states,
            complete=False,
            incomplete_reason=reason,
        ),
        fallback_reason=reason,
    )


def _fingerprint_digest(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _semantic_plain(value: Any) -> Any:
    """Normalize detached audit metadata for deterministic semantic hashing."""

    if isinstance(value, Mapping):
        return {
            str(key): _semantic_plain(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_candidate_b_section_census(
    canonical_layout_audit: Mapping[str, Any] | None,
) -> CandidateBSectionCensus:
    """Build a fail-closed section census from ``canonical_layout_audit()``.

    The function accepts the public detached audit, rather than context or
    projection objects, so planning cannot mutate extraction state.
    """

    if not isinstance(canonical_layout_audit, Mapping):
        return _incomplete_census("canonical_audit_missing")
    corrected = canonical_layout_audit.get("corrected_evidence_conservation")
    if not isinstance(corrected, Mapping) or corrected.get("valid") is not True:
        return _incomplete_census("corrected_evidence_conservation_invalid")
    subset = canonical_layout_audit.get("canonical_subset_conservation")
    if not isinstance(subset, Mapping) or subset.get("valid") is not True:
        return _incomplete_census("canonical_subset_conservation_invalid")

    unresolved = _strict_sequence(canonical_layout_audit.get("unresolved_pages"))
    if unresolved is None:
        return _incomplete_census("canonical_unresolved_pages_missing")
    unresolved_pages = tuple(
        page
        for value in unresolved
        for page in (_strict_positive_int(value),)
        if page is not None
    )
    if len(unresolved_pages) != len(unresolved):
        return _incomplete_census("canonical_unresolved_pages_invalid")
    if len(unresolved_pages) != len(set(unresolved_pages)):
        return _incomplete_census("canonical_unresolved_pages_invalid")
    unresolved_page_set = set(unresolved_pages)
    unresolved_reason = (
        "canonical_unresolved_pages:"
        + ",".join(map(str, sorted(unresolved_pages)))
        if unresolved_pages
        else ""
    )

    conserved_raw = _strict_sequence(corrected.get("conserved_logical_pages"))
    if conserved_raw is None:
        return _incomplete_census("conserved_logical_pages_missing")
    conserved_pages = tuple(
        page
        for value in conserved_raw
        for page in (_strict_positive_int(value),)
        if page is not None
    )
    if len(conserved_pages) != len(conserved_raw) or len(conserved_pages) != len(
        set(conserved_pages)
    ):
        return _incomplete_census("conserved_logical_pages_invalid")

    registrations_raw = _strict_sequence(
        canonical_layout_audit.get("registrations")
    )
    if registrations_raw is None or not registrations_raw:
        return _incomplete_census("canonical_registrations_missing")

    registration_rows: dict[int, dict[str, Any]] = {}
    observed_sections: set[str] = set()
    for raw_registration in registrations_raw:
        if not isinstance(raw_registration, Mapping):
            return _incomplete_census("canonical_registration_invalid")
        logical_page = _strict_positive_int(raw_registration.get("logical_page"))
        source_page = _strict_positive_int(raw_registration.get("source_page"))
        if logical_page is None or source_page is None:
            return _incomplete_census("canonical_registration_page_invalid")
        if logical_page in registration_rows:
            return _incomplete_census(
                f"canonical_registration_duplicate_page:{logical_page}"
            )
        status = str(raw_registration.get("status") or "").strip()
        template_id = str(raw_registration.get("template_id") or "").strip()
        basis = str(raw_registration.get("basis") or "").strip()
        if not basis:
            return _incomplete_census(
                f"canonical_registration_basis_missing:{logical_page}"
            )

        if logical_page in unresolved_page_set or status == "unresolved":
            if status not in {"registered", "unresolved"}:
                return _incomplete_census(
                    f"canonical_registration_unresolved:{logical_page}"
                )
            registration_rows[logical_page] = {
                "logical_page": logical_page,
                "source_page": source_page,
                "template_id": "unresolved",
                "basis": basis,
                "sections": (),
                "datasets": (),
                "stage_names": (),
                "table_owners": (),
                "table_owner_contract_digests": (),
                "printed_identity": None,
                "blank": False,
                "unresolved": True,
            }
            continue

        if status == "blank" and template_id == BLANK_SECTION:
            registration_rows[logical_page] = {
                "logical_page": logical_page,
                "source_page": source_page,
                "template_id": template_id,
                "basis": basis,
                "sections": (),
                "datasets": (),
                "stage_names": (),
                "table_owners": (),
                "table_owner_contract_digests": (),
                "printed_identity": None,
                "blank": True,
                "unresolved": False,
            }
            continue
        if status != "registered":
            return _incomplete_census(
                f"canonical_registration_unresolved:{logical_page}"
            )

        dataset_names = _strict_names(
            raw_registration.get("affected_source_datasets")
        )
        if not dataset_names:
            return _incomplete_census(
                f"canonical_registration_datasets_missing:{logical_page}"
            )
        unknown_datasets = sorted(
            set(dataset_names).difference(CANONICAL_DATASET_TO_SECTION)
        )
        if unknown_datasets:
            return _incomplete_census(
                "canonical_dataset_unknown:" + ",".join(unknown_datasets)
            )

        table_owners: tuple[tuple[str, str], ...] = ()
        table_owner_contract_digests: tuple[tuple[str, str], ...] = ()
        if template_id == MIXED_SECTION_ENVELOPE:
            raw_owners = raw_registration.get("section_table_owners")
            if not isinstance(raw_owners, Mapping) or not raw_owners:
                return _incomplete_census(
                    f"mixed_section_owners_missing:{logical_page}"
                )
            normalized_owners: list[tuple[str, str]] = []
            normalized_owner_contracts: list[tuple[str, str]] = []
            owner_sections: set[str] = set()
            for raw_table_id, raw_owner in raw_owners.items():
                table_id = str(raw_table_id or "").strip()
                if not table_id or not isinstance(raw_owner, Mapping):
                    return _incomplete_census(
                        f"mixed_section_owner_invalid:{logical_page}"
                    )
                owner_template = str(raw_owner.get("template_id") or "").strip()
                if owner_template not in _SECTION_DATASETS:
                    return _incomplete_census(
                        f"mixed_section_owner_unknown:{logical_page}:{owner_template or 'missing'}"
                    )
                normalized_owners.append((table_id, owner_template))
                normalized_owner_contracts.append(
                    (
                        table_id,
                        _fingerprint_digest(
                            _semantic_plain(dict(raw_owner))
                        ),
                    )
                )
                owner_sections.add(owner_template)
            if len(normalized_owners) != len({row[0] for row in normalized_owners}):
                return _incomplete_census(
                    f"mixed_section_owner_duplicate:{logical_page}"
                )
            expected_datasets = frozenset(
                dataset_name
                for section_name in owner_sections
                for dataset_name in _SECTION_DATASETS[section_name]
            )
            if frozenset(dataset_names) != expected_datasets:
                return _incomplete_census(
                    f"mixed_section_dataset_ownership_ambiguous:{logical_page}"
                )
            sections = tuple(sorted(owner_sections))
            table_owners = tuple(sorted(normalized_owners))
            table_owner_contract_digests = tuple(
                sorted(normalized_owner_contracts)
            )
        elif template_id in _SECTION_DATASETS:
            sections = (template_id,)
            if frozenset(dataset_names) != _SECTION_DATASETS[template_id]:
                return _incomplete_census(
                    f"canonical_section_dataset_contract_mismatch:{logical_page}:{template_id}"
                )
        else:
            return _incomplete_census(
                f"canonical_template_unknown:{logical_page}:{template_id or 'missing'}"
            )

        stage_names = _ordered_names(
            stage_name
            for dataset_name in dataset_names
            for stage_name in CANONICAL_DATASET_TO_STAGE_NAMES[dataset_name]
        )
        printed_page = _strict_positive_int(raw_registration.get("printed_page"))
        printed_total = _strict_positive_int(raw_registration.get("printed_total"))
        if (printed_page is None) != (printed_total is None):
            return _incomplete_census(
                f"canonical_printed_identity_incomplete:{logical_page}"
            )
        printed_identity = (
            (printed_page, printed_total)
            if printed_page is not None and printed_total is not None
            else None
        )
        if printed_identity is not None and printed_identity[0] > printed_identity[1]:
            return _incomplete_census(
                f"canonical_printed_identity_invalid:{logical_page}"
            )
        observed_sections.update(sections)
        registration_rows[logical_page] = {
            "logical_page": logical_page,
            "source_page": source_page,
            "template_id": template_id,
            "basis": basis,
            "sections": sections,
            "datasets": dataset_names,
            "stage_names": stage_names,
            "table_owners": table_owners,
            "table_owner_contract_digests": (
                table_owner_contract_digests
            ),
            "printed_identity": printed_identity,
            "blank": False,
            "unresolved": False,
        }

    conserved_page_set = set(conserved_pages)
    extra_registration_pages = set(registration_rows) - conserved_page_set
    # Canonical repair can retain an explicit trailing blank fragment that has
    # no corrected business evidence and therefore is intentionally absent
    # from the conserved plane.  It is not a section owner and must not make
    # every real page fall back to eager extraction.  Any extra non-blank page,
    # or any conserved page without a registration, still fails closed.
    if extra_registration_pages and all(
        registration_rows[page]["blank"]
        and not registration_rows[page]["unresolved"]
        for page in extra_registration_pages
    ):
        for page in extra_registration_pages:
            registration_rows.pop(page)
    if set(registration_rows) != conserved_page_set:
        return _incomplete_census("canonical_registration_page_census_mismatch")
    if not unresolved_pages and REPORT_HEADER_SECTION not in observed_sections:
        return _incomplete_census("required_header_section_not_observed")

    fragment_groups_raw = _strict_sequence(
        canonical_layout_audit.get("fragment_groups")
    )
    if fragment_groups_raw is None:
        return _incomplete_census("canonical_fragment_groups_missing")
    fragment_by_page: dict[int, dict[str, Any]] = {}
    fragment_members_seen: set[int] = set()
    non_reusable_fragment_pages: set[int] = set()
    for raw_group in fragment_groups_raw:
        if not isinstance(raw_group, Mapping):
            return _incomplete_census("canonical_fragment_group_invalid")
        logicals_raw = _strict_sequence(raw_group.get("fragment_logical_pages"))
        if logicals_raw is None or not logicals_raw:
            return _incomplete_census("canonical_fragment_members_missing")
        logicals = tuple(
            page
            for value in logicals_raw
            for page in (_strict_positive_int(value),)
            if page is not None
        )
        if len(logicals) != len(logicals_raw) or len(logicals) != len(set(logicals)):
            return _incomplete_census("canonical_fragment_members_invalid")
        if any(page not in registration_rows for page in logicals):
            return _incomplete_census("canonical_fragment_owner_missing")
        if any(page in fragment_members_seen for page in logicals):
            return _incomplete_census("canonical_fragment_members_overlap")
        fragment_members_seen.update(logicals)
        group_rows = [registration_rows[page] for page in logicals]
        if any(row["unresolved"] for row in group_rows):
            # A fragment contract is atomic: if one member is unresolved, none
            # of its members can safely be reused.  Unrelated, independently
            # registered fragments remain available to repair planning.
            non_reusable_fragment_pages.update(logicals)
            continue
        if all(row["blank"] for row in group_rows):
            if str(raw_group.get("template_id") or "").strip() not in {
                "",
                BLANK_SECTION,
            }:
                return _incomplete_census(
                    "canonical_blank_fragment_template_mismatch"
                )
            continue
        if any(row["blank"] for row in group_rows):
            return _incomplete_census("canonical_fragment_owner_missing")
        if str(raw_group.get("coverage_status") or "") != "full":
            return _incomplete_census(
                "canonical_fragment_coverage_incomplete:"
                + ",".join(map(str, sorted(logicals)))
            )
        try:
            coverage_ratio = float(raw_group.get("coverage_ratio"))
        except (TypeError, ValueError):
            return _incomplete_census("canonical_fragment_coverage_invalid")
        if coverage_ratio < 0.985:
            return _incomplete_census("canonical_fragment_coverage_invalid")
        group_template = str(raw_group.get("template_id") or "").strip()
        if {registration_rows[page]["template_id"] for page in logicals} != {
            group_template
        }:
            return _incomplete_census("canonical_fragment_template_mismatch")
        canonical_page = _strict_positive_int(raw_group.get("canonical_page"))
        if canonical_page is None:
            return _incomplete_census("canonical_fragment_page_invalid")
        group = {
            "fragment_logical_pages": tuple(sorted(logicals)),
            "canonical_page": canonical_page,
            "fragment_contract_digest": _fingerprint_digest(
                _semantic_plain(dict(raw_group))
            ),
        }
        for page in logicals:
            fragment_by_page[page] = group

    registered_pages = {
        page
        for page, row in registration_rows.items()
        if (
            not row["blank"]
            and not row["unresolved"]
            and page not in non_reusable_fragment_pages
        )
    }
    if set(fragment_by_page) != registered_pages:
        return _incomplete_census("canonical_fragment_page_census_mismatch")

    page_ownership: list[PageOwnershipFingerprint] = []
    for logical_page in sorted(registration_rows):
        row = registration_rows[logical_page]
        if row["unresolved"] or logical_page in non_reusable_fragment_pages:
            continue
        fragment = fragment_by_page.get(
            logical_page,
            {
                "fragment_logical_pages": (logical_page,),
                "canonical_page": logical_page,
                "fragment_contract_digest": _fingerprint_digest(
                    {
                        "canonical_page": logical_page,
                        "fragment_logical_pages": (logical_page,),
                        "registration_status": (
                            "blank" if row["blank"] else "registered"
                        ),
                    }
                ),
            },
        )
        fingerprint_payload = {
            "logical_page": logical_page,
            "source_page": row["source_page"],
            "template_id": row["template_id"],
            "registration_basis": row["basis"],
            "sections": row["sections"],
            "canonical_datasets": row["datasets"],
            "stage_names": row["stage_names"],
            "table_owners": row["table_owners"],
            "table_owner_contract_digests": row[
                "table_owner_contract_digests"
            ],
            "fragment_logical_pages": fragment["fragment_logical_pages"],
            "canonical_page": fragment["canonical_page"],
            "fragment_contract_digest": fragment[
                "fragment_contract_digest"
            ],
            "printed_identity": row["printed_identity"],
        }
        page_ownership.append(
            PageOwnershipFingerprint(
                **fingerprint_payload,
                digest=_fingerprint_digest(fingerprint_payload),
            )
        )

    states = {
        section_name: (
            SectionState.OBSERVED
            if section_name in observed_sections
            else (
                SectionState.UNRESOLVED
                if unresolved_pages
                else SectionState.ABSENT_PROVEN
            )
        )
        for section_name in _SECTION_DATASETS
    }
    return CandidateBSectionCensus(
        census=SectionCensus.from_mapping(
            states,
            complete=not unresolved_pages,
            incomplete_reason=unresolved_reason,
        ),
        page_ownership=tuple(page_ownership),
        fallback_reason=unresolved_reason,
    )


def section_census_from_canonical_audit(
    canonical_layout_audit: Mapping[str, Any] | None,
) -> CandidateBSectionCensus:
    """Stable public adapter for Candidate B canonical-layout audits."""

    return build_candidate_b_section_census(canonical_layout_audit)


def plan_candidate_b_initial_extraction(
    canonical_layout_audit: Mapping[str, Any] | None,
    *,
    strategy: ExtractionStrategy | None = None,
    requested_stage_names: Iterable[str] = (),
) -> tuple[CandidateBSectionCensus, ExtractionPlan]:
    """Build the census and its initial materialization plan in one call."""

    census = build_candidate_b_section_census(canonical_layout_audit)
    planner = strategy or LazyExtractionStrategy()
    return (
        census,
        planner.plan(
            CANDIDATE_B_STAGE_REGISTRY,
            census.initial_request(
                requested_stage_names=requested_stage_names,
            ),
        ),
    )


def _repair_fallback(
    census: SectionCensus,
    *,
    affected_pages: tuple[int, ...],
    changed_pages: Iterable[int] = (),
    reason: str,
) -> CandidateBRepairScope:
    return CandidateBRepairScope(
        census=census,
        affected_pages=affected_pages,
        expanded_pages=affected_pages,
        dirty_sections=(),
        dirty_stage_names=(),
        ownership_changed_pages=tuple(sorted(set(changed_pages))),
        fallback_reason=reason,
    )


def resolve_candidate_b_repair_scope(
    discovery: CandidateBSectionCensus,
    repaired: CandidateBSectionCensus,
    *,
    affected_pages: Iterable[int],
    repair_dataset_names: Iterable[str] = (),
) -> CandidateBRepairScope:
    """Resolve a repair to safely reusable sections/stages, or replay eagerly.

    Reuse is allowed only when the complete semantic ownership fingerprint is
    unchanged for every page.  A repaired page still expands to its whole
    canonical fragment group.  When the repair supplies explicit, unambiguous
    dataset owners, however, only those canonical sections are dirtied; without
    dataset owners the conservative page-wide section behaviour is preserved.
    Source rows remain page-wide, and all dependents of the selected section
    stages are rebuilt.
    """

    normalized_pages: list[int] = []
    for raw_page in affected_pages:
        page = _strict_positive_int(raw_page)
        if page is None or page in normalized_pages:
            fallback_pages = tuple(sorted(normalized_pages))
            return _repair_fallback(
                repaired.census,
                affected_pages=fallback_pages,
                reason="repair_affected_pages_invalid",
            )
        normalized_pages.append(page)
    affected = tuple(sorted(normalized_pages))
    if not affected:
        return _repair_fallback(
            repaired.census,
            affected_pages=(),
            reason="repair_affected_pages_missing",
        )
    if isinstance(repair_dataset_names, (str, bytes, bytearray)):
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repair_dataset_names_invalid",
        )
    try:
        repair_datasets = tuple(repair_dataset_names)
    except Exception:
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repair_dataset_names_invalid",
        )

    def ownership_is_usable(census: CandidateBSectionCensus) -> bool:
        return census.complete or (
            census.fallback_reason.startswith("canonical_unresolved_pages:")
            and bool(census.page_ownership)
        )

    if not ownership_is_usable(discovery):
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="discovery_census_incomplete:"
            + (discovery.fallback_reason or "unknown"),
        )
    if not ownership_is_usable(repaired):
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repaired_census_incomplete:"
            + (repaired.fallback_reason or "unknown"),
        )

    discovery_by_page = discovery.fingerprint_by_page
    repaired_by_page = repaired.fingerprint_by_page
    membership_changed = set(discovery_by_page).symmetric_difference(repaired_by_page)
    membership_changed_outside = sorted(membership_changed.difference(affected))
    if membership_changed_outside:
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            changed_pages=membership_changed_outside,
            reason="repair_page_census_changed:"
            + ",".join(map(str, membership_changed_outside)),
        )
    missing_affected = sorted(
        set(affected).difference(set(discovery_by_page) | set(repaired_by_page))
    )
    if missing_affected and not repair_datasets:
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repair_page_ownership_missing:"
            + ",".join(map(str, missing_affected)),
        )

    changed_pages = tuple(
        page
        for page in sorted(set(discovery_by_page).intersection(repaired_by_page))
        if discovery_by_page[page].digest != repaired_by_page[page].digest
    )
    changed_outside = tuple(sorted(set(changed_pages).difference(affected)))
    if changed_outside:
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            changed_pages=changed_pages,
            reason="repair_ownership_changed_outside_scope:"
            + ",".join(map(str, changed_outside)),
        )
    ownership_changed_pages = tuple(
        sorted(set(changed_pages) | membership_changed.intersection(affected))
    )

    expanded_pages = set(affected)
    for page in affected:
        for ownership_by_page in (discovery_by_page, repaired_by_page):
            owner = ownership_by_page.get(page)
            if owner is not None:
                expanded_pages.update(owner.fragment_logical_pages)
    known_ownership_pages = set(discovery_by_page) | set(repaired_by_page)
    unresolved_expanded_pages = expanded_pages.difference(known_ownership_pages)
    if unresolved_expanded_pages.difference(affected):
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repair_fragment_ownership_incomplete",
        )

    page_owned_sections = {
        section_name
        for page in expanded_pages
        for ownership_by_page in (discovery_by_page, repaired_by_page)
        for owner in (ownership_by_page.get(page),)
        if owner is not None
        for section_name in owner.sections
    }
    if not page_owned_sections and not repair_datasets:
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repair_page_has_no_owned_section",
        )

    selected_sections: set[str] = set()
    selected_stage_names: set[str] = set()
    explicit_dataset_owner = False
    for raw_dataset_name in repair_datasets:
        explicit_dataset_owner = True
        dataset_name = str(raw_dataset_name or "").strip()
        if not dataset_name:
            return _repair_fallback(
                repaired.census,
                affected_pages=affected,
                reason="repair_dataset_name_missing",
            )
        canonical_name = _REPAIR_DATASET_ALIASES.get(dataset_name, dataset_name)
        stage_names = CANONICAL_DATASET_TO_STAGE_NAMES.get(canonical_name)
        if stage_names is None:
            stage_names = _DERIVED_REPAIR_STAGES.get(
                canonical_name,
                _DERIVED_REPAIR_STAGES.get(dataset_name),
            )
        if stage_names is None:
            return _repair_fallback(
                repaired.census,
                affected_pages=affected,
                reason=f"repair_dataset_unknown:{dataset_name}",
            )
        canonical_section = CANONICAL_DATASET_TO_SECTION.get(canonical_name)
        if canonical_section is None:
            stage_sections = {
                stage.section
                for stage in CANDIDATE_B_STAGE_REGISTRY.stages
                if stage.name in stage_names
                if stage.section is not None
            }
            if len(stage_sections) != 1:
                return _repair_fallback(
                    repaired.census,
                    affected_pages=affected,
                    reason=f"repair_dataset_ownership_ambiguous:{dataset_name}",
                )
            canonical_section = next(iter(stage_sections))
        if (
            canonical_section not in page_owned_sections
            and not missing_affected
            and not ownership_changed_pages
        ):
            return _repair_fallback(
                repaired.census,
                affected_pages=affected,
                reason=(
                    "repair_dataset_page_ownership_mismatch:"
                    f"{dataset_name}:{canonical_section}"
                ),
            )
        selected_sections.add(canonical_section)
        selected_stage_names.update(stage_names)

    dirty_sections = (
        selected_sections if explicit_dataset_owner else page_owned_sections
    )
    if ownership_changed_pages:
        dirty_sections.update(page_owned_sections)
    if explicit_dataset_owner and not dirty_sections:
        return _repair_fallback(
            repaired.census,
            affected_pages=affected,
            reason="repair_dataset_ownership_missing",
        )

    dirty_roots = set(
        CANDIDATE_B_STAGE_REGISTRY.stages_for_sections(
            dirty_sections.intersection(CANDIDATE_B_STAGE_REGISTRY.sections)
        )
    )
    dirty_roots.update(selected_stage_names)
    # Source rows are page-wide and therefore change for every admitted repair.
    dirty_roots.add("source_rows")

    dirty_stages = CANDIDATE_B_STAGE_REGISTRY.dependent_closure(dirty_roots)
    return CandidateBRepairScope(
        census=repaired.census,
        affected_pages=affected,
        expanded_pages=tuple(sorted(expanded_pages)),
        dirty_sections=tuple(sorted(dirty_sections)),
        dirty_stage_names=CANDIDATE_B_STAGE_REGISTRY.ordered(dirty_stages),
        ownership_changed_pages=ownership_changed_pages,
    )


def _plan_value(plan: Any, name: str, default: Any) -> Any:
    if isinstance(plan, Mapping):
        return plan.get(name, default)
    return getattr(plan, name, default)


def candidate_b_repair_scope(
    plan: Any,
    discovery_audit: Mapping[str, Any] | None,
    repaired_audit: Mapping[str, Any] | None,
) -> CandidateBRepairScope:
    """Adapt a business-repair plan and two audits to a safe stage scope.

    ``plan`` is intentionally duck-typed to avoid coupling this metadata-only
    module to the repair coordinator.  Candidate B may pass its live
    ``BusinessRepairPlan`` or a detached mapping with ``affected_pages`` and
    ``uncertainties``.  Only page-scoped uncertainty datasets participate;
    unscoped findings were not repaired and must not widen an incremental
    materialization decision.
    """

    discovery = section_census_from_canonical_audit(discovery_audit)
    repaired = section_census_from_canonical_audit(repaired_audit)
    if plan is None:
        return _repair_fallback(
            repaired.census,
            affected_pages=(),
            reason="repair_plan_missing",
        )

    raw_pages = _strict_sequence(_plan_value(plan, "affected_pages", None))
    if raw_pages is None:
        return _repair_fallback(
            repaired.census,
            affected_pages=(),
            reason="repair_plan_affected_pages_invalid",
        )
    affected_pages: list[int] = []
    for raw_page in raw_pages:
        page = _strict_positive_int(raw_page)
        if page is None or page in affected_pages:
            return _repair_fallback(
                repaired.census,
                affected_pages=tuple(sorted(affected_pages)),
                reason="repair_plan_affected_pages_invalid",
            )
        affected_pages.append(page)
    affected_set = set(affected_pages)

    raw_reconstruction = _plan_value(plan, "reconstruction_evidence", None)
    if raw_reconstruction is None:
        # Compatibility for detached/legacy plans that predate the explicit
        # reconstruction ledger: their affected pages remain page-replay scope.
        materialization_pages = tuple(affected_pages)
    elif isinstance(raw_reconstruction, Mapping):
        parsed_reconstruction_pages = [
            _strict_positive_int(raw_page) for raw_page in raw_reconstruction
        ]
        if any(page is None for page in parsed_reconstruction_pages):
            return _repair_fallback(
                repaired.census,
                affected_pages=tuple(sorted(affected_pages)),
                reason="repair_reconstruction_pages_invalid",
            )
        materialization_pages = tuple(
            sorted({int(page) for page in parsed_reconstruction_pages if page is not None})
        )
    else:
        return _repair_fallback(
            repaired.census,
            affected_pages=tuple(sorted(affected_pages)),
            reason="repair_reconstruction_evidence_invalid",
        )
    materialization_set = set(materialization_pages)

    raw_uncertainties = _strict_sequence(
        _plan_value(plan, "uncertainties", ())
    )
    if raw_uncertainties is None:
        return _repair_fallback(
            repaired.census,
            affected_pages=tuple(sorted(affected_pages)),
            reason="repair_plan_uncertainties_invalid",
        )
    repair_datasets: set[str] = set()
    for uncertainty in raw_uncertainties:
        raw_logical_pages = _strict_sequence(
            _plan_value(uncertainty, "logical_pages", ())
        )
        if raw_logical_pages is None:
            return _repair_fallback(
                repaired.census,
                affected_pages=tuple(sorted(affected_pages)),
                reason="repair_uncertainty_pages_invalid",
            )
        logical_pages = {
            page
            for raw_page in raw_logical_pages
            for page in (_strict_positive_int(raw_page),)
            if page is not None
        }
        if len(logical_pages) != len(raw_logical_pages):
            return _repair_fallback(
                repaired.census,
                affected_pages=tuple(sorted(affected_pages)),
                reason="repair_uncertainty_pages_invalid",
            )
        if not logical_pages.intersection(affected_set):
            continue
        dataset_name = str(
            _plan_value(uncertainty, "dataset_name", "") or ""
        ).strip()
        if dataset_name and logical_pages.intersection(materialization_set):
            repair_datasets.add(dataset_name)

    if not materialization_pages:
        # Field-local deterministic/context-rich overlays consume the discovery
        # payload directly.  They must not invalidate any extraction stage.
        return CandidateBRepairScope(
            census=repaired.census,
            affected_pages=tuple(sorted(affected_pages)),
            expanded_pages=(),
            dirty_sections=(),
            dirty_stage_names=(),
        )

    return resolve_candidate_b_repair_scope(
        discovery,
        repaired,
        affected_pages=materialization_pages,
        repair_dataset_names=repair_datasets,
    )


__all__ = [
    "ACCOUNT_SECTION",
    "AGREEMENT_SECTION",
    "BLANK_SECTION",
    "CANDIDATE_B_STAGE_REGISTRY",
    "CANONICAL_DATASET_TO_SECTION",
    "CANONICAL_DATASET_TO_STAGE_NAMES",
    "DATASET_TO_SECTION",
    "CandidateBRepairScope",
    "CandidateBSectionCensus",
    "INQUIRY_SECTION",
    "LIABILITY_SECTION",
    "MIXED_SECTION_ENVELOPE",
    "POSTPAID_SECTION",
    "PUBLIC_SECTION",
    "PageOwnershipFingerprint",
    "REPORT_EXPLANATION_SECTION",
    "REPORT_HEADER_SECTION",
    "SECTION_TO_CANONICAL_DATASETS",
    "SUMMARY_SECTION",
    "build_candidate_b_section_census",
    "candidate_b_repair_scope",
    "plan_candidate_b_initial_extraction",
    "resolve_candidate_b_repair_scope",
    "section_census_from_canonical_audit",
    "stage_names_for_datasets",
]
