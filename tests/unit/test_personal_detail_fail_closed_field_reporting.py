# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trust-boundary tests for source-bound field omission reporting."""

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


def _metadata_issue(
    metadata_id: str,
    *,
    field_name: str = "subject_name",
    target_dataset: str = "personal_report_metadata",
    target_record_id: str | None = None,
    source_refs: list[dict[str, object]] | None = None,
    status: str = "requires_review",
) -> dict[str, object]:
    return make_issue(
        category="ocr_cell_level_error",
        issue_code="page_one_consensus_unresolved",
        message="withheld report-header field",
        status=status,
        parser_stage="page_one_consensus",
        target_dataset=target_dataset,
        target_record_id=target_record_id or metadata_id,
        field_name=field_name,
        observed_value=[],
        source_refs=(
            source_refs
            if source_refs is not None
            else [
                {
                    "source": "native_detail_table_cell",
                    "logical_page": 1,
                    "source_page": 1,
                    "table_id": "report-header",
                    "row": 1,
                    "column": 0,
                    "bbox": [20.0, 40.0, 120.0, 60.0],
                    "geometry_scope": "cell",
                    "evidence_ids": ["native:report-header:1:0"],
                    "binding": "canonical_header_column",
                    "binding_quality": "canonical_header_column",
                    "field_name": field_name,
                    "canonical_template_id": "report_header_and_identity",
                }
            ]
        ),
        reason_codes=("page_one_consensus", "normalized_value_withheld"),
    )


def _canonical_layout_audit(
    *owners: tuple[str, tuple[tuple[int, int], ...]],
) -> dict[str, object]:
    if not owners:
        owners = (("report_header_and_identity", ((1, 1),)),)
    return {
        "architecture": "canonical_template_registration_v3_static",
        "registrations": [
            {
                "logical_page": logical_page,
                "source_page": source_page,
                "template_id": template_id,
                "status": "registered",
            }
            for template_id, fragments in owners
            for logical_page, source_page in fragments
        ],
        "fragment_groups": [
            {
                "canonical_page": index,
                "template_id": template_id,
                "fragment_logical_pages": [logical for logical, _source in fragments],
                "source_pages": sorted({source for _logical, source in fragments}),
                "coverage_status": "full",
            }
            for index, (template_id, fragments) in enumerate(owners, start=1)
        ],
    }


def _corrected_metadata_ref(
    *,
    field_name: str = "subject_name",
    logical_page: int = 17,
    source_page: int = 9,
    bbox: tuple[float, float, float, float] = (20.0, 40.0, 120.0, 60.0),
    evidence_ids: tuple[str, ...] = ("corrected:header:subject-name",),
) -> dict[str, object]:
    return {
        "source": "personal_detail_corrected_page_cell",
        "logical_page": logical_page,
        "source_page": source_page,
        "bbox": list(bbox),
        "geometry_scope": "cell",
        "evidence_ids": list(evidence_ids),
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_cell_slot",
        "field_name": field_name,
        "canonical_template_id": "report_header_and_identity",
    }


def _base_profile_content(
    issue: dict[str, object],
    *,
    metadata_id: str | None = None,
    profile_name: str | None = None,
    canonical_layout_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata_id = metadata_id or str(issue.get("target_record_id") or "metadata:1")
    return {
        "facts": {
            "personal_detail_canonical_layout_audit": (
                canonical_layout_audit
                if canonical_layout_audit is not None
                else _canonical_layout_audit()
            ),
            **(
                {"subject_name": profile_name}
                if profile_name is not None
                else {}
            ),
        },
        "datasets": {
            "personal_report_metadata": [
                {
                    "record_id": metadata_id,
                    "personal_report_metadata_id": metadata_id,
                    "subject_name": None,
                    "primary_id_type": None,
                    "primary_id_number": None,
                }
            ],
            "personal_profile": [
                {
                    "record_id": "personal_profile:1",
                    "personal_profile_id": "personal_profile:1",
                    **(
                        {"subject_name": profile_name}
                        if profile_name is not None
                        else {}
                    ),
                }
            ],
            "personal_detail_extraction_issues": [issue],
        },
    }


def _production_timing_profile_content(
    issue: dict[str, object], audit: dict[str, object]
) -> dict[str, object]:
    content = _base_profile_content(issue)
    facts = content["facts"]
    assert isinstance(facts, dict)
    facts.pop("personal_detail_canonical_layout_audit")
    facts["_personal_detail_canonical_layout_owner_census"] = audit
    return content


def _exact_mobile_ref(
    *,
    field_name: str,
    row: int = 2,
    column: int = 1,
    logical_page: int = 1,
    evidence_ids: tuple[str, ...] | None = None,
    sequence: int = 1,
) -> dict[str, object]:
    return {
        "source": "candidate_b_raw_profile_field_cell",
        "logical_page": logical_page,
        "source_page": 1,
        "table_id": "mobile-table",
        "row": row,
        "column": column,
        "bbox": [20.0, 40.0, 120.0, 60.0],
        "geometry_scope": "cell",
        "evidence_ids": list(
            evidence_ids
            if evidence_ids is not None
            else (f"raw:mobile:{sequence}:{row}:{column}:{field_name}",)
        ),
        "binding": "printed_profile_field",
        "binding_quality": "printed_profile_field",
        "canonical_template_id": "report_header_and_identity",
        "dataset_name": "mobile_phone_records",
        "component": "mobile",
        "sequence": sequence,
        "field_name": field_name,
    }


def _exact_mobile_sequence_ref(sequence: int) -> dict[str, object]:
    return {
        "source": "candidate_b_raw_profile_sequence_cell",
        "logical_page": 1,
        "source_page": 1,
        "table_id": "mobile-table",
        "row": sequence + 1,
        "column": 0,
        "bbox": [5.0, 40.0 + sequence * 20.0, 15.0, 60.0 + sequence * 20.0],
        "geometry_scope": "cell",
        "evidence_ids": [f"raw:mobile:sequence:{sequence}"],
        "binding": "printed_profile_sequence",
        "binding_quality": "printed_profile_sequence",
        "canonical_template_id": "report_header_and_identity",
        "dataset_name": "mobile_phone_records",
        "component": "mobile",
        "sequence": sequence,
        "field_name": "sequence",
    }


def _mobile_content(
    *,
    observations: dict[str, object],
    endpoint: int = 1,
    emitted: list[dict[str, object]] | None = None,
    existing_issues: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sealed_observations = deepcopy(observations)
    observed_sequences: list[int] = []
    for raw_observation in sealed_observations.values():
        if not isinstance(raw_observation, dict):
            continue
        sequence = raw_observation.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
            continue
        observed_sequences.append(sequence)
        raw_observation.setdefault(
            "mobile_phone_record_id",
            stable_record_id("personal_mobile_phone", sequence),
        )
        raw_observation.setdefault(
            "canonical_template_id", "report_header_and_identity"
        )
        raw_observation.setdefault(
            "canonical_header_fields_by_component",
            {
                "mobile": [
                    "sequence",
                    "mobile_phone",
                    "information_updated_date",
                    "data_provider",
                ]
            },
        )
        raw_observation.setdefault(
            "source_refs", [_exact_mobile_sequence_ref(sequence)]
        )
    return {
        "facts": {
            "personal_detail_source_completeness_ledger": {
                "sequence_endpoints": {"mobile_phone_records": endpoint},
                "sequence_observed_sequences": {
                    "mobile_phone_records": sorted(observed_sequences)
                },
                "sequence_ordinal_observations": {
                    "mobile_phone_records": sealed_observations
                },
            }
        },
        "datasets": {
            "mobile_phone_records": list(emitted or []),
            "personal_detail_extraction_issues": list(existing_issues or []),
        },
    }


def _issues(content: dict[str, object], code: str) -> list[dict[str, object]]:
    datasets = content["datasets"]
    assert isinstance(datasets, dict)
    return [
        row
        for row in datasets.get("personal_detail_extraction_issues", [])
        if isinstance(row, dict) and row.get("issue_code") == code
    ]


def test_profile_metadata_failure_is_mirrored_after_profile_materialization() -> None:
    metadata_id = "personal_report_metadata:exact"
    content = _base_profile_content(_metadata_issue(metadata_id))

    prepare_personal_detail_source_collections(content)
    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert {
        "target_dataset": mirrors[0]["target_dataset"],
        "target_record_id": mirrors[0]["target_record_id"],
        "field_name": mirrors[0]["field_name"],
    } == {
        "target_dataset": "personal_profile",
        "target_record_id": "personal_profile:1",
        "field_name": "subject_name",
    }
    assert mirrors[0]["source_refs"] == [
        {
            "source": "native_detail_table_cell",
            "logical_page": 1,
            "source_page": 1,
            "table_id": "report-header",
            "row": 1,
            "column": 0,
            "bbox": [20.0, 40.0, 120.0, 60.0],
            "geometry_scope": "cell",
            "evidence_ids": ["native:report-header:1:0"],
            "binding": "canonical_header_column",
            "binding_quality": "canonical_header_column",
            "field_name": "subject_name",
            "canonical_template_id": "report_header_and_identity",
        }
    ]
    observations = content["datasets"]["personal_detail_field_observations"]
    assert sum(
        row.get("dataset_name") == "personal_profile"
        and row.get("business_record_id") == "personal_profile:1"
        and row.get("field_name") == "subject_name"
        for row in observations
    ) == 1


def test_corrected_header_cell_failure_is_mirrored_on_arbitrary_owned_page() -> None:
    metadata_id = "personal_report_metadata:corrected-exact"
    ref = _corrected_metadata_ref()
    content = _base_profile_content(
        _metadata_issue(metadata_id, source_refs=[ref]),
        canonical_layout_audit=_canonical_layout_audit(
            ("information_summary", ((1, 1),)),
            ("report_header_and_identity", ((17, 9),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_owner",
        "wrong_source",
        "wrong_binding",
        "replayed_evidence",
        "multiple_slots",
    ),
)
def test_corrected_header_mirror_rejects_nonunique_or_foreign_evidence(
    mutation: str,
) -> None:
    metadata_id = f"personal_report_metadata:corrected-{mutation}"
    first = _corrected_metadata_ref()
    refs = [first]
    audit = _canonical_layout_audit(
        ("report_header_and_identity", ((17, 9),)),
    )
    if mutation == "wrong_owner":
        audit = _canonical_layout_audit(
            ("information_summary", ((17, 9),)),
        )
    elif mutation == "wrong_source":
        first["source"] = "personal_detail_corrected_page_rows"
    elif mutation == "wrong_binding":
        first["binding"] = "canonical_label_slot"
    elif mutation == "replayed_evidence":
        second = deepcopy(first)
        second["bbox"] = [140.0, 40.0, 240.0, 60.0]
        refs.append(second)
    elif mutation == "multiple_slots":
        second = _corrected_metadata_ref(
            bbox=(140.0, 40.0, 240.0, 60.0),
            evidence_ids=("corrected:header:subject-name:second",),
        )
        refs.append(second)
    content = _base_profile_content(
        _metadata_issue(metadata_id, source_refs=refs),
        canonical_layout_audit=audit,
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_source_dataset",
        "wrong_source_record",
        "noncanonical_field",
        "resolved",
        "broad_ref",
        "outside_header_owner",
        "profile_value_present",
    ),
)
def test_profile_mirror_rejects_non_exact_or_nonmissing_sources(mutation: str) -> None:
    metadata_id = "personal_report_metadata:exact"
    kwargs: dict[str, object] = {}
    profile_name = None
    if mutation == "wrong_source_dataset":
        kwargs["target_dataset"] = "report_query"
    elif mutation == "wrong_source_record":
        kwargs["target_record_id"] = "personal_report_metadata:other"
    elif mutation == "noncanonical_field":
        kwargs["field_name"] = "query_reason"
    elif mutation == "resolved":
        kwargs["status"] = "resolved"
    elif mutation == "broad_ref":
        kwargs["source_refs"] = [
            {
                "source": "candidate_b_source_coverage_ledger",
                "logical_page": 1,
                "source_page": 1,
                "geometry_scope": "logical_page",
            }
        ]
    elif mutation == "outside_header_owner":
        ref = _exact_mobile_ref(field_name="subject_name", logical_page=2)
        ref["table_id"] = "report-header"
        ref["canonical_template_id"] = "report_header_and_identity"
        kwargs["source_refs"] = [ref]
    elif mutation == "profile_value_present":
        profile_name = "Example Subject"
    issue = _metadata_issue(metadata_id, **kwargs)
    content = _base_profile_content(
        issue,
        metadata_id=metadata_id,
        profile_name=profile_name,
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_accepts_exact_header_owner_on_page_three() -> None:
    metadata_id = "personal_report_metadata:page-three"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["logical_page"] = 3
    ref["source_page"] = 3
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(
        issue,
        canonical_layout_audit=_canonical_layout_audit(
            ("information_summary", ((1, 1),)),
            ("report_header_and_identity", ((3, 3),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


def test_profile_mirror_does_not_require_cell_to_self_declare_template() -> None:
    metadata_id = "personal_report_metadata:audit-owned-template"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref.pop("canonical_template_id")
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(issue)

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


def test_profile_mirror_rejects_exact_page_one_non_header_owner() -> None:
    metadata_id = "personal_report_metadata:non-header"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["canonical_template_id"] = "information_summary"
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(
        issue,
        canonical_layout_audit=_canonical_layout_audit(
            ("information_summary", ((1, 1),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


@pytest.mark.parametrize(
    ("coordinate", "value"),
    (("logical_page", None), ("logical_page", 0), ("source_page", None), ("source_page", 0)),
)
def test_profile_mirror_rejects_invalid_owner_coordinates_without_raising(
    coordinate: str,
    value: object,
) -> None:
    metadata_id = f"personal_report_metadata:invalid-{coordinate}-{value}"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref[coordinate] = value
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(issue)

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_rejects_cell_template_claim_conflicting_with_owner_census() -> None:
    metadata_id = "personal_report_metadata:conflicting-cell-claim"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["canonical_template_id"] = "information_summary"
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(issue)

    prepare_personal_detail_source_collections(content)

    # A self-declared template never grants authority, but an explicit conflict
    # with the complete canonical owner census must still veto the mirror.
    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_accepts_one_field_owner_among_multiple_profile_groups() -> None:
    metadata_id = "personal_report_metadata:duplicate-owners"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["logical_page"] = 3
    ref["source_page"] = 2
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(
        issue,
        canonical_layout_audit=_canonical_layout_audit(
            ("report_header_and_identity", ((3, 2),)),
            ("report_header_and_identity", ((7, 4),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


def test_profile_mirror_rejects_duplicate_same_field_owners() -> None:
    metadata_id = "personal_report_metadata:duplicate-field-owners"
    first = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(first, dict)
    first["logical_page"] = 3
    first["source_page"] = 2
    second = deepcopy(first)
    second["logical_page"] = 7
    second["source_page"] = 4
    second["table_id"] = "second-report-header"
    second["evidence_ids"] = ["native:second-report-header:1:0"]
    issue = _metadata_issue(metadata_id, source_refs=[first, second])
    content = _base_profile_content(
        issue,
        canonical_layout_audit=_canonical_layout_audit(
            ("report_header_and_identity", ((3, 2),)),
            ("report_header_and_identity", ((7, 4),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_production_timing_owner_census_mirrors_page_three_with_multiple_groups() -> None:
    metadata_id = "personal_report_metadata:production-page-three"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["logical_page"] = 3
    ref["source_page"] = 3
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _production_timing_profile_content(
        issue,
        _canonical_layout_audit(
            ("report_header_and_identity", ((1, 1),)),
            ("report_header_and_identity", ((3, 3),)),
            ("information_summary", ((4, 4),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


def test_production_timing_owner_census_mirrors_two_up_profile_fragment() -> None:
    metadata_id = "personal_report_metadata:production-two-up"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["logical_page"] = 8
    ref["source_page"] = 4
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _production_timing_profile_content(
        issue,
        _canonical_layout_audit(
            ("report_header_and_identity", ((1, 1),)),
            ("report_header_and_identity", ((8, 4),)),
            ("information_summary", ((9, 4),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


def test_production_timing_owner_census_vetoes_duplicate_same_field_owner() -> None:
    metadata_id = "personal_report_metadata:production-duplicate-owner"
    first = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(first, dict)
    first["logical_page"] = 3
    first["source_page"] = 3
    second = deepcopy(first)
    second["logical_page"] = 8
    second["source_page"] = 4
    second["table_id"] = "second-report-header"
    second["evidence_ids"] = ["native:second-report-header:1:0"]
    issue = _metadata_issue(metadata_id, source_refs=[first, second])
    content = _production_timing_profile_content(
        issue,
        _canonical_layout_audit(
            ("report_header_and_identity", ((3, 3),)),
            ("report_header_and_identity", ((8, 4),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_accepts_two_up_fragment_with_distinct_page_identities() -> None:
    metadata_id = "personal_report_metadata:two-up"
    ref = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(ref, dict)
    ref["logical_page"] = 8
    ref["source_page"] = 4
    issue = _metadata_issue(metadata_id, source_refs=[ref])
    content = _base_profile_content(
        issue,
        canonical_layout_audit=_canonical_layout_audit(
            ("report_header_and_identity", ((8, 4),)),
            ("information_summary", ((9, 4),)),
        ),
    )

    prepare_personal_detail_source_collections(content)

    mirrors = _issues(content, "source_bound_profile_field_omitted")
    assert len(mirrors) == 1
    assert mirrors[0]["source_refs"] == [ref]


def test_profile_mirror_rejects_multiple_field_slots_on_one_owner() -> None:
    metadata_id = "personal_report_metadata:multiple-slots"
    first = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(first, dict)
    second = deepcopy(first)
    second["column"] = 1
    second["evidence_ids"] = ["native:report-header:1:1"]
    issue = _metadata_issue(metadata_id, source_refs=[first, second])
    content = _base_profile_content(issue)

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_accepts_owner_census_on_metadata_record() -> None:
    metadata_id = "personal_report_metadata:metadata-owned-census"
    content = _base_profile_content(_metadata_issue(metadata_id))
    facts = content["facts"]
    datasets = content["datasets"]
    assert isinstance(facts, dict)
    assert isinstance(datasets, dict)
    audit = facts.pop("personal_detail_canonical_layout_audit")
    metadata = datasets["personal_report_metadata"][0]
    assert isinstance(metadata, dict)
    metadata["canonical_layout"] = audit

    prepare_personal_detail_source_collections(content)

    assert len(_issues(content, "source_bound_profile_field_omitted")) == 1


@pytest.mark.parametrize("mutation", ("missing", "incomplete", "conflicting"))
def test_profile_mirror_requires_one_complete_owner_census(mutation: str) -> None:
    metadata_id = f"personal_report_metadata:{mutation}-census"
    content = _base_profile_content(_metadata_issue(metadata_id))
    facts = content["facts"]
    assert isinstance(facts, dict)
    if mutation == "missing":
        facts.pop("personal_detail_canonical_layout_audit")
    elif mutation == "incomplete":
        facts["canonical_layout"] = {"registrations": []}
        facts.pop("personal_detail_canonical_layout_audit")
    else:
        facts["canonical_layout"] = _canonical_layout_audit(
            ("report_header_and_identity", ((3, 3),)),
        )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_rejects_same_table_slot_on_distinct_owner_fragments() -> None:
    metadata_id = "personal_report_metadata:repeated-slot"
    first = deepcopy(_metadata_issue(metadata_id)["source_refs"][0])
    assert isinstance(first, dict)
    first["logical_page"] = 8
    first["source_page"] = 4
    second = deepcopy(first)
    second["logical_page"] = 9
    second["evidence_ids"] = ["native:report-header:second-fragment"]
    issue = _metadata_issue(metadata_id, source_refs=[first, second])
    content = _base_profile_content(
        issue,
        canonical_layout_audit=_canonical_layout_audit(
            ("report_header_and_identity", ((8, 4), (9, 4))),
        ),
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_profile_mirror_requires_one_unambiguous_metadata_and_profile_identity() -> None:
    metadata_id = "personal_report_metadata:exact"
    content = _base_profile_content(_metadata_issue(metadata_id))
    datasets = content["datasets"]
    assert isinstance(datasets, dict)
    datasets["personal_report_metadata"].append(
        {
            "record_id": "personal_report_metadata:second",
            "personal_report_metadata_id": "personal_report_metadata:second",
        }
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_bound_profile_field_omitted")


def test_mobile_exact_cell_omissions_are_stable_idempotent_and_public() -> None:
    target_id = stable_record_id("personal_mobile_phone", 1)
    observation = {
        "sequence": 1,
        "canonical_header_fields": sorted(
            {
                "sequence",
                "mobile_phone",
                "information_updated_date",
                "data_provider",
            }
        ),
        "printed_fields": [
            "mobile_phone",
            "information_updated_date",
            "data_provider",
        ],
        "field_source_refs": {
            "mobile_phone": [
                _exact_mobile_ref(field_name="mobile_phone", column=1)
            ],
            "information_updated_date": [
                _exact_mobile_ref(
                    field_name="information_updated_date", column=2
                )
            ],
            "data_provider": [
                _exact_mobile_ref(field_name="data_provider", column=3)
            ],
        },
    }
    content = _mobile_content(observations={"1": observation})

    prepare_personal_detail_source_collections(content)
    prepare_personal_detail_source_collections(content)

    source_issues = _issues(content, "source_mobile_field_omitted")
    assert len(source_issues) == 3
    assert {
        (row["target_record_id"], row["field_name"]) for row in source_issues
    } == {
        (target_id, "mobile_phone"),
        (target_id, "information_updated_date"),
        (target_id, "data_provider"),
    }
    assert all(
        ref.get("geometry_scope") == "cell"
        and ref.get("field_name") == row["field_name"]
        and ref.get("evidence_ids")
        for row in source_issues
        for ref in row["source_refs"]
    )

    projected = project_personal_detail_datasets(content["datasets"])
    public = [
        row.get("normalized", row)
        for row in projected["extraction_issues"]
        if row.get("normalized", row).get("issue_code")
        == "source_mobile_field_omitted"
    ]
    assert {
        (row["target_dataset"], row["target_record_id"], row["field_name"])
        for row in public
    } == {
        ("subject_mobile_phones", target_id, "mobile_phone"),
        ("subject_mobile_phones", target_id, "information_updated_date"),
        ("subject_mobile_phones", target_id, "data_provider"),
    }
    public_ids = {row["extraction_issue_id"] for row in public}
    evidence = [
        row
        for row in projected["extraction_issue_evidence"]
        if row["extraction_issue_id"] in public_ids
    ]
    assert evidence
    assert public_ids == {row["extraction_issue_id"] for row in evidence}
    assert any(
        row["evidence_kind"] == "observed"
        and row["evidence_path"] == "source_field_observed"
        and row.get("boolean_value") is True
        for row in evidence
    )
    assert any(
        row["evidence_kind"] == "reason"
        and row.get("string_value") == "exact_source_cell"
        for row in evidence
    )


def test_mobile_reports_only_missing_printed_fields_on_exact_emitted_identity() -> None:
    target_id = stable_record_id("personal_mobile_phone", 1)
    content = _mobile_content(
        observations={
            "1": {
                "sequence": 1,
                "canonical_header_fields": sorted(
                    {
                        "sequence",
                        "mobile_phone",
                        "information_updated_date",
                        "data_provider",
                    }
                ),
                "printed_fields": ["mobile_phone", "data_provider"],
                "field_source_refs": {
                    "mobile_phone": [
                        _exact_mobile_ref(field_name="mobile_phone", column=1)
                    ],
                    "data_provider": [
                        _exact_mobile_ref(field_name="data_provider", column=3)
                    ],
                },
            }
        },
        emitted=[
            {
                "record_id": target_id,
                "mobile_phone_record_id": target_id,
                "sequence": 1,
                "mobile_phone": "13800138000",
            }
        ],
    )

    prepare_personal_detail_source_collections(content)

    issues = _issues(content, "source_mobile_field_omitted")
    assert [(row["target_record_id"], row["field_name"]) for row in issues] == [
        (target_id, "data_provider")
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "endpoint_only",
        "missing_ordinal_observation",
        "partial_header",
        "unknown_printed_field",
        "table_scope",
        "wrong_ref_field",
        "missing_evidence",
        "derived_slot",
        "duplicate_emitted_ordinal",
        "wrong_emitted_identity",
    ),
)
def test_mobile_does_not_promote_broad_ambiguous_or_noncanonical_evidence(
    mutation: str,
) -> None:
    target_id = stable_record_id("personal_mobile_phone", 1)
    observation: dict[str, object] = {
        "sequence": 1,
        "canonical_header_fields": sorted(
            {
                "sequence",
                "mobile_phone",
                "information_updated_date",
                "data_provider",
            }
        ),
        "printed_fields": ["mobile_phone"],
        "field_source_refs": {
            "mobile_phone": [_exact_mobile_ref(field_name="mobile_phone")]
        },
    }
    observations: dict[str, object] = {"1": observation}
    emitted: list[dict[str, object]] = []
    if mutation == "endpoint_only":
        observations = {}
    elif mutation == "missing_ordinal_observation":
        observations = {
            "2": {
                **deepcopy(observation),
                "sequence": 2,
            }
        }
    elif mutation == "partial_header":
        observation["canonical_header_fields_by_component"] = {
            "mobile": ["sequence", "mobile_phone"]
        }
    elif mutation == "unknown_printed_field":
        observation["printed_fields"] = ["raw_mobile_blob"]
        observation["field_source_refs"] = {
            "raw_mobile_blob": [
                _exact_mobile_ref(field_name="raw_mobile_blob")
            ]
        }
    elif mutation == "table_scope":
        ref = _exact_mobile_ref(field_name="mobile_phone")
        ref["geometry_scope"] = "table"
        observation["field_source_refs"] = {"mobile_phone": [ref]}
    elif mutation == "wrong_ref_field":
        observation["field_source_refs"] = {
            "mobile_phone": [_exact_mobile_ref(field_name="data_provider")]
        }
    elif mutation == "missing_evidence":
        observation["field_source_refs"] = {
            "mobile_phone": [
                _exact_mobile_ref(field_name="mobile_phone", evidence_ids=())
            ]
        }
    elif mutation == "derived_slot":
        ref = _exact_mobile_ref(field_name="mobile_phone")
        ref["source"] = "native_detail_table"
        ref["geometry_scope"] = "canonical_field_slot"
        ref.pop("bbox")
        observation["field_source_refs"] = {"mobile_phone": [ref]}
    elif mutation == "duplicate_emitted_ordinal":
        emitted = [
            {
                "record_id": target_id,
                "mobile_phone_record_id": target_id,
                "sequence": 1,
            },
            {
                "record_id": target_id,
                "mobile_phone_record_id": target_id,
                "sequence": 1,
            },
        ]
    elif mutation == "wrong_emitted_identity":
        emitted = [
            {
                "record_id": "mobile:guessed",
                "mobile_phone_record_id": "mobile:guessed",
                "sequence": 1,
            }
        ]
    content = _mobile_content(
        observations=observations,
        endpoint=1,
        emitted=emitted,
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_mobile_field_omitted")


def test_mobile_existing_active_field_issue_prevents_duplicate_reporting() -> None:
    target_id = stable_record_id("personal_mobile_phone", 1)
    existing = make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_mobile_row_unresolved",
        message="mobile withheld",
        target_dataset="mobile_phone_records",
        target_record_id=target_id,
        field_name="mobile_phone",
        source_refs=[_exact_mobile_ref(field_name="mobile_phone")],
        reason_codes=("normalized_value_withheld",),
    )
    content = _mobile_content(
        observations={
            "1": {
                "sequence": 1,
                "canonical_header_fields": sorted(
                    {
                        "sequence",
                        "mobile_phone",
                        "information_updated_date",
                        "data_provider",
                    }
                ),
                "printed_fields": ["mobile_phone"],
                "field_source_refs": {
                    "mobile_phone": [
                        _exact_mobile_ref(field_name="mobile_phone")
                    ]
                },
            }
        },
        existing_issues=[existing],
    )

    prepare_personal_detail_source_collections(content)

    assert not _issues(content, "source_mobile_field_omitted")
    matching_field_issues = [
        row
        for row in content["datasets"]["personal_detail_extraction_issues"]
        if row.get("target_dataset") == "mobile_phone_records"
        and row.get("target_record_id") == target_id
        and row.get("field_name") == "mobile_phone"
    ]
    assert len(matching_field_issues) == 1
    assert matching_field_issues[0]["issue_code"] == "candidate_b_mobile_row_unresolved"
    # Field-local coverage does not assert that the row itself was emitted.
    # The independent ordinal therefore still receives its own row-local gap.
    assert len(_issues(content, "source_mobile_record_omitted")) == 1


def test_new_field_issue_names_belong_to_the_closed_canonical_dictionary() -> None:
    dictionary = personal_detail_data_dictionary()["datasets"]
    source_to_canonical = {
        "personal_profile": "subject_profile",
        "mobile_phone_records": "subject_mobile_phones",
    }
    cases = [
        ("personal_profile", "subject_name"),
        ("personal_profile", "primary_id_type"),
        ("personal_profile", "primary_id_number"),
        ("mobile_phone_records", "mobile_phone"),
        ("mobile_phone_records", "information_updated_date"),
        ("mobile_phone_records", "data_provider"),
    ]

    assert all(
        field_name
        in dictionary[source_to_canonical[source_dataset]]["columns"]
        for source_dataset, field_name in cases
    )
