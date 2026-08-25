from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)


def _values(record: dict[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, dict) else record


def _field_ref(field_name: str) -> dict[str, Any]:
    return {
        "source": "native_detail_table_cell",
        "logical_page": 1,
        "source_page": 1,
        "table_id": "profile-phone-contract",
        "row": 1,
        "column": 1,
        "bbox": [10.0, 20.0, 110.0, 35.0],
        "coordinate_system": "pdf_points_top_left",
        "geometry_scope": "cell",
        "geometry_status": "exact",
        "evidence_ids": [f"profile-phone:{field_name}"],
        "field_name": field_name,
        "binding": "canonical_header_column",
        "binding_quality": "canonical_header_column",
    }


def _profile_phone_source(
    field_name: str,
    normalized_value: str,
    *,
    raw_value: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    raw_value = raw_value if raw_value is not None else normalized_value
    profile_id = "personal_profile:1"
    ref = _field_ref(field_name)
    return {
        "personal_profile": [
            {
                "record_id": profile_id,
                "normalized": {
                    "personal_profile_id": profile_id,
                    field_name: normalized_value,
                },
                "canonical_raw": {field_name: raw_value},
                "source_refs": [ref],
            }
        ],
        "personal_detail_field_observations": [
            {
                "record_id": f"personal_profile_field:{field_name}",
                "normalized": {
                    "field_observation_id": f"personal_profile_field:{field_name}",
                    "dataset_name": "personal_profile",
                    "business_record_id": profile_id,
                    "field_name": field_name,
                    "raw_value": raw_value,
                    "normalized_value": normalized_value,
                    "observation_status": "normalized",
                },
                "source_refs": [ref],
            }
        ],
    }


@pytest.mark.parametrize(
    (
        "profile_field",
        "phone_value",
        "source_dataset",
        "target_dataset",
        "id_field",
        "target_field",
        "record_id",
    ),
    [
        (
            "mobile_phone",
            "13800138007",
            "mobile_phone_records",
            "subject_mobile_phones",
            "mobile_phone_record_id",
            "mobile_phone",
            "personal_mobile_phone:existing",
        ),
        (
            "work_phone",
            "010-12345678",
            "employment_records",
            "subject_employment",
            "employment_record_id",
            "employer_phone",
            "credit_employment:existing",
        ),
        (
            "residence_phone",
            "010-87654321",
            "residence_records",
            "subject_residences",
            "residence_record_id",
            "residential_phone",
            "credit_residence:existing",
        ),
    ],
)
def test_profile_phone_routes_only_to_one_stable_existing_relation(
    profile_field: str,
    phone_value: str,
    source_dataset: str,
    target_dataset: str,
    id_field: str,
    target_field: str,
    record_id: str,
) -> None:
    source = _profile_phone_source(profile_field, phone_value)
    source[source_dataset] = [
        {
            "record_id": record_id,
            "normalized": {id_field: record_id, "sequence": 1},
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert profile_field not in _values(projected["subject_profile"][0])
    target = projected[target_dataset][0]
    assert target["record_id"] == record_id
    assert _values(target)[target_field] == phone_value
    observation = next(
        _values(row)
        for row in projected["field_observations"]
        if _values(row).get("field_name") == target_field
    )
    assert observation["dataset_name"] == target_dataset
    assert observation["business_record_id"] == record_id
    assert not any(
        _values(issue).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
        for issue in projected.get("extraction_issues", ())
    )


def test_standalone_profile_mobile_is_issue_only_without_printed_ordinal_grain() -> None:
    raw_value = "+86 138-0013-8007"
    source = _profile_phone_source(
        "mobile_phone",
        "13800138007",
        raw_value=raw_value,
    )

    projected = project_personal_detail_datasets(source)

    assert "subject_mobile_phones" not in projected
    assert "mobile_phone" not in _values(projected["subject_profile"][0])
    issue = next(
        row
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    values = _values(issue)
    assert values["target_dataset"] == "subject_mobile_phones"
    assert values["field_name"] == "mobile_phone"
    assert "target_record_id" not in values
    assert values["observed_value"] == raw_value
    assert values["candidate_value"] == "13800138007"
    assert issue["source_refs"] == [_field_ref("mobile_phone")]


@pytest.mark.parametrize(
    ("profile_field", "phone_value", "target_dataset", "target_field"),
    [
        (
            "work_phone",
            "010-12345678",
            "subject_employment",
            "employer_phone",
        ),
        (
            "residence_phone",
            "010-87654321",
            "subject_residences",
            "residential_phone",
        ),
    ],
)
def test_profile_work_and_residence_phones_never_invent_entities(
    profile_field: str,
    phone_value: str,
    target_dataset: str,
    target_field: str,
) -> None:
    projected = project_personal_detail_datasets(
        _profile_phone_source(profile_field, phone_value)
    )

    assert target_dataset not in projected
    issue = next(
        _values(row)
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    assert issue["target_dataset"] == target_dataset
    assert issue["field_name"] == target_field
    assert "target_record_id" not in issue


def test_profile_phone_conflict_does_not_overwrite_existing_relation() -> None:
    source = _profile_phone_source("work_phone", "010-12345678")
    target_id = "credit_employment:existing"
    source["employment_records"] = [
        {
            "record_id": target_id,
            "normalized": {
                "employment_record_id": target_id,
                "sequence": 1,
                "employer_phone": "010-99999999",
            },
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert (
        _values(projected["subject_employment"][0])["employer_phone"]
        == "010-99999999"
    )
    issue = next(
        _values(row)
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    assert issue["target_record_id"] == target_id
    assert issue["field_name"] == "employer_phone"


def test_profile_phone_does_not_choose_between_multiple_existing_entities() -> None:
    source = _profile_phone_source("residence_phone", "010-87654321")
    source["residence_records"] = [
        {
            "record_id": f"credit_residence:{sequence}",
            "normalized": {
                "residence_record_id": f"credit_residence:{sequence}",
                "sequence": sequence,
            },
        }
        for sequence in (1, 2)
    ]

    projected = project_personal_detail_datasets(source)

    assert all(
        "residential_phone" not in _values(row)
        for row in projected["subject_residences"]
    )
    issue = next(
        _values(row)
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    assert issue["target_dataset"] == "subject_residences"
    assert issue["field_name"] == "residential_phone"
    assert "target_record_id" not in issue


@pytest.mark.parametrize(
    "mutation",
    ["wrong_field", "derived", "missing_evidence", "wrong_coordinate"],
)
def test_profile_phone_rejects_nonexact_or_foreign_source_refs(
    mutation: str,
) -> None:
    source = _profile_phone_source("work_phone", "010-12345678")
    ref = source["personal_profile"][0]["source_refs"][0]
    if mutation == "wrong_field":
        ref["field_name"] = "birth_date"
        source["personal_profile"][0]["source_refs_by_field"] = {
            "work_phone": [ref]
        }
        source["personal_profile"][0].pop("source_refs", None)
    elif mutation == "derived":
        ref["geometry_status"] = "derived"
    elif mutation == "missing_evidence":
        ref["evidence_ids"] = []
    elif mutation == "wrong_coordinate":
        ref["coordinate_system"] = "image_pixels"
    source["personal_detail_field_observations"][0]["source_refs"] = [ref]
    source["employment_records"] = [
        {
            "record_id": "credit_employment:existing",
            "normalized": {
                "employment_record_id": "credit_employment:existing",
                "sequence": 1,
            },
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert "employer_phone" not in _values(projected["subject_employment"][0])
    issue_record = next(
        row
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    issue = _values(issue_record)
    assert issue["field_name"] == "employer_phone"
    assert issue_record.get("source_refs", []) == []


def test_matching_phone_from_two_profile_owners_is_not_collapsed() -> None:
    source = _profile_phone_source("work_phone", "010-12345678")
    second = deepcopy(source["personal_profile"][0])
    second["record_id"] = "personal_profile:2"
    second["normalized"]["personal_profile_id"] = "personal_profile:2"
    second["source_refs"][0]["evidence_ids"] = ["profile-phone:work_phone:2"]
    source["personal_profile"].append(second)
    source["employment_records"] = [
        {
            "record_id": "credit_employment:existing",
            "normalized": {
                "employment_record_id": "credit_employment:existing",
                "sequence": 1,
            },
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert "employer_phone" not in _values(projected["subject_employment"][0])
    assert any(
        _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
        for row in projected["extraction_issues"]
    )


def test_profile_phone_requires_agreeing_relation_identity_layers() -> None:
    source = _profile_phone_source("work_phone", "010-12345678")
    source["employment_records"] = [
        {
            "record_id": "credit_employment:outer",
            "normalized": {
                "employment_record_id": "credit_employment:inner",
                "sequence": 1,
            },
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert "employer_phone" not in _values(projected["subject_employment"][0])
    issue = next(
        _values(row)
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    assert "target_record_id" not in issue


@pytest.mark.parametrize(
    ("normalized_value", "raw_value"),
    [
        ("010--12345678", "010--12345678"),
        ("010-12345678", "010-99999999"),
    ],
)
def test_profile_phone_terminal_contract_rejects_malformed_or_conflicting_raw(
    normalized_value: str,
    raw_value: str,
) -> None:
    source = _profile_phone_source(
        "work_phone",
        normalized_value,
        raw_value=raw_value,
    )
    source["employment_records"] = [
        {
            "record_id": "credit_employment:existing",
            "normalized": {
                "employment_record_id": "credit_employment:existing",
                "sequence": 1,
            },
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert "employer_phone" not in _values(projected["subject_employment"][0])
    assert any(
        _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
        for row in projected["extraction_issues"]
    )


@pytest.mark.parametrize("raw_value", ["(010) 12345678", "0086 010-12345678"])
def test_profile_phone_raw_presentation_variants_share_one_semantic_number(
    raw_value: str,
) -> None:
    source = _profile_phone_source(
        "work_phone",
        "010-12345678",
        raw_value=raw_value,
    )
    source["employment_records"] = [
        {
            "record_id": "credit_employment:existing",
            "normalized": {
                "employment_record_id": "credit_employment:existing",
                "sequence": 1,
            },
        }
    ]

    projected = project_personal_detail_datasets(source)

    assert (
        _values(projected["subject_employment"][0])["employer_phone"]
        == "010-12345678"
    )


def test_orphan_profile_phone_control_cannot_retarget_by_relation_count_alone() -> None:
    source = {
        "employment_records": [
            {
                "record_id": "credit_employment:only",
                "normalized": {
                    "employment_record_id": "credit_employment:only",
                    "sequence": 1,
                    "employer_phone": "010-99999999",
                },
            }
        ],
        "personal_detail_field_observations": [
            {
                "record_id": "profile-phone:orphan",
                "normalized": {
                    "field_observation_id": "profile-phone:orphan",
                    "dataset_name": "personal_profile",
                    "business_record_id": "personal_profile:missing",
                    "field_name": "work_phone",
                    "raw_value": "010-12345678",
                    "normalized_value": "010-12345678",
                    "observation_status": "observed",
                },
                "source_refs": [
                    {
                        **_field_ref("birth_date"),
                        "source": "foreign_plane",
                    }
                ],
            }
        ],
    }

    projected = project_personal_detail_datasets(source)

    assert (
        _values(projected["subject_employment"][0])["employer_phone"]
        == "010-99999999"
    )
    assert not any(
        _values(observation).get("field_observation_id") == "profile-phone:orphan"
        for observation in projected.get("field_observations", [])
    )
    issue_record = next(
        row
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
    )
    issue = _values(issue_record)
    assert issue["target_dataset"] == "subject_employment"
    assert issue["field_name"] == "employer_phone"
    assert "target_record_id" not in issue
    assert issue["observed_value"] == "010-12345678"
    assert issue_record.get("source_refs", []) == []
    assert any(
        _values(evidence).get("string_value")
        == "profile_phone_source_ref_contract_failed"
        for evidence in projected.get("extraction_issue_evidence", [])
        if _values(evidence).get("extraction_issue_id")
        == issue["extraction_issue_id"]
    )


def test_orphan_profile_phone_control_retargets_only_with_exact_matching_authority() -> None:
    source = {
        "employment_records": [
            {
                "record_id": "credit_employment:only",
                "normalized": {
                    "employment_record_id": "credit_employment:only",
                    "sequence": 1,
                    "employer_phone": "010-12345678",
                },
            }
        ],
        "personal_detail_field_observations": [
            {
                "record_id": "profile-phone:orphan",
                "normalized": {
                    "field_observation_id": "profile-phone:orphan",
                    "dataset_name": "personal_profile",
                    "business_record_id": "personal_profile:missing",
                    "field_name": "work_phone",
                    "raw_value": "(010) 12345678",
                    "normalized_value": "010-12345678",
                    "observation_status": "normalized",
                },
                "source_refs": [_field_ref("work_phone")],
            }
        ],
    }

    projected = project_personal_detail_datasets(source)

    observation = next(
        _values(row)
        for row in projected["field_observations"]
        if _values(row).get("field_observation_id") == "profile-phone:orphan"
    )
    assert observation["dataset_name"] == "subject_employment"
    assert observation["business_record_id"] == "credit_employment:only"
    assert observation["field_name"] == "employer_phone"
    assert not any(
        _values(issue).get("issue_code")
        == "canonical_profile_phone_relation_unresolved"
        for issue in projected.get("extraction_issues", [])
    )


def _metadata(report_time: str) -> dict[str, Any]:
    metadata_id = f"metadata:{report_time}"
    return {
        "record_id": metadata_id,
        "normalized": {
            "personal_report_metadata_id": metadata_id,
            "report_number": "2025052510051518624525",
            "report_time": report_time,
            "subject_name": "张三",
            "primary_id_type": "护照",
            "primary_id_number": "AB123456",
            "query_institution": "中国人民银行征信中心",
            "query_reason": "本人查询",
        },
    }


def _birth_profile(birth_date: str, raw_value: str) -> dict[str, Any]:
    ref = _field_ref("birth_date")
    return {
        "record_id": "personal_profile:birth",
        "normalized": {
            "personal_profile_id": "personal_profile:birth",
            "birth_date": birth_date,
        },
        "canonical_raw": {"birth_date": raw_value},
        "source_refs": [ref],
    }


def test_future_birth_date_is_withheld_against_exact_report_date() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_report_metadata": [_metadata("2024-06-30T12:00:00+08:00")],
            "personal_profile": [_birth_profile("2099-12-31", "2099.12.31")],
        }
    )

    profile = projected["subject_profile"][0]
    assert _values(profile)["birth_date"] is None
    assert profile["canonical_raw"]["birth_date"] == "2099.12.31"
    issue = next(
        row
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_birth_date_after_report_date"
    )
    assert _values(issue)["target_dataset"] == "subject_profile"
    assert _values(issue)["target_record_id"] == "personal_profile:birth"
    assert _values(issue)["field_name"] == "birth_date"
    assert issue["source_refs"] == [_field_ref("birth_date")]


def test_future_birth_gate_never_targets_a_conflicting_profile_identity() -> None:
    profile = _birth_profile("2099-12-31", "2099.12.31")
    profile["record_id"] = "personal_profile:outer"
    profile["normalized"]["personal_profile_id"] = "personal_profile:inner"

    projected = project_personal_detail_datasets(
        {
            "personal_report_metadata": [_metadata("2024-06-30T12:00:00+08:00")],
            "personal_profile": [profile],
        }
    )

    values = _values(projected["subject_profile"][0])
    assert values["subject_profile_id"] is None
    assert values["birth_date"] is None
    birth_issue = next(
        _values(row)
        for row in projected["extraction_issues"]
        if _values(row).get("issue_code")
        == "canonical_birth_date_after_report_date"
    )
    assert birth_issue["target_record_id"] == "personal_profile:outer"


@pytest.mark.parametrize(
    ("metadata_rows", "birth_date"),
    [
        ([_metadata("2024-06-30T12:00:00+08:00")], "2024-06-30"),
        ([], "2099-12-31"),
        (
            [
                _metadata("2024-06-30T12:00:00+08:00"),
                _metadata("2025-06-30T12:00:00+08:00"),
            ],
            "2099-12-31",
        ),
    ],
)
def test_birth_date_semantic_gate_requires_one_exact_report_date(
    metadata_rows: list[dict[str, Any]],
    birth_date: str,
) -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_report_metadata": metadata_rows,
            "personal_profile": [_birth_profile(birth_date, birth_date)],
        }
    )

    assert _values(projected["subject_profile"][0])["birth_date"] == birth_date
    assert not any(
        _values(issue).get("issue_code")
        == "canonical_birth_date_after_report_date"
        for issue in projected.get("extraction_issues", ())
    )
