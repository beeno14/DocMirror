# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.community_plugin import (
    _compact_personal_detail_public_projection,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


def _ref(kind: str, *, field_name: str, ordinal: int = 3) -> dict[str, object]:
    common: dict[str, object] = {
        "logical_page": 12,
        "source_page": 6,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "evidence_ids": ["ocr:exact:3"],
    }
    if kind == "account":
        return {**common, "source": "candidate_b_account_anchor"}
    if kind == "agreement":
        return {
            **common,
            "source": "candidate_b_source_coverage_ledger",
            "geometry_scope": "line",
            "binding": "printed_credit_agreement_ordinal",
            "binding_quality": "printed_credit_agreement_ordinal",
            "sequence": ordinal,
        }
    if kind == "inquiry":
        return {
            **common,
            "source": "native_detail_inquiry_token_ordinal",
            "geometry_scope": "token",
            "binding": "printed_inquiry_ordinal_token",
            "binding_quality": "exact_token_in_sequence_cell",
            "sequence": ordinal,
            "table_id": "pt_12_0",
            "row": 3,
            "column": 0,
        }
    return {
        **common,
        "source": "native_detail_table_cell",
        "geometry_scope": "cell",
        "binding": "monthly_grid_cell",
        "binding_quality": "monthly_grid_cell",
        "field_name": field_name,
        "grid_id": "grid-7",
        "performance_month": "2024-03",
        "table_id": "monthly-7",
        "row": 2,
        "column": 3,
    }


def _evidence(
    issue_id: str,
    path: str,
    value: object,
    *,
    kind: str = "observed",
    page: int = 12,
) -> dict[str, object]:
    if type(value) is bool:
        value_type, value_key = "boolean", "boolean_value"
    elif type(value) is int:
        value_type, value_key = "integer", "integer_value"
    else:
        value_type, value_key = "string", "string_value"
    return {
        "record_id": f"evidence:{issue_id}:{kind}:{path}",
        "normalized": {
            "extraction_issue_id": issue_id,
            "evidence_kind": kind,
            "evidence_path": path,
            "value_type": value_type,
            value_key: value,
        },
        "source": {"page_range": [page, page]},
    }


def _case(kind: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    issue_id = f"issue:{kind}"
    if kind == "account":
        values = {
            "extraction_issue_id": issue_id,
            "issue_code": "source_account_record_omitted",
            "status": "requires_review",
            "target_dataset": "credit_accounts",
            "target_record_id": "credit_account:credit_card:3",
            "field_name": "account_id",
        }
        evidence = [
            _evidence(issue_id, "account_type", "credit_card"),
            _evidence(issue_id, "category_sequence", 3),
        ]
    elif kind == "agreement":
        values = {
            "extraction_issue_id": issue_id,
            "issue_code": "source_credit_agreement_record_omitted",
            "status": "requires_review",
            "target_dataset": "credit_agreements",
            "target_record_id": "credit_agreement:3",
            "field_name": "credit_agreement_id",
        }
        evidence = [_evidence(issue_id, "credit_agreement_sequence", 3)]
    elif kind == "inquiry":
        values = {
            "extraction_issue_id": issue_id,
            "issue_code": "source_inquiry_record_omitted",
            "status": "requires_review",
            "target_dataset": "inquiries",
            "target_record_id": "credit_inquiry:institution:3",
            "field_name": "inquiry_id",
        }
        evidence = [
            _evidence(issue_id, "inquiry_type", "institution"),
            _evidence(issue_id, "sequence", 3),
        ]
    else:
        values = {
            "extraction_issue_id": issue_id,
            "issue_code": "candidate_b_monthly_grid_contract_missing_field",
            "status": "requires_review",
            "target_dataset": "credit_account_monthly_performance",
            "target_record_id": "grid-7:2024-03",
            "field_name": "status_code",
        }
        evidence = [
            _evidence(issue_id, "grid_id", "grid-7"),
            _evidence(issue_id, "performance_month", "2024-03"),
        ]
    source = {
        "record_id": issue_id,
        "normalized": dict(values),
        "source_refs": [_ref(kind, field_name=str(values["field_name"]))],
    }
    return source, evidence


def _range_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    issue_id = "issue:source-account-month-range"
    account_id = "credit_account:credit_card:3"
    month = "2024-03"
    owner_hash = stable_record_id(
        "source_account_month_owner", account_id
    ).split(":", 1)[-1]
    values: dict[str, object] = {
        "extraction_issue_id": issue_id,
        "issue_code": "candidate_b_monthly_account_range_missing_month",
        "status": "requires_review",
        "target_dataset": "credit_account_monthly_performance",
        "target_record_id": f"source_account_month:{owner_hash}:{month}",
        "field_name": "performance_month",
    }
    source: dict[str, object] = {
        "record_id": issue_id,
        "normalized": values,
        "source_refs": [
            {
                "source": "candidate_b_monthly_anchor_ledger",
                "logical_page": 12,
                "source_page": 6,
                "bbox": [10.0, 20.0, 300.0, 40.0],
                "geometry_scope": "line",
                "binding": "source_account_month_range",
                "binding_quality": "source_account_month_range",
                "field_name": "performance_month",
                "account_id": account_id,
                "performance_month": month,
                "evidence_ids": ["ocr:exact:monthly-range:3"],
            }
        ],
    }
    return source, [
        _evidence(issue_id, "account_id", account_id),
        _evidence(issue_id, "performance_month", month),
    ]


def _owned_grid_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    issue_id = "issue:source-account-month-owned-grid"
    account_id = "credit_account:revolving_loan_account:4"
    month = "2023-11"
    owner_hash = stable_record_id(
        "source_account_month_owner", account_id
    ).split(":", 1)[-1]
    values: dict[str, object] = {
        "extraction_issue_id": issue_id,
        "issue_code": "candidate_b_monthly_owned_grid_missing_field",
        "status": "requires_review",
        "target_dataset": "credit_account_monthly_performance",
        "target_record_id": f"source_account_month:{owner_hash}:{month}",
        "field_name": "performance_month",
    }
    source: dict[str, object] = {
        "record_id": issue_id,
        "normalized": values,
        "source_refs": [
            {
                "source": "candidate_b_monthly_owned_grid_cell",
                "source_origin": "sealed_native_physical_table_cell",
                "logical_page": 12,
                "source_page": 6,
                "bbox": [42.0, 120.0, 58.0, 132.0],
                "geometry_scope": "cell",
                "binding": "source_account_month_identity",
                "binding_quality": "source_account_month_identity",
                "field_name": "performance_month",
                "account_id": account_id,
                "grid_id": "monthly-grid-4",
                "performance_month": month,
                "table_id": "monthly-table-4",
                "row": 3,
                "column": 11,
                "evidence_ids": ["native:monthly-grid-4:r3:c11"],
            }
        ],
    }
    return source, [
        _evidence(issue_id, "account_id", account_id),
        _evidence(issue_id, "performance_month", month),
    ]


def _compact(
    source_issue: dict[str, object], evidence: list[dict[str, object]]
) -> dict[str, object]:
    public_issue = {
        "record_id": source_issue["record_id"],
        "normalized": deepcopy(source_issue["normalized"]),
        "canonical_raw": {},
        "raw": {},
        "source": {"page_range": [12, 12]},
    }
    payload = {
        "datasets": [
            {"name": "extraction_issues", "rows": [public_issue], "columns": []},
            {"name": "extraction_issue_evidence", "rows": [], "columns": []},
        ]
    }
    _compact_personal_detail_public_projection(
        payload,
        source_datasets=[
            SimpleNamespace(public={"name": "extraction_issues"}, rows=[source_issue]),
            SimpleNamespace(
                public={"name": "extraction_issue_evidence"}, rows=evidence
            ),
        ],
    )
    return payload["datasets"][0]["rows"][0]


@pytest.mark.parametrize("kind", ["account", "agreement", "inquiry", "monthly"])
def test_exact_omission_issue_retains_only_its_auditable_source_ref(kind: str) -> None:
    source, evidence = _case(kind)

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["page_range"] == [12, 12]


def test_account_month_range_omission_retains_exact_identity_line_ref() -> None:
    source, evidence = _range_case()

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["page_range"] == [12, 12]


def test_owned_grid_month_omission_retains_one_exact_identity_cell_ref() -> None:
    source, evidence = _owned_grid_case()

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["page_range"] == [12, 12]


def test_owned_grid_month_ref_survives_schema_projection_and_community_compaction() -> None:
    source, evidence = _owned_grid_case()
    values = source["normalized"]
    raw_issue = {
        "record_id": source["record_id"],
        **values,
        "observed_value": {
            "account_id": evidence[0]["normalized"]["string_value"],
            "performance_month": evidence[1]["normalized"]["string_value"],
        },
        "reason_codes": ["exact_owner_bound_grid_month"],
        "source_refs": deepcopy(source["source_refs"]),
    }
    projected = project_personal_detail_datasets(
        {"personal_detail_extraction_issues": [raw_issue]}
    )
    projected_issue = projected["extraction_issues"][0]
    public_payload = {
        "datasets": [
            {
                "name": "extraction_issues",
                "rows": [
                    {
                        "record_id": projected_issue["record_id"],
                        "normalized": deepcopy(projected_issue),
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": [12, 12]},
                    }
                ],
                "columns": [],
            },
            {
                "name": "extraction_issue_evidence",
                "rows": [],
                "columns": [],
            },
        ]
    }
    _compact_personal_detail_public_projection(
        public_payload,
        source_datasets=[
            SimpleNamespace(
                public={"name": "extraction_issues"},
                rows=projected["extraction_issues"],
            ),
            SimpleNamespace(
                public={"name": "extraction_issue_evidence"},
                rows=projected["extraction_issue_evidence"],
            ),
        ],
    )

    compact_issue = public_payload["datasets"][0]["rows"][0]
    assert compact_issue["source"]["source_refs"] == source["source_refs"]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_code",
        "wrong_dataset",
        "status_field",
        "malformed_target",
        "wrong_hash",
        "duplicate_account_evidence",
        "duplicate_month_evidence",
        "unexpected_observed_evidence",
        "candidate_evidence",
        "wrong_month_evidence",
        "evidence_page_mismatch",
        "ref_page_mismatch",
        "wrong_source",
        "wrong_binding",
        "wrong_binding_quality",
        "wrong_ref_account",
        "empty_grid",
        "wrong_ref_month",
        "wrong_ref_field",
        "broad_geometry",
        "missing_table",
        "negative_row",
        "boolean_column",
        "invalid_bbox",
        "missing_evidence_ids",
        "duplicate_evidence_ids",
        "duplicate_ref",
    ],
)
def test_owned_grid_month_ref_restoration_fails_closed(mutation: str) -> None:
    source, evidence = _owned_grid_case()
    values = source["normalized"]
    ref = source["source_refs"][0]
    if mutation == "wrong_code":
        values["issue_code"] = "candidate_b_monthly_grid_contract_missing_field"
    elif mutation == "wrong_dataset":
        values["target_dataset"] = "credit_accounts"
    elif mutation == "status_field":
        values["field_name"] = "status_code"
        ref["field_name"] = "status_code"
    elif mutation == "malformed_target":
        values["target_record_id"] = "source_account_month:not-a-hash:2023-11"
    elif mutation == "wrong_hash":
        values["target_record_id"] = "source_account_month:0000000000000000:2023-11"
    elif mutation == "duplicate_account_evidence":
        evidence.append(deepcopy(evidence[0]))
    elif mutation == "duplicate_month_evidence":
        evidence.append(deepcopy(evidence[1]))
    elif mutation == "unexpected_observed_evidence":
        evidence.append(_evidence(str(source["record_id"]), "grid_id", "monthly-grid-4"))
    elif mutation == "candidate_evidence":
        evidence.append(
            _evidence(
                str(source["record_id"]),
                "performance_month",
                "2023-11",
                kind="candidate",
            )
        )
    elif mutation == "wrong_month_evidence":
        evidence[1]["normalized"]["string_value"] = "2023-10"
    elif mutation == "evidence_page_mismatch":
        evidence[1]["source"]["page_range"] = [11, 11]
    elif mutation == "ref_page_mismatch":
        ref["logical_page"] = 11
    elif mutation == "wrong_source":
        ref["source"] = "native_detail_table_cell"
    elif mutation == "wrong_binding":
        ref["binding"] = "monthly_grid_cell"
    elif mutation == "wrong_binding_quality":
        ref["binding_quality"] = "monthly_grid_cell"
    elif mutation == "wrong_ref_account":
        ref["account_id"] = "credit_account:revolving_loan_account:5"
    elif mutation == "empty_grid":
        ref["grid_id"] = ""
    elif mutation == "wrong_ref_month":
        ref["performance_month"] = "2023-10"
    elif mutation == "wrong_ref_field":
        ref["field_name"] = "status_code"
    elif mutation == "broad_geometry":
        ref["geometry_scope"] = "row"
    elif mutation == "missing_table":
        ref["table_id"] = ""
    elif mutation == "negative_row":
        ref["row"] = -1
    elif mutation == "boolean_column":
        ref["column"] = True
    elif mutation == "invalid_bbox":
        ref["bbox"] = [42.0, 120.0, 42.0, 132.0]
    elif mutation == "missing_evidence_ids":
        ref["evidence_ids"] = []
    elif mutation == "duplicate_evidence_ids":
        ref["evidence_ids"] = ["native:duplicate", "native:duplicate"]
    elif mutation == "duplicate_ref":
        source["source_refs"].append(deepcopy(ref))

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]


@pytest.mark.parametrize(
    "mutation",
    [
        "status_diagnostic",
        "malformed_target",
        "wrong_hash",
        "duplicate_account_evidence",
        "duplicate_month_evidence",
        "wrong_month_evidence",
        "evidence_page_mismatch",
        "ref_page_mismatch",
        "wrong_ref_account",
        "wrong_ref_month",
        "missing_evidence_ids",
        "duplicate_ref",
        "mixed_ref",
        "status_binding",
    ],
)
def test_account_month_range_ref_restoration_fails_closed(mutation: str) -> None:
    source, evidence = _range_case()
    values = source["normalized"]
    ref = source["source_refs"][0]
    if mutation == "status_diagnostic":
        values["issue_code"] = (
            "candidate_b_monthly_account_range_status_grid_unavailable"
        )
        values["field_name"] = "status_code"
        ref["field_name"] = "status_code"
        ref["binding"] = "source_account_month_identity"
        ref["binding_quality"] = "source_account_month_identity"
    elif mutation == "malformed_target":
        values["target_record_id"] = "source_account_month:123:2024-03"
    elif mutation == "wrong_hash":
        values["target_record_id"] = "source_account_month:0000000000000000:2024-03"
    elif mutation == "duplicate_account_evidence":
        evidence.append(deepcopy(evidence[0]))
    elif mutation == "duplicate_month_evidence":
        evidence.append(deepcopy(evidence[1]))
    elif mutation == "wrong_month_evidence":
        evidence[1]["normalized"]["string_value"] = "2024-02"
    elif mutation == "evidence_page_mismatch":
        evidence[1]["source"]["page_range"] = [11, 11]
    elif mutation == "ref_page_mismatch":
        ref["logical_page"] = 11
    elif mutation == "wrong_ref_account":
        ref["account_id"] = "credit_account:credit_card:2"
    elif mutation == "wrong_ref_month":
        ref["performance_month"] = "2024-02"
    elif mutation == "missing_evidence_ids":
        ref["evidence_ids"] = []
    elif mutation == "duplicate_ref":
        source["source_refs"].append(deepcopy(ref))
    elif mutation == "mixed_ref":
        source["source_refs"].append({**ref, "geometry_scope": "cell"})
    elif mutation == "status_binding":
        ref["binding"] = "monthly_grid_cell"

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]


@pytest.mark.parametrize(
    "mutation",
    [
        "unsupported_code",
        "resolved",
        "wrong_target",
        "wrong_field",
        "malformed_identity",
        "mismatched_public_identity",
        "mismatched_record_id",
        "missing_identity_evidence",
        "wrong_identity_evidence",
        "evidence_page_mismatch",
        "boolean_page",
        "missing_bbox",
        "nonfinite_bbox",
        "empty_evidence_ids",
        "duplicate_evidence_ids",
        "broad_geometry",
        "wrong_binding",
        "extra_untrusted_ref",
    ],
)
def test_exact_omission_ref_restoration_fails_closed(mutation: str) -> None:
    source, evidence = _case("monthly")
    values = source["normalized"]
    ref = source["source_refs"][0]
    if mutation == "unsupported_code":
        values["issue_code"] = "candidate_b_monthly_grid_contract_unresolved"
    elif mutation == "resolved":
        values["status"] = "resolved"
    elif mutation == "wrong_target":
        values["target_dataset"] = "credit_accounts"
    elif mutation == "wrong_field":
        values["field_name"] = "status_amount"
    elif mutation == "malformed_identity":
        values["target_record_id"] = "grid-7:not-a-month"
    elif mutation == "mismatched_public_identity":
        row = _compact(source, evidence)
        assert row["source"].get("source_refs")
        source["normalized"]["status"] = "open"
        public = deepcopy(source)
        public["normalized"]["status"] = "requires_review"
        # The helper below receives one public copy, so emulate a rich/public
        # disagreement through the source record identity instead.
        source["normalized"]["extraction_issue_id"] = "issue:foreign"
    elif mutation == "mismatched_record_id":
        source["record_id"] = "issue:foreign"
    elif mutation == "missing_identity_evidence":
        evidence.pop()
    elif mutation == "wrong_identity_evidence":
        evidence[-1]["normalized"]["string_value"] = "2024-02"
    elif mutation == "evidence_page_mismatch":
        evidence[-1]["source"]["page_range"] = [11, 11]
    elif mutation == "boolean_page":
        ref["logical_page"] = True
    elif mutation == "missing_bbox":
        ref.pop("bbox")
    elif mutation == "nonfinite_bbox":
        ref["bbox"] = [10.0, 20.0, float("inf"), 40.0]
    elif mutation == "empty_evidence_ids":
        ref["evidence_ids"] = []
    elif mutation == "duplicate_evidence_ids":
        ref["evidence_ids"] = ["ocr:exact:3", "ocr:exact:3"]
    elif mutation == "broad_geometry":
        ref["geometry_scope"] = "grid"
    elif mutation == "wrong_binding":
        ref["binding"] = "canonical_header_row"
    elif mutation == "extra_untrusted_ref":
        source["source_refs"].append({**ref, "geometry_scope": "grid"})

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]


def _exact_field_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    issue_id = "issue:exact-field"
    values: dict[str, object] = {
        "extraction_issue_id": issue_id,
        "issue_code": "candidate_b_exact_slot_value_invalid",
        "status": "requires_review",
        "target_dataset": "subject_employment",
        "target_record_id": "credit_employment:1",
        "field_name": "employer_phone",
    }
    source: dict[str, object] = {
        "record_id": issue_id,
        "normalized": values,
        "source_refs": [
            {
                "source": "native_detail_table_cell",
                "logical_page": 12,
                "source_page": 6,
                "table_id": "pt_2_1",
                "row": 1,
                "column": 4,
                "geometry_scope": "cell",
                "bbox": [335.5, 290.0, 385.0, 310.0],
                "evidence_ids": ["ocr:sp0001:lp0002:0054"],
                "binding": "canonical_field_slot",
                "binding_quality": "canonical_header_column",
                "field_name": "employer_phone",
            }
        ],
    }
    return source, []


def _inquiry_raw_position_case(
    *, split_band: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    issue_id = "issue:inquiry-raw-position"
    evidence_ids = ["ocr:sp0006:lp0012:0031", "ocr:sp0006:lp0012:0032"]
    target_record_id = stable_record_id(
        "source_inquiry_raw_physical_position",
        12,
        6,
        "pt_12_0",
        3,
        *sorted(evidence_ids),
    )
    values: dict[str, object] = {
        "extraction_issue_id": issue_id,
        "issue_code": "source_inquiry_physical_record_omitted",
        "status": "requires_review",
        "target_dataset": "inquiries",
        "target_record_id": target_record_id,
        "field_name": "inquiry_id",
        "observed_value_type": "object",
        "candidate_value_type": "object",
    }
    source: dict[str, object] = {
        "record_id": issue_id,
        "normalized": values,
        "source_refs": [
            {
                "source": "native_detail_inquiry_raw_physical_row",
                "logical_page": 12,
                "source_page": 6,
                "table_id": "pt_12_0",
                "row": 3,
                "geometry_scope": (
                    "token_y_band" if split_band else "exact_source_row_band"
                ),
                "bbox": [80.0, 220.0, 300.0, 232.0],
                "evidence_ids": evidence_ids,
                "binding": "sealed_raw_inquiry_registered_lattice_band",
                "binding_quality": (
                    "all_exact_tokens_uniquely_partitioned"
                    if split_band
                    else "sealed_exact_physical_position"
                ),
            }
        ],
    }
    evidence = [
        _evidence(issue_id, "source_row_observed", True),
        _evidence(
            issue_id,
            "source_physical_row_id",
            target_record_id,
            kind="candidate",
        ),
    ]
    return source, evidence


@pytest.mark.parametrize("split_band", [False, True])
def test_inquiry_raw_position_issue_retains_only_exact_row_band_ref(
    split_band: bool,
) -> None:
    source, evidence = _inquiry_raw_position_case(split_band=split_band)

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["page_range"] == [12, 12]
    assert not {
        "inquiry_type",
        "sequence",
        "value",
        "raw_value",
        "normalized_value",
    }.intersection(row["source"]["source_refs"][0])


@pytest.mark.parametrize("split_band", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_code",
        "wrong_dataset",
        "wrong_field",
        "canonical_namespace",
        "malformed_target",
        "wrong_target_hash",
        "wrong_observed",
        "wrong_observed_type",
        "wrong_candidate",
        "wrong_candidate_type",
        "missing_observed",
        "missing_candidate",
        "duplicate_observed",
        "duplicate_candidate",
        "unexpected_observed_path",
        "unexpected_candidate_path",
        "unexpected_evidence_kind",
        "missing_object_type",
        "wrong_evidence_page",
        "wrong_ref_page",
        "wrong_source",
        "wrong_scope",
        "wrong_binding",
        "wrong_binding_quality",
        "crossed_scope_quality",
        "missing_table",
        "nonstring_table",
        "negative_row",
        "boolean_row",
        "unexpected_column",
        "invalid_bbox",
        "boolean_bbox",
        "missing_evidence_ids",
        "duplicate_evidence_ids",
        "reordered_hash_input",
        "forbidden_type",
        "forbidden_ordinal",
        "forbidden_value",
        "extra_metadata",
        "extra_ref",
    ],
)
def test_inquiry_raw_position_restoration_fails_closed(
    split_band: bool,
    mutation: str,
) -> None:
    source, evidence = _inquiry_raw_position_case(split_band=split_band)
    values = source["normalized"]
    ref = source["source_refs"][0]
    if mutation == "wrong_code":
        values["issue_code"] = "source_inquiry_record_omitted"
    elif mutation == "wrong_dataset":
        values["target_dataset"] = "credit_accounts"
    elif mutation == "wrong_field":
        values["field_name"] = "sequence"
    elif mutation == "canonical_namespace":
        values["target_record_id"] = "source_inquiry_physical_row:0123456789abcdef"
        candidate = next(
            row for row in evidence if row["normalized"]["evidence_kind"] == "candidate"
        )
        candidate["normalized"]["string_value"] = values["target_record_id"]
    elif mutation == "malformed_target":
        values["target_record_id"] = "source_inquiry_raw_physical_position:not-a-digest"
    elif mutation == "wrong_target_hash":
        values["target_record_id"] = "source_inquiry_raw_physical_position:ffffffffffffffff"
        candidate = next(
            row for row in evidence if row["normalized"]["evidence_kind"] == "candidate"
        )
        candidate["normalized"]["string_value"] = values["target_record_id"]
    elif mutation == "wrong_observed":
        evidence[0]["normalized"]["boolean_value"] = False
    elif mutation == "wrong_observed_type":
        normalized = evidence[0]["normalized"]
        normalized["value_type"] = "integer"
        normalized.pop("boolean_value")
        normalized["integer_value"] = 1
    elif mutation == "wrong_candidate":
        evidence[1]["normalized"]["string_value"] = "source_inquiry_raw_physical_position:ffffffffffffffff"
    elif mutation == "wrong_candidate_type":
        normalized = evidence[1]["normalized"]
        normalized["value_type"] = "integer"
        normalized.pop("string_value")
        normalized["integer_value"] = 1
    elif mutation == "missing_observed":
        evidence.pop(0)
    elif mutation == "missing_candidate":
        evidence.pop()
    elif mutation == "duplicate_observed":
        evidence.append(deepcopy(evidence[0]))
    elif mutation == "duplicate_candidate":
        evidence.append(deepcopy(evidence[1]))
    elif mutation == "unexpected_observed_path":
        evidence[0]["normalized"]["evidence_path"] = "sequence"
    elif mutation == "unexpected_candidate_path":
        evidence[1]["normalized"]["evidence_path"] = "inquiry_type"
    elif mutation == "unexpected_evidence_kind":
        evidence[0]["normalized"]["evidence_kind"] = "raw"
    elif mutation == "missing_object_type":
        values.pop("candidate_value_type")
    elif mutation == "wrong_evidence_page":
        evidence[0]["source"] = {"page_range": [13, 13]}
    elif mutation == "wrong_ref_page":
        ref["logical_page"] = 11
    elif mutation == "wrong_source":
        ref["source"] = "native_detail_inquiry_physical_row"
    elif mutation == "wrong_scope":
        ref["geometry_scope"] = "cell"
    elif mutation == "wrong_binding":
        ref["binding"] = "canonical_header_row"
    elif mutation == "wrong_binding_quality":
        ref["binding_quality"] = "exact_token_row"
    elif mutation == "crossed_scope_quality":
        ref["binding_quality"] = (
            "sealed_exact_physical_position"
            if split_band
            else "all_exact_tokens_uniquely_partitioned"
        )
    elif mutation == "missing_table":
        ref.pop("table_id")
    elif mutation == "nonstring_table":
        ref["table_id"] = 7
        target = stable_record_id(
            "source_inquiry_raw_physical_position",
            ref["logical_page"],
            ref["source_page"],
            ref["table_id"],
            ref["row"],
            *sorted(ref["evidence_ids"]),
        )
        values["target_record_id"] = target
        evidence[1]["normalized"]["string_value"] = target
    elif mutation == "negative_row":
        ref["row"] = -1
    elif mutation == "boolean_row":
        ref["row"] = True
    elif mutation == "unexpected_column":
        ref["column"] = 0
    elif mutation == "invalid_bbox":
        ref["bbox"] = [80.0, 220.0, 80.0, 232.0]
    elif mutation == "boolean_bbox":
        ref["bbox"] = [True, 220.0, 300.0, 232.0]
    elif mutation == "missing_evidence_ids":
        ref["evidence_ids"] = []
    elif mutation == "duplicate_evidence_ids":
        ref["evidence_ids"] = ["ocr:exact:1", "ocr:exact:1"]
    elif mutation == "reordered_hash_input":
        ref["evidence_ids"] = list(reversed(ref["evidence_ids"])) + ["ocr:extra"]
    elif mutation == "forbidden_type":
        ref["inquiry_type"] = "institution"
    elif mutation == "forbidden_ordinal":
        ref["sequence"] = 3
    elif mutation == "forbidden_value":
        ref["value"] = "withheld"
    elif mutation == "extra_metadata":
        ref["owner"] = "adjacent_inquiry"
    elif mutation == "extra_ref":
        source["source_refs"].append(deepcopy(ref))

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]


def test_inquiry_raw_position_survives_schema_and_community_projection() -> None:
    source, evidence = _inquiry_raw_position_case()
    values = source["normalized"]
    raw_issue = {
        "record_id": source["record_id"],
        **values,
        "observed_value": {"source_row_observed": True},
        "candidate_value": {
            "source_physical_row_id": values["target_record_id"]
        },
        "reason_codes": ["sealed_raw_inquiry_physical_position"],
        "source_refs": deepcopy(source["source_refs"]),
    }
    projected = project_personal_detail_datasets(
        {"personal_detail_extraction_issues": [raw_issue]}
    )
    projected_issue = projected["extraction_issues"][0]
    public_payload = {
        "datasets": [
            {
                "name": "extraction_issues",
                "rows": [
                    {
                        "record_id": projected_issue["record_id"],
                        "normalized": deepcopy(projected_issue),
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": [12, 12]},
                    }
                ],
                "columns": [],
            },
            {"name": "extraction_issue_evidence", "rows": [], "columns": []},
        ]
    }

    _compact_personal_detail_public_projection(
        public_payload,
        source_datasets=[
            SimpleNamespace(
                public={"name": "extraction_issues"},
                rows=projected["extraction_issues"],
            ),
            SimpleNamespace(
                public={"name": "extraction_issue_evidence"},
                rows=projected["extraction_issue_evidence"],
            ),
        ],
    )

    compact_issue = public_payload["datasets"][0]["rows"][0]
    assert compact_issue["source"]["source_refs"] == source["source_refs"]


def _inquiry_physical_field_case(
    *, typed: bool, raw: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    issue_id = (
        f"issue:inquiry-physical:{'typed' if typed else 'anonymous'}:"
        f"{'raw' if raw else 'canonical'}"
    )
    target_record_id = (
        "credit_inquiry:institution:3"
        if typed
        else (
            "source_inquiry_raw_physical_position:0123456789abcdef"
            if raw
            else "source_inquiry_physical_row:0123456789abcdef"
        )
    )
    candidate_value: dict[str, object] = (
        {"inquiry_type": "institution", "sequence": 3}
        if typed
        else {"source_physical_row_id": target_record_id}
    )
    values: dict[str, object] = {
        "extraction_issue_id": issue_id,
        "issue_code": "source_inquiry_field_omitted",
        "status": "requires_review",
        "target_dataset": "inquiries",
        "target_record_id": target_record_id,
        "field_name": "institution",
        "observed_value_type": "object",
        "candidate_value_type": "object",
    }
    source: dict[str, object] = {
        "record_id": issue_id,
        "normalized": values,
        "source_refs": [
            {
                "source": (
                    "native_detail_inquiry_raw_physical_field"
                    if raw
                    else "native_detail_inquiry_physical_field"
                ),
                "logical_page": 12,
                "source_page": 6,
                "table_id": "pt_12_0",
                "row": 3,
                "column": 2,
                "geometry_scope": "token_y_band",
                "bbox": [110.0, 220.0, 180.0, 232.0],
                "evidence_ids": ["ocr:sp0006:lp0012:0037"],
                "binding": (
                    "sealed_raw_inquiry_role_y_band"
                    if raw
                    else "canonical_inquiry_column_y_band"
                ),
                "binding_quality": (
                    "exact_tokens_uniquely_partitioned_in_registered_lattice"
                    if raw
                    else "exact_tokens_uniquely_owned_by_date_band"
                ),
                "field_name": "institution",
            }
        ],
    }
    evidence = [
        _evidence(issue_id, "source_field_observed", True),
        *[
            _evidence(issue_id, path, value, kind="candidate")
            for path, value in candidate_value.items()
        ],
    ]
    return source, evidence


@pytest.mark.parametrize("typed", [False, True])
def test_inquiry_physical_field_finding_retains_exact_token_band_ref(
    typed: bool,
) -> None:
    source, evidence = _inquiry_physical_field_case(typed=typed)

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["page_range"] == [12, 12]


@pytest.mark.parametrize("typed", [False, True])
def test_inquiry_raw_physical_field_retains_only_exact_named_band_ref(
    typed: bool,
) -> None:
    source, evidence = _inquiry_physical_field_case(typed=typed, raw=True)

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["source_refs"][0]["source"] == (
        "native_detail_inquiry_raw_physical_field"
    )
    assert row["source"]["page_range"] == [12, 12]


@pytest.mark.parametrize(
    ("target_namespace", "raw_ref"),
    [
        ("source_inquiry_physical_row", True),
        ("source_inquiry_raw_physical_position", False),
    ],
)
def test_inquiry_anonymous_field_rejects_cross_namespace_ref_tuple(
    target_namespace: str,
    raw_ref: bool,
) -> None:
    source, evidence = _inquiry_physical_field_case(typed=False, raw=raw_ref)
    target = f"{target_namespace}:0123456789abcdef"
    source["normalized"]["target_record_id"] = target
    candidate = next(
        row for row in evidence if row["normalized"]["evidence_kind"] == "candidate"
    )
    candidate["normalized"]["string_value"] = target

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]


def test_inquiry_raw_field_survives_schema_and_community_projection() -> None:
    source, _evidence_rows = _inquiry_physical_field_case(typed=False, raw=True)
    values = source["normalized"]
    target = str(values["target_record_id"])
    raw_issue = {
        "record_id": source["record_id"],
        **values,
        "observed_value": {"source_field_observed": True},
        "candidate_value": {"source_physical_row_id": target},
        "reason_codes": ["sealed_raw_inquiry_role_y_band"],
        "source_refs": deepcopy(source["source_refs"]),
    }
    projected = project_personal_detail_datasets(
        {"personal_detail_extraction_issues": [raw_issue]}
    )
    projected_issue = projected["extraction_issues"][0]
    public_payload = {
        "datasets": [
            {
                "name": "extraction_issues",
                "rows": [
                    {
                        "record_id": projected_issue["record_id"],
                        "normalized": deepcopy(projected_issue),
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": [12, 12]},
                    }
                ],
                "columns": [],
            },
            {"name": "extraction_issue_evidence", "rows": [], "columns": []},
        ]
    }

    _compact_personal_detail_public_projection(
        public_payload,
        source_datasets=[
            SimpleNamespace(
                public={"name": "extraction_issues"},
                rows=projected["extraction_issues"],
            ),
            SimpleNamespace(
                public={"name": "extraction_issue_evidence"},
                rows=projected["extraction_issue_evidence"],
            ),
        ],
    )

    compact_issue = public_payload["datasets"][0]["rows"][0]
    assert compact_issue["source"]["source_refs"] == source["source_refs"]


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "unsupported_code",
        "wrong_dataset",
        "wrong_field",
        "malformed_target",
        "wrong_candidate",
        "wrong_candidate_type",
        "wrong_observed",
        "wrong_observed_type",
        "missing_issue_evidence",
        "wrong_evidence_page",
        "unexpected_candidate_path",
        "unexpected_evidence_kind",
        "duplicate_candidate_evidence",
        "missing_object_type",
        "page_mismatch",
        "wrong_source",
        "cell_scope",
        "wrong_binding",
        "wrong_binding_quality",
        "wrong_ref_field",
        "missing_table",
        "nonstring_table",
        "negative_row",
        "boolean_column",
        "invalid_bbox",
        "boolean_bbox",
        "duplicate_evidence",
        "extra_metadata",
        "duplicate_ref",
    ],
)
def test_inquiry_physical_field_ref_restoration_fails_closed(
    typed: bool,
    raw: bool,
    mutation: str,
) -> None:
    source, evidence = _inquiry_physical_field_case(typed=typed, raw=raw)
    values = source["normalized"]
    ref = source["source_refs"][0]
    if mutation == "unsupported_code":
        values["issue_code"] = "candidate_b_exact_slot_value_invalid"
    elif mutation == "wrong_dataset":
        values["target_dataset"] = "credit_accounts"
    elif mutation == "wrong_field":
        values["field_name"] = "sequence"
        ref["field_name"] = "sequence"
    elif mutation == "malformed_target":
        values["target_record_id"] = (
            "source_inquiry_raw_physical_position:not-a-digest"
            if raw
            else "source_inquiry_physical_row:not-a-digest"
        )
    elif mutation == "wrong_candidate":
        candidate = next(
            row
            for row in evidence
            if row["normalized"]["evidence_kind"] == "candidate"
        )
        normalized = candidate["normalized"]
        if normalized["value_type"] == "integer":
            normalized["integer_value"] = 99
        else:
            normalized["string_value"] = "foreign"
    elif mutation == "wrong_candidate_type":
        candidate = next(
            row
            for row in evidence
            if row["normalized"]["evidence_kind"] == "candidate"
            and row["normalized"]["value_type"] == ("integer" if typed else "string")
        )
        normalized = candidate["normalized"]
        if typed:
            normalized["value_type"] = "number"
            normalized["number_value"] = float(normalized.pop("integer_value"))
        else:
            normalized["value_type"] = "integer"
            normalized.pop("string_value")
            normalized["integer_value"] = 1
    elif mutation == "wrong_observed":
        observed = next(
            row
            for row in evidence
            if row["normalized"]["evidence_kind"] == "observed"
        )
        observed["normalized"]["boolean_value"] = False
    elif mutation == "wrong_observed_type":
        observed = next(
            row
            for row in evidence
            if row["normalized"]["evidence_kind"] == "observed"
        )
        normalized = observed["normalized"]
        normalized["value_type"] = "integer"
        normalized.pop("boolean_value")
        normalized["integer_value"] = 1
    elif mutation == "missing_issue_evidence":
        evidence.clear()
    elif mutation == "wrong_evidence_page":
        evidence[0]["source"] = {"page_range": [13, 13]}
    elif mutation == "unexpected_candidate_path":
        candidate = next(
            row
            for row in evidence
            if row["normalized"]["evidence_kind"] == "candidate"
        )
        candidate["normalized"]["evidence_path"] = "foreign_identity"
    elif mutation == "unexpected_evidence_kind":
        evidence[0]["normalized"]["evidence_kind"] = "raw"
    elif mutation == "duplicate_candidate_evidence":
        candidate = next(
            row
            for row in evidence
            if row["normalized"]["evidence_kind"] == "candidate"
        )
        evidence.append(deepcopy(candidate))
    elif mutation == "missing_object_type":
        values.pop("candidate_value_type")
    elif mutation == "page_mismatch":
        ref["logical_page"] = 11
    elif mutation == "wrong_source":
        ref["source"] = "native_detail_table_cell"
    elif mutation == "cell_scope":
        ref["geometry_scope"] = "cell"
    elif mutation == "wrong_binding":
        ref["binding"] = "canonical_header_column"
    elif mutation == "wrong_binding_quality":
        ref["binding_quality"] = "exact_cell_in_sequence_column"
    elif mutation == "wrong_ref_field":
        ref["field_name"] = "reason"
    elif mutation == "missing_table":
        ref.pop("table_id")
    elif mutation == "nonstring_table":
        ref["table_id"] = 7
    elif mutation == "negative_row":
        ref["row"] = -1
    elif mutation == "boolean_column":
        ref["column"] = True
    elif mutation == "invalid_bbox":
        ref["bbox"] = [110.0, 220.0, 110.0, 232.0]
    elif mutation == "boolean_bbox":
        ref["bbox"] = [True, 220.0, 180.0, 232.0]
    elif mutation == "duplicate_evidence":
        ref["evidence_ids"] = ["ocr:exact:1", "ocr:exact:1"]
    elif mutation == "extra_metadata":
        ref["inquiry_type"] = "institution"
    elif mutation == "duplicate_ref":
        source["source_refs"].append(deepcopy(ref))

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]


def test_exact_field_finding_retains_its_auditable_cell_ref() -> None:
    source, evidence = _exact_field_case()

    row = _compact(source, evidence)

    assert row["source"]["source_refs"] == source["source_refs"]
    assert row["source"]["page_range"] == [12, 12]


@pytest.mark.parametrize(
    "mutation",
    [
        "aggregate_code",
        "missing_target",
        "wrong_field",
        "page_scope",
        "missing_table",
        "missing_bbox",
        "missing_evidence",
        "wrong_binding",
        "mixed_ref",
    ],
)
def test_exact_field_finding_ref_restoration_fails_closed(mutation: str) -> None:
    source, evidence = _exact_field_case()
    values = source["normalized"]
    ref = source["source_refs"][0]
    if mutation == "aggregate_code":
        values["issue_code"] = "candidate_b_employment_component_missing"
    elif mutation == "missing_target":
        values["target_record_id"] = None
    elif mutation == "wrong_field":
        ref["field_name"] = "employer_address"
    elif mutation == "page_scope":
        ref["geometry_scope"] = "table"
    elif mutation == "missing_table":
        ref.pop("table_id")
    elif mutation == "missing_bbox":
        ref.pop("bbox")
    elif mutation == "missing_evidence":
        ref["evidence_ids"] = []
    elif mutation == "wrong_binding":
        ref["binding"] = "nearest_row"
    elif mutation == "mixed_ref":
        source["source_refs"].append({**ref, "geometry_scope": "table"})

    row = _compact(source, evidence)

    assert "source_refs" not in row["source"]
