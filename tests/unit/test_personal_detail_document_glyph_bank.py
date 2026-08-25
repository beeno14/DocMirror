# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.document_glyph_bank import (
    _explicit_zero,
    _parse_year_month,
    apply_document_local_status_glyph_bank,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)


@pytest.mark.parametrize("value", ("0,0", "0,,000", "0000,000", "0,00.0"))
def test_glyph_bank_zero_proof_rejects_malformed_grouping(value: str) -> None:
    assert not _explicit_zero(value)


@pytest.mark.parametrize("value", ("2024-01x", "2024.01.02", "2024/01/extra"))
def test_glyph_bank_month_owner_requires_a_complete_month(value: str) -> None:
    assert _parse_year_month(value) == ""


def test_glyph_bank_accepts_registered_zero_and_month_presentations() -> None:
    assert _explicit_zero("0,000.00")
    assert _parse_year_month("2024.01") == "2024-01"


def _glyph(label: str) -> np.ndarray:
    glyph = np.zeros((32, 32), dtype=np.float32)
    if label == "N":
        glyph[5:27, 7:10] = 1
        glyph[5:27, 21:24] = 1
        for y in range(5, 27):
            x = 9 + int((y - 5) * 12 / 21)
            glyph[y, x - 1 : x + 2] = 1
        return glyph
    glyph[15:18, 6:27] = 1
    glyph[6:27, 15:18] = 1
    for offset in range(-9, 10):
        glyph[16 + offset, 15 + offset : 18 + offset] = 1
        glyph[16 + offset, 15 - offset : 18 - offset] = 1
    return glyph


def _record(
    repayment_id: str,
    *,
    status: str,
    amount: str = "0",
    review: bool = False,
    page: int = 1,
) -> dict[str, Any]:
    grid_id, performance_month = repayment_id.rsplit(":", 1)
    year, month = (int(value) for value in performance_month.split("-"))
    record: dict[str, Any] = {
        "repayment_id": repayment_id,
        "grid_id": grid_id,
        "account_id": "account:1",
        "year": year,
        "month": month,
        "status": status,
        "overdue_amount": amount,
        "source_cell_refs": [
            {
                "page": page,
                "logical_page": page,
                "grid_id": grid_id,
                "row": 1,
                "col": month,
                "field_name": "status",
                "geometry_scope": "cell",
                "bbox": [10.0, 10.0, 20.0, 20.0],
            }
        ],
    }
    if review:
        _mark_static_review(record)
    return record


def _mark_static_review(record: dict[str, Any]) -> None:
    record.update(
        {
            "extraction_status": "review",
            "recognition_source": "static_glyph_shape_unresolved",
            "audit": {
                "reason": "zero_status_static_corroboration_unavailable",
                "field_name": "status_code",
                "observed_status": record["status"],
                "reported_value_retained": True,
            },
        }
    )


def _observation(
    repayment_id: str,
    *,
    label: str,
    page: int,
    decisive: bool,
    template: np.ndarray | None = None,
    exact_geometry: bool = True,
) -> dict[str, Any]:
    grid_id, performance_month = repayment_id.rsplit(":", 1)
    year, month = (int(value) for value in performance_month.split("-"))
    return {
        "repayment_id": repayment_id,
        "grid_id": grid_id,
        "page": page,
        "year": year,
        "month": month,
        "observed_status": label,
        "resolved_status": label,
        "template": _glyph(label) if template is None else template,
        "decisive_label": label if decisive else "",
        "decisive_confidence": 0.97 if decisive else 0.0,
        "classifier_conflict": False,
        "alignment_exact": True,
        "exact_status_geometry": exact_geometry,
        "amount": "0",
        "amount_pair_exact": True,
        "status_amount_conflict": False,
        "source_ref": {
            "page": page,
            "logical_page": page,
            "grid_id": grid_id,
            "row": 1,
            "col": month,
            "field_name": "status",
            "geometry_scope": "cell",
            "bbox": [10.0, 10.0, 20.0, 20.0],
        },
    }


def _fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for label, prefix, year in (("N", "n", 2022), ("*", "star", 2023)):
        for grid_number, page in ((1, 1), (2, 2), (3, 2)):
            for slot in (1, 2):
                repayment_id = f"{prefix}_grid_{grid_number}:{year}-{grid_number * 2 + slot - 2:02d}"
                records.append(_record(repayment_id, status=label, page=page))
                observations.append(
                    _observation(
                        repayment_id,
                        label=label,
                        page=page,
                        decisive=True,
                    )
                )
    candidate_id = "candidate_grid:2024-06"
    records.append(_record(candidate_id, status="N", review=True, page=3))
    observations.append(_observation(candidate_id, label="N", page=3, decisive=False))
    return records, observations, candidate_id


def _apply(
    records: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    native: list[dict[str, Any]] | None = None,
    corrected: list[dict[str, Any]] | None = None,
    issues: list[dict[str, Any]] | None = None,
    accounts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return apply_document_local_status_glyph_bank(
        records,
        observations,
        accounts=(
            accounts if accounts is not None else [{"account_id": "account:1", "account_lifecycle_state": "active"}]
        ),
        issues=issues or [],
        native_plane_records=native if native is not None else deepcopy(records),
        corrected_plane_records=(corrected if corrected is not None else deepcopy(records)),
    )


def _candidate(records: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    return next(row for row in records if row["repayment_id"] == candidate_id)


def test_strict_document_bank_clears_only_the_static_review() -> None:
    records, observations, candidate_id = _fixture()

    audit = _apply(records, observations)

    candidate = _candidate(records, candidate_id)
    assert audit["enabled"] is True
    assert audit["promoted_count"] == 1
    assert candidate["status"] == "N"
    assert candidate["overdue_amount"] == "0"
    assert "extraction_status" not in candidate
    assert "audit" not in candidate
    assert audit["labels"]["N"]["seed_count"] == 7
    assert audit["labels"]["N"]["seed_basis_counts"] == {
        "decisive_static_classifier": 6,
        "exact_row_label_review": 1,
    }
    assert audit["labels"]["N"]["decisive_anchor_count"] == 6
    assert audit["labels"]["N"]["decisive_anchor_grid_count"] == 3
    assert audit["labels"]["N"]["decisive_anchor_page_count"] == 2
    assert audit["labels"]["*"]["decisive_anchor_count"] == 6
    assert audit["labels"]["*"]["decisive_anchor_grid_count"] == 3
    assert audit["labels"]["*"]["decisive_anchor_page_count"] == 2
    assert audit["labels"]["*"]["grid_count"] == 3
    assert audit["validation_mode"] == "all_selected_seeds_must_pass"


def test_bank_disabled_is_field_equivalent_and_one_label_cannot_enable_it() -> None:
    records, observations, _candidate_id = _fixture()
    observations = [row for row in observations if row.get("decisive_label") != "*"]
    before = deepcopy(records)

    audit = _apply(records, observations)

    assert audit["enabled"] is False
    assert audit["disabled_reason"] == "insufficient_cross_grid_page_seed_coverage"
    assert list(audit["labels"]) == ["N", "*"]
    assert audit["labels"]["N"]["seed_count"] == 7
    assert audit["labels"]["*"] == {
        "seed_count": 0,
        "grid_count": 0,
        "page_count": 0,
        "seed_basis_counts": {},
        "seed_record_ids": [],
        "seed_source_refs": [],
        "decisive_anchor_count": 0,
        "decisive_anchor_grid_count": 0,
        "decisive_anchor_page_count": 0,
        "decisive_anchor_record_ids": [],
        "decisive_anchor_source_refs": [],
    }
    assert audit["insufficient_labels"] == ["*"]
    assert audit["seed_selection"]["prepared_observation_count"] == 7
    assert audit["seed_selection"]["selected_seed_count"] == 7
    assert audit["seed_selection"]["rejection_counts"] == {}
    assert audit["seed_selection"]["labels"]["*"] == {
        "decisive_observation_count": 0,
        "exact_row_review_observation_count": 0,
        "eligible_seed_count_before_grid_cap": 0,
        "selected_seed_count": 0,
        "eligible_seed_basis_counts": {},
        "selected_seed_basis_counts": {},
        "rejection_counts": {},
    }
    assert records == before


def test_cross_label_templates_disable_the_bank() -> None:
    records, observations, _candidate_id = _fixture()
    for observation in observations:
        if observation.get("decisive_label") == "*":
            observation["template"] = _glyph("N")
    before = deepcopy(records)

    audit = _apply(records, observations)

    assert audit["enabled"] is False
    assert audit["disabled_reason"] == "cross_label_seed_margin_below_threshold"
    assert set(audit["validation_failures"]) == {"N", "*"}
    assert all("minimum_cross_label_margin" in audit["labels"][label] for label in ("N", "*"))
    assert records == before


@pytest.mark.parametrize(
    "veto_kind",
    ["noise", "plane_conflict", "active_issue", "geometry", "nonzero_amount"],
)
def test_candidate_vetoes_leave_review_unchanged(veto_kind: str) -> None:
    records, observations, candidate_id = _fixture()
    native = deepcopy(records)
    issues: list[dict[str, Any]] = []
    candidate_observation = next(row for row in observations if row["repayment_id"] == candidate_id)
    candidate_record = _candidate(records, candidate_id)
    if veto_kind == "noise":
        rng = np.random.default_rng(12)
        candidate_observation["template"] = (rng.random((32, 32)) > 0.55).astype(np.float32)
    elif veto_kind == "plane_conflict":
        _candidate(native, candidate_id)["status"] = "C"
    elif veto_kind == "active_issue":
        issues.append(
            {
                "status": "requires_review",
                "issue_code": "candidate_b_independent_plane_repayment_status_conflict",
                "target_dataset": "repayment_records",
                "target_record_id": candidate_id,
                "field_name": "status_code",
            }
        )
    elif veto_kind == "geometry":
        candidate_observation["exact_status_geometry"] = False
    else:
        candidate_observation["amount"] = "10"
        candidate_record["overdue_amount"] = "10"
    before = deepcopy(candidate_record)

    audit = _apply(records, observations, native=native, issues=issues)

    if veto_kind == "noise":
        assert audit["enabled"] is False
        assert audit["disabled_reason"] == "leave_one_grid_out_similarity_below_threshold"
    else:
        assert audit["enabled"] is True
    assert _candidate(records, candidate_id) == before
    assert audit["promoted_count"] == 0


def test_one_label_only_decisive_anchors_cannot_enable_review_seed_bank() -> None:
    records, observations, candidate_id = _fixture()
    for observation in observations:
        if observation["observed_status"] != "*":
            continue
        observation["decisive_label"] = ""
        _mark_static_review(_candidate(records, observation["repayment_id"]))

    before = deepcopy(records)
    audit = _apply(records, observations)

    assert audit["enabled"] is False
    assert audit["disabled_reason"] == (
        "insufficient_independent_decisive_anchor_coverage"
    )
    assert audit["insufficient_anchor_labels"] == ["*"]
    assert audit["labels"]["*"]["seed_count"] == 6
    assert audit["labels"]["*"]["seed_basis_counts"] == {
        "exact_row_label_review": 6
    }
    assert audit["seed_selection"]["labels"]["*"][
        "exact_row_review_observation_count"
    ] == 6
    assert audit["seed_selection"]["selected_seed_basis_counts"] == {
        "decisive_static_classifier": 6,
        "exact_row_label_review": 7,
    }
    assert audit["labels"]["*"]["decisive_anchor_count"] == 0
    assert audit["labels"]["N"]["decisive_anchor_count"] == 6
    assert audit["promoted_count"] == 0
    assert records == before
    assert _candidate(records, candidate_id)["extraction_status"] == "review"


def test_exact_minimum_decisive_anchors_per_label_can_enable_bank() -> None:
    records, observations, candidate_id = _fixture()
    for label, prefix in (("N", "n_grid_"), ("*", "star_grid_")):
        label_observations = sorted(
            (
                observation
                for observation in observations
                if observation["grid_id"].startswith(prefix)
            ),
            key=lambda observation: (
                observation["page"],
                observation["grid_id"],
                observation["month"],
            ),
        )
        anchors = {
            label_observations[0]["repayment_id"],
            next(
                observation["repayment_id"]
                for observation in label_observations
                if observation["page"] == 2
            ),
        }
        for observation in label_observations:
            if observation["repayment_id"] in anchors:
                continue
            observation["decisive_label"] = ""
            _mark_static_review(
                _candidate(records, observation["repayment_id"])
            )

    audit = _apply(records, observations)

    assert audit["enabled"] is True
    for label in ("N", "*"):
        assert audit["labels"][label]["decisive_anchor_count"] == 2
        assert audit["labels"][label]["decisive_anchor_grid_count"] == 2
        assert audit["labels"][label]["decisive_anchor_page_count"] == 2
    assert audit["thresholds"]["minimum_decisive_anchors_per_label"] == 2
    assert audit["thresholds"]["minimum_decisive_anchor_grids_per_label"] == 2
    assert audit["thresholds"]["minimum_decisive_anchor_pages_per_label"] == 2
    assert "extraction_status" not in _candidate(records, candidate_id)


def test_globally_inverted_review_only_bank_cannot_self_validate() -> None:
    records, observations, _candidate_id = _fixture()
    for observation in observations:
        observation["decisive_label"] = ""
        observation["template"] = _glyph(
            "*" if observation["observed_status"] == "N" else "N"
        )
        _mark_static_review(_candidate(records, observation["repayment_id"]))
    before = deepcopy(records)

    audit = _apply(records, observations)

    assert audit["enabled"] is False
    assert audit["disabled_reason"] == (
        "insufficient_independent_decisive_anchor_coverage"
    )
    assert audit["insufficient_anchor_labels"] == ["N", "*"]
    assert audit["labels"]["N"]["decisive_anchor_count"] == 0
    assert audit["labels"]["*"]["decisive_anchor_count"] == 0
    assert audit["promoted_count"] == 0
    assert records == before


@pytest.mark.parametrize("sentinel", ["unknown", "unresolved"])
def test_non_substantive_plane_sentinel_does_not_veto_exact_row_seed(
    sentinel: str,
) -> None:
    records, observations, candidate_id = _fixture()
    native = deepcopy(records)
    _candidate(native, candidate_id)["status"] = sentinel

    audit = _apply(records, observations, native=native)

    assert audit["enabled"] is True
    assert audit["promoted_count"] == 1
    assert "extraction_status" not in _candidate(records, candidate_id)


@pytest.mark.parametrize("conflicting_field", ["observed_status", "resolved_status", "decisive_label"])
def test_row_seed_requires_every_available_label_to_agree(
    conflicting_field: str,
) -> None:
    records, observations, candidate_id = _fixture()
    observation = next(
        row for row in observations if row["repayment_id"] == candidate_id
    )
    observation[conflicting_field] = "*"
    before = deepcopy(_candidate(records, candidate_id))

    audit = _apply(records, observations)

    assert audit["enabled"] is True
    assert audit["labels"]["N"]["seed_count"] == 6
    assert audit["seed_selection"]["labels"]["N"][
        "exact_row_review_observation_count"
    ] == 0
    assert audit["promoted_count"] == 0
    assert _candidate(records, candidate_id) == before


def test_row_seed_cap_is_source_ordered_and_capped_out_mislabel_cannot_promote() -> None:
    records, observations, _candidate_id = _fixture()
    expected_star_seed_ids: list[str] = []
    capped_out_ids: list[str] = []
    for grid_number, page in ((1, 1), (2, 2), (3, 2)):
        grid_id = f"star_grid_{grid_number}"
        grid_records = sorted(
            (row for row in records if row["grid_id"] == grid_id),
            key=lambda row: row["repayment_id"],
        )
        expected_star_seed_ids.extend(row["repayment_id"] for row in grid_records)

        repayment_id = f"{grid_id}:2023-{grid_number + 8:02d}"
        records.append(
            _record(
                repayment_id,
                status="*",
                review=True,
                page=page,
            )
        )
        extra = _observation(
            repayment_id,
            label="*",
            page=page,
            decisive=False,
            template=_glyph("N"),
        )
        # A confidence spike must not influence the deterministic cap.
        extra["decisive_confidence"] = 1.0
        observations.append(extra)
        capped_out_ids.append(repayment_id)

    audit = _apply(records, observations)

    assert audit["enabled"] is True
    assert audit["labels"]["*"]["seed_record_ids"] == expected_star_seed_ids
    assert audit["seed_selection"]["labels"]["*"]["rejection_counts"] == {
        "per_grid_seed_cap": 3
    }
    assert all(
        _candidate(records, repayment_id)["extraction_status"] == "review"
        for repayment_id in capped_out_ids
    )


def test_one_incoherent_exact_row_seed_disables_the_whole_bank() -> None:
    records, observations, candidate_id = _fixture()
    candidate_observation = next(
        observation
        for observation in observations
        if observation["repayment_id"] == candidate_id
    )
    candidate_observation["template"] = _glyph("*")
    before = deepcopy(records)

    audit = _apply(records, observations)

    assert audit["enabled"] is False
    assert audit["labels"]["N"]["decisive_anchor_count"] == 6
    assert audit["labels"]["*"]["decisive_anchor_count"] == 6
    assert audit["validation_mode"] == "all_selected_seeds_must_pass"
    assert audit["disabled_reason"] in {
        "leave_one_grid_out_similarity_below_threshold",
        "cross_label_seed_margin_below_threshold",
        "within_label_variation_above_threshold",
    }
    assert records == before


def test_watermark_line_is_suppressed_but_unstructured_noise_is_not_promoted() -> None:
    records, observations, candidate_id = _fixture()
    candidate_observation = next(row for row in observations if row["repayment_id"] == candidate_id)
    glyph_with_grid_line = _glyph("N")
    glyph_with_grid_line[1, :] = 1
    candidate_observation["template"] = glyph_with_grid_line

    audit = _apply(records, observations)

    assert audit["promoted_count"] == 1
    assert "extraction_status" not in _candidate(records, candidate_id)


def test_account_lifecycle_issue_vetoes_but_unrelated_account_field_does_not() -> None:
    records, observations, candidate_id = _fixture()
    unrelated = {
        "status": "requires_review",
        "issue_code": "candidate_b_exact_slot_value_invalid",
        "target_dataset": "credit_accounts",
        "target_record_id": "account:1",
        "field_name": "management_institution",
    }

    audit = _apply(records, observations, issues=[unrelated])

    assert audit["promoted_count"] == 1
    assert "extraction_status" not in _candidate(records, candidate_id)

    records, observations, candidate_id = _fixture()
    lifecycle = {
        **unrelated,
        "issue_code": "candidate_b_account_lifecycle_conflict",
        "field_name": "account_status_code",
    }
    audit = _apply(records, observations, issues=[lifecycle])
    assert audit["promoted_count"] == 0
    assert _candidate(records, candidate_id)["extraction_status"] == "review"

    records, observations, candidate_id = _fixture()
    missing_anchor_grid = {
        **unrelated,
        "issue_code": "candidate_b_monthly_anchor_grid_missing",
        "field_name": "account_id",
    }
    audit = _apply(records, observations, issues=[missing_anchor_grid])
    assert audit["promoted_count"] == 0
    assert _candidate(records, candidate_id)["extraction_status"] == "review"

    records, observations, candidate_id = _fixture()
    audit = _apply(
        records,
        observations,
        accounts=[{"account_id": "account:1", "account_lifecycle_state": "settled"}],
    )
    assert audit["promoted_count"] == 0
    assert _candidate(records, candidate_id)["extraction_status"] == "review"


@pytest.mark.parametrize(
    ("issue_row", "expected_promotion"),
    [(1, False), (2, True), (None, False)],
)
def test_issue_ref_veto_is_status_row_aware_and_missing_row_is_conservative(
    issue_row: int | None,
    expected_promotion: bool,
) -> None:
    records, observations, candidate_id = _fixture()
    source_ref: dict[str, Any] = {
        "page": 3,
        "logical_page": 3,
        "grid_id": "candidate_grid",
        "col": 6,
        "field_name": "status",
        "geometry_scope": "cell",
        "bbox": [10.0, 10.0, 20.0, 20.0],
    }
    if issue_row is not None:
        source_ref["row"] = issue_row
    issue = {
        "status": "requires_review",
        "issue_code": "candidate_b_independent_plane_repayment_status_conflict",
        "target_dataset": "repayment_records",
        "target_record_id": "candidate_grid:2023-06",
        "field_name": "status_code",
        "source_refs": [source_ref],
    }

    audit = _apply(records, observations, issues=[issue])

    assert audit["enabled"] is True
    assert audit["promoted_count"] == int(expected_promotion)
    candidate = _candidate(records, candidate_id)
    assert ("extraction_status" not in candidate) is expected_promotion
    if expected_promotion:
        assert audit["veto_counts"] == {}
    else:
        assert audit["veto_counts"] == {"business_contract:active_issue_block": 1}


def test_templates_never_leak_to_records_or_v2_projection() -> None:
    records, observations, candidate_id = _fixture()
    observations.append(
        _observation(
            "invalid_glyph_grid:2024-07",
            label="N",
            page=4,
            decisive=True,
            template=np.zeros((32, 32), dtype=np.float32),
        )
    )

    audit = _apply(records, observations)
    projected = project_personal_detail_datasets({"repayment_records": records})
    serialized_records = json.dumps(records, ensure_ascii=False)
    serialized_projection = json.dumps(projected, ensure_ascii=False)
    serialized_audit = json.dumps(audit, ensure_ascii=False)
    candidate = next(
        row for row in projected["credit_account_monthly_performance"] if row["monthly_performance_id"] == candidate_id
    )

    assert "template" not in serialized_records
    assert "bitmap" not in serialized_records
    assert "vector" not in serialized_records
    assert "document_local_status_glyph_bank" not in serialized_projection
    assert "template" not in serialized_audit
    assert "bitmap" not in serialized_audit
    assert "vector" not in serialized_audit
    assert audit["observation_preparation"]["rejection_counts"] == {"glyph_normalization_failed": 1}
    assert candidate["status_code"] == "N"
    assert "review" not in candidate
    assert "audit" not in candidate
