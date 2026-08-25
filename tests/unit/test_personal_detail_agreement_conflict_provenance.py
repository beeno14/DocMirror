from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction

_IDENTIFIER = "B11911000H00016100000010048606842"


def test_agreement_matching_does_not_delete_institution_glyphs() -> None:
    assert native_extraction._agreement_strong_field_matches(
        {"institution": "导中国银行股份有限公司"},
        {"institution": "中国银行股份有限公司"},
    ) == 0


def test_agreement_matching_requires_registered_amount_grouping() -> None:
    assert native_extraction._agreement_strong_field_matches(
        {"total_limit": "1,2"},
        {"total_limit": 12},
    ) == 0
    assert native_extraction._agreement_strong_field_matches(
        {"total_limit": "1,200"},
        {"total_limit": 1200},
    ) == 1


def test_agreement_identifier_matching_rejects_deleted_residue() -> None:
    assert native_extraction._agreement_identifier_text("坏ABC12345678") == ""
    assert native_extraction._agreement_identifier_text("$ABC12345678") == ""
    assert native_extraction._agreement_identifier_text("ABC-12345678") == "ABC-12345678"


def _cross_plane_identifier_record(
    identifier: str,
    *,
    source: str,
    complete: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "account_identifier": identifier,
        "institution": "示例银行",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "_printed_sequence": 1,
        "_canonical_card_key": "credit_agreement:1",
        "_canonical_card_anchor_refs": [
            {
                "source": source,
                "binding": "canonical_card_anchor",
            }
        ],
        "source_refs": [],
        "source_refs_by_field": {},
        "_field_binding_quality": {},
        "_observed_fields": [
            "account_identifier",
            "institution",
            "facility_type",
            "effective_date",
        ],
    }
    if complete:
        record["due_date"] = "2025-01-01"
        record["_observed_fields"].append("due_date")
    return record


def test_agreement_cross_plane_anchor_does_not_merge_hyphen_colliding_identifiers() -> None:
    native = _cross_plane_identifier_record(
        "AB-CD1234",
        source="native_detail_canonical_anchor_text",
    )
    corrected = _cross_plane_identifier_record(
        "ABC-D1234",
        source="personal_detail_corrected_page_cell",
        complete=True,
    )

    authorized = native_extraction._agreement_authorized_cross_plane_anchors(
        [native, corrected]
    )
    assert authorized == set()
    assert not native_extraction._agreement_exact_anchor_authorizes_merge(
        native,
        corrected,
        {"credit_agreement:1"},
    )

    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [deepcopy(native), deepcopy(corrected)],
    )
    assert {row["account_identifier"] for row in reconciled} == {
        "AB-CD1234",
        "ABC-D1234",
    }


def test_agreement_cross_plane_anchor_still_accepts_one_exact_identifier() -> None:
    native = _cross_plane_identifier_record(
        "AB-CD1234",
        source="native_detail_canonical_anchor_text",
    )
    corrected = _cross_plane_identifier_record(
        "ab- cd1234",
        source="personal_detail_corrected_page_cell",
        complete=True,
    )

    authorized = native_extraction._agreement_authorized_cross_plane_anchors(
        [native, corrected]
    )
    assert authorized == {"credit_agreement:1"}
    assert native_extraction._agreement_exact_anchor_authorizes_merge(
        native,
        corrected,
        authorized,
    )

    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [deepcopy(native), deepcopy(corrected)],
    )
    assert len(reconciled) == 1
    assert reconciled[0]["account_identifier"] == "AB-CD1234"


def _anchor_ref() -> dict[str, Any]:
    return {
        "source": "native_detail_canonical_anchor_text",
        "logical_page": 48,
        "source_page": 24,
        "bbox": [56.0, 482.0, 88.0, 493.0],
        "geometry_scope": "text",
        "field_name": "sequence",
        "binding": "canonical_card_anchor",
    }


def _table_ref(*, table_id: str = "agreement-9") -> dict[str, Any]:
    return {
        "source": "native_detail_tolerant_table",
        "logical_page": 48,
        "source_page": 24,
        "table_id": table_id,
        "bbox": [54.0, 491.0, 400.0, 559.0],
        "geometry_scope": "table",
    }


def _cell_ref(
    *,
    table_id: str = "agreement-9",
    row: int = 0,
    column: int = 0,
    geometry_scope: str = "cell",
) -> dict[str, Any]:
    return {
        "source": "native_detail_tolerant_table_cell",
        "logical_page": 48,
        "source_page": 24,
        "table_id": table_id,
        "bbox": [55.0 + column * 70.0, 498.0 + row * 20.0, 124.0 + column * 70.0, 530.0 + row * 20.0],
        "geometry_scope": geometry_scope,
        "row": row,
        "column": column,
        "field_name": "institution",
        "binding": "label_column",
        "evidence_ids": [f"ocr:agreement-9:{row}:{column}"],
    }


def _agreement(
    institution: str,
    *,
    field_refs: list[dict[str, Any]],
    source_refs: list[dict[str, Any]] | None = None,
    binding: str = "native_label_column",
) -> dict[str, Any]:
    return {
        "credit_line_id": "credit_line:provisional",
        "account_identifier": _IDENTIFIER,
        "institution": institution,
        "facility_type": "信用卡共享额度",
        "effective_date": "2024-01-08",
        "due_date": None,
        "validity_type": "perpetual",
        "total_limit": 38_000,
        "credit_limit": None,
        "used_limit": 0,
        "limit_identifier": None,
        "currency": "CNY",
        "_printed_sequence": 9,
        "_canonical_card_key": "credit_agreement:9",
        "_canonical_card_anchor_refs": [_anchor_ref()],
        "source_refs": list(source_refs if source_refs is not None else [_table_ref()]),
        "source_refs_by_field": {"institution": field_refs},
        "_field_binding_quality": {"institution": binding},
        "_observed_fields": ["institution"],
    }


def _corrected_conflict() -> dict[str, Any]:
    ref = {
        "source": "personal_detail_corrected_page_cell",
        "logical_page": 48,
        "source_page": 24,
        "bbox": [58.0, 516.0, 122.0, 530.0],
        "geometry_scope": "cell",
        "field_name": "institution",
        "binding": "canonical_label_slot",
        "evidence_ids": ["reocr:agreement-9:institution"],
    }
    return _agreement(
        "样例银行股份有限公司",
        field_refs=[ref],
        source_refs=[
            {
                "source": "personal_detail_corrected_page_rows",
                "logical_page": 48,
                "source_page": 24,
                "geometry_scope": "logical_page",
            }
        ],
        binding="canonical_cell_slot",
    )


def _institution_required_issues(context: SimpleNamespace) -> list[dict[str, Any]]:
    return [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_credit_agreement_required_field_unresolved"
        and issue.get("field_name") == "institution"
    ]


def test_agreement_conflict_preserves_one_exact_card_owned_field_slot() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    native = _agreement("示例银行股份有限公司", field_refs=[_cell_ref()])

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [native, _corrected_conflict()],
    )

    assert len(rows) == 1
    assert rows[0]["institution"] is None
    issues = _institution_required_issues(context)
    assert len(issues) == 1
    assert issues[0]["target_record_id"] == rows[0]["credit_line_id"]
    assert issues[0]["source_refs"] == [_cell_ref()]


@pytest.mark.parametrize(
    "case",
    ["broad", "ambiguous_owner", "duplicate_slots", "wrong_owner", "wrong_anchor"],
)
def test_agreement_conflict_never_promotes_unproved_field_slot(case: str) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    native = _agreement("示例银行股份有限公司", field_refs=[_cell_ref()])
    if case == "broad":
        native["source_refs_by_field"]["institution"][0]["geometry_scope"] = "table"
    elif case == "ambiguous_owner":
        native["source_refs"].append(_table_ref(table_id="agreement-10"))
    elif case == "duplicate_slots":
        native["source_refs_by_field"]["institution"].append(_cell_ref(row=1))
    elif case == "wrong_owner":
        native["source_refs_by_field"]["institution"] = [_cell_ref(table_id="agreement-10")]
    elif case == "wrong_anchor":
        native["_canonical_card_anchor_refs"][0]["bbox"] = [56.0, 300.0, 88.0, 311.0]
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [deepcopy(native), _corrected_conflict()],
    )

    assert len(rows) == 1
    assert rows[0]["institution"] is None
    assert _institution_required_issues(context) == []


def test_agreement_conflict_never_chooses_between_two_exact_owned_slots() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    first = _agreement("示例银行股份有限公司", field_refs=[_cell_ref()])
    second = _agreement(
        "另一银行股份有限公司",
        field_refs=[_cell_ref(table_id="agreement-duplicate")],
        source_refs=[_table_ref(table_id="agreement-duplicate")],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(context, [first, second])

    assert len(rows) == 1
    assert rows[0]["institution"] is None
    assert _institution_required_issues(context) == []
