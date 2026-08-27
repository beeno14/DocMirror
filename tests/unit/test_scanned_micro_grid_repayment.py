# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

import docmirror.plugins.credit_report.repayment_grid as repayment_mod
from docmirror.models.entities.parse_result import DocumentEntities, ParseResult
from docmirror.models.mirror.page_evidence_bundles import (
    domain_specific_with_page_bundles,
    merge_micro_grid_structures_into_bundles,
    micro_grid_structures_from_bundles,
    page_evidence_bundle,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.ocr.micro_grid.cell_recognition import normalize_allowlist_text
from docmirror.ocr.micro_grid.detect import detect_micro_grid_candidates
from docmirror.ocr.micro_grid.models import OCRToken
from docmirror.output.mirror_projector import project_mirror
from docmirror.plugins._base.kv_community_enrich import enrich_credit_report_output
from docmirror.plugins.credit_report.micro_grid_materialize import (
    augment_credit_repayment_evidence_bundles,
    materialize_credit_repayment_micro_grids_from_bundles,
)
from docmirror.plugins.credit_report.repayment_grid import (
    dedupe_repayment_records,
    extract_credit_repayment_records,
    records_from_micro_grid_dict,
)
from docmirror.server.edition_outputs import build_all_projections, write_outputs


def _micro_grid_bundle_domain(
    *,
    page: int = 4,
    page_width: int = 834,
    page_height: int = 1207,
    lines=None,
    tokens=None,
    **extra,
):
    lines = lines if lines is not None else _credit_page4_lines()
    tokens = tokens if tokens is not None else [token.to_dict() for token in _credit_page4_tokens()]
    ds = domain_specific_with_page_bundles(
        page_evidence_bundle(
            page,
            page_width=page_width,
            page_height=page_height,
            micro_grid_evidence={
                "page": page,
                "page_width": page_width,
                "page_height": page_height,
                "lines": lines,
                "tokens": tokens,
            },
        ),
        **extra,
    )
    materialize_credit_repayment_micro_grids_from_bundles(ds)
    return ds


def _credit_page4_lines():
    return [
        {
            "content": "2020年09月-2021年02月的还款记录",
            "bbox": [280.46, 194.67, 510.65, 217.78],
            "confidence": 1.0,
        },
        {
            "content": "1 122689 113.45710",
            "bbox": [130.84, 222.65, 733.57, 241.51],
            "confidence": 1.0,
        },
        {
            "content": "CN.",
            "bbox": [136.90, 249.42, 206.56, 267.06],
            "confidence": 1.0,
        },
        {
            "content": "2021",
            "bbox": [75.71, 262.80, 112.67, 280.44],
            "confidence": 1.0,
        },
        {
            "content": "NN N N",
            "bbox": [559.11, 302.34, 731.75, 319.38],
            "confidence": 1.0,
        },
        {
            "content": "2020",
            "bbox": [75.11, 315.12, 109.64, 332.76],
            "confidence": 1.0,
        },
        {
            "content": "000 0",
            "bbox": [561.53, 327.89, 729.93, 345.54],
            "confidence": 1.0,
        },
    ]


def _credit_page4_tokens():
    tokens = []
    for idx, line in enumerate(_credit_page4_lines()):
        x0, y0, x1, y1 = line["bbox"]
        tokens.append(
            OCRToken(
                token_id=f"ocr_p4_t{idx}",
                text=line["content"],
                bbox=(x0, y0, x1, y1),
                confidence=line["confidence"],
                page=4,
                source="rapidocr_test",
                raw_bbox=(x0 * 2, y0 * 2, x1 * 2, y1 * 2),
            )
        )
    return tokens


def _record_tuples(records):
    return [(r["year"], r["month"], r["status"], r["overdue_amount"]) for r in records]


def _expected_repayment_tuples():
    return [
        (2021, 1, "N", None),
        (2021, 2, "C", None),
        (2020, 9, "N", "0"),
        (2020, 10, "N", "0"),
        (2020, 11, "N", "0"),
        (2020, 12, "N", "0"),
    ]


def _micro_grid_structure_from_document(document: dict, *, page: int = 4) -> dict:
    from docmirror.models.mirror.page_access import micro_grid_structures_from_document

    for grid in micro_grid_structures_from_document(document):
        if int(grid.get("page") or 0) == page:
            return grid
    return {}


def test_records_from_micro_grid_dict_matches_line_extraction():
    out = extract_credit_repayment_records(_credit_page4_lines(), page=4, tokens=_credit_page4_tokens())
    projected = records_from_micro_grid_dict(out["micro_grid"])
    assert _record_tuples(projected) == _record_tuples(out["repayment_records"])
    assert all(
        "field_geometry_exact" not in (cell.get("recognition_audit") or {})
        for row in out["micro_grid"]["cells"]
        for cell in row
    )


def test_credit_enrich_from_micro_grids_only_without_scanned_evidence():
    out = extract_credit_repayment_records(_credit_page4_lines(), page=4, tokens=_credit_page4_tokens())
    ds: dict = {}
    merge_micro_grid_structures_into_bundles(ds, [out["micro_grid"]])
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=ds,
        )
    )
    enriched = enrich_credit_report_output({"data": {}}, parse_result=pr)
    assert _record_tuples(enriched["data"]["repayment_records"]) == _expected_repayment_tuples()


def test_credit_enrich_skips_smg_rebuild_when_structure_exists(monkeypatch):
    out = extract_credit_repayment_records(_credit_page4_lines(), page=4, tokens=_credit_page4_tokens())
    ds = _micro_grid_bundle_domain()
    merge_micro_grid_structures_into_bundles(ds, [out["micro_grid"]])
    calls: list[int] = []

    def _forbidden_rebuild(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("SMG rebuild should not run when micro_grid_structures exist")

    monkeypatch.setattr(repayment_mod, "reconstruct_repayment_micro_grid_from_lines", _forbidden_rebuild)
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=ds,
        )
    )
    enriched = enrich_credit_report_output({"data": {}}, parse_result=pr)
    assert calls == []
    assert _record_tuples(enriched["data"]["repayment_records"]) == _expected_repayment_tuples()


def test_credit_repayment_micro_grid_from_line_bboxes():
    out = extract_credit_repayment_records(_credit_page4_lines(), page=4)

    assert out["micro_grid"]
    assert len(out["micro_grid"]["col_bands"]) == 13
    assert out["micro_grid"]["col_bands"][0]["role"] == "year"
    year_cells = [cell for row in out["micro_grid"]["cells"] for cell in row if cell.get("role") == "year"]
    assert [cell["text"] for cell in year_cells] == ["2021", "2020"]
    assert _record_tuples(out["repayment_records"]) == _expected_repayment_tuples()
    assert all(record["source_cell_refs"] for record in out["repayment_records"])


def test_opted_in_hash_status_is_a_canonical_business_value_without_month_shift():
    lines = [
        {
            "content": "2023\u5e7401\u6708-2023\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            "bbox": [280, 195, 510, 218],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130, 223, 734, 242],
            "confidence": 1.0,
        },
        {"content": "NN###NNNNNNN", "bbox": [137, 249, 730, 267], "confidence": 1.0},
        {"content": "2023", "bbox": [76, 263, 113, 280], "confidence": 1.0},
        {
            "content": "0 0 0 0 0 0 0 0 0 0 0 0",
            "bbox": [137, 303, 730, 320],
            "confidence": 1.0,
        },
    ]

    out = extract_credit_repayment_records(
        lines,
        page=4,
        extra_status_chars={"#"},
    )
    records = records_from_micro_grid_dict(out["micro_grid"])

    assert [(record["month"], record["status"]) for record in records] == [
        (1, "N"),
        (2, "N"),
        (3, "#"),
        (4, "#"),
        (5, "#"),
        (6, "N"),
        (7, "N"),
        (8, "N"),
        (9, "N"),
        (10, "N"),
        (11, "N"),
        (12, "N"),
    ]
    assert all(record.get("extraction_status") != "review" for record in records)
    hash_cells = [
        cell
        for row in out["micro_grid"]["cells"]
        for cell in row
        if cell.get("role") == "status" and cell.get("text") == "#"
    ]
    assert len(hash_cells) == 3
    assert {cell.get("recognition_source") for cell in hash_cells} == {"canonical_row_sequence"}


def test_unapproved_hash_remains_alignment_only_and_is_not_an_unknown_business_value():
    lines = [
        {
            "content": "2023\u5e7401\u6708-2023\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            "bbox": [280, 195, 510, 218],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130, 223, 734, 242],
            "confidence": 1.0,
        },
        {"content": "NN###NNNNNNN", "bbox": [137, 249, 730, 267], "confidence": 1.0},
        {"content": "2023", "bbox": [76, 263, 113, 280], "confidence": 1.0},
        {
            "content": "0 0 0 0 0 0 0 0 0 0 0 0",
            "bbox": [137, 303, 730, 320],
            "confidence": 1.0,
        },
    ]

    out = extract_credit_repayment_records(lines, page=4)
    records = records_from_micro_grid_dict(out["micro_grid"])

    assert all(record["status"] != "#" for record in records)
    assert {record["month"] for record in records if record["status"] == "unknown"} == {3, 4, 5}


def test_credit_repayment_micro_grid_prefers_ocr_tokens_when_available():
    out = extract_credit_repayment_records(_credit_page4_lines(), page=4, tokens=_credit_page4_tokens())

    assert out["micro_grid"]["geometry_source"].startswith("ocr_tokens")
    assert out["micro_grid"]["audit"]["source_token_count"] == len(_credit_page4_tokens())
    year_cells = [cell for row in out["micro_grid"]["cells"] for cell in row if cell.get("role") == "year"]
    assert year_cells[0]["text"] == "2021"
    assert year_cells[0]["bbox"]
    assert _record_tuples(out["repayment_records"]) == _expected_repayment_tuples()


def test_zero_status_preserves_contradictory_nonzero_amount_evidence_for_review():
    lines = [dict(line) for line in _credit_page4_lines()[:-1]]
    # This is a common OCR merge: the calendar-year label and four amount
    # cells arrive as one line. The final nonzero amount must not be replaced
    # by the semantic zero implied by status N.
    lines[-1] = {
        "content": "2020 0005",
        "bbox": [75.0, 315.12, 733.57, 332.76],
        "confidence": 1.0,
    }

    out = extract_credit_repayment_records(lines, page=4)
    direct = next(record for record in out["repayment_records"] if record["year"] == 2020 and record["month"] == 12)
    projected = next(
        record
        for record in records_from_micro_grid_dict(out["micro_grid"])
        if record["year"] == 2020 and record["month"] == 12
    )

    for record in (direct, projected):
        assert record["status"] == "N"
        assert record["overdue_amount"] == "5"
        assert record["extraction_status"] == "review"
        assert record["audit"]["reason"] == "zero_status_conflicts_with_observed_nonzero_amount"


def test_grid_dedupe_preserves_conflicting_months_for_relationship_reporting():
    records = [
        {
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "confidence": 0.8,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "a"}],
        },
        {
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "1",
            "overdue_amount": "100",
            "confidence": 0.9,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "b"}],
        },
    ]

    assert dedupe_repayment_records(records) == records


def test_grid_dedupe_silently_merges_normalized_identical_months():
    records = [
        {
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0.00",
            "confidence": 0.8,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "a"}],
        },
        {
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "confidence": 0.9,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "b"}],
        },
    ]

    deduped = dedupe_repayment_records(records)

    assert len(deduped) == 1
    assert deduped[0]["confidence"] == 0.9
    assert deduped[0]["source_cell_refs"] == [
        {"grid_id": "grid:1", "cell": "a"},
        {"grid_id": "grid:1", "cell": "b"},
    ]
    assert "audit" not in deduped[0]


def test_micro_grid_candidate_detector_is_anchor_gated():
    candidates = detect_micro_grid_candidates(_credit_page4_tokens(), lines=_credit_page4_lines(), page=4)
    assert candidates
    assert "anchor_temporal_record" in candidates[0].reason_codes

    negative = extract_credit_repayment_records(
        [
            {"content": "个人消费贷款", "bbox": [80, 120, 200, 140]},
            {"content": "NN N N", "bbox": [300, 180, 520, 200]},
            {"content": "000 0", "bbox": [300, 210, 520, 230]},
        ],
        page=4,
    )
    assert negative["micro_grid"] is None
    assert negative["repayment_records"] == []


def test_allowlist_normalization_filters_ocr_noise():
    assert normalize_allowlist_text("ＣN.O〇x", {"C", "N", "0"}, max_chars=4) == "N000"
    assert normalize_allowlist_text("O,OOO.50元", set("0123456789.,"), max_chars=16) == "0,000.50"


def test_hash_is_uncertainty_not_a_credit_status():
    assert "#" not in repayment_mod._STATUS_CHARS
    assert "/" in repayment_mod._STATUS_CHARS
    # R2-only status A is enabled by the personal-detail caller; the shared
    # extractor default remains unchanged for other credit-report variants.
    assert "A" not in repayment_mod._STATUS_CHARS
    assert "A" not in repayment_mod._ZERO_OVERDUE_STATUSES


def test_unreadable_boundary_cell_uses_two_matching_neighbors_only():
    statuses = {7: "#", 8: "N", 9: "N", 10: "N", 11: "*", 12: "*"}

    assert repayment_mod._neighbor_status_fallback(statuses, 7) == "N"
    assert repayment_mod._neighbor_status_fallback({9: "*", 10: "#", 11: "N"}, 10) == ""


def test_visually_confirmed_numeric_status_is_not_quarantined():
    grid = {
        "grid_id": "mg_p1_repayment_0",
        "page": 1,
        "anchor_text": "2024年01月-2024年01月的还款记录",
        "audit": {"date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 1}},
        "col_bands": [{"index": 1, "header": "1"}],
        "cells": [
            [
                {"row_index": 2, "col_index": 0, "text": "2024", "role": "year"},
                {
                    "row_index": 2,
                    "col_index": 1,
                    "text": "2",
                    "role": "status",
                    "recognition_source": "cell_crop_consensus",
                    "recognition_audit": {"consensus_count": 2},
                },
            ],
            [
                {
                    "row_index": 3,
                    "col_index": 1,
                    "text": "100",
                    "role": "overdue_amount",
                }
            ],
        ],
    }

    records = records_from_micro_grid_dict(grid)

    assert records[0]["status"] == "2"
    assert records[0]["recognition_source"] == "cell_crop_consensus"


def test_candidate_b_exact_row_numeric_status_without_positive_amount_is_rejected():
    status_ref = {
        "page": 3,
        "logical_page": 3,
        "bbox": [100, 200, 120, 220],
        "geometry_scope": "cell",
        "field_name": "status",
    }
    amount_ref = {
        "page": 3,
        "logical_page": 3,
        "bbox": [100, 220, 120, 240],
        "geometry_scope": "cell",
        "field_name": "overdue_amount",
    }
    grid = {
        "grid_id": "mg_p3_repayment_0",
        "page": 3,
        "anchor_text": "2024年01月-2024年01月的还款记录",
        "audit": {
            "date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 1},
            "zero_overdue_statuses": ["*", "/", "N", "C", "A"],
        },
        "col_bands": [{"index": 1, "header": "1"}],
        "cells": [
            [
                {"row_index": 2, "col_index": 0, "text": "2024", "role": "year"},
                {
                    "row_index": 2,
                    "col_index": 1,
                    "bbox": [100, 200, 120, 220],
                    "text": "2",
                    "role": "status",
                    "recognition_source": "canonical_row_sequence",
                    "recognition_audit": {
                        "alignment_status": "exact",
                        "expected_cell_count": 1,
                        "observed_status_count": 1,
                        "source_ref": status_ref,
                    },
                },
            ],
            [
                {
                    "row_index": 3,
                    "col_index": 1,
                    "bbox": [100, 220, 120, 240],
                    "text": "",
                    "role": "overdue_amount",
                    "recognition_audit": {"source_ref": amount_ref},
                }
            ],
        ],
    }

    shared_default = records_from_micro_grid_dict(grid)
    candidate_b = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=True)

    assert shared_default[0]["status"] == "unknown"
    assert candidate_b[0]["status"] == "unknown"
    assert candidate_b[0]["repayment_id"] == "mg_p3_repayment_0:2024-01"
    assert candidate_b[0]["grid_id"] == "mg_p3_repayment_0"
    assert candidate_b[0]["overdue_amount"] is None
    assert candidate_b[0]["source_cell_refs"] == [status_ref, amount_ref]
    assert candidate_b[0]["audit"]["field_name"] == "status_code"
    assert candidate_b[0]["audit"]["unresolved_fields"] == [
        "status_code",
        "overdue_amount",
    ]


def _candidate_b_row_role_grid(
    status: str,
    amount: str,
    *,
    amount_row_index: int = 3,
    duplicate_amount_cell: bool = False,
    duplicate_status_row: bool = False,
) -> dict:
    status_cell = {
        "row_index": 2,
        "col_index": 1,
        "bbox": [100, 200, 120, 220],
        "text": status,
        "role": "status",
        "recognition_source": "canonical_row_sequence",
        "recognition_audit": {
            "alignment_status": "exact",
            "expected_cell_count": 1,
            "observed_status_count": 1,
        },
    }
    amount_cell = {
        "row_index": amount_row_index,
        "col_index": 1,
        "bbox": [100, 220, 120, 240],
        "text": amount,
        "role": "overdue_amount",
    }
    cells = [
        [
            {"row_index": 2, "col_index": 0, "text": "2024", "role": "year"},
            status_cell,
        ],
        [amount_cell, *([dict(amount_cell)] if duplicate_amount_cell else [])],
    ]
    if duplicate_status_row:
        cells.append(
            [
                {"row_index": 4, "col_index": 0, "text": "2024", "role": "year"},
                {**status_cell, "row_index": 4},
            ]
        )
    return {
        "grid_id": "mg_p3_repayment_row_role",
        "page": 3,
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 1,
            },
            "zero_overdue_statuses": ["*", "/", "N", "C", "#"],
        },
        "col_bands": [{"index": 1, "header": "1"}],
        "cells": cells,
    }


def test_candidate_b_numeric_status_requires_positive_same_column_amount() -> None:
    rejected = records_from_micro_grid_dict(
        _candidate_b_row_role_grid("2", "0"),
        accept_exact_row_numeric_status=True,
    )[0]
    accepted = records_from_micro_grid_dict(
        _candidate_b_row_role_grid("2", "150"),
        accept_exact_row_numeric_status=True,
    )[0]

    assert rejected["status"] == "unknown"
    assert rejected["overdue_amount"] == "0"
    assert rejected["audit"]["field_name"] == "status_code"
    assert rejected["audit"]["unresolved_fields"] == ["status_code"]
    assert accepted["status"] == "2"
    assert accepted["overdue_amount"] == "150"


def _candidate_b_two_month_amount_lines(
    statuses: tuple[str, str],
    *amount_lines: dict,
) -> list[dict]:
    return [
        {
            "content": "2024年01月-2024年02月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": statuses[0], "bbox": [145.0, 249.0, 165.0, 274.0], "confidence": 1.0},
        {"content": statuses[1], "bbox": [195.0, 249.0, 215.0, 274.0], "confidence": 1.0},
        {"content": "2024", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
        *amount_lines,
    ]


def test_candidate_b_does_not_split_one_undelimited_amount_across_numeric_status_cells() -> None:
    lines = _candidate_b_two_month_amount_lines(
        ("1", "2"),
        # One undelimited number spans two month columns. It may be a single
        # amount of 12 with the adjacent cell blank; its digits are not two
        # independently observed positive amount cells.
        {
            "content": "12",
            "bbox": [142.0, 270.0, 218.0, 288.0],
            "confidence": 1.0,
        },
    )
    extracted = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        enable_candidate_b_amount_pairing=True,
    )

    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [(row["status"], row["overdue_amount"]) for row in rows] == [
        ("unknown", None),
        ("unknown", None),
    ]
    assert all(row["extraction_status"] == "review" for row in rows)
    assert all(row["_amount_pairing"]["status"] == "duplicate_or_ambiguous_cell" for row in rows)


def test_candidate_b_splits_all_zero_run_only_across_unique_consecutive_month_bands() -> None:
    lines = _candidate_b_two_month_amount_lines(
        ("C", "C"),
        {
            "content": "00",
            "bbox": [142.0, 270.0, 218.0, 288.0],
            "confidence": 1.0,
        },
    )
    extracted = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        enable_candidate_b_amount_pairing=True,
    )

    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [(row["status"], row["overdue_amount"]) for row in rows] == [
        ("C", "0"),
        ("C", "0"),
    ]
    assert all(row.get("extraction_status") != "review" for row in rows)


def test_candidate_b_duplicate_zero_runs_do_not_create_exact_amount_cells() -> None:
    lines = _candidate_b_two_month_amount_lines(
        ("1", "2"),
        {
            "content": "00",
            "bbox": [142.0, 270.0, 218.0, 288.0],
            "confidence": 1.0,
        },
        {
            "content": "00",
            "bbox": [142.0, 271.0, 218.0, 289.0],
            "confidence": 1.0,
        },
    )
    extracted = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        enable_candidate_b_amount_pairing=True,
    )

    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [(row["status"], row["overdue_amount"]) for row in rows] == [
        ("unknown", None),
        ("unknown", None),
    ]
    assert all(row["extraction_status"] == "review" for row in rows)


def _candidate_b_direct_amount_pair(
    *amount_lines: dict,
    excluded_line_indices: set[int] | None = None,
) -> dict:
    month_cols = [
        {"header": "1", "bbox": [130.0, 222.0, 180.0, 241.0]},
        {"header": "2", "bbox": [180.0, 222.0, 230.0, 241.0]},
    ]
    year_line = {
        "idx": 10,
        "text": "2024",
        "bbox": [75.0, 270.0, 112.0, 288.0],
        "confidence": 1.0,
        "source_logical_page": 4,
    }
    status_line = {
        "idx": 20,
        "text": "CC",
        "bbox": [145.0, 249.0, 215.0, 274.0],
        "confidence": 1.0,
        "source_logical_page": 4,
        "status_source_line_indices": [20],
    }
    return repayment_mod._candidate_b_amount_row_pair(
        [year_line, *amount_lines],
        year_line,
        month_cols=month_cols,
        active_months=[1, 2],
        page=4,
        excluded_line_indices=excluded_line_indices or {20},
        year_lines=[year_line],
        status_line=status_line,
    )


def _candidate_b_direct_amount_line(
    index: int,
    text: str,
    bbox: list[float],
) -> dict:
    return {
        "idx": index,
        "text": text,
        "bbox": bbox,
        "confidence": 1.0,
        "source_logical_page": 4,
    }


def test_candidate_b_unique_fragment_cover_recovers_one_printed_amount_band() -> None:
    pairing = _candidate_b_direct_amount_pair(
        _candidate_b_direct_amount_line(11, "0", [148.0, 277.0, 162.0, 281.0]),
        # A different glyph height and baseline forms a second raw y-cluster,
        # but its x geometry uniquely owns the adjacent month.
        _candidate_b_direct_amount_line(12, "25", [198.0, 283.0, 212.0, 289.0]),
    )

    assert pairing["status"] == "exact"
    assert pairing["cell_status_by_month"] == {"1": "exact", "2": "exact"}
    assert [token.text for token in pairing["tokens"]] == ["0", "25"]
    assert pairing["source_line_indices"] == [11, 12]


def test_candidate_b_partial_month_geometry_keeps_owned_amount_field_local() -> None:
    """A proved cell must not make an unproved sibling addressable."""

    year_line = {
        "idx": 10,
        "text": "2024",
        "bbox": [75.0, 270.0, 112.0, 288.0],
        "confidence": 1.0,
        "source_logical_page": 4,
    }
    amount_line = _candidate_b_direct_amount_line(
        11,
        "25",
        [198.0, 277.0, 212.0, 289.0],
    )
    pairing = repayment_mod._candidate_b_amount_row_pair(
        [year_line, amount_line],
        year_line,
        month_cols=[{"header": "2", "bbox": [180.0, 222.0, 230.0, 241.0]}],
        active_months=[1, 2],
        page=4,
        excluded_line_indices=set(),
        year_lines=[year_line],
    )

    assert pairing["status"] == "exact"
    assert pairing["cell_status_by_month"] == {
        "1": "month_geometry_unowned",
        "2": "exact",
    }
    assert pairing["unowned_geometry_months"] == [1]
    assert [token.text for token in pairing["tokens"]] == ["25"]


@pytest.mark.parametrize(
    "amount_lines",
    [
        # Two OCR fragments claim the same month.
        [
            _candidate_b_direct_amount_line(11, "0", [146.0, 277.0, 158.0, 281.0]),
            _candidate_b_direct_amount_line(12, "5", [152.0, 283.0, 164.0, 289.0]),
        ],
        # A full-row sequence competes with otherwise valid partial fragments.
        [
            _candidate_b_direct_amount_line(11, "0", [148.0, 277.0, 162.0, 281.0]),
            _candidate_b_direct_amount_line(12, "25", [198.0, 283.0, 212.0, 289.0]),
            _candidate_b_direct_amount_line(13, "0 25", [142.0, 285.0, 218.0, 291.0]),
        ],
        # Distinct vertical bands cannot be joined merely because x is disjoint.
        [
            _candidate_b_direct_amount_line(11, "0", [148.0, 277.0, 162.0, 281.0]),
            _candidate_b_direct_amount_line(12, "25", [198.0, 298.0, 212.0, 304.0]),
        ],
        # A narrow fragment centred exactly on a column boundary has two owners.
        [
            _candidate_b_direct_amount_line(11, "0", [173.0, 277.0, 187.0, 281.0]),
            _candidate_b_direct_amount_line(12, "25", [198.0, 283.0, 212.0, 289.0]),
        ],
    ],
)
def test_candidate_b_unique_fragment_cover_rejects_nonunique_geometry(
    amount_lines: list[dict],
) -> None:
    pairing = _candidate_b_direct_amount_pair(*amount_lines)

    assert pairing["status"] == "ambiguous_immediate_rows"
    assert pairing["tokens"] == []
    assert pairing["cell_status_by_month"] == {
        "1": "ambiguous_immediate_rows",
        "2": "ambiguous_immediate_rows",
    }


def test_candidate_b_unique_fragment_cover_quarantines_role_known_numeric_lines() -> None:
    pairing = _candidate_b_direct_amount_pair(
        _candidate_b_direct_amount_line(11, "0", [148.0, 277.0, 162.0, 281.0]),
        _candidate_b_direct_amount_line(12, "25", [198.0, 283.0, 212.0, 289.0]),
        # A year-like OCR token overlaps the amount slot but is not business data.
        _candidate_b_direct_amount_line(13, "2023", [148.0, 284.0, 180.0, 290.0]),
        # Numeric status/header evidence is excluded by its resolved source id.
        _candidate_b_direct_amount_line(30, "7", [148.0, 285.0, 162.0, 291.0]),
        excluded_line_indices={20, 30},
    )

    assert pairing["status"] == "exact"
    assert pairing["observed_texts"] == ["0", "25"]
    assert [token.text for token in pairing["tokens"]] == ["0", "25"]


def test_candidate_b_unique_fragment_cover_keeps_uncovered_month_null() -> None:
    pairing = _candidate_b_direct_amount_pair(
        _candidate_b_direct_amount_line(11, "7", [148.0, 277.0, 162.0, 281.0]),
        # This creates a second raw cluster but is explicitly a year token, not
        # evidence that the blank second amount cell contains zero.
        _candidate_b_direct_amount_line(13, "2023", [148.0, 284.0, 180.0, 290.0]),
    )

    assert pairing["status"] == "exact"
    assert pairing["cell_status_by_month"] == {
        "1": "exact",
        "2": "blank_amount_cell",
    }
    assert [token.text for token in pairing["tokens"]] == ["7"]


@pytest.mark.parametrize("status", ["N", "*", "C", "#", "/", "A", "B", "M", "D", "Z", "G"])
def test_candidate_b_clean_symbolic_status_preserves_explicit_paired_zero(status: str) -> None:
    record = records_from_micro_grid_dict(
        _candidate_b_row_role_grid(status, "0.00"),
        accept_exact_row_numeric_status=True,
    )[0]

    assert record["status"] == status
    assert record["overdue_amount"] == "0"
    assert record.get("extraction_status") != "review"


def test_candidate_b_blank_paired_amount_is_not_inferred_as_zero() -> None:
    record = records_from_micro_grid_dict(
        _candidate_b_row_role_grid("C", ""),
        accept_exact_row_numeric_status=True,
    )[0]

    assert record["status"] == "C"
    assert record["overdue_amount"] is None


def test_candidate_b_n_star_retains_reported_candidate_without_static_corroboration() -> None:
    unresolved_grid = _candidate_b_row_role_grid("N", "0")
    unresolved_grid["audit"]["static_status_validation"] = {"enabled": True}
    corroborated_grid = _candidate_b_row_role_grid("N", "0")
    corroborated_grid["audit"]["static_status_validation"] = {"enabled": True}
    corroborated_grid["cells"][0][1]["recognition_source"] = "static_grid_template_consensus"

    unresolved = records_from_micro_grid_dict(
        unresolved_grid,
        accept_exact_row_numeric_status=True,
    )[0]
    corroborated = records_from_micro_grid_dict(
        corroborated_grid,
        accept_exact_row_numeric_status=True,
    )[0]

    assert unresolved["status"] == "N"
    assert unresolved["extraction_status"] == "review"
    assert unresolved["audit"]["field_name"] == "status_code"
    assert unresolved["audit"]["reported_value_retained"] is True
    assert corroborated["status"] == "N"
    assert corroborated["overdue_amount"] == "0"


@pytest.mark.parametrize(
    ("grid", "reason"),
    [
        (
            _candidate_b_row_role_grid("2", "200", amount_row_index=4),
            "numeric_status_row_role_or_month_geometry_unresolved",
        ),
        (
            _candidate_b_row_role_grid("2", "200", duplicate_amount_cell=True),
            "numeric_status_row_role_or_month_geometry_unresolved",
        ),
        (
            _candidate_b_row_role_grid("2", "200", duplicate_status_row=True),
            "numeric_status_row_role_or_month_geometry_unresolved",
        ),
    ],
)
def test_candidate_b_numeric_status_rejects_missing_or_ambiguous_row_role(
    grid: dict,
    reason: str,
) -> None:
    record = records_from_micro_grid_dict(
        grid,
        accept_exact_row_numeric_status=True,
    )[0]

    assert record["status"] == "unknown"
    assert record["overdue_amount"] is None
    assert record["audit"]["reason"] == reason
    assert record["audit"]["field_name"] == "status_code"
    assert record["audit"]["unresolved_fields"] == [
        "status_code",
        "overdue_amount",
    ]


def _candidate_b_boundary_lines() -> list[dict]:
    return [
        {
            "content": "2019年12月-2020年02月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        # The OCR split Jan/Feb into separate words and misread printed C as N.
        {"content": "N", "bbox": [145.0, 249.0, 165.0, 274.0], "confidence": 1.0},
        {"content": "N", "bbox": [195.0, 249.0, 215.0, 274.0], "confidence": 1.0},
        {"content": "2020", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
        {"content": "0", "bbox": [148.0, 270.0, 162.0, 288.0], "confidence": 1.0},
        {"content": "0", "bbox": [198.0, 270.0, 212.0, 288.0], "confidence": 1.0},
        {"content": "*", "bbox": [695.0, 302.0, 715.0, 327.0], "confidence": 1.0},
        {"content": "2019", "bbox": [75.0, 323.0, 112.0, 341.0], "confidence": 1.0},
        {"content": "0", "bbox": [698.0, 323.0, 712.0, 341.0], "confidence": 1.0},
    ]


def test_candidate_b_mixed_year_boundary_keeps_month_cells_and_row_bands_distinct() -> None:
    import cv2
    import numpy as np

    image = np.full((600, 834, 3), 255, dtype=np.uint8)

    def draw_glyph(glyph: str, center_x: float, center_y: float) -> None:
        (width, height), _baseline = cv2.getTextSize(
            glyph,
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            1,
        )
        cv2.putText(
            image,
            glyph,
            (round(center_x - width / 2), round(center_y + height / 2)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    draw_glyph("N", 155.0, 259.0)
    draw_glyph("C", 205.0, 259.0)
    draw_glyph("*", 705.0, 312.0)
    out = extract_credit_repayment_records(
        _candidate_b_boundary_lines(),
        page=4,
        page_width=834,
        page_height=600,
        page_image=image,
        enable_static_status_validation=True,
        enable_candidate_b_amount_pairing=True,
    )
    rows = records_from_micro_grid_dict(
        out["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    by_month = {(row["year"], row["month"]): row for row in rows}

    assert set(by_month) == {(2019, 12), (2020, 1), (2020, 2)}
    assert by_month[(2019, 12)]["status"] == "*"
    assert by_month[(2020, 1)]["status"] == "N"
    assert by_month[(2020, 2)]["status"] == "C"
    assert all(row["overdue_amount"] == "0" for row in by_month.values())
    assert by_month[(2020, 2)]["recognition_source"] == "static_glyph_shape_validation"
    assert out["micro_grid"]["audit"]["static_status_validation"]["corrections"] == 1

    for row in by_month.values():
        refs = {ref.get("field_name"): ref for ref in row["source_cell_refs"]}
        assert refs["status"]["bbox"][3] <= refs["overdue_amount"]["bbox"][1]
    assert by_month[(2020, 1)]["source_cell_refs"][0]["col"] == 1
    assert by_month[(2020, 2)]["source_cell_refs"][0]["col"] == 2
    assert by_month[(2019, 12)]["source_cell_refs"][0]["col"] == 12


def _candidate_b_boundary_page_with_visual_amount(
    *,
    target: tuple[int, int],
    observed_amount: str,
    visual_amount: str,
) -> tuple[list[dict], object]:
    """Replay one real-report failure mode: watermark noise changes amount OCR."""

    import cv2
    import numpy as np

    lines = deepcopy(_candidate_b_boundary_lines())
    target_bbox = {
        (2020, 1): [148.0, 270.0, 162.0, 288.0],
        (2019, 12): [698.0, 323.0, 712.0, 341.0],
    }[target]
    target_line = next(line for line in lines if line.get("bbox") == target_bbox)
    target_line["content"] = observed_amount

    image = np.full((600, 834, 3), 255, dtype=np.uint8)

    def draw_glyph(glyph: str, center_x: float, center_y: float) -> None:
        (width, height), _baseline = cv2.getTextSize(
            glyph,
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            1,
        )
        cv2.putText(
            image,
            glyph,
            (round(center_x - width / 2), round(center_y + height / 2)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # These are the three printed status cells in the business excerpt.  The
    # amount glyph is drawn independently from its OCR token so this exercises
    # deterministic visual validation, not a second OCR invocation.
    draw_glyph("N", 155.0, 259.0)
    draw_glyph("C", 205.0, 259.0)
    draw_glyph("*", 705.0, 312.0)
    amount_center = (155.0, 279.0) if target == (2020, 1) else (705.0, 332.0)
    draw_glyph(visual_amount, *amount_center)
    return lines, image


@pytest.mark.parametrize(
    ("target", "observed_amount", "expected_audit_text"),
    [
        ((2020, 1), "o", ""),
        ((2019, 12), "10", "10"),
        ((2019, 12), "20", "20"),
    ],
)
def test_candidate_b_static_amount_zero_repairs_one_glyph_ocr_noise_without_reocr(
    monkeypatch,
    target: tuple[int, int],
    observed_amount: str,
    expected_audit_text: str,
) -> None:
    lines, image = _candidate_b_boundary_page_with_visual_amount(
        target=target,
        observed_amount=observed_amount,
        visual_amount="0",
    )

    def forbidden_ocr(*_args, **_kwargs):
        raise AssertionError("static amount validation must not invoke OCR")

    monkeypatch.setattr(repayment_mod, "recognize_micro_cell_from_image", forbidden_ocr)
    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        page_image=image,
        enable_cell_ocr=False,
        enable_static_status_validation=True,
        enable_candidate_b_amount_pairing=True,
    )
    record = next(
        row
        for row in records_from_micro_grid_dict(
            out["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if (row["year"], row["month"]) == target
    )

    assert record["overdue_amount"] == "0"
    amount_cell = next(
        cell
        for row in out["micro_grid"]["cells"]
        for cell in row
        if cell.get("role") == "overdue_amount" and cell.get("col_index") == target[1] and cell.get("text") == "0"
    )
    assert amount_cell["recognition_source"] == "static_amount_zero_glyph_validation"
    assert amount_cell["recognition_audit"]["observed_amount_text"] == expected_audit_text
    assert out["micro_grid"]["audit"]["static_amount_zero_validation"]["corrections"] >= 1
    assert out["micro_grid"]["audit"]["cell_crop_ocr"] == {
        "enabled": False,
        "attempts": 0,
        "hits": 0,
    }


def test_candidate_b_static_amount_zero_uses_exact_source_lattice_when_row_is_noisy(
    monkeypatch,
) -> None:
    """A noisy sibling status must not disable an exact target-cell repair."""

    import cv2
    import numpy as np

    lines = _source_owned_base_lines(year=2022, status_text="#" + "N" * 11)
    lines[-1]["content"] = "0 " * 11 + "10"
    lines[0]["bbox"] = [250.0, 5.0, 500.0, 10.0]
    lines[1]["bbox"] = [70.0, 15.0, 670.0, 20.0]
    lines[2]["bbox"] = [70.0, 30.0, 670.0, 35.0]
    lines[3]["bbox"] = [25.0, 37.0, 60.0, 42.0]
    lines[4]["bbox"] = [70.0, 44.0, 670.0, 49.0]
    image = np.full((100, 700, 3), 255, dtype=np.uint8)

    def draw_glyph(glyph: str, center_x: float, center_y: float) -> None:
        (width, height), _baseline = cv2.getTextSize(
            glyph,
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            1,
        )
        cv2.putText(
            image,
            glyph,
            (round(center_x - width / 2), round(center_y + height / 2)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    draw_glyph("N", 645.0, 34.0)
    draw_glyph("0", 645.0, 52.0)

    def forbidden_ocr(*_args, **_kwargs):
        raise AssertionError("static amount validation must not invoke OCR")

    monkeypatch.setattr(repayment_mod, "recognize_micro_cell_from_image", forbidden_ocr)
    geometry = _continuation_source_table_geometry(
        table_id="pt_13_0",
        logical_page=13,
        year_left=20.0,
        column_pitch=50.0,
        row_edges=(25.0, 43.0, 61.0),
    )
    geometry["bbox"] = [20.0, 25.0, 670.0, 61.0]
    out = extract_credit_repayment_records(
        lines,
        page=13,
        page_width=700,
        page_height=100,
        page_image=image,
        enable_cell_ocr=False,
        enable_static_status_validation=True,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={"13": [geometry]},
    )
    record = next(
        row
        for row in records_from_micro_grid_dict(
            out["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if (row["year"], row["month"]) == (2022, 12)
    )

    assert record["status"] == "N"
    assert record["overdue_amount"] == "0"
    assert record["recognition_source"] == "static_glyph_shape_validation"
    assert out["micro_grid"]["audit"]["static_amount_zero_validation"]["corrections"] == 1


@pytest.mark.parametrize("printed_amount", ["6", "8", "9", "10", "20"])
def test_candidate_b_static_amount_zero_rejects_genuine_printed_nonzero(
    printed_amount: str,
) -> None:
    lines, image = _candidate_b_boundary_page_with_visual_amount(
        target=(2019, 12),
        observed_amount=printed_amount,
        visual_amount=printed_amount,
    )
    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        page_image=image,
        enable_cell_ocr=False,
        enable_static_status_validation=True,
        enable_candidate_b_amount_pairing=True,
    )
    record = next(
        row
        for row in records_from_micro_grid_dict(
            out["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if (row["year"], row["month"]) == (2019, 12)
    )

    assert record["overdue_amount"] == printed_amount
    amount_cell = next(
        cell
        for row in out["micro_grid"]["cells"]
        for cell in row
        if cell.get("role") == "overdue_amount" and cell.get("col_index") == 12 and cell.get("text") == printed_amount
    )
    assert amount_cell["recognition_audit"]["reason"] == ("zero_status_conflicts_with_observed_nonzero_amount")


@pytest.mark.parametrize(
    ("amount_lines", "expected_pair_status"),
    [
        ([], "missing_amount_row"),
        (
            [{"content": "0", "bbox": [148.0, 316.0, 162.0, 334.0], "confidence": 1.0}],
            "non_immediate_amount_row",
        ),
        (
            [
                {"content": "0", "bbox": [146.0, 270.0, 158.0, 288.0], "confidence": 1.0},
                {"content": "0", "bbox": [152.0, 270.0, 164.0, 288.0], "confidence": 1.0},
            ],
            "duplicate_or_ambiguous_cell",
        ),
        (
            [{"content": "0", "bbox": [198.0, 270.0, 212.0, 288.0], "confidence": 1.0}],
            "blank_amount_cell",
        ),
    ],
)
def test_candidate_b_never_infers_zero_from_unresolved_amount_geometry(
    amount_lines: list[dict],
    expected_pair_status: str,
) -> None:
    lines = [
        {
            "content": "2024年01月-2024年01月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": "C", "bbox": [145.0, 249.0, 165.0, 274.0], "confidence": 1.0},
        {"content": "2024", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
        *amount_lines,
    ]
    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        enable_candidate_b_amount_pairing=True,
    )
    record = records_from_micro_grid_dict(
        out["micro_grid"],
        accept_exact_row_numeric_status=True,
    )[0]

    assert record["status"] == "C"
    assert record["overdue_amount"] is None
    assert record["extraction_status"] == "review"
    assert record["_amount_pairing"]["status"] == expected_pair_status


def test_repayment_projection_binds_amount_to_same_year_row():
    def row(row_index, year, status, amount):
        return [
            [
                {"row_index": row_index, "col_index": 0, "text": str(year), "role": "year"},
                {"row_index": row_index, "col_index": 1, "text": status, "role": "status"},
            ],
            [
                {"row_index": row_index + 1, "col_index": 1, "text": amount, "role": "overdue_amount"},
            ],
        ]

    grid = {
        "grid_id": "mg_p1_repayment_0",
        "page": 1,
        "anchor_text": "2023年01月-2024年01月的还款记录",
        "audit": {"date_range": {"start_year": 2023, "start_month": 1, "end_year": 2024, "end_month": 1}},
        "col_bands": [{"index": 1, "header": "1"}],
        "cells": [*row(2, 2024, "N", "0"), *row(4, 2023, "N", "200")],
    }

    records = records_from_micro_grid_dict(grid)
    by_year = {record["year"]: record for record in records if record["month"] == 1}

    assert by_year[2024]["overdue_amount"] == "0"
    assert by_year[2023]["overdue_amount"] == "200"


def test_repayment_mapper_is_credit_plugin_not_core_export():
    import importlib

    import docmirror.ocr.micro_grid as micro_grid

    assert "extract_credit_repayment_records" not in micro_grid.__all__
    assert not hasattr(micro_grid, "extract_credit_repayment_records")
    try:
        importlib.import_module("docmirror.ocr.micro_grid.repayment")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("credit repayment mapper must not live under core.ocr.micro_grid")


def test_cell_crop_ocr_fills_missing_target_cell(monkeypatch):
    lines = [
        {"content": "2020年09月-2021年02月的还款记录", "bbox": [280.46, 194.67, 510.65, 217.78], "confidence": 1.0},
        {"content": "1 122689 113.45710", "bbox": [130.84, 222.65, 733.57, 241.51], "confidence": 1.0},
        {"content": "C", "bbox": [186.90, 249.42, 206.56, 267.06], "confidence": 1.0},
        {"content": "2021", "bbox": [75.71, 262.80, 112.67, 280.44], "confidence": 1.0},
    ]

    class FakeRecognition:
        text = "N"
        confidence = 0.91
        source = "cell_crop_ocr"
        raw_text = "N"
        audit = {"region": (1, 2, 3, 4)}

    def fake_recognize(*args, **kwargs):
        return FakeRecognition()

    class FakeImage:
        shape = (1200, 834, 3)

    monkeypatch.setattr(repayment_mod, "recognize_micro_cell_from_image", fake_recognize)
    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=1207,
        page_image=FakeImage(),
        enable_cell_ocr=True,
    )

    assert (2021, 1, "N") in [(r["year"], r["month"], r["status"]) for r in out["repayment_records"]]
    resolved = next(r for r in out["repayment_records"] if (r["year"], r["month"], r["status"]) == (2021, 1, "N"))
    assert resolved["overdue_amount"] is None
    assert out["micro_grid"]["audit"]["cell_crop_ocr"]["attempts"] >= 1
    assert out["micro_grid"]["audit"]["cell_crop_ocr"]["hits"] >= 1
    status_cells = [
        cell
        for row in out["micro_grid"]["cells"]
        for cell in row
        if cell["role"] == "status" and cell["col_index"] == 1
    ]
    assert status_cells[0]["recognition_source"] == "cell_crop_ocr"


def test_word_level_month_header_recovers_realistic_static_grid_geometry(monkeypatch) -> None:
    import cv2
    import numpy as np

    centres = [92.0 + 27.0 * index for index in range(12)]
    month_words = [
        {
            "content": str(month),
            "bbox": [centres[month - 1] - 4.0, 222.0, centres[month - 1] + 4.0, 238.0],
            "confidence": 1.0,
        }
        for month in range(1, 8)
    ]
    # Lin's saved source structure contains one noisy cell spanning months 8/9.
    month_words.append(
        {
            "content": "供 8 9",
            "bbox": [centres[7] - 13.5, 222.0, centres[8] + 13.5, 238.0],
            "confidence": 1.0,
        }
    )
    month_words.extend(
        {
            "content": str(month),
            "bbox": [centres[month - 1] - 5.0, 222.0, centres[month - 1] + 5.0, 238.0],
            "confidence": 1.0,
        }
        for month in range(10, 13)
    )
    lines = [
        {
            "content": "2024年01月-2024年12月的还款记录",
            "bbox": [200.0, 194.0, 360.0, 217.0],
            "confidence": 1.0,
        },
        *month_words,
        {"content": "N" * 12, "bbox": [78.5, 249.0, 402.5, 267.0], "confidence": 1.0},
        {"content": "2024", "bbox": [52.0, 270.0, 75.0, 288.0], "confidence": 1.0},
    ]
    image = np.full((600, 455, 3), 255, dtype=np.uint8)
    for boundary in [78.5 + 27.0 * index for index in range(13)]:
        cv2.line(image, (round(boundary), 217), (round(boundary), 310), (0, 0, 0), 2)
    observed_crops: list[tuple[float, float, float, float]] = []

    def capture_template(_image, bbox, **_kwargs):
        observed_crops.append(tuple(float(value) for value in bbox))
        return {"captured_bbox": bbox}

    monkeypatch.setattr(repayment_mod, "extract_micro_cell_glyph_template", capture_template)
    monkeypatch.setattr(
        repayment_mod,
        "_static_n_star_glyph_classification",
        lambda _template: ("*", 0.99, {"classification_basis": "test_grid_geometry"}),
    )

    out = extract_credit_repayment_records(
        lines,
        page=14,
        page_width=455,
        page_height=600,
        page_image=image,
        enable_static_status_validation=True,
    )

    month_bands = [band for band in out["micro_grid"]["col_bands"] if band.get("role") == "month"]
    assert len(month_bands) == 12
    assert month_bands[0]["bbox"][0] == pytest.approx(78.5, abs=2.0)
    assert month_bands[-1]["bbox"][2] == pytest.approx(402.5, abs=2.0)
    assert min(band["bbox"][2] - band["bbox"][0] for band in month_bands) > 20.0
    assert out["micro_grid"]["audit"]["month_header_geometry"] == "word_center_sequence_exact"
    assert out["micro_grid"]["audit"]["visual_month_geometry"]["usable"] is True
    assert len(out["repayment_records"]) == 12
    assert [row["status"] for row in out["repayment_records"]] == ["*"] * 12
    assert len(observed_crops) == 12
    assert min(bbox[2] - bbox[0] for bbox in observed_crops) > 20.0


def test_candidate_b_never_promotes_word_header_digit_when_status_row_is_missing() -> None:
    centers = [92.0 + 27.0 * index for index in range(12)]
    lines = [
        {
            "content": "2024年01月-2024年01月的还款记录",
            "bbox": [200.0, 194.0, 360.0, 217.0],
            "confidence": 1.0,
        },
        *[
            {
                "content": str(month),
                "bbox": [
                    centers[month - 1] - 4.0,
                    222.0,
                    centers[month - 1] + 4.0,
                    238.0,
                ],
                "confidence": 1.0,
            }
            for month in range(1, 13)
        ],
        {"content": "2024", "bbox": [52.0, 260.0, 75.0, 278.0], "confidence": 1.0},
    ]
    out = extract_credit_repayment_records(
        lines,
        page=14,
        page_width=455,
        page_height=600,
        enable_candidate_b_amount_pairing=True,
    )
    rows = records_from_micro_grid_dict(
        out["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [(row["year"], row["month"], row["status"]) for row in rows] == [(2024, 1, "unknown")]
    assert rows[0]["extraction_status"] == "review"


def _multi_page_geometry_lines(year_pages: list[tuple[int, int]]) -> list[dict]:
    years = [year for year, _page in year_pages]
    lines = [
        {
            "content": (f"{min(years)}\u5e7401\u6708-{max(years)}\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55"),
            "bbox": [280.0, 10.0, 510.0, 15.0],
            "confidence": 1.0,
            "source_logical_page": 1,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 18.0, 730.0, 22.0],
            "confidence": 1.0,
            "source_logical_page": 1,
        },
    ]
    for index, (year, logical_page) in enumerate(year_pages):
        y0 = 30.0 + index * 15.0 + (100.0 if logical_page != 1 else 0.0)
        lines.extend(
            [
                {
                    "content": "B" * 12,
                    "bbox": [130.0, y0, 730.0, y0 + 5.0],
                    "confidence": 1.0,
                    "source_logical_page": logical_page,
                },
                {
                    "content": str(year),
                    "bbox": [75.0, y0 + 7.0, 112.0, y0 + 12.0],
                    "confidence": 1.0,
                    "source_logical_page": logical_page,
                },
            ]
        )
    return lines


def _run_geometry_cache_witness(monkeypatch, year_pages: list[tuple[int, int]]):
    import numpy as np

    images = {logical_page: np.full((100, 800, 3), 255, dtype=np.uint8) for _year, logical_page in year_pages}
    calls: list[tuple[int, float, float, tuple[int, ...]]] = []
    original = repayment_mod._visual_month_col_bands

    def counted_visual_month_cols(month_cols, **kwargs):
        image = kwargs["page_image"]
        calls.append(
            (
                id(image),
                float(kwargs["page_width"]),
                float(kwargs["page_height"]),
                tuple(int(value) for value in image.shape),
            )
        )
        return original(month_cols, **kwargs)

    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        counted_visual_month_cols,
    )

    def resolver(logical_page: int):
        return {
            "image": images[logical_page],
            "page_width": 800,
            "page_height": 100,
        }

    out = extract_credit_repayment_records(
        _multi_page_geometry_lines(year_pages),
        page=1,
        page_width=800,
        page_height=100,
        page_image_resolver=resolver,
    )
    projected = records_from_micro_grid_dict(
        out["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    return out, projected, calls, images


def test_visual_month_geometry_keeps_distinct_continuation_rois_separate(monkeypatch) -> None:
    year_pages = [(2024, 1), (2023, 1), (2022, 2), (2021, 2)]
    out, projected, calls, images = _run_geometry_cache_witness(monkeypatch, year_pages)

    assert len(calls) == 3
    assert [call[0] for call in calls] == [id(images[1]), id(images[2]), id(images[2])]
    assert all(call[1:] == (800.0, 100.0, (100, 800, 3)) for call in calls)
    expected = [(year, month, "B", None) for year, _logical_page in year_pages for month in range(1, 13)]
    assert _record_tuples(out["repayment_records"]) == expected
    assert _record_tuples(projected) == expected


def test_visual_month_geometry_does_not_merge_distinct_continuation_pages(monkeypatch) -> None:
    year_pages = [(2024, 1), (2023, 2), (2022, 2), (2021, 3), (2020, 3)]
    out, projected, calls, images = _run_geometry_cache_witness(monkeypatch, year_pages)

    assert len(calls) == 5
    assert [call[0] for call in calls] == [
        id(images[1]),
        id(images[2]),
        id(images[2]),
        id(images[3]),
        id(images[3]),
    ]
    assert _record_tuples(projected) == _record_tuples(out["repayment_records"])
    assert len(projected) == 60


def test_candidate_b_continuation_source_refs_match_month_cells_with_year_rule() -> None:
    import cv2
    import numpy as np

    def line(content: str, bbox: list[float], logical_page: int) -> dict:
        return {
            "content": content,
            "bbox": bbox,
            "confidence": 1.0,
            "source_logical_page": logical_page,
        }

    lines = [
        line(
            "2019\u5e7401\u6708-2020\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            [100.0, 5.0, 260.0, 10.0],
            1,
        ),
        line(
            "1 2 3 4 5 6 7 8 9 10 11 12",
            [40.0, 15.0, 280.0, 20.0],
            1,
        ),
        line("N" * 12, [40.0, 30.0, 280.0, 35.0], 1),
        line("2020", [5.0, 37.0, 20.0, 42.0], 1),
        line("0 " * 12, [40.0, 44.0, 280.0, 49.0], 1),
        # Continuation evidence is shifted by one page height in the joined
        # extraction plane and must be localized back to logical page 2.
        # Page 2 has a different left margin.  Its own physical lattice must
        # drive assignment and refs; inheriting page 1 would shift the row.
        line("N" * 12, [55.0, 120.0, 295.0, 125.0], 2),
        line("2019", [35.0, 127.0, 53.0, 132.0], 2),
        line("0 " * 12, [55.0, 134.0, 295.0, 139.0], 2),
    ]
    images = {}
    for logical_page in (1, 2):
        image = np.full((100, 340, 3), 255, dtype=np.uint8)
        # Fourteen rules: one year column plus twelve month columns.  The
        # continuation page is shifted 15 points to the right.
        offset = 0 if logical_page == 1 else 15
        for x in range(20 + offset, 281 + offset, 20):
            cv2.line(image, (x, 0), (x, 99), (0, 0, 0), 2)
        images[logical_page] = image

    extracted = extract_credit_repayment_records(
        lines,
        page=1,
        page_width=340,
        page_height=100,
        page_image_resolver=lambda logical_page: {
            "image": images[logical_page],
            "page_width": 340,
            "page_height": 100,
        },
        enable_candidate_b_amount_pairing=True,
        continuation_logical_pages=(2,),
        source_table_geometry_by_page={
            "2": [
                _continuation_source_table_geometry(
                    logical_page=2,
                    year_left=35.0,
                    row_edges=(18.0, 30.0, 42.0),
                )
            ]
        },
        grid_index=0,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = {row["month"]: row for row in rows if row["year"] == 2019}

    assert set(continuation) == set(range(1, 13))
    for month, row in continuation.items():
        expected_left = 55.0 + (month - 1) * 20.0
        expected_right = expected_left + 20.0
        refs = row["source_cell_refs"]
        assert {ref["field_name"] for ref in refs} == {"status", "overdue_amount"}
        assert all(ref["logical_page"] == 2 for ref in refs)
        assert all(ref["col"] == month for ref in refs)
        assert all(ref["bbox"][0] == pytest.approx(expected_left, abs=2) for ref in refs)
        assert all(ref["bbox"][2] == pytest.approx(expected_right, abs=2) for ref in refs)
        assert all(
            ref["geometry_provenance"]["selection_basis"] == "source_table_year_plus_twelve_ownership" for ref in refs
        )
        assert all(
            ref["geometry_provenance"]["source_table_comparison"] == "agree"
            and ref["geometry_provenance"]["value_inputs_used"] is False
            for ref in refs
        )
    continuation_audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"]["2"]
    assert continuation_audit["selection_basis"] == ("source_table_year_plus_twelve_ownership")
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        project_personal_detail_datasets,
    )

    semantic = project_personal_detail_datasets(
        {"repayment_records": [{**continuation[1], "account_id": "credit_account:test"}]}
    )["credit_account_monthly_performance"][0]
    assert (
        semantic["source_cell_refs"][0]["geometry_provenance"]["selection_basis"]
        == "source_table_year_plus_twelve_ownership"
    )


def test_candidate_b_continuation_without_image_fails_closed_only_when_enabled() -> None:
    def line(content: str, bbox: list[float], logical_page: int) -> dict:
        return {
            "content": content,
            "bbox": bbox,
            "confidence": 1.0,
            "source_logical_page": logical_page,
        }

    lines = [
        line(
            "2019\u5e7401\u6708-2020\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            [100.0, 5.0, 260.0, 10.0],
            1,
        ),
        line("1 2 3 4 5 6 7 8 9 10 11 12", [40.0, 15.0, 280.0, 20.0], 1),
        line("N" * 12, [40.0, 30.0, 280.0, 35.0], 1),
        line("2020", [5.0, 37.0, 20.0, 42.0], 1),
        line("0 " * 12, [40.0, 44.0, 280.0, 49.0], 1),
        line("N" * 12, [55.0, 120.0, 295.0, 125.0], 2),
        line("2019", [20.0, 127.0, 35.0, 132.0], 2),
        line("0 " * 12, [55.0, 134.0, 295.0, 139.0], 2),
    ]

    shared = extract_credit_repayment_records(
        lines,
        page=1,
        page_width=340,
        page_height=100,
    )
    strict = extract_credit_repayment_records(
        lines,
        page=1,
        page_width=340,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
    )
    shared_rows = records_from_micro_grid_dict(shared["micro_grid"])
    strict_rows = records_from_micro_grid_dict(
        strict["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert all(row["status"] == "N" for row in shared_rows if row["year"] == 2019)
    continuation = [row for row in strict_rows if row["year"] == 2019]
    assert len(continuation) == 12
    assert all(row["status"] == "unknown" for row in continuation)
    assert all(row["extraction_status"] == "review" for row in continuation)
    geometry_audit = strict["micro_grid"]["audit"]["visual_month_geometry_by_page"]["2"]
    assert geometry_audit["source"] == "rejected_month_geometry"
    assert geometry_audit["usable"] is False
    assert geometry_audit["reason"] == "physical_month_column_ownership_unavailable"
    assert geometry_audit["source_logical_page"] == 2


def test_candidate_b_exact_row_cell_disagreement_is_localized_unknown(
    monkeypatch,
) -> None:
    import numpy as np

    class ConflictingRecognition:
        text = "M"
        confidence = 0.99
        source = "cell_crop_consensus"
        raw_text = "M"
        audit = {"consensus_count": 2, "engines": ["shape", "ocr"]}

    monkeypatch.setattr(
        repayment_mod,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: ConflictingRecognition(),
    )
    extracted = extract_credit_repayment_records(
        _exact_year_status_lines("N" * 12),
        page=4,
        page_width=834,
        page_height=600,
        page_image=np.full((600, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=True,
        enable_candidate_b_amount_pairing=True,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [row["month"] for row in rows] == list(range(1, 13))
    assert all(row["status"] == "unknown" for row in rows)
    for row in rows:
        assert row["extraction_status"] == "review"
        assert row["recognition_source"] == ("candidate_b_exact_row_cell_status_conflict")
        assert row["audit"]["reason"] == "exact_row_cell_status_disagreement"
        assert row["audit"]["row_status"] == "N"
        assert row["audit"]["cell_status"] == "M"
        assert row["audit"]["resolution"] == "withheld_unknown_review"
    assert not extracted["repayment_records"]


def _p19_shaped_owned_status_lines(*, duplicate_month_8: bool = False) -> list[dict]:
    lines = [
        {
            "content": ("2024\u5e7401\u6708-2024\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55"),
            "bbox": [90.0, 5.0, 270.0, 15.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [55.0, 25.0, 295.0, 35.0],
            "confidence": 1.0,
        },
        # The source-like row is deliberately not globally exact: month 1 is
        # noisy and month 7 is absent, while month 8 is one independently
        # positioned OCR word in its physical cell.
        {"content": "W", "bbox": [60.0, 50.0, 70.0, 62.0], "confidence": 1.0},
        *[
            {
                "content": "N",
                "bbox": [
                    55.0 + (month - 1) * 20.0 + 5.0,
                    50.0,
                    55.0 + (month - 1) * 20.0 + 15.0,
                    62.0,
                ],
                "confidence": 1.0,
            }
            for month in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
        ],
        *(
            [
                {
                    "content": "N",
                    "bbox": [211.0, 50.5, 214.0, 62.5],
                    "confidence": 1.0,
                }
            ]
            if duplicate_month_8
            else []
        ),
        {"content": "2024", "bbox": [20.0, 70.0, 50.0, 82.0], "confidence": 1.0},
        {
            "content": "000000000000",
            "bbox": [55.0, 84.0, 295.0, 96.0],
            "confidence": 1.0,
        },
    ]
    return lines


def _owned_month_rule_image():
    import cv2
    import numpy as np

    image = np.full((140, 340, 3), 255, dtype=np.uint8)
    # Fourteen physical rules: year-left, year/month boundary, and twelve
    # month-right boundaries.
    for x in range(35, 296, 20):
        cv2.line(image, (x, 0), (x, 139), (0, 0, 0), 2)
    return image


def _target_month_recognizer(target_text: str):
    class Recognition:
        confidence = 0.99
        source = "cell_crop_consensus"

        def __init__(self, text: str) -> None:
            self.text = text
            self.raw_text = text
            self.audit = {
                "consensus_count": 2 if text else 0,
                "engines": ["shape", "ocr"] if text else [],
            }

    def recognize(_image, bbox, **_kwargs):
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        return Recognition(target_text if abs(center_x - 205.0) <= 3.0 else "")

    return recognize


def test_candidate_b_owned_single_token_cell_disagreement_is_localized(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repayment_mod,
        "recognize_micro_cell_from_image",
        _target_month_recognizer("M"),
    )
    extracted = extract_credit_repayment_records(
        _p19_shaped_owned_status_lines(),
        page=19,
        page_width=340,
        page_height=140,
        page_image=_owned_month_rule_image(),
        enable_cell_ocr=True,
        enable_candidate_b_amount_pairing=True,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    target = next(row for row in rows if row["month"] == 8)

    assert target["status"] == "unknown"
    assert target["extraction_status"] == "review"
    assert target["recognition_source"] == ("candidate_b_owned_token_cell_status_conflict")
    assert target["audit"]["reason"] == ("owned_month_token_cell_status_disagreement")
    assert target["audit"]["token_status"] == "N"
    assert target["audit"]["cell_status"] == "M"
    assert target["audit"]["month_geometry_selection_basis"] == ("year_plus_twelve_rule_ownership")
    assert target["audit"]["resolution"] == "withheld_unknown_review"


def test_candidate_b_owned_single_token_agreement_preserves_status(monkeypatch) -> None:
    monkeypatch.setattr(
        repayment_mod,
        "recognize_micro_cell_from_image",
        _target_month_recognizer("N"),
    )
    extracted = extract_credit_repayment_records(
        _p19_shaped_owned_status_lines(),
        page=19,
        page_width=340,
        page_height=140,
        page_image=_owned_month_rule_image(),
        enable_cell_ocr=True,
        enable_candidate_b_amount_pairing=True,
    )
    target = next(
        row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if row["month"] == 8
    )

    assert target["status"] == "N"
    assert target["recognition_source"] == "cell_crop_consensus"
    assert target.get("extraction_status") != "review"


def test_candidate_b_owned_multiple_tokens_do_not_create_conflict_witness(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repayment_mod,
        "recognize_micro_cell_from_image",
        _target_month_recognizer("M"),
    )
    extracted = extract_credit_repayment_records(
        _p19_shaped_owned_status_lines(duplicate_month_8=True),
        page=19,
        page_width=340,
        page_height=140,
        page_image=_owned_month_rule_image(),
        enable_cell_ocr=True,
        enable_candidate_b_amount_pairing=True,
    )
    target = next(
        row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if row["month"] == 8
    )

    assert target["status"] == "M"
    assert target["recognition_source"] == "cell_crop_consensus"
    assert target.get("audit", {}).get("reason") != ("owned_month_token_cell_status_disagreement")


def test_owned_token_conflict_quarantine_is_candidate_b_only(monkeypatch) -> None:
    import numpy as np

    monkeypatch.setattr(
        repayment_mod,
        "recognize_micro_cell_from_image",
        lambda *_args, **_kwargs: type(
            "Recognition",
            (),
            {
                "text": "M",
                "confidence": 0.99,
                "source": "cell_crop_consensus",
                "raw_text": "M",
                "audit": {"consensus_count": 2},
            },
        )(),
    )
    extracted = extract_credit_repayment_records(
        _exact_year_status_lines("N" * 12),
        page=4,
        page_width=834,
        page_height=600,
        page_image=np.full((600, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=True,
    )
    rows = records_from_micro_grid_dict(extracted["micro_grid"])

    assert rows
    assert all(row["status"] == "M" for row in rows)
    assert all(row.get("recognition_source") == "cell_crop_consensus" for row in rows)


def test_static_n_star_classifier_separates_canonical_glyph_shapes() -> None:
    import cv2
    import numpy as np

    def rendered(glyph: str):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        (width, height), _baseline = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_DUPLEX, 0.72, 1)
        cv2.putText(
            image,
            glyph,
            ((80 - width) // 2, (80 + height) // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.72,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return repayment_mod.extract_micro_cell_glyph_template(image, (0, 0, 80, 80), page_width=80, page_height=80)

    n_glyph = rendered("N")
    star_glyph = rendered("*")
    hash_glyph = rendered("#")
    one_glyph = rendered("1")

    assert n_glyph is not None
    assert star_glyph is not None
    assert repayment_mod._static_n_star_glyph_classification(n_glyph)[0] == "N"
    assert repayment_mod._static_n_star_glyph_classification(star_glyph)[0] == "*"
    assert repayment_mod._static_n_star_glyph_classification(hash_glyph) is None
    assert repayment_mod._static_n_star_glyph_classification(one_glyph) is None


def test_static_n_star_classifier_rejects_other_status_and_noise_glyphs() -> None:
    import cv2
    import numpy as np

    for glyph in "HMBAZGC1234567#+.X/":
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        (width, height), _baseline = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.putText(
            image,
            glyph,
            ((64 - width) // 2, (64 + height) // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        template = repayment_mod.extract_micro_cell_glyph_template(
            image,
            (0, 0, 64, 64),
            page_width=64,
            page_height=64,
        )
        assert template is not None
        assert repayment_mod._static_n_star_glyph_classification(template) is None, glyph


def test_static_status_validation_corrects_n_star_without_calling_ocr(monkeypatch) -> None:
    import cv2
    import numpy as np

    lines = [
        {
            "content": "2024年01月-2024年01月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": "N", "bbox": [130.0, 249.0, 180.0, 267.0], "confidence": 1.0},
        {"content": "2024", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
    ]
    star_image = np.full((80, 80, 3), 255, dtype=np.uint8)
    (star_width, star_height), _baseline = cv2.getTextSize("*", cv2.FONT_HERSHEY_DUPLEX, 0.72, 1)
    cv2.putText(
        star_image,
        "*",
        ((80 - star_width) // 2, (80 + star_height) // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    star_glyph = repayment_mod.extract_micro_cell_glyph_template(
        star_image, (0, 0, 80, 80), page_width=80, page_height=80
    )
    assert star_glyph is not None

    monkeypatch.setattr(
        repayment_mod,
        "extract_micro_cell_glyph_template",
        lambda *_args, **_kwargs: star_glyph,
    )

    def forbidden_ocr(*_args, **_kwargs):
        raise AssertionError("static status validation must not invoke OCR")

    monkeypatch.setattr(repayment_mod, "recognize_micro_cell_from_image", forbidden_ocr)
    page_image = np.full((1207, 834, 3), 255, dtype=np.uint8)

    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=1207,
        page_image=page_image,
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    assert [(row["year"], row["month"], row["status"]) for row in out["repayment_records"]] == [
        (2024, 1, "*"),
    ]
    audit = out["micro_grid"]["audit"]
    assert audit["cell_crop_ocr"] == {"enabled": False, "attempts": 0, "hits": 0}
    assert audit["static_status_validation"]["corrections"] == 1
    status_cell = next(
        cell
        for row in out["micro_grid"]["cells"]
        for cell in row
        if cell.get("role") == "status" and cell.get("col_index") == 1
    )
    assert status_cell["recognition_source"] == "static_glyph_shape_validation"


def test_static_status_validation_reports_but_retains_row_evidence_when_image_is_unavailable() -> None:
    lines = [
        {
            "content": "2024年01月-2024年01月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": "N", "bbox": [130.0, 249.0, 180.0, 267.0], "confidence": 1.0},
        {"content": "2024", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
    ]

    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=1207,
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    assert [(row["year"], row["month"], row["status"]) for row in out["repayment_records"]] == [(2024, 1, "N")]
    projected = records_from_micro_grid_dict(out["micro_grid"])
    assert len(projected) == 1
    assert projected[0]["status"] == "N"
    assert projected[0]["extraction_status"] == "review"
    assert projected[0]["audit"]["reported_value_retained"] is True
    assert projected[0]["source_cell_refs"][0]["field_name"] == "status"
    assert projected[0]["source_cell_refs"][0]["geometry_scope"] == "cell"
    audit = out["micro_grid"]["audit"]["static_status_validation"]
    assert audit["attempts"] == 1
    assert audit["unresolved"] == 1
    assert audit["unavailable"] == 1


def _rendered_status_template(glyph: str):
    import cv2
    import numpy as np

    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    (width, height), _baseline = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_DUPLEX, 0.72, 1)
    cv2.putText(
        image,
        glyph,
        ((80 - width) // 2, (80 + height) // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return repayment_mod.extract_micro_cell_glyph_template(image, (0, 0, 80, 80), page_width=80, page_height=80)


def _exact_year_status_lines(statuses: str, *, amount_text: str = "2024") -> list[dict]:
    return [
        {
            "content": "2024年01月-2024年12月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": statuses, "bbox": [130.0, 249.0, 730.0, 267.0], "confidence": 1.0},
        {"content": amount_text, "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
    ]


def _install_static_grid_witnesses(monkeypatch, statuses: str, seed_indices: set[int]) -> None:
    templates = {"N": _rendered_status_template("N"), "*": _rendered_status_template("*")}
    assert all(template is not None for template in templates.values())
    calls = {"template": 0, "classification": 0}

    def template_for_cell(*_args, **_kwargs):
        index = calls["template"]
        calls["template"] += 1
        return templates[statuses[index]]

    def classify_cell(_template):
        index = calls["classification"]
        calls["classification"] += 1
        if index not in seed_indices:
            return None
        return statuses[index], 0.97, {"classification_basis": "test_static_seed"}

    monkeypatch.setattr(repayment_mod, "extract_micro_cell_glyph_template", template_for_cell)
    monkeypatch.setattr(repayment_mod, "_static_n_star_glyph_classification", classify_cell)


@pytest.mark.parametrize("status", ["*", "N"])
def test_static_grid_template_consensus_recovers_exact_uniform_row(monkeypatch, status: str) -> None:
    import numpy as np

    statuses = status * 12
    _install_static_grid_witnesses(monkeypatch, statuses, {0, 1})
    out = extract_credit_repayment_records(
        _exact_year_status_lines(statuses),
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    rows = records_from_micro_grid_dict(out["micro_grid"])
    assert [row["status"] for row in rows] == [status] * 12
    assert sum(row.get("recognition_source") == "static_grid_template_consensus" for row in rows) == 10
    audit = out["micro_grid"]["audit"]["static_status_validation"]
    assert audit["template_consensus_resolved"] == 10
    assert audit["unresolved"] == 0


def test_static_grid_template_consensus_keeps_mixed_classes_separate(monkeypatch) -> None:
    import numpy as np

    statuses = "NNNNNN******"
    _install_static_grid_witnesses(monkeypatch, statuses, {0, 1, 6, 7})
    out = extract_credit_repayment_records(
        _exact_year_status_lines(statuses),
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    assert [row["status"] for row in records_from_micro_grid_dict(out["micro_grid"])] == list(statuses)


def test_static_grid_template_consensus_stops_on_classifier_contradiction(monkeypatch) -> None:
    import numpy as np

    statuses = "N" * 12
    templates = {"N": _rendered_status_template("N"), "*": _rendered_status_template("*")}
    calls = {"template": 0, "classification": 0}

    def template_for_cell(*_args, **_kwargs):
        calls["template"] += 1
        return templates["N"]

    def classify_cell(_template):
        index = calls["classification"]
        calls["classification"] += 1
        if index == 0:
            return "*", 0.97, {"classification_basis": "test_contradiction"}
        if index in {1, 2}:
            return "N", 0.97, {"classification_basis": "test_static_seed"}
        return None

    monkeypatch.setattr(repayment_mod, "extract_micro_cell_glyph_template", template_for_cell)
    monkeypatch.setattr(repayment_mod, "_static_n_star_glyph_classification", classify_cell)
    out = extract_credit_repayment_records(
        _exact_year_status_lines(statuses),
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    rows = records_from_micro_grid_dict(out["micro_grid"])
    assert [row["status"] for row in rows[:3]] == ["*", "N", "N"]
    assert all(row["status"] == "N" for row in rows[3:])
    assert all(row.get("extraction_status") == "review" for row in rows[3:])
    assert all(row["audit"]["reported_value_retained"] is True for row in rows[3:])
    assert out["micro_grid"]["audit"]["static_status_validation"]["contradicted_observed_symbols"] == ["N"]


def test_static_grid_template_consensus_rejects_nonzero_amount_conflict(monkeypatch) -> None:
    import numpy as np

    statuses = "N" * 12
    _install_static_grid_witnesses(monkeypatch, statuses, {0, 1})
    monkeypatch.setattr(repayment_mod, "_normalize_amount_text", lambda _value: "5")
    out = extract_credit_repayment_records(
        _exact_year_status_lines(statuses),
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    assert out["micro_grid"]["audit"]["static_status_validation"]["template_consensus_resolved"] == 0
    assert not any(
        row.get("recognition_source") == "static_grid_template_consensus"
        for row in records_from_micro_grid_dict(out["micro_grid"])
    )


def test_static_grid_template_consensus_rejects_geometry_reused_across_years(monkeypatch) -> None:
    import numpy as np

    statuses = "N" * 24
    _install_static_grid_witnesses(monkeypatch, statuses, {0, 1})
    lines = [
        {
            "content": "2023年01月-2024年12月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": "N" * 12, "bbox": [130.0, 249.0, 730.0, 267.0], "confidence": 1.0},
        {"content": "2024", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
        {"content": "2023", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
    ]
    out = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
    )

    audit = out["micro_grid"]["audit"]["static_status_validation"]
    assert audit["template_consensus_resolved"] == 0
    assert not any(
        row.get("recognition_source") == "static_grid_template_consensus"
        for row in records_from_micro_grid_dict(out["micro_grid"])
    )


@pytest.mark.parametrize("statuses", ["N" * 11, "#" * 12, "1" * 12])
def test_static_grid_template_consensus_does_not_expand_misaligned_hash_or_digit_rows(
    monkeypatch,
    statuses: str,
) -> None:
    import numpy as np

    template_statuses = statuses if len(statuses) == 12 and set(statuses) <= {"N", "*"} else "N" * 12
    _install_static_grid_witnesses(monkeypatch, template_statuses, {0, 1})
    out = extract_credit_repayment_records(
        _exact_year_status_lines(statuses),
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
        extra_status_chars={"#"},
    )

    assert not any(
        row.get("recognition_source") == "static_grid_template_consensus"
        for row in records_from_micro_grid_dict(out["micro_grid"])
    )


def test_candidate_b_glyph_observations_use_private_sink_only(monkeypatch) -> None:
    import json

    import numpy as np

    template = _rendered_status_template("N")
    assert template is not None
    monkeypatch.setattr(
        repayment_mod,
        "extract_micro_cell_glyph_template",
        lambda *_args, **_kwargs: template,
    )
    monkeypatch.setattr(
        repayment_mod,
        "_static_candidate_b_zero_status_glyph_classification",
        lambda _template: (
            "N",
            0.97,
            {"classification_basis": "test_candidate_b_decisive_seed"},
        ),
    )
    observations: list[dict] = []

    out = extract_credit_repayment_records(
        _exact_year_status_lines("N" * 12),
        page=4,
        page_width=834,
        page_height=1207,
        page_image=np.full((1207, 834, 3), 255, dtype=np.uint8),
        enable_cell_ocr=False,
        enable_static_status_validation=True,
        enable_candidate_b_amount_pairing=True,
        candidate_b_status_glyph_observations=observations,
    )

    assert len(observations) == 12
    assert all(observation["template"].shape == (32, 32) for observation in observations)
    assert all(observation["decisive_label"] == "N" for observation in observations)
    serialized = json.loads(json.dumps(out, ensure_ascii=False))

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert "template" not in set(keys(serialized))
    assert "candidate_b_status_glyph_observations" not in out


def test_forensic_api_exports_micro_grids_without_domain_semantics():
    out = extract_credit_repayment_records(_credit_page4_lines(), page=4)
    ds: dict = {"credit_repayment_records": out["repayment_records"]}
    merge_micro_grid_structures_into_bundles(ds, [out["micro_grid"]])
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=ds,
        )
    )

    standard = project_mirror(seal_parse_result(pr), mirror_level="standard")
    forensic = project_mirror(seal_parse_result(pr), mirror_level="forensic")

    assert "repayment_records" not in standard
    standard_grid = _micro_grid_structure_from_document(standard)
    standard_cell = next(cell for row in standard_grid["cells"] for cell in row if cell.get("role") == "status")
    assert standard_grid["grid_type_hint"] == "credit_repayment_record"
    assert standard_cell["text"] == "N"
    assert standard_cell["bbox"]
    assert "token_ids" not in standard_cell
    assert "audit" not in standard_grid
    forensic_grid = _micro_grid_structure_from_document(forensic)
    assert forensic_grid["grid_type_hint"] == "credit_repayment_record"


def test_credit_plugin_maps_generic_scanned_micro_grid_evidence():
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=_micro_grid_bundle_domain(),
        )
    )
    output = {"data": {}, "document": {}}

    enriched = enrich_credit_report_output(output, parse_result=pr)

    assert [
        (r["year"], r["month"], r["status"], r["overdue_amount"]) for r in enriched["data"]["repayment_records"]
    ] == [
        (2021, 1, "N", None),
        (2021, 2, "C", None),
        (2020, 9, "N", "0"),
        (2020, 10, "N", "0"),
        (2020, 11, "N", "0"),
        (2020, 12, "N", "0"),
    ]
    ds = pr.entities.domain_specific
    assert ds["credit_repayment_records"]
    assert micro_grid_structures_from_bundles(ds)
    assert "_micro_grids" not in ds


def test_forensic_api_exports_generic_scanned_micro_grid_evidence_only():
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=_micro_grid_bundle_domain(),
        )
    )

    standard = project_mirror(seal_parse_result(pr), mirror_level="standard")
    forensic = project_mirror(seal_parse_result(pr), mirror_level="forensic")

    assert "scanned_micro_grid_evidence" not in standard
    forensic_doc = forensic
    assert forensic_doc["scanned_ocr_pages"][0]["page"] == 4
    assert forensic_doc["scanned_ocr_pages"][0]["line_count"] > 0
    assert forensic_doc["scanned_ocr_pages"][0]["token_count"] > 0
    assert forensic_doc["scanned_ocr_pages"][0]["payload"] == "external_evidence_bundle"
    evidence = forensic_doc["scanned_micro_grid_evidence"][0]
    assert evidence["page"] == 4
    assert evidence["ocr_page_ref"] == forensic_doc["scanned_ocr_pages"][0]["ocr_page_id"]
    assert "lines" not in evidence
    assert "tokens" not in evidence


def test_four_file_forensic_mirror_includes_plugin_primed_micro_grids_without_semantics():
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=_micro_grid_bundle_domain(),
        )
    )

    outputs = build_all_projections(seal_parse_result(pr))
    document = project_mirror(seal_parse_result(pr), mirror_level="forensic")
    assert "repayment_records" not in document
    grid = _micro_grid_structure_from_document(document)
    assert grid["grid_type_hint"] == "credit_repayment_record"
    assert grid["cells"][0][0]["bbox"]
    page4 = next(p for p in document["pages"] if p.get("page_number") == 4)
    assert any(r.get("kind") == "micro_grid" for r in page4.get("regions") or [])
    repayment_index = next(item for item in outputs["community"]["datasets"] if item["name"] == "repayment_records")
    repayment_dataset = next(
        item for item in outputs["community_bundle"].datasets if item.public["name"] == "repayment_records"
    )
    assert len(repayment_index["rows"]) == len(repayment_dataset.rows)
    assert repayment_index["row_count"] == len(repayment_dataset.rows)
    assert repayment_dataset.rows[0]["normalized"]["status"] == "N"


def test_four_file_standard_mirror_includes_compact_plugin_primed_micro_grids():
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=_micro_grid_bundle_domain(),
        )
    )

    outputs = build_all_projections(seal_parse_result(pr))

    document = outputs["mirror"]
    grid = _micro_grid_structure_from_document(document)
    status_cell = next(cell for row in grid["cells"] for cell in row if cell.get("role") == "status")
    assert "repayment_records" not in document
    assert "scanned_micro_grid_evidence" not in document
    page4 = next(p for p in document["pages"] if p.get("page_number") == 4)
    assert page4.get("flow") is not None
    assert any(r.get("kind") == "micro_grid" for r in page4.get("regions") or [])
    assert grid["grid_id"] == "mg_p4_repayment_0"
    assert status_cell["text"] == "N"
    assert status_cell["bbox"]
    assert "token_ids" not in status_cell
    repayment_index = next(item for item in outputs["community"]["datasets"] if item["name"] == "repayment_records")
    repayment_dataset = next(
        item for item in outputs["community_bundle"].datasets if item.public["name"] == "repayment_records"
    )
    assert len(repayment_index["rows"]) == len(repayment_dataset.rows)
    assert repayment_index["row_count"] == len(repayment_dataset.rows)
    assert repayment_dataset.rows[0]["normalized"]["status"] == "N"


def test_write_outputs_standard_mirror_includes_plugin_primed_micro_grids(tmp_path):
    pr = ParseResult(
        entities=DocumentEntities(
            document_type="credit_report",
            domain_specific=_micro_grid_bundle_domain(),
        )
    )

    _task_id, written = write_outputs(
        pr,
        tmp_path,
        task_id="task_micro_grid",
    )

    mirror = json.loads(written["mirror"].read_text(encoding="utf-8"))
    document = mirror
    assert "repayment_records" not in document
    grid = _micro_grid_structure_from_document(document)
    assert grid["grid_type_hint"] == "credit_repayment_record"
    assert grid["cells"][0][0]["bbox"]


def _continuation_source_table_geometry(
    *,
    table_id: str = "pt_2_0",
    logical_page: int = 2,
    year_left: float = 20.0,
    column_pitch: float = 20.0,
    row_edges: tuple[float, ...] = (8.0, 27.0, 45.0),
    status_row: int = 0,
    amount_row: int = 1,
    year_anchor_row: int = 0,
    year_row_span: int = 2,
    active_months: tuple[int, ...] = tuple(range(1, 13)),
) -> dict:
    vertical_lines = [year_left + column_pitch * index for index in range(14)]
    row_count = len(row_edges) - 1
    cell_bboxes: list[list[list[float] | None]] = []
    cell_geometry_status: list[list[str]] = []
    for row in range(row_count):
        boxes: list[list[float] | None] = []
        statuses: list[str] = []
        for col in range(13):
            boxes.append(
                [
                    vertical_lines[col],
                    row_edges[row],
                    vertical_lines[col + 1],
                    row_edges[row + 1],
                ]
            )
            statuses.append("exact")
        cell_bboxes.append(boxes)
        cell_geometry_status.append(statuses)

    for row in range(row_count):
        cell_bboxes[row][0] = None
        cell_geometry_status[row][0] = "derived"
    cell_bboxes[year_anchor_row][0] = [
        vertical_lines[0],
        row_edges[year_anchor_row],
        vertical_lines[1],
        row_edges[year_anchor_row + year_row_span],
    ]
    cell_geometry_status[year_anchor_row][0] = "exact"
    for month in set(range(1, 13)).difference(active_months):
        cell_bboxes[status_row][month] = None
        cell_bboxes[amount_row][month] = None
        cell_geometry_status[status_row][month] = "derived"
        cell_geometry_status[amount_row][month] = "derived"

    return {
        "table_id": table_id,
        "logical_page": logical_page,
        "source_page": logical_page,
        "bbox": [20.0, row_edges[0], 280.0, row_edges[-1]],
        "extraction_layer": "scanned_image_line_grid",
        "geometry_source": "scanned_image_line_grid",
        "coordinate_system": "pdf_points_top_left",
        "vertical_lines": vertical_lines,
        "horizontal_lines": list(row_edges),
        # These are the exact axis-band shapes emitted by the scanned-table
        # reconstructor; they deliberately do not contain synthetic bboxes.
        "col_bands": [
            {
                "index": index,
                "x0": vertical_lines[index],
                "x1": vertical_lines[index + 1],
            }
            for index in range(13)
        ],
        "row_bands": [{"index": row, "y0": row_edges[row], "y1": row_edges[row + 1]} for row in range(row_count)],
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": cell_geometry_status,
        "cell_spans": [
            {
                "row": year_anchor_row,
                "col": 0,
                "row_span": year_row_span,
                "col_span": 1,
                "bbox": cell_bboxes[year_anchor_row][0],
            }
        ],
        # Adversarial source values are outside the geometry contract and must
        # never influence status or amount materialization.
        "raw_rows": [["2099", "M", "999"]],
    }


def _source_owned_continuation_lines(*, three_row_fragment: bool = False) -> list[dict]:
    def line(content: str, bbox: list[float], logical_page: int) -> dict:
        return {
            "content": content,
            "bbox": bbox,
            "confidence": 1.0,
            "source_logical_page": logical_page,
        }

    start = "2021\u5e7405\u6708" if three_row_fragment else "2019\u5e7401\u6708"
    end = "2022\u5e7412\u6708" if three_row_fragment else "2020\u5e7412\u6708"
    base_year = "2022" if three_row_fragment else "2020"
    continuation_year = "2021" if three_row_fragment else "2019"
    lines = [
        line(
            f"{start}-{end}\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            [80.0, 5.0, 260.0, 10.0],
            1,
        ),
        line("1 2 3 4 5 6 7 8 9 10 11 12", [40.0, 15.0, 280.0, 20.0], 1),
        line("N" * 12, [40.0, 30.0, 280.0, 35.0], 1),
        line(base_year, [20.0, 37.0, 38.0, 42.0], 1),
        line("0 " * 12, [40.0, 44.0, 280.0, 49.0], 1),
    ]
    if three_row_fragment:
        lines.extend(
            [
                # A preceding amount fragment must not become the 2021 row.
                line("0 " * 12, [40.0, 103.0, 280.0, 109.0], 2),
                line("N" * 8, [120.0, 115.0, 280.0, 122.0], 2),
                line(continuation_year, [20.0, 123.0, 38.0, 128.0], 2),
                line("0 " * 8, [120.0, 134.0, 280.0, 139.0], 2),
            ]
        )
    else:
        lines.extend(
            [
                line("N" * 12, [40.0, 112.0, 280.0, 120.0], 2),
                line(continuation_year, [20.0, 121.0, 38.0, 126.0], 2),
                line("0 " * 12, [40.0, 130.0, 280.0, 138.0], 2),
            ]
        )
    return lines


def _extract_source_owned_continuation(
    *,
    lines: list[dict] | None = None,
    tokens: list[dict] | None = None,
    geometry: dict | None = None,
    continuation_logical_pages: tuple[int, ...] = (2,),
    page_image_resolver=None,
    enable_cell_ocr: bool = False,
):
    return extract_credit_repayment_records(
        lines or _source_owned_continuation_lines(),
        tokens=tokens,
        page=1,
        page_width=320,
        page_height=100,
        page_image_resolver=page_image_resolver,
        enable_cell_ocr=enable_cell_ocr,
        enable_candidate_b_amount_pairing=True,
        continuation_logical_pages=continuation_logical_pages,
        source_table_geometry_by_page={"2": [geometry or _continuation_source_table_geometry()]},
    )


def _source_owned_base_lines(*, year: int, status_text: str) -> list[dict]:
    def source_line(content: str, bbox: list[float]) -> dict:
        return {
            "content": content,
            "bbox": bbox,
            "confidence": 1.0,
        }

    return [
        source_line(
            f"{year}\u5e7401\u6708-{year}\u5e7412\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            [80.0, 5.0, 260.0, 10.0],
        ),
        source_line("1 2 3 4 5 6 7 8 9 10 11 12", [40.0, 15.0, 280.0, 20.0]),
        source_line(status_text, [40.0, 30.0, 280.0, 35.0]),
        source_line(str(year), [20.0, 37.0, 38.0, 42.0]),
        source_line("0 " * 12, [40.0, 44.0, 280.0, 49.0]),
    ]


def _base_source_geometry(*, page: int) -> dict:
    geometry = _continuation_source_table_geometry(
        table_id=f"pt_{page}_0",
        logical_page=page,
        row_edges=(25.0, 40.0, 52.0),
    )
    geometry["raw_rows"] = [["2099", "PRIVATE_SOURCE_VALUE", "999999"]]
    return geometry


def test_repayment_grid_preserves_raw_anchor_identity_without_using_it_for_values() -> None:
    """The upstream-authenticated anchor survives registration as opaque audit data."""

    lines = _source_owned_base_lines(year=2021, status_text="N" * 12)
    identity = {
        "coordinate_system": "pdf_points_top_left",
        "coordinate_plane": "raw_logical_page",
        "source_logical_page": 9,
        "source_page": 5,
        "evidence_ids": ["sealed-anchor-word-1", "sealed-anchor-word-2"],
        "bbox": [180.0, 205.0, 360.0, 210.0],
        "date_range": [2021, 1, 2021, 12],
    }
    options = {
        "page": 10,
        "page_width": 320,
        "page_height": 100,
        "enable_candidate_b_amount_pairing": True,
        "source_table_geometry_by_page": {"10": [_base_source_geometry(page=10)]},
    }
    unannotated = extract_credit_repayment_records(lines, **options)
    assert "printed_anchor_provenance" not in unannotated["micro_grid"]["audit"]
    lines[0]["printed_anchor_identity"] = deepcopy(identity)

    annotated = extract_credit_repayment_records(lines, **options)
    provenance = annotated["micro_grid"]["audit"]["printed_anchor_provenance"]

    assert provenance == identity
    assert annotated["repayment_records"] == unannotated["repayment_records"]
    assert annotated["micro_grid"]["cells"] == unannotated["micro_grid"]["cells"]
    assert lines[0]["printed_anchor_identity"] == identity
    lines[0]["printed_anchor_identity"]["evidence_ids"].append("later-input-mutation")
    assert provenance == identity
    provenance["bbox"][0] = -1.0
    assert lines[0]["printed_anchor_identity"]["bbox"] == identity["bbox"]


@pytest.mark.parametrize("untyped_identity", (None, [], "unverified-anchor"))
def test_repayment_line_adapter_does_not_invent_printed_anchor_identity(untyped_identity) -> None:
    line = {
        "content": "2021年01月-2021年12月的还款记录",
        "bbox": [80.0, 5.0, 260.0, 10.0],
        "source_bbox": [180.0, 205.0, 360.0, 210.0],
        "source_logical_page": 9,
        "coordinate_logical_page": 10,
        "evidence_ids": ["unverified-generic-id"],
        "printed_anchor_identity": untyped_identity,
    }

    [adapted] = repayment_mod._line_items([line])

    assert "printed_anchor_identity" not in adapted
    assert adapted["source_logical_page"] == 10
    assert adapted["source_origin_logical_page"] == 9


def _owned_visual_cols(
    *,
    shift: float,
    owned_rule_hits: int = 12,
    residual_shift_months: float = 0.52,
) -> tuple[list[dict], dict]:
    return (
        [
            {
                "index": month,
                "header": str(month),
                "role": "month",
                "bbox": [
                    40.0 + (month - 1) * 20.0 + shift,
                    15.0,
                    40.0 + month * 20.0 + shift,
                    20.0,
                ],
            }
            for month in range(1, 13)
        ],
        {
            "source": "vertical_rule_projection",
            "usable": True,
            "selection_basis": "year_plus_twelve_rule_ownership",
            "rule_hits": owned_rule_hits,
            "owned_month_rule_hits": owned_rule_hits,
            "residual_shift_months": residual_shift_months,
            "offset": shift,
            "right_offset": shift,
        },
    )


def test_candidate_b_visual_lattice_source_requirement_is_disjunctive() -> None:
    base_audit = {
        "source": "vertical_rule_projection",
        "usable": True,
        "selection_basis": "year_plus_twelve_rule_ownership",
    }

    assert repayment_mod._candidate_b_visual_lattice_needs_source_table(
        {**base_audit, "owned_month_rule_hits": 12, "residual_shift_months": 0.1}
    )
    assert repayment_mod._candidate_b_visual_lattice_needs_source_table(
        {**base_audit, "owned_month_rule_hits": 13, "residual_shift_months": 0.49}
    )
    assert not repayment_mod._candidate_b_visual_lattice_needs_source_table(
        {**base_audit, "owned_month_rule_hits": 13, "residual_shift_months": 0.1}
    )


@pytest.mark.parametrize(
    ("page", "year", "status_by_month", "targets"),
    (
        (
            10,
            2022,
            {6: "N", 7: "C"},
            ((6, "C"),),
        ),
        (10, 2020, {8: "N", 9: "M"}, ((8, "M"),)),
        (17, 2020, {9: "N", 10: "Z"}, ((9, "Z"),)),
        (18, 2019, {9: "N", 10: "G"}, ((9, "G"),)),
    ),
    ids=(
        "p10_2022_june",
        "p10_2020_august",
        "p17_2020_september",
        "p18_2019_september",
    ),
)
def test_candidate_b_base_source_lattice_repairs_plus_one_month_visual_shift(
    monkeypatch,
    page: int,
    year: int,
    status_by_month: dict[int, str],
    targets: tuple[tuple[int, str], ...],
) -> None:
    status_text = "".join(status_by_month.get(month, "N") for month in range(1, 13))
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(shift=20.0),
    )

    extracted = extract_credit_repayment_records(
        _source_owned_base_lines(year=year, status_text=status_text),
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = {
        row["month"]: row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
    }

    for month, formerly_shifted_value in targets:
        assert rows[month]["status"] == status_by_month[month]
        assert rows[month]["status"] != formerly_shifted_value
        status_ref = next(ref for ref in rows[month]["source_cell_refs"] if ref["field_name"] == "status")
        provenance = status_ref["geometry_provenance"]
        assert status_ref["col"] == month
        assert status_ref["bbox"] == [
            40.0 + (month - 1) * 20.0,
            25.0,
            40.0 + month * 20.0,
            40.0,
        ]
        assert provenance["source"] == "source_table_geometry"
        assert provenance["table_id"] == f"pt_{page}_0"
        assert provenance["source_table_comparison"] == ("source_over_ambiguous_visual")
        assert provenance["visual_owned_month_rule_hits"] == 12
        assert provenance["visual_residual_shift_months"] == 0.52
        assert provenance["corroborated_by_source_table_geometry"] is False
        assert provenance["ambiguous_visual_geometry_superseded"] is True
        assert provenance["value_inputs_used"] is False
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"][str(page)]
    assert audit["reason"] == "exact_source_table_month_lattice_calibration"
    assert audit["source_table_comparison"] == "source_over_ambiguous_visual"
    assert audit["corroborated_by_source_table_geometry"] is False
    assert audit["ambiguous_visual_geometry_superseded"] is True
    assert audit["value_inputs_used"] is False

    from types import SimpleNamespace

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        collect_extraction_issues,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        PersonalDetailOCRCorrectionOverlay,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        project_personal_detail_datasets,
    )

    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    corrected = overlay.correct_business_candidates(
        {"repayment_records": [{**row, "account_id": "credit_account:test"} for row in rows.values()]},
        stage="candidate_b_final_validation",
    )
    corrected["personal_detail_extraction_issues"] = collect_extraction_issues(
        SimpleNamespace(
            _personal_detail_extraction_issues=[],
            ocr_correction_audit=overlay.audit,
        )
    )
    projected = project_personal_detail_datasets(corrected)
    active_issues = projected.get("extraction_issues", [])
    for month, _formerly_shifted_value in targets:
        target_record_id = f"mg_p{page}_repayment_0:{year:04d}-{month:02d}"
        assert any(
            row.get("monthly_performance_id") == target_record_id and row.get("status_code") == status_by_month[month]
            for row in projected["credit_account_monthly_performance"]
        ), target_record_id
        assert not any(
            issue.get("target_dataset") == "credit_account_monthly_performance"
            and issue.get("target_record_id") == target_record_id
            and issue.get("field_name") == "status_code"
            and issue.get("issue_code")
            and str(issue.get("status") or "requires_review")
            not in {"resolved", "suppressed_redundant", "informational"}
            for issue in active_issues
        ), target_record_id


def test_candidate_b_independently_exact_visual_source_disagreement_still_blocks(
    monkeypatch,
) -> None:
    page = 10
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(
            shift=20.0,
            owned_rule_hits=13,
            residual_shift_months=0.1,
        ),
    )

    extracted = extract_credit_repayment_records(
        _source_owned_base_lines(
            year=2022,
            status_text="NNNNNNCNNNNN",
        ),
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = {
        row["month"]: row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
    }

    assert rows[6]["status"] == "unknown"
    status_ref = next(ref for ref in rows[6]["source_cell_refs"] if ref["field_name"] == "status")
    assert status_ref["geometry_rejection"] == {
        "source": "rejected_month_geometry",
        "reason": "source_table_month_geometry_plane_conflict",
        "source_table_id": "pt_10_0",
        "source_table_comparison": "disagree",
        "logical_page": 10,
        "value_inputs_used": False,
    }


@pytest.mark.parametrize(
    "exact_source",
    (
        "exact_native_source_table_status_cell",
        "exact_corrected_source_table_status_cell",
    ),
)
def test_candidate_b_exact_cell_atoms_resolve_visual_source_disagreement(
    monkeypatch,
    exact_source: str,
) -> None:
    """A third exact plane authorizes only its independently owned cells."""

    page = 12
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(
            shift=20.0,
            owned_rule_hits=13,
            residual_shift_months=0.1,
        ),
    )
    tokens = [
        {
            "token_id": f"exact_status_{month}",
            "content": "N",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                28.0,
                56.0 + (month - 1) * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": exact_source,
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
    ]

    extracted = extract_credit_repayment_records(
        _source_owned_base_lines(year=2023, status_text="N" * 12),
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert len(rows) == 12
    assert all(row["status"] == "N" for row in rows)
    assert all(row["overdue_amount"] is None for row in rows)
    for row in rows:
        status_ref = next(ref for ref in row["source_cell_refs"] if ref["field_name"] == "status")
        proof = status_ref["geometry_provenance"]
        assert proof["source_table_comparison"] == ("source_over_conflicting_visual_exact_cell_atoms")
        assert proof["conflicting_visual_geometry_superseded"] is True
        assert proof["exact_source_atom_geometry_months"] == list(range(1, 13))


def test_candidate_b_corrected_and_exact_status_disagreement_is_withheld(
    monkeypatch,
) -> None:
    """Neither contradictory status plane may silently overwrite the other."""

    page = 12
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(
            shift=20.0,
            owned_rule_hits=13,
            residual_shift_months=0.1,
        ),
    )
    lines = _source_owned_base_lines(year=2024, status_text="MCN" + "N" * 9)
    lines[0]["content"] = "2024年01月-2024年03月的还款记录"
    tokens = [
        {
            "token_id": f"exact_status_{month}",
            "content": "N",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                28.0,
                56.0 + (month - 1) * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": "exact_native_source_table_status_cell",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 4)
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [(row["month"], row["status"]) for row in rows] == [
        (1, "unknown"),
        (2, "unknown"),
        (3, "N"),
    ]
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"][str(page)]
    assert audit["source_table_comparison"] == (
        "source_over_conflicting_visual_exact_cell_atoms"
    )
    assert audit["exact_source_atom_geometry_months"] == [1, 2, 3]


def test_candidate_b_exact_atom_withholds_conflicting_status_and_preserves_amount(
    monkeypatch,
) -> None:
    """A native/corrected status conflict remains field-level downstream."""

    page = 19
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(
            shift=0.0,
            owned_rule_hits=13,
            residual_shift_months=0.1,
        ),
    )
    lines = _source_owned_base_lines(
        year=2022,
        status_text="N" * 7 + "M" + "N" * 4,
    )
    exact_native = {
        "token_id": "exact_status_8",
        "content": "N",
        "bbox": [184.0, 28.0, 196.0, 36.0],
        "confidence": 0.99,
        "source": "exact_native_source_table_status_cell",
        "coordinate_system": "pdf_points_top_left",
    }

    extracted = extract_credit_repayment_records(
        lines,
        tokens=[exact_native],
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    august = next(row for row in rows if row["month"] == 8)

    assert august["status"] == "unknown"
    assert august["overdue_amount"] == "0"


def test_candidate_b_partial_exact_atom_override_does_not_address_siblings(
    monkeypatch,
) -> None:
    """Partial third-plane proof remains field-local through amount pairing."""

    page = 12
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(
            shift=20.0,
            owned_rule_hits=13,
            residual_shift_months=0.1,
        ),
    )
    lines = _source_owned_base_lines(
        year=2023,
        status_text="N N N N 福 N N N N N N N",
    )
    lines[-1] = {
        "content": "0",
        "bbox": [124.0, 44.0, 136.0, 49.0],
        "confidence": 1.0,
    }
    tokens = [
        {
            "token_id": "exact_status_5",
            "content": "N",
            "bbox": [124.0, 28.0, 136.0, 36.0],
            "confidence": 0.99,
            "source": "exact_native_source_table_status_cell",
            "coordinate_system": "pdf_points_top_left",
        }
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )

    pairing = extracted["micro_grid"]["audit"]["candidate_b_amount_pairing"]["2023"]
    assert pairing["cell_status_by_month"]["5"] == "month_geometry_unowned"
    assert pairing["cell_status_by_month"]["4"] == "month_geometry_unowned"
    assert pairing["cell_status_by_month"]["6"] == "month_geometry_unowned"
    assert pairing["unowned_geometry_months"] == list(range(1, 13))
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"][str(page)]
    assert audit["source_table_comparison"] == (
        "source_over_conflicting_visual_exact_cell_atoms"
    )
    assert audit["exact_source_atom_geometry_months"] == [5]


@pytest.mark.parametrize(
    ("visual", "expected_comparison"),
    (
        (_owned_visual_cols(shift=0.0, owned_rule_hits=13, residual_shift_months=0.1), "agree"),
        (_owned_visual_cols(shift=20.0), "source_over_ambiguous_visual"),
    ),
    ids=("complete_planes_agree", "visual_plane_is_ambiguous"),
)
def test_candidate_b_partial_repair_atoms_do_not_narrow_complete_source_plane(
    monkeypatch,
    visual: tuple[list[dict], dict],
    expected_comparison: str,
) -> None:
    """Field atoms repair values; complete-plane arbitration owns geometry."""

    page = 12
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: visual,
    )
    lines = _source_owned_base_lines(
        year=2023,
        status_text="N N N N 福 N N N N N N N",
    )
    lines[-1] = {
        "content": "0",
        "bbox": [124.0, 44.0, 136.0, 49.0],
        "confidence": 1.0,
    }
    tokens = [
        {
            "token_id": "exact_status_5",
            "content": "N",
            "bbox": [124.0, 28.0, 136.0, 36.0],
            "confidence": 0.99,
            "source": "exact_native_source_table_status_cell",
            "coordinate_system": "pdf_points_top_left",
        }
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )

    pairing = extracted["micro_grid"]["audit"]["candidate_b_amount_pairing"]["2023"]
    assert pairing["cell_status_by_month"]["5"] == "exact"
    assert pairing["cell_status_by_month"]["4"] == "blank_amount_cell"
    assert "unowned_geometry_months" not in pairing
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"][str(page)]
    assert audit["source_table_comparison"] == expected_comparison
    assert audit["exact_source_atom_geometry_months"] == []


def test_candidate_b_base_source_lattice_calibrates_agreement_and_exposes_proof(
    monkeypatch,
) -> None:
    page = 10
    status_text = "NNNNNNCNMNNN"
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(shift=0.0),
    )

    extracted = extract_credit_repayment_records(
        _source_owned_base_lines(year=2020, status_text=status_text),
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = {
        row["month"]: row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
    }

    assert rows[6]["status"] == "N"
    assert rows[8]["status"] == "N"
    for month in (6, 8):
        status_ref = next(ref for ref in rows[month]["source_cell_refs"] if ref["field_name"] == "status")
        provenance = status_ref["geometry_provenance"]
        assert status_ref["col"] == month
        assert status_ref["bbox"] == [
            40.0 + (month - 1) * 20.0,
            25.0,
            40.0 + month * 20.0,
            40.0,
        ]
        assert provenance["selection_basis"] == ("source_table_year_plus_twelve_ownership")
        assert provenance["source_table_comparison"] == "agree"
        assert provenance["calibrated_from_source_table_geometry"] is True
        assert provenance["corroborated_by_source_table_geometry"] is True
        assert provenance["ambiguous_visual_geometry_superseded"] is False
        assert provenance["visual_owned_month_rule_hits"] == 12
        assert provenance["visual_residual_shift_months"] == 0.52
        assert provenance["value_inputs_used"] is False
    assert "PRIVATE_SOURCE_VALUE" not in json.dumps(extracted["micro_grid"], ensure_ascii=False)

    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        project_personal_detail_datasets,
    )

    projected = project_personal_detail_datasets(
        {"repayment_records": [{**row, "account_id": "credit_account:test"} for row in rows.values()]}
    )["credit_account_monthly_performance"]
    assert len(projected) == 12
    for projected_row in projected:
        month = int(projected_row["performance_month"][-2:])
        for ref in projected_row["source_cell_refs"]:
            assert ref["col"] == month
            assert ref["bbox"][0] == 40.0 + (month - 1) * 20.0
            assert ref["bbox"][2] == 40.0 + month * 20.0
            assert ref["geometry_provenance"]["selection_basis"] == ("source_table_year_plus_twelve_ownership")
            assert ref["geometry_provenance"]["value_inputs_used"] is False


def test_candidate_b_underdetermined_owned_visual_lattice_without_source_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(shift=0.0),
    )

    extracted = extract_credit_repayment_records(
        _source_owned_base_lines(year=2020, status_text="N" * 12),
        page=10,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert all(row["status"] == "unknown" for row in rows)
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"]["10"]
    assert audit["reason"] == "source_table_month_ownership_required"
    assert audit["visual_owned_month_rule_hits"] == 12
    assert audit["visual_residual_shift_months"] == 0.52


def test_source_table_geometry_is_default_off_and_private_values_never_leak() -> None:
    lines = _source_owned_base_lines(year=2020, status_text="N" * 12)
    baseline = extract_credit_repayment_records(
        lines,
        page=10,
        page_width=320,
        page_height=100,
    )
    protected = extract_credit_repayment_records(
        lines,
        page=10,
        page_width=320,
        page_height=100,
        source_table_geometry_by_page={"10": [_base_source_geometry(page=10)]},
    )

    assert protected == baseline
    assert "PRIVATE_SOURCE_VALUE" not in json.dumps(protected, ensure_ascii=False)


def test_candidate_b_headerless_continuation_uses_exact_source_table_geometry() -> None:
    extracted = _extract_source_owned_continuation()
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = {row["month"]: row for row in rows if row["year"] == 2019}

    assert set(continuation) == set(range(1, 13))
    assert all(row["status"] == "N" for row in continuation.values())
    assert all(row["overdue_amount"] == "0" for row in continuation.values())
    for month, row in continuation.items():
        refs = {ref["field_name"]: ref for ref in row["source_cell_refs"]}
        assert refs["status"]["col"] == month
        assert refs["overdue_amount"]["col"] == month
        assert refs["status"]["bbox"] == [
            40.0 + (month - 1) * 20.0,
            8.0,
            40.0 + month * 20.0,
            27.0,
        ]
        assert refs["overdue_amount"]["bbox"] == [
            40.0 + (month - 1) * 20.0,
            27.0,
            40.0 + month * 20.0,
            45.0,
        ]
        assert refs["status"]["geometry_provenance"]["selection_basis"] == ("source_table_year_plus_twelve_ownership")
        assert refs["status"]["geometry_provenance"]["value_inputs_used"] is False
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"]["2"]
    assert audit["table_id"] == "pt_2_0"
    assert audit["status_row_index"] == 0
    assert audit["amount_row_index"] == 1


def test_candidate_b_continuation_uses_boundary_straddling_year_singletons() -> None:
    geometry = _continuation_source_table_geometry(
        year_anchor_row=0,
        year_row_span=1,
    )
    geometry["cell_bboxes"][1][0] = [20.0, 27.0, 40.0, 45.0]
    geometry["cell_geometry_status"][1][0] = "exact"
    geometry["cell_spans"].append(
        {
            "row": 1,
            "col": 0,
            "row_span": 1,
            "col_span": 1,
            "bbox": geometry["cell_bboxes"][1][0],
        }
    )
    lines = [dict(line) for line in _source_owned_continuation_lines()]
    continuation_year = next(
        line for line in lines if line.get("source_logical_page") == 2 and line.get("content") == "2019"
    )
    # In local continuation coordinates this is y=23..32, crossing the exact
    # singleton boundary at y=27 with material glyph support on both sides.
    continuation_year["bbox"] = [20.0, 123.0, 38.0, 132.0]

    extracted = _extract_source_owned_continuation(
        lines=lines,
        geometry=geometry,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = {row["month"]: row for row in rows if row["year"] == 2019}

    assert set(continuation) == set(range(1, 13))
    assert all(row["status"] == "N" for row in continuation.values())
    assert all(row["overdue_amount"] == "0" for row in continuation.values())
    for month, row in continuation.items():
        refs = {ref["field_name"]: ref for ref in row["source_cell_refs"]}
        assert refs["status"]["logical_page"] == 2
        assert refs["status"]["geometry_scope"] == "cell"
        assert refs["status"]["col"] == month
        assert refs["overdue_amount"]["col"] == month
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"]["2"]
    assert audit["year_anchor_mode"] == ("boundary_straddling_singleton_year_cells")
    assert audit["table_id"] == "pt_2_0"
    assert audit["value_inputs_used"] is False


def test_candidate_b_three_row_continuation_binds_only_active_status_pair() -> None:
    geometry = _continuation_source_table_geometry(
        row_edges=(0.0, 12.0, 30.0, 45.0),
        status_row=1,
        amount_row=2,
        year_anchor_row=0,
        year_row_span=3,
        active_months=tuple(range(5, 13)),
    )
    extracted = _extract_source_owned_continuation(
        lines=_source_owned_continuation_lines(three_row_fragment=True),
        geometry=geometry,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = {row["month"]: row for row in rows if row["year"] == 2021}

    assert set(continuation) == set(range(5, 13))
    assert all(row["status"] == "N" for row in continuation.values())
    assert all(row["overdue_amount"] == "0" for row in continuation.values())
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"]["2"]
    assert audit["status_row_index"] == 1
    assert audit["amount_row_index"] == 2
    assert audit["selection_basis"] == "source_table_year_plus_twelve_ownership"


def test_candidate_b_three_row_continuation_keeps_adjacent_month_columns_distinct() -> None:
    geometry = _continuation_source_table_geometry(
        row_edges=(0.0, 12.0, 30.0, 45.0),
        status_row=1,
        amount_row=2,
        year_anchor_row=0,
        year_row_span=3,
        active_months=tuple(range(5, 13)),
    )
    lines = [dict(line) for line in _source_owned_continuation_lines(three_row_fragment=True)]
    continuation_status = next(
        line
        for line in lines
        if line.get("source_logical_page") == 2 and line.get("bbox") == [120.0, 115.0, 280.0, 122.0]
    )
    # The source table supplies geometry only.  Even when the status content is
    # identical, August and September must retain their own physical columns;
    # the separate native-cell guard handles any source-value disagreement.
    continuation_status["content"] = "********"

    extracted = _extract_source_owned_continuation(lines=lines, geometry=geometry)
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = {row["month"]: row for row in rows if row["year"] == 2021}

    assert continuation[8]["status"] == "*"
    assert continuation[9]["status"] == "*"
    assert {ref["col"] for ref in continuation[8]["source_cell_refs"] if ref["field_name"] == "status"} == {8}
    assert {ref["col"] for ref in continuation[9]["source_cell_refs"] if ref["field_name"] == "status"} == {9}
    assert all(
        ref["geometry_provenance"]["value_inputs_used"] is False
        for month in (8, 9)
        for ref in continuation[month]["source_cell_refs"]
    )


@pytest.mark.parametrize("failure_mode", ["not_adjacent", "thirteen_rules", "ambiguous"])
def test_candidate_b_source_table_geometry_bridge_fails_closed(failure_mode: str) -> None:
    geometry = _continuation_source_table_geometry()
    continuation_pages = (2,)
    tables = [geometry]
    if failure_mode == "not_adjacent":
        continuation_pages = ()
    elif failure_mode == "thirteen_rules":
        geometry["vertical_lines"] = geometry["vertical_lines"][:-1]
    else:
        tables.append({**geometry, "table_id": "pt_2_1"})

    extracted = extract_credit_repayment_records(
        _source_owned_continuation_lines(),
        page=1,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        continuation_logical_pages=continuation_pages,
        source_table_geometry_by_page={"2": tables},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = [row for row in rows if row["year"] == 2019]

    assert len(continuation) == 12
    assert all(row["status"] == "unknown" for row in continuation)
    assert all(row["extraction_status"] == "review" for row in continuation)


def test_candidate_b_source_geometry_preserves_local_cell_conflict(monkeypatch) -> None:
    import numpy as np

    class Recognition:
        confidence = 0.99
        source = "cell_crop_consensus"

        def __init__(self, text: str) -> None:
            self.text = text
            self.raw_text = text
            self.audit = {"consensus_count": 2, "engines": ["shape", "ocr"]}

    def recognize(_image, bbox, **_kwargs):
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        month = int((center_x - 40.0) // 20.0) + 1
        return Recognition("M" if month == 8 else "N")

    monkeypatch.setattr(repayment_mod, "recognize_micro_cell_from_image", recognize)
    image = np.full((100, 320, 3), 255, dtype=np.uint8)
    extracted = _extract_source_owned_continuation(
        page_image_resolver=lambda _page: {
            "image": image,
            "page_width": 320,
            "page_height": 100,
        },
        enable_cell_ocr=True,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = {row["month"]: row for row in rows if row["year"] == 2019}

    assert continuation[8]["status"] == "unknown"
    assert continuation[8]["extraction_status"] == "review"
    assert continuation[8]["audit"]["reason"] == "exact_row_cell_status_disagreement"
    assert all(row["status"] == "N" for month, row in continuation.items() if month != 8)


@pytest.mark.parametrize("source_token_evidence", (False, True))
def test_candidate_b_source_geometry_supersedes_high_residual_visual_lattice(
    source_token_evidence: bool,
) -> None:
    import cv2
    import numpy as np

    base_image = np.full((100, 320, 3), 255, dtype=np.uint8)
    continuation_image = base_image.copy()
    for x in range(35, 296, 20):
        cv2.line(continuation_image, (x, 0), (x, 99), (0, 0, 0), 2)
    extracted = _extract_source_owned_continuation(
        tokens=(
            [
                {
                    "token_id": f"continuation-status-{month}",
                    "content": "N",
                    "bbox": [45.0 + 20.0 * (month - 1), 112.0, 55.0 + 20.0 * (month - 1), 120.0],
                    "confidence": 0.99,
                    "source": "rapidocr_word",
                    "coordinate_system": "pdf_points_top_left",
                }
                for month in range(1, 13)
            ]
            if source_token_evidence
            else None
        ),
        lines=[
            {
                **line,
                "bbox": (
                    [55.0, line["bbox"][1], 295.0, line["bbox"][3]]
                    if line.get("source_logical_page") == 2 and str(line.get("content") or "").startswith("N")
                    else line["bbox"]
                ),
            }
            for line in _source_owned_continuation_lines()
        ],
        page_image_resolver=lambda logical_page: {
            "image": continuation_image if logical_page == 2 else base_image,
            "page_width": 320,
            "page_height": 100,
        },
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    continuation = [row for row in rows if row["year"] == 2019]

    # Source-table geometry is value-free. A drifted merged row alone cannot
    # authenticate twelve glyphs; independently positioned words can do so.
    assert all(row["status"] == ("N" if source_token_evidence else "unknown") for row in continuation)
    assert all(row["overdue_amount"] == "0" for row in continuation)
    audit = extracted["micro_grid"]["audit"]["visual_month_geometry_by_page"]["2"]
    assert audit["reason"] == "exact_source_table_month_lattice_calibration"
    assert audit["source_table_comparison"] == "source_over_ambiguous_visual"
    assert audit["visual_owned_month_rule_hits"] == 13
    assert audit["visual_residual_shift_months"] >= 0.45
    assert audit["corroborated_by_source_table_geometry"] is False
    assert audit["ambiguous_visual_geometry_superseded"] is True


def test_candidate_b_visual_source_geometry_allows_sub_cell_calibration_drift() -> None:
    visual_edges = [
        72.85045993160904,
        99.8053271473418,
        126.76019436307456,
        153.71506157880734,
        180.6699287945401,
        207.62479601027286,
        234.57966322600564,
        261.5345304417384,
        288.48939765747116,
        315.4442648732039,
        342.39913208893665,
        369.3539993046694,
        396.30886652040226,
    ]
    source_edges = [
        73.0,
        100.0,
        127.0,
        154.0,
        181.0,
        208.0,
        235.0,
        261.0,
        288.0,
        316.0,
        343.0,
        370.0,
        398.5,
    ]

    def month_cols(edges: list[float]) -> list[dict[str, object]]:
        return [
            {
                "header": str(month),
                "bbox": [edges[month - 1], 0.0, edges[month], 1.0],
            }
            for month in range(1, 13)
        ]

    assert repayment_mod._month_geometry_planes_agree(month_cols(visual_edges), month_cols(source_edges))
    assert not repayment_mod._month_geometry_planes_agree(
        month_cols(visual_edges),
        month_cols([edge + 13.5 for edge in source_edges]),
    )

    within_one_tenth = list(source_edges)
    within_one_tenth[6] += 27.0 * 0.099
    beyond_one_tenth = list(source_edges)
    beyond_one_tenth[6] += 27.0 * 0.101
    assert repayment_mod._month_geometry_planes_agree(
        month_cols(source_edges),
        month_cols(within_one_tenth),
    )
    assert not repayment_mod._month_geometry_planes_agree(
        month_cols(source_edges),
        month_cols(beyond_one_tenth),
    )


def test_candidate_b_bundle_augmentation_plumbs_detached_continuation_geometry() -> None:
    joined_lines = _source_owned_continuation_lines()
    base_lines = [line for line in joined_lines if line.get("source_logical_page") == 1]
    continuation_lines = [
        {
            **line,
            "bbox": [
                line["bbox"][0],
                line["bbox"][1] - 100.0,
                line["bbox"][2],
                line["bbox"][3] - 100.0,
            ],
        }
        for line in joined_lines
        if line.get("source_logical_page") == 2
    ]
    geometry = _continuation_source_table_geometry()
    domain_specific = domain_specific_with_page_bundles(
        page_evidence_bundle(
            1,
            page_width=320,
            page_height=100,
            micro_grid_evidence={
                "page": 1,
                "page_width": 320,
                "page_height": 100,
                "lines": base_lines,
                "tokens": base_lines,
            },
        ),
        page_evidence_bundle(
            2,
            page_width=320,
            page_height=100,
            micro_grid_evidence={
                "page": 2,
                "page_width": 320,
                "page_height": 100,
                "lines": continuation_lines,
                "tokens": continuation_lines,
                "source_table_geometry": [geometry],
            },
        ),
    )

    augment_credit_repayment_evidence_bundles(domain_specific)
    first_evidence = domain_specific["_page_evidence_bundles"][0]["micro_grid_evidence"]
    copied = first_evidence["continuation_source_table_geometry_by_page"]["2"]
    domain_specific["_page_evidence_bundles"][1]["micro_grid_evidence"]["source_table_geometry"][0]["table_id"] = (
        "mutated_after_augmentation"
    )

    assert first_evidence["continuation_logical_pages"] == [2]
    assert copied[0]["table_id"] == "pt_2_0"
    materialize_credit_repayment_micro_grids_from_bundles(
        domain_specific,
        enable_candidate_b_amount_pairing=True,
    )
    grids = micro_grid_structures_from_bundles(domain_specific)
    rows = [
        row
        for grid in grids
        for row in records_from_micro_grid_dict(
            grid,
            accept_exact_row_numeric_status=True,
        )
    ]
    continuation = [row for row in rows if row["year"] == 2019]

    assert len(continuation) == 12
    assert all(row["status"] == "N" for row in continuation)


def test_candidate_b_bundle_augmentation_shifts_continuation_word_tokens() -> None:
    joined_lines = _source_owned_continuation_lines()
    base_lines = [line for line in joined_lines if line.get("source_logical_page") == 1]
    continuation_lines = [
        {
            **line,
            "content": ("N N 福 N 限" if str(line.get("content") or "") == "N" * 12 else line["content"]),
            "bbox": [
                line["bbox"][0],
                line["bbox"][1] - 100.0,
                line["bbox"][2],
                line["bbox"][3] - 100.0,
            ],
        }
        for line in joined_lines
        if line.get("source_logical_page") == 2
    ]
    continuation_tokens = [
        {
            "token_id": f"continuation_status_{month}",
            "content": "N",
            "bbox": [
                24.0 + month * 20.0,
                12.0,
                36.0 + month * 20.0,
                20.0,
            ],
            "confidence": 0.99,
            "source": "rapidocr_word",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
    ]
    domain_specific = domain_specific_with_page_bundles(
        page_evidence_bundle(
            1,
            page_width=320,
            page_height=100,
            micro_grid_evidence={
                "page": 1,
                "page_width": 320,
                "page_height": 100,
                "lines": base_lines,
                "tokens": base_lines,
            },
        ),
        page_evidence_bundle(
            2,
            page_width=320,
            page_height=100,
            micro_grid_evidence={
                "page": 2,
                "page_width": 320,
                "page_height": 100,
                "lines": continuation_lines,
                "tokens": continuation_tokens,
                "source_table_geometry": [_continuation_source_table_geometry()],
            },
        ),
    )

    augment_credit_repayment_evidence_bundles(domain_specific)
    first_evidence = domain_specific["_page_evidence_bundles"][0]["micro_grid_evidence"]
    shifted_tokens = [token for token in first_evidence["tokens"] if token.get("source_logical_page") == 2]

    assert len(shifted_tokens) == 12
    assert all(token["coordinate_status"] == "cross_page_y_shift" for token in shifted_tokens)
    assert all(token["bbox"][1] >= 100.0 for token in shifted_tokens)

    materialize_credit_repayment_micro_grids_from_bundles(
        domain_specific,
        enable_candidate_b_amount_pairing=True,
    )
    rows = [
        row
        for grid in micro_grid_structures_from_bundles(domain_specific)
        for row in records_from_micro_grid_dict(
            grid,
            accept_exact_row_numeric_status=True,
        )
        if row["year"] == 2019
    ]

    assert len(rows) == 12
    assert all(row["status"] == "N" for row in rows)


def test_candidate_b_bundle_materializer_plumbs_base_source_geometry() -> None:
    page = 10
    geometry = _base_source_geometry(page=page)
    domain_specific = domain_specific_with_page_bundles(
        page_evidence_bundle(
            page,
            page_width=320,
            page_height=100,
            micro_grid_evidence={
                "page": page,
                "page_width": 320,
                "page_height": 100,
                "lines": _source_owned_base_lines(year=2020, status_text="N" * 12),
                "tokens": [],
                "source_table_geometry": [geometry],
            },
        )
    )

    materialize_credit_repayment_micro_grids_from_bundles(
        domain_specific,
        enable_candidate_b_amount_pairing=True,
    )
    [grid] = micro_grid_structures_from_bundles(domain_specific)
    rows = records_from_micro_grid_dict(
        grid,
        accept_exact_row_numeric_status=True,
    )

    assert len(rows) == 12
    assert all(row["status"] == "N" for row in rows)
    assert all(
        ref["geometry_provenance"]["selection_basis"] == "source_table_year_plus_twelve_ownership"
        for row in rows
        for ref in row["source_cell_refs"]
    )
    assert "PRIVATE_SOURCE_VALUE" not in json.dumps(grid, ensure_ascii=False)


def _noisy_active_quarter_source_fixture() -> tuple[list[dict], list[dict]]:
    lines = [
        {
            "content": "2019\u5e7404\u6708-2024\u5e7403\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            "bbox": [80.0, 5.0, 260.0, 10.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [40.0, 15.0, 280.0, 20.0],
            "confidence": 1.0,
        },
        {
            # Representative p17/p18 line: one unrelated glyph makes the
            # merged OCR row unusable, and an out-of-range numeric glyph must
            # not contaminate the active January-March cells.
            "content": "N N \u798f N \u9650 2",
            "bbox": [40.0, 28.0, 240.0, 36.0],
            "confidence": 0.55,
        },
        {"content": "2024", "bbox": [20.0, 37.0, 38.0, 42.0], "confidence": 1.0},
        {"content": "0 0 0", "bbox": [40.0, 44.0, 100.0, 49.0], "confidence": 1.0},
    ]
    tokens = [
        {
            "token_id": f"status_{month}",
            "content": "N",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                28.0,
                56.0 + (month - 1) * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": "rapidocr_word",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 4)
    ]
    tokens.extend(
        [
            {
                "token_id": "status_noise_month_3",
                "content": "\u798f",
                "bbox": [82.0, 28.0, 87.0, 36.0],
                "confidence": 0.35,
                "source": "rapidocr_word",
                "coordinate_system": "pdf_points_top_left",
            },
            {
                "token_id": "inactive_numeric_noise",
                "content": "2",
                "bbox": [224.0, 28.0, 236.0, 36.0],
                "confidence": 0.45,
                "source": "rapidocr_word",
                "coordinate_system": "pdf_points_top_left",
            },
        ]
    )
    return lines, tokens


def test_candidate_b_noisy_active_quarter_recovers_only_exact_source_owned_n_zero_cells() -> None:
    page = 17
    lines, tokens = _noisy_active_quarter_source_fixture()
    geometry = _base_source_geometry(page=page)

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [geometry]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    rows = [row for row in rows if row["year"] == 2024]

    assert [(row["year"], row["month"]) for row in rows] == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
    ]
    assert all(row["status"] == "N" for row in rows)
    assert all(row["overdue_amount"] == "0" for row in rows)
    assert all(row["audit"]["row_repair"] == "exact_active_source_lattice_token_ownership" for row in rows)
    for row in rows:
        refs = {ref["field_name"]: ref for ref in row["source_cell_refs"]}
        assert set(refs) == {"status", "overdue_amount"}
        assert refs["status"]["col"] == row["month"]
        assert refs["overdue_amount"]["col"] == row["month"]
        assert all(ref["geometry_scope"] == "cell" for ref in refs.values())
        assert all(
            ref["geometry_provenance"]["selection_basis"] == "source_table_year_plus_twelve_ownership"
            for ref in refs.values()
        )
        assert all(ref["geometry_provenance"]["value_inputs_used"] is False for ref in refs.values())
    assert "PRIVATE_SOURCE_VALUE" not in json.dumps(
        extracted["micro_grid"],
        ensure_ascii=False,
    )


def test_candidate_b_incomplete_status_line_does_not_block_exact_source_owned_tokens() -> None:
    """A partial line candidate must not mask a complete exact token row.

    Real scanned reports can expose one isolated status as a usable OCR line
    while retaining all twelve independently positioned word tokens.  The
    value-free source lattice proves each token's month; the incomplete line
    is weaker evidence and must not disable that deterministic repair.
    """

    page = 19
    lines = _source_owned_base_lines(year=2023, status_text="N")
    tokens = [
        {
            "token_id": f"status_{month}",
            "content": "N",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                28.0,
                56.0 + (month - 1) * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": "rapidocr_word",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert len(rows) == 12
    assert all(row["status"] == "N" for row in rows)
    assert all(row["overdue_amount"] == "0" for row in rows)
    assert all(row["audit"]["row_repair"] == "exact_active_source_lattice_token_ownership" for row in rows)


def test_candidate_b_exact_status_atom_survives_same_value_line_duplicate() -> None:
    """A canonical duplicate must not degrade its identical immutable atom."""

    page = 19
    lines = _source_owned_base_lines(year=2023, status_text="N" * 12)
    tokens = []
    for month in range(1, 13):
        bbox = [
            44.0 + (month - 1) * 20.0,
            28.0,
            56.0 + (month - 1) * 20.0,
            36.0,
        ]
        tokens.extend(
            [
                {
                    "token_id": f"canonical_status_{month}",
                    "content": "N",
                    "bbox": bbox,
                    "confidence": 0.91,
                    "source": "canonical_page_line",
                    "coordinate_system": "pdf_points_top_left",
                },
                {
                    "token_id": f"exact_status_{month}",
                    "content": "N",
                    "bbox": bbox,
                    "confidence": 0.99,
                    "source": "exact_native_source_table_status_cell",
                    "coordinate_system": "pdf_points_top_left",
                },
            ]
        )

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert len(rows) == 12
    assert all(row["status"] == "N" for row in rows)
    assert all(row["overdue_amount"] == "0" for row in rows)
    assert all(row["audit"]["row_repair"] == "exact_active_source_lattice_token_ownership" for row in rows)


@pytest.mark.parametrize(
    "exact_amount_source",
    (
        "exact_native_source_table_amount_cell",
        "exact_corrected_source_table_amount_cell",
    ),
)
def test_candidate_b_exact_zero_atoms_fill_only_missing_amount_cells(
    exact_amount_source: str,
) -> None:
    page = 17
    lines, status_tokens = _noisy_active_quarter_source_fixture()
    lines = [line for line in lines if line["content"] != "0 0 0"]
    for token in status_tokens:
        if token["token_id"].startswith("status_"):
            token["source"] = "exact_native_source_table_status_cell"
    zero_tokens = [
        {
            "token_id": f"exact_amount_{month}",
            "content": "0",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                43.0,
                56.0 + (month - 1) * 20.0,
                49.0,
            ],
            "confidence": 0.99,
            "source": exact_amount_source,
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 4)
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=[*status_tokens, *zero_tokens],
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    rows = [row for row in rows if row["year"] == 2024]

    assert [(row["month"], row["status"], row["overdue_amount"]) for row in rows] == [
        (1, "N", "0"),
        (2, "N", "0"),
        (3, "N", "0"),
    ]
    assert extracted["micro_grid"]["audit"]["candidate_b_amount_pairing"]["2024"]["exact_source_zero_months"] == [
        1,
        2,
        3,
    ]
    amount_cells = [
        cell
        for row in extracted["micro_grid"]["cells"]
        for cell in row
        if cell["role"] == "overdue_amount" and cell["col_index"] in {1, 2, 3}
    ]
    assert all(cell["recognition_source"] == "exact_source_zero_cell" for cell in amount_cells)


def test_candidate_b_exact_zero_atom_does_not_override_exact_nonzero_amount() -> None:
    page = 17
    lines, status_tokens = _noisy_active_quarter_source_fixture()
    next(line for line in lines if line["content"] == "0 0 0")["content"] = "50 0 0"
    for token in status_tokens:
        if token["token_id"].startswith("status_"):
            token["source"] = "exact_native_source_table_status_cell"
    zero_token = {
        "token_id": "contradicting_exact_zero",
        "content": "0",
        "bbox": [44.0, 43.0, 56.0, 49.0],
        "confidence": 0.99,
        "source": "exact_native_source_table_amount_cell",
        "coordinate_system": "pdf_points_top_left",
    }

    extracted = extract_credit_repayment_records(
        lines,
        tokens=[*status_tokens, zero_token],
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    row = next(
        row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if row["year"] == 2024 and row["month"] == 1
    )

    assert row["overdue_amount"] == "50"
    assert row["extraction_status"] == "review"
    assert "exact_source_zero_months" not in extracted["micro_grid"]["audit"]["candidate_b_amount_pairing"]["2024"]


@pytest.mark.parametrize(
    "failure_mode",
    (
        "missing_source_geometry",
        "competing_source_geometry",
        "merged_token_only",
    ),
)
def test_candidate_b_noisy_active_quarter_repair_fails_closed(
    failure_mode: str,
) -> None:
    page = 17
    lines, tokens = _noisy_active_quarter_source_fixture()
    geometry = _base_source_geometry(page=page)
    source_tables = [geometry]
    if failure_mode == "missing_source_geometry":
        source_tables = []
    elif failure_mode == "competing_source_geometry":
        duplicate = deepcopy(geometry)
        duplicate["table_id"] = "pt_17_duplicate"
        source_tables.append(duplicate)
    else:
        tokens = [
            {
                "token_id": "merged_noisy_status",
                "content": "N N \u798f N",
                "bbox": [40.0, 28.0, 100.0, 36.0],
                "confidence": 0.55,
                "source": "rapidocr_line",
                "coordinate_system": "pdf_points_top_left",
            }
        ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): source_tables},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    rows = [row for row in rows if row["year"] == 2024]

    assert [(row["year"], row["month"]) for row in rows] == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
    ]
    assert all(row["status"] == "unknown" for row in rows)
    assert all(
        row["overdue_amount"] == ("0" if failure_mode == "merged_token_only" else None)
        for row in rows
    )
    assert all(row["extraction_status"] == "review" for row in rows)


@pytest.mark.parametrize("failure_mode", ("missing_active_token", "duplicate_active_token"))
def test_candidate_b_source_owned_missing_or_duplicate_status_is_not_neighbor_imputed(
    failure_mode: str,
) -> None:
    page = 17
    lines, tokens = _noisy_active_quarter_source_fixture()
    if failure_mode == "missing_active_token":
        tokens = [token for token in tokens if token["token_id"] != "status_2"]
    else:
        duplicate_token = deepcopy(next(token for token in tokens if token["token_id"] == "status_2"))
        duplicate_token["token_id"] = "status_2_duplicate"
        tokens.append(duplicate_token)

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = {
        row["month"]: row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if row["year"] == 2024
    }

    assert rows[1]["status"] == "N"
    assert rows[3]["status"] == "N"
    assert rows[2]["status"] == "unknown"
    assert rows[2]["overdue_amount"] == "0"
    assert rows[2].get("recognition_source") != "row_neighbor_consensus"
    assert rows[1]["audit"]["row_repair"] == ("exact_active_source_lattice_token_ownership")
    assert rows[3]["audit"]["row_repair"] == ("exact_active_source_lattice_token_ownership")


def test_candidate_b_source_owned_status_conflict_remains_field_local() -> None:
    page = 17
    lines, tokens = _noisy_active_quarter_source_fixture()
    conflicting_token = deepcopy(next(token for token in tokens if token["token_id"] == "status_2"))
    conflicting_token["token_id"] = "status_2_conflict"
    conflicting_token["content"] = "C"
    tokens.append(conflicting_token)

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = {
        row["month"]: row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if row["year"] == 2024
    }

    assert rows[1]["status"] == "N"
    assert rows[3]["status"] == "N"
    assert rows[2]["status"] == "unknown"
    assert rows[2]["overdue_amount"] == "0"
    assert rows[2]["extraction_status"] == "review"


def test_candidate_b_noisy_full_year_withholds_missing_status_without_erasing_amount() -> None:
    page = 17
    lines = _source_owned_base_lines(year=2023, status_text="NNNNNSNNNNNNN")
    tokens = [
        {
            "token_id": f"status_{month}",
            "content": "N",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                28.0,
                56.0 + (month - 1) * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": "rapidocr_word",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
        if month != 6
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert len(rows) == 12
    by_month = {row["month"]: row for row in rows}
    assert all(row["status"] == "N" for month, row in by_month.items() if month != 6)
    assert by_month[6]["status"] == "unknown"
    assert by_month[6]["overdue_amount"] == "0"


def test_candidate_b_partial_status_band_disambiguates_singleton_year_boundary() -> None:
    """A p17/p18-shaped partial status line must retain exact table ownership."""

    page = 17
    lines = [
        {
            "content": "2024\u5e7401\u6708-2024\u5e7403\u6708\u7684\u8fd8\u6b3e\u8bb0\u5f55",
            "bbox": [80.0, 5.0, 260.0, 10.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [40.0, 15.0, 280.0, 20.0],
            "confidence": 1.0,
        },
        {"content": "NNN", "bbox": [40.0, 30.0, 100.0, 35.0], "confidence": 1.0},
        {"content": "2024", "bbox": [20.0, 41.0, 38.0, 48.0], "confidence": 1.0},
        {"content": "0 0 0", "bbox": [40.0, 44.0, 100.0, 49.0], "confidence": 1.0},
    ]
    geometry = _continuation_source_table_geometry(
        table_id="pt_17_0",
        logical_page=page,
        row_edges=(25.0, 40.0, 52.0, 64.0),
        status_row=0,
        amount_row=1,
        year_anchor_row=1,
        year_row_span=1,
        active_months=(1, 2, 3),
    )

    extracted = extract_credit_repayment_records(
        lines,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [geometry]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert [(row["year"], row["month"]) for row in rows] == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
    ]
    assert all(row["status"] == "N" for row in rows)
    assert all(row["overdue_amount"] == "0" for row in rows)
    for row in rows:
        refs = {ref["field_name"]: ref for ref in row["source_cell_refs"]}
        assert refs["status"]["geometry_scope"] == "cell"
        assert refs["status"]["col"] == row["month"]
        assert refs["overdue_amount"]["col"] == row["month"]
        assert refs["status"]["geometry_provenance"]["table_id"] == "pt_17_0"


@pytest.mark.parametrize("missing_amount", (False, True))
@pytest.mark.parametrize("damaged_sibling_geometry", (False, True))
def test_candidate_b_singleton_year_disambiguation_checks_status_before_materializing(
    missing_amount: bool,
    damaged_sibling_geometry: bool,
) -> None:
    """Lin-like singleton years must not let late row ownership bypass repair.

    This is a minimized business-shaped fixture, not a claim that the simulated
    glyph boxes below were recovered from the original PDF. The source table
    has two geometrically plausible row pairs at the singleton year boundary;
    the canonical status band disambiguates them. September's canonical M and
    independently owned raw N conflict, even when September's amount is absent.
    """

    page = 19
    lines = _source_owned_base_lines(
        year=2021,
        status_text="N N N N N N N N M N N N",
    )
    lines[3]["bbox"] = [20.0, 41.0, 38.0, 48.0]
    # Keep amount positions explicit: deleting one glyph from a merged line
    # must not accidentally change the spacing of all following months.
    lines = lines[:4] + [
        {
            "content": "0",
            "bbox": [44.0 + (month - 1) * 20.0, 44.0, 56.0 + (month - 1) * 20.0, 49.0],
            "confidence": 1.0,
        }
        for month in range(1, 13)
        if not (missing_amount and month == 9)
    ]
    geometry = _continuation_source_table_geometry(
        table_id="pt_19_0",
        logical_page=page,
        row_edges=(25.0, 40.0, 52.0, 64.0),
        status_row=0,
        amount_row=1,
        year_anchor_row=1,
        year_row_span=1,
    )
    if damaged_sibling_geometry:
        geometry["cell_bboxes"][0][4] = [
            geometry["vertical_lines"][4], 25.0, geometry["vertical_lines"][6], 40.0
        ]
        geometry["cell_bboxes"][0][5] = None
        geometry["cell_geometry_status"][0][5] = "derived"
        geometry["cell_spans"].append(
            {
                "row": 0,
                "col": 4,
                "row_span": 1,
                "col_span": 2,
                "bbox": geometry["cell_bboxes"][0][4],
            }
        )
    tokens = [
        {
            "token_id": f"raw_status_{month}",
            "content": "N",
            "bbox": [44.0 + (month - 1) * 20.0, 28.0, 56.0 + (month - 1) * 20.0, 36.0],
            "confidence": 0.99,
            "source": "exact_native_source_table_status_cell",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
    ]
    tokens += [
        {
            "token_id": f"raw_amount_{month}",
            "content": "0",
            "bbox": [44.0 + (month - 1) * 20.0, 44.0, 56.0 + (month - 1) * 20.0, 49.0],
            "confidence": 0.99,
            "source": "exact_native_source_table_amount_cell",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
        if not (missing_amount and month == 9)
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [geometry]},
    )
    rows = {
        row["month"]: row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"],
            accept_exact_row_numeric_status=True,
        )
        if row["year"] == 2021
    }

    assert set(rows) == set(range(1, 13))
    assert rows[9]["status"] == "unknown"
    assert rows[9]["extraction_status"] == "review"
    assert rows[9]["overdue_amount"] == (None if missing_amount else "0")
    assert rows[9]["audit"]["unresolved_fields"] == (
        ["status_code", "overdue_amount"] if missing_amount else ["status_code"]
    )
    assert rows[9]["audit"]["observations"] == {
        "fallback": ["M"],
        "exact_source_cell": ["N"],
        "ordinary": [],
    }
    for month in set(range(1, 13)) - {9, 4, 5}:
        assert rows[month]["status"] == "N"
        assert rows[month]["overdue_amount"] == "0"
    assert rows[4]["status"] == ("unknown" if damaged_sibling_geometry else "N")
    assert rows[5]["status"] == ("unknown" if damaged_sibling_geometry else "N")
    [status_ref] = [ref for ref in rows[9]["source_cell_refs"] if ref["field_name"] == "status"]
    assert status_ref["col"] == 9
    assert status_ref["geometry_provenance"]["table_id"] == "pt_19_0"
    assert status_ref["geometry_provenance"]["status_row_index"] == 0


def test_candidate_b_repairs_exact_status_cells_without_requiring_whole_row_geometry() -> None:
    """One merged source cell must not discard ten independent sibling cells."""

    page = 19
    lines = _source_owned_base_lines(
        year=2023,
        status_text="N N N N N 福 N N N N N N",
    )
    geometry = _base_source_geometry(page=page)
    vertical_lines = geometry["vertical_lines"]
    status_boxes = geometry["cell_bboxes"][0]
    status_states = geometry["cell_geometry_status"][0]
    status_boxes[6] = [
        vertical_lines[6],
        25.0,
        vertical_lines[8],
        40.0,
    ]
    status_boxes[7] = None
    status_states[6] = "exact"
    status_states[7] = "derived"
    geometry["cell_spans"].append(
        {
            "row": 0,
            "col": 6,
            "row_span": 1,
            "col_span": 2,
            "bbox": status_boxes[6],
        }
    )
    tokens = [
        {
            "token_id": f"status_{month}",
            "content": "N",
            "bbox": [
                24.0 + month * 20.0,
                28.0,
                36.0 + month * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": "exact_native_source_table_status_cell",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in (*range(1, 6), *range(8, 13))
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [geometry]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    by_month = {row["month"]: row for row in rows if row["year"] == 2023}

    assert set(by_month) == set(range(1, 13))
    for month in (*range(1, 6), *range(8, 13)):
        assert by_month[month]["status"] == "N"
        assert by_month[month]["recognition_source"] == ("candidate_b_source_cell_repair")
        [status_ref] = [ref for ref in by_month[month]["source_cell_refs"] if ref["field_name"] == "status"]
        assert status_ref["geometry_scope"] == "cell"
        assert status_ref["col"] == month
    for month in (6, 7):
        assert by_month[month]["status"] == "unknown"
        assert by_month[month].get("recognition_source") != "row_neighbor_consensus"
        [status_ref] = [ref for ref in by_month[month]["source_cell_refs"] if ref["field_name"] == "status"]
        assert status_ref["geometry_scope"] == "logical_page"
        assert status_ref["geometry_status"] == "unresolved"


def test_candidate_b_partial_word_tokens_do_not_degrade_a_complete_clean_row() -> None:
    page = 17
    lines = _source_owned_base_lines(year=2023, status_text="N" * 12)
    tokens = [
        {
            "token_id": f"status_{month}",
            "content": "N",
            "bbox": [
                44.0 + (month - 1) * 20.0,
                28.0,
                56.0 + (month - 1) * 20.0,
                36.0,
            ],
            "confidence": 0.99,
            "source": "rapidocr_word",
            "coordinate_system": "pdf_points_top_left",
        }
        for month in range(1, 13)
        if month != 6
    ]

    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )

    assert len(rows) == 12
    assert all(row["status"] == "N" for row in rows)


def _explicit_field_ref(grid: dict, cell: dict, field_name: str) -> dict:
    return {
        "page": grid["page"],
        "logical_page": grid["page"],
        "grid_id": grid["grid_id"],
        "row": cell["row_index"],
        "col": cell["col_index"],
        "field_name": field_name,
        "geometry_scope": "cell",
        "geometry_status": "exact",
        "coordinate_system": "pdf_points_top_left",
        "bbox": list(cell["bbox"]),
    }


def test_empty_status_field_materialization_is_scoped_to_candidate_b() -> None:
    grid = _candidate_b_row_role_grid("", "150")

    [generic_record] = records_from_micro_grid_dict(grid)
    assert generic_record["status"] == "unknown"
    assert generic_record["overdue_amount"] is None
    assert generic_record["source"] == "repayment_grid_date_range_placeholder"
    [record] = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=True)
    assert record["status"] == "unknown"
    assert record["overdue_amount"] == "150"
    assert record["extraction_status"] == "review"


def test_candidate_b_amount_only_direct_record_requires_its_field_geometry_mask() -> None:
    lines = [
        {"content": "2024年01月-2024年03月的还款记录", "bbox": [80.0, 5.0, 260.0, 10.0]},
        {"content": "月份模糊", "bbox": [40.0, 18.0, 280.0, 26.0]},
        {"content": "N", "bbox": [40.0, 28.0, 60.0, 36.0]},
        {"content": "C", "bbox": [80.0, 28.0, 100.0, 36.0]},
        {"content": "2024", "bbox": [20.0, 37.0, 38.0, 42.0]},
        {"content": "0 150 0", "bbox": [40.0, 44.0, 100.0, 49.0]},
    ]
    extracted = extract_credit_repayment_records(
        lines,
        page=17,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
    )
    amount_cell = next(
        cell
        for row in extracted["micro_grid"]["cells"]
        for cell in row
        if cell["role"] == "overdue_amount" and cell["col_index"] == 2
    )
    assert amount_cell["text"] == "150"
    pairing = extracted["micro_grid"]["audit"]["candidate_b_amount_pairing"]["2024"]
    assert pairing["cell_status_by_month"]["2"] == "exact"
    assert amount_cell["recognition_audit"]["field_geometry_exact"] is False
    assert all(row["month"] != 2 for row in extracted["repayment_records"])
    february = next(
        row
        for row in records_from_micro_grid_dict(
            extracted["micro_grid"], accept_exact_row_numeric_status=True,
        )
        if row["month"] == 2
    )
    assert february["status"] == "unknown"
    assert february["overdue_amount"] is None


@pytest.mark.parametrize("status", ("", "N"))
def test_candidate_b_exact_amount_does_not_require_status_field_geometry(status: str) -> None:
    grid = _candidate_b_row_role_grid(status, "150")
    status_cell = grid["cells"][0][1]
    status_cell["recognition_audit"]["field_geometry_exact"] = False
    [record] = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=True)
    assert record["status"] == "unknown"
    assert record["overdue_amount"] == "150"
    assert record["extraction_status"] == "review"


@pytest.mark.parametrize("amount_mask", (False, "true"))
@pytest.mark.parametrize("strict_argument", (False, True))
def test_candidate_b_status_does_not_authorize_an_unowned_amount_field(
    amount_mask: object, strict_argument: bool,
) -> None:
    grid = _candidate_b_row_role_grid("N", "150")
    amount_cell = grid["cells"][1][0]
    amount_cell["recognition_audit"] = {"field_geometry_exact": amount_mask}
    [record] = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=strict_argument)
    assert record["status"] == "N"
    assert record["overdue_amount"] is None


@pytest.mark.parametrize("bad_ref", ("wrong_row", "wrong_month", "wrong_page_type", "rejected", "nonfinite"))
def test_candidate_b_amount_materialization_validates_its_own_ref(bad_ref: str) -> None:
    grid = _candidate_b_row_role_grid("", "150")
    amount_cell = grid["cells"][1][0]
    ref = _explicit_field_ref(grid, amount_cell, "overdue_amount")
    if bad_ref == "wrong_row":
        ref["row"] += 1
    elif bad_ref == "wrong_month":
        ref["col"] += 1
    elif bad_ref == "wrong_page_type":
        ref["page"] = True
    elif bad_ref == "rejected":
        ref["geometry_status"] = "rejected"
    else:
        ref["bbox"][0] = float("nan")
    amount_cell["recognition_audit"] = {"field_geometry_exact": True, "source_ref": ref}
    [record] = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=True)
    assert record["status"] == "unknown"
    assert record["overdue_amount"] is None


def test_candidate_b_exact_field_refs_supersede_only_stale_global_visual_rejection() -> None:
    grid = _candidate_b_row_role_grid("", "150")
    grid["audit"]["visual_month_geometry"] = {"usable": False, "source": "rejected_month_geometry"}
    for cell, field_name in ((grid["cells"][0][1], "status"), (grid["cells"][1][0], "overdue_amount")):
        cell.setdefault("recognition_audit", {}).update(
            {"field_geometry_exact": True, "source_ref": _explicit_field_ref(grid, cell, field_name)}
        )
    [record] = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=True)
    assert record["status"] == "unknown"
    assert record["overdue_amount"] == "150"

    del grid["cells"][1][0]["recognition_audit"]["source_ref"]
    [unproved] = records_from_micro_grid_dict(grid, accept_exact_row_numeric_status=True)
    assert unproved["overdue_amount"] is None


@pytest.mark.parametrize("status_mode", ("absent", "all_conflicted"))
@pytest.mark.parametrize("amount_mode", ("printed_line", "exact_atoms"))
def test_candidate_b_geometry_only_status_row_retains_independent_amounts(
    status_mode: str,
    amount_mode: str,
) -> None:
    """A three-month business excerpt must not lose amounts with its status OCR."""
    page = 17
    lines = _source_owned_base_lines(year=2024, status_text="NNN")
    lines[0]["content"] = "2024年01月-2024年03月的还款记录"
    lines[2]["bbox"] = [40.0, 28.0, 100.0, 36.0]
    lines[-1].update(content="0 0 0", bbox=[40.0, 44.0, 100.0, 49.0])
    tokens = []
    if status_mode == "absent":
        lines.pop(2)
    else:
        tokens.extend(
            {
                "token_id": f"exact_status_conflict_{month}",
                "content": "C",
                "bbox": [24.0 + month * 20.0, 28.0, 36.0 + month * 20.0, 36.0],
                "confidence": 0.99,
                "source": "exact_native_source_table_status_cell",
                "coordinate_system": "pdf_points_top_left",
            }
            for month in range(1, 4)
        )
    if amount_mode == "exact_atoms":
        lines = [line for line in lines if line["content"] != "0 0 0"]
        tokens.extend(
            {
                "token_id": f"exact_amount_{month}",
                "content": "0",
                "bbox": [24.0 + month * 20.0, 44.0, 36.0 + month * 20.0, 49.0],
                "confidence": 0.99,
                "source": "exact_native_source_table_amount_cell",
                "coordinate_system": "pdf_points_top_left",
            }
            for month in range(1, 4)
        )
    extracted = extract_credit_repayment_records(
        lines,
        tokens=tokens,
        page=page,
        page_width=320,
        page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    projected = records_from_micro_grid_dict(extracted["micro_grid"], accept_exact_row_numeric_status=True)
    for records in (extracted["repayment_records"], projected):
        assert [(row["year"], row["month"]) for row in records] == [(2024, 1), (2024, 2), (2024, 3)]
        assert all(row["status"] == "unknown" for row in records)
        assert all(row["overdue_amount"] == "0" for row in records)
        assert all(row["extraction_status"] == "review" for row in records)


@pytest.mark.parametrize("exact_field", ("status", "overdue_amount"))
def test_candidate_b_conflicting_planes_keep_status_and_amount_masks_independent(monkeypatch, exact_field: str) -> None:
    monkeypatch.setattr(
        repayment_mod,
        "_visual_month_col_bands",
        lambda *_args, **_kwargs: _owned_visual_cols(shift=20.0, owned_rule_hits=13, residual_shift_months=0.1),
    )
    page = 12
    lines = _source_owned_base_lines(year=2024, status_text="NNN")
    lines[0]["content"] = "2024年01月-2024年03月的还款记录"
    lines[2]["bbox"] = [40.0, 28.0, 100.0, 36.0]
    lines[-1].update(content="0", bbox=[44.0, 44.0, 56.0, 49.0])
    token = {
        "token_id": f"exact_{exact_field}_1",
        "content": "N" if exact_field == "status" else "0",
        "bbox": [44.0, 28.0, 56.0, 36.0] if exact_field == "status" else [44.0, 44.0, 56.0, 49.0],
        "confidence": 0.99,
        "source": "exact_native_source_table_status_cell" if exact_field == "status" else "exact_native_source_table_amount_cell",
        "coordinate_system": "pdf_points_top_left",
    }
    extracted = extract_credit_repayment_records(
        lines, tokens=[token], page=page, page_width=320, page_height=100,
        enable_candidate_b_amount_pairing=True,
        source_table_geometry_by_page={str(page): [_base_source_geometry(page=page)]},
    )
    rows = records_from_micro_grid_dict(extracted["micro_grid"], accept_exact_row_numeric_status=True)
    first = next(row for row in rows if row["month"] == 1)
    assert first["status"] == ("N" if exact_field == "status" else "unknown")
    assert first["overdue_amount"] == (None if exact_field == "status" else "0")
    assert all(row["status"] == "unknown" and row["overdue_amount"] is None for row in rows if row["month"] != 1)


def _registered_test_line(idx: int, text: str, raw_bbox: list[float], *, origin: int = 3) -> dict:
    scale_x, scale_y, offset_x, offset_y, stack = 1.25, 0.75, 8.0, 12.0, 100.0
    return {
        "idx": idx,
        "text": text,
        "bbox": [
            offset_x + raw_bbox[0] * scale_x,
            offset_y + stack + raw_bbox[1] * scale_y,
            offset_x + raw_bbox[2] * scale_x,
            offset_y + stack + raw_bbox[3] * scale_y,
        ],
        "confidence": 0.99,
        "source_logical_page": 2,
        "coordinate_logical_page": 2,
        "source_origin_logical_page": origin,
        "coordinate_status": "cross_page_y_shift",
        "source_bbox": list(raw_bbox),
    }


def test_synthesized_status_and_amount_rows_preserve_composed_coordinate_inverse() -> None:
    year = _registered_test_line(0, "2024", [20.0, 20.0, 38.0, 28.0])
    statuses = [
        _registered_test_line(1, "N", [44.0, 10.0, 56.0, 18.0]),
        _registered_test_line(2, "C", [64.0, 10.0, 76.0, 18.0]),
    ]
    amounts = [
        _registered_test_line(3, "0", [44.0, 30.0, 56.0, 38.0]),
        _registered_test_line(4, "0", [64.0, 30.0, 76.0, 38.0]),
    ]
    cols = [
        {"index": month, "header": str(month), "bbox": _registered_test_line(9, "", [20.0 + month * 20.0, 2.0, 40.0 + month * 20.0, 8.0])["bbox"]}
        for month in (1, 2)
    ]
    status_row = repayment_mod._candidate_b_status_row(
        statuses, year, month_cols=cols, status_charset={"N", "C"}, page=1, excluded_line_indices=set(),
    )
    amount_pair = repayment_mod._candidate_b_amount_row_pair(
        amounts, year, month_cols=cols, active_months=[1, 2], page=1, excluded_line_indices=set(),
    )
    assert status_row is not None
    for row, raw_bbox in ((status_row, [44.0, 10.0, 76.0, 18.0]), (amount_pair["line"], [44.0, 30.0, 76.0, 38.0])):
        assert row["coordinate_logical_page"] == 2
        assert row["source_origin_logical_page"] == 3
        assert row["coordinate_status"] == "cross_page_y_shift"
        assert row["source_bbox"] == pytest.approx(raw_bbox)
        visual = repayment_mod._visual_page_context(
            source_line=row, bbox=tuple(row["bbox"]), base_page=1,
            base_page_width=320, base_page_height=100, page_image=None,
            page_image_resolver=lambda origin: {"image": object(), "page_width": 320, "page_height": 100} if origin == 3 else None,
        )
        assert visual is not None
        assert visual[1] == pytest.approx(raw_bbox)
        assert visual[4] == 3
        local = repayment_mod._local_page_bbox(
            tuple(row["bbox"]), logical_page=2, base_page=1, base_page_height=100,
            coordinates_already_registered=True, coordinate_status=row["coordinate_status"],
        )
        assert local == pytest.approx([row["bbox"][0], row["bbox"][1] - 100, row["bbox"][2], row["bbox"][3] - 100])

    statuses[1]["source_origin_logical_page"] = 4
    unproved = repayment_mod._candidate_b_status_row(
        statuses, year, month_cols=cols, status_charset={"N", "C"}, page=1, excluded_line_indices=set(),
    )
    assert unproved is not None and unproved["coordinate_logical_page"] == 2
    assert "source_bbox" not in unproved


def test_visual_detector_uses_one_raw_plane_and_cache_is_roi_specific(monkeypatch) -> None:
    year = _registered_test_line(0, "2024", [20.0, 20.0, 38.0, 28.0])
    status = _registered_test_line(1, "NN", [40.0, 10.0, 80.0, 18.0])
    cols = [
        {"index": month, "header": str(month), "bbox": _registered_test_line(9, "", [20.0 + month * 20.0, 2.0, 40.0 + month * 20.0, 8.0])["bbox"]}
        for month in range(1, 13)
    ]
    image = object()
    calls = []

    def detector(raw_cols, **kwargs):
        calls.append((deepcopy(raw_cols), kwargs))
        assert kwargs["page_image"] is image
        assert kwargs["year_column_bbox"] == pytest.approx([20.0, 20.0, 38.0, 28.0])
        return deepcopy(raw_cols), {"source": "vertical_rule_projection", "usable": True, "offset": 0.0}

    monkeypatch.setattr(repayment_mod, "_visual_month_col_bands", detector)
    cache = {}
    kwargs = dict(
        source_lines=[status, year], base_page=1, base_page_width=320, base_page_height=100,
        page_image=None, page_image_resolver=lambda origin: {"image": image, "page_width": 320, "page_height": 100} if origin == 3 else None,
        y0=115.75, y1=142.0, year_column_bbox=year["bbox"], cache=cache,
        require_physical_month_ownership=True,
    )
    bands, audit = repayment_mod._visual_month_col_bands_in_registered_plane(cols, **kwargs)
    assert bands == cols
    assert calls[0][0][0]["bbox"] == pytest.approx([40.0, 2.0, 60.0, 8.0])
    assert calls[0][1]["y0"] == pytest.approx(5.0)
    assert calls[0][1]["y1"] == pytest.approx(40.0)
    assert audit["source_logical_page"] == 3
    repayment_mod._visual_month_col_bands_in_registered_plane(cols, **kwargs)
    assert len(calls) == 1
    repayment_mod._visual_month_col_bands_in_registered_plane(cols, **{**kwargs, "y1": 145.75})
    assert len(calls) == 2


def test_registered_continuation_keeps_exact_source_cells_in_stack_and_refs_local() -> None:
    lines = _source_owned_continuation_lines()
    for line in lines:
        logical_page = line["source_logical_page"]
        line["coordinate_logical_page"] = logical_page
        line["source_origin_logical_page"] = logical_page
        line["source_bbox"] = list(line["bbox"])
        if logical_page == 2:
            line["coordinate_status"] = "cross_page_y_shift"
            line["source_bbox"][1] -= 100.0
            line["source_bbox"][3] -= 100.0
    extracted = _extract_source_owned_continuation(lines=lines)
    rows = records_from_micro_grid_dict(extracted["micro_grid"], accept_exact_row_numeric_status=True)
    continuation = [row for row in rows if row["year"] == 2019]
    assert len(continuation) == 12
    assert all(row["status"] == "N" and row["overdue_amount"] == "0" for row in continuation)
    for row in continuation:
        refs = {ref["field_name"]: ref for ref in row["source_cell_refs"]}
        assert refs["status"]["page"] == 2
        assert refs["status"]["bbox"][1:4:2] == [8.0, 27.0]
        assert refs["overdue_amount"]["bbox"][1:4:2] == [27.0, 45.0]
        assert row["status_bbox"][1] >= 100.0
