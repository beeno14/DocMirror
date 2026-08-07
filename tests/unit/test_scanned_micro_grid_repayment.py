# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

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
        (2021, 1, "N", "0"),
        (2021, 2, "C", "0"),
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
    direct = next(
        record
        for record in out["repayment_records"]
        if record["year"] == 2020 and record["month"] == 12
    )
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
            ]
        ],
    }

    records = records_from_micro_grid_dict(grid)

    assert records[0]["status"] == "2"
    assert records[0]["recognition_source"] == "cell_crop_consensus"


def test_candidate_b_exact_row_numeric_status_is_kept_without_false_zero_amount():
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
    assert candidate_b[0]["status"] == "2"
    assert candidate_b[0]["repayment_id"] == "mg_p3_repayment_0:2024-01"
    assert candidate_b[0]["grid_id"] == "mg_p3_repayment_0"
    assert candidate_b[0]["overdue_amount"] is None
    assert candidate_b[0]["source_cell_refs"] == [status_ref, amount_ref]


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
    assert resolved["overdue_amount"] == "0"
    assert out["micro_grid"]["audit"]["cell_crop_ocr"]["attempts"] >= 1
    assert out["micro_grid"]["audit"]["cell_crop_ocr"]["hits"] >= 1
    status_cells = [
        cell
        for row in out["micro_grid"]["cells"]
        for cell in row
        if cell["role"] == "status" and cell["col_index"] == 1
    ]
    assert status_cells[0]["recognition_source"] == "cell_crop_ocr"


def test_static_n_star_classifier_separates_canonical_glyph_shapes() -> None:
    import cv2
    import numpy as np

    def rendered(glyph: str):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        (width, height), _baseline = cv2.getTextSize(
            glyph, cv2.FONT_HERSHEY_DUPLEX, 0.72, 1
        )
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
        return repayment_mod.extract_micro_cell_glyph_template(
            image, (0, 0, 80, 80), page_width=80, page_height=80
        )

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
        (width, height), _baseline = cv2.getTextSize(
            glyph, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2
        )
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
    (star_width, star_height), _baseline = cv2.getTextSize(
        "*", cv2.FONT_HERSHEY_DUPLEX, 0.72, 1
    )
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


def test_static_status_validation_quarantines_when_page_image_is_unavailable() -> None:
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

    assert out["repayment_records"] == []
    projected = records_from_micro_grid_dict(out["micro_grid"])
    assert len(projected) == 1
    assert projected[0]["status"] == "unknown"
    assert projected[0]["extraction_status"] == "review"
    assert projected[0]["source_cell_refs"][0]["field_name"] == "status"
    assert projected[0]["source_cell_refs"][0]["geometry_scope"] == "cell"
    audit = out["micro_grid"]["audit"]["static_status_validation"]
    assert audit["attempts"] == 1
    assert audit["unresolved"] == 1
    assert audit["unavailable"] == 1


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
        (2021, 1, "N", "0"),
        (2021, 2, "C", "0"),
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
