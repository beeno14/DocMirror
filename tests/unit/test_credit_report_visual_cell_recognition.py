# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from docmirror.ocr.micro_grid.cell_recognition import (
    extract_micro_cell_glyph_template,
    normalize_allowlist_text,
    recognize_micro_cell_from_image,
)
from docmirror.ocr.micro_grid.reconstruct import equal_col_bands
from docmirror.plugins.credit_report.repayment_grid import _visual_month_col_bands


class _EmptyEngine:
    def force_recognize_regions(self, *_args, **_kwargs):
        return []


def _cell_image(character: str, *, noise: bool = False):
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 35), (210, 145), (0, 0, 0), 2)
    if character == "*":
        centre = (120, 90)
        for endpoint in ((120, 70), (120, 110), (101, 79), (139, 101), (101, 101), (139, 79)):
            cv2.line(image, centre, endpoint, (0, 0, 0), 4)
    else:
        cv2.putText(image, character, (96, 111), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
    if noise:
        cv2.line(image, (75, 125), (170, 55), (150, 150, 150), 2)
    return image


def test_status_allowlist_preserves_hash_semantics():
    assert normalize_allowlist_text("#", {"#", "*"}, max_chars=1) == "#"


def test_cell_shape_recognizes_star_without_ocr(monkeypatch):
    import docmirror.ocr.vision.rapidocr_engine as rapidocr_engine

    monkeypatch.setattr(rapidocr_engine, "get_ocr_engine", lambda: _EmptyEngine())
    image = _cell_image("*")
    result = recognize_micro_cell_from_image(
        image,
        (30, 35, 210, 145),
        page_width=240,
        page_height=180,
        allowed_charset={"*", "N", "1", "2"},
        max_chars=1,
    )
    assert result.text == "*"
    assert result.source == "cell_crop_consensus"


def test_document_template_and_shape_recognize_noisy_n(monkeypatch):
    import docmirror.ocr.vision.rapidocr_engine as rapidocr_engine

    monkeypatch.setattr(rapidocr_engine, "get_ocr_engine", lambda: _EmptyEngine())
    reference_image = _cell_image("N")
    reference = extract_micro_cell_glyph_template(
        reference_image,
        (30, 35, 210, 145),
        page_width=240,
        page_height=180,
    )
    result = recognize_micro_cell_from_image(
        _cell_image("N", noise=True),
        (30, 35, 210, 145),
        page_width=240,
        page_height=180,
        allowed_charset={"*", "N", "1", "2"},
        max_chars=1,
        reference_templates={"N": [reference]},
    )
    assert result.text == "N"


def test_visual_month_geometry_uses_table_rules_not_header_text_bbox():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(40, 281, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    legacy = equal_col_bands((55, 20, 295, 30), count=12, start_index=1, role="month")

    refined, audit = _visual_month_col_bands(
        legacy,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
    )

    assert audit["source"] == "vertical_rule_projection"
    assert refined[0]["bbox"][0] == pytest.approx(40, abs=2)
    assert refined[-1]["bbox"][2] == pytest.approx(280, abs=2)


@pytest.mark.parametrize(
    "rule_positions",
    (
        range(20, 281, 20),  # Extra year-column rule to the left.
        range(40, 301, 20),  # Extra non-month rule to the right.
    ),
)
def test_candidate_b_visual_month_geometry_prefers_validated_header_on_14_rule_near_tie(
    rule_positions,
):
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in rule_positions:
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    validated_header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        validated_header,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        max_left_shift_months=1.85,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=True,
        max_residual_shift_months=0.5,
    )

    assert audit["source"] == "vertical_rule_projection"
    assert audit["selection_basis"] == "validated_header_near_tie"
    assert audit["residual_shift_months"] < 0.5
    assert refined[0]["bbox"][0] == pytest.approx(40, abs=2)
    assert refined[-1]["bbox"][2] == pytest.approx(280, abs=2)


def test_candidate_b_visual_month_geometry_preserves_ordinary_13_rule_lattice():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(40, 281, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    validated_header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        validated_header,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        prefer_validated_header_lattice=True,
        max_residual_shift_months=0.5,
    )

    assert audit["source"] == "vertical_rule_projection"
    assert refined[0]["bbox"][0] == pytest.approx(40, abs=2)
    assert refined[-1]["bbox"][2] == pytest.approx(280, abs=2)


@pytest.mark.parametrize(
    ("source_grid", "page_width", "year_left", "month_left", "month_right"),
    (
        ("mg_p5_repayment_2", 455.0, 53.5, 81.0, 403.5),
        ("mg_p16_repayment_0", 455.0, 52.5, 80.5, 403.0),
        ("mg_p19_repayment_0", 455.0, 52.0, 79.5, 403.0),
        ("mg_p20_repayment_0", 421.0, 33.5, 61.5, 388.5),
        ("mg_p22_repayment_0", 455.0, 52.5, 80.0, 402.5),
    ),
)
def test_candidate_b_physical_year_column_owns_source_shaped_month_lattice(
    source_grid,
    page_width,
    year_left,
    month_left,
    month_right,
):
    scale = 2
    image = np.full((240, int(page_width * scale), 3), 255, dtype=np.uint8)
    rules = [year_left, *np.linspace(month_left, month_right, 13)]
    for index, x in enumerate(rules):
        # Reproduce the source-page trap: the year rule is strongest and the
        # far-right month-12 rule is weakest, so an unconstrained projection
        # prefers year+months 1..11 over physical months 1..12.
        thickness = 4 if index == 0 else 1 if index == len(rules) - 1 else 2
        cv2.line(
            image,
            (round(x * scale), 30),
            (round(x * scale), 210),
            (0, 0, 0),
            thickness,
        )
    header = equal_col_bands(
        (month_left, 20, month_right, 30),
        count=12,
        start_index=1,
        role="month",
    )

    generic, _generic_audit = _visual_month_col_bands(
        header,
        page_image=image,
        page_width=page_width,
        page_height=120,
        y0=15,
        y1=105,
        max_left_shift_months=1.85,
    )
    refined, audit = _visual_month_col_bands(
        header,
        page_image=image,
        page_width=page_width,
        page_height=120,
        y0=15,
        y1=105,
        year_column_bbox=[year_left - 25.0, 40.0, month_left - 2.0, 60.0],
        require_physical_month_ownership=True,
        max_left_shift_months=1.85,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=False,
        max_residual_shift_months=0.5,
    )

    assert generic[0]["bbox"][0] == pytest.approx(year_left, abs=2), source_grid
    assert audit["source"] == "vertical_rule_projection"
    assert audit["selection_basis"] == "year_plus_twelve_rule_ownership"
    assert audit["owned_month_rule_hits"] >= 11
    assert refined[0]["bbox"][0] == pytest.approx(month_left, abs=2)
    assert refined[-1]["bbox"][2] == pytest.approx(month_right, abs=2)


def test_candidate_b_physical_month_ownership_rejects_unowned_decisive_lattice():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(40, 281, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    estimated_header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        estimated_header,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        year_column_bbox=[5, 40, 20, 60],
        require_physical_month_ownership=True,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=False,
    )

    assert refined == []
    assert audit["source"] == "rejected_month_geometry"
    assert audit["reason"] == "physical_month_column_ownership_unproven"


def test_candidate_b_physical_ownership_repairs_header_shifted_one_month_left():
    page_width = 455.0
    scale = 2
    year_left = 53.5
    month_left = 81.0
    month_right = 403.5
    month_step = (month_right - month_left) / 12.0
    image = np.full((240, int(page_width * scale), 3), 255, dtype=np.uint8)
    for index, x in enumerate([year_left, *np.linspace(month_left, month_right, 13)]):
        thickness = 4 if index == 0 else 1 if index == 13 else 2
        cv2.line(
            image,
            (round(x * scale), 30),
            (round(x * scale), 210),
            (0, 0, 0),
            thickness,
        )
    # Reproduce p5_2: header-derived bands are a full cell left of the
    # physical months and the strongest unconstrained lattice starts at the
    # year rule.
    shifted_header = equal_col_bands(
        (month_left - month_step, 20, month_right - month_step, 30),
        count=12,
        start_index=1,
        role="month",
    )

    refined, audit = _visual_month_col_bands(
        shifted_header,
        page_image=image,
        page_width=page_width,
        page_height=120,
        y0=15,
        y1=105,
        year_column_bbox=[28.0, 40.0, month_left - 2.0, 60.0],
        require_physical_month_ownership=True,
        max_left_shift_months=1.85,
        max_right_shift_months=1.10,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=False,
        max_residual_shift_months=0.5,
    )

    assert audit["selection_basis"] == "year_plus_twelve_rule_ownership"
    assert audit["year_glyph_left_of_month_coverage"] >= 0.72
    assert refined[0]["bbox"][0] == pytest.approx(month_left, abs=2)
    assert refined[-1]["bbox"][2] == pytest.approx(month_right, abs=2)


def test_candidate_b_continuation_ownership_fails_closed_without_page_image():
    header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    shared, shared_audit = _visual_month_col_bands(
        header,
        page_image=None,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
    )
    strict, strict_audit = _visual_month_col_bands(
        header,
        page_image=None,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        year_column_bbox=[5, 40, 20, 60],
        require_physical_month_ownership=True,
        allow_unowned_header_fallback=False,
    )

    assert shared == header
    assert shared_audit["source"] == "header_geometry"
    assert strict == []
    assert strict_audit["source"] == "rejected_month_geometry"
    assert strict_audit["reason"] == "physical_month_column_ownership_unavailable"


def test_candidate_b_observed_month_centers_retain_exact_header_without_year_rule():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(40, 281, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    observed_month_centers = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        observed_month_centers,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        year_column_bbox=[5, 40, 20, 60],
        require_physical_month_ownership=True,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=True,
    )

    assert refined == observed_month_centers
    assert audit["source"] == "header_geometry"
    assert audit["reason"] == "physical_rule_ownership_unavailable_exact_header_retained"


def test_candidate_b_visual_month_geometry_retains_exact_header_after_large_visual_shift():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(20, 261, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    validated_header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        validated_header,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        max_left_shift_months=1.85,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=True,
        max_residual_shift_months=0.5,
    )

    assert refined == validated_header
    assert audit["source"] == "header_geometry"
    assert audit["usable"] is True
    assert audit["reason"] == "visual_lattice_residual_rejected_header_retained"
    assert audit["residual_shift_months"] > 0.5


def test_visual_month_geometry_still_rejects_large_shift_without_validated_header():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(20, 261, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    estimated_header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        estimated_header,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        max_left_shift_months=1.85,
        prefer_validated_header_lattice=False,
        max_residual_shift_months=0.5,
    )

    assert refined == []
    assert audit["source"] == "rejected_month_geometry"
    assert audit["usable"] is False
    assert audit["reason"] == "month_lattice_residual_shift_exceeds_bound"


def test_visual_month_geometry_does_not_retain_content_only_merged_header():
    image = np.full((120, 340, 3), 255, dtype=np.uint8)
    for x in range(20, 261, 20):
        cv2.line(image, (x, 15), (x, 105), (0, 0, 0), 2)
    estimated_merged_header = equal_col_bands(
        (40, 20, 280, 30), count=12, start_index=1, role="month"
    )

    refined, audit = _visual_month_col_bands(
        estimated_merged_header,
        page_image=image,
        page_width=340,
        page_height=120,
        y0=15,
        y1=105,
        max_left_shift_months=1.85,
        prefer_validated_header_lattice=True,
        retain_validated_header_on_residual=False,
        max_residual_shift_months=0.5,
    )

    assert refined == []
    assert audit["source"] == "rejected_month_geometry"
    assert audit["usable"] is False
    assert audit["reason"] == "month_lattice_residual_shift_exceeds_bound"


def test_visual_month_geometry_rejects_collapsed_ocr_word_before_cropping():
    image = np.full((600, 455, 3), 255, dtype=np.uint8)
    for x in range(79, 404, 27):
        cv2.line(image, (x, 210), (x, 330), (0, 0, 0), 2)
    collapsed = equal_col_bands(
        (282.64, 225.0, 298.10, 238.5),
        count=12,
        start_index=1,
        role="month",
    )

    refined, audit = _visual_month_col_bands(
        collapsed,
        page_image=image,
        page_width=455,
        page_height=600,
        y0=210,
        y1=330,
    )

    assert refined == []
    assert audit["source"] == "rejected_month_geometry"
    assert audit["usable"] is False
    assert audit["reason"] in {"implausible_page_coverage", "implausible_cell_aspect"}
