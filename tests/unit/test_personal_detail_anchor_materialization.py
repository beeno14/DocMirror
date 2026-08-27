from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.micro_grid_materialize import (
    _date_range_only_grid,
    materialize_credit_repayment_micro_grids,
)


def _anchor() -> dict:
    return {
        "text": "2021年1月至2021年12月还款记录（状态及逾期金额）",
        "bbox": [80.0, 200.0, 460.0, 216.0],
        "evidence_ids": ["sealed-range-start", "sealed-range-end"],
    }


def _identity() -> dict:
    return {
        "coordinate_system": "pdf_points_top_left",
        "coordinate_plane": "raw_logical_page",
        "source_logical_page": 9,
        "source_page": 5,
        "evidence_ids": ["sealed-range-start", "sealed-range-end"],
        "bbox": [40.0, 100.0, 230.0, 108.0],
        "date_range": [2021, 1, 2021, 12],
    }


@pytest.mark.parametrize("object_anchor", (False, True))
def test_range_only_grid_preserves_anchor_proof_without_inventing_cell_geometry(object_anchor: bool) -> None:
    raw_anchor = _anchor()
    baseline = _date_range_only_grid(raw_anchor, page=10, grid_index=2, page_width=600)
    raw_anchor["printed_anchor_identity"] = deepcopy(_identity())
    anchor = SimpleNamespace(**raw_anchor) if object_anchor else raw_anchor

    grid = _date_range_only_grid(anchor, page=10, grid_index=2, page_width=600)

    assert grid is not None and baseline is not None
    provenance = grid["audit"].pop("printed_anchor_provenance")
    assert provenance == _identity()
    assert grid == baseline
    assert grid["cells"] == grid["row_bands"] == []
    assert all(col["geometry_status"] == "unresolved" for col in grid["col_bands"])
    raw_anchor["printed_anchor_identity"]["evidence_ids"].append("later-mutation")
    assert provenance == _identity()
    provenance["bbox"][0] = -100.0
    assert raw_anchor["printed_anchor_identity"]["bbox"] == _identity()["bbox"]


@pytest.mark.parametrize("unverified_identity", (None, [], "unverified"))
def test_range_only_grid_does_not_manufacture_a_printed_anchor_identity(unverified_identity) -> None:
    anchor = {**_anchor(), "printed_anchor_identity": unverified_identity}

    grid = _date_range_only_grid(anchor, page=10, grid_index=0, page_width=600)

    assert grid is not None
    assert "printed_anchor_provenance" not in grid["audit"]


def test_real_materializer_retains_anchor_when_no_month_grid_can_be_built() -> None:
    anchor = {**_anchor(), "printed_anchor_identity": _identity()}
    before = deepcopy(anchor)

    [grid] = materialize_credit_repayment_micro_grids(
        lines=[anchor],
        page=10,
        page_width=600,
        page_height=800,
        enable_cell_ocr=False,
        enable_static_status_validation=False,
    )

    assert grid["geometry_source"] == "date_range_anchor_only"
    assert grid["audit"]["printed_anchor_provenance"] == _identity()
    assert grid["cells"] == []
    assert grid["audit"]["date_range"] == {
        "start_year": 2021,
        "start_month": 1,
        "end_year": 2021,
        "end_month": 12,
    }
    assert anchor == before
