# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_employment_records,
)

_PROFILE_TEMPLATE = "report_header_and_identity"
_SUMMARY_TEMPLATE = "information_summary"


def _geometry(rows: list[list[str]], widths: list[float]) -> dict[str, object]:
    row_edges = [12.0]
    for row in rows:
        row_edges.append(row_edges[-1] + 8.0 + max(len(str(value)) for value in row) * 0.03)
    column_edges = [17.0]
    for width in widths:
        column_edges.append(column_edges[-1] + width)
    bboxes = [
        [
            [column_edges[column], row_edges[row], column_edges[column + 1], row_edges[row + 1]]
            for column in range(len(widths))
        ]
        for row in range(len(rows))
    ]
    evidence = [
        [
            [f"employment:{row}:{column}"] if str(rows[row][column]).strip() else []
            for column in range(len(widths))
        ]
        for row in range(len(rows))
    ]
    return {
        "coordinate_system": "pdf_points_top_left",
        "row_bands": [
            {"index": row, "y0": row_edges[row], "y1": row_edges[row + 1]}
            for row in range(len(rows))
        ],
        "col_bands": [
            {
                "index": column,
                "x0": column_edges[column],
                "x1": column_edges[column + 1],
            }
            for column in range(len(widths))
        ],
        "cell_bboxes": bboxes,
        "cell_geometry_status": [["exact"] * len(widths) for _row in rows],
        "cell_evidence_ids": evidence,
        "cell_token_ids": deepcopy(evidence),
        "cell_spans": [],
    }


def _owned_result(
    rows: list[list[str]],
    *,
    widths: list[float],
    template: str = _PROFILE_TEMPLATE,
    heading: str | None = None,
) -> SimpleNamespace:
    geometry = _geometry(rows, widths)
    table = SimpleNamespace(
        table_id="employment-generic",
        metadata={
            "raw_rows": rows,
            "geometry": geometry,
            "canonical_template_id": template,
            "source_logical_page": 9,
            "source_page": 4,
        },
        headers=[],
        rows=[],
        bbox=[17.0, geometry["row_bands"][0]["y0"], sum(widths) + 17.0, geometry["row_bands"][-1]["y1"]],
    )
    texts = []
    if heading is not None:
        texts.extend(
            [
                SimpleNamespace(
                    content=heading,
                    bbox=[90.0, 1.0, 180.0, 8.0],
                    evidence_ids=["employment-section-heading"],
                    source_logical_page=9,
                ),
                SimpleNamespace(
                    content="信息概要",
                    bbox=[90.0, table.bbox[3] + 5.0, 180.0, table.bbox[3] + 12.0],
                    evidence_ids=["summary-section-heading"],
                    source_logical_page=9,
                ),
            ]
        )
    page = SimpleNamespace(
        page_number=9,
        source_page_number=4,
        canonical_template_id=template,
        canonical_fragment_logical_pages=(9,),
        coordinate_transform={
            "kind": "plugin_canonical_template",
            "source_page_numbers": [4],
        },
        tables=[table],
        texts=texts,
    )
    return SimpleNamespace(
        pages=[page],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )


def _reordered_rows() -> list[list[str]]:
    return [
        ["单位电话", "单位地址", "编号", "工作单位", "单位性质", "保留空槽", ""],
        ["010-87654321", "福建省福州市鼓楼区通湖路8号", "1", "泛用科技有限公司", "国有企业", "", ""],
        ["职称", "信息更新日期", "编号", "行业", "进入本单位年份", "职业", "职务"],
        ["中级", "2025.03.04", "1", "制造业", "2019", "专业技术人员", "一般员工"],
        ["编号", "数据发生机构名称", "", "", "", "", ""],
        ["1", "泛用银行股份有限公司", "", "", "", "", ""],
    ]


@pytest.mark.parametrize(
    ("widths", "heading"),
    [
        ([61.0, 173.0, 37.0, 149.0, 83.0, 29.0, 47.0], None),
        ([122.0, 55.0, 46.0, 207.0, 71.0, 96.0, 34.0], "（九）职业信息"),
    ],
)
def test_owned_exact_employment_accepts_reordered_unequal_lattices(
    widths: list[float],
    heading: str | None,
) -> None:
    template = _PROFILE_TEMPLATE if heading is None else _SUMMARY_TEMPLATE
    result = _owned_result(
        _reordered_rows(),
        widths=widths,
        template=template,
        heading=heading,
    )

    records = _extract_employment_records(result)

    assert len(records) == 1
    assert {
        field: records[0][field]
        for field in (
            "employer",
            "employer_type",
            "employer_address",
            "employer_phone",
            "occupation",
            "industry",
            "position",
            "professional_title",
            "entry_year",
            "information_updated_date",
        )
    } == {
        "employer": "泛用科技有限公司",
        "employer_type": "国有企业",
        "employer_address": "福建省福州市鼓楼区通湖路8号",
        "employer_phone": "01087654321",
        "occupation": "专业技术人员",
        "industry": "制造业",
        "position": "一般员工",
        "professional_title": "中级",
        "entry_year": 2019,
        "information_updated_date": "2025-03-04",
    }
    for refs in records[0]["source_refs_by_field"].values():
        assert all(ref.get("geometry_status") == "exact" for ref in refs)
        assert all(ref.get("canonical_template_id") == template for ref in refs)


@pytest.mark.parametrize(
    "mutation",
    [
        "unregistered",
        "foreign_section",
        "missing_transition_heading",
        "duplicate_heading_owner",
        "shared_header_evidence",
        "ambiguous_header_span",
        "unsealed_value_cell",
    ],
)
def test_employment_owner_gate_rejects_unowned_or_ambiguous_tables(mutation: str) -> None:
    result = _owned_result(
        _reordered_rows(),
        widths=[61.0, 173.0, 37.0, 149.0, 83.0, 29.0, 47.0],
    )
    page = result.pages[0]
    table = page.tables[0]
    geometry = table.metadata["geometry"]
    if mutation == "unregistered":
        delattr(page, "canonical_template_id")
        table.metadata.pop("canonical_template_id")
    elif mutation == "foreign_section":
        page.canonical_template_id = "credit_account_detail"
        table.metadata["canonical_template_id"] = "credit_account_detail"
    elif mutation == "missing_transition_heading":
        page.canonical_template_id = _SUMMARY_TEMPLATE
        table.metadata["canonical_template_id"] = _SUMMARY_TEMPLATE
    elif mutation == "duplicate_heading_owner":
        page.canonical_template_id = _SUMMARY_TEMPLATE
        table.metadata["canonical_template_id"] = _SUMMARY_TEMPLATE
        page.texts = [
            SimpleNamespace(
                content="职业信息",
                bbox=[90.0, top, 180.0, top + 5.0],
                evidence_ids=[f"heading:{top}"],
                source_logical_page=9,
            )
            for top in (1.0, 6.0)
        ] + [
            SimpleNamespace(
                content="信息概要",
                bbox=[90.0, table.bbox[3] + 5.0, 180.0, table.bbox[3] + 12.0],
                evidence_ids=["summary-heading"],
                source_logical_page=9,
            )
        ]
    elif mutation == "shared_header_evidence":
        geometry["cell_evidence_ids"][0][3] = geometry["cell_evidence_ids"][0][4]
        geometry["cell_token_ids"][0][3] = geometry["cell_token_ids"][0][4]
    elif mutation == "ambiguous_header_span":
        geometry["cell_spans"] = [
            {"row": 0, "col": 3, "row_span": 1, "col_span": 2},
            {"row": 0, "col": 3, "row_span": 1, "col_span": 2},
        ]
    elif mutation == "unsealed_value_cell":
        geometry["cell_geometry_status"][1][3] = "projected"

    records = _extract_employment_records(result)
    if mutation in {"unregistered", "foreign_section", "missing_transition_heading", "duplicate_heading_owner"}:
        assert records == []
    elif mutation in {"shared_header_evidence", "ambiguous_header_span"}:
        assert len(records) == 1
        assert "employer" not in records[0]
        assert records[0]["occupation"] == "专业技术人员"
        assert records[0]["data_provider"] == "泛用银行股份有限公司"
    else:
        assert len(records) == 1
        assert "employer" not in records[0]
        assert records[0]["employer_type"] == "国有企业"


def test_damaged_ordinal_header_requires_exact_local_ordinal_owners() -> None:
    rows = [
        ["", "工作单位", "单位性质", "单位地址", "单位电话", ""],
        ["1", "泛用科技有限公司", "国有企业", "福建省福州市鼓楼区通湖路8号", "010-87654321", ""],
        ["2", "通用实业有限公司", "私营企业", "福建省福州市台江区江滨路3号", "010-76543210", ""],
    ]
    result = _owned_result(
        rows,
        widths=[43.0, 191.0, 73.0, 211.0, 102.0, 31.0],
    )

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    result.pages[0].tables[0].metadata["geometry"]["cell_geometry_status"][2][0] = "projected"
    assert _extract_employment_records(result) == []


def test_collapsed_basic_emits_only_from_one_exact_owned_cluster() -> None:
    rows = [
        ["编号工作单位单位性质单位地址单位电话", "", ""],
        ["1 泛用科技有限公司 国有企业 福建省福州市鼓楼区通湖路8号 01087654321", "", ""],
        ["2 通用实业有限公司 私营企业 福建省福州市台江区江滨路3号 01076543210", "", ""],
    ]
    result = _owned_result(rows, widths=[307.0, 89.0, 42.0])

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    assert records[0]["employer"] == "泛用科技有限公司"
    result.pages[0].tables[0].metadata["geometry"]["cell_geometry_status"][1][0] = "projected"
    assert [record["sequence"] for record in _extract_employment_records(result)] == [2]


def test_clustered_detail_requires_unique_header_and_value_owners() -> None:
    rows = [
        ["编号", "工作单位", "单位性质", "单位地址", "单位电话"],
        ["1", "泛用科技有限公司", "国有企业", "福建省福州市鼓楼区通湖路8号", "01087654321"],
        ["编号", "行业 职业", "", "进入本单位年份 职称 职务", "信息更新日期"],
        ["1", "制造业 专业技术人员", "", "2019 一般员工 中级", "2025.03.04"],
    ]
    result = _owned_result(rows, widths=[41.0, 225.0, 35.0, 187.0, 105.0])

    records = _extract_employment_records(result)

    assert records[0]["occupation"] == "专业技术人员"
    geometry = result.pages[0].tables[0].metadata["geometry"]
    geometry["cell_evidence_ids"][2][1] = geometry["cell_evidence_ids"][2][3]
    geometry["cell_token_ids"][2][1] = geometry["cell_token_ids"][2][3]
    records = _extract_employment_records(result)
    assert "occupation" not in records[0]
