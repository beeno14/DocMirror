# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact physical provenance recovery for grid-standard logical rows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.bank_statement.styles.grid_standard import (
    _infer_row_sources,
    _logical_table_row_sources,
    _with_internal_row_sources,
)


def _cell(text: str, *, refs: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(text=text, evidence_ids=[], source_cell_refs=list(refs or []))


def _logical_row(
    values: list[str],
    *,
    raw_row: int,
    source_row_index: int,
) -> SimpleNamespace:
    refs = [
        {"page": 2, "table_id": "pt_2_0", "row": source_row_index, "raw_row": raw_row, "col": col}
        for col in range(len(values))
    ]
    return SimpleNamespace(
        cells=[_cell(value, refs=[refs[col]]) for col, value in enumerate(values)],
        source_page=2,
        source_physical_id="pt_2_0",
        source_row_index=source_row_index,
        source_cell_refs=refs,
    )


def _geometry(raw_rows: list[list[str]]) -> dict:
    return {
        "cell_bboxes": [
            [[float(col * 10), float(row * 20), float(col * 10 + 8), float(row * 20 + 6)] for col in range(len(values))]
            for row, values in enumerate(raw_rows)
        ],
        "cell_evidence_ids": [
            [[f"ev:{row}:{col}"] for col in range(len(values))]
            for row, values in enumerate(raw_rows)
        ],
    }


def _parse_result(
    raw_rows: list[list[str]],
    logical_rows: list[SimpleNamespace],
    *,
    physical_rows: list[SimpleNamespace] | None = None,
    geometry: dict | None = None,
) -> SimpleNamespace:
    physical_table = SimpleNamespace(
        table_id="pt_2_0",
        headers=list(raw_rows[0]),
        rows=list(physical_rows or []),
        metadata={"raw_rows": raw_rows, "geometry": geometry if geometry is not None else _geometry(raw_rows)},
    )
    logical_table = SimpleNamespace(
        rows=logical_rows,
        provenance=[
            SimpleNamespace(
                source_page=row.source_page,
                source_table_id=row.source_physical_id,
                source_row_index=row.source_row_index,
            )
            for row in logical_rows
        ],
    )
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=2, source_page_number=2, tables=[physical_table])],
        logical_tables=[logical_table],
    )


@pytest.mark.parametrize(
    ("raw_row_index", "shifted_raw_row", "logical_row_index", "expected_physical_row"),
    [
        (0, 1, 0, 0),
        (1, 2, 1, 0),
    ],
)
def test_logical_source_uses_unique_exact_physical_raw_row_geometry(
    raw_row_index: int,
    shifted_raw_row: int,
    logical_row_index: int,
    expected_physical_row: int,
) -> None:
    promoted = ["acct", "2023-10-08 18:41:49", "3793.86", "2599.34"]
    body = ["acct", "2023-10-13 13:39:37", "298.81", "2300.53"]
    raw_rows = [promoted, body]
    physical_body = _logical_row(body, raw_row=1, source_row_index=0)
    logical = _logical_row(
        raw_rows[raw_row_index],
        raw_row=shifted_raw_row,
        source_row_index=logical_row_index,
    )

    source = _logical_table_row_sources(
        _parse_result(raw_rows, [logical], physical_rows=[physical_body])
    )[0]

    assert source["bbox"] == [0.0, float(raw_row_index * 20), 38.0, float(raw_row_index * 20 + 6)]
    assert source["evidence_ids"] == [f"ev:{raw_row_index}:{col}" for col in range(4)]
    assert {ref["raw_row"] for ref in source["source_cell_refs"]} == {raw_row_index}
    assert {ref["row"] for ref in source["source_cell_refs"]} == {expected_physical_row}
    assert source["source_refs"][0]["bbox"] == source["bbox"]


@pytest.mark.parametrize("raw_rows", [[['same', 'row'], ['same', 'row']], [['different', 'row']]])
def test_logical_source_does_not_borrow_ambiguous_or_nonmatching_geometry(
    raw_rows: list[list[str]],
) -> None:
    original_ref = {"page": 2, "table_id": "pt_2_0", "row": 7, "raw_row": 8, "col": 0}
    logical = SimpleNamespace(
        cells=[_cell("same", refs=[original_ref]), _cell("row")],
        source_page=2,
        source_physical_id="pt_2_0",
        source_row_index=7,
        source_cell_refs=[original_ref],
    )

    source = _logical_table_row_sources(_parse_result(raw_rows, [logical]))[0]

    assert "bbox" not in source
    assert "evidence_ids" not in source
    assert source["source_cell_refs"] == [original_ref]


def test_true_header_geometry_is_not_attached_to_transaction_source() -> None:
    headers = ["账号", "交易时间", "交易金额", "余额"]
    values = ["acct", "2023-10-13 13:39:37", "+298.81", "2300.53"]
    header_row = _logical_row(headers, raw_row=0, source_row_index=0)
    data_row = _logical_row(values, raw_row=1, source_row_index=1)
    parse_result = _parse_result([headers, values], [header_row, data_row])

    sources = _infer_row_sources([dict(zip(headers, values, strict=True))], parse_result)

    assert len(sources) == 1
    assert sources[0]["evidence_ids"] == [f"ev:1:{col}" for col in range(4)]
    assert {ref["raw_row"] for ref in sources[0]["source_cell_refs"]} == {1}
    assert not any(evidence_id.startswith("ev:0:") for evidence_id in sources[0]["evidence_ids"])


def test_existing_logical_transaction_source_is_repaired_before_early_return() -> None:
    promoted = ["acct", "2023-10-08 18:41:49", "+3793.86", "2599.34"]
    body = ["acct", "2023-10-13 13:39:37", "+298.81", "2300.53"]
    physical_body = _logical_row(body, raw_row=1, source_row_index=0)
    logical_body = _logical_row(body, raw_row=2, source_row_index=1)
    parse_result = _parse_result([promoted, body], [logical_body], physical_rows=[physical_body])
    transaction = dict(zip(["账号", "交易时间", "交易金额", "余额"], body, strict=True))
    transaction["_source"] = {
        "source": "canonical_table",
        "source_page": 2,
        "table_id": "pt_2_0",
        "source_row_index": 1,
        "source_cell_refs": logical_body.source_cell_refs,
    }

    sourced = _with_internal_row_sources([transaction], parse_result)

    source = sourced[0]["_source"]
    assert source["bbox"] == [0.0, 20.0, 38.0, 26.0]
    assert source["evidence_ids"] == [f"ev:1:{col}" for col in range(4)]
    assert {ref["raw_row"] for ref in source["source_cell_refs"]} == {1}
    assert {ref["row"] for ref in source["source_cell_refs"]} == {0}


def test_stitched_source_refs_recover_each_exact_physical_fragment_geometry() -> None:
    first_values = ["2023-05-19", "35464.67", "上海赫程国际旅行"]
    second_values = ["16:24:44", "", "社有限公司南通分公司"]

    def physical_table(page: int, values: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            table_id=f"pt_{page}_0",
            headers=[],
            rows=[],
            metadata={"raw_rows": [values], "geometry": _geometry([values])},
        )

    fragments = []
    combined_refs = []
    for page, values in ((1, first_values), (2, second_values)):
        refs = [
            {"page": page, "table_id": f"pt_{page}_0", "row": 0, "raw_row": 0, "col": col}
            for col in range(len(values))
        ]
        combined_refs.extend(refs)
        fragments.append(
            {
                "source_page": page,
                "table_id": f"pt_{page}_0",
                "source_row_index": 0,
                "source_cell_refs": refs,
            }
        )

    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[physical_table(1, first_values)]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[physical_table(2, second_values)]),
        ],
        logical_tables=[],
    )
    transaction = {
        "交易时间": "2023-05-19\n16:24:44",
        "收入金额": "35464.67",
        "对方户名": "上海赫程国际旅行\n社有限公司南通分公司",
        "_source": {
            "source": "canonical_table",
            "source_page": 1,
            "table_id": "pt_1_0",
            "source_row_index": 0,
            "page_range": [1, 2],
            "source_refs": fragments,
            "source_cell_refs": combined_refs,
        },
    }

    source = _with_internal_row_sources([transaction], parse_result)[0]["_source"]

    assert [ref["bbox"] for ref in source["source_refs"]] == [
        [0.0, 0.0, 28.0, 6.0],
        [0.0, 0.0, 28.0, 6.0],
    ]
    assert [ref["evidence_ids"] for ref in source["source_refs"]] == [
        ["ev:0:0", "ev:0:1", "ev:0:2"],
        ["ev:0:0", "ev:0:1", "ev:0:2"],
    ]
    assert source["evidence_ids"] == ["ev:0:0", "ev:0:1", "ev:0:2"]


def test_unique_match_without_physical_geometry_keeps_logical_provenance() -> None:
    values = ["acct", "2023-10-13 13:39:37"]
    logical = _logical_row(values, raw_row=4, source_row_index=3)
    logical.cells[0].evidence_ids = ["logical:evidence"]

    source = _logical_table_row_sources(
        _parse_result([values], [logical], geometry={})
    )[0]

    assert "bbox" not in source
    assert source["evidence_ids"] == ["logical:evidence"]
    assert {ref["raw_row"] for ref in source["source_cell_refs"]} == {4}


def test_native_wide_internal_source_recovers_exact_physical_evidence_and_refs() -> None:
    headers = ["账号", "交易时间", "交易金额", "余额"]
    values = ["acct", "2023-10-13 13:39:37", "+298.81", "2300.53"]
    physical = _logical_row(values, raw_row=1, source_row_index=0)
    parse_result = _parse_result([headers, values], [], physical_rows=[physical])
    transaction = dict(zip(headers, values, strict=True))
    transaction.update(
        {
            "_source_page": "2",
            "_source_bbox": "11.000,22.000,33.000,44.000",
            "_source_table_id": "native:p2:t0",
            "_source_row_index": "1",
        }
    )

    source = _with_internal_row_sources([transaction], parse_result)[0]["_source"]

    assert source["table_id"] == "native:p2:t0"
    assert source["source_row_index"] == 1
    assert source["bbox"] == [11.0, 22.0, 33.0, 44.0]
    assert source["evidence_ids"] == [f"ev:1:{col}" for col in range(4)]
    assert {ref["table_id"] for ref in source["source_cell_refs"]} == {"pt_2_0"}
    assert {ref["raw_row"] for ref in source["source_cell_refs"]} == {1}
    assert {ref["row"] for ref in source["source_cell_refs"]} == {0}
    assert source["source_refs"][0]["bbox"] == source["bbox"]


@pytest.mark.parametrize(
    ("raw_rows", "table_id"),
    [
        ([['acct', '2023-10-13'], ['acct', '2023-10-13']], "native:p2:t0"),
        ([['different', 'row']], "native:p2:t0"),
        ([['acct', '2023-10-13']], "native:p2:t1"),
        ([['acct', '2023-10-13']], "native:p3:t0"),
    ],
)
def test_native_wide_source_alias_fails_closed_without_unique_page_table_row_proof(
    raw_rows: list[list[str]],
    table_id: str,
) -> None:
    transaction = {
        "账号": "acct",
        "交易时间": "2023-10-13",
        "_source_page": "2",
        "_source_bbox": "11.000,22.000,33.000,44.000",
        "_source_table_id": table_id,
        "_source_row_index": "1",
    }

    source = _with_internal_row_sources([transaction], _parse_result(raw_rows, []))[0]["_source"]

    assert source["bbox"] == [11.0, 22.0, 33.0, 44.0]
    assert "evidence_ids" not in source
    assert "source_cell_refs" not in source


def test_native_wide_source_alias_requires_canonical_physical_table_identity() -> None:
    values = ["acct", "2023-10-13"]
    parse_result = _parse_result([values], [])
    parse_result.pages[0].tables[0].table_id = "geo_table_0"
    transaction = {
        "账号": values[0],
        "交易时间": values[1],
        "_source_page": "2",
        "_source_bbox": "11.000,22.000,33.000,44.000",
        "_source_table_id": "native:p2:t0",
        "_source_row_index": "0",
    }

    source = _with_internal_row_sources([transaction], parse_result)[0]["_source"]

    assert source["bbox"] == [11.0, 22.0, 33.0, 44.0]
    assert "evidence_ids" not in source
    assert "source_cell_refs" not in source
