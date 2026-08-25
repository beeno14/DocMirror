from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.models.entities.parse_result import (
    DocumentEntities,
    PageContent,
    ParseResult,
)
from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.community_plugin import _CreditReportCommunityBundle
from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessRepairPlan,
    BusinessUncertaintyRepairCoordinator,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import CandidateBPipeline
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)


def _sealed_context(
    body_rows: list[list[object]],
    *,
    headers: list[object] | None = None,
    column_widths: tuple[float, ...] | None = None,
    scale: float = 1.0,
    table_top: float = 60.0,
    table_id: str = "raw-inquiry",
    duplicate_start: bool = False,
    replay_cell_evidence: bool = False,
) -> SimpleNamespace:
    rows = ([headers] if headers is not None else []) + body_rows
    column_count = len(rows[0])
    assert all(len(row) == column_count for row in rows)
    widths = column_widths or tuple(70.0 + index * 9.0 for index in range(column_count))
    assert len(widths) == column_count
    left = 30.0 * scale
    top = table_top * scale
    row_height = 18.0 * scale
    edges = [left]
    for width in widths:
        edges.append(edges[-1] + width * scale)

    atoms: list[EvidenceAtom] = []
    cell_bboxes: list[list[list[float]]] = []
    cell_status: list[list[str]] = []
    cell_evidence: list[list[list[str]]] = []
    cell_tokens: list[list[list[str]]] = []
    raw_rows: list[list[str]] = []
    first_body_token_id = ""
    for row_index, row in enumerate(rows):
        raw_line: list[str] = []
        bbox_line: list[list[float]] = []
        status_line: list[str] = []
        evidence_line: list[list[str]] = []
        token_line: list[list[str]] = []
        for column, raw_cell in enumerate(row):
            token_texts = (
                [str(value) for value in raw_cell]
                if isinstance(raw_cell, (list, tuple))
                else ([str(raw_cell)] if str(raw_cell or "") else [])
            )
            raw_line.append(" ".join(token_texts))
            y0 = top + row_index * row_height
            y1 = y0 + row_height
            bbox = [edges[column], y0, edges[column + 1], y1]
            bbox_line.append(bbox)
            status_line.append("exact")
            ids: list[str] = []
            for token_index, token_text in enumerate(token_texts):
                token_id = f"cell:{table_id}:{row_index}:{column}:{token_index}"
                if replay_cell_evidence and row_index == len(rows) - 1 and column == column_count - 1:
                    token_id = first_body_token_id
                if not first_body_token_id and row_index >= int(headers is not None):
                    first_body_token_id = token_id
                token_width = (bbox[2] - bbox[0]) / max(len(token_texts), 1)
                token_bbox = [
                    bbox[0] + token_index * token_width + 2.0 * scale,
                    bbox[1] + 4.0 * scale,
                    bbox[0] + (token_index + 1) * token_width - 2.0 * scale,
                    bbox[3] - 4.0 * scale,
                ]
                ids.append(token_id)
                atoms.append(EvidenceAtom(id=token_id, text=token_text, bbox=token_bbox))
            evidence_line.append(ids)
            token_line.append(list(ids))
        raw_rows.append(raw_line)
        cell_bboxes.append(bbox_line)
        cell_status.append(status_line)
        cell_evidence.append(evidence_line)
        cell_tokens.append(token_line)

    table_bottom = top + len(rows) * row_height
    geometry = {
        "row_bands": [
            {"index": index, "y0": top + index * row_height, "y1": top + (index + 1) * row_height}
            for index in range(len(rows))
        ],
        "col_bands": [
            {"index": index, "x0": edges[index], "x1": edges[index + 1]}
            for index in range(column_count)
        ],
        "cell_bboxes": cell_bboxes,
        "cell_geometry_status": cell_status,
        "cell_evidence_ids": cell_evidence,
        "cell_token_ids": cell_tokens,
        "cell_spans": [],
    }
    table = SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": raw_rows, "geometry": geometry},
        bbox=[left, top, edges[-1], table_bottom],
        headers=[],
        rows=[],
    )

    def heading(text: str, y0: float, suffix: str) -> SimpleNamespace:
        evidence_id = f"heading:{suffix}"
        bbox = [left, y0 * scale, left + 100.0 * scale, (y0 + 8.0) * scale]
        atoms.append(EvidenceAtom(id=evidence_id, text=text, bbox=bbox))
        return SimpleNamespace(content=text, bbox=bbox, evidence_ids=[evidence_id])

    texts = [
        heading("四查询记录", 10.0, "start"),
        heading("机构查询记录明细", 28.0, "institution"),
        heading("报告说明", table_bottom / scale + 20.0, "boundary"),
    ]
    if duplicate_start:
        texts.insert(1, heading("五查询记录", 19.0, "competing-start"))
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=edges[-1] + 30.0 * scale,
        height=(table_bottom + 50.0 * scale),
        texts=texts,
        tables=[table],
    )
    return SimpleNamespace(
        pages=[],
        _frozen_logical_pages={1: page},
        reading_order_resolution={"resolved": True, "authoritative": True},
        reading_order_by_logical={1: 1},
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
    )


def _inquiry_row(sequence: int, *, personal: bool = False) -> list[str]:
    return [
        str(sequence),
        f"2024.{((sequence - 1) % 12) + 1:02d}.{((sequence - 1) % 27) + 1:02d}",
        "本人" if personal else f"机构{sequence}",
        "本人查询(互联网个人信用信息服务平台)" if personal else "贷后管理",
    ]


def _chain_heading(
    atoms: list[EvidenceAtom],
    *,
    page_number: int,
    text: str,
    y0: float,
    suffix: str,
) -> SimpleNamespace:
    evidence_id = f"heading:{page_number}:{suffix}"
    bbox = [30.0, y0, 230.0, y0 + 8.0]
    atoms.append(EvidenceAtom(id=evidence_id, text=text, bbox=bbox))
    return SimpleNamespace(content=text, bbox=bbox, evidence_ids=[evidence_id])


def _sealed_chain_context(
    institutional_batches: list[list[list[object]]],
    *,
    personal_rows: list[list[object]] | None = None,
    continuation_headers: dict[int, list[object]] | None = None,
    institutional_column_widths: dict[int, tuple[float, ...]] | None = None,
) -> SimpleNamespace:
    assert institutional_batches
    continuation_headers = continuation_headers or {}
    institutional_column_widths = institutional_column_widths or {}
    pages: list[SimpleNamespace] = []
    atoms: list[EvidenceAtom] = []
    for batch_index, batch in enumerate(institutional_batches):
        page_number = batch_index + 1
        header = (
            ["编号", "查询日期", "查询机构", "查询原因"]
            if batch_index == 0
            else continuation_headers.get(batch_index)
        )
        table_id = f"inst-{batch_index}"
        fixture = _sealed_context(
            batch,
            headers=header,
            table_id=table_id,
            table_top=60.0,
            column_widths=institutional_column_widths.get(batch_index),
        )
        table = fixture._frozen_logical_pages[1].tables[0]
        atoms.extend(
            atom
            for atom in fixture.evidence_plane.evidence.text_atoms
            if atom.id.startswith(f"cell:{table_id}:")
        )
        texts: list[SimpleNamespace] = []
        if batch_index == 0:
            texts.extend(
                [
                    _chain_heading(
                        atoms,
                        page_number=page_number,
                        text="四查询记录",
                        y0=10.0,
                        suffix="start",
                    ),
                    _chain_heading(
                        atoms,
                        page_number=page_number,
                        text="机构查询记录明细",
                        y0=28.0,
                        suffix="institution",
                    ),
                ]
            )
        tables = [table]
        page_bottom = float(table.bbox[3])
        if batch_index == len(institutional_batches) - 1 and personal_rows is not None:
            personal_heading_y = page_bottom + 12.0
            texts.append(
                _chain_heading(
                    atoms,
                    page_number=page_number,
                    text="本人查询记录明细",
                    y0=personal_heading_y,
                    suffix="personal",
                )
            )
            personal_top = personal_heading_y + 18.0
            personal_id = f"personal-{batch_index}"
            personal_fixture = _sealed_context(
                personal_rows,
                headers=["编号", "查询日期", "查询机构", "查询原因"],
                table_id=personal_id,
                table_top=personal_top,
            )
            personal_table = personal_fixture._frozen_logical_pages[1].tables[0]
            tables.append(personal_table)
            atoms.extend(
                atom
                for atom in personal_fixture.evidence_plane.evidence.text_atoms
                if atom.id.startswith(f"cell:{personal_id}:")
            )
            page_bottom = float(personal_table.bbox[3])
        if batch_index == len(institutional_batches) - 1:
            texts.append(
                _chain_heading(
                    atoms,
                    page_number=page_number,
                    text="报告说明",
                    y0=page_bottom + 20.0,
                    suffix="boundary",
                )
            )
        pages.append(
            SimpleNamespace(
                page_number=page_number,
                source_page_number=page_number,
                width=460.0,
                height=page_bottom + 60.0,
                texts=texts,
                tables=tables,
            )
        )
    return SimpleNamespace(
        pages=[],
        _frozen_logical_pages={page.page_number: page for page in pages},
        reading_order_resolution={"resolved": True, "authoritative": True},
        reading_order_by_logical={
            page.page_number: index
            for index, page in enumerate(pages, start=1)
        },
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
    )


def _merge_exact_chain_cells(
    context: SimpleNamespace,
    *,
    page_number: int,
    row: int,
    column: int,
    column_span: int = 2,
) -> None:
    table = context._frozen_logical_pages[page_number].tables[0]
    geometry = table.metadata["geometry"]
    last_column = column + column_span - 1
    owner_bbox = geometry["cell_bboxes"][row][column]
    final_bbox = geometry["cell_bboxes"][row][last_column]
    owner_bbox[2] = final_bbox[2]
    evidence_ids = [
        evidence_id
        for source_column in range(column, column + column_span)
        for evidence_id in geometry["cell_evidence_ids"][row][source_column]
    ]
    token_ids = [
        evidence_id
        for source_column in range(column, column + column_span)
        for evidence_id in geometry["cell_token_ids"][row][source_column]
    ]
    geometry["cell_evidence_ids"][row][column] = evidence_ids
    geometry["cell_token_ids"][row][column] = token_ids
    table.metadata["raw_rows"][row][column] = " ".join(
        value
        for value in table.metadata["raw_rows"][row][column : column + column_span]
        if value
    )
    for covered_column in range(column + 1, column + column_span):
        geometry["cell_geometry_status"][row][covered_column] = "derived"
        geometry["cell_bboxes"][row][covered_column] = None
        geometry["cell_evidence_ids"][row][covered_column] = []
        geometry["cell_token_ids"][row][covered_column] = []
        table.metadata["raw_rows"][row][covered_column] = ""
    geometry["cell_spans"].append(
        {
            "row": row,
            "col": column,
            "row_span": 1,
            "col_span": column_span,
        }
    )


def _replace_exact_chain_cell_text(
    context: SimpleNamespace,
    *,
    page_number: int,
    row: int,
    column: int,
    text: str,
) -> None:
    table = context._frozen_logical_pages[page_number].tables[0]
    geometry = table.metadata["geometry"]
    token_ids = geometry["cell_token_ids"][row][column]
    assert len(token_ids) == 1
    token_id = token_ids[0]
    table.metadata["raw_rows"][row][column] = text
    atoms = context.evidence_plane.evidence.text_atoms
    atom_index = next(index for index, atom in enumerate(atoms) if atom.id == token_id)
    atom = atoms[atom_index]
    atoms[atom_index] = EvidenceAtom(
        id=atom.id,
        text=text,
        bbox=list(atom.bbox),
    )


def _blank_exact_chain_cell(
    context: SimpleNamespace,
    *,
    page_number: int,
    row: int,
    column: int,
    derived: bool = False,
) -> None:
    table = context._frozen_logical_pages[page_number].tables[0]
    geometry = table.metadata["geometry"]
    token_ids = list(geometry["cell_token_ids"][row][column])
    assert token_ids
    table.metadata["raw_rows"][row][column] = ""
    geometry["cell_evidence_ids"][row][column] = []
    geometry["cell_token_ids"][row][column] = []
    if derived:
        geometry["cell_geometry_status"][row][column] = "derived"
        geometry["cell_bboxes"][row][column] = None
    context.evidence_plane.evidence.text_atoms = [
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id not in token_ids
    ]


def _population_only_chain_context(
    continuation_rows: list[list[object]],
    *,
    exact_widths: bool = False,
    seed_rows: list[list[object]] | None = None,
) -> SimpleNamespace:
    seed_widths = (40.0, 79.0, 154.5, 78.5)
    context = _sealed_chain_context(
        [
            seed_rows
            or [
                ["1", "2024.07.01", "测试银行", "贷后管理"],
                ["2", "2024.06.01", "测试银行", "贷后管理"],
            ],
            continuation_rows,
        ],
        institutional_column_widths={
            0: seed_widths,
            1: seed_widths if exact_widths else (39.0, 78.0, 154.0, 79.0),
        },
    )
    # Withhold the independent headerless descriptor and reproduce the exact
    # sequence/date owner that starts the live Yang continuation's noisy tail.
    _merge_exact_chain_cells(
        context,
        page_number=2,
        row=0,
        column=0,
    )
    _replace_exact_chain_cell_text(
        context,
        page_number=2,
        row=1,
        column=1,
        text="2024,04.01",
    )
    return context


def _append_chain_furniture_table(
    context: SimpleNamespace,
    *,
    table_id: str,
) -> None:
    page = context._frozen_logical_pages[2]
    table_top = float(page.tables[0].bbox[3]) + 4.0
    fixture = _sealed_context(
        [
            ["5", "2024.03.01", "复核", "备注"],
            ["6", "2024.02.01", "制表人", "以下空白"],
        ],
        column_widths=(40.0, 79.0, 154.5, 78.5),
        table_id="furniture-evidence-owner",
        table_top=table_top,
    )
    furniture_table = fixture._frozen_logical_pages[1].tables[0]
    furniture_table.table_id = table_id
    page.tables.append(furniture_table)
    context.evidence_plane.evidence.text_atoms.extend(
        atom
        for atom in fixture.evidence_plane.evidence.text_atoms
        if atom.id.startswith("cell:furniture-evidence-owner:")
    )

    boundary = page.texts[-1]
    boundary_bbox = [
        30.0,
        float(furniture_table.bbox[3]) + 20.0,
        230.0,
        float(furniture_table.bbox[3]) + 28.0,
    ]
    boundary.bbox = boundary_bbox
    boundary_id = boundary.evidence_ids[0]
    atoms = context.evidence_plane.evidence.text_atoms
    atom_index = next(index for index, atom in enumerate(atoms) if atom.id == boundary_id)
    atom = atoms[atom_index]
    atoms[atom_index] = EvidenceAtom(
        id=atom.id,
        text=atom.text,
        bbox=boundary_bbox,
    )
    page.height = boundary_bbox[3] + 40.0


def _chain_envelope_and_table_entries(
    context: SimpleNamespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    envelope = native_extraction._sealed_raw_inquiry_envelope(context)
    assert envelope is not None
    entries: list[dict[str, Any]] = []
    for page_rank, (page_position, _logical_page, _source_page, page) in enumerate(
        envelope["page_rows"]
    ):
        for table in page.tables:
            table_bbox = native_extraction._exact_geometry_bbox(table.bbox)
            assert table_bbox is not None
            table_key = (
                *page_position,
                table_bbox[1],
                table_bbox[0],
                table_bbox[3],
                table_bbox[2],
            )
            if not (envelope["lower_key"] < table_key < envelope["upper_key"]):
                continue
            grid = native_extraction._sealed_raw_inquiry_cell_grid(
                context,
                page,
                table,
            )
            descriptor = (
                native_extraction._sealed_raw_inquiry_table_descriptor(table, grid)
                if grid is not None
                else None
            )
            entries.append(
                {
                    "table_key": table_key,
                    "page_rank": page_rank,
                    "page": page,
                    "table": table,
                    "table_bbox": table_bbox,
                    "grid": grid,
                    "descriptor": descriptor,
                }
            )
    entries.sort(key=lambda entry: entry["table_key"])
    return envelope, entries


def _yu_packed_split_header_context(*, mismatched_reason_bands: bool = False) -> SimpleNamespace:
    institution_id = "yu-inst"
    institution_fixture = _sealed_context(
        [
            [
                ["1", "2"],
                ["2024.01.01", "2024.01.02"],
                ["机构一", "机构二"],
                ["贷后管理"] if mismatched_reason_bands else ["贷后管理", "贷款审批"],
            ]
        ],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
        table_id=institution_id,
        table_top=60.0,
    )
    institution_table = institution_fixture._frozen_logical_pages[1].tables[0]
    institution_atoms: list[EvidenceAtom] = []
    for atom in institution_fixture.evidence_plane.evidence.text_atoms:
        if not atom.id.startswith(f"cell:{institution_id}:"):
            continue
        parts = atom.id.split(":")
        if len(parts) == 5 and parts[2] == "1":
            token_index = int(parts[4])
            bbox = list(atom.bbox)
            bbox[1] = 80.0 + token_index * 8.0
            bbox[3] = bbox[1] + 6.0
            institution_atoms.append(EvidenceAtom(id=atom.id, text=atom.text, bbox=bbox))
        else:
            institution_atoms.append(atom)

    personal_top = float(institution_table.bbox[3]) + 30.0
    personal_id = "yu-personal"
    personal_fixture = _sealed_context(
        [
            ["", "查询日期", "查询机构", "查询原因"],
            _inquiry_row(1, personal=True),
        ],
        headers=["编号", "", "", ""],
        table_id=personal_id,
        table_top=personal_top,
    )
    personal_table = personal_fixture._frozen_logical_pages[1].tables[0]
    personal_geometry = personal_table.metadata["geometry"]
    first_band = personal_geometry["row_bands"][0]
    second_band = personal_geometry["row_bands"][1]
    owner_bbox = personal_geometry["cell_bboxes"][0][0]
    owner_bbox[1] = first_band["y0"]
    owner_bbox[3] = second_band["y1"]
    personal_geometry["cell_geometry_status"][1][0] = "derived"
    personal_geometry["cell_bboxes"][1][0] = None
    personal_geometry["cell_evidence_ids"][1][0] = []
    personal_geometry["cell_token_ids"][1][0] = []
    personal_geometry["cell_spans"] = [
        {
            "row": 0,
            "col": 0,
            "row_span": 2,
            "col_span": 1,
        }
    ]
    personal_atoms = [
        atom
        for atom in personal_fixture.evidence_plane.evidence.text_atoms
        if atom.id.startswith(f"cell:{personal_id}:")
    ]

    atoms = [*institution_atoms, *personal_atoms]
    texts = [
        _chain_heading(
            atoms,
            page_number=1,
            text="四查询记录",
            y0=10.0,
            suffix="start",
        ),
        _chain_heading(
            atoms,
            page_number=1,
            text="机构查询记录明细",
            y0=28.0,
            suffix="institution",
        ),
        _chain_heading(
            atoms,
            page_number=1,
            text="本人查询记录明细",
            y0=float(institution_table.bbox[3]) + 12.0,
            suffix="personal",
        ),
        _chain_heading(
            atoms,
            page_number=1,
            text="报告说明",
            y0=float(personal_table.bbox[3]) + 20.0,
            suffix="boundary",
        ),
    ]
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        width=460.0,
        height=float(personal_table.bbox[3]) + 60.0,
        texts=texts,
        tables=[institution_table, personal_table],
    )
    return SimpleNamespace(
        pages=[],
        _frozen_logical_pages={1: page},
        reading_order_resolution={"resolved": True, "authoritative": True},
        reading_order_by_logical={1: 1},
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
    )


def _yu_static_split_26_source_parse_result() -> SimpleNamespace:
    seed_fixture = _sealed_context(
        [_inquiry_row(sequence) for sequence in range(1, 10)],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
        column_widths=(39.0, 76.5, 152.5, 78.0),
        table_id="pt_16_4",
        table_top=60.0,
    )
    seed_table = seed_fixture._frozen_logical_pages[1].tables[0]
    continuation_rows: list[list[object]] = [
        [
            ["10", "11", "12"],
            ["2021.06.19", "2021.05.18", "2021.05.18"],
            ["机构10", "机构11", "机构12"],
            ["贷后管理", "贷款审批", "信用卡审批"],
        ],
        _inquiry_row(13),
        _inquiry_row(14),
        ["15", "2021,03.06", "机构15", "信用卡审批"],
        [
            ["16", "17"],
            ["2021.01.31", "2020.12.23"],
            ["机构16", "机构17"],
            ["贷款审批", "贷后管理"],
        ],
        *[_inquiry_row(sequence) for sequence in (18, 19, 20)],
        [["噪", "21"], ["2020.12.12"], ["机构21"], ["贷款审批"]],
        *[_inquiry_row(sequence) for sequence in (22, 23, 24)],
    ]
    continuation_fixture = _sealed_context(
        continuation_rows,
        column_widths=(40.5, 79.5, 152.0, 78.5),
        table_id="pt_17_0",
        table_top=20.0,
    )
    continuation_table = continuation_fixture._frozen_logical_pages[1].tables[0]

    personal_top = float(continuation_table.bbox[3]) + 36.0
    personal_fixture = _sealed_context(
        [
            ["", "查询日期", "查询机构", "查询原因"],
            _inquiry_row(1, personal=True),
            _inquiry_row(2, personal=True),
        ],
        headers=["编号", "", "", ""],
        table_id="pt_17_1",
        table_top=personal_top,
    )
    personal_table = personal_fixture._frozen_logical_pages[1].tables[0]
    personal_geometry = personal_table.metadata["geometry"]
    personal_rows = personal_table.metadata["raw_rows"]
    personal_geometry["cell_bboxes"][0][0][3] = personal_geometry["row_bands"][1]["y1"]
    personal_geometry["cell_geometry_status"][1][0] = "derived"
    personal_geometry["cell_bboxes"][1][0] = None
    personal_geometry["cell_evidence_ids"][1][0] = []
    personal_geometry["cell_token_ids"][1][0] = []
    personal_geometry["cell_spans"] = [
        {"row": 0, "col": 0, "row_span": 2, "col_span": 1}
    ]
    bottom = float(personal_table.bbox[3])
    personal_rows.append(["", "", "", ""])
    personal_geometry["row_bands"].append(
        {"index": len(personal_rows) - 1, "y0": bottom, "y1": bottom + 3.0}
    )
    personal_geometry["cell_bboxes"].append(
        [
            [
                personal_geometry["col_bands"][column]["x0"],
                bottom,
                personal_geometry["col_bands"][column]["x1"],
                bottom + 3.0,
            ]
            if column < 3
            else None
            for column in range(4)
        ]
    )
    personal_geometry["cell_geometry_status"].append(
        ["exact", "exact", "exact", "derived"]
    )
    personal_geometry["cell_evidence_ids"].append([[], [], [], []])
    personal_geometry["cell_token_ids"].append([[], [], [], []])
    personal_table.bbox[3] = bottom + 3.0

    atoms = [
        atom
        for fixture, prefix in (
            (seed_fixture, "cell:pt_16_4:"),
            (continuation_fixture, "cell:pt_17_0:"),
            (personal_fixture, "cell:pt_17_1:"),
        )
        for atom in fixture.evidence_plane.evidence.text_atoms
        if atom.id.startswith(prefix)
    ]
    atoms_by_id = {atom.id: atom for atom in atoms}
    continuation_geometry = continuation_table.metadata["geometry"]
    for row_index, token_row in enumerate(continuation_geometry["cell_token_ids"]):
        for column, token_ids in enumerate(token_row):
            cell_bbox = continuation_geometry["cell_bboxes"][row_index][column]
            for token_index, token_id in enumerate(token_ids):
                atom = atoms_by_id[token_id]
                if row_index == 8 and column == 0:
                    y0 = float(cell_bbox[1]) + 5.0
                    y1 = float(cell_bbox[3]) - 5.0
                else:
                    slot = (float(cell_bbox[3]) - float(cell_bbox[1])) / len(token_ids)
                    y0 = float(cell_bbox[1]) + token_index * slot + 1.0
                    y1 = float(cell_bbox[1]) + (token_index + 1) * slot - 1.0
                token_bbox = [float(atom.bbox[0]), y0, float(atom.bbox[2]), y1]
                if row_index == 1 and column == 2:
                    # The producer assigns exact scanned-grid membership by
                    # token centre; this glyph box deliberately crosses a rule.
                    token_bbox[0] = float(cell_bbox[0]) - 14.0
                    token_bbox[2] = float(cell_bbox[0]) + 30.0
                atoms_by_id[token_id] = EvidenceAtom(
                    id=atom.id,
                    text=atom.text,
                    bbox=token_bbox,
                )
    atoms = list(atoms_by_id.values())

    width = 460.0
    height = 400.0

    def spread_transform(segment: int) -> dict[str, object]:
        return {
            "source_page_number": 1,
            "source_crop_bbox": [segment * width, 0.0, (segment + 1) * width, height],
            "display_width": width,
            "display_height": height,
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "inverse_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "decomposition": {
                "kind": "two_page_spread",
                "segment_index": segment,
                "selected_rotation": 0,
                "confidence": 1.0,
            },
        }

    page_texts = [
        [
            _chain_heading(
                atoms,
                page_number=1,
                text="四查询记录",
                y0=10.0,
                suffix="start",
            ),
            _chain_heading(
                atoms,
                page_number=1,
                text="机构查询记录明细",
                y0=28.0,
                suffix="institution",
            ),
        ],
        [
            _chain_heading(
                atoms,
                page_number=2,
                text="本人查询记录明细",
                y0=float(continuation_table.bbox[3]) + 12.0,
                suffix="personal",
            ),
            _chain_heading(
                atoms,
                page_number=2,
                text="报告说明",
                y0=float(personal_table.bbox[3]) + 18.0,
                suffix="boundary",
            ),
        ],
    ]
    pages: list[SimpleNamespace] = []
    for page_number, (texts, tables) in enumerate(
        zip(page_texts, ([seed_table], [continuation_table, personal_table]), strict=True),
        start=1,
    ):
        footer = f"第 {page_number} 页，共 2 页"
        footer_id = f"footer:{page_number}"
        footer_bbox = [180.0, height - 18.0, 300.0, height - 6.0]
        texts.append(
            SimpleNamespace(content=footer, bbox=footer_bbox, evidence_ids=[footer_id])
        )
        atoms.append(EvidenceAtom(id=footer_id, text=footer, bbox=footer_bbox))
        pages.append(
            SimpleNamespace(
                page_number=page_number,
                source_page_number=1,
                width=width,
                height=height,
                coordinate_transform=spread_transform(page_number - 1),
                texts=texts,
                tables=tables,
            )
        )
    return SimpleNamespace(
        pages=pages,
        entities=SimpleNamespace(domain_specific={}),
        evidence_plane=SimpleNamespace(evidence=EvidenceStore(text_atoms=atoms)),
    )


def test_static_split_yu_population_survives_candidate_b_second_pass_and_public_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parse_result = _yu_static_split_26_source_parse_result()
    context = build_personal_detail_extraction_context(parse_result)

    def force_static_second_pass(
        _coordinator: BusinessUncertaintyRepairCoordinator,
        _payload: object,
        **_kwargs: object,
    ) -> BusinessRepairPlan:
        return BusinessRepairPlan(affected_pages=(2,))

    def retain_static_page_evidence(
        _coordinator: BusinessUncertaintyRepairCoordinator,
        plan: BusinessRepairPlan,
        **_kwargs: object,
    ) -> BusinessRepairPlan:
        page = context._frozen_logical_pages[2]
        plan.page_evidence[2] = {
            "page": 2,
            "source_page": 1,
            "page_width": page.width,
            "page_height": page.height,
            "lines": [
                {
                    "text": text.content,
                    "bbox": list(text.bbox),
                    "evidence_ids": list(text.evidence_ids),
                }
                for text in page.texts
            ],
        }
        plan.page_decisions.append(
            {
                "logical_page": 2,
                "mode": "existing_complete_page_evidence",
                "ocr_invocations": 0,
                "target_count": 1,
            }
        )
        return plan

    monkeypatch.setattr(
        BusinessUncertaintyRepairCoordinator,
        "plan",
        force_static_second_pass,
    )
    monkeypatch.setattr(
        BusinessUncertaintyRepairCoordinator,
        "resolve_page_evidence",
        retain_static_page_evidence,
    )

    result = CandidateBPipeline(context, "").run()

    assert result.audit["schema_extraction_pass_count"] == 2
    assert result.audit["parse_result_mutated"] is False
    assert context._business_repair_active is True
    topology = result.audit["page_topology"]
    assert {
        key: topology[key]
        for key in (
            "logical_page_count",
            "source_page_count",
            "double_page_sources",
            "static_split_recoveries",
            "topology_frozen_before_reocr",
        )
    } == {
        "logical_page_count": 2,
        "source_page_count": 1,
        "double_page_sources": 1,
        "static_split_recoveries": [],
        "topology_frozen_before_reocr": True,
    }
    repair_audit = result.audit["ocr_correction"]["business_repair"]
    assert repair_audit["field_triggered_ocr_requests"] == 0
    assert repair_audit["page_decisions"] == [
        {
            "logical_page": 2,
            "mode": "existing_complete_page_evidence",
            "ocr_invocations": 0,
            "target_count": 1,
        }
    ]
    owners_by_page = {
        registration["logical_page"]: set(registration["section_table_owners"])
        for registration in result.audit["canonical_layout"]["registrations"]
    }
    assert owners_by_page == {1: {"pt_16_4"}, 2: {"pt_17_1"}}
    assert len(result.business["inquiry_records"]) == 9

    ledger = result.section_content["facts"][
        "personal_detail_source_completeness_ledger"
    ]
    raw_positions = ledger["inquiry_raw_physical_positions"]
    field_observations = ledger["inquiry_physical_field_observations"]
    assert ledger["inquiry_records"] == 26
    assert ledger["inquiry_sequence_endpoints"] == {"institution": 9}
    assert len(raw_positions) == 26
    assert len(field_observations) == 25
    assert all(
        not {"inquiry_type", "sequence", "value"}.intersection(position)
        for position in raw_positions
    )
    assert Counter(
        position["source_refs"][0]["table_id"] for position in raw_positions
    ) == Counter({"pt_16_4": 9, "pt_17_0": 15, "pt_17_1": 2})
    assert Counter(
        observation["source_refs"][0]["table_id"]
        for observation in field_observations
    ) == Counter({"pt_16_4": 9, "pt_17_0": 14, "pt_17_1": 2})
    typed_ordinals = ledger["inquiry_ordinal_observations"]
    assert set(typed_ordinals) == {"institution"}
    assert set(typed_ordinals["institution"]) == {
        str(sequence) for sequence in range(1, 10)
    }
    assert {
        observation["source_refs"][0]["table_id"]
        for observation in typed_ordinals["institution"].values()
    } == {"pt_16_4"}

    inquiry_status = next(
        row
        for row in result.section_content["datasets"][
            "personal_detail_dataset_status"
        ]
        if row["dataset_name"] == "inquiry_records"
    )
    assert {
        key: inquiry_status[key]
        for key in (
            "presence_status",
            "observed_row_count",
            "expected_row_count",
            "reason",
        )
    } == {
        "presence_status": "partial",
        "observed_row_count": 9,
        "expected_row_count": 26,
        "reason": "source_sequence_or_count_gap",
    }

    source = {
        **deepcopy(result.section_content["datasets"]),
        **{
            name: deepcopy(rows)
            for name, rows in result.business.items()
            if isinstance(rows, list)
        },
    }
    projected_datasets = project_personal_detail_datasets(source)
    semantic = personal_detail_semantic_extensions()
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "personal_credit_report",
            "report_subtype": "personal_detail",
            "content_mode": "scanned_ocr",
            "data_dictionary": personal_detail_data_dictionary(),
            **{
                f"personal_detail_v2_expected_{name}_count": len(rows)
                for name, rows in projected_datasets.items()
            },
        },
        "semantic": semantic,
        "datasets": projected_datasets,
        "sections": [],
    }
    sealed_result = seal_parse_result(
        ParseResult(
            entities=DocumentEntities(
                document_type="personal_credit_report_detailed"
            ),
            pages=[PageContent(page_number=1)],
        )
    )
    source_pdf = tmp_path / "yu-static-split.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    projected = project_community_bundle(
        sealed_result,
        file_path=str(source_pdf),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    )
    payload = _CreditReportCommunityBundle(
        schema=projected.schema,
        document=projected.document,
        sections=projected.sections,
        datasets=projected.datasets,
        files=projected.files,
        warnings=projected.warnings,
        result=projected.result,
        source_fingerprint=projected.source_fingerprint,
        parse_result_schema=projected.parse_result_schema,
        classification=projected.classification,
        domain=projected.domain,
        diagnostics=projected.diagnostics,
        content_markdown_override=projected.content_markdown_override,
    ).json_payload()
    public_datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    assert len(public_datasets["inquiries"]["rows"]) == 9
    assert public_datasets["inquiries"]["status"] == "partial"
    assert public_datasets["inquiries"]["completeness"] == {
        "expected_row_count": 26,
        "emitted_row_count": 9,
        "omitted_row_count": 17,
        "verified": False,
        "basis": "personal_detail_dataset_status:partial",
    }
    assert any(
        row["normalized"]["issue_code"] == "source_sequence_or_count_gap"
        and row["normalized"]["target_dataset"] == "inquiries"
        for row in public_datasets["extraction_issues"]["rows"]
    )


@pytest.mark.parametrize(
    ("count", "scale", "column_widths"),
    [
        (2, 0.75, (55.0, 120.0, 80.0, 65.0)),
        (3, 1.0, (90.0, 72.0, 140.0, 88.0)),
        (5, 1.8, (65.0, 95.0, 170.0, 74.0)),
    ],
)
def test_raw_inquiry_census_accepts_variable_population_width_and_scale(
    count: int,
    scale: float,
    column_widths: tuple[float, ...],
) -> None:
    # Semantic columns are deliberately not in the sample report's order.
    headers = ["查询机构", "查询原因", "编号", "查询日期"]
    rows = [
        [f"机构{index}", "贷款审批", str(index + 1), f"2024.01.{index + 1:02d}"]
        for index in range(count)
    ]
    context = _sealed_context(
        rows,
        headers=headers,
        scale=scale,
        column_widths=column_widths,
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert len(census["raw_physical_positions"]) == count
    assert len(census["physical_field_observations"]) == count
    assert all(
        not {"inquiry_type", "sequence", "value"}.intersection(position)
        for position in census["raw_physical_positions"]
    )


def test_headerless_six_column_lattice_requires_one_unique_four_role_map() -> None:
    rows = [
        ["", str(index), f"2024.02.{index:02d}", f"机构{index}", "贷后管理", ""]
        for index in range(1, 4)
    ]
    context = _sealed_context(rows, column_widths=(20, 45, 80, 150, 75, 20))

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert len(census["raw_physical_positions"]) == 3
    assert len(census["physical_field_observations"]) == 3


def test_headerless_three_column_date_institution_merge_is_token_local() -> None:
    rows = [
        [str(index), [f"2024.03.{index:02d}", f"机构{index}"], "信用卡审批"]
        for index in range(1, 4)
    ]
    context = _sealed_context(rows, column_widths=(45, 190, 85))

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert len(census["raw_physical_positions"]) == 3
    assert len(census["physical_field_observations"]) == 3
    assert all(
        observation["field_source_refs"]["inquiry_date"][0]["column"] == 1
        and observation["field_source_refs"]["institution"][0]["column"] == 1
        for observation in census["physical_field_observations"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "competing_envelope",
        "replayed_evidence",
        "topology_gap",
        "topology_reversal",
        "competing_table_owner",
    ],
)
def test_raw_inquiry_census_fails_closed_on_adversarial_ownership(mutation: str) -> None:
    headers = ["编号", "查询日期", "查询机构", "查询原因"]
    context = _sealed_context(
        [
            ["1", "2024.01.01", "机构一", "贷款审批"],
            ["2", "2024.01.02", "机构二", "贷款审批"],
        ],
        headers=headers,
        duplicate_start=mutation == "competing_envelope",
        replay_cell_evidence=mutation == "replayed_evidence",
    )
    if mutation == "topology_gap":
        context.reading_order_by_logical = {1: 2}
    if mutation == "topology_reversal":
        first_page = context._frozen_logical_pages[1]
        boundary = first_page.texts.pop()
        context._frozen_logical_pages[2] = SimpleNamespace(
            page_number=2,
            source_page_number=2,
            width=first_page.width,
            height=first_page.height,
            texts=[boundary],
            tables=[],
        )
        context.reading_order_by_logical = {1: 2, 2: 1}
    if mutation == "competing_table_owner":
        context._frozen_logical_pages[1].tables.append(
            deepcopy(context._frozen_logical_pages[1].tables[0])
        )
        context._frozen_logical_pages[1].tables[1].table_id = "competing-table"

    assert native_extraction._sealed_raw_inquiry_population_census(context) is None


def test_malformed_registered_header_conserves_anonymous_rows_without_semantic_roles() -> None:
    context = _sealed_context(
        [
            ["1", "2024.01.01", "机构一", "贷款审批"],
            ["2", "2024.01.02", "机构二", "贷款审批"],
        ],
        headers=["编号", "? 查询日期 查询机构 X", "", "查询原因"],
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert census["typed_ordinal_groups"] == []
    assert len(census["raw_physical_positions"]) == 2
    assert census["physical_field_observations"] == []
    assert all(
        position["source_refs"][0]["binding"]
        == "sealed_raw_inquiry_anonymous_exact_grid_row"
        for position in census["raw_physical_positions"]
    )
    assert coverage["expected_row_count"] == 2
    assert "sequence_endpoints" not in coverage


def test_malformed_registered_header_does_not_admit_furniture_body_rows() -> None:
    context = _sealed_context(
        [
            ["签字", "复核", "页码", "备注"],
            ["制表人", "审核人", "第1页", "以下空白"],
        ],
        headers=["编号", "? 查询日期 查询机构 X", "", "查询原因"],
    )

    assert native_extraction._sealed_raw_inquiry_population_census(context) is None


def test_exact_headed_population_is_typed_without_materialized_rows() -> None:
    context = _sealed_context(
        [
            ["1", "2024.01.01", "机构一", "贷款审批"],
            ["2", "2024.01.02", "机构二", "贷款审批"],
        ],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )

    coverage = native_extraction._inquiry_source_coverage(context)

    assert len(coverage["raw_physical_positions"]) == 2
    assert coverage["sequence_endpoints"] == {"institution": 2}
    assert coverage["expected_row_count"] == 2
    assert sorted(coverage["ordinal_observations"]["institution"]) == ["1", "2"]


@pytest.mark.parametrize(
    ("institutional_ranges", "personal_count", "expected"),
    [
        (((1, 21), (22, 57), (58, 92), (93, 113)), 4, 117),
        (((1, 29), (30, 59), (60, 89)), 1, 90),
    ],
    ids=("yang-p21-through-p24", "lin-four-column-chain"),
)
def test_sealed_headerless_chain_and_personal_restart_are_materialization_independent(
    institutional_ranges: tuple[tuple[int, int], ...],
    personal_count: int,
    expected: int,
) -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(sequence) for sequence in range(first, last + 1)]
            for first, last in institutional_ranges
        ],
        personal_rows=[
            _inquiry_row(sequence, personal=True)
            for sequence in range(1, personal_count + 1)
        ],
    )

    coverage = native_extraction._inquiry_source_coverage(context)

    institutional_endpoint = institutional_ranges[-1][1]
    assert coverage["sequence_endpoints"] == {
        "institution": institutional_endpoint,
        "personal": personal_count,
    }
    assert coverage["expected_row_count"] == expected
    assert coverage["observed_sequences"]["institution"] == list(
        range(1, institutional_endpoint + 1)
    )
    assert coverage["observed_sequences"]["personal"] == list(
        range(1, personal_count + 1)
    )
    assert len(coverage["raw_physical_positions"]) == expected
    assert all(
        observation["source_refs"][0]["geometry_scope"] == "token"
        for observations in coverage["ordinal_observations"].values()
        for observation in observations.values()
    )


def test_lin_bounded_damaged_chain_conserves_all_90_physical_positions() -> None:
    def institutional_row(sequence: int) -> list[str]:
        row = _inquiry_row(sequence)
        row[2] = "测试银行"
        return row

    context = _sealed_chain_context(
        [
            [institutional_row(sequence) for sequence in range(1, 17)],
            [institutional_row(sequence) for sequence in range(17, 55)],
            [institutional_row(sequence) for sequence in range(55, 90)],
        ],
        personal_rows=[_inquiry_row(1, personal=True)],
        institutional_column_widths={
            0: (40.0, 77.5, 154.5, 78.5),
            1: (40.0, 77.5, 156.5, 80.0),
            2: (40.0, 77.0, 155.0, 78.5),
        },
    )

    # Exact source-visible damage from the Lin continuation chain.  The proof
    # may conserve these physical rows, but it must not repair any ordinal or
    # date, infer an inquiry type, or publish a business value from them.
    for page_number, row, column, text in (
        (1, 2, 1, "2023.01.03 20"),
        (1, 3, 0, ""),
        (1, 4, 0, ""),
        (1, 8, 0, "8 火"),
        (1, 9, 0, "花9"),
        (2, 6, 1, "2022.08:25"),
        (2, 10, 0, "%"),
        (2, 29, 0, "花拾"),
        (3, 11, 0, "6%"),
        (3, 23, 1, "2021:05.18"),
        (3, 27, 1, "202104.25"),
        (3, 32, 0, "0k87"),
    ):
        if text:
            _replace_exact_chain_cell_text(
                context,
                page_number=page_number,
                row=row,
                column=column,
                text=text,
            )
        else:
            _blank_exact_chain_cell(
                context,
                page_number=page_number,
                row=row,
                column=column,
                derived=page_number == 1,
            )
    _blank_exact_chain_cell(context, page_number=3, row=12, column=0)

    envelope, table_entries = _chain_envelope_and_table_entries(context)
    entries_by_id = {
        entry["grid"]["table_id"]: entry
        for entry in table_entries
        if entry.get("grid") is not None
    }
    assert entries_by_id["inst-0"]["descriptor"] is not None
    assert entries_by_id["inst-1"]["descriptor"] is None
    assert entries_by_id["inst-2"]["descriptor"] is None
    assert native_extraction._sealed_raw_inquiry_anonymous_continuation_markers(
        envelope,
        table_entries,
    ) == frozenset({(2, 2, "inst-1"), (3, 3, "inst-2")})

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert coverage["expected_row_count"] == 90
    assert len(census["raw_physical_positions"]) == 90
    position_counts = Counter(
        position["source_refs"][0]["table_id"]
        for position in census["raw_physical_positions"]
    )
    assert position_counts == {
        "inst-0": 16,
        "inst-1": 38,
        "inst-2": 35,
        "personal-2": 1,
    }
    continuation_positions = [
        position
        for position in census["raw_physical_positions"]
        if position["source_refs"][0]["table_id"] in {"inst-1", "inst-2"}
    ]
    assert len(continuation_positions) == 73
    assert all(
        position["source_refs"][0]["binding"]
        == "sealed_raw_inquiry_registered_lattice_band"
        and position["source_refs"][0]["binding_quality"]
        == "sealed_exact_physical_position"
        and not {
            "inquiry_type",
            "sequence",
            "value",
            "raw_value",
            "normalized_value",
            "printed_fields",
            "field_source_refs",
        }.intersection(position)
        for position in continuation_positions
    )
    assert not any(
        proof["table_id"] in {"inst-1", "inst-2"}
        for proof in census["typed_table_proofs"]
    )
    assert not any(
        observation["source_refs"][0]["table_id"] in {"inst-1", "inst-2"}
        for observation in census["physical_field_observations"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "readable_ordinal_gap",
        "too_many_damaged_ordinals",
        "missing_date_shape",
        "no_exact_institution_anchors",
        "two_semantic_furniture_rows",
    ],
)
def test_bounded_damaged_physical_chain_stays_fail_closed(
    mutation: str,
) -> None:
    def institutional_row(sequence: int) -> list[str]:
        row = _inquiry_row(sequence)
        row[2] = "测试银行"
        return row

    context = _sealed_chain_context(
        [
            [institutional_row(sequence) for sequence in range(1, 17)],
            [institutional_row(sequence) for sequence in range(17, 33)],
        ]
    )
    # Force the continuation onto the new physical-position-only path.
    _replace_exact_chain_cell_text(
        context,
        page_number=2,
        row=6,
        column=1,
        text="2024.07:07",
    )
    _replace_exact_chain_cell_text(
        context,
        page_number=2,
        row=8,
        column=0,
        text="%",
    )
    if mutation == "readable_ordinal_gap":
        _replace_exact_chain_cell_text(
            context,
            page_number=2,
            row=5,
            column=0,
            text="23",
        )
    elif mutation == "too_many_damaged_ordinals":
        for row in range(1, 6):
            _replace_exact_chain_cell_text(
                context,
                page_number=2,
                row=row,
                column=0,
                text="x",
            )
    elif mutation == "missing_date_shape":
        _replace_exact_chain_cell_text(
            context,
            page_number=2,
            row=7,
            column=1,
            text="复核",
        )
    elif mutation == "no_exact_institution_anchors":
        for row in range(16):
            _replace_exact_chain_cell_text(
                context,
                page_number=2,
                row=row,
                column=2,
                text="复核",
            )
    elif mutation == "two_semantic_furniture_rows":
        for row in (4, 5):
            _replace_exact_chain_cell_text(
                context,
                page_number=2,
                row=row,
                column=2,
                text="复核",
            )
            _replace_exact_chain_cell_text(
                context,
                page_number=2,
                row=row,
                column=3,
                text="备注",
            )

    envelope, table_entries = _chain_envelope_and_table_entries(context)
    continuation_entry = next(
        entry
        for entry in table_entries
        if (entry.get("grid") or {}).get("table_id") == "inst-1"
    )
    assert continuation_entry["descriptor"] is None
    assert (2, 2, "inst-1") not in (
        native_extraction._sealed_raw_inquiry_anonymous_continuation_markers(
            envelope,
            table_entries,
        )
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    assert not any(
        position["source_refs"][0]["table_id"] == "inst-1"
        for position in census["raw_physical_positions"]
    )


def test_exact_continuation_population_survives_business_field_decode_failures() -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [
                ["3", "not-a-date", "%%%", "not-a-reason"],
                _inquiry_row(4),
            ],
        ]
    )

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage["sequence_endpoints"] == {"institution": 4}
    assert coverage["expected_row_count"] == 4
    assert sorted(coverage["ordinal_observations"]["institution"]) == [
        "1",
        "2",
        "3",
        "4",
    ]
    assert len(coverage["raw_physical_positions"]) == 4


def test_yu_packed_rows_and_split_two_row_personal_header_share_one_contract() -> None:
    context = _yu_packed_split_header_context()

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage["sequence_endpoints"] == {"institution": 2, "personal": 1}
    assert coverage["expected_row_count"] == 3
    assert len(coverage["raw_physical_positions"]) == 3
    assert {
        proof["kind"]
        for proof in native_extraction._sealed_raw_inquiry_population_census(context)[
            "typed_table_proofs"
        ]
    } == {"exact_header_seed"}


def test_split_personal_header_ignores_only_an_empty_derived_terminal_border_cell() -> None:
    context = _yu_packed_split_header_context()
    page = context._frozen_logical_pages[1]
    table = page.tables[1]
    geometry = table.metadata["geometry"]
    rows = table.metadata["raw_rows"]
    bottom = float(geometry["row_bands"][-1]["y1"])
    row_height = 3.0
    rows.append(["", "", "", ""])
    geometry["row_bands"].append(
        {"index": len(rows) - 1, "y0": bottom, "y1": bottom + row_height}
    )
    geometry["cell_bboxes"].append(
        [
            [
                float(geometry["col_bands"][column]["x0"]),
                bottom,
                float(geometry["col_bands"][column]["x1"]),
                bottom + row_height,
            ]
            if column < 3
            else None
            for column in range(4)
        ]
    )
    geometry["cell_geometry_status"].append(["exact", "exact", "exact", "derived"])
    geometry["cell_evidence_ids"].append([[], [], [], []])
    geometry["cell_token_ids"].append([[], [], [], []])
    table.bbox[3] = bottom + row_height

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert census["typed_population_complete"] is True
    assert len(census["raw_physical_positions"]) == 3
    assert len(census["physical_field_observations"]) == 3

    geometry["cell_geometry_status"][-1][3] = "exact"
    geometry["cell_bboxes"][-1][3] = [
        float(geometry["col_bands"][3]["x0"]),
        bottom,
        float(geometry["col_bands"][3]["x1"]),
        bottom + row_height,
    ]
    geometry["cell_geometry_status"][-1][1] = "derived"
    geometry["cell_bboxes"][-1][1] = None
    assert native_extraction._sealed_raw_inquiry_cell_grid(context, page, table) is None


def test_exact_grid_replays_the_producer_token_center_owner_contract() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    page = context._frozen_logical_pages[1]
    table = page.tables[0]
    geometry = table.metadata["geometry"]
    token_id = geometry["cell_token_ids"][1][2][0]
    cell_bbox = geometry["cell_bboxes"][1][2]
    atoms = context.evidence_plane.evidence.text_atoms
    token_index = next(index for index, atom in enumerate(atoms) if atom.id == token_id)
    original = atoms[token_index]
    # The bordered-table producer assigns a token to this exact cell by its
    # centre.  Let the glyph box cross the left rule while retaining that one
    # unambiguous centre owner, matching real scanned-grid output.
    atoms[token_index] = EvidenceAtom(
        id=original.id,
        text=original.text,
        bbox=[cell_bbox[0] - 14.0, original.bbox[1], cell_bbox[0] + 30.0, original.bbox[3]],
    )

    grid = native_extraction._sealed_raw_inquiry_cell_grid(context, page, table)

    assert grid is not None
    assert len(native_extraction._sealed_raw_inquiry_population_census(context)["raw_physical_positions"]) == 1

    atoms[token_index] = EvidenceAtom(
        id=original.id,
        text=original.text,
        bbox=[cell_bbox[0] - 50.0, original.bbox[1], cell_bbox[0] - 10.0, original.bbox[3]],
    )
    assert native_extraction._sealed_raw_inquiry_cell_grid(context, page, table) is None


def test_derived_empty_cell_inside_an_inquiry_business_row_stays_fail_closed() -> None:
    context = _yu_packed_split_header_context()
    page = context._frozen_logical_pages[1]
    table = page.tables[1]
    geometry = table.metadata["geometry"]
    geometry["cell_geometry_status"][2][3] = "derived"
    geometry["cell_bboxes"][2][3] = None
    geometry["cell_evidence_ids"][2][3] = []
    geometry["cell_token_ids"][2][3] = []

    assert native_extraction._sealed_raw_inquiry_cell_grid(context, page, table) is None


def test_yu_packed_cardinality_mismatch_stays_one_anonymous_physical_position() -> None:
    context = _yu_packed_split_header_context(mismatched_reason_bands=True)

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert len(census["raw_physical_positions"]) == 2
    assert not any(
        proof["inquiry_type"] == "institution"
        for proof in census["typed_table_proofs"]
    )
    assert coverage["expected_row_count"] == 2
    assert "sequence_endpoints" not in coverage


@pytest.mark.parametrize(
    ("continuation_sequences", "mutation"),
    [
        ((4, 5), "gap"),
        ((2, 3), "duplicate"),
        ((4, 3), "reorder"),
    ],
)
def test_exact_continuation_rejects_non_dense_ordinals_but_conserves_physical_rows(
    continuation_sequences: tuple[int, int],
    mutation: str,
) -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [_inquiry_row(sequence) for sequence in continuation_sequences],
        ]
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None, mutation
    assert census["typed_population_complete"] is False
    assert len(census["raw_physical_positions"]) == 4
    assert coverage["expected_row_count"] == 4
    assert "sequence_endpoints" not in coverage


def test_exact_headerless_continuation_with_role_contamination_is_conserved_anonymously() -> None:
    def exact_institutional_row(sequence: int) -> list[str]:
        return [
            str(sequence),
            f"2024.{((sequence - 1) % 12) + 1:02d}.{((sequence - 1) % 27) + 1:02d}",
            "测试银行",
            "贷后管理",
        ]

    context = _sealed_chain_context(
        [
            [exact_institutional_row(1), exact_institutional_row(2)],
            [exact_institutional_row(sequence) for sequence in range(3, 7)],
        ]
    )
    continuation_page = context._frozen_logical_pages[2]
    continuation_table = continuation_page.tables[0]
    rows = continuation_table.metadata["raw_rows"]
    geometry = continuation_table.metadata["geometry"]

    # Reproduce the two independent traits from the fresh Yang continuation
    # grids.  One exact ruled cell spans the sequence/date columns and owns both
    # tokens, while one other date token retains a source comma.  The complete
    # four-column business descriptor must therefore stay withheld even though
    # the registered subsection, continuation lattice, cells, and token owners
    # are all exact.
    merged_row = 2
    merged_ids = [
        *geometry["cell_token_ids"][merged_row][0],
        *geometry["cell_token_ids"][merged_row][1],
    ]
    rows[merged_row][0] = f"{rows[merged_row][1]} {rows[merged_row][0]}"
    rows[merged_row][1] = ""
    geometry["cell_bboxes"][merged_row][0][2] = geometry["cell_bboxes"][merged_row][1][2]
    geometry["cell_evidence_ids"][merged_row][0] = list(merged_ids)
    geometry["cell_token_ids"][merged_row][0] = list(merged_ids)
    geometry["cell_geometry_status"][merged_row][1] = "derived"
    geometry["cell_bboxes"][merged_row][1] = None
    geometry["cell_evidence_ids"][merged_row][1] = []
    geometry["cell_token_ids"][merged_row][1] = []
    geometry["cell_spans"] = [
        {"row": merged_row, "col": 0, "row_span": 1, "col_span": 2}
    ]

    noisy_date_row = 1
    noisy_date_id = geometry["cell_token_ids"][noisy_date_row][1][0]
    rows[noisy_date_row][1] = "2024,01.04"
    atoms = context.evidence_plane.evidence.text_atoms
    noisy_atom_index = next(
        index for index, atom in enumerate(atoms) if atom.id == noisy_date_id
    )
    noisy_atom = atoms[noisy_atom_index]
    atoms[noisy_atom_index] = EvidenceAtom(
        id=noisy_atom.id,
        text=rows[noisy_date_row][1],
        bbox=list(noisy_atom.bbox),
    )

    grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        continuation_page,
        continuation_table,
    )
    assert grid is not None
    assert (
        native_extraction._sealed_raw_inquiry_table_descriptor(
            continuation_table,
            grid,
        )
        is None
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert len(census["raw_physical_positions"]) == 6
    continuation_positions = [
        position
        for position in census["raw_physical_positions"]
        if position["source_refs"][0]["table_id"] == "inst-1"
    ]
    assert len(continuation_positions) == 4
    assert {
        (
            position["source_refs"][0]["binding"],
            position["source_refs"][0]["binding_quality"],
        )
        for position in continuation_positions
    } == {
        (
            "sealed_raw_inquiry_registered_lattice_band",
            "sealed_exact_physical_position",
        )
    }
    assert all(
        not {
            "inquiry_type",
            "sequence",
            "value",
            "raw_value",
            "normalized_value",
            "printed_fields",
            "field_source_refs",
        }.intersection(position)
        for position in continuation_positions
    )
    assert not any(
        observation["source_refs"][0]["table_id"] == "inst-1"
        for observation in census["physical_field_observations"]
    )
    assert coverage["expected_row_count"] == 6
    assert "sequence_endpoints" not in coverage

    content = _issue_content(census, include_fields=True)
    prepare_personal_detail_source_collections(content)
    continuation_issues = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        and (issue.get("source_refs") or [{}])[0].get("table_id") == "inst-1"
    ]
    assert len(continuation_issues) == 4
    assert {
        issue["target_record_id"] for issue in continuation_issues
    } == {
        position["source_physical_row_id"] for position in continuation_positions
    }
    assert all(
        issue["observed_value"] == {"source_row_observed": True}
        and issue["candidate_value"]
        == {"source_physical_row_id": issue["target_record_id"]}
        and {
            "inquiry_type_withheld",
            "ordinal_withheld",
            "normalized_value_withheld",
        }.issubset(issue["reason_codes"])
        for issue in continuation_issues
    )


def test_yang_population_only_continuation_conserves_all_117_live_shaped_rows() -> None:
    def institutional_row(sequence: int) -> list[object]:
        date = f"2024.{((sequence - 1) % 12) + 1:02d}.{((sequence - 1) % 27) + 1:02d}"
        row: list[object] = [str(sequence), date, "测试银行", "贷后管理"]
        if sequence in {22, 28}:
            row[3] = "法人代表、负责人、高管等资审查"
        if sequence in {23, 26}:
            row[1] = date.replace(".", ",", 1)
        if sequence == 29:
            row[1] = [date, "正"]
        if sequence == 45:
            row[1] = ["5", date]
        if sequence == 51:
            row[0] = "051"
        return row

    context = _sealed_chain_context(
        [
            [institutional_row(sequence) for sequence in range(1, 22)],
            [institutional_row(sequence) for sequence in range(22, 58)],
            [institutional_row(sequence) for sequence in range(58, 93)],
            [institutional_row(sequence) for sequence in range(93, 114)],
        ],
        personal_rows=[
            _inquiry_row(sequence, personal=True)
            for sequence in range(1, 5)
        ],
        institutional_column_widths={
            0: (40.0, 79.0, 154.5, 78.5),
            1: (39.0, 78.0, 154.0, 79.0),
            2: (40.0, 79.0, 154.0, 79.5),
            3: (39.0, 78.0, 154.0, 78.5),
        },
    )
    # The exact headed seed contains the institution/reason spans seen live.
    _merge_exact_chain_cells(context, page_number=1, row=5, column=2)
    # The page-22 continuation combines sequence/date once and
    # date/institution three times; all are exact two-column source owners.
    _merge_exact_chain_cells(context, page_number=2, row=22, column=0)
    for row in (24, 33, 35):
        _merge_exact_chain_cells(context, page_number=2, row=row, column=1)

    seed_page = context._frozen_logical_pages[1]
    continuation_page = context._frozen_logical_pages[2]
    seed_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        seed_page,
        seed_page.tables[0],
    )
    continuation_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        continuation_page,
        continuation_page.tables[0],
    )
    assert seed_grid is not None
    assert continuation_grid is not None
    seed_descriptor = native_extraction._sealed_raw_inquiry_table_descriptor(
        seed_page.tables[0],
        seed_grid,
    )
    assert seed_descriptor is not None
    assert (
        native_extraction._sealed_raw_inquiry_table_descriptor(
            continuation_page.tables[0],
            continuation_grid,
        )
        is None
    )
    assert (
        native_extraction._sealed_raw_inquiry_table_descriptor(
            continuation_page.tables[0],
            continuation_grid,
        )
        is None
    )
    assert not native_extraction._sealed_raw_inquiry_same_column_lattice(
        continuation_grid,
        seed_grid,
    )
    assert native_extraction._sealed_raw_inquiry_same_population_column_lattice(
        continuation_grid,
        seed_grid,
    )
    assert (
        native_extraction._sealed_raw_inquiry_population_only_continuation_endpoint(
            continuation_grid,
            seed_grid,
            seed_descriptor,
            expected_first_sequence=22,
        )
        == 57
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert len(census["raw_physical_positions"]) == 117
    continuation_positions = [
        position
        for position in census["raw_physical_positions"]
        if position["source_refs"][0]["table_id"] == "inst-1"
    ]
    assert len(continuation_positions) == 36
    assert [
        position["source_refs"][0]["row"]
        for position in continuation_positions
    ] == list(range(36))
    assert all(
        position["source_refs"][0]["binding"]
        == "sealed_raw_inquiry_registered_lattice_band"
        and not {
            "inquiry_type",
            "sequence",
            "value",
            "raw_value",
            "normalized_value",
            "printed_fields",
            "field_source_refs",
        }.intersection(position)
        for position in continuation_positions
    )
    assert not any(
        observation["source_refs"][0]["table_id"] == "inst-1"
        for observation in census["physical_field_observations"]
    )
    assert not any(
        proof["table_id"] == "inst-1"
        for proof in census["typed_table_proofs"]
    )
    assert coverage["expected_row_count"] == 117


@pytest.mark.parametrize(
    ("mutation", "continuation_rows"),
    [
        (
            "numbered_date_furniture",
            [
                ["3", "2024.05.01", "复核", "贷后管理"],
                ["4", "2024.04.01", "制表人", "贷后管理"],
            ],
        ),
        (
            "ordinal_gap",
            [
                ["4", "2024.05.01", "测试银行", "贷后管理"],
                ["5", "2024.04.01", "测试银行", "贷后管理"],
            ],
        ),
        (
            "ordinal_reorder",
            [
                ["4", "2024.05.01", "测试银行", "贷后管理"],
                ["3", "2024.04.01", "测试银行", "贷后管理"],
            ],
        ),
        (
            "missing_reason_anchors",
            [
                ["3", "2024.05.01", "测试银行", "备注"],
                ["4", "2024.04.01", "测试银行", "复核"],
            ],
        ),
        (
            "valid_prefix_with_furniture_tail",
            [
                ["3", "2024.05.01", "测试银行", "贷后管理"],
                ["4", "2024.04.01", "测试银行", "贷款审批"],
                ["5", "2024.03.01", "复核", "备注"],
                ["6", "2024.02.01", "制表人", "以下空白"],
            ],
        ),
    ],
)
def test_population_only_continuation_stays_closed_on_adversarial_tables(
    mutation: str,
    continuation_rows: list[list[object]],
) -> None:
    context = _population_only_chain_context(continuation_rows)

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None, mutation
    assert not any(
        position["source_refs"][0]["table_id"] == "inst-1"
        for position in census["raw_physical_positions"]
    ), mutation


def test_population_only_continuation_requires_dense_endpoint_at_exact_width() -> None:
    context = _population_only_chain_context(
        [
            ["4", "2024.05.01", "测试银行", "贷后管理"],
            ["5", "2024.04.01", "测试银行", "贷款审批"],
        ],
        exact_widths=True,
    )
    seed_page = context._frozen_logical_pages[1]
    continuation_page = context._frozen_logical_pages[2]
    seed_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        seed_page,
        seed_page.tables[0],
    )
    continuation_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        continuation_page,
        continuation_page.tables[0],
    )
    assert seed_grid is not None
    assert continuation_grid is not None
    seed_descriptor = native_extraction._sealed_raw_inquiry_table_descriptor(
        seed_page.tables[0],
        seed_grid,
    )
    assert seed_descriptor is not None
    assert (
        native_extraction._sealed_raw_inquiry_table_descriptor(
            continuation_page.tables[0],
            continuation_grid,
        )
        is None
    )
    assert native_extraction._sealed_raw_inquiry_headerless_continuation_grid(
        continuation_grid,
        seed_grid,
        seed_descriptor,
    )
    assert (
        native_extraction._sealed_raw_inquiry_population_only_continuation_endpoint(
            continuation_grid,
            seed_grid,
            seed_descriptor,
            expected_first_sequence=3,
        )
        is None
    )
    envelope, table_entries = _chain_envelope_and_table_entries(context)
    continuation_marker = (2, 2, "inst-1")
    assert continuation_marker not in (
        native_extraction._sealed_raw_inquiry_anonymous_continuation_markers(
            envelope,
            table_entries,
        )
    )
    assert not any(
        proof["table_id"] == "inst-1"
        for proof in native_extraction._sealed_raw_inquiry_typed_population(
            envelope,
            table_entries,
        )["proofs"]
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert not any(
        position["source_refs"][0]["table_id"] == "inst-1"
        for position in census["raw_physical_positions"]
    )


def test_population_only_continuation_requires_authenticated_seed_endpoint() -> None:
    context = _population_only_chain_context(
        [
            ["3", "2024.05.01", "测试银行", "贷后管理"],
            ["4", "2024.04.01", "测试银行", "贷款审批"],
        ],
        exact_widths=True,
        seed_rows=[
            ["1", "2024.07.01", "机构一", "贷后管理"],
            ["2", "2024.06.01", "机构二", "贷款审批"],
        ],
    )
    seed_page = context._frozen_logical_pages[1]
    continuation_page = context._frozen_logical_pages[2]
    seed_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        seed_page,
        seed_page.tables[0],
    )
    continuation_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        continuation_page,
        continuation_page.tables[0],
    )
    assert seed_grid is not None
    assert continuation_grid is not None
    seed_descriptor = native_extraction._sealed_raw_inquiry_table_descriptor(
        seed_page.tables[0],
        seed_grid,
    )
    assert seed_descriptor is not None
    assert (
        native_extraction._sealed_raw_inquiry_table_descriptor(
            continuation_page.tables[0],
            continuation_grid,
        )
        is None
    )
    assert native_extraction._sealed_raw_inquiry_headerless_continuation_grid(
        continuation_grid,
        seed_grid,
        seed_descriptor,
    )
    assert (
        native_extraction._sealed_raw_inquiry_population_only_sequence_endpoint(
            seed_grid,
            seed_descriptor,
            expected_first_sequence=1,
        )
        is None
    )
    envelope, table_entries = _chain_envelope_and_table_entries(context)
    assert (2, 2, "inst-1") not in (
        native_extraction._sealed_raw_inquiry_anonymous_continuation_markers(
            envelope,
            table_entries,
        )
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert not any(
        position["source_refs"][0]["table_id"] == "inst-1"
        for position in census["raw_physical_positions"]
    )


@pytest.mark.parametrize(
    ("continuation_sequences", "admitted"),
    [
        ((3, 4), False),
        ((5, 6), True),
    ],
)
def test_population_only_chain_advances_endpoint_across_repeated_exact_header(
    continuation_sequences: tuple[int, int],
    admitted: bool,
) -> None:
    def exact_institutional_row(sequence: int) -> list[str]:
        return [
            str(sequence),
            f"2024.{((sequence - 1) % 12) + 1:02d}.{((sequence - 1) % 27) + 1:02d}",
            "测试银行",
            "贷后管理" if sequence % 2 else "贷款审批",
        ]

    widths = (40.0, 79.0, 154.5, 78.5)
    context = _sealed_chain_context(
        [
            [exact_institutional_row(1), exact_institutional_row(2)],
            [exact_institutional_row(3), exact_institutional_row(4)],
            [exact_institutional_row(sequence) for sequence in continuation_sequences],
        ],
        continuation_headers={1: ["编号", "查询日期", "查询机构", "查询原因"]},
        institutional_column_widths={0: widths, 1: widths, 2: widths},
    )
    _merge_exact_chain_cells(
        context,
        page_number=3,
        row=0,
        column=0,
    )
    _replace_exact_chain_cell_text(
        context,
        page_number=3,
        row=1,
        column=1,
        text="2024,06.06",
    )
    envelope, table_entries = _chain_envelope_and_table_entries(context)
    continuation_markers = (
        native_extraction._sealed_raw_inquiry_anonymous_continuation_markers(
            envelope,
            table_entries,
        )
    )
    assert ((3, 3, "inst-2") in continuation_markers) is admitted
    tail_entry = next(
        entry for entry in table_entries if entry["grid"]["table_id"] == "inst-2"
    )
    assert tail_entry["descriptor"] is None
    typed = native_extraction._sealed_raw_inquiry_typed_population(
        envelope,
        table_entries,
    )
    assert not any(proof["table_id"] == "inst-2" for proof in typed["proofs"])

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    continuation_positions = [
        position
        for position in census["raw_physical_positions"]
        if position["source_refs"][0]["table_id"] == "inst-2"
    ]
    assert len(continuation_positions) == (2 if admitted else 0)
    if admitted:
        assert all(
            position["source_refs"][0]["binding"]
            == "sealed_raw_inquiry_registered_lattice_band"
            and not {
                "inquiry_type",
                "sequence",
                "value",
                "raw_value",
                "normalized_value",
                "printed_fields",
                "field_source_refs",
            }.intersection(position)
            for position in continuation_positions
        )
        assert not any(
            observation["source_refs"][0]["table_id"] == "inst-2"
            for observation in census["physical_field_observations"]
        )
        assert not any(
            proof["table_id"] == "inst-2" for proof in census["typed_table_proofs"]
        )


def test_population_only_continuation_stays_closed_on_replayed_evidence() -> None:
    context = _population_only_chain_context(
        [
            ["3", "2024.05.01", "测试银行", "贷后管理"],
            ["4", "2024.04.01", "测试银行", "贷后管理"],
        ]
    )
    seed_id = "cell:inst-0:1:0:0"
    continuation_geometry = context._frozen_logical_pages[2].tables[0].metadata[
        "geometry"
    ]
    continuation_geometry["cell_evidence_ids"][1][3] = [seed_id]
    continuation_geometry["cell_token_ids"][1][3] = [seed_id]

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert not any(
        position["source_refs"][0]["table_id"] == "inst-1"
        for position in census["raw_physical_positions"]
    )


@pytest.mark.parametrize(
    ("furniture_table_id", "duplicate_identity"),
    [
        ("inst-1", True),
        ("inst-furniture", False),
    ],
    ids=("duplicate-id-fails-closed", "distinct-id-preserved"),
)
def test_population_only_marker_cannot_authorize_a_second_table_with_colliding_identity(
    furniture_table_id: str,
    duplicate_identity: bool,
) -> None:
    context = _population_only_chain_context(
        [
            ["3", "2024.05.01", "测试银行", "贷后管理"],
            ["4", "2024.04.01", "测试银行", "贷款审批"],
        ]
    )
    _append_chain_furniture_table(context, table_id=furniture_table_id)
    envelope, table_entries = _chain_envelope_and_table_entries(context)
    continuation_entries = [
        entry
        for entry in table_entries
        if entry["grid"] is not None and entry["grid"]["logical_page"] == 2
    ]
    assert len(continuation_entries) == 2
    assert all(entry["descriptor"] is None for entry in continuation_entries)
    assert set(continuation_entries[0]["grid"]["evidence_ids"]).isdisjoint(
        continuation_entries[1]["grid"]["evidence_ids"]
    )
    markers = native_extraction._sealed_raw_inquiry_anonymous_continuation_markers(
        envelope,
        table_entries,
    )
    assert (2, 2, "inst-1") in markers
    assert (2, 2, "inst-furniture") not in markers

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    if duplicate_identity:
        assert census is None
    else:
        assert census is not None
        assert Counter(
            position["source_refs"][0]["table_id"]
            for position in census["raw_physical_positions"]
        ) == Counter({"inst-0": 2, "inst-1": 2})


def test_exact_headerless_continuation_rejects_four_column_furniture() -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [
                ["签字", "复核", "页码", "备注"],
                ["制表人", "审核人", "第2页", "以下空白"],
            ],
        ]
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)
    coverage = native_extraction._inquiry_source_coverage(context)

    assert census is not None
    assert len(census["raw_physical_positions"]) == 2
    assert not any(
        position["source_refs"][0]["table_id"] == "inst-1"
        for position in census["raw_physical_positions"]
    )
    assert coverage["expected_row_count"] == 2


def test_exact_headerless_continuation_requires_seed_column_lattice() -> None:
    seed_context = _sealed_context(
        [_inquiry_row(1), _inquiry_row(2)],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    candidate_context = _sealed_context(
        [_inquiry_row(3), _inquiry_row(4)],
        column_widths=(35.0, 180.0, 55.0, 120.0),
    )
    seed_page = seed_context._frozen_logical_pages[1]
    candidate_page = candidate_context._frozen_logical_pages[1]
    seed_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        seed_context,
        seed_page,
        seed_page.tables[0],
    )
    candidate_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        candidate_context,
        candidate_page,
        candidate_page.tables[0],
    )
    assert seed_grid is not None
    assert candidate_grid is not None
    seed_descriptor = native_extraction._sealed_raw_inquiry_table_descriptor(
        seed_page.tables[0],
        seed_grid,
    )
    assert seed_descriptor is not None

    assert not native_extraction._sealed_raw_inquiry_headerless_continuation_grid(
        candidate_grid,
        seed_grid,
        seed_descriptor,
    )


def test_yu_date_classification_does_not_widen_continuation_lattice() -> None:
    def grid(widths: tuple[float, ...]) -> dict[str, object]:
        edges = [30.0]
        for width in widths:
            edges.append(edges[-1] + width)
        return {
            "table_bbox": (edges[0], 0.0, edges[-1], 20.0),
            "column_bands": {
                index: (edges[index], edges[index + 1])
                for index in range(len(widths))
            },
        }

    seed = grid((39.0, 76.5, 152.5, 78.0))
    yu_continuation = grid((40.5, 79.5, 152.0, 78.5))
    over_cap = grid((39.0, 81.0, 148.0, 78.0))
    narrow_seed = grid((40.0, 5.0, 225.0, 80.0))
    narrow_shift = grid((43.0, 5.0, 222.0, 80.0))

    assert not native_extraction._raw_inquiry_sequence_cell("2021,03.06")
    assert native_extraction._sealed_raw_inquiry_source_date_token("2021,03.06")
    assert native_extraction._raw_inquiry_exact_date_tokens("2021,03.06") == ()
    assert not native_extraction._sealed_raw_inquiry_same_column_lattice(
        yu_continuation,
        seed,
    )
    assert not native_extraction._sealed_raw_inquiry_same_column_lattice(
        over_cap,
        seed,
    )
    # The classification-only repair must not relax either ordinary or narrow
    # distorted continuation lattices.
    assert not native_extraction._sealed_raw_inquiry_same_column_lattice(
        narrow_shift,
        narrow_seed,
    )


def test_invalid_comma_calendar_cannot_authorize_headerless_inquiry_descriptor() -> None:
    context = _sealed_context(
        [
            ["10", "2021.03.06", "机构十", "贷款审批"],
            ["11", "2021,99,99", "机构十一", "贷后管理"],
            ["12", "2021.01.03", "机构十二", "信用卡审批"],
        ]
    )
    page = context._frozen_logical_pages[1]
    table = page.tables[0]
    grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        page,
        table,
    )

    assert grid is not None
    assert not native_extraction._sealed_raw_inquiry_source_date_token("2021,99,99")
    assert native_extraction._raw_inquiry_sequence_cell("2021,99,99")
    assert (
        native_extraction._sealed_raw_inquiry_table_descriptor(
            table,
            grid,
        )
        is None
    )


def test_packed_inquiry_position_split_requires_complete_role_band_cardinality() -> None:
    context = _sealed_context(
        [
            [
                ["1", "2", "3"],
                ["2024.01.01", "2024.01.02", "2024.01.03"],
                ["机构一", "机构二"],
                ["贷款审批", "贷后管理", "信用卡审批"],
            ]
        ],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    page = context._frozen_logical_pages[1]
    table = page.tables[0]
    grid = native_extraction._sealed_raw_inquiry_cell_grid(
        context,
        page,
        table,
    )
    assert grid is not None
    descriptor = native_extraction._sealed_raw_inquiry_table_descriptor(
        table,
        grid,
    )
    assert descriptor is not None

    positions, field_observations = native_extraction._sealed_raw_inquiry_row_observations(
        grid,
        descriptor,
        row_index=1,
    )

    assert len(positions) == 1
    assert positions[0]["source_refs"][0]["geometry_scope"] == "exact_source_row_band"
    assert field_observations == []


def test_exact_headerless_continuation_rejects_inactive_column_span() -> None:
    widths = (20.0, 45.0, 85.0, 145.0, 75.0, 20.0)
    seed_context = _sealed_context(
        [["", "1", "2024.01.01", "机构一", "贷款审批", ""]],
        headers=["", "编号", "查询日期", "查询机构", "查询原因", ""],
        column_widths=widths,
    )
    candidate_context = _sealed_context(
        [["", "2", "2024.01.02", "机构二", "贷后管理", ""]],
        column_widths=widths,
    )
    candidate_page = candidate_context._frozen_logical_pages[1]
    candidate_table = candidate_page.tables[0]
    geometry = candidate_table.metadata["geometry"]
    reason_id = geometry["cell_token_ids"][0][4][0]
    geometry["cell_bboxes"][0][4][2] = geometry["cell_bboxes"][0][5][2]
    geometry["cell_geometry_status"][0][5] = "derived"
    geometry["cell_bboxes"][0][5] = None
    geometry["cell_evidence_ids"][0][5] = []
    geometry["cell_token_ids"][0][5] = []
    geometry["cell_spans"] = [
        {"row": 0, "col": 4, "row_span": 1, "col_span": 2}
    ]
    inactive_band = geometry["col_bands"][5]
    candidate_context.evidence_plane.evidence.text_atoms = [
        EvidenceAtom(
            id=atom.id,
            text=atom.text,
            bbox=[
                float(inactive_band["x0"]) + 2.0,
                float(atom.bbox[1]),
                float(inactive_band["x1"]) - 2.0,
                float(atom.bbox[3]),
            ],
        )
        if atom.id == reason_id
        else atom
        for atom in candidate_context.evidence_plane.evidence.text_atoms
    ]
    seed_page = seed_context._frozen_logical_pages[1]
    seed_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        seed_context,
        seed_page,
        seed_page.tables[0],
    )
    candidate_grid = native_extraction._sealed_raw_inquiry_cell_grid(
        candidate_context,
        candidate_page,
        candidate_table,
    )
    assert seed_grid is not None
    assert candidate_grid is not None
    seed_descriptor = native_extraction._sealed_raw_inquiry_table_descriptor(
        seed_page.tables[0],
        seed_grid,
    )
    assert seed_descriptor is not None

    assert not native_extraction._sealed_raw_inquiry_headerless_continuation_grid(
        candidate_grid,
        seed_grid,
        seed_descriptor,
    )


def test_exact_continuation_rejects_malformed_residual_header() -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [_inquiry_row(3), _inquiry_row(4)],
        ],
        continuation_headers={
            1: ["编号", "? 查询日期 查询机构 X", "", "查询原因"]
        },
    )

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert not any(
        proof["table_id"] == "inst-1"
        for proof in census["typed_table_proofs"]
    )


def test_exact_continuation_rejects_nonadjacent_page_topology() -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [_inquiry_row(3), _inquiry_row(4)],
        ]
    )
    second_page = context._frozen_logical_pages.pop(2)
    second_page.page_number = 3
    second_page.source_page_number = 3
    context._frozen_logical_pages[2] = SimpleNamespace(
        page_number=2,
        source_page_number=2,
        width=460.0,
        height=200.0,
        texts=[],
        tables=[],
    )
    context._frozen_logical_pages[3] = second_page
    context.reading_order_by_logical = {1: 1, 2: 2, 3: 3}

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert len(census["raw_physical_positions"]) == 4


def test_exact_continuation_stops_at_registered_section_boundary() -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [_inquiry_row(3), _inquiry_row(4)],
        ]
    )
    final_page = context._frozen_logical_pages[2]
    boundary = final_page.texts.pop()
    first_page = context._frozen_logical_pages[1]
    boundary.bbox = [30.0, first_page.tables[0].bbox[3] + 10.0, 230.0, first_page.tables[0].bbox[3] + 18.0]
    first_page.texts.append(boundary)

    coverage = native_extraction._inquiry_source_coverage(context)

    assert coverage["sequence_endpoints"] == {"institution": 2}
    assert coverage["expected_row_count"] == 2
    assert len(coverage["raw_physical_positions"]) == 2


def test_replayed_table_evidence_cannot_authorize_continuation() -> None:
    context = _sealed_chain_context(
        [
            [_inquiry_row(1), _inquiry_row(2)],
            [_inquiry_row(3), _inquiry_row(4)],
        ]
    )
    seed_id = "cell:inst-0:1:0:0"
    continuation_geometry = context._frozen_logical_pages[2].tables[0].metadata["geometry"]
    continuation_geometry["cell_evidence_ids"][0][0] = [seed_id]
    continuation_geometry["cell_token_ids"][0][0] = [seed_id]

    census = native_extraction._sealed_raw_inquiry_population_census(context)

    assert census is not None
    assert census["typed_population_complete"] is False
    assert not any(
        proof["table_id"] == "inst-1"
        for proof in census["typed_table_proofs"]
    )


def _issue_content(
    census: dict[str, object],
    *,
    include_fields: bool,
    inquiry_records: list[dict[str, object]] | None = None,
    extra_ledger: dict[str, object] | None = None,
) -> dict[str, object]:
    ledger: dict[str, object] = {
        "inquiry_raw_physical_positions": deepcopy(census["raw_physical_positions"]),
    }
    if include_fields:
        ledger["inquiry_physical_field_observations"] = deepcopy(
            census["physical_field_observations"]
        )
    ledger.update(extra_ledger or {})
    return {
        "facts": {"personal_detail_source_completeness_ledger": ledger},
        "datasets": {"inquiry_records": inquiry_records or []},
    }


def _typed_ordinal_ref(
    context: SimpleNamespace,
    census: dict[str, object],
) -> dict[str, object]:
    position = census["raw_physical_positions"][0]
    owner_ids = set(position["source_refs"][0]["evidence_ids"])
    field_ids = {
        evidence_id
        for refs in census["physical_field_observations"][0]["field_source_refs"].values()
        for evidence_id in refs[0]["evidence_ids"]
    }
    sequence_id = next(iter(owner_ids - field_ids))
    sequence_atom = next(
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id == sequence_id
    )
    return {
        "source": "native_detail_inquiry_token_ordinal",
        "logical_page": 1,
        "source_page": 1,
        "table_id": "raw-inquiry",
        "row": 1,
        "column": 0,
        "sequence": 1,
        "bbox": list(sequence_atom.bbox),
        "geometry_scope": "token",
        "evidence_ids": [sequence_id],
        "binding": "printed_inquiry_ordinal_token",
        "binding_quality": "exact_token_in_sequence_cell",
    }


def _typed_identity_content(
    census: dict[str, object],
    source_ref: dict[str, object],
) -> dict[str, object]:
    return _issue_content(
        census,
        include_fields=False,
        extra_ledger={
            "inquiry_sequence_endpoints": {"institution": 1},
            "inquiry_observed_sequences": {"institution": [1]},
            "inquiry_ordinal_observations": {
                "institution": {
                    "1": {
                        "sequence": 1,
                        "inquiry_type": "institution",
                        "source_refs": [source_ref],
                    }
                }
            },
        },
    )


def test_complete_raw_fields_replace_anonymous_identity_issue() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    content = _issue_content(census, include_fields=True)

    prepare_personal_detail_source_collections(content)

    issues = content["datasets"]["personal_detail_extraction_issues"]
    physical_id = census["raw_physical_positions"][0]["source_physical_row_id"]
    assert not any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        and issue.get("target_record_id") == physical_id
        for issue in issues
    )
    assert sum(
        issue.get("issue_code") == "source_inquiry_field_omitted"
        and issue.get("target_record_id") == physical_id
        for issue in issues
    ) == 3


def test_anonymous_row_issue_contains_no_guessed_identity_or_value() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    content = _issue_content(census, include_fields=False)

    prepare_personal_detail_source_collections(content)

    issues = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_inquiry_physical_record_omitted"
    ]
    assert len(issues) == 1
    assert issues[0]["observed_value"] == {"source_row_observed": True}
    assert set(issues[0]["candidate_value"]) == {"source_physical_row_id"}
    assert len(issues[0]["source_refs"]) == 1


def test_malformed_header_anonymous_row_is_localized_once_until_consumed() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "? 查询日期 查询机构 X", "", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    position = census["raw_physical_positions"][0]
    source_ref = position["source_refs"][0]
    assert source_ref["binding"] == "sealed_raw_inquiry_anonymous_exact_grid_row"

    content = _issue_content(census, include_fields=False)
    prepare_personal_detail_source_collections(content)

    issues = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_inquiry_physical_record_omitted"
    ]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["target_record_id"] == position["source_physical_row_id"]
    assert issue["observed_value"] == {"source_row_observed": True}
    assert issue["candidate_value"] == {
        "source_physical_row_id": position["source_physical_row_id"]
    }
    assert issue["source_refs"] == [source_ref]
    assert {
        "inquiry_type_withheld",
        "ordinal_withheld",
        "normalized_value_withheld",
    }.issubset(issue["reason_codes"])
    assert not {
        "inquiry_type",
        "sequence",
        "value",
        "raw_value",
        "normalized_value",
    }.intersection(issue["observed_value"] | issue["candidate_value"])

    consumed = _issue_content(
        census,
        include_fields=False,
        inquiry_records=[{"source_refs": [deepcopy(source_ref)]}],
    )
    prepare_personal_detail_source_collections(consumed)
    assert not any(
        item.get("issue_code") == "source_inquiry_physical_record_omitted"
        for item in consumed["datasets"]["personal_detail_extraction_issues"]
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("source", "native_detail_inquiry_row"),
        ("geometry_scope", "row"),
        ("binding", "sealed_raw_inquiry_registered_lattice_band"),
        ("binding_quality", "sealed_exact_physical_position"),
    ],
)
def test_malformed_header_anonymous_row_contract_fails_closed(
    field_name: str,
    bad_value: str,
) -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "? 查询日期 查询机构 X", "", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    mutated = deepcopy(census)
    mutated["raw_physical_positions"][0]["source_refs"][0][field_name] = bad_value
    content = _issue_content(mutated, include_fields=False)

    prepare_personal_detail_source_collections(content)

    assert not any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def test_emitted_or_typed_evidence_suppresses_anonymous_identity_issue() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    position = census["raw_physical_positions"][0]
    owner_ids = set(position["source_refs"][0]["evidence_ids"])
    field_ids = {
        evidence_id
        for refs in census["physical_field_observations"][0]["field_source_refs"].values()
        for evidence_id in refs[0]["evidence_ids"]
    }
    sequence_id = next(iter(owner_ids - field_ids))
    sequence_atom = next(
        atom
        for atom in context.evidence_plane.evidence.text_atoms
        if atom.id == sequence_id
    )
    typed_ref = {
        "source": "native_detail_inquiry_token_ordinal",
        "logical_page": 1,
        "source_page": 1,
        "table_id": "raw-inquiry",
        "row": 1,
        "column": 0,
        "sequence": 1,
        "bbox": list(sequence_atom.bbox),
        "geometry_scope": "token",
        "evidence_ids": [sequence_id],
        "binding": "printed_inquiry_ordinal_token",
        "binding_quality": "exact_token_in_sequence_cell",
    }
    typed_content = _issue_content(
        census,
        include_fields=False,
        extra_ledger={
            "inquiry_sequence_endpoints": {"institution": 1},
            "inquiry_observed_sequences": {"institution": [1]},
            "inquiry_ordinal_observations": {
                "institution": {
                    "1": {
                        "sequence": 1,
                        "inquiry_type": "institution",
                        "source_refs": [typed_ref],
                    }
                }
            },
        },
    )
    prepare_personal_detail_source_collections(typed_content)
    typed_issues = typed_content["datasets"]["personal_detail_extraction_issues"]
    assert not any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        for issue in typed_issues
    )
    assert sum(
        issue.get("issue_code") == "source_inquiry_record_omitted"
        for issue in typed_issues
    ) == 1

    emitted_content = _issue_content(
        census,
        include_fields=False,
        inquiry_records=[
            {
                "source_refs": [deepcopy(position["source_refs"][0])]
            }
        ],
    )
    prepare_personal_detail_source_collections(emitted_content)
    assert not any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        for issue in emitted_content["datasets"]["personal_detail_extraction_issues"]
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("logical_page", 2),
        ("source_page", 2),
        ("table_id", "other-table"),
        ("row", 2),
        ("column", True),
        ("sequence", 2),
    ],
)
def test_cross_row_typed_ordinal_cannot_suppress_anonymous_identity_issue(
    field_name: str,
    bad_value: object,
) -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    position = census["raw_physical_positions"][0]
    typed_ref = _typed_ordinal_ref(context, census)
    typed_ref[field_name] = bad_value
    content = _typed_identity_content(census, typed_ref)

    prepare_personal_detail_source_collections(content)

    assert any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        and issue.get("target_record_id") == position["source_physical_row_id"]
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def test_typed_ordinal_requires_one_exact_evidence_atom() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    typed_ref = _typed_ordinal_ref(context, census)
    typed_ref["evidence_ids"] = [*typed_ref["evidence_ids"], "foreign-atom"]
    content = _typed_identity_content(census, typed_ref)

    prepare_personal_detail_source_collections(content)

    assert any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


@pytest.mark.parametrize(
    ("partial_evidence", "wrong_row", "expected_omission"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_typed_full_row_requires_complete_same_row_owner(
    partial_evidence: bool,
    wrong_row: bool,
    expected_omission: bool,
) -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    owner_ref = census["raw_physical_positions"][0]["source_refs"][0]
    evidence_ids = list(owner_ref["evidence_ids"])
    typed_ref = {
        "source": "native_detail_table",
        "logical_page": owner_ref["logical_page"],
        "source_page": owner_ref["source_page"],
        "table_id": owner_ref["table_id"],
        "row": owner_ref["row"] + int(wrong_row),
        "bbox": list(owner_ref["bbox"]),
        "geometry_scope": "row",
        "evidence_ids": evidence_ids[:1] if partial_evidence else evidence_ids,
        "binding": "canonical_header_row",
    }
    content = _typed_identity_content(census, typed_ref)

    prepare_personal_detail_source_collections(content)

    assert (
        any(
            issue.get("issue_code") == "source_inquiry_physical_record_omitted"
            for issue in content["datasets"]["personal_detail_extraction_issues"]
        )
        is expected_omission
    )


def test_stitched_cross_row_emitted_evidence_cannot_suppress_anonymous_issue() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    owner_ref = census["raw_physical_positions"][0]["source_refs"][0]
    evidence_ids = list(owner_ref["evidence_ids"])
    split = max(1, len(evidence_ids) // 2)
    first_ref = deepcopy(owner_ref)
    first_ref["evidence_ids"] = evidence_ids[:split]
    second_ref = deepcopy(owner_ref)
    second_ref["row"] += 1
    second_ref["evidence_ids"] = evidence_ids[split:]
    content = _issue_content(
        census,
        include_fields=False,
        inquiry_records=[{"source_refs": [first_ref, second_ref]}],
    )

    prepare_personal_detail_source_collections(content)

    assert any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )


def test_partial_emitted_evidence_cannot_suppress_anonymous_identity_issue() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "? 查询日期 查询机构 X", "", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    position = census["raw_physical_positions"][0]
    owner_ref = deepcopy(position["source_refs"][0])
    owner_ref["evidence_ids"] = owner_ref["evidence_ids"][:1]
    content = _issue_content(
        census,
        include_fields=False,
        inquiry_records=[{"source_refs": [owner_ref]}],
    )

    prepare_personal_detail_source_collections(content)

    issues = [
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "source_inquiry_physical_record_omitted"
    ]
    assert len(issues) == 1
    assert issues[0]["target_record_id"] == position["source_physical_row_id"]


def test_replayed_raw_position_evidence_earns_no_issue_credit() -> None:
    context = _sealed_context(
        [["1", "2024.01.01", "机构一", "贷款审批"]],
        headers=["编号", "查询日期", "查询机构", "查询原因"],
    )
    census = native_extraction._sealed_raw_inquiry_population_census(context)
    assert census is not None
    replayed = deepcopy(census)
    replayed["raw_physical_positions"].append(
        deepcopy(replayed["raw_physical_positions"][0])
    )
    content = _issue_content(replayed, include_fields=False)

    prepare_personal_detail_source_collections(content)

    assert not any(
        issue.get("issue_code") == "source_inquiry_physical_record_omitted"
        for issue in content["datasets"]["personal_detail_extraction_issues"]
    )
