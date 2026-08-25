# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build route-specific ``StyleContext`` instances for bank style detection.

Aggregates table cell matrices, full text, institution hints, LTRO reconstruction
meta, and page count into a single context object passed to ``BankStyleDetector``
and style parsers.

Pipeline role: first derivation step inside the post-seal bank-statement
projector, before style detection and parser dispatch.

Key exports: ``StyleContext``, ``build_digital_style_context``,
``build_scanned_style_context``, ``collect_tables_from_parse_result``.

Dependencies: ``ltro``, ``institution_authority``, ``BaseTableParser._collect_tables``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docmirror.plugins.bank_statement.extraction_dispatch import (
    DIGITAL_POLICY,
    SCANNED_POLICY,
    BankExtractionPolicy,
    BankExtractionRoute,
)
from docmirror.plugins.bank_statement.header_resolve import align_bank_ledger_row
from docmirror.plugins.bank_statement.institution_authority import resolve_institution_from_context
from docmirror.plugins.bank_statement.ltro import (
    ReconstructionMeta,
    reconstruct_digital_tables,
    reconstruct_scanned_tables,
)


def _structure_spe_from_parse_result(parse_result: Any) -> dict | None:
    from docmirror.evidence.spe_consumer import read_structure_spe

    return read_structure_spe(parse_result)


@dataclass
class StyleContext:
    tables: list[list[list[str]]]
    full_text: str
    institution: str | None
    page_count: int
    parse_result: Any = None
    reconstruction: ReconstructionMeta | None = None
    institution_authority: str = ""
    prefer_context_tables: bool = False
    extraction_route: BankExtractionRoute = BankExtractionRoute.DIGITAL
    extraction_policy: BankExtractionPolicy = DIGITAL_POLICY

    def __post_init__(self) -> None:
        if self.extraction_policy.route is not self.extraction_route:
            raise ValueError(
                "bank StyleContext route/policy mismatch: "
                f"{self.extraction_route.value} != {self.extraction_policy.route.value}"
            )


def collect_tables_from_parse_result(parse_result: Any) -> list[list[list[str]]]:
    from docmirror.plugins._base.base_table_parser import BaseTableParser

    class _Collector(BaseTableParser):
        @property
        def domain_name(self):
            return "bank_statement"

        @property
        def display_name(self):
            return "collector"

        @property
        def column_registry(self):
            return {}

        @property
        def standard_fields(self):
            return []

    tables = _Collector()._collect_tables(parse_result)
    if tables:
        return tables
    return []


def collect_physical_tables_from_parse_result(parse_result: Any) -> list[list[list[str]]]:
    """Collect page-local source tables without logical-table precedence."""
    tables: list[list[list[str]]] = []
    for page in getattr(parse_result, "pages", []) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", []) or []:
            matrix: list[list[str]] = []
            headers = [str(header or "") for header in getattr(table, "headers", []) or []]
            table_id = str(getattr(table, "table_id", "") or "")
            if headers:
                matrix.append([*headers, "_source_page", "_source_table_id", "_source_row_index"])
            for row_index, row in enumerate(getattr(table, "rows", []) or []):
                values = [
                    str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "")
                    for cell in getattr(row, "cells", []) or []
                ]
                if any(value.strip() for value in values):
                    values = align_bank_ledger_row(headers, values)
                    source_page = int(getattr(row, "source_page", 0) or page_number)
                    source_table_id = str(getattr(row, "source_physical_id", "") or table_id)
                    source_row_index = int(getattr(row, "source_row_index", -1))
                    if source_row_index < 0:
                        source_row_index = row_index
                    matrix.append(
                        [
                            *values,
                            str(source_page),
                            source_table_id,
                            str(source_row_index),
                        ]
                    )
            if matrix:
                tables.append(matrix)
    return tables


def _physical_row_geometry_index(
    table: Any,
    row: Any,
    row_index: int,
    values: list[str],
) -> int | None:
    """Resolve one body row to ``metadata.geometry`` without guessing.

    Physical-table geometry commonly includes the promoted header at raw row
    zero while ``table.rows`` starts with body row zero.  Prefer an explicit
    ``raw_row`` cell reference, then accept the structural offset only when
    ``metadata.raw_rows`` proves the exact source values.  A failed proof must
    not borrow header geometry for a body row.
    """

    metadata = getattr(table, "metadata", None) or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
    if not isinstance(raw_rows, list):
        return None

    refs = [
        *(getattr(row, "source_cell_refs", []) or []),
        *(
            ref
            for cell in (getattr(row, "cells", []) or [])
            for ref in (getattr(cell, "source_cell_refs", []) or [])
        ),
    ]
    candidates: list[int] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        try:
            raw_index = int(ref.get("raw_row", -1))
        except (TypeError, ValueError):
            continue
        if raw_index >= 0 and raw_index not in candidates:
            candidates.append(raw_index)

    headers = list(getattr(table, "headers", []) or [])
    structural_index = row_index + (1 if headers else 0)
    if structural_index not in candidates:
        candidates.append(structural_index)

    normalized_values = [str(value or "").strip() for value in values]
    for candidate in candidates:
        if candidate < 0 or candidate >= len(raw_rows) or not isinstance(raw_rows[candidate], list):
            continue
        raw_values = [str(value or "").strip() for value in raw_rows[candidate]]
        if raw_values == normalized_values:
            return candidate
    return None


def _physical_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return None
    return bbox


def _physical_row_bbox(cells: list[Any], geometry_boxes: Any) -> list[float] | None:
    boxes = [bbox for cell in cells if (bbox := _physical_bbox(getattr(cell, "bbox", None))) is not None]
    if isinstance(geometry_boxes, list):
        boxes.extend(bbox for value in geometry_boxes if (bbox := _physical_bbox(value)) is not None)
    if not boxes:
        return None
    return [
        min(bbox[0] for bbox in boxes),
        min(bbox[1] for bbox in boxes),
        max(bbox[2] for bbox in boxes),
        max(bbox[3] for bbox in boxes),
    ]


def _physical_row_evidence_ids(cells: list[Any], geometry_evidence: Any) -> list[str]:
    values = [
        str(evidence_id)
        for cell in cells
        for evidence_id in (getattr(cell, "evidence_ids", []) or [])
        if str(evidence_id)
    ]
    if isinstance(geometry_evidence, list):
        values.extend(
            str(evidence_id)
            for cell_evidence in geometry_evidence
            for evidence_id in (cell_evidence if isinstance(cell_evidence, list) else [])
            if str(evidence_id)
        )
    return list(dict.fromkeys(values))


def _physical_row_source_cell_refs(
    row: Any,
    cells: list[Any],
    *,
    page_number: int,
    table_id: str,
    source_row_index: int,
    raw_row_index: int | None,
    values: list[str],
    geometry_boxes: Any,
    geometry_evidence: Any,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for ref in [
        *(getattr(row, "source_cell_refs", []) or []),
        *(ref for cell in cells for ref in (getattr(cell, "source_cell_refs", []) or [])),
    ]:
        if isinstance(ref, dict) and ref not in refs:
            refs.append(dict(ref))
    if refs or raw_row_index is None:
        return refs

    boxes = geometry_boxes if isinstance(geometry_boxes, list) else []
    evidence = geometry_evidence if isinstance(geometry_evidence, list) else []
    width = max(len(values), len(boxes), len(evidence))
    for col_index in range(width):
        has_bbox = col_index < len(boxes) and _physical_bbox(boxes[col_index]) is not None
        has_evidence = (
            col_index < len(evidence)
            and isinstance(evidence[col_index], list)
            and any(str(item) for item in evidence[col_index])
        )
        if not (has_bbox or has_evidence):
            continue
        refs.append(
            {
                "source": "canonical_physical_table",
                "page": page_number,
                "table_id": table_id,
                "row": source_row_index,
                "raw_row": raw_row_index,
                "col": col_index,
            }
        )
    return refs


def collect_physical_table_row_sources_from_parse_result(parse_result: Any) -> list[dict[str, Any]]:
    """Collect row-local provenance for the physical-table candidate plane.

    The matrix collector intentionally carries only scalar parser columns.  This
    companion preserves the exact row geometry/evidence and is attached after
    parsing by row identity/signature.  Header or whole-table geometry is never
    used as a body-row fallback.
    """

    sources: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", []) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", []) or []:
            headers = [str(header or "") for header in (getattr(table, "headers", []) or [])]
            table_id = str(getattr(table, "table_id", "") or "")
            metadata = getattr(table, "metadata", None) or {}
            geometry = metadata.get("geometry") if isinstance(metadata, dict) else None
            geometry = geometry if isinstance(geometry, dict) else {}
            cell_bboxes = geometry.get("cell_bboxes")
            cell_evidence_ids = geometry.get("cell_evidence_ids")

            for row_index, row in enumerate(getattr(table, "rows", []) or []):
                cells = list(getattr(row, "cells", []) or [])
                values = [str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "") for cell in cells]
                if not any(value.strip() for value in values):
                    continue
                aligned_values = align_bank_ledger_row(headers, values)
                source_page = int(getattr(row, "source_page", 0) or page_number)
                source_table_id = str(getattr(row, "source_physical_id", "") or table_id)
                try:
                    source_row_index = int(getattr(row, "source_row_index", -1))
                except (TypeError, ValueError):
                    source_row_index = -1
                if source_row_index < 0:
                    source_row_index = row_index

                raw_row_index = _physical_row_geometry_index(table, row, row_index, values)
                geometry_boxes = (
                    cell_bboxes[raw_row_index]
                    if raw_row_index is not None
                    and isinstance(cell_bboxes, list)
                    and raw_row_index < len(cell_bboxes)
                    else []
                )
                geometry_evidence = (
                    cell_evidence_ids[raw_row_index]
                    if raw_row_index is not None
                    and isinstance(cell_evidence_ids, list)
                    and raw_row_index < len(cell_evidence_ids)
                    else []
                )
                bbox = _physical_row_bbox(cells, geometry_boxes)
                evidence_ids = _physical_row_evidence_ids(cells, geometry_evidence)
                source_cell_refs = _physical_row_source_cell_refs(
                    row,
                    cells,
                    page_number=source_page,
                    table_id=source_table_id,
                    source_row_index=source_row_index,
                    raw_row_index=raw_row_index,
                    values=values,
                    geometry_boxes=geometry_boxes,
                    geometry_evidence=geometry_evidence,
                )
                source: dict[str, Any] = {
                    "source": "canonical_physical_table",
                    "source_page": source_page,
                    "page_id": f"page:{source_page:04d}",
                    "page_range": [source_page, source_page],
                    **({"table_id": source_table_id} if source_table_id else {}),
                    "source_row_index": source_row_index,
                    **({"bbox": bbox} if bbox is not None else {}),
                    **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                    **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
                    "row_values": list(aligned_values),
                }
                if (
                    headers
                    and len(headers) == len(aligned_values)
                    and all(header.strip() for header in headers)
                    and len(set(headers)) == len(headers)
                ):
                    source["source_raw"] = dict(zip(headers, aligned_values, strict=True))
                sources.append(source)
    return sources


def _collect_tables_from_vnext_mirror(mirror: Any) -> list[list[list[str]]]:
    if mirror is None:
        return []
    if hasattr(mirror, "model_dump"):
        payload = mirror.model_dump(by_alias=True, exclude_none=True)
    elif isinstance(mirror, dict):
        payload = mirror
    else:
        return []

    tables: list[list[list[str]]] = []
    for block in payload.get("blocks") or []:
        if block.get("type") != "table":
            continue
        grid = (block.get("content") or {}).get("grid") or {}
        cells = grid.get("cells") or []
        if not cells:
            continue
        max_row = max((int(cell.get("row", 0) or 0) for cell in cells), default=-1)
        max_col = max((int(cell.get("col", 0) or 0) for cell in cells), default=-1)
        if max_row < 0 or max_col < 0:
            continue
        rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            row_idx = int(cell.get("row", 0) or 0)
            col_idx = int(cell.get("col", 0) or 0)
            if row_idx > max_row or col_idx > max_col:
                continue
            rows[row_idx][col_idx] = str(cell.get("text") or "")
        if rows and any(any(value.strip() for value in row) for row in rows):
            tables.append(rows)
    return tables


def _build_style_context(
    parse_result: Any,
    full_text: str,
    *,
    policy: BankExtractionPolicy,
) -> StyleContext:
    text = full_text or getattr(parse_result, "full_text", "") or ""
    institution, authority = resolve_institution_from_context(parse_result, text)

    pages = getattr(parse_result, "pages", []) or []
    canonical_tables = collect_tables_from_parse_result(parse_result)
    structure_spe = _structure_spe_from_parse_result(parse_result)
    reconstruct = (
        reconstruct_digital_tables
        if policy.route is BankExtractionRoute.DIGITAL
        else reconstruct_scanned_tables
    )
    tables, reconstruction = reconstruct(
        canonical_tables,
        text,
        page_count=len(pages),
        structure_spe=structure_spe,
        parse_result=parse_result,
    )

    return StyleContext(
        tables=tables,
        full_text=text,
        institution=institution,
        page_count=len(pages),
        parse_result=parse_result,
        reconstruction=reconstruction,
        institution_authority=authority,
        extraction_route=policy.route,
        extraction_policy=policy,
    )


def build_digital_style_context(parse_result: Any, full_text: str = "") -> StyleContext:
    """Build a context using only native/digital reconstruction strategies."""

    return _build_style_context(parse_result, full_text, policy=DIGITAL_POLICY)


def build_scanned_style_context(parse_result: Any, full_text: str = "") -> StyleContext:
    """Build a context using only OCR/scanned reconstruction strategies."""

    return _build_style_context(parse_result, full_text, policy=SCANNED_POLICY)


def build_style_context(parse_result: Any, full_text: str = "") -> StyleContext:
    """Compatibility wrapper that still honors sealed acquisition metadata."""

    from docmirror.plugins.bank_statement.extraction_dispatch import resolve_bank_extraction_route

    route = resolve_bank_extraction_route(parse_result)
    builder = build_digital_style_context if route is BankExtractionRoute.DIGITAL else build_scanned_style_context
    return builder(parse_result, full_text)
