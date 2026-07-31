# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build ``StyleContext`` from Mirror ``ParseResult`` for bank style detection.

Aggregates table cell matrices, full text, institution hints, LTRO reconstruction
meta, and page count into a single context object passed to ``BankStyleDetector``
and style parsers.

Pipeline role: first derivation step inside the post-seal bank-statement
projector, before style detection and parser dispatch.

Key exports: ``StyleContext``, ``build_style_context``, ``collect_tables_from_parse_result``.

Dependencies: ``ltro``, ``institution_authority``, ``BaseTableParser._collect_tables``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docmirror.plugins.bank_statement.header_resolve import align_bank_ledger_row
from docmirror.plugins.bank_statement.institution_authority import resolve_institution_from_context
from docmirror.plugins.bank_statement.ltro import ReconstructionMeta, reconstruct_tables


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


def build_style_context(parse_result: Any, full_text: str = "") -> StyleContext:
    text = full_text or getattr(parse_result, "full_text", "") or ""
    institution, authority = resolve_institution_from_context(parse_result, text)

    pages = getattr(parse_result, "pages", []) or []
    canonical_tables = collect_tables_from_parse_result(parse_result)
    structure_spe = _structure_spe_from_parse_result(parse_result)
    tables, reconstruction = reconstruct_tables(
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
    )
