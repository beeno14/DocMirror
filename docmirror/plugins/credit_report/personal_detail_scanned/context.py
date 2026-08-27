# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-page extraction context for personal detailed credit reports.

The context is post-seal and source conserving.  It owns the one cross-page
decode performed for a detailed report and memoizes expensive variant-owned
extractors without exposing mutable cached values to their consumers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

from docmirror.plugins.credit_report.personal_detail_scanned.page_reocr import (
    OneShotPageReOCRRegistry,
)
from docmirror.plugins.credit_report.personal_detail_scanned.page_topology import (
    PersonalDetailLogicalPageImageResolver,
    PersonalDetailPageTopology,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    CreditReportUnit,
    TransitionAction,
    TransitionHypothesis,
    UnitKind,
    decode_credit_report_units,
    score_credit_report_transition,
)

_T = TypeVar("_T")

_SECTION_MARKERS = (
    "个人基本信息",
    "信息概要",
    "信贷交易信息明细",
    "非信贷交易信息明细",
    "公共信息明细",
    "查询记录",
    "报告说明",
)
_FAMILY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inquiry", ("查询日期", "查询机构", "查询原因")),
    ("public_record", ("欠税记录", "民事判决", "强制执行", "行政处罚", "住房公积金")),
    ("liability", ("相关还款责任", "责任金额", "责任余额")),
    ("credit_line", ("授信协议信息", "授信协议标识", "授信额度用途")),
    ("residence", ("居住地址", "居住状况", "住宅电话")),
    ("employment", ("工作单位", "单位地址", "职业", "职务")),
    ("repayment", ("还款记录", "还款状态", "逾期金额", "月份")),
    ("account", ("账户标识", "账户状态", "管理机构", "发卡机构", "开立日期")),
    ("summary", ("信息概要", "账户数", "业务类型")),
)
_STRONG_FAMILY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("liability", ("相关还款责任", "保证合同编号")),
    ("credit_line", ("授信协议",)),
    ("residence", ("居住地址",)),
    ("employment", ("工作单位",)),
    ("inquiry", ("查询记录",)),
    ("public_record", ("欠税记录", "民事判决", "强制执行", "行政处罚", "执行法院", "立案法院")),
    ("repayment", ("还款记录",)),
    ("account", ("贷账户", "贷记卡账户", "准贷记卡账户")),
    ("summary", ("信息概要",)),
)
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*\d+\s*页(?:[,，]\s*共\s*\d+\s*页)?|page\s*\d+)", re.I)
_PRINTED_PAGE_RE = re.compile(r"第\s*(?P<page>\d{1,3})\s*页\s*[,，]?\s*共\s*(?P<total>\d{1,3})\s*页")
_PRINTED_PAGE_ONLY_RE = re.compile(r"^\s*第\s*(?P<page>\d{1,3})\s*页\s*[,，。.]?\s*$")
_NUMBERED_RE = re.compile(r"^\s*\d{1,4}[.、)]")
_ACCOUNT_ANCHOR_RE = re.compile(r"(?:账户|业务)\s*[（(]?\s*(\d{1,3})\s*[）)]?")
_BUSINESS_HEADING_RE = re.compile(
    r"^(?:"
    r"[（(][一二三四五六七八九十]+[）)].{0,24}"
    r"|账户\s*\d{1,3}(?:[（(].{0,80}[）)])?"
    r"|授信协议\s*\d{1,3}"
    r")$"
)
_CONTINUATIONS = frozenset({"same_table", "table_to_text_related", "text_to_table_related", "same_text_section"})
_SPLIT_FOR: dict[TransitionAction, TransitionAction] = {
    "same_table": "different_table",
    "table_to_text_related": "table_to_text_unrelated",
    "text_to_table_related": "text_to_table_unrelated",
    "same_text_section": "different_text_section",
}
_PAGE_OCR_ANCHORS = (
    "个人信用报告",
    "报告编号",
    "个人基本信息",
    "信息概要",
    "信贷交易信息明细",
    "查询记录",
    "账户",
    "管理机构",
    "查询日期",
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _finite(value: Any) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


_MONTHLY_STATUS_ATOMS = frozenset(
    {"*", "/", "N", "C", "1", "2", "3", "4", "5", "6", "7", "B", "M", "D", "Z", "G", "A", "#"}
)
_MONTHLY_REPAIR_ATOMS = _MONTHLY_STATUS_ATOMS | {"0"}
_MONTHLY_STATUS_TRANSLATION = str.maketrans({"☆": "*", "★": "*", "＊": "*"})


def _exact_source_table_repair_tokens_by_page(
    owner: Any,
    pages: Any,
    selected_pages: set[int],
) -> dict[int, list[dict[str, Any]]]:
    """Expose independently owned status/zero atoms from native table cells.

    Flat table values never participate. Every candidate is re-resolved from
    the immutable token plane and must remain uniquely owned by one exact native
    cell. Only one audited ``parse_result_table_cell`` correction atom may
    change that immutable value; a canonical typed cell is corroboration, never
    correction authority. The returned atoms are consumed only by the later
    source-lattice field repair; they do not replace canonical page lines.
    A declared vertical merge or multiword cell may expose one raw repair atom
    only after resolving its complete token set. Its own token box, never the
    merged cell box or typed text, determines the later field ownership.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
        _exact_native_table_cell_tokens,
        _native_table_cell_span,
    )

    def member(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    def unique_ids(value: Any) -> tuple[str, ...] | None:
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            return None
        return tuple(value)

    def exact_vertical_span(
        geometry: Mapping[str, Any],
        *,
        row: int,
        column: int,
        span: tuple[int, int],
    ) -> bool:
        """Authenticate the covered rows that the generic span reader skips."""

        row_span, column_span = span
        if row_span <= 1 or column_span != 1:
            return False
        spans = geometry.get("cell_spans")
        if not isinstance(spans, list):
            return False
        owners = 0
        for declaration in spans:
            if not isinstance(declaration, Mapping):
                return False
            values = tuple(declaration.get(key) for key in ("row", "col", "row_span", "col_span"))
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                return False
            other_row, other_col, other_rows, other_cols = values
            if other_row < 0 or other_col < 0 or other_rows < 1 or other_cols < 1:
                return False
            if (
                other_col <= column < other_col + other_cols
                and other_row < row + row_span
                and other_row + other_rows > row
            ):
                if values != (row, column, row_span, 1):
                    return False
                owners += 1
        if owners != 1:
            return False
        for covered_row in range(row + 1, row + row_span):
            for key, expected in (
                ("cell_bboxes", None),
                ("cell_geometry_status", "derived"),
                ("cell_evidence_ids", []),
                ("cell_token_ids", []),
            ):
                grid = geometry.get(key)
                if (
                    not isinstance(grid, list)
                    or covered_row >= len(grid)
                    or not isinstance(grid[covered_row], list)
                    or column >= len(grid[covered_row])
                    or grid[covered_row][column] != expected
                ):
                    return False
        return True

    def exact_bbox(value: Any) -> tuple[float, float, float, float] | None:
        try:
            bbox = tuple(float(coordinate) for coordinate in value)
        except (TypeError, ValueError):
            return None
        if (
            len(bbox) != 4
            or not all(math.isfinite(coordinate) for coordinate in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            return None
        return cast(tuple[float, float, float, float], bbox)

    def same_bbox(left: Any, right: Any) -> bool:
        left_bbox = exact_bbox(left)
        right_bbox = exact_bbox(right)
        return bool(
            left_bbox is not None
            and right_bbox is not None
            and all(
                abs(left_coordinate - right_coordinate) <= 1e-3
                for left_coordinate, right_coordinate in zip(
                    left_bbox,
                    right_bbox,
                    strict=True,
                )
            )
        )

    def projected_bbox(table: Any, value: Any) -> tuple[float, float, float, float] | None:
        bbox = exact_bbox(value)
        if bbox is None:
            return None
        metadata = member(table, "metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        affine = metadata.get("source_to_canonical_affine")
        if not isinstance(affine, Mapping):
            return bbox
        try:
            scale_x = float(affine["scale_x"])
            scale_y = float(affine["scale_y"])
            offset_x = float(affine["offset_x"])
            offset_y = float(affine["offset_y"])
        except (KeyError, TypeError, ValueError):
            return None
        projected = (
            offset_x + bbox[0] * scale_x,
            offset_y + bbox[1] * scale_y,
            offset_x + bbox[2] * scale_x,
            offset_y + bbox[3] * scale_y,
        )
        return exact_bbox(projected)

    parse_result = getattr(owner, "parse_result", owner)
    parser_info = getattr(parse_result, "parser_info", None)
    parser_options = getattr(parser_info, "options", None)
    correction_audit = (
        parser_options.get("ocr_corrections")
        if isinstance(parser_options, Mapping)
        else None
    )
    correction_events = (
        correction_audit.get("events")
        if isinstance(correction_audit, Mapping)
        else None
    )
    applied_correction_events_by_token: dict[str, list[Mapping[str, Any]]] = {}
    for event in correction_events or ():
        if not isinstance(event, Mapping) or str(event.get("action") or "") != "applied":
            continue
        source_ref = str(event.get("source_ref") or "")
        if source_ref:
            applied_correction_events_by_token.setdefault(source_ref, []).append(event)

    def canonical_cell_observation(
        table: Any,
        *,
        table_id: str,
        raw_row_index: int,
        column_index: int,
        source_logical_page: int,
        token_ids: tuple[str, ...],
        source_cell_bbox: tuple[float, ...],
        expected_span: tuple[int, int] = (1, 1),
        allow_multiword_text: bool = False,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return one exact typed-cell corroboration, never correction authority."""

        scalar_types = (str, bytes, int, float, bool)

        def row_cells(row: Any) -> list[Any]:
            cells = member(row, "cells")
            if not isinstance(cells, (list, tuple)) and isinstance(row, (list, tuple)):
                cells = row
            return list(cells) if isinstance(cells, (list, tuple)) else []

        candidate_cells: list[Any] = []
        candidate_ids: set[int] = set()

        def add_candidate(cell: Any) -> None:
            if cell is None or isinstance(cell, scalar_types) or id(cell) in candidate_ids:
                return
            candidate_ids.add(id(cell))
            candidate_cells.append(cell)

        # Projected tables retain an explicitly raw-row-aligned matrix. Header
        # slots are None and therefore cannot shift the first typed body row.
        aligned_rows = member(table, "source_cell_objects")
        if isinstance(aligned_rows, (list, tuple)) and raw_row_index < len(aligned_rows):
            cells = row_cells(aligned_rows[raw_row_index])
            if column_index < len(cells):
                add_candidate(cells[column_index])

        # Native/lightweight tables retain typed rows without the preserved
        # header. Infer the offset only from an explicit raw-row cardinality.
        typed_rows = member(table, "rows")
        metadata = member(table, "metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        raw_rows = metadata.get("raw_rows")
        if isinstance(typed_rows, (list, tuple)):
            row_offset = (
                1
                if isinstance(raw_rows, (list, tuple))
                and len(raw_rows) == len(typed_rows) + 1
                else 0
            )
            typed_slot = raw_row_index - row_offset
            if 0 <= typed_slot < len(typed_rows):
                cells = row_cells(typed_rows[typed_slot])
                if column_index < len(cells):
                    add_candidate(cells[column_index])

            # Explicit raw_row ownership is stronger than container position
            # and covers sparse typed-row containers.
            for row in typed_rows:
                for cell in row_cells(row):
                    refs = member(cell, "source_cell_refs")
                    if not isinstance(refs, (list, tuple)):
                        continue
                    if any(
                        isinstance(ref, Mapping)
                        and ref.get("raw_row") == raw_row_index
                        and ref.get("col") == column_index
                        for ref in refs
                    ):
                        add_candidate(cell)

        if not candidate_cells:
            return False, None
        if len(candidate_cells) != 1:
            return True, None
        cell = candidate_cells[0]
        typed_row_index = member(cell, "row_index")
        typed_column_index = member(cell, "col_index")
        typed_span = (member(cell, "row_span", 1), member(cell, "col_span", 1))
        if (
            not isinstance(typed_row_index, int)
            or isinstance(typed_row_index, bool)
            or typed_row_index < 0
            or not isinstance(typed_column_index, int)
            or isinstance(typed_column_index, bool)
            or typed_column_index != column_index
            or str(member(cell, "geometry_status") or "") != "exact"
            or any(not isinstance(value, int) or isinstance(value, bool) for value in typed_span)
            or typed_span != expected_span
        ):
            return True, None
        typed_cell_bbox = exact_bbox(member(cell, "bbox"))
        if typed_cell_bbox is None or not same_bbox(typed_cell_bbox, source_cell_bbox):
            return True, None
        evidence_ids = unique_ids(member(cell, "evidence_ids"))
        typed_token_ids = unique_ids(member(cell, "token_ids"))
        if (
            evidence_ids is None
            or typed_token_ids is None
            or set(evidence_ids) != set(token_ids)
            or set(typed_token_ids) != set(token_ids)
        ):
            return True, None
        source_cell_refs = member(cell, "source_cell_refs")
        if not isinstance(source_cell_refs, (list, tuple)) or len(source_cell_refs) != 1:
            return True, None
        [source_cell_ref] = source_cell_refs
        if (
            not isinstance(source_cell_ref, Mapping)
            or str(source_cell_ref.get("table_id") or "") != table_id
            or source_cell_ref.get("row") != typed_row_index
            or source_cell_ref.get("raw_row") != raw_row_index
            or source_cell_ref.get("col") != column_index
            or source_cell_ref.get("page") != source_logical_page
        ):
            return True, None
        repair_atom = str(member(cell, "text") or "").strip().translate(_MONTHLY_STATUS_TRANSLATION)
        if not allow_multiword_text and (len(repair_atom) != 1 or repair_atom not in _MONTHLY_REPAIR_ATOMS):
            return True, None
        raw_confidence = member(cell, "geometry_confidence")
        if raw_confidence is None:
            raw_confidence = member(cell, "confidence")
        return True, {
            "content": repair_atom,
            "bbox": typed_cell_bbox,
            "confidence": min(1.0, max(0.0, _finite(raw_confidence))),
            "evidence_ids": list(token_ids),
            "typed_row_index": typed_row_index,
        }

    page_values = list(pages) if isinstance(pages, (list, tuple)) else []
    candidates_by_token: dict[str, list[tuple[dict[str, Any], tuple[str, ...]]]] = {}
    declared_token_owners: Counter[str] = Counter()

    confidence_observations_by_token_id: dict[str, list[tuple[str, float]]] = {}
    corrected_observations_by_cell: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}
    corrected_claims_by_cell: set[tuple[str, str]] = set()
    raw_only_correction_claims: set[tuple[str, str]] = set()
    plane = getattr(owner, "evidence_plane", None)
    if plane is None:
        plane = getattr(getattr(owner, "parse_result", None), "evidence_plane", None)
    evidence = getattr(plane, "evidence", None)
    text_atoms = getattr(evidence, "text_atoms", None)
    if isinstance(text_atoms, list):
        for atom in text_atoms:
            raw_confidence = atom.get("confidence") if isinstance(atom, Mapping) else getattr(atom, "confidence", None)
            raw_text = atom.get("text") if isinstance(atom, Mapping) else getattr(atom, "text", None)
            repair_atom = str(raw_text or "").strip().translate(_MONTHLY_STATUS_TRANSLATION)
            is_repair_atom = len(repair_atom) == 1 and repair_atom in _MONTHLY_REPAIR_ATOMS
            raw_source_refs = (
                atom.get("source_refs") if isinstance(atom, Mapping) else getattr(atom, "source_refs", None)
            )
            source_refs = (
                tuple(dict.fromkeys(str(value) for value in raw_source_refs if str(value or "")))
                if isinstance(raw_source_refs, (list, tuple))
                else ()
            )
            if not source_refs:
                # Older/lightweight planes used the raw token ID as the atom ID.
                # This is confidence metadata only; exact cell ownership is still
                # independently re-resolved below from the immutable token plane.
                legacy_id = str(
                    (atom.get("id") if isinstance(atom, Mapping) else getattr(atom, "id", "")) or ""
                )
                source_refs = (legacy_id,) if legacy_id else ()
            confidence = min(1.0, max(0.0, _finite(raw_confidence)))
            if is_repair_atom:
                for source_ref in source_refs:
                    confidence_observations_by_token_id.setdefault(source_ref, []).append(
                        (repair_atom, confidence)
                    )

            source_kind = str(
                (atom.get("source_kind") if isinstance(atom, Mapping) else getattr(atom, "source_kind", ""))
                or ""
            )
            atom_metadata = (
                atom.get("metadata") if isinstance(atom, Mapping) else getattr(atom, "metadata", None)
            )
            if source_kind != "parse_result_table_cell" or not isinstance(atom_metadata, Mapping):
                continue
            if not any(
                key in atom_metadata
                for key in (
                    "ocr_original_text",
                    "ocr_corrected_text",
                    "ocr_correction_id",
                    "ocr_correction_action",
                    "ocr_correction_rule_id",
                )
            ):
                # Every typed cell produces a table atom, even when no
                # correction was attempted. That ordinary observation cannot
                # turn an exact raw token into a failed correction claim.
                # Present but empty/incomplete correction metadata still
                # reaches the fail-closed claim and event checks below.
                continue
            table_id = str(atom_metadata.get("table_id") or "")
            if table_id:
                raw_only_correction_claims.update((table_id, source_ref) for source_ref in source_refs)
            row_index = atom_metadata.get("row_index")
            column_index = atom_metadata.get("col_index")
            metadata_token_ids = atom_metadata.get("token_ids")
            normalized_metadata_token_ids = (
                tuple(dict.fromkeys(str(value) for value in metadata_token_ids if str(value or "")))
                if isinstance(metadata_token_ids, (list, tuple))
                else ()
            )
            if (
                not table_id
                or not isinstance(row_index, int)
                or isinstance(row_index, bool)
                or row_index < 0
                or not isinstance(column_index, int)
                or isinstance(column_index, bool)
                or column_index < 0
                or str(atom_metadata.get("geometry_status") or "") != "exact"
                or len(source_refs) != 1
                or normalized_metadata_token_ids != source_refs
            ):
                continue
            cell_key = (table_id, source_refs[0])
            corrected_claims_by_cell.add(cell_key)
            if not is_repair_atom:
                continue
            raw_bbox = atom.get("bbox") if isinstance(atom, Mapping) else getattr(atom, "bbox", None)
            atom_bbox = exact_bbox(raw_bbox)
            if atom_bbox is None:
                continue
            atom_id = str((atom.get("id") if isinstance(atom, Mapping) else getattr(atom, "id", "")) or "")
            if not atom_id:
                continue
            corrected_observations_by_cell.setdefault(
                cell_key,
                [],
            ).append(
                {
                    "atom_id": atom_id,
                    "content": repair_atom,
                    "bbox": atom_bbox,
                    "confidence": confidence,
                    "row_index": row_index,
                    "column_index": column_index,
                    "page_id": str(
                        (atom.get("page_id") if isinstance(atom, Mapping) else getattr(atom, "page_id", ""))
                        or ""
                    ),
                    "metadata": dict(atom_metadata),
                }
            )

    for fallback_page, page_value in enumerate(page_values, start=1):
        page_number = (
            page_value.get("page_number")
            if isinstance(page_value, Mapping)
            else getattr(page_value, "page_number", None)
        )
        logical_page = (
            page_number
            if isinstance(page_number, int) and not isinstance(page_number, bool) and page_number > 0
            else fallback_page
        )
        if logical_page not in selected_pages:
            continue
        tables = page_value.get("tables") if isinstance(page_value, Mapping) else getattr(page_value, "tables", None)
        if not isinstance(tables, (list, tuple)):
            continue
        for table in tables:
            table_id = str(
                (table.get("table_id") if isinstance(table, Mapping) else getattr(table, "table_id", ""))
                or ""
            )
            metadata = table.get("metadata") if isinstance(table, Mapping) else getattr(table, "metadata", None)
            metadata = metadata if isinstance(metadata, Mapping) else {}
            nested_source_geometry = metadata.get("geometry")
            source_geometry = (
                nested_source_geometry
                if isinstance(nested_source_geometry, Mapping)
                else metadata
            )
            token_grid = source_geometry.get("cell_token_ids")
            if not isinstance(token_grid, list):
                token_grid = metadata.get("cell_token_ids")
            if isinstance(token_grid, list):
                # An invalid competing cell is still an ownership claim. Do
                # not make a reused token unique by filtering that owner out.
                for claimed_row in token_grid:
                    if not isinstance(claimed_row, list):
                        continue
                    for claimed_ids in claimed_row:
                        if isinstance(claimed_ids, (list, tuple)):
                            declared_token_owners.update(
                                token_id for token_id in claimed_ids if isinstance(token_id, str) and token_id
                            )
            coordinate_geometry = metadata.get("canonical_geometry")
            has_canonical_geometry = isinstance(coordinate_geometry, Mapping)
            coordinate_geometry = (
                coordinate_geometry
                if has_canonical_geometry
                else source_geometry
            )
            if (
                str(source_geometry.get("coordinate_system") or "pdf_points_top_left")
                != "pdf_points_top_left"
                or str(coordinate_geometry.get("coordinate_system") or "pdf_points_top_left")
                != "pdf_points_top_left"
            ):
                continue
            source_cell_bboxes = source_geometry.get("cell_bboxes")
            if not isinstance(source_cell_bboxes, list):
                source_cell_bboxes = metadata.get("source_cell_bboxes")
            if (
                not isinstance(source_cell_bboxes, list)
                and not has_canonical_geometry
            ):
                # An unprojected lightweight table has only one (raw) plane.
                source_cell_bboxes = metadata.get("cell_bboxes")
            cell_bboxes = coordinate_geometry.get("cell_bboxes")
            if not isinstance(cell_bboxes, list):
                cell_bboxes = metadata.get("cell_bboxes")
            source_logical_page = metadata.get("source_logical_page", logical_page)
            if (
                not table_id
                or not isinstance(source_logical_page, int)
                or isinstance(source_logical_page, bool)
                or source_logical_page <= 0
                or not isinstance(token_grid, list)
                or not isinstance(source_cell_bboxes, list)
                or not isinstance(cell_bboxes, list)
            ):
                continue
            for row_index, token_row in enumerate(token_grid):
                if (
                    not isinstance(token_row, list)
                    or row_index >= len(cell_bboxes)
                    or not isinstance(cell_bboxes[row_index], list)
                    or row_index >= len(source_cell_bboxes)
                    or not isinstance(source_cell_bboxes[row_index], list)
                ):
                    continue
                for column_index, token_ids in enumerate(token_row):
                    if (
                        not isinstance(token_ids, list)
                        or column_index >= len(cell_bboxes[row_index])
                        or column_index >= len(source_cell_bboxes[row_index])
                    ):
                        continue
                    normalized_cell_token_ids = unique_ids(token_ids)
                    if normalized_cell_token_ids is None:
                        continue
                    evidence_grid = source_geometry.get("cell_evidence_ids")
                    if (
                        not isinstance(evidence_grid, list)
                        or row_index >= len(evidence_grid)
                        or not isinstance(evidence_grid[row_index], list)
                        or column_index >= len(evidence_grid[row_index])
                    ):
                        continue
                    cell_evidence_ids = unique_ids(evidence_grid[row_index][column_index])
                    if cell_evidence_ids is None or set(cell_evidence_ids) != set(normalized_cell_token_ids):
                        continue
                    allowed_span = _native_table_cell_span(table, row=row_index, column=column_index)
                    if allowed_span is not None and not exact_vertical_span(
                        source_geometry,
                        row=row_index,
                        column=column_index,
                        span=allowed_span,
                    ):
                        continue
                    resolved = _exact_native_table_cell_tokens(
                        owner,
                        table,
                        row=row_index,
                        column=column_index,
                        allowed_span=allowed_span,
                        logical_page=source_logical_page,
                        require_raw_tokens=True,
                    )
                    if (
                        resolved is None
                        or len(resolved) != len(normalized_cell_token_ids)
                        or {item[2] for item in resolved} != set(normalized_cell_token_ids)
                    ):
                        continue
                    raw_only_cell = allowed_span is not None or len(normalized_cell_token_ids) > 1
                    if raw_only_cell:
                        # A whole-cell correction cannot authorize a subfield
                        # inside a merge. Resolve all words, but expose exactly
                        # one independently positioned raw repair glyph.
                        if any(
                            (table_id, source_id) in raw_only_correction_claims
                            or applied_correction_events_by_token.get(source_id)
                            for source_id in normalized_cell_token_ids
                        ):
                            continue
                        eligible = [
                            item
                            for item in resolved
                            if str(item[0] or "").strip().translate(_MONTHLY_STATUS_TRANSLATION)
                            in _MONTHLY_REPAIR_ATOMS
                        ]
                        if len(eligible) != 1:
                            continue
                        raw_text, bbox, token_id = eligible[0]
                    else:
                        raw_text, bbox, token_id = resolved[0]
                    source_cell_bbox = exact_bbox(source_cell_bboxes[row_index][column_index])
                    cell_bbox = exact_bbox(cell_bboxes[row_index][column_index])
                    if source_cell_bbox is None or cell_bbox is None:
                        continue
                    if raw_only_cell and not same_bbox(projected_bbox(table, source_cell_bbox), cell_bbox):
                        continue
                    typed_cell_present, typed_cell_observation = canonical_cell_observation(
                        table,
                        table_id=table_id,
                        raw_row_index=row_index,
                        column_index=column_index,
                        source_logical_page=source_logical_page,
                        token_ids=normalized_cell_token_ids,
                        source_cell_bbox=source_cell_bbox,
                        expected_span=allowed_span or (1, 1),
                        allow_multiword_text=len(normalized_cell_token_ids) > 1,
                    )
                    if typed_cell_present and typed_cell_observation is None:
                        continue
                    cell_key = (table_id, token_id)
                    raw_token_text = str(raw_text or "").strip()
                    applied_events = applied_correction_events_by_token.get(token_id, [])
                    corrected: list[dict[str, Any]] = []
                    for observation in corrected_observations_by_cell.get(cell_key, []):
                        observation_metadata = observation.get("metadata")
                        observation_metadata = (
                            observation_metadata
                            if isinstance(observation_metadata, Mapping)
                            else {}
                        )
                        source_refs = observation_metadata.get("source_cell_refs")
                        if not isinstance(source_refs, (list, tuple)) or len(source_refs) != 1:
                            continue
                        source_ref = source_refs[0]
                        if not isinstance(source_ref, Mapping):
                            continue
                        typed_row_index = source_ref.get("row")
                        if (
                            observation.get("column_index") != column_index
                            or source_ref.get("raw_row") != row_index
                            or source_ref.get("col") != column_index
                            or source_ref.get("table_id") != table_id
                            or source_ref.get("page") != source_logical_page
                            or observation.get("row_index") != typed_row_index
                            or observation.get("page_id") != f"page:{source_logical_page:04d}"
                            or not same_bbox(observation.get("bbox"), source_cell_bbox)
                            or len(applied_events) != 1
                        ):
                            continue
                        event = applied_events[0]
                        event_id = str(event.get("event_id") or "")
                        rule_id = str(event.get("rule_id") or "")
                        repair_atom = str(observation.get("content") or "")
                        if (
                            not event_id
                            or not rule_id
                            or str(event.get("source_ref") or "") != token_id
                            or str(event.get("action") or "") != "applied"
                            or str(event.get("original") or "") != raw_token_text
                            or str(event.get("corrected") or "") != repair_atom
                            or str(observation_metadata.get("ocr_correction_action") or "")
                            != "applied"
                            or str(observation_metadata.get("ocr_original_text") or "")
                            != raw_token_text
                            or str(observation_metadata.get("ocr_corrected_text") or "")
                            != repair_atom
                            or str(observation_metadata.get("ocr_correction_id") or "")
                            != event_id
                            or str(observation_metadata.get("ocr_correction_rule_id") or "")
                            != rule_id
                            or (
                                typed_cell_observation is not None
                                and str(typed_cell_observation.get("content") or "")
                                != repair_atom
                            )
                        ):
                            continue
                        projected_observation_bbox = projected_bbox(
                            table,
                            observation.get("bbox"),
                        )
                        if projected_observation_bbox is None or not same_bbox(
                            projected_observation_bbox,
                            cell_bbox,
                        ):
                            continue
                        corrected.append(
                            {
                                **observation,
                                "correction_identity": (event_id, rule_id),
                            }
                        )
                    if len(corrected) > 1:
                        continue
                    if corrected:
                        observation = corrected[0]
                        repair_atom = str(observation["content"])
                        output_bbox = list(cell_bbox)
                        confidence = min(
                            float(observation["confidence"]),
                            float(typed_cell_observation.get("confidence", 1.0))
                            if typed_cell_observation is not None
                            else 1.0,
                        )
                        source = (
                            "exact_corrected_source_table_status_cell"
                            if repair_atom in _MONTHLY_STATUS_ATOMS
                            else "exact_corrected_source_table_amount_cell"
                        )
                        evidence_ids = [
                            token_id,
                            *dict.fromkeys(
                                str(value.get("atom_id") or "")
                                for value in corrected
                                if str(value.get("atom_id") or "")
                            ),
                        ]
                    else:
                        if cell_key in corrected_claims_by_cell or applied_events:
                            # A table-cell correction was asserted for this
                            # owner but failed its audit/geometry contract. Do
                            # not silently fall back through that ambiguity.
                            continue
                        repair_atom = str(raw_text or "").strip().translate(_MONTHLY_STATUS_TRANSLATION)
                        if len(repair_atom) != 1 or repair_atom not in _MONTHLY_REPAIR_ATOMS:
                            continue
                        matching_confidences = [
                            confidence
                            for observed_atom, confidence in confidence_observations_by_token_id.get(token_id, [])
                            if observed_atom == repair_atom
                        ]
                        confidence = min(matching_confidences, default=0.0)
                        projected_token_bbox = projected_bbox(table, bbox)
                        if projected_token_bbox is None:
                            continue
                        output_bbox = list(projected_token_bbox)
                        source = (
                            "exact_native_source_table_status_cell"
                            if repair_atom in _MONTHLY_STATUS_ATOMS
                            else "exact_native_source_table_amount_cell"
                        )
                        evidence_ids = [token_id]
                    candidates_by_token.setdefault(token_id, []).append(
                        (
                            {
                                "token_id": token_id,
                                "content": repair_atom,
                                "bbox": output_bbox,
                                "confidence": confidence,
                                "page": logical_page,
                                "source_logical_page": logical_page,
                                "source_origin_logical_page": source_logical_page,
                                "coordinate_system": "pdf_points_top_left",
                                "source": source,
                                "evidence_ids": evidence_ids,
                            },
                            normalized_cell_token_ids,
                        )
                    )

    output: dict[int, list[dict[str, Any]]] = {}
    for candidates in candidates_by_token.values():
        if len(candidates) != 1:
            continue
        candidate, cell_token_ids = candidates[0]
        if any(declared_token_owners[token_id] != 1 for token_id in cell_token_ids):
            continue
        output.setdefault(int(candidate["page"]), []).append(candidate)
    for page_tokens in output.values():
        page_tokens.sort(
            key=lambda token: (
                float(token["bbox"][1]),
                float(token["bbox"][0]),
                str(token["token_id"]),
            )
        )
    return output


def _printed_monthly_anchor_shape(
    line: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...], str] | None:
    """Read one complete printed range line without accepting projected boxes."""

    from docmirror.plugins.credit_report.personal_detail_scanned.relations import _printed_repayment_range

    text = str(line.get("text") or line.get("content") or "").strip()
    if (
        not text
        or (line.get("text") and line.get("content") and str(line["text"]).strip() != str(line["content"]).strip())
        or text.count("还款记录") != 1
        or str(line.get("coordinate_system") or "pdf_points_top_left") != "pdf_points_top_left"
    ):
        return None
    printed_range = _printed_repayment_range(text)
    if printed_range is None:
        return None
    date_range = tuple(printed_range[key] for key in ("start_year", "start_month", "end_year", "end_month"))
    ids = line.get("evidence_ids")
    if (
        not isinstance(ids, (list, tuple))
        or not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        return None
    token_ids = line.get("token_ids")
    if token_ids is not None and (
        not isinstance(token_ids, (list, tuple))
        or any(not isinstance(value, str) or not value for value in token_ids)
        or len(set(token_ids)) != len(token_ids)
        or set(token_ids) != set(ids)
    ):
        return None
    raw_bbox = line.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        bbox = tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return tuple(sorted(ids)), bbox, date_range, text


def _authenticated_printed_monthly_anchor_inventory(
    owner: Any,
) -> tuple[dict[tuple[int, tuple[str, ...]], dict[str, Any]], bool]:
    """Authenticate and close the original raw repayment-range population.

    Source bbox fields on transformed lines have several different historical
    meanings. None are consulted here: only the immutable raw OCR words and
    their complete original line can establish the printed-grid identity.
    A valid duplicate view cannot hide a malformed original range line. The
    completeness bit describes this sealed source inventory, not whether any
    detector or canonical registration subsequently materialized its grids.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.exact_evidence import (
        resolve_exact_page_token_atoms,
    )

    def positive_page(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def range_line(line: Mapping[str, Any]) -> bool:
        # A damaged range is still part of the source census. Do not require
        # its dates, IDs or geometry to pass the authenticator before deciding
        # that a failed original line makes the population incomplete.
        return any(
            "还款记录" in _compact(value)
            and re.search(r"\d{4}\s*年", str(value or "")) is not None
            for value in (line.get("text"), line.get("content"))
        )

    parse_result = getattr(owner, "parse_result", owner)
    bundles = _domain_specific(parse_result).get("_page_evidence_bundles")
    if not isinstance(bundles, (list, tuple)) or not bundles:
        return {}, False
    complete = True
    page_counts = Counter(
        bundle["page"]
        for bundle in bundles
        if isinstance(bundle, Mapping) and positive_page(bundle.get("page"))
    )
    raw_pages: dict[int, list[Any]] = {}
    for page in getattr(parse_result, "pages", None) or ():
        page_number = getattr(page, "page_number", None)
        if positive_page(page_number):
            raw_pages.setdefault(page_number, []).append(page)
        else:
            complete = False
    if (
        not raw_pages
        or set(raw_pages) != set(page_counts)
        or any(len(pages) != 1 or page_counts.get(logical_page) != 1 for logical_page, pages in raw_pages.items())
    ):
        complete = False
    output: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            complete = False
            continue
        logical_page = bundle.get("page")
        if not positive_page(logical_page) or page_counts[logical_page] != 1:
            complete = False
            continue
        pages = raw_pages.get(logical_page) or []
        if len(pages) != 1:
            complete = False
            continue
        source_claims = [bundle.get("source_page_number"), getattr(pages[0], "source_page_number", None)]
        evidence_views = [
            view
            for key in ("local_structure_evidence", "micro_grid_evidence")
            if isinstance(view := bundle.get(key), Mapping)
        ]
        if not evidence_views or any(
            bundle.get(key) is not None and not isinstance(bundle.get(key), Mapping)
            for key in ("local_structure_evidence", "micro_grid_evidence")
        ):
            complete = False
        source_claims.extend(view.get("source_page") for view in evidence_views)
        source_claims = [value for value in source_claims if value is not None]
        if (
            not source_claims
            or any(not positive_page(value) for value in source_claims)
            or len(set(source_claims)) != 1
            or any(
                not positive_page(view.get("page", logical_page))
                or view.get("page", logical_page) != logical_page
                for view in evidence_views
            )
        ):
            complete = False
            continue
        source_page = source_claims[0]
        for view in evidence_views:
            lines = view.get("lines")
            if not isinstance(lines, (list, tuple)):
                complete = False
                continue
            for line in lines:
                if not isinstance(line, Mapping):
                    complete = False
                    continue
                if line.get("coordinate_status") == "cross_page_y_shift":
                    # The shared pre-plugin enricher can retain explicit
                    # next-page aliases in a raw bundle. They are not new
                    # original lines; the source page's own bundle is already
                    # required by the closed-world page coverage check above.
                    continue
                if not range_line(line):
                    continue
                shape = _printed_monthly_anchor_shape(line)
                if shape is None:
                    complete = False
                    continue
                if any(
                    not positive_page(line.get(key, logical_page))
                    or line.get(key, logical_page) != logical_page
                    for key in ("page", "source_logical_page", "coordinate_logical_page")
                ):
                    complete = False
                    continue
                ids, bbox, date_range, text = shape
                resolved = resolve_exact_page_token_atoms(
                    owner,
                    ids,
                    logical_page=logical_page,
                    require_raw_tokens=True,
                )
                if resolved is None or len(resolved) != len(ids) or {item[2] for item in resolved} != set(ids):
                    complete = False
                    continue
                words = sorted(resolved, key=lambda item: (item[1][0], item[1][1], item[2]))
                raw_bbox = (
                    min(word[1][0] for word in words),
                    min(word[1][1] for word in words),
                    max(word[1][2] for word in words),
                    max(word[1][3] for word in words),
                )
                if _compact(" ".join(word[0] for word in words)) != _compact(text) or any(
                    abs(left - right) > 1e-3 for left, right in zip(raw_bbox, bbox, strict=True)
                ):
                    complete = False
                    continue
                output[(logical_page, ids)] = {
                    "coordinate_system": "pdf_points_top_left",
                    "coordinate_plane": "raw_logical_page",
                    "source_logical_page": logical_page,
                    "source_page": source_page,
                    "evidence_ids": list(ids),
                    "bbox": list(raw_bbox),
                    "date_range": list(date_range),
                }
    return output, complete


def _authenticated_printed_monthly_anchors(
    owner: Any,
) -> dict[tuple[int, tuple[str, ...]], dict[str, Any]]:
    """Compatibility index over individually authenticated original anchors."""

    return _authenticated_printed_monthly_anchor_inventory(owner)[0]


def _lines_with_printed_monthly_anchors(
    lines: Iterable[Mapping[str, Any]],
    *,
    logical_page: Any,
    source_page: Any,
    anchors: Mapping[tuple[int, tuple[str, ...]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Stamp only authenticated raw lines before any coordinate projection."""

    output: list[dict[str, Any]] = []
    for value in lines:
        if not isinstance(value, Mapping):
            continue
        line = deepcopy(dict(value))
        line.pop("printed_anchor_identity", None)
        shape = _printed_monthly_anchor_shape(line)
        if (
            shape is not None
            and isinstance(logical_page, int)
            and not isinstance(logical_page, bool)
            and isinstance(source_page, int)
            and not isinstance(source_page, bool)
            and (anchor := anchors.get((logical_page, shape[0]))) is not None
            and anchor.get("source_page") == source_page
            and anchor.get("bbox") == list(shape[1])
            and anchor.get("date_range") == list(shape[2])
            and line.get("coordinate_status") != "cross_page_y_shift"
            and all(
                type(line.get(key, logical_page)) is int and line.get(key, logical_page) == logical_page
                for key in ("page", "source_logical_page", "coordinate_logical_page")
            )
        ):
            line["printed_anchor_identity"] = deepcopy(dict(anchor))
        output.append(line)
    return output


def _page_ocr_score(words: Iterable[dict[str, Any]], *, image_shape: Any = None) -> float:
    texts = [str(word.get("text") or "").strip() for word in words if str(word.get("text") or "").strip()]
    joined = " ".join(texts)
    anchors = sum(joined.count(marker) for marker in _PAGE_OCR_ANCHORS)
    long_cjk = sum(1 for text in texts if len(re.findall(r"[\u3400-\u9fff]", text)) >= 2)
    confidences = [float(word.get("confidence") or 0.0) for word in words if word.get("text")]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    horizontal_chars = 0
    vertical_chars = 0
    for word in words:
        text = str(word.get("text") or "").strip()
        box = word.get("bbox")
        if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
        weight = max(1, len(text))
        if width >= height * 1.15:
            horizontal_chars += weight
        elif height >= width * 1.5:
            vertical_chars += weight
    portrait_score = 0.0
    if isinstance(image_shape, (list, tuple)) and len(image_shape) >= 2:
        height = float(image_shape[0] or 0.0)
        width = float(image_shape[1] or 0.0)
        if height > 0 and width > 0:
            # Canonical detailed-report logical pages are portrait.  This
            # prior breaks dense-table ties where rotating an already upright
            # page produces more, but vertically fragmented, OCR tokens.
            portrait_score = 240.0 if height >= width else -240.0
    return float(
        len(joined)
        + anchors * 80
        + long_cjk * 4
        + mean_confidence * 30
        + horizontal_chars * 2.5
        - vertical_chars * 3.0
        + portrait_score
    )


def _single_page_ocr(image: Any) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Deskew and OCR one already-oriented, frozen logical subpage once."""

    from docmirror.layout.normalization import hough_deskew_image, inverse_project_hough_bbox
    from docmirror.ocr.repair.recognizers import rapidocr_recognize

    deskewed, deskew = hough_deskew_image(image)
    words = rapidocr_recognize(deskewed, source="personal_detail_page_reocr_once")
    shape = getattr(image, "shape", None)
    if deskew.get("applied") is True and isinstance(shape, (list, tuple)) and len(shape) >= 2:
        restored: list[dict[str, Any]] = []
        for word in words:
            mapped = inverse_project_hough_bbox(
                word.get("bbox"),
                deskew,
                width=float(shape[1]),
                height=float(shape[0]),
            )
            if mapped is None:
                continue
            restored.append({**word, "bbox": mapped})
        words = restored
    audit = {key: deepcopy(value) for key, value in deskew.items() if key not in {"forward_matrix", "inverse_matrix"}}
    return words, _page_ocr_score(words, image_shape=shape), audit


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox") if isinstance(value, dict) else getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    x0, y0, x1, y1 = (_finite(item) for item in raw[:4])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _bottom_furniture_geometry(value: Any, *, page_height: Any) -> bool:
    """Require one exact local bbox in the page's narrow bottom band."""

    box = _bbox(value)
    height = _finite(page_height)
    if box is None or height <= 0.0:
        return False
    tolerance = max(2.0, height * 0.01)
    return bool(
        box[1] >= height * 0.85
        and box[3] >= height * 0.90
        and box[3] <= height + tolerance
        and box[3] - box[1] <= height * 0.08
    )


def _authoritative_reading_order(resolution: Any) -> bool:
    return bool(
        isinstance(resolution, Mapping)
        and resolution.get("resolved") is True
        and resolution.get("authoritative") is True
    )


def _matrix3(value: Any) -> list[list[float]] | None:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)
    ):
        return None
    return [[float(item) for item in row] for row in value]


def _transform_bbox(
    matrix: list[list[float]],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    points = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        points.append(
            (
                matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
                matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_fraction(
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(bbox[2], crop[2]) - max(bbox[0], crop[0])) * max(
        0.0, min(bbox[3], crop[3]) - max(bbox[1], crop[1])
    )
    area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return intersection / area


def _slice_assignment(
    source_bbox: tuple[float, float, float, float],
    recovered_by_segment: Mapping[int, Mapping[str, Any]],
) -> tuple[int | None, bool]:
    ranked = sorted(
        (
            (
                _overlap_fraction(
                    source_bbox,
                    cast(
                        tuple[float, float, float, float],
                        tuple(float(value) for value in recovered["source_crop_bbox"]),
                    ),
                ),
                segment,
            )
            for segment, recovered in recovered_by_segment.items()
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < 0.55:
        return None, True
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    ambiguous = second >= 0.20 and ranked[0][0] - second < 0.35
    return (None, True) if ambiguous else (ranked[0][1], False)


def _project_table_for_static_slice(
    table: Any,
    *,
    base_to_source: list[list[float]],
    source_to_slice: list[list[float]],
) -> Any:
    def project(raw: Any) -> list[float] | Any:
        box = _bbox({"bbox": raw})
        if box is None:
            return raw
        source_box = _transform_bbox(base_to_source, box)
        return list(_transform_bbox(source_to_slice, source_box))

    metadata = deepcopy(dict(getattr(table, "metadata", None) or {}))
    cell_boxes = metadata.get("cell_bboxes")
    if isinstance(cell_boxes, list):
        metadata["cell_bboxes"] = [
            [project(box) for box in row] if isinstance(row, list) else row for row in cell_boxes
        ]
    raw_rows = _raw_rows(table)
    if raw_rows:
        metadata["raw_rows"] = [list(row) for row in raw_rows]
    table_box = _bbox(table)
    return SimpleNamespace(
        table_id=str(getattr(table, "table_id", "") or ""),
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=project(table_box) if table_box is not None else None,
        confidence=getattr(table, "confidence", None),
    )


def _raw_rows(table: Any) -> tuple[tuple[str, ...], ...]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw_rows = metadata.get("raw_rows")
    if isinstance(raw_rows, list) and raw_rows:
        return tuple(
            tuple(str(value or "").replace("\n", "").strip() for value in row)
            for row in raw_rows
            if isinstance(row, list)
        )
    rows: list[tuple[str, ...]] = []
    headers = tuple(str(value or "").replace("\n", "").strip() for value in getattr(table, "headers", None) or [])
    if headers:
        rows.append(headers)
    for row in getattr(table, "rows", None) or []:
        rows.append(
            tuple(
                str(getattr(cell, "text", cell) or "").replace("\n", "").strip()
                for cell in getattr(row, "cells", None) or []
            )
        )
    return tuple(rows)


def _owned_by_table(
    box: tuple[float, float, float, float] | None,
    table_boxes: Iterable[tuple[float, float, float, float]],
) -> bool:
    if box is None:
        return False
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    for table in table_boxes:
        intersection = max(0.0, min(box[2], table[2]) - max(box[0], table[0])) * max(
            0.0, min(box[3], table[3]) - max(box[1], table[1])
        )
        center_inside = table[0] <= center[0] <= table[2] and table[1] <= center[1] <= table[3]
        if center_inside or (area > 0 and intersection / area >= 0.65):
            return True
    return False


def _geometry_owner(
    box: tuple[float, float, float, float] | None,
    owners: Iterable[tuple[tuple[float, float, float, float], str]],
) -> str:
    if box is None:
        return ""
    for owner_box, unit_id in owners:
        if _owned_by_table(box, (owner_box,)):
            return unit_id
    return ""


def _kind(text: str) -> UnitKind:
    compact = _compact(text).strip(":：")
    if any(marker in compact and len(compact) <= len(marker) + 12 for marker in _SECTION_MARKERS) or (
        len(compact) <= 96 and _BUSINESS_HEADING_RE.fullmatch(compact)
    ):
        return "heading"
    if _NUMBERED_RE.match(str(text or "")):
        return "ledger"
    return "text"


def _family(units: Iterable[CreditReportUnit]) -> str:
    text = _compact("\n".join(unit.text for unit in units))
    for family, markers in _STRONG_FAMILY_MARKERS:
        if any(marker in text for marker in markers):
            return family
    ranked = sorted(
        ((sum(marker in text for marker in markers), family) for family, markers in _FAMILY_MARKERS),
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 2 else ""


def _new_account_boundary(left_text: str, right_text: str) -> bool:
    left = _ACCOUNT_ANCHOR_RE.search(left_text)
    right = _ACCOUNT_ANCHOR_RE.search(right_text)
    return bool(left and right and left.group(1) != right.group(1))


def _families_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left == right:
        return True
    # Repayment/status grids are components of an account card, not a separate
    # business entity, even when their local schema contains no account header.
    return {left, right} == {"account", "repayment"}


class PersonalDetailTransitionPolicy:
    """Re-rank shared transition hypotheses using personal-report contracts."""

    def __init__(self, *, minimum_confidence: float = 0.46, minimum_margin: float = 0.06) -> None:
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self._cache: dict[tuple[tuple[str, ...], str, str], tuple[TransitionHypothesis, ...]] = {}

    def score(
        self,
        open_units: tuple[CreditReportUnit, ...],
        candidate: CreditReportUnit,
        lookahead: CreditReportUnit | None,
    ) -> tuple[TransitionHypothesis, ...]:
        key = (tuple(unit.unit_id for unit in open_units), candidate.unit_id, lookahead.unit_id if lookahead else "")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = score_credit_report_transition(
            open_units,
            candidate,
            report_family="personal_detail",
            lookahead=lookahead,
        ).hypotheses
        tail = open_units[-1]
        crosses_page = candidate.page != tail.page

        weights = {hypothesis.action: max(hypothesis.score, 0.000001) for hypothesis in base}
        signals = {hypothesis.action: list(hypothesis.signals) for hypothesis in base}
        continuation = next((item for item in base if item.action in _CONTINUATIONS), None)
        if continuation is None:
            self._cache[key] = base
            return base

        split_action = _SPLIT_FOR[continuation.action]
        left_family = _family(open_units)
        right_family = _family((candidate,))
        hard_split = candidate.kind == "heading" or _new_account_boundary(
            _compact("\n".join(unit.text for unit in open_units)),
            _compact(candidate.text),
        )
        if not _families_compatible(left_family, right_family):
            hard_split = True
            signals[split_action].append("personal_detail_family_mismatch")
        elif left_family and right_family:
            weights[continuation.action] *= 1.35
            signals[continuation.action].append(f"personal_detail_family_continues:{left_family}:{right_family}")
            if crosses_page and {left_family, right_family} <= {"account", "repayment"}:
                # One account card is composed of several differently shaped
                # grids. A page break can therefore change column count and
                # header schema without opening a new business entity.
                weights[continuation.action] *= 2.25
                weights[split_action] *= 0.50
                signals[continuation.action].append("personal_detail_account_card_continues")

        if hard_split:
            weights[continuation.action] *= 0.01
            weights[split_action] *= 5.0
            signals[split_action].append("personal_detail_semantic_veto")

        normalized = self._normalize(weights, signals)
        selected = normalized[0]
        if crosses_page and selected.action in _CONTINUATIONS:
            runner_up = normalized[1].score if len(normalized) > 1 else 0.0
            if selected.score < self.minimum_confidence or selected.score - runner_up < self.minimum_margin:
                weights[split_action] = max(weights[split_action], weights[selected.action] * 1.05)
                signals[split_action].append("personal_detail_conservative_boundary")
                normalized = self._normalize(weights, signals)
        self._cache[key] = normalized
        return normalized

    @staticmethod
    def _normalize(
        weights: dict[TransitionAction, float],
        signals: dict[TransitionAction, list[str]],
    ) -> tuple[TransitionHypothesis, ...]:
        total = sum(max(value, 0.0) for value in weights.values()) or 1.0
        return tuple(
            sorted(
                (
                    TransitionHypothesis(
                        action=action,
                        score=round(max(value, 0.0) / total, 6),
                        signals=tuple(signals[action]),
                    )
                    for action, value in weights.items()
                ),
                key=lambda item: (-item.score, item.action),
            )
        )


def _domain_specific(parse_result: Any) -> dict[str, Any]:
    value = getattr(getattr(parse_result, "entities", None), "domain_specific", None)
    return value if isinstance(value, dict) else {}


def _printed_reading_order_resolution(
    parse_result: Any,
    topology: PersonalDetailPageTopology | None = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Resolve printed reading order without disguising fallback as proof.

    Detailed reports are commonly scanned as two-page spreads. Physical
    sheets can be out of order even though each half retains the report's own
    ``第 N 页，共 M 页`` footer. Provenance page numbers remain unchanged; only
    continuation and evidence traversal use this order.

    A page-only ``第 N 页`` marker is accepted only after other complete
    footers establish one coherent document total. Its other spread half may
    then be inferred from frozen topology. Every observed logical page must
    ultimately own one unique, in-range printed number. A trailing scan half
    with no sealed text or table content may follow that complete permutation;
    it is retained after the report pages but never assigned a manufactured
    printed number. Failure returns sealed identity order together with explicit
    unresolved provenance so downstream ownership code cannot mistake the
    fallback for an authoritative mapping.
    """
    text_evidence_by_page: dict[int, list[tuple[str, Any]]] = {}
    page_heights_by_logical: dict[int, list[float]] = {}
    observed_pages: set[int] = set()
    source_by_logical: dict[int, int] = {}
    native_pages_by_logical: dict[int, list[Any]] = {}
    evidence_bundles_by_logical: dict[int, list[Mapping[str, Any]]] = {}

    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        logical = int(getattr(page, "page_number", 0) or page_index)
        observed_pages.add(logical)
        native_pages_by_logical.setdefault(logical, []).append(page)
        transform = dict(getattr(page, "coordinate_transform", None) or {})
        source_by_logical[logical] = int(
            transform.get("source_page_number") or getattr(page, "source_page_number", 0) or logical
        )
        page_height = _finite(getattr(page, "height", 0))
        if page_height > 0.0:
            page_heights_by_logical.setdefault(logical, []).append(page_height)
        text_evidence_by_page.setdefault(logical, []).extend(
            (str(getattr(block, "content", "") or ""), block) for block in getattr(page, "texts", None) or []
        )

    for bundle in _domain_specific(parse_result).get("_page_evidence_bundles") or []:
        if not isinstance(bundle, dict):
            continue
        local = bundle.get("local_structure_evidence")
        logical = int(bundle.get("page") or (local.get("page") if isinstance(local, Mapping) else 0) or 0)
        if logical > 0:
            evidence_bundles_by_logical.setdefault(logical, []).append(bundle)
        if not isinstance(local, dict):
            continue
        if logical <= 0:
            continue
        observed_pages.add(logical)
        source_by_logical.setdefault(
            logical,
            int(bundle.get("source_page_number") or local.get("source_page") or logical),
        )
        page_height = _finite(
            local.get("page_height") or bundle.get("page_height") or local.get("height") or bundle.get("height")
        )
        if page_height > 0.0:
            page_heights_by_logical.setdefault(logical, []).append(page_height)
        text_evidence_by_page.setdefault(logical, []).extend(
            (str(line.get("text") or line.get("content") or ""), line)
            for line in local.get("lines") or []
            if isinstance(line, dict)
        )

    identity = {page: page for page in sorted(observed_pages)}

    def table_source_content_present(table: Any) -> bool:
        """Check every sealed table representation, not only its preferred rows."""

        def member(owner: Any, name: str, default: Any = None) -> Any:
            return owner.get(name, default) if isinstance(owner, Mapping) else getattr(owner, name, default)

        def scalar_present(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(_compact(value))
            return True

        metadata = member(table, "metadata")
        raw_row_representations = [member(table, "raw_rows")]
        if isinstance(metadata, Mapping):
            raw_row_representations.append(metadata.get("raw_rows"))
        for raw_rows in raw_row_representations:
            if raw_rows in (None, [], ()):
                continue
            if not isinstance(raw_rows, (list, tuple)):
                return True
            for raw_row in raw_rows:
                if not isinstance(raw_row, (list, tuple)):
                    return True
                if any(scalar_present(cell) for cell in raw_row):
                    return True

        if any(scalar_present(header) for header in member(table, "headers", ()) or ()):
            return True
        if scalar_present(member(table, "caption")):
            return True
        for row_collection_name in ("rows", "row_models", "logical_rows"):
            rows = member(table, row_collection_name)
            if rows in (None, [], ()):
                continue
            if not isinstance(rows, (list, tuple)):
                return True
            for row in rows:
                cells = member(row, "cells")
                if cells is None and isinstance(row, (list, tuple)):
                    cells = row
                if cells is None or not isinstance(cells, (list, tuple)):
                    return True
                for cell in cells:
                    if isinstance(cell, Mapping):
                        values = (
                            cell.get("text"),
                            cell.get("cleaned"),
                            cell.get("numeric"),
                            cell.get("value"),
                            cell.get("content"),
                        )
                    elif isinstance(cell, (str, int, float, bool)) or cell is None:
                        values = (cell,)
                    else:
                        values = (
                            getattr(cell, "text", None),
                            getattr(cell, "cleaned", None),
                            getattr(cell, "numeric", None),
                        )
                    if any(scalar_present(value) for value in values):
                        return True
        return False

    def source_evidence_empty(logical: int) -> bool:
        """Prove that one registered logical page has no sealed source content."""

        native_pages = native_pages_by_logical.get(logical) or []
        if len(native_pages) != 1:
            return False
        if any(_compact(text) for text, _evidence in text_evidence_by_page.get(logical, ())):
            return False
        page = native_pages[0]
        if any(
            _compact(pair.get("key") if isinstance(pair, Mapping) else getattr(pair, "key", ""))
            or _compact(pair.get("value") if isinstance(pair, Mapping) else getattr(pair, "value", ""))
            for pair in getattr(page, "key_values", None) or []
        ):
            return False
        if any(
            _compact(
                (line.get("content") or line.get("text"))
                if isinstance(line, Mapping)
                else getattr(line, "content", getattr(line, "text", ""))
            )
            for line in getattr(page, "lines", None) or []
        ):
            return False
        if any(table_source_content_present(table) for table in getattr(page, "tables", None) or []):
            return False
        sealed_bundle_content_keys = (
            "lines",
            "tokens",
            "candidates",
            "structures",
            "micro_grid_structures",
            "source_table_geometry",
            "tables",
            "table_rows",
            "rows",
            "cells",
        )
        for bundle in evidence_bundles_by_logical.get(logical, ()):
            owners = (
                bundle,
                bundle.get("local_structure_evidence"),
                bundle.get("micro_grid_evidence"),
            )
            if any(
                isinstance(owner, Mapping) and any(owner.get(key) for key in sealed_bundle_content_keys)
                for owner in owners
            ):
                return False
            region_detect = bundle.get("region_detect")
            if isinstance(region_detect, Mapping) and region_detect.get("region_detect_candidates"):
                return False
            morphology_summary = bundle.get("morphology_summary")
            if isinstance(morphology_summary, Mapping) and any(
                value not in (None, "", 0, 0.0, False) for value in morphology_summary.values()
            ):
                return False
        return True

    def unresolved(
        reason: str,
        *,
        printed_by_logical: Mapping[int, int] | None = None,
        expected_total: int | None = None,
        full_footer_pages: Iterable[int] = (),
        page_only_footer_pages: Iterable[int] = (),
        paired_inferred_pages: Iterable[int] = (),
        blank_logical_pages: Iterable[int] = (),
    ) -> tuple[dict[int, int], dict[str, Any]]:
        observed = sorted(observed_pages)
        printed = {int(logical): int(page) for logical, page in (printed_by_logical or {}).items()}
        duplicate_printed_pages = sorted(page for page, count in Counter(printed.values()).items() if count > 1)
        return identity, {
            "resolved": False,
            "authoritative": False,
            "basis": "unresolved_identity_fallback",
            "reason": reason,
            "observed_logical_pages": observed,
            "identity_fallback": True,
            "printed_page_by_logical": printed,
            "unresolved_logical_pages": sorted(set(observed) - set(printed)),
            "duplicate_printed_pages": duplicate_printed_pages,
            "full_footer_logical_pages": sorted(set(full_footer_pages)),
            "page_only_footer_logical_pages": sorted(set(page_only_footer_pages)),
            "paired_inferred_logical_pages": sorted(set(paired_inferred_pages)),
            "blank_logical_pages": sorted(set(blank_logical_pages)),
            **({"printed_total": expected_total} if expected_total is not None else {}),
        }

    if len(observed_pages) < 2:
        return identity, {
            "resolved": True,
            "authoritative": True,
            "basis": "single_or_empty_page",
            "reason": "cross_page_order_not_required",
            "observed_logical_pages": sorted(observed_pages),
            "identity_fallback": False,
            "printed_page_by_logical": dict(identity),
            "unresolved_logical_pages": [],
            "duplicate_printed_pages": [],
            "full_footer_logical_pages": [],
            "page_only_footer_logical_pages": [],
            "paired_inferred_logical_pages": [],
            "blank_logical_pages": [],
        }

    printed_by_logical: dict[int, int] = {}
    totals: list[int] = []
    full_footer_pages: set[int] = set()
    ambiguous_full_footer_pages: set[int] = set()
    for logical in sorted(observed_pages):
        page_height = max(page_heights_by_logical.get(logical) or (0.0,))
        exact_matches = {
            (int(match.group("page")), int(match.group("total")))
            for text, evidence in text_evidence_by_page.get(logical, ())
            if _bottom_furniture_geometry(evidence, page_height=page_height)
            for match in _PRINTED_PAGE_RE.finditer(text)
        }
        if len(exact_matches) == 1:
            printed, total = next(iter(exact_matches))
            if 1 <= printed <= total:
                printed_by_logical[logical] = printed
                totals.append(total)
                full_footer_pages.add(logical)
        elif len(exact_matches) > 1:
            ambiguous_full_footer_pages.add(logical)

    if ambiguous_full_footer_pages:
        return unresolved(
            "ambiguous_full_footer",
            printed_by_logical=printed_by_logical,
            full_footer_pages=full_footer_pages,
        )

    observed_totals = set(totals)
    if len(observed_totals) != 1 or next(iter(observed_totals), 0) <= 0:
        return unresolved(
            "printed_total_missing_or_conflicting",
            printed_by_logical=printed_by_logical,
            full_footer_pages=full_footer_pages,
        )
    expected_total = next(iter(observed_totals))

    page_only_footer_pages: set[int] = set()
    ambiguous_page_only_pages: set[int] = set()
    for logical in sorted(observed_pages - set(printed_by_logical)):
        page_height = max(page_heights_by_logical.get(logical) or (0.0,))
        page_only_matches = {
            int(match.group("page"))
            for text, evidence in text_evidence_by_page.get(logical, ())
            if _bottom_furniture_geometry(evidence, page_height=page_height)
            if (match := _PRINTED_PAGE_ONLY_RE.fullmatch(text)) is not None
            and 1 <= int(match.group("page")) <= expected_total
        }
        if len(page_only_matches) == 1:
            printed_by_logical[logical] = next(iter(page_only_matches))
            page_only_footer_pages.add(logical)
        elif len(page_only_matches) > 1:
            ambiguous_page_only_pages.add(logical)

    if ambiguous_page_only_pages:
        return unresolved(
            "ambiguous_page_only_footer",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
        )

    # A sparse sealed excerpt (printed total larger than its observed page
    # population) cannot prove an extra scan half. Its empty synthetic/partial
    # siblings remain eligible for bounded footer inference. Blank-tail
    # certification is available only when the observed population could
    # already contain the whole numbered document.
    blank_logical_pages = (
        {logical for logical in observed_pages - set(printed_by_logical) if source_evidence_empty(logical)}
        if expected_total <= len(observed_pages)
        else set()
    )
    paired_inferred_pages = _infer_paired_printed_pages(
        printed_by_logical,
        source_by_logical,
        topology=topology,
        expected_total=expected_total,
        full_footer_logical_pages=full_footer_pages,
        excluded_logical_pages=blank_logical_pages,
    )
    unprinted_nonblank_pages = observed_pages - set(printed_by_logical) - blank_logical_pages
    if unprinted_nonblank_pages:
        return unresolved(
            "logical_page_footer_unresolved",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )
    if len(set(printed_by_logical.values())) != len(printed_by_logical):
        return unresolved(
            "printed_page_nonunique",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )

    printed_pages = set(printed_by_logical.values())
    if max(printed_pages, default=0) > expected_total or min(printed_pages, default=1) < 1:
        return unresolved(
            "printed_page_out_of_range",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )

    # Partial sealed contexts may legitimately contain a sparse, uniquely
    # numbered report excerpt.  Once an unnumbered empty scan half is accepted,
    # however, require the numbered side to be the complete 1..M document so a
    # missing business page can never be disguised as that blank tail.
    if blank_logical_pages and printed_pages != set(range(1, expected_total + 1)):
        return unresolved(
            "printed_page_permutation_incomplete",
            printed_by_logical=printed_by_logical,
            expected_total=expected_total,
            full_footer_pages=full_footer_pages,
            page_only_footer_pages=page_only_footer_pages,
            paired_inferred_pages=paired_inferred_pages,
            blank_logical_pages=blank_logical_pages,
        )

    ordered_report_pages = sorted(
        printed_by_logical,
        key=lambda page: (printed_by_logical[page], page),
    )
    ordered_logical_pages = ordered_report_pages + sorted(blank_logical_pages)
    order = {logical: index for index, logical in enumerate(ordered_logical_pages, start=1)}
    has_blank_tail = bool(blank_logical_pages)
    return order, {
        "resolved": True,
        "authoritative": True,
        "basis": (
            "complete_unique_printed_page_permutation_with_blank_tail"
            if has_blank_tail
            else "complete_unique_printed_page_permutation"
        ),
        "reason": (
            "full_page_total_bounded_pair_resolution_and_source_empty_tail"
            if has_blank_tail
            else "full_page_total_and_bounded_pair_resolution"
        ),
        "observed_logical_pages": sorted(observed_pages),
        "identity_fallback": False,
        "printed_page_by_logical": dict(sorted(printed_by_logical.items())),
        "unresolved_logical_pages": [],
        "duplicate_printed_pages": [],
        "full_footer_logical_pages": sorted(full_footer_pages),
        "page_only_footer_logical_pages": sorted(page_only_footer_pages),
        "paired_inferred_logical_pages": sorted(paired_inferred_pages),
        "blank_logical_pages": sorted(blank_logical_pages),
        "printed_total": expected_total,
    }


def _printed_reading_order(
    parse_result: Any,
    topology: PersonalDetailPageTopology | None = None,
) -> dict[int, int]:
    """Compatibility view of the authoritative order or sealed fallback."""

    return _printed_reading_order_resolution(parse_result, topology)[0]


def _infer_paired_printed_pages(
    printed_by_logical: dict[int, int],
    source_by_logical: dict[int, int],
    *,
    topology: PersonalDetailPageTopology | None = None,
    expected_total: int | None = None,
    full_footer_logical_pages: Iterable[int] = (),
    excluded_logical_pages: Iterable[int] = (),
) -> set[int]:
    """Infer unread spread footers only from a repeated imposition profile.

    A core ``two_page_spread`` label proves that two logical fragments share a
    source surface; it does not prove report-page order.  Establish that order
    from at least two independent spreads whose two full printed footers are
    consecutive left-to-right and whose validated crop/orientation profiles
    agree.  Booklet scans, top/bottom crops, and a lone example therefore stay
    unresolved rather than manufacturing an ``N +/- 1`` identity.
    """
    inferred: set[int] = set()
    if topology is None:
        return inferred
    excluded = {int(value) for value in excluded_logical_pages}
    full_footers = {int(value) for value in full_footer_logical_pages}
    logicals_by_source: dict[int, list[int]] = {}
    for logical, source in source_by_logical.items():
        logicals_by_source.setdefault(source, []).append(logical)

    def affine_signature(logical: int) -> tuple[float, ...] | None:
        page = topology.page(logical)
        transform = getattr(page, "coordinate_transform", None) if page is not None else None
        transform = transform if isinstance(transform, Mapping) else {}
        matrix = _matrix3(transform.get("matrix"))
        if matrix is None:
            return None
        source_x_norm = math.hypot(matrix[0][0], matrix[1][0])
        source_y_norm = math.hypot(matrix[0][1], matrix[1][1])
        if source_x_norm <= 1e-12 or source_y_norm <= 1e-12:
            return None
        return (
            matrix[0][0] / source_x_norm,
            matrix[1][0] / source_x_norm,
            matrix[0][1] / source_y_norm,
            matrix[1][1] / source_y_norm,
            source_x_norm / source_y_norm,
        )

    def numeric_profile_matches(left: Iterable[float], right: Iterable[float]) -> bool:
        left_values = tuple(float(value) for value in left)
        right_values = tuple(float(value) for value in right)
        return len(left_values) == len(right_values) and all(
            math.isclose(left_value, right_value, rel_tol=0.02, abs_tol=0.01)
            for left_value, right_value in zip(left_values, right_values, strict=True)
        )

    def spread_profiles_match(
        left: tuple[int, tuple[float, ...], tuple[float, ...]],
        right: tuple[int, tuple[float, ...], tuple[float, ...]],
    ) -> bool:
        return (
            left[0] == right[0]
            and numeric_profile_matches(left[1], right[1])
            and numeric_profile_matches(left[2], right[2])
        )

    def spread_profile(
        logicals: Iterable[int],
    ) -> tuple[int, tuple[float, ...], tuple[float, ...]] | None:
        ordered = topology.ordered_pair(logicals)
        if ordered is None:
            return None
        left, right = ordered
        left_geometry = topology.geometry(left)
        right_geometry = topology.geometry(right)
        if (
            left_geometry is None
            or right_geometry is None
            or not left_geometry.transform_usable
            or not right_geometry.transform_usable
            or left_geometry.split_kind != "two_page_spread"
            or right_geometry.split_kind != "two_page_spread"
            or left_geometry.segment_index != 0
            or right_geometry.segment_index != 1
            or left_geometry.selected_rotation != right_geometry.selected_rotation
            or left_geometry.source_crop_bbox is None
            or right_geometry.source_crop_bbox is None
        ):
            return None
        left_crop = left_geometry.source_crop_bbox
        right_crop = right_geometry.source_crop_bbox
        left_height = left_crop[3] - left_crop[1]
        right_height = right_crop[3] - right_crop[1]
        vertical_overlap = max(
            0.0,
            min(left_crop[3], right_crop[3]) - max(left_crop[1], right_crop[1]),
        )
        # Crop coordinates are source evidence.  Both halves must describe a
        # horizontal imposition; a top/bottom partition cannot establish
        # left-to-right report reading order.
        if vertical_overlap < min(left_height, right_height) * 0.98:
            return None
        if left_crop[2] <= right_crop[0]:
            source_crop_order = 1
        elif right_crop[2] <= left_crop[0]:
            source_crop_order = -1
        else:
            return None
        if source_crop_order != 1:
            return None
        left_transform = affine_signature(left)
        right_transform = affine_signature(right)
        if (
            left_transform is None
            or right_transform is None
            or not numeric_profile_matches(left_transform, right_transform)
        ):
            return None

        union_x0 = min(left_crop[0], right_crop[0])
        union_y0 = min(left_crop[1], right_crop[1])
        union_x1 = max(left_crop[2], right_crop[2])
        union_y1 = max(left_crop[3], right_crop[3])
        union_width = union_x1 - union_x0
        union_height = union_y1 - union_y0
        display_width = left_geometry.width + right_geometry.width
        display_height = max(left_geometry.height, right_geometry.height)
        if union_width <= 0 or union_height <= 0 or display_width <= 0 or display_height <= 0:
            return None

        def normalized_crop(crop: tuple[float, float, float, float]) -> tuple[float, ...]:
            return (
                (crop[0] - union_x0) / union_width,
                (crop[1] - union_y0) / union_height,
                (crop[2] - union_x0) / union_width,
                (crop[3] - union_y0) / union_height,
            )

        crop_and_dimension_profile = (
            *normalized_crop(left_crop),
            *normalized_crop(right_crop),
            union_width / union_height,
            left_geometry.width / display_width,
            right_geometry.width / display_width,
            left_geometry.height / display_height,
            right_geometry.height / display_height,
        )
        return (
            left_geometry.selected_rotation,
            left_transform,
            crop_and_dimension_profile,
        )

    fully_printed_profiles: list[tuple[int, tuple[float, ...], tuple[float, ...]]] = []
    for logicals in logicals_by_source.values():
        ordered = topology.ordered_pair(logicals)
        if ordered is None or not set(ordered).issubset(full_footers):
            continue
        left, right = ordered
        profile = spread_profile(logicals)
        if profile is None or printed_by_logical.get(right) != printed_by_logical.get(left, 0) + 1:
            # One nonconsecutive or geometrically incompatible fully printed
            # spread disproves a document-wide sequential imposition profile.
            return inferred
        fully_printed_profiles.append(profile)

    if len(fully_printed_profiles) < 2 or not all(
        spread_profiles_match(fully_printed_profiles[0], profile) for profile in fully_printed_profiles[1:]
    ):
        return inferred
    proved_profile = fully_printed_profiles[0]

    for logicals in logicals_by_source.values():
        ordered = topology.ordered_pair(logicals)
        profile = spread_profile(logicals)
        if ordered is None or profile is None or not spread_profiles_match(profile, proved_profile):
            continue
        left, right = ordered
        if left in excluded or right in excluded:
            continue
        if left in printed_by_logical and right not in printed_by_logical:
            candidate = printed_by_logical[left] + 1
            if (
                expected_total is None or 1 <= candidate <= expected_total
            ) and candidate not in printed_by_logical.values():
                printed_by_logical[right] = candidate
                inferred.add(right)
        elif right in printed_by_logical and left not in printed_by_logical:
            candidate = printed_by_logical[right] - 1
            if (
                expected_total is None or 1 <= candidate <= expected_total
            ) and candidate not in printed_by_logical.values():
                printed_by_logical[left] = candidate
                inferred.add(left)
    return inferred


def _evidence_key(page: int, line: dict[str, Any], index: int) -> str:
    evidence_ids = tuple(str(value) for value in line.get("evidence_ids") or [] if value)
    if evidence_ids:
        return f"evidence:{'|'.join(evidence_ids)}"
    box = _bbox(line)
    box_key = ":".join(f"{value:.2f}" for value in box) if box else ""
    return f"evidence:p{page}:{index}:{box_key}:{_compact(line.get('text') or line.get('content'))[:80]}"


def _collect_personal_detail_units(
    parse_result: Any,
    *,
    topology: PersonalDetailPageTopology | None = None,
    registered_reading_order: Mapping[int, int] | None = None,
    registered_reading_order_resolution: Mapping[str, Any] | None = None,
) -> tuple[
    tuple[CreditReportUnit, ...],
    tuple[str, ...],
    dict[str, str],
    dict[int, int],
    dict[int, int],
    dict[str, Any],
]:
    candidates: list[CreditReportUnit] = []
    furniture: set[str] = set()
    evidence_units: dict[str, str] = {}
    source_pages: dict[int, int] = {}
    pages = list(getattr(parse_result, "pages", None) or [])
    table_owners: dict[int, list[tuple[tuple[float, float, float, float], str]]] = {}
    text_owners: dict[int, list[tuple[str, tuple[float, float, float, float] | None, str]]] = {}
    edge_occurrences: dict[str, list[tuple[int, str]]] = {}
    if registered_reading_order is None:
        reading_order, reading_order_resolution = _printed_reading_order_resolution(
            parse_result,
            topology,
        )
    else:
        reading_order = {int(logical): int(position) for logical, position in registered_reading_order.items()}
        reading_order_resolution = deepcopy(dict(registered_reading_order_resolution or {}))

    for page_index, page in enumerate(pages, start=1):
        logical = int(getattr(page, "page_number", 0) or page_index)
        source = int(getattr(page, "source_page_number", 0) or logical)
        source_pages[logical] = source
        width = _finite(getattr(page, "width", 0))
        height = _finite(getattr(page, "height", 0))
        positioned: list[tuple[float, int, int, CreditReportUnit]] = []
        table_boxes = tuple(box for table in getattr(page, "tables", None) or [] if (box := _bbox(table)))
        for table_index, table in enumerate(getattr(page, "tables", None) or []):
            rows = _raw_rows(table)
            if not rows:
                continue
            table_id = str(getattr(table, "table_id", "") or f"p{logical}:t{table_index}")
            box = _bbox(table)
            unit_id = f"personal_detail:table:p{logical}:{table_id}"
            unit = CreditReportUnit(
                unit_id=unit_id,
                page=reading_order.get(logical, logical),
                order=0,
                source_index=table_index,
                kind="table",
                text="\n".join(" | ".join(row) for row in rows),
                bbox=box,
                page_width=width,
                page_height=height,
                table_id=table_id,
                rows=rows,
            )
            positioned.append((box[1] if box else 10000.0 + table_index, 1, table_index, unit))
            if box is not None:
                table_owners.setdefault(logical, []).append((box, unit_id))

        for text_index, block in enumerate(getattr(page, "texts", None) or []):
            content = str(getattr(block, "content", "") or "").strip()
            box = _bbox(block)
            if not content or _owned_by_table(box, table_boxes):
                continue
            unit_id = f"personal_detail:text:p{logical}:{text_index}"
            unit = CreditReportUnit(
                unit_id=unit_id,
                page=reading_order.get(logical, logical),
                order=0,
                source_index=text_index,
                kind=_kind(content),
                text=content,
                bbox=box,
                page_width=width,
                page_height=height,
            )
            positioned.append((box[1] if box else float(text_index), 0, text_index, unit))
            text_owners.setdefault(logical, []).append((_compact(content), box, unit_id))
            if _PAGE_NUMBER_RE.match(_compact(content)):
                furniture.add(unit_id)
            elif box is not None and height > 0 and (box[1] / height <= 0.08 or box[3] / height >= 0.90):
                compact = _compact(content)
                if len(compact) <= 80:
                    edge_occurrences.setdefault(compact, []).append((logical, unit_id))

        for order, (_top, _kind_order, _index, unit) in enumerate(sorted(positioned)):
            candidates.append(replace(unit, order=order))

    bundles = _domain_specific(parse_result).get("_page_evidence_bundles") or []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        local = bundle.get("local_structure_evidence")
        if not isinstance(local, dict):
            continue
        logical = int(bundle.get("page") or local.get("page") or 0)
        if logical <= 0:
            continue
        source_pages.setdefault(logical, int(bundle.get("source_page_number") or local.get("source_page") or logical))
        lines = [dict(line) for line in local.get("lines") or [] if isinstance(line, dict)]
        width = _finite(
            local.get("page_width") or bundle.get("page_width") or local.get("width") or bundle.get("width")
        )
        height = _finite(
            local.get("page_height") or bundle.get("page_height") or local.get("height") or bundle.get("height")
        )
        ordered_lines = sorted(
            lines,
            key=lambda item: (
                (_bbox(item) or (0, 0, 0, 0))[1],
                (_bbox(item) or (0, 0, 0, 0))[0],
            ),
        )
        for line_index, line in enumerate(ordered_lines):
            content = str(line.get("text") or line.get("content") or "").strip()
            if not content:
                continue
            evidence_key = _evidence_key(logical, line, line_index)
            box = _bbox(line)
            owner = _geometry_owner(box, table_owners.get(logical, ()))
            if not owner:
                compact = _compact(content)
                owner = next(
                    (
                        unit_id
                        for native_text, native_box, unit_id in text_owners.get(logical, ())
                        if native_text == compact
                        or (native_box is not None and _geometry_owner(box, ((native_box, unit_id),)))
                    ),
                    "",
                )
            if owner:
                evidence_units[evidence_key] = owner
                continue
            unit_id = f"personal_detail:evidence:p{logical}:{line_index}"
            evidence_units[evidence_key] = unit_id
            candidates.append(
                CreditReportUnit(
                    unit_id=unit_id,
                    page=reading_order.get(logical, logical),
                    order=line_index,
                    source_index=line_index,
                    kind=_kind(content),
                    text=content,
                    bbox=box,
                    page_width=width,
                    page_height=height,
                )
            )
            if _PAGE_NUMBER_RE.match(_compact(content)):
                furniture.add(unit_id)
            elif box is not None and height > 0 and (box[1] / height <= 0.08 or box[3] / height >= 0.90):
                compact = _compact(content)
                if len(compact) <= 80:
                    edge_occurrences.setdefault(compact, []).append((logical, unit_id))

    page_count = len(source_pages)
    recurrence_minimum = max(2, math.ceil(page_count * 0.5))
    for occurrences in edge_occurrences.values():
        if len({page for page, _unit_id in occurrences}) >= recurrence_minimum:
            furniture.update(unit_id for _page, unit_id in occurrences)

    active_by_page: dict[int, list[CreditReportUnit]] = {}
    for unit in candidates:
        if unit.unit_id not in furniture:
            active_by_page.setdefault(unit.page, []).append(unit)
    active: list[CreditReportUnit] = []
    for page in sorted(active_by_page):
        page_units = sorted(
            active_by_page[page],
            key=lambda unit: (
                unit.bbox[1] if unit.bbox is not None else 10000.0 + unit.order,
                1 if unit.kind == "table" else 0,
                unit.source_index,
                unit.unit_id,
            ),
        )
        active.extend(replace(unit, order=order) for order, unit in enumerate(page_units))
    return (
        tuple(active),
        tuple(sorted(furniture)),
        evidence_units,
        source_pages,
        reading_order,
        reading_order_resolution,
    )


class PersonalDetailExtractionContext:
    """Variant-owned logical page graph and copy-on-read extraction cache."""

    def __init__(
        self,
        parse_result: Any,
        entity_context: CreditReportEntityContext,
        *,
        evidence_unit_ids: dict[str, str],
        source_page_by_logical: dict[int, int],
        reading_order_by_logical: dict[int, int],
        reading_order_resolution: Mapping[str, Any],
        page_topology: PersonalDetailPageTopology,
    ) -> None:
        self.parse_result = parse_result
        # Canonical template pages are detached from ParseResult, but exact
        # merged-cell decoders still need the immutable OCR atom store.  Expose
        # the lossless runtime plane on the context so every source pass uses
        # the same evidence IDs/bboxes rather than falling back to raw cell text.
        canonical_plane = getattr(parse_result, "evidence_plane", None)
        self.evidence_plane = (
            canonical_plane.to_runtime()
            if canonical_plane is not None and callable(getattr(canonical_plane, "to_runtime", None))
            else canonical_plane
        )
        self.entity_context = entity_context
        self.evidence_unit_ids = MappingProxyType(dict(evidence_unit_ids))
        # Plugin-owned logical subpages are added during static topology
        # construction, so these two ledgers intentionally remain mutable while
        # the sealed ParseResult and evidence IDs stay immutable.
        self.source_page_by_logical = dict(source_page_by_logical)
        self._conserved_source_page_by_logical = dict(source_page_by_logical)
        self.reading_order_by_logical = dict(reading_order_by_logical)
        self.reading_order_resolution = deepcopy(dict(reading_order_resolution))
        self.page_topology = page_topology
        self._cache: dict[str, Any] = {}
        self._candidate_b_printed_anchor_inventory: list[dict[str, Any]] = []
        self._candidate_b_printed_anchor_inventory_complete = False
        self._frozen_logical_pages: dict[int, Any] = {
            int(getattr(page, "page_number", 0) or index): page
            for index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1)
        }
        self._topology_recovery_issues: list[dict[str, Any]] = []
        self._initial_personal_detail_extraction_issues = deepcopy(
            getattr(self, "_personal_detail_extraction_issues", [])
        )
        self._canonical_layout_projection_cache: Any | None = None
        self._canonical_projection_conservation_by_phase: dict[str, dict[str, Any]] = {}
        self._conserved_corrected_evidence_pages_cache: tuple[dict[str, Any], ...] | None = None
        self._conserved_corrected_evidence_sha256 = ""
        self._pboc_layout_profile_cache: Any | None = None
        self._canonical_entity_context_ready = False
        self._business_repair_plan: Any | None = None
        self._business_repair_evidence_by_page: dict[int, dict[str, Any]] = {}
        self._business_repair_active = False
        # The contract is one attempt per frozen logical page, not a document-
        # wide quota.  A fixed quota silently deprived later uncertain business
        # fields of their only field-aware correction pass on longer reports.
        self._page_ocr_max_requests = max(
            len(self._frozen_logical_pages),
            len(self.source_page_by_logical),
        )
        self._page_reocr_registry = OneShotPageReOCRRegistry(max_pages=self._page_ocr_max_requests)
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            PersonalDetailOCRCorrectionOverlay,
        )

        self._page_image_resolver = PersonalDetailLogicalPageImageResolver(
            parse_result,
            topology=page_topology,
        )
        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(parse_result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parse_result, name)

    @property
    def pages(self) -> list[Any]:
        """Return detached canonical pages, never the sealed ParseResult pages."""
        return list(self._canonical_layout_projection().pages)

    def cached(self, key: str, factory: Callable[[], _T]) -> _T:
        if key not in self._cache:
            self._cache[key] = deepcopy(factory())
        return cast(_T, deepcopy(self._cache[key]))

    def _sealed_printed_monthly_anchor_index(self) -> dict[tuple[int, tuple[str, ...]], dict[str, Any]]:
        """Publish the original source census independently of detector output."""

        anchors, complete = self.cached(
            "authenticated_printed_monthly_anchor_inventory",
            lambda: _authenticated_printed_monthly_anchor_inventory(self),
        )
        # Deduplicate only the complete immutable proof. Date ranges, page-local
        # IDs, detector IDs and transformed boxes alone never identify a grid.
        inventory = {
            json.dumps(anchor, sort_keys=True, ensure_ascii=False): deepcopy(anchor)
            for anchor in anchors.values()
        }
        self._candidate_b_printed_anchor_inventory = [inventory[key] for key in sorted(inventory)]
        self._candidate_b_printed_anchor_inventory_complete = complete is True
        return anchors

    def account_collections(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import _extract_accounts

        return cast(
            tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
            self.cached("account_collections", lambda: _extract_accounts(self)),
        )

    def native_business(self, full_text: str) -> dict[str, Any]:
        """Compatibility view of the single Candidate B business result."""
        return deepcopy(self.candidate_b_extraction(full_text).business)

    def scanned_business(self, full_text: str) -> dict[str, Any]:
        """Compatibility view; no shared scanned extractor is invoked."""
        return deepcopy(self.candidate_b_extraction(full_text).business)

    def candidate_b_extraction(self, full_text: str) -> Any:
        """Build and retain the only business extraction for this document."""
        key = "candidate_b_extraction"
        if key not in self._cache:
            from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
                CandidateBPipeline,
            )

            self._cache[key] = CandidateBPipeline(self, full_text).run()
        return self._cache[key]

    def prepare_candidate_b_business_repair(self, payload: dict[str, Any]) -> bool:
        """Plan the sole post-schema page repair and prepare a second pass."""
        from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
            BusinessUncertaintyRepairCoordinator,
        )

        coordinator = BusinessUncertaintyRepairCoordinator(self.parse_result, monthly_context=self)
        plan = coordinator.plan(
            payload,
            canonical_audit=self.canonical_layout_audit(),
            extraction_issues=(
                dict(issue)
                for issue in getattr(self, "_personal_detail_extraction_issues", ())
                if isinstance(issue, Mapping)
            ),
        )
        plan = coordinator.resolve_page_evidence(
            plan,
            source_pages=self._source_evidence_pages(),
            page_ocr_loader=self.full_page_ocr_evidence,
        )
        self._business_repair_plan = plan
        reconstruction_evidence = getattr(
            plan,
            "reconstruction_evidence",
            plan.page_evidence,
        )
        self._business_repair_evidence_by_page = deepcopy(reconstruction_evidence)
        if not plan.requires_second_pass:
            return False

        # The first pass exists only to discover schema uncertainty.  A true
        # template failure installs reconstructed pages for the affected stage;
        # ordinary field repairs retain the discovery pages and consult their
        # independent OCR acquisition only through the exact-field overlay.
        self._business_repair_active = True
        self._personal_detail_extraction_issues = deepcopy(self._initial_personal_detail_extraction_issues)
        self._cache.clear()
        if reconstruction_evidence:
            # Account anchors are cached directly on the context because
            # several account/table consumers share the same source skeleton.
            # A genuine template reconstruction can split or recover an anchor,
            # so only that path invalidates the canonical page plane.  A
            # field-only repair continues to use the discovery projection.
            discovery_account_skeletons = self.__dict__.get("_candidate_b_account_anchor_skeleton_cache")
            if isinstance(discovery_account_skeletons, list):
                self._candidate_b_pre_repair_account_anchor_inventory = tuple(deepcopy(discovery_account_skeletons))
            self.__dict__.pop("_candidate_b_account_anchor_skeleton_cache", None)
            self._canonical_layout_projection_cache = None
            self._pboc_layout_profile_cache = None
            self._canonical_entity_context_ready = False
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            PersonalDetailOCRCorrectionOverlay,
        )

        self._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(self.parse_result)
        self._ocr_correction_overlay.install_business_repair_evidence(
            plan.page_evidence.values(),
            affected_pages=plan.affected_pages,
            allowed_target_refs=(
                {**dict(ref), "field_name": repair.field_name}
                for repair in getattr(plan, "field_repairs", ())
                for ref in repair.source_refs
            ),
        )
        return True

    def candidate_b_planned_field_repair(
        self,
        *,
        dataset_name: str,
        record_id: str,
        field_name: str,
        observed_value: Any,
        source_refs: Iterable[Mapping[str, Any]],
        mode: str | None = None,
    ) -> Any | None:
        """Return the one coordinator directive bound to this exact field."""

        plan = self._business_repair_plan
        if not self._business_repair_active or plan is None:
            return None
        resolver = getattr(plan, "field_repair_for", None)
        if not callable(resolver):
            return None
        return resolver(
            dataset_name=str(dataset_name),
            record_id=str(record_id or ""),
            field_name=str(field_name),
            observed_value=observed_value,
            source_refs=source_refs,
            mode=mode,
        )

    def candidate_b_field_repair(
        self,
        value: Any,
        *,
        dataset_name: str,
        record_id: str,
        field_name: str,
        source_refs: Iterable[Mapping[str, Any]],
    ) -> tuple[str, Any | None]:
        """Apply one planned repair; acquisition and mutation scopes stay separate."""

        refs = tuple(dict(ref) for ref in source_refs if isinstance(ref, Mapping))
        repair = self.candidate_b_planned_field_repair(
            dataset_name=dataset_name,
            record_id=record_id,
            field_name=field_name,
            observed_value=value,
            source_refs=refs,
        )
        if repair is None:
            return str(value or ""), None
        return self._ocr_correction_overlay.repair_planned_text(
            value,
            repair=repair,
            source_refs=refs,
        )

    def correct_candidate_b_datasets(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Correct and validate all final v2 source datasets exactly once."""
        plan = self._business_repair_plan
        if plan is not None and plan.affected_pages:
            self._ocr_correction_overlay.install_business_repair_evidence(
                plan.page_evidence.values(),
                affected_pages=plan.affected_pages,
                allowed_target_refs=(
                    {**dict(ref), "field_name": repair.field_name}
                    for repair in getattr(plan, "field_repairs", ())
                    for ref in repair.source_refs
                ),
            )
        return self._ocr_correction_overlay.correct_business_candidates(
            payload,
            stage="candidate_b_final_validation",
        )

    def corrected_repayment_records(self) -> list[dict[str, Any]]:
        """Rebuild monthly cells from the same canonical page evidence as all fields.

        No cell-level OCR is permitted here. One-shot page re-OCR and
        canonical fragment registration happen before the repayment grid is
        materialized, so monthly performance cannot bypass the template layer.
        The sealed evidence plane remains unchanged.
        """

        def rebuild() -> list[dict[str, Any]]:
            from docmirror.models.mirror.domain_access import (
                micro_grid_structures_from_domain_specific,
            )
            from docmirror.plugins.credit_report.micro_grid_materialize import (
                augment_credit_repayment_evidence_bundles,
                materialize_credit_repayment_micro_grids_from_bundles,
            )
            from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
                report_localized_monthly_omissions,
            )
            from docmirror.plugins.credit_report.repayment_grid import (
                dedupe_repayment_records,
                records_from_micro_grid_dict,
            )
            from docmirror.plugins.credit_report.source_table_month_lattice import (
                detached_source_table_geometry_by_page,
            )

            def strip_cross_page_augmentation(bundle: dict[str, Any]) -> None:
                grid_evidence = bundle.get("micro_grid_evidence")
                if not isinstance(grid_evidence, dict):
                    return
                evidence_page = str(grid_evidence.get("page") or bundle.get("page") or "").strip()
                for evidence_key in ("lines", "tokens"):
                    if evidence_key not in grid_evidence:
                        continue
                    grid_evidence[evidence_key] = [
                        item
                        for item in grid_evidence.get(evidence_key) or []
                        if not (
                            isinstance(item, dict)
                            and (
                                item.get("coordinate_status") == "cross_page_y_shift"
                                or (str(item.get("source_logical_page") or "") not in {"", evidence_page})
                            )
                        )
                    ]
                grid_evidence.pop("credit_cross_page_augmented", None)
                grid_evidence.pop("continuation_logical_pages", None)
                grid_evidence.pop(
                    "continuation_source_table_geometry_by_page",
                    None,
                )

            def detached_geometry_from_unique_pages(
                pages: Any,
                selected_pages: set[int],
            ) -> tuple[dict[int, list[dict[str, Any]]], set[int]]:
                """Detach geometry only when page/table ownership is unique."""

                page_values = list(pages) if isinstance(pages, (list, tuple)) else []

                def freeze_geometry(value: Any) -> Any:
                    if isinstance(value, Mapping):
                        return tuple(
                            sorted(
                                (str(key), freeze_geometry(item))
                                for key, item in value.items()
                                if key not in {"table_id", "logical_page", "source_page"}
                            )
                        )
                    if isinstance(value, (list, tuple)):
                        return tuple(freeze_geometry(item) for item in value)
                    return value

                def raw_physical_signature(table: Any) -> Any | None:
                    metadata = table.get("metadata") if isinstance(table, Mapping) else getattr(table, "metadata", None)
                    metadata = metadata if isinstance(metadata, Mapping) else {}
                    geometry = metadata.get("geometry")
                    geometry = geometry if isinstance(geometry, Mapping) else {}
                    cell_bboxes = geometry.get("cell_bboxes")
                    if not isinstance(cell_bboxes, (list, tuple)):
                        cell_bboxes = metadata.get("cell_bboxes")
                    bbox = table.get("bbox") if isinstance(table, Mapping) else getattr(table, "bbox", None)
                    if not isinstance(bbox, (list, tuple)) and not isinstance(cell_bboxes, (list, tuple)):
                        return None
                    return freeze_geometry(
                        {
                            "bbox": bbox if isinstance(bbox, (list, tuple)) else None,
                            "cell_bboxes": (cell_bboxes if isinstance(cell_bboxes, (list, tuple)) else None),
                        }
                    )

                page_counts: Counter[int] = Counter()
                page_values_by_logical: dict[int, list[Any]] = {}
                for fallback_page, page_value in enumerate(page_values, start=1):
                    page_number = (
                        page_value.get("page_number")
                        if isinstance(page_value, Mapping)
                        else getattr(page_value, "page_number", None)
                    )
                    logical_page = (
                        page_number
                        if isinstance(page_number, int) and not isinstance(page_number, bool)
                        else fallback_page
                    )
                    if logical_page in selected_pages:
                        page_counts[logical_page] += 1
                        page_values_by_logical.setdefault(logical_page, []).append(page_value)

                blocked_pages = {page for page in selected_pages if page_counts.get(page, 0) != 1}
                raw_table_owners: dict[str, set[int]] = {}
                for page in selected_pages - blocked_pages:
                    page_value = page_values_by_logical[page][0]
                    raw_tables = (
                        page_value.get("tables")
                        if isinstance(page_value, Mapping)
                        else getattr(page_value, "tables", None)
                    )
                    if not isinstance(raw_tables, (list, tuple)):
                        blocked_pages.add(page)
                        continue
                    raw_table_ids = [
                        str(
                            (
                                table.get("table_id") or table.get("id")
                                if isinstance(table, Mapping)
                                else getattr(table, "table_id", None) or getattr(table, "id", None)
                            )
                            or ""
                        ).strip()
                        for table in raw_tables
                    ]
                    nonempty_raw_ids = [table_id for table_id in raw_table_ids if table_id]
                    raw_signatures = [
                        signature for table in raw_tables if (signature := raw_physical_signature(table)) is not None
                    ]
                    for table_id in set(nonempty_raw_ids):
                        raw_table_owners.setdefault(table_id, set()).add(page)
                    if len(nonempty_raw_ids) != len(set(nonempty_raw_ids)) or len(raw_signatures) != len(
                        set(raw_signatures)
                    ):
                        blocked_pages.add(page)
                        continue
                for owners in raw_table_owners.values():
                    if len(owners) > 1:
                        blocked_pages.update(owners)

                detached_by_page = detached_source_table_geometry_by_page(page_values)
                geometry_by_page: dict[int, list[dict[str, Any]]] = {}
                table_owners: dict[str, set[int]] = {}

                for page in selected_pages - blocked_pages:
                    tables = detached_by_page.get(page) or []
                    if not isinstance(tables, list) or not all(isinstance(table, Mapping) for table in tables):
                        blocked_pages.add(page)
                        continue
                    table_ids = [str(table.get("table_id") or "").strip() for table in tables]
                    nonempty_ids = [table_id for table_id in table_ids if table_id]
                    signatures = [freeze_geometry(table) for table in tables]
                    if len(nonempty_ids) != len(set(nonempty_ids)) or len(signatures) != len(set(signatures)):
                        blocked_pages.add(page)
                        continue
                    geometry_by_page[page] = deepcopy(tables)
                    for table_id in nonempty_ids:
                        table_owners.setdefault(table_id, set()).add(page)

                # Repeated physical layouts are expected across report pages;
                # only one table identity claiming multiple pages is ambiguous.
                for owners in table_owners.values():
                    if len(owners) > 1:
                        blocked_pages.update(owners)
                for page in blocked_pages:
                    geometry_by_page.pop(page, None)
                return geometry_by_page, blocked_pages

            detached = deepcopy(_domain_specific(self.parse_result))
            detached.pop("credit_repayment_records", None)
            cross_page_order_authoritative = _authoritative_reading_order(
                getattr(self, "reading_order_resolution", None)
            )
            for bundle in detached.get("_page_evidence_bundles") or []:
                if isinstance(bundle, dict):
                    bundle.pop("micro_grid_structures", None)
                    strip_cross_page_augmentation(bundle)

            canonical_pages: dict[int, dict[str, Any]] = {}
            geometry_blocked_pages: set[int] = set()
            for canonical_page in self.corrected_evidence_pages():
                if not isinstance(canonical_page, dict):
                    continue
                page = canonical_page.get("page")
                if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
                    continue
                if page in canonical_pages:
                    geometry_blocked_pages.add(page)
                    continue
                canonical_pages[page] = canonical_page

            selected_pages = set(canonical_pages)
            canonical_projection = getattr(
                self,
                "_canonical_layout_projection_cache",
                None,
            )
            if canonical_projection is None:
                # Lightweight/legacy contexts do not own a canonical projection;
                # retain their sealed, ordinary-page geometry behavior.
                geometry_pages = getattr(self.parse_result, "pages", None)
            else:
                # The evidence lines and physical cells must come from the same
                # transformed canonical page plane.  Never mix in sealed values.
                geometry_pages = getattr(canonical_projection, "pages", None)
            source_table_geometry_by_page, nonunique_geometry_pages = detached_geometry_from_unique_pages(
                geometry_pages, selected_pages
            )
            geometry_blocked_pages.update(nonunique_geometry_pages)
            for page in geometry_blocked_pages:
                source_table_geometry_by_page.pop(page, None)
            exact_repair_tokens_by_page = _exact_source_table_repair_tokens_by_page(
                self,
                geometry_pages,
                selected_pages - geometry_blocked_pages,
            )

            def monthly_evidence_tokens(
                page: int,
                lines: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                exact_tokens = deepcopy(exact_repair_tokens_by_page.get(page) or [])
                exact_ids = {str(token.get("token_id") or "") for token in exact_tokens}
                exact_visual_keys = {
                    (
                        str(token.get("content") or ""),
                        tuple(round(float(value), 3) for value in token.get("bbox") or ()),
                    )
                    for token in exact_tokens
                }
                retained_lines: list[dict[str, Any]] = []
                for line in deepcopy(lines):
                    evidence_ids = tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
                    text = str(line.get("text") or line.get("content") or "").strip()
                    bbox = tuple(round(float(value), 3) for value in line.get("bbox") or ())
                    if (len(evidence_ids) == 1 and evidence_ids[0] in exact_ids) or (text, bbox) in exact_visual_keys:
                        continue
                    retained_lines.append(line)
                return retained_lines + exact_tokens

            def monthly_coordinate_lines(
                page: int,
                values: Iterable[Mapping[str, Any]],
            ) -> list[dict[str, Any]]:
                """Expose one coordinate page while retaining fragment origin."""

                normalized: list[dict[str, Any]] = []
                for value in values:
                    line = deepcopy(dict(value))
                    coordinate_page = line.get("coordinate_logical_page")
                    if coordinate_page is None:
                        normalized.append(line)
                        continue
                    if (
                        not isinstance(coordinate_page, int)
                        or isinstance(coordinate_page, bool)
                        or coordinate_page != page
                    ):
                        # A transformed line claiming another coordinate frame
                        # cannot safely enter this page-local reconstruction.
                        continue
                    source_origin = line.get("source_logical_page")
                    if (
                        isinstance(source_origin, int)
                        and not isinstance(source_origin, bool)
                        and source_origin > 0
                    ):
                        line["source_origin_logical_page"] = source_origin
                    line["source_logical_page"] = coordinate_page
                    normalized.append(line)
                return normalized

            observed_pages: set[int] = set()
            for bundle in detached.get("_page_evidence_bundles") or []:
                if not isinstance(bundle, dict):
                    continue
                local = bundle.get("local_structure_evidence")
                page = int(bundle.get("page") or (local or {}).get("page") or 0)
                canonical = canonical_pages.get(page)
                if canonical is None:
                    continue
                observed_pages.add(page)
                lines = monthly_coordinate_lines(page, canonical.get("lines") or ())
                if not isinstance(local, dict):
                    local = {}
                    bundle["local_structure_evidence"] = local
                local.update(
                    {
                        "page": page,
                        "source_page": int(canonical.get("source_page") or page),
                        "page_width": canonical.get("page_width"),
                        "page_height": canonical.get("page_height"),
                        "lines": lines,
                    }
                )
                grid_evidence = bundle.get("micro_grid_evidence")
                if not isinstance(grid_evidence, dict):
                    grid_evidence = {}
                    bundle["micro_grid_evidence"] = grid_evidence
                # Canonical lines replace the prior evidence view, so any
                # augmentation marker on that prior view is stale. Rebuild the
                # adjacent continuation and its detached geometry atomically.
                grid_evidence.pop("credit_cross_page_augmented", None)
                grid_evidence.pop("continuation_logical_pages", None)
                grid_evidence.pop(
                    "continuation_source_table_geometry_by_page",
                    None,
                )
                grid_evidence.update(
                    {
                        "page": page,
                        "page_width": canonical.get("page_width"),
                        "page_height": canonical.get("page_height"),
                        "lines": deepcopy(lines),
                        # Ordinary sealed tokens do not outrank the canonical
                        # page. Exact singleton table atoms are reintroduced
                        # only for the later source-lattice field repair.
                        "tokens": monthly_evidence_tokens(page, lines),
                        "source_table_geometry": deepcopy(source_table_geometry_by_page.get(page) or []),
                    }
                )
            for page, canonical in canonical_pages.items():
                if page in observed_pages:
                    continue
                lines = monthly_coordinate_lines(page, canonical.get("lines") or ())
                detached.setdefault("_page_evidence_bundles", []).append(
                    {
                        "page": page,
                        "source_page_number": int(canonical.get("source_page") or page),
                        "local_structure_evidence": {
                            "page": page,
                            "source_page": int(canonical.get("source_page") or page),
                            "page_width": canonical.get("page_width"),
                            "page_height": canonical.get("page_height"),
                            "lines": deepcopy(lines),
                        },
                        "micro_grid_evidence": {
                            "page": page,
                            "page_width": canonical.get("page_width"),
                            "page_height": canonical.get("page_height"),
                            "lines": lines,
                            "tokens": monthly_evidence_tokens(page, lines),
                            "source_table_geometry": deepcopy(source_table_geometry_by_page.get(page) or []),
                        },
                    }
                )
            if cross_page_order_authoritative:
                augment_credit_repayment_evidence_bundles(
                    detached,
                    reading_order_by_logical=dict(self.reading_order_by_logical),
                )
            status_glyph_observations: list[dict[str, Any]] = []
            materialize_credit_repayment_micro_grids_from_bundles(
                detached,
                page_image_resolver=getattr(self, "_page_image_resolver", None),
                enable_cell_ocr=False,
                enable_static_status_validation=True,
                extra_status_chars={"A", "#"},
                enable_candidate_b_amount_pairing=True,
                candidate_b_status_glyph_observations=status_glyph_observations,
            )
            # Ephemeral visual evidence is held only on this plugin context.
            # It is never inserted into the detached ParseResult view or any
            # micro-grid/dataset projection.
            self._candidate_b_status_glyph_observations = status_glyph_observations
            corrected_grids = micro_grid_structures_from_domain_specific(detached)
            self._corrected_repayment_micro_grids = deepcopy(corrected_grids)
            records = [
                record
                for grid in corrected_grids
                for record in records_from_micro_grid_dict(
                    grid,
                    accept_exact_row_numeric_status=True,
                )
            ]
            deduped = dedupe_repayment_records(records)

            # Build an OCR-free structural witness from the detached sealed
            # page evidence.  It is never returned and none of its cell values
            # can enter the business projection; it only detects complete grid
            # ranges that canonical registration may have missed.
            source_baseline = deepcopy(_domain_specific(self.parse_result))
            source_baseline.pop("credit_repayment_records", None)
            printed_anchors = self._sealed_printed_monthly_anchor_index()
            for bundle in source_baseline.get("_page_evidence_bundles") or []:
                if isinstance(bundle, dict):
                    bundle.pop("micro_grid_structures", None)
                    strip_cross_page_augmentation(bundle)
                    for evidence_key in ("local_structure_evidence", "micro_grid_evidence"):
                        evidence = bundle.get(evidence_key)
                        if isinstance(evidence, dict):
                            evidence["lines"] = _lines_with_printed_monthly_anchors(
                                evidence.get("lines") or (),
                                logical_page=bundle.get("page"),
                                source_page=bundle.get("source_page_number", evidence.get("source_page")),
                                anchors=printed_anchors,
                            )
            if cross_page_order_authoritative:
                augment_credit_repayment_evidence_bundles(
                    source_baseline,
                    reading_order_by_logical=dict(self.reading_order_by_logical),
                )
            materialize_credit_repayment_micro_grids_from_bundles(
                source_baseline,
                page_image_resolver=None,
                enable_cell_ocr=False,
                extra_status_chars={"A", "#"},
            )
            source_structure_grids = micro_grid_structures_from_domain_specific(source_baseline)
            source_structure_records = dedupe_repayment_records(
                [record for grid in source_structure_grids for record in records_from_micro_grid_dict(grid)]
            )
            # Keep the detached structural plane private and value-inert.  The
            # relationship layer may use only its printed range/cell geometry
            # to reconcile extraction-local grid aliases after a unique account
            # owner is proven; no detached status value enters business rows.
            self._candidate_b_monthly_source_structure_grids = deepcopy(source_structure_grids)
            self._candidate_b_monthly_source_structure_records = deepcopy(source_structure_records)
            self._candidate_b_printed_grid_census_required = True
            source_structure_count = len(source_structure_records)

            def exact_monthly_key(
                record: Mapping[str, Any],
            ) -> tuple[str, int, int] | None:
                refs = [ref for ref in record.get("source_cell_refs") or () if isinstance(ref, Mapping)]
                grid_ids = {
                    str(value).strip()
                    for value in (
                        record.get("grid_id"),
                        *(ref.get("grid_id") for ref in refs),
                    )
                    if str(value or "").strip()
                }
                raw_month = str(record.get("performance_month") or "").strip()
                if raw_month:
                    match = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])", raw_month)
                    if match is None:
                        return None
                    year, month = int(match.group(1)), int(match.group(2))
                else:
                    raw_year = record.get("year")
                    raw_month_number = record.get("month")
                    if (
                        isinstance(raw_year, bool)
                        or not isinstance(raw_year, int)
                        or isinstance(raw_month_number, bool)
                        or not isinstance(raw_month_number, int)
                    ):
                        return None
                    year, month = raw_year, raw_month_number
                if len(grid_ids) != 1 or not 2000 <= year <= 2099 or not 1 <= month <= 12:
                    return None
                return next(iter(grid_ids)), year, month

            canonical_keys = {
                key
                for record in deduped
                if isinstance(record, Mapping) and (key := exact_monthly_key(record)) is not None
            }
            source_records_by_key: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
            for record in source_structure_records:
                if not isinstance(record, Mapping):
                    continue
                key = exact_monthly_key(record)
                if key is not None:
                    source_records_by_key.setdefault(key, []).append(record)
            source_keys = set(source_records_by_key)
            missing_source_keys = source_keys - canonical_keys
            # Grid IDs are parser-local. Physical aliases cannot be collapsed
            # safely until the relationship layer has proved one unique
            # account owner for both positions, so this early structural
            # ledger intentionally retains every missing key.
            unreconciled_source_keys = missing_source_keys
            grids_by_id = {
                str(grid.get("grid_id") or ""): grid
                for grid in source_structure_grids
                if isinstance(grid, Mapping) and str(grid.get("grid_id") or "")
            }
            for grid_id in sorted({key[0] for key in unreconciled_source_keys}):
                localized_keys = sorted(key for key in unreconciled_source_keys if key[0] == grid_id)
                report_localized_monthly_omissions(
                    self,
                    issue_code="canonical_monthly_source_structure_missing_field",
                    message=(
                        "A detached source-structure grid/month position was absent from the deduplicated "
                        "canonical monthly population."
                    ),
                    parser_stage="canonical_monthly_grid_materialization",
                    grid_id=grid_id,
                    months=((key[1], key[2]) for key in localized_keys),
                    source_records=(record for key in localized_keys for record in source_records_by_key[key]),
                    grid=grids_by_id.get(grid_id),
                    observed_context={"source_structure_key_count": len(localized_keys)},
                    reason_codes=(
                        "detached_source_structure_exact_key",
                        "canonical_deduplicated_key_missing",
                        "source_structure_is_audit_only",
                        "account_month_owner_reconciliation_pending",
                        "dataset_incomplete",
                    ),
                )

            # Grid IDs are extraction-local aliases, and gaps between two
            # printed ranges are not implied months.  Report only the detached
            # source-position delta here; the relationship layer owns the
            # unique account/month reconciliation and the public denominator.
            unreconciled_source_position_count = len(unreconciled_source_keys)
            if unreconciled_source_position_count:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                record_issue(
                    self,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="canonical_monthly_reconstruction_incomplete",
                        message=(
                            "The canonical pass omitted detached grid/month source positions; they remain "
                            "audit-only until a unique account owner proves canonical account/month identities."
                        ),
                        parser_stage="canonical_monthly_grid_materialization",
                        target_dataset="repayment_records",
                        observed_value={"canonical_row_count": len(deduped)},
                        candidate_value={
                            "source_structure_row_count": source_structure_count,
                            "unreconciled_source_position_count": (unreconciled_source_position_count),
                            "account_month_expected_row_count": None,
                            "localization_status": ("pending_unique_account_owner_reconciliation"),
                        },
                        reason_codes=(
                            "cell_level_ocr_disabled",
                            "canonical_page_evidence_only",
                            "source_structure_is_audit_only",
                            "raw_grid_positions_not_a_population_denominator",
                            "printed_ranges_do_not_imply_intervening_months",
                            "dataset_incomplete",
                        ),
                    ),
                )
            account_gap_issues = [
                issue
                for issue in getattr(self, "_personal_detail_extraction_issues", ())
                if isinstance(issue, Mapping)
                and issue.get("issue_code") == "candidate_b_account_sequence_gap"
                and str(issue.get("status") or "open") != "resolved"
            ]
            if account_gap_issues:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                missing_by_family = {
                    str((issue.get("observed_value") or {}).get("account_type") or "unknown"): list(
                        (issue.get("candidate_value") or {}).get("missing_category_sequences") or ()
                    )
                    for issue in account_gap_issues
                }
                record_issue(
                    self,
                    make_issue(
                        category="schema_incompleteness",
                        issue_code="monthly_population_incomplete_from_account_gap",
                        message=(
                            "Monthly performance cannot be population-complete while one or more account-family "
                            "ordinals are unresolved; no months were invented for the missing accounts."
                        ),
                        parser_stage="canonical_monthly_grid_materialization",
                        target_dataset="repayment_records",
                        observed_value={"canonical_grid_row_count": len(deduped)},
                        candidate_value={"missing_account_category_sequences": missing_by_family},
                        source_refs=(
                            dict(ref)
                            for issue in account_gap_issues
                            for ref in issue.get("source_refs") or ()
                            if isinstance(ref, Mapping)
                        ),
                        reason_codes=(
                            "credit_account_population_incomplete",
                            "monthly_population_cannot_be_complete",
                            "missing_months_not_invented",
                        ),
                    ),
                )
            return deduped

        return self.cached("corrected_repayment_records", rebuild)

    def corrected_repayment_micro_grids(self) -> list[dict[str, Any]]:
        """Return the exact canonical grids used to materialize monthly rows."""
        self.corrected_repayment_records()
        return deepcopy(getattr(self, "_corrected_repayment_micro_grids", []))

    def candidate_b_status_glyph_observations(self) -> list[dict[str, Any]]:
        """Return private document-local glyph evidence for the final gate."""
        self.corrected_repayment_records()
        return deepcopy(getattr(self, "_candidate_b_status_glyph_observations", []))

    def section_content(self, full_text: str) -> dict[str, Any]:
        """Return supplemental datasets from the same Candidate B result."""
        return deepcopy(self.candidate_b_extraction(full_text).section_content)

    def corrected_evidence_pages(self) -> list[dict[str, Any]]:
        """Return only extraction-safe, registered canonical evidence.

        This is intentionally a canonical subset.  Call
        :meth:`conserved_corrected_evidence_pages` when auditing lossless page
        conservation; unresolved or explicitly blank source pages must never
        be reopened to generic extraction merely to make the counts equal.
        """
        return deepcopy(list(self._canonical_layout_projection().evidence_pages))

    def conserved_corrected_evidence_pages(self) -> list[dict[str, Any]]:
        """Return the immutable, lossless corrected source-page plane.

        One ordinary OCR bundle contributes exactly one page.  A source page
        may contribute two pages only through the registered static two-way
        split path.  Business-repair evidence is deliberately excluded: the
        discovery and repaired canonical projections may change their admitted
        subset, but they cannot rewrite this frozen source census.
        """

        if self._conserved_corrected_evidence_pages_cache is None:
            frozen = tuple(deepcopy(self._build_source_evidence_pages()))
            self._conserved_corrected_evidence_pages_cache = frozen
            serialized = json.dumps(
                frozen,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            self._conserved_corrected_evidence_sha256 = hashlib.sha256(serialized).hexdigest()
        return deepcopy(list(self._conserved_corrected_evidence_pages_cache))

    def _source_evidence_pages(self) -> list[dict[str, Any]]:
        """Return the phase-local evidence used to register canonical pages."""

        conserved = self.conserved_corrected_evidence_pages()
        if not self._business_repair_active:
            return conserved
        return self._build_business_repaired_source_evidence_pages(conserved)

    def _canonical_layout_projection(self) -> Any:
        if self._canonical_layout_projection_cache is None:
            from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
                PBOCCanonicalTemplateAssembler,
            )

            assembler = PBOCCanonicalTemplateAssembler(
                self.parse_result,
                topology=self.page_topology,
                reading_order_by_logical=self.reading_order_by_logical,
                source_evidence_loader=self._source_evidence_pages,
                issue_owner=self,
                source_page_loader=lambda: list(self._frozen_logical_pages.values()),
            )
            self._canonical_layout_projection_cache = assembler.build()
            phase = "business_repair" if self._business_repair_active else "discovery"
            phase_audit = self._canonical_subset_conservation_audit(
                self._canonical_layout_projection_cache,
                phase=phase,
            )
            # Preserve the first completed audit for each phase.  A later
            # repaired projection must not erase discovery-pass withholds.
            self._canonical_projection_conservation_by_phase.setdefault(
                phase,
                deepcopy(phase_audit),
            )
            self._adopt_canonical_entity_context()
        return self._canonical_layout_projection_cache

    def _adopt_canonical_entity_context(self) -> None:
        """Rebuild continuation identities over detached canonical pages."""
        if self._canonical_entity_context_ready or self._canonical_layout_projection_cache is None:
            return
        adapter = SimpleNamespace(
            pages=list(self._canonical_layout_projection_cache.pages),
            entities=SimpleNamespace(domain_specific={}),
        )
        (
            units,
            furniture,
            evidence_units,
            source_pages,
            _reading_order,
            _reading_order_resolution,
        ) = _collect_personal_detail_units(
            adapter,
            topology=self.page_topology,
            registered_reading_order=self.reading_order_by_logical,
            registered_reading_order_resolution=self.reading_order_resolution,
        )
        if units:
            policy = PersonalDetailTransitionPolicy()
            self.entity_context = decode_credit_report_units(
                units,
                report_family="personal_detail",
                furniture_unit_ids=furniture,
                transition_scorer=policy.score,
                entity_prefix="personal_detail_canonical",
            )
            self.evidence_unit_ids = MappingProxyType(dict(evidence_units))
            self.source_page_by_logical.clear()
            self.source_page_by_logical.update(source_pages)
        self._canonical_entity_context_ready = True

    def _build_source_evidence_pages(self) -> list[dict[str, Any]]:
        domain_specific = _domain_specific(self.parse_result)
        printed_anchors = self._sealed_printed_monthly_anchor_index()
        pages: list[dict[str, Any]] = []
        for bundle in domain_specific.get("_page_evidence_bundles") or []:
            if not isinstance(bundle, dict):
                continue
            local = bundle.get("local_structure_evidence")
            if not isinstance(local, dict):
                continue
            lines = _lines_with_printed_monthly_anchors(
                local.get("lines") or (),
                logical_page=bundle.get("page", local.get("page")),
                source_page=bundle.get("source_page_number", local.get("source_page")),
                anchors=printed_anchors,
            )
            if not lines:
                continue
            pages.append(
                {
                    "page": int(bundle.get("page") or local.get("page") or 0),
                    "source_page": int(bundle.get("source_page_number") or local.get("source_page") or 0),
                    "page_width": _finite(
                        local.get("page_width") or bundle.get("page_width") or local.get("width") or bundle.get("width")
                    ),
                    "page_height": _finite(
                        local.get("page_height")
                        or bundle.get("page_height")
                        or local.get("height")
                        or bundle.get("height")
                    ),
                    "lines": sorted(
                        lines,
                        key=lambda line: (
                            (_bbox(line) or (0, 0, 0, 0))[1],
                            (_bbox(line) or (0, 0, 0, 0))[0],
                        ),
                    ),
                }
            )
        ordered = sorted(
            pages,
            key=lambda item: (
                self.reading_order_by_logical.get(item["page"], item["page"]),
                item["page"],
            ),
        )
        conserved = self._construct_static_topology_pages(ordered)
        self._conserved_source_page_by_logical = {
            int(page.get("page") or 0): int(page.get("source_page") or 0) for page in conserved
        }
        return conserved

    def _build_business_repaired_source_evidence_pages(
        self,
        conserved_pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Overlay repair evidence without mutating the conserved page census."""

        merged: list[dict[str, Any]] = []
        affected = set(self._business_repair_evidence_by_page)
        for source in conserved_pages:
            logical = int(source.get("page") or 0)
            replacement = self._business_repair_evidence_by_page.get(logical)
            if replacement is None:
                merged.append(source)
                continue
            merged.append(
                {
                    **source,
                    **deepcopy(replacement),
                    "page": logical,
                    "logical_page": logical,
                    "source_page": int(replacement.get("source_page") or source.get("source_page") or logical),
                }
            )
        known = {int(page.get("page") or 0) for page in merged}
        merged.extend(
            deepcopy(page) for logical, page in self._business_repair_evidence_by_page.items() if logical not in known
        )
        # Line-level normalization is permitted only now: the first schema
        # pass selected these pages as business-uncertain.
        selected = [page for page in merged if int(page.get("page") or 0) in affected]
        corrected_by_page = {
            int(page.get("page") or 0): page for page in self._ocr_correction_overlay.corrected_evidence_pages(selected)
        }
        return [corrected_by_page.get(int(page.get("page") or 0), page) for page in merged]

    def _construct_static_topology_pages(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Freeze statically split pages and partition existing evidence."""
        audit = self.page_topology.audit()
        source_candidates = {
            int(source)
            for source, logicals in (audit.get("logical_pages_by_source") or {}).items()
            if len(logicals) == 1
            and (
                (geometry := self.page_topology.geometry(int(logicals[0]))) is not None
                and (geometry.split_kind == "two_page_spread" or geometry.split_confidence >= 0.55)
            )
        }
        if not source_candidates:
            return pages
        split_pages = self._page_image_resolver.static_split_slices(source_candidates)
        for decision in self._page_image_resolver.audit().get("static_split_decisions") or []:
            if not isinstance(decision, Mapping) or decision.get("status") not in {"uncertain", "failed"}:
                continue
            source = int(decision.get("source_page") or 0)
            failed = decision.get("status") == "failed"
            issue_code = "static_page_split_validation_failed" if failed else "static_page_split_uncertain"
            if any(
                item.get("code") == issue_code and int(item.get("source_page") or 0) == source
                for item in self._topology_recovery_issues
            ):
                continue
            self._topology_recovery_issues.append(
                {
                    "code": issue_code,
                    "message": (
                        "Static page-split validation could not run; the original logical page was preserved."
                        if failed
                        else "Static image geometry could not safely confirm or reject a potential page split."
                    ),
                    **deepcopy(dict(decision)),
                }
            )
        if not split_pages:
            return pages

        base_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in pages:
            base_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)
        static_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in split_pages:
            static_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)

        next_logical = max(self.source_page_by_logical, default=0) + 1
        merged: list[dict[str, Any]] = []
        for source in sorted(set(base_by_source) | set(static_by_source)):
            base_pages = base_by_source.get(source, [])
            constructed_pages = sorted(
                static_by_source.get(source, []),
                key=lambda item: int(item.get("segment_index") or 0),
            )
            if not constructed_pages:
                merged.extend(base_pages)
                continue
            logicals = self.page_topology.logicals_for_source(source)
            geometries = {
                int(geometry.segment_index): logical
                for logical in logicals
                if (geometry := self.page_topology.geometry(logical)) is not None
                and geometry.split_kind == "two_page_spread"
                and geometry.segment_index in {0, 1}
            }
            base_by_segment: dict[int, dict[str, Any]] = {}
            for page in base_pages:
                geometry = self.page_topology.geometry(int(page.get("page") or 0))
                if (
                    geometry is not None
                    and geometry.split_kind == "two_page_spread"
                    and geometry.segment_index in {0, 1}
                ):
                    base_by_segment[int(geometry.segment_index)] = page

            split_by_segment = {int(page.get("segment_index") or 0): page for page in constructed_pages}
            available_segments = set(base_by_segment) | set(split_by_segment)
            if available_segments != {0, 1}:
                # Never replace a source with only one half of a statically
                # confirmed spread. Preserve the original evidence and report
                # the topology uncertainty without invoking OCR.
                merged.extend(base_pages)
                if not any(
                    item.get("code") == "static_split_pair_incomplete" and int(item.get("source_page") or 0) == source
                    for item in self._topology_recovery_issues
                ):
                    self._topology_recovery_issues.append(
                        {
                            "code": "static_split_pair_incomplete",
                            "message": "Static spread validation did not produce both logical subpages.",
                            "source_page": source,
                            "expected_segments": [0, 1],
                            "observed_segments": sorted(available_segments),
                        }
                    )
                continue
            unsplit_logical = logicals[0] if len(logicals) == 1 and not geometries else 0
            partitioned_lines: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
            partitioned_tables: dict[int, list[Any]] = {0: [], 1: []}
            ambiguous_items = 0
            if unsplit_logical and base_pages:
                base_evidence = base_pages[0]
                raw_page = self._frozen_logical_pages.get(unsplit_logical)
                transform = dict(getattr(raw_page, "coordinate_transform", None) or {})
                base_to_source = _matrix3(transform.get("inverse_matrix"))
                if base_to_source is None:
                    merged.extend(base_pages)
                    self._topology_recovery_issues.append(
                        {
                            "code": "static_split_evidence_transform_unusable",
                            "message": (
                                "The original page evidence could not be projected into the static subpages; "
                                "the unsplit page was preserved."
                            ),
                            "source_page": source,
                            "logical_page": unsplit_logical,
                        }
                    )
                    continue
                for line in base_evidence.get("lines") or []:
                    if not isinstance(line, Mapping) or (box := _bbox(line)) is None:
                        ambiguous_items += 1
                        continue
                    source_box = _transform_bbox(base_to_source, box)
                    segment, ambiguous = _slice_assignment(source_box, split_by_segment)
                    if ambiguous or segment is None:
                        ambiguous_items += 1
                        continue
                    source_to_slice = _matrix3(split_by_segment[segment].get("source_to_logical"))
                    if source_to_slice is None:
                        ambiguous_items += 1
                        continue
                    local = deepcopy(dict(line))
                    local["source_bbox"] = list(source_box)
                    local["bbox"] = list(_transform_bbox(source_to_slice, source_box))
                    partitioned_lines[segment].append(local)
                for table in getattr(raw_page, "tables", None) or []:
                    if (box := _bbox(table)) is None:
                        ambiguous_items += 1
                        continue
                    source_box = _transform_bbox(base_to_source, box)
                    segment, ambiguous = _slice_assignment(source_box, split_by_segment)
                    if ambiguous or segment is None:
                        ambiguous_items += 1
                        continue
                    source_to_slice = _matrix3(split_by_segment[segment].get("source_to_logical"))
                    if source_to_slice is None:
                        ambiguous_items += 1
                        continue
                    partitioned_tables[segment].append(
                        _project_table_for_static_slice(
                            table,
                            base_to_source=base_to_source,
                            source_to_slice=source_to_slice,
                        )
                    )
            if ambiguous_items:
                self._topology_recovery_issues.append(
                    {
                        "code": "static_split_boundary_ambiguous",
                        "message": "Evidence crossing the static split boundary was withheld from both subpages.",
                        "source_page": source,
                        "ambiguous_item_count": ambiguous_items,
                    }
                )
            for segment in (0, 1):
                if segment in base_by_segment:
                    merged.append(base_by_segment[segment])
                    continue
                static_page = split_by_segment.get(segment)
                if static_page is None:
                    continue
                logical = geometries.get(segment, 0)
                if not logical and unsplit_logical and segment == 0:
                    logical = unsplit_logical
                if not logical:
                    logical = next_logical
                    next_logical += 1
                static_page = dict(static_page)
                static_page["page"] = logical
                static_page["logical_page"] = logical
                static_page["plugin_static_subpage"] = True
                static_page["lines"] = sorted(
                    partitioned_lines.get(segment, []),
                    key=lambda line: ((_bbox(line) or (0, 0, 0, 0))[1], (_bbox(line) or (0, 0, 0, 0))[0]),
                )
                self.source_page_by_logical[logical] = source
                self._page_image_resolver.register_static_logical_page(logical, static_page)
                # The resolver owns the frozen pixel surface. Canonical
                # evidence remains lightweight and serializable.
                static_page.pop("image", None)
                self._frozen_logical_pages[logical] = SimpleNamespace(
                    page_number=logical,
                    source_page_number=source,
                    width=float(static_page.get("page_width") or 0.0),
                    height=float(static_page.get("page_height") or 0.0),
                    coordinate_transform=deepcopy(static_page.get("coordinate_transform") or {}),
                    tables=partitioned_tables.get(segment, []),
                    texts=[
                        SimpleNamespace(
                            content=str(line.get("text") or ""),
                            bbox=list(line.get("bbox") or []),
                        )
                        for line in static_page["lines"]
                    ],
                )
                merged.append(static_page)

        order_keys: dict[int, tuple[int, int, int]] = {}
        for logical, source in self.source_page_by_logical.items():
            source_logicals = self.page_topology.logicals_for_source(source)
            source_order = min(
                (self.reading_order_by_logical.get(item, item) for item in source_logicals),
                default=self.reading_order_by_logical.get(logical, logical),
            )
            geometry = self.page_topology.geometry(logical)
            segment = int(geometry.segment_index) if geometry and geometry.segment_index in {0, 1} else 0
            static_page = next((item for item in merged if int(item.get("page") or 0) == logical), None)
            if static_page and static_page.get("plugin_static_subpage"):
                segment = int(static_page.get("segment_index") or 0)
            order_keys[logical] = (source_order, segment, logical)
        self.reading_order_by_logical.clear()
        self.reading_order_by_logical.update(
            {
                logical: index
                for index, logical in enumerate(sorted(order_keys, key=lambda item: order_keys[item]), start=1)
            }
        )
        return sorted(
            merged,
            key=lambda item: (
                self.reading_order_by_logical.get(int(item.get("page") or 0), int(item.get("page") or 0)),
                int(item.get("segment_index") or 0),
            ),
        )

    def full_page_ocr_evidence(
        self,
        logical_pages: Iterable[int],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        """Return the one cached re-OCR result for each frozen logical page."""
        if os.environ.get("DOCMIRROR_PERSONAL_DETAIL_PAGE_OCR", "1") == "0":
            return []
        requested = sorted({int(page) for page in logical_pages if int(page) > 0})
        output: list[dict[str, Any]] = []
        for logical in requested:
            rendered = self._page_image_resolver(logical)
            page_key = self._page_image_resolver.page_key(logical)
            if not page_key:
                continue

            def produce() -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
                if not rendered:
                    return None, "render_failed", {"ocr_invocations": 0}
                image = rendered.get("image")
                shape = getattr(image, "shape", None)
                if not shape or len(shape) < 2 or not shape[0] or not shape[1]:
                    return None, "invalid_image", {"ocr_invocations": 0}
                ocr_result = _single_page_ocr(image)
                if isinstance(ocr_result, tuple) and len(ocr_result) == 3:
                    words, page_score, deskew = ocr_result
                else:  # Backward-compatible test seam.
                    words, page_score = ocr_result
                    deskew = {
                        "method": "hough_lines_p_v1",
                        "applied": False,
                        "angle": 0.0,
                        "reason": "test_seam_not_evaluated",
                    }
                page_width = float(rendered.get("page_width") or 0.0)
                page_height = float(rendered.get("page_height") or 0.0)
                if page_width <= 0 or page_height <= 0:
                    zoom = float(rendered.get("zoom") or 1.0)
                    page_width = float(shape[1]) / zoom
                    page_height = float(shape[0]) / zoom
                scale_x = page_width / float(shape[1])
                scale_y = page_height / float(shape[0])
                lines: list[dict[str, Any]] = []
                for index, word in enumerate(words):
                    text = str(word.get("text") or "").strip()
                    bbox = word.get("bbox")
                    confidence = float(word.get("confidence") or 0.0)
                    if not text or confidence < 0.45 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    lines.append(
                        {
                            "text": text,
                            "confidence": confidence,
                            "bbox": [
                                float(bbox[0]) * scale_x,
                                float(bbox[1]) * scale_y,
                                float(bbox[2]) * scale_x,
                                float(bbox[3]) * scale_y,
                            ],
                            "evidence_ids": [f"personal_detail_page_reocr:{page_key}:w{index}"],
                            "source": "personal_detail_page_reocr_once",
                        }
                    )
                details = {
                    "ocr_invocations": 1,
                    "word_count": len(lines),
                    "page_score": page_score,
                    "deskew": deepcopy(deskew),
                }
                if not lines:
                    return None, "ocr_empty", details
                transform = dict(rendered.get("coordinate_transform") or {})
                decomposition = dict(transform.get("decomposition") or {})
                return (
                    {
                        "page": logical,
                        "logical_page": logical,
                        "page_key": page_key,
                        "source_page": int(
                            rendered.get("source_page") or self.source_page_by_logical.get(logical, logical)
                        ),
                        "page_width": page_width,
                        "page_height": page_height,
                        "selected_rotation": int(decomposition.get("selected_rotation") or 0),
                        "deskew_method": str(deskew.get("method") or "hough_lines_p_v1"),
                        "deskew_applied": deskew.get("applied") is True,
                        "deskew_angle": float(deskew.get("angle") or 0.0),
                        "deskew_reason": str(deskew.get("reason") or "not_evaluated"),
                        "lines": sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0])),
                    },
                    "completed",
                    details,
                )

            page = self._page_reocr_registry.resolve(
                page_key=page_key,
                logical_page=logical,
                reason=str(reason),
                producer=produce,
            )
            if page is not None:
                output.append(page)
        return output

    def ocr_correction_audit(self) -> dict[str, Any]:
        """Return a detached audit snapshot for diagnostics and regression tests."""
        registry = self._page_reocr_registry.audit()
        requests = registry.pop("page_reocr_requests", [])
        core_deskew: list[dict[str, Any]] = []
        parse_result = self.__dict__.get("parse_result")
        for page in getattr(parse_result, "pages", None) or []:
            transform = getattr(page, "coordinate_transform", None) or {}
            if not isinstance(transform, Mapping) or not transform.get("deskew_method"):
                continue
            decomposition = transform.get("decomposition") or {}
            core_deskew.append(
                {
                    "logical_page": int(getattr(page, "page_number", 0) or 0),
                    "source_page": int(getattr(page, "source_page_number", 0) or 0),
                    "method": str(transform.get("deskew_method") or ""),
                    "applied": transform.get("deskew_applied") is True,
                    "angle": float(transform.get("deskew_angle") or 0.0),
                    "reason": str(transform.get("deskew_reason") or ""),
                    "support_line_count": (
                        int(decomposition.get("deskew_support_line_count") or 0)
                        if isinstance(decomposition, Mapping)
                        else 0
                    ),
                }
            )
        conservation = (
            self.corrected_evidence_conservation_audit()
            if "_conserved_corrected_evidence_pages_cache" in self.__dict__
            else {
                "schema": "docmirror.personal_detail.corrected_evidence_conservation.v1",
                "available": False,
                "reason": "lightweight_context_without_source_census",
            }
        )
        return {
            **deepcopy(self._ocr_correction_overlay.audit()),
            **registry,
            "corrected_evidence_conservation": conservation,
            "canonical_projection_phase_history": self._canonical_projection_phase_history(),
            "core_ocr_deskew": core_deskew,
            "page_reocr_deskew": [
                {
                    "page_key": str(row.get("page_key") or ""),
                    "logical_page": int(row.get("logical_page") or 0),
                    **deepcopy(dict(row.get("deskew") or {})),
                }
                for row in requests
                if isinstance(row, Mapping) and isinstance(row.get("deskew"), Mapping)
            ],
            "business_repair": (
                deepcopy(self.__dict__["_business_repair_plan"].audit())
                if self.__dict__.get("_business_repair_plan") is not None
                else {
                    "architecture": "schema_triggered_field_local_repair_v2",
                    "first_pass_uncertainty_count": 0,
                    "affected_pages": [],
                    "second_schema_pass_required": False,
                }
            ),
            "page_reocr_failures": [
                row
                for row in requests
                if isinstance(row, Mapping) and row.get("status") not in {"completed", "requested"}
            ],
        }

    def corrected_evidence_conservation_audit(self) -> dict[str, Any]:
        """Reconcile every raw logical OCR bundle to the frozen source plane."""

        raw_pages: list[dict[str, int]] = []
        for bundle in _domain_specific(self.parse_result).get("_page_evidence_bundles") or ():
            if not isinstance(bundle, Mapping):
                continue
            local = bundle.get("local_structure_evidence")
            if not isinstance(local, Mapping) or not any(
                isinstance(line, Mapping) for line in local.get("lines") or ()
            ):
                continue
            raw_pages.append(
                {
                    "logical_page": int(bundle.get("page") or local.get("page") or 0),
                    "source_page": int(bundle.get("source_page_number") or local.get("source_page") or 0),
                }
            )

        conserved_pages = self.conserved_corrected_evidence_pages()
        frozen_source_map = getattr(
            self,
            "_conserved_source_page_by_logical",
            self.source_page_by_logical,
        )
        raw_by_source: dict[int, list[int]] = {}
        for page in raw_pages:
            raw_by_source.setdefault(page["source_page"], []).append(page["logical_page"])
        conserved_by_source: dict[int, list[dict[str, Any]]] = {}
        for page in conserved_pages:
            conserved_by_source.setdefault(int(page.get("source_page") or 0), []).append(page)

        raw_logicals = [page["logical_page"] for page in raw_pages]
        conserved_logicals = [int(page.get("page") or 0) for page in conserved_pages]
        duplicate_raw_logicals = sorted(
            logical for logical, count in Counter(raw_logicals).items() if logical <= 0 or count != 1
        )
        duplicate_conserved_logicals = sorted(
            logical for logical, count in Counter(conserved_logicals).items() if logical <= 0 or count != 1
        )
        source_mappings: list[dict[str, Any]] = []
        static_replacement_sources: list[int] = []
        unexpected_conserved_logicals: set[int] = set(conserved_logicals).difference(raw_logicals)
        mapping_valid = True
        for source in sorted(set(raw_by_source) | set(conserved_by_source)):
            raw_for_source = sorted(raw_by_source.get(source, ()))
            conserved_for_source = sorted(
                conserved_by_source.get(source, ()),
                key=lambda page: (
                    int(page.get("segment_index") or 0),
                    int(page.get("page") or 0),
                ),
            )
            conserved_for_source_logicals = [int(page.get("page") or 0) for page in conserved_for_source]
            segments = sorted(int(page.get("segment_index") or 0) for page in conserved_for_source)
            has_static_registration = any(page.get("plugin_static_subpage") is True for page in conserved_for_source)
            registered_to_source = all(
                frozen_source_map.get(logical) == source and logical in self._frozen_logical_pages
                for logical in conserved_for_source_logicals
            )
            if (
                len(raw_for_source) == 1
                and len(conserved_for_source_logicals) == 2
                and segments == [0, 1]
                and has_static_registration
                and raw_for_source[0] in conserved_for_source_logicals
                and registered_to_source
            ):
                status = "registered_static_two_way_replacement"
                valid = True
                static_replacement_sources.append(source)
                unexpected_conserved_logicals.difference_update(
                    set(conserved_for_source_logicals).difference(raw_for_source)
                )
            else:
                status = "one_to_one_preserved"
                valid = (
                    raw_for_source == sorted(conserved_for_source_logicals)
                    and len(raw_for_source) == len(conserved_for_source_logicals)
                    and not has_static_registration
                    and registered_to_source
                )
                if not valid:
                    status = "invalid_or_incomplete_mapping"
            mapping_valid = mapping_valid and valid
            source_mappings.append(
                {
                    "source_page": source,
                    "status": status,
                    "valid": valid,
                    "raw_logical_pages": raw_for_source,
                    "conserved_logical_pages": conserved_for_source_logicals,
                    "conserved_segments": segments,
                    "plugin_static_registration": has_static_registration,
                    "registered_to_source": registered_to_source,
                }
            )

        dropped_raw_logicals = sorted(set(raw_logicals).difference(conserved_logicals))
        valid = (
            mapping_valid
            and not duplicate_raw_logicals
            and not duplicate_conserved_logicals
            and not dropped_raw_logicals
            and not unexpected_conserved_logicals
            and all(source > 0 for source in set(raw_by_source) | set(conserved_by_source))
        )
        return {
            "schema": "docmirror.personal_detail.corrected_evidence_conservation.v1",
            "valid": valid,
            "frozen_before_business_repair": True,
            "business_repair_can_mutate_conserved_plane": False,
            "raw_bundle_count": len(raw_pages),
            "conserved_page_count": len(conserved_pages),
            "raw_source_page_count": len(raw_by_source),
            "conserved_source_page_count": len(conserved_by_source),
            "raw_logical_pages": sorted(raw_logicals),
            "conserved_logical_pages": sorted(conserved_logicals),
            "static_replacement_sources": static_replacement_sources,
            "source_mappings": source_mappings,
            "duplicate_raw_logical_pages": duplicate_raw_logicals,
            "duplicate_conserved_logical_pages": duplicate_conserved_logicals,
            "dropped_raw_logical_pages": dropped_raw_logicals,
            "unexpected_conserved_logical_pages": sorted(unexpected_conserved_logicals),
            "conserved_plane_sha256": self._conserved_corrected_evidence_sha256,
        }

    def _canonical_subset_conservation_audit(
        self,
        projection: Any,
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Localize every source page omitted from one canonical projection."""

        conserved = self.conserved_corrected_evidence_pages()
        source_by_logical = {int(page.get("page") or 0): int(page.get("source_page") or 0) for page in conserved}
        registrations = {
            int(row.get("logical_page") or 0): row
            for row in getattr(projection, "registrations", ())
            if isinstance(row, Mapping)
        }
        admitted_projection_logicals = {
            int(logical)
            for group in getattr(projection, "fragment_groups", ())
            if isinstance(group, Mapping)
            for logical in group.get("fragment_logical_pages") or ()
        }
        conserved_logicals = set(source_by_logical)
        admitted_logicals = admitted_projection_logicals.intersection(conserved_logicals)
        unresolved_projection_logicals = {int(logical) for logical in getattr(projection, "unresolved_pages", ())}
        withheld_pages: list[dict[str, Any]] = []
        for logical in sorted(set(source_by_logical).difference(admitted_logicals)):
            registration = registrations.get(logical, {})
            registration_status = str(registration.get("status") or "missing")
            basis = str(registration.get("basis") or "")
            if registration_status == "blank" and basis == "explicitly_blank_fragment":
                issue_code = "canonical_blank_fragment_explicitly_withheld"
                localized = True
            elif registration_status == "unresolved" and basis:
                issue_code = "canonical_page_registration_failed"
                localized = True
            elif logical in unresolved_projection_logicals and basis:
                # A complete printed-page fragment group can be withheld after
                # its members were individually registered (for example, due
                # to conflicting template IDs).  The group-level issue is still
                # a localized canonical disposition, not a silent page loss.
                issue_code = "canonical_fragment_group_withheld"
                localized = True
            else:
                issue_code = "canonical_page_omission_unlocalized"
                localized = False
            withheld_pages.append(
                {
                    "logical_page": logical,
                    "source_page": source_by_logical[logical],
                    "registration_status": registration_status,
                    "template_id": str(registration.get("template_id") or ""),
                    "basis": basis,
                    "signals": list(registration.get("signals") or ()),
                    "localization_issue_code": issue_code,
                    "localized": localized,
                    "source_refs": [
                        {
                            "source": "canonical_template_registration",
                            "logical_page": logical,
                            "source_page": source_by_logical[logical],
                            "geometry_scope": "logical_page",
                        }
                    ],
                }
            )
        registered_logicals = set(registrations)
        partition_valid = (
            admitted_logicals.isdisjoint({row["logical_page"] for row in withheld_pages})
            and admitted_logicals | {row["logical_page"] for row in withheld_pages} == conserved_logicals
        )
        return {
            "phase": phase,
            "valid": (
                self.corrected_evidence_conservation_audit()["valid"]
                and registered_logicals.issuperset(conserved_logicals)
                and partition_valid
                and all(row["localized"] is True for row in withheld_pages)
            ),
            "conserved_plane_sha256": self._conserved_corrected_evidence_sha256,
            "canonical_page_count": len(getattr(projection, "pages", ())),
            "admitted_logical_page_count": len(admitted_logicals),
            "admitted_logical_pages": sorted(admitted_logicals),
            "admitted_non_census_logical_pages": sorted(admitted_projection_logicals.difference(conserved_logicals)),
            "withheld_logical_page_count": len(withheld_pages),
            "withheld_logical_pages": [row["logical_page"] for row in withheld_pages],
            "withheld_pages": withheld_pages,
            "unregistered_conserved_logical_pages": sorted(conserved_logicals.difference(registered_logicals)),
        }

    def _canonical_projection_phase_history(self) -> list[dict[str, Any]]:
        order = {"discovery": 0, "business_repair": 1}
        return [
            deepcopy(audit)
            for phase, audit in sorted(
                self.__dict__.get(
                    "_canonical_projection_conservation_by_phase",
                    {},
                ).items(),
                key=lambda item: (order.get(item[0], 99), item[0]),
            )
        ]

    def page_topology_audit(self) -> dict[str, Any]:
        """Return the plugin's detached logical-page validation result."""
        # Freezing the conserved plane may run the static, OCR-free split
        # validator.  Capture the resolver audit only after those decisions so
        # the returned snapshot cannot lag the page census it contains.
        conservation = self.corrected_evidence_conservation_audit()
        audit = deepcopy(self._page_image_resolver.audit())
        audit["issues"] = [*(audit.get("issues") or []), *deepcopy(self._topology_recovery_issues)]
        audit["ocr_used_for_topology"] = False
        audit["topology_frozen_before_reocr"] = True
        audit["corrected_evidence_conservation"] = conservation
        return audit

    def canonical_layout_audit(self) -> dict[str, Any]:
        """Return the detached template-registration and fragment audit."""
        projection = self._canonical_layout_projection()
        audit = deepcopy(projection.audit())
        audit["reading_order_resolution"] = deepcopy(self.reading_order_resolution)
        audit["pboc_layout_profile"] = self.pboc_layout_profile().audit()
        phase = "business_repair" if self._business_repair_active else "discovery"
        audit["corrected_evidence_conservation"] = self.corrected_evidence_conservation_audit()
        audit["canonical_subset_conservation"] = self._canonical_subset_conservation_audit(
            projection,
            phase=phase,
        )
        audit["canonical_projection_phase_history"] = self._canonical_projection_phase_history()
        return audit

    def pboc_layout_profile(self) -> Any:
        """Return the immutable evidence-detected PBOC layout profile."""

        if self._pboc_layout_profile_cache is None:
            from docmirror.plugins.credit_report.personal_detail_scanned.layout_profile import (
                detect_pboc_layout_profile,
            )

            self._pboc_layout_profile_cache = detect_pboc_layout_profile(self.pages)
        return self._pboc_layout_profile_cache

    def tables_continue(self, left_table_id: str, right_table_id: str) -> bool | None:
        left_unit_id = self.entity_context.table_unit_id(left_table_id)
        right_unit_id = self.entity_context.table_unit_id(right_table_id)
        if not left_unit_id or not right_unit_id:
            return None
        units_by_id = {unit.unit_id: unit for unit in self.entity_context.units}
        left_unit = units_by_id.get(left_unit_id)
        right_unit = units_by_id.get(right_unit_id)
        if left_unit is None or right_unit is None:
            return None
        left = self.entity_context.entity_for_unit(left_unit_id)
        right = self.entity_context.entity_for_unit(right_unit_id)
        left_entity_id = str(getattr(left, "entity_id", "") or "")
        right_entity_id = str(getattr(right, "entity_id", "") or "")
        if not left_entity_id or not right_entity_id:
            return None
        if left_unit.page != right_unit.page:
            if not _authoritative_reading_order(getattr(self, "reading_order_resolution", None)) or not (
                isinstance(left_unit.page, int)
                and not isinstance(left_unit.page, bool)
                and isinstance(right_unit.page, int)
                and not isinstance(right_unit.page, bool)
                and right_unit.page == left_unit.page + 1
            ):
                return False
        return left_entity_id == right_entity_id

    def pages_adjacent_in_reading_order(self, left_page: int, right_page: int) -> bool:
        if not _authoritative_reading_order(getattr(self, "reading_order_resolution", None)):
            return False
        if (
            not isinstance(left_page, int)
            or isinstance(left_page, bool)
            or left_page <= 0
            or not isinstance(right_page, int)
            or isinstance(right_page, bool)
            or right_page <= 0
        ):
            return False
        registered_positions = list(self.reading_order_by_logical.values())
        registered_logical_pages = set(getattr(self, "source_page_by_logical", {}) or {})
        if (
            any(
                not isinstance(position, int) or isinstance(position, bool) or position <= 0
                for position in registered_positions
            )
            or len(registered_positions) != len(set(registered_positions))
            or (registered_logical_pages and not registered_logical_pages.issubset(self.reading_order_by_logical))
        ):
            return False
        left_order = self.reading_order_by_logical.get(left_page)
        right_order = self.reading_order_by_logical.get(right_page)
        if (
            not isinstance(left_order, int)
            or isinstance(left_order, bool)
            or not isinstance(right_order, int)
            or isinstance(right_order, bool)
        ):
            return False
        return right_order == left_order + 1

    def allows_scanned_line_transition(
        self,
        left_page: int,
        left_line: dict[str, Any],
        left_index: int,
        right_page: int,
        right_line: dict[str, Any],
        right_index: int,
    ) -> bool | None:
        if left_page == right_page:
            return True
        if not _authoritative_reading_order(getattr(self, "reading_order_resolution", None)):
            return False
        if not self.pages_adjacent_in_reading_order(left_page, right_page):
            return False
        left_id = self.evidence_unit_ids.get(_evidence_key(left_page, left_line, left_index))
        right_id = self.evidence_unit_ids.get(_evidence_key(right_page, right_line, right_index))
        if not left_id or not right_id:
            # Page-edge placement corroborates a known continuation, but it
            # cannot create entity identity.  Adjacent PBOC sections routinely
            # end and begin near these same margins, so accepting geometry alone
            # can join unrelated records across a real section boundary.
            return None
        left = self.entity_context.entity_for_unit(left_id)
        right = self.entity_context.entity_for_unit(right_id)
        return bool(left is not None and right is not None and left.entity_id == right.entity_id)


def build_personal_detail_extraction_context(parse_result: Any) -> PersonalDetailExtractionContext:
    """Build the detailed-report logical-page graph exactly once."""
    if isinstance(parse_result, PersonalDetailExtractionContext):
        return parse_result
    page_topology = PersonalDetailPageTopology(parse_result)
    (
        units,
        furniture,
        evidence_units,
        source_pages,
        reading_order,
        reading_order_resolution,
    ) = _collect_personal_detail_units(parse_result, topology=page_topology)
    policy = PersonalDetailTransitionPolicy()
    entity_context = decode_credit_report_units(
        units,
        report_family="personal_detail",
        furniture_unit_ids=furniture,
        transition_scorer=policy.score,
        entity_prefix="personal_detail",
    )
    return PersonalDetailExtractionContext(
        parse_result,
        entity_context,
        evidence_unit_ids=evidence_units,
        source_page_by_logical=source_pages,
        reading_order_by_logical=reading_order,
        reading_order_resolution=reading_order_resolution,
        page_topology=page_topology,
    )


__all__ = [
    "PersonalDetailExtractionContext",
    "PersonalDetailTransitionPolicy",
    "build_personal_detail_extraction_context",
]
