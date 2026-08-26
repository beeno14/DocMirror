# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative physical table reconstruction for OCR-only pages."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from docmirror.models.entities.domain import Block

_NUMBER_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?")
_LEADING_NUMBER_AND_LABEL_RE = re.compile(r"^\s*([+-]?\d[\d,，\s]*(?:\.\s*\d+)?)\s*([\u3400-\u9fff].+)$")
_LABEL_AND_ORDINAL_HEADER_RE = re.compile(r"^\s*(.+?)(行次|栏次|序号|次行)\s*$")
_GRID_NUMERIC_RE = re.compile(r"^(?:[-+−—一]?\d[\d,]*(?:\.\d+)?|[-−—一]+)$")
_GRID_ORDINAL_RE = re.compile(r"^\d{1,3}[A-Za-z]?(?:[=＝+−≤≥<>-].+)?$")
_ORDINAL_HEADER_RE = re.compile(r"^(?:行次|栏次|序号|次行)$")
_SHORT_NOISE_RE = re.compile(r"^[A-Za-z0-9]{1,3}$")
_EXACT_DECIMAL_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}$")
_CJK_ORDINALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
_CJK_ORDINAL_RE = re.compile(r"^(?P<open>[（(]?)(?P<ordinal>[一二三四五六七八九十]+)[）)](?P<suffix>.*)$")
_MISSING_CJK_ORDINAL_RE = re.compile(r"^[）)](?P<suffix>.+)$")
_LEADING_CJK_SECTION_FRAGMENT_RE = re.compile(r"^[、，,:：](?P<body>.+)$")
_LEADING_CJK_SECTION_LABEL_RE = re.compile(r"^(?P<ordinal>[一二三四五六七八九十]+)[、，,:：](?P<body>.+)$")
_ARABIC_LIST_LABEL_RE = re.compile(r"^(?P<ordinal>\d{1,2})[.、](?P<suffix>.+)$")
_MISSING_ARABIC_LIST_LABEL_RE = re.compile(r"^[.、](?P<suffix>.+)$")
_PARENTHETICAL_CJK_SECTION_RE = re.compile(r"^[（(][一二三四五六七八九十]+[）)]")
_CJK_SECTION_LABEL_RE = re.compile(r"^[一二三四五六七八九十]+[、.]")
_SUMMARY_ROW_RE = re.compile(r"合计|总计|小计|余额|净额")


@dataclass(frozen=True)
class _Token:
    text: str
    bbox: tuple[float, float, float, float]
    evidence_id: str
    confidence: float

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def h(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


@dataclass
class _Row:
    tokens: list[_Token]

    @property
    def cy(self) -> float:
        return statistics.median(token.cy for token in self.tokens)

    @property
    def h(self) -> float:
        return statistics.median(token.h for token in self.tokens)

    @property
    def text(self) -> str:
        return " ".join(token.text for token in sorted(self.tokens, key=lambda t: t.bbox[0])).strip()


def needs_high_precision_grid_review(blocks: list[Block]) -> bool:
    """Return whether a dense numeric grid benefits from a sharper review image."""

    tokens = [token for block in blocks if (token := _block_to_token(block)) is not None]
    if len(tokens) < 12:
        return False
    numeric_tokens = [token for token in tokens if _is_grid_numeric_token(token.text)]
    if len(numeric_tokens) < 6:
        return False
    numeric_rows = _cluster_rows(numeric_tokens)
    if len(numeric_rows) < 3:
        return False
    lane_width = max(12.0, statistics.median(token.h for token in numeric_tokens) * 1.5)
    lane_counts = Counter(round(token.cx / lane_width) for token in numeric_tokens)
    return sum(count >= 3 for count in lane_counts.values()) >= 1


def _recover_scanned_table_tokens(
    page_image: Any | None,
    tokens: list[_Token],
    *,
    page_width: float,
    page_height: float,
    page_number: int,
    table_index: int,
    grid_geometry: tuple[list[int], list[int], float, float] | None = None,
) -> tuple[list[_Token], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the shared structural and numeric token recovery sequence."""

    tokens, weak_text_events = _recover_weak_text_tokens(
        page_image,
        tokens,
        page_width=page_width,
        page_height=page_height,
        page_number=page_number,
        table_index=table_index,
    )
    tokens, numeric_events = _recover_numeric_ocr_tokens(
        page_image,
        tokens,
        page_width=page_width,
        page_height=page_height,
        page_number=page_number,
        table_index=table_index,
        grid_geometry=grid_geometry,
    )
    return tokens, weak_text_events, numeric_events


def reconstruct_scanned_bordered_tables(
    page_image: Any,
    blocks: list[Block],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    start_order: int = 0,
) -> list[Block]:
    """Reconstruct physical bordered tables without assigning business meaning."""
    if page_image is None or getattr(page_image, "size", 0) == 0:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    image = page_image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    height_px, width_px = gray.shape[:2]
    if width_px < 80 or height_px < 80 or page_width <= 0 or page_height <= 0:
        return []
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width_px // 28), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, height_px // 36)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8), iterations=1)

    contours, _hierarchy = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < width_px * 0.22 or height < height_px * 0.035:
            continue
        if width * height < width_px * height_px * 0.012:
            continue
        candidates.append((x, y, x + width, y + height))
    candidates = _merge_nested_table_candidates(candidates)

    sx = page_width / float(width_px)
    sy = page_height / float(height_px)
    tokens = [token for block in blocks if (token := _block_to_token(block)) is not None]
    out: list[Block] = []
    for table_index, pixel_bbox in enumerate(sorted(candidates, key=lambda value: (value[1], value[0]))):
        original_x0 = pixel_bbox[0]
        pixel_bbox = _extend_open_left_candidate(pixel_bbox, tokens, sx=sx, sy=sy, width_px=width_px)
        pixel_bbox = _extend_open_right_candidate(pixel_bbox, tokens, sx=sx, sy=sy, width_px=width_px)
        open_left_column = pixel_bbox[0] < original_x0 - 3
        x0, y0, x1, y1 = pixel_bbox
        x_lines = _projection_line_positions(vertical[y0:y1, x0:x1], axis=0, offset=x0)
        y_lines = _projection_line_positions(horizontal[y0:y1, original_x0:x1], axis=1, offset=y0)
        x_lines = _ensure_outer_lines(x_lines, x0, x1)
        y_lines = _ensure_outer_lines(y_lines, y0, y1)
        if len(x_lines) < 3 or len(y_lines) < 3:
            continue
        if len(x_lines) - 1 < 2 or len(y_lines) - 1 < 2:
            continue

        preserve_open_left_rows = bool(
            open_left_column
            and not _open_left_column_has_horizontal_dividers(
                horizontal,
                x0=x0,
                original_x0=original_x0,
                y_lines=y_lines,
            )
        )
        groups, merge_diagnostics = _merged_cell_groups(
            horizontal,
            vertical,
            x_lines,
            y_lines,
            preserve_left_column_rows=preserve_open_left_rows,
        )
        grid_rows = len(y_lines) - 1
        grid_cols = len(x_lines) - 1
        original_grid_rows = grid_rows
        original_grid_cols = grid_cols
        table_bbox_points = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        table_tokens = [token for token in tokens if _center_in_bbox(token.cx, token.cy, table_bbox_points)]
        table_tokens, weak_text_recovery_events, numeric_recovery_events = _recover_scanned_table_tokens(
            page_image,
            table_tokens,
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            table_index=table_index,
            grid_geometry=(x_lines, y_lines, sx, sy),
        )
        recovered_x_lines = _recover_missing_token_column_lines(
            table_tokens,
            vertical=vertical,
            x_lines=x_lines,
            y_lines=y_lines,
            sx=sx,
            sy=sy,
        )
        forced_vertical_lines = set(recovered_x_lines).difference(x_lines)
        x_lines = recovered_x_lines
        table_tokens = _split_tokens_at_supported_column_boundaries(
            table_tokens,
            x_lines=x_lines,
            sx=sx,
            trusted_boundaries=forced_vertical_lines,
        )
        table_tokens, single_cell_punctuation_events = _normalize_single_cell_numeric_punctuation(
            table_tokens,
            x_lines=x_lines,
            sx=sx,
            page_number=page_number,
            table_index=table_index,
        )
        numeric_recovery_events.extend(single_cell_punctuation_events)
        recovered_y_lines = _recover_missing_token_row_lines(
            table_tokens,
            x_lines=x_lines,
            y_lines=y_lines,
            sx=sx,
            sy=sy,
        )
        forced_horizontal_lines = set(recovered_y_lines).difference(y_lines)
        y_lines = recovered_y_lines
        grid_rows = len(y_lines) - 1
        grid_cols = len(x_lines) - 1
        if forced_vertical_lines or forced_horizontal_lines:
            groups, recovered_merge_diagnostics = _merged_cell_groups(
                horizontal,
                vertical,
                x_lines,
                y_lines,
                preserve_left_column_rows=preserve_open_left_rows,
                forced_vertical_lines=forced_vertical_lines,
                forced_horizontal_lines=forced_horizontal_lines,
            )
            recovered_merge_diagnostics["token_recovered_column_line_count"] = grid_cols - original_grid_cols
            recovered_merge_diagnostics["token_recovered_row_line_count"] = grid_rows - original_grid_rows
            merge_diagnostics = recovered_merge_diagnostics
        groups = _recover_sparse_vertical_label_groups(
            groups,
            table_tokens,
            x_lines=x_lines,
            y_lines=y_lines,
            sx=sx,
            sy=sy,
        )
        group_tokens: dict[int, list[_Token]] = {index: [] for index in range(len(groups))}
        numeric_columns_by_row: dict[int, set[int]] = {index: set() for index in range(grid_rows)}
        for token in table_tokens:
            row_index = _band_index(token.cy / sy, y_lines)
            col_index = _band_index(token.cx / sx, x_lines)
            if row_index is None or col_index is None:
                continue
            group_index = next(
                (idx for idx, cells in enumerate(groups) if (row_index, col_index) in cells),
                None,
            )
            if group_index is not None:
                group_tokens[group_index].append(token)
                if _is_grid_numeric_token(token.text):
                    numeric_columns_by_row[row_index].add(col_index)
        numeric_column_counts: dict[int, int] = {}
        for columns in numeric_columns_by_row.values():
            for column in columns:
                numeric_column_counts[column] = numeric_column_counts.get(column, 0) + 1
        stable_numeric_columns = {column for column, count in numeric_column_counts.items() if count >= 3}

        raw = [["" for _col in range(grid_cols)] for _row in range(grid_rows)]
        cell_bboxes: list[list[list[float] | None]] = [[None for _col in range(grid_cols)] for _row in range(grid_rows)]
        cell_evidence_ids: list[list[list[str]]] = [[[] for _col in range(grid_cols)] for _row in range(grid_rows)]
        cell_confidences: list[list[float | None]] = [[None for _col in range(grid_cols)] for _row in range(grid_rows)]
        cell_geometry_status = [["derived" for _col in range(grid_cols)] for _row in range(grid_rows)]
        cell_geometry_loss_reason: list[list[str | None]] = [
            ["covered_by_merged_cell" for _col in range(grid_cols)] for _row in range(grid_rows)
        ]
        cell_spans: list[dict[str, Any]] = []
        owned_ids: set[str] = set()
        token_split_vertical_merge_count = 0
        token_split_horizontal_merge_count = 0
        token_split_rectangular_merge_count = 0
        token_reanchored_vertical_merge_count = 0
        for group_index, cells in enumerate(groups):
            rows = sorted({cell[0] for cell in cells})
            cols = sorted({cell[1] for cell in cells})
            anchor_row, anchor_col = rows[0], cols[0]
            bbox = [
                round(x_lines[anchor_col] * sx, 4),
                round(y_lines[anchor_row] * sy, 4),
                round(x_lines[cols[-1] + 1] * sx, 4),
                round(y_lines[rows[-1] + 1] * sy, 4),
            ]
            assigned = _tokens_in_reading_order(group_tokens[group_index])
            row_buckets: dict[int, list[_Token]] = {row: [] for row in rows}
            for token in assigned:
                token_row = _band_index(token.cy / sy, y_lines)
                if token_row in row_buckets:
                    row_buckets[token_row].append(token)
            populated_buckets = [bucket for bucket in row_buckets.values() if bucket]
            populated_rows = [row for row, bucket in row_buckets.items() if bucket]
            numeric_rows_in_group = [
                row for row, bucket in row_buckets.items() if any(_looks_numeric(token.text) for token in bucket)
            ]
            typed_numeric_rows_in_group = [
                row
                for row, bucket in row_buckets.items()
                if any(_is_grid_numeric_token(token.text) for token in bucket)
            ]
            aligned_numeric_rows = [
                row for row in populated_rows if numeric_columns_by_row.get(row, set()).difference(cols)
            ]
            numeric_rows_aligned_elsewhere = set(numeric_rows_in_group).intersection(aligned_numeric_rows)
            column_buckets: dict[int, list[_Token]] = {col: [] for col in cols}
            cell_token_buckets: dict[tuple[int, int], list[_Token]] = {(row, col): [] for row in rows for col in cols}
            for token in assigned:
                token_col = _band_index(token.cx / sx, x_lines)
                if token_col in column_buckets:
                    column_buckets[token_col].append(token)
                token_row = _band_index(token.cy / sy, y_lines)
                if (token_row, token_col) in cell_token_buckets:
                    cell_token_buckets[(token_row, token_col)].append(token)
            populated_columns = {col for col, bucket in column_buckets.items() if bucket}
            nearby_numeric_columns = set().union(
                *(
                    numeric_columns_by_row[row]
                    for row in range(max(0, anchor_row - 2), min(grid_rows, anchor_row + 3))
                    if row != anchor_row
                )
            )
            numeric_like_columns = {
                col
                for col, bucket in column_buckets.items()
                if bucket
                and all(
                    _is_grid_numeric_token(token.text) or re.fullmatch(r"""[-—−一二工"']+""", token.text.strip())
                    for token in bucket
                )
            }
            supported_numeric_split = bool(
                numeric_like_columns == populated_columns
                and (
                    populated_columns <= nearby_numeric_columns.union(stable_numeric_columns)
                    or (len(populated_columns) > 2 and len(populated_columns.intersection(nearby_numeric_columns)) >= 3)
                )
            )
            ordered_columns = sorted(populated_columns)
            supported_label_numeric_split = bool(
                len(ordered_columns) >= 2
                and ordered_columns == list(range(ordered_columns[0], ordered_columns[-1] + 1))
                and numeric_like_columns == set(ordered_columns[1:])
                and set(ordered_columns[1:]) <= nearby_numeric_columns
                and ordered_columns[0] not in nearby_numeric_columns
            )
            ordinal_header_columns = {
                col for col, bucket in column_buckets.items() if _normalized_ordinal_header_text(bucket)
            }
            supported_label_ordinal_header_split = bool(
                len(ordered_columns) == 2
                and ordered_columns == list(range(ordered_columns[0], ordered_columns[-1] + 1))
                and ordinal_header_columns == {ordered_columns[1]}
                and ordered_columns[1] in nearby_numeric_columns
                and ordered_columns[0] not in nearby_numeric_columns
            )
            split_rectangular_merge = bool(
                len(rows) > 1 and len(cols) > 1 and _supports_rectangular_header_data_split(cell_token_buckets)
            )
            if split_rectangular_merge:
                token_split_rectangular_merge_count += 1
                for (row, col), bucket in cell_token_buckets.items():
                    if not bucket:
                        continue
                    bucket = _tokens_in_reading_order(bucket)
                    raw[row][col] = _join_cell_tokens(bucket)
                    cell_bboxes[row][col] = [
                        round(x_lines[col] * sx, 4),
                        round(y_lines[row] * sy, 4),
                        round(x_lines[col + 1] * sx, 4),
                        round(y_lines[row + 1] * sy, 4),
                    ]
                    cell_evidence_ids[row][col] = [token.evidence_id for token in bucket]
                    cell_confidences[row][col] = round(
                        sum(token.confidence * max(1, len(token.text)) for token in bucket)
                        / sum(max(1, len(token.text)) for token in bucket),
                        4,
                    )
                    cell_geometry_status[row][col] = "exact"
                    cell_geometry_loss_reason[row][col] = None
                    owned_ids.update(token.evidence_id for token in bucket)
                continue
            split_horizontal_merge = bool(
                len(rows) == 1
                and len(cols) > 1
                and len(populated_columns) >= 2
                and (supported_numeric_split or supported_label_numeric_split or supported_label_ordinal_header_split)
            )
            if split_horizontal_merge:
                token_split_horizontal_merge_count += 1
                for col, bucket in column_buckets.items():
                    if not bucket:
                        continue
                    bucket = _tokens_in_reading_order(bucket)
                    raw[anchor_row][col] = _normalized_ordinal_header_text(bucket) or _join_cell_tokens(bucket)
                    cell_bboxes[anchor_row][col] = [
                        round(x_lines[col] * sx, 4),
                        round(y_lines[anchor_row] * sy, 4),
                        round(x_lines[col + 1] * sx, 4),
                        round(y_lines[anchor_row + 1] * sy, 4),
                    ]
                    cell_evidence_ids[anchor_row][col] = [token.evidence_id for token in bucket]
                    cell_confidences[anchor_row][col] = round(
                        sum(token.confidence * max(1, len(token.text)) for token in bucket)
                        / sum(max(1, len(token.text)) for token in bucket),
                        4,
                    )
                    cell_geometry_status[anchor_row][col] = "exact"
                    cell_geometry_loss_reason[anchor_row][col] = None
                    owned_ids.update(token.evidence_id for token in bucket)
                continue
            reanchor_vertical_merge = bool(
                len(rows) > 1
                and len(cols) == 1
                and len(populated_buckets) == 1
                and populated_rows[0] != anchor_row
                and populated_rows[0] in aligned_numeric_rows
                and all(_is_grid_numeric_token(token.text) for token in populated_buckets[0])
            )
            if reanchor_vertical_merge:
                token_reanchored_vertical_merge_count += 1
                target_row = populated_rows[0]
                bucket = _tokens_in_reading_order(populated_buckets[0])
                raw[target_row][anchor_col] = _join_cell_tokens(bucket)
                cell_bboxes[target_row][anchor_col] = [
                    round(x_lines[anchor_col] * sx, 4),
                    round(y_lines[target_row] * sy, 4),
                    round(x_lines[anchor_col + 1] * sx, 4),
                    round(y_lines[target_row + 1] * sy, 4),
                ]
                cell_evidence_ids[target_row][anchor_col] = [token.evidence_id for token in bucket]
                cell_confidences[target_row][anchor_col] = round(
                    sum(token.confidence * max(1, len(token.text)) for token in bucket)
                    / sum(max(1, len(token.text)) for token in bucket),
                    4,
                )
                cell_geometry_status[target_row][anchor_col] = "exact"
                cell_geometry_loss_reason[target_row][anchor_col] = None
                owned_ids.update(token.evidence_id for token in bucket)
                continue
            split_vertical_merge = bool(
                len(rows) > 1
                and len(cols) == 1
                and len(populated_buckets) >= 2
                and (
                    len(numeric_rows_in_group) >= 2
                    or len(typed_numeric_rows_in_group) >= 2
                    or len(aligned_numeric_rows) >= 2
                    or bool(numeric_rows_aligned_elsewhere)
                    or _supports_standalone_label_row_split(row_buckets)
                    or _supports_header_data_label_row_split(row_buckets, aligned_numeric_rows)
                )
            )
            if split_vertical_merge:
                token_split_vertical_merge_count += 1
                for row, bucket in row_buckets.items():
                    if not bucket:
                        continue
                    bucket = _tokens_in_reading_order(bucket)
                    bucket_evidence_ids = [token.evidence_id for token in bucket]
                    bucket_confidence = sum(token.confidence * max(1, len(token.text)) for token in bucket) / sum(
                        max(1, len(token.text)) for token in bucket
                    )
                    raw[row][anchor_col] = _join_cell_tokens(bucket)
                    cell_bboxes[row][anchor_col] = [
                        round(x_lines[anchor_col] * sx, 4),
                        round(y_lines[row] * sy, 4),
                        round(x_lines[anchor_col + 1] * sx, 4),
                        round(y_lines[row + 1] * sy, 4),
                    ]
                    cell_evidence_ids[row][anchor_col] = bucket_evidence_ids
                    cell_confidences[row][anchor_col] = round(float(bucket_confidence), 4)
                    cell_geometry_status[row][anchor_col] = "exact"
                    cell_geometry_loss_reason[row][anchor_col] = None
                    owned_ids.update(bucket_evidence_ids)
                continue
            text = _join_cell_tokens(assigned)
            evidence_ids = [token.evidence_id for token in assigned]
            confidence = (
                sum(token.confidence * max(1, len(token.text)) for token in assigned)
                / sum(max(1, len(token.text)) for token in assigned)
                if assigned
                else None
            )
            raw[anchor_row][anchor_col] = text
            cell_bboxes[anchor_row][anchor_col] = bbox
            cell_evidence_ids[anchor_row][anchor_col] = evidence_ids
            cell_confidences[anchor_row][anchor_col] = round(float(confidence), 4) if confidence is not None else None
            cell_geometry_status[anchor_row][anchor_col] = "exact"
            cell_geometry_loss_reason[anchor_row][anchor_col] = None
            owned_ids.update(evidence_ids)
            if len(rows) > 1 or len(cols) > 1:
                cell_spans.append(
                    {
                        "row": anchor_row,
                        "col": anchor_col,
                        "row_span": len(rows),
                        "col_span": len(cols),
                        "bbox": bbox,
                        "evidence_ids": evidence_ids,
                    }
                )
        merge_diagnostics["token_split_vertical_merge_count"] = token_split_vertical_merge_count
        merge_diagnostics["token_split_horizontal_merge_count"] = token_split_horizontal_merge_count
        merge_diagnostics["token_split_rectangular_merge_count"] = token_split_rectangular_merge_count
        merge_diagnostics["token_reanchored_vertical_merge_count"] = token_reanchored_vertical_merge_count

        non_empty = sum(1 for row in raw for value in row if value.strip())
        if non_empty < 3:
            continue
        raw, ordinal_recovery_events = _recover_ordinal_cells(
            page_image,
            raw,
            cell_bboxes=cell_bboxes,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            cell_geometry_status=cell_geometry_status,
            cell_geometry_loss_reason=cell_geometry_loss_reason,
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            table_index=table_index,
        )
        raw, section_label_recovery_events = _recover_leading_cjk_section_labels(
            page_image,
            raw,
            cell_bboxes=cell_bboxes,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            table_index=table_index,
        )
        raw, marker_recovery_events = _recover_cjk_ordinal_markers(
            raw,
            cell_bboxes=cell_bboxes,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            page_number=page_number,
            table_index=table_index,
        )
        raw, arabic_marker_recovery_events = _recover_arabic_ordinal_markers(
            raw,
            cell_bboxes=cell_bboxes,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            page_number=page_number,
            table_index=table_index,
        )
        raw, placeholder_events = _recover_dash_placeholder_cells(
            page_image,
            raw,
            cell_bboxes=cell_bboxes,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            cell_geometry_status=cell_geometry_status,
            cell_geometry_loss_reason=cell_geometry_loss_reason,
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            table_index=table_index,
        )
        correction_mode = _ocr_correction_mode(blocks)
        correction_scope = _ocr_correction_scope(blocks)
        raw, correction_events, correction_processed_count = _correct_table_grid(
            raw,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            domain=None,
            first_column_role=None,
            mode=correction_mode,
            **correction_scope,
            page_number=page_number,
            table_index=table_index,
        )
        raw, cell_punctuation_events = _normalize_numeric_cell_punctuation(
            raw,
            cell_bboxes=cell_bboxes,
            cell_evidence_ids=cell_evidence_ids,
            cell_confidences=cell_confidences,
            column_bands=[
                {"x0": float(x_lines[column] * sx), "x1": float(x_lines[column + 1] * sx)}
                for column in range(len(x_lines) - 1)
            ],
            page_number=page_number,
            table_index=table_index,
        )
        correction_events = [
            *_source_correction_events(blocks, owned_ids),
            *numeric_recovery_events,
            *weak_text_recovery_events,
            *ordinal_recovery_events,
            *section_label_recovery_events,
            *marker_recovery_events,
            *arabic_marker_recovery_events,
            *placeholder_events,
            *cell_punctuation_events,
            *correction_events,
        ]
        h_strength = _line_projection_strength(horizontal, x0, y0, x1, y1, axis=1, positions=y_lines)
        v_strength = _line_projection_strength(vertical, x0, y0, x1, y1, axis=0, positions=x_lines)
        assignment_ratio = len(owned_ids) / max(
            1, sum(1 for token in tokens if _center_in_bbox(token.cx, token.cy, table_bbox_points))
        )
        geometry_confidence = round(
            max(0.0, min(1.0, 0.45 * h_strength + 0.45 * v_strength + 0.10 * assignment_ratio)), 4
        )
        bbox_points = tuple(round(value, 4) for value in table_bbox_points)
        out.append(
            Block(
                block_id=f"scanned_grid:p{page_number:04d}:{table_index:04d}",
                block_type="table",
                bbox=bbox_points,
                reading_order=start_order + table_index,
                page=page_number,
                raw_content=raw,
                attrs={
                    "extraction_layer": "scanned_image_line_grid",
                    "extraction_confidence": geometry_confidence,
                    "confidence": geometry_confidence,
                    "geometry": {
                        "geometry_source": "scanned_image_line_grid",
                        "geometry_confidence": geometry_confidence,
                        "coordinate_system": "pdf_points_top_left",
                        "cell_bboxes": cell_bboxes,
                        "cell_geometry_status": cell_geometry_status,
                        "cell_geometry_loss_reason": cell_geometry_loss_reason,
                        "cell_evidence_ids": cell_evidence_ids,
                        "cell_token_ids": cell_evidence_ids,
                        "cell_confidences": cell_confidences,
                        "cell_spans": cell_spans,
                        "row_bands": [
                            {
                                "index": index,
                                "y0": round(y_lines[index] * sy, 4),
                                "y1": round(y_lines[index + 1] * sy, 4),
                            }
                            for index in range(grid_rows)
                        ],
                        "col_bands": [
                            {
                                "index": index,
                                "x0": round(x_lines[index] * sx, 4),
                                "x1": round(x_lines[index + 1] * sx, 4),
                            }
                            for index in range(grid_cols)
                        ],
                        "horizontal_lines": [round(value * sy, 4) for value in y_lines],
                        "vertical_lines": [round(value * sx, 4) for value in x_lines],
                        "merge_diagnostics": merge_diagnostics,
                    },
                    "role": "physical_table",
                    "preserve_headers": False,
                    "source": "scanned_bordered_table_reconstructor",
                    "ocr_correction_mode": correction_mode,
                    "ocr_correction_processed_count": correction_processed_count,
                    **({"ocr_corrections": correction_events} if correction_events else {}),
                    "page_width": page_width,
                    "page_height": page_height,
                },
                evidence_ids=tuple(sorted(owned_ids)),
            )
        )
    return out


def _merge_nested_table_candidates(candidates: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ordered = sorted(candidates, key=lambda value: (-(value[2] - value[0]) * (value[3] - value[1]), value[1]))
    kept: list[tuple[int, int, int, int]] = []
    for candidate in ordered:
        if any(
            outer[0] <= candidate[0]
            and outer[1] <= candidate[1]
            and outer[2] >= candidate[2]
            and outer[3] >= candidate[3]
            for outer in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _extend_open_left_candidate(
    candidate: tuple[int, int, int, int],
    tokens: list[_Token],
    *,
    sx: float,
    sy: float,
    width_px: int,
) -> tuple[int, int, int, int]:
    """Include an unruled first label column adjacent to a detected numeric grid."""
    x0, y0, x1, y1 = candidate
    x0_points, y0_points, y1_points = x0 * sx, y0 * sy, y1 * sy
    max_extension = min(110.0, width_px * sx * 0.18)
    left_tokens = [
        token
        for token in tokens
        if x0_points - max_extension <= token.cx < x0_points and y0_points <= token.cy <= y1_points
    ]
    has_project_header = any(re.sub(r"\s+", "", token.text) in {"项", "目", "项目"} for token in left_tokens)
    row_centers = {int(round(token.cy / max(3.0, token.h))) for token in left_tokens}
    if not has_project_header or len(left_tokens) < 2 or len(row_centers) < 2:
        return candidate
    extended_x0 = max(0, int((min(token.bbox[0] for token in left_tokens) - 2.0) / sx))
    return (extended_x0, y0, x1, y1)


def _extend_open_right_candidate(
    candidate: tuple[int, int, int, int],
    tokens: list[_Token],
    *,
    sx: float,
    sy: float,
    width_px: int,
) -> tuple[int, int, int, int]:
    """Include a row-aligned numeric column beside an open right table edge."""

    x0, y0, x1, y1 = candidate
    x1_points, y0_points, y1_points = x1 * sx, y0 * sy, y1 * sy
    max_extension = min(110.0, width_px * sx * 0.18)
    right_tokens = [
        token
        for token in tokens
        if x1_points < token.cx <= x1_points + max_extension and y0_points <= token.cy <= y1_points
    ]
    numeric_tokens = [token for token in right_tokens if _is_grid_numeric_token(token.text)]
    row_centers = {int(round(token.cy / max(3.0, token.h))) for token in numeric_tokens}
    if len(numeric_tokens) < 3 or len(row_centers) < 3:
        return candidate
    centers = [token.cx for token in numeric_tokens]
    if max(centers) - min(centers) > max(45.0, max_extension * 0.65):
        return candidate
    extended_x1 = min(width_px, int((max(token.bbox[2] for token in right_tokens) + 2.0) / sx))
    return (x0, y0, extended_x1, y1) if extended_x1 - x1 >= 8 else candidate


def _open_left_column_has_horizontal_dividers(
    horizontal: Any,
    *,
    x0: int,
    original_x0: int,
    y_lines: list[int],
) -> bool:
    """Return whether the restored left column contains physical row-group rules."""

    if original_x0 - x0 < 8:
        return False
    pad = max(1, int((original_x0 - x0) * 0.08))
    left, right = x0 + pad, max(x0 + pad + 1, original_x0 - pad)
    for y in y_lines[1:-1]:
        segment = horizontal[max(0, y - 1) : y + 2, left:right]
        if segment.size and float((segment > 0).mean()) >= 0.30:
            return True
    return False


def _split_tokens_at_supported_column_boundaries(
    tokens: list[_Token],
    *,
    x_lines: list[int],
    sx: float,
    trusted_boundaries: set[int] | None = None,
) -> list[_Token]:
    """Split OCR text only where stable neighboring column types support it."""
    boundaries = [value * sx for value in x_lines[1:-1]]
    profiles: dict[int, list[bool]] = {}
    for token in tokens:
        compact = re.sub(r"\s+", "", token.text).replace("，", ",")
        crosses_boundary = any(token.bbox[0] + 2 < boundary < token.bbox[2] - 2 for boundary in boundaries)
        if crosses_boundary and not _EXACT_DECIMAL_RE.fullmatch(compact):
            continue
        column = _band_index(token.cx / sx, x_lines)
        if column is not None:
            profiles.setdefault(column, []).append(bool(_GRID_NUMERIC_RE.fullmatch(compact)))

    output: list[_Token] = []
    for token in tokens:
        numeric_label_match = _LEADING_NUMBER_AND_LABEL_RE.fullmatch(token.text)
        label_ordinal_match = _LABEL_AND_ORDINAL_HEADER_RE.fullmatch(token.text)
        crossed = [
            (index, boundary)
            for index, boundary in enumerate(boundaries, start=1)
            if token.bbox[0] + 2 < boundary < token.bbox[2] - 2
        ]
        amount_pair = _split_concatenated_decimal_values(token.text)
        if len(crossed) != 1:
            output.append(token)
            continue
        right_column, boundary = crossed[0]
        left_profile = profiles.get(right_column - 1, [])
        right_profile = profiles.get(right_column, [])
        numeric_pair_supported = bool(
            amount_pair
            and (
                (trusted_boundaries and any(abs(boundary - value * sx) <= 2 for value in trusted_boundaries))
                or (sum(left_profile) >= 2 and sum(right_profile) >= 2)
            )
        )
        numeric_label_supported = bool(
            numeric_label_match
            and len(left_profile) >= 3
            and sum(left_profile) / len(left_profile) >= 0.75
            and len(right_profile) >= 3
            and sum(not value for value in right_profile) / len(right_profile) >= 0.75
        )
        label_ordinal_supported = bool(
            label_ordinal_match
            and len(left_profile) >= 3
            and sum(not value for value in left_profile) / len(left_profile) >= 0.75
            and len(right_profile) >= 3
            and sum(right_profile) / len(right_profile) >= 0.75
        )
        if not numeric_pair_supported and not numeric_label_supported and not label_ordinal_supported:
            output.append(token)
            continue
        if numeric_pair_supported and amount_pair is not None:
            prefix, suffix = amount_pair
        else:
            match = numeric_label_match or label_ordinal_match
            if match is None:
                output.append(token)
                continue
            prefix, suffix = match.groups()
            if label_ordinal_supported and suffix == "次行":
                suffix = "行次"
        output.extend(
            (
                _Token(
                    prefix.strip(),
                    (token.bbox[0], token.bbox[1], boundary, token.bbox[3]),
                    token.evidence_id,
                    token.confidence,
                ),
                _Token(
                    suffix.strip(),
                    (boundary, token.bbox[1], token.bbox[2], token.bbox[3]),
                    token.evidence_id,
                    token.confidence,
                ),
            )
        )
    return output


def _recover_missing_token_column_lines(
    tokens: list[_Token],
    *,
    vertical: Any,
    x_lines: list[int],
    y_lines: list[int],
    sx: float,
    sy: float,
) -> list[int]:
    """Restore omitted column rules from repeated token lanes or a local rule segment."""

    if len(x_lines) < 3 or len(y_lines) < 3:
        return x_lines
    candidates: dict[int, list[float]] = {}
    joined_candidates: dict[int, list[float]] = {}
    local_rule_candidates: dict[int, list[float]] = {}
    tokens_by_row: dict[int, list[_Token]] = {}
    for token in tokens:
        row = _band_index(token.cy / sy, y_lines)
        if row is not None:
            tokens_by_row.setdefault(row, []).append(token)

    for row_tokens in tokens_by_row.values():
        ordered = sorted(row_tokens, key=lambda token: token.cx)
        for left, right in zip(ordered, ordered[1:]):
            band = _band_index(left.cx / sx, x_lines)
            if band is None or _band_index(right.cx / sx, x_lines) != band:
                continue
            if not _is_grid_numeric_token(right.text):
                continue
            left_numeric = _is_grid_numeric_token(left.text)
            left_text = re.sub(r"\s+", "", left.text)
            if not left_numeric and (len(left_text) > 24 or not re.search(r"[A-Za-z\u3400-\u9fff]", left_text)):
                continue
            if right.cx - left.cx < max(18.0, (x_lines[band + 1] - x_lines[band]) * sx * 0.18):
                continue
            gap_midpoint = (left.bbox[2] + right.bbox[0]) / 2.0
            if left.bbox[2] >= right.bbox[0]:
                gap_midpoint = (left.cx + right.cx) / 2.0
            candidates.setdefault(band, []).append(gap_midpoint / sx)

    for token in tokens:
        amount_pair = _split_concatenated_decimal_values(token.text)
        band = _band_index(token.cx / sx, x_lines)
        if amount_pair is None or band is None:
            continue
        compact = _normalize_recovered_amount(token.text)
        estimated = token.bbox[0] + (token.bbox[2] - token.bbox[0]) * len(amount_pair[0]) / len(compact)
        joined_candidates.setdefault(band, []).append(estimated / sx)

        row = _band_index(token.cy / sy, y_lines)
        if row is None:
            continue
        pixel_left = max(x_lines[band] + 4, int(token.bbox[0] / sx) + 2)
        pixel_right = min(x_lines[band + 1] - 4, int(token.bbox[2] / sx) - 2)
        pixel_top = max(y_lines[row] + 2, int(token.bbox[1] / sy) - 2)
        pixel_bottom = min(y_lines[row + 1] - 2, int(token.bbox[3] / sy) + 2)
        if pixel_right <= pixel_left or pixel_bottom <= pixel_top:
            continue
        projection = (vertical[pixel_top:pixel_bottom, pixel_left:pixel_right] > 0).mean(axis=0)
        indices = [pixel_left + index for index, strength in enumerate(projection) if float(strength) >= 0.35]
        groups: list[list[int]] = []
        for index in indices:
            if not groups or index - groups[-1][-1] > 3:
                groups.append([index])
            else:
                groups[-1].append(index)
        centers = [sum(group) / len(group) for group in groups]
        centers = [center for center in centers if all(abs(center - line) >= 6 for line in x_lines)]
        if len(centers) == 1:
            local_rule_candidates.setdefault(band, []).append(centers[0])

    insertions: list[int] = []
    for band in range(len(x_lines) - 1):
        width = x_lines[band + 1] - x_lines[band]
        tolerance = max(4.0, width * 0.08)
        supported = _stable_position(candidates.get(band, []), minimum=3, tolerance=tolerance)
        if supported is None:
            supported = _stable_position(joined_candidates.get(band, []), minimum=3, tolerance=tolerance)
        if supported is None:
            supported = _stable_position(local_rule_candidates.get(band, []), minimum=1, tolerance=tolerance)
        if supported is None:
            continue
        boundary = int(round(supported))
        if boundary - x_lines[band] >= 12 and x_lines[band + 1] - boundary >= 12:
            insertions.append(boundary)
    return sorted(set(x_lines).union(insertions))


def _stable_position(values: list[float], *, minimum: int, tolerance: float) -> float | None:
    """Return the median of one compact position cluster."""

    if len(values) < minimum:
        return None
    ordered = sorted(values)
    clusters: list[list[float]] = []
    for value in ordered:
        if not clusters or value - statistics.median(clusters[-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    eligible = [cluster for cluster in clusters if len(cluster) >= minimum]
    if len(eligible) != 1:
        return None
    return statistics.median(eligible[0])


def _recover_missing_token_row_lines(
    tokens: list[_Token],
    *,
    x_lines: list[int],
    y_lines: list[int],
    sx: float,
    sy: float,
) -> list[int]:
    """Restore row rules supported by separate label and numeric token baselines."""

    numeric_counts: dict[int, int] = {}
    for token in tokens:
        column = _band_index(token.cx / sx, x_lines)
        if column is not None and _is_grid_numeric_token(token.text):
            numeric_counts[column] = numeric_counts.get(column, 0) + 1
    stable_numeric_columns = {column for column, count in numeric_counts.items() if count >= 3}
    recovered = list(y_lines)
    insertions: list[int] = []
    for upper, lower in zip(y_lines, y_lines[1:]):
        band_tokens = [token for token in tokens if upper < token.cy / sy < lower]
        if len(band_tokens) < 2:
            continue
        median_height = statistics.median(max(1.0, token.h) for token in band_tokens)
        ordered = sorted(band_tokens, key=lambda token: token.cy)
        clusters: list[list[_Token]] = [[ordered[0]]]
        for token in ordered[1:]:
            previous_center = statistics.median(item.cy for item in clusters[-1])
            if token.cy - previous_center > max(4.0, median_height * 0.60):
                clusters.append([token])
            else:
                clusters[-1].append(token)
        if len(clusters) < 2:
            continue
        inferred_ordered_boundaries = _inferred_ordered_label_boundaries(
            clusters,
            tokens=tokens,
            upper=upper,
            x_lines=x_lines,
            sx=sx,
            sy=sy,
        )
        for pair_index, (first, second) in enumerate(zip(clusters, clusters[1:])):
            first_columns = {_band_index(token.cx / sx, x_lines) for token in first}
            second_columns = {_band_index(token.cx / sx, x_lines) for token in second}
            if 0 not in first_columns or 0 not in second_columns:
                continue
            first_numeric = {
                column
                for token in first
                if (column := _band_index(token.cx / sx, x_lines)) in stable_numeric_columns
                and _is_grid_numeric_token(token.text)
            }
            second_numeric = {
                column
                for token in second
                if (column := _band_index(token.cx / sx, x_lines)) in stable_numeric_columns
                and _is_grid_numeric_token(token.text)
            }
            header_then_data = _is_semantic_header_cluster(first) and bool(second_numeric)
            data_then_standalone_label = bool(first_numeric) and _cluster_has_standalone_row_label(
                second,
                x_lines,
                sx,
            )
            standalone_label_pair = _cluster_has_standalone_row_label(
                first,
                x_lines,
                sx,
            ) and _cluster_has_standalone_row_label(
                second,
                x_lines,
                sx,
            )
            inferred_ordered_label_pair = pair_index in inferred_ordered_boundaries
            if not (
                (first_numeric and second_numeric)
                or header_then_data
                or data_then_standalone_label
                or standalone_label_pair
                or inferred_ordered_label_pair
            ):
                continue
            first_bottom = max(token.bbox[3] for token in first)
            second_top = min(token.bbox[1] for token in second)
            boundary = int(round(((first_bottom + second_top) / 2.0) / sy))
            if upper + 2 < boundary < lower - 2:
                insertions.append(boundary)
    for boundary in sorted(set(insertions)):
        if all(abs(boundary - existing) > 2 for existing in recovered):
            recovered.append(boundary)
    return sorted(recovered)


def _is_semantic_header_cluster(tokens: list[_Token]) -> bool:
    """Return whether a baseline contains several independent text headers."""

    labels = [re.sub(r"\s+", "", token.text) for token in tokens if not _is_grid_numeric_token(token.text)]
    return len([label for label in labels if len(label) >= 2]) >= 3


_STANDALONE_ROW_RE = re.compile(
    r"^(?:(?:加|减)[:：]|其他$|合计$|[一二三四五六七八九十]+[、.]|"
    r"[（(][一二三四五六七八九十]+[）)]|\d+[.、]|\.[\u3400-\u9fff])"
)


def _cluster_first_column_label(tokens: list[_Token], x_lines: list[int], sx: float) -> str:
    labels = [token.text for token in tokens if _band_index(token.cx / sx, x_lines) == 0]
    return re.sub(r"\s+", "", "".join(labels))


def _inferred_ordered_label_boundaries(
    clusters: list[list[_Token]],
    *,
    tokens: list[_Token],
    upper: int,
    x_lines: list[int],
    sx: float,
    sy: float,
) -> set[int]:
    """Recover missing rules inside an OCR-damaged numbered-list run.

    A run is accepted only when a preceding visible ordinal and a later visible
    ordinal account exactly for the intervening aligned label baselines. The
    rule restores row boundaries but does not invent the missing ordinal text.
    """

    labels = [_cluster_first_column_label(cluster, x_lines, sx) for cluster in clusters]
    if len(labels) < 3 or any(not label for label in labels):
        return set()
    explicit = next(
        (
            (index, int(match.group(1)))
            for index, label in enumerate(labels)
            if (match := re.match(r"^(\d{1,2})[.、]", label)) is not None
        ),
        None,
    )
    if explicit is None or explicit[0] < 2:
        return set()
    explicit_index, explicit_ordinal = explicit
    previous_candidates: list[tuple[float, int]] = []
    for token in tokens:
        if _band_index(token.cx / sx, x_lines) != 0 or token.cy / sy >= upper:
            continue
        match = re.match(r"^(\d{1,2})[.、]", re.sub(r"\s+", "", token.text))
        if match is not None:
            previous_candidates.append((token.cy, int(match.group(1))))
    if not previous_candidates:
        return set()
    previous_ordinal = max(previous_candidates, key=lambda item: item[0])[1]
    if previous_ordinal + explicit_index + 1 != explicit_ordinal:
        return set()
    inferred_prefix = clusters[:explicit_index]
    if any(any(_is_grid_numeric_token(token.text) for token in cluster) for cluster in inferred_prefix):
        return set()
    prefix = clusters[: explicit_index + 1]
    left_edges = [min(token.bbox[0] for token in cluster) for cluster in prefix]
    median_height = statistics.median(token.h for cluster in prefix for token in cluster)
    if max(left_edges) - min(left_edges) > max(8.0, median_height * 0.8):
        return set()
    return set(range(explicit_index))


def _cluster_has_standalone_row_label(tokens: list[_Token], x_lines: list[int], sx: float) -> bool:
    label = _cluster_first_column_label(tokens, x_lines, sx)
    return bool(label and _STANDALONE_ROW_RE.search(label))


def _supports_standalone_label_row_split(row_buckets: dict[int, list[_Token]]) -> bool:
    """Split a vertical merge only when each populated lane is a standalone list row."""

    populated = [bucket for bucket in row_buckets.values() if bucket]
    if len(populated) < 2:
        return False
    return all(
        not any(_is_grid_numeric_token(token.text) for token in bucket)
        and bool(_STANDALONE_ROW_RE.search(re.sub(r"\s+", "", _join_cell_tokens(bucket))))
        for bucket in populated
    )


def _supports_header_data_label_row_split(
    row_buckets: dict[int, list[_Token]],
    aligned_numeric_rows: list[int],
) -> bool:
    """Separate a vertically merged label header from a numeric data row."""

    populated = [(row, bucket) for row, bucket in sorted(row_buckets.items()) if bucket]
    if len(populated) < 2:
        return False
    first_row, first_bucket = populated[0]
    first_label = re.sub(r"\s+", "", _join_cell_tokens(first_bucket))
    if first_row >= populated[-1][0] or not first_label or len(first_label) > 8 or _looks_numeric(first_label):
        return False
    return any(row in aligned_numeric_rows for row, _bucket in populated[1:])


def _supports_rectangular_header_data_split(
    cell_buckets: dict[tuple[int, int], list[_Token]],
) -> bool:
    """Split a missing-rule rectangle when it contains header cells above numeric cells."""

    occupied = {cell: bucket for cell, bucket in cell_buckets.items() if bucket}
    if len(occupied) < 4:
        return False
    numeric_cells = {
        cell for cell, bucket in occupied.items() if any(_is_grid_numeric_token(token.text) for token in bucket)
    }
    if len(numeric_cells) < 2 or len({column for _row, column in numeric_cells}) < 2:
        return False
    first_numeric_row = min(row for row, _column in numeric_cells)
    header_cells = {
        cell
        for cell, bucket in occupied.items()
        if cell[0] < first_numeric_row and all(not _is_grid_numeric_token(token.text) for token in bucket)
    }
    return len(header_cells) >= 2 and len({column for _row, column in header_cells}) >= 2


def _split_concatenated_decimal_values(value: str) -> tuple[str, str] | None:
    """Return one unambiguous pair of exact decimals from a joined OCR token."""

    compact = _normalize_recovered_amount(value)
    candidates = [
        (compact[:index], compact[index:])
        for index in range(1, len(compact))
        if _EXACT_DECIMAL_RE.fullmatch(compact[:index]) and _EXACT_DECIMAL_RE.fullmatch(compact[index:])
    ]
    return candidates[0] if len(candidates) == 1 else None


def _recover_numeric_ocr_tokens(
    page_image: Any,
    tokens: list[_Token],
    *,
    page_width: float,
    page_height: float,
    page_number: int,
    table_index: int,
    max_repairs: int = 32,
    grid_geometry: tuple[list[int], list[int], float, float] | None = None,
) -> tuple[list[_Token], list[dict[str, Any]]]:
    """Reinspect suspicious decimal tokens using crop-local OCR consensus."""

    tokens, spacing_events = _normalize_numeric_token_spacing(
        tokens,
        page_number=page_number,
        table_index=table_index,
    )
    numeric_count = sum(bool(_NUMBER_RE.search(token.text.replace("，", ","))) for token in tokens)
    if numeric_count < 3:
        return tokens, spacing_events

    from docmirror.ocr.micro_grid.cell_recognition import recognize_micro_cell_from_image

    review_candidates: list[tuple[tuple[int, float, int], int]] = []
    row_texts_by_index: dict[int, str] = {}
    for index, token in enumerate(tokens):
        row_text = _token_row_text(token, tokens)
        if not _should_reinspect_numeric_token(token, row_text):
            continue
        row_texts_by_index[index] = row_text
        review_candidates.append((_numeric_review_priority(token, row_text, index), index))
    selected_indexes = {index for _priority, index in sorted(review_candidates, key=lambda item: item[0])[:max_repairs]}

    out: list[_Token] = []
    events: list[dict[str, Any]] = list(spacing_events)
    for index, token in enumerate(tokens):
        if index not in selected_indexes:
            out.append(token)
            continue
        row_text = row_texts_by_index[index]
        recognition = recognize_micro_cell_from_image(
            page_image,
            token.bbox,
            page_width=page_width,
            page_height=page_height,
            allowed_charset="0123456789,.-()−",
            min_confidence=0.5,
        )
        candidate = _normalize_recovered_amount(recognition.text)
        original = _normalize_recovered_amount(token.text)
        accepted = _accept_numeric_recovery(
            original,
            candidate,
            original_confidence=token.confidence,
            recognition_confidence=recognition.confidence,
            consensus_count=int(recognition.audit.get("consensus_count") or 0),
            summary_row=bool(_SUMMARY_ROW_RE.search(row_text)),
        )
        reason_code = "numeric_cell_crop_consensus"
        if not accepted and (
            _decimal_value_is_visibly_incomplete(original)
            or _decimal_value_has_invalid_punctuation(original)
            or _looks_like_zero_digit_confusion(original)
        ):
            cell_bbox = _containing_grid_cell_bbox(token, grid_geometry)
            if cell_bbox is not None:
                cell_recognition = recognize_micro_cell_from_image(
                    page_image,
                    cell_bbox,
                    page_width=page_width,
                    page_height=page_height,
                    allowed_charset="0123456789,.-()−",
                    min_confidence=0.5,
                )
                recovered = _reconstruct_decimal_from_cell_votes(original, cell_recognition.audit)
                reason_code = "numeric_cell_digit_lattice_consensus"
                if recovered is None:
                    recovered = _reconstruct_zero_from_cell_votes(original, cell_recognition.audit)
                    reason_code = "numeric_zero_glyph_consensus"
                if recovered is not None:
                    from docmirror.ocr.micro_grid.cell_recognition import CellRecognition

                    candidate, recovered_confidence, recovered_votes = recovered
                    recognition = CellRecognition(
                        text=candidate,
                        confidence=recovered_confidence,
                        source=cell_recognition.source,
                        raw_text=cell_recognition.raw_text,
                        audit={**cell_recognition.audit, "consensus_count": recovered_votes},
                    )
                    accepted = True
        if not accepted:
            out.append(token)
            continue
        out.append(
            _Token(
                candidate,
                token.bbox,
                token.evidence_id,
                float(recognition.confidence),
            )
        )
        events.append(
            {
                "action": "corrected",
                "input_text": token.text,
                "output_text": candidate,
                "confidence": round(float(recognition.confidence), 4),
                "reason_codes": [reason_code],
                "source_ref": token.evidence_id,
                "audit": recognition.to_dict(),
                "target": {
                    "kind": "ocr_token",
                    "page": page_number,
                    "table": table_index,
                    "bbox": list(token.bbox),
                },
            }
        )
    return out, events


def _normalize_numeric_token_spacing(
    tokens: list[_Token],
    *,
    page_number: int,
    table_index: int,
) -> tuple[list[_Token], list[dict[str, Any]]]:
    """Normalize OCR-only spacing and an unmatched trailing bracket in decimal tokens."""

    out: list[_Token] = []
    events: list[dict[str, Any]] = []
    for token in tokens:
        compact = _normalize_recovered_amount(token.text)
        reason_code = "numeric_spacing_normalized"
        punctuation_repair = (
            None
            if _split_concatenated_decimal_values(compact) is not None
            else _repair_unambiguous_decimal_punctuation(compact)
        )
        if punctuation_repair is not None:
            compact = punctuation_repair
            reason_code = "numeric_punctuation_structure"
        if compact.endswith("]") and _EXACT_DECIMAL_RE.fullmatch(compact[:-1]):
            compact = compact[:-1]
            reason_code = "numeric_unmatched_trailing_bracket_removed"
        supported = bool(
            compact != token.text and (_EXACT_DECIMAL_RE.fullmatch(compact) or _GRID_ORDINAL_RE.fullmatch(compact))
        )
        if not supported:
            out.append(token)
            continue
        out.append(_Token(compact, token.bbox, token.evidence_id, token.confidence))
        events.append(
            {
                "action": "corrected",
                "input_text": token.text,
                "output_text": compact,
                "confidence": round(float(token.confidence), 4),
                "reason_codes": [reason_code],
                "source_ref": token.evidence_id,
                "target": {
                    "kind": "ocr_token",
                    "page": page_number,
                    "table": table_index,
                    "bbox": list(token.bbox),
                },
            }
        )
    return out, events


def _token_row_text(target: _Token, tokens: list[_Token]) -> str:
    tolerance = max(target.h * 0.8, 4.0)
    row = [token for token in tokens if abs(token.cy - target.cy) <= max(tolerance, token.h * 0.8)]
    return " ".join(token.text for token in sorted(row, key=lambda item: item.bbox[0]))


def _should_reinspect_numeric_token(token: _Token, row_text: str) -> bool:
    text = _normalize_recovered_amount(token.text)
    if not text or not any(character.isdigit() for character in text):
        return False
    if not re.fullmatch(r"[-()\[\]\d,.]+", text):
        return False
    visibly_incomplete = (
        _decimal_value_is_visibly_incomplete(text)
        or _decimal_value_has_invalid_punctuation(text)
        or "]" in text
        or " " in token.text
    )
    low_confidence = token.confidence < 0.9
    summary_row = bool(_SUMMARY_ROW_RE.search(row_text))
    return visibly_incomplete or low_confidence or summary_row


def _numeric_review_priority(token: _Token, row_text: str, index: int) -> tuple[int, float, int]:
    text = _normalize_recovered_amount(token.text)
    visibly_incomplete = (
        _decimal_value_is_visibly_incomplete(text)
        or _decimal_value_has_invalid_punctuation(text)
        or "]" in text
        or " " in token.text
    )
    low_confidence = token.confidence < 0.9
    summary_row = bool(_SUMMARY_ROW_RE.search(row_text))
    if low_confidence and _looks_like_zero_digit_confusion(text):
        tier = 0
    elif summary_row and low_confidence:
        tier = 0
    elif visibly_incomplete:
        tier = 1
    elif summary_row:
        tier = 2
    else:
        tier = 3
    return tier, token.confidence, index


def _looks_like_zero_digit_confusion(value: str) -> bool:
    """Return whether one low-order digit may be a misread zero glyph."""

    text = _normalize_recovered_amount(value).lstrip("+-")
    return bool(re.fullmatch(r"0\.0[1-9]", text))


def _recover_ordinal_cells(
    page_image: Any,
    raw: list[list[str]],
    *,
    cell_bboxes: list[list[list[float] | None]],
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    cell_geometry_status: list[list[str]],
    cell_geometry_loss_reason: list[list[str | None]],
    page_width: float,
    page_height: float,
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Recover a missing line ordinal only when crop OCR confirms its sequence."""

    if len(raw) < 4:
        return raw, []
    header_rows = raw[: min(5, len(raw))]
    ordinal_columns = {
        column
        for row in header_rows
        for column, value in enumerate(row)
        if re.search(r"行次|栏次", re.sub(r"\s+", "", str(value or "")))
    }
    if not ordinal_columns:
        return raw, []

    from docmirror.ocr.micro_grid.cell_recognition import recognize_micro_cell_from_image

    out = [list(row) for row in raw]
    events: list[dict[str, Any]] = []
    for column in sorted(ordinal_columns):
        data_rows = [
            row_index
            for row_index, row in enumerate(out)
            if row_index >= 1
            and column < len(row)
            and any(index != column and _looks_numeric(value) for index, value in enumerate(row))
        ]
        for position, row_index in enumerate(data_rows):
            original = str(out[row_index][column] or "").strip()
            expected, support_count = _expected_ordinal_from_lattice(out, data_rows, position, column)
            if expected is None:
                expected = _expected_missing_ordinal(out, data_rows, position, column)
                support_count = 2 if expected is not None else 0
            if expected is None:
                continue
            leading = re.match(r"^(\d{1,3})", original)
            if leading is not None and int(leading.group(1)) == expected:
                continue
            bbox = cell_bboxes[row_index][column] if row_index < len(cell_bboxes) else None
            if bbox is None:
                continue
            candidate = re.sub(r"^\d{1,3}", str(expected), original) if leading is not None else str(expected)
            recognition = None
            if leading is None:
                recognition = recognize_micro_cell_from_image(
                    page_image,
                    tuple(float(value) for value in bbox),
                    page_width=page_width,
                    page_height=page_height,
                    allowed_charset="0123456789",
                    max_chars=3,
                    isolate_glyph=True,
                    min_confidence=0.5,
                )
            crop_confirmed = bool(
                recognition is not None
                and str(recognition.text or "").strip() == str(expected)
                and int(recognition.audit.get("consensus_count") or 0) >= 2
                and recognition.confidence >= 0.75
            )
            lattice_confirmed = support_count >= 4 and support_count / max(len(data_rows), 1) >= 0.75
            if not crop_confirmed and not lattice_confirmed:
                continue
            confidence = float(recognition.confidence) if crop_confirmed and recognition is not None else 0.9
            reason_code = "ordinal_cell_crop_consensus" if crop_confirmed else "ordinal_sequence_lattice"
            out[row_index][column] = candidate
            cell_confidences[row_index][column] = round(confidence, 4)
            cell_geometry_status[row_index][column] = "exact"
            cell_geometry_loss_reason[row_index][column] = None
            source_ref = f"table:p{page_number}:t{table_index}:r{row_index}:c{column}"
            events.append(
                {
                    "action": "corrected",
                    "input_text": original,
                    "output_text": candidate,
                    "confidence": round(confidence, 4),
                    "reason_codes": [reason_code],
                    "source_ref": source_ref,
                    "audit": (
                        recognition.to_dict()
                        if recognition is not None
                        else {"sequence_support_count": support_count, "data_row_count": len(data_rows)}
                    ),
                    "target": {
                        "kind": "table_cell",
                        "page": page_number,
                        "table": table_index,
                        "row": row_index,
                        "column": column,
                        "bbox": list(bbox),
                    },
                }
            )
            if not cell_evidence_ids[row_index][column]:
                cell_evidence_ids[row_index][column] = [source_ref]
    return out, events


def _expected_ordinal_from_lattice(
    raw: list[list[str]],
    data_rows: list[int],
    position: int,
    column: int,
) -> tuple[int | None, int]:
    bases: list[int] = []
    for anchor_position, row_index in enumerate(data_rows):
        value = str(raw[row_index][column] if column < len(raw[row_index]) else "").strip()
        match = re.match(r"^(\d{1,3})", value)
        if match:
            bases.append(int(match.group(1)) - anchor_position)
    if not bases:
        return None, 0
    base, support_count = Counter(bases).most_common(1)[0]
    expected = base + position
    return (expected, support_count) if 1 <= expected <= 200 else (None, support_count)


def _recover_leading_cjk_section_labels(
    page_image: Any,
    raw: list[list[str]],
    *,
    cell_bboxes: list[list[list[float] | None]],
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    page_width: float,
    page_height: float,
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Restore a missing leading CJK ordinal from crop-level text consensus.

    The repair is limited to text-only table rows whose OCR text begins with
    punctuation. The recovered candidate must keep
    the complete observed suffix, so the function never infers an ordinal from
    row position or a document-specific title dictionary.
    """

    from docmirror.ocr.micro_grid.cell_recognition import recognize_micro_cell_from_image

    out = [list(row) for row in raw]
    events: list[dict[str, Any]] = []
    ordinal_charset = "".join(_CJK_ORDINALS)
    punctuation = "、，,:：。.;；（）()-—"
    for row_index, row in enumerate(out):
        if not row or any(_looks_numeric(value) for value in row[1:]):
            continue
        original = str(row[0] or "").strip()
        fragment = _LEADING_CJK_SECTION_FRAGMENT_RE.fullmatch(original)
        if fragment is None or len(re.findall(r"[\u3400-\u9fff]", fragment.group("body"))) < 4:
            continue
        bbox = cell_bboxes[row_index][0] if row_index < len(cell_bboxes) and cell_bboxes[row_index] else None
        if bbox is None:
            continue
        allowed_charset = set(original + ordinal_charset + punctuation)
        recognition = recognize_micro_cell_from_image(
            page_image,
            tuple(float(value) for value in bbox),
            page_width=page_width,
            page_height=page_height,
            allowed_charset=allowed_charset,
            min_confidence=0.5,
        )
        observed_body = _comparable_section_label(fragment.group("body"))
        consensus = _leading_section_ordinal_consensus(recognition, observed_body)
        if consensus is None:
            continue
        ordinal, confidence, support_count = consensus
        candidate = f"{ordinal}{original}"

        out[row_index][0] = candidate
        cell_confidences[row_index][0] = round(confidence, 4)
        evidence_ids = cell_evidence_ids[row_index][0]
        source_ref = evidence_ids[0] if evidence_ids else f"table:p{page_number}:t{table_index}:r{row_index}:c0"
        if not evidence_ids:
            evidence_ids.append(source_ref)
        events.append(
            {
                "action": "corrected",
                "input_text": original,
                "output_text": candidate,
                "confidence": round(confidence, 4),
                "reason_codes": ["leading_cjk_ordinal_label_crop_consensus"],
                "source_ref": source_ref,
                "audit": {
                    "recognition": recognition.to_dict(),
                    "ordinal_support_count": support_count,
                },
                "target": {
                    "kind": "table_cell",
                    "page": page_number,
                    "table": table_index,
                    "row": row_index,
                    "column": 0,
                    "bbox": list(bbox),
                },
            }
        )
    return out, events


def _recover_weak_text_tokens(
    page_image: Any | None,
    tokens: list[_Token],
    *,
    page_width: float,
    page_height: float,
    page_number: int,
    table_index: int,
    max_repairs: int = 8,
) -> tuple[list[_Token], list[dict[str, Any]]]:
    """Repair structurally suspicious text tokens using local glyph evidence."""

    if page_image is None or max_repairs <= 0:
        return tokens, []
    from docmirror.ocr.micro_grid.cell_recognition import recover_empty_quote_dash_from_image

    output: list[_Token] = []
    events: list[dict[str, Any]] = []
    for token in tokens:
        if len(events) >= max_repairs:
            output.append(token)
            continue
        recognition = recover_empty_quote_dash_from_image(
            page_image,
            token.text,
            token.bbox,
            page_width=page_width,
            page_height=page_height,
        )
        if recognition is None or recognition.text == token.text:
            output.append(token)
            continue
        confidence = min(float(token.confidence), float(recognition.confidence))
        output.append(_Token(recognition.text, token.bbox, token.evidence_id, confidence))
        events.append(
            {
                "action": "corrected",
                "input_text": token.text,
                "output_text": recognition.text,
                "confidence": round(confidence, 4),
                "reason_codes": ["empty_quote_horizontal_dash_shape"],
                "source_ref": token.evidence_id,
                "audit": recognition.to_dict(),
                "target": {
                    "kind": "ocr_token",
                    "page": page_number,
                    "table": table_index,
                    "bbox": list(token.bbox),
                },
            }
        )
    return output, events


def _comparable_section_label(value: str) -> str:
    """Return a punctuation-stable comparison key for one table label."""

    compact = re.sub(r"\s+", "", str(value or "")).translate(str.maketrans({"：": ":", "，": ","}))
    return compact.rstrip(":,.;；")


def _leading_section_ordinal_consensus(
    recognition: Any,
    observed_body: str,
) -> tuple[str, float, int] | None:
    """Return an ordinal supported by at least two suffix-matching OCR variants."""

    if not observed_body:
        return None
    votes = [vote for vote in recognition.audit.get("votes") or [] if isinstance(vote, dict)]
    observations = votes or [
        {
            "text": recognition.text,
            "confidence": recognition.confidence,
            "support_count": int(recognition.audit.get("consensus_count") or 0),
        }
    ]
    supported: dict[str, list[float]] = {}
    explicit_support: dict[str, int] = {}
    for observation in observations:
        text = str(observation.get("text") or observation.get("raw_text") or "").strip()
        match = _LEADING_CJK_SECTION_LABEL_RE.fullmatch(text)
        confidence = float(observation.get("confidence") or 0.0)
        if match is None or confidence < 0.5:
            continue
        if _comparable_section_label(match.group("body")) != observed_body:
            continue
        ordinal = match.group("ordinal")
        supported.setdefault(ordinal, []).append(confidence)
        explicit_support[ordinal] = max(explicit_support.get(ordinal, 0), int(observation.get("support_count") or 0))
    if not supported:
        return None
    ordinal, confidences = max(supported.items(), key=lambda item: (len(item[1]), max(item[1])))
    support_count = max(len(confidences), explicit_support.get(ordinal, 0))
    confidence = max(confidences)
    if support_count < 2 or confidence < 0.75:
        return None
    return ordinal, confidence, support_count


def _recover_cjk_ordinal_markers(
    raw: list[list[str]],
    *,
    cell_bboxes: list[list[list[float] | None]],
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Repair damaged parenthesized CJK ordinals from a proven same-column sequence."""

    out = [list(row) for row in raw]
    events: list[dict[str, Any]] = []
    width = max((len(row) for row in out), default=0)
    for column in range(width):
        candidates: list[tuple[int, int | None, str]] = []
        for row_index, row in enumerate(out):
            value = str(row[column] if column < len(row) else "").strip()
            match = _CJK_ORDINAL_RE.match(value)
            if match is not None:
                ordinal = _cjk_ordinal_value(match.group("ordinal"))
                if ordinal is not None:
                    candidates.append((row_index, ordinal, match.group("suffix")))
                continue
            missing = _MISSING_CJK_ORDINAL_RE.match(value)
            if missing is not None:
                candidates.append((row_index, None, missing.group("suffix")))
        known = [(position, ordinal) for position, (_row, ordinal, _suffix) in enumerate(candidates) if ordinal]
        if len(known) < 3:
            continue
        bases = [ordinal - position for position, ordinal in known]
        base, support = Counter(bases).most_common(1)[0]
        if support != len(known):
            continue
        for position, (row_index, observed_ordinal, suffix) in enumerate(candidates):
            expected = base + position
            if not 1 <= expected <= len(_CJK_ORDINALS):
                continue
            expected_text = f"（{_CJK_ORDINALS[expected - 1]}）{suffix}"
            original = str(out[row_index][column] or "")
            if original == expected_text or (observed_ordinal is not None and observed_ordinal != expected):
                continue
            out[row_index][column] = expected_text
            cell_confidences[row_index][column] = max(float(cell_confidences[row_index][column] or 0.0), 0.95)
            bbox = cell_bboxes[row_index][column] if row_index < len(cell_bboxes) else None
            evidence_ids = cell_evidence_ids[row_index][column] if row_index < len(cell_evidence_ids) else []
            source_ref = (
                evidence_ids[0] if evidence_ids else f"table:p{page_number}:t{table_index}:r{row_index}:c{column}"
            )
            events.append(
                {
                    "action": "corrected",
                    "input_text": original,
                    "output_text": expected_text,
                    "confidence": 0.95,
                    "reason_codes": ["cjk_ordinal_sequence_lattice"],
                    "source_ref": source_ref,
                    "audit": {
                        "known_ordinals": len(known),
                        "sequence_support_count": support,
                    },
                    "target": {
                        "kind": "table_cell",
                        "page": page_number,
                        "table": table_index,
                        "row": row_index,
                        "column": column,
                        **({"bbox": list(bbox)} if bbox is not None else {}),
                    },
                }
            )
    return out, events


def _normalize_single_cell_numeric_punctuation(
    tokens: list[_Token],
    *,
    x_lines: list[int],
    sx: float,
    page_number: int,
    table_index: int,
) -> tuple[list[_Token], list[dict[str, Any]]]:
    """Repair an invalid amount only after geometry proves it occupies one cell."""

    out: list[_Token] = []
    events: list[dict[str, Any]] = []
    for token in tokens:
        compact = _normalize_recovered_amount(token.text)
        candidate = _repair_unambiguous_decimal_punctuation(compact)
        column = _band_index(token.cx / sx, x_lines)
        if candidate is None or column is None:
            out.append(token)
            continue
        left = token.bbox[0] / sx
        right = token.bbox[2] / sx
        cell_left = x_lines[column]
        cell_right = x_lines[column + 1]
        tolerance = max(2.0, (cell_right - cell_left) * 0.04)
        if left < cell_left - tolerance or right > cell_right + tolerance:
            out.append(token)
            continue
        out.append(_Token(candidate, token.bbox, token.evidence_id, token.confidence))
        events.append(
            {
                "action": "corrected",
                "input_text": token.text,
                "output_text": candidate,
                "confidence": round(float(token.confidence), 4),
                "reason_codes": ["numeric_single_cell_punctuation_structure"],
                "source_ref": token.evidence_id,
                "target": {
                    "kind": "ocr_token",
                    "page": page_number,
                    "table": table_index,
                    "bbox": list(token.bbox),
                },
            }
        )
    return out, events


def _normalize_numeric_cell_punctuation(
    raw: list[list[str]],
    *,
    cell_bboxes: list[list[list[float] | None]],
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    column_bands: list[dict[str, Any]],
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Repair punctuation only after OCR tokens have been assigned to one table cell."""

    out = [list(row) for row in raw]
    events: list[dict[str, Any]] = []
    for row_index, row in enumerate(out):
        for column, value in enumerate(row):
            original = str(value or "").strip()
            candidate = _repair_unambiguous_decimal_punctuation(original)
            bbox = cell_bboxes[row_index][column] if row_index < len(cell_bboxes) else None
            if candidate is None or bbox is None or column >= len(column_bands):
                continue
            band = column_bands[column]
            tolerance = max(2.0, (float(band["x1"]) - float(band["x0"])) * 0.04)
            within_anchor_band = bbox[0] >= float(band["x0"]) - tolerance and bbox[2] <= float(band["x1"]) + tolerance
            if not within_anchor_band and not _cell_bbox_has_repeated_column_support(
                cell_bboxes,
                row_index=row_index,
                column=column,
                bbox=bbox,
            ):
                continue
            out[row_index][column] = candidate
            cell_confidences[row_index][column] = max(float(cell_confidences[row_index][column] or 0.0), 0.95)
            evidence_ids = cell_evidence_ids[row_index][column] if row_index < len(cell_evidence_ids) else []
            source_ref = (
                evidence_ids[0] if evidence_ids else f"table:p{page_number}:t{table_index}:r{row_index}:c{column}"
            )
            events.append(
                {
                    "action": "corrected",
                    "input_text": original,
                    "output_text": candidate,
                    "confidence": 0.95,
                    "reason_codes": ["numeric_materialized_cell_punctuation_structure"],
                    "source_ref": source_ref,
                    "target": {
                        "kind": "table_cell",
                        "page": page_number,
                        "table": table_index,
                        "row": row_index,
                        "column": column,
                        **({"bbox": list(bbox)} if bbox is not None else {}),
                    },
                }
            )
    return out, events


def _cell_bbox_has_repeated_column_support(
    cell_bboxes: list[list[list[float] | None]],
    *,
    row_index: int,
    column: int,
    bbox: list[float],
) -> bool:
    """Check whether other rows independently prove the same physical column boundaries."""

    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    tolerance = max(3.0, width * 0.08)
    support = 0
    for other_row_index, row in enumerate(cell_bboxes):
        if other_row_index == row_index or column >= len(row) or row[column] is None:
            continue
        other = row[column]
        if abs(float(other[0]) - float(bbox[0])) <= tolerance and abs(float(other[2]) - float(bbox[2])) <= tolerance:
            support += 1
            if support >= 2:
                return True
    return False


def _recover_arabic_ordinal_markers(
    raw: list[list[str]],
    *,
    cell_bboxes: list[list[list[float] | None]],
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Repair list ordinals from section-bounded same-column sequence evidence."""

    out = [list(row) for row in raw]
    events: list[dict[str, Any]] = []
    width = max((len(row) for row in out), default=0)
    for column in range(width):
        runs: list[list[int]] = []
        for parent_index, row in enumerate(out):
            parent = str(row[column] if column < len(row) else "").strip()
            normalized_parent = unicodedata.normalize("NFKC", parent)
            if not _PARENTHETICAL_CJK_SECTION_RE.match(normalized_parent):
                continue
            run: list[int] = []
            for row_index in range(parent_index + 1, len(out)):
                label = str(out[row_index][column] if column < len(out[row_index]) else "").strip()
                normalized_label = unicodedata.normalize("NFKC", label)
                if not normalized_label and all(not str(value or "").strip() for value in out[row_index]):
                    continue
                if _PARENTHETICAL_CJK_SECTION_RE.match(normalized_label) or _CJK_SECTION_LABEL_RE.match(
                    normalized_label
                ):
                    break
                if _ARABIC_LIST_LABEL_RE.match(normalized_label) or _MISSING_ARABIC_LIST_LABEL_RE.match(
                    normalized_label
                ):
                    run.append(row_index)
                    continue
                if normalized_label == "其他" and run:
                    run.append(row_index)
                break
            if len(run) < 2:
                continue
            first_label = unicodedata.normalize("NFKC", str(out[run[0]][column] or "").strip())
            first = _ARABIC_LIST_LABEL_RE.match(first_label)
            if first is not None and int(first.group("ordinal")) == 1:
                runs.append(run)

        supported_runs = 0
        total_support = 0
        run_support: dict[int, int] = {}
        for run_index, run in enumerate(runs):
            support = 0
            for position, row_index in enumerate(run, start=1):
                label = str(out[row_index][column] or "").strip()
                match = _ARABIC_LIST_LABEL_RE.match(unicodedata.normalize("NFKC", label))
                if match is not None and int(match.group("ordinal")) == position:
                    support += 1
            run_support[run_index] = support
            total_support += support
            if support >= 2:
                supported_runs += 1
        convention_proven = supported_runs >= 2 and total_support >= 4

        for run_index, run in enumerate(runs):
            if run_support.get(run_index, 0) < 2 and not convention_proven:
                continue
            for expected, row_index in enumerate(run, start=1):
                original = str(out[row_index][column] or "").strip()
                normalized_original = unicodedata.normalize("NFKC", original)
                explicit = _ARABIC_LIST_LABEL_RE.match(normalized_original)
                missing = _MISSING_ARABIC_LIST_LABEL_RE.match(normalized_original)
                if explicit is not None and int(explicit.group("ordinal")) == expected:
                    continue
                if explicit is not None:
                    candidate = f"{expected}.{explicit.group('suffix')}"
                elif missing is not None:
                    candidate = f"{expected}.{missing.group('suffix')}"
                elif normalized_original == "其他":
                    candidate = f"{expected}.其他"
                else:
                    continue
                out[row_index][column] = candidate
                cell_confidences[row_index][column] = max(float(cell_confidences[row_index][column] or 0.0), 0.95)
                bbox = cell_bboxes[row_index][column] if row_index < len(cell_bboxes) else None
                evidence_ids = cell_evidence_ids[row_index][column] if row_index < len(cell_evidence_ids) else []
                source_ref = (
                    evidence_ids[0] if evidence_ids else f"table:p{page_number}:t{table_index}:r{row_index}:c{column}"
                )
                events.append(
                    {
                        "action": "corrected",
                        "input_text": original,
                        "output_text": candidate,
                        "confidence": 0.95,
                        "reason_codes": ["arabic_ordinal_section_sequence_lattice"],
                        "source_ref": source_ref,
                        "audit": {
                            "run_length": len(run),
                            "same_column_support_count": run_support.get(run_index, 0),
                            "table_convention_support_count": total_support,
                        },
                        "target": {
                            "kind": "table_cell",
                            "page": page_number,
                            "table": table_index,
                            "row": row_index,
                            "column": column,
                            **({"bbox": list(bbox)} if bbox is not None else {}),
                        },
                    }
                )
    return out, events


def _cjk_ordinal_value(value: str) -> int | None:
    try:
        return _CJK_ORDINALS.index(value) + 1
    except ValueError:
        return None


def _consecutive_runs(values: list[int], *, max_gap: int) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values:
        if not runs or value - runs[-1][-1] > max_gap + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def _recover_dash_placeholder_cells(
    page_image: Any,
    raw: list[list[str]],
    *,
    cell_bboxes: list[list[list[float] | None]],
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    cell_geometry_status: list[list[str]],
    cell_geometry_loss_reason: list[list[str | None]],
    page_width: float,
    page_height: float,
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Restore numeric-column dash placeholders only from crop-level shape evidence."""

    placeholder_columns = _numeric_placeholder_columns(raw)
    if not placeholder_columns:
        return raw, []

    from docmirror.ocr.micro_grid.cell_recognition import recognize_micro_cell_from_image

    out = [list(row) for row in raw]
    source_bboxes = [[list(bbox) if bbox is not None else None for bbox in row] for row in cell_bboxes]
    events: list[dict[str, Any]] = []
    suspect_values = {"", "-", "—", "一", "二", "工", "|", "I", '"', "'", "“", "”"}
    for row_index in range(len(out)):
        row = out[row_index]
        if not any(_looks_numeric(value) or _is_grid_numeric_token(value) for value in row):
            continue
        for column in placeholder_columns:
            if column >= len(row) or str(row[column] or "").strip() not in suspect_values:
                continue
            review_bbox, bbox_was_derived = _placeholder_review_bbox(
                source_bboxes,
                row_index=row_index,
                column=column,
            )
            if review_bbox is None:
                continue
            recognition = recognize_micro_cell_from_image(
                page_image,
                review_bbox,
                page_width=page_width,
                page_height=page_height,
                allowed_charset="-—",
                max_chars=1,
                isolate_glyph=True,
                min_confidence=0.5,
            )
            shape_confirmed = any(
                vote.get("variant") == "glyph_shape_dash" and float(vote.get("confidence") or 0.0) >= 0.8
                for vote in recognition.audit.get("votes") or []
            )
            if recognition.text not in {"-", "—"} or recognition.confidence < 0.8 or not shape_confirmed:
                continue
            original = str(row[column] or "")
            out[row_index][column] = "—"
            cell_confidences[row_index][column] = round(float(recognition.confidence), 4)
            cell_bboxes[row_index][column] = [round(value, 4) for value in review_bbox]
            cell_geometry_status[row_index][column] = "derived" if bbox_was_derived else "exact"
            cell_geometry_loss_reason[row_index][column] = None
            source_ref = f"table:p{page_number}:t{table_index}:r{row_index}:c{column}"
            events.append(
                {
                    "action": "corrected",
                    "input_text": original,
                    "output_text": "—",
                    "confidence": round(float(recognition.confidence), 4),
                    "reason_codes": ["numeric_placeholder_cell_crop_shape"],
                    "source_ref": source_ref,
                    "audit": recognition.to_dict(),
                    "target": {
                        "kind": "table_cell",
                        "page": page_number,
                        "table": table_index,
                        "row": row_index,
                        "column": column,
                        "bbox": list(review_bbox),
                    },
                }
            )
            if not cell_evidence_ids[row_index][column]:
                cell_evidence_ids[row_index][column] = [source_ref]
    return out, events


def _numeric_placeholder_columns(raw: list[list[str]]) -> list[int]:
    """Infer amount columns from repeated decimal values, independent of document type."""

    width = max((len(row) for row in raw), default=0)
    columns: list[int] = []
    for column in range(width):
        values = [
            _normalize_recovered_amount(row[column])
            for row in raw
            if column < len(row) and str(row[column] or "").strip()
        ]
        if sum(bool(_EXACT_DECIMAL_RE.fullmatch(value)) for value in values) >= 3:
            columns.append(column)
    if columns:
        rightmost = max(columns)
        columns.extend(column for column in range(rightmost + 1, width) if column not in columns)
    return columns


def _placeholder_review_bbox(
    cell_bboxes: list[list[list[float] | None]],
    *,
    row_index: int,
    column: int,
) -> tuple[tuple[float, float, float, float] | None, bool]:
    """Split a merged placeholder crop using stable column bounds from adjacent rows."""

    bbox = tuple(float(value) for value in cell_bboxes[row_index][column] or ())
    if not bbox:
        peer_boxes = [
            tuple(float(value) for value in row[column] or ())
            for index, row in enumerate(cell_bboxes)
            if index != row_index and column < len(row) and row[column] is not None
        ]
        peer_boxes = [peer for peer in peer_boxes if len(peer) == 4 and peer[0] < peer[2]]
        row_boxes = [tuple(float(value) for value in value or ()) for value in cell_bboxes[row_index]]
        row_boxes = [value for value in row_boxes if len(value) == 4 and value[1] < value[3]]
        if not peer_boxes or not row_boxes:
            return None, False
        minimum_width = min(peer[2] - peer[0] for peer in peer_boxes)
        narrow_peers = [peer for peer in peer_boxes if peer[2] - peer[0] <= minimum_width * 1.25]
        return (
            (
                float(statistics.median(peer[0] for peer in narrow_peers)),
                float(statistics.median(value[1] for value in row_boxes)),
                float(statistics.median(peer[2] for peer in narrow_peers)),
                float(statistics.median(value[3] for value in row_boxes)),
            ),
            True,
        )
    if len(bbox) != 4:
        return None, False
    row_boxes = cell_bboxes[row_index]
    duplicate_count = sum(
        other is not None and all(abs(float(left) - float(right)) < 0.01 for left, right in zip(other, bbox))
        for other in row_boxes
    )
    if duplicate_count < 2:
        return bbox, False

    peer_boxes = [
        tuple(float(value) for value in row[column] or ())
        for index, row in enumerate(cell_bboxes)
        if index != row_index and column < len(row) and row[column] is not None
    ]
    peer_boxes = [
        peer
        for peer in peer_boxes
        if len(peer) == 4
        and bbox[0] - 0.01 <= peer[0] < peer[2] <= bbox[2] + 0.01
        and peer[2] - peer[0] < (bbox[2] - bbox[0]) * 0.75
    ]
    if not peer_boxes:
        return bbox, False
    derived = (
        float(statistics.median(peer[0] for peer in peer_boxes)),
        bbox[1],
        float(statistics.median(peer[2] for peer in peer_boxes)),
        bbox[3],
    )
    return (derived, True) if derived[0] < derived[2] else (bbox, False)


def _expected_missing_ordinal(
    raw: list[list[str]],
    data_rows: list[int],
    position: int,
    column: int,
) -> int | None:
    previous = _integer_cell(raw, data_rows[position - 1], column) if position > 0 else None
    following = _integer_cell(raw, data_rows[position + 1], column) if position + 1 < len(data_rows) else None
    if previous is not None and following is not None and previous + 2 == following:
        return previous + 1
    if previous is None and following == 2:
        return 1
    return None


def _integer_cell(raw: list[list[str]], row: int, column: int) -> int | None:
    value = str(raw[row][column] if row < len(raw) and column < len(raw[row]) else "").strip()
    return int(value) if re.fullmatch(r"\d{1,3}", value) else None


def _normalize_recovered_amount(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("，", ",").replace("−", "-")


def _decimal_value_is_visibly_incomplete(value: str) -> bool:
    return bool(re.search(r"\.\d?$", _normalize_recovered_amount(value)))


def _decimal_value_has_invalid_punctuation(value: str) -> bool:
    """Return whether a numeric OCR token has an impossible decimal layout."""

    text = _normalize_recovered_amount(value)
    return bool(re.fullmatch(r"[-()\d,.]+", text) and text.count(".") > 1)


def _repair_unambiguous_decimal_punctuation(value: str) -> str | None:
    """Repair separators only when one exact grouped amount is possible."""

    text = _normalize_recovered_amount(value)
    sign = ""
    if text.startswith(("-", "+")):
        sign, text = text[0], text[1:]
    if text.count(".") <= 1 or not re.fullmatch(r"\d[\d,.]+", text):
        return None
    integer_part, decimal_part = text.rsplit(".", 1)
    groups = re.split(r"[,.]", integer_part)
    if not re.fullmatch(r"\d{2}", decimal_part):
        return None
    if (
        not groups
        or not re.fullmatch(r"\d{1,3}", groups[0])
        or any(not re.fullmatch(r"\d{3}", group) for group in groups[1:])
    ):
        return None
    candidate = f"{sign}{','.join(groups)}.{decimal_part}"
    return candidate if _EXACT_DECIMAL_RE.fullmatch(candidate) and candidate != f"{sign}{text}" else None


def _containing_grid_cell_bbox(
    token: _Token,
    grid_geometry: tuple[list[int], list[int], float, float] | None,
) -> tuple[float, float, float, float] | None:
    if grid_geometry is None:
        return None
    x_lines, y_lines, sx, sy = grid_geometry
    row = _band_index(token.cy / sy, y_lines)
    column = _band_index(token.cx / sx, x_lines)
    if row is None or column is None:
        return None
    return (
        x_lines[column] * sx,
        y_lines[row] * sy,
        x_lines[column + 1] * sx,
        y_lines[row + 1] * sy,
    )


def _reconstruct_decimal_from_cell_votes(
    original: str,
    audit: dict[str, Any],
) -> tuple[str, float, int] | None:
    """Recover two decimals when two cell variants prove the complete digit lattice."""

    match = re.fullmatch(r"(?P<sign>-?)(?P<integer>(?:\d{1,3}(?:,\d{3})+|\d+))\.(?P<fraction>\d?)", original)
    if match is None:
        return None
    integer_digits = re.sub(r"\D", "", match.group("integer"))
    expected_digit_count = len(integer_digits) + 2
    votes_by_digits: dict[str, list[float]] = {}
    for vote in audit.get("votes") or []:
        confidence = float(vote.get("confidence") or 0.0)
        digits = re.sub(r"\D", "", str(vote.get("text") or ""))
        if confidence < 0.5 or len(digits) != expected_digit_count or not digits.startswith(integer_digits):
            continue
        votes_by_digits.setdefault(digits, []).append(confidence)
    if not votes_by_digits:
        return None
    digits, confidences = max(votes_by_digits.items(), key=lambda item: (len(item[1]), max(item[1])))
    if len(confidences) < 2:
        return None
    integer = digits[:-2]
    grouped = f"{int(integer):,}"
    return f"{match.group('sign')}{grouped}.{digits[-2:]}", max(confidences), len(confidences)


def _reconstruct_zero_from_cell_votes(
    original: str,
    audit: dict[str, Any],
) -> tuple[str, float, int] | None:
    """Recover zero when repeated cell views disprove one uncertain trailing digit."""

    if not _looks_like_zero_digit_confusion(original):
        return None
    confidences = [
        float(vote.get("confidence") or 0.0)
        for vote in audit.get("votes") or []
        if float(vote.get("confidence") or 0.0) >= 0.5
        and re.fullmatch(r"0(?:\.0?)?", _normalize_recovered_amount(str(vote.get("text") or "")))
    ]
    if len(confidences) < 3 or max(confidences) < 0.7:
        return None
    sign = "-" if original.startswith("-") else ""
    return f"{sign}0.00", max(confidences), len(confidences)


def _accept_numeric_recovery(
    original: str,
    candidate: str,
    *,
    original_confidence: float,
    recognition_confidence: float,
    consensus_count: int,
    summary_row: bool,
) -> bool:
    if not candidate or candidate == original:
        return False
    if not _EXACT_DECIMAL_RE.fullmatch(candidate):
        return False
    if _decimal_value_is_visibly_incomplete(original):
        original_integer = re.sub(r"\D", "", original.split(".", 1)[0])
        candidate_integer = re.sub(r"\D", "", candidate.split(".", 1)[0])
        if (
            original_integer == candidate_integer
            and consensus_count >= 3
            and recognition_confidence >= 0.8
            and original_confidence < 0.9
            and recognition_confidence >= original_confidence - 0.02
        ):
            return True
    if consensus_count < 2 or recognition_confidence < 0.9:
        return False
    if candidate.startswith("-") and candidate[1:] == original.lstrip("+"):
        return summary_row and recognition_confidence >= original_confidence
    if _EXACT_DECIMAL_RE.fullmatch(original):
        return (
            _looks_like_zero_digit_confusion(original)
            and bool(re.fullmatch(r"-?0\.00", candidate.replace(",", "")))
            and consensus_count >= 3
            and recognition_confidence >= 0.95
            and original_confidence < 0.9
        )
    original_digits = re.sub(r"\D", "", original)
    candidate_digits = re.sub(r"\D", "", candidate)
    digit_evidence_agrees = (
        candidate_digits.startswith(original_digits)
        or original_digits.startswith(candidate_digits)
        or recognition_confidence >= original_confidence + 0.05
    )
    return digit_evidence_agrees and recognition_confidence >= original_confidence


def _projection_line_positions(mask: Any, *, axis: int, offset: int) -> list[int]:
    import numpy as np

    projection = (mask > 0).mean(axis=axis)
    peak = float(projection.max()) if projection.size else 0.0
    threshold = max(0.16, min(0.35, peak * 0.68))
    indices = np.where(projection >= threshold)[0].tolist()
    groups: list[list[int]] = []
    for index in indices:
        if not groups or index - groups[-1][-1] > 3:
            groups.append([index])
        else:
            groups[-1].append(index)
    return [offset + int(round(sum(group) / len(group))) for group in groups]


def _ensure_outer_lines(lines: list[int], start: int, end: int) -> list[int]:
    """Deduplicate detected rules and restore open table outer boundaries."""
    result = sorted({int(value) for value in lines if start <= int(value) <= end})
    if not result:
        return [start, end] if end - start >= 12 else []
    if result[0] - start >= 6:
        result.insert(0, start)
    else:
        result[0] = start
    if end - result[-1] >= 6:
        result.append(end)
    else:
        result[-1] = end
    return [value for index, value in enumerate(result) if index == 0 or value - result[index - 1] >= 6]


def _recover_sparse_vertical_label_groups(
    groups: list[set[tuple[int, int]]],
    tokens: list[_Token],
    *,
    x_lines: list[int],
    y_lines: list[int],
    sx: float,
    sy: float,
) -> list[set[tuple[int, int]]]:
    """Coalesce a sparse first-column label over consecutive numeric detail rows."""

    if len(x_lines) < 3 or len(y_lines) < 4:
        return groups
    unit_left_rows = {row for group in groups if len(group) == 1 for row, column in group if column == 0}
    if len(unit_left_rows) < 2:
        return groups

    tokens_by_cell: dict[tuple[int, int], list[_Token]] = {}
    numeric_columns_by_row: dict[int, set[int]] = {}
    for token in tokens:
        row = _band_index(token.cy / sy, y_lines)
        column = _band_index(token.cx / sx, x_lines)
        if row is None or column is None:
            continue
        tokens_by_cell.setdefault((row, column), []).append(token)
        if column > 0 and _is_grid_numeric_token(token.text):
            numeric_columns_by_row.setdefault(row, set()).add(column)

    detail_rows = sorted(row for row in unit_left_rows if len(numeric_columns_by_row.get(row, set())) >= 2)
    runs = _consecutive_runs(detail_rows, max_gap=0)
    merge_runs: list[list[int]] = []
    for run in runs:
        if len(run) < 2:
            continue
        populated = [row for row in run if tokens_by_cell.get((row, 0))]
        if len(populated) == 1:
            merge_runs.append(run)
    if not merge_runs:
        return groups

    merged_cells = {cell for run in merge_runs for cell in ((row, 0) for row in run)}
    out = [group for group in groups if not (len(group) == 1 and next(iter(group)) in merged_cells)]
    out.extend({(row, 0) for row in run} for run in merge_runs)
    return out


def _merged_cell_groups(
    horizontal: Any,
    vertical: Any,
    x_lines: list[int],
    y_lines: list[int],
    *,
    preserve_left_column_rows: bool = False,
    forced_vertical_lines: set[int] | None = None,
    forced_horizontal_lines: set[int] | None = None,
) -> tuple[list[set[tuple[int, int]]], dict[str, int]]:
    """Return rectangular merged-cell groups validated against original masks."""
    rows, cols = len(y_lines) - 1, len(x_lines) - 1
    parent = list(range(rows * cols))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in range(rows):
        for col in range(cols - 1):
            x = x_lines[col + 1]
            if forced_vertical_lines and x in forced_vertical_lines:
                continue
            y0, y1 = y_lines[row], y_lines[row + 1]
            pad = max(1, int((y1 - y0) * 0.12))
            strength = float((vertical[y0 + pad : max(y0 + pad + 1, y1 - pad), max(0, x - 1) : x + 2] > 0).mean())
            if strength < 0.30:
                union(row * cols + col, row * cols + col + 1)
    for row in range(rows - 1):
        for col in range(cols):
            if preserve_left_column_rows and col == 0:
                continue
            y = y_lines[row + 1]
            if forced_horizontal_lines and y in forced_horizontal_lines:
                continue
            x0, x1 = x_lines[col], x_lines[col + 1]
            pad = max(1, int((x1 - x0) * 0.08))
            strength = float((horizontal[max(0, y - 1) : y + 2, x0 + pad : max(x0 + pad + 1, x1 - pad)] > 0).mean())
            if strength < 0.30:
                union(row * cols + col, (row + 1) * cols + col)

    candidates: dict[int, set[tuple[int, int]]] = {}
    for row in range(rows):
        for col in range(cols):
            candidates.setdefault(find(row * cols + col), set()).add((row, col))

    accepted: list[set[tuple[int, int]]] = []
    diagnostics = {
        "merge_candidate_count": 0,
        "accepted_merge_count": 0,
        "rejected_non_rectangular_count": 0,
        "rejected_internal_divider_count": 0,
        "rejected_full_table_count": 0,
        "fallback_unit_cell_count": 0,
    }
    for cells in candidates.values():
        if len(cells) == 1:
            accepted.append(cells)
            continue
        diagnostics["merge_candidate_count"] += 1
        row_values = sorted({row for row, _col in cells})
        col_values = sorted({col for _row, col in cells})
        rectangle = {(row, col) for row in row_values for col in col_values}
        if cells != rectangle:
            diagnostics["rejected_non_rectangular_count"] += 1
            accepted.extend({cell} for cell in sorted(cells))
            diagnostics["fallback_unit_cell_count"] += len(cells)
            continue
        if len(cells) == rows * cols and rows > 1 and cols > 1:
            diagnostics["rejected_full_table_count"] += 1
            accepted.extend({cell} for cell in sorted(cells))
            diagnostics["fallback_unit_cell_count"] += len(cells)
            continue
        if _merged_rectangle_has_internal_divider(
            horizontal,
            vertical,
            x_lines,
            y_lines,
            row_values=row_values,
            col_values=col_values,
        ):
            diagnostics["rejected_internal_divider_count"] += 1
            accepted.extend({cell} for cell in sorted(cells))
            diagnostics["fallback_unit_cell_count"] += len(cells)
            continue
        accepted.append(cells)
        diagnostics["accepted_merge_count"] += 1
    return accepted, diagnostics


def _merged_rectangle_has_internal_divider(
    horizontal: Any,
    vertical: Any,
    x_lines: list[int],
    y_lines: list[int],
    *,
    row_values: list[int],
    col_values: list[int],
) -> bool:
    """Whether a candidate rectangle still contains a material internal rule."""
    for col in range(col_values[0] + 1, col_values[-1] + 1):
        x = x_lines[col]
        for row in row_values:
            y0, y1 = y_lines[row], y_lines[row + 1]
            pad = max(1, int((y1 - y0) * 0.12))
            segment = vertical[y0 + pad : max(y0 + pad + 1, y1 - pad), max(0, x - 1) : x + 2]
            if segment.size and float((segment > 0).mean()) >= 0.30:
                return True
    for row in range(row_values[0] + 1, row_values[-1] + 1):
        y = y_lines[row]
        for col in col_values:
            x0, x1 = x_lines[col], x_lines[col + 1]
            pad = max(1, int((x1 - x0) * 0.08))
            segment = horizontal[max(0, y - 1) : y + 2, x0 + pad : max(x0 + pad + 1, x1 - pad)]
            if segment.size and float((segment > 0).mean()) >= 0.30:
                return True
    return False


def _center_in_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _band_index(value: float, lines: list[int]) -> int | None:
    for index, (start, end) in enumerate(zip(lines, lines[1:], strict=False)):
        if start <= value <= end:
            return index
    return None


def _line_projection_strength(
    mask: Any,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    axis: int,
    positions: list[int],
) -> float:
    values: list[float] = []
    for position in positions:
        if axis == 1:
            values.append(float((mask[max(0, position - 1) : position + 2, x0:x1] > 0).mean()))
        else:
            values.append(float((mask[y0:y1, max(0, position - 1) : position + 2] > 0).mean()))
    return sum(values) / len(values) if values else 0.0


def _block_to_token(block: Block) -> _Token | None:
    attrs = block.attrs or {}
    text = str(block.raw_content or "").strip()
    correction = attrs.get("ocr_correction")
    original = attrs.get("ocr_original_text")
    if (
        isinstance(correction, dict)
        and correction.get("rule_id") == "unicode.normalize"
        and set(correction.get("reason_codes") or ()) == {"unicode_normalization"}
        and isinstance(original, str)
        and original.strip()
    ):
        text = original.strip()
    if not text:
        return None
    bbox = tuple(float(v) for v in (block.bbox or (0.0, 0.0, 0.0, 0.0)))
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    confidence = float(attrs.get("confidence") or 1.0)
    evidence_ids = list(block.evidence_ids or ())
    return _Token(
        text=text, bbox=bbox, evidence_id=evidence_ids[0] if evidence_ids else block.block_id, confidence=confidence
    )


def _ocr_correction_mode(blocks: list[Block]) -> str:
    for block in blocks:
        mode = str((block.attrs or {}).get("ocr_correction_mode") or "")
        if mode in {"off", "safe", "suggest"}:
            return mode
    return "safe"


def _ocr_correction_scope(blocks: list[Block]) -> dict[str, Any]:
    for block in blocks:
        attrs = block.attrs or {}
        if attrs.get("ocr_correction_processed"):
            return {
                "language": str(attrs.get("ocr_correction_language") or "") or None,
                "country": str(attrs.get("ocr_correction_country") or "") or None,
                "locale": str(attrs.get("ocr_correction_locale") or "") or None,
                "pack_ids": tuple(str(value) for value in attrs.get("ocr_correction_pack_ids") or []),
            }
    return {"language": None, "country": None, "locale": None, "pack_ids": ()}


def _correct_table_grid(
    raw: list[list[str]],
    *,
    cell_evidence_ids: list[list[list[str]]],
    cell_confidences: list[list[float | None]],
    domain: str | None,
    first_column_role: str | None,
    mode: str,
    language: str | None,
    country: str | None,
    locale: str | None,
    pack_ids: tuple[str, ...],
    page_number: int,
    table_index: int,
) -> tuple[list[list[str]], list[dict[str, Any]], int]:
    from docmirror.ocr.correction import CorrectionContext, SafeOCRCorrector

    if not raw:
        return raw, [], 0
    corrector = SafeOCRCorrector()
    headers = [str(value or "") for value in raw[0]]
    out = [list(row) for row in raw]
    events: list[dict[str, Any]] = []
    processed_count = 0
    for row_index, row in enumerate(out):
        for col_index, value in enumerate(row):
            text = str(value or "").strip()
            if not text:
                continue
            role = _table_cell_role(
                row_index=row_index,
                col_index=col_index,
                headers=headers,
                first_column_role=first_column_role,
            )
            if role == "data":
                continue
            processed_count += 1
            evidence_ids = (
                cell_evidence_ids[row_index][col_index]
                if row_index < len(cell_evidence_ids) and col_index < len(cell_evidence_ids[row_index])
                else []
            )
            confidence = (
                cell_confidences[row_index][col_index]
                if row_index < len(cell_confidences) and col_index < len(cell_confidences[row_index])
                else None
            )
            source_ref = (
                evidence_ids[0] if evidence_ids else f"table:p{page_number}:t{table_index}:r{row_index}:c{col_index}"
            )
            decision = corrector.correct(
                text,
                CorrectionContext(
                    role=role,
                    domain=domain,
                    source_ref=source_ref,
                    ocr_confidence=confidence,
                    mode=mode if mode in {"off", "safe", "suggest"} else "safe",
                    language=language,
                    country=country,
                    locale=locale,
                    pack_ids=pack_ids,
                    metadata={"field_type": role if role in {"date", "amount"} else ""},
                ),
            )
            row[col_index] = decision.output_text
            if decision.action != "unchanged":
                event = decision.to_dict()
                event["target"] = {
                    "kind": "table_cell",
                    "page": page_number,
                    "table": table_index,
                    "row": row_index,
                    "column": col_index,
                }
                events.append(event)
    return out, events, processed_count


def _table_cell_role(
    *,
    row_index: int,
    col_index: int,
    headers: list[str],
    first_column_role: str | None,
) -> str:
    if row_index == 0:
        return "table_header"
    if col_index == 0 and first_column_role:
        return first_column_role
    header = headers[col_index] if col_index < len(headers) else ""
    if re.search(r"日期|时间|期限|年月日", header):
        return "date"
    if re.search(r"金额|余额|价款|合计|总计|收入|支出|借方|贷方", header):
        return "amount"
    if re.search(r"代码|编号|证号|税号|账号|卡号", header):
        return "code"
    return "data"


def _source_correction_events(blocks: list[Block], owned_ids: set[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in blocks:
        if not (_block_evidence_ids(block) & owned_ids):
            continue
        event = (block.attrs or {}).get("ocr_correction")
        if isinstance(event, dict):
            events.append(dict(event))
    return events


def _block_evidence_ids(block: Block) -> set[str]:
    return set(block.evidence_ids or ()) or ({block.block_id} if block.block_id else set())


def _looks_numeric(text: str) -> bool:
    return bool(_NUMBER_RE.search(text.replace("，", ",")))


def _is_grid_numeric_token(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).replace("，", ",")
    return bool(_GRID_NUMERIC_RE.fullmatch(compact) or _GRID_ORDINAL_RE.fullmatch(compact))


def _cluster_rows(tokens: list[_Token]) -> list[_Row]:
    rows: list[_Row] = []
    for token in sorted(tokens, key=lambda item: (item.cy, item.bbox[0])):
        placed = False
        for row in rows[-5:]:
            tolerance = max(4.0, min(12.0, max(row.h, token.h) * 0.65))
            if abs(token.cy - row.cy) <= tolerance:
                row.tokens.append(token)
                placed = True
                break
        if not placed:
            rows.append(_Row(tokens=[token]))
    for row in rows:
        row.tokens.sort(key=lambda item: item.bbox[0])
    return rows


def _join_cell_tokens(tokens: list[_Token]) -> str:
    parts = [token.text for token in _tokens_in_reading_order(tokens)]
    text = ""
    for part in parts:
        if not text:
            text = part
            continue
        if text[-1].isdigit() and part[0].isdigit():
            text += part
        elif _is_cjk(text[-1]) or _is_cjk(part[0]):
            text += part
        else:
            text += " " + part
    return text.strip()


def _normalized_ordinal_header_text(tokens: list[_Token]) -> str | None:
    """Normalize a complete ordinal header split into vertically arranged OCR tokens."""

    text = re.sub(r"\s+", "", _join_cell_tokens(tokens))
    if text == "次行":
        return "行次"
    return text if _ORDINAL_HEADER_RE.fullmatch(text) else None


def _tokens_in_reading_order(tokens: list[_Token]) -> list[_Token]:
    """Order cell-local OCR tokens by visual line, then from left to right."""

    if len(tokens) < 2:
        return list(tokens)
    return [token for row in _cluster_rows(tokens) for token in sorted(row.tokens, key=lambda item: item.bbox[0])]


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"
