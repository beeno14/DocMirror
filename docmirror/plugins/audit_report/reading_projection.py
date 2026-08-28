"""Audit-only cleanup for the source-complete enhanced reading view."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from docmirror.output.markdown_renderer import render_semantic_source_overlay_markdown
from docmirror.plugins.audit_report.table_projection import (
    normalize_audit_display_text,
    normalize_audit_display_value,
    normalize_audit_text,
)

_FINANCIAL_DATASETS = (
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "owners_equity_changes",
)
_PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?\d{1,3}(?:\s*页)?$")
_NUMERIC_VALUE_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|元)?$")
_SEMANTIC_TABLE_MARKERS = re.compile(
    r"项目|附注|期末|期初|本期|上期|余额|金额|资产|负债|收入|利润|现金流量|所有者权益|股东权益"
)
_PERIOD_CAPTION_RE = re.compile(r"^20\d{2}(?:年度|年\d{1,2}月\d{1,2}日)$")
_PARAGRAPH_HEADING_RE = re.compile(r"^(?:[一二三四五六七八九十百]{1,4}|\d{1,2})[、.．]\S")
_LIST_ITEM_RE = re.compile(r"^[（(]\d{1,2}[）)]")
_ENUMERATED_PREFIX_RE = re.compile(r"^(?:[（(][一二三四五六七八九十百\d]{1,3}[）)]|[①②③④⑤⑥⑦⑧⑨⑩])")
_STATEMENT_CAPTION_RE = re.compile(r"^(?:资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表)(?:（续）)?$")


@dataclass
class _DatasetTableProjection:
    projected_table_ids: set[str] = field(default_factory=set)
    suppressed_table_ids: set[str] = field(default_factory=set)
    suppressed_block_ids: set[str] = field(default_factory=set)


def render_audit_reading_markdown(semantic: dict[str, Any]) -> str:
    """Render an audit report from a cleaned copy of its public semantic source."""

    rendered = render_semantic_source_overlay_markdown(prepare_audit_reading_semantic(semantic))
    rendered = _strip_synthetic_preamble(rendered)
    rendered = _normalize_display_glyphs(rendered)
    return _collapse_rendered_statement_titles(rendered)


def prepare_audit_reading_semantic(semantic: dict[str, Any]) -> dict[str, Any]:
    """Prepare a display-only source overlay without changing JSON facts."""

    projected = copy.deepcopy(semantic)
    structure = projected.get("structure") if isinstance(projected.get("structure"), dict) else {}
    blocks = [block for block in structure.get("blocks") or [] if isinstance(block, dict)]
    source_tables = [table for table in structure.get("source_tables") or [] if isinstance(table, dict)]
    domain = projected.get("domain") if isinstance(projected.get("domain"), dict) else {}
    extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
    top_level = projected.get("enhanced_markdown") if isinstance(projected.get("enhanced_markdown"), dict) else {}
    enhanced = (
        extensions.get("enhanced_markdown") if isinstance(extensions.get("enhanced_markdown"), dict) else top_level
    )
    raw_dimensions = enhanced.get("page_dimensions") if isinstance(enhanced.get("page_dimensions"), dict) else {}
    page_heights = {
        int(page): float(dimensions.get("height") or 0)
        for page, dimensions in raw_dimensions.items()
        if isinstance(dimensions, dict) and dimensions.get("height")
    }

    _normalize_dataset_display_values(projected.get("datasets") or [])
    _normalize_source_tables(source_tables)
    dataset_projection = _project_dataset_tables(projected, source_tables, blocks)
    dataset_projection.suppressed_block_ids.update(_source_table_header_fragment_block_ids(blocks, source_tables))
    recovered_table_ids = dataset_projection.projected_table_ids
    unusable_table_ids = {
        str(table.get("id") or "")
        for table in source_tables
        if str(table.get("id") or "") not in recovered_table_ids and _unusable_landscape_table(table)
    }
    unusable_pages = {
        int(table.get("page") or 1) for table in source_tables if str(table.get("id") or "") in unusable_table_ids
    }
    appendix_pages = _appendix_pages(structure)
    blocks = _curate_blocks(
        blocks,
        suppressed_table_ids=dataset_projection.suppressed_table_ids,
        suppressed_block_ids=dataset_projection.suppressed_block_ids,
        unusable_table_ids=unusable_table_ids,
        unusable_pages=unusable_pages,
        appendix_pages=appendix_pages,
        page_heights=page_heights,
    )
    blocks = _reflow_paragraph_blocks(blocks)
    page_count = max(0, int((projected.get("document") or {}).get("page_count") or 0))
    blocks = _preserve_page_sequence(blocks, page_count)
    blocks.sort(key=lambda block: (int(block.get("page") or 1), int(block.get("order") or 0), str(block.get("id"))))
    for order, block in enumerate(blocks, start=1):
        block["order"] = order

    structure["blocks"] = blocks
    structure["source_tables"] = [
        table for table in source_tables if str(table.get("id") or "") not in dataset_projection.suppressed_table_ids
    ]
    structure["reading_flows"] = [
        {"id": "audit:enhanced-reading", "type": "main", "node_ids": [str(block["id"]) for block in blocks]}
    ]
    valid_ids = {str(block["id"]) for block in blocks}
    for section in structure.get("sections") or []:
        if isinstance(section, dict) and isinstance(section.get("block_refs"), list):
            section["block_refs"] = [str(block_id) for block_id in section["block_refs"] if str(block_id) in valid_ids]
    return projected


def _normalize_dataset_display_values(datasets: list[Any]) -> None:
    """Clean values on the display-only copy consumed by dataset overlays."""

    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        for column in dataset.get("columns") or []:
            if isinstance(column, dict) and "label" in column:
                column["label"] = _display_audit_text(column["label"])
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            for value_field in ("canonical_raw", "raw", "normalized"):
                values = row.get(value_field)
                if isinstance(values, dict):
                    row[value_field] = {key: _display_audit_text(value) for key, value in values.items()}


def _normalize_source_tables(source_tables: list[dict[str, Any]]) -> None:
    for table in source_tables:
        table["headers"] = [_display_audit_text(value) for value in table.get("headers") or []]
        table["rows"] = [
            [_display_audit_text(value) for value in row] for row in table.get("rows") or [] if isinstance(row, list)
        ]


def _project_dataset_tables(
    semantic: dict[str, Any],
    source_tables: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> _DatasetTableProjection:
    by_id = {str(table.get("id") or ""): table for table in source_tables if table.get("id")}
    result = _DatasetTableProjection()
    for dataset in semantic.get("datasets") or []:
        if not isinstance(dataset, dict) or not _dataset_projection_allowed(dataset):
            continue
        dataset_name = str(dataset.get("name") or "")
        columns = [
            column
            for column in dataset.get("columns") or []
            if isinstance(column, dict) and column.get("key") and column.get("key") != "period_role"
        ]
        if not columns:
            continue
        base_headers = [_display_audit_text(column.get("label") or column["key"]) for column in columns]
        dataset_rows = [row for row in dataset.get("rows") or [] if isinstance(row, dict)]
        default_by_page = _dataset_default_tables(dataset_rows, by_id)
        grouped: dict[str, list[tuple[dict[str, Any], list[str]]]] = defaultdict(list)
        grouped_periods: dict[str, set[str]] = defaultdict(set)
        for row in dataset_rows:
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            page = max(1, int(source.get("page") or 1))
            table_id = _row_projection_table_id(
                source,
                page=page,
                dataset_name=dataset_name,
                default_by_page=default_by_page,
                by_id=by_id,
            )
            logical_id = str(source.get("table_id") or "").split(":segment_", 1)[0]
            if logical_id.startswith("lt_") and logical_id != table_id:
                result.suppressed_table_ids.add(logical_id)
            result.suppressed_table_ids.update(
                str(value) for value in (source.get("suppressed_source_table_ids") or []) if value
            )
            physical_ids = [str(value) for value in (source.get("physical_table_ids") or []) if value]
            if len(physical_ids) > 1:
                result.suppressed_table_ids.update(physical_ids[1:])
            raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else row.get("raw")
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            raw = raw if isinstance(raw, dict) else {}
            period_role = str(normalized.get("period_role") or "")
            if period_role:
                grouped_periods[table_id].add(period_role)
            grouped[table_id].append(
                (
                    row,
                    [
                        _dataset_display_value(
                            dataset_name=dataset_name,
                            column=column,
                            raw=raw,
                            normalized=normalized,
                        )
                        for column in columns
                    ],
                )
            )
        for table_id, row_entries in grouped.items():
            rows = [values for _row, values in row_entries]
            table = by_id.get(table_id)
            if table is None:
                page = max(1, int((row_entries[0][0].get("source") or {}).get("page") or 1))
                table = {"id": table_id, "page": page, "headers": [], "rows": []}
                source_tables.append(table)
                by_id[table_id] = table
                matching_block = _matching_source_text_block(blocks, page=page, rows=rows)
                order = int((matching_block or {}).get("order") or _last_page_order(blocks, page))
                block_id = f"block:{table_id}"
                blocks.append(
                    {
                        "id": block_id,
                        "kind": "physical_table",
                        "role": "body",
                        "order": order,
                        "page": page,
                        "text": "",
                        "source_table_ref": table_id,
                    }
                )
                if matching_block is not None:
                    result.suppressed_block_ids.add(str(matching_block.get("id") or ""))
            periods = grouped_periods.get(table_id) or set()
            period_label = "本期金额" if periods == {"current"} else "上期金额" if periods == {"previous"} else ""
            table["headers"] = [
                header if index == 0 or not period_label else f"{header}（{period_label}）"
                for index, header in enumerate(base_headers)
            ]
            table["rows"] = rows
            extensions = table.get("extensions") if isinstance(table.get("extensions"), dict) else {}
            extensions["audit_reading_projection"] = "source_backed_dataset_rows"
            extensions["dataset_name"] = dataset_name
            table["extensions"] = extensions
            result.projected_table_ids.add(table_id)
            result.suppressed_block_ids.update(
                _table_fragment_block_ids(blocks, table=table, columns=columns, row_entries=row_entries)
            )
    result.suppressed_table_ids.difference_update(result.projected_table_ids)
    result.suppressed_block_ids.discard("")
    return result


def _dataset_projection_allowed(dataset: dict[str, Any]) -> bool:
    status = str(dataset.get("status") or "")
    completeness = dataset.get("completeness") if isinstance(dataset.get("completeness"), dict) else {}
    return (not status or status == "complete") and completeness.get("verified") is not False


def _dataset_default_tables(
    rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[int, str]:
    candidates: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        page = max(1, int(source.get("page") or 1))
        table_id = str(source.get("physical_table_id") or source.get("table_id") or "").split(":segment_", 1)[0]
        if table_id in by_id:
            candidates[page][table_id] += 1
    return {page: counts.most_common(1)[0][0] for page, counts in candidates.items() if counts}


def _row_projection_table_id(
    source: dict[str, Any],
    *,
    page: int,
    dataset_name: str,
    default_by_page: dict[int, str],
    by_id: dict[str, dict[str, Any]],
) -> str:
    physical_ids = [str(value) for value in (source.get("physical_table_ids") or []) if value]
    candidates = [*physical_ids[:1], str(source.get("physical_table_id") or ""), str(source.get("table_id") or "")]
    for candidate in candidates:
        table_id = candidate.split(":segment_", 1)[0]
        if table_id in by_id:
            return table_id
    if page in default_by_page:
        return default_by_page[page]
    return f"audit:{dataset_name}:page:{page}"


def _matching_source_text_block(
    blocks: list[dict[str, Any]],
    *,
    page: int,
    rows: list[list[str]],
) -> dict[str, Any] | None:
    values = [_compact_text(value) for row in rows for value in row if _compact_text(value)]
    for block in blocks:
        if int(block.get("page") or 1) != page or block.get("source_table_ref"):
            continue
        text = _compact_text(block.get("text") or block.get("value"))
        matched = sum(value in text for value in values)
        if text and matched >= min(2, len(values)):
            return block
    return None


def _last_page_order(blocks: list[dict[str, Any]], page: int) -> int:
    return max((int(block.get("order") or 0) for block in blocks if int(block.get("page") or 1) == page), default=0)


def _table_fragment_block_ids(
    blocks: list[dict[str, Any]],
    *,
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    row_entries: list[tuple[dict[str, Any], list[str]]],
) -> set[str]:
    page = int(table.get("page") or 1)
    header_labels = [_compact_text(column.get("label") or column.get("key")) for column in columns]
    header_tokens = {
        token
        for label in header_labels
        for token in (label, *(part for part in re.split(r"[/／|]", label) if part))
        if token
    }
    tokens = set(header_tokens)
    tokens.update(_compact_text(value) for _row, values in row_entries for value in values)
    projection_bbox = _projection_table_bbox(table, row_entries=row_entries, column_count=len(columns))
    evidence_ids = {
        str(evidence_id)
        for row, _values in row_entries
        for evidence_id in ((row.get("source") or {}).get("evidence_ids") or [])
        if evidence_id
    }
    return {
        str(block.get("id") or "")
        for block in blocks
        if int(block.get("page") or 1) == page
        and not block.get("source_table_ref")
        and _compact_text(block.get("text") or block.get("value")) in tokens
        and not _period_caption_above_table(block, projection_bbox)
        and (
            _bbox_within(block.get("bbox"), projection_bbox)
            or bool(evidence_ids.intersection(str(value) for value in (block.get("evidence_refs") or []) if value))
            or _follows_projected_table(
                block,
                blocks=blocks,
                table_id=str(table.get("id") or ""),
                page=page,
                max_distance=max(8, len(columns) * 4),
            )
            or (
                _compact_text(block.get("text") or block.get("value")) in header_tokens
                and str(block.get("kind") or "") not in {"heading", "title"}
                and str(block.get("role") or "") not in {"heading", "title"}
            )
        )
    }


def _source_table_header_fragment_block_ids(
    blocks: list[dict[str, Any]],
    source_tables: list[dict[str, Any]],
) -> set[str]:
    """Suppress standalone source-header fragments even when a dataset is not verified."""

    headers_by_page: dict[int, set[str]] = defaultdict(set)
    for table in source_tables:
        page = max(1, int(table.get("page") or 1))
        headers_by_page[page].update(_compact_text(value) for value in table.get("headers") or [] if value)
    return {
        str(block.get("id") or "")
        for block in blocks
        if not block.get("source_table_ref")
        and str(block.get("kind") or "") not in {"heading", "title"}
        and str(block.get("role") or "") not in {"heading", "title"}
        and not _PERIOD_CAPTION_RE.fullmatch(_compact_text(block.get("text") or block.get("value")))
        and _compact_text(block.get("text") or block.get("value"))
        in headers_by_page.get(max(1, int(block.get("page") or 1)), set())
    }


def _period_caption_above_table(block: dict[str, Any], table_bbox: list[float] | None) -> bool:
    text = _compact_text(block.get("text") or block.get("value"))
    bbox = block.get("bbox")
    return bool(
        _PERIOD_CAPTION_RE.fullmatch(text)
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and isinstance(table_bbox, list)
        and len(table_bbox) == 4
        and float(bbox[3]) <= float(table_bbox[1]) + 2
    )


def _follows_projected_table(
    block: dict[str, Any],
    *,
    blocks: list[dict[str, Any]],
    table_id: str,
    page: int,
    max_distance: int,
) -> bool:
    block_order = int(block.get("order") or 0)
    table_orders = [
        int(candidate.get("order") or 0)
        for candidate in blocks
        if int(candidate.get("page") or 1) == page and str(candidate.get("source_table_ref") or "") == table_id
    ]
    return any(0 < block_order - table_order <= max_distance for table_order in table_orders)


def _projection_table_bbox(
    table: dict[str, Any],
    *,
    row_entries: list[tuple[dict[str, Any], list[str]]],
    column_count: int,
) -> list[float] | None:
    boxes = [table.get("bbox")]
    boxes.extend(
        source_ref.get("bbox")
        for row, _values in row_entries
        for source_ref in ((row.get("source") or {}).get("source_cell_refs") or [])
        if isinstance(source_ref, dict)
    )
    valid = [list(map(float, box)) for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return None
    union = [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]
    horizontal_padding = (union[2] - union[0]) / max(1, column_count)
    union[0] -= horizontal_padding
    union[2] += horizontal_padding
    return union


def _bbox_within(inner: Any, outer: Any) -> bool:
    return bool(
        isinstance(inner, (list, tuple))
        and len(inner) == 4
        and isinstance(outer, (list, tuple))
        and len(outer) == 4
        and float(inner[0]) >= float(outer[0]) - 2
        and float(inner[1]) >= float(outer[1]) - 2
        and float(inner[2]) <= float(outer[2]) + 2
        and float(inner[3]) <= float(outer[3]) + 2
    )


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", _display_audit_text(value))


def _curate_blocks(
    blocks: list[dict[str, Any]],
    *,
    suppressed_table_ids: set[str],
    suppressed_block_ids: set[str],
    unusable_table_ids: set[str],
    unusable_pages: set[int],
    appendix_pages: set[int],
    page_heights: dict[int, float],
) -> list[dict[str, Any]]:
    normalized_blocks = [_normalize_block(block) for block in blocks]
    repeated_headers = _running_headers(normalized_blocks)
    seen_running: set[str] = set()
    seen_same_page: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    curated: list[dict[str, Any]] = []
    for block in normalized_blocks:
        if str(block.get("id") or "") in suppressed_block_ids:
            continue
        page = max(1, int(block.get("page") or 1))
        text = _display_audit_text(block.get("text") or block.get("value"))
        compact = re.sub(r"\s+", "", text)
        table_id = str(block.get("source_table_ref") or "")
        if table_id in suppressed_table_ids:
            continue
        if table_id in unusable_table_ids:
            block.pop("source_table_ref", None)
            block.update(
                kind="key_value",
                role="artifact",
                key="",
                value=f"（第{page}页横向财务表格识别置信度不足，请核对源 PDF）",
                text=f"第{page}页横向财务表格识别置信度不足",
            )
            text = str(block["value"])
            compact = re.sub(r"\s+", "", text)
        if _printed_page_number(block, text, page_height=page_heights.get(page)):
            continue
        if page in unusable_pages and not table_id and _low_confidence_fragment(block, text, threshold=0.75):
            continue
        if page in appendix_pages and _low_confidence_fragment(block, text):
            continue
        if compact in repeated_headers:
            if compact in seen_running:
                continue
            seen_running.add(compact)
        key = (page, compact)
        if compact and _overlaps_existing(block.get("bbox"), seen_same_page[key]):
            continue
        if compact:
            bbox = block.get("bbox")
            seen_same_page[key].append(list(bbox) if isinstance(bbox, list) and len(bbox) == 4 else [])
        if _NUMERIC_VALUE_RE.fullmatch(compact) and str(block.get("kind") or "") not in {"heading", "title"}:
            block.update(kind="key_value", role="body", key="", value=text)
        curated.append(block)
    return curated


def _normalize_block(block: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(block)
    for key in ("text", "key", "value"):
        if key in normalized:
            lines = [line for line in str(normalized[key] or "").splitlines() if line.strip()]
            normalized[key] = "\n".join(
                _collapse_repeated_statement_title(_display_audit_text(line)) for line in lines
            )
    text_lines = str(normalized.get("text") or "").splitlines()
    if len(text_lines) == 2 and text_lines[1].startswith("中国注册会计师"):
        normalized.update(kind="key_value", role="body", key=text_lines[0], value=text_lines[1])
    return normalized


def _reflow_paragraph_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join same-page physical lines only when geometry proves one paragraph flow."""

    reflowed: list[dict[str, Any]] = []
    for block in blocks:
        if reflowed and _paragraph_continuation(reflowed[-1], block):
            _merge_paragraph_block(reflowed[-1], block)
        else:
            reflowed.append(copy.deepcopy(block))
    return reflowed


def _paragraph_continuation(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page") or 1) != int(right.get("page") or 1):
        return False
    if not _reflowable_paragraph(left) or not _reflowable_paragraph(right):
        return False
    left_box, right_box = left.get("bbox"), right.get("bbox")
    if not all(isinstance(box, (list, tuple)) and len(box) == 4 for box in (left_box, right_box)):
        return False
    line_height = max(1.0, float(left_box[3]) - float(left_box[1]))
    vertical_gap = float(right_box[1]) - float(left_box[3])
    if vertical_gap < -2 or vertical_gap > min(13.0, line_height * 0.8):
        return False
    right_text = _compact_text(right.get("text"))
    if _LIST_ITEM_RE.match(right_text) or _PARAGRAPH_HEADING_RE.match(right_text):
        return False
    if _standalone_subheading(left):
        return False
    return float(right_box[0]) <= float(left_box[0]) + 4


def _reflowable_paragraph(block: dict[str, Any]) -> bool:
    if str(block.get("kind") or "paragraph") != "paragraph" or str(block.get("role") or "body") != "body":
        return False
    if block.get("source_table_ref"):
        return False
    text = _compact_text(block.get("text"))
    return bool(
        text
        and not _PARAGRAPH_HEADING_RE.match(text)
        and not _STATEMENT_CAPTION_RE.fullmatch(text)
        and not _PERIOD_CAPTION_RE.fullmatch(text)
        and not text.startswith(("编制单位", "法定代表人", "主管会计工作负责人"))
    )


def _merge_paragraph_block(target: dict[str, Any], source: dict[str, Any]) -> None:
    left = str(target.get("text") or "")
    right = str(source.get("text") or "")
    separator = " " if left[-1:].isascii() and left[-1:].isalnum() and right[:1].isascii() and right[:1].isalnum() else ""
    target["text"] = f"{left}{separator}{right}"
    target["bbox"] = _bbox_union(target.get("bbox"), source.get("bbox"))
    for key in ("evidence_refs", "source_refs"):
        values = [str(value) for value in [*(target.get(key) or []), *(source.get(key) or [])] if value]
        if values:
            target[key] = list(dict.fromkeys(values))


def _standalone_subheading(block: dict[str, Any]) -> bool:
    text = _compact_text(block.get("text"))
    bbox = block.get("bbox")
    return bool(
        _ENUMERATED_PREFIX_RE.match(text)
        and text[-1:] not in "。！？；："
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and float(bbox[2]) - float(bbox[0]) < 400
    )


def _bbox_union(left: Any, right: Any) -> list[float]:
    return [
        min(float(left[0]), float(right[0])),
        min(float(left[1]), float(right[1])),
        max(float(left[2]), float(right[2])),
        max(float(left[3]), float(right[3])),
    ]


def _running_headers(blocks: list[dict[str, Any]]) -> set[str]:
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    top_by_text: dict[str, bool] = defaultdict(bool)
    for block in blocks:
        text = re.sub(r"\s+", "", normalize_audit_text(block.get("text")))
        bbox = block.get("bbox")
        if not text or len(text) > 60 or str(block.get("kind") or "") == "physical_table":
            continue
        pages_by_text[text].add(max(1, int(block.get("page") or 1)))
        top_by_text[text] = top_by_text[text] or bool(
            isinstance(bbox, list) and len(bbox) == 4 and float(bbox[1]) <= 80
        )
    return {text for text, pages in pages_by_text.items() if len(pages) >= 3 and top_by_text[text]}


def _printed_page_number(block: dict[str, Any], text: str, *, page_height: float | None = None) -> bool:
    if not _PAGE_NUMBER_RE.fullmatch(text.replace(" ", "")):
        return False
    bbox = block.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return True
    threshold = page_height * 0.9 if page_height else 700.0
    return float(bbox[1]) >= threshold


def _display_audit_text(value: Any) -> str:
    return normalize_audit_display_value(value)


def _normalize_display_glyphs(text: str) -> str:
    return "".join(normalize_audit_text(char) if 0x2E80 <= ord(char) <= 0x2FFF else char for char in text)


def _strip_synthetic_preamble(markdown: str) -> str:
    """Keep renderer profile markers while letting page-one source blocks own the visible title."""

    page_start = markdown.find("<!-- docmirror:page ")
    if page_start < 0:
        return markdown
    markers = re.findall(r"<!-- docmirror:(?:markdown-profile|reading-profile|audit-reading)[^>]*-->", markdown[:page_start])
    return "\n\n".join([*markers, markdown[page_start:].strip()]) + "\n"


def _collapse_rendered_statement_titles(markdown: str) -> str:
    """Collapse adjacent source nodes that jointly repeat one statement caption."""

    for title in ("资产负债表", "利润表", "现金流量表", "所有者权益变动表", "股东权益变动表"):
        pattern = re.compile(rf"(?:{title}(?:（续）)?\s*){{2,}}")
        markdown = pattern.sub(
            lambda match: f"{title}（续）\n\n" if "续" in match.group(0) else f"{title}\n\n",
            markdown,
        )
    return markdown


def _collapse_repeated_statement_title(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    for title in ("资产负债表", "利润表", "现金流量表", "所有者权益变动表", "股东权益变动表"):
        if compact.count(title) < 2:
            continue
        remainder = compact.replace(title, "")
        if re.sub(r"[()（）续]", "", remainder):
            continue
        return f"{title}（续）" if "续" in remainder else title
    parts = text.split()
    if len(parts) > 1 and len(set(parts)) == 1:
        return parts[0]
    return text


def _dataset_display_value(
    *,
    dataset_name: str,
    column: dict[str, Any],
    raw: dict[str, Any],
    normalized: dict[str, Any],
) -> str:
    key = str(column.get("key") or "")
    value = raw[key] if key in raw else normalized.get(key)
    displayed = _display_audit_text(value)
    if dataset_name.startswith("owners_equity_changes") and key == "item":
        displayed = re.sub(r"^-(?=\d+[.、])", "", displayed)
    return displayed


def _low_confidence_fragment(block: dict[str, Any], text: str, *, threshold: float = 0.55) -> bool:
    confidence = block.get("confidence")
    try:
        low = float(confidence) < threshold
    except (TypeError, ValueError):
        low = False
    return low and len(text) <= 80 and str(block.get("kind") or "") not in {"image", "figure", "physical_table"}


def _overlaps_existing(bbox: Any, existing: list[list[float]]) -> bool:
    if not existing:
        return False
    if not isinstance(bbox, list) or len(bbox) != 4:
        return any(not candidate for candidate in existing)
    return any(candidate and _intersection_ratio(bbox, candidate) >= 0.8 for candidate in existing)


def _intersection_ratio(left: list[float], right: list[float]) -> float:
    x0, y0 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    x1, y1 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    minimum = min(left_area, right_area)
    return intersection / minimum if minimum else 0.0


def _unusable_landscape_table(table: dict[str, Any]) -> bool:
    rows = [row for row in table.get("rows") or [] if isinstance(row, list)]
    width = max([len(table.get("headers") or []), *(len(row) for row in rows)], default=0)
    if width < 8:
        return False
    values = [str(value or "") for row in rows for value in row if str(value or "")]
    semantic = sum(bool(_SEMANTIC_TABLE_MARKERS.search(normalize_audit_text(value))) for value in values)
    confidences = []
    for row_model in table.get("row_models") or []:
        if not isinstance(row_model, dict):
            continue
        for cell in row_model.get("cells") or []:
            if isinstance(cell, dict) and cell.get("text") and cell.get("confidence") is not None:
                try:
                    confidences.append(float(cell["confidence"]))
                except (TypeError, ValueError):
                    pass
    average_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    characters = "".join(values)
    cjk = sum("\u3400" <= char <= "\u9fff" for char in characters)
    cjk_ratio = cjk / max(1, sum(not char.isspace() for char in characters))
    return semantic < 2 and average_confidence < 0.65 and cjk_ratio < 0.15


def _appendix_pages(structure: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    for section in structure.get("sections") or []:
        if not isinstance(section, dict) or section.get("type") != "appendix":
            continue
        page_range = section.get("page_range") or []
        if len(page_range) == 2:
            pages.update(range(max(1, int(page_range[0])), max(1, int(page_range[1])) + 1))
    return pages


def _preserve_page_sequence(blocks: list[dict[str, Any]], page_count: int) -> list[dict[str, Any]]:
    present = Counter(max(1, int(block.get("page") or 1)) for block in blocks)
    projected = list(blocks)
    for page in range(1, page_count + 1):
        if present[page]:
            continue
        projected.append(
            {
                "id": f"audit:blank-page:{page:04d}",
                "kind": "paragraph",
                "role": "artifact",
                "page": page,
                "order": page * 1_000_000,
                "text": "",
            }
        )
    return projected


__all__ = ["prepare_audit_reading_semantic", "render_audit_reading_markdown"]
