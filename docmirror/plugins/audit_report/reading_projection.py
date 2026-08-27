"""Audit-only cleanup for the source-complete enhanced reading view."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from typing import Any

from docmirror.output.markdown_renderer import render_semantic_source_overlay_markdown
from docmirror.plugins.audit_report.table_projection import normalize_audit_text

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


def render_audit_reading_markdown(semantic: dict[str, Any]) -> str:
    """Render an audit report from a cleaned copy of its public semantic source."""

    return render_semantic_source_overlay_markdown(prepare_audit_reading_semantic(semantic))


def prepare_audit_reading_semantic(semantic: dict[str, Any]) -> dict[str, Any]:
    """Prepare a display-only source overlay without changing JSON facts."""

    projected = copy.deepcopy(semantic)
    structure = projected.get("structure") if isinstance(projected.get("structure"), dict) else {}
    blocks = [block for block in structure.get("blocks") or [] if isinstance(block, dict)]
    source_tables = [table for table in structure.get("source_tables") or [] if isinstance(table, dict)]

    _normalize_source_tables(source_tables)
    recovered_table_ids = _project_financial_tables(projected, source_tables)
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
        title=normalize_audit_text((projected.get("document") or {}).get("title")),
        unusable_table_ids=unusable_table_ids,
        unusable_pages=unusable_pages,
        appendix_pages=appendix_pages,
    )
    page_count = max(0, int((projected.get("document") or {}).get("page_count") or 0))
    blocks = _preserve_page_sequence(blocks, page_count)
    blocks.sort(key=lambda block: (int(block.get("page") or 1), int(block.get("order") or 0), str(block.get("id"))))
    for order, block in enumerate(blocks, start=1):
        block["order"] = order

    structure["blocks"] = blocks
    structure["source_tables"] = source_tables
    structure["reading_flows"] = [
        {"id": "audit:enhanced-reading", "type": "main", "node_ids": [str(block["id"]) for block in blocks]}
    ]
    valid_ids = {str(block["id"]) for block in blocks}
    for section in structure.get("sections") or []:
        if isinstance(section, dict) and isinstance(section.get("block_refs"), list):
            section["block_refs"] = [str(block_id) for block_id in section["block_refs"] if str(block_id) in valid_ids]
    return projected


def _normalize_source_tables(source_tables: list[dict[str, Any]]) -> None:
    for table in source_tables:
        table["headers"] = [normalize_audit_text(value) for value in table.get("headers") or []]
        table["rows"] = [
            [normalize_audit_text(value) for value in row] for row in table.get("rows") or [] if isinstance(row, list)
        ]


def _project_financial_tables(
    semantic: dict[str, Any],
    source_tables: list[dict[str, Any]],
) -> set[str]:
    by_id = {str(table.get("id") or ""): table for table in source_tables if table.get("id")}
    recovered: set[str] = set()
    for dataset in semantic.get("datasets") or []:
        if not isinstance(dataset, dict) or not str(dataset.get("name") or "").startswith(_FINANCIAL_DATASETS):
            continue
        columns = [column for column in dataset.get("columns") or [] if isinstance(column, dict) and column.get("key")]
        keys = [str(column["key"]) for column in columns]
        headers = [normalize_audit_text(column.get("label") or column["key"]) for column in columns]
        grouped: dict[str, list[list[str]]] = defaultdict(list)
        recovered_groups: set[str] = set()
        for row in dataset.get("rows") or []:
            if not isinstance(row, dict):
                continue
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            table_id = str(source.get("physical_table_id") or source.get("table_id") or "").split(":segment_", 1)[0]
            if table_id not in by_id:
                continue
            raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), dict) else row.get("raw")
            normalized = row.get("normalized") if isinstance(row.get("normalized"), dict) else {}
            raw = raw if isinstance(raw, dict) else {}
            grouped[table_id].append(
                [normalize_audit_text(raw[key] if key in raw else normalized.get(key)) for key in keys]
            )
            if source.get("recovery"):
                recovered_groups.add(table_id)
        for table_id, rows in grouped.items():
            table = by_id[table_id]
            table["headers"] = headers
            table["rows"] = rows
            extensions = table.get("extensions") if isinstance(table.get("extensions"), dict) else {}
            extensions["audit_reading_projection"] = "source_backed_financial_rows"
            table["extensions"] = extensions
            if table_id in recovered_groups:
                recovered.add(table_id)
    return recovered


def _curate_blocks(
    blocks: list[dict[str, Any]],
    *,
    title: str,
    unusable_table_ids: set[str],
    unusable_pages: set[int],
    appendix_pages: set[int],
) -> list[dict[str, Any]]:
    normalized_blocks = [_normalize_block(block) for block in blocks]
    repeated_headers = _running_headers(normalized_blocks)
    seen_running: set[str] = set()
    seen_same_page: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    curated: list[dict[str, Any]] = []
    for block in normalized_blocks:
        page = max(1, int(block.get("page") or 1))
        text = normalize_audit_text(block.get("text") or block.get("value"))
        compact = re.sub(r"\s+", "", text)
        table_id = str(block.get("source_table_ref") or "")
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
        if page == 1 and title and compact == re.sub(r"\s+", "", title):
            continue
        if _printed_page_number(block, text):
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
            normalized[key] = normalize_audit_text(normalized[key])
    return normalized


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


def _printed_page_number(block: dict[str, Any], text: str) -> bool:
    if not _PAGE_NUMBER_RE.fullmatch(text.replace(" ", "")):
        return False
    bbox = block.get("bbox")
    return not isinstance(bbox, list) or len(bbox) != 4 or float(bbox[1]) >= 700


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
                "kind": "key_value",
                "role": "artifact",
                "page": page,
                "order": page * 1_000_000,
                "text": "（空白页）",
                "key": "",
                "value": "（空白页）",
            }
        )
    return projected


__all__ = ["prepare_audit_reading_semantic", "render_audit_reading_markdown"]
