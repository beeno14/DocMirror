# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical connected-component IR for digital enterprise credit reports.

The IR is the only input accepted by the enterprise business-schema decoder.
It contains copied text/table facts and scored continuation decisions; it does
not retain or proxy the source ``ParseResult``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from docmirror.plugins.credit_report.enterprise_native.continuation import TableFragment
from docmirror.plugins.credit_report.enterprise_native.header_visual_recovery import (
    recover_enterprise_header_visual_fields,
)
from docmirror.plugins.credit_report.enterprise_native.input_quality import (
    assess_enterprise_parse_result,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    CreditReportUnit,
    decode_credit_report_units,
)

_PAGE_NUMBER_RE = re.compile(
    r"^(?:\u7b2c\s*\d+\s*\u9875(?:\s*[,\uff0c]\s*\u5171\s*\d+\s*\u9875)?|page\s*\d+\s*(?:of|/)\s*\d+)$",
    re.IGNORECASE,
)
_LEDGER_RE = re.compile(r"^\s*\d{1,4}\s*(?:(?:19|20)\d{2}(?:\u5e74|[-/.]))")
_HEADING_ROLES = frozenset({"heading", "title", "section_title", "h1", "h2", "h3"})
_NOISE_ROLES = frozenset({"header", "footer", "watermark"})
_CANONICAL_HEADINGS = frozenset(
    {
        "\u62a5\u544a\u4fe1\u606f",
        "\u8bf4\u660e",
        "\u8eab\u4efd\u6807\u8bc6",
        "\u4fe1\u606f\u6982\u8981",
        "\u57fa\u672c\u4fe1\u606f",
        "\u4fe1\u8d37\u8bb0\u5f55",
        "\u4fe1\u8d37\u8bb0\u5f55\u660e\u7ec6",
        "\u76f8\u5173\u8fd8\u6b3e\u8d23\u4efb\u4fe1\u606f",
        "\u975e\u4fe1\u8d37\u4ea4\u6613\u8bb0\u5f55",
        "\u516c\u5171\u8bb0\u5f55",
        "\u516c\u5171\u8bb0\u5f55\u660e\u7ec6",
        "\u67e5\u8be2\u8bb0\u5f55",
        "\u673a\u6784\u67e5\u8be2\u8bb0\u5f55\u660e\u7ec6",
        "\u4fe1\u7528\u8bb0\u5f55\u8865\u5145\u4fe1\u606f",
        "\u9644\u4ef6",
    }
)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _positive_int(value: Any, fallback: int = 0) -> int:
    try:
        candidate = int(value or 0)
    except (TypeError, ValueError):
        candidate = 0
    return candidate if candidate > 0 else fallback


def _finite(value: Any) -> float:
    try:
        candidate = float(value or 0.0)
    except (TypeError, ValueError):
        candidate = 0.0
    return candidate if math.isfinite(candidate) else 0.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = _value(value, "bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    x0, y0, x1, y1 = (_finite(item) for item in raw[:4])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _cell_text(cell: Any) -> str:
    cleaned = _value(cell, "cleaned")
    value = cleaned if cleaned not in (None, "") else _value(cell, "text", cell)
    return str(value or "").replace("\n", "").strip()


def _physical_rows(table: Any) -> tuple[tuple[str, ...], ...]:
    metadata = _value(table, "metadata") or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, Mapping) else None
    if isinstance(raw_rows, (list, tuple)) and raw_rows:
        return tuple(
            tuple(str(value or "").replace("\n", "").strip() for value in row)
            for row in raw_rows
            if isinstance(row, (list, tuple))
        )
    rows: list[tuple[str, ...]] = []
    headers = tuple(
        str(value or "").replace("\n", "").strip()
        for value in (_value(table, "headers") or ())
    )
    if headers:
        rows.append(headers)
    for row in _value(table, "rows") or ():
        cells = _value(row, "cells") or ()
        rows.append(tuple(_cell_text(cell) for cell in cells))
    return tuple(rows)


def _logical_rows(table: Any) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    headers = tuple(
        str(value or "").replace("\n", "").strip()
        for value in (_value(table, "headers") or ())
    )
    if headers:
        rows.append(headers)
    for row in _value(table, "rows") or ():
        cells = _value(row, "cells") or ()
        rows.append(tuple(_cell_text(cell) for cell in cells))
    return tuple(rows)


def _unit_kind(text: str, source: Any = None, *, flow_type: str = "") -> str:
    role = str(_value(source, "role", "") or "").strip().lower()
    level = str(_value(source, "level", "") or "").strip().lower()
    compact = _compact(text).strip(":\uff1a")
    if (
        flow_type == "heading"
        or role in _HEADING_ROLES
        or level in _HEADING_ROLES
        or compact in _CANONICAL_HEADINGS
        or any(
            compact.startswith(marker) and len(compact) <= len(marker) + 12
            for marker in _CANONICAL_HEADINGS
        )
    ):
        return "heading"
    if _LEDGER_RE.match(compact):
        return "ledger"
    return "text"


def _is_page_furniture(text: str) -> bool:
    return bool(_PAGE_NUMBER_RE.fullmatch(_compact(text).lower()))


def _make_unique(value: str, used: set[str], suffix: str) -> str:
    candidate = value or suffix
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{candidate}@{suffix}:{index}" in used:
        index += 1
    unique = f"{candidate}@{suffix}:{index}"
    used.add(unique)
    return unique


def _bbox_payload(value: tuple[float, float, float, float] | None) -> list[float] | None:
    return list(value) if value is not None else None


def _decision_payload(decision: Any) -> dict[str, Any]:
    return {
        "left_component_id": str(decision.left_unit_id),
        "right_component_id": str(decision.right_unit_id),
        "from_page": int(decision.from_page),
        "to_page": int(decision.to_page),
        "selected": str(decision.selected),
        "confidence": float(decision.confidence),
        "continues_component": bool(decision.continues_entity),
        "hypotheses": [
            {
                "kind": str(hypothesis.action),
                "score": float(hypothesis.score),
                "signals": list(hypothesis.signals),
            }
            for hypothesis in decision.hypotheses
        ],
    }


def _segment_payload(
    unit: CreditReportUnit,
    origin: CanonicalEnterpriseSourceUnit | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": unit.unit_id,
        "kind": unit.kind,
        "source_page": unit.page,
        "source_order": unit.order,
        "source_index": unit.source_index,
        "bbox": _bbox_payload(unit.bbox),
        "text": unit.text,
    }
    if unit.table_id:
        payload["table_id"] = unit.table_id
    if unit.rows:
        payload["rows"] = [list(row) for row in unit.rows]
    if origin is not None:
        payload.update(
            {
                "source_id": origin.source_id,
                "source_view": origin.source_view,
                "page_instance": origin.page_instance,
            }
        )
        if origin.key or origin.value:
            payload["key"] = origin.key
            payload["value"] = origin.value
    return payload


@dataclass(frozen=True)
class CanonicalEnterpriseSourceUnit:
    """One losslessly copied source view and its canonical disposition.

    Multiple ParseResult views may describe the same fact.  ``canonical_unit_id``
    identifies the unit fed to reconstruction, while ``duplicate_of`` records an
    alternate representation without decoding the business value twice.
    """

    source_id: str
    source_view: str
    kind: str
    source_page: int
    page_instance: int
    source_order: int
    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    key: str = ""
    value: str = ""
    table_id: str = ""
    canonical_unit_id: str = ""
    duplicate_of: str = ""
    disposition: str = "canonical_unit"

    @property
    def represented(self) -> bool:
        return bool(self.canonical_unit_id or self.duplicate_of) or self.disposition in {
            "logical_table_view",
            "explicit_furniture",
            "structural_reference",
        }

    def to_debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "source_view": self.source_view,
            "kind": self.kind,
            "source_page": self.source_page,
            "page_instance": self.page_instance,
            "source_order": self.source_order,
            "disposition": self.disposition,
        }
        if self.text:
            payload["text"] = self.text
        if self.rows:
            payload["rows"] = [list(row) for row in self.rows]
        if self.key or self.value:
            payload["key"] = self.key
            payload["value"] = self.value
        if self.table_id:
            payload["table_id"] = self.table_id
        if self.canonical_unit_id:
            payload["canonical_unit_id"] = self.canonical_unit_id
        if self.duplicate_of:
            payload["duplicate_of"] = self.duplicate_of
        return payload


@dataclass(frozen=True)
class CanonicalEnterpriseLogicalTable:
    """Page-free composed table copied from ``ParseResult.logical_tables``."""

    logical_table_id: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source_physical_ids: tuple[str, ...]
    source_pages: tuple[int, ...]
    row_source_pages: tuple[int, ...]
    confidence: float
    merge_confidence: float
    quality_passed: bool
    source_unit_ids: tuple[str, ...] = ()

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "logical_table_id": self.logical_table_id,
            "headers": list(self.headers),
            "rows": [list(row) for row in self.rows],
            "source_physical_ids": list(self.source_physical_ids),
            "source_pages": list(self.source_pages),
            "row_source_pages": list(self.row_source_pages),
            "confidence": self.confidence,
            "merge_confidence": self.merge_confidence,
            "quality_passed": self.quality_passed,
            "source_unit_ids": list(self.source_unit_ids),
        }


@dataclass(frozen=True)
class _CollectedEnterpriseSource:
    units: tuple[CreditReportUnit, ...]
    source_units: tuple[CanonicalEnterpriseSourceUnit, ...]
    logical_tables: tuple[CanonicalEnterpriseLogicalTable, ...]
    furniture_unit_ids: tuple[str, ...]
    source_pages: tuple[int, ...]
    page_modes: Mapping[int, str]
    origins: Mapping[str, CanonicalEnterpriseSourceUnit]
    recovery_quality_flags: tuple[dict[str, Any], ...] = ()


def _flow_inventory(parse_result: Any) -> tuple[dict[str, int], list[Any]]:
    graph = _value(parse_result, "document_flow")
    nodes = list(_value(graph, "nodes") or ())
    ranks: dict[str, int] = {}
    rank = 0
    flows = list(_value(graph, "reading_flow") or ())
    preferred = [
        flow
        for flow in flows
        if str(_value(flow, "type", "") or "") == "main_reading_order"
    ]
    for flow in preferred or flows[:1]:
        for node_id in _value(flow, "node_ids") or ():
            key = str(node_id or "")
            if key and key not in ranks:
                rank += 1
                ranks[key] = rank
    for node in sorted(
        nodes,
        key=lambda value: (
            _positive_int(_value(value, "page"), 1),
            _positive_int(_value(value, "reading_order"), 1_000_000),
            str(_value(value, "node_id", "") or ""),
        ),
    ):
        node_id = str(_value(node, "node_id", "") or "")
        if node_id and node_id not in ranks:
            rank += 1
            ranks[node_id] = rank
    return ranks, nodes


def _source_page(page: Any, fallback: int) -> int:
    return _positive_int(
        _value(page, "source_page_number"),
        _positive_int(_value(page, "page_number"), fallback),
    )


def _logical_table_view(table: Any, index: int, used_ids: set[str]) -> CanonicalEnterpriseLogicalTable:
    raw_id = str(_value(table, "logical_id") or _value(table, "table_id") or f"logical_table:{index}")
    logical_id = _make_unique(raw_id, used_ids, f"logical:{index}")
    rows = _logical_rows(table)
    headers = tuple(
        str(value or "").replace("\n", "").strip()
        for value in (_value(table, "headers") or ())
    )
    source_pages = tuple(
        dict.fromkeys(
            _positive_int(value)
            for value in (_value(table, "source_pages") or ())
            if _positive_int(value)
        )
    )
    provenance = list(_value(table, "provenance") or ())
    row_pages = tuple(
        _positive_int(_value(item, "source_page"))
        for item in provenance
        if _positive_int(_value(item, "source_page"))
    )
    if not source_pages and row_pages:
        source_pages = tuple(dict.fromkeys(row_pages))
    if not source_pages:
        page_span = _value(table, "page_span") or ()
        if isinstance(page_span, (list, tuple)) and page_span:
            first = _positive_int(page_span[0])
            last = _positive_int(page_span[-1], first)
            if first:
                source_pages = tuple(range(first, max(first, last) + 1))
    first_page = source_pages[0] if source_pages else 1
    if headers:
        row_pages = (first_page, *row_pages)
    if len(row_pages) < len(rows):
        row_pages = (*row_pages, *((source_pages[-1] if source_pages else first_page,) * (len(rows) - len(row_pages))))
    return CanonicalEnterpriseLogicalTable(
        logical_table_id=logical_id,
        headers=headers,
        rows=rows,
        source_physical_ids=tuple(
            str(value or "") for value in (_value(table, "source_physical_ids") or ()) if str(value or "")
        ),
        source_pages=source_pages,
        row_source_pages=tuple(row_pages[: len(rows)]),
        confidence=_finite(_value(table, "confidence", 1.0)) or 1.0,
        merge_confidence=_finite(_value(table, "merge_confidence", 1.0)) or 1.0,
        quality_passed=bool(_value(table, "quality_passed", True)),
    )


def _collect_enterprise_source(parse_result: Any) -> _CollectedEnterpriseSource:
    """Copy every business-bearing ParseResult view into an immutable ledger."""
    flow_ranks, flow_nodes = _flow_inventory(parse_result)
    candidates: dict[str, dict[str, Any]] = {}
    source_entries: list[dict[str, Any]] = []
    aliases: dict[str, list[str]] = {}
    used_unit_ids: set[str] = set()
    used_table_ids: set[str] = set()
    used_source_ids: set[str] = set()
    page_modes: dict[int, str] = {}
    source_pages: set[int] = set()
    furniture_ids: list[str] = []

    def register_alias(alias: str, unit_id: str) -> None:
        if alias:
            aliases.setdefault(alias, []).append(unit_id)

    def add_source(
        *,
        source_id: str,
        source_view: str,
        kind: str,
        source_page: int,
        page_instance: int,
        source_order: int,
        text: str = "",
        rows: tuple[tuple[str, ...], ...] = (),
        key: str = "",
        value: str = "",
        table_id: str = "",
        canonical_unit_id: str = "",
        duplicate_of: str = "",
        disposition: str = "canonical_unit",
    ) -> str:
        unique_source_id = _make_unique(source_id, used_source_ids, f"source:{len(source_entries) + 1}")
        source_entries.append(
            {
                "source_id": unique_source_id,
                "source_view": source_view,
                "kind": kind,
                "source_page": source_page,
                "page_instance": page_instance,
                "source_order": source_order,
                "text": text,
                "rows": rows,
                "key": key,
                "value": value,
                "table_id": table_id,
                "canonical_unit_id": canonical_unit_id,
                "duplicate_of": duplicate_of,
                "disposition": disposition,
            }
        )
        return unique_source_id

    def add_candidate(
        *,
        unit_id: str,
        source_view: str,
        kind: str,
        source_page: int,
        page_instance: int,
        source_index: int,
        text: str,
        rows: tuple[tuple[str, ...], ...] = (),
        table_id: str = "",
        bbox: tuple[float, float, float, float] | None = None,
        page_width: float = 0.0,
        page_height: float = 0.0,
        reading_order: int = 0,
        flow_rank: int = 0,
        source_id: str = "",
        source_key: str = "",
        source_value: str = "",
    ) -> str:
        unique_unit_id = _make_unique(unit_id, used_unit_ids, f"unit:{len(candidates) + 1}")
        top = bbox[1] if bbox is not None else float(source_index)
        candidates[unique_unit_id] = {
            "unit_id": unique_unit_id,
            "source_view": source_view,
            "kind": kind,
            "source_page": source_page,
            "page_instance": page_instance,
            "source_index": source_index,
            "text": text,
            "rows": rows,
            "table_id": table_id,
            "bbox": bbox,
            "page_width": page_width,
            "page_height": page_height,
            "reading_order": reading_order,
            "flow_rank": flow_rank,
            "top": top,
        }
        source_id = add_source(
            source_id=source_id or unique_unit_id,
            source_view=source_view,
            kind=kind,
            source_page=source_page,
            page_instance=page_instance,
            source_order=reading_order or source_index,
            text=text,
            rows=rows,
            key=source_key,
            value=source_value,
            table_id=table_id,
            canonical_unit_id=unique_unit_id,
        )
        candidates[unique_unit_id]["source_id"] = source_id
        register_alias(unique_unit_id, unique_unit_id)
        return unique_unit_id

    pages = list(_value(parse_result, "pages") or ())
    for page_instance, page in enumerate(pages, start=1):
        source_page = _source_page(page, page_instance)
        source_pages.add(source_page)
        page_modes.setdefault(source_page, str(_value(page, "page_mode", "native_text") or "native_text"))
        width = _finite(_value(page, "width"))
        height = _finite(_value(page, "height"))
        page_tag = f"lp{page_instance:04d}:sp{source_page:04d}"

        for text_index, block in enumerate(_value(page, "texts") or ()):
            content = str(_value(block, "content", "") or "").strip()
            if not content:
                continue
            source_id = f"physical_text:{page_tag}:{text_index:04d}"
            if _is_page_furniture(content):
                furniture_ids.append(source_id)
                add_source(
                    source_id=source_id,
                    source_view="physical_text",
                    kind="text",
                    source_page=source_page,
                    page_instance=page_instance,
                    source_order=text_index,
                    text=content,
                    disposition="explicit_furniture",
                )
                continue
            unit_id = add_candidate(
                unit_id=f"text:{page_tag}:{text_index:04d}",
                source_view="physical_text",
                kind=_unit_kind(content, block),
                source_page=source_page,
                page_instance=page_instance,
                source_index=text_index,
                text=content,
                bbox=_bbox(block),
                page_width=width,
                page_height=height,
                reading_order=_positive_int(_value(block, "reading_order")),
                source_id=source_id,
            )
            register_alias(f"text:p{source_page}:{text_index}", unit_id)
            for evidence_id in _value(block, "evidence_ids") or ():
                register_alias(str(evidence_id or ""), unit_id)

        for table_index, table in enumerate(_value(page, "tables") or ()):
            rows = _physical_rows(table)
            raw_table_id = str(_value(table, "table_id", "") or f"p{source_page}:t{table_index}")
            table_id = _make_unique(raw_table_id, used_table_ids, page_tag)
            if rows:
                unit_id = add_candidate(
                    unit_id=f"table:{page_tag}:{table_index:04d}",
                    source_view="physical_table",
                    kind="table",
                    source_page=source_page,
                    page_instance=page_instance,
                    source_index=table_index,
                    text="\n".join(" | ".join(row) for row in rows),
                    rows=rows,
                    table_id=table_id,
                    bbox=_bbox(table),
                    page_width=width,
                    page_height=height,
                    reading_order=_positive_int(_value(table, "reading_order")),
                    source_id=f"physical_table:{page_tag}:{table_index:04d}",
                )
                register_alias(raw_table_id, unit_id)
                register_alias(f"table:{raw_table_id}", unit_id)
                for evidence_id in _value(table, "evidence_ids") or ():
                    register_alias(str(evidence_id or ""), unit_id)
            else:
                add_source(
                    source_id=f"physical_table:{page_tag}:{table_index:04d}",
                    source_view="physical_table",
                    kind="table",
                    source_page=source_page,
                    page_instance=page_instance,
                    source_order=table_index,
                    table_id=table_id,
                    disposition="structural_reference",
                )
            caption = str(_value(table, "caption", "") or "").strip()
            if caption:
                matches = [
                    item["unit_id"]
                    for item in candidates.values()
                    if item["source_page"] == source_page and _compact(item["text"]) == _compact(caption)
                ]
                if matches:
                    add_source(
                        source_id=f"table_caption:{page_tag}:{table_index:04d}",
                        source_view="table_caption",
                        kind="caption",
                        source_page=source_page,
                        page_instance=page_instance,
                        source_order=table_index,
                        text=caption,
                        table_id=table_id,
                        duplicate_of=matches[0],
                        disposition="duplicate_view",
                    )
                else:
                    add_candidate(
                        unit_id=f"caption:{page_tag}:{table_index:04d}",
                        source_view="table_caption",
                        kind="text",
                        source_page=source_page,
                        page_instance=page_instance,
                        source_index=table_index,
                        text=caption,
                        bbox=_bbox(table),
                        page_width=width,
                        page_height=height,
                        reading_order=max(1, _positive_int(_value(table, "reading_order")) - 1),
                        source_id=f"table_caption:{page_tag}:{table_index:04d}",
                    )

        for kv_index, pair in enumerate(_value(page, "key_values") or ()):
            key = str(_value(pair, "key", "") or "").strip()
            value = str(_value(pair, "value", "") or "").strip()
            if not key and not value:
                continue
            content = "\n".join(item for item in (key, value) if item)
            unit_id = add_candidate(
                unit_id=f"key_value:{page_tag}:{kv_index:04d}",
                source_view="key_value",
                kind="text",
                source_page=source_page,
                page_instance=page_instance,
                source_index=kv_index,
                text=content,
                bbox=_bbox(pair),
                page_width=width,
                page_height=height,
                reading_order=_positive_int(_value(pair, "reading_order")),
                source_id=f"key_value:{page_tag}:{kv_index:04d}",
                source_key=key,
                source_value=value,
            )
            for evidence_id in _value(pair, "evidence_ids") or ():
                register_alias(str(evidence_id or ""), unit_id)

    header_recovery = recover_enterprise_header_visual_fields(
        parse_result,
        existing_text="\n".join(
            str(item.get("text") or "") for item in candidates.values() if item.get("text")
        ),
    )
    for recovery_index, field in enumerate(header_recovery.fields, start=1):
        label = {"report_number": "\u62a5\u544a\u7f16\u53f7"}.get(
            field.field_name,
            field.field_name,
        )
        source_pages.add(field.source_page)
        page_modes.setdefault(field.source_page, "native_text")
        add_candidate(
            unit_id=f"visual_header:{field.field_name}:p{field.source_page}",
            source_view="bounded_header_visual_recovery",
            kind="text",
            source_page=field.source_page,
            page_instance=1,
            source_index=recovery_index,
            text=f"{label}\uff1a{field.value}",
            bbox=field.bbox,
            reading_order=recovery_index,
            source_id=f"bounded_header_visual_recovery:{field.field_name}:p{field.source_page}",
            source_key=label,
            source_value=field.value,
        )

    logical_views: list[CanonicalEnterpriseLogicalTable] = []
    used_logical_ids: set[str] = set()
    logical_aliases: dict[str, str] = {}
    for logical_index, table in enumerate(_value(parse_result, "logical_tables") or (), start=1):
        view = _logical_table_view(table, logical_index, used_logical_ids)
        source_pages.update(view.source_pages)
        matched_ids: list[str] = []
        for physical_id in view.source_physical_ids:
            matched_ids.extend(aliases.get(physical_id, ()))
            matched_ids.extend(aliases.get(f"table:{physical_id}", ()))
        matched_ids = list(dict.fromkeys(matched_ids))
        source_id = f"logical_table:{view.logical_table_id}"
        canonical_unit_id = ""
        disposition = "logical_table_view"
        if view.rows and not matched_ids:
            page = view.source_pages[0] if view.source_pages else 1
            source_pages.add(page)
            table_id = _make_unique(view.logical_table_id, used_table_ids, f"logical:{logical_index}")
            canonical_unit_id = add_candidate(
                unit_id=f"logical_table_unit:{logical_index:04d}",
                source_view="logical_table",
                kind="table",
                source_page=page,
                page_instance=0,
                source_index=logical_index,
                text="\n".join(" | ".join(row) for row in view.rows),
                rows=view.rows,
                table_id=table_id,
                reading_order=logical_index,
                source_id=source_id,
            )
            disposition = "canonical_unit"
        else:
            add_source(
                source_id=source_id,
                source_view="logical_table",
                kind="logical_table",
                source_page=view.source_pages[0] if view.source_pages else 0,
                page_instance=0,
                source_order=logical_index,
                rows=view.rows,
                table_id=view.logical_table_id,
                duplicate_of=matched_ids[0] if matched_ids else "",
                disposition=disposition,
            )
        view = CanonicalEnterpriseLogicalTable(
            **{
                **view.__dict__,
                "source_unit_ids": tuple(matched_ids or ([canonical_unit_id] if canonical_unit_id else [])),
            }
        )
        logical_views.append(view)
        logical_aliases[view.logical_table_id] = source_id
        for alias in (
            str(_value(table, "logical_id", "") or ""),
            str(_value(table, "table_id", "") or ""),
        ):
            if alias:
                logical_aliases[alias] = source_id

    for node_index, node in enumerate(flow_nodes, start=1):
        node_id = str(_value(node, "node_id", "") or f"flow_node:{node_index}")
        flow_type = str(_value(node, "type", "paragraph") or "paragraph")
        role = str(_value(node, "role", "body") or "body").lower()
        source_page = _positive_int(_value(node, "page"), 1)
        source_pages.add(source_page)
        text = str(_value(node, "text", "") or "").strip()
        matched: list[str] = []
        for reference in _value(node, "fact_refs") or ():
            matched.extend(aliases.get(str(reference or ""), ()))
        metadata = _value(node, "metadata") or {}
        if isinstance(metadata, Mapping):
            for key in ("table_id", "physical_table_id", "source_table_id"):
                matched.extend(aliases.get(str(metadata.get(key) or ""), ()))
        if not matched and text:
            matched.extend(
                item["unit_id"]
                for item in candidates.values()
                if item["source_page"] == source_page and _compact(item["text"]) == _compact(text)
            )
        matched = list(dict.fromkeys(matched))
        if matched:
            for unit_id in matched:
                current = int(candidates[unit_id].get("flow_rank") or 0)
                rank = int(flow_ranks.get(node_id) or 0)
                if rank and (not current or rank < current):
                    candidates[unit_id]["flow_rank"] = rank
            add_source(
                source_id=f"document_flow:{node_id}",
                source_view="document_flow",
                kind=flow_type,
                source_page=source_page,
                page_instance=0,
                source_order=int(flow_ranks.get(node_id) or _positive_int(_value(node, "reading_order"))),
                text=text,
                duplicate_of=matched[0],
                disposition="duplicate_view",
            )
            continue
        if flow_type == "logical_table":
            logical_ref = ""
            if isinstance(metadata, Mapping):
                logical_ref = str(metadata.get("logical_id") or metadata.get("table_id") or "")
            duplicate = logical_aliases.get(logical_ref, "")
            add_source(
                source_id=f"document_flow:{node_id}",
                source_view="document_flow",
                kind=flow_type,
                source_page=source_page,
                page_instance=0,
                source_order=int(flow_ranks.get(node_id) or node_index),
                text=text,
                duplicate_of=duplicate,
                disposition="duplicate_view" if duplicate else "structural_reference",
            )
            continue
        if flow_type in {"header", "footer", "watermark"} or role in _NOISE_ROLES:
            add_source(
                source_id=f"document_flow:{node_id}",
                source_view="document_flow",
                kind=flow_type,
                source_page=source_page,
                page_instance=0,
                source_order=int(flow_ranks.get(node_id) or node_index),
                text=text,
                disposition="explicit_furniture",
            )
            continue
        if text:
            unit_id = add_candidate(
                unit_id=f"flow:{node_id}",
                source_view="document_flow",
                kind=_unit_kind(text, node, flow_type=flow_type),
                source_page=source_page,
                page_instance=0,
                source_index=node_index,
                text=text,
                bbox=_bbox(node),
                reading_order=_positive_int(_value(node, "reading_order")),
                flow_rank=int(flow_ranks.get(node_id) or 0),
                source_id=f"document_flow:{node_id}",
            )
            register_alias(node_id, unit_id)
        else:
            add_source(
                source_id=f"document_flow:{node_id}",
                source_view="document_flow",
                kind=flow_type,
                source_page=source_page,
                page_instance=0,
                source_order=int(flow_ranks.get(node_id) or node_index),
                disposition="structural_reference",
            )

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            item["source_page"],
            0 if item["flow_rank"] else 1,
            item["flow_rank"] or _positive_int(item["reading_order"], 1_000_000),
            item["page_instance"],
            item["top"],
            1 if item["kind"] == "table" else 0,
            item["source_index"],
            item["unit_id"],
        ),
    )
    units = tuple(
        CreditReportUnit(
            unit_id=item["unit_id"],
            page=item["source_page"],
            order=order,
            source_index=item["source_index"],
            kind=item["kind"],
            text=item["text"],
            bbox=item["bbox"],
            page_width=item["page_width"],
            page_height=item["page_height"],
            table_id=item["table_id"],
            rows=item["rows"],
        )
        for order, item in enumerate(ordered)
    )
    source_units = tuple(CanonicalEnterpriseSourceUnit(**entry) for entry in source_entries)
    origin_by_source = {entry.source_id: entry for entry in source_units}
    origins = {
        item["unit_id"]: origin_by_source[item["source_id"]]
        for item in ordered
        if item.get("source_id") in origin_by_source
    }
    return _CollectedEnterpriseSource(
        units=units,
        source_units=source_units,
        logical_tables=tuple(logical_views),
        furniture_unit_ids=tuple(furniture_ids),
        source_pages=tuple(sorted(page for page in source_pages if page > 0)),
        page_modes=MappingProxyType(page_modes),
        origins=MappingProxyType(origins),
        recovery_quality_flags=tuple(
            flag.to_payload() for flag in header_recovery.quality_flags
        ),
    )


@dataclass(frozen=True)
class CanonicalEnterpriseComponent:
    """One connected logical component after continuation reconstruction."""

    component_id: str
    kind: str
    global_order: int
    segment_ids: tuple[str, ...]
    source_pages: tuple[int, ...]
    confidence: float
    text: str
    rows: tuple[tuple[str, ...], ...]
    segments: tuple[dict[str, Any], ...]

    def to_debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.component_id,
            "kind": self.kind,
            "global_order": self.global_order,
            "segment_ids": list(self.segment_ids),
            "source_pages": list(self.source_pages),
            "confidence": self.confidence,
            "text": self.text,
            "segments": [dict(segment) for segment in self.segments],
        }
        if self.rows:
            payload["rows"] = [list(row) for row in self.rows]
        return payload


@dataclass(frozen=True)
class CanonicalEnterpriseSourcePage:
    """Minimal source-page inventory retained for completeness auditing."""

    page_number: int
    source_page_number: int
    page_mode: str = "native_text"
    page_instances: tuple[int, ...] = ()


@dataclass(frozen=True)
class CanonicalEnterpriseDocumentIR:
    """Pagination-independent enterprise document consumed by one schema pass."""

    schema_id: str
    schema_version: str
    components: tuple[CanonicalEnterpriseComponent, ...]
    connections: tuple[dict[str, Any], ...]
    continuation_decisions: tuple[dict[str, Any], ...]
    table_rows: tuple[tuple[int, str, tuple[tuple[str, ...], ...]], ...]
    page_texts: Mapping[int, str]
    table_headings: Mapping[str, str]
    page_flow: tuple[tuple[int, str, Any], ...]
    continuation_fragments: tuple[TableFragment, ...]
    entity_context: CreditReportEntityContext
    source_units: tuple[CanonicalEnterpriseSourceUnit, ...]
    logical_tables: tuple[CanonicalEnterpriseLogicalTable, ...]
    pages: tuple[CanonicalEnterpriseSourcePage, ...]
    full_text: str
    source_page_count: int
    confidence: float
    input_quality_flags: tuple[dict[str, Any], ...]

    @property
    def unassigned_source_unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.source_id for unit in self.source_units if not unit.represented)

    @property
    def content_conserved(self) -> bool:
        """Whether all source views and all reconstruction units are accounted for."""
        return self.entity_context.content_conserved and not self.unassigned_source_unit_ids

    @property
    def page_free_table_rows(self) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
        """Return authoritative composed tables, or reconstructed physical views."""
        usable = tuple(table for table in self.logical_tables if table.rows and table.quality_passed)
        if usable:
            return tuple((table.logical_table_id, table.rows) for table in usable)
        return tuple(
            (component.component_id, component.rows)
            for component in self.components
            if component.rows and component.kind in {"logical_table", "logical_mixed"}
        )

    def to_debug_payload(self) -> dict[str, Any]:
        """Return the preserved JSON-safe IR used by regression diagnostics."""
        return {
            "schema": {"id": self.schema_id, "version": self.schema_version},
            "source_page_count": self.source_page_count,
            "source_pages": [page.source_page_number for page in self.pages],
            "component_count": len(self.components),
            "confidence": self.confidence,
            "input_quality_flags": [dict(flag) for flag in self.input_quality_flags],
            "content_conserved": self.content_conserved,
            "unassigned_component_ids": [
                *self.entity_context.unassigned_unit_ids,
                *self.unassigned_source_unit_ids,
            ],
            "source_view_conservation": {
                "source_unit_count": len(self.source_units),
                "represented_source_unit_count": sum(unit.represented for unit in self.source_units),
                "unassigned_source_unit_ids": list(self.unassigned_source_unit_ids),
                "source_view_counts": {
                    view: sum(unit.source_view == view for unit in self.source_units)
                    for view in sorted({unit.source_view for unit in self.source_units})
                },
            },
            "full_text": self.full_text,
            "source_units": [unit.to_debug_payload() for unit in self.source_units],
            "logical_tables": [table.to_debug_payload() for table in self.logical_tables],
            "page_free_table_rows": [
                {
                    "logical_table_id": table_id,
                    "rows": [list(row) for row in rows],
                }
                for table_id, rows in self.page_free_table_rows
            ],
            "components": [component.to_debug_payload() for component in self.components],
            "connections": [dict(connection) for connection in self.connections],
            "continuation_decisions": [dict(decision) for decision in self.continuation_decisions],
            "table_rows": [
                {
                    "source_page": page,
                    "table_id": table_id,
                    "rows": [list(row) for row in rows],
                }
                for page, table_id, rows in self.table_rows
            ],
            "page_texts": {str(page): text for page, text in self.page_texts.items()},
            "table_headings": dict(self.table_headings),
            "page_flow": [
                {"source_page": page, "kind": kind, "value": value}
                for page, kind, value in self.page_flow
            ],
            "continuation_fragments": [
                {
                    "index": fragment.index,
                    "source_page": fragment.page,
                    "table_id": fragment.table_id,
                    "rows": [list(row) for row in fragment.rows],
                    "bbox": list(fragment.bbox) if fragment.bbox is not None else None,
                    "page_width": fragment.page_width,
                    "page_height": fragment.page_height,
                    "first_on_page": fragment.first_on_page,
                    "last_on_page": fragment.last_on_page,
                }
                for fragment in self.continuation_fragments
            ],
        }

    def to_debug_json(self) -> str:
        return json.dumps(self.to_debug_payload(), ensure_ascii=False, sort_keys=True)


def _component_kind(kind: str) -> str:
    return {
        "table": "logical_table",
        "text": "logical_text",
        "mixed": "logical_mixed",
    }.get(kind, "logical_component")


def _table_headings(context: CreditReportEntityContext) -> dict[str, str]:
    headings: dict[str, str] = {}
    by_page: dict[int, list[CreditReportUnit]] = {}
    for unit in context.units:
        by_page.setdefault(unit.page, []).append(unit)
    for units in by_page.values():
        ordered = sorted(units, key=lambda unit: (unit.order, unit.source_index))
        for index, unit in enumerate(ordered):
            if unit.kind != "table" or not unit.table_id:
                continue
            preceding = [candidate for candidate in ordered[:index] if candidate.kind != "table" and candidate.text]
            if preceding:
                headings[unit.table_id] = preceding[-1].text.replace("\n", "").strip()
    return headings


def _continuation_fragments(context: CreditReportEntityContext) -> tuple[TableFragment, ...]:
    table_units = [unit for unit in context.units if unit.kind == "table" and unit.rows]
    page_tables: dict[int, list[str]] = {}
    for unit in table_units:
        page_tables.setdefault(unit.page, []).append(unit.unit_id)
    fragments: list[TableFragment] = []
    for unit in table_units:
        ids = page_tables[unit.page]
        fragments.append(
            TableFragment(
                index=len(fragments),
                page=unit.page,
                table_id=unit.table_id,
                rows=unit.rows,
                bbox=unit.bbox,
                page_width=unit.page_width,
                page_height=unit.page_height,
                first_on_page=unit.unit_id == ids[0],
                last_on_page=unit.unit_id == ids[-1],
            )
        )
    return tuple(fragments)


def build_canonical_enterprise_document(parse_result: Any) -> CanonicalEnterpriseDocumentIR:
    """Score continuations and copy the report into the canonical enterprise IR."""
    if isinstance(parse_result, CanonicalEnterpriseDocumentIR):
        return parse_result

    structural_input_quality_flags = tuple(
        flag.to_payload() for flag in assess_enterprise_parse_result(parse_result)
    )
    collected = _collect_enterprise_source(parse_result)
    input_quality_flags = tuple(
        [*structural_input_quality_flags, *collected.recovery_quality_flags]
    )
    context = decode_credit_report_units(
        collected.units,
        report_family="enterprise",
        furniture_unit_ids=collected.furniture_unit_ids,
        entity_prefix="enterprise",
    )
    units_by_id = {unit.unit_id: unit for unit in context.units}
    components: list[CanonicalEnterpriseComponent] = []
    connections: list[dict[str, Any]] = []
    for global_order, entity in enumerate(context.entities, start=1):
        units = tuple(units_by_id[unit_id] for unit_id in entity.unit_ids if unit_id in units_by_id)
        rows = tuple(row for unit in units if unit.kind == "table" for row in unit.rows)
        text = "\n".join(unit.text for unit in units if unit.text).strip()
        components.append(
            CanonicalEnterpriseComponent(
                component_id=entity.entity_id,
                kind=_component_kind(entity.kind),
                global_order=global_order,
                segment_ids=tuple(unit.unit_id for unit in units),
                source_pages=entity.pages,
                confidence=float(entity.confidence),
                text=text,
                rows=rows,
                segments=tuple(
                    _segment_payload(unit, collected.origins.get(unit.unit_id))
                    for unit in units
                ),
            )
        )
        for left, right in zip(units, units[1:]):
            decision = context.decision_between(left.unit_id, right.unit_id)
            connections.append(
                {
                    "kind": "continues",
                    "from": left.unit_id,
                    "to": right.unit_id,
                    "component_id": entity.entity_id,
                    "confidence": float(decision.confidence) if decision is not None else float(entity.confidence),
                }
            )
    for left, right in zip(components, components[1:]):
        connections.append(
            {
                "kind": "next",
                "from": left.component_id,
                "to": right.component_id,
                "confidence": 1.0,
            }
        )

    table_rows = tuple(
        (unit.page, unit.table_id, unit.rows)
        for unit in context.units
        if unit.kind == "table" and unit.rows
    )
    page_text_parts: dict[int, list[str]] = {}
    for unit in context.units:
        if unit.kind != "table" and unit.text:
            page_text_parts.setdefault(unit.page, []).append(unit.text)
    page_texts = {page: "\n".join(parts) for page, parts in sorted(page_text_parts.items())}
    full_text = "\n".join(unit.text for unit in context.units if unit.text).strip()
    source_pages = set(collected.source_pages)
    source_pages.update(unit.page for unit in context.units if unit.page > 0)
    page_instances: dict[int, set[int]] = {}
    for source_unit in collected.source_units:
        if source_unit.source_page > 0 and source_unit.page_instance > 0:
            page_instances.setdefault(source_unit.source_page, set()).add(source_unit.page_instance)
    confidence = min((entity.confidence for entity in context.entities), default=1.0)
    return CanonicalEnterpriseDocumentIR(
        schema_id="canonical_enterprise_document_ir",
        schema_version="2.0.0",
        components=tuple(components),
        connections=tuple(connections),
        continuation_decisions=tuple(_decision_payload(decision) for decision in context.decisions),
        table_rows=table_rows,
        page_texts=MappingProxyType(page_texts),
        table_headings=MappingProxyType(_table_headings(context)),
        page_flow=context.ordered_page_flow(),
        continuation_fragments=_continuation_fragments(context),
        entity_context=context,
        source_units=collected.source_units,
        logical_tables=collected.logical_tables,
        pages=tuple(
            CanonicalEnterpriseSourcePage(
                page_number=page,
                source_page_number=page,
                page_mode=str(collected.page_modes.get(page) or "native_text"),
                page_instances=tuple(sorted(page_instances.get(page, ()))),
            )
            for page in sorted(source_pages)
        ),
        full_text=full_text,
        source_page_count=len(source_pages),
        confidence=round(float(confidence), 4),
        input_quality_flags=input_quality_flags,
    )


__all__ = [
    "CanonicalEnterpriseComponent",
    "CanonicalEnterpriseDocumentIR",
    "CanonicalEnterpriseSourcePage",
    "build_canonical_enterprise_document",
]
