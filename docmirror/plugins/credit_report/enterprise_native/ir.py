# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical connected-component IR for digital enterprise credit reports.

The IR is the only input accepted by the enterprise business-schema decoder.
It contains copied text/table facts and scored continuation decisions; it does
not retain or proxy the source ``ParseResult``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from docmirror.plugins.credit_report.enterprise_native.continuation import TableFragment
from docmirror.plugins.credit_report.enterprise_native.input_quality import (
    assess_enterprise_parse_result,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    CreditReportUnit,
    decode_credit_report_entities,
)


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


def _segment_payload(unit: CreditReportUnit) -> dict[str, Any]:
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
    return payload


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
    pages: tuple[CanonicalEnterpriseSourcePage, ...]
    full_text: str
    source_page_count: int
    confidence: float
    input_quality_flags: tuple[dict[str, Any], ...]

    def to_debug_payload(self) -> dict[str, Any]:
        """Return the preserved JSON-safe IR used by regression diagnostics."""
        return {
            "schema": {"id": self.schema_id, "version": self.schema_version},
            "source_page_count": self.source_page_count,
            "source_pages": [page.page_number for page in self.pages],
            "component_count": len(self.components),
            "confidence": self.confidence,
            "input_quality_flags": [dict(flag) for flag in self.input_quality_flags],
            "content_conserved": self.entity_context.content_conserved,
            "unassigned_component_ids": list(self.entity_context.unassigned_unit_ids),
            "full_text": self.full_text,
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

    input_quality_flags = tuple(
        flag.to_payload() for flag in assess_enterprise_parse_result(parse_result)
    )
    context = decode_credit_report_entities(parse_result, report_family="enterprise")
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
                segments=tuple(_segment_payload(unit) for unit in units),
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
    source_pages = {unit.page for unit in context.units if unit.page > 0}
    confidence = min((entity.confidence for entity in context.entities), default=1.0)
    return CanonicalEnterpriseDocumentIR(
        schema_id="canonical_enterprise_document_ir",
        schema_version="1.0.0",
        components=tuple(components),
        connections=tuple(connections),
        continuation_decisions=tuple(_decision_payload(decision) for decision in context.decisions),
        table_rows=table_rows,
        page_texts=MappingProxyType(page_texts),
        table_headings=MappingProxyType(_table_headings(context)),
        page_flow=context.ordered_page_flow(),
        continuation_fragments=_continuation_fragments(context),
        entity_context=context,
        pages=tuple(
            CanonicalEnterpriseSourcePage(page_number=page, source_page_number=page)
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
