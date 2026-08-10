# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-world subset contract for native PBOC enterprise reports.

The published PBOC enterprise layout is the complete grammar.  A concrete
report may omit any optional section, but a populated section is decoded only
inside its canonical slice.  Physical pages are deliberately absent from the
contract: section boundaries are reconstructed from the ordered source units
in the canonical enterprise IR.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable

from docmirror.plugins.credit_report.enterprise_native.continuation import TableFragment
from docmirror.plugins.credit_report.enterprise_native.ir import (
    CanonicalEnterpriseComponent,
    CanonicalEnterpriseDocumentIR,
    CanonicalEnterpriseSourcePage,
)


@dataclass(frozen=True)
class CanonicalEnterpriseSectionSpec:
    """One ordered, optionally present section in the canonical layout."""

    key: str
    section_id: str
    title: str
    section_type: str
    aliases: tuple[str, ...]
    datasets: tuple[str, ...]


CANONICAL_ENTERPRISE_SECTIONS: tuple[CanonicalEnterpriseSectionSpec, ...] = (
    CanonicalEnterpriseSectionSpec(
        "report_metadata",
        "sec_enterprise_report_metadata",
        "报告信息",
        "report_metadata",
        ("企业信用报告", "企业信用报告（自主查询版）", "企业信用报告(自主查询版)"),
        ("enterprise_report_metadata",),
    ),
    CanonicalEnterpriseSectionSpec(
        "report_notes",
        "sec_enterprise_report_notes",
        "说明",
        "notes",
        ("报告说明",),
        ("report_notes", "enterprise_exchange_rates"),
    ),
    CanonicalEnterpriseSectionSpec(
        "identity",
        "sec_enterprise_identity",
        "身份标识",
        "identity",
        ("身份标识",),
        ("enterprise_report_identity", "enterprise_dispute_overview"),
    ),
    CanonicalEnterpriseSectionSpec(
        "information_overview",
        "sec_enterprise_summary",
        "信息概要",
        "credit_summary",
        ("信息概要",),
        (
            "enterprise_credit_overview",
            "enterprise_public_record_counts",
            "enterprise_recovery_summary",
            "enterprise_overdue_summary",
            "enterprise_current_credit_summary",
            "enterprise_facility_summary",
            "enterprise_repayment_responsibility_summary",
            "enterprise_closed_credit_summary",
        ),
    ),
    CanonicalEnterpriseSectionSpec(
        "basic_information",
        "sec_enterprise_profile",
        "基本信息",
        "basic_information",
        ("基本信息",),
        (
            "enterprise_profile",
            "enterprise_capital_summary",
            "enterprise_contributors",
            "enterprise_key_personnel",
            "enterprise_relationships",
        ),
    ),
    CanonicalEnterpriseSectionSpec(
        "credit_details",
        "sec_enterprise_credit",
        "信贷记录明细",
        "credit_details",
        ("信贷记录明细",),
        (
            "enterprise_credit_detail_groups",
            "enterprise_credit_accounts",
            "enterprise_interest_arrears",
            "enterprise_displayed_credit_summary",
            "enterprise_credit_facilities",
            "enterprise_repayment_responsibility_accounts",
            "enterprise_repayment_responsibility_group_details",
            "enterprise_account_annotations",
        ),
    ),
    CanonicalEnterpriseSectionSpec(
        "non_credit_records",
        "sec_enterprise_non_credit",
        "非信贷记录明细",
        "non_credit_records",
        ("非信贷记录明细",),
        ("enterprise_public_utility_payment_records",),
    ),
    CanonicalEnterpriseSectionSpec(
        "public_records",
        "sec_enterprise_public",
        "公共记录明细",
        "public_records",
        ("公共记录明细",),
        (
            "enterprise_public_tax_arrears_records",
            "enterprise_public_civil_judgment_records",
            "enterprise_public_enforcement_records",
            "enterprise_public_administrative_penalty_records",
            "enterprise_public_housing_fund_payment_records",
            "enterprise_public_social_security_payment_records",
            "enterprise_public_license_records",
            "enterprise_public_certification_records",
            "enterprise_public_qualification_records",
            "enterprise_public_award_records",
            "enterprise_public_export_quality_records",
            "enterprise_public_inspection_exemption_records",
            "enterprise_public_regulatory_supervision_records",
            "enterprise_public_patent_records",
            "enterprise_public_financing_restriction_records",
        ),
    ),
    CanonicalEnterpriseSectionSpec(
        "statements_and_disputes",
        "sec_enterprise_statements",
        "声明及异议标注信息",
        "statements_and_disputes",
        ("声明及异议标注信息",),
        (
            "enterprise_public_data_provider_statement_records",
            "enterprise_public_credit_bureau_statement_records",
            "enterprise_public_subject_statement_records",
            "enterprise_public_dispute_annotation_records",
        ),
    ),
    CanonicalEnterpriseSectionSpec(
        "attachment",
        "sec_enterprise_supplement",
        "附件",
        "credit_supplement",
        ("附件", "附件1：信用记录补充信息", "附件1:信用记录补充信息", "信用记录补充信息"),
        (
            "enterprise_attachment_accounts",
            "enterprise_credit_supplement",
            "enterprise_attachment_credit_details",
            "enterprise_special_transactions",
            "enterprise_utility_payment_history",
            "enterprise_housing_fund_history",
        ),
    ),
)

_SPEC_BY_KEY = MappingProxyType({spec.key: spec for spec in CANONICAL_ENTERPRISE_SECTIONS})
_SPEC_INDEX = MappingProxyType(
    {spec.key: index for index, spec in enumerate(CANONICAL_ENTERPRISE_SECTIONS)}
)


def _compact(value: Any) -> str:
    return "".join(str(value or "").split()).strip()


def _heading_candidate(value: Any) -> str:
    text = _compact(value)
    text = re.sub(
        r"^(?:(?:第?[一二三四五六七八九十]+[、.．])|(?:[（(][一二三四五六七八九十0-9]+[）)]))",
        "",
        text,
    )
    return text


def _section_key_for_line(value: Any) -> str | None:
    line = _heading_candidate(value)
    if not line:
        return None
    for spec in CANONICAL_ENTERPRISE_SECTIONS:
        if line in spec.aliases:
            return spec.key
    attachment_match = re.fullmatch(r"附件\d*[:：]?信用记录补充信息", line)
    return "attachment" if attachment_match else None


@dataclass(frozen=True)
class _OrderedSourceUnit:
    unit_id: str
    page: int
    kind: str
    text: str
    table_id: str
    rows: tuple[tuple[str, ...], ...]
    bbox: tuple[float, float, float, float] | None
    page_width: float
    page_height: float


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = list(value or ())
    if len(raw) < 4:
        return None
    try:
        return tuple(float(item) for item in raw[:4])  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _ordered_units(document: CanonicalEnterpriseDocumentIR) -> tuple[_OrderedSourceUnit, ...]:
    """Return each copied source unit once, in document order."""
    units: list[tuple[tuple[int, float, int, int], _OrderedSourceUnit]] = []
    seen: set[str] = set()
    for component in sorted(document.components, key=lambda item: item.global_order):
        for local_index, segment in enumerate(component.segments):
            unit_id = str(segment.get("id") or f"{component.component_id}:{local_index}")
            if unit_id in seen:
                continue
            seen.add(unit_id)
            page = int(segment.get("source_page") or 0)
            try:
                source_order = float(segment.get("source_order") or 0)
            except (TypeError, ValueError):
                source_order = float(local_index)
            source_index = int(segment.get("source_index") or local_index)
            rows = tuple(
                tuple(_compact(cell) for cell in row)
                for row in (segment.get("rows") or ())
                if isinstance(row, (list, tuple))
            )
            table_id = str(segment.get("table_id") or "")
            unit = _OrderedSourceUnit(
                unit_id=unit_id,
                page=page,
                kind=str(segment.get("kind") or "text"),
                text=str(segment.get("text") or ""),
                table_id=table_id,
                rows=rows,
                bbox=_bbox(segment.get("bbox")),
                page_width=float(segment.get("page_width") or 0.0),
                page_height=float(segment.get("page_height") or 0.0),
            )
            units.append(((page, source_order, source_index, component.global_order), unit))
    units.sort(key=lambda item: item[0])
    return tuple(unit for _order, unit in units)


@dataclass(frozen=True)
class CanonicalEnterpriseSectionSlice:
    """Page-free connected source units assigned to one canonical section."""

    spec: CanonicalEnterpriseSectionSpec
    heading_detected: bool
    unit_ids: tuple[str, ...]
    source_pages: tuple[int, ...]
    table_ids: tuple[str, ...]
    text: str
    page_texts: MappingProxyType
    page_flow: tuple[tuple[int, str, Any], ...]
    fragments: tuple[TableFragment, ...]

    @property
    def present(self) -> bool:
        return self.heading_detected or bool(self.table_ids or self.text.strip())


@dataclass(frozen=True)
class CanonicalEnterpriseSubset:
    """One report interpreted as an ordered subset of the canonical grammar."""

    source: CanonicalEnterpriseDocumentIR
    sections: tuple[CanonicalEnterpriseSectionSlice, ...]

    def section(self, key: str) -> CanonicalEnterpriseSectionSlice:
        return self.sections[_SPEC_INDEX[key]]

    def view(self, key: str) -> CanonicalEnterpriseDocumentIR:
        """Return an IR view containing only one canonical business section."""
        section = self.section(key)
        table_ids = set(section.table_ids)
        source_pages = set(section.source_pages)
        fragments = tuple(
            replace(
                fragment,
                index=index,
                first_on_page=not any(
                    prior.page == fragment.page for prior in section.fragments[:index]
                ),
                last_on_page=not any(
                    later.page == fragment.page for later in section.fragments[index + 1 :]
                ),
            )
            for index, fragment in enumerate(section.fragments)
        )
        table_rows = tuple(
            row for row in self.source.table_rows if len(row) >= 2 and str(row[1]) in table_ids
        )
        source_segment_queues: dict[tuple[int, str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for source_component in sorted(
            self.source.components,
            key=lambda item: item.global_order,
        ):
            for source_segment in source_component.segments:
                source_kind = str(source_segment.get("kind") or "")
                source_page = int(source_segment.get("source_page") or 0)
                source_value = (
                    str(source_segment.get("table_id") or "")
                    if source_kind == "table"
                    else str(source_segment.get("text") or "")
                )
                source_segment_queues[(source_page, source_kind, source_value)].append(
                    dict(source_segment)
                )
        segments: list[dict[str, Any]] = []
        component_rows: list[tuple[str, ...]] = []
        component_text: list[str] = []
        for flow_index, (page, kind, value) in enumerate(section.page_flow):
            lookup_value = str(value[0]) if kind == "table" else str(value)
            source_queue = source_segment_queues.get((page, kind, lookup_value))
            segment: dict[str, Any] = (
                dict(source_queue.popleft())
                if source_queue
                else {
                    "id": f"canonical_subset:{key}:{flow_index}",
                    "kind": kind,
                    "source_page": page,
                    "source_order": flow_index,
                    "source_index": flow_index,
                    "bbox": None,
                }
            )
            if kind == "table":
                table_id, rows = value
                normalized_rows = tuple(tuple(str(cell) for cell in row) for row in rows)
                segment.update(
                    {
                        "table_id": str(table_id),
                        "rows": [list(row) for row in normalized_rows],
                        "text": "\n".join(" | ".join(row) for row in normalized_rows),
                    }
                )
                component_rows.extend(normalized_rows)
            else:
                segment["text"] = str(value)
                component_text.append(str(value))
            segments.append(segment)
        component = CanonicalEnterpriseComponent(
            component_id=f"canonical_subset:{key}",
            kind=(
                "logical_mixed"
                if table_ids and component_text
                else "logical_table"
                if table_ids
                else "logical_text"
            ),
            global_order=_SPEC_INDEX[key] + 1,
            segment_ids=tuple(str(segment["id"]) for segment in segments),
            source_pages=tuple(sorted(source_pages)),
            confidence=self.source.confidence,
            text="\n".join(component_text).strip(),
            rows=tuple(component_rows),
            segments=tuple(segments),
        )
        updates: dict[str, Any] = {
            "components": (component,) if segments else (),
            "connections": (),
            "continuation_decisions": (),
            "table_rows": table_rows,
            "page_texts": section.page_texts,
            "table_headings": MappingProxyType(
                {
                    table_id: heading
                    for table_id, heading in self.source.table_headings.items()
                    if table_id in table_ids
                }
            ),
            "page_flow": section.page_flow,
            "continuation_fragments": fragments,
            "pages": tuple(
                CanonicalEnterpriseSourcePage(page_number=page, source_page_number=page)
                for page in sorted(source_pages)
            ),
            "full_text": section.text,
            "source_page_count": len(source_pages),
        }
        # Logical ParseResult tables are authoritative when their quality gate
        # passed.  Keep only logical tables backed by physical tables in this
        # canonical slice so a section decoder cannot consume a neighbour.
        fields = getattr(self.source, "__dataclass_fields__", {})
        if "logical_tables" in fields:
            logical_tables = getattr(self.source, "logical_tables", ())
            updates["logical_tables"] = tuple(
                table
                for table in logical_tables
                if _logical_table_intersects(table, table_ids)
            )
        return replace(self.source, **updates)

    def debug_payload(self) -> dict[str, Any]:
        return {
            "contract": "pboc-enterprise-canonical-subset",
            "version": "1.0.0",
            "sections": [
                {
                    "key": section.spec.key,
                    "title": section.spec.title,
                    "heading_detected": section.heading_detected,
                    "present": section.present,
                    "source_pages": list(section.source_pages),
                    "unit_ids": list(section.unit_ids),
                    "table_ids": list(section.table_ids),
                    "datasets": list(section.spec.datasets),
                }
                for section in self.sections
            ],
        }


def _logical_table_intersects(value: Any, table_ids: set[str]) -> bool:
    if isinstance(value, dict):
        ids = value.get("source_table_ids") or value.get("table_ids") or ()
        return bool(set(str(item) for item in ids) & table_ids)
    ids = getattr(value, "source_table_ids", None) or getattr(value, "table_ids", None)
    if ids:
        return bool(set(str(item) for item in ids) & table_ids)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return str(value[1]) in table_ids
    return False


def _table_heading_lines(unit: _OrderedSourceUnit) -> Iterable[str]:
    for row in unit.rows[:3]:
        for cell in row:
            if cell:
                yield cell


def build_canonical_enterprise_subset(
    document: CanonicalEnterpriseDocumentIR,
) -> CanonicalEnterpriseSubset:
    """Partition a report into the monotonic optional sections of the layout."""
    if not isinstance(document, CanonicalEnterpriseDocumentIR):
        raise TypeError("canonical subset reconstruction requires CanonicalEnterpriseDocumentIR")

    units = _ordered_units(document)
    states: dict[str, dict[str, Any]] = {
        spec.key: {
            "heading": False,
            "unit_ids": [],
            "pages": set(),
            "table_ids": [],
            "text": [],
            "page_texts": {},
            "page_flow": [],
            "fragments": [],
        }
        for spec in CANONICAL_ENTERPRISE_SECTIONS
    }
    current_key = "report_metadata"
    last_index = 0

    def assign_text(key: str, unit: _OrderedSourceUnit, value: str) -> None:
        if not value:
            return
        state = states[key]
        if unit.unit_id not in state["unit_ids"]:
            state["unit_ids"].append(unit.unit_id)
        if unit.page > 0:
            state["pages"].add(unit.page)
        state["text"].append(value)
        state["page_texts"].setdefault(unit.page, []).append(value)
        state["page_flow"].append((unit.page, "text", value))

    for unit in units:
        if unit.kind != "table":
            chunk_key = current_key
            chunk: list[str] = []
            for raw_line in unit.text.splitlines():
                candidate_key = _section_key_for_line(raw_line)
                if candidate_key is not None:
                    candidate_index = _SPEC_INDEX[candidate_key]
                    # Canonical reports are closed-world ordered subsets.
                    # Reject a backward prose mention instead of rewinding.
                    if candidate_index >= last_index:
                        if chunk:
                            assign_text(chunk_key, unit, "\n".join(chunk))
                            chunk = []
                        current_key = candidate_key
                        chunk_key = current_key
                        last_index = candidate_index
                        states[current_key]["heading"] = True
                chunk.append(raw_line)
            if chunk:
                assign_text(chunk_key, unit, "\n".join(chunk))
            continue

        detected = [
            key for value in _table_heading_lines(unit) if (key := _section_key_for_line(value))
        ]
        if detected:
            candidate_key = detected[-1]
            candidate_index = _SPEC_INDEX[candidate_key]
            if candidate_index >= last_index:
                current_key = candidate_key
                last_index = candidate_index
                states[current_key]["heading"] = True

        state = states[current_key]
        if unit.unit_id not in state["unit_ids"]:
            state["unit_ids"].append(unit.unit_id)
        if unit.page > 0:
            state["pages"].add(unit.page)
        if unit.text:
            state["text"].append(unit.text)
            state["page_texts"].setdefault(unit.page, []).append(unit.text)
        if unit.table_id and unit.rows:
            state["table_ids"].append(unit.table_id)
            state["page_flow"].append((unit.page, "table", (unit.table_id, unit.rows)))
            state["fragments"].append(
                TableFragment(
                    index=len(state["fragments"]),
                    page=unit.page,
                    table_id=unit.table_id,
                    rows=unit.rows,
                    bbox=unit.bbox,
                    page_width=unit.page_width,
                    page_height=unit.page_height,
                    first_on_page=True,
                    last_on_page=True,
                )
            )

    # The cover is the initial canonical section and has no standalone heading
    # in many native files.  Its copied content is sufficient presence proof.
    states["report_metadata"]["heading"] = bool(states["report_metadata"]["unit_ids"])

    slices: list[CanonicalEnterpriseSectionSlice] = []
    for spec in CANONICAL_ENTERPRISE_SECTIONS:
        state = states[spec.key]
        page_texts = MappingProxyType(
            {
                int(page): "\n".join(parts)
                for page, parts in sorted(state["page_texts"].items())
            }
        )
        slices.append(
            CanonicalEnterpriseSectionSlice(
                spec=spec,
                heading_detected=bool(state["heading"]),
                unit_ids=tuple(state["unit_ids"]),
                source_pages=tuple(sorted(state["pages"])),
                table_ids=tuple(dict.fromkeys(state["table_ids"])),
                text="\n".join(state["text"]).strip(),
                page_texts=page_texts,
                page_flow=tuple(state["page_flow"]),
                fragments=tuple(state["fragments"]),
            )
        )
    return CanonicalEnterpriseSubset(source=document, sections=tuple(slices))


__all__ = [
    "CANONICAL_ENTERPRISE_SECTIONS",
    "CanonicalEnterpriseSectionSlice",
    "CanonicalEnterpriseSectionSpec",
    "CanonicalEnterpriseSubset",
    "build_canonical_enterprise_subset",
]
