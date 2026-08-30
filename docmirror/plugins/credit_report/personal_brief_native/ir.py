# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, pagination-independent IR for PBOC personal brief reports.

The IR is the only boundary consumed by the personal-brief business decoder.
It preserves every parser source unit once, records scored continuation
decisions, and reconstructs the borderless inquiry ledger as one authoritative
logical table.  Physical pages survive only as provenance.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

from docmirror.plugins.credit_report.personal_brief_native.date_rules import (
    PERSONAL_BRIEF_DATE_PATTERN,
    PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN,
    normalize_personal_brief_date,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    CreditReportUnit,
    decode_credit_report_entities,
)

ComponentKind = Literal[
    "heading",
    "paragraph",
    "key_value_group",
    "logical_table",
    "numbered_record",
]
ConnectionKind = Literal["continues", "contains", "next"]

PERSONAL_BRIEF_IR_SCHEMA_ID = "canonical_personal_brief_document"
PERSONAL_BRIEF_IR_SCHEMA_VERSION = "1.1.0"
_GLOBAL_ORDER_STRIDE = 1000

CANONICAL_PERSONAL_BRIEF_SECTIONS: tuple[tuple[str, str], ...] = (
    ("report_header", "报告信息"),
    ("credit_summary", "信息概要"),
    ("asset_disposition", "资产处置信息"),
    ("guarantor_compensation", "垫款信息"),
    ("credit_cards", "信用卡"),
    ("loans", "贷款"),
    ("other_business", "其他业务"),
    ("repayment_liability", "相关还款责任信息"),
    ("non_credit_transactions", "非信贷交易记录"),
    ("public_records", "公共记录"),
    ("tax_arrears", "欠税记录"),
    ("civil_judgments", "民事判决记录"),
    ("enforcements", "强制执行记录"),
    ("administrative_penalties", "行政处罚记录"),
    ("institution_statements", "机构说明"),
    ("institution_inquiries", "机构查询记录明细"),
    ("personal_inquiries", "个人查询记录明细"),
    ("report_notes", "说明"),
)

_SECTION_LABELS = dict(CANONICAL_PERSONAL_BRIEF_SECTIONS)
_PAGE_NUMBER_RE = re.compile(r"^第\s*\d+\s*页\s*[,，]\s*共\s*\d+\s*页$")
_DATE_RE = re.compile(PERSONAL_BRIEF_DATE_PATTERN)
_DIRECT_INQUIRY_RE = re.compile(
    r"^\s*(?P<sequence>\d{1,4})[.、]?\s*"
    rf"(?P<date>{PERSONAL_BRIEF_DATE_PATTERN})\s*"
    r"(?P<tail>.+)$",
    re.DOTALL,
)
_UNNUMBERED_INQUIRY_RE = re.compile(
    rf"^\s*(?P<date>{PERSONAL_BRIEF_DATE_PATTERN})\s*"
    r"(?P<tail>.+)$",
    re.DOTALL,
)
_STREAM_INQUIRY_RE = re.compile(
    r"(?<!\d)(?P<sequence>\d{1,4})[.、]?\s+"
    rf"(?P<date>{PERSONAL_BRIEF_DATE_PATTERN})\s+"
)
_GENERIC_INSTITUTION_REASON_RE = re.compile(
    r"((?:个人|企业|信用卡|融资|授信|担保|法人|负责人|高管|贷后|保前|资信|客户|风险|关联|异议|账户|商户)"
    r"[^，,。；;]{0,24}(?:审批|审查|管理|查询|核查|复核|评估|授信|准入)"
    r"(?:（[^）]{1,40}）)?)$"
)
_INQUIRY_REASONS = tuple(
    sorted(
        {
            "法人代表、负责人、高管等资信审查",
            "本人查询（互联网个人信用信息服务平台）",
            "本人查询（商业银行网上银行）",
            "本人查询（自助查询机）",
            "本人查询（临柜）",
            "担保资格审查",
            "保前审查",
            "资信审查",
            "融资审批",
            "信用卡审批",
            "贷款审批",
            "贷后管理",
        },
        key=len,
        reverse=True,
    )
)
_INQUIRY_HEADINGS = frozenset(
    {
        "编号",
        "查询日期",
        "查询机构",
        "查询原因",
        "机构查询记录明细",
        "个人查询记录明细",
        "本人查询记录明细",
    }
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        box = tuple(float(item) for item in raw[:4])
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def _iso_date(value: str) -> str:
    return normalize_personal_brief_date(value)


def _normalize_reason(value: str, inquiry_type: str) -> str:
    compact = _compact(value).replace("(", "（").replace(")", "）")
    compact = compact.replace("担保资格审建", "担保资格审查")
    for reason in _INQUIRY_REASONS:
        if _compact(reason) in compact:
            return reason
    if compact.startswith("担保资格审"):
        return "担保资格审查"
    if inquiry_type == "personal" and compact.startswith("本人查询"):
        channel = re.search(r"本人查询（([^）]+)）", compact)
        return f"本人查询（{channel.group(1)}）" if channel else "本人查询"
    if inquiry_type == "institution":
        matches = list(_GENERIC_INSTITUTION_REASON_RE.finditer(compact))
        match = min(matches, key=lambda item: len(item.group(1))) if matches else None
        if match is not None:
            return match.group(1)
    return ""


_SECTION_HEADING_TARGETS = {
    "信息概要": "credit_summary",
    "信用卡": "credit_cards",
    "贷款": "loans",
    "其他业务": "other_business",
    "资产处置信息": "asset_disposition",
    "保证人代偿信息": "guarantor_compensation",
    "垫款信息": "guarantor_compensation",
    "相关还款责任信息": "repayment_liability",
    "非信贷交易记录": "non_credit_transactions",
    "欠税记录": "tax_arrears",
    "民事判决记录": "civil_judgments",
    "强制执行记录": "enforcements",
    "行政处罚记录": "administrative_penalties",
    "公共记录": "public_records",
    "机构说明": "institution_statements",
    "机构查询记录明细": "institution_inquiries",
    "个人查询记录明细": "personal_inquiries",
    "本人查询记录明细": "personal_inquiries",
    "说明": "report_notes",
}


def _section_for_heading(text: str, current: str, kind: str = "text") -> str:
    # A one-cell or narrow summary table may contain a canonical label such as
    # ``其他业务``.  Only text headings are allowed to transition sections.
    if kind == "table":
        compact_table = _compact(text)
        if current == "credit_records_container" and (
            any(
                marker in compact_table
                for marker in (
                    "未结清/未销户账户数",
                    "发生过逾期的账户数",
                    "相关还款责任账户数",
                )
            )
            or (
                "资产处置信息" in compact_table
                and "垫款信息" in compact_table
                and "账户数" in compact_table
            )
        ):
            return "credit_summary"
        return current
    compact = _compact(text).strip(":：")
    first_line = _compact(str(text or "").splitlines()[0]).strip(":：")
    if compact in _SECTION_HEADING_TARGETS:
        return _SECTION_HEADING_TARGETS[compact]
    if first_line in _SECTION_HEADING_TARGETS:
        return _SECTION_HEADING_TARGETS[first_line]
    # Containers are structural scaffolding, not evidence that any optional
    # canonical child section is present.
    if compact == "信贷记录" or first_line == "信贷记录":
        return "credit_records_container"
    if compact == "查询记录" or first_line == "查询记录":
        return "inquiries_container"
    return current


def _section_after_trailing_headings(text: str, current: str) -> str:
    """Carry an exact heading at the tail of one parser unit into the next unit.

    Native PDFs occasionally bundle the final heading on a page into the preceding
    text unit.  The preceding content must keep its original section ownership, but
    the following unit must start in the section introduced by that trailing heading.
    """
    section = current
    for raw_line in str(text or "").splitlines()[1:]:
        label = _compact(raw_line).strip(":：")
        target = _SECTION_HEADING_TARGETS.get(label)
        if target is not None:
            section = target
        elif label == "信贷记录":
            section = "credit_records_container"
        elif label == "查询记录":
            section = "inquiries_container"
    return section


@dataclass(frozen=True)
class PersonalBriefSourceRef:
    source_page: int
    page_instance: int
    source_order: int
    bbox: tuple[float, float, float, float] | None = None
    unit_id: str = ""
    node_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    table_id: str = ""
    row_index: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox) if self.bbox is not None else None
        payload["node_ids"] = list(self.node_ids)
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload


@dataclass(frozen=True)
class CanonicalPersonalBriefRow:
    values: tuple[str, ...]
    source_refs: tuple[PersonalBriefSourceRef, ...]
    status: str = "reported"

    @property
    def source_pages(self) -> tuple[int, ...]:
        return tuple(sorted({ref.source_page for ref in self.source_refs}))

    def to_payload(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "source_refs": [ref.to_payload() for ref in self.source_refs],
            "status": self.status,
        }


@dataclass(frozen=True)
class CanonicalPersonalBriefComponent:
    component_id: str
    kind: ComponentKind
    section_key: str
    global_order: int
    text: str
    rows: tuple[CanonicalPersonalBriefRow, ...] = ()
    source_refs: tuple[PersonalBriefSourceRef, ...] = ()
    source_unit_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    semantic_role: str = ""

    @property
    def source_pages(self) -> tuple[int, ...]:
        pages = {ref.source_page for ref in self.source_refs}
        pages.update(page for row in self.rows for page in row.source_pages)
        return tuple(sorted(pages))

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.component_id,
            "type": self.kind,
            "section_key": self.section_key,
            "section_label": _SECTION_LABELS.get(self.section_key, self.section_key),
            "content": self.text,
            "rows": [row.to_payload() for row in self.rows],
            "global_order": self.global_order,
            "source_refs": [ref.to_payload() for ref in self.source_refs],
            "source_unit_ids": list(self.source_unit_ids),
            "confidence": self.confidence,
            "semantic_role": self.semantic_role,
        }


@dataclass(frozen=True)
class PersonalBriefConnection:
    kind: ConnectionKind
    source_id: str
    target_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class PersonalBriefContinuationDecision:
    left_unit_id: str
    right_unit_id: str
    selected: str
    best_score: float
    runner_up_score: float
    margin: float
    signals: tuple[str, ...]
    from_page: int
    to_page: int


@dataclass(frozen=True)
class _TextView:
    content: str
    bbox: list[float] | None
    evidence_ids: list[str]
    confidence: float = 1.0


@dataclass(frozen=True)
class _CellView:
    text: str


@dataclass(frozen=True)
class _RowView:
    cells: tuple[_CellView, ...]


@dataclass(frozen=True)
class _TableView:
    table_id: str
    headers: tuple[str, ...]
    rows: tuple[_RowView, ...]
    page: int
    bbox: list[float] | None
    evidence_ids: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PageView:
    page_number: int
    source_page_number: int
    width: float
    height: float
    texts: tuple[_TextView, ...]
    tables: tuple[_TableView, ...]
    key_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _ParserInfoView:
    structure: dict[str, int]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalPersonalBriefDocumentIR:
    schema_id: str
    schema_version: str
    components: tuple[CanonicalPersonalBriefComponent, ...]
    connections: tuple[PersonalBriefConnection, ...]
    continuation_decisions: tuple[PersonalBriefContinuationDecision, ...]
    section_presence: dict[str, str]
    furniture_source_refs: tuple[PersonalBriefSourceRef, ...]
    unassigned_source_unit_ids: tuple[str, ...]
    source_page_count: int
    source_unit_count: int
    confidence: float
    _page_dimensions: tuple[tuple[int, float, float], ...] = field(default=(), repr=False)

    report_family: str = field(default="personal_brief", init=False)

    @property
    def full_text(self) -> str:
        """Canonical narrative text; logical tables remain typed, not flattened."""
        return "\n".join(
            component.text
            for component in sorted(self.components, key=lambda item: item.global_order)
            if component.kind != "logical_table" and component.text.strip()
        )

    @property
    def raw_text(self) -> str:
        return self.full_text

    @property
    def content_conserved(self) -> bool:
        owned = [unit_id for component in self.components for unit_id in component.source_unit_ids]
        return (
            not self.unassigned_source_unit_ids
            and len(owned) == len(set(owned))
            and len(owned) == self.source_unit_count
        )

    @property
    def entity_context(self) -> CanonicalPersonalBriefDocumentIR:
        """Compatibility view for low-level rigid helpers; still IR-only."""
        return self

    @property
    def document_flow(self) -> None:
        return None

    @property
    def parser_info(self) -> _ParserInfoView:
        """Expose canonical page cardinality to shared audit helpers."""
        return _ParserInfoView(structure={"source_page_count": self.source_page_count})

    def components_for(self, section_key: str) -> tuple[CanonicalPersonalBriefComponent, ...]:
        return tuple(component for component in self.components if component.section_key == section_key)

    def logical_table(self, semantic_role: str) -> CanonicalPersonalBriefComponent | None:
        return next(
            (
                component
                for component in self.components
                if component.kind == "logical_table" and component.semantic_role == semantic_role
            ),
            None,
        )

    def ordered_text_blocks(self) -> tuple[tuple[int, str], ...]:
        blocks: list[tuple[int, str]] = []
        for component in sorted(self.components, key=lambda item: item.global_order):
            if component.kind == "logical_table" or not component.text.strip():
                continue
            page = component.source_pages[0] if component.source_pages else 1
            blocks.append((page, component.text))
        return tuple(blocks)

    def ordered_page_flow(self) -> tuple[tuple[int, str, Any], ...]:
        flow: list[tuple[int, str, Any]] = []
        for component in sorted(self.components, key=lambda item: item.global_order):
            page = component.source_pages[0] if component.source_pages else 1
            if component.kind == "logical_table":
                flow.append(
                    (
                        page,
                        "table",
                        (component.component_id, [list(row.values) for row in component.rows]),
                    )
                )
            else:
                flow.append((page, "text", component.text))
        return tuple(flow)

    @property
    def pages(self) -> tuple[_PageView, ...]:
        dimensions = {page: (width, height) for page, width, height in self._page_dimensions}
        page_texts: dict[int, list[_TextView]] = {page: [] for page in range(1, self.source_page_count + 1)}
        page_tables: dict[int, list[_TableView]] = {page: [] for page in range(1, self.source_page_count + 1)}
        for component in sorted(self.components, key=lambda item: item.global_order):
            if component.kind != "logical_table":
                if not component.text.strip():
                    continue
                ref = component.source_refs[0] if component.source_refs else None
                page = ref.source_page if ref is not None else 1
                page_texts.setdefault(page, []).append(
                    _TextView(
                        content=component.text,
                        bbox=list(ref.bbox) if ref is not None and ref.bbox is not None else None,
                        evidence_ids=list(ref.evidence_ids) if ref is not None else [],
                        confidence=component.confidence,
                    )
                )
                continue
            rows_by_page: dict[int, list[CanonicalPersonalBriefRow]] = {}
            for row in component.rows:
                page = row.source_pages[0] if row.source_pages else (component.source_pages[0] if component.source_pages else 1)
                rows_by_page.setdefault(page, []).append(row)
            for page, rows in rows_by_page.items():
                page_refs = [
                    ref
                    for row in rows
                    for ref in row.source_refs
                    if ref.source_page == page
                ]
                evidence_ids = list(
                    dict.fromkeys(
                        evidence_id
                        for row in rows
                        for ref in row.source_refs
                        for evidence_id in ref.evidence_ids
                    )
                )
                raw_rows = [list(row.values) for row in rows]
                source_table_ids = tuple(
                    dict.fromkeys(ref.table_id for ref in page_refs if ref.table_id)
                )
                table_id = (
                    source_table_ids[0]
                    if len(source_table_ids) == 1
                    else component.component_id
                )
                boxes = [ref.bbox for ref in page_refs if ref.bbox is not None]
                table_bbox = (
                    [
                        min(box[0] for box in boxes),
                        min(box[1] for box in boxes),
                        max(box[2] for box in boxes),
                        max(box[3] for box in boxes),
                    ]
                    if boxes
                    else None
                )
                is_inquiry_table = component.semantic_role in {
                    "institution_inquiries",
                    "personal_inquiries",
                }
                canonical_headers = (
                    ("编号", "查询日期", "查询机构", "查询原因")
                    if is_inquiry_table
                    else tuple(rows[0].values) if rows else ()
                )
                canonical_rows = rows if is_inquiry_table else rows[1:]
                page_tables.setdefault(page, []).append(
                    _TableView(
                        table_id=table_id,
                        # Source-table headers are split from body rows while
                        # ``raw_rows`` remains lossless.  This avoids duplicate
                        # header rows without discarding header semantics.
                        headers=canonical_headers,
                        rows=tuple(
                            _RowView(tuple(_CellView(value) for value in row.values))
                            for row in canonical_rows
                        ),
                        page=page,
                        bbox=table_bbox,
                        evidence_ids=evidence_ids,
                        metadata={
                            "raw_rows": raw_rows,
                            "semantic_role": component.semantic_role,
                            "canonical_component_id": component.component_id,
                        },
                    )
                )
        return tuple(
            _PageView(
                page_number=page,
                source_page_number=page,
                width=dimensions.get(page, (0.0, 0.0))[0],
                height=dimensions.get(page, (0.0, 0.0))[1],
                texts=tuple(page_texts.get(page, ())),
                tables=tuple(page_tables.get(page, ())),
            )
            for page in range(1, self.source_page_count + 1)
        )

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "schema": {"id": self.schema_id, "version": self.schema_version},
            "components": [component.to_payload() for component in self.components],
            "connections": [asdict(connection) for connection in self.connections],
            "continuation_decisions": [asdict(decision) for decision in self.continuation_decisions],
            "section_presence": dict(self.section_presence),
            "furniture_source_refs": [ref.to_payload() for ref in self.furniture_source_refs],
            "unassigned_source_unit_ids": list(self.unassigned_source_unit_ids),
            "source_page_count": self.source_page_count,
            "source_unit_count": self.source_unit_count,
            "content_conserved": self.content_conserved,
            "confidence": self.confidence,
        }

    def to_debug_json(self) -> str:
        return json.dumps(self.to_debug_payload(), ensure_ascii=False, sort_keys=True)


def _node_ids_by_evidence(parse_result: Any) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for node in getattr(getattr(parse_result, "document_flow", None), "nodes", None) or []:
        node_id = str(getattr(node, "node_id", "") or "")
        if not node_id:
            continue
        for evidence_id in getattr(node, "evidence_refs", None) or []:
            mapping.setdefault(str(evidence_id), []).append(node_id)
    return {key: tuple(dict.fromkeys(values)) for key, values in mapping.items()}


def _unit_source_refs(
    parse_result: Any,
    context: CreditReportEntityContext,
) -> tuple[dict[str, PersonalBriefSourceRef], tuple[PersonalBriefSourceRef, ...]]:
    pages = list(getattr(parse_result, "pages", None) or [])
    pages_by_number = {
        int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or index): page
        for index, page in enumerate(pages, start=1)
    }
    nodes = _node_ids_by_evidence(parse_result)
    refs: dict[str, PersonalBriefSourceRef] = {}
    for unit in context.units:
        page = pages_by_number.get(unit.page)
        source: Any | None = None
        if page is not None and unit.kind == "table":
            source = next(
                (
                    table
                    for table in getattr(page, "tables", None) or []
                    if str(getattr(table, "table_id", "") or "") == unit.table_id
                ),
                None,
            )
        elif page is not None:
            blocks = list(getattr(page, "texts", None) or [])
            source = blocks[unit.source_index] if 0 <= unit.source_index < len(blocks) else None
        evidence_ids = tuple(str(value) for value in getattr(source, "evidence_ids", None) or [] if value)
        refs[unit.unit_id] = PersonalBriefSourceRef(
            source_page=unit.page,
            page_instance=unit.page,
            source_order=unit.order,
            bbox=unit.bbox,
            unit_id=unit.unit_id,
            node_ids=tuple(dict.fromkeys(node for evidence_id in evidence_ids for node in nodes.get(evidence_id, ()))),
            evidence_ids=evidence_ids,
            table_id=unit.table_id,
        )
    furniture: list[PersonalBriefSourceRef] = []
    for page_index, page in enumerate(pages, start=1):
        page_number = int(
            getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or page_index
        )
        for text_index, block in enumerate(getattr(page, "texts", None) or []):
            content = _compact(getattr(block, "content", ""))
            if not _PAGE_NUMBER_RE.fullmatch(content):
                continue
            evidence_ids = tuple(str(value) for value in getattr(block, "evidence_ids", None) or [] if value)
            furniture.append(
                PersonalBriefSourceRef(
                    source_page=page_number,
                    page_instance=page_number,
                    source_order=text_index,
                    bbox=_bbox(block),
                    unit_id=f"furniture:text:p{page_number}:{text_index}",
                    node_ids=tuple(
                        dict.fromkeys(node for evidence_id in evidence_ids for node in nodes.get(evidence_id, ()))
                    ),
                    evidence_ids=evidence_ids,
                )
            )
    return refs, tuple(furniture)


@dataclass(frozen=True)
class _InquiryCandidate:
    inquiry_type: str
    rows: tuple[CanonicalPersonalBriefRow, ...]
    consumed_unit_ids: tuple[str, ...]
    score: float
    method: str


def _section_units(
    units: tuple[CreditReportUnit, ...],
    inquiry_type: str,
) -> tuple[CreditReportUnit, ...]:
    start_markers = (
        ("机构查询记录明细",)
        if inquiry_type == "institution"
        else ("个人查询记录明细", "本人查询记录明细")
    )
    end_markers = (
        ("个人查询记录明细", "本人查询记录明细", "说明")
        if inquiry_type == "institution"
        else ("说明",)
    )
    start = next(
        (index for index, unit in enumerate(units) if _compact(unit.text) in start_markers),
        -1,
    )
    if start < 0:
        return ()
    end = next(
        (
            index
            for index, unit in enumerate(units[start + 1 :], start=start + 1)
            if _compact(unit.text) in end_markers
        ),
        len(units),
    )
    return units[start + 1 : end]


def _split_inquiry_tail(tail: str, inquiry_type: str) -> tuple[str, str]:
    compact = _compact(tail).replace("(", "（").replace(")", "）")
    reason = _normalize_reason(compact, inquiry_type)
    if not reason:
        return compact, ""
    reason_at = compact.rfind(_compact(reason))
    institution = compact[:reason_at] if reason_at >= 0 else ""
    return institution, reason


def _direct_inquiry_candidate(
    section: tuple[CreditReportUnit, ...],
    inquiry_type: str,
    refs: dict[str, PersonalBriefSourceRef],
    *,
    starting_sequence: int = 1,
) -> _InquiryCandidate:
    rows: list[CanonicalPersonalBriefRow] = []
    consumed: list[str] = []
    expected_sequence = max(1, starting_sequence)
    for unit in section:
        match = _DIRECT_INQUIRY_RE.match(unit.text)
        explicit_sequence = int(match.group("sequence")) if match else None
        if match is None:
            match = _UNNUMBERED_INQUIRY_RE.match(unit.text)
        if match is None:
            continue
        institution, reason = _split_inquiry_tail(match.group("tail"), inquiry_type)
        if inquiry_type == "personal" and not institution:
            institution = "本人"
        sequence = explicit_sequence or expected_sequence
        ref = refs[unit.unit_id]
        rows.append(
            CanonicalPersonalBriefRow(
                values=(
                    str(sequence),
                    _iso_date(match.group("date")),
                    institution,
                    reason,
                ),
                source_refs=(ref,),
                status=(
                    "reported"
                    if explicit_sequence is not None and institution and reason
                    else "unresolved"
                ),
            )
        )
        consumed.append(unit.unit_id)
        expected_sequence = sequence + 1
    return _score_inquiry_candidate(inquiry_type, rows, consumed, "direct_text")


def _stream_inquiry_candidate(
    section: tuple[CreditReportUnit, ...],
    inquiry_type: str,
    refs: dict[str, PersonalBriefSourceRef],
) -> _InquiryCandidate:
    """Read rows split across native text units from one ordered section stream."""
    parts: list[str] = []
    unit_spans: list[tuple[int, int, CreditReportUnit]] = []
    cursor = 0
    for unit in section:
        if parts:
            cursor += 1  # the newline inserted by ``join``
        text = str(unit.text or "")
        start = cursor
        parts.append(text)
        cursor += len(text)
        unit_spans.append((start, cursor, unit))
    stream = "\n".join(parts)
    matches = list(_STREAM_INQUIRY_RE.finditer(stream))
    rows: list[CanonicalPersonalBriefRow] = []
    consumed: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stream)
        tail = _compact(stream[match.end() : end])
        reason = _normalize_reason(tail, inquiry_type)
        if not reason:
            continue
        reason_at = tail.find(_compact(reason))
        institution = "本人" if inquiry_type == "personal" else tail[:reason_at]
        if not institution or len(institution) > 100:
            continue
        used = [
            unit
            for start, stop, unit in unit_spans
            if stop > match.start() and start < end
        ]
        row_refs = tuple(refs[unit.unit_id] for unit in used)
        rows.append(
            CanonicalPersonalBriefRow(
                values=(
                    str(int(match.group("sequence"))),
                    _iso_date(match.group("date")),
                    institution,
                    reason,
                ),
                source_refs=row_refs,
            )
        )
        consumed.extend(unit.unit_id for unit in used)
    return _score_inquiry_candidate(inquiry_type, rows, consumed, "ordered_stream")


def _center(unit: CreditReportUnit) -> float:
    return (unit.bbox[1] + unit.bbox[3]) / 2.0 if unit.bbox is not None else float(unit.order)


def _is_reason_unit(unit: CreditReportUnit, inquiry_type: str) -> bool:
    return bool(_normalize_reason(unit.text, inquiry_type))


def _geometry_inquiry_candidate(
    section: tuple[CreditReportUnit, ...],
    inquiry_type: str,
    refs: dict[str, PersonalBriefSourceRef],
) -> _InquiryCandidate:
    rows: list[CanonicalPersonalBriefRow] = []
    consumed: list[str] = []
    expected_sequence = 1
    for page in sorted({unit.page for unit in section}):
        page_units = [
            unit
            for unit in section
            if unit.page == page
            and unit.bbox is not None
            and _compact(unit.text) not in _INQUIRY_HEADINGS
            and not _PAGE_NUMBER_RE.fullmatch(_compact(unit.text))
        ]
        direct_units = [
            unit
            for unit in page_units
            if _DIRECT_INQUIRY_RE.match(unit.text)
            or _UNNUMBERED_INQUIRY_RE.match(unit.text)
        ]
        direct_ids: set[str] = set()
        if direct_units:
            direct = _direct_inquiry_candidate(
                tuple(direct_units),
                inquiry_type,
                refs,
                starting_sequence=expected_sequence,
            )
            rows.extend(direct.rows)
            consumed.extend(direct.consumed_unit_ids)
            direct_ids.update(direct.consumed_unit_ids)
            if direct.rows:
                expected_sequence = int(direct.rows[-1].values[0]) + 1
        remaining = [unit for unit in page_units if unit.unit_id not in direct_ids]
        anchors = [
            unit
            for unit in remaining
            if _DATE_RE.fullmatch(_compact(unit.text))
            or _is_reason_unit(unit, inquiry_type)
            or (
                unit.bbox is not None
                and unit.bbox[0] < 110
                and re.fullmatch(r"\d{1,3}[.]?", _compact(unit.text))
            )
        ]
        if not anchors:
            continue
        centers: list[float] = []
        for anchor in sorted(anchors, key=_center):
            value = _center(anchor)
            if not centers or value - centers[-1] > 7.0:
                centers.append(value)
            else:
                centers[-1] = (centers[-1] + value) / 2.0
        for ordinal, row_center in enumerate(centers):
            upper = -float("inf") if ordinal == 0 else (centers[ordinal - 1] + row_center) / 2.0
            lower = float("inf") if ordinal + 1 == len(centers) else (row_center + centers[ordinal + 1]) / 2.0
            local = [unit for unit in remaining if upper < _center(unit) <= lower]
            date_unit = min(
                (unit for unit in local if _DATE_RE.fullmatch(_compact(unit.text))),
                key=lambda unit: abs(_center(unit) - row_center),
                default=None,
            )
            reason_unit = min(
                (unit for unit in local if _is_reason_unit(unit, inquiry_type)),
                key=lambda unit: abs(_center(unit) - row_center),
                default=None,
            )
            sequence_units = [
                unit
                for unit in local
                if unit.bbox is not None
                and unit.bbox[0] < 110
                and re.fullmatch(r"\d{1,3}[.]?", _compact(unit.text))
            ]
            explicit_sequence = None
            if sequence_units:
                sequence_unit = min(
                    sequence_units,
                    key=lambda unit: abs(_center(unit) - row_center),
                )
                candidate = int(re.sub(r"\D", "", sequence_unit.text) or 0)
                explicit_sequence = candidate if candidate > 0 else None
            sequence = explicit_sequence or expected_sequence
            reason = _normalize_reason(reason_unit.text if reason_unit is not None else "", inquiry_type)
            institution_units = [
                unit
                for unit in local
                if unit.bbox is not None
                and 245 <= unit.bbox[0] < 405
                and unit not in sequence_units
                and unit is not date_unit
                and unit is not reason_unit
                and unit.bbox[3] - unit.bbox[1] <= 28
                and len(_compact(unit.text)) <= 60
            ]
            institution_units.sort(key=lambda unit: (_center(unit), unit.bbox[0] if unit.bbox else 0.0))
            institution = "".join(_compact(unit.text) for unit in institution_units)
            if inquiry_type == "personal":
                institution = "本人" if "本人" in institution or not institution else institution
            elif len(institution) < 4:
                # Short OCR fragments such as ``中心`` are not defensible
                # institution identities; preserve their evidence but leave
                # the business cell explicitly unresolved.
                institution = ""
            # A row must contain at least two independent business columns.
            business_columns = sum(bool(value) for value in (date_unit, institution, reason))
            if business_columns < 2:
                continue
            used = [*sequence_units, *institution_units]
            if date_unit is not None:
                used.append(date_unit)
            if reason_unit is not None:
                used.append(reason_unit)
            unique_used = list({unit.unit_id: unit for unit in used}.values())
            row_refs = tuple(refs[unit.unit_id] for unit in unique_used)
            status = (
                "reported"
                if date_unit is not None and institution and reason
                else "unresolved"
            )
            rows.append(
                CanonicalPersonalBriefRow(
                    values=(
                        str(sequence),
                        _iso_date(date_unit.text if date_unit is not None else ""),
                        institution,
                        reason,
                    ),
                    source_refs=row_refs,
                    status=status,
                )
            )
            consumed.extend(unit.unit_id for unit in unique_used)
            expected_sequence = sequence + 1
    return _score_inquiry_candidate(inquiry_type, rows, consumed, "geometry")


def _native_dfg_inquiry_candidate(
    parse_result: Any,
    inquiry_type: str,
    refs: dict[str, PersonalBriefSourceRef],
) -> _InquiryCandidate:
    if inquiry_type != "institution":
        return _score_inquiry_candidate(inquiry_type, (), (), "native_dfg")
    from docmirror.plugins.credit_report.inquiry_reading_order import (
        reconstruct_institution_inquiry_rows,
    )

    rows: list[CanonicalPersonalBriefRow] = []
    consumed: list[str] = []
    for source in reconstruct_institution_inquiry_rows(parse_result):
        sequence = int(source.get("sequence") or 0)
        date = _iso_date(str(source.get("query_date_text") or ""))
        institution = _compact(source.get("institution"))
        reason = _normalize_reason(str(source.get("reason") or ""), inquiry_type)
        if sequence <= 0 or not date or not institution or not reason:
            continue
        source_node_ids = {
            str(value) for value in source.get("source_node_ids") or [] if value
        }
        row_refs = tuple(
            ref
            for ref in refs.values()
            if source_node_ids.intersection(ref.node_ids)
        )
        if not row_refs:
            evidence_ids = {
                str(value) for value in source.get("evidence_ids") or [] if value
            }
            row_refs = tuple(
                ref
                for ref in refs.values()
                if evidence_ids.intersection(ref.evidence_ids)
            )
        rows.append(
            CanonicalPersonalBriefRow(
                values=(str(sequence), date, institution, reason),
                source_refs=row_refs,
            )
        )
        consumed.extend(ref.unit_id for ref in row_refs if ref.unit_id)
    return _score_inquiry_candidate(inquiry_type, rows, consumed, "native_dfg")


def _native_bundle_inquiry_candidate(
    parse_result: Any,
    section: tuple[CreditReportUnit, ...],
    inquiry_type: str,
    refs: dict[str, PersonalBriefSourceRef],
) -> _InquiryCandidate:
    """Resolve native bundled rows and wrapped rows into one row sequence."""
    direct = _direct_inquiry_candidate(section, inquiry_type, refs)
    supplement = _native_dfg_inquiry_candidate(parse_result, inquiry_type, refs)
    by_sequence = {
        int(row.values[0]): row
        for row in direct.rows
        if row.values and row.values[0].isdigit()
    }
    for row in supplement.rows:
        if row.values and row.values[0].isdigit():
            by_sequence[int(row.values[0])] = row
    rows = tuple(by_sequence[sequence] for sequence in sorted(by_sequence))
    consumed = tuple(
        dict.fromkeys((*direct.consumed_unit_ids, *supplement.consumed_unit_ids))
    )
    return _score_inquiry_candidate(inquiry_type, rows, consumed, "native_bundle")


def _canonical_lattice_inquiry_candidate(
    inquiry_type: str,
    candidates: tuple[_InquiryCandidate, ...],
) -> _InquiryCandidate:
    """Select one best native observation for every printed row ordinal."""
    method_rank = {"native_bundle": 3, "ordered_stream": 2, "direct_text": 1}
    selected: dict[int, tuple[tuple[int, int, int, int], CanonicalPersonalBriefRow]] = {}
    for candidate in candidates:
        for row in candidate.rows:
            if not row.values or not row.values[0].isdigit():
                continue
            sequence = int(row.values[0])
            populated = sum(bool(value) for value in row.values[1:4])
            score = (
                1 if row.status == "reported" else 0,
                populated,
                method_rank.get(candidate.method, 0),
                len(row.source_refs),
            )
            if sequence not in selected or score > selected[sequence][0]:
                selected[sequence] = (score, row)
    rows = tuple(selected[sequence][1] for sequence in sorted(selected))
    consumed = tuple(
        dict.fromkeys(
            ref.unit_id
            for row in rows
            for ref in row.source_refs
            if ref.unit_id
        )
    )
    return _score_inquiry_candidate(
        inquiry_type,
        rows,
        consumed,
        "canonical_lattice",
    )


def _score_inquiry_candidate(
    inquiry_type: str,
    rows: Iterable[CanonicalPersonalBriefRow],
    consumed: Iterable[str],
    method: str,
) -> _InquiryCandidate:
    materialized = tuple(rows)
    sequences = [int(row.values[0]) for row in materialized if row.values and row.values[0].isdigit()]
    contiguous = bool(sequences) and sequences == list(range(1, len(sequences) + 1))
    complete = sum(row.status == "reported" for row in materialized)
    score = (
        min(len(materialized), 100) * 0.01
        + (complete / len(materialized) * 0.62 if materialized else 0.0)
        + (0.32 if contiguous else 0.0)
    )
    return _InquiryCandidate(
        inquiry_type=inquiry_type,
        rows=materialized,
        consumed_unit_ids=tuple(dict.fromkeys(consumed)),
        score=round(score, 6),
        method=method,
    )


def _inquiry_candidates(
    parse_result: Any,
    context: CreditReportEntityContext,
    refs: dict[str, PersonalBriefSourceRef],
) -> tuple[_InquiryCandidate, ...]:
    selected: list[_InquiryCandidate] = []
    for inquiry_type in ("institution", "personal"):
        section = _section_units(context.units, inquiry_type)
        if not section:
            continue
        native_observations = (
            _native_bundle_inquiry_candidate(parse_result, section, inquiry_type, refs),
            _stream_inquiry_candidate(section, inquiry_type, refs),
            _direct_inquiry_candidate(section, inquiry_type, refs),
        )
        alternatives = (
            _canonical_lattice_inquiry_candidate(inquiry_type, native_observations),
            _geometry_inquiry_candidate(section, inquiry_type, refs),
        )
        method_rank = {"canonical_lattice": 2, "geometry": 1}
        best = max(
            alternatives,
            key=lambda item: (item.score, len(item.rows), method_rank.get(item.method, 0)),
        )
        if best.rows:
            selected.append(best)
    return tuple(selected)


_NUMBERED_ORDINAL_RE = re.compile(r"^\s*(?P<sequence>\d{1,4})[.、]\s*$")
_LIABILITY_RECORD_START_RE = re.compile(
    rf"(?<!\d){PERSONAL_BRIEF_DATE_PATTERN}\s*[，,]?\s*为"
)
_LIABILITY_RECORD_BOUNDARY_RE = re.compile(
    r"(?:(?<!\d)(?P<sequence>\d{1,4})[.、]\s*)?"
    rf"(?P<record>(?<!\d){PERSONAL_BRIEF_DATE_PATTERN}\s*[，,]?\s*为)"
)


@dataclass(frozen=True)
class _EvidenceLine:
    text: str
    source_page: int
    bbox: tuple[float, float, float, float] | None
    evidence_ids: tuple[str, ...]
    unit_id: str
    source_ref: PersonalBriefSourceRef
    source_order: int
    line_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class _NumberedNarrativeReconstruction:
    components: tuple[CanonicalPersonalBriefComponent, ...]
    consumed_unit_ids: tuple[str, ...]
    expected_fragment_ids: tuple[str, ...]
    continuation_decisions: tuple[PersonalBriefContinuationDecision, ...]
    first_source_index: int


def _section_by_unit(units: Iterable[CreditReportUnit]) -> dict[str, str]:
    """Assign sections before replacing unreliable parser block boundaries."""
    materialized = tuple(units)
    current = "report_header"
    assignments: dict[str, str] = {}
    for unit in materialized:
        current = _section_for_heading(unit.text, current, unit.kind)
        assignments[unit.unit_id] = current
        if unit.kind != "table":
            current = _section_after_trailing_headings(unit.text, current)

    # Coarse native blocks often contain many records and a trailing heading,
    # while the visual ordinal column remains split into later parser units.
    # Recover each ordinal's section from the date-bearing block at the same
    # vertical band instead of inheriting the trailing heading's new section.
    for unit in materialized:
        ordinal_lines = tuple(
            line.strip() for line in str(unit.text or "").splitlines() if line.strip()
        )
        if (
            not ordinal_lines
            or not all(_NUMBERED_ORDINAL_RE.fullmatch(line) for line in ordinal_lines)
            or unit.bbox is None
        ):
            continue
        center_y = (unit.bbox[1] + unit.bbox[3]) / 2.0
        aligned = [
            candidate
            for candidate in materialized
            if candidate.unit_id != unit.unit_id
            and candidate.page == unit.page
            and candidate.bbox is not None
            and candidate.bbox[1] - 2.0 <= center_y <= candidate.bbox[3] + 2.0
            and _DATE_RE.search(candidate.text)
        ]
        if aligned:
            body = min(aligned, key=lambda item: item.bbox[3] - item.bbox[1])
            assignments[unit.unit_id] = assignments[body.unit_id]
    return assignments


def _evidence_text_atoms(parse_result: Any) -> dict[str, Any]:
    plane = getattr(parse_result, "evidence_plane", None)
    store = getattr(plane, "evidence", None)
    atoms = getattr(store, "text_atoms", None)
    if atoms is None and isinstance(store, dict):
        atoms = store.get("text_atoms")
    out: dict[str, Any] = {}
    for atom in atoms or ():
        evidence_id = str(
            atom.get("id", "") if isinstance(atom, dict) else getattr(atom, "id", "")
        )
        if evidence_id:
            out[evidence_id] = atom
    return out


def _atom_value(atom: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(atom, dict):
        return atom.get(field_name, default)
    return getattr(atom, field_name, default)


def _interpolated_bbox(
    bbox: tuple[float, float, float, float] | None,
    index: int,
    count: int,
) -> tuple[float, float, float, float] | None:
    if bbox is None or count <= 1:
        return bbox
    line_height = (bbox[3] - bbox[1]) / count
    top = bbox[1] + line_height * index
    return (bbox[0], top, bbox[2], top + line_height)


def _evidence_lines_for_unit(
    unit: CreditReportUnit,
    ref: PersonalBriefSourceRef,
    atoms_by_id: dict[str, Any],
) -> tuple[_EvidenceLine, ...]:
    resolved_atoms = [atoms_by_id.get(evidence_id) for evidence_id in ref.evidence_ids]
    use_evidence = bool(ref.evidence_ids) and all(atom is not None for atom in resolved_atoms)
    if use_evidence:
        atom_text = "".join(
            str(_atom_value(atom, "text", "") or "") for atom in resolved_atoms
        )
        use_evidence = _compact(atom_text) == _compact(unit.text)

    lines: list[_EvidenceLine] = []
    if use_evidence:
        atom_entries = [
            {
                "evidence_id": evidence_id,
                "text": str(_atom_value(atom, "text", "") or ""),
                "bbox": _bbox(atom),
                "confidence": float(_atom_value(atom, "confidence", 1.0) or 0.0),
                "index": atom_index,
            }
            for atom_index, (evidence_id, atom) in enumerate(
                zip(ref.evidence_ids, resolved_atoms, strict=True)
            )
            if str(_atom_value(atom, "text", "") or "").strip()
        ]
        use_evidence = bool(atom_entries) and all(
            entry["bbox"] is not None for entry in atom_entries
        )

    if use_evidence:
        visual_lines: list[list[dict[str, Any]]] = []
        for entry in sorted(
            atom_entries,
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
                item["index"],
            ),
        ):
            box = entry["bbox"]
            matching_group: list[dict[str, Any]] | None = None
            for group in reversed(visual_lines):
                group_boxes = [item["bbox"] for item in group]
                group_top = min(item[1] for item in group_boxes)
                group_bottom = max(item[3] for item in group_boxes)
                overlap = min(group_bottom, box[3]) - max(group_top, box[1])
                minimum_height = min(group_bottom - group_top, box[3] - box[1])
                group_center = (group_top + group_bottom) / 2.0
                box_center = (box[1] + box[3]) / 2.0
                if overlap >= minimum_height * 0.45 or abs(group_center - box_center) <= 2.0:
                    matching_group = group
                    break
                if box[1] > group_bottom + 2.0:
                    break
            if matching_group is None:
                visual_lines.append([entry])
            else:
                matching_group.append(entry)

        for visual_index, group in enumerate(visual_lines):
            ordered = sorted(group, key=lambda item: (item["bbox"][0], item["index"]))
            raw_text = "".join(str(item["text"]).strip() for item in ordered)
            atom_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if not atom_lines and raw_text.strip():
                atom_lines = [raw_text.strip()]
            boxes = [item["bbox"] for item in group]
            visual_bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            evidence_ids = tuple(str(item["evidence_id"]) for item in ordered)
            source_order = min(int(item["index"]) for item in group)
            confidence = min(float(item["confidence"]) for item in group)
            for line_index, text in enumerate(atom_lines):
                lines.append(
                    _EvidenceLine(
                        text=text,
                        source_page=ref.source_page,
                        bbox=_interpolated_bbox(visual_bbox, line_index, len(atom_lines)),
                        evidence_ids=evidence_ids,
                        unit_id=unit.unit_id,
                        source_ref=ref,
                        source_order=(
                            ref.source_order * _GLOBAL_ORDER_STRIDE
                            + source_order * 100
                            + line_index
                        ),
                        line_id=f"{unit.unit_id}:visual-line:{visual_index}:{line_index}",
                        confidence=confidence,
                    )
                )
        return tuple(lines)

    raw_lines = [line.strip() for line in str(unit.text or "").splitlines() if line.strip()]
    if not raw_lines and unit.text.strip():
        raw_lines = [unit.text.strip()]
    return tuple(
        _EvidenceLine(
            text=text,
            source_page=ref.source_page,
            bbox=_interpolated_bbox(ref.bbox, index, len(raw_lines)),
            evidence_ids=(),
            unit_id=unit.unit_id,
            source_ref=ref,
            source_order=ref.source_order * _GLOBAL_ORDER_STRIDE + index,
            line_id=f"{unit.unit_id}:line:{index}",
        )
        for index, text in enumerate(raw_lines)
    )


def _line_sort_key(line: _EvidenceLine) -> tuple[int, float, float, int]:
    if line.bbox is None:
        return (line.source_page, float(line.source_order), 0.0, line.source_order)
    return (line.source_page, line.bbox[1], line.bbox[0], line.source_order)


def _ref_for_lines(lines: Iterable[_EvidenceLine]) -> tuple[PersonalBriefSourceRef, ...]:
    grouped: dict[str, list[_EvidenceLine]] = {}
    for line in lines:
        grouped.setdefault(line.unit_id, []).append(line)
    refs: list[PersonalBriefSourceRef] = []
    for unit_id, unit_lines in grouped.items():
        base = unit_lines[0].source_ref
        boxes = [line.bbox for line in unit_lines if line.bbox is not None]
        bbox = (
            (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            if boxes
            else base.bbox
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for line in unit_lines
                for evidence_id in line.evidence_ids
                if evidence_id
            )
        )
        refs.append(
            PersonalBriefSourceRef(
                **{
                    **asdict(base),
                    "bbox": bbox,
                    "unit_id": unit_id,
                    "evidence_ids": evidence_ids or base.evidence_ids,
                }
            )
        )
    return tuple(refs)


def _aligned_ordinal(
    line: _EvidenceLine,
    ordinals: Iterable[tuple[int, _EvidenceLine]],
) -> tuple[int, _EvidenceLine] | None:
    if line.bbox is None:
        return None
    candidates = [
        (sequence, ordinal)
        for sequence, ordinal in ordinals
        if ordinal.source_page == line.source_page
        and ordinal.bbox is not None
        and abs(ordinal.bbox[1] - line.bbox[1]) <= 5.0
    ]
    return min(candidates, key=lambda item: abs(item[1].bbox[1] - line.bbox[1])) if candidates else None


def _liability_record_is_open(text: str) -> bool:
    compact = _compact(text)
    return bool(
        compact.endswith(("保证合同编号：", "保证合同编号:"))
        or compact.count("（") > compact.count("）")
        or "截至" not in compact
        or "余额" not in compact
    )


def _score_liability_continuation(
    previous_text: str,
    incoming_text: str,
    *,
    previous_page: int,
    incoming: _EvidenceLine,
    page_heights: dict[int, float],
    bounded_by_next_record: bool = False,
) -> tuple[float, float, tuple[str, ...]]:
    compact_previous = _compact(previous_text)
    compact_incoming = _compact(incoming_text)
    signals: list[str] = []
    score = 0.0
    if incoming.source_page == previous_page + 1:
        score += 0.18
        signals.append("adjacent_page")
    page_height = page_heights.get(incoming.source_page, 0.0)
    if incoming.bbox is not None and page_height > 0 and incoming.bbox[1] / page_height <= 0.12:
        score += 0.16
        signals.append("page_top")
    if _liability_record_is_open(previous_text):
        score += 0.24
        signals.append("open_liability_record")
    if compact_previous.endswith(("保证合同编号：", "保证合同编号:")):
        score += 0.24
        signals.append("open_contract_number")
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9._/-]{5,}）", compact_incoming):
        score += 0.13
        signals.append("contract_number_completion")
    if re.match(
        rf"^截至{PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN}[，,]",
        compact_incoming,
    ):
        score += 0.18
        signals.append("snapshot_tail")
    if "余额" in compact_incoming:
        score += 0.12
        signals.append("balance_tail")
    if compact_previous.count("（") > compact_previous.count("）") and "）" in compact_incoming:
        score += 0.12
        signals.append("delimiter_completion")
    if any(
        marker in compact_incoming
        for marker in ("责任人类型为", "相关还款责任金额", "保证合同编号", "截至")
    ):
        score += 0.08
        signals.append("canonical_field_tail")
    if bounded_by_next_record:
        score += 0.08
        signals.append("bounded_by_next_record")
    if not _LIABILITY_RECORD_START_RE.search(incoming_text):
        score += 0.05
        signals.append("no_new_record_anchor")
    unresolved_score = 0.45
    if incoming.source_page != previous_page + 1:
        unresolved_score = 0.85
    elif not _liability_record_is_open(previous_text):
        unresolved_score = 0.75
    return (
        round(min(score, 1.0), 6),
        round(unresolved_score, 6),
        tuple(signals),
    )


def _exact_heading_target(text: str) -> str | None:
    label = _compact(text).strip(":：")
    if label in _SECTION_HEADING_TARGETS:
        return _SECTION_HEADING_TARGETS[label]
    if label == "信贷记录":
        return "credit_records_container"
    if label == "查询记录":
        return "inquiries_container"
    return None


def _reconstruct_repayment_liability_records(
    parse_result: Any,
    context: CreditReportEntityContext,
    refs: dict[str, PersonalBriefSourceRef],
) -> _NumberedNarrativeReconstruction | None:
    """Replace coarse liability blocks with canonical, evidence-backed records."""
    assignments = _section_by_unit(context.units)
    units = tuple(
        unit
        for unit in context.units
        if assignments.get(unit.unit_id) == "repayment_liability"
        and unit.kind != "table"
        and _compact(unit.text).strip(":：") != "相关还款责任信息"
    )
    if not units:
        return None

    atoms_by_id = _evidence_text_atoms(parse_result)
    lines = sorted(
        (
            line
            for unit in units
            for line in _evidence_lines_for_unit(unit, refs[unit.unit_id], atoms_by_id)
        ),
        key=_line_sort_key,
    )
    ordinals: list[tuple[int, _EvidenceLine, str]] = []
    narrative: list[tuple[_EvidenceLine, str]] = []
    for line in lines:
        ordinal = _NUMBERED_ORDINAL_RE.fullmatch(line.text)
        if ordinal:
            ordinals.append(
                (
                    int(ordinal.group("sequence")),
                    line,
                    f"fragment:{line.line_id}:slice:0",
                )
            )
            continue
        narrative.append((line, line.text))

    builders: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    continuation_decisions: list[PersonalBriefContinuationDecision] = []
    fragment_counts: dict[str, int] = {}
    fragment_source_text = {
        fragment_id: line.text for _sequence, line, fragment_id in ordinals
    }
    fragment_line_ids = {
        fragment_id: line.line_id for _sequence, line, fragment_id in ordinals
    }
    used_ordinal_fragments: set[str] = set()
    active_section = "repayment_liability"
    page_heights = {
        int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or index):
        float(getattr(page, "height", 0.0) or 0.0)
        for index, page in enumerate(getattr(parse_result, "pages", None) or (), start=1)
    }

    def next_fragment_id(line: _EvidenceLine, source_text: str) -> str:
        slice_index = fragment_counts.get(line.line_id, 0)
        fragment_counts[line.line_id] = slice_index + 1
        fragment_id = f"fragment:{line.line_id}:slice:{slice_index}"
        fragment_source_text[fragment_id] = source_text
        fragment_line_ids[fragment_id] = line.line_id
        return fragment_id

    def fragment_sort_key(
        line: _EvidenceLine,
        fragment_id: str,
    ) -> tuple[int, float, float, int, int]:
        slice_index = int(fragment_id.rsplit(":slice:", 1)[-1])
        return (*_line_sort_key(line), slice_index)

    def finish_current() -> None:
        nonlocal current
        if current is not None:
            builders.append(current)
            current = None

    def add_residual(
        line: _EvidenceLine,
        text: str,
        *,
        section_key: str,
        semantic_role: str,
        status: str,
        fragment_id: str | None = None,
    ) -> None:
        if not text.strip():
            return
        owned_fragment_id = fragment_id or next_fragment_id(line, text)
        residuals.append(
            {
                "line": line,
                "text": text.strip(),
                "section_key": section_key,
                "semantic_role": semantic_role,
                "status": status,
                "fragment_id": owned_fragment_id,
                "sort_key": fragment_sort_key(line, owned_fragment_id),
            }
        )

    def append_span(
        span: list[tuple[_EvidenceLine, str, str]],
        *,
        bounded_by_next_record: bool = False,
    ) -> bool:
        nonlocal current
        materialized = [item for item in span if item[1].strip()]
        if not materialized:
            return False
        line, _first_text, _first_fragment_id = materialized[0]
        combined_text = "\n".join(text.strip() for _line, text, _fragment_id in materialized)
        if current is None:
            for fragment_line, fragment_text, fragment_id in materialized:
                add_residual(
                    fragment_line,
                    fragment_text,
                    section_key="repayment_liability",
                    semantic_role="unresolved_liability_fragment",
                    status="unresolved",
                    fragment_id=fragment_id,
                )
            return False
        previous_page = int(current["last_page"])
        if line.source_page != previous_page:
            score, runner_up, signals = _score_liability_continuation(
                "\n".join(current["parts"]),
                combined_text,
                previous_page=previous_page,
                incoming=line,
                page_heights=page_heights,
                bounded_by_next_record=bounded_by_next_record,
            )
            eligible_tail = bool(
                "snapshot_tail" in signals
                or (
                    "open_contract_number" in signals
                    and "contract_number_completion" in signals
                )
                or "delimiter_completion" in signals
                or "canonical_field_tail" in signals
            )
            if not eligible_tail or score < 0.70 or score - runner_up < 0.10:
                if _liability_record_is_open("\n".join(current["parts"])):
                    current["status"] = "unresolved"
                for fragment_line, fragment_text, fragment_id in materialized:
                    add_residual(
                        fragment_line,
                        fragment_text,
                        section_key="repayment_liability",
                        semantic_role="unresolved_liability_fragment",
                        status="unresolved",
                        fragment_id=fragment_id,
                    )
                return False
            left_line = current["lines"][-1]
            continuation_decisions.append(
                PersonalBriefContinuationDecision(
                    left_unit_id=left_line.unit_id,
                    right_unit_id=line.unit_id,
                    selected="continue_numbered_record",
                    best_score=score,
                    runner_up_score=runner_up,
                    margin=round(max(0.0, score - runner_up), 6),
                    signals=signals,
                    from_page=previous_page,
                    to_page=line.source_page,
                )
            )
            current["continuation_scores"].append(score)
        for fragment_line, fragment_text, fragment_id in materialized:
            current["parts"].append(fragment_text.strip())
            current["lines"].append(fragment_line)
            current["fragment_ids"].append(fragment_id)
            current["last_page"] = fragment_line.source_page
        return True

    def append_fragment(
        line: _EvidenceLine,
        text: str,
        *,
        bounded_by_next_record: bool = False,
        fragment_id: str | None = None,
    ) -> bool:
        return append_span(
            [(line, text, fragment_id or next_fragment_id(line, text))],
            bounded_by_next_record=bounded_by_next_record,
        )

    narrative_index = 0
    while narrative_index < len(narrative):
        line, text = narrative[narrative_index]
        heading_target = _exact_heading_target(text)
        if heading_target is not None:
            if heading_target != "repayment_liability":
                finish_current()
            add_residual(
                line,
                text,
                section_key=heading_target,
                semantic_role="section_heading",
                status="reported",
            )
            active_section = heading_target
            narrative_index += 1
            continue
        if active_section != "repayment_liability":
            add_residual(
                line,
                text,
                section_key=active_section,
                semantic_role="reconstructed_source_fragment",
                status="reported",
            )
            narrative_index += 1
            continue

        starts = list(_LIABILITY_RECORD_BOUNDARY_RE.finditer(text))
        if not starts:
            if current is not None and line.source_page != int(current["last_page"]):
                page_prefix: list[tuple[_EvidenceLine, str, str]] = []
                scan_index = narrative_index
                while scan_index < len(narrative):
                    prefix_line, prefix_text = narrative[scan_index]
                    if prefix_line.source_page != line.source_page:
                        break
                    if _exact_heading_target(prefix_text) is not None:
                        break
                    if _LIABILITY_RECORD_BOUNDARY_RE.search(prefix_text):
                        break
                    page_prefix.append(
                        (
                            prefix_line,
                            prefix_text,
                            next_fragment_id(prefix_line, prefix_text),
                        )
                    )
                    scan_index += 1
                bounded = bool(
                    scan_index < len(narrative)
                    and narrative[scan_index][0].source_page == line.source_page
                    and _LIABILITY_RECORD_BOUNDARY_RE.search(narrative[scan_index][1])
                )
                append_span(page_prefix, bounded_by_next_record=bounded)
                narrative_index = scan_index
                continue
            append_fragment(line, text)
            narrative_index += 1
            continue
        prefix = text[: starts[0].start()].strip()
        if prefix:
            append_fragment(line, prefix, bounded_by_next_record=True)
        for index, start in enumerate(starts):
            finish_current()
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            aligned = (
                _aligned_ordinal(
                    line,
                    ((sequence, ordinal) for sequence, ordinal, _fragment_id in ordinals),
                )
                if index == 0
                else None
            )
            inline_sequence = start.group("sequence")
            sequence = (
                int(inline_sequence)
                if inline_sequence is not None
                else aligned[0]
                if index == 0 and aligned is not None
                else int(builders[-1]["sequence"]) + 1
                if builders
                else 1
            )
            ordinal_fragment_id = next(
                (
                    fragment_id
                    for ordinal_sequence, ordinal_line, fragment_id in ordinals
                    if aligned is not None
                    and ordinal_sequence == aligned[0]
                    and ordinal_line == aligned[1]
                ),
                None,
            )
            if ordinal_fragment_id is not None:
                used_ordinal_fragments.add(ordinal_fragment_id)
            body = text[start.start("record") : end].strip()
            body_fragment_id = next_fragment_id(line, text[start.start() : end])
            current = {
                "sequence": sequence,
                "parts": [body],
                "lines": [line],
                "fragment_ids": [body_fragment_id],
                "ordinal": aligned[1] if aligned is not None else None,
                "ordinal_fragment_id": ordinal_fragment_id,
                "last_page": line.source_page,
                "status": "reported",
                "continuation_scores": [],
                "sort_key": fragment_sort_key(line, body_fragment_id),
            }
        narrative_index += 1
    finish_current()
    if not builders:
        return None

    for sequence, ordinal_line, fragment_id in ordinals:
        if fragment_id in used_ordinal_fragments:
            continue
        residuals.append(
            {
                "line": ordinal_line,
                "text": "",
                "section_key": "repayment_liability",
                "semantic_role": "record_ordinal",
                "status": "structural",
                "fragment_id": fragment_id,
                "sort_key": fragment_sort_key(ordinal_line, fragment_id),
                "sequence": sequence,
            }
        )

    unit_index = {unit.unit_id: index for index, unit in enumerate(context.units, start=1)}
    consumed = tuple(dict.fromkeys(line.unit_id for line in lines))
    first_source_index = min((unit_index[unit_id] for unit_id in consumed), default=1)
    ordered_components: list[
        tuple[tuple[int, float, float, int, int], CanonicalPersonalBriefComponent]
    ] = []
    for record_index, builder in enumerate(builders, start=1):
        text = "\n".join(builder["parts"]).strip()
        narrative_refs = _ref_for_lines(builder["lines"])
        ordinal = builder.get("ordinal")
        component_refs = tuple(
            dict.fromkeys(
                (
                    *narrative_refs,
                    *(_ref_for_lines((ordinal,)) if ordinal is not None else ()),
                )
            )
        )
        complete = bool(
            "承担相关还款责任" in _compact(text)
            and "余额" in _compact(text)
            and not _liability_record_is_open(text)
        )
        status = "reported" if complete and builder["status"] == "reported" else "unresolved"
        observed_confidence = min(
            (
                *(line.confidence for line in builder["lines"]),
                *builder["continuation_scores"],
            ),
            default=1.0,
        )
        confidence = min(0.99, observed_confidence) if status == "reported" else min(0.58, observed_confidence)
        sequence = int(builder["sequence"])
        source_fragment_ids = tuple(
            dict.fromkeys(
                (
                    *builder["fragment_ids"],
                    *((builder["ordinal_fragment_id"],) if builder.get("ordinal_fragment_id") else ()),
                )
            )
        )
        ordered_components.append(
            (
                builder["sort_key"],
            CanonicalPersonalBriefComponent(
                component_id=f"personal_brief:numbered_record:repayment_liability:{record_index:04d}",
                kind="numbered_record",
                section_key="repayment_liability",
                global_order=0,
                text=f"{sequence}. {text}",
                rows=(
                    CanonicalPersonalBriefRow(
                        values=(str(sequence), text),
                        source_refs=narrative_refs,
                        status=status,
                    ),
                ),
                source_refs=component_refs,
                source_unit_ids=source_fragment_ids,
                confidence=confidence,
                semantic_role="repayment_liability_record",
            ),
            )
        )
    for residual_index, residual in enumerate(residuals, start=1):
        line = residual["line"]
        semantic_role = str(residual["semantic_role"])
        text = str(residual["text"])
        rows = (
            (
                CanonicalPersonalBriefRow(
                    values=(f"{int(residual['sequence'])}.",),
                    source_refs=_ref_for_lines((line,)),
                    status="structural",
                ),
            )
            if semantic_role == "record_ordinal"
            else ()
        )
        ordered_components.append(
            (
                residual["sort_key"],
                CanonicalPersonalBriefComponent(
                    component_id=f"personal_brief:reconstructed_fragment:{residual_index:04d}",
                    kind=(
                        "heading"
                        if semantic_role == "section_heading"
                        else "numbered_record"
                        if semantic_role == "record_ordinal"
                        else "paragraph"
                    ),
                    section_key=str(residual["section_key"]),
                    global_order=0,
                    text=text,
                    rows=rows,
                    source_refs=_ref_for_lines((line,)),
                    source_unit_ids=(str(residual["fragment_id"]),),
                    confidence=line.confidence if residual["status"] == "reported" else min(0.45, line.confidence),
                    semantic_role=semantic_role,
                ),
            )
        )
    ordered_components.sort(key=lambda item: item[0])
    components = tuple(
        CanonicalPersonalBriefComponent(
            **{
                **asdict(component),
                "global_order": first_source_index * _GLOBAL_ORDER_STRIDE + index,
                "rows": component.rows,
                "source_refs": component.source_refs,
                "source_unit_ids": component.source_unit_ids,
            }
        )
        for index, (_source_order, component) in enumerate(ordered_components, start=1)
    )
    uncovered_fragment_ids: list[str] = []
    for line in lines:
        line_fragment_ids = sorted(
            (
                fragment_id
                for fragment_id, line_id in fragment_line_ids.items()
                if line_id == line.line_id
            ),
            key=lambda fragment_id: int(fragment_id.rsplit(":slice:", 1)[-1]),
        )
        reconstructed_text = "".join(
            fragment_source_text[fragment_id] for fragment_id in line_fragment_ids
        )
        if _compact(reconstructed_text) != _compact(line.text):
            uncovered_fragment_ids.append(f"uncovered:{line.line_id}")

    return _NumberedNarrativeReconstruction(
        components=components,
        consumed_unit_ids=consumed,
        expected_fragment_ids=tuple(
            dict.fromkeys((*fragment_source_text, *uncovered_fragment_ids))
        ),
        continuation_decisions=tuple(continuation_decisions),
        first_source_index=first_source_index,
    )


def _component_kind(unit: CreditReportUnit) -> ComponentKind:
    if unit.kind == "heading":
        return "heading"
    if unit.kind == "table":
        return "logical_table"
    if re.match(r"^\s*\d{1,4}[.、]", unit.text):
        return "numbered_record"
    return "paragraph"


def _decision_payload(
    context: CreditReportEntityContext,
    *,
    excluded_unit_ids: Iterable[str] = (),
) -> tuple[PersonalBriefContinuationDecision, ...]:
    excluded = set(excluded_unit_ids)
    decisions: list[PersonalBriefContinuationDecision] = []
    for decision in context.decisions:
        if decision.left_unit_id in excluded or decision.right_unit_id in excluded:
            continue
        hypotheses = sorted(decision.hypotheses, key=lambda item: item.score, reverse=True)
        best = hypotheses[0] if hypotheses else None
        runner_up = hypotheses[1] if len(hypotheses) > 1 else None
        best_score = float(best.score if best is not None else decision.confidence)
        runner_up_score = float(runner_up.score if runner_up is not None else 0.0)
        selected_hypothesis = next(
            (item for item in hypotheses if item.action == decision.selected),
            best,
        )
        decisions.append(
            PersonalBriefContinuationDecision(
                left_unit_id=decision.left_unit_id,
                right_unit_id=decision.right_unit_id,
                selected=decision.selected,
                best_score=best_score,
                runner_up_score=runner_up_score,
                margin=round(max(0.0, best_score - runner_up_score), 6),
                signals=tuple(selected_hypothesis.signals if selected_hypothesis is not None else ()),
                from_page=decision.from_page,
                to_page=decision.to_page,
            )
        )
    return tuple(decisions)


def build_canonical_personal_brief_document(parse_result: Any) -> CanonicalPersonalBriefDocumentIR:
    """Reconstruct one canonical long document from an immutable ParseResult."""
    if isinstance(parse_result, CanonicalPersonalBriefDocumentIR):
        return parse_result
    context = decode_credit_report_entities(parse_result, report_family="personal_brief", beam_width=7)
    refs, furniture = _unit_source_refs(parse_result, context)
    inquiry_tables = _inquiry_candidates(parse_result, context, refs)
    liability_reconstruction = _reconstruct_repayment_liability_records(
        parse_result,
        context,
        refs,
    )
    liability_consumed = set(
        liability_reconstruction.consumed_unit_ids
        if liability_reconstruction is not None
        else ()
    )
    global_index_by_unit = {
        unit.unit_id: index for index, unit in enumerate(context.units, start=1)
    }
    consumed_by_inquiry = {
        unit_id: candidate
        for candidate in inquiry_tables
        for unit_id in candidate.consumed_unit_ids
    }
    first_consumed = {
        min(
            (
                global_index_by_unit[unit_id]
                for unit_id in candidate.consumed_unit_ids
                if unit_id in global_index_by_unit
            ),
            default=len(context.units),
        ): candidate
        for candidate in inquiry_tables
    }

    components: list[CanonicalPersonalBriefComponent] = []
    unit_to_component: dict[str, str] = {}
    unit_sections = _section_by_unit(context.units)
    for global_index, unit in enumerate(context.units, start=1):
        current_section = unit_sections[unit.unit_id]
        if (
            liability_reconstruction is not None
            and global_index == liability_reconstruction.first_source_index
        ):
            components.extend(liability_reconstruction.components)
        if unit.unit_id in liability_consumed:
            continue
        inquiry = first_consumed.get(global_index)
        if inquiry is not None:
            role = f"{inquiry.inquiry_type}_inquiries"
            component_id = f"personal_brief:logical_table:{role}"
            source_refs = tuple(
                dict.fromkeys(
                    ref
                    for row in inquiry.rows
                    for ref in row.source_refs
                )
            )
            component = CanonicalPersonalBriefComponent(
                component_id=component_id,
                kind="logical_table",
                section_key=role,
                global_order=global_index * _GLOBAL_ORDER_STRIDE,
                text="\n".join(" | ".join(row.values) for row in inquiry.rows),
                rows=inquiry.rows,
                source_refs=source_refs,
                source_unit_ids=inquiry.consumed_unit_ids,
                confidence=max(
                    0.0,
                    min(
                        1.0,
                        inquiry.score
                        / max(min(len(inquiry.rows), 100) * 0.01 + 0.94, 1.0),
                    ),
                ),
                semantic_role=role,
            )
            components.append(component)
            for unit_id in inquiry.consumed_unit_ids:
                unit_to_component[unit_id] = component_id
        if unit.unit_id in consumed_by_inquiry:
            continue
        entity = context.entity_for_unit(unit.unit_id)
        table_units = (
            tuple(
                candidate
                for candidate in context.units_for_entity(entity.entity_id)
                if candidate.kind == "table" and candidate.unit_id not in consumed_by_inquiry
            )
            if entity is not None and entity.kind == "table"
            else ()
        )
        if len(table_units) > 1:
            if unit.unit_id != table_units[0].unit_id:
                continue
            component_id = f"personal_brief:component:{len(components) + 1:05d}"
            source_refs = tuple(refs[candidate.unit_id] for candidate in table_units)
            first_header = (
                tuple(_compact(value) for value in table_units[0].rows[0])
                if table_units[0].rows
                else ()
            )
            rows: list[CanonicalPersonalBriefRow] = []
            for table_index, candidate in enumerate(table_units):
                ref = refs[candidate.unit_id]
                for row_index, row in enumerate(candidate.rows):
                    repeated_header = bool(
                        table_index > 0
                        and row_index == 0
                        and first_header
                        and tuple(_compact(value) for value in row) == first_header
                    )
                    rows.append(
                        CanonicalPersonalBriefRow(
                            values=tuple(row),
                            source_refs=(
                                PersonalBriefSourceRef(
                                    **{
                                        **asdict(ref),
                                        "row_index": row_index,
                                    }
                                ),
                            ),
                            status="repeated_header" if repeated_header else "reported",
                        )
                    )
            component = CanonicalPersonalBriefComponent(
                component_id=component_id,
                kind="logical_table",
                section_key=current_section,
                global_order=global_index * _GLOBAL_ORDER_STRIDE,
                text="\n".join(candidate.text for candidate in table_units),
                rows=tuple(rows),
                source_refs=source_refs,
                source_unit_ids=tuple(candidate.unit_id for candidate in table_units),
                confidence=float(entity.confidence),
                semantic_role="source_table",
            )
            components.append(component)
            for candidate in table_units:
                unit_to_component[candidate.unit_id] = component_id
            continue
        ref = refs[unit.unit_id]
        rows = (
            tuple(
                CanonicalPersonalBriefRow(
                    values=tuple(row),
                    source_refs=(
                        PersonalBriefSourceRef(
                            **{
                                **asdict(ref),
                                "row_index": row_index,
                            }
                        ),
                    ),
                )
                for row_index, row in enumerate(unit.rows)
            )
            if unit.kind == "table"
            else ()
        )
        component_id = f"personal_brief:component:{len(components) + 1:05d}"
        component = CanonicalPersonalBriefComponent(
            component_id=component_id,
            kind=_component_kind(unit),
            section_key=current_section,
            global_order=global_index * _GLOBAL_ORDER_STRIDE,
            text=unit.text,
            rows=rows,
            source_refs=(ref,),
            source_unit_ids=(unit.unit_id,),
            confidence=float(
                getattr(context.entity_for_unit(unit.unit_id), "confidence", 1.0) or 0.0
            ),
            semantic_role="source_table" if unit.kind == "table" else "",
        )
        components.append(component)
        unit_to_component[unit.unit_id] = component_id

    components.sort(key=lambda item: (item.global_order, item.component_id))
    connections: list[PersonalBriefConnection] = []
    for left, right in zip(components, components[1:]):
        connections.append(PersonalBriefConnection("next", left.component_id, right.component_id, 1.0))
    for section_key, _label in CANONICAL_PERSONAL_BRIEF_SECTIONS:
        for component in components:
            if component.section_key == section_key:
                connections.append(
                    PersonalBriefConnection("contains", f"section:{section_key}", component.component_id, 1.0)
                )
    for decision in context.decisions:
        if not decision.continues_entity:
            continue
        source = unit_to_component.get(decision.left_unit_id)
        target = unit_to_component.get(decision.right_unit_id)
        if source and target and source != target:
            connections.append(
                PersonalBriefConnection("continues", source, target, float(decision.confidence))
            )

    owned = [unit_id for component in components for unit_id in component.source_unit_ids]
    expected_unit_ids = [
        unit.unit_id for unit in context.units if unit.unit_id not in liability_consumed
    ]
    if liability_reconstruction is not None:
        expected_unit_ids.extend(liability_reconstruction.expected_fragment_ids)
    unassigned = tuple(
        unit_id
        for unit_id in (
            *(
                unit_id
                for unit_id in context.unassigned_unit_ids
                if unit_id not in liability_consumed
            ),
            *(unit_id for unit_id in expected_unit_ids if owned.count(unit_id) != 1),
        )
        if unit_id
    )
    section_presence = {
        section_key: (
            "present"
            if any(component.section_key == section_key for component in components)
            else "absent_from_report"
        )
        for section_key, _label in CANONICAL_PERSONAL_BRIEF_SECTIONS
    }
    pages = list(getattr(parse_result, "pages", None) or [])
    dimensions = tuple(
        (
            int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or index),
            float(getattr(page, "width", 0.0) or 0.0),
            float(getattr(page, "height", 0.0) or 0.0),
        )
        for index, page in enumerate(pages, start=1)
    )
    decisions = list(
        _decision_payload(context, excluded_unit_ids=liability_consumed)
    )
    if liability_reconstruction is not None:
        decisions.extend(liability_reconstruction.continuation_decisions)
    decision_confidences = [decision.best_score for decision in decisions]
    confidence = min(decision_confidences, default=1.0)
    return CanonicalPersonalBriefDocumentIR(
        schema_id=PERSONAL_BRIEF_IR_SCHEMA_ID,
        schema_version=PERSONAL_BRIEF_IR_SCHEMA_VERSION,
        components=tuple(components),
        connections=tuple(connections),
        continuation_decisions=tuple(decisions),
        section_presence=section_presence,
        furniture_source_refs=furniture,
        unassigned_source_unit_ids=tuple(dict.fromkeys(unassigned)),
        source_page_count=len(pages),
        source_unit_count=len(expected_unit_ids),
        confidence=max(0.0, min(1.0, confidence)),
        _page_dimensions=dimensions,
    )


__all__ = [
    "CANONICAL_PERSONAL_BRIEF_SECTIONS",
    "CanonicalPersonalBriefComponent",
    "CanonicalPersonalBriefDocumentIR",
    "CanonicalPersonalBriefRow",
    "PERSONAL_BRIEF_IR_SCHEMA_ID",
    "PERSONAL_BRIEF_IR_SCHEMA_VERSION",
    "PersonalBriefConnection",
    "PersonalBriefContinuationDecision",
    "PersonalBriefSourceRef",
    "build_canonical_personal_brief_document",
]
