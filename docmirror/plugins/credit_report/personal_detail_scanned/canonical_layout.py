# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-owned canonical layout registration for detailed PBOC reports.

The sealed :class:`ParseResult` is source evidence, not the document model used
by the detailed-report extractors.  This module projects its arbitrary logical
fragments onto the fixed PBOC page family using static source evidence and
exposes detached pages and tables that all downstream extractors share.  OCR
acquisition is forbidden here; schema-triggered page repair happens later.

Templates describe semantic page roles and dynamic tables.  They deliberately
do not encode subject names, institution names, account identifiers, or any
other report-specific business value.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

_PRINTED_PAGE_RE = re.compile(r"第\s*(?P<page>\d+)\s*页\s*[,，]?\s*共\s*(?P<total>\d+)\s*页")
_MONTHLY_GRID_RE = re.compile(
    r"20\d{2}\s*年\s*\d{1,2}\s*月\s*[-—一至到~～]\s*20\d{2}\s*年\s*\d{1,2}\s*月.*(?:还款|缴费)记录"
)


@dataclass(frozen=True)
class CanonicalTemplateSpec:
    template_id: str
    anchor_groups: tuple[tuple[str, ...], ...]
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalLayoutProjection:
    pages: tuple[Any, ...]
    evidence_pages: tuple[dict[str, Any], ...]
    registrations: tuple[dict[str, Any], ...]
    fragment_groups: tuple[dict[str, Any], ...]
    unresolved_pages: tuple[int, ...]

    def audit(self) -> dict[str, Any]:
        return {
            "architecture": "canonical_template_registration_v3_static",
            "parse_result_mutated": False,
            "page_count": len(self.pages),
            "registrations": deepcopy(list(self.registrations)),
            "fragment_groups": deepcopy(list(self.fragment_groups)),
            "unresolved_pages": list(self.unresolved_pages),
            "all_extractors_share_canonical_evidence": True,
            "cell_level_ocr_enabled": False,
            "topology_ocr_free": True,
            "template_registration_ocr_used": False,
            "business_repair_after_schema": True,
        }


_TEMPLATES: tuple[CanonicalTemplateSpec, ...] = (
    CanonicalTemplateSpec(
        "report_header_and_identity",
        (
            ("个人信用报告", "报告编号"),
            ("个人基本信息",),
            ("身份信息", "手机号码"),
            ("居住信息", "职业信息"),
            ("配偶信息",),
            ("工作单位", "单位性质"),
            ("居住地址", "居住状况"),
            ("手机号码", "信息更新日期"),
        ),
        (
            "personal_report_metadata",
            "personal_profile",
            "identity_document_records",
            "mobile_phone_records",
            "spouse_records",
            "residence_records",
            "employment_records",
        ),
    ),
    CanonicalTemplateSpec(
        "information_summary",
        (
            ("信息概要",),
            ("信贷交易信息提示",),
            ("信贷交易授信及负债信息概要",),
            ("查询记录概要",),
            ("账户数", "余额", "授信总额"),
        ),
        (
            "personal_detail_summary_records",
            "personal_detail_summary_cells",
        ),
    ),
    CanonicalTemplateSpec(
        "credit_account_detail",
        (
            ("信贷交易信息明细",),
            ("管理机构", "账户标识"),
            ("发卡机构", "账户标识"),
            ("账户状态", "余额"),
            ("当前逾期期数", "当前逾期总额"),
            ("还款记录",),
            ("被追偿信息",),
            ("非循环贷账户",),
            ("循环贷账户一",),
            ("循环贷账户二",),
            ("贷记卡账户",),
            ("准贷记卡账户",),
        ),
        (
            "credit_accounts",
            "repayment_records",
            "personal_detail_account_events",
            "recovery_account_details",
        ),
    ),
    CanonicalTemplateSpec(
        "repayment_responsibility",
        (
            ("相关还款责任信息",),
            ("责任人类型", "还款责任金额"),
            ("主业务借款人", "保证合同编号"),
        ),
        ("repayment_liability_records",),
    ),
    CanonicalTemplateSpec(
        "credit_agreement",
        (
            ("授信协议信息",),
            ("管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"),
            ("授信额度", "授信限额", "授信限额编号", "已用额度", "币种"),
        ),
        ("credit_lines",),
    ),
    CanonicalTemplateSpec(
        "postpaid_detail",
        (
            ("非信贷交易信息明细",),
            ("后付费记录",),
            ("机构名称", "业务类型", "当前缴费状态"),
            ("缴费记录",),
        ),
        ("postpaid_accounts", "postpaid_payment_history"),
    ),
    CanonicalTemplateSpec(
        "public_information",
        (
            ("公共信息明细",),
            ("欠税记录",),
            ("民事判决记录",),
            ("强制执行记录",),
            ("行政处罚记录",),
            ("住房公积金参缴记录",),
            ("执业资格记录",),
            ("行政奖励记录",),
        ),
        ("public_records",),
    ),
    CanonicalTemplateSpec(
        "annotations_and_inquiries",
        (
            ("异议标注",),
            ("查询记录", "查询日期", "查询机构", "查询原因"),
            ("机构查询记录明细",),
            ("本人查询记录明细",),
        ),
        ("statements", "annotations", "inquiry_records"),
    ),
    CanonicalTemplateSpec(
        "report_explanation",
        (
            ("报告说明",),
            ("还款状态说明",),
            ("贷记卡账户", "还款状态说明"),
            ("后付费业务", "还款状态说明"),
        ),
        ("report_notes",),
    ),
)


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox") if isinstance(value, Mapping) else getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    box = tuple(_finite(item) for item in raw[:4])
    return box if box[2] > box[0] and box[3] > box[1] else None


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _raw_rows(table: Any) -> list[list[str]]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw = metadata.get("raw_rows")
    if isinstance(raw, list) and raw:
        return [[str(cell or "").replace("\n", "").strip() for cell in row] for row in raw if isinstance(row, list)]
    rows: list[list[str]] = []
    headers = [str(value or "") for value in getattr(table, "headers", None) or []]
    if headers:
        rows.append(headers)
    for row in getattr(table, "rows", None) or []:
        rows.append([str(getattr(cell, "text", cell) or "") for cell in getattr(row, "cells", None) or []])
    return rows


def _page_text(page: Any, evidence: Mapping[str, Any]) -> str:
    values = [str(line.get("text") or line.get("content") or "") for line in evidence.get("lines") or []]
    for table in getattr(page, "tables", None) or []:
        values.extend(cell for row in _raw_rows(table) for cell in row)
    values.extend(str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or [])
    return "\n".join(value for value in values if value)


def _template_score(text: str, spec: CanonicalTemplateSpec) -> tuple[int, int]:
    compact = _compact(text)
    matched = [group for group in spec.anchor_groups if all(marker in compact for marker in group)]
    strength = sum(len(group) for group in matched)
    return len(matched), strength


def _classify(text: str) -> tuple[str, float, tuple[str, ...]] | None:
    compact = _compact(text)
    # A canonical top-level heading is stronger evidence than repeated
    # category names.  The latter also occur in information-summary tables
    # and in the report explanation, so allowing them to win by frequency
    # registers otherwise valid pages to the wrong canonical role.
    explicit_roles = (
        ("report_explanation", ("报告说明", "还款状态说明")),
        ("annotations_and_inquiries", ("查询记录明细", "异议标注")),
        ("public_information", ("公共信息明细",)),
        ("postpaid_detail", ("非信贷交易信息明细",)),
        ("repayment_responsibility", ("相关还款责任信息",)),
        ("credit_agreement", ("授信协议信息",)),
        ("credit_account_detail", ("信贷交易信息明细",)),
        ("information_summary", ("信息概要",)),
        ("report_header_and_identity", ("个人基本信息", "个人信用报告")),
    )
    for template_id, headings in explicit_roles:
        matched = tuple(heading for heading in headings if heading in compact)
        if matched:
            return template_id, min(0.99, 0.92 + 0.03 * len(matched)), matched

    # A physical page may begin between account cards and therefore omit the
    # enclosing section heading.  The numbered card heading plus its printed
    # agreement-id label is a canonical account-detail signature, not a
    # footer-based or generic previous-page guess.
    if re.search(
        r"\u8d26\u6237\s*\d{1,3}\s*[\uff08(][^\uff09)]{0,40}\u6388\u4fe1\u534f\u8bae\u6807\u8bc6",
        compact,
    ):
        return "credit_account_detail", 0.94, ("canonical_numbered_account_heading",)

    if len(compact) < 8:
        return None

    ranked = sorted(
        (
            (*_template_score(text, spec), spec)
            for spec in _TEMPLATES
        ),
        key=lambda item: (-item[0], -item[1], item[2].template_id),
    )
    count, strength, spec = ranked[0]
    if count <= 0:
        if _MONTHLY_GRID_RE.search(compact):
            return "credit_account_detail", 0.90, ("monthly_grid_signature",)
        return None
    confidence = min(0.99, 0.70 + 0.08 * count + 0.02 * strength)
    signals = tuple("+".join(group) for group in spec.anchor_groups if all(marker in compact for marker in group))
    return spec.template_id, confidence, signals


def _template_spec(template_id: str) -> CanonicalTemplateSpec | None:
    return next((spec for spec in _TEMPLATES if spec.template_id == template_id), None)


def _printed_identity(evidence: Mapping[str, Any]) -> tuple[int, int] | None:
    matches = {
        (int(match.group("page")), int(match.group("total")))
        for line in evidence.get("lines") or []
        if isinstance(line, Mapping)
        for match in _PRINTED_PAGE_RE.finditer(str(line.get("text") or line.get("content") or ""))
    }
    valid = {(page, total) for page, total in matches if 1 <= page <= total}
    return next(iter(valid)) if len(valid) == 1 else None


def _overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    return intersection / area


def _line_in_box(line: Mapping[str, Any], box: Sequence[float]) -> bool:
    candidate = _bbox(line)
    if candidate is None:
        return False
    center = ((candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0)
    return (
        box[0] <= center[0] <= box[2]
        and box[1] <= center[1] <= box[3]
    ) or _overlap_ratio(candidate, box) >= 0.35


def _project_table(
    table: Any,
    *,
    template_id: str,
    transform: Callable[[Sequence[float]], list[float]],
) -> Any:
    metadata = deepcopy(dict(getattr(table, "metadata", None) or {}))
    rows = _raw_rows(table)
    cell_boxes = metadata.get("cell_bboxes")
    if rows:
        metadata["raw_rows"] = rows
    if isinstance(cell_boxes, list):
        metadata["source_cell_bboxes"] = deepcopy(cell_boxes)
        metadata["cell_bboxes"] = [
            [transform(box) if isinstance(box, (list, tuple)) and len(box) == 4 else box for box in row]
            if isinstance(row, list)
            else row
            for row in cell_boxes
        ]
    metadata["canonical_template_id"] = template_id
    table_box = _bbox(table)
    return SimpleNamespace(
        table_id=str(getattr(table, "table_id", "") or ""),
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=transform(table_box) if table_box is not None else None,
        confidence=getattr(table, "confidence", None),
    )


class PBOCCanonicalTemplateAssembler:
    """Register arbitrary ParseResult fragments against canonical PBOC pages."""

    def __init__(
        self,
        parse_result: Any,
        *,
        topology: Any,
        reading_order_by_logical: Mapping[int, int],
        source_evidence_loader: Callable[[], list[dict[str, Any]]],
        issue_owner: Any,
        source_page_loader: Callable[[], Iterable[Any]] | None = None,
    ) -> None:
        self.parse_result = parse_result
        self.topology = topology
        self.reading_order = {int(key): int(value) for key, value in reading_order_by_logical.items()}
        self.source_evidence_loader = source_evidence_loader
        self.issue_owner = issue_owner
        self.source_page_loader = source_page_loader

    def build(self) -> CanonicalLayoutProjection:
        source_evidence = self.source_evidence_loader()
        source_pages = (
            list(self.source_page_loader())
            if callable(self.source_page_loader)
            else list(getattr(self.parse_result, "pages", None) or [])
        )
        raw_pages = {
            int(getattr(page, "page_number", 0) or index): page
            for index, page in enumerate(source_pages, start=1)
        }
        evidence = {
            int(page.get("page") or 0): deepcopy(page)
            for page in source_evidence
            if isinstance(page, Mapping) and int(page.get("page") or 0) > 0
        }
        for logical, page in raw_pages.items():
            if logical in evidence:
                continue
            evidence[logical] = {
                "page": logical,
                "source_page": int(getattr(page, "source_page_number", 0) or logical),
                "page_width": _finite(getattr(page, "width", 0)),
                "page_height": _finite(getattr(page, "height", 0)),
                "lines": [
                    {
                        "text": str(getattr(block, "content", "") or ""),
                        "bbox": list(_bbox(block) or ()),
                        "source": "sealed_page_text_fallback",
                    }
                    for block in getattr(page, "texts", None) or []
                    if str(getattr(block, "content", "") or "").strip() and _bbox(block) is not None
                ],
            }
        registrations: dict[int, dict[str, Any]] = {}
        for logical, page_evidence in evidence.items():
            page = raw_pages.get(logical, SimpleNamespace(tables=[], texts=[]))
            result = _classify(_page_text(page, page_evidence))
            if result is not None:
                template_id, confidence, signals = result
                registrations[logical] = self._registration(
                    logical, template_id, confidence, "source_page_evidence", signals, page_evidence
                )
                continue
            if len(_compact(_page_text(page, page_evidence))) < 8:
                registrations[logical] = self._registration(
                    logical,
                    "blank_fragment",
                    1.0,
                    "explicitly_blank_fragment",
                    ("no_business_content",),
                    page_evidence,
                    status="blank",
                )

        # Canonical pages often start in the middle of an account or repeated
        # table. A remaining fragment may inherit only when an explicit
        # canonical continuation relation is present; document proximity or
        # page shape alone is not sufficient.
        ordered = sorted(evidence, key=lambda value: self.reading_order.get(value, value))
        active_template = ""
        active_logical = 0
        for logical in ordered:
            registration = registrations.get(logical)
            if registration and registration["status"] == "registered":
                active_template = str(registration["template_id"])
                active_logical = logical
                continue
            if registration:
                continue
            text = _compact(_page_text(raw_pages.get(logical, SimpleNamespace(tables=[], texts=[])), evidence[logical]))
            continuation_signals: list[str] = []
            continuation_check = getattr(self.issue_owner, "tables_continue", None)
            previous_page = raw_pages.get(active_logical)
            current_page = raw_pages.get(logical)
            if active_template and callable(continuation_check) and previous_page is not None and current_page is not None:
                previous_ids = [
                    str(getattr(table, "table_id", "") or "")
                    for table in getattr(previous_page, "tables", None) or ()
                ]
                current_ids = [
                    str(getattr(table, "table_id", "") or "")
                    for table in getattr(current_page, "tables", None) or ()
                ]
                if any(
                    left and right and continuation_check(left, right) is True
                    for left in previous_ids
                    for right in current_ids
                ):
                    continuation_signals.append("explicit_table_continuation")
            if active_template == "credit_account_detail" and _MONTHLY_GRID_RE.search(text):
                continuation_signals.append("canonical_monthly_grid_continuation")
            if active_template == "annotations_and_inquiries" and re.search(
                r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}.*(?:贷后管理|贷款审批|信用卡审批|本人查询)",
                text,
            ):
                continuation_signals.append("canonical_inquiry_row_continuation")
            if active_template == "postpaid_detail" and (
                "缴费记录" in text
                or bool(re.search(r"20\d{2}[N1-7*]{4,}", text))
            ):
                continuation_signals.append("canonical_postpaid_history_continuation")
            if active_template == "report_explanation" and re.search(
                r"(?:N-正常|C-结清|G-结束|#-账户|1-逾期)",
                text,
            ):
                continuation_signals.append("canonical_explanation_continuation")
            # A table merely existing, or a page merely containing enough
            # text, is not proof that it continues the preceding section.
            # Under the closed canonical catalog an unproven page stays
            # unresolved and becomes eligible for the one business-repair
            # OCR pass instead of being silently registered to the wrong role.
            if active_template and continuation_signals:
                registrations[logical] = self._registration(
                    logical,
                    active_template,
                    0.78,
                    "canonical_flow_continuation",
                    (
                        "preceding_template_role",
                        *continuation_signals,
                        "no_conflicting_section_anchor",
                    ),
                    evidence[logical],
                )
                active_logical = logical

        unresolved = tuple(
            logical
            for logical in ordered
            if logical not in registrations or registrations[logical].get("status") == "unresolved"
        )
        for logical in unresolved:
            registrations[logical] = self._registration(
                logical,
                "unresolved",
                0.0,
                "canonical_registration_exhausted",
                ("no_generic_layout_fallback",),
                evidence[logical],
                status="unresolved",
            )
            self._record_registration_failure(logical, registrations[logical])

        groups = self._fragment_groups(evidence, registrations)
        canonical_pages: list[Any] = []
        canonical_evidence: list[dict[str, Any]] = []
        group_audits: list[dict[str, Any]] = []
        for group in groups:
            if group["status"] == "blank":
                continue
            if group["status"] == "unresolved":
                # No generic extraction is allowed from an unregistered page.
                continue
            page, page_evidence, audit = self._assemble_group(group, raw_pages, evidence, registrations)
            canonical_pages.append(page)
            canonical_evidence.append(page_evidence)
            group_audits.append(audit)

        return CanonicalLayoutProjection(
            pages=tuple(canonical_pages),
            evidence_pages=tuple(canonical_evidence),
            registrations=tuple(registrations[key] for key in sorted(registrations, key=lambda v: self.reading_order.get(v, v))),
            fragment_groups=tuple(group_audits),
            unresolved_pages=unresolved,
        )

    def _registration(
        self,
        logical: int,
        template_id: str,
        confidence: float,
        basis: str,
        signals: Iterable[str],
        evidence: Mapping[str, Any],
        *,
        status: str = "registered",
    ) -> dict[str, Any]:
        printed = _printed_identity(evidence)
        spec = _template_spec(template_id)
        return {
            "logical_page": logical,
            "source_page": int(evidence.get("source_page") or logical),
            "template_id": template_id,
            "status": status,
            "confidence": round(float(confidence), 4),
            "basis": basis,
            "signals": list(signals),
            **({"printed_page": printed[0], "printed_total": printed[1]} if printed else {}),
            **({"affected_source_datasets": list(spec.datasets)} if spec else {}),
        }

    def _fragment_groups(
        self,
        evidence: Mapping[int, Mapping[str, Any]],
        registrations: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        printed_by_source: dict[int, list[tuple[int, int, str]]] = {}
        for logical, registration in registrations.items():
            printed_page = int(registration.get("printed_page") or 0)
            source_page = int(registration.get("source_page") or 0)
            if printed_page and source_page:
                printed_by_source.setdefault(source_page, []).append(
                    (printed_page, logical, str(registration.get("template_id") or ""))
                )
        grouped: dict[tuple[str, int], list[int]] = {}
        for logical in sorted(evidence, key=lambda value: self.reading_order.get(value, value)):
            registration = registrations[logical]
            printed_page = int(registration.get("printed_page") or 0)
            if not printed_page and evidence[logical].get("plugin_static_subpage"):
                source_page = int(registration.get("source_page") or evidence[logical].get("source_page") or 0)
                template_id = str(registration.get("template_id") or "")
                compatible = {
                    page
                    for page, _seed_logical, seed_template in printed_by_source.get(source_page, ())
                    if seed_template == template_id
                }
                if len(compatible) == 1:
                    # This is another crop of the same canonical printed page,
                    # not a new template variant.  Its source crop determines
                    # where it lands on the virtual page canvas.
                    printed_page = next(iter(compatible))
            key = ("printed", printed_page) if printed_page else ("logical", logical)
            grouped.setdefault(key, []).append(logical)
        result: list[dict[str, Any]] = []
        for key, logicals in grouped.items():
            statuses = {str(registrations[logical].get("status") or "unresolved") for logical in logicals}
            status = "unresolved" if "unresolved" in statuses else "blank" if statuses == {"blank"} else "registered"
            template_ids = [
                str(registrations[logical].get("template_id") or "")
                for logical in logicals
                if registrations[logical].get("status") == "registered"
            ]
            template_id = template_ids[0] if template_ids else "unresolved"
            result.append(
                {
                    "group_key": f"{key[0]}:{key[1]}",
                    "canonical_page": key[1] if key[0] == "printed" else self.reading_order.get(logicals[0], logicals[0]),
                    "logical_pages": logicals,
                    "template_id": template_id,
                    "status": status,
                }
            )
        return result

    def _assemble_group(
        self,
        group: Mapping[str, Any],
        raw_pages: Mapping[int, Any],
        evidence: Mapping[int, Mapping[str, Any]],
        registrations: Mapping[int, Mapping[str, Any]],
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        logicals = [int(value) for value in group["logical_pages"]]
        representative = logicals[0]
        template_id = str(group["template_id"])
        transforms, width, height, coverage = self._fragment_transforms(logicals, evidence)
        output_lines: list[dict[str, Any]] = []
        tables: list[Any] = []
        table_boxes: list[tuple[float, float, float, float]] = []

        for logical in logicals:
            local_evidence = evidence[logical]
            transform = transforms[logical]
            local_lines = [line for line in local_evidence.get("lines") or [] if isinstance(line, Mapping)]
            raw_page = raw_pages.get(logical)
            if raw_page is not None:
                for table in getattr(raw_page, "tables", None) or []:
                    projected = _project_table(
                        table,
                        template_id=template_id,
                        transform=transform,
                    )
                    projected.metadata["source_logical_page"] = logical
                    projected.metadata["source_page"] = int(local_evidence.get("source_page") or logical)
                    tables.append(projected)
                    if (box := _bbox(projected)) is not None:
                        table_boxes.append(box)
            for line in local_lines:
                box = _bbox(line)
                if box is None:
                    continue
                output_lines.append(
                    {
                        **deepcopy(dict(line)),
                        "source_bbox": list(box),
                        "bbox": transform(box),
                        "page": representative,
                        "source_logical_page": logical,
                        "canonical_page": int(group["canonical_page"]),
                        "canonical_template_id": template_id,
                    }
                )

        output_lines.sort(key=lambda line: ((_bbox(line) or (0, 0, 0, 0))[1], (_bbox(line) or (0, 0, 0, 0))[0]))
        texts = [
            SimpleNamespace(content=str(line.get("text") or ""), bbox=list(line.get("bbox") or []))
            for line in output_lines
            if not any(_line_in_box(line, box) for box in table_boxes)
        ]
        source_pages = sorted({int(evidence[logical].get("source_page") or logical) for logical in logicals})
        page = SimpleNamespace(
            page_number=representative,
            source_page_number=source_pages[0] if source_pages else representative,
            width=width,
            height=height,
            tables=tables,
            texts=texts,
            coordinate_transform={
                "kind": "plugin_canonical_template",
                "canonical_page": int(group["canonical_page"]),
                "fragment_logical_pages": logicals,
                "source_page_numbers": source_pages,
            },
            canonical_template_id=template_id,
            canonical_fragment_logical_pages=tuple(logicals),
        )
        page_evidence = {
            "page": representative,
            "canonical_page": int(group["canonical_page"]),
            "source_page": source_pages[0] if source_pages else representative,
            "source_pages": source_pages,
            "fragment_logical_pages": logicals,
            "page_width": width,
            "page_height": height,
            "canonical_template_id": template_id,
            "canonical_coverage_status": "full" if coverage >= 0.985 else "partial",
            "canonical_coverage_ratio": round(coverage, 4),
            "lines": output_lines,
        }
        audit = {
            "canonical_page": int(group["canonical_page"]),
            "template_id": template_id,
            "fragment_logical_pages": logicals,
            "source_pages": source_pages,
            "coverage_ratio": round(coverage, 4),
            "coverage_status": "full" if coverage >= 0.985 else "partial",
            "joined_fragment_count": len(logicals),
        }
        return page, page_evidence, audit

    def _fragment_transforms(
        self,
        logicals: Sequence[int],
        evidence: Mapping[int, Mapping[str, Any]],
    ) -> tuple[dict[int, Callable[[Sequence[float]], list[float]]], float, float, float]:
        placements: dict[int, tuple[float, float, float, float, float, float]] = {}
        source_boxes: list[tuple[float, float, float, float]] = []
        for logical in logicals:
            geometry = self.topology.geometry(logical) if self.topology is not None else None
            evidence_crop = evidence[logical].get("source_crop_bbox")
            crop = (
                tuple(float(value) for value in evidence_crop)
                if isinstance(evidence_crop, (list, tuple)) and len(evidence_crop) == 4
                else getattr(geometry, "source_crop_bbox", None)
            )
            local_width = max(1.0, _finite(evidence[logical].get("page_width")))
            local_height = max(1.0, _finite(evidence[logical].get("page_height")))
            if crop is not None:
                source_box = tuple(float(value) for value in crop)
            elif len(logicals) == 1:
                source_box = (0.0, 0.0, local_width, local_height)
            else:
                prior_bottom = max((box[3] for box in source_boxes), default=0.0)
                source_box = (0.0, prior_bottom, local_width, prior_bottom + local_height)
            source_boxes.append(source_box)
            placements[logical] = (*source_box, local_width, local_height)

        union = (
            min(box[0] for box in source_boxes),
            min(box[1] for box in source_boxes),
            max(box[2] for box in source_boxes),
            max(box[3] for box in source_boxes),
        )
        transforms: dict[int, Callable[[Sequence[float]], list[float]]] = {}
        for logical, placement in placements.items():
            sx0, sy0, sx1, sy1, local_width, local_height = placement
            scale_x = (sx1 - sx0) / local_width
            scale_y = (sy1 - sy0) / local_height

            def transform(
                box: Sequence[float],
                *,
                x0: float = sx0,
                y0: float = sy0,
                fx: float = scale_x,
                fy: float = scale_y,
                ux0: float = union[0],
                uy0: float = union[1],
            ) -> list[float]:
                return [
                    x0 - ux0 + float(box[0]) * fx,
                    y0 - uy0 + float(box[1]) * fy,
                    x0 - ux0 + float(box[2]) * fx,
                    y0 - uy0 + float(box[3]) * fy,
                ]

            transforms[logical] = transform

        union_area = max(1.0, (union[2] - union[0]) * (union[3] - union[1]))
        covered_area = sum(max(0.0, (box[2] - box[0]) * (box[3] - box[1])) for box in source_boxes)
        # Overlaps can make the simple sum exceed the union.  They represent
        # duplicated evidence rather than additional coverage.
        coverage = min(1.0, covered_area / union_area)
        return transforms, union[2] - union[0], union[3] - union[1], coverage

    def _record_registration_failure(self, logical: int, registration: Mapping[str, Any]) -> None:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            make_issue,
            record_issue,
        )

        record_issue(
            self.issue_owner,
            make_issue(
                category="ocr_structure_correction",
                issue_code="canonical_page_registration_failed",
                message=(
                    "The logical page could not be registered to a canonical PBOC layout from static source "
                    "evidence; generic reconstruction was not used and business repair may retry the page."
                ),
                parser_stage="canonical_template_registration",
                observed_value={
                    "logical_page": logical,
                    "source_page": registration.get("source_page"),
                },
                confidence=0.0,
                source_refs=(
                    {
                        "source": "canonical_template_registration",
                        "logical_page": logical,
                        "source_page": int(registration.get("source_page") or logical),
                        "geometry_scope": "logical_page",
                    },
                ),
                reason_codes=(
                    "canonical_layout_unresolved_from_source_evidence",
                    "schema_triggered_page_repair_eligible",
                    "no_generic_layout_fallback",
                    "normalized_values_withheld_for_page",
                ),
            ),
        )


__all__ = [
    "CanonicalLayoutProjection",
    "CanonicalTemplateSpec",
    "PBOCCanonicalTemplateAssembler",
]
