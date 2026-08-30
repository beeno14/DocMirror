# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-world record decoders for canonical PBOC enterprise reports.

The canonical enterprise layout contains a few business record families that
do not share the ordinary credit-account table shape.  This module decodes
those families from :class:`CanonicalEnterpriseDocumentIR` only.  It never
consults ``ParseResult`` or physical page objects, so page boundaries are
provenance rather than parsing boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator

from docmirror.plugins.credit_report.currency_codes import normalize_currency_code
from docmirror.plugins.credit_report.enterprise_native.business_values import opaque_identifier
from docmirror.plugins.credit_report.enterprise_native.ir import (
    CanonicalEnterpriseDocumentIR,
)
from docmirror.plugins.credit_report.value_utils import (
    compact_text,
    parse_number,
    stable_record_id,
)

RECOVERED_BUSINESS_DATASET = "enterprise_recovered_business_accounts"
ACCOUNT_ANNOTATION_DATASET = "enterprise_account_annotations"
GROUPED_RESPONSIBILITY_DATASET = "enterprise_repayment_responsibility_group_details"
CREDIT_DETAIL_GROUP_DATASET = "enterprise_credit_detail_groups"

_MISSING_MARKERS = frozenset({"", "--", "-", "—"})
_ACCOUNT_IDENTIFIER_RE = re.compile(r"[0-9A-Z]{6,}")
_DATE_PATTERN = r"((?:18|19|20|21)\d{2})[-年./](\d{1,2})[-月./](\d{1,2})日?"
_ACCOUNT_NOTE_ANCHOR_RE = re.compile(r"(?:对于)?账户编号为[“\"']?(?P<identifier>[0-9A-Z]{6,})[”\"']?")
_ACCOUNT_NOTE_TERMINATORS = (
    "未结清信贷",
    "已结清信贷",
    "相关还款责任",
    "非信贷记录明细",
    "公共记录明细",
    "声明及异议标注信息",
)

_RECOVERY_SECONDARY_FIELDS = (
    "five_tier_class",
    "last_repayment_date",
    "last_repayment_amount",
    "repayment_method",
    "history_status",
    "original_creditor_name",
    "original_debt_type",
)

_CREDIT_DETAIL_GROUP_TOKEN_RE = re.compile(
    r"(?P<phase>未结清信贷|已结清信贷|相关还款责任)|"
    r"(?P<label>为担保交易承担的相关还款责任|除贴现外的其他业务|"
    r"银行承兑汇票和信用证|银行保函及其他业务|被追偿业务|"
    r"欠息|中长期借款|短期借款|循环透支|贴现|授信信息)"
    r"共(?P<count>\d+)笔"
)

_CREDIT_DETAIL_GROUP_TARGETS = {
    "被追偿业务": ("account_card", "enterprise_credit_accounts"),
    "欠息": ("interest_arrears", "enterprise_interest_arrears"),
    "中长期借款": ("account_card", "enterprise_credit_accounts"),
    "短期借款": ("account_card", "enterprise_credit_accounts"),
    "循环透支": ("account_card", "enterprise_credit_accounts"),
    "贴现": ("grouped_credit", "enterprise_displayed_credit_summary"),
    "银行承兑汇票和信用证": (
        "grouped_credit",
        "enterprise_displayed_credit_summary",
    ),
    "银行保函及其他业务": (
        "grouped_credit",
        "enterprise_displayed_credit_summary",
    ),
    "授信信息": ("facility", "enterprise_credit_facilities"),
    "除贴现外的其他业务": (
        "identified_repayment_responsibility",
        "enterprise_repayment_responsibility_accounts",
    ),
    "为担保交易承担的相关还款责任": (
        "grouped_repayment_responsibility",
        "enterprise_repayment_responsibility_group_details",
    ),
}

_ACCOUNT_ANNOTATION_LABELS = {
    "data_provider_statement": "数据提供机构说明",
    "subject_statement": "信息主体声明",
    "credit_bureau_statement": "征信中心说明",
    "dispute_processing": "异议处理中",
}


@dataclass(frozen=True)
class _FlowItem:
    page: int
    kind: str
    value: Any
    component_id: str
    unit_id: str

    def source_ref(self, *, table_id: str = "", row: int | None = None) -> dict[str, Any]:
        ref: dict[str, Any] = {
            "source": "canonical_enterprise_document_ir",
            "page": self.page,
            "component_id": self.component_id,
            "unit_id": self.unit_id,
        }
        if table_id:
            ref["table_id"] = table_id
        if row is not None:
            ref["row"] = row
        return ref


@dataclass(frozen=True)
class _TextSpan:
    start: int
    end: int
    ref: dict[str, Any]


def _require_document(document: CanonicalEnterpriseDocumentIR) -> None:
    if not isinstance(document, CanonicalEnterpriseDocumentIR):
        raise TypeError("canonical record extraction requires CanonicalEnterpriseDocumentIR")


def _flow_items(document: CanonicalEnterpriseDocumentIR) -> Iterator[_FlowItem]:
    """Yield component-local IR segments in canonical reading order."""

    emitted = False
    for component in sorted(document.components, key=lambda item: item.global_order):
        segments = sorted(
            component.segments,
            key=lambda segment: (
                int(segment.get("source_page") or 0),
                int(segment.get("source_order") or 0),
                int(segment.get("source_index") or 0),
            ),
        )
        for segment in segments:
            kind = str(segment.get("kind") or "")
            page = int(segment.get("source_page") or 0)
            unit_id = str(segment.get("id") or "")
            if kind == "table":
                rows = tuple(tuple(compact_text(cell) for cell in row) for row in segment.get("rows") or ())
                if not rows:
                    continue
                value: Any = (str(segment.get("table_id") or ""), rows)
                yield _FlowItem(page, "table", value, component.component_id, unit_id)
                emitted = True
                continue
            text = str(segment.get("text") or "")
            if text:
                yield _FlowItem(page, "text", text, component.component_id, unit_id)
                emitted = True

    if emitted:
        return

    # A defensive fallback for hand-built IR fixtures.  Production IR always
    # has component segments, and this still remains an IR-only input path.
    for index, (page, kind, value) in enumerate(document.page_flow, start=1):
        normalized: Any = value
        if kind == "table":
            table_id, rows = value
            normalized = (
                str(table_id or ""),
                tuple(tuple(compact_text(cell) for cell in row) for row in rows),
            )
        yield _FlowItem(
            int(page or 0),
            str(kind),
            normalized,
            "canonical_page_flow",
            f"canonical_page_flow:{index}",
        )


def _date(value: Any) -> str | None:
    raw = compact_text(value)
    match = re.fullmatch(_DATE_PATTERN, raw)
    if not match:
        return None
    try:
        return date(*(int(match.group(index)) for index in range(1, 4))).isoformat()
    except ValueError:
        return None


def _identifier(value: Any) -> str:
    raw = compact_text(value).upper()
    match = _ACCOUNT_IDENTIFIER_RE.search(raw)
    return match.group(0) if match else ""


def _source_state(value: Any, normalized: Any) -> str:
    return (
        "not_reported"
        if compact_text(value) in _MISSING_MARKERS
        else ("reported" if normalized not in (None, "") else "unresolved")
    )


def _column_source_state(index: int | None, value: Any, normalized: Any) -> str:
    return "not_applicable" if index is None else _source_state(value, normalized)


def _currency(value: Any) -> str:
    raw = compact_text(value)
    return normalize_currency_code(raw) or raw


def _amount_unit(currency: str) -> str:
    return f"{currency}_10K" if currency else "10K"


def _column_index(headers: tuple[str, ...], *labels: str) -> int | None:
    normalized_labels = tuple(compact_text(label) for label in labels)
    for index, header in enumerate(headers):
        if compact_text(header) in normalized_labels:
            return index
    for index, header in enumerate(headers):
        normalized = compact_text(header)
        if any(label and label in normalized for label in normalized_labels):
            return index
    return None


def _cell(row: tuple[str, ...], index: int | None) -> str:
    return row[index] if index is not None and index < len(row) else ""


def _align_row(row: tuple[str, ...], headers: tuple[str, ...]) -> tuple[str, ...]:
    if len(row) + 1 == len(headers) and headers and not compact_text(headers[0]):
        return ("", *row)
    if len(row) < len(headers):
        return (*row, *("" for _ in range(len(headers) - len(row))))
    return row


def _append_ref(record: dict[str, Any], ref: dict[str, Any]) -> None:
    refs = record.setdefault("source_refs", [])
    if ref not in refs:
        refs.append(ref)


def _source_bounds(record: dict[str, Any]) -> None:
    refs = [ref for ref in record.get("source_refs", ()) if int(ref.get("page") or 0) > 0]
    if not refs:
        return
    pages = [int(ref["page"]) for ref in refs]
    record["source_page"] = min(pages)
    record["source_page_end"] = max(pages)
    table_ids = [str(ref.get("table_id") or "") for ref in refs if ref.get("table_id")]
    if table_ids:
        record["source_table_id"] = table_ids[0]
        record["source_table_id_end"] = table_ids[-1]


def _is_recovery_primary_header(row: tuple[str, ...]) -> bool:
    signature = "".join(row)
    return all(
        marker in signature
        for marker in (
            "账户编号",
            "债权机构",
            "业务种类",
            "接收日期",
            "借款金额",
            "余额",
            "信息报告日期",
        )
    )


def _is_recovery_secondary_header(row: tuple[str, ...]) -> bool:
    signature = "".join(row)
    return all(marker in signature for marker in ("五级分类", "最近一次还款日期", "初始债权人名称", "原债权种类"))


def extract_recovered_business_accounts(
    document: CanonicalEnterpriseDocumentIR,
) -> list[dict[str, Any]]:
    """Decode every canonical ``被追偿业务`` account card.

    The source card has two labelled rows.  The second row may be placed in a
    later IR table segment after pagination reconstruction; a pending record is
    therefore closed only by the next primary account row or the end of flow.
    """

    _require_document(document)
    records: list[dict[str, Any]] = []
    primary_headers: tuple[str, ...] | None = None
    secondary_headers: tuple[str, ...] | None = None
    pending: dict[str, Any] | None = None
    in_recovery_section = False

    def finish_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        for field in _RECOVERY_SECONDARY_FIELDS:
            pending.setdefault(field, None)
            status_field = "history_status_source_state" if field == "history_status" else f"{field}_status"
            pending.setdefault(status_field, "unresolved")
        pending["sequence"] = len(records) + 1
        _source_bounds(pending)
        records.append(pending)
        pending = None

    for item in _flow_items(document):
        if item.kind == "text":
            text = compact_text(item.value)
            if re.search(r"被追偿业务共\d+笔", text):
                in_recovery_section = True
            elif in_recovery_section and any(
                marker in text
                for marker in (
                    "未结清信贷",
                    "已结清信贷",
                    "欠息共",
                    "中长期借款共",
                    "短期借款共",
                    "循环透支共",
                    "贴现共",
                    "相关还款责任",
                    "非信贷记录明细",
                )
            ):
                finish_pending()
                primary_headers = None
                secondary_headers = None
                in_recovery_section = False
            continue
        if item.kind != "table":
            continue
        table_id, raw_rows = item.value
        for row_index, raw_row in enumerate(raw_rows):
            row = tuple(compact_text(value) for value in raw_row)
            if _is_recovery_primary_header(row):
                in_recovery_section = True
                primary_headers = row
                continue
            row_signature = "".join(row)
            if (
                in_recovery_section
                and "账户编号" in row_signature
                and "授信机构" in row_signature
                and "开立日期" in row_signature
            ):
                finish_pending()
                primary_headers = None
                secondary_headers = None
                in_recovery_section = False
                continue
            if _is_recovery_secondary_header(row):
                secondary_headers = row
                continue
            if not in_recovery_section or primary_headers is None or secondary_headers is None:
                continue

            primary_row = _align_row(row, primary_headers)
            account_index = _column_index(primary_headers, "账户编号")
            account_identifier = _identifier(_cell(primary_row, account_index))
            receive_index = _column_index(primary_headers, "接收日期")
            receive_date = _date(_cell(primary_row, receive_index))
            if account_identifier and receive_date:
                finish_pending()
                creditor_index = _column_index(primary_headers, "债权机构")
                business_index = _column_index(primary_headers, "业务种类", "业务类型")
                currency_index = _column_index(primary_headers, "币种")
                amount_index = _column_index(primary_headers, "借款金额")
                balance_index = _column_index(primary_headers, "余额")
                close_index = _column_index(primary_headers, "关闭日期")
                snapshot_index = _column_index(primary_headers, "信息报告日期")
                raw_currency = _cell(primary_row, currency_index)
                currency = _currency(raw_currency)
                raw_amount = _cell(primary_row, amount_index)
                raw_balance = _cell(primary_row, balance_index)
                raw_close_date = _cell(primary_row, close_index)
                raw_snapshot_date = _cell(primary_row, snapshot_index)
                loan_amount = parse_number(raw_amount)
                balance = parse_number(raw_balance)
                close_date = _date(raw_close_date)
                snapshot_date = _date(raw_snapshot_date)
                creditor = _cell(primary_row, creditor_index)
                business_type = _cell(primary_row, business_index)
                pending = {
                    "recovery_account_id": stable_record_id(
                        "enterprise_recovered_business_account", account_identifier
                    ),
                    "account_id": f"credit_account:{account_identifier}",
                    "account_identifier": account_identifier,
                    "business_category": "被追偿业务",
                    "institution": creditor,
                    "creditor_institution": creditor,
                    "business_type": business_type,
                    "receive_date": receive_date,
                    "receive_date_status": _source_state(_cell(primary_row, receive_index), receive_date),
                    "currency": currency,
                    "currency_status": _source_state(raw_currency, currency),
                    "amount_unit": _amount_unit(currency),
                    "loan_amount": loan_amount,
                    "loan_amount_status": _source_state(raw_amount, loan_amount),
                    "balance": balance,
                    "balance_status": _source_state(raw_balance, balance),
                    "close_date": close_date,
                    "close_date_status": _source_state(raw_close_date, close_date),
                    "status": "settled" if close_date else "active",
                    "status_source_state": "derived",
                    "snapshot_date": snapshot_date,
                    "snapshot_date_status": _source_state(raw_snapshot_date, snapshot_date),
                    "source": "canonical_recovered_business_account",
                    "confidence": 1.0,
                    "source_refs": [item.source_ref(table_id=table_id, row=row_index)],
                }
                continue

            if pending is None:
                continue
            secondary_row = _align_row(row, secondary_headers)
            class_index = _column_index(secondary_headers, "五级分类")
            repay_date_index = _column_index(secondary_headers, "最近一次还款日期")
            repay_amount_index = _column_index(secondary_headers, "最近一次还款总额")
            repayment_index = _column_index(secondary_headers, "最近一次还款形式")
            history_index = _column_index(secondary_headers, "历史表现")
            original_creditor_index = _column_index(secondary_headers, "初始债权人名称")
            original_type_index = _column_index(secondary_headers, "原债权种类")
            raw_class = _cell(secondary_row, class_index)
            raw_repay_date = _cell(secondary_row, repay_date_index)
            raw_repay_amount = _cell(secondary_row, repay_amount_index)
            raw_repayment = _cell(secondary_row, repayment_index)
            raw_history = _cell(secondary_row, history_index)
            raw_original_creditor = _cell(secondary_row, original_creditor_index)
            raw_original_type = _cell(secondary_row, original_type_index)
            repayment_date = _date(raw_repay_date)
            repayment_amount = parse_number(raw_repay_amount)
            pending.update(
                {
                    "five_tier_class": (None if raw_class in _MISSING_MARKERS else raw_class),
                    "five_tier_class_status": _source_state(raw_class, raw_class),
                    "last_repayment_date": repayment_date,
                    "last_repayment_date_status": _source_state(raw_repay_date, repayment_date),
                    "last_repayment_amount": repayment_amount,
                    "last_repayment_amount_status": _source_state(raw_repay_amount, repayment_amount),
                    "repayment_method": (None if raw_repayment in _MISSING_MARKERS else raw_repayment),
                    "repayment_method_status": _source_state(raw_repayment, raw_repayment),
                    "history_status": (None if raw_history in _MISSING_MARKERS else raw_history),
                    "history_status_source_state": _source_state(raw_history, raw_history),
                    "original_creditor_name": (
                        None if raw_original_creditor in _MISSING_MARKERS else raw_original_creditor
                    ),
                    "original_creditor_name_status": _source_state(raw_original_creditor, raw_original_creditor),
                    "original_debt_type": (None if raw_original_type in _MISSING_MARKERS else raw_original_type),
                    "original_debt_type_status": _source_state(raw_original_type, raw_original_type),
                }
            )
            _append_ref(pending, item.source_ref(table_id=table_id, row=row_index))
            finish_pending()

    finish_pending()
    return records


def _joined_account_note_text(
    document: CanonicalEnterpriseDocumentIR,
) -> tuple[str, tuple[_TextSpan, ...]]:
    chunks: list[str] = []
    spans: list[_TextSpan] = []
    offset = 0
    for item in _flow_items(document):
        if item.kind != "text":
            continue
        chunk = compact_text(item.value)
        if not chunk:
            continue
        chunks.append(chunk)
        spans.append(_TextSpan(offset, offset + len(chunk), item.source_ref()))
        offset += len(chunk)
    return "".join(chunks), tuple(spans)


def _refs_for_interval(
    spans: tuple[_TextSpan, ...],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for span in spans:
        if span.end <= start or span.start >= end:
            continue
        if span.ref not in refs:
            refs.append(dict(span.ref))
    return refs


def _account_note_end(text: str, start: int, next_anchor: int) -> int:
    end = min(len(text), next_anchor)
    for marker in _ACCOUNT_NOTE_TERMINATORS:
        marker_index = text.find(marker, start, end)
        if marker_index >= 0:
            end = marker_index
    return end


def _clean_annotation_content(value: Any) -> str:
    return compact_text(value).strip("；")


def extract_account_annotations(
    document: CanonicalEnterpriseDocumentIR,
) -> list[dict[str, Any]]:
    """Extract account-linked provider/subject statements and dispute notes."""

    _require_document(document)
    text, spans = _joined_account_note_text(document)
    anchors = tuple(_ACCOUNT_NOTE_ANCHOR_RE.finditer(text))
    records: list[dict[str, Any]] = []

    def append_annotation(
        *,
        account_identifier: str,
        annotation_type: str,
        issuer: str,
        annotation_date: str | None,
        content: str,
        start: int,
        end: int,
        dispute_status: str | None = None,
        date_not_applicable: bool = False,
    ) -> None:
        normalized_content = _clean_annotation_content(content)
        if not normalized_content:
            return
        record: dict[str, Any] = {
            "account_annotation_id": stable_record_id(
                "enterprise_account_annotation",
                account_identifier,
                annotation_type,
                annotation_date,
                normalized_content,
            ),
            "sequence": len(records) + 1,
            "account_id": f"credit_account:{account_identifier}",
            "account_identifier": account_identifier,
            "annotation_type": annotation_type,
            "annotation_type_label": _ACCOUNT_ANNOTATION_LABELS[annotation_type],
            "issuer": issuer or None,
            "annotation_date": annotation_date,
            "annotation_date_status": (
                "reported" if annotation_date else "not_applicable" if date_not_applicable else "not_reported"
            ),
            "annotation_content": normalized_content,
            "source": "canonical_account_bound_annotation",
            "confidence": 1.0,
            "source_refs": _refs_for_interval(spans, start, end),
        }
        if dispute_status:
            record["dispute_status"] = dispute_status
        _source_bounds(record)
        records.append(record)

    for index, anchor in enumerate(anchors):
        identifier = _identifier(anchor.group("identifier"))
        if not identifier:
            continue
        next_anchor = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        end = _account_note_end(text, anchor.end(), next_anchor)
        note = text[anchor.end() : end]

        provider_pattern = re.compile(
            rf"(?:[”\"']?的业务[,，])?"
            rf"(?P<issuer>[^，；。]+?)于(?P<date>{_DATE_PATTERN})[做作]出说明[:：]"
            rf"(?P<content>.*?)(?=；(?:信息主体|征信中心|数据提供机构|该业务|该账户)|$)"
        )
        for match in provider_pattern.finditer(note):
            parsed_date = _date(match.group("date"))
            append_annotation(
                account_identifier=identifier,
                annotation_type="data_provider_statement",
                issuer=_clean_annotation_content(match.group("issuer")).lstrip("，的业务"),
                annotation_date=parsed_date,
                content=match.group("content"),
                start=anchor.start(),
                end=end,
            )

        subject_pattern = re.compile(
            rf"信息主体于(?P<date>{_DATE_PATTERN})提出声明[:：]"
            rf"(?P<content>.*?)(?=；(?:征信中心|数据提供机构|该业务|该账户)|$)"
        )
        for match in subject_pattern.finditer(note):
            append_annotation(
                account_identifier=identifier,
                annotation_type="subject_statement",
                issuer="信息主体",
                annotation_date=_date(match.group("date")),
                content=match.group("content"),
                start=anchor.start(),
                end=end,
            )

        bureau_pattern = re.compile(
            rf"征信中心(?:于(?P<date>{_DATE_PATTERN}))?[做作]出说明[:：]"
            rf"(?P<content>.*?)(?=；(?:信息主体|数据提供机构|该业务|该账户)|$)"
        )
        for match in bureau_pattern.finditer(note):
            append_annotation(
                account_identifier=identifier,
                annotation_type="credit_bureau_statement",
                issuer="征信中心",
                annotation_date=_date(match.group("date")) if match.group("date") else None,
                content=match.group("content"),
                start=anchor.start(),
                end=end,
            )

        dispute_pattern = re.compile(r"(?P<content>该(?:业务|账户)(?:处于异议处理期|正在异议处理中|存在异议))")
        for match in dispute_pattern.finditer(note):
            append_annotation(
                account_identifier=identifier,
                annotation_type="dispute_processing",
                issuer="信息主体",
                annotation_date=None,
                content=match.group("content"),
                start=anchor.start(),
                end=end,
                dispute_status="in_progress",
                date_not_applicable=True,
            )

    return records


def _is_grouped_responsibility_header(row: tuple[str, ...]) -> bool:
    signature = "".join(row)
    return (
        "账户编号" not in signature
        and all(
            marker in signature
            for marker in (
                "责任类型",
                "保证合同编号",
                "还款责任金额",
                "授信机构",
                "业务种类",
                "五级分类",
                "账户数",
                "余额",
            )
        )
        and ("借款金额" in signature or "担保金额" in signature)
    )


def _responsibility_group(text: str, headers: tuple[str, ...] | None = None) -> str:
    compact = compact_text(text)
    header_text = "".join(headers or ())
    if "担保交易" in compact or "担保金额" in header_text:
        return "guarantee"
    if "贴现" in compact:
        return "discount"
    return "borrowing"


def _heading_account_count(text: str) -> int | None:
    match = re.search(r"共(\d+)笔", compact_text(text))
    return int(match.group(1)) if match else None


def _join_continuation_text(left: str, right: str) -> str:
    if not right:
        return left
    if not left:
        return right
    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + right


def extract_grouped_repayment_responsibility_details(
    document: CanonicalEnterpriseDocumentIR,
) -> list[dict[str, Any]]:
    """Decode grouped borrowing, discount, and guarantee responsibility rows."""

    _require_document(document)
    records: list[dict[str, Any]] = []
    current_heading = ""
    expected_account_count: int | None = None
    headers: tuple[str, ...] | None = None
    group = ""
    pending: dict[str, Any] | None = None
    pending_raw: dict[str, str] | None = None

    def finish_pending() -> None:
        nonlocal pending, pending_raw
        if pending is None:
            return
        pending["sequence"] = len(records) + 1
        _source_bounds(pending)
        records.append(pending)
        pending = None
        pending_raw = None

    def reset_group() -> None:
        nonlocal headers, group, expected_account_count
        finish_pending()
        headers = None
        group = ""
        expected_account_count = None

    for item in _flow_items(document):
        if item.kind == "text":
            text = compact_text(item.value)
            if any(marker in text for marker in ("非信贷记录明细", "公共记录明细", "声明及异议标注信息", "附件")):
                reset_group()
                current_heading = ""
                continue
            if (
                "共" in text
                and "笔" in text
                and any(marker in text for marker in ("贴现", "担保交易", "其他借贷交易", "其他业务"))
            ):
                finish_pending()
                headers = None
                current_heading = text
                expected_account_count = _heading_account_count(text)
            continue

        table_id, raw_rows = item.value
        recognized_table = False
        for row_index, raw_row in enumerate(raw_rows):
            row = tuple(compact_text(value) for value in raw_row)
            if _is_grouped_responsibility_header(row):
                finish_pending()
                headers = row
                group = _responsibility_group(current_heading, headers)
                recognized_table = True
                continue
            if headers is None:
                continue
            aligned = _align_row(row, headers)
            responsibility_index = _column_index(headers, "责任类型")
            contract_index = _column_index(headers, "保证合同编号")
            responsibility_amount_index = _column_index(headers, "还款责任金额")
            institution_index = _column_index(headers, "授信机构")
            business_index = _column_index(headers, "业务种类", "业务类型")
            class_index = _column_index(headers, "五级分类")
            account_count_index = _column_index(headers, "账户数")
            loan_amount_index = _column_index(headers, "借款金额")
            guarantee_amount_index = _column_index(headers, "担保金额")
            balance_index = _column_index(headers, "余额")
            overdue_total_index = _column_index(headers, "逾期总额")
            overdue_principal_index = _column_index(headers, "逾期本金")
            raw_values = {
                "responsibility_type": _cell(aligned, responsibility_index),
                "contract_number": _cell(aligned, contract_index),
                "responsibility_amount": _cell(aligned, responsibility_amount_index),
                "institution": _cell(aligned, institution_index),
                "business_type": _cell(aligned, business_index),
                "five_tier_class": _cell(aligned, class_index),
                "account_count": _cell(aligned, account_count_index),
                "loan_amount": _cell(aligned, loan_amount_index),
                "guarantee_amount": _cell(aligned, guarantee_amount_index),
                "balance": _cell(aligned, balance_index),
                "overdue_total": _cell(aligned, overdue_total_index),
                "overdue_principal": _cell(aligned, overdue_principal_index),
            }
            account_count = parse_number(raw_values["account_count"])
            nonempty_numeric = any(
                compact_text(raw_values[key]) not in _MISSING_MARKERS
                for key in (
                    "responsibility_amount",
                    "account_count",
                    "loan_amount",
                    "guarantee_amount",
                    "balance",
                    "overdue_total",
                    "overdue_principal",
                )
            )
            is_continuation = bool(
                pending is not None
                and raw_values["responsibility_type"]
                and account_count is None
                and not nonempty_numeric
            )
            if is_continuation and pending is not None and pending_raw is not None:
                for key in ("responsibility_type", "institution", "business_type"):
                    piece = raw_values[key]
                    if not piece:
                        continue
                    pending_raw[key] = _join_continuation_text(pending_raw.get(key, ""), piece)
                    pending[key] = pending_raw[key]
                _append_ref(pending, item.source_ref(table_id=table_id, row=row_index))
                recognized_table = True
                continue

            if not raw_values["responsibility_type"] or account_count is None:
                continue
            finish_pending()
            contract_number = opaque_identifier(raw_values["contract_number"])
            responsibility_amount = parse_number(raw_values["responsibility_amount"])
            loan_amount = parse_number(raw_values["loan_amount"])
            guarantee_amount = parse_number(raw_values["guarantee_amount"])
            pending_raw = raw_values
            pending = {
                "responsibility_group_detail_id": stable_record_id(
                    "enterprise_repayment_responsibility_group_detail",
                    group,
                    raw_values["responsibility_type"],
                    contract_number,
                    raw_values["institution"],
                    raw_values["business_type"],
                    len(records) + 1,
                ),
                "transaction_group": group,
                "responsibility_type": raw_values["responsibility_type"],
                "contract_number": contract_number or None,
                "contract_number_status": _source_state(raw_values["contract_number"], contract_number),
                "responsibility_amount": responsibility_amount,
                "responsibility_amount_status": _source_state(
                    raw_values["responsibility_amount"], responsibility_amount
                ),
                "institution": raw_values["institution"],
                "business_type": raw_values["business_type"],
                "five_tier_class": raw_values["five_tier_class"],
                "account_count": int(account_count),
                "loan_amount": loan_amount,
                "loan_amount_status": _column_source_state(loan_amount_index, raw_values["loan_amount"], loan_amount),
                "guarantee_amount": guarantee_amount,
                "guarantee_amount_status": _column_source_state(
                    guarantee_amount_index,
                    raw_values["guarantee_amount"],
                    guarantee_amount,
                ),
                "amount_kind": ("guarantee_amount" if guarantee_amount_index is not None else "loan_amount"),
                "balance": parse_number(raw_values["balance"]),
                "balance_status": _column_source_state(
                    balance_index,
                    raw_values["balance"],
                    parse_number(raw_values["balance"]),
                ),
                "overdue_total": parse_number(raw_values["overdue_total"]),
                "overdue_total_status": _column_source_state(
                    overdue_total_index,
                    raw_values["overdue_total"],
                    parse_number(raw_values["overdue_total"]),
                ),
                "overdue_principal": parse_number(raw_values["overdue_principal"]),
                "overdue_principal_status": _column_source_state(
                    overdue_principal_index,
                    raw_values["overdue_principal"],
                    parse_number(raw_values["overdue_principal"]),
                ),
                "source_group_account_count": expected_account_count,
                "currency": "CNY",
                "amount_unit": "CNY_10K",
                "source": "canonical_grouped_repayment_responsibility_detail",
                "confidence": 1.0,
                "source_refs": [item.source_ref(table_id=table_id, row=row_index)],
            }
            recognized_table = True

        if headers is not None and not recognized_table:
            reset_group()

    finish_pending()
    return records


def extract_credit_detail_groups(
    document: CanonicalEnterpriseDocumentIR,
) -> list[dict[str, Any]]:
    """Retain every printed ``共 N 笔`` heading at its canonical grain.

    These source-declared counts are business data as well as reconstruction
    contracts.  A heading may occur as native text, inside a physical table,
    or in both representations; the canonical phase/category key emits it
    once while preserving every source reference.
    """

    _require_document(document)
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    phase = "unspecified"

    def consume(value: Any, item: _FlowItem, *, table_id: str = "", row: int | None = None) -> None:
        nonlocal phase
        signature = compact_text(value)
        if not signature:
            return
        for match in _CREDIT_DETAIL_GROUP_TOKEN_RE.finditer(signature):
            phase_heading = match.group("phase")
            if phase_heading:
                phase = {
                    "未结清信贷": "active",
                    "已结清信贷": "settled",
                    "相关还款责任": "repayment_responsibility",
                }[phase_heading]
                continue
            label = str(match.group("label") or "")
            count = int(match.group("count"))
            record_phase = phase
            if label == "被追偿业务":
                record_phase = "recovered"
            elif label == "欠息" and record_phase == "unspecified":
                record_phase = "active"
            elif label in {
                "除贴现外的其他业务",
                "为担保交易承担的相关还款责任",
            }:
                record_phase = "repayment_responsibility"
                phase = record_phase

            group_kind, represented_dataset = _CREDIT_DETAIL_GROUP_TARGETS[label]
            if record_phase == "repayment_responsibility" and label == "贴现":
                group_kind = "grouped_repayment_responsibility"
                represented_dataset = GROUPED_RESPONSIBILITY_DATASET
            key = (record_phase, label)
            source_ref = item.source_ref(table_id=table_id, row=row)
            existing = by_key.get(key)
            if existing is not None:
                _append_ref(existing, source_ref)
                # Conflicting duplicates are retained as an explicit state;
                # no source assertion is silently overwritten.
                if existing["reported_record_count"] != count:
                    existing["reported_record_count_status"] = "conflict"
                    existing_conflicts = existing.get("reported_record_count_conflicts") or ()
                    if not isinstance(existing_conflicts, (list, tuple, set)):
                        existing_conflicts = (existing_conflicts,)
                    conflict_values = {
                        int(value)
                        for value in existing_conflicts
                        if str(value).isdigit()
                    }
                    conflict_values.update(
                        {int(existing["reported_record_count"]), count}
                    )
                    existing["reported_record_count_conflicts"] = sorted(conflict_values)
                continue
            record = {
                "credit_detail_group_id": stable_record_id(
                    "enterprise_credit_detail_group",
                    record_phase,
                    label,
                ),
                "group_phase": record_phase,
                "business_category": label,
                "group_kind": group_kind,
                "represented_dataset": represented_dataset,
                "reported_record_count": count,
                "reported_record_count_status": "reported",
                "source": "canonical_credit_detail_group_heading",
                "source_refs": [source_ref],
                "confidence": 1.0,
            }
            by_key[key] = record
            records.append(record)

    for item in _flow_items(document):
        if item.kind == "text":
            consume(item.value, item)
            continue
        if item.kind != "table":
            continue
        table_id, rows = item.value
        for row_index, row in enumerate(rows):
            consume("".join(row), item, table_id=table_id, row=row_index)

    for sequence, record in enumerate(records, start=1):
        record["sequence"] = sequence
        _source_bounds(record)
    return records


def extract_canonical_enterprise_record_families(
    document: CanonicalEnterpriseDocumentIR,
) -> dict[str, list[dict[str, Any]]]:
    """Return the canonical business families owned by this decoder."""

    _require_document(document)
    return {
        RECOVERED_BUSINESS_DATASET: extract_recovered_business_accounts(document),
        ACCOUNT_ANNOTATION_DATASET: extract_account_annotations(document),
        GROUPED_RESPONSIBILITY_DATASET: (extract_grouped_repayment_responsibility_details(document)),
        CREDIT_DETAIL_GROUP_DATASET: extract_credit_detail_groups(document),
    }


__all__ = [
    "ACCOUNT_ANNOTATION_DATASET",
    "CREDIT_DETAIL_GROUP_DATASET",
    "GROUPED_RESPONSIBILITY_DATASET",
    "RECOVERED_BUSINESS_DATASET",
    "extract_account_annotations",
    "extract_canonical_enterprise_record_families",
    "extract_credit_detail_groups",
    "extract_grouped_repayment_responsibility_details",
    "extract_recovered_business_accounts",
]
