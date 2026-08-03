# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical-table refinements for native enterprise credit reports.

Enterprise account cards use stacked header/detail/repayment rows and may
continue across a page boundary.  This module interprets those already-sealed
physical tables without changing or supplementing ``ParseResult``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

from docmirror.plugins.credit_report.currency_codes import (
    CURRENCY_CODE_BY_ALIAS,
    normalize_currency_code,
)
from docmirror.plugins.credit_report.enterprise_native.continuation import (
    ACCOUNT_SETTLED_DETAIL_CONTRACT,
    ATTACHMENT_HISTORY_BODY_CONTRACT,
    CLOSED_SUMMARY_BODY_CONTRACT,
    FACILITY_VALUE_CONTRACT,
    EnterpriseContinuationResolver,
    TableFragment,
)
from docmirror.plugins.credit_report.shared.entity_decoder import (
    CreditReportEntityContext,
    decode_credit_report_entities,
)
from docmirror.plugins.credit_report.value_utils import (
    compact_text as _compact,
)
from docmirror.plugins.credit_report.value_utils import (
    parse_number as _number,
)
from docmirror.plugins.credit_report.value_utils import (
    stable_record_id as _stable_id,
)

_ACCOUNT_CATEGORIES = frozenset({"中长期借款", "短期借款", "循环透支", "贴现"})
_FIVE_TIER_CLASSES = frozenset({"正常", "关注", "次级", "可疑", "损失", "违约", "未分类"})
_MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH = 6
_CURRENCY_CODES = {
    "人民币": "CNY",
    "人民币元": "CNY",
    "美元": "USD",
    "欧元": "EUR",
    "港币": "HKD",
}
_MISSING_MARKERS = frozenset({"", "--", "-", "—"})


def _date(value: Any) -> str:
    raw = _compact(value)
    match = re.fullmatch(r"((?:18|19|20|21)\d{2})[-年./](\d{1,2})[-月./](\d{1,2})日?", raw)
    if not match:
        return ""
    year, month, day = (int(match.group(index)) for index in range(1, 4))
    try:
        parsed = date(year, month, day)
    except ValueError:
        return ""
    return parsed.isoformat()


def _percentage(value: Any) -> float | int | None:
    raw = _compact(value)
    number = _number(raw)
    if number is None:
        return None
    return float(number) / 100 if raw.endswith("%") else number


def _identifier(value: Any) -> str:
    return re.sub(r"[^0-9A-Z]", "", _compact(value).upper())


def _currency_code(value: Any) -> str:
    """Normalize known currencies while preserving future source labels."""
    raw = _compact(value)
    return normalize_currency_code(raw) or _CURRENCY_CODES.get(raw, raw)


def _amount_unit_for_currency(currency: str, *, scale: str = "10K") -> str:
    """Keep amount scale independent from the source currency."""
    normalized = _currency_code(currency)
    return f"{normalized}_{scale}" if normalized else scale


def _looks_like_currency(value: Any) -> bool:
    raw = _compact(value)
    upper = raw.upper()
    return bool(
        raw
        and (
            raw in _CURRENCY_CODES
            or upper in CURRENCY_CODE_BY_ALIAS
            or re.fullmatch(r"[A-Z]{3}", upper)
            or raw.endswith(("元", "币"))
        )
    )


def _account_category_heading(row: list[str]) -> tuple[str, str] | None:
    """Recognize an account-card heading without a closed category allowlist."""
    values = [_compact(value) for value in row]
    signature = "".join(values)
    count_match = re.search(r"共\s*\d+\s*笔", signature)
    if not count_match:
        return None
    category = next(
        (
            value
            for value in values
            if value
            and not re.fullmatch(r"共\s*\d+\s*笔", value)
            and not any(marker in value for marker in ("账户编号", "授信机构", "业务种类", "业务类型"))
        ),
        "",
    )
    if not category:
        return None
    status = "settled" if "已结清" in signature else "active" if "未结清" in signature else ""
    return category, status


def _raw_table_rows(table: Any) -> list[list[str]]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw_rows = metadata.get("raw_rows")
    if isinstance(raw_rows, list) and raw_rows:
        return [[_compact(value) for value in row] for row in raw_rows if isinstance(row, list)]
    rows: list[list[str]] = []
    headers = list(getattr(table, "headers", None) or [])
    if headers:
        rows.append([_compact(value) for value in headers])
    for row in getattr(table, "rows", None) or []:
        rows.append([_compact(getattr(cell, "text", cell)) for cell in (getattr(row, "cells", None) or [])])
    return rows


def _table_stream(parse_result: Any):
    if isinstance(parse_result, EnterpriseExtractionContext):
        for page, table_id, rows in parse_result.table_rows:
            yield page, table_id, [list(row) for row in rows]
        return
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", None) or []:
            table_rows = _raw_table_rows(table)
            if table_rows:
                yield page_number, str(getattr(table, "table_id", "") or ""), table_rows


def _table_headings(parse_result: Any) -> dict[str, str]:
    """Return the closest preceding page heading for each physical table."""
    if isinstance(parse_result, EnterpriseExtractionContext):
        return dict(parse_result.table_headings)
    headings: dict[str, str] = {}
    for page in getattr(parse_result, "pages", None) or []:
        text_blocks: list[tuple[float, str]] = []
        for block in getattr(page, "texts", None) or []:
            bbox = list(getattr(block, "bbox", None) or [])
            content = _compact(getattr(block, "content", ""))
            if len(bbox) >= 4 and content:
                text_blocks.append((float(bbox[3]), content))
        for table in getattr(page, "tables", None) or []:
            table_id = str(getattr(table, "table_id", "") or "")
            bbox = list(getattr(table, "bbox", None) or [])
            if not table_id or len(bbox) < 2:
                continue
            table_top = float(bbox[1])
            preceding = [(bottom, content) for bottom, content in text_blocks if bottom <= table_top]
            if preceding:
                headings[table_id] = max(preceding, key=lambda item: item[0])[1]
    return headings


def _page_texts(parse_result: Any) -> dict[int, str]:
    """Return compact source text by page for enterprise metadata recovery."""
    if isinstance(parse_result, EnterpriseExtractionContext):
        return dict(parse_result.page_texts)
    values: dict[int, str] = {}
    for index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or index)
        values[page_number] = "\n".join(
            str(getattr(block, "content", "") or "")
            for block in getattr(page, "texts", None) or []
            if getattr(block, "content", None)
        )
    return values


@dataclass(frozen=True)
class EnterpriseExtractionContext:
    """Immutable indexes reused by every enterprise extractor in one projection."""

    parse_result: Any
    table_rows: tuple[tuple[int, str, tuple[tuple[str, ...], ...]], ...]
    page_texts: Mapping[int, str]
    table_headings: Mapping[str, str]
    page_flow: tuple[tuple[int, str, Any], ...]
    continuation_fragments: tuple[TableFragment, ...]
    entity_context: CreditReportEntityContext

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parse_result, name)


def build_enterprise_extraction_context(parse_result: Any) -> EnterpriseExtractionContext:
    """Index immutable page/table views once without altering the sealed result."""
    if isinstance(parse_result, EnterpriseExtractionContext):
        return parse_result

    entity_context = decode_credit_report_entities(
        parse_result,
        report_family="enterprise",
    )
    table_rows: list[tuple[int, str, tuple[tuple[str, ...], ...]]] = []
    page_texts: dict[int, str] = {}
    table_headings: dict[str, str] = {}
    fragments: list[TableFragment] = []

    def normalized_bbox(value: Any) -> tuple[float, float, float, float] | None:
        raw_bbox = list(getattr(value, "bbox", None) or [])
        if len(raw_bbox) < 4:
            return None
        try:
            x0, y0, x1, y1 = (float(item) for item in raw_bbox[:4])
        except (TypeError, ValueError):
            return None
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or page_index)
        blocks = list(getattr(page, "texts", None) or [])
        page_texts[page_number] = "\n".join(
            str(getattr(block, "content", "") or "") for block in blocks if getattr(block, "content", None)
        )
        positioned_text = [
            (
                float((list(getattr(block, "bbox", None) or [0, 0, 0, 0]))[3]),
                _compact(getattr(block, "content", "")),
            )
            for block in blocks
            if len(list(getattr(block, "bbox", None) or [])) >= 4 and _compact(getattr(block, "content", ""))
        ]
        for table_index, table in enumerate(getattr(page, "tables", None) or []):
            rows = _raw_table_rows(table)
            if not rows:
                continue
            table_id = str(getattr(table, "table_id", "") or "")
            immutable_rows = tuple(tuple(value for value in row) for row in rows)
            table_rows.append((page_number, table_id, immutable_rows))
            fragments.append(
                TableFragment(
                    index=len(fragments),
                    page=page_number,
                    table_id=table_id,
                    rows=immutable_rows,
                    bbox=normalized_bbox(table),
                    page_width=float(getattr(page, "width", 0) or 0),
                    page_height=float(getattr(page, "height", 0) or 0),
                    first_on_page=table_index == 0,
                    last_on_page=table_index == len(list(getattr(page, "tables", None) or [])) - 1,
                )
            )
            bbox = list(getattr(table, "bbox", None) or [])
            if table_id and len(bbox) >= 2:
                preceding = [(bottom, content) for bottom, content in positioned_text if bottom <= float(bbox[1])]
                if preceding:
                    table_headings[table_id] = max(preceding, key=lambda item: item[0])[1]
    return EnterpriseExtractionContext(
        parse_result=parse_result,
        table_rows=tuple(table_rows),
        page_texts=MappingProxyType(page_texts),
        table_headings=MappingProxyType(table_headings),
        page_flow=entity_context.ordered_page_flow(),
        continuation_fragments=tuple(fragments),
        entity_context=entity_context,
    )


def _table_source_metadata(rows: list[list[str]]) -> dict[str, str]:
    """Parse source institution and update date from a table footer."""
    for row in rows:
        text = "".join(_compact(value) for value in row)
        if "信息来源机构" not in text:
            continue
        source_match = re.search(
            r"信息来源机构[:：]?(.*?)(?=更新日期[:：]?|$)",
            text,
        )
        date_match = re.search(
            r"更新日期[:：]?((?:19|20)\d{2}[-年./]\d{1,2}[-月./]\d{1,2})日?",
            text,
        )
        metadata: dict[str, str] = {}
        if source_match and _compact(source_match.group(1)):
            metadata["source_institution"] = _compact(source_match.group(1))
        if date_match:
            metadata["update_date"] = _date(date_match.group(1))
        return metadata
    return {}


def _source_ref(page: int, table_id: str, row: int) -> dict[str, Any]:
    return {
        "source": "canonical_physical_table",
        "page": page,
        "table_id": table_id,
        "row": row,
    }


def _append_ref(record: dict[str, Any], page: int, table_id: str, row: int) -> None:
    ref = _source_ref(page, table_id, row)
    refs = record.setdefault("source_refs", [])
    if ref not in refs:
        refs.append(ref)


def _is_account_header(row: list[str]) -> bool:
    return bool(
        row
        and "账户编号" in row[0]
        and any("授信机构" in cell for cell in row)
        and any("业务种类" in cell or "业务类型" in cell for cell in row)
        and any("开立日期" in cell or "开户日期" in cell for cell in row)
        and any("借款金额" in cell or "信用额度" in cell for cell in row)
    )


def _is_primary_account_row(
    row: list[str],
    *,
    open_date_index: int,
    due_date_index: int,
    amount_index: int,
) -> bool:
    return bool(
        len(row) > max(open_date_index, due_date_index, amount_index)
        and len(_identifier(row[0])) >= _MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH
        and _date(row[open_date_index])
        and _date(row[due_date_index])
        and _number(row[amount_index]) is not None
    )


def _primary_account_field_indexes(
    row: list[str],
    *,
    institution_index: int,
    business_type_index: int,
    open_date_index: int,
    due_date_index: int,
    currency_index: int,
    amount_index: int,
) -> dict[str, int] | None:
    """Resolve one primary account row even when a page break shifts columns.

    Some settled-account cards end a page after the primary header and resume
    with a secondary header whose spacer columns alter the physical row width.
    Prefer the declared header positions, then recover the date/currency/amount
    positions semantically without weakening the account-card guards.
    """

    preferred = {
        "institution": institution_index,
        "business_type": business_type_index,
        "open_date": open_date_index,
        "due_date": due_date_index,
        "currency": currency_index,
        "amount": amount_index,
    }
    if _is_primary_account_row(
        row,
        open_date_index=open_date_index,
        due_date_index=due_date_index,
        amount_index=amount_index,
    ):
        return preferred

    if len(_identifier(row[0] if row else "")) < _MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH:
        return None
    date_indexes = [index for index, value in enumerate(row) if _date(value)]
    if len(date_indexes) < 2:
        return None
    detected_currency_index = next(
        (index for index, value in enumerate(row) if _looks_like_currency(value)),
        -1,
    )
    if detected_currency_index < 0:
        return None
    detected_amount_index = next(
        (index for index in range(detected_currency_index + 1, len(row)) if _number(row[index]) is not None),
        -1,
    )
    if detected_amount_index < 0:
        return None
    return {
        "institution": institution_index,
        "business_type": business_type_index,
        "open_date": date_indexes[0],
        "due_date": date_indexes[1],
        "currency": detected_currency_index,
        "amount": detected_amount_index,
    }


def _is_account_detail_row(row: list[str]) -> bool:
    classification = _compact(row[3]) if len(row) > 3 else ""
    return bool(
        len(row) >= 4
        and _number(row[2]) is not None
        and classification not in _MISSING_MARKERS
        and not any(marker in classification for marker in ("五级分类", "账户编号", "还款"))
    )


def _is_account_continuation_row(row: list[str]) -> bool:
    values = [_compact(value) for value in row]
    signature = "".join(values)
    if "关闭日期" in signature and "五级分类" in signature:
        return True
    if _is_account_detail_row(values):
        return True
    date_indexes = [index for index, value in enumerate(values) if _date(value)]
    if len(date_indexes) >= 2:
        first_date, second_date = date_indexes[:2]
        classification = next(
            (value for value in values[first_date + 1 : second_date] if value not in _MISSING_MARKERS),
            "",
        )
        if classification and not any(marker in classification for marker in ("五级分类", "关闭日期")):
            return True
    if len(_identifier(values[0] if values else "")) >= _MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH:
        return bool(
            sum(bool(_date(value)) for value in values) >= 2
            and any(_looks_like_currency(value) for value in values)
            and any(_number(value) is not None for value in values[1:])
        )
    return bool(
        any(_date(value) for value in values)
        and any("还款" in value for value in values)
    )


def _finalize_account(record: dict[str, Any] | None, out: list[dict[str, Any]]) -> None:
    if not record:
        return
    identifier = _identifier(record.get("account_identifier"))
    if len(identifier) < _MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH:
        return
    record["account_identifier"] = identifier
    record["account_id"] = f"credit_account:{identifier}"
    record["sequence"] = len(out) + 1
    out.append(record)


def _append_account_identifier_suffix(
    record: dict[str, Any],
    value: Any,
    *,
    continuation_authorized: bool,
) -> None:
    """Append a guarded row-level suffix to an account identifier."""
    suffix = _identifier(value)
    identifier = _identifier(record.get("account_identifier"))
    if (
        not identifier
        or not suffix
        or identifier.endswith(suffix)
        or len(identifier) + len(suffix) > 64
        or (not continuation_authorized and len(suffix) > 8)
    ):
        return
    record["account_identifier"] = f"{identifier}{suffix}"


def extract_enterprise_accounts_from_tables(parse_result: Any) -> list[dict[str, Any]]:
    """Read current and settled enterprise account cards.

    Account-history and appendix tables deliberately do not qualify because
    their headers lack the source account-card amount labels.
    """
    accounts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    category = ""
    amount_field = "loan_amount"
    header_seen = False
    settled_schema = False
    institution_index = 1
    business_type_index = 2
    open_date_index = 3
    due_date_index = 4
    currency_index = 5
    amount_index = 6
    issuance_form_index = -1
    header_page = 0
    header_table_id = ""
    header_row_index = 0
    resolver = EnterpriseContinuationResolver(parse_result)
    fragment_by_table_id = {fragment.table_id: fragment for fragment in resolver.fragments}
    active_fragment_index = -1
    active_fragment_page = 0
    account_status_hint = ""
    settled_continuation_rows: set[tuple[str, int]] = set()
    for fragment in resolver.fragments:
        match = resolver.following_row(
            fragment,
            ACCOUNT_SETTLED_DETAIL_CONTRACT,
        )
        if match is not None:
            settled_continuation_rows.add((match.fragment.table_id, match.row_index))

    for page, table_id, rows in _table_stream(parse_result):
        fragment = fragment_by_table_id.get(table_id)
        fragment_index = fragment.index if fragment is not None else -1

        def follows_active_fragment() -> bool:
            if active_fragment_index < 0 or fragment_index < 0:
                return True
            if fragment_index == active_fragment_index:
                return True
            if fragment_index != active_fragment_index + 1 or not 0 <= page - active_fragment_page <= 1:
                return False
            return resolver.table_continues(
                resolver.fragments[active_fragment_index],
                fragment,
                candidate_validator=_is_account_continuation_row,
                candidate_validator_scope="fragment",
                context="enterprise_account_card",
            )

        for row_index, row in enumerate(rows):
            cells = [*row, *([""] * max(0, 8 - len(row)))]
            first = cells[0]
            category_heading = _account_category_heading(cells)
            if category_heading is not None:
                _finalize_account(current, accounts)
                current = None
                category, heading_status = category_heading
                header_seen = False
                settled_schema = False
                account_status_hint = heading_status
                active_fragment_index = fragment_index
                active_fragment_page = page
                continue
            if _is_account_header(cells):
                _finalize_account(current, accounts)
                current = None
                header_seen = True
                header_page = page
                header_table_id = table_id
                header_row_index = row_index
                amount_field = "credit_limit" if any("信用额度" in cell for cell in cells) else "loan_amount"
                institution_index = next(
                    (index for index, value in enumerate(cells) if "授信机构" in value),
                    1,
                )
                business_type_index = next(
                    (index for index, value in enumerate(cells) if "业务种类" in value or "业务类型" in value),
                    2,
                )
                open_date_index = next(
                    (index for index, value in enumerate(cells) if "开立日期" in value or "开户日期" in value),
                    3,
                )
                due_date_index = next(
                    (index for index, value in enumerate(cells) if "到期日" in value),
                    4,
                )
                currency_index = next(
                    (index for index, value in enumerate(cells) if value == "币种"),
                    5,
                )
                amount_index = next(
                    (index for index, value in enumerate(cells) if "借款金额" in value or "信用额度" in value),
                    6,
                )
                issuance_form_index = next(
                    (index for index, value in enumerate(cells) if "发放形式" in value),
                    -1,
                )
                settled_schema = any(
                    "关闭日期" in cell for following in rows[row_index + 1 : row_index + 3] for cell in following
                )
                if account_status_hint == "settled":
                    settled_schema = True
                active_fragment_index = fragment_index
                active_fragment_page = page
                continue
            if (
                header_seen
                and follows_active_fragment()
                and any("关闭日期" in cell for cell in cells)
                and any("五级分类" in cell for cell in cells)
            ):
                settled_schema = True
                account_status_hint = "settled"
                active_fragment_index = fragment_index
                active_fragment_page = page
                continue
            primary_indexes = (
                _primary_account_field_indexes(
                    cells,
                    institution_index=institution_index,
                    business_type_index=business_type_index,
                    open_date_index=open_date_index,
                    due_date_index=due_date_index,
                    currency_index=currency_index,
                    amount_index=amount_index,
                )
                if header_seen and follows_active_fragment()
                else None
            )
            if primary_indexes is not None:
                resolved_institution_index = primary_indexes["institution"]
                resolved_business_type_index = primary_indexes["business_type"]
                resolved_open_date_index = primary_indexes["open_date"]
                resolved_due_date_index = primary_indexes["due_date"]
                resolved_currency_index = primary_indexes["currency"]
                resolved_amount_index = primary_indexes["amount"]
                _finalize_account(current, accounts)
                identifier = _identifier(first)
                source_refs = [_source_ref(page, table_id, row_index)]
                header_ref = _source_ref(header_page, header_table_id, header_row_index)
                if header_table_id and header_ref not in source_refs:
                    source_refs.insert(0, header_ref)
                current = {
                    "account_id": f"credit_account:{identifier}",
                    "account_identifier": identifier,
                    "account_type": "enterprise_credit",
                    "business_category": category,
                    "management_institution": _compact(cells[resolved_institution_index]),
                    "business_type": _compact(cells[resolved_business_type_index]),
                    "open_date": _date(cells[resolved_open_date_index]),
                    "due_date": _date(cells[resolved_due_date_index]),
                    "currency": _currency_code(cells[resolved_currency_index]),
                    amount_field: _number(cells[resolved_amount_index]),
                    "amount_unit": _amount_unit_for_currency(cells[resolved_currency_index]),
                    "account_status": account_status_hint or ("settled" if settled_schema else "active"),
                    "source": "canonical_enterprise_account_card",
                    "source_refs": source_refs,
                    "confidence": 1.0,
                }
                if 0 <= issuance_form_index < len(cells):
                    issuance_form = _compact(cells[issuance_form_index])
                    if issuance_form not in {"", "--", "-", "—"}:
                        current["issuance_form"] = issuance_form
                active_fragment_index = fragment_index
                active_fragment_page = page
                continue
            if (
                current
                and current.get("account_status") == "settled"
                and follows_active_fragment()
                and len(cells) >= 8
                and _date(cells[1])
            ):
                _append_account_identifier_suffix(
                    current,
                    first,
                    continuation_authorized=(
                        table_id,
                        row_index,
                    )
                    in settled_continuation_rows,
                )
                current["close_date"] = _date(cells[1])
                classification = _compact(cells[2])
                if classification not in _MISSING_MARKERS and "五级分类" not in classification:
                    current["five_tier_class"] = classification
                current["last_repayment_date"] = _date(cells[3])
                current["repayment_method"] = _compact(cells[5])
                current["history_status"] = _compact(cells[7])
                current["payoff_state"] = "settled"
                current["account_state"] = "closed"
                _append_ref(current, page, table_id, row_index)
                _finalize_account(current, accounts)
                current = None
                active_fragment_index = fragment_index
                active_fragment_page = page
                continue
            if current and _is_account_detail_row(cells):
                _append_account_identifier_suffix(
                    current,
                    first,
                    continuation_authorized=True,
                )
                current["guarantee_type"] = _compact(cells[1])
                current["balance"] = _number(cells[2])
                current["five_tier_class"] = _compact(cells[3])
                overdue_values = (
                    _number(cells[4]),
                    _number(cells[5]),
                    _number(cells[6]),
                )
                current["current_overdue_amount"] = overdue_values[0]
                current["overdue_principal"] = overdue_values[1]
                current["current_overdue_periods"] = overdue_values[2]
                current["last_repayment_date"] = _date(cells[7])
                reported_values = [value for value in overdue_values if value is not None]
                if len(reported_values) == len(overdue_values):
                    current["current_overdue"] = any(value != 0 for value in reported_values)
                    current["current_overdue_status"] = "overdue" if current["current_overdue"] else "not_overdue"
                elif not reported_values:
                    current.pop("current_overdue", None)
                    current["current_overdue_status"] = "not_reported"
                else:
                    current["current_overdue_status"] = "partially_reported"
                    if any(value != 0 for value in reported_values):
                        current["current_overdue"] = True
                _append_ref(current, page, table_id, row_index)
                continue
            if current and len(cells) >= 3 and _number(cells[1]) is not None and "还款" in cells[2]:
                current["actual_payment"] = _number(cells[1])
                current["repayment_method"] = cells[2]
                if category == "循环透支":
                    current["remaining_periods"] = _number(cells[3])
                    current["special_transaction"] = "" if cells[4] == "--" else cells[4]
                    current["credit_agreement_identifier"] = _identifier(cells[5])
                    current["history_status"] = "" if cells[6] == "--" else cells[6]
                else:
                    current["special_transaction"] = "" if cells[3] == "--" else cells[3]
                    current["credit_agreement_identifier"] = _identifier(cells[4])
                    current["history_status"] = "" if cells[5] == "--" else cells[5]
                report_date = next(
                    (_date(value) for value in reversed(cells) if _date(value)),
                    "",
                )
                if report_date:
                    current["snapshot_date"] = report_date
                _append_ref(current, page, table_id, row_index)

    _finalize_account(current, accounts)
    return accounts


def _facility_summary_lines(parse_result: Any) -> list[dict[str, Any]]:
    resolver = EnterpriseContinuationResolver(parse_result)
    for fragment in resolver.fragments:
        page = fragment.page
        table_id = fragment.table_id
        rows = [[_compact(value) for value in row] for row in fragment.rows]
        for row_index, row in enumerate(rows):
            if not (len(row) >= 6 and "非循环信用额度" in row[0] and "循环信用额度" in row[3]):
                continue
            if row_index + 1 >= len(rows):
                continue
            labels = rows[row_index + 1]
            if len(labels) < 6:
                continue
            if [_compact(value) for value in labels[:6]] != [
                "总额",
                "已用额度",
                "剩余可用额度",
                "总额",
                "已用额度",
                "剩余可用额度",
            ]:
                continue
            value_page = page
            value_table_id = table_id
            value_row_index = row_index + 2
            values = rows[value_row_index] if value_row_index < len(rows) else []
            numeric_values = [_number(value) for value in values if _number(value) is not None]
            if len(numeric_values) < 6:
                match = resolver.following_row(fragment, FACILITY_VALUE_CONTRACT)
                if match is not None:
                    value_page = match.fragment.page
                    value_table_id = match.fragment.table_id
                    value_row_index = match.row_index
                    values = [_compact(value) for value in match.row]
                    numeric_values = [_number(value) for value in values if _number(value) is not None]
            if len(numeric_values) < 6:
                continue
            numbers = numeric_values[:6]
            source_refs = [
                _source_ref(page, table_id, row_index),
                _source_ref(value_page, value_table_id, value_row_index),
            ]
            return [
                {
                    "credit_line_id": _stable_id(
                        "credit_line",
                        "non_revolving",
                        table_id,
                        value_table_id,
                    ),
                    "facility_type": "non_revolving",
                    "total_limit": numbers[0],
                    "used_limit": numbers[1],
                    "available_limit": numbers[2],
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                    "source": "canonical_enterprise_facility_summary",
                    "source_refs": list(source_refs),
                    "confidence": 1.0,
                },
                {
                    "credit_line_id": _stable_id(
                        "credit_line",
                        "revolving",
                        table_id,
                        value_table_id,
                    ),
                    "facility_type": "revolving",
                    "total_limit": numbers[3],
                    "used_limit": numbers[4],
                    "available_limit": numbers[5],
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                    "source": "canonical_enterprise_facility_summary",
                    "source_refs": list(source_refs),
                    "confidence": 1.0,
                },
            ]
    return []


def extract_enterprise_facility_summary(parse_result: Any) -> list[dict[str, Any]]:
    """Return one typed row for each facility-summary class."""
    rows = _facility_summary_lines(parse_result)
    for sequence, row in enumerate(rows, start=1):
        row["sequence"] = sequence
        total = row.get("total_limit")
        used = row.get("used_limit")
        if isinstance(total, (int, float)) and total:
            row["utilization_rate"] = round(float(used or 0) / float(total), 6)
    return rows


def _facility_details(parse_result: Any) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    seen_agreements: set[str] = set()
    for page, table_id, rows in _table_stream(parse_result):
        for row_index, row in enumerate(rows):
            if not row:
                continue
            if row[0].startswith("授信信息"):
                header_index = row_index + 1
            elif "授信协议编号" in row[0]:
                header_index = row_index
            else:
                continue
            if header_index + 3 >= len(rows):
                continue
            header = rows[header_index]
            subheader = rows[header_index + 1]
            if not (header and "授信协议编号" in header[0] and "授信额度" in subheader and "已用额度" in subheader):
                continue
            detail_index = header_index + 2
            while detail_index + 1 < len(rows):
                primary = rows[detail_index]
                amounts = rows[detail_index + 1]
                agreement = _identifier(primary[0] if primary else "")
                if (
                    len(agreement) < 12
                    or len(primary) < 7
                    or len(amounts) < 4
                    or _number(amounts[2]) is None
                    or _number(amounts[3]) is None
                ):
                    break
                if agreement not in seen_agreements:
                    seen_agreements.add(agreement)
                    revolving = _compact(primary[3]) in {"是", "Y", "YES", "循环"}
                    details.append(
                        {
                            "credit_line_id": f"credit_line:{agreement}",
                            "account_identifier": agreement,
                            "management_institution": _compact(primary[1]),
                            "facility_type": "revolving" if revolving else "non_revolving",
                            "facility_product": _compact(primary[2]),
                            "revolving_flag": revolving,
                            "effective_date": _date(primary[4]),
                            "due_date": _date(primary[5]),
                            "snapshot_date": _date(primary[6]),
                            "currency": _CURRENCY_CODES.get(
                                _compact(amounts[1]),
                                _compact(amounts[1]),
                            ),
                            "total_limit": _number(amounts[2]),
                            "used_limit": _number(amounts[3]),
                            "facility_limit": _number(amounts[4]) if len(amounts) > 4 else None,
                            "limit_identifier": (_identifier(amounts[5]) if len(amounts) > 5 else ""),
                            "amount_unit": "CNY_10K",
                            "source": "canonical_enterprise_facility_detail",
                            "source_refs": [
                                _source_ref(page, table_id, detail_index),
                                _source_ref(page, table_id, detail_index + 1),
                            ],
                            "confidence": 1.0,
                        }
                    )
                detail_index += 2
    return details


def _reported_credit_line_count(parse_result: Any) -> int | None:
    """Read the source-declared facility-detail count independently."""
    counts: list[int] = []
    for _page, _table_id, rows in _table_stream(parse_result):
        for row in rows:
            if not row or not row[0].startswith("授信信息"):
                continue
            match = re.search(r"共\s*(\d+)\s*笔", "".join(row))
            if match:
                counts.append(int(match.group(1)))
                break
    return sum(counts) if counts else None


def extract_enterprise_credit_lines_from_tables(
    parse_result: Any,
    accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return source-grained facility cards, never aggregate pseudo-records."""
    details = _facility_details(parse_result)
    for line in details:
        institution = _compact(line.get("management_institution"))
        candidates = [
            account
            for account in accounts
            if institution and _compact(account.get("management_institution")) == institution
        ]
        if len(candidates) == 1:
            line["account_id"] = candidates[0]["account_id"]
            line["account_status"] = candidates[0].get("account_status")
    return details


_PROFILE_LABELS = frozenset(
    {
        "经济类型",
        "组织机构类型",
        "企业规模",
        "所属行业",
        "成立年份",
        "登记证书有效截止日期",
        "登记地址",
        "办公/经营地址",
        "存续状态",
    }
)

_IDENTITY_LABELS = {
    "企业名称": "subject_name",
    "中征码": "zhongzheng_code",
    "统一社会信用代码": "unified_social_credit_code",
    "组织机构代码": "organization_code",
    "机构信用代码": "institution_credit_code",
    "工商注册号": "business_registration_number",
    "纳税人识别号（国税）": "national_tax_id",
    "纳税人识别号（地税）": "local_tax_id",
}


def extract_enterprise_identity_facts(parse_result: Any) -> dict[str, Any]:
    """Read identity fields across split canonical identity tables.

    A page break or PDF producer may split the label/value form into multiple
    physical tables.  Conflicting values are withheld rather than silently
    selecting whichever table happened to be visited last.
    """
    facts: dict[str, Any] = {}
    conflicts: set[str] = set()
    page_texts = _page_texts(parse_result)
    last_identity_page: int | None = None
    for page, _table_id, rows in _table_stream(parse_result):
        recognized = [row for row in rows if len(row) >= 2 and _compact(row[0]) in _IDENTITY_LABELS]
        page_signature = _compact(page_texts.get(page, ""))
        in_identity_context = bool(
            "身份标识" in page_signature
            or len(recognized) >= 2
            or (last_identity_page is not None and 0 <= page - last_identity_page <= 1)
        )
        if not recognized or not in_identity_context:
            continue
        last_identity_page = page
        for row in recognized:
            field = _IDENTITY_LABELS.get(_compact(row[0]))
            value = _compact(row[1])
            if not field or value in _MISSING_MARKERS:
                continue
            if field in facts and facts[field] != value:
                conflicts.add(field)
                continue
            facts[field] = value
    for field in conflicts:
        facts.pop(field, None)
    return facts


def extract_enterprise_overview(parse_result: Any) -> dict[str, Any]:
    """Read overview facts that are represented as compact canonical tables."""
    summary: dict[str, Any] = {}
    overview_labels = {
        "首次有信贷交易的年份": "first_credit_year",
        "发生信贷交易的机构数": "credit_institution_count",
        "当前有未结清信贷交易的机构数": "active_credit_institution_count",
        "首次有相关还款责任的年份": "first_repayment_responsibility_year",
    }
    public_labels = {
        "非信贷交易账户数": "non_credit_accounts",
        "欠税记录条数": "tax_arrears",
        "民事判决记录条数": "civil_judgments",
        "强制执行记录条数": "enforcements",
        "行政处罚记录条数": "administrative_penalties",
    }
    for _page, _table_id, rows in _table_stream(parse_result):
        if not rows:
            continue
        headers = rows[0]
        if len(rows) >= 2 and all(any(label in header for header in headers) for label in overview_labels):
            values = rows[1]
            for index, header in enumerate(headers):
                field = next((key for label, key in overview_labels.items() if label in header), "")
                if not field:
                    continue
                value = values[index] if index < len(values) else ""
                number = _number(value)
                if number is not None:
                    summary[field] = int(number)
                elif field == "first_repayment_responsibility_year" and value == "--":
                    summary["first_repayment_responsibility_year_status"] = "not_reported"
        if len(rows) >= 2 and all(any(label in header for header in headers) for label in public_labels):
            values = rows[1]
            counts: dict[str, int] = {}
            for index, header in enumerate(headers):
                field = next((key for label, key in public_labels.items() if label in header), "")
                number = _number(values[index] if index < len(values) else "")
                if field and number is not None:
                    counts[field] = int(number)
            if counts:
                summary["public_record_counts"] = counts
        signature = "".join(headers)
        if "借贷交易" not in signature or "担保交易" not in signature:
            continue
        for row in rows[1:]:
            if len(row) >= 4 and row[0] == "余额" and row[2] == "余额":
                summary["credit_balance"] = _number(row[1])
                summary["guarantee_balance"] = _number(row[3])
                summary["amount_unit"] = "CNY_10K"
            for index, value in enumerate(row[:-1]):
                number = _number(row[index + 1])
                if number is None:
                    continue
                if "被追偿余额" in value:
                    summary["recovered_debt_balance"] = number
                elif "关注类余额" in value:
                    target = "credit_attention_balance" if index < 2 else "guarantee_attention_balance"
                    summary[target] = number
                elif "不良类余额" in value:
                    target = "credit_adverse_balance" if index < 2 else "guarantee_adverse_balance"
                    summary[target] = number
    return summary


def _classification_group(label: str) -> str:
    compact = _compact(label)
    if "正常" in compact:
        return "normal"
    if "关注" in compact:
        return "attention"
    if any(marker in compact for marker in ("不良", "次级", "可疑", "损失")):
        return "adverse"
    if "合计" in compact or "总计" in compact:
        return "total"
    return ""


def _current_summary_schema(header_rows: list[list[str]]) -> dict[str, int]:
    """Bind classification summary columns by labels, not physical widths."""
    if len(header_rows) < 2:
        return {}
    groups, metrics = header_rows[:2]
    active_group = ""
    schema: dict[str, int] = {}
    for index in range(max(len(groups), len(metrics))):
        group = _classification_group(groups[index] if index < len(groups) else "")
        if group:
            active_group = group
        metric = _compact(metrics[index] if index < len(metrics) else "")
        if not active_group:
            continue
        if "账户数" in metric or metric in {"户数", "笔数"}:
            schema[f"{active_group}_account_count"] = index
        elif "余额" in metric or "金额" in metric:
            schema[f"{active_group}_balance"] = index
    required = {
        "normal_account_count",
        "normal_balance",
        "attention_account_count",
        "attention_balance",
        "adverse_account_count",
        "adverse_balance",
        "total_account_count",
        "total_balance",
    }
    return schema if required <= schema.keys() else {}


def _values_for_schema(row: list[str], schema: dict[str, int]) -> dict[str, int | float] | None:
    values: dict[str, int | float] = {}
    for field, index in schema.items():
        parsed = _number(row[index] if index < len(row) else "")
        if parsed is None:
            return None
        values[field] = parsed
    return values


def _closed_summary_schema(header: list[str]) -> dict[str, int]:
    schema: dict[str, int] = {}
    for index, label in enumerate(header):
        group = _classification_group(label)
        if group and ("账户" in _compact(label) or "户数" in _compact(label) or group == "total"):
            schema[group] = index
    required = {"normal", "attention", "adverse", "total"}
    return schema if required <= schema.keys() else {}


def _closed_summary_values(
    row: list[str],
    schema: dict[str, int],
) -> tuple[int | float, int | float, int | float, int | float] | None:
    ordered = []
    for group in ("normal", "attention", "adverse", "total"):
        index = schema[group]
        value = _number(row[index] if index < len(row) else "")
        ordered.append(value)
    if any(value is None for value in ordered):
        # Merged cells can move spacer columns on a continuation page.  The
        # semantic contract still requires exactly four reported counts.
        dense = [_number(value) for value in row if _number(value) is not None]
        if len(dense) != 4:
            return None
        ordered = dense
    return tuple(value for value in ordered if value is not None)  # type: ignore[return-value]


def _enterprise_transaction_group(
    categories: set[str],
    heading: str = "",
    *,
    summary_kind: str = "current",
) -> str:
    signature = "".join(sorted(categories)) + _compact(heading)
    if any(marker in signature for marker in ("借款", "透支", "贴现", "融资", "借贷")):
        return "借贷交易"
    if any(marker in signature for marker in ("承兑", "信用证")):
        return "银行承兑汇票和信用证" if summary_kind == "closed" else "担保交易"
    if any(marker in signature for marker in ("保函", "担保")):
        return "银行保函及其他业务"
    # Preserve new PBOC categories instead of dropping a structurally valid
    # table just because its business label is not in today's vocabulary.
    return "其他"


def extract_enterprise_summary_datasets(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Project the page-level responsibility and closed-credit summaries."""
    current_rows: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    displayed_rows: list[dict[str, Any]] = []
    responsibility_rows: list[dict[str, Any]] = []
    resolver = EnterpriseContinuationResolver(parse_result)
    headings = _table_headings(parse_result)
    for fragment in resolver.fragments:
        page = fragment.page
        table_id = fragment.table_id
        rows = [[_compact(value) for value in row] for row in fragment.rows]
        if not rows:
            continue
        headers = rows[0]
        displayed_header_index = next(
            (
                row_index
                for row_index, row in enumerate(rows[1:], start=1)
                if "账户数" in "".join(row)
                and "五级分类" in "".join(row)
                and any(marker in "".join(row) for marker in ("授信机构", "债权机构"))
                and any(marker in "".join(row) for marker in ("业务种类", "业务类型"))
                and any(marker in "".join(row) for marker in ("余额", "贴现金额", "垫款标志"))
            ),
            -1,
        )
        if displayed_header_index >= 0:
            title_rows = rows[:displayed_header_index]
            title_signature = "".join(value for row in title_rows for value in row)
            group_count_match = re.search(r"共(\d+)笔", title_signature)
            summary_title = next(
                (value for row in title_rows for value in row if value and not re.fullmatch(r"共\d+笔", value)),
                "",
            )
            if group_count_match and summary_title:
                displayed_headers = rows[displayed_header_index]

                def displayed_index(*markers: str) -> int:
                    return next(
                        (
                            index
                            for index, value in enumerate(displayed_headers)
                            if any(marker in value for marker in markers)
                        ),
                        -1,
                    )

                displayed_indexes = {
                    "institution": displayed_index("授信机构", "债权机构"),
                    "business_type": displayed_index("业务种类", "业务类型"),
                    "five_tier_class": displayed_index("五级分类"),
                    "source_account_count": displayed_index("账户数"),
                    "balance": displayed_index("余额"),
                    "discount_amount": displayed_index("贴现金额"),
                    "overdue_total": displayed_index("逾期总额"),
                    "overdue_principal": displayed_index("逾期本金"),
                    "advance_flag": displayed_index("垫款标志"),
                }
                if displayed_indexes["discount_amount"] >= 0:
                    settlement_status = "settled"
                    amount_kind = "discount_amount"
                    amount_index = displayed_indexes["discount_amount"]
                elif displayed_indexes["advance_flag"] >= 0:
                    settlement_status = "settled"
                    amount_kind = "not_applicable"
                    amount_index = -1
                else:
                    settlement_status = "active"
                    amount_kind = "balance"
                    amount_index = displayed_indexes["balance"]
                if any(marker in summary_title for marker in ("借款", "贴现", "透支")):
                    transaction_group = "借贷交易"
                elif any(marker in summary_title for marker in ("承兑", "信用证", "保函", "担保")):
                    transaction_group = "担保交易"
                else:
                    transaction_group = "其他"

                emitted = 0
                for row_index, row in enumerate(rows[displayed_header_index + 1 :], start=displayed_header_index + 1):
                    account_count_index = displayed_indexes["source_account_count"]
                    account_count = _number(row[account_count_index]) if 0 <= account_count_index < len(row) else None
                    institution_index = displayed_indexes["institution"]
                    business_type_index = displayed_indexes["business_type"]
                    five_tier_index = displayed_indexes["five_tier_class"]
                    if (
                        account_count is None
                        or institution_index < 0
                        or institution_index >= len(row)
                        or business_type_index < 0
                        or business_type_index >= len(row)
                        or five_tier_index < 0
                        or five_tier_index >= len(row)
                    ):
                        continue

                    def displayed_value(field: str) -> str:
                        index = displayed_indexes[field]
                        return _compact(row[index]) if 0 <= index < len(row) else ""

                    source_reported_amount = _number(row[amount_index]) if 0 <= amount_index < len(row) else None
                    displayed_rows.append(
                        {
                            "displayed_summary_id": _stable_id(
                                "enterprise_displayed_credit_summary",
                                settlement_status,
                                summary_title,
                                displayed_value("institution"),
                                displayed_value("business_type"),
                                page,
                                table_id,
                                row_index,
                            ),
                            "sequence": len(displayed_rows) + 1,
                            "settlement_status": settlement_status,
                            "transaction_group": transaction_group,
                            "business_category": summary_title,
                            "institution": displayed_value("institution"),
                            "business_type": displayed_value("business_type"),
                            "five_tier_class": displayed_value("five_tier_class"),
                            "source_group_account_count": int(group_count_match.group(1)),
                            "source_account_count": int(account_count),
                            "source_reported_amount": source_reported_amount,
                            "amount_kind": amount_kind,
                            "overdue_total": _number(displayed_value("overdue_total")),
                            "overdue_principal": _number(displayed_value("overdue_principal")),
                            "advance_flag": displayed_value("advance_flag"),
                            "summary_scope": "displayed_detail_section",
                            "currency": "CNY",
                            "amount_unit": "CNY_10K",
                            "source_page": page,
                            "source_table_id": table_id,
                            "source": "canonical_enterprise_displayed_credit_summary",
                            "source_refs": [
                                _source_ref(page, table_id, title_row_index)
                                for title_row_index in range(displayed_header_index + 1)
                            ]
                            + [_source_ref(page, table_id, row_index)],
                            "confidence": 1.0,
                        }
                    )
                    emitted += 1
                if emitted:
                    continue
        current_schema = _current_summary_schema(rows[:2])
        if len(rows) >= 3 and current_schema:
            categories = {_compact(row[0]) for row in rows[2:] if row and _compact(row[0])}
            transaction_group = _enterprise_transaction_group(
                categories,
                headings.get(table_id, ""),
            )
            for row_index, row in enumerate(rows[2:], start=2):
                category = _compact(row[0]) if row else ""
                values = _values_for_schema(row, current_schema)
                if not category or values is None:
                    continue
                current_rows.append(
                    {
                                "current_summary_id": _stable_id(
                                    "enterprise_current_summary",
                                    transaction_group,
                                    category,
                                    page,
                                    table_id,
                                ),
                                "sequence": len(current_rows) + 1,
                                "transaction_group": transaction_group,
                                "business_category": category,
                                "normal_account_count": int(values["normal_account_count"]),
                                "normal_balance": values["normal_balance"],
                                "attention_account_count": int(values["attention_account_count"]),
                                "attention_balance": values["attention_balance"],
                                "adverse_account_count": int(values["adverse_account_count"]),
                                "adverse_balance": values["adverse_balance"],
                                "total_account_count": int(values["total_account_count"]),
                                "total_balance": values["total_balance"],
                                "currency": "CNY",
                                "amount_unit": "CNY_10K",
                                "is_total": category == "合计",
                                "source_page": page,
                                "source_table_id": table_id,
                                "source": "canonical_enterprise_current_credit_summary",
                                "source_refs": [_source_ref(page, table_id, row_index)],
                                "confidence": 1.0,
                            }
                        )
            continue
        closed_schema = _closed_summary_schema(headers)
        if closed_schema:
            closed_data: list[tuple[int, str, int, list[str]]] = [
                (page, table_id, row_index, row) for row_index, row in enumerate(rows[1:], start=1)
            ]
            if not closed_data:
                match = resolver.following_row(
                    fragment,
                    CLOSED_SUMMARY_BODY_CONTRACT,
                )
                if match is not None:
                    candidate_rows = [[_compact(value) for value in row] for row in match.fragment.rows]
                    if all(_closed_summary_values(row, closed_schema) is not None for row in candidate_rows):
                        closed_data = [
                            (
                                match.fragment.page,
                                match.fragment.table_id,
                                row_index,
                                row,
                            )
                            for row_index, row in enumerate(candidate_rows)
                        ]
            categories = {_compact(row[0]) for _, _, _, row in closed_data if row}
            transaction_group = _enterprise_transaction_group(
                categories,
                headings.get(table_id, ""),
                summary_kind="closed",
            )
            for source_page, source_table_id, row_index, row in closed_data:
                category = next((_compact(value) for value in row if _compact(value) and _number(value) is None), "")
                counts = _closed_summary_values(row, closed_schema)
                if not category or counts is None:
                    continue
                closed_rows.append(
                    {
                        "closed_summary_id": _stable_id(
                            "enterprise_closed_summary",
                            transaction_group,
                            category,
                            source_page,
                            source_table_id,
                        ),
                        "sequence": len(closed_rows) + 1,
                        "transaction_group": transaction_group,
                        "business_category": category,
                        "normal_account_count": int(counts[0]),
                        "attention_account_count": int(counts[1]),
                        "adverse_account_count": int(counts[2]),
                        "total_account_count": int(counts[3]),
                        "is_total": category == "合计",
                        "source_page": source_page,
                        "source_table_id": source_table_id,
                        "source": "canonical_enterprise_closed_credit_summary",
                        "source_refs": [
                            _source_ref(page, table_id, 0),
                            _source_ref(source_page, source_table_id, row_index),
                        ],
                        "confidence": 1.0,
                    }
                )
            continue
        responsibility_signature = "".join(value for row in rows[:2] for value in row)
        grouped_responsibility_header = (
            "被追偿业务" in responsibility_signature and "其他借贷交易" in responsibility_signature
        )
        flat_responsibility_header = (
            responsibility_signature.count("还款责任金额") >= 2
            and responsibility_signature.count("账户数") >= 2
            and responsibility_signature.count("余额") >= 4
        )
        if (
            len(headers) >= 9
            and (grouped_responsibility_header or flat_responsibility_header)
            and "还款责任金额" in responsibility_signature
            and "账户数" in responsibility_signature
            and "余额" in responsibility_signature
        ):
            data_start = 2 if len(rows) > 1 and "还款责任金额" in "".join(rows[1]) else 1
            for row_index, row in enumerate(rows[data_start:], start=data_start):
                if len(row) < 9 or not _compact(row[0]):
                    continue
                values = [_number(value) for value in row[1:9]]
                responsibility_type = _compact(row[0])
                responsibility_rows.append(
                    {
                        "responsibility_summary_id": _stable_id(
                            "enterprise_responsibility_summary",
                            responsibility_type,
                            page,
                            table_id,
                        ),
                        "sequence": len(responsibility_rows) + 1,
                        "responsibility_type": responsibility_type,
                        "recovered_responsibility_amount": values[0],
                        "recovered_account_count": (int(values[1]) if values[1] is not None else None),
                        "recovered_balance": values[2],
                        "other_credit_responsibility_amount": values[3],
                        "other_credit_account_count": (int(values[4]) if values[4] is not None else None),
                        "other_credit_balance": values[5],
                        "other_credit_attention_balance": values[6],
                        "other_credit_adverse_balance": values[7],
                        "is_total": responsibility_type == "合计",
                        "currency": "CNY",
                        "amount_unit": "CNY_10K",
                        "source_page": page,
                        "source_table_id": table_id,
                        "source": "canonical_enterprise_responsibility_summary",
                        "source_refs": [_source_ref(page, table_id, row_index)],
                        "confidence": 1.0,
                    }
                )
            continue
        if (
            len(headers) >= 6
            and "责任类型" in responsibility_signature
            and "担保交易" in responsibility_signature
            and "还款责任金额" in responsibility_signature
            and "账户数" in responsibility_signature
            and responsibility_signature.count("余额") >= 3
        ):
            data_start = 2 if len(rows) > 1 and "还款责任金额" in "".join(rows[1]) else 1
            for row_index, row in enumerate(rows[data_start:], start=data_start):
                if len(row) < 6 or not _compact(row[0]):
                    continue
                values = [_number(value) for value in row[1:6]]
                responsibility_type = _compact(row[0])
                responsibility_rows.append(
                    {
                        "responsibility_summary_id": _stable_id(
                            "enterprise_responsibility_summary",
                            "担保交易",
                            responsibility_type,
                            page,
                            table_id,
                        ),
                        "sequence": len(responsibility_rows) + 1,
                        "responsibility_type": responsibility_type,
                        "transaction_group": "担保交易",
                        "guarantee_responsibility_amount": values[0],
                        "guarantee_account_count": (int(values[1]) if values[1] is not None else None),
                        "guarantee_balance": values[2],
                        "guarantee_attention_balance": values[3],
                        "guarantee_adverse_balance": values[4],
                        "is_total": responsibility_type == "合计",
                        "currency": "CNY",
                        "amount_unit": "CNY_10K",
                        "source_page": page,
                        "source_table_id": table_id,
                        "source": "canonical_enterprise_responsibility_summary",
                        "source_refs": [_source_ref(page, table_id, row_index)],
                        "confidence": 1.0,
                    }
                )
    return {
        "enterprise_current_credit_summary": current_rows,
        "enterprise_closed_credit_summary": closed_rows,
        "enterprise_displayed_credit_summary": displayed_rows,
        "enterprise_repayment_responsibility_summary": responsibility_rows,
    }


def extract_enterprise_repayment_liability_records(
    parse_result: Any,
) -> list[dict[str, Any]]:
    """Project related-repayment detail cards, including page continuations."""
    records: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    detail_section = False

    def finish_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        pending.pop("_fragment_index", None)
        pending.pop("_row_index", None)
        pending["sequence"] = len(records) + 1
        pending["continuation_complete"] = bool(pending.get("snapshot_date"))
        records.append(pending)
        pending = None

    resolver = EnterpriseContinuationResolver(parse_result)

    def continuation_row_shape(row: list[str]) -> bool:
        values = [_compact(value) for value in row]
        return bool(
            len(values) >= 9
            and not values[0]
            and _number(values[1]) is not None
            and _number(values[2]) is not None
            and _date(values[8])
        )

    for fragment in resolver.fragments:
        page = fragment.page
        table_id = fragment.table_id
        rows = [[_compact(value) for value in row] for row in fragment.rows]
        for row_index, row in enumerate(rows):
            signature = "".join(row)
            if (
                "账户编号" in signature
                and "责任类型" in signature
                and "保证合同编号" in signature
                and "还款责任金额" in signature
                and ("授信机构" in signature or "债权机构" in signature)
            ):
                detail_section = True
                continue

            is_immediate_continuation = bool(
                pending is not None
                and (
                    (pending.get("_fragment_index") == fragment.index and pending.get("_row_index") == row_index - 1)
                    or (
                        pending.get("_fragment_index") is not None
                        and row_index == 0
                        and resolver.table_continues(
                            resolver.fragments[int(pending["_fragment_index"])],
                            fragment,
                            candidate_row_index=row_index,
                            candidate_validator=continuation_row_shape,
                            context="repayment_liability_detail",
                        )
                    )
                )
            )
            if pending is not None and is_immediate_continuation and len(row) >= 9 and not _compact(row[0]):
                loan_or_credit_amount = _number(row[1])
                balance = _number(row[2])
                snapshot_date = _date(row[8])
                if loan_or_credit_amount is not None and balance is not None and snapshot_date:
                    overdue_value = _number(row[6])
                    pending.update(
                        {
                            "loan_or_credit_amount": loan_or_credit_amount,
                            "balance": balance,
                            "five_tier_class": ("" if _compact(row[3]) in {"--", "-", "—"} else _compact(row[3])),
                            "overdue_total": _number(row[4]),
                            "overdue_principal": _number(row[5]),
                            "overdue_months_or_repayment_status": (
                                overdue_value
                                if overdue_value is not None
                                else ("" if _compact(row[6]) in {"--", "-", "—"} else _compact(row[6]))
                            ),
                            "remaining_periods": _number(row[7]),
                            "snapshot_date": snapshot_date,
                            "source_page_end": page,
                            "source_table_id_end": table_id,
                        }
                    )
                    _append_ref(pending, page, table_id, row_index)
                    finish_pending()
                    continue
            elif pending is not None and not is_immediate_continuation:
                finish_pending()

            if not detail_section or len(row) < 10:
                continue
            account_identifier = _identifier(row[0])
            contract_number = _identifier(row[2])
            responsibility_amount = _number(row[4])
            open_date = _date(row[7])
            due_date = _date(row[8])
            if (
                len(account_identifier) < 12
                or not open_date
                or not _compact(row[1])
                or not _looks_like_currency(row[3])
            ):
                continue
            finish_pending()
            currency = _currency_code(row[3])
            pending = {
                "liability_id": f"repayment_liability:{account_identifier}",
                "account_identifier": account_identifier,
                "responsibility_type": _compact(row[1]),
                "contract_number": contract_number,
                "contract_number_status": ("reported" if contract_number else "not_reported"),
                "currency": currency,
                "amount_unit": _amount_unit_for_currency(currency),
                "responsibility_amount": responsibility_amount,
                "responsibility_amount_reported": responsibility_amount is not None,
                "responsibility_amount_status": ("reported" if responsibility_amount is not None else "not_reported"),
                "institution": _compact(row[5]),
                "business_type": _compact(row[6]),
                "open_date": open_date,
                "due_date": due_date,
                "due_date_status": "reported" if due_date else "not_reported",
                "source_page": page,
                "source_table_id": table_id,
                "source": "canonical_enterprise_repayment_liability_detail",
                "source_refs": [_source_ref(page, table_id, row_index)],
                "confidence": 1.0,
                "_fragment_index": fragment.index,
                "_row_index": row_index,
            }
    finish_pending()
    return records


def extract_enterprise_continuation_audit(
    parse_result: Any,
    *,
    datasets: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Reconcile source contract rows with document-wide logical records."""
    resolved = {str(name): list(records) for name, records in (datasets or {}).items()}
    resolver = EnterpriseContinuationResolver(parse_result)
    expected = {
        "current_credit_summary": 0,
        "closed_credit_summary": 0,
        "repayment_responsibility_summary": 0,
        "repayment_liability": 0,
        "attachment_account": 0,
    }
    for fragment in resolver.fragments:
        rows = [[_compact(value) for value in row] for row in fragment.rows]
        if not rows:
            continue
        first_two = "".join(value for row in rows[:2] for value in row)
        current_schema = _current_summary_schema(rows[:2])
        if len(rows) >= 3 and current_schema:
            expected["current_credit_summary"] += sum(
                1
                for row in rows[2:]
                if row and _compact(row[0]) and _values_for_schema(row, current_schema) is not None
            )
        closed_schema = _closed_summary_schema(rows[0])
        if closed_schema:
            body = rows[1:]
            if not body:
                match = resolver.following_row(fragment, CLOSED_SUMMARY_BODY_CONTRACT)
                body = [[_compact(value) for value in row] for row in match.fragment.rows] if match is not None else []
            expected["closed_credit_summary"] += sum(
                1
                for row in body
                if _closed_summary_values(row, closed_schema) is not None
            )
        if (
            len(rows[0]) >= 9
            and "还款责任金额" in first_two
            and "账户数" in first_two
            and "余额" in first_two
            and (("被追偿业务" in first_two and "其他借贷交易" in first_two) or first_two.count("还款责任金额") >= 2)
        ):
            start = 2 if len(rows) > 1 and "还款责任金额" in "".join(rows[1]) else 1
            expected["repayment_responsibility_summary"] += sum(
                1 for row in rows[start:] if len(row) >= 9 and _compact(row[0])
            )
        elif len(rows[0]) >= 6 and "责任类型" in first_two and "担保交易" in first_two and "还款责任金额" in first_two:
            start = 2 if len(rows) > 1 and "还款责任金额" in "".join(rows[1]) else 1
            expected["repayment_responsibility_summary"] += sum(
                1 for row in rows[start:] if len(row) >= 6 and _compact(row[0])
            )
        for row in rows:
            if (
                len(row) >= 10
                and len(_identifier(row[0])) >= 12
                and _compact(row[1])
                and _compact(row[3]) in _CURRENCY_CODES
                and _date(row[7])
            ):
                expected["repayment_liability"] += 1

    attachment_started = False
    for _page, kind, value in _page_flow(parse_result):
        if kind != "text":
            continue
        text = _compact(value)
        if text == "附件" or re.search(r"附件\d*[:：]?信用记录补充信息", text):
            attachment_started = True
        if not attachment_started:
            continue
        expected["attachment_account"] += len(
            re.findall(
                r"\d+\.(?:已结清|未结清)(?:账户编号[:：]?[0-9A-Z]{6,}|业务)",
                text,
            )
        )

    if not any(
        key in resolved
        for key in (
            "enterprise_current_credit_summary",
            "enterprise_closed_credit_summary",
            "enterprise_repayment_responsibility_summary",
        )
    ):
        resolved.update(extract_enterprise_summary_datasets(parse_result))
    if "repayment_liability_records" not in resolved:
        resolved["repayment_liability_records"] = extract_enterprise_repayment_liability_records(parse_result)
    if "enterprise_attachment_accounts" not in resolved or "enterprise_attachment_credit_details" not in resolved:
        attachment_datasets = extract_enterprise_attachment_datasets(parse_result)
        for name, records in attachment_datasets.items():
            resolved.setdefault(name, records)
    actual = {
        "current_credit_summary": len(resolved.get("enterprise_current_credit_summary") or []),
        "closed_credit_summary": len(resolved.get("enterprise_closed_credit_summary") or []),
        "repayment_responsibility_summary": len(resolved.get("enterprise_repayment_responsibility_summary") or []),
        "repayment_liability": len(resolved.get("repayment_liability_records") or []),
        "attachment_account": len(resolved.get("enterprise_attachment_accounts") or []),
    }
    audits: list[dict[str, Any]] = []
    for sequence, family in enumerate(expected, start=1):
        unresolved = max(0, expected[family] - actual[family])
        unexpected = max(0, actual[family] - expected[family])
        audits.append(
            {
                "audit_id": f"enterprise_continuation_audit:{family}",
                "sequence": sequence,
                "continuation_family": family,
                "expected_record_count": expected[family],
                "extracted_record_count": actual[family],
                "unresolved_record_count": unresolved,
                "unexpected_record_count": unexpected,
                "reconciliation_status": ("complete" if not unresolved and not unexpected else "unresolved"),
                "source": "enterprise_continuation_contract_audit",
                "confidence": 1.0,
            }
        )
    page_text_values = tuple(_page_texts(parse_result).values())
    detail_population_not_comparable = any(
        ("受篇幅所限" in text and "只展示部分信贷记录" in text) or "展示样式" in text for text in page_text_values
    )
    if not detail_population_not_comparable:
        closed_summaries = list(resolved.get("enterprise_closed_credit_summary") or [])
        attachment_contexts = list(resolved.get("enterprise_attachment_accounts") or [])
        attachment_details = list(resolved.get("enterprise_attachment_credit_details") or [])
        settled_business_categories = sorted(
            {
                str(context.get("business_category") or "")
                for context in attachment_contexts
                if context.get("attachment_record_type") == "business"
                and context.get("account_status") == "settled"
                and context.get("business_category")
            }
        )
        for business_category in settled_business_categories:
            expected_count = sum(
                int(record.get("total_account_count") or 0)
                for record in closed_summaries
                if not record.get("is_total") and record.get("transaction_group") == business_category
            )
            if expected_count <= 0:
                continue
            extracted_count = sum(
                1
                for record in attachment_details
                if record.get("account_status") == "settled" and record.get("business_category") == business_category
            )
            unresolved = max(0, expected_count - extracted_count)
            unexpected = max(0, extracted_count - expected_count)
            audits.append(
                {
                    "audit_id": (f"enterprise_continuation_audit:attachment_credit_detail:{business_category}:settled"),
                    "sequence": len(audits) + 1,
                    "continuation_family": "attachment_credit_detail",
                    "business_category": business_category,
                    "account_status": "settled",
                    "expected_record_count": expected_count,
                    "extracted_record_count": extracted_count,
                    "unresolved_record_count": unresolved,
                    "unexpected_record_count": unexpected,
                    "reconciliation_status": ("complete" if not unresolved and not unexpected else "unresolved"),
                    "source": "enterprise_continuation_contract_audit",
                    "confidence": 1.0,
                }
            )
    return audits


def _placeholder_row(row: list[str]) -> bool:
    meaningful = [_compact(value) for value in row if _compact(value)]
    return bool(meaningful) and all(value in {"--", "-", "—"} for value in meaningful)


def extract_enterprise_capital_summary(parse_result: Any) -> list[dict[str, Any]]:
    """Separate registered capital from contributor-record availability."""
    headings = _table_headings(parse_result)
    page_texts = _page_texts(parse_result)
    for page, table_id, rows in _table_stream(parse_result):
        if not rows:
            continue
        signature = "".join(rows[0])
        if not all(marker in signature for marker in ("类型", "出资方", "身份标识号码")):
            continue
        data_rows = [row for row in rows[1:] if row and not row[0].startswith("信息来源")]
        contributor_status = (
            "no_records" if data_rows and all(_placeholder_row(row) for row in data_rows) else "reported"
        )
        capital_match: re.Match[str] | None = None
        capital_page = page
        for candidate_page in range(max(1, page - 1), page + 1):
            source_text = "\n".join(
                (
                    headings.get(table_id, "") if candidate_page == page else "",
                    page_texts.get(candidate_page, ""),
                )
            )
            capital_match = re.search(
                r"注册资本(?:折人民币)?(?:合计)?[:：]?\s*([0-9][0-9,.]*)\s*(万元|元)",
                source_text,
            )
            if capital_match:
                capital_page = candidate_page
                break
        capital_amount = _number(capital_match.group(1)) if capital_match else None
        amount_unit = "CNY_10K" if capital_match and capital_match.group(2) == "万元" else "CNY"
        record: dict[str, Any] = {
            "sequence": 1,
            "contributor_status": contributor_status,
            "contributor_count": sum(1 for row in data_rows if not _placeholder_row(row)),
            "source_page": capital_page,
            "source": "canonical_enterprise_capital_table",
            "source_refs": [_source_ref(page, table_id, 0)],
            "confidence": 1.0,
            **_table_source_metadata(rows),
        }
        if capital_page != page:
            record["contributor_source_page"] = page
            record["source_refs"].insert(
                0,
                {
                    "source": "native_text_registered_capital",
                    "page": capital_page,
                },
            )
        if capital_amount is not None:
            record.update(
                {
                    "registered_capital_amount": capital_amount,
                    "currency": "CNY",
                    "amount_unit": amount_unit,
                }
            )
        return [record]
    return []


def extract_enterprise_profile_status(parse_result: Any) -> dict[str, Any]:
    """Compatibility helper exposing the accurately scoped contributor status."""
    rows = extract_enterprise_capital_summary(parse_result)
    if not rows:
        return {}
    return {"contributor_status": rows[0]["contributor_status"]}


def extract_enterprise_report_metadata(
    parse_result: Any,
    full_text: str = "",
) -> dict[str, Any]:
    """Recover report-edition and exchange-rate metadata from native text."""
    text = full_text or "\n".join(_page_texts(parse_result).values())
    compact = _compact(text)
    facts: dict[str, Any] = {}
    if "自主查询版" in compact:
        facts["report_edition"] = "independent_query"
    exchange_match = re.search(
        r"汇率[（(]美元折人民币[）)][:：]?([0-9]+(?:\.[0-9]+)?)"
        r".{0,30}?有效期[:：]?((?:19|20)\d{2}[-/.]\d{1,2})",
        compact,
    )
    if exchange_match:
        facts["exchange_rate_usd_cny"] = _number(exchange_match.group(1))
        period = exchange_match.group(2).replace("/", "-").replace(".", "-")
        year, month = period.split("-", 1)
        facts["exchange_rate_effective_period"] = f"{int(year):04d}-{int(month):02d}"
    return facts


def extract_enterprise_report_metadata_records(
    parse_result: Any,
    full_text: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Return cover and report-note metadata at their source-page grain."""
    metadata = extract_enterprise_report_metadata(parse_result, full_text)
    page_texts = _page_texts(parse_result)
    datasets: dict[str, list[dict[str, Any]]] = {}

    report_edition = metadata.get("report_edition")
    if report_edition:
        page = next(
            (page_number for page_number, text in page_texts.items() if "自主查询版" in _compact(text)),
            None,
        )
        record: dict[str, Any] = {
            "sequence": 1,
            "report_edition": report_edition,
            "source": "enterprise_report_cover",
            "confidence": 1.0,
        }
        if page is not None:
            record["source_page"] = page
            record["source_refs"] = [{"source": "native_text_report_edition", "page": page}]
        datasets["enterprise_report_metadata"] = [record]

    exchange_rate = metadata.get("exchange_rate_usd_cny")
    exchange_period = metadata.get("exchange_rate_effective_period")
    if exchange_rate is not None or exchange_period:
        page = next(
            (
                page_number
                for page_number, text in page_texts.items()
                if "汇率" in _compact(text) and "有效期" in _compact(text)
            ),
            None,
        )
        record = {
            "sequence": 1,
            "exchange_rate_usd_cny": exchange_rate,
            "exchange_rate_effective_period": exchange_period,
            "source": "enterprise_report_notes",
            "confidence": 1.0,
        }
        if page is not None:
            record["source_page"] = page
            record["source_refs"] = [{"source": "native_text_exchange_rate", "page": page}]
        datasets["enterprise_exchange_rates"] = [record]
    return datasets


def extract_enterprise_report_identity_records(
    parse_result: Any,
    full_text: str = "",
) -> list[dict[str, Any]]:
    """Copy cover and identity-table business identifiers into one dataset."""
    page_texts = _page_texts(parse_result)
    text = full_text or "\n".join(page_texts.values())
    compact = _compact(text)
    identity = dict(extract_enterprise_identity_facts(parse_result))
    metadata = extract_enterprise_report_metadata(parse_result, text)
    query_match = re.search(
        r"查询机构[:：]?(.+?)(?=报告时间[:：]?|$)",
        compact,
    )
    report_time_match = re.search(
        r"报告时间[:：]?((?:19|20)\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
        compact,
    )
    cover_name_match = re.search(
        r"企业名称[:：]?(.+?)(?=中征码[:：]?)",
        compact,
    )
    if cover_name_match and not identity.get("subject_name"):
        identity["subject_name"] = _compact(cover_name_match.group(1))
    if query_match:
        identity["query_institution"] = _compact(query_match.group(1))
    if report_time_match:
        identity["report_time"] = report_time_match.group(1)
    identity.update(metadata)
    if not identity:
        return []
    identity["enterprise_name"] = identity.get("subject_name")
    identity["sequence"] = 1
    identity["enterprise_identity_id"] = _stable_id(
        "enterprise_report_identity",
        identity.get("subject_name"),
        identity.get("unified_social_credit_code"),
        identity.get("report_time"),
    )
    source_pages = [
        page
        for page, value in page_texts.items()
        if any(marker in _compact(value) for marker in ("企业名称", "身份标识", "报告时间"))
    ]
    if source_pages:
        identity["source_page"] = min(source_pages)
        identity["source_page_end"] = max(source_pages)
        identity["source_refs"] = [
            {"source": "enterprise_cover_or_identity", "page": page}
            for page in sorted(set(source_pages))
        ]
    identity["source"] = "enterprise_report_identity"
    identity["confidence"] = 1.0
    return [identity]


def extract_enterprise_overview_datasets(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Return report-level dispute, credit overview, recovery, and overdue rows."""
    datasets: dict[str, list[dict[str, Any]]] = {}
    page_texts = _page_texts(parse_result)
    overview = extract_enterprise_overview(parse_result)
    if overview:
        public_record_counts = overview.pop("public_record_counts", None)
        record = {
            "enterprise_credit_overview_id": _stable_id(
                "enterprise_credit_overview",
                overview.get("first_credit_year"),
                overview.get("credit_balance"),
                overview.get("guarantee_balance"),
            ),
            "sequence": 1,
            **overview,
            "currency": "CNY",
            "source_page": 3,
            "source": "enterprise_information_overview",
            "source_refs": [{"source": "enterprise_information_overview", "page": 3}],
            "confidence": 1.0,
        }
        datasets["enterprise_credit_overview"] = [record]
        if isinstance(public_record_counts, dict):
            count_type_aliases = {
                "civil_judgments": "civil_judgment",
                "enforcements": "enforcement",
                "administrative_penalties": "administrative_penalty",
            }
            datasets["enterprise_public_record_counts"] = [
                {
                    "enterprise_public_record_count_id": _stable_id(
                        "enterprise_public_record_count",
                        record_type,
                    ),
                    "sequence": index,
                    "record_type": count_type_aliases.get(record_type, record_type),
                    "record_count": int(count),
                    "source_page": 3,
                    "source": "enterprise_information_overview",
                    "source_refs": [
                        {"source": "enterprise_information_overview", "page": 3}
                    ],
                    "confidence": 1.0,
                }
                for index, (record_type, count) in enumerate(
                    public_record_counts.items(),
                    start=1,
                )
            ]

    for page, text in page_texts.items():
        match = re.search(
            r"提出了\s*(\d+)\s*笔异议且正在处理中",
            _compact(text),
        )
        if not match:
            continue
        count = int(match.group(1))
        datasets["enterprise_dispute_overview"] = [
            {
                "enterprise_dispute_overview_id": _stable_id(
                    "enterprise_dispute_overview",
                    page,
                    count,
                ),
                "sequence": 1,
                "in_progress_dispute_count": count,
                "dispute_status": "in_progress",
                "source_page": page,
                "source": "enterprise_dispute_notice",
                "source_refs": [{"source": "enterprise_dispute_notice", "page": page}],
                "confidence": 1.0,
            }
        ]
        break

    recovery_rows: list[dict[str, Any]] = []
    overdue_rows: list[dict[str, Any]] = []
    for page, table_id, rows in _table_stream(parse_result):
        if len(rows) < 3:
            continue
        title_index = next(
            (
                index
                for index, row in enumerate(rows)
                if "由资产管理公司处置的债务" in "".join(row)
                and "垫款" in "".join(row)
                and "逾期" in "".join(row)
            ),
            -1,
        )
        if title_index < 0 or title_index + 2 >= len(rows):
            continue
        values = rows[title_index + 2]
        if len(values) < 9:
            continue
        source_ref = _source_ref(page, table_id, title_index + 2)
        for recovery_type, offset, activity_field in (
            ("asset_management_disposed_debt", 0, "latest_disposal_date"),
            ("advance", 3, "latest_repayment_date"),
        ):
            account_count = _number(values[offset])
            amount = _number(values[offset + 1])
            activity_date = _date(values[offset + 2])
            if account_count is None and amount is None and not activity_date:
                continue
            row = {
                "enterprise_recovery_summary_id": _stable_id(
                    "enterprise_recovery_summary",
                    "active",
                    recovery_type,
                    page,
                ),
                "sequence": len(recovery_rows) + 1,
                "settlement_status": "active",
                "recovery_type": recovery_type,
                "account_count": int(account_count) if account_count is not None else None,
                "balance": amount,
                "currency": "CNY",
                "amount_unit": "CNY_10K",
                activity_field: activity_date or None,
                "source_page": page,
                "source_table_id": table_id,
                "source": "enterprise_recovery_summary",
                "source_refs": [source_ref],
                "confidence": 1.0,
            }
            recovery_rows.append(row)
        overdue_rows.append(
            {
                "enterprise_overdue_summary_id": _stable_id(
                    "enterprise_overdue_summary",
                    page,
                    table_id,
                ),
                "sequence": 1,
                "overdue_principal": _number(values[6]),
                "overdue_interest_and_other": _number(values[7]),
                "overdue_total": _number(values[8]),
                "currency": "CNY",
                "amount_unit": "CNY_10K",
                "source_page": page,
                "source_table_id": table_id,
                "source": "enterprise_overdue_summary",
                "source_refs": [source_ref],
                "confidence": 1.0,
            }
        )
        break
    if recovery_rows:
        datasets["enterprise_recovery_summary"] = recovery_rows
    if overdue_rows:
        datasets["enterprise_overdue_summary"] = overdue_rows
    return datasets


def extract_enterprise_interest_arrears(
    parse_result: Any,
) -> list[dict[str, Any]]:
    """Return one row per source 欠息 record."""
    records: list[dict[str, Any]] = []
    headings = _table_headings(parse_result)
    for page, table_id, rows in _table_stream(parse_result):
        heading = _compact(headings.get(table_id))
        for row_index, row in enumerate(rows):
            if len(row) < 6:
                continue
            arrears_type = _compact(row[1])
            currency = _CURRENCY_CODES.get(_compact(row[2]))
            balance = _number(row[3])
            balance_change_date = _date(row[4])
            snapshot_date = _date(row[5])
            if (
                not currency
                or balance is None
                or not balance_change_date
                or not snapshot_date
                or (arrears_type not in {"表内", "表外"} and "欠息" not in heading)
            ):
                continue
            institution = _compact(row[0])
            records.append(
                {
                    "interest_arrears_id": _stable_id(
                        "enterprise_interest_arrears",
                        institution,
                        arrears_type,
                        snapshot_date,
                    ),
                    "sequence": len(records) + 1,
                    "institution": institution,
                    "arrears_type": arrears_type,
                    "currency": currency,
                    "amount_unit": "CNY_10K",
                    "arrears_balance": balance,
                    "balance_change_date": balance_change_date,
                    "snapshot_date": snapshot_date,
                    "source_page": page,
                    "source_table_id": table_id,
                    "source": "enterprise_interest_arrears",
                    "source_refs": [_source_ref(page, table_id, row_index)],
                    "confidence": 1.0,
                }
            )
    return records


def extract_enterprise_non_credit_history_datasets(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Return monthly utility and housing-fund attachment histories."""
    datasets: dict[str, list[dict[str, Any]]] = {}
    page_texts = _page_texts(parse_result)
    for page, table_id, rows in _table_stream(parse_result):
        page_text = _compact(page_texts.get(page))
        history_type = (
            "housing_fund"
            if "住房公积金历史缴费记录明细" in page_text
            else "utility"
            if "公用事业历史缴费记录明细" in page_text
            else ""
        )
        if not history_type:
            continue
        records = datasets.setdefault(
            (
                "enterprise_housing_fund_history"
                if history_type == "housing_fund"
                else "enterprise_utility_payment_history"
            ),
            [],
        )
        for row_index, row in enumerate(rows):
            if len(row) < 5 or not re.fullmatch(r"(?:19|20)\d{2}-\d{2}", _compact(row[0])):
                continue
            statistics_month = _compact(row[0])
            records.append(
                {
                    f"{history_type}_history_id": _stable_id(
                        f"enterprise_{history_type}_history",
                        statistics_month,
                        page,
                        row_index,
                    ),
                    "sequence": len(records) + 1,
                    "statistics_month": statistics_month,
                    "payment_status": _compact(row[1]),
                    "amount_due": _number(row[2]),
                    "amount_paid": _number(row[3]),
                    "cumulative_arrears": _number(row[4]),
                    "currency": "CNY",
                    "amount_unit": "CNY_1",
                    "source_page": page,
                    "source_table_id": table_id,
                    "source": f"enterprise_{history_type}_history",
                    "source_refs": [_source_ref(page, table_id, row_index)],
                    "confidence": 1.0,
                }
            )
    return {name: rows for name, rows in datasets.items() if rows}


def extract_enterprise_report_notes(parse_result: Any) -> list[dict[str, Any]]:
    """Transcribe numbered enterprise report notes from their source page."""
    for page, page_text in _page_texts(parse_result).items():
        heading = re.search(r"报告说明", page_text)
        if heading is None:
            continue
        note_text = page_text[heading.end() :]
        note_text = re.split(r"汇率[（(]美元折人民币[）)]", note_text, maxsplit=1)[0]
        note_text = re.sub(r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页", "", note_text)
        notes: list[dict[str, Any]] = []
        for match in re.finditer(
            r"(?ms)(?<!\d)(\d{1,2})[.．、]\s*(.*?)(?=(?<!\d)\d{1,2}[.．、]\s*|\Z)",
            note_text,
        ):
            sequence = int(match.group(1))
            content = re.sub(r"\s*\n\s*", "", match.group(2))
            content = re.sub(r"[ \t]+", " ", content).strip()
            if not content:
                continue
            notes.append(
                {
                    "note_id": _stable_id("enterprise_report_note", sequence, content),
                    "sequence": sequence,
                    "content": content,
                    "source_page": page,
                    "source": "enterprise_report_notes",
                    "source_refs": [{"source": "native_text_report_note", "page": page}],
                    "confidence": 1.0,
                }
            )
        if notes:
            return notes
    return []


def extract_enterprise_profile_datasets(parse_result: Any) -> dict[str, list[dict[str, Any]]]:
    """Extract enterprise-only profile and related-party tables."""
    profile: list[dict[str, Any]] = []
    stakeholders: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    headings = _table_headings(parse_result)
    for page, table_id, rows in _table_stream(parse_result):
        if not rows:
            continue
        headers = rows[0]
        signature = "".join(headers)
        source_metadata = _table_source_metadata(rows)
        if any(row and row[0] in _PROFILE_LABELS for row in rows):
            for row_index, row in enumerate(rows):
                if len(row) < 2 or row[0] not in _PROFILE_LABELS:
                    continue
                profile.append(
                    {
                        "sequence": len(profile) + 1,
                        "field": row[0],
                        "value": row[1],
                        "source_institution": row[3] if len(row) > 3 else "",
                        "page": page,
                        "source": "canonical_enterprise_profile_table",
                        "source_refs": [_source_ref(page, table_id, row_index)],
                        "confidence": 1.0,
                    }
                )
            continue
        if all(marker in signature for marker in ("类型", "出资方", "身份标识号码")):
            for row_index, row in enumerate(rows[1:], start=1):
                if len(row) < 4 or not row[0] or row[0].startswith("信息来源") or _placeholder_row(row):
                    continue
                stakeholders.append(
                    {
                        "sequence": len(stakeholders) + 1,
                        "role": row[0],
                        "name": row[1],
                        "identity_type": row[2],
                        "identity_number": row[3],
                        "ownership_percentage": row[4] if len(row) > 4 else "",
                        **source_metadata,
                        "page": page,
                        "source": "canonical_enterprise_stakeholder_table",
                        "source_refs": [_source_ref(page, table_id, row_index)],
                        "confidence": 1.0,
                    }
                )
            continue
        management_header = all(marker in signature for marker in ("职位", "姓名", "证件号码"))
        management_continuation = not management_header and any(
            len(row) >= 4 and row[2] in {"身份证", "护照", "港澳居民来往内地通行证"} and len(_identifier(row[3])) >= 8
            for row in rows
        )
        if management_header or management_continuation:
            start_index = 1 if management_header else 0
            for row_index, row in enumerate(rows[start_index:], start=start_index):
                if len(row) < 4 or not row[0] or row[0].startswith("信息来源"):
                    continue
                stakeholders.append(
                    {
                        "sequence": len(stakeholders) + 1,
                        "role": row[0],
                        "name": row[1],
                        "identity_type": row[2],
                        "identity_number": row[3],
                        **source_metadata,
                        "page": page,
                        "source": "canonical_enterprise_management_table",
                        "source_refs": [_source_ref(page, table_id, row_index)],
                        "confidence": 1.0,
                    }
                )
            continue
        relationship_headers = all(marker in signature for marker in ("类型", "名称", "身份标识号码")) or all(
            marker in signature for marker in ("名称", "身份标识类型", "身份标识号码")
        )
        if relationship_headers:
            for row_index, row in enumerate(rows[1:], start=1):
                if len(row) < 3 or not row[0] or row[0].startswith("信息来源") or _placeholder_row(row):
                    continue
                has_type_column = "类型" == headers[0]
                heading = headings.get(table_id, "")
                relationship_type = (
                    row[0]
                    if has_type_column
                    else ("actual_controller" if "实际控制人" in heading else "related_enterprise")
                )
                relationships.append(
                    {
                        "sequence": len(relationships) + 1,
                        "relationship_type": relationship_type,
                        "name": row[1] if has_type_column else row[0],
                        "identity_type": row[2] if has_type_column else row[1],
                        "identity_number": row[3] if has_type_column and len(row) > 3 else row[2],
                        **source_metadata,
                        "page": page,
                        "source": "canonical_enterprise_relationship_table",
                        "source_refs": [_source_ref(page, table_id, row_index)],
                        "confidence": 1.0,
                    }
                )
    return {
        "enterprise_profile_fields": profile,
        "enterprise_stakeholders": stakeholders,
        "enterprise_relationships": relationships,
    }


_ATTACHMENT_CATEGORY_MARKERS = {
    "被追偿业务的历史表现": "被追偿业务",
    "中长期借款的历史表现": "中长期借款",
    "短期借款的历史表现": "短期借款",
    "循环透支的历史表现": "循环透支",
    "贴现的历史表现": "贴现",
    "贴现的信贷明细": "贴现",
    "银行承兑汇票和信用证的信贷明细": "银行承兑汇票和信用证",
    "银行保函及其他业务的信贷明细": "银行保函及其他业务",
}


def _vertical_position(value: Any, fallback: float) -> float:
    bbox = list(getattr(value, "bbox", None) or [])
    if len(bbox) >= 2:
        try:
            return float(bbox[1])
        except (TypeError, ValueError):
            pass
    return fallback


def _page_flow(parse_result: Any) -> Iterator[tuple[int, str, Any]]:
    """Yield page text and tables in visual order, preserving continuations."""
    if isinstance(parse_result, EnterpriseExtractionContext):
        yield from parse_result.page_flow
        return
    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or page_index)
        items: list[tuple[float, int, int, str, Any]] = []
        for index, block in enumerate(getattr(page, "texts", None) or []):
            content = str(getattr(block, "content", "") or "")
            if content:
                items.append(
                    (
                        _vertical_position(block, float(index)),
                        0,
                        index,
                        "text",
                        content,
                    )
                )
        for index, table in enumerate(getattr(page, "tables", None) or []):
            rows = _raw_table_rows(table)
            if rows:
                items.append(
                    (
                        _vertical_position(table, 10000.0 + float(index)),
                        1,
                        index,
                        "table",
                        (
                            str(getattr(table, "table_id", "") or ""),
                            rows,
                        ),
                    )
                )
        for _position, _kind_order, _index, kind, value in sorted(items):
            yield page_number, kind, value


def _attachment_label(text: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}[:：](.*?)(?="
        r"授信机构[:：]|业务种类[:：]|五级分类[:：]|历史表现|"
        r"第\d+页/?共\d+页|$)",
        text,
    )
    return _compact(match.group(1)) if match else ""


def _history_table(rows: list[list[str]]) -> bool:
    signature = "".join(rows[0]) if rows else ""
    return "信息报告日期" in signature and "余额" in signature and "余额变化日期" in signature


def _attachment_contexts_and_records(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    contexts: list[dict[str, Any]] = []
    history_records: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    special_records: list[dict[str, Any]] = []
    current_category = ""
    current: dict[str, Any] | None = None
    pending_history: tuple[list[str], dict[str, Any], int, str, int] | None = None
    pending_detail_header: tuple[list[list[str]], dict[str, Any]] | None = None
    resolver = EnterpriseContinuationResolver(parse_result)
    allowed_history_continuation_tables: set[str] = set()
    for fragment in resolver.fragments:
        fragment_rows = [[_compact(value) for value in row] for row in fragment.rows]
        if not _history_table(fragment_rows):
            continue
        match = resolver.following_row(
            fragment,
            ATTACHMENT_HISTORY_BODY_CONTRACT,
        )
        if match is not None:
            allowed_history_continuation_tables.add(match.fragment.table_id)

    allowed_detail_continuation_tables: set[str] = set()
    for fragment in resolver.fragments:
        fragment_rows = [[_compact(value) for value in row] for row in fragment.rows]
        header_index = next(
            (
                index
                for index, row in enumerate(fragment_rows)
                if ("开户日期" in row or "开立日期" in row)
                and any(marker in "".join(row) for marker in ("贴现金额", "借款金额", "信用额度", "金额"))
            ),
            -1,
        )
        if header_index < 0:
            continue
        header = fragment_rows[header_index]
        open_date_index = next(
            (index for index, value in enumerate(header) if "开户日期" in value or "开立日期" in value),
            -1,
        )
        amount_index = next(
            (
                index
                for index, value in enumerate(header)
                if any(marker in value for marker in ("贴现金额", "借款金额", "信用额度", "金额"))
            ),
            -1,
        )

        def valid_detail_row(row: list[str]) -> bool:
            return bool(
                0 <= open_date_index < len(row)
                and 0 <= amount_index < len(row)
                and _date(row[open_date_index])
                and _number(row[amount_index]) is not None
            )

        for candidate in resolver.following_fragments(
            fragment,
            candidate_validator=valid_detail_row,
            context="attachment_credit_detail",
        ):
            candidate_rows = [[_compact(value) for value in row] for row in candidate.rows]
            candidate_starts_new_table = any(
                ("开户日期" in row or "开立日期" in row)
                and any(marker in "".join(row) for marker in ("贴现金额", "借款金额", "信用额度", "金额"))
                for row in candidate_rows
            )
            if candidate_starts_new_table or not any(valid_detail_row(row) for row in candidate_rows):
                break
            allowed_detail_continuation_tables.add(candidate.table_id)

    def start_context(
        *,
        source_page: int,
        source_sequence: int,
        record_type: str,
        status: str,
        account_identifier: str = "",
    ) -> dict[str, Any]:
        nonlocal pending_detail_header, pending_history
        pending_history = None
        pending_detail_header = None
        identifier = _identifier(account_identifier)
        attachment_account_id = _stable_id(
            "enterprise_attachment_account",
            current_category,
            record_type,
            source_sequence,
            identifier,
            source_page,
        )
        context = {
            "attachment_account_id": attachment_account_id,
            "sequence": len(contexts) + 1,
            "source_sequence": source_sequence,
            "attachment_record_type": record_type,
            "account_status": status,
            "business_category": current_category,
            "source_page": source_page,
            "source_page_end": source_page,
            "source": "native_enterprise_attachment_heading",
            "source_refs": [
                {
                    "source": "native_enterprise_attachment_heading",
                    "page": source_page,
                }
            ],
            "confidence": 1.0,
        }
        if identifier:
            context["account_identifier"] = identifier
            context["account_id"] = f"credit_account:{identifier}"
        contexts.append(context)
        return context

    def update_context_from_text(context: dict[str, Any], text: str) -> None:
        institution = _attachment_label(text, "授信机构")
        business_type = _attachment_label(text, "业务种类")
        five_tier_class = _attachment_label(text, "五级分类")
        if institution:
            context["institution"] = institution
        if business_type:
            context["business_type"] = business_type
        if five_tier_class:
            context["five_tier_class"] = five_tier_class

    def emit_history(
        primary: list[str],
        secondary: list[str],
        context: dict[str, Any],
        first_page: int,
        first_table: str,
        first_row: int,
        second_page: int,
        second_table: str,
        second_row: int,
    ) -> None:
        primary_values = [_compact(value) for value in primary if _compact(value)]
        secondary_values = [_compact(value) for value in secondary if _compact(value)]
        if len(primary_values) < 7 or len(secondary_values) < 6:
            return
        report_date = _date(primary_values[0])
        if not report_date:
            return
        revolving = len(primary_values) >= 8
        secondary_offset = 0 if revolving else 1
        attachment_account_id = str(context["attachment_account_id"])
        history_records.append(
            {
                "supplement_id": _stable_id(
                    "enterprise_history",
                    attachment_account_id,
                    report_date,
                    len(history_records) + 1,
                ),
                "sequence": len(history_records) + 1,
                "attachment_account_id": attachment_account_id,
                "account_id": context.get("account_id"),
                "account_identifier": context.get("account_identifier", ""),
                "institution": context.get("institution", ""),
                "business_type": context.get("business_type", ""),
                "business_category": context.get("business_category", ""),
                "account_status": context.get("account_status", ""),
                "report_date": report_date,
                "balance": _number(primary_values[1]),
                "balance_change_date": _date(primary_values[2]),
                "five_tier_class": ("" if primary_values[3] == "--" else primary_values[3]),
                "classification_date": _date(primary_values[4]),
                "overdue_total": _number(primary_values[5]),
                "overdue_principal": _number(primary_values[6]),
                "overdue_months": _number(primary_values[7] if revolving else secondary_values[0]),
                "scheduled_repayment_date": _date(secondary_values[secondary_offset]),
                "scheduled_repayment_amount": _number(secondary_values[secondary_offset + 1]),
                "actual_repayment_date": _date(secondary_values[secondary_offset + 2]),
                "actual_repayment_amount": _number(secondary_values[secondary_offset + 3]),
                "repayment_method": (
                    "" if secondary_values[secondary_offset + 4] == "--" else secondary_values[secondary_offset + 4]
                ),
                "remaining_periods": (
                    _number(secondary_values[secondary_offset + 5])
                    if len(secondary_values) > secondary_offset + 5
                    else None
                ),
                "currency": "CNY",
                "amount_unit": "CNY_10K",
                "source_page": first_page,
                "source_table_id": first_table,
                "source": "canonical_enterprise_monthly_history",
                "source_refs": [
                    _source_ref(first_page, first_table, first_row),
                    _source_ref(second_page, second_table, second_row),
                ],
                "confidence": 1.0,
            }
        )

    def consume_history(
        rows: list[list[str]],
        context: dict[str, Any],
        page: int,
        table_id: str,
    ) -> None:
        nonlocal pending_history
        offset = 2 if _history_table(rows) else 0
        for row_index, row in enumerate(rows[offset:], start=offset):
            values = [_compact(value) for value in row]
            if _date(values[0] if values else ""):
                pending_history = (values, context, page, table_id, row_index)
                continue
            if pending_history and pending_history[1] is context and values and not values[0]:
                primary, owner, first_page, first_table, first_row = pending_history
                emit_history(
                    primary,
                    values,
                    owner,
                    first_page,
                    first_table,
                    first_row,
                    page,
                    table_id,
                    row_index,
                )
                pending_history = None

    def consume_credit_details(
        rows: list[list[str]],
        context: dict[str, Any],
        page: int,
        table_id: str,
        *,
        allow_headerless: bool = False,
    ) -> bool:
        nonlocal pending_detail_header
        header_markers = (
            "账户编号",
            "开户日期",
            "开立日期",
            "到期日",
            "币种",
            "贴现金额",
            "借款金额",
            "信用额度",
            "金额",
            "关闭日期",
            "担保方式",
            "反担保方式",
            "保证金比例",
            "保证金率",
            "余额",
            "风险敞口",
            "五级分类",
            "授信协议编号",
            "信用协议编号",
            "信息报告日期",
            "最后一次还款日期",
            "最后一次还款形式",
            "垫款标志",
        )

        def is_header_row(row: list[str]) -> bool:
            signature = "".join(_compact(value) for value in row)
            return bool(signature and any(marker in signature for marker in header_markers))

        header_index = next(
            (
                index
                for index, row in enumerate(rows)
                if any(value in {"开户日期", "开立日期"} for value in row)
                and any(marker in value for value in row for marker in ("贴现金额", "金额", "借款金额", "信用额度"))
            ),
            -1,
        )
        if header_index >= 0:
            header_band = [rows[header_index]]
            data_start = header_index + 1
            while data_start < len(rows) and is_header_row(rows[data_start]):
                header_band.append(rows[data_start])
                data_start += 1
            data_rows = list(enumerate(rows[data_start:], start=data_start))
        elif allow_headerless and pending_detail_header and pending_detail_header[1] is context:
            header_band = [list(row) for row in pending_detail_header[0]]
            data_start = 0
            while data_start < len(rows) and is_header_row(rows[data_start]):
                header_band.append(rows[data_start])
                data_start += 1
            data_rows = list(enumerate(rows[data_start:], start=data_start))
        else:
            return False

        def location_of(*markers: str, exact: bool = True) -> tuple[int, int] | None:
            return next(
                (
                    (header_row_index, column_index)
                    for header_row_index, header_row in enumerate(header_band)
                    for column_index, value in enumerate(header_row)
                    if any(_compact(value) == marker if exact else marker in _compact(value) for marker in markers)
                ),
                None,
            )

        locations = {
            "account_identifier": location_of("账户编号"),
            "open_date": location_of("开户日期", "开立日期"),
            "due_date": location_of("到期日", "到期日期"),
            "currency": location_of("币种"),
            "amount": location_of("贴现金额", "借款金额", "信用额度", "金额"),
            "close_date": location_of("关闭日期"),
            "guarantee_type": location_of("担保方式", "保证方式"),
            "counter_guarantee_type": location_of("反担保方式", "反担保措施"),
            "deposit_ratio": location_of("保证金比例", "保证金率"),
            "balance": location_of("余额"),
            "risk_exposure_amount": location_of("风险敞口", "风险敞口金额"),
            "five_tier_class": location_of("五级分类"),
            "credit_agreement_identifier": location_of(
                "授信协议编号",
                "信用协议编号",
            ),
            "snapshot_date": location_of("信息报告日期"),
            "last_repayment_date": location_of("最后一次还款日期"),
            "repayment_method": location_of("最后一次还款形式"),
            "advance_flag": location_of("垫款标志"),
        }

        open_date_location = locations["open_date"]
        amount_location = locations["amount"]

        def is_primary_row(row: list[str]) -> bool:
            if open_date_location is None or amount_location is None:
                return False
            open_date_column = open_date_location[1]
            amount_column = amount_location[1]
            identifier_location = locations["account_identifier"]
            identifier_valid = True
            if identifier_location is not None:
                identifier_column = identifier_location[1]
                identifier_valid = (
                    identifier_column < len(row)
                    and len(_identifier(row[identifier_column])) >= _MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH
                )
            return bool(
                identifier_valid
                and open_date_column < len(row)
                and amount_column < len(row)
                and _date(row[open_date_column])
                and _number(row[amount_column]) is not None
            )

        logical_rows: list[list[tuple[int, list[str]] | None]] = []
        row_cursor = 0
        while row_cursor < len(data_rows):
            primary_row_index, primary_row = data_rows[row_cursor]
            if not any(_compact(value) for value in primary_row) or not is_primary_row(primary_row):
                row_cursor += 1
                continue
            group: list[tuple[int, list[str]] | None] = [(primary_row_index, primary_row)]
            row_cursor += 1
            for _header_row_index in range(1, len(header_band)):
                while row_cursor < len(data_rows) and not any(_compact(value) for value in data_rows[row_cursor][1]):
                    row_cursor += 1
                if row_cursor >= len(data_rows) or is_primary_row(data_rows[row_cursor][1]):
                    group.append(None)
                    continue
                group.append(data_rows[row_cursor])
                row_cursor += 1
            logical_rows.append(group)

        emitted = 0
        for logical_row in logical_rows:
            primary = logical_row[0]
            if primary is None:
                continue
            row_index, _primary_row = primary

            def value(field: str) -> str:
                location = locations[field]
                if location is None:
                    return ""
                header_row_index, column_index = location
                if header_row_index >= len(logical_row):
                    return ""
                physical_row = logical_row[header_row_index]
                if physical_row is None:
                    return ""
                row = physical_row[1]
                return _compact(row[column_index]) if column_index < len(row) else ""

            amount = _number(value("amount"))
            open_date = _date(value("open_date"))
            if amount is None or not open_date:
                continue
            row_identifier = _identifier(value("account_identifier"))
            identifier = row_identifier or str(context.get("account_identifier") or "")
            detail_five_tier_class = value("five_tier_class")
            five_tier_class = "" if detail_five_tier_class in {"--", "-", "—"} else detail_five_tier_class
            classification_source = "detail_table" if five_tier_class else ""
            inherited_classification = False
            if locations["five_tier_class"] is None and not five_tier_class and context.get("five_tier_class"):
                five_tier_class = str(context["five_tier_class"])
                classification_source = "parent_attachment_heading"
                inherited_classification = True

            agreement_value = value("credit_agreement_identifier")
            agreement_missing = agreement_value in {"", "--", "-", "—"}
            if locations["credit_agreement_identifier"] is None:
                agreement_status = "not_applicable"
            elif agreement_missing:
                agreement_status = "not_reported"
            else:
                agreement_status = "reported"

            source_refs = [
                _source_ref(page, table_id, physical_row[0]) for physical_row in logical_row if physical_row is not None
            ]
            if inherited_classification:
                for source_ref in context.get("source_refs") or []:
                    if source_ref not in source_refs:
                        source_refs.append(source_ref)
            guarantee_type = value("guarantee_type")
            counter_guarantee_type = value("counter_guarantee_type")
            detail_records.append(
                {
                    "attachment_detail_id": _stable_id(
                        "enterprise_attachment_detail",
                        context["attachment_account_id"],
                        identifier,
                        open_date,
                        page,
                        table_id,
                        row_index,
                    ),
                    "sequence": len(detail_records) + 1,
                    "attachment_account_id": context["attachment_account_id"],
                    "account_identifier": identifier,
                    "institution": context.get("institution", ""),
                    "business_type": context.get("business_type", ""),
                    "business_category": context.get("business_category", ""),
                    "account_status": context.get("account_status", ""),
                    "open_date": open_date,
                    "due_date": _date(value("due_date")),
                    "currency": _CURRENCY_CODES.get(value("currency"), value("currency")),
                    "amount": amount,
                    "amount_unit": "CNY_10K",
                    "close_date": _date(value("close_date")),
                    "guarantee_type": ("" if guarantee_type in {"--", "-", "—"} else guarantee_type),
                    "counter_guarantee_type": (
                        "" if counter_guarantee_type in {"--", "-", "—"} else counter_guarantee_type
                    ),
                    "deposit_ratio": _percentage(value("deposit_ratio")),
                    "balance": _number(value("balance")),
                    "risk_exposure_amount": _number(value("risk_exposure_amount")),
                    "five_tier_class": five_tier_class,
                    "five_tier_class_source": classification_source,
                    "credit_agreement_identifier": ("" if agreement_missing else _identifier(agreement_value)),
                    "credit_agreement_status": agreement_status,
                    "snapshot_date": _date(value("snapshot_date")),
                    "last_repayment_date": _date(value("last_repayment_date")),
                    "repayment_method": ("" if value("repayment_method") == "--" else value("repayment_method")),
                    "advance_flag": ("" if value("advance_flag") == "--" else value("advance_flag")),
                    "source_page": page,
                    "source_table_id": table_id,
                    "source": "canonical_enterprise_attachment_credit_detail",
                    "source_refs": source_refs,
                    "confidence": 1.0,
                }
            )
            emitted += 1
        if emitted or header_index >= 0:
            pending_detail_header = (header_band, context)
        return True

    def consume_special_transactions(
        rows: list[list[str]],
        context: dict[str, Any],
        page: int,
        table_id: str,
    ) -> bool:
        header_index = next(
            (index for index, row in enumerate(rows) if "交易类型" in row and "交易日期" in row),
            -1,
        )
        if header_index < 0:
            return False
        header = rows[header_index]

        def index_of(marker: str) -> int:
            return next(
                (index for index, value in enumerate(header) if marker in value),
                -1,
            )

        indexes = {
            "transaction_type": index_of("交易类型"),
            "transaction_date": index_of("交易日期"),
            "transaction_amount": index_of("交易金额"),
            "due_date_change_months": index_of("到期日期变更月数"),
            "transaction_detail": index_of("交易明细信息"),
        }
        for row_index, row in enumerate(rows[header_index + 1 :], start=header_index + 1):

            def value(field: str) -> str:
                index = indexes[field]
                return _compact(row[index]) if 0 <= index < len(row) else ""

            transaction_type = value("transaction_type")
            transaction_date = _date(value("transaction_date"))
            if not transaction_type or not transaction_date:
                continue
            special_records.append(
                {
                    "special_transaction_id": _stable_id(
                        "enterprise_special_transaction",
                        context["attachment_account_id"],
                        transaction_type,
                        transaction_date,
                        page,
                        table_id,
                        row_index,
                    ),
                    "sequence": len(special_records) + 1,
                    "attachment_account_id": context["attachment_account_id"],
                    "account_identifier": context.get("account_identifier", ""),
                    "institution": context.get("institution", ""),
                    "business_type": context.get("business_type", ""),
                    "business_category": context.get("business_category", ""),
                    "transaction_type": transaction_type,
                    "transaction_date": transaction_date,
                    "transaction_amount": _number(value("transaction_amount")),
                    "due_date_change_months": _number(value("due_date_change_months")),
                    "transaction_detail": ("" if value("transaction_detail") == "--" else value("transaction_detail")),
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                    "source_page": page,
                    "source_table_id": table_id,
                    "source": "canonical_enterprise_special_transaction",
                    "source_refs": [_source_ref(page, table_id, row_index)],
                    "confidence": 1.0,
                }
            )
        return True

    attachment_started = False
    for page, kind, value in _page_flow(parse_result):
        if kind == "text":
            text = _compact(value)
            if text == "附件" or re.search(r"附件\d*[:：]?信用记录补充信息", text):
                attachment_started = True
            if not attachment_started:
                continue
            for marker, category in _ATTACHMENT_CATEGORY_MARKERS.items():
                if marker in text:
                    current_category = category
            account_match = re.search(
                r"(\d+)\.(已结清|未结清)账户编号[:：]?([0-9A-Z]{6,})",
                text,
            )
            business_match = re.search(r"(\d+)\.(已结清|未结清)业务", text)
            if account_match:
                current = start_context(
                    source_page=page,
                    source_sequence=int(account_match.group(1)),
                    record_type="account",
                    status="settled" if account_match.group(2) == "已结清" else "active",
                    account_identifier=account_match.group(3),
                )
            elif business_match:
                current = start_context(
                    source_page=page,
                    source_sequence=int(business_match.group(1)),
                    record_type="business",
                    status="settled" if business_match.group(2) == "已结清" else "active",
                )
            if current is not None:
                update_context_from_text(current, text)
            continue

        if not attachment_started or current is None:
            continue
        table_id, rows = value
        current["source_page_end"] = max(int(current.get("source_page_end") or page), page)
        if (
            pending_detail_header
            and pending_detail_header[1] is current
            and table_id in allowed_detail_continuation_tables
        ):
            if consume_credit_details(
                rows,
                current,
                page,
                table_id,
                allow_headerless=True,
            ):
                continue
        if _history_table(rows) or table_id in allowed_history_continuation_tables:
            consume_history(rows, current, page, table_id)
            consume_special_transactions(rows, current, page, table_id)
            continue
        if consume_credit_details(rows, current, page, table_id):
            continue
        consume_special_transactions(rows, current, page, table_id)

    return {
        "enterprise_attachment_accounts": contexts,
        "enterprise_credit_supplement": history_records,
        "enterprise_attachment_credit_details": detail_records,
        "enterprise_special_transactions": special_records,
    }


def extract_enterprise_attachment_datasets(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Return all business-grained datasets from the enterprise attachment."""
    return _attachment_contexts_and_records(parse_result)


def extract_enterprise_supplement_rows(parse_result: Any) -> list[dict[str, Any]]:
    """Return account-bound monthly histories from the full attachment."""
    return extract_enterprise_attachment_datasets(parse_result)["enterprise_credit_supplement"]


_PUBLIC_TABLE_TYPES = {
    "公用事业单位名称": "utility_payment",
    "主管税务机关": "tax_arrears",
    "许可部门": "license",
    "认证部门": "certification",
    "认定部门": "qualification",
    "奖励部门": "award",
    "批准部门出口商品名称生效日期": "export_quality",
    "批准部门免验商品名称免验号截止日期": "inspection_exemption",
    "监管部门管辖直属局监管级别生效日期截止日期": "regulatory_supervision",
    "专利名称": "patent",
    "所属名录": "financing_restriction",
    "数据提供机构说明": "data_provider_statement",
    "征信中心说明": "credit_bureau_statement",
    "信息主体声明": "subject_statement",
    "异议标注": "dispute_annotation",
}

_PUBLIC_CONTINUATION_RULES = {
    "administrative_penalty": {
        "start": frozenset({"处罚机构", "处罚决定书文号"}),
        "continuation": frozenset(
            {"违法行为", "处罚日期", "处罚决定", "处罚金额（元）", "处罚执行情况", "行政复议结果"}
        ),
    },
    "civil_judgment": {
        "start": frozenset({"立案法院", "案号"}),
        "continuation": frozenset({"结案方式", "判决/调解结果", "判决/调解生效日期"}),
    },
    "enforcement": {
        "start": frozenset({"执行法院", "案号"}),
        "continuation": frozenset({"案件状态", "结案方式", "已执行标的", "已执行标的金额（元）"}),
    },
}

_PUBLIC_ATTRIBUTE_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "utility_payment": {
        "utility_provider": ("公用事业单位名称",),
        "utility_type": ("业务类型", "公用事业类型"),
        "utility_account_identifier": ("账户编号", "用户编号"),
        "payment_status": ("缴费状态",),
        "cumulative_arrears": ("累计欠费金额（元）", "累计欠费金额", "累计欠费"),
        "statistics_month": ("统计年月",),
        "history_status": ("查看过去24个月缴费情况", "查看过去 24 个月缴费情况"),
    },
    "tax_arrears": {
        "tax_authority": ("主管税务机关",),
        "tax_arrears_amount": ("欠税总额（元）", "欠税总额"),
        "tax_arrears_statistics_date": ("欠税统计日期",),
    },
    "civil_judgment": {
        "filing_court": ("立案法院",),
        "filing_date": ("立案日期",),
        "cause_of_action": ("案由",),
        "litigation_position": ("诉讼地位",),
        "case_number": ("案号",),
        "trial_procedure": ("审判程序",),
        "claim_subject": ("诉讼标的",),
        "claim_amount": ("诉讼标的金额（元）", "诉讼标的金额"),
        "closure_method": ("结案方式",),
        "judgment_effective_date": ("判决/调解生效日期",),
        "judgment_result": ("判决/调解结果",),
    },
    "enforcement": {
        "enforcement_court": ("执行法院",),
        "enforcement_filing_date": ("立案日期",),
        "enforcement_cause": ("执行案由",),
        "enforcement_case_number": ("案号",),
        "requested_enforcement_subject": ("申请执行标的",),
        "requested_enforcement_amount": ("申请执行标的金额（元）", "申请执行标的金额"),
        "case_status": ("案件状态",),
        "enforcement_closure_method": ("结案方式",),
        "executed_subject": ("已执行标的",),
        "executed_amount": ("已执行标的金额（元）", "已执行标的金额"),
    },
    "administrative_penalty": {
        "penalty_authority": ("处罚机构",),
        "penalty_decision_number": ("处罚决定书文号",),
        "violation": ("违法行为",),
        "penalty_date": ("处罚日期",),
        "penalty_decision": ("处罚决定",),
        "penalty_amount": ("处罚金额（元）", "处罚金额"),
        "penalty_execution_status": ("处罚执行情况",),
        "administrative_review_result": ("行政复议结果",),
    },
    "social_security_payment": {
        "statistics_month": ("统计年月", "统计月份"),
        "initial_contribution_month": ("初缴年月",),
        "employee_count": ("职工人数",),
        "contribution_base": ("缴费基数（元）", "缴存基数（元）", "缴存基数"),
        "last_contribution_date": ("最近一次缴费日期", "最近一次缴存日期"),
        "paid_through_month": ("缴至年月",),
        "payment_status": ("缴费状态", "缴存状态"),
        "cumulative_arrears": ("累计欠费金额（元）", "累计欠缴金额（元）", "累计欠缴金额"),
    },
    "license": {
        "licensing_authority": ("许可部门", "许可机关"),
        "license_type": ("许可类型",),
        "license_date": ("许可日期",),
        "license_expiry_date": ("截止日期", "有效截止日期"),
        "license_content": ("许可内容",),
    },
    "certification": {
        "certification_authority": ("认证部门", "认证机关"),
        "certification_type": ("认证类型",),
        "certification_date": ("认证日期",),
        "certification_expiry_date": ("截止日期", "有效截止日期"),
        "certification_content": ("认证内容",),
    },
    "qualification": {
        "qualification_authority": ("认定部门", "认定机关"),
        "qualification_type": ("资质类型", "认定类型"),
        "qualification_approval_date": ("批准日期", "认定日期"),
        "qualification_expiry_date": ("截止日期", "有效截止日期"),
        "qualification_content": ("资质内容", "认定内容"),
    },
    "award": {
        "award_authority": ("奖励部门", "授予部门"),
        "award_name": ("奖励名称",),
        "award_date": ("授予日期", "奖励日期"),
        "award_expiry_date": ("截止日期", "有效截止日期"),
        "award_fact": ("奖励事实", "奖励内容"),
    },
    "export_quality": {
        "approval_authority": ("批准部门",),
        "export_product_name": ("出口商品名称",),
        "effective_date": ("生效日期",),
        "expiry_date": ("截止日期",),
    },
    "inspection_exemption": {
        "approval_authority": ("批准部门",),
        "inspection_exempt_product_name": ("免验商品名称",),
        "inspection_exemption_number": ("免验号",),
        "expiry_date": ("截止日期",),
    },
    "regulatory_supervision": {
        "regulatory_authority": ("监管部门",),
        "direct_supervising_bureau": ("管辖直属局",),
        "supervision_level": ("监管级别",),
        "effective_date": ("生效日期",),
        "expiry_date": ("截止日期",),
    },
    "patent": {
        "patent_name": ("专利名称",),
        "patent_number": ("专利号",),
        "application_date": ("申请日期",),
        "grant_date": ("授予日期", "授权日期"),
        "validity_years": ("专利有效期（单位：年）", "专利有效期"),
    },
    "financing_restriction": {
        "catalog": ("所属名录",),
        "control_type": ("融资控制类型",),
        "year": ("年度",),
        "scale": ("规模",),
    },
    "data_provider_statement": {
        "statement_content": ("数据提供机构说明",),
        "added_date": ("添加日期",),
    },
    "credit_bureau_statement": {
        "statement_content": ("征信中心说明",),
        "added_date": ("添加日期",),
    },
    "subject_statement": {
        "statement_content": ("信息主体声明",),
        "added_date": ("添加日期",),
    },
    "dispute_annotation": {
        "annotation_content": ("异议标注",),
        "added_date": ("添加日期",),
    },
}

_PUBLIC_DATE_ATTRIBUTE_FIELDS = frozenset(
    {
        "statistics_date",
        "tax_arrears_statistics_date",
        "filing_date",
        "judgment_effective_date",
        "enforcement_filing_date",
        "penalty_date",
        "last_contribution_date",
        "license_date",
        "license_expiry_date",
        "certification_date",
        "certification_expiry_date",
        "qualification_approval_date",
        "qualification_expiry_date",
        "award_date",
        "award_expiry_date",
        "effective_date",
        "expiry_date",
        "application_date",
        "grant_date",
        "added_date",
    }
)
_PUBLIC_INTEGER_ATTRIBUTE_FIELDS = frozenset({"employee_count", "validity_years", "year"})
_PUBLIC_NUMBER_ATTRIBUTE_FIELDS = frozenset(
    {
        "tax_arrears_amount",
        "claim_amount",
        "requested_enforcement_amount",
        "executed_amount",
        "penalty_amount",
        "contribution_base",
        "cumulative_arrears",
    }
)

_PUBLIC_RECORD_TYPE_LABELS = {
    "utility_payment": "公用事业缴费记录",
    "tax_arrears": "欠税记录",
    "civil_judgment": "民事判决记录",
    "enforcement": "强制执行记录",
    "administrative_penalty": "行政处罚记录",
    "social_security_payment": "社会保障及住房公积金缴费记录",
    "license": "行政许可记录",
    "certification": "认证记录",
    "qualification": "资质认定记录",
    "award": "行政奖励记录",
    "export_quality": "出口商品质量记录",
    "inspection_exemption": "免验商品记录",
    "regulatory_supervision": "监管记录",
    "patent": "专利记录",
    "financing_restriction": "融资限制记录",
    "data_provider_statement": "数据提供机构说明",
    "credit_bureau_statement": "征信中心说明",
    "subject_statement": "信息主体声明",
    "dispute_annotation": "异议标注",
}


def enterprise_public_record_dataset_specs() -> dict[str, dict[str, Any]]:
    """Return lossless, chart-grained dataset schemas for public records."""

    specs: dict[str, dict[str, Any]] = {}
    for record_type, field_map in _PUBLIC_ATTRIBUTE_FIELDS.items():
        dataset_id = f"enterprise_public_{record_type}_records"
        columns: dict[str, dict[str, str]] = {}
        for field, labels in field_map.items():
            if field in _PUBLIC_INTEGER_ATTRIBUTE_FIELDS:
                field_type = "integer"
            elif field in _PUBLIC_NUMBER_ATTRIBUTE_FIELDS:
                field_type = "money"
            elif field in _PUBLIC_DATE_ATTRIBUTE_FIELDS:
                # Public-record charts use placeholders such as "--".  These
                # source cells are intentionally strings so the normalized JSON
                # remains lossless while valid dates retain ISO text.
                field_type = "string"
            else:
                field_type = "string"
            columns[field] = {"label": labels[0], "type": field_type}
        label = _PUBLIC_RECORD_TYPE_LABELS.get(record_type, record_type)
        specs[record_type] = {
            "dataset_id": dataset_id,
            "label": label,
            "definition": f"{label}源表中的一行。",
            "columns": columns,
        }
    return specs


def _public_attributes(record_type: str, details: dict[str, Any]) -> dict[str, Any]:
    """Return stable, type-specific JSON fields for one public record."""

    field_map = _PUBLIC_ATTRIBUTE_FIELDS.get(record_type, {})
    normalized_details = {_compact(key): value for key, value in details.items()}
    attributes: dict[str, Any] = {}
    for field, labels in field_map.items():
        raw = next(
            (normalized_details[label] for label in labels if label in normalized_details),
            None,
        )
        if raw in (None, ""):
            continue
        if field in _PUBLIC_DATE_ATTRIBUTE_FIELDS:
            value: Any = _date(raw) or _compact(raw)
        elif field in _PUBLIC_INTEGER_ATTRIBUTE_FIELDS:
            number = _number(raw)
            value = int(number) if number is not None else None
        elif field in _PUBLIC_NUMBER_ATTRIBUTE_FIELDS:
            value = _number(raw)
        else:
            value = _compact(raw)
        if value not in (None, ""):
            attributes[field] = value
    return attributes


def project_enterprise_public_record_datasets(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Expose public records only as typed, business-grained chart tables."""

    datasets: dict[str, list[dict[str, Any]]] = {}
    specs = enterprise_public_record_dataset_specs()
    for record in records:
        record_type = str(record.get("record_type") or "")
        spec = specs.get(record_type)
        if not spec:
            continue
        dataset_id = str(spec["dataset_id"])
        typed_records = datasets.setdefault(dataset_id, [])
        attributes = record.get("attributes")
        if not isinstance(attributes, dict):
            attributes = _public_attributes(
                record_type,
                record.get("details") if isinstance(record.get("details"), dict) else {},
            )
        typed_record: dict[str, Any] = {
            "public_record_id": record.get("public_record_id"),
            "sequence": len(typed_records) + 1,
        }
        for field in spec["columns"]:
            if field in attributes:
                typed_record[field] = attributes[field]
        if record.get("page") is not None:
            typed_record["source_page"] = record["page"]
        if record.get("source_table_id"):
            typed_record["source_table_id"] = record["source_table_id"]
        for key in ("source", "source_refs", "confidence"):
            if key in record:
                typed_record[key] = record[key]
        typed_records.append(typed_record)
    return datasets


def extract_enterprise_public_record_datasets(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Extract public records and expose each source chart as its own dataset."""

    records = extract_enterprise_public_records_from_tables(parse_result)
    return project_enterprise_public_record_datasets(records)


def _public_type(headers: list[str], rows: list[list[str]]) -> str:
    signature = "".join(headers)
    for marker, record_type in _PUBLIC_TABLE_TYPES.items():
        if marker in signature:
            return record_type
    text = "".join("".join(row) for row in rows[:3])
    if "立案法院" in text:
        return "civil_judgment"
    if "执行法院" in text:
        return "enforcement"
    if "处罚机构" in text or "违法行为" in text or "处罚决定" in text:
        return "administrative_penalty"
    if "初缴年月" in text or ("职工人数" in text and "缴费基数" in text):
        return "social_security_payment"
    return ""


def _split_label_value(value: str) -> tuple[str, str]:
    if "：" in value:
        label, content = value.split("：", 1)
        return _compact(label), _compact(content)
    if ":" in value:
        label, content = value.split(":", 1)
        return _compact(label), _compact(content)
    return "", _compact(value)


def _public_date(details: dict[str, Any], *labels: str) -> str:
    for label in labels:
        for key, value in details.items():
            if label in key:
                parsed = _date(value)
                if parsed:
                    return parsed
    return ""


def extract_enterprise_public_records_from_tables(parse_result: Any) -> list[dict[str, Any]]:
    """Project typed public-record tables without cross-section text bleed."""
    records: list[dict[str, Any]] = []
    resolver = EnterpriseContinuationResolver(parse_result)
    fragment_by_table_id = {fragment.table_id: fragment for fragment in resolver.fragments}
    for page, table_id, rows in _table_stream(parse_result):
        if not rows:
            continue
        headers = rows[0]
        record_type = _public_type(headers, rows)
        if not record_type:
            continue
        structured_table = any(marker in "".join(headers) for marker in _PUBLIC_TABLE_TYPES)
        key_value_table = not structured_table and any("：" in cell or ":" in cell for row in rows for cell in row)
        entries: list[tuple[int, dict[str, Any]]] = []
        if key_value_table:
            details: dict[str, Any] = {}
            for row in rows:
                for value in row:
                    label, content = _split_label_value(value)
                    if label and content:
                        details[label] = content
            if details:
                entries.append((0, details))
        else:
            for row_index, row in enumerate(rows[1:], start=1):
                if not any(_compact(value) for value in row):
                    continue
                details = {
                    _compact(label): _compact(value)
                    for label, value in zip(headers, row)
                    if _compact(label) and _compact(value)
                }
                if details:
                    entries.append((row_index, details))
        for row_index, details in entries:
            attributes = _public_attributes(record_type, details)
            authority = next(
                (
                    value
                    for key, value in details.items()
                    if any(marker in key for marker in ("部门", "机关", "法院", "单位名称", "机构"))
                ),
                "",
            )
            category = next(
                (
                    value
                    for key, value in details.items()
                    if any(marker in key for marker in ("类型", "案由", "名称", "级别"))
                ),
                "",
            )
            content = "；".join(f"{key}：{value}" for key, value in details.items())
            record = {
                "public_record_id": _stable_id(
                    "public_record",
                    record_type,
                    page,
                    table_id,
                    row_index,
                    content,
                ),
                "record_type": record_type,
                "authority": authority,
                "category": category,
                "start_date": _public_date(
                    details,
                    "生效日期",
                    "许可日期",
                    "认证日期",
                    "批准日期",
                    "授予日期",
                    "申请日期",
                    "统计日期",
                    "添加日期",
                    "立案日期",
                    "处罚日期",
                    "最近一次缴费日期",
                ),
                "end_date": _public_date(details, "截止日期", "有效截止日期"),
                "content": content,
                "details": details,
                "attributes": attributes,
                **attributes,
                "page": page,
                "source_table_id": table_id,
                "source": "canonical_enterprise_public_record_table",
                "source_refs": [_source_ref(page, table_id, row_index)],
                "confidence": 1.0,
            }
            rules = _PUBLIC_CONTINUATION_RULES.get(record_type)
            previous = records[-1] if records else None
            raw_previous_details = previous.get("details") if isinstance(previous, dict) else None
            previous_details: dict[str, Any] = (
                dict(raw_previous_details) if isinstance(raw_previous_details, dict) else {}
            )
            detail_labels = frozenset(str(key) for key in details)
            previous_labels = frozenset(str(key) for key in previous_details)
            previous_table_id = str(previous.get("source_table_id") if isinstance(previous, dict) else "")
            scored_continuation = bool(
                previous_table_id in fragment_by_table_id
                and table_id in fragment_by_table_id
                and resolver.table_continues(
                    fragment_by_table_id[previous_table_id],
                    fragment_by_table_id[table_id],
                    candidate_row_index=row_index,
                    candidate_validator=lambda _row: bool(
                        rules
                        and previous_labels & rules["start"]
                        and not detail_labels & rules["start"]
                        and detail_labels & rules["continuation"]
                    ),
                    context=f"public_record:{record_type}",
                )
            )
            guarded_continuation = bool(
                rules
                and previous
                and previous.get("record_type") == record_type
                and scored_continuation
                and previous_labels & rules["start"]
                and not detail_labels & rules["start"]
                and detail_labels & rules["continuation"]
            )
            if guarded_continuation:
                previous = records[-1]
                previous["details"].update(details)
                merged_attributes = _public_attributes(record_type, previous["details"])
                previous["attributes"] = merged_attributes
                previous.update(merged_attributes)
                previous["content"] = "；".join(f"{key}：{value}" for key, value in previous["details"].items())
                previous["source_refs"].extend(record["source_refs"])
                previous["source_page_end"] = page
                previous["source_table_id_end"] = table_id
                continue
            records.append(record)
    for index, record in enumerate(records, start=1):
        record["sequence"] = index
    return records


def _reported_account_summary(parse_result: Any) -> dict[str, Any]:
    for page, table_id, rows in _table_stream(parse_result):
        for row_index, row in enumerate(rows):
            if not (len(row) >= 9 and row[0] == "" and row[1] == "正常类" and row[7] == "合计"):
                continue
            counts: dict[str, int] = {}
            balances: dict[str, int | float] = {}
            for values in rows[row_index + 1 :]:
                if len(values) < 9:
                    continue
                category = values[0]
                if category in _ACCOUNT_CATEGORIES:
                    count = _number(values[7])
                    balance = _number(values[8])
                    if isinstance(count, int):
                        counts[category] = count
                    if balance is not None:
                        balances[category] = balance
                elif category == "合计":
                    count = _number(values[7])
                    balance = _number(values[8])
                    if count is not None:
                        return {
                            "reported_account_count": int(count),
                            "reported_account_counts": counts,
                            "reported_account_balances": balances,
                            "reported_account_balance": balance,
                            "source_account_summary_table_id": table_id,
                            "source_account_summary_page": page,
                        }
    return {}


def _merge_accounts(
    original: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    expected: int | None,
) -> list[dict[str, Any]]:
    if not canonical:
        return original

    def identifier(record: dict[str, Any]) -> str:
        return _identifier(record.get("account_identifier"))

    def compatible(candidate: dict[str, Any]) -> bool:
        """Require main-account identity evidence before filling a gap."""
        return bool(
            len(identifier(candidate)) >= _MIN_ENTERPRISE_ACCOUNT_IDENTIFIER_LENGTH
            and _compact(candidate.get("management_institution"))
            and _compact(candidate.get("business_type"))
            and (_date(candidate.get("open_date")) or _date(candidate.get("due_date")))
        )

    def merge_record(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        merged = dict(primary)
        for key, value in secondary.items():
            if key == "source_refs":
                refs = list(merged.get("source_refs") or [])
                for ref in value or []:
                    if ref not in refs:
                        refs.append(ref)
                merged["source_refs"] = refs
            elif merged.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                merged[key] = value
        return merged

    original_by_id = {
        identifier(record): record
        for record in original
        if isinstance(record, dict) and compatible(record)
    }
    merged: list[dict[str, Any]] = []
    consumed_original: set[str] = set()
    for record in canonical:
        canonical_id = identifier(record)
        exact = original_by_id.get(canonical_id)
        prefix_id = next(
            (
                candidate_id
                for candidate_id in original_by_id
                if candidate_id != canonical_id
                and (candidate_id.startswith(canonical_id) or canonical_id.startswith(candidate_id))
            ),
            "",
        )
        fallback = exact or original_by_id.get(prefix_id)
        if fallback is not None:
            consumed_original.add(identifier(fallback))
            record = merge_record(record, fallback)
            fallback_id = identifier(fallback)
            if len(fallback_id) > len(canonical_id) and fallback_id.startswith(canonical_id):
                record["account_identifier"] = fallback_id
                record["account_id"] = f"credit_account:{fallback_id}"
        merged.append(record)

    if expected is None or len(merged) == expected:
        return merged

    candidates = [
        record
        for candidate_id, record in original_by_id.items()
        if candidate_id not in consumed_original
        and not any(
            candidate_id.startswith(identifier(item)) or identifier(item).startswith(candidate_id)
            for item in merged
        )
    ]
    reconciled = [*merged, *candidates]
    if len(reconciled) == expected:
        return reconciled
    if len(original_by_id) == expected:
        # The fallback population exactly reconciles with the report total and
        # every row passed the main-account evidence guard.
        return list(original_by_id.values())
    # Do not guess which extra appendix/history candidate belongs to the main
    # account population.  Returning the strongest canonical population keeps
    # the mismatch visible to the extraction reconciliation audit.
    return merged


def _enterprise_source_display_limited(text: str) -> bool:
    compact = _compact(text)
    partial_credit = bool(
        re.search(
            r"(?:受?篇幅(?:所限|限制)?.{0,30})?"
            r"(?:只|仅)?(?:展示|列示|显示|提供).{0,12}部分.{0,12}(?:信贷|信用)记录",
            compact,
        )
        or re.search(r"部分(?:信贷|信用)记录.{0,12}(?:展示|列示|显示|提供)", compact)
    )
    limited_period = bool(
        re.search(r"(?:仅|只)?展示.{0,12}(?:期限|期间|年限|时间范围).{0,20}已结清信贷信息", compact)
        and any(marker in compact for marker in ("非信贷信息", "公共信息", "公共记录"))
    )
    return partial_credit or limited_period


def refine_enterprise_business(
    parse_result: Any,
    business: dict[str, Any],
) -> dict[str, Any]:
    """Replace heuristic enterprise candidates when canonical cards reconcile."""
    refined = dict(business)
    summary = dict(refined.get("credit_summary") or {})
    summary.update(extract_enterprise_overview(parse_result))
    reported = _reported_account_summary(parse_result)
    summary.update(reported)
    expected = reported.get("reported_account_count")
    canonical_accounts = extract_enterprise_accounts_from_tables(parse_result)
    accounts = _merge_accounts(
        list(refined.get("credit_accounts") or []),
        canonical_accounts,
        int(expected) if isinstance(expected, int) else None,
    )
    credit_lines = extract_enterprise_credit_lines_from_tables(parse_result, accounts)
    repayment_liabilities = extract_enterprise_repayment_liability_records(parse_result)
    public_records = extract_enterprise_public_records_from_tables(parse_result)
    refined["credit_lines"] = credit_lines
    if repayment_liabilities:
        refined["repayment_liability_records"] = repayment_liabilities
    reported_credit_lines = _reported_credit_line_count(parse_result)
    if reported_credit_lines is not None:
        summary["reported_credit_line_count"] = reported_credit_lines
    if public_records:
        refined["public_records"] = public_records
        summary["public_record_count"] = len(public_records)
        public_record_counts: dict[str, int] = {}
        for record in public_records:
            record_type = str(record.get("record_type") or "unknown")
            public_record_counts[record_type] = public_record_counts.get(record_type, 0) + 1
        summary["public_record_type_counts"] = public_record_counts
    refined["credit_accounts"] = accounts
    summary["account_population_comparable"] = bool(isinstance(expected, int) and expected == len(accounts))
    summary["account_population_reconciliation_status"] = (
        "complete"
        if isinstance(expected, int) and expected == len(accounts)
        else "not_reported"
        if not isinstance(expected, int)
        else "unresolved"
    )
    attachment_datasets = extract_enterprise_attachment_datasets(parse_result)
    attachment_accounts = attachment_datasets["enterprise_attachment_accounts"]
    attachment_details = attachment_datasets["enterprise_attachment_credit_details"]
    attachment_transactions = attachment_datasets["enterprise_special_transactions"]
    source_display_limited = any(
        _enterprise_source_display_limited(text)
        for text in _page_texts(parse_result).values()
    )
    summary.update(
        {
            "account_dataset_scope": "main_report_account_cards",
            "account_dataset_scope_note": (
                "信贷账户数据集对应报告正文展示的账户卡片；"
                + ("源报告明确说明信息展示范围受限。" if source_display_limited else "")
                + "附件账户、月度历史、信贷明细及特定交易分别在企业附件数据集中列示。"
            ),
            "source_display_limited": source_display_limited,
            "attachment_account_count": len(attachment_accounts),
            "attachment_credit_detail_count": len(attachment_details),
            "attachment_special_transaction_count": len(attachment_transactions),
        }
    )
    facility_summaries = _facility_summary_lines(parse_result)
    if facility_summaries:
        summary["facility_summary"] = {
            str(line["facility_type"]): {
                key: line.get(key)
                for key in ("total_limit", "used_limit", "available_limit", "currency", "amount_unit")
            }
            for line in facility_summaries
        }
        summary["facility_summary_record_count"] = len(facility_summaries)
    summary.update(
        {
            "extracted_account_count": len(accounts),
            "canonical_table_account_count": len(canonical_accounts),
            "credit_line_count": len(refined.get("credit_lines") or []),
        }
    )
    refined["credit_summary"] = summary
    return refined


def _enterprise_text_fallback(full_text: str, parse_result: Any) -> dict[str, Any]:
    """Extract text-only enterprise records without entering the personal pipeline."""
    text = str(full_text or "")
    compact = re.sub(r"\s+", "", text)
    summary: dict[str, Any] = {}
    overview_match = re.search(
        r"首次有相关还款责任的年份\s*(\d{4})\s+(\d+)\s+(\d+)\s+(\d{4})",
        text,
    )
    if overview_match:
        summary.update(
            {
                "first_credit_year": int(overview_match.group(1)),
                "credit_institution_count": int(overview_match.group(2)),
                "active_credit_institution_count": int(overview_match.group(3)),
                "first_repayment_responsibility_year": int(overview_match.group(4)),
            }
        )
    balance_match = re.search(
        r"借贷交易担保交易余额([0-9,.]+)余额([0-9,.]+)其中[：:]?被追偿余额([0-9,.]+)",
        compact,
    )
    if balance_match:
        summary.update(
            {
                "credit_balance": _number(balance_match.group(1)),
                "guarantee_balance": _number(balance_match.group(2)),
                "recovered_debt_balance": _number(balance_match.group(3)),
            }
        )

    page_texts = _page_texts(parse_result)

    def source_page(value: str) -> int | None:
        needle = _compact(value)
        return next(
            (page for page, page_text in page_texts.items() if needle and needle in _compact(page_text)),
            None,
        )

    accounts: list[dict[str, Any]] = []
    account_matches = list(
        re.finditer(r"未结清账户编号[：:]\s*([0-9A-Z]{6,})", text, flags=re.IGNORECASE)
    )
    for index, match in enumerate(account_matches):
        end = account_matches[index + 1].start() if index + 1 < len(account_matches) else len(text)
        segment = text[match.end() : end]
        account_identifier = _identifier(match.group(1))
        institution_match = re.search(r"授信机构[：:]\s*([^\r\n]+)", segment)
        business_match = re.search(r"业务种类[：:]\s*([^\r\n]+)", segment)
        page = source_page(account_identifier)
        accounts.append(
            {
                "account_id": _stable_id("enterprise_credit_account", account_identifier),
                "account_identifier": account_identifier,
                "account_status": "active",
                "status": "active",
                "institution": _compact(institution_match.group(1)) if institution_match else "",
                "business_type": _compact(business_match.group(1)) if business_match else "",
                "source_page": page,
                "source": "enterprise_native_text_fallback",
                "source_refs": (
                    [{"source": "native_text_enterprise_account", "page": page}]
                    if page is not None
                    else []
                ),
                "confidence": 0.95,
            }
        )

    public_records: list[dict[str, Any]] = []
    public_match = re.search(
        r"公共记录明细(.*?)(?:附件\s*\d*[：:]|信用记录补充信息|\Z)",
        text,
        flags=re.DOTALL,
    )
    public_text = public_match.group(1) if public_match else ""
    date_token = r"(?:19|20)\d{2}-\d{2}-\d{2}"

    def append_public_record(
        *,
        record_type: str,
        authority: str,
        category: str,
        start_date: str,
        end_date: str,
        content: str,
    ) -> None:
        page = source_page(authority)
        public_records.append(
            {
                "public_record_id": _stable_id(
                    "enterprise_public_record",
                    record_type,
                    authority,
                    start_date,
                    end_date,
                    content,
                ),
                "record_type": record_type,
                "authority": authority,
                "category": category,
                "start_date": "" if start_date == "--" else start_date,
                "end_date": "" if end_date == "--" else end_date,
                "content": content,
                "source_page": page,
                "source": "enterprise_native_text_fallback",
                "source_refs": (
                    [{"source": "native_text_enterprise_public_record", "page": page}]
                    if page is not None
                    else []
                ),
                "confidence": 0.9,
            }
        )

    license_match = re.search(
        r"许可部门.*?许可内容(.*?)(?=认证部门|资质部门|\Z)",
        public_text,
        flags=re.DOTALL,
    )
    if license_match:
        license_body = re.sub(r"\s+", "", license_match.group(1))
        license_pattern = re.compile(
            rf"(?P<authority>.+?)(?P<category>普通)(?P<start>{date_token})"
            rf"(?P<end>{date_token})(?P<content>.+?)(?=(?:.+?普通{date_token})|\Z)"
        )
        for match in license_pattern.finditer(license_body):
            authority = re.sub(
                r"^.*?许可(?=[^许可]*(?:省|市|县|区))",
                "",
                match.group("authority"),
            )
            append_public_record(
                record_type="license",
                authority=authority,
                category=match.group("category"),
                start_date=match.group("start"),
                end_date=match.group("end"),
                content=match.group("content"),
            )

    certification_match = re.search(
        r"认证部门.*?认证内容(.*?)(?=资质部门|附件\s*\d*[：:]|\Z)",
        public_text,
        flags=re.DOTALL,
    )
    if certification_match:
        certification_body = re.sub(r"\s+", "", certification_match.group(1))
        certification_pattern = re.compile(
            rf"(?P<authority>.+?)(?P<category>纳税信用A(?:级)?纳税人)"
            rf"(?P<start>{date_token}|--)(?P<end>{date_token}|--)"
            rf"(?P<content>.+?)(?=(?:.+?纳税信用A(?:级)?纳税人(?:{date_token}|--))|\Z)"
        )
        for match in certification_pattern.finditer(certification_body):
            append_public_record(
                record_type="certification",
                authority=match.group("authority"),
                category=match.group("category"),
                start_date=match.group("start"),
                end_date=match.group("end"),
                content=match.group("content"),
            )

    if public_records:
        summary["public_record_count"] = len(public_records)
        counts: dict[str, int] = {}
        for record in public_records:
            record_type = str(record["record_type"])
            counts[record_type] = counts.get(record_type, 0) + 1
        summary["public_record_type_counts"] = counts
    return {
        "credit_accounts": accounts,
        "credit_lines": [],
        "repayment_liability_records": [],
        "repayment_records": [],
        "overdue_records": [],
        "inquiry_records": [],
        "public_records": public_records,
        "credit_summary": summary,
    }


def extract_enterprise_native_business(
    parse_result: Any,
    full_text: str,
) -> dict[str, Any]:
    """Transform a ParseResult into enterprise-native business candidates."""
    candidates = _enterprise_text_fallback(full_text, parse_result)
    return refine_enterprise_business(parse_result, candidates)


__all__ = [
    "EnterpriseExtractionContext",
    "build_enterprise_extraction_context",
    "extract_enterprise_accounts_from_tables",
    "extract_enterprise_credit_lines_from_tables",
    "extract_enterprise_native_business",
    "extract_enterprise_facility_summary",
    "extract_enterprise_identity_facts",
    "extract_enterprise_interest_arrears",
    "extract_enterprise_non_credit_history_datasets",
    "extract_enterprise_overview",
    "extract_enterprise_overview_datasets",
    "extract_enterprise_capital_summary",
    "extract_enterprise_continuation_audit",
    "extract_enterprise_profile_datasets",
    "extract_enterprise_profile_status",
    "extract_enterprise_public_records_from_tables",
    "extract_enterprise_repayment_liability_records",
    "extract_enterprise_report_metadata",
    "extract_enterprise_report_metadata_records",
    "extract_enterprise_report_identity_records",
    "extract_enterprise_report_notes",
    "extract_enterprise_supplement_rows",
    "refine_enterprise_business",
]
