# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical-table refinements for native enterprise credit reports.

Enterprise account cards use stacked header/detail/repayment rows and may
continue across a page boundary.  This module interprets those already-sealed
physical tables without changing or supplementing ``ParseResult``.
"""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.plugins.credit_report.enterprise_native.continuation import (
    CLOSED_SUMMARY_BODY_CONTRACT,
    EnterpriseContinuationResolver,
    FACILITY_VALUE_CONTRACT,
)

_ACCOUNT_CATEGORIES = frozenset({"中长期借款", "短期借款", "循环透支", "贴现"})
_FIVE_TIER_CLASSES = frozenset({"正常", "关注", "次级", "可疑", "损失", "违约", "未分类"})
_CURRENCY_CODES = {
    "人民币": "CNY",
    "人民币元": "CNY",
    "美元": "USD",
    "欧元": "EUR",
    "港币": "HKD",
}


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _number(value: Any) -> int | float | None:
    raw = re.sub(r"[^0-9.-]", "", str(value or "").replace(",", ""))
    if not raw or raw in {"-", ".", "-."}:
        return None
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return None
    return int(number) if number == number.to_integral_value() else float(number)


def _date(value: Any) -> str:
    raw = _compact(value)
    match = re.fullmatch(r"(20\d{2})[-年./](\d{1,2})[-月./](\d{1,2})日?", raw)
    if not match:
        return ""
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _identifier(value: Any) -> str:
    return re.sub(r"[^0-9A-Z]", "", _compact(value).upper())


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(_compact(part).upper() for part in parts)
    return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


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
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", None) or []:
            rows = _raw_table_rows(table)
            if rows:
                yield page_number, str(getattr(table, "table_id", "") or ""), rows


def _table_headings(parse_result: Any) -> dict[str, str]:
    """Return the closest preceding page heading for each physical table."""
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
    values: dict[int, str] = {}
    for index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(
            getattr(page, "source_page_number", 0)
            or getattr(page, "page_number", 0)
            or index
        )
        values[page_number] = "\n".join(
            str(getattr(block, "content", "") or "")
            for block in getattr(page, "texts", None) or []
            if getattr(block, "content", None)
        )
    return values


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
        and len(_identifier(row[0])) >= 10
        and _date(row[open_date_index])
        and _date(row[due_date_index])
        and _number(row[amount_index]) is not None
    )


def _is_account_detail_row(row: list[str]) -> bool:
    return bool(len(row) >= 4 and _number(row[2]) is not None and row[3] in _FIVE_TIER_CLASSES)


def _finalize_account(record: dict[str, Any] | None, out: list[dict[str, Any]]) -> None:
    if not record:
        return
    identifier = _identifier(record.get("account_identifier"))
    if len(identifier) < 12:
        return
    record["account_identifier"] = identifier
    record["account_id"] = f"credit_account:{identifier}"
    record["sequence"] = len(out) + 1
    out.append(record)


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

    for page, table_id, rows in _table_stream(parse_result):
        for row_index, row in enumerate(rows):
            cells = [*row, *([""] * max(0, 8 - len(row)))]
            first = cells[0]
            if first in _ACCOUNT_CATEGORIES and any("共" in cell and "笔" in cell for cell in cells):
                _finalize_account(current, accounts)
                current = None
                category = first
                header_seen = False
                settled_schema = False
                continue
            if _is_account_header(cells):
                _finalize_account(current, accounts)
                current = None
                header_seen = True
                amount_field = "credit_limit" if any("信用额度" in cell for cell in cells) else "loan_amount"
                institution_index = next(
                    (index for index, value in enumerate(cells) if "授信机构" in value),
                    1,
                )
                business_type_index = next(
                    (
                        index
                        for index, value in enumerate(cells)
                        if "业务种类" in value or "业务类型" in value
                    ),
                    2,
                )
                open_date_index = next(
                    (
                        index
                        for index, value in enumerate(cells)
                        if "开立日期" in value or "开户日期" in value
                    ),
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
                    (
                        index
                        for index, value in enumerate(cells)
                        if "借款金额" in value or "信用额度" in value
                    ),
                    6,
                )
                settled_schema = any(
                    "关闭日期" in cell
                    for following in rows[row_index + 1 : row_index + 3]
                    for cell in following
                )
                continue
            if header_seen and _is_primary_account_row(
                cells,
                open_date_index=open_date_index,
                due_date_index=due_date_index,
                amount_index=amount_index,
            ):
                _finalize_account(current, accounts)
                identifier = _identifier(first)
                current = {
                    "account_id": f"credit_account:{identifier}",
                    "account_identifier": identifier,
                    "account_type": "enterprise_credit",
                    "business_category": category,
                    "management_institution": _compact(cells[institution_index]),
                    "business_type": _compact(cells[business_type_index]),
                    "open_date": _date(cells[open_date_index]),
                    "due_date": _date(cells[due_date_index]),
                    "currency": _CURRENCY_CODES.get(
                        _compact(cells[currency_index]),
                        _compact(cells[currency_index]),
                    ),
                    amount_field: _number(cells[amount_index]),
                    "amount_unit": "CNY_10K",
                    "account_status": "settled" if settled_schema else "active",
                    "source": "canonical_enterprise_account_card",
                    "source_refs": [_source_ref(page, table_id, row_index)],
                    "confidence": 1.0,
                }
                continue
            if current and settled_schema and len(cells) >= 8 and _date(cells[1]):
                current["close_date"] = _date(cells[1])
                if cells[2] in _FIVE_TIER_CLASSES:
                    current["five_tier_class"] = cells[2]
                current["last_repayment_date"] = _date(cells[3])
                current["repayment_method"] = _compact(cells[5])
                current["history_status"] = _compact(cells[7])
                current["payoff_state"] = "settled"
                current["account_state"] = "closed"
                _append_ref(current, page, table_id, row_index)
                _finalize_account(current, accounts)
                current = None
                continue
            if current and _is_account_detail_row(cells):
                suffix = _identifier(first)
                if suffix:
                    current["account_identifier"] = f"{_identifier(current.get('account_identifier'))}{suffix}"
                current["guarantee_type"] = _compact(cells[1])
                current["balance"] = _number(cells[2])
                current["five_tier_class"] = cells[3]
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
                    current["current_overdue_status"] = (
                        "overdue" if current["current_overdue"] else "not_overdue"
                    )
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
            if len(values) < 6:
                match = resolver.following_row(fragment, FACILITY_VALUE_CONTRACT)
                if match is not None:
                    value_page = match.fragment.page
                    value_table_id = match.fragment.table_id
                    value_row_index = match.row_index
                    values = [_compact(value) for value in match.row]
            if len(values) < 6:
                continue
            numbers = [_number(value) for value in values[:6]]
            if any(value is None for value in numbers):
                continue
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
            if not (
                header
                and "授信协议编号" in header[0]
                and "授信额度" in subheader
                and "已用额度" in subheader
            ):
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
                            "limit_identifier": (
                                _identifier(amounts[5]) if len(amounts) > 5 else ""
                            ),
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
    "工商注册号": "business_registration_number",
    "纳税人识别号（国税）": "national_tax_id",
    "纳税人识别号（地税）": "local_tax_id",
}


def extract_enterprise_identity_facts(parse_result: Any) -> dict[str, Any]:
    """Read all enterprise identity codes from the canonical identity table."""
    facts: dict[str, Any] = {}
    for _page, _table_id, rows in _table_stream(parse_result):
        labels = {row[0] for row in rows if len(row) >= 2}
        if not {"企业名称", "中征码", "统一社会信用代码"} <= labels:
            continue
        for row in rows:
            if len(row) < 2:
                continue
            field = _IDENTITY_LABELS.get(row[0])
            if field and row[1] and row[1] != "--":
                facts[field] = row[1]
        break
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
        if len(rows) >= 2 and all(
            any(label in header for header in headers) for label in overview_labels
        ):
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
        if len(rows) >= 2 and all(
            any(label in header for header in headers) for label in public_labels
        ):
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
                    target = (
                        "credit_attention_balance"
                        if index < 2
                        else "guarantee_attention_balance"
                    )
                    summary[target] = number
                elif "不良类余额" in value:
                    target = (
                        "credit_adverse_balance"
                        if index < 2
                        else "guarantee_adverse_balance"
                    )
                    summary[target] = number
    return summary


def extract_enterprise_summary_datasets(
    parse_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Project the page-level responsibility and closed-credit summaries."""
    current_rows: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    responsibility_rows: list[dict[str, Any]] = []
    resolver = EnterpriseContinuationResolver(parse_result)
    for fragment in resolver.fragments:
        page = fragment.page
        table_id = fragment.table_id
        rows = [[_compact(value) for value in row] for row in fragment.rows]
        if not rows:
            continue
        headers = rows[0]
        signature = "".join(headers)
        if (
            len(rows) >= 3
            and len(headers) >= 9
            and all(marker in signature for marker in ("正常类", "关注类", "不良类", "合计"))
        ):
            subheaders = rows[1]
            if (
                len(subheaders) >= 9
                and subheaders[1:9]
                == ["账户数", "余额", "账户数", "余额", "账户数", "余额", "账户数", "余额"]
            ):
                categories = {_compact(row[0]) for row in rows[2:] if row}
                if categories & {"中长期借款", "短期借款", "循环透支", "贴现"}:
                    transaction_group = "借贷交易"
                elif categories & {"银行承兑汇票", "信用证"}:
                    transaction_group = "担保交易"
                elif categories & {"银行保函", "其他担保交易"}:
                    transaction_group = "银行保函及其他业务"
                else:
                    transaction_group = ""
                if transaction_group:
                    for row_index, row in enumerate(rows[2:], start=2):
                        if len(row) < 9 or not _compact(row[0]):
                            continue
                        values = [_number(value) for value in row[1:9]]
                        if any(value is None for value in values):
                            continue
                        category = _compact(row[0])
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
                                "normal_account_count": int(values[0]),
                                "normal_balance": values[1],
                                "attention_account_count": int(values[2]),
                                "attention_balance": values[3],
                                "adverse_account_count": int(values[4]),
                                "adverse_balance": values[5],
                                "total_account_count": int(values[6]),
                                "total_balance": values[7],
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
        if (
            "正常类账户数" in signature
            and "关注类账户数" in signature
            and "不良类账户数" in signature
            and "合计" in signature
        ):
            closed_data: list[tuple[int, str, int, list[str]]] = [
                (page, table_id, row_index, row)
                for row_index, row in enumerate(rows[1:], start=1)
            ]
            if not closed_data:
                match = resolver.following_row(
                    fragment,
                    CLOSED_SUMMARY_BODY_CONTRACT,
                )
                if match is not None:
                    candidate_rows = [
                        [_compact(value) for value in row]
                        for row in match.fragment.rows
                    ]
                    if all(
                        len(row) == 5
                        and _compact(row[0])
                        and all(_number(value) is not None for value in row[1:5])
                        for row in candidate_rows
                    ):
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
            if categories & {"中长期借款", "短期借款", "贴现"}:
                transaction_group = "借贷交易"
            elif "银行承兑汇票" in categories:
                transaction_group = "银行承兑汇票和信用证"
            elif "其他担保交易" in categories:
                transaction_group = "银行保函及其他业务"
            else:
                transaction_group = "其他"
            for source_page, source_table_id, row_index, row in closed_data:
                if len(row) < 5 or not _compact(row[0]):
                    continue
                counts = [_number(value) for value in row[1:5]]
                if any(value is None for value in counts):
                    continue
                category = _compact(row[0])
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
        responsibility_signature = "".join(
            value
            for row in rows[:2]
            for value in row
        )
        grouped_responsibility_header = (
            "被追偿业务" in responsibility_signature
            and "其他借贷交易" in responsibility_signature
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
                        "recovered_account_count": (
                            int(values[1]) if values[1] is not None else None
                        ),
                        "recovered_balance": values[2],
                        "other_credit_responsibility_amount": values[3],
                        "other_credit_account_count": (
                            int(values[4]) if values[4] is not None else None
                        ),
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
                        "guarantee_account_count": (
                            int(values[1]) if values[1] is not None else None
                        ),
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
                    (
                        pending.get("_fragment_index") == fragment.index
                        and pending.get("_row_index") == row_index - 1
                    )
                    or (
                        pending.get("_fragment_index") == fragment.index - 1
                        and row_index == 0
                        and page
                        - int(pending.get("source_page") or page)
                        in {0, 1}
                    )
                )
            )
            if (
                pending is not None
                and is_immediate_continuation
                and len(row) >= 9
                and not _compact(row[0])
            ):
                loan_or_credit_amount = _number(row[1])
                balance = _number(row[2])
                snapshot_date = _date(row[8])
                if (
                    loan_or_credit_amount is not None
                    and balance is not None
                    and snapshot_date
                ):
                    overdue_value = _number(row[6])
                    pending.update(
                        {
                            "loan_or_credit_amount": loan_or_credit_amount,
                            "balance": balance,
                            "five_tier_class": (
                                "" if _compact(row[3]) in {"--", "-", "—"} else _compact(row[3])
                            ),
                            "overdue_total": _number(row[4]),
                            "overdue_principal": _number(row[5]),
                            "overdue_months_or_repayment_status": (
                                overdue_value
                                if overdue_value is not None
                                else (
                                    ""
                                    if _compact(row[6]) in {"--", "-", "—"}
                                    else _compact(row[6])
                                )
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
                or _compact(row[3]) not in _CURRENCY_CODES
            ):
                continue
            finish_pending()
            currency = _CURRENCY_CODES.get(_compact(row[3]), _compact(row[3]))
            pending = {
                "liability_id": f"repayment_liability:{account_identifier}",
                "account_identifier": account_identifier,
                "responsibility_type": _compact(row[1]),
                "contract_number": contract_number,
                "contract_number_status": (
                    "reported" if contract_number else "not_reported"
                ),
                "currency": currency,
                "amount_unit": "CNY_10K",
                "responsibility_amount": responsibility_amount,
                "responsibility_amount_reported": responsibility_amount is not None,
                "responsibility_amount_status": (
                    "reported" if responsibility_amount is not None else "not_reported"
                ),
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
    resolved = datasets or {}
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
        signature = "".join(rows[0])
        first_two = "".join(value for row in rows[:2] for value in row)
        if (
            len(rows) >= 3
            and len(rows[0]) >= 9
            and all(marker in signature for marker in ("正常类", "关注类", "不良类", "合计"))
            and len(rows[1]) >= 9
            and rows[1][1:9]
            == ["账户数", "余额", "账户数", "余额", "账户数", "余额", "账户数", "余额"]
        ):
            categories = {_compact(row[0]) for row in rows[2:] if row}
            if categories & {
                "中长期借款",
                "短期借款",
                "循环透支",
                "贴现",
                "银行承兑汇票",
                "信用证",
                "银行保函",
                "其他担保交易",
            }:
                expected["current_credit_summary"] += sum(
                    1
                    for row in rows[2:]
                    if len(row) >= 9
                    and _compact(row[0])
                    and all(_number(value) is not None for value in row[1:9])
                )
        if (
            "正常类账户数" in signature
            and "关注类账户数" in signature
            and "不良类账户数" in signature
            and "合计" in signature
        ):
            body = rows[1:]
            if not body:
                match = resolver.following_row(fragment, CLOSED_SUMMARY_BODY_CONTRACT)
                body = (
                    [[_compact(value) for value in row] for row in match.fragment.rows]
                    if match is not None
                    else []
                )
            expected["closed_credit_summary"] += sum(
                1
                for row in body
                if len(row) == 5
                and _compact(row[0])
                and all(_number(value) is not None for value in row[1:5])
            )
        if (
            len(rows[0]) >= 9
            and "还款责任金额" in first_two
            and "账户数" in first_two
            and "余额" in first_two
            and (
                ("被追偿业务" in first_two and "其他借贷交易" in first_two)
                or first_two.count("还款责任金额") >= 2
            )
        ):
            start = 2 if len(rows) > 1 and "还款责任金额" in "".join(rows[1]) else 1
            expected["repayment_responsibility_summary"] += sum(
                1 for row in rows[start:] if len(row) >= 9 and _compact(row[0])
            )
        elif (
            len(rows[0]) >= 6
            and "责任类型" in first_two
            and "担保交易" in first_two
            and "还款责任金额" in first_two
        ):
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
        resolved["repayment_liability_records"] = (
            extract_enterprise_repayment_liability_records(parse_result)
        )
    if "enterprise_attachment_accounts" not in resolved:
        resolved.update(extract_enterprise_attachment_datasets(parse_result))
    actual = {
        "current_credit_summary": len(
            resolved.get("enterprise_current_credit_summary") or []
        ),
        "closed_credit_summary": len(
            resolved.get("enterprise_closed_credit_summary") or []
        ),
        "repayment_responsibility_summary": len(
            resolved.get("enterprise_repayment_responsibility_summary") or []
        ),
        "repayment_liability": len(
            resolved.get("repayment_liability_records") or []
        ),
        "attachment_account": len(
            resolved.get("enterprise_attachment_accounts") or []
        ),
    }
    audits: list[dict[str, Any]] = []
    for sequence, family in enumerate(expected, start=1):
        unresolved = max(0, expected[family] - actual[family])
        audits.append(
            {
                "audit_id": f"enterprise_continuation_audit:{family}",
                "sequence": sequence,
                "continuation_family": family,
                "expected_record_count": expected[family],
                "extracted_record_count": actual[family],
                "unresolved_record_count": unresolved,
                "reconciliation_status": (
                    "complete"
                    if expected[family] == actual[family]
                    else "unresolved"
                ),
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
        data_rows = [
            row
            for row in rows[1:]
            if row and not row[0].startswith("信息来源")
        ]
        contributor_status = (
            "no_records"
            if data_rows and all(_placeholder_row(row) for row in data_rows)
            else "reported"
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
            "contributor_count": sum(
                1 for row in data_rows if not _placeholder_row(row)
            ),
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
            (
                page_number
                for page_number, text in page_texts.items()
                if "自主查询版" in _compact(text)
            ),
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
            len(row) >= 4
            and row[2] in {"身份证", "护照", "港澳居民来往内地通行证"}
            and len(_identifier(row[3])) >= 8
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
        relationship_headers = (
            all(marker in signature for marker in ("类型", "名称", "身份标识号码"))
            or all(marker in signature for marker in ("名称", "身份标识类型", "身份标识号码"))
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


def _page_flow(parse_result: Any):
    """Yield page text and tables in visual order, preserving continuations."""
    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(
            getattr(page, "source_page_number", 0)
            or getattr(page, "page_number", 0)
            or page_index
        )
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
    return (
        "信息报告日期" in signature
        and "余额" in signature
        and "余额变化日期" in signature
    )


def _history_continuation(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    first = rows[0]
    dense = [_compact(value) for value in first if _compact(value)]
    return bool(
        dense
        and (
            _date(first[0] if first else "")
            or (not _compact(first[0] if first else "") and _number(dense[0]) is not None)
        )
        and 6 <= len(dense) <= 8
    )


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
    pending_detail_header: tuple[list[str], dict[str, Any]] | None = None
    resolver = EnterpriseContinuationResolver(parse_result)
    allowed_detail_continuation_tables: set[str] = set()
    for fragment in resolver.fragments:
        fragment_rows = [
            [_compact(value) for value in row] for row in fragment.rows
        ]
        header_index = next(
            (
                index
                for index, row in enumerate(fragment_rows)
                if ("开户日期" in row or "开立日期" in row)
                and any(
                    marker in "".join(row)
                    for marker in ("贴现金额", "借款金额", "信用额度", "金额")
                )
            ),
            -1,
        )
        if header_index < 0:
            continue
        header = fragment_rows[header_index]
        open_date_index = next(
            (
                index
                for index, value in enumerate(header)
                if "开户日期" in value or "开立日期" in value
            ),
            -1,
        )
        amount_index = next(
            (
                index
                for index, value in enumerate(header)
                if any(
                    marker in value
                    for marker in ("贴现金额", "借款金额", "信用额度", "金额")
                )
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

        if any(valid_detail_row(row) for row in fragment_rows[header_index + 1 :]):
            continue
        if fragment.index + 1 >= len(resolver.fragments):
            continue
        candidate = resolver.fragments[fragment.index + 1]
        if candidate.page - fragment.page not in {0, 1}:
            continue
        candidate_rows = [
            [_compact(value) for value in row] for row in candidate.rows
        ]
        candidate_starts_new_table = any(
            ("开户日期" in row or "开立日期" in row)
            and any(
                marker in "".join(row)
                for marker in ("贴现金额", "借款金额", "信用额度", "金额")
            )
            for row in candidate_rows
        )
        if (
            not candidate_starts_new_table
            and any(valid_detail_row(row) for row in candidate_rows)
        ):
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
                "five_tier_class": (
                    "" if primary_values[3] == "--" else primary_values[3]
                ),
                "classification_date": _date(primary_values[4]),
                "overdue_total": _number(primary_values[5]),
                "overdue_principal": _number(primary_values[6]),
                "overdue_months": _number(
                    primary_values[7] if revolving else secondary_values[0]
                ),
                "scheduled_repayment_date": _date(
                    secondary_values[secondary_offset]
                ),
                "scheduled_repayment_amount": _number(
                    secondary_values[secondary_offset + 1]
                ),
                "actual_repayment_date": _date(
                    secondary_values[secondary_offset + 2]
                ),
                "actual_repayment_amount": _number(
                    secondary_values[secondary_offset + 3]
                ),
                "repayment_method": (
                    ""
                    if secondary_values[secondary_offset + 4] == "--"
                    else secondary_values[secondary_offset + 4]
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
            if (
                pending_history
                and pending_history[1] is context
                and values
                and not values[0]
            ):
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
    ) -> bool:
        nonlocal pending_detail_header
        header_index = next(
            (
                index
                for index, row in enumerate(rows)
                if any(value in {"开户日期", "开立日期"} for value in row)
                and any(
                    marker in value
                    for value in row
                    for marker in ("贴现金额", "金额", "借款金额", "信用额度")
                )
            ),
            -1,
        )
        if header_index >= 0:
            header = rows[header_index]
            data_rows = list(
                enumerate(rows[header_index + 1 :], start=header_index + 1)
            )
        elif pending_detail_header and pending_detail_header[1] is context:
            header = pending_detail_header[0]
            data_rows = list(enumerate(rows))
        else:
            return False

        def index_of(*markers: str) -> int:
            return next(
                (
                    index
                    for index, value in enumerate(header)
                    if any(marker in value for marker in markers)
                ),
                -1,
            )

        indexes = {
            "account_identifier": index_of("账户编号"),
            "open_date": index_of("开户日期", "开立日期"),
            "due_date": index_of("到期日"),
            "currency": index_of("币种"),
            "amount": index_of("贴现金额", "借款金额", "信用额度", "金额"),
            "close_date": index_of("关闭日期"),
            "five_tier_class": index_of("五级分类"),
            "last_repayment_date": index_of("最后一次还款日期"),
            "repayment_method": index_of("最后一次还款形式"),
            "advance_flag": index_of("垫款标志"),
        }
        emitted = 0
        for row_index, row in data_rows:
            if not any(_compact(value) for value in row):
                continue

            def value(field: str) -> str:
                index = indexes[field]
                return _compact(row[index]) if 0 <= index < len(row) else ""

            amount = _number(value("amount"))
            open_date = _date(value("open_date"))
            if amount is None or not open_date:
                continue
            row_identifier = _identifier(value("account_identifier"))
            identifier = row_identifier or str(context.get("account_identifier") or "")
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
                    "five_tier_class": (
                        "" if value("five_tier_class") == "--" else value("five_tier_class")
                    ),
                    "last_repayment_date": _date(value("last_repayment_date")),
                    "repayment_method": (
                        "" if value("repayment_method") == "--" else value("repayment_method")
                    ),
                    "advance_flag": (
                        "" if value("advance_flag") == "--" else value("advance_flag")
                    ),
                    "source_page": page,
                    "source_table_id": table_id,
                    "source": "canonical_enterprise_attachment_credit_detail",
                    "source_refs": [_source_ref(page, table_id, row_index)],
                    "confidence": 1.0,
                }
            )
            emitted += 1
        if emitted:
            pending_detail_header = None
        elif header_index >= 0:
            pending_detail_header = (header, context)
        return True

    def consume_special_transactions(
        rows: list[list[str]],
        context: dict[str, Any],
        page: int,
        table_id: str,
    ) -> bool:
        header_index = next(
            (
                index
                for index, row in enumerate(rows)
                if "交易类型" in row and "交易日期" in row
            ),
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
                    "transaction_detail": (
                        ""
                        if value("transaction_detail") == "--"
                        else value("transaction_detail")
                    ),
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
            if (
                text == "附件"
                or re.search(r"附件\d*[:：]?信用记录补充信息", text)
            ):
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
            if consume_credit_details(rows, current, page, table_id):
                continue
        if _history_table(rows) or _history_continuation(rows):
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
    return extract_enterprise_attachment_datasets(parse_result)[
        "enterprise_credit_supplement"
    ]


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
    fragment_index_by_table_id = {
        fragment.table_id: fragment.index for fragment in resolver.fragments
    }
    for page, table_id, rows in _table_stream(parse_result):
        if not rows:
            continue
        headers = rows[0]
        record_type = _public_type(headers, rows)
        if not record_type:
            continue
        structured_table = any(marker in "".join(headers) for marker in _PUBLIC_TABLE_TYPES)
        key_value_table = not structured_table and any(
            "：" in cell or ":" in cell for row in rows for cell in row
        )
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
                "page": page,
                "source_table_id": table_id,
                "source": "canonical_enterprise_public_record_table",
                "source_refs": [_source_ref(page, table_id, row_index)],
                "confidence": 1.0,
            }
            rules = _PUBLIC_CONTINUATION_RULES.get(record_type)
            previous = records[-1] if records else None
            previous_details = (
                previous.get("details")
                if isinstance(previous, dict) and isinstance(previous.get("details"), dict)
                else {}
            )
            detail_labels = frozenset(str(key) for key in details)
            previous_labels = frozenset(str(key) for key in previous_details)
            previous_table_id = str(
                previous.get("source_table_id") if isinstance(previous, dict) else ""
            )
            adjacent_physical_table = (
                previous_table_id in fragment_index_by_table_id
                and table_id in fragment_index_by_table_id
                and fragment_index_by_table_id[table_id]
                == fragment_index_by_table_id[previous_table_id] + 1
            )
            guarded_continuation = bool(
                rules
                and previous
                and previous.get("record_type") == record_type
                and page - int(previous.get("page") or page) in {0, 1}
                and adjacent_physical_table
                and previous_labels & rules["start"]
                and not detail_labels & rules["start"]
                and detail_labels & rules["continuation"]
            )
            if guarded_continuation:
                previous = records[-1]
                previous["details"].update(details)
                previous["content"] = "；".join(
                    f"{key}：{value}" for key, value in previous["details"].items()
                )
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
    # Canonical account cards and heuristic history/appendix candidates have
    # different grains.  Once cards exist, mixing the two populations creates
    # duplicates and field-shifted pseudo-accounts.
    return canonical


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
    summary["account_population_comparable"] = bool(
        isinstance(expected, int) and expected == len(canonical_accounts)
    )
    attachment_datasets = extract_enterprise_attachment_datasets(parse_result)
    attachment_accounts = attachment_datasets["enterprise_attachment_accounts"]
    attachment_details = attachment_datasets["enterprise_attachment_credit_details"]
    attachment_transactions = attachment_datasets["enterprise_special_transactions"]
    source_display_limited = any(
        "受篇幅所限" in text and "只展示部分信贷记录" in text
        for text in _page_texts(parse_result).values()
    )
    summary.update(
        {
            "account_dataset_scope": "main_report_account_cards",
            "account_dataset_scope_note": (
                "信贷账户数据集对应报告正文展示的账户卡片；"
                + (
                    "源报告明确说明受篇幅限制仅展示部分信贷记录。"
                    if source_display_limited
                    else ""
                )
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


def extract_enterprise_native_business(
    parse_result: Any,
    full_text: str,
) -> dict[str, Any]:
    """Transform a ParseResult into enterprise-native business candidates."""
    from docmirror.plugins.credit_report.business_records import (
        extract_enterprise_native_candidates,
    )

    candidates = extract_enterprise_native_candidates(parse_result, full_text)
    return refine_enterprise_business(parse_result, candidates)


__all__ = [
    "extract_enterprise_accounts_from_tables",
    "extract_enterprise_credit_lines_from_tables",
    "extract_enterprise_native_business",
    "extract_enterprise_facility_summary",
    "extract_enterprise_identity_facts",
    "extract_enterprise_overview",
    "extract_enterprise_capital_summary",
    "extract_enterprise_continuation_audit",
    "extract_enterprise_profile_datasets",
    "extract_enterprise_profile_status",
    "extract_enterprise_public_records_from_tables",
    "extract_enterprise_repayment_liability_records",
    "extract_enterprise_report_metadata",
    "extract_enterprise_report_metadata_records",
    "extract_enterprise_report_notes",
    "extract_enterprise_supplement_rows",
    "refine_enterprise_business",
]
