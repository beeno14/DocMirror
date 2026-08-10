# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-facing business records for public credit-report variants.

Native personal brief reports express accounts as prose and inquiries as a
borderless ledger. Scanned personal detail reports continue to use the
geometry-aware local-structure and repayment-grid projectors; this module only
derives their overdue view from already-projected records.
"""

from __future__ import annotations

import re
from typing import Any

from docmirror.plugins.credit_report.currency_codes import normalize_currency_code
from docmirror.plugins.credit_report.value_utils import (
    compact_text as _compact,
)
from docmirror.plugins.credit_report.value_utils import (
    linear_text as _linear,
)
from docmirror.plugins.credit_report.value_utils import (
    parse_number as _number,
)
from docmirror.plugins.credit_report.value_utils import (
    stable_record_id as _stable_id,
)

_DATE_CN_RE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_ACCOUNT_DATE_PATTERN = r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
_ACCOUNT_ACTION_PATTERN = r"(?:发放(?:了)?(?:的)?|办理(?:了)?(?:的)?|开立(?:了)?(?:的)?|提供(?:了)?(?:的)?|为(?=.{0,40}贷款授信))"
_ACCOUNT_START_RE = re.compile(
    rf"{_ACCOUNT_DATE_PATTERN}"
    rf"(?=(?:(?!{_ACCOUNT_DATE_PATTERN}).){{4,180}}?{_ACCOUNT_ACTION_PATTERN})"
)
_PERSONAL_BRIEF_CARD_TYPE_RE = re.compile(r"^(?P<business_type>准贷记卡|贷记卡)(?=[（(，,。；;账户]|$)")
_INQUIRY_REASONS = tuple(
    sorted(
        {
            "法人代表、负责人、高管等资信审查",
            "本人查询（互联网个人信用信息服务平台）",
            "本人查询(互联网个人信用信息服务平台)",
            "本人查询（商业银行网上银行）",
            "本人查询(商业银行网上银行)",
            "本人查询（自助查询机）",
            "本人查询(自助查询机)",
            "担保资格审查",
            "保前审查",
            "资信审查",
            "融资审批",
            "信用卡审批",
            "贷款审批",
            "贷后管理",
            "本人查询（临柜）",
            "本人查询(临柜)",
        },
        key=len,
        reverse=True,
    )
)


def _iso_date(value: str) -> str:
    match = _DATE_CN_RE.search(str(value or ""))
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = re.search(r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})", str(value or ""))
    if not match:
        return ""
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _iso_month(value: str) -> str:
    match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", str(value or ""))
    if not match:
        return ""
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"


def _page_texts(parse_result: Any) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for index, page in enumerate(getattr(parse_result, "pages", []) or [], start=1):
        parts = [str(getattr(block, "content", "") or "") for block in getattr(page, "texts", []) or []]
        for kv in getattr(page, "key_values", []) or []:
            parts.extend((str(getattr(kv, "key", "") or ""), str(getattr(kv, "value", "") or "")))
        for table in getattr(page, "tables", []) or []:
            parts.extend(str(header or "") for header in getattr(table, "headers", []) or [])
            for row in getattr(table, "rows", []) or []:
                parts.extend(str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", []) or [])
        text = "\n".join(part for part in parts if part.strip())
        out.append((int(getattr(page, "page_number", 0) or index), _linear(text), _compact(text)))
    return out


def _source_page(page_texts: list[tuple[int, str, str]], fragment: str) -> int:
    needle = _compact(fragment)[:36]
    if not needle:
        return 0
    for page_number, _linear_text, compact_text in page_texts:
        if needle in compact_text:
            return page_number
    return 0


def _source_refs(page: int, method: str) -> list[dict[str, Any]]:
    ref: dict[str, Any] = {"source": method}
    if page > 0:
        ref["page"] = page
    return [ref]


def _cell_texts(row: Any) -> list[str]:
    return [str(getattr(cell, "text", "") or "").strip() for cell in (getattr(row, "cells", None) or [])]


def _reported_count(value: str) -> int | None:
    compact = _compact(value)
    if not compact or compact == "--":
        return None
    number = _number(value)
    return int(number) if isinstance(number, int | float) else None


def _personal_brief_summary_from_canonical_tables(
    parse_result: Any,
    fallback_text: str = "",
    *,
    expected_account_count: int | None = None,
) -> dict[str, Any]:
    """Read source-defined personal-brief counts without changing ``--`` to zero."""
    row_names = {
        "账户数": "source_account_counts",
        "未结清/未销户账户数": "source_unclosed_account_counts",
        "发生过逾期的账户数": "source_overdue_account_counts",
        "发生过90天以上逾期的账户数": "source_over_90_days_account_counts",
    }
    column_names = ("credit_card", "housing_loan", "other_loan", "other_business")
    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "page_number", 0) or page_index)
        for table in getattr(page, "tables", None) or []:
            extracted: dict[str, Any] = {}
            for row in getattr(table, "rows", None) or []:
                cells = _cell_texts(row)
                if not cells:
                    continue
                row_key = row_names.get(_compact(cells[0]))
                if not row_key:
                    continue
                values = [_reported_count(cells[index]) if index < len(cells) else None for index in range(1, 5)]
                extracted[row_key] = dict(zip(column_names, values, strict=True))
            if "source_unclosed_account_counts" not in extracted:
                continue
            unclosed = extracted["source_unclosed_account_counts"]
            account_counts = extracted.get("source_account_counts", {})
            overdue = extracted.get("source_overdue_account_counts", {})
            over_90 = extracted.get("source_over_90_days_account_counts", {})
            summary = {
                **extracted,
                "source_account_count": (
                    sum(value for value in account_counts.values() if value is not None)
                    if any(value is not None for value in account_counts.values())
                    else None
                ),
                "source_unclosed_account_count": sum(value for value in unclosed.values() if value is not None),
                "source_overdue_account_count": (
                    sum(value for value in overdue.values() if value is not None)
                    if any(value is not None for value in overdue.values())
                    else None
                ),
                "source_overdue_account_count_status": (
                    "reported" if any(value is not None for value in overdue.values()) else "not_reported"
                ),
                "source_over_90_days_account_count": (
                    sum(value for value in over_90.values() if value is not None)
                    if any(value is not None for value in over_90.values())
                    else None
                ),
                "source_over_90_days_account_count_status": (
                    "reported" if any(value is not None for value in over_90.values()) else "not_reported"
                ),
                "source_summary_table_id": str(getattr(table, "table_id", "") or ""),
                "source_summary_page": page_number,
            }
            for candidate in getattr(page, "tables", None) or []:
                raw_rows = (getattr(candidate, "metadata", None) or {}).get("raw_rows") or []
                for row in raw_rows:
                    cells = [_compact(cell) for cell in row]
                    if cells[:1] == ["账户数"] and len(cells) >= 3:
                        headers = [_compact(value) for value in (getattr(candidate, "headers", None) or [])]
                        if "资产处置信息" in headers and "垫款信息" in headers:
                            summary["source_asset_disposition_count"] = _reported_count(cells[1])
                            summary["source_guarantor_compensation_count"] = _reported_count(cells[2])
                    if cells[:1] == ["相关还款责任账户数"] and len(cells) >= 3:
                        summary["source_personal_liability_count"] = _reported_count(cells[1])
                        summary["source_enterprise_liability_count"] = _reported_count(cells[2])
            if (
                expected_account_count is None
                or summary.get("source_account_count") == expected_account_count
            ):
                return summary
    text = _linear(fallback_text)
    if not text:
        entity_context = getattr(parse_result, "entity_context", None)
        ordered_text_blocks = getattr(entity_context, "ordered_text_blocks", None)
        if callable(ordered_text_blocks):
            text = _linear(
                "\n".join(
                    str(content or "")
                    for _page, content in ordered_text_blocks()
                    if str(content or "").strip()
                )
            )
    row_candidates: dict[str, list[tuple[int, dict[str, int | None]]]] = {}
    for source_label, row_key in row_names.items():
        for match in re.finditer(re.escape(source_label), text):
            values_match = re.match(
                r"\s*(--|\d+)\s+(--|\d+)\s+(--|\d+)\s+(--|\d+)(?=\s|$)",
                text[match.end() :],
            )
            if not values_match:
                continue
            values = {
                name: _reported_count(value)
                for name, value in zip(column_names, values_match.groups(), strict=True)
            }
            row_candidates.setdefault(row_key, []).append((match.start(), values))
    account_candidates = row_candidates.get("source_account_counts") or []
    if expected_account_count is not None:
        matching_accounts = [
            candidate
            for candidate in account_candidates
            if sum(value for value in candidate[1].values() if value is not None)
            == expected_account_count
        ]
        if matching_accounts:
            account_candidates = matching_accounts
    if not account_candidates:
        return {}
    account_position, account_counts = account_candidates[0]
    if (
        expected_account_count is not None
        and sum(value for value in account_counts.values() if value is not None)
        != expected_account_count
    ):
        return {}
    extracted: dict[str, Any] = {"source_account_counts": account_counts}
    for row_key, candidates in row_candidates.items():
        if row_key == "source_account_counts" or not candidates:
            continue
        _position, values = min(candidates, key=lambda candidate: abs(candidate[0] - account_position))
        extracted[row_key] = values
    if "source_unclosed_account_counts" not in extracted:
        return {}
    unclosed = extracted["source_unclosed_account_counts"]
    account_counts = extracted.get("source_account_counts", {})
    overdue = extracted.get("source_overdue_account_counts", {})
    over_90 = extracted.get("source_over_90_days_account_counts", {})
    page_texts = _page_texts(parse_result)
    return {
        **extracted,
        "source_account_count": (
            sum(value for value in account_counts.values() if value is not None)
            if any(value is not None for value in account_counts.values())
            else None
        ),
        "source_unclosed_account_count": sum(value for value in unclosed.values() if value is not None),
        "source_overdue_account_count": (
            sum(value for value in overdue.values() if value is not None)
            if any(value is not None for value in overdue.values())
            else None
        ),
        "source_overdue_account_count_status": (
            "reported" if any(value is not None for value in overdue.values()) else "not_reported"
        ),
        "source_over_90_days_account_count": (
            sum(value for value in over_90.values() if value is not None)
            if any(value is not None for value in over_90.values())
            else None
        ),
        "source_over_90_days_account_count_status": (
            "reported" if any(value is not None for value in over_90.values()) else "not_reported"
        ),
        "source_summary_table_id": "",
        "source_summary_page": _source_page(page_texts, "未结清/未销户账户数") or 1,
        "source_summary_extraction_method": "ordered_text_fallback",
    }


def _personal_brief_text(parse_result: Any, full_text: str) -> str:
    """Prefer the variant's geometry-aware reading order over serializer text."""
    entity_context = getattr(parse_result, "entity_context", None)
    ordered_text_blocks = getattr(entity_context, "ordered_text_blocks", None)
    if callable(ordered_text_blocks):
        ordered = [content for _page, content in ordered_text_blocks() if str(content or "").strip()]
        if ordered:
            return _linear("\n".join(ordered))
    return _linear(full_text)


def extract_native_credit_business(
    parse_result: Any,
    full_text: str,
    *,
    report_subtype: str,
    content_mode: str,
) -> dict[str, Any]:
    """Compatibility entry point delegated to the resolved document variant."""
    from docmirror.plugins.credit_report.variant_router import (
        resolve_credit_report_variant,
    )

    variant = resolve_credit_report_variant(report_subtype, content_mode)
    return variant.extract_native_business(
        parse_result,
        full_text,
        content_mode=content_mode,
    )


def _personal_brief_credit_lines(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for account in accounts:
        if account.get("account_type") != "credit_line":
            continue
        account_id = str(account.get("account_id") or "")
        out.append(
            {
                "credit_line_id": _stable_id("credit_line", account_id),
                "account_id": account_id,
                "account_identifier": account.get("account_identifier"),
                "management_institution": account.get("management_institution"),
                "facility_type": account.get("business_type") or "贷款授信",
                "total_limit": account.get("credit_limit"),
                "used_limit": account.get("balance"),
                "currency": account.get("currency") or "CNY",
                "account_currency": account.get("account_currency") or account.get("currency") or "CNY",
                "reporting_amount_currency": account.get("reporting_amount_currency") or "CNY",
                "amount_unit": account.get("amount_unit") or "yuan",
                "reporting_amount_unit": account.get("reporting_amount_unit") or "yuan",
                "validity_type": account.get("credit_line_validity_type"),
                "expiry_date": account.get("credit_line_expiry_date"),
                "account_status": account.get("account_status"),
                "source": "personal_brief_narrative",
                "source_refs": list(account.get("source_refs") or []),
                "confidence": account.get("confidence", 0.9),
            }
        )
    return out


def _personal_brief_overdue_section_state(text: str, account_start: int) -> bool | None:
    """Apply only an explicit, still-open account-group overdue heading."""
    prefix = text[:account_start]
    states = list(
        re.finditer(
            r"(?P<never>从未发生过逾期|从未逾期过)|(?P<ever>发生过逾期)"
            r".{0,80}?(?:账户|卡片|贷款).{0,30}?(?:明细|如下)",
            prefix,
        )
    )
    if not states:
        return None
    state = states[-1]
    intervening = prefix[state.end() :]
    # A new business/record section closes the group.  Numbered account rows,
    # page footers and repeated group titles do not.
    if re.search(
        r"(?:相关还款责任信息|非信贷交易记录|公共记录|查询记录|资产处置信息|保证人代偿信息)",
        intervening,
    ):
        return None
    return state.group("never") is None


def _personal_brief_accounts(text: str, page_texts: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    # Account narratives can continue on later pages even after a page-one
    # column has already introduced non-credit or responsibility headings.
    # The date+institution+issuance anchor is narrow enough to scan the whole
    # report without treating query dates or repayment-liability prose as an
    # account start.
    starts = list(_ACCOUNT_START_RE.finditer(text))
    accounts: list[dict[str, Any]] = []
    source_sequences = {"credit_card": 0, "loan": 0}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        for marker in ("相关还款责任信息", "非信贷交易记录", "公共记录", "查询记录"):
            marker_at = text.find(marker, match.end(), end)
            if marker_at >= 0:
                end = min(end, marker_at)
        chunk = text[match.start() : end].strip()
        if "承担相关还款责任" in _compact(chunk):
            continue
        if not any(marker in chunk for marker in ("贷记卡", "准贷记卡", "贷款", "授信")):
            continue
        if not re.search(_ACCOUNT_ACTION_PATTERN, chunk):
            continue
        account = _personal_brief_account_from_chunk(chunk, page_texts)
        if not account or not account.get("account_id"):
            continue
        # The printed report owns two numbered account lists: credit cards and
        # loans. Loan facilities share the loan list. Preserve those source
        # ordinals in Community data instead of inventing one document-wide
        # sequence, while retaining the group in the surrogate identity.
        source_group = "credit_card" if account.get("account_type") == "credit_card" else "loan"
        source_sequences[source_group] += 1
        account["sequence"] = source_sequences[source_group]
        account["account_id"] = _stable_id(
            "credit_account",
            account["account_id"],
            source_group,
            account["sequence"],
        )
        if "ever_overdue" not in account:
            section_state = _personal_brief_overdue_section_state(text, match.start())
            if section_state is not None:
                account["ever_overdue"] = section_state
                if not section_state:
                    account["overdue_months_last_5y"] = 0
        accounts.append(account)
    return accounts


def _personal_brief_account_from_chunk(
    chunk: str,
    page_texts: list[tuple[int, str, str]],
) -> dict[str, Any] | None:
    opened = _DATE_CN_RE.match(chunk)
    if not opened:
        return None
    open_date = _iso_date(opened.group(0))
    remainder = chunk[opened.end() :]
    action_match = re.search(_ACCOUNT_ACTION_PATTERN, remainder)
    if not action_match:
        return None
    institution = re.sub(r"\s+", "", remainder[: action_match.start()]).strip("，,。.;；")
    if len(institution) < 4 or len(institution) > 80:
        return None
    body = remainder[action_match.end() :]
    compact_chunk = _compact(chunk)
    compact_body = _compact(body)

    card_type_match = _PERSONAL_BRIEF_CARD_TYPE_RE.match(compact_body)
    if card_type_match:
        account_type = "credit_card"
        business_type = card_type_match.group("business_type")
    elif "授信" in compact_body and re.search(r"贷款授信", compact_body):
        account_type = "credit_line"
        business_match = re.search(r"([\u3400-\u9fff（）()]{1,24}贷款)授信", compact_body)
        business_type = business_match.group(1) if business_match else "贷款授信"
    else:
        account_type = "loan"
        business_match = re.search(r"[）)]([\u3400-\u9fff（）()]{1,30}贷款)", compact_body)
        if not business_match:
            business_match = re.search(r"([\u3400-\u9fff（）()]{1,24}贷款)", compact_body)
        business_type = business_match.group(1) if business_match else "贷款"

    currency = _personal_brief_account_currency(compact_body)
    card_tail_match = re.search(r"卡片尾号[:：]?(\d{3,8})", compact_body)
    card_tail = card_tail_match.group(1) if card_tail_match else ""
    page = _source_page(page_texts, chunk[:120])
    is_card = account_type == "credit_card"
    card_activation_state = (
        "not_activated"
        if is_card and "尚未激活" in compact_body
        else "activated"
        if is_card and "已激活" in compact_body
        else "not_reported"
        if is_card
        else "not_applicable"
    )
    account_status = (
        "closed"
        if "销户" in compact_body
        else "settled"
        if "已结清" in compact_body
        else "transferred_out"
        if "已转出" in compact_body
        else "inactive"
        if card_activation_state == "not_activated"
        else "active"
    )
    lifecycle_state = (
        "closed"
        if account_status == "closed"
        else "settled"
        if account_status == "settled"
        else "transferred_out"
        if account_status == "transferred_out"
        else "open"
    )
    account: dict[str, Any] = {
        "account_type": account_type,
        "management_institution": institution,
        "business_type": business_type,
        "open_date": open_date,
        # ``currency`` remains as a compatibility alias for the account's
        # denomination. Personal Brief monetary values are reported in CNY.
        "currency": currency,
        "account_currency": currency,
        "reporting_amount_currency": "CNY",
        "amount_unit": "yuan",
        "reporting_amount_unit": "yuan",
        "reporting_amount_precision": 0,
        "account_status": account_status,
        "account_lifecycle_state": lifecycle_state,
        "card_activation_state": card_activation_state,
        "credit_quality_status": "bad_debt" if "呆账" in compact_body else "not_reported",
        "source": "personal_brief_narrative",
        "source_refs": _source_refs(page, "native_text_narrative"),
        "confidence": 0.94,
    }
    if is_card:
        account["credit_card_type"] = (
            "quasi_credit_card" if business_type == "准贷记卡" else "credit_card"
        )
    if card_tail:
        account["card_tail"] = card_tail

    patterns = {
        "loan_amount": r"发放的([\d,]+(?:\.\d+)?)元",
        "credit_limit": r"信用额度([\d,]+(?:\.\d+)?)",
        "balance": r"(?<!大额专项分期)(?<!分期)余额(?:为)?([\d,]+(?:\.\d+)?)",
        "used_amount": r"已使用额度([\d,]+(?:\.\d+)?)",
        "unbilled_installment_balance": r"(?:未出单的)?大额专项分期余额([\d,]+(?:\.\d+)?)",
    }
    for field, pattern in patterns.items():
        amount_match = re.search(pattern, compact_chunk)
        if amount_match:
            account[field] = _number(amount_match.group(1))
    if account_type == "loan" and "loan_amount" not in account:
        issued_amount = re.match(r"([\d,]+(?:\.\d+)?)元", compact_body)
        if issued_amount:
            account["loan_amount"] = _number(issued_amount.group(1))

    due_match = re.search(r"(20\d{2}年\d{1,2}月\d{1,2}日)到期", compact_chunk)
    if due_match:
        maturity_date = _iso_date(due_match.group(1))
        account["due_date"] = maturity_date
        account["contract_maturity_date"] = maturity_date
    close_match = re.search(r"(20\d{2}年\d{1,2}月)(?:已结清|销户)", compact_chunk)
    if close_match:
        account["close_date"] = _iso_month(close_match.group(1))
        account["termination_event_date"] = account["close_date"]
        account["termination_event_type"] = (
            "account_closed" if account_status == "closed" else "debt_settled"
        )
    transfer_match = re.search(r"(20\d{2}年\d{1,2}月)已转出", compact_chunk)
    if transfer_match:
        account["transfer_out_date"] = _iso_month(transfer_match.group(1))
        account["termination_event_date"] = account["transfer_out_date"]
        account["termination_event_type"] = "transferred_out"
    validity_match = re.search(r"额度有效期至(20\d{2}年\d{1,2}月\d{1,2}日)", compact_chunk)
    if validity_match:
        account["credit_line_validity_type"] = "fixed_term"
        account["credit_line_expiry_date"] = _iso_date(validity_match.group(1))
    elif "额度长期有效" in compact_chunk:
        account["credit_line_validity_type"] = "perpetual"
    as_of_match = re.search(r"截至(20\d{2}年\d{1,2}月(?:\d{1,2}日)?)", compact_chunk)
    if as_of_match:
        account["information_as_of"] = _iso_date(as_of_match.group(1)) or _iso_month(as_of_match.group(1))

    overdue_months = re.search(r"最近5年内有(\d+)个月处于逾期状态", compact_chunk)
    if overdue_months:
        account["overdue_months_last_5y"] = int(overdue_months.group(1))
        account["ever_overdue"] = True
    elif "从未发生过逾期" in compact_chunk or "从未逾期过" in compact_chunk:
        account["overdue_months_last_5y"] = 0
        account["ever_overdue"] = False
    if "当前无逾期" in compact_chunk:
        account["current_overdue"] = False
    elif "当前有逾期" in compact_chunk:
        account["current_overdue"] = True
    over_90_months = re.search(r"其中(\d+)个月逾期超过90天", compact_chunk)
    if over_90_months:
        account["over_90_days_months"] = int(over_90_months.group(1))
        account["over_90_days"] = int(over_90_months.group(1)) > 0
    elif "没有发生过90天以上" in compact_chunk or "没有发生过90天以上的逾期" in compact_chunk:
        account["over_90_days_months"] = 0
        account["over_90_days"] = False
    elif "发生过90天以上" in compact_chunk:
        account["over_90_days"] = True
    account["account_id"] = _stable_id(
        "credit_account",
        open_date,
        institution,
        business_type,
        currency,
        card_tail,
        account.get("contract_maturity_date"),
        account.get("credit_line_expiry_date"),
        account.get("credit_line_validity_type"),
        account.get("credit_limit"),
        account.get("loan_amount"),
    )
    return account


def _personal_brief_account_currency(compact_body: str) -> str:
    """Read the explicitly printed account currency, defaulting only if absent."""
    for match in re.finditer(r"[（(]([^()（）]{1,48})[）)]", compact_body):
        segment = match.group(1).split("，", 1)[0].split(",", 1)[0]
        has_account_marker = segment.endswith("账户")
        label = segment.removesuffix("账户").removesuffix("币种")
        normalized = normalize_currency_code(label)
        if normalized:
            return normalized
        # A future or non-ISO account label is still more truthful than CNY.
        if has_account_marker and label:
            return label
    return "CNY"


def _personal_brief_repayment_liabilities(
    text: str,
    page_texts: list[tuple[int, str, str]],
) -> list[dict[str, Any]]:
    """Extract personal-brief related repayment-responsibility narratives."""
    section_start = text.find("相关还款责任信息")
    if section_start < 0:
        return []
    section_end = len(text)
    for marker in ("非信贷交易记录", "公共记录", "查询记录"):
        marker_at = text.find(marker, section_start + len("相关还款责任信息"))
        if marker_at >= 0:
            section_end = min(section_end, marker_at)
    section = text[section_start:section_end]
    starts = list(
        re.finditer(
            rf"{_ACCOUNT_DATE_PATTERN}(?=(?:(?!{_ACCOUNT_DATE_PATTERN}).){{0,400}}?承担相关还款责任)",
            section,
        )
    )
    records: list[dict[str, Any]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(section)
        chunk = section[match.start() : end].strip()
        compact = _compact(chunk)
        core = re.search(
            r"^(20\d{2}年\d{1,2}月\d{1,2}日)[，,]?为"
            r"(.+?)[（(]证件类型[:：](.+?)[，,]证件号码[:：](.+?)[）)]"
            r"在(.+?)办理(?:的)?(.+?)承担相关还款责任[，,]"
            r"责任人类型为(.+?)[，,]相关还款责任金额([\d,.]+|--)"
            r"(?:[（(]保证合同编号[:：](.+?)[）)])?[。.]",
            compact,
        )
        if not core:
            continue
        remainder = compact[core.end() :]
        snapshot = re.search(r"截至(20\d{2}年\d{1,2}月(?:\d{1,2}日)?)，", remainder)
        snapshot_business = re.search(
            r"截至20\d{2}年\d{1,2}月(?:\d{1,2}日)?，(.+?)余额",
            remainder,
        )
        balance = re.search(r"余额([\d,.]+)(?:（?人民币元）?)?", remainder)
        liability_date = _iso_date(core.group(1))
        party_name = core.group(2)
        party_id_type = core.group(3)
        party_id_number = core.group(4)
        institution = core.group(5)
        business_type = core.group(6)
        responsibility_type = core.group(7)
        amount_text = core.group(8)
        contract_number = core.group(9) or ""
        page = _source_page(page_texts, chunk[:120])
        sequence = len(records) + 1
        record: dict[str, Any] = {
            "liability_id": _stable_id(
                "repayment_liability",
                liability_date,
                party_id_number,
                institution,
                contract_number,
                amount_text,
                sequence,
            ),
            "sequence": sequence,
            "liability_date": liability_date,
            "related_party_name": party_name,
            "related_party_id_type": party_id_type,
            "related_party_id_number": party_id_number,
            "management_institution": institution,
            "business_type": business_type,
            "underlying_business_type": business_type,
            "snapshot_balance_business_type": (
                snapshot_business.group(1) if snapshot_business else business_type
            ),
            "responsibility_type": responsibility_type,
            "responsibility_amount": _number(amount_text),
            "responsibility_amount_reported": amount_text != "--",
            "currency": "CNY",
            "reporting_amount_currency": "CNY",
            "amount_unit": "yuan",
            "reporting_amount_unit": "yuan",
            "source": "personal_brief_repayment_liability",
            "source_refs": _source_refs(page, "native_text_narrative"),
            "confidence": 0.94,
        }
        if contract_number:
            record["contract_number"] = contract_number
        if snapshot:
            record["snapshot_date"] = _iso_date(snapshot.group(1)) or _iso_month(snapshot.group(1))
        if balance:
            record["balance"] = _number(balance.group(1))
        records.append(record)
    return records


def _personal_brief_inquiries(
    parse_result: Any,
    text: str,
    page_texts: list[tuple[int, str, str]],
) -> list[dict[str, Any]]:
    if "机构查询记录明细" not in text:
        return []
    institution_start = text.index("机构查询记录明细")
    personal_starts = [
        position
        for heading in ("本人查询记录明细", "个人查询记录明细")
        if (position := text.find(heading, institution_start)) >= 0
    ]
    personal_start = min(personal_starts) if personal_starts else -1
    institution_text = text[institution_start : personal_start if personal_start >= 0 else len(text)]
    personal_text = text[personal_start:] if personal_start >= 0 else ""
    institution_records = _merge_reconstructed_personal_brief_institution_inquiries(
        _personal_brief_inquiry_rows(institution_text, "institution", page_texts),
        _reconstructed_personal_brief_institution_inquiries(parse_result),
    )
    records = [
        *institution_records,
        *_personal_brief_inquiry_rows(personal_text, "personal", page_texts),
    ]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("inquiry_id") or "")
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        out.append(record)
    return sorted(
        out,
        key=lambda record: (
            0 if record.get("inquiry_type") == "institution" else 1,
            int(record.get("sequence") or 0),
            str(record.get("inquiry_date") or ""),
        ),
    )


def _reconstructed_personal_brief_institution_inquiries(
    parse_result: Any,
) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.inquiry_reading_order import (
        reconstruct_institution_inquiry_rows,
    )

    records: list[dict[str, Any]] = []
    for row in reconstruct_institution_inquiry_rows(parse_result):
        query_date = _iso_date(str(row.get("query_date_text") or ""))
        institution = _compact(str(row.get("institution") or ""))
        reason = str(row.get("reason") or "").replace("(", "（").replace(")", "）")
        if not query_date or not institution or not reason:
            continue
        source_ref: dict[str, Any] = {"source": "native_dfg_inquiry_ledger"}
        if page := int(row.get("page") or 0):
            source_ref["page"] = page
        if node_ids := [str(value) for value in row.get("source_node_ids") or [] if value]:
            source_ref["node_ids"] = node_ids
        if evidence_ids := [str(value) for value in row.get("evidence_ids") or [] if value]:
            source_ref["evidence_ids"] = evidence_ids
        sequence = int(row.get("sequence") or 0)
        records.append(
            {
                "inquiry_id": _stable_id(
                    "credit_inquiry",
                    "institution",
                    sequence,
                    query_date,
                    institution,
                    reason,
                ),
                "sequence": sequence,
                "inquiry_type": "institution",
                "inquiry_date": query_date,
                "institution": institution,
                "reason": reason,
                "source": "personal_brief_inquiry_ledger",
                "source_refs": [source_ref],
                "confidence": float(row.get("confidence") or 0.98),
            }
        )
    return records


def _merge_reconstructed_personal_brief_institution_inquiries(
    fallback: list[dict[str, Any]],
    reconstructed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer DFG rows while retaining text-only rows as compatibility fallback."""

    def key(record: dict[str, Any]) -> tuple[int, str]:
        return int(record.get("sequence") or 0), str(record.get("inquiry_date") or "")

    reconstructed_by_key = {key(record): record for record in reconstructed}
    merged = [reconstructed_by_key.pop(key(record), record) for record in fallback]
    merged.extend(reconstructed_by_key.values())
    return merged


def _personal_brief_inquiry_rows(
    section: str,
    inquiry_type: str,
    page_texts: list[tuple[int, str, str]],
) -> list[dict[str, Any]]:
    matches = list(
        re.finditer(
            r"(?<!\d)(\d{1,4})\s+(20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)\s+",
            section,
        )
    )
    out: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        rest = _compact(section[match.end() : end])
        source_reason = next((candidate for candidate in _INQUIRY_REASONS if candidate in rest), "")
        if not source_reason:
            source_reason = _personal_brief_inquiry_reason(rest, inquiry_type)
        if not source_reason:
            continue
        source_reason = source_reason.replace("(", "（").replace(")", "）")
        institution = "本人" if inquiry_type == "personal" else rest.split(source_reason, 1)[0]
        if not institution or len(institution) > 100:
            continue
        query_date = _iso_date(match.group(2))
        page = _source_page(page_texts, f"{match.group(1)}{match.group(2)}{institution[:12]}")
        query_channel_match = re.search(r"[（(]([^）)]+)[）)]", source_reason)
        query_channel = query_channel_match.group(1) if query_channel_match else ""
        reason = "本人查询" if inquiry_type == "personal" else source_reason
        out.append(
            {
                "inquiry_id": _stable_id(
                    "credit_inquiry",
                    inquiry_type,
                    int(match.group(1)),
                    query_date,
                    institution,
                    source_reason,
                ),
                "sequence": int(match.group(1)),
                "inquiry_type": inquiry_type,
                "inquiry_date": query_date,
                "institution": institution,
                "reason": reason,
                "source_reason": source_reason,
                **({"query_channel": query_channel} if query_channel else {}),
                "source": "personal_brief_inquiry_ledger",
                "source_refs": _source_refs(page, "native_text_ledger"),
                "confidence": 0.97,
            }
        )
    return out


def _personal_brief_inquiry_reason(rest: str, inquiry_type: str) -> str:
    """Read new PBOC reason labels conservatively from the ledger tail."""
    normalized = rest.replace("(", "（").replace(")", "）")
    if inquiry_type == "personal":
        match = re.search(r"(本人查询(?:（[^）]{1,40}）)?)$", normalized)
        return match.group(1) if match else ""
    matches = list(re.finditer(
        r"((?:个人|企业|信用卡|融资|授信|担保|法人|负责人|高管|贷后|保前|资信|客户|风险|关联|异议|账户|商户)"
        r"[^，,。；;]{0,24}(?:审批|审查|管理|查询|核查|复核|评估|授信|准入)"
        r"(?:（[^）]{1,40}）)?)$",
        normalized,
    ))
    match = min(matches, key=lambda item: len(item.group(1))) if matches else None
    return match.group(1) if match else ""


def _overdue_from_personal_brief_accounts(
    accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for account in accounts:
        if not account.get("ever_overdue"):
            continue
        account_id = str(account.get("account_id") or "")
        current_overdue = account.get("current_overdue")
        reported_overdue_months = account.get("overdue_months_last_5y")
        out.append(
            {
                "overdue_id": _stable_id("credit_overdue", account_id, "last_5_years"),
                "account_id": account_id,
                "sequence": account.get("sequence"),
                "account_type": account.get("account_type"),
                "management_institution": account.get("management_institution"),
                "business_type": account.get("business_type"),
                "card_tail": account.get("card_tail"),
                "open_date": account.get("open_date"),
                "currency": account.get("currency"),
                "period_scope": "last_5_years",
                "overdue_months": (
                    int(reported_overdue_months)
                    if reported_overdue_months is not None
                    else None
                ),
                "over_90_days_months": account.get("over_90_days_months"),
                "over_90_days": account.get("over_90_days"),
                "current_overdue": current_overdue,
                "current_overdue_status": (
                    "overdue"
                    if current_overdue is True
                    else "not_overdue"
                    if current_overdue is False
                    else "not_reported"
                ),
                "source": "personal_brief_account_narrative",
                "source_refs": list(account.get("source_refs") or []),
                "confidence": account.get("confidence", 0.9),
            }
        )
    return out


def _plain(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("normalized_value", "normalized", "value", "raw_value", "raw"):
        if value.get(key) not in (None, ""):
            return value[key]
    return None


def _record_value(record: dict[str, Any], key: str) -> Any:
    value = _plain(record.get(key))
    if value not in (None, ""):
        return value
    normalized = record.get("normalized")
    if isinstance(normalized, dict):
        return _plain(normalized.get(key))
    return None


def derive_overdue_records(
    credit_accounts: list[Any],
    repayment_records: list[Any],
    existing: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one canonical overdue view from narrative, account, and month-grid facts."""
    out = [dict(item) for item in existing or [] if isinstance(item, dict)]
    seen = {str(item.get("overdue_id") or "") for item in out}
    account_types: dict[str, str] = {}
    for account in credit_accounts or []:
        if not isinstance(account, dict):
            continue
        account_id = str(_record_value(account, "account_id") or account.get("source_structure_id") or "")
        account_type = str(_record_value(account, "account_type") or _record_value(account, "credit_card_type") or "")
        if account_id and account_type:
            account_types[account_id] = account_type
        status = str(_record_value(account, "account_status") or _record_value(account, "status") or "")
        five_tier = str(_record_value(account, "five_tier_class") or "")
        overdue_amount = _number(str(_record_value(account, "overdue_amount") or ""))
        if status not in {"逾期", "overdue"} and five_tier not in {"关注", "次级", "可疑", "损失", "违约"}:
            if not overdue_amount:
                continue
        overdue_id = _stable_id("credit_overdue", account_id, "account_snapshot")
        if overdue_id in seen:
            continue
        seen.add(overdue_id)
        out.append(
            {
                "overdue_id": overdue_id,
                "account_id": account_id,
                "period_scope": "account_snapshot",
                "overdue_amount": overdue_amount,
                "five_tier_class": five_tier,
                "source": "credit_account_snapshot",
                "source_refs": list(account.get("source_refs") or []),
                "confidence": account.get("confidence", 0.8),
            }
        )
    for record in repayment_records or []:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "")
        if status not in {"1", "2", "3", "4", "5", "6", "7"}:
            continue
        account_id = str(record.get("account_id") or record.get("grid_id") or "")
        # The personal-report legend assigns quasi-credit-card codes 1 and 2
        # to normal balance states; only codes 3 through 7 are overdue.
        if account_types.get(account_id) == "quasi_credit_card" and status in {"1", "2"}:
            continue
        try:
            year = int(record.get("year") or 0)
            month = int(record.get("month") or 0)
        except (TypeError, ValueError):
            continue
        overdue_id = _stable_id("credit_overdue", account_id, year, month)
        if overdue_id in seen:
            continue
        seen.add(overdue_id)
        out.append(
            {
                "overdue_id": overdue_id,
                "account_id": account_id,
                "period_scope": "month",
                "year": year,
                "month": month,
                "overdue_level": int(status),
                "overdue_amount": _number(str(record.get("overdue_amount") or "")),
                "source": "repayment_micro_grid",
                "source_cell_refs": list(record.get("source_cell_refs") or []),
                "confidence": record.get("confidence", 0.8),
            }
        )
    return out


__all__ = [
    "derive_overdue_records",
    "extract_native_credit_business",
]
