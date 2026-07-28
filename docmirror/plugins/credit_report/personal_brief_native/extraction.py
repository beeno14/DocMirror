# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variant-owned extraction entry points for native personal brief reports."""

from __future__ import annotations

import re
from typing import Any


def extract_personal_brief_native_business(
    parse_result: Any,
    full_text: str,
) -> dict[str, Any]:
    """Transform a ParseResult into personal-brief business candidates."""
    from docmirror.plugins.credit_report.business_records import (
        _linear,
        _overdue_from_personal_brief_accounts,
        _page_texts,
        _personal_brief_accounts,
        _personal_brief_credit_lines,
        _personal_brief_inquiries,
        _personal_brief_repayment_liabilities,
        _personal_brief_summary_from_canonical_tables,
    )

    text = _linear(full_text)
    page_texts = _page_texts(parse_result)
    accounts = _personal_brief_accounts(text, page_texts)
    liabilities = _personal_brief_repayment_liabilities(text, page_texts)
    inquiries = _personal_brief_inquiries(parse_result, text, page_texts)
    overdue = _overdue_from_personal_brief_accounts(accounts)
    credit_lines = _personal_brief_credit_lines(accounts)
    source_summary = _personal_brief_summary_from_canonical_tables(parse_result)
    return {
        "credit_accounts": accounts,
        "credit_lines": credit_lines,
        "repayment_liability_records": liabilities,
        "overdue_records": overdue,
        "inquiry_records": inquiries,
        "credit_summary": {
            "source": "personal_brief_native_text",
            "account_count": len(accounts),
            "active_account_count": sum(account.get("account_status") == "active" for account in accounts),
            "active_account_count_basis": "derived_account_status_active",
            "activated_credit_card_account_count": sum(
                account.get("account_type") == "credit_card" and account.get("account_status") == "active"
                for account in accounts
            ),
            "inactive_credit_card_account_count": sum(
                account.get("account_type") == "credit_card" and account.get("account_status") == "inactive"
                for account in accounts
            ),
            "settled_account_count": sum(
                account.get("account_status") in {"settled", "closed"} for account in accounts
            ),
            "derived_ever_overdue_account_count": len(overdue),
            "repayment_liability_count": len(liabilities),
            "inquiry_count": len(inquiries),
            "institution_inquiry_count": sum(item.get("inquiry_type") == "institution" for item in inquiries),
            "personal_inquiry_count": sum(item.get("inquiry_type") == "personal" for item in inquiries),
            **source_summary,
        },
    }


def extract_personal_brief_section_content(
    parse_result: Any,
    full_text: str,
) -> dict[str, Any]:
    """Return personal-brief-only facts and supplemental records."""
    from docmirror.plugins.credit_report.business_records import (
        _compact,
        _linear,
        _page_texts,
        _source_page,
        _source_refs,
        _stable_id,
    )

    blocks: list[tuple[int, str]] = []
    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "page_number", 0) or page_index)
        for block in getattr(page, "texts", None) or []:
            content = str(getattr(block, "content", "") or "").strip()
            if content:
                blocks.append((page_number, content))

    def statement_after(heading: str) -> tuple[int, str]:
        for index, (page, content) in enumerate(blocks):
            if _compact(content) != heading:
                continue
            for next_page, candidate in blocks[index + 1 :]:
                if next_page != page:
                    break
                compact = _compact(candidate)
                if compact and not re.fullmatch(r"第\d+页，共\d+页", compact):
                    return page, re.sub(r"\s+", " ", candidate).strip()
        text = _linear(full_text)
        marker = text.find(heading)
        if marker < 0:
            return 0, ""
        remainder = text[marker + len(heading) :]
        end = min(
            [
                position
                for value in ("非信贷交易记录", "公共记录", "查询记录", "说明")
                if (position := remainder.find(value)) >= 0
            ]
            or [len(remainder)]
        )
        return _source_page(_page_texts(parse_result), heading), remainder[:end].strip()

    non_credit_page, non_credit_statement = statement_after("非信贷交易记录")
    public_page, public_statement = statement_after("公共记录")
    notes: list[dict[str, Any]] = []
    for index, (page, content) in enumerate(blocks):
        if _compact(content) != "说明":
            continue
        note_text = "\n".join(
            candidate
            for next_page, candidate in blocks[index + 1 :]
            if next_page == page and not re.fullmatch(r"\s*第\s*\d+\s*页，共\s*\d+\s*页\s*", candidate)
        )
        for match in re.finditer(r"(?ms)(\d+)\.\s*(.*?)(?=^\d+\.|\Z)", note_text):
            sequence = int(match.group(1))
            content_value = re.sub(r"\s+", " ", match.group(2)).strip()
            if not content_value:
                continue
            notes.append(
                {
                    "note_id": _stable_id("credit_report_note", sequence, content_value),
                    "sequence": sequence,
                    "content": content_value,
                    "source": "personal_brief_notes",
                    "source_refs": _source_refs(page, "native_text_note"),
                    "confidence": 1.0,
                }
            )
        break
    return {
        "non_credit_transaction_summary": {
            "record_status": "no_records" if "没有" in non_credit_statement else "reported",
            "lookback_years": 5 if "5年" in non_credit_statement else None,
            "source_statement": non_credit_statement,
            "source_page": non_credit_page or None,
        },
        "public_record_summary": {
            "record_status": "no_records" if "没有" in public_statement else "reported",
            "lookback_years": 5 if "5年" in public_statement else None,
            "source_statement": public_statement,
            "source_page": public_page or None,
        },
        "report_notes": notes,
    }


__all__ = [
    "extract_personal_brief_native_business",
    "extract_personal_brief_section_content",
]
