# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite text-line fallback for the canonical PBOC credit overview.

Some scans preserve the first summary header and row as positioned text while
the remaining rows are emitted as a table.  This module only decodes the exact
closed business categories printed by that canonical overview.  It deliberately
does not perform fuzzy label repair or infer a missing count/month.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_OVERVIEW_HEADER_LABELS = ("业务类型", "账户数", "首笔业务发放月份")
_OVERVIEW_CATEGORIES = tuple(
    sorted(
        (
            "个人住房贷款",
            "个人商用房贷款(包括商住两用房)",
            "个人商用房贷款（包括商住两用房）",
            "其他类贷款",
            "贷记卡",
            "准贷记卡",
        ),
        key=len,
        reverse=True,
    )
)
_MONTH_RE = re.compile(r"((?:19|20)\d{2})[.:](\d{1,2})")


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def is_credit_business_overview_text_header(value: Any) -> bool:
    compact = _compact(value)
    return all(label in compact for label in _OVERVIEW_HEADER_LABELS)


def decode_credit_business_overview_text_line(value: Any) -> dict[str, Any] | None:
    """Decode one exact category/count/month line, or withhold it entirely."""

    compact = _compact(value)
    matches = [category for category in _OVERVIEW_CATEGORIES if category in compact]
    # The short label 贷记卡 is contained in 准贷记卡.  Longest-match removes
    # that one lexical overlap; every other multi-category line is ambiguous.
    if "准贷记卡" in matches and "贷记卡" in matches:
        matches.remove("贷记卡")
    if len(matches) != 1:
        return None
    category = matches[0]
    tail = compact.split(category, 1)[1]
    month_matches = list(_MONTH_RE.finditer(tail))
    if len(month_matches) != 1:
        return None
    month_match = month_matches[0]
    year, month = (int(part) for part in month_match.groups())
    if not 1 <= month <= 12:
        return None
    prefix = tail[: month_match.start()]
    suffix = tail[month_match.end() :]
    if suffix or not re.fullmatch(r"\d{1,3}", prefix):
        return None
    return {
        "business_type": category.translate(str.maketrans({"(": "（", ")": "）"})),
        "account_count": int(prefix),
        "first_business_issue_month": f"{year:04d}-{month:02d}",
    }


def decode_credit_business_overview_text_lines(
    lines: Iterable[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Decode exact rows after the canonical header and retain failed witnesses."""

    active = False
    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for value in lines:
        text = str(value or "").strip()
        if is_credit_business_overview_text_header(text):
            active = True
            continue
        if not active:
            continue
        if "汇总" in _compact(text):
            break
        category_like = any(category in _compact(text) for category in _OVERVIEW_CATEGORIES)
        decoded = decode_credit_business_overview_text_line(text)
        if decoded is not None:
            rows.append(decoded)
        elif category_like:
            unresolved.append(text)
    return rows, unresolved


__all__ = [
    "decode_credit_business_overview_text_line",
    "decode_credit_business_overview_text_lines",
    "is_credit_business_overview_text_header",
]
