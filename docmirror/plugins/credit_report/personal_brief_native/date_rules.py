# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed date grammar for digital PBOC Personal Brief reports.

The source prints four-digit years.  Accept 1900-2099 exactly as printed and
never infer a century from a two-digit value.
"""

from __future__ import annotations

import re
from typing import Any

PERSONAL_BRIEF_YEAR_PATTERN = r"(?:19|20)\d{2}"
PERSONAL_BRIEF_DATE_PATTERN = (
    rf"{PERSONAL_BRIEF_YEAR_PATTERN}\s*年\s*"
    r"\d{1,2}\s*月\s*\d{1,2}\s*日"
)
PERSONAL_BRIEF_MONTH_PATTERN = rf"{PERSONAL_BRIEF_YEAR_PATTERN}\s*年\s*\d{{1,2}}\s*月"
PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN = rf"{PERSONAL_BRIEF_MONTH_PATTERN}(?:\s*\d{{1,2}}\s*日)?"

_CHINESE_DATE_RE = re.compile(
    rf"(?P<year>{PERSONAL_BRIEF_YEAR_PATTERN})\s*年\s*"
    r"(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日"
)
_SEPARATED_DATE_RE = re.compile(
    rf"(?P<year>{PERSONAL_BRIEF_YEAR_PATTERN})\s*[-./]\s*"
    r"(?P<month>\d{1,2})\s*[-./]\s*(?P<day>\d{1,2})"
)
_CHINESE_MONTH_RE = re.compile(
    rf"(?P<year>{PERSONAL_BRIEF_YEAR_PATTERN})\s*年\s*"
    r"(?P<month>\d{1,2})\s*月"
)
_SEPARATED_MONTH_RE = re.compile(
    rf"(?P<year>{PERSONAL_BRIEF_YEAR_PATTERN})\s*[-./]\s*"
    r"(?P<month>\d{1,2})(?!\s*[-./]\s*\d)"
)


def normalize_personal_brief_date(value: Any) -> str:
    """Return an ISO date while preserving the source's explicit century."""

    raw = str(value or "")
    match = _CHINESE_DATE_RE.search(raw) or _SEPARATED_DATE_RE.search(raw)
    if match is None:
        return ""
    return f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"


def normalize_personal_brief_month(value: Any) -> str:
    """Return an ISO month while preserving the source's explicit century."""

    raw = str(value or "")
    match = _CHINESE_MONTH_RE.search(raw) or _SEPARATED_MONTH_RE.search(raw)
    if match is None:
        return ""
    return f"{int(match.group('year')):04d}-{int(match.group('month')):02d}"


__all__ = [
    "PERSONAL_BRIEF_DATE_PATTERN",
    "PERSONAL_BRIEF_MONTH_OR_DATE_PATTERN",
    "PERSONAL_BRIEF_MONTH_PATTERN",
    "PERSONAL_BRIEF_YEAR_PATTERN",
    "normalize_personal_brief_date",
    "normalize_personal_brief_month",
]
