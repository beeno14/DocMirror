# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite OCR aliases for PBOC credit-agreement headings and labels.

This module is deliberately closed-world.  It contains only independently
observed OCR renderings of canonical agreement text; it performs no fuzzy,
edit-distance, or document-specific matching.
"""

from __future__ import annotations

import re
from typing import Any

CANONICAL_CREDIT_AGREEMENT_PREFIX = "授信"
CREDIT_AGREEMENT_OCR_PREFIXES = (
    CANONICAL_CREDIT_AGREEMENT_PREFIX,
    "授伯",
    "授值",
    "投信",
    "投值",
    "投伯",
    "投借",
)

_CREDIT_AGREEMENT_HEADING_STEMS = (
    *(f"{prefix}协议" for prefix in CREDIT_AGREEMENT_OCR_PREFIXES),
    "投值协这",
)
CREDIT_AGREEMENT_CARD_HEADING_RE = re.compile(
    rf"(?:{'|'.join(re.escape(value) for value in _CREDIT_AGREEMENT_HEADING_STEMS)})"
    rf"\s*(?P<sequence>[1-9]\d{{0,2}})(?!\d)"
)

_CHINESE_SECTION_ORDINAL = r"[〇零一二三四五六七八九十百]{1,5}"
_OPTIONAL_SECTION_ORDINAL = (
    rf"(?:(?:[（(]{_CHINESE_SECTION_ORDINAL}[）)])|{_CHINESE_SECTION_ORDINAL})?"
)
CREDIT_AGREEMENT_SECTION_HEADING_RE = re.compile(
    rf"^{_OPTIONAL_SECTION_ORDINAL}"
    rf"(?:{'|'.join(re.escape(value) for value in _CREDIT_AGREEMENT_HEADING_STEMS)})"
    r"信息$"
)

_CANONICAL_PREFIX_LABEL_SUFFIXES = (
    "协议标识",
    "额度用途",
    "额度",
    "限额",
    "限额编号",
)
CREDIT_AGREEMENT_LABEL_OCR_ALIASES = {
    f"{prefix}{suffix}": f"{CANONICAL_CREDIT_AGREEMENT_PREFIX}{suffix}"
    for prefix in CREDIT_AGREEMENT_OCR_PREFIXES
    if prefix != CANONICAL_CREDIT_AGREEMENT_PREFIX
    for suffix in _CANONICAL_PREFIX_LABEL_SUFFIXES
}
CREDIT_AGREEMENT_LABEL_OCR_ALIASES.update(
    {
        "授值协仪标识": "授信协议标识",
        "授信标度用途": "授信额度用途",
    }
)


def credit_agreement_label_aliases(canonical_label: str) -> tuple[str, ...]:
    """Return the exact canonical label and its finite OCR aliases."""

    return (
        canonical_label,
        *(
            alias
            for alias, canonical in CREDIT_AGREEMENT_LABEL_OCR_ALIASES.items()
            if canonical == canonical_label
        ),
    )


CREDIT_AGREEMENT_PRIMARY_LABELS = credit_agreement_label_aliases("授信协议标识")
CREDIT_AGREEMENT_PURPOSE_LABELS = credit_agreement_label_aliases("授信额度用途")
CREDIT_AGREEMENT_AMOUNT_LABELS = (
    *credit_agreement_label_aliases("授信额度"),
    *credit_agreement_label_aliases("授信限额"),
)


def canonical_credit_agreement_heading(value: Any) -> str | None:
    """Canonicalize one complete finite-alias card heading, or fail closed."""

    compact = re.sub(r"\s+", "", str(value or ""))
    match = CREDIT_AGREEMENT_CARD_HEADING_RE.fullmatch(compact)
    if match is None:
        return None
    return f"授信协议{int(match.group('sequence'))}"


def canonical_credit_agreement_section_heading(value: Any) -> str | None:
    """Canonicalize one complete finite-alias PBOC agreement section title."""

    compact = re.sub(r"\s+", "", str(value or "")).rstrip("：:")
    if CREDIT_AGREEMENT_SECTION_HEADING_RE.fullmatch(compact) is None:
        return None
    return "授信协议信息"
