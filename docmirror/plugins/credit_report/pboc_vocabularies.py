# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, cross-variant PBOC personal-report vocabularies.

These values are semantic contracts of the PBOC report family.  They are not
OCR correction rules: a glyph-near value is never admitted by this module, and
an acquisition-specific alias must retain its observed provenance before a
caller may propose a correction.
"""

from __future__ import annotations

import re

PBOC_INSTITUTION_NAME_SUFFIXES = frozenset(
    {
        "银行卡业务部(牡丹卡中心)",
        "农村信用合作联社",
        "农村信用社联合社",
        "股份有限公司",
        "有限责任公司",
        "信用卡中心",
        "个人信贷部",
        "管理中心",
        "股份公司",
        "有限公司",
        "营业管理部",
        "营业部",
        "管理部",
        "信用社",
        "合作社",
        "分行",
        "支行",
        "银行",
        "中心",
        "公司",
        "联社",
        "机构",
        "信托",
    }
)

PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_ID = "pboc.personal_credit_report.inquiry_reason"
PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_VERSION = "1.0.0"

# Institution-query reasons printed across supported PBOC personal-report
# revisions.  The registry is intentionally finite and exact.  New official
# codes are added as a contract revision; they are not inferred from suffixes.
PBOC_INSTITUTION_INQUIRY_REASONS = frozenset(
    {
        "贷后管理",
        "贷款审批",
        "信用卡审批",
        "担保资格审查",
        "融资审批",
        "保前审查",
        "保后管理",
        "客户准入资格审查",
        "资信审查",
        "法人代表、负责人、高管等资信审查",
        "特约商户实名审查",
        "异议处理",
        "司法调查",
        "公积金提取复核",
        "其他",
    }
)

PBOC_SELF_INQUIRY_CHANNELS = frozenset(
    {
        "本人查询",
        "本人查询(自助查询机)",
        "本人查询(商业银行网上银行)",
        "本人查询(互联网个人信用信息服务平台)",
        "本人查询(征信中心柜台)",
        "本人查询(临柜)",
    }
)

PBOC_INQUIRY_REASON_ROOTS = frozenset(
    {"本人查询", *PBOC_INSTITUTION_INQUIRY_REASONS}
)
PBOC_INQUIRY_REASON_FORMS = frozenset(
    {*PBOC_INSTITUTION_INQUIRY_REASONS, *PBOC_SELF_INQUIRY_CHANNELS}
)

_PARENTHESIS_TRANSLATION = str.maketrans({"（": "(", "）": ")"})


def is_pboc_institution_name(value: object) -> bool:
    """Return whether one exact schema-owned value has a PBOC institution shape.

    This is a semantic field contract, not an OCR repair rule.  It requires a
    complete registered organizational suffix and never searches a larger
    string for a plausible institution substring.  Exact source ownership and
    cross-cell contamination remain caller responsibilities.
    """

    compact = re.sub(r"\s+", "", str(value or "")).translate(
        _PARENTHESIS_TRANSLATION
    )
    if compact == "本人":
        return True
    return bool(
        2 <= len(compact) <= 100
        and re.fullmatch(r"[A-Za-z0-9\u3400-\u9fff（）()·\-‐‑‒–—―－]+", compact)
        and re.search(r"[\u3400-\u9fff]", compact)
        and any(compact.endswith(suffix) for suffix in PBOC_INSTITUTION_NAME_SUFFIXES)
    )


def canonical_pboc_inquiry_reason(value: object) -> str | None:
    """Return one exact PBOC reason form after typographic normalization.

    Whitespace and full-width parentheses are presentation artifacts inside a
    schema-bound reason cell.  No spelling correction or substring matching is
    performed.  Unknown exact source text therefore remains reviewable.
    """

    compact = re.sub(r"\s+", "", str(value or "")).translate(
        _PARENTHESIS_TRANSLATION
    )
    return compact if compact in PBOC_INQUIRY_REASON_FORMS else None


def pboc_inquiry_reason_root(value: object) -> str | None:
    """Return the semantic root of one exact registered PBOC reason form."""

    canonical = canonical_pboc_inquiry_reason(value)
    if canonical is None:
        return None
    return "本人查询" if canonical.startswith("本人查询") else canonical


def longest_pboc_inquiry_reason_suffix(value: object) -> str | None:
    """Return one registered reason occupying the complete text suffix.

    This helper is for already PBOC-owned inquiry rows.  It never searches for
    a reason in the middle of an institution or prose string.
    """

    match = pboc_inquiry_reason_suffix(value)
    return match[0] if match is not None else None


def pboc_inquiry_reason_suffix(value: object) -> tuple[str, str, int] | None:
    """Return ``(form, root, source_start)`` for one complete row suffix.

    The returned offset points into the original source text, so callers can
    split institution and reason without searching for a normalized spelling.
    Whitespace and full-width parentheses are normalized only for comparison.
    The generic code ``其他`` additionally requires a source boundary when it
    follows other text; otherwise an institution name ending in those two
    characters could be truncated.
    """

    source = str(value or "")
    normalized: list[str] = []
    source_indexes: list[int] = []
    for index, character in enumerate(source):
        if character.isspace():
            continue
        normalized.append(character.translate(_PARENTHESIS_TRANSLATION))
        source_indexes.append(index)
    compact = "".join(normalized)
    matches = sorted(
        (reason for reason in PBOC_INQUIRY_REASON_FORMS if compact.endswith(reason)),
        key=len,
        reverse=True,
    )
    for reason in matches:
        normalized_start = len(compact) - len(reason)
        source_start = source_indexes[normalized_start] if source_indexes else 0
        if reason == "其他" and source_start > 0 and not source[source_start - 1].isspace():
            continue
        root = "本人查询" if reason.startswith("本人查询") else reason
        return reason, root, source_start
    return None


__all__ = [
    "PBOC_INSTITUTION_NAME_SUFFIXES",
    "PBOC_INQUIRY_REASON_FORMS",
    "PBOC_INQUIRY_REASON_ROOTS",
    "PBOC_INSTITUTION_INQUIRY_REASONS",
    "PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_ID",
    "PBOC_PERSONAL_INQUIRY_REASON_CONTRACT_VERSION",
    "PBOC_SELF_INQUIRY_CHANNELS",
    "canonical_pboc_inquiry_reason",
    "is_pboc_institution_name",
    "longest_pboc_inquiry_reason_suffix",
    "pboc_inquiry_reason_suffix",
    "pboc_inquiry_reason_root",
]
