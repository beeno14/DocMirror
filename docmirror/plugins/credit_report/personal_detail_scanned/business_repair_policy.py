# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure, bounded policy predicates for personal-detail business repair.

These helpers do not read a document, start OCR, or mutate an extraction.  They
only answer whether one already-localized field has exactly one deterministic
candidate.  Callers remain responsible for proving exact field ownership and,
when no candidate is returned, acquiring an independent context-rich OCR view.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

_INQUIRY_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[.,/:-]?(\d{2})[.,/:-]?(\d{2})(?!\d)"
)
_BOUNDED_EDGE_NOISE_RE = re.compile(r"(?:[A-Za-z]{1,3}|[\u3400-\u9fff])")
_INSTITUTION_LEGAL_ENDINGS = (
    "银行卡业务部(牡丹卡中心)",
    "银行卡业务部（牡丹卡中心）",
    "信用卡中心",
    "个人信贷部",
    "农村信用合作联社",
    "农村信用社联合社",
    "股份有限公司",
    "有限责任公司",
    "股份公司",
    "有限公司",
    "管理中心",
    "支行",
    "分行",
)
_AGREEMENT_ADJACENT_LABELS = (
    "授信额度用途",
    "授信协议标识",
    "生效日期",
    "到期日期",
    "授信额度",
    "已用额度",
    "授信限额",
    "授信限额编号",
    "币种",
)
_LIABILITY_BUSINESS_TYPES = frozenset(
    {
        "贷款",
        "个人住房贷款",
        "个人住房商业贷款",
        "个人住房公积金贷款",
        "个人商用房贷款",
        "个人经营性贷款",
        "企业经营贷款",
        "个人消费贷款",
        "个人汽车消费贷款",
        "其他个人消费贷款",
        "国家助学贷款",
        "农户贷款",
        "其他贷款",
        "其他类贷款",
        "循环贷款",
        "融资租赁",
        "融资租赁业务",
        "贷记卡",
        "准贷记卡",
        "票据贴现",
        "银行保函",
    }
)


def _plain(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\u200b", "").strip()


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", _plain(value))


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def deterministic_inquiry_date_candidate(value: Any) -> str | None:
    """Return one date after removing only short nonnumeric edge residue.

    Numeric residue is deliberately excluded.  A value such as
    ``2023.01.03 20`` could contain a damaged second date or ordinal and must be
    sent to the context-rich OCR acquisition instead of being substring-cleaned.
    """

    text = _plain(value).replace(",", ".")
    candidates: list[tuple[tuple[int, int], str]] = []
    for match in _INQUIRY_DATE_RE.finditer(text):
        candidate = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        if _valid_date(candidate):
            candidates.append((match.span(), candidate))
    unique = {(span, candidate) for span, candidate in candidates}
    if len(unique) != 1:
        return None
    (start, end), candidate = next(iter(unique))
    left_residue = re.sub(r"\s+", "", text[:start])
    right_residue = re.sub(r"\s+", "", text[end:])
    residues = [residue for residue in (left_residue, right_residue) if residue]
    if (
        len(residues) != 1
        or _BOUNDED_EDGE_NOISE_RE.fullmatch(residues[0]) is None
    ):
        return None
    return candidate


def bounded_inquiry_sequence_noise_candidate(value: Any) -> tuple[int, str] | None:
    """Return one digit ordinal with exactly one Han/Latin edge glyph."""

    text = _plain(value)
    prefixed = re.fullmatch(r"[A-Za-z\u3400-\u9fff]\s*(\d{1,4})", text)
    if prefixed is not None:
        sequence = int(prefixed.group(1))
        return (sequence, "prefixed_noise") if sequence > 0 else None
    suffixed = re.fullmatch(r"(\d{1,4})\s*[A-Za-z\u3400-\u9fff]", text)
    if suffixed is not None:
        sequence = int(suffixed.group(1))
        return (sequence, "suffix_noise") if sequence > 0 else None
    return None


def deterministic_agreement_institution_candidate(value: Any) -> str | None:
    """Remove one complete adjacent agreement label from a legal-name suffix.

    Whitespace folding alone is ordinary presentation normalization.  This
    policy handles only a known label copied after one otherwise complete legal
    institution name; arbitrary trailing glyphs and reordered names require an
    independent OCR acquisition.
    """

    compact = _compact(value).strip("-_:：,，;；")
    matches = [label for label in _AGREEMENT_ADJACENT_LABELS if compact.endswith(label)]
    if len(matches) != 1:
        return None
    label = matches[0]
    candidate = compact[: -len(label)]
    if not candidate or not any(candidate.endswith(ending) for ending in _INSTITUTION_LEGAL_ENDINGS):
        return None
    if any(other in candidate for other in _AGREEMENT_ADJACENT_LABELS):
        return None
    return candidate


def deterministic_liability_business_type_candidate(value: Any) -> str | None:
    """Return one closed business type after deleting one edge noise glyph."""

    compact = _compact(value)
    candidates: set[str] = set()
    for candidate in _LIABILITY_BUSINESS_TYPES:
        if compact == candidate:
            continue
        if compact.endswith(candidate):
            residue = compact[: -len(candidate)]
        elif compact.startswith(candidate):
            residue = compact[len(candidate) :]
        else:
            continue
        if re.fullmatch(r"[A-Za-z\u3400-\u9fff]", residue):
            candidates.add(candidate)
    return next(iter(candidates)) if len(candidates) == 1 else None


def liability_business_type_is_valid(value: Any) -> bool:
    return _compact(value) in _LIABILITY_BUSINESS_TYPES


def separated_leading_han_company_boundary(value: Any) -> bool:
    """Identify an ambiguous leading-Han boundary on one company-name field."""

    text = _plain(value)
    match = re.fullmatch(r"[\u3400-\u9fff]\s+([\u3400-\u9fffA-Za-z0-9（）()·-]{4,119})", text)
    if match is None:
        return False
    remainder = match.group(1)
    return any(remainder.endswith(ending) for ending in _INSTITUTION_LEGAL_ENDINGS)


__all__ = [
    "bounded_inquiry_sequence_noise_candidate",
    "deterministic_agreement_institution_candidate",
    "deterministic_inquiry_date_candidate",
    "deterministic_liability_business_type_candidate",
    "liability_business_type_is_valid",
    "separated_leading_han_company_boundary",
]
