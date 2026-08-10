# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed decoding for packed PBOC repayment-responsibility rows.

The helper deliberately performs no fuzzy OCR correction. It accepts only a
complete canonical header (plus the small, explicit label-alias set below).
Within that contract, uniquely typed and positionally supported value spans may
survive unrelated OCR residue; every unconsumed or ambiguous span keeps the row
explicitly unresolved so callers cannot silently project guessed values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping, Sequence

from docmirror.plugins.credit_report.currency_codes import (
    CURRENCY_CODE_BY_ALIAS,
    ISO_4217_CURRENT_CODES,
)

CANONICAL_LIABILITY_LABELS = (
    "管理机构",
    "业务种类",
    "开立日期",
    "到期日期",
    "责任人类型",
    "还款责任金额",
    "币种",
    "保证合同编号",
)

_LABEL_ALIASES = {
    "贵任人类型": "责任人类型",
    "还款贵任金额": "还款责任金额",
    "成立日期": "开立日期",
}
_SOURCE_LABELS = frozenset((*CANONICAL_LIABILITY_LABELS, *_LABEL_ALIASES))
_HEADER_LABEL_PATTERN = re.compile(
    "|".join(re.escape(label) for label in sorted(_SOURCE_LABELS, key=len, reverse=True))
)
_HEADER_SEPARATOR_PATTERN = re.compile(r"[\s|｜:：,，;；]*")

# These are field vocabularies from the fixed PBOC report family, not observed
# institution/value registries.  Unknown printed values are intentionally not
# inferred from neighboring tokens.
_BUSINESS_TYPES = frozenset(
    {
        "贷款",
        "个人住房贷款",
        "个人商用房贷款",
        "个人经营性贷款",
        "个人消费贷款",
        "其他个人消费贷款",
        "个人汽车贷款",
        "助学贷款",
        "农户贷款",
        "其他贷款",
        "循环贷款",
        "融资租赁",
        "融资租赁业务",
        "贷记卡",
        "准贷记卡",
    }
)
_RESPONSIBILITY_TYPES = frozenset(
    {
        "本人",
        "保证",
        "保证人",
        "担保",
        "担保人",
        "共同借款人",
        "共同还款人",
        "抵押",
        "抵押人",
        "质押",
        "质押人",
        "其他",
        "未知",
    }
)
_CURRENCIES = {
    **CURRENCY_CODE_BY_ALIAS,
    **{code: code for code in ISO_4217_CURRENT_CODES},
    "RMB": "CNY",
}

_DATE_PATTERN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[.年/-]\s*(\d{1,2})\s*[.月/-]\s*(\d{1,2})\s*日?(?!\d)"
)
_INSTITUTION_CHARACTERS = re.compile(r"[\u3400-\u9fffA-Za-z0-9（）()·&＆\-]+")
_INSTITUTION_ENDING = re.compile(
    r"(?:有限责任公司|股份有限公司|有限公司|农村信用合作联社|信用合作联社|农村信用合作社|"
    r"信用社|银行|分行|支行|管理中心|中心)$"
)
_AMOUNT_PATTERN = re.compile(r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)")
_AMOUNT_TOKEN_PATTERN = re.compile(
    r"(?<![A-Z0-9,+\-])(?:[1-9]\d{0,2}(?:,\d{3})+|0|[1-9]\d*)(?![A-Z0-9,])"
)
_IDENTIFIER_PATTERN = re.compile(r"[A-Z0-9*][A-Z0-9*/_\-]{7,99}")
_LAYOUT_SEPARATORS = re.compile(r"[\s|｜]+")

LiabilityValue = str | int
LiabilityStatusField = Literal["overdue_months", "repayment_status_code"]


@dataclass(frozen=True, slots=True)
class PackedLiabilityDecode:
    """Typed result for one packed liability header/value pair."""

    fields: Mapping[str, LiabilityValue]
    normalized_labels: tuple[str, ...]
    unresolved_reason: str | None = None

    @property
    def resolved(self) -> bool:
        """Return whether every slot was uniquely typed with no OCR residue."""

        return self.unresolved_reason is None


def _line_text(line: str | Sequence[str]) -> str:
    if isinstance(line, str):
        return line
    return " | ".join(str(value) for value in line)


def normalize_packed_liability_header(
    header_line: str | Sequence[str],
) -> tuple[str, ...] | None:
    """Return the canonical eight labels only for a complete unique header.

    Text other than layout separators and the exact known labels is rejected.
    Consequently, a new OCR misspelling cannot be treated as a canonical slot
    without an explicit future contract change.
    """

    text = _line_text(header_line)
    matches = list(_HEADER_LABEL_PATTERN.finditer(text))
    normalized = tuple(_LABEL_ALIASES.get(match.group(0), match.group(0)) for match in matches)
    if normalized != CANONICAL_LIABILITY_LABELS:
        return None

    cursor = 0
    for match in matches:
        if _HEADER_SEPARATOR_PATTERN.fullmatch(text[cursor : match.start()]) is None:
            return None
        cursor = match.end()
    if _HEADER_SEPARATOR_PATTERN.fullmatch(text[cursor:]) is None:
        return None
    return normalized


def _unresolved(
    reason: str,
    normalized_labels: tuple[str, ...] = (),
) -> PackedLiabilityDecode:
    return PackedLiabilityDecode(
        fields={},
        normalized_labels=normalized_labels,
        unresolved_reason=reason,
    )


def _compact_layout(text: str, *, uppercase: bool = False) -> str:
    compact = _LAYOUT_SEPARATORS.sub("", text)
    return compact.upper() if uppercase else compact


def _valid_institution(value: str) -> bool:
    if not 4 <= len(value) <= 100:
        return False
    if _INSTITUTION_CHARACTERS.fullmatch(value) is None:
        return False
    if len(re.findall(r"[\u3400-\u9fff]", value)) < 2:
        return False
    return _INSTITUTION_ENDING.search(value) is not None


def _split_institution_and_business(
    prefix: str,
) -> tuple[dict[str, LiabilityValue], str]:
    """Retain only unique exact prefix fields and return unconsumed residue."""

    compact = _compact_layout(prefix)
    business_matches = [value for value in _BUSINESS_TYPES if compact.endswith(value)]
    if not business_matches:
        return {}, compact

    longest = max(len(value) for value in business_matches)
    business_candidates = {value for value in business_matches if len(value) == longest}
    if len(business_candidates) != 1:
        return {}, compact
    business_type = next(iter(business_candidates))
    institution_and_residue = compact[: -len(business_type)]

    institution_endpoints = [
        end
        for end in range(4, len(institution_and_residue) + 1)
        if _valid_institution(institution_and_residue[:end])
    ]
    fields: dict[str, LiabilityValue] = {"business_type": business_type}
    if not institution_endpoints:
        return fields, institution_and_residue

    # A legal name may itself contain an earlier suffix (e.g. ``银行`` inside
    # ``银行股份有限公司``). The last legal ending is the only positionally
    # complete candidate before the exact business token.
    institution = institution_and_residue[: max(institution_endpoints)]
    fields["institution"] = institution
    return fields, institution_and_residue[len(institution) :]


def _normalized_date(match: re.Match[str]) -> str | None:
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _parse_amount(value: str) -> int | None:
    if _AMOUNT_PATTERN.fullmatch(value) is None:
        return None
    return int(value.replace(",", ""))


def _valid_identifier(value: str) -> bool:
    return _IDENTIFIER_PATTERN.fullmatch(value) is not None and any(character.isdigit() for character in value)


def _unique_currency_span(
    compact: str,
    *,
    source_text: str,
) -> tuple[int, int, str] | None:
    matches: set[tuple[int, int, str]] = set()
    for token, currency_code in _CURRENCIES.items():
        if token.isascii():
            continue
        start = compact.find(token)
        while start >= 0:
            end = start + len(token)
            left = compact[start - 1] if start > 0 else ""
            right = compact[end] if end < len(compact) else ""
            ascii_boundary_valid = not token.isascii() or (
                (not left or not left.isascii() or not left.isalnum())
                and (not right or not right.isascii() or not right.isalnum())
            )
            chinese_boundary_valid = not re.fullmatch(
                r"[\u3400-\u9fff]", left
            ) and not re.fullmatch(r"[\u3400-\u9fff]", right)
            if ascii_boundary_valid and chinese_boundary_valid:
                matches.add((start, end, currency_code))
            start = compact.find(token, start + 1)

    upper_source = source_text.upper()
    for token, currency_code in _CURRENCIES.items():
        if not token.isascii():
            continue
        for match in re.finditer(
            rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])",
            upper_source,
        ):
            start = len(_compact_layout(upper_source[: match.start()], uppercase=True))
            end = start + len(token)
            matches.add((start, end, currency_code))
    if not matches:
        return None

    # Prefer the longest exact token at an identical start (e.g. 人民币元 over
    # 人民币), but never choose between separate occurrences.
    by_start: dict[int, tuple[int, int, str]] = {}
    for match in matches:
        existing = by_start.get(match[0])
        if existing is None or match[1] - match[0] > existing[1] - existing[0]:
            by_start[match[0]] = match
    return next(iter(by_start.values())) if len(by_start) == 1 else None


def _identifier_span(
    compact: str,
    *,
    search_start: int,
) -> tuple[int, int, str] | None:
    """Find a typed identifier suffix without absorbing Chinese OCR residue."""

    suffix = compact[search_start:]
    if not suffix:
        return None
    last_non_identifier = max(
        (index for index, value in enumerate(suffix) if not re.fullmatch(r"[A-Z0-9*/_\-]", value)),
        default=-1,
    )
    start = search_start + last_non_identifier + 1
    identifier = compact[start:]
    if not _valid_identifier(identifier):
        return None
    return start, len(compact), identifier


def _split_typed_tail(tail: str) -> tuple[dict[str, LiabilityValue], str]:
    """Decode independent exact tail spans and return all leftover text."""

    compact = _compact_layout(tail, uppercase=True)
    fields: dict[str, LiabilityValue] = {}
    consumed = [False] * len(compact)

    responsibility_matches = [value for value in _RESPONSIBILITY_TYPES if compact.startswith(value)]
    if responsibility_matches:
        longest = max(len(value) for value in responsibility_matches)
        candidates = {value for value in responsibility_matches if len(value) == longest}
        if len(candidates) == 1:
            responsibility_type = next(iter(candidates))
            fields["responsibility_type"] = responsibility_type
            consumed[: len(responsibility_type)] = [True] * len(responsibility_type)

    currency_span = _unique_currency_span(compact, source_text=tail)
    if currency_span is not None:
        currency_start, currency_end, currency_code = currency_span
        fields["currency"] = currency_code
        consumed[currency_start:currency_end] = [True] * (currency_end - currency_start)

    identifier_search_start = currency_span[1] if currency_span is not None else 0
    identifier_span = _identifier_span(compact, search_start=identifier_search_start)
    if identifier_span is not None:
        identifier_start, identifier_end, identifier = identifier_span
        fields["contract_number"] = identifier
        consumed[identifier_start:identifier_end] = [True] * (identifier_end - identifier_start)

    amount_limit = min(
        value
        for value in (
            currency_span[0] if currency_span is not None else len(compact),
            identifier_span[0] if identifier_span is not None else len(compact),
        )
    )
    amount_matches = [
        match
        for match in _AMOUNT_TOKEN_PATTERN.finditer(compact, 0, amount_limit)
        if _parse_amount(match.group(0)) is not None
    ]
    if len(amount_matches) == 1:
        amount_match = amount_matches[0]
        amount = _parse_amount(amount_match.group(0))
        if amount is not None:
            fields["responsibility_amount"] = amount
            consumed[amount_match.start() : amount_match.end()] = [True] * (
                amount_match.end() - amount_match.start()
            )

    residue = "".join(value for index, value in enumerate(compact) if not consumed[index])
    return fields, residue


def decode_packed_liability_row(
    header_line: str | Sequence[str],
    value_line: str | Sequence[str],
) -> PackedLiabilityDecode:
    """Decode exact typed spans under one canonical liability-row contract."""

    normalized_labels = normalize_packed_liability_header(header_line)
    if normalized_labels is None:
        return _unresolved("header_not_canonical")

    text = _line_text(value_line)
    date_matches = list(_DATE_PATTERN.finditer(text))
    if len(date_matches) != 2:
        return _unresolved("date_segmentation_not_unique", normalized_labels)
    open_date = _normalized_date(date_matches[0])
    due_date = _normalized_date(date_matches[1])
    if open_date is None or due_date is None:
        return _unresolved("date_value_invalid", normalized_labels)

    prefix_fields, prefix_residue = _split_institution_and_business(
        text[: date_matches[0].start()]
    )
    interdate_text = text[date_matches[0].end() : date_matches[1].start()]
    interdate_residue = (
        ""
        if _HEADER_SEPARATOR_PATTERN.fullmatch(interdate_text) is not None
        else _compact_layout(interdate_text)
    )
    tail_fields, tail_residue = _split_typed_tail(text[date_matches[1].end() :])
    fields: dict[str, LiabilityValue] = {
        **prefix_fields,
        "open_date": open_date,
        "due_date": due_date,
        **tail_fields,
    }
    residue = "".join((prefix_residue, interdate_residue, tail_residue))
    unresolved_reason: str | None = None
    if residue:
        unresolved_reason = "typed_spans_with_ocr_residue"
    elif len(fields) != len(CANONICAL_LIABILITY_LABELS):
        unresolved_reason = "typed_spans_incomplete_or_ambiguous"
    return PackedLiabilityDecode(
        fields=fields,
        normalized_labels=normalized_labels,
        unresolved_reason=unresolved_reason,
    )


def liability_status_projection_field(source_label: str) -> LiabilityStatusField | None:
    """Preserve the distinct business meaning of the two printed status slots."""

    compact = re.sub(r"\s+", "", str(source_label or ""))
    if compact == "逾期月数":
        return "overdue_months"
    if compact == "还款状态":
        return "repayment_status_code"
    return None


__all__ = [
    "CANONICAL_LIABILITY_LABELS",
    "PackedLiabilityDecode",
    "decode_packed_liability_row",
    "liability_status_projection_field",
    "normalize_packed_liability_header",
]
