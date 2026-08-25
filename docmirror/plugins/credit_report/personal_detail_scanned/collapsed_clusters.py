# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure decoders for canonical PBOC fields collapsed into whole OCR cells.

The decoders in this module have no parser or projection dependencies.  They
only emit a business field when the finite vocabulary, field type, and
canonical cluster topology leave one possible segmentation.  Everything else
remains in ``unresolved_residue`` for the caller to report as uncertainty.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

from docmirror.plugins.credit_report.currency_codes import (
    CURRENCY_CODE_BY_ALIAS,
    ISO_4217_CURRENT_CODES,
)

ClusterValue = str | int
LabeledClusterKind = Literal["account_terms", "special_transaction", "large_installment"]


@dataclass(frozen=True, slots=True)
class ClusterDecodeResult:
    """Uniquely decoded fields plus source text that could not be assigned."""

    fields: dict[str, ClusterValue]
    unresolved_residue: str
    unresolved_fields: tuple[str, ...]


_EMPLOYER_TYPES = (
    "机关事业单位",
    "国有企业",
    "集体企业",
    "外资企业",
    "私营企业",
    "民营企业",
    "个体工商户",
    "个体、私营企业",
    "其他（包括三资企业、民营企业、民间团体等）",
    "未知",
)
_ORGANIZATION_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "分公司",
    "合作联社",
    "合作社",
    "委员会",
    "事务所",
    "工作室",
    "集团",
    "企业",
    "银行",
    "信用社",
    "中心",
    "学校",
    "学院",
    "医院",
    "机关",
    "单位",
    "商店",
    "经营部",
    "公司",
    "工厂",
    "厂",
    "局",
    "所",
    "社",
    "院",
)
_ADDRESS_ENDINGS = (
    "单元",
    "室",
    "层",
    "楼",
    "栋",
    "幢",
    "座",
    "号",
    "路",
    "街",
    "道",
    "巷",
    "村",
    "镇",
    "乡",
    "区",
    "县",
    "市",
    "省",
)
_ADDRESS_MARKER_RE = re.compile(r"[省市区县镇乡街路道巷村号楼栋幢单元室层座]")
_PHONE_RE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")


@dataclass(frozen=True, slots=True)
class _SpanValue:
    start: int
    end: int
    raw: str
    value: ClusterValue


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    field_name: str
    label: str
    value_type: Literal["date", "money", "integer", "currency", "text"]


_CLUSTER_SPECS: dict[LabeledClusterKind, tuple[_FieldSpec, ...]] = {
    "account_terms": (
        _FieldSpec("due_date", "到期日期", "date"),
        _FieldSpec("loan_amount", "借款金额", "money"),
        _FieldSpec("currency", "账户币种", "currency"),
    ),
    "special_transaction": (
        _FieldSpec("transaction_type", "特殊交易类型", "text"),
        _FieldSpec("event_date", "发生日期", "date"),
        _FieldSpec("changed_months", "变更月数", "integer"),
        _FieldSpec("amount", "发生金额", "money"),
        _FieldSpec("details", "明细记录", "text"),
    ),
    "large_installment": (
        _FieldSpec("installment_limit", "大额专项分期额度", "money"),
        _FieldSpec("effective_date", "分期额度生效日期", "date"),
        _FieldSpec("expiry_date", "分期额度到期日期", "date"),
        _FieldSpec("used_installment_amount", "已用分期金额", "money"),
    ),
}
_ALLOWED_LABEL_SEQUENCES: dict[LabeledClusterKind, frozenset[tuple[str, ...]]] = {
    "account_terms": frozenset({("到期日期", "借款金额", "账户币种")}),
    "special_transaction": frozenset(
        {
            ("特殊交易类型", "发生日期", "变更月数", "发生金额", "明细记录"),
            ("变更月数", "发生日期", "发生金额", "明细记录", "特殊交易类型"),
        }
    ),
    "large_installment": frozenset(
        {
            ("大额专项分期额度", "分期额度生效日期"),
            ("分期额度生效日期", "大额专项分期额度"),
            ("分期额度到期日期", "已用分期金额"),
            ("大额专项分期额度", "分期额度生效日期", "分期额度到期日期", "已用分期金额"),
        }
    ),
}
_ALL_CLUSTER_LABELS = tuple(
    sorted(
        {spec.label for specs in _CLUSTER_SPECS.values() for spec in specs},
        key=len,
        reverse=True,
    )
)

_DATE_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}[./-]\d{1,2}(?:[./-]\d{1,2})?(?!\d)")
_NUMBER_RE = re.compile(r"(?<![\dA-Za-z\u3400-\u9fff])(?:\d{1,3}(?:,\d{3})+|\d+)(?![\dA-Za-z\u3400-\u9fff])")
_CURRENCY_TERMS = {
    **CURRENCY_CODE_BY_ALIAS,
    **{code: code for code in ISO_4217_CURRENT_CODES},
    "RMB": "CNY",
}


def _compact_employment(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unconsumed_text(text: str, spans: Sequence[tuple[int, int]]) -> str:
    if not text:
        return ""
    consumed = [False] * len(text)
    for start, end in spans:
        for index in range(max(0, start), min(len(text), end)):
            consumed[index] = True
    residue = "".join(" " if consumed[index] else char for index, char in enumerate(text))
    return _clean_text(residue).strip(" ,，;；|/")


def _unique_vocabulary_span(text: str, vocabulary: Sequence[str]) -> tuple[int, int, str] | None:
    matches: list[tuple[int, int, str]] = []
    for candidate in vocabulary:
        marker = _compact_employment(candidate)
        start = 0
        while marker and (position := text.find(marker, start)) >= 0:
            matches.append((position, position + len(marker), candidate))
            start = position + 1
    maximal = [
        match
        for match in matches
        if not any(
            other[0] <= match[0]
            and match[1] <= other[1]
            and (other[1] - other[0]) > (match[1] - match[0])
            for other in matches
        )
    ]
    return maximal[0] if len(maximal) == 1 else None


def _is_organization(value: str) -> bool:
    return (
        len(re.findall(r"[\u3400-\u9fff]", value)) >= 3
        and value.endswith(_ORGANIZATION_SUFFIXES)
        and not _PHONE_RE.search(value)
    )


def _is_address(value: str) -> bool:
    return (
        len(value) >= 4
        and bool(_ADDRESS_MARKER_RE.search(value))
        and value.endswith(_ADDRESS_ENDINGS)
        and not value.endswith(_ORGANIZATION_SUFFIXES)
        and not _PHONE_RE.search(value)
    )


def _organization_endpoints(value: str) -> list[int]:
    endpoints: set[int] = set()
    for suffix in _ORGANIZATION_SUFFIXES:
        start = 0
        while (position := value.find(suffix, start)) >= 0:
            end = position + len(suffix)
            if _is_organization(value[:end]):
                endpoints.add(end)
            start = position + 1
    return sorted(endpoints)


def decode_employment_basic_cluster(value: object) -> ClusterDecodeResult:
    """Decode the four canonical employment-basic fields from one OCR blob."""

    text = _compact_employment(value)
    if not text:
        return ClusterDecodeResult({}, "", ("employer", "employer_type", "employer_address", "employer_phone"))

    fields: dict[str, ClusterValue] = {}
    consumed: list[tuple[int, int]] = []
    phone_matches = list(_PHONE_RE.finditer(text))
    phone_span: tuple[int, int] | None = None
    if len(phone_matches) == 1:
        phone = phone_matches[0]
        phone_span = phone.span()
        fields["employer_phone"] = phone.group(0)
        consumed.append(phone_span)

    employer_type = _unique_vocabulary_span(text, _EMPLOYER_TYPES)
    if employer_type is not None:
        type_start, type_end, type_value = employer_type
        employer_candidate = text[:type_start]
        cluster_end = phone_span[0] if phone_span is not None else len(text)
        organization_endpoints = _organization_endpoints(text[:cluster_end])
        type_is_nested = any(endpoint > type_end for endpoint in organization_endpoints) or (
            not _is_organization(employer_candidate)
            and type_end in organization_endpoints
        )
        if not type_is_nested:
            fields["employer_type"] = type_value
            consumed.append((type_start, type_end))
            if _is_organization(employer_candidate):
                fields["employer"] = employer_candidate
                consumed.append((0, type_start))
            tail_end = phone_span[0] if phone_span is not None and phone_span[0] > type_end else len(text)
            address_candidate = text[type_end:tail_end]
            if _is_address(address_candidate):
                fields["employer_address"] = address_candidate
                consumed.append((type_end, tail_end))
    else:
        cluster_end = phone_span[0] if phone_span is not None else len(text)
        cluster = text[:cluster_end]
        segmentations: list[tuple[int, str, str | None]] = []
        for endpoint in _organization_endpoints(cluster):
            employer_candidate = cluster[:endpoint]
            address_candidate = cluster[endpoint:]
            address = address_candidate if _is_address(address_candidate) else None
            segmentations.append((2 if address is not None else 1, employer_candidate, address))
        if segmentations:
            best_score = max(item[0] for item in segmentations)
            best = [item for item in segmentations if item[0] == best_score]
            distinct = {(employer, address) for _score, employer, address in best}
            if len(distinct) == 1:
                employer, address = next(iter(distinct))
                fields["employer"] = employer
                employer_end = len(employer)
                consumed.append((0, employer_end))
                if address is not None:
                    fields["employer_address"] = address
                    consumed.append((employer_end, cluster_end))

    field_order = ("employer", "employer_type", "employer_address", "employer_phone")
    return ClusterDecodeResult(
        fields,
        _unconsumed_text(text, consumed),
        tuple(field for field in field_order if field not in fields),
    )


def _leading_label_sequence(text: str) -> tuple[tuple[str, ...], list[tuple[int, int]], int]:
    labels: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        whitespace = re.match(r"[\s|,，;；/]*", text[cursor:])
        cursor += len(whitespace.group(0)) if whitespace else 0
        label = next((candidate for candidate in _ALL_CLUSTER_LABELS if text.startswith(candidate, cursor)), None)
        if label is None:
            break
        labels.append(label)
        spans.append((cursor, cursor + len(label)))
        cursor += len(label)
    return tuple(labels), spans, cursor


def _typed_candidates(text: str, start: int, value_type: str) -> list[_SpanValue]:
    if value_type == "date":
        candidates: list[_SpanValue] = []
        for match in _DATE_RE.finditer(text, start):
            parts = re.split(r"[./-]", match.group(0))
            normalized = f"{int(parts[0]):04d}-{int(parts[1]):02d}"
            if len(parts) == 3:
                normalized += f"-{int(parts[2]):02d}"
            candidates.append(_SpanValue(match.start(), match.end(), match.group(0), normalized))
        return candidates
    if value_type == "currency":
        candidates = []
        for raw, normalized in _CURRENCY_TERMS.items():
            offset = start
            while (position := text.find(raw, offset)) >= 0:
                end = position + len(raw)
                if not raw.isascii() or (
                    (position == 0 or not text[position - 1].isascii() or not text[position - 1].isalnum())
                    and (end == len(text) or not text[end].isascii() or not text[end].isalnum())
                ):
                    candidates.append(_SpanValue(position, end, raw, normalized))
                offset = position + 1
        # 人民币 is nested inside 人民币元; retain only maximal spans.
        return [
            candidate
            for candidate in candidates
            if not any(
                other.start <= candidate.start
                and candidate.end <= other.end
                and (other.end - other.start) > (candidate.end - candidate.start)
                for other in candidates
            )
        ]
    raise ValueError(f"Unsupported independent candidate type: {value_type}")


def _numeric_candidates(text: str, start: int, date_spans: Sequence[_SpanValue]) -> list[_SpanValue]:
    candidates: list[_SpanValue] = []
    for match in _NUMBER_RE.finditer(text, start):
        if any(match.start() < date.end and date.start < match.end() for date in date_spans):
            continue
        raw = match.group(0)
        candidates.append(_SpanValue(match.start(), match.end(), raw, int(raw.replace(",", ""))))
    return candidates


def _stable_assignments(
    specs: Sequence[_FieldSpec],
    candidates: dict[str, list[_SpanValue]],
) -> dict[str, _SpanValue]:
    active = [spec for spec in specs if candidates.get(spec.field_name)]
    if not active:
        return {}
    solutions: list[dict[str, _SpanValue]] = []
    for combination in product(*(candidates[spec.field_name] for spec in active)):
        spans = {(candidate.start, candidate.end) for candidate in combination}
        if len(spans) != len(combination):
            continue
        solutions.append(dict(zip((spec.field_name for spec in active), combination, strict=True)))
    if not solutions:
        return {}
    stable: dict[str, _SpanValue] = {}
    for spec in active:
        observed = {
            (solution[spec.field_name].start, solution[spec.field_name].end)
            for solution in solutions
        }
        if len(observed) == 1:
            stable[spec.field_name] = solutions[0][spec.field_name]
    return stable


def _decode_labeled_fragment(text: str, kind: LabeledClusterKind) -> ClusterDecodeResult:
    specs = _CLUSTER_SPECS[kind]
    labels, label_spans, value_start = _leading_label_sequence(text)
    if labels not in _ALLOWED_LABEL_SEQUENCES[kind]:
        observed = {spec.field_name for spec in specs if spec.label in labels}
        unresolved = tuple(spec.field_name for spec in specs if spec.field_name in observed)
        return ClusterDecodeResult({}, text, unresolved)

    specs_by_label = {spec.label: spec for spec in specs}
    observed_specs = [specs_by_label[label] for label in labels]
    date_candidates = _typed_candidates(text, value_start, "date")
    candidates: dict[str, list[_SpanValue]] = {}
    for spec in observed_specs:
        if spec.value_type == "date":
            candidates[spec.field_name] = date_candidates
        elif spec.value_type == "currency":
            candidates[spec.field_name] = _typed_candidates(text, value_start, "currency")

    numeric = _numeric_candidates(text, value_start, date_candidates)
    for spec in observed_specs:
        if spec.value_type == "money":
            candidates[spec.field_name] = numeric
        elif spec.value_type == "integer":
            candidates[spec.field_name] = [
                candidate
                for candidate in numeric
                if "," not in candidate.raw and isinstance(candidate.value, int) and candidate.value <= 999
            ]

    nontext_specs = [spec for spec in observed_specs if spec.value_type != "text"]
    assigned = _stable_assignments(nontext_specs, candidates)
    consumed = [*label_spans, *((value.start, value.end) for value in assigned.values())]
    fields: dict[str, ClusterValue] = {
        field_name: candidate.value for field_name, candidate in assigned.items()
    }

    text_specs = [spec for spec in observed_specs if spec.value_type == "text"]
    if text_specs and len(assigned) == len(nontext_specs):
        typed_spans = sorted(assigned.values(), key=lambda candidate: candidate.start)
        if typed_spans:
            middle_gaps = [
                text[left.end : right.start]
                for left, right in zip(typed_spans, typed_spans[1:])
            ]
            if all(not _clean_text(gap).strip(" ,，;；|/") for gap in middle_gaps):
                prefix = text[value_start : typed_spans[0].start].strip(" ,，;；|/")
                suffix = text[typed_spans[-1].end :].strip(" ,，;；|/")
                if (
                    {spec.field_name for spec in text_specs} == {"transaction_type", "details"}
                    and prefix
                    and suffix
                    and re.search(r"[\u3400-\u9fff]", prefix)
                    and re.search(r"[\u3400-\u9fff]", suffix)
                ):
                    fields["transaction_type"] = _clean_text(prefix)
                    fields["details"] = _clean_text(suffix)
                    consumed.extend(
                        (
                            (value_start, typed_spans[0].start),
                            (typed_spans[-1].end, len(text)),
                        )
                    )

    unresolved = tuple(spec.field_name for spec in observed_specs if spec.field_name not in fields)
    return ClusterDecodeResult(fields, _unconsumed_text(text, consumed), unresolved)


def decode_labeled_cluster(
    value: object | Sequence[object],
    *,
    kind: LabeledClusterKind,
) -> ClusterDecodeResult:
    """Decode one or more exact canonical label/value clusters.

    Multiple fragments are useful for the two independently collapsed large-
    installment pairs.  Conflicting observations are withheld rather than
    resolved by fragment order.
    """

    raw_fragments = list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    fragments = [_clean_text(fragment) for fragment in raw_fragments if _clean_text(fragment)]
    if not fragments:
        return ClusterDecodeResult({}, "", ())

    decoded = [_decode_labeled_fragment(fragment, kind) for fragment in fragments]
    observations: dict[str, list[ClusterValue]] = {}
    for result in decoded:
        for field_name, field_value in result.fields.items():
            observations.setdefault(field_name, []).append(field_value)

    fields: dict[str, ClusterValue] = {}
    conflicts: list[str] = []
    conflict_residue: list[str] = []
    for field_name, values in observations.items():
        distinct = list(dict.fromkeys(values))
        if len(distinct) == 1:
            fields[field_name] = distinct[0]
        else:
            conflicts.append(field_name)
            conflict_residue.extend(str(item) for item in distinct)

    unresolved = tuple(
        dict.fromkeys(
            [field for result in decoded for field in result.unresolved_fields]
            + conflicts
        )
    )
    residue = _clean_text(
        " ".join(
            [result.unresolved_residue for result in decoded if result.unresolved_residue]
            + conflict_residue
        )
    )
    return ClusterDecodeResult(fields, residue, unresolved)


__all__ = [
    "ClusterDecodeResult",
    "decode_employment_basic_cluster",
    "decode_labeled_cluster",
]
