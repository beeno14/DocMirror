# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-directed profile extraction for Candidate B.

Profile values are selected from the registered canonical page plane.  The
extractor never asks a prose/generic credit-report parser for a second opinion:
table cells and full-page OCR lines are merely two observations of the same
canonical page.  Conflicting or invalid observations are retained as review
evidence and are not emitted as normalized business values.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
    record_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
    is_explicit_source_absence,
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "gender": ("性别",),
    "birth_date": ("出生日期", "出生年月"),
    "marital_status": ("婚姻状况",),
    "employment_status": ("就业状况",),
    "education_level": ("学历",),
    "degree": ("学位",),
    "nationality": ("国籍", "国籍/地区"),
    "mobile_phone": ("手机号码", "移动电话", "手机号"),
    "work_phone": ("单位电话",),
    "residence_phone": ("住宅电话",),
    "email": ("电子邮箱", "邮箱"),
    "mailing_address": ("通讯地址", "通信地址"),
    "household_address": ("户籍地址",),
}
_LABEL_TO_FIELD = {
    re.sub(r"[\s:：，,；;()（）\[\]【】]", "", alias): field
    for field, aliases in _ALIASES.items()
    for alias in aliases
}
_VOCAB_ROLES = {
    "gender": "gender",
    "marital_status": "marital_status",
    "employment_status": "employment_status",
    "education_level": "education_level",
    "degree": "degree",
}
_PHONE_FIELDS = frozenset({"mobile_phone", "work_phone", "residence_phone"})
_ADDRESS_FIELDS = frozenset({"mailing_address", "household_address"})
_PROFILE_TEMPLATE_ID = "report_header_and_identity"
_PROFILE_TABLE_ANCHORS = (
    "个人基本信息",
    "身份信息",
    "性别",
    "出生日期",
    "出生年月",
    "婚姻状况",
    "就业状况",
    "学历",
    "学位",
    "国籍",
    "国籍/地区",
    "电子邮箱",
)
_NON_PROFILE_TABLE_ANCHOR_GROUPS = (
    ("居住信息",),
    ("职业信息",),
    ("配偶信息",),
    ("居住地址", "居住状况"),
    ("工作单位", "单位地址"),
    ("工作单位", "单位性质"),
    ("工作单位", "职业"),
)
_DATE_RE = re.compile(r"(19|20)\d{2}[./年-]\d{1,2}[./月-]\d{1,2}日?")
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_EMAIL_SEARCH_RE = re.compile(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}")
_ADDRESS_MARKER_RE = re.compile(r"[省市县区镇乡村路街道巷号栋室楼]")
_SYMBOL_NOISE_RE = re.compile(r"[#=*<>]{2,}|[\"“”]{2,}")
_PROVINCE_LEVEL_RE = re.compile(
    r"北京市|天津市|上海市|重庆市|"
    r"(?:河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|"
    r"湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾)省|"
    r"内蒙古自治区|广西壮族自治区|西藏自治区|宁夏回族自治区|新疆维吾尔自治区|"
    r"香港特别行政区|澳门特别行政区"
)
_SUBORDINATE_REGION_PREFIX_RE = re.compile(
    r"^[\u3400-\u9fff]{1,10}(?:市|自治州|地区|盟|县|自治县|区|旗|镇|乡|街道|村)"
)
_ADDRESS_ADJACENT_FIELD_RE = re.compile(
    r"通讯地址|通信地址|户籍地址|"
    r"数据发生机构(?:名称)?|数据提供机构(?:名称)?|"
    r"信息提供机构(?:名称)?|数据报送机构(?:名称)?"
)
_PROFILE_PROVIDER_LABEL = "数据发生机构名称"
_PROFILE_PROVIDER_SUFFIX_RE = re.compile(
    r"[A-Za-z0-9\u3400-\u9fff（）()·]{2,100}(?:银行|支行|分行|信用社|"
    r"农村信用合作联社|农村信用社联合社|股份有限公司|"
    r"有限责任公司|有限公司|征信中心|信用卡中心|个人信贷部|管理中心)"
)


def _compact(value: Any) -> str:
    return re.sub(r"[\s:：，,；;()（）\[\]【】]", "", str(value or "")).strip()


def _address_has_region_or_provider_contamination(value: str) -> bool:
    """Detect a second address/provider field without rejecting proper names.

    Repeating a province name inside a community or government-compound name
    is legitimate.  It becomes a second region sequence only when a repeated
    province-level anchor starts another subordinate administrative chain.
    A different province-level anchor is independently conflicting.
    """

    compact = _compact(value)
    if _ADDRESS_ADJACENT_FIELD_RE.search(compact):
        return True
    anchors = list(_PROVINCE_LEVEL_RE.finditer(compact))
    if len(anchors) < 2:
        return False
    first_region = anchors[0].group(0)
    for anchor in anchors[1:]:
        if anchor.group(0) != first_region:
            return True
        following = compact[anchor.end() :]
        if _SUBORDINATE_REGION_PREFIX_RE.match(following):
            return True
    return False


def _address_candidate_valid(value: str) -> bool:
    text = str(value or "").strip()
    compact = _compact(text)
    return bool(
        len(compact) >= 6
        and _ADDRESS_MARKER_RE.search(text)
        and not _SYMBOL_NOISE_RE.search(text)
        and not _address_has_region_or_provider_contamination(text)
    )


def _profile_provider_evidence(value: str) -> dict[str, Any]:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        institution_slot_is_unambiguous,
        normalize_institution_name,
    )

    raw = str(value or "").strip()
    if is_explicit_source_absence(raw):
        return {"raw": raw, "normalized_value": None, "observation_status": "source_absent"}
    normalized = normalize_institution_name(raw)
    valid = bool(
        institution_slot_is_unambiguous(raw)
        and institution_slot_is_unambiguous(normalized)
        and _PROFILE_PROVIDER_SUFFIX_RE.fullmatch(_compact(normalized))
    )
    return {
        "raw": raw,
        "normalized_value": normalized if valid else None,
        "observation_status": (
            "normalized" if valid and raw != normalized else "observed" if valid else "unreadable"
        ),
    }


def _split_profile_address_provider(value: str) -> dict[str, Any] | None:
    """Decode one exact address/provider boundary without losing either side."""

    raw = str(value or "").strip()
    marker_count = raw.count(_PROFILE_PROVIDER_LABEL)
    if marker_count == 0:
        return None
    if marker_count != 1:
        return {
            "address": None,
            "provider_evidence": {
                "raw": raw,
                "normalized_value": None,
                "observation_status": "ambiguous",
            },
        }
    address_raw, provider_raw = (part.strip() for part in raw.split(_PROFILE_PROVIDER_LABEL, 1))
    return {
        "address": address_raw if _address_candidate_valid(address_raw) else None,
        "provider_evidence": _profile_provider_evidence(provider_raw),
    }


def _rows(table: Any) -> list[list[str]]:
    metadata = getattr(table, "metadata", None) or {}
    raw = metadata.get("raw_rows") if isinstance(metadata, Mapping) else None
    if isinstance(raw, list) and raw:
        return [[str(cell or "") for cell in row] for row in raw if isinstance(row, list)]
    headers = [str(value or "") for value in getattr(table, "headers", None) or []]
    body = [
        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", None) or []]
        for row in getattr(table, "rows", None) or []
    ]
    return ([headers] if headers else []) + body


def _canonical_template_id(page: Any, table: Any) -> str:
    metadata = getattr(table, "metadata", None) or {}
    table_template = str(metadata.get("canonical_template_id") or "") if isinstance(metadata, Mapping) else ""
    page_template = str(getattr(page, "canonical_template_id", "") or "")
    return table_template or page_template


def _contains_marker(cells: Iterable[str], marker: str) -> bool:
    compact_marker = _compact(marker)
    return any(compact_marker in cell for cell in cells)


def _is_profile_table(page: Any, table: Any, rows: list[list[str]]) -> bool:
    """Return whether a table belongs to the canonical subject-profile block.

    The header/identity canonical page also owns residence, employment, and
    spouse tables.  Their phone/address labels must not compete with the
    subject's identity-table fields merely because they share a page.
    """

    if _canonical_template_id(page, table) != _PROFILE_TEMPLATE_ID:
        return False
    cells = tuple(_compact(cell) for row in rows for cell in row if _compact(cell))
    if any(all(_contains_marker(cells, marker) for marker in group) for group in _NON_PROFILE_TABLE_ANCHOR_GROUPS):
        return False
    return any(_contains_marker(cells, marker) for marker in _PROFILE_TABLE_ANCHORS)


def _source_ref(page: Any, table: Any, row: int, col: int) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "source": "candidate_b_canonical_table",
        "logical_page": int(getattr(page, "page_number", 0) or 0),
        "source_page": int(
            getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0
        ),
        "table_id": str(getattr(table, "table_id", "") or ""),
        "row": row,
        "column": col,
        "canonical_row": row,
        "canonical_column": col,
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_header_column",
        "geometry_scope": "canonical_field_slot",
    }
    metadata = getattr(table, "metadata", None) or {}
    bbox = None
    if isinstance(metadata, Mapping):
        cell_bboxes = metadata.get("source_cell_bboxes") or metadata.get("cell_bboxes")
        if (
            isinstance(cell_bboxes, list)
            and 0 <= row < len(cell_bboxes)
            and isinstance(cell_bboxes[row], list)
            and 0 <= col < len(cell_bboxes[row])
        ):
            candidate = cell_bboxes[row][col]
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                bbox = candidate
        evidence_ids = metadata.get("cell_evidence_ids")
        if (
            isinstance(evidence_ids, list)
            and 0 <= row < len(evidence_ids)
            and isinstance(evidence_ids[row], list)
            and 0 <= col < len(evidence_ids[row])
            and isinstance(evidence_ids[row][col], list)
        ):
            ref["evidence_ids"] = [str(value) for value in evidence_ids[row][col] if value]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        ref["bbox"] = list(bbox)
        ref["geometry_scope"] = "cell"
    confidence = None
    if isinstance(metadata, Mapping):
        confidence_rows = metadata.get("cell_confidences")
        if (
            isinstance(confidence_rows, list)
            and 0 <= row < len(confidence_rows)
            and isinstance(confidence_rows[row], list)
            and 0 <= col < len(confidence_rows[row])
        ):
            confidence = confidence_rows[row][col]
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        ref["confidence"] = max(0.0, min(1.0, float(confidence)))
    return ref


def _candidate_valid(field: str, value: str) -> tuple[bool, str | None]:
    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        normalize_pboc_field,
        validate_pboc_field,
    )

    text = str(value or "").strip()
    compact = _compact(text)
    if not compact:
        return False, None
    if field in _VOCAB_ROLES:
        role = _VOCAB_ROLES[field]
        contract = validate_pboc_field(text, role)
        return contract.valid, normalize_pboc_field(text, role) if contract.valid else None
    if field == "birth_date":
        match = _DATE_RE.search(text)
        if not match:
            return False, None
        parts = [int(value) for value in re.findall(r"\d+", match.group(0))[:3]]
        if len(parts) != 3 or not 1 <= parts[1] <= 12 or not 1 <= parts[2] <= 31:
            return False, None
        return True, f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
    if field == "mobile_phone":
        if any(character.isalpha() for character in text):
            return False, None
        digits = re.sub(r"\D", "", text)
        return bool(re.fullmatch(r"1[3-9]\d{9}", digits)), digits if len(digits) == 11 else None
    if field in {"work_phone", "residence_phone"}:
        if any(character.isalpha() for character in text):
            return False, None
        digits = re.sub(r"\D", "", text)
        return 5 <= len(digits) <= 16, text if 5 <= len(digits) <= 16 else None
    if field == "email":
        return bool(_EMAIL_RE.fullmatch(text)), text if _EMAIL_RE.fullmatch(text) else None
    if field in _ADDRESS_FIELDS:
        valid = _address_candidate_valid(text)
        return valid, text if valid else None
    if field == "nationality":
        valid = bool(re.search(r"[\u3400-\u9fffA-Za-z]", compact)) and len(compact) <= 30
        return valid, text if valid else None
    return True, text


def _label_fields(value: Any) -> list[str]:
    """Return canonical roles printed in one possibly collapsed header cell."""

    marker = _compact(value)
    exact = _LABEL_TO_FIELD.get(marker)
    if exact:
        return [exact]
    located: list[tuple[int, int, str]] = []
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            compact_alias = _compact(alias)
            position = marker.find(compact_alias)
            if position >= 0:
                located.append((position, -len(compact_alias), field))
                break
    ordered: list[str] = []
    for _position, _negative_length, field in sorted(located):
        if field not in ordered:
            ordered.append(field)
    return ordered


def _row_label_fields(row: list[str]) -> list[tuple[int, str]]:
    """Apply the closed canonical identity-row signature to damaged headers."""

    by_column = {column: _label_fields(value) for column, value in enumerate(row)}
    observed = {field for fields in by_column.values() for field in fields}
    if {"degree", "education_level", "email"} <= observed and "nationality" not in observed:
        degree_column = next(
            (column for column, fields in by_column.items() if {"degree", "education_level"} <= set(fields)),
            None,
        )
        email_column = next(
            (column for column, fields in by_column.items() if "email" in fields),
            None,
        )
        if degree_column is not None and email_column is not None and email_column - degree_column == 2:
            by_column.setdefault(degree_column + 1, []).append("nationality")
    if "mailing_address" in observed and "household_address" not in observed:
        mailing_column = next(
            (column for column, fields in by_column.items() if "mailing_address" in fields),
            None,
        )
        unknown_columns = [
            column
            for column, fields in by_column.items()
            if mailing_column is not None
            and column > mailing_column
            and str(row[column] or "").strip()
            and not fields
        ]
        if len(unknown_columns) == 1:
            by_column[unknown_columns[0]].append("household_address")
    return [
        (column, field)
        for column, fields in by_column.items()
        for field in fields
    ]


def _field_fragments(field: str, value: str) -> list[tuple[str, str]]:
    """Decode field-specific values from one OCR-collapsed canonical cell."""

    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        normalize_pboc_field,
        pboc_controlled_vocabulary,
        validate_pboc_field,
    )

    text = str(value or "").strip()
    if not text:
        return []
    business_text = (
        text
        if field in _ADDRESS_FIELDS
        else re.split(r"数据发生机构(?:名称)?", text, maxsplit=1)[0].strip()
    )
    if not business_text:
        return []
    text = business_text
    role = _VOCAB_ROLES.get(field)
    if role:
        compact = _compact(text)
        matches: list[tuple[int, int, str]] = []
        for candidate in pboc_controlled_vocabulary(role):
            marker = _compact(candidate)
            position = compact.find(marker)
            if position >= 0:
                matches.append((position, -len(marker), candidate))
        selected: list[str] = []
        for _position, _negative_length, candidate in sorted(matches):
            if any(_compact(candidate) in _compact(existing) for existing in selected):
                continue
            if any(_compact(existing) in _compact(candidate) for existing in selected):
                selected = [existing for existing in selected if _compact(existing) not in _compact(candidate)]
            if candidate not in selected:
                selected.append(candidate)
        if field == "education_level" and any(
            candidate not in {"无", "未知", "其他"} for candidate in selected
        ):
            selected = [candidate for candidate in selected if candidate not in {"无", "未知", "其他"}]
        decoded = [
            (candidate, normalize_pboc_field(candidate, role))
            for candidate in selected
            if validate_pboc_field(candidate, role).valid
        ]
        if decoded:
            return decoded
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            normalize_role_candidate,
        )

        corrected_tokens: list[tuple[str, str]] = []
        for token in re.split(r"\s+", text):
            token = token.strip("：:，,；;()（）[]【】")
            if not token:
                continue
            corrected = normalize_role_candidate(token, role)
            if (
                corrected != token
                and validate_pboc_field(corrected, role).valid
                and corrected not in {value for _raw, value in corrected_tokens}
            ):
                corrected_tokens.append((token, corrected))
        return corrected_tokens
    if field == "birth_date":
        match = _DATE_RE.search(text)
        if match:
            valid, normalized = _candidate_valid(field, match.group(0))
            return [(match.group(0), str(normalized))] if valid and normalized else []
    if field == "email":
        match = _EMAIL_SEARCH_RE.search(text)
        if match:
            valid, normalized = _candidate_valid(field, match.group(0))
            return [(match.group(0), str(normalized))] if valid and normalized else []
    if field == "mobile_phone":
        if any(character.isalpha() for character in text):
            return []
        match = re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text)
        if match:
            return [(match.group(0), match.group(0))]
    if field in {"work_phone", "residence_phone"}:
        if any(character.isalpha() for character in text):
            return []
        match = re.search(r"(?<!\d)(?:\d{3,4}[-—]?)?\d{5,12}(?!\d)", text)
        if match:
            valid, normalized = _candidate_valid(field, match.group(0))
            return [(match.group(0), str(normalized))] if valid and normalized else []
    if field == "nationality":
        match = re.search(r"中国(?:[（(]含港澳台[）)])?", text)
        if match:
            return [(match.group(0), match.group(0))]
    valid, normalized = _candidate_valid(field, text)
    return [(text, str(normalized))] if valid and normalized is not None else []


def _table_candidates(pages: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        for table in getattr(page, "tables", None) or ():
            rows = _rows(table)
            if not _is_profile_table(page, table, rows):
                continue
            for row_index, row in enumerate(rows):
                labels = _row_label_fields(row)
                fields_by_column: dict[int, set[str]] = defaultdict(set)
                for label_column, label_field in labels:
                    fields_by_column[label_column].add(label_field)
                for column, field in labels:
                    choices: list[tuple[str, int, int]] = []
                    next_label_column = next(
                        (candidate_column for candidate_column, _candidate_field in labels if candidate_column > column),
                        len(row),
                    )
                    inline_values = (
                        [
                            (str(row[candidate_column]), row_index, candidate_column)
                            for candidate_column in range(column + 1, next_label_column)
                            if str(row[candidate_column] or "").strip()
                            and not _label_fields(row[candidate_column])
                        ]
                        if len(fields_by_column[column]) == 1
                        else []
                    )
                    # An inline slot is exact only when exactly one physical
                    # cell lies between this label and the next canonical role.
                    if len(inline_values) == 1:
                        choices.extend(inline_values)
                    if row_index + 1 < len(rows) and column < len(rows[row_index + 1]):
                        below = str(rows[row_index + 1][column] or "")
                        if below.strip():
                            choices.append((below, row_index + 1, column))
                    if not choices:
                        # The canonical label is visible but its value cell is
                        # empty/unreadable.  Retain that distinction from a
                        # profile role which is not printed in this report
                        # layout at all; only the former requires review.
                        ref = _source_ref(page, table, row_index, column)
                        ref["field_name"] = field
                        candidates[field].append(
                            {
                                "raw": "",
                                "normalized": None,
                                "valid": False,
                                "source_absent": False,
                                "label_observed": True,
                                "source_refs": [ref],
                                "confidence": ref.get("confidence"),
                            }
                        )
                    for raw, value_row, value_col in choices:
                        if _compact(raw) in _LABEL_TO_FIELD:
                            continue
                        ref = _source_ref(page, table, value_row, value_col)
                        ref["field_name"] = field
                        address_split = (
                            _split_profile_address_provider(raw)
                            if field in _ADDRESS_FIELDS
                            else None
                        )
                        if address_split is not None:
                            provider_evidence = dict(address_split["provider_evidence"])
                            provider_evidence["source_refs"] = [ref]
                            address = address_split.get("address")
                            candidates[field].append(
                                {
                                    "raw": raw.strip(),
                                    "normalized": address,
                                    "valid": address is not None,
                                    "source_absent": False,
                                    "provider_evidence": provider_evidence,
                                    "source_refs": [ref],
                                    "confidence": ref.get("confidence"),
                                }
                            )
                            continue
                        fragments = _field_fragments(field, raw)
                        if not fragments:
                            valid, normalized = _candidate_valid(field, raw)
                            fragments = [(raw.strip(), str(normalized))] if valid and normalized is not None else []
                        if not fragments:
                            candidates[field].append(
                                {
                                    "raw": raw.strip(),
                                    "normalized": None,
                                    "valid": False,
                                    "source_absent": is_explicit_source_absence(raw),
                                    "source_refs": [ref],
                                    "confidence": ref.get("confidence"),
                                }
                            )
                        for fragment, normalized in fragments:
                            candidates[field].append(
                                {
                                    "raw": fragment.strip(),
                                    "normalized": normalized,
                                    "valid": True,
                                    "source_absent": is_explicit_source_absence(fragment),
                                    "source_refs": [ref],
                                    "confidence": ref.get("confidence"),
                                }
                            )
    return candidates


def _dedupe_candidates(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in values:
        marker = str(candidate.get("normalized") or candidate.get("raw") or "").strip()
        if not marker:
            continue
        if marker not in merged:
            merged[marker] = dict(candidate)
            continue
        refs = list(merged[marker].get("source_refs") or ())
        refs.extend(candidate.get("source_refs") or ())
        merged[marker]["source_refs"] = refs
        confidences = [
            value
            for value in (merged[marker].get("confidence"), candidate.get("confidence"))
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if confidences:
            merged[marker]["confidence"] = max(float(value) for value in confidences)
    return list(merged.values())


def extract_candidate_b_profile(context: Any) -> dict[str, Any]:
    """Return one uncertainty-aware profile observation map."""
    result: dict[str, Any] = {}
    candidates_by_field = _table_candidates(context.pages)
    if not candidates_by_field:
        return result
    for field in _ALIASES:
        raw_candidates = candidates_by_field.get(field, [])
        if not raw_candidates:
            # PBOC detailed reports are canonical subsets: roles omitted by a
            # particular printed identity table are not extraction failures.
            # A visible label with a missing value is represented above by an
            # explicit label-only candidate and remains reviewable.
            continue
        for raw_candidate in raw_candidates:
            provider_evidence = raw_candidate.get("provider_evidence")
            if not isinstance(provider_evidence, Mapping):
                continue
            if provider_evidence.get("observation_status") not in {"unreadable", "ambiguous"}:
                continue
            record_issue(
                context,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_profile_provider_contract_unresolved",
                    message="An exact profile provider boundary was observed, but its provider value was not unambiguous.",
                    parser_stage="candidate_b_profile_extraction",
                    target_dataset="personal_profile",
                    target_record_id="personal_profile:1",
                    field_name=f"{field}.data_provider",
                    observed_value=provider_evidence.get("raw"),
                    source_refs=provider_evidence.get("source_refs") or (),
                    reason_codes=(
                        "exact_profile_provider_boundary",
                        "institution_contract_failed",
                        "provider_value_withheld",
                    ),
                ),
            )
        candidates = _dedupe_candidates(raw_candidates)
        valid = [
            candidate
            for candidate in candidates
            if candidate.get("valid") and not candidate.get("source_absent")
        ]
        normalized_values = {str(candidate.get("normalized") or "") for candidate in valid}
        if candidates and all(candidate.get("source_absent") for candidate in candidates):
            result[field] = {
                "value": None,
                "normalized_value": None,
                "raw": [
                    str(candidate.get("raw") or "")
                    for candidate in candidates
                    if candidate.get("raw")
                ],
                "source_refs": [
                    ref
                    for candidate in raw_candidates
                    for ref in candidate.get("source_refs") or ()
                ],
                "observation_status": "source_absent",
            }
            continue
        selected = valid[0] if len(normalized_values) == 1 and valid else None
        if selected is not None:
            entry: dict[str, Any] = {
                "value": selected["normalized"],
                "normalized_value": selected["normalized"],
                "raw": selected["raw"],
                "source_refs": list(selected.get("source_refs") or ()),
                "observation_status": (
                    "normalized" if str(selected.get("raw")) != str(selected.get("normalized")) else "observed"
                ),
            }
            confidence = selected.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                entry["confidence"] = max(0.0, min(1.0, float(confidence)))
                entry["confidence_basis"] = "canonical_table_cell"
            if isinstance(selected.get("provider_evidence"), Mapping):
                entry["provider_evidence"] = dict(selected["provider_evidence"])
            result[field] = entry
            continue

        observed = [str(candidate.get("raw") or "") for candidate in candidates if candidate.get("raw")]
        refs = [
            ref
            for candidate in raw_candidates
            for ref in candidate.get("source_refs") or ()
        ]
        status = "ambiguous" if len(normalized_values) > 1 else "unreadable"
        result[field] = {
            "value": None,
            "normalized_value": None,
            "raw": observed,
            "source_refs": refs,
            "observation_status": status,
            "reason": "candidate_b_profile_contract_unresolved",
        }
        provider_evidence = [
            dict(candidate["provider_evidence"])
            for candidate in candidates
            if isinstance(candidate.get("provider_evidence"), Mapping)
        ]
        if provider_evidence:
            result[field]["provider_evidence"] = provider_evidence
        record_issue(
            context,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_profile_contract_unresolved",
                message="Canonical profile observations were invalid or conflicting; the value was withheld.",
                parser_stage="candidate_b_profile_extraction",
                target_dataset="personal_profile",
                target_record_id="personal_profile:1",
                field_name=field,
                observed_value=observed,
                source_refs=refs,
                reason_codes=("canonical_template_cell", "schema_field_validation", "normalized_value_withheld"),
            ),
        )
    return result


__all__ = ["extract_candidate_b_profile"]
