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
from dataclasses import dataclass
from datetime import date
from typing import Any

from docmirror.plugins.credit_report.pboc_vocabularies import is_pboc_institution_name
from docmirror.plugins.credit_report.personal_detail_scanned.exact_evidence import (
    exact_cell_visually_contains_dash_pair,
    resolve_exact_page_token_atoms,
)
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
    re.sub(r"[\s:：，,；;()（）\[\]【】]", "", alias): field for field, aliases in _ALIASES.items() for alias in aliases
}
_VOCAB_ROLES = {
    "gender": "gender",
    "marital_status": "marital_status",
    "employment_status": "employment_status",
    "education_level": "education_level",
    "degree": "degree",
}
_STRICT_PROFILE_SCALAR_FIELDS = frozenset({*_VOCAB_ROLES, "birth_date", "nationality"})
_PHONE_FIELDS = frozenset({"mobile_phone", "work_phone", "residence_phone"})
_EXACT_PROFILE_TOKEN_FIELDS = frozenset({*_STRICT_PROFILE_SCALAR_FIELDS, *_PHONE_FIELDS, "email"})
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
_EMAIL_RE = re.compile(
    r"(?=.{3,254}\Z)"
    r"(?=[^@]{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}"
)
_NONMOBILE_PHONE_RE = re.compile(
    r"(?:"
    r"\d{7,8}"
    r"|0\d{2,3}\d{7,8}"
    r"|0\d{2,3}-\d{7,8}"
    r"|0\d{2,3} \d{7,8}"
    r"|\(0\d{2,3}\)[ -]?\d{7,8}"
    r"|(?:\+86|0086)[ -](?:"
    r"0\d{2,3}-\d{7,8}"
    r"|0\d{2,3} \d{7,8}"
    r"|\(0\d{2,3}\)[ -]?\d{7,8}"
    r")"
    r")"
)
_ADDRESS_MARKER_RE = re.compile(r"[省市县区镇乡村路街道巷号栋室楼]")
_SYMBOL_NOISE_RE = re.compile(r"[#=*<>]{2,}|[\"“”]{2,}")
_PROVINCE_LEVEL_RE = re.compile(
    r"北京市|天津市|上海市|重庆市|"
    r"(?:河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|"
    r"湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾)省|"
    r"内蒙古自治区|广西壮族自治区|西藏自治区|宁夏回族自治区|新疆维吾尔自治区|"
    r"香港特别行政区|澳门特别行政区"
)
_SUBORDINATE_REGION_PREFIX_RE = re.compile(r"^[\u3400-\u9fff]{1,10}(?:市|自治州|地区|盟|县|自治县|区|旗|镇|乡|街道|村)")
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
_HOUSEHOLD_ADDRESS_LABEL = "\u6237\u7c4d\u5730\u5740"


@dataclass(frozen=True, slots=True)
class _MergedHouseholdHeaderTrait:
    """Source-owned proof that one merged header contains ``鎴风睄鍦板潃``."""

    row: int
    column: int
    evidence_ids: tuple[str, ...]
    bbox: tuple[float, float, float, float]


def _finite_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return None
    if not all(
        coordinate == coordinate and abs(coordinate) != float("inf") for coordinate in (left, top, right, bottom)
    ):
        return None
    return (left, top, right, bottom) if right > left and bottom > top else None


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
        and _compact(normalized) == _compact(raw)
        and _PROFILE_PROVIDER_SUFFIX_RE.fullmatch(_compact(normalized))
    )
    return {
        "raw": raw,
        "normalized_value": normalized if valid else None,
        "observation_status": ("normalized" if valid and raw != normalized else "observed" if valid else "unreadable"),
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
    metadata = getattr(table, "metadata", None) or {}
    source_logical_page = int(metadata.get("source_logical_page") or 0) if isinstance(metadata, Mapping) else 0
    source_page = int(metadata.get("source_page") or 0) if isinstance(metadata, Mapping) else 0
    ref: dict[str, Any] = {
        "source": "candidate_b_canonical_table",
        "logical_page": source_logical_page or int(getattr(page, "page_number", 0) or 0),
        "source_page": int(
            source_page or getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0
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
    bbox = None
    geometry_status = None
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
        geometry_rows = metadata.get("cell_geometry_status")
        if (
            isinstance(geometry_rows, list)
            and 0 <= row < len(geometry_rows)
            and isinstance(geometry_rows[row], list)
            and 0 <= col < len(geometry_rows[row])
        ):
            geometry_status = str(geometry_rows[row][col] or "")
        evidence_ids = metadata.get("cell_evidence_ids")
        if (
            isinstance(evidence_ids, list)
            and 0 <= row < len(evidence_ids)
            and isinstance(evidence_ids[row], list)
            and 0 <= col < len(evidence_ids[row])
            and isinstance(evidence_ids[row][col], list)
        ):
            ref["evidence_ids"] = [str(value) for value in evidence_ids[row][col] if value]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and geometry_status == "exact":
        ref["bbox"] = list(bbox)
        ref["geometry_scope"] = "cell"
        ref["geometry_status"] = geometry_status
        ref["coordinate_system"] = "pdf_points_top_left"
    confidence = None
    if isinstance(metadata, Mapping):
        confidence_rows = metadata.get("cell_geometry_confidences") or metadata.get("cell_confidences")
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


def _exact_profile_cell(table: Any, row: int, col: int) -> Any | None:
    """Return one evidence-sealed exact native cell behind a canonical slot."""

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    native_cells = getattr(table, "source_cell_objects", None)
    if (
        not isinstance(native_cells, list)
        or not (0 <= row < len(native_cells))
        or not isinstance(native_cells[row], list)
        or not (0 <= col < len(native_cells[row]))
    ):
        return None
    cell = native_cells[row][col]
    if (
        cell is None
        or str(getattr(cell, "geometry_status", "") or "") != "exact"
        or not getattr(cell, "evidence_ids", None)
    ):
        return None
    bbox = getattr(cell, "bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(value == value and abs(value) != float("inf") for value in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    return cell


def _exact_profile_cell_tokens(
    context: Any,
    table: Any,
    row: int,
    col: int,
    *,
    logical_page: int | None,
) -> tuple[tuple[str, tuple[float, float, float, float], str], ...] | None:
    """Return the independently boxed tokens owned by one exact profile cell."""

    cell = _exact_profile_cell(table, row, col)
    if cell is None:
        return None
    evidence_ids = tuple(str(value) for value in getattr(cell, "evidence_ids", None) or () if str(value or ""))
    token_ids = tuple(str(value) for value in getattr(cell, "token_ids", None) or () if str(value or ""))
    if (
        not token_ids
        or len(token_ids) != len(set(token_ids))
        or len(evidence_ids) != len(set(evidence_ids))
        or set(token_ids) != set(evidence_ids)
    ):
        return None

    resolved_tokens = resolve_exact_page_token_atoms(
        context,
        token_ids,
        logical_page=logical_page,
    )
    if resolved_tokens is None:
        return None

    cell_bbox = tuple(float(value) for value in getattr(cell, "bbox"))
    output: list[tuple[str, tuple[float, float, float, float], str]] = []
    for text, bbox, token_id in resolved_tokens:
        if (
            bbox[0] < cell_bbox[0] - 1.0
            or bbox[1] < cell_bbox[1] - 1.0
            or bbox[2] > cell_bbox[2] + 1.0
            or bbox[3] > cell_bbox[3] + 1.0
        ):
            return None
        output.append((text, bbox, token_id))
    return tuple(sorted(output, key=lambda item: (item[1][0], item[1][1], item[2])))


def _profile_token_boxes_are_disjoint(
    tokens: Iterable[tuple[str, tuple[float, float, float, float], str]],
) -> bool:
    boxes = [token[1] for token in tokens]
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if left[0] < right[2] and right[0] < left[2] and left[1] < right[3] and right[1] < left[3]:
                return False
    return True


def _profile_token_ids_have_unique_cell_owners(
    context: Any,
    token_ids: Iterable[str],
) -> bool:
    """Reject evidence IDs replayed by another physical source cell."""

    required = frozenset(str(value) for value in token_ids if str(value or ""))
    if not required:
        return False
    owner_counts = {token_id: 0 for token_id in required}
    for page in getattr(context, "pages", None) or ():
        for table in getattr(page, "tables", None) or ():
            source_cells = getattr(table, "source_cell_objects", None)
            if not isinstance(source_cells, list):
                continue
            for row in source_cells:
                if not isinstance(row, list):
                    continue
                for cell in row:
                    if cell is None:
                        continue
                    owned = {
                        str(value)
                        for attribute in ("evidence_ids", "token_ids")
                        for value in getattr(cell, attribute, None) or ()
                        if str(value or "")
                    }
                    for token_id in required & owned:
                        owner_counts[token_id] += 1
    return all(count == 1 for count in owner_counts.values())


def _bounded_profile_mailing_header_residue(
    context: Any,
    table: Any,
    row_index: int,
    column: int,
) -> bool:
    """Prove ``通讯地址`` beside one stray Han atom in an exact header cell.

    Prefer the independently resolved two-token partition.  Some native table
    producers retain the exact cell and its two immutable owners while their
    token boxes overlap at the OCR seam.  In that bounded case the complete
    raw-cell signature plus unique ownership is still sufficient to recover
    the *header role*; no value cell is repaired by this fallback.
    """

    cell = _exact_profile_cell(table, row_index, column)
    rows = _rows(table)
    if not (
        cell is not None
        and getattr(cell, "row_span", 1) == 1
        and getattr(cell, "col_span", 1) == 1
        and 0 <= row_index < len(rows)
        and 0 <= column < len(rows[row_index])
    ):
        return False
    raw_header = str(rows[row_index][column] or "").strip()
    bounded_raw_residue = bool(
        re.fullmatch(
            r"[\u3400-\u9fff]\s+(?:通讯地址|通信地址)",
            raw_header,
        )
    )
    metadata = getattr(table, "metadata", None) or {}
    logical_page = int(metadata.get("source_logical_page") or 0) if isinstance(metadata, Mapping) else 0
    tokens = _exact_profile_cell_tokens(
        context,
        table,
        row_index,
        column,
        logical_page=logical_page,
    )
    if tokens is not None and len(tokens) == 2:
        strict_partition = bool(
            _profile_token_boxes_are_disjoint(tokens)
            and _compact("".join(text for text, _bbox, _token_id in tokens)) == _compact(raw_header)
        )
        mailing_tokens = [token for token in tokens if _label_fields(token[0]) == ["mailing_address"]]
        residue_tokens = [token for token in tokens if token not in mailing_tokens]
        if strict_partition and len(mailing_tokens) == len(residue_tokens) == 1:
            mailing = mailing_tokens[0]
            residue = residue_tokens[0]
            cell_bbox = _finite_bbox(getattr(cell, "bbox", None))
            vertical_delta = (
                abs((residue[1][1] + residue[1][3]) / 2.0 - (mailing[1][1] + mailing[1][3]) / 2.0)
                if cell_bbox is not None
                else float("inf")
            )
            token_ids = tuple(token_id for _text, _bbox, token_id in tokens)
            if (
                re.fullmatch(r"[\u3400-\u9fff]", _compact(residue[0])) is not None
                and not _label_fields(residue[0])
                and residue[1][2] <= mailing[1][0]
                and cell_bbox is not None
                and vertical_delta <= max(2.0, (cell_bbox[3] - cell_bbox[1]) * 0.35)
                and len(token_ids) == len(set(token_ids))
                and _profile_token_ids_have_unique_cell_owners(
                    context,
                    token_ids,
                )
            ):
                return True

    evidence_ids = tuple(str(value) for value in getattr(cell, "evidence_ids", None) or () if str(value or ""))
    token_ids = tuple(str(value) for value in getattr(cell, "token_ids", None) or () if str(value or ""))
    return bool(
        bounded_raw_residue
        and len(evidence_ids) == len(token_ids) == 2
        and len(set(evidence_ids)) == 2
        and len(set(token_ids)) == 2
        and set(evidence_ids) == set(token_ids)
        and _profile_token_ids_have_unique_cell_owners(context, token_ids)
    )


def _exact_profile_scalar_token_candidate(
    context: Any,
    page: Any,
    table: Any,
    *,
    header_row: int,
    header_column: int,
    header_fields: Iterable[str],
    value_row: int,
    value_column: int,
    raw: str,
    field: str,
) -> tuple[str, str, dict[str, Any]] | None:
    """Bind one strict scalar to a closed, residue-free token partition.

    This capability applies only when multiple registered scalar labels share
    one exact header cell.  Every header and value token must have a distinct
    immutable owner, every value token must type as exactly one registered
    role, and the complete physical-cell text must be consumed.
    """

    observed_roles = tuple(str(role) for role in header_fields)
    roles = tuple(dict.fromkeys(observed_roles))
    if (
        field not in roles
        or len(roles) < 2
        or len(roles) != len(observed_roles)
        or any(role not in _EXACT_PROFILE_TOKEN_FIELDS for role in roles)
    ):
        return None
    logical_page = int(getattr(page, "page_number", 0) or 0)
    header_tokens = _exact_profile_cell_tokens(
        context,
        table,
        header_row,
        header_column,
        logical_page=logical_page,
    )
    value_tokens = _exact_profile_cell_tokens(
        context,
        table,
        value_row,
        value_column,
        logical_page=logical_page,
    )
    if (
        header_tokens is None
        or value_tokens is None
        or len(header_tokens) != len(roles)
        or len(value_tokens) != len(roles)
        or not _profile_token_boxes_are_disjoint(header_tokens)
        or not _profile_token_boxes_are_disjoint(value_tokens)
        or not _profile_token_boxes_are_disjoint((*header_tokens, *value_tokens))
    ):
        return None

    rows = _rows(table)
    if not (0 <= header_row < len(rows) and 0 <= header_column < len(rows[header_row])):
        return None
    header_raw = str(rows[header_row][header_column] or "")
    if re.sub(r"\s+", "", "".join(token[0] for token in header_tokens)) != re.sub(r"\s+", "", header_raw) or re.sub(
        r"\s+", "", "".join(token[0] for token in value_tokens)
    ) != re.sub(r"\s+", "", str(raw or "")):
        return None

    header_token_roles = tuple(_LABEL_TO_FIELD.get(_compact(text)) for text, _bbox, _token_id in header_tokens)
    if (
        any(role is None for role in header_token_roles)
        or len(set(header_token_roles)) != len(header_token_roles)
        or set(header_token_roles) != set(roles)
    ):
        return None

    assignments: dict[
        str,
        tuple[str, str, tuple[float, float, float, float], str],
    ] = {}
    for header_role, (token_text, token_bbox, token_id) in zip(
        header_token_roles,
        value_tokens,
        strict=True,
    ):
        matches: list[tuple[str, str]] = []
        for role in roles:
            valid, normalized = _candidate_valid(role, token_text)
            if valid and normalized is not None:
                matches.append((role, str(normalized)))
        if len(matches) != 1 or matches[0][0] != header_role or header_role in assignments:
            return None
        assignments[header_role] = (
            token_text,
            matches[0][1],
            token_bbox,
            token_id,
        )
    if set(assignments) != set(roles):
        return None

    all_token_ids = tuple(token_id for _text, _bbox, token_id in (*header_tokens, *value_tokens))
    if len(all_token_ids) != len(set(all_token_ids)) or not _profile_token_ids_have_unique_cell_owners(
        context, all_token_ids
    ):
        return None
    token_text, normalized, token_bbox, token_id = assignments[field]
    ref = _source_ref(page, table, value_row, value_column)
    ref.update(
        {
            "field_name": field,
            "bbox": list(token_bbox),
            "evidence_ids": [token_id],
            "geometry_scope": "token",
            "geometry_status": "exact",
            "coordinate_system": "pdf_points_top_left",
            "binding": "exact_profile_scalar_token_owner",
            "binding_quality": "exact_profile_scalar_token_owner",
        }
    )
    return token_text, normalized, ref


def _bounded_profile_address_sequence_residue(
    context: Any,
    table: Any,
    row: int,
    col: int,
    raw: str,
    *,
    logical_page: int | None,
) -> str | None:
    """Remove one independently owned row-number residue from an address slot.

    A profile address has no sequence field.  A nearby provider-row ordinal may
    nevertheless be pulled into its merged cell when a horizontal rule is faint.
    Recovery is allowed only when the cell owns exactly two distinct OCR atoms:
    one tiny left-edge integer and one complete region-led address.  The atoms'
    boxes, IDs, order, and full raw concatenation must all agree.  This is not a
    general leading-number heuristic.
    """

    cell = _exact_profile_cell(table, row, col)
    tokens = _exact_profile_cell_tokens(
        context,
        table,
        row,
        col,
        logical_page=logical_page,
    )
    if cell is None or tokens is None or len(tokens) != 2:
        return None
    row_span = getattr(cell, "row_span", 1)
    col_span = getattr(cell, "col_span", 1)
    if row_span != 1 or col_span != 2:
        return None
    ordinal_candidates = [token for token in tokens if re.fullmatch(r"[1-9]\d?", token[0])]
    address_candidates = [token for token in tokens if token not in ordinal_candidates]
    if len(ordinal_candidates) != 1 or len(address_candidates) != 1:
        return None
    ordinal = ordinal_candidates[0]
    address = address_candidates[0]
    address_text = address[0].strip()
    if (
        not re.match(r"^[\u3400-\u9fff]{1,12}(?:省|市|自治区|特别行政区)", address_text)
        or not _address_candidate_valid(address_text)
        or _compact(raw) != _compact(ordinal[0] + address_text)
    ):
        return None
    cell_bbox = tuple(float(value) for value in getattr(cell, "bbox"))
    cell_width = cell_bbox[2] - cell_bbox[0]
    ordinal_width = ordinal[1][2] - ordinal[1][0]
    address_width = address[1][2] - address[1][0]
    if (
        ordinal_width > cell_width * 0.12
        or address_width < cell_width * 0.45
        or abs(((ordinal[1][1] + ordinal[1][3]) / 2.0) - ((address[1][1] + address[1][3]) / 2.0))
        > max(3.0, (cell_bbox[3] - cell_bbox[1]) * 0.35)
    ):
        return None
    edge_tolerance = max(1.0, cell_width * 0.02)
    separated_at_left = ordinal[1][2] + edge_tolerance <= address[1][0]
    separated_at_right = address[1][2] + edge_tolerance <= ordinal[1][0]
    if not (separated_at_left or separated_at_right):
        return None
    return address_text


def _bounded_visual_dash_only_cell(context: Any, table: Any, row: int, col: int) -> bool:
    """Verify a printed ``--`` in one exact canonical profile cell."""
    cell = _exact_profile_cell(table, row, col)
    if cell is None:
        return False
    metadata = getattr(table, "metadata", None) or {}
    logical_page = int(metadata.get("source_logical_page") or 0) if isinstance(metadata, Mapping) else 0
    return exact_cell_visually_contains_dash_pair(context, cell, logical_page=logical_page)


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
        match = _DATE_RE.fullmatch(text)
        if not match:
            return False, None
        parts = [int(value) for value in re.findall(r"\d+", match.group(0))[:3]]
        if len(parts) != 3:
            return False, None
        try:
            parsed = date(parts[0], parts[1], parts[2])
        except ValueError:
            return False, None
        return True, parsed.isoformat()
    if field == "mobile_phone":
        match = re.fullmatch(
            r"(?:(?:\+?86|0086)[ -]?)?(1[3-9]\d[ -]?\d{4}[ -]?\d{4})",
            text,
        )
        if match is None:
            return False, None
        return True, re.sub(r"[ -]", "", match.group(1))
    if field in {"work_phone", "residence_phone"}:
        valid = _NONMOBILE_PHONE_RE.fullmatch(text) is not None
        return valid, text if valid else None
    if field == "email":
        return bool(_EMAIL_RE.fullmatch(text)), text if _EMAIL_RE.fullmatch(text) else None
    if field in _ADDRESS_FIELDS:
        valid = _address_candidate_valid(text)
        return valid, text if valid else None
    if field == "nationality":
        contract = validate_pboc_field(text, "country_or_region_code")
        valid = contract.valid and not is_pboc_institution_name(compact)
        return valid, normalize_pboc_field(text, "country_or_region_code") if valid else None
    return True, text


def canonical_profile_phone(field: str, value: Any) -> str | None:
    """Return one canonical profile phone under the extraction grammar."""

    if field not in _PHONE_FIELDS or isinstance(value, (Mapping, list, tuple, set)):
        return None
    valid, normalized = _candidate_valid(field, str(value or ""))
    return str(normalized) if valid and normalized is not None else None


def _label_fields(value: Any) -> list[str]:
    """Return roles only from a residue-free whole-cell alias grammar."""

    marker = _compact(value)
    exact = _LABEL_TO_FIELD.get(marker)
    if exact:
        return [exact]
    if not marker:
        return []

    alias_roles = tuple(
        sorted(
            {(_compact(alias), field) for field, aliases in _ALIASES.items() for alias in aliases if _compact(alias)},
            key=lambda item: (-len(item[0]), item[0], item[1]),
        )
    )
    memo: dict[int, set[tuple[str, ...]]] = {}

    def complete_covers(offset: int) -> set[tuple[str, ...]]:
        if offset == len(marker):
            return {()}
        if offset in memo:
            return memo[offset]
        covers: set[tuple[str, ...]] = set()
        for alias, field in alias_roles:
            if not marker.startswith(alias, offset):
                continue
            for tail in complete_covers(offset + len(alias)):
                candidate = (field, *tail)
                if len(candidate) == len(set(candidate)):
                    covers.add(candidate)
        memo[offset] = covers
        return covers

    covers = complete_covers(0)
    if len(covers) != 1:
        return []
    fields = next(iter(covers))
    # Household-address repair still requires independently owned exact token
    # geometry; a concatenated raw header is not sufficient authority.
    if "household_address" in fields:
        return []
    return list(fields)


def _merged_household_header_trait(
    context: Any,
    table: Any,
    row_index: int,
    column: int,
) -> _MergedHouseholdHeaderTrait | None:
    """Prove one merged household-address label from immutable token atoms.

    The canonical raw string is not authority for a role repair.  The source
    cell must be exact, must own a unique closed set of token IDs, and exactly
    one of those independently boxed tokens must itself print ``鎴风睄鍦板潃``.
    """

    cell = _exact_profile_cell(table, row_index, column)
    metadata = getattr(table, "metadata", None) or {}
    logical_page = int(metadata.get("source_logical_page") or 0) if isinstance(metadata, Mapping) else 0
    tokens = _exact_profile_cell_tokens(
        context,
        table,
        row_index,
        column,
        logical_page=logical_page,
    )
    if cell is None or tokens is None or len(tokens) < 2:
        return None
    rows = _rows(table)
    if not (
        0 <= row_index < len(rows)
        and 0 <= column < len(rows[row_index])
        and _compact("".join(text for text, _bbox, _token_id in tokens)) == _compact(rows[row_index][column])
        and _profile_token_boxes_are_disjoint(tokens)
    ):
        return None
    token_roles = [_label_fields(text) for text, _bbox, _token_id in tokens]
    if any(len(roles) != 1 for roles in token_roles):
        return None
    recognized_fields = [roles[0] for roles in token_roles]
    if (
        len(set(recognized_fields)) != len(recognized_fields)
        or len(recognized_fields) < 2
        or recognized_fields.count("household_address") != 1
    ):
        return None
    evidence_ids = tuple(token_id for _text, _bbox, token_id in tokens)
    if len(evidence_ids) != len(set(evidence_ids)) or not _profile_token_ids_have_unique_cell_owners(
        context, evidence_ids
    ):
        return None
    bbox = _finite_bbox(getattr(cell, "bbox", None))
    if bbox is None:
        return None
    return _MergedHouseholdHeaderTrait(
        row=row_index,
        column=column,
        evidence_ids=evidence_ids,
        bbox=bbox,
    )


def _exact_clipped_nationality_column(
    context: Any,
    table: Any,
    *,
    header_row: int,
    column: int,
) -> bool:
    """Prove the clipped ``国`` header and its adjacent nationality value."""

    rows = _rows(table)
    value_row = header_row + 1
    if not (
        0 <= header_row < len(rows)
        and 0 <= value_row < len(rows)
        and 0 <= column < len(rows[header_row])
        and 0 <= column < len(rows[value_row])
        and _compact(rows[header_row][column]) == "国"
    ):
        return False
    valid, _normalized = _candidate_valid("nationality", rows[value_row][column])
    if not valid:
        return False

    metadata = getattr(table, "metadata", None) or {}
    logical_page = int(metadata.get("source_logical_page") or 0) if isinstance(metadata, Mapping) else 0
    header_cell = _exact_profile_cell(table, header_row, column)
    value_cell = _exact_profile_cell(table, value_row, column)
    header_tokens = _exact_profile_cell_tokens(
        context,
        table,
        header_row,
        column,
        logical_page=logical_page,
    )
    value_tokens = _exact_profile_cell_tokens(
        context,
        table,
        value_row,
        column,
        logical_page=logical_page,
    )
    if (
        header_cell is None
        or value_cell is None
        or header_tokens is None
        or value_tokens is None
        or len(header_tokens) != 1
        or not value_tokens
        or not _profile_token_boxes_are_disjoint(value_tokens)
        or _compact(header_tokens[0][0]) != "国"
        or _compact("".join(text for text, _bbox, _token_id in value_tokens)) != _compact(rows[value_row][column])
    ):
        return False

    header_bbox = _finite_bbox(getattr(header_cell, "bbox", None))
    value_bbox = _finite_bbox(getattr(value_cell, "bbox", None))
    if header_bbox is None or value_bbox is None:
        return False
    horizontal_overlap = min(header_bbox[2], value_bbox[2]) - max(header_bbox[0], value_bbox[0])
    if horizontal_overlap <= 0:
        return False

    token_ids = tuple(token_id for _text, _bbox, token_id in (*header_tokens, *value_tokens))
    return bool(
        len(token_ids) == len(set(token_ids)) and _profile_token_ids_have_unique_cell_owners(context, token_ids)
    )


def _row_label_fields(
    row: list[str],
    *,
    context: Any | None = None,
    table: Any | None = None,
    row_index: int | None = None,
) -> list[tuple[int, str]]:
    """Apply the closed canonical identity-row signature to damaged headers."""

    by_column = {column: _label_fields(value) for column, value in enumerate(row)}
    observed = {field for fields in by_column.values() for field in fields}
    if "nationality" not in observed and len(row) == 4:
        education_columns = [column for column, fields in by_column.items() if fields == ["education_level"]]
        degree_columns = [column for column, fields in by_column.items() if fields == ["degree"]]
        email_columns = [column for column, fields in by_column.items() if fields == ["email"]]
        if (
            education_columns == [0]
            and degree_columns == [1]
            and email_columns == [3]
            and not by_column[2]
            and _compact(row[2]) == "国"
        ):
            # Position is only a locator.  Material role authority comes from
            # exact, uniquely owned header and value atoms in the same column.
            if (
                context is not None
                and table is not None
                and isinstance(row_index, int)
                and _exact_clipped_nationality_column(
                    context,
                    table,
                    header_row=row_index,
                    column=2,
                )
            ):
                by_column[2].append("nationality")
    household_columns = [column for column, fields in by_column.items() if "household_address" in fields]
    if len(household_columns) > 1:
        for column in household_columns:
            by_column[column] = [field for field in by_column[column] if field != "household_address"]
    elif not household_columns and context is not None and table is not None and isinstance(row_index, int):
        traits = [
            trait
            for column in range(len(row))
            if (
                trait := _merged_household_header_trait(
                    context,
                    table,
                    row_index,
                    column,
                )
            )
            is not None
        ]
        # Duplicate physical owners make the semantic column ambiguous.
        if len(traits) == 1:
            by_column[traits[0].column].append("household_address")
    mailing_columns = [column for column, fields in by_column.items() if "mailing_address" in fields]
    household_columns = [column for column, fields in by_column.items() if "household_address" in fields]
    if (
        not mailing_columns
        and len(household_columns) == 1
        and len(row) == 4
        and context is not None
        and table is not None
        and isinstance(row_index, int)
    ):
        residue_columns = [
            column
            for column in range(len(row))
            if _bounded_profile_mailing_header_residue(
                context,
                table,
                row_index,
                column,
            )
        ]
        if len(residue_columns) == 1:
            by_column[residue_columns[0]].append("mailing_address")
    return [(column, field) for column, fields in by_column.items() for field in fields]


def _field_fragments(field: str, value: str) -> list[tuple[str, str]]:
    """Decode field-specific values from one OCR-collapsed canonical cell."""

    text = str(value or "").strip()
    if not text:
        return []
    if field in _EXACT_PROFILE_TOKEN_FIELDS:
        valid, normalized = _candidate_valid(field, text)
        return [(text, str(normalized))] if valid and normalized is not None else []
    business_text = (
        text if field in _ADDRESS_FIELDS else re.split(r"数据发生机构(?:名称)?", text, maxsplit=1)[0].strip()
    )
    if not business_text:
        return []
    text = business_text
    valid, normalized = _candidate_valid(field, text)
    return [(text, str(normalized))] if valid and normalized is not None else []


def _table_candidates(context: Any, pages: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        for table in getattr(page, "tables", None) or ():
            rows = _rows(table)
            if not _is_profile_table(page, table, rows):
                continue
            for row_index, row in enumerate(rows):
                labels = _row_label_fields(
                    row,
                    context=context,
                    table=table,
                    row_index=row_index,
                )
                fields_by_column: dict[int, set[str]] = defaultdict(set)
                for label_column, label_field in labels:
                    fields_by_column[label_column].add(label_field)
                for column, field in labels:
                    header_fields = tuple(label_field for label_column, label_field in labels if label_column == column)
                    choices: list[tuple[str, int, int]] = []
                    next_label_column = next(
                        (
                            candidate_column
                            for candidate_column, _candidate_field in labels
                            if candidate_column > column
                        ),
                        len(row),
                    )
                    inline_values = (
                        [
                            (str(row[candidate_column]), row_index, candidate_column)
                            for candidate_column in range(column + 1, next_label_column)
                            if str(row[candidate_column] or "").strip() and not _label_fields(row[candidate_column])
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
                        address_without_residue = (
                            _bounded_profile_address_sequence_residue(
                                context,
                                table,
                                value_row,
                                value_col,
                                raw,
                                logical_page=int(getattr(page, "page_number", 0) or 0),
                            )
                            if field == "mailing_address"
                            else None
                        )
                        if address_without_residue is not None:
                            candidates[field].append(
                                {
                                    "raw": raw.strip(),
                                    "normalized": address_without_residue,
                                    "valid": True,
                                    "source_absent": False,
                                    "source_refs": [ref],
                                    "confidence": ref.get("confidence"),
                                    "recognition_source": "exact_token_owned_sequence_residue",
                                }
                            )
                            continue
                        if (
                            field in {"degree", "household_address"}
                            and not is_explicit_source_absence(raw)
                            and _bounded_visual_dash_only_cell(context, table, value_row, value_col)
                        ):
                            candidates[field].append(
                                {
                                    "raw": "--",
                                    "normalized": None,
                                    "valid": False,
                                    "source_absent": True,
                                    "source_refs": [ref],
                                    "confidence": ref.get("confidence"),
                                    "recognition_source": "static_visual_dash_shape",
                                }
                            )
                            continue
                        address_split = _split_profile_address_provider(raw) if field in _ADDRESS_FIELDS else None
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
                        strict_multi_role = field in _EXACT_PROFILE_TOKEN_FIELDS and len(header_fields) > 1
                        strict_token_candidate = (
                            _exact_profile_scalar_token_candidate(
                                context,
                                page,
                                table,
                                header_row=row_index,
                                header_column=column,
                                header_fields=header_fields,
                                value_row=value_row,
                                value_column=value_col,
                                raw=raw,
                                field=field,
                            )
                            if strict_multi_role
                            else None
                        )
                        if strict_token_candidate is not None:
                            token_raw, token_normalized, token_ref = strict_token_candidate
                            candidates[field].append(
                                {
                                    "raw": token_raw.strip(),
                                    "normalized": token_normalized,
                                    "valid": True,
                                    "source_absent": False,
                                    "source_refs": [token_ref],
                                    "confidence": token_ref.get("confidence"),
                                    "recognition_source": "exact_profile_scalar_token_owner",
                                }
                            )
                            continue
                        fragments = [] if strict_multi_role else _field_fragments(field, raw)
                        if not fragments and not strict_multi_role:
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
    candidates_by_field = _table_candidates(context, context.pages)
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
        valid = [candidate for candidate in candidates if candidate.get("valid") and not candidate.get("source_absent")]
        normalized_values = {str(candidate.get("normalized") or "") for candidate in valid}
        if candidates and all(candidate.get("source_absent") for candidate in candidates):
            result[field] = {
                "value": None,
                "normalized_value": None,
                "raw": [str(candidate.get("raw") or "") for candidate in candidates if candidate.get("raw")],
                "source_refs": [ref for candidate in raw_candidates for ref in candidate.get("source_refs") or ()],
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
        refs = [ref for candidate in raw_candidates for ref in candidate.get("source_refs") or ()]
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


__all__ = ["canonical_profile_phone", "extract_candidate_b_profile"]
