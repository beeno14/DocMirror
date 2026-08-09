# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic quality contracts for scanned personal reports.

This module deliberately contains no OCR calls.  It is shared by extraction
and the final v2 projection so that both stages make the same decision about a
value without importing either orchestration layer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from docmirror.plugins.credit_report.currency_codes import (
    CURRENCY_CODE_BY_ALIAS,
    ISO_4217_CURRENT_CODES,
)

_CURRENCY_CODES = {
    **CURRENCY_CODE_BY_ALIAS,
    **{code: code for code in ISO_4217_CURRENT_CODES},
    "RMB": "CNY",
}

_HEADER_LABEL_NOISE = (
    "被查询者姓名",
    "被查询者证件类型",
    "被查询者证件号码",
    "查询机构",
    "查询原因",
    "报告时间",
)


def normalize_currency(value: Any) -> Any:
    """Return an ISO currency code when the PBOC label is unambiguous."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    return _CURRENCY_CODES.get(text, value)


def cn_identity_number_valid(value: Any) -> bool:
    """Validate an 18-character PRC resident identity number and checksum."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    if not re.fullmatch(r"\d{17}[0-9X]", text):
        return False
    try:
        date(int(text[6:10]), int(text[10:12]), int(text[12:14]))
    except ValueError:
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    check = "10X98765432"
    return check[sum(int(char) * weight for char, weight in zip(text[:17], weights, strict=True)) % 11] == text[-1]


def header_field_valid(field_name: str, value: Any, *, id_type: Any = None) -> bool:
    """Validate a page-one field conservatively; unknown is never guessed."""
    text = str(value or "").strip()
    if not text or any(label in text for label in _HEADER_LABEL_NOISE):
        return False
    compact = re.sub(r"\s+", "", text)
    if field_name == "subject_name":
        return bool(re.fullmatch(r"[\u3400-\u9fff·]{2,30}", compact))
    if field_name in {"primary_id_type", "document_type", "id_type"}:
        return compact in {
            "身份证",
            "居民身份证",
            "军官证",
            "士兵证",
            "护照",
            "港澳居民来往内地通行证",
            "台湾居民来往大陆通行证",
            "外国人永久居留证",
            "其他证件",
        }
    if field_name in {"primary_id_number", "document_number", "id_number"}:
        return cn_identity_number_valid(compact) if "身份" in str(id_type or "") else bool(
            re.fullmatch(r"[0-9A-Za-z()（）-]{5,40}", compact)
        )
    if field_name == "query_institution":
        return compact == "本人" or bool(
            re.search(r"[\u3400-\u9fff]", compact)
            and len(compact) >= 4
            and re.search(r"(?:银行|中心|公司|联社|机构|本人|分行|支行|营业部|信用社)$", compact)
        )
    if field_name == "report_number":
        return bool(re.fullmatch(r"\d{18,30}", compact))
    if field_name == "report_time":
        try:
            datetime.fromisoformat(text)
            return True
        except ValueError:
            return False
    return True


def valid_iso_date(value: Any, *, month_precision: bool = False) -> bool:
    text = str(value or "").strip()
    if month_precision:
        match = re.fullmatch(r"(\d{4})-(\d{2})", text)
        return bool(match and 1 <= int(match.group(2)) <= 12)
    try:
        date.fromisoformat(text)
        return True
    except (TypeError, ValueError):
        return False


def decode_mapping(value: Any) -> dict[str, Any] | None:
    """Decode a mapping or a JSON-serialized mapping without accepting lists."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip().startswith("{"):
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return dict(decoded) if isinstance(decoded, Mapping) else None


__all__ = [
    "cn_identity_number_valid",
    "decode_mapping",
    "header_field_valid",
    "normalize_currency",
    "valid_iso_date",
]
