# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-seal OCR correction overlay for personal detailed credit reports.

The overlay never mutates ``ParseResult``.  It produces corrected copies of
OCR evidence and extraction candidates, keyed by stable semantic roles rather
than Community JSON paths.  Domain extractors can therefore consume corrected
evidence while the sealed source remains auditable and schema redesigns remain
independent from OCR policy.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable, Mapping

import yaml

_DATE_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)")
_DATE_LOOSE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[.,/-]?(\d{2})[.,/-](\d{2})(?!\d)")
_MONTH_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[./-](\d{1,2})(?![\d./-])")
_DATETIME_DIGITS_RE = re.compile(r"(?<!\d)(20\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})(?!\d)")
_CN_ID_RE = re.compile(r"^\d{17}[0-9X]$")
_MOBILE_RE = re.compile(r"^1[3-9]\d{9}$")
_ACCOUNT_IDENTIFIER_RE = re.compile(r"^[A-Z0-9-]{8,80}$")
_AMOUNT_RE = re.compile(r"^-?(?:0|[1-9]\d{0,14})(?:\.\d{1,2})?$")
_INQUIRY_DATE_LIKE_RE = re.compile(r"20\d{2}(?:[.,/:;-]?\d{1,2}[.,/:;-]\d{1,2}|[.]\d{4})")
_VALID_INQUIRY_REASONS = (
    "本人查询",
    "贷后管理",
    "贷款审批",
    "信用卡审批",
    "担保资格审查",
    "融资审批",
    "保前审查",
    "保后管理",
    "客户准入资格审查",
    "资信审查",
    "法人代表、负责人、高管",
)
_INQUIRY_REASON_MARKERS = _VALID_INQUIRY_REASONS + (
    "费后管理",
    "货后管理",
    "货款审批",
    "资款审批",
    "信用卡审抵",
    "担保资格申查",
    "贷后智理",
    "磁资审批",
)
_INSTITUTION_SUFFIX_RE = re.compile(
    r"[A-Za-z0-9\u3400-\u9fff（）()·]{2,100}?(?:"
    r"银行卡业务部[（(]牡丹卡中心[）)]|信用卡中心|个人信贷部|"
    r"支行|分行|"
    r"农村信用合作联社|农村信用社联合社|股份有限公司|股份公司|有限责任公司|有限公司|管理中心"
    r")"
)
_LEADING_ROW_NOISE_RE = re.compile(r"^[\s\W_]*(?:[A-Za-z\u3400-\u9fff]{1,2}\s+)?(?=\d{0,3}\s*20\d{2})")
_TRAILING_INSTITUTION_NOISE_RE = re.compile(r"(?:\s+[A-Za-z0-9￥¥?$]{1,3})+$")
_REPAYMENT_STATUSES = frozenset({"*", "/", "N", "A", "C", "M", "B", "D", "Z", "G", "#", *"1234567"})
_PLACEHOLDERS = frozenset({"-", "--"})
_ACCOUNT_TYPE_LABELS = (
    "非循环贷账户",
    "循环贷账户一",
    "循环贷账户二",
    "贷记卡账户",
    "准贷记卡账户",
)
_ACCOUNT_STATES = frozenset(
    {
        "正常",
        "逾期",
        "结清",
        "呆账",
        "转出",
        "结束",
        "冻结",
        "止付",
        "银行止付",
        "销户",
        "未激活",
        "司法追偿",
        "担保物不足",
        "强制平仓",
        "催收",
        "open",
        "closed",
        "settled",
        "transferred_out",
        "unknown",
    }
)
_FIVE_TIER_CLASSES = frozenset({"正常", "关注", "次级", "可疑", "损失", "违约", "未分类", "unknown"})

_FIELD_ROLES: dict[str, str] = {
    "report_time": "report_datetime",
    "query_time": "report_datetime",
    "primary_id_number": "identity_document_number",
    "id_number": "identity_document_number",
    "document_number": "identity_document_number",
    "mobile_phone": "mobile_phone",
    "phone": "phone",
    "work_phone": "phone",
    "residence_phone": "phone",
    "residential_phone": "phone",
    "employer_phone": "phone",
    "account_identifier": "account_identifier",
    "credit_agreement_identifier": "account_identifier",
    "institution": "institution_name",
    "management_institution": "institution_name",
    "query_institution": "institution_name",
    "data_provider": "institution_name",
    "数据发生机构名称": "institution_name",
    "信息更新日期": "date",
    "open_date": "date_or_month",
    "due_date": "date_or_month",
    "snapshot_date": "date_or_month",
    "close_date": "date_or_month",
    "inquiry_date": "date",
    "birth_date": "date",
    "event_date": "date_or_month",
    "repayment_date": "date_or_month",
    "balance": "amount",
    "loan_amount": "amount",
    "credit_limit": "amount",
    "used_amount": "amount",
    "overdue_amount": "amount",
    "repayment_status": "repayment_status",
    "account_state": "account_state",
    "account_lifecycle_state": "account_state",
    "five_tier_class": "five_tier_class",
    "current_overdue_periods": "nonnegative_integer",
    "overdue_months": "nonnegative_integer",
    "remaining_periods": "nonnegative_integer",
    "repayment_periods": "nonnegative_integer",
    "gender": "gender",
    "marital_status": "marital_status",
    "employment_status": "employment_status",
    "education_level": "education_level",
    "degree": "degree",
    "currency": "currency",
    "account_currency": "currency",
    "responsibility_type": "responsibility_type",
    "responsible_person_type": "responsibility_type",
    "query_reason": "inquiry_reason",
    "residence_status": "residence_status",
    "postal_code": "postal_code",
    "organization_code": "organization_code",
    "nationality": "country_or_region_code",
}
_RAW_OR_PROVENANCE_KEYS = frozenset(
    {
        "raw",
        "canonical_raw",
        "raw_value",
        "raw_values",
        "raw_text",
        "raw_detail_text",
        "raw_detail_lines",
        "source_refs",
        "source_cell_refs",
        "audit",
        "ocr_correction",
        "ocr_corrections",
    }
)


@dataclass(frozen=True)
class PersonalDetailCorrectionDecision:
    correction_id: str
    role: str
    original: str
    corrected: str
    action: str
    method: str
    reason_codes: tuple[str, ...]
    confidence: float
    source_refs: tuple[dict[str, Any], ...] = ()
    candidates: tuple[str, ...] = ()
    pack_id: str = "pboc.personal_detail.zh-CN"
    pack_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PersonalDetailCellAnomaly:
    anomaly_id: str
    stage: str
    path: str
    role: str
    value: str
    reason_codes: tuple[str, ...]
    dataset_name: str = ""
    record_id: str = ""
    field_name: str = ""
    extraction_status: str = "unreadable"
    normalized_value_withheld: bool = False
    source_refs: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _pack() -> dict[str, Any]:
    payload = (
        yaml.safe_load(
            files("docmirror.plugins.credit_report.personal_detail_scanned")
            .joinpath("ocr_corrections.yaml")
            .read_text(encoding="utf-8")
        )
        or {}
    )
    if not isinstance(payload, dict) or int(payload.get("version") or 0) < 1:
        raise ValueError("invalid personal-detail OCR correction pack")
    return payload


def _plain_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\u200b", "").strip()


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _valid_date_or_month(value: str) -> bool:
    if _valid_date(value):
        return True
    try:
        datetime.strptime(value, "%Y-%m")
        return True
    except ValueError:
        return False


def _normalize_date(value: str) -> str:
    text = _plain_text(value).replace(",", ".")
    match = _DATE_TOKEN_RE.search(text) or _DATE_LOOSE_RE.search(text)
    if not match:
        return text
    candidate = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return candidate if _valid_date(candidate) else text


def _normalize_datetime(value: str) -> str:
    text = _plain_text(value).replace(",", ".")
    match = _DATETIME_DIGITS_RE.search(text)
    if not match:
        digits = re.sub(r"\D", "", text)
        if len(digits) == 14 and digits.startswith("20"):
            parts = (digits[:4], digits[4:6], digits[6:8], digits[8:10], digits[10:12], digits[12:14])
        else:
            return text
    else:
        parts = match.groups()
    candidate = f"{parts[0]}-{parts[1]}-{parts[2]}T{parts[3]}:{parts[4]}:{parts[5]}+08:00"
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return text
    return candidate


def _normalize_date_or_month(value: str) -> str:
    date = _normalize_date(value)
    if _valid_date(date):
        return date
    text = _plain_text(value).replace(",", ".")
    match = _MONTH_TOKEN_RE.search(text)
    if not match:
        return text
    candidate = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    return candidate if _valid_date_or_month(candidate) else text


def _cn_id_checksum_valid(value: str) -> bool:
    if not _CN_ID_RE.fullmatch(value):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checks = "10X98765432"
    return checks[sum(int(ch) * weight for ch, weight in zip(value[:17], weights, strict=True)) % 11] == value[-1]


def _normalize_identity(value: str) -> str:
    text = re.sub(r"[\s-]+", "", _plain_text(value)).upper()
    if _cn_id_checksum_valid(text):
        return text
    substitutions = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"})
    candidate = text.translate(substitutions)
    return candidate if _cn_id_checksum_valid(candidate) else text


def _normalize_phone(value: str, *, mobile: bool) -> str:
    text = _plain_text(value)
    digits = re.sub(r"\D", "", text)
    if mobile and _MOBILE_RE.fullmatch(digits):
        return digits
    if not mobile and 5 <= len(digits) <= 16 and not re.search(r"[A-Za-z\u3400-\u9fff]", text):
        return digits
    return text


def _normalize_identifier(value: str) -> str:
    text = re.sub(r"[\s:：,，]+", "", _plain_text(value)).upper()
    return text if _ACCOUNT_IDENTIFIER_RE.fullmatch(text) else _plain_text(value)


def _normalize_amount(value: str) -> str:
    text = _plain_text(value).replace("，", ",").replace(",", "")
    if re.fullmatch(r"[+-]?0+(?:\.0+)?", text):
        return "0"
    if text.startswith("+"):
        text = text[1:]
    if not _AMOUNT_RE.fullmatch(text):
        substitutions = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
        candidate = text.translate(substitutions)
        if not _AMOUNT_RE.fullmatch(candidate):
            return _plain_text(value)
        text = candidate
    try:
        Decimal(text)
    except InvalidOperation:
        return _plain_text(value)
    return text


def _normalize_nonnegative_integer(value: str, *, allow_placeholder: bool = False) -> str:
    text = _plain_text(value).replace(",", "").replace("，", "")
    if allow_placeholder and text in _PLACEHOLDERS:
        return text
    if re.fullmatch(r"\d{1,12}", text):
        return str(int(text))
    unit_match = re.fullmatch(r"(\d{1,12})(?:个)?(?:月|期|次|笔|家|户)", text)
    if unit_match:
        return str(int(unit_match.group(1)))
    substitutions = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
    candidate = text.upper().translate(substitutions)
    if re.fullmatch(r"\d{1,12}", candidate):
        return str(int(candidate))
    chinese_digits = {
        "〇": "0",
        "零": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }
    if text and len(text) <= 4 and all(char in chinese_digits for char in text):
        return str(int("".join(chinese_digits[char] for char in text)))
    return _plain_text(value)


def _normalize_amount_or_placeholder(value: str) -> str:
    text = _plain_text(value)
    return text if text in _PLACEHOLDERS else _normalize_amount(text)


def _normalize_business_enum(value: str, candidates: Iterable[str]) -> str:
    text = re.sub(r"\s+", "", _plain_text(value)).strip("-_:：,，;；")
    options = tuple(dict.fromkeys(str(item) for item in candidates if item))
    if text in options or not text:
        return text
    scored = sorted(
        ((SequenceMatcher(None, text, option).ratio(), option) for option in options),
        reverse=True,
    )
    if not scored:
        return text
    best_score, best = scored[0]
    runner_score = scored[1][0] if len(scored) > 1 else 0.0
    length_delta = abs(len(text) - len(best))
    if best_score >= 0.86 and best_score - runner_score >= 0.08 and length_delta <= 2:
        return best
    return text


def normalize_institution_name(value: str) -> str:
    """Return a conservative institution-name correction."""
    text = re.sub(r"\s+", " ", _plain_text(value)).strip(" -_:：,，;；")
    for original, corrected in dict(_pack().get("institution_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    text = re.sub(r"^[中福装R$证芬心多离囍版真苏德会食守]\s+(?=.{2,})", "", text)
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = _TRAILING_INSTITUTION_NOISE_RE.sub("", text).strip()
    matches = list(_INSTITUTION_SUFFIX_RE.finditer(text))
    if matches:
        # OCR debris tends to occur before or after the legal-name span. Prefer
        # the longest suffix-terminated Chinese span and preserve branch names.
        selected_match = max(matches, key=lambda match: len(match.group(0)))
        selected = selected_match.group(0)
        trailing = text[selected_match.end() :]
        specialized_tail = re.match(
            r"(?:信用卡中心|个人信贷部|银行卡业务部[（(]牡丹卡中心[）)]|"
            r"[A-Za-z0-9\u3400-\u9fff（）()·]{1,24}(?:支行|分行))",
            trailing,
        )
        if specialized_tail:
            selected += specialized_tail.group(0)
        selected = re.sub(r"^[证芬心多离囍版真苏德会食守](?=[\u3400-\u9fff]{4,})", "", selected)
        text = selected
    return text


def _normalize_inquiry_line(value: str) -> str:
    text = re.sub(r"\s+", " ", _plain_text(value)).strip()
    text = re.sub(r"(?<=\d)[,:;](?=\d)", ".", text)
    text = re.sub(r"(?<!\d)(20\d{2})[.](\d{2})(\d{2})(?!\d)", r"\1.\2.\3", text)
    text = re.sub(r"(?<!\d)(20\d{2})(\d{2})[.](\d{2})(?!\d)", r"\1.\2.\3", text)
    for original, corrected in dict(_pack().get("inquiry_reason_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    return text


def _normalize_account_line(value: str) -> str:
    text = _plain_text(value)
    for original, corrected in dict(_pack().get("account_label_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    text = re.sub(r"^[^\u3400-\u9fff0-9]*[账联][户广尸]\s*(?=\d{1,3}\b)", "账户 ", text)
    return text


def _is_valid_for_role(value: str, role: str) -> bool:
    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        validate_pboc_field,
    )

    contract = validate_pboc_field(value, role)
    if contract.assessed:
        return contract.valid
    if role == "date":
        return _valid_date(value)
    if role == "date_or_month":
        return _valid_date_or_month(value)
    if role == "report_datetime":
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            return False
    if role == "identity_document_number":
        return _cn_id_checksum_valid(value)
    if role == "mobile_phone":
        return bool(_MOBILE_RE.fullmatch(value))
    if role == "phone":
        return 5 <= len(re.sub(r"\D", "", value)) <= 16
    if role == "account_identifier":
        return bool(_ACCOUNT_IDENTIFIER_RE.fullmatch(value))
    if role == "amount":
        return bool(_AMOUNT_RE.fullmatch(value))
    if role == "amount_or_placeholder":
        return value in _PLACEHOLDERS or bool(_AMOUNT_RE.fullmatch(value))
    if role == "nonnegative_integer":
        return bool(re.fullmatch(r"\d{1,12}", value))
    if role == "integer_or_placeholder":
        return value in _PLACEHOLDERS or bool(re.fullmatch(r"\d{1,12}", value))
    if role == "institution_name":
        compact = re.sub(r"\s+", "", value)
        return bool(_INSTITUTION_SUFFIX_RE.fullmatch(compact) or compact == "本人")
    if role == "repayment_status":
        return value in _REPAYMENT_STATUSES
    if role == "account_type_label":
        return value in _ACCOUNT_TYPE_LABELS
    if role == "account_state":
        return value in _ACCOUNT_STATES
    if role == "five_tier_class":
        return value in _FIVE_TIER_CLASSES
    if role == "inquiry_row":
        return bool(_DATE_TOKEN_RE.search(value) and any(marker in value for marker in _VALID_INQUIRY_REASONS))
    return bool(value)


def _normalize_role(value: str, role: str) -> str:
    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        normalize_pboc_field,
    )

    controlled = normalize_pboc_field(value, role)
    if controlled != str(value or "").strip():
        return controlled
    if role == "date":
        return _normalize_date(value)
    if role == "date_or_month":
        return _normalize_date_or_month(value)
    if role == "report_datetime":
        return _normalize_datetime(value)
    if role == "identity_document_number":
        return _normalize_identity(value)
    if role == "mobile_phone":
        return _normalize_phone(value, mobile=True)
    if role == "phone":
        return _normalize_phone(value, mobile=False)
    if role == "account_identifier":
        return _normalize_identifier(value)
    if role == "amount":
        return _normalize_amount(value)
    if role == "amount_or_placeholder":
        return _normalize_amount_or_placeholder(value)
    if role == "nonnegative_integer":
        return _normalize_nonnegative_integer(value)
    if role == "integer_or_placeholder":
        return _normalize_nonnegative_integer(value, allow_placeholder=True)
    if role == "institution_name":
        return normalize_institution_name(value)
    if role == "repayment_status":
        return _plain_text(value).upper()
    if role == "account_type_label":
        return _normalize_business_enum(value, _ACCOUNT_TYPE_LABELS)
    if role == "account_state":
        return _normalize_business_enum(value, _ACCOUNT_STATES)
    if role == "five_tier_class":
        return _normalize_business_enum(value, _FIVE_TIER_CLASSES)
    if role == "inquiry_row":
        return _normalize_inquiry_line(value)
    if role == "account_line":
        return _normalize_account_line(value)
    return _plain_text(value)


def _source_refs(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _summary_cell_role(value: Mapping[str, Any]) -> str | None:
    label = re.sub(r"\s+", "", _plain_text(value.get("column_label")))
    if not label:
        return None
    if "账户类型" in label:
        return "account_type_label"
    if any(marker in label for marker in ("金额", "余额", "额度", "总额", "本金")):
        return "amount_or_placeholder"
    if any(marker in label for marker in ("账户数", "月份数", "月数", "笔数", "次数", "机构数", "记录数")):
        return "integer_or_placeholder"
    return None


def _mapping_role(owner: Mapping[str, Any], key: str) -> str | None:
    # These are plugin audit booleans, not PBOC count cells.  The suffix is
    # descriptive and must not route ``True`` through the integer contract.
    if key in {
        "status_inferred_from_adjacent_months",
        "normalized_value_withheld",
        "is_primary",
        "ocr_corrected",
    }:
        return None
    if key == "value":
        return _summary_cell_role(owner)
    if key == "reason" and any(name in owner for name in ("inquiry_date", "inquiry_type")):
        return "inquiry_reason"
    role = _FIELD_ROLES.get(key)
    if role:
        return role
    if key == "status" and {"year", "month"} <= set(owner):
        return "repayment_status"
    if key.endswith("_date") and not key.endswith("_date_text"):
        return "date_or_month"
    if key.endswith("_amount") or key in {
        "actual_payment",
        "balance",
        "credit_limit",
        "loan_amount",
        "maximum_overdraft_balance",
        "maximum_used_amount",
        "scheduled_payment",
        "shared_credit_limit",
        "total_limit",
        "unbilled_installment_balance",
        "used_amount",
        "used_limit",
    }:
        return "amount"
    if key.endswith(("_count", "_periods", "_months")) or key in {"sequence", "year", "month"}:
        return "nonnegative_integer"
    return None


def _cell_scoped(refs: Iterable[dict[str, Any]]) -> bool:
    return any(ref.get("geometry_scope") == "cell" for ref in refs)


def _boxes_associate(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> bool:
    """Return whether a page-wide OCR token belongs to a schema field box."""
    tx0, ty0, tx1, ty1 = target
    cx0, cy0, cx1, cy1 = candidate
    intersection_width = max(0.0, min(tx1, cx1) - max(tx0, cx0))
    intersection_height = max(0.0, min(ty1, cy1) - max(ty0, cy0))
    intersection = intersection_width * intersection_height
    target_area = max(1.0, (tx1 - tx0) * (ty1 - ty0))
    candidate_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
    if intersection / min(target_area, candidate_area) >= 0.25:
        return True
    # OCR boxes can move slightly between complete-page passes.  Permit a
    # bounded half-line-height halo while keeping the association local.
    halo = max(2.0, min(12.0, (ty1 - ty0) * 0.5))
    center_x = (cx0 + cx1) / 2.0
    center_y = (cy0 + cy1) / 2.0
    return tx0 - halo <= center_x <= tx1 + halo and ty0 - halo <= center_y <= ty1 + halo


class PersonalDetailOCRCorrectionOverlay:
    """Schema-independent corrected view over one sealed personal report."""

    def __init__(
        self,
        parse_result: Any,
        *,
        page_image_resolver: Any | None = None,
        full_page_ocr_loader: Any | None = None,
        repair_engine: Any | None = None,
        enable_targeted_ocr: bool | None = None,
        max_targeted_requests: int = 8,
    ) -> None:
        self.parse_result = parse_result
        self._page_image_resolver = page_image_resolver
        self._full_page_ocr_loader = full_page_ocr_loader
        self._repair_engine = repair_engine
        self.enable_targeted_ocr = (
            os.environ.get("DOCMIRROR_PERSONAL_DETAIL_TARGETED_OCR", "1") != "0"
            if enable_targeted_ocr is None
            else bool(enable_targeted_ocr)
        )
        self.max_targeted_requests = max(0, int(max_targeted_requests))
        self._targeted_requests = 0
        self._decisions: list[PersonalDetailCorrectionDecision] = []
        self._decision_keys: set[tuple[str, str, str, str]] = set()
        self._audited_cells: set[tuple[str, str, str, str]] = set()
        self._cell_anomalies: list[PersonalDetailCellAnomaly] = []
        self._cell_anomaly_keys: set[tuple[str, str, str, str, str]] = set()

    @property
    def decisions(self) -> tuple[PersonalDetailCorrectionDecision, ...]:
        return tuple(self._decisions)

    def audit(self) -> dict[str, Any]:
        counts = Counter(decision.action for decision in self._decisions)
        return {
            "pack_id": str(_pack().get("pack_id") or "pboc.personal_detail.zh-CN"),
            "pack_version": int(_pack().get("version") or 1),
            "decision_count": len(self._decisions),
            "applied_count": counts.get("applied", 0),
            "suggested_count": counts.get("suggested", 0),
            "targeted_ocr_requests": self._targeted_requests,
            "audited_cell_count": len(self._audited_cells),
            "abnormal_cell_count": len(self._cell_anomalies),
            "cell_anomalies": [anomaly.to_dict() for anomaly in self._cell_anomalies],
            "decisions": [decision.to_dict() for decision in self._decisions],
        }

    def _audit_cell(
        self,
        *,
        stage: str,
        path: str,
        role: str,
        dataset_name: str,
        record_id: str,
        field_name: str,
        value: Any,
        refs: tuple[dict[str, Any], ...],
        valid: bool,
        normalized_value_withheld: bool = False,
        reason_codes: tuple[str, ...] | None = None,
    ) -> None:
        text = str(value or "")
        marker = (stage, path, role, repr(refs))
        self._audited_cells.add(marker)
        if valid:
            return
        anomaly_marker = (stage, path, role, text, repr(refs))
        if anomaly_marker in self._cell_anomaly_keys:
            return
        self._cell_anomaly_keys.add(anomaly_marker)
        digest = hashlib.sha256("\x1f".join(anomaly_marker).encode("utf-8")).hexdigest()[:16]
        self._cell_anomalies.append(
            PersonalDetailCellAnomaly(
                anomaly_id=f"personal_detail_cell:{digest}",
                stage=stage,
                path=path,
                role=role,
                value=text,
                reason_codes=reason_codes
                or (
                    "role_validation_failed",
                    "normalized_value_withheld" if normalized_value_withheld else "preserved_unresolved_value",
                ),
                dataset_name=dataset_name,
                record_id=record_id,
                field_name=field_name,
                extraction_status="unreadable",
                normalized_value_withheld=normalized_value_withheld,
                source_refs=refs,
            )
        )

    def _record(
        self,
        *,
        role: str,
        original: str,
        corrected: str,
        method: str,
        reason_codes: tuple[str, ...],
        confidence: float,
        refs: tuple[dict[str, Any], ...],
        candidates: tuple[str, ...] = (),
        action: str = "applied",
    ) -> PersonalDetailCorrectionDecision:
        marker = (role, original, corrected, repr(refs))
        digest = hashlib.sha256("\x1f".join(marker).encode("utf-8")).hexdigest()[:16]
        decision = PersonalDetailCorrectionDecision(
            correction_id=f"personal_detail_ocr:{digest}",
            role=role,
            original=original,
            corrected=corrected,
            action=action,
            method=method,
            reason_codes=reason_codes,
            confidence=max(0.0, min(1.0, float(confidence))),
            source_refs=refs,
            candidates=candidates,
            pack_id=str(_pack().get("pack_id") or "pboc.personal_detail.zh-CN"),
            pack_version=int(_pack().get("version") or 1),
        )
        if marker not in self._decision_keys:
            self._decision_keys.add(marker)
            self._decisions.append(decision)
        return decision

    def correct_text(
        self,
        value: Any,
        *,
        role: str,
        source_refs: Iterable[dict[str, Any]] = (),
        confidence: float | None = None,
        allow_targeted_ocr: bool = False,
    ) -> tuple[str, PersonalDetailCorrectionDecision | None]:
        original = str(value or "")
        refs = _source_refs(source_refs)
        corrected = _normalize_role(original, role)
        if corrected != original and _is_valid_for_role(corrected, role):
            return corrected, self._record(
                role=role,
                original=original,
                corrected=corrected,
                method="typed_deterministic",
                reason_codes=("role_scoped_normalization", "typed_validation"),
                confidence=max(0.98, float(confidence or 0.0)),
                refs=refs,
            )
        if allow_targeted_ocr and not _is_valid_for_role(corrected, role):
            repaired = self._targeted_repair(original, role=role, refs=refs)
            if repaired is not None:
                return repaired
        return original, None

    def _targeted_repair(
        self,
        original: str,
        *,
        role: str,
        refs: tuple[dict[str, Any], ...],
    ) -> tuple[str, PersonalDetailCorrectionDecision] | None:
        if not self.enable_targeted_ocr or self._targeted_requests >= self.max_targeted_requests:
            return None
        ref = next(
            (
                item
                for item in refs
                if isinstance(item.get("bbox"), (list, tuple))
                and len(item["bbox"]) == 4
                and int(item.get("logical_page") or item.get("page") or 0) > 0
            ),
            None,
        )
        if ref is None:
            return None
        logical_page = int(ref.get("logical_page") or ref.get("page") or 0)
        self._targeted_requests += 1
        if not callable(self._full_page_ocr_loader):
            return None
        # Re-OCR the complete logical page once and cache it in the extraction
        # context.  The bbox is only a schema-aware association hint; it never
        # becomes the OCR input region.
        return self._repair_from_full_page(
            original,
            role=role,
            ref=ref,
            refs=refs,
            logical_page=logical_page,
        )

    def _repair_from_full_page(
        self,
        original: str,
        *,
        role: str,
        ref: dict[str, Any],
        refs: tuple[dict[str, Any], ...],
        logical_page: int,
    ) -> tuple[str, PersonalDetailCorrectionDecision] | None:
        """Select a typed value from one cached complete-page OCR pass.

        The source bbox is used only to associate the already page-wide OCR
        result with its schema-assigned field.  OCR itself is never rerun on a
        crop here.
        """
        pages = self._full_page_ocr_loader(
            {logical_page},
            reason=f"schema_role_repair:{role}",
        )
        target = tuple(float(item) for item in ref.get("bbox") or ())
        if len(target) != 4:
            return None
        candidates: list[tuple[float, str]] = []
        for page in pages or ():
            if int(page.get("page") or 0) != logical_page:
                continue
            selected: list[tuple[float, float, str, float]] = []
            for line in page.get("lines") or ():
                if not isinstance(line, dict):
                    continue
                bbox = line.get("bbox")
                text = str(line.get("text") or line.get("content") or "").strip()
                if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                candidate_box = tuple(float(item) for item in bbox)
                if not _boxes_associate(target, candidate_box):
                    continue
                confidence = float(line.get("confidence") or 0.0)
                selected.append((candidate_box[1], candidate_box[0], text, confidence))
                normalized = _normalize_role(text, role)
                if _is_valid_for_role(normalized, role):
                    candidates.append((confidence, normalized))
            if selected:
                joined = " ".join(item[2] for item in sorted(selected))
                normalized = _normalize_role(joined, role)
                if _is_valid_for_role(normalized, role):
                    candidates.append((min(item[3] for item in selected), normalized))
        valid = sorted(candidates, reverse=True)
        if not valid or valid[0][0] < 0.72:
            return None
        if len(valid) > 1 and valid[0][1] != valid[1][1] and valid[0][0] - valid[1][0] < 0.08:
            return None
        selected = valid[0][1]
        return selected, self._record(
            role=role,
            original=original,
            corrected=selected,
            method="full_page_ocr_role_reparse",
            reason_codes=("complete_logical_page_rerendered", "schema_role_validation", "candidate_margin"),
            confidence=valid[0][0],
            refs=refs,
            candidates=tuple(dict.fromkeys(value for _score, value in valid)),
        )

    @staticmethod
    def _line_role(text: str) -> str | None:
        compact = re.sub(r"\s+", "", text)
        has_inquiry_reason = any(marker in compact for marker in _INQUIRY_REASON_MARKERS)
        has_inquiry_date = bool(_INQUIRY_DATE_LIKE_RE.search(text))
        has_damaged_inquiry_date = bool(re.search(r"(?:^|\D)\d{0,3}\D{0,3}20\d{2}", text))
        if has_inquiry_reason and (has_inquiry_date or has_damaged_inquiry_date):
            return "inquiry_row"
        if re.search(r"^[^\u3400-\u9fff0-9]*[账联][户广尸]\s*\d{1,3}\b", text):
            return "account_line"
        return None

    def corrected_evidence_pages(self, pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        corrected_pages: list[dict[str, Any]] = []
        for raw_page in pages:
            page = deepcopy(raw_page)
            corrected_lines: list[dict[str, Any]] = []
            for raw_line in page.get("lines") or []:
                line = dict(raw_line)
                original = str(line.get("text") or line.get("content") or "")
                role = self._line_role(original)
                if role:
                    corrected, decision = self.correct_text(
                        original,
                        role=role,
                        source_refs=(
                            {
                                "source": "scanned_ocr_line",
                                "logical_page": int(page.get("page") or 0),
                                "source_page": int(page.get("source_page") or 0),
                                "bbox": list(line.get("bbox") or []),
                                "evidence_ids": list(line.get("evidence_ids") or []),
                            },
                        ),
                        confidence=float(line.get("confidence") or 0.0),
                        allow_targeted_ocr=role == "inquiry_row",
                    )
                    if decision is not None:
                        line["ocr_original_text"] = original
                        line["ocr_correction"] = decision.to_dict()
                        line["text"] = corrected
                        if "content" in line:
                            line["content"] = corrected
                corrected_lines.append(line)
            page["lines"] = corrected_lines
            corrected_pages.append(page)
        return corrected_pages

    def correct_business_candidates(self, payload: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
        corrected = deepcopy(dict(payload))
        self._walk(corrected, parent="", refs=(), stage=stage)
        self._enforce_cross_field_contracts(corrected, stage=stage)
        self._apply_institution_consensus(corrected)
        self._promote_account_identifier_candidates(corrected)
        return corrected

    def enforce_cross_field_contracts(self, payload: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
        """Validate merged records without replaying all single-field corrections."""
        corrected = deepcopy(dict(payload))
        self._enforce_cross_field_contracts(corrected, stage=stage)
        return corrected

    def _enforce_cross_field_contracts(self, payload: dict[str, Any], *, stage: str) -> None:
        """Withhold individually valid values that violate the dataset schema."""
        for index, record in enumerate(payload.get("credit_lines") or [], start=1):
            if not isinstance(record, dict):
                continue
            pools = [record]
            if isinstance(record.get("normalized"), dict):
                pools.append(record["normalized"])
            original: Any | None = None
            for pool in pools:
                try:
                    total_limit = Decimal(str(pool.get("total_limit")))
                    used_limit = Decimal(str(pool.get("used_limit")))
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if total_limit < 0 or used_limit < 0 or used_limit <= total_limit:
                    continue
                if original is None:
                    original = pool.get("used_limit")
                pool["used_limit"] = None
                if "used_limit_status" in pool:
                    pool["used_limit_status"] = "unknown"
            if original is None:
                continue
            refs = _source_refs(record.get("source_refs"))
            record_id = str(record.get("record_id") or record.get("credit_line_id") or f"row:{index}")
            self._audit_cell(
                stage=stage,
                path=f"credit_lines[{record_id}].used_limit",
                role="amount",
                dataset_name="credit_lines",
                record_id=record_id,
                field_name="used_limit",
                value=original,
                refs=refs,
                valid=False,
                normalized_value_withheld=True,
                reason_codes=(
                    "cross_field_contract_failed",
                    "used_limit_exceeds_total_limit",
                    "normalized_value_withheld",
                ),
            )

    def _walk(
        self,
        value: Any,
        *,
        parent: str | None,
        refs: tuple[dict[str, Any], ...],
        stage: str,
    ) -> None:
        if isinstance(value, list):
            for item in value:
                self._walk(item, parent=parent, refs=refs, stage=stage)
            return
        if not isinstance(value, dict):
            return
        local_refs = _source_refs(value.get("source_refs")) or refs
        confidence = value.get("confidence")
        node_id = str(value.get("record_id") or value.get("summary_cell_id") or value.get("account_id") or "")
        base_path = f"{parent}[{node_id}]" if node_id else str(parent or "")
        dataset_name = str(parent or "").split(".", 1)[0].split("[", 1)[0]
        for key, item in list(value.items()):
            field_path = f"{base_path}.{key}".lstrip(".")
            role = _mapping_role(value, str(key))
            if role and item not in (None, "") and not isinstance(item, (dict, list)):
                cell_target = (
                    stage == "native_business"
                    and _cell_scoped(local_refs)
                    and role
                    in {
                        "account_type_label",
                        "amount_or_placeholder",
                        "integer_or_placeholder",
                    }
                )
                updated, decision = self.correct_text(
                    item,
                    role=role,
                    source_refs=local_refs,
                    confidence=float(confidence or 0.0),
                    allow_targeted_ocr=len(local_refs) == 1
                    and (
                        cell_target
                        or role
                        in {
                            "identity_document_number",
                            "report_datetime",
                            "date",
                            "date_or_month",
                        }
                    ),
                )
                if decision is not None:
                    value[key] = updated
                final_value = value[key]
                valid = _is_valid_for_role(str(final_value), role)
                withhold_invalid = bool(
                    not valid and stage == "native_business" and len(local_refs) == 1 and _cell_scoped(local_refs)
                )
                if withhold_invalid:
                    value[key] = None
                self._audit_cell(
                    stage=stage,
                    path=field_path,
                    role=role,
                    dataset_name=dataset_name,
                    record_id=node_id,
                    field_name=str(key),
                    value=final_value,
                    refs=local_refs,
                    valid=valid,
                    normalized_value_withheld=withhold_invalid,
                )
            elif isinstance(item, dict) and role and item.get("value") not in (None, ""):
                nested_refs = _source_refs(item.get("source_refs")) or local_refs
                updated, decision = self.correct_text(
                    item["value"],
                    role=role,
                    source_refs=nested_refs,
                    confidence=float(item.get("confidence") or confidence or 0.0),
                    allow_targeted_ocr=role in {"identity_document_number", "report_datetime", "date", "date_or_month"},
                )
                if decision is not None:
                    item.setdefault("raw", item["value"])
                    item["value"] = updated
                final_value = item["value"]
                valid = _is_valid_for_role(str(final_value), role)
                withhold_invalid = bool(
                    not valid and stage == "native_business" and len(nested_refs) == 1 and _cell_scoped(nested_refs)
                )
                if withhold_invalid:
                    item.setdefault("raw", final_value)
                    item["value"] = None
                self._audit_cell(
                    stage=stage,
                    path=f"{field_path}.value",
                    role=role,
                    dataset_name=dataset_name,
                    record_id=node_id,
                    field_name=str(key),
                    value=final_value,
                    refs=nested_refs,
                    valid=valid,
                    normalized_value_withheld=withhold_invalid,
                )
            if isinstance(item, (dict, list)) and str(key) not in _RAW_OR_PROVENANCE_KEYS:
                self._walk(item, parent=field_path, refs=local_refs, stage=stage)

    def _apply_institution_consensus(self, payload: dict[str, Any]) -> None:
        fields: list[tuple[dict[str, Any], str, str]] = []

        def collect(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if _FIELD_ROLES.get(str(key)) == "institution_name" and isinstance(item, str) and item:
                        fields.append((value, str(key), item))
                    collect(item)

        collect(payload)
        counts = Counter(normalize_institution_name(item) for _owner, _key, item in fields)
        references = [
            name for name, count in counts.items() if count >= 1 and _is_valid_for_role(name, "institution_name")
        ]
        for owner, key, original in fields:
            current = normalize_institution_name(original)
            scored = sorted(
                (
                    (SequenceMatcher(None, current, candidate).ratio(), counts[candidate], candidate)
                    for candidate in references
                ),
                reverse=True,
            )
            if not scored:
                continue
            best_score, best_count, best = scored[0]
            runner_score = scored[1][0] if len(scored) > 1 else 0.0
            if (
                best != current
                and best_score >= 0.94
                and best_score - runner_score >= 0.015
                and best_count >= counts[current]
            ):
                owner[key] = best
                self._record(
                    role="institution_name",
                    original=original,
                    corrected=best,
                    method="document_internal_consensus",
                    reason_codes=("typed_legal_suffix", "document_candidate_match", "candidate_margin"),
                    confidence=best_score,
                    refs=(),
                    candidates=tuple(item[2] for item in scored[:3]),
                )

    def _promote_account_identifier_candidates(self, payload: dict[str, Any]) -> None:
        accounts = payload.get("credit_accounts")
        if not isinstance(accounts, list):
            return
        for account in accounts:
            if not isinstance(account, dict) or account.get("account_identifier"):
                continue
            raw_candidates = account.get("account_identifier_candidates")
            if not isinstance(raw_candidates, list):
                continue
            candidates = tuple(
                dict.fromkeys(
                    normalized
                    for item in raw_candidates
                    if (normalized := _normalize_identifier(str(item or "")))
                    and _is_valid_for_role(normalized, "account_identifier")
                )
            )
            if len(candidates) != 1:
                continue
            account["account_identifier"] = candidates[0]
            self._record(
                role="account_identifier",
                original="",
                corrected=candidates[0],
                method="unique_typed_candidate",
                reason_codes=("missing_typed_field", "single_valid_candidate", "identifier_character_set"),
                confidence=float(account.get("confidence") or 0.9),
                refs=_source_refs(account.get("source_refs")),
                candidates=candidates,
            )


__all__ = [
    "PersonalDetailCorrectionDecision",
    "PersonalDetailOCRCorrectionOverlay",
    "normalize_institution_name",
]
