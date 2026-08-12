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
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib.resources import files
from math import isfinite
from typing import Any, Iterable, Mapping

import yaml

_DATE_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)")
_DATE_LOOSE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[.,/-]?(\d{2})[.,/-](\d{2})(?!\d)")
_MONTH_TOKEN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[./-](\d{1,2})(?![\d./-])")
_DATETIME_DIGITS_RE = re.compile(r"(?<!\d)(20\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})\D*(\d{2})(?!\d)")
_CN_ID_RE = re.compile(r"^\d{17}[0-9X]$")
_IDENTITY_NUMBER_TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "primary_id_number": ("primary_id_type", "id_type", "document_type"),
    "id_number": ("id_type", "primary_id_type", "document_type"),
    "document_number": ("document_type", "primary_id_type", "id_type"),
}
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
_INSTITUTION_INTERNAL_DASHES = "-‐‑‒–—―－"
_INSTITUTION_SUFFIX_RE = re.compile(
    rf"[A-Za-z0-9\u3400-\u9fff（）()·{re.escape(_INSTITUTION_INTERNAL_DASHES)}]{{2,100}}?(?:"
    r"银行卡业务部[（(]牡丹卡中心[）)]|信用卡中心|个人信贷部|"
    r"支行|分行|"
    r"农村信用合作联社|农村信用社联合社|股份有限公司|股份公司|有限责任公司|有限公司|管理中心"
    r")"
)
_LEADING_ROW_NOISE_RE = re.compile(r"^[\s\W_]*(?:[A-Za-z\u3400-\u9fff]{1,2}\s+)?(?=\d{0,3}\s*20\d{2})")
_TRAILING_INSTITUTION_NOISE_RE = re.compile(r"(?:\s+[A-Za-z0-9￥¥?$]{1,3})+$")
_REPAYMENT_STATUSES = frozenset(
    {"*", "/", "#", "N", "A", "C", "M", "B", "D", "Z", "G", "unknown", *"1234567"}
)
_INSTITUTION_ROOT_SUFFIX_RE = re.compile(
    r"农村信用合作联社|农村信用社联合社|股份有限公司|股份公司|"
    r"有限责任公司|有限公司|管理中心"
)
_INSTITUTION_BRANCH_SUFFIX_RE = re.compile(r"信用卡中心|个人信贷部|支行|分行")
_INSTITUTION_ADJACENT_LABELS = frozenset(
    {
        "管理机构",
        "发卡机构",
        "机构名称",
        "数据发生机构名称",
        "账户标识",
        "开立日期",
        "生效日期",
        "到期日期",
        "账户授信额度",
        "共享授信额度",
        "授信额度用途",
        "币种",
        "业务种类",
        "担保方式",
        "还款频率",
        "还款方式",
    }
)
_PLACEHOLDERS = frozenset({"-", "--"})
_ACCOUNT_TYPE_LABELS = (
    "非循环贷账户",
    "循环贷账户一",
    "循环贷账户二",
    "贷记卡账户",
    "准贷记卡账户",
)
_SUMMARY_BUSINESS_CATEGORIES = (
    "个人住房贷款",
    "个人商用房贷款",
    "其他类贷款",
    "贷记卡",
    "准贷记卡",
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
_ACCOUNT_STATUS_CODES = frozenset(
    {
        "active",
        "settled",
        "closed",
        "transferred_out",
        "inactive",
        "frozen",
        "suspended",
        "recovery",
        "collateral_shortfall",
        "forced_liquidation",
        "collection",
        "unknown",
    }
)
_FIVE_TIER_CLASSES = frozenset({"正常", "关注", "次级", "可疑", "损失", "违约", "未分类", "unknown"})

_FIELD_ROLES: dict[str, str] = {
    "subject_name": "person_name",
    "holder_name": "person_name",
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
    "account_status": "account_status_code",
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
    "address": "address",
    "mailing_address": "address",
    "household_address": "address",
    "communication_address": "address",
    "employer": "employer_name",
    "work_unit": "employer_name",
    "facility_type": "facility_type",
    "guarantee_type": "guarantee_type",
    "repayment_frequency": "repayment_frequency",
    "repayment_method": "repayment_method",
    "industry": "employment_industry",
    "occupation": "employment_occupation",
    "position": "employment_position",
    "job_title": "employment_descriptor",
    "professional_title": "employment_professional_title",
    "primary_id_type": "identity_document_type",
    "document_type": "identity_document_type",
    "related_party_id_type": "identity_document_type",
    "co_borrower_flag": "boolean_flag",
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
        "source_refs_by_field",
        "_field_binding_quality",
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


def _normalized_date_candidate_spans(value: str) -> list[tuple[tuple[int, int], str]]:
    text = _plain_text(value).replace(",", ".")
    candidates: dict[tuple[int, int], str] = {}
    for pattern in (_DATE_TOKEN_RE, _DATE_LOOSE_RE):
        for match in pattern.finditer(text):
            candidate = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
            if _valid_date(candidate):
                candidates.setdefault(match.span(), candidate)
    return sorted(candidates.items())


def _normalized_date_candidates(value: str) -> list[str]:
    return [candidate for _span, candidate in _normalized_date_candidate_spans(value)]


def _short_ascii_date_residue(value: str) -> bool:
    residue = re.sub(r"\s+", "", value)
    if not residue:
        return True
    return bool(
        residue.isascii()
        and len(residue) <= 3
        and not any(character.isdigit() for character in residue)
        and sum(character.isalpha() for character in residue) <= 2
        and all(
            character.isalpha() or character in r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
            for character in residue
        )
    )


def _one_safe_date_candidate(value: str) -> str | None:
    text = _plain_text(value).replace(",", ".")
    candidates = _normalized_date_candidate_spans(text)
    if len(candidates) != 1:
        return None
    (start, end), candidate = candidates[0]
    residue = text[:start] + text[end:]
    return candidate if _short_ascii_date_residue(residue) else None


def _normalize_date(value: str) -> str:
    text = _plain_text(value).replace(",", ".")
    return _one_safe_date_candidate(text) or text


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
    text = _plain_text(value).replace(",", ".")
    date_candidates = _normalized_date_candidate_spans(text)
    if len(date_candidates) == 1:
        (start, end), candidate = date_candidates[0]
        residue = text[:start] + text[end:]
        return candidate if _short_ascii_date_residue(residue) else text
    if date_candidates:
        return text
    month_candidates = [
        (match.span(), f"{int(match.group(1)):04d}-{int(match.group(2)):02d}")
        for match in _MONTH_TOKEN_RE.finditer(text)
    ]
    valid_months = [item for item in month_candidates if _valid_date_or_month(item[1])]
    if len(valid_months) != 1:
        return text
    (start, end), candidate = valid_months[0]
    residue = text[:start] + text[end:]
    return candidate if _short_ascii_date_residue(residue) else text


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
    digit_groups = re.findall(r"\d+", text)
    if len(digit_groups) > 1 and re.search(r"\d\s+\d", text) and len(digits) > 12:
        # A common table-collapse failure joins a telephone number to a
        # neighbouring numeric cell.  Flattening that sequence would turn
        # structure damage into a plausible-looking telephone number.
        return text
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


def _normalize_business_enum(value: str, _candidates: Iterable[str]) -> str:
    return re.sub(r"\s+", "", _plain_text(value)).strip("-_:：,，;；")


def _normalize_summary_business_category(value: str) -> str:
    text = re.sub(r"\s+", "", _plain_text(value)).strip("-_:：,，;；")
    return text


def institution_slot_is_unambiguous(value: str) -> bool:
    """Return whether one slot contains exactly one institution-name span."""

    text = re.sub(r"\s+", "", _plain_text(value))
    if not text or any(label in text for label in _INSTITUTION_ADJACENT_LABELS):
        return False
    root_count = len(_INSTITUTION_ROOT_SUFFIX_RE.findall(text))
    if root_count > 1:
        return False
    if root_count == 0 and len(_INSTITUTION_BRANCH_SUFFIX_RE.findall(text)) > 1:
        return False
    return True


def institution_name_has_separated_leading_han(value: str) -> bool:
    """Return whether OCR separated one leading Han glyph from the name.

    That boundary is not self-interpreting: it can be an intra-name OCR space
    (``中 国银行``) or a glyph copied from the neighbouring cell
    (``福 中信银行``).  Callers must therefore require independent,
    source-bound corroboration before silently publishing the joined value.
    """

    return bool(re.match(r"^[\u3400-\u9fff]\s+(?=[\u3400-\u9fff])", _plain_text(value).strip()))


def normalize_institution_name(value: str) -> str:
    """Return a conservative institution-name correction."""
    text = re.sub(r"\s+", " ", _plain_text(value)).strip(" -_:：,，;；")
    if not institution_slot_is_unambiguous(text):
        return text
    for original, corrected in dict(_pack().get("institution_substitutions") or {}).items():
        text = text.replace(str(original), str(corrected))
    if not institution_slot_is_unambiguous(text):
        return text
    isolated_suffix_fragment = re.fullmatch(
        r"(?:[A-Za-z]|[有限责任股份公司])\s+(.{4,}(?:有限公司|股份有限公司|有限责任公司|公司))",
        text,
    )
    if isolated_suffix_fragment:
        # A separated single glyph copied from a neighbouring legal suffix is
        # OCR boundary debris, not part of the individualized institution.
        text = isolated_suffix_fragment.group(1)
    # Never discard a separated Han glyph here.  Text alone cannot distinguish
    # an OCR word break from cross-cell debris; the schema caller resolves that
    # boundary from independent source-bound observations.
    text = re.sub(r"^[R$]\s+(?=.{2,})", "", text)
    text = re.sub(
        r"^[导务]\s*(?=.{2,}(?:银行|公司|中心|支行|分行|营业部))",
        "",
        text,
    )
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(
        rf"\s*([{re.escape(_INSTITUTION_INTERNAL_DASHES)}])\s*",
        r"\1",
        text,
    )
    text = _TRAILING_INSTITUTION_NOISE_RE.sub("", text).strip()
    matches = list(_INSTITUTION_SUFFIX_RE.finditer(text))
    if matches:
        # OCR debris tends to occur before or after the legal-name span. Prefer
        # the longest suffix-terminated Chinese span and preserve branch names.
        # Expand every legal-root candidate with its immediately following
        # branch before ranking.  Otherwise a standalone branch span can be
        # longer than the bank/company root and incorrectly discard that root.
        expanded: list[str] = []
        for match in matches:
            selected = match.group(0)
            trailing = text[match.end() :]
            specialized_tail = re.match(
                r"(?:信用卡中心|个人信贷部|银行卡业务部[（(]牡丹卡中心[）)]|"
                rf"[A-Za-z0-9\u3400-\u9fff（）()·{re.escape(_INSTITUTION_INTERNAL_DASHES)}]{{1,24}}(?:支行|分行))",
                trailing,
            )
            if specialized_tail:
                selected += specialized_tail.group(0)
            expanded.append(selected)
        text = max(expanded, key=len)
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
    if role.startswith("identity_document_number::"):
        document_type = role.partition("::")[2]
        compact = re.sub(r"\s+", "", _plain_text(value)).upper()
        if document_type in {"身份证", "居民身份证"}:
            return _cn_id_checksum_valid(compact)
        if document_type == "护照":
            return bool(
                re.fullmatch(r"(?=.{5,20}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+", compact)
            ) and not bool(_CN_ID_RE.fullmatch(compact))
        if document_type in {"军官证", "士兵证"}:
            return bool(
                re.fullmatch(
                    r"(?:[A-Z](?=[A-Z0-9-]{4,19}$)(?=.*\d)[A-Z0-9-]+|"
                    r"[\u3400-\u9fff]{1,4}字第?\d{4,12}号?)",
                    compact,
                )
            )
        if document_type == "港澳居民来往内地通行证":
            return bool(re.fullmatch(r"[HM]\d{8,10}", compact))
        if document_type == "台湾居民来往大陆通行证":
            return bool(re.fullmatch(r"(?:[A-Z]\d{7,9}|\d{8,10})", compact))
        if document_type == "外国人永久居留证":
            return bool(re.fullmatch(r"(?:[A-Z]{3}\d{12}|9\d{16}[0-9X])", compact))
        if document_type == "统一社会信用代码":
            return bool(re.fullmatch(r"[0-9A-HJ-NPQRTUWXY]{18}", compact))
        if document_type == "中征码":
            return bool(re.fullmatch(r"\d{16}", compact))
        if document_type == "组织机构代码":
            return bool(re.fullmatch(r"[A-Z0-9]{8}-?[A-Z0-9]", compact))
        # `其他证件` and `未知` do not identify a safe number grammar.
        return False
    if role == "identity_document_number":
        return _cn_id_checksum_valid(value)
    if role == "mobile_phone":
        return bool(_MOBILE_RE.fullmatch(value))
    if role == "phone":
        digits = re.sub(r"\D", "", value)
        if re.search(r"[A-Za-z\u3400-\u9fff]", value):
            return False
        if len(re.findall(r"\d+", value)) > 1 and re.search(r"\d\s+\d", value) and len(digits) > 12:
            return False
        return 5 <= len(digits) <= 16
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
        return bool(
            institution_slot_is_unambiguous(compact)
            and (_INSTITUTION_SUFFIX_RE.fullmatch(compact) or compact == "本人")
        )
    if role == "repayment_status":
        # ``unknown`` is an extraction state, never a printed PBOC monthly
        # symbol.  Treating it as valid prevented the only permitted page
        # repair from being selected and made a silent miss look complete.
        return value != "unknown" and value in _REPAYMENT_STATUSES
    if role == "account_type_label":
        return value in _ACCOUNT_TYPE_LABELS
    if role == "summary_business_category":
        return value in _PLACEHOLDERS or value in _SUMMARY_BUSINESS_CATEGORIES
    if role == "account_state":
        return value != "unknown" and value in _ACCOUNT_STATES
    if role == "account_status_code":
        return value != "unknown" and value in _ACCOUNT_STATUS_CODES
    if role == "five_tier_class":
        return value != "unknown" and value in _FIVE_TIER_CLASSES
    if role.startswith("employment_"):
        # The native extractor and the final correction plane must share the
        # same closed PBOC vocabularies.  A generic text-shape contract both
        # rejected canonical comma-delimited occupations and admitted
        # arbitrary Han near-matches after extraction had finished.
        from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
            _EMPLOYMENT_INDUSTRIES,
            _EMPLOYMENT_OCCUPATIONS,
            _EMPLOYMENT_POSITIONS,
            _EMPLOYMENT_TITLES,
        )

        vocabulary = {
            "employment_industry": _EMPLOYMENT_INDUSTRIES,
            "employment_occupation": _EMPLOYMENT_OCCUPATIONS,
            "employment_position": _EMPLOYMENT_POSITIONS,
            "employment_professional_title": _EMPLOYMENT_TITLES,
        }.get(role)
        return vocabulary is not None and value in vocabulary
    if role == "inquiry_row":
        return bool(_DATE_TOKEN_RE.search(value) and any(marker in value for marker in _VALID_INQUIRY_REASONS))
    return bool(value)


def _normalize_role(value: str, role: str) -> str:
    from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
        normalize_pboc_field,
    )

    text = _plain_text(value)
    substitutions = dict(_pack().get("profile_field_substitutions") or {}).get(role)
    if isinstance(substitutions, Mapping):
        corrected = substitutions.get(text)
        if corrected is not None:
            text = str(corrected)
    controlled = normalize_pboc_field(text, role)
    if controlled != text:
        return controlled
    if role == "date":
        return _normalize_date(value)
    if role == "date_or_month":
        return _normalize_date_or_month(value)
    if role == "report_datetime":
        return _normalize_datetime(value)
    if role.startswith("identity_document_number::"):
        document_type = role.partition("::")[2]
        if document_type in {"身份证", "居民身份证"}:
            return _normalize_identity(value)
        return re.sub(r"\s+", "", _plain_text(value)).upper().translate(
            str.maketrans({"―": "-", "‐": "-", "‑": "-", "–": "-", "—": "-", "－": "-"})
        )
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
    if role == "employer_name":
        return normalize_institution_name(value)
    if role == "repayment_status":
        return _plain_text(value).upper()
    if role == "account_type_label":
        return _normalize_business_enum(value, _ACCOUNT_TYPE_LABELS)
    if role == "summary_business_category":
        return _normalize_summary_business_category(value)
    if role == "account_state":
        return _normalize_business_enum(value, _ACCOUNT_STATES)
    if role == "account_status_code":
        return _normalize_business_enum(value, _ACCOUNT_STATUS_CODES)
    if role == "five_tier_class":
        return _normalize_business_enum(value, _FIVE_TIER_CLASSES)
    if role == "inquiry_row":
        return _normalize_inquiry_line(value)
    if role == "inquiry_reason":
        text = _plain_text(value)
        for original, corrected in dict(_pack().get("inquiry_reason_substitutions") or {}).items():
            text = text.replace(str(original), str(corrected))
        return _normalize_business_enum(text, _VALID_INQUIRY_REASONS)
    if role == "account_line":
        return _normalize_account_line(value)
    return text


def normalize_role_candidate(value: Any, role: str) -> str:
    """Expose the field-specific normalizer to the page repair coordinator."""
    return _normalize_role(str(value or ""), str(role or ""))


def role_candidate_is_valid(value: Any, role: str) -> bool:
    """Return whether one candidate is admissible for a PBOC semantic role."""
    normalized = normalize_role_candidate(value, role)
    return bool(normalized) and _is_valid_for_role(normalized, str(role or ""))


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
    if "业务类型" in label or "业务类别" in label:
        return "summary_business_category"
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
    if key in _IDENTITY_NUMBER_TYPE_FIELDS:
        from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
            normalize_pboc_field,
            validate_pboc_field,
        )

        document_type = next(
            (
                str(owner.get(type_field) or "")
                for type_field in _IDENTITY_NUMBER_TYPE_FIELDS[key]
                if owner.get(type_field) not in (None, "")
            ),
            "",
        )
        normalized_type = normalize_pboc_field(document_type, "identity_document_type")
        if not validate_pboc_field(normalized_type, "identity_document_type").valid:
            normalized_type = "unknown"
        return f"identity_document_number::{normalized_type}"
    if key == "value":
        return _summary_cell_role(owner)
    if key == "reason" and any(name in owner for name in ("inquiry_date", "inquiry_type")):
        return "inquiry_reason"
    if key == "business_type" and any(
        name in owner
        for name in (
            "account_id",
            "account_identifier",
            "account_number",
            "guarantee_type",
            "repayment_frequency",
            "repayment_method",
        )
    ):
        return "account_business_type"
    if key == "status_code" and any(
        name in owner for name in ("performance_month", "year", "month")
    ):
        return "repayment_status"
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


def _candidate_is_contained_by_field_cell(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> bool:
    """Require candidate-local containment for a missing-field repair.

    The generic page-evidence selector deliberately permits a small OCR halo
    and row-scale boxes for other repair roles.  A previously blank currency
    slot needs stronger proof: the selected token's center must be in the
    canonical value cell and at least half of that token must overlap the cell.
    """

    tx0, ty0, tx1, ty1 = target
    cx0, cy0, cx1, cy1 = candidate
    target_width = tx1 - tx0
    target_height = ty1 - ty0
    candidate_width = cx1 - cx0
    candidate_height = cy1 - cy0
    if min(target_width, target_height, candidate_width, candidate_height) <= 0.0:
        return False
    center_x = (cx0 + cx1) / 2.0
    center_y = (cy0 + cy1) / 2.0
    if not (tx0 <= center_x <= tx1 and ty0 <= center_y <= ty1):
        return False
    intersection_width = max(0.0, min(tx1, cx1) - max(tx0, cx0))
    intersection_height = max(0.0, min(ty1, cy1) - max(ty0, cy0))
    candidate_area = candidate_width * candidate_height
    return intersection_width * intersection_height / candidate_area >= 0.5


class PersonalDetailOCRCorrectionOverlay:
    """Schema-aware corrected view over one sealed personal report.

    This object never starts OCR.  The business-repair coordinator may install
    complete-page evidence after first-pass validation; typed correction then
    selects candidates from that fixed evidence plane.
    """

    def __init__(
        self,
        parse_result: Any,
    ) -> None:
        self.parse_result = parse_result
        self._repair_evidence_by_page: dict[int, dict[str, Any]] = {}
        self._repair_evidence_reparse_attempts = 0
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
            "repair_evidence_page_count": len(self._repair_evidence_by_page),
            "repair_evidence_reparse_attempt_count": self._repair_evidence_reparse_attempts,
            "ocr_started_by_correction_overlay": False,
            "audited_cell_count": len(self._audited_cells),
            "abnormal_cell_count": len(self._cell_anomalies),
            "cell_anomalies": [anomaly.to_dict() for anomaly in self._cell_anomalies],
            "decisions": [decision.to_dict() for decision in self._decisions],
        }

    def install_business_repair_evidence(
        self,
        pages: Iterable[Mapping[str, Any]],
        *,
        affected_pages: Iterable[int],
    ) -> None:
        """Install the coordinator's fixed evidence without acquiring any OCR."""
        allowed = {int(page) for page in affected_pages if int(page) > 0}
        self._repair_evidence_by_page = {
            int(page.get("page") or 0): deepcopy(dict(page))
            for page in pages
            if isinstance(page, Mapping)
            and int(page.get("page") or 0) in allowed
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
        if not _is_valid_for_role(corrected, role):
            repaired = self._repair_from_installed_page_evidence(original, role=role, refs=refs)
            if repaired is not None:
                return repaired
        return original, None

    def _repair_from_installed_page_evidence(
        self,
        original: str,
        *,
        role: str,
        refs: tuple[dict[str, Any], ...],
    ) -> tuple[str, PersonalDetailCorrectionDecision] | None:
        ref = next(
            (
                item
                for item in refs
                if isinstance(item.get("bbox"), (list, tuple))
                and len(item["bbox"]) == 4
                and int(item.get("logical_page") or item.get("page") or 0) > 0
                and (
                    item.get("geometry_scope") == "cell"
                    or item.get("binding") == "canonical_field_slot"
                )
            ),
            None,
        )
        if ref is None:
            return None
        logical_page = int(ref.get("logical_page") or ref.get("page") or 0)
        evidence = self._repair_evidence_by_page.get(logical_page)
        if evidence is None:
            return None
        self._repair_evidence_reparse_attempts += 1
        return self._repair_from_page_evidence(
            original,
            role=role,
            ref=ref,
            refs=refs,
            page=evidence,
        )

    def _repair_from_page_evidence(
        self,
        original: str,
        *,
        role: str,
        ref: dict[str, Any],
        refs: tuple[dict[str, Any], ...],
        page: Mapping[str, Any],
    ) -> tuple[str, PersonalDetailCorrectionDecision] | None:
        """Select a typed value from already-acquired complete-page evidence."""
        selection = self._select_page_evidence_candidate(
            role=role,
            ref=ref,
            page=page,
        )
        if selection is None:
            return None
        selected = str(selection["normalized"])
        return selected, self._record(
            role=role,
            original=original,
            corrected=selected,
            method="schema_bound_page_evidence_reparse",
            reason_codes=("business_uncertainty_trigger", "schema_role_validation", "candidate_margin"),
            confidence=float(selection["confidence"]),
            refs=refs,
            candidates=tuple(selection["candidates"]),
        )

    @staticmethod
    def _select_page_evidence_candidate(
        *,
        role: str,
        ref: Mapping[str, Any],
        page: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return one margin-qualified typed candidate and its exact OCR evidence."""

        target = tuple(float(item) for item in ref.get("bbox") or ())
        if len(target) != 4 or not all(isfinite(item) for item in target):
            return None
        candidates: list[dict[str, Any]] = []
        associated: list[dict[str, Any]] = []
        for line in page.get("lines") or ():
            if not isinstance(line, dict):
                continue
            bbox = line.get("bbox")
            text = str(line.get("text") or line.get("content") or "").strip()
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            candidate_box = tuple(float(item) for item in bbox)
            if not all(isfinite(item) for item in candidate_box):
                continue
            if not _boxes_associate(target, candidate_box):
                continue
            confidence = float(line.get("confidence") or 0.0)
            evidence_ids = tuple(
                str(item)
                for item in line.get("evidence_ids") or ()
                if str(item or "")
            )
            associated.append(
                {
                    "top": candidate_box[1],
                    "left": candidate_box[0],
                    "raw": text,
                    "confidence": confidence,
                    "bbox": candidate_box,
                    "evidence_ids": evidence_ids,
                }
            )
            normalized = _normalize_role(text, role)
            if _is_valid_for_role(normalized, role):
                candidates.append(
                    {
                        "confidence": confidence,
                        "normalized": normalized,
                        "raw": text,
                        "bbox": candidate_box,
                        "evidence_ids": evidence_ids,
                    }
                )
        if associated:
            ordered = sorted(
                associated,
                key=lambda item: (float(item["top"]), float(item["left"])),
            )
            joined = " ".join(str(item["raw"]) for item in ordered)
            normalized = _normalize_role(joined, role)
            if _is_valid_for_role(normalized, role):
                boxes = [tuple(item["bbox"]) for item in ordered]
                candidates.append(
                    {
                        "confidence": min(float(item["confidence"]) for item in ordered),
                        "normalized": normalized,
                        "raw": joined,
                        "bbox": (
                            min(box[0] for box in boxes),
                            min(box[1] for box in boxes),
                            max(box[2] for box in boxes),
                            max(box[3] for box in boxes),
                        ),
                        "evidence_ids": tuple(
                            dict.fromkeys(
                                evidence_id
                                for item in ordered
                                for evidence_id in item["evidence_ids"]
                            )
                        ),
                    }
                )
        best_by_value: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            normalized = str(candidate["normalized"])
            prior = best_by_value.get(normalized)
            if prior is None or (
                float(candidate["confidence"]),
                len(str(candidate["raw"])),
            ) > (
                float(prior["confidence"]),
                len(str(prior["raw"])),
            ):
                best_by_value[normalized] = candidate
        valid = sorted(
            best_by_value.values(),
            key=lambda item: (float(item["confidence"]), str(item["normalized"])),
            reverse=True,
        )
        if not valid or float(valid[0]["confidence"]) < 0.72:
            return None
        if (
            len(valid) > 1
            and float(valid[0]["confidence"]) - float(valid[1]["confidence"]) < 0.08
        ):
            return None
        selected = dict(valid[0])
        selected["candidates"] = tuple(str(item["normalized"]) for item in valid)
        return selected

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

    @staticmethod
    def _missing_account_currency_ref(
        record: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return one proved blank account-currency value-slot reference."""

        aliases = {"currency", "account_currency"}
        owners = (values, record) if values is not record else (record,)
        unresolved = {
            str(field_name)
            for owner in owners
            for field_name in owner.get("_unresolved_fields") or ()
        }
        invalid = {
            str(field_name)
            for owner in owners
            for field_name in owner.get("_invalid_observation_fields") or ()
        }
        conflicts = {
            str(field_name)
            for owner in owners
            for field_name in owner.get("_reported_field_conflicts") or ()
        }
        source_absent = {
            str(field_name)
            for owner in owners
            for field_name in owner.get("_source_absent_fields") or ()
        }
        if (
            not aliases.intersection(unresolved)
            or not aliases.intersection(invalid)
            or aliases.intersection(conflicts)
            or aliases.intersection(source_absent)
        ):
            return None

        raw_observed = False
        raw_items: list[Any] = []
        for owner in owners:
            canonical_raw = owner.get("canonical_raw")
            if not isinstance(canonical_raw, Mapping):
                continue
            for alias in aliases:
                if alias not in canonical_raw:
                    continue
                raw_observed = True
                raw = canonical_raw.get(alias)
                raw_items.extend(raw if isinstance(raw, list) else (raw,))
        if not raw_observed or any(str(item or "").strip() for item in raw_items):
            # This recovery is only for a blank/unreadable exact slot.  Invalid
            # prose and unsupported printed tokens remain explicit failures.
            return None

        refs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for owner in owners:
            refs_by_field = owner.get("source_refs_by_field")
            if not isinstance(refs_by_field, Mapping):
                continue
            for alias in aliases:
                for raw_ref in refs_by_field.get(alias) or ():
                    if not isinstance(raw_ref, Mapping):
                        continue
                    ref = dict(raw_ref)
                    marker = repr(sorted(ref.items(), key=lambda item: str(item[0])))
                    if marker in seen:
                        continue
                    seen.add(marker)
                    bbox = ref.get("bbox")
                    try:
                        exact_bbox = (
                            tuple(float(item) for item in bbox)
                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
                            else ()
                        )
                    except (TypeError, ValueError):
                        exact_bbox = ()
                    row = ref.get("row")
                    label_row = ref.get("canonical_label_row")
                    value_row = ref.get("canonical_value_row")
                    if (
                        str(ref.get("source") or "") != "native_detail_table_cell"
                        or str(ref.get("binding") or "") != "canonical_field_slot"
                        or str(ref.get("binding_quality") or "")
                        != "canonical_header_column"
                        or str(ref.get("field_slot_role") or "") != "value"
                        or not exact_bbox
                        or not all(isfinite(item) for item in exact_bbox)
                        or int(ref.get("logical_page") or 0) <= 0
                        or not isinstance(row, int)
                        or not isinstance(label_row, int)
                        or not isinstance(value_row, int)
                        or row != value_row
                        or label_row >= value_row
                    ):
                        continue
                    refs.append(ref)
        return refs[0] if len(refs) == 1 else None

    def _recover_missing_account_currencies(
        self,
        payload: dict[str, Any],
        *,
        stage: str,
    ) -> None:
        """Recover only blank exact account-currency slots from installed evidence."""

        if stage != "candidate_b_final_validation":
            return
        accounts = payload.get("credit_accounts")
        if not isinstance(accounts, list):
            return
        for record in accounts:
            if not isinstance(record, dict):
                continue
            normalized = record.get("normalized")
            values = normalized if isinstance(normalized, dict) else record
            if any(
                values.get(field_name) not in (None, "")
                for field_name in (
                    "currency",
                    "account_currency",
                    "reporting_amount_currency",
                )
            ):
                continue
            ref = self._missing_account_currency_ref(record, values)
            if ref is None:
                continue
            logical_page = int(ref.get("logical_page") or 0)
            page = self._repair_evidence_by_page.get(logical_page)
            if page is None:
                continue
            self._repair_evidence_reparse_attempts += 1
            selection = self._select_page_evidence_candidate(
                role="currency",
                ref=ref,
                page=page,
            )
            if (
                selection is None
                or not selection.get("evidence_ids")
            ):
                continue
            try:
                target_bbox = tuple(float(item) for item in ref.get("bbox") or ())
                candidate_bbox = tuple(float(item) for item in selection.get("bbox") or ())
            except (TypeError, ValueError):
                continue
            if (
                len(target_bbox) != 4
                or len(candidate_bbox) != 4
                or not all(isfinite(item) for item in (*target_bbox, *candidate_bbox))
                or not _candidate_is_contained_by_field_cell(
                    target_bbox,
                    candidate_bbox,
                )
            ):
                continue
            from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
                _currency_token,
            )

            currency_codes: set[str] = set()
            candidates_are_exact = True
            for candidate in selection.get("candidates") or ():
                currency_code, residue, resolution = _currency_token(candidate)
                if currency_code is None or residue or resolution != "exact":
                    candidates_are_exact = False
                    break
                currency_codes.add(currency_code)
            if not candidates_are_exact or len(currency_codes) != 1:
                continue
            currency = next(iter(currency_codes))
            raw_currency = str(selection["raw"])
            corrected_ref: dict[str, Any] = {
                "source": "personal_detail_corrected_page_cell",
                "logical_page": logical_page,
                "source_page": int(
                    page.get("source_page")
                    or ref.get("source_page")
                    or logical_page
                ),
                "bbox": list(selection["bbox"]),
                "geometry_scope": "cell",
                "evidence_ids": list(selection["evidence_ids"]),
                "binding": "canonical_field_slot",
                "binding_quality": "canonical_field_slot",
                "field_slot_role": "value",
                "evidence_plane": "business_repair",
            }
            coordinate_system = ref.get("coordinate_system")
            if isinstance(coordinate_system, str) and coordinate_system.strip():
                corrected_ref["coordinate_system"] = coordinate_system.strip()

            provenance = record if values is not record else values
            refs_by_field = provenance.setdefault("source_refs_by_field", {})
            canonical_raw = provenance.setdefault("canonical_raw", {})
            for field_name in (
                "currency",
                "account_currency",
                "reporting_amount_currency",
            ):
                values[field_name] = currency
                if values is not record:
                    # Candidate wrappers retain flat compatibility slots beside
                    # ``normalized``.  Keep them synchronized so a stale null
                    # or pre-repair value cannot override the canonical view in
                    # a downstream Community projection.
                    record[field_name] = currency
                canonical_raw[field_name] = raw_currency
                field_ref = {**corrected_ref, "field_name": field_name}
                field_refs = refs_by_field.setdefault(field_name, [])
                if field_ref not in field_refs:
                    field_refs.append(field_ref)
            self._record(
                role="currency",
                original="",
                corrected=currency,
                method="schema_bound_missing_account_currency_reparse",
                reason_codes=(
                    "exact_blank_account_currency_value_slot",
                    "coordinator_installed_page_evidence",
                    "unique_finite_currency_candidate",
                    "candidate_margin",
                ),
                confidence=float(selection["confidence"]),
                refs=(ref, {**corrected_ref, "field_name": "currency"}),
                candidates=(currency,),
            )

    def correct_business_candidates(self, payload: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
        corrected = deepcopy(dict(payload))
        # Institution names are individualized values.  Repetition elsewhere
        # in the report is not evidence for this cell; only the same canonical
        # field slot on the one-shot page observation may correct it.
        self._recover_missing_account_currencies(corrected, stage=stage)
        self._walk(corrected, parent="", refs=(), stage=stage)
        self._enforce_cross_field_contracts(corrected, stage=stage)
        self._promote_account_identifier_candidates(corrected)
        return corrected

    def enforce_cross_field_contracts(self, payload: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
        """Validate merged records without replaying all single-field corrections."""
        corrected = deepcopy(dict(payload))
        self._enforce_cross_field_contracts(corrected, stage=stage)
        return corrected

    def _enforce_cross_field_contracts(self, payload: dict[str, Any], *, stage: str) -> None:
        """Withhold individually valid values that violate the dataset schema."""
        for index, record in enumerate(payload.get("credit_accounts") or [], start=1):
            if not isinstance(record, dict):
                continue
            values = record.get("normalized") if isinstance(record.get("normalized"), dict) else record
            if values.get("account_identifier") not in (None, ""):
                continue
            refs_by_field = record.get("source_refs_by_field")
            field_refs = (
                _source_refs(refs_by_field.get("account_identifier"))
                if isinstance(refs_by_field, Mapping)
                else ()
            )
            cell_refs = _source_refs(record.get("source_cell_refs"))
            field_refs = field_refs or tuple(
                ref
                for ref in cell_refs
                if not ref.get("field_name") or ref.get("field_name") == "account_identifier"
            ) or _source_refs(record.get("source_refs"))
            record_id = str(
                values.get("account_id")
                or record.get("record_id")
                or f"credit_accounts:{index}"
            )
            self._audit_cell(
                stage=stage,
                path=f"credit_accounts[{record_id}].account_identifier",
                role="account_identifier",
                dataset_name="credit_accounts",
                record_id=record_id,
                field_name="account_identifier",
                value=None,
                refs=field_refs,
                valid=False,
                reason_codes=(
                    "required_field_missing",
                    "canonical_account_identifier_unresolved",
                    "preserved_unknown_value",
                ),
            )
        for index, record in enumerate(payload.get("repayment_records") or [], start=1):
            if not isinstance(record, dict):
                continue
            values = record.get("normalized") if isinstance(record.get("normalized"), dict) else record
            pairing = record.get("_amount_pairing") or values.get("_amount_pairing")
            if not isinstance(pairing, Mapping) or values.get("overdue_amount") not in (None, ""):
                continue
            status = str(values.get("status_code") or values.get("status") or "").strip()
            if not status or status in {"1", "2", "3", "4", "5", "6", "7"}:
                continue
            refs = _source_refs(record.get("source_cell_refs")) or _source_refs(
                record.get("source_refs")
            )
            amount_refs = tuple(
                ref for ref in refs if ref.get("field_name") == "overdue_amount"
            ) or refs
            record_id = str(
                values.get("repayment_id")
                or record.get("repayment_id")
                or record.get("record_id")
                or f"repayment_records:{index}"
            )
            pair_status = str(pairing.get("status") or "amount_pair_geometry_unresolved")
            self._audit_cell(
                stage=stage,
                path=f"repayment_records[{record_id}].overdue_amount",
                role="amount",
                dataset_name="repayment_records",
                record_id=record_id,
                field_name="overdue_amount",
                value=None,
                refs=amount_refs,
                valid=False,
                reason_codes=(
                    "monthly_status_amount_unresolved",
                    "candidate_b_immediate_amount_pair_required",
                    pair_status,
                    "blank_amount_not_inferred_as_zero",
                    "preserved_unknown_value",
                ),
            )

        for index, record in enumerate(payload.get("repayment_records") or [], start=1):
            if not isinstance(record, dict):
                continue
            values = record.get("normalized") if isinstance(record.get("normalized"), dict) else record
            pairing = record.get("_amount_pairing") or values.get("_amount_pairing")
            pair_status = (
                str(pairing.get("status") or "") if isinstance(pairing, Mapping) else ""
            )
            # ``status_code`` is the canonical key when it is already present;
            # source monthly rows use ``status``.  Respect that shape instead
            # of creating a second, competing status field.
            status_key = next(
                (
                    key
                    for key in ("status_code", "status")
                    if values.get(key) not in (None, "")
                ),
                "status" if "status" in values else "status_code",
            )
            status = str(values.get(status_key) or "").strip()
            if status not in {"1", "2", "3", "4", "5", "6", "7"}:
                continue

            amount = values.get("overdue_amount")
            decimal_amount: Decimal | None = None
            try:
                normalized_amount = _normalize_amount(str(amount or ""))
                decimal_amount = Decimal(normalized_amount)
                amount_is_positive = bool(
                    _is_valid_for_role(normalized_amount, "amount")
                    and decimal_amount.is_finite()
                    and decimal_amount > 0
                )
            except (InvalidOperation, ValueError):
                amount_is_positive = False
            if amount_is_positive:
                continue

            refs = _source_refs(record.get("source_cell_refs")) or _source_refs(record.get("source_refs"))
            refs_by_field = record.get("source_refs_by_field")
            status_refs = (
                _source_refs(refs_by_field.get(status_key))
                if isinstance(refs_by_field, Mapping)
                else ()
            )
            status_refs = status_refs or tuple(
                ref
                for ref in refs
                if ref.get("field_name") in {None, "", status_key, "status", "status_code"}
            ) or refs
            amount_refs = tuple(ref for ref in refs if ref.get("field_name") == "overdue_amount") or refs
            record_id = str(
                values.get("repayment_id")
                or record.get("repayment_id")
                or record.get("record_id")
                or f"repayment_records:{index}"
            )

            # The digit remains candidate evidence, but without a positive
            # paired overdue amount it is not an admissible business status.
            raw_values = record.setdefault("canonical_raw", {})
            if isinstance(raw_values, dict):
                raw_values.setdefault(status_key, status)
            values[status_key] = "unknown"
            self._audit_cell(
                stage=stage,
                path=f"repayment_records[{record_id}].status_code",
                role="repayment_status",
                dataset_name="repayment_records",
                record_id=record_id,
                field_name="status_code",
                value=status,
                refs=status_refs,
                valid=False,
                normalized_value_withheld=True,
                reason_codes=(
                    "monthly_status_amount_unresolved",
                    "numeric_overdue_status_requires_amount_evidence",
                    "positive_validated_overdue_amount_required",
                    "raw_evidence_preserved",
                    "normalized_value_withheld",
                ),
            )

            # A missing amount is independently incomplete.  A printed zero,
            # however, can be perfectly legible evidence that the status digit
            # is wrong, so do not falsely report the amount in that case.  An
            # invalid non-empty amount was already reported and withheld by
            # the ordinary field walk above.
            raw_amount = (
                raw_values.get("overdue_amount")
                if isinstance(raw_values, Mapping)
                else None
            )
            if amount in (None, "") and raw_amount in (None, ""):
                self._audit_cell(
                    stage=stage,
                    path=f"repayment_records[{record_id}].overdue_amount",
                    role="amount",
                    dataset_name="repayment_records",
                    record_id=record_id,
                    field_name="overdue_amount",
                    value=None,
                    refs=amount_refs,
                    valid=False,
                    reason_codes=(
                        "monthly_status_amount_unresolved",
                        "numeric_overdue_status_requires_amount_evidence",
                        *(
                            ("candidate_b_immediate_amount_pair_required", pair_status)
                            if pair_status
                            else ()
                        ),
                        "preserved_unknown_value",
                    ),
                )
            elif (
                decimal_amount is not None
                and decimal_amount.is_finite()
                and decimal_amount < 0
            ):
                if isinstance(raw_values, dict):
                    raw_values.setdefault("overdue_amount", amount)
                values["overdue_amount"] = None
                self._audit_cell(
                    stage=stage,
                    path=f"repayment_records[{record_id}].overdue_amount",
                    role="amount",
                    dataset_name="repayment_records",
                    record_id=record_id,
                    field_name="overdue_amount",
                    value=amount,
                    refs=amount_refs,
                    valid=False,
                    normalized_value_withheld=True,
                    reason_codes=(
                        "monthly_status_amount_unresolved",
                        "negative_overdue_amount_invalid",
                        "raw_evidence_preserved",
                        "normalized_value_withheld",
                    ),
                )
        # PBOC agreement cards can legitimately report 已用额度 above the
        # printed 授信额度 (for example after limit changes or shared-limit
        # accounting).  Column provenance, not an invented inequality, is the
        # authority for these two independent source fields.

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
        row_refs = _source_refs(value.get("source_refs"))
        cell_refs = _source_refs(value.get("source_cell_refs"))
        local_refs = row_refs or cell_refs or refs
        refs_by_field = value.get("source_refs_by_field")
        confidence = value.get("confidence")
        raw_node_id = (
            value.get("record_id")
            or value.get("summary_cell_id")
            or next(
                (
                    item
                    for key, item in value.items()
                    if str(key).endswith("_id")
                    and str(key) != "account_id"
                    and item not in (None, "")
                    and not isinstance(item, (dict, list))
                ),
                "",
            )
            or value.get("account_id")
        )
        node_id = str(raw_node_id or "")
        base_path = f"{parent}[{node_id}]" if node_id else str(parent or "")
        dataset_name = str(parent or "").split(".", 1)[0].split("[", 1)[0]
        for key, item in list(value.items()):
            field_path = f"{base_path}.{key}".lstrip(".")
            role = _mapping_role(value, str(key))
            configured_field_refs = (
                _source_refs(refs_by_field.get(str(key)))
                if isinstance(refs_by_field, Mapping)
                else ()
            )
            tagged_cell_refs = tuple(
                ref
                for ref in cell_refs
                if not ref.get("field_name") or ref.get("field_name") == str(key)
            )
            field_refs = configured_field_refs or tagged_cell_refs or tuple(
                ref
                for ref in local_refs
                if not ref.get("field_name") or ref.get("field_name") == str(key)
            ) or local_refs
            if role and item not in (None, "") and not isinstance(item, (dict, list)):
                if (
                    stage == "candidate_b_final_validation"
                    and role in {"institution_name", "employer_name"}
                    and institution_name_has_separated_leading_han(str(item))
                    and normalize_institution_name(str(item))
                    == re.sub(r"\s+", "", _plain_text(item)).strip("-_:：,，;；")
                ):
                    raw_values = value.setdefault("canonical_raw", {})
                    if isinstance(raw_values, dict):
                        raw_values.setdefault(str(key), item)
                    value[key] = None
                    self._audit_cell(
                        stage=stage,
                        path=field_path,
                        role=role,
                        dataset_name=dataset_name,
                        record_id=node_id,
                        field_name=str(key),
                        value=item,
                        refs=field_refs,
                        valid=False,
                        normalized_value_withheld=True,
                        reason_codes=(
                            "separated_leading_han_boundary",
                            "independent_source_corroboration_missing",
                            "normalized_value_withheld",
                        ),
                    )
                    continue
                updated, decision = self.correct_text(
                    item,
                    role=role,
                    source_refs=field_refs,
                    confidence=float(confidence or 0.0),
                )
                if decision is not None:
                    value[key] = updated
                final_value = value[key]
                valid = _is_valid_for_role(str(final_value), role)
                withhold_invalid = bool(
                    not valid
                    and (
                        stage == "candidate_b_final_validation"
                        or (
                            stage == "native_business"
                            and len(field_refs) == 1
                            and _cell_scoped(field_refs)
                        )
                    )
                )
                if withhold_invalid:
                    raw_values = value.setdefault("canonical_raw", {})
                    if isinstance(raw_values, dict):
                        raw_values.setdefault(str(key), final_value)
                    value[key] = None
                self._audit_cell(
                    stage=stage,
                    path=field_path,
                    role=role,
                    dataset_name=dataset_name,
                    record_id=node_id,
                    field_name=str(key),
                    value=final_value,
                    refs=field_refs,
                    valid=valid,
                    normalized_value_withheld=withhold_invalid,
                )
            elif isinstance(item, dict) and role and item.get("value") not in (None, ""):
                nested_refs = _source_refs(item.get("source_refs")) or local_refs
                if (
                    stage == "candidate_b_final_validation"
                    and role in {"institution_name", "employer_name"}
                    and institution_name_has_separated_leading_han(str(item["value"]))
                    and normalize_institution_name(str(item["value"]))
                    == re.sub(r"\s+", "", _plain_text(item["value"])).strip("-_:：,，;；")
                ):
                    item.setdefault("raw", item["value"])
                    final_value = item["value"]
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
                        valid=False,
                        normalized_value_withheld=True,
                        reason_codes=(
                            "separated_leading_han_boundary",
                            "independent_source_corroboration_missing",
                            "normalized_value_withheld",
                        ),
                    )
                    continue
                updated, decision = self.correct_text(
                    item["value"],
                    role=role,
                    source_refs=nested_refs,
                    confidence=float(item.get("confidence") or confidence or 0.0),
                )
                if decision is not None:
                    item.setdefault("raw", item["value"])
                    item["value"] = updated
                final_value = item["value"]
                valid = _is_valid_for_role(str(final_value), role)
                withhold_invalid = bool(
                    not valid
                    and (
                        stage == "candidate_b_final_validation"
                        or (
                            stage == "native_business"
                            and len(nested_refs) == 1
                            and _cell_scoped(nested_refs)
                        )
                    )
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
            if (
                isinstance(item, (dict, list))
                and str(key) not in _RAW_OR_PROVENANCE_KEYS
                and not str(key).startswith("_")
            ):
                self._walk(item, parent=field_path, refs=local_refs, stage=stage)

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
    "institution_name_has_separated_leading_han",
    "institution_slot_is_unambiguous",
    "normalize_institution_name",
    "normalize_role_candidate",
    "role_candidate_is_valid",
]
