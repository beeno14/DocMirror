# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned PBOC instructed-cell validation for personal credit reports.

Contracts are intentionally broad across known PBOC report revisions.  A
contract failure is diagnostic only: callers must retain the OCR value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docmirror.plugins.credit_report.currency_codes import (
    CURRENCY_CODE_BY_ALIAS,
    ISO_4217_CURRENT_CODES,
)
from docmirror.plugins.credit_report.pboc_vocabularies import (
    PBOC_INQUIRY_REASON_FORMS,
    is_pboc_institution_name,
)

CONTRACT_ID = "pboc.personal_credit_report.instructed_cells"
CONTRACT_VERSION = "2026-08-08"

_PLACEHOLDERS = frozenset({"-", "--", "---", "未报告", "不详", "未知"})
_SOURCE_ABSENCE_DASH_RE = re.compile(r"[-－‐‑‒–—―]+")

_VOCABULARIES: dict[str, frozenset[str]] = {
    "gender": frozenset({"男", "女", "未说明的性别", "未知"}),
    "marital_status": frozenset(
        {"未婚", "已婚", "初婚", "再婚", "复婚", "离婚", "丧偶", "分居", "未知", "未说明"}
    ),
    "employment_status": frozenset(
        {
            "在职", "离职", "退休", "失业", "无业", "自由职业", "学生", "务农", "个体经营",
            "个体经营者", "职员", "专业技术人员", "其他", "未知", "未说明",
        }
    ),
    "education_level": frozenset(
        {
            "研究生", "大学本科", "大学专科和专科学校", "中等专业学校或中等技术学校", "技术学校",
            "高中", "初中", "小学", "文盲或半文盲", "其他", "未知", "本科", "专科", "硕士研究生",
            "博士研究生", "大专", "中专、职高、技校", "无",
        }
    ),
    "degree": frozenset({"名誉博士", "博士", "硕士", "学士", "其他", "无", "未知", "未说明"}),
    "currency": frozenset(
        {*CURRENCY_CODE_BY_ALIAS, *ISO_4217_CURRENT_CODES, "RMB"}
    ),
    "responsibility_type": frozenset(
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
    ),
    "inquiry_reason": PBOC_INQUIRY_REASON_FORMS,
    "residence_status": frozenset(
        {"自置", "按揭", "亲属楼宇", "集体宿舍", "租房", "共有住宅", "其他", "未知"}
    ),
    "facility_type": frozenset(
        {
            "非循环贷款额度",
            "循环贷款额度",
            "循环额度",
            "信用卡共享额度",
            "信用卡独立额度",
            "其他额度",
        }
    ),
    "account_business_type": frozenset(
        {
            "个人住房贷款",
            "个人住房商业贷款",
            "个人住房公积金贷款",
            "个人商用房贷款",
            "个人经营性贷款",
            "个人消费贷款",
            "个人汽车消费贷款",
            "其他个人消费贷款",
            "国家助学贷款",
            "农户贷款",
            "其他贷款",
            "其他类贷款",
            "循环贷款",
            "融资租赁业务",
            "贷记卡",
            "准贷记卡",
        }
    ),
    "guarantee_type": frozenset(
        {
            "信用/无担保",
            "信用/免担保",
            "抵押",
            "质押",
            "保证",
            "组合（含保证）",
            "组合（不含保证）",
            "农户联保",
            "其他",
        }
    ),
    "repayment_frequency": frozenset(
        {
            "日",
            "周",
            "月",
            "季",
            "半年",
            "年",
            "一次性",
            "不定期",
            "其他",
        }
    ),
    "repayment_method": frozenset(
        {
            "按期结息，到期还本",
            "按期结息，自由还本",
            "到期还本分期结息",
            "分期等额本息",
            "分期等额本金",
            "等额本息",
            "等额本金",
            "先息后本",
            "一次性还本付息",
            "按期计算还本付息",
            "随借随还",
            "不区分还款方式",
            "无",
            "其他",
        }
    ),
    "identity_document_type": frozenset(
        {
            "身份证",
            "居民身份证",
            "军官证",
            "士兵证",
            "护照",
            "统一社会信用代码",
            "中征码",
            "组织机构代码",
            "港澳居民来往内地通行证",
            "台湾居民来往大陆通行证",
            "外国人永久居留证",
            "其他证件",
            "未知",
        }
    ),
    "boolean_flag": frozenset({"有", "无", "是", "否", "未知"}),
}

_TYPOGRAPHIC_ALIASES: dict[str, dict[str, str]] = {
    "guarantee_type": {
        "组合(含保证)": "组合（含保证）",
        "组合(不含保证)": "组合（不含保证）",
    },
    "repayment_method": {
        "按期结息,到期还本": "按期结息，到期还本",
        "按期结息,自由还本": "按期结息，自由还本",
    },
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    "postal_code": re.compile(r"^\d{6}$"),
    "organization_code": re.compile(r"^[0-9A-Z-]{8,32}$"),
    "country_or_region_code": re.compile(r"^[A-Z]{2,3}$|^[\u3400-\u9fff]{2,20}(?:[（(][\u3400-\u9fff]+[）)])?$"),
}


@dataclass(frozen=True)
class FieldContractResult:
    assessed: bool
    valid: bool
    contract_id: str = CONTRACT_ID
    contract_version: str = CONTRACT_VERSION
    reason_code: str = ""


def _controlled_marker(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).translate(str.maketrans({"（": "(", "）": ")"}))


def is_explicit_source_absence(value: object) -> bool:
    """Return whether a nonblank cell contains only a printed dash sentinel.

    Blank OCR is deliberately not source absence: an empty canonical slot is
    unreadable evidence and must remain reviewable.  The accepted glyphs are
    the finite dash variants used by PBOC renderers, without treating prose
    such as ``未知`` as a statement that the source omitted the field.
    """

    marker = re.sub(r"\s+", "", str(value or ""))
    return bool(marker and _SOURCE_ABSENCE_DASH_RE.fullmatch(marker))


def normalize_pboc_field(value: str, role: str) -> str:
    """Return a canonical spelling for exact or sanctioned typographic aliases.

    Business enums are deliberately not spell-corrected or fuzzy matched.  An
    OCR-near value may be a different printed category or may include residue
    from a neighbouring cell; that evidence must be reported, not silently
    replaced by the nearest vocabulary member.
    """
    text = str(value or "").strip()
    candidates = _VOCABULARIES.get(role)
    if candidates is None:
        return text
    marker = _controlled_marker(text)
    matches = sorted(candidate for candidate in candidates if _controlled_marker(candidate) == marker)
    if matches:
        return matches[0]
    aliases = _TYPOGRAPHIC_ALIASES.get(role, {})
    alias_matches = sorted(
        canonical for alias, canonical in aliases.items() if _controlled_marker(alias) == marker
    )
    if alias_matches:
        return alias_matches[0]
    return text


def pboc_controlled_vocabulary(role: str) -> frozenset[str]:
    """Return the finite printed vocabulary for one canonical field role."""

    return _VOCABULARIES.get(str(role), frozenset())


def _custom_contract(text: str, role: str) -> FieldContractResult | None:
    compact = re.sub(r"\s+", "", text)
    if role == "person_name":
        valid = bool(re.fullmatch(r"[\u3400-\u9fff·]{2,30}", compact))
    elif role == "institution_name":
        valid = is_pboc_institution_name(compact)
    elif role == "address":
        date_prefix = bool(re.match(r"^(?:19|20)\d{2}[./年-]\d{1,2}(?:[./月-]\d{1,2})?", compact))
        phone_fragment = bool(re.search(r"(?<!\d)\d{7,16}(?!\d)", compact))
        residue = bool(re.search(r"[\"'*]{2,}|[?？]{2,}", text))
        valid = bool(re.search(r"[\u3400-\u9fff]", compact)) and 3 <= len(compact) <= 160 and not (
            date_prefix or phone_fragment or residue
        )
    elif role == "employer_name":
        leading_ordinal = bool(re.match(r"^[\W_]*\d{1,3}\s+", text))
        date_or_phone = bool(
            re.search(r"(?:19|20)\d{2}[./年-]\d{1,2}|(?<!\d)\d{7,16}(?!\d)", compact)
        )
        appended_position = bool(
            re.search(r"(?:公司|单位|银行|中心|学校|医院)(?:董事长|总经理|经理|职员|负责人|主任|主管)$", compact)
        )
        trailing_token = bool(re.search(r"\s+[A-Za-z\u3400-\u9fff*?]{1}\s*$", text))
        valid = (
            bool(re.search(r"[\u3400-\u9fffA-Za-z]", compact))
            and 2 <= len(compact) <= 120
            and not (leading_ordinal or date_or_phone or appended_position or trailing_token)
        )
    elif role == "employment_descriptor":
        valid = bool(re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9·（）()\-]{1,40}", compact)) and bool(
            re.search(r"[\u3400-\u9fffA-Za-z]", compact)
        )
    else:
        return None
    return FieldContractResult(
        assessed=True,
        valid=valid,
        reason_code="field_semantic_shape_match" if valid else "cross_cell_contamination_or_shape_mismatch",
    )


def validate_pboc_field(value: str, role: str) -> FieldContractResult:
    """Validate one instructed cell without correcting or discarding it."""
    text = normalize_pboc_field(value, role)
    if text in _PLACEHOLDERS or is_explicit_source_absence(text):
        return FieldContractResult(assessed=True, valid=True, reason_code="explicit_placeholder")
    custom = _custom_contract(text, role)
    if custom is not None:
        return custom
    if role not in _VOCABULARIES and role not in _PATTERNS:
        return FieldContractResult(assessed=False, valid=True)
    if role in _VOCABULARIES:
        valid = text in _VOCABULARIES[role]
        return FieldContractResult(
            assessed=True,
            valid=valid,
            reason_code="controlled_vocabulary_match" if valid else "controlled_vocabulary_mismatch",
        )
    valid = bool(_PATTERNS[role].fullmatch(text))
    return FieldContractResult(
        assessed=True,
        valid=valid,
        reason_code="field_pattern_match" if valid else "field_pattern_mismatch",
    )


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "FieldContractResult",
    "is_explicit_source_absence",
    "normalize_pboc_field",
    "pboc_controlled_vocabulary",
    "validate_pboc_field",
]
