# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned PBOC instructed-cell validation for personal credit reports.

Contracts are intentionally broad across known PBOC report revisions.  A
contract failure is diagnostic only: callers must retain the OCR value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

CONTRACT_ID = "pboc.personal_credit_report.instructed_cells"
CONTRACT_VERSION = "2026-08-07"

_PLACEHOLDERS = frozenset({"-", "--", "---", "未报告", "不详", "未知"})

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
        {"人民币", "人民币元", "美元", "欧元", "日元", "港元", "英镑", "CNY", "RMB", "USD", "EUR", "JPY", "HKD", "GBP"}
    ),
    "responsibility_type": frozenset(
        {"本人", "保证人", "担保人", "共同借款人", "共同还款人", "抵押人", "质押人", "其他", "未知"}
    ),
    "inquiry_reason": frozenset(
        {
            "本人查询", "本人查询(自助查询机)", "本人查询（自助查询机）", "贷后管理", "保后管理", "贷款审批", "信用卡审批", "担保资格审查", "融资审批", "保前审查",
            "本人查询(商业银行网上银行)", "本人查询（商业银行网上银行）",
            "本人查询(互联网个人信用信息服务平台)", "本人查询（互联网个人信用信息服务平台）",
            "本人查询(征信中心柜台)", "本人查询（征信中心柜台）",
            "客户准入资格审查", "资信审查", "法人代表、负责人、高管等资信审查", "异议处理", "司法调查",
            "公积金提取复核", "特约商户实名审查", "其他",
        }
    ),
    "residence_status": frozenset(
        {"自置", "按揭", "亲属楼宇", "集体宿舍", "租房", "共有住宅", "其他", "未知"}
    ),
    "facility_type": frozenset(
        {
            "非循环贷款额度",
            "循环贷款额度",
            "信用卡共享额度",
            "信用卡独立额度",
            "其他额度",
        }
    ),
    "account_business_type": frozenset(
        {
            "个人住房贷款",
            "个人商用房贷款",
            "个人经营性贷款",
            "个人消费贷款",
            "其他个人消费贷款",
            "其他贷款",
            "循环贷款",
            "贷记卡",
            "准贷记卡",
        }
    ),
    "identity_document_type": frozenset(
        {
            "身份证",
            "居民身份证",
            "军官证",
            "士兵证",
            "护照",
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


def normalize_pboc_field(value: str, role: str) -> str:
    """Return a canonical vocabulary spelling when only layout whitespace differs."""
    text = str(value or "").strip()
    candidates = _VOCABULARIES.get(role)
    if candidates is None:
        return text
    if role == "facility_type":
        # Short sequence/watermark fragments frequently trail the intended
        # value after a damaged table boundary.  Strip only a separated,
        # non-lexical suffix; never delete characters inside the value.
        text = re.sub(r"\s+[%*?#]?\d{1,2}\s*$", "", text).strip()
    marker = _controlled_marker(text)
    matches = sorted(candidate for candidate in candidates if _controlled_marker(candidate) == marker)
    if matches:
        return matches[0]
    if role in {"facility_type", "account_business_type"}:
        scored = sorted(
            ((SequenceMatcher(None, marker, _controlled_marker(candidate)).ratio(), candidate) for candidate in candidates),
            reverse=True,
        )
        if scored:
            best_score, best = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if best_score >= 0.84 and best_score - runner >= 0.08 and abs(len(marker) - len(_controlled_marker(best))) <= 2:
                return best
    return text


def _custom_contract(text: str, role: str) -> FieldContractResult | None:
    compact = re.sub(r"\s+", "", text)
    if role == "person_name":
        valid = bool(re.fullmatch(r"[\u3400-\u9fff·]{2,30}", compact))
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
    elif role == "repayment_method":
        date_text = bool(re.search(r"截至\s*(?:19|20)\d{2}|(?:19|20)\d{2}[./年-]\d{1,2}", compact))
        duration_only = bool(re.fullmatch(r"\d{1,4}(?:月|期|次|年)", compact))
        valid = bool(re.search(r"[\u3400-\u9fff]", compact)) and not date_text and not duration_only
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
    if text in _PLACEHOLDERS:
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
    "normalize_pboc_field",
    "validate_pboc_field",
]
