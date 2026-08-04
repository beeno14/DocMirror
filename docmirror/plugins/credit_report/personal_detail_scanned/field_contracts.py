# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned PBOC instructed-cell validation for personal credit reports.

Contracts are intentionally broad across known PBOC report revisions.  A
contract failure is diagnostic only: callers must retain the OCR value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTRACT_ID = "pboc.personal_credit_report.instructed_cells"
CONTRACT_VERSION = "2026-08-04"

_PLACEHOLDERS = frozenset({"-", "--", "---", "未报告", "不详", "未知"})

_VOCABULARIES: dict[str, frozenset[str]] = {
    "gender": frozenset({"男", "女", "未说明的性别", "未知"}),
    "marital_status": frozenset(
        {"未婚", "已婚", "初婚", "再婚", "复婚", "离婚", "丧偶", "分居", "未知", "未说明"}
    ),
    "employment_status": frozenset(
        {
            "在职", "离职", "退休", "失业", "无业", "自由职业", "学生", "务农", "个体经营",
            "其他", "未知", "未说明",
        }
    ),
    "education_level": frozenset(
        {
            "研究生", "大学本科", "大学专科和专科学校", "中等专业学校或中等技术学校", "技术学校",
            "高中", "初中", "小学", "文盲或半文盲", "其他", "未知", "本科", "专科", "硕士研究生",
            "博士研究生",
        }
    ),
    "degree": frozenset({"名誉博士", "博士", "硕士", "学士", "其他", "无", "未知", "未说明"}),
    "currency": frozenset(
        {"人民币", "美元", "欧元", "日元", "港元", "英镑", "CNY", "RMB", "USD", "EUR", "JPY", "HKD", "GBP"}
    ),
    "responsibility_type": frozenset(
        {"本人", "保证人", "担保人", "共同借款人", "共同还款人", "抵押人", "质押人", "其他", "未知"}
    ),
    "inquiry_reason": frozenset(
        {
            "本人查询", "贷后管理", "贷款审批", "信用卡审批", "担保资格审查", "融资审批", "保前审查",
            "客户准入资格审查", "资信审查", "法人代表、负责人、高管等资信审查", "异议处理", "司法调查",
            "公积金提取复核", "特约商户实名审查", "其他",
        }
    ),
    "residence_status": frozenset(
        {"自置", "按揭", "亲属楼宇", "集体宿舍", "租房", "共有住宅", "其他", "未知"}
    ),
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    "postal_code": re.compile(r"^\d{6}$"),
    "organization_code": re.compile(r"^[0-9A-Z-]{8,32}$"),
    "country_or_region_code": re.compile(r"^[A-Z]{2,3}$|^[\u3400-\u9fff]{2,20}$"),
}


@dataclass(frozen=True)
class FieldContractResult:
    assessed: bool
    valid: bool
    contract_id: str = CONTRACT_ID
    contract_version: str = CONTRACT_VERSION
    reason_code: str = ""


def validate_pboc_field(value: str, role: str) -> FieldContractResult:
    """Validate one instructed cell without correcting or discarding it."""
    text = str(value or "").strip()
    if role not in _VOCABULARIES and role not in _PATTERNS:
        return FieldContractResult(assessed=False, valid=True)
    if text in _PLACEHOLDERS:
        return FieldContractResult(assessed=True, valid=True, reason_code="explicit_placeholder")
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


__all__ = ["CONTRACT_ID", "CONTRACT_VERSION", "FieldContractResult", "validate_pboc_field"]
