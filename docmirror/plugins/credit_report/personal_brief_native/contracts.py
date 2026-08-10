# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed business contracts for digital PBOC personal-brief reports.

This module is intentionally personal-brief-only.  It is the single runtime
source for public enum membership and reporting-amount unit invariants.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

PERSONAL_BRIEF_REPORTING_CURRENCY = "CNY"
PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT = "CNY_1"
PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION = 0

PERSONAL_BRIEF_AMOUNT_UNIT_LABELS: dict[str, str] = {
    PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT: "元（人民币）",
}

_REPORTING_STATUS_LABELS = {
    "reported": "已报告",
    "not_reported": "未报告",
    "not_applicable": "不适用",
    "derived": "由账户记录派生",
}
_AMOUNT_STATUS_LABELS = {
    "reported": "已报告",
    "not_reported": "未报告",
    "not_applicable": "不适用",
}
_ACCOUNT_TYPE_LABELS = {
    "credit_card": "信用卡",
    "loan": "贷款",
    "credit_line": "贷款授信",
    "other_business": "其他业务",
}
_CAUSE_STATUS_LABELS = {
    "reported": "已报告",
    "not_reported": "未报告",
}


# Keys are dataset-qualified because a field name may have different business
# domains in different source sections (notably ``business_category``).
PERSONAL_BRIEF_ENUM_CONTRACT: dict[tuple[str, str], dict[str, str]] = {
    ("personal_report_metadata", "marital_status"): {
        "unmarried": "未婚",
        "married": "已婚",
        "divorced": "离婚",
        "widowed": "丧偶",
        "other": "其他",
        "not_reported": "未说明",
    },
    ("personal_credit_summary_records", "metric"): {
        "account_count": "账户数",
        "unclosed_account_count": "未结清/未销户账户数",
        "ever_overdue_account_count": "发生过逾期的账户数",
        "over_90_days_account_count": "发生过90天以上逾期的账户数",
        "asset_disposition_count": "资产处置信息账户数",
        "guarantor_compensation_count": "垫款信息账户数",
        "personal_repayment_liability_count": "为个人承担相关还款责任的账户数",
        "enterprise_repayment_liability_count": "为企业承担相关还款责任的账户数",
    },
    ("personal_credit_summary_records", "business_category"): {
        "credit_card": "信用卡",
        "housing_loan": "购房贷款",
        "other_loan": "其他贷款",
        "other_business": "其他业务",
        "all": "全部",
    },
    ("personal_credit_summary_records", "reporting_status"): dict(
        _REPORTING_STATUS_LABELS
    ),
    ("guarantor_compensation_records", "settlement_state"): {
        "settled": "已结清",
        "not_reported": "未报告",
    },
    ("credit_accounts", "account_type"): dict(_ACCOUNT_TYPE_LABELS),
    ("credit_accounts", "business_category"): {
        "credit_cards": "信用卡",
        "loans": "贷款",
        "other_business": "其他业务",
    },
    ("credit_accounts", "credit_card_type"): {
        "credit_card": "贷记卡",
        "quasi_credit_card": "准贷记卡",
    },
    ("credit_accounts", "credit_line_validity_type"): {
        "fixed_term": "固定期限",
        "perpetual": "长期有效",
        "not_reported": "未报告",
    },
    ("credit_accounts", "termination_event_type"): {
        "debt_settled": "债务结清",
        "account_closed": "信用卡销户",
        "transferred_out": "账户转出",
    },
    ("credit_accounts", "credit_limit_status"): dict(_AMOUNT_STATUS_LABELS),
    ("credit_accounts", "used_amount_status"): dict(_AMOUNT_STATUS_LABELS),
    ("credit_accounts", "loan_amount_status"): dict(_AMOUNT_STATUS_LABELS),
    ("credit_accounts", "balance_status"): dict(_AMOUNT_STATUS_LABELS),
    ("credit_accounts", "account_lifecycle_state"): {
        "open": "未关闭",
        "settled": "已结清",
        "closed": "已销户",
        "transferred_out": "已转出",
    },
    ("credit_accounts", "card_activation_state"): {
        "activated": "已激活",
        "not_activated": "尚未激活",
        "not_reported": "未报告",
        "not_applicable": "不适用",
    },
    ("credit_accounts", "payoff_state"): {
        "outstanding": "未结清",
        "settled": "已结清",
        "not_applicable": "不适用",
        "unknown": "未知（如账户已转出）",
    },
    ("credit_accounts", "credit_quality_status"): {
        "bad_debt": "呆账",
        "not_reported": "未报告",
    },
    ("overdue_records", "account_type"): dict(_ACCOUNT_TYPE_LABELS),
    ("overdue_records", "period_scope"): {
        "last_5_years": "最近5年",
    },
    ("overdue_records", "current_overdue_status"): {
        "overdue": "当前有逾期",
        "not_overdue": "当前无逾期",
        "not_reported": "未报告",
    },
    ("postpaid_records", "payment_status"): {
        "正常": "正常",
        "欠费": "欠费",
    },
    ("civil_judgment_records", "cause_status"): dict(_CAUSE_STATUS_LABELS),
    ("enforcement_records", "cause_status"): dict(_CAUSE_STATUS_LABELS),
    ("administrative_penalty_records", "administrative_review_result_status"): dict(
        _CAUSE_STATUS_LABELS
    ),
    ("inquiry_records", "inquiry_type"): {
        "institution": "机构查询",
        "personal": "个人查询",
    },
}


PERSONAL_BRIEF_MONEY_FIELDS: dict[str, tuple[str, ...]] = {
    "asset_disposition_records": ("received_debt_amount", "balance"),
    "guarantor_compensation_records": ("cumulative_compensation_amount",),
    "credit_accounts": (
        "credit_limit",
        "used_amount",
        "loan_amount",
        "balance",
        "unbilled_installment_balance",
    ),
    "repayment_liability_records": ("responsibility_amount", "balance"),
    "postpaid_records": ("current_arrears_amount",),
    "tax_arrears_records": ("arrears_amount",),
    "civil_judgment_records": ("claim_amount",),
    "enforcement_records": ("requested_amount", "executed_amount"),
    "administrative_penalty_records": ("penalty_amount",),
}


class PersonalBriefContractError(ValueError):
    """Raised when a personal-brief business row violates its closed contract."""


def canonical_personal_brief_amount_unit(value: Any) -> str:
    """Map legacy internal spelling to the one canonical public unit code."""

    unit = str(value or "")
    if unit in {"yuan", PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT}:
        return PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT
    raise PersonalBriefContractError(
        "PERSONAL_BRIEF_AMOUNT_UNIT_CONTRACT_VIOLATION: "
        f"actual={value!r}, allowed={[PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT]!r}"
    )


def canonicalize_personal_brief_reporting_units(
    datasets: Mapping[str, Sequence[MutableMapping[str, Any]]],
    *,
    amount_policy: MutableMapping[str, Any] | None = None,
) -> None:
    """Canonicalize personal-brief amount-unit aliases before semantic output."""

    updates: list[tuple[MutableMapping[str, Any], str, str]] = []
    if amount_policy is not None:
        updates.append(
            (
                amount_policy,
                "reporting_amount_unit",
                canonical_personal_brief_amount_unit(
                    amount_policy.get("reporting_amount_unit")
                ),
            )
        )
    for rows in datasets.values():
        for row in rows:
            for field in ("reporting_amount_unit", "amount_unit"):
                if field in row and row[field] not in (None, ""):
                    updates.append(
                        (row, field, canonical_personal_brief_amount_unit(row[field]))
                    )
    for target, field, value in updates:
        target[field] = value


def validate_personal_brief_public_record(
    dataset_name: str,
    record_id: str,
    values: Mapping[str, Any],
) -> None:
    """Validate closed enums and reporting-unit invariants for one public row."""

    for (rule_dataset, field), labels in PERSONAL_BRIEF_ENUM_CONTRACT.items():
        if rule_dataset != dataset_name or field not in values:
            continue
        value = values[field]
        if value in (None, ""):
            continue
        if not isinstance(value, str) or value not in labels:
            raise PersonalBriefContractError(
                "PERSONAL_BRIEF_ENUM_CONTRACT_VIOLATION: "
                f"dataset={dataset_name!r}, record_id={record_id!r}, field={field!r}, "
                f"actual={value!r}, allowed={list(labels)!r}"
            )

    if (
        dataset_name == "credit_accounts"
        and values.get("payoff_state") == "unknown"
        and values.get("account_lifecycle_state") != "transferred_out"
    ):
        raise PersonalBriefContractError(
            "PERSONAL_BRIEF_ENUM_RELATION_CONTRACT_VIOLATION: "
            f"dataset={dataset_name!r}, record_id={record_id!r}, "
            f"payoff_state={values.get('payoff_state')!r}, "
            f"account_lifecycle_state={values.get('account_lifecycle_state')!r}"
        )

    if dataset_name == "personal_report_metadata":
        currency = values.get("reporting_currency")
        unit = values.get("reporting_amount_unit")
        precision = values.get("reporting_amount_precision")
        if (
            currency != PERSONAL_BRIEF_REPORTING_CURRENCY
            or unit != PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT
            or precision != PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION
        ):
            raise PersonalBriefContractError(
                "PERSONAL_BRIEF_AMOUNT_POLICY_CONTRACT_VIOLATION: "
                f"dataset={dataset_name!r}, record_id={record_id!r}, "
                f"currency={currency!r}, unit={unit!r}, precision={precision!r}"
            )

    if dataset_name not in PERSONAL_BRIEF_MONEY_FIELDS:
        return
    currency = values.get("reporting_amount_currency")
    unit = values.get("reporting_amount_unit")
    if currency != PERSONAL_BRIEF_REPORTING_CURRENCY or unit != PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT:
        raise PersonalBriefContractError(
            "PERSONAL_BRIEF_MONEY_UNIT_CONTRACT_VIOLATION: "
            f"dataset={dataset_name!r}, record_id={record_id!r}, "
            f"currency={currency!r}, unit={unit!r}, "
            f"money_fields={list(PERSONAL_BRIEF_MONEY_FIELDS[dataset_name])!r}"
        )


__all__ = [
    "PERSONAL_BRIEF_AMOUNT_UNIT_LABELS",
    "PERSONAL_BRIEF_ENUM_CONTRACT",
    "PERSONAL_BRIEF_MONEY_FIELDS",
    "PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION",
    "PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT",
    "PERSONAL_BRIEF_REPORTING_CURRENCY",
    "PersonalBriefContractError",
    "canonical_personal_brief_amount_unit",
    "canonicalize_personal_brief_reporting_units",
    "validate_personal_brief_public_record",
]
