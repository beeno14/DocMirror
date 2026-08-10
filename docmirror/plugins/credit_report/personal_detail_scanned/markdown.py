# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Business-only Markdown for scanned PBOC personal detailed reports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_DATASET_LABELS = {
    "report_metadata": "报告信息",
    "report_query": "查询信息",
    "subject_profile": "个人基本资料",
    "subject_identity_documents": "身份信息",
    "subject_mobile_phones": "手机号码历史",
    "subject_spouses": "配偶信息",
    "subject_residences": "居住信息",
    "subject_employment": "职业信息",
    "credit_business_overview": "信贷业务概要",
    "credit_accounts": "信贷账户",
    "credit_account_monthly_performance": "月度还款表现",
    "credit_agreements": "授信协议",
    "public_records": "公共信息",
    "inquiries": "查询记录",
}
_DATASET_ORDER = tuple(_DATASET_LABELS)
_DIAGNOSTIC_DATASETS = frozenset(
    {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "dataset_status",
        "personal_detail_source_rows",
    }
)
_SOURCE_BUSINESS_IDENTIFIERS = frozenset(
    {
        "account_identifier",
        "agreement_identifier",
        "certificate_number",
        "contract_number",
        "guarantee_contract_number",
        "id_number",
        "limit_identifier",
        "primary_id_number",
        "report_number",
    }
)
_TECHNICAL_FIELDS = frozenset(
    {
        "business_record_id",
        "canonical_dataset_schema",
        "confidence",
        "confidence_basis",
        "confidence_status",
        "dataset_name",
        "extraction_status",
        "field_name",
        "geometry_scope",
        "logical_page",
        "observation_status",
        "parser_stage",
        "reason",
        "reason_codes",
        "record_id",
        "review_status",
        "sequence",
        "source_dataset_name",
        "source_page",
        "source_page_number",
        "status_reason",
    }
)
_VALUE_LABELS = {
    "true": "是",
    "false": "否",
    "credit_card": "贷记卡",
    "quasi_credit_card": "准贷记卡",
    "non_revolving_loan": "非循环贷款",
    "revolving_loan": "循环贷款",
    "revolving_credit_line": "循环额度",
    "active": "正常",
    "closed": "结清",
    "settled": "结清",
    "unknown": "待核验",
}
_CHINESE_RE = re.compile(r"[\u3400-\u9fff]")


def _escape(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _is_system_field(key: str) -> bool:
    if key in _SOURCE_BUSINESS_IDENTIFIERS:
        return False
    if key in _TECHNICAL_FIELDS:
        return True
    return key.endswith("_id") or key.endswith("_record_id") or key.endswith("_observation_id")


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    text = str(value).strip()
    return _VALUE_LABELS.get(text.lower(), text)


def _column_label(column: Mapping[str, Any]) -> str | None:
    key = str(column.get("key") or "")
    if not key or _is_system_field(key):
        return None
    label = str(column.get("label") or "").strip()
    return label if _CHINESE_RE.search(label) else None


def _dataset_rows(dataset: Mapping[str, Any]) -> tuple[list[tuple[str, str]], list[list[str]]]:
    columns = [column for column in dataset.get("columns") or () if isinstance(column, Mapping)]
    selected = [
        (str(column.get("key") or ""), label)
        for column in columns
        if (label := _column_label(column)) is not None
    ]
    rendered: list[list[str]] = []
    for row in dataset.get("rows") or ():
        if not isinstance(row, Mapping):
            continue
        normalized = row.get("normalized") if isinstance(row.get("normalized"), Mapping) else {}
        canonical_raw = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), Mapping) else {}
        values = [
            _display_value(normalized.get(key) if normalized.get(key) is not None else canonical_raw.get(key))
            for key, _label in selected
        ]
        if any(value for value in values):
            rendered.append(values)
    return selected, rendered


def render_personal_detail_business_markdown(semantic: Mapping[str, Any]) -> str:
    """Render Chinese business data while keeping semantic identifiers private."""

    datasets = {
        str(dataset.get("name") or ""): dataset
        for dataset in semantic.get("datasets") or ()
        if isinstance(dataset, Mapping)
        and str(dataset.get("name") or "") not in _DIAGNOSTIC_DATASETS
    }
    lines = ["# 个人信用报告"]
    for name in _DATASET_ORDER:
        dataset = datasets.get(name)
        if dataset is None:
            continue
        columns, rows = _dataset_rows(dataset)
        if not columns or not rows:
            continue
        lines.extend(("", f"## {_DATASET_LABELS[name]}", ""))
        labels = [_escape(label) for _key, label in columns]
        lines.append("| " + " | ".join(labels) + " |")
        lines.append("| " + " | ".join("---" for _ in labels) + " |")
        lines.extend("| " + " | ".join(_escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_personal_detail_business_markdown"]
