# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, straight-through personal-detail extraction.

This reader uses the same PBOC labels, section names and v2 business dictionary
as the checked pipeline, but not its repair, reconciliation or quality gates.
It reads the *initial* sealed text/cells; it never opens a PDF or requests OCR.
An unparseable scalar remains the observed string, not a rejected/null value.

Label/column binding and calendar coordinates are still necessary extraction
operations. There is deliberately no fuzzy label repair, neighbor-month fill,
source-population assertion, value-based deduplication or confidence threshold.
The normal Community envelope is used; the mode is recorded only in the log.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.credit_report.currency_codes import CURRENCY_CODE_BY_ALIAS, ISO_4217_CURRENT_CODES
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _CREDIT_AGREEMENT_FIELD_NAMES,
    _LIABILITY_LABEL_TO_FIELD,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    _ACCOUNT_TYPE_CODES,
    PBOC_DATASET_ORDER,
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
)
from docmirror.plugins.credit_report.personal_detail_scanned.section_headings import (
    canonical_account_family_heading,
    canonical_registered_section_heading,
    canonical_registered_subsection_heading,
)
from docmirror.plugins.credit_report.report_profile import detect_credit_report_content_mode
from docmirror.plugins.credit_report.value_utils import stable_record_id

logger = logging.getLogger(__name__)

CONTROL_DATASETS = frozenset(
    {"field_observations", "extraction_issues", "extraction_issue_evidence", "pboc_extension_fields", "dataset_status"}
)

# These are parser assessments, not business statuses. In particular do NOT
# remove administrative_review_result, payment_status, case_status, account
# state, repayment status, or every key containing the word "status"/"review".
VALIDATION_FIELDS = frozenset(
    {
        "audit",
        "review",
        "review_required",
        "requires_review",
        "review_reason",
        "confidence",
        "confidence_status",
        "confidence_basis",
        "binding_quality",
        "validation",
        "validation_status",
        "validation_errors",
        "validation_warnings",
        "verified",
        "completeness",
        "quality",
        "quality_flags",
        "diagnostics",
        "warnings",
        "errors",
        "extraction_status",
        "extraction_report",
        "audit_report",
        "observation_status",
        "mapping_status",
        "source_projection_status",
        "dataset_status_semantics",
        "sparse_dataset_status_semantics",
        "absence_dataset",
        "uncertainty_dataset",
        "extraction_issue_dataset",
        "absence_requires_explicit_source_evidence",
        "empty_dataset_means_absent",
        "uncertainty_coverage",
        "dataset_audit_csv",
    }
)


def _assessment_key(key: str) -> bool:
    return key in VALIDATION_FIELDS or key.startswith(("personal_detail_expected_", "personal_detail_v2_expected_"))


def omit_validation(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy output, removing assessments without touching business status fields.

    Both public JSON and semantic JSON pass through here *after* the generic
    serializer, which otherwise inserts completeness/status defaults. No mode
    marker or replacement "unvalidated" status is added to either envelope.
    """

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            # A column descriptor or a section item can name an assessment.
            return {
                str(key): clean(item)
                for key, item in value.items()
                if not _assessment_key(str(key)) and str(key) not in CONTROL_DATASETS
            }
        if isinstance(value, (list, tuple)):
            return [
                clean(item)
                for item in value
                if not (
                    isinstance(item, Mapping)
                    and (_assessment_key(str(item.get("key") or "")) or item.get("name") in CONTROL_DATASETS)
                )
                and not (isinstance(item, str) and item in CONTROL_DATASETS)
            ]
        return deepcopy(value)

    result = clean(payload)
    for dataset in result.get("datasets") or []:
        if isinstance(dataset, dict):
            # Dataset status is an extraction verdict; row status is business.
            dataset.pop("status", None)
    return result


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _label(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value)).strip("：:")


def _box(value: Any) -> tuple[float, ...] | None:
    """Unusable geometry falls back to text; it never rejects business values."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return coordinates if all(isfinite(item) for item in coordinates) else None


# Exact aliases already used by native_extraction/profile_extraction. They
# rename printed labels only; no OCR-value correction is performed here.
_ALIASES: dict[str, dict[str, str]] = {
    "report_metadata": {
        "被查询者姓名": "subject_name",
        "姓名": "subject_name",
        "被查询者证件类型": "primary_id_type",
        "证件类型": "primary_id_type",
        "被查询者证件号码": "primary_id_number",
        "证件号码": "primary_id_number",
        "查询机构": "query_institution",
        "查询原因": "query_reason",
    },
    "subject_profile": {"证件类型": "primary_id_type", "证件号码": "primary_id_number"},
    "credit_accounts": {
        "账户状态": "account_state",
        "币种": "account_currency",
        "发卡机构": "management_institution",
        "贷款金额": "loan_amount",
        "发放金额": "loan_amount",
        "账户关闭日期": "close_date",
        "还款期数": "repayment_periods",
        "已用额度": "used_amount",
        # Label-only excerpt of native_extraction._apply_account_facts.
        "借款金额": "loan_amount",
        "账户授信额度": "credit_limit",
        "账户币种": "account_currency",
        "业务种类": "business_type",
        "透支余额": "balance",
        "剩余还款期数": "remaining_periods",
        "剩余分期期数": "remaining_periods",
        "应还款日": "scheduled_payment_date",
        "逾期31—60天未还本金": "overdue_principal_31_60",
        "逾期31－60天未还本金": "overdue_principal_31_60",
        "逾期61—90天未还本金": "overdue_principal_61_90",
        "逾期61－90天未还本金": "overdue_principal_61_90",
        "逾期91—180天未还本金": "overdue_principal_91_180",
        "逾期91－180天未还本金": "overdue_principal_91_180",
        "透支180天以上未付余额": "overdue_principal_over_180",
        "最近6个月平均使用额度": "recent_6_month_average_used_amount",
        "最近6个月平均透支余额": "recent_6_month_average_overdraft_balance",
        "未出单的大额专项分期余额": "unbilled_installment_balance",
        "销户日期": "close_date",
        "转出月份": "transfer_out_date",
    },
    "credit_agreements": {
        **{
            label: "facility_limit" if key == "total_limit" else key
            for label, key in _CREDIT_AGREEMENT_FIELD_NAMES.items()
        },
        "授信协议标识": "account_identifier",
        "授信机构": "institution",
        "授信额度用途": "facility_type",
        "授信协议生效日期": "effective_date",
        "授信协议到期日期": "due_date",
    },
    "repayment_responsibilities": {
        **{label: key for label, key in _LIABILITY_LABEL_TO_FIELD.items() if not label.startswith("__")},
        "保证合同编号": "contract_number",
        "主业务借款人": "related_party_name",
        "主业务借款人姓名": "related_party_name",
        "主业务借款人名称": "related_party_name",
        "主业务借款人证件类型": "related_party_id_type",
        "主业务借款人证件号码": "related_party_id_number",
        "责任人类型": "responsibility_type",
        "还款责任金额": "responsibility_amount",
        "还款状态": "repayment_status_code",
    },
    "inquiries": {"编号": "sequence", "序号": "sequence", "查询机构名称": "institution"},
    "postpaid_accounts": {
        "机构名称": "institution",
        "缴费状态": "payment_status",
        "欠费金额": "current_arrears_amount",
    },
    "annotation_statements": {"声明内容": "text", "说明内容": "text", "标注内容": "text"},
}

_TITLES = {
    "个人信用报告": "report_metadata",
    "个人信用报告（本人版）": "report_metadata",
    "个人基本信息": "subject_profile",
    "身份信息": "subject_profile",
    "其他证件信息": "subject_identity_documents",
    "手机号码": "subject_mobile_phones",
    "信息概要": "credit_business_overview",
    "信贷交易信息明细": "credit_accounts",
    "信贷交易授信及负债信息概要": "credit_business_overview",
    "授信协议信息": "credit_agreements",
    "相关还款责任信息": "repayment_responsibilities",
    "非信贷交易信息明细": "postpaid_accounts",
    "后付费记录": "postpaid_accounts",
    "后付费记录账户": "postpaid_accounts",
    "查询记录明细": "inquiries",
    "机构查询记录明细": "inquiries",
    "本人查询记录明细": "inquiries",
    "机构说明": "annotation_statements",
    "本人声明": "annotation_statements",
    "信息主体声明": "annotation_statements",
    "异议标注": "annotation_statements",
    "最近一次还款信息": "credit_account_latest_repayments",
    "还款记录": "credit_account_monthly_performance",
    "还款记录明细": "credit_account_monthly_performance",
    "特殊交易": "credit_account_special_transactions",
    "特殊事件": "credit_account_special_events",
    "大额专项分期信息": "credit_card_large_installments",
}
_CARD_DATASETS = frozenset(
    {
        "report_metadata",
        "report_query",
        "subject_profile",
        "subject_spouse",
        "credit_accounts",
        "credit_agreements",
        "repayment_responsibilities",
        "postpaid_accounts",
    }
)
_ACCOUNT_CHILDREN = frozenset(
    {
        "credit_account_latest_repayments",
        "credit_account_special_transactions",
        "credit_account_special_events",
        "credit_card_large_installments",
        "credit_account_history_windows",
        "credit_account_snapshots",
        "credit_account_monthly_performance",
    }
)
_SECTION_TYPES = {
    "report_header_and_identity": "basic_information",
    "information_summary": "credit_summary",
    "credit_account_detail": "credit_details",
    "credit_agreement": "credit_details",
    "repayment_responsibility": "credit_details",
    "postpaid_detail": "non_credit_transactions",
    "public_information": "public_records",
    "annotations_and_inquiries": "inquiries",
    "report_explanation": "report_explanation",
}


def _scalar(value: str, descriptor: Mapping[str, Any], field_name: str = "") -> Any:
    """Presentation-only decoding; failure returns the original observation."""
    raw = value.strip()
    kind = descriptor.get("type")
    if field_name in {"account_currency", "currency", "reporting_amount_currency"}:
        compact = _label(raw).upper()
        if compact in ISO_4217_CURRENT_CODES:
            return compact
        return CURRENCY_CODE_BY_ALIAS.get(compact, "CNY" if compact == "RMB" else raw)
    if kind == "integer" and re.fullmatch(r"[+-]?\d+", raw):
        try:
            return int(raw)
        except ValueError:
            return raw
    if kind in {"money", "decimal", "number"} and re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", raw):
        return raw.replace(",", "")
    if kind in {"date", "datetime"}:
        match = re.fullmatch(
            r"(\d{4})[-年/.](\d{1,2})[-月/.](\d{1,2})日?(?:[ T](\d{1,2})[:：](\d{2})(?:[:：](\d{2}))?)?", raw
        )
        if match:
            year, month, day, hour, minute, second = match.groups()
            result = f"{year}-{int(month):02d}-{int(day):02d}"
            if hour is not None:
                result += f"T{int(hour):02d}:{minute}:{second or '00'}"
            return result
    return raw


def _table_rows(table: Any) -> list[list[str]]:
    """Raw-cell excerpt of native_extraction._table_rows, retaining numeric 0."""
    metadata = getattr(table, "metadata", None) or {}
    raw = metadata.get("raw_rows") if isinstance(metadata, Mapping) else None
    if isinstance(raw, list) and raw:
        return [[_text(cell) for cell in row] for row in raw if isinstance(row, (list, tuple))]
    headers = [_text(cell) for cell in getattr(table, "headers", ()) or ()]
    rows = [[_text(getattr(cell, "text", "")) for cell in row.cells] for row in getattr(table, "rows", ()) or ()]
    return ([headers] if headers else []) + rows


def _line_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if "|" in line:
            stripped = line.strip()
            cells = stripped.split("|")
            if stripped.startswith("|"):
                cells = cells[1:]
            if stripped.endswith("|"):
                cells = cells[:-1]
            if all(re.fullmatch(r"[ :\-]+", cell) for cell in cells):
                continue
        elif "\t" in line:
            cells = line.split("\t")
        else:
            cells = re.split(r"\s{2,}", line.strip())
        rows.append([cell.strip() for cell in cells])
    return rows


@dataclass
class _BoundRecord:
    values: dict[str, str]
    raw: dict[str, str]
    row: int
    block: int


def _bind_rows(
    rows: list[list[str]],
    aliases: Mapping[str, str],
    *,
    card: bool,
    initial_header: list[str],
) -> tuple[list[_BoundRecord], list[str]]:
    """Read column headers or adjacent/inline KV pairs, without value gates.

    Empty columns retain their positions. Repeated data rows are never merged
    by value; only consecutive, disjoint header blocks of one card are joined.
    """
    if not aliases:
        return [], []
    labels = sorted(aliases, key=len, reverse=True)
    inline = re.compile(r"(" + "|".join(re.escape(label) for label in labels) + r")\s*[:：]\s*")
    result: list[_BoundRecord] = []
    header = list(initial_header)
    block = -1

    def append(values: dict[str, str], raw: dict[str, str], row: int, *, new_row: bool) -> None:
        if not values:
            return
        if card and result and not new_row and not (result[-1].values.keys() & values.keys()):
            result[-1].values.update(values)
            result[-1].raw.update(raw)
        else:
            result.append(_BoundRecord(values, raw, row, block))

    for row_index, row in enumerate(rows):
        nonempty = [cell for cell in row if cell.strip()]
        if not nonempty:
            continue
        exact = [aliases.get(_label(cell)) for cell in row]
        # Unknown header columns are allowed alongside two recognized ones.
        is_header = bool(any(exact)) and (
            all(_label(cell) in aliases for cell in nonempty)
            or (
                any(exact[index] and exact[index + 1] for index in range(len(exact) - 1))
                and not any("：" in cell or ":" in cell for cell in row)
            )
        )
        if is_header:
            header = row
            block = row_index
            continue

        values: dict[str, str] = {}
        raw_values: dict[str, str] = {}
        for column, cell in enumerate(row):
            matches = list(inline.finditer(cell))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(cell)
                value = cell[match.end() : end].strip().rstrip(";；")
                values[aliases[match.group(1)]] = value
                raw_values[match.group(1)] = value
            if not matches and exact[column] and column + 1 < len(row) and not exact[column + 1]:
                values[exact[column]] = row[column + 1].strip()
                raw_values[cell.strip()] = row[column + 1].strip()
        if values:
            header = []
            block = row_index
            append(values, raw_values, row_index, new_row=False)
        elif header:
            for column, label in enumerate(header):
                value = row[column].strip() if column < len(row) else ""
                if _label(label) in aliases:
                    values[aliases[_label(label)]] = value
                if label.strip():
                    raw_values[label.strip()] = value
            append(values, raw_values, row_index, new_row=bool(result and result[-1].block == block))
    return result, header


@dataclass
class _Reader:
    dictionary: dict[str, Any]
    titles: dict[str, str]
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    datasets: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    sections: list[dict[str, Any]] = field(default_factory=list)
    dataset: str = "report_metadata"
    title: str = ""
    family: str = ""
    account: dict[str, Any] | None = None
    postpaid: dict[str, Any] | None = None
    unit: int = 0
    unbound_units: int = 0
    header_dataset: str = ""
    last_header: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name, schema in self.dictionary["datasets"].items():
            aliases = {
                _label(info.get("label")): key
                for key, info in schema.get("columns", {}).items()
                if info.get("label") and not key.endswith("_id") and not _assessment_key(key)
            }
            if "sequence" in schema.get("columns", {}):
                aliases.update({"编号": "sequence", "序号": "sequence"})
            aliases.update(_ALIASES.get(name, {}))
            self.aliases[name] = aliases

    def heading(self, text: str, page: int) -> bool:
        compact = _label(text)
        family = canonical_account_family_heading(text)
        section = canonical_registered_section_heading(text)
        subsection = canonical_registered_subsection_heading(text)
        title = section or (subsection[1] if subsection else compact)
        dataset = "credit_accounts" if family else self.titles.get(title)
        if dataset is None and section is None:
            return False
        if dataset != self.dataset or family:
            self.header_dataset = ""
            self.last_header = []
        self.title = title
        if family:
            self.family = family
            self.account = None
        elif dataset not in _ACCOUNT_CHILDREN:
            self.account = None
            self.family = ""
        if dataset not in {"postpaid_accounts", "postpaid_monthly_performance", "postpaid_history_windows"}:
            self.postpaid = None
        self.dataset = dataset or ""
        if section:
            from docmirror.plugins.credit_report.personal_detail_scanned.section_headings import (
                REGISTERED_SECTION_TEMPLATE_BY_TITLE,
            )

            self.sections.append(
                {
                    "id": f"section_{len(self.sections) + 1}",
                    "title": title,
                    "type": _SECTION_TYPES[REGISTERED_SECTION_TEMPLATE_BY_TITLE[section]],
                    "page_range": [page, page],
                }
            )
        return True

    def infer_dataset(self, rows: list[list[str]]) -> str:
        labels = {_label(cell).split("：", 1)[0].split(":", 1)[0] for row in rows for cell in row}
        # Distinctive printed headers take precedence over a previous section.
        anchors = (
            ({"被查询者姓名", "报告编号"}, "report_metadata"),
            ({"查询日期"}, "inquiries"),
            ({"保证合同编号", "主业务借款人", "责任人类型"}, "repayment_responsibilities"),
            ({"授信限额编号", "授信额度用途"}, "credit_agreements"),
            ({"居住地址", "居住状况"}, "subject_residences"),
            ({"单位性质", "单位地址", "进入本单位年份"}, "subject_employment"),
            ({"手机号码"}, "subject_mobile_phones"),
            ({"出生日期", "性别", "婚姻状况"}, "subject_profile"),
            ({"当前欠费金额", "业务开通日期"}, "postpaid_accounts"),
        )
        for markers, name in anchors:
            if labels & markers:
                return name
        if labels & {"账户标识"}:
            return "credit_agreements" if self.dataset == "credit_agreements" else "credit_accounts"
        if "授信协议标识" in labels and self.dataset != "credit_accounts":
            return "credit_agreements"
        if self.dataset in self.aliases:
            return self.dataset
        scores = {name: len(labels & aliases.keys()) for name, aliases in self.aliases.items()}
        return max(scores, key=scores.get) if scores and max(scores.values()) >= 2 else ""

    def emit(self, name: str, values: dict[str, Any], raw: dict[str, Any], page: int, row: int) -> dict[str, Any]:
        identity = stable_record_id(name, page, self.unit, row, len(self.datasets[name]))
        columns = self.dictionary["datasets"][name].get("columns", {})
        id_keys = [
            key
            for key in columns
            if key.endswith("_id") and key not in {"grid_id", "source_table_id", "target_record_id"}
        ]
        if name in _ACCOUNT_CHILDREN or name == "credit_account_monthly_performance":
            id_keys = [key for key in id_keys if key != "account_id"]
        if name == "postpaid_monthly_performance":
            id_keys = [key for key in id_keys if key != "postpaid_record_id"]
        id_key = id_keys[0] if id_keys else ""
        if name == "credit_business_overview":
            id_key = "credit_business_overview_id"
        normalized = {
            key: _scalar(value, columns.get(key, {}), key) if isinstance(value, str) else value
            for key, value in values.items()
        }
        if id_key:
            normalized[id_key] = identity
        if name in {"report_metadata", "report_query", "subject_profile"} and self.datasets[name]:
            previous = self.datasets[name][-1]
            # One document-level card can span text blocks and tables. Join
            # only disjoint fields; duplicate/conflicting observations remain
            # separate source records and are never resolved by confidence.
            if not (previous["canonical_raw"].keys() & values.keys()):
                normalized.pop(id_key, None)
                previous["normalized"].update(normalized)
                previous["canonical_raw"].update(values)
                previous["raw"].update(raw)
                pages = [*previous["source"]["page_range"], page]
                previous["source"]["page_range"] = [min(pages), max(pages)]
                return previous
        record = {
            "record_id": identity,
            "normalized": normalized,
            "canonical_raw": dict(values),
            "raw": dict(raw),
            "source": {"page_range": [page, page]},
        }
        self.datasets[name].append(record)
        return record

    def records(self, rows: list[list[str]], page: int, offset: int) -> None:
        if not rows:
            return
        name = self.infer_dataset(rows)
        if name == "credit_business_overview":
            self.summary(rows, page, offset)
            return
        if name not in self.aliases:
            self.unbound_units += 1
            return
        records, header = _bind_rows(
            rows,
            self.aliases[name],
            card=name in _CARD_DATASETS,
            initial_header=self.last_header if self.header_dataset == name else [],
        )
        self.header_dataset = name
        self.last_header = header
        if not records:
            if name == "annotation_statements":
                values = {
                    "text": "\n".join(" ".join(row) for row in rows),
                    "annotation_kind": "annotation" if "异议" in self.title else "statement",
                }
                self.emit(name, values, values, page, offset)
            else:
                self.unbound_units += 1
            return
        self.dataset = name
        for bound in records:
            values: dict[str, Any] = dict(bound.values)
            if name == "report_metadata":
                query = dict(values)
                values = {
                    key: value for key, value in values.items() if key not in {"query_institution", "query_reason"}
                }
                self.emit("report_query", query, bound.raw, page, offset + bound.row)
            if name == "credit_accounts" and self.family:
                code, label = _ACCOUNT_TYPE_CODES[self.family]
                values.update(account_type=self.family, pboc_account_type_code=code, pboc_account_type_label=label)
            if name in _ACCOUNT_CHILDREN and self.account:
                values["account_id"] = self.account["record_id"]
            record = self.emit(name, values, bound.raw, page, offset + bound.row)
            if name == "credit_accounts":
                self.account = record
                self.dataset = name
            elif name == "postpaid_accounts":
                self.postpaid = record
                self.dataset = name

    def summary(self, rows: list[list[str]], page: int, offset: int) -> None:
        if len(rows) < 2:
            self.unbound_units += 1
            return
        headers = rows[0]
        for index, row in enumerate(rows[1:], start=1):
            for column, value in enumerate(row[1:], start=1):
                metric = headers[column] if column < len(headers) else ""
                values = {
                    "title": self.title,
                    "metric_name": metric,
                    "business_dimension_value": row[0],
                    "source_value": value,
                    "row_index": index,
                    "column_index": column,
                }
                numeric = _scalar(value, {"type": "money"})
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", _text(numeric)):
                    values.update(numeric_value=numeric, value_type="number")
                else:
                    values.update(text_value=value, value_type="text")
                self.emit("credit_business_overview", values, {metric or str(column): value}, page, offset + index)

    @staticmethod
    def month_columns(row: list[str]) -> dict[int, int]:
        if not row:
            return {}
        result = {}
        for column, value in enumerate(row):
            match = re.fullmatch(r"(0?[1-9]|1[0-2])月?", _label(value))
            if match:
                result[column] = int(match.group(1))
        # Calendar coordinates, not a status whitelist or a 13-column schema.
        other_labels = [_label(cell) for column, cell in enumerate(row) if column not in result]
        calendar_labels = {"", "年", "年份", "月份", "月", "状态", "还款状态", "缴费状态", "备注"}
        explicit_calendar = bool(other_labels) or len(result) >= 5 or any("月" in cell for cell in row)
        return (
            result
            if len(result) >= 2 and explicit_calendar and all(label in calendar_labels for label in other_labels)
            else {}
        )

    def monthly(self, rows: list[list[str]], page: int, offset: int) -> None:
        months = self.month_columns(rows[0])
        postpaid = self.dataset in {"postpaid_accounts", "postpaid_monthly_performance", "postpaid_history_windows"}
        name = "postpaid_monthly_performance" if postpaid else "credit_account_monthly_performance"
        parent = self.postpaid if postpaid else self.account
        grid_id = stable_record_id("grid", page, self.unit, offset)
        index = 1
        while index < len(rows):
            row = rows[index]
            year = next(
                (
                    match.group(1)
                    for column, cell in enumerate(row)
                    if column not in months and (match := re.fullmatch(r"(20\d{2})年?", _label(cell)))
                ),
                None,
            )
            if year is None:
                index += 1
                continue
            amounts: list[str] | None = None
            if not postpaid and index + 1 < len(rows):
                following = rows[index + 1]
                prefix = "".join(cell for column, cell in enumerate(following) if column not in months)
                if not prefix.strip() or "金额" in prefix or "余额" in prefix:
                    amounts = following
            for column, month in months.items():
                if column >= len(row) and (amounts is None or column >= len(amounts)):
                    continue
                values = {
                    "performance_month": f"{year}-{month:02d}",
                    "status_code": row[column] if column < len(row) else "",
                }
                if parent:
                    values["postpaid_record_id" if postpaid else "account_id"] = parent["record_id"]
                    identifier = parent["normalized"].get("account_identifier")
                    if identifier and not postpaid:
                        values["account_identifier"] = identifier
                if not postpaid:
                    values["grid_id"] = grid_id
                    if amounts is not None and column < len(amounts):
                        values["status_amount"] = amounts[column]
                self.emit(name, values, values, page, offset + index)
            index += 2 if amounts is not None else 1

    def consume(self, rows: list[list[str]], page: int) -> None:
        self.unit += 1
        start = 0
        index = 0
        while index < len(rows):
            nonempty = [cell for cell in rows[index] if cell.strip()]
            if len(nonempty) == 1:
                # Flush before changing owner; headings never become values.
                text = nonempty[0]
                is_heading = bool(
                    canonical_account_family_heading(text)
                    or canonical_registered_section_heading(text)
                    or canonical_registered_subsection_heading(text)
                    or _label(text) in self.titles
                )
                if is_heading:
                    self.records(rows[start:index], page, start)
                    self.heading(text, page)
                    index += 1
                    start = index
                    continue
            months = self.month_columns(rows[index]) if rows[index] else {}
            if months and self.dataset in {
                "credit_accounts",
                "credit_account_monthly_performance",
                "postpaid_accounts",
                "postpaid_monthly_performance",
            }:
                self.records(rows[start:index], page, start)
                end = index + 1
                while end < len(rows):
                    candidate = rows[end]
                    has_year = any(re.fullmatch(r"20\d{2}年?", _label(cell)) for cell in candidate)
                    prefix = "".join(cell for column, cell in enumerate(candidate) if column not in months)
                    if not (has_year or not prefix.strip() or "金额" in prefix or "余额" in prefix):
                        break
                    end += 1
                self.monthly(rows[index:end], page, index)
                index = end
                start = end
                continue
            target = self.infer_dataset([rows[index]])
            if (
                target != self.dataset
                and target in self.aliases
                and any(_label(cell) in self.aliases[target] for cell in rows[index])
            ):
                # OCR often combines several printed tables into one physical
                # block. A new distinctive header starts its own decoder.
                self.records(rows[start:index], page, start)
                if target not in _ACCOUNT_CHILDREN and target != "credit_accounts":
                    self.account = None
                    self.family = ""
                if target not in {"postpaid_accounts", "postpaid_monthly_performance", "postpaid_history_windows"}:
                    self.postpaid = None
                self.dataset = target
                start = index
            index += 1
        self.records(rows[start:], page, start)


def _positioned_text_rows(lines: list[dict[str, Any]], page: int, labels: set[str]) -> list[tuple[float, list[str]]]:
    """Reuse native row grouping; keep missing slots at their header columns."""
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import PBOCPersonalDetailNativeParser

    grouped = PBOCPersonalDetailNativeParser._ocr_positioned_rows({"page": page, "lines": lines})
    result: list[tuple[float, list[str]]] = []
    centers: list[float] = []
    for group in grouped:
        texts = [cell.text for cell in group]
        top = min(cell.bbox[1] for cell in group)
        is_header = sum(_label(text) in labels for text in texts) >= 2 or (
            len(texts) == 1 and _label(texts[0]) in labels
        )
        months = _Reader.month_columns(texts)
        if is_header or months:
            centers = [cell.center_x for cell in group]
            if months and len(months) == len(texts):
                # The year header is often blank, so OCR has no token for it.
                gap = centers[1] - centers[0]
                centers.insert(0, centers[0] - gap)
                texts.insert(0, "")
        elif centers:
            aligned = [""] * len(centers)
            for cell in group:
                column = min(range(len(centers)), key=lambda index: abs(centers[index] - cell.center_x))
                aligned[column] = " ".join(part for part in (aligned[column], cell.text) if part)
            texts = aligned
        elif len(group) == 1:
            centers = []
            texts = _line_rows(group[0].text)[0]
        result.append((top, texts))
    return result


def _source_units(parse_result: Any, full_text: str, labels: set[str]) -> Iterator[tuple[int, list[list[str]]]]:
    """Read initial physical tables, KV and OCR lines without the repair context.

    Prefer table cells for text covered by that table, avoiding double decoding
    the same physical content. Other repeated source observations are retained.
    """
    pages = list(getattr(parse_result, "pages", ()) or ())
    ds = getattr(getattr(parse_result, "entities", None), "domain_specific", {}) or {}
    bundles = {
        int(bundle.get("page") or 0): bundle
        for bundle in ds.get("_page_evidence_bundles", ())
        if isinstance(bundle, Mapping)
    }
    emitted = False
    for page in pages:
        page_number = int(getattr(page, "page_number", 1) or 1)
        tables = list(getattr(page, "tables", ()) or ())
        local = bundles.get(page_number, {}).get("local_structure_evidence") or {}
        lines = local.get("lines") or []
        texts = (
            [
                (str(line.get("text") or line.get("content") or ""), line.get("bbox"))
                for line in lines
                if isinstance(line, Mapping)
            ]
            if lines
            else [
                (str(getattr(block, "content", "") or ""), getattr(block, "bbox", None))
                for block in getattr(page, "texts", ()) or ()
            ]
        )
        units: list[tuple[float, int, list[list[str]]]] = []
        for index, table in enumerate(tables):
            bbox = _box(getattr(table, "bbox", None))
            top = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else float(index + 1)
            caption = str(getattr(table, "caption", "") or "")
            rows = ([[caption]] if caption else []) + _table_rows(table)
            units.append((top, index, rows))
        positioned: list[dict[str, Any]] = []
        for index, (text, bbox) in enumerate(texts):
            bbox = _box(bbox)
            covered = False
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                for table in tables:
                    box = _box(getattr(table, "bbox", None))
                    if (
                        isinstance(box, (list, tuple))
                        and len(box) == 4
                        and box[0] <= bbox[0] <= bbox[2] <= box[2]
                        and box[1] <= bbox[1] <= bbox[3] <= box[3]
                        and any(_label(text) in _label("".join(row)) for row in _table_rows(table))
                    ):
                        covered = True
                        break
            if not covered and text.strip():
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    positioned.append({"text": text, "bbox": bbox})
                else:
                    units.append((0.0, len(tables) + index, _line_rows(text)))
        for index, (top, row) in enumerate(_positioned_text_rows(positioned, page_number, labels)):
            units.append((top, len(tables) + len(texts) + index, [row]))
        kv = [[_text(item.key), _text(item.value)] for item in getattr(page, "key_values", ()) or ()]
        if kv:
            units.append((0.0, -1, kv))
        # Consecutive OCR text blocks form a matrix too (header on one line,
        # values on the next). Table boundaries stay intact.
        text_rows: list[list[str]] = []
        for _top, order, rows in sorted(units, key=lambda item: (item[0], item[1])):
            if order >= len(tables) or order == -1:
                text_rows.extend(rows)
            else:
                if text_rows:
                    emitted = True
                    yield page_number, text_rows
                    text_rows = []
                emitted = True
                yield page_number, rows
        if text_rows:
            emitted = True
            yield page_number, text_rows
    if not emitted and full_text.strip():
        yield 1, _line_rows(full_text)


def derive_unvalidated_personal_detail(plugin: Any, parse_result: Any, full_text: str = "") -> ProjectionData:
    """Produce ordinary PBOC v2 business data without checked-pipeline work."""
    logger.warning("Personal detailed PBOC: unvalidated=on; extraction only; repair, re-OCR and validation disabled.")
    dictionary = omit_validation(personal_detail_data_dictionary())
    semantic = omit_validation(personal_detail_semantic_extensions())
    overrides = semantic["community_projection_overrides"]
    overrides.pop("completeness", None)
    overrides["publish_empty_datasets"] = []
    titles = {**{label: name for name, label in overrides["dataset_labels"].items()}, **_TITLES}
    reader = _Reader(dictionary, titles)
    labels = {label for aliases in reader.aliases.values() for label in aliases}
    for page, rows in _source_units(parse_result, full_text, labels):
        reader.consume(rows, page)
    if reader.unbound_units:
        logger.info(
            "Personal detailed PBOC extraction-only: %d source units had no label/column binding.", reader.unbound_units
        )
    datasets = {name: reader.datasets[name] for name in PBOC_DATASET_ORDER if reader.datasets.get(name)}
    metadata = (datasets.get("report_metadata") or [{}])[0].get("normalized", {})
    entities = {key: metadata[key] for key in ("subject_name", "report_number", "report_time") if key in metadata}
    if "primary_id_number" in metadata:
        entities["subject_id"] = metadata["primary_id_number"]
    return ProjectionData(
        projector_id=plugin.projector_id,
        document_type="personal_credit_report_detailed",
        entity_fields=entities,
        domain_facts={
            "document_label": "个人信用报告",
            "report_subtype": "personal_detail",
            "content_mode": detect_credit_report_content_mode(parse_result),
            "data_dictionary": dictionary,
        },
        semantic=semantic,
        datasets=datasets,
        sections=tuple(reader.sections),
    )
