# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical table extraction for native personal detailed credit reports.

The personal-detailed layout is primarily a sequence of labelled tables.  A
native PDF must not be routed through the prose-oriented personal-brief
extractor: doing so drops the account cards entirely.  This module projects
the stable table grammar into the same canonical dataset collections used by
the scanned/OCR path and also emits a lossless source-row ledger.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any

from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import validate_pboc_field
from docmirror.plugins.credit_report.personal_detail_scanned.quality import (
    cn_identity_number_valid,
    header_field_valid,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id

_DATE_RE = re.compile(r"(20\d{2})[.年/-]\s*(\d{1,2})(?:[.月/-]\s*(\d{1,2}))?")
_AS_OF_RE = re.compile(r"截至\s*(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_STATUS_CODES = frozenset({"*", "N", "1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "G", "M", "Z", "#"})
_INQUIRY_REASONS = (
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
    "法人代表、负责人、高管等资信审查",
    "特约商户实名审查",
    "异议处理",
    "司法调查",
)
_INQUIRY_REASON_REPAIRS = {
    "货后管理": "贷后管理",
    "贷后智理": "贷后管理",
    "货款审批": "贷款审批",
    "资款审批": "贷款审批",
    "信用卡审抛": "信用卡审批",
}

_ACCOUNT_LABELS = frozenset(
    {
        "管理机构",
        "发卡机构",
        "账户标识",
        "开立日期",
        "到期日期",
        "借款金额",
        "账户授信额度",
        "共享授信额度",
        "账户币种",
        "币种",
        "业务种类",
        "担保方式",
        "还款期数",
        "还款频率",
        "还款方式",
        "共同借款标志",
        "账户状态",
        "状态",
        "五级分类",
        "余额",
        "透支余额",
        "剩余还款期数",
        "剩余分期期数",
        "本月应还款",
        "应还款日",
        "账单日",
        "本月实还款",
        "最近一次还款日期",
        "当前逾期期数",
        "当前逾期总额",
        "逾期31—60天未还本金",
        "逾期31－60天未还本金",
        "逾期61—90天未还本金",
        "逾期61－90天未还本金",
        "逾期91—180天未还本金",
        "逾期91－180天未还本金",
        "逾期180天以上未还本金",
        "透支180天以上未付余额",
        "已用额度",
        "最近6个月平均使用额度",
        "最大使用额度",
        "最近6个月平均透支余额",
        "最大透支余额",
        "未出单的大额专项分期余额",
        "账户关闭日期",
        "销户日期",
        "转出月份",
        "大额专项分期额度",
        "分期额度生效日期",
        "分期额度到期日期",
        "已用分期金额",
        "还款日期",
        "还款金额",
        "当前还款状态",
        "授信协议标识",
        "生效日期",
        "授信额度用途",
        "授信额度",
        "授信限额",
        "授信限额编号",
        "责任人类型",
        "还款责任金额",
        "保证合同编号",
        "主业务借款人",
        "主业务借款人证件类型",
        "主业务借款人证件号码",
        "还款状态",
        "逾期月数",
        "机构名称",
        "业务类型",
        "业务开通日期",
        "当前缴费状态",
        "当前欠费金额",
        "记账年月",
        "债权接收日期",
        "原债权人",
        "原债务业务种类",
        "债权金额",
        "债权转移时的还款状态",
        "特殊交易类型",
        "发生日期",
        "变更月数",
        "发生金额",
        "明细记录",
    }
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _business_text(value: Any) -> str:
    """Remove PDF line-wrap whitespace from compact domain labels and names."""
    return _compact(value)


def _identifier(value: Any) -> str | None:
    compact = re.sub(r"\s+", "", str(value or "")).strip()
    return compact if compact and compact != "--" else None


def _typed_identifier(value: Any) -> str | None:
    identifier = _identifier(value)
    if not identifier or not re.fullmatch(r"[A-Z0-9-]{8,80}", identifier, re.IGNORECASE):
        return None
    if not re.search(r"[A-Z]", identifier, re.IGNORECASE) or not re.search(r"\d", identifier):
        return None
    return identifier


def _dedupe_prefixed_identifiers(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        identifier = str(record.get(field) or "")
        replaced = False
        for index, existing in enumerate(kept):
            prior = str(existing.get(field) or "")
            if identifier and prior and (identifier.startswith(prior) or prior.startswith(identifier)):
                if len(identifier) > len(prior):
                    kept[index] = record
                replaced = True
                break
        if not replaced:
            kept.append(record)
    return kept


def _dedupe_liability_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suppress full-page OCR replays of an already parsed liability card.

    OCR can transpose chunks of a long guarantee-contract identifier, so an
    identifier-prefix comparison alone is not sufficient.  A replay is only
    treated as the same source card when at least three independent business
    fields agree.  Earlier native-table rows win over later tolerant OCR rows.
    """
    kept: list[dict[str, Any]] = []
    comparison_fields = (
        "related_party_id_number",
        "responsibility_amount",
        "balance",
        "open_date",
        "due_date",
    )
    for record in records:
        contract_number = str(record.get("contract_number") or "")
        duplicate = False
        for existing in kept:
            prior_contract = str(existing.get("contract_number") or "")
            if contract_number and prior_contract and (
                contract_number.startswith(prior_contract) or prior_contract.startswith(contract_number)
            ):
                duplicate = True
                break
            matching_fields = sum(
                1
                for field in comparison_fields
                if record.get(field) not in (None, "")
                and existing.get(field) not in (None, "")
                and record.get(field) == existing.get(field)
            )
            if matching_fields >= 3:
                duplicate = True
                break
        if not duplicate:
            kept.append(record)
    for sequence, record in enumerate(kept, start=1):
        record["sequence"] = sequence
    return kept


def _number(value: Any) -> int | str | None:
    raw = _compact(value).replace(",", "")
    if raw in {"", "--", "-"}:
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def _date(value: Any) -> str | None:
    raw = _compact(value)
    if raw in {"", "--", "长期"}:
        return None
    match = _DATE_RE.search(raw)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}" + (f"-{int(day):02d}" if day else "")


def _currency(value: Any) -> str | None:
    raw = _compact(value)
    if not raw or raw == "--":
        return None
    if "人民币" in raw:
        return "CNY"
    if "美元" in raw:
        return "USD"
    if "欧元" in raw:
        return "EUR"
    if "港" in raw and "元" in raw:
        return "HKD"
    return raw


def _table_rows(table: Any) -> list[list[str]]:
    metadata = getattr(table, "metadata", None) or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
    if isinstance(raw_rows, list) and raw_rows:
        return [[str(cell or "") for cell in row] for row in raw_rows if isinstance(row, list)]
    headers = [str(value or "") for value in getattr(table, "headers", None) or []]
    rows = [
        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", None) or []]
        for row in getattr(table, "rows", None) or []
    ]
    return ([headers] if headers else []) + rows


def _source_ref(
    page: Any,
    table: Any,
    *,
    row: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "source": "native_detail_table",
        "logical_page": int(getattr(page, "page_number", 0) or 0),
        "source_page": int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0),
        "table_id": str(getattr(table, "table_id", "") or ""),
    }
    if row is not None:
        ref["row"] = row
    if column is not None:
        ref["column"] = column
    metadata = getattr(table, "metadata", None) or {}
    bbox = None
    if row is not None and column is not None and isinstance(metadata, dict):
        cell_bboxes = metadata.get("cell_bboxes")
        if (
            isinstance(cell_bboxes, list)
            and 0 <= row < len(cell_bboxes)
            and isinstance(cell_bboxes[row], list)
            and 0 <= column < len(cell_bboxes[row])
        ):
            candidate = cell_bboxes[row][column]
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                bbox = candidate
                ref["source"] = "native_detail_table_cell"
                ref["geometry_scope"] = "cell"
        cell_evidence_ids = metadata.get("cell_evidence_ids")
        if (
            isinstance(cell_evidence_ids, list)
            and 0 <= row < len(cell_evidence_ids)
            and isinstance(cell_evidence_ids[row], list)
            and 0 <= column < len(cell_evidence_ids[row])
            and isinstance(cell_evidence_ids[row][column], list)
        ):
            ref["evidence_ids"] = [str(item) for item in cell_evidence_ids[row][column] if item]
    if bbox is None:
        bbox = getattr(table, "bbox", None)
        if bbox and len(bbox) == 4:
            ref["geometry_scope"] = "table"
    if bbox and len(bbox) == 4:
        ref["bbox"] = list(bbox)
    return ref


def _nonempty(row: list[str]) -> list[str]:
    return [_clean(cell) for cell in row if _clean(cell)]


def _pairs(rows: list[list[str]]) -> dict[str, str]:
    """Return labelled values from adjacent rows without collapsing grid columns."""
    out: dict[str, str] = {}
    for index in range(len(rows) - 1):
        label_cells = [
            (column, _compact(cell)) for column, cell in enumerate(rows[index]) if _compact(cell) in _ACCOUNT_LABELS
        ]
        value_cells = [
            (column, _clean(cell))
            for column, cell in enumerate(rows[index + 1])
            if _clean(cell) and _compact(cell) not in _ACCOUNT_LABELS
        ]
        if not value_cells:
            continue
        if not label_cells:
            labels = _nonempty(rows[index])
            values = _nonempty(rows[index + 1])
            if labels and len(labels) == len(values):
                for label, value in zip(labels, values, strict=True):
                    out.setdefault(_compact(label), value)
            continue
        if len(label_cells) == len(value_cells):
            for (_label_column, label), (_value_column, value) in zip(label_cells, value_cells, strict=True):
                out.setdefault(label, value)
            continue
        unused = set(range(len(value_cells)))
        for label_column, label in label_cells:
            if not unused:
                break
            value_index = min(unused, key=lambda item: abs(value_cells[item][0] - label_column))
            value_column, value = value_cells[value_index]
            # A value belongs to the closest visual label. This prevents a
            # missing cell from shifting every value that follows it.
            closest_label = min(label_cells, key=lambda item: abs(item[0] - value_column))
            if closest_label[0] != label_column:
                continue
            out.setdefault(label, value)
            unused.remove(value_index)
    return out


def _label_row(row: list[str]) -> bool:
    return sum(_compact(cell) in _ACCOUNT_LABELS for cell in row) >= 2


def _column_centers(table: Any, width: int) -> list[float]:
    metadata = getattr(table, "metadata", None) or {}
    geometry = metadata.get("geometry") if isinstance(metadata, dict) else None
    bands = geometry.get("col_bands") if isinstance(geometry, dict) else None
    if isinstance(bands, list) and bands:
        centers: list[float] = []
        for band in bands:
            if isinstance(band, list) and len(band) >= 3:
                centers.append((float(band[0]) + float(band[2])) / 2.0)
            else:
                centers.append(float(len(centers)))
        return centers
    return [float(index) for index in range(width)]


def _month_centers(table: Any, rows: list[list[str]]) -> dict[int, float]:
    width = max((len(row) for row in rows), default=0)
    centers = _column_centers(table, width)
    for row in rows:
        candidates = {
            int(_compact(cell)): centers[index]
            for index, cell in enumerate(row)
            if index < len(centers) and re.fullmatch(r"(?:[1-9]|1[0-2])", _compact(cell))
        }
        if len(candidates) >= 5:
            return candidates
    return {}


def _nearest_month(column: int, centers: list[float], months: dict[int, float]) -> int | None:
    if not months or column >= len(centers):
        return None
    x = centers[column]
    return min(months, key=lambda month: abs(months[month] - x))


def _field(facts: dict[str, str], *labels: str) -> str | None:
    for label in labels:
        value = facts.get(_compact(label))
        if value not in (None, ""):
            return value
    return None


def _status_fields(raw_status: Any) -> dict[str, Any]:
    raw = _compact(raw_status)
    if not raw:
        return {}
    status_labels = {
        "正常": "active",
        "逾期": "active",
        "呆账": "active",
        "结清": "settled",
        "销户": "closed",
        "转出": "transferred_out",
        "结束": "closed",
        "未激活": "inactive",
        "冻结": "frozen",
        "止付": "suspended",
        "银行止付": "suspended",
        "司法追偿": "recovery",
        "担保物不足": "collateral_shortfall",
        "强制平仓": "forced_liquidation",
        "催收": "collection",
    }
    source_label = raw
    status = status_labels.get(raw)
    repaired_noise = False
    if status is None:
        matches = [label for label in status_labels if label in raw]
        if len(matches) == 1:
            remaining = raw.replace(matches[0], "", 1)
            if len(remaining) <= 2:
                source_label = matches[0]
                status = status_labels[source_label]
                repaired_noise = True
    if status is None:
        status = raw
    lifecycle = {
        "active": "open",
        "inactive": "open",
        "settled": "settled",
        "closed": "closed",
        "transferred_out": "transferred_out",
    }.get(status)
    result: dict[str, Any] = {
        "account_status": status,
        "account_status_raw": raw,
        "account_status_resolution": "ocr_noise_normalized" if repaired_noise else "resolved"
        if status in status_labels.values()
        else "unresolved",
    }
    if lifecycle:
        result["account_lifecycle_state"] = lifecycle
    if source_label == "未激活":
        result["card_activation_state"] = "not_activated"
    elif source_label == "呆账":
        result["credit_quality_status"] = "bad_debt"
    if source_label == "逾期":
        result["current_overdue"] = True
        result["current_overdue_status"] = "overdue"
    elif source_label in {"正常", "结清", "销户", "转出"}:
        result["current_overdue"] = False
        result["current_overdue_status"] = "not_overdue"
    return result


def _apply_account_facts(account: dict[str, Any], rows: list[list[str]]) -> None:
    facts = _pairs(rows)
    mappings: tuple[tuple[str, tuple[str, ...], Any], ...] = (
        ("management_institution", ("管理机构", "发卡机构"), _business_text),
        ("account_identifier", ("账户标识",), _identifier),
        ("open_date", ("开立日期",), _date),
        ("due_date", ("到期日期",), _date),
        ("loan_amount", ("借款金额",), _number),
        ("credit_limit", ("账户授信额度", "账户授信额度"), _number),
        ("shared_credit_limit", ("共享授信额度",), _number),
        ("business_type", ("业务种类",), _business_text),
        ("guarantee_type", ("担保方式",), _business_text),
        ("repayment_periods", ("还款期数",), _number),
        ("repayment_frequency", ("还款频率",), _business_text),
        ("repayment_method", ("还款方式",), _business_text),
        ("co_borrower_flag", ("共同借款标志",), _business_text),
        ("balance", ("余额", "透支余额"), _number),
        ("remaining_periods", ("剩余还款期数", "剩余分期期数"), _number),
        ("scheduled_payment", ("本月应还款",), _number),
        ("actual_payment", ("本月实还款",), _number),
        ("scheduled_payment_date", ("应还款日",), _date),
        ("billing_date", ("账单日",), _date),
        ("last_repayment_date", ("最近一次还款日期",), _date),
        ("current_overdue_periods", ("当前逾期期数",), _number),
        ("current_overdue_amount", ("当前逾期总额",), _number),
        ("overdue_principal_31_60", ("逾期31—60天未还本金", "逾期31－60天未还本金"), _number),
        ("overdue_principal_61_90", ("逾期61—90天未还本金", "逾期61－90天未还本金"), _number),
        ("overdue_principal_91_180", ("逾期91—180天未还本金", "逾期91－180天未还本金"), _number),
        ("overdue_principal_over_180", ("逾期180天以上未还本金", "透支180天以上未付余额"), _number),
        ("used_amount", ("已用额度",), _number),
        ("recent_6_month_average_used_amount", ("最近6个月平均使用额度",), _number),
        ("maximum_used_amount", ("最大使用额度",), _number),
        ("recent_6_month_average_overdraft_balance", ("最近6个月平均透支余额",), _number),
        ("maximum_overdraft_balance", ("最大透支余额",), _number),
        ("unbilled_installment_balance", ("未出单的大额专项分期余额",), _number),
        ("five_tier_class", ("五级分类",), _clean),
        ("close_date", ("账户关闭日期", "销户日期"), _date),
        ("transfer_out_date", ("转出月份",), _date),
    )
    for target, labels, converter in mappings:
        raw = _field(facts, *labels)
        if raw is None:
            continue
        value = converter(raw)
        if value not in (None, ""):
            account[target] = value
            account.setdefault("canonical_raw", {})[target] = raw

    raw_currency = _field(facts, "账户币种", "币种")
    currency = _currency(raw_currency)
    if currency:
        account["currency"] = currency
        account["account_currency"] = currency
        account.setdefault("canonical_raw", {})["currency"] = raw_currency

    raw_status = _field(facts, "账户状态", "状态")
    if raw_status:
        account.update(_status_fields(raw_status))
        account.setdefault("canonical_raw", {})["account_status"] = raw_status

    for row in rows:
        text = _clean(" ".join(row))
        as_of = _AS_OF_RE.search(text)
        if as_of:
            account["snapshot_date"] = f"{int(as_of.group(1)):04d}-{int(as_of.group(2)):02d}-{int(as_of.group(3)):02d}"
        inactive = re.search(r"账户状态为[“\"]([^”\"]+)", text)
        if inactive:
            account.update(_status_fields(inactive.group(1)))

    if account.get("due_date"):
        account["contract_maturity_date"] = account["due_date"]
    account.setdefault("reporting_amount_currency", "CNY")
    account.setdefault("amount_unit", "yuan")
    account.setdefault("reporting_amount_unit", "yuan")
    account.setdefault("reporting_amount_precision", 0)


def _repayment_records(
    page: Any,
    table: Any,
    rows: list[list[str]],
    account: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = account.setdefault("_repayment_context", {})
    observed_months = _month_centers(table, rows)
    if observed_months:
        context["month_centers"] = observed_months
    months = dict(context.get("month_centers") or {})
    width = max((len(row) for row in rows), default=0)
    centers = _column_centers(table, width)

    def amount_values(row: list[str]) -> dict[int, str]:
        values: dict[int, str] = {}
        for column, cell in enumerate(row):
            raw = _compact(cell)
            if not raw or not re.fullmatch(r"(?:--|-?\d[\d,]*)", raw):
                continue
            month = _nearest_month(column, centers, months)
            if month is not None:
                values[month] = raw
        return values

    def make_record(
        *,
        year: int,
        month: int,
        status: str,
        amount_raw: str = "",
        row_index: int,
        source_refs: list[dict[str, Any]] | None = None,
        confidence: float = 1.0,
        inferred: bool = False,
    ) -> dict[str, Any]:
        account_id = str(account["account_id"])
        return {
            "repayment_id": stable_record_id("credit_repayment", account_id, year, month),
            "account_id": account_id,
            "account_identifier": account.get("account_identifier"),
            "grid_id": str(getattr(table, "table_id", "") or ""),
            "year": year,
            "month": month,
            "status": status,
            "overdue_amount": None if amount_raw in {"", "--"} else amount_raw.replace(",", ""),
            "source": "native_detail_repayment_grid_gap_recovery" if inferred else "native_detail_repayment_grid",
            "source_refs": source_refs or [_source_ref(page, table, row=row_index)],
            "confidence": confidence,
            **({"status_inferred_from_adjacent_months": True} if inferred else {}),
        }

    records: list[dict[str, Any]] = []
    pending = context.pop("pending_status", None)
    if isinstance(pending, dict) and rows and months:
        first_values = amount_values(rows[0])
        if first_values:
            for month, status in pending.get("statuses", {}).items():
                records.append(
                    make_record(
                        year=int(pending["year"]),
                        month=int(month),
                        status=str(status),
                        amount_raw=first_values.get(int(month), ""),
                        row_index=0,
                        source_refs=[
                            *list(pending.get("source_refs") or []),
                            _source_ref(page, table, row=0),
                        ],
                    )
                )

    for row_index, row in enumerate(rows):
        year_cell = next(
            ((index, _compact(cell)) for index, cell in enumerate(row) if re.fullmatch(r"20\d{2}", _compact(cell))),
            None,
        )
        if year_cell is None:
            continue
        year_col, year_raw = year_cell
        status_cells = [
            (index, _compact(cell))
            for index, cell in enumerate(row)
            if index != year_col and _compact(cell) in _STATUS_CODES
        ]
        if not status_cells:
            continue
        statuses: dict[int, str] = {}
        for column, status in status_cells:
            month = _nearest_month(column, centers, months)
            if month is not None:
                statuses[month] = status
        if not statuses:
            continue

        amount_row_index = row_index + 1
        amounts = (
            amount_values(rows[amount_row_index])
            if amount_row_index < len(rows)
            and not any(re.fullmatch(r"20\d{2}", _compact(cell)) for cell in rows[amount_row_index])
            else {}
        )
        # A native grid can occasionally omit one status glyph while keeping
        # the paired amount. Recover only a sandwiched value whose two adjacent
        # statuses agree; this is deterministic and intentionally conservative.
        inferred_months: set[int] = set()
        for month in sorted(amounts):
            if month in statuses or month <= 1 or month >= 12:
                continue
            previous = statuses.get(month - 1)
            following = statuses.get(month + 1)
            if previous == following and previous in {"1", "2", "3", "4", "5", "6", "7"}:
                statuses[month] = str(previous)
                inferred_months.add(month)

        if amounts or amount_row_index < len(rows):
            for month, status in sorted(statuses.items()):
                records.append(
                    make_record(
                        year=int(year_raw),
                        month=month,
                        status=status,
                        amount_raw=amounts.get(month, ""),
                        row_index=row_index,
                        confidence=0.8 if month in inferred_months else 1.0,
                        inferred=month in inferred_months,
                    )
                )
        else:
            context["pending_status"] = {
                "year": int(year_raw),
                "statuses": statuses,
                "source_refs": [_source_ref(page, table, row=row_index)],
            }
    return records, context


def _account_base(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows[:4] for cell in row))
    return "账户标识" in compact and ("管理机构" in compact or "发卡机构" in compact) and not _other_entity_table(rows)


def _other_entity_table(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows[:4] for cell in row))
    return any(
        marker in compact
        for marker in (
            "保证合同编号",
            "授信协议标识",
            "机构名称业务类型业务开通日期",
            "主管税务机关",
            "立案法院",
            "执行法院",
            "处罚机构",
            "查询日期查询机构查询原因",
        )
    )


def _account_heading_for_table(page: Any, table: Any) -> dict[str, str]:
    bbox = getattr(table, "bbox", None) or []
    table_top = float(bbox[1]) if len(bbox) == 4 else float("inf")
    candidates: list[tuple[float, str]] = []
    for text_block in getattr(page, "texts", None) or []:
        text = _clean(getattr(text_block, "content", "") or getattr(text_block, "text", "") or "")
        compact_text = _compact(text)
        if not re.match(r"^(?:账户|业务)\s*\d{1,3}", compact_text):
            continue
        text_bbox = getattr(text_block, "bbox", None) or []
        bottom = float(text_bbox[3]) if len(text_bbox) == 4 else 0.0
        if bottom <= table_top + 1.0:
            candidates.append((bottom, text))
    if not candidates:
        return {}
    bottom, raw_text = max(candidates, key=lambda item: item[0])
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    maximum_gap = max(72.0, page_height * 0.10) if page_height else 72.0
    if table_top != float("inf") and table_top - bottom > maximum_gap:
        return {}
    text = _compact(raw_text)
    tail = re.search(r"卡片尾号[：:]?([0-9]{4})", text)
    agreement = re.search(r"授信协议标识[：:]?([A-Z0-9]+)", text, re.IGNORECASE)
    return {
        **({"card_tail": tail.group(1)} if tail else {}),
        **({"credit_agreement_identifier": agreement.group(1)} if agreement else {}),
    }


def _account_events(
    account: dict[str, Any],
    page: Any,
    table: Any,
    rows: list[list[str]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    page_number = int(getattr(page, "page_number", 0) or 0)
    for row_index, row in enumerate(rows):
        compact = _compact("".join(row))
        event_type: str | None = None
        if "特殊交易类型" in compact:
            event_type = "special_transaction"
        elif "大额专项分期额度" in compact:
            event_type = "large_installment"
        elif "还款日期" in compact and "还款金额" in compact and "当前还款状态" in compact:
            event_type = "latest_repayment"
        elif "特殊事件说明" in compact:
            event_type = "special_event_note"
        if event_type is None or row_index + 1 >= len(rows):
            continue
        value_row = rows[row_index + 1]
        values = _nonempty(value_row)
        if not values:
            continue
        facts = _pairs([row, value_row])
        if event_type == "special_transaction":
            payload = {
                "transaction_type": _field(facts, "特殊交易类型") or values[0],
                "event_date": _date(_field(facts, "发生日期")),
                "changed_months": _number(_field(facts, "变更月数")),
                "amount": _number(_field(facts, "发生金额")),
                "details": None if _compact(_field(facts, "明细记录")) == "--" else _clean(_field(facts, "明细记录")),
            }
        elif event_type == "large_installment":
            payload = {
                "installment_limit": _number(_field(facts, "大额专项分期额度")),
                "effective_date": _date(_field(facts, "分期额度生效日期")),
                "expiry_date": _date(_field(facts, "分期额度到期日期")),
                "used_installment_amount": _number(_field(facts, "已用分期金额")),
            }
        elif event_type == "latest_repayment":
            payload = {
                "five_tier_class": _clean(_field(facts, "五级分类")),
                "balance": _number(_field(facts, "余额")),
                "repayment_date": _date(_field(facts, "还款日期")),
                "repayment_amount": _number(_field(facts, "还款金额")),
                "repayment_status": _clean(_field(facts, "当前还款状态")),
            }
        else:
            payload = {"details": values[0]}
        payload = {key: value for key, value in payload.items() if value not in (None, "")}
        event_id = stable_record_id(
            "personal_detail_account_event",
            account.get("account_id"),
            event_type,
            page_number,
            row_index,
        )
        events.append(
            {
                "record_id": event_id,
                "account_event_id": event_id,
                "account_id": account.get("account_id"),
                "event_type": event_type,
                **payload,
                "source": "native_personal_detail_account_event",
                "source_refs": [_source_ref(page, table, row=row_index)],
                "confidence": 1.0,
            }
        )
    return events


def _extract_table_accounts(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accounts: list[dict[str, Any]] = []
    repayments: list[dict[str, Any]] = []
    account_events: list[dict[str, Any]] = []
    category_counts: defaultdict[str, int] = defaultdict(int)
    phase = "non_revolving_loan"
    current: dict[str, Any] | None = None
    pending_labels: list[str] | None = None
    current_table_id = ""
    current_logical_page = 0
    continuation_check = getattr(parse_result, "tables_continue", None)

    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            compact = _compact(" ".join(cell for row in rows[:6] for cell in row))
            if _account_base(rows):
                if "发卡机构" in compact:
                    account_type = "credit_card" if "业务种类" in compact else "quasi_credit_card"
                    phase = account_type
                elif "账户授信额度" in compact:
                    account_type = "revolving_loan_subaccount"
                    phase = account_type
                elif phase in {"revolving_loan_subaccount", "revolving_loan_account"}:
                    account_type = "revolving_loan_account"
                    phase = account_type
                else:
                    account_type = "non_revolving_loan"

                category_counts[account_type] += 1
                current = {
                    "account_id": f"credit_account:{account_type}:{category_counts[account_type]}",
                    "sequence": len(accounts) + 1,
                    "category_sequence": category_counts[account_type],
                    "account_type": account_type,
                    "source": "native_detail_account_table",
                    "source_refs": [_source_ref(page, table)],
                    "confidence": 1.0,
                    "canonical_raw": {},
                }
                current.update(_account_heading_for_table(page, table))
                if account_type in {"credit_card", "quasi_credit_card"}:
                    current["credit_card_type"] = account_type
                _apply_account_facts(current, rows)
                accounts.append(current)
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                account_events.extend(_account_events(current, page, table, rows))
                pending_labels = rows[-1] if rows and _label_row(rows[-1]) else None
                current_table_id = str(getattr(table, "table_id", "") or "")
                current_logical_page = int(getattr(page, "page_number", 0) or 0)
                continue

            logical_page = int(getattr(page, "page_number", 0) or 0)
            crosses_page = bool(current_logical_page and logical_page != current_logical_page)
            if current is not None and crosses_page:
                candidate_table_id = str(getattr(table, "table_id", "") or "")
                continuation = (
                    continuation_check(current_table_id, candidate_table_id) if callable(continuation_check) else None
                )
                if continuation is not True:
                    current = None
                    pending_labels = None
                    current_table_id = ""
                    current_logical_page = 0

            if current is not None and not _other_entity_table(rows):
                continuation_values = [value for row in rows for value in _nonempty(row)]
                if (
                    len(continuation_values) == 1
                    and continuation_values[0] == "期"
                    and account_events
                    and account_events[-1].get("account_id") == current.get("account_id")
                    and str(account_events[-1].get("transaction_type") or "").endswith("分")
                ):
                    account_events[-1]["transaction_type"] = (
                        str(account_events[-1]["transaction_type"]) + continuation_values[0]
                    )
                fact_rows = ([pending_labels] if pending_labels else []) + rows
                _apply_account_facts(current, fact_rows)
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                account_events.extend(_account_events(current, page, table, rows))
                pending_labels = rows[-1] if rows and _label_row(rows[-1]) else None
                current_table_id = str(getattr(table, "table_id", "") or current_table_id)
                current_logical_page = logical_page or current_logical_page

    account_identifiers = {str(account["account_id"]): account.get("account_identifier") for account in accounts}
    deduped: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in repayments:
        if not record.get("account_identifier"):
            record["account_identifier"] = account_identifiers.get(str(record["account_id"]))
        key = (str(record["account_id"]), int(record["year"]), int(record["month"]))
        deduped[key] = record
    for account in accounts:
        account.pop("_repayment_context", None)
    return accounts, list(deduped.values()), account_events


_ACCOUNT_SECTION_END = (
    "授信协议信息",
    "相关还款责任信息",
    "非信贷交易信息明细",
    "公共信息明细",
    "查询记录",
    "报告说明",
)


def _account_family_from_heading(value: Any) -> tuple[str, str] | None:
    """Resolve the printed PBOC family without collapsing R1 and R2."""
    compact = re.sub(r"[\s（）()：:、，,。._-]+", "", str(value or ""))
    if "非循环贷账户" in compact:
        return "non_revolving_loan", "exact"
    if "循环贷账户一" in compact:
        return "revolving_loan_subaccount", "exact"
    if "循环贷账户二" in compact:
        return "revolving_loan_account", "exact"
    if "准贷记卡账户" in compact:
        return "quasi_credit_card", "exact"
    if "贷记卡账户" in compact:
        return "credit_card", "exact"
    if "循环贷账户" in compact:
        return "revolving_loan_subaccount", "ambiguous_missing_variant"
    return None
_ACCOUNT_ANCHOR_RE = re.compile(
    r"^[^\u3400-\u9fff0-9]*账户\s*(\d{1,3})?\s*(?:[：:(（]|$|\s)"
)
_ACCOUNT_IDENTIFIER_PREFIX_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z][0-9]{8}[A-Z][A-Z0-9]{0,20})(?![A-Z0-9])",
    re.I,
)
_ACCOUNT_IDENTIFIER_SUFFIX_RE = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9]{4,})(?=[A-Z0-9]*\d)[A-Z0-9]{4,}(?![A-Z0-9])",
    re.I,
)


def _account_identifier_from_detail(detail: list[dict[str, Any]]) -> str | None:
    """Recover one identifier from the canonical account table row text.

    Whole-page OCR commonly emits the ten-character identifier prefix as its
    own line and the numeric suffix inside the following multi-cell row.  The
    printed open date is the canonical boundary that prevents credit limits or
    other numeric fields from being absorbed into the identifier.
    """
    header = next(
        (
            index
            for index, line in enumerate(detail)
            if "账户标识" in _compact(str(line.get("text") or ""))
        ),
        None,
    )
    if header is None:
        return None

    prefix = ""
    suffix_parts: list[str] = []
    saw_date_boundary = False
    for line in detail[header + 1 : header + 9]:
        text = str(line.get("text") or "").upper()
        if "授信协议标识" in _compact(text):
            continue
        date_match = re.search(r"(?:19|20)\d{2}\s*(?:[./,，-]|年)\s*\d{1,2}", text)
        before_date = text[: date_match.start()] if date_match else text
        matches = list(_ACCOUNT_IDENTIFIER_PREFIX_RE.finditer(before_date))
        if matches:
            if prefix or len(matches) != 1:
                return None
            prefix = matches[0].group(1).upper()
            before_date = before_date[matches[0].end() :]
        if prefix:
            suffix_parts.extend(match.group(0) for match in _ACCOUNT_IDENTIFIER_SUFFIX_RE.finditer(before_date))
        if date_match and prefix:
            saw_date_boundary = True
            break

    candidate = prefix + "".join(suffix_parts)
    if saw_date_boundary and 24 <= len(candidate) <= 80 and re.fullmatch(r"[A-Z0-9]+", candidate):
        return candidate
    return None


def _account_anchor_skeletons(parse_result: Any) -> list[dict[str, Any]]:
    """Build the canonical account row skeleton from printed account anchors."""
    evidence_loader = getattr(parse_result, "corrected_evidence_pages", None)
    if not callable(evidence_loader):
        return []
    flattened: list[dict[str, Any]] = []
    active_type = ""
    active_family_quality = ""
    for page in evidence_loader():
        lines = [line for line in page.get("lines") or () if isinstance(line, dict)]
        for index, line in enumerate(lines):
            text = str(line.get("text") or line.get("content") or "")
            compact = _compact(text)
            if active_type and any(marker in compact for marker in _ACCOUNT_SECTION_END):
                active_type = ""
                active_family_quality = ""
            family = _account_family_from_heading(compact)
            marker = family[0] if family else None
            if marker is not None:
                active_type = marker
                active_family_quality = family[1]
            flattened.append(
                {
                    **line,
                    "text": text,
                    "page": int(page.get("page") or 0),
                    "source_page": int(page.get("source_page") or 0),
                    "account_type": active_type,
                    "account_family_quality": active_family_quality,
                    "line_index": index,
                }
            )

    starts = [
        index
        for index, line in enumerate(flattened)
        if line.get("account_type") and _ACCOUNT_ANCHOR_RE.search(str(line.get("text") or ""))
    ]
    category_counts: defaultdict[str, int] = defaultdict(int)
    used: defaultdict[str, set[int]] = defaultdict(set)
    skeletons: list[dict[str, Any]] = []
    for position, start in enumerate(starts):
        anchor = flattened[start]
        account_type = str(anchor["account_type"])
        next_start = starts[position + 1] if position + 1 < len(starts) else len(flattened)
        end = next_start
        for index in range(start + 1, next_start):
            if flattened[index].get("account_type") != account_type:
                end = index
                break
        detail = flattened[start:end]
        match = _ACCOUNT_ANCHOR_RE.search(str(anchor.get("text") or ""))
        detected = int(match.group(1)) if match and match.group(1) else 0
        category_counts[account_type] += 1
        ordinal = detected if detected > 0 and detected not in used[account_type] else category_counts[account_type]
        while ordinal in used[account_type]:
            ordinal += 1
        used[account_type].add(ordinal)
        account_id = f"credit_account:{account_type}:{ordinal}"
        skeleton = {
                "account_id": account_id,
                "sequence": len(skeletons) + 1,
                "category_sequence": ordinal,
                "account_type": account_type,
                "account_family_quality": str(anchor.get("account_family_quality") or ""),
                "account_state": "unknown",
                "source": "candidate_b_account_anchor",
                "page": int(anchor.get("page") or 0),
                "source_page": int(anchor.get("source_page") or 0),
                "bbox": list(anchor.get("bbox") or ()),
                "source_refs": [
                    {
                        "source": "candidate_b_account_anchor",
                        "logical_page": int(anchor.get("page") or 0),
                        "source_page": int(anchor.get("source_page") or 0),
                        "bbox": list(anchor.get("bbox") or ()),
                        "evidence_ids": list(anchor.get("evidence_ids") or ()),
                    }
                ],
                "raw_detail_lines": [
                    {
                        "logical_page": int(line.get("page") or 0),
                        "source_page": int(line.get("source_page") or 0),
                        "text": str(line.get("text") or ""),
                        "bbox": list(line.get("bbox") or ()),
                        "evidence_ids": list(line.get("evidence_ids") or ()),
                    }
                    for line in detail
                ],
                "confidence": float(anchor.get("confidence") or 0.0),
            }
        reconstructed_identifier = _account_identifier_from_detail(detail)
        if reconstructed_identifier:
            skeleton["account_identifier"] = reconstructed_identifier
            skeleton["account_identifier_source"] = "canonical_anchor_table_row"
        skeletons.append(skeleton)
    return skeletons


def _account_stream_position(record: dict[str, Any]) -> tuple[int, float] | None:
    """Return the canonical logical-page position of an account observation."""
    page = int(record.get("page") or 0)
    bbox = record.get("bbox")
    if not page:
        for ref in record.get("source_refs") or ():
            if not isinstance(ref, dict):
                continue
            page = int(ref.get("logical_page") or 0)
            if not bbox:
                bbox = ref.get("bbox")
            if page:
                break
    if not page:
        return None
    try:
        top = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0.0
    except (TypeError, ValueError):
        top = 0.0
    return page, top


def _match_account_table_observations(
    skeletons: list[dict[str, Any]],
    table_accounts: list[dict[str, Any]],
) -> dict[int, int]:
    """Match table observations to printed anchors in canonical stream order.

    OCR table classification is deliberately not an identity key.  For
    example, a table headed ``账户授信额度`` can occur inside a type-R2 card
    and look like an R1 table in isolation.  The printed anchor and registered
    logical-page stream own identity; the table only supplies business cells.
    """
    matches: dict[int, int] = {}
    consumed_tables: set[int] = set()
    positioned_tables = sorted(
        (
            (position, table_index)
            for table_index, table in enumerate(table_accounts)
            if (position := _account_stream_position(table)) is not None
        ),
        key=lambda item: item[0],
    )
    positioned_skeletons = {
        index: position
        for index, skeleton in enumerate(skeletons)
        if (position := _account_stream_position(skeleton)) is not None
    }

    for table_position, table_index in positioned_tables:
        eligible: list[tuple[tuple[int, float], int]] = []
        for skeleton_index, skeleton_position in positioned_skeletons.items():
            same_page_precedes = (
                skeleton_position[0] == table_position[0]
                and skeleton_position[1] <= table_position[1] + 24.0
            )
            earlier_page = skeleton_position[0] < table_position[0]
            if same_page_precedes or earlier_page:
                eligible.append((skeleton_position, skeleton_index))
        if eligible:
            _position, skeleton_index = max(eligible, key=lambda item: item[0])
            # The immediately preceding printed anchor owns this canonical
            # stream segment.  If it already has a base-table observation,
            # this is a replay/secondary table, not evidence for an older
            # unmatched anchor.
            if skeleton_index in matches:
                continue
            matches[skeleton_index] = table_index
            consumed_tables.add(table_index)

    # Geometry can be unavailable in synthetic/native-table ParseResults.
    # Exact canonical keys are a safe secondary match, but never override a
    # stream-position match.
    for skeleton_index, skeleton in enumerate(skeletons):
        if skeleton_index in matches:
            continue
        key = (str(skeleton.get("account_type") or ""), int(skeleton.get("category_sequence") or 0))
        if not key[0] or key[1] <= 0:
            continue
        for table_index, table in enumerate(table_accounts):
            if table_index in consumed_tables:
                continue
            table_key = (str(table.get("account_type") or ""), int(table.get("category_sequence") or 0))
            if table_key == key:
                matches[skeleton_index] = table_index
                consumed_tables.add(table_index)
                break
    return matches


def _extract_accounts(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize one account population from anchors enriched by tables."""
    table_accounts, repayments, events = _extract_table_accounts(parse_result)
    skeletons = _account_anchor_skeletons(parse_result)
    if not skeletons:
        return table_accounts, repayments, events

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    table_matches = _match_account_table_observations(skeletons, table_accounts)
    emitted: list[dict[str, Any]] = []
    consumed: set[int] = set()
    account_id_remap: dict[str, str] = {}
    for skeleton_index, skeleton in enumerate(skeletons):
        table_index = table_matches.get(skeleton_index)
        table = table_accounts[table_index] if table_index is not None else None
        if table is not None:
            record = dict(table)
            # Printed anchors own population identity and canonical ordering;
            # the table contributes only decoded business cells.
            for field_name in (
                "account_id",
                "sequence",
                "category_sequence",
                "account_type",
                "account_family_quality",
                "page",
                "source_page",
                "bbox",
            ):
                record[field_name] = deepcopy(skeleton.get(field_name))
            if not record.get("account_identifier") and skeleton.get("account_identifier"):
                record["account_identifier"] = skeleton["account_identifier"]
                record["account_identifier_source"] = skeleton.get("account_identifier_source")
            record["source_refs"] = [
                *(table.get("source_refs") or ()),
                *(skeleton.get("source_refs") or ()),
            ]
            record["raw_detail_lines"] = list(skeleton.get("raw_detail_lines") or ())
            emitted.append(record)
            consumed.add(table_index)
            prior_account_id = str(table.get("account_id") or "")
            canonical_account_id = str(skeleton.get("account_id") or "")
            if prior_account_id and canonical_account_id:
                account_id_remap[prior_account_id] = canonical_account_id
            continue

        emitted.append(skeleton)
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_account_table_missing",
                message="A printed account anchor was present but its canonical account table was not recoverable.",
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                target_record_id=str(skeleton.get("account_id") or ""),
                observed_value={
                    "account_type": skeleton.get("account_type"),
                    "category_sequence": skeleton.get("category_sequence"),
                },
                source_refs=skeleton.get("source_refs") or (),
                reason_codes=("printed_account_anchor", "canonical_table_missing", "record_partial"),
            ),
        )

    anchored_types = {
        str(skeleton.get("account_type") or "")
        for skeleton in skeletons
        if skeleton.get("account_type")
    }
    for table_index, table in enumerate(table_accounts):
        if table_index in consumed:
            continue
        account_type = str(table.get("account_type") or "")
        structurally_missing_category = bool(account_type) and account_type not in anchored_types
        if structurally_missing_category:
            record = dict(table)
            record["sequence"] = len(emitted) + 1
            emitted.append(record)
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code=(
                    "candidate_b_account_category_anchor_missing"
                    if structurally_missing_category
                    else "candidate_b_unmatched_account_table_suppressed"
                ),
                message=(
                    "A canonical account category had table evidence but no recoverable printed anchor; "
                    "the table row was retained with uncertainty."
                    if structurally_missing_category
                    else "An unmatched table row duplicated an already anchored account category and was suppressed."
                ),
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                target_record_id=str(table.get("account_id") or ""),
                source_refs=table.get("source_refs") or (),
                reason_codes=(
                    "canonical_account_table",
                    "printed_anchor_missing",
                    "record_requires_review" if structurally_missing_category else "duplicate_candidate_suppressed",
                ),
            ),
        )

    for account in emitted:
        if account.get("account_family_quality") != "ambiguous_missing_variant":
            continue
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_account_family_variant_unresolved",
                message=(
                    "The printed revolving-loan family heading did not preserve its one/two variant; "
                    "the conservative R1 family was retained and explicitly marked uncertain."
                ),
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                target_record_id=str(account.get("account_id") or ""),
                field_name="account_type",
                observed_value="循环贷账户",
                candidate_value=account.get("account_type"),
                source_refs=account.get("source_refs") or (),
                reason_codes=("account_family_variant_missing", "normalized_value_requires_review"),
            ),
        )

    emitted_by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for account in emitted:
        account_type = str(account.get("account_type") or "")
        try:
            ordinal = int(account.get("category_sequence") or 0)
        except (TypeError, ValueError):
            ordinal = 0
        if account_type and ordinal > 0:
            emitted_by_family[account_type].append(account)
    for account_type, family_accounts in emitted_by_family.items():
        observed = sorted({int(account["category_sequence"]) for account in family_accounts})
        if not observed:
            continue
        # PBOC category ordinals start at one and are dense. Bound the gap
        # expansion so a single OCR-joined outlier cannot manufacture hundreds
        # of supposed missing records; the outlier is still reported below.
        credible_ceiling = max(12, len(observed) * 3)
        bounded_endpoint = min(max(observed), credible_ceiling)
        missing = sorted(set(range(1, bounded_endpoint + 1)) - set(observed))
        outliers = [ordinal for ordinal in observed if ordinal > credible_ceiling]
        if not missing and not outliers:
            continue
        refs = [
            dict(ref)
            for account in family_accounts
            for ref in account.get("source_refs") or ()
            if isinstance(ref, dict)
        ]
        record_issue(
            parse_result,
            make_issue(
                category="schema_incompleteness",
                issue_code="candidate_b_account_sequence_gap",
                message=(
                    "Printed account ordinals in one PBOC account family were not dense; no missing records "
                    "were invented and the ordinal ambiguity was retained for downstream review."
                ),
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                observed_value={
                    "account_type": account_type,
                    "observed_category_sequences": observed,
                },
                candidate_value={
                    "missing_category_sequences": missing,
                    "outlier_category_sequences": outliers,
                },
                source_refs=refs,
                reason_codes=(
                    "printed_category_ordinals_not_dense",
                    "missing_records_not_invented",
                    "business_population_uncertain",
                ),
            ),
        )

    account_identifiers = {
        str(account.get("account_id") or ""): account.get("account_identifier")
        for account in emitted
        if account.get("account_id")
    }
    for related_record in [*repayments, *events]:
        prior_account_id = str(related_record.get("account_id") or "")
        canonical_account_id = account_id_remap.get(prior_account_id)
        if not canonical_account_id:
            continue
        related_record["account_id"] = canonical_account_id
        if not related_record.get("account_identifier"):
            related_record["account_identifier"] = account_identifiers.get(canonical_account_id)
    return emitted, repayments, events


def _extract_credit_lines(parse_result: Any) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    records: list[dict[str, Any]] = []
    for candidate in PBOCPersonalDetailNativeParser(parse_result).records("credit_lines"):
        facts = candidate.fields
        identifier = _typed_identifier(_field(facts, "授信协议标识"))
        if not identifier:
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_identifier_unresolved",
                    message="A canonical credit-agreement card was observed but its agreement identifier was not safely extractable.",
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    field_name="account_identifier",
                    observed_value=_field(facts, "授信协议标识") or None,
                    source_refs=candidate.source_refs,
                    reason_codes=("canonical_credit_agreement_card", "typed_identifier_failed", "record_withheld"),
                ),
            )
            continue
        due_raw = _field(facts, "到期日期")
        currency = _currency(_field(facts, "币种")) or "CNY"
        printed_sequence_raw = _field(facts, "__printed_sequence")
        printed_sequence = int(printed_sequence_raw) if str(printed_sequence_raw).isdigit() else None
        raw_limit_identifier = _field(facts, "授信限额编号")
        limit_identifier = _typed_identifier(raw_limit_identifier)
        if raw_limit_identifier and not limit_identifier:
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_credit_limit_identifier_unresolved",
                    message="The credit-limit identifier cell contained multiple or invalid identifier tokens and was withheld.",
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=stable_record_id("credit_line", identifier),
                    field_name="limit_identifier",
                    observed_value=raw_limit_identifier,
                    source_refs=candidate.source_refs,
                    reason_codes=(
                        "canonical_credit_agreement_card",
                        "typed_identifier_failed",
                        "normalized_value_withheld",
                    ),
                ),
            )
        records.append(
            {
                "credit_line_id": stable_record_id("credit_line", identifier),
                "_printed_sequence": printed_sequence,
                "account_identifier": identifier,
                "institution": _clean(_field(facts, "管理机构")),
                "facility_type": _clean(_field(facts, "授信额度用途")),
                "effective_date": _date(_field(facts, "生效日期")),
                "due_date": _date(due_raw),
                "validity_type": "perpetual" if _compact(due_raw) == "长期" else "fixed_term",
                "total_limit": _number(_field(facts, "授信额度")),
                "used_limit": _number(_field(facts, "已用额度")),
                "limit_identifier": limit_identifier,
                "currency": currency,
                "account_currency": currency,
                "reporting_amount_currency": "CNY",
                "amount_unit": "yuan",
                "reporting_amount_unit": "yuan",
                "status": "active",
                "source": "candidate_b_credit_agreement_schema",
                "source_refs": list(candidate.source_refs),
                "confidence": candidate.confidence,
            }
        )
    return _dedupe_prefixed_identifiers(records, "account_identifier")


def _agreement_identifier_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _agreement_identifier_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    left_counts = Counter(left)
    right_counts = Counter(right)
    shared = sum((left_counts & right_counts).values())
    multiset = 2.0 * shared / max(1, len(left) + len(right))
    return max(sequence, multiset)


def _agreement_strong_field_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import normalize_institution_name

    matches = 0
    for field_name in (
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "used_limit",
        "currency",
    ):
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        if left_value in (None, "") or right_value in (None, ""):
            continue
        if field_name == "institution":
            equal = normalize_institution_name(str(left_value)) == normalize_institution_name(str(right_value))
        elif field_name in {"total_limit", "used_limit"}:
            equal = str(left_value).replace(",", "") == str(right_value).replace(",", "")
        else:
            equal = _compact(left_value) == _compact(right_value)
        matches += int(equal)
    return matches


def _agreement_source_pages(record: Mapping[str, Any]) -> set[int]:
    return {
        int(ref.get("logical_page") or ref.get("page") or 0)
        for ref in record.get("source_refs") or ()
        if isinstance(ref, dict) and (ref.get("logical_page") or ref.get("page"))
    }


def _agreement_printed_sequences_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_sequence = left.get("_printed_sequence")
    right_sequence = right.get("_printed_sequence")
    return (
        left_sequence in (None, "")
        or right_sequence in (None, "")
        or int(left_sequence) == int(right_sequence)
    )


def reconcile_candidate_b_credit_lines(
    parse_result: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the one-row-per-agreement schema constraint after correction."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    prototypes: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(records):
        record = dict(source)
        raw_identifier = _agreement_identifier_text(record.get("account_identifier"))
        identifier = _typed_identifier(record.get("account_identifier"))
        if identifier:
            record["account_identifier"] = identifier
            record["credit_line_id"] = stable_record_id("credit_line", identifier)
        identity = str(record.get("credit_line_id") or f"candidate_b_credit_line_row:{index + 1}")
        if identity not in groups and raw_identifier:
            compatible = [
                candidate_identity
                for candidate_identity, prototype in prototypes.items()
                if _agreement_identifier_similarity(
                    raw_identifier, _agreement_identifier_text(prototype.get("account_identifier"))
                )
                >= 0.90
                and _agreement_strong_field_matches(record, prototype) >= 3
                and _agreement_printed_sequences_compatible(record, prototype)
                and (
                    not identifier
                    or bool(_agreement_source_pages(record) & _agreement_source_pages(prototype))
                )
            ]
            if len(compatible) == 1:
                identity = compatible[0]
        if identity not in groups:
            groups[identity] = []
            order.append(identity)
            prototypes[identity] = record
        groups[identity].append(record)

    reconciled: list[dict[str, Any]] = []
    ignored_fields = {
        "source",
        "source_refs",
        "confidence",
        "credit_line_id",
        "sequence",
        "_printed_sequence",
    }
    for identity in order:
        observations = groups[identity]
        ranked = sorted(
            observations,
            key=lambda row: (
                sum(value not in (None, "", [], {}) for key, value in row.items() if key not in ignored_fields),
                float(row.get("confidence") or 0.0),
                len(str(row.get("account_identifier") or "")),
            ),
            reverse=True,
        )
        selected = dict(ranked[0])
        printed_sequences = {
            int(observation["_printed_sequence"])
            for observation in observations
            if observation.get("_printed_sequence") not in (None, "")
        }
        if len(printed_sequences) == 1:
            selected["sequence"] = next(iter(printed_sequences))
        selected.pop("_printed_sequence", None)
        conflicts: set[str] = set()
        merged_refs: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for observation in ranked:
            for ref in observation.get("source_refs") or ():
                if not isinstance(ref, dict):
                    continue
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    merged_refs.append(dict(ref))
            for key, value in observation.items():
                if key in ignored_fields or value in (None, "", [], {}):
                    continue
                retained = selected.get(key)
                if retained in (None, "", [], {}):
                    selected[key] = deepcopy(value)
                elif retained != value and not (
                    key == "account_identifier"
                    and _agreement_identifier_similarity(
                        _agreement_identifier_text(retained), _agreement_identifier_text(value)
                    )
                    >= 0.90
                ):
                    conflicts.add(key)
        if merged_refs:
            selected["source_refs"] = merged_refs
        reconciled.append(selected)
        if len(observations) > 1 and conflicts:
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_observation_conflict",
                    message=(
                        "Multiple corrected observations resolved to one canonical credit-agreement identity; "
                        "the most complete observation was retained and conflicting fields require review."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=identity,
                    observed_value={
                        "candidate_count": len(observations),
                        "conflicting_fields": sorted(conflicts),
                    },
                    source_refs=merged_refs,
                    reason_codes=(
                        "canonical_identity_collision",
                        "single_schema_record_retained",
                        "conflicting_fields_reported",
                    ),
                ),
            )
    sequence_counts = Counter(
        int(record["sequence"])
        for record in reconciled
        if record.get("sequence") not in (None, "")
    )
    for record in reconciled:
        sequence = record.get("sequence")
        reason_codes: tuple[str, ...]
        if sequence not in (None, "") and sequence_counts[int(sequence)] == 1:
            continue
        if sequence not in (None, ""):
            record.pop("sequence", None)
            reason_codes = (
                "printed_sequence_collision",
                "row_order_not_used",
                "normalized_value_withheld",
            )
        else:
            reason_codes = (
                "printed_sequence_unreadable",
                "row_order_not_used",
                "normalized_value_withheld",
            )
        record_issue(
            parse_result,
            make_issue(
                category="schema_incompleteness",
                issue_code="candidate_b_credit_agreement_sequence_unresolved",
                message="The printed credit-agreement ordinal was missing or non-unique; encounter order was not emitted as business data.",
                parser_stage="candidate_b_credit_agreement_schema",
                target_dataset="credit_lines",
                target_record_id=str(record.get("credit_line_id") or "") or None,
                field_name="sequence",
                observed_value=sequence,
                source_refs=record.get("source_refs") or (),
                reason_codes=reason_codes,
            ),
        )
    return reconciled


def _extract_liabilities(parse_result: Any) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    records: list[dict[str, Any]] = []
    for candidate in PBOCPersonalDetailNativeParser(parse_result).records("repayment_liability_records"):
        facts = candidate.fields
        contract_number = _typed_identifier(_field(facts, "保证合同编号"))
        related_id = _identifier(_field(facts, "主业务借款人证件号码"))
        responsibility_amount = _number(_field(facts, "还款责任金额"))
        if contract_number is None and not isinstance(responsibility_amount, int):
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_repayment_responsibility_identity_unresolved",
                    message="A canonical repayment-responsibility card was observed but neither its contract identifier nor responsibility amount was safely extractable.",
                    parser_stage="candidate_b_repayment_responsibility_schema",
                    target_dataset="repayment_liability_records",
                    observed_value={
                        "contract_number": _field(facts, "保证合同编号") or None,
                        "responsibility_amount": _field(facts, "还款责任金额") or None,
                    },
                    source_refs=candidate.source_refs,
                    reason_codes=("canonical_responsibility_card", "record_identity_unresolved", "record_withheld"),
                ),
            )
            continue
        identifier = contract_number or stable_record_id("liability_source", len(records) + 1)
        currency = _currency(_field(facts, "币种")) or "CNY"
        records.append(
            {
                "liability_id": stable_record_id("repayment_liability", identifier),
                "sequence": len(records) + 1,
                "open_date": _date(_field(facts, "开立日期")),
                "due_date": _date(_field(facts, "到期日期")),
                "related_party_name": _clean(_field(facts, "主业务借款人")),
                "related_party_id_type": _clean(_field(facts, "主业务借款人证件类型")),
                "related_party_id_number": related_id,
                "institution": _clean(_field(facts, "管理机构")),
                "business_type": _clean(_field(facts, "业务种类")),
                "responsibility_type": _clean(_field(facts, "责任人类型")),
                "responsibility_amount": responsibility_amount,
                "responsibility_amount_reported": True,
                "contract_number": contract_number,
                "snapshot_date": _date(_field(facts, "报告日期")),
                "balance": _number(_field(facts, "余额")),
                "five_tier_class": _clean(_field(facts, "五级分类")),
                "overdue_months_or_repayment_status": _clean(_field(facts, "逾期月数", "还款状态")),
                "currency": currency,
                "reporting_amount_currency": currency,
                "amount_unit": "yuan",
                "reporting_amount_unit": "yuan",
                "source": "candidate_b_repayment_responsibility_schema",
                "source_refs": list(candidate.source_refs),
                "confidence": candidate.confidence,
            }
        )
    deduped = _dedupe_liability_records(records)
    if len(deduped) < len(records):
        record_issue(
            parse_result,
            make_issue(
                category="schema_incompleteness",
                issue_code="candidate_b_duplicate_responsibility_observation_suppressed",
                message=(
                    "Two canonical observations matched the same repayment-responsibility card on at least "
                    "three independent business fields; the redundant observation was suppressed."
                ),
                severity="info",
                status="suppressed_redundant",
                parser_stage="candidate_b_repayment_responsibility_schema",
                target_dataset="repayment_liability_records",
                observed_value={"candidate_count": len(records), "emitted_count": len(deduped)},
                reason_codes=(
                    "single_schema_decoder",
                    "three_field_semantic_match",
                    "duplicate_observation_suppressed",
                ),
            ),
        )
    return deduped


def _normalize_inquiry_reason(value: str) -> str:
    text = str(value or "")
    for observed, canonical in _INQUIRY_REASON_REPAIRS.items():
        text = text.replace(observed, canonical)
    return text


def _inquiry_geometry_groups(lines: Any) -> list[list[dict[str, Any]]]:
    """Join OCR tokens that occupy one canonical inquiry-table row."""
    positioned: list[tuple[float, float, dict[str, Any]]] = []
    for index, line in enumerate(lines or ()):
        if not isinstance(line, dict):
            continue
        bbox = line.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                top, left = float(bbox[1]), float(bbox[0])
            except (TypeError, ValueError):
                top, left = float(index * 20), 0.0
        else:
            top, left = float(index * 20), 0.0
        positioned.append((top, left, line))
    positioned.sort(key=lambda item: (item[0], item[1]))

    groups: list[list[tuple[float, float, dict[str, Any]]]] = []
    for top, left, line in positioned:
        if groups and abs(top - sum(item[0] for item in groups[-1]) / len(groups[-1])) <= 3.5:
            groups[-1].append((top, left, line))
        else:
            groups.append([(top, left, line)])
    return [
        [line for _top, _left, line in sorted(group, key=lambda item: item[1])]
        for group in groups
    ]


def _canonical_inquiry_line_rows(parse_result: Any) -> list[dict[str, Any]]:
    """Reconstruct known inquiry-template rows from canonical line geometry."""
    evidence_loader = getattr(parse_result, "corrected_evidence_pages", None)
    if not callable(evidence_loader):
        return []
    rows: list[dict[str, Any]] = []
    last_sequence = {"institution": 0, "personal": 0}
    inferred_sequences: defaultdict[str, list[int]] = defaultdict(list)
    for page in evidence_loader():
        if str(page.get("canonical_template_id") or "") != "annotations_and_inquiries":
            continue
        for group in _inquiry_geometry_groups(page.get("lines") or ()):
            text = _normalize_inquiry_reason(
                " ".join(str(line.get("text") or line.get("content") or "").strip() for line in group)
            )
            date_match = re.search(r"20\d{2}[.,/-]\d{1,2}[.,/-]\d{1,2}", text)
            if date_match is None:
                continue
            reason = next(
                (candidate for candidate in _INQUIRY_REASONS if candidate in text[date_match.end() :]),
                "",
            )
            if not reason:
                continue
            reason_index = text.find(reason, date_match.end())
            institution = re.sub(
                r"^[^\u3400-\u9fffA-Za-z]+",
                "",
                text[date_match.end() : reason_index],
            ).strip()
            inquiry_type = "personal" if reason.startswith("本人查询") or institution == "本人" else "institution"
            if inquiry_type == "personal" and not institution:
                institution = "本人"
            if not institution:
                continue
            expected = last_sequence[inquiry_type] + 1
            sequence_tokens = re.findall(r"\d{1,4}", text[: date_match.start()])
            detected = int(sequence_tokens[-1]) if sequence_tokens else 0
            inferred_sequence = detected == 0
            corrected_sequence = detected > expected and str(detected).endswith(str(expected))
            sequence = expected if inferred_sequence or corrected_sequence else detected
            if sequence <= last_sequence[inquiry_type]:
                continue
            last_sequence[inquiry_type] = sequence
            if inferred_sequence:
                inferred_sequences[inquiry_type].append(sequence)
            inquiry_date = _date(date_match.group(0).replace(",", "."))
            if inquiry_date is None:
                continue
            boxes = [line.get("bbox") for line in group if isinstance(line.get("bbox"), list) and len(line["bbox"]) == 4]
            bbox = (
                [
                    min(float(box[0]) for box in boxes),
                    min(float(box[1]) for box in boxes),
                    max(float(box[2]) for box in boxes),
                    max(float(box[3]) for box in boxes),
                ]
                if boxes
                else []
            )
            source_ref = {
                "source": "candidate_b_canonical_inquiry_line",
                "logical_page": int(page.get("page") or 0),
                "source_page": int(page.get("source_page") or 0),
                "bbox": bbox,
                "evidence_ids": list(
                    dict.fromkeys(
                        str(evidence_id)
                        for line in group
                        for evidence_id in (line.get("evidence_ids") or ())
                        if evidence_id
                    )
                ),
            }
            row = {
                "inquiry_id": stable_record_id(
                    "credit_inquiry", inquiry_type, sequence, inquiry_date, institution, reason
                ),
                "sequence": sequence,
                "inquiry_date": inquiry_date,
                "institution": institution,
                "reason": reason,
                "source_reason": text[reason_index:].strip(),
                "query_channel": inquiry_type,
                "inquiry_type": inquiry_type,
                "source": "candidate_b_canonical_inquiry_line",
                "source_refs": [source_ref],
                "confidence": min(
                    min((float(line.get("confidence") or 0.0) for line in group), default=0.0),
                    0.8,
                ),
            }
            if corrected_sequence or inferred_sequence:
                row["extraction_status"] = "review"
                row["audit"] = {
                    "reason": (
                        "sequence_missing_inferred_by_template_row_order"
                        if inferred_sequence
                        else "sequence_prefix_noise_corrected_by_template_contract"
                    ),
                    "raw_sequence": detected or None,
                }
            rows.append(row)
    if inferred_sequences:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

        for inquiry_type, sequences in inferred_sequences.items():
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_inquiry_sequence_inferred_from_row_order",
                    message="One or more inquiry row numbers were unreadable and were inferred from canonical table order.",
                    parser_stage="candidate_b_inquiry_schema",
                    target_dataset="inquiry_records",
                    field_name="sequence",
                    observed_value={"inquiry_type": inquiry_type, "missing_ocr_sequence": True},
                    candidate_value={"inferred_sequences": sequences},
                    reason_codes=("canonical_four_column_table", "contiguous_row_order", "sequence_requires_review"),
                ),
            )
    return rows


def _normalized_inquiry_field(field_name: str, value: Any) -> str:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        normalize_institution_name,
        normalize_role_candidate,
    )

    if field_name == "institution":
        normalized = normalize_institution_name(str(value or ""))
        compact = re.sub(r"\s+", "", normalized)
        # PBOC personal-query rows use 本人 as the institution.  Permit only a
        # tiny amount of surrounding OCR debris so bank names cannot collapse
        # into this special value.
        if "本人" in compact and len(compact.replace("本人", "")) <= 2:
            return "本人"
        return normalized

    if field_name == "reason":
        normalized = normalize_role_candidate(value, "inquiry_reason")
        compact = re.sub(r"\s+", "", normalized)
        for canonical in (
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
        ):
            if canonical in compact:
                return canonical
        return normalized

    role = {
        "inquiry_date": "date",
    }[field_name]
    return normalize_role_candidate(value, role)


def _inquiry_business_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if any(
        _normalized_inquiry_field(field, left.get(field))
        != _normalized_inquiry_field(field, right.get(field))
        for field in ("inquiry_date", "reason")
    ):
        return False
    left_institution = re.sub(r"\s+", "", _normalized_inquiry_field("institution", left.get("institution")))
    right_institution = re.sub(r"\s+", "", _normalized_inquiry_field("institution", right.get("institution")))
    if left_institution == right_institution:
        return True
    return (
        abs(len(left_institution) - len(right_institution)) <= 2
        and bool(left_institution)
        and bool(right_institution)
        and (left_institution in right_institution or right_institution in left_institution)
    )


def _inquiry_observation_score(record: Mapping[str, Any]) -> tuple[int, int, float]:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import role_candidate_is_valid

    valid = sum(
        role_candidate_is_valid(record.get(field), role)
        for field, role in (
            ("inquiry_date", "date"),
            ("institution", "institution_name"),
            ("reason", "inquiry_reason"),
        )
    )
    populated = sum(record.get(field) not in (None, "") for field in ("inquiry_date", "institution", "reason"))
    return valid, populated, float(record.get("confidence") or 0.0)


def _extract_inquiries(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    active_canonical_table = False
    for page in getattr(parse_result, "pages", None) or []:
        page_is_canonical_inquiry = (
            str(getattr(page, "canonical_template_id", "") or "") == "annotations_and_inquiries"
        )
        page_had_inquiry_table = False
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            header = _nonempty(rows[0])
            compact_header = _compact("".join(header))
            has_header = all(marker in compact_header for marker in ("查询日期", "查询机构", "查询原因"))
            is_continuation = (
                not has_header
                and page_is_canonical_inquiry
                and active_canonical_table
                and bool(_nonempty(rows[0]))
                and re.fullmatch(r"\d{1,4}", _nonempty(rows[0])[0]) is not None
            )
            if not has_header and not is_continuation:
                continue
            page_had_inquiry_table = True
            active_canonical_table = True
            start = 1 if has_header else 0
            for row_index, row in enumerate(rows[start:], start=start):
                cells = [str(value or "").strip() for value in row]
                values = _nonempty(cells)
                if not values or re.fullmatch(r"\d{1,4}", values[0]) is None:
                    continue
                sequence = int(values[0])
                date_cell = cells[1] if len(cells) > 1 else (values[1] if len(values) > 1 else "")
                date_match = re.search(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}", date_cell)
                if date_match is None:
                    continue
                inquiry_date = _date(date_match.group(0))
                institution = cells[2] if len(cells) > 2 else ""
                source_reason = cells[3] if len(cells) > 3 else ""
                if not institution:
                    # A border miss can merge the date and institution while
                    # leaving the reason in its correct final column.  The
                    # date contract gives an unambiguous, value-agnostic split.
                    institution = date_cell[date_match.end() :].strip(" :：|/")
                if (not institution or not source_reason) and len(values) >= 4:
                    institution = institution or values[2]
                    source_reason = source_reason or values[3]
                if not institution or not source_reason or inquiry_date is None:
                    continue
                inquiry_type = (
                    "personal" if institution == "本人" or source_reason.startswith("本人查询") else "institution"
                )
                records.append(
                    {
                        "inquiry_id": stable_record_id(
                            "credit_inquiry", inquiry_type, sequence, inquiry_date, institution, source_reason
                        ),
                        "sequence": sequence,
                        "inquiry_date": inquiry_date,
                        "institution": institution,
                        "reason": source_reason,
                        "source_reason": source_reason,
                        "query_channel": "personal" if inquiry_type == "personal" else "institution",
                        "inquiry_type": inquiry_type,
                        "source": "native_detail_inquiry_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "confidence": 1.0,
                    }
                )
        if not page_is_canonical_inquiry and not page_had_inquiry_table:
            active_canonical_table = False
    records.extend(_canonical_inquiry_line_rows(parse_result))
    best: dict[tuple[str, int], dict[str, Any]] = {}
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    for record in records:
        key = (str(record.get("inquiry_type") or ""), int(record.get("sequence") or 0))
        current = best.get(key)
        if current is None:
            best[key] = record
            continue
        current_values = tuple(current.get(field) for field in ("inquiry_date", "institution", "reason"))
        candidate_values = tuple(record.get(field) for field in ("inquiry_date", "institution", "reason"))
        current_score = _inquiry_observation_score(current)
        candidate_score = _inquiry_observation_score(record)
        equivalent = _inquiry_business_equivalent(current, record)
        if not equivalent and current_score[:2] == candidate_score[:2]:
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_inquiry_row_conflict",
                    message="Canonical table and line observations disagree for the same printed inquiry row.",
                    parser_stage="candidate_b_inquiry_schema",
                    target_dataset="inquiry_records",
                    target_record_id=str(current.get("inquiry_id") or record.get("inquiry_id") or ""),
                    observed_value={"table_or_first": current_values, "line_or_second": candidate_values},
                    source_refs=[*(current.get("source_refs") or ()), *(record.get("source_refs") or ())],
                    reason_codes=("canonical_observation_conflict", "requires_review"),
                ),
            )
        if candidate_score > current_score:
            best[key] = record
    # Full-page OCR occasionally prefixes a row number with a neighbouring
    # watermark digit (89 -> 789).  If the suffix row already exists in the
    # same canonical population, the prefixed observation is redundant rather
    # than a new sequence endpoint.
    for key, record in list(best.items()):
        inquiry_type, sequence = key
        if sequence < 100:
            continue
        suffix = sequence % 100
        suffix_key = (inquiry_type, suffix)
        if suffix <= 0 or suffix_key not in best:
            continue
        suffix_record = best[suffix_key]
        if record.get("inquiry_date") != suffix_record.get("inquiry_date"):
            continue
        best.pop(key, None)
        record_issue(
            parse_result,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="candidate_b_inquiry_sequence_prefix_suppressed",
                message="A prefixed OCR row number duplicated an existing canonical inquiry row.",
                severity="info",
                status="resolved",
                parser_stage="candidate_b_inquiry_schema",
                target_dataset="inquiry_records",
                target_record_id=str(record.get("inquiry_id") or ""),
                observed_value={"raw_sequence": sequence},
                candidate_value={"canonical_sequence": suffix},
                source_refs=record.get("source_refs") or (),
                reason_codes=("sequence_prefix_noise", "canonical_duplicate_suppressed"),
            ),
        )

    ordered = sorted(best.values(), key=lambda row: (str(row.get("inquiry_type") or ""), int(row.get("sequence") or 0)))
    for inquiry_type in sorted({str(row.get("inquiry_type") or "unknown") for row in ordered}):
        sequences = {
            int(row.get("sequence") or 0)
            for row in ordered
            if str(row.get("inquiry_type") or "unknown") == inquiry_type and int(row.get("sequence") or 0) > 0
        }
        missing = sorted(set(range(1, max(sequences) + 1)) - sequences) if sequences else []
        if not missing:
            continue
        record_issue(
            parse_result,
            make_issue(
                category="page_continuation",
                issue_code="canonical_inquiry_sequence_gap",
                message="Canonical inquiry rows contain a printed sequence gap; no missing row was invented.",
                parser_stage="candidate_b_inquiry_schema",
                target_dataset="inquiry_records",
                observed_value={
                    "inquiry_type": inquiry_type,
                    "observed_row_count": len(sequences),
                    "observed_sequences": sorted(sequences),
                },
                candidate_value={"source_sequence_endpoint": max(sequences), "missing_sequences": missing},
                reason_codes=("canonical_sequence_not_contiguous", "missing_row_not_invented", "dataset_incomplete"),
            ),
        )
    return ordered


def _extract_public_records(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            compact = _compact(" ".join(cell for row in rows for cell in row))
            page_number = int(getattr(page, "page_number", 0) or 0)
            typed_rows: list[tuple[str, str, dict[str, Any]]] = []
            if "主管税务机关" in compact and "欠税总额" in compact:
                for row in rows[1:]:
                    values = _nonempty(row)
                    if len(values) >= 4 and values[0].isdigit():
                        typed_rows.append(
                            (
                                "tax_arrears",
                                values[1],
                                {"arrears_amount": _number(values[2]), "statistics_date": _date(values[3])},
                            )
                        )
            elif "立案法院" in compact and "判决/调解结果" in compact:
                base: dict[str, dict[str, Any]] = {}
                detail_mode = False
                for row in rows[1:]:
                    values = _nonempty(row)
                    if values and values[0] == "编号":
                        detail_mode = True
                        continue
                    if len(values) < 2 or not values[0].isdigit():
                        continue
                    if not detail_mode and len(values) >= 5:
                        base[values[0]] = {
                            "authority": values[1],
                            "cause": None if values[2] == "--" else values[2],
                            "filing_date": _date(values[3]),
                            "closure_method": values[4],
                        }
                    elif detail_mode and len(values) >= 5:
                        item = base.setdefault(values[0], {})
                        item.update(
                            {
                                "judgment_result": values[1],
                                "effective_date": _date(values[2]),
                                "claim_subject": values[3],
                                "claim_amount": _number(values[4]),
                            }
                        )
                typed_rows.extend(("civil_judgment", item.get("authority", ""), item) for item in base.values())
            elif "执行法院" in compact and "申请执行标的" in compact:
                base = {}
                detail_mode = False
                for row in rows[1:]:
                    values = _nonempty(row)
                    if values and values[0] == "编号":
                        detail_mode = True
                        continue
                    if len(values) < 2 or not values[0].isdigit():
                        continue
                    if not detail_mode and len(values) >= 5:
                        base[values[0]] = {
                            "authority": values[1],
                            "cause": None if values[2] == "--" else values[2],
                            "filing_date": _date(values[3]),
                            "closure_method": values[4],
                        }
                    elif detail_mode and len(values) >= 7:
                        item = base.setdefault(values[0], {})
                        item.update(
                            {
                                "case_status": values[1],
                                "closure_date": _date(values[2]),
                                "requested_subject": values[3],
                                "requested_amount": _number(values[4]),
                                "executed_subject": values[5],
                                "executed_amount": _number(values[6]),
                            }
                        )
                typed_rows.extend(("enforcement", item.get("authority", ""), item) for item in base.values())
            elif "处罚机构" in compact and "处罚内容" in compact:
                for row in rows[1:]:
                    values = _nonempty(row)
                    if len(values) >= 7 and values[0].isdigit():
                        typed_rows.append(
                            (
                                "administrative_penalty",
                                values[1],
                                {
                                    "penalty_content": values[2],
                                    "penalty_amount": _number(values[3]),
                                    "effective_date": _date(values[4]),
                                    "end_date": _date(values[5]),
                                    "administrative_review_result": None if values[6] == "--" else values[6],
                                },
                            )
                        )
            elif "参缴地" in compact and "月缴存额" in compact:
                values = _nonempty(rows[1]) if len(rows) > 1 else []
                unit_values = _nonempty(rows[3]) if len(rows) > 3 else []
                if len(values) >= 8:
                    typed_rows.append(
                        (
                            "housing_fund",
                            unit_values[0] if unit_values else "",
                            {
                                "contribution_location": values[0],
                                "participation_date": _date(values[1]),
                                "first_contribution_month": _date(values[2]),
                                "paid_through_month": _date(values[3]),
                                "payment_status": values[4],
                                "monthly_contribution": _number(values[5]),
                                "personal_contribution_ratio": values[6],
                                "employer_contribution_ratio": values[7],
                                "employer": unit_values[0] if unit_values else None,
                                "information_updated_month": _date(unit_values[1]) if len(unit_values) > 1 else None,
                            },
                        )
                    )
            elif "执业资格名称" in compact and "颁发机构" in compact:
                for row in rows[1:]:
                    values = _nonempty(row)
                    if len(values) >= 8 and values[0].isdigit():
                        typed_rows.append(
                            (
                                "professional_qualification",
                                values[6],
                                {
                                    "qualification_name": values[1],
                                    "level": values[2],
                                    "obtained_date": _date(values[3]),
                                    "expiry_date": _date(values[4]),
                                    "revocation_date": _date(values[5]),
                                    "issuing_authority": values[6],
                                    "authority_location": values[7],
                                },
                            )
                        )
            elif "奖励机构" in compact and "奖励内容" in compact:
                for row in rows[1:]:
                    values = _nonempty(row)
                    if len(values) >= 5 and values[0].isdigit():
                        typed_rows.append(
                            (
                                "award",
                                values[1],
                                {
                                    "award_content": values[2],
                                    "effective_date": _date(values[3]),
                                    "end_date": _date(values[4]),
                                },
                            )
                        )
            for record_type, authority, content in typed_rows:
                sequence = 1 + sum(record.get("record_type") == record_type for record in records)
                records.append(
                    {
                        "public_record_id": stable_record_id(
                            "public_record", record_type, sequence, authority, page_number
                        ),
                        "sequence": sequence,
                        "record_type": record_type,
                        "authority": authority,
                        "start_date": content.get("filing_date") or content.get("effective_date"),
                        "end_date": content.get("closure_date") or content.get("end_date"),
                        # The shared canonical assembler treats dictionaries as
                        # value wrappers. Keep the personal-detail payload as a
                        # deterministic JSON value so no type-specific fields
                        # are discarded before community serialization.
                        "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
                        "source": "native_detail_public_table",
                        "source_refs": [_source_ref(page, table)],
                        "confidence": 1.0,
                    }
                )
    return records


def _canonical_header_slots(
    row: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]]
) -> dict[str, int]:
    """Map semantic roles to distinct physical columns in a printed header."""
    candidates: dict[str, int] = {}
    by_column: defaultdict[int, list[str]] = defaultdict(list)
    for role, labels in aliases.items():
        for column, cell in enumerate(row):
            text = _compact(cell)
            if text and any(_compact(label) in text for label in labels):
                candidates[role] = column
                by_column[column].append(role)
                break
    # A single OCR blob containing several labels is not a column graph.
    duplicate_columns = {column for column, roles in by_column.items() if len(roles) > 1}
    return {role: column for role, column in candidates.items() if column not in duplicate_columns}


def _slot_value(row: tuple[str, ...], slots: Mapping[str, int], role: str) -> str:
    column = slots.get(role)
    return _clean(row[column]) if column is not None and column < len(row) else ""


def _sequence_value(row: tuple[str, ...], slots: Mapping[str, int]) -> int | None:
    value = _slot_value(row, slots, "sequence")
    match = re.fullmatch(r"\D*(\d{1,3})\D*", value)
    return int(match.group(1)) if match else None


def _report_required_row_failure(
    parse_result: Any,
    *,
    issue_code: str,
    dataset: str,
    sequence: int,
    field_name: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code=issue_code,
            message="A canonical table row was observed but a required business cell could not be assigned to its printed column.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            target_record_id=f"{dataset}:{sequence}",
            field_name=field_name,
            observed_value={"sequence": sequence, "physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("canonical_column_graph", "required_cell_unresolved", "normalized_value_withheld"),
        ),
    )


def _report_header_graph_failure(
    parse_result: Any,
    *,
    dataset: str,
    page: Any,
    table: Any,
    row_index: int,
    row: tuple[str, ...],
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_canonical_header_graph_unresolved",
            message="Canonical header labels were observed, but OCR did not preserve distinct physical columns.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            observed_value={"physical_header_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("merged_header_cells", "column_roles_unresolved", "page_ocr_eligible"),
        ),
    )


def _report_unkeyed_fragment(
    parse_result: Any,
    *,
    dataset: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="page_continuation",
            issue_code="candidate_b_continuation_sequence_unresolved",
            message="A continued provider fragment lacked a trustworthy printed sequence and was not attached by row order.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            field_name="sequence",
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("continuation_fragment", "printed_sequence_unreadable", "row_order_not_used"),
        ),
    )


def _extract_residence_records(parse_result: Any) -> list[dict[str, Any]]:
    aliases = {
        "sequence": ("编号",),
        "address": ("居住地址",),
        "residential_phone": ("住宅电话",),
        "residence_status": ("居住状况",),
        "information_updated_date": ("信息更新日期",),
    }
    records: dict[int, dict[str, Any]] = {}
    providers: dict[int, tuple[str, dict[str, Any]]] = {}
    active_slots: dict[str, int] = {}
    mode = ""
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            table_id = str(getattr(table, "table_id", "") or "")
            table_text = _compact(" ".join(cell for row in rows for cell in row))
            continues = bool(
                previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            residence_schema_markers = ("居住地址", "住宅电话", "居住状况")
            other_section_markers = ("学历", "学位", "电子邮箱", "通讯地址", "户籍地址", "工作单位", "职业")
            if (
                mode in {"residence", "provider"}
                and not any(marker in table_text for marker in residence_schema_markers)
                and any(marker in table_text for marker in other_section_markers)
            ):
                continues = False
            if not continues:
                active_slots = {}
                mode = ""
            if not continues and not any(marker in table_text for marker in residence_schema_markers):
                previous_table_id = table_id or previous_table_id
                continue
            if continues and mode == "residence" and rows:
                nonempty_columns = [
                    [index for index, value in enumerate(row) if _clean(value)] for row in rows
                ]
                if nonempty_columns and all(len(columns) == 2 and columns[0] == 0 for columns in nonempty_columns):
                    active_slots = {"sequence": 0, "data_provider": nonempty_columns[0][1]}
                    mode = "provider"
            for row_index, row in enumerate(rows):
                residence_slots = _canonical_header_slots(row, aliases)
                compact_header = _compact("".join(row))
                if all(marker in compact_header for marker in ("编号", "居住地址", "信息更新日期")) and not {
                    "sequence",
                    "address",
                    "information_updated_date",
                } <= set(residence_slots):
                    _report_header_graph_failure(
                        parse_result,
                        dataset="residence_records",
                        page=page,
                        table=table,
                        row_index=row_index,
                        row=row,
                    )
                    active_slots = {}
                    mode = ""
                    continue
                if {"sequence", "address", "information_updated_date"} <= set(residence_slots):
                    active_slots = residence_slots
                    mode = "residence"
                    continue
                if "数据发生机构名称" in _compact("".join(row)) and mode in {"residence", "provider"}:
                    provider_slots = _canonical_header_slots(
                        row, {"sequence": ("编号",), "data_provider": ("数据发生机构名称",)}
                    )
                    active_slots = provider_slots
                    mode = "provider"
                    continue
                if not active_slots or mode not in {"residence", "provider"}:
                    continue
                sequence = _sequence_value(row, active_slots)
                if sequence is None and mode == "provider":
                    first = _clean(row[0] if row else "")
                    sequence = int(first) if first.isdigit() else None
                if sequence is None:
                    nonempty = _nonempty(row)
                    if (
                        mode == "provider"
                        and continues
                        and 0 < len(nonempty) <= 2
                        and any(
                            re.search(r"(?:银行|信用社|有限公司|管理中心|征信中心)", value)
                            for value in nonempty
                        )
                    ):
                        _report_unkeyed_fragment(
                            parse_result,
                            dataset="residence_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    continue
                ref = _source_ref(page, table, row=row_index)
                if mode == "provider":
                    provider = _slot_value(row, active_slots, "data_provider")
                    if not provider or provider.isdigit():
                        provider = next(
                            (_clean(value) for index, value in enumerate(row) if index > 0 and _clean(value)), ""
                        )
                    if provider:
                        providers[sequence] = (provider, ref)
                    continue
                address = _slot_value(row, active_slots, "address")
                updated_raw = _slot_value(row, active_slots, "information_updated_date")
                updated = _date(updated_raw)
                record = records.setdefault(
                    sequence,
                    {
                        "sequence": sequence,
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_residence_table",
                        "source_refs": [],
                        "confidence": 1.0,
                    },
                )
                record["source_refs"].append(ref)
                if address and address != "--":
                    record["address"] = address
                else:
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_residence_required_cell_unresolved",
                        dataset="residence_records",
                        sequence=sequence,
                        field_name="address",
                        row=row,
                        page=page,
                        table=table,
                        row_index=row_index,
                    )
                phone = _slot_value(row, active_slots, "residential_phone")
                status = _slot_value(row, active_slots, "residence_status")
                if phone and phone != "--":
                    record["residential_phone"] = phone
                if status and status != "--":
                    record["residence_status"] = status
                if updated and updated != "--":
                    record["information_updated_date"] = updated
            previous_table_id = table_id or previous_table_id
    for sequence, record in records.items():
        if sequence in providers:
            record["data_provider"] = providers[sequence][0]
            record["source_refs"].append(providers[sequence][1])
        residence_id = stable_record_id(
            "credit_residence", sequence, record.get("address"), record.get("information_updated_date")
        )
        record["record_id"] = residence_id
        record["residence_record_id"] = residence_id
    return [records[key] for key in sorted(records)]


_EMPLOYER_TYPES = (
    "国有企业",
    "集体企业",
    "私营企业",
    "外资企业",
    "事业单位",
    "国家机关",
    "个体经营",
    "其他",
)


def _split_employment_basic_row(row: tuple[str, ...]) -> dict[str, Any]:
    """Recover a basic employment row even when OCR merged its columns."""
    cells = [_clean(value) for value in row]
    positional_phone = next(
        (value for value in reversed(cells[4:]) if len(re.sub(r"\D", "", value)) >= 7),
        "",
    )
    if len(cells) >= 5 and cells[1] and (cells[2] or positional_phone):
        # Preserve a table that already has positional columns. Applying the
        # collapsed-row heuristic here can detach an area-code prefix from the
        # phone and shift every employment-detail field that follows it.
        return {
            "employer": cells[1],
            **({"employer_type": cells[2]} if cells[2] and cells[2] != "--" else {}),
            **({"employer_address": cells[3]} if cells[3] and cells[3] != "--" else {}),
            **({"employer_phone": positional_phone} if positional_phone else {}),
        }

    text = re.sub(r"\s+", " ", " ".join(_clean(value) for value in row[1:] if _clean(value))).strip()
    parsed: dict[str, Any] = {}
    phone_match = re.search(r"(?<!\d)(\d{7,16})(?!\d)\s*$", text)
    if phone_match:
        parsed["employer_phone"] = phone_match.group(1)
        text = text[: phone_match.start()].strip()

    employer_type = next((value for value in _EMPLOYER_TYPES if value in text), "")
    if employer_type:
        employer, address = text.split(employer_type, 1)
        parsed["employer"] = employer.strip()
        parsed["employer_type"] = employer_type
        if address.strip() and address.strip() != "--":
            parsed["employer_address"] = address.strip()
        return parsed

    # Most legal names have a reliable suffix even when every table column was
    # collapsed into one OCR cell. Keep uncertain trailing text as the address
    # instead of silently appending it to the employer name.
    legal = re.match(
        r"(.{2,60}?(?:有限责任公司|股份有限公司|有限公司|工业公司|学校|公司))(?:\s+|(?=[省市县区镇路街]))?(.*)$", text
    )
    if legal:
        parsed["employer"] = legal.group(1).strip()
        trailing = legal.group(2).strip()
        if trailing and trailing != "--":
            parsed["employer_address"] = trailing
    elif text and text != "--":
        parsed["employer"] = text
    return parsed


def _extract_employment_records(parse_result: Any) -> list[dict[str, Any]]:
    basic_aliases = {
        "sequence": ("编号",),
        "employer": ("工作单位",),
        "employer_type": ("单位性质",),
        "employer_address": ("单位地址",),
        "employer_phone": ("单位电话",),
    }
    detail_aliases = {
        "sequence": ("编号",),
        "occupation": ("职业",),
        "industry": ("行业",),
        "position": ("职务",),
        "professional_title": ("职称",),
        "entry_year": ("进入本单位年份",),
        "information_updated_date": ("信息更新日期",),
    }
    records: dict[int, dict[str, Any]] = {}
    active_slots: dict[str, int] = {}
    mode = ""
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            table_id = str(getattr(table, "table_id", "") or "")
            continues = bool(
                previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            if not continues:
                active_slots = {}
                mode = ""
            table_text = _compact(" ".join(cell for row in rows for cell in row))
            table_has_employment_schema = any(
                marker in table_text for marker in ("工作单位", "单位性质", "单位电话", "职业", "职务", "职称")
            )
            if not table_has_employment_schema and not continues:
                previous_table_id = table_id or previous_table_id
                continue
            if continues and mode in {"basic", "detail"} and rows:
                nonempty_columns = [
                    [index for index, value in enumerate(row) if _clean(value)] for row in rows
                ]
                if nonempty_columns and all(len(columns) == 2 and columns[0] == 0 for columns in nonempty_columns):
                    active_slots = {"sequence": 0, "data_provider": nonempty_columns[0][1]}
                    mode = "provider"
            for row_index, row in enumerate(rows):
                basic_slots = _canonical_header_slots(row, basic_aliases)
                detail_slots = _canonical_header_slots(row, detail_aliases)
                compact_header = _compact("".join(row))
                broken_basic = all(
                    marker in compact_header for marker in ("编号", "工作单位", "单位性质", "单位电话")
                ) and not {"sequence", "employer", "employer_type", "employer_phone"} <= set(basic_slots)
                broken_detail = all(
                    marker in compact_header for marker in ("编号", "职业", "行业", "职务", "职称")
                ) and not {"sequence", "occupation", "industry", "position", "professional_title"} <= set(
                    detail_slots
                )
                if broken_basic or broken_detail:
                    _report_header_graph_failure(
                        parse_result,
                        dataset="employment_records",
                        page=page,
                        table=table,
                        row_index=row_index,
                        row=row,
                    )
                    active_slots = {}
                    mode = ""
                    continue
                if {"sequence", "employer", "employer_type", "employer_phone"} <= set(basic_slots):
                    active_slots = basic_slots
                    mode = "basic"
                    continue
                if {"sequence", "occupation", "industry", "position", "professional_title"} <= set(detail_slots):
                    active_slots = detail_slots
                    mode = "detail"
                    continue
                if "数据发生机构名称" in _compact("".join(row)):
                    active_slots = _canonical_header_slots(
                        row, {"sequence": ("编号",), "data_provider": ("数据发生机构名称",)}
                    )
                    mode = "provider"
                    continue
                if not active_slots:
                    continue
                sequence = _sequence_value(row, active_slots)
                if sequence is None and mode == "provider":
                    first = _clean(row[0] if row else "")
                    sequence = int(first) if first.isdigit() else None
                if sequence is None:
                    if mode == "provider" and _nonempty(row):
                        _report_unkeyed_fragment(
                            parse_result,
                            dataset="employment_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    continue
                record = records.setdefault(
                    sequence,
                    {
                        "sequence": sequence,
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_employment_table",
                        "source_refs": [],
                        "confidence": 1.0,
                    },
                )
                record["source_refs"].append(_source_ref(page, table, row=row_index))
                if mode == "provider":
                    provider = _slot_value(row, active_slots, "data_provider")
                    if not provider or provider.isdigit():
                        provider = next(
                            (_clean(value) for index, value in enumerate(row) if index > 0 and _clean(value)), ""
                        )
                    if provider:
                        record["data_provider"] = provider
                    continue
                roles = basic_aliases if mode == "basic" else detail_aliases
                for role in roles:
                    if role == "sequence":
                        continue
                    raw = _slot_value(row, active_slots, role)
                    if not raw or raw == "--":
                        continue
                    if role == "entry_year":
                        match = re.fullmatch(r"\D*((?:19|20)\d{2})\D*", raw)
                        if match:
                            record[role] = int(match.group(1))
                    elif role == "information_updated_date":
                        converted = _date(raw)
                        if converted:
                            record[role] = converted
                    else:
                        record[role] = raw
                required_role = "employer" if mode == "basic" else "occupation"
                if not record.get(required_role):
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_employment_required_cell_unresolved",
                        dataset="employment_records",
                        sequence=sequence,
                        field_name=required_role,
                        row=row,
                        page=page,
                        table=table,
                        row_index=row_index,
                    )
            previous_table_id = table_id or previous_table_id
    for sequence, record in records.items():
        record["employment_record_id"] = stable_record_id(
            "credit_employment",
            sequence,
            {key: value for key, value in record.items() if key not in {"source_refs", "confidence"}},
        )
    return [records[key] for key in sorted(records)]


def _credible_sequence_endpoint(values: set[int]) -> tuple[int | None, list[int]]:
    """Return a dense sequence endpoint and isolate implausible OCR outliers."""
    if not values:
        return None, []
    # Printed ordinals are dense within a PBOC account family. Permit a small
    # number of missing observations, but never let one account identifier or
    # OCR-joined number inflate the expected row count by dozens of records.
    ceiling = max(3, len(values) + max(2, len(values) // 4))
    credible = {value for value in values if 1 <= value <= ceiling}
    outliers = sorted(values - credible)
    return (max(credible) if credible else None), outliers


def _source_completeness_ledger(parse_result: Any) -> dict[str, Any]:
    """Count printed sequence evidence independently of emitted records."""
    sequences: dict[str, set[int]] = {
        "residence_records": set(),
        "employment_records": set(),
    }
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows for cell in row))
            targets: list[str] = []
            if any(marker in compact for marker in ("居住地址", "住宅电话", "居住状况")):
                targets.append("residence_records")
            if any(marker in compact for marker in ("工作单位", "单位性质", "进入本单位年份")) or all(
                marker in compact for marker in ("职业", "行业", "职务", "职称")
            ):
                targets.append("employment_records")
            for row in rows:
                first = _clean(row[0] if row else "")
                numbers = [int(value) for value in re.findall(r"\d{1,3}", first)]
                sequence = numbers[0] if numbers and len(set(numbers)) == 1 else None
                if sequence is not None and 1 <= sequence <= 20:
                    for target in targets:
                        sequences[target].add(sequence)

    # Account numbers restart for each PBOC account family, so count the
    # maximum printed endpoint within each observed family instead of taking
    # one document-wide maximum.
    family = ""
    family_sequences: defaultdict[str, set[int]] = defaultdict(set)
    loader = getattr(parse_result, "corrected_evidence_pages", None)
    pages = loader() if callable(loader) else []
    for page in pages or []:
        for line in page.get("lines") or []:
            if not isinstance(line, dict):
                continue
            text = _compact(line.get("text") or line.get("content") or "")
            heading = _account_family_from_heading(text)
            if heading is not None:
                family = heading[0]
            match = re.match(r"^(?:账户|业务)[（(]?(\d{1,3})(?:[）)]|\D|$)", text)
            if match and family:
                family_sequences[family].add(int(match.group(1)))

    endpoints: dict[str, int] = {}
    sequence_outliers: dict[str, list[int]] = {}
    for account_family, values in family_sequences.items():
        endpoint, outliers = _credible_sequence_endpoint(values)
        if endpoint is not None:
            endpoints[account_family] = endpoint
        if outliers:
            sequence_outliers[account_family] = outliers

    return {
        "sequence_endpoints": {
            name: max(values)
            for name, values in sequences.items()
            if values
        },
        **({"credit_accounts": sum(endpoints.values())} if endpoints else {}),
        **({"account_family_endpoints": dict(endpoints)} if endpoints else {}),
        **({"account_family_sequence_outliers": sequence_outliers} if sequence_outliers else {}),
    }


def _extract_profile_detail_records(parse_result: Any) -> dict[str, list[dict[str, Any]]]:
    mobile_aliases = {
        "sequence": ("编号",),
        "mobile_phone": ("手机号码",),
        "information_updated_date": ("信息更新日期",),
        "data_provider": ("数据发生机构名称",),
    }
    spouse_aliases = {
        "name": ("姓名",),
        "document_type": ("证件类型",),
        "document_number": ("证件号码",),
        "employer": ("工作单位",),
        "phone": ("联系电话",),
    }
    mobile_records: dict[tuple[int, str, str], dict[str, Any]] = {}
    spouse_records: dict[tuple[str, str], dict[str, Any]] = {}
    active_slots: dict[str, int] = {}
    mode = ""
    last_spouse_key: tuple[str, str] | None = None
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            active_slots = {}
            mode = ""
            last_spouse_key = None
            for row_index, row in enumerate(rows):
                mobile_slots = _canonical_header_slots(row, mobile_aliases)
                spouse_slots = _canonical_header_slots(row, spouse_aliases)
                compact_header = _compact("".join(row))
                broken_mobile = all(
                    marker in compact_header
                    for marker in ("编号", "手机号码", "信息更新日期", "数据发生机构名称")
                ) and not set(mobile_aliases) <= set(mobile_slots)
                broken_spouse = all(
                    marker in compact_header for marker in ("姓名", "证件类型", "证件号码", "工作单位", "联系电话")
                ) and not set(spouse_aliases) <= set(spouse_slots)
                if broken_mobile or broken_spouse:
                    _report_header_graph_failure(
                        parse_result,
                        dataset="mobile_phone_records" if broken_mobile else "spouse_records",
                        page=page,
                        table=table,
                        row_index=row_index,
                        row=row,
                    )
                    active_slots = {}
                    mode = ""
                    continue
                if set(mobile_aliases) <= set(mobile_slots):
                    active_slots = mobile_slots
                    mode = "mobile"
                    continue
                if set(spouse_aliases) <= set(spouse_slots):
                    active_slots = spouse_slots
                    mode = "spouse"
                    continue
                if "数据发生机构名称" in _compact("".join(row)) and last_spouse_key is not None:
                    active_slots = _canonical_header_slots(
                        row, {"sequence": ("编号",), "data_provider": ("数据发生机构名称",)}
                    )
                    mode = "spouse_provider"
                    continue
                if not active_slots:
                    continue
                if mode == "mobile":
                    sequence = _sequence_value(row, active_slots)
                    phone = re.sub(r"\D", "", _slot_value(row, active_slots, "mobile_phone"))
                    updated = _date(_slot_value(row, active_slots, "information_updated_date"))
                    if sequence is None:
                        continue
                    if not re.fullmatch(r"1[3-9]\d{9}", phone) or not updated:
                        _report_required_row_failure(
                            parse_result,
                            issue_code="candidate_b_mobile_row_unresolved",
                            dataset="mobile_phone_records",
                            sequence=sequence,
                            field_name="mobile_phone" if not re.fullmatch(r"1[3-9]\d{9}", phone) else "information_updated_date",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                        continue
                    key = (sequence, phone, str(updated))
                    mobile_id = stable_record_id("personal_mobile_phone", *key)
                    mobile_records[key] = {
                        "record_id": mobile_id,
                        "mobile_phone_record_id": mobile_id,
                        "sequence": sequence,
                        "mobile_phone": phone,
                        "information_updated_date": updated,
                        **(
                            {"data_provider": provider}
                            if (provider := _slot_value(row, active_slots, "data_provider")) and provider != "--"
                            else {}
                        ),
                        "source": "native_personal_detail_profile_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "confidence": 1.0,
                    }
                elif mode == "spouse":
                    name = _slot_value(row, active_slots, "name")
                    if not name or name == "--":
                        continue
                    document_number = _slot_value(row, active_slots, "document_number")
                    key = (name, document_number)
                    spouse_id = stable_record_id("personal_spouse", *key)
                    spouse_records[key] = {
                        "record_id": spouse_id,
                        "spouse_record_id": spouse_id,
                        "name": name,
                        **{
                            role: value
                            for role in ("document_type", "document_number", "employer", "phone")
                            if (value := _slot_value(row, active_slots, role)) and value != "--"
                        },
                        "source": "native_personal_detail_profile_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "confidence": 1.0,
                    }
                    last_spouse_key = key
                elif mode == "spouse_provider" and last_spouse_key in spouse_records:
                    provider = _slot_value(row, active_slots, "data_provider")
                    if not provider:
                        provider = next((_clean(value) for value in row if _clean(value)), "")
                    if provider and provider != "--":
                        spouse_records[last_spouse_key]["data_provider"] = provider
                        spouse_records[last_spouse_key]["source_refs"].append(
                            _source_ref(page, table, row=row_index)
                        )
    return {
        "mobile_phone_records": list(mobile_records.values()),
        "spouse_records": list(spouse_records.values()),
    }


def _extract_personal_notes(parse_result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations: list[dict[str, Any]] = []
    statements: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            for row_index, row in enumerate(rows):
                compact = _compact("".join(row))
                if all(marker in compact for marker in ("编号", "标注内容", "添加日期")):
                    for data_index, data_row in enumerate(rows[row_index + 1 :], start=row_index + 1):
                        values = _nonempty(data_row)
                        if len(values) < 3 or not values[0].isdigit():
                            break
                        text = values[1]
                        annotations.append(
                            {
                                "id": stable_record_id("personal_detail_annotation", page_number, text, values[2]),
                                "note_type": "report_annotation",
                                "text": text,
                                "added_date": _date(values[2]),
                                "logical_page": page_number,
                                "source_page": source_page,
                                "source": "native_personal_detail_note_table",
                                "source_refs": [_source_ref(page, table, row=data_index)],
                                "confidence": 1.0,
                            }
                        )
                marker = next(
                    (candidate for candidate in ("异议标注", "本人声明", "机构说明") if candidate in compact),
                    None,
                )
                if marker is None or row_index + 1 >= len(rows):
                    continue
                values = _nonempty(rows[row_index + 1])
                if not values:
                    continue
                text = values[0]
                added_date = _date(values[-1]) if len(values) > 1 else None
                target = annotations if marker == "异议标注" else statements
                target.append(
                    {
                        "id": stable_record_id("personal_detail_note", marker, page_number, text, added_date),
                        "note_type": {
                            "异议标注": "dispute_annotation",
                            "本人声明": "subject_statement",
                            "机构说明": "institution_statement",
                        }[marker],
                        "text": text,
                        "added_date": added_date,
                        "logical_page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_note_table",
                        "source_refs": [_source_ref(page, table, row=row_index + 1)],
                        "confidence": 1.0,
                    }
                )
    return annotations, statements


def _extract_source_rows(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", None) or []:
            table_id = str(getattr(table, "table_id", "") or "")
            for row_index, row in enumerate(_table_rows(table)):
                cells = [_clean(cell) for cell in row]
                records.append(
                    {
                        "source_table_row_id": stable_record_id(
                            "personal_detail_source_row", page_number, table_id, row_index
                        ),
                        "page": page_number,
                        "table_id": table_id,
                        "row_index": row_index,
                        "cells": cells,
                        "nonempty_cells": [cell for cell in cells if cell],
                        "source": "native_detail_source_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "confidence": 1.0,
                    }
                )
    return records


def _extract_postpaid_records(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            compact = _compact(" ".join(cell for row in rows[:2] for cell in row))
            if not all(
                marker in compact
                for marker in ("机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月")
            ):
                continue
            facts = _pairs(rows)
            institution = _clean(_field(facts, "机构名称"))
            business_type = _clean(_field(facts, "业务类型"))
            if not institution or not business_type:
                continue
            records.append(
                {
                    "postpaid_record_id": stable_record_id(
                        "postpaid", institution, business_type, _field(facts, "记账年月")
                    ),
                    "sequence": len(records) + 1,
                    "institution": institution,
                    "business_type": business_type,
                    "service_start_date": _date(_field(facts, "业务开通日期")),
                    "payment_status": _clean(_field(facts, "当前缴费状态")),
                    "current_arrears_amount": _number(_field(facts, "当前欠费金额")),
                    "billing_month": _date(_field(facts, "记账年月")),
                    "reporting_amount_currency": "CNY",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_postpaid_table",
                    "source_refs": [_source_ref(page, table)],
                    "confidence": 1.0,
                }
            )
    return records


def _extract_postpaid_payment_history(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:3] for cell in row))
            if not all(marker in compact for marker in ("机构名称", "业务类型", "缴费记录")):
                continue
            facts = _pairs(rows[:3])
            institution = _clean(_field(facts, "机构名称"))
            business_type = _clean(_field(facts, "业务类型"))
            if not institution or not business_type:
                continue
            months = _month_centers(table, rows)
            centers = _column_centers(table, max((len(row) for row in rows), default=0))
            postpaid_id = stable_record_id("postpaid", institution, business_type, _field(facts, "记账年月"))
            for row_index, row in enumerate(rows):
                year_cell = next(
                    (
                        (column, _compact(cell))
                        for column, cell in enumerate(row)
                        if re.fullmatch(r"20\d{2}", _compact(cell))
                    ),
                    None,
                )
                if year_cell is None:
                    continue
                year_column, year_raw = year_cell
                for column, cell in enumerate(row):
                    status = _compact(cell)
                    if column == year_column or status not in _STATUS_CODES:
                        continue
                    month = _nearest_month(column, centers, months)
                    if month is None:
                        continue
                    history_id = stable_record_id("postpaid_payment_history", postpaid_id, year_raw, month)
                    records.append(
                        {
                            "record_id": history_id,
                            "postpaid_payment_history_id": history_id,
                            "postpaid_record_id": postpaid_id,
                            "institution": institution,
                            "business_type": business_type,
                            "year": int(year_raw),
                            "month": month,
                            "status": status,
                            "source": "native_personal_detail_postpaid_history",
                            "source_refs": [_source_ref(page, table, row=row_index)],
                            "confidence": 1.0,
                        }
                    )
    return records


def _is_summary_anchor(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows for cell in row))
    return bool(
        "汇总" in compact or ("账户数" in compact and "首笔业务发放月份" in compact) or "最近1个月内的查询" in compact
    )


def _summary_title(rows: list[list[str]]) -> str:
    compact = _compact(" ".join(cell for row in rows for cell in row))
    return next(
        (_clean(cell) for row in rows for cell in row if "汇总" in _compact(cell)),
        "信用业务概要" if "首笔业务发放月份" in compact else "查询记录概要",
    )


def _summary_row_has_values(row: list[str]) -> bool:
    """Distinguish business rows from textual/group header rows."""
    for value in row:
        compact = _compact(value)
        if compact in {"--", "-"} or re.fullmatch(r"[-+]?\d[\d,./年月-]*", compact):
            return True
    return False


def _expanded_summary_headers(row: list[str], width: int) -> list[str]:
    values = [_clean(row[index] if index < len(row) else "") for index in range(width)]
    populated = [index for index, value in enumerate(values) if value]
    if not populated:
        return [""] * width
    expanded = [""] * width
    for position, start in enumerate(populated):
        end = populated[position + 1] if position + 1 < len(populated) else width
        for column in range(start, end):
            expanded[column] = values[start]
    if populated[0] > 0:
        for column in range(populated[0]):
            expanded[column] = values[populated[0]]
    return expanded


def _summary_business_rows(
    fragments: list[tuple[Any, Any, list[list[str]]]],
) -> list[tuple[Any, Any, int, list[str], list[str]]]:
    width = max((len(row) for _page, _table, rows in fragments for row in rows), default=0)
    header_paths: list[list[str]] = [[] for _column in range(width)]
    output: list[tuple[Any, Any, int, list[str], list[str]]] = []
    for page, table, rows in fragments:
        for source_row_index, row in enumerate(rows):
            nonempty = _nonempty(row)
            if len(nonempty) == 1 and "汇总" in _compact(nonempty[0]):
                continue
            if _summary_row_has_values(row):
                labels = ["/".join(path) for path in header_paths]
                output.append((page, table, source_row_index, row, labels))
                continue
            expanded = _expanded_summary_headers(row, width)
            distinct = {value for value in expanded if value}
            if len(distinct) == 1:
                label = next(iter(distinct))
                header_paths = [[label] for _column in range(width)]
                continue
            for column, label in enumerate(expanded):
                if label and (not header_paths[column] or header_paths[column][-1] != label):
                    header_paths[column].append(label)
    return output


def _extract_summary_datasets(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project logical summary grids, including headerless cross-page fragments."""
    records: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    physical: list[tuple[Any, Any, list[list[str]]]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if rows:
                physical.append((page, table, rows))

    continuation_check = getattr(parse_result, "tables_continue", None)
    reading_order = dict(getattr(parse_result, "reading_order_by_logical", {}) or {})
    consumed: set[int] = set()
    for index, (page, table, rows) in enumerate(physical):
        if index in consumed or not _is_summary_anchor(rows):
            continue
        fragments = [(page, table, rows)]
        anchor_width = max((len(row) for row in rows), default=0)
        cursor = index + 1
        while cursor < len(physical):
            next_page, next_table, next_rows = physical[cursor]
            if _is_summary_anchor(next_rows):
                break
            previous_page, previous_table, _previous_rows = fragments[-1]
            previous_page_number = int(getattr(previous_page, "page_number", 0) or 0)
            next_page_number = int(getattr(next_page, "page_number", 0) or 0)
            next_width = max((len(row) for row in next_rows), default=0)
            previous_table_id = str(getattr(previous_table, "table_id", "") or "")
            next_table_id = str(getattr(next_table, "table_id", "") or "")
            previous_order = reading_order.get(previous_page_number, previous_page_number)
            next_order = reading_order.get(next_page_number, next_page_number)
            if (
                next_order != previous_order + 1
                or not callable(continuation_check)
                or continuation_check(previous_table_id, next_table_id) is not True
                or (anchor_width and next_width != anchor_width)
            ):
                break
            fragments.append((next_page, next_table, next_rows))
            consumed.add(cursor)
            cursor += 1

        title = _summary_title(rows)
        table_id = str(getattr(table, "table_id", "") or "")
        page_number = int(getattr(page, "page_number", 0) or 0)
        summary_id = stable_record_id("personal_detail_summary", page_number, table_id, title)
        business_rows = _summary_business_rows(fragments)
        records.append(
            {
                "record_id": summary_id,
                "summary_record_id": summary_id,
                "summary_type": re.sub(r"信息?汇总$", "", title) or title,
                "title": title,
                "source_table_id": table_id,
                "source_column_count": max(
                    (len(row) for _fragment_page, _fragment_table, fragment_rows in fragments for row in fragment_rows),
                    default=0,
                ),
                "source_row_count": len(business_rows),
                "source": "native_personal_detail_summary_table",
                "source_refs": [
                    _source_ref(fragment_page, fragment_table) for fragment_page, fragment_table, _ in fragments
                ],
                "confidence": 1.0,
            }
        )
        for logical_row_index, (source_page, source_table, source_row_index, row, labels) in enumerate(
            business_rows, start=1
        ):
            for column_index, value in enumerate(row, start=1):
                value = _clean(value)
                if not value:
                    continue
                header = labels[column_index - 1] if column_index <= len(labels) else ""
                cell_id = stable_record_id(
                    "personal_detail_summary_cell",
                    summary_id,
                    logical_row_index,
                    column_index,
                    value,
                )
                cells.append(
                    {
                        "record_id": cell_id,
                        "summary_cell_id": cell_id,
                        "summary_record_id": summary_id,
                        "summary_type": re.sub(r"信息?汇总$", "", title) or title,
                        "title": title,
                        "row_index": logical_row_index,
                        "column_index": column_index,
                        "column_label": header or None,
                        "value": value,
                        "source": "native_personal_detail_summary_cell",
                        "source_refs": [
                            _source_ref(
                                source_page,
                                source_table,
                                row=source_row_index,
                                column=column_index - 1,
                            )
                        ],
                        "confidence": 1.0,
                    }
                )
    return records, cells


def _extract_recovery_records(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:3] for cell in row))
            if "债权接收日期" not in compact or "原债权人" not in compact:
                continue
            facts = _pairs(rows)
            institution = _clean(_field(facts, "管理机构"))
            received_date = _date(_field(facts, "债权接收日期"))
            records.append(
                {
                    "recovery_record_id": stable_record_id("recovery", institution, received_date, len(records) + 1),
                    "sequence": len(records) + 1,
                    "institution": institution,
                    "business_type": _clean(_field(facts, "业务种类")),
                    "debt_received_date": received_date,
                    "original_creditor": _clean(_field(facts, "原债权人")),
                    "original_business_type": _clean(_field(facts, "原债务业务种类")),
                    "debt_amount": _number(_field(facts, "债权金额")),
                    "transfer_repayment_status": _clean(_field(facts, "债权转移时的还款状态")),
                    "snapshot_date": next(
                        (
                            _date("-".join(match.groups()))
                            for row in rows
                            if (match := _AS_OF_RE.search(_clean(" ".join(row))))
                        ),
                        None,
                    ),
                    "account_status": _clean(_field(facts, "账户状态")),
                    "balance": _number(_field(facts, "余额")),
                    "last_repayment_date": _date(_field(facts, "最近一次还款日期")),
                    "close_date": _date(_field(facts, "账户关闭日期")),
                    "reporting_amount_currency": "CNY",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_recovery_table",
                    "source_refs": [_source_ref(page, table)],
                    "confidence": 1.0,
                }
            )
    return records


def _extract_header_datasets(parse_result: Any, full_text: str) -> dict[str, list[dict[str, Any]]]:
    compact = _compact(full_text)
    report_number = next(iter(re.findall(r"报告编号[:：]?(\d{18,30})", compact)), None)
    time_match = re.search(
        r"报告时间[:：]?(20\d{2})[.年/-](\d{1,2})[.月/-](\d{1,2})日?(\d{1,2}):(\d{2}):(\d{2})",
        compact,
    )
    report_time = (
        f"{int(time_match.group(1)):04d}-{int(time_match.group(2)):02d}-{int(time_match.group(3)):02d}"
        f"T{int(time_match.group(4)):02d}:{time_match.group(5)}:{time_match.group(6)}+08:00"
        if time_match
        else None
    )
    field_candidates: dict[str, list[str]] = defaultdict(list)
    other_documents: list[tuple[str, str]] = []
    for page in list(getattr(parse_result, "pages", None) or [])[:1]:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            for row_index, row in enumerate(rows[:-1]):
                labels = _nonempty(row)
                values = _nonempty(rows[row_index + 1])
                label_text = _compact("".join(labels))
                if (
                    all(
                        marker in label_text
                        for marker in ("被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因")
                    )
                    and len(values) >= 5
                ):
                    for key, value in zip(
                        ("subject_name", "primary_id_type", "primary_id_number", "query_institution", "query_reason"),
                        values[:5],
                        strict=True,
                    ):
                        field_candidates[key].append(value)
                elif labels == ["证件类型", "证件号码"] and len(values) >= 2:
                    other_documents.append((values[0], values[1]))
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import PBOCPersonalDetailNativeParser

    parser = PBOCPersonalDetailNativeParser(parse_result)
    label_to_key = {
        "被查询者姓名": "subject_name",
        "被查询者证件类型": "primary_id_type",
        "被查询者证件号码": "primary_id_number",
        "查询机构": "query_institution",
        "查询原因": "query_reason",
        "报告编号": "report_number",
        "报告时间": "report_time",
    }
    for candidate in parser.records("report_header"):
        for label, key in label_to_key.items():
            value = candidate.fields.get(label)
            if value not in (None, ""):
                field_candidates[key].append(str(value))
    if report_number:
        field_candidates["report_number"].append(report_number)
    if report_time:
        field_candidates["report_time"].append(report_time)

    def candidate_valid(key: str, value: str, selected_id_type: str | None = None) -> bool:
        if key == "query_reason":
            return validate_pboc_field(value, "inquiry_reason").valid
        return header_field_valid(key, value, id_type=selected_id_type)

    provisional_id_types = {
        value.strip()
        for value in field_candidates.get("primary_id_type", [])
        if candidate_valid("primary_id_type", value)
    }
    selected_id_type = next(iter(provisional_id_types)) if len(provisional_id_types) == 1 else None
    def select(key: str, selected_type: str | None = None) -> str | None:
        observed = tuple(dict.fromkeys(value.strip() for value in field_candidates.get(key, []) if value.strip()))
        valid = tuple(value for value in observed if candidate_valid(key, value, selected_type))
        if len(valid) == 1:
            return valid[0]
        record_issue(
            parse_result,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="page_one_consensus_unresolved",
                message="Page-one header evidence was missing, invalid, or conflicting; the normalized value was withheld.",
                parser_stage="page_one_consensus",
                target_dataset="personal_report_metadata",
                target_record_id="personal_report_metadata:primary",
                field_name=key,
                observed_value=list(observed),
                source_refs=(
                    {
                        "source": "candidate_b_page_one_business_fields",
                        "logical_page": 1,
                        "source_page": 1,
                        "geometry_scope": "logical_page",
                    },
                ),
                reason_codes=("page_one_consensus", "schema_field_validation", "normalized_value_withheld"),
            ),
        )
        return None

    subject_name = select("subject_name")
    id_type = select("primary_id_type")
    id_number = select("primary_id_number", id_type or "身份证")
    observed_primary_numbers = tuple(
        dict.fromkeys(
            value.strip()
            for value in field_candidates.get("primary_id_number", [])
            if value.strip()
        )
    )
    source_primary_id_number = (
        id_number
        or (observed_primary_numbers[0] if len(observed_primary_numbers) == 1 else None)
    )
    if id_type is None and cn_identity_number_valid(id_number):
        id_type = "身份证"
        for issue in getattr(parse_result, "_personal_detail_extraction_issues", []) or []:
            if (
                isinstance(issue, dict)
                and issue.get("issue_code") == "page_one_consensus_unresolved"
                and issue.get("field_name") == "primary_id_type"
            ):
                issue["status"] = "resolved"
                issue["severity"] = "info"
                issue["reason_codes"] = [
                    *issue.get("reason_codes", []),
                    "checksum_valid_resident_identity_implies_document_type",
                ]
    query_institution = select("query_institution")
    query_reason = select("query_reason")
    report_number = select("report_number")
    report_time = select("report_time")
    metadata = [
        {
            "personal_report_metadata_id": stable_record_id(
                "personal_report_metadata", report_number, report_time, subject_name
            ),
            "report_number": report_number,
            "report_time": report_time,
            "subject_name": subject_name,
            "primary_id_type": id_type,
            "primary_id_number": id_number,
            "query_institution": query_institution,
            "query_reason": query_reason,
            "reporting_currency": "CNY",
            "reporting_amount_unit": "yuan",
            "reporting_amount_precision": 0,
            "amount_policy_source": "personal_detailed_report_notes",
            "source": "native_detail_header",
            "source_refs": [{"source": "native_detail_header", "logical_page": 1, "source_page": 1}],
            "confidence": 1.0,
        }
    ]
    identities: list[dict[str, Any]] = []
    if id_type and source_primary_id_number:
        identities.append(
            {
                "identity_document_id": stable_record_id(
                    "identity_document", "primary", id_type, source_primary_id_number
                ),
                "sequence": 1,
                "holder_name": subject_name,
                "document_type": id_type,
                # Preserve the source-visible row even when its displayed
                # number fails a checksum/type contract.  The v2 quality gate
                # withholds the normalized field and publishes the linked
                # uncertainty instead of silently deleting the identity row.
                "document_number": source_primary_id_number,
                "is_primary": True,
                "source": "native_detail_header",
                "source_refs": [{"source": "native_detail_header", "logical_page": 1, "source_page": 1}],
                "confidence": 1.0,
            }
        )
    for other_type, other_number in other_documents:
        identities.append(
            {
                "identity_document_id": stable_record_id("identity_document", "other", other_type, other_number),
                "sequence": len(identities) + 1,
                "holder_name": subject_name,
                "document_type": other_type,
                "document_number": other_number,
                "is_primary": False,
                "source": "native_detail_header",
                "source_refs": [{"source": "native_detail_header", "logical_page": 1, "source_page": 1}],
                "confidence": 1.0,
            }
        )
    return {"personal_report_metadata": metadata, "identity_documents": identities}


def extract_personal_detail_native_business(parse_result: Any, full_text: str) -> dict[str, Any]:
    """Return the business slice of the authoritative Candidate B result."""
    from copy import deepcopy

    from docmirror.plugins.credit_report.personal_detail_scanned.context import (
        build_personal_detail_extraction_context,
    )

    context = build_personal_detail_extraction_context(parse_result)
    return deepcopy(context.candidate_b_extraction(full_text).business)


def extract_personal_detail_section_content(parse_result: Any, full_text: str) -> dict[str, Any]:
    """Return the supplemental slice of the authoritative Candidate B result."""
    from copy import deepcopy

    from docmirror.plugins.credit_report.personal_detail_scanned.context import (
        build_personal_detail_extraction_context,
    )

    context = build_personal_detail_extraction_context(parse_result)
    return deepcopy(context.candidate_b_extraction(full_text).section_content)


def extract_personal_detail_common_datasets(
    parse_result: Any,
    full_text: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return report metadata datasets for any content mode."""
    return _extract_header_datasets(parse_result, full_text)


__all__ = [
    "extract_personal_detail_common_datasets",
    "extract_personal_detail_native_business",
    "extract_personal_detail_section_content",
]
