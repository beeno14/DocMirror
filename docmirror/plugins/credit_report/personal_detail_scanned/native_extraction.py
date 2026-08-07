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
from datetime import date
from typing import Any, Iterable, Mapping

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


_LIABILITY_CANONICAL_FIELDS = (
    "institution",
    "business_type",
    "open_date",
    "due_date",
    "responsibility_type",
    "responsibility_amount",
    "currency",
    "contract_number",
    "related_party_name",
    "related_party_id_type",
    "related_party_id_number",
    "snapshot_date",
    "balance",
    "five_tier_class",
    "overdue_months_or_repayment_status",
)


def _liability_exact_replay_key(record: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return an exact business fingerprint, never a fuzzy record identity."""

    contract_number = _compact(record.get("contract_number"))
    if not contract_number:
        return None
    return (
        contract_number,
        *(
            _compact(record.get(field_name))
            for field_name in _LIABILITY_CANONICAL_FIELDS
            if field_name != "contract_number"
        ),
    )


def _dedupe_liability_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suppress only byte-equivalent canonical liability observations.

    A former implementation merged identifier prefixes or any observations
    sharing three business fields.  Different guarantee contracts commonly do
    share borrower, dates, and amounts, so that policy could silently erase a
    real record.  Candidate B now reconciles only exact contract/ordinal
    identities; this compatibility helper removes exact replays only.
    """

    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        replay_key = _liability_exact_replay_key(record)
        if replay_key is not None and replay_key in seen:
            continue
        if replay_key is not None:
            seen.add(replay_key)
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
    normalized_year = int(year)
    normalized_month = int(month)
    if not 1 <= normalized_month <= 12:
        return None
    if day:
        normalized_day = int(day)
        try:
            date(normalized_year, normalized_month, normalized_day)
        except ValueError:
            return None
        return f"{normalized_year:04d}-{normalized_month:02d}-{normalized_day:02d}"
    return f"{normalized_year:04d}-{normalized_month:02d}"


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
        # Canonical projection keeps both registered-page coordinates and the
        # originating source-page coordinates.  A business-field repair must
        # point at the originating cell, never at the enclosing table.
        cell_bboxes = metadata.get("source_cell_bboxes") or metadata.get("cell_bboxes")
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
        ref["binding_quality"] = "canonical_header_column"
        ref["binding"] = "canonical_field_slot"
        ref["canonical_row"] = row
        ref["canonical_column"] = column
        if bbox is None:
            # Row/column identity is still exact even when a synthetic fixture
            # has no pixel geometry.  Do not substitute a table-sized bbox:
            # that would falsely authorize a broad field-level correction.
            ref["geometry_scope"] = "canonical_field_slot"
    if bbox is None and (row is None or column is None):
        bbox = getattr(table, "bbox", None)
        if bbox and len(bbox) == 4:
            ref["geometry_scope"] = "table"
    if bbox and len(bbox) == 4:
        ref["bbox"] = list(bbox)
    return ref


def _nonempty(row: list[str]) -> list[str]:
    return [_clean(cell) for cell in row if _clean(cell)]


def _exact_label_observations(
    rows: list[list[str]],
) -> tuple[dict[str, list[tuple[str, int, int]]], set[tuple[str, int, int]]]:
    """Bind canonical labels only to the value in the same physical column.

    Missing cells are not allowed to shift later values left/right, and two
    equally plausible rows are retained as separate observations so the
    field-level reconciler can report a conflict instead of overwriting it.
    """

    observations: defaultdict[str, list[tuple[str, int, int]]] = defaultdict(list)
    unresolved: set[tuple[str, int, int]] = set()
    for index in range(len(rows) - 1):
        for column, cell in enumerate(rows[index]):
            label = _compact(cell)
            if label not in _ACCOUNT_LABELS:
                continue
            value = _clean(rows[index + 1][column]) if column < len(rows[index + 1]) else ""
            if not value or _compact(value) in _ACCOUNT_LABELS:
                unresolved.add((label, index, column))
                continue
            observations[label].append((value, index + 1, column))
    return dict(observations), unresolved


def _pairs(rows: list[list[str]]) -> dict[str, str]:
    """Return only unambiguous exact-column label/value bindings."""

    observations, _unresolved = _exact_label_observations(rows)
    out: dict[str, str] = {}
    for label, values in observations.items():
        distinct = {_compact(value) for value, _row, _column in values}
        if len(distinct) == 1:
            out[label] = values[0][0]
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


def _append_internal_field(record: dict[str, Any], key: str, field_name: str) -> None:
    values = record.setdefault(key, [])
    if field_name not in values:
        values.append(field_name)


def _mark_source_absent(record: dict[str, Any], field_name: str, raw: str = "--") -> None:
    _append_internal_field(record, "_source_absent_fields", field_name)
    record.setdefault("canonical_raw", {})[field_name] = raw


def _merge_exact_observation(
    parse_result: Any,
    record: dict[str, Any],
    *,
    dataset: str,
    target_record_id: str,
    field_name: str,
    value: Any,
    raw: str,
    source_ref: Mapping[str, Any],
    parser_stage: str,
) -> None:
    """Merge one exact-slot observation without last-write-wins behavior."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    ref = {**dict(source_ref), "field_name": field_name}
    refs = record.setdefault("source_refs_by_field", {}).setdefault(field_name, [])
    if ref not in refs:
        refs.append(ref)
    observations = record.setdefault("_field_observations", {}).setdefault(field_name, [])
    marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if not any(item.get("marker") == marker for item in observations):
        observations.append({"marker": marker, "value": value, "raw": raw, "source_refs": [ref]})
    else:
        prior = next(item for item in observations if item.get("marker") == marker)
        if ref not in prior.setdefault("source_refs", []):
            prior["source_refs"].append(ref)
    distinct = {str(item.get("marker") or "") for item in observations}
    if len(distinct) == 1 and field_name not in record.get("_invalid_observation_fields", []):
        record[field_name] = value
        record.setdefault("canonical_raw", {})[field_name] = raw
        return

    record.pop(field_name, None)
    _append_internal_field(record, "_unresolved_fields", field_name)
    prior_raw = record.setdefault("canonical_raw", {}).get(field_name)
    raw_values = prior_raw if isinstance(prior_raw, list) else ([prior_raw] if prior_raw not in (None, "") else [])
    for item in observations:
        candidate_raw = str(item.get("raw") or "")
        if candidate_raw not in raw_values:
            raw_values.append(candidate_raw)
    record.setdefault("canonical_raw", {})[field_name] = raw_values
    reported = record.setdefault("_reported_field_conflicts", [])
    if field_name in reported:
        return
    reported.append(field_name)
    conflict_refs = [
        candidate_ref
        for item in observations
        for candidate_ref in item.get("source_refs") or ()
        if isinstance(candidate_ref, Mapping)
    ]
    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_exact_slot_value_conflict",
            message="Canonical observations for one business field disagreed; the normalized value was withheld.",
            parser_stage=parser_stage,
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value=raw_values,
            source_refs=conflict_refs,
            reason_codes=("canonical_field_slot", "conflicting_exact_observations", "normalized_value_withheld"),
        ),
    )


def _reject_exact_observation(
    parse_result: Any,
    record: dict[str, Any],
    *,
    dataset: str,
    target_record_id: str,
    field_name: str,
    raw: str,
    source_ref: Mapping[str, Any],
    parser_stage: str,
) -> None:
    """Retain an invalid exact cell as issue evidence, never as business data."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    ref = {**dict(source_ref), "field_name": field_name}
    refs = record.setdefault("source_refs_by_field", {}).setdefault(field_name, [])
    if ref not in refs:
        refs.append(ref)
    record.pop(field_name, None)
    _append_internal_field(record, "_unresolved_fields", field_name)
    _append_internal_field(record, "_invalid_observation_fields", field_name)
    prior = record.setdefault("canonical_raw", {}).get(field_name)
    raw_values = prior if isinstance(prior, list) else ([prior] if prior not in (None, "") else [])
    if raw not in raw_values:
        raw_values.append(raw)
    record["canonical_raw"][field_name] = raw_values
    reported = record.setdefault("_reported_invalid_fields", [])
    if field_name in reported:
        return
    reported.append(field_name)
    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_exact_slot_value_invalid",
            message="A value was present in the canonical field slot but failed its field contract and was withheld.",
            parser_stage=parser_stage,
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value=raw_values,
            source_refs=(ref,),
            reason_codes=("canonical_field_slot", "field_contract_failed", "normalized_value_withheld"),
        ),
    )


def _apply_account_facts(
    parse_result: Any,
    account: dict[str, Any],
    rows: list[list[str]],
    *,
    page: Any,
    table: Any,
    physical_row_indices: list[int | None] | None = None,
) -> None:
    observations, unresolved = _exact_label_observations(rows)
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
    dataset = "credit_accounts"
    target_record_id = str(account.get("account_id") or "")
    parser_stage = "candidate_b_account_canonical_slots"

    def physical_row(row_index: int) -> int | None:
        if physical_row_indices is None:
            return row_index
        if 0 <= row_index < len(physical_row_indices):
            return physical_row_indices[row_index]
        return None

    for target, labels, converter in mappings:
        values = [item for label in labels for item in observations.get(_compact(label), ())]
        for raw, value_row_index, column in values:
            source_row = physical_row(value_row_index)
            if source_row is None:
                continue
            ref = _source_ref(page, table, row=source_row, column=column)
            if _compact(raw) in {"-", "--"}:
                _mark_source_absent(account, target, raw)
                continue
            value = converter(raw)
            valid = value not in (None, "")
            if converter is _number:
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif converter is _date:
                valid = isinstance(value, str) and bool(re.fullmatch(r"20\d{2}-\d{2}(?:-\d{2})?", value))
            elif converter is _business_text:
                compact_value = _compact(value)
                valid = bool(value) and not any(label in compact_value for label in _ACCOUNT_LABELS)
                if target == "management_institution":
                    valid = valid and not bool(
                        _DATE_RE.search(compact_value)
                        or re.search(r"[A-Z][A-Z0-9-]{7,}\d", compact_value, re.IGNORECASE)
                    )
            elif target == "account_identifier":
                typed_value = _typed_identifier(value)
                valid = typed_value is not None
                value = typed_value
            if not valid:
                _reject_exact_observation(
                    parse_result,
                    account,
                    dataset=dataset,
                    target_record_id=target_record_id,
                    field_name=target,
                    raw=raw,
                    source_ref=ref,
                    parser_stage=parser_stage,
                )
                continue
            _merge_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=target,
                value=value,
                raw=raw,
                source_ref=ref,
                parser_stage=parser_stage,
            )

    for raw_currency, value_row_index, column in [
        item for label in ("账户币种", "币种") for item in observations.get(label, ())
    ]:
        source_row = physical_row(value_row_index)
        if source_row is None:
            continue
        ref = _source_ref(page, table, row=source_row, column=column)
        if _compact(raw_currency) in {"-", "--"}:
            _mark_source_absent(account, "currency", raw_currency)
            _mark_source_absent(account, "account_currency", raw_currency)
            continue
        currency = _currency(raw_currency)
        if not currency or not (currency in {"CNY", "USD", "EUR", "HKD"} or re.fullmatch(r"[A-Z]{3}", currency)):
            _reject_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name="currency",
                raw=raw_currency,
                source_ref=ref,
                parser_stage=parser_stage,
            )
            continue
        for field_name in ("currency", "account_currency"):
            _merge_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=field_name,
                value=currency,
                raw=raw_currency,
                source_ref=ref,
                parser_stage=parser_stage,
            )

    for raw_status, value_row_index, column in [
        item for label in ("账户状态", "状态") for item in observations.get(label, ())
    ]:
        source_row = physical_row(value_row_index)
        if source_row is None:
            continue
        ref = _source_ref(page, table, row=source_row, column=column)
        if _compact(raw_status) in {"-", "--"}:
            _mark_source_absent(account, "account_status", raw_status)
            continue
        status_values = _status_fields(raw_status)
        if status_values.get("account_status_resolution") == "unresolved":
            _reject_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name="account_status",
                raw=raw_status,
                source_ref=ref,
                parser_stage=parser_stage,
            )
            continue
        for field_name, value in status_values.items():
            _merge_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=field_name,
                value=value,
                raw=raw_status,
                source_ref=ref,
                parser_stage=parser_stage,
            )

    # Do not report the final label row yet: it may be a verified continuation
    # whose value row begins the next logical page.  Internal missing cells are
    # immediately repair-eligible.
    for label, label_row_index, column in unresolved:
        if label_row_index == len(rows) - 1:
            continue
        target = next((field for field, labels, _converter in mappings if label in labels), None)
        if target is None:
            target = {"账户币种": "currency", "币种": "currency", "账户状态": "account_status", "状态": "account_status"}.get(label)
        source_row = physical_row(label_row_index)
        if not target or source_row is None:
            continue
        _reject_exact_observation(
            parse_result,
            account,
            dataset=dataset,
            target_record_id=target_record_id,
            field_name=target,
            raw="",
            source_ref=_source_ref(page, table, row=source_row, column=column),
            parser_stage=parser_stage,
        )

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
    if account.get("currency"):
        account["reporting_amount_currency"] = account["currency"]
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
    issue_owner: Any,
    account: dict[str, Any],
    page: Any,
    table: Any,
    rows: list[list[str]],
) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

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
        target_dataset = {
            "special_transaction": "credit_account_special_transactions",
            "large_installment": "credit_card_large_installments",
            "latest_repayment": "credit_account_latest_repayments",
            "special_event_note": "credit_account_special_events",
        }[event_type]
        value_row = rows[row_index + 1]
        event_id = stable_record_id(
            "personal_detail_account_event",
            account.get("account_id"),
            event_type,
            page_number,
            row_index,
        )
        record: dict[str, Any] = {
            "record_id": event_id,
            "account_event_id": event_id,
            "account_id": account.get("account_id"),
            "event_type": event_type,
            "source": "native_personal_detail_account_event",
            "source_refs": [_source_ref(page, table, row=row_index)],
            "source_refs_by_field": {},
            "canonical_raw": {},
            "confidence": float(getattr(table, "confidence", None) or 0.9),
        }
        observations, unresolved_slots = _exact_label_observations([row, value_row])

        def bind_exact(field_name: str, label: str, converter: Any) -> None:
            candidates = observations.get(label) or []
            distinct = {_compact(raw) for raw, _physical_row, _column in candidates if _compact(raw)}
            if len(distinct) != 1:
                record_issue(
                    issue_owner,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_account_event_slot_unresolved",
                        message=(
                            "A canonical account-event field was missing or ambiguous; no adjacent value "
                            "was shifted into the slot."
                        ),
                        parser_stage="candidate_b_account_event_canonical_slots",
                        target_dataset=target_dataset,
                        target_record_id=event_id,
                        field_name=field_name,
                        observed_value={
                            "label": label,
                            "candidate_values": [raw for raw, _physical_row, _column in candidates],
                        },
                        source_refs=(_source_ref(page, table, row=row_index),),
                        reason_codes=(
                            "canonical_event_label_observed",
                            "exact_value_slot_unresolved",
                            "positional_fallback_forbidden",
                            "normalized_value_withheld",
                        ),
                    ),
                )
                _append_internal_field(record, "_unresolved_fields", field_name)
                return
            raw, physical_row, column = candidates[0]
            source_ref = _source_ref(page, table, row=row_index + physical_row, column=column)
            if _compact(raw) in {"-", "--"}:
                _mark_source_absent(record, field_name, raw)
                record["source_refs_by_field"].setdefault(field_name, []).append(
                    {**source_ref, "field_name": field_name}
                )
                return
            value = converter(raw)
            if value in (None, ""):
                _reject_exact_observation(
                    issue_owner,
                    record,
                    dataset=target_dataset,
                    target_record_id=event_id,
                    field_name=field_name,
                    raw=raw,
                    source_ref=source_ref,
                    parser_stage="candidate_b_account_event_canonical_slots",
                )
                return
            record[field_name] = value
            record["canonical_raw"][field_name] = raw
            record["source_refs_by_field"].setdefault(field_name, []).append(
                {**source_ref, "field_name": field_name}
            )

        field_contracts: dict[str, tuple[tuple[str, str, Any], ...]] = {
            "special_transaction": (
                ("transaction_type", "特殊交易类型", _clean),
                ("event_date", "发生日期", _date),
                ("changed_months", "变更月数", _number),
                ("amount", "发生金额", _number),
                ("details", "明细记录", _clean),
            ),
            "large_installment": (
                ("installment_limit", "大额专项分期额度", _number),
                ("effective_date", "分期额度生效日期", _date),
                ("expiry_date", "分期额度到期日期", _date),
                ("used_installment_amount", "已用分期金额", _number),
            ),
            "latest_repayment": (
                ("five_tier_class", "五级分类", _clean),
                ("balance", "余额", _number),
                ("repayment_date", "还款日期", _date),
                ("repayment_amount", "还款金额", _number),
                ("repayment_status", "当前还款状态", _clean),
            ),
            "special_event_note": (("details", "特殊事件说明", _clean),),
        }
        for field_name, label, converter in field_contracts[event_type]:
            bind_exact(field_name, label, converter)
        # Keep the observed row even when one slot failed: the normalized
        # omissions are linked to explicit issues and the source record count
        # remains conserved.
        if unresolved_slots:
            record.setdefault("_unresolved_source_slots", []).extend(
                sorted({label for label, _row, _column in unresolved_slots})
            )
        events.append(record)
    return events


def _extract_table_accounts(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accounts: list[dict[str, Any]] = []
    repayments: list[dict[str, Any]] = []
    account_events: list[dict[str, Any]] = []
    phase = "non_revolving_loan"
    current: dict[str, Any] | None = None
    pending_labels: list[str] | None = None
    current_table_id = ""
    current_logical_page = 0
    continuation_check = getattr(parse_result, "tables_continue", None)

    for page in getattr(parse_result, "pages", None) or []:
        for source_table_index, table in enumerate(getattr(page, "tables", None) or []):
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

                table_ref = _source_ref(page, table)
                table_observation_id = stable_record_id(
                    "credit_account_table_observation",
                    account_type,
                    table_ref.get("logical_page"),
                    table_ref.get("source_page"),
                    table_ref.get("table_id"),
                    table_ref.get("bbox"),
                    source_table_index,
                )
                current = {
                    # A table is an observation, not an account identity.  Only
                    # a printed anchor can contribute the family ordinal.  The
                    # source-position digest stays stable when the ordinal is
                    # unreadable and cannot silently masquerade as one.
                    "account_id": table_observation_id,
                    "_table_observation_id": table_observation_id,
                    "sequence": len(accounts) + 1,
                    "account_type": account_type,
                    "source": "native_detail_account_table",
                    "source_refs": [table_ref],
                    "confidence": 1.0,
                    "canonical_raw": {},
                }
                current.update(_account_heading_for_table(page, table))
                if account_type in {"credit_card", "quasi_credit_card"}:
                    current["credit_card_type"] = account_type
                _apply_account_facts(
                    parse_result,
                    current,
                    rows,
                    page=page,
                    table=table,
                    physical_row_indices=list(range(len(rows))),
                )
                accounts.append(current)
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                account_events.extend(_account_events(parse_result, current, page, table, rows))
                pending_labels = rows[-1] if rows and _label_row(rows[-1]) else None
                current_table_id = str(getattr(table, "table_id", "") or "")
                current_logical_page = int(getattr(page, "page_number", 0) or 0)
                continue

            logical_page = int(getattr(page, "page_number", 0) or 0)
            candidate_table_id = str(getattr(table, "table_id", "") or "")
            continuation = None
            if current is not None and callable(continuation_check) and current_table_id and candidate_table_id:
                continuation = continuation_check(current_table_id, candidate_table_id)

            # A neighbouring table is never assigned to the current account by
            # page order alone.  Both same-page fragments and cross-page
            # fragments require the entity graph's affirmative continuation
            # decision.  Unknown is a veto, not permission to absorb cells.
            if current is not None and continuation is True and not _other_entity_table(rows):
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
                _apply_account_facts(
                    parse_result,
                    current,
                    fact_rows,
                    page=page,
                    table=table,
                    physical_row_indices=([None] if pending_labels else []) + list(range(len(rows))),
                )
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                account_events.extend(_account_events(parse_result, current, page, table, rows))
                pending_labels = rows[-1] if rows and _label_row(rows[-1]) else None
                current_table_id = str(getattr(table, "table_id", "") or current_table_id)
                current_logical_page = logical_page or current_logical_page
                continue

            if current is not None:
                compact_values = _compact(" ".join(cell for row in rows for cell in row))
                looks_like_account_fragment = (
                    any(label in compact_values for label in _ACCOUNT_LABELS)
                    or "还款记录" in compact_values
                    or "逾期金额" in compact_values
                )
                if looks_like_account_fragment and not _other_entity_table(rows):
                    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                        make_issue,
                        record_issue,
                    )

                    record_issue(
                        parse_result,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_account_table_fragment_owner_unresolved",
                            message=(
                                "An account-like table fragment was not joined because canonical continuation "
                                "ownership was not affirmatively established."
                            ),
                            parser_stage="candidate_b_account_table_ownership",
                            target_dataset="credit_accounts",
                            target_record_id=str(current.get("account_id") or "") or None,
                            source_refs=(_source_ref(page, table),),
                            observed_value={
                                "left_table_id": current_table_id,
                                "candidate_table_id": candidate_table_id,
                                "left_logical_page": current_logical_page,
                                "candidate_logical_page": logical_page,
                                "continuation_decision": continuation,
                            },
                            reason_codes=(
                                "canonical_continuation_not_verified",
                                "table_fragment_withheld",
                                "neighbour_order_not_used",
                            ),
                        ),
                    )
                current = None
                pending_labels = None
                current_table_id = ""
                current_logical_page = 0

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
    printed_ordinals: list[int | None] = []
    ordinal_counts: Counter[tuple[str, int]] = Counter()
    for start in starts:
        anchor = flattened[start]
        match = _ACCOUNT_ANCHOR_RE.search(str(anchor.get("text") or ""))
        detected = int(match.group(1)) if match and match.group(1) else None
        printed_ordinals.append(detected)
        if detected is not None and detected > 0:
            ordinal_counts[(str(anchor.get("account_type") or ""), detected)] += 1

    transition_check = getattr(parse_result, "allows_scanned_line_transition", None)
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
        # A physical/logical page boundary belongs to the account segment only
        # when the canonical entity context affirms the evidence transition.
        # Adjacency by itself is deliberately insufficient.
        verified_end = end
        for index in range(start + 1, end):
            left = flattened[index - 1]
            right = flattened[index]
            left_page = int(left.get("page") or 0)
            right_page = int(right.get("page") or 0)
            if left_page == right_page:
                continue
            decision = None
            if callable(transition_check):
                decision = transition_check(
                    left_page,
                    left,
                    int(left.get("line_index") or 0),
                    right_page,
                    right,
                    int(right.get("line_index") or 0),
                )
            if decision is not True:
                verified_end = index
                break
        detail = flattened[start:verified_end]
        boundary = flattened[verified_end] if verified_end < len(flattened) else None

        page_segments: list[dict[str, Any]] = []
        segment_pages: list[int] = []
        for line in detail:
            page_number = int(line.get("page") or 0)
            if page_number and page_number not in segment_pages:
                segment_pages.append(page_number)
        anchor_bbox = anchor.get("bbox")
        anchor_top = (
            float(anchor_bbox[1])
            if isinstance(anchor_bbox, (list, tuple)) and len(anchor_bbox) == 4
            else None
        )
        for page_index, page_number in enumerate(segment_pages):
            upper: float | None = None
            if boundary is not None and int(boundary.get("page") or 0) == page_number:
                boundary_bbox = boundary.get("bbox")
                if isinstance(boundary_bbox, (list, tuple)) and len(boundary_bbox) == 4:
                    upper = float(boundary_bbox[1])
            page_segments.append(
                {
                    "logical_page": page_number,
                    "min_y": anchor_top if page_index == 0 else 0.0,
                    "max_y": upper,
                    "continuation_verified": page_index > 0,
                }
            )

        detected = printed_ordinals[position]
        ordinal_is_unique = bool(
            detected is not None
            and detected > 0
            and ordinal_counts[(account_type, detected)] == 1
        )
        if ordinal_is_unique:
            account_id = f"credit_account:{account_type}:{detected}"
            ordinal_status = "printed_unique"
        else:
            ordinal_status = "printed_duplicate" if detected else "printed_unreadable"
            account_id = stable_record_id(
                "credit_account_provisional",
                account_type,
                anchor.get("source_page"),
                anchor.get("page"),
                anchor.get("bbox"),
                anchor.get("evidence_ids"),
                anchor.get("line_index"),
                anchor.get("text"),
            )
        skeleton = {
                "account_id": account_id,
                "sequence": len(skeletons) + 1,
                **({"category_sequence": detected} if ordinal_is_unique else {}),
                "account_type": account_type,
                "account_family_quality": str(anchor.get("account_family_quality") or ""),
                "_printed_ordinal_status": ordinal_status,
                "_canonical_segment": {
                    "ownership_basis": "printed_anchor_to_next_anchor",
                    "anchor_logical_page": int(anchor.get("page") or 0),
                    "anchor_bbox": list(anchor.get("bbox") or ()),
                    "pages": page_segments,
                    "cross_page_continuation_verified": len(page_segments) > 1,
                },
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
        if not ordinal_is_unique:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_account_printed_ordinal_unresolved",
                    message=(
                        "A printed account ordinal was missing or duplicated; encounter order was not emitted "
                        "as the account family sequence and a source-stable provisional identity was used."
                    ),
                    parser_stage="candidate_b_account_anchor_ownership",
                    target_dataset="credit_accounts",
                    target_record_id=account_id,
                    field_name="category_sequence",
                    observed_value={
                        "account_type": account_type,
                        "printed_ordinal": detected,
                        "ordinal_status": ordinal_status,
                    },
                    source_refs=skeleton.get("source_refs") or (),
                    reason_codes=(
                        "printed_ordinal_duplicate" if detected else "printed_ordinal_unreadable",
                        "encounter_order_not_used",
                        "stable_provisional_identity",
                        "normalized_ordinal_withheld",
                    ),
                ),
            )
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
    if not page or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        top = float(bbox[1])
    except (TypeError, ValueError):
        return None
    return page, top


def _match_account_table_observations(
    skeletons: list[dict[str, Any]],
    table_accounts: list[dict[str, Any]],
) -> dict[int, int]:
    """Match tables only inside an anchor-owned canonical stream segment.

    The exact ownership interval begins at the printed anchor and ends at the
    next printed anchor (or the registered section boundary).  Later logical
    pages are eligible only when ``_account_anchor_skeletons`` recorded an
    affirmative continuation transition.  Family labels, encounter order and
    a globally nearest predecessor are deliberately not identity keys.
    """
    matches: dict[int, int] = {}
    consumed_tables: set[int] = set()
    fallback_segments: dict[int, tuple[int, float, float | None]] = {}
    positioned_skeletons = [
        (index, position)
        for index, skeleton in enumerate(skeletons)
        if (position := _account_stream_position(skeleton)) is not None
    ]
    for offset, (skeleton_index, (page, top)) in enumerate(positioned_skeletons):
        upper: float | None = None
        for _next_index, (next_page, next_top) in positioned_skeletons[offset + 1 :]:
            if next_page == page:
                upper = next_top
                break
            if next_page > page:
                break
        fallback_segments[skeleton_index] = (page, top, upper)

    def owns(skeleton_index: int, page: int, top: float) -> bool:
        skeleton = skeletons[skeleton_index]
        segment = skeleton.get("_canonical_segment")
        pages = segment.get("pages") if isinstance(segment, Mapping) else None
        if isinstance(pages, list):
            for page_segment in pages:
                if not isinstance(page_segment, Mapping):
                    continue
                if int(page_segment.get("logical_page") or 0) != page:
                    continue
                minimum = page_segment.get("min_y")
                maximum = page_segment.get("max_y")
                if minimum is None:
                    return False
                try:
                    lower = float(minimum)
                    upper = float(maximum) if maximum is not None else None
                except (TypeError, ValueError):
                    return False
                return top + 8.0 >= lower and (upper is None or top < upper)
            return False
        fallback = fallback_segments.get(skeleton_index)
        if fallback is None:
            return False
        segment_page, lower, upper = fallback
        return page == segment_page and top + 8.0 >= lower and (upper is None or top < upper)

    positioned_tables = sorted(
        (
            (position, table_index)
            for table_index, table in enumerate(table_accounts)
            if (position := _account_stream_position(table)) is not None
        ),
        key=lambda item: item[0],
    )
    for (page, top), table_index in positioned_tables:
        candidates = [
            skeleton_index
            for skeleton_index in range(len(skeletons))
            if owns(skeleton_index, page, top)
        ]
        if len(candidates) != 1:
            continue
        skeleton_index = candidates[0]
        if skeleton_index in matches or table_index in consumed_tables:
            continue
        matches[skeleton_index] = table_index
        consumed_tables.add(table_index)
    return matches


def _extract_accounts(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize one account population from anchors enriched by tables."""
    table_accounts, repayments, events = _extract_table_accounts(parse_result)
    skeletons = _account_anchor_skeletons(parse_result)
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    if not skeletons:
        # Preserve otherwise useful canonical cells as explicitly partial
        # observations, but never promote table encounter order to an account
        # family identity when the printed anchors were not recovered.
        for table in table_accounts:
            table.pop("category_sequence", None)
            table["extraction_status"] = "review"
            table["_ownership_status"] = "printed_anchor_missing"
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_account_anchor_population_missing",
                    message=(
                        "Canonical account-table cells were retained as a partial observation, but no printed "
                        "account anchor was available to establish family identity or ordinal."
                    ),
                    parser_stage="candidate_b_account_anchor_ownership",
                    target_dataset="credit_accounts",
                    target_record_id=str(table.get("account_id") or "") or None,
                    field_name="category_sequence",
                    source_refs=table.get("source_refs") or (),
                    reason_codes=(
                        "printed_anchor_missing",
                        "stable_table_observation_identity",
                        "encounter_order_not_used",
                        "record_partial",
                    ),
                ),
            )
        return table_accounts, repayments, events

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
                "_printed_ordinal_status",
                "_canonical_segment",
                "page",
                "source_page",
                "bbox",
            ):
                if field_name in skeleton:
                    record[field_name] = deepcopy(skeleton[field_name])
                else:
                    record.pop(field_name, None)
            if skeleton.get("account_identifier"):
                anchor_refs = [
                    ref for ref in skeleton.get("source_refs") or () if isinstance(ref, Mapping)
                ]
                if anchor_refs:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="credit_accounts",
                        target_record_id=str(skeleton.get("account_id") or ""),
                        field_name="account_identifier",
                        value=str(skeleton["account_identifier"]),
                        raw=str(skeleton["account_identifier"]),
                        source_ref={
                            **dict(anchor_refs[0]),
                            "binding": "canonical_account_anchor",
                            "binding_quality": "canonical_account_anchor",
                        },
                        parser_stage="candidate_b_account_canonical_slots",
                    )
                elif not record.get("account_identifier") and "account_identifier" not in record.get(
                    "_unresolved_fields", []
                ):
                    record["account_identifier"] = skeleton["account_identifier"]
                if record.get("account_identifier"):
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
            record.pop("category_sequence", None)
            record["extraction_status"] = "review"
            record["_ownership_status"] = "printed_category_anchor_missing"
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
                target_record_id=(
                    str(table.get("account_id") or "") or None
                    if structurally_missing_category
                    else None
                ),
                observed_value={
                    "table_observation_id": table.get("_table_observation_id") or table.get("account_id"),
                    "account_type_candidate": account_type or None,
                },
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
        if account_type:
            emitted_by_family[account_type].append(account)
    for account_type, family_accounts in emitted_by_family.items():
        observed: list[int] = []
        unresolved_accounts: list[dict[str, Any]] = []
        for account in family_accounts:
            try:
                ordinal = int(account.get("category_sequence") or 0)
            except (TypeError, ValueError):
                ordinal = 0
            if ordinal > 0:
                observed.append(ordinal)
            else:
                unresolved_accounts.append(account)
        observed = sorted(set(observed))
        # PBOC category ordinals start at one and are dense. Bound the gap
        # expansion so a single OCR-joined outlier cannot manufacture hundreds
        # of supposed missing records; the outlier is still reported below.
        credible_ceiling = max(12, len(observed) * 3)
        bounded_endpoint = min(max(observed), credible_ceiling) if observed else 0
        missing = sorted(set(range(1, bounded_endpoint + 1)) - set(observed))
        outliers = [ordinal for ordinal in observed if ordinal > credible_ceiling]
        if not missing and not outliers and not unresolved_accounts:
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
                    "Printed account ordinals in one PBOC account family were not uniquely recoverable or "
                    "not dense; no ordinal or missing record was invented."
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
                    "unresolved_printed_ordinal_count": len(unresolved_accounts),
                    "provisional_account_ids": [
                        str(account.get("account_id") or "")
                        for account in unresolved_accounts
                        if account.get("account_id")
                    ],
                },
                source_refs=refs,
                reason_codes=(
                    (
                        "printed_category_ordinals_not_dense"
                        if missing or outliers
                        else "printed_category_ordinal_unresolved"
                    ),
                    *(
                        ("printed_category_ordinal_unresolved",)
                        if unresolved_accounts and (missing or outliers)
                        else ()
                    ),
                    "missing_records_not_invented",
                    "business_population_uncertain",
                ),
            ),
        )

    for account in emitted:
        if account.get("source") != "native_detail_account_table":
            continue
        source_absent = set(account.get("_source_absent_fields") or ())
        for field_name in ("management_institution", "account_identifier", "open_date", "currency"):
            if account.get(field_name) not in (None, "") or field_name in source_absent:
                continue
            _append_internal_field(account, "_unresolved_fields", field_name)
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_account_required_field_unresolved",
                    message="A required canonical account field was not safely decoded; the field remains withheld.",
                    parser_stage="candidate_b_account_canonical_slots",
                    target_dataset="credit_accounts",
                    target_record_id=str(account.get("account_id") or ""),
                    field_name=field_name,
                    source_refs=(
                        account.get("source_refs_by_field", {}).get(field_name)
                        or account.get("source_refs")
                        or ()
                    ),
                    reason_codes=(
                        "canonical_account_template",
                        "required_field_missing",
                        "normalized_value_withheld",
                    ),
                ),
            )

    account_identifiers = {
        str(account.get("account_id") or ""): account.get("account_identifier")
        for account in emitted
        if account.get("account_id")
    }
    accepted_table_account_ids = {
        str(table_accounts[index].get("account_id") or "")
        for index in consumed
        if 0 <= index < len(table_accounts)
    }
    accepted_table_account_ids.update(
        str(account.get("account_id") or "")
        for account in emitted
        if account.get("_ownership_status") == "printed_category_anchor_missing"
    )
    filtered_repayments: list[dict[str, Any]] = []
    filtered_events: list[dict[str, Any]] = []
    for related_record, target in [
        *((record, filtered_repayments) for record in repayments),
        *((record, filtered_events) for record in events),
    ]:
        prior_account_id = str(related_record.get("account_id") or "")
        if prior_account_id not in accepted_table_account_ids:
            continue
        canonical_account_id = account_id_remap.get(prior_account_id)
        if canonical_account_id:
            related_record["account_id"] = canonical_account_id
            if not related_record.get("account_identifier"):
                related_record["account_identifier"] = account_identifiers.get(canonical_account_id)
        target.append(related_record)
    return emitted, filtered_repayments, filtered_events


def _extract_credit_lines(parse_result: Any) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    field_names = {
        "授信协议标识": "account_identifier",
        "管理机构": "institution",
        "授信额度用途": "facility_type",
        "生效日期": "effective_date",
        "到期日期": "due_date",
        "授信额度": "total_limit",
        "授信限额": "credit_limit",
        "已用额度": "used_limit",
        "授信限额编号": "limit_identifier",
        "币种": "currency",
    }
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
        currency = _currency(_field(facts, "币种"))
        printed_sequence_raw = _field(facts, "__printed_sequence")
        printed_sequence = int(printed_sequence_raw) if str(printed_sequence_raw).isdigit() else None
        candidate_refs_by_field = getattr(candidate, "source_refs_by_field", {})
        candidate_bindings_by_field = getattr(candidate, "binding_quality_by_field", {})
        anchor_refs = [
            dict(ref)
            for ref in candidate_refs_by_field.get("__printed_sequence", ())
            if isinstance(ref, Mapping)
            and ref.get("binding") == "canonical_card_anchor"
        ] if isinstance(candidate_refs_by_field, Mapping) else []
        anchor_binding = (
            str(candidate_bindings_by_field.get("__printed_sequence") or "")
            if isinstance(candidate_bindings_by_field, Mapping)
            else ""
        )
        canonical_card_key = (
            f"credit_agreement:{printed_sequence}"
            if printed_sequence is not None
            and anchor_refs
            and anchor_binding == "canonical_card_anchor"
            else None
        )
        raw_limit_identifier = _field(facts, "授信限额编号")
        limit_identifier = _typed_identifier(raw_limit_identifier)
        if (
            raw_limit_identifier
            and not _agreement_source_absent(raw_limit_identifier)
            and not limit_identifier
        ):
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
        raw_due_compact = _compact(due_raw)
        source_refs_by_field: dict[str, list[dict[str, Any]]] = {}
        for label, refs in candidate_refs_by_field.items():
            field_name = field_names.get(str(label))
            if not field_name:
                continue
            source_refs_by_field[field_name] = [
                {**dict(ref), "field_name": field_name}
                for ref in refs
                if isinstance(ref, Mapping)
            ]
        binding_quality_by_field = {
            field_names[label]: quality
            for label, quality in candidate_bindings_by_field.items()
            if label in field_names
        }
        source_absent_fields = {
            field_names[label]
            for label in field_names
            if _agreement_source_absent(_field(facts, label))
        }
        unresolved_fields = {
            field_names[label]
            for label in getattr(candidate, "unresolved_labels", frozenset())
            if label in field_names
        }
        observed_fields = {
            field_names[label]
            for label in getattr(candidate, "observed_labels", frozenset())
            if label in field_names
        }
        record = {
                "credit_line_id": stable_record_id("credit_line", identifier),
                "_printed_sequence": printed_sequence,
                "_canonical_card_key": canonical_card_key,
                "_canonical_card_anchor_refs": anchor_refs,
                "account_identifier": identifier,
                "institution": _clean(_field(facts, "管理机构")),
                "facility_type": _clean(_field(facts, "授信额度用途")),
                "effective_date": _date(_field(facts, "生效日期")),
                "due_date": _date(due_raw),
                "validity_type": (
                    "perpetual"
                    if raw_due_compact == "长期"
                    else "fixed_term"
                    if _date(due_raw)
                    else "unknown"
                ),
                "total_limit": (
                    None
                    if _agreement_source_absent(_field(facts, "授信额度"))
                    else _number(_field(facts, "授信额度"))
                ),
                "credit_limit": (
                    None
                    if _agreement_source_absent(_field(facts, "授信限额"))
                    else _number(_field(facts, "授信限额"))
                ),
                "used_limit": (
                    None
                    if _agreement_source_absent(_field(facts, "已用额度"))
                    else _number(_field(facts, "已用额度"))
                ),
                "limit_identifier": limit_identifier,
                "currency": currency,
                "account_currency": currency,
                "reporting_amount_currency": currency,
                "amount_unit": "yuan",
                "reporting_amount_unit": "yuan",
                "source": "candidate_b_credit_agreement_schema",
                "source_refs": list(candidate.source_refs),
                "confidence": candidate.confidence,
                "source_refs_by_field": source_refs_by_field,
                "_field_binding_quality": binding_quality_by_field,
                "_source_absent_fields": sorted(source_absent_fields),
                "_unresolved_fields": sorted(unresolved_fields),
                "_observed_fields": sorted(observed_fields),
            }
        records.append(record)
    return records


def _agreement_source_absent(value: Any) -> bool:
    """Return whether an agreement cell explicitly prints only dash glyphs."""

    text = re.sub(r"\s+", "", str(value or ""))
    return bool(text and re.fullmatch(r"[-－‐‑‒–—―]+", text))


def _agreement_identifier_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _agreement_strong_field_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import normalize_institution_name

    matches = 0
    for field_name in (
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "currency",
    ):
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        if left_value in (None, "") or right_value in (None, ""):
            continue
        if field_name == "institution":
            equal = normalize_institution_name(str(left_value)) == normalize_institution_name(str(right_value))
        elif field_name in {"total_limit", "credit_limit", "used_limit"}:
            equal = str(left_value).replace(",", "") == str(right_value).replace(",", "")
        else:
            equal = _compact(left_value) == _compact(right_value)
        matches += int(equal)
    return matches


def _agreement_field_value_key(field_name: str, value: Any) -> str:
    if field_name == "institution":
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            normalize_institution_name,
        )
        return normalize_institution_name(str(value or ""))
    if field_name in {"total_limit", "credit_limit", "used_limit"}:
        parsed = _number(value)
        return str(parsed) if parsed is not None else ""
    return _compact(value)


def _agreement_printed_sequences_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_sequence = left.get("_printed_sequence")
    right_sequence = right.get("_printed_sequence")
    return bool(
        left_sequence not in (None, "")
        and right_sequence not in (None, "")
        and int(left_sequence) == int(right_sequence)
    )


def _agreement_single_leading_ocr_insertion(
    left_identifier: str,
    right_identifier: str,
) -> str | None:
    """Return the shorter exact identifier after one leading OCR insertion.

    This is deliberately not edit-distance matching: every character of the
    canonical candidate must be an exact suffix of the other observation.
    """

    longer, shorter = sorted(
        (left_identifier, right_identifier),
        key=lambda value: (len(value), value),
        reverse=True,
    )
    if (
        len(shorter) >= 20
        and len(longer) == len(shorter) + 1
        and longer[0].isalpha()
        and longer[1:] == shorter
    ):
        return shorter
    return None


def _agreement_same_source_page(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def pages(record: Mapping[str, Any]) -> set[int]:
        return {
            int(ref.get("source_page") or ref.get("logical_page") or 0)
            for ref in _agreement_observation_refs(record)
            if int(ref.get("source_page") or ref.get("logical_page") or 0) > 0
        }

    return bool(pages(left) & pages(right))


def _agreement_observation_refs(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs = [ref for ref in record.get("source_refs") or () if isinstance(ref, Mapping)]
    for field_refs in (record.get("source_refs_by_field") or {}).values():
        refs.extend(ref for ref in field_refs or () if isinstance(ref, Mapping))
    return refs


def _agreement_canonical_card_key(record: Mapping[str, Any]) -> str:
    key = str(record.get("_canonical_card_key") or "")
    sequence = record.get("_printed_sequence")
    if sequence in (None, "") or not str(sequence).isdigit():
        return ""
    expected = f"credit_agreement:{int(sequence)}"
    return key if key == expected else ""


def _agreement_anchor_planes(record: Mapping[str, Any]) -> set[str]:
    planes: set[str] = set()
    for ref in record.get("_canonical_card_anchor_refs") or ():
        if not isinstance(ref, Mapping) or ref.get("binding") != "canonical_card_anchor":
            continue
        source = str(ref.get("source") or "")
        if source.startswith("personal_detail_corrected_page"):
            planes.add("corrected_page")
        elif source.startswith("native_detail"):
            planes.add("native_table")
    return planes


def _agreement_anchor_geometry_overlaps(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_refs = [
        ref
        for ref in left.get("_canonical_card_anchor_refs") or ()
        if isinstance(ref, Mapping) and ref.get("binding") == "canonical_card_anchor"
    ]
    right_refs = [
        ref
        for ref in right.get("_canonical_card_anchor_refs") or ()
        if isinstance(ref, Mapping) and ref.get("binding") == "canonical_card_anchor"
    ]
    left_evidence = {
        str(evidence_id)
        for ref in left_refs
        for evidence_id in ref.get("evidence_ids") or ()
        if evidence_id
    }
    right_evidence = {
        str(evidence_id)
        for ref in right_refs
        for evidence_id in ref.get("evidence_ids") or ()
        if evidence_id
    }
    if left_evidence & right_evidence:
        return True
    for left_ref in left_refs:
        left_box = left_ref.get("bbox")
        left_page = int(left_ref.get("source_page") or left_ref.get("logical_page") or 0)
        if left_page <= 0 or not isinstance(left_box, (list, tuple)) or len(left_box) != 4:
            continue
        for right_ref in right_refs:
            right_box = right_ref.get("bbox")
            right_page = int(right_ref.get("source_page") or right_ref.get("logical_page") or 0)
            if right_page != left_page or not isinstance(right_box, (list, tuple)) or len(right_box) != 4:
                continue
            intersection = max(
                0.0,
                min(float(left_box[2]), float(right_box[2]))
                - max(float(left_box[0]), float(right_box[0])),
            ) * max(
                0.0,
                min(float(left_box[3]), float(right_box[3]))
                - max(float(left_box[1]), float(right_box[1])),
            )
            left_area = max(
                1.0,
                (float(left_box[2]) - float(left_box[0]))
                * (float(left_box[3]) - float(left_box[1])),
            )
            right_area = max(
                1.0,
                (float(right_box[2]) - float(right_box[0]))
                * (float(right_box[3]) - float(right_box[1])),
            )
            if intersection / min(left_area, right_area) >= 0.60:
                return True
    return False


def _agreement_authorized_cross_plane_anchors(
    records: list[dict[str, Any]],
) -> set[str]:
    """Return exact card anchors that are unique inside every evidence plane."""

    identifiers_by_key_and_plane: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        key = _agreement_canonical_card_key(record)
        identifier = _agreement_identifier_text(record.get("account_identifier"))
        if not key or not identifier:
            continue
        for plane in _agreement_anchor_planes(record):
            identifiers_by_key_and_plane[key][plane].add(identifier)
    return {
        key
        for key, identifiers_by_plane in identifiers_by_key_and_plane.items()
        if len(identifiers_by_plane) >= 2
        and all(len(identifiers) == 1 for identifiers in identifiers_by_plane.values())
    }


def _agreement_overlapping_business_conflicts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> set[str]:
    conflicts: set[str] = set()
    for field_name in (
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "currency",
    ):
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        if left_value in (None, "") or right_value in (None, ""):
            continue
        if _agreement_field_value_key(field_name, left_value) != _agreement_field_value_key(
            field_name,
            right_value,
        ):
            conflicts.add(field_name)
    return conflicts


def _agreement_has_strict_business_field_superset(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    field_names = {
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "currency",
    }
    left_fields = {field_name for field_name in field_names if left.get(field_name) not in (None, "")}
    right_fields = {field_name for field_name in field_names if right.get(field_name) not in (None, "")}
    return left_fields < right_fields or right_fields < left_fields


def _agreement_exact_anchor_authorizes_merge(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    authorized_anchors: set[str],
) -> bool:
    left_key = _agreement_canonical_card_key(left)
    right_key = _agreement_canonical_card_key(right)
    left_planes = _agreement_anchor_planes(left)
    right_planes = _agreement_anchor_planes(right)
    return bool(
        left_key
        and left_key == right_key
        and left_key in authorized_anchors
        and left_planes
        and right_planes
        and left_planes.isdisjoint(right_planes)
        and _agreement_strong_field_matches(left, right) >= 3
        and not _agreement_overlapping_business_conflicts(left, right)
        and _agreement_has_strict_business_field_superset(left, right)
    )


def _agreement_verified_observation_relation(
    issue_owner: Any,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Require shared canonical geometry or an explicit continuation edge.

    Page proximity and identifier resemblance are intentionally insufficient:
    several distinct agreements can share a page, institution, dates, and
    amount fields.  This relation only establishes that two observations came
    from the same canonical card region (or two tables explicitly joined by
    the plugin's continuation graph).
    """

    left_refs = _agreement_observation_refs(left)
    right_refs = _agreement_observation_refs(right)
    left_tables = {str(ref.get("table_id") or "") for ref in left_refs if ref.get("table_id")}
    right_tables = {str(ref.get("table_id") or "") for ref in right_refs if ref.get("table_id")}
    if left_tables & right_tables:
        return True

    continuation_check = getattr(issue_owner, "tables_continue", None)
    if callable(continuation_check) and any(
        continuation_check(left_table, right_table) is True
        for left_table in left_tables
        for right_table in right_tables
    ):
        return True

    for left_ref in left_refs:
        left_box = left_ref.get("bbox")
        if not isinstance(left_box, (list, tuple)) or len(left_box) != 4:
            continue
        left_page = int(left_ref.get("source_page") or left_ref.get("logical_page") or 0)
        for right_ref in right_refs:
            right_page = int(right_ref.get("source_page") or right_ref.get("logical_page") or 0)
            right_box = right_ref.get("bbox")
            if left_page <= 0 or left_page != right_page:
                continue
            if not isinstance(right_box, (list, tuple)) or len(right_box) != 4:
                continue
            intersection = max(0.0, min(float(left_box[2]), float(right_box[2])) - max(float(left_box[0]), float(right_box[0]))) * max(
                0.0, min(float(left_box[3]), float(right_box[3])) - max(float(left_box[1]), float(right_box[1]))
            )
            left_area = max(1.0, (float(left_box[2]) - float(left_box[0])) * (float(left_box[3]) - float(left_box[1])))
            right_area = max(1.0, (float(right_box[2]) - float(right_box[0])) * (float(right_box[3]) - float(right_box[1])))
            if intersection / min(left_area, right_area) >= 0.60:
                return True
    return False


def _agreement_field_provenance_quality(record: Mapping[str, Any], field_name: str) -> int:
    bindings = record.get("_field_binding_quality")
    binding = str(bindings.get(field_name) or "") if isinstance(bindings, Mapping) else ""
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    cell_scoped = any(
        isinstance(ref, Mapping)
        and ref.get("geometry_scope") == "cell"
        and ref.get("binding") in {"canonical_label_slot", "label_column"}
        for ref in refs or ()
    )
    if binding == "canonical_cell_slot" and cell_scoped:
        return 4
    if binding == "native_label_column" and cell_scoped:
        return 3
    if binding == "native_label_column":
        return 2
    # Backward-compatible synthetic observations remain comparable in unit
    # tests, but can never outrank a label-bound production observation.
    return 1 if record.get(field_name) not in (None, "") else 0


def _agreement_observation_quality(record: Mapping[str, Any]) -> tuple[int, int, int]:
    """Rank one agreement observation by schema validity before OCR confidence.

    Repeated observations of a card can contain a fully populated canonical row
    and a damaged continuation fragment.  Counting non-empty cells alone lets
    the fragment win when an OCR artifact happens to look numeric.  Candidate B
    therefore ranks observations by field contracts first and uses confidence
    only as a later tie-breaker.
    """
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        _is_valid_for_role,
    )

    role_fields = (
        ("account_identifier", "account_identifier"),
        ("institution", "institution_name"),
        ("facility_type", "facility_type"),
        ("effective_date", "date"),
        ("due_date", "date"),
        ("total_limit", "amount"),
        ("credit_limit", "amount"),
        ("used_limit", "amount"),
        ("currency", "currency"),
    )
    valid_fields = 0
    valid_identity_fields = 0
    provenance = 0
    for field_name, role in role_fields:
        value = record.get(field_name)
        if value in (None, ""):
            continue
        if _is_valid_for_role(str(value), role):
            valid_fields += 1
            provenance += _agreement_field_provenance_quality(record, field_name)
            if field_name in {"institution", "facility_type"}:
                valid_identity_fields += 1
    return provenance, valid_identity_fields, valid_fields


def reconcile_candidate_b_credit_lines(
    parse_result: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the one-row-per-agreement schema constraint after correction."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
        register_issue_target_remap,
    )

    authorized_cross_plane_anchors = _agreement_authorized_cross_plane_anchors(records)
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    prototypes: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(records):
        record = dict(source)
        if record.get("_printed_sequence") in (None, "") and record.get("sequence") not in (None, ""):
            record["_printed_sequence"] = record.get("sequence")
        issue_target_ids = {
            str(value)
            for value in (source.get("credit_line_id"), source.get("record_id"))
            if value
        }
        raw_identifier = _agreement_identifier_text(record.get("account_identifier"))
        identifier = _typed_identifier(record.get("account_identifier"))
        if identifier:
            record["account_identifier"] = identifier
            record["credit_line_id"] = stable_record_id("credit_line", identifier)
            issue_target_ids.add(str(record["credit_line_id"]))
        if issue_target_ids:
            record["_issue_target_ids"] = sorted(issue_target_ids)
        identity = str(record.get("credit_line_id") or f"candidate_b_credit_line_row:{index + 1}")
        if identity not in groups and raw_identifier:
            compatible: list[str] = []
            for candidate_identity, prototype in prototypes.items():
                prototype_identifier = _agreement_identifier_text(prototype.get("account_identifier"))
                strong_match_count = _agreement_strong_field_matches(record, prototype)
                sequence_match = _agreement_printed_sequences_match(record, prototype)
                verified_relation = _agreement_verified_observation_relation(
                    parse_result, prototype, record
                )
                leading_insertion_identifier = _agreement_single_leading_ocr_insertion(
                    raw_identifier,
                    prototype_identifier,
                )
                exact_leading_insertion_relation = bool(
                    leading_insertion_identifier
                    and sequence_match
                    and _agreement_same_source_page(prototype, record)
                    and strong_match_count >= 2
                )
                exact_anchor_relation = _agreement_exact_anchor_authorizes_merge(
                    prototype,
                    record,
                    authorized_cross_plane_anchors,
                )
                if (
                    raw_identifier != prototype_identifier
                    and (
                        (
                            sequence_match
                            and verified_relation
                            and strong_match_count >= 3
                        )
                        or exact_leading_insertion_relation
                        or exact_anchor_relation
                    )
                ):
                    if exact_leading_insertion_relation:
                        prototype["_identifier_leading_insertion_canonical"] = leading_insertion_identifier
                        record["_identifier_leading_insertion_canonical"] = leading_insertion_identifier
                    if exact_anchor_relation:
                        prototype["_canonical_card_anchor_verified"] = True
                        record["_canonical_card_anchor_verified"] = True
                    compatible.append(candidate_identity)
                    continue
                same_authorized_card_key = bool(
                    _agreement_canonical_card_key(record)
                    and _agreement_canonical_card_key(record)
                    == _agreement_canonical_card_key(prototype)
                )
                if raw_identifier != prototype_identifier and (
                    sequence_match
                    or same_authorized_card_key
                    or leading_insertion_identifier
                    or _agreement_anchor_geometry_overlaps(prototype, record)
                ):
                    record_issue(
                        parse_result,
                        make_issue(
                            category="schema_incompleteness",
                            issue_code="candidate_b_credit_agreement_identity_ambiguous",
                            message=(
                                "Two credit-agreement observations had different identifiers without the exact "
                                "ordinal-and-source relation required to prove one canonical card; both were retained."
                            ),
                            parser_stage="candidate_b_credit_agreement_schema",
                            target_dataset="credit_lines",
                            observed_value={
                                "left_identifier": prototype.get("account_identifier"),
                                "right_identifier": record.get("account_identifier"),
                                "left_sequence": prototype.get("_printed_sequence"),
                                "right_sequence": record.get("_printed_sequence"),
                            },
                            source_refs=[
                                *(prototype.get("source_refs") or ()),
                                *(record.get("source_refs") or ()),
                            ],
                            reason_codes=(
                                "different_identifiers",
                                "exact_card_identity_not_proven",
                                (
                                    "canonical_card_anchor_business_conflict"
                                    if same_authorized_card_key
                                    and _agreement_canonical_card_key(record)
                                    in authorized_cross_plane_anchors
                                    else "canonical_card_anchor_not_cross_plane_unique"
                                    if same_authorized_card_key
                                    else "canonical_card_anchor_unavailable"
                                ),
                                "fuzzy_identifier_merge_forbidden",
                                "records_conservatively_retained",
                            ),
                        ),
                    )
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
        "source_refs_by_field",
        "confidence",
        "credit_line_id",
        "sequence",
        "_printed_sequence",
        "_issue_target_ids",
        "_field_binding_quality",
        "_source_absent_fields",
        "_unresolved_fields",
        "_observed_fields",
        "_identifier_leading_insertion_canonical",
        "_canonical_card_key",
        "_canonical_card_anchor_refs",
        "_canonical_card_anchor_verified",
    }
    provenance_fields = (
        "account_identifier",
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "currency",
    )
    for identity in order:
        observations = groups[identity]
        ranked = sorted(
            observations,
            key=lambda row: (
                _agreement_observation_quality(row),
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
        selected.pop("_issue_target_ids", None)
        conflicts: set[str] = set()
        conflict_values: dict[str, list[Any]] = {}
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
                if key in provenance_fields:
                    continue
                retained = selected.get(key)
                if retained in (None, "", [], {}):
                    selected[key] = deepcopy(value)
                elif retained != value:
                    conflicts.add(key)

        selected_field_refs: dict[str, list[dict[str, Any]]] = {}
        selected_binding_quality: dict[str, str] = {}
        source_absent_fields = {
            str(field_name)
            for observation in observations
            for field_name in observation.get("_source_absent_fields") or ()
        }
        unresolved_fields = {
            str(field_name)
            for observation in observations
            for field_name in observation.get("_unresolved_fields") or ()
        }
        observed_fields = {
            str(field_name)
            for observation in observations
            for field_name in observation.get("_observed_fields") or ()
        }
        for field_name in provenance_fields:
            if field_name == "account_identifier":
                insertion_candidates = {
                    str(observation.get("_identifier_leading_insertion_canonical") or "")
                    for observation in observations
                    if observation.get("_identifier_leading_insertion_canonical")
                }
                if len(insertion_candidates) == 1:
                    canonical_identifier = next(iter(insertion_candidates))
                    selected[field_name] = canonical_identifier
                    chosen_observations = [
                        observation
                        for observation in observations
                        if _agreement_identifier_text(observation.get(field_name)) == canonical_identifier
                    ]
                    selected_field_refs[field_name] = [
                        dict(ref)
                        for observation in chosen_observations
                        for ref in (
                            (observation.get("source_refs_by_field") or {}).get(field_name)
                            if isinstance(observation.get("source_refs_by_field"), Mapping)
                            else ()
                        )
                        if isinstance(ref, Mapping)
                    ]
                    continue
            candidates = [
                (
                    _agreement_field_provenance_quality(observation, field_name),
                    float(observation.get("confidence") or 0.0),
                    observation.get(field_name),
                    observation,
                )
                for observation in observations
                if observation.get(field_name) not in (None, "")
            ]
            if not candidates:
                selected[field_name] = None
                continue
            best_quality = max(candidate[0] for candidate in candidates)
            best = [candidate for candidate in candidates if candidate[0] == best_quality]
            by_value: dict[str, list[tuple[int, float, Any, Mapping[str, Any]]]] = defaultdict(list)
            for candidate in best:
                by_value[_agreement_field_value_key(field_name, candidate[2])].append(candidate)
            if len(by_value) != 1:
                selected[field_name] = None
                unresolved_fields.add(field_name)
                conflicts.add(field_name)
                conflict_values[field_name] = [
                    candidate[2]
                    for candidate in sorted(best, key=lambda item: (item[1], str(item[2])), reverse=True)
                ]
                continue
            chosen = max(best, key=lambda item: (item[1], len(str(item[2]))))
            selected[field_name] = deepcopy(chosen[2])
            chosen_observations = by_value[next(iter(by_value))]
            field_refs: list[dict[str, Any]] = []
            field_ref_markers: set[str] = set()
            for _quality, _confidence, _value, observation in chosen_observations:
                refs_by_field = observation.get("source_refs_by_field")
                refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
                for ref in refs or ():
                    if not isinstance(ref, Mapping):
                        continue
                    marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                    if marker in field_ref_markers:
                        continue
                    field_ref_markers.add(marker)
                    field_refs.append(dict(ref))
            if field_refs:
                selected_field_refs[field_name] = field_refs
            binding_map = chosen[3].get("_field_binding_quality")
            if isinstance(binding_map, Mapping) and binding_map.get(field_name):
                selected_binding_quality[field_name] = str(binding_map[field_name])

        selected_identifier = _typed_identifier(selected.get("account_identifier"))
        if selected_identifier:
            selected["account_identifier"] = selected_identifier
            selected["credit_line_id"] = stable_record_id("credit_line", selected_identifier)
        else:
            selected["account_identifier"] = None
            selected["credit_line_id"] = stable_record_id(
                "credit_line_unresolved",
                next(iter(printed_sequences)) if len(printed_sequences) == 1 else identity,
                sorted(
                    {
                        _agreement_identifier_text(observation.get("account_identifier"))
                        for observation in observations
                        if _agreement_identifier_text(observation.get("account_identifier"))
                    }
                ),
            )
            unresolved_fields.add("account_identifier")
        final_target_id = str(selected["credit_line_id"])
        for observation in observations:
            for prior_target_id in observation.get("_issue_target_ids") or ():
                register_issue_target_remap(parse_result, prior_target_id, final_target_id)

        selected["source_refs_by_field"] = selected_field_refs
        selected["_field_binding_quality"] = selected_binding_quality
        selected["_source_absent_fields"] = sorted(source_absent_fields)
        selected["_unresolved_fields"] = sorted(unresolved_fields)
        selected["_observed_fields"] = sorted(observed_fields)
        currency = selected.get("currency")
        selected["account_currency"] = currency
        selected["reporting_amount_currency"] = currency
        due_date = selected.get("due_date")
        if due_date not in (None, ""):
            selected["validity_type"] = "fixed_term"
        elif any(
            _compact(observation.get("canonical_raw", {}).get("due_date") if isinstance(observation.get("canonical_raw"), Mapping) else "") == "长期"
            for observation in observations
        ):
            selected["validity_type"] = "perpetual"
        elif selected.get("validity_type") != "perpetual":
            selected["validity_type"] = "unknown"
        if merged_refs:
            selected["source_refs"] = merged_refs
        anchor_verified = bool(selected.pop("_canonical_card_anchor_verified", False))
        selected.pop("_canonical_card_key", None)
        selected.pop("_canonical_card_anchor_refs", None)
        reconciled.append(selected)
        identifier_variants = {
            _agreement_identifier_text(observation.get("account_identifier"))
            for observation in observations
            if _agreement_identifier_text(observation.get("account_identifier"))
        }
        if len(identifier_variants) > 1:
            leading_insertion_canonical = selected.get("_identifier_leading_insertion_canonical")
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_credit_agreement_identifier_variant",
                    message=(
                        "Source-verified observations of one agreement contained different identifiers; "
                        "only a uniquely stronger canonical field observation may be retained."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=str(selected.get("credit_line_id") or identity),
                    field_name="account_identifier",
                    observed_value=sorted(identifier_variants),
                    candidate_value=selected.get("account_identifier"),
                    source_refs=merged_refs,
                    reason_codes=(
                        (
                            "exact_canonical_card_anchor_cross_plane"
                            if anchor_verified
                            else "same_printed_sequence"
                            if len(printed_sequences) == 1
                            else "verified_source_relation"
                        ),
                        "different_identifier_observations",
                        (
                            "exact_leading_ocr_insertion_corrected"
                            if leading_insertion_canonical
                            else "canonical_anchor_provenance_selection"
                            if anchor_verified
                            else "fuzzy_identifier_selection_forbidden"
                        ),
                        (
                            "normalized_value_withheld"
                            if selected.get("account_identifier") in (None, "")
                            else "higher_provenance_value_retained_for_review"
                        ),
                    ),
                ),
            )
        if len(observations) > 1 and conflicts:
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_observation_conflict",
                    message=(
                        "Multiple corrected observations resolved to one canonical credit-agreement identity; "
                        "equally source-bound values disagreed; the conflicting fields were withheld for review."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=str(selected.get("credit_line_id") or identity),
                    observed_value={
                        "candidate_count": len(observations),
                        "conflicting_fields": sorted(conflicts),
                        "field_candidates": conflict_values,
                    },
                    source_refs=merged_refs,
                    reason_codes=(
                        "canonical_identity_collision",
                        "field_provenance_tie",
                        "conflicting_values_withheld",
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
    for record in reconciled:
        source_absent_fields = set(record.get("_source_absent_fields") or ())
        expected_fields = (
            "account_identifier",
            "institution",
            "facility_type",
            "effective_date",
            "due_date",
            "total_limit",
            "credit_limit",
            "used_limit",
            "limit_identifier",
            "currency",
        )
        for field_name in expected_fields:
            if record.get(field_name) not in (None, ""):
                continue
            if field_name in source_absent_fields:
                continue
            if field_name == "due_date" and record.get("validity_type") == "perpetual":
                continue
            refs_by_field = record.get("source_refs_by_field")
            field_refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_required_field_unresolved",
                    message=(
                        "A required credit-agreement field remained unavailable after all canonical "
                        "observations were reconciled; no business value was invented."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=str(record.get("credit_line_id") or "") or None,
                    field_name=field_name,
                    source_refs=field_refs or record.get("source_refs") or (),
                    reason_codes=(
                        "required_field_missing",
                        "canonical_credit_agreement_field_unresolved",
                        "field_slot_not_safely_bound",
                        "preserved_unknown_value",
                    ),
                ),
            )
    return reconciled


_LIABILITY_LABEL_TO_FIELD = {
    "管理机构": "institution",
    "业务种类": "business_type",
    "开立日期": "open_date",
    "到期日期": "due_date",
    "责任人类型": "responsibility_type",
    "还款责任金额": "responsibility_amount",
    "币种": "currency",
    "保证合同编号": "contract_number",
    "主业务借款人": "related_party_name",
    "主业务借款人证件类型": "related_party_id_type",
    "主业务借款人证件号码": "related_party_id_number",
    "__snapshot_date": "snapshot_date",
    "余额": "balance",
    "五级分类": "five_tier_class",
    "逾期月数": "overdue_months_or_repayment_status",
    "还款状态": "overdue_months_or_repayment_status",
}
_LIABILITY_PLACEHOLDERS = frozenset({"", "-", "--", "---", "未报告", "不详"})
_LIABILITY_RESPONSIBILITY_TYPES = frozenset(
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
)
_LIABILITY_ID_TYPES = frozenset(
    {
        "身份证",
        "居民身份证",
        "统一社会信用代码",
        "中征码",
        "组织机构代码",
        "护照",
        "军官证",
        "士兵证",
        "其他证件",
        "未知",
    }
)
_LIABILITY_FIVE_TIER_CLASSES = frozenset({"正常", "关注", "次级", "可疑", "损失", "未分类", "未知"})


def _liability_source_absent(value: Any) -> bool:
    return _compact(value) in _LIABILITY_PLACEHOLDERS


def _liability_currency(value: Any) -> str | None:
    compact = _compact(value).upper()
    aliases = {
        "人民币": "CNY",
        "人民币元": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "美元": "USD",
        "USD": "USD",
        "欧元": "EUR",
        "EUR": "EUR",
        "港元": "HKD",
        "HKD": "HKD",
        "日元": "JPY",
        "JPY": "JPY",
        "英镑": "GBP",
        "GBP": "GBP",
    }
    return aliases.get(compact)


def _liability_date(value: Any) -> str | None:
    match = re.search(r"((?:19|20)\d{2})[.年/-]\s*(\d{1,2})[.月/-]\s*(\d{1,2})", _compact(value))
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}" if 1 <= month <= 12 and 1 <= day <= 31 else None


def _liability_identifier(value: Any, *, contract: bool = False) -> str | None:
    compact = re.sub(r"\s+", "", str(value or "")).upper()
    minimum = 8 if contract else 6
    if not (minimum <= len(compact) <= 100):
        return None
    if not re.fullmatch(r"[A-Z0-9*\-]+", compact):
        return None
    if contract and not re.search(r"[A-Z0-9]", compact):
        return None
    return compact


def _liability_text(value: Any, *, minimum: int = 1, maximum: int = 120) -> str | None:
    text = _business_text(value)
    if not (minimum <= len(text) <= maximum):
        return None
    if not re.search(r"[\u3400-\u9fffA-Za-z]", text):
        return None
    if re.search(r"[\"'?？]{2,}", text):
        return None
    return text


def _liability_convert(field_name: str, raw_value: Any) -> Any | None:
    compact = _compact(raw_value)
    if _liability_source_absent(raw_value):
        return None
    if field_name in {"responsibility_amount", "balance"}:
        parsed = _number(raw_value)
        return parsed if isinstance(parsed, int) and parsed >= 0 else None
    if field_name in {"open_date", "due_date", "snapshot_date"}:
        return _liability_date(raw_value)
    if field_name == "currency":
        return _liability_currency(raw_value)
    if field_name == "contract_number":
        return _liability_identifier(raw_value, contract=True)
    if field_name == "related_party_id_number":
        return _liability_identifier(raw_value)
    if field_name == "responsibility_type":
        return compact if compact in _LIABILITY_RESPONSIBILITY_TYPES else None
    if field_name == "related_party_id_type":
        return compact if compact in _LIABILITY_ID_TYPES else None
    if field_name == "five_tier_class":
        return compact if compact in _LIABILITY_FIVE_TIER_CLASSES else None
    if field_name == "overdue_months_or_repayment_status":
        upper = compact.upper()
        if upper in _STATUS_CODES or re.fullmatch(r"\d{1,2}", upper):
            return upper
        return compact if compact in {"正常", "逾期", "结清", "未知"} else None
    if field_name == "related_party_name":
        return _liability_text(raw_value, minimum=2)
    if field_name == "institution":
        return _liability_text(raw_value, minimum=2)
    if field_name == "business_type":
        return _liability_text(raw_value, minimum=2, maximum=60)
    return None


def _liability_party_category(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("_party_category") or "")
    if explicit in {"person", "organization"}:
        return explicit
    id_type = _compact(record.get("related_party_id_type"))
    identifier = _compact(record.get("related_party_id_number")).upper()
    if "身份" in id_type or re.fullmatch(r"\d{17}[0-9X]", identifier):
        return "person"
    if any(marker in id_type for marker in ("统一社会信用", "中征码", "组织机构")):
        return "organization"
    return "unknown"


def _liability_field_provenance_quality(record: Mapping[str, Any], field_name: str) -> int:
    bindings = record.get("_field_binding_quality")
    binding = str(bindings.get(field_name) or "") if isinstance(bindings, Mapping) else ""
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    cell_scoped = any(
        isinstance(ref, Mapping) and ref.get("geometry_scope") == "cell" for ref in refs or ()
    )
    if binding in {"canonical_cell_slot", "canonical_snapshot_date_cell"} and cell_scoped:
        return 4
    if binding == "native_label_column" and cell_scoped:
        return 3
    if binding in {"native_label_column", "canonical_snapshot_date_cell"}:
        return 2
    return 1 if record.get(field_name) not in (None, "") else 0


def _liability_value_key(field_name: str, value: Any) -> str:
    if field_name in {"responsibility_amount", "balance"}:
        return str(_number(value))
    return _compact(value).upper()


def _liability_records_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_contract = _compact(left.get("contract_number")).upper()
    right_contract = _compact(right.get("contract_number")).upper()
    if left_contract and right_contract and left_contract == right_contract:
        return True
    left_sequence = left.get("_printed_sequence")
    right_sequence = right.get("_printed_sequence")
    left_category = _liability_party_category(left)
    right_category = _liability_party_category(right)
    return bool(
        str(left_sequence or "").isdigit()
        and str(right_sequence or "").isdigit()
        and int(left_sequence) == int(right_sequence)
        and left_category != "unknown"
        and left_category == right_category
    )


def _liability_field_refs(record: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    return [dict(ref) for ref in refs or () if isinstance(ref, Mapping)]


def reconcile_candidate_b_liabilities(
    parse_result: Any,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconcile fixed-layout liability cards without fuzzy business matching."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    groups: list[list[dict[str, Any]]] = []
    for observation in observations:
        compatible = [
            group for group in groups if any(_liability_records_compatible(observation, prior) for prior in group)
        ]
        if len(compatible) == 1:
            compatible[0].append(observation)
        else:
            # Ambiguous bridges are never used to collapse business records.
            groups.append([observation])

    reconciled: list[dict[str, Any]] = []
    for output_sequence, group in enumerate(groups, start=1):
        merged_refs: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for observation in group:
            for ref in observation.get("source_refs") or ():
                if not isinstance(ref, Mapping):
                    continue
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    merged_refs.append(dict(ref))

        printed_sequences = {
            int(observation["_printed_sequence"])
            for observation in group
            if str(observation.get("_printed_sequence") or "").isdigit()
        }
        group_contracts = {
            str(observation.get("contract_number") or "")
            for observation in group
            if observation.get("contract_number") not in (None, "")
        }
        group_categories = {
            _liability_party_category(observation)
            for observation in group
            if _liability_party_category(observation) != "unknown"
        }
        identity_seed: Any = next(iter(group_contracts)) if len(group_contracts) == 1 else None
        if identity_seed is None and len(printed_sequences) == 1 and len(group_categories) == 1:
            identity_seed = f"{next(iter(group_categories))}:{next(iter(printed_sequences))}"
        if identity_seed is None:
            identity_seed = json.dumps(merged_refs, ensure_ascii=False, sort_keys=True, default=str)
        liability_id = stable_record_id("repayment_liability", identity_seed or output_sequence)
        selected: dict[str, Any] = {
            "liability_id": liability_id,
            "sequence": output_sequence,
            "source": "candidate_b_repayment_responsibility_schema",
            "source_refs": merged_refs,
            "confidence": max(float(observation.get("confidence") or 0.0) for observation in group),
            "amount_unit": "yuan",
            "reporting_amount_unit": "yuan",
        }
        selected_field_refs: dict[str, list[dict[str, Any]]] = {}
        selected_bindings: dict[str, str] = {}
        selected_raw: dict[str, Any] = {}
        source_absent_fields = {
            str(field_name)
            for observation in group
            for field_name in observation.get("_source_absent_fields") or ()
        }
        unresolved_fields = {
            str(field_name)
            for observation in group
            for field_name in observation.get("_unresolved_fields") or ()
        }
        source_absent_quality = {
            field_name: max(
                (
                    _liability_field_provenance_quality(observation, field_name)
                    for observation in group
                    if field_name in set(observation.get("_source_absent_fields") or ())
                ),
                default=0,
            )
            for field_name in source_absent_fields
        }
        unresolved_quality = {
            field_name: max(
                (
                    _liability_field_provenance_quality(observation, field_name)
                    for observation in group
                    if field_name in set(observation.get("_unresolved_fields") or ())
                ),
                default=0,
            )
            for field_name in unresolved_fields
        }
        invalid_raw: dict[str, list[Any]] = defaultdict(list)
        for observation in group:
            raw_map = observation.get("_invalid_raw_by_field")
            if not isinstance(raw_map, Mapping):
                continue
            for field_name, values in raw_map.items():
                invalid_raw[str(field_name)].extend(values if isinstance(values, list) else [values])

        conflict_fields: set[str] = set()
        for field_name in _LIABILITY_CANONICAL_FIELDS:
            candidates = [
                (
                    _liability_field_provenance_quality(observation, field_name),
                    float(observation.get("confidence") or 0.0),
                    observation.get(field_name),
                    observation,
                )
                for observation in group
                if observation.get(field_name) not in (None, "")
            ]
            if not candidates:
                selected[field_name] = None
                continue
            best_quality = max(candidate[0] for candidate in candidates)
            best = [candidate for candidate in candidates if candidate[0] == best_quality]
            by_value: dict[str, list[tuple[int, float, Any, Mapping[str, Any]]]] = defaultdict(list)
            for candidate in best:
                by_value[_liability_value_key(field_name, candidate[2])].append(candidate)
            if len(by_value) != 1:
                selected[field_name] = None
                unresolved_fields.add(field_name)
                conflict_fields.add(field_name)
                conflict_refs = [
                    ref
                    for _quality, _confidence, _value, observation in best
                    for ref in _liability_field_refs(observation, field_name)
                ]
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_repayment_responsibility_field_conflict",
                        message=(
                            "Equally source-bound observations disagreed for one canonical repayment-"
                            "responsibility field; the value was withheld instead of guessed."
                        ),
                        parser_stage="candidate_b_repayment_responsibility_schema",
                        target_dataset="repayment_liability_records",
                        target_record_id=liability_id,
                        field_name=field_name,
                        observed_value=[candidate[2] for candidate in best],
                        source_refs=conflict_refs or merged_refs,
                        reason_codes=(
                            "closed_canonical_liability_slot",
                            "equal_provenance_conflict",
                            "conflicting_value_withheld",
                        ),
                    ),
                )
                continue
            chosen = max(best, key=lambda candidate: candidate[1])
            selected[field_name] = deepcopy(chosen[2])
            chosen_group = by_value[next(iter(by_value))]
            field_refs: list[dict[str, Any]] = []
            field_ref_markers: set[str] = set()
            for _quality, _confidence, _value, observation in chosen_group:
                for ref in _liability_field_refs(observation, field_name):
                    marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                    if marker not in field_ref_markers:
                        field_ref_markers.add(marker)
                        field_refs.append(ref)
            if field_refs:
                selected_field_refs[field_name] = field_refs
            bindings = chosen[3].get("_field_binding_quality")
            if isinstance(bindings, Mapping) and bindings.get(field_name):
                selected_bindings[field_name] = str(bindings[field_name])
            raw_values = chosen[3].get("canonical_raw")
            if isinstance(raw_values, Mapping) and field_name in raw_values:
                selected_raw[field_name] = deepcopy(raw_values[field_name])

        selected["responsibility_amount_reported"] = isinstance(selected.get("responsibility_amount"), int)
        selected["reporting_amount_currency"] = selected.get("currency")
        selected["source_refs_by_field"] = selected_field_refs
        selected["_field_binding_quality"] = selected_bindings
        selected["_source_absent_fields"] = sorted(source_absent_fields)
        selected["_unresolved_fields"] = sorted(unresolved_fields)
        if selected_raw:
            selected["canonical_raw"] = selected_raw

        for field_name in _LIABILITY_CANONICAL_FIELDS:
            if selected.get(field_name) not in (None, ""):
                continue
            explicit_absence_is_best_evidence = bool(
                field_name in source_absent_fields
                and source_absent_quality.get(field_name, 0) >= unresolved_quality.get(field_name, 0)
            )
            if explicit_absence_is_best_evidence or field_name in conflict_fields:
                continue
            field_invalid_raw = list(dict.fromkeys(invalid_raw.get(field_name) or ()))
            field_refs = selected_field_refs.get(field_name) or [
                ref
                for observation in group
                for ref in _liability_field_refs(observation, field_name)
            ]
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error" if field_invalid_raw else "schema_incompleteness",
                    issue_code=(
                        "candidate_b_repayment_responsibility_field_invalid"
                        if field_invalid_raw
                        else "candidate_b_repayment_responsibility_required_field_unresolved"
                    ),
                    message=(
                        "A canonical repayment-responsibility field failed its field contract and was withheld."
                        if field_invalid_raw
                        else "A fixed-layout repayment-responsibility field remained unresolved after correction."
                    ),
                    parser_stage="candidate_b_repayment_responsibility_schema",
                    target_dataset="repayment_liability_records",
                    target_record_id=liability_id,
                    field_name=field_name,
                    observed_value=field_invalid_raw or None,
                    source_refs=field_refs or merged_refs,
                    reason_codes=(
                        "closed_canonical_liability_slot",
                        "field_contract_failed" if field_invalid_raw else "required_field_missing",
                        "value_withheld_not_invented",
                    ),
                ),
            )
        reconciled.append(selected)
    return reconciled


def _extract_liabilities(parse_result: Any) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    observations: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(
        PBOCPersonalDetailNativeParser(parse_result).records("repayment_liability_records"), start=1
    ):
        facts = candidate.fields
        observation: dict[str, Any] = {
            "_printed_sequence": facts.get("__printed_sequence"),
            "_party_category": facts.get("__party_category"),
            "source": "candidate_b_repayment_responsibility_observation",
            "source_refs": list(candidate.source_refs),
            "confidence": candidate.confidence,
            "source_refs_by_field": {},
            "_field_binding_quality": {},
            "_source_absent_fields": [],
            "_unresolved_fields": [],
            "_invalid_raw_by_field": {},
            "canonical_raw": {},
        }
        source_absent_fields: set[str] = set()
        unresolved_fields = {
            _LIABILITY_LABEL_TO_FIELD[label]
            for label in candidate.unresolved_labels
            if label in _LIABILITY_LABEL_TO_FIELD
        }
        for label, field_name in _LIABILITY_LABEL_TO_FIELD.items():
            raw_value = facts.get(label)
            if raw_value in (None, ""):
                continue
            observation["canonical_raw"][field_name] = raw_value
            if _liability_source_absent(raw_value):
                source_absent_fields.add(field_name)
                observation[field_name] = None
            else:
                converted = _liability_convert(field_name, raw_value)
                observation[field_name] = converted
                if converted is None:
                    unresolved_fields.add(field_name)
                    observation["_invalid_raw_by_field"].setdefault(field_name, []).append(raw_value)

            raw_refs = candidate.source_refs_by_field.get(label) or ()
            if raw_refs:
                observation["source_refs_by_field"].setdefault(field_name, []).extend(
                    dict(ref) for ref in raw_refs if isinstance(ref, Mapping)
                )
            binding = candidate.binding_quality_by_field.get(label)
            if binding:
                observation["_field_binding_quality"][field_name] = binding

        observation["_source_absent_fields"] = sorted(source_absent_fields)
        observation["_unresolved_fields"] = sorted(unresolved_fields)
        has_identity = observation.get("contract_number") not in (None, "")
        has_amount = isinstance(observation.get("responsibility_amount"), int)
        has_sequence = str(observation.get("_printed_sequence") or "").isdigit()
        if not (has_identity or has_amount or has_sequence):
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_repayment_responsibility_identity_unresolved",
                    message=(
                        "A canonical repayment-responsibility card was observed but no contract, printed "
                        "ordinal, or valid responsibility amount could identify it."
                    ),
                    parser_stage="candidate_b_repayment_responsibility_schema",
                    target_dataset="repayment_liability_records",
                    observed_value=observation.get("_invalid_raw_by_field") or None,
                    source_refs=candidate.source_refs,
                    reason_codes=(
                        "canonical_responsibility_card",
                        "record_identity_unresolved",
                        "record_withheld",
                    ),
                ),
            )
            continue
        # Keep observations separate until exact contract/ordinal reconciliation.
        observation["_observation_index"] = candidate_index
        observations.append(observation)
    return reconcile_candidate_b_liabilities(parse_result, observations)


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
                "geometry_scope": "row",
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
                    "credit_inquiry", inquiry_type, sequence
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
    return bool(left_institution) and left_institution == right_institution


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
    refs_by_field = record.get("source_refs_by_field")
    exact_fields = sum(
        any(
            isinstance(ref, Mapping)
            and ref.get("geometry_scope") == "cell"
            and ref.get("binding") == "canonical_header_column"
            for ref in refs_by_field.get(field_name) or ()
        )
        for field_name in ("inquiry_date", "institution", "reason")
    ) if isinstance(refs_by_field, Mapping) else 0
    return exact_fields * 10 + valid, populated, float(record.get("confidence") or 0.0)


def _extract_inquiries(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    active_canonical_table = False
    active_slots: dict[str, int] = {}
    inquiry_aliases = {
        "sequence": ("编号", "序号"),
        "inquiry_date": ("查询日期",),
        "institution": ("查询机构",),
        "reason": ("查询原因",),
    }
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
            header_slots = _canonical_header_slots(tuple(str(value or "") for value in rows[0]), inquiry_aliases)
            has_exact_header = has_header and set(header_slots) == set(inquiry_aliases)
            is_continuation = (
                not has_header
                and page_is_canonical_inquiry
                and active_canonical_table
                and set(active_slots) == set(inquiry_aliases)
                and bool(_nonempty(rows[0]))
                and re.fullmatch(r"\d{1,4}", _nonempty(rows[0])[0]) is not None
            )
            if has_header and not has_exact_header:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_inquiry_header_columns_unresolved",
                        message="The canonical inquiry header was visible but its four fields did not resolve to distinct columns.",
                        parser_stage="candidate_b_inquiry_schema",
                        target_dataset="inquiry_records",
                        observed_value={"header": rows[0], "resolved_roles": sorted(header_slots)},
                        source_refs=(_source_ref(page, table, row=0),),
                        reason_codes=(
                            "closed_canonical_inquiry_template",
                            "distinct_header_columns_required",
                            "rows_withheld_until_page_repair",
                        ),
                    ),
                )
                active_canonical_table = False
                active_slots = {}
                continue
            if not has_exact_header and not is_continuation:
                continue
            page_had_inquiry_table = True
            active_canonical_table = True
            if has_exact_header:
                active_slots = header_slots
            slots = dict(active_slots)
            start = 1 if has_exact_header else 0
            for row_index, row in enumerate(rows[start:], start=start):
                cells = tuple(str(value or "").strip() for value in row)
                raw_sequence = _slot_value(cells, slots, "sequence")
                sequence_match = re.fullmatch(r"\D*(\d{1,4})\D*", raw_sequence)
                if sequence_match is None:
                    continue
                sequence = int(sequence_match.group(1))
                date_cell = _slot_value(cells, slots, "inquiry_date")
                date_match = re.search(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}", date_cell)
                inquiry_date = _date(date_match.group(0)) if date_match is not None else None
                institution = _slot_value(cells, slots, "institution")
                source_reason = _slot_value(cells, slots, "reason")
                if not institution or not source_reason or inquiry_date is None:
                    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                        make_issue,
                        record_issue,
                    )

                    missing_fields = [
                        field_name
                        for field_name, value in (
                            ("inquiry_date", inquiry_date),
                            ("institution", institution),
                            ("reason", source_reason),
                        )
                        if value in (None, "")
                    ]
                    record_issue(
                        parse_result,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_inquiry_row_cells_unresolved",
                            message="A printed inquiry row could not be assigned completely to the canonical four-column schema.",
                            parser_stage="candidate_b_inquiry_schema",
                            target_dataset="inquiry_records",
                            observed_value={"sequence": sequence, "row": list(cells)},
                            candidate_value={"missing_fields": missing_fields},
                            source_refs=(_source_ref(page, table, row=row_index),),
                            reason_codes=(
                                "printed_inquiry_sequence_observed",
                                "exact_header_column_binding_failed",
                                "record_not_invented",
                            ),
                        ),
                    )
                    continue
                inquiry_type = (
                    "personal" if institution == "本人" or source_reason.startswith("本人查询") else "institution"
                )
                inquiry_id = stable_record_id("credit_inquiry", inquiry_type, sequence)
                refs_by_field = {
                    field_name: [
                        {
                            **_source_ref(page, table, row=row_index, column=slots[field_name]),
                            "field_name": field_name,
                            "binding": "canonical_header_column",
                        }
                    ]
                    for field_name in ("inquiry_date", "institution", "reason")
                }
                records.append(
                    {
                        "inquiry_id": inquiry_id,
                        "sequence": sequence,
                        "inquiry_date": inquiry_date,
                        "institution": institution,
                        "reason": source_reason,
                        "source_reason": source_reason,
                        "query_channel": "personal" if inquiry_type == "personal" else "institution",
                        "inquiry_type": inquiry_type,
                        "source": "native_detail_inquiry_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "source_refs_by_field": refs_by_field,
                        "confidence": float(getattr(table, "confidence", None) or 0.9),
                    }
                )
        if not page_is_canonical_inquiry and not page_had_inquiry_table:
            active_canonical_table = False
            active_slots = {}
    records.extend(_canonical_inquiry_line_rows(parse_result))
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    for record in records:
        key = (str(record.get("inquiry_type") or ""), int(record.get("sequence") or 0))
        if key[0] and key[1] > 0:
            grouped[key].append(record)

    best: dict[tuple[str, int], dict[str, Any]] = {}
    for key, observations in grouped.items():
        selected = deepcopy(max(observations, key=_inquiry_observation_score))
        selected["inquiry_id"] = stable_record_id("credit_inquiry", key[0], key[1])
        merged_refs: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for observation in observations:
            for ref in observation.get("source_refs") or ():
                if not isinstance(ref, Mapping):
                    continue
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    merged_refs.append(dict(ref))
        selected["source_refs"] = merged_refs
        selected_refs_by_field: dict[str, list[dict[str, Any]]] = {}
        for field_name in ("inquiry_date", "institution", "reason"):
            candidates: list[tuple[int, float, Any, Mapping[str, Any]]] = []
            for observation in observations:
                value = observation.get(field_name)
                if value in (None, ""):
                    continue
                refs_by_field = observation.get("source_refs_by_field")
                refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
                exact = any(
                    isinstance(ref, Mapping)
                    and ref.get("geometry_scope") == "cell"
                    and ref.get("binding") == "canonical_header_column"
                    for ref in refs or ()
                )
                candidates.append((2 if exact else 1, float(observation.get("confidence") or 0.0), value, observation))
            if not candidates:
                selected[field_name] = None
                continue
            top_quality = max(candidate[0] for candidate in candidates)
            top = [candidate for candidate in candidates if candidate[0] == top_quality]
            normalized: defaultdict[str, list[tuple[int, float, Any, Mapping[str, Any]]]] = defaultdict(list)
            for candidate in top:
                normalized[_normalized_inquiry_field(field_name, candidate[2])].append(candidate)
            if len(normalized) > 1:
                selected[field_name] = None
                selected["extraction_status"] = "review"
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_inquiry_field_conflict",
                        message="Equally source-bound observations disagree for one canonical inquiry field; the value was withheld.",
                        parser_stage="candidate_b_inquiry_schema",
                        target_dataset="inquiry_records",
                        target_record_id=selected["inquiry_id"],
                        field_name=field_name,
                        observed_value=[candidate[2] for candidate in top],
                        source_refs=merged_refs,
                        reason_codes=(
                            "same_printed_inquiry_row",
                            "equal_field_provenance",
                            "conflicting_values_withheld",
                        ),
                    ),
                )
                continue
            chosen = max(top, key=lambda candidate: (candidate[1], len(str(candidate[2]))))
            selected[field_name] = chosen[2]
            for _quality, _confidence, _value, observation in normalized[next(iter(normalized))]:
                refs_by_field = observation.get("source_refs_by_field")
                for ref in refs_by_field.get(field_name) or () if isinstance(refs_by_field, Mapping) else ():
                    if isinstance(ref, Mapping):
                        selected_refs_by_field.setdefault(field_name, []).append(dict(ref))
        if selected_refs_by_field:
            selected["source_refs_by_field"] = selected_refs_by_field
        best[key] = selected
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
                target_record_id=str(suffix_record.get("inquiry_id") or ""),
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


_PUBLIC_CANONICAL_LAYOUTS: tuple[dict[str, Any], ...] = (
    {
        "name": "tax_arrears",
        "record_type": "tax_arrears",
        "signature": ("tax_authority", "arrears_amount"),
        "aliases": {
            "sequence": ("编号",),
            "tax_authority": ("主管税务机关",),
            "arrears_amount": ("欠税总额",),
            "statistics_date": ("欠税统计日期",),
        },
        "fields": {
            "tax_authority": ("tax_authority", "text"),
            "arrears_amount": ("arrears_amount", "number"),
            "statistics_date": ("statistics_date", "date"),
        },
    },
    {
        "name": "civil_judgment_base",
        "record_type": "civil_judgment",
        "signature": ("filing_court", "closure_method"),
        "aliases": {
            "sequence": ("编号",),
            "filing_court": ("立案法院",),
            "cause": ("案由",),
            "filing_date": ("立案日期",),
            "closure_method": ("结案方式",),
        },
        "fields": {
            "filing_court": ("filing_court", "text"),
            "cause": ("cause", "text"),
            "filing_date": ("filing_date", "date"),
            "closure_method": ("closure_method", "text"),
        },
    },
    {
        "name": "civil_judgment_detail",
        "record_type": "civil_judgment",
        "signature": ("judgment_result", "claim_amount"),
        "aliases": {
            "sequence": ("编号",),
            "judgment_result": ("判决/调解结果", "判决／调解结果"),
            "judgment_effective_date": ("判决/调解生效日期", "判决／调解生效日期"),
            "claim_subject": ("诉讼标的",),
            "claim_amount": ("诉讼标的金额",),
        },
        "fields": {
            "judgment_result": ("judgment_result", "text"),
            "judgment_effective_date": ("judgment_effective_date", "date"),
            "claim_subject": ("claim_subject", "text"),
            "claim_amount": ("claim_amount", "number"),
        },
    },
    {
        "name": "enforcement_base",
        "record_type": "enforcement",
        "signature": ("court", "closure_method"),
        "aliases": {
            "sequence": ("编号",),
            "court": ("执行法院",),
            "cause": ("执行案由", "案由"),
            "filing_date": ("立案日期",),
            "closure_method": ("结案方式",),
        },
        "fields": {
            "court": ("court", "text"),
            "cause": ("cause", "text"),
            "filing_date": ("filing_date", "date"),
            "closure_method": ("closure_method", "text"),
        },
    },
    {
        "name": "enforcement_detail",
        "record_type": "enforcement",
        "signature": ("case_status", "requested_subject", "executed_subject"),
        "aliases": {
            "sequence": ("编号",),
            "case_status": ("案件状态",),
            "closure_date": ("结案日期",),
            "requested_subject": ("申请执行标的",),
            "requested_amount": ("申请执行标的金额", "申请执行标的的价值"),
            "executed_subject": ("已执行标的",),
            "executed_amount": ("已执行标的金额",),
        },
        "fields": {
            "case_status": ("case_status", "text"),
            "closure_date": ("closure_date", "date"),
            "requested_subject": ("requested_subject", "text"),
            "requested_amount": ("requested_amount", "number"),
            "executed_subject": ("executed_subject", "text"),
            "executed_amount": ("executed_amount", "number"),
        },
    },
    {
        "name": "administrative_penalty",
        "record_type": "administrative_penalty",
        "signature": ("authority", "penalty_content", "penalty_amount"),
        "aliases": {
            "sequence": ("编号",),
            "authority": ("处罚机构",),
            "penalty_content": ("处罚内容",),
            "penalty_amount": ("处罚金额",),
            "effective_date": ("生效日期",),
            "end_date": ("截止日期",),
            "administrative_review_result": ("行政复议结果",),
        },
        "fields": {
            "authority": ("authority", "text"),
            "penalty_content": ("penalty_content", "text"),
            "penalty_amount": ("penalty_amount", "number"),
            "effective_date": ("effective_date", "date"),
            "end_date": ("end_date", "date"),
            "administrative_review_result": ("administrative_review_result", "text"),
        },
    },
    {
        "name": "housing_fund_base",
        "record_type": "housing_fund",
        "signature": ("contribution_location", "monthly_contribution"),
        "sequence_optional": True,
        "record_boundary": "start",
        "aliases": {
            "contribution_location": ("参缴地",),
            "participation_date": ("参缴日期",),
            "first_contribution_month": ("初缴月份", "初缴日期"),
            "paid_through_month": ("缴至月份",),
            "payment_status": ("缴费状态",),
            "monthly_contribution": ("月缴存额",),
            "personal_contribution_ratio": ("个人缴存比例",),
            "employer_contribution_ratio": ("单位缴存比例",),
        },
        "fields": {
            "contribution_location": ("contribution_location", "text"),
            "participation_date": ("participation_date", "date"),
            "first_contribution_month": ("first_contribution_month", "date"),
            "paid_through_month": ("paid_through_month", "date"),
            "payment_status": ("payment_status", "text"),
            "monthly_contribution": ("monthly_contribution", "number"),
            "personal_contribution_ratio": ("personal_contribution_ratio", "text"),
            "employer_contribution_ratio": ("employer_contribution_ratio", "text"),
        },
    },
    {
        "name": "housing_fund_provider",
        "record_type": "housing_fund",
        "signature": ("employer", "information_updated_month"),
        "sequence_optional": True,
        "record_boundary": "continuation",
        "aliases": {
            "employer": ("缴费单位",),
            "information_updated_month": ("信息更新日期", "信息更新月份"),
        },
        "fields": {
            "employer": ("employer", "employer_name"),
            "information_updated_month": ("information_updated_month", "date"),
        },
    },
    {
        "name": "professional_qualification",
        "record_type": "professional_qualification",
        "signature": ("qualification_name", "issuing_authority"),
        "aliases": {
            "sequence": ("编号",),
            "qualification_name": ("执业资格名称",),
            "level": ("等级",),
            "obtained_date": ("获得日期",),
            "expiry_date": ("到期日期",),
            "revocation_date": ("吊销日期",),
            "issuing_authority": ("颁发机构",),
            "authority_location": ("机构所在地",),
        },
        "fields": {
            "qualification_name": ("qualification_name", "text"),
            "level": ("level", "text"),
            "obtained_date": ("obtained_date", "date"),
            "expiry_date": ("expiry_date", "date"),
            "revocation_date": ("revocation_date", "date"),
            "issuing_authority": ("issuing_authority", "text"),
            "authority_location": ("authority_location", "text"),
        },
    },
    {
        "name": "award",
        "record_type": "award",
        "signature": ("authority", "award_content"),
        "aliases": {
            "sequence": ("编号",),
            "authority": ("奖励机构",),
            "award_content": ("奖励内容",),
            "effective_date": ("生效日期",),
            "end_date": ("截止日期",),
        },
        "fields": {
            "authority": ("authority", "text"),
            "award_content": ("award_content", "text"),
            "effective_date": ("effective_date", "date"),
            "end_date": ("end_date", "date"),
        },
    },
)


def _public_value(raw: str, kind: str) -> Any:
    if kind == "number":
        value = _number(raw)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if kind == "date":
        return _date(raw)
    if kind == "employer_name":
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            normalize_role_candidate,
            role_candidate_is_valid,
        )

        value = normalize_role_candidate(raw, "employer_name")
        return value if role_candidate_is_valid(value, "employer_name") else None
    return _clean(raw)


_PUBLIC_FINAL_DATASETS = {
    "tax_arrears": "tax_arrears_records",
    "civil_judgment": "civil_judgment_records",
    "enforcement": "enforcement_records",
    "administrative_penalty": "administrative_penalty_records",
    "housing_fund": "housing_fund_records",
    "professional_qualification": "professional_qualification_records",
    "award": "administrative_award_records",
}


def _consume_public_canonical_row(
    parse_result: Any,
    working: dict[tuple[str, int], dict[str, Any]],
    *,
    page: Any,
    table: Any,
    layout: Mapping[str, Any],
    slots: Mapping[str, int],
    data_row: tuple[str, ...],
    row_index: int,
    sequence: int,
) -> None:
    record_type = str(layout["record_type"])
    target_dataset = _PUBLIC_FINAL_DATASETS[record_type]
    target_id = stable_record_id("public_record", record_type, sequence)
    item = working.setdefault(
        (record_type, sequence),
        {
            "public_record_id": target_id,
            "sequence": sequence,
            "record_type": record_type,
            "source": "native_detail_public_table",
            "source_refs": [],
            "confidence": 1.0,
        },
    )
    row_ref = _source_ref(page, table, row=row_index)
    if row_ref not in item["source_refs"]:
        item["source_refs"].append(row_ref)
    for role, (field_name, kind) in layout["fields"].items():
        raw = _slot_value(data_row, slots, role)
        ref = _source_ref(page, table, row=row_index, column=slots[role])
        if not raw:
            _report_required_row_failure(
                parse_result,
                issue_code="candidate_b_public_record_cell_unresolved",
                dataset=target_dataset,
                sequence=sequence,
                field_name=field_name,
                row=data_row,
                page=page,
                table=table,
                row_index=row_index,
                target_record_id=target_id,
            )
            _append_internal_field(item, "_unresolved_fields", field_name)
            continue
        if _compact(raw) in {"-", "--", "---"}:
            _mark_source_absent(item, field_name, raw)
            continue
        value = _public_value(raw, kind)
        if value in (None, ""):
            _reject_exact_observation(
                parse_result,
                item,
                dataset=target_dataset,
                target_record_id=target_id,
                field_name=field_name,
                raw=raw,
                source_ref=ref,
                parser_stage="candidate_b_public_canonical_slots",
            )
            continue
        _merge_exact_observation(
            parse_result,
            item,
            dataset=target_dataset,
            target_record_id=target_id,
            field_name=field_name,
            value=value,
            raw=raw,
            source_ref=ref,
            parser_stage="candidate_b_public_canonical_slots",
        )


def _extract_public_records(parse_result: Any) -> list[dict[str, Any]]:
    # All accepted layouts are enumerated from the canonical PBOC report.  A
    # missed physical cell therefore becomes uncertainty, never a left-shifted
    # value guessed from the remaining non-empty cells.
    working: dict[tuple[str, int], dict[str, Any]] = {}
    optional_sequence_counters: defaultdict[str, int] = defaultdict(int)
    pending_optional_sequences: dict[str, dict[str, int]] = {}
    reading_order = dict(getattr(parse_result, "reading_order_by_logical", {}) or {})

    def report_pending(record_type: str) -> None:
        pending = pending_optional_sequences.pop(record_type, None)
        if pending is None:
            return
        sequence = int(pending["sequence"])
        _report_optional_public_continuation_missing(
            parse_result,
            dataset=_PUBLIC_FINAL_DATASETS[record_type],
            target_record_id=stable_record_id("public_record", record_type, sequence),
            sequence=sequence,
            source_refs=working.get((record_type, sequence), {}).get("source_refs", []),
        )

    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = [tuple(row) for row in _table_rows(table)]
            for header_index, header in enumerate(rows):
                matched: dict[str, Any] | None = None
                slots: dict[str, int] = {}
                for layout in _PUBLIC_CANONICAL_LAYOUTS:
                    candidate = _canonical_header_slots(header, layout["aliases"])
                    if all(role in candidate for role in layout["signature"]):
                        matched = layout
                        slots = candidate
                        break
                if matched is None:
                    continue
                matched_record_type = str(matched["record_type"])
                matched_boundary = str(matched.get("record_boundary") or "standalone")
                for pending_record_type in tuple(pending_optional_sequences):
                    if not (
                        matched_record_type == pending_record_type
                        and matched_boundary == "continuation"
                    ):
                        report_pending(pending_record_type)
                expected = set(matched["aliases"])
                if matched.get("sequence_optional"):
                    expected.discard("sequence")
                if not expected.issubset(slots):
                    _report_header_graph_failure(
                        parse_result,
                        dataset=_PUBLIC_FINAL_DATASETS[str(matched["record_type"])],
                        page=page,
                        table=table,
                        row_index=header_index,
                        row=header,
                    )
                    continue
                if header_index + 1 >= len(rows):
                    _report_header_graph_failure(
                        parse_result,
                        dataset=_PUBLIC_FINAL_DATASETS[str(matched["record_type"])],
                        page=page,
                        table=table,
                        row_index=header_index,
                        row=header,
                    )
                    continue
                for data_index in range(header_index + 1, len(rows)):
                    data_row = rows[data_index]
                    if any(
                        all(
                            role in _canonical_header_slots(data_row, candidate_layout["aliases"])
                            for role in candidate_layout["signature"]
                        )
                        for candidate_layout in _PUBLIC_CANONICAL_LAYOUTS
                    ):
                        break
                    if matched.get("sequence_optional") and not any(_clean(cell) for cell in data_row):
                        continue
                    if matched.get("sequence_optional"):
                        record_type = matched_record_type
                        boundary = matched_boundary
                        if boundary == "start":
                            optional_sequence_counters[record_type] += 1
                            sequence = optional_sequence_counters[record_type]
                            logical_page = int(getattr(page, "page_number", 0) or 0)
                            pending_optional_sequences[record_type] = {
                                "sequence": sequence,
                                "logical_page": logical_page,
                                "reading_order": int(
                                    reading_order.get(logical_page, logical_page) or logical_page
                                ),
                            }
                        elif boundary == "continuation":
                            pending = pending_optional_sequences.get(record_type)
                            if pending is None:
                                _report_unowned_optional_public_fragment(
                                    parse_result,
                                    dataset=_PUBLIC_FINAL_DATASETS[record_type],
                                    row=data_row,
                                    page=page,
                                    table=table,
                                    row_index=data_index,
                                )
                                break
                            logical_page = int(getattr(page, "page_number", 0) or 0)
                            current_order = int(
                                reading_order.get(logical_page, logical_page) or logical_page
                            )
                            start_page = int(pending["logical_page"])
                            adjacent = logical_page == start_page or (
                                current_order == int(pending["reading_order"]) + 1
                            )
                            if not adjacent:
                                report_pending(record_type)
                                _report_unowned_optional_public_fragment(
                                    parse_result,
                                    dataset=_PUBLIC_FINAL_DATASETS[record_type],
                                    row=data_row,
                                    page=page,
                                    table=table,
                                    row_index=data_index,
                                )
                                break
                            sequence = int(pending["sequence"])
                        else:
                            optional_sequence_counters[record_type] += 1
                            sequence = optional_sequence_counters[record_type]
                    else:
                        sequence = _sequence_value(data_row, slots)
                    if sequence is None:
                        raw_sequence = _slot_value(data_row, slots, "sequence")
                        if raw_sequence or (
                            data_index == header_index + 1
                            and sum(bool(_clean(cell)) for cell in data_row) >= 2
                        ):
                            _report_unkeyed_business_row(
                                parse_result,
                                dataset=_PUBLIC_FINAL_DATASETS[str(matched["record_type"])],
                                row=data_row,
                                page=page,
                                table=table,
                                row_index=data_index,
                            )
                        break
                    _consume_public_canonical_row(
                        parse_result,
                        working,
                        page=page,
                        table=table,
                        layout=matched,
                        slots=slots,
                        data_row=data_row,
                        row_index=data_index,
                        sequence=sequence,
                    )
                    if (
                        matched.get("sequence_optional")
                        and matched.get("record_boundary") == "continuation"
                    ):
                        pending_optional_sequences.pop(str(matched["record_type"]), None)
                    if matched.get("sequence_optional"):
                        break

    for record_type in tuple(pending_optional_sequences):
        report_pending(record_type)

    records: list[dict[str, Any]] = []
    authority_fields = {
        "tax_arrears": "tax_authority",
        "civil_judgment": "filing_court",
        "enforcement": "court",
        "administrative_penalty": "authority",
        "housing_fund": "employer",
        "professional_qualification": "issuing_authority",
        "award": "authority",
    }
    internal = {
        "public_record_id",
        "sequence",
        "record_type",
        "source",
        "source_refs",
        "source_refs_by_field",
        "confidence",
    }
    for (record_type, _sequence), item in sorted(working.items(), key=lambda pair: pair[0]):
        content = {
            key: value
            for key, value in item.items()
            if key not in internal and not key.startswith("_") and key != "canonical_raw"
        }
        for field_name in item.get("_source_absent_fields", []):
            content.setdefault(field_name, None)
        if record_type in {"tax_arrears", "civil_judgment", "enforcement", "administrative_penalty", "housing_fund"}:
            content["reporting_amount_currency"] = "CNY"
            content["reporting_amount_unit"] = "yuan"
        if "cause" in item:
            content["cause_status"] = "reported"
        elif "cause" in item.get("_source_absent_fields", []):
            content["cause_status"] = "not_reported"
        if "administrative_review_result" in item:
            content["administrative_review_result_status"] = "reported"
        elif "administrative_review_result" in item.get("_source_absent_fields", []):
            content["administrative_review_result_status"] = "not_reported"
        record = {
            **item,
            "authority": item.get(authority_fields[record_type]),
            "start_date": item.get("filing_date") or item.get("effective_date"),
            "end_date": item.get("closure_date") or item.get("end_date"),
            "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
        }
        records.append(record)
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


def _continuation_rows_are_provider_shaped(
    rows: list[list[str]],
    *,
    known_sequences: set[int],
) -> tuple[bool, int | None]:
    """Recognize a headerless provider tail without consuming business rows.

    A provider tail has exactly two populated columns, institution-shaped
    values, and either repeats already observed sequence numbers or contains
    at least two consistently unreadable sequence glyphs.  A new numbered
    residence/employment row therefore stays in its current canonical mode.
    """

    if not rows or not known_sequences:
        return False, None
    nonempty_columns = [
        [index for index, value in enumerate(row) if _clean(value)] for row in rows
    ]
    if not all(len(columns) == 2 and columns[0] == 0 for columns in nonempty_columns):
        return False, None
    provider_column = nonempty_columns[0][1]
    if any(columns[1] != provider_column for columns in nonempty_columns):
        return False, None
    provider_pattern = re.compile(
        r"(?:银行(?:股份)?有限公司|银行|信用社|消费金融(?:股份)?有限公司|财务有限公司|管理中心|征信中心)$"
    )
    if not all(provider_pattern.search(_clean(row[provider_column])) for row in rows):
        return False, None
    parsed_sequences: list[int] = []
    unreadable_sequences = 0
    for row in rows:
        first = _clean(row[0] if row else "")
        if first.isdigit():
            parsed_sequences.append(int(first))
        else:
            unreadable_sequences += 1
    if parsed_sequences:
        return (
            unreadable_sequences == 0 and set(parsed_sequences) <= known_sequences,
            provider_column,
        )
    return len(rows) >= 2 and unreadable_sequences == len(rows), provider_column


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
    target_record_id: str | None = None,
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
            target_record_id=target_record_id or f"{dataset}:{sequence}",
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


def _report_optional_public_continuation_missing(
    parse_result: Any,
    *,
    dataset: str,
    target_record_id: str,
    sequence: int,
    source_refs: Iterable[Mapping[str, Any]],
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="page_continuation",
            issue_code="candidate_b_public_record_continuation_missing",
            message="A canonical public-record start block was preserved, but its required continuation block was not observed.",
            parser_stage="candidate_b_public_record_boundaries",
            target_dataset=dataset,
            target_record_id=target_record_id,
            observed_value={"sequence": sequence, "start_block_observed": True},
            candidate_value={
                "missing_fields": ["employer", "information_updated_month"]
            },
            source_refs=tuple(source_refs),
            reason_codes=(
                "canonical_start_block_observed",
                "required_continuation_not_observed",
                "partial_record_preserved",
            ),
        ),
    )


def _report_unowned_optional_public_fragment(
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
            issue_code="candidate_b_public_record_continuation_unowned",
            message="A canonical public-record continuation block had no preceding start block and was not attached by proximity.",
            parser_stage="candidate_b_public_record_boundaries",
            target_dataset=dataset,
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=(
                "canonical_continuation_block_observed",
                "preceding_start_block_not_observed",
                "proximity_attachment_forbidden",
                "record_not_invented",
            ),
        ),
    )


def _report_unkeyed_business_row(
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
            category="ocr_cell_level_error",
            issue_code="candidate_b_sequence_cell_unresolved",
            message="A canonical business row was present but its printed sequence cell was unreadable; no row identity was invented.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            field_name="sequence",
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("canonical_business_row", "printed_sequence_unreadable", "record_not_silently_dropped"),
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
    providers: defaultdict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    provider_source_absent: set[int] = set()
    unkeyed_providers: list[
        tuple[str, dict[str, Any], tuple[str, ...], Any, Any, int]
    ] = []
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
                provider_shaped, provider_column = _continuation_rows_are_provider_shaped(
                    rows,
                    known_sequences=set(records),
                )
                if provider_shaped and provider_column is not None:
                    active_slots = {"sequence": 0, "data_provider": provider_column}
                    mode = "provider"
            for row_index, row in enumerate(rows):
                residence_slots = _canonical_header_slots(row, aliases)
                compact_header = _compact("".join(row))
                header_anchor = len(residence_slots) >= 2
                if (
                    all(marker in compact_header for marker in ("编号", "居住地址", "信息更新日期"))
                    and not set(aliases) <= set(residence_slots)
                ) or (header_anchor and not set(aliases) <= set(residence_slots)):
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
                if set(aliases) <= set(residence_slots):
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
                        institution = next(
                            (
                                value
                                for value in reversed(nonempty)
                                if re.search(
                                    r"(?:银行|信用社|有限公司|管理中心|征信中心)",
                                    value,
                                )
                            ),
                            "",
                        )
                        if institution:
                            unkeyed_providers.append(
                                (
                                    institution,
                                    _source_ref(page, table, row=row_index),
                                    tuple(row),
                                    page,
                                    table,
                                    row_index,
                                )
                            )
                        else:
                            _report_unkeyed_fragment(
                                parse_result,
                                dataset="residence_records",
                                row=row,
                                page=page,
                                table=table,
                                row_index=row_index,
                            )
                    elif mode == "provider" and nonempty:
                        _report_unkeyed_fragment(
                            parse_result,
                            dataset="residence_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    elif mode == "residence" and nonempty:
                        _report_unkeyed_business_row(
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
                    provider_column = active_slots.get("data_provider")
                    if provider == "--":
                        provider_source_absent.add(sequence)
                    elif provider and provider_column is not None:
                        providers[sequence].append(
                            (provider, _source_ref(page, table, row=row_index, column=provider_column))
                        )
                    continue
                residence_id = stable_record_id("credit_residence", sequence)
                address = _slot_value(row, active_slots, "address")
                updated_raw = _slot_value(row, active_slots, "information_updated_date")
                updated = _date(updated_raw)
                collapsed_updated: str | None = None
                if address and not updated and not updated_raw:
                    date_matches = list(_DATE_RE.finditer(address))
                    if len(date_matches) == 1:
                        candidate_date = _date(date_matches[0].group(0))
                        residual = (
                            address[: date_matches[0].start()]
                            + " "
                            + address[date_matches[0].end() :]
                        )
                        residual = re.sub(
                            r"^[\s\"“”'*#=—–-]+|[\s\"“”'*#=—–-]+$",
                            "",
                            residual,
                        )
                        if candidate_date and len(_compact(residual)) >= 4:
                            address = residual
                            collapsed_updated = candidate_date
                            updated = candidate_date
                record = records.setdefault(
                    sequence,
                    {
                        "sequence": sequence,
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_residence_table",
                        "source_refs": [],
                        "record_id": residence_id,
                        "residence_record_id": residence_id,
                        "confidence": 1.0,
                    },
                )
                record["source_refs"].append(ref)
                if address == "--":
                    _mark_source_absent(record, "address", address)
                elif address:
                    address_ref = _source_ref(
                        page, table, row=row_index, column=active_slots["address"]
                    )
                    suspicious_address = bool(
                        _DATE_RE.search(address)
                        or re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", address)
                        or any(
                            label in _compact(address)
                            for label_names in aliases.values()
                            for label in label_names
                        )
                    )
                    if suspicious_address:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="address",
                            raw=address,
                            source_ref=address_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="address",
                            value=address,
                            raw=address,
                            source_ref=address_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
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
                        target_record_id=residence_id,
                    )
                phone = _slot_value(row, active_slots, "residential_phone")
                status = _slot_value(row, active_slots, "residence_status")
                if phone == "--":
                    _mark_source_absent(record, "residential_phone", phone)
                elif phone:
                    phone_digits = re.sub(r"\D", "", phone)
                    phone_ref = _source_ref(
                        page, table, row=row_index, column=active_slots["residential_phone"]
                    )
                    if any(character.isalpha() for character in phone) or not 5 <= len(phone_digits) <= 16:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="residential_phone",
                            raw=phone,
                            source_ref=phone_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="residential_phone",
                            value=phone,
                            raw=phone,
                            source_ref=phone_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                if status == "--":
                    _mark_source_absent(record, "residence_status", status)
                elif status:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="residence_records",
                        target_record_id=residence_id,
                        field_name="residence_status",
                        value=status,
                        raw=status,
                        source_ref=_source_ref(
                            page, table, row=row_index, column=active_slots["residence_status"]
                        ),
                        parser_stage="candidate_b_residence_canonical_slots",
                    )
                if updated:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="residence_records",
                        target_record_id=residence_id,
                        field_name="information_updated_date",
                        value=updated,
                        raw=updated_raw or collapsed_updated or updated,
                        source_ref=_source_ref(
                            page,
                            table,
                            row=row_index,
                            column=(
                                active_slots["address"]
                                if collapsed_updated
                                else active_slots["information_updated_date"]
                            ),
                        ),
                        parser_stage="candidate_b_residence_canonical_slots",
                    )
                elif updated_raw == "--":
                    _mark_source_absent(record, "information_updated_date", updated_raw)
                elif updated_raw:
                    _reject_exact_observation(
                        parse_result,
                        record,
                        dataset="residence_records",
                        target_record_id=residence_id,
                        field_name="information_updated_date",
                        raw=updated_raw,
                        source_ref=_source_ref(
                            page, table, row=row_index, column=active_slots["information_updated_date"]
                        ),
                        parser_stage="candidate_b_residence_canonical_slots",
                    )
                elif not updated_raw:
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_residence_required_cell_unresolved",
                        dataset="residence_records",
                        sequence=sequence,
                        field_name="information_updated_date",
                        row=row,
                        page=page,
                        table=table,
                        row_index=row_index,
                        target_record_id=residence_id,
                    )
            previous_table_id = table_id or previous_table_id

    keyed_sequences = {sequence for sequence, values in providers.items() if values}
    missing_provider_sequences = [
        sequence for sequence in sorted(records) if sequence not in keyed_sequences
    ]
    if (
        unkeyed_providers
        and len(unkeyed_providers) == len(missing_provider_sequences)
        and sorted(records) == list(range(1, max(records, default=0) + 1))
        and missing_provider_sequences
        == list(range(min(missing_provider_sequences), max(records) + 1))
    ):
        for sequence, (provider, provider_ref, _row, _page, _table, _row_index) in zip(
            missing_provider_sequences,
            unkeyed_providers,
            strict=True,
        ):
            providers[sequence].append((provider, provider_ref))
        unkeyed_providers = []
    for _provider, _ref, row, page, table, row_index in unkeyed_providers:
        _report_unkeyed_fragment(
            parse_result,
            dataset="residence_records",
            row=row,
            page=page,
            table=table,
            row_index=row_index,
        )
    for sequence, record in records.items():
        residence_id = str(record["residence_record_id"])
        if sequence in provider_source_absent:
            _mark_source_absent(record, "data_provider", "--")
        for provider, provider_ref in providers.get(sequence, ()):
            record["source_refs"].append(provider_ref)
            _merge_exact_observation(
                parse_result,
                record,
                dataset="residence_records",
                target_record_id=residence_id,
                field_name="data_provider",
                value=provider,
                raw=provider,
                source_ref=provider_ref,
                parser_stage="candidate_b_residence_canonical_slots",
            )
        if (
            not record.get("data_provider")
            and "data_provider" not in set(record.get("_source_absent_fields") or ())
        ):
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record["extraction_status"] = "review"
            record_issue(
                parse_result,
                make_issue(
                    category="page_continuation",
                    issue_code="candidate_b_residence_provider_missing",
                    message="A canonical residence row has no safely bound provider row.",
                    parser_stage="candidate_b_residence_canonical_slots",
                    target_dataset="residence_records",
                    target_record_id=residence_id,
                    field_name="data_provider",
                    observed_value={"sequence": sequence},
                    source_refs=tuple(record.get("source_refs") or ()),
                    reason_codes=(
                        "canonical_residence_row_observed",
                        "provider_component_not_bound",
                        "row_marked_for_review",
                    ),
                ),
            )
    return [records[key] for key in sorted(records)]


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
    employment_section_active = False
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            table_id = str(getattr(table, "table_id", "") or "")
            table_text = _compact(" ".join(cell for row in rows for cell in row))
            table_has_employment_schema = any(
                marker in table_text
                for marker in ("工作单位", "单位性质", "单位电话", "职业", "职务", "职称")
            )
            continues = bool(
                previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            other_section_markers = (
                "居住地址",
                "住宅电话",
                "居住状况",
                "手机号码",
                "配偶",
                "学历",
                "学位",
                "电子邮箱",
                "通讯地址",
                "户籍地址",
            )
            if (
                employment_section_active
                and not table_has_employment_schema
                and any(marker in table_text for marker in other_section_markers)
            ):
                continues = False
                employment_section_active = False
            if not continues:
                active_slots = {}
                mode = ""
                employment_section_active = table_has_employment_schema
            elif table_has_employment_schema:
                employment_section_active = True
            if not employment_section_active and not table_has_employment_schema:
                previous_table_id = table_id or previous_table_id
                continue
            if continues and mode in {"basic", "detail"} and rows:
                provider_shaped, provider_column = _continuation_rows_are_provider_shaped(
                    rows,
                    known_sequences=set(records),
                )
                if provider_shaped and provider_column is not None:
                    active_slots = {"sequence": 0, "data_provider": provider_column}
                    mode = "provider"
            for row_index, row in enumerate(rows):
                basic_slots = _canonical_header_slots(row, basic_aliases)
                detail_slots = _canonical_header_slots(row, detail_aliases)
                compact_header = _compact("".join(row))
                basic_anchor = len(basic_slots) >= 2
                detail_anchor = len(detail_slots) >= 2
                broken_basic = (
                    all(marker in compact_header for marker in ("编号", "工作单位", "单位性质", "单位电话"))
                    or basic_anchor
                ) and not set(basic_aliases) <= set(basic_slots)
                broken_detail = (
                    all(marker in compact_header for marker in ("编号", "职业", "行业", "职务", "职称"))
                    or detail_anchor
                ) and not set(detail_aliases) <= set(detail_slots)
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
                    employment_section_active = True
                    continue
                if set(basic_aliases) <= set(basic_slots):
                    active_slots = basic_slots
                    mode = "basic"
                    continue
                if set(detail_aliases) <= set(detail_slots):
                    active_slots = detail_slots
                    mode = "detail"
                    continue
                if (
                    employment_section_active
                    and "数据发生机构名称" in _compact("".join(row))
                ):
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
                    elif mode in {"basic", "detail"} and _nonempty(row):
                        _report_unkeyed_business_row(
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
                        "record_id": stable_record_id("credit_employment", sequence),
                        "employment_record_id": stable_record_id("credit_employment", sequence),
                        "sequence": sequence,
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_employment_table",
                        "source_refs": [],
                        "confidence": 1.0,
                    },
                )
                record["source_refs"].append(_source_ref(page, table, row=row_index))
                observed_components = record.setdefault("_observed_components", [])
                if mode not in observed_components:
                    observed_components.append(mode)
                if mode == "provider":
                    provider = _slot_value(row, active_slots, "data_provider")
                    provider_column = active_slots.get("data_provider")
                    if provider == "--":
                        _mark_source_absent(record, "data_provider", provider)
                    elif provider and provider_column is not None:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=str(record["employment_record_id"]),
                            field_name="data_provider",
                            value=provider,
                            raw=provider,
                            source_ref=_source_ref(
                                page, table, row=row_index, column=provider_column
                            ),
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                    continue
                roles = basic_aliases if mode == "basic" else detail_aliases
                for role in roles:
                    if role == "sequence":
                        continue
                    raw = _slot_value(row, active_slots, role)
                    if raw == "--":
                        _mark_source_absent(record, role, raw)
                        continue
                    if not raw:
                        continue
                    ref = _source_ref(page, table, row=row_index, column=active_slots[role])
                    target_record_id = str(record["employment_record_id"])
                    if role == "entry_year":
                        match = re.fullmatch(r"\D*((?:19|20)\d{2})\D*", raw)
                        if match:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=int(match.group(1)),
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role == "information_updated_date":
                        converted = _date(raw)
                        if converted:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=converted,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role == "employer_phone":
                        digits = re.sub(r"\D", "", raw)
                        if any(character.isalpha() for character in raw) or not 5 <= len(digits) <= 16:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=raw,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role == "employer" and (
                        _DATE_RE.search(raw) or re.search(r"(?<!\d)\d{7,16}(?!\d)", raw)
                    ):
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=role,
                            raw=raw,
                            source_ref=ref,
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=role,
                            value=raw,
                            raw=raw,
                            source_ref=ref,
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                required_roles = ("employer",) if mode == "basic" else (
                    "occupation",
                    "information_updated_date",
                )
                for required_role in required_roles:
                    if (
                        record.get(required_role)
                        or required_role in record.get("_unresolved_fields", [])
                        or required_role in record.get("_source_absent_fields", [])
                    ):
                        continue
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
                        target_record_id=str(record["employment_record_id"]),
                    )
            previous_table_id = table_id or previous_table_id
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    for sequence, record in records.items():
        observed_components = set(record.get("_observed_components") or ())
        missing_components = [
            component
            for component in ("basic", "detail", "provider")
            if component not in observed_components
        ]
        if not missing_components:
            continue
        record["extraction_status"] = "review"
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_employment_component_missing",
                message="A canonical employment row is missing one or more schema components.",
                parser_stage="candidate_b_employment_canonical_slots",
                target_dataset="employment_records",
                target_record_id=str(record["employment_record_id"]),
                field_name="employment_components",
                observed_value={
                    "sequence": sequence,
                    "observed_components": sorted(observed_components),
                },
                candidate_value={"missing_components": missing_components},
                source_refs=tuple(record.get("source_refs") or ()),
                reason_codes=(
                    "canonical_employment_row_observed",
                    "component_population_incomplete",
                    "row_marked_for_review",
                ),
            ),
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


def _inquiry_sequence_endpoint(
    values: set[int],
    dates_by_sequence: Mapping[int, set[str]] | None = None,
) -> tuple[int | None, list[int]]:
    """Trust exact inquiry ordinal cells while suppressing prefix bleed.

    Unlike account-family discovery, a canonical inquiry sequence is already
    bound to the exact leftmost header column, so a sparse high ordinal is
    valid evidence of preceding rows.  The only bounded correction here is the
    known watermark/neighbour prefix form (for example 89 read as 789).
    """

    retained = {value for value in values if 1 <= value <= 9999}
    rejected: list[int] = []
    dates = dates_by_sequence or {}
    for value in sorted(retained):
        if value < 100:
            continue
        suffix = value % 100
        if suffix <= 0 or suffix >= value:
            continue
        actual_neighbours = value - 1 in retained or value + 1 in retained
        same_date_duplicate = bool(
            value >= 300
            and suffix in retained
            and dates.get(value)
            and dates.get(suffix)
            and dates[value] & dates[suffix]
            and not actual_neighbours
        )
        bounded_prefix_between_neighbours = bool(
            value >= 300
            and suffix - 1 in retained
            and suffix + 1 in retained
            and not actual_neighbours
        )
        if same_date_duplicate or bounded_prefix_between_neighbours:
            retained.discard(value)
            rejected.append(value)
    return (max(retained) if retained else None), rejected


def _inquiry_source_coverage(parse_result: Any) -> dict[str, Any]:
    """Inventory printed inquiry ordinals before any business row is emitted.

    Inquiry rows are closed-world canonical records: institutional and personal
    inquiry ordinals each restart at one.  Completeness therefore comes from
    the printed ordinal populations, not from the subset of rows whose date,
    institution, and reason all happened to decode.  Headerless physical-page
    continuations remain attached to the last canonical four-column table.
    """

    aliases = {
        "sequence": ("编号", "序号"),
        "inquiry_date": ("查询日期",),
        "institution": ("查询机构",),
        "reason": ("查询原因",),
    }
    groups: list[dict[str, Any]] = []
    active_group: dict[str, Any] | None = None
    active_slots: dict[str, int] = {}

    for page in getattr(parse_result, "pages", None) or []:
        canonical_page = (
            str(getattr(page, "canonical_template_id", "") or "")
            == "annotations_and_inquiries"
        )
        page_had_inquiry_table = False
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            header_text = _compact("".join(_nonempty(rows[0])))
            has_header = all(
                marker in header_text
                for marker in ("查询日期", "查询机构", "查询原因")
            )
            header_slots = _canonical_header_slots(
                tuple(str(value or "") for value in rows[0]), aliases
            )
            first_value = _clean(rows[0][0] if rows[0] else "")
            is_continuation = bool(
                not has_header
                and canonical_page
                and active_group is not None
                and "sequence" in active_slots
                and re.fullmatch(r"\D*\d{1,4}\D*", first_value)
            )
            if has_header:
                # Even a damaged/merged header is useful as a section boundary.
                # The canonical printed sequence is the leftmost column, but no
                # other field is guessed when its header binding is unresolved.
                active_slots = dict(header_slots)
                active_slots.setdefault("sequence", 0)
                active_group = {"observations": [], "source_refs": []}
                groups.append(active_group)
                start = 1
            elif is_continuation:
                start = 0
            else:
                continue

            page_had_inquiry_table = True
            table_ref = _source_ref(page, table)
            active_group["source_refs"].append(table_ref)
            for row_index, row in enumerate(rows[start:], start=start):
                cells = tuple(str(value or "").strip() for value in row)
                raw_sequence = _slot_value(cells, active_slots, "sequence")
                match = re.fullmatch(r"\D*(\d{1,4})\D*", raw_sequence)
                if match is None:
                    continue
                sequence = int(match.group(1))
                if sequence <= 0:
                    continue
                institution = _slot_value(cells, active_slots, "institution")
                reason = _slot_value(cells, active_slots, "reason")
                compact_institution = _compact(
                    _normalized_inquiry_field("institution", institution)
                )
                compact_reason = _compact(_normalized_inquiry_field("reason", reason))
                inquiry_type: str | None
                if compact_institution == "本人" or compact_reason.startswith("本人查询"):
                    inquiry_type = "personal"
                elif compact_institution or compact_reason:
                    inquiry_type = "institution"
                else:
                    inquiry_type = None
                active_group["observations"].append(
                    {
                        "sequence": sequence,
                        "inquiry_type": inquiry_type,
                        "inquiry_date": _slot_value(cells, active_slots, "inquiry_date"),
                        "source_ref": _source_ref(page, table, row=row_index),
                    }
                )

        if not canonical_page and not page_had_inquiry_table:
            active_group = None
            active_slots = {}

    # A header can repeat at a physical-page boundary.  If all readable rows in
    # that group have one type, their unreadable neighbours share that canonical
    # population.  A headerless group beginning above one likewise continues
    # the nearest preceding typed population.  Mixed groups stay unclassified.
    group_types: list[str | None] = []
    for group in groups:
        types = {
            str(observation.get("inquiry_type"))
            for observation in group["observations"]
            if observation.get("inquiry_type")
        }
        group_types.append(next(iter(types)) if len(types) == 1 else None)
    for index, group in enumerate(groups):
        if group_types[index] is not None:
            continue
        sequences = {
            int(observation["sequence"])
            for observation in group["observations"]
            if int(observation.get("sequence") or 0) > 0
        }
        if sequences and min(sequences) > 1:
            preceding = next(
                (group_types[cursor] for cursor in range(index - 1, -1, -1) if group_types[cursor]),
                None,
            )
            if preceding:
                group_types[index] = preceding

    sequences_by_type: defaultdict[str, set[int]] = defaultdict(set)
    dates_by_type: defaultdict[str, defaultdict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    unclassified_endpoints: list[int] = []
    all_refs: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group_type = group_types[index]
        known_types = {
            str(observation.get("inquiry_type"))
            for observation in group["observations"]
            if observation.get("inquiry_type")
        }
        group_sequences: set[int] = set()
        for observation in group["observations"]:
            explicit_type = observation.get("inquiry_type")
            inquiry_type = group_type if len(known_types) <= 1 else explicit_type
            sequence = int(observation.get("sequence") or 0)
            if inquiry_type and sequence > 0:
                sequences_by_type[str(inquiry_type)].add(sequence)
                raw_date = _compact(observation.get("inquiry_date"))
                if raw_date:
                    dates_by_type[str(inquiry_type)][sequence].add(raw_date)
            elif sequence > 0:
                group_sequences.add(sequence)
        if group_sequences:
            endpoint, _outliers = _inquiry_sequence_endpoint(group_sequences)
            if endpoint:
                unclassified_endpoints.append(endpoint)
        all_refs.extend(
            dict(ref) for ref in group.get("source_refs") or () if isinstance(ref, Mapping)
        )

    endpoints: dict[str, int] = {}
    outliers: dict[str, list[int]] = {}
    observed: dict[str, list[int]] = {}
    for inquiry_type, values in sequences_by_type.items():
        endpoint, rejected = _inquiry_sequence_endpoint(
            values, dates_by_type.get(inquiry_type)
        )
        if endpoint:
            endpoints[inquiry_type] = endpoint
        observed[inquiry_type] = sorted(values)
        if rejected:
            outliers[inquiry_type] = rejected

    # Keep issue evidence compact and deterministic: one source table reference
    # is sufficient to select each affected page for the one-shot repair pass.
    unique_refs: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for ref in all_refs:
        marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen_refs:
            seen_refs.add(marker)
            unique_refs.append(ref)
    expected = sum(endpoints.values()) + sum(unclassified_endpoints)
    return {
        **({"sequence_endpoints": endpoints} if endpoints else {}),
        **({"observed_sequences": observed} if observed else {}),
        **({"sequence_outliers": outliers} if outliers else {}),
        **({"unclassified_sequence_endpoints": unclassified_endpoints} if unclassified_endpoints else {}),
        **({"expected_row_count": expected} if expected else {}),
        **({"source_refs": unique_refs} if unique_refs else {}),
    }


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

    # Agreement cards are repeated labelled records, and a parser that emits
    # one valid card cannot use that partial success as evidence that the
    # section is complete.  Count both printed ordinals and repeated primary
    # labels directly from canonical page evidence before business repair.
    agreement_sequences: set[int] = set()
    agreement_label_count = 0
    agreement_refs: list[dict[str, Any]] = []
    active_agreements = False
    agreement_heading = "授信协议信息"
    agreement_label = "授信协议标识"
    agreement_anchor = re.compile(r"授信协议\s*(\d{1,3})")
    agreement_end_markers = (
        "非信贷交易信息明细",
        "公共信息明细",
        "异议标注",
        "查询记录",
        "报告说明",
    )
    for page in pages or []:
        logical_page = int(page.get("page") or 0)
        source_page = int(page.get("source_page") or logical_page)
        page_observed = False
        for line in page.get("lines") or []:
            if not isinstance(line, Mapping):
                continue
            text = _compact(line.get("text") or line.get("content") or "")
            if agreement_heading in text:
                active_agreements = True
                page_observed = True
            if not active_agreements and agreement_anchor.search(text):
                # A damaged/missing section heading must not hide an explicit
                # agreement ordinal.  The primary label alone is insufficient:
                # it also appears in ordinary account-detail headings.
                active_agreements = True
            if not active_agreements:
                continue
            matches = [int(match.group(1)) for match in agreement_anchor.finditer(text)]
            if matches:
                agreement_sequences.update(value for value in matches if 1 <= value <= 100)
                page_observed = True
            label_occurrences = text.count(agreement_label)
            if label_occurrences:
                agreement_label_count += label_occurrences
                page_observed = True
            if any(marker in text for marker in agreement_end_markers):
                active_agreements = False
        if page_observed and logical_page > 0:
            agreement_refs.append(
                {
                    "source": "candidate_b_source_coverage_ledger",
                    "logical_page": logical_page,
                    "source_page": source_page,
                    "geometry_scope": "logical_page",
                }
            )

    endpoints: dict[str, int] = {}
    sequence_outliers: dict[str, list[int]] = {}
    for account_family, values in family_sequences.items():
        endpoint, outliers = _credible_sequence_endpoint(values)
        if endpoint is not None:
            endpoints[account_family] = endpoint
        if outliers:
            sequence_outliers[account_family] = outliers

    agreement_endpoint, agreement_outliers = _credible_sequence_endpoint(agreement_sequences)
    inquiry_coverage = _inquiry_source_coverage(parse_result)
    # Every canonical agreement card has a printed ordinal.  Once at least one
    # credible ordinal is available, its dense endpoint is a stronger
    # population witness than raw primary-label occurrences: corrected logical
    # pages can contain overlapping evidence fragments and therefore repeat the
    # same ``授信协议标识`` label.  Treating those duplicate labels as extra cards
    # creates a false partial-dataset report even when all printed ordinals were
    # recovered.  Label counting remains the fallback for a section whose
    # ordinal headings are completely unreadable.
    agreement_expected = agreement_endpoint or agreement_label_count
    source_refs: dict[str, list[dict[str, Any]]] = {}
    if agreement_refs:
        source_refs["credit_lines"] = agreement_refs
    if inquiry_coverage.get("source_refs"):
        source_refs["inquiry_records"] = list(inquiry_coverage["source_refs"])
    return {
        "sequence_endpoints": {
            name: max(values)
            for name, values in sequences.items()
            if values
        },
        **({"credit_accounts": sum(endpoints.values())} if endpoints else {}),
        **({"credit_agreements": agreement_expected} if agreement_expected else {}),
        **({"credit_agreement_sequence_endpoint": agreement_endpoint} if agreement_endpoint else {}),
        **({"credit_agreement_sequence_outliers": agreement_outliers} if agreement_outliers else {}),
        **(
            {"inquiry_records": int(inquiry_coverage["expected_row_count"])}
            if inquiry_coverage.get("expected_row_count")
            else {}
        ),
        **(
            {"inquiry_sequence_endpoints": dict(inquiry_coverage["sequence_endpoints"])}
            if inquiry_coverage.get("sequence_endpoints")
            else {}
        ),
        **(
            {"inquiry_observed_sequences": dict(inquiry_coverage["observed_sequences"])}
            if inquiry_coverage.get("observed_sequences")
            else {}
        ),
        **(
            {"inquiry_sequence_outliers": dict(inquiry_coverage["sequence_outliers"])}
            if inquiry_coverage.get("sequence_outliers")
            else {}
        ),
        **(
            {
                "inquiry_unclassified_sequence_endpoints": list(
                    inquiry_coverage["unclassified_sequence_endpoints"]
                )
            }
            if inquiry_coverage.get("unclassified_sequence_endpoints")
            else {}
        ),
        **({"account_family_endpoints": dict(endpoints)} if endpoints else {}),
        **({"account_family_sequence_outliers": sequence_outliers} if sequence_outliers else {}),
        **({"source_refs": source_refs} if source_refs else {}),
    }


def _record_pre_repair_source_gaps(
    parse_result: Any,
    datasets: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish source-population gaps early enough to select repair pages.

    The final completeness pass remains authoritative.  This early pass exists
    solely so a record that was never instantiated can still trigger the one
    allowed page repair instead of disappearing silently.
    """
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    ledger = _source_completeness_ledger(parse_result)
    source_refs = ledger.get("source_refs") if isinstance(ledger.get("source_refs"), Mapping) else {}
    checks = (
        ("credit_lines", "credit_agreements"),
        ("inquiry_records", "inquiry_records"),
    )
    for dataset_name, ledger_name in checks:
        expected = ledger.get(ledger_name)
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            continue
        observed = sum(isinstance(row, Mapping) for row in datasets.get(dataset_name) or ())
        if observed >= expected:
            continue
        record_issue(
            parse_result,
            make_issue(
                category="page_continuation",
                issue_code="source_sequence_or_count_gap",
                message=(
                    "Independent canonical section evidence contains more records than the first schema pass; "
                    "the affected pages must be repaired before completeness is decided."
                ),
                parser_stage="candidate_b_pre_repair_source_coverage",
                target_dataset=dataset_name,
                observed_value={"observed_row_count": observed},
                candidate_value={
                    "source_expected_row_count": expected,
                    **(
                        {
                            "source_sequence_endpoints": ledger.get(
                                "inquiry_sequence_endpoints", {}
                            ),
                            "unclassified_sequence_endpoints": ledger.get(
                                "inquiry_unclassified_sequence_endpoints", []
                            ),
                        }
                        if dataset_name == "inquiry_records"
                        else {}
                    ),
                },
                source_refs=(source_refs.get(dataset_name) or ()),
                reason_codes=(
                    "independent_source_ledger",
                    "missing_business_records",
                    "schema_triggered_page_repair_eligible",
                    "no_missing_row_invented",
                ),
            ),
        )
    return ledger


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
    mobile_records: dict[int, dict[str, Any]] = {}
    spouse_record: dict[str, Any] | None = None
    active_slots: dict[str, int] = {}
    mode = ""
    spouse_observed = False
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
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
                spouse_observed = False
            for row_index, row in enumerate(rows):
                mobile_slots = _canonical_header_slots(row, mobile_aliases)
                spouse_slots = _canonical_header_slots(row, spouse_aliases)
                compact_header = _compact("".join(row))
                broken_mobile = all(
                    marker in compact_header
                    for marker in ("编号", "手机号码", "信息更新日期", "数据发生机构名称")
                ) and not set(mobile_aliases) <= set(mobile_slots)
                broken_mobile = broken_mobile or (
                    len(mobile_slots) >= 2
                    and not set(mobile_aliases) <= set(mobile_slots)
                )
                broken_spouse = all(
                    marker in compact_header for marker in ("姓名", "证件类型", "证件号码", "工作单位", "联系电话")
                ) and not set(spouse_aliases) <= set(spouse_slots)
                broken_spouse = broken_spouse or (
                    len(spouse_slots) >= 2
                    and not set(spouse_aliases) <= set(spouse_slots)
                )
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
                if "数据发生机构名称" in _compact("".join(row)) and spouse_observed:
                    active_slots = _canonical_header_slots(
                        row, {"sequence": ("编号",), "data_provider": ("数据发生机构名称",)}
                    )
                    mode = "spouse_provider"
                    continue
                if not active_slots:
                    continue
                if mode == "mobile":
                    sequence = _sequence_value(row, active_slots)
                    if sequence is None:
                        if _nonempty(row):
                            _report_unkeyed_business_row(
                                parse_result,
                                dataset="mobile_phone_records",
                                row=row,
                                page=page,
                                table=table,
                                row_index=row_index,
                            )
                        continue
                    mobile_id = stable_record_id("personal_mobile_phone", sequence)
                    record = mobile_records.setdefault(
                        sequence,
                        {
                            "record_id": mobile_id,
                            "mobile_phone_record_id": mobile_id,
                            "sequence": sequence,
                            "source": "native_personal_detail_profile_table",
                            "source_refs": [],
                            "confidence": 1.0,
                        },
                    )
                    record["source_refs"].append(_source_ref(page, table, row=row_index))
                    raw_phone = _slot_value(row, active_slots, "mobile_phone")
                    phone = re.sub(r"\D", "", raw_phone)
                    phone_ref = _source_ref(
                        page, table, row=row_index, column=active_slots["mobile_phone"]
                    )
                    if not raw_phone:
                        _report_required_row_failure(
                            parse_result,
                            issue_code="candidate_b_mobile_row_unresolved",
                            dataset="mobile_phone_records",
                            sequence=sequence,
                            field_name="mobile_phone",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                            target_record_id=mobile_id,
                        )
                    elif raw_phone == "--":
                        _mark_source_absent(record, "mobile_phone", raw_phone)
                    elif (
                        any(character.isalpha() for character in raw_phone)
                        or not re.fullmatch(r"1[3-9]\d{9}", phone)
                    ):
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="mobile_phone_records",
                            target_record_id=mobile_id,
                            field_name="mobile_phone",
                            raw=raw_phone,
                            source_ref=phone_ref,
                            parser_stage="candidate_b_mobile_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="mobile_phone_records",
                            target_record_id=mobile_id,
                            field_name="mobile_phone",
                            value=phone,
                            raw=raw_phone,
                            source_ref=phone_ref,
                            parser_stage="candidate_b_mobile_canonical_slots",
                        )

                    raw_updated = _slot_value(row, active_slots, "information_updated_date")
                    updated = _date(raw_updated)
                    updated_ref = _source_ref(
                        page, table, row=row_index, column=active_slots["information_updated_date"]
                    )
                    if updated:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="mobile_phone_records",
                            target_record_id=mobile_id,
                            field_name="information_updated_date",
                            value=updated,
                            raw=raw_updated,
                            source_ref=updated_ref,
                            parser_stage="candidate_b_mobile_canonical_slots",
                        )
                    elif raw_updated == "--":
                        _mark_source_absent(record, "information_updated_date", raw_updated)
                    elif raw_updated:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="mobile_phone_records",
                            target_record_id=mobile_id,
                            field_name="information_updated_date",
                            raw=raw_updated,
                            source_ref=updated_ref,
                            parser_stage="candidate_b_mobile_canonical_slots",
                        )
                    elif not raw_updated:
                        _report_required_row_failure(
                            parse_result,
                            issue_code="candidate_b_mobile_row_unresolved",
                            dataset="mobile_phone_records",
                            sequence=sequence,
                            field_name="information_updated_date",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                            target_record_id=mobile_id,
                        )

                    provider = _slot_value(row, active_slots, "data_provider")
                    if provider == "--":
                        _mark_source_absent(record, "data_provider", provider)
                    elif provider:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="mobile_phone_records",
                            target_record_id=mobile_id,
                            field_name="data_provider",
                            value=provider,
                            raw=provider,
                            source_ref=_source_ref(
                                page, table, row=row_index, column=active_slots["data_provider"]
                            ),
                            parser_stage="candidate_b_mobile_canonical_slots",
                        )
                elif mode == "spouse":
                    values = {
                        role: _slot_value(row, active_slots, role)
                        for role in spouse_aliases
                    }
                    if not any(value and value != "--" for value in values.values()):
                        continue
                    spouse_id = stable_record_id("personal_spouse", 1)
                    if spouse_record is None:
                        spouse_record = {
                            "record_id": spouse_id,
                            "spouse_record_id": spouse_id,
                            "source": "native_personal_detail_profile_table",
                            "source_refs": [],
                            "confidence": 1.0,
                        }
                    spouse_record["source_refs"].append(_source_ref(page, table, row=row_index))
                    spouse_observed = True
                    for role, raw in values.items():
                        if raw == "--":
                            _mark_source_absent(spouse_record, role, raw)
                            continue
                        if not raw:
                            continue
                        ref = _source_ref(page, table, row=row_index, column=active_slots[role])
                        valid = True
                        normalized = raw
                        if role == "phone":
                            digits = re.sub(r"\D", "", raw)
                            valid = not any(character.isalpha() for character in raw) and 5 <= len(digits) <= 16
                        elif role == "name":
                            valid = bool(re.search(r"[\u3400-\u9fffA-Za-z]", raw)) and _compact(raw) not in _ACCOUNT_LABELS
                        elif role == "document_number":
                            compact_number = re.sub(r"\s+", "", raw)
                            valid = bool(re.fullmatch(r"[A-Za-z0-9()（）-]{4,40}", compact_number))
                            normalized = compact_number
                        if valid:
                            _merge_exact_observation(
                                parse_result,
                                spouse_record,
                                dataset="spouse_records",
                                target_record_id=spouse_id,
                                field_name=role,
                                value=normalized,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_spouse_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                spouse_record,
                                dataset="spouse_records",
                                target_record_id=spouse_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_spouse_canonical_slots",
                            )
                    if not spouse_record.get("name"):
                        _report_required_row_failure(
                            parse_result,
                            issue_code="candidate_b_spouse_row_unresolved",
                            dataset="spouse_records",
                            sequence=1,
                            field_name="name",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                            target_record_id=spouse_id,
                        )
                elif mode == "spouse_provider" and spouse_record is not None:
                    provider = _slot_value(row, active_slots, "data_provider")
                    if provider and provider != "--":
                        ref = _source_ref(
                            page, table, row=row_index, column=active_slots["data_provider"]
                        )
                        spouse_record["source_refs"].append(ref)
                        _merge_exact_observation(
                            parse_result,
                            spouse_record,
                            dataset="spouse_records",
                            target_record_id=str(spouse_record["spouse_record_id"]),
                            field_name="data_provider",
                            value=provider,
                            raw=provider,
                            source_ref=ref,
                            parser_stage="candidate_b_spouse_canonical_slots",
                        )
            previous_table_id = table_id or previous_table_id
    return {
        "mobile_phone_records": [mobile_records[key] for key in sorted(mobile_records)],
        "spouse_records": [spouse_record] if spouse_record is not None else [],
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
                    slots = _canonical_header_slots(
                        tuple(row),
                        {"sequence": ("编号",), "text": ("标注内容",), "added_date": ("添加日期",)},
                    )
                    if set(slots) != {"sequence", "text", "added_date"}:
                        _report_header_graph_failure(
                            parse_result,
                            dataset="annotation_statements",
                            page=page,
                            table=table,
                            row_index=row_index,
                            row=tuple(row),
                        )
                        continue
                    for data_index, data_row in enumerate(rows[row_index + 1 :], start=row_index + 1):
                        physical = tuple(data_row)
                        sequence = _sequence_value(physical, slots)
                        if sequence is None:
                            if _slot_value(physical, slots, "sequence"):
                                _report_unkeyed_business_row(
                                    parse_result,
                                    dataset="annotation_statements",
                                    row=physical,
                                    page=page,
                                    table=table,
                                    row_index=data_index,
                                )
                            break
                        target_id = stable_record_id("personal_detail_annotation", sequence)
                        record: dict[str, Any] = {
                            "id": target_id,
                            "annotation_id": target_id,
                            "sequence": sequence,
                            "note_type": "report_annotation",
                            "logical_page": page_number,
                            "source_page": source_page,
                            "source": "native_personal_detail_note_table",
                            "source_refs": [_source_ref(page, table, row=data_index)],
                            "confidence": float(getattr(table, "confidence", None) or 0.9),
                        }
                        for role, field_name, kind in (
                            ("text", "text", "text"),
                            ("added_date", "added_date", "date"),
                        ):
                            raw = _slot_value(physical, slots, role)
                            ref = _source_ref(page, table, row=data_index, column=slots[role])
                            value = _date(raw) if kind == "date" else _clean(raw)
                            if value in (None, ""):
                                _reject_exact_observation(
                                    parse_result,
                                    record,
                                    dataset="annotation_statements",
                                    target_record_id=f"annotation_statement:{target_id}",
                                    field_name=field_name,
                                    raw=raw,
                                    source_ref=ref,
                                    parser_stage="candidate_b_note_canonical_slots",
                                )
                            else:
                                _merge_exact_observation(
                                    parse_result,
                                    record,
                                    dataset="annotation_statements",
                                    target_record_id=f"annotation_statement:{target_id}",
                                    field_name=field_name,
                                    value=value,
                                    raw=raw,
                                    source_ref=ref,
                                    parser_stage="candidate_b_note_canonical_slots",
                                )
                        annotations.append(record)
                marker = next(
                    (candidate for candidate in ("异议标注", "本人声明", "机构说明") if candidate in compact),
                    None,
                )
                if marker is None or row_index + 1 >= len(rows):
                    continue
                # A standalone canonical heading is followed by one narrative
                # cell (and, for annotation forms, optionally one date cell).
                # Multiple free-floating cells are not concatenated into an
                # unstructured text blob.
                if any(label in compact for label in ("编号", "标注内容", "添加日期")):
                    continue
                next_compact = _compact("".join(rows[row_index + 1]))
                if all(label in next_compact for label in ("编号", "标注内容", "添加日期")):
                    continue
                values = [
                    (column, _clean(value))
                    for column, value in enumerate(rows[row_index + 1])
                    if _clean(value)
                ]
                if not values:
                    continue
                target = annotations if marker == "异议标注" else statements
                target_id = stable_record_id("personal_detail_note", marker, page_number, row_index)
                record = {
                    "id": target_id,
                    (
                        "annotation_id"
                        if marker == "异议标注"
                        else "statement_id"
                    ): target_id,
                    "note_type": {
                        "异议标注": "dispute_annotation",
                        "本人声明": "subject_statement",
                        "机构说明": "institution_statement",
                    }[marker],
                    "logical_page": page_number,
                    "source_page": source_page,
                    "source": "native_personal_detail_note_table",
                    "source_refs": [_source_ref(page, table, row=row_index + 1)],
                    "confidence": float(getattr(table, "confidence", None) or 0.9),
                }
                text_column, text = values[0]
                if len(values) > 2 or (len(values) == 2 and _date(values[1][1]) is None):
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_note_narrative_cells_unresolved",
                        dataset="annotation_statements",
                        sequence=len(target) + 1,
                        field_name="text",
                        row=tuple(rows[row_index + 1]),
                        page=page,
                        table=table,
                        row_index=row_index + 1,
                        target_record_id=f"annotation_statement:{target_id}",
                    )
                    _append_internal_field(record, "_unresolved_fields", "text")
                else:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="annotation_statements",
                        target_record_id=f"annotation_statement:{target_id}",
                        field_name="text",
                        value=text,
                        raw=text,
                        source_ref=_source_ref(page, table, row=row_index + 1, column=text_column),
                        parser_stage="candidate_b_note_narrative_slot",
                    )
                    if len(values) == 2:
                        date_column, date_raw = values[1]
                        record["added_date"] = _date(date_raw)
                        record.setdefault("source_refs_by_field", {})["added_date"] = [
                            _source_ref(page, table, row=row_index + 1, column=date_column)
                        ]
                target.append(record)
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


def _decode_exact_label_card(
    parse_result: Any,
    *,
    page: Any,
    table: Any,
    dataset: str,
    target_record_id: str,
    fields: Mapping[str, tuple[tuple[str, ...], str]],
) -> dict[str, Any]:
    """Decode a canonical label/value card without compacting physical cells."""

    rows = _table_rows(table)
    observations, unresolved = _exact_label_observations(rows)
    record: dict[str, Any] = {"source_refs": [_source_ref(page, table)]}
    for field_name, (labels, kind) in fields.items():
        candidates = [item for label in labels for item in observations.get(_compact(label), ())]
        if not candidates:
            label_location = next(
                ((row, column) for label, row, column in unresolved if label in {_compact(item) for item in labels}),
                None,
            )
            row_index = label_location[0] if label_location is not None else 0
            _report_required_row_failure(
                parse_result,
                issue_code="candidate_b_canonical_card_field_unresolved",
                dataset=dataset,
                sequence=1,
                field_name=field_name,
                row=tuple(rows[row_index]) if row_index < len(rows) else (),
                page=page,
                table=table,
                row_index=row_index,
                target_record_id=target_record_id,
            )
            _append_internal_field(record, "_unresolved_fields", field_name)
            continue
        for raw, row_index, column in candidates:
            ref = _source_ref(page, table, row=row_index, column=column)
            if _compact(raw) in {"-", "--", "---"}:
                _mark_source_absent(record, field_name, raw)
                continue
            if kind == "date":
                value = _date(raw)
            elif kind == "number":
                candidate = _number(raw)
                value = candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
            else:
                value = _clean(raw)
            if value in (None, ""):
                _reject_exact_observation(
                    parse_result,
                    record,
                    dataset=dataset,
                    target_record_id=target_record_id,
                    field_name=field_name,
                    raw=raw,
                    source_ref=ref,
                    parser_stage="candidate_b_canonical_label_card",
                )
                continue
            _merge_exact_observation(
                parse_result,
                record,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=field_name,
                value=value,
                raw=raw,
                source_ref=ref,
                parser_stage="candidate_b_canonical_label_card",
            )
    return record


_POSTPAID_CARD_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "institution": (("机构名称",), "text"),
    "business_type": (("业务类型",), "text"),
    "service_start_date": (("业务开通日期",), "date"),
    "payment_status": (("当前缴费状态",), "text"),
    "current_arrears_amount": (("当前欠费金额", "当前欠款金额"), "number"),
    "billing_month": (("记账年月",), "date"),
}


def _postpaid_id(page: Any, table: Any, record: Mapping[str, Any]) -> str:
    identity = (record.get("institution"), record.get("business_type"), record.get("billing_month"))
    if all(value not in (None, "") for value in identity):
        return stable_record_id("postpaid", *identity)
    return stable_record_id(
        "postpaid_unresolved",
        int(getattr(page, "page_number", 0) or 0),
        str(getattr(table, "table_id", "") or ""),
    )


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
            provisional_id = stable_record_id(
                "postpaid_unresolved",
                int(getattr(page, "page_number", 0) or 0),
                str(getattr(table, "table_id", "") or ""),
            )
            record = _decode_exact_label_card(
                parse_result,
                page=page,
                table=table,
                dataset="postpaid_records",
                target_record_id=provisional_id,
                fields=_POSTPAID_CARD_FIELDS,
            )
            postpaid_id = _postpaid_id(page, table, record)
            record.update(
                {
                    "postpaid_record_id": postpaid_id,
                    "sequence": len(records) + 1,
                    "reporting_amount_currency": "CNY",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_postpaid_table",
                    "confidence": float(getattr(table, "confidence", None) or 0.9),
                }
            )
            # Keep issue links valid if the canonical identity became readable.
            if postpaid_id != provisional_id:
                for issue in getattr(parse_result, "_personal_detail_extraction_issues", []) or []:
                    if isinstance(issue, dict) and issue.get("target_record_id") == provisional_id:
                        issue["target_record_id"] = postpaid_id
            records.append(record)
    return records


def _extract_postpaid_payment_history(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:3] for cell in row))
            if not all(marker in compact for marker in ("机构名称", "业务类型", "缴费记录")):
                continue
            observations, _unresolved = _exact_label_observations(rows)
            identity: dict[str, Any] = {}
            for field_name, label in (("institution", "机构名称"), ("business_type", "业务类型"), ("billing_month", "记账年月")):
                candidates = observations.get(label) or []
                distinct = {_compact(value) for value, _row, _column in candidates}
                if len(distinct) == 1:
                    raw = candidates[0][0]
                    identity[field_name] = _date(raw) if field_name == "billing_month" else _clean(raw)
            postpaid_id = _postpaid_id(page, table, identity)
            month_header = next(
                (
                    (row_index, {column: int(_compact(cell)) for column, cell in enumerate(row) if re.fullmatch(r"(?:[1-9]|1[0-2])", _compact(cell))})
                    for row_index, row in enumerate(rows)
                    if sum(bool(re.fullmatch(r"(?:[1-9]|1[0-2])", _compact(cell))) for cell in row) >= 5
                ),
                None,
            )
            if month_header is None:
                _report_header_graph_failure(
                    parse_result,
                    dataset="postpaid_payment_history",
                    page=page,
                    table=table,
                    row_index=0,
                    row=tuple(rows[0]) if rows else (),
                )
                continue
            month_header_index, months_by_column = month_header
            for row_index, row in enumerate(rows):
                if row_index <= month_header_index:
                    continue
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
                for column, month in months_by_column.items():
                    status = _compact(row[column]) if column < len(row) else ""
                    history_id = stable_record_id("postpaid_payment_history", postpaid_id, year_raw, month)
                    record = {
                        "record_id": history_id,
                        "postpaid_payment_history_id": history_id,
                        "postpaid_record_id": postpaid_id,
                        "institution": identity.get("institution"),
                        "business_type": identity.get("business_type"),
                        "year": int(year_raw),
                        "month": month,
                        "source": "native_personal_detail_postpaid_history",
                        "source_refs": [_source_ref(page, table, row=row_index, column=column)],
                        "confidence": float(getattr(table, "confidence", None) or 0.9),
                    }
                    if status in _STATUS_CODES:
                        record["status"] = status
                    else:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="postpaid_payment_history",
                            target_record_id=history_id,
                            field_name="status",
                            raw=status,
                            source_ref=_source_ref(page, table, row=row_index, column=column),
                            parser_stage="candidate_b_postpaid_month_slots",
                        )
                    records.append(record)
    return records


def _is_summary_anchor(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows for cell in row))
    return bool(
        "汇总" in compact or ("账户数" in compact and "首笔业务发放月份" in compact) or "最近1个月内的查询" in compact
    )


_CANONICAL_SUMMARY_LEAF_LABELS = frozenset(
    {
        "业务类型",
        "账户类型",
        "信息类型",
        "账户数",
        "记录数",
        "月份数",
        "首笔业务发放月份",
        "余额",
        "欠费金额",
        "涉及金额",
        "单月最高逾期/透支总额",
        "最长逾期/透支月数",
        "管理机构数",
        "发卡机构数",
        "授信总额",
        "单家机构最高授信额",
        "单家机构最低授信额",
        "已用额度",
        "透支余额",
        "最近6个月平均应还款",
        "最近6个月平均使用额度",
        "最近6个月平均透支余额",
        "担保金额",
        "还款责任金额",
        "贷款审批",
        "信用卡审批",
        "本人查询",
        "贷后管理",
        "担保资格审查",
        "特约商户实名审查",
    }
)
_CANONICAL_SUMMARY_GROUP_LABELS = frozenset(
    {
        "为个人",
        "为企业",
        "担保责任",
        "其他相关还款责任",
        "最近1个月内的查询机构数",
        "最近1个月内的查询次数",
        "最近2年内的查询次数",
    }
)
_CANONICAL_SUMMARY_HEADER_LABELS = frozenset(
    _compact(value)
    for value in (*_CANONICAL_SUMMARY_LEAF_LABELS, *_CANONICAL_SUMMARY_GROUP_LABELS)
)

_CANONICAL_SUMMARY_COLUMN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "信用业务概要": ("业务分组", "业务类型", "账户数", "首笔业务发放月份"),
    "逾期(透支)信息汇总": (
        "账户类型",
        "账户数",
        "月份数",
        "单月最高逾期/透支总额",
        "最长逾期/透支月数",
    ),
    "非循环贷账户信息汇总": (
        "管理机构数",
        "账户数",
        "授信总额",
        "余额",
        "最近6个月平均应还款",
    ),
    "循环贷账户一信息汇总": (
        "管理机构数",
        "账户数",
        "授信总额",
        "余额",
        "最近6个月平均应还款",
    ),
    "循环贷账户二信息汇总": (
        "管理机构数",
        "账户数",
        "授信总额",
        "余额",
        "最近6个月平均应还款",
    ),
    "贷记卡账户信息汇总": (
        "发卡机构数",
        "账户数",
        "授信总额",
        "单家机构最高授信额",
        "单家机构最低授信额",
        "已用额度",
        "最近6个月平均使用额度",
    ),
    "准贷记卡账户信息汇总": (
        "发卡机构数",
        "账户数",
        "授信总额",
        "单家机构最高授信额",
        "单家机构最低授信额",
        "透支余额",
        "最近6个月平均透支余额",
    ),
}


def _canonical_summary_columns(title: str, width: int) -> tuple[str, ...] | None:
    key = _compact(title).translate(str.maketrans({"（": "(", "）": ")"}))
    columns = _CANONICAL_SUMMARY_COLUMN_TEMPLATES.get(key)
    return columns if columns is not None and len(columns) == width else None


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
    if any(_compact(values[index]) not in _CANONICAL_SUMMARY_HEADER_LABELS for index in populated):
        return [""] * width
    expanded = [""] * width
    for position, start in enumerate(populated):
        end = populated[position + 1] if position + 1 < len(populated) else width
        for column in range(start, end):
            expanded[column] = values[start]
    return expanded


def _summary_business_rows(
    fragments: list[tuple[Any, Any, list[list[str]]]],
    *,
    title: str,
) -> tuple[
    list[tuple[Any, Any, int, list[str], list[str]]],
    list[tuple[Any, Any, int, list[str], str]],
]:
    width = max((len(row) for _page, _table, rows in fragments for row in rows), default=0)
    canonical_columns = _canonical_summary_columns(title, width)
    if canonical_columns is not None:
        output: list[tuple[Any, Any, int, list[str], list[str]]] = []
        rejected: list[tuple[Any, Any, int, list[str], str]] = []
        header_conflict = False
        for page, table, rows in fragments:
            for source_row_index, row in enumerate(rows):
                if _summary_row_has_values(row):
                    output.append(
                        (page, table, source_row_index, row, list(canonical_columns))
                    )
                    continue
                for column, value in enumerate(row[:width]):
                    compact = _compact(value)
                    if not compact or compact in _CANONICAL_SUMMARY_GROUP_LABELS or "汇总" in compact:
                        continue
                    if compact not in _CANONICAL_SUMMARY_HEADER_LABELS:
                        continue
                    if compact != _compact(canonical_columns[column]):
                        header_conflict = True
                        rejected.append(
                            (page, table, source_row_index, row, "canonical_template_header_conflict")
                        )
        if header_conflict:
            return [], rejected
        return output, rejected

    header_paths: list[list[str]] = [[] for _column in range(width)]
    output: list[tuple[Any, Any, int, list[str], list[str]]] = []
    rejected: list[tuple[Any, Any, int, list[str], str]] = []
    for page, table, rows in fragments:
        for source_row_index, row in enumerate(rows):
            nonempty = _nonempty(row)
            if len(nonempty) == 1 and "汇总" in _compact(nonempty[0]):
                continue
            if _summary_row_has_values(row):
                labels = ["/".join(path) for path in header_paths]
                missing_label_columns = [
                    column
                    for column, value in enumerate(row)
                    if _clean(value) and (column >= len(labels) or not labels[column])
                ]
                if missing_label_columns:
                    rejected.append(
                        (page, table, source_row_index, row, "value_without_exact_canonical_header")
                    )
                    continue
                output.append((page, table, source_row_index, row, labels))
                continue
            expanded = _expanded_summary_headers(row, width)
            if nonempty and not any(expanded):
                rejected.append((page, table, source_row_index, row, "unknown_summary_header"))
                continue
            distinct = {value for value in expanded if value}
            if len(distinct) == 1:
                label = next(iter(distinct))
                header_paths = [[label] for _column in range(width)]
                continue
            for column, label in enumerate(expanded):
                if label and (not header_paths[column] or header_paths[column][-1] != label):
                    header_paths[column].append(label)
    return output, rejected


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
        business_rows, rejected_rows = _summary_business_rows(fragments, title=title)
        if rejected_rows:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            for rejected_page, rejected_table, rejected_row_index, rejected_row, reason in rejected_rows:
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_summary_layout_unresolved",
                        message=(
                            "A canonical summary row could not be bound to the finite summary template; "
                            "its cells were withheld instead of inheriting positional labels."
                        ),
                        parser_stage="candidate_b_summary_canonical_slots",
                        target_dataset="personal_detail_summary_cells",
                        observed_value={"row": rejected_row, "reason": reason},
                        source_refs=(
                            _source_ref(
                                rejected_page,
                                rejected_table,
                                row=rejected_row_index,
                            ),
                        ),
                        reason_codes=(
                            "canonical_summary_table_observed",
                            reason,
                            "header_fill_inference_forbidden",
                            "normalized_cells_withheld",
                        ),
                    ),
                )
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
                if _compact(title) == "信用业务概要" and header == "业务分组":
                    # 贷款/信用卡/其他 is a merged visual group label, not the
                    # row's individualized business category.
                    continue
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
            sequence = len(records) + 1
            provisional_id = stable_record_id(
                "recovery_unresolved",
                int(getattr(page, "page_number", 0) or 0),
                str(getattr(table, "table_id", "") or ""),
            )
            record = _decode_exact_label_card(
                parse_result,
                page=page,
                table=table,
                dataset="recovery_records",
                target_record_id=provisional_id,
                fields={
                    "institution": (("管理机构",), "text"),
                    "business_type": (("业务种类",), "text"),
                    "debt_received_date": (("债权接收日期",), "date"),
                    "original_creditor": (("原债权人",), "text"),
                    "original_business_type": (("原债务业务种类",), "text"),
                    "debt_amount": (("债权金额",), "number"),
                    "transfer_repayment_status": (("债权转移时的还款状态",), "text"),
                    "account_status": (("账户状态",), "text"),
                    "balance": (("余额",), "number"),
                    "last_repayment_date": (("最近一次还款日期",), "date"),
                    "close_date": (("账户关闭日期",), "date"),
                },
            )
            received_date = record.get("debt_received_date")
            institution = record.get("institution")
            recovery_id = (
                stable_record_id("recovery", institution, received_date, sequence)
                if institution and received_date
                else provisional_id
            )
            record.update(
                {
                    "recovery_record_id": recovery_id,
                    "sequence": sequence,
                    "snapshot_date": next(
                        (
                            _date("-".join(match.groups()))
                            for row in rows
                            if (match := _AS_OF_RE.search(_clean(" ".join(row))))
                        ),
                        None,
                    ),
                    "reporting_amount_currency": "CNY",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_recovery_table",
                    "confidence": float(getattr(table, "confidence", None) or 0.9),
                }
            )
            if recovery_id != provisional_id:
                for issue in getattr(parse_result, "_personal_detail_extraction_issues", []) or []:
                    if isinstance(issue, dict) and issue.get("target_record_id") == provisional_id:
                        issue["target_record_id"] = recovery_id
            records.append(record)
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
                # Page-one business values are consumed only through exact
                # physical columns.  Compacting away empty cells and zipping
                # the survivors silently shifted every later value whenever
                # OCR missed one cell in the five-column header.
                if "被查询者" in _compact("".join(row)):
                    continue
                slots = _canonical_header_slots(
                    tuple(row),
                    {
                        "document_type": ("证件类型",),
                        "document_number": ("证件号码",),
                    },
                )
                if set(slots) != {"document_type", "document_number"}:
                    continue
                value_row = tuple(rows[row_index + 1])
                document_type = _slot_value(value_row, slots, "document_type")
                document_number = _slot_value(value_row, slots, "document_number")
                if document_type and document_number:
                    other_documents.append((document_type, document_number))
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

    def select(key: str, selected_type: str | None = None) -> str | None:
        observed = tuple(dict.fromkeys(value.strip() for value in field_candidates.get(key, []) if value.strip()))
        valid = tuple(value for value in observed if candidate_valid(key, value, selected_type))
        if len(valid) == 1:
            return valid[0]
        target_dataset = (
            "report_query"
            if key in {"query_institution", "query_reason"}
            else "report_metadata"
        )
        record_issue(
            parse_result,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="page_one_consensus_unresolved",
                message="Page-one header evidence was missing, invalid, or conflicting; the normalized value was withheld.",
                parser_stage="page_one_consensus",
                target_dataset=target_dataset,
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
