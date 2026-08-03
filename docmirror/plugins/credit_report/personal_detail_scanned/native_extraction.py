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
from collections import defaultdict
from typing import Any

from docmirror.plugins.credit_report.value_utils import stable_record_id

_DATE_RE = re.compile(r"(20\d{2})[.年/-]\s*(\d{1,2})(?:[.月/-]\s*(\d{1,2}))?")
_AS_OF_RE = re.compile(r"截至\s*(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_STATUS_CODES = frozenset({"*", "N", "1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "G", "M", "Z", "#"})

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
    status = {
        "正常": "active",
        "逾期": "active",
        "呆账": "active",
        "结清": "settled",
        "销户": "closed",
        "转出": "transferred_out",
        "结束": "closed",
        "未激活": "inactive",
    }.get(raw, raw)
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
    }
    if lifecycle:
        result["account_lifecycle_state"] = lifecycle
    if raw == "未激活":
        result["card_activation_state"] = "not_activated"
    elif raw == "呆账":
        result["credit_quality_status"] = "bad_debt"
    if raw == "逾期":
        result["current_overdue"] = True
        result["current_overdue_status"] = "overdue"
    elif raw in {"正常", "结清", "销户", "转出"}:
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
    return "账户标识" in compact and ("管理机构" in compact or "发卡机构" in compact)


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


def _extract_accounts(
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
                    continuation_check(current_table_id, candidate_table_id)
                    if callable(continuation_check)
                    else None
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


def _extract_credit_lines(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:4] for cell in row))
            if "授信协议标识" not in compact or "授信额度用途" not in compact:
                continue
            facts = _pairs(rows)
            identifier = _identifier(_field(facts, "授信协议标识"))
            if not identifier:
                continue
            due_raw = _field(facts, "到期日期")
            currency = _currency(_field(facts, "币种")) or "CNY"
            record = {
                "credit_line_id": stable_record_id("credit_line", identifier),
                "account_identifier": identifier,
                "institution": _clean(_field(facts, "管理机构")),
                "facility_type": _clean(_field(facts, "授信额度用途")),
                "effective_date": _date(_field(facts, "生效日期")),
                "due_date": _date(due_raw),
                "validity_type": "perpetual" if _compact(due_raw) == "长期" else "fixed_term",
                "total_limit": _number(_field(facts, "授信额度")),
                "used_limit": _number(_field(facts, "已用额度")),
                "limit_identifier": _identifier(_field(facts, "授信限额编号")),
                "currency": currency,
                "account_currency": currency,
                "reporting_amount_currency": "CNY",
                "amount_unit": "yuan",
                "reporting_amount_unit": "yuan",
                "status": "active",
                "source": "native_detail_credit_agreement_table",
                "source_refs": [_source_ref(page, table)],
                "confidence": 1.0,
            }
            records.append(record)
    return records


def _extract_liabilities(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:5] for cell in row))
            if "责任人类型" not in compact or "保证合同编号" not in compact:
                continue
            facts = _pairs(rows)
            contract_number = _identifier(_field(facts, "保证合同编号"))
            related_id = _identifier(_field(facts, "主业务借款人证件号码"))
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
                    "responsibility_amount": _number(_field(facts, "还款责任金额")),
                    "responsibility_amount_reported": True,
                    "contract_number": contract_number,
                    "snapshot_date": next(
                        (
                            _date("-".join(match.groups()))
                            for row in rows
                            if (match := _AS_OF_RE.search(_clean(" ".join(row))))
                        ),
                        None,
                    ),
                    "balance": _number(_field(facts, "余额")),
                    "five_tier_class": _clean(_field(facts, "五级分类")),
                    "overdue_months_or_repayment_status": _clean(_field(facts, "逾期月数", "还款状态")),
                    "currency": currency,
                    "reporting_amount_currency": currency,
                    "amount_unit": "yuan",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_liability_table",
                    "source_refs": [_source_ref(page, table)],
                    "confidence": 1.0,
                }
            )
    return records


def _extract_inquiries(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            header = _nonempty(rows[0])
            compact_header = _compact("".join(header))
            if not all(marker in compact_header for marker in ("查询日期", "查询机构", "查询原因")):
                continue
            for row_index, row in enumerate(rows[1:], start=1):
                values = _nonempty(row)
                if len(values) < 4 or not re.fullmatch(r"\d+", values[0]):
                    continue
                inquiry_date = _date(values[1])
                institution = values[2]
                source_reason = values[3]
                inquiry_type = (
                    "personal" if institution == "本人" or source_reason.startswith("本人查询") else "institution"
                )
                records.append(
                    {
                        "inquiry_id": stable_record_id(
                            "credit_inquiry", inquiry_type, values[0], inquiry_date, institution, source_reason
                        ),
                        "sequence": int(values[0]),
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
    return records


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


def _extract_residence_records(parse_result: Any) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    provider_rows: dict[int, tuple[str, dict[str, Any]]] = {}
    residence_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            header_index = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if all(
                        marker in _compact("".join(row))
                        for marker in ("编号", "居住地址", "住宅电话", "居住状况", "信息更新日期")
                    )
                ),
                None,
            )
            if header_index is not None:
                residence_table_id = str(getattr(table, "table_id", "") or residence_table_id)
                provider_header = next(
                    (
                        index
                        for index, row in enumerate(rows[header_index + 1 :], start=header_index + 1)
                        if "数据发生机构名称" in _compact("".join(row))
                    ),
                    len(rows),
                )
                for row_index, row in enumerate(rows[header_index + 1 : provider_header], start=header_index + 1):
                    sequence_match = re.search(r"\d+", str(row[0] if row else ""))
                    if sequence_match is None:
                        break
                    sequence = int(sequence_match.group(0))
                    address = _clean(row[1] if len(row) > 1 else "")
                    phone = _clean(row[2] if len(row) > 2 else "")
                    status = _clean(row[3] if len(row) > 3 else "")
                    updated = _date(row[4] if len(row) > 4 else "")
                    if not address:
                        continue
                    residence_id = stable_record_id("credit_residence", sequence, address, updated)
                    records[sequence] = {
                        "record_id": residence_id,
                        "residence_record_id": residence_id,
                        "sequence": sequence,
                        "address": address,
                        **({"residential_phone": phone} if phone and phone != "--" else {}),
                        **({"residence_status": status} if status and status != "--" else {}),
                        **({"information_updated_date": updated} if updated and updated != "--" else {}),
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_residence_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "confidence": 1.0,
                    }
                available_sequences = iter(sorted(records))
                used_provider_sequences: set[int] = set()
                for row_index, row in enumerate(rows[provider_header + 1 :], start=provider_header + 1):
                    nonempty = [(index, _clean(value)) for index, value in enumerate(row) if _clean(value)]
                    if not nonempty:
                        continue
                    sequence_match = re.search(r"\d+", nonempty[0][1]) if nonempty[0][0] == 0 else None
                    sequence = int(sequence_match.group(0)) if sequence_match else None
                    institution_parts = [
                        value
                        for index, value in nonempty
                        if not (
                            index == 0
                            and (
                                sequence_match is not None
                                or (len(value) <= 2 and len(nonempty) > 1)
                            )
                        )
                    ]
                    institution = " ".join(institution_parts).strip()
                    if sequence not in records:
                        sequence = next(
                            (candidate for candidate in available_sequences if candidate not in used_provider_sequences),
                            None,
                        )
                    if sequence is None or sequence not in records or not institution:
                        continue
                    institution = re.sub(r"^[A-Za-z0-9#*.,'\"\s]+(?=[\u3400-\u9fff])", "", institution)
                    provider_rows[sequence] = (institution, _source_ref(page, table, row=row_index))
                    used_provider_sequences.add(sequence)
            table_id = str(getattr(table, "table_id", "") or "")
            if (
                records
                and residence_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(residence_table_id, table_id) is True
            ):
                candidates: dict[int, tuple[str, dict[str, Any]]] = {}
                for row_index, row in enumerate(rows):
                    values = _nonempty(row)
                    if len(values) != 2 or not values[0].isdigit():
                        candidates = {}
                        break
                    candidates[int(values[0])] = (values[1], _source_ref(page, table, row=row_index))
                if candidates and set(candidates) <= set(records):
                    provider_rows.update(candidates)
    for sequence, record in records.items():
        provider = provider_rows.get(sequence)
        if provider:
            record["data_provider"] = provider[0]
            record["source_refs"].append(provider[1])
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
        (
            value
            for value in reversed(cells[4:])
            if len(re.sub(r"\D", "", value)) >= 7
        ),
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
    legal = re.match(r"(.{2,60}?(?:有限责任公司|股份有限公司|有限公司|工业公司|学校|公司))(?:\s+|(?=[省市县区镇路街]))?(.*)$", text)
    if legal:
        parsed["employer"] = legal.group(1).strip()
        trailing = legal.group(2).strip()
        if trailing and trailing != "--":
            parsed["employer_address"] = trailing
    elif text and text != "--":
        parsed["employer"] = text
    return parsed


def _extract_employment_records(parse_result: Any) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            basic_header = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if all(marker in _compact("".join(row)) for marker in ("编号", "工作单位", "单位性质", "单位电话"))
                ),
                None,
            )
            if basic_header is None:
                continue
            detail_header = next(
                (
                    index
                    for index, row in enumerate(rows[basic_header + 1 :], start=basic_header + 1)
                    if all(marker in _compact("".join(row)) for marker in ("编号", "职业", "行业", "职务", "职称"))
                ),
                None,
            )
            if detail_header is None:
                continue
            institution_header = next(
                (
                    index
                    for index, row in enumerate(rows[detail_header + 1 :], start=detail_header + 1)
                    if "数据发生机构名称" in _compact("".join(row))
                ),
                len(rows),
            )
            for row_index, row in enumerate(rows[basic_header + 1 : detail_header], start=basic_header + 1):
                sequence_match = re.search(r"\d+", str(row[0] if row else ""))
                if sequence_match is None:
                    continue
                sequence = int(sequence_match.group(0))
                parsed = _split_employment_basic_row(row)
                records[sequence] = {
                    "employment_record_id": stable_record_id("credit_employment", sequence, parsed),
                    "sequence": sequence,
                    **{key: value for key, value in parsed.items() if value and value != "--"},
                    "page": page_number,
                    "source_page": source_page,
                    "source": "native_personal_detail_employment_table",
                    "source_refs": [_source_ref(page, table, row=row_index)],
                    "confidence": 1.0,
                }
            for row_index, row in enumerate(rows[detail_header + 1 : institution_header], start=detail_header + 1):
                sequence_match = re.search(r"\d+", str(row[0] if row else ""))
                if sequence_match is None:
                    continue
                sequence = int(sequence_match.group(0))
                if sequence not in records:
                    continue
                cells = [_clean(value) for value in row]
                if len(cells) >= 7 and _date(cells[6]):
                    entry_match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", cells[5])
                    detail = {
                        "occupation": cells[1],
                        "industry": cells[2],
                        "position": cells[3],
                        "professional_title": cells[4],
                        "entry_year": int(entry_match.group(0)) if entry_match else None,
                        "information_updated_date": _date(cells[6]),
                    }
                    records[sequence].update(
                        {key: value for key, value in detail.items() if value and value != "--"}
                    )
                    records[sequence]["source_refs"].append(
                        _source_ref(page, table, row=row_index)
                    )
                    continue
                role_text = _clean(row[1] if len(row) > 1 else "")
                industry = "批发和零售业" if "批发和零售业" in role_text else ""
                occupation = role_text.replace(industry, "").strip(" #*-.")
                trailing = re.sub(
                    r"\s+",
                    " ",
                    " ".join(_clean(value) for value in row[3:] if _clean(value)),
                ).strip()
                date_match = re.search(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}", trailing)
                updated = _date(date_match.group(0)) if date_match else None
                before_date = trailing[: date_match.start()].strip() if date_match else trailing
                entry_match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", before_date)
                detail = {
                    "occupation": occupation,
                    "industry": industry,
                    "position": _clean(row[2] if len(row) > 2 else ""),
                    "professional_title": re.sub(r"[\s#*\-.]+", "", before_date),
                    "entry_year": int(entry_match.group(0)) if entry_match else None,
                    "information_updated_date": updated,
                }
                records[sequence].update({key: value for key, value in detail.items() if value and value != "--"})
                records[sequence]["source_refs"].append(_source_ref(page, table, row=row_index))
            used_provider_sequences: set[int] = set()
            for row_index, row in enumerate(rows[institution_header + 1 :], start=institution_header + 1):
                cells = [_clean(value) for value in row]
                first = cells[0] if cells else ""
                sequence_match = re.search(r"\d+", first)
                sequence = int(sequence_match.group(0)) if sequence_match else None
                institution = " ".join(
                    value
                    for index, value in enumerate(cells)
                    if value and not (index == 0 and sequence_match is not None)
                ).strip()
                if sequence not in records:
                    sequence = next(
                        (candidate for candidate in sorted(records) if candidate not in used_provider_sequences),
                        None,
                    )
                if sequence is None or sequence not in records or not institution:
                    continue
                institution = re.sub(r"^[A-Za-z0-9#*.,'\"\s]+(?=[\u3400-\u9fff])", "", institution)
                records[sequence]["data_provider"] = institution
                records[sequence]["source_refs"].append(_source_ref(page, table, row=row_index))
                used_provider_sequences.add(sequence)
    for sequence, record in records.items():
        record["employment_record_id"] = stable_record_id(
            "credit_employment",
            sequence,
            {key: value for key, value in record.items() if key not in {"source_refs", "confidence"}},
        )
    return [records[key] for key in sorted(records)]


def _extract_profile_detail_records(parse_result: Any) -> dict[str, list[dict[str, Any]]]:
    mobile_records: list[dict[str, Any]] = []
    spouse_records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            for row_index, row in enumerate(rows):
                compact = _compact("".join(row))
                if all(marker in compact for marker in ("编号", "手机号码", "信息更新日期", "数据发生机构名称")):
                    for ordinal, (data_index, data_row) in enumerate(
                        enumerate(rows[row_index + 1 :], start=row_index + 1),
                        start=1,
                    ):
                        cells = [_clean(value) for value in data_row]
                        phone = next((re.sub(r"\D", "", value) for value in cells if len(re.sub(r"\D", "", value)) == 11), "")
                        updated = next((value for value in cells if re.fullmatch(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}", value)), "")
                        if not phone or not updated:
                            break
                        sequence_match = re.fullmatch(r"\d{1,3}", cells[0] if cells else "")
                        sequence = int(sequence_match.group(0)) if sequence_match else ordinal
                        provider = next(
                            (
                                value
                                for value in reversed(cells)
                                if value and value not in {phone, updated} and re.search(r"[\u3400-\u9fff]", value)
                            ),
                            None,
                        )
                        mobile_id = stable_record_id("personal_mobile_phone", sequence, phone, updated)
                        mobile_records.append(
                            {
                                "record_id": mobile_id,
                                "mobile_phone_record_id": mobile_id,
                                "sequence": sequence,
                                "mobile_phone": phone,
                                "information_updated_date": _date(updated),
                                "data_provider": provider,
                                "source": "native_personal_detail_profile_table",
                                "source_refs": [_source_ref(page, table, row=data_index)],
                                "confidence": 1.0,
                            }
                        )
                if all(marker in compact for marker in ("姓名", "证件类型", "证件号码", "工作单位", "联系电话")):
                    if row_index + 1 >= len(rows):
                        continue
                    values = [_clean(value) for value in rows[row_index + 1]]
                    if not values or not values[0]:
                        continue
                    provider = None
                    if row_index + 3 < len(rows) and "数据发生机构名称" in _compact("".join(rows[row_index + 2])):
                        provider_values = _nonempty(rows[row_index + 3])
                        provider = provider_values[0] if provider_values else None
                    values.extend([""] * (5 - len(values)))
                    spouse_id = stable_record_id("personal_spouse", values[0], values[2])
                    spouse_records.append(
                        {
                            "record_id": spouse_id,
                            "spouse_record_id": spouse_id,
                            "name": values[0],
                            **({"document_type": values[1]} if values[1] and values[1] != "--" else {}),
                            **({"document_number": values[2]} if values[2] and values[2] != "--" else {}),
                            **({"employer": values[3]} if values[3] and values[3] != "--" else {}),
                            **({"phone": values[4]} if values[4] and values[4] != "--" else {}),
                            "data_provider": provider,
                            "source": "native_personal_detail_profile_table",
                            "source_refs": [_source_ref(page, table, row=row_index + 1)],
                            "confidence": 1.0,
                        }
                    )
    return {
        "mobile_phone_records": mobile_records,
        "spouse_records": spouse_records,
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
        "汇总" in compact
        or ("账户数" in compact and "首笔业务发放月份" in compact)
        or "最近1个月内的查询" in compact
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
            if (
                next_page_number != previous_page_number + 1
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
                "source_refs": [_source_ref(fragment_page, fragment_table) for fragment_page, fragment_table, _ in fragments],
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
    subject_name = id_type = id_number = query_institution = query_reason = None
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
                    subject_name, id_type, id_number, query_institution, query_reason = values[:5]
                elif labels == ["证件类型", "证件号码"] and len(values) >= 2:
                    other_documents.append((values[0], values[1]))
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
    if id_type and id_number:
        identities.append(
            {
                "identity_document_id": stable_record_id("identity_document", "primary", id_type, id_number),
                "sequence": 1,
                "holder_name": subject_name,
                "document_type": id_type,
                "document_number": id_number,
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
    """Return canonical base collections for a native detailed report."""
    account_loader = getattr(parse_result, "account_collections", None)
    accounts, repayments, _account_event_rows = (
        account_loader() if callable(account_loader) else _extract_accounts(parse_result)
    )
    credit_lines = _extract_credit_lines(parse_result)
    liabilities = _extract_liabilities(parse_result)
    inquiries = _extract_inquiries(parse_result)
    public_records = _extract_public_records(parse_result)
    return {
        "credit_accounts": accounts,
        "credit_lines": credit_lines,
        "repayment_liability_records": liabilities,
        "repayment_records": repayments,
        "inquiry_records": inquiries,
        "public_records": public_records,
        "credit_summary": {
            "source": "personal_detail_native_tables",
            "reported_account_count": len(accounts),
            "projected_account_count": len(accounts),
            "repayment_liability_count": len(liabilities),
            "inquiry_count": len(inquiries),
        },
    }


def extract_personal_detail_section_content(parse_result: Any, full_text: str) -> dict[str, Any]:
    """Return supplemental canonical datasets and lossless source rows."""
    from docmirror.plugins.credit_report.business_records import derive_overdue_records
    from docmirror.plugins.credit_report.scanned_business import extract_scanned_credit_business

    scanned_loader = getattr(parse_result, "scanned_business", None)
    scanned_compatible = (
        scanned_loader(full_text)
        if callable(scanned_loader)
        else extract_scanned_credit_business(parse_result, full_text)
    )
    account_loader = getattr(parse_result, "account_collections", None)
    accounts, repayments, account_events = (
        account_loader() if callable(account_loader) else _extract_accounts(parse_result)
    )
    native_loader = getattr(parse_result, "native_business", None)
    native_compatible = (
        native_loader(full_text)
        if callable(native_loader)
        else extract_personal_detail_native_business(parse_result, full_text)
    )
    annotations, statements = _extract_personal_notes(parse_result)
    employment_records = _extract_employment_records(parse_result)
    residence_records = _extract_residence_records(parse_result)
    summary_records, summary_cells = _extract_summary_datasets(parse_result)
    subject_profile = dict(scanned_compatible.get("subject_profile") or {})
    profile_facts = {
        key: value.get("normalized_value", value.get("value"))
        for key, value in subject_profile.items()
        if isinstance(value, dict) and value.get("normalized_value", value.get("value")) not in (None, "")
    }
    if profile_facts.get("birth_date"):
        birth_match = re.fullmatch(
            r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})",
            _compact(profile_facts["birth_date"]),
        )
        if birth_match:
            profile_facts["birth_date"] = (
                f"{int(birth_match.group(1)):04d}-{int(birth_match.group(2)):02d}-{int(birth_match.group(3)):02d}"
            )
    profile_detail_records = _extract_profile_detail_records(parse_result)
    header = extract_personal_detail_common_datasets(parse_result, full_text)
    datasets: dict[str, list[dict[str, Any]]] = {
        **header,
        "recovery_records": _extract_recovery_records(parse_result),
        "postpaid_records": _extract_postpaid_records(parse_result),
        "postpaid_payment_history": _extract_postpaid_payment_history(parse_result),
        "personal_detail_account_events": account_events,
        "personal_detail_summary_records": summary_records,
        "personal_detail_summary_cells": summary_cells,
        **profile_detail_records,
        "residence_records": residence_records or list(scanned_compatible.get("residence_records") or []),
        "employment_records": employment_records or list(scanned_compatible.get("employment_records") or []),
        "annotations": annotations or list(scanned_compatible.get("annotations") or []),
        "statements": statements or list(scanned_compatible.get("statements") or []),
    }
    expected_counts = {
        **{name: len(rows) for name, rows in datasets.items()},
        "credit_lines": len(native_compatible.get("credit_lines") or []),
        "repayment_liability_records": len(native_compatible.get("repayment_liability_records") or []),
        "repayment_records": len(repayments),
        "overdue_records": len(derive_overdue_records(accounts, repayments)),
        "public_records": len(native_compatible.get("public_records") or []),
    }
    return {
        "facts": {
            "subject_profile": subject_profile,
            **profile_facts,
            "canonical_dataset_schema": "personal_credit_report_detailed.v1",
            **{f"personal_detail_expected_{name}_count": count for name, count in expected_counts.items()},
        },
        "datasets": {name: rows for name, rows in datasets.items() if rows},
    }


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
