# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover high-confidence split debit/credit ledgers from canonical evidence atoms."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from statistics import median
from typing import Any

_DATE_ANY_RE = re.compile(r"20\d{6}|20\d{2}[-/]\d{1,2}[-/]\d{1,2}")
_MONEY_RE = re.compile(r"^-?\d[\d,]*\.\d{2}$")
_MONEY_ANY_RE = re.compile(r"-?\d[\d,]*\.\d{2}")
_COMPOSITE_MARKERS = ("支出", "收入", "账户余额")
_RECOVERY_CACHE_KEY = "_bank_evidence_atom_recovery"
_GEOMETRY_FOOTER_MARKERS = (
    "本页合计",
    "本页支出",
    "本页收入",
    "总收入笔数",
    "总收入金额",
    "总支出笔数",
    "总支出金额",
    "当前账单借方发生数",
    "当前账单贷方发生数",
    "本月累计借方发生数",
    "本月累计贷方发生数",
    "借方发生额汇总",
    "贷方发生额汇总",
    "回单编号",
    "打印完毕",
    "友情提示",
    "风险提示",
    "本回单",
)
_OUTPUT_HEADER = [
    "序号",
    "交易日期",
    "交易流水号",
    "支出金额",
    "收入金额",
    "余额",
    "对方账号",
    "对方户名",
    "对方行号",
    "对方行名",
    "交易渠道",
    "用途",
    "摘要",
]

_POSITIONED_BLOCK_HEADER_MARKERS = (
    "\u5e8f\u53f7",
    "\u4ea4\u6613\u65e5\u671f",
    "\u4ea4\u6613\u91d1\u989d",
    "\u8d26\u6237\u4f59\u989d",
)
_POSITIONED_BLOCK_INCOME_MARKERS = (
    "\u5165\u8d26",
    "\u5b58\u5165",
    "\u6536\u6b3e",
    "\u8f6c\u5165",
    "\u7ed3\u606f",
    "\u6536\u5165",
)
_POSITIONED_BLOCK_EXPENSE_MARKERS = (
    "\u652f\u53d6",
    "\u652f\u51fa",
    "\u8f6c\u51fa",
    "\u6d88\u8d39",
    "\u6263\u6b3e",
    "\u624b\u7eed\u8d39",
    "\u8fd8\u6b3e",
)
_POSITIONED_BLOCK_ACCOUNT_RE = re.compile(r"(?<![\d*])\d[\d*]{5,22}\d(?![\d*])")
_COLUMN_AGGREGATE_HEADER_MARKERS = {
    "sequence": ("序号", "编号"),
    "summary": ("摘要", "交易摘要", "备注"),
    "currency": ("币别", "币种"),
    "cash": ("钞汇", "钞/汇"),
    "date": ("交易日期", "交易时间", "记账日期", "日期"),
    "amount": ("交易金额", "发生额", "金额"),
    "balance": ("账户余额", "余额"),
    "location": ("交易地点/附言", "交易地点", "附言"),
    "counterparty": ("对方账号与户名", "对方账户", "对方账号", "对手方"),
}


@dataclass(frozen=True)
class PositionedBlockRecovery:
    """A page-positioned record-block recovery result for candidate selection."""

    tables: list[list[list[str]]]
    row_sources: list[dict[str, Any]]
    expected_rows: int


def recover_positioned_record_block_bank_tables(parse_result: Any) -> PositionedBlockRecovery:
    """Recover rotated or column-major ledgers where one positioned block is one record.

    Some native PDFs write each visual ledger row as a vertically arranged text
    block.  Their table grid can therefore collapse into one long value per
    column even though the positioned text has already retained the record
    boundary.  This recovery path uses only generic ledger header, date, money,
    and balance evidence; it has no institution-specific rules.
    """
    tables: list[list[list[str]]] = []
    row_sources: list[dict[str, Any]] = []
    expected_rows = 0
    previous_page_record: dict[str, Any] | None = None
    page_candidates: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for page_id, atoms in sorted(_positioned_atoms_by_page(parse_result).items()):
        page_records = [record for atom in atoms if (record := _positioned_block_record(page_id, atom)) is not None]
        if len(page_records) < 3:
            aggregate_records = _column_aggregate_block_records(parse_result, page_id, atoms)
            if aggregate_records:
                page_records = aggregate_records
        if page_records:
            _sort_positioned_block_records(page_records)
        page_candidates.append((page_id, atoms, page_records))

    has_strong_layout = any(
        len(page_records) >= 3 or any(_is_positioned_block_header(str(atom.get("text") or "")) for atom in atoms)
        for _, atoms, page_records in page_candidates
    )
    if not has_strong_layout:
        return PositionedBlockRecovery(tables=[], row_sources=[], expected_rows=0)

    for page_index, (page_id, atoms, page_records) in enumerate(page_candidates):
        if not _positioned_page_candidate_supported(page_candidates, page_index):
            continue
        expected_rows += len(page_records)
        source_headers = _positioned_source_headers(atoms)
        for record in page_records:
            if not isinstance(record.get("source_raw"), dict):
                record["source_raw"] = _positioned_record_source_raw(record, source_headers)
        _infer_positioned_block_directions(page_records, preceding_record=previous_page_record)
        previous_page_record = page_records[-1]
        rows: list[list[str]] = []
        for record in page_records:
            direction = str(record.get("direction") or "")
            if direction not in {"income", "expense"}:
                continue
            amount = str(record["amount"])
            rows.append(
                [
                    str(record.get("sequence_no") or ""),
                    str(record["date"]),
                    "",
                    amount if direction == "expense" else "",
                    amount if direction == "income" else "",
                    str(record["balance"]),
                    str(record.get("counter_account") or ""),
                    str(record.get("counter_party") or ""),
                    "",
                    "",
                    "",
                    "",
                    str(record.get("summary") or ""),
                ]
            )
            row_sources.append(_positioned_block_row_source(page_id, record, rows[-1]))
        if rows:
            tables.append([_OUTPUT_HEADER, *rows])
    return PositionedBlockRecovery(tables=tables, row_sources=row_sources, expected_rows=expected_rows)


def _positioned_page_candidate_supported(
    page_candidates: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    page_index: int,
) -> bool:
    """Accept a short page only when document-local evidence proves the ledger continuation."""
    _page_id, atoms, records = page_candidates[page_index]
    if not records:
        return False
    if len(records) >= 3:
        return True
    if any(_is_positioned_block_header(str(atom.get("text") or "")) for atom in atoms):
        return True
    if page_index > 0:
        previous = page_candidates[page_index - 1][2]
        if previous and _is_sequence_continuation(previous[-1], records[0]):
            return True
    if page_index + 1 < len(page_candidates):
        following = page_candidates[page_index + 1][2]
        if following and _is_sequence_continuation(records[-1], following[0]):
            return True
    return False


def _column_aggregate_block_records(
    parse_result: Any,
    page_id: str,
    atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join positioned record spines with a physical table collapsed by column.

    PDF producers sometimes retain the visual text block for the left side of a
    transaction (sequence, summary and date), while their table extractor puts
    every value for each remaining column into one newline-delimited cell. The
    two artifacts originate from the same page and share an ordinal record
    boundary, so they can be joined without bank-specific layout constants.
    """
    if not any(_is_column_aggregate_spine_header(atom) for atom in atoms):
        return []
    spines = [record for atom in atoms if (record := _positioned_record_spine(page_id, atom)) is not None]
    if len(spines) < 3:
        return []
    columns = _collapsed_page_table_columns(parse_result, page_id)
    required = ("sequence", "summary", "date", "amount", "balance")
    if any(not columns.get(name) for name in required):
        return []
    expected_count = len(spines)
    if any(len(columns[name]) != expected_count for name in required):
        return []
    source_headers = _collapsed_page_table_headers(parse_result, page_id)
    records: list[dict[str, Any]] = []
    for index, spine in enumerate(spines):
        sequence = _positioned_block_sequence([columns["sequence"][index]])
        if sequence is None or sequence != spine["sequence_no"]:
            return []
        date = _normalize_block_date(columns["date"][index])
        amount_match = _MONEY_ANY_RE.search(columns["amount"][index])
        balance_match = _MONEY_ANY_RE.search(columns["balance"][index])
        if not date or amount_match is None or balance_match is None:
            return []
        amount_raw = amount_match.group(0).replace(",", "")
        try:
            amount = abs(float(amount_raw))
            balance = float(balance_match.group(0).replace(",", ""))
        except ValueError:
            return []
        if amount <= 0:
            return []
        row_atoms = _positioned_source_row_atoms(parse_result, page_id, spines, index)
        column_axis = _positioned_column_axis(spines)
        counterparty = ""
        counter_account = ""
        counterparty_values = columns.get("counterparty") or []
        if len(counterparty_values) == expected_count:
            counterparty_value = columns["counterparty"][index]
            counter_account = _positioned_block_counter_account(counterparty_value)
            counterparty = _positioned_block_counterparty(counterparty_value, counter_account)
        else:
            counterparty_value = _positioned_source_counterparty(row_atoms, column_axis)
            counter_account = _positioned_block_counter_account(counterparty_value)
            counterparty = _positioned_block_counterparty(counterparty_value, counter_account)
        summary = columns["summary"][index] or spine["summary"]
        record = {
            "page_id": page_id,
            "atom": spine["atom"],
            "sequence_no": sequence,
            "date": date,
            "amount": f"{amount:.2f}",
            "amount_raw": columns["amount"][index],
            "balance": f"{balance:.2f}",
            "balance_raw": columns["balance"][index],
            "summary": summary,
            "direction": _positioned_block_direction(amount_raw, summary),
            "counter_account": counter_account,
            "counter_party": counterparty,
        }
        record["source_raw"] = _column_aggregate_source_raw(
            parse_result,
            page_id,
            spines,
            index,
            source_headers,
            columns,
            row_atoms=row_atoms,
            column_axis=column_axis,
        )
        records.append(record)
    return records


def _is_column_aggregate_spine_header(atom: dict[str, Any]) -> bool:
    compact = re.sub(r"\s+", "", str(atom.get("text") or ""))
    return all(marker in compact for marker in ("序号", "摘要", "交易日期"))


def _positioned_record_spine(page_id: str, atom: dict[str, Any]) -> dict[str, Any] | None:
    text = str(atom.get("text") or "").strip()
    if not text or _is_column_aggregate_spine_header(atom) or _is_geometry_footer_text(text):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sequence = _positioned_block_sequence(lines)
    date_line = next((line for line in lines if _DATE_ANY_RE.fullmatch(line)), "")
    if sequence is None or not date_line:
        return None
    date = _normalize_block_date(date_line)
    if not date:
        return None
    return {
        "page_id": page_id,
        "atom": atom,
        "sequence_no": sequence,
        "summary": lines[1] if len(lines) > 1 else "",
        "date": date,
    }


def _collapsed_page_table_columns(parse_result: Any, page_id: str) -> dict[str, list[str]]:
    """Return newline-separated physical columns keyed by generic ledger role."""
    try:
        page_number = int(page_id.rsplit(":", 1)[-1])
    except ValueError:
        return {}
    page = next(
        (
            candidate
            for candidate in getattr(parse_result, "pages", []) or []
            if int(getattr(candidate, "source_page_number", 0) or getattr(candidate, "page_number", 0) or 0)
            == page_number
        ),
        None,
    )
    if page is None:
        return {}
    for table in getattr(page, "tables", []) or []:
        headers = [str(header or "").strip() for header in getattr(table, "headers", []) or []]
        header_map = _column_aggregate_header_map(headers)
        if not all(name in header_map for name in ("sequence", "summary", "date", "amount", "balance")):
            continue
        rows = list(getattr(table, "rows", []) or [])
        if len(rows) != 1:
            continue
        values = [
            str(getattr(cell, "cleaned", None) or getattr(cell, "text", "") or "")
            for cell in getattr(rows[0], "cells", []) or []
        ]
        if len(values) < len(headers):
            continue
        return {
            name: _split_collapsed_column(values[index]) for name, index in header_map.items() if index < len(values)
        }
    return {}


def _column_aggregate_header_map(headers: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, header in enumerate(headers):
        compact = re.sub(r"\s+", "", header)
        for name, markers in _COLUMN_AGGREGATE_HEADER_MARKERS.items():
            if name not in result and any(marker in compact for marker in markers):
                result[name] = index
    return result


def _collapsed_page_table_headers(parse_result: Any, page_id: str) -> list[str]:
    """Return the original physical headers for a column-aggregated page."""
    try:
        page_number = int(page_id.rsplit(":", 1)[-1])
    except ValueError:
        return []
    page = next(
        (
            candidate
            for candidate in getattr(parse_result, "pages", []) or []
            if int(getattr(candidate, "source_page_number", 0) or getattr(candidate, "page_number", 0) or 0)
            == page_number
        ),
        None,
    )
    if page is None:
        return []
    for table in getattr(page, "tables", []) or []:
        headers = [str(header or "").strip() for header in getattr(table, "headers", []) or []]
        header_map = _column_aggregate_header_map(headers)
        if all(name in header_map for name in ("sequence", "summary", "date", "amount", "balance")):
            return headers
    return []


def _column_aggregate_source_raw(
    parse_result: Any,
    page_id: str,
    spines: list[dict[str, Any]],
    index: int,
    headers: list[str],
    columns: dict[str, list[str]],
    *,
    row_atoms: list[dict[str, Any]] | None = None,
    column_axis: int = 0,
) -> dict[str, str]:
    """Rebuild one source row using the physical table's original columns."""
    if not headers:
        return {}
    header_map = _column_aggregate_header_map(headers)
    row_atoms = (
        row_atoms if row_atoms is not None else _positioned_source_row_atoms(parse_result, page_id, spines, index)
    )
    raw: dict[str, str] = {}
    for column_index, header in enumerate(headers):
        role = next((name for name, value in header_map.items() if value == column_index), "")
        values = columns.get(role) or []
        if len(values) == len(spines):
            raw[header] = str(values[index] or "").strip()
        elif role == "location":
            raw[header] = _positioned_source_location(row_atoms, column_axis)
        elif role == "counterparty":
            raw[header] = _positioned_source_counterparty(row_atoms, column_axis)
        else:
            raw[header] = ""
    return raw


def _positioned_source_row_atoms(
    parse_result: Any,
    page_id: str,
    spines: list[dict[str, Any]],
    index: int,
) -> list[dict[str, Any]]:
    """Return token atoms inside one positioned record boundary."""
    token_atoms = _atoms_by_page(parse_result).get(page_id, [])
    if not token_atoms or index >= len(spines):
        return []
    centers = [
        (_x_center(record["atom"]), _y_center(record["atom"]))
        for record in spines
        if isinstance(record.get("atom"), dict) and isinstance(record["atom"].get("bbox"), list)
    ]
    if not centers:
        return []
    x_span = max(x for x, _ in centers) - min(x for x, _ in centers)
    y_span = max(y for _, y in centers) - min(y for _, y in centers)
    record_axis = 0 if x_span > y_span else 1
    current = centers[index][record_axis]
    ordered = sorted(center[record_axis] for center in centers)
    position = ordered.index(current)
    lower = (ordered[position - 1] + current) / 2 if position else current - 10.0
    upper = (current + ordered[position + 1]) / 2 if position + 1 < len(ordered) else current + 10.0
    result: list[dict[str, Any]] = []
    for atom in token_atoms:
        bbox = atom.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        atom_center = _x_center(atom) if record_axis == 0 else _y_center(atom)
        if lower < atom_center < upper:
            result.append(atom)
    return result


def _positioned_column_axis(spines: list[dict[str, Any]]) -> int:
    centers = [
        (_x_center(record["atom"]), _y_center(record["atom"]))
        for record in spines
        if isinstance(record.get("atom"), dict) and isinstance(record["atom"].get("bbox"), list)
    ]
    if not centers:
        return 0
    x_span = max(x for x, _ in centers) - min(x for x, _ in centers)
    y_span = max(y for _, y in centers) - min(y for _, y in centers)
    return 1 if x_span > y_span else 0


def _positioned_source_location(row_atoms: list[dict[str, Any]], column_axis: int) -> str:
    """Extract source location/remark text between balance and counterparty."""
    money_atoms = [atom for atom in row_atoms if _MONEY_ANY_RE.fullmatch(str(atom.get("text") or "").strip())]
    account_atoms = [
        atom
        for atom in row_atoms
        if _POSITIONED_BLOCK_ACCOUNT_RE.search(str(atom.get("text") or ""))
        and not _DATE_ANY_RE.fullmatch(str(atom.get("text") or "").strip())
    ]
    if len(money_atoms) < 2 or not account_atoms:
        return ""

    def coordinate(atom: dict[str, Any]) -> float:
        return _x_center(atom) if column_axis == 0 else _y_center(atom)

    money_atoms.sort(key=coordinate)
    balance_atom = money_atoms[1]
    counter_atom = min(account_atoms, key=lambda atom: abs(coordinate(atom) - coordinate(balance_atom)))
    left = min(coordinate(balance_atom), coordinate(counter_atom))
    right = max(coordinate(balance_atom), coordinate(counter_atom))
    selected = [
        atom
        for atom in row_atoms
        if left < coordinate(atom) < right
        and atom is not balance_atom
        and atom is not counter_atom
        and not _MONEY_ANY_RE.fullmatch(str(atom.get("text") or "").strip())
    ]
    line_axis = 1 if column_axis == 0 else 0
    selected.sort(
        key=lambda atom: (
            _y_center(atom) if line_axis == 1 else _x_center(atom),
            coordinate(atom),
        )
    )
    lines: list[list[str]] = []
    line_centers: list[float] = []
    for atom in selected:
        text = str(atom.get("text") or "").strip()
        if not text:
            continue
        line_center = _y_center(atom) if line_axis == 1 else _x_center(atom)
        if line_centers and abs(line_center - line_centers[-1]) <= 1.5:
            lines[-1].append(text)
        else:
            line_centers.append(line_center)
            lines.append([text])
    return "\n".join("".join(line) for line in lines)


def _positioned_source_counterparty(row_atoms: list[dict[str, Any]], column_axis: int) -> str:
    """Return the original counterparty cell when a collapsed column lost blanks."""
    account_atoms = [
        atom
        for atom in row_atoms
        if _POSITIONED_BLOCK_ACCOUNT_RE.search(str(atom.get("text") or ""))
        and not _DATE_ANY_RE.fullmatch(str(atom.get("text") or "").strip())
    ]
    if not account_atoms:
        return ""

    def coordinate(atom: dict[str, Any]) -> float:
        return _x_center(atom) if column_axis == 0 else _y_center(atom)

    money_atoms = [atom for atom in row_atoms if _MONEY_ANY_RE.fullmatch(str(atom.get("text") or "").strip())]
    if len(money_atoms) >= 2:
        balance_coordinate = sorted(coordinate(atom) for atom in money_atoms)[1]
        after_balance = [atom for atom in account_atoms if coordinate(atom) > balance_coordinate]
        counter_atom = min(after_balance or account_atoms, key=lambda atom: abs(coordinate(atom) - balance_coordinate))
    else:
        counter_atom = min(account_atoms, key=coordinate)
    return str(counter_atom.get("text") or "").strip()


def _split_collapsed_column(value: str) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def _is_positioned_block_header(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return all(marker in compact for marker in _POSITIONED_BLOCK_HEADER_MARKERS)


def _positioned_block_record(page_id: str, atom: dict[str, Any]) -> dict[str, Any] | None:
    text = str(atom.get("text") or "").strip()
    if not text or _is_positioned_block_header(text) or _is_geometry_footer_text(text):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Account numbers can contain an eight-digit sequence beginning with
    # ``20``. A ledger date is a complete line within this rotated block;
    # scanning the whole block would mistake that account substring for a
    # second date and discard an otherwise auditable transaction.
    date_lines = [line for line in lines if _DATE_ANY_RE.fullmatch(line)]
    money = list(_MONEY_ANY_RE.finditer(text))
    if len(date_lines) != 1 or len(money) < 2 or len(lines) < 4:
        return None
    date = _normalize_block_date(date_lines[0])
    if not date:
        return None
    sequence_no = _positioned_block_sequence(lines)
    date_line = next((index for index, line in enumerate(lines) if _DATE_ANY_RE.fullmatch(line)), -1)
    if sequence_no is None and date_line < 1:
        return None
    amount_raw = money[0].group(0).replace(",", "")
    balance_raw = money[1].group(0).replace(",", "")
    try:
        amount = abs(float(amount_raw))
        balance = float(balance_raw)
    except ValueError:
        return None
    if amount <= 0:
        return None
    summary = lines[1] if sequence_no is not None and len(lines) > 1 else lines[0]
    direction = _positioned_block_direction(amount_raw, summary)
    counter_account = _positioned_block_counter_account(text)
    counter_party = _positioned_block_counterparty(text, counter_account)
    return {
        "page_id": page_id,
        "atom": atom,
        "source_lines": lines,
        "sequence_no": sequence_no,
        "date": date,
        "amount": f"{amount:.2f}",
        "amount_raw": amount_raw,
        "balance": f"{balance:.2f}",
        "balance_raw": balance_raw,
        "summary": summary,
        "direction": direction,
        "counter_account": counter_account,
        "counter_party": counter_party,
    }


def _positioned_source_headers(atoms: list[dict[str, Any]]) -> list[str]:
    """Read a vertical source header without falling back to a fixed schema."""
    header = next(
        (atom for atom in atoms if _is_positioned_block_header(str(atom.get("text") or ""))),
        None,
    )
    if header is None:
        return []
    return [line.strip() for line in str(header.get("text") or "").splitlines() if line.strip()]


def _positioned_record_source_raw(record: dict[str, Any], headers: list[str]) -> dict[str, str]:
    """Build source raw values from a positioned record and its real header."""
    lines = [str(value or "").strip() for value in record.get("source_lines") or []]
    if headers and len(lines) == len(headers):
        return {header: lines[index] for index, header in enumerate(headers)}

    fallback = {
        "序号": str(record.get("sequence_no") or ""),
        "摘要": str(record.get("summary") or ""),
        "交易日期": str(record.get("date") or ""),
        "交易金额": str(record.get("amount_raw") or record.get("amount") or ""),
        "账户余额": str(record.get("balance_raw") or record.get("balance") or ""),
    }
    account = str(record.get("counter_account") or "").strip()
    party = str(record.get("counter_party") or "").strip()
    if account or party:
        fallback["对方账号与户名"] = f"{account}/{party}" if account and party else account or party
    if headers:
        role_map = _column_aggregate_header_map(headers)
        mapped: dict[str, str] = {}
        for header_index, header in enumerate(headers):
            role = next((name for name, value in role_map.items() if value == header_index), "")
            if role == "sequence":
                mapped[header] = fallback["序号"]
            elif role == "summary":
                mapped[header] = fallback["摘要"]
            elif role == "date":
                mapped[header] = fallback["交易日期"]
            elif role == "amount":
                mapped[header] = fallback["交易金额"]
            elif role == "balance":
                mapped[header] = fallback["账户余额"]
            elif role == "counterparty":
                mapped[header] = fallback.get("对方账号与户名", "")
            else:
                mapped[header] = ""
        return mapped
    return fallback


def _normalize_block_date(value: str) -> str:
    compact = re.sub(r"\D", "", value)
    if not re.fullmatch(r"20\d{6}", compact):
        return ""
    return f"{compact[:4]}{compact[4:6]}{compact[6:8]}"


def _positioned_block_sequence(lines: list[str]) -> int | None:
    if not lines or not re.fullmatch(r"\d{1,6}", lines[0]):
        return None
    return int(lines[0])


def _positioned_block_direction(amount_raw: str, summary: str) -> str:
    if amount_raw.startswith("-"):
        return "expense"
    if amount_raw.startswith("+"):
        return "income"
    if any(marker in summary for marker in _POSITIONED_BLOCK_INCOME_MARKERS):
        return "income"
    if any(marker in summary for marker in _POSITIONED_BLOCK_EXPENSE_MARKERS):
        return "expense"
    return ""


def _positioned_block_counter_account(text: str) -> str:
    money_spans = [match.span() for match in _MONEY_ANY_RE.finditer(text)]
    candidates = []
    for match in _POSITIONED_BLOCK_ACCOUNT_RE.finditer(text):
        value = match.group(0)
        if _DATE_ANY_RE.fullmatch(value):
            continue
        if any(start <= match.start() and match.end() <= end for start, end in money_spans):
            continue
        candidates.append(match)
    if not candidates:
        return ""

    # A source account joined to a party name is stronger than an earlier
    # transaction/reference number in the same positioned block.
    for match in candidates:
        line_suffix = text[match.end() :].splitlines()[0] if text[match.end() :] else ""
        if line_suffix.lstrip().startswith("/"):
            return match.group(0)

    balance_end = money_spans[1][1] if len(money_spans) >= 2 else 0
    after_balance = [match for match in candidates if match.start() >= balance_end]
    if after_balance:
        return after_balance[0].group(0)
    return candidates[0].group(0)


def _positioned_block_counterparty(text: str, counter_account: str) -> str:
    if not counter_account:
        return ""
    suffix = text.split(counter_account, 1)[-1]
    suffix_lines = suffix.splitlines()
    candidate = suffix_lines[0].strip().lstrip("/ ") if suffix_lines else ""
    return re.sub(r"\s+", "", candidate)


def _sort_positioned_block_records(records: list[dict[str, Any]]) -> None:
    centers = [(_x_center(record["atom"]), _y_center(record["atom"])) for record in records]
    x_span = max(x for x, _ in centers) - min(x for x, _ in centers)
    y_span = max(y for _, y in centers) - min(y for _, y in centers)
    axis = 0 if x_span > y_span else 1
    records.sort(
        key=lambda record: (
            _x_center(record["atom"]) if axis == 0 else _y_center(record["atom"]),
            int(record.get("sequence_no") or 0),
        )
    )


def _infer_positioned_block_directions(
    records: list[dict[str, Any]],
    *,
    preceding_record: dict[str, Any] | None = None,
) -> None:
    """Use an adjacent, sequence-continuous balance to infer an unsigned direction."""
    for index, record in enumerate(records):
        previous = records[index - 1] if index else preceding_record
        if previous is None:
            continue
        previous_sequence = previous.get("sequence_no")
        current_sequence = record.get("sequence_no")
        if previous_sequence is not None or current_sequence is not None:
            if not _is_sequence_continuation(previous, record):
                continue
        elif index == 0:
            # Page-local geometry can prove adjacency, but two page-edge rows
            # without a sequence cannot safely bridge a missing transaction.
            continue
        try:
            delta = round(float(record["balance"]) - float(previous["balance"]), 2)
            amount = round(float(record["amount"]), 2)
        except (TypeError, ValueError):
            continue
        if abs(delta - amount) <= 0.05:
            record["direction"] = "income"
        elif abs(delta + amount) <= 0.05:
            record["direction"] = "expense"


def _is_sequence_continuation(previous: dict[str, Any], record: dict[str, Any]) -> bool:
    """Return whether two page-boundary records prove an adjacent ledger row."""
    try:
        return abs(int(record["sequence_no"]) - int(previous["sequence_no"])) == 1
    except (KeyError, TypeError, ValueError):
        return False


def _positioned_block_row_source(
    page_id: str,
    record: dict[str, Any],
    row: list[str],
) -> dict[str, Any]:
    atom = record["atom"]
    try:
        source_page = int(atom.get("source_page_number") or atom.get("source_page") or page_id.rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        source_page = 0
    evidence_ids = [str(value) for value in [atom.get("id"), *(atom.get("evidence_ids") or [])] if str(value or "")]
    source = {
        "source": "positioned_record_block",
        "page_id": page_id,
        "row_values": list(row),
    }
    if isinstance(record.get("source_raw"), dict):
        source["source_raw"] = dict(record["source_raw"])
    if source_page > 0:
        source["source_page"] = source_page
        source["page_range"] = [source_page, source_page]
    bbox = atom.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        source["bbox"] = [float(value) for value in bbox[:4]]
    if evidence_ids:
        source["evidence_ids"] = list(dict.fromkeys(evidence_ids))
    return source


def recover_evidence_atom_bank_tables(parse_result: Any) -> list[list[list[str]]]:
    """Return one canonical table when issuer headers and column geometry agree."""
    atoms_by_page = _normalize_page_orientations(parse_result, _atoms_by_page(parse_result))
    if not atoms_by_page:
        _store_recovery_cache(parse_result, [], [], 0)
        return []
    all_atoms = [atom for atoms in atoms_by_page.values() for atom in atoms]
    geometry_fallback, geometry_sources, geometry_expected = _recover_geometry_bank_tables(
        parse_result,
        atoms_by_page,
    )

    header_names = (
        "序号",
        "交易日期",
        "交易流水号",
        "对方账号",
        "对方户名",
        "对方行号",
        "对方行名",
        "交易渠道",
        "用途",
        "摘要",
    )
    headers = {name: _first_exact(all_atoms, name) for name in header_names}
    composite_header = next(
        (atom for atom in all_atoms if all(marker in str(atom.get("text") or "") for marker in _COMPOSITE_MARKERS)),
        None,
    )
    if composite_header is None or any(atom is None for atom in headers.values()):
        _store_recovery_cache(parse_result, geometry_fallback, geometry_sources, geometry_expected)
        return geometry_fallback

    composite_left = float(composite_header["bbox"][0])
    composite_right = float(composite_header["bbox"][2])
    endpoints = _money_column_endpoints(all_atoms, composite_left, composite_right)
    if len(endpoints) != 3:
        _store_recovery_cache(parse_result, geometry_fallback, geometry_sources, geometry_expected)
        return geometry_fallback
    expense_end, income_end, balance_end = endpoints

    anchors = {name: float(atom["bbox"][0]) for name, atom in headers.items() if atom is not None}
    if [anchors[name] for name in header_names] != sorted(anchors[name] for name in header_names):
        _store_recovery_cache(parse_result, geometry_fallback, geometry_sources, geometry_expected)
        return geometry_fallback
    sequence_right = (anchors["序号"] + anchors["交易日期"]) / 2
    date_x = anchors["交易日期"]
    reference_left = (anchors["交易日期"] + anchors["交易流水号"]) / 2
    reference_right = (anchors["交易流水号"] + composite_left) / 2
    account_left = (composite_right + anchors["对方账号"]) / 2
    text_columns = ("对方账号", "对方户名", "对方行号", "对方行名", "交易渠道", "用途", "摘要")
    text_bounds = [
        account_left,
        *((anchors[left] + anchors[right]) / 2 for left, right in zip(text_columns, text_columns[1:])),
        float("inf"),
    ]

    rows: list[list[str]] = []
    row_sources: list[dict[str, Any]] = []
    expected_rows = 0
    for page_id in sorted(atoms_by_page):
        atoms = atoms_by_page[page_id]
        dates = sorted(
            (
                (_y_center(atom), str(atom.get("text") or "").strip())
                for atom in atoms
                if abs(float(atom["bbox"][0]) - date_x) <= 12.0
                and _DATE_ANY_RE.search(str(atom.get("text") or "").strip())
            ),
            key=lambda item: item[0],
        )
        expected_rows += len(dates)
        for index, (row_y, date) in enumerate(dates):
            if index + 1 < len(dates):
                row_end = dates[index + 1][0]
            else:
                footer_starts = [
                    float(atom["bbox"][1])
                    for atom in atoms
                    if float(atom["bbox"][1]) > row_y and _is_geometry_footer_text(str(atom.get("text") or ""))
                ]
                row_end = min(footer_starts, default=float("inf"))
            row_atoms = [atom for atom in atoms if row_y - 0.5 <= _y_center(atom) < row_end - 0.5]
            money = [
                atom
                for atom in row_atoms
                if composite_left - 2.0 <= float(atom["bbox"][0])
                and float(atom["bbox"][2]) <= composite_right + 3.0
                and _MONEY_RE.fullmatch(str(atom.get("text") or "").strip())
            ]
            expense = _money_at_endpoint(money, expense_end)
            income = _money_at_endpoint(money, income_end)
            balance = _money_at_endpoint(money, balance_end)
            if not balance or bool(expense) == bool(income):
                continue
            row = [
                _column_text(row_atoms, float("-inf"), sequence_right),
                date,
                _column_text(row_atoms, reference_left, reference_right),
                expense,
                income,
                balance,
                *[
                    _column_text(row_atoms, text_bounds[column_index], text_bounds[column_index + 1])
                    for column_index in range(len(text_columns))
                ],
            ]
            rows.append(row)
            row_sources.append(_row_source(page_id, row_atoms, row))
    if rows:
        tables = [[_OUTPUT_HEADER, *rows]]
        _store_recovery_cache(parse_result, tables, row_sources, expected_rows)
        return tables
    _store_recovery_cache(parse_result, geometry_fallback, geometry_sources, geometry_expected)
    return geometry_fallback


def recovered_evidence_atom_row_sources(parse_result: Any) -> list[dict[str, Any]]:
    """Return row provenance aligned with recovered evidence-atom table rows."""
    cache = _recovery_cache(parse_result)
    sources = cache.get("row_sources") if cache else None
    return deepcopy(sources) if isinstance(sources, list) else []


def recovered_evidence_atom_expected_row_count(parse_result: Any) -> int:
    """Return the independent positioned-date candidate count from recovery."""
    cache = _recovery_cache(parse_result)
    return int(cache.get("expected_row_count") or 0) if cache else 0


def _recover_geometry_bank_tables(
    parse_result: Any,
    atoms_by_page: dict[str, list[dict[str, Any]]],
) -> tuple[list[list[list[str]]], list[dict[str, Any]], int]:
    """Rebuild borderless bank grids from header and row geometry."""
    tables: list[list[list[str]]] = []
    row_sources: list[dict[str, Any]] = []
    expected_rows = 0
    for page_id in sorted(atoms_by_page):
        atoms = atoms_by_page[page_id]
        header = _geometry_header(atoms)
        if header is None:
            continue
        header_atoms, col_map = header
        header_cells = [str(atom.get("text") or "").strip() for atom in header_atoms]
        centers = [_x_center(atom) for atom in header_atoms]
        bounds = [float("-inf"), *((left + right) / 2 for left, right in zip(centers, centers[1:])), float("inf")]
        horizontal_rules = _page_horizontal_rules(
            parse_result,
            page_id,
            atoms,
            header_atoms,
        )
        date_idx = col_map.get("date", col_map.get("timestamp"))
        if date_idx is None:
            continue
        date_left, date_right = bounds[date_idx], bounds[date_idx + 1]
        header_bottom = max(float(atom["bbox"][3]) for atom in header_atoms)
        page_footer_y = min(
            (
                _y_center(atom)
                for atom in atoms
                if float(atom["bbox"][1]) > header_bottom and _is_geometry_footer_text(str(atom.get("text") or ""))
            ),
            default=float("inf"),
        )
        date_anchors = sorted(
            (
                atom
                for atom in atoms
                if float(atom["bbox"][1]) > header_bottom
                and _y_center(atom) < page_footer_y
                and date_left <= _x_center(atom) < date_right
                and _DATE_ANY_RE.search(str(atom.get("text") or "").strip())
            ),
            key=lambda atom: float(atom["bbox"][1]),
        )
        expected_rows += len(date_anchors)
        rows: list[list[str]] = []
        row_atom_groups: list[list[dict[str, Any]]] = []
        for idx, anchor in enumerate(date_anchors):
            anchor_y = _y_center(anchor)
            next_y = _y_center(date_anchors[idx + 1]) if idx + 1 < len(date_anchors) else float("inf")
            footer_limit: float | None = None
            if next_y == float("inf"):
                footer_starts = [
                    _y_center(atom)
                    for atom in atoms
                    if float(atom["bbox"][1]) > anchor_y and _is_geometry_footer_text(str(atom.get("text") or ""))
                ]
                footer_limit = min(footer_starts, default=float("inf"))
            previous_rule = max((rule for rule in horizontal_rules if rule < anchor_y), default=None)
            next_rule = min((rule for rule in horizontal_rules if rule > anchor_y), default=None)
            if previous_rule is not None and next_rule is not None:
                # Native ledger rules are the strongest boundary evidence:
                # wrapped cells may begin above the date baseline or end well
                # below it, but cannot cross a full-width separator.
                row_top = previous_rule + 0.5
                row_bottom = next_rule - 0.5
            else:
                # Without vector rules, assign text to the closest preceding
                # date anchor and stop at the next date/footer.
                row_top = header_bottom if idx == 0 else anchor_y - 0.5
                if next_y != float("inf"):
                    row_bottom = next_y - 0.5
                else:
                    row_bottom = footer_limit if footer_limit is not None else float("inf")
            row_atoms = [
                atom
                for atom in atoms
                if row_top < _y_center(atom) < row_bottom and float(atom["bbox"][1]) > header_bottom
            ]
            row: list[str] = []
            for col_idx in range(len(header_cells)):
                selected = sorted(
                    (atom for atom in row_atoms if bounds[col_idx] <= _x_center(atom) < bounds[col_idx + 1]),
                    key=lambda atom: (float(atom["bbox"][1]), float(atom["bbox"][0])),
                )
                row.append("".join(str(atom.get("text") or "").strip() for atom in selected))
            date_match = _DATE_ANY_RE.search(row[date_idx])
            if date_match:
                sequence_idx = col_map.get("sequence_no")
                prefix = row[date_idx][: date_match.start()]
                if sequence_idx is not None and sequence_idx < len(row) and not row[sequence_idx] and prefix.isdigit():
                    row[sequence_idx] = prefix
                row[date_idx] = row[date_idx][date_match.start() :]
            if _geometry_row_is_transaction(row):
                rows.append(row)
                row_atom_groups.append(row_atoms)
        if rows:
            _repair_geometry_rows(rows, col_map)
            row_sources.extend(_row_source(page_id, row_atoms, row) for row_atoms, row in zip(row_atom_groups, rows))
            tables.append([header_cells, *rows])
    return tables, row_sources, expected_rows


def _page_horizontal_rules(
    parse_result: Any,
    page_id: str,
    text_atoms: list[dict[str, Any]],
    header_atoms: list[dict[str, Any]],
) -> list[float]:
    """Return full-width native horizontal rules aligned with a ledger header."""
    from docmirror.plugins._runtime.evidence_access import evidence_payload

    payload = evidence_payload(parse_result)
    vector_atoms = [
        atom
        for atom in payload.get("vector_atoms") or []
        if isinstance(atom, dict)
        and str(atom.get("page_id") or "") == page_id
        and isinstance(atom.get("bbox"), list)
        and len(atom["bbox"]) >= 4
    ]
    if not vector_atoms:
        return []

    rotation = int(text_atoms[0].get("_geometry_rotation") or 0) if text_atoms else 0
    page_number = int(page_id.rsplit(":", 1)[-1])
    page = next(
        (
            item
            for item in getattr(parse_result, "pages", []) or []
            if int(getattr(item, "page_number", 0) or 0) == page_number
        ),
        None,
    )
    width = float(getattr(page, "width", 0) or 0)
    height = float(getattr(page, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return []

    rotated = [_rotated_atom(atom, rotation, width, height) for atom in vector_atoms]
    horizontal = [
        atom
        for atom in rotated
        if abs(float(atom["bbox"][3]) - float(atom["bbox"][1])) <= 1.0
        and abs(float(atom["bbox"][2]) - float(atom["bbox"][0])) >= 8.0
    ]
    if not horizontal:
        return []

    header_left = min(float(atom["bbox"][0]) for atom in header_atoms)
    header_right = max(float(atom["bbox"][2]) for atom in header_atoms)
    required_span = max((header_right - header_left) * 0.75, 1.0)
    groups: list[list[dict[str, Any]]] = []
    for atom in sorted(horizontal, key=lambda item: float(item["bbox"][1])):
        y = float(atom["bbox"][1])
        if not groups:
            groups.append([atom])
            continue
        baseline = sum(float(item["bbox"][1]) for item in groups[-1]) / len(groups[-1])
        if abs(y - baseline) <= 1.0:
            groups[-1].append(atom)
        else:
            groups.append([atom])

    rules: list[float] = []
    for group in groups:
        left = min(float(atom["bbox"][0]) for atom in group)
        right = max(float(atom["bbox"][2]) for atom in group)
        if right - left < required_span:
            continue
        rules.append(sum(float(atom["bbox"][1]) for atom in group) / len(group))
    return rules


def _normalize_page_orientations(
    parse_result: Any,
    atoms_by_page: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Choose the page rotation that yields the strongest generic ledger geometry."""
    dimensions = {
        f"page:{int(getattr(page, 'page_number', 1) or 1):04d}": (
            float(getattr(page, "width", 0) or 0),
            float(getattr(page, "height", 0) or 0),
        )
        for page in (getattr(parse_result, "pages", []) or [])
    }
    normalized: dict[str, list[dict[str, Any]]] = {}
    for page_id, atoms in atoms_by_page.items():
        fallback_width = max((float(atom["bbox"][2]) for atom in atoms), default=0.0)
        fallback_height = max((float(atom["bbox"][3]) for atom in atoms), default=0.0)
        width, height = dimensions.get(page_id, (0.0, 0.0))
        width = width or fallback_width
        height = height or fallback_height
        candidates = [
            _split_stacked_atoms(
                _expand_composite_header_atoms([_rotated_atom(atom, rotation, width, height) for atom in atoms])
            )
            for rotation in (0, 90, 180, 270)
        ]
        normalized[page_id] = max(
            enumerate(candidates),
            key=lambda item: (_orientation_score(item[1]), -item[0]),
        )[1]
    return normalized


def _rotated_atom(
    atom: dict[str, Any],
    rotation: int,
    width: float,
    height: float,
) -> dict[str, Any]:
    cloned = dict(atom)
    source_bbox = [float(value) for value in atom["bbox"][:4]]
    x0, y0, x1, y1 = source_bbox
    if rotation == 90:
        bbox = [y0, width - x1, y1, width - x0]
    elif rotation == 180:
        bbox = [width - x1, height - y1, width - x0, height - y0]
    elif rotation == 270:
        bbox = [height - y1, x0, height - y0, x1]
    else:
        bbox = source_bbox
    cloned["bbox"] = bbox
    cloned["_source_bbox"] = source_bbox
    cloned["_geometry_rotation"] = rotation
    return cloned


def _expand_composite_header_atoms(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split common OCR-merged labels into virtual geometry-only header cells."""
    patterns = {
        "序号交易日期": ("序号", "交易日期"),
        "序号交易日期交易类型": ("序号", "交易日期", "交易类型"),
        "收入/支出交易金额": ("收入/支出", "交易金额"),
        "收入支出交易金额": ("收入/支出", "交易金额"),
        "交易类型收入/支出交易金额": ("交易类型", "收入/支出", "交易金额"),
        "序号交易日期交易类型收入/支出交易金额": (
            "序号",
            "交易日期",
            "交易类型",
            "收入/支出",
            "交易金额",
        ),
    }
    expanded: list[dict[str, Any]] = []
    for atom in atoms:
        normalized_text = re.sub(r"\s+", "", str(atom.get("text") or ""))
        parts = patterns.get(normalized_text)
        if not parts:
            expanded.append(atom)
            continue
        x0, y0, x1, y1 = [float(value) for value in atom["bbox"][:4]]
        total_weight = sum(len(part) for part in parts)
        cursor = x0
        for index, part in enumerate(parts):
            right = x1 if index == len(parts) - 1 else cursor + (x1 - x0) * len(part) / total_weight
            virtual = dict(atom)
            virtual["id"] = f"{atom.get('id') or 'atom'}:split:{index}"
            virtual["text"] = part
            virtual["bbox"] = [cursor, y0, right, y1]
            expanded.append(virtual)
            cursor = right
    return expanded


def _split_stacked_atoms(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split OCR blocks that vertically merged repeated row values."""
    heights = [
        float(atom["bbox"][3]) - float(atom["bbox"][1])
        for atom in atoms
        if float(atom["bbox"][3]) > float(atom["bbox"][1])
    ]
    typical_height = median(heights) if heights else 0.0
    split_atoms: list[dict[str, Any]] = []
    for atom in atoms:
        text = str(atom.get("text") or "").strip()
        height = float(atom["bbox"][3]) - float(atom["bbox"][1])
        if typical_height <= 0 or height < typical_height * 1.55:
            split_atoms.append(atom)
            continue
        parts = _stacked_text_parts(text)
        if len(parts) < 2:
            split_atoms.append(atom)
            continue
        x0, y0, x1, y1 = [float(value) for value in atom["bbox"][:4]]
        step = (y1 - y0) / len(parts)
        for index, part in enumerate(parts):
            virtual = dict(atom)
            virtual["id"] = f"{atom.get('id') or 'atom'}:stack:{index}"
            virtual["text"] = part
            virtual["bbox"] = [x0, y0 + step * index, x1, y0 + step * (index + 1)]
            split_atoms.append(virtual)
    return split_atoms


def _stacked_text_parts(text: str) -> list[str]:
    date_matches = list(_DATE_ANY_RE.finditer(text))
    if len(date_matches) >= 2:
        return [
            text[match.start() : date_matches[index + 1].start() if index + 1 < len(date_matches) else len(text)]
            for index, match in enumerate(date_matches)
        ]
    money_matches = list(_MONEY_ANY_RE.finditer(text))
    if len(money_matches) >= 2:
        remainder = _MONEY_ANY_RE.sub("", text).strip(" ,，")
        if not remainder:
            return [match.group(0) for match in money_matches]
    directions = re.findall(r"收入|支出|收人|支山|攴出", text)
    if len(directions) >= 2 and "".join(directions) == text:
        return directions
    return []


def _orientation_score(atoms: list[dict[str, Any]]) -> float:
    labelled = [(atom, _header_roles(str(atom.get("text") or ""))) for atom in atoms]
    labelled = [(atom, roles) for atom, roles in labelled if roles]
    best = -1.0
    for anchor, _anchor_roles in labelled:
        baseline = _y_center(anchor)
        group = [(atom, roles) for atom, roles in labelled if abs(_y_center(atom) - baseline) <= 8.0]
        roles = set().union(*(roles for _atom, roles in group))
        if not {"date", "amount", "balance"}.issubset(roles):
            continue
        xs = [_x_center(atom) for atom, _roles in group]
        spread = max(xs, default=0.0) - min(xs, default=0.0)
        score = len(roles) * 10.0 + min(spread / 20.0, 20.0)
        best = max(best, score)
    return best


def _header_roles(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text)
    roles: set[str] = set()
    if any(marker in normalized for marker in ("交易日期", "记账日期", "交易时间", "日期")):
        roles.add("date")
    if any(marker in normalized for marker in ("交易金额", "发生额", "支出金额", "收入金额")):
        roles.add("amount")
    if "余额" in normalized:
        roles.add("balance")
    if any(marker in normalized for marker in ("收入/支出", "收/支", "借贷", "借/贷")):
        roles.add("direction")
    if "序号" in normalized:
        roles.add("sequence")
    if any(marker in normalized for marker in ("对方账号", "对方账户", "对方户名", "交易对手")):
        roles.add("counterparty")
    return roles


def _row_source(page_id: str, atoms: list[dict[str, Any]], row: list[str]) -> dict[str, Any]:
    source_boxes = [
        atom.get("_source_bbox") or atom.get("bbox")
        for atom in atoms
        if isinstance(atom.get("_source_bbox") or atom.get("bbox"), list)
    ]
    bbox = _union_bbox(source_boxes)
    evidence_ids = list(
        dict.fromkeys(
            str(evidence_id)
            for atom in atoms
            for evidence_id in [atom.get("id"), *(atom.get("evidence_ids") or [])]
            if str(evidence_id or "")
        )
    )
    try:
        source_page = int(page_id.rsplit(":", 1)[-1])
    except ValueError:
        source_page = 0
    return {
        "source": "canonical_evidence_table",
        "page_id": page_id,
        **({"source_page": source_page} if source_page > 0 else {}),
        **({"bbox": bbox} if bbox else {}),
        **({"evidence_ids": evidence_ids} if evidence_ids else {}),
        "row_values": list(row),
    }


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    valid = [box for box in boxes if len(box) >= 4]
    if not valid:
        return []
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def _domain_specific(parse_result: Any) -> dict[str, Any] | None:
    entities = getattr(parse_result, "entities", None)
    domain = getattr(entities, "domain_specific", None) if entities is not None else None
    return domain if isinstance(domain, dict) else None


def _store_recovery_cache(
    parse_result: Any,
    tables: list[list[list[str]]],
    row_sources: list[dict[str, Any]],
    expected_row_count: int,
) -> None:
    domain = _domain_specific(parse_result)
    if domain is None:
        return
    domain[_RECOVERY_CACHE_KEY] = {
        "status": "ready",
        "table_count": len(tables),
        "row_count": sum(max(len(table) - 1, 0) for table in tables),
        "expected_row_count": max(int(expected_row_count or 0), 0),
        "row_sources": deepcopy(row_sources),
    }


def _recovery_cache(parse_result: Any) -> dict[str, Any]:
    domain = _domain_specific(parse_result)
    cache = domain.get(_RECOVERY_CACHE_KEY) if domain else None
    return cache if isinstance(cache, dict) and cache.get("status") == "ready" else {}


def _geometry_header(
    atoms: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    from docmirror.plugins._base.column_registry import ColumnMatcher
    from docmirror.plugins.bank_statement.community_plugin import BANK_COLUMN_REGISTRY
    from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell

    matcher = ColumnMatcher(BANK_COLUMN_REGISTRY)
    best: tuple[list[dict[str, Any]], dict[str, int]] | None = None
    for group in _baseline_groups(atoms):
        ordered = sorted(group, key=lambda atom: float(atom["bbox"][0]))
        cells = [normalize_header_cell(str(atom.get("text") or "")) for atom in ordered]
        joined = "".join(cells)
        if not any(marker in joined for marker in ("交易日期", "记账日期", "交易时间", "日期")):
            continue
        if not any(marker in joined for marker in ("交易金额", "发生额", "支出金额", "收入金额")):
            continue
        if "余额" not in joined:
            continue
        col_map = matcher.match(cells)
        fields = set(col_map)
        valid = {"amount", "balance"}.issubset(fields) and bool(fields.intersection({"date", "timestamp"}))
        if valid and (best is None or len(col_map) > len(best[1])):
            expanded = _expand_staggered_header(atoms, ordered)
            expanded_cells = [normalize_header_cell(str(atom.get("text") or "")) for atom in expanded]
            expanded_map = matcher.match(expanded_cells)
            expanded_fields = set(expanded_map)
            expanded_valid = {"amount", "balance"}.issubset(expanded_fields) and bool(
                expanded_fields.intersection({"date", "timestamp"})
            )
            best = (
                (expanded, expanded_map) if expanded_valid and len(expanded_map) >= len(col_map) else (ordered, col_map)
            )
    return best


def _expand_staggered_header(
    atoms: list[dict[str, Any]],
    header_atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge lower labels from multi-tier headers without pulling in English rows."""
    baseline = sum(float(atom["bbox"][1]) for atom in header_atoms) / len(header_atoms)
    band = [atom for atom in atoms if baseline - 4.0 <= float(atom["bbox"][1]) <= baseline + 6.0]
    return sorted(band, key=lambda atom: float(atom["bbox"][0]))


def _baseline_groups(atoms: list[dict[str, Any]], tolerance: float = 3.0) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for atom in sorted(atoms, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]))):
        y = float(atom["bbox"][1])
        if not groups:
            groups.append([atom])
            continue
        baseline = sum(float(item["bbox"][1]) for item in groups[-1]) / len(groups[-1])
        if abs(y - baseline) <= tolerance:
            groups[-1].append(atom)
        else:
            groups.append([atom])
    return groups


def _geometry_row_is_transaction(row: list[str]) -> bool:
    from docmirror.plugins.bank_statement.row_extract import row_has_transaction_data
    from docmirror.plugins.bank_statement.wide_table_recovery import is_footer_or_total_row

    return not is_footer_or_total_row(row) and row_has_transaction_data(row)


def _is_geometry_footer_text(text: str) -> bool:
    return any(marker in text for marker in _GEOMETRY_FOOTER_MARKERS) or bool(
        re.search(r"第\s*\d+\s*页\s*(?:[/／-]\s*)?共\s*\d+\s*页", text)
    )


def _repair_geometry_rows(rows: list[list[str]], col_map: dict[str, int]) -> None:
    sequence_idx = col_map.get("sequence_no")
    direction_idx = col_map.get("direction")
    amount_idx = col_map.get("amount")
    balance_idx = col_map.get("balance")
    summary_indexes = [
        index for key in ("summary", "purpose", "counter_party") if (index := col_map.get(key)) is not None
    ]

    for row in rows:
        for index in (amount_idx, balance_idx):
            if index is not None and index < len(row):
                row[index] = _repair_malformed_money(row[index])
    if sequence_idx is not None:
        _repair_sequence_values(rows, sequence_idx)
    if direction_idx is None or amount_idx is None or balance_idx is None:
        return

    previous_balance: float | None = None
    for row in rows:
        if max(direction_idx, amount_idx, balance_idx) >= len(row):
            continue
        direction_text = str(row[direction_idx] or "")
        if any(marker in direction_text for marker in ("收入", "收人", "转入", "贷")):
            direction = "收入"
            inferred_direction = False
        elif any(marker in direction_text for marker in ("支出", "支山", "攴出", "转出", "借")):
            direction = "支出"
            inferred_direction = False
        else:
            inferred_direction = True
            context = "".join(row[index] for index in summary_indexes if index < len(row))
            if any(marker in context for marker in ("转入", "收入")):
                direction = "收入"
            elif any(marker in context for marker in ("转出", "支出")):
                direction = "支出"
            else:
                direction = ""
        amount = _money_float(row[amount_idx])
        balance = _money_float(row[balance_idx])
        if not direction and previous_balance is not None and amount is not None and balance is not None:
            income_error = abs(previous_balance + amount - balance)
            expense_error = abs(previous_balance - amount - balance)
            if min(income_error, expense_error) <= 0.05:
                direction = "收入" if income_error < expense_error else "支出"
        if direction and inferred_direction:
            row[direction_idx] = direction
        if balance is not None:
            previous_balance = balance


def _repair_sequence_values(rows: list[list[str]], sequence_idx: int) -> None:
    values: list[int | None] = []
    for row in rows:
        text = str(row[sequence_idx] or "").strip() if sequence_idx < len(row) else ""
        values.append(int(text) if re.fullmatch(r"\d{1,6}", text) else None)
    for index, value in enumerate(values):
        if value is not None:
            continue
        previous = next((position for position in range(index - 1, -1, -1) if values[position] is not None), None)
        following = next((position for position in range(index + 1, len(values)) if values[position] is not None), None)
        inferred: int | None = None
        if previous is not None and following is not None:
            if values[following] - values[previous] == following - previous:
                inferred = int(values[previous] or 0) + index - previous
        elif following is not None:
            inferred = int(values[following] or 0) - (following - index)
        elif previous is not None:
            inferred = int(values[previous] or 0) + index - previous
        if inferred is not None and inferred > 0:
            values[index] = inferred
            rows[index][sequence_idx] = str(inferred)


def _repair_malformed_money(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    if text.count(".") <= 1:
        return text
    integer, decimal = text.rsplit(".", 1)
    if len(decimal) == 2 and re.fullmatch(r"\d[\d,.]*", integer):
        return f"{integer.replace('.', ',')}.{decimal}"
    return text


def _money_float(value: str) -> float | None:
    try:
        return float(str(value or "").replace(",", ""))
    except ValueError:
        return None


def _x_center(atom: dict[str, Any]) -> float:
    return (float(atom["bbox"][0]) + float(atom["bbox"][2])) / 2


def _y_center(atom: dict[str, Any]) -> float:
    return (float(atom["bbox"][1]) + float(atom["bbox"][3])) / 2


def _atoms_by_page(parse_result: Any) -> dict[str, list[dict[str, Any]]]:
    from docmirror.plugins._runtime.evidence_access import text_atoms

    atoms = text_atoms(parse_result)
    if not atoms:
        atoms = _page_text_atoms(parse_result)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        page_id = str(atom.get("page_id") or "")
        bbox = atom.get("bbox")
        text = str(atom.get("text") or "").strip()
        if page_id and text and isinstance(bbox, list) and len(bbox) >= 4:
            grouped[page_id].append(atom)
    return dict(grouped)


def _positioned_atoms_by_page(parse_result: Any) -> dict[str, list[dict[str, Any]]]:
    """Prefer page text blocks when recovering one-block-per-record layouts.

    The evidence plane often tokenizes a native PDF into individual visual text
    atoms.  That is ideal for grid geometry, but it loses the record boundary
    retained by ``PageContent.texts``.  This recovery path specifically needs
    that boundary, so it prefers the page blocks while preserving the evidence
    atom path when page blocks are unavailable.
    """
    page_atoms = _page_text_atoms(parse_result)
    if page_atoms:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for atom in page_atoms:
            grouped[str(atom["page_id"])].append(atom)
        for page_id, atoms in _atoms_by_page(parse_result).items():
            if page_id not in grouped:
                grouped[page_id].extend(atoms)
        return dict(grouped)
    return _atoms_by_page(parse_result)


def _page_text_atoms(parse_result: Any) -> list[dict[str, Any]]:
    """Adapt positioned OCR text blocks when the sealed evidence plane has no atoms.

    Bank projection runs before every parser configuration has promoted page OCR
    blocks into ``evidence_plane.text_atoms``. Page text blocks are the same
    canonical extraction facts and retain their physical-page bboxes.
    """
    atoms: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", []) or []:
        logical_page_number = int(getattr(page, "page_number", 1) or 1)
        source_page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 1) or 1)
        page_id = f"page:{logical_page_number:04d}"
        for index, block in enumerate(getattr(page, "texts", []) or []):
            text = str(getattr(block, "content", "") or "").strip()
            bbox = getattr(block, "bbox", None)
            if not text or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            evidence_ids = [
                str(evidence_id)
                for evidence_id in (getattr(block, "evidence_ids", None) or [])
                if str(evidence_id or "")
            ]
            atoms.append(
                {
                    "id": evidence_ids[0] if evidence_ids else f"{page_id}:text:{index}",
                    "page_id": page_id,
                    "source_page_number": source_page_number,
                    "text": text,
                    "bbox": [float(value) for value in bbox[:4]],
                    "evidence_ids": evidence_ids,
                }
            )
    return atoms


def _first_exact(atoms: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    return next((atom for atom in atoms if str(atom.get("text") or "").strip() == text), None)


def _money_column_endpoints(atoms: list[dict[str, Any]], left: float, right: float) -> list[float]:
    rounded_ends = [
        round(float(atom["bbox"][2]), 1)
        for atom in atoms
        if left - 2.0 <= float(atom["bbox"][0])
        and float(atom["bbox"][2]) <= right + 3.0
        and _MONEY_RE.fullmatch(str(atom.get("text") or "").strip())
    ]
    common = [value for value, count in Counter(rounded_ends).most_common() if count >= 2]
    return sorted(common) if len(common) == 3 else []


def _money_at_endpoint(atoms: list[dict[str, Any]], endpoint: float) -> str:
    atom = next((atom for atom in atoms if abs(float(atom["bbox"][2]) - endpoint) <= 1.0), None)
    return str(atom.get("text") or "").strip() if atom else ""


def _column_text(atoms: list[dict[str, Any]], left: float, right: float) -> str:
    selected = sorted(
        (atom for atom in atoms if left <= float(atom["bbox"][0]) < right),
        key=lambda atom: (float(atom["bbox"][1]), float(atom["bbox"][0])),
    )
    return "".join(str(atom.get("text") or "").strip() for atom in selected)


__all__ = [
    "PositionedBlockRecovery",
    "recover_evidence_atom_bank_tables",
    "recover_positioned_record_block_bank_tables",
    "recovered_evidence_atom_expected_row_count",
    "recovered_evidence_atom_row_sources",
]
