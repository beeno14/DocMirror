# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover high-confidence split debit/credit ledgers from canonical evidence atoms."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from copy import deepcopy
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


def recover_evidence_atom_bank_tables(parse_result: Any) -> list[list[list[str]]]:
    """Return one canonical table when issuer headers and column geometry agree."""
    atoms_by_page = _normalize_page_orientations(parse_result, _atoms_by_page(parse_result))
    if not atoms_by_page:
        _store_recovery_cache(parse_result, [], [], 0)
        return []
    all_atoms = [atom for atoms in atoms_by_page.values() for atom in atoms]
    geometry_fallback, geometry_sources, geometry_expected = _recover_geometry_bank_tables(atoms_by_page)

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
                    if float(atom["bbox"][1]) > row_y
                    and _is_geometry_footer_text(str(atom.get("text") or ""))
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
        date_idx = col_map.get("date", col_map.get("timestamp"))
        if date_idx is None:
            continue
        date_left, date_right = bounds[date_idx], bounds[date_idx + 1]
        header_bottom = max(float(atom["bbox"][3]) for atom in header_atoms)
        page_footer_y = min(
            (
                _y_center(atom)
                for atom in atoms
                if float(atom["bbox"][1]) > header_bottom
                and _is_geometry_footer_text(str(atom.get("text") or ""))
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
            previous_y = _y_center(date_anchors[idx - 1]) if idx > 0 else header_bottom
            next_y = _y_center(date_anchors[idx + 1]) if idx + 1 < len(date_anchors) else float("inf")
            footer_limit: float | None = None
            if next_y == float("inf"):
                footer_starts = [
                    _y_center(atom)
                    for atom in atoms
                    if float(atom["bbox"][1]) > anchor_y
                    and _is_geometry_footer_text(str(atom.get("text") or ""))
                ]
                footer_limit = min(footer_starts, default=float("inf"))
            row_top = (previous_y + anchor_y) / 2
            if next_y != float("inf"):
                row_bottom = (anchor_y + next_y) / 2
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
            row_sources.extend(
                _row_source(page_id, row_atoms, row)
                for row_atoms, row in zip(row_atom_groups, rows)
            )
            tables.append([header_cells, *rows])
    return tables, row_sources, expected_rows


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
            best = (expanded, expanded_map) if expanded_valid and len(expanded_map) >= len(col_map) else (ordered, col_map)
    return best


def _expand_staggered_header(
    atoms: list[dict[str, Any]],
    header_atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge lower labels from multi-tier headers without pulling in English rows."""
    baseline = sum(float(atom["bbox"][1]) for atom in header_atoms) / len(header_atoms)
    band = [
        atom
        for atom in atoms
        if baseline - 4.0 <= float(atom["bbox"][1]) <= baseline + 6.0
    ]
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
        re.search(r"第\s*\d+\s*页\s*共\s*\d+\s*页", text)
    )


def _repair_geometry_rows(rows: list[list[str]], col_map: dict[str, int]) -> None:
    sequence_idx = col_map.get("sequence_no")
    direction_idx = col_map.get("direction")
    amount_idx = col_map.get("amount")
    balance_idx = col_map.get("balance")
    summary_indexes = [
        index
        for key in ("summary", "purpose", "counter_party")
        if (index := col_map.get(key)) is not None
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


def _page_text_atoms(parse_result: Any) -> list[dict[str, Any]]:
    """Adapt positioned OCR text blocks when the sealed evidence plane has no atoms.

    Bank projection runs before every parser configuration has promoted page OCR
    blocks into ``evidence_plane.text_atoms``. Page text blocks are the same
    canonical extraction facts and retain their physical-page bboxes.
    """
    atoms: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", []) or []:
        page_number = int(getattr(page, "page_number", 1) or 1)
        page_id = f"page:{page_number:04d}"
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
    "recover_evidence_atom_bank_tables",
    "recovered_evidence_atom_expected_row_count",
    "recovered_evidence_atom_row_sources",
]
