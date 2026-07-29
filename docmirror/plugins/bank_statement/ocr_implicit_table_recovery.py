# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover OCR implicit ledger tables from canonical facts and evidence."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

from docmirror.plugins.bank_statement.header_resolve import normalize_header_cell

logger = logging.getLogger(__name__)

_STANDARD_HEADER = [
    "交易日期",
    "收/支",
    "交易金额",
    "余额",
    "摘要",
    "对方账号",
    "对方户名",
    "机构",
    "柜员",
    "备注",
    "_source_page",
]
_DATE_RE = re.compile(r"(?<!\d)(20\d{6}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})(?!\d)")
_DIRECTION_RE = re.compile(r"(收入|收人|支出|支山|支鼎|攴出)")
_AMOUNT_TOKEN_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)\.\s*\d{2}")
_ACCOUNT_RE = re.compile(r"(?<!\d)\d{7,24}(?!\d)")
_NOISE_RE = re.compile(r"第\s*\d+\s*页|业务用章|交易机构|产品说明|账号序号|起始日期|终止日期")
_CACHE_KEY = "_bank_ocr_implicit_recovery"
_TEXT_BLOCK_TYPES = {"paragraph", "list", "footer", "unknown", "text"}
_MIN_DISTRIBUTED_ROWS = 3
_MIN_DISTRIBUTED_VALID_RATIO = 0.6
_SWAP_ORIENTATION_PENALTY = 0.02
_REPAIR_META_PREFIX = "__docmirror_repair_meta__:"
_PAGE_MARKER_RE = re.compile(r"<!--\s*docmirror:page\b[^>]*-->")
_OPENING_BALANCE_MARKERS = (
    "上页余额",
    "期初余额",
    "起始余额",
    "opening balance",
)
_SUMMARY_KEYWORDS = (
    "网银转账",
    "跨行转账",
    "实时汇款",
    "短信通费",
    "归还本息",
    "账户费",
    "手续费",
    "活期结息",
    "发放贷款",
    "贷款受让",
    "证券转银行",
    "第三方支付",
    "还信用卡",
    "网络付款",
    "网络收款",
    "POS消费",
    "微信转账",
    "扫二维码",
    "美团支付",
    "转账",
    "结息",
    "付息",
)
_COUNTERPARTY_STOP_MARKERS = (
    *_SUMMARY_KEYWORDS,
    "普通汇兑",
    "业务功能描述",
    "贷款还息",
    "用途:",
    "用途：",
    "附言:",
    "附言：",
    "备注:",
    "备注：",
)
_CONCATENATED_SUMMARY_MARKERS = (
    "网银转账",
    "跨行转账",
    "实时汇款",
    "短信通费",
    "归还本息",
    "账户费",
    "活期结息",
    "发放贷款",
    "贷款受让",
)
_AMOUNT_HEADER_MARKERS = (
    "交易金额",
    "交易发生金额",
    "发生金额",
    "发生额",
)


def recover_ocr_implicit_ledger_tables(parse_result: Any, full_text: str = "") -> list[list[list[str]]]:
    """Build bank ledger tables from canonical tables and positioned text."""
    cached = _cached_tables(parse_result)
    if cached is not None:
        logger.debug("[BankOCRImplicitRecovery] cache hit rows=%d", _recovered_row_count(cached))
        return cached

    canonical = _canonical_payload(parse_result)
    tables = _extract_canonical_tables(canonical)
    recovered: list[list[list[str]]] = []
    recovered.extend(_extract_paragraph_ledger_tables(canonical))
    for table in tables:
        normalized = _normalize_implicit_table(table)
        if len(normalized) > 1:
            recovered.append(normalized)
    if recovered:
        _store_cache(parse_result, recovered, source="canonical_facts")
        logger.info("[BankOCRImplicitRecovery] recovered %d OCR implicit table(s)", len(recovered))
        return recovered
    recovered = _recover_from_text(full_text)
    _store_cache(parse_result, recovered, source="full_text" if recovered else "none")
    return recovered


def recovered_ocr_implicit_row_count(parse_result: Any) -> int:
    """Return cached OCR implicit recovery row count without triggering Mirror rebuild."""
    cached = _cached_tables(parse_result)
    return _recovered_row_count(cached or [])


def _domain_specific(parse_result: Any) -> dict[str, Any] | None:
    entities = getattr(parse_result, "entities", None)
    ds = getattr(entities, "domain_specific", None) if entities is not None else None
    return ds if isinstance(ds, dict) else None


def _cached_tables(parse_result: Any) -> list[list[list[str]]] | None:
    ds = _domain_specific(parse_result)
    if ds is None:
        return None
    cache = ds.get(_CACHE_KEY)
    if not isinstance(cache, dict) or cache.get("status") != "ready":
        return None
    tables = cache.get("tables")
    if not _is_table_list(tables):
        return None
    return deepcopy(tables)


def _store_cache(parse_result: Any, tables: list[list[list[str]]], *, source: str) -> None:
    ds = _domain_specific(parse_result)
    if ds is None:
        return
    ds[_CACHE_KEY] = {
        "status": "ready",
        "source": source,
        "table_count": len(tables),
        "row_count": _recovered_row_count(tables),
        "tables": deepcopy(tables),
    }


def _is_table_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(table, list) and all(isinstance(row, list) for row in table) for table in value
    )


def _recovered_row_count(tables: list[list[list[str]]]) -> int:
    return sum(max(len(table) - 1, 0) for table in tables)


def _canonical_payload(parse_result: Any) -> dict[str, Any]:
    if parse_result is None:
        return {}
    from docmirror.plugins._runtime.evidence_access import evidence_payload

    blocks: list[dict[str, Any]] = []
    local_lines_by_page = _local_structure_lines_by_page(parse_result)
    for page in getattr(parse_result, "pages", []) or []:
        page_no = int(getattr(page, "page_number", 1) or 1)
        page_id = f"page:{page_no:04d}"
        for table in getattr(page, "tables", []) or []:
            cells: list[dict[str, Any]] = []
            for col, text in enumerate(getattr(table, "headers", []) or []):
                cells.append({"row": 0, "col": col, "text": str(text)})
            for row_index, row in enumerate(getattr(table, "rows", []) or [], start=1):
                for col, cell in enumerate(getattr(row, "cells", []) or []):
                    cells.append({"row": row_index, "col": col, "text": str(getattr(cell, "text", ""))})
            if cells:
                blocks.append(
                    {
                        "type": "table",
                        "page_ids": [page_id],
                        "content": {"grid": {"cells": cells}},
                    }
                )
        local_lines = local_lines_by_page.get(page_no) or []
        if local_lines:
            for line in local_lines:
                blocks.append(
                    {
                        "type": "paragraph",
                        "page_ids": [page_id],
                        "bbox": line.get("bbox"),
                        "text": str(line.get("text") or line.get("content") or "").strip(),
                    }
                )
        else:
            for text in getattr(page, "texts", []) or []:
                content = str(getattr(text, "content", "") or "").strip()
                if content:
                    blocks.append(
                        {
                            "type": "paragraph",
                            "page_ids": [page_id],
                            "bbox": getattr(text, "bbox", None),
                            "text": content,
                        }
                    )
    return {"blocks": blocks, "evidence": evidence_payload(parse_result)}


def _local_structure_lines_by_page(parse_result: Any) -> dict[int, list[dict[str, Any]]]:
    """Return positioned OCR line evidence keyed by its actual source page."""
    domain_specific = _domain_specific(parse_result) or {}
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for bundle in domain_specific.get("_page_evidence_bundles") or []:
        if not isinstance(bundle, dict):
            continue
        local = bundle.get("local_structure_evidence")
        if not isinstance(local, dict):
            continue
        for line in local.get("lines") or []:
            if not isinstance(line, dict):
                continue
            text = str(line.get("text") or line.get("content") or "").strip()
            bbox = line.get("bbox")
            try:
                page_no = int(line.get("page") or local.get("page") or bundle.get("page") or 0)
            except (TypeError, ValueError):
                continue
            if page_no > 0 and text and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                by_page[page_no].append(line)
    return dict(by_page)


def _extract_canonical_tables(payload: dict[str, Any]) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    for block in payload.get("blocks") or []:
        if not isinstance(block, dict) or block.get("type") != "table":
            continue
        cells = ((block.get("content") or {}).get("grid") or {}).get("cells") or []
        if not cells:
            continue
        max_row = max((int(cell.get("row", 0) or 0) for cell in cells), default=-1)
        max_col = max((int(cell.get("col", 0) or 0) for cell in cells), default=-1)
        if max_row < 0 or max_col < 0:
            continue
        rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            row = int(cell.get("row", 0) or 0)
            col = int(cell.get("col", 0) or 0)
            if row <= max_row and col <= max_col:
                rows[row][col] = _clean_cell(cell.get("text"))
        if rows:
            tables.append(rows)
    return tables


def _normalize_implicit_table(table: list[list[str]]) -> list[list[str]]:
    if not table:
        return []
    header_idx = next((idx for idx, row in enumerate(table[:6]) if _is_implicit_ledger_header(row)), -1)
    if header_idx < 0:
        return []
    header = table[header_idx]
    mapping = _header_mapping(header)
    out = [_STANDARD_HEADER]
    for row in table[header_idx + 1 :]:
        normalized = _normalize_row(row, mapping)
        if normalized:
            out.append(normalized)
    return out if len(out) > 1 else []


def _extract_paragraph_ledger_tables(payload: dict[str, Any]) -> list[list[list[str]]]:
    """Recover ledger rows that vNext kept as paragraph/list/footer text blocks.

    Scanned first pages often have valid OCR tokens and bboxes, but fail table-region
    reconstruction because stamps, page marks, or mixed multi-line cells disrupt the
    grid.  This path treats the visible ledger header as a column-role anchor and then
    uses domain invariants to parse the following text blocks into rows.
    """
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in payload.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        page_ids = block.get("page_ids") or []
        if not page_ids:
            continue
        by_page[str(page_ids[0])].append(block)

    tables: list[list[list[str]]] = []
    for page_id, blocks in sorted(by_page.items()):
        page_no = _page_number_from_id(page_id)
        ordered = sorted(blocks, key=_block_sort_key)
        header_idx = next((idx for idx, block in enumerate(ordered) if _is_paragraph_ledger_header(block)), -1)
        if header_idx < 0:
            header_idx = _distributed_header_end(ordered)
        if header_idx < 0:
            continue
        counterparty_before_account = _counterparty_precedes_account(ordered[: header_idx + 1])
        rows: list[list[str]] = []
        pending_counterparty: tuple[str, str] | None = None
        sequence_counterparties: dict[int, tuple[str, str]] = {}
        prev_balance: float | None = None
        for block in ordered[header_idx + 1 :]:
            if str(block.get("type") or "") not in _TEXT_BLOCK_TYPES:
                continue
            text = _clean_cell(block.get("text"))
            if not text or _is_header_or_meta_text(text):
                continue
            if not _DATE_RE.search(text):
                pending_counterparty = _counterparty_hint_from_text(text) or pending_counterparty
                continue
            leading_orphan = _leading_orphan_text(text)
            if leading_orphan:
                orphan_hint = _counterparty_hint_from_text(leading_orphan)
                first_sequence = _first_fragment_sequence(text)
                if orphan_hint and first_sequence is not None and first_sequence > 1:
                    sequence_counterparties.setdefault(first_sequence - 1, orphan_hint)
            for fragment in _ledger_fragments(text):
                row = _parse_paragraph_ledger_fragment(
                    fragment,
                    prev_balance=prev_balance,
                    counterparty_before_account=counterparty_before_account,
                )
                if not row:
                    continue
                if pending_counterparty and _row_counterparty_missing(row):
                    _fill_row_counterparty(row, pending_counterparty)
                    pending_counterparty = None
                _set_row_source_page(row, page_no)
                rows.append(row)
                try:
                    prev_balance = float(row[3])
                except ValueError:
                    prev_balance = None
        if len(rows) < _MIN_DISTRIBUTED_ROWS:
            distributed_rows = _extract_distributed_ledger_rows(
                ordered,
                header_idx,
                counterparty_before_account=counterparty_before_account,
            )
            if distributed_rows:
                for row in distributed_rows:
                    _set_row_source_page(row, page_no)
                rows = distributed_rows
        if rows:
            _apply_sequence_counterparties(rows, sequence_counterparties)
            rows = _sort_rows_by_sequence(rows)
            rows = _repair_balance_chain_rows(rows)
            tables.append([_STANDARD_HEADER, *rows])
    return tables


def _distributed_header_end(ordered: list[dict[str, Any]]) -> int:
    """Find a ledger header whose column labels were emitted as separate blocks."""
    accumulated: list[str] = []
    support_groups = (
        ("摘要",),
        ("收/支", "收支", "收入/支出", "借/贷", "借贷"),
        ("对方账号", "对方账户"),
        ("对方户名", "对方名称"),
        ("序号",),
        ("交易流水号", "流水号", "凭证号", "支票号码"),
    )
    for idx, block in enumerate(ordered[:80]):
        if str(block.get("type") or "") not in _TEXT_BLOCK_TYPES:
            continue
        text = normalize_header_cell(str(block.get("text") or ""))
        if not text:
            continue
        accumulated.append(text)
        joined = "".join(accumulated)
        has_date = _has_ledger_date_header(joined)
        has_amount = any(
            marker in joined
            for marker in (
                *_AMOUNT_HEADER_MARKERS,
                "收入金额",
                "支出金额",
                "借方/贷方金额",
                "借贷方金额",
            )
        )
        has_balance = "余额" in joined
        support_count = sum(any(marker in joined for marker in group) for group in support_groups)
        if has_date and has_amount and has_balance and support_count >= 2:
            return idx
    return -1


def _has_ledger_date_header(text: str) -> bool:
    """Return whether text contains an explicit or OCR-split transaction-date header."""
    compact = re.sub(r"\s+", "", text)
    if any(marker in compact for marker in ("交易日期", "交易时间", "记账日期")):
        return True
    # Narrow scanned columns are commonly OCR'd as two stacked fragments:
    # ``交易日 借贷标`` / ``期 志``.  Keep the fuzzy rule anchored to
    # ``交易`` and the trailing ``期`` so unrelated ``起止日期`` metadata
    # cannot satisfy the ledger-header gate.
    return bool(re.search(r"交易[日曰口][^0-9]{0,8}期", compact))


def _extract_distributed_ledger_rows(
    ordered: list[dict[str, Any]],
    header_idx: int,
    *,
    counterparty_before_account: bool = False,
) -> list[list[str]]:
    """Parse rows whose fields are spread across positioned text blocks."""
    values: list[str] = []
    for block in ordered[header_idx + 1 :]:
        if str(block.get("type") or "") not in _TEXT_BLOCK_TYPES:
            continue
        text = _clean_cell(block.get("text"))
        if not text or _is_header_or_meta_text(text):
            continue
        values.append(text)
    fragments = _date_anchored_fragments(" ".join(values))
    if not fragments:
        return []

    rows: list[list[str]] = []
    prev_balance: float | None = None
    for fragment in fragments:
        row = _parse_paragraph_ledger_fragment(
            fragment,
            prev_balance=prev_balance,
            counterparty_before_account=counterparty_before_account,
        )
        if not row:
            continue
        rows.append(row)
        try:
            prev_balance = float(row[3])
        except ValueError:
            prev_balance = None
    if len(rows) < _MIN_DISTRIBUTED_ROWS:
        return []
    if len(rows) / len(fragments) < _MIN_DISTRIBUTED_VALID_RATIO:
        return []
    return rows


def _page_number_from_id(page_id: str) -> int:
    match = re.search(r"(\d+)$", str(page_id or ""))
    return int(match.group(1)) if match else 1


def _date_anchored_fragments(text: str) -> list[str]:
    """Split distributed rows from each date anchor to the next date anchor."""
    matches = _transaction_date_matches(text)
    starts = [_date_fragment_start(text, match.start()) for match in matches]
    return [
        text[starts[idx] : starts[idx + 1] if idx + 1 < len(starts) else len(text)].strip()
        for idx, match in enumerate(matches)
    ]


def _date_fragment_start(text: str, date_start: int) -> int:
    prefix = text[:date_start]
    match = re.search(r"(?:^|\s)(\d{1,5})\s+$", prefix)
    if not match:
        return date_start
    return match.start(1)


def _block_sort_key(block: dict[str, Any]) -> tuple[float, float]:
    bbox = block.get("bbox") or [0, 0, 0, 0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
        return (0.0, 0.0)
    return (float(bbox[1] or 0), float(bbox[0] or 0))


def _is_paragraph_ledger_header(block: dict[str, Any]) -> bool:
    text = normalize_header_cell(str(block.get("text") or ""))
    has_date = "交易日期" in text or "交易时间" in text
    has_direction = "收/支" in text or "收支" in text or _has_signed_amount_header(text)
    has_balance = "账户余额" in text or "余额" in text
    return has_date and has_direction and _has_amount_header(text) and has_balance


def _has_signed_amount_header(text: str) -> bool:
    return _has_amount_header(text) and "收/支" not in text and "收支" not in text


def _has_amount_header(text: str) -> bool:
    return any(marker in text for marker in _AMOUNT_HEADER_MARKERS)


def _is_header_or_meta_text(text: str) -> bool:
    normalized = normalize_header_cell(text)
    if "交易日期" in normalized and _has_amount_header(normalized):
        return True
    return bool(_NOISE_RE.search(text) and not _DATE_RE.search(text))


def _ledger_fragments(text: str) -> list[str]:
    """Split a text block into date-centered ledger fragments."""
    if _has_sequence_date_rows(text):
        return _date_anchored_fragments(text)
    matches = _transaction_date_matches(text)
    if not matches:
        return []
    fragments: list[str] = []
    for idx, match in enumerate(matches):
        prev_end = matches[idx - 1].end() if idx > 0 else 0
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        start = prev_end
        end = next_start
        fragment = text[start:end].strip()
        if match.group(1) not in fragment:
            fragment = f"{match.group(1)} {fragment}"
        fragments.append(fragment)
    return fragments


def _transaction_date_matches(text: str) -> list[re.Match[str]]:
    """Return row-date anchors while excluding dates inside narrative periods."""
    matches: list[re.Match[str]] = []
    for match in _DATE_RE.finditer(text):
        before = text[max(0, match.start() - 24) : match.start()]
        after = text[match.end() : match.end() + 4]
        compact_before = re.sub(r"\s+", "", before)
        compact_after = re.sub(r"\s+", "", after)
        if compact_before.endswith("至") or compact_after.startswith("至"):
            continue
        if any(marker in compact_before for marker in ("所属时间", "计息期间", "利息期间", "起止日期", "账期")):
            continue
        matches.append(match)
    return matches


def _leading_orphan_text(text: str) -> str:
    first_date = _DATE_RE.search(text)
    if first_date is None:
        return ""
    start = _date_fragment_start(text, first_date.start())
    return _clean_cell(text[:start])


def _first_fragment_sequence(text: str) -> int | None:
    first_date = _DATE_RE.search(text)
    if first_date is None:
        return None
    start = _date_fragment_start(text, first_date.start())
    return _extract_sequence_no(text[start : first_date.end()])


def _has_sequence_date_rows(text: str) -> bool:
    return bool(re.search(r"(?:^|\s)\d{1,5}\s+20\d{6}(?!\d)", _clean_cell(text)))


def _counterparty_precedes_account(header_blocks: list[dict[str, Any]]) -> bool:
    """Return whether the visible source header orders name before account and summary."""
    header = "".join(normalize_header_cell(str(block.get("text") or "")) for block in header_blocks)
    party_pos = header.rfind("对方户名")
    account_positions = [header.rfind(marker) for marker in ("对方账号", "对方账户")]
    account_pos = max(account_positions)
    summary_pos = header.rfind("摘要")
    return party_pos >= 0 and account_pos > party_pos and summary_pos > account_pos


def _parse_paragraph_ledger_fragment(
    fragment: str,
    *,
    prev_balance: float | None,
    counterparty_before_account: bool = False,
) -> list[str]:
    date_raw, _direction_raw = _split_date_direction(fragment)
    date = _normalize_date(date_raw)
    direction = _normalize_direction(fragment)
    if not date or not direction:
        return []

    amounts = _amount_tokens(fragment)
    if len(amounts) < 2:
        return []
    amount, balance = _choose_amount_balance(amounts, direction, prev_balance)
    if amount is None or balance is None:
        return []

    counter_account = _extract_counter_account(fragment)
    ordered_fields = (
        _extract_name_account_summary_order(fragment, counter_account) if counterparty_before_account else None
    )
    if ordered_fields is not None:
        counterparty, summary = ordered_fields
    else:
        summary = _extract_summary(fragment)
        counterparty = _extract_counterparty_from_fragment(fragment, counter_account)
    return [
        date,
        direction,
        f"{amount:.2f}",
        f"{balance:.2f}",
        summary,
        counter_account,
        counterparty,
        "",
        "",
        _repair_metadata(
            sequence_no=_extract_sequence_no(fragment),
            signed_amount=_has_signed_amount_token(amounts),
        ),
        "",
    ]


def _extract_name_account_summary_order(fragment: str, counter_account: str) -> tuple[str, str] | None:
    """Extract fields for headers ordered as counterparty name, account, then summary."""
    if not counter_account:
        return None
    account_start = fragment.find(counter_account)
    if account_start < 0:
        return None
    amount_matches = list(_AMOUNT_TOKEN_RE.finditer(fragment[:account_start]))
    if len(amount_matches) < 2:
        return None
    party = _clean_counterparty(fragment[amount_matches[1].end() : account_start])
    summary = _clean_cell(fragment[account_start + len(counter_account) :])
    summary = re.sub(r"^(?:摘要|用途)\s*[:：]?\s*", "", summary).strip()
    return party, summary


def _amount_tokens(text: str) -> list[tuple[str, float, int]]:
    out: list[tuple[str, float, int]] = []
    for match in _AMOUNT_TOKEN_RE.finditer(text):
        raw = re.sub(r"\s+", "", match.group(0)).replace(",", "")
        try:
            out.append((raw, float(raw), match.start()))
        except ValueError:
            continue
    return out


def _has_signed_amount_token(amounts: list[tuple[str, float, int]]) -> bool:
    return any(raw.startswith(("+", "-")) for raw, _, _ in amounts)


def _extract_sequence_no(fragment: str) -> int | None:
    match = re.search(r"(?:^|\s)(\d{1,5})\s+20\d{6}(?!\d)", _clean_cell(fragment))
    return int(match.group(1)) if match else None


def _choose_amount_balance(
    amounts: list[tuple[str, float, int]],
    direction: str,
    prev_balance: float | None,
) -> tuple[float | None, float | None]:
    if prev_balance is not None:
        best: tuple[float, float, float] | None = None
        for amount_idx, (_, raw_amount, _) in enumerate(amounts):
            amount = abs(raw_amount)
            if amount <= 0:
                continue
            for balance_idx, (_, balance, _) in enumerate(amounts):
                if balance_idx == amount_idx:
                    continue
                expected = prev_balance + amount if direction == "收入" else prev_balance - amount
                error = abs(round(expected - balance, 2))
                candidate = (error, amount, balance)
                if best is None or candidate < best:
                    best = candidate
        if best is not None and best[0] <= 0.05:
            return best[1], best[2]

    if len(amounts) >= 2:
        signed = next((item for item in amounts if item[0].startswith(("+", "-"))), None)
        if signed is not None:
            balance = next((item for item in amounts if item is not signed), None)
            if balance is not None:
                return abs(signed[1]), balance[1]
        return abs(amounts[0][1]), amounts[1][1]
    return None, None


def _repair_balance_chain_rows(rows: list[list[str]]) -> list[list[str]]:
    """Choose amount/balance orientation that minimizes page-local chain breaks."""
    if len(rows) < 2:
        return rows
    candidate_rows: list[list[list[str]]] = []
    for row in rows:
        candidates = [row]
        try:
            amount = float(row[2])
            balance = float(row[3])
        except (TypeError, ValueError):
            candidate_rows.append(candidates)
            continue
        if not _repair_meta(row).get("signed_amount") and abs(amount - balance) > 0.001:
            swapped = list(row)
            swapped[2] = f"{balance:.2f}"
            swapped[3] = f"{amount:.2f}"
            candidates.append(swapped)
        candidate_rows.append(candidates)

    forward_rows, forward_cost = _select_balance_chain_candidates(candidate_rows)
    reverse_rows, reverse_cost = _select_balance_chain_candidates(candidate_rows, reverse=True)
    if reverse_cost + 0.001 < forward_cost:
        return reverse_rows
    return forward_rows


def _select_balance_chain_candidates(
    candidate_rows: list[list[list[str]]],
    *,
    reverse: bool = False,
) -> tuple[list[list[str]], float]:
    working = list(reversed(candidate_rows)) if reverse else candidate_rows
    # dp[row][candidate] = (cost, previous_candidate_index)
    dp: list[list[tuple[float, int | None]]] = []
    dp.append([(_SWAP_ORIENTATION_PENALTY * idx, None) for idx, _candidate in enumerate(working[0])])
    for row_idx in range(1, len(working)):
        current_scores: list[tuple[float, int | None]] = []
        for cand_idx, candidate in enumerate(working[row_idx]):
            best: tuple[float, int | None] | None = None
            for prev_idx, prev_candidate in enumerate(working[row_idx - 1]):
                prev_cost = dp[row_idx - 1][prev_idx][0]
                transition_cost = _balance_transition_cost(prev_candidate, candidate)
                swap_penalty = _SWAP_ORIENTATION_PENALTY * cand_idx
                score = prev_cost + transition_cost + swap_penalty
                if best is None or score < best[0]:
                    best = (score, prev_idx)
            current_scores.append(best or (9999.0, None))
        dp.append(current_scores)

    last_idx = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx][0])
    total_cost = dp[-1][last_idx][0]
    selected = [0 for _ in working]
    selected[-1] = last_idx
    for row_idx in range(len(working) - 1, 0, -1):
        prev_idx = dp[row_idx][selected[row_idx]][1]
        selected[row_idx - 1] = int(prev_idx or 0)
    rows_out = [working[row_idx][selected[row_idx]] for row_idx in range(len(working))]
    if reverse:
        rows_out = list(reversed(rows_out))
    return [_clear_repair_marker(row) for row in rows_out], total_cost


def _clear_repair_marker(row: list[str]) -> list[str]:
    cleaned = list(row)
    if len(cleaned) > 9 and str(cleaned[9]).startswith(_REPAIR_META_PREFIX):
        cleaned[9] = ""
    return cleaned


def _sort_rows_by_sequence(rows: list[list[str]]) -> list[list[str]]:
    keyed: list[tuple[int, int, list[str]]] = []
    for index, row in enumerate(rows):
        sequence = _repair_meta(row).get("sequence_no")
        if sequence is None:
            continue
        keyed.append((int(sequence), index, row))
    if len(keyed) < 3 or len(keyed) / len(rows) < 0.8:
        return rows
    if len({sequence for sequence, _, _row in keyed}) < len(keyed) * 0.8:
        return rows
    keyed_by_index = {index: row for _sequence, index, row in keyed}
    sorted_rows = [row for _sequence, _index, row in sorted(keyed, key=lambda item: item[0])]
    leftovers = [row for index, row in enumerate(rows) if index not in keyed_by_index]
    return sorted_rows + leftovers


def _set_row_source_page(row: list[str], page_no: int) -> None:
    while len(row) < len(_STANDARD_HEADER):
        row.append("")
    row[10] = str(page_no)


def _apply_sequence_counterparties(rows: list[list[str]], sequence_counterparties: dict[int, tuple[str, str]]) -> None:
    if not sequence_counterparties:
        return
    for row in rows:
        sequence = _repair_meta(row).get("sequence_no")
        if not isinstance(sequence, int):
            continue
        hint = sequence_counterparties.get(sequence)
        if hint and _row_counterparty_missing(row):
            _fill_row_counterparty(row, hint)


def _counterparty_hint_from_text(text: str) -> tuple[str, str] | None:
    if _DATE_RE.search(text) or not _ACCOUNT_RE.search(text):
        return None
    account = _extract_counter_account(text)
    party = _extract_counterparty_from_fragment(text, account)
    if not account and not party:
        return None
    return account, party


def _row_counterparty_missing(row: list[str]) -> bool:
    return len(row) > 6 and not str(row[5] or "").strip() and not str(row[6] or "").strip()


def _fill_row_counterparty(row: list[str], hint: tuple[str, str]) -> None:
    account, party = hint
    if account and not str(row[5] or "").strip():
        row[5] = account
    if party and not str(row[6] or "").strip():
        row[6] = party


def _repair_metadata(*, sequence_no: int | None, signed_amount: bool) -> str:
    parts: list[str] = []
    if sequence_no is not None:
        parts.append(f"seq={sequence_no}")
    if signed_amount:
        parts.append("signed=1")
    return _REPAIR_META_PREFIX + ";".join(parts) if parts else ""


def _repair_meta(row: list[str]) -> dict[str, int | bool]:
    if len(row) <= 9:
        return {}
    text = str(row[9] or "")
    if not text.startswith(_REPAIR_META_PREFIX):
        return {}
    payload = text[len(_REPAIR_META_PREFIX) :]
    out: dict[str, int | bool] = {}
    for part in payload.split(";"):
        key, _, value = part.partition("=")
        if key == "seq" and value.isdigit():
            out["sequence_no"] = int(value)
        elif key == "signed" and value == "1":
            out["signed_amount"] = True
    return out


def _balance_transition_cost(previous: list[str], current: list[str]) -> float:
    try:
        prev_balance = float(previous[3])
        amount = float(current[2])
        balance = float(current[3])
    except (TypeError, ValueError):
        return 1.0
    direction = _normalize_direction(current[1])
    if direction == "收入":
        expected = prev_balance + amount
    elif direction == "支出":
        expected = prev_balance - amount
    else:
        return 1.0
    error = abs(round(expected - balance, 2))
    return 0.0 if error <= 0.01 else min(1.0, error / max(amount, 1.0))


def _extract_counter_account(fragment: str) -> str:
    amount_spans = [match.span() for match in _AMOUNT_TOKEN_RE.finditer(fragment)]
    for match in _ACCOUNT_RE.finditer(fragment):
        token = match.group(0)
        if _DATE_RE.fullmatch(token):
            continue
        if any(start <= match.start() and match.end() <= end for start, end in amount_spans):
            continue
        return token
    return ""


def _extract_summary(fragment: str) -> str:
    for keyword in _SUMMARY_KEYWORDS:
        if keyword in fragment:
            return keyword
    return ""


def _extract_counterparty_from_fragment(fragment: str, counter_account: str) -> str:
    if not counter_account:
        return ""
    account_position = fragment.find(counter_account)
    if account_position < 0:
        return ""
    account_end = account_position + len(counter_account)
    marker_pattern = (
        r"(?<!\S)(?:" + "|".join(re.escape(marker) for marker in _COUNTERPARTY_STOP_MARKERS) + r")(?=\s|;|$)"
    )
    marker_matches = list(re.finditer(marker_pattern, fragment))
    marker_before = next(
        (match for match in reversed(marker_matches) if match.end() <= account_position),
        None,
    )
    marker_after = next(
        (match for match in marker_matches if match.start() >= account_end),
        None,
    )
    marker_after_position = marker_after.start() if marker_after is not None else None
    suffix = fragment[account_end:]
    if marker_after_position is None:
        concatenated_positions = [
            (position, marker) for marker in _CONCATENATED_SUMMARY_MARKERS if (position := suffix.find(marker)) >= 0
        ]
        if concatenated_positions:
            position, _marker = min(concatenated_positions, key=lambda item: item[0])
            marker_after_position = account_end + position
    if marker_after_position is not None:
        text = fragment[account_end:marker_after_position]
    elif marker_before is not None:
        text = fragment[marker_before.end() : account_position]
    else:
        prefix = fragment[:account_position]
        suffix = fragment[account_end:]
        text = suffix if re.search(r"[\u4e00-\u9fff]", suffix) else prefix
    narrative_stop = re.search(r"\s+(?:普通汇兑|业务功能描述|贷款还息|用途|附言|备注)\s*[:：]", text)
    if narrative_stop is not None:
        text = text[: narrative_stop.start()]
    for value in _DATE_RE.findall(text):
        text = text.replace(value, " ")
    text = _remove_amount_tokens(text)
    if counter_account:
        text = text.replace(counter_account, " ")
    text = re.sub(r"\b(?:00)?98\b|\b(?:NY|YL)\d{4}\b|业务用章|第\s*\d+\s*页", " ", text)
    text = re.sub(r"[A-Za-z0-9&°]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _clean_counterparty(text)


def _remove_amount_tokens(text: str) -> str:
    return _AMOUNT_TOKEN_RE.sub(" ", text)


def _is_implicit_ledger_header(row: list[str]) -> bool:
    joined = "".join(normalize_header_cell(cell) for cell in row)
    has_date = "交易日期" in joined or "交易时间" in joined
    has_amount = "交易金额" in joined or "金额" in joined
    has_balance = "账户余额" in joined or "余额" in joined
    has_direction = "收/支" in joined or "收支" in joined or any("收/支" in str(cell) for cell in row)
    return has_date and has_amount and has_balance and has_direction


def _header_mapping(header: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, cell in enumerate(header):
        key = normalize_header_cell(cell)
        if ("交易日期" in key or "交易时间" in key) and ("收/支" in cell or "收支" in key):
            mapping["combined_date_direction"] = idx
        elif "交易日期" in key or "交易时间" in key:
            mapping["date"] = idx
        elif "收/支" in cell or "收支" in key or key in {"月收/支", "月收支"}:
            mapping["direction"] = idx
        elif "交易金额" in key or key == "金额":
            mapping["amount"] = idx
        elif "账户余额" in key or key == "余额":
            mapping["balance"] = idx
        elif "摘要" in key:
            mapping["summary"] = idx
        elif "对方账号" in key:
            mapping["counter_account"] = idx
        elif "对方户名" in key:
            mapping["counter_party"] = idx
        elif key == "机构":
            mapping["institution"] = idx
        elif "柜员" in key:
            mapping["teller"] = idx
        elif "备注" in key:
            mapping["remark"] = idx
    return mapping


def _normalize_row(row: list[str], mapping: dict[str, int]) -> list[str]:
    combined = _value(row, mapping.get("combined_date_direction"))
    date = _value(row, mapping.get("date"))
    direction = _value(row, mapping.get("direction"))
    if combined:
        parsed_date, parsed_direction = _split_date_direction(combined)
        date = date or parsed_date
        direction = direction or parsed_direction
    else:
        parsed_date, parsed_direction = _split_date_direction(f"{date}{direction}")
        date = parsed_date or date
        direction = parsed_direction or direction

    date = _normalize_date(date)
    direction = _normalize_direction(direction)
    amount = _normalize_amount_text(_value(row, mapping.get("amount")))
    balance = _normalize_amount_text(_value(row, mapping.get("balance")))
    if not date or not direction or not amount or not balance:
        return []
    return [
        date,
        direction,
        amount,
        balance,
        _value(row, mapping.get("summary")),
        _value(row, mapping.get("counter_account")),
        _clean_counterparty(_value(row, mapping.get("counter_party"))),
        _value(row, mapping.get("institution")),
        _value(row, mapping.get("teller")),
        _value(row, mapping.get("remark")),
    ]


def _split_date_direction(value: str) -> tuple[str, str]:
    text = _clean_cell(value)
    date_m = _DATE_RE.search(text)
    direction_m = _DIRECTION_RE.search(text)
    return (
        date_m.group(1) if date_m else "",
        direction_m.group(1) if direction_m else "",
    )


def _normalize_date(value: str) -> str:
    text = _clean_cell(value)
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    m = re.fullmatch(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ""


def _normalize_direction(value: str) -> str:
    text = _clean_cell(value)
    date_match = _DATE_RE.search(text)
    signed_amount = _AMOUNT_TOKEN_RE.search(text)
    direction_zone_start = date_match.end() if date_match is not None else 0
    direction_zone_end = signed_amount.start() if signed_amount is not None else len(text)
    direction_zone = text[direction_zone_start:direction_zone_end]
    debit_credit = ""
    if "借" in direction_zone:
        debit_credit = "借"
    elif "贷" in direction_zone:
        debit_credit = "贷"

    if debit_credit:
        is_negative = bool(signed_amount and signed_amount.group(0).startswith("-"))
        if debit_credit == "借":
            return "收入" if is_negative else "支出"
        return "支出" if is_negative else "收入"
    if signed_amount and signed_amount.group(0).startswith("-"):
        return "支出"
    if signed_amount and signed_amount.group(0).startswith("+"):
        return "收入"
    if any(token in text for token in ("收入", "收人")):
        return "收入"
    if any(token in text for token in ("支出", "支山", "支鼎", "攴出")):
        return "支出"
    return ""


def _normalize_amount_text(value: str) -> str:
    text = _clean_cell(value).replace(",", "")
    text = re.sub(r"\s+", "", text)
    m = re.search(r"\d+(?:\.\d{1,2})?", text)
    return m.group(0) if m else ""


def _clean_counterparty(value: str) -> str:
    text = _clean_cell(value)
    text = re.sub(r"(?:00)?98", "", text)
    text = re.sub(r"\b(?:NY|YL)\d{4}\b", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff）)])\s+(?=[\u4e00-\u9fff（(])", "", text)
    text = re.sub(r"^[\s+\-.,，。'\"`]+", "", text)
    text = re.sub(r"[\s+.,，。'\"`]+$", "", text)
    return text.strip()


def _value(row: list[str], idx: int | None) -> str:
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return _clean_cell(row[idx])


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip()


def _recover_from_text(full_text: str) -> list[list[list[str]]]:
    """Recover a borderless ledger emitted as one OCR text block per page."""
    if not full_text.strip():
        return []

    pages = [part for part in _PAGE_MARKER_RE.split(full_text) if part.strip()] or [full_text]
    tables: list[list[list[str]]] = []
    for page_text in pages:
        lines = [_clean_cell(line) for line in page_text.splitlines() if _clean_cell(line)]
        blocks = [{"type": "paragraph", "text": line} for line in lines]
        header_idx = _distributed_header_end(blocks)
        if header_idx < 0:
            continue

        body = " ".join(lines[header_idx + 1 :])
        fragments = _date_suffix_fragments(body)
        if not fragments:
            continue
        opening_balance = _opening_balance(" ".join(lines[: header_idx + 1]))
        rows = _parse_unpositioned_rows(fragments, opening_balance=opening_balance)
        if len(rows) < _MIN_DISTRIBUTED_ROWS:
            continue
        if len(rows) / len(fragments) < _MIN_DISTRIBUTED_VALID_RATIO:
            continue
        tables.append([_STANDARD_HEADER, *rows])
    return tables


def _date_suffix_fragments(text: str) -> list[str]:
    """Split OCR text where the transaction date may trail the other row fields."""
    matches = _transaction_date_matches(text)
    if not matches:
        return []
    fragments: list[str] = []
    start = 0
    for match in matches:
        fragment = text[start : match.end()].strip()
        start = match.end()
        if fragment:
            fragments.append(fragment)
    return fragments


def _opening_balance(text: str) -> float | None:
    normalized = text.lower()
    for marker in _OPENING_BALANCE_MARKERS:
        marker_idx = normalized.find(marker.lower())
        if marker_idx < 0:
            continue
        amounts = _amount_tokens(text[marker_idx : marker_idx + 80])
        if amounts:
            return amounts[0][1]
    return None


def _parse_unpositioned_rows(
    fragments: list[str],
    *,
    opening_balance: float | None,
) -> list[list[str]]:
    rows: list[list[str]] = []
    prev_balance = opening_balance
    for fragment in fragments:
        row = _parse_unpositioned_fragment(fragment, prev_balance=prev_balance)
        if not row:
            continue
        rows.append(row)
        try:
            prev_balance = float(row[3])
        except ValueError:
            prev_balance = None
    return _repair_balance_chain_rows(rows)


def _parse_unpositioned_fragment(fragment: str, *, prev_balance: float | None) -> list[str]:
    date_match = _DATE_RE.search(fragment)
    amounts = _amount_tokens(fragment)
    if date_match is None or len(amounts) < 2:
        return []

    explicit_direction = _normalize_direction(fragment)
    marker_directions = [_amount_marker_direction(fragment, token[2], token[0]) for token in amounts]
    debit_amount_indexes = {idx for idx, direction in enumerate(marker_directions) if direction == "支出"}
    candidates: list[tuple[float, str, float, float]] = []
    for amount_idx, amount_token in enumerate(amounts):
        if debit_amount_indexes and amount_idx not in debit_amount_indexes:
            continue
        amount = abs(amount_token[1])
        if amount <= 0:
            continue
        marker_direction = marker_directions[amount_idx]
        directions = (
            [explicit_direction or marker_direction] if explicit_direction or marker_direction else ["收入", "支出"]
        )
        for balance_idx, balance_token in enumerate(amounts):
            if balance_idx == amount_idx:
                continue
            balance = balance_token[1]
            for direction in directions:
                score = 0.03 * max(abs(amount_idx - balance_idx) - 1, 0)
                if marker_directions[balance_idx] == "支出":
                    score += 2.0
                if prev_balance is not None:
                    expected = prev_balance + amount if direction == "收入" else prev_balance - amount
                    score += min(abs(expected - balance) / max(amount, 1.0), 20.0)
                elif balance_idx < amount_idx:
                    score += 0.01
                candidates.append((score, direction, amount, balance))
    if not candidates:
        return []

    _, direction, amount, balance = min(candidates, key=lambda item: item[0])
    counter_account = _extract_counter_account(fragment)
    return [
        _normalize_date(date_match.group(1)),
        direction,
        f"{amount:.2f}",
        f"{balance:.2f}",
        _extract_summary(fragment),
        counter_account,
        _extract_counterparty_from_fragment(fragment, counter_account),
        "",
        "",
        "",
        "",
    ]


def _amount_marker_direction(text: str, start: int, raw_amount: str) -> str:
    tail = text[start + len(raw_amount) : start + len(raw_amount) + 8]
    if "借" in tail:
        return "支出"
    if "贷" in tail:
        return "收入"
    return ""


__all__ = ["recover_ocr_implicit_ledger_tables", "recovered_ocr_implicit_row_count"]
