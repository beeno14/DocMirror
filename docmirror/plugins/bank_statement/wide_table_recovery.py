# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover native-PDF wide debit/credit bank ledger tables.

This is a guarded candidate source for bank statements where the primary
Mirror/LTRO table candidate is sparse or malformed, but the source PDF still
contains a reliable native table. It is intentionally schema-driven rather than
bank-name-driven: a candidate must expose row number/date/debit/credit/balance
semantics before it is returned.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from docmirror.evidence.repair import RepairRequest
from docmirror.plugins.bank_statement.header_resolve import has_split_debit_credit_headers, normalize_header_cell

logger = logging.getLogger(__name__)

_DEBIT_CREDIT_REQUIRED = ("借方发生额", "贷方发生额", "余额")
_INCOME_EXPENSE_REQUIRED = ("支出金额", "收入金额", "余额")
_AMOUNT_HEADERS = (
    "交易金额",
    "发生额",
    "借方/贷方金额",
    "收入/支出金额",
    "支出/收入金额",
    "收/支金额",
    "支/收交易金额",
)
_ROW_ANCHOR_HEADERS = ("序号", "交易日期", "交易时间", "记账日期", "会计日期", "日期")
_BORDERLESS_DATE_RE = re.compile(r"(?:20\d{6}|20\d{2}[-/.]\d{2}[-/.]\d{2})")
_BORDERLESS_SIGNED_AMOUNT_RE = re.compile(r"[+-]\d[\d,]*(?:\.\d{1,2})?")
_BORDERLESS_BALANCE_RE = re.compile(r"-?\d[\d,]*(?:\.\d{1,2})?")
_BORDERLESS_ROW_RE = re.compile(
    r"^\s*(?:\d{1,6}\s+)?(?:20\d{6}|20\d{2}[-/.]\d{2}[-/.]\d{2})"
    r"(?:\s+(?:\d{6}|\d{1,2}:\d{2}:\d{2}))?.*?"
    r"[+-]?\d[\d,]*(?:\.\d{1,2})?\s+"
    r"-?\d[\d,]*(?:\.\d{1,2})?(?:\s|$)"
)
_BORDERLESS_FOOTER_MARKERS = ("数据缺失", "明细内容仅供参考", "本页合计")
_FOOTER_MARKERS = (
    "当前账单借方发生数",
    "当前账单贷方发生数",
    "本月累计借方发生数",
    "本月累计贷方发生数",
    "本月累计借方发生额",
    "本月累计贷方发生额",
    "总收入笔数",
    "总收入金额",
    "总支出笔数",
    "总支出金额",
    "出单截至日期",
    "以下此页无正文",
    "合计",
    "小计",
    "总计",
)
_COUNT_PATTERNS = (re.compile(r"(?:总条数|交易总笔数|总笔数|合计笔数)[:：]\s*(?P<count>\d+)"),)
_PAGE_COUNT_PATTERN = re.compile(r"本页交易笔数\s*[:：]\s*(?P<count>\d+)")
_SOURCE_PAGE_RE = re.compile(r"第\s*(?P<page>\d+)\s*页\s*(?:[/／-]\s*)?共\s*(?P<total>\d+)\s*页")
_NATIVE_DATETIME_RE = re.compile(r"(?P<date>20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\s+(?P<time>\d{1,2}:\d{2}:\d{2})")
_NATIVE_SIGNED_MONEY_RE = re.compile(r"[+-](?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")
_NATIVE_UNSIGNED_MONEY_RE = re.compile(r"(?<![\d,])(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")
_COMBINED_SIGNED_AMOUNT_HEADERS = ("收入/支出金额", "支出/收入金额", "收/支金额", "支/收交易金额")
_SPLIT_COUNT_PATTERNS = (
    re.compile(
        r"借方合计笔数[:：]\s*(?P<debit>\d+)\s*笔?.*?"
        r"贷方合计笔数[:：]\s*(?P<credit>\d+)\s*笔?",
        re.S,
    ),
    re.compile(r"借方笔数[:：]\s*(?P<debit>\d+).*?贷方笔数[:：]\s*(?P<credit>\d+)", re.S),
    re.compile(r"当前账单借方发生数[:：]\s*(?P<debit>\d+).*?当前账单贷方发生数[:：]\s*(?P<credit>\d+)", re.S),
    re.compile(r"本月累计借方发生数[:：]\s*(?P<debit>\d+).*?本月累计贷方发生数[:：]\s*(?P<credit>\d+)", re.S),
    re.compile(r"支出总笔数[:：]\s*(?P<debit>\d+).*?收入总笔数[:：]\s*(?P<credit>\d+)", re.S),
    re.compile(r"收入总笔数[:：]\s*(?P<credit>\d+).*?支出总笔数[:：]\s*(?P<debit>\d+)", re.S),
    re.compile(
        r"总收入笔数\s*[:：]?\s*(?P<credit>\d+).*?"
        r"总支出笔数\s*[:：]?\s*(?P<debit>\d+)",
        re.S,
    ),
    re.compile(
        r"总支出笔数\s*[:：]?\s*(?P<debit>\d+).*?"
        r"总收入笔数\s*[:：]?\s*(?P<credit>\d+)",
        re.S,
    ),
)
_DEBIT_TOTAL_PATTERNS = (
    re.compile(r"借方发生总额[:：]\s*(?P<value>[\d,]+\.\d{2})"),
    re.compile(r"本月累计借方发生额[:：]\s*(?P<value>[\d,]+\.\d{2})"),
    re.compile(r"支出总金额[:：]\s*(?P<value>[\d,]+\.\d{2})"),
    re.compile(r"本页支出合计\s*[:：]\s*(?P<value>[\d,]+\.\d{1,2})"),
)
_CREDIT_TOTAL_PATTERNS = (
    re.compile(r"贷方发生总额[:：]\s*(?P<value>[\d,]+\.\d{2})"),
    re.compile(r"本月累计贷方发生额[:：]\s*(?P<value>[\d,]+\.\d{2})"),
    re.compile(r"收入总金额[:：]\s*(?P<value>[\d,]+\.\d{2})"),
    re.compile(r"本页收入合计\s*[:：]\s*(?P<value>[\d,]+\.\d{1,2})"),
)


@dataclass(frozen=True)
class RowCountEvidence:
    """A transaction-count fact together with its source and confidence."""

    count: int
    source: str
    confidence: float
    page: int | None = None
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> RowCountEvidence:
        """Return an explicit no-evidence value."""
        return cls(count=0, source="none", confidence=0.0)


_SINGLE_LINE_COUNT_PATTERN = re.compile(
    r"(?:\u603b\u6761\u6570|\u4ea4\u6613\u603b\u7b14\u6570|\u603b\u7b14\u6570|\u5408\u8ba1\u7b14\u6570)"
    r"[ \t]*[:\uff1a][ \t]*(?P<count>\d+)"
)


def page_texts_from_parse_result(parse_result: Any) -> list[tuple[int, str]]:
    """Build page-local text scopes without relying on flattened PDF reading order."""
    if parse_result is None:
        return []
    result = getattr(parse_result, "to_read_view", lambda: parse_result)()
    page_texts: list[tuple[int, str]] = []
    for page in getattr(result, "pages", []) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        if page_number <= 0:
            continue
        parts = [
            str(getattr(block, "content", "") or getattr(block, "text", "") or "").strip()
            for block in getattr(page, "texts", []) or []
        ]
        for table in getattr(page, "tables", []) or []:
            for row in getattr(table, "rows", []) or []:
                cells = [str(getattr(cell, "text", "") or "").strip() for cell in getattr(row, "cells", []) or []]
                if any(cells):
                    parts.append(" ".join(cells))
        text = "\n".join(part for part in parts if part)
        if text:
            page_texts.append((page_number, text))
    return page_texts


def _count_scopes(text: str, page_texts: Iterable[tuple[int, str]] | None) -> list[tuple[int | None, str]]:
    scoped = [(int(page), str(value or "")) for page, value in (page_texts or ()) if str(value or "").strip()]
    if scoped:
        return scoped
    parts = [part for part in re.split(r"\f", str(text or "")) if part.strip()]
    return [(None, part) for part in (parts or [str(text or "")])]


def resolve_row_count_evidence(
    text: str,
    *,
    page_texts: Iterable[tuple[int, str]] | None = None,
) -> RowCountEvidence:
    """Resolve a transaction count only from a bounded, semantically labelled scope.

    Flattened PDF text is deliberately used only as a compatibility fallback. Counts
    that require a newline between the label and value are not accepted from that
    fallback because page numbers commonly occupy the next text position.
    """
    scopes = _count_scopes(text, page_texts)
    has_page_scopes = any(page is not None for page, _ in scopes)

    for page, scoped_text in scopes:
        for pattern in _SPLIT_COUNT_PATTERNS:
            match = pattern.search(scoped_text)
            if match:
                count = _safe_count(int(match.group("debit")) + int(match.group("credit")))
                if count:
                    return RowCountEvidence(count, "split_footer", 0.98, page)

    for page, scoped_text in scopes:
        patterns = _COUNT_PATTERNS if page is not None else (_SINGLE_LINE_COUNT_PATTERN,)
        for pattern in patterns:
            match = pattern.search(scoped_text)
            if match:
                count = _safe_count(int(match.group("count")))
                if count:
                    return RowCountEvidence(count, "header_total", 0.94, page)

    page_counts: list[tuple[int | None, int]] = []
    for page, scoped_text in scopes:
        for match in _PAGE_COUNT_PATTERN.finditer(scoped_text):
            count = _safe_count(int(match.group("count")))
            if count:
                page_counts.append((page, count))
    if page_counts:
        return RowCountEvidence(
            count=_safe_count(sum(count for _, count in page_counts)),
            source="page_footer",
            confidence=0.90,
            page=page_counts[0][0] if len(page_counts) == 1 else None,
        )

    # PageContent can be sparse even when the flattened canonical text retained
    # the footer. Only same-line or explicitly page-local labels are accepted in
    # this compatibility fallback, so a following page number cannot become a count.
    if has_page_scopes:
        if match := _SINGLE_LINE_COUNT_PATTERN.search(str(text or "")):
            count = _safe_count(int(match.group("count")))
            if count:
                return RowCountEvidence(count, "header_total", 0.90)
        flattened_page_counts = [
            _safe_count(int(match.group("count"))) for match in _PAGE_COUNT_PATTERN.finditer(str(text or ""))
        ]
        if flattened_page_counts and all(flattened_page_counts):
            return RowCountEvidence(sum(flattened_page_counts), "page_footer", 0.86)

    anchored_counts = [
        _count_borderless_transaction_anchors(scoped_text)
        for _, scoped_text in scopes
        if _has_borderless_source_header(scoped_text)
    ]
    if anchored_counts and all(count > 0 for count in anchored_counts):
        return RowCountEvidence(
            count=_safe_count(sum(anchored_counts)),
            source="page_transaction_anchors",
            confidence=0.93,
        )

    return RowCountEvidence.empty()


def recover_wide_bank_tables(parse_result: Any, full_text: str = "") -> list[list[list[str]]]:
    """Return high-confidence wide debit/credit table candidates from source PDF."""
    pdf_path = _source_pdf_path(parse_result)
    if not pdf_path:
        return []
    try:
        import pdfplumber
    except ImportError:
        logger.debug("[BankWideTableRecovery] pdfplumber unavailable")
        return []

    page_tables: list[list[list[str]]] = []
    native_money_hints = _native_money_hints(pdf_path)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                native_tables_found = False
                try:
                    native_tables = page.find_tables() or []
                except Exception:
                    native_tables = []
                for table_index, table in enumerate(native_tables):
                    normalized = _normalize_native_grid_table(
                        table,
                        page_number=page_number,
                        table_index=table_index,
                        money_hints=native_money_hints.get(page_number, {}),
                    )
                    if normalized:
                        page_tables.append(normalized)
                        native_tables_found = native_tables_found or bool(_select_wide_bank_table(normalized))
                if not native_tables:
                    for table_index, table in enumerate(page.extract_tables() or []):
                        normalized = _annotate_native_grid_matrix(
                            _normalize_table(table),
                            page_number=page_number,
                            table_index=table_index,
                            money_hints=native_money_hints.get(page_number, {}),
                        )
                        if normalized:
                            page_tables.append(normalized)
                            native_tables_found = native_tables_found or bool(_select_wide_bank_table(normalized))
                if not native_tables_found:
                    borderless = _recover_borderless_native_page(page, page_number)
                    if borderless:
                        page_tables.append(borderless)
    except Exception as exc:
        logger.debug("[BankWideTableRecovery] native PDF table recovery failed: %s", exc)
        return []

    candidates = _recover_cross_page_wide_tables(page_tables) if len(page_tables) > 1 else []
    for table in page_tables:
        wide = _select_wide_bank_table(table)
        if wide:
            candidates.append(wide)
    candidates = _dedupe_tables(candidates)

    if candidates:
        logger.info("[BankWideTableRecovery] recovered %d native wide table(s)", len(candidates))
    return candidates


def _native_money_hints(pdf_path: Path) -> dict[int, dict[tuple[str, str], list[tuple[str, str]]]]:
    """Extract page-local signed amount and balance hints from native reading order."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}

    hints: dict[int, dict[tuple[str, str], list[tuple[str, str]]]] = {}
    try:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = str(page.extract_text() or "")
            anchors = list(_NATIVE_DATETIME_RE.finditer(text))
            page_hints: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for index, anchor in enumerate(anchors):
                end = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
                fragment = text[anchor.end() : end]
                signed = _NATIVE_SIGNED_MONEY_RE.search(fragment)
                if signed is None:
                    continue
                balance = _NATIVE_UNSIGNED_MONEY_RE.search(fragment, signed.end())
                if balance is None:
                    continue
                key = (_normalize_native_date(anchor.group("date")), _normalize_native_time(anchor.group("time")))
                page_hints.setdefault(key, []).append((signed.group(0), balance.group(0)))
            if page_hints:
                hints[page_number] = page_hints
    except Exception as exc:
        logger.debug("[BankWideTableRecovery] native text amount hints unavailable: %s", exc)
        return {}
    return hints


def _normalize_native_grid_table(
    table: Any,
    *,
    page_number: int,
    table_index: int,
    money_hints: dict[tuple[str, str], list[tuple[str, str]]],
) -> list[list[str]]:
    """Normalize a pdfplumber grid and retain page-local row provenance."""
    try:
        matrix = _normalize_table(table.extract() or [])
    except Exception:
        return []
    row_bboxes = [_native_table_row_bbox(row) for row in getattr(table, "rows", []) or []]
    return _annotate_native_grid_matrix(
        matrix,
        page_number=page_number,
        table_index=table_index,
        money_hints=money_hints,
        row_bboxes=row_bboxes,
    )


def _annotate_native_grid_matrix(
    matrix: list[list[str]],
    *,
    page_number: int,
    table_index: int,
    money_hints: dict[tuple[str, str], list[tuple[str, str]]],
    row_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[list[str]]:
    """Attach source facts and clean only a confirmed combined signed-amount column."""
    if not matrix:
        return []
    header_index = next((index for index, row in enumerate(matrix[:8]) if is_wide_bank_header(row)), -1)
    if header_index < 0:
        return matrix
    headers = matrix[header_index]
    date_column = _source_date_column(headers)
    balance_column = _source_balance_column(headers)
    signed_amount_column = _combined_signed_amount_column(headers)
    if date_column < 0 or balance_column < 0 or signed_amount_column < 0:
        return matrix

    source_headers = [*headers, "_source_page", "_source_bbox", "_source_table_id", "_source_row_index"]
    out = [*matrix[:header_index], source_headers]
    hint_queues = {key: list(values) for key, values in money_hints.items()}
    table_id = f"native:p{page_number}:t{table_index}"
    for row_index, source_row in enumerate(matrix[header_index + 1 :], start=header_index + 1):
        row = list(source_row)
        key = _native_row_datetime_key(row, date_column)
        if key is None or not hint_queues.get(key):
            row_time = _native_row_time(row, date_column)
            matching_keys = [
                candidate for candidate, values in hint_queues.items() if values and candidate[1] == row_time
            ]
            if len(matching_keys) == 1:
                key = matching_keys[0]
        queue = hint_queues.get(key) if key is not None else None
        if queue:
            amount, balance = queue.pop(0)
            row[date_column] = f"{key[0]} {key[1]}"
            row[signed_amount_column] = amount
            row[balance_column] = balance
        else:
            cleaned_amount = _extract_native_signed_money(row[signed_amount_column])
            cleaned_balance = _extract_native_balance(row[balance_column])
            if cleaned_amount:
                row[signed_amount_column] = cleaned_amount
            if cleaned_balance:
                row[balance_column] = cleaned_balance
        bbox = row_bboxes[row_index] if row_bboxes and row_index < len(row_bboxes) else (0.0, 0.0, 0.0, 0.0)
        out.append(
            [
                *row,
                str(page_number),
                ",".join(f"{value:.3f}" for value in bbox),
                table_id,
                str(row_index),
            ]
        )
    return out


def _combined_signed_amount_column(headers: list[str]) -> int:
    for index, header in enumerate(headers):
        normalized = normalize_header_cell(header)
        if any(marker in normalized for marker in _COMBINED_SIGNED_AMOUNT_HEADERS):
            return index
    return -1


def _native_row_datetime_key(row: list[str], date_column: int) -> tuple[str, str] | None:
    if date_column < 0 or date_column >= len(row):
        return None
    match = _NATIVE_DATETIME_RE.search(str(row[date_column] or ""))
    if match is None:
        return None
    return _normalize_native_date(match.group("date")), _normalize_native_time(match.group("time"))


def _native_row_time(row: list[str], date_column: int) -> str:
    if date_column < 0 or date_column >= len(row):
        return ""
    match = re.search(r"(?<!\d)(?P<time>\d{1,2}:\d{2}:\d{2})(?!\d)", str(row[date_column] or ""))
    return _normalize_native_time(match.group("time")) if match else ""


def _normalize_native_date(value: str) -> str:
    parts = re.split(r"[-/.]", str(value or "").strip())
    if len(parts) != 3:
        return str(value or "").strip()
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def _normalize_native_time(value: str) -> str:
    parts = str(value or "").strip().split(":")
    if len(parts) != 3:
        return str(value or "").strip()
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"


def _extract_native_signed_money(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or ""))
    match = _NATIVE_SIGNED_MONEY_RE.search(compact)
    return match.group(0) if match else ""


def _extract_native_balance(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or ""))
    match = _NATIVE_UNSIGNED_MONEY_RE.search(compact)
    return match.group(0) if match else ""


def _native_table_row_bbox(row: Any) -> tuple[float, float, float, float]:
    cells = [cell for cell in (getattr(row, "cells", []) or []) if isinstance(cell, (list, tuple)) and len(cell) == 4]
    if not cells:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(float(cell[0]) for cell in cells),
        min(float(cell[1]) for cell in cells),
        max(float(cell[2]) for cell in cells),
        max(float(cell[3]) for cell in cells),
    )


def _count_borderless_transaction_anchors(text: str) -> int:
    """Count validated source rows from a page-local borderless ledger."""
    return sum(1 for line in str(text or "").splitlines() if _BORDERLESS_ROW_RE.search(line))


def _has_borderless_source_header(text: str) -> bool:
    return any(_looks_like_borderless_header_text(line) for line in str(text or "").splitlines())


def _recover_borderless_native_page(page: Any, page_number: int) -> list[list[str]]:
    """Recover a native source-column ledger from word coordinates.

    The branch is deliberately gated by the complete source header. It does not
    run for generic prose, payment documents, scanned OCR, or ledgers whose
    source column roles are ambiguous.
    """
    try:
        words = [
            dict(word)
            for word in page.extract_words(
                x_tolerance=1,
                y_tolerance=1,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            if str(word.get("text") or "").strip()
        ]
    except Exception:
        return []
    if not words:
        return []

    lines = _group_native_words_by_line(words)
    header_spec = next(
        ((line, spec) for line in lines if (spec := _borderless_header_spec(line)) is not None),
        None,
    )
    if header_spec is None:
        return []
    header_words, (source_headers, starts) = header_spec
    date_column = _source_date_column(source_headers)
    amount_columns = _source_amount_columns(source_headers)
    balance_column = _source_balance_column(source_headers)

    header_bottom = max(float(word.get("bottom") or word.get("top") or 0.0) for word in header_words)
    column_words = [
        (word, _column_index(float(word.get("x0") or 0.0), starts))
        for word in words
        if float(word.get("top") or 0.0) > header_bottom
    ]
    anchors = [
        word
        for word, column in column_words
        if column == date_column and _BORDERLESS_DATE_RE.fullmatch(str(word.get("text") or "").strip())
    ]
    anchors.sort(key=_word_vertical_center)
    if not anchors:
        return []

    centers = [_word_vertical_center(word) for word in anchors]
    gaps = [current - previous for previous, current in zip(centers, centers[1:]) if current > previous]
    typical_gap = median(gaps) if gaps else 18.0
    footer_top = min(
        (
            float(word.get("top") or 0.0)
            for word in words
            if _word_vertical_center(word) > centers[-1]
            and any(
                marker in normalize_header_cell(str(word.get("text") or "")) for marker in _BORDERLESS_FOOTER_MARKERS
            )
        ),
        default=float(getattr(page, "height", centers[-1] + typical_gap)),
    )

    header = [*source_headers, "_source_page", "_source_bbox"]
    rows: list[list[str]] = [list(header)]
    for index, anchor in enumerate(anchors):
        lower = header_bottom if index == 0 else (centers[index - 1] + centers[index]) / 2.0
        if index + 1 < len(anchors):
            upper = (centers[index] + centers[index + 1]) / 2.0
        else:
            upper = min(footer_top, centers[index] + max(typical_gap, 18.0))
        row_words = [(word, column) for word, column in column_words if lower <= _word_vertical_center(word) < upper]
        cells = [
            _join_native_cell_words([word for word, col in row_words if col == column])
            for column in range(len(source_headers))
        ]
        anchor_text = str(anchor.get("text") or "").strip()
        if not _BORDERLESS_DATE_RE.search(cells[date_column]):
            cells[date_column] = anchor_text
        if not _valid_borderless_row(
            cells,
            date_column=date_column,
            amount_columns=amount_columns,
            balance_column=balance_column,
        ):
            continue
        bbox = _native_row_bbox([word for word, _ in row_words])
        rows.append([*cells, str(page_number), ",".join(f"{value:.3f}" for value in bbox)])
    return rows if len(rows) > 1 else []


def _group_native_words_by_line(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item.get("top") or 0.0), float(item.get("x0") or 0.0))):
        top = float(word.get("top") or 0.0)
        if not lines or abs(top - float(lines[-1][0].get("top") or 0.0)) > 1.5:
            lines.append([word])
        else:
            lines[-1].append(word)
    return lines


def _borderless_header_spec(words: list[dict[str, Any]]) -> tuple[list[str], list[float]] | None:
    if not words:
        return None
    ordered = sorted(words, key=lambda item: float(item.get("x0") or 0.0))
    groups: list[list[dict[str, Any]]] = []
    for word in ordered:
        if not groups:
            groups.append([word])
            continue
        previous = groups[-1][-1]
        gap = float(word.get("x0") or 0.0) - float(previous.get("x1") or previous.get("x0") or 0.0)
        if gap <= 4.0:
            groups[-1].append(word)
        else:
            groups.append([word])
    headers = [
        unicodedata.normalize("NFKC", "".join(str(word.get("text") or "").strip() for word in group))
        for group in groups
    ]
    if len(headers) < 4 or not is_wide_bank_header(headers):
        return None
    if _source_date_column(headers) < 0 or not _source_amount_columns(headers) or _source_balance_column(headers) < 0:
        return None
    support_markers = ("摘要", "对方", "对手", "渠道", "附言", "用途", "借贷", "收支")
    joined = normalize_header_cell("".join(headers))
    if sum(marker in joined for marker in support_markers) < 1:
        return None
    return headers, [min(float(word.get("x0") or 0.0) for word in group) for group in groups]


def _looks_like_borderless_header_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", normalize_header_cell(text))
    has_date = any(marker in compact for marker in ("交易日期", "交易时间", "记账日期", "日期"))
    has_single_amount = any(marker in compact for marker in ("交易金额", "发生额"))
    has_split_amount = any(marker in compact for marker in ("收入", "贷方")) and any(
        marker in compact for marker in ("支出", "借方")
    )
    has_support = any(marker in compact for marker in ("摘要", "附言", "对方", "对手"))
    return has_date and (has_single_amount or has_split_amount) and "余额" in compact and has_support


def _source_date_column(headers: list[str]) -> int:
    for index, header in enumerate(headers):
        compact = normalize_header_cell(header)
        if any(marker in compact for marker in ("交易日期", "记账日期", "会计日期")):
            return index
    for index, header in enumerate(headers):
        if "日期" in normalize_header_cell(header):
            return index
    return -1


def _source_amount_columns(headers: list[str]) -> list[int]:
    exact_amount_headers = {
        "金额",
        "交易金额",
        "发生额",
        "收入",
        "支出",
        "收入金额",
        "支出金额",
        "收入/支出金额",
        "支出/收入金额",
        "收/支金额",
        "支/收交易金额",
        "借方",
        "贷方",
        "借方发生额",
        "贷方发生额",
        "转入金额",
        "转出金额",
    }
    indexes: list[int] = []
    for index, header in enumerate(headers):
        normalized = normalize_header_cell(header)
        if normalized in exact_amount_headers or any(
            marker in normalized
            for marker in (
                "交易金额",
                "借方发生额",
                "贷方发生额",
                "收入金额",
                "支出金额",
                "收入/支出金额",
                "支出/收入金额",
                "收/支金额",
                "支/收交易金额",
                "转入金额",
                "转出金额",
            )
        ):
            indexes.append(index)
    return indexes


def _source_balance_column(headers: list[str]) -> int:
    return next((index for index, header in enumerate(headers) if "余额" in normalize_header_cell(header)), -1)


def _column_index(x0: float, starts: list[float]) -> int:
    boundaries = [(left + right) / 2.0 for left, right in zip(starts, starts[1:])]
    return sum(x0 >= boundary for boundary in boundaries)


def _word_vertical_center(word: dict[str, Any]) -> float:
    top = float(word.get("top") or 0.0)
    bottom = float(word.get("bottom") or top)
    return (top + bottom) / 2.0


def _join_native_cell_words(words: list[dict[str, Any]]) -> str:
    if not words:
        return ""
    lines = _group_native_words_by_line(words)
    return "\n".join(
        "".join(
            str(word.get("text") or "").strip() for word in sorted(line, key=lambda item: float(item.get("x0") or 0.0))
        )
        for line in lines
    )


def _valid_borderless_row(
    cells: list[str],
    *,
    date_column: int,
    amount_columns: list[int],
    balance_column: int,
) -> bool:
    amount_values = [cells[index].replace(" ", "") for index in amount_columns if index < len(cells)]
    return bool(
        date_column >= 0
        and balance_column >= 0
        and date_column < len(cells)
        and balance_column < len(cells)
        and _BORDERLESS_DATE_RE.search(cells[date_column])
        and any(
            _BORDERLESS_SIGNED_AMOUNT_RE.fullmatch(value) or _BORDERLESS_BALANCE_RE.fullmatch(value)
            for value in amount_values
            if value
        )
        and _BORDERLESS_BALANCE_RE.fullmatch(cells[balance_column].replace(" ", ""))
    )


def _native_row_bbox(words: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    if not words:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(float(word.get("x0") or 0.0) for word in words),
        min(float(word.get("top") or 0.0) for word in words),
        max(float(word.get("x1") or word.get("x0") or 0.0) for word in words),
        max(float(word.get("bottom") or word.get("top") or 0.0) for word in words),
    )


def count_expected_rows_from_bank_footer(
    text: str,
    *,
    page_texts: Iterable[tuple[int, str]] | None = None,
) -> int:
    """Return the compatible integer form of :func:`resolve_row_count_evidence`."""
    return resolve_row_count_evidence(text, page_texts=page_texts).count


def audit_bank_statement_invariants(
    records: list[dict[str, Any]],
    text: str,
    *,
    page_texts: Iterable[tuple[int, str]] | None = None,
) -> list[str]:
    """Hard semantic gates for bank ledger rows against source footer totals."""
    failures: list[str] = []
    if page_gap_warning := _source_page_gap_warning(text):
        failures.append(page_gap_warning)
    expected = count_expected_rows_from_bank_footer(text, page_texts=page_texts)
    if expected > 0 and len(records) != expected:
        failures.append(f"bank_invariant_failed:row_count:{len(records)}/{expected}")

    normalized = [rec.get("normalized") or {} for rec in records]
    debit_rows = [row for row in normalized if row.get("direction") == "expense"]
    credit_rows = [row for row in normalized if row.get("direction") == "income"]
    reported_counts = _reported_direction_counts(text, page_texts=page_texts)
    if reported_counts is not None:
        expected_debit, expected_credit = reported_counts
        if len(debit_rows) != expected_debit:
            failures.append(f"bank_invariant_failed:debit_count:{len(debit_rows)}/{expected_debit}")
        if len(credit_rows) != expected_credit:
            failures.append(f"bank_invariant_failed:credit_count:{len(credit_rows)}/{expected_credit}")
    debit_total = _footer_amount(text, _DEBIT_TOTAL_PATTERNS)
    credit_total = _footer_amount(text, _CREDIT_TOTAL_PATTERNS)
    if debit_total is not None:
        actual = round(sum(_float(row.get("amount")) for row in debit_rows), 2)
        if abs(actual - debit_total) > 0.01:
            failures.append(f"bank_invariant_failed:debit_total:{actual:.2f}/{debit_total:.2f}")
    if credit_total is not None:
        actual = round(sum(_float(row.get("amount")) for row in credit_rows), 2)
        if abs(actual - credit_total) > 0.01:
            failures.append(f"bank_invariant_failed:credit_total:{actual:.2f}/{credit_total:.2f}")

    breaks, checked = _best_balance_chain_breaks(normalized)
    if checked > 0 and breaks > 0:
        failures.append(f"bank_invariant_failed:balance_chain:{breaks}/{checked}")
        failures.extend(_balance_chain_break_review_items(normalized, limit=3))
    return failures


def _reported_direction_counts(
    text: str,
    *,
    page_texts: Iterable[tuple[int, str]] | None = None,
) -> tuple[int, int] | None:
    for _, source in _count_scopes(text, page_texts):
        for pattern in _SPLIT_COUNT_PATTERNS:
            match = pattern.search(source)
            if match:
                return int(match.group("debit")), int(match.group("credit"))
    return None


def _best_balance_chain_breaks(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return min breaks across chronological and reverse-chronological order."""
    forward = _balance_chain_breaks(rows)
    backward = _balance_chain_breaks(list(reversed(rows)))
    if backward[1] > forward[1]:
        return backward
    if forward[1] > backward[1]:
        return forward
    return min(forward, backward, key=lambda item: item[0])


def _balance_chain_breaks(rows: list[dict[str, Any]]) -> tuple[int, int]:
    checked = 0
    breaks = 0
    prev_balance: float | None = None
    prev_sequence: int | None = None
    for row in rows:
        direction = row.get("direction")
        if direction not in ("income", "expense"):
            continue
        balance = row.get("balance")
        amount = row.get("amount")
        if balance in (None, "") or amount in (None, ""):
            continue
        balance_f = _float(balance)
        amount_f = _float(amount)
        sequence = _sequence_number(row)
        sequence_is_contiguous = prev_sequence is None or sequence is None or abs(sequence - prev_sequence) == 1
        if prev_balance is not None and sequence_is_contiguous:
            checked += 1
            expected_balance = prev_balance + amount_f if direction == "income" else prev_balance - amount_f
            if abs(round(expected_balance - balance_f, 2)) > 0.01:
                breaks += 1
        prev_balance = balance_f
        prev_sequence = sequence
    return breaks, checked


def _balance_chain_break_review_items(rows: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    items: list[str] = []
    prev_balance: float | None = None
    prev_row: dict[str, Any] | None = None
    prev_sequence: int | None = None
    for row_index, row in enumerate(rows, start=1):
        direction = row.get("direction")
        if direction not in ("income", "expense"):
            continue
        balance = row.get("balance")
        amount = row.get("amount")
        if balance in (None, "") or amount in (None, ""):
            continue
        balance_f = _float(balance)
        amount_f = _float(amount)
        sequence = _sequence_number(row)
        sequence_is_contiguous = prev_sequence is None or sequence is None or abs(sequence - prev_sequence) == 1
        if prev_balance is not None and sequence_is_contiguous:
            expected_balance = prev_balance + amount_f if direction == "income" else prev_balance - amount_f
            delta = round(balance_f - expected_balance, 2)
            if abs(delta) > 0.01:
                items.append(
                    "bank_review:balance_chain_gap:"
                    f"row={row_index}:"
                    f"date={row.get('date') or row.get('transaction_date') or ''}:"
                    f"direction={direction}:"
                    f"amount={amount_f:.2f}:"
                    f"prev_balance={prev_balance:.2f}:"
                    f"expected_balance={expected_balance:.2f}:"
                    f"actual_balance={balance_f:.2f}:"
                    f"delta={delta:.2f}"
                )
                missing_candidate = _single_missing_row_candidate(
                    previous_row=prev_row,
                    current_row=row,
                    current_row_index=row_index,
                    previous_balance=prev_balance,
                    current_balance=balance_f,
                    current_amount=amount_f,
                )
                if missing_candidate:
                    items.append(missing_candidate)
                    repair_request = _single_missing_row_repair_request(
                        previous_row=prev_row,
                        current_row=row,
                        current_row_index=row_index,
                    )
                    items.append(_repair_request_review_item(repair_request))
                if len(items) >= limit:
                    break
        prev_balance = balance_f
        prev_row = row
        prev_sequence = sequence
    return items


def _sequence_number(row: dict[str, Any]) -> int | None:
    value = row.get("sequence_no")
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _source_page_gap_warning(text: str) -> str:
    matches = list(_SOURCE_PAGE_RE.finditer(text or ""))
    if len(matches) < 2:
        return ""
    observed = {int(match.group("page")) for match in matches}
    declared_total = max(int(match.group("total")) for match in matches)
    if declared_total <= 0:
        return ""
    missing = [page for page in range(1, declared_total + 1) if page not in observed]
    if not missing:
        return ""
    return (
        "bank_review:source_page_gap:"
        f"observed={len(observed)}/{declared_total}:"
        f"missing_ranges={_compact_ranges(missing)}:"
        "action=manual_review"
    )


def _compact_ranges(values: list[int]) -> str:
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _single_missing_row_candidate(
    *,
    previous_row: dict[str, Any] | None,
    current_row: dict[str, Any],
    current_row_index: int,
    previous_balance: float,
    current_balance: float,
    current_amount: float,
) -> str:
    """Return a review-only candidate when one missing row can bridge a gap."""
    if previous_row is None:
        return ""
    current_direction = current_row.get("direction")
    if current_direction == "income":
        bridge_balance = current_balance - current_amount
    elif current_direction == "expense":
        bridge_balance = current_balance + current_amount
    else:
        return ""

    missing_delta = round(bridge_balance - previous_balance, 2)
    if abs(missing_delta) <= 0.01:
        return ""
    missing_direction = "income" if missing_delta > 0 else "expense"
    missing_amount = abs(missing_delta)
    if missing_amount <= 0 or missing_amount > 1_000_000_000:
        return ""

    previous_date = previous_row.get("date") or previous_row.get("transaction_date") or ""
    current_date = current_row.get("date") or current_row.get("transaction_date") or ""
    return (
        "bank_review:missing_row_candidate:"
        f"before_row={current_row_index}:"
        f"date_range={previous_date}..{current_date}:"
        f"direction={missing_direction}:"
        f"amount={missing_amount:.2f}:"
        f"balance={bridge_balance:.2f}:"
        "evidence=balance_chain_only:"
        "action=manual_review:not_auto_adopted"
    )


def _single_missing_row_repair_request(
    *,
    previous_row: dict[str, Any] | None,
    current_row: dict[str, Any],
    current_row_index: int,
) -> RepairRequest:
    previous_date = ""
    if previous_row is not None:
        previous_date = previous_row.get("date") or previous_row.get("transaction_date") or ""
    current_date = current_row.get("date") or current_row.get("transaction_date") or ""
    return RepairRequest(
        request_id=f"bank-ledger-balance-gap-before-row-{current_row_index}",
        domain="bank_statement",
        kind="missing_ledger_row_local_ocr",
        expected_schema=("date", "direction", "amount", "balance"),
        constraints=(
            "bank.balance_chain_consistency",
            "bank.date_order",
            "bank.amount_format",
            "bank.no_duplicate_transaction",
        ),
        context={
            "before_row": current_row_index,
            "date_range": f"{previous_date}..{current_date}",
            "previous_date": previous_date,
            "current_date": current_date,
        },
        reason="balance_chain_gap_single_missing_row_candidate",
    )


def _repair_request_review_item(request: RepairRequest) -> str:
    data = request.to_dict()
    return (
        "bank_review:repair_request:"
        f"id={data['request_id']}:"
        f"kind={data['kind']}:"
        f"can_render={str(data['can_render']).lower()}:"
        "action=manual_review:"
        "reason=missing_page_bbox"
    )


def is_footer_or_total_row(row: list[str] | tuple[str, ...] | None) -> bool:
    """Return true when a table row is a footer/total rather than a transaction."""
    if not row:
        return False
    joined = " ".join(str(cell or "").strip() for cell in row if str(cell or "").strip())
    return bool(joined and any(marker in joined for marker in _FOOTER_MARKERS))


def is_wide_bank_header(row: list[str] | tuple[str, ...] | None) -> bool:
    if not row:
        return False
    headers = [normalize_header_cell(str(cell or "")) for cell in row]
    joined = "".join(headers)
    has_required = all(normalize_header_cell(item) in joined for item in _DEBIT_CREDIT_REQUIRED) or all(
        normalize_header_cell(item) in joined for item in _INCOME_EXPENSE_REQUIRED
    )
    has_required = has_required or (
        normalize_header_cell("余额") in joined
        and any(normalize_header_cell(item) in joined for item in _AMOUNT_HEADERS)
    )
    has_required = has_required or (
        normalize_header_cell("余额") in joined and has_split_debit_credit_headers([[list(row)]])
    )
    has_anchor = any(normalize_header_cell(item) in joined for item in _ROW_ANCHOR_HEADERS)
    return has_required and has_anchor


def _select_wide_bank_table(table: list[list[str]]) -> list[list[str]]:
    if not table:
        return []
    for idx, row in enumerate(table[:8]):
        if not is_wide_bank_header(row):
            continue
        header = [str(cell or "").strip() for cell in row]
        rows = [header]
        for data_row in table[idx + 1 :]:
            if not data_row or not any(str(cell or "").strip() for cell in data_row):
                continue
            if is_footer_or_total_row(data_row):
                continue
            if _looks_like_transaction_row(data_row):
                rows.append([str(cell or "").strip() for cell in data_row])
        if len(rows) > 1:
            return rows
    return []


def _looks_like_transaction_row(row: list[str]) -> bool:
    joined = " ".join(str(cell or "").strip() for cell in row)
    if not re.search(r"(?<!\d)\d{8}(?!\d)|\d{4}[-/]\d{1,2}[-/]\d{1,2}", joined):
        return False
    if not re.search(r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{1,2}", joined):
        return False
    return True


def _recover_cross_page_wide_tables(page_tables: list[list[list[str]]]) -> list[list[list[str]]]:
    """Compose first-header + continuation native PDF tables into one logical ledger."""
    recovered: list[list[list[str]]] = []
    current_header: list[str] | None = None
    current_rows: list[list[str]] = []
    previous_seq = 0

    def flush() -> None:
        nonlocal current_header, current_rows, previous_seq
        if current_header and current_rows:
            recovered.append([current_header, *current_rows])
        current_header = None
        current_rows = []
        previous_seq = 0

    for table in page_tables:
        if not table:
            continue
        table = [[_clean_native_cell(cell) for cell in row] for row in table]
        header_idx = next((idx for idx, row in enumerate(table[:8]) if is_wide_bank_header(row)), -1)
        if header_idx >= 0:
            next_header = [str(cell or "").strip() for cell in table[header_idx]]
            if current_rows and current_header != next_header:
                flush()
            if current_header is None:
                current_header = next_header
            data_rows = table[header_idx + 1 :]
        elif current_header and _is_continuation_table(table, current_header, previous_seq):
            data_rows = table
        else:
            continue

        for row in data_rows:
            if not row or is_footer_or_total_row(row) or not _looks_like_transaction_row(row):
                continue
            normalized = _fit_row_width([str(cell or "").strip() for cell in row], len(current_header))
            seq = _row_sequence(normalized)
            if previous_seq and seq and seq != previous_seq + 1:
                flush()
                current_header = [str(cell or "").strip() for cell in table[header_idx]] if header_idx >= 0 else None
                if current_header is None:
                    continue
            current_rows.append(normalized)
            if seq:
                previous_seq = seq
    flush()
    return recovered


def _is_continuation_table(table: list[list[str]], header: list[str], previous_seq: int) -> bool:
    data_rows = [row for row in table if row and not is_footer_or_total_row(row) and _looks_like_transaction_row(row)]
    if not data_rows:
        return False
    width_ok = abs(max((len(row) for row in data_rows), default=0) - len(header)) <= 2
    first_seq = _row_sequence(data_rows[0])
    sequence_ok = not previous_seq or not first_seq or first_seq == previous_seq + 1
    return width_ok and sequence_ok


def _row_sequence(row: list[str]) -> int:
    first = str(row[0] or "").strip()
    return int(first) if re.fullmatch(r"\d{1,6}", first) else 0


def _fit_row_width(row: list[str], width: int) -> list[str]:
    if len(row) > width:
        return row[:width]
    if len(row) < width:
        return [*row, *([""] * (width - len(row)))]
    return row


def _dedupe_tables(tables: list[list[list[str]]]) -> list[list[list[str]]]:
    out: list[list[list[str]]] = []
    seen: set[tuple[int, str, str]] = set()
    for table in tables:
        if not table:
            continue
        key = (
            len(table),
            "|".join(table[0]),
            "|".join(table[-1] if len(table) > 1 else []),
        )
        if key in seen:
            continue
        if any(_table_contains(existing, table) for existing in out):
            continue
        contained = [index for index, existing in enumerate(out) if _table_contains(table, existing)]
        if contained:
            insert_at = contained[0]
            out = [existing for index, existing in enumerate(out) if index not in contained]
            out.insert(insert_at, table)
        else:
            out.append(table)
        seen.add(key)
    return out


def _table_contains(larger: list[list[str]], smaller: list[list[str]]) -> bool:
    if len(larger) < len(smaller) or not larger or not smaller:
        return False
    if _table_row_signature(larger[0]) != _table_row_signature(smaller[0]):
        return False
    large_rows = {_table_row_signature(row) for row in larger[1:]}
    return all(_table_row_signature(row) in large_rows for row in smaller[1:])


def _table_row_signature(row: list[str]) -> str:
    return "|".join(re.sub(r"\s+", "", str(cell or "")) for cell in row)


def _normalize_table(table: list[list[Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    width = max((len(row or []) for row in table or []), default=0)
    for row in table or []:
        values = [_clean_native_cell(cell) for cell in row or []]
        if len(values) < width:
            values.extend([""] * (width - len(values)))
        if any(values):
            rows.append(values)
    return rows


def _clean_native_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d{4}-\d{2})-\s+(\d{1,2})", r"\1-\2", text)
    text = re.sub(r"(\d{1,2}:\d{2}:\d)\s+(\d)\b", r"\1\2", text)
    return text.strip()


def _footer_amount(text: str, patterns: tuple[re.Pattern[str], ...]) -> float | None:
    for pat in patterns:
        matches = list(pat.finditer(text or ""))
        if matches:
            return round(sum(_float(match.group("value")) for match in matches), 2)
    return None


def _safe_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return count if 0 < count <= 10000 else 0


def _float(value: Any) -> float:
    try:
        return float(str(value or "0").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _source_pdf_path(parse_result: Any) -> Path | None:
    candidates = [
        getattr(parse_result, "file_path", None),
        getattr(parse_result, "source_path", None),
    ]
    provenance = getattr(parse_result, "provenance", None)
    if provenance is not None:
        props = getattr(provenance, "document_properties", None)
        if isinstance(props, dict):
            candidates.extend([props.get("file_path"), props.get("source_path"), props.get("path")])

    parser_info = getattr(parse_result, "parser_info", None)
    if parser_info is not None:
        opts = getattr(parser_info, "options", None)
        if isinstance(opts, dict):
            candidates.extend([opts.get("file_path"), opts.get("source_path")])

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.is_file() and path.suffix.lower() == ".pdf":
            return path
    return None
