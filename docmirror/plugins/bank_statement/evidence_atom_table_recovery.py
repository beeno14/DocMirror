# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover high-confidence split debit/credit ledgers from canonical evidence atoms."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any

_DATE_ANY_RE = re.compile(r"20\d{6}|20\d{2}[-/]\d{1,2}[-/]\d{1,2}")
_MONEY_RE = re.compile(r"^-?\d[\d,]*\.\d{2}$")
_MONEY_ANY_RE = re.compile(r"-?\d[\d,]*\.\d{2}")
_COMPOSITE_MARKERS = ("支出", "收入", "账户余额")
_RECOVERY_CACHE_KEY = "_bank_evidence_atom_recovery"
_NATIVE_DATETIME_CENSUS_SOURCE = "native_page_datetime_census"
_NATIVE_SIGNED_LEDGER_CENSUS_SOURCE = "native_page_signed_ledger_census"
_SOURCE_COVERAGE_CONFIDENCE = 0.80
_NATIVE_DATETIME_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_NATIVE_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_NATIVE_ACCOUNT_RE = re.compile(r"^(?:账号|账户号)\s*[:：]\s*(?P<account>[\d*]{6,40})$")
_NATIVE_LEDGER_TITLE_RE = re.compile(r"^\S{1,24}银行交易明细$")
_NATIVE_LEDGER_HEADERS = ("日期", "支出", "收入", "余额", "对方账户", "对方户名", "摘要/附言")
_CIB_NATIVE_LEDGER_TITLE = "兴业银行交易流水"
_CIB_NATIVE_ACCOUNT_RE = re.compile(r"^号[:：](?P<account>[\d*]{6,40})$")
_CIB_NATIVE_LEDGER_HEADERS = ("交易日期", "记账日期", "摘要", "对方户名", "对方账户/对方银行")
_CIB_NATIVE_COMPOUND_MONEY_HEADER = "支/收交易金额账户余额交易地点"
_CIB_NATIVE_FOOTER_PREFIX = "说明：交易明细涉及您的个人隐私"
_GEOMETRY_FOOTER_MARKERS = (
    "本页合计",
    "本页支出",
    "本页收入",
    "总收入笔数",
    "总收入金额",
    "总支出笔数",
    "总支出金额",
    "收入交易笔数",
    "收入金额合计",
    "支出交易笔数",
    "支出金额合计",
    "支出交易总额",
    "收入交易总额",
    "合计笔数",
    "收入总额",
    "支出总额",
    "当前账单借方发生数",
    "当前账单贷方发生数",
    "本月累计借方发生数",
    "本月累计贷方发生数",
    "借方发生额汇总",
    "贷方发生额汇总",
    "回单编号",
    "打印时间",
    "打印日期",
    "打印柜员",
    "打印机构",
    "打印完毕",
    "友情提示",
    "重要提示",
    "风险提示",
    "说明：交易明细涉及",
    "本回单",
    "对账单专用章",
    "https://",
    "http://",
)
_COMMON_BANK_SUMMARY_OCR_CORRECTIONS = {
    "网银路行互联": "网银跨行互联",
}
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

_GEOMETRY_OVERLAY_LABELS_BY_ROLE = {
    "date": ("交易日期", "业务日期", "日期"),
    "timestamp": ("交易时间", "交易日期时间", "日期时间"),
    "posting_date": ("记账日期", "会计日期", "入账日期"),
    "summary": ("交易摘要", "摘要"),
    "purpose": ("交易用途", "用途"),
    "direction": ("支/收", "收/支", "借/贷", "借贷"),
    "amount": ("交易金额", "发生金额", "发生额", "金额"),
    "balance": ("账户余额", "本次余额", "账面余额", "余额"),
    "transaction_location": ("交易地点", "交易场所"),
    "institution": ("交易机构", "交易网点", "交易地点"),
    "counter_party": ("对方户名", "对方名称", "对手方名称"),
    "counter_account": (
        "对方账户/对方银行",
        "对方账号/对方银行",
        "对方账户",
        "对方账号",
    ),
}


@dataclass(frozen=True)
class PositionedBlockRecovery:
    """A page-positioned record-block recovery result for candidate selection."""

    tables: list[list[list[str]]]
    row_sources: list[dict[str, Any]]
    expected_rows: int


def recovered_native_datetime_row_evidence(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> tuple[int, str, float]:
    """Count structurally witnessed rows in a bounded native ledger plane.

    This is a source-coverage signal, not an independent completeness proof.
    It deliberately recognizes only narrow, well-witnessed native layouts.
    One requires date/time anchors and reconciled debit/credit page totals; the
    other requires transaction/posting dates and a signed amount/balance chain.
    Both require the same account, declared page numbering, exact ledger headers,
    source-row geometry, and footer boundaries on every observed page.  These
    checks prove internal representation consistency but cannot rule out a row
    omitted symmetrically from all observed planes, so the result intentionally
    remains below the authoritative-count threshold.  Repeated values remain
    distinct because rows are matched as bbox-backed observations.
    """
    if source_route not in (None, "digital"):
        return 0, "", 0.0
    if not _native_document_plane_is_bounded(parse_result):
        return 0, "", 0.0

    from docmirror.plugins._runtime.evidence_access import text_atoms

    native_atoms = [
        atom
        for atom in text_atoms(parse_result)
        if isinstance(atom, dict)
        and str(atom.get("source_kind") or "").strip().lower() == "pdf_native"
        and str(atom.get("page_id") or "")
        and str(atom.get("text") or "").strip()
        and isinstance(atom.get("bbox"), list)
        and len(atom["bbox"]) >= 4
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in native_atoms:
        grouped[str(atom["page_id"])].append(atom)

    page_ids = _native_document_page_ids(parse_result)
    if not page_ids or set(grouped) != set(page_ids):
        return 0, "", 0.0

    expected_account = ""
    total_rows = 0
    datetime_layout_valid = True
    for page_number, page_id in enumerate(page_ids, start=1):
        page_coverage = _bounded_native_datetime_page_coverage(
            grouped[page_id],
            page_number=page_number,
            page_count=len(page_ids),
        )
        if page_coverage is None:
            datetime_layout_valid = False
            break
        page_rows, account = page_coverage
        if expected_account and account != expected_account:
            datetime_layout_valid = False
            break
        expected_account = account
        total_rows += page_rows

    if datetime_layout_valid and total_rows > 0 and expected_account:
        return total_rows, _NATIVE_DATETIME_CENSUS_SOURCE, _SOURCE_COVERAGE_CONFIDENCE

    cib_coverage = _bounded_cib_native_document_coverage(grouped, page_ids)
    if cib_coverage is not None:
        return cib_coverage[0], _NATIVE_SIGNED_LEDGER_CENSUS_SOURCE, _SOURCE_COVERAGE_CONFIDENCE
    return 0, "", 0.0


def _native_document_plane_is_bounded(parse_result: Any) -> bool:
    parser_info = getattr(parse_result, "parser_info", None)
    raw_method = getattr(parser_info, "extraction_method", "")
    extraction_method = str(getattr(raw_method, "value", raw_method) or "").strip().lower()
    if extraction_method and extraction_method != "digital":
        return False

    page_ids = _native_document_page_ids(parse_result)
    if not page_ids:
        return False
    expected_numbers = list(range(1, len(page_ids) + 1))
    if page_ids != [f"page:{number:04d}" for number in expected_numbers]:
        return False

    plane = getattr(parse_result, "evidence_plane", None)
    plane_pages = list(getattr(plane, "pages", None) or [])
    if len(plane_pages) != len(page_ids):
        return False
    for expected, page in zip(expected_numbers, plane_pages):
        if int(getattr(page, "page_number", 0) or 0) != expected:
            return False
        mode = str(getattr(page, "content_mode", "") or "").strip().lower()
        if mode and mode != "unknown" and "native" not in mode:
            return False

    options = getattr(parser_info, "options", None)
    if hasattr(options, "model_dump"):
        options = options.model_dump(mode="python")
    options = options if isinstance(options, dict) else {}
    source_page_count = int(options.get("source_page_count") or 0)
    if source_page_count and source_page_count != len(page_ids):
        return False
    selected_pages = options.get("selected_source_pages")
    if isinstance(selected_pages, list) and selected_pages:
        try:
            selected = [int(value) for value in selected_pages]
        except (TypeError, ValueError):
            return False
        if selected != expected_numbers:
            return False
    return True


def _native_document_page_ids(parse_result: Any) -> list[str]:
    pages = list(getattr(parse_result, "pages", None) or [])
    numbered: list[tuple[int, str]] = []
    for page in pages:
        page_number = int(getattr(page, "page_number", 0) or 0)
        if page_number <= 0:
            return []
        numbered.append((page_number, f"page:{page_number:04d}"))
    numbered.sort()
    if [number for number, _page_id in numbered] != list(range(1, len(numbered) + 1)):
        return []
    return [page_id for _number, page_id in numbered]


def _bounded_native_datetime_page_coverage(
    atoms: list[dict[str, Any]],
    *,
    page_number: int,
    page_count: int,
) -> tuple[int, str] | None:
    ordered = sorted(
        atoms,
        key=lambda atom: (
            float(atom["bbox"][1]),
            float(atom["bbox"][0]),
            str(atom.get("id") or ""),
        ),
    )
    header_atoms = [_first_exact(ordered, label) for label in _NATIVE_LEDGER_HEADERS]
    if any(atom is None for atom in header_atoms):
        return None
    headers = [atom for atom in header_atoms if atom is not None]
    header_y = median(_y_center(atom) for atom in headers)
    if any(abs(_y_center(atom) - header_y) > 1.5 for atom in headers):
        return None
    header_centers = [_x_center(atom) for atom in headers]
    if any(right <= left for left, right in zip(header_centers, header_centers[1:])):
        return None
    bounds = [
        float("-inf"),
        *((left + right) / 2 for left, right in zip(header_centers, header_centers[1:])),
        float("inf"),
    ]

    title = next(
        (
            atom
            for atom in ordered
            if _NATIVE_LEDGER_TITLE_RE.fullmatch(str(atom.get("text") or "").strip()) and _y_center(atom) < header_y
        ),
        None,
    )
    account_atoms = [
        (atom, match.group("account"))
        for atom in ordered
        if _y_center(atom) < header_y
        and (match := _NATIVE_ACCOUNT_RE.fullmatch(str(atom.get("text") or "").strip())) is not None
    ]
    if title is None or len(account_atoms) != 1:
        return None
    account = account_atoms[0][1]

    header_texts = [str(atom.get("text") or "").strip() for atom in ordered if _y_center(atom) < header_y]
    required_page_tokens = (f"第{page_number}", "/", str(page_count), "页")
    if any(token not in header_texts for token in required_page_tokens):
        return None

    footer_labels = [
        atom
        for atom in ordered
        if str(atom.get("text") or "").strip() in {"合计:", "合计："} and _y_center(atom) > header_y
    ]
    if len(footer_labels) != 1:
        return None
    footer_y = _y_center(footer_labels[0])
    footer_texts = [str(atom.get("text") or "").strip() for atom in ordered if _y_center(atom) > footer_y]
    if not all(
        any(text.startswith(prefix) for text in footer_texts) for prefix in ("打印操作员", "打印日期", "打印时间")
    ):
        return None

    body = [atom for atom in ordered if header_y + 2.0 < _y_center(atom) < footer_y - 2.0]
    date_atoms = [
        atom
        for atom in body
        if _NATIVE_DATETIME_RE.fullmatch(str(atom.get("text") or "").strip())
        and bounds[0] <= _x_center(atom) < bounds[1]
    ]
    time_atoms = [
        atom
        for atom in body
        if _NATIVE_TIME_RE.fullmatch(str(atom.get("text") or "").strip()) and bounds[0] <= _x_center(atom) < bounds[1]
    ]
    if not date_atoms or len(date_atoms) != len(time_atoms):
        return None

    # The date column is the page's structural row-boundary plane.  Preserve
    # its native layout cadence as part of coverage checking: debit/credit totals alone
    # cannot witness a missing zero-value transaction.  Real rows may grow by
    # one or two wrapped text lines, hence the deliberately broad multiples of
    # the minimum source pitch rather than a fixed case-specific row height.
    row_centers = sorted(_y_center(atom) for atom in date_atoms)
    if len(row_centers) > 1:
        row_gaps = [right - left for left, right in zip(row_centers, row_centers[1:])]
        source_pitch = min(row_gaps)
        if not 15.0 <= source_pitch <= 60.0 or any(
            gap > (2.25 * source_pitch) or abs((gap / source_pitch) - round(gap / source_pitch)) > 0.2
            for gap in row_gaps
        ):
            return None
        if row_centers[0] - header_y > 1.5 * source_pitch:
            return None
        # A final page can legitimately end well above its fixed footer band.
        # On a filled page, however, removing the terminal row creates a blank
        # band wider than any complete wrapped source-row slot.
        tail_gap = footer_y - row_centers[-1]
        if tail_gap <= 3.0 * source_pitch and tail_gap > 2.25 * source_pitch:
            return None

    money_atoms = [
        atom
        for atom in body
        if _MONEY_RE.fullmatch(str(atom.get("text") or "").strip()) and bounds[1] <= _x_center(atom) < bounds[4]
    ]
    unused_times = set(range(len(time_atoms)))
    unused_money = set(range(len(money_atoms)))
    debit_total = Decimal("0")
    credit_total = Decimal("0")

    for date_atom in sorted(date_atoms, key=lambda atom: (_y_center(atom), _x_center(atom))):
        raw_date = str(date_atom.get("text") or "").strip()
        try:
            calendar_date.fromisoformat(raw_date)
        except ValueError:
            return None
        date_y = _y_center(date_atom)
        matching_times = [
            index
            for index in unused_times
            if 6.0 <= _y_center(time_atoms[index]) - date_y <= 14.0
            and abs(_x_center(time_atoms[index]) - _x_center(date_atom)) <= 15.0
        ]
        if len(matching_times) != 1:
            return None
        time_index = matching_times[0]
        raw_time = str(time_atoms[time_index].get("text") or "").strip()
        try:
            int(raw_time[:2]) < 24 and int(raw_time[3:5]) < 60 and int(raw_time[6:8]) < 60
        except (TypeError, ValueError):
            return None
        if not (int(raw_time[:2]) < 24 and int(raw_time[3:5]) < 60 and int(raw_time[6:8]) < 60):
            return None
        unused_times.remove(time_index)

        row_money = [index for index in unused_money if abs(_y_center(money_atoms[index]) - date_y) <= 1.5]
        amount_indexes = [index for index in row_money if bounds[1] <= _x_center(money_atoms[index]) < bounds[3]]
        balance_indexes = [index for index in row_money if bounds[3] <= _x_center(money_atoms[index]) < bounds[4]]
        if len(amount_indexes) != 1 or len(balance_indexes) != 1:
            return None
        amount_index = amount_indexes[0]
        amount_atom = money_atoms[amount_index]
        try:
            amount = Decimal(str(amount_atom.get("text") or "").replace(",", ""))
            Decimal(str(money_atoms[balance_indexes[0]].get("text") or "").replace(",", ""))
        except InvalidOperation:
            return None
        if bounds[1] <= _x_center(amount_atom) < bounds[2]:
            debit_total += amount
        elif bounds[2] <= _x_center(amount_atom) < bounds[3]:
            credit_total += amount
        else:
            return None
        unused_money.difference_update({amount_index, balance_indexes[0]})

    if unused_times or unused_money:
        return None

    footer_money = [
        atom
        for atom in ordered
        if abs(_y_center(atom) - footer_y) <= 1.5
        and _MONEY_RE.fullmatch(str(atom.get("text") or "").strip())
        and bounds[1] <= _x_center(atom) < bounds[3]
    ]
    debit_footer = [atom for atom in footer_money if bounds[1] <= _x_center(atom) < bounds[2]]
    credit_footer = [atom for atom in footer_money if bounds[2] <= _x_center(atom) < bounds[3]]
    if len(debit_footer) != 1 or len(credit_footer) != 1:
        return None
    try:
        expected_debit = Decimal(str(debit_footer[0].get("text") or "").replace(",", ""))
        expected_credit = Decimal(str(credit_footer[0].get("text") or "").replace(",", ""))
    except InvalidOperation:
        return None
    if debit_total != expected_debit or credit_total != expected_credit:
        return None
    return len(date_atoms), account


def _bounded_cib_native_document_coverage(
    grouped: dict[str, list[dict[str, Any]]],
    page_ids: list[str],
) -> tuple[int, str] | None:
    """Count internally consistent signed-ledger rows in the native source plane.

    The exact title/header signature selects the layout, while structural quality
    is checked on every observed page: each has the same account and footer,
    and every bounded transaction anchor owns one posting date, signed amount,
    and balance.  The balance chain must close across pages.  This cannot prove
    that a terminal row absent from the source plane was never omitted upstream.
    """
    expected_account = ""
    ledger_rows: list[tuple[Decimal, Decimal]] = []
    page_count = len(page_ids)
    for page_number, page_id in enumerate(page_ids, start=1):
        ordered = sorted(
            grouped.get(page_id, []),
            key=lambda atom: (
                float(atom["bbox"][1]),
                float(atom["bbox"][0]),
                str(atom.get("id") or ""),
            ),
        )
        title = _first_exact(ordered, _CIB_NATIVE_LEDGER_TITLE)
        header_atoms = [_first_exact(ordered, label) for label in _CIB_NATIVE_LEDGER_HEADERS]
        if title is None or any(atom is None for atom in header_atoms):
            return None
        headers = [atom for atom in header_atoms if atom is not None]
        header_y = median(_y_center(atom) for atom in headers)
        if _y_center(title) >= header_y or any(abs(_y_center(atom) - header_y) > 1.5 for atom in headers):
            return None
        header_centers = [_x_center(atom) for atom in headers]
        if any(right <= left for left, right in zip(header_centers, header_centers[1:])):
            return None

        money_header_atoms = [
            atom
            for atom in ordered
            if _y_center(atom) < header_y + 1.5
            and "".join(str(atom.get("text") or "").split())
            in {
                _CIB_NATIVE_COMPOUND_MONEY_HEADER,
                "支/收交易金额",
                "账户余额",
                "交易地点",
            }
        ]
        compact_money_headers = {"".join(str(atom.get("text") or "").split()) for atom in money_header_atoms}
        if not (
            _CIB_NATIVE_COMPOUND_MONEY_HEADER in compact_money_headers
            or {"支/收交易金额", "账户余额", "交易地点"}.issubset(compact_money_headers)
        ):
            return None

        account_facts = {
            match.group("account")
            for atom in ordered
            if _y_center(atom) < header_y
            and (match := _CIB_NATIVE_ACCOUNT_RE.fullmatch(str(atom.get("text") or "").strip())) is not None
        }
        if len(account_facts) != 1:
            return None
        account = account_facts.pop()
        if expected_account and account != expected_account:
            return None
        expected_account = account

        page_marker = f"第{page_number}页/共{page_count}页"
        marker_atoms = [atom for atom in ordered if str(atom.get("text") or "").strip() == page_marker]
        footer_atoms = [
            atom for atom in ordered if str(atom.get("text") or "").strip().startswith(_CIB_NATIVE_FOOTER_PREFIX)
        ]
        if len(marker_atoms) != 1 or len(footer_atoms) != 1:
            return None
        footer_y = _y_center(footer_atoms[0])
        if footer_y <= header_y or _y_center(marker_atoms[0]) <= footer_y:
            return None

        transaction_header, posting_header, summary_header, party_header, _counter_header = headers
        body = [atom for atom in ordered if header_y + 2.0 < _y_center(atom) < footer_y - 2.0]
        transaction_dates = [
            atom
            for atom in body
            if _NATIVE_DATETIME_RE.fullmatch(str(atom.get("text") or "").strip())
            and float(transaction_header["bbox"][0]) - 2.0
            <= _x_center(atom)
            <= float(transaction_header["bbox"][2]) + 10.0
        ]
        posting_dates = [
            atom
            for atom in body
            if _NATIVE_DATETIME_RE.fullmatch(str(atom.get("text") or "").strip())
            and float(posting_header["bbox"][0]) - 2.0 <= _x_center(atom) <= float(posting_header["bbox"][2]) + 10.0
        ]
        money_atoms = [
            atom
            for atom in body
            if _MONEY_RE.fullmatch(str(atom.get("text") or "").strip())
            and float(summary_header["bbox"][2]) < _x_center(atom) < float(party_header["bbox"][0])
        ]
        if not transaction_dates or not (
            len(transaction_dates) == len(posting_dates) and len(money_atoms) == 2 * len(transaction_dates)
        ):
            return None
        ordered_row_centers = sorted(_y_center(atom) for atom in transaction_dates)
        if (
            not 15.0 <= ordered_row_centers[0] - header_y <= 55.0
            or not 15.0 <= footer_y - ordered_row_centers[-1] <= 55.0
            or any(
                not 20.0 <= right - left <= 55.0 for left, right in zip(ordered_row_centers, ordered_row_centers[1:])
            )
        ):
            return None

        unused_postings = set(range(len(posting_dates)))
        unused_money = set(range(len(money_atoms)))
        for date_atom in sorted(transaction_dates, key=lambda atom: (_y_center(atom), _x_center(atom))):
            raw_date = str(date_atom.get("text") or "").strip()
            try:
                calendar_date.fromisoformat(raw_date)
            except ValueError:
                return None
            row_y = _y_center(date_atom)
            posting_matches = [
                index for index in unused_postings if abs(_y_center(posting_dates[index]) - row_y) <= 1.5
            ]
            row_money = [index for index in unused_money if abs(_y_center(money_atoms[index]) - row_y) <= 1.5]
            if len(posting_matches) != 1 or len(row_money) != 2:
                return None
            posting_index = posting_matches[0]
            try:
                calendar_date.fromisoformat(str(posting_dates[posting_index].get("text") or "").strip())
            except ValueError:
                return None
            ordered_money = sorted(row_money, key=lambda index: _x_center(money_atoms[index]))
            try:
                amount = Decimal(str(money_atoms[ordered_money[0]].get("text") or "").replace(",", ""))
                balance = Decimal(str(money_atoms[ordered_money[1]].get("text") or "").replace(",", ""))
            except InvalidOperation:
                return None
            if amount == 0:
                return None
            ledger_rows.append((amount, balance))
            unused_postings.remove(posting_index)
            unused_money.difference_update(row_money)
        if unused_postings or unused_money:
            return None

    if not expected_account or not ledger_rows:
        return None
    if any(
        abs((previous_balance + amount) - balance) > Decimal("0.01")
        for previous_balance, (amount, balance) in zip((row[1] for row in ledger_rows), ledger_rows[1:])
    ):
        return None
    return len(ledger_rows), expected_account


def recover_positioned_record_block_bank_tables(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> PositionedBlockRecovery:
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
    for page_id, atoms in sorted(_positioned_atoms_by_page(parse_result, source_route=source_route).items()):
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


def recover_evidence_atom_bank_tables(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> list[list[list[str]]]:
    """Return one canonical table when issuer headers and column geometry agree."""
    atoms_by_page = _normalize_page_orientations(
        parse_result,
        _atoms_by_page(parse_result, source_route=source_route),
    )
    if not atoms_by_page:
        _store_recovery_cache(parse_result, [], [], 0, source_route=source_route)
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
        _store_geometry_recovery_cache(
            parse_result,
            geometry_fallback,
            geometry_sources,
            geometry_expected,
            source_route=source_route,
        )
        return geometry_fallback

    composite_left = float(composite_header["bbox"][0])
    composite_right = float(composite_header["bbox"][2])
    endpoints = _money_column_endpoints(all_atoms, composite_left, composite_right)
    if len(endpoints) != 3:
        _store_geometry_recovery_cache(
            parse_result,
            geometry_fallback,
            geometry_sources,
            geometry_expected,
            source_route=source_route,
        )
        return geometry_fallback
    expense_end, income_end, balance_end = endpoints

    anchors = {name: float(atom["bbox"][0]) for name, atom in headers.items() if atom is not None}
    if [anchors[name] for name in header_names] != sorted(anchors[name] for name in header_names):
        _store_geometry_recovery_cache(
            parse_result,
            geometry_fallback,
            geometry_sources,
            geometry_expected,
            source_route=source_route,
        )
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
        _store_recovery_cache(
            parse_result,
            tables,
            row_sources,
            expected_rows,
            source_route=source_route,
        )
        return tables
    _store_geometry_recovery_cache(
        parse_result,
        geometry_fallback,
        geometry_sources,
        geometry_expected,
        source_route=source_route,
    )
    return geometry_fallback


def recovered_evidence_atom_row_sources(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> list[dict[str, Any]]:
    """Return row provenance aligned with recovered evidence-atom table rows."""
    cache = _recovery_cache(parse_result, source_route=source_route)
    sources = cache.get("row_sources") if cache else None
    return deepcopy(sources) if isinstance(sources, list) else []


def recovered_evidence_atom_expected_row_count(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> int:
    """Return the independent positioned-date candidate count from recovery."""
    cache = _recovery_cache(parse_result, source_route=source_route)
    return int(cache.get("expected_row_count") or 0) if cache else 0


def recovered_evidence_atom_expected_row_evidence(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> tuple[int, str, float]:
    """Return count, source, and confidence for evidence-atom row anchors."""
    cache = _recovery_cache(parse_result, source_route=source_route)
    if not cache:
        return 0, "", 0.0
    return (
        int(cache.get("expected_row_count") or 0),
        str(cache.get("expected_row_source") or ""),
        float(cache.get("expected_row_confidence") or 0.0),
    )


def _recover_geometry_bank_tables(
    parse_result: Any,
    atoms_by_page: dict[str, list[dict[str, Any]]],
) -> tuple[list[list[list[str]]], list[dict[str, Any]], int]:
    """Rebuild borderless bank grids from header and row geometry."""
    tables: list[list[list[str]]] = []
    row_sources: list[dict[str, Any]] = []
    expected_rows = 0
    previous_balance: float | None = None
    carried_header: (
        tuple[list[dict[str, Any]], list[str], dict[str, int], list[float], dict[str, Any] | None] | None
    ) = None
    for page_id in sorted(atoms_by_page):
        atoms = atoms_by_page[page_id]
        header = _geometry_header(atoms)
        header_is_carried = header is None
        if header is not None:
            header_atoms, col_map = header
            header_cells = [str(atom.get("text") or "").strip() for atom in header_atoms]
            centers = [_x_center(atom) for atom in header_atoms]
            auxiliary_header = _geometry_auxiliary_header(atoms, header_atoms)
            carried_header = (header_atoms, header_cells, col_map, centers, auxiliary_header)
        elif carried_header is not None:
            header_atoms, header_cells, col_map, centers, auxiliary_header = carried_header
        else:
            continue
        bounds = [float("-inf"), *((left + right) / 2 for left, right in zip(centers, centers[1:])), float("inf")]
        geometry_atoms = _expand_geometry_data_atoms(atoms, bounds, centers, col_map)
        horizontal_rules = _page_horizontal_rules(
            parse_result,
            page_id,
            atoms,
            header_atoms,
        )
        date_idx = col_map.get("date", col_map.get("timestamp", col_map.get("posting_date")))
        if date_idx is None:
            continue
        if header_is_carried:
            header_bottom = _carried_geometry_header_bottom(geometry_atoms, bounds, col_map)
            if header_bottom is None:
                continue
        else:
            header_bottom = max(
                float(atom["bbox"][3]) for atom in [*header_atoms, *([auxiliary_header] if auxiliary_header else [])]
            )
        source_headers = [
            *header_cells,
            *([str(auxiliary_header.get("text") or "").strip()] if auxiliary_header else []),
        ]
        footer_atoms = [
            atom
            for atom in atoms
            if float(atom["bbox"][1]) > header_bottom and _is_geometry_footer_text(str(atom.get("text") or ""))
        ]
        first_footer_atom = min(footer_atoms, key=_y_center) if footer_atoms else None
        page_footer_y = _y_center(first_footer_atom) if first_footer_atom is not None else float("inf")
        typical_atom_height = median(
            [
                float(atom["bbox"][3]) - float(atom["bbox"][1])
                for atom in atoms
                if float(atom["bbox"][3]) > float(atom["bbox"][1])
            ]
            or [0.0]
        )
        first_footer_text = str(first_footer_atom.get("text") or "") if first_footer_atom is not None else ""
        footer_guard = (
            max(typical_atom_height * 1.5, 8.0)
            if "打印" in first_footer_text or re.search(r"第\s*\d+\s*页", first_footer_text)
            else 0.0
        )
        footer_content_bottom = page_footer_y - footer_guard if page_footer_y < float("inf") else page_footer_y
        table_bottom = max(
            (rule for rule in horizontal_rules if header_bottom < rule < footer_content_bottom),
            default=footer_content_bottom,
        )
        row_anchors, anchor_type = _geometry_row_anchors(
            geometry_atoms,
            bounds,
            col_map,
            header_bottom=header_bottom,
            table_bottom=table_bottom,
        )
        expected_rows += len(row_anchors)
        rows: list[list[str]] = []
        row_atom_groups: list[list[dict[str, Any]]] = []
        original_rows: list[list[str]] = []
        overlay_repairs_by_row: list[list[dict[str, str]]] = []
        boundary_atoms_owned_by_previous: set[int] = set()
        for idx, anchor in enumerate(row_anchors):
            anchor_y = _geometry_transaction_anchor_y(anchor, typical_atom_height)
            next_y = (
                _geometry_transaction_anchor_y(row_anchors[idx + 1], typical_atom_height)
                if idx + 1 < len(row_anchors)
                else float("inf")
            )
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
            anchors_between_rules = (
                sum(
                    previous_rule < _geometry_transaction_anchor_y(candidate, typical_atom_height) < next_rule
                    for candidate in row_anchors
                )
                if previous_rule is not None and next_rule is not None
                else 0
            )
            anchor_is_bracketed = (
                previous_rule is not None
                and next_rule is not None
                and float(anchor["bbox"][1]) > previous_rule
                and float(anchor["bbox"][3]) < next_rule
            )
            uses_row_rules = anchor_is_bracketed and anchors_between_rules == 1
            if uses_row_rules:
                # Native ledger rules are the strongest boundary evidence:
                # wrapped cells may begin above the date baseline or end well
                # below it, but cannot cross a full-width separator.  A page
                # frame enclosing several anchors is not a row separator and
                # must fall back to the date-anchor Voronoi boundaries below.
                row_top = previous_rule + 0.5
                row_bottom = next_rule - 0.5
            else:
                # Without vector rules, use Voronoi boundaries between date
                # anchors. Wrapped cells can start above their own date
                # baseline, so assigning everything to the preceding date
                # leaks the next transaction into the previous row.
                previous_y = (
                    _geometry_transaction_anchor_y(row_anchors[idx - 1], typical_atom_height) if idx > 0 else None
                )
                # A documented second-tier column is printed below its core
                # transaction baseline and therefore needs a small downward
                # bias.  Ordinary wrapped cells use the true nearest-anchor
                # midpoint; applying the auxiliary bias globally would pull
                # leading text from the next transaction into the prior row.
                voronoi_bias = max(typical_atom_height * 0.20, 1.0) if auxiliary_header else 0.0
                row_top = header_bottom if previous_y is None else (previous_y + anchor_y) / 2 + voronoi_bias
                if next_y != float("inf"):
                    row_bottom = (anchor_y + next_y) / 2 + voronoi_bias
                else:
                    row_bottom = footer_limit if footer_limit is not None else float("inf")
            if footer_limit is not None:
                row_bottom = min(row_bottom, footer_limit)
            semantic_voronoi_boundary = (
                not uses_row_rules
                and auxiliary_header is None
                and idx + 1 < len(row_anchors)
            )
            previous_owned_boundary_atoms = {
                id(atom)
                for atom in geometry_atoms
                if semantic_voronoi_boundary
                and _geometry_transaction_anchor_y(atom, typical_atom_height) == row_bottom
                and _boundary_account_fragment_owner(
                    atom,
                    geometry_atoms,
                    boundary_y=row_bottom,
                    previous_anchor_y=anchor_y,
                    next_anchor_y=next_y,
                    bounds=bounds,
                    col_map=col_map,
                    typical_atom_height=typical_atom_height,
                )
                == "previous"
            }
            boundary_atoms_owned_by_previous.update(previous_owned_boundary_atoms)
            row_atoms = [
                atom
                for atom in geometry_atoms
                if (
                    (
                        row_top < _geometry_transaction_anchor_y(atom, typical_atom_height) < row_bottom
                    )
                    or (
                        _geometry_transaction_anchor_y(atom, typical_atom_height) == row_top
                        and id(atom) not in boundary_atoms_owned_by_previous
                    )
                    or (
                        id(atom) in previous_owned_boundary_atoms
                        and _geometry_transaction_anchor_y(atom, typical_atom_height) == row_bottom
                    )
                )
                and _geometry_transaction_anchor_y(atom, typical_atom_height) > header_bottom
            ]
            auxiliary_atoms = _geometry_auxiliary_row_atoms(
                row_atoms,
                anchor_y=anchor_y,
                auxiliary_header=auxiliary_header,
                typical_atom_height=typical_atom_height,
            )
            auxiliary_atom_ids = {id(atom) for atom in auxiliary_atoms}
            core_row_atoms = [atom for atom in row_atoms if id(atom) not in auxiliary_atom_ids]
            row: list[str] = []
            for col_idx in range(len(header_cells)):
                selected = [atom for atom in core_row_atoms if bounds[col_idx] <= _x_center(atom) < bounds[col_idx + 1]]
                row.append(
                    _join_geometry_atoms(
                        selected,
                        line_tolerance=1.5,
                    )
                )
            if auxiliary_header is not None:
                row.append(_join_geometry_atoms(auxiliary_atoms, line_tolerance=1.5))
            date_match = _DATE_ANY_RE.search(row[date_idx])
            if date_match:
                sequence_idx = col_map.get("sequence_no")
                prefix = row[date_idx][: date_match.start()]
                if sequence_idx is not None and sequence_idx < len(row) and not row[sequence_idx] and prefix.isdigit():
                    row[sequence_idx] = prefix
                row[date_idx] = row[date_idx][date_match.start() :]
            overlay_repairs = _strip_geometry_page_header_overlay(row, col_map)
            original_row = list(row)
            _repair_geometry_cell_spill(row, col_map)
            if _geometry_row_is_transaction(row):
                rows.append(row)
                row_atom_groups.append(row_atoms)
                original_rows.append(original_row)
                overlay_repairs_by_row.append(overlay_repairs)
        if rows:
            previous_balance = _repair_geometry_rows(
                rows,
                col_map,
                source_headers=source_headers,
                previous_balance=previous_balance,
            )
            for row_atoms, original_row, row, overlay_repairs in zip(
                row_atom_groups,
                original_rows,
                rows,
                overlay_repairs_by_row,
            ):
                source = _row_source(
                    page_id,
                    row_atoms,
                    row,
                    source_headers=source_headers,
                    source_values=original_row,
                )
                source["row_anchor_type"] = anchor_type
                repairs = [
                    {
                        "field": source_headers[index],
                        "ocr_raw": original,
                        "reconstructed": repaired,
                    }
                    for index, (original, repaired) in enumerate(zip(original_row, row))
                    if original != repaired
                ]
                if repairs:
                    source["reconstruction_repairs"] = repairs
                if overlay_repairs:
                    source["source_overlay_repairs"] = overlay_repairs
                row_sources.append(source)
            tables.append([source_headers, *rows])
    return tables, row_sources, expected_rows


def _carried_geometry_header_bottom(
    atoms: list[dict[str, Any]],
    bounds: list[float],
    col_map: dict[str, int],
) -> float | None:
    """Return a safe top boundary when a prior-page ledger schema continues.

    A carried schema is only admitted when the new page independently exposes
    a transaction-shaped baseline in the same date, amount, and balance
    columns.  This lets issuer layouts omit repeated headers without turning a
    date and money mentioned in prose into a ledger row.
    """
    date_idx = col_map.get("date", col_map.get("timestamp", col_map.get("posting_date")))
    amount_idx = col_map.get("amount")
    balance_idx = col_map.get("balance")
    if date_idx is None or amount_idx is None or balance_idx is None:
        return None

    candidates = sorted(
        (
            atom
            for atom in atoms
            if bounds[date_idx] <= _x_center(atom) < bounds[date_idx + 1]
            and not _is_geometry_footer_text(str(atom.get("text") or ""))
            and _DATE_ANY_RE.search(str(atom.get("text") or "").strip())
        ),
        key=_y_center,
    )
    typical_height = median(
        [
            float(atom["bbox"][3]) - float(atom["bbox"][1])
            for atom in atoms
            if float(atom["bbox"][3]) > float(atom["bbox"][1])
        ]
        or [8.0]
    )
    baseline_tolerance = max(typical_height * 0.65, 3.0)
    for anchor in candidates:
        anchor_y = _y_center(anchor)
        baseline_atoms = [atom for atom in atoms if abs(_y_center(atom) - anchor_y) <= baseline_tolerance]
        row = [
            _join_geometry_atoms(
                [atom for atom in baseline_atoms if bounds[index] <= _x_center(atom) < bounds[index + 1]],
                line_tolerance=1.5,
            )
            for index in range(len(bounds) - 1)
        ]
        amount = row[amount_idx] if amount_idx < len(row) else ""
        balance = row[balance_idx] if balance_idx < len(row) else ""
        if not (_MONEY_ANY_RE.search(amount) and _MONEY_ANY_RE.search(balance)):
            continue
        candidate_row = list(row)
        _repair_geometry_cell_spill(candidate_row, col_map)
        if not _geometry_row_is_transaction(candidate_row):
            continue

        # Keep nearby wrapped text above the first date baseline, while
        # excluding distant page furniture such as ``Page 2 of 72``.
        row_cluster_top = float(anchor["bbox"][1])
        preceding_groups = [
            group
            for group in _baseline_groups(atoms)
            if max(float(item["bbox"][3]) for item in group) <= float(anchor["bbox"][1])
        ]
        for group in reversed(preceding_groups):
            group_bottom = max(float(item["bbox"][3]) for item in group)
            if row_cluster_top - group_bottom > max(typical_height * 1.8, 12.0):
                break
            if any(
                bounds[date_idx] <= _x_center(item) < bounds[date_idx + 1]
                and _DATE_ANY_RE.search(str(item.get("text") or ""))
                for item in group
            ):
                break
            if any(_is_geometry_footer_text(str(item.get("text") or "")) for item in group):
                break
            row_cluster_top = min(float(item["bbox"][1]) for item in group)
        return row_cluster_top - 0.5
    return None


def _geometry_auxiliary_header(
    atoms: list[dict[str, Any]],
    header_atoms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a bounded second-tier business column below a main ledger header."""
    markers = {
        "对方账号户名/附言",
        "对方账户户名/附言",
        "对方账号与户名/附言",
    }
    header_top = min(float(atom["bbox"][1]) for atom in header_atoms)
    header_bottom = max(float(atom["bbox"][3]) for atom in header_atoms)
    typical_height = median([float(atom["bbox"][3]) - float(atom["bbox"][1]) for atom in header_atoms] or [8.0])
    candidates = [
        atom
        for atom in atoms
        if re.sub(r"\s+", "", str(atom.get("text") or "")) in markers
        and header_top <= float(atom["bbox"][1]) <= header_bottom + max(typical_height * 2.5, 16.0)
    ]
    return min(candidates, key=lambda atom: float(atom["bbox"][1]), default=None)


def _geometry_auxiliary_row_atoms(
    row_atoms: list[dict[str, Any]],
    *,
    anchor_y: float,
    auxiliary_header: dict[str, Any] | None,
    typical_atom_height: float,
) -> list[dict[str, Any]]:
    """Return the same-row value band belonging to a second-tier header."""
    if auxiliary_header is None:
        return []
    x0, _, x1, _ = [float(value) for value in auxiliary_header["bbox"][:4]]
    groups = [
        group
        for group in _baseline_groups(row_atoms)
        if sum(_y_center(atom) for atom in group) / len(group) > anchor_y + max(typical_atom_height * 0.65, 3.0)
        and any(float(atom["bbox"][2]) >= x0 - 8.0 and float(atom["bbox"][0]) <= x1 + 8.0 for atom in group)
    ]
    if not groups:
        return []
    nearest = min(groups, key=lambda group: sum(_y_center(atom) for atom in group) / len(group))
    return [atom for atom in nearest if float(atom["bbox"][2]) >= x0 - 8.0 and float(atom["bbox"][0]) <= x1 + 120.0]


def _expand_geometry_data_atoms(
    atoms: list[dict[str, Any]],
    bounds: list[float],
    centers: list[float],
    col_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Split source-fragmented ledger values using the proven header geometry.

    Native PDF text can split one date vertically (``2023-03-`` + ``01``)
    or glue adjacent date/money cells into one horizontal atom.  The source
    header already proves the semantic columns, so virtual fragments may be
    placed at those column centers while retaining the original atom bbox and
    evidence ids.  No values are inferred or copied between rows.
    """
    coalesced = _coalesce_fragmented_geometry_dates(atoms, bounds, col_map)
    expanded: list[dict[str, Any]] = []
    for atom in coalesced:
        fragments = _split_glued_geometry_data_atom(atom, bounds, centers, col_map)
        expanded.extend(fragments or [atom])
    return expanded


def _coalesce_fragmented_geometry_dates(
    atoms: list[dict[str, Any]],
    bounds: list[float],
    col_map: dict[str, int],
) -> list[dict[str, Any]]:
    date_indexes = {index for key in ("date", "timestamp", "posting_date") if (index := col_map.get(key)) is not None}
    if not date_indexes:
        return list(atoms)

    typical_height = median([float(atom["bbox"][3]) - float(atom["bbox"][1]) for atom in atoms] or [8.0])
    ordered = sorted(enumerate(atoms), key=lambda item: (_y_center(item[1]), float(item[1]["bbox"][0])))
    used: set[int] = set()
    merged_by_index: dict[int, dict[str, Any]] = {}
    for ordered_pos, (first_index, first) in enumerate(ordered):
        if first_index in used:
            continue
        first_text = re.sub(r"\s+", "", str(first.get("text") or ""))
        if not re.match(r"^(?:19|20)\d{2}", first_text) or _DATE_ANY_RE.fullmatch(first_text):
            continue
        first_col = next(
            (index for index in date_indexes if bounds[index] <= _x_center(first) < bounds[index + 1]),
            None,
        )
        if first_col is None:
            continue
        first_x0 = float(first["bbox"][0])
        first_bottom = float(first["bbox"][3])
        for second_index, second in ordered[ordered_pos + 1 :]:
            if second_index in used:
                continue
            second_top = float(second["bbox"][1])
            gap = second_top - first_bottom
            if gap > max(typical_height * 0.8, 6.0):
                break
            if gap < -1.0 or abs(float(second["bbox"][0]) - first_x0) > 4.0:
                continue
            if not (bounds[first_col] <= _x_center(second) < bounds[first_col + 1]):
                continue
            second_text = re.sub(r"\s+", "", str(second.get("text") or ""))
            if not re.fullmatch(r"[\d./年月日-]{1,8}", second_text):
                continue
            combined = f"{first_text}{second_text}"
            if _DATE_ANY_RE.fullmatch(combined) is None or not _normalize_block_date(combined):
                continue
            virtual = dict(first)
            virtual["text"] = combined
            virtual["bbox"] = _union_bbox([first["bbox"], second["bbox"]])
            virtual["_source_bbox"] = _union_bbox(
                [
                    first.get("_source_bbox") or first["bbox"],
                    second.get("_source_bbox") or second["bbox"],
                ]
            )
            virtual["evidence_ids"] = list(
                dict.fromkeys(
                    str(value)
                    for source in (first, second)
                    for value in [source.get("id"), *(source.get("evidence_ids") or [])]
                    if str(value or "")
                )
            )
            # This bbox is the union of two source-visible date fragments, not
            # one unusually tall native glyph.  Mark it explicitly so row
            # anchoring can use the union midpoint without changing the
            # lower-baseline behavior required by genuine tall PDF atoms.
            virtual["_coalesced_fragmented_date_anchor"] = True
            merged_by_index[first_index] = virtual
            used.update({first_index, second_index})
            break

    return [
        merged_by_index[index] if index in merged_by_index else atom
        for index, atom in enumerate(atoms)
        if index not in used or index in merged_by_index
    ]


def _split_glued_geometry_data_atom(
    atom: dict[str, Any],
    bounds: list[float],
    centers: list[float],
    col_map: dict[str, int],
) -> list[dict[str, Any]]:
    text = re.sub(r"\s+", "", str(atom.get("text") or ""))
    if not text:
        return []

    date_idx = col_map.get("date", col_map.get("timestamp"))
    posting_idx = col_map.get("posting_date")
    summary_idx = next(
        (index for key in ("summary", "purpose") if (index := col_map.get(key)) is not None),
        None,
    )
    date_matches = list(_DATE_ANY_RE.finditer(text))
    if (
        date_idx is not None
        and posting_idx is not None
        and summary_idx is not None
        and len(date_matches) >= 2
        and date_matches[0].start() == 0
        and date_matches[1].start() == date_matches[0].end()
    ):
        residue = text[date_matches[1].end() :]
        if residue:
            return [
                _virtual_geometry_data_atom(atom, date_matches[0].group(0), centers[date_idx]),
                _virtual_geometry_data_atom(atom, date_matches[1].group(0), centers[posting_idx]),
                _virtual_geometry_data_atom(atom, residue, centers[summary_idx]),
            ]

    direction_idx = col_map.get("direction")
    if summary_idx is not None and direction_idx is not None and summary_idx < direction_idx:
        x0, _, x1, _ = [float(value) for value in atom["bbox"][:4]]
        marker = text[-1:] if text[-1:] in {"收", "支", "借", "贷"} else ""
        summary_text = text[:-1]
        summary_owns_left = bounds[summary_idx] <= x0 < bounds[summary_idx + 1]
        crosses_direction = float(atom["bbox"][2]) >= bounds[direction_idx]
        if marker and summary_text and summary_owns_left and crosses_direction:
            # A native word can straddle the proven summary/direction boundary
            # (for example ``快捷支付支``). Keep the exact compound in source
            # provenance, but expose its two source-visible business roles in
            # the canonical row rather than polluting the summary or losing
            # the dedicated direction.
            return [
                _virtual_geometry_data_atom(atom, summary_text, centers[summary_idx]),
                _virtual_geometry_data_atom(atom, marker, centers[direction_idx]),
            ]

    amount_idx = col_map.get("amount")
    balance_idx = col_map.get("balance")
    location_idx = next(
        (index for key in ("transaction_location", "institution") if (index := col_map.get(key)) is not None),
        None,
    )
    if amount_idx is None or balance_idx is None:
        return []
    money_matches = list(_MONEY_ANY_RE.finditer(text))
    if not money_matches or money_matches[0].start() != 0:
        return []
    x0, _, x1, _ = [float(value) for value in atom["bbox"][:4]]
    if (
        len(money_matches) >= 2
        and money_matches[1].start() == money_matches[0].end()
        and balance_idx == amount_idx + 1
        and x0 <= centers[amount_idx] + 12.0
        and x1 >= centers[balance_idx] - 12.0
    ):
        residue = text[money_matches[1].end() :]
        if residue and location_idx != balance_idx + 1:
            return []
        fragments = [
            _virtual_geometry_data_atom(atom, money_matches[0].group(0), centers[amount_idx]),
            _virtual_geometry_data_atom(atom, money_matches[1].group(0), centers[balance_idx]),
        ]
        if residue:
            fragments.append(_virtual_geometry_data_atom(atom, residue, centers[location_idx]))
        return fragments
    residue = text[money_matches[0].end() :]
    balance_owns_left = bounds[balance_idx] - 12.0 <= x0 < bounds[balance_idx + 1]
    if (
        residue
        and location_idx is not None
        and location_idx == balance_idx + 1
        and balance_owns_left
        and x1 >= centers[location_idx] - 12.0
    ):
        return [
            _virtual_geometry_data_atom(atom, money_matches[0].group(0), centers[balance_idx]),
            _virtual_geometry_data_atom(atom, residue, centers[location_idx]),
        ]
    return []


def _virtual_geometry_data_atom(
    atom: dict[str, Any],
    text: str,
    center: float,
) -> dict[str, Any]:
    virtual = dict(atom)
    virtual["text"] = text
    _, y0, _, y1 = [float(value) for value in atom["bbox"][:4]]
    virtual["bbox"] = [center - 0.5, y0, center + 0.5, y1]
    virtual["_source_bbox"] = list(atom.get("_source_bbox") or atom["bbox"])
    virtual["evidence_ids"] = list(
        dict.fromkeys(str(value) for value in [atom.get("id"), *(atom.get("evidence_ids") or [])] if str(value or ""))
    )
    return virtual


def _geometry_row_anchors(
    atoms: list[dict[str, Any]],
    bounds: list[float],
    col_map: dict[str, int],
    *,
    header_bottom: float,
    table_bottom: float,
) -> tuple[list[dict[str, Any]], str]:
    """Return reliable physical-row spines, preferring issuer sequence cells."""
    sequence_idx = col_map.get("sequence_no")
    if sequence_idx is not None:
        sequence_anchors = sorted(
            (
                atom
                for atom in atoms
                if _y_center(atom) > header_bottom
                and _y_center(atom) < table_bottom
                and bounds[sequence_idx] <= _x_center(atom) < bounds[sequence_idx + 1]
                and re.fullmatch(r"\d{1,6}", re.sub(r"\s+", "", str(atom.get("text") or "")))
            ),
            key=_y_center,
        )
        if _sequence_anchors_are_reliable(sequence_anchors):
            return sequence_anchors, "sequence"

    date_idx = col_map.get("date", col_map.get("timestamp", col_map.get("posting_date")))
    if date_idx is None:
        return [], ""
    return (
        sorted(
            (
                atom
                for atom in atoms
                if _y_center(atom) > header_bottom
                and _y_center(atom) < table_bottom
                and bounds[date_idx] <= _x_center(atom) < bounds[date_idx + 1]
                and not _is_geometry_footer_text(str(atom.get("text") or ""))
                and _DATE_ANY_RE.search(str(atom.get("text") or "").strip())
            ),
            key=_y_center,
        ),
        "date",
    )


def _sequence_anchors_are_reliable(anchors: list[dict[str, Any]]) -> bool:
    """Return whether positioned sequence cells form one physical ledger spine."""
    if len(anchors) < 2:
        return False
    values = [int(re.sub(r"\s+", "", str(atom.get("text") or ""))) for atom in anchors]
    if len(values) != len(set(values)):
        return False
    continuity = sum(current - previous == 1 for previous, current in zip(values, values[1:]))
    return continuity / max(len(values) - 1, 1) >= 0.8


def _repair_geometry_cell_spill(row: list[str], col_map: dict[str, int]) -> None:
    """Separate OCR atoms that combine balance, summary, or a malformed date."""
    date_idx = col_map.get("date", col_map.get("timestamp", col_map.get("posting_date")))
    if date_idx is not None and date_idx < len(row):
        row[date_idx] = _repair_malformed_date(row[date_idx])

    balance_idx = col_map.get("balance")
    summary_idx = next(
        (index for key in ("summary", "purpose") if (index := col_map.get(key)) is not None),
        None,
    )
    if balance_idx is None or summary_idx is None or max(balance_idx, summary_idx) >= len(row):
        return

    amount_idx = col_map.get("amount")
    if amount_idx is not None and amount_idx < len(row) and amount_idx == summary_idx + 1:
        summary_prefix, signed_money = _signed_money_summary_prefix(row[amount_idx])
        if summary_prefix and signed_money:
            # A short narrative atom can begin just over the summary/amount
            # Voronoi boundary and therefore join the following signed-money
            # atom (for example ``Apple`` + ``-19.00``).  Under an already
            # proven adjacent summary/amount schema, move only a bounded
            # alphabetic prefix back to the summary.  The pre-repair source
            # cell and evidence ids remain in ``source_raw``/repair provenance.
            row[summary_idx] = _merge_geometry_text(row[summary_idx], summary_prefix)
            row[amount_idx] = signed_money

    balance_money, balance_residue = _money_and_residue(row[balance_idx])
    summary_money, summary_residue = _money_and_residue(row[summary_idx])
    if balance_money:
        row[balance_idx] = balance_money
        if balance_residue:
            row[summary_idx] = _merge_geometry_text(
                _clean_geometry_residue(balance_residue, balance_money),
                row[summary_idx],
            )
    elif summary_money:
        row[balance_idx] = summary_money
        row[summary_idx] = _clean_geometry_residue(summary_residue, summary_money)
    _repair_geometry_summary_prefix(row, col_map, balance_idx, summary_idx)
    row[summary_idx] = _repair_common_bank_summary_ocr(row[summary_idx])


def _signed_money_summary_prefix(value: str) -> tuple[str, str]:
    """Split a bounded narrative prefix from one terminal explicitly signed amount."""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    match = re.fullmatch(
        r"(?P<prefix>[A-Za-z\u3400-\u9fff][A-Za-z\u3400-\u9fff._:/()（）-]{0,63})"
        r"(?P<money>[+-]\d[\d,]*\.\d{2})",
        compact,
    )
    if match is None:
        return "", ""
    prefix = match.group("prefix")
    if prefix.upper() in {"CNY", "RMB", "USD", "HKD", "EUR", "JPY"} or prefix in {"人民币", "美元"}:
        return "", ""
    return prefix, match.group("money")


def _strip_geometry_page_header_overlay(
    row: list[str],
    col_map: dict[str, int],
) -> list[dict[str, str]]:
    """Remove page-header furniture fused into a source-backed first row.

    Some native PDF writers expose a tall word/block spanning the page header
    and the first ledger row. Once an ordered ledger header has proven the
    column roles, a repeated field label within that cell is a safe boundary:
    only the value after the last label belongs to the transaction. Numeric
    date/amount/balance roles are additionally bounded by their final typed
    token. The source bbox/evidence ids remain attached to the reconstructed
    row, and the discarded compound is retained in repair provenance.
    """

    repairs: list[dict[str, str]] = []
    handled_indexes: set[int] = set()
    for role, labels in _GEOMETRY_OVERLAY_LABELS_BY_ROLE.items():
        index = col_map.get(role)
        if index is None or index >= len(row) or index in handled_indexes:
            continue
        original = str(row[index] or "")
        cleaned = _geometry_overlay_business_value(original, role=role, labels=labels)
        if cleaned == original or not cleaned:
            continue
        row[index] = cleaned
        handled_indexes.add(index)
        repairs.append({"field": role, "source_compound": original, "reconstructed": cleaned})
    return repairs


def _geometry_overlay_business_value(
    value: str,
    *,
    role: str,
    labels: tuple[str, ...],
) -> str:
    original = str(value or "")
    compact, source_ends = _compact_nfkc_with_source_ends(original)
    if not compact:
        return original

    normalized_labels = tuple(
        sorted(
            {re.sub(r"\s+", "", unicodedata.normalize("NFKC", label)) for label in labels},
            key=len,
            reverse=True,
        )
    )
    marker_matches = [
        (position, label)
        for label in normalized_labels
        if (position := compact.rfind(label)) >= 0
    ]
    marker_end = max((position + len(label) for position, label in marker_matches), default=-1)

    if role in {"date", "timestamp", "posting_date"}:
        matches = list(_DATE_ANY_RE.finditer(compact))
        if not matches or (len(matches) == 1 and marker_end < 0):
            return original
        match = matches[-1]
        end = match.end()
        time_match = re.match(r"(?:[T ])?(\d{1,2}:\d{2}(?::\d{2})?)", compact[end:])
        if time_match:
            end += time_match.end()
        return _source_compact_slice(original, source_ends, match.start(), end)

    if role in {"amount", "balance"}:
        matches = list(_MONEY_ANY_RE.finditer(compact))
        if not matches:
            return original
        match = matches[-1]
        prefix_is_overlay = match.start() > 0 and (
            marker_end >= 0
            or bool(_DATE_ANY_RE.search(compact[: match.start()]))
            or any(marker in compact[: match.start()] for marker in ("账单", "流水", "账户"))
        )
        if not prefix_is_overlay:
            return original
        return _source_compact_slice(original, source_ends, match.start(), match.end())

    if role == "direction":
        direction_match = re.search(r"(?:收入|支出|借|贷|收|支)$", compact)
        if direction_match is None:
            return original
        prefix = compact[: direction_match.start()]
        if marker_end < 0 and not re.search(r"\d{1,2}:\d{2}(?::\d{2})?", prefix):
            return original
        return _source_compact_slice(original, source_ends, direction_match.start(), direction_match.end())

    if marker_end < 0 or marker_end >= len(compact):
        return original
    return _source_compact_slice(original, source_ends, marker_end, len(compact)).lstrip(":：")


def _compact_nfkc_with_source_ends(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    source_ends: list[int] = []
    for source_index, char in enumerate(str(value or "")):
        for normalized_char in unicodedata.normalize("NFKC", char):
            if normalized_char.isspace():
                continue
            chars.append(normalized_char)
            source_ends.append(source_index + 1)
    return "".join(chars), source_ends


def _source_compact_slice(
    original: str,
    source_ends: list[int],
    compact_start: int,
    compact_end: int,
) -> str:
    if compact_start < 0 or compact_end <= compact_start or compact_end > len(source_ends):
        return str(original or "")
    source_start = 0 if compact_start == 0 else source_ends[compact_start - 1]
    source_end = source_ends[compact_end - 1]
    return str(original or "")[source_start:source_end].strip()


def _repair_common_bank_summary_ocr(value: str) -> str:
    """Repair exact, stable OCR confusions in common interbank summary phrases."""
    text = str(value or "").strip()
    for malformed, corrected in _COMMON_BANK_SUMMARY_OCR_CORRECTIONS.items():
        text = text.replace(malformed, corrected)
    return text


def _repair_malformed_date(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    match = re.fullmatch(r"(20\d{2})([-/])(\d{1,2})\2{2}(\d{1,2})", text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(2)}{match.group(4)}"
    return text


def _money_and_residue(value: str) -> tuple[str, str]:
    text = re.sub(r"\s+", "", str(value or ""))
    match = _MONEY_ANY_RE.search(text)
    if match is None:
        return "", text
    money = _repair_malformed_money(match.group(0))
    return money, f"{text[: match.start()]}{text[match.end() :]}"


def _clean_geometry_residue(residue: str, money: str) -> str:
    """Remove a duplicated decimal digit before keeping source narrative text."""
    text = str(residue or "").strip()
    if len(text) >= 2 and text[0].isdigit() and re.match(r"[\u3400-\u9fff]", text[1:]):
        decimal_digits = re.sub(r"\D", "", money.rsplit(".", 1)[-1])
        if decimal_digits and text[0] == decimal_digits[-1]:
            text = text[1:]
    return text


def _merge_geometry_text(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left
    if left.endswith(right):
        return left
    if right.startswith(left):
        return right
    return f"{left}{right}"


def _repair_geometry_summary_prefix(
    row: list[str],
    col_map: dict[str, int],
    balance_idx: int,
    summary_idx: int,
) -> None:
    """Remove one OCR bleed digit before a strongly semantic summary phrase."""
    sequence_idx = col_map.get("sequence_no")
    if sequence_idx is None or max(sequence_idx, balance_idx, summary_idx) >= len(row):
        return
    if not re.fullmatch(r"\d{1,6}", str(row[sequence_idx] or "").strip()):
        return
    summary = str(row[summary_idx] or "").strip()
    match = re.fullmatch(r"(\d)([\u3400-\u9fff].+)", summary)
    if match is None:
        return
    balance_digits = re.sub(r"\D", "", str(row[balance_idx] or ""))
    semantic_summary = match.group(2).startswith(
        ("电子银行", "网银", "城商行", "普通汇兑", "税费", "转账", "付息", "代扣", "现金")
    )
    if (balance_digits and match.group(1) == balance_digits[-1]) or semantic_summary:
        row[summary_idx] = match.group(2)


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
        "余额摘要": ("余额", "摘要"),
        "序号交易日期": ("序号", "交易日期"),
        "序号交易日期交易类型": ("序号", "交易日期", "交易类型"),
        "收入/支出交易金额": ("收入/支出", "交易金额"),
        "收入支出交易金额": ("收入/支出", "交易金额"),
        "支/收交易金额": ("支/收", "交易金额"),
        "支收交易金额": ("支/收", "交易金额"),
        "交易金额账户余额": ("交易金额", "账户余额"),
        "账户余额交易地点": ("账户余额", "交易地点"),
        "交易金额账户余额交易地点": ("交易金额", "账户余额", "交易地点"),
        "对方户名对方账户/对方银行": ("对方户名", "对方账户/对方银行"),
        "对方户名对方账号/对方银行": ("对方户名", "对方账号/对方银行"),
        "对方名称对方账户/对方银行": ("对方名称", "对方账户/对方银行"),
        "对方名称对方账号/对方银行": ("对方名称", "对方账号/对方银行"),
        "交易类型收入/支出交易金额": ("交易类型", "收入/支出", "交易金额"),
        "支/收交易金额账户余额交易地点": ("支/收", "交易金额", "账户余额", "交易地点"),
        "支收交易金额账户余额交易地点": ("支/收", "交易金额", "账户余额", "交易地点"),
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
        normalized_text = re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", str(atom.get("text") or "")),
        )
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
        # Some native PDFs emit one *horizontal* business row as a tall
        # bounding box containing ``transaction date + posting date +
        # summary``.  Splitting that box vertically destroys the proven
        # column position: both date fragments retain the source-wide x
        # span and can no longer act as the transaction spine.  Leave this
        # source-backed compound intact so the header-aware geometry pass
        # can place its three fragments at their semantic column centers.
        if (
            date_matches[0].start() == 0
            and date_matches[1].start() == date_matches[0].end()
            and text[date_matches[1].end() :].strip()
        ):
            return []
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
    header = _geometry_header(atoms)
    if header is None:
        return -1.0
    header_atoms, col_map = header
    centers = [_x_center(atom) for atom in header_atoms]
    bounds = [float("-inf"), *((left + right) / 2 for left, right in zip(centers, centers[1:])), float("inf")]
    header_bottom = max(float(atom["bbox"][3]) for atom in header_atoms)
    row_anchors, anchor_type = _geometry_row_anchors(
        atoms,
        bounds,
        col_map,
        header_bottom=header_bottom,
        table_bottom=float("inf"),
    )
    spread = max(centers, default=0.0) - min(centers, default=0.0)
    return (
        len(col_map) * 10.0
        + min(spread / 20.0, 20.0)
        + min(len(row_anchors), 20) * 5.0
        + (10.0 if anchor_type == "sequence" else 0.0)
    )


def _row_source(
    page_id: str,
    atoms: list[dict[str, Any]],
    row: list[str],
    *,
    source_headers: list[str] | None = None,
    source_values: list[str] | None = None,
) -> dict[str, Any]:
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
    source = {
        "source": "canonical_evidence_table",
        "page_id": page_id,
        **({"source_page": source_page} if source_page > 0 else {}),
        **({"bbox": bbox} if bbox else {}),
        **({"evidence_ids": evidence_ids} if evidence_ids else {}),
        "row_values": list(row),
    }
    raw_values = source_values if source_values is not None else row
    if (
        source_headers
        and len(source_headers) == len(raw_values)
        and all(str(header or "").strip() for header in source_headers)
        and len(set(source_headers)) == len(source_headers)
    ):
        # Preserve the physical source roles separately from parser keys.  A
        # bare ICBC ``账号`` is an own-account field and must never be relabelled
        # as ``对方账号`` in the lossless raw layer.
        source["source_raw"] = {
            str(header).strip(): str(value or "").strip() for header, value in zip(source_headers, raw_values)
        }
    return source


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
    *,
    expected_row_source: str = "positioned_date_anchors",
    expected_row_confidence: float = 0.80,
    source_route: str | None = None,
) -> None:
    domain = _domain_specific(parse_result)
    if domain is None:
        return
    domain[_recovery_cache_key(source_route)] = {
        "status": "ready",
        "table_count": len(tables),
        "row_count": sum(max(len(table) - 1, 0) for table in tables),
        "expected_row_count": max(int(expected_row_count or 0), 0),
        "expected_row_source": expected_row_source,
        "expected_row_confidence": max(0.0, min(float(expected_row_confidence), 1.0)),
        "row_sources": deepcopy(row_sources),
    }


def _store_geometry_recovery_cache(
    parse_result: Any,
    tables: list[list[list[str]]],
    row_sources: list[dict[str, Any]],
    expected_row_count: int,
    *,
    source_route: str | None = None,
) -> None:
    # Sequence-labelled geometry rows prove that this recovery is internally
    # ordered; they do not prove that the source did not contain a terminal
    # row that every retained geometry row missed.  Keep this candidate-local
    # signal at structural confidence and never relabel it as independent
    # page-transaction evidence.
    _store_recovery_cache(
        parse_result,
        tables,
        row_sources,
        expected_row_count,
        expected_row_source="positioned_date_anchors",
        expected_row_confidence=0.80,
        source_route=source_route,
    )


def _recovery_cache_key(source_route: str | None) -> str:
    route = str(source_route or "").strip().lower()
    return f"{_RECOVERY_CACHE_KEY}:{route}" if route in {"digital", "scanned"} else _RECOVERY_CACHE_KEY


def _recovery_cache(parse_result: Any, *, source_route: str | None = None) -> dict[str, Any]:
    domain = _domain_specific(parse_result)
    cache = domain.get(_recovery_cache_key(source_route)) if domain else None
    return cache if isinstance(cache, dict) and cache.get("status") == "ready" else {}


def _geometry_header(
    atoms: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    from docmirror.plugins._base.column_registry import ColumnMatcher
    from docmirror.plugins.bank_statement.community_plugin import BANK_COLUMN_REGISTRY
    from docmirror.plugins.bank_statement.header_resolve import (
        has_split_debit_credit_headers,
        normalize_header_cell,
        prefer_explicit_direction_column,
    )

    matcher = ColumnMatcher(BANK_COLUMN_REGISTRY)
    best: tuple[list[dict[str, Any]], dict[str, int]] | None = None
    for group in _baseline_groups(atoms):
        ordered = sorted(
            (atom for atom in group if not _is_geometry_transaction_value(str(atom.get("text") or ""))),
            key=lambda atom: float(atom["bbox"][0]),
        )
        if not ordered:
            continue
        cells = [normalize_header_cell(str(atom.get("text") or "")) for atom in ordered]
        joined = "".join(cells)
        if not any(marker in joined for marker in ("交易日期", "记账日期", "交易时间", "日期")):
            continue
        split_debit_credit = has_split_debit_credit_headers([[cells]])
        if not split_debit_credit and not any(
            marker in joined for marker in ("交易金额", "发生额", "支出金额", "收入金额")
        ):
            continue
        if "余额" not in joined:
            continue
        col_map = matcher.match(cells)
        col_map = prefer_explicit_direction_column(cells, col_map)
        fields = set(col_map)
        valid = (
            "balance" in fields
            and ("amount" in fields or split_debit_credit)
            and bool(fields.intersection({"date", "timestamp", "posting_date"}))
        )
        if valid and (best is None or len(col_map) > len(best[1])):
            expanded = _expand_staggered_header(atoms, ordered)
            expanded_cells = [normalize_header_cell(str(atom.get("text") or "")) for atom in expanded]
            expanded_map = matcher.match(expanded_cells)
            expanded_map = prefer_explicit_direction_column(expanded_cells, expanded_map)
            expanded_fields = set(expanded_map)
            expanded_split_debit_credit = has_split_debit_credit_headers([[expanded_cells]])
            expanded_valid = (
                "balance" in expanded_fields
                and ("amount" in expanded_fields or expanded_split_debit_credit)
                and bool(expanded_fields.intersection({"date", "timestamp", "posting_date"}))
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
    band = [
        atom
        for atom in atoms
        if baseline - 4.0 <= float(atom["bbox"][1]) <= baseline + 6.0
        and not _is_geometry_transaction_value(str(atom.get("text") or ""))
    ]
    direction = _staggered_direction_header_atom(atoms, baseline)
    if direction is not None:
        direction_ids = set(direction.get("evidence_ids") or [])
        band = [atom for atom in band if str(atom.get("id") or "") not in direction_ids]
        band.append(direction)
    return sorted(band, key=lambda atom: float(atom["bbox"][0]))


def _staggered_direction_header_atom(
    atoms: list[dict[str, Any]],
    baseline: float,
) -> dict[str, Any] | None:
    candidates = [atom for atom in atoms if baseline - 10.0 <= float(atom["bbox"][1]) <= baseline + 10.0]
    for upper in candidates:
        upper_text = re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", str(upper.get("text") or "")),
        )
        expected_lower = {"支/": ("收", "支/收"), "收/": ("支", "收/支")}.get(upper_text)
        if expected_lower is None:
            continue
        lower_text, combined = expected_lower
        for lower in candidates:
            normalized_lower = re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", str(lower.get("text") or "")),
            )
            if normalized_lower != lower_text or float(lower["bbox"][1]) <= float(upper["bbox"][1]):
                continue
            if abs(_x_center(lower) - _x_center(upper)) > 5.0:
                continue
            virtual = dict(upper)
            virtual["text"] = combined
            virtual["bbox"] = _union_bbox([upper["bbox"], lower["bbox"]])
            virtual["_source_bbox"] = _union_bbox(
                [
                    upper.get("_source_bbox") or upper["bbox"],
                    lower.get("_source_bbox") or lower["bbox"],
                ]
            )
            virtual["evidence_ids"] = list(
                dict.fromkeys(
                    str(value)
                    for source in (upper, lower)
                    for value in [source.get("id"), *(source.get("evidence_ids") or [])]
                    if str(value or "")
                )
            )
            return virtual
    return None


def _is_geometry_transaction_value(text: str) -> bool:
    """Return whether an overlapping atom is row data, not a column label."""
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    if _DATE_ANY_RE.fullmatch(compact) or _MONEY_ANY_RE.fullmatch(compact):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}[-/.]\d{1,2}[-/.]?", compact):
        return True
    date_matches = list(_DATE_ANY_RE.finditer(compact))
    if len(date_matches) >= 2 and date_matches[0].start() == 0 and date_matches[1].start() == date_matches[0].end():
        return True
    money_matches = list(_MONEY_ANY_RE.finditer(compact))
    if (
        len(money_matches) >= 2
        and money_matches[0].start() == 0
        and money_matches[1].start() == money_matches[0].end()
    ):
        # Tall first-row atoms can overlap the visual header band.  A glued
        # amount+balance (optionally followed by a location) is transaction
        # data, never an extra statement column label.
        return True
    return compact in {"收", "支", "借", "贷", "Cr", "Dr"}


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


def _join_geometry_atoms(atoms: list[dict[str, Any]], *, line_tolerance: float) -> str:
    """Join atoms in visual reading order without letting font baselines reorder characters."""
    lines = _baseline_groups(atoms, tolerance=line_tolerance)
    return "".join(
        str(atom.get("text") or "").strip()
        for line in lines
        for atom in sorted(line, key=lambda item: float(item["bbox"][0]))
    )


def _geometry_row_is_transaction(row: list[str]) -> bool:
    from docmirror.plugins.bank_statement.row_extract import row_has_transaction_data
    from docmirror.plugins.bank_statement.wide_table_recovery import is_footer_or_total_row

    return not is_footer_or_total_row(row) and row_has_transaction_data(row)


def _is_geometry_footer_text(text: str) -> bool:
    return any(marker in text for marker in _GEOMETRY_FOOTER_MARKERS) or bool(
        re.search(r"第\s*\d+\s*页\s*(?:[/／-]\s*)?共\s*\d+\s*页", text)
    )


def _repair_geometry_rows(
    rows: list[list[str]],
    col_map: dict[str, int],
    *,
    source_headers: list[str] | None = None,
    previous_balance: float | None = None,
) -> float | None:
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
    if sequence_idx is not None and source_headers is not None and sequence_idx < len(source_headers):
        _repair_sequence_values(
            rows,
            sequence_idx,
            source_header=source_headers[sequence_idx],
        )
    if amount_idx is None or balance_idx is None:
        return previous_balance

    for row in rows:
        required_indexes = [amount_idx, balance_idx, *([direction_idx] if direction_idx is not None else [])]
        if max(required_indexes) >= len(row):
            continue
        direction_text = str(row[direction_idx] or "") if direction_idx is not None else ""
        if direction_idx is not None and any(marker in direction_text for marker in ("收入", "收人", "转入", "贷")):
            direction, inferred_direction = "收入", False
        elif direction_idx is not None and any(
            marker in direction_text for marker in ("支出", "支山", "攴出", "转出", "借")
        ):
            direction, inferred_direction = "支出", False
        else:
            inferred_direction = True
            context = "".join(row[index] for index in summary_indexes if index < len(row))
            if any(marker in context for marker in ("转入", "收入", "入账", "利息")):
                direction = "收入"
            elif any(marker in context for marker in ("转出", "支出", "出账", "支取", "消费", "付款")):
                direction = "支出"
            else:
                direction = ""
        amount = _money_float(row[amount_idx])
        balance = _money_float(row[balance_idx])
        balance_direction = ""
        if previous_balance is not None and amount is not None and balance is not None:
            absolute_amount = abs(amount)
            income_error = abs(previous_balance + absolute_amount - balance)
            expense_error = abs(previous_balance - absolute_amount - balance)
            if min(income_error, expense_error) <= 0.05 < max(income_error, expense_error):
                balance_direction = "收入" if income_error < expense_error else "支出"
        if balance_direction and (not direction or balance_direction != direction):
            direction = balance_direction
            inferred_direction = True
        if direction_idx is not None and direction and inferred_direction:
            row[direction_idx] = direction
        elif direction_idx is None and balance_direction and amount is not None:
            amount_text = str(row[amount_idx] or "").strip()
            if balance_direction == "支出" and amount > 0 and not amount_text.startswith("-"):
                row[amount_idx] = f"-{amount_text.lstrip('+')}"
            elif balance_direction == "收入" and amount < 0:
                row[amount_idx] = amount_text.lstrip("-")
        if balance is not None:
            previous_balance = balance
    return previous_balance


def _is_ordinal_sequence_header(source_header: str) -> bool:
    """Return whether a source column is a repairable row ordinal."""

    compact = re.sub(
        r"[\s.:：]+",
        "",
        unicodedata.normalize("NFKC", str(source_header or "")),
    ).casefold()
    return compact in {"序号", "交易序号", "no", "sequence", "序号/no", "no/序号"}


def _nearest_sequence_anchor(
    texts: list[str],
    values: list[int | None],
    *,
    start: int,
    step: int,
) -> int | None:
    """Find an ordinal anchor without crossing a non-empty opaque value."""

    position = start + step
    while 0 <= position < len(values):
        if values[position] is not None:
            return position
        if texts[position]:
            return None
        position += step
    return None


def _repair_sequence_values(
    rows: list[list[str]],
    sequence_idx: int,
    *,
    source_header: str,
) -> None:
    # ``sequence_no`` is a shared canonical field, but issuer columns such as
    # 日志号 and 流水号 contain opaque identifiers rather than row ordinals.
    # Only an explicit ordinal header permits interpolation, and even then a
    # non-empty source value is immutable.
    if not _is_ordinal_sequence_header(source_header):
        return

    texts: list[str] = []
    values: list[int | None] = []
    for row in rows:
        text = str(row[sequence_idx] or "").strip() if sequence_idx < len(row) else ""
        texts.append(text)
        values.append(int(text) if re.fullmatch(r"\d{1,6}", text) else None)
    for index, value in enumerate(values):
        if value is not None or texts[index]:
            continue
        previous = _nearest_sequence_anchor(texts, values, start=index, step=-1)
        following = _nearest_sequence_anchor(texts, values, start=index, step=1)
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


def _geometry_transaction_anchor_y(atom: dict[str, Any], typical_height: float) -> float:
    """Return the visual baseline for a date spine with a possibly tall PDF bbox."""
    top = float(atom["bbox"][1])
    bottom = float(atom["bbox"][3])
    if atom.get("_coalesced_fragmented_date_anchor") is True:
        return (top + bottom) / 2
    height = bottom - top
    if typical_height > 0 and height > max(typical_height * 1.8, 16.0):
        return bottom - typical_height / 2
    return (top + bottom) / 2


def _boundary_account_fragment_owner(
    atom: dict[str, Any],
    geometry_atoms: list[dict[str, Any]],
    *,
    boundary_y: float,
    previous_anchor_y: float,
    next_anchor_y: float,
    bounds: list[float],
    col_map: dict[str, int],
    typical_atom_height: float,
) -> str:
    """Resolve an exact Voronoi tie from adjacent account/bank text."""
    account_idx = col_map.get("counter_account")
    if account_idx is None or account_idx + 1 >= len(bounds):
        return "default"
    if not (bounds[account_idx] <= _x_center(atom) < bounds[account_idx + 1]):
        return "default"

    candidate = unicodedata.normalize("NFKC", re.sub(r"\s+", "", str(atom.get("text") or "")))
    evidence_span = max(typical_atom_height * 2.0, 16.0)
    previous_atoms = [
        target
        for target in geometry_atoms
        if id(target) != id(atom)
        and bounds[account_idx] <= _x_center(target) < bounds[account_idx + 1]
        and previous_anchor_y - evidence_span
        <= _geometry_transaction_anchor_y(target, typical_atom_height)
        < boundary_y
    ]
    target_atoms = [
        target
        for target in geometry_atoms
        if id(target) != id(atom)
        and bounds[account_idx] <= _x_center(target) < bounds[account_idx + 1]
        and boundary_y
        < _geometry_transaction_anchor_y(target, typical_atom_height)
        <= next_anchor_y + evidence_span
    ]
    previous_text = unicodedata.normalize(
        "NFKC",
        re.sub(r"\s+", "", _join_geometry_atoms(previous_atoms, line_tolerance=1.5)),
    )
    target_text = unicodedata.normalize(
        "NFKC",
        re.sub(r"\s+", "", _join_geometry_atoms(target_atoms, line_tolerance=1.5)),
    )
    if (
        re.fullmatch(r"[\u3400-\u9fff]{1,4}", candidate)
        and _is_account_bank_compound(previous_text + candidate)
        and not _is_account_bank_compound(previous_text)
    ):
        return "previous"
    if (
        re.fullmatch(r"[0-9A-Za-z*._-]{6,64}", candidate)
        and re.search(r"\d", candidate)
        and _is_account_bank_compound(candidate + target_text)
        and not _is_account_bank_compound(target_text)
    ):
        return "next"
    return "default"


def _is_account_bank_compound(value: str) -> bool:
    match = re.fullmatch(r"(?P<account>[0-9A-Za-z*._-]{6,64})(?P<bank>[\u3400-\u9fff].+)", value)
    if match is None or re.search(r"\d", match.group("account")) is None:
        return False
    return match.group("bank").endswith(
        ("银行", "支行", "分行", "营业部", "信用社", "合作社", "财务公司", "有限公司", "有限责任公司")
    )


def _atoms_by_page(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from docmirror.plugins._runtime.evidence_access import text_atoms

    page_modes = _evidence_page_modes(parse_result)
    atoms = [
        atom
        for atom in text_atoms(parse_result)
        if _atom_matches_source_route(
            atom,
            source_route,
            page_mode=page_modes.get(str(atom.get("page_id") or "")),
        )
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        page_id = str(atom.get("page_id") or "")
        bbox = atom.get("bbox")
        text = str(atom.get("text") or "").strip()
        if page_id and text and isinstance(bbox, list) and len(bbox) >= 4:
            grouped[page_id].append(atom)
    fallback_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in _page_text_atoms(parse_result, source_route=source_route):
        fallback_by_page[str(atom.get("page_id") or "")].append(atom)
    for page_id, fallback_atoms in fallback_by_page.items():
        if page_id and page_id not in grouped:
            grouped[page_id].extend(fallback_atoms)
    return dict(grouped)


def _positioned_atoms_by_page(
    parse_result: Any,
    *,
    source_route: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Prefer page text blocks when recovering one-block-per-record layouts.

    The evidence plane often tokenizes a native PDF into individual visual text
    atoms.  That is ideal for grid geometry, but it loses the record boundary
    retained by ``PageContent.texts``.  This recovery path specifically needs
    that boundary, so it prefers the page blocks while preserving the evidence
    atom path when page blocks are unavailable.
    """
    page_atoms = _page_text_atoms(parse_result, source_route=source_route)
    if page_atoms:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for atom in page_atoms:
            grouped[str(atom["page_id"])].append(atom)
        for page_id, atoms in _atoms_by_page(parse_result, source_route=source_route).items():
            if page_id not in grouped:
                grouped[page_id].extend(atoms)
        return dict(grouped)
    return _atoms_by_page(parse_result, source_route=source_route)


def _page_text_atoms(parse_result: Any, *, source_route: str | None = None) -> list[dict[str, Any]]:
    """Adapt positioned OCR text blocks when the sealed evidence plane has no atoms.

    Bank projection runs before every parser configuration has promoted page OCR
    blocks into ``evidence_plane.text_atoms``. Page text blocks are the same
    canonical extraction facts and retain their physical-page bboxes.
    """
    atoms: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", []) or []:
        if not _page_matches_source_route(getattr(page, "page_mode", None), source_route):
            continue
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
                    "source_kind": ("page_ocr_text" if source_route == "scanned" else "parse_result_text"),
                }
            )
    return atoms


def _atom_matches_source_route(
    atom: dict[str, Any],
    source_route: str | None,
    *,
    page_mode: Any = None,
) -> bool:
    if source_route not in {"digital", "scanned"}:
        return True
    source_kind = str(atom.get("source_kind") or "").lower()
    is_ocr = "ocr" in source_kind or "image" in source_kind or source_kind.endswith(("_token", "_line"))
    is_generic_parse_result = not source_kind or source_kind.startswith("parse_result_")
    page_mode_is_known = str(page_mode or "").strip().lower() not in {"", "unknown"}
    if page_mode_is_known:
        page_matches = _page_matches_source_route(page_mode, source_route)
        if not page_matches:
            return False
        # Generic ParseResult atom labels describe the canonical container, not
        # necessarily the acquisition source. A scanned page may therefore use
        # generic canonical atoms, but an explicitly native atom still belongs
        # exclusively to the digital route.
        if source_route == "scanned":
            return is_ocr or is_generic_parse_result
        return not is_ocr
    if source_route == "scanned":
        return is_ocr
    return not is_ocr


def _evidence_page_modes(parse_result: Any) -> dict[str, str]:
    plane = getattr(parse_result, "evidence_plane", None)
    modes = {
        str(getattr(page, "page_id", "") or ""): str(getattr(page, "content_mode", "") or "")
        for page in (getattr(plane, "pages", None) or [])
        if str(getattr(page, "page_id", "") or "")
    }
    for page in getattr(parse_result, "pages", []) or []:
        page_number = int(getattr(page, "page_number", 1) or 1)
        page_id = f"page:{page_number:04d}"
        modes.setdefault(page_id, str(getattr(page, "page_mode", "") or ""))
    return modes


def _page_matches_source_route(page_mode: Any, source_route: str | None) -> bool:
    if source_route not in {"digital", "scanned"}:
        return True
    mode = str(page_mode or "").strip().lower()
    if not mode or mode == "unknown":
        # Older cached ParseResults may omit page_mode. The document-level
        # extraction route remains authoritative when no contradictory
        # page-level provenance exists.
        return True
    scanned = any(marker in mode for marker in ("scan", "ocr", "image"))
    return scanned if source_route == "scanned" else not scanned


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
    selected = [atom for atom in atoms if left <= float(atom["bbox"][0]) < right]
    return _join_geometry_atoms(selected, line_tolerance=1.5)


__all__ = [
    "PositionedBlockRecovery",
    "recover_evidence_atom_bank_tables",
    "recover_positioned_record_block_bank_tables",
    "recovered_evidence_atom_expected_row_count",
    "recovered_evidence_atom_expected_row_evidence",
    "recovered_evidence_atom_row_sources",
    "recovered_native_datetime_row_evidence",
]
