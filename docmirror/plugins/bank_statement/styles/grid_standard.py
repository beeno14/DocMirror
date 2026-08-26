# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standard multi-column grid bank ledger style parser.

Default parser for clearly tabulated ledgers: strict header detection, split
debit/credit column merging, and per-row normalization through plugin column registry.

Pipeline role: primary and fallback parser in ``style_registry``; also used by
``signed_amount`` for shared row harvest paths.

Key exports: ``PARSER_ID``, ``STYLE_ID``, ``normalize_split_debit_credit``,
``extract_transactions``.

Dependencies: ``header_resolve``, ``row_extract``, ``institution``, ``standardizer``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from datetime import datetime
from typing import Any

from docmirror.plugins._base.standardizer import normalize_amount, normalize_timestamp
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.header_resolve import (
    detect_headers,
    has_split_debit_credit_headers,
    normalize_header_cell,
)
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
from docmirror.plugins.bank_statement.row_extract import (
    extract_all_tables,
    extract_logical_rows_with_provenance,
    extract_rows_from_header,
    row_has_transaction_data,
)
from docmirror.plugins.bank_statement.wide_table_recovery import is_footer_or_total_row

PARSER_ID = "grid_standard"
STYLE_ID = "grid_standard"

_SPLIT_DEBIT_KEYS = ("支出", "支出金额", "借方发生额", "借方", "转出金额")
_SPLIT_CREDIT_KEYS = ("收入", "收入金额", "贷方发生额", "贷方", "转入金额")
_DIRECTION_KEYS = (
    "收/支",
    "收支",
    "方向",
    "交易方向",
    "交易类别",
    "交易类型",
    "收入/支出",
    "月收/支",
    "借贷",
    "借/贷",
    "借贷标志",
    "Dc Flg",
)
_MONEY_PREFIX_RE = re.compile(r"^[^\d+-]*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)")
_COUNTERPARTY_KEYS = (
    "对方户名",
    "对方名称",
    "对手信息",
    "对手名称",
    "交易对方",
    "交易对手",
    "Counterparty Name",
    "counter_party",
)
_COUNTER_ACCOUNT_KEYS = ("对方账户", "对方账号", "counter_account")
_COUNTERPARTY_RECOVERY_BOUNDARY_MARKERS = (
    "序号交易日期",
    "对方账号",
    "对方账户",
    "对方户名",
    "清单支出算术合计",
    "清单收入算术合计",
    "打印渠道",
    "打印机构",
    "打印柜员",
    "打印时间",
    "本页支出算术合计",
    "本页收入算术合计",
    "交易提示",
    "CPKYG",
)


def _cell_value(raw_txn: dict[str, str], *needles: str) -> str:
    compact_needles = {
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(needle or ""))).lower() for needle in needles
    }
    for key, value in raw_txn.items():
        compact_key = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(key or ""))).lower()
        if compact_key in compact_needles:
            return str(value or "").strip()
    for key, value in raw_txn.items():
        norm_key = normalize_header_cell(key)
        for needle in needles:
            norm_needle = normalize_header_cell(needle)
            if norm_key == norm_needle or norm_needle in norm_key:
                return str(value or "").strip()
    return ""


def _registry_field_keys(plugin: Any, canonical_header: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable field aliases from the plugin registry plus parser fallbacks."""
    registry = getattr(plugin, "column_registry", None)
    mapping = registry.get(canonical_header) if isinstance(registry, dict) else None
    aliases = getattr(mapping, "aliases", ()) or ()
    return tuple(dict.fromkeys((canonical_header, *aliases, *fallback)))


def _explicit_source_column_value(raw_txn: dict[str, str], aliases: tuple[str, ...]) -> str:
    def compact(value: Any) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).lower()

    compact_aliases = {compact(alias) for alias in aliases}
    for raw_header, value in raw_txn.items():
        compact_header = compact(raw_header)
        header_parts = {compact(part) for part in str(raw_header or "").splitlines() if compact(part)}
        if compact_header in compact_aliases or compact_aliases.intersection(header_parts):
            return str(value or "").strip()
    return ""


_DATE_COLUMN_ALIASES = ("交易日期", "记账日", "记账日期", "日期", "Date")
_TIME_COLUMN_ALIASES = ("交易时间", "时间", "Time")


def _header_matches_aliases(header: str, aliases: tuple[str, ...]) -> bool:
    def compact(value: Any) -> str:
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).lower()

    compact_aliases = {compact(alias) for alias in aliases}
    compact_header = compact(header)
    header_parts = {compact(part) for part in str(header or "").splitlines() if compact(part)}
    return compact_header in compact_aliases or bool(compact_aliases.intersection(header_parts))


def _combine_separate_date_time(raw_txn: dict[str, str]) -> str:
    """Combine explicit bank date and time columns without guessing six-digit semantics."""
    date_value = _explicit_source_column_value(raw_txn, _DATE_COLUMN_ALIASES)
    time_value = _explicit_source_column_value(raw_txn, _TIME_COLUMN_ALIASES)
    if not date_value or not time_value:
        return ""

    normalized_date = normalize_timestamp(date_value)
    if not re.match(r"^\d{4}-\d{2}-\d{2}", normalized_date):
        return ""
    date_part = normalized_date[:10]
    compact_time = re.sub(r"\s+", "", unicodedata.normalize("NFKC", time_value))
    time_formats = (
        (r"\d{6}", "%H%M%S"),
        (r"\d{1,2}:\d{2}:\d{2}", "%H:%M:%S"),
        (r"\d{1,2}:\d{2}", "%H:%M"),
    )
    for pattern, time_format in time_formats:
        if not re.fullmatch(pattern, compact_time):
            continue
        try:
            parsed_time = datetime.strptime(compact_time, time_format).time()
            return datetime.combine(datetime.fromisoformat(date_part).date(), parsed_time).isoformat()
        except ValueError:
            return ""
    return ""


def _normalize_with_temporal_context(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any]:
    combined_timestamp = _combine_separate_date_time(raw_txn)
    if not combined_timestamp:
        return plugin._normalize(raw_txn)

    prepared: dict[str, str] = {"交易时间": combined_timestamp}
    prepared.update(
        (key, value) for key, value in raw_txn.items() if not _header_matches_aliases(key, _TIME_COLUMN_ALIASES)
    )
    return plugin._normalize(prepared)


def _explicit_amount_column(raw_txn: dict[str, str], aliases: tuple[str, ...]) -> tuple[bool, str, float | None]:
    """Return source-column presence, raw text, and parsed amount without defaulting blanks to zero."""
    for header, value in raw_txn.items():
        if not _header_matches_aliases(header, aliases):
            continue
        raw_value = str(value or "").strip()
        return True, raw_value, _normalize_monetary_cell(raw_value) if raw_value else None
    return False, "", None


def _normalize_source_counterparty_columns(
    raw_txn: dict[str, str],
    normalized: dict[str, Any],
) -> None:
    """Prefer explicit source columns over fuzzy base matches."""
    summary = _explicit_source_column_value(raw_txn, ("摘要描述", "交易摘要", "摘要"))
    if summary:
        normalized["summary"] = _clean_wrapped_text(summary)

    remark = _explicit_source_column_value(raw_txn, ("交易附言", "附言", "用途", "备注"))
    if remark and re.match(r"^(?:用途|附言)\s*[:：]", remark):
        normalized["purpose"] = _clean_wrapped_text(remark)

    counter_account = _explicit_source_column_value(raw_txn, _COUNTER_ACCOUNT_KEYS)
    if counter_account:
        normalized["counter_account"] = _clean_account(counter_account)

    counter_party = _explicit_source_column_value(
        raw_txn,
        (
            "对方户名",
            "对方名称",
            "对手信息",
            "对手名称",
            "交易对方",
            "Counterparty Name",
            "对方账号与户名",
        ),
    )
    if counter_party:
        cleaned_party = _clean_wrapped_text(counter_party)
        cleaned_party, embedded_account = _split_embedded_counter_account(cleaned_party)
        if embedded_account and not normalized.get("counter_account"):
            normalized["counter_account"] = embedded_account
        compact_party = re.sub(r"\s+", "", cleaned_party)
        if re.fullmatch(r"[0-9*＊]{6,32}", compact_party):
            normalized["counter_account"] = compact_party
            normalized["counter_party"] = ""
        elif compact_party in {"--", "-"}:
            normalized["counter_party"] = ""
        else:
            normalized["counter_party"] = cleaned_party

    counter_bank = _explicit_source_column_value(
        raw_txn,
        ("对方行名", "对手机构", "对方开户行", "对方银行名称", "Counterparty Institution"),
    )
    if counter_bank:
        normalized["counter_bank_name"] = _clean_wrapped_text(counter_bank)


def normalize_split_debit_credit(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any] | None:
    """Parse separate debit/credit columns into amount + direction."""
    income_column, _income_raw, income = _explicit_amount_column(raw_txn, _SPLIT_CREDIT_KEYS)
    expense_column, _expense_raw, expense = _explicit_amount_column(raw_txn, _SPLIT_DEBIT_KEYS)
    if not income_column and not expense_column:
        directed = _normalize_direction_amount(raw_txn, plugin)
        if directed is not None:
            return directed
        return _normalize_embedded_direction_amount(raw_txn, plugin)

    normalized = _normalize_with_temporal_context(raw_txn, plugin)
    _normalize_source_counterparty_columns(raw_txn, normalized)
    if normalized.get("counter_party"):
        normalized["counter_party"] = _clean_wrapped_text(str(normalized.get("counter_party") or ""))
    if normalized.get("counter_account"):
        normalized["counter_account"] = _clean_account(str(normalized.get("counter_account") or ""))
    balance = _normalize_monetary_cell(_cell_value(raw_txn, "余额", "账户余额", "本次余额", "账面余额"))
    if balance is not None:
        normalized["balance"] = float(balance)
    reference = _cell_value(raw_txn, "交易流水号", "流水号", "Reference")
    if reference:
        normalized["reference"] = reference

    if not str(normalized.get("counter_party", "") or "").strip():
        cp = _cell_value(
            raw_txn,
            *_registry_field_keys(plugin, "对方户名", _COUNTERPARTY_KEYS),
            "对方账号与户名",
            "备注",
            "Remarks",
        )
        if cp:
            normalized["counter_party"] = _clean_wrapped_text(cp)

    explicit_direction = _normalize_direction_text(_explicit_source_column_value(raw_txn, _DIRECTION_KEYS))
    income_nonzero = income is not None and income != 0
    expense_nonzero = expense is not None and expense != 0
    if income_nonzero and expense_nonzero:
        if explicit_direction == "income":
            selected_amount = abs(float(income))
        elif explicit_direction == "expense":
            selected_amount = abs(float(expense))
        else:
            normalized["amount"] = None
            normalized["amount_cny"] = None
            normalized["direction"] = ""
            return normalized
        normalized["amount"] = selected_amount
        normalized["amount_cny"] = selected_amount
        normalized["direction"] = explicit_direction
    elif income_nonzero:
        selected_amount = abs(float(income))
        normalized["amount"] = selected_amount
        normalized["amount_cny"] = selected_amount
        normalized["direction"] = "income"
    elif expense_nonzero:
        selected_amount = abs(float(expense))
        normalized["amount"] = selected_amount
        normalized["amount_cny"] = selected_amount
        normalized["direction"] = "expense"
    elif income is not None or expense is not None:
        normalized["amount"] = 0.0
        normalized["amount_cny"] = 0.0
        if explicit_direction in {"income", "expense"}:
            normalized["direction"] = explicit_direction
        elif income is not None and expense is None:
            normalized["direction"] = "income"
        elif expense is not None and income is None:
            normalized["direction"] = "expense"
        else:
            normalized["direction"] = ""
    else:
        normalized["amount"] = None
        normalized["amount_cny"] = None
        normalized["direction"] = explicit_direction if explicit_direction in {"income", "expense"} else ""
    return normalized


def _normalize_direction_amount(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any] | None:
    direction_raw = _cell_value(raw_txn, *_DIRECTION_KEYS)
    amount = _normalize_monetary_cell(_cell_value(raw_txn, "交易金额", "金额", "发生额", "Amount"))
    if not direction_raw or amount is None:
        return None
    direction = _normalize_direction_text(direction_raw)
    if direction not in ("income", "expense"):
        return None

    normalized = _normalize_with_temporal_context(raw_txn, plugin)
    _normalize_source_counterparty_columns(raw_txn, normalized)
    normalized["amount"] = abs(float(amount))
    normalized["amount_cny"] = abs(float(amount))
    normalized["direction"] = direction
    balance = _normalize_monetary_cell(_cell_value(raw_txn, "余额", "账户余额", "本次余额", "账面余额"))
    if balance is not None:
        normalized["balance"] = float(balance)
    if normalized.get("counter_party"):
        normalized["counter_party"] = _clean_wrapped_text(str(normalized.get("counter_party") or ""))
    if normalized.get("counter_account"):
        normalized["counter_account"] = _clean_account(str(normalized.get("counter_account") or ""))
    return normalized


def _normalize_embedded_direction_amount(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any] | None:
    value = _cell_value(raw_txn, "交易金额", "金额", "发生额", "Amount")
    amount = _normalize_monetary_cell(value)
    if amount is None:
        return None
    match = _MONEY_PREFIX_RE.search(str(value or "").strip())
    suffix = str(value or "")[match.end() :] if match else ""
    markers = re.findall(r"[收支借贷]", suffix)
    if not markers:
        summary = _cell_value(raw_txn, "摘要", "交易摘要", "Description", "Memo")
        trailing = re.search(r"([收支借贷])\s*$", summary)
        markers = [trailing.group(1)] if trailing else []
    marker = markers[-1] if markers else ""
    if marker in {"收", "贷"}:
        direction = "income"
    elif marker in {"支", "借"} or ("付" in suffix and "收" not in suffix):
        direction = "expense"
    else:
        return None

    normalized = _normalize_with_temporal_context(raw_txn, plugin)
    _normalize_source_counterparty_columns(raw_txn, normalized)
    normalized["amount"] = float(amount)
    normalized["amount_cny"] = float(amount)
    normalized["direction"] = direction
    balance = _normalize_monetary_cell(_cell_value(raw_txn, "余额", "账户余额", "本次余额", "账面余额"))
    if balance is not None:
        normalized["balance"] = float(balance)
    return normalized


def _normalize_monetary_cell(value: str) -> float | None:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).strip()
    match = _MONEY_PREFIX_RE.search(compact)
    return normalize_amount(match.group(1)) if match else None


def _normalize_direction_text(value: str) -> str:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "").strip()))
    if any(token in text for token in ("收入", "转入", "收人")):
        return "income"
    if any(token in text for token in ("支出", "转出", "支山", "支鼎", "攴出")):
        return "expense"
    if "贷" in text or re.search(r"\bCr\b", text, re.IGNORECASE):
        return "income"
    if "借" in text or re.search(r"\bDr\b", text, re.IGNORECASE):
        return "expense"
    return ""


def _clean_wrapped_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff(（])", "", text)
    return re.sub(r"(?<=[)）])\s+(?=[\u4e00-\u9fff])", "", text)


def _clean_account(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[0-9*\s＊]+", text):
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text)


def _split_embedded_counter_account(value: str) -> tuple[str, str]:
    """Split a trailing counterparty account from a collapsed source cell."""
    text = _clean_wrapped_text(value)
    compact = re.sub(r"\s+", "", text)
    match = re.search(r"(?<!\d)([0-9*＊]{8,32})$", compact)
    if match is None:
        return text, ""

    account = match.group(1)
    prefix_length = match.start(1)
    compact_seen = 0
    prefix_chars: list[str] = []
    for char in text:
        if not char.isspace():
            if compact_seen >= prefix_length:
                break
            compact_seen += 1
        prefix_chars.append(char)
    return "".join(prefix_chars).rstrip(), account


def _extract_split_grid_records(
    tables: list[list[list[str]]],
    header_row_idx: int,
    raw_headers: list[str],
) -> list[dict[str, str]]:
    transactions: list[dict[str, str]] = []
    for tbl in tables:
        if not tbl or header_row_idx >= len(tbl):
            continue
        for row in tbl[header_row_idx + 1 :]:
            if not row or not any(str(c).strip() for c in row):
                continue
            first_cell = str(row[0] or "").strip()
            if any(kw in first_cell for kw in ("合计", "小计", "本页", "总计")) or is_footer_or_total_row(row):
                continue
            if not row_has_transaction_data(row, strict_first_col=False):
                continue
            txn: dict[str, str] = {}
            for idx, cell in enumerate(row):
                header = raw_headers[idx] if idx < len(raw_headers) else f"col_{idx}"
                txn[header] = str(cell or "").strip()
            income = _normalize_monetary_cell(_cell_value(txn, *_SPLIT_CREDIT_KEYS))
            expense = _normalize_monetary_cell(_cell_value(txn, *_SPLIT_DEBIT_KEYS))
            if float(income or 0) <= 0 and float(expense or 0) <= 0:
                continue
            transactions.append(txn)
    return transactions


def _with_internal_row_sources(
    transactions: list[dict[str, Any]],
    parse_result: Any | None = None,
) -> list[dict[str, Any]]:
    inferred_sources = _infer_row_sources(transactions, parse_result)
    for transaction in transactions:
        source = transaction.get("_source")
        if isinstance(source, dict) and _positive_int(source.get("source_page")) is not None:
            source_page = _positive_int(source.get("source_page"))
            source.setdefault("page_range", [source_page, source_page])
            _ensure_row_bbox_source_ref(source)
            continue
        source_page = _internal_source_page(transaction)
        if source_page is None:
            inferred = inferred_sources.pop(0) if inferred_sources else None
            if inferred is not None:
                transaction["_source"] = inferred
            continue
        row_source: dict[str, Any] = {
            "source_page": source_page,
            "page_range": [source_page, source_page],
            **({"table_id": table_id} if (table_id := _internal_source_value(transaction, "_source_table_id")) else {}),
            **(
                {"source_row_index": int(row_index)}
                if (row_index := _internal_source_value(transaction, "_source_row_index")).isdigit()
                else {}
            ),
        }
        bbox_value = _internal_source_value(transaction, "_source_bbox")
        try:
            bbox = [float(value) for value in bbox_value.split(",")]
        except ValueError:
            bbox = []
        if len(bbox) == 4:
            row_source["bbox"] = bbox
            _ensure_row_bbox_source_ref(row_source)
        transaction["_source"] = row_source
    return transactions


def _ensure_row_bbox_source_ref(source: dict[str, Any]) -> None:
    """Expose a recovered row bbox through the standard audit source-ref path."""
    bbox = source.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return
    refs = source.get("source_refs")
    if not isinstance(refs, list):
        refs = []
        source["source_refs"] = refs
    if any(isinstance(ref, dict) and ref.get("bbox") == list(bbox) for ref in refs):
        return
    refs.append(
        {
            "source_page": source.get("source_page"),
            "page_range": list(source.get("page_range") or []),
            "bbox": list(bbox),
            "source": "native_pdf_words",
        }
    )


def _finalize_transactions(
    transactions: list[dict[str, Any]],
    parse_result: Any | None = None,
    full_text: str = "",
) -> list[dict[str, Any]]:
    sourced = _with_internal_row_sources(transactions, parse_result)
    _recover_missing_counterparties_from_page_text(sourced, parse_result, full_text)
    return sourced


def _recover_missing_counterparties_from_page_text(
    transactions: list[dict[str, Any]],
    parse_result: Any | None,
    full_text: str = "",
) -> None:
    if not transactions:
        return
    parse_result = _read_view(parse_result)
    page_texts = dict(_parse_result_page_texts(parse_result)) if parse_result is not None else {}
    pdf_page_texts = dict(_pdf_page_texts_from_provenance(parse_result)) if parse_result is not None else {}
    fallback_text = str(full_text or "")
    if not page_texts and not pdf_page_texts and not fallback_text:
        return

    by_page: dict[int, list[dict[str, Any]]] = {}
    for transaction in transactions:
        page = _transaction_source_page(transaction)
        if page is not None:
            by_page.setdefault(page, []).append(transaction)

    for page, page_transactions in by_page.items():
        candidate_texts = []
        for candidate_text in (page_texts.get(page, ""), pdf_page_texts.get(page, ""), fallback_text):
            if candidate_text and candidate_text not in candidate_texts:
                candidate_texts.append(candidate_text)
        for page_text in candidate_texts:
            if not page_text:
                continue
            page_index = _compact_text_with_offsets(page_text)
            locations = _locate_page_transactions(page_index[0], page_transactions)
            for index, transaction in enumerate(page_transactions):
                if not _needs_counterparty_recovery(transaction):
                    continue
                location = locations[index] if index < len(locations) else {}
                account_end = _positive_or_zero_int(location.get("account_end")) if location else None
                if account_end is None:
                    continue
                next_start = _next_located_row_start(locations, index, len(page_index[0]))
                candidate = _slice_original_by_compact(page_text, page_index[1], account_end, next_start)
                recovered = _clean_recovered_counterparty(candidate)
                if (
                    recovered
                    and not _recovered_counterparty_repeats_transaction_fields(recovered, transaction)
                    and _is_safe_recovered_counterparty(recovered)
                ):
                    _set_counterparty(transaction, recovered)


def _transaction_source_page(transaction: dict[str, Any]) -> int | None:
    source = transaction.get("_source") if isinstance(transaction.get("_source"), dict) else {}
    return _positive_int(source.get("source_page")) or _internal_source_page(transaction)


def _needs_counterparty_recovery(transaction: dict[str, Any]) -> bool:
    if not _cell_value(transaction, *_COUNTER_ACCOUNT_KEYS):
        return False
    counterparty = _cell_value(transaction, *_COUNTERPARTY_KEYS)
    return not counterparty or _looks_like_incomplete_counterparty(counterparty)


def _looks_like_incomplete_counterparty(value: str) -> bool:
    compact = _signature_value(value)
    if compact in {"入", "收", "收入", "出", "支", "限公司", "有限公司", "代收)", "代收）"}:
        return True
    if compact.startswith(("限公司", "代收)", "代收）")):
        return True
    fee_tail = "电子渠道跨行转账手续费收"
    if fee_tail in compact and not compact.startswith(("企业电子渠道", "个人电子渠道", "电子渠道")):
        return True
    return False


def _compact_text_with_offsets(text: str) -> tuple[str, list[int]]:
    compact_chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(str(text or "")):
        normalized = unicodedata.normalize("NFKC", char)
        for normalized_char in normalized:
            if normalized_char.isspace():
                continue
            compact_chars.append(normalized_char)
            offsets.append(index)
    return "".join(compact_chars), offsets


def _locate_page_transactions(
    compact_text: str,
    transactions: list[dict[str, Any]],
) -> list[dict[str, int]]:
    cursor = 0
    locations: list[dict[str, int]] = []
    for transaction in transactions:
        account = _signature_value(_cell_value(transaction, *_COUNTER_ACCOUNT_KEYS))
        if not account:
            row_start = _best_row_start_position(compact_text, cursor, transaction)
            if row_start >= 0:
                locations.append({"row_start": row_start})
                cursor = row_start
            else:
                locations.append({})
            continue
        account_pos = _best_account_position(compact_text, account, cursor, transaction)
        if account_pos < 0:
            locations.append({})
            continue
        row_start = _infer_transaction_start(compact_text, transaction, account_pos)
        account_end = account_pos + len(account)
        locations.append({"row_start": row_start, "account_end": account_end})
        cursor = account_end
    return locations


def _best_account_position(
    compact_text: str,
    account: str,
    cursor: int,
    transaction: dict[str, Any],
) -> int:
    positions = _find_all_after(compact_text, account, cursor)
    if not positions:
        return -1
    anchors = _transaction_row_start_anchor_groups(transaction)
    if not anchors:
        return positions[0]

    best_position = positions[0]
    best_score = -1
    for position in positions[:20]:
        window = compact_text[max(0, position - 260) : position]
        score = sum(1 for variants in anchors if any(variant and variant in window for variant in variants))
        if score > best_score:
            best_position = position
            best_score = score
    return best_position


def _best_row_start_position(compact_text: str, cursor: int, transaction: dict[str, Any]) -> int:
    anchors = _transaction_row_start_anchor_groups(transaction)
    positions: list[int] = []
    for variants in anchors:
        for variant in variants:
            if not variant:
                continue
            position = compact_text.find(variant, max(cursor, 0))
            if position >= 0:
                positions.append(position)
    return min(positions) if positions else -1


def _find_all_after(text: str, needle: str, start: int) -> list[int]:
    positions: list[int] = []
    position = text.find(needle, max(start, 0))
    while position >= 0:
        positions.append(position)
        position = text.find(needle, position + 1)
    return positions


def _transaction_row_start_anchor_groups(transaction: dict[str, Any]) -> list[set[str]]:
    groups: list[set[str]] = []
    sequence = _cell_value(transaction, "序号", "sequence_no")
    if sequence:
        groups.append({_signature_value(sequence)})
    date = _cell_value(transaction, "交易日期", "日期", "记账日期", "date")
    if date:
        groups.append(_date_anchor_variants(date))
    timestamp = _cell_value(transaction, "交易时间", "时间", "timestamp")
    if timestamp:
        groups.append(_time_anchor_variants(timestamp))
    summary = _cell_value(transaction, "摘要", "交易摘要", "备注", "summary")
    if summary:
        groups.append({_signature_value(summary)})
    return [group for group in groups if any(group)]


def _infer_transaction_start(compact_text: str, transaction: dict[str, Any], account_pos: int) -> int:
    window_start = max(0, account_pos - 260)
    candidates: list[int] = []
    for variants in _transaction_row_start_anchor_groups(transaction):
        for variant in variants:
            if not variant:
                continue
            position = compact_text.rfind(variant, window_start, account_pos)
            if position >= 0:
                candidates.append(position)
    return min(candidates) if candidates else account_pos


def _next_located_row_start(locations: list[dict[str, int]], index: int, fallback: int) -> int:
    current_end = locations[index].get("account_end", 0) if index < len(locations) else 0
    for location in locations[index + 1 :]:
        row_start = location.get("row_start", 0)
        if row_start > current_end:
            return row_start
    return fallback


def _slice_original_by_compact(text: str, offsets: list[int], start: int, end: int) -> str:
    if start >= end or start >= len(offsets):
        return ""
    bounded_end = min(end, len(offsets))
    original_start = offsets[start]
    original_end = offsets[bounded_end - 1] + 1
    return text[original_start:original_end]


def _clean_recovered_counterparty(value: str) -> str:
    text = _clean_wrapped_text(value)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[（(])", "", text)
    text = re.sub(r"(?<=[（(])\s+", "", text)
    text = re.sub(r"^(?:null|None|无|--|[-:：|]+)+", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return ""
    text = _strip_recovered_counterparty_after_marker(text)
    text = text.strip(" -:：|")
    compact = _signature_value(text)
    if compact in {"入", "收", "出", "支", "限公司", "有限公司", "代收)", "代收）", "null", "None"}:
        return ""
    if re.fullmatch(r"(?:入|收|收入|出|支)\d{1,8}", compact):
        return ""
    if _looks_like_transaction_fragment(compact):
        return ""
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", text):
        return ""
    return text


def _looks_like_transaction_fragment(compact: str) -> bool:
    if re.match(r"^\d{1,8}20\d{2}[-/.]?\d{2}[-/.]?\d{2}", compact):
        return True
    if re.match(r"^20\d{2}[-/.]?\d{2}[-/.]?\d{2}", compact) and re.search(r"\d+(?:,\d{3})*\.\d{2}", compact):
        return True
    return bool(re.match(r"^\d{1,8}20\d{6}", compact) and re.search(r"\d{10,}", compact))


def _strip_recovered_counterparty_after_marker(value: str) -> str:
    compact, _offsets = _compact_text_with_offsets(value)
    markers = (*_COUNTERPARTY_RECOVERY_BOUNDARY_MARKERS, "借方笔数", "贷方笔数", "合计笔数")
    positions = [compact.find(marker) for marker in markers if compact.find(marker) >= 0]
    if not positions:
        return value
    return _prefix_by_compact_length(value, min(positions))


def _is_safe_recovered_counterparty(value: str) -> bool:
    compact = _signature_value(value)
    if not compact or len(compact) > 120:
        return False
    if any(marker in compact for marker in _COUNTERPARTY_RECOVERY_BOUNDARY_MARKERS):
        return False
    if len(re.findall(r"(?<!\d)\d{8,}(?!\d)", compact)) > 1:
        return False
    if re.fullmatch(r"[\d*＊,./:：-]{8,}", compact):
        return False
    repeated_channels = sum(compact.count(marker) for marker in ("WL财付通", "WL支付宝", "微信转账"))
    return repeated_channels <= 2


def _recovered_counterparty_repeats_transaction_fields(
    value: str,
    transaction: dict[str, Any],
) -> bool:
    """Reject a source-null counterparty slice made from later row columns."""
    compact = _signature_value(value)
    summary = _signature_value(_cell_value(transaction, "交易摘要", "摘要", "备注", "用途"))
    if not summary or not compact.startswith(summary):
        return False
    if compact == summary:
        return True
    monetary_values = (
        _cell_value(transaction, "交易金额", "金额", "发生额", *_SPLIT_DEBIT_KEYS, *_SPLIT_CREDIT_KEYS),
        _cell_value(transaction, "余额", "账户余额", "本次余额", "账面余额"),
    )
    return any(_signature_value(value) in compact for value in monetary_values if value)


def _prefix_by_compact_length(value: str, compact_length: int) -> str:
    seen = 0
    chars: list[str] = []
    for char in value:
        normalized = unicodedata.normalize("NFKC", char)
        char_width = len([part for part in normalized if not part.isspace()])
        if seen + char_width > compact_length:
            break
        chars.append(char)
        seen += char_width
    return "".join(chars).strip()


def _set_counterparty(transaction: dict[str, Any], value: str) -> None:
    for key in _COUNTERPARTY_KEYS:
        if key in transaction:
            transaction[key] = value
            return
    transaction["对方户名"] = value


def _infer_row_sources(transactions: list[dict[str, Any]], parse_result: Any | None) -> list[dict[str, Any]]:
    if not transactions or parse_result is None:
        return []
    parse_result = _read_view(parse_result)
    logical_sources = _logical_table_row_sources(parse_result)
    if not logical_sources:
        logical_sources = _physical_table_row_sources(parse_result)
    if len(logical_sources) == len(transactions):
        return [_public_source(source) for source in logical_sources]

    by_signature: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for source in logical_sources:
        signature = tuple(source.pop("_signature", ()))
        if signature:
            by_signature.setdefault(signature, []).append(source)

    inferred: list[dict[str, Any]] = []
    for transaction in transactions:
        candidates = by_signature.get(_transaction_signature(transaction), [])
        inferred.append(candidates.pop(0) if candidates else {})
    if len([source for source in inferred if source]) == len(transactions):
        return [source for source in inferred if source]

    text_sources = _text_page_row_sources(transactions, parse_result)
    return text_sources if len(text_sources) == len(transactions) else []


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key != "_signature"}


def _logical_table_row_sources(parse_result: Any) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for table in getattr(parse_result, "logical_tables", []) or []:
        rows = list(getattr(table, "rows", []) or [])
        provenance = list(getattr(table, "provenance", []) or [])
        for row_index, row in enumerate(rows):
            row_source = provenance[row_index] if row_index < len(provenance) else None
            source_page = _positive_int(
                getattr(row_source, "source_page", 0) if row_source is not None else 0
            ) or _positive_int(getattr(row, "source_page", 0))
            if source_page is None:
                continue
            source_table_id = str(
                getattr(row, "source_physical_id", "")
                or (getattr(row_source, "source_table_id", "") if row_source is not None else "")
                or ""
            )
            row_source_index = _positive_or_zero_int(getattr(row, "source_row_index", -1))
            if row_source_index is None and row_source is not None:
                row_source_index = _positive_or_zero_int(getattr(row_source, "source_row_index", -1))
            row_source_index = row_source_index if row_source_index is not None else row_index
            cells = list(getattr(row, "cells", []) or [])
            source_cell_refs = _row_source_cell_refs(row, cells)
            evidence_ids = _row_evidence_ids(cells)
            sources.append(
                {
                    "source_page": source_page,
                    "page_id": f"page:{source_page:04d}",
                    **({"table_id": source_table_id} if source_table_id else {}),
                    "source_row_index": row_source_index,
                    "page_range": [source_page, source_page],
                    **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
                    **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                    "_signature": _row_signature([getattr(cell, "text", "") for cell in cells]),
                }
            )
    return sources


def _physical_table_row_sources(parse_result: Any) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", []) or []:
        page_number = _positive_int(getattr(page, "page_number", 0))
        if page_number is None:
            continue
        for table_index, table in enumerate(getattr(page, "tables", []) or []):
            rows = list(getattr(table, "rows", []) or [])
            for row_index, row in enumerate(rows):
                cells = list(getattr(row, "cells", []) or [])
                if not any(str(getattr(cell, "text", "") or "").strip() for cell in cells):
                    continue
                source_cell_refs = _row_source_cell_refs(row, cells)
                evidence_ids = _row_evidence_ids(cells)
                sources.append(
                    {
                        "source_page": page_number,
                        "page_id": f"page:{page_number:04d}",
                        "table_id": str(getattr(table, "table_id", "") or f"pt_{page_number}_{table_index}"),
                        "source_row_index": _positive_or_zero_int(getattr(row, "source_row_index", -1))
                        if _positive_or_zero_int(getattr(row, "source_row_index", -1)) is not None
                        else row_index,
                        "page_range": [page_number, page_number],
                        **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
                        **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                        "_signature": _row_signature([getattr(cell, "text", "") for cell in cells]),
                    }
                )
    return sources


def _text_page_row_sources(transactions: list[dict[str, Any]], parse_result: Any) -> list[dict[str, Any]]:
    page_texts = _parse_result_page_texts(parse_result)
    if not page_texts:
        return []
    page_compact = [(page, _signature_value(text)) for page, text in page_texts]
    page_counters: dict[int, int] = {}
    last_page = page_compact[0][0]
    sources: list[dict[str, Any]] = []
    for transaction in transactions:
        anchors = _transaction_page_anchor_groups(transaction)
        if len(anchors) < 3:
            return []
        best_page = 0
        best_score = -1
        threshold = min(3, len(anchors))
        for page, text in page_compact:
            if page < last_page:
                continue
            score = sum(1 for variants in anchors if any(variant and variant in text for variant in variants))
            if score > best_score:
                best_page = page
                best_score = score
        if best_page <= 0 or best_score < threshold:
            return []
        row_index = page_counters.get(best_page, 0)
        page_counters[best_page] = row_index + 1
        last_page = best_page
        sources.append(
            {
                "source": "full_text_page_anchor",
                "source_page": best_page,
                "page_id": f"page:{best_page:04d}",
                "source_row_index": row_index,
                "page_range": [best_page, best_page],
            }
        )
    return sources


def _parse_result_page_texts(parse_result: Any) -> list[tuple[int, str]]:
    parse_result = _read_view(parse_result)
    page_texts: list[tuple[int, str]] = []
    for page in getattr(parse_result, "pages", []) or []:
        page_number = _positive_int(getattr(page, "source_page_number", None)) or _positive_int(
            getattr(page, "page_number", 0)
        )
        if page_number is None:
            continue
        parts = [
            str(getattr(text, "content", "") or "").strip()
            for text in getattr(page, "texts", []) or []
            if str(getattr(text, "content", "") or "").strip()
        ]
        for table in getattr(page, "tables", []) or []:
            for row in getattr(table, "rows", []) or []:
                cells = [str(getattr(cell, "text", "") or "").strip() for cell in getattr(row, "cells", []) or []]
                if any(cells):
                    parts.append(" ".join(cells))
        if parts:
            page_texts.append((page_number, "\n".join(parts)))
    return page_texts


def _pdf_page_texts_from_provenance(parse_result: Any) -> list[tuple[int, str]]:
    parse_result = _read_view(parse_result)
    provenance = getattr(parse_result, "provenance", None)
    file_path = str(getattr(provenance, "file_path", "") or getattr(parse_result, "file_path", "") or "").strip()
    if not file_path or not file_path.lower().endswith(".pdf"):
        return []
    try:
        from pathlib import Path

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return []
        import fitz

        with fitz.open(path) as doc:
            return [
                (index + 1, text)
                for index, page in enumerate(doc)
                if (text := str(page.get_text("text") or "").strip())
            ]
    except Exception:
        return []


def _read_view(parse_result: Any) -> Any:
    if parse_result is None:
        return None
    if getattr(parse_result, "pages", None) is not None or getattr(parse_result, "provenance", None) is not None:
        return parse_result
    to_read_view = getattr(parse_result, "to_read_view", None)
    if callable(to_read_view):
        try:
            return to_read_view()
        except Exception:
            return parse_result
    return parse_result


def _transaction_page_anchor_groups(transaction: dict[str, Any]) -> list[set[str]]:
    groups: list[set[str]] = []
    for key in ("序号", "流水号"):
        value = str(transaction.get(key) or "").strip()
        if value:
            groups.append({_signature_value(value)})
            break
    for key in ("交易日期", "日期", "记账日期"):
        value = str(transaction.get(key) or "").strip()
        if value:
            groups.append(_date_anchor_variants(value))
            break
    for key in ("交易时间", "时间", "timestamp"):
        value = str(transaction.get(key) or "").strip()
        if value:
            groups.append(_time_anchor_variants(value))
            break
    for key in ("借方发生额", "贷方发生额", "交易金额", "金额"):
        value = str(transaction.get(key) or "").strip()
        if value:
            groups.append(_money_anchor_variants(value))
            break
    for key in ("余额", "账户余额", "本次余额"):
        value = str(transaction.get(key) or "").strip()
        if value:
            groups.append(_money_anchor_variants(value))
            break
    for key in ("对方账户", "对方账号", "counter_account"):
        value = str(transaction.get(key) or "").strip()
        if value:
            groups.append({_signature_value(value)})
            break
    return [group for group in groups if any(group)]


def _date_anchor_variants(value: str) -> set[str]:
    compact = _signature_value(value)
    variants = {compact}
    match = re.search(r"(20\d{2})[-/.]?(\d{2})[-/.]?(\d{2})", compact)
    if match:
        y, m, d = match.groups()
        variants.update({f"{y}{m}{d}", f"{y}-{m}-{d}", f"{y}/{m}/{d}"})
    return {_signature_value(variant) for variant in variants if variant}


def _time_anchor_variants(value: str) -> set[str]:
    compact = _signature_value(value)
    variants = {compact}
    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", value)
    if match:
        h, m, s = match.groups()
        variants.update({f"{h}:{m}:{s}", f"{int(h):02d}:{m}:{s}", f"{int(h):02d}{m}{s}"})
    return {_signature_value(variant) for variant in variants if variant}


def _money_anchor_variants(value: str) -> set[str]:
    compact = _signature_value(value).lstrip("+-")
    variants = {compact}
    amount = normalize_amount(compact)
    if amount is not None:
        variants.add(f"{amount:.2f}")
        variants.add(f"{amount:,.2f}")
    return {_signature_value(variant).lstrip("+-") for variant in variants if variant}


def _row_source_cell_refs(row: Any, cells: list[Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for ref in [
        *(getattr(row, "source_cell_refs", []) or []),
        *(ref for cell in cells for ref in (getattr(cell, "source_cell_refs", []) or [])),
    ]:
        if isinstance(ref, dict) and ref not in refs:
            refs.append(dict(ref))
    return refs


def _row_evidence_ids(cells: list[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(evidence_id)
            for cell in cells
            for evidence_id in (getattr(cell, "evidence_ids", []) or [])
            if str(evidence_id)
        )
    )


def _transaction_signature(transaction: dict[str, Any]) -> tuple[str, ...]:
    return _row_signature(value for key, value in transaction.items() if not str(key).startswith("_"))


def _row_signature(values: Any) -> tuple[str, ...]:
    return tuple(_signature_value(value) for value in values if _signature_value(value))


def _signature_value(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))


def _positive_int(value: Any) -> int | None:
    try:
        number = int(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_or_zero_int(value: Any) -> int | None:
    try:
        number = int(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _internal_source_page(transaction: dict[str, Any]) -> int | None:
    try:
        page = int(_internal_source_value(transaction, "_source_page"))
    except ValueError:
        return None
    return page if page > 0 else None


def _internal_source_value(transaction: dict[str, Any], field_name: str) -> str:
    for key, value in transaction.items():
        if str(key).strip() == field_name:
            return str(value or "").strip()
    return ""


def _extract_internal_source_grid_records(
    tables: list[list[list[str]]],
    parse_result: Any | None = None,
    full_text: str = "",
    *,
    require_bbox: bool = False,
) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    for tbl in tables:
        if not tbl:
            continue
        header_idx = next(
            (
                idx
                for idx, row in enumerate(tbl[:10])
                if any(str(cell or "").strip() == "_source_page" for cell in row)
                and (not require_bbox or any(str(cell or "").strip() == "_source_bbox" for cell in row))
                and any("交易" in str(cell or "") for cell in row)
            ),
            -1,
        )
        if header_idx < 0:
            continue
        raw_headers = [str(cell or "").strip() for cell in tbl[header_idx]]
        for row in tbl[header_idx + 1 :]:
            if not row or not any(str(cell or "").strip() for cell in row):
                continue
            if is_footer_or_total_row(row):
                continue
            txn = {
                raw_headers[idx] if idx < len(raw_headers) and raw_headers[idx] else f"col_{idx}": str(
                    cell or ""
                ).strip()
                for idx, cell in enumerate(row)
            }
            if any(value for key, value in txn.items() if key != "_source_page"):
                transactions.append(txn)
    return _finalize_transactions(transactions, parse_result, full_text)


def extract_transactions(ctx: StyleContext, plugin: Any) -> list[dict[str, Any]]:
    variant = match_institution(ctx.full_text, ctx.institution)

    if (
        ctx.parse_result is not None
        and ctx.reconstruction is not None
        and ctx.reconstruction.source in {"canonical_table", "none"}
        and not ctx.prefer_context_tables
    ):
        logical_stats: dict[str, int] = {}
        logical_transactions = extract_logical_rows_with_provenance(
            ctx.parse_result,
            plugin.column_registry,
            strict_first_col=True,
            stats=logical_stats,
        )
        if logical_transactions:
            stitched_count = int(logical_stats.get("stitched_continuation_rows") or 0)
            if stitched_count > 0:
                ctx.reconstruction = replace(
                    ctx.reconstruction,
                    stitched_continuation_rows=stitched_count,
                )
            return _finalize_transactions(logical_transactions, ctx.parse_result, ctx.full_text)

    split_txns: list[dict[str, str]] = []
    for tbl in ctx.tables:
        if not tbl:
            continue
        for row_idx, row in enumerate(tbl[:15]):
            if row_has_transaction_data(row, strict_first_col=False):
                continue
            raw_headers = [str(c or "").strip() for c in row]
            if has_split_debit_credit_headers([[raw_headers]]):
                split_txns.extend(_extract_split_grid_records([tbl], row_idx, raw_headers))
                break
    if split_txns:
        return _finalize_transactions(split_txns, ctx.parse_result, ctx.full_text)

    internal_source_batch = _extract_internal_source_grid_records(
        ctx.tables,
        ctx.parse_result,
        ctx.full_text,
        require_bbox=True,
    )
    if internal_source_batch:
        return internal_source_batch
    tables = normalize_table_headers(ctx.tables, variant=variant)
    internal_source_batch = _extract_internal_source_grid_records(tables, ctx.parse_result, ctx.full_text)
    if internal_source_batch:
        return internal_source_batch

    batch = extract_all_tables(
        tables,
        plugin.column_registry,
        prefer_strict=True,
        strict_first_col=True,
    )
    if batch:
        return _finalize_transactions(batch, ctx.parse_result, ctx.full_text)

    header = detect_headers(tables, plugin.column_registry, prefer_strict=True)
    if header is None:
        header_row_idx, raw_headers, col_map = plugin._detect_headers(tables)
        return _finalize_transactions(
            plugin._extract_records(tables, header_row_idx, raw_headers, col_map),
            ctx.parse_result,
            ctx.full_text,
        )

    raw_headers = header.raw_headers
    if has_split_debit_credit_headers([[raw_headers]]):
        return _finalize_transactions(
            _extract_split_grid_records(tables, header.row_index, raw_headers),
            ctx.parse_result,
            ctx.full_text,
        )

    rows = extract_rows_from_header(
        tables,
        header,
        plugin.column_registry,
        strict_first_col=True,
    )
    if rows:
        return _finalize_transactions(rows, ctx.parse_result, ctx.full_text)

    header_row_idx, raw_headers, col_map = plugin._detect_headers(tables)
    return _finalize_transactions(
        plugin._extract_records(tables, header_row_idx, raw_headers, col_map),
        ctx.parse_result,
        ctx.full_text,
    )


def _normalize_wrapped_temporal_fields(
    normalized: dict[str, Any],
    raw_txn: dict[str, str],
) -> dict[str, Any]:
    out = dict(normalized)
    date_value = _cell_value(raw_txn, "交易日期", "记账日", "记账日期", "日期", "Date")
    timestamp_value = _cell_value(raw_txn, "交易时间", "时间", "Time")
    timestamp_compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", timestamp_value))
    date_compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", date_value))

    temporal_candidate = timestamp_compact
    if timestamp_compact and not re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", timestamp_compact):
        temporal_candidate = f"{date_compact}{timestamp_compact}" if date_compact else ""
    elif not temporal_candidate:
        temporal_candidate = date_compact

    if temporal_candidate:
        parsed = normalize_timestamp(temporal_candidate)
        if re.match(r"^\d{4}-\d{2}-\d{2}", parsed):
            out["date"] = parsed[:10]
            if ":" in temporal_candidate or re.fullmatch(r"\d{6}", timestamp_compact):
                out["timestamp"] = parsed

    balance = _normalize_monetary_cell(_cell_value(raw_txn, "余额", "账户余额", "本次余额", "账面余额"))
    if balance is not None:
        out["balance"] = float(balance)
    source_sequence = _internal_source_value(raw_txn, "_source_sequence_no")
    if re.fullmatch(r"\d{1,9}", source_sequence):
        out["sequence_no"] = source_sequence
    return out


def normalize_record(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any]:
    if raw_txn.get("_compact") == "1":
        from docmirror.plugins.bank_statement.styles.compact_merged import normalize_record as compact_norm

        return _normalize_wrapped_temporal_fields(compact_norm(raw_txn), raw_txn)

    split = normalize_split_debit_credit(raw_txn, plugin)
    if split is not None:
        return _normalize_wrapped_temporal_fields(split, raw_txn)

    from docmirror.plugins.bank_statement.styles.signed_amount import parse_signed_amount

    for key, value in raw_txn.items():
        normalized_key = normalize_header_cell(key)
        if any(n in normalized_key for n in ("金额", "发生", "Amount")) and str(value).strip().startswith(("+", "-")):
            amount, direction = parse_signed_amount(str(value))
            if amount is not None:
                normalized = _normalize_with_temporal_context(raw_txn, plugin)
                _normalize_source_counterparty_columns(raw_txn, normalized)
                normalized["amount"] = amount
                normalized["amount_cny"] = amount
                normalized["direction"] = direction
                return _normalize_wrapped_temporal_fields(normalized, raw_txn)

    normalized = _normalize_with_temporal_context(raw_txn, plugin)
    _normalize_source_counterparty_columns(raw_txn, normalized)
    return _normalize_wrapped_temporal_fields(normalized, raw_txn)


def refine_missing_directions_from_balance_chain(records: list[dict[str, Any]]) -> None:
    """Infer or correct directions when the source balance chain is unique."""
    source_order = _record_source_order(records)
    for index, record in enumerate(records):
        normalized = record.get("normalized") or {}
        raw = record.get("raw") or {}
        source_amount = _normalize_monetary_cell(_cell_value(raw, "交易金额", "金额", "发生额", "Amount"))
        negative_reversal = source_amount is not None and source_amount < 0
        amount = _safe_float(normalized.get("amount"))
        balance = _safe_float(normalized.get("balance"))
        candidates: set[str] = set()
        if amount is not None and amount > 0 and balance is not None:
            if source_order != "reverse" and index > 0:
                previous_balance = _safe_float((records[index - 1].get("normalized") or {}).get("balance"))
                inferred = _direction_from_balance(previous_balance, amount, balance)
                if inferred:
                    candidates.add(inferred)
            if source_order == "reverse" and index + 1 < len(records):
                next_balance = _safe_float((records[index + 1].get("normalized") or {}).get("balance"))
                inferred = _direction_from_balance(next_balance, amount, balance)
                if inferred:
                    candidates.add(inferred)
        if len(candidates) == 1:
            normalized["direction"] = candidates.pop()
            continue
        if normalized.get("direction") in {"income", "expense"} and not negative_reversal:
            continue
        if not negative_reversal:
            semantic_direction = _direction_from_source_semantics(raw)
            if semantic_direction:
                normalized["direction"] = semantic_direction


def _direction_from_source_semantics(raw: dict[str, Any]) -> str:
    text = "".join(
        _explicit_source_column_value(raw, aliases)
        for aliases in (
            ("摘要描述", "交易摘要", "摘要", "摘要/附言"),
            ("交易附言", "附言", "用途", "备注"),
        )
    )
    income = any(marker in text for marker in ("转入", "收入", "入账", "入息", "收款", "贷方"))
    expense = any(marker in text for marker in ("转出", "支出", "出账", "付款", "支付", "借方"))
    if income != expense:
        return "income" if income else "expense"
    return ""


def _record_source_order(records: list[dict[str, Any]]) -> str:
    dates = [str((record.get("normalized") or {}).get("date") or "") for record in records]
    valid_dates = len(dates) >= 2 and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in dates)
    date_increases = valid_dates and any(current > previous for previous, current in zip(dates, dates[1:]))
    date_decreases = valid_dates and any(current < previous for previous, current in zip(dates, dates[1:]))
    sequence_values: list[int] = []
    for record in records:
        value = str((record.get("normalized") or {}).get("sequence_no") or "").strip()
        if re.fullmatch(r"\d{1,9}", value):
            sequence_values.append(int(value))
    if len(sequence_values) == len(records) and len(sequence_values) >= 2:
        deltas = [current - previous for previous, current in zip(sequence_values, sequence_values[1:])]
        if all(delta > 0 for delta in deltas) and not date_decreases:
            return "forward"
        if all(delta < 0 for delta in deltas) and not date_increases:
            return "reverse"

    if valid_dates:
        if all(current >= previous for previous, current in zip(dates, dates[1:])):
            return "forward"
        if all(current <= previous for previous, current in zip(dates, dates[1:])):
            return "reverse"
    return "forward"


def _direction_from_balance(previous_balance: float | None, amount: float, balance: float) -> str:
    if previous_balance is None:
        return ""
    if abs(previous_balance + amount - balance) <= 0.01:
        return "income"
    if abs(previous_balance - amount - balance) <= 0.01:
        return "expense"
    return ""


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
