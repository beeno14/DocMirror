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

import json
import math
import re
import unicodedata
from dataclasses import replace
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.plugins._base.standardizer import normalize_amount, normalize_timestamp
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.header_resolve import (
    detect_headers,
    has_split_debit_credit_headers,
    normalize_bank_matching_text,
    normalize_header_cell,
)
from docmirror.plugins.bank_statement.institution import match_institution, normalize_table_headers
from docmirror.plugins.bank_statement.row_extract import (
    extract_all_tables,
    extract_logical_rows_with_provenance,
    extract_rows_from_header,
    row_has_transaction_data,
)
from docmirror.plugins.bank_statement.wide_table_recovery import (
    _validated_native_source_repair_working_map,
    is_footer_or_total_row,
)

PARSER_ID = "grid_standard"
STYLE_ID = "grid_standard"
_NATIVE_SOURCE_RAW_JSON_COLUMN = "_source_raw_json"
_NATIVE_SOURCE_REPAIR_JSON_COLUMN = "_source_repair_json"
_NATIVE_SOURCE_REPAIR_KIND = "adjacent_summary_signed_amount_spill"
_NATIVE_SOURCE_REPAIR_KEYS = frozenset(
    {
        "kind",
        "summary_header",
        "amount_header",
        "summary_prefix",
        "source_summary",
        "source_amount",
        "working_summary",
        "working_amount",
        "working_transform",
    }
)

_SPLIT_DEBIT_KEYS = ("支出", "支出金额", "借方发生额", "借方", "转出金额")
_SPLIT_CREDIT_KEYS = ("收入", "收入金额", "贷方发生额", "贷方", "转入金额")
_DEDICATED_DIRECTION_KEYS = (
    "收/支",
    "支/收",
    "收支",
    "方向",
    "交易方向",
    "收入/支出",
    "月收/支",
    "借贷",
    "借/贷",
    "借贷标志",
    "Dc Flg",
)
_CLASSIFICATION_DIRECTION_KEYS = ("交易类别", "交易类型")
_DIRECTION_KEYS = (*_DEDICATED_DIRECTION_KEYS, *_CLASSIFICATION_DIRECTION_KEYS)
_GENERIC_SIGNED_AMOUNT_KEYS = ("交易金额", "金额", "发生额", "Amount")
_MONEY_PREFIX_RE = re.compile(r"^[^\d+-]*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)")
_STRICT_SOURCE_MONEY_RE = re.compile(
    r"^(?:CNY|RMB|人民币|[¥￥])?"
    r"(?P<amount>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    r"(?:元)?$",
    re.IGNORECASE,
)
_COUNTERPARTY_KEYS = (
    "对方户名",
    "对方账户名",
    "对方名称",
    "对手信息",
    "对手名称",
    "交易对方",
    "交易对手",
    "Counterparty Name",
    "counter_party",
)
_COUNTER_ACCOUNT_KEYS = ("对方账户", "对方账号", "counter_account")
_ACCOUNT_BANK_COMPOUND_KEYS = ("对方账户/对方银行", "对方账号/对方银行")
_COMPOUND_COUNTERPARTY_KEYS = ("交易对手信息", "对方账号与户名", "对方户名/账号")
_COMPOUND_COUNTERPARTY_NUMBER_RE = re.compile(r"(?<!\d)([0-9*＊]{6,32})(?!\d)")
_SHORT_MONTH_DAY_RE = re.compile(r"^(?P<month>\d{2})[-/](?P<day>\d{2})$")
_STATEMENT_PERIOD_RE = re.compile(
    r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?:\d{1,2}日)?"
    r"(?:\s*[-~至]\s*20\d{2}年\d{1,2}月\d{1,2}日)?"
)
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
_INCOME_ONLY_STATEMENT_TITLES = ("中国建设银行个人活期账户收入交易明细",)
_CSCB_SOURCE_HEADERS = frozenset({"交易日期", "交易金额", "账户余额", "对方户名", "对方账号", "摘要/备注", "编号"})
_BOC_SOURCE_HEADERS = frozenset(
    {
        "序号",
        "记账日",
        "起息日",
        "交易类型",
        "凭证",
        "凭证号码/业务编号/用途/摘要",
        "借方发生额",
        "贷方发生额",
        "余额",
        "机构/柜员/流水",
        "备注",
    }
)
_BOJS_SOURCE_HEADERS = frozenset(
    {
        "序号",
        "摘要/附言",
        "币别",
        "交易日期",
        "交易类型",
        "交易金额",
        "账户余额",
        "对方账号",
        "对方户名",
    }
)
_BOJS_REFERENCE_RE = re.compile(r"[0-9S]{6,31}", re.IGNORECASE)
_PAB_BILINGUAL_SOURCE_HEADER_SIGNATURES = (
    # Native PDF tables retain the bilingual labels as stacked text.  The
    # signature helper removes whitespace but deliberately does not invent a
    # slash that is absent from the source.
    frozenset(
        {
            "序号No.",
            "交易日期Date",
            "交易金额TransactionAmount",
            "余额Balance",
            "交易地点TradingPlace",
            "摘要Remark",
            "备注Notes",
        }
    ),
    # Pipe/OCR projections may losslessly serialize the same stacked header
    # with an explicit separator.  Keep this as a second exact representation,
    # not a looser keyword match.
    frozenset(
        {
            "序号/No.",
            "交易日期/Date",
            "交易金额/TransactionAmount",
            "余额/Balance",
            "交易地点/TradingPlace",
            "摘要/Remark",
            "备注/Notes",
        }
    ),
)
_DATE_ONLY_SUMMARY_LAYOUT_HEADERS = frozenset(
    {"序号", "记账日期", "交易金额", "账户余额", "摘要描述", "对方户名"}
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
    if compact_header in compact_aliases or bool(compact_aliases.intersection(header_parts)):
        return True
    # Units and harmless parenthetical qualifiers are part of the physical
    # header, not its semantic role (for example ``支出（元）``).  Strip only a
    # terminal qualifier and keep exact role matching otherwise so debit and
    # credit columns remain distinct.
    without_qualifier = re.sub(r"[（(][^（）()]{1,12}[）)]$", "", compact_header)
    return without_qualifier in compact_aliases


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


def _exact_source_column_value(raw_txn: dict[str, str], aliases: tuple[str, ...]) -> str:
    """Return only a whole-header match, never a stacked-header component."""
    compact_aliases = {re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(alias or ""))).lower() for alias in aliases}
    for raw_header, value in raw_txn.items():
        compact_header = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(raw_header or ""))).lower()
        if compact_header in compact_aliases:
            return str(value or "").strip()
    return ""


def _is_bounded_direction_label(value: str) -> bool:
    """Return whether a classification cell contains only a direction label."""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).strip()
    if compact in {"0", "1", "收", "支", "收入", "支出", "转入", "转出", "收人"}:
        return True
    return bool(
        re.fullmatch(
            r"(?:借(?:方)?(?:[/／()（）-]?Dr)?|贷(?:方)?(?:[/／()（）-]?Cr)?|"
            r"Dr(?:[/／()（）-]?借(?:方)?)?|Cr(?:[/／()（）-]?贷(?:方)?)?)",
            compact,
            re.IGNORECASE,
        )
    )


def _explicit_direction_source_value(raw_txn: dict[str, Any]) -> str:
    """Prefer a dedicated B3 direction column over transaction classification.

    ``交易类型`` and ``交易类别`` are useful compatibility fallbacks only when
    the entire source value is a bounded direction label.  A business name such
    as ``贷款到期归还`` must never outrank a later explicit ``借贷=借 Dr`` cell.
    """
    for header, value in raw_txn.items():
        if _header_matches_aliases(header, _DEDICATED_DIRECTION_KEYS):
            explicit = str(value or "").strip()
            if explicit:
                return explicit
    for header, value in raw_txn.items():
        if not _header_matches_aliases(header, _CLASSIFICATION_DIRECTION_KEYS):
            continue
        fallback = str(value or "").strip()
        if _is_bounded_direction_label(fallback):
            return fallback
    return ""


def _source_header_signature(raw_txn: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(header or "")))
        for header in raw_txn
        if not str(header).startswith("_")
    )


def _is_pab_bilingual_source_layout(raw_txn: dict[str, Any]) -> bool:
    source_headers = _source_header_signature(raw_txn)
    return any(signature.issubset(source_headers) for signature in _PAB_BILINGUAL_SOURCE_HEADER_SIGNATURES)


def _has_distinct_own_counter_account_headers(raw_txn: dict[str, Any]) -> bool:
    """Return whether exact source headers disambiguate own and counter accounts."""
    source_headers = _source_header_signature(raw_txn)
    return "账号" in source_headers and bool({"对方账号", "对方账户"}.intersection(source_headers))


def _enforce_distinct_account_header_roles(
    raw_txn: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    """Keep exact own/counter account columns in their declared source roles."""
    if not _has_distinct_own_counter_account_headers(raw_txn):
        return

    own_account = _exact_source_column_value(raw_txn, ("账号",))
    normalized["own_account"] = _clean_account(own_account) if own_account else ""

    # Clear the base mapper's permissive substring copy from bare ``账号`` while
    # preserving an independently declared sub-account column when present.
    sub_account = _exact_source_column_value(raw_txn, ("子账号", "子账户"))
    normalized["sub_account"] = _clean_account(sub_account) if sub_account else ""

    counter_account = _explicit_source_column_value(raw_txn, _COUNTER_ACCOUNT_KEYS)
    embedded = _decompose_account_with_trailing_party(counter_account)
    if embedded:
        normalized["counter_account"] = embedded["counter_account"]
        if not str(normalized.get("counter_party") or "").strip():
            normalized["counter_party"] = embedded["counter_party"]
    else:
        # The distinct header is direct role evidence.  Therefore a blank cell
        # is an authoritative source-null, not permission to borrow the
        # statement-owned account from the shorter fuzzy header match.
        normalized["counter_account"] = _clean_account(counter_account) if counter_account else ""


def _pab_labelled_fund_transfer_reference(raw_txn: dict[str, Any]) -> str:
    """Return a labelled reference only under the complete bilingual source layout."""
    if not _is_pab_bilingual_source_layout(raw_txn):
        return ""
    note = _exact_source_column_value(raw_txn, ("备注\nNotes", "备注Notes", "备注/Notes"))
    return _labelled_fund_transfer_reference(note) if note else ""


def _normalized_temporal_header_view(raw_txn: dict[str, str]) -> dict[str, str]:
    """Add canonical temporal keys without changing the retained raw row.

    Some digital statements use CJK compatibility glyphs or line wrapping in
    otherwise normal headers (for example ``交易⽇期`` or ``交易时\n间``).
    Header detection already canonicalizes those representations, so temporal
    normalization must use the same view.  The source dictionary remains the
    lossless ``raw`` payload; the aliases below exist only in this derived
    matching view.
    """

    view = dict(raw_txn)
    for raw_header, value in raw_txn.items():
        if str(raw_header).startswith("_"):
            continue
        source_header = str(raw_header or "")
        compatibility_header = normalize_bank_matching_text(source_header)
        matching_header = normalize_header_cell(compatibility_header)
        if matching_header in {
            "交易日期",
            "交易时间",
            "记账日期",
            "会计日期",
            "起息日",
            "起息日期",
        }:
            view.setdefault(matching_header, value)
    return view


def _normalize_with_temporal_context(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any]:
    matching_view = _normalized_temporal_header_view(raw_txn)
    short_date = _explicit_source_column_value(matching_view, _DATE_COLUMN_ALIASES)
    if materialized_date := _materialize_page_scoped_short_date(raw_txn, short_date):
        matching_view["交易日期"] = materialized_date
    combined_timestamp = _combine_separate_date_time(matching_view)
    if not combined_timestamp:
        return plugin._normalize(matching_view)

    prepared: dict[str, str] = {"交易时间": combined_timestamp}
    prepared.update(
        (key, value) for key, value in matching_view.items() if not _header_matches_aliases(key, _TIME_COLUMN_ALIASES)
    )
    return plugin._normalize(prepared)


def _materialize_page_scoped_short_date(raw_txn: dict[str, Any], value: str) -> str:
    """Materialize ``MM-DD`` only from an explicit same-page statement period."""
    match = _SHORT_MONTH_DAY_RE.fullmatch(re.sub(r"\s+", "", str(value or "")))
    if match is None:
        return ""
    scope = str(raw_txn.get("_source_page_scope_text") or "")
    periods = list(_STATEMENT_PERIOD_RE.finditer(scope))
    if not periods:
        return ""
    month = int(match.group("month"))
    day = int(match.group("day"))
    scoped_periods = {
        (int(period.group("year")), int(period.group("month")))
        for period in periods
        if int(period.group("month")) == month
    }
    if len(scoped_periods) != 1:
        return ""
    year, _month = scoped_periods.pop()
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def _apply_document_scope_direction(
    normalized: dict[str, Any],
    raw_txn: dict[str, Any],
) -> None:
    """Apply direction only for an exact, source-proven filtered statement."""
    scope_text = re.sub(
        r"\s+",
        "",
        unicodedata.normalize("NFKC", str(raw_txn.get("_document_scope_text") or "")),
    )
    if not any(title in scope_text for title in _INCOME_ONLY_STATEMENT_TITLES):
        return
    explicit = _normalize_direction_text(_explicit_direction_source_value(raw_txn))
    # An explicit source contradiction stays visible instead of being silently
    # overwritten; candidate semantic gates can then fail closed.
    normalized["direction"] = explicit if explicit == "expense" else "income"


def _normalize_directional_payer_payee(
    raw_txn: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    """Map payer/payee columns from the statement account's perspective.

    For an incoming row the payer is the counterparty; for an outgoing row the
    payee is the counterparty. The opposite side identifies the statement
    account. Missing selected-side facts stay source-null, and an exact self pair
    is never presented as a counterparty.
    """
    if normalized.get("counter_account") or normalized.get("counter_party"):
        return
    direction = str(normalized.get("direction") or "")
    if direction == "income":
        account_aliases = ("付款账号", "付款账户")
        party_aliases = ("付款账户名", "付款户名")
        own_account_aliases = ("收款账号", "收款账户")
        own_party_aliases = ("收款账户名", "收款户名")
    elif direction == "expense":
        account_aliases = ("收款账号", "收款账户")
        party_aliases = ("收款账户名", "收款户名")
        own_account_aliases = ("付款账号", "付款账户")
        own_party_aliases = ("付款账户名", "付款户名")
    else:
        return

    account = _clean_account(_explicit_source_column_value(raw_txn, account_aliases))
    party = _clean_wrapped_text(_explicit_source_column_value(raw_txn, party_aliases))
    if not account or not party:
        return
    own_account = _clean_account(_explicit_source_column_value(raw_txn, own_account_aliases))
    own_party = _clean_wrapped_text(_explicit_source_column_value(raw_txn, own_party_aliases))
    if account == own_account and party == own_party:
        return
    normalized["counter_account"] = account
    normalized["counter_party"] = party


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
    source_headers = _source_header_signature(raw_txn)
    is_cscb_layout = _CSCB_SOURCE_HEADERS.issubset(source_headers)
    is_boc_layout = _BOC_SOURCE_HEADERS.issubset(source_headers)
    is_bojs_layout = _BOJS_SOURCE_HEADERS.issubset(source_headers)
    is_pab_bilingual_layout = _is_pab_bilingual_source_layout(raw_txn)
    is_date_only_summary_layout = _DATE_ONLY_SUMMARY_LAYOUT_HEADERS.issubset(source_headers)
    if is_cscb_layout:
        compound_summary = _exact_source_column_value(raw_txn, ("摘要/备注",))
        reference = _exact_source_column_value(raw_txn, ("编号",))
        normalized["summary"] = _clean_wrapped_text(compound_summary) if compound_summary else ""
        normalized["note"] = ""
        normalized["reference"] = re.sub(r"\s+", "", reference) if reference else ""

    if is_boc_layout:
        _normalize_boc_business_columns(raw_txn, normalized)

    if is_bojs_layout:
        _normalize_bojs_business_columns(raw_txn, normalized)

    if not is_cscb_layout and not is_boc_layout and not is_bojs_layout:
        summary = _explicit_source_column_value(raw_txn, ("摘要描述", "交易摘要", "摘要"))
        if summary:
            normalized["summary"] = _clean_wrapped_text(summary)
            unique_business_id = re.search(r"业务唯一编号\s*([0-9A-Z-]{6,40})", summary, re.I)
            if unique_business_id and not normalized.get("reference"):
                normalized["reference"] = unique_business_id.group(1)

    purpose = _explicit_source_column_value(raw_txn, ("用途", "交易用途"))
    if purpose:
        normalized["purpose"] = _clean_wrapped_text(purpose)
    remittance_note = _explicit_source_column_value(raw_txn, ("交易附言", "附言"))
    if remittance_note:
        normalized["remittance_note"] = _clean_wrapped_text(remittance_note)
    note = (
        _exact_source_column_value(raw_txn, ("备注\nNotes", "备注Notes", "备注/Notes"))
        if is_pab_bilingual_layout
        else _explicit_source_column_value(raw_txn, ("备注", "Remarks", "Notes"))
    )
    if note and not is_cscb_layout and not is_boc_layout and not is_bojs_layout:
        cleaned_note = _clean_wrapped_text(note)
        if cleaned_note in {"/", "／", "-", "--"}:
            normalized["note"] = ""
        elif re.match(r"^用途\s*[:：]", cleaned_note):
            normalized["purpose"] = cleaned_note
        else:
            normalized["note"] = cleaned_note
        if cleaned_note not in {"/", "／", "-", "--"}:
            _decompose_labelled_business_note(cleaned_note, normalized)
            if not str(normalized.get("business_system_reference") or "").strip():
                transfer_reference = _pab_labelled_fund_transfer_reference(raw_txn)
                if transfer_reference:
                    normalized["business_system_reference"] = transfer_reference

    compound_counterparty = _exact_source_column_value(raw_txn, _COMPOUND_COUNTERPARTY_KEYS)
    compound_fields = _decompose_compound_counterparty(compound_counterparty)
    ambiguous_compound = bool(compound_counterparty and not compound_fields)
    if ambiguous_compound:
        # The base registry may have fuzzily copied the compound source cell
        # into a business role.  An unproven decomposition must stay raw-only.
        normalized["counter_party"] = ""
        normalized["counter_account"] = ""
        normalized["counter_bank_name"] = ""
        normalized["counter_bank_code"] = ""

    counter_account = _explicit_source_column_value(raw_txn, _COUNTER_ACCOUNT_KEYS)
    embedded_account_party = _decompose_account_with_trailing_party(counter_account)
    if embedded_account_party:
        normalized["counter_account"] = embedded_account_party["counter_account"]
    elif counter_account:
        normalized["counter_account"] = _clean_account(counter_account)
    elif compound_fields:
        normalized["counter_account"] = compound_fields["counter_account"]

    account_bank = _exact_source_column_value(raw_txn, _ACCOUNT_BANK_COMPOUND_KEYS)
    account_bank_fields = _decompose_account_and_bank(account_bank)
    if account_bank:
        if account_bank_fields:
            normalized["counter_account"] = account_bank_fields["counter_account"]
            normalized["counter_bank_name"] = account_bank_fields["counter_bank_name"]
        else:
            # The base registry must not label an unsplit compound source cell
            # as an account.  Keep it raw-only until the adjacent exact party
            # cell proves that its trailing token is the account and therefore
            # that this remaining cell is the bank portion.
            normalized["counter_account"] = ""
            normalized["counter_bank_name"] = ""

    counter_party = _explicit_source_column_value(
        raw_txn,
        (
            "对方户名",
            "对方账户名",
            "对方名称",
            "对手信息",
            "对手名称",
            "交易对方",
            "Counterparty Name",
        ),
    )
    if counter_party and not ambiguous_compound:
        cleaned_party = _clean_wrapped_text(counter_party)
        if _looks_like_counterparty_contamination(cleaned_party):
            # Keep the lossless source cell in ``raw``, but never expose a
            # neighboring transaction or page furniture as a business party.
            normalized["counter_party"] = ""
        else:
            cleaned_party, embedded_account = _split_embedded_counter_account(cleaned_party)
            if embedded_account and not account_bank_fields:
                normalized["counter_account"] = embedded_account
                if account_bank:
                    # Some native PDFs place the account suffix in the party
                    # atom across the proven column boundary and leave only
                    # the bank name in the exact account/bank column.  The two
                    # source cells together fully consume all three business
                    # roles, so no document-wide alias or row inference is
                    # needed.
                    normalized["counter_bank_name"] = _clean_wrapped_text(account_bank)
            elif embedded_account and not normalized.get("counter_account"):
                normalized["counter_account"] = embedded_account
            compact_party = re.sub(r"\s+", "", cleaned_party)
            if re.fullmatch(r"[0-9*＊]{6,32}", compact_party):
                normalized["counter_account"] = compact_party
                normalized["counter_party"] = ""
            elif compact_party in {"--", "-"}:
                normalized["counter_party"] = ""
            else:
                normalized["counter_party"] = cleaned_party
    elif embedded_account_party:
        normalized["counter_party"] = embedded_account_party["counter_party"]
    elif compound_fields:
        normalized["counter_party"] = compound_fields["counter_party"]

    counter_bank = _explicit_source_column_value(
        raw_txn,
        ("对方行名", "对手机构", "对方开户行", "对方银行名称", "Counterparty Institution"),
    )
    if counter_bank:
        normalized["counter_bank_name"] = _clean_wrapped_text(counter_bank)
    elif account_bank_fields:
        normalized["counter_bank_name"] = account_bank_fields["counter_bank_name"]
    elif compound_fields:
        normalized["counter_bank_name"] = compound_fields["counter_bank_name"]
        normalized["counter_bank_code"] = compound_fields["counter_bank_code"]

    distinct_business_fields = {
        "transaction_location": ("交易地点", "交易场所", "Trading Place"),
        "currency": ("币种", "币别", "货币", "Currency"),
        "cash_remittance": ("钞汇", "现/转", "现金/转账", "现转标志", "现金/转账标志"),
        "voucher_type": ("凭证种类", "凭证类型"),
        "voucher_number": ("凭证号", "凭证号码"),
        "transaction_code": ("交易代码", "业务代码"),
        "transaction_institution": ("交易机构", "经办机构"),
        "teller_id": ("柜员号", "柜员"),
        "posting_date": ("记账日期", "会计日期", "Accounting Date"),
        "sequence_no": ("序号", "序 号", "交易序号", "Sequence"),
        "transaction_name": ("交易名称", "交易描述", "交易类型", "Transaction Name"),
        "reference": ("交易流水号", "流水号", "电子回单编号", "Reference", "Reference No."),
    }
    for field, aliases in distinct_business_fields.items():
        if is_bojs_layout and field == "transaction_name":
            continue
        value = _explicit_source_column_value(raw_txn, aliases)
        if value:
            normalized[field] = (
                re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
                if field == "voucher_number"
                else _clean_wrapped_text(value)
            )
    # This source signature declares one transaction date only. The base
    # profile alias is useful for header discovery but must not duplicate the
    # same source cell into a second posting-date business fact.
    if is_date_only_summary_layout:
        normalized["posting_date"] = ""
    _enforce_distinct_account_header_roles(raw_txn, normalized)


def _decompose_bojs_summary(
    raw_txn: dict[str, Any],
    *,
    normalize_values: bool = True,
) -> dict[str, Any]:
    """Split the exact four-part BOJS memo grammar without guessing.

    The full source cell remains a distinct ``business_detail``.  Promoted
    fields are substrings of that cell and are emitted only when both the code
    family and transaction-kind grammar are proven.
    """
    compound = _exact_source_column_value(raw_txn, ("摘要/附言",))
    if not compound:
        return {}
    fields: dict[str, Any] = {}
    parts = str(compound).split("#")
    if len(parts) != 4:
        fields["summary"] = _clean_wrapped_text(compound) if normalize_values else compound
        return fields

    compact = [re.sub(r"\s+", "", unicodedata.normalize("NFKC", part)) for part in parts]
    code, reference, transaction_name, summary = compact
    wl_code = code in {"0WL", "_0WL"}
    unionpay_code = code == "1银联"
    grammar_valid = (
        bool(_BOJS_REFERENCE_RE.fullmatch(reference))
        and bool(summary)
        and (
            (wl_code and transaction_name in {"WL协议", "WL付款", "WL退款"})
            or (unionpay_code and transaction_name == "银联贷记")
        )
    )
    if not grammar_valid:
        fields["summary"] = _clean_wrapped_text(compound) if normalize_values else compound
        return fields

    fields["business_detail"] = _clean_wrapped_text(compound) if normalize_values else compound

    if normalize_values:
        fields.update(
            {
                "transaction_code": code.removeprefix("_"),
                "reference": reference,
                "transaction_name": transaction_name,
                "summary": _clean_wrapped_text(parts[3]),
            }
        )
    else:
        fields.update(
            {
                "transaction_code": parts[0],
                "reference": parts[1],
                "transaction_name": parts[2],
                "summary": parts[3],
            }
        )
    return fields


def _normalize_bojs_business_columns(raw_txn: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Apply source roles declared by the exact BOJS statement header."""
    compound = _exact_source_column_value(raw_txn, ("摘要/附言",))
    fields = _decompose_bojs_summary(raw_txn)
    normalized.update(fields)
    if compound and "summary" not in normalized:
        normalized["summary"] = _clean_wrapped_text(compound)
    normalized["direction"] = _normalize_direction_text(
        _exact_source_column_value(raw_txn, ("交易类型",))
    )
    normalized["currency"] = _clean_wrapped_text(_exact_source_column_value(raw_txn, ("币别",)))
    normalized["timestamp"] = ""
    # ``交易类型`` is the source direction in this signature.  A transaction
    # name exists only when the compound memo proves one.
    if "transaction_name" not in fields:
        normalized["transaction_name"] = ""


def _decompose_boc_business_columns(
    raw_txn: dict[str, Any],
    *,
    normalize_values: bool = True,
) -> dict[str, Any]:
    """Return deterministic source-backed roles for the exact BOC layout."""
    fields: dict[str, Any] = {}
    value_date = _exact_source_column_value(raw_txn, ("起息日",))
    if value_date:
        if normalize_values:
            compact_value_date = re.sub(r"\s+", "", value_date)
            if match := re.fullmatch(r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})", compact_value_date):
                parsed = f"20{match.group('year')}-{match.group('month')}-{match.group('day')}"
            else:
                parsed = normalize_timestamp(value_date)
            fields["value_date"] = parsed[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", parsed) else parsed
        else:
            fields["value_date"] = value_date

    transaction_name = _exact_source_column_value(raw_txn, ("交易类型",))
    if transaction_name:
        fields["transaction_name"] = _clean_wrapped_text(transaction_name)

    institution_serial = _exact_source_column_value(raw_txn, ("机构/柜员/流水",))
    if institution_serial:
        parts = [_clean_wrapped_text(part) for part in re.split(r"[/／]", institution_serial) if part.strip()]
        if len(parts) == 3:
            fields["transaction_institution"] = parts[0]
            fields["teller_id"] = parts[1]
            fields["bank_serial"] = re.sub(r"\s+", "", parts[2])

    compound = _exact_source_column_value(raw_txn, ("凭证号码/业务编号/用途/摘要",))
    fields["business_detail"] = _clean_wrapped_text(compound) if compound else ""
    if compound:
        parts = [_clean_wrapped_text(part) for part in re.split(r"[/／]", compound) if part.strip()]
        if len(parts) >= 2 and _looks_like_boc_business_reference(parts[0]):
            fields["reference"] = re.sub(r"\s+", "", parts[0])
            tail = parts[1:]
            if len(tail) >= 2 and _looks_like_boc_system_reference(tail[-1]):
                fields["business_system_reference"] = re.sub(r"\s+", "", tail.pop())
            # A single slash tail can itself contain an account, tax ID and
            # authority text.  Keep that source context intact unless it is a
            # concise, high-specificity business purpose. With several slash
            # parts, the final human phrase is an explicitly delimited purpose.
            if (
                tail
                and (len(tail) >= 2 or _is_concise_boc_business_purpose(tail[-1]))
                and _looks_like_boc_business_purpose(tail[-1])
            ):
                fields["purpose"] = tail[-1]
                tail.pop()
            if tail:
                fields["business_context"] = " / ".join(tail)
        elif len(parts) >= 2 and _looks_like_boc_system_reference(parts[-1]):
            fields["purpose"] = parts[0]
            fields["business_system_reference"] = re.sub(r"\s+", "", parts[-1])
        else:
            fields["purpose"] = _clean_wrapped_text(compound)

    note = _exact_source_column_value(raw_txn, ("备注",))
    if note:
        fields["note"] = _clean_wrapped_text(note)
        note_parts = [_clean_wrapped_text(part) for part in re.split(r"[/／]", note, maxsplit=1)]
        fields["counter_party"] = note_parts[0]
        counter_bank_name = note_parts[1] if len(note_parts) == 2 and _looks_like_bank_name(note_parts[1]) else ""
        if normalize_values and counter_bank_name:
            counter_bank_name = _complete_boc_counter_bank_name(
                counter_bank_name,
                counter_party=note_parts[0],
                business_detail=compound,
            )
        fields["counter_bank_name"] = counter_bank_name
    return fields


def _normalize_boc_business_columns(raw_txn: dict[str, Any], normalized: dict[str, Any]) -> None:
    """Decompose the exact BOC compound layout without duplicating roles."""
    boc_fields = _decompose_boc_business_columns(raw_txn)
    normalized.update(boc_fields)
    normalized["summary"] = ""
    normalized["purpose"] = str(boc_fields.get("purpose") or "")
    normalized.setdefault("business_context", "")
    normalized.setdefault("business_system_reference", "")
    # ``记账日`` is the transaction date in this issuer layout; it is not a
    # second posting-date fact. The base substring mapper otherwise duplicates
    # it into both roles.
    normalized["posting_date"] = ""


def _looks_like_boc_business_reference(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    return bool(re.fullmatch(r"(?=.*\d)[A-Z0-9]{8,64}|[A-Z]{4}[A-Z0-9 ]{8,64}", compact, re.I))


def _looks_like_boc_system_reference(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    return bool(re.fullmatch(r"(?:OBSS|GIRO)[A-Z0-9]{12,64}", compact, re.I))


def _looks_like_boc_business_purpose(value: str) -> bool:
    text = _clean_wrapped_text(value)
    return bool(re.search(r"[\u4e00-\u9fff]", text)) and not _looks_like_boc_system_reference(text)


def _is_concise_boc_business_purpose(value: str) -> bool:
    text = _clean_wrapped_text(value)
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 24 or len(re.findall(r"\d{6,}", compact)) > 0:
        return False
    return any(
        marker in compact
        for marker in (
            "货款",
            "往来",
            "工资",
            "服务费",
            "手续费",
            "转账",
            "保险",
            "税款",
            "退款",
            "报销",
            "备用金",
        )
    )


def _looks_like_bank_name(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    return any(marker in compact for marker in ("银行", "信用社", "信用合作", "农商行", "金库", "支库"))


def _complete_boc_counter_bank_name(
    bank_prefix: str,
    *,
    counter_party: str,
    business_detail: str,
) -> str:
    """Complete a printed bank prefix only from a unique same-row source fact."""
    prefix = _clean_wrapped_text(bank_prefix)
    candidates: set[str] = set()
    party = _clean_wrapped_text(counter_party)
    if len(party) > len(prefix) and party.startswith(prefix) and _looks_like_bank_name(party):
        candidates.add(party)
    for part in re.split(r"[/／]", str(business_detail or "")):
        candidate = _clean_wrapped_text(part)
        if len(candidate) > len(prefix) and candidate.startswith(prefix) and _looks_like_bank_name(candidate):
            candidates.add(candidate)
    return next(iter(candidates)) if len(candidates) == 1 else prefix


def _decompose_labelled_business_note(
    note: str,
    normalized: dict[str, Any],
) -> None:
    """Promote explicitly labelled business facts while preserving the note.

    The full source note remains in ``note``. Only high-specificity labels are
    decomposed, and an explicit source column already mapped to the target role
    always wins.
    """
    text = _clean_wrapped_text(note)
    if not text:
        return
    label_roles = (
        (("业务编号", "网银流水号", "业务标识号"), "reference"),
        (("用途",), "purpose"),
        (("附言",), "remittance_note"),
    )
    for labels, field in label_roles:
        if str(normalized.get(field) or "").strip():
            continue
        for label in labels:
            match = re.search(
                rf"(?:^|[;；])\s*{re.escape(label)}\s*[:：]\s*([^;；]+)",
                text,
            )
            if match is not None:
                value = _clean_wrapped_text(match.group(1)).strip(" ;；")
                if value:
                    normalized[field] = value
                break
def _labelled_fund_transfer_reference(note: str) -> str:
    match = re.search(
        r"资金划拨\s*[:：]\s*(MQ[0-9A-Z]{6,40})(?=[;；]|$)",
        _clean_wrapped_text(note),
        re.IGNORECASE,
    )
    return match.group(1) if match is not None else ""


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
    explicit_balance = _explicit_source_column_value(raw_txn, ("余额", "账户余额", "本次余额", "账面余额"))
    explicit_reference = _explicit_source_column_value(raw_txn, ("交易流水号", "流水号", "Reference"))
    merged_balance_reference = _exact_source_column_value(raw_txn, ("账户余额流水号", "余额流水号"))
    split_balance, split_reference = _decompose_merged_balance_reference(merged_balance_reference)
    balance = _normalize_monetary_cell(explicit_balance) if explicit_balance else split_balance
    if balance is not None:
        normalized["balance"] = float(balance)
    elif merged_balance_reference:
        # A collapsed source role that cannot be decomposed is not a balance.
        normalized.pop("balance", None)
    reference = explicit_reference or split_reference
    if reference:
        normalized["reference"] = re.sub(r"\s+", "", reference)

    compound_counterparty = _exact_source_column_value(raw_txn, _COMPOUND_COUNTERPARTY_KEYS)
    if not compound_counterparty and not str(normalized.get("counter_party", "") or "").strip():
        cp = _cell_value(
            raw_txn,
            *_registry_field_keys(plugin, "对方户名", _COUNTERPARTY_KEYS),
        )
        if cp and not _looks_like_counterparty_contamination(cp):
            normalized["counter_party"] = _clean_wrapped_text(cp)

    explicit_direction = _normalize_direction_text(_explicit_direction_source_value(raw_txn))
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
    direction_raw = _explicit_direction_source_value(raw_txn)
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
    # Direction is represented separately in the canonical schema.  Preserve
    # the signed source token in ``raw``/``canonical_raw`` while keeping the
    # normalized business amount as a non-negative magnitude, exactly as the
    # explicit-direction and split debit/credit paths do.
    normalized["amount"] = abs(float(amount))
    normalized["amount_cny"] = abs(float(amount))
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
    # Some corporate ledgers encode the dedicated 借/贷 flag numerically.
    # This interpretation is safe only after a direction-column lookup has
    # supplied the value; arbitrary numeric cells never reach this helper.
    if text == "1":
        return "income"
    if text == "0":
        return "expense"
    if text == "收":
        return "income"
    if text == "支":
        return "expense"
    if any(token in text for token in ("收入", "转入", "收人")):
        return "income"
    if any(token in text for token in ("支出", "转出", "支山", "支鼎", "攴出")):
        return "expense"
    if "贷" in text or re.search(r"\bCr\b", text, re.IGNORECASE):
        return "income"
    if "借" in text or re.search(r"\bDr\b", text, re.IGNORECASE):
        return "expense"
    return ""


def _source_provenanced_signed_amount(
    raw_txn: dict[str, Any],
    normalized: dict[str, Any],
    canonical_raw: dict[str, Any],
) -> tuple[float, str | None] | None:
    """Return a source-proven signed amount and any split-column direction."""
    if not all(isinstance(pool, dict) for pool in (raw_txn, normalized, canonical_raw)):
        return None

    try:
        normalized_amount = Decimal(str(normalized.get("amount")))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not normalized_amount.is_finite() or normalized_amount < 0:
        return None

    def matching_nonblank(aliases: tuple[str, ...]) -> list[str]:
        return [
            str(value).strip()
            for header, value in raw_txn.items()
            if _header_matches_aliases(header, aliases) and str(value or "").strip()
        ]

    income_values = matching_nonblank(_SPLIT_CREDIT_KEYS)
    expense_values = matching_nonblank(_SPLIT_DEBIT_KEYS)
    generic_values = matching_nonblank(_GENERIC_SIGNED_AMOUNT_KEYS)
    if len(income_values) > 1 or len(expense_values) > 1 or len(generic_values) > 1:
        return None
    income_nonblank = bool(income_values)
    expense_nonblank = bool(expense_values)

    if income_nonblank or expense_nonblank:
        if income_nonblank == expense_nonblank or generic_values:
            return None
        split_direction = "income" if income_nonblank else "expense"
        source_amount_raw = income_values[0] if income_nonblank else expense_values[0]
    else:
        split_direction = None
        if len(generic_values) != 1:
            return None
        source_amount_raw = generic_values[0]

    canonical_amount_raw = canonical_raw.get("amount")
    if (
        isinstance(canonical_amount_raw, (bool, list, tuple, dict))
        or canonical_amount_raw in (None, "")
    ):
        return None
    source_text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", source_amount_raw)).strip()
    canonical_text = re.sub(
        r"\s+",
        "",
        unicodedata.normalize("NFKC", str(canonical_amount_raw)),
    ).strip()
    if source_text != canonical_text:
        return None
    source_match = _STRICT_SOURCE_MONEY_RE.fullmatch(source_text)
    canonical_match = _STRICT_SOURCE_MONEY_RE.fullmatch(canonical_text)
    if source_match is None or canonical_match is None:
        return None
    try:
        source_amount = Decimal(source_match.group("amount").replace(",", ""))
        canonical_amount = Decimal(canonical_match.group("amount").replace(",", ""))
    except InvalidOperation:
        return None
    if source_amount != canonical_amount:
        return None
    if abs(abs(source_amount) - normalized_amount) > Decimal("0.005"):
        return None
    return float(source_amount), split_direction


def source_provenanced_signed_amount(
    raw_txn: dict[str, Any],
    normalized: dict[str, Any],
    canonical_raw: dict[str, Any],
) -> float | None:
    """Return an exact source/canonical signed amount, without inferring its side."""
    fact = _source_provenanced_signed_amount(raw_txn, normalized, canonical_raw)
    return fact[0] if fact is not None else None


def source_owned_signed_directional_amount(
    raw_txn: dict[str, Any],
    normalized: dict[str, Any],
    canonical_raw: dict[str, Any],
) -> tuple[str, float] | None:
    """Return a signed reversal fact only with independently owned direction."""
    amount_fact = _source_provenanced_signed_amount(raw_txn, normalized, canonical_raw)
    if amount_fact is None:
        return None
    signed_amount, split_direction = amount_fact

    normalized_direction = str(normalized.get("direction") or "").strip()
    if normalized_direction not in {"income", "expense"}:
        return None
    explicit_direction_value = _explicit_direction_source_value(raw_txn)
    explicit_direction = ""
    if explicit_direction_value:
        if not _is_bounded_direction_label(explicit_direction_value):
            return None
        explicit_direction = _normalize_direction_text(explicit_direction_value)
        if explicit_direction not in {"income", "expense"}:
            return None

    source_direction = split_direction or explicit_direction
    if not source_direction or source_direction != normalized_direction:
        return None
    if split_direction and explicit_direction and split_direction != explicit_direction:
        return None

    canonical_direction_raw = str(canonical_raw.get("direction") or "").strip()
    if canonical_direction_raw:
        canonical_direction = (
            canonical_direction_raw
            if canonical_direction_raw in {"income", "expense"}
            else _normalize_direction_text(canonical_direction_raw)
        )
        if canonical_direction != source_direction:
            return None
    return source_direction, signed_amount


def _clean_wrapped_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff(（])", "", text)
    return re.sub(r"(?<=[)）])\s+(?=[\u4e00-\u9fff])", "", text)


def _clean_account(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[0-9*\s＊]+", text):
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text)


def _decompose_merged_balance_reference(value: str) -> tuple[float | None, str]:
    """Split only the proven ``money(2dp)+serial`` collapsed source form."""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).upper()
    if not compact:
        return None, ""
    match = re.fullmatch(
        r"(?P<balance>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})(?P<reference>[0-9A-Z]{10,40})",
        compact,
    )
    if match is None:
        return None, ""
    balance = normalize_amount(match.group("balance"))
    return balance, match.group("reference") if balance is not None else ""


def _decompose_compound_counterparty(value: str) -> dict[str, str]:
    """Split a proven party/account/bank/code source cell without guessing.

    Two issuer layouts place all four roles in one physical column.  We only
    decompose when the source contains exactly two bounded identifier tokens:
    the first is the counter-account and the final token is the clearing/bank
    code.  Ambiguous cells remain available losslessly in ``raw`` and are not
    assigned invented business roles.
    """
    text = _clean_wrapped_text(str(value or ""))
    if not text:
        return {}
    matches = list(_COMPOUND_COUNTERPARTY_NUMBER_RE.finditer(text))
    if len(matches) == 1 and matches[0].span() == (0, len(text)):
        # Under a proven compound header, an exact identifier-only cell is an
        # account with a source-null party, not an unknown party name.
        return {
            "counter_party": "",
            "counter_account": _clean_account(matches[0].group(1)),
            "counter_bank_name": "",
            "counter_bank_code": "",
        }
    if len(matches) != 2:
        # CCB's explicitly slash-delimited ``account/name`` form contains one
        # identifier and is equally deterministic.
        slash_match = re.fullmatch(r"\s*([0-9A-Za-z*＊]{6,40})\s*[/／]\s*(.+?)\s*", text)
        if slash_match is not None and any(char.isdigit() for char in slash_match.group(1)):
            return {
                "counter_party": _clean_wrapped_text(slash_match.group(2)),
                "counter_account": _clean_account(slash_match.group(1)),
                "counter_bank_name": "",
                "counter_bank_code": "",
            }
        # CMBC prints the inverse, equally explicit ``party/account`` form.
        reverse_slash_match = re.fullmatch(r"\s*(.+?)\s*[/／]\s*([0-9*＊]{6,32})\s*", text)
        if reverse_slash_match is None:
            return {}
        return {
            "counter_party": _clean_wrapped_text(reverse_slash_match.group(1)),
            "counter_account": _clean_account(reverse_slash_match.group(2)),
            "counter_bank_name": "",
            "counter_bank_code": "",
        }

    first, second = matches
    party = text[: first.start()].strip(" /／,，;；")
    bank = text[first.end() : second.start()].strip(" /／,，;；")
    if not party or not bank:
        return {}
    return {
        "counter_party": _clean_wrapped_text(party),
        "counter_account": _clean_account(first.group(1)),
        "counter_bank_name": _clean_wrapped_text(bank),
        "counter_bank_code": _clean_account(second.group(1)),
    }


def _decompose_account_with_trailing_party(value: str) -> dict[str, str]:
    """Split an account/name pair proven within one dedicated account cell.

    Explicit slash-delimited values use the existing D2/D3 compound grammar.
    The separator-free form is accepted only when a numeric account is followed
    by a CJK name, which covers wrapped native rows without treating an
    alphanumeric account identifier as a party.
    """
    compound = _decompose_compound_counterparty(value)
    if compound.get("counter_account") and compound.get("counter_party"):
        return {
            "counter_account": compound["counter_account"],
            "counter_party": compound["counter_party"],
        }

    text = _clean_wrapped_text(str(value or ""))
    match = re.fullmatch(
        r"\s*(?P<account>[0-9*＊]{6,32})\s*(?P<party>[^0-9*＊]*[\u4e00-\u9fff][^0-9*＊]*)\s*",
        text,
    )
    if match is None:
        return {}
    party = _clean_wrapped_text(match.group("party")).strip(" /／,，;；")
    if not party:
        return {}
    return {
        "counter_account": _clean_account(match.group("account")),
        "counter_party": party,
    }


def _decompose_account_and_bank(value: str) -> dict[str, str]:
    """Split an exact combined account/bank source cell at its identifier."""
    text = _clean_wrapped_text(unicodedata.normalize("NFKC", str(value or "")))
    if not text:
        return {}

    # Prefer an explicit source separator.  A no-space split is allowed only
    # at the first CJK bank-name glyph and only for an account-like ASCII token
    # containing at least one digit.  This supports PDF word atoms that were
    # concatenated during visual row reconstruction, including alphanumeric
    # payment accounts, without guessing a boundary inside ordinary prose.
    match = re.fullmatch(
        r"\s*(?P<account>[0-9A-Za-z*＊._-]{6,64})\s+(?P<bank>.+?)\s*",
        text,
    )
    if match is None:
        match = re.fullmatch(
            r"\s*(?P<account>[0-9A-Za-z*＊._-]{6,64})(?P<bank>[\u3400-\u9fff].+?)\s*",
            text,
        )
    if match is None or not re.search(r"\d", match.group("account")):
        return {}
    bank = _clean_wrapped_text(match.group("bank")).strip(" /／,，;；")
    if not bank:
        return {}
    return {
        "counter_account": _clean_account(match.group("account")),
        "counter_bank_name": bank,
    }


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
            short_date_row = bool(row and _SHORT_MONTH_DAY_RE.fullmatch(re.sub(r"\s+", "", str(row[0] or ""))))
            if not row_has_transaction_data(row, strict_first_col=False) and not short_date_row:
                continue
            txn: dict[str, str] = {}
            for idx, cell in enumerate(row):
                header = raw_headers[idx] if idx < len(raw_headers) else f"col_{idx}"
                txn[header] = str(cell or "").strip()
            income = _normalize_monetary_cell(_cell_value(txn, *_SPLIT_CREDIT_KEYS))
            expense = _normalize_monetary_cell(_cell_value(txn, *_SPLIT_DEBIT_KEYS))
            if float(income or 0) <= 0 and float(expense or 0) <= 0:
                continue
            if (
                short_date_row
                and _normalize_monetary_cell(_cell_value(txn, "余额", "账户余额", "本次余额", "账面余额")) is None
            ):
                continue
            transactions.append(txn)
    return transactions


def _with_internal_row_sources(
    transactions: list[dict[str, Any]],
    parse_result: Any | None = None,
) -> list[dict[str, Any]]:
    parse_result = _read_view(parse_result)
    physical_raw_rows = _physical_raw_row_geometry_sources(parse_result)
    inferred_sources = _infer_row_sources(
        transactions,
        parse_result,
        physical_raw_rows=physical_raw_rows,
    )
    for transaction in transactions:
        source = transaction.get("_source")
        if isinstance(source, dict) and _positive_int(source.get("source_page")) is not None:
            _augment_source_from_exact_physical_raw_row(
                transaction,
                source,
                physical_raw_rows,
            )
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
        _augment_source_from_exact_physical_raw_row(
            transaction,
            row_source,
            physical_raw_rows,
        )
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
    page_scopes = dict(_pdf_page_texts_from_provenance(parse_result)) if parse_result is not None else {}
    if full_text:
        for transaction in sourced:
            transaction.setdefault("_document_scope_text", full_text)
    for transaction in sourced:
        source_page = _transaction_source_page(transaction)
        if source_page is not None and page_scopes.get(source_page):
            transaction["_source_page_scope_text"] = page_scopes[source_page]
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
                if next_start is None:
                    # Never skip over an unlocated or overlapping adjacent row:
                    # doing so turns that row's business fields into a party.
                    continue
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
    # A same-row D2/D3 compound cell is authoritative.  Flattened page text
    # cannot improve it and must not invent a dedicated raw party column.
    if _exact_source_column_value(transaction, _COMPOUND_COUNTERPARTY_KEYS):
        return False
    counter_account = _cell_value(transaction, *_COUNTER_ACCOUNT_KEYS)
    if not counter_account:
        return False
    if _decompose_account_with_trailing_party(counter_account):
        return False
    counterparty = _cell_value(transaction, *_COUNTERPARTY_KEYS)
    return not counterparty or _looks_like_incomplete_counterparty(counterparty)


def _looks_like_incomplete_counterparty(value: str) -> bool:
    if _looks_like_counterparty_contamination(value):
        return True
    compact = _signature_value(value)
    if compact in {"入", "收", "收入", "出", "支", "限公司", "有限公司", "代收)", "代收）"}:
        return True
    if compact.startswith(("限公司", "代收)", "代收）")):
        return True
    fee_tail = "电子渠道跨行转账手续费收"
    if fee_tail in compact and not compact.startswith(("企业电子渠道", "个人电子渠道", "电子渠道")):
        return True
    return False


def _looks_like_counterparty_contamination(value: str) -> bool:
    """Detect only independently provable row/page material in a party cell."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    transaction_time = re.search(
        r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s*[T ]?\s*\d{1,2}:\d{2}(?::\d{2})?)?",
        text,
    )
    if transaction_time is not None:
        tail = text[transaction_time.end() :]
        money_values = re.findall(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}(?!\d)", tail)
        if len(money_values) >= 2:
            return True

    compact = _signature_value(text)
    page_footer = bool(
        re.search(r"第\d+页(?:[/／]?共\d+页|[/／]共\d+页)", compact)
        or re.search(r"Page\d+(?:of|/)\d+", compact, re.IGNORECASE)
    )
    if not page_footer:
        return False
    header_markers = (
        "交易日期",
        "交易时间",
        "收入金额",
        "支出金额",
        "交易金额",
        "账户余额",
        "对方账号",
        "对方户名",
    )
    return sum(marker in compact for marker in header_markers) >= 2


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
        row_start = max(cursor, _infer_transaction_start(compact_text, transaction, account_pos))
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


def _next_located_row_start(
    locations: list[dict[str, int]],
    index: int,
    fallback: int,
) -> int | None:
    """Return the immediate row boundary, never a later row after a gap."""
    current_end = locations[index].get("account_end", 0) if index < len(locations) else 0
    if index + 1 >= len(locations):
        return fallback if fallback >= current_end else None
    row_start = locations[index + 1].get("row_start", 0)
    return row_start if row_start >= current_end else None


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
    if _looks_like_counterparty_contamination(compact):
        return True
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
    if _looks_like_counterparty_contamination(value):
        return False
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


def _infer_row_sources(
    transactions: list[dict[str, Any]],
    parse_result: Any | None,
    *,
    physical_raw_rows: dict[tuple[int, str, tuple[str, ...]], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    if not transactions or parse_result is None:
        return []
    parse_result = _read_view(parse_result)
    logical_sources = _logical_table_row_sources(
        parse_result,
        physical_raw_rows=physical_raw_rows,
    )
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


def _logical_table_row_sources(
    parse_result: Any,
    *,
    physical_raw_rows: dict[tuple[int, str, tuple[str, ...]], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    physical_raw_rows = (
        physical_raw_rows
        if physical_raw_rows is not None
        else _physical_raw_row_geometry_sources(parse_result)
    )
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
            bbox: list[float] | None = None
            values = [str(getattr(cell, "text", "") or "").strip() for cell in cells]
            physical_matches = physical_raw_rows.get(
                (source_page, source_table_id, tuple(values)),
                [],
            )
            if len(physical_matches) == 1:
                physical = physical_matches[0]
                bbox = physical.get("bbox")
                if physical.get("evidence_ids"):
                    evidence_ids = list(physical["evidence_ids"])
                physical_row_index = physical.get("physical_row_index")
                ref_row_index = (
                    physical_row_index if isinstance(physical_row_index, int) else row_source_index
                )
                if physical.get("sourced_column_indexes"):
                    source_cell_refs = _physical_geometry_cell_refs(
                        physical,
                        page=source_page,
                        table_id=source_table_id,
                        row=ref_row_index,
                    )

            source = {
                "source_page": source_page,
                "page_id": f"page:{source_page:04d}",
                **({"table_id": source_table_id} if source_table_id else {}),
                "source_row_index": row_source_index,
                "page_range": [source_page, source_page],
                **({"bbox": bbox} if bbox is not None else {}),
                **({"source_cell_refs": source_cell_refs} if source_cell_refs else {}),
                **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                "_signature": _row_signature(values),
            }
            _ensure_row_bbox_source_ref(source)
            sources.append(source)
    return sources


def _augment_source_from_exact_physical_raw_row(
    transaction: dict[str, Any],
    source: dict[str, Any],
    physical_raw_rows: dict[tuple[int, str, tuple[str, ...]], list[dict[str, Any]]],
) -> None:
    """Repair a row source only from one exact physical raw-row match."""

    source_page = _positive_int(source.get("source_page"))
    table_id = str(source.get("table_id") or "")
    if source_page is None or not table_id:
        return
    values = tuple(
        str(value or "").strip()
        for key, value in transaction.items()
        if not str(key).startswith("_")
    )
    matches = physical_raw_rows.get((source_page, table_id, values), [])
    if len(matches) != 1:
        return
    physical = matches[0]
    physical_table_id = str(physical.get("physical_table_id") or table_id)
    is_table_alias = physical_table_id != table_id
    if physical.get("bbox") is not None and (not is_table_alias or source.get("bbox") is None):
        source["bbox"] = list(physical["bbox"])
    if physical.get("evidence_ids"):
        source["evidence_ids"] = list(physical["evidence_ids"])
    if not physical.get("sourced_column_indexes"):
        return
    physical_row_index = physical.get("physical_row_index")
    fallback_row_index = _positive_or_zero_int(source.get("source_row_index", -1))
    ref_row_index = (
        physical_row_index
        if isinstance(physical_row_index, int)
        else (fallback_row_index if fallback_row_index is not None else physical["raw_row_index"])
    )
    source["source_cell_refs"] = _physical_geometry_cell_refs(
        physical,
        page=source_page,
        table_id=physical_table_id,
        row=ref_row_index,
    )


def _physical_geometry_cell_refs(
    physical: dict[str, Any],
    *,
    page: int,
    table_id: str,
    row: int,
) -> list[dict[str, Any]]:
    return [
        {
            "page": page,
            "table_id": table_id,
            "row": row,
            "raw_row": physical["raw_row_index"],
            "col": col_index,
        }
        for col_index in physical.get("sourced_column_indexes", [])
    ]


def _physical_raw_row_geometry_sources(
    parse_result: Any,
) -> dict[tuple[int, str, tuple[str, ...]], list[dict[str, Any]]]:
    """Index exact physical raw rows without trusting logical row offsets.

    Logical-table assembly can promote a transaction-valued first raw row into
    ``table.headers`` on continuation pages.  Its logical cell references then
    retain the right page/table but may be shifted by one raw row.  Geometry is
    safe to restore only when the complete ordered logical row uniquely equals
    one ``metadata.raw_rows`` entry inside that declared physical table.
    """

    indexed: dict[tuple[int, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for page in getattr(parse_result, "pages", []) or []:
        page_number = _positive_int(
            getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0)
        )
        if page_number is None:
            continue
        for table_index, table in enumerate(getattr(page, "tables", []) or []):
            table_id = str(getattr(table, "table_id", "") or f"pt_{page_number}_{table_index}")
            metadata = getattr(table, "metadata", None) or {}
            if not isinstance(metadata, dict):
                continue
            raw_rows = metadata.get("raw_rows")
            geometry = metadata.get("geometry")
            if not isinstance(raw_rows, list) or not isinstance(geometry, dict):
                continue
            cell_bboxes = geometry.get("cell_bboxes")
            cell_evidence_ids = geometry.get("cell_evidence_ids")
            physical_row_indexes = _physical_row_indexes_by_raw_row(table)

            for raw_row_index, raw_values in enumerate(raw_rows):
                if not isinstance(raw_values, list):
                    continue
                values = tuple(str(value or "").strip() for value in raw_values)
                geometry_boxes = (
                    cell_bboxes[raw_row_index]
                    if isinstance(cell_bboxes, list) and raw_row_index < len(cell_bboxes)
                    else []
                )
                geometry_evidence = (
                    cell_evidence_ids[raw_row_index]
                    if isinstance(cell_evidence_ids, list) and raw_row_index < len(cell_evidence_ids)
                    else []
                )
                bbox = _geometry_row_bbox(geometry_boxes)
                evidence_ids = _geometry_row_evidence_ids(geometry_evidence)
                sourced_column_indexes = _geometry_sourced_column_indexes(
                    geometry_boxes,
                    geometry_evidence,
                )
                physical = {
                    "physical_table_id": table_id,
                    "raw_row_index": raw_row_index,
                    "physical_row_index": physical_row_indexes.get(raw_row_index),
                    **({"bbox": bbox} if bbox is not None else {}),
                    **({"evidence_ids": evidence_ids} if evidence_ids else {}),
                    "sourced_column_indexes": sourced_column_indexes,
                }
                table_ids = {table_id}
                if table_id == f"pt_{page_number}_{table_index}":
                    table_ids.add(f"native:p{page_number}:t{table_index}")
                for source_table_id in table_ids:
                    indexed.setdefault((page_number, source_table_id, values), []).append(physical)
    return indexed


def _physical_row_indexes_by_raw_row(table: Any) -> dict[int, int]:
    indexes: dict[int, int] = {}
    ambiguous: set[int] = set()
    for position, row in enumerate(getattr(table, "rows", []) or []):
        cells = list(getattr(row, "cells", []) or [])
        refs = _row_source_cell_refs(row, cells)
        raw_indexes = {
            raw_index
            for ref in refs
            if (raw_index := _positive_or_zero_int(ref.get("raw_row", -1))) is not None
        }
        if len(raw_indexes) != 1:
            continue
        raw_index = next(iter(raw_indexes))
        row_index = _positive_or_zero_int(getattr(row, "source_row_index", -1))
        row_index = row_index if row_index is not None else position
        if raw_index in indexes and indexes[raw_index] != row_index:
            ambiguous.add(raw_index)
            continue
        indexes[raw_index] = row_index
    for raw_index in ambiguous:
        indexes.pop(raw_index, None)
    return indexes


def _geometry_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return None
    return bbox


def _geometry_row_bbox(geometry_boxes: Any) -> list[float] | None:
    if not isinstance(geometry_boxes, list):
        return None
    boxes = [bbox for value in geometry_boxes if (bbox := _geometry_bbox(value)) is not None]
    if not boxes:
        return None
    return [
        min(bbox[0] for bbox in boxes),
        min(bbox[1] for bbox in boxes),
        max(bbox[2] for bbox in boxes),
        max(bbox[3] for bbox in boxes),
    ]


def _geometry_row_evidence_ids(geometry_evidence: Any) -> list[str]:
    if not isinstance(geometry_evidence, list):
        return []
    return list(
        dict.fromkeys(
            str(evidence_id)
            for cell_evidence in geometry_evidence
            for evidence_id in (cell_evidence if isinstance(cell_evidence, list) else [])
            if str(evidence_id)
        )
    )


def _geometry_sourced_column_indexes(
    geometry_boxes: Any,
    geometry_evidence: Any,
) -> list[int]:
    boxes = geometry_boxes if isinstance(geometry_boxes, list) else []
    evidence = geometry_evidence if isinstance(geometry_evidence, list) else []
    indexes: list[int] = []
    for col_index in range(max(len(boxes), len(evidence))):
        has_bbox = col_index < len(boxes) and _geometry_bbox(boxes[col_index]) is not None
        has_evidence = (
            col_index < len(evidence)
            and isinstance(evidence[col_index], list)
            and any(str(value) for value in evidence[col_index])
        )
        if has_bbox or has_evidence:
            indexes.append(col_index)
    return indexes


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
    for key in ("交易日期", "日期", "记账日期", "记账日"):
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


def _decode_native_source_repair(
    transaction: dict[str, Any],
    raw_headers: list[str],
) -> None:
    """Decode a source-preserving native cell repair under a strict contract."""
    source_json = transaction.pop(_NATIVE_SOURCE_RAW_JSON_COLUMN, "")
    repair_json = transaction.pop(_NATIVE_SOURCE_REPAIR_JSON_COLUMN, "")
    if not source_json and not repair_json:
        return
    if (
        raw_headers.count(_NATIVE_SOURCE_RAW_JSON_COLUMN) != 1
        or raw_headers.count(_NATIVE_SOURCE_REPAIR_JSON_COLUMN) != 1
    ):
        return

    business_headers = [header for header in raw_headers if not str(header).startswith("_")]
    if not business_headers or len(business_headers) != len(set(business_headers)):
        return
    try:
        source_raw = json.loads(str(source_json))
        manifest = json.loads(str(repair_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    if (
        not isinstance(source_raw, dict)
        or list(source_raw) != business_headers
        or not all(isinstance(value, str) for value in source_raw.values())
        or not isinstance(manifest, dict)
        or set(manifest) != _NATIVE_SOURCE_REPAIR_KEYS
        or not all(isinstance(value, str) for value in manifest.values())
        or manifest.get("kind") != _NATIVE_SOURCE_REPAIR_KIND
    ):
        return

    expected_working = _validated_native_source_repair_working_map(
        source_raw,
        business_headers,
        manifest,
    )
    actual_working = {header: transaction.get(header) for header in business_headers}
    if expected_working is None or actual_working != expected_working:
        return
    transaction["_source_raw"] = source_raw
    transaction["_source_repair_manifest"] = manifest
    transaction["_canonical_raw_from_working"] = True


def _source_roles_for_normalization(transaction: dict[str, Any]) -> dict[str, Any]:
    """Use repaired roles only while the complete working-row contract holds."""
    source_raw = transaction.get("_source_raw")
    if not isinstance(source_raw, dict):
        return transaction
    manifest = transaction.get("_source_repair_manifest")
    if (
        transaction.get("_canonical_raw_from_working") is not True
        or not isinstance(manifest, dict)
        or set(manifest) != _NATIVE_SOURCE_REPAIR_KEYS
        or manifest.get("kind") != _NATIVE_SOURCE_REPAIR_KIND
    ):
        return source_raw
    working_roles = {header: transaction.get(header) for header in source_raw}
    expected_working = _validated_native_source_repair_working_map(
        source_raw,
        list(source_raw),
        manifest,
    )
    if expected_working is None or working_roles != expected_working:
        return source_raw
    return working_roles


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
            _decode_native_source_repair(txn, raw_headers)
            if any(value for key, value in txn.items() if key != "_source_page"):
                transactions.append(txn)
    return _finalize_transactions(transactions, parse_result, full_text)


def _extract_bojs_source_grid_records(tables: list[list[list[str]]]) -> list[dict[str, str]]:
    """Harvest the exact BOJS layout without replacing its source headers."""
    transactions: list[dict[str, str]] = []
    for table in tables:
        header_index = next(
            (
                index
                for index, row in enumerate(table[:10])
                if _BOJS_SOURCE_HEADERS.issubset(
                    {
                        re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(cell or "")))
                        for cell in row
                    }
                )
            ),
            -1,
        )
        if header_index < 0:
            continue
        headers = [str(cell or "").strip() for cell in table[header_index]]
        for row in table[header_index + 1 :]:
            if not row or not any(str(cell or "").strip() for cell in row):
                continue
            if is_footer_or_total_row(row) or not row_has_transaction_data(row, strict_first_col=True):
                continue
            transaction = {
                headers[index] if index < len(headers) and headers[index] else f"col_{index}": str(cell or "").strip()
                for index, cell in enumerate(row)
            }
            if not _BOJS_SOURCE_HEADERS.issubset(_source_header_signature(transaction)):
                continue
            transactions.append(transaction)
    return transactions


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

    bojs_transactions = _extract_bojs_source_grid_records(ctx.tables)
    if bojs_transactions:
        return _finalize_transactions(bojs_transactions, ctx.parse_result, ctx.full_text)

    split_txns: list[dict[str, str]] = []
    for tbl in ctx.tables:
        if not tbl:
            continue
        header = detect_headers([tbl], plugin.column_registry, prefer_strict=True)
        if header is not None and has_split_debit_credit_headers([[header.raw_headers]]):
            split_txns.extend(_extract_split_grid_records([tbl], header.row_index, header.raw_headers))
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


def _validated_source_date(value: str) -> str:
    """Return one calendar-valid source date in ISO form, otherwise empty."""
    source = str(value or "").strip()
    parsed = normalize_timestamp(source)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", parsed):
        date_value = parsed
    elif (
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T00:00:00", parsed)
        and "T" not in source
        and not re.search(r"\d:\d", source)
    ):
        # The shared normalizer represents an ISO date as midnight.  Preserve
        # the source's date-only semantics instead of inventing a timestamp.
        date_value = parsed[:10]
    else:
        return ""
    try:
        return datetime.fromisoformat(date_value).date().isoformat()
    except ValueError:
        return ""


def _validated_source_datetime(value: str) -> str:
    """Normalize a full source datetime after removing arbitrary line wraps."""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    if not compact:
        return ""

    parsed = normalize_timestamp(compact)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?", parsed):
        try:
            return datetime.fromisoformat(parsed).isoformat()
        except ValueError:
            return ""

    # Native grids commonly wrap ``YYYYMMDD`` and ``HH:MM:SS`` inside one
    # physical cell.  Whitespace compaction therefore produces a valid but
    # otherwise unsupported ``YYYYMMDDHH:MM:SS`` representation.
    match = re.fullmatch(
        r"(?P<date>(?:19|20)\d{6})(?P<time>\d{1,2}:\d{2}(?::\d{2})?)",
        compact,
    )
    if match is None:
        return ""
    date_value = _validated_source_date(match.group("date"))
    if not date_value:
        return ""
    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            parsed_time = datetime.strptime(match.group("time"), time_format).time()
            return datetime.combine(datetime.fromisoformat(date_value).date(), parsed_time).isoformat()
        except ValueError:
            continue
    return ""


def _validated_source_time(value: str) -> str:
    """Return one time-only source value in canonical display form."""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    for pattern, time_format in (
        (r"\d{1,2}:\d{2}:\d{2}", "%H:%M:%S"),
        (r"\d{1,2}:\d{2}", "%H:%M"),
        (r"\d{6}", "%H%M%S"),
        (r"\d{4}", "%H%M"),
    ):
        if not re.fullmatch(pattern, compact):
            continue
        try:
            return datetime.strptime(compact, time_format).time().isoformat()
        except ValueError:
            return ""
    return ""


def _normalize_wrapped_temporal_fields(
    normalized: dict[str, Any],
    raw_txn: dict[str, str],
) -> dict[str, Any]:
    out = dict(normalized)
    date_value = _cell_value(raw_txn, "交易日期", "记账日", "记账日期", "日期", "Date")
    timestamp_value = _cell_value(raw_txn, "交易时间", "时间", "Time")
    timestamp_compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", timestamp_value))
    date_compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", date_value))

    # A full transaction datetime is self-contained.  A separate posting date
    # may legitimately differ, so never prepend it to an already dated value.
    source_timestamp = _validated_source_datetime(timestamp_compact)
    if source_timestamp:
        out["date"] = source_timestamp[:10]
        out["timestamp"] = source_timestamp
    else:
        date_column_value = _materialize_page_scoped_short_date(raw_txn, date_compact) or _validated_source_date(
            date_compact
        )
        source_time = _validated_source_time(timestamp_compact)
        # A six-digit time such as ``120101`` is also a syntactically valid
        # YYMMDD date to the shared normalizer.  Under explicit, separate date
        # and time columns, the source header roles remove that ambiguity: use
        # the validated date column as context before considering a date-only
        # value in the timestamp column.
        compact_time_only = bool(re.fullmatch(r"(?:\d{4}|\d{6}|\d{1,2}:\d{2}(?::\d{2})?)", timestamp_compact))
        timestamp_date = "" if date_column_value and source_time and compact_time_only else _validated_source_date(
            timestamp_compact
        )
        source_date = timestamp_date or date_column_value
        if source_date:
            out["date"] = source_date
            if source_time:
                try:
                    parsed_time = datetime.strptime(source_time, "%H:%M:%S").time()
                    out["timestamp"] = datetime.combine(
                        datetime.fromisoformat(source_date).date(),
                        parsed_time,
                    ).isoformat()
                except ValueError:
                    pass

    balance = _normalize_monetary_cell(_cell_value(raw_txn, "余额", "账户余额", "本次余额", "账面余额"))
    if balance is not None:
        out["balance"] = float(balance)
    return out


def normalize_record(raw_txn: dict[str, str], plugin: Any) -> dict[str, Any]:
    def finalize(normalized: dict[str, Any]) -> dict[str, Any]:
        normalized.pop("amount_cny", None)
        _apply_document_scope_direction(normalized, raw_txn)
        source_roles = _source_roles_for_normalization(raw_txn)
        if source_roles is not raw_txn:
            # Geometry repairs may standardize a compound source header or
            # reconstruct a missing direction for candidate scoring.  The
            # row-scoped exact source map remains authoritative for roles it
            # explicitly owns.  Overlay only those business roles; keep the
            # repaired date, amount, balance, and row geometry untouched.
            _normalize_source_counterparty_columns(source_roles, normalized)
            source_direction = _normalize_direction_text(_explicit_direction_source_value(source_roles))
            if source_direction in {"income", "expense"}:
                normalized["direction"] = source_direction
        _normalize_directional_payer_payee(source_roles, normalized)
        _enforce_distinct_account_header_roles(source_roles, normalized)
        return normalized

    if raw_txn.get("_compact") == "1":
        from docmirror.plugins.bank_statement.styles.compact_merged import normalize_record as compact_norm

        normalized = _normalize_wrapped_temporal_fields(compact_norm(raw_txn), raw_txn)
        return finalize(normalized)

    split = normalize_split_debit_credit(raw_txn, plugin)
    if split is not None:
        normalized = _normalize_wrapped_temporal_fields(split, raw_txn)
        return finalize(normalized)

    from docmirror.plugins.bank_statement.styles.signed_amount import parse_signed_amount

    for key, value in raw_txn.items():
        normalized_key = normalize_header_cell(key)
        if any(n in normalized_key for n in ("金额", "发生", "Amount")) and str(value).strip().startswith(("+", "-")):
            source_value = str(value).strip()
            amount, direction = parse_signed_amount(source_value)
            if amount is None:
                # Repeated PDF watermarks, verification URLs, or print
                # furniture can share a text object with the first signed
                # amount on a continuation page.  The bounded numeric prefix
                # and its explicit sign remain direct source facts even when
                # the trailing overlay prevents whole-cell parsing.
                prefix_amount = _normalize_monetary_cell(source_value)
                if prefix_amount is not None:
                    amount = abs(float(prefix_amount))
                    direction = "expense" if source_value.startswith("-") else "income"
            if amount is not None:
                normalized = _normalize_with_temporal_context(raw_txn, plugin)
                _normalize_source_counterparty_columns(raw_txn, normalized)
                normalized["amount"] = abs(float(amount))
                normalized["amount_cny"] = abs(float(amount))
                normalized["direction"] = direction
                normalized = _normalize_wrapped_temporal_fields(normalized, raw_txn)
                return finalize(normalized)

    normalized = _normalize_with_temporal_context(raw_txn, plugin)
    _normalize_source_counterparty_columns(raw_txn, normalized)
    normalized = _normalize_wrapped_temporal_fields(normalized, raw_txn)
    return finalize(normalized)


def refine_missing_directions_from_balance_chain(records: list[dict[str, Any]]) -> None:
    """Infer or correct directions when the source balance chain is unique."""
    source_order = _record_source_order(records)
    for index, record in enumerate(records):
        normalized = record.get("normalized") or {}
        raw = record.get("raw") or {}
        scope_text = re.sub(
            r"\s+",
            "",
            unicodedata.normalize(
                "NFKC",
                str(record.get("_document_scope_text") or raw.get("_document_scope_text") or ""),
            ),
        )
        if any(title in scope_text for title in _INCOME_ONLY_STATEMENT_TITLES):
            # This is an explicitly filtered income report.  Omitted debit rows
            # make adjacent balance deltas discontinuous, so the title-scoped
            # direction must not be rewritten by balance or summary inference.
            normalized["direction"] = "income"
            continue
        explicit_direction = _normalize_direction_text(_explicit_direction_source_value(raw))
        if explicit_direction in {"income", "expense"}:
            # A dedicated source direction is an independent business fact.
            # Negative reversal amounts can make the balance move opposite to
            # that label, but balance inference must not rewrite the source.
            normalized["direction"] = explicit_direction
            continue
        _income_present, _income_raw, explicit_income = _explicit_amount_column(raw, _SPLIT_CREDIT_KEYS)
        _expense_present, _expense_raw, explicit_expense = _explicit_amount_column(raw, _SPLIT_DEBIT_KEYS)
        income_nonzero = explicit_income is not None and explicit_income != 0
        expense_nonzero = explicit_expense is not None and explicit_expense != 0
        if income_nonzero != expense_nonzero:
            # Split debit/credit cells are direct source semantics. Same-time
            # batches are not guaranteed to be printed in balance order, so an
            # adjacent delta may never override this explicit row direction.
            normalized["direction"] = "income" if income_nonzero else "expense"
            continue
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
