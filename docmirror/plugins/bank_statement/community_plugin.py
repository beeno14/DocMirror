# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bank statement community plugin — style-aware ledger extract.

Premium community plugin for ``bank_statement`` documents. Extends ``BaseTableParser``
with a style detection pipeline (``BankStyleDetector`` → ``BankStyleParserRegistry``)
that selects among grid, compact merged, signed amount, borderless OCR, and KV
identity parsers before building canonical transaction facts.

Pipeline role: registered as ``plugin`` for post-seal registry discovery; the projector
invokes ``derive`` on canonical tables and OCR evidence fallback.

Key exports: ``BankStatementCommunityPlugin``, ``plugin``, column/identity config constants.

Dependencies: ``_base.base_table_parser``, ``bank_statement.extract_pipeline``, ``ProjectionData``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from docmirror.plugins._base.base_table_parser import BaseTableParser
from docmirror.plugins._base.column_registry import ColumnMapping
from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins.bank_statement.extract_pipeline import run_bank_statement_extract

BANK_COLUMN_REGISTRY: dict[str, ColumnMapping] = {
    "序号": ColumnMapping(field="sequence_no", aliases=["No.", "序列号"]),
    "交易日期": ColumnMapping(field="date", format_hint="date", aliases=["日期", "记账日期", "记账日", "Date"]),
    "交易时间": ColumnMapping(field="timestamp", format_hint="datetime", aliases=["时间", "Time"]),
    "收/支": ColumnMapping(
        field="direction",
        enum_map={
            "收入": "income",
            "转入": "income",
            "收人": "income",
            "支出": "expense",
            "转出": "expense",
            "支山": "expense",
            "支鼎": "expense",
            "攴出": "expense",
            "贷": "income",
            "贷Cr": "income",
            "Cr": "income",
            "借": "expense",
            "借Dr": "expense",
            "Dr": "expense",
        },
        aliases=[
            "收支",
            "方向",
            "交易方向",
            "交易类别",
            "交易类型",
            "收入/支出",
            "月收/支",
            "月收支",
            "借贷",
            "借/贷",
            "借贷标志",
            "Dc Flg",
        ],
    ),
    "摘要": ColumnMapping(field="summary", aliases=["交易摘要", "备注", "Description", "Memo"]),
    "交易金额": ColumnMapping(
        field="amount",
        unit="CNY",
        aliases=["金额", "发生额", "Amount", "借方发生额", "贷方发生额", "收入金额", "支出金额"],
    ),
    "余额": ColumnMapping(field="balance", unit="CNY", aliases=["账户余额", "Balance"]),
    "对方户名": ColumnMapping(
        field="counter_party",
        aliases=[
            "对方名称",
            "交易对方",
            "Counter party",
            "Remarks",
            "对方账号与户名",
        ],
    ),
    "对方账号": ColumnMapping(field="counter_account", aliases=["对方账户", "Counter account"]),
    "对方行号": ColumnMapping(field="counter_bank_code", aliases=["对方银行行号"]),
    "对方行名": ColumnMapping(field="counter_bank_name", aliases=["对方开户行", "对方银行名称"]),
    "交易渠道": ColumnMapping(field="channel", aliases=["渠道", "交易方式"]),
    "用途": ColumnMapping(field="purpose", aliases=["交易用途"]),
}

BANK_STANDARD_FIELDS = [
    "date",
    "timestamp",
    "summary",
    "direction",
    "amount",
    "balance",
    "counter_party",
    "counter_account",
    "sequence_no",
    "counter_bank_code",
    "counter_bank_name",
    "channel",
    "purpose",
    "counterparty_status",
]

BANK_IDENTITY_FIELDS: Sequence[tuple[str, Sequence[str]]] = (
    ("account_holder", ("Account holder", "Account name", "Card holder", "Customer name", "户名", "账户名")),
    ("account_number", ("Account number", "Card number", "Customer account number", "账号", "账户号", "卡号")),
    ("bank_name", ("Bank name", "Bank branch", "银行名称")),
    ("query_period", ("Query period", "From/to date", "Period", "查询时间段", "交易时段")),
    ("print_date", ("打印日期",)),
    ("total_transactions", ("总笔数", "总条数")),
    ("currency", ("Currency", "币种")),
)


class BankStatementCommunityPlugin(BaseTableParser):
    """Community edition plugin for bank statement document processing."""

    @property
    def domain_name(self) -> str:
        return "bank_statement"

    @property
    def display_name(self) -> str:
        return "Bank Statement (Community)"

    @property
    def column_registry(self) -> dict[str, ColumnMapping]:
        return BANK_COLUMN_REGISTRY

    @property
    def standard_fields(self) -> list[str]:
        return BANK_STANDARD_FIELDS

    @property
    def identity_fields(self) -> Sequence[tuple[str, Sequence[str]]]:
        return BANK_IDENTITY_FIELDS

    def _recover_identity_from_evidence(self, parse_result) -> dict[str, dict[str, object]]:
        atoms_by_page = self._evidence_text_atoms_by_page(parse_result)
        if not atoms_by_page:
            return {}
        page_id = sorted(atoms_by_page)[0]
        atoms = sorted(
            atoms_by_page[page_id],
            key=lambda atom: (float(atom["bbox"][1]), float(atom["bbox"][0])),
        )
        text = " ".join(str(atom.get("text") or "").strip() for atom in atoms)
        patterns = {
            "print_date": ("打印日期", r"打印日期\s*[:：]\s*(20\d{2}-\d{2}-\d{2})"),
            "query_period": (
                "交易时段",
                r"交易时段\s*[:：]\s*(20\d{2}-\d{2}-\d{2})\s*至\s*(20\d{2}-\d{2}-\d{2})",
            ),
            "total_transactions": ("总条数", r"(?:总笔数|总条数)\s*[:：]\s*(\d+)"),
            "account_holder": (
                "客户名称",
                r"(?:户名|客户名称|客户姓名|账户名称)\s*[:：]\s*(.+?)(?=\s+(?:账号|卡号|起始日期|结束日期)\s*[:：])",
            ),
            "account_number": ("账号", r"账号\s*[:：]\s*([0-9*]+)"),
            "currency": ("币种", r"币种\s*[:：]\s*([^\s]+)"),
        }
        recovered: dict[str, dict[str, object]] = {}
        for field_name, (label, pattern) in patterns.items():
            match = re.search(pattern, text)
            if not match:
                continue
            value = " 至 ".join(match.groups()) if field_name == "query_period" else match.group(1).strip()
            if value:
                recovered[field_name] = self._evidence_identity_detail(field_name, label, value, page_id=page_id)
        title_atom = next(
            (atom for atom in atoms if "账户交易明细表" in str(atom.get("text") or "")),
            None,
        )
        if title_atom is not None:
            title = str(title_atom.get("text") or "").strip()
            recovered["statement_title"] = self._evidence_identity_detail(
                "statement_title",
                "document_title",
                title,
                page_id=page_id,
                evidence_ids=[str(title_atom.get("id") or "")],
            )
        return recovered

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        """Run the style-aware extractor and return projector-local facts."""
        result = run_bank_statement_extract(parse_result, text, self)
        records = _sanitize_bank_records(result.records)
        summary = self._build_summary(records)
        period = summary.get("period", {})
        period_detail = result.identity_fields.get("query_period")
        if isinstance(period_detail, dict):
            period_value = next(
                (
                    str(period_detail.get(candidate) or "")
                    for candidate in ("normalized_value", "value", "raw_value")
                    if period_detail.get(candidate) not in (None, "")
                ),
                "",
            )
            period_dates = re.findall(r"20\d{2}-\d{2}-\d{2}", period_value)
            if len(period_dates) >= 2:
                period = {"start": period_dates[0], "end": period_dates[1]}
        projection = self._projection_data_from_components(
            identity_fields=result.identity_fields,
            records=records,
            raw_headers=[],
            summary=summary,
            period=period,
            extra_domain_facts=result.style_meta.to_properties(),
            warnings=result.warnings,
            confidence=1.0 if result.style_meta.extract_status != "degraded" else 0.5,
        )
        identity_values: dict[str, str] = {}
        for field_name, detail in result.identity_fields.items():
            value = detail
            if isinstance(detail, dict):
                value = next(
                    (
                        detail.get(candidate)
                        for candidate in ("normalized_value", "value", "raw_value")
                        if detail.get(candidate) not in (None, "")
                    ),
                    None,
                )
            if value not in (None, ""):
                identity_values[field_name] = str(value)
        entity_fields = {
            target: identity_values[source]
            for source, target in (
                ("account_holder", "subject_name"),
                ("account_number", "subject_id"),
                ("bank_name", "organization"),
            )
            if identity_values.get(source)
        }
        return projection.model_copy(
            update={
                "entity_fields": entity_fields,
                "content_markdown_override": _render_bank_statement_content_markdown(
                    records,
                    identity_values,
                    period,
                    text,
                ),
            }
        )


def _render_bank_statement_content_markdown(
    records: list[dict],
    identity: dict[str, str],
    period: str | dict,
    source_text: str = "",
) -> str:
    """Render a record-complete bank statement Markdown view from canonical plugin facts."""
    if not records:
        return ""
    rows_by_page: dict[int, list[dict]] = {}
    for record in records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        page = int(source.get("source_page") or (source.get("page_range") or [1])[0] or 1)
        rows_by_page.setdefault(page, []).append(record)

    parts = ['<!-- docmirror:markdown-profile version="1.0" -->']
    page_numbers = sorted(rows_by_page) or [1]
    raw_headers = _raw_statement_table_headers(records, source_text)
    for page in page_numbers:
        parts.append(f'<!-- docmirror:page logical="{page}" source="{page}" -->')
        page_records = rows_by_page.get(page, [])
        if raw_headers:
            header_lines = _raw_statement_header_lines(identity, period, source_text)
            if header_lines:
                parts.append("  \n".join(header_lines))
            parts.append(_render_raw_statement_table(page_records, raw_headers))
            after_table_lines = _raw_statement_after_table_lines(source_text, page)
            if after_table_lines:
                parts.append("  \n".join(after_table_lines))
        else:
            parts.append(f"## 第 {page} 页")
            if page == page_numbers[0]:
                parts.extend(_bank_statement_header_lines(identity, period))
            parts.append(_render_bank_statement_table(page_records))
    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def _sanitize_bank_records(records: list[dict]) -> list[dict]:
    """Remove page furniture accidentally captured in bank transaction fields."""
    sanitized: list[dict] = []
    for record in records:
        copied = {
            key: dict(value) if key in {"raw", "normalized", "canonical_raw"} and isinstance(value, dict) else value
            for key, value in dict(record).items()
        }
        for pool_name in ("raw", "normalized", "canonical_raw"):
            pool = copied.get(pool_name)
            if not isinstance(pool, dict):
                continue
            _sanitize_bank_value_pool(pool)
        sanitized.append(copied)
    counterparty_aliases = _stable_counterparty_aliases(sanitized)
    for record in sanitized:
        _sanitize_record_counterparty(record, counterparty_aliases)
    return sanitized


def _sanitize_bank_value_pool(pool: dict) -> None:
    for key, value in list(pool.items()):
        if not isinstance(value, str):
            continue
        key_text = str(key)
        text = _clean_footer_text(value)
        if key_text in {"balance", "amount", "amount_cny", "余额", "交易金额"}:
            text = _clean_money_text(text)
        if key_text in {"counter_party", "对方户名", "对方名称", "交易对手"}:
            text = _clean_counterparty_text(text)
        pool[key] = text


def _stable_counterparty_aliases(records: list[dict]) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for record in records:
        account = _record_counter_account(record)
        if not account:
            continue
        for value in _record_counterparty_values(record):
            cleaned = _clean_counterparty_text(value)
            if _usable_counterparty_alias(cleaned):
                candidates.setdefault(account, set()).add(cleaned)

    aliases: dict[str, str] = {}
    for account, values in candidates.items():
        ordered = sorted(values, key=lambda item: (len(item), item))
        if len(ordered) == 1:
            aliases[account] = ordered[0]
            continue
        for candidate in ordered:
            if any(other != candidate and other.startswith(candidate) for other in values):
                aliases[account] = candidate
                break
    return aliases


def _sanitize_record_counterparty(record: dict, aliases: dict[str, str]) -> None:
    account = _record_counter_account(record)
    summary = _record_summary(record)
    alias = aliases.get(account, "") if account else ""
    for pool_name in ("raw", "normalized", "canonical_raw"):
        pool = record.get(pool_name)
        if not isinstance(pool, dict):
            continue
        for key in ("对方户名", "对方名称", "交易对手", "counter_party"):
            value = pool.get(key)
            if not isinstance(value, str):
                continue
            cleaned = _clean_counterparty_text(value)
            if alias and (
                not cleaned or _looks_like_counterparty_residue(cleaned) or _looks_like_counterparty_residue(value)
            ):
                cleaned = alias
            elif alias and cleaned.startswith(alias) and len(cleaned) > len(alias) + 1:
                cleaned = alias
            if _is_fee_residue_counterparty(cleaned, summary):
                cleaned = ""
            pool[key] = cleaned
    normalized = record.get("normalized")
    if isinstance(normalized, dict):
        counter_party = str(normalized.get("counter_party") or "").strip()
        counter_account = str(normalized.get("counter_account") or "").strip()
        normalized["counterparty_status"] = "present" if counter_party or counter_account else "source_null"


def _record_counter_account(record: dict) -> str:
    for pool_name in ("normalized", "canonical_raw", "raw"):
        pool = record.get(pool_name)
        if not isinstance(pool, dict):
            continue
        for key in ("counter_account", "对方账号", "对方账户"):
            value = str(pool.get(key) or "").strip()
            if value:
                return re.sub(r"\s+", "", value)
    return ""


def _record_summary(record: dict) -> str:
    for pool_name in ("normalized", "canonical_raw", "raw"):
        pool = record.get(pool_name)
        if not isinstance(pool, dict):
            continue
        for key in ("summary", "摘要", "备注", "摘要/附言"):
            value = str(pool.get(key) or "").strip()
            if value:
                return value
    return ""


def _record_counterparty_values(record: dict) -> list[str]:
    values: list[str] = []
    for pool_name in ("normalized", "canonical_raw", "raw"):
        pool = record.get(pool_name)
        if not isinstance(pool, dict):
            continue
        for key in ("counter_party", "对方户名", "对方名称", "交易对手"):
            value = str(pool.get(key) or "").strip()
            if value:
                values.append(value)
    return values


def _usable_counterparty_alias(value: str) -> bool:
    text = str(value or "").strip()
    return len(text) > 1 and not _looks_like_counterparty_pollution(text)


def _clean_counterparty_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    if text in {"入", "收", "出", "支", "限公司", "有限公司", "代收）", "代收)"}:
        return ""
    text = _strip_counterparty_header_fragment(text)
    text = _strip_fee_tail(text)
    text = _strip_tax_escrow_tail(text)
    return text.strip()


def _strip_counterparty_header_fragment(value: str) -> str:
    compact = _compat_compact(value)
    marker = "序号交易日期"
    if marker not in compact:
        return value
    prefix_len = compact.index(marker)
    if prefix_len <= 0:
        return ""
    return _prefix_by_compact_length(value, prefix_len)


def _strip_fee_tail(value: str) -> str:
    compact = _compat_compact(value)
    match = re.search(r"(?:企业|个人)?电子渠道(?:跨行)?转账手续费(?:收入|收)?$", compact)
    if match is None or match.start() <= 1:
        return value
    return _prefix_by_compact_length(value, match.start())


def _strip_tax_escrow_tail(value: str) -> str:
    marker = "待报解预算收入"
    compact = _compat_compact(value)
    if marker not in compact:
        return value
    prefix_len = compact.index(marker)
    if prefix_len <= 1:
        return value
    return _prefix_by_compact_length(value, prefix_len)


def _prefix_by_compact_length(value: str, compact_length: int) -> str:
    seen = 0
    chars: list[str] = []
    for char in value:
        if char.isspace():
            chars.append(char)
            continue
        if seen >= compact_length:
            break
        chars.append(char)
        seen += 1
    return "".join(chars).strip()


def _looks_like_counterparty_pollution(value: str) -> bool:
    compact = _compat_compact(value)
    return (
        not compact
        or compact in {"入", "收", "出", "支", "限公司", "有限公司", "代收)", "代收）"}
        or "序号交易日期" in compact
    )


def _looks_like_counterparty_residue(value: str) -> bool:
    compact = _compat_compact(value)
    if compact in {"入", "收", "出", "支", "限公司", "有限公司", "代收)", "代收）"}:
        return True
    if re.fullmatch(r"(?:入|收|收入|出|支)\d{1,8}(?:第页)?", compact):
        return True
    return compact.startswith(("限公司", "代收)", "代收）"))


def _is_fee_residue_counterparty(value: str, summary: str) -> bool:
    compact_value = _compat_compact(value)
    compact_summary = _compat_compact(summary)
    return compact_summary == "收费" and compact_value in {"入", "收", "收入", "手续费收", "手续费收入"}


def _clean_money_text(value: str) -> str:
    text = str(value or "").strip()
    match = re.match(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    return match.group(0) if match else text


def _bank_statement_header_lines(identity: dict[str, str], period: str | dict) -> list[str]:
    lines = ["# 银行流水"]
    labels = [
        ("银行名称", identity.get("bank_name") or ""),
        ("开户行/客户行", identity.get("bank_branch") or ""),
        ("户名", identity.get("account_holder") or ""),
        ("账号", identity.get("account_number") or ""),
        ("币种", identity.get("currency") or ""),
    ]
    for label, value in labels:
        if value:
            lines.append(f"**{label}:** {_markdown_cell(value)}")
    if isinstance(period, dict):
        start = str(period.get("start") or "")
        end = str(period.get("end") or "")
        if start or end:
            lines.append(f"**账期:** {_markdown_cell(start)} 至 {_markdown_cell(end)}")
    elif period:
        lines.append(f"**账期:** {_markdown_cell(period)}")
    return lines


_SOURCE_TABLE_HEADER_LAYOUTS = [
    ["交易日期", "交易金额", "交易类别", "账户余额", "对方账号", "对方户名", "备注", "交易机构"],
    ["交易日期", "交易时间", "交易摘要", "交易金额", "本次余额", "对手信息", "日志号", "交易渠道", "交易附言"],
]
_RAW_REQUIRED_HEADERS = {"交易日期", "交易金额", "账户余额"}
_SOURCE_REQUIRED_HEADERS = {"交易日期", "交易金额"}
_RAW_TABLE_EXCLUDED_HEADERS = {"序号", "币别", "币种", "货币"}
_GENERIC_RAW_HEADER_ORDER = (
    ("交易日期", "日期", "记账日期", "记账日"),
    ("交易时间", "时间"),
    ("交易类型", "交易类别", "收/支", "收支", "方向", "交易方向", "借贷", "借/贷", "借贷标志", "收入/支出"),
    ("交易金额", "金额", "发生额"),
    ("账户余额", "本次余额", "余额"),
    ("对方账号", "对方账户"),
    ("对方户名", "对方名称", "交易对手"),
    ("交易摘要", "摘要/附言", "摘要", "备注"),
    ("交易渠道", "渠道"),
    ("交易附言", "附言", "用途"),
    ("交易机构", "机构"),
)
_RAW_DIRECTION_KEYS = (
    "收/支",
    "收支",
    "方向",
    "交易方向",
    "交易类别",
    "交易类型",
    "收入/支出",
    "月收/支",
    "月收支",
    "借贷",
    "借/贷",
    "借贷标志",
    "Dc Flg",
)
_HEADER_VALUE_KEYS = {
    "交易日期": (("交易日期", "日期"), ("date",)),
    "交易时间": (("交易时间", "时间"), ("timestamp",)),
    "交易摘要": (("交易摘要", "摘要"), ("summary",)),
    "交易金额": (("交易金额", "金额"), ("amount",)),
    "本次余额": (("本次余额", "账户余额", "余额"), ("balance",)),
    "账户余额": (("账户余额", "本次余额", "余额"), ("balance",)),
    "对手信息": (("对手信息", "对方户名", "对方账号"), ("counter_party", "counter_account")),
    "日志号": (("日志号", "日 志号", "序号", "交易流水号", "流水号"), ("sequence_no",)),
    "交易渠道": (("交易渠道", "渠道"), ("channel",)),
    "交易附言": (("交易附言", "附言", "用途"), ("purpose",)),
    "交易类别": (("交易类别", "交易类型", "收/支"), ("direction",)),
    "交易类型": (("交易类型", "交易类别", "收/支"), ("direction",)),
    "对方账号": (("对方账号", "对方账户"), ("counter_account",)),
    "对方户名": (("对方户名", "对方名称", "交易对手"), ("counter_party",)),
    "备注": (("备注", "摘要"), ("summary",)),
    "摘要/附言": (("摘要/附言", "摘要", "交易摘要", "备注"), ("summary",)),
    "币别": (("币别", "币种"), ("currency",)),
    "交易机构": (("交易机构", "机构"), ()),
}


def _raw_statement_table_headers(records: list[dict], source_text: str = "") -> list[str]:
    """Return source-table headers when records already carry a readable bank ledger shape."""
    if not records:
        return []
    source_headers = _source_statement_table_headers(source_text)
    if source_headers and _records_support_source_headers(records, source_headers):
        return source_headers

    supporting_rows = 0
    for record in records[:20]:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        if _raw_record_has_ledger_shape(raw):
            supporting_rows += 1
    if supporting_rows:
        known_headers = _known_raw_source_headers(records)
        if known_headers:
            return known_headers
        return _generic_raw_statement_headers(records)
    return []


def _raw_record_has_ledger_shape(raw: dict) -> bool:
    if not raw:
        return False
    has_required = all(_source_header_value(raw, {}, header) for header in _RAW_REQUIRED_HEADERS)
    has_direction = any(raw.get(key) not in (None, "") for key in _RAW_DIRECTION_KEYS)
    return has_required and has_direction


def _generic_raw_statement_headers(records: list[dict]) -> list[str]:
    present_headers: list[str] = []
    for record in records[:20]:
        raw = _record_raw(record)
        if not _raw_record_has_ledger_shape(raw):
            continue
        for key, value in raw.items():
            if key in _RAW_TABLE_EXCLUDED_HEADERS or value in (None, ""):
                continue
            if key not in present_headers:
                present_headers.append(key)
    ordered: list[str] = []
    for candidates in _GENERIC_RAW_HEADER_ORDER:
        header = _first_present_header(present_headers, candidates)
        if header and header not in ordered:
            ordered.append(header)
    ordered.extend(header for header in present_headers if header not in ordered)
    return ordered


def _known_raw_source_headers(records: list[dict]) -> list[str]:
    best_headers: list[str] = []
    best_score = 0
    for record in records[:20]:
        raw = _record_raw(record)
        for headers in _SOURCE_TABLE_HEADER_LAYOUTS:
            score = sum(1 for header in headers if raw.get(header) not in (None, ""))
            if score > best_score:
                best_score = score
                best_headers = headers
    if best_score < 6:
        return []
    return [header for header in best_headers if any(_record_raw(record).get(header) not in (None, "") for record in records)]


def _first_present_header(headers: list[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        for header in headers:
            if header == candidate:
                return header
    return ""


def _source_statement_table_headers(source_text: str) -> list[str]:
    lines = [line.strip() for line in str(source_text or "").splitlines() if line.strip()]
    for idx in range(len(lines)):
        window = " ".join(lines[idx : idx + 12])
        compact = _compat_compact(window)
        for headers in _SOURCE_TABLE_HEADER_LAYOUTS:
            if all(_compat_compact(header) in compact for header in headers):
                return headers
    return []


def _records_support_source_headers(records: list[dict], headers: list[str]) -> bool:
    if not _SOURCE_REQUIRED_HEADERS.issubset(headers):
        return False
    support = 0
    for record in records[:20]:
        raw = _record_raw(record)
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        if _source_header_value(raw, normalized, "交易日期") and _source_header_value(raw, normalized, "交易金额"):
            support += 1
    return support > 0


def _raw_statement_header_lines(identity: dict[str, str], period: str | dict, source_text: str) -> list[str]:
    source_lines = _source_statement_header_block(source_text)
    if source_lines:
        return source_lines

    title = _source_statement_title(source_text)
    holder, branch = _holder_and_branch(identity)
    lines = [title] if title else []
    print_date = _source_label_value(source_text, "打印日期") or identity.get("print_date") or ""
    account_number = identity.get("account_number") or _source_label_value(source_text, "卡/账号") or ""
    if print_date:
        lines.append(f"打印日期：{_compact_date(print_date)}")
    if account_number:
        lines.append(f"卡/账号： {_markdown_cell(account_number)}")
    if holder:
        lines.append(f"户名： {_markdown_cell(holder)}")
    if branch:
        lines.append(f"开户行： {_markdown_cell(branch)}")
    start, end = _period_bounds(period, source_text)
    if start or end:
        lines.append(f"起始日期：{_compact_date(start)} 终止日期：{_compact_date(end)}")
    return lines


def _source_statement_header_block(source_text: str) -> list[str]:
    lines = [line.strip() for line in str(source_text or "").splitlines()]
    for idx, line in enumerate(lines):
        text = line.strip()
        if not _is_statement_title_line(text):
            continue
        out = [text]
        for next_line in lines[idx + 1 : idx + 10]:
            candidate = next_line.strip()
            if not candidate:
                continue
            if _looks_like_source_table_header(candidate) or _looks_like_transaction_line(candidate):
                break
            if _is_footer_line(candidate):
                break
            out.append(candidate)
        return out
    return []


def _holder_and_branch(identity: dict[str, str]) -> tuple[str, str]:
    holder = str(identity.get("account_holder") or "").strip()
    branch = str(identity.get("bank_branch") or "").strip()
    match = re.match(r"(.+?)\s*开户行\s*[:：]\s*(.+)", holder)
    if match:
        holder = match.group(1).strip()
        branch = branch or match.group(2).strip()
    return holder, branch


def _period_bounds(period: str | dict, source_text: str) -> tuple[str, str]:
    start = _source_label_value(source_text, "起始日期")
    end = _source_label_value(source_text, "终止日期")
    if start or end:
        return start, end
    if isinstance(period, dict):
        return str(period.get("start") or ""), str(period.get("end") or "")
    dates = re.findall(r"20\d{2}[-/]?\d{2}[-/]?\d{2}", str(period or ""))
    if len(dates) >= 2:
        return dates[0], dates[1]
    return "", ""


def _source_statement_title(source_text: str) -> str:
    for line in str(source_text or "").splitlines():
        text = line.strip()
        if _is_statement_title_line(text):
            return text
    return ""


def _source_label_value(source_text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}\s*[:：]\s*([^\n\r]+?)(?=\s+(?:打印日期|卡/账号|账号|户名|开户行|起始日期|终止日期)\s*[:：]|\n|\r|$)"
    match = re.search(pattern, str(source_text or ""))
    return match.group(1).strip() if match else ""


def _compact_date(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", text):
        return text.replace("-", "")
    if re.fullmatch(r"20\d{2}/\d{2}/\d{2}", text):
        return text.replace("/", "")
    return text


def _raw_statement_after_table_lines(source_text: str, page: int) -> list[str]:
    lines = [line.strip() for line in str(source_text or "").splitlines()]
    for idx, line in enumerate(lines):
        if not _is_bank_footer_line(line):
            continue
        footer_page = _footer_page_number(lines[idx + 1] if idx + 1 < len(lines) else "")
        if footer_page != page:
            continue
        out: list[str] = []
        for prev in reversed(lines[max(0, idx - 5) : idx]):
            if _is_statement_note_line(prev):
                out.insert(0, prev)
        out.extend(_raw_statement_footer_lines(source_text, page))
        return out
    return _raw_statement_footer_lines(source_text, page)


def _raw_statement_footer_lines(source_text: str, page: int) -> list[str]:
    lines = [line.strip() for line in str(source_text or "").splitlines()]
    for idx, line in enumerate(lines):
        if not _is_bank_footer_line(line):
            continue
        page_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if _footer_page_number(page_line) == page:
            return [line.strip(), page_line]
    return []


def _source_header_value(raw: dict, normalized: dict, header: str) -> object:
    raw_keys, normalized_keys = _HEADER_VALUE_KEYS.get(header, ((header,), ()))
    for key in raw_keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    for key in normalized_keys:
        value = normalized.get(key)
        if value in (None, ""):
            continue
        if header in {"交易类别", "交易类型"}:
            return _display_direction(value)
        if header == "交易金额":
            raw_amount = raw.get("交易金额")
            return raw_amount if raw_amount not in (None, "") else value
        return value
    return ""


def _looks_like_source_table_header(text: str) -> bool:
    compact = _compat_compact(text)
    return any(sum(_compat_compact(header) in compact for header in headers) >= 4 for headers in _SOURCE_TABLE_HEADER_LAYOUTS)


def _looks_like_transaction_line(text: str) -> bool:
    return bool(re.match(r"^20\d{6}(?:\s|$)", _compat_text(text)))


def _is_statement_title_line(text: str) -> bool:
    compact = _compat_compact(text)
    return bool(compact) and any(marker in compact for marker in ("交易流水", "交易明细", "对账单", "明细清单"))


def _is_statement_note_line(text: str) -> bool:
    compact = _compat_compact(text)
    return any(marker in compact for marker in ("截至打印时间", "无其他明细", "交易明细截止", "打印时间下方"))


def _is_footer_line(text: str) -> bool:
    return _is_bank_footer_line(text) or _footer_page_number(text) is not None


def _is_bank_footer_line(text: str) -> bool:
    compact = _compat_compact(text)
    return compact.startswith("@") and "银行" in compact


def _footer_page_number(text: str) -> int | None:
    match = re.search(r"第\s*(\d+)\s*页", _compat_text(text))
    return int(match.group(1)) if match else None


def _compat_compact(value: object) -> str:
    return re.sub(r"\s+", "", _compat_text(value))


def _compat_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or ""))


def _render_raw_statement_table(records: list[dict], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        raw = _record_raw(record)
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        values = [_raw_markdown_cell(_source_header_value(raw, normalized, header)) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _record_raw(record: dict) -> dict:
    raw = record.get("raw")
    return raw if isinstance(raw, dict) else {}


def _raw_markdown_cell(value: object) -> str:
    text = str(value or "").replace("|", "\\|").strip()
    return re.sub(r"\s*\n\s*", "", text)


def _render_bank_statement_table(records: list[dict]) -> str:
    headers = ["日期", "收/支", "交易金额", "账户余额", "对方户名", "对方账号", "摘要"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        values = [
            _first_value(raw, normalized, "交易日期", "date"),
            _display_direction(_first_raw_value(raw, "收/支", "交易类型", "交易类别") or _first_value(raw, normalized, "收/支", "direction")),
            _display_amount(raw, normalized),
            _first_value(raw, normalized, "余额", "balance"),
            _first_value(raw, normalized, "对方户名", "counter_party"),
            _first_value(raw, normalized, "对方账号", "counter_account"),
            _clean_footer_text(_first_value(raw, normalized, "摘要", "summary")),
        ]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    return "\n".join(lines)


def _first_value(raw: dict, normalized: dict, raw_key: str, normalized_key: str) -> object:
    raw_value = raw.get(raw_key)
    if raw_value not in (None, ""):
        return _clean_footer_text(str(raw_value))
    value = normalized.get(normalized_key)
    return _clean_footer_text(str(value)) if value not in (None, "") else ""


def _first_raw_value(raw: dict, *raw_keys: str) -> object:
    for raw_key in raw_keys:
        raw_value = raw.get(raw_key)
        if raw_value not in (None, ""):
            return _clean_footer_text(str(raw_value))
    return ""


def _display_amount(raw: dict, normalized: dict) -> str:
    amount = str(raw.get("交易金额") or normalized.get("amount") or "").strip()
    direction = str(_first_raw_value(raw, "收/支", "交易类型", "交易类别") or normalized.get("direction") or "").strip()
    if not amount:
        return ""
    if amount.startswith(("+", "-")):
        return amount
    if direction in {"收入", "income"}:
        return f"+{amount}"
    if direction in {"支出", "expense"}:
        return f"-{amount}"
    return amount


def _display_direction(value: object) -> str:
    text = str(value or "").strip()
    if text == "income":
        return "收入"
    if text == "expense":
        return "支出"
    return text


def _clean_footer_text(value: str) -> str:
    text = re.sub(r"(?:当前页|总页数|生成时间)[:：]?.*$", "", str(value or "")).strip()
    return text


def _markdown_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


plugin = BankStatementCommunityPlugin()
