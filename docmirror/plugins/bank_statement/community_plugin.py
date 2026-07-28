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
                ),
            }
        )


def _render_bank_statement_content_markdown(
    records: list[dict],
    identity: dict[str, str],
    period: str | dict,
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
    for page in page_numbers:
        parts.append(f'<!-- docmirror:page logical="{page}" source="{page}" -->')
        if page == page_numbers[0]:
            parts.extend(_bank_statement_header_lines(identity, period))
        parts.append(_render_bank_statement_table(rows_by_page.get(page, [])))
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
    return sanitized


def _sanitize_bank_value_pool(pool: dict) -> None:
    for key, value in list(pool.items()):
        if not isinstance(value, str):
            continue
        key_text = str(key)
        text = _clean_footer_text(value)
        if key_text in {"balance", "amount", "amount_cny", "余额", "交易金额"}:
            text = _clean_money_text(text)
        pool[key] = text


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


def _render_bank_statement_table(records: list[dict]) -> str:
    headers = ["序号", "日期", "收/支", "交易金额", "账户余额", "对方户名", "对方账号", "摘要"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        values = [
            _first_value(raw, normalized, "序号", "sequence_no"),
            _first_value(raw, normalized, "交易日期", "date"),
            _display_direction(_first_value(raw, normalized, "收/支", "direction")),
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


def _display_amount(raw: dict, normalized: dict) -> str:
    amount = str(raw.get("交易金额") or normalized.get("amount") or "").strip()
    direction = str(raw.get("收/支") or normalized.get("direction") or "").strip()
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
