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

import json
import logging
import re
import unicodedata
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
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
    "交易发生金额",
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
_BORDERLESS_FOOTER_MARKERS = (
    "数据缺失",
    "明细内容仅供参考",
    "银行提示",
    "合计:",
    "合计：",
    "本页合计",
    "交易总金额",
    "收入总额",
    "打印日期",
    "打印时间",
    "生成时间",
    "说明",
    "借方合计笔数",
    "贷方合计笔数",
    "打印渠道",
    "打印柜员",
    "借方发生总额",
    "贷方发生总额",
    "合计笔数",
    "期末余额",
)
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
    "总支出入笔数",
    "总支出金额",
    "出单截至日期",
    "以下此页无正文",
    "合计",
    "小计",
    "总计",
)
_COUNT_PATTERNS = (
    re.compile(r"(?:总条数|记录数|交易总笔数|交易总笔额|总笔数|合计笔数)[:：]\s*(?P<count>\d+)"),
    re.compile(r"汇总交易笔数\s*(?P<count>\d+)\s*笔"),
)
_SPDB_STATEMENT_TOTAL_PATTERN = re.compile(
    r"汇总交易笔数\s*(?:\n?Total number of transactions\s*)?(?P<count>\d+)\s*笔",
    re.S,
)
_PAGE_COUNT_PATTERN = re.compile(r"本页交易笔数\s*[:：]\s*(?P<count>\d+)")
_SOURCE_PAGE_RE = re.compile(r"第\s*(?P<page>\d+)\s*页\s*(?:[,，/／-]\s*)?共\s*(?P<total>\d+)\s*页")
_CUMULATIVE_TRANSACTION_FOOTER_RE = re.compile(r"第\s*(?P<through>\d+)\s*笔\s*[,，]?\s*共\s*(?P<total>\d+)\s*笔")
_STATEMENT_FIRST_PAGE_RE = re.compile(r"第\s*1\s*页\s*[,，]?\s*共\s*\d+\s*页")
_SPDB_MONTHLY_CONTEXT = ("客户名称", "账号", "账单币种", "账单类型", "月账单")
_SPDB_ANNUAL_CONTEXT = (
    "客户名称 Customer Name",
    "账号 Account Number",
    "账单币种 Currency",
    "账单统计日期 Start Time & End Time",
)
_SPDB_CN_DATE_RE = re.compile(r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日")
_SPDB_SLASH_DATE_RE = re.compile(r"(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})")
_CUMULATIVE_HEADER = ("交易日期", "交易金额", "账户余额", "对方户名", "对方账号", "摘要/备注", "编号")
_CUMULATIVE_IDENTITY_LABELS = ("账户名称:", "客户账号:", "开户机构:", "币种:")
_CUMULATIVE_PERIOD_RE = re.compile(r"20\d{6}\s*-\s*20\d{6}")
_NATIVE_DATETIME_RE = re.compile(r"(?P<date>20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\s+(?P<time>\d{1,2}:\d{2}:\d{2})")
_NATIVE_TIME_RE = re.compile(r"\d{1,2}:\d{2}:\d{2}")
_NATIVE_SIGNED_MONEY_RE = re.compile(r"[+-](?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")
_NATIVE_UNSIGNED_MONEY_RE = re.compile(r"(?<![\d,])(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")
_COMBINED_SIGNED_AMOUNT_HEADERS = ("收入/支出金额", "支出/收入金额", "收/支金额", "支/收交易金额")
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
_NATIVE_SUMMARY_SIGNED_MONEY_SPILL = re.compile(
    r"(?P<prefix>_?[A-Za-z\u3400-\u9fff][A-Za-z\u3400-\u9fff._:/()（）-]{0,63})"
    r"(?P<money>[+-](?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2})"
)
_NATIVE_CURRENCY_PREFIXES = frozenset({"CNY", "RMB", "USD", "HKD", "EUR", "JPY", "人民币", "美元"})
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
        r"总支出(?:入)?笔数\s*[:：]?\s*(?P<debit>\d+)",
        re.S,
    ),
    re.compile(
        r"总支出(?:入)?笔数\s*[:：]?\s*(?P<debit>\d+).*?"
        r"总收入笔数\s*[:：]?\s*(?P<credit>\d+)",
        re.S,
    ),
)
_COLUMN_MAJOR_DIRECTION_TOTAL_PATTERN = re.compile(
    r"支出总笔数\s*[:：]?\s*(?P<debit>\d+)\s*"
    r"支出总金额\s*[:：]?\s*收入总笔数\s*[:：]?\s*"
    r"(?P<debit_total>[\d,]+\.\d{2})\s+(?P<credit>\d+)\s*"
    r"收入总金额\s*[:：]?\s*(?P<credit_total>[\d,]+\.\d{2})",
    re.S,
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
    r"(?:\u603b\u6761\u6570|\u8bb0\u5f55\u6570|\u4ea4\u6613\u603b\u7b14\u6570|\u4ea4\u6613\u603b\u7b14\u989d|\u603b\u7b14\u6570|\u5408\u8ba1\u7b14\u6570)"
    r"[ \t]*[:\uff1a][ \t]*(?P<count>\d+)"
)

_CCB_PRIMARY_HEADER = ("序号", "摘要", "币别", "钞汇", "交易日期")
_CCB_DUPLICATE_HEADER = (*_CCB_PRIMARY_HEADER, "交易金额", "账户余额")
_CMB_PRIMARY_HEADER = ("记账日期", "货币", "交易金额", "联机余额", "交易摘要", "对手信息")
_CMB_ENGLISH_HEADER = ("Date", "Currency", "Transaction", "Amount", "Balance", "Transaction Type", "Counter Party")
_CMB_DUPLICATE_TITLE = "Transaction Statement of China Merchants Bank"
_CMB_PAGE_MARKER_RE = re.compile(r"(?P<page>\d+)\s*/\s*(?P<total>\d+)")
_PRIMARY_MONEY_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")
_ISO_CURRENCY_RE = re.compile(r"[A-Z]{3}")
_PRIMARY_DATE_SHAPE_RE = re.compile(r"(?:19|20)\d{2}(?:(?:[-/.]\d{1,2}){2}|\d{4})")
_SPDB_PERIOD_VALUE_RE = re.compile(
    r"(?:20\d{2}年\d{1,2}月\d{1,2}日(?:\s*[-至~]\s*20\d{2}年\d{1,2}月\d{1,2}日)?|"
    r"20\d{2}/\d{1,2}/\d{1,2}\s*[-至~]\s*20\d{2}/\d{1,2}/\d{1,2})"
)

_ISSUER_ROW_COUNT_SOURCES = frozenset(
    {
        "split_footer",
        "header_total",
        "statement_header_totals",
        "cumulative_footer_total",
        "page_footer",
    }
)
_ACCOUNT_IDENTITY_RE = re.compile(
    r"(?<!对方)(?:客户账号|账户账号|账号|账户代号|账户号|卡号/账号|卡号|Account\s+Number)"
    r"\s*[:：]?\s*(?:\n\s*)?"
    r"(?P<value>[0-9*＊-]{6,})",
    re.I,
)
_HOLDER_IDENTITY_RE = re.compile(
    r"(?<!对方)(?:户名|账户名称|客户名称|客户姓名|企业名称|用户所属公司|Customer\s+Name)\s*[:：]?\s*"
    r"(?:\n\s*)?(?P<value>[^\r\n:：]{2,80})",
    re.I,
)
_PERIOD_RE = re.compile(
    r"(?:起止日期|查询日期|交易日期|数据时间范围|交易时段|账单统计日期|Start\s+Time\s*&\s*End\s+Time)"
    r"\s*[:：]?\s*(?:\n\s*)?(?P<value>"
    r"(?:20\d{2}(?:年|[-/.])?\d{1,2}(?:月|[-/.])?\d{1,2}日?)"
    r"\s*(?:-|至|~|—|–)\s*"
    r"(?:20\d{2}(?:年|[-/.])?\d{1,2}(?:月|[-/.])?\d{1,2}日?)"
    r")",
    re.I,
)
_START_DATE_RE = re.compile(
    r"(?:起始日期|开始日期)\s*[:：]?\s*(?:\n\s*)?(?P<value>20\d{6}|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})"
)
_END_DATE_RE = re.compile(
    r"(?:终止日期|结束日期)\s*[:：]?\s*(?:\n\s*)?(?P<value>20\d{6}|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})"
)
_REVERSED_START_END_RE = re.compile(
    r"(?:起始日期|开始日期)\s*[:：]?\s*(?P<start>20\d{6}|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\s*\r?\n"
    r"\s*(?P<end>20\d{6}|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\s*\r?\n\s*(?:终止日期|结束日期)\s*[:：]?"
)
_PRINT_TIMESTAMP_RE = re.compile(
    r"(?:打印日期|打印时间)\s*[:：]?\s*(?:\n\s*)?20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}"
)
_LEDGER_HEADER_DATE_MARKERS = ("交易日期", "记账日期", "会计日期", "交易时间", "日期", "Date")
_LEDGER_HEADER_AMOUNT_MARKERS = (
    "交易发生金额",
    "交易金额",
    "发生额",
    "收入金额",
    "支出金额",
    "收入",
    "支出",
    "借方",
    "贷方",
    "Amount",
    "Debit",
    "Credit",
)
_LEDGER_HEADER_BALANCE_MARKERS = ("账户余额", "余额", "Balance")
_LEDGER_HEADER_BUSINESS_MARKERS = (
    "摘要",
    "备注",
    "附言",
    "用途",
    "对方",
    "对手",
    "交易类型",
    "交易渠道",
    "交易描述",
    "描述",
    "渠道",
    "流水",
    "凭证",
    "Description",
    "Counter",
    "Reference",
    "Remarks",
    "Type",
    "Channel",
)
_LEDGER_HEADER_ALLOWED_WORDS = tuple(
    sorted(
        {
            *_LEDGER_HEADER_DATE_MARKERS,
            *_LEDGER_HEADER_AMOUNT_MARKERS,
            *_LEDGER_HEADER_BALANCE_MARKERS,
            *_LEDGER_HEADER_BUSINESS_MARKERS,
            "序号",
            "编号",
            "币种",
            "货币",
            "钞汇",
            "现转标志",
            "现/转",
            "借/贷",
            "收支标志",
            "交易方向",
            "被冲标志",
            "经办机构",
            "交易机构",
            "交易代码",
            "代码",
            "核心流水号",
            "柜员流水号",
            "柜员流水",
            "交易流水号",
            "流水号",
            "日志号",
            "对方开户行",
            "对方账号/卡号",
            "对方户名/账号",
            "收(付)方名称",
            "收(付)方账号",
            "对方账户",
            "对方账号",
            "对方户名",
            "对方名称",
            "对方行名",
            "对方行号",
            "凭证类型",
            "凭证种类",
            "凭证号码",
            "凭证号",
            "Currency",
            "Transaction",
            "Counterparty",
            "Name",
            "Serial Number",
            "SerialNumber",
            "Teller's",
            "时间",
            "号",
            "交易",
            "本次",
            "信息",
            "元",
            "出账",
            "入账",
        },
        key=len,
        reverse=True,
    )
)
_LEDGER_HEADER_VALUE_RE = re.compile(
    r"(?:19|20)\d{2}(?:\d{4}|[-/.]\d{1,2}[-/.]\d{1,2})|[+-]?\d[\d,]*\.\d"
)
_LEDGER_HEADER_TOKEN_RE = re.compile(
    "|".join(re.escape(marker) for marker in _LEDGER_HEADER_ALLOWED_WORDS),
    re.I,
)
_LEDGER_HEADER_NOISE_RE = re.compile(r"[\s/\\|,，、:：;；()（）\[\]【】._\-]+")
_LEDGER_TITLE_MARKERS = ("对账单", "交易明细", "账户明细", "账户交易", "交易流水", "Transaction Statement")
_DIRECTION_AMOUNT_TOTAL_PATTERNS = {
    "debit": (
        re.compile(r"(?:借方(?:发生|累计|合计)?(?:总)?(?:额|金额)|支出(?:交易)?总额|支出总金额|总支出金额)\s*[:：]?\s*(?P<value>[\d,]+\.\d{1,2})"),
        re.compile(r"本页支出合计\s*[:：]?\s*(?P<value>[\d,]+\.\d{1,2})"),
    ),
    "credit": (
        re.compile(r"(?:贷方(?:发生|累计|合计)?(?:总)?(?:额|金额)|收入(?:交易)?总额|收入总金额|总收入金额)\s*[:：]?\s*(?P<value>[\d,]+\.\d{1,2})"),
        re.compile(r"本页收入合计\s*[:：]?\s*(?P<value>[\d,]+\.\d{1,2})"),
    ),
}
_DIRECTION_COUNT_TOTAL_PATTERNS = {
    "debit": (
        re.compile(
            r"(?:借方(?:发生|累计|合计)?(?:总)?笔数|"
            r"支出(?:交易)?总笔数|总支出(?:入)?笔数)"
            r"[ \t]*[:：][ \t]*(?P<value>\d+)"
        ),
    ),
    "credit": (
        re.compile(
            r"(?:贷方(?:发生|累计|合计)?(?:总)?笔数|"
            r"收入(?:交易)?总笔数|总收入笔数)"
            r"[ \t]*[:：][ \t]*(?P<value>\d+)"
        ),
    ),
}
_DECLARED_TRANSACTION_TOTAL_PATTERNS = (
    re.compile(
        r"(?<!借方)(?<!贷方)(?<!支出)(?<!收入)"
        r"(?:交易总笔数|合计笔数|总笔数)"
        r"[ \t]*[:：][ \t]*(?P<value>\d+)"
    ),
)


def _source_atom_page_number(page_id: Any) -> int:
    match = re.search(r"(?P<page>\d+)$", str(page_id or ""))
    return int(match.group("page")) if match else 0


def _spdb_header_facts_from_source_atoms(parse_result: Any) -> dict[int, tuple[str, ...]]:
    """Recover a small, geometry-bound SPDB header contract omitted from page text."""
    try:
        from docmirror.plugins._runtime.evidence_access import text_atoms
    except ImportError:
        return {}

    atoms = [
        atom
        for atom in text_atoms(parse_result)
        if str(atom.get("source_kind") or "").strip().casefold() == "pdf_native"
        and float(atom.get("confidence") or 0.0) >= 0.9
        and isinstance(atom.get("bbox"), list)
        and len(atom["bbox"]) >= 4
        and _source_atom_page_number(atom.get("page_id")) > 0
    ]
    by_page: dict[int, list[dict[str, Any]]] = {}
    for atom in atoms:
        by_page.setdefault(_source_atom_page_number(atom.get("page_id")), []).append(atom)

    facts: dict[int, tuple[str, ...]] = {}
    for page, page_atoms in by_page.items():
        page_facts: list[str] = []
        for label_text, value_pattern in (
            ("客户名称", None),
            ("账号", re.compile(r"[0-9*＊-]{6,}")),
            ("账单币种", None),
            ("账单统计日期", _SPDB_PERIOD_VALUE_RE),
        ):
            labels = [atom for atom in page_atoms if str(atom.get("text") or "").strip() == label_text]
            if len(labels) != 1:
                page_facts = []
                break
            label_bbox = [float(value) for value in labels[0]["bbox"][:4]]
            label_center = (label_bbox[1] + label_bbox[3]) / 2.0
            same_line = sorted(
                (
                    atom
                    for atom in page_atoms
                    if atom is not labels[0]
                    and abs((float(atom["bbox"][1]) + float(atom["bbox"][3])) / 2.0 - label_center) <= 2.0
                    and float(atom["bbox"][0]) >= label_bbox[2] - 2.0
                ),
                key=lambda atom: float(atom["bbox"][0]),
            )
            if label_text == "账单统计日期":
                candidates = {
                    str(atom.get("text") or "").strip()
                    for atom in same_line
                    if value_pattern is not None and value_pattern.fullmatch(str(atom.get("text") or "").strip())
                }
                if not candidates:
                    compact = "".join(str(atom.get("text") or "").strip() for atom in same_line[:12])
                    candidates = {match.group(0) for match in _SPDB_PERIOD_VALUE_RE.finditer(compact)}
                if len(candidates) != 1:
                    page_facts = []
                    break
                page_facts.append(f"账单统计日期 {candidates.pop()}")
                continue

            candidates = [str(atom.get("text") or "").strip() for atom in same_line if str(atom.get("text") or "").strip()]
            if value_pattern is not None:
                candidates = [value for value in candidates if value_pattern.fullmatch(value)]
            else:
                candidates = [value for value in candidates if value not in {"Customer", "Name", "Currency"}]
            if not candidates:
                page_facts = []
                break
            page_facts.append(f"{label_text} {candidates[0]}")
        if page_facts:
            facts[page] = tuple(page_facts)
    return facts


def page_texts_from_parse_result(parse_result: Any) -> list[tuple[int, str]]:
    """Build page-local text scopes without relying on flattened PDF reading order."""
    if parse_result is None:
        return []
    header_facts = _spdb_header_facts_from_source_atoms(parse_result)
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
            headers = [str(header or "").strip() for header in getattr(table, "headers", []) or []]
            if any(headers):
                parts.append(" ".join(headers))
            for row in getattr(table, "rows", []) or []:
                cells = [str(getattr(cell, "text", "") or "").strip() for cell in getattr(row, "cells", []) or []]
                if any(cells):
                    parts.append(" ".join(cells))
        text = "\n".join(part for part in parts if part)
        for header_fact in header_facts.get(page_number, ()):
            if header_fact not in text:
                text = f"{text}\n{header_fact}" if text else header_fact
        if text:
            page_texts.append((page_number, text))
    return page_texts


def _count_scopes(text: str, page_texts: Iterable[tuple[int, str]] | None) -> list[tuple[int | None, str]]:
    scoped = [(int(page), str(value or "")) for page, value in (page_texts or ()) if str(value or "").strip()]
    if scoped:
        return scoped
    parts = [part for part in re.split(r"\f", str(text or "")) if part.strip()]
    return [(None, part) for part in (parts or [str(text or "")])]


def _single_source_page_fact(text: str) -> tuple[int, int] | None:
    facts = {
        (int(match.group("page")), int(match.group("total"))) for match in _SOURCE_PAGE_RE.finditer(str(text or ""))
    }
    return facts.pop() if len(facts) == 1 else None


def _spdb_statement_total_facts(text: str) -> set[int]:
    return {
        count
        for match in _SPDB_STATEMENT_TOTAL_PATTERN.finditer(str(text or ""))
        if (count := _safe_count(int(match.group("count"))))
    }


def _has_labelled_count_context(text: str) -> bool:
    """Return whether text proves an actual ledger document, not just ledger-ish words."""
    source = str(text or "")
    return _ledger_document_contract(source) is not None or _strict_count_header_contract(source)


def _normalized_contract_value(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _period_contract_value(text: str) -> str:
    periods = {_normalized_contract_value(match.group("value")) for match in _PERIOD_RE.finditer(str(text or ""))}
    if len(periods) == 1:
        return periods.pop()
    starts = {_normalized_contract_value(match.group("value")) for match in _START_DATE_RE.finditer(str(text or ""))}
    ends = {_normalized_contract_value(match.group("value")) for match in _END_DATE_RE.finditer(str(text or ""))}
    if len(starts) == len(ends) == 1:
        return f"{starts.pop()}~{ends.pop()}"
    reversed_periods = {
        (
            _normalized_contract_value(match.group("start")),
            _normalized_contract_value(match.group("end")),
        )
        for match in _REVERSED_START_END_RE.finditer(str(text or ""))
    }
    if len(reversed_periods) == 1:
        start, end = reversed_periods.pop()
        return f"{start}~{end}"
    return ""


def _unique_identity_value(text: str, pattern: re.Pattern[str]) -> str:
    values = {
        _normalized_contract_value(match.group("value"))
        for match in pattern.finditer(str(text or ""))
        if _normalized_contract_value(match.group("value"))
    }
    return values.pop() if len(values) == 1 else ""


def _header_identity_contract(text: str) -> tuple[str, str]:
    try:
        from docmirror.plugins.bank_statement.institution_authority import extract_identity_from_header

        identity = extract_identity_from_header(str(text or ""))
    except (ImportError, TypeError, ValueError):
        return "", ""
    return (
        _normalized_contract_value(str(identity.get("account_number") or "")),
        _normalized_contract_value(str(identity.get("account_holder") or "")),
    )


def _identity_pair_from_text(text: str) -> tuple[str, str]:
    source = str(text or "")
    account = _unique_identity_value(source, _ACCOUNT_IDENTITY_RE)
    holder = _unique_identity_value(source, _HOLDER_IDENTITY_RE)
    if account and holder:
        return account, holder
    parsed_account, parsed_holder = _header_identity_contract(source)
    account = account or parsed_account
    holder = holder or parsed_holder
    if account and holder:
        return account, holder

    # Some column-major covers serialize the values immediately before their
    # labels. Bind only the exact adjacent pair; never scan business rows for a
    # convenient account/name token.
    reverse = re.search(
        r"(?m)^\s*(?P<holder>[^\r\n:：]{2,80})\s*\r?\n\s*(?:账户名称|客户名称)\s*[:：]?\s*\r?\n"
        r"\s*(?P<account>[0-9*＊-]{6,})\s*\r?\n\s*账号\s*[:：]?\s*$",
        source,
    )
    if reverse is not None:
        return (
            _normalized_contract_value(reverse.group("account")),
            _normalized_contract_value(reverse.group("holder")),
        )
    return "", ""


def _ledger_header_block_is_bounded(lines: list[str]) -> bool:
    """Recognize one compact ordered ledger header, never document-wide keywords."""
    if not lines:
        return False
    normalized_lines = [re.sub(r"\s+", "", unicodedata.normalize("NFKC", line)) for line in lines]
    compact = "".join(normalized_lines)
    if not compact or len(compact) > 256 or _LEDGER_HEADER_VALUE_RE.search(compact):
        return False
    tokens: list[tuple[str, int]] = []
    line_breaks: list[int] = []
    offset = 0
    for line_index, line in enumerate(normalized_lines):
        line_breaks.extend([line_index] * len(line))
        offset += len(line)
    cursor = 0
    for match in _LEDGER_HEADER_TOKEN_RE.finditer(compact):
        if _LEDGER_HEADER_NOISE_RE.sub("", compact[cursor : match.start()]):
            return False
        token_line = line_breaks[match.start()] if match.start() < len(line_breaks) else len(lines) - 1
        tokens.append((match.group(0), token_line))
        cursor = match.end()
    if _LEDGER_HEADER_NOISE_RE.sub("", compact[cursor:]) or len(tokens) < 4:
        return False

    roles: list[str] = []
    for token, _line_index in tokens:
        if any(marker.casefold() in token.casefold() for marker in _LEDGER_HEADER_DATE_MARKERS):
            roles.append("date")
        elif any(marker.casefold() in token.casefold() for marker in _LEDGER_HEADER_AMOUNT_MARKERS):
            roles.append("amount")
        elif any(marker.casefold() in token.casefold() for marker in _LEDGER_HEADER_BALANCE_MARKERS):
            roles.append("balance")
        elif any(marker.casefold() in token.casefold() for marker in _LEDGER_HEADER_BUSINESS_MARKERS):
            roles.append("business")
        else:
            roles.append("neutral")
    semantic_roles = [role for role in roles if role != "neutral"]
    required = {"date", "amount", "balance", "business"}
    if not required.issubset(semantic_roles):
        return False

    # Ordinary ledger headers begin with their date/time plane; business fields
    # may appear before or after amount/balance, but amount must precede balance.
    first_date = semantic_roles.index("date")
    first_amount = semantic_roles.index("amount")
    first_balance = semantic_roles.index("balance")
    if first_date == 0 and first_amount < first_balance:
        recognized_by_line = {line_index for _token, line_index in tokens}
        if len(lines) > 1 and len(recognized_by_line) != len(lines):
            return False
        return True

    # A few rotated, column-major native layouts serialize their headers in
    # visual rather than left-to-right order. Admit only the dense, explicit
    # form: many labelled columns, a direction field, and multiple date roles.
    explicit_direction = any(
        marker in compact for marker in ("借/贷", "借贷", "收支标志", "交易方向", "被冲标志")
    )
    return bool(
        len(tokens) >= 10
        and explicit_direction
        and semantic_roles.count("date") >= 2
        and first_amount < first_balance < len(semantic_roles) - 1
        and semantic_roles[-1] == "date"
    )


def _bounded_ledger_header_span(text: str, *, max_lines: int = 16) -> tuple[int, int] | None:
    """Return the first compact source-header span in document reading order."""
    lines = _nonempty_source_lines(text)
    for start in range(len(lines)):
        for end in range(start + 1, min(len(lines), start + max_lines) + 1):
            if _ledger_header_block_is_bounded(lines[start:end]):
                return start, end
            # Once a transaction value enters the window it cannot become a
            # source header by appending more unrelated text.
            if _LEDGER_HEADER_VALUE_RE.search("".join(lines[start:end])):
                break
    return None


def _has_bounded_ledger_header(text: str) -> bool:
    return _bounded_ledger_header_span(text) is not None


def _ledger_document_contract(text: str) -> tuple[str, str, str] | None:
    """Return stable identity/period only for a complete transaction-ledger layout."""
    source = str(text or "")
    account, holder = _identity_pair_from_text(source)
    period = _period_contract_value(source)
    if not account or not holder or not period or not _has_bounded_ledger_header(source):
        return None
    return account, holder, period


def _strict_count_header_contract(text: str) -> bool:
    """Admit a count-only cover page only when it directly owns a full ledger header."""
    source = str(text or "")
    holder = _unique_identity_value(source, _HOLDER_IDENTITY_RE)
    if not holder or _PRINT_TIMESTAMP_RE.search(source) is None:
        return False
    count_match = next((match for pattern in _COUNT_PATTERNS for match in pattern.finditer(source)), None)
    if count_match is None:
        return False
    tail = source[count_match.end() :]
    header_span = _bounded_ledger_header_span(tail)
    return header_span is not None and header_span[0] <= 2


def _paired_direction_amount_totals(text: str) -> tuple[float, float] | None:
    """Read both direction totals from one bounded aggregate scope."""
    source = str(text or "")
    values: dict[str, float] = {}
    for direction, patterns in _DIRECTION_AMOUNT_TOTAL_PATTERNS.items():
        facts = {
            round(_float(match.group("value")), 2)
            for pattern in patterns
            for match in pattern.finditer(source)
        }
        if len(facts) != 1:
            return None
        values[direction] = facts.pop()
    return values["debit"], values["credit"]


def _paired_direction_count_totals(text: str) -> tuple[int, int] | None:
    """Read one debit count and one credit count from the same summary scope."""
    source = str(text or "")
    values: dict[str, int] = {}
    for direction, patterns in _DIRECTION_COUNT_TOTAL_PATTERNS.items():
        facts: set[int] = set()
        for pattern in patterns:
            for match in pattern.finditer(source):
                count = int(match.group("value"))
                if 0 <= count <= 10000:
                    facts.add(count)
        if len(facts) != 1:
            return None
        values[direction] = facts.pop()
    return values["debit"], values["credit"]


def _declared_transaction_total(text: str) -> int | None:
    """Return one direction-neutral transaction total from a bounded scope."""
    facts = {
        count
        for pattern in _DECLARED_TRANSACTION_TOTAL_PATTERNS
        for match in pattern.finditer(str(text or ""))
        if (count := _safe_count(int(match.group("value"))))
    }
    return facts.pop() if len(facts) == 1 else None


def _stable_repeated_header_summary_evidence(
    scopes: list[tuple[int | None, str]],
) -> RowCountEvidence | None:
    """Resolve a repeated issuer summary only from its complete source plane.

    Some native ledgers repeat a geometry-bound statement header on every page.
    The generic count patterns also see the debit and credit subtotals inside
    labels such as ``借方总笔数``.  Treating any one of those numbers as the
    document total is unsafe.  Instead, require every ordered page scope to
    repeat the same identity, period, ordered ledger header, direction counts,
    and direction amounts, then reconcile the neutral total to both counts.
    """
    if not scopes or [page for page, _text in scopes] != list(range(1, len(scopes) + 1)):
        return None

    contracts: list[tuple[str, str, str]] = []
    summaries: list[tuple[int, int, int, float, float]] = []
    for _page, scoped_text in scopes:
        contract = _ledger_document_contract(scoped_text)
        direction_counts = _paired_direction_count_totals(scoped_text)
        direction_amounts = _paired_direction_amount_totals(scoped_text)
        declared_total = _declared_transaction_total(scoped_text)
        if contract is None or direction_counts is None or direction_amounts is None or declared_total is None:
            return None
        debit_count, credit_count = direction_counts
        if debit_count + credit_count != declared_total:
            return None
        contracts.append(contract)
        summaries.append((declared_total, debit_count, credit_count, *direction_amounts))

    if len(set(contracts)) != 1 or len(set(summaries)) != 1:
        return None
    return RowCountEvidence(summaries[0][0], "header_total", 0.98, scopes[0][0])


def _header_total_scope_is_bounded(
    scoped_text: str,
    *,
    document_contract: tuple[str, str, str] | None,
    repeated_count_scope: bool,
    terminal_scope: bool,
) -> bool:
    if _strict_count_header_contract(scoped_text):
        return True
    if document_contract is None:
        return False
    if _paired_direction_amount_totals(scoped_text) is not None:
        return terminal_scope
    # A stable count repeated by the issuer on every page is a header fact, not
    # an incidental number in a business row. Identity/period/header structure
    # is still proved independently by ``document_contract``.
    return repeated_count_scope and _has_bounded_ledger_header(scoped_text)


def _source_label_value(text: str, label_pattern: str) -> str:
    match = re.search(
        rf"(?:^|\n){label_pattern}[ \t]*(?:\n[ \t]*)?(?P<value>[^\r\n]+)",
        str(text or ""),
        re.M,
    )
    return str(match.group("value") if match else "").strip()


def _spdb_scope_identity(text: str) -> tuple[str, str, str] | None:
    customer_values = _source_label_values(text, r"客户名称(?:[ \t]+Customer[ \t]+Name)?")
    account_values = {
        value
        for value in _source_label_values(text, r"账号(?:[ \t]+Account[ \t]+Number)?")
        if re.fullmatch(r"[0-9*＊-]{6,}", value)
    }
    currency_values = _source_label_values(text, r"账单币种(?:[ \t]+Currency)?")
    customer = next(iter(customer_values), "") if len(customer_values) == 1 else ""
    account = next(iter(account_values), "") if len(account_values) == 1 else ""
    if not customer or not account:
        return None
    currency_tokens = {
        token
        for value in currency_values
        for token in value.split()
        if token in {"人民币", "CNY"}
    }
    currency_token = "人民币" if "人民币" in currency_tokens else "CNY" if "CNY" in currency_tokens else ""
    if not currency_token:
        return None
    return customer, account, currency_token


def _parse_spdb_date(value: str, pattern: re.Pattern[str]) -> date | None:
    match = pattern.fullmatch(value.strip())
    if match is None:
        return None
    try:
        return date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except ValueError:
        return None


def _spdb_scope_period(text: str, *, monthly: bool) -> tuple[date, date] | None:
    raw_facts: set[str] = set()
    for line in str(text or "").splitlines():
        if re.match(r"^\s*\u8d26\u5355\u7edf\u8ba1\u65e5\u671f", line) is None:
            continue
        matches = [match.group(0).strip() for match in _SPDB_PERIOD_VALUE_RE.finditer(line)]
        raw_facts.update(matches)
    periods: set[tuple[date, date]] = set()
    for raw in raw_facts:
        if "年" in raw:
            parts = re.split(r"\s*[-至~]\s*", raw, maxsplit=1)
        else:
            parts = re.split(r"(?<=\d)\s*[-至~]\s*(?=20\d{2}/)", raw, maxsplit=1)
        if len(parts) == 2:
            left, right = (part.strip() for part in parts)
            pattern = _SPDB_CN_DATE_RE if "年" in raw else _SPDB_SLASH_DATE_RE
            start, end = _parse_spdb_date(left, pattern), _parse_spdb_date(right, pattern)
        else:
            end = _parse_spdb_date(raw, _SPDB_CN_DATE_RE)
            start = date(end.year, end.month, 1) if end is not None else None
        if start is not None and end is not None and start <= end:
            periods.add((start, end))
    if len(periods) != 1:
        return None
    start, end = periods.pop()
    if monthly and (
        start.day != 1
        or (start.year, start.month) != (end.year, end.month)
        or end.day != monthrange(end.year, end.month)[1]
    ):
        return None
    return start, end


def _spdb_statement_segment_evidence(
    scopes: list[tuple[int | None, str]],
) -> RowCountEvidence | None:
    """Sum complete, identity-stable and period-contiguous statement segments."""
    if not scopes or [page for page, _text in scopes] != list(range(1, len(scopes) + 1)):
        return None
    monthly_layout = all(all(label in text for label in _SPDB_MONTHLY_CONTEXT) for _page, text in scopes)
    annual_layout = all(all(label in text for label in _SPDB_ANNUAL_CONTEXT) for _page, text in scopes)
    if not monthly_layout and not annual_layout:
        return None
    identities = [_spdb_scope_identity(text) for _page, text in scopes]
    if any(identity is None for identity in identities) or len(set(identities)) != 1:
        return None

    markers = [_single_source_page_fact(scoped_text) for _page, scoped_text in scopes]
    if any(marker is None for marker in markers):
        return None

    statement_totals: list[tuple[int, int]] = []
    statement_periods: list[tuple[date, date]] = []
    cursor = 0
    while cursor < len(scopes):
        first_marker = markers[cursor]
        if first_marker is None:
            return None
        local_page, declared_pages = first_marker
        segment_end = cursor + declared_pages
        if local_page != 1 or declared_pages <= 0 or segment_end > len(scopes):
            return None
        if markers[cursor:segment_end] != [(page, declared_pages) for page in range(1, declared_pages + 1)]:
            return None

        first_facts = _spdb_statement_total_facts(scopes[cursor][1])
        continuation_facts = [
            _spdb_statement_total_facts(scoped_text) for _page, scoped_text in scopes[cursor + 1 : segment_end]
        ]
        if len(first_facts) != 1 or any(continuation_facts):
            return None
        segment_periods = [
            _spdb_scope_period(text, monthly=monthly_layout) for _page, text in scopes[cursor:segment_end]
        ]
        if any(period is None for period in segment_periods) or len(set(segment_periods)) != 1:
            return None
        statement_periods.append(segment_periods[0])  # type: ignore[arg-type]
        statement_totals.append((int(scopes[cursor][0] or 0), first_facts.pop()))
        cursor = segment_end

    if not statement_totals:
        return None
    if any(current[0] != previous[1] + timedelta(days=1) for previous, current in zip(statement_periods, statement_periods[1:])):
        return None
    if len(statement_totals) == 1:
        page, count = statement_totals[0]
        return RowCountEvidence(count, "header_total", 0.94, page)
    return RowCountEvidence(
        _safe_count(sum(count for _page, count in statement_totals)),
        "statement_header_totals",
        0.97,
    )


def _cumulative_footer_evidence(scopes: list[tuple[int | None, str]]) -> RowCountEvidence | None:
    """Resolve a cumulative footer only from every page of its exact statement plane."""
    if not scopes or [page for page, _text in scopes] != list(range(1, len(scopes) + 1)):
        return None

    cumulative_facts: list[tuple[int, int]] = []
    source_contracts: list[tuple[str, ...]] = []
    for scope_index, (page, scoped_text) in enumerate(scopes):
        source_contract = _cumulative_scope_contract(scoped_text)
        if source_contract is None:
            return None
        source_contracts.append(source_contract)
        if _single_source_page_fact(scoped_text) != (page, len(scopes)):
            return None
        facts = {
            (int(match.group("through")), int(match.group("total")))
            for match in _CUMULATIVE_TRANSACTION_FOOTER_RE.finditer(scoped_text)
        }
        if len(facts) != 1:
            return None
        cumulative_facts.append(facts.pop())

    if len(set(source_contracts)) != 1:
        return None
    totals = {total for _through, total in cumulative_facts}
    through_values = [through for through, _total in cumulative_facts]
    if (
        len(totals) != 1
        or any(through <= 0 or through > total for through, total in cumulative_facts)
        or through_values != sorted(through_values)
        or len(set(through_values)) != len(through_values)
        or through_values[-1] != next(iter(totals))
    ):
        return None
    return RowCountEvidence(
        count=through_values[-1],
        source="cumulative_footer_total",
        confidence=0.99,
        page=int(scopes[-1][0] or 0),
    )


def _source_label_values(text: str, label_pattern: str) -> set[str]:
    return {
        str(match.group("value") or "").strip()
        for match in re.finditer(
            rf"(?:^|\n){label_pattern}[ \t]*(?:\n[ \t]*)?(?P<value>[^\r\n]+)",
            str(text or ""),
            re.M,
        )
        if str(match.group("value") or "").strip()
    }


def _cumulative_identity_value_map(text: str) -> dict[str, set[str]]:
    """Tokenize every vertical or inline identity fact on a cumulative page."""
    labels = tuple(item[:-1] for item in _CUMULATIVE_IDENTITY_LABELS)
    label_re = re.compile(rf"(?P<label>{'|'.join(map(re.escape, labels))})[:：]")
    values: dict[str, set[str]] = {label: set() for label in labels}
    lines = str(text or "").splitlines()
    for line_index, line in enumerate(lines):
        matches = list(label_re.finditer(line))
        for match_index, match in enumerate(matches):
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(line)
            value = line[match.end() : end].strip()
            if not value:
                next_index = line_index + 1
                while next_index < len(lines) and not lines[next_index].strip():
                    next_index += 1
                if next_index < len(lines):
                    next_line = lines[next_index].strip()
                    if not label_re.search(next_line):
                        value = next_line
            if value:
                values[str(match.group("label"))].add(value)
    return values


def _cumulative_scope_contract(text: str) -> tuple[str, ...] | None:
    """Return stable issuer identity only for the bounded ordered ledger layout."""
    source = str(text or "")
    if re.search(r"(?m)^\s*单位账户明细对账单\s*$", source) is None:
        return None
    header_pattern = r"(?m)^\s*" + r"\s*\r?\n\s*".join(map(re.escape, _CUMULATIVE_HEADER)) + r"\s*$"
    if re.search(header_pattern, source) is None:
        return None
    periods = {match.group(0).replace(" ", "") for match in _CUMULATIVE_PERIOD_RE.finditer(source)}
    if len(periods) != 1:
        return None
    identity: list[str] = [periods.pop()]
    identity_values = _cumulative_identity_value_map(source)
    for label in _CUMULATIVE_IDENTITY_LABELS:
        values = identity_values[label[:-1]]
        if len(values) != 1:
            return None
        identity.append(values.pop())
    return tuple(identity)


def _page_footer_scope_contract(text: str) -> tuple[str, str, str, str] | None:
    """Return a stable per-page contract for additive page transaction totals."""
    source = str(text or "")
    lines = _nonempty_source_lines(source)
    if not lines or not _has_bounded_ledger_header(source):
        return None
    account = _unique_identity_value(source, _ACCOUNT_IDENTITY_RE)
    holder = _unique_identity_value(source, _HOLDER_IDENTITY_RE)
    period = _period_contract_value(source)
    if not account or not holder or not period or _paired_direction_amount_totals(source) is None:
        return None
    title = _normalized_contract_value(lines[0])
    if (
        not title
        or not any(_normalized_contract_value(marker) in title for marker in _LEDGER_TITLE_MARKERS)
        or _SOURCE_PAGE_RE.search(lines[0])
        or _PAGE_COUNT_PATTERN.search(lines[0])
    ):
        return None
    return title, account, holder, period


def _complete_page_footer_counts(
    scopes: list[tuple[int | None, str]],
) -> list[tuple[int, int]] | None:
    """Read one issuer page-count fact from every contracted, numbered page."""
    if not scopes:
        return None
    facts: list[tuple[int, int]] = []
    contracts: list[tuple[str, str, str, str]] = []
    declared_total = 0
    for physical_page, scoped_text in scopes:
        contract = _page_footer_scope_contract(scoped_text)
        marker = _single_source_page_fact(scoped_text)
        counts = {
            count
            for match in _PAGE_COUNT_PATTERN.finditer(scoped_text)
            if (count := _safe_count(int(match.group("count"))))
        }
        if contract is None or marker is None or len(counts) != 1:
            return None
        contracts.append(contract)
        source_page, source_total = marker
        if physical_page is not None and int(physical_page) != source_page:
            return None
        if declared_total and source_total != declared_total:
            return None
        declared_total = source_total
        facts.append((source_page, counts.pop()))
    if (
        len(set(contracts)) != 1
        or declared_total != len(facts)
        or [page for page, _count in facts] != list(range(1, declared_total + 1))
    ):
        return None
    return facts


def _nonempty_source_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _consecutive_line_index(lines: list[str], expected: tuple[str, ...]) -> int:
    width = len(expected)
    return next(
        (
            index
            for index in range(0, max(len(lines) - width + 1, 0))
            if tuple(lines[index : index + width]) == expected
        ),
        -1,
    )


def _valid_compact_date(value: str) -> bool:
    if re.fullmatch(r"20\d{6}", value) is None:
        return False
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return False
    return True


def _valid_iso_date(value: str) -> bool:
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value) is None:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _ccb_duplicate_header_matches(line: str) -> bool:
    tokens = tuple(token for token in re.split(r"[\s|]+", str(line or "").strip(" |")) if token)
    return tokens[: len(_CCB_DUPLICATE_HEADER)] == _CCB_DUPLICATE_HEADER


def _ccb_duplicate_sequence_planes(lines: list[str], footer_index: int) -> list[list[int]] | None:
    """Read independently serialized ordinal columns under complete table headers."""
    header_indexes = [
        index for index in range(footer_index + 1, len(lines)) if _ccb_duplicate_header_matches(lines[index])
    ]
    if not header_indexes:
        return None

    planes: list[list[int]] = []
    for position, header_index in enumerate(header_indexes):
        limit = header_indexes[position + 1] if position + 1 < len(header_indexes) else len(lines)
        sequences: list[int] = []
        for line in lines[header_index + 1 : limit]:
            match = re.fullmatch(r"\|?\s*(?P<sequence>\d{1,6})(?:\s*\|\s*.*|\s+.*)?", line)
            if match is None:
                break
            sequence = int(match.group("sequence"))
            if sequences and sequence != sequences[-1] + 1:
                return None
            sequences.append(sequence)
        if not sequences:
            return None
        planes.append(sequences)
    return planes


def _ccb_primary_source_sequence(scopes: list[tuple[int | None, str]]) -> RowCountEvidence | None:
    """Measure CCB row-plane coverage when both serialized planes agree.

    Agreement catches extraction corruption but cannot prove that a shared
    terminal prefix was not omitted, so this is deliberately non-authoritative.
    """
    if not scopes or any(page is None for page, _text in scopes):
        return None
    pages = [int(page or 0) for page, _text in scopes]
    if pages != list(range(1, len(scopes) + 1)):
        return None

    page_sequences: list[list[int]] = []
    for _page, scoped_text in scopes:
        lines = _nonempty_source_lines(scoped_text)
        if len(lines) < 8 or not lines[0].startswith("卡号/账号:") or not lines[1].startswith("客户名称："):
            return None
        header_index = _consecutive_line_index(lines, _CCB_PRIMARY_HEADER)
        footer_index = next(
            (index for index, line in enumerate(lines) if line.startswith("生成时间:")),
            -1,
        )
        if header_index < 0 or footer_index <= header_index + len(_CCB_PRIMARY_HEADER):
            return None
        primary = lines[header_index + len(_CCB_PRIMARY_HEADER) : footer_index]
        if len(primary) % 5:
            return None
        sequences: list[int] = []
        for offset in range(0, len(primary), 5):
            sequence, summary, currency, cash_remittance, transaction_date = primary[offset : offset + 5]
            if (
                re.fullmatch(r"\d{1,6}", sequence) is None
                or not summary
                or currency != "人民币元"
                or cash_remittance not in {"钞", "汇"}
                or not _valid_compact_date(transaction_date)
            ):
                return None
            sequences.append(int(sequence))
        if not sequences or sequences != list(range(sequences[0], sequences[-1] + 1)):
            return None
        duplicate_planes = _ccb_duplicate_sequence_planes(lines, footer_index)
        if duplicate_planes is None or any(duplicate != sequences for duplicate in duplicate_planes):
            return None
        page_sequences.append(sequences)

    flattened = [sequence for values in page_sequences for sequence in values]
    if not flattened or flattened != list(range(flattened[0], flattened[-1] + 1)):
        return None
    return RowCountEvidence(len(flattened), "ccb_primary_source_sequence", 0.80)


def _cmb_duplicate_row_facts(lines: list[str]) -> list[tuple[str, str, str, str]] | None:
    """Read structural row identities from CMB's line-oriented duplicate plane."""
    facts: list[tuple[str, str, str, str]] = []
    for line in lines:
        cells = line.split(maxsplit=4)
        if (
            len(cells) != 5
            or not cells[4].strip()
            or not _valid_iso_date(cells[0])
            or _ISO_CURRENCY_RE.fullmatch(cells[1]) is None
            or _PRIMARY_MONEY_RE.fullmatch(cells[2]) is None
            or _PRIMARY_MONEY_RE.fullmatch(cells[3]) is None
        ):
            return None
        facts.append((cells[0], cells[1], cells[2], cells[3]))
    return facts or None


def _cmb_primary_source_rows(scopes: list[tuple[int | None, str]]) -> RowCountEvidence | None:
    """Measure CMB row-plane coverage when every serialized duplicate agrees.

    Duplicate agreement is a structural quality check, not an independent
    issuer denominator; symmetric omissions remain possible.
    """
    if not scopes or any(page is None for page, _text in scopes):
        return None
    pages = [int(page or 0) for page, _text in scopes]
    if pages != list(range(1, len(scopes) + 1)):
        return None

    total_rows = 0
    for page, scoped_text in scopes:
        lines = _nonempty_source_lines(scoped_text)
        marker_indexes = [
            index
            for index, line in enumerate(lines)
            if (match := _CMB_PAGE_MARKER_RE.fullmatch(line))
            and int(match.group("page")) == page
            and int(match.group("total")) == len(scopes)
        ]
        if len(marker_indexes) != 1:
            return None
        marker_index = marker_indexes[0]
        header_index = _consecutive_line_index(lines[:marker_index], _CMB_PRIMARY_HEADER)
        if header_index < 0:
            return None
        if page == 1 and (
            "招商银行交易流水" not in lines[:header_index]
            or _CMB_DUPLICATE_TITLE not in lines[:header_index]
        ):
            return None
        english_index = header_index + len(_CMB_PRIMARY_HEADER)
        if tuple(lines[english_index : english_index + len(_CMB_ENGLISH_HEADER)]) != _CMB_ENGLISH_HEADER:
            return None
        primary = lines[english_index + len(_CMB_ENGLISH_HEADER) : marker_index]
        row_starts = [index for index, value in enumerate(primary) if _valid_iso_date(value)]
        structural_starts = [
            index
            for index in range(max(len(primary) - 3, 0))
            if _ISO_CURRENCY_RE.fullmatch(primary[index + 1])
            and _PRIMARY_MONEY_RE.fullmatch(primary[index + 2])
            and _PRIMARY_MONEY_RE.fullmatch(primary[index + 3])
        ]
        date_shaped_starts = [index for index, value in enumerate(primary) if _PRIMARY_DATE_SHAPE_RE.fullmatch(value)]
        if not row_starts or structural_starts != row_starts or date_shaped_starts != row_starts:
            return None
        row_ends = [*row_starts[1:], len(primary)]
        if any(
            end - start < 6
            or _ISO_CURRENCY_RE.fullmatch(primary[start + 1]) is None
            or _PRIMARY_MONEY_RE.fullmatch(primary[start + 2]) is None
            or _PRIMARY_MONEY_RE.fullmatch(primary[start + 3]) is None
            for start, end in zip(row_starts, row_ends)
        ):
            # A source row must also carry at least the summary and counterparty
            # region after its four structural fields.  This rejects incidental
            # date/currency/money furniture without requiring those fields to be
            # populated in every real transaction.
            return None

        primary_facts = [
            (primary[start], primary[start + 1], primary[start + 2], primary[start + 3]) for start in row_starts
        ]
        post_marker = lines[marker_index + 1 :]
        if page == 1 or post_marker:
            if (
                len(post_marker) < 4
                or post_marker[0] != _CMB_DUPLICATE_TITLE
                or tuple(post_marker[1].split()) != _CMB_PRIMARY_HEADER
                or post_marker[2] != " ".join(_CMB_ENGLISH_HEADER)
            ):
                return None
            duplicate_facts = _cmb_duplicate_row_facts(post_marker[3:])
            if duplicate_facts != primary_facts:
                return None
        total_rows += len(row_starts)
    return RowCountEvidence(total_rows, "cmb_primary_source_rows", 0.80)


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
    complete_page_scope_sequence = (
        not has_page_scopes or [page for page, _scoped_text in scopes] == list(range(1, len(scopes) + 1))
    )
    scoped_contracts = [
        contract
        for _page, scoped_text in scopes
        if (contract := _ledger_document_contract(scoped_text)) is not None
    ]
    unique_scoped_contracts = list(dict.fromkeys(scoped_contracts))
    document_contract = (
        unique_scoped_contracts[0]
        if len(unique_scoped_contracts) == 1
        else _ledger_document_contract(str(text or ""))
        if not has_page_scopes
        else None
    )

    if evidence := _cumulative_footer_evidence(scopes):
        return evidence

    if evidence := _stable_repeated_header_summary_evidence(scopes):
        return evidence

    for page, scoped_text in scopes:
        column_major = _column_major_direction_totals(scoped_text)
        scope_contract = _ledger_document_contract(scoped_text)
        if column_major is not None and document_contract is not None and scope_contract == document_contract:
            debit, credit, _debit_total, _credit_total = column_major
            return RowCountEvidence(debit + credit, "split_footer", 0.98, page)
        for pattern in _SPLIT_COUNT_PATTERNS:
            match = pattern.search(scoped_text)
            if (
                match
                and document_contract is not None
                and scope_contract == document_contract
                and _paired_direction_amount_totals(scoped_text) is not None
            ):
                count = _safe_count(int(match.group("debit")) + int(match.group("credit")))
                if count:
                    return RowCountEvidence(count, "split_footer", 0.98, page)

    # Repeated statement headers can contain page- or period-local totals.  A
    # first-match policy turns whichever page happened to be read first into a
    # document total.  Accept a labelled total only when all occurrences agree;
    # otherwise continue to page-footers or transaction anchors.
    header_totals: list[tuple[int | None, int]] = []
    header_totals_conflict = False
    raw_header_facts: list[set[int]] = []
    for scope_index, (page, scoped_text) in enumerate(scopes):
        patterns = _COUNT_PATTERNS if page is not None else (_SINGLE_LINE_COUNT_PATTERN,)
        raw_header_facts.append(
            {
                count
                for pattern in patterns
                for match in pattern.finditer(scoped_text)
                if (count := _safe_count(int(match.group("count"))))
            }
        )
    repeated_count_scope = (
        len(scopes) > 1
        and complete_page_scope_sequence
        and all(len(facts) == 1 for facts in raw_header_facts)
        and len({next(iter(facts)) for facts in raw_header_facts}) == 1
    )
    for scope_index, (page, scoped_text) in enumerate(scopes):
        patterns = _COUNT_PATTERNS if page is not None else (_SINGLE_LINE_COUNT_PATTERN,)
        page_facts: set[int] = set()
        if _header_total_scope_is_bounded(
            scoped_text,
            document_contract=document_contract,
            repeated_count_scope=repeated_count_scope,
            terminal_scope=complete_page_scope_sequence and scope_index == len(scopes) - 1,
        ):
            for pattern in patterns:
                for match in pattern.finditer(scoped_text):
                    count = _safe_count(int(match.group("count")))
                    if count:
                        page_facts.add(count)
        if len(page_facts) > 1:
            header_totals_conflict = True
            header_totals = []
            break
        if page_facts:
            header_totals.append((page, page_facts.pop()))
    # Some rotated SPDB statements expose the four summary values only in the
    # flattened text.  Use that compatibility plane strictly as a fallback:
    # appending its duplicate facts to bounded page facts destroys the page
    # provenance needed to prove that several first-page statement totals are
    # additive.
    if not header_totals and not header_totals_conflict:
        flattened_facts = (
            {
                count
                for match in _SPDB_STATEMENT_TOTAL_PATTERN.finditer(str(text or ""))
                if (count := _safe_count(int(match.group("count"))))
            }
            if _has_labelled_count_context(text)
            else set()
        )
        if len(flattened_facts) > 1:
            header_totals_conflict = True
        elif flattened_facts:
            header_totals.append((None, flattened_facts.pop()))
    scoped_spdb_totals = any(_SPDB_STATEMENT_TOTAL_PATTERN.search(scoped_text) for _page, scoped_text in scopes)
    if scoped_spdb_totals and not header_totals_conflict:
        # Additive statement totals require a complete, independently paginated
        # segment census.  A subset of first pages or repeated snippets cannot
        # establish how many statements belong to the document.
        if evidence := _spdb_statement_segment_evidence(scopes):
            return evidence
        header_totals_conflict = True
    header_totals_conflict = header_totals_conflict or len({count for _, count in header_totals}) > 1
    if header_totals and not header_totals_conflict:
        page, count = header_totals[0]
        return RowCountEvidence(count, "header_total", 0.94, page)

    page_counts = _complete_page_footer_counts(scopes)
    if page_counts is not None:
        return RowCountEvidence(
            count=_safe_count(sum(count for _page, count in page_counts)),
            source="page_footer",
            confidence=0.90,
            page=page_counts[0][0] if len(page_counts) == 1 else None,
        )

    # PageContent can be sparse even when the flattened canonical text retained
    # the footer. Only same-line or explicitly page-local labels are accepted in
    # this compatibility fallback, so a following page number cannot become a count.
    if has_page_scopes:
        flattened_contract = _ledger_document_contract(str(text or ""))
        flattened_header_totals = (
            [
                _safe_count(int(match.group("count")))
                for match in _SINGLE_LINE_COUNT_PATTERN.finditer(str(text or ""))
            ]
            if _header_total_scope_is_bounded(
                str(text or ""),
                document_contract=flattened_contract,
                repeated_count_scope=repeated_count_scope,
                terminal_scope=complete_page_scope_sequence,
            )
            else []
        )
        flattened_header_totals = [count for count in flattened_header_totals if count]
        if not header_totals_conflict and flattened_header_totals and len(set(flattened_header_totals)) == 1:
            return RowCountEvidence(flattened_header_totals[0], "header_total", 0.90)
        # Additive page counts are accepted only from page-scoped contracts.
        # Flattening destroys the per-page identity, header and aggregate proof.

    # These issuer templates expose a complete, independently countable source
    # plane followed by a second flattened/duplicated representation.  Generic
    # date-anchor counting sees the query period or only part of a column-major
    # page.  Count only the strictly bounded primary plane after all labelled
    # aggregate counts and page footers have had priority.
    for source_census in (_ccb_primary_source_sequence, _cmb_primary_source_rows):
        evidence = source_census(scopes)
        if evidence is not None:
            return evidence

    header_scopes = [(page, scoped_text) for page, scoped_text in scopes if _has_borderless_source_header(scoped_text)]
    anchored_counts = [_count_borderless_transaction_anchors(scoped_text) for _, scoped_text in header_scopes]
    # A transaction anchor count is document evidence only when every populated
    # page scope carries the ledger header.  One late-page header followed by a
    # single row (seen in a 9-page CIB continuation ledger) is page-local—not a
    # defensible denominator for the whole statement.
    complete_header_coverage = not has_page_scopes or len(header_scopes) == len(scopes)
    if complete_header_coverage and anchored_counts and all(count > 0 for count in anchored_counts):
        return RowCountEvidence(
            count=_safe_count(sum(anchored_counts)),
            source="page_transaction_anchors",
            # Complete header coverage bounds the observed anchor plane, but a
            # terminal source row can still be absent from that plane. Keep it
            # below public count authority.
            confidence=0.80,
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

    # Keep grid and borderless candidates in separate page-order streams.  A
    # source PDF can expose a header-only grid (or one page-sized aggregate
    # cell) while its positioned words retain every physical row.  Interleaving
    # those two representations makes the cross-page composer flush at every
    # page boundary, so each representation is composed independently.
    native_page_tables: list[list[list[str]]] = []
    borderless_page_tables: list[list[list[str]]] = []
    carried_borderless_header: tuple[list[str], list[float]] | None = None
    page_scope_titles: list[str] = []
    native_money_hints = _native_money_hints(pdf_path)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_scope_titles.append(_native_statement_scope_title(page))
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
                        native_page_tables.append(normalized)
                if not native_tables:
                    for table_index, table in enumerate(page.extract_tables() or []):
                        normalized = _annotate_native_grid_matrix(
                            _normalize_table(table),
                            page_number=page_number,
                            table_index=table_index,
                            money_hints=native_money_hints.get(page_number, {}),
                        )
                        if normalized:
                            native_page_tables.append(normalized)

                borderless, detected_header = _recover_borderless_native_page_with_header(
                    page,
                    page_number,
                    inherited_header=carried_borderless_header,
                )
                if detected_header is not None:
                    carried_borderless_header = detected_header
                if borderless:
                    borderless_page_tables.append(borderless)
    except Exception as exc:
        logger.debug("[BankWideTableRecovery] native PDF table recovery failed: %s", exc)
        return []

    scope_title = _stable_native_statement_scope_title(page_scope_titles)
    if scope_title:
        native_page_tables = [
            _attach_private_table_column(table, "_document_scope_text", scope_title) for table in native_page_tables
        ]
        borderless_page_tables = [
            _attach_private_table_column(table, "_document_scope_text", scope_title) for table in borderless_page_tables
        ]

    candidates: list[list[list[str]]] = []
    for page_tables in (native_page_tables, borderless_page_tables):
        if len(page_tables) > 1:
            candidates.extend(_recover_cross_page_wide_tables(page_tables))
    for table in [*native_page_tables, *borderless_page_tables]:
        wide = _select_wide_bank_table(table)
        if wide:
            candidates.append(wide)
    candidates = _dedupe_tables(candidates)

    if candidates:
        logger.info("[BankWideTableRecovery] recovered %d native wide table(s)", len(candidates))
    return candidates


_NATIVE_STATEMENT_SCOPE_TITLES = ("中国建设银行个人活期账户收入交易明细",)


def _native_statement_scope_title(page: Any) -> str:
    """Return an exact source-PDF title used to scope transaction direction."""
    try:
        text = str(page.extract_text() or "")
    except Exception:
        return ""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return next((title for title in _NATIVE_STATEMENT_SCOPE_TITLES if title in compact), "")


def _stable_native_statement_scope_title(titles: list[str]) -> str:
    nonempty = [title for title in titles if title]
    if not nonempty or len(set(nonempty)) != 1:
        return ""
    # Require every populated page to repeat the exact title when there is more
    # than one page.  A nearby prose mention cannot scope a whole document.
    return nonempty[0] if len(nonempty) == len(titles) else ""


def _attach_private_table_column(table: list[list[str]], name: str, value: str) -> list[list[str]]:
    if not table or not value:
        return table
    header = list(table[0])
    if name in header:
        return table
    return [[*header, name], *[[*list(row), value] for row in table[1:]]]


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
    amount_columns = _source_amount_columns(headers)
    if date_column < 0 or balance_column < 0 or not amount_columns:
        return matrix

    source_headers = [*headers, "_source_page", "_source_bbox", "_source_table_id", "_source_row_index"]
    out = [*matrix[:header_index], source_headers]
    hint_queues = {key: list(values) for key, values in money_hints.items()}
    table_id = f"native:p{page_number}:t{table_index}"
    for row_index, source_row in enumerate(matrix[header_index + 1 :], start=header_index + 1):
        row = list(source_row)
        if signed_amount_column >= 0:
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
    lines = [str(line or "").strip() for line in str(text or "").splitlines()]
    header_index = next((index for index, line in enumerate(lines) if _looks_like_borderless_header_text(line)), -1)
    if header_index < 0:
        return 0
    transaction_lines: list[str] = []
    for line in lines[header_index + 1 :]:
        compact = re.sub(r"\s+", "", normalize_header_cell(line))
        if _is_transaction_count_footer(compact):
            break
        transaction_lines.append(line)

    inline_count = sum(1 for line in transaction_lines if _BORDERLESS_ROW_RE.search(line))
    stacked_count = 0
    for index, line in enumerate(transaction_lines):
        if not _BORDERLESS_DATE_RE.fullmatch(line):
            continue
        lookahead = transaction_lines[index + 1 : index + 5]
        money_values = [value for value in lookahead if _BORDERLESS_BALANCE_RE.fullmatch(re.sub(r"\s+", "", value))]
        compact_date = re.sub(r"\s+", "", line)
        inline_duplicate = any(
            compact_date in re.sub(r"\s+", "", candidate)
            and all(re.sub(r"\s+", "", value) in re.sub(r"\s+", "", candidate) for value in money_values[:2])
            for candidate in transaction_lines[index + 1 :]
        )
        if len(money_values) >= 2 and not inline_duplicate:
            stacked_count += 1
    return inline_count + stacked_count


def _is_transaction_count_footer(compact: str) -> bool:
    """Return whether a page-local line starts the bank transaction footer."""
    markers = (
        "总收入笔数",
        "总支出笔数",
        "总支出入笔数",
        "收入总笔数",
        "支出总笔数",
        "借方合计笔数",
        "贷方合计笔数",
        "本页交易笔数",
        "本页合计",
    )
    return any(marker in compact for marker in markers)


def _has_borderless_source_header(text: str) -> bool:
    return any(_looks_like_borderless_header_text(line) for line in str(text or "").splitlines())


def _is_header_continuation_line(words: list[dict[str, Any]]) -> bool:
    """Return whether a line after a source header is still bilingual header text."""
    raw = re.sub(r"\s+", "", "".join(str(word.get("text") or "") for word in words)).lower()
    normalized = re.sub(r"\s+", "", normalize_header_cell(raw))
    markers = (
        "date",
        "currency",
        "transaction",
        "amount",
        "balance",
        "type",
        "counter",
        "party",
        "日期",
        "币种",
        "金额",
        "余额",
        "摘要",
        "对方",
        "对手",
    )
    return any(marker in raw or marker in normalized for marker in markers)


def _recover_borderless_native_page(page: Any, page_number: int) -> list[list[str]]:
    """Recover a native source-column ledger from word coordinates.

    The branch is deliberately gated by the complete source header. It does not
    run for generic prose, payment documents, scanned OCR, or ledgers whose
    source column roles are ambiguous.
    """
    table, _ = _recover_borderless_native_page_with_header(page, page_number)
    return table


def _extract_native_words(page: Any, *, use_text_flow: bool) -> list[dict[str, Any]]:
    """Read native words while excluding proven colored stamp overlays.

    A few issuer PDFs place a red electronic-seal identifier directly over a
    black ledger value.  pdfplumber exposes both as ordinary words unless color
    metadata is requested.  Exclude only a chromatic word whose bbox overlaps
    a neutral source word; colored ledger values that do not overlap another
    value remain untouched.
    """
    kwargs = {
        "x_tolerance": 1,
        "y_tolerance": 1,
        "keep_blank_chars": False,
        "use_text_flow": use_text_flow,
    }
    try:
        extracted = page.extract_words(
            **kwargs,
            extra_attrs=["fontname", "non_stroking_color", "stroking_color"],
        )
    except (KeyError, TypeError, ValueError):
        # Compatibility for synthetic/older page providers that do not expose
        # character color attributes.  Their source words remain lossless.
        extracted = page.extract_words(**kwargs)
    words = [dict(word) for word in extracted if str(word.get("text") or "").strip()]
    return [
        word
        for word in words
        if not (
            _native_word_is_chromatic(word)
            and any(
                other is not word and _native_word_is_neutral(other) and _native_word_bbox_overlaps(word, other)
                for other in words
            )
        )
    ]


def _native_word_color(word: dict[str, Any]) -> tuple[float, ...] | None:
    value = word.get("non_stroking_color")
    if not isinstance(value, (tuple, list)) or not value:
        return None
    try:
        return tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None


def _native_word_is_chromatic(word: dict[str, Any]) -> bool:
    color = _native_word_color(word)
    return color is not None and len(color) >= 3 and max(color[:3]) - min(color[:3]) >= 0.35


def _native_word_is_neutral(word: dict[str, Any]) -> bool:
    color = _native_word_color(word)
    return color is not None and (len(color) == 1 or max(color[:3]) - min(color[:3]) <= 0.08)


def _native_word_bbox_overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_x0 = float(left.get("x0") or 0.0)
    left_x1 = float(left.get("x1") or left_x0)
    left_top = float(left.get("top") or 0.0)
    left_bottom = float(left.get("bottom") or left_top)
    right_x0 = float(right.get("x0") or 0.0)
    right_x1 = float(right.get("x1") or right_x0)
    right_top = float(right.get("top") or 0.0)
    right_bottom = float(right.get("bottom") or right_top)
    return (
        min(left_x1, right_x1) - max(left_x0, right_x0) > 0.5
        and min(left_bottom, right_bottom) - max(left_top, right_top) > 0.5
    )


def _recover_borderless_native_page_with_header(
    page: Any,
    page_number: int,
    *,
    inherited_header: tuple[list[str], list[float]] | None = None,
) -> tuple[list[list[str]], tuple[list[str], list[float]] | None]:
    """Recover one native page and return the reusable document header.

    Some issuer exports print the full ledger header only on page one.  The
    physical x coordinates are stable on continuation pages, so the detected
    header can be carried forward within the same source PDF.  The inherited
    header is never used until transaction anchors and valid money/balance
    cells independently prove that a continuation page contains ledger rows.
    """
    try:
        words = _extract_native_words(page, use_text_flow=False)
    except Exception:
        return [], None
    if not words:
        return [], None

    lines = _group_native_words_by_line(words)
    header_spec = next(
        ((line, spec) for line in lines if (spec := _borderless_header_spec(line)) is not None),
        None,
    )
    reusable_header: tuple[list[str], list[float]] | None = None
    if header_spec is not None:
        header_words, (source_headers, starts) = header_spec
        # Slight font-baseline differences can split one visual header into two
        # extracted lines.  Rebuild the source spec from the narrow visual band
        # before concluding that an adjacent field is absent.
        header_top = min(float(word.get("top") or 0.0) for word in header_words)
        header_band_bottom = max(float(word.get("top") or 0.0) for word in header_words) + 3.0
        expanded_header_words = [
            word for word in words if header_top - 0.5 <= float(word.get("top") or 0.0) <= header_band_bottom
        ]
        expanded_spec = _borderless_header_spec(expanded_header_words)
        if expanded_spec is not None and len(expanded_spec[0]) >= len(source_headers):
            header_words = expanded_header_words
            source_headers, starts = expanded_spec

        local_header = (list(source_headers), list(starts))
        if inherited_header is not None and not _is_usable_native_header(*local_header):
            # Preserve the first complete document header.  Some later pages
            # render adjacent headings without a measurable gap; accepting that
            # malformed local variant would merge amount/balance/location and
            # flush the cross-page ledger.
            source_headers, starts = (list(inherited_header[0]), list(inherited_header[1]))
            reusable_header = (list(source_headers), list(starts))
        elif _is_usable_native_header(*local_header):
            if inherited_header is not None and _same_native_header_labels(inherited_header[0], local_header[0]):
                # Keep stable source labels for cross-page composition while
                # using this page's coordinates for word-to-cell assignment.
                source_headers = list(inherited_header[0])
                reusable_header = (list(inherited_header[0]), list(inherited_header[1]))
            else:
                reusable_header = local_header
        elif inherited_header is not None:
            source_headers, starts = (list(inherited_header[0]), list(inherited_header[1]))
            reusable_header = (list(source_headers), list(starts))
        else:
            return [], None

        header_index = next((index for index, line in enumerate(lines) if line == header_words), -1)
        header_bottom = max(float(word.get("bottom") or word.get("top") or 0.0) for word in header_words)
        # ``header_words`` may be the expanded band rather than an exact member
        # of ``lines``.  Locate the original line for continuation detection.
        if header_index < 0:
            header_index = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if any(word in header_words for word in line) and _borderless_header_spec(line) is not None
                ),
                -1,
            )
        if header_index >= 0:
            for continuation in lines[header_index + 1 :]:
                if any(_BORDERLESS_DATE_RE.fullmatch(str(word.get("text") or "").strip()) for word in continuation):
                    break
                if not _is_header_continuation_line(continuation):
                    break
                header_bottom = max(
                    header_bottom,
                    max(float(word.get("bottom") or word.get("top") or 0.0) for word in continuation),
                )
    elif inherited_header is not None:
        source_headers, starts = (list(inherited_header[0]), list(inherited_header[1]))
        header_words = []
        header_bottom = 0.0
        header_index = -1
        reusable_header = (list(source_headers), list(starts))
    else:
        return [], None

    date_column = _source_date_column(source_headers)
    amount_columns = _source_amount_columns(source_headers)
    balance_column = _source_balance_column(source_headers)
    if date_column < 0 or not amount_columns or balance_column < 0:
        return [], reusable_header

    column_boundaries = _native_vertical_column_boundaries(page, source_headers, starts)

    column_words = [
        (
            word,
            _semantic_native_column_index(
                word,
                source_headers,
                starts,
                column_boundaries=column_boundaries,
            ),
        )
        for word in words
        if float(word.get("top") or 0.0) > header_bottom
    ]
    sequence_column = _source_sequence_column(source_headers)
    anchor_column = sequence_column if sequence_column >= 0 else date_column
    if sequence_column >= 0:
        anchors = [
            word
            for word, column in column_words
            if column == anchor_column and re.fullmatch(r"\d{1,9}", str(word.get("text") or "").strip())
        ]
    else:
        anchors = [
            word
            for word, column in column_words
            if column == anchor_column and _BORDERLESS_DATE_RE.fullmatch(str(word.get("text") or "").strip())
        ]
    anchors.sort(key=_word_vertical_center)
    if sequence_column < 0 and anchors:
        footer_marker_centers = [
            _word_vertical_center(word)
            for word in words
            if any(
                marker in normalize_header_cell(str(word.get("text") or "")) for marker in _BORDERLESS_FOOTER_MARKERS
            )
        ]
        # A print/query date is sometimes positioned in the transaction-date
        # column immediately before a footer label.  It is not a row anchor.
        # Drop only a trailing, visually isolated date next to a proven footer
        # marker; ordinary consecutive transaction dates remain untouched.
        while anchors and any(
            abs(marker_center - _word_vertical_center(anchors[-1])) <= 16.0 for marker_center in footer_marker_centers
        ):
            if len(anchors) == 1:
                anchors.pop()
                break
            centers_so_far = [_word_vertical_center(word) for word in anchors]
            trailing_gap = centers_so_far[-1] - centers_so_far[-2]
            prior_gaps = [
                current - previous
                for previous, current in zip(centers_so_far, centers_so_far[1:-1])
                if current > previous
            ]
            expected_gap = median(prior_gaps) if prior_gaps else 12.0
            if trailing_gap <= max(expected_gap * 2.0, 24.0):
                break
            anchors.pop()
    if not anchors:
        return [], reusable_header

    centers = [_word_vertical_center(word) for word in anchors]
    gaps = [current - previous for previous, current in zip(centers, centers[1:]) if current > previous]
    typical_gap = median(gaps) if gaps else 18.0
    # Narrative cells are often three lines high while the date/sequence sits
    # on their first or second line.  A midpoint boundary cuts the final bank or
    # account line off the row.  Native ledgers consistently begin the next
    # row's content shortly before its anchor, so reserve a small leading band
    # for that next row and let the preceding row own everything before it.
    row_leading_band = min(6.0, max(3.0, typical_gap * 0.25))
    footer_candidates = [
        float(word.get("top") or 0.0)
        for word in words
        if _word_vertical_center(word) > centers[-1]
        and (
            any(marker in normalize_header_cell(str(word.get("text") or "")) for marker in _BORDERLESS_FOOTER_MARKERS)
            or re.fullmatch(r"[_—–-]{20,}", str(word.get("text") or "").strip()) is not None
        )
    ]
    for line in lines:
        line_top = min(float(word.get("top") or 0.0) for word in line)
        if line_top <= centers[-1]:
            continue
        joined_line = re.sub(r"\s+", "", "".join(str(word.get("text") or "") for word in line))
        if _SOURCE_PAGE_RE.search(joined_line):
            footer_candidates.append(line_top)
    footer_top = min(
        footer_candidates,
        default=float(getattr(page, "height", centers[-1] + typical_gap)),
    )
    if footer_candidates:
        # A footer marker may be serialized after its bank/date prefix.  If a
        # large blank band separates the final transaction from that prefix,
        # rewind the boundary to the first word in the footer cluster.  This
        # keeps flow-order validation from seeing a footer date as another row
        # anchor while preserving ordinary wrapped row tails near the anchor.
        pre_footer_tops = sorted(
            {
                float(word.get("top") or 0.0)
                for word in words
                if centers[-1] < float(word.get("top") or 0.0) < footer_top
            }
        )
        previous_top = centers[-1]
        rewind_threshold = max(typical_gap * 2.0, 24.0)
        for candidate_top in pre_footer_tops:
            if candidate_top - previous_top > rewind_threshold:
                footer_top = candidate_top
                break
            previous_top = candidate_top

    # Native content-stream order is a stronger row-ownership signal than the
    # y baseline for ledgers whose wrapped party/account lines begin above the
    # following row's date.  Keep geometric words for headers and x columns,
    # but use a second flow-ordered view to bound a row from its date token to
    # the next date token.  The branch is accepted only when it proves exactly
    # the same anchors and every slice is a valid ledger row; otherwise the
    # generic geometric path below remains authoritative.
    flow_slices: list[list[tuple[dict[str, Any], int]]] = []
    try:
        flow_words = [
            word
            for word in _extract_native_words(page, use_text_flow=True)
            if header_bottom < _word_vertical_center(word) < footer_top
        ]
    except Exception:
        flow_words = []
    flow_column_words = [
        (
            word,
            _semantic_native_column_index(
                word,
                source_headers,
                starts,
                column_boundaries=column_boundaries,
            ),
        )
        for word in flow_words
    ]
    flow_anchor_indexes = [
        index
        for index, (word, column) in enumerate(flow_column_words)
        if column == anchor_column
        and (
            re.fullmatch(r"\d{1,9}", str(word.get("text") or "").strip())
            if sequence_column >= 0
            else _BORDERLESS_DATE_RE.fullmatch(str(word.get("text") or "").strip())
        )
    ]
    if len(flow_anchor_indexes) == len(anchors):
        # Some native ledgers serialize the columns before the date anchor in
        # source order (voucher type/number, then date/time). Date-to-date
        # slicing assigns the following row's prefix to the preceding row. Move
        # each start backward over the contiguous, visually co-located prefix
        # columns and use those starts as the row boundaries instead.
        prefix_band = min(8.0, max(3.0, typical_gap * 0.45))
        flow_row_starts: list[int] = []
        for anchor_index in flow_anchor_indexes:
            anchor_center = _word_vertical_center(flow_column_words[anchor_index][0])
            row_start = anchor_index
            cursor = anchor_index - 1
            while cursor >= 0:
                prefix_word, prefix_column = flow_column_words[cursor]
                if prefix_column >= anchor_column:
                    break
                if abs(_word_vertical_center(prefix_word) - anchor_center) > prefix_band:
                    break
                if _is_header_continuation_line([prefix_word]):
                    break
                row_start = cursor
                cursor -= 1
            flow_row_starts.append(row_start)

        first_anchor = flow_anchor_indexes[0]
        first_row_start = flow_row_starts[0]
        first_anchor_center = _word_vertical_center(flow_column_words[first_anchor][0])
        leading_flow_words = [
            item
            for item in flow_column_words[:first_row_start]
            if header_bottom < _word_vertical_center(item[0])
            and first_anchor_center - max(row_leading_band, 8.0)
            <= _word_vertical_center(item[0])
            <= first_anchor_center + 1.0
            and not _is_header_continuation_line([item[0]])
        ]
        candidate_slices: list[list[tuple[dict[str, Any], int]]] = []
        for index, start in enumerate(flow_row_starts):
            end = flow_row_starts[index + 1] if index + 1 < len(flow_row_starts) else len(flow_column_words)
            row_slice = list(flow_column_words[start:end])
            if index == 0 and leading_flow_words:
                row_slice = [*leading_flow_words, *row_slice]
            candidate_slices.append(row_slice)
        candidate_cells: list[list[str]] = []
        for row_words in candidate_slices:
            cells = [
                _join_native_cell_words([word for word, col in row_words if col == column])
                for column in range(len(source_headers))
            ]
            _repair_native_summary_signed_money_spill(
                cells,
                source_headers,
                amount_columns=amount_columns,
            )
            candidate_cells.append(cells)
        if all(
            _valid_borderless_row(
                cells,
                date_column=date_column,
                amount_columns=amount_columns,
                balance_column=balance_column,
            )
            for cells in candidate_cells
        ):
            flow_slices = candidate_slices

    header = [
        *source_headers,
        _NATIVE_SOURCE_RAW_JSON_COLUMN,
        _NATIVE_SOURCE_REPAIR_JSON_COLUMN,
        "_source_page",
        "_source_bbox",
    ]
    rows: list[list[str]] = [list(header)]
    for index, anchor in enumerate(anchors):
        # Wrapped fields in these ledgers are commonly printed above and below
        # the date/sequence baseline.  Midpoints are safe for the dense regular
        # rows, while the first row must also include a wrapped fragment between
        # the header and its baseline.
        if flow_slices:
            row_words = flow_slices[index]
        else:
            lower = header_bottom if index == 0 else centers[index] - row_leading_band
            if index + 1 < len(anchors):
                upper = centers[index + 1] - row_leading_band
            else:
                upper = min(footer_top, centers[index] + max(typical_gap, 18.0))
            row_words = [
                (word, column) for word, column in column_words if lower <= _word_vertical_center(word) < upper
            ]
        cells = [
            _join_native_cell_words([word for word, col in row_words if col == column])
            for column in range(len(source_headers))
        ]
        anchor_text = str(anchor.get("text") or "").strip()
        if sequence_column >= 0:
            if not re.fullmatch(r"\d{1,9}", re.sub(r"\s+", "", cells[sequence_column])):
                cells[sequence_column] = anchor_text
        elif not _BORDERLESS_DATE_RE.search(cells[date_column]):
            cells[date_column] = anchor_text
        source_raw = dict(zip(source_headers, cells, strict=True))
        repair = _repair_native_summary_signed_money_spill(
            cells,
            source_headers,
            amount_columns=amount_columns,
        )
        if not _valid_borderless_row(
            cells,
            date_column=date_column,
            amount_columns=amount_columns,
            balance_column=balance_column,
        ):
            continue
        bbox = _native_row_bbox([word for word, _ in row_words])
        source_raw_json = (
            json.dumps(source_raw, ensure_ascii=False, separators=(",", ":")) if repair is not None else ""
        )
        repair_json = json.dumps(repair, ensure_ascii=False, separators=(",", ":")) if repair is not None else ""
        rows.append(
            [
                *cells,
                source_raw_json,
                repair_json,
                str(page_number),
                ",".join(f"{value:.3f}" for value in bbox),
            ]
        )
    return (rows if len(rows) > 1 else []), reusable_header


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
        previous_group = groups[-1]
        previous = previous_group[-1]
        gap = float(word.get("x0") or 0.0) - float(previous.get("x1") or previous.get("x0") or 0.0)
        previous_text = unicodedata.normalize(
            "NFKC",
            "".join(str(item.get("text") or "").strip() for item in previous_group),
        )
        word_text = unicodedata.normalize("NFKC", str(word.get("text") or "").strip())
        # Direction and amount are distinct business fields even when the
        # issuer typesets their headings almost flush (for example
        # ``支/收`` immediately followed by ``交易金额``).
        force_boundary = _must_split_adjacent_headers(previous_text, word_text)
        if gap <= 4.0 and not force_boundary:
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


def _is_direction_header(value: str) -> bool:
    compact = re.sub(r"\s+", "", normalize_header_cell(value)).lower()
    return compact in {
        "收/支",
        "支/收",
        "借/贷",
        "借贷",
        "收支",
        "收入/支出",
        "支出/收入",
        "dcflg",
    }


def _is_amount_header(value: str) -> bool:
    compact = re.sub(r"\s+", "", normalize_header_cell(value)).lower()
    return compact in {"金额", "交易金额", "发生额", "amount"}


def _must_split_adjacent_headers(previous: str, current: str) -> bool:
    """Keep complete business headings atomic even when their glyphs touch."""
    previous_compact = re.sub(r"\s+", "", normalize_header_cell(previous)).lower()
    current_compact = re.sub(r"\s+", "", normalize_header_cell(current)).lower()
    if _is_direction_header(previous_compact) and _is_amount_header(current_compact):
        return True
    atomic = {
        "交易日期",
        "记账日期",
        "摘要",
        "交易金额",
        "发生额",
        "账户余额",
        "余额",
        "流水号",
        "交易流水号",
        "现转标志",
        "现金/转账标志",
        "交易地点",
        "对方户名",
        "对方账户",
        "对方银行",
        "渠道",
        "交易渠道",
        "交易代码",
        "经办机构",
    }
    return previous_compact in atomic and current_compact in atomic


def _looks_like_borderless_header_text(text: str) -> bool:
    return _ledger_header_block_is_bounded([str(text or "")])


def _source_date_column(headers: list[str]) -> int:
    for index, header in enumerate(headers):
        compact = normalize_header_cell(header)
        if any(marker in compact for marker in ("交易日期", "记账日期", "会计日期")):
            return index
    for index, header in enumerate(headers):
        if "交易时间" in normalize_header_cell(header):
            return index
    for index, header in enumerate(headers):
        if "日期" in normalize_header_cell(header):
            return index
    return -1


def _source_amount_columns(headers: list[str]) -> list[int]:
    exact_amount_headers = {
        "金额",
        "交易金额",
        "交易发生金额",
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


def _source_sequence_column(headers: list[str]) -> int:
    return next(
        (
            index
            for index, header in enumerate(headers)
            if normalize_header_cell(header) in {"序号", "交易序号", "流水序号", "no", "no."}
        ),
        -1,
    )


def _is_usable_native_header(headers: list[str], starts: list[float]) -> bool:
    """Return whether the header keeps required source roles in distinct cells."""
    if len(headers) != len(starts) or len(headers) < 4:
        return False
    date_column = _source_date_column(headers)
    amount_columns = _source_amount_columns(headers)
    balance_column = _source_balance_column(headers)
    return bool(
        date_column >= 0
        and amount_columns
        and balance_column >= 0
        and balance_column not in amount_columns
        and date_column != balance_column
        and date_column not in amount_columns
    )


def _native_header_roles(headers: list[str]) -> list[str]:
    roles: list[str] = []
    amount_columns = set(_source_amount_columns(headers))
    for index, header in enumerate(headers):
        normalized = normalize_header_cell(header)
        if index == _source_sequence_column(headers):
            roles.append("sequence")
        elif index == _source_date_column(headers):
            roles.append("date")
        elif index in amount_columns:
            roles.append("amount")
        elif index == _source_balance_column(headers):
            roles.append("balance")
        elif _is_direction_header(normalized):
            roles.append("direction")
        elif any(marker in normalized for marker in ("摘要", "备注", "附言", "用途")):
            roles.append("summary")
        elif any(marker in normalized for marker in ("对方", "对手")):
            roles.append("counterparty")
        else:
            roles.append(normalized)
    return roles


def _same_native_header_labels(inherited: list[str], detected: list[str]) -> bool:
    """Recognize a repeated page header despite small wording differences."""
    return len(inherited) == len(detected) and _native_header_roles(inherited) == _native_header_roles(detected)


def _native_vertical_column_boundaries(
    page: Any,
    headers: list[str],
    starts: list[float],
) -> list[float] | None:
    """Return explicit grid boundaries when native PDF line geometry proves them.

    The coordinates are optional evidence, not a new extraction branch.  Every
    header start must land in its corresponding interval and every boundary
    must recur at least twice, which excludes glyph outlines and decoration.
    """
    if len(headers) != len(starts) or not headers:
        return None
    try:
        edges = list(getattr(page, "edges", []) or [])
    except Exception:
        return None

    x_counts: dict[float, int] = {}
    for edge in edges:
        if str(edge.get("orientation") or "") != "v":
            continue
        if str(edge.get("object_type") or "") not in {"line", "rect_edge"}:
            continue
        top = float(edge.get("top") or 0.0)
        bottom = float(edge.get("bottom") or top)
        if bottom - top < 4.0:
            continue
        x0 = float(edge.get("x0") or 0.0)
        x1 = float(edge.get("x1") or x0)
        if abs(x1 - x0) > 0.75:
            continue
        x = round((x0 + x1) / 2.0, 1)
        x_counts[x] = x_counts.get(x, 0) + 1

    candidates = sorted(x for x, count in x_counts.items() if count >= 2)
    width = len(headers) + 1
    if len(candidates) < width:
        return None
    valid_windows: list[list[float]] = []
    for offset in range(len(candidates) - width + 1):
        boundaries = candidates[offset : offset + width]
        if all(boundaries[index] - 1.0 <= starts[index] <= boundaries[index + 1] + 1.0 for index in range(len(starts))):
            valid_windows.append(boundaries)
    if len(valid_windows) != 1:
        return None
    return valid_windows[0]


def _semantic_native_column_index(
    word: dict[str, Any],
    headers: list[str],
    starts: list[float],
    *,
    column_boundaries: list[float] | None = None,
) -> int:
    """Assign a word to a source column using its left edge and semantic role.

    Centered headings are not reliable cell boundaries: a long value in the
    next column can begin to the left of the midpoint between heading starts.
    Prefer exact value roles first, then use left-edge starts.  The latter also
    keeps a sequence and its following summary separate when the summary begins
    inside the visually wide sequence-heading cell.
    """
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(word.get("text") or "")))
    x0 = float(word.get("x0") or 0.0)
    x1 = float(word.get("x1") or x0)

    transaction_time_column = next(
        (
            index
            for index, header in enumerate(headers)
            if re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(header or "")))
            in {"交易时间", "时间", "TransactionTime"}
        ),
        -1,
    )
    if 0 <= transaction_time_column < len(starts):
        right = starts[transaction_time_column + 1] if transaction_time_column + 1 < len(starts) else float("inf")
        if _NATIVE_TIME_RE.fullmatch(text) and starts[transaction_time_column] - 1.0 <= x0 and x1 <= right + 1.5:
            return transaction_time_column
        next_header = (
            re.sub(
                r"\s+",
                "",
                unicodedata.normalize("NFKC", str(headers[transaction_time_column + 1] or "")),
            )
            if transaction_time_column + 1 < len(headers)
            else ""
        )
        # A long summary can begin left of its centered heading and overlap the
        # apparent time-cell midpoint.  Only an exact time token belongs to the
        # dedicated transaction-time field; narrative crossing the following
        # summary start belongs to that source summary column.
        if (
            not _NATIVE_TIME_RE.fullmatch(text)
            and not _BORDERLESS_DATE_RE.fullmatch(text)
            and "摘要" in next_header
            and x0 < right
            and x1 >= right - 1.0
        ):
            return transaction_time_column + 1

    date_column = _source_date_column(headers)
    if 0 <= date_column < len(starts) and _NATIVE_TIME_RE.fullmatch(text):
        date_header = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(headers[date_column] or "")))
        right = starts[date_column + 1] if date_column + 1 < len(starts) else float("inf")
        # A centered following heading can put the time token just past the
        # midpoint even though its complete bbox remains in the source
        # transaction-time cell.
        if "交易时间" in date_header and starts[date_column] - 1.0 <= x0 and x1 <= right + 1.5:
            return date_column

    note_column = next(
        (
            index
            for index, header in enumerate(headers)
            if normalize_header_cell(header) in {"交易备注", "备注", "附言", "交易附言"}
        ),
        -1,
    )
    if 0 <= note_column < len(starts) and re.fullmatch(r"\d{3}", text):
        right = starts[note_column + 1] if note_column + 1 < len(starts) else float("inf")
        # Short transaction-note codes can begin just left of a centered note
        # heading. Resolve this exact role before applying the neighboring
        # compound-counterparty tolerance.
        if starts[note_column] - 4.0 <= x0 < right:
            return note_column

    compound_column = next(
        (
            index
            for index, header in enumerate(headers)
            if any(marker in normalize_header_cell(header) for marker in ("交易对手信息", "对方账号与户名"))
        ),
        -1,
    )
    if 0 <= compound_column < len(starts):
        left = (
            column_boundaries[compound_column]
            if column_boundaries is not None and len(column_boundaries) == len(headers) + 1
            else starts[compound_column]
        )
        right = (
            column_boundaries[compound_column + 1]
            if column_boundaries is not None and len(column_boundaries) == len(headers) + 1
            else starts[compound_column + 1]
            if compound_column + 1 < len(starts)
            else float("inf")
        )
        # Centered headers can start a few points to the right of the actual
        # compound body cell.  Preserve a whole wrapped legal-name token when
        # it overlaps that physical interval instead of assigning it to balance.
        if x0 >= left - 4.0 and x1 > left and x0 < right:
            return compound_column

    if column_boundaries is not None and len(column_boundaries) == len(headers) + 1:
        for index, (left, right) in enumerate(zip(column_boundaries, column_boundaries[1:])):
            if left - 0.5 <= x0 < right - 0.5:
                return index
        return 0 if x0 < column_boundaries[0] else len(headers) - 1

    if (sequence_column := _source_sequence_column(headers)) >= 0 and re.fullmatch(r"\d{1,9}", text):
        if sequence_column < len(starts) and abs(x0 - starts[sequence_column]) <= 24.0:
            return sequence_column

    for index, header in enumerate(headers):
        normalized = normalize_header_cell(header)
        if "余额" in normalized and _BORDERLESS_BALANCE_RE.fullmatch(text):
            if index < len(starts) and abs(x0 - starts[index]) <= 24.0:
                return index
        if index in _source_amount_columns(headers) and (
            _BORDERLESS_SIGNED_AMOUNT_RE.fullmatch(text) or _BORDERLESS_BALANCE_RE.fullmatch(text)
        ):
            if index < len(starts) and abs(x0 - starts[index]) <= 24.0:
                return index
        if any(marker in normalized for marker in ("交易日期", "记账日期", "会计日期", "日期")):
            if _BORDERLESS_DATE_RE.fullmatch(text) and index < len(starts) and abs(x0 - starts[index]) <= 24.0:
                return index

    counterparty_column = next(
        (
            index
            for index, header in enumerate(headers)
            if any(marker in normalize_header_cell(header) for marker in ("交易对手", "对方账户", "对手信息"))
        ),
        -1,
    )
    if 0 <= counterparty_column < len(starts) - 1:
        # Compound counterparty cells often indent wrapped account/bank lines;
        # those lines remain part of the source field until the next heading.
        looks_like_counterparty_detail = bool(re.fullmatch(r"[\d*\-]{8,}", text) or "银行" in text)
        if looks_like_counterparty_detail and starts[counterparty_column] - 6.0 <= x0 < starts[counterparty_column + 1]:
            return counterparty_column

    # Centered text headings can sit far to the right of their physical cells.
    # Once the balance has been identified, assign following narrative fields
    # by their ordinal header midpoints, with the first narrative cell beginning
    # immediately after the numeric balance area.
    balance_column = _source_balance_column(headers)
    if 0 <= balance_column < len(starts) - 1 and x0 >= starts[balance_column] + 24.0:
        trailing_starts = starts[balance_column + 1 :]
        trailing_boundaries = [(left + right) / 2.0 for left, right in zip(trailing_starts, trailing_starts[1:])]
        return balance_column + 1 + sum(x0 >= boundary for boundary in trailing_boundaries)

    return _column_index(x0, starts)


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
    line_values: list[str] = []
    for line in lines:
        fragments: list[str] = []
        for word in sorted(line, key=lambda item: float(item.get("x0") or 0.0)):
            fragment = str(word.get("text") or "").strip()
            compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", fragment))
            if len(compact) >= 4 and any(compact in re.sub(r"\s+", "", value) for value in fragments):
                continue
            fragments.append(fragment)
        value = "".join(fragments)
        compact_value = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
        if len(compact_value) >= 4 and any(
            compact_value in re.sub(r"\s+", "", unicodedata.normalize("NFKC", prior)) for prior in line_values
        ):
            continue
        line_values.append(value)
    return "\n".join(line_values)


def _repair_native_summary_signed_money_spill(
    cells: list[str],
    headers: list[str],
    *,
    amount_columns: list[int],
) -> dict[str, str] | None:
    """Repair one bounded narrative prefix fused to an adjacent signed amount.

    The source header must independently prove adjacent summary and amount
    roles.  Exactly one contaminated amount cell may be repaired; ambiguous
    multi-column rows fail closed.  The returned manifest describes the two
    working-cell changes so callers can retain the untouched source row.
    """
    if len(cells) != len(headers):
        return None
    roles = _native_header_roles(headers)
    candidates: list[tuple[int, int, str, str]] = []
    for amount_index in amount_columns:
        summary_index = amount_index - 1
        if (
            summary_index < 0
            or amount_index >= len(cells)
            or amount_index >= len(roles)
            or roles[summary_index] != "summary"
            or roles[amount_index] != "amount"
        ):
            continue
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(cells[amount_index] or "")))
        match = _NATIVE_SUMMARY_SIGNED_MONEY_SPILL.fullmatch(compact)
        if match is None:
            continue
        prefix = match.group("prefix")
        if prefix.lstrip("_").upper() in _NATIVE_CURRENCY_PREFIXES:
            continue
        candidates.append((summary_index, amount_index, prefix, match.group("money")))
    if len(candidates) != 1:
        return None

    summary_index, amount_index, prefix, signed_money = candidates[0]
    source_summary = str(cells[summary_index] or "")
    source_amount = str(cells[amount_index] or "")
    working_summary = f"{source_summary}{prefix}"
    cells[summary_index] = working_summary
    cells[amount_index] = signed_money
    return {
        "kind": _NATIVE_SOURCE_REPAIR_KIND,
        "summary_header": headers[summary_index],
        "amount_header": headers[amount_index],
        "summary_prefix": prefix,
        "source_summary": source_summary,
        "source_amount": source_amount,
        "working_summary": working_summary,
        "working_amount": signed_money,
        "working_transform": "identity",
    }


def _validated_native_source_repair_working_map(
    source_raw: dict[str, Any],
    business_headers: list[str],
    manifest: dict[str, Any],
) -> dict[str, str] | None:
    """Rederive the complete working row for one source-preserving repair.

    The source map is authoritative.  A caller may use the returned working
    values only when the bounded repair replays from that exact map and the
    manifest names either no transform or the deterministic native-cell clean
    applied to every business cell.  Comparing the complete map prevents a
    valid two-cell repair from authorizing unrelated working-value changes.
    """
    if (
        not business_headers
        or len(business_headers) != len(set(business_headers))
        or any(not isinstance(header, str) or header.startswith("_") for header in business_headers)
        or not isinstance(source_raw, dict)
        or list(source_raw) != business_headers
        or not all(isinstance(value, str) for value in source_raw.values())
        or not isinstance(manifest, dict)
        or set(manifest) != _NATIVE_SOURCE_REPAIR_KEYS
        or not all(isinstance(value, str) for value in manifest.values())
        or manifest.get("kind") != _NATIVE_SOURCE_REPAIR_KIND
    ):
        return None

    expected_values = [source_raw[header] for header in business_headers]
    rederived_manifest = _repair_native_summary_signed_money_spill(
        expected_values,
        business_headers,
        amount_columns=_source_amount_columns(business_headers),
    )
    if rederived_manifest is None:
        return None

    transform = manifest.get("working_transform")
    if transform == "native_cell_clean_v1":
        expected_values = [_clean_native_cell(value) for value in expected_values]
        expected_manifest = dict(rederived_manifest)
        summary_index = business_headers.index(rederived_manifest["summary_header"])
        amount_index = business_headers.index(rederived_manifest["amount_header"])
        expected_manifest["working_summary"] = expected_values[summary_index]
        expected_manifest["working_amount"] = expected_values[amount_index]
        expected_manifest["working_transform"] = transform
    elif transform == "identity":
        expected_manifest = rederived_manifest
    else:
        return None

    if manifest != expected_manifest:
        return None
    return dict(zip(business_headers, expected_values, strict=True))


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


def _complete_source_signed_direction_totals(
    records: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """Return a complete source-sign plane with owned direction on every row.

    Every amount must agree across raw, canonical_raw, and normalized magnitude.
    Every component also needs an independent source-owned side; a normalized
    direction alone cannot make the alternate reconciliation plane auditable.
    """
    from docmirror.plugins.bank_statement.styles.grid_standard import (
        source_owned_signed_directional_amount,
        source_provenanced_signed_amount,
    )

    totals = {"expense": 0.0, "income": 0.0}
    saw_negative = False
    if not records:
        return None
    for record in records:
        raw = record.get("raw")
        normalized = record.get("normalized")
        canonical_raw = record.get("canonical_raw")
        if not isinstance(raw, dict) or not isinstance(normalized, dict) or not isinstance(canonical_raw, dict):
            return None
        direction = str(normalized.get("direction") or "").strip()
        signed_amount = source_provenanced_signed_amount(raw, normalized, canonical_raw)
        if direction not in totals or signed_amount is None:
            return None
        owned_fact = source_owned_signed_directional_amount(raw, normalized, canonical_raw)
        if owned_fact != (direction, signed_amount):
            return None
        totals[direction] += signed_amount
        saw_negative = saw_negative or signed_amount < 0
    if not saw_negative:
        return None
    return round(totals["expense"], 2), round(totals["income"], 2)


def audit_bank_statement_invariants(
    records: list[dict[str, Any]],
    text: str,
    *,
    page_texts: Iterable[tuple[int, str]] | None = None,
    row_count_evidence: RowCountEvidence | None = None,
) -> list[str]:
    """Hard semantic gates for bank ledger rows against source footer totals."""
    failures: list[str] = []
    if page_gap_warning := _source_page_gap_warning(
        text,
        records=records,
        page_texts=page_texts,
    ):
        failures.append(page_gap_warning)
    resolved_evidence = resolve_row_count_evidence(text, page_texts=page_texts)
    supplied_evidence = row_count_evidence or RowCountEvidence.empty()
    authoritative_evidence = (
        supplied_evidence
        if supplied_evidence.count > 0
        and supplied_evidence.confidence >= 0.85
        and supplied_evidence.source in _ISSUER_ROW_COUNT_SOURCES
        else resolved_evidence
    )
    expected = (
        authoritative_evidence.count
        if authoritative_evidence.count > 0
        and authoritative_evidence.confidence >= 0.85
        and authoritative_evidence.source in _ISSUER_ROW_COUNT_SOURCES
        else 0
    )
    if expected > 0 and len(records) != expected:
        failures.append(f"bank_invariant_failed:row_count:{len(records)}/{expected}")

    normalized = [rec.get("normalized") or {} for rec in records]
    debit_rows = [row for row in normalized if row.get("direction") == "expense"]
    credit_rows = [row for row in normalized if row.get("direction") == "income"]
    aggregate_contract = authoritative_evidence.source == "split_footer"
    reported_counts = _reported_direction_counts(text, page_texts=page_texts) if aggregate_contract else None
    if reported_counts is not None:
        expected_debit, expected_credit = reported_counts
        direction_count_conflict = len(debit_rows) != expected_debit or len(credit_rows) != expected_credit
        balance_breaks, balance_checked = _best_balance_chain_breaks(normalized)
        explicit_direction_source = all(
            str((record.get("canonical_raw") or {}).get("direction") or "").strip() for record in records
        )
        # A few issuer PDFs print debit/credit footer counts that conflict with
        # their own row labels while every retained source-labelled row closes
        # the balance chain and amount totals.  In that internally inconsistent
        # source, row-level facts are the auditable SSOT; do not degrade a
        # complete dataset solely to match the contradictory aggregate.
        footer_counts_contradict_source_rows = (
            direction_count_conflict
            and explicit_direction_source
            and balance_checked >= max(3, len(records) - 2)
            and balance_breaks == 0
        )
        if not footer_counts_contradict_source_rows:
            if len(debit_rows) != expected_debit:
                failures.append(f"bank_invariant_failed:debit_count:{len(debit_rows)}/{expected_debit}")
            if len(credit_rows) != expected_credit:
                failures.append(f"bank_invariant_failed:credit_count:{len(credit_rows)}/{expected_credit}")
    column_major_totals = next(
        (
            parsed
            for _page, source in _count_scopes(text, page_texts)
            if (parsed := _column_major_direction_totals(source)) is not None
        ),
        None,
    ) if aggregate_contract else None
    paired_totals = _paired_direction_amount_totals(text) if aggregate_contract else None
    debit_total = column_major_totals[2] if column_major_totals else paired_totals[0] if paired_totals else None
    credit_total = column_major_totals[3] if column_major_totals else paired_totals[1] if paired_totals else None
    actual_debit_total = round(sum(_float(row.get("amount")) for row in debit_rows), 2)
    actual_credit_total = round(sum(_float(row.get("amount")) for row in credit_rows), 2)
    signed_totals_close = False
    # Signed reconciliation is an all-or-nothing alternate aggregate plane.  It
    # cannot rescue a single side, an incomplete direction census, or a footer
    # that lacks both counts and both totals.
    if (
        aggregate_contract
        and reported_counts == (len(debit_rows), len(credit_rows))
        and len(debit_rows) + len(credit_rows) == len(records)
        and debit_total is not None
        and credit_total is not None
    ):
        source_signed_totals = _complete_source_signed_direction_totals(records)
        signed_totals_close = bool(
            source_signed_totals is not None
            and abs(source_signed_totals[0] - debit_total) <= 0.01
            and abs(source_signed_totals[1] - credit_total) <= 0.01
        )
    if debit_total is not None:
        if abs(actual_debit_total - debit_total) > 0.01 and not signed_totals_close:
            failures.append(f"bank_invariant_failed:debit_total:{actual_debit_total:.2f}/{debit_total:.2f}")
    if credit_total is not None:
        if abs(actual_credit_total - credit_total) > 0.01 and not signed_totals_close:
            failures.append(f"bank_invariant_failed:credit_total:{actual_credit_total:.2f}/{credit_total:.2f}")

    filtered_income_scope = any(
        title
        in re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", str(record.get("_document_scope_text") or "")),
        )
        for record in records
        for title in _NATIVE_STATEMENT_SCOPE_TITLES
    )
    if not filtered_income_scope:
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
        column_major = _column_major_direction_totals(source)
        if column_major is not None:
            return column_major[0], column_major[1]
        for pattern in _SPLIT_COUNT_PATTERNS:
            match = pattern.search(source)
            if match:
                return int(match.group("debit")), int(match.group("credit"))
    return None


def _column_major_direction_totals(text: str) -> tuple[int, int, float, float] | None:
    """Parse the exact column-major debit/credit summary tuple.

    Some rotated exports serialize the debit total between the credit-count
    label and value. Generic cross-label regexes then mistake a money amount for
    the credit count. This strict tuple requires both counts and both 2-decimal
    totals in the observed source order before returning any evidence.
    """
    match = _COLUMN_MAJOR_DIRECTION_TOTAL_PATTERN.search(str(text or ""))
    if match is None:
        return None
    debit = _safe_count(match.group("debit"))
    credit = _safe_count(match.group("credit"))
    if not debit or not credit:
        return None
    return debit, credit, _float(match.group("debit_total")), _float(match.group("credit_total"))


def _best_balance_chain_breaks(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return min breaks across chronological and reverse-chronological order."""
    forward = _balance_chain_breaks(rows)
    backward = _balance_chain_breaks(list(reversed(rows)))
    if backward[1] > forward[1]:
        return backward
    if forward[1] > backward[1]:
        return forward
    return min(forward, backward, key=lambda item: item[0])


def _explicit_timestamp_batch_members(rows: list[dict[str, Any]]) -> list[bool]:
    """Mark contiguous equal-timestamp rows whose printed order is ambiguous."""
    timestamps: list[str] = []
    for row in rows:
        timestamp = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(row.get("timestamp") or ""))).strip()
        if (
            re.fullmatch(
                r"(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?"
                r"(?:Z|[+-]\d{2}:?\d{2})?",
                timestamp,
            )
            is None
        ):
            timestamp = ""
        timestamps.append(timestamp)
    return [
        bool(
            timestamp
            and (
                (index > 0 and timestamps[index - 1] == timestamp)
                or (index + 1 < len(timestamps) and timestamps[index + 1] == timestamp)
            )
        )
        for index, timestamp in enumerate(timestamps)
    ]


def _balance_chain_breaks(rows: list[dict[str, Any]]) -> tuple[int, int]:
    checked = 0
    breaks = 0
    prev_balance: float | None = None
    prev_sequence: int | None = None
    prev_index: int | None = None
    ambiguous_batch = _explicit_timestamp_batch_members(rows)
    for row_index, row in enumerate(rows):
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
        timestamp_order_is_known = (
            prev_index is None or not ambiguous_batch[prev_index] and not ambiguous_batch[row_index]
        )
        if prev_balance is not None and sequence_is_contiguous and timestamp_order_is_known:
            checked += 1
            expected_balance = prev_balance + amount_f if direction == "income" else prev_balance - amount_f
            if abs(round(expected_balance - balance_f, 2)) > 0.01:
                breaks += 1
        prev_balance = balance_f
        prev_sequence = sequence
        prev_index = row_index
    return breaks, checked


def _balance_chain_break_review_items(rows: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    items: list[str] = []
    prev_balance: float | None = None
    prev_row: dict[str, Any] | None = None
    prev_sequence: int | None = None
    prev_index: int | None = None
    ambiguous_batch = _explicit_timestamp_batch_members(rows)
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
        current_index = row_index - 1
        timestamp_order_is_known = (
            prev_index is None or not ambiguous_batch[prev_index] and not ambiguous_batch[current_index]
        )
        if prev_balance is not None and sequence_is_contiguous and timestamp_order_is_known:
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
        prev_index = current_index
    return items


def _sequence_number(row: dict[str, Any]) -> int | None:
    value = row.get("sequence_no")
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _source_page_gap_warning(
    text: str,
    *,
    records: list[dict[str, Any]] | None = None,
    page_texts: Iterable[tuple[int, str]] | None = None,
) -> str:
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
    parsed_pages = {int(page) for page, _source in (page_texts or ()) if isinstance(page, int) or str(page).isdigit()}
    record_pages = {page for record in records or [] if (page := _record_source_page(record)) is not None}
    # Some issuer PDFs are intentionally exported as a contiguous slice of a
    # larger statement.  This is not a parser page gap when the printed page
    # ordinals form one slice and every supplied parser page owns source rows.
    # This check proves page presence only; it deliberately makes no claim
    # about transaction-row completeness.
    ordered_printed_pages = sorted(observed)
    consecutive_printed_slice = len(ordered_printed_pages) == len(parsed_pages) and ordered_printed_pages == list(
        range(ordered_printed_pages[0], ordered_printed_pages[0] + len(ordered_printed_pages))
    )
    if (
        consecutive_printed_slice
        and parsed_pages == set(range(1, len(parsed_pages) + 1))
        and record_pages == parsed_pages
    ):
        return ""
    # Some issuer exports omit the printed footer on their final page even
    # though the parser retained that page and its source-bounded rows.  A
    # footer absence is not a page absence when all declared page scopes were
    # parsed and every footer-missing page owns source-provenanced records.
    # Row-count completeness remains the responsibility of independently
    # propagated row-count evidence; emitted record sequences are not evidence
    # here.  Keep warning on subsets, unproven pages, and genuine page gaps.
    if (
        parsed_pages == set(range(1, declared_total + 1))
        and set(missing).issubset(record_pages)
    ):
        return ""
    return (
        "bank_review:source_page_gap:"
        f"observed={len(observed)}/{declared_total}:"
        f"missing_ranges={_compact_ranges(missing)}:"
        "action=manual_review"
    )


def _record_source_page(record: dict[str, Any]) -> int | None:
    source = record.get("source")
    if isinstance(source, dict):
        value = source.get("source_page")
        try:
            page = int(str(value or "").strip())
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
        page_range = source.get("page_range")
        if isinstance(page_range, (tuple, list)) and len(page_range) == 2:
            try:
                start, end = (int(page_range[0]), int(page_range[1]))
            except (TypeError, ValueError):
                start = end = 0
            if start > 0 and start == end:
                return start
    for pool_name in ("raw", "canonical_raw", "normalized"):
        pool = record.get(pool_name)
        if not isinstance(pool, dict):
            continue
        try:
            page = int(str(pool.get("_source_page") or "").strip())
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
    return None


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
    if not re.search(
        r"(?<!\d)\d{8}(?!\d)|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?<!\d)\d{2}[-/]\d{2}(?!\d)",
        joined,
    ):
        return False
    if not re.search(r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{1,2}", joined):
        return False
    return True


def _clean_cross_page_native_table(table: list[list[str]]) -> list[list[str]]:
    """Clean working cells while retaining exact private source JSON."""
    metadata_header = next(
        (
            [str(cell or "").strip() for cell in row]
            for row in table[:8]
            if _NATIVE_SOURCE_RAW_JSON_COLUMN in row and _NATIVE_SOURCE_REPAIR_JSON_COLUMN in row
        ),
        None,
    )
    if metadata_header is None:
        return [[_clean_native_cell(cell) for cell in row] for row in table]

    source_index = metadata_header.index(_NATIVE_SOURCE_RAW_JSON_COLUMN)
    repair_index = metadata_header.index(_NATIVE_SOURCE_REPAIR_JSON_COLUMN)
    cleaned_table: list[list[str]] = []
    for row in table:
        raw_values = [str(cell or "") for cell in row]
        cleaned = [_clean_native_cell(cell) for cell in raw_values]
        if max(source_index, repair_index) < len(raw_values):
            # These compact JSON cells are source/provenance, not display
            # text. Never collapse whitespace inside exact source values.
            cleaned[source_index] = raw_values[source_index].strip()
            cleaned[repair_index] = raw_values[repair_index].strip()
            synchronized = _synchronize_cross_page_repair_manifest(
                raw_values,
                cleaned,
                metadata_header,
            )
            if synchronized:
                cleaned[repair_index] = synchronized
        cleaned_table.append(cleaned)
    return cleaned_table


def _synchronize_cross_page_repair_manifest(
    raw_row: list[str],
    cleaned_row: list[str],
    headers: list[str],
) -> str:
    """Clean only proven working values inside a native repair manifest."""
    if (
        headers.count(_NATIVE_SOURCE_RAW_JSON_COLUMN) != 1
        or headers.count(_NATIVE_SOURCE_REPAIR_JSON_COLUMN) != 1
    ):
        return ""
    source_index = headers.index(_NATIVE_SOURCE_RAW_JSON_COLUMN)
    repair_index = headers.index(_NATIVE_SOURCE_REPAIR_JSON_COLUMN)
    source_json = raw_row[source_index].strip()
    repair_json = raw_row[repair_index].strip()
    if not source_json or not repair_json:
        return ""
    business_indexes = [index for index, header in enumerate(headers) if not header.startswith("_")]
    business_headers = [headers[index] for index in business_indexes]
    if not business_headers or len(business_headers) != len(set(business_headers)):
        return ""
    try:
        source_raw = json.loads(source_json)
        manifest = json.loads(repair_json)
    except (TypeError, ValueError):
        return ""
    if (
        not isinstance(source_raw, dict)
        or list(source_raw) != business_headers
        or not all(isinstance(value, str) for value in source_raw.values())
        or not isinstance(manifest, dict)
        or set(manifest) != _NATIVE_SOURCE_REPAIR_KEYS
        or not all(isinstance(value, str) for value in manifest.values())
        or manifest.get("kind") != _NATIVE_SOURCE_REPAIR_KIND
    ):
        return ""

    expected_working = [source_raw[header] for header in business_headers]
    expected_manifest = _repair_native_summary_signed_money_spill(
        expected_working,
        business_headers,
        amount_columns=_source_amount_columns(business_headers),
    )
    raw_business = [raw_row[index] for index in business_indexes]
    if expected_manifest != manifest or raw_business != expected_working:
        return ""
    cleaned_business = [cleaned_row[index] for index in business_indexes]
    expected_cleaned = [_clean_native_cell(value) for value in expected_working]
    if cleaned_business != expected_cleaned:
        return ""

    synchronized = dict(manifest)
    summary_header = manifest["summary_header"]
    amount_header = manifest["amount_header"]
    synchronized["working_summary"] = cleaned_business[business_headers.index(summary_header)]
    synchronized["working_amount"] = cleaned_business[business_headers.index(amount_header)]
    synchronized["working_transform"] = "native_cell_clean_v1"
    return json.dumps(synchronized, ensure_ascii=False, separators=(",", ":"))


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
        table = _clean_cross_page_native_table(table)
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
