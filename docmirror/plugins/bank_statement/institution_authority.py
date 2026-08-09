# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Institution Authority Stack (IAS) for bank statement plugin.

Resolves issuing-bank hints from Mirror entities, header text, and layout profile
variants — never from transaction-body counterparty names alone.

Pipeline role: ``style_detector`` and ``build_style_context`` call
``resolve_institution_hint`` / ``resolve_institution_from_context``.

Key exports: ``resolve_institution_hint``, ``extract_identity_from_header``.
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docmirror.plugins.bank_statement.header_resolve import normalize_bank_matching_text
from docmirror.plugins.bank_statement.institution import detect_registered_institution, match_institution

if TYPE_CHECKING:
    from docmirror.plugins.bank_statement.context import StyleContext

_HEADER_LIMIT = 4000
_HEADER_BANK_PATTERNS = (
    re.compile(
        r"(?<!对方)开户行(?:\s+The Bank(?:\s+of Account Opening)?)?[:：]?\s*"
        r"([\u4e00-\u9fa5A-Za-z（）()·\s]{4,60}?)(?:\s{2,}|账单统计日期|起始日期|From\(|\n|$)",
        re.I,
    ),
    re.compile(r"开户机构[:：]?\s*([\u4e00-\u9fa5A-Za-z（）()·\s]{4,60}?)(?:\s{2,}|币种|年份|月份|账号|户名|\n)"),
    re.compile(r"(?<!对方)开户行\s+([\u4e00-\u9fa5A-Za-z（）()·\s]{4,40}?)(?:\s{2,}|起始日期|From\(|\n)"),
    re.compile(
        r"(?<!对方)开户行[:：]?\s*"
        r"([\u4e00-\u9fa5A-Za-z（）()·\s]{4,60}?)(?:\s{2,}|币种|账号|户名|\n)"
    ),
    re.compile(r"Bank Name\s+([\u4e00-\u9fa5A-Za-z（）()·\s]{4,60}?)(?:\s{2,}|From\(|\n)", re.I),
)
_IDENTITY_SKIP_KEYWORDS = (
    "币种",
    "交易",
    "申请",
    "账号",
    "开户行",
    "起始",
    "截止",
    "日期",
    "页码",
    "年份",
    "月份",
    "户名",
    "开户机构",
    "明细",
    "人民币",
    "序号",
    "摘要",
    "余额",
    "对方",
    "借方",
    "贷方",
    "承前",
    "承后",
    "合计",
    "小计",
    "总计",
    "发生数",
    "发生额",
)
_NON_INSTITUTION_TOKENS = (
    "网上银行",
    "网银",
    "网银结算",
    "支付系统",
    "财税库行",
    "清算中心",
    "个人银行",
    "个人银行结算账户",
)


def _header_text(full_text: str) -> str:
    text = normalize_bank_matching_text(full_text or "")
    if not text:
        return ""
    identity_anchors = (
        "客户名称 Customer Name",
        "账户名称 Account Name",
        "账号 Account Number",
        "账单统计日期 Start Time",
    )
    anchor_positions = [text.find(anchor) for anchor in identity_anchors if text.find(anchor) >= 0]
    if anchor_positions:
        start = max(min(anchor_positions) - 300, 0)
        text = text[start : start + _HEADER_LIMIT]
        table_start = _transaction_header_start(text)
        return text[:table_start] if table_start is not None else text
    ledger_start = text.find("|序号|")
    if ledger_start < 0:
        ledger_start = text.find("|No.")
    if ledger_start > 0:
        return text[:ledger_start]
    text = text[:_HEADER_LIMIT]
    table_start = _transaction_header_start(text)
    return text[:table_start] if table_start is not None else text


def _transaction_header_start(text: str) -> int | None:
    """Return the start of a multi-line transaction header cluster."""
    role_patterns = {
        "sequence": re.compile(r"^(?:序号|Serial|Num)$", re.I),
        "date": re.compile(r"^(?:交易日期|记账日期|Trans\s*Date)$", re.I),
        "amount": re.compile(
            r"^(?:交易金额|发生额|借方(?:\s*[（(]出账[）)])?|贷方(?:\s*[（(]入账[）)])?|Trans\s*Amt|Amount)$",
            re.I,
        ),
        "balance": re.compile(r"^(?:余额(?:摘要)?|账户余额|Balance)$", re.I),
        "direction": re.compile(
            r"^(?:借贷|借贷标志|借方(?:\s*[（(]出账[）)])?|贷方(?:\s*[（(]入账[）)])?|收/支|收入/支出|Dc\s*Flg)$",
            re.I,
        ),
    }
    matches: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        compact = re.sub(r"\s+", "", stripped)
        line_roles = {role for role, pattern in role_patterns.items() if pattern.fullmatch(stripped)}
        if not line_roles:
            if (
                any(marker in compact for marker in ("交易日期", "记账日期"))
                and (
                    any(marker in compact for marker in ("交易金额", "发生额"))
                    or ("借方" in compact and "贷方" in compact)
                )
                and "余额" in compact
            ):
                return offset
        else:
            matches.extend((offset, role) for role in line_roles)
        offset += len(line)
    roles = {role for _, role in matches}
    if {"date", "amount", "balance"}.issubset(roles) and roles.intersection({"sequence", "direction"}):
        return min(position for position, _ in matches)
    return None


def _header_bank_name(full_text: str) -> str | None:
    header = _header_text(full_text)
    for pat in _HEADER_BANK_PATTERNS:
        m = pat.search(header)
        if m:
            name = m.group(1).strip()
            if _looks_like_institution(name):
                return name
    return None


def _filename_bank_token(file_path: str | None) -> str | None:
    if not file_path:
        return None
    stem = Path(file_path).stem
    match = re.search(r"([\u4e00-\u9fa5]{2,20}银行[\u4e00-\u9fa5A-Za-z（）()·\s]*)", stem)
    if match:
        name = match.group(1).strip()
        if _looks_like_institution(name):
            return name
    return None


def resolve_institution_from_context(parse_result: Any, full_text: str) -> tuple[str | None, str]:
    """Resolve institution for StyleContext (IAS v2 stack)."""
    header_bank = _header_bank_name(full_text)
    if header_bank:
        return header_bank, "header.kv"

    entities = getattr(parse_result, "entities", None)
    if entities is not None:
        org = getattr(entities, "organization", None)
        if org and _looks_like_institution(str(org)):
            return str(org), "entities.organization"

    file_path = getattr(parse_result, "file_path", None) if parse_result is not None else None
    filename_bank = _filename_bank_token(str(file_path) if file_path else None)
    if filename_bank:
        return filename_bank, "filename.token"

    registered = detect_registered_institution(full_text)
    if registered:
        return registered, "plugin.institutions"

    if entities is not None:
        domain = getattr(entities, "domain_specific", None) or {}
        inst = domain.get("institution")
        if inst and _looks_like_institution(str(inst)):
            return str(inst), "domain_specific.institution"

    variant = match_institution(full_text, None)
    if variant and _looks_like_institution(variant.display_name) and _header_text(full_text):
        header_only = _header_text(full_text)
        if any(kw in header_only for kw in variant.keywords):
            return variant.display_name, "layout_profile.variant"

    return None, ""


def resolve_institution_hint(
    ctx: StyleContext,
    keyword_map: dict[str, list[str]],
) -> tuple[str | None, str]:
    """Resolve institution hint for style metadata (IAS full stack)."""
    if ctx.institution and _looks_like_institution(ctx.institution):
        return ctx.institution, "entities.organization"

    header = _header_text(ctx.full_text)
    header_bank = _header_bank_name(ctx.full_text)
    if header_bank:
        return header_bank, "header.kv"

    file_path = getattr(ctx.parse_result, "file_path", None) if ctx.parse_result is not None else None
    filename_bank = _filename_bank_token(str(file_path) if file_path else None)
    if filename_bank:
        return filename_bank, "filename.token"

    variant = match_institution(header, None)
    if variant and _looks_like_institution(variant.display_name):
        return variant.display_name, "layout_profile.variant"

    registered = detect_registered_institution(header)
    if registered:
        return registered, "plugin.institutions"

    if header and keyword_map:
        sorted_banks = sorted(
            keyword_map.items(),
            key=lambda kv: max(len(k) for k in kv[1]),
            reverse=True,
        )
        for name, keywords in sorted_banks:
            if any(kw in header for kw in keywords):
                return name, "institution_keywords.header"

    return None, ""


def _looks_like_institution(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if any(token in text for token in _NON_INSTITUTION_TOKENS):
        return False
    return any(marker in text for marker in ("银行", "信用社", "信用合作联社", "农村信用联社"))


def extract_identity_from_header(full_text: str) -> dict[str, str]:
    """Extract account holder / number / period from document header region."""
    header = _header_text(full_text)
    out: dict[str, str] = {}
    lines = [line.strip() for line in header.splitlines() if line.strip()]

    m = re.search(
        r"(?<![账客方])户名[:：]?\s*([\u4e00-\u9fa5A-Za-z（）()·\s]{2,80}?)"
        r"(?:\s{2,}|币种|年份|月份|交易|明细|账号|开户|$|\n)",
        header,
    )
    if m and _looks_like_holder_name(m.group(1)):
        out["account_holder"] = m.group(1).strip()

    if not out.get("account_holder"):
        m = re.search(r"客户姓名[:：]?\s*([\u4e00-\u9fa5A-Za-z（）()·\s]{2,40})(?:\s{2,}|账号|客户账号|$|\n)", header)
        if m and _looks_like_holder_name(m.group(1)):
            out["account_holder"] = m.group(1).strip()

    if not out.get("account_holder"):
        m = re.search(
            r"客户名称(?:\s*Customer\s+Name)?[:：]?\s*([\u4e00-\u9fa5A-Za-z（）()·\s]{2,40}?)"
            r"(?:\s{2,}|账号|卡号|币种|开户机构|开户行|起始日期|终止日期|交易|$|\n)",
            header,
            re.I,
        )
        if m and _looks_like_holder_name(m.group(1)):
            out["account_holder"] = m.group(1).strip()

    m = re.search(
        r"账户名称(?:\s*Account\s+Name)?[:：]?\s*([\u4e00-\u9fa5A-Za-z（）()·\s]{2,60}?)"
        r"(?:\s{2,}|账号|开户行|Bank Name|Account No|借方笔数)",
        header,
        re.I,
    )
    if m and not out.get("account_holder"):
        out["account_holder"] = m.group(1).strip()

    if not out.get("account_holder"):
        m = re.search(r"Account Name\s+([\u4e00-\u9fa5A-Za-z（）()·\s]{2,60}?)(?:\s{2,}|Bank Name)", header, re.I)
        if m:
            out["account_holder"] = m.group(1).strip()

    if not out.get("account_holder") or out.get("account_holder") in {"往来款", "收费", "工资", "转账", "报销"}:
        holder = _previous_line_before_label(lines, "账户名称")
        if holder and not re.fullmatch(r"[0-9*＊\s]{6,}", holder):
            out["account_holder"] = holder
    if not out.get("account_holder"):
        for label in ("户名", "客户名称", "客户姓名"):
            holder = _previous_line_before_label(lines, label)
            if holder and _looks_like_holder_name(holder):
                out["account_holder"] = holder
                break
    if not out.get("account_holder"):
        nearby_holder = _nearby_holder_after_label(header)
        if nearby_holder:
            out["account_holder"] = nearby_holder

    m = re.search(
        r"(?<!贷款)(?<!对方)(?:客户)?(?:账号|账户|账\s*号|账\s*户)(?:\s*Account\s+Number)?"
        r"(?:\s*/\s*卡号)?[:：]?[ \t]*([0-9*＊\\ \t-]{8,40})",
        header,
        re.I,
    )
    if m:
        out["account_number"] = _normalize_account_token(m.group(1))
    else:
        m = re.search(r"Account No\.?[ \t]+([0-9*＊\\ \t]{8,40})", header, re.I)
        if m:
            out["account_number"] = _normalize_account_token(m.group(1))
    if not out.get("account_number"):
        account = _previous_line_before_label(lines, "账号")
        if account and re.fullmatch(r"[0-9*＊\s]{8,30}", account):
            out["account_number"] = _normalize_account_token(account)
    if not out.get("account_number"):
        account = _nearby_account_after_label(lines)
        if account:
            out["account_number"] = account

    start_m = re.search(r"起始日期[:：]?\s*(\d{4}-\d{2}-\d{2}|\d{8})", header)
    end_m = re.search(r"(?:截止日期|终止日期)[:：]?\s*(\d{4}-\d{2}-\d{2}|\d{8})", header)
    if start_m and not end_m:
        end_value = _previous_line_before_label(lines, "终止日期") or _previous_line_before_label(lines, "截止日期")
        if end_value:
            end_m = re.match(r"(\d{4}-\d{2}-\d{2}|\d{8})", end_value)
    if start_m and end_m:
        s, e = start_m.group(1), end_m.group(1)
        if "-" not in s:
            s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        if "-" not in e:
            e = f"{e[:4]}-{e[4:6]}-{e[6:8]}"
        out["query_period"] = f"{s} ~ {e}"
    elif vertical_period := _vertical_query_period(lines):
        out["query_period"] = vertical_period
    elif query_period := _extract_query_period(header):
        out["query_period"] = query_period
    elif period := _extract_year_month_period(header):
        out["query_period"] = period

    if "currency" not in out and "人民币" in header:
        out["currency"] = "CNY"

    bank = _header_bank_name(full_text)
    if bank:
        out["bank_name"] = bank

    return out


def _vertical_query_period(lines: list[str]) -> str:
    """Read a query range when values precede bilingual labels in text flow."""
    start = ""
    end = ""
    for label in ("查询起日", "查询开始日期", "起始日期"):
        if value := _previous_line_before_label(lines, label):
            start = _normalize_date_token(value)
            if start:
                break
    for label in ("查询止日", "查询结束日期", "终止日期", "截止日期"):
        if value := _previous_line_before_label(lines, label):
            end = _normalize_date_token(value)
            if end:
                break
    return f"{start} ~ {end}" if start and end else ""


def _previous_line_before_label(lines: list[str], label: str) -> str:
    """Read vertical/native text headers where value appears above its label."""
    for idx, line in enumerate(lines):
        if label not in line:
            continue
        for prev_idx in range(idx - 1, -1, -1):
            value = lines[prev_idx].strip().strip(":：")
            if not value or label in value:
                continue
            if any(stop in value for stop in ("起始日期", "终止日期", "借方笔数", "贷方笔数", "合计笔数")):
                continue
            return value
    return ""


def _nearby_holder_after_label(header: str) -> str | None:
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if not re.search(r"(户名|账户名称|客户名称|客户姓名|Account Name|Customer Name)", line, re.I):
            continue
        inline = re.sub(
            r"^.*?(?:户名|账户名称|客户名称|客户姓名|Account Name|Customer Name)\s*[:：]?",
            "",
            line,
            flags=re.I,
        ).strip()
        inline = re.split(r"(?:币种|Currency|账号|开户行|交易)", inline, maxsplit=1)[0].strip()
        if _looks_like_holder_name(inline):
            return inline
        for candidate in lines[idx + 1 : idx + 8]:
            if _looks_like_holder_name(candidate):
                return candidate
    return None


def _nearby_account_after_label(lines: list[str]) -> str:
    for idx, line in enumerate(lines):
        if "账号" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 5]:
            text = candidate.strip().strip(":：")
            if any(marker in text for marker in ("序号", "交易日期", "交易金额", "账户余额")):
                break
            if re.fullmatch(r"[0-9*＊\s]{8,40}", text):
                return _normalize_account_token(text)
    return ""


def _normalize_account_token(value: str) -> str:
    text = str(value or "").replace("\\", "").replace("＊", "*")
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_holder_name(value: str) -> bool:
    text = value.strip(" :：|")
    if not text:
        return False
    if any(keyword in text for keyword in _IDENTITY_SKIP_KEYWORDS):
        return False
    if text in {"机构", "柜员", "备注信息", "对方户名", "个人银行", "个人银行结算账户"}:
        return False
    if re.search(r"\d", text):
        return False
    if any(keyword in text for keyword in ("公司", "企业", "集团", "银行", "合作社", "中心", "学校", "医院", "分公司")):
        return bool(re.fullmatch(r"[\u4e00-\u9fa5A-Za-z（）()·\s]{4,80}", text))
    if re.fullmatch(r"[\u4e00-\u9fa5·]{2,8}", text):
        return True
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", text))


def _extract_year_month_period(header: str) -> str:
    m = re.search(r"年份[:：]?\s*(\d{4}).{0,20}?月份[:：]?\s*(\d{1,2})", header, re.S)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
    else:
        lines = [line.strip().strip(":：") for line in header.splitlines() if line.strip()]
        year = _nearby_int_after_label(lines, "年份", min_value=1900, max_value=2100)
        month = _nearby_int_after_label(lines, "月份", min_value=1, max_value=12)
        if not year or not month:
            return ""
    if month < 1 or month > 12:
        return ""
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01 ~ {year:04d}-{month:02d}-{last_day:02d}"


def _extract_query_period(header: str) -> str:
    date_token = r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{8}|\d{4}年\d{1,2}月\d{1,2}日)"
    m = re.search(
        r"(?:查询日期|起止日期|起讫日期|日期范围|交易时间|账单统计日期(?:\s*Start Time\s*&\s*End Time)?)[:：]?\s*"
        rf"({date_token})\s*(?:至|~|—|-)\s*({date_token})",
        header,
        re.I,
    )
    if not m:
        return ""
    start = _normalize_date_token(m.group(1))
    end = _normalize_date_token(m.group(2))
    return f"{start} ~ {end}" if start and end else ""


def _normalize_date_token(value: str) -> str:
    text = str(value or "").replace("/", "-").strip()
    chinese = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if chinese:
        return f"{int(chinese.group(1)):04d}-{int(chinese.group(2)):02d}-{int(chinese.group(3)):02d}"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ""


def _nearby_int_after_label(lines: list[str], label: str, *, min_value: int, max_value: int) -> int:
    for idx, line in enumerate(lines):
        if label not in line:
            continue
        candidates = list(lines[idx + 1 : idx + 12]) + list(reversed(lines[max(0, idx - 6) : idx]))
        for candidate in candidates:
            text = candidate.strip()
            if not re.fullmatch(r"\d{1,4}", text):
                continue
            value = int(text)
            if min_value <= value <= max_value:
                return value
    return 0
