"""Source-backed statement header recovery and transaction context linking.

The transaction table and the statement header are two different source
planes.  This module keeps them separate: it recovers business header facts
from bounded, positioned source text, emits one header record per statement
scope, and links transaction records to the applicable scope without
overwriting row-local facts.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Sequence

from docmirror.plugins._runtime.evidence_access import text_atoms
from docmirror.plugins.bank_statement.header_resolve import normalize_bank_matching_text

_SOURCE_UNITEMIZED_DERIVATION = "source_unitemized_reconciliation"
_SOURCE_UNITEMIZED_PROVENANCE = "derived.bank_statement.source_unitemized"
_MONEY_EPSILON = Decimal("0.005")
_ANCHOR_DEPENDENT_SELECTED_SOURCES = {
    "canonical_evidence_table",
    "ocr_implicit_table",
    "positioned_record_block",
}

_DATE_TOKEN_RE = re.compile(r"(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|20\d{2}年\d{1,2}月\d{1,2}日|20\d{6})")
_MONTH_TOKEN_RE = re.compile(r"(?:20\d{2}[-/.年]\d{1,2}(?:月)?|20\d{4})")
_MONEY_TOKEN_RE = re.compile(r"^[+-]?(?:[¥￥$]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")
_LOCAL_FIRST_PAGE_RES = (
    re.compile(r"第\s*1\s*页\s*[,，/]?\s*(?:共|of)\s*\d+\s*页", re.I),
    re.compile(r"(?:页码|page)\s*[:：]?\s*1\s*[/／]\s*\d+", re.I),
    re.compile(r"(?:页码|page)\s*[:：]?\s*1\s*[-—]\s*1(?:\D|$)", re.I),
    re.compile(r"page\s*1\s*(?:of|/)\s*\d+", re.I),
)
_TITLE_MARKERS = (
    "账户明细",
    "账户交易",
    "交易明细",
    "历史明细",
    "活期账户",
    "银行流水",
    "交易流水",
    "对账单",
    "账单",
    "月结单",
    "statement",
    "account activity",
    "transaction detail",
    "transaction history",
    "交易清单",
    "客户交易清单",
    "account details",
)
_TRANSACTION_HEADER_MARKERS = {
    "序号",
    "交易日期",
    "交易时间",
    "记账日期",
    "金额",
    "交易金额",
    "余额",
    "账户余额",
    "摘要",
    "摘要代码",
    "备注",
    "附言",
    "用途",
    "交易类型",
    "交易地点",
    "交易渠道",
    "交易机构",
    "交易信息",
    "交易对手信息",
    "对手机构",
    "对手名称",
    "对方账号",
    "对方户名",
    "凭证号",
    "流水号",
    "编号",
    "币种",
    "借方发生额",
    "贷方发生额",
    "收/支",
}


def _nfkc(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _compact(value: Any) -> str:
    matching = normalize_bank_matching_text(_nfkc(value))
    return re.sub(r"[\s:：._()（）\[\]【】]+", "", matching).casefold()


def _page_number(page_id: Any) -> int:
    match = re.search(r"(\d+)$", str(page_id or ""))
    return int(match.group(1)) if match else 0


def _normalize_date(value: Any) -> str:
    text = _nfkc(value)
    chinese = re.fullmatch(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
    if chinese:
        parts = tuple(int(chinese.group(index)) for index in range(1, 4))
        try:
            return calendar_date(*parts).isoformat()
        except ValueError:
            return text
    compact = re.sub(r"\D", "", text)
    if len(compact) == 8 and compact.startswith("20"):
        parts = (int(compact[:4]), int(compact[4:6]), int(compact[6:8]))
        try:
            return calendar_date(*parts).isoformat()
        except ValueError:
            return text
    slash = re.fullmatch(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if slash:
        parts = tuple(int(slash.group(index)) for index in range(1, 4))
        try:
            return calendar_date(*parts).isoformat()
        except ValueError:
            return text
    return text


def _valid_date_tokens(value: Any) -> list[str]:
    valid: list[str] = []
    for token in _DATE_TOKEN_RE.findall(_nfkc(value)):
        normalized = _normalize_date(token)
        try:
            calendar_date.fromisoformat(normalized)
        except ValueError:
            continue
        valid.append(normalized)
    return valid


def _period_dates_are_valid(value: Any, *, minimum: int, maximum: int = 2) -> bool:
    dates = _valid_date_tokens(value)
    if not minimum <= len(dates) <= maximum:
        return False
    return len(dates) < 2 or dates[0] <= dates[1]


def _normalize_money(value: Any) -> str:
    text = re.sub(r"[\s,¥￥$]", "", _nfkc(value))
    try:
        number = Decimal(text)
    except InvalidOperation:
        return _nfkc(value)
    return format(number, "f")


def _normalize_currency(value: Any) -> str:
    text = _compact(value).upper()
    aliases = {
        "人民币": "CNY",
        "人民币元": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "美元": "USD",
        "USD": "USD",
        "港币": "HKD",
        "港元": "HKD",
        "HKD": "HKD",
        "欧元": "EUR",
        "EUR": "EUR",
        "日元": "JPY",
        "JPY": "JPY",
    }
    return aliases.get(text, _nfkc(value))


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "statement_title": ("标题", "流水标题", "账单标题", "Statement Title", "Document Title"),
    "document_date": ("单据日期", "报表日期", "Document Date"),
    "print_date": ("打印日期", "制表日期", "生成日期", "Print Date", "Printed Date", "Printing Date"),
    "print_timestamp": ("打印时间", "Print Time", "Printing Time"),
    "application_time": ("申请时间", "申请日期", "Application Time", "Application Date"),
    "query_date": ("查询日期", "Query Date"),
    "issue_date": ("出具日期", "开具日期", "Issue Date"),
    "issue_timestamp": ("开立日期", "Issuing Date"),
    "period_start": (
        "开始日期",
        "起始日期",
        "查询起日",
        "查询开始日期",
        "账期开始",
        "From Date",
        "Start Date",
        "Query Starting Date",
    ),
    "period_end": (
        "结束日期",
        "截止日期",
        "终止日期",
        "查询止日",
        "查询结束日期",
        "账期结束",
        "To Date",
        "End Date",
        "Query Ending Date",
    ),
    "query_period": (
        "查询期间",
        "查询时间段",
        "查询日期范围",
        "交易时段",
        "交易期间",
        "起止日期",
        "起讫日期",
        "时间范围",
        "日期范围",
        "对账周期",
        "Statement Period",
        "Query Period",
        "Start Time & End Time",
    ),
    "statement_period": ("账单统计日期", "账单期间", "Statement Statistics Date"),
    "statement_cutoff_date": ("出单截至日期", "Statement Cutoff Date"),
    "account_holder": (
        "户名",
        "账户名",
        "账户名称",
        "本方户名",
        "本方账号户名",
        "客户名称",
        "客户姓名",
        "客户名",
        "企业名称",
        "Account Name",
        "Customer Name",
        "Account Holder",
        "Card Holder",
    ),
    "account_number": (
        "账号",
        "账户号",
        "银行账号",
        "客户账号",
        "账户代号",
        "卡号/账号",
        "账号/卡号",
        "Account No",
        "Account/Card No",
        "Account Number",
        "Card Number",
        "Customer Account Number",
    ),
    "card_number": ("卡号", "银行卡号", "Card No", "Card Number"),
    "internal_account": ("账户账号", "内部账号", "Internal Account"),
    "customer_number": ("客户号", "客户编号", "Customer Number", "Customer No"),
    "bank_name": ("银行名称", "Bank Name"),
    "branch_name": (
        "开户银行",
        "开户行",
        "开户机构",
        "打印机构",
        "开户网点",
        "营业机构",
        "网点名称",
        "本方账号开户行",
        "Branch",
        "Opening Branch",
        "Sub Branch",
    ),
    "transaction_institution": ("交易机构", "经办机构", "Transaction Institution"),
    "accepting_branch": ("受理行", "Accepting Branch"),
    "currency": ("币种", "账单币种", "币别", "Currency"),
    "amount_unit": ("单位", "金额单位", "币种单位", "Amount Unit"),
    "account_type": ("账户类型", "账户类别", "Account Type"),
    "deposit_type": ("存款种类", "存款类型", "产品名称", "储种", "Deposit Type", "Product Name"),
    "cash_remittance": ("钞汇", "钞汇标识", "钞汇标志", "现转标志", "Cash/Remittance"),
    "statement_number": ("账单号", "结单号", "对账单号", "Statement No", "Statement Number"),
    "statement_code": ("对账单代码", "报表代码", "Statement Code"),
    "statement_type": ("账单类型", "Statement Type"),
    "list_number": ("清单编号", "List Number"),
    "statement_month": ("账单月份", "对账月份", "结单月份", "Statement Month"),
    "statement_year": ("年份", "Year"),
    "statement_month_number": ("月份", "Month"),
    "electronic_serial": ("电子流水号", "电子序列号", "Electronic Serial"),
    "verification_code": ("验证码", "验证编号", "Verification Code"),
    "proof_number": ("证明编号", "编号", "Proof Number"),
    "wechat_id": ("微信号", "WeChat ID"),
    "id_type": ("证件种类", "证件类型", "ID Type"),
    "id_number": ("证件号码", "身份证号", "身份证号码", "ID Number"),
    "filter_condition": ("筛选条件", "流水范围", "查询范围", "Filter", "Scope"),
    "direction_filter": ("借贷方向", "借/贷标记", "交易方向", "Direction"),
    "sort_order": ("排序方向", "排序方式", "Sort Order"),
    "print_channel": ("打印渠道", "Print Channel"),
    "print_teller": ("打印柜员", "柜员号", "Print Teller"),
    "print_count": ("已打印次数", "打印次数", "Print Count"),
    "print_method": ("打印方式", "Print Method"),
    "device_number": ("设备编号", "设备号", "Device Number"),
    "query_teller": ("查询柜员", "Search Teller", "柜员 Search Teller"),
    "query_timestamp": ("查询时间", "Query Time"),
    "department": ("部门", "Department"),
    "customer_branch": ("客户行", "Customer Branch"),
    "account_balance": ("账户余额", "当前余额", "Account Balance"),
    "summary_code": ("摘要代号", "摘要代码", "Summary Code"),
    "amount_upper_limit": ("最高金额", "金额上限", "Maximum Amount"),
    "amount_lower_limit": ("最低金额", "金额下限", "Minimum Amount"),
    "total_transactions": (
        "总笔数",
        "合计笔数",
        "总条数",
        "记录数",
        "交易总笔数",
        "汇总交易笔数",
    ),
    "total_amount": ("总金额", "交易总金额", "汇总交易金额", "Total Amount"),
    "debit_count": (
        "借方总笔数",
        "借方笔数",
        "借方合计笔数",
        "本月累计借方发生数",
        "支出总笔数",
        "支出笔数",
        "支出交易笔数",
    ),
    "debit_total": (
        "借方总金额",
        "借方发生额",
        "借方发生总额",
        "借方合计金额",
        "本月累计借方发生额",
        "汇总借方发生",
        "汇总借方发生额",
        "支出总金额",
        "支出总额",
        "支出金额",
        "支出金额合计",
    ),
    "credit_count": (
        "贷方总笔数",
        "贷方笔数",
        "贷方合计笔数",
        "本月累计贷方发生数",
        "收入总笔数",
        "收入笔数",
        "收入交易笔数",
    ),
    "credit_total": (
        "贷方总金额",
        "贷方发生额",
        "贷方发生总额",
        "贷方合计金额",
        "本月累计贷方发生额",
        "汇总贷方发生",
        "汇总贷方发生额",
        "收入总金额",
        "收入总额",
        "收入金额",
        "收入金额合计",
    ),
    "opening_balance": ("期初余额", "上期余额", "Opening Balance"),
    "closing_balance": ("期末余额", "Closing Balance"),
    "brought_forward_balance": ("承前余额", "承前", "承上余额", "Brought Forward Balance"),
}

_NORMALIZED_ALIAS_TO_FIELD = {
    _compact(alias): field_name for field_name, aliases in _FIELD_ALIASES.items() for alias in aliases
}
_FIELD_ALIAS_PARTS = {
    field_name: {_compact(alias) for alias in aliases if _compact(alias)}
    for field_name, aliases in _FIELD_ALIASES.items()
}
_MAX_ALIAS_ATOMS = 5
_COUNT_FIELDS = {"total_transactions", "debit_count", "credit_count", "print_count"}
_MONEY_FIELDS = {
    "total_amount",
    "debit_total",
    "credit_total",
    "opening_balance",
    "closing_balance",
    "brought_forward_balance",
    "account_balance",
    "amount_upper_limit",
    "amount_lower_limit",
}
_DATE_FIELDS = {
    "print_date",
    "period_start",
    "period_end",
    "statement_cutoff_date",
    "document_date",
    "query_date",
    "issue_date",
}
_DATETIME_FIELDS = {"print_timestamp", "application_time", "query_timestamp", "issue_timestamp"}
_SINGLE_ATOM_VALUE_FIELDS = (
    _COUNT_FIELDS
    | _MONEY_FIELDS
    | _DATE_FIELDS
    | {
        "currency",
        "account_number",
        "card_number",
        "internal_account",
        "customer_number",
        "statement_number",
        "statement_code",
        "electronic_serial",
        "verification_code",
        "proof_number",
        "list_number",
        "wechat_id",
        "id_number",
        "device_number",
    }
)
_CONTEXT_SIGNATURE_FIELDS = (
    "account_number",
    "card_number",
    "statement_number",
    "statement_month",
    "statement_year",
    "statement_month_number",
    "statement_period",
    "period_start",
    "period_end",
    "query_period",
)
_TRANSACTION_CONTEXT_FIELDS = (
    "account_holder",
    "bank_name",
    "currency",
    "statement_title",
    "period_start",
    "period_end",
    "print_date",
    "document_date",
)
_POSITIONED_FOOTER_FIELDS = {
    "print_count",
    "print_date",
    "print_timestamp",
    "print_method",
    "device_number",
    "print_teller",
    "period_end",
    "statement_cutoff_date",
    "total_transactions",
    "total_amount",
    "debit_count",
    "debit_total",
    "credit_count",
    "credit_total",
    "closing_balance",
}
_STABLE_CROSS_SCOPE_FIELDS = {
    "statement_title",
    "bank_name",
    "account_holder",
    "account_number",
    "card_number",
    "internal_account",
    "customer_number",
    "branch_name",
    "currency",
    "amount_unit",
}


@dataclass(frozen=True)
class _HeaderFact:
    field_key: str
    raw_name: str
    raw_value: str
    normalized_value: Any
    page: int
    page_id: str
    bbox: tuple[float, float, float, float] | None
    evidence_ids: tuple[str, ...]
    source_kind: str = "canonical_evidence_atoms"


@dataclass(frozen=True)
class _LabelSpan:
    start: int
    end: int
    field_key: str
    raw_name: str
    inline_value: str = ""
    known: bool = True


def _normalize_field_value(field_key: str, value: Any) -> Any:
    text = _nfkc(value)
    if field_key in _DATE_FIELDS:
        return _normalize_date(text)
    if field_key in _DATETIME_FIELDS:
        match = re.search(
            r"(20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日)?)(?:\s+|T)?(\d{1,2}:\d{2}(?::\d{2})?)?",
            text,
        )
        if match:
            date = _normalize_date(match.group(1))
            time = match.group(2) or ""
            return f"{date} {time}".strip()
        return text
    if field_key == "statement_month":
        separated = re.search(r"(20\d{2})[-/.年](\d{1,2})(?:月)?", text)
        if separated:
            return f"{separated.group(1)}-{int(separated.group(2)):02d}"
        compact_month = re.search(r"(?<!\d)(20\d{4})(?!\d)", re.sub(r"\s+", "", text))
        if compact_month:
            digits = compact_month.group(1)
            return f"{digits[:4]}-{int(digits[4:6]):02d}"
        return text
    if field_key in {"statement_year", "statement_month_number"}:
        digits = re.sub(r"\D", "", text)
        return int(digits) if digits else text
    if field_key == "currency":
        return _normalize_currency(text)
    if field_key in _COUNT_FIELDS:
        digits = re.sub(r"[^0-9]", "", text)
        return int(digits) if digits else text
    if field_key in _MONEY_FIELDS:
        return _normalize_money(text)
    if field_key in {
        "account_number",
        "card_number",
        "internal_account",
        "customer_number",
        "statement_number",
        "statement_code",
        "electronic_serial",
        "verification_code",
        "proof_number",
        "list_number",
        "id_number",
        "device_number",
    }:
        return re.sub(r"\s+", "", text)
    if field_key in {"query_period", "statement_period"}:
        dates = _valid_date_tokens(text)
        if len(dates) >= 2:
            return f"{dates[0]} ~ {dates[1]}"
        if len(dates) == 1:
            return dates[0]
    return text


def _clean_header_value(field_key: str, value: Any) -> str:
    text = _nfkc(value)
    if field_key in {
        "query_period",
        "statement_period",
        "period_start",
        "period_end",
        "statement_cutoff_date",
        "query_date",
    }:
        text = re.sub(
            r"(?:第\s*\d+\s*页\s*[,，/]?\s*(?:共\s*)?\d+\s*页|页码\s*[:：]?\s*\d+\s*[/／]\s*\d+).*$",
            "",
            text,
            flags=re.I,
        ).strip()
    return text


def _fact_value_is_plausible(field_key: str, value: Any) -> bool:
    """Reject ledger headings and neighbouring labels mistaken for values."""
    text = _nfkc(value)
    compact = _compact(text)
    if not text or compact in _NORMALIZED_ALIAS_TO_FIELD:
        return False
    if field_key == "query_period":
        return _period_dates_are_valid(text, minimum=2)
    if field_key in {"statement_period", "query_date"}:
        return _period_dates_are_valid(text, minimum=1)
    if field_key in _DATE_FIELDS:
        return bool(_DATE_TOKEN_RE.fullmatch(text)) and _period_dates_are_valid(text, minimum=1, maximum=1)
    if field_key in _DATETIME_FIELDS:
        return _period_dates_are_valid(text, minimum=1, maximum=1)
    if field_key == "statement_month":
        return bool(_MONTH_TOKEN_RE.search(text))
    if field_key in _COUNT_FIELDS:
        return bool(re.fullmatch(r"\s*\d{1,9}\s*(?:笔|条|次)?\s*", text))
    if field_key in _MONEY_FIELDS:
        return bool(_MONEY_TOKEN_RE.fullmatch(text.replace(" ", "")))
    if field_key == "currency":
        return compact in {
            "人民币",
            "人民币元",
            "rmb",
            "cny",
            "美元",
            "usd",
            "港币",
            "港元",
            "hkd",
            "欧元",
            "eur",
            "日元",
            "jpy",
        }
    if field_key in {
        "account_number",
        "card_number",
        "internal_account",
        "customer_number",
        "statement_number",
        "statement_code",
        "electronic_serial",
        "verification_code",
        "proof_number",
        "list_number",
        "wechat_id",
        "id_number",
        "device_number",
    }:
        return bool(re.fullmatch(r"[A-Za-z0-9*_.\-/]{3,80}", re.sub(r"\s+", "", text)))
    if field_key in {"statement_year", "statement_month_number"}:
        return bool(re.fullmatch(r"\d{1,4}(?:年|月)?", text))
    if field_key == "account_holder":
        return (
            len(text) <= 100
            and not _DATE_TOKEN_RE.search(text)
            and not re.search(
                r"(?:用途|备注|对方|账号|账户余额|打印日期|查询日期)",
                text,
            )
        )
    if field_key == "bank_name":
        return len(text) <= 100 and "银行" in text and not _DATE_TOKEN_RE.search(text)
    return not bool(_is_header_only_value(text))


def _is_header_only_value(value: str) -> bool:
    compact = _compact(value)
    if _field_for_label(value):
        return True
    header_markers = {_compact(marker) for marker in _TRANSACTION_HEADER_MARKERS}
    return compact in header_markers


def _group_atoms(parse_result: Any) -> dict[int, list[dict[str, Any]]]:
    parser_info = getattr(parse_result, "parser_info", None)
    options = getattr(parser_info, "options", None)
    selected_pages = {int(page) for page in ((options or {}).get("selected_source_pages") or []) if str(page).isdigit()}
    rejected_native_pages = {
        int(page) for page in ((options or {}).get("native_text_ocr_fallback_pages") or []) if str(page).isdigit()
    }
    native_kinds = {"pdf_native", "pdf_native_pypdf", "parse_result_text"}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for atom in text_atoms(parse_result):
        page_id = str(atom.get("page_id") or "")
        page = _page_number(page_id)
        bbox = atom.get("bbox")
        text = _nfkc(atom.get("text"))
        if page <= 0 or not text or not isinstance(bbox, list) or len(bbox) < 4:
            continue
        if selected_pages and page not in selected_pages:
            continue
        source_kind = str(atom.get("source_kind") or "")
        if page in rejected_native_pages and source_kind in native_kinds:
            continue
        fingerprint = (
            page,
            _compact(text),
            *(round(float(value), 1) for value in bbox[:4]),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        grouped[page].append(
            {
                **atom,
                "page_id": page_id,
                "text": text,
                "bbox": [float(value) for value in bbox[:4]],
            }
        )
    # When native text was explicitly rejected in favour of OCR, the canonical
    # evidence plane may still retain only the rejected native atoms.  The
    # positioned OCR TextBlocks on PageContent are then the selected source
    # plane and must be considered instead of leaving the business header
    # empty.  This is bounded to pages named by the parser's fallback decision;
    # it never mixes OCR with an accepted native plane.
    for page_content in getattr(parse_result, "pages", None) or ():
        page = int(getattr(page_content, "source_page_number", 0) or getattr(page_content, "page_number", 0) or 0)
        if page <= 0 or page not in rejected_native_pages or (selected_pages and page not in selected_pages):
            continue
        for block_index, block in enumerate(getattr(page_content, "texts", None) or ()):
            text = _nfkc(getattr(block, "content", ""))
            bbox = getattr(block, "bbox", None)
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            fingerprint = (page, _compact(text), *(round(float(value), 1) for value in bbox[:4]))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            evidence_ids = [str(value) for value in (getattr(block, "evidence_ids", None) or ()) if str(value)]
            grouped[page].append(
                {
                    "id": evidence_ids[0] if evidence_ids else f"context:ocr:p{page:04d}:{block_index:06d}",
                    "page_id": f"page:{page:04d}",
                    "text": text,
                    "bbox": [float(value) for value in bbox[:4]],
                    "confidence": float(getattr(block, "confidence", 1.0) or 0.0),
                    "source_kind": "parse_result_ocr_text",
                    "_context_source_kind": "parse_result_ocr_text",
                }
            )
    for atoms in grouped.values():
        atoms.sort(key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
    return dict(grouped)


def _context_source_kind(atoms: Sequence[dict[str, Any]]) -> str:
    if any(str(atom.get("_context_source_kind") or "") == "parse_result_ocr_text" for atom in atoms):
        return "parse_result_ocr_text"
    return "canonical_evidence_atoms"


def _baseline_rows(atoms: Sequence[dict[str, Any]], tolerance: float = 3.5) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for atom in sorted(atoms, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]))):
        baseline = float(atom["bbox"][1])
        target = next(
            (
                row
                for row in reversed(rows[-4:])
                if abs(sum(float(item["bbox"][1]) for item in row) / len(row) - baseline) <= tolerance
            ),
            None,
        )
        if target is None:
            rows.append([atom])
        else:
            target.append(atom)
    for row in rows:
        row.sort(key=lambda item: float(item["bbox"][0]))
    return rows


def _is_transaction_data_row(row: Sequence[dict[str, Any]]) -> bool:
    # A dense header band can contain account-sized digits, a date range, and
    # monetary-looking values on one baseline. Two independently recognized
    # business labels prove that this is a KV header row, not a transaction.
    known_spans = [span for span in _label_spans(row) if span.known and span.field_key]
    if len(known_spans) >= 2:
        return False
    texts = [_nfkc(atom.get("text")) for atom in row]
    has_date = any(_DATE_TOKEN_RE.fullmatch(text) and _valid_date_tokens(text) for text in texts)
    money_count = sum(bool(_MONEY_TOKEN_RE.fullmatch(text.replace(" ", ""))) for text in texts)
    has_sequence = any(re.fullmatch(r"\d{1,7}", text) for text in texts)
    return has_date and money_count >= 1 and (has_sequence or money_count >= 2)


_TRANSACTION_HEADER_ROLE_MARKERS = {
    "temporal": ("交易日期", "交易时间", "记账日期", "会计日期", "起息日", "date", "time"),
    "amount": ("交易金额", "发生额", "收入金额", "支出金额", "借方", "贷方", "amount"),
    "balance": ("余额", "balance"),
    "business": (
        "摘要",
        "备注",
        "附言",
        "用途",
        "对方",
        "对手",
        "交易信息",
        "交易机构",
        "渠道",
        "description",
        "memo",
    ),
    "sequence": ("序号", "流水号", "编号", "凭证号", "sequence", "reference"),
}
_TRANSACTION_HEADER_OVERLAP_FIELDS = {
    "account_number",
    "account_balance",
    "cash_remittance",
    "currency",
    "summary_code",
    "transaction_institution",
}


def _transaction_header_roles(value: Any) -> set[str]:
    compact = _compact(value)
    return {
        role
        for role, markers in _TRANSACTION_HEADER_ROLE_MARKERS.items()
        if any(_compact(marker) in compact for marker in markers)
    }


def _is_transaction_header_row(row: Sequence[dict[str, Any]]) -> bool:
    """Reject a ledger schema row before overlapping aliases become KV facts."""

    texts = [_nfkc(atom.get("text")) for atom in row if _nfkc(atom.get("text"))]
    if len(texts) < 3 or any(re.search(r"[:：]\s*\S|\d{4,}", text) for text in texts):
        return False
    roles = set().union(*(_transaction_header_roles(text) for text in texts))
    if {"temporal", "amount", "balance"} <= roles:
        return True
    header_like = 0
    for text in texts:
        field_key = _field_for_label(text)
        if _transaction_header_roles(text) or field_key in _TRANSACTION_HEADER_OVERLAP_FIELDS:
            header_like += 1
    return (
        header_like == len(texts)
        and len(roles) >= 3
        and "balance" in roles
        and "business" in roles
    )


def _header_rows(atoms: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows = _baseline_rows(atoms)
    first_data_top = next(
        (min(float(atom["bbox"][1]) for atom in row) for row in rows if _is_transaction_data_row(row)),
        None,
    )
    if first_data_top is None:
        if not rows:
            return []
        page_top = min(float(atom["bbox"][1]) for row in rows for atom in row)
        page_bottom = max(float(atom["bbox"][3]) for row in rows for atom in row)
        cutoff = page_top + max(120.0, (page_bottom - page_top) * 0.38)
    else:
        cutoff = first_data_top
    return [row for row in rows if min(float(atom["bbox"][1]) for atom in row) < cutoff]


def _known_label_span(row: Sequence[dict[str, Any]], start: int) -> _LabelSpan | None:
    best: _LabelSpan | None = None
    for end in range(start, min(len(row), start + _MAX_ALIAS_ATOMS)):
        raw = "".join(_nfkc(row[index].get("text")) for index in range(start, end + 1))
        field_key = _field_for_label(raw)
        if field_key:
            best = _LabelSpan(start, end, field_key, raw)
        inline = re.match(r"^(.{1,50}?)[\s]*[:：][\s]*(.+)$", raw)
        if inline:
            field_key = _field_for_label(inline.group(1))
            # A value may share the final atom with a fragmented label
            # (for example ``账`` + ``号:123``).  Do not, however, consume a
            # following label atom as that value: bilingual forms such as
            # ``List Number:  Issuing Date:`` otherwise become the false pair
            # ``List Number = Issuing`` before the second label is complete.
            final_atom_text = _nfkc(row[end].get("text"))
            final_atom_has_inline_value = bool(re.search(r"[:：]\s*\S", final_atom_text))
            if field_key and final_atom_has_inline_value and best is None:
                separator = re.search(r"[:：]", raw)
                raw_name = raw[: separator.end()].strip() if separator else inline.group(1)
                return _LabelSpan(start, end, field_key, raw_name, inline.group(2), True)
    return best


def _field_for_label(value: Any) -> str:
    """Resolve exact aliases, including a Chinese+English bilingual pair."""
    compact = _compact(value)
    if not compact:
        return ""
    if field_key := _NORMALIZED_ALIAS_TO_FIELD.get(compact):
        return field_key
    for field_key, aliases in _FIELD_ALIAS_PARTS.items():
        for left in aliases:
            if compact.startswith(left) and compact[len(left) :] in aliases:
                return field_key
    return ""


def _label_spans(row: Sequence[dict[str, Any]]) -> list[_LabelSpan]:
    spans: list[_LabelSpan] = []
    index = 0
    while index < len(row):
        span = _known_label_span(row, index)
        if span is None:
            raw = _nfkc(row[index].get("text"))
            generic = re.fullmatch(r"([^:：]{1,40})[:：]\s*(.*)", raw)
            if generic and re.search(r"[\u4e00-\u9fffA-Za-z]", generic.group(1)):
                span = _LabelSpan(index, index, "", generic.group(1), generic.group(2), known=False)
        if span is None:
            index += 1
            continue
        # Bilingual headers often print the Chinese and English aliases as two
        # adjacent label atoms.  Treat them as one label, never as each other's
        # value.
        next_span = _known_label_span(row, span.end + 1) if span.end + 1 < len(row) else None
        if next_span is not None and next_span.field_key == span.field_key:
            span = _LabelSpan(
                span.start,
                next_span.end,
                span.field_key,
                " ".join((span.raw_name, next_span.raw_name)),
                span.inline_value or next_span.inline_value,
                True,
            )
        spans.append(span)
        index = span.end + 1
    return spans


def _join_value_atoms(atoms: Sequence[dict[str, Any]]) -> str:
    parts = [_nfkc(atom.get("text")) for atom in atoms if _nfkc(atom.get("text"))]
    if not parts:
        return ""
    if all(not re.search(r"[A-Za-z]", part) for part in parts):
        return "".join(parts)
    return " ".join(parts)


def _bbox_union(atoms: Sequence[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    boxes = [atom.get("bbox") for atom in atoms if isinstance(atom.get("bbox"), list) and len(atom["bbox"]) >= 4]
    if not boxes:
        return None
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _facts_from_row(row: Sequence[dict[str, Any]], page: int) -> list[_HeaderFact]:
    if _is_transaction_header_row(row):
        return []
    spans = _label_spans(row)
    facts: list[_HeaderFact] = []
    for position, span in enumerate(spans):
        value_atoms = list(row[span.end + 1 : spans[position + 1].start if position + 1 < len(spans) else len(row)])
        field_key = span.field_key
        if not span.inline_value and field_key in _DATETIME_FIELDS:
            date_position = next(
                (
                    index
                    for index, atom in enumerate(value_atoms)
                    if _DATE_TOKEN_RE.fullmatch(_nfkc(atom.get("text")))
                    and _valid_date_tokens(_nfkc(atom.get("text")))
                ),
                None,
            )
            if date_position is not None:
                selected = [value_atoms[date_position]]
                if date_position + 1 < len(value_atoms):
                    next_atom = value_atoms[date_position + 1]
                    next_text = _nfkc(next_atom.get("text"))
                    gap = float(next_atom["bbox"][0]) - float(selected[-1]["bbox"][2])
                    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", next_text) and -2.0 <= gap <= 80.0:
                        selected.append(next_atom)
                value_atoms = selected
        if not span.inline_value and field_key in _SINGLE_ATOM_VALUE_FIELDS:
            single_value_atom = next(
                (atom for atom in value_atoms if _fact_value_is_plausible(field_key, _nfkc(atom.get("text")))),
                None,
            )
            if single_value_atom is not None:
                value_atoms = [single_value_atom]
        trailing_value = _join_value_atoms(value_atoms)
        value = span.inline_value or trailing_value
        used_value_atoms = [] if span.inline_value else value_atoms
        if span.inline_value and field_key in {"query_period", "statement_period"}:
            inline_dates = _valid_date_tokens(span.inline_value)
            trailing_period = re.fullmatch(
                rf"\s*(?:-|--|—|~|～|至|到)\s*({_DATE_TOKEN_RE.pattern})\s*",
                trailing_value,
            )
            if len(inline_dates) == 1 and trailing_period and value_atoms:
                gap = float(value_atoms[0]["bbox"][0]) - float(row[span.end]["bbox"][2])
                candidate = f"{span.inline_value} {trailing_value}"
                if -2.0 <= gap <= 180.0 and _period_dates_are_valid(candidate, minimum=2):
                    value = candidate
                    used_value_atoms = value_atoms
        if (
            span.inline_value
            and field_key in _DATETIME_FIELDS
            and re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", trailing_value)
        ):
            value = f"{span.inline_value} {trailing_value}"
            used_value_atoms = value_atoms
        if not field_key and _compact(span.raw_name) in {"交易日期", "交易明细对应时间段"}:
            if len(_DATE_TOKEN_RE.findall(_nfkc(value))) >= 2:
                field_key = "query_period"
        recognized_field = bool(field_key)
        field_key = field_key or f"source_header_{_compact(span.raw_name)}"
        adjacent_month = ""
        if field_key == "statement_year":
            year_month = re.fullmatch(r"\s*(20\d{2})\D*(0?[1-9]|1[0-2])\s*", value)
            if year_month:
                value, adjacent_month = year_month.groups()
        value = _clean_header_value(field_key, value)
        if not value:
            continue
        if len(value) > 240 or _compact(value) in _NORMALIZED_ALIAS_TO_FIELD:
            continue
        if recognized_field and not _fact_value_is_plausible(field_key, value):
            continue
        supporting = [*row[span.start : span.end + 1], *used_value_atoms]
        page_id = str(row[span.start].get("page_id") or f"page:{page:04d}")
        primary_fact = _HeaderFact(
            field_key=field_key,
            raw_name=span.raw_name,
            raw_value=value,
            normalized_value=_normalize_field_value(field_key, value),
            page=page,
            page_id=page_id,
            bbox=_bbox_union(supporting),
            evidence_ids=tuple(
                dict.fromkeys(str(atom.get("id") or "") for atom in supporting if str(atom.get("id") or ""))
            ),
            source_kind=_context_source_kind(supporting),
        )
        facts.append(primary_fact)
        if adjacent_month:
            facts.append(
                _HeaderFact(
                    field_key="statement_month_number",
                    raw_name="unlabelled_month_number",
                    raw_value=adjacent_month,
                    normalized_value=int(adjacent_month),
                    page=page,
                    page_id=page_id,
                    bbox=primary_fact.bbox,
                    evidence_ids=primary_fact.evidence_ids,
                    source_kind=primary_fact.source_kind,
                )
            )
    return facts


def _paired_cumulative_direction_facts(
    row: Sequence[dict[str, Any]],
    page: int,
) -> list[_HeaderFact]:
    """Recover a paired cumulative debit/credit footer whose right label wraps."""
    debit_label: dict[str, Any] | None = None
    credit_label: dict[str, Any] | None = None
    suffix = ""
    for atom in row:
        compact = _compact(atom.get("text"))
        match = re.fullmatch(r"本月累计借方发生(?P<suffix>数|额)", compact)
        if match:
            debit_label, suffix = atom, match.group("suffix")
        elif compact == "本月累计贷方":
            credit_label = atom
    if debit_label is None or credit_label is None:
        return []
    debit_field, credit_field = ("debit_count", "credit_count") if suffix == "数" else ("debit_total", "credit_total")

    def value_atom_after(label: dict[str, Any], field_key: str) -> dict[str, Any] | None:
        candidates = [
            atom
            for atom in row
            if float(atom["bbox"][0]) >= float(label["bbox"][2])
            and _fact_value_is_plausible(field_key, _nfkc(atom.get("text")))
        ]
        return min(candidates, key=lambda atom: float(atom["bbox"][0]), default=None)

    debit_value = value_atom_after(debit_label, debit_field)
    credit_value = value_atom_after(credit_label, credit_field)
    if debit_value is None or credit_value is None or debit_value is credit_value:
        return []
    facts: list[_HeaderFact] = []
    for field_key, label, value in (
        (debit_field, debit_label, debit_value),
        (credit_field, credit_label, credit_value),
    ):
        supporting = [label, value]
        raw_value = _nfkc(value.get("text"))
        facts.append(
            _HeaderFact(
                field_key,
                _nfkc(label.get("text")),
                raw_value,
                _normalize_field_value(field_key, raw_value),
                page,
                str(label.get("page_id") or f"page:{page:04d}"),
                _bbox_union(supporting),
                tuple(str(atom.get("id") or "") for atom in supporting if str(atom.get("id") or "")),
                _context_source_kind(supporting),
            )
        )
    return facts


def _title_fact(header_rows: Sequence[Sequence[dict[str, Any]]], page: int) -> _HeaderFact | None:
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    ledger_header_top = min(
        (
            min(float(atom["bbox"][1]) for atom in row)
            for row in header_rows
            if row and _is_transaction_header_row(row)
        ),
        default=float("inf"),
    )
    for row in header_rows:
        for atom in row:
            text = _nfkc(atom.get("text"))
            compact = _compact(text)
            if float(atom["bbox"][1]) >= ledger_header_top:
                continue
            if not (4 <= len(compact) <= 120) or ":" in text or "：" in text:
                continue
            if _field_for_label(text):
                continue
            if compact in {_compact(marker) for marker in _TRANSACTION_HEADER_MARKERS}:
                continue
            marker_score = sum(marker.casefold() in text.casefold() for marker in _TITLE_MARKERS)
            if marker_score:
                candidates.append((marker_score, -float(atom["bbox"][1]), atom))
    if not candidates:
        return None
    # Document headings precede ledger cells and disclaimers.  Marker richness
    # breaks ties inside the same top band; it must not let a lower transaction
    # value such as ``交易流水`` replace the visible statement title.
    atom = max(candidates, key=lambda item: (item[1], item[0], len(_nfkc(item[2].get("text")))))[2]
    value = _nfkc(atom.get("text"))
    page_id = str(atom.get("page_id") or f"page:{page:04d}")
    return _HeaderFact(
        "statement_title",
        "document_title",
        value,
        value,
        page,
        page_id,
        _bbox_union([atom]),
        (str(atom.get("id") or ""),) if str(atom.get("id") or "") else (),
        _context_source_kind([atom]),
    )


def _bank_name_from_title_fact(title: _HeaderFact) -> _HeaderFact | None:
    """Recover an issuer only when its name is visibly embedded in the title."""
    value = _nfkc(title.raw_value)
    match = re.match(
        r"^(?P<bank>[\u4e00-\u9fff·]{2,22}?银行)(?=.{0,16}(?:账户|交易|明细|对账|流水|账单|客户|存款))",
        value,
    )
    if match is None:
        return None
    bank_name = match.group("bank")
    return _HeaderFact(
        "bank_name",
        "statement_title_issuer",
        bank_name,
        bank_name,
        title.page,
        title.page_id,
        title.bbox,
        title.evidence_ids,
        title.source_kind,
    )


def _bank_name_from_mark_fact(
    atoms: Sequence[dict[str, Any]],
    page: int,
) -> _HeaderFact | None:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for atom in atoms:
        text = _nfkc(atom.get("text"))
        match = re.fullmatch(
            r"(?P<bank>[\u4e00-\u9fff·]{2,22}?银行)\s*[（(]\s*(?:银行)?(?:签章|盖章|公章)\s*[)）]\s*[:：]?",
            text,
        )
        if match:
            candidates.append((match.group("bank"), atom))
    names = {name for name, _atom in candidates}
    if len(names) != 1:
        return None
    bank_name = next(iter(names))
    atoms = [atom for name, atom in candidates if name == bank_name]
    first = atoms[0]
    return _HeaderFact(
        "bank_name",
        "issuer_mark",
        bank_name,
        bank_name,
        page,
        str(first.get("page_id") or f"page:{page:04d}"),
        _bbox_union(atoms),
        tuple(dict.fromkeys(str(atom.get("id") or "") for atom in atoms if str(atom.get("id") or ""))),
        _context_source_kind(atoms),
    )


def _bank_name_from_header_logo_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    title: _HeaderFact | None,
    page: int,
) -> _HeaderFact | None:
    """Bind a plain issuer name only when it is in the statement-title band."""
    if title is None or title.bbox is None:
        return None
    candidates: list[dict[str, Any]] = []
    for row in header_rows:
        for atom in row:
            text = _nfkc(atom.get("text"))
            bbox = atom.get("bbox")
            if not re.fullmatch(r"[一-鿿·]{2,22}银行", text):
                continue
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            if float(bbox[1]) <= title.bbox[3] and float(bbox[3]) >= title.bbox[1] - 45.0:
                candidates.append(atom)
    names = {_nfkc(atom.get("text")) for atom in candidates}
    if len(names) != 1:
        return None
    bank_name = next(iter(names))
    supporting = [atom for atom in candidates if _nfkc(atom.get("text")) == bank_name]
    first = supporting[0]
    return _HeaderFact(
        "bank_name",
        "issuer_title_band",
        bank_name,
        bank_name,
        page,
        str(first.get("page_id") or f"page:{page:04d}"),
        _bbox_union(supporting),
        tuple(dict.fromkeys(str(atom.get("id") or "") for atom in supporting if str(atom.get("id") or ""))),
        _context_source_kind(supporting),
    )


def _unlabelled_document_date_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    facts: Sequence[_HeaderFact],
    page: int,
) -> _HeaderFact | None:
    labelled_dates = {
        _normalize_date(fact.raw_value)
        for fact in facts
        if fact.field_key
        in {
            "print_date",
            "query_date",
            "issue_date",
            "period_start",
            "period_end",
            "query_period",
            "statement_period",
        }
    }
    label_top = min(
        (fact.bbox[1] for fact in facts if fact.bbox and fact.field_key not in {"statement_title", "document_date"}),
        default=float("inf"),
    )
    candidates: list[dict[str, Any]] = []
    for row in header_rows:
        for atom in row:
            text = _nfkc(atom.get("text"))
            if not _DATE_TOKEN_RE.fullmatch(text) or not _valid_date_tokens(text):
                continue
            normalized = _normalize_date(text)
            if normalized in labelled_dates or float(atom["bbox"][1]) >= label_top:
                continue
            candidates.append(atom)
    if len(candidates) != 1:
        return None
    atom = candidates[0]
    value = _nfkc(atom.get("text"))
    return _HeaderFact(
        "document_date",
        "unlabelled_header_date",
        value,
        _normalize_date(value),
        page,
        str(atom.get("page_id") or f"page:{page:04d}"),
        _bbox_union([atom]),
        (str(atom.get("id") or ""),) if str(atom.get("id") or "") else (),
        _context_source_kind([atom]),
    )


def _unlabelled_period_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    facts: Sequence[_HeaderFact],
    page: int,
) -> _HeaderFact | None:
    if any(
        fact.field_key in {"query_period", "statement_period", "query_date", "period_start", "period_end"}
        for fact in facts
    ):
        return None
    candidates: list[tuple[str, Sequence[dict[str, Any]]]] = []
    for row in header_rows:
        if _is_transaction_data_row(row):
            continue
        row_text = _join_value_atoms(row)
        compact_row = _compact(row_text)
        header_marker_count = sum(_compact(marker) in compact_row for marker in _TRANSACTION_HEADER_MARKERS)
        if header_marker_count >= 2:
            continue
        match = re.fullmatch(
            rf"\s*({_DATE_TOKEN_RE.pattern})\s*(?:-|--|—|~|～|至|到)\s*({_DATE_TOKEN_RE.pattern})\s*",
            row_text,
        )
        if match and len(row_text) <= 80 and _period_dates_are_valid(row_text, minimum=2):
            candidates.append((row_text, row))
    title_bottom = max(
        (fact.bbox[3] for fact in facts if fact.field_key == "statement_title" and fact.bbox),
        default=float("-inf"),
    )
    title_band_candidates = [
        candidate
        for candidate in candidates
        if min(float(atom["bbox"][1]) for atom in candidate[1]) <= title_bottom + 20.0
    ]
    if title_band_candidates:
        candidates = title_band_candidates
    unique = {
        (_compact(value), tuple(str(atom.get("id") or "") for atom in row)): (value, row) for value, row in candidates
    }
    if len(unique) != 1:
        return None
    value, row = next(iter(unique.values()))
    return _HeaderFact(
        "query_period",
        "unlabelled_header_period",
        value,
        _normalize_field_value("query_period", value),
        page,
        str(row[0].get("page_id") or f"page:{page:04d}"),
        _bbox_union(row),
        tuple(dict.fromkeys(str(atom.get("id") or "") for atom in row if str(atom.get("id") or ""))),
        _context_source_kind(row),
    )


def _unlabelled_statement_month_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    facts: Sequence[_HeaderFact],
    page: int,
) -> _HeaderFact | None:
    if any(fact.field_key == "statement_month" for fact in facts):
        return None
    candidates: list[dict[str, Any]] = []
    for row in header_rows:
        for atom in row:
            text = _nfkc(atom.get("text"))
            if re.fullmatch(r"20\d{2}年\d{1,2}月", text):
                candidates.append(atom)
    if len(candidates) != 1:
        return None
    atom = candidates[0]
    value = _nfkc(atom.get("text"))
    return _HeaderFact(
        "statement_month",
        "unlabelled_statement_month",
        value,
        _normalize_field_value("statement_month", value),
        page,
        str(atom.get("page_id") or f"page:{page:04d}"),
        _bbox_union([atom]),
        (str(atom.get("id") or ""),) if str(atom.get("id") or "") else (),
        _context_source_kind([atom]),
    )


def _adjacent_unlabelled_month_number_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    facts: Sequence[_HeaderFact],
    page: int,
) -> _HeaderFact | None:
    """Bind an omitted month label only when its value is beside a labelled year."""
    if any(fact.field_key in {"statement_month", "statement_month_number"} for fact in facts):
        return None
    year_facts = [fact for fact in facts if fact.field_key == "statement_year" and fact.bbox]
    if len(year_facts) != 1:
        return None
    year_fact = year_facts[0]
    candidates: list[dict[str, Any]] = []
    for row in header_rows:
        for atom in row:
            text = _nfkc(atom.get("text"))
            bbox = atom.get("bbox")
            if not re.fullmatch(r"0?[1-9]|1[0-2]", text) or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            vertical_overlap = min(float(bbox[3]), year_fact.bbox[3]) - max(float(bbox[1]), year_fact.bbox[1])
            horizontal_gap = float(bbox[0]) - year_fact.bbox[2]
            if vertical_overlap >= -2.0 and 0.0 <= horizontal_gap <= 140.0:
                candidates.append(atom)
    if len(candidates) != 1:
        return None
    atom = candidates[0]
    value = _nfkc(atom.get("text"))
    return _HeaderFact(
        "statement_month_number",
        "unlabelled_month_number",
        value,
        int(value),
        page,
        str(atom.get("page_id") or f"page:{page:04d}"),
        _bbox_union([atom]),
        (str(atom.get("id") or ""),) if str(atom.get("id") or "") else (),
        _context_source_kind([atom]),
    )


def _unlabelled_holder_next_to_account_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    facts: Sequence[_HeaderFact],
    page: int,
) -> _HeaderFact | None:
    if any(fact.field_key == "account_holder" for fact in facts):
        return None
    accounts = [fact for fact in facts if fact.field_key == "account_number" and fact.bbox]
    if len(accounts) != 1:
        return None
    account = accounts[0]
    counterparty_owned_values = {
        _nfkc(fact.raw_value)
        for fact in facts
        if re.search(r"(?:对方|对手|counterparty)", _compact(fact.raw_name), re.I)
    }
    candidates: list[dict[str, Any]] = []
    for row in header_rows:
        for atom in row:
            text = _nfkc(atom.get("text"))
            bbox = atom.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4 or _field_for_label(text):
                continue
            if text in counterparty_owned_values:
                continue
            if not re.search(r"(?:公司|企业|中心|事务所|合作社|经营部|委员会|学校|医院|政府|局|厂|店|部)$", text):
                continue
            vertical_overlap = min(float(bbox[3]), account.bbox[3]) - max(float(bbox[1]), account.bbox[1])
            horizontal_gap = float(bbox[0]) - account.bbox[2]
            if vertical_overlap >= -2.0 and 0.0 <= horizontal_gap <= 360.0:
                candidates.append(atom)
    if len(candidates) != 1:
        return None
    atom = candidates[0]
    value = _nfkc(atom.get("text"))
    if not _fact_value_is_plausible("account_holder", value):
        return None
    return _HeaderFact(
        "account_holder",
        "unlabelled_account_holder",
        value,
        value,
        page,
        str(atom.get("page_id") or f"page:{page:04d}"),
        _bbox_union([atom]),
        (str(atom.get("id") or ""),) if str(atom.get("id") or "") else (),
        _context_source_kind([atom]),
    )


def _page_header_facts(parse_result: Any) -> tuple[dict[int, list[_HeaderFact]], dict[int, list[str]]]:
    facts_by_page: dict[int, list[_HeaderFact]] = {}
    lines_by_page: dict[int, list[str]] = {}
    for page, atoms in sorted(_group_atoms(parse_result).items()):
        rows = _header_rows(atoms)
        lines = [" ".join(_nfkc(atom.get("text")) for atom in row if _nfkc(atom.get("text"))) for row in rows]
        lines_by_page[page] = [line for line in lines if line]
        facts = [fact for row in rows for fact in _facts_from_row(row, page)]
        header_bottom = max(
            (float(atom["bbox"][3]) for row in rows for atom in row),
            default=float("-inf"),
        )
        # Some statements place source business facts (printing metadata and
        # terminal cumulative totals) below the ledger.  Scan that positioned
        # footer plane, but admit only explicit business roles; arbitrary
        # transaction prose and unknown ``x:y`` cells remain out of scope.
        footer_rows = [
            row
            for row in _baseline_rows(atoms)
            if min(float(atom["bbox"][1]) for atom in row) > header_bottom and not _is_transaction_data_row(row)
        ]
        facts.extend(fact for row in footer_rows for fact in _paired_cumulative_direction_facts(row, page))
        facts.extend(
            fact
            for row in footer_rows
            for fact in _facts_from_row(row, page)
            if fact.field_key in _POSITIONED_FOOTER_FIELDS
        )
        title = _title_fact(rows, page)
        if title:
            facts.append(title)
            if bank_name := _bank_name_from_title_fact(title):
                facts.append(bank_name)
        if logo_bank := _bank_name_from_header_logo_fact(rows, title, page):
            facts.append(logo_bank)
        if bank_mark := _bank_name_from_mark_fact(atoms, page):
            facts.append(bank_mark)
        if period := _unlabelled_period_fact(rows, facts, page):
            facts.append(period)
        if statement_month := _unlabelled_statement_month_fact(rows, facts, page):
            facts.append(statement_month)
        if month_number := _adjacent_unlabelled_month_number_fact(rows, facts, page):
            facts.append(month_number)
        if holder := _unlabelled_holder_next_to_account_fact(rows, facts, page):
            facts.append(holder)
        if document_date := _unlabelled_document_date_fact(rows, facts, page):
            facts.append(document_date)
        unique: dict[tuple[str, str, str], _HeaderFact] = {}
        for fact in facts:
            unique.setdefault((fact.field_key, _nfkc(fact.raw_name), _nfkc(fact.raw_value)), fact)
        facts_by_page[page] = list(unique.values())
    return facts_by_page, lines_by_page


def page_texts_with_business_headers(
    parse_result: Any,
    page_texts: Sequence[tuple[int, str]] | None = None,
) -> list[tuple[int, str]]:
    """Prepend geometry-bounded header lines to ordinary page text scopes."""
    facts_by_page, lines_by_page = _page_header_facts(parse_result)
    base = {int(page): str(text or "") for page, text in (page_texts or ())}
    pages = sorted(set(base) | set(lines_by_page))
    return [
        (
            page,
            "\n".join(
                part
                for part in (
                    "\n".join(
                        f"{fact.raw_name.rstrip(':：')}：{fact.raw_value}"
                        for fact in facts_by_page.get(page, [])
                        if fact.field_key != "statement_title"
                    ),
                    "\n".join(lines_by_page.get(page, [])),
                    base.get(page, ""),
                )
                if part.strip()
            ),
        )
        for page in pages
    ]


def _source_page_texts(parse_result: Any) -> dict[int, str]:
    try:
        from docmirror.plugins.bank_statement.wide_table_recovery import page_texts_from_parse_result

        return {int(page): str(text or "") for page, text in page_texts_from_parse_result(parse_result)}
    except Exception:
        return {}


def _fact_values(facts: Iterable[_HeaderFact], key: str) -> set[str]:
    return {_nfkc(fact.normalized_value) for fact in facts if fact.field_key == key and _nfkc(fact.normalized_value)}


def _facts_conflict(left: Sequence[_HeaderFact], right: Sequence[_HeaderFact]) -> bool:
    for key in _CONTEXT_SIGNATURE_FIELDS:
        left_values, right_values = _fact_values(left, key), _fact_values(right, key)
        if left_values and right_values and left_values != right_values:
            return True
    return False


def _is_local_first_page(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in _LOCAL_FIRST_PAGE_RES)


def _context_page_groups(parse_result: Any, facts_by_page: dict[int, list[_HeaderFact]]) -> list[list[int]]:
    source_texts = _source_page_texts(parse_result)
    parsed_pages = {
        int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        for page in (getattr(parse_result, "pages", None) or [])
    }
    pages = sorted((set(facts_by_page) | set(source_texts) | parsed_pages) - {0})
    if not pages:
        return [[1]]
    groups: list[list[int]] = []
    current: list[int] = []
    current_explicit_facts: list[_HeaderFact] = []
    for page in pages:
        page_facts = facts_by_page.get(page, [])
        boundary = bool(current) and (
            _is_local_first_page(source_texts.get(page, "")) or _facts_conflict(current_explicit_facts, page_facts)
        )
        if boundary:
            groups.append(current)
            current = []
            current_explicit_facts = []
        current.append(page)
        if page_facts:
            current_explicit_facts.extend(page_facts)
    if current:
        groups.append(current)
    return groups


def _identity_facts(identity_fields: dict[str, dict[str, Any]], page: int = 1) -> list[_HeaderFact]:
    facts: list[_HeaderFact] = []
    for field_key, detail in identity_fields.items():
        if field_key not in _FIELD_ALIASES and field_key not in {"statement_title", "document_date"}:
            continue
        mapping = detail if isinstance(detail, dict) else {"normalized_value": detail}
        raw_value = next(
            (
                mapping.get(candidate)
                for candidate in ("raw_value", "value", "normalized_value")
                if mapping.get(candidate) not in (None, "")
            ),
            "",
        )
        if raw_value in (None, ""):
            continue
        raw_name = str(mapping.get("raw_name") or field_key)
        if not _identity_detail_is_source_bound(field_key, raw_name, raw_value, mapping):
            continue
        refs = [ref for ref in (mapping.get("source_refs") or []) if isinstance(ref, dict)]
        source_page = next(
            (
                int(ref.get("source_page") or ref.get("page") or _page_number(ref.get("page_id")) or 0)
                for ref in refs
                if int(ref.get("source_page") or ref.get("page") or _page_number(ref.get("page_id")) or 0) > 0
            ),
            page,
        )
        facts.append(
            _HeaderFact(
                field_key,
                raw_name,
                str(raw_value),
                _normalize_field_value(field_key, raw_value),
                source_page,
                f"page:{source_page:04d}",
                None,
                tuple(str(item) for item in (mapping.get("evidence_ids") or []) if str(item)),
                str(mapping.get("source") or "identity_fields"),
            )
        )
    return facts


def _identity_detail_is_source_bound(
    field_key: str,
    raw_name: str,
    raw_value: Any,
    mapping: dict[str, Any],
) -> bool:
    """Admit only explicit source labels, never layout/institution guesses."""
    source = str(mapping.get("source") or "").casefold()
    if field_key == "total_transactions" and source.startswith("row_count_evidence."):
        return _fact_value_is_plausible(field_key, raw_value)
    direct_sources = {
        "canonical_evidence_atoms",
        "parse_result_ocr_text",
        "page_headers",
        "header.kv",
    }
    refs = [ref for ref in (mapping.get("source_refs") or []) if isinstance(ref, dict)]
    direct_ref = any(
        str(ref.get("source") or "").casefold() in direct_sources
        and (ref.get("bbox") or mapping.get("evidence_ids"))
        for ref in refs
    )
    direct_source = source in direct_sources or source.startswith("statement_header") or direct_ref
    if not direct_source or _compact(raw_name) == _compact(field_key):
        return False
    if _field_for_label(raw_name) != field_key:
        return False
    return _fact_value_is_plausible(field_key, raw_value)


def _field_source(facts: Sequence[_HeaderFact]) -> dict[str, Any]:
    refs = []
    evidence_ids: list[str] = []
    for fact in facts:
        ref: dict[str, Any] = {"source": fact.source_kind, "source_page": fact.page}
        if fact.bbox:
            ref["bbox"] = list(fact.bbox)
        refs.append(ref)
        evidence_ids.extend(fact.evidence_ids)
    first = facts[0]
    return {
        "raw_name": first.raw_name,
        "source": first.source_kind,
        "source_refs": list({repr(ref): ref for ref in refs}.values()),
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
    }


def _raw_header_map(facts: Sequence[_HeaderFact]) -> dict[str, Any]:
    grouped: dict[str, list[_HeaderFact]] = defaultdict(list)
    for fact in facts:
        grouped[fact.raw_name].append(fact)
    out: dict[str, Any] = {}
    for raw_name, raw_facts in grouped.items():
        values = list(dict.fromkeys(_nfkc(fact.raw_value) for fact in raw_facts if _nfkc(fact.raw_value)))
        if len(values) == 1:
            out[raw_name] = values[0]
        elif values:
            out[raw_name] = [{"page": fact.page, "value": _nfkc(fact.raw_value)} for fact in raw_facts]
    return out


def _record_from_facts(facts: Sequence[_HeaderFact], pages: Sequence[int], index: int) -> dict[str, Any]:
    grouped: dict[str, list[_HeaderFact]] = defaultdict(list)
    for fact in facts:
        grouped[fact.field_key].append(fact)
    normalized: dict[str, Any] = {}
    canonical_raw: dict[str, Any] = {}
    field_sources: dict[str, Any] = {}
    for field_key, field_facts in grouped.items():
        if field_key.startswith("source_header_"):
            continue
        selected_facts = list(field_facts)
        if field_key == "brought_forward_balance":
            first_page = min(fact.page for fact in field_facts)
            selected_facts = [fact for fact in field_facts if fact.page == first_page]
        elif field_key == "statement_cutoff_date":
            last_page = max(fact.page for fact in field_facts)
            selected_facts = [fact for fact in field_facts if fact.page == last_page]
        elif field_key in {"debit_count", "debit_total", "credit_count", "credit_total"} and all(
            "累计" in _nfkc(fact.raw_name) for fact in field_facts
        ):
            last_page = max(fact.page for fact in field_facts)
            selected_facts = [fact for fact in field_facts if fact.page == last_page]
        normalized_values = list(
            dict.fromkeys(
                _nfkc(fact.normalized_value) if not isinstance(fact.normalized_value, int) else fact.normalized_value
                for fact in selected_facts
                if fact.normalized_value not in (None, "")
            )
        )
        raw_values = list(dict.fromkeys(_nfkc(fact.raw_value) for fact in selected_facts if _nfkc(fact.raw_value)))
        if len(normalized_values) != 1 or not raw_values:
            continue
        normalized[field_key] = normalized_values[0]
        canonical_raw[field_key] = raw_values[0] if len(raw_values) == 1 else raw_values
        field_sources[field_key] = _field_source(selected_facts)
    if (
        "statement_month" not in normalized
        and {
            "statement_year",
            "statement_month_number",
        }
        <= normalized.keys()
    ):
        year = int(normalized["statement_year"])
        month = int(normalized["statement_month_number"])
        if 1 <= month <= 12:
            normalized["statement_month"] = f"{year:04d}-{month:02d}"
            canonical_raw["statement_month"] = (
                f"{canonical_raw['statement_year']} {canonical_raw['statement_month_number']}"
            )
            supporting = [fact for key in ("statement_year", "statement_month_number") for fact in grouped.get(key, [])]
            if supporting:
                field_sources["statement_month"] = _field_source(supporting)
    period_source_key = next(
        (
            key
            for key in ("query_period", "statement_period", "query_date")
            if len(_DATE_TOKEN_RE.findall(str(normalized.get(key) or ""))) >= 2
        ),
        "",
    )
    if period_source_key:
        dates = [_normalize_date(value) for value in _DATE_TOKEN_RE.findall(str(normalized[period_source_key]))]
        raw_period_dates = _DATE_TOKEN_RE.findall(str(canonical_raw.get(period_source_key) or ""))
        if len(dates) >= 2:
            for position, (key, value) in enumerate((("period_start", dates[0]), ("period_end", dates[1]))):
                normalized.setdefault(key, value)
                if position < len(raw_period_dates):
                    canonical_raw.setdefault(key, raw_period_dates[position])
                field_sources.setdefault(key, dict(field_sources.get(period_source_key) or {}))
    if "period_start" in normalized and "period_end" in normalized:
        normalized.setdefault("query_period", f"{normalized['period_start']} ~ {normalized['period_end']}")
    record_id = f"statement_header:r{index:06d}"
    page_range = [min(pages), max(pages)] if pages else [1, 1]
    refs = [{"source": "statement_header_scope", "source_page": page, "page_range": [page, page]} for page in pages]
    return {
        "record_id": record_id,
        "normalized": normalized,
        "canonical_raw": canonical_raw,
        "raw": _raw_header_map(facts),
        "source": {
            "source": "statement_header_scope",
            "page_range": page_range,
            "source_refs": refs,
            "field_sources": field_sources,
        },
        "confidence": 1.0 if any(fact.source_kind == "canonical_evidence_atoms" for fact in facts) else 0.85,
    }


def build_statement_header_records(
    parse_result: Any,
    identity_fields: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return one source-preserving header row per resolved statement scope."""
    facts_by_page, _lines = _page_header_facts(parse_result)
    groups = _context_page_groups(parse_result, facts_by_page)
    global_identity = _identity_facts(dict(identity_fields or {}))
    stable_cross_scope = _stable_cross_scope_facts(facts_by_page, groups)
    records: list[dict[str, Any]] = []
    for index, pages in enumerate(groups, start=1):
        facts = [fact for page in pages for fact in facts_by_page.get(page, [])]
        # Stable identity fields apply to every statement scope.  Period and
        # declared-count fields are document-global only when there is exactly
        # one scope; otherwise a segment must carry its own source fact.
        for fact in global_identity:
            if len(groups) > 1 and fact.page not in pages:
                continue
            if not _fact_values(facts, fact.field_key):
                facts.append(fact)
        for field_key, stable_facts in stable_cross_scope.items():
            if not _fact_values(facts, field_key):
                facts.extend(stable_facts)
        if not facts:
            continue
        records.append(_record_from_facts(facts, pages, index))
    if not records and global_identity:
        page_count = max(1, len(getattr(parse_result, "pages", None) or []))
        records.append(_record_from_facts(global_identity, list(range(1, page_count + 1)), 1))
    return records


def _stable_cross_scope_facts(
    facts_by_page: dict[int, list[_HeaderFact]],
    groups: Sequence[Sequence[int]],
) -> dict[str, list[_HeaderFact]]:
    """Carry a fact only after two distinct scopes independently agree."""
    if len(groups) < 2:
        return {}
    grouped_facts: dict[str, list[_HeaderFact]] = defaultdict(list)
    supporting_groups: dict[str, set[int]] = defaultdict(set)
    for group_index, pages in enumerate(groups):
        for page in pages:
            for fact in facts_by_page.get(page, []):
                if fact.field_key not in _STABLE_CROSS_SCOPE_FIELDS:
                    continue
                grouped_facts[fact.field_key].append(fact)
                supporting_groups[fact.field_key].add(group_index)
    stable: dict[str, list[_HeaderFact]] = {}
    for field_key, facts in grouped_facts.items():
        if len(supporting_groups[field_key]) < 2:
            continue
        if len(_fact_values(facts, field_key)) == 1:
            stable[field_key] = facts
    return stable


def _record_source_page(record: dict[str, Any]) -> int:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    for candidate in (
        source.get("source_page"),
        source.get("page"),
        (source.get("page_range") or [None])[0],
        record.get("source_page"),
    ):
        try:
            page = int(candidate or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
    return 0


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _as_exact_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _money_values_match(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) < _MONEY_EPSILON


def _normalized_money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):.2f}"


def _row_local_source_page(record: dict[str, Any]) -> int:
    """Return a row-local page only; broad statement ranges are not row evidence."""
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    for candidate in (source.get("source_page"), source.get("page"), record.get("source_page")):
        try:
            page = int(candidate or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            return page
    page_id = _page_number(source.get("page_id"))
    if page_id > 0:
        return page_id
    page_range = source.get("page_range")
    if isinstance(page_range, (list, tuple)) and len(page_range) >= 2:
        try:
            start, end = int(page_range[0]), int(page_range[1])
        except (TypeError, ValueError):
            return 0
        if start > 0 and start == end:
            return start
    return 0


def _cached_independent_row_anchor_evidence(
    parse_result: Any,
    *,
    source_route: str | None,
) -> dict[str, Any]:
    """Read the already-materialized positioned-row census without rerunning extraction."""
    try:
        from docmirror.plugins.bank_statement.evidence_atom_table_recovery import (
            recovered_evidence_atom_expected_row_evidence,
            recovered_evidence_atom_row_sources,
        )

        row_sources = recovered_evidence_atom_row_sources(parse_result, source_route=source_route)
        expected, source, confidence = recovered_evidence_atom_expected_row_evidence(
            parse_result,
            source_route=source_route,
        )
    except (AttributeError, TypeError, ValueError):
        return {}
    pages = [_row_local_source_page({"source": item}) for item in row_sources if isinstance(item, dict)]
    if (
        not row_sources
        or len(pages) != len(row_sources)
        or any(page <= 0 for page in pages)
        or int(expected or 0) != len(row_sources)
        or not str(source or "").strip()
        or float(confidence or 0.0) < 0.80
    ):
        return {}
    return {
        "expected_rows": int(expected),
        "source": str(source),
        "confidence": float(confidence),
        "row_sources": row_sources,
        "pages": pages,
    }


def _header_scope(header: dict[str, Any]) -> tuple[int, int] | None:
    source = header.get("source") if isinstance(header.get("source"), dict) else {}
    page_range = source.get("page_range")
    if not isinstance(page_range, (list, tuple)) or len(page_range) < 2:
        return None
    try:
        start, end = int(page_range[0]), int(page_range[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if 0 < start <= end else None


def _scope_rows(
    records: Sequence[dict[str, Any]],
    header: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]] | None:
    header_id = str(header.get("record_id") or "")
    scope = _header_scope(header)
    if not header_id or scope is None:
        return None
    scoped = [
        record
        for record in records
        if str((record.get("normalized") or {}).get("statement_header_id") or "") == header_id
    ]
    if not scoped:
        return None
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in scoped:
        page = _row_local_source_page(record)
        if page <= 0 or not scope[0] <= page <= scope[1]:
            return None
        by_page[page].append(record)
    return scoped, dict(by_page)


def _visible_direction_totals(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
    totals: dict[str, dict[str, Any]] = {
        "expense": {"count": 0, "amount": Decimal("0")},
        "income": {"count": 0, "amount": Decimal("0")},
    }
    for record in rows:
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        direction = str(normalized.get("direction") or "")
        amount = _as_decimal(normalized.get("amount"))
        if direction not in totals or amount is None or amount < 0:
            return None
        totals[direction]["count"] += 1
        totals[direction]["amount"] += amount
    return totals


def _carry_fact_value(facts: Sequence[_HeaderFact]) -> tuple[Decimal, list[_HeaderFact]] | None:
    carry_facts = [fact for fact in facts if fact.field_key == "brought_forward_balance"]
    values = {_as_decimal(fact.normalized_value) for fact in carry_facts}
    values.discard(None)
    if len(values) != 1:
        return None
    value = next(iter(values))
    return value, [fact for fact in carry_facts if _as_decimal(fact.normalized_value) == value]


def _previous_row_source(record: dict[str, Any], page: int) -> dict[str, Any]:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    ref: dict[str, Any] = {
        "source": str(source.get("source") or "transaction_row"),
        "source_page": page,
    }
    for key in ("bbox", "evidence_ids", "table_id", "source_table_id", "source_row_index"):
        if source.get(key) not in (None, "", [], {}):
            ref[key] = deepcopy(source[key])
    return ref


def _carry_boundaries(
    facts_by_page: dict[int, list[_HeaderFact]],
    rows_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    pages = sorted(rows_by_page)
    if len(pages) < 2:
        return None
    boundaries: list[dict[str, Any]] = []
    for previous_page, next_page in zip(pages, pages[1:]):
        if next_page != previous_page + 1:
            return None
        previous_record = rows_by_page[previous_page][-1]
        previous_normalized = (
            previous_record.get("normalized") if isinstance(previous_record.get("normalized"), dict) else {}
        )
        previous_balance = _as_decimal(previous_normalized.get("balance"))
        carry = _carry_fact_value(facts_by_page.get(next_page, []))
        if previous_balance is None or carry is None:
            return None
        brought_forward, carry_facts = carry
        signed_gap = brought_forward - previous_balance
        if abs(signed_gap) < _MONEY_EPSILON:
            continue
        direction = "income" if signed_gap > 0 else "expense"
        amount = abs(signed_gap)
        boundaries.append(
            {
                "from_source_page": previous_page,
                "to_source_page": next_page,
                "last_visible_balance": _normalized_money(previous_balance),
                "brought_forward_balance": _normalized_money(brought_forward),
                "direction": direction,
                "amount": _normalized_money(amount),
                "previous_row_source": _previous_row_source(previous_record, previous_page),
                "brought_forward_source": _field_source(carry_facts),
            }
        )
    return boundaries


def _source_ref_pages(detail: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    for ref in detail.get("source_refs") or []:
        if not isinstance(ref, dict):
            continue
        try:
            page = int(ref.get("source_page") or ref.get("page") or _page_number(ref.get("page_id")) or 0)
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            pages.add(page)
    return pages


def _terminal_aggregate_sources(
    header: dict[str, Any],
    *,
    count_field: str,
    total_field: str,
    terminal_page: int,
) -> dict[str, Any] | None:
    canonical_raw = header.get("canonical_raw") if isinstance(header.get("canonical_raw"), dict) else {}
    source = header.get("source") if isinstance(header.get("source"), dict) else {}
    field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
    count_source = field_sources.get(count_field)
    total_source = field_sources.get(total_field)
    if (
        canonical_raw.get(count_field) in (None, "")
        or canonical_raw.get(total_field) in (None, "")
        or not isinstance(count_source, dict)
        or not isinstance(total_source, dict)
        or terminal_page not in _source_ref_pages(count_source)
        or terminal_page not in _source_ref_pages(total_source)
    ):
        return None
    return {count_field: deepcopy(count_source), total_field: deepcopy(total_source)}


def _direction_residuals(
    header: dict[str, Any],
    visible: dict[str, dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
    *,
    terminal_page: int,
) -> dict[str, dict[str, Any]] | None:
    normalized = header.get("normalized") if isinstance(header.get("normalized"), dict) else {}
    field_pairs = {
        "expense": ("debit_count", "debit_total"),
        "income": ("credit_count", "credit_total"),
    }
    result: dict[str, dict[str, Any]] = {}
    for direction, (count_field, total_field) in field_pairs.items():
        declared_count = _as_exact_int(normalized.get(count_field))
        declared_total = _as_decimal(normalized.get(total_field))
        direction_gaps = [boundary for boundary in boundaries if boundary["direction"] == direction]
        gap_total = sum((_as_decimal(boundary["amount"]) or Decimal("0") for boundary in direction_gaps), Decimal("0"))
        if declared_count is None and declared_total is None and not direction_gaps:
            continue
        if declared_count is None or declared_total is None:
            return None
        count_residual = declared_count - int(visible[direction]["count"])
        amount_residual = declared_total - Decimal(visible[direction]["amount"])
        if count_residual < 0 or amount_residual < -_MONEY_EPSILON:
            return None
        if abs(amount_residual) < _MONEY_EPSILON:
            amount_residual = Decimal("0")
        if count_residual == 0 and amount_residual == 0 and not direction_gaps:
            continue
        if count_residual <= 0 or amount_residual <= 0 or not direction_gaps:
            return None
        if not _money_values_match(amount_residual, gap_total):
            return None
        terminal_sources = _terminal_aggregate_sources(
            header,
            count_field=count_field,
            total_field=total_field,
            terminal_page=terminal_page,
        )
        if terminal_sources is None:
            return None
        result[direction] = {
            "count": count_residual,
            "amount": amount_residual,
            "declared_count": declared_count,
            "declared_amount": declared_total,
            "visible_count": int(visible[direction]["count"]),
            "visible_amount": Decimal(visible[direction]["amount"]),
            "count_field": count_field,
            "total_field": total_field,
            "terminal_aggregate_sources": terminal_sources,
            "carry_boundaries": direction_gaps,
        }
    return result or None


def _scope_anchor_evidence(
    anchor_evidence: dict[str, Any],
    rows_by_page: dict[int, list[dict[str, Any]]],
    scope: tuple[int, int],
) -> dict[str, Any] | None:
    anchor_counts = Counter(
        page for page in anchor_evidence.get("pages") or [] if scope[0] <= int(page) <= scope[1]
    )
    visible_counts = Counter({page: len(rows) for page, rows in rows_by_page.items()})
    if anchor_counts != visible_counts:
        return None
    return {
        "source": anchor_evidence["source"],
        "confidence": anchor_evidence["confidence"],
        "document_expected_rows": anchor_evidence["expected_rows"],
        "scope_anchored_rows": sum(anchor_counts.values()),
        "scope_visible_rows": sum(visible_counts.values()),
        "page_counts": {str(page): count for page, count in sorted(anchor_counts.items())},
    }


def _derived_source_refs(residual: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for field_name, detail in residual["terminal_aggregate_sources"].items():
        for source_ref in detail.get("source_refs") or []:
            if isinstance(source_ref, dict):
                refs.append({**deepcopy(source_ref), "role": f"terminal_aggregate.{field_name}"})
    for boundary in residual["carry_boundaries"]:
        refs.append({**deepcopy(boundary["previous_row_source"]), "role": "carry_boundary.previous_balance"})
        for source_ref in boundary["brought_forward_source"].get("source_refs") or []:
            if isinstance(source_ref, dict):
                refs.append({**deepcopy(source_ref), "role": "carry_boundary.brought_forward_balance"})
    return list({repr(ref): ref for ref in refs}.values())


def _residual_field_source(
    residual: dict[str, Any],
    *,
    field_name: str,
    value: Any,
    anchor_evidence: dict[str, Any],
    source_route: str | None,
    selected_source: str,
) -> dict[str, Any]:
    return {
        "source": _SOURCE_UNITEMIZED_PROVENANCE,
        "derivation": _SOURCE_UNITEMIZED_DERIVATION,
        "field": field_name,
        "value": value,
        "declared_count": residual["declared_count"],
        "declared_amount": _normalized_money(residual["declared_amount"]),
        "visible_count": residual["visible_count"],
        "visible_amount": _normalized_money(residual["visible_amount"]),
        "residual_count": residual["count"],
        "residual_amount": _normalized_money(residual["amount"]),
        "terminal_aggregate_sources": deepcopy(residual["terminal_aggregate_sources"]),
        "carry_boundaries": deepcopy(residual["carry_boundaries"]),
        "independent_row_anchors": deepcopy(anchor_evidence),
        "source_route": str(source_route or ""),
        "selected_source": selected_source,
        "source_refs": _derived_source_refs(residual),
    }


def reconcile_source_unitemized_residuals(
    parse_result: Any,
    records: Sequence[dict[str, Any]],
    header_records: Sequence[dict[str, Any]],
    *,
    source_route: str | None = None,
    selected_source: str = "",
) -> list[dict[str, Any]]:
    """Publish only source-proven unitemized residuals on statement headers.

    A balance discontinuity is not a transaction row.  This reconciliation
    therefore keeps the visible transaction dataset unchanged and adds explicit
    source-scoped aggregate residuals only when terminal issuer aggregates,
    page-local carry balances, and an independent row-anchor census agree.
    """
    copied_headers = [deepcopy(header) for header in header_records]
    if (
        not records
        or not copied_headers
        or str(selected_source or "") in _ANCHOR_DEPENDENT_SELECTED_SOURCES
    ):
        return copied_headers
    anchor_evidence = _cached_independent_row_anchor_evidence(parse_result, source_route=source_route)
    if not anchor_evidence:
        return copied_headers
    try:
        facts_by_page, _lines = _page_header_facts(parse_result)
    except (AttributeError, TypeError, ValueError):
        return copied_headers
    for header in copied_headers:
        scope = _header_scope(header)
        scoped = _scope_rows(records, header)
        if scope is None or scoped is None:
            continue
        scoped_rows, rows_by_page = scoped
        scope_anchors = _scope_anchor_evidence(anchor_evidence, rows_by_page, scope)
        visible = _visible_direction_totals(scoped_rows)
        boundaries = _carry_boundaries(facts_by_page, rows_by_page)
        if scope_anchors is None or visible is None or boundaries is None:
            continue
        residuals = _direction_residuals(header, visible, boundaries, terminal_page=scope[1])
        if not residuals:
            continue
        normalized = header.get("normalized") if isinstance(header.get("normalized"), dict) else {}
        source = header.get("source") if isinstance(header.get("source"), dict) else {}
        field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
        for direction, residual in residuals.items():
            prefix = "debit" if direction == "expense" else "credit"
            count_field = f"source_unitemized_{prefix}_count"
            amount_field = f"source_unitemized_{prefix}_amount"
            amount_value = _normalized_money(residual["amount"])
            normalized[count_field] = residual["count"]
            normalized[amount_field] = amount_value
            field_sources[count_field] = _residual_field_source(
                residual,
                field_name=count_field,
                value=residual["count"],
                anchor_evidence=scope_anchors,
                source_route=source_route,
                selected_source=selected_source,
            )
            field_sources[amount_field] = _residual_field_source(
                residual,
                field_name=amount_field,
                value=amount_value,
                anchor_evidence=scope_anchors,
                source_route=source_route,
                selected_source=selected_source,
            )
        header["normalized"] = normalized
        source["field_sources"] = field_sources
        header["source"] = source
    return copied_headers


def _matching_header(record: dict[str, Any], headers: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not headers:
        return None
    page = _record_source_page(record)
    matches = []
    for header in headers:
        source = header.get("source") if isinstance(header.get("source"), dict) else {}
        page_range = source.get("page_range") if isinstance(source.get("page_range"), list) else []
        if page > 0 and len(page_range) >= 2 and int(page_range[0]) <= page <= int(page_range[1]):
            matches.append(header)
    if len(matches) == 1:
        return matches[0]
    if page == 0 and len(headers) == 1:
        return headers[0]
    return None


def _single_context_date(value: Any) -> calendar_date | None:
    dates = _valid_date_tokens(value)
    if len(dates) != 1:
        return None
    try:
        return calendar_date.fromisoformat(dates[0])
    except ValueError:
        return None


def _period_coherence_by_header(
    records: Sequence[dict[str, Any]],
    matched_headers: Sequence[dict[str, Any] | None],
) -> dict[int, bool]:
    """Check each page-matched header period against all parseable row dates."""
    headers: dict[int, dict[str, Any]] = {}
    dates_by_header: dict[int, list[calendar_date]] = defaultdict(list)
    for record, header in zip(records, matched_headers):
        if header is None:
            continue
        key = id(header)
        headers[key] = header
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        row_date = _single_context_date(normalized.get("date"))
        if row_date is None:
            row_date = _single_context_date(normalized.get("timestamp"))
        if row_date is not None:
            dates_by_header[key].append(row_date)

    coherent: dict[int, bool] = {}
    for key, header in headers.items():
        normalized = header.get("normalized") if isinstance(header.get("normalized"), dict) else {}
        start_value = normalized.get("period_start")
        end_value = normalized.get("period_end")
        start = _single_context_date(start_value)
        end = _single_context_date(end_value)
        bounds_are_parseable = (start_value in (None, "") or start is not None) and (
            end_value in (None, "") or end is not None
        )
        bounds_are_ordered = start is None or end is None or start <= end
        coherent[key] = bounds_are_parseable and bounds_are_ordered and all(
            (start is None or row_date >= start) and (end is None or row_date <= end)
            for row_date in dates_by_header.get(key, [])
        )
    return coherent


def attach_statement_context(
    records: Sequence[dict[str, Any]],
    header_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link header scopes and fill only missing, applicable row context."""
    attached: list[dict[str, Any]] = []
    matched_headers = [_matching_header(record, header_records) for record in records]
    coherent_periods = _period_coherence_by_header(records, matched_headers)
    for source_record, header in zip(records, matched_headers):
        record = {
            **source_record,
            "normalized": dict(source_record.get("normalized") or {}),
            "canonical_raw": dict(source_record.get("canonical_raw") or {}),
            "raw": dict(source_record.get("raw") or {}),
            "source": dict(source_record.get("source") or {}),
        }
        if header is None:
            attached.append(record)
            continue
        header_id = str(header.get("record_id") or "")
        normalized = record["normalized"]
        canonical_raw = record["canonical_raw"]
        header_normalized = header.get("normalized") if isinstance(header.get("normalized"), dict) else {}
        header_raw = header.get("canonical_raw") if isinstance(header.get("canonical_raw"), dict) else {}
        header_source = header.get("source") if isinstance(header.get("source"), dict) else {}
        header_field_sources = (
            header_source.get("field_sources") if isinstance(header_source.get("field_sources"), dict) else {}
        )
        normalized["statement_header_id"] = header_id
        source_field_sources = (
            dict(record["source"].get("field_sources") or {})
            if isinstance(record["source"].get("field_sources"), dict)
            else {}
        )
        source_field_sources["statement_header_id"] = {
            "source": "statement_header_link",
            "record_id": header_id,
        }
        context_pairs = [("account_number", "own_account"), *[(key, key) for key in _TRANSACTION_CONTEXT_FIELDS]]
        for source_key, target_key in context_pairs:
            if source_key in {"period_start", "period_end"} and not coherent_periods.get(id(header), True):
                continue
            value = header_normalized.get(source_key)
            if value in (None, "") or normalized.get(target_key) not in (None, ""):
                continue
            normalized[target_key] = value
            if header_raw.get(source_key) not in (None, ""):
                canonical_raw[target_key] = header_raw[source_key]
            if isinstance(header_field_sources.get(source_key), dict):
                source_field_sources[target_key] = dict(header_field_sources[source_key])
        record["source"]["field_sources"] = source_field_sources
        attached.append(record)
    return attached


__all__ = [
    "attach_statement_context",
    "build_statement_header_records",
    "page_texts_with_business_headers",
    "reconcile_source_unitemized_residuals",
]
