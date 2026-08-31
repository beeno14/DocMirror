"""Source-backed statement header recovery and transaction context linking.

The transaction table and the statement header are two different source
planes.  This module keeps them separate: it recovers business header facts
from bounded, positioned source text, emits one header record per statement
scope, and links transaction records to the applicable scope without
overwriting row-local facts.
"""

from __future__ import annotations

import calendar
import math
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Sequence

from docmirror.plugins._runtime.evidence_access import text_atoms
from docmirror.plugins.bank_statement.embedded_metadata import extract_embedded_business_metadata
from docmirror.plugins.bank_statement.header_resolve import normalize_bank_matching_text
from docmirror.plugins.bank_statement.work_cache import memoize_bank_document_work

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
_TIME_TOKEN_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
_MONEY_TOKEN_RE = re.compile(r"^[+-]?(?:[¥￥$]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")
_PAGE_LABEL_ATOM_RE = re.compile(r"(?:页码|page)\s*[:：]?", re.IGNORECASE)
_PAGE_VALUE_ATOM_RE = re.compile(r"\d{1,5}\s*(?:[/／]|[-—])\s*\d{1,5}")
_SEAL_CODE_RE = re.compile(r"(?=[A-Z0-9]{8,32}\Z)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+")
_LOCAL_FIRST_PAGE_RES = (
    re.compile(r"第\s*1\s*页\s*[,，/]?\s*(?:共|of)\s*\d+\s*页", re.I),
    re.compile(r"(?:页码|page)\s*[:：]?\s*1\s*[/／]\s*\d+", re.I),
    re.compile(r"(?:页码|page)\s*[:：]?\s*1\s*[-—]\s*1(?:\D|$)", re.I),
    re.compile(r"page\s*1\s*(?:of|/)\s*\d+", re.I),
)
_TITLE_MARKERS = (
    "账务明细清单",
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
_TITLE_DISCLAIMER_MARKERS = (
    "仅供参考",
    "数据缺失",
    "可能导致",
    "不作为",
    "免责声明",
    "法律效力",
    "截至打印时间下方无其他明细内容",
    "最终解释权",
    "重要提示",
    "for reference only",
    "not legal proof",
    "does not constitute legal proof",
)
_TITLE_DISCLAIMER_PREFIX_RE = re.compile(
    r"^\s*(?:说明|提示|注|声明|disclaimer|notice)\s*[:：]",
    re.IGNORECASE,
)
_FOOTER_CUTOFF_RE = re.compile(
    r"(?P<notice>截至打印时间下方无其他明细内容)"
    r"[,，。；;\s]*"
    r"(?:交易明细截止|明细截止)"
    r"(?P<timestamp>20\d{2}(?:年\d{1,2}月\d{1,2}日|[-/.]\d{1,2}[-/.]\d{1,2})"
    r"(?:\s|T)*(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}时\d{1,2}分(?:\d{1,2}秒)?))"
)
_BUSINESS_STAMP_NAMES = (
    "零售业务电子凭证专用章",
    "电子回单业务专用章",
    "账户明细专用章",
    "明细回单专用章",
    "对账单专用章",
    "回单专用章",
    "业务专用章",
    "对账专用章",
    "电子专用章",
    "会计业务章",
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


def _source_atom_text(atom: dict[str, Any]) -> str:
    source_text = atom.get("_source_text")
    return str(atom.get("text") if source_text is None else source_text).strip()


def _compact(value: Any) -> str:
    matching = normalize_bank_matching_text(_nfkc(value))
    return re.sub(r"[\s:：._()（）\[\]【】]+", "", matching).casefold()


def _length_preserving_matching_text(value: Any) -> str:
    source = _nfkc(value)
    matching = normalize_bank_matching_text(source)
    return matching if len(matching) == len(source) else ""


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


def _normalize_time(value: Any) -> str:
    text = _nfkc(value)
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if match is None:
        return text
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return text
    return f"{hour:02d}:{minute:02d}:{second:02d}"


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
    "print_timestamp": ("打印时间", "下单时间", "生成时间", "Print Time", "Printing Time"),
    "print_time": ("打印时刻", "打印具体时间", "Print Clock Time"),
    "issuing_bank": ("签发银行", "印章银行", "Issuing Bank"),
    "issuing_branch": ("签发机构", "印章机构", "Issuing Branch"),
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
        "交易明细对应时间段",
        "起止日期",
        "起讫日期",
        "时间范围",
        "日期范围",
        "对账周期",
        "账单所属期间",
        "Statement Period",
        "Statement Covered Period",
        "Query Period",
        "Start Time & End Time",
    ),
    "statement_period": ("账单统计日期", "账单期间", "Statement Statistics Date"),
    "statement_cutoff_date": ("出单截至日期", "Statement Cutoff Date"),
    "statement_cutoff_timestamp": ("交易明细截止时间", "明细截止时间", "Statement Cutoff Time"),
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
        "账户",
        "账户号",
        "银行账号",
        "客户账号",
        "账户代号",
        "卡/账号",
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
    "branch_number": ("网点号", "网点编号", "Branch Number", "Outlet Number"),
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
    "currency": ("币种", "幣種", "账单币种", "币别", "货币", "Currency"),
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
    "verification_code": ("验证码", "验证编号", "核验编号", "Verification Code"),
    "proof_number": ("证明编号", "编号", "Proof Number"),
    "wechat_id": ("微信号", "WeChat ID"),
    "id_type": ("证件种类", "证件类型", "ID Type"),
    "id_number": ("证件号码", "身份证号", "身份证号码", "ID Number"),
    "filter_condition": ("筛选条件", "流水范围", "查询范围", "Filter", "Scope"),
    "direction_filter": ("借贷方向", "借/贷标记", "交易方向", "收支类别", "Direction"),
    "transaction_type_filter": ("交易类型",),
    "transfer_amount_filter": ("转账金额区间",),
    "counterparty_name_filter": ("对方户名",),
    "counterparty_account_filter": ("对方账号",),
    "purpose_note_filter": ("用途/备注",),
    "sort_order": ("排序方向", "排序方式", "Sort Order"),
    "print_channel": ("打印渠道", "Print Channel"),
    "print_teller": ("打印柜员", "打印操作员", "柜员号", "Print Teller"),
    "print_count": ("已打印次数", "打印次数", "Print Count"),
    "print_method": ("打印方式", "Print Method"),
    "device_number": ("设备编号", "设备号", "Device Number"),
    "query_teller": (
        "查询柜员SearchTeller",
        "查询柜员",
        "Search Teller",
        "柜员 Search Teller",
    ),
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
        "本页支出笔数",
        "总支出笔数",
        "总支出入笔数",
    ),
    "debit_total": (
        "借方总金额",
        "借方发生额",
        "借方发生额汇总",
        "借方发生总额",
        "借方合计金额",
        "本月累计借方发生额",
        "汇总借方发生",
        "汇总借方发生额",
        "支出总金额",
        "支出总额",
        "支出金额",
        "支出金额合计",
        "本页支出算数合计",
        "本页支出算术合计",
        "总支出金额",
    ),
    "credit_count": (
        "贷方总笔数",
        "贷方笔数",
        "贷方合计笔数",
        "本月累计贷方发生数",
        "收入总笔数",
        "收入笔数",
        "收入交易笔数",
        "本页收入笔数",
        "总收入笔数",
    ),
    "credit_total": (
        "贷方总金额",
        "贷方发生额",
        "贷方发生额汇总",
        "贷方发生总额",
        "贷方合计金额",
        "本月累计贷方发生额",
        "汇总贷方发生",
        "汇总贷方发生额",
        "收入总金额",
        "收入总额",
        "收入金额",
        "收入金额合计",
        "本页收入算数合计",
        "本页收入算术合计",
        "总收入金额",
    ),
    "opening_balance": ("期初余额", "上期余额", "Opening Balance"),
    "closing_balance": ("期末余额", "Closing Balance"),
    "brought_forward_balance": (
        "承前余额",
        "承前",
        "承上余额",
        "上页余额",
        "Brought Forward Balance",
        "Last balance",
    ),
    "seal_type": ("印章类型", "Seal Type"),
}

_NORMALIZED_ALIAS_TO_FIELD = {
    _compact(alias): field_name for field_name, aliases in _FIELD_ALIASES.items() for alias in aliases
}
_FIELD_ALIAS_PARTS = {
    field_name: {_compact(alias) for alias in aliases if _compact(alias)}
    for field_name, aliases in _FIELD_ALIASES.items()
}
_OPENING_BRANCH_ROW_LABELS = {
    _compact(alias)
    for alias in ("开户银行", "开户行", "开户机构", "开户网点", "Opening Branch", "Sub Branch")
}
_ISOLATED_CURRENCY_METADATA_FIELDS = {
    "statement_year",
    "statement_month_number",
    "source_header_page_label",
}
_PAGE_LOCAL_DIRECTION_LABELS = {
    "debit_count": {_compact("本页支出笔数")},
    "debit_total": {_compact("本页支出算数合计"), _compact("本页支出算术合计")},
    "credit_count": {_compact("本页收入笔数")},
    "credit_total": {_compact("本页收入算数合计"), _compact("本页收入算术合计")},
}
_FILTER_CONTEXT_FIELDS = {
    "direction_filter",
    "transaction_type_filter",
    "transfer_amount_filter",
    "counterparty_name_filter",
    "counterparty_account_filter",
    "purpose_note_filter",
}
_INLINE_KV_ALIAS_RE = re.compile(
    r"(?P<prefix>^|[\s/／,，;；(（])(?P<label>"
    + "|".join(
        sorted(
            {
                re.escape(matching_alias)
                for aliases in _FIELD_ALIASES.values()
                for alias in aliases
                if (matching_alias := _length_preserving_matching_text(alias))
            },
            key=len,
            reverse=True,
        )
    )
    + r")\s*[:：]",
    re.IGNORECASE,
)
_CERTIFICATE_SUBJECT_RE = re.compile(
    r"^\s*兹证明\s*[:：]\s*"
    r"(?P<holder>[\u4e00-\u9fffA-Za-z·•\s]{1,100}?)\s*"
    r"[（(]\s*(?P<id_label>身份证(?:号|号码)?|证件号码)\s*[:：]\s*"
    r"(?P<id_number>[A-Za-z0-9*]{6,30})\s*[)）]\s*[,，]?\s*"
    r"在其微信号\s*[:：]\s*(?P<wechat_id>[A-Za-z0-9_.-]{3,80})\s*"
    r"中的交易明细(?:信息)?如下\s*[:：]?\s*$",
    re.IGNORECASE,
)
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
_DATETIME_FIELDS = {
    "print_timestamp",
    "statement_cutoff_timestamp",
    "application_time",
    "query_timestamp",
    "issue_timestamp",
}
_TIME_FIELDS = {"print_time"}
_SINGLE_ATOM_VALUE_FIELDS = (
    _COUNT_FIELDS
    | _MONEY_FIELDS
    | _DATE_FIELDS
    | _TIME_FIELDS
    | {
        "currency",
        "account_number",
        "card_number",
        "internal_account",
        "customer_number",
        "branch_number",
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
    "print_channel",
    "print_count",
    "print_date",
    "print_timestamp",
    "print_time",
    "print_method",
    "device_number",
    "print_teller",
    "period_end",
    "statement_cutoff_date",
    "statement_cutoff_timestamp",
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
    "statement_disclaimer",
    "bank_name",
    "account_holder",
    "account_number",
    "card_number",
    "internal_account",
    "customer_number",
    "branch_name",
    "branch_number",
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
    derivation: str = ""
    source_detail: dict[str, Any] | None = None


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
    if field_key in _TIME_FIELDS:
        return _normalize_time(text)
    if field_key in _DATETIME_FIELDS:
        text = re.sub(
            r"(\d{1,2})时(\d{1,2})分(?:(\d{1,2})秒)?",
            lambda match: (
                f"{int(match.group(1)):02d}:{int(match.group(2)):02d}:"
                f"{int(match.group(3) or 0):02d}"
            ),
            text,
        )
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
        "branch_number",
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


def _bounded_explicit_filter_text(value: Any, *, maximum_length: int = 100) -> bool:
    text = _nfkc(value)
    return bool(
        1 <= len(text) <= maximum_length
        and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text)
        and not _INLINE_KV_ALIAS_RE.search(text)
        and not _is_header_only_value(text)
    )


def _transfer_amount_filter_is_plausible(value: Any) -> bool:
    text = _nfkc(value)
    if _compact(text) in {"无", "全部", "不限"}:
        return True
    parts = [part.strip() for part in re.split(r"\s*(?:-|—|~|～|至|到)\s*", text) if part.strip()]
    if len(parts) == 1:
        return _strict_source_money(parts[0], nonnegative=True) is not None
    if len(parts) != 2:
        return False
    lower = _strict_source_money(parts[0], nonnegative=True)
    upper = _strict_source_money(parts[1], nonnegative=True)
    return lower is not None and upper is not None and lower <= upper


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
    if field_key in _TIME_FIELDS:
        normalized = _normalize_time(text)
        return bool(re.fullmatch(r"\d{2}:\d{2}:\d{2}", normalized)) and normalized != text or bool(
            re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?", text)
        )
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
    if field_key == "amount_unit":
        return compact in {"元", "角", "分", "千元", "万元", "百万元", "亿元"}
    if field_key == "seal_code":
        return bool(_SEAL_CODE_RE.fullmatch(text))
    if field_key == "direction_filter":
        return compact in {"全部", "收入", "支出", "收支", "借方", "贷方"}
    if field_key == "transaction_type_filter":
        return compact in {"无", "全部", "不限"} or _bounded_explicit_filter_text(text, maximum_length=80)
    if field_key == "transfer_amount_filter":
        return _transfer_amount_filter_is_plausible(text)
    if field_key == "counterparty_name_filter":
        return compact in {"无", "全部", "不限"} or _bounded_explicit_filter_text(text)
    if field_key == "counterparty_account_filter":
        token = re.sub(r"\s+", "", text)
        return compact in {"无", "全部", "不限"} or bool(re.fullmatch(r"[A-Za-z0-9*_.\-/]{3,80}", token))
    if field_key == "purpose_note_filter":
        return compact in {"无", "全部", "不限"} or _bounded_explicit_filter_text(text)
    if field_key in {
        "account_number",
        "card_number",
        "internal_account",
        "customer_number",
        "branch_number",
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
                r"(?:用途|备注|对方|账号|账户余额|上页余额|打印日期|查询日期|Last\s+balance|Counterparty)",
                text,
                re.IGNORECASE,
            )
        )
    if field_key == "bank_name":
        return len(text) <= 100 and "银行" in text and not _DATE_TOKEN_RE.search(text)
    return not bool(_is_header_only_value(text))


def _labelled_fact_value_is_plausible(field_key: str, raw_name: Any, value: Any) -> bool:
    if not _fact_value_is_plausible(field_key, value):
        return False
    if field_key == "account_number" and _compact(raw_name) == _compact("账户"):
        token = re.sub(r"\s+", "", _nfkc(value))
        return bool(
            6 <= len(token) <= 80
            and re.search(r"\d", token)
            and re.fullmatch(r"[A-Za-z0-9*_.\-/]+", token)
        )
    return True


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
        source_text = str(atom.get("text") or "").strip()
        text = _nfkc(source_text)
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
                "_source_text": source_text,
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
            source_text = str(getattr(block, "content", "") or "").strip()
            text = _nfkc(source_text)
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
                    "_source_text": source_text,
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
    ledger_header_top = next(
        (min(float(atom["bbox"][1]) for atom in row) for row in rows if _is_transaction_header_row(row)),
        None,
    )
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
    if ledger_header_top is not None:
        cutoff = min(cutoff, ledger_header_top)
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
            if _PAGE_LABEL_ATOM_RE.fullmatch(raw):
                span = _LabelSpan(index, index, "source_header_page_label", _source_atom_text(row[index]))
            else:
                generic = re.fullmatch(r"([^:：]{2,40})[:：]\s*(.*)", raw)
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


def _bounded_page_value_atom(
    label_atom: dict[str, Any],
    value_atoms: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return one source page token tightly bounded to an explicit page label."""

    if not _PAGE_LABEL_ATOM_RE.fullmatch(_nfkc(label_atom.get("text"))):
        return None
    label_bbox = label_atom.get("bbox")
    if not isinstance(label_bbox, (list, tuple)) or len(label_bbox) < 4:
        return None
    candidates: list[dict[str, Any]] = []
    for atom in value_atoms:
        value = _nfkc(atom.get("text"))
        bbox = atom.get("bbox")
        if not _PAGE_VALUE_ATOM_RE.fullmatch(value) or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        horizontal_gap = float(bbox[0]) - float(label_bbox[2])
        vertical_overlap = min(float(bbox[3]), float(label_bbox[3])) - max(
            float(bbox[1]),
            float(label_bbox[1]),
        )
        if -2.0 <= horizontal_gap <= 48.0 and vertical_overlap >= 1.0:
            candidates.append(atom)
    return candidates[0] if len(candidates) == 1 else None


def _isolated_currency_context(
    row: Sequence[dict[str, Any]],
    spans: Sequence[_LabelSpan],
) -> tuple[
    dict[str, Any],
    _LabelSpan,
    _LabelSpan,
    list[dict[str, Any]],
    list[dict[str, Any]],
] | None:
    """Recognize a detached currency cell in a strictly structured metadata row."""

    opening_spans = [
        span
        for span in spans
        if span.field_key == "branch_name" and _compact(span.raw_name) in _OPENING_BRANCH_ROW_LABELS
    ]
    if len(opening_spans) != 1 or any(span.field_key == "currency" for span in spans):
        return None
    branch_span = opening_spans[0]
    next_span = next((span for span in spans if span.start > branch_span.end), None)
    if next_span is None or next_span.field_key not in _ISOLATED_CURRENCY_METADATA_FIELDS:
        return None
    between = list(row[branch_span.end + 1 : next_span.start])
    if not between:
        return None
    currency_atom = between[-1]
    if not _fact_value_is_plausible("currency", _nfkc(currency_atom.get("text"))):
        return None
    branch_value_atoms = between[:-1]
    if not branch_span.inline_value and not branch_value_atoms:
        return None
    predecessor = branch_value_atoms[-1] if branch_value_atoms else row[branch_span.end]
    separation = float(currency_atom["bbox"][0]) - float(predecessor["bbox"][2])
    metadata_gap = float(row[next_span.start]["bbox"][0]) - float(currency_atom["bbox"][2])
    if not 12.0 <= separation <= 180.0 or not -2.0 <= metadata_gap <= 180.0:
        return None
    supporting = [
        currency_atom,
        *row[branch_span.start : branch_span.end + 1],
        *branch_value_atoms,
        *row[next_span.start : next_span.end + 1],
    ]
    return currency_atom, branch_span, next_span, branch_value_atoms, supporting


def _inline_header_transaction_time_period(
    span: _LabelSpan,
    row: Sequence[dict[str, Any]],
    value_atoms: Sequence[dict[str, Any]],
) -> str:
    """Resolve only the explicit header form ``交易时间:<date>`` ``至`` ``<date>``."""
    start = _nfkc(span.inline_value).strip()
    if (
        span.field_key
        or _compact(span.raw_name) != _compact("交易时间")
        or not _DATE_TOKEN_RE.fullmatch(start)
        or len(_valid_date_tokens(start)) != 1
        or len(value_atoms) != 2
    ):
        return ""
    separator, end_atom = value_atoms
    end = _nfkc(end_atom.get("text")).strip()
    if (
        _nfkc(separator.get("text")).strip() != "至"
        or not _DATE_TOKEN_RE.fullmatch(end)
        or len(_valid_date_tokens(end)) != 1
    ):
        return ""
    try:
        separator_gap = float(separator["bbox"][0]) - float(row[span.end]["bbox"][2])
        end_gap = float(end_atom["bbox"][0]) - float(separator["bbox"][2])
    except (KeyError, IndexError, TypeError, ValueError):
        return ""
    candidate = f"{start} 至 {end}"
    if not (-2.0 <= separator_gap <= 180.0 and -2.0 <= end_gap <= 180.0):
        return ""
    return candidate if _period_dates_are_valid(candidate, minimum=2) else ""


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


def _positioned_atom_fact(
    atom: dict[str, Any],
    page: int,
    field_key: str,
    raw_name: str,
    raw_value: str,
) -> _HeaderFact:
    return _HeaderFact(
        field_key=field_key,
        raw_name=raw_name,
        raw_value=raw_value,
        normalized_value=_normalize_field_value(field_key, raw_value),
        page=page,
        page_id=str(atom.get("page_id") or f"page:{page:04d}"),
        bbox=_bbox_union([atom]),
        evidence_ids=tuple([str(atom.get("id"))] if str(atom.get("id") or "") else []),
        source_kind=_context_source_kind([atom]),
    )


def _multi_inline_kv_facts(row: Sequence[dict[str, Any]], page: int) -> list[_HeaderFact]:
    """Recover distinct explicit KVs printed inside one positioned text atom."""

    facts: list[_HeaderFact] = []
    for atom in row:
        source_text = str(atom.get("text") or "").strip()
        text = _nfkc(source_text)
        matching_text = _length_preserving_matching_text(text)
        # Match compatibility/Traditional label glyphs without losing the
        # source-text offsets used to slice raw labels and values.  Should a
        # future matching normalization expand or contract text, fail closed
        # instead of attaching a value to the wrong source span.
        if not matching_text:
            continue
        matches = list(_INLINE_KV_ALIAS_RE.finditer(matching_text))
        if len(matches) < 2 or text[: matches[0].start("label")].strip():
            continue
        atom_facts: list[_HeaderFact] = []
        valid = True
        for index, match in enumerate(matches):
            raw_label = text[match.start("label") : match.end("label")]
            field_key = _field_for_label(raw_label)
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            raw_value = text[match.end() : value_end].strip(" \t/／,，;；()（）")
            if (
                not field_key
                or not raw_value
                or len(raw_value) > 240
                or re.search(r"[:：]", raw_value)
                or not _labelled_fact_value_is_plausible(field_key, raw_label, raw_value)
            ):
                valid = False
                break
            atom_facts.append(
                _positioned_atom_fact(
                    atom,
                    page,
                    field_key,
                    text[match.start("label") : match.end()].strip(),
                    raw_value,
                )
            )
        field_keys = {fact.field_key for fact in atom_facts}
        mixed_filter_namespace = bool(field_keys & _FILTER_CONTEXT_FIELDS) and not field_keys <= _FILTER_CONTEXT_FIELDS
        if valid and not mixed_filter_namespace and len(field_keys) >= 2:
            facts.extend(atom_facts)
    return facts


def _certificate_subject_facts(row: Sequence[dict[str, Any]], page: int) -> list[_HeaderFact]:
    """Promote the explicit subject identity of a payment-detail certificate."""

    for atom in row:
        match = _CERTIFICATE_SUBJECT_RE.fullmatch(_nfkc(atom.get("text")))
        if match is None:
            continue
        values = {
            "account_holder": match.group("holder").strip(),
            "id_number": match.group("id_number").strip(),
            "wechat_id": match.group("wechat_id").strip(),
        }
        chinese_id = values["id_number"]
        if not re.fullmatch(r"(?:\d{15}|[0-9*]{17}[0-9Xx*])", chinese_id):
            continue
        if not all(_fact_value_is_plausible(field_key, value) for field_key, value in values.items()):
            continue
        return [
            _positioned_atom_fact(atom, page, "account_holder", "兹证明:", values["account_holder"]),
            _positioned_atom_fact(
                atom,
                page,
                "id_number",
                f"{match.group('id_label')}:",
                values["id_number"],
            ),
            _positioned_atom_fact(atom, page, "wechat_id", "微信号:", values["wechat_id"]),
        ]
    return []


def _facts_from_row(row: Sequence[dict[str, Any]], page: int) -> list[_HeaderFact]:
    if _is_transaction_header_row(row):
        return []
    spans = _label_spans(row)
    isolated_currency = _isolated_currency_context(row, spans)
    facts: list[_HeaderFact] = []
    for position, span in enumerate(spans):
        value_atoms = list(row[span.end + 1 : spans[position + 1].start if position + 1 < len(spans) else len(row)])
        field_key = span.field_key
        if isolated_currency is not None and span == isolated_currency[1]:
            value_atoms = [atom for atom in value_atoms if atom is not isolated_currency[0]]
        if not span.inline_value and field_key == "source_header_page_label":
            page_value_atom = _bounded_page_value_atom(row[span.end], value_atoms)
            value_atoms = [page_value_atom] if page_value_atom is not None else []
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
                (
                    atom
                    for atom in value_atoms
                    if _labelled_fact_value_is_plausible(
                        field_key,
                        span.raw_name,
                        _nfkc(atom.get("text")),
                    )
                ),
                None,
            )
            if single_value_atom is not None:
                value_atoms = [single_value_atom]
        trailing_value = (
            _source_atom_text(value_atoms[0])
            if field_key == "source_header_page_label" and len(value_atoms) == 1
            else _join_value_atoms(value_atoms)
        )
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
        if header_transaction_period := _inline_header_transaction_time_period(span, row, value_atoms):
            field_key = "query_period"
            value = header_transaction_period
            used_value_atoms = value_atoms
        if (
            span.inline_value
            and field_key in _DATETIME_FIELDS
            and re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", trailing_value)
        ):
            value = f"{span.inline_value} {trailing_value}"
            used_value_atoms = value_atoms
        if field_key == "print_timestamp" and _TIME_TOKEN_RE.fullmatch(_nfkc(value)):
            # The same visible label is used both for a full timestamp and for
            # a clock-only footer value. Preserve that source distinction.
            field_key = "print_time"
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
        if recognized_field and not _labelled_fact_value_is_plausible(field_key, span.raw_name, value):
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
    if isolated_currency is not None:
        currency_atom, branch_span, metadata_span, branch_value_atoms, supporting = isolated_currency
        raw_value = _source_atom_text(currency_atom)
        source_kind = _context_source_kind(supporting)
        bbox = _bbox_union(supporting)
        evidence_ids = tuple(
            dict.fromkeys(str(atom.get("id") or "") for atom in supporting if str(atom.get("id") or ""))
        )
        facts.append(
            _HeaderFact(
                field_key="currency",
                raw_name="unlabelled_currency",
                raw_value=raw_value,
                normalized_value=_normalize_field_value("currency", raw_value),
                page=page,
                page_id=str(currency_atom.get("page_id") or f"page:{page:04d}"),
                bbox=bbox,
                evidence_ids=evidence_ids,
                source_kind=source_kind,
                derivation="isolated_currency_in_opening_branch_metadata_row",
                source_detail={
                    "page": page,
                    "raw_value": raw_value,
                    "opening_branch_label": branch_span.raw_name,
                    "branch_value_texts": [
                        *([branch_span.inline_value] if branch_span.inline_value else []),
                        *[_source_atom_text(atom) for atom in branch_value_atoms],
                    ],
                    "branch_value_evidence_ids": [
                        str(atom.get("id") or "") for atom in branch_value_atoms if str(atom.get("id") or "")
                    ],
                    "metadata_label": metadata_span.raw_name,
                    "value_evidence_id": str(currency_atom.get("id") or ""),
                    "evidence_ids": list(evidence_ids),
                    "bbox": list(bbox) if bbox else None,
                    "source": source_kind,
                    "normalized_only": False,
                },
            )
        )
    multi_inline_facts = _multi_inline_kv_facts(row, page)
    multi_inline_candidate_evidence = {
        str(atom.get("id"))
        for atom in row
        if str(atom.get("id") or "")
        and (matching_text := _length_preserving_matching_text(atom.get("text")))
        and len(list(_INLINE_KV_ALIAS_RE.finditer(matching_text))) >= 2
    }
    if multi_inline_candidate_evidence:
        facts = [
            fact
            for fact in facts
            if fact.field_key.startswith("source_header_")
            or not multi_inline_candidate_evidence.intersection(fact.evidence_ids)
        ]
    facts.extend(multi_inline_facts)
    facts.extend(_certificate_subject_facts(row, page))
    unique: dict[tuple[str, str, str], _HeaderFact] = {}
    for fact in facts:
        unique.setdefault((fact.field_key, _nfkc(fact.raw_name), _nfkc(fact.raw_value)), fact)
    return list(unique.values())


def _paired_cumulative_direction_facts(
    row: Sequence[dict[str, Any]],
    page: int,
) -> list[_HeaderFact]:
    """Recover a paired cumulative debit/credit footer whose right label wraps."""
    debit_labels: list[tuple[dict[str, Any], str]] = []
    credit_labels: list[dict[str, Any]] = []
    for atom in row:
        compact = _compact(atom.get("text"))
        match = re.fullmatch(r"本月累计借方发生(?P<suffix>数|额)", compact)
        if match:
            debit_labels.append((atom, match.group("suffix")))
        elif compact == "本月累计贷方":
            credit_labels.append(atom)
    if len(debit_labels) != 1 or len(credit_labels) != 1:
        return []
    debit_label, suffix = debit_labels[0]
    credit_label = credit_labels[0]
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

    def ordered_neighbour(
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        max_gap: float,
    ) -> bool:
        left_bbox = left["bbox"]
        right_bbox = right["bbox"]
        gap = float(right_bbox[0]) - float(left_bbox[2])
        vertical_overlap = min(float(left_bbox[3]), float(right_bbox[3])) - max(
            float(left_bbox[1]),
            float(right_bbox[1]),
        )
        return -2.0 <= gap <= max_gap and vertical_overlap >= 1.0

    if not (
        ordered_neighbour(debit_label, debit_value, max_gap=80.0)
        and ordered_neighbour(debit_value, credit_label, max_gap=250.0)
        and ordered_neighbour(credit_label, credit_value, max_gap=80.0)
    ):
        return []
    debit_supporting = [debit_label, debit_value]
    debit_raw_value = _nfkc(debit_value.get("text"))
    debit_fact = _HeaderFact(
        debit_field,
        _nfkc(debit_label.get("text")),
        debit_raw_value,
        _normalize_field_value(debit_field, debit_raw_value),
        page,
        str(debit_label.get("page_id") or f"page:{page:04d}"),
        _bbox_union(debit_supporting),
        tuple(str(atom.get("id") or "") for atom in debit_supporting if str(atom.get("id") or "")),
        _context_source_kind(debit_supporting),
    )

    credit_raw_name = f"本月累计贷方发生{suffix}"
    credit_supporting = [credit_label, debit_label, credit_value]
    credit_raw_value = _nfkc(credit_value.get("text"))
    credit_bbox = _bbox_union(credit_supporting)
    credit_evidence_ids = tuple(
        str(atom.get("id") or "") for atom in credit_supporting if str(atom.get("id") or "")
    )
    credit_source_kind = _context_source_kind(credit_supporting)
    credit_fact = _HeaderFact(
        credit_field,
        credit_raw_name,
        credit_raw_value,
        _normalize_field_value(credit_field, credit_raw_value),
        page,
        str(credit_label.get("page_id") or f"page:{page:04d}"),
        credit_bbox,
        credit_evidence_ids,
        credit_source_kind,
        derivation="structural_pair_suffix_from_explicit_left_label",
        source_detail={
            "page": page,
            "derived_raw_name": credit_raw_name,
            "left_label": _nfkc(debit_label.get("text")),
            "right_label": _nfkc(credit_label.get("text")),
            "raw_value": credit_raw_value,
            "value_evidence_id": str(credit_value.get("id") or ""),
            "evidence_ids": list(credit_evidence_ids),
            "bbox": list(credit_bbox) if credit_bbox else None,
            "source": credit_source_kind,
            "normalized_only": False,
        },
    )
    return [debit_fact, credit_fact]


def _is_statement_disclaimer(value: Any) -> bool:
    text = _nfkc(value)
    compact = _compact(text)
    return _TITLE_DISCLAIMER_PREFIX_RE.search(text) is not None or any(
        _compact(marker) in compact for marker in _TITLE_DISCLAIMER_MARKERS
    )


def _atomic_disclaimer_value(value: Any) -> str:
    text = _nfkc(value)
    match = _FOOTER_CUTOFF_RE.search(text)
    return match.group("notice") if match else text


def _statement_disclaimer_fact(
    header_rows: Sequence[Sequence[dict[str, Any]]],
    page: int,
) -> _HeaderFact | None:
    row_candidates: list[tuple[int, list[dict[str, Any]], str]] = []
    for row_index, row in enumerate(header_rows):
        ordered = sorted(row, key=lambda atom: float(atom["bbox"][0]))
        text = "".join(_nfkc(atom.get("text")) for atom in ordered if _nfkc(atom.get("text")))
        if text:
            row_candidates.append((row_index, ordered, text))

    seed_indexes = {
        index
        for index, (_row_index, _atoms, text) in enumerate(row_candidates)
        if _is_statement_disclaimer(text)
    }
    if not seed_indexes:
        return None

    selected_indexes = set(seed_indexes)
    for seed_index in sorted(seed_indexes):
        cursor = seed_index
        while cursor + 1 < len(row_candidates):
            _row_index, current_atoms, current_text = row_candidates[cursor]
            next_row_index, next_atoms, next_text = row_candidates[cursor + 1]
            if next_row_index != row_candidates[cursor][0] + 1:
                break
            if not re.search(r"[,，;；:：]\s*$", current_text):
                break
            current_bottom = max(float(atom["bbox"][3]) for atom in current_atoms)
            next_top = min(float(atom["bbox"][1]) for atom in next_atoms)
            current_height = max(float(atom["bbox"][3]) - float(atom["bbox"][1]) for atom in current_atoms)
            if next_top - current_bottom > max(6.0, current_height * 1.5):
                break
            if re.search(r"^\s*第\s*\d+\s*页", next_text) or _is_transaction_data_row(next_atoms):
                break
            selected_indexes.add(cursor + 1)
            cursor += 1

    runs: list[list[int]] = []
    for index in sorted(selected_indexes):
        if not runs or index > runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)

    candidates: list[tuple[list[dict[str, Any]], str]] = []
    for run in runs:
        atoms = [atom for index in run for atom in row_candidates[index][1]]
        value = "".join(_atomic_disclaimer_value(row_candidates[index][2]) for index in run)
        if value:
            candidates.append((atoms, value))
    values = {value for _atoms, value in candidates if value}
    if len(values) != 1:
        return None
    value = next(iter(values))
    atoms = [atom for candidate_atoms, candidate_value in candidates if candidate_value == value for atom in candidate_atoms]
    first = atoms[0]
    composite = next(
        (_nfkc(atom.get("text")) for atom in atoms if _FOOTER_CUTOFF_RE.search(_nfkc(atom.get("text")))),
        "",
    )
    return _HeaderFact(
        "statement_disclaimer",
        "document_disclaimer",
        value,
        value,
        page,
        str(first.get("page_id") or f"page:{page:04d}"),
        _bbox_union(atoms),
        tuple(dict.fromkeys(str(atom.get("id") or "") for atom in atoms if str(atom.get("id") or ""))),
        _context_source_kind(atoms),
        derivation=(
            "atomic_notice_from_composite_footer"
            if composite
            else "joined_adjacent_statement_disclaimer_lines"
            if len(atoms) > 1
            else ""
        ),
        source_detail=(
            {
                "page": page,
                "source_text": composite or "".join(_nfkc(atom.get("text")) for atom in atoms),
                "raw_value": value,
                "source": _context_source_kind(atoms),
                "normalized_only": False,
            }
            if composite or len(atoms) > 1
            else None
        ),
    )


def _statement_cutoff_timestamp_fact(
    rows: Sequence[Sequence[dict[str, Any]]],
    page: int,
) -> _HeaderFact | None:
    candidates: list[tuple[str, list[dict[str, Any]], str]] = []
    for row in rows:
        source_text = "".join(_source_atom_text(atom) for atom in row)
        match = _FOOTER_CUTOFF_RE.search(_nfkc(source_text))
        if match:
            candidates.append((match.group("timestamp"), list(row), source_text))
    normalized = {
        _normalize_field_value("statement_cutoff_timestamp", raw_value)
        for raw_value, _atoms, _source_text in candidates
    }
    if len(normalized) != 1 or not candidates:
        return None
    normalized_value = next(iter(normalized))
    matching = [item for item in candidates if _normalize_field_value("statement_cutoff_timestamp", item[0]) == normalized_value]
    raw_values = list(dict.fromkeys(item[0] for item in matching))
    if len(raw_values) != 1:
        return None
    raw_value = raw_values[0]
    supporting = [atom for _value, atoms, _source_text in matching for atom in atoms]
    first = supporting[0]
    return _HeaderFact(
        field_key="statement_cutoff_timestamp",
        raw_name="交易明细截止",
        raw_value=raw_value,
        normalized_value=normalized_value,
        page=page,
        page_id=str(first.get("page_id") or f"page:{page:04d}"),
        bbox=_bbox_union(supporting),
        evidence_ids=tuple(
            dict.fromkeys(str(atom.get("id") or "") for atom in supporting if str(atom.get("id") or ""))
        ),
        source_kind=_context_source_kind(supporting),
        derivation="timestamp_from_explicit_composite_footer",
        source_detail={
            "page": page,
            "source_text": matching[0][2],
            "raw_value": raw_value,
            "source": _context_source_kind(supporting),
            "normalized_only": False,
        },
    )


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
            if _is_statement_disclaimer(text) or re.search(r"[。；;！？!?]", text):
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


def _reconciliation_stamp_groups(
    atoms: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return tightly contiguous source atoms spelling a business seal marker."""

    targets = {_compact(name) for name in _BUSINESS_STAMP_NAMES}
    groups: list[list[dict[str, Any]]] = []
    for row in _baseline_rows(atoms):
        for start in range(len(row)):
            assembled = ""
            group: list[dict[str, Any]] = []
            for atom in row[start:]:
                segment = _compact(atom.get("text"))
                candidate = f"{assembled}{segment}"
                if not segment or not any(target.startswith(candidate) for target in targets):
                    break
                if group:
                    previous_bbox = group[-1]["bbox"]
                    bbox = atom["bbox"]
                    horizontal_gap = float(bbox[0]) - float(previous_bbox[2])
                    vertical_overlap = min(float(bbox[3]), float(previous_bbox[3])) - max(
                        float(bbox[1]),
                        float(previous_bbox[1]),
                    )
                    if not -3.0 <= horizontal_gap <= 12.0 or vertical_overlap < 1.0:
                        break
                assembled += segment
                group.append(atom)
                if assembled in targets:
                    groups.append(group)
                    break
    return groups


def _seal_type_facts(atoms: Sequence[dict[str, Any]], page: int) -> list[_HeaderFact]:
    """Preserve each explicit native seal marker as atomic business metadata."""

    facts: list[_HeaderFact] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for stamp_atoms in _reconciliation_stamp_groups(atoms):
        raw_value = "".join(_source_atom_text(atom) for atom in stamp_atoms)
        evidence_ids = tuple(
            dict.fromkeys(str(atom.get("id") or "") for atom in stamp_atoms if str(atom.get("id") or ""))
        )
        key = (_compact(raw_value), evidence_ids)
        if not raw_value or key in seen:
            continue
        seen.add(key)
        first = stamp_atoms[0]
        facts.append(
            _HeaderFact(
                field_key="seal_type",
                raw_name="印章类型",
                raw_value=raw_value,
                normalized_value=raw_value,
                page=page,
                page_id=str(first.get("page_id") or f"page:{page:04d}"),
                bbox=_bbox_union(stamp_atoms),
                evidence_ids=evidence_ids,
                source_kind=_context_source_kind(stamp_atoms),
                derivation="explicit_business_stamp_text",
                source_detail={
                    "page": page,
                    "stamp_texts": [_source_atom_text(atom) for atom in stamp_atoms],
                    "source": _context_source_kind(stamp_atoms),
                    "normalized_only": False,
                },
            )
        )
    return facts


def _seal_code_fact(atoms: Sequence[dict[str, Any]], page: int) -> _HeaderFact | None:
    """Promote one strict code immediately below an explicit reconciliation stamp."""

    pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for stamp_atoms in _reconciliation_stamp_groups(atoms):
        stamp_bbox = _bbox_union(stamp_atoms)
        if stamp_bbox is None:
            continue
        for atom in atoms:
            if atom in stamp_atoms or not _SEAL_CODE_RE.fullmatch(_source_atom_text(atom)):
                continue
            bbox = atom.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            vertical_gap = float(bbox[1]) - stamp_bbox[3]
            horizontal_overlap = min(float(bbox[2]), stamp_bbox[2]) - max(float(bbox[0]), stamp_bbox[0])
            code_width = max(0.0, float(bbox[2]) - float(bbox[0]))
            code_center = (float(bbox[0]) + float(bbox[2])) / 2.0
            if (
                0.0 <= vertical_gap <= 18.0
                and stamp_bbox[0] <= code_center <= stamp_bbox[2]
                and horizontal_overlap >= max(4.0, code_width * 0.4)
            ):
                pairs.append((atom, stamp_atoms))
    unique_pairs = {
        (
            str(code.get("id") or ""),
            tuple(str(atom.get("id") or "") for atom in stamp_atoms),
        ): (code, stamp_atoms)
        for code, stamp_atoms in pairs
    }
    if len(unique_pairs) != 1:
        return None
    code, stamp_atoms = next(iter(unique_pairs.values()))
    supporting = [code, *stamp_atoms]
    raw_value = _source_atom_text(code)
    bbox = _bbox_union(supporting)
    evidence_ids = tuple(
        dict.fromkeys(str(atom.get("id") or "") for atom in supporting if str(atom.get("id") or ""))
    )
    source_kind = _context_source_kind(supporting)
    stamp_name = "".join(_source_atom_text(atom) for atom in stamp_atoms)
    reconciliation_stamp = _compact(stamp_name) == _compact("对账专用章")
    return _HeaderFact(
        field_key="seal_code",
        raw_name=stamp_name,
        raw_value=raw_value,
        normalized_value=raw_value,
        page=page,
        page_id=str(code.get("page_id") or f"page:{page:04d}"),
        bbox=bbox,
        evidence_ids=evidence_ids,
        source_kind=source_kind,
        derivation=(
            "seal_code_adjacent_to_explicit_reconciliation_stamp"
            if reconciliation_stamp
            else "seal_code_adjacent_to_explicit_business_stamp"
        ),
        source_detail={
            "page": page,
            "stamp_texts": [_source_atom_text(atom) for atom in stamp_atoms],
            "raw_value": raw_value,
            "value_evidence_id": str(code.get("id") or ""),
            "evidence_ids": list(evidence_ids),
            "bbox": list(bbox) if bbox else None,
            "source": source_kind,
            "normalized_only": False,
        },
    )


def _seal_issuer_facts(atoms: Sequence[dict[str, Any]], page: int) -> list[_HeaderFact]:
    """Recover an issuer/branch only from text tightly attached to a seal."""

    candidates: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {
        "issuing_bank": [],
        "issuing_branch": [],
    }
    for stamp_atoms in _reconciliation_stamp_groups(atoms):
        stamp_bbox = _bbox_union(stamp_atoms)
        if stamp_bbox is None:
            continue
        for atom in atoms:
            if atom in stamp_atoms:
                continue
            bbox = atom.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            text = _nfkc(atom.get("text")).strip()
            field_key = ""
            if re.fullmatch(r"[\u4e00-\u9fff·]{2,30}(?:支行|分行|营业部|信用社|信用联社)", text):
                field_key = "issuing_branch"
            elif re.fullmatch(r"[\u4e00-\u9fff·]{2,30}(?:银行股份有限公司|银行|信用联社)", text):
                field_key = "issuing_bank"
            if not field_key:
                continue
            vertical_gap = float(stamp_bbox[1]) - float(bbox[3])
            horizontal_overlap = min(float(bbox[2]), stamp_bbox[2]) - max(float(bbox[0]), stamp_bbox[0])
            value_width = max(float(bbox[2]) - float(bbox[0]), 1.0)
            if 0.0 <= vertical_gap <= 30.0 and horizontal_overlap >= max(4.0, min(24.0, value_width * 0.25)):
                candidates[field_key].append((atom, stamp_atoms))

    facts: list[_HeaderFact] = []
    for field_key, matches in candidates.items():
        values = {_nfkc(atom.get("text")).strip() for atom, _stamp in matches}
        if len(values) != 1:
            continue
        value = next(iter(values))
        matching = [(atom, stamp) for atom, stamp in matches if _nfkc(atom.get("text")).strip() == value]
        if len(matching) != 1:
            continue
        atom, stamp_atoms = matching[0]
        supporting = [atom, *stamp_atoms]
        facts.append(
            _HeaderFact(
                field_key=field_key,
                raw_name="业务印章签发机构" if field_key == "issuing_branch" else "业务印章签发银行",
                raw_value=value,
                normalized_value=value,
                page=page,
                page_id=str(atom.get("page_id") or f"page:{page:04d}"),
                bbox=_bbox_union(supporting),
                evidence_ids=tuple(
                    dict.fromkeys(str(item.get("id") or "") for item in supporting if str(item.get("id") or ""))
                ),
                source_kind=_context_source_kind(supporting),
                derivation="issuer_text_adjacent_to_explicit_business_stamp",
                source_detail={
                    "page": page,
                    "stamp_texts": [_source_atom_text(item) for item in stamp_atoms],
                    "source": _context_source_kind(supporting),
                    "normalized_only": False,
                },
            )
        )
    return facts


def _physical_split_ledger_schema(headers: Sequence[str]) -> dict[str, Any] | None:
    compact_headers = [_compact(header) for header in headers]
    date_columns = [
        index
        for index, header in enumerate(compact_headers)
        if "日期" in header or header in {"交易时间", "入账时间", "记账时间", "timestamp"}
    ]
    debit_columns = [
        index
        for index, header in enumerate(compact_headers)
        if "借方" in header or header in {"支出", "支出金额", "付款金额", "debit"}
    ]
    credit_columns = [
        index
        for index, header in enumerate(compact_headers)
        if "贷方" in header or header in {"收入", "收入金额", "收款金额", "credit"}
    ]
    balance_columns = [
        index for index, header in enumerate(compact_headers) if "余额" in header or header == "balance"
    ]
    if not date_columns or len(debit_columns) != 1 or len(credit_columns) != 1 or len(balance_columns) != 1:
        return None
    return {
        "date_columns": date_columns,
        "debit_column": debit_columns[0],
        "credit_column": credit_columns[0],
        "balance_column": balance_columns[0],
    }


def _source_owned_data_row_index(
    row: Any,
    *,
    page_number: int,
    table_id: str,
) -> int | None:
    row_type = getattr(row, "row_type", None)
    row_type = getattr(row_type, "value", row_type)
    try:
        source_page = int(getattr(row, "source_page"))
        source_row_index = int(getattr(row, "source_row_index"))
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        str(row_type or "").casefold() != "data"
        or source_page != page_number
        or str(getattr(row, "source_physical_id", "") or "").strip() != table_id
        or source_row_index < 0
    ):
        return None
    return source_row_index


def _strict_cell_bbox(cell: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(cell, "bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values) or values[2] < values[0] or values[3] < values[1]:
        return None
    return values


def _strict_physical_row_bbox(cells: Sequence[Any]) -> tuple[float, float, float, float] | None:
    valid_boxes: list[tuple[float, float, float, float]] = []
    for cell in cells:
        box = _strict_cell_bbox(cell)
        if box is not None:
            valid_boxes.append(box)
            continue
        if str(getattr(cell, "text", "") or "").strip() or any(
            str(evidence_id or "").strip()
            for evidence_id in (getattr(cell, "evidence_ids", None) or [])
        ):
            return None
    if not valid_boxes:
        return None
    return (
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    )


def _strict_source_cell_ref(
    cell: Any,
    *,
    page_number: int,
    table_id: str,
    source_row_index: int,
    column: int,
) -> dict[str, Any] | None:
    refs = getattr(cell, "source_cell_refs", None)
    if not isinstance(refs, (list, tuple)) or len(refs) != 1 or not isinstance(refs[0], dict):
        return None
    ref = refs[0]
    try:
        ref_page = int(ref.get("page") or ref.get("source_page") or 0)
        ref_row = int(ref.get("row"))
        raw_row = int(ref.get("raw_row"))
        ref_column = int(ref.get("col"))
    except (TypeError, ValueError):
        return None
    ref_source = str(ref.get("source") or "").strip()
    if (
        ref_source not in {"", "canonical_physical_table"}
        or ref_page != page_number
        or str(ref.get("table_id") or "").strip() != table_id
        or ref_row != source_row_index
        or raw_row < 0
        or ref_column != column
    ):
        return None
    return deepcopy(ref)


def _strict_source_date(value: Any) -> str | None:
    raw_value = _nfkc(value)
    if not _DATE_TOKEN_RE.fullmatch(raw_value):
        return None
    normalized = _normalize_date(raw_value)
    try:
        calendar_date.fromisoformat(normalized)
    except ValueError:
        return None
    return normalized


def _strict_source_money(value: Any, *, nonnegative: bool) -> Decimal | None:
    raw_value = _nfkc(value)
    if not _MONEY_TOKEN_RE.fullmatch(raw_value):
        return None
    parsed = _as_decimal(raw_value)
    if parsed is None or (nonnegative and parsed < 0):
        return None
    return parsed


def _evidence_order(evidence_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", evidence_id)
    return (int(match.group(1)) if match else 2**31 - 1, evidence_id)


def _physical_cell_evidence_ids(
    cell: Any,
    raw_value: str,
    *,
    page_atoms: Sequence[dict[str, Any]],
    table_evidence_ids: set[str],
    allow_first_exact_match: bool,
    allow_rotated_exact_match: bool,
) -> tuple[str, ...]:
    """Bind a source-owned physical cell when rotated geometry lost its IDs."""

    declared = tuple(
        dict.fromkeys(
            str(evidence_id)
            for evidence_id in (getattr(cell, "evidence_ids", None) or [])
            if str(evidence_id or "").strip()
        )
    )
    if declared:
        return declared
    exact = [
        str(atom.get("id") or "")
        for atom in page_atoms
        if _compact(atom.get("text")) == _compact(raw_value)
        and (
            str(atom.get("id") or "") in table_evidence_ids
            or allow_rotated_exact_match
        )
    ]
    exact = list(dict.fromkeys(item for item in exact if item))
    if len(exact) == 1:
        return tuple(exact)
    if allow_first_exact_match and exact:
        return (min(exact, key=_evidence_order),)
    return ()


def _physical_brought_forward_facts(parse_result: Any) -> dict[int, list[_HeaderFact]]:
    """Recover source-owned carry balances from strict physical ledger rows."""

    carry_aliases = {_compact(alias) for alias in _FIELD_ALIASES["brought_forward_balance"]}
    facts_by_page: dict[int, list[_HeaderFact]] = defaultdict(list)
    page_atoms_by_page = _group_atoms(parse_result)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        if page_number <= 0:
            continue
        page_candidates: list[_HeaderFact] = []
        for table in getattr(page, "tables", None) or []:
            table_id = str(getattr(table, "table_id", "") or "").strip()
            metadata = getattr(table, "metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            if (
                not table_id
                or metadata.get("header_source") == "data_row"
                or metadata.get("preserve_headers") is False
            ):
                continue
            raw_headers = [str(header or "") for header in (getattr(table, "headers", None) or [])]
            schema = _physical_split_ledger_schema(raw_headers)
            if schema is None:
                continue
            date_columns = schema["date_columns"]
            debit_column = schema["debit_column"]
            credit_column = schema["credit_column"]
            balance_column = schema["balance_column"]
            table_evidence_ids = {
                str(evidence_id)
                for evidence_id in (getattr(table, "evidence_ids", None) or [])
                if str(evidence_id or "").strip()
            }
            transaction_facts: list[dict[str, Any]] = []
            carry_candidates: list[tuple[int, int, _HeaderFact]] = []
            for row in getattr(table, "rows", None) or []:
                source_row_index = _source_owned_data_row_index(
                    row,
                    page_number=page_number,
                    table_id=table_id,
                )
                if source_row_index is None:
                    continue
                cells = list(getattr(row, "cells", None) or [])
                if len(cells) != len(raw_headers):
                    continue
                values = [str(getattr(cell, "text", "") or "").strip() for cell in cells]
                balance_value = _strict_source_money(values[balance_column], nonnegative=False)
                transaction_fact = _physical_split_transaction_fact(
                    row,
                    page_number=page_number,
                    table_id=table_id,
                    headers=raw_headers,
                    schema=schema,
                    require_cell_evidence=False,
                )
                if transaction_fact is not None:
                    transaction_facts.append(transaction_fact)
                carry_columns = [
                    index for index, value in enumerate(values) if _compact(value) in carry_aliases
                ]
                if len(carry_columns) != 1 or balance_value is None:
                    continue
                carry_column = carry_columns[0]
                if carry_column in {*date_columns, debit_column, credit_column, balance_column} or any(
                    value for index, value in enumerate(values) if index not in {carry_column, balance_column}
                ):
                    continue
                supporting_cells = [cells[carry_column], cells[balance_column]]
                supporting_boxes = [_strict_cell_bbox(cell) for cell in supporting_cells]
                physical_row_bbox = _strict_physical_row_bbox(cells)
                rotated_logical_row = bool(
                    physical_row_bbox
                    and physical_row_bbox[3] - physical_row_bbox[1]
                    > max(50.0, (physical_row_bbox[2] - physical_row_bbox[0]) * 3.0)
                )
                page_atoms = page_atoms_by_page.get(page_number, [])
                supporting_id_groups = [
                    _physical_cell_evidence_ids(
                        cell,
                        values[column],
                        page_atoms=page_atoms,
                        table_evidence_ids=table_evidence_ids,
                        allow_first_exact_match=column == balance_column,
                        allow_rotated_exact_match=rotated_logical_row,
                    )
                    for cell, column in zip(
                        supporting_cells,
                        (carry_column, balance_column),
                        strict=True,
                    )
                ]
                supporting_refs = [
                    _strict_source_cell_ref(
                        cell,
                        page_number=page_number,
                        table_id=table_id,
                        source_row_index=source_row_index,
                        column=column,
                    )
                    for cell, column in zip(
                        supporting_cells,
                        (carry_column, balance_column),
                        strict=True,
                    )
                ]
                if (
                    any(ref is None for ref in supporting_refs)
                    or any(not evidence_ids for evidence_ids in supporting_id_groups)
                    or any(box is None for box in supporting_boxes)
                ):
                    continue
                raw_rows = {int(ref["raw_row"]) for ref in supporting_refs if ref is not None}
                if len(raw_rows) != 1:
                    continue
                raw_row = next(iter(raw_rows))
                valid_boxes = [box for box in supporting_boxes if box is not None]
                bbox = (
                    min(box[0] for box in valid_boxes),
                    min(box[1] for box in valid_boxes),
                    max(box[2] for box in valid_boxes),
                    max(box[3] for box in valid_boxes),
                )
                raw_value = values[balance_column]
                evidence_ids = tuple(
                    dict.fromkeys(evidence_id for group in supporting_id_groups for evidence_id in group)
                )
                source_ref = {
                    "source": "canonical_physical_table",
                    "source_page": page_number,
                    "page_range": [page_number, page_number],
                    "table_id": table_id,
                    "source_row_index": source_row_index,
                    "raw_row": raw_row,
                    "bbox": list(bbox),
                    "evidence_ids": list(evidence_ids),
                    "source_cell_refs": [ref for ref in supporting_refs if ref is not None],
                }
                carry_candidates.append(
                    (
                        source_row_index,
                        raw_row,
                        _HeaderFact(
                            field_key="brought_forward_balance",
                            raw_name=values[carry_column],
                            raw_value=raw_value,
                            normalized_value=_normalized_money(balance_value),
                            page=page_number,
                            page_id=f"page:{page_number:04d}",
                            bbox=bbox,
                            evidence_ids=evidence_ids,
                            source_kind="canonical_physical_table",
                            derivation="physical_brought_forward_row",
                            source_detail={
                                "page": page_number,
                                "raw_name": values[carry_column],
                                "raw_value": raw_value,
                                "source": "canonical_physical_table",
                                "source_ref": source_ref,
                                "normalized_only": False,
                            },
                        ),
                    )
                )
            if (
                len(carry_candidates) != 1
                or not transaction_facts
                or not _physical_row_order_is_consistent(transaction_facts)
            ):
                continue
            carry_row_index, carry_raw_row, carry_fact = carry_candidates[0]
            first_transaction = min(transaction_facts, key=lambda fact: int(fact["source_row_index"]))
            first_transaction_index = int(first_transaction["source_row_index"])
            first_transaction_raw_row = int(first_transaction["raw_row"])
            first_transaction_bbox = first_transaction["bbox"]
            carry_precedes_transaction_geometry = bool(
                carry_fact.bbox
                and (
                    carry_fact.bbox[3] <= first_transaction_bbox[1]
                    or carry_fact.bbox[2] <= first_transaction_bbox[0]
                    or carry_fact.bbox[0] >= first_transaction_bbox[2]
                )
            )
            if (
                carry_row_index >= first_transaction_index
                or carry_raw_row >= first_transaction_raw_row
                or carry_raw_row - carry_row_index != first_transaction_raw_row - first_transaction_index
                or not carry_precedes_transaction_geometry
            ):
                continue
            page_candidates.append(carry_fact)
        if len(page_candidates) == 1:
            facts_by_page[page_number].append(page_candidates[0])
    return dict(facts_by_page)


@memoize_bank_document_work
def _page_header_facts(parse_result: Any) -> tuple[dict[int, list[_HeaderFact]], dict[int, list[str]]]:
    facts_by_page: dict[int, list[_HeaderFact]] = {}
    lines_by_page: dict[int, list[str]] = {}
    physical_carry_facts = _physical_brought_forward_facts(parse_result)
    for page, atoms in sorted(_group_atoms(parse_result).items()):
        rows = _header_rows(atoms)
        lines = [" ".join(_nfkc(atom.get("text")) for atom in row if _nfkc(atom.get("text"))) for row in rows]
        lines_by_page[page] = [line for line in lines if line]
        facts = [
            *physical_carry_facts.get(page, []),
            *(fact for row in rows for fact in _facts_from_row(row, page)),
        ]
        header_bottom = max(
            (float(atom["bbox"][3]) for row in rows for atom in row),
            default=float("-inf"),
        )
        # Some statements place source business facts (printing metadata and
        # terminal cumulative totals) below the ledger.  Scan that positioned
        # footer plane. Explicit unknown label/value pairs are retained as
        # scoped business metadata; transaction rows and unlabelled prose stay
        # excluded by the row boundary above.
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
            if fact.field_key in _POSITIONED_FOOTER_FIELDS or fact.field_key.startswith("source_header_")
        )
        for positioned_rows in (rows, footer_rows):
            if disclaimer := _statement_disclaimer_fact(positioned_rows, page):
                facts.append(disclaimer)
            if cutoff := _statement_cutoff_timestamp_fact(positioned_rows, page):
                facts.append(cutoff)
        facts.extend(_seal_type_facts(atoms, page))
        if seal_code := _seal_code_fact(atoms, page):
            facts.append(seal_code)
        facts.extend(_seal_issuer_facts(atoms, page))
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

    embedded = extract_embedded_business_metadata(parse_result)
    for fact in embedded.facts:
        facts_by_page.setdefault(fact.page, []).append(
            _HeaderFact(
                field_key=fact.field_key,
                raw_name=fact.source_label,
                raw_value=fact.value,
                normalized_value=fact.value,
                page=fact.page,
                page_id=fact.page_id,
                bbox=fact.bbox,
                evidence_ids=tuple(
                    dict.fromkeys((fact.evidence_id, *fact.supporting_evidence_ids))
                ),
                source_kind="embedded_image_ocr",
                derivation="targeted_embedded_business_seal_ocr",
                source_detail={
                    "page": fact.page,
                    "confidence": fact.confidence,
                    "evidence_id": fact.evidence_id,
                    "source": "embedded_image_ocr",
                    "normalized_only": False,
                },
            )
        )
    for page, facts in list(facts_by_page.items()):
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


def statement_scope_count(parse_result: Any) -> int:
    """Return the independently resolved number of statement/account scopes."""

    if parse_result is None:
        return 0
    facts_by_page, _lines = _page_header_facts(parse_result)
    if not any(
        _fact_values(page_facts, key)
        for page_facts in facts_by_page.values()
        for key in _CONTEXT_SIGNATURE_FIELDS
    ):
        return 0
    for page_facts in facts_by_page.values():
        if any(len(_fact_values(page_facts, key)) > 1 for key in _CONTEXT_SIGNATURE_FIELDS):
            return 2
    return len(_context_page_groups(parse_result, facts_by_page))


def _identity_facts(identity_fields: dict[str, dict[str, Any]], page: int = 1) -> list[_HeaderFact]:
    facts: list[_HeaderFact] = []
    for field_key, detail in identity_fields.items():
        if field_key not in _FIELD_ALIASES and field_key not in {"statement_title", "document_date"}:
            continue
        mapping = detail if isinstance(detail, dict) else {"normalized_value": detail}
        if field_key == "query_period" and mapping.get("derivation") == "source_period_envelope":
            facts.extend(_source_period_envelope_facts(mapping, page))
            continue
        if field_key == "query_period" and mapping.get("derivation") == "source_year_month_period":
            facts.extend(_source_year_month_period_facts(mapping, page))
            continue
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


def _source_period_envelope_facts(mapping: dict[str, Any], default_page: int) -> list[_HeaderFact]:
    """Materialize exact source pairs that share one normalized min/max envelope."""
    components = [item for item in (mapping.get("source_components") or []) if isinstance(item, dict)]
    if not components or mapping.get("raw_value") not in (None, ""):
        return []
    normalized_components: list[tuple[str, str, int, dict[str, Any]]] = []
    for component in components:
        raw_start = str(component.get("raw_start") or "").strip()
        raw_end = str(component.get("raw_end") or "").strip()
        raw_start_name = str(component.get("raw_start_name") or "").strip()
        raw_end_name = str(component.get("raw_end_name") or "").strip()
        normalized_start = _valid_date_tokens(raw_start)
        normalized_end = _valid_date_tokens(raw_end)
        if len(normalized_start) != 1 or len(normalized_end) != 1 or normalized_start[0] > normalized_end[0]:
            return []
        component_mapping = {
            "source": component.get("source") or mapping.get("source"),
            "source_refs": component.get("source_refs") or mapping.get("source_refs") or [],
            "evidence_ids": component.get("evidence_ids") or [],
        }
        if (
            str(component_mapping["source"] or "").casefold() == "canonical_evidence_atoms"
            and not component_mapping["evidence_ids"]
        ):
            return []
        same_period_label = raw_start_name == raw_end_name and _identity_detail_is_source_bound(
            "query_period",
            raw_start_name,
            f"{raw_start} {raw_end}",
            component_mapping,
        )
        separate_bounds = (
            _identity_detail_is_source_bound("period_start", raw_start_name, raw_start, component_mapping)
            and _identity_detail_is_source_bound("period_end", raw_end_name, raw_end, component_mapping)
        )
        if not same_period_label and not separate_bounds:
            return []
        try:
            source_page = int(
                component.get("source_page")
                or component.get("page")
                or _page_number(component.get("page_id"))
                or default_page
            )
        except (TypeError, ValueError):
            return []
        if source_page < 1:
            return []
        normalized_components.append((normalized_start[0], normalized_end[0], source_page, component))
    for previous, current in zip(normalized_components, normalized_components[1:]):
        previous_end = calendar_date.fromisoformat(previous[1])
        current_start = calendar_date.fromisoformat(current[0])
        if current[2] <= previous[2] or current_start.toordinal() != previous_end.toordinal() + 1:
            return []
    expected_envelope = (
        f"{min(start for start, _end, _page, _component in normalized_components)} ~ "
        f"{max(end for _start, end, _page, _component in normalized_components)}"
    )
    if _normalize_field_value("query_period", mapping.get("normalized_value")) != expected_envelope:
        return []
    facts: list[_HeaderFact] = []
    for _start, _end, source_page, component in normalized_components:
        raw_start = str(component["raw_start"]).strip()
        raw_end = str(component["raw_end"]).strip()
        source_detail = {
            "page": source_page,
            "page_id": str(component.get("page_id") or f"page:{source_page:04d}"),
            "raw_name": str(component.get("raw_name") or "source period pair"),
            "raw_start_name": str(component["raw_start_name"]),
            "raw_start": raw_start,
            "raw_end_name": str(component["raw_end_name"]),
            "raw_end": raw_end,
            "evidence_ids": list(dict.fromkeys(str(item) for item in component.get("evidence_ids") or [] if str(item))),
            "source": str(component.get("source") or mapping.get("source") or "identity_fields"),
        }
        facts.append(
            _HeaderFact(
                field_key="query_period",
                raw_name=source_detail["raw_name"],
                raw_value=f"{raw_start} {raw_end}",
                normalized_value=f"{_start} ~ {_end}",
                page=source_page,
                page_id=source_detail["page_id"],
                bbox=None,
                evidence_ids=tuple(source_detail["evidence_ids"]),
                source_kind=source_detail["source"],
                derivation="source_period_envelope",
                source_detail=source_detail,
            )
        )
    return facts


def _source_period_mapping_for_pages(
    mapping: dict[str, Any],
    pages: Sequence[int],
) -> dict[str, Any] | None:
    """Restrict a normalized-only period envelope to one resolved statement scope."""
    page_set = {int(page) for page in pages if int(page) > 0}
    components = []
    normalized_bounds: list[tuple[str, str]] = []
    for component in mapping.get("source_components") or []:
        if not isinstance(component, dict):
            continue
        try:
            source_page = int(
                component.get("source_page")
                or component.get("page")
                or _page_number(component.get("page_id"))
            )
        except (TypeError, ValueError):
            continue
        if source_page not in page_set:
            continue
        components.append(deepcopy(component))
        starts = _valid_date_tokens(component.get("raw_start"))
        ends = _valid_date_tokens(component.get("raw_end"))
        if len(starts) == 1 and len(ends) == 1:
            normalized_bounds.append((starts[0], ends[0]))
    if not components:
        return None
    scoped = deepcopy(mapping)
    scoped["source_components"] = components
    if len(normalized_bounds) == len(components):
        scoped["normalized_value"] = (
            f"{min(start for start, _end in normalized_bounds)} ~ "
            f"{max(end for _start, end in normalized_bounds)}"
        )
    refs: list[dict[str, Any]] = []
    for ref in mapping.get("source_refs") or []:
        if not isinstance(ref, dict):
            continue
        try:
            ref_page = int(
                ref.get("source_page") or ref.get("page") or _page_number(ref.get("page_id")) or 0
            )
        except (TypeError, ValueError):
            continue
        if ref_page in page_set:
            refs.append(deepcopy(ref))
    if mapping.get("source_refs") is not None:
        scoped["source_refs"] = refs
    return scoped


def _source_year_month_period_facts(mapping: dict[str, Any], default_page: int) -> list[_HeaderFact]:
    """Retain exact year/month KVs that support a normalized calendar period."""
    components = [item for item in (mapping.get("source_components") or []) if isinstance(item, dict)]
    if not components or mapping.get("raw_value") not in (None, ""):
        return []
    resolved: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for component in components:
        raw_year_name = str(component.get("raw_year_name") or "").strip()
        raw_year = str(component.get("raw_year") or "").strip()
        raw_month_name = str(component.get("raw_month_name") or "").strip()
        raw_month = str(component.get("raw_month") or "").strip()
        component_mapping = {
            "source": component.get("source") or mapping.get("source"),
            "source_refs": component.get("source_refs") or mapping.get("source_refs") or [],
            "evidence_ids": component.get("evidence_ids") or [],
        }
        if (
            str(component_mapping["source"] or "").casefold() == "canonical_evidence_atoms"
            and not component_mapping["evidence_ids"]
        ):
            return []
        if not _identity_detail_is_source_bound(
            "statement_year", raw_year_name, raw_year, component_mapping
        ) or not _identity_detail_is_source_bound(
            "statement_month_number", raw_month_name, raw_month, component_mapping
        ):
            return []
        if not re.fullmatch(r"20\d{2}", raw_year) or not re.fullmatch(r"\d{1,2}", raw_month):
            return []
        year = int(raw_year)
        month = int(raw_month)
        if not 1 <= month <= 12:
            return []
        resolved.append((year, month, component, component_mapping))
    starts = [f"{year:04d}-{month:02d}-01" for year, month, _component, _mapping in resolved]
    ends = [
        f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
        for year, month, _component, _mapping in resolved
    ]
    expected_envelope = f"{min(starts)} ~ {max(ends)}"
    if _normalize_field_value("query_period", mapping.get("normalized_value")) != expected_envelope:
        return []
    facts: list[_HeaderFact] = []
    for year, month, component, component_mapping in resolved:
        source_page = int(
            component.get("source_page")
            or component.get("page")
            or _page_number(component.get("page_id"))
            or default_page
        )
        page_id = str(component.get("page_id") or f"page:{source_page:04d}")
        evidence_ids = tuple(
            dict.fromkeys(str(item) for item in component.get("evidence_ids") or [] if str(item))
        )
        source_kind = str(component_mapping.get("source") or "identity_fields")
        for field_key, raw_name, raw_value, normalized_value in (
            ("statement_year", str(component["raw_year_name"]), str(component["raw_year"]), year),
            (
                "statement_month_number",
                str(component["raw_month_name"]),
                str(component["raw_month"]),
                month,
            ),
        ):
            facts.append(
                _HeaderFact(
                    field_key=field_key,
                    raw_name=raw_name,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    page=source_page,
                    page_id=page_id,
                    bbox=None,
                    evidence_ids=evidence_ids,
                    source_kind=source_kind,
                    derivation="source_year_month_period",
                    source_detail={
                        "page": source_page,
                        "page_id": page_id,
                        "raw_name": raw_name,
                        "raw_value": raw_value,
                        "evidence_ids": list(evidence_ids),
                        "source": source_kind,
                    },
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
    resolved_field = _field_for_label(raw_name)
    range_reclassified = (
        field_key == "query_period"
        and _compact(raw_name) == _compact("交易时间")
        and _period_dates_are_valid(raw_value, minimum=2)
    )
    if resolved_field != field_key and not range_reclassified:
        return False
    return _labelled_fact_value_is_plausible(field_key, raw_name, raw_value)


def _field_source(facts: Sequence[_HeaderFact]) -> dict[str, Any]:
    refs = []
    evidence_ids: list[str] = []
    for fact in facts:
        ref: dict[str, Any] = {"source": fact.source_kind, "source_page": fact.page}
        explicit_ref = (fact.source_detail or {}).get("source_ref")
        if isinstance(explicit_ref, dict):
            ref.update(deepcopy(explicit_ref))
            ref["source"] = fact.source_kind
            ref["source_page"] = fact.page
        if fact.bbox:
            ref["bbox"] = list(fact.bbox)
        refs.append(ref)
        evidence_ids.extend(fact.evidence_ids)
    first = facts[0]
    detail = {
        "raw_name": first.raw_name,
        "source": first.source_kind,
        "source_refs": list({repr(ref): ref for ref in refs}.values()),
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
    }
    derivations = list(dict.fromkeys(fact.derivation for fact in facts if fact.derivation))
    if derivations:
        detail["derivation"] = derivations[0] if len(derivations) == 1 else derivations
        detail["normalized_only"] = all(
            bool((fact.source_detail or {}).get("normalized_only", True))
            for fact in facts
            if fact.derivation
        )
        detail["components"] = [dict(fact.source_detail) for fact in facts if fact.source_detail]
        detail["component_count"] = len(detail["components"])
    return detail


def _raw_header_map(facts: Sequence[_HeaderFact]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
    for fact in facts:
        if fact.derivation == "source_period_envelope" and fact.source_detail:
            for name_key, value_key in (
                ("raw_start_name", "raw_start"),
                ("raw_end_name", "raw_end"),
            ):
                raw_name = _nfkc(fact.source_detail.get(name_key))
                raw_value = _nfkc(fact.source_detail.get(value_key))
                if raw_name and raw_value:
                    grouped[raw_name].append((fact.page, raw_value, True))
            continue
        raw_value = str(fact.raw_value).strip() if fact.field_key == "source_header_page_label" else _nfkc(fact.raw_value)
        grouped[fact.raw_name].append((fact.page, raw_value, False))
    out: dict[str, Any] = {}
    for raw_name, raw_entries in grouped.items():
        entries = list(dict.fromkeys(entry for entry in raw_entries if entry[1]))
        values = list(dict.fromkeys(value for _page, value, _force_array in entries))
        preserve_page_array = len(entries) > 1 and (
            any(force_array for _page, _value, force_array in entries)
            or any(_compact(raw_name) in aliases for aliases in _PAGE_LOCAL_DIRECTION_LABELS.values())
        )
        if preserve_page_array:
            out[raw_name] = [{"page": page, "value": value} for page, value, _force_array in entries]
        elif len(values) == 1:
            out[raw_name] = values[0]
        elif values:
            out[raw_name] = [{"page": page, "value": value} for page, value, _force_array in entries]
    return out


def _explicit_page_direction_aggregates(
    facts: Sequence[_HeaderFact],
    pages: Sequence[int],
) -> dict[str, tuple[Any, list[_HeaderFact]]]:
    """Sum a complete source-explicit totals pair or quartet from every scope page."""

    scope_pages = sorted({int(page) for page in pages if int(page) > 0})
    if not scope_pages or scope_pages != list(range(scope_pages[0], scope_pages[-1] + 1)):
        return {}
    direction_fields = set(_PAGE_LOCAL_DIRECTION_LABELS)
    direction_facts = [fact for fact in facts if fact.field_key in direction_fields]
    if any(
        _compact(fact.raw_name) not in _PAGE_LOCAL_DIRECTION_LABELS[fact.field_key]
        or fact.page not in scope_pages
        for fact in direction_facts
    ):
        return {}
    by_field_page: dict[str, dict[int, list[_HeaderFact]]] = {
        field_key: defaultdict(list) for field_key in direction_fields
    }
    for fact in direction_facts:
        by_field_page[fact.field_key][fact.page].append(fact)
    total_fields = ("debit_total", "credit_total")
    count_fields = ("debit_count", "credit_count")
    if any(
        len(by_field_page[field_key].get(page, [])) != 1
        for field_key in total_fields
        for page in scope_pages
    ):
        return {}
    count_fact_count = sum(
        len(by_field_page[field_key].get(page, []))
        for field_key in count_fields
        for page in scope_pages
    )
    include_counts = count_fact_count > 0
    if include_counts and any(
        len(by_field_page[field_key].get(page, [])) != 1
        for field_key in count_fields
        for page in scope_pages
    ):
        return {}
    aggregate_fields = (*count_fields, *total_fields) if include_counts else total_fields
    components = {
        field_key: [by_field_page[field_key][page][0] for page in scope_pages]
        for field_key in aggregate_fields
    }
    count_values: dict[str, list[int]] = {}
    if include_counts:
        for field_key in count_fields:
            parsed = [_as_exact_int(fact.normalized_value) for fact in components[field_key]]
            if any(value is None or value < 0 for value in parsed):
                return {}
            count_values[field_key] = [int(value) for value in parsed if value is not None]
    money_values: dict[str, list[Decimal]] = {}
    for field_key in total_fields:
        parsed = [_as_decimal(fact.normalized_value) for fact in components[field_key]]
        if any(value is None for value in parsed):
            return {}
        money_values[field_key] = [value for value in parsed if value is not None]
    debit_values = money_values["debit_total"]
    if any(value > 0 for value in debit_values) and any(value < 0 for value in debit_values):
        return {}
    if any(value < 0 for value in money_values["credit_total"]):
        return {}
    aggregates: dict[str, tuple[Any, list[_HeaderFact]]] = {}
    if include_counts:
        aggregates.update(
            {
                "debit_count": (sum(count_values["debit_count"]), components["debit_count"]),
                "credit_count": (sum(count_values["credit_count"]), components["credit_count"]),
            }
        )
    aggregates.update(
        {
            "debit_total": (
                _normalized_money(abs(sum(debit_values, Decimal("0")))),
                components["debit_total"],
            ),
            "credit_total": (
                _normalized_money(sum(money_values["credit_total"], Decimal("0"))),
                components["credit_total"],
            ),
        }
    )
    return aggregates


def _explicit_page_aggregate_source(field_key: str, facts: Sequence[_HeaderFact]) -> dict[str, Any]:
    source = _field_source(facts)
    source.update(
        {
            "source": "derived_explicit_page_aggregate",
            "derivation": "sum_explicit_page_totals",
            "components": [
                {
                    "page": fact.page,
                    "raw_name": fact.raw_name,
                    "raw_value": fact.raw_value,
                    "normalized_value": fact.normalized_value,
                    "bbox": list(fact.bbox) if fact.bbox else None,
                    "evidence_ids": list(fact.evidence_ids),
                    "source": fact.source_kind,
                }
                for fact in facts
            ],
        }
    )
    if field_key == "debit_total":
        debit_values = [_as_decimal(fact.normalized_value) for fact in facts]
        source["sign_normalization"] = (
            "magnitude_from_nonpositive_expense_page_totals"
            if any(value is not None and value < 0 for value in debit_values)
            else "magnitude_from_nonnegative_expense_page_totals"
        )
    return source


def _record_from_facts(
    facts: Sequence[_HeaderFact],
    pages: Sequence[int],
    index: int,
    *,
    allow_page_direction_aggregates: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[_HeaderFact]] = defaultdict(list)
    for fact in facts:
        grouped[fact.field_key].append(fact)
    normalized: dict[str, Any] = {}
    canonical_raw: dict[str, Any] = {}
    field_sources: dict[str, Any] = {}
    page_direction_aggregates = (
        _explicit_page_direction_aggregates(facts, pages) if allow_page_direction_aggregates else {}
    )
    has_page_local_direction_facts = any(
        fact.field_key in _PAGE_LOCAL_DIRECTION_LABELS
        and _compact(fact.raw_name) in _PAGE_LOCAL_DIRECTION_LABELS[fact.field_key]
        for fact in facts
    )
    for field_key, field_facts in grouped.items():
        if field_key.startswith("source_header_"):
            continue
        if field_key in page_direction_aggregates:
            aggregate, component_facts = page_direction_aggregates[field_key]
            normalized[field_key] = aggregate
            field_sources[field_key] = _explicit_page_aggregate_source(field_key, component_facts)
            continue
        if has_page_local_direction_facts and field_key in _PAGE_LOCAL_DIRECTION_LABELS:
            continue
        if field_key == "query_period" and field_facts and all(
            fact.derivation == "source_period_envelope" for fact in field_facts
        ):
            component_starts = [
                _valid_date_tokens((fact.source_detail or {}).get("raw_start")) for fact in field_facts
            ]
            component_ends = [
                _valid_date_tokens((fact.source_detail or {}).get("raw_end")) for fact in field_facts
            ]
            if all(len(values) == 1 for values in (*component_starts, *component_ends)):
                normalized[field_key] = (
                    f"{min(values[0] for values in component_starts)} ~ "
                    f"{max(values[0] for values in component_ends)}"
                )
                field_sources[field_key] = _field_source(field_facts)
            continue
        selected_facts = list(field_facts)
        if field_key in {"brought_forward_balance", "statement_title"}:
            first_page = min(fact.page for fact in field_facts)
            selected_facts = [fact for fact in field_facts if fact.page == first_page]
        elif field_key in {"statement_cutoff_date", "statement_cutoff_timestamp"}:
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
        if not all(fact.derivation == "source_period_envelope" for fact in selected_facts):
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
            supporting = [fact for key in ("statement_year", "statement_month_number") for fact in grouped.get(key, [])]
            if supporting:
                month_source = _field_source(supporting)
                month_source.update(
                    {
                        "derivation": "year_month_calendar",
                        "normalized_only": True,
                    }
                )
                field_sources["statement_month"] = month_source
            if not any(
                len(_DATE_TOKEN_RE.findall(str(normalized.get(key) or ""))) >= 2
                for key in ("query_period", "statement_period", "query_date")
            ) and not ({"period_start", "period_end"} & normalized.keys()):
                period_start = f"{year:04d}-{month:02d}-01"
                period_end = f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
                normalized["period_start"] = period_start
                normalized["period_end"] = period_end
                normalized["query_period"] = f"{period_start} ~ {period_end}"
                if supporting:
                    period_source = _field_source(supporting)
                    period_source.update(
                        {
                            "derivation": "year_month_calendar",
                            "normalized_only": True,
                        }
                    )
                    for field_key in ("period_start", "period_end", "query_period"):
                        field_sources[field_key] = dict(period_source)
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
        if not raw_period_dates:
            raw_period_dates = [
                raw_date
                for fact in grouped.get(period_source_key, [])
                for raw_date in _DATE_TOKEN_RE.findall(fact.raw_value)
            ]
        if len(dates) >= 2:
            for key, value in (("period_start", dates[0]), ("period_end", dates[1])):
                normalized.setdefault(key, value)
                matching_raw = next(
                    (raw_date for raw_date in raw_period_dates if _normalize_date(raw_date) == value),
                    "",
                )
                if matching_raw:
                    canonical_raw.setdefault(key, matching_raw)
                matching_fact = next(
                    (
                        fact
                        for fact in grouped.get(period_source_key, [])
                        if any(_normalize_date(raw_date) == value for raw_date in _DATE_TOKEN_RE.findall(fact.raw_value))
                    ),
                    None,
                )
                if matching_fact is not None:
                    field_sources.setdefault(key, _field_source([matching_fact]))
                else:
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
        "confidence": (
            1.0
            if any(
                fact.source_kind == "canonical_evidence_atoms" and (fact.evidence_ids or fact.bbox)
                for fact in facts
            )
            else 0.85
        ),
    }


def _has_complete_source_page_selection(parse_result: Any) -> bool:
    parser_info = getattr(parse_result, "parser_info", None)
    options = getattr(parser_info, "options", None)
    options = options if isinstance(options, dict) else {}
    try:
        source_page_count = int(options.get("source_page_count") or 0)
    except (TypeError, ValueError):
        source_page_count = 0
    selected_pages = sorted(
        {
            int(page)
            for page in (options.get("selected_source_pages") or [])
            if str(page).isdigit() and int(page) > 0
        }
    )
    parsed_pages = sorted(
        {
            int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
            for page in (getattr(parse_result, "pages", None) or [])
            if int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0) > 0
        }
    )
    if source_page_count > 0:
        expected = list(range(1, source_page_count + 1))
        return parsed_pages == expected and (not selected_pages or selected_pages == expected)
    if selected_pages:
        return False
    try:
        parser_page_count = int(getattr(parser_info, "page_count", 0) or 0)
    except (TypeError, ValueError):
        parser_page_count = 0
    return parser_page_count > 0 and parsed_pages == list(range(1, parser_page_count + 1))


def build_statement_header_records(
    parse_result: Any,
    identity_fields: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return one source-preserving header row per resolved statement scope."""
    facts_by_page, _lines = _page_header_facts(parse_result)
    groups = _context_page_groups(parse_result, facts_by_page)
    identity_mapping = dict(identity_fields or {})
    global_identity = _identity_facts(identity_mapping)
    source_period_mapping = identity_mapping.get("query_period")
    has_source_period_envelope = bool(
        isinstance(source_period_mapping, dict)
        and source_period_mapping.get("derivation") == "source_period_envelope"
    )
    stable_cross_scope = _stable_cross_scope_facts(facts_by_page, groups)
    allow_page_direction_aggregates = _has_complete_source_page_selection(parse_result)
    records: list[dict[str, Any]] = []
    for index, pages in enumerate(groups, start=1):
        facts = [fact for page in pages for fact in facts_by_page.get(page, [])]
        scoped_identity = global_identity
        if has_source_period_envelope:
            scoped_identity = [fact for fact in global_identity if fact.field_key != "query_period"]
            scoped_mapping = _source_period_mapping_for_pages(source_period_mapping, pages)
            if scoped_mapping is not None:
                scoped_identity.extend(_source_period_envelope_facts(scoped_mapping, pages[0]))
        # Stable identity fields apply to every statement scope.  Period and
        # declared-count fields are document-global only when there is exactly
        # one scope; otherwise a segment must carry its own source fact.
        for fact in scoped_identity:
            if len(groups) > 1 and fact.page not in pages:
                continue
            existing_field_facts = [item for item in facts if item.field_key == fact.field_key]
            if not existing_field_facts or (
                fact.derivation == "source_period_envelope"
                and all(item.derivation == "source_period_envelope" for item in existing_field_facts)
            ):
                facts.append(fact)
        for field_key, stable_facts in stable_cross_scope.items():
            if not _fact_values(facts, field_key):
                facts.extend(stable_facts)
        if not facts:
            continue
        records.append(
            _record_from_facts(
                facts,
                pages,
                index,
                allow_page_direction_aggregates=allow_page_direction_aggregates,
            )
        )
    if not records and global_identity:
        page_count = max(1, len(getattr(parse_result, "pages", None) or []))
        records.append(_record_from_facts(global_identity, list(range(1, page_count + 1)), 1))
    return records


def _header_record_for_page(
    page: int,
    header_records: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for record in header_records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        page_range = source.get("page_range") if isinstance(source.get("page_range"), (list, tuple)) else []
        if len(page_range) < 2:
            continue
        try:
            if int(page_range[0]) <= page <= int(page_range[1]):
                matches.append(record)
        except (TypeError, ValueError):
            continue
    return matches[0] if len(matches) == 1 else None


def _fact_is_represented_by_header(
    fact: _HeaderFact,
    header_records: Sequence[dict[str, Any]],
) -> bool:
    header = _header_record_for_page(fact.page, header_records)
    if header is None:
        return False
    normalized = header.get("normalized") if isinstance(header.get("normalized"), dict) else {}
    expected = fact.normalized_value
    is_disclaimer_fact = fact.field_key == "statement_disclaimer" or _is_statement_disclaimer(
        f"{fact.raw_name}:{fact.raw_value}"
    )
    if is_disclaimer_fact:
        # Positioned text extractors can expose both the first physical line
        # and the reconstructed complete disclaimer.  Once the complete header
        # fact contains that exact normalized fragment, emitting the fragment
        # again as generic source metadata is redundant rather than lossless.
        observed_compact = _compact(normalized.get("statement_disclaimer"))
        expected_compact = _compact(expected)
        if expected_compact and expected_compact in observed_compact:
            return True
    if fact.field_key.startswith("source_header_"):
        return False
    observed = normalized.get(fact.field_key)
    if isinstance(observed, (int, float)) or isinstance(expected, (int, float)):
        return observed == expected
    return _nfkc(observed) == _nfkc(expected) and bool(_nfkc(expected))


def _metadata_scope(page_numbers: Sequence[int], all_pages: Sequence[int]) -> str:
    pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    document_pages = sorted({int(page) for page in all_pages if int(page) > 0})
    if pages and pages == document_pages:
        return "document"
    return "page" if len(pages) == 1 else "pages"


def _contiguous_fact_runs(facts: Sequence[_HeaderFact]) -> list[list[_HeaderFact]]:
    """Split one repeated fact into exact contiguous source-page ranges."""

    ordered = sorted(facts, key=lambda fact: fact.page)
    runs: list[list[_HeaderFact]] = []
    for fact in ordered:
        if not runs or fact.page > runs[-1][-1].page + 1:
            runs.append([fact])
        else:
            runs[-1].append(fact)
    return runs


def build_source_metadata_records(
    parse_result: Any,
    header_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return lossless, scoped metadata facts not representable by one header row.

    The dataset is deliberately long-form: every value is atomic and every
    source-page occurrence is explicit.  Stable facts already represented by
    ``statement_header`` are omitted to avoid duplicating consumer data.
    """

    facts_by_page, _lines = _page_header_facts(parse_result)
    facts = [fact for page in sorted(facts_by_page) for fact in facts_by_page[page]]
    all_pages = sorted(
        set(facts_by_page)
        | {
            int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
            for page in (getattr(parse_result, "pages", None) or [])
            if int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0) > 0
        }
    )
    raw_values_by_field: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        raw_values_by_field[fact.field_key].add(_nfkc(fact.raw_value))

    groups: dict[tuple[str, str, str, str, int], list[_HeaderFact]] = defaultdict(list)
    for fact in facts:
        page_local = (
            fact.field_key == "source_header_page_label"
            or fact.field_key in _PAGE_LOCAL_DIRECTION_LABELS
            or len(raw_values_by_field[fact.field_key]) > 1
        )
        if not page_local and _fact_is_represented_by_header(fact, header_records):
            continue
        normalized_signature = repr(fact.normalized_value)
        groups[
            (
                fact.field_key,
                _nfkc(fact.raw_name),
                _nfkc(fact.raw_value),
                normalized_signature,
                fact.page if page_local else 0,
            )
        ].append(fact)

    records: list[dict[str, Any]] = []
    grouped_runs = [
        (key, run)
        for key, grouped_facts in groups.items()
        for run in _contiguous_fact_runs(grouped_facts)
    ]
    for index, ((_field_key, raw_name, raw_value, _signature, _page_scope), grouped_facts) in enumerate(
        grouped_runs, start=1
    ):
        first = grouped_facts[0]
        pages = sorted({fact.page for fact in grouped_facts})
        normalized: dict[str, Any] = {
            "metadata_field": (
                "page_label"
                if first.field_key == "source_header_page_label"
                else "other"
                if first.field_key.startswith("source_header_")
                else first.field_key
            ),
            "metadata_name": raw_name,
            "metadata_value": raw_value,
            "source_page_start": min(pages),
            "source_page_end": max(pages),
            "scope": _metadata_scope(pages, all_pages),
        }
        normalized_value = first.normalized_value
        normalized["normalized_value"] = (
            normalized_value if normalized_value not in (None, "") else raw_value
        )
        source_refs = [
            {
                "source": fact.source_kind,
                "source_page": fact.page,
                "page_range": [fact.page, fact.page],
                **({"bbox": list(fact.bbox)} if fact.bbox else {}),
                **({"evidence_ids": list(fact.evidence_ids)} if fact.evidence_ids else {}),
            }
            for fact in grouped_facts
        ]
        records.append(
            {
                "record_id": f"source_metadata:r{index:06d}",
                "normalized": normalized,
                "canonical_raw": {"metadata_name": raw_name, "metadata_value": raw_value},
                "raw": {"metadata_name": raw_name, "metadata_value": raw_value},
                "source": {
                    "source": "source_business_metadata_fact",
                    "page_range": [min(pages), max(pages)],
                    "source_refs": source_refs,
                },
                "confidence": min(
                    1.0 if fact.source_kind == "canonical_evidence_atoms" else 0.95
                    for fact in grouped_facts
                ),
            }
        )
    return records


def audit_source_fact_conservation(
    parse_result: Any,
    header_records: Sequence[dict[str, Any]],
    metadata_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Check that every recovered source metadata fact reaches a business dataset."""

    facts_by_page, _lines = _page_header_facts(parse_result)
    facts = [fact for page in sorted(facts_by_page) for fact in facts_by_page[page]]
    metadata_index: set[tuple[str, str, int]] = set()
    for record in metadata_records:
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        name = _nfkc(normalized.get("metadata_name"))
        value = _nfkc(normalized.get("metadata_value"))
        try:
            page_start = int(normalized.get("source_page_start") or 0)
            page_end = int(normalized.get("source_page_end") or page_start)
        except (TypeError, ValueError):
            page_start = page_end = 0
        if page_start > 0 and page_end >= page_start:
            for page in range(page_start, page_end + 1):
                metadata_index.add((name, value, page))
    uncovered = [
        fact
        for fact in facts
        if not _fact_is_represented_by_header(fact, header_records)
        and (_nfkc(fact.raw_name), _nfkc(fact.raw_value), fact.page) not in metadata_index
    ]
    embedded = extract_embedded_business_metadata(parse_result)
    return {
        "source_fact_count": len(facts),
        "represented_fact_count": len(facts) - len(uncovered),
        "unrepresented_fact_count": len(uncovered),
        "unrepresented_fields": sorted({fact.field_key for fact in uncovered}),
        "embedded_candidate_images": embedded.candidate_images,
        "embedded_ocr_images": embedded.ocr_images,
        "embedded_ocr_status": embedded.status,
    }


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


def _canonical_physical_lineage_key(
    source: dict[str, Any],
    *,
    required_columns: set[int] | None = None,
    required_raw_row: int | None = None,
) -> tuple[int, str, int] | None:
    """Return one exact physical-row key only when its source lineage is complete."""

    if str(source.get("source") or "") != "canonical_physical_table":
        return None
    try:
        page = int(source.get("source_page") or 0)
        row_index = int(source.get("source_row_index"))
    except (TypeError, ValueError):
        return None
    table_id = str(source.get("table_id") or "").strip()
    page_range = source.get("page_range")
    bbox = source.get("bbox")
    evidence_ids = source.get("evidence_ids")
    refs = source.get("source_cell_refs")
    if (
        page <= 0
        or row_index < 0
        or not table_id
        or list(page_range or []) != [page, page]
        or not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not isinstance(evidence_ids, (list, tuple))
        or not evidence_ids
        or not all(str(item or "").strip() for item in (evidence_ids or []))
        or not isinstance(refs, list)
        or not refs
    ):
        return None
    try:
        numeric_bbox = [float(value) for value in bbox]
        if (
            not all(math.isfinite(value) for value in numeric_bbox)
            or numeric_bbox[2] < numeric_bbox[0]
            or numeric_bbox[3] < numeric_bbox[1]
        ):
            return None
    except (TypeError, ValueError):
        return None

    columns: set[int] = set()
    raw_rows: set[int] = set()
    for ref in refs:
        if not isinstance(ref, dict):
            return None
        try:
            ref_page = int(ref.get("page") or ref.get("source_page") or 0)
            ref_row = int(ref.get("row"))
            raw_row = int(ref.get("raw_row"))
            column = int(ref.get("col"))
        except (TypeError, ValueError):
            return None
        ref_source = str(ref.get("source") or "").strip()
        if (
            ref_source not in {"", "canonical_physical_table"}
            or ref_page != page
            or str(ref.get("table_id") or "").strip() != table_id
            or ref_row != row_index
            or raw_row < 0
            or column < 0
        ):
            return None
        raw_rows.add(raw_row)
        columns.add(column)
    if (
        len(raw_rows) != 1
        or len(columns) != len(refs)
        or len(columns) < 3
        or (required_columns and not required_columns <= columns)
        or (required_raw_row is not None and raw_rows != {required_raw_row})
    ):
        return None
    return page, table_id, row_index


def _physical_split_transaction_fact(
    row: Any,
    *,
    page_number: int,
    table_id: str,
    headers: Sequence[str],
    schema: dict[str, Any],
    require_cell_evidence: bool = True,
) -> dict[str, Any] | None:
    """Return exact source semantics for one ordinary physical split-ledger row."""

    source_row_index = _source_owned_data_row_index(
        row,
        page_number=page_number,
        table_id=table_id,
    )
    cells = list(getattr(row, "cells", None) or [])
    if source_row_index is None or len(cells) != len(headers):
        return None
    values = [str(getattr(cell, "text", "") or "").strip() for cell in cells]
    populated_dates: list[tuple[int, str | None]] = [
        (column, _strict_source_date(values[column]))
        for column in schema["date_columns"]
        if values[column]
    ]
    if not populated_dates or any(value is None for _column, value in populated_dates):
        return None
    date_column, source_date = populated_dates[0]
    if source_date is None:
        return None
    debit_column = int(schema["debit_column"])
    credit_column = int(schema["credit_column"])
    debit_raw, credit_raw = values[debit_column], values[credit_column]
    if bool(debit_raw) == bool(credit_raw):
        return None
    active_column = debit_column if debit_raw else credit_column
    amount = _strict_source_money(values[active_column], nonnegative=True)
    balance_column = int(schema["balance_column"])
    balance = _strict_source_money(values[balance_column], nonnegative=False)
    if amount is None or balance is None:
        return None
    required_columns = {date_column, active_column, balance_column}
    required_refs: list[dict[str, Any]] = []
    required_evidence_ids: list[str] = []
    for column in sorted(required_columns):
        cell = cells[column]
        ref = _strict_source_cell_ref(
            cell,
            page_number=page_number,
            table_id=table_id,
            source_row_index=source_row_index,
            column=column,
        )
        evidence_ids = tuple(
            dict.fromkeys(
                str(evidence_id)
                for evidence_id in (getattr(cell, "evidence_ids", None) or [])
                if str(evidence_id or "").strip()
            )
        )
        if (
            ref is None
            or (require_cell_evidence and not evidence_ids)
            or _strict_cell_bbox(cell) is None
        ):
            return None
        required_refs.append(ref)
        required_evidence_ids.extend(evidence_ids)
    raw_rows = {int(ref["raw_row"]) for ref in required_refs}
    row_bbox = _strict_physical_row_bbox(cells)
    if len(raw_rows) != 1 or row_bbox is None:
        return None
    raw_row = next(iter(raw_rows))
    balance_ref = next(ref for ref in required_refs if int(ref["col"]) == balance_column)
    direction = "expense" if debit_raw else "income"
    header_roles = {
        "date": {"column": date_column, "header": str(headers[date_column])},
        "debit_amount": {"column": debit_column, "header": str(headers[debit_column])},
        "credit_amount": {"column": credit_column, "header": str(headers[credit_column])},
        "balance": {"column": balance_column, "header": str(headers[balance_column])},
    }
    source_roles = {
        "date": deepcopy(header_roles["date"]),
        "amount": {
            "column": active_column,
            "header": str(headers[active_column]),
            "direction": direction,
        },
        "balance": deepcopy(header_roles["balance"]),
    }
    return {
        "key": (page_number, table_id, source_row_index),
        "source_page": page_number,
        "table_id": table_id,
        "source_row_index": source_row_index,
        "raw_row": raw_row,
        "date": source_date,
        "direction": direction,
        "amount": amount,
        "balance": balance,
        "active_column": active_column,
        "balance_column": balance_column,
        "required_columns": sorted(required_columns),
        "required_evidence_ids": list(dict.fromkeys(required_evidence_ids)),
        "required_source_cell_refs": required_refs,
        "balance_source_cell_ref": balance_ref,
        "bbox": row_bbox,
        "source_headers": [str(header) for header in headers],
        "header_roles": header_roles,
        "source_roles": source_roles,
        "active_amount_role": "debit_amount" if direction == "expense" else "credit_amount",
    }


def _physical_row_order_is_consistent(facts: Sequence[dict[str, Any]]) -> bool:
    """Prove one physical table has a single row-index/raw-row/geometry order."""

    if not facts:
        return False
    ordered: list[tuple[int, int, tuple[float, float, float, float]]] = []
    table_keys: set[tuple[int, str]] = set()
    try:
        for fact in facts:
            source_row_index = int(fact["source_row_index"])
            raw_row = int(fact["raw_row"])
            bbox = tuple(float(value) for value in fact["bbox"])
            table_keys.add((int(fact["source_page"]), str(fact["table_id"])))
            if (
                source_row_index < 0
                or raw_row < 0
                or len(bbox) != 4
                or not all(math.isfinite(value) for value in bbox)
                or bbox[2] < bbox[0]
                or bbox[3] < bbox[1]
            ):
                return False
            ordered.append((source_row_index, raw_row, bbox))
    except (KeyError, TypeError, ValueError):
        return False
    if (
        len(table_keys) != 1
        or len({item[0] for item in ordered}) != len(ordered)
        or len({item[1] for item in ordered}) != len(ordered)
        or len({item[1] - item[0] for item in ordered}) != 1
    ):
        return False
    ordered.sort(key=lambda item: item[0])
    vertical_order = all(
        previous[0] < current[0]
        and previous[1] < current[1]
        and previous[2][1] < current[2][1]
        for previous, current in zip(ordered, ordered[1:])
    )
    horizontal_order = all(
        previous[0] < current[0]
        and previous[1] < current[1]
        and previous[2][0] != current[2][0]
        for previous, current in zip(ordered, ordered[1:])
    ) and (
        all(
            previous[2][0] < current[2][0]
            for previous, current in zip(ordered, ordered[1:])
        )
        or all(
            previous[2][0] > current[2][0]
            for previous, current in zip(ordered, ordered[1:])
        )
    )
    return vertical_order or horizontal_order


def _emitted_record_matches_physical_fact(record: dict[str, Any], fact: dict[str, Any]) -> bool:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    required_columns = {int(column) for column in fact["required_columns"]}
    if _canonical_physical_lineage_key(
        source,
        required_columns=required_columns,
        required_raw_row=int(fact["raw_row"]),
    ) != fact["key"]:
        return False
    try:
        source_bbox = tuple(float(value) for value in source["bbox"])
    except (KeyError, TypeError, ValueError):
        return False
    if any(abs(left - right) > 1e-6 for left, right in zip(source_bbox, fact["bbox"], strict=True)):
        return False
    emitted_evidence = {str(item) for item in source.get("evidence_ids") or [] if str(item)}
    if not set(fact["required_evidence_ids"]) <= emitted_evidence:
        return False
    for layer_name in ("normalized", "canonical_raw"):
        layer = record.get(layer_name) if isinstance(record.get(layer_name), dict) else {}
        if (
            _strict_source_date(layer.get("date")) != fact["date"]
            or str(layer.get("direction") or "") != fact["direction"]
            or _as_decimal(layer.get("amount")) != fact["amount"]
            or _as_decimal(layer.get("balance")) != fact["balance"]
        ):
            return False
    return True


def _canonical_physical_row_census_evidence(
    parse_result: Any,
    records: Sequence[dict[str, Any]],
    *,
    source_route: str | None,
    selected_source: str,
) -> dict[str, Any]:
    """Certify complete visible-row coverage from an exact physical-table census."""

    if (
        str(source_route or "") != "digital"
        or str(selected_source or "") != "canonical_table"
        or not records
        or not _has_complete_source_page_selection(parse_result)
    ):
        return {}
    parser_info = getattr(parse_result, "parser_info", None)
    structure = getattr(parser_info, "structure", None)
    structure = structure if isinstance(structure, dict) else {}
    gate = structure.get("table_reconstruction_gate")
    gate = gate if isinstance(gate, dict) else {}
    try:
        physical_count = int(structure.get("physical_table_count") or 0)
        candidate_count = int(gate.get("candidate_count") or 0)
        gate_physical_count = int(gate.get("physical_table_count") or 0)
    except (TypeError, ValueError):
        return {}
    actual_table_count = sum(
        len(getattr(page, "tables", None) or []) for page in (getattr(parse_result, "pages", None) or [])
    )
    if (
        structure.get("table_extraction") != "full"
        or gate.get("applicable") is not True
        or gate.get("passed") is not True
        or physical_count <= 0
        or physical_count != candidate_count
        or physical_count != gate_physical_count
        or physical_count != actual_table_count
    ):
        return {}

    table_keys: set[tuple[int, str]] = set()
    physical_facts: dict[tuple[int, str, int], dict[str, Any]] = {}
    transaction_tables: dict[int, set[str]] = defaultdict(set)
    source_pages: set[int] = set()
    table_pages: set[int] = set()
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        if page_number <= 0 or page_number in source_pages:
            return {}
        source_pages.add(page_number)
        for table in getattr(page, "tables", None) or []:
            table_pages.add(page_number)
            table_id = str(getattr(table, "table_id", "") or "").strip()
            table_key = (page_number, table_id)
            if not table_id or table_key in table_keys:
                return {}
            table_keys.add(table_key)
            headers = [str(header or "") for header in (getattr(table, "headers", None) or [])]
            schema = _physical_split_ledger_schema(headers)
            metadata = getattr(table, "metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            promoted_table = metadata.get("header_source") == "data_row" or metadata.get("preserve_headers") is False
            row_indices: set[int] = set()
            table_facts: list[dict[str, Any]] = []
            for row in getattr(table, "rows", None) or []:
                source_row_index = _source_owned_data_row_index(
                    row,
                    page_number=page_number,
                    table_id=table_id,
                )
                if source_row_index is None or source_row_index in row_indices:
                    return {}
                row_indices.add(source_row_index)
                fact = (
                    _physical_split_transaction_fact(
                        row,
                        page_number=page_number,
                        table_id=table_id,
                        headers=headers,
                        schema=schema,
                    )
                    if schema is not None
                    else None
                )
                if fact is None:
                    continue
                if promoted_table:
                    return {}
                table_facts.append(fact)
            if table_facts:
                if not _physical_row_order_is_consistent(table_facts):
                    return {}
                for fact in table_facts:
                    if fact["key"] in physical_facts:
                        return {}
                    physical_facts[fact["key"]] = fact
                transaction_tables[page_number].add(table_id)

    if (
        not source_pages
        or not table_pages
        or any(len(transaction_tables.get(page, set())) != 1 for page in table_pages)
    ):
        return {}

    try:
        from docmirror.plugins.bank_statement.canonical_quality import physical_transaction_row_estimate

        physical_estimate = physical_transaction_row_estimate(parse_result)
    except (AttributeError, TypeError, ValueError):
        return {}
    census_keys = sorted(physical_facts)
    emitted_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for record in records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        key = _canonical_physical_lineage_key(source)
        if key is None or key in emitted_by_key or key not in physical_facts:
            return {}
        if not _emitted_record_matches_physical_fact(record, physical_facts[key]):
            return {}
        emitted_by_key[key] = record
    if (
        not census_keys
        or int(physical_estimate or 0) != len(census_keys)
        or set(census_keys) != set(emitted_by_key)
    ):
        return {}
    page_counts = Counter(key[0] for key in census_keys)
    row_sources = []
    for key in census_keys:
        fact = physical_facts[key]
        row_sources.append(
            {
                "source_page": fact["source_page"],
                "table_id": fact["table_id"],
                "source_row_index": fact["source_row_index"],
                "raw_row": fact["raw_row"],
                "date": fact["date"],
                "direction": fact["direction"],
                "amount": _normalized_money(fact["amount"]),
                "balance": _normalized_money(fact["balance"]),
                "active_column": fact["active_column"],
                "balance_column": fact["balance_column"],
                "required_columns": list(fact["required_columns"]),
                "required_evidence_ids": list(fact["required_evidence_ids"]),
                "required_source_cell_refs": deepcopy(fact["required_source_cell_refs"]),
                "source_cell_ref_owner": "canonical_physical_table",
                "balance_source_cell_ref": deepcopy(fact["balance_source_cell_ref"]),
                "bbox": list(fact["bbox"]),
                "source_headers": list(fact["source_headers"]),
                "header_roles": deepcopy(fact["header_roles"]),
                "source_roles": deepcopy(fact["source_roles"]),
                "active_amount_role": fact["active_amount_role"],
            }
        )
    return {
        "expected_rows": len(census_keys),
        "source": "canonical_physical_table_row_census",
        "confidence": 1.0,
        "row_sources": row_sources,
        "pages": [key[0] for key in census_keys],
        "census": {
            "exact_lineage_match": True,
            "exact_semantic_match": True,
            "consistent_physical_order": True,
            "unique_transaction_table_per_page": True,
            "physical_table_count": physical_count,
            "transaction_row_count": len(census_keys),
            "table_ids": sorted({key[1] for key in census_keys}),
            "page_counts": {str(page): count for page, count in sorted(page_counts.items())},
        },
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


def _source_lineage_triplet(source: dict[str, Any]) -> tuple[int, str, int] | None:
    try:
        page = int(source.get("source_page") or 0)
        row_index = int(source.get("source_row_index"))
    except (TypeError, ValueError):
        return None
    table_id = str(source.get("table_id") or source.get("source_table_id") or "").strip()
    if page <= 0 or row_index < 0 or not table_id:
        return None
    return page, table_id, row_index


def _terminal_visible_record(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    lineaged: list[tuple[tuple[int, str, int], dict[str, Any]]] = []
    for record in rows:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        key = _source_lineage_triplet(source)
        if key is None:
            lineaged = []
            break
        lineaged.append((key, record))
    if lineaged:
        if len({key for key, _record in lineaged}) != len(lineaged):
            return None
        table_keys = {(key[0], key[1]) for key, _record in lineaged}
        if len(table_keys) != 1:
            return None
        if all(
            str((record.get("source") or {}).get("source") or "") == "canonical_physical_table"
            for _key, record in lineaged
        ):
            physical_order_facts = []
            for key, record in lineaged:
                source = record.get("source") if isinstance(record.get("source"), dict) else {}
                if _canonical_physical_lineage_key(source) != key:
                    return None
                try:
                    raw_rows = {int(ref["raw_row"]) for ref in source["source_cell_refs"]}
                except (KeyError, TypeError, ValueError):
                    return None
                if len(raw_rows) != 1:
                    return None
                physical_order_facts.append(
                    {
                        "source_page": key[0],
                        "table_id": key[1],
                        "source_row_index": key[2],
                        "raw_row": next(iter(raw_rows)),
                        "bbox": source.get("bbox"),
                    }
                )
            if not _physical_row_order_is_consistent(physical_order_facts):
                return None
        return max(lineaged, key=lambda item: item[0][2])[1]

    positioned: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for record in rows:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        bbox = source.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            numeric = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        if (
            not all(math.isfinite(value) for value in numeric)
            or numeric[2] < numeric[0]
            or numeric[3] < numeric[1]
        ):
            return None
        positioned.append(((numeric[3], numeric[1]), record))
    if not positioned:
        return None
    maximum = max(position for position, _record in positioned)
    matches = [record for position, record in positioned if position == maximum]
    return matches[0] if len(matches) == 1 else None


def _previous_row_source(
    record: dict[str, Any],
    page: int,
    *,
    anchor_row: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    ref: dict[str, Any] = {
        "source": str(source.get("source") or "transaction_row"),
        "source_page": page,
    }
    for key in (
        "page_range",
        "bbox",
        "evidence_ids",
        "table_id",
        "source_table_id",
        "source_row_index",
        "source_cell_refs",
    ):
        if source.get(key) not in (None, "", [], {}):
            ref[key] = deepcopy(source[key])
    if anchor_row is not None:
        balance_ref = anchor_row.get("balance_source_cell_ref")
        if not isinstance(balance_ref, dict):
            return None
        source_refs = source.get("source_cell_refs")
        if not isinstance(source_refs, list) or balance_ref not in source_refs:
            return None
        ref["source_cell_refs"] = [deepcopy(balance_ref)]
        ref["balance_source_cell_ref"] = deepcopy(balance_ref)
    return ref


def _carry_boundaries(
    facts_by_page: dict[int, list[_HeaderFact]],
    rows_by_page: dict[int, list[dict[str, Any]]],
    *,
    anchor_evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    pages = sorted(rows_by_page)
    if len(pages) < 2:
        return None
    exact_anchor_rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    if (anchor_evidence or {}).get("source") == "canonical_physical_table_row_census":
        for item in (anchor_evidence or {}).get("row_sources") or []:
            if not isinstance(item, dict):
                return None
            key = _source_lineage_triplet(item)
            if key is None or key in exact_anchor_rows:
                return None
            exact_anchor_rows[key] = item
    boundaries: list[dict[str, Any]] = []
    for previous_page, next_page in zip(pages, pages[1:]):
        if next_page != previous_page + 1:
            return None
        previous_record = _terminal_visible_record(rows_by_page[previous_page])
        if previous_record is None:
            return None
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
        previous_source = (
            previous_record.get("source") if isinstance(previous_record.get("source"), dict) else {}
        )
        previous_key = _source_lineage_triplet(previous_source)
        anchor_row = exact_anchor_rows.get(previous_key) if previous_key is not None else None
        if exact_anchor_rows and anchor_row is None:
            return None
        previous_row_source = _previous_row_source(
            previous_record,
            previous_page,
            anchor_row=anchor_row,
        )
        if previous_row_source is None:
            return None
        boundaries.append(
            {
                "from_source_page": previous_page,
                "to_source_page": next_page,
                "last_visible_balance": _normalized_money(previous_balance),
                "brought_forward_balance": _normalized_money(brought_forward),
                "direction": direction,
                "amount": _normalized_money(amount),
                "previous_row_source": previous_row_source,
                "brought_forward_source": _field_source(carry_facts),
            }
        )
    return boundaries


def _has_direct_positioned_terminal_source(detail: dict[str, Any], terminal_page: int) -> bool:
    direct_sources = {
        "canonical_evidence_atoms",
        "canonical_physical_table",
        "header.kv",
        "page_headers",
        "parse_result_ocr_text",
    }
    detail_source = str(detail.get("source") or "").casefold()
    evidence_ids = detail.get("evidence_ids")
    refs = detail.get("source_refs")
    derivation = detail.get("derivation")
    if (
        detail_source not in direct_sources
        or detail_source == "derived_explicit_page_aggregate"
        or derivation == "sum_explicit_page_totals"
        or (isinstance(derivation, (list, tuple)) and "sum_explicit_page_totals" in derivation)
        or detail.get("normalized_only") is True
        or not isinstance(evidence_ids, (list, tuple))
        or not evidence_ids
        or not all(str(evidence_id or "").strip() for evidence_id in evidence_ids)
        or not isinstance(refs, (list, tuple))
        or not refs
    ):
        return False
    for ref in refs:
        if not isinstance(ref, dict) or str(ref.get("source") or "").casefold() != detail_source:
            return False
        try:
            source_page = int(ref.get("source_page") or ref.get("page") or 0)
            bbox = tuple(float(value) for value in ref.get("bbox") or ())
        except (TypeError, ValueError):
            return False
        if (
            source_page != terminal_page
            or len(bbox) != 4
            or not all(math.isfinite(value) for value in bbox)
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            return False
    return True


def _terminal_aggregate_sources(
    header: dict[str, Any],
    *,
    count_field: str,
    total_field: str,
    terminal_page: int,
) -> dict[str, Any] | None:
    normalized = header.get("normalized") if isinstance(header.get("normalized"), dict) else {}
    canonical_raw = header.get("canonical_raw") if isinstance(header.get("canonical_raw"), dict) else {}
    source = header.get("source") if isinstance(header.get("source"), dict) else {}
    field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
    count_source = field_sources.get(count_field)
    total_source = field_sources.get(total_field)
    normalized_count = _as_exact_int(normalized.get(count_field))
    canonical_count = _as_exact_int(canonical_raw.get(count_field))
    normalized_total = _as_decimal(normalized.get(total_field))
    canonical_total = _as_decimal(canonical_raw.get(total_field))
    if (
        normalized_count is None
        or canonical_count is None
        or normalized_count != canonical_count
        or normalized_total is None
        or canonical_total is None
        or normalized_total != canonical_total
        or not isinstance(count_source, dict)
        or not isinstance(total_source, dict)
        or not _has_direct_positioned_terminal_source(count_source, terminal_page)
        or not _has_direct_positioned_terminal_source(total_source, terminal_page)
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
    scoped_evidence = {
        "source": anchor_evidence["source"],
        "confidence": anchor_evidence["confidence"],
        "document_expected_rows": anchor_evidence["expected_rows"],
        "scope_anchored_rows": sum(anchor_counts.values()),
        "scope_visible_rows": sum(visible_counts.values()),
        "page_counts": {str(page): count for page, count in sorted(anchor_counts.items())},
    }
    if anchor_evidence.get("source") == "canonical_physical_table_row_census":
        scoped_row_sources = []
        for item in anchor_evidence.get("row_sources") or []:
            if not isinstance(item, dict):
                return None
            try:
                source_page = int(item.get("source_page") or 0)
            except (TypeError, ValueError):
                return None
            if scope[0] <= source_page <= scope[1]:
                scoped_row_sources.append(deepcopy(item))
        anchor_keys = [_source_lineage_triplet(item) for item in scoped_row_sources]
        visible_keys = [
            _source_lineage_triplet(
                record.get("source") if isinstance(record.get("source"), dict) else {}
            )
            for page_rows in rows_by_page.values()
            for record in page_rows
        ]
        if (
            any(key is None for key in anchor_keys)
            or any(key is None for key in visible_keys)
            or Counter(anchor_keys) != Counter(visible_keys)
        ):
            return None
        scoped_evidence.update(
            {
                "exact_lineage_match": True,
                "exact_semantic_match": True,
                "row_sources": scoped_row_sources,
            }
        )
    return scoped_evidence


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
        anchor_evidence = _canonical_physical_row_census_evidence(
            parse_result,
            records,
            source_route=source_route,
            selected_source=selected_source,
        )
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
        boundaries = _carry_boundaries(
            facts_by_page,
            rows_by_page,
            anchor_evidence=scope_anchors,
        )
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
    "audit_source_fact_conservation",
    "attach_statement_context",
    "build_source_metadata_records",
    "build_statement_header_records",
    "page_texts_with_business_headers",
    "reconcile_source_unitemized_residuals",
    "statement_scope_count",
]
