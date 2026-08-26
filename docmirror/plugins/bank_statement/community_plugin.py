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

import calendar
import copy
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from docmirror.plugins._base.base_table_parser import BaseTableParser
from docmirror.plugins._base.column_registry import ColumnMapping
from docmirror.plugins._base.projector import ProjectionData
from docmirror.plugins._base.standardizer import normalize_timestamp
from docmirror.plugins.bank_statement.canonical_quality import audit_amount_consistency, audit_row_accounting
from docmirror.plugins.bank_statement.extract_pipeline import (
    authoritative_issuer_transaction_count_detail,
    is_authoritative_issuer_row_count,
    is_source_bound_issuer_detail,
    run_bank_statement_extract,
)
from docmirror.plugins.bank_statement.header_resolve import normalize_bank_matching_text, normalize_header_cell
from docmirror.plugins.bank_statement.statement_context import (
    attach_statement_context,
    build_statement_header_records,
    page_texts_with_business_headers,
    reconcile_source_unitemized_residuals,
)
from docmirror.plugins.bank_statement.wide_table_recovery import (
    page_texts_from_parse_result,
    resolve_row_count_evidence,
)

BANK_COLUMN_REGISTRY: dict[str, ColumnMapping] = {
    "序号": ColumnMapping(
        field="sequence_no",
        aliases=["序 号", "No.", "序列号", "日志号", "日 志 号", "交易序号", "Sequence"],
    ),
    "交易日期": ColumnMapping(field="date", format_hint="date", aliases=["日期", "记账日", "Date"]),
    "交易时间": ColumnMapping(
        field="timestamp",
        format_hint="datetime",
        aliases=["时间", "日期时间", "交易日期时间", "Date Time", "Datetime", "Time"],
    ),
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
            "收": "income",
            "借": "expense",
            "借Dr": "expense",
            "Dr": "expense",
            "支": "expense",
        },
        aliases=[
            "收支",
            "方向",
            "交易方向",
            "交易类别",
            "收入/支出",
            "支/收",
            "月收/支",
            "月收支",
            "借贷",
            "借/贷",
            "借贷标志",
            "Dc Flg",
        ],
    ),
    "摘要": ColumnMapping(field="summary", aliases=["摘要代码", "摘要描述", "交易摘要", "Description", "Memo"]),
    "交易金额": ColumnMapping(
        field="amount",
        unit="CNY",
        aliases=[
            "金额",
            "发生额",
            "交易发生金额",
            "Amount",
            "借方发生额",
            "贷方发生额",
            "收入金额",
            "支出金额",
            "收入/支出金额",
            "支出/收入金额",
            "收/支金额",
            "支/收交易金额",
        ],
    ),
    "余额": ColumnMapping(field="balance", unit="CNY", aliases=["账户余额", "本次余额", "Balance"]),
    "账号": ColumnMapping(field="own_account", aliases=["本方账号", "本方账户"]),
    "储种": ColumnMapping(field="deposit_type", aliases=["存款种类"]),
    "地区": ColumnMapping(field="region_code", aliases=["地区代码"]),
    "对方户名": ColumnMapping(
        field="counter_party",
        aliases=[
            "收(付)方名称",
            "收（付）方名称",
            "对方名称",
            "对手信息",
            "对手名称",
            "交易对方",
            "交易对手信息",
            "Counter party",
            "Counterparty Name",
            "对方账号与户名",
        ],
    ),
    "对方账号": ColumnMapping(
        field="counter_account",
        aliases=["收(付)方账号", "收（付）方账号", "对方账户", "对方账户/对方银行", "Counter account"],
    ),
    "对方行号": ColumnMapping(field="counter_bank_code", aliases=["对方银行行号"]),
    "对方行名": ColumnMapping(
        field="counter_bank_name",
        aliases=["对方开户行", "对方银行名称", "对手机构", "Counterparty Institution"],
    ),
    "交易渠道": ColumnMapping(field="channel", aliases=["渠道", "交易方式"]),
    "用途": ColumnMapping(field="purpose", aliases=["交易用途"]),
    "交易流水号": ColumnMapping(
        field="reference",
        aliases=["柜员流水号", "流水号", "电子回单编号", "Reference", "Reference No."],
    ),
    "交易附言": ColumnMapping(field="remittance_note", aliases=["附言"]),
    "备注": ColumnMapping(field="note", aliases=["Remarks", "Notes"]),
    "交易地点": ColumnMapping(field="transaction_location", aliases=["交易场所", "Trading Place"]),
    "币种": ColumnMapping(field="currency", aliases=["币别", "货币", "Currency"]),
    "子账号": ColumnMapping(field="sub_account", aliases=["子账户"]),
    "钞汇": ColumnMapping(
        field="cash_remittance",
        aliases=["现/转", "现金/转账", "现转标志", "现金/转账标志"],
    ),
    "凭证种类": ColumnMapping(field="voucher_type", aliases=["凭证类型"]),
    "凭证号": ColumnMapping(field="voucher_number", aliases=["凭证号码"]),
    "交易代码": ColumnMapping(field="transaction_code", aliases=["业务代码"]),
    "交易机构": ColumnMapping(field="transaction_institution", aliases=["经办机构"]),
    "柜员号": ColumnMapping(field="teller_id", aliases=["柜员"]),
    "记账日期": ColumnMapping(field="posting_date", format_hint="date", aliases=["会计日期", "Accounting Date"]),
    "交易名称": ColumnMapping(field="transaction_name", aliases=["交易描述", "Transaction Name"]),
    "起息日": ColumnMapping(field="value_date", format_hint="date", aliases=["起息日期", "Value Date"]),
    "银行流水": ColumnMapping(field="bank_serial", aliases=["Bank Serial"]),
    "业务明细": ColumnMapping(field="business_detail", aliases=["Business Detail"]),
    "业务背景": ColumnMapping(field="business_context", aliases=["Business Context"]),
    "业务系统参考号": ColumnMapping(field="business_system_reference", aliases=["System Reference"]),
}

BANK_STANDARD_FIELDS = [
    "date",
    "timestamp",
    "summary",
    "direction",
    "amount",
    "balance",
    "own_account",
    "deposit_type",
    "region_code",
    "sub_account",
    "counter_party",
    "counter_account",
    "sequence_no",
    "counter_bank_code",
    "counter_bank_name",
    "channel",
    "purpose",
    "counterparty_status",
    "cash_remittance",
    "currency",
    "note",
    "posting_date",
    "reference",
    "remittance_note",
    "teller_id",
    "transaction_code",
    "transaction_institution",
    "transaction_location",
    "transaction_name",
    "value_date",
    "bank_serial",
    "business_detail",
    "business_context",
    "business_system_reference",
    "voucher_number",
    "voucher_type",
]

BANK_DATA_DICTIONARY: dict[str, Any] = {
    "fields": {
        "organization": {"label": "银行名称", "type": "string"},
        "subject_name": {"label": "账户名称", "type": "string"},
        "subject_id": {"label": "账户标识", "type": "long_id", "sensitive": True, "display": "masked"},
        "account_holder": {"label": "账户名称", "type": "string"},
        "account_number": {"label": "账号", "type": "long_id", "sensitive": True, "display": "masked"},
        "customer_number": {
            "label": "客户号",
            "type": "long_id",
            "sensitive": True,
            "display": "masked",
        },
        "bank_name": {"label": "开户银行", "type": "string"},
        "branch_name": {"label": "开户机构", "type": "string"},
        "account_type": {"label": "账户类型", "type": "string"},
        "deposit_type": {"label": "存款种类", "type": "string"},
        "statement_number": {"label": "账单号", "type": "string"},
        "query_period": {"label": "查询期间", "type": "string"},
        "period_start": {"label": "账期开始", "type": "date"},
        "period_end": {"label": "账期结束", "type": "date"},
        "print_date": {"label": "打印日期", "type": "date"},
        "document_date": {"label": "单据日期", "type": "date"},
        "total_transactions": {"label": "交易总笔数", "type": "integer"},
        "total_amount": {"label": "交易总金额", "type": "money"},
        "debit_count": {"label": "借方总笔数", "type": "integer"},
        "debit_total": {"label": "借方总金额", "type": "money"},
        "credit_count": {"label": "贷方总笔数", "type": "integer"},
        "credit_total": {"label": "贷方总金额", "type": "money"},
        "source_unitemized_debit_count": {
            "label": "来源未逐笔列示借方笔数",
            "type": "integer",
            "definition": "来源借方汇总与可见逐笔交易之间、经跨页承前余额独立核对的笔数差额。",
            "derived": True,
        },
        "source_unitemized_debit_amount": {
            "label": "来源未逐笔列示借方金额",
            "type": "money",
            "definition": "来源借方汇总与可见逐笔交易之间、经跨页承前余额独立核对的金额差额。",
            "derived": True,
        },
        "source_unitemized_credit_count": {
            "label": "来源未逐笔列示贷方笔数",
            "type": "integer",
            "definition": "来源贷方汇总与可见逐笔交易之间、经跨页承前余额独立核对的笔数差额。",
            "derived": True,
        },
        "source_unitemized_credit_amount": {
            "label": "来源未逐笔列示贷方金额",
            "type": "money",
            "definition": "来源贷方汇总与可见逐笔交易之间、经跨页承前余额独立核对的金额差额。",
            "derived": True,
        },
        "opening_balance": {"label": "期初余额", "type": "money"},
        "closing_balance": {"label": "期末余额", "type": "money"},
        "currency": {"label": "币种", "type": "string"},
        "statement_title": {"label": "流水标题", "type": "string"},
        "style_id": {"label": "版式标识", "type": "string"},
        "style_confidence": {"label": "版式置信度", "type": "number"},
        "parser_chain": {"label": "解析链", "type": "string"},
        "institution_hint": {"label": "识别银行", "type": "string"},
        "secondary_styles": {"label": "备选版式", "type": "string"},
        "reconstruction_source": {"label": "重建来源", "type": "string"},
        "expected_primary_rows": {"label": "预期交易笔数", "type": "integer"},
        "extracted_rows": {"label": "实际提取笔数", "type": "integer"},
        "coverage_ratio": {"label": "交易覆盖率", "type": "percentage"},
        "institution_authority": {"label": "银行识别依据", "type": "string"},
        "pipe_parse_failed": {"label": "管道表解析失败", "type": "boolean"},
        "canonical_expected": {"label": "Canonical 预期笔数", "type": "integer"},
        "canonical_extracted": {"label": "Canonical 提取笔数", "type": "integer"},
        "canonical_ratio": {"label": "Canonical 覆盖率", "type": "percentage"},
        "extract_status": {"label": "提取状态", "type": "string"},
        "blo_tables_parsed": {"label": "BLO 已解析表数", "type": "integer"},
        "blo_tables_skipped": {"label": "BLO 已跳过表数", "type": "integer"},
        "extraction_route": {"label": "提取路线", "type": "string"},
        "source_reported_transaction_count": {"label": "原文报告交易笔数", "type": "integer"},
        "document_scene_refined": {"label": "修正文档场景", "type": "string"},
        "layout_profile_id_refined": {"label": "修正版式配置", "type": "string"},
        "layout_profile_refine_confidence": {"label": "版式修正置信度", "type": "number"},
    },
    "record_columns": {
        "statement_header_id": {
            "label": "流水表头记录ID",
            "type": "string",
            "definition": "关联 statement_header 数据集中的来源流水表头记录。",
        },
        "statement_title": {"label": "流水标题", "type": "string"},
        "account_holder": {"label": "账户名称", "type": "string"},
        "bank_name": {"label": "开户银行", "type": "string"},
        "query_period": {"label": "查询期间", "type": "string"},
        "period_start": {"label": "账期开始", "type": "date"},
        "period_end": {"label": "账期结束", "type": "date"},
        "print_date": {"label": "打印日期", "type": "date"},
        "document_date": {"label": "单据日期", "type": "date"},
        "amount": {"label": "交易金额", "type": "money"},
        "balance": {"label": "账户余额", "type": "money"},
        "channel": {"label": "交易渠道", "type": "string"},
        "own_account": {"label": "本方账号", "type": "long_id"},
        "deposit_type": {"label": "储种", "type": "string"},
        "region_code": {"label": "地区代码", "type": "string"},
        "sub_account": {"label": "子账号", "type": "long_id"},
        "counter_account": {"label": "对方账号", "type": "long_id"},
        "counter_bank_code": {"label": "对方银行代码", "type": "string"},
        "counter_bank_name": {"label": "对方银行名称", "type": "string"},
        "counter_party": {"label": "对方户名", "type": "string"},
        "counterparty_status": {"label": "对方信息状态", "type": "string"},
        "date": {"label": "交易日期", "type": "date"},
        "direction": {"label": "收支方向", "type": "string"},
        "purpose": {"label": "交易用途", "type": "string"},
        "reference": {"label": "交易参考号", "type": "string"},
        "sequence_no": {"label": "序号", "type": "long_id"},
        "summary": {"label": "摘要", "type": "string"},
        "timestamp": {"label": "交易时间", "type": "datetime"},
        "cash_remittance": {"label": "钞汇/现转", "type": "string"},
        "currency": {"label": "币种", "type": "string"},
        "note": {"label": "备注", "type": "string"},
        "posting_date": {"label": "记账日期", "type": "date"},
        "remittance_note": {"label": "附言", "type": "string"},
        "teller_id": {"label": "柜员号", "type": "string"},
        "transaction_code": {"label": "交易代码", "type": "string"},
        "transaction_institution": {"label": "交易机构", "type": "string"},
        "transaction_location": {"label": "交易地点", "type": "string"},
        "transaction_name": {"label": "交易名称", "type": "string"},
        "value_date": {"label": "起息日", "type": "date"},
        "bank_serial": {"label": "银行流水", "type": "string"},
        "business_detail": {"label": "业务明细", "type": "string"},
        "business_context": {"label": "业务背景", "type": "string"},
        "business_system_reference": {"label": "业务系统参考号", "type": "string"},
        "voucher_number": {"label": "凭证号", "type": "string"},
        "voucher_type": {"label": "凭证种类", "type": "string"},
    },
    "datasets": {
        "statement_header": {
            "definition": "一行对应一个来源银行流水表头或账户账期范围。",
            "columns": {
                "statement_title": {"label": "流水标题", "type": "string"},
                "bank_name": {"label": "开户银行", "type": "string"},
                "account_holder": {"label": "账户名称", "type": "string"},
                "account_number": {
                    "label": "账号",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "card_number": {
                    "label": "卡号",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "internal_account": {
                    "label": "内部账号",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "customer_number": {
                    "label": "客户号",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "branch_name": {"label": "开户机构", "type": "string"},
                "transaction_institution": {"label": "交易机构", "type": "string"},
                "accepting_branch": {"label": "受理机构", "type": "string"},
                "account_type": {"label": "账户类型", "type": "string"},
                "deposit_type": {"label": "存款种类", "type": "string"},
                "cash_remittance": {"label": "钞汇/现转", "type": "string"},
                "statement_number": {"label": "账单号", "type": "string"},
                "statement_code": {"label": "账单代码", "type": "string"},
                "statement_type": {"label": "账单类型", "type": "string"},
                "list_number": {"label": "清单编号", "type": "string"},
                "statement_month": {"label": "账单月份", "type": "string"},
                "statement_year": {"label": "账单年份", "type": "integer"},
                "statement_month_number": {"label": "账单月", "type": "integer"},
                "electronic_serial": {"label": "电子流水号", "type": "string"},
                "verification_code": {"label": "验证码", "type": "string"},
                "proof_number": {"label": "证明编号", "type": "string"},
                "wechat_id": {"label": "微信号", "type": "string"},
                "id_type": {"label": "证件类型", "type": "string"},
                "id_number": {
                    "label": "证件号码",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "amount_unit": {"label": "金额单位", "type": "string"},
                "currency": {"label": "币种", "type": "string"},
                "query_period": {"label": "查询期间", "type": "string"},
                "statement_period": {"label": "账单统计期间", "type": "string"},
                "statement_cutoff_date": {"label": "出单截至日期", "type": "date"},
                "period_start": {"label": "账期开始", "type": "date"},
                "period_end": {"label": "账期结束", "type": "date"},
                "query_date": {"label": "查询日期", "type": "string"},
                "print_date": {"label": "打印日期", "type": "date"},
                "print_timestamp": {"label": "打印时间", "type": "datetime"},
                "query_timestamp": {"label": "查询时间", "type": "datetime"},
                "application_time": {"label": "申请时间", "type": "datetime"},
                "issue_date": {"label": "开立日期", "type": "date"},
                "issue_timestamp": {"label": "开立时间", "type": "datetime"},
                "document_date": {"label": "单据日期", "type": "date"},
                "filter_condition": {"label": "筛选条件", "type": "string"},
                "direction_filter": {"label": "交易方向筛选", "type": "string"},
                "sort_order": {"label": "排序方向", "type": "string"},
                "print_channel": {"label": "打印渠道", "type": "string"},
                "print_teller": {"label": "打印柜员", "type": "string"},
                "print_count": {"label": "打印次数", "type": "integer"},
                "print_method": {"label": "打印方式", "type": "string"},
                "device_number": {"label": "设备编号", "type": "string"},
                "query_teller": {"label": "查询柜员", "type": "string"},
                "department": {"label": "部门", "type": "string"},
                "customer_branch": {"label": "客户行", "type": "string"},
                "total_transactions": {"label": "交易总笔数", "type": "integer"},
                "total_amount": {"label": "交易总金额", "type": "money"},
                "debit_count": {"label": "借方总笔数", "type": "integer"},
                "debit_total": {"label": "借方总金额", "type": "money"},
                "credit_count": {"label": "贷方总笔数", "type": "integer"},
                "credit_total": {"label": "贷方总金额", "type": "money"},
                "source_unitemized_debit_count": {
                    "label": "来源未逐笔列示借方笔数",
                    "type": "integer",
                    "derived": True,
                },
                "source_unitemized_debit_amount": {
                    "label": "来源未逐笔列示借方金额",
                    "type": "money",
                    "derived": True,
                },
                "source_unitemized_credit_count": {
                    "label": "来源未逐笔列示贷方笔数",
                    "type": "integer",
                    "derived": True,
                },
                "source_unitemized_credit_amount": {
                    "label": "来源未逐笔列示贷方金额",
                    "type": "money",
                    "derived": True,
                },
                "opening_balance": {"label": "期初余额", "type": "money"},
                "closing_balance": {"label": "期末余额", "type": "money"},
                "brought_forward_balance": {"label": "承前余额", "type": "money"},
                "account_balance": {"label": "账户余额", "type": "money"},
                "summary_code": {"label": "摘要代码", "type": "string"},
                "amount_upper_limit": {"label": "金额上限", "type": "money"},
                "amount_lower_limit": {"label": "金额下限", "type": "money"},
            },
        }
    },
    "enums": {
        "direction": {"income": "收入", "expense": "支出"},
        "counterparty_status": {"present": "已提供", "source_null": "原文未提供"},
        "extract_status": {
            "success": "成功",
            "low_coverage": "覆盖率偏低",
            "degraded": "降级",
            "failed": "失败",
        },
        "pipe_parse_failed": {"true": "是", "false": "否"},
        "document_type": {
            "bank_statement": "银行流水",
            "bank_reconciliation": "银行对账单",
        },
        "document_scene_refined": {
            "bank_statement": "银行流水",
            "bank_reconciliation": "银行对账单",
        },
        "layout_profile_id_refined": {
            "borderless_ledger_bank": "无框银行流水版式",
        },
    },
}

BANK_IDENTITY_FIELDS: Sequence[tuple[str, Sequence[str]]] = (
    ("account_holder", ("Account holder", "Account name", "Card holder", "Customer name", "户名", "账户名")),
    ("account_number", ("Account number", "Card number", "Customer account number", "账号", "账户号", "卡号")),
    ("bank_name", ("Bank name", "Issuer bank", "银行名称")),
    ("branch_name", ("Bank branch", "Opening branch", "开户银行", "开户行", "开户机构", "打印机构")),
    ("query_period", ("Query period", "From/to date", "Period", "查询时间段", "交易时段")),
    ("print_date", ("打印日期",)),
    ("total_transactions", ("总笔数", "总条数")),
    ("currency", ("Currency", "币种")),
)


def _exact_source_value(raw_txn: dict[str, Any], aliases: Sequence[str]) -> Any:
    """Return an exact or stacked-header source value, never a fuzzy substring."""

    def source_header_identity(value: Any) -> str:
        # Canonical-raw mapping must retain the source field's exact semantic
        # identity.  The layout registry intentionally collapses roles such as
        # ``交易时间`` into ``交易日期`` for parser matching; using that profile
        # here would falsely claim a date-only source cell as a timestamp.
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()

    normalized_aliases = {source_header_identity(alias) for alias in aliases if str(alias or "").strip()}
    if not normalized_aliases:
        return ""
    for raw_header, value in raw_txn.items():
        if str(raw_header).startswith("_"):
            continue
        header = source_header_identity(raw_header)
        parts = {source_header_identity(part) for part in str(raw_header or "").splitlines() if str(part).strip()}
        if header in normalized_aliases or parts.intersection(normalized_aliases):
            return value
    return ""


def _normalize_source_business_date(value: str) -> str:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    if match := re.fullmatch(r"(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})", compact):
        year = 2000 + int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        try:
            calendar.monthrange(year, month)
            if not 1 <= day <= calendar.monthrange(year, month)[1]:
                return compact
        except (ValueError, OverflowError):
            return compact
        return f"{year:04d}-{month:02d}-{day:02d}"
    parsed = normalize_timestamp(compact)
    return parsed[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", parsed) else parsed


def _is_explicit_account_identity_label(value: str) -> bool:
    """Return whether a KV key explicitly labels the statement account."""
    label = re.sub(r"\s+", " ", str(value or "")).strip(" :：").casefold()
    explicit_labels = (
        "账号",
        "账户号",
        "账户账号",
        "银行账号",
        "客户账号",
        "卡号",
        "账号/卡号",
        "卡号/账号",
        "account number",
        "card number",
        "customer account number",
    )
    return any(label == candidate or label.startswith(f"{candidate} ") for candidate in explicit_labels)


def _evidence_transaction_header_top(atoms: list[dict[str, Any]]) -> float | None:
    """Locate the first transaction header band using source header semantics."""
    for anchor in sorted(atoms, key=lambda atom: float(atom["bbox"][1])):
        anchor_text = normalize_bank_matching_text(str(anchor.get("text") or ""))
        if not any(marker in anchor_text for marker in ("交易日期", "记账日期", "交易时间")):
            continue
        baseline = float(anchor["bbox"][1])
        band = [atom for atom in atoms if abs(float(atom["bbox"][1]) - baseline) <= 12.0]
        joined = "".join(normalize_bank_matching_text(str(atom.get("text") or "")) for atom in band)
        has_amount_structure = any(marker in joined for marker in ("交易金额", "发生额")) or (
            "借方" in joined and "贷方" in joined
        )
        if has_amount_structure and "余额" in joined:
            return min(float(atom["bbox"][1]) for atom in band)
    return None


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

    def _normalize(self, raw_txn: dict[str, str]) -> dict[str, Any]:
        """Use the shared mapper without inventing an unevidenced FX field."""
        normalized = super()._normalize(raw_txn)
        normalized.pop("amount_cny", None)
        # ICBC's electronic debit-account history uses the bare source headers
        # ``账号`` / ``储种`` / ``地区`` for account-owned attributes.  The base
        # mapper's intentionally permissive substring fallback can otherwise
        # copy bare ``账号`` into ``对方账号``.  Exact source roles win, and an
        # ICBC row with no counterparty header must not acquire one by inference.
        deposit_type = _exact_source_value(raw_txn, ("储种",))
        region_code = _exact_source_value(raw_txn, ("地区",))
        source_headers = {
            re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(header or "")))
            for header in raw_txn
            if not str(header).startswith("_")
        }
        source_owned_bare_account = {"账号", "储种", "地区"}.issubset(source_headers)
        own_account = _exact_source_value(
            raw_txn,
            ("账号", "本方账号", "本方账户") if source_owned_bare_account else ("本方账号", "本方账户"),
        )
        if own_account not in (None, ""):
            normalized["own_account"] = re.sub(r"\s+", "", str(own_account))
        else:
            # Adding the shorter canonical header ``账号`` must not make the
            # base substring matcher copy an explicit ``对方账号`` into this
            # source-owned role.
            normalized["own_account"] = ""
        if deposit_type not in (None, ""):
            normalized["deposit_type"] = str(deposit_type).strip()
        if region_code not in (None, ""):
            normalized["region_code"] = re.sub(r"\s+", "", str(region_code))
        if source_owned_bare_account:
            normalized["sub_account"] = ""
            normalized["counter_party"] = ""
            normalized["counter_account"] = ""
            normalized["counter_bank_name"] = ""
            normalized["counter_bank_code"] = ""
        value_date = _exact_source_value(raw_txn, ("起息日", "起息日期", "Value Date"))
        if value_date not in (None, ""):
            normalized["value_date"] = _normalize_source_business_date(str(value_date))
        return normalized

    @property
    def identity_fields(self) -> Sequence[tuple[str, Sequence[str]]]:
        return BANK_IDENTITY_FIELDS

    def _extract_identity(self, parse_result) -> dict[str, dict]:
        """Keep transaction-body account references out of statement identity."""
        fields = super()._extract_identity(parse_result)
        account_detail = fields.get("account_number")
        if isinstance(account_detail, dict):
            raw_name = str(account_detail.get("raw_name") or "")
            if not _is_explicit_account_identity_label(raw_name):
                fields.pop("account_number", None)
        return fields

    def _canonical_raw_values(
        self,
        raw_txn: dict[str, Any],
        normalized: dict[str, Any],
    ) -> dict[str, Any]:
        """Map exact source cells to canonical roles without derived backfill."""
        values: dict[str, Any] = {}
        for canonical_name, mapping in self.column_registry.items():
            value = _exact_source_value(raw_txn, (canonical_name, *(mapping.aliases or [])))
            if value not in (None, ""):
                values[mapping.field] = value

        # These source columns deliberately combine multiple business roles.
        # Preserve the original cell in ``raw`` and expose only deterministic
        # source-backed substrings in ``canonical_raw``; never label the whole
        # compound as both a party and an account.
        compound_value = _exact_source_value(
            raw_txn,
            ("交易对手信息", "对方账号与户名", "对方户名/账号"),
        )
        if compound_value not in (None, ""):
            from docmirror.plugins.bank_statement.styles.grid_standard import _decompose_compound_counterparty

            values.pop("counter_party", None)
            values.pop("counter_account", None)
            values.pop("counter_bank_name", None)
            values.pop("counter_bank_code", None)
            values.update(_decompose_compound_counterparty(str(compound_value)))

        account_bank_value = _exact_source_value(
            raw_txn,
            ("对方账户/对方银行", "对方账号/对方银行"),
        )
        if account_bank_value not in (None, ""):
            from docmirror.plugins.bank_statement.styles.grid_standard import _decompose_account_and_bank

            values.pop("counter_account", None)
            values.pop("counter_bank_name", None)
            values.update(_decompose_account_and_bank(str(account_bank_value)))

        exact_party_value = _exact_source_value(
            raw_txn,
            ("对方户名", "对方账户名", "对方名称", "对手信息", "对手名称", "交易对方"),
        )
        if exact_party_value not in (None, ""):
            from docmirror.plugins.bank_statement.styles.grid_standard import _split_embedded_counter_account

            exact_party, embedded_account = _split_embedded_counter_account(str(exact_party_value))
            if embedded_account:
                values["counter_party"] = exact_party
                values["counter_account"] = embedded_account
                if account_bank_value not in (None, "") and not _decompose_account_and_bank(
                    str(account_bank_value)
                ):
                    # The exact party cell supplied the missing account suffix;
                    # under the adjacent exact compound header the remaining
                    # source cell is therefore the bank value, not an account.
                    values["counter_bank_name"] = str(account_bank_value).strip()

        source_headers = {
            re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(header or "")))
            for header in raw_txn
            if not str(header).startswith("_")
        }
        if not {"账号", "储种", "地区"}.issubset(source_headers):
            bare_account = _exact_source_value(raw_txn, ("账号",))
            if bare_account not in (None, ""):
                values.pop("own_account", None)
        cscb_headers = {
            "交易日期",
            "交易金额",
            "账户余额",
            "对方户名",
            "对方账号",
            "摘要/备注",
            "编号",
        }
        if cscb_headers.issubset(source_headers):
            summary = _exact_source_value(raw_txn, ("摘要/备注",))
            reference = _exact_source_value(raw_txn, ("编号",))
            values.pop("note", None)
            values.pop("timestamp", None)
            if summary not in (None, ""):
                values["summary"] = summary
            if reference not in (None, ""):
                values["reference"] = reference

        boc_headers = {
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
        if boc_headers.issubset(source_headers):
            from docmirror.plugins.bank_statement.styles.grid_standard import _decompose_boc_business_columns

            values.pop("summary", None)
            values.pop("posting_date", None)
            values.pop("purpose", None)
            values.pop("reference", None)
            values.update(_decompose_boc_business_columns(raw_txn, normalize_values=False))

        bojs_headers = {
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
        if bojs_headers.issubset(source_headers):
            from docmirror.plugins.bank_statement.styles.grid_standard import _decompose_bojs_summary

            values.pop("timestamp", None)
            values.pop("summary", None)
            values.pop("transaction_name", None)
            values.pop("transaction_code", None)
            values.pop("reference", None)
            date = _exact_source_value(raw_txn, ("交易日期",))
            direction = _exact_source_value(raw_txn, ("交易类型",))
            currency = _exact_source_value(raw_txn, ("币别",))
            if date not in (None, ""):
                values["date"] = date
            if direction not in (None, ""):
                values["direction"] = direction
            if currency not in (None, ""):
                values["currency"] = currency
            values.update(_decompose_bojs_summary(raw_txn, normalize_values=False))

        if "business_system_reference" not in values:
            from docmirror.plugins.bank_statement.styles.grid_standard import (
                _pab_labelled_fund_transfer_reference,
            )

            if transfer_reference := _pab_labelled_fund_transfer_reference(raw_txn):
                values["business_system_reference"] = transfer_reference

        # Split debit/credit columns are two independent source facts. Choose
        # the exact non-zero source cell only after its explicit direction was
        # resolved; never repair this later from normalized output.
        direction = str(normalized.get("direction") or "")
        amount_aliases = (
            ("收入", "收入金额", "贷方发生额", "贷方", "转入金额")
            if direction == "income"
            else ("支出", "支出金额", "借方发生额", "借方", "转出金额")
            if direction == "expense"
            else ()
        )
        source_amount = _exact_source_value(raw_txn, amount_aliases)
        if source_amount not in (None, ""):
            values["amount"] = source_amount
        return values

    def _recover_identity_from_evidence(self, parse_result) -> dict[str, dict[str, object]]:
        atoms_by_page = self._evidence_text_atoms_by_page(parse_result)
        if not atoms_by_page:
            return {}
        page_id = sorted(atoms_by_page)[0]
        atoms = sorted(
            atoms_by_page[page_id],
            key=lambda atom: (float(atom["bbox"][1]), float(atom["bbox"][0])),
        )
        identity_atoms = atoms
        header_top = _evidence_transaction_header_top(atoms)
        if header_top is not None:
            identity_atoms = [atom for atom in atoms if float(atom["bbox"][3]) <= header_top]
        text = " ".join(str(atom.get("text") or "").strip() for atom in identity_atoms)
        patterns = {
            "print_date": ("打印日期", r"打印日期\s*[:：]\s*(20\d{2}-\d{2}-\d{2})"),
            "query_period": (
                "交易时段",
                r"(?:交易时段|交易时间|起止日期)\s*[:：]\s*"
                r"(20\d{2}(?:-\d{2}-\d{2}|年\d{1,2}月\d{1,2}日)|20\d{6})\s*"
                r"(?:至|~|-)\s*"
                r"(20\d{2}(?:-\d{2}-\d{2}|年\d{1,2}月\d{1,2}日)|20\d{6})",
            ),
            "total_transactions": ("总条数", r"(?:总笔数|总条数)\s*[:：]\s*(\d+)"),
            "account_holder": (
                "客户名称",
                r"(?:户名|客户名称|客户姓名|账户名称)\s*[:：]\s*(.+?)"
                r"(?=\s*(?:开户机构|开户行|账号\s*/\s*卡号|卡号\s*/\s*账号|账号|卡号|"
                r"币种|年份|月份|验证码|结单号|客户行|"
                r"起始日期|终止日期|结束日期|交易日期)\s*[:：])",
            ),
            "account_number": (
                "账号",
                r"(?<!贷款)(?<!对方)(?:客户)?"
                r"(?:账\s*号(?:\s*/\s*卡\s*号)?|卡\s*号(?:\s*/\s*账\s*号)?)"
                r"\s*[:：]\s*([0-9*＊-]+)",
            ),
            "currency": ("币种", r"币种\s*[:：]\s*([^\s]+)"),
            "branch_name": (
                "开户行",
                r"开户行\s*(?:The Bank(?:\s+of Account Opening)?)?\s*"
                r"([\u4e00-\u9fa5]{2,30}银行[\u4e00-\u9fa5]{0,30}?)(?=\s+(?:客户号|Customer Number|账号|Account Number))",
            ),
            "bank_name": (
                "银行名称",
                r"(?:银行名称|Bank Name)\s*[:：]?\s*"
                r"([\u4e00-\u9fa5A-Za-z（）()·\s]{2,40}?(?:银行|Bank)[\u4e00-\u9fa5A-Za-z（）()·\s]{0,24}?)"
                r"(?=\s+(?:客户号|Customer Number|账号|Account Number|户名|币种)|$)",
            ),
        }
        recovered: dict[str, dict[str, object]] = {}
        for field_name, (label, pattern) in patterns.items():
            match = re.search(pattern, text)
            if not match:
                continue
            value = (
                " 至 ".join(_normalize_evidence_date(group) for group in match.groups())
                if field_name == "query_period"
                else match.group(1).strip()
            )
            if value:
                detail = self._evidence_identity_detail(field_name, label, value, page_id=page_id)
                if field_name == "currency" and "人民" in normalize_bank_matching_text(value):
                    detail["normalized_value"] = "CNY"
                recovered[field_name] = detail

        if "query_period" not in recovered:
            year_month = re.search(r"年份\s*[:：]\s*(\d{4}).{0,30}?月份\s*[:：]\s*(\d{1,2})", text)
            if year_month:
                year = int(year_month.group(1))
                month = int(year_month.group(2))
                if 1 <= month <= 12:
                    last_day = calendar.monthrange(year, month)[1]
                    value = f"{year:04d}-{month:02d}-01 至 {year:04d}-{month:02d}-{last_day:02d}"
                    recovered["query_period"] = self._evidence_identity_detail(
                        "query_period",
                        "年份/月",
                        value,
                        page_id=page_id,
                    )

        document_period = _evidence_document_query_period(atoms_by_page)
        if document_period is not None:
            period_value, period_page_ids, period_evidence_ids = document_period
            detail = self._evidence_identity_detail(
                "query_period",
                "起始日期/截止日期",
                period_value,
                page_id=period_page_ids[0],
                evidence_ids=period_evidence_ids,
            )
            detail["source_refs"] = [
                {"source": "canonical_evidence_atoms", "page_id": source_page_id} for source_page_id in period_page_ids
            ]
            recovered["query_period"] = detail

        if "total_transactions" not in recovered:
            expected_evidence = resolve_row_count_evidence(text)
            if is_authoritative_issuer_row_count(expected_evidence):
                recovered["total_transactions"] = self._evidence_identity_detail(
                    "total_transactions",
                    "directional_counts",
                    str(expected_evidence.count),
                    page_id=page_id,
                )

        def _right_nearby_value(label_text: str, predicate) -> dict[str, Any] | None:
            labels = [
                atom
                for atom in identity_atoms
                if re.fullmatch(
                    rf"{re.escape(label_text)}\s*[:：]?",
                    str(atom.get("text") or "").strip(),
                )
            ]
            candidates: list[tuple[float, dict[str, Any]]] = []
            for label_atom in labels:
                label_bbox = label_atom["bbox"]
                for candidate in atoms:
                    candidate_text = str(candidate.get("text") or "").strip()
                    candidate_bbox = candidate["bbox"]
                    if not predicate(candidate_text):
                        continue
                    if float(candidate_bbox[0]) <= float(label_bbox[2]):
                        continue
                    y_distance = abs(float(candidate_bbox[1]) - float(label_bbox[1]))
                    if y_distance > 15.0:
                        continue
                    x_distance = float(candidate_bbox[0]) - float(label_bbox[2])
                    candidates.append((y_distance + x_distance / 1000.0, candidate))
            return min(candidates, key=lambda item: item[0])[1] if candidates else None

        account_atom = next(
            (
                atom
                for label in ("银行账号", "账户账号", "账号/卡号", "卡号/账号", "账号")
                if (
                    atom := _right_nearby_value(
                        label,
                        lambda value: bool(re.fullmatch(r"[0-9*＊-]{8,40}", re.sub(r"\s+", "", value))),
                    )
                )
            ),
            None,
        )
        if account_atom is not None:
            account_value = re.sub(r"\s+", "", str(account_atom.get("text") or ""))
            recovered["account_number"] = self._evidence_identity_detail(
                "account_number",
                "账号",
                account_value,
                page_id=page_id,
                evidence_ids=[str(account_atom.get("id") or "")],
            )

        holder_atom = next(
            (
                atom
                for label in ("账户名称", "客户名称", "客户姓名", "户名")
                if (
                    atom := _right_nearby_value(
                        label,
                        lambda value: (
                            2 <= len(re.sub(r"\s+", "", value)) <= 80
                            and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", value))
                            and not any(
                                marker in value for marker in ("账号", "币种", "存款种类", "交易日期", "账户余额")
                            )
                            and normalize_bank_matching_text(value).upper() not in {"人民币", "CNY", "RMB"}
                        ),
                    )
                )
            ),
            None,
        )
        if holder_atom is not None:
            recovered["account_holder"] = self._evidence_identity_detail(
                "account_holder",
                "账户名称",
                str(holder_atom.get("text") or "").strip(),
                page_id=page_id,
                evidence_ids=[str(holder_atom.get("id") or "")],
            )

        currency_atom = _right_nearby_value(
            "币种",
            lambda value: (
                normalize_bank_matching_text(value).upper()
                in {"人民币", "CNY", "RMB", "美元", "USD", "港币", "HKD", "欧元", "EUR", "日元", "JPY"}
            ),
        )
        if currency_atom is not None:
            currency_value = str(currency_atom.get("text") or "").strip()
            detail = self._evidence_identity_detail(
                "currency",
                "币种",
                currency_value,
                page_id=page_id,
                evidence_ids=[str(currency_atom.get("id") or "")],
            )
            if normalize_bank_matching_text(currency_value).upper() in {"人民币", "CNY", "RMB"}:
                detail["normalized_value"] = "CNY"
            recovered["currency"] = detail

        issuer_atom = _right_nearby_value(
            "银行名称",
            lambda value: any(marker in value for marker in ("银行", "信用社", "信用合作联社")) and len(value) <= 40,
        )
        if issuer_atom is not None:
            recovered["bank_name"] = self._evidence_identity_detail(
                "bank_name",
                "银行名称",
                str(issuer_atom.get("text") or "").strip(),
                page_id=page_id,
                evidence_ids=[str(issuer_atom.get("id") or "")],
            )

        branch_match = next(
            (
                (label, atom)
                for label in ("开户行", "开户机构", "打印机构")
                if (
                    atom := _right_nearby_value(
                        label,
                        lambda value: (
                            any(marker in value for marker in ("银行", "信用社", "信用合作联社"))
                            and len(value) <= 40
                        ),
                    )
                )
            ),
            None,
        )
        if branch_match is not None:
            branch_label, branch_atom = branch_match
            recovered["branch_name"] = self._evidence_identity_detail(
                "branch_name",
                branch_label,
                str(branch_atom.get("text") or "").strip(),
                page_id=page_id,
                evidence_ids=[str(branch_atom.get("id") or "")],
            )
        count_atom = _right_nearby_value(
            "汇总交易笔数",
            lambda value: bool(re.fullmatch(r"\d+\s*笔", value)),
        )
        if count_atom is not None:
            count_value = re.sub(r"\D", "", str(count_atom.get("text") or ""))
            recovered["total_transactions"] = self._evidence_identity_detail(
                "total_transactions",
                "汇总交易笔数",
                count_value,
                page_id=page_id,
                evidence_ids=[str(count_atom.get("id") or "")],
            )
        title_atom = next(
            (
                atom
                for atom in atoms
                if any(
                    marker in str(atom.get("text") or "")
                    for marker in ("账户交易明细表", "电子对账单", "对公账户对账单", "企业账户对账单")
                )
            ),
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
        if not is_source_bound_issuer_detail(result.identity_fields.get("bank_name")):
            result.identity_fields.pop("bank_name", None)
        records = _sanitize_bank_records(result.records)
        result.emitted_rows = len(records)
        accounting_warnings = audit_row_accounting(
            parsed_rows=result.parsed_rows,
            canonical_rows=result.canonical_rows,
            emitted_rows=result.emitted_rows,
        )
        amount_warnings = audit_amount_consistency(records)
        if accounting_warnings or amount_warnings:
            result.style_meta.extract_status = "degraded"
        projection_warnings = [
            *result.warnings,
            *accounting_warnings,
            *amount_warnings,
        ]
        summary = self._build_summary(records)
        # Transaction bounds remain useful summary analytics, but they are not
        # an issuer-stated account period and must not be projected as one.
        period: dict[str, str] = {}
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
        extra_domain_facts = result.style_meta.to_properties()
        extra_domain_facts.pop("institution_hint", None)
        extra_domain_facts.pop("institution_authority", None)
        extra_domain_facts["extraction_route"] = result.extraction_route.value
        extra_domain_facts["data_dictionary"] = BANK_DATA_DICTIONARY
        bank_detail = result.identity_fields.get("bank_name")
        if isinstance(bank_detail, dict):
            bank_value = next(
                (
                    str(bank_detail.get(candidate) or "")
                    for candidate in ("normalized_value", "value", "raw_value")
                    if bank_detail.get(candidate) not in (None, "")
                ),
                "",
            )
            if bank_value:
                extra_domain_facts["institution_hint"] = bank_value
                extra_domain_facts["institution_authority"] = "identity.bank_name"
        source_reported_evidence = resolve_row_count_evidence(
            result.ctx.full_text,
            page_texts=page_texts_with_business_headers(
                parse_result,
                page_texts_from_parse_result(parse_result),
            ),
        )
        # Defend the public projection boundary independently of the shared
        # pipeline. Generic identity routes may observe a count label, but only
        # issuer evidence may publish an exact transaction total.
        result.identity_fields.pop("total_transactions", None)
        issuer_count_detail = authoritative_issuer_transaction_count_detail(source_reported_evidence)
        if issuer_count_detail is not None:
            result.identity_fields["total_transactions"] = issuer_count_detail
            extra_domain_facts["source_reported_transaction_count"] = source_reported_evidence.count
        statement_header_records = build_statement_header_records(
            parse_result,
            result.identity_fields,
        )
        records = attach_statement_context(records, statement_header_records)
        statement_header_records = reconcile_source_unitemized_residuals(
            parse_result,
            records,
            statement_header_records,
            source_route=result.extraction_route.value,
            selected_source=str(getattr(result.ctx.reconstruction, "source", "") or ""),
        )
        projection = self._projection_data_from_components(
            identity_fields=result.identity_fields,
            records=records,
            raw_headers=[],
            summary=summary,
            period=period,
            extra_domain_facts=extra_domain_facts,
            warnings=projection_warnings,
            confidence={"success": 1.0, "low_coverage": 0.65, "degraded": 0.35}.get(
                result.style_meta.extract_status,
                0.35,
            ),
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
        datasets: dict[str, list[dict[str, Any]]] = {}
        if statement_header_records:
            datasets["statement_header"] = statement_header_records
        datasets.update(projection.datasets)
        semantic = dict(projection.semantic)
        semantic["dataset_document_order"] = ["statement_header", "transactions"]
        return projection.model_copy(
            update={
                "entity_fields": entity_fields,
                "semantic": semantic,
                "datasets": datasets,
                "content_markdown_override": _render_bank_statement_content_markdown(
                    records,
                    identity_values,
                    period,
                    text,
                    document_type=str(getattr(parse_result.entities, "document_type", "") or ""),
                    source_pages=_parse_result_source_pages(parse_result),
                    source_headers=_parse_result_source_table_headers(parse_result),
                ),
            }
        )


def _render_bank_statement_content_markdown(
    records: list[dict],
    identity: dict[str, str],
    period: str | dict,
    source_text: str = "",
    *,
    document_type: str = "bank_statement",
    source_pages: dict[int, str] | None = None,
    source_headers: list[str] | None = None,
) -> str:
    """Render a record-complete bank statement Markdown view from canonical plugin facts."""
    if not records:
        return ""
    identity = dict(identity)
    if document_type == "bank_reconciliation" and not identity.get("statement_title"):
        identity["statement_title"] = "银行对账单"
    rows_by_page: dict[int, list[dict]] = {}
    for record in records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        page = int(source.get("source_page") or (source.get("page_range") or [1])[0] or 1)
        rows_by_page.setdefault(page, []).append(record)

    parts = ['<!-- docmirror:markdown-profile version="1.0" -->']
    source_pages = source_pages or {}
    page_numbers = sorted(set(rows_by_page) | set(source_pages)) or [1]
    positioned_headers = source_headers or []
    raw_headers = _raw_statement_table_headers(records, source_text, source_headers=positioned_headers)
    use_positioned_source_values = bool(positioned_headers and positioned_headers == raw_headers)
    for page in page_numbers:
        parts.append(f'<!-- docmirror:page logical="{page}" source="{page}" -->')
        page_records = rows_by_page.get(page, [])
        if raw_headers:
            page_source_text = source_pages.get(page, "")
            header_lines = _raw_statement_header_lines(
                identity,
                period,
                page_source_text or source_text,
                allow_identity_fallback=page == page_numbers[0],
            )
            statement_title = str(identity.get("statement_title") or "").strip()
            if (
                page == page_numbers[0]
                and statement_title
                and not any(statement_title in line for line in header_lines)
            ):
                header_lines.insert(0, statement_title)
            if header_lines:
                parts.append("  \n".join(header_lines))
            parts.append(
                _render_raw_statement_table(
                    page_records,
                    raw_headers,
                    allow_semantic_fallback=use_positioned_source_values,
                )
            )
            after_table_lines = (
                _source_statement_note_lines(page_source_text)
                if not page_records
                else _raw_statement_after_table_lines(page_source_text or source_text, page)
            )
            if after_table_lines:
                parts.append("  \n".join(after_table_lines))
        else:
            parts.append(f"## 第 {page} 页")
            if page == page_numbers[0]:
                parts.extend(_bank_statement_header_lines(identity, period))
            parts.append(_render_bank_statement_table(page_records))
    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def _parse_result_source_pages(parse_result) -> dict[int, str]:
    """Collect page-local source text for faithful non-transaction page rendering."""
    page_texts: dict[int, str] = {}
    for page in getattr(parse_result, "pages", []) or []:
        page_number = int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0)
        if page_number <= 0:
            continue
        parts = [
            str(getattr(block, "content", "") or "").strip()
            for block in getattr(page, "texts", []) or []
            if str(getattr(block, "content", "") or "").strip()
        ]
        for table in getattr(page, "tables", []) or []:
            for row in getattr(table, "rows", []) or []:
                cells = [str(getattr(cell, "text", "") or "").strip() for cell in getattr(row, "cells", []) or []]
                if any(cells):
                    parts.append(" ".join(cells))
        if parts:
            page_texts[page_number] = "\n".join(parts)
    return page_texts


def _parse_result_source_table_headers(parse_result) -> list[str]:
    """Recover a signed-ledger header order from page-local OCR coordinates."""
    entities = getattr(parse_result, "entities", None)
    domain_specific = getattr(entities, "domain_specific", None)
    if not isinstance(domain_specific, dict):
        return []
    for bundle in domain_specific.get("_page_evidence_bundles") or []:
        local = bundle.get("local_structure_evidence") if isinstance(bundle, dict) else None
        tokens = local.get("tokens") if isinstance(local, dict) else None
        headers = _positioned_signed_source_headers(tokens or [])
        if headers:
            return headers
    return []


def _sanitize_bank_records(records: list[dict]) -> list[dict]:
    """Sanitize derived display fields while preserving both source layers."""
    sanitized: list[dict] = []
    for record in records:
        copied = copy.deepcopy(dict(record))
        normalized = copied.get("normalized")
        if isinstance(normalized, dict):
            _sanitize_bank_value_pool(normalized)
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
        if key_text in {"counter_party", "对方户名", "对方名称", "对手信息", "对手名称", "交易对手"}:
            text = _clean_counterparty_text(text)
        pool[key] = text


def _repair_split_amount_canonical_raw(record: dict) -> None:
    canonical_raw = record.get("canonical_raw")
    raw = record.get("raw")
    normalized = record.get("normalized")
    if not isinstance(canonical_raw, dict) or not isinstance(raw, dict) or not isinstance(normalized, dict):
        return
    direction = str(normalized.get("direction") or "")
    if direction not in {"income", "expense"}:
        return
    keys = (
        ("收入", "收入金额", "贷方发生额", "贷方", "转入金额")
        if direction == "income"
        else ("支出", "支出金额", "借方发生额", "借方", "转出金额")
    )
    for value in _raw_values_matching_headers(raw, keys):
        cleaned = _clean_money_text(value)
        if not cleaned:
            continue
        try:
            source_amount = abs(float(cleaned.replace(",", "")))
            normalized_amount = abs(float(normalized.get("amount")))
        except (TypeError, ValueError):
            continue
        if abs(source_amount - normalized_amount) > 0.000001:
            continue
        canonical_raw["amount"] = cleaned
        if "amount_cny" in canonical_raw:
            canonical_raw["amount_cny"] = cleaned
        return


def _raw_values_matching_headers(raw: dict, aliases: tuple[str, ...]) -> list[str]:
    """Read values from exact or stacked bilingual source headers."""
    values: list[str] = []
    for alias in aliases:
        compact_alias = _compat_compact(alias)
        for raw_header, value in raw.items():
            if value in (None, ""):
                continue
            header_text = str(raw_header)
            compact_header = _compat_compact(header_text)
            header_parts = {_compat_compact(part) for part in header_text.splitlines()}
            if compact_header == compact_alias or compact_alias in header_parts:
                values.append(str(value))
    return values


def _repair_counterparty_canonical_raw(record: dict) -> None:
    """Preserve explicit stacked source party and institution values for audit."""
    canonical_raw = record.get("canonical_raw")
    raw = record.get("raw")
    if not isinstance(canonical_raw, dict) or not isinstance(raw, dict):
        return
    explicit_headers = {
        "对方户名": ("对方户名", "对方名称", "对手名称", "交易对方", "Counterparty Name", "对方账号与户名"),
        "对方行名": ("对方行名", "对手机构", "对方开户行", "对方银行名称", "Counterparty Institution"),
    }
    for canonical_header, aliases in explicit_headers.items():
        mapping = BANK_COLUMN_REGISTRY[canonical_header]
        values = _raw_values_matching_headers(raw, aliases)
        value = next((item for item in values if item.strip()), "")
        if value:
            canonical_raw[mapping.field] = _clean_counterparty_text(value)


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
    for pool_name in ("normalized",):
        pool = record.get(pool_name)
        if not isinstance(pool, dict):
            continue
        for key in ("对方户名", "对方名称", "交易对手", "counter_party"):
            value = pool.get(key)
            if not isinstance(value, str):
                continue
            cleaned = _clean_counterparty_text(value)
            polluted = _looks_like_counterparty_pollution(value) or _looks_like_counterparty_pollution(cleaned)
            residue = _looks_like_counterparty_residue(cleaned)
            if alias and cleaned and cleaned.startswith(alias):
                # The alias only shortens a value that is already present in
                # this source row.  It must never populate an empty/residual
                # party from another transaction that happens to share the
                # same counter-account.
                cleaned = alias
            elif polluted or residue:
                cleaned = ""
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
    if unicodedata.normalize("NFKC", text).casefold() == "null":
        return ""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff(（])", "", text)
    text = re.sub(r"(?<=[)）])\s+(?=[\u4e00-\u9fff])", "", text)
    if text in {"入", "收", "出", "支", "限公司", "有限公司", "代收）", "代收)"}:
        return ""
    text = _strip_counterparty_header_fragment(text)
    text = _strip_fee_tail(text)
    text = _strip_tax_escrow_tail(text)
    return text.strip()


def _strip_counterparty_header_fragment(value: str) -> str:
    compact = _compat_compact(value)
    markers = (
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
    )
    positions = [compact.find(marker) for marker in markers if compact.find(marker) >= 0]
    if not positions:
        return value
    prefix_len = min(positions)
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
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    compact = _compat_compact(value)
    long_number_count = len(re.findall(r"(?<!\d)\d{8,}(?!\d)", normalized))
    return (
        not compact
        or compact in {"入", "收", "出", "支", "限公司", "有限公司", "代收)", "代收）"}
        or "序号交易日期" in compact
        # One long identifier can be a legitimate party suffix; two usually mean
        # that an adjacent account or transaction row leaked into the name cell.
        or long_number_count > 1
        or bool(re.fullmatch(r"[\d*＊,./:：-]{8,}", compact))
        or sum(compact.count(marker) for marker in ("WL财付通", "WL支付宝", "微信转账")) > 2
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
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).strip()
    match = re.match(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    return match.group(0) if match else text


def _bank_statement_header_lines(identity: dict[str, str], period: str | dict) -> list[str]:
    statement_title = str(identity.get("statement_title") or "").strip()
    lines = [f"# {_markdown_cell(statement_title)}" if statement_title else "# 银行流水"]
    labels = [
        ("银行名称", identity.get("bank_name") or ""),
        ("开户行/客户行", identity.get("branch_name") or identity.get("bank_branch") or ""),
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
_RAW_TABLE_EXCLUDED_HEADERS = {"_style_id", "_source_page"}
# Vocabulary used only to detect source table header lines. Raw Markdown columns
# deliberately retain the insertion order of the source record.
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
_RAW_SPLIT_DIRECTION_KEYS = (
    "收入",
    "收入金额",
    "贷方发生额",
    "贷方",
    "转入金额",
    "支出",
    "支出金额",
    "借方发生额",
    "借方",
    "转出金额",
)
_HEADER_VALUE_KEYS = {
    "序号": (("序号", "日志号", "交易流水号", "流水号"), ("sequence_no",)),
    "日期": (("日期", "交易日期", "交易时间", "记账日期"), ("date",)),
    "交易日期": (("交易日期", "日期"), ("date",)),
    "交易时间": (("交易时间", "时间"), ("timestamp",)),
    "交易摘要": (("交易摘要", "摘要"), ("summary",)),
    "交易金额": (("交易金额", "金额"), ("amount",)),
    "借/贷方发生额": (("借/贷方发生额", "借贷方发生额", "交易金额", "金额"), ("amount",)),
    "本次余额": (("本次余额", "账户余额", "余额"), ("balance",)),
    "账户余额": (("账户余额", "本次余额", "余额"), ("balance",)),
    "余额": (("余额", "账户余额", "本次余额"), ("balance",)),
    "对手信息": (("对手信息", "对方户名", "对方账号"), ("counter_party", "counter_account")),
    "日志号": (("日志号", "日 志号", "序号", "交易流水号", "流水号"), ("sequence_no",)),
    "交易渠道": (("交易渠道", "渠道"), ("channel",)),
    "交易附言": (("交易附言", "附言", "用途"), ("purpose",)),
    "交易类别": (("交易类别", "交易类型", "收/支"), ("direction",)),
    "交易类型": (("交易类型", "交易类别", "收/支"), ("direction",)),
    "对方账号": (("对方账号", "对方账户"), ("counter_account",)),
    "对方账户": (("对方账户", "对方账号"), ("counter_account",)),
    "对方户名": (("对方户名", "对方名称", "交易对手"), ("counter_party",)),
    "传票号": (("传票号", "凭证号"), ("reference",)),
    "备注": (("备注", "摘要"), ("summary",)),
    "摘要/附言": (("摘要/附言", "摘要", "交易摘要", "备注"), ("summary",)),
    "币别": (("币别", "币种"), ("currency",)),
    "交易机构": (("交易机构", "机构"), ()),
}

_SIGNED_AMOUNT_HEADERS = ("借/贷方发生额", "借贷方发生额", "借贷发生额")
_SIGNED_SOURCE_HEADER_GROUPS = (
    ("序号", "流水号"),
    ("交易日期", "记账日期", "日期"),
    ("交易时间", "时间"),
    _SIGNED_AMOUNT_HEADERS,
    ("账户余额", "余额"),
    ("对方户名", "对方名称", "交易对手"),
    ("对方账户", "对方账号"),
    ("传票号", "凭证号"),
    ("交易摘要", "摘要"),
)


def _positioned_signed_source_headers(tokens: list[dict]) -> list[str]:
    """Return source headers ordered by their horizontal OCR positions."""
    candidates: list[tuple[float, float, float, int, str]] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        bbox = token.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        text = _compat_compact(token.get("text") or token.get("content"))
        for group_index, aliases in enumerate(_SIGNED_SOURCE_HEADER_GROUPS):
            matches = [alias for alias in aliases if alias in text]
            if not matches:
                continue
            x0, y0, _x1, y1 = (float(value) for value in bbox[:4])
            candidates.append((x0, (y0 + y1) / 2, max(y1 - y0, 1.0), group_index, max(matches, key=len)))
            break

    best: list[tuple[float, float, float, int, str]] = []
    for candidate in candidates:
        band = [other for other in candidates if abs(other[1] - candidate[1]) <= max(other[2], candidate[2])]
        unique = {item[3]: item for item in sorted(band, key=lambda item: abs(item[1] - candidate[1]))}
        positioned = sorted(unique.values(), key=lambda item: item[0])
        groups = {item[3] for item in positioned}
        if len(positioned) > len(best) and 3 in groups and 4 in groups and groups.intersection({1, 2}):
            best = positioned
    return [item[4] for item in best] if len(best) >= 5 else []


def _raw_statement_table_headers(
    records: list[dict],
    source_text: str = "",
    *,
    source_headers: list[str] | None = None,
) -> list[str]:
    """Return source-table headers when records carry a readable bank ledger shape."""
    if not records:
        return []
    source_headers = source_headers or _source_statement_table_headers(source_text)
    if source_headers and _records_support_source_headers(records, source_headers):
        return source_headers

    supporting_rows = 0
    for record in records[:20]:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        if _raw_record_has_ledger_shape(raw, normalized):
            supporting_rows += 1
    if supporting_rows:
        known_headers = _known_raw_source_headers(records)
        if known_headers:
            return known_headers
        return _generic_raw_statement_headers(records, source_text)
    return []


def _raw_record_has_ledger_shape(raw: dict, normalized: dict | None = None) -> bool:
    if not raw:
        return False
    normalized = normalized or {}
    has_required = all(_source_header_value(raw, normalized, header) for header in _RAW_REQUIRED_HEADERS)
    has_direction = any(raw.get(key) not in (None, "") for key in _RAW_DIRECTION_KEYS)
    has_direction = has_direction or any(raw.get(key) not in (None, "") for key in _RAW_SPLIT_DIRECTION_KEYS)
    has_direction = has_direction or normalized.get("direction") in {"income", "expense"}
    return has_required and has_direction


def _generic_raw_statement_headers(records: list[dict], source_text: str = "") -> list[str]:
    """Return source-backed raw columns without schema-padding placeholders."""
    present_headers: list[str] = []
    source_lines = {_compat_compact(line) for line in str(source_text or "").splitlines() if line.strip()}
    raw_rows = [_record_raw(record) for record in records]
    split_direction_present = any(
        any(raw.get(key) not in (None, "") for raw in raw_rows) for key in _RAW_SPLIT_DIRECTION_KEYS
    )
    for record in records[:20]:
        raw = _record_raw(record)
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        if not _raw_record_has_ledger_shape(raw, normalized):
            continue
        for key in raw:
            if key in _RAW_TABLE_EXCLUDED_HEADERS or str(key).startswith("_"):
                continue
            if key not in present_headers:
                has_value = any(row.get(key) not in (None, "") for row in raw_rows)
                explicitly_declared = _compat_compact(key) in source_lines
                paired_split_amount = key in _RAW_SPLIT_DIRECTION_KEYS and split_direction_present
                if has_value or explicitly_declared or paired_split_amount:
                    present_headers.append(key)
    return present_headers


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
    return [
        header
        for header in best_headers
        if any(_record_raw(record).get(header) not in (None, "") for record in records)
    ]


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
    compact_headers = [_compat_compact(header) for header in headers]
    if not any("日期" in header or "时间" in header for header in compact_headers):
        return False
    if not any("金额" in header or "发生额" in header for header in compact_headers):
        return False
    support = 0
    for record in records[:20]:
        raw = _record_raw(record)
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        if _source_header_value(raw, normalized, "交易日期") and _source_header_value(raw, normalized, "交易金额"):
            support += 1
    return support > 0


def _raw_statement_header_lines(
    identity: dict[str, str],
    period: str | dict,
    source_text: str,
    *,
    allow_identity_fallback: bool = True,
) -> list[str]:
    source_lines = _source_statement_header_block(source_text)
    source_identity = _compat_compact(" ".join(source_lines))
    has_empty_source_label = any(re.search(r"[:：]\s*$", line) for line in source_lines)
    if (
        source_lines
        and not has_empty_source_label
        and any(
            marker in source_identity
            for marker in ("客户名称", "账户名称", "账号", "账户", "户名", "起止日期", "账单统计日期")
        )
    ):
        return source_lines

    if not allow_identity_fallback:
        return source_lines

    source_title = _source_statement_title(source_text)
    identity_title = str(identity.get("statement_title") or "").strip()
    holder, branch = _holder_and_branch(identity)
    lines = [source_title] if source_title else ([f"# {identity_title}"] if identity_title else [])
    print_date = _source_label_value(source_text, "打印日期") or identity.get("print_date") or ""
    account_number = identity.get("account_number") or _source_label_value(source_text, "卡/账号") or ""
    if print_date:
        lines.append(f"打印日期：{_compact_date(print_date)}")
    if account_number:
        account_label = "卡/账号" if "卡/账号" in source_text else "账号"
        lines.append(f"{account_label}：{_markdown_cell(account_number)}")
    if holder:
        lines.append(f"户名：{_markdown_cell(holder)}")
    if branch:
        lines.append(f"开户行：{_markdown_cell(branch)}")
    currency = str(identity.get("currency") or "").strip()
    if currency:
        lines.append(f"币种：{_markdown_cell(currency)}")
    start, end = _period_bounds(period, source_text)
    if start or end:
        lines.append(f"账期：{_compact_date(start)} 至 {_compact_date(end)}")
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
            if candidate.count("|") >= 3:
                break
            if (
                _looks_like_source_table_header(candidate)
                or _looks_like_source_table_header_fragment(candidate)
                or _looks_like_source_table_row_ordinal(candidate)
                or _looks_like_transaction_line(candidate)
            ):
                break
            if _is_footer_line(candidate):
                break
            out.append(candidate)
        return _merge_source_page_number_lines(_merge_source_header_fragments(out))
    return []


def _merge_source_header_fragments(lines: list[str]) -> list[str]:
    """Repair common two-character KV labels split by OCR reading order."""
    merged = [line.strip() for line in lines]
    label_pairs = (("币种", "币", "种"), ("单位", "单", "位"))
    for canonical, first, second in label_pairs:
        first_indexes = [index for index, line in enumerate(merged) if _compat_compact(line) == first]
        value_matches = [
            (index, match)
            for index, line in enumerate(merged)
            if (match := re.fullmatch(rf"{re.escape(second)}\s*([:：])\s*(.+)", line)) is not None
        ]
        if not first_indexes or not value_matches:
            continue
        value_index, match = min(
            value_matches,
            key=lambda item: min(abs(item[0] - first_index) for first_index in first_indexes),
        )
        first_index = min(first_indexes, key=lambda item: abs(item - value_index))
        if abs(first_index - value_index) > 8:
            continue
        target = min(first_index, value_index)
        merged[target] = f"{canonical}{match.group(1)}{match.group(2).strip()}"
        merged[max(first_index, value_index)] = ""
    return [line for line in merged if line]


def _merge_source_page_number_lines(lines: list[str]) -> list[str]:
    """Coalesce a positioned ``第 N / M 页`` header split into text fragments."""
    merged: list[str] = []
    index = 0
    while index < len(lines):
        if (
            index + 2 < len(lines)
            and re.fullmatch(r"第\s*\d+", lines[index])
            and lines[index + 1] == "/"
            and re.fullmatch(r"\d+\s*页", lines[index + 2])
        ):
            merged.append(f"{lines[index]} / {lines[index + 2]}")
            index += 3
            continue
        merged.append(lines[index])
        index += 1
    return merged


def _source_statement_note_lines(page_text: str) -> list[str]:
    """Return source remarks from a page that contains no business transactions."""
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    starts = [
        index
        for index, line in enumerate(lines)
        if "提示" in line and ("Remarks" in line or _compat_compact(line).startswith("提示"))
    ]
    if not starts:
        return []
    # Page text can contain both native text blocks and reconstructed table text.
    # The final occurrence is normally the complete table-backed remarks block.
    start = starts[-1]

    generation_label = next(
        (line for line in lines if "Statement Generation Date" in line or "账单生成日期" in line), ""
    )
    source_dates = re.findall(r"20\d{2}[/.-]\d{2}[/.-]\d{2}", str(page_text or ""))
    generation_date = source_dates[-1] if source_dates else ""
    out: list[str] = []
    for line in lines[start:]:
        compact = _compat_compact(line)
        if re.fullmatch(r"_+", compact) or re.fullmatch(r"[_A-Z0-9]{8,}", compact):
            continue
        if "Statement Generation Date" in line or "账单生成日期" in line:
            continue
        if generation_date and line == generation_date:
            continue
        line = re.sub(r"(?<=/company)[ _]+e[ _]+bank(?=/)", "_e_bank", line)
        out.append(line.replace("_", r"\_"))
    if generation_label:
        out.append(f"{generation_label} {generation_date}".strip())
    return out


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


def _normalize_evidence_date(value: str) -> str:
    text = str(value or "").strip()
    chinese = re.fullmatch(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
    if chinese:
        return f"{int(chinese.group(1)):04d}-{int(chinese.group(2)):02d}-{int(chinese.group(3)):02d}"
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def _evidence_document_query_period(
    atoms_by_page: dict[str, list[dict[str, Any]]],
) -> tuple[str, list[str], list[str]] | None:
    """Aggregate issuer-stated page periods without using transaction dates."""
    periods: list[tuple[str, str, str, list[str]]] = []
    for page_id, page_atoms in sorted(atoms_by_page.items()):
        atoms = sorted(page_atoms, key=lambda atom: (float(atom["bbox"][1]), float(atom["bbox"][0])))
        header_top = _evidence_transaction_header_top(atoms)
        header_atoms = atoms if header_top is None else [atom for atom in atoms if float(atom["bbox"][3]) <= header_top]
        text = " ".join(str(atom.get("text") or "").strip() for atom in header_atoms)
        start_match = re.search(r"起始日期\s*[:：]\s*(20\d{6}|20\d{2}[-/]\d{2}[-/]\d{2})", text)
        end_match = re.search(r"(?:截止日期|终止日期)\s*[:：]\s*(20\d{6}|20\d{2}[-/]\d{2}[-/]\d{2})", text)
        if start_match is None or end_match is None:
            continue
        start = _normalize_evidence_date(start_match.group(1).replace("/", "-"))
        end = _normalize_evidence_date(end_match.group(1).replace("/", "-"))
        evidence_ids = [
            str(atom.get("id") or "")
            for atom in header_atoms
            if any(marker in str(atom.get("text") or "") for marker in ("起始日期", "截止日期", "终止日期"))
            and str(atom.get("id") or "")
        ]
        periods.append((start, end, page_id, evidence_ids))
    if not periods:
        return None
    starts = [period[0] for period in periods]
    ends = [period[1] for period in periods]
    page_ids = list(dict.fromkeys(period[2] for period in periods))
    evidence_ids = list(dict.fromkeys(evidence_id for period in periods for evidence_id in period[3]))
    return f"{min(starts)} 至 {max(ends)}", page_ids, evidence_ids


def _raw_statement_after_table_lines(source_text: str, page: int) -> list[str]:
    lines = [line.strip() for line in str(source_text or "").splitlines()]
    disclaimers = list(dict.fromkeys(line for line in lines if _is_statement_disclaimer(line)))
    for idx, line in enumerate(lines):
        if not _is_bank_footer_line(line):
            continue
        footer_page = _footer_page_number(line)
        if footer_page is None:
            footer_page = _footer_page_number(lines[idx + 1] if idx + 1 < len(lines) else "")
        if footer_page != page:
            continue
        out: list[str] = []
        for prev in reversed(lines[max(0, idx - 5) : idx]):
            if _is_statement_note_line(prev):
                out.insert(0, prev)
        out.extend(_raw_statement_footer_lines(source_text, page))
        out.extend(disclaimers)
        return list(dict.fromkeys(out))
    return list(dict.fromkeys([*_raw_statement_footer_lines(source_text, page), *disclaimers]))


def _raw_statement_footer_lines(source_text: str, page: int) -> list[str]:
    lines = [line.strip() for line in str(source_text or "").splitlines()]
    out = [line for line in lines if _is_statement_summary_line(line)]
    for idx, line in enumerate(lines):
        if not _is_bank_footer_line(line):
            continue
        if _footer_page_number(line) == page:
            prefix = next(
                (
                    previous.strip()
                    for previous in reversed(lines[max(0, idx - 2) : idx])
                    if "第" in previous and any(marker in previous for marker in ("银行", "回单", "对账单"))
                ),
                "",
            )
            out.append(f"{prefix}{line.strip()}" if prefix else line.strip())
            continue
        page_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if _footer_page_number(page_line) == page:
            out.extend([line.strip(), page_line])
    return list(dict.fromkeys(line for line in out if line))


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


def _source_raw_header_value(raw: dict, header: str) -> object:
    """Return only source-backed values for the original-table Markdown view."""
    raw_keys, _ = _HEADER_VALUE_KEYS.get(header, ((header,), ()))
    for key in (header, *raw_keys):
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return ""


def _looks_like_source_table_header(text: str) -> bool:
    compact = _compat_compact(text)
    if any(
        sum(_compat_compact(header) in compact for header in headers) >= 4 for headers in _SOURCE_TABLE_HEADER_LAYOUTS
    ):
        return True
    generic_headers = {header for group in _GENERIC_RAW_HEADER_ORDER for header in group}
    generic_headers.update({"序号", "流水号", "收入", "支出", "附言"})
    return sum(_compat_compact(header) in compact for header in generic_headers) >= 4


def _looks_like_source_table_header_fragment(text: str) -> bool:
    """Detect one or more source column labels emitted as a text block."""
    compact = _compat_compact(text)
    headers = {header for group in _GENERIC_RAW_HEADER_ORDER for header in group}
    headers.update(header for layout in _SOURCE_TABLE_HEADER_LAYOUTS for header in layout)
    headers.update({"序号", "流水号", "收入", "支出", "附言", "对手信息", "日志号"})
    compact_headers = {_compat_compact(header) for header in headers}
    if compact in compact_headers:
        return True
    if len(compact) >= 3 and any(header.startswith(compact) or header.endswith(compact) for header in compact_headers):
        return True
    if len(compact) <= 12 and not re.search(r"[:：]", text):
        table_markers = ("借贷", "收/支", "收支", "金额", "余额", "对方", "摘要", "序号", "流水号")
        if any(marker in compact for marker in table_markers) or compact.startswith("交易"):
            return True

    token_counts = [-1] * (len(compact) + 1)
    token_counts[0] = 0
    for start in range(len(compact)):
        count = token_counts[start]
        if count < 0:
            continue
        for header in compact_headers:
            if compact.startswith(header, start):
                end = start + len(header)
                token_counts[end] = max(token_counts[end], count + 1)
    return token_counts[-1] >= 2


def _looks_like_transaction_line(text: str) -> bool:
    normalized = _compat_text(text).strip()
    return bool(
        re.match(
            r"^(?:\d{1,7}\s+)?20\d{2}(?:\d{4}|[-/.]\d{1,2}[-/.]\d{1,2})(?:\s|$)",
            normalized,
        )
    )


def _looks_like_source_table_row_ordinal(text: str) -> bool:
    """Detect an isolated source row number emitted before the remaining cells."""
    return bool(re.fullmatch(r"\d{1,7}\s*[.、)]", _compat_text(text).strip()))


def _is_statement_title_line(text: str) -> bool:
    compact = _compat_compact(text)
    if (
        not compact
        or len(compact) > 60
        or _is_statement_disclaimer(text)
        or re.match(r"^\d+[、.．]", str(text or "").strip())
    ):
        return False
    if "对账单" in compact or any(marker in compact for marker in ("交易明细", "明细清单")):
        return True
    return "交易流水" in compact and any(marker in compact for marker in ("银行", "账户", "个人", "企业"))


def _is_statement_disclaimer(text: str) -> bool:
    compact = _compat_compact(text)
    return (
        ("数据缺失" in compact and "仅供参考" in compact)
        or compact.startswith("风险提示:")
        or "不具有法律效力" in compact
        or ("客户实际交易" in compact and "不符" in compact)
    )


def _is_statement_note_line(text: str) -> bool:
    compact = _compat_compact(text)
    return any(marker in compact for marker in ("截至打印时间", "无其他明细", "交易明细截止", "打印时间下方"))


def _is_statement_summary_line(text: str) -> bool:
    compact = _compat_compact(text)
    return any(
        marker in compact
        for marker in (
            "当前账单借方发生数",
            "当前账单贷方发生数",
            "本月累计借方发生额",
            "本月累计贷方发生额",
            "出单截至日期",
        )
    )


def _is_footer_line(text: str) -> bool:
    return _is_statement_disclaimer(text) or _is_bank_footer_line(text) or _footer_page_number(text) is not None


def _is_bank_footer_line(text: str) -> bool:
    compact = _compat_compact(text)
    if compact.startswith("@") and "银行" in compact:
        return True
    if _footer_page_number(text) is not None and re.search(r"页[/／]共?\d+", compact):
        return True
    return _footer_page_number(text) is not None and any(
        marker in compact for marker in ("本页支出合计", "本页收入合计", "本页交易笔数")
    )


def _footer_page_number(text: str) -> int | None:
    match = re.search(r"第\s*(\d+)\s*页", _compat_text(text))
    if match:
        return int(match.group(1))
    current_total = re.search(r"(?<!共)(\d+)\s*页\s*[/／]\s*共?\s*\d+\s*页?", _compat_text(text))
    return int(current_total.group(1)) if current_total else None


def _compat_compact(value: object) -> str:
    return re.sub(r"\s+", "", _compat_text(value))


def _compat_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or ""))


def _render_raw_statement_table(
    records: list[dict],
    headers: list[str],
    *,
    allow_semantic_fallback: bool = False,
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        raw = _record_raw(record)
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        values = [
            _raw_markdown_cell(
                _source_markdown_value(raw, normalized, header, allow_semantic_fallback=allow_semantic_fallback)
            )
            for header in headers
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _source_markdown_value(
    raw: dict,
    normalized: dict,
    header: str,
    *,
    allow_semantic_fallback: bool,
) -> object:
    value = _source_raw_header_value(raw, header)
    if allow_semantic_fallback and _compat_compact(header) in _SIGNED_AMOUNT_HEADERS:
        if value in (None, ""):
            value = _source_header_value(raw, normalized, header)
        text = str(value or "").strip()
        if text and not text.startswith(("+", "-")):
            direction = str(normalized.get("direction") or "")
            text = ("+" if direction == "income" else "-" if direction == "expense" else "") + text
        return text
    if value not in (None, "") or not allow_semantic_fallback:
        return value
    return _source_header_value(raw, normalized, header)


def _record_raw(record: dict) -> dict:
    raw = record.get("raw")
    return raw if isinstance(raw, dict) else {}


def _raw_markdown_cell(value: object) -> str:
    text = str(value or "").replace("|", "\\|").strip()
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return ""
    out = parts[0]
    numeric_fragment = re.compile(r"^[+-]?[\d,.*-]+$")
    for part in parts[1:]:
        continues_wrapped_text = bool(
            re.search(r"[\u3400-\u9fff]$", out) and re.match(r"^[\u3400-\u9fff(（]", part)
        ) or bool(re.search(r"[)）]$", out) and re.match(r"^[\u3400-\u9fff]", part))
        join_without_space = continues_wrapped_text or bool(
            numeric_fragment.fullmatch(out) and numeric_fragment.fullmatch(part)
        )
        out += part if join_without_space else f" {part}"
    return out


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
            _display_direction(
                _first_raw_value(raw, "收/支", "交易类型", "交易类别")
                or _first_value(raw, normalized, "收/支", "direction")
            ),
            _display_amount(raw, normalized),
            _first_value(raw, normalized, "余额", "balance"),
            _first_value(raw, normalized, "对方户名", "counter_party"),
            _record_counter_account(record),
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
