"""Audit-specific table projection and conservative cross-page recovery."""

from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from docmirror.models.entities.parse_result import CellValue, TableBlock, TableRow
from docmirror.plugins._base.financial_source_projection import (
    ProjectedSegment,
    add_review_reason,
    amount_like,
    data_dictionary,
    normalize_scalar,
    row_cells_by_column,
    table_width,
)
from docmirror.plugins.financial_statement.projection import project_statement_table

_AMOUNT_HEADER_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_AMOUNT_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}")
_FULL_AMOUNT_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}$")
_NOTE_RE = re.compile(r"[一二三四五六七八九十百]+[、.．]\s*\d{1,3}")
_PERIOD_HEADER_RE = re.compile(r"^20\d{2}年(?:\d{1,2}月\d{1,2}日|度)$")
_TABLE_HEADER_MARKERS = re.compile(r"项目|名称|期末|期初|本期|上期|余额|金额|比例|账面")
_GENERIC_TABLE_DATASET_RE = re.compile(r"^(?:table_\d+(?:__split_\d+)?|records)$")
_AUDIT_NOTE_HEADING_RE = re.compile(r"^(?:[一二三四五六七八九十百]{1,4}|\d{1,2})[、.．]\s*\S")
_AUDIT_TABLE_CAPTION_RE = re.compile(
    r"列示|披露|分类|情况|明细|账龄|组合|性质|变动|计提|余额|构成|投资|交易|控制人|子公司|关联方"
)
_PROMOTED_HEADER_MARKERS = re.compile(
    r"项目|名称|单位|期末|期初|本期|上期|余额|金额|比例|账龄|增减变动|投资|损益|收益|日期|起始|到期|是否"
)
_OWNER_EQUITY_MARKERS = re.compile(
    r"所有者权益|股东权益|实收资本|股本|资本公积|其他综合收益|盈余公积|未分配利润|期初余额|期末余额"
)
_OWNER_EQUITY_ROW_MARKERS = re.compile(r"余额|变动|收益|投入|分配|结转|资本|公积|权益|政策|差错|提取|利润|其他")
_OWNER_EQUITY_COLUMN_LABELS = {
    "paid_in_capital": "实收资本（或股本）",
    "preferred_shares": "优先股",
    "perpetual_bonds": "永续债",
    "other_equity_instruments": "其他权益工具",
    "capital_reserve": "资本公积",
    "treasury_shares": "减：库存股",
    "other_comprehensive_income": "其他综合收益",
    "special_reserve": "专项储备",
    "surplus_reserve": "盈余公积",
    "retained_earnings": "未分配利润",
    "total_equity": "所有者权益合计",
}
_STATEMENT_TITLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:合并(?:及母公司)?|母公司)?资产负债表(?:\(续\))?$"), "balance_sheet"),
    (re.compile(r"^(?:合并(?:及母公司)?|母公司)?利润表(?:\(续\))?$"), "income_statement"),
    (re.compile(r"^(?:合并(?:及母公司)?|母公司)?现金流量表(?:\(续\))?$"), "cash_flow_statement"),
    (
        re.compile(
            r"^(?:合并(?:及母公司)?|母公司)?(?:所有者(?:\(股东\))?权益|所有者权益\(股东权益\)|股东权益)"
            r"变动表(?:\(续\))?$"
        ),
        "owners_equity_changes",
    ),
)
_RADICAL_TRANSLATION = str.maketrans(
    {
        "⺠": "民",
        "⻅": "见",
        "⻛": "风",
        "⻓": "长",
        "⻢": "马",
        "⻜": "飞",
        "⻔": "门",
        "⻩": "黄",
        "⻄": "西",
        "⻋": "车",
        "⻮": "齿",
    }
)
_STATEMENT_KINDS = {"balance_sheet", "income_statement", "cash_flow_statement", "owners_equity_changes"}
_ASCII_FIELD_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_CJK_LAYOUT_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")
_AUDIT_COLUMN_ALIASES = {
    "项目": "item",
    "名称": "name",
    "类别": "category",
    "账龄": "aging_bucket",
    "投资方": "investor",
    "股份数(万股)": "share_count_ten_thousand",
    "持股比例(%)": "shareholding_ratio",
    "确定组合的依据": "portfolio_basis",
    "损失准备的计提方法": "loss_allowance_method",
    "资产类别": "asset_category",
    "预计使用寿命(年)": "estimated_useful_life_years",
    "预计净残值率(%)": "estimated_residual_value_ratio",
    "年折旧率(%)": "annual_depreciation_ratio",
    "税种": "tax_type",
    "计税依据": "tax_basis",
    "税率(%)": "tax_rate",
    "单位名称": "entity_name",
    "预付对象": "prepayment_counterparty",
    "款项性质": "payment_nature",
    "被投资单位": "investee",
    "股东名称": "shareholder_name",
    "办公设备": "office_equipment",
    "运输设备": "transportation_equipment",
    "房屋建筑物": "buildings",
    "软件": "software",
    "实际控制人": "ultimate_controller",
    "认缴出资额(万元)": "subscribed_capital_ten_thousand_yuan",
    "子公司名称": "subsidiary_name",
    "主要经营地": "principal_place_of_business",
    "注册地": "registered_address",
    "业务性质": "business_nature",
    "直接持股比例(%)": "direct_shareholding_ratio",
    "间接持股比例(%)": "indirect_shareholding_ratio",
    "合计持股比例(%)": "total_shareholding_ratio",
    "关联方": "related_party",
    "关联方名称": "related_party_name",
    "其他关联方名称": "other_related_party_name",
    "其他关联方与本企业关系": "related_party_relationship",
    "关联交易内容": "related_transaction_description",
    "对本公司的持股比例(%)": "company_shareholding_ratio",
    "对本公司的表决权比例(%)": "company_voting_rights_ratio",
    "合计": "total",
    "坏账准备": "loss_allowance",
    "坏账准备期末余额": "loss_allowance_ending_balance",
    "减值准备期末余额": "impairment_allowance_ending_balance",
    "期初余额": "opening_balance",
    "期末余额": "ending_balance",
    "期初账面余额": "opening_book_balance",
    "期末账面余额": "ending_book_balance",
    "本期": "current_period",
    "上期": "previous_period",
    "本期发生额": "current_period_amount",
    "上期发生额": "previous_period_amount",
    "本期增加": "current_period_increase",
    "本期增加额": "current_period_increase",
    "本期减少": "current_period_decrease",
    "本期减少额": "current_period_decrease",
    "本期摊销": "current_period_amortization",
    "其他减少": "other_decrease",
}
_AUDIT_COLUMN_COMPONENT_ALIASES = {
    **_AUDIT_COLUMN_ALIASES,
    "金额": "amount",
    "比例(%)": "ratio",
    "账面余额": "book_balance",
    "账面价值": "carrying_amount",
    "坏账准备": "loss_allowance",
    "计提比例(%)": "provision_ratio",
    "递延所得税资产": "deferred_tax_asset",
    "递延所得税负债": "deferred_tax_liability",
    "可抵扣暂时性差异": "deductible_temporary_difference",
    "应纳税暂时性差异": "taxable_temporary_difference",
    "应收账款": "accounts_receivable",
    "收入": "revenue",
    "成本": "cost",
    "本期变动金额": "current_period_change_amount",
    "核销": "write_off",
    "计提": "provision",
    "收回或转回": "reversal",
    "本期增减变动": "current_period_changes",
    "追加投资": "additional_investment",
    "减少投资": "investment_reduction",
    "权益法下确认的投资损益": "equity_method_investment_income",
    "其他综合收益调整": "other_comprehensive_income_adjustment",
    "其他权益变动": "other_equity_changes",
    "宣告发放现金股利或利润": "declared_cash_dividends_or_profit",
    "计提减值准备": "impairment_provision",
    "其他": "other",
}
_SHARE_RATIO_SUBJECTS = {
    "预付款项": "prepayments",
    "应收账款": "accounts_receivable",
    "其他应收": "other_receivables",
    "其他应收款": "other_receivables",
}
_NOTE_SUBJECT_SLUGS: tuple[tuple[str, str], ...] = (
    ("一年内到期的非流动负债", "current_portion_of_non_current_liabilities"),
    ("历史沿革及资本变更情况", "capital_change_history"),
    ("本公司的实际控制人情况", "ultimate_controller"),
    ("营业收入和营业成本", "operating_revenue_and_cost"),
    ("本公司的子公司情况", "subsidiaries"),
    ("长期待摊费用", "long_term_deferred_expenses"),
    ("长期股权投资", "long_term_equity_investments"),
    ("其他关联方情况", "other_related_parties"),
    ("关联交易情况", "related_party_transactions"),
    ("金融资产减值", "financial_asset_impairment"),
    ("递延所得税资产", "deferred_tax"),
    ("应付职工薪酬", "employee_benefits_payable"),
    ("其他流动资产", "other_current_assets"),
    ("其他流动负债", "other_current_liabilities"),
    ("信用减值损失", "credit_impairment_losses"),
    ("承诺及或有事项", "commitments_and_contingencies"),
    ("货币资金", "cash_and_cash_equivalents"),
    ("其他应收款", "other_receivables"),
    ("其他应付款", "other_payables"),
    ("使用权资产", "right_of_use_assets"),
    ("无形资产", "intangible_assets"),
    ("短期借款", "short_term_borrowings"),
    ("合同负债", "contract_liabilities"),
    ("应付票据", "notes_payable"),
    ("应付账款", "accounts_payable"),
    ("应收账款", "accounts_receivable"),
    ("预付款项", "prepayments"),
    ("应交税费", "taxes_payable"),
    ("租赁负债", "lease_liabilities"),
    ("实收资本", "paid_in_capital"),
    ("资本公积", "capital_reserve"),
    ("盈余公积", "surplus_reserve"),
    ("未分配利润", "retained_earnings"),
    ("税金及附加", "taxes_and_surcharges"),
    ("销售费用", "selling_expenses"),
    ("管理费用", "administrative_expenses"),
    ("研发费用", "research_and_development_expenses"),
    ("财务费用", "finance_expenses"),
    ("其他收益", "other_income"),
    ("投资收益", "investment_income"),
    ("营业外收入", "non_operating_income"),
    ("营业外支出", "non_operating_expenses"),
    ("所得税费用", "income_tax_expense"),
    ("固定资产", "fixed_assets"),
    ("税项", "taxes"),
)
_BALANCE_SECTION_ANCHORS = {
    "流动资产": ("货币资金", "交易性金融资产"),
    "非流动资产": ("债权投资", "可供出售金融资产", "长期应收款", "长期股权投资"),
    "流动负债": ("短期借款", "交易性金融负债", "应付票据"),
    "非流动负债": ("长期借款", "应付债券", "租赁负债"),
    "所有者(股东)权益": ("实收资本(股本)", "实收资本", "股本"),
}
_STANDARD_BALANCE_ITEMS = (
    "交易性金融负债",
    "衍生金融负债",
    "应付票据",
)
_DATASET_BLOCKING_WARNING_PREFIXES = (
    "AUDIT_FINANCIAL_STATEMENT_UNRESOLVED",
    "AUDIT_OWNER_EQUITY_UNRESOLVED",
    "AUDIT_AMOUNT_SPLIT_INFERRED",
    "AUDIT_AMOUNT_FRAGMENT_INFERRED",
    "AUDIT_AMOUNT_SPLIT_UNRESOLVED",
    "AUDIT_STATEMENT_AMOUNT_COLUMNS_EMPTY",
    "AUDIT_STATEMENT_ITEM_MISSING",
    "AUDIT_MERGED_ITEM_ROWS_INFERRED",
    "AUDIT_COLUMN_SCHEMA_UNRESOLVED",
    "AUDIT_COLUMN_SCHEMA_COLLISION",
    "AUDIT_NUMERIC_COLUMN_NAME",
    "AUDIT_GENERIC_COLUMNS_EXCESSIVE",
    "AUDIT_EVIDENCE_MISSING",
    "AUDIT_CANONICAL_RAW_MISSING",
    "AUDIT_FIELD_SOURCE_MISSING",
    "AUDIT_NORMALIZED_NUMERIC_MISMATCH",
    "AUDIT_NOTE_SOURCE_COLUMN_OMITTED",
    "AUDIT_NOTE_SOURCE_CELL_OMITTED",
    "AUDIT_CROSS_PAGE_ORPHAN_ROW_UNRESOLVED",
)
_DOCUMENT_BLOCKING_WARNING_PREFIXES = (
    "AUDIT_SECTION_MISSING",
    "AUDIT_DOCUMENT_NUMBER_MISSING",
    "AUDIT_DOCUMENT_NUMBER_CONFLICT",
    "AUDIT_NORMALIZED_GLYPH_VARIANT_REMAINS",
    *_DATASET_BLOCKING_WARNING_PREFIXES,
)


@dataclass(frozen=True)
class _SourceTable:
    table: Any
    page: int
    index: int


@dataclass(frozen=True)
class _OrphanTotalCandidate:
    dataset_name: str
    keys: tuple[str, ...]
    source_table: _SourceTable
    rows: tuple[Any, ...]


@dataclass(frozen=True)
class _PhysicalRowMatch:
    source_table: _SourceTable
    row: Any
    row_index: int
    source_row_index: int
    matched_values: int
    coverage: float
    header_score: int


@dataclass(frozen=True)
class _RecoveredCategory:
    value: str
    page: int
    bbox: tuple[float, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    table_id: str = "canonical_text"
    source_row_index: int = -1


@dataclass(frozen=True)
class _TextAtom:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    evidence_ids: tuple[str, ...]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass(frozen=True)
class _StatementCandidate:
    mode: str
    segment: ProjectedSegment
    warnings: tuple[str, ...]


def normalize_audit_text(value: Any) -> str:
    """Normalize layout glyphs used by audit PDFs while preserving raw fields."""

    normalized = _normalize_audit_text(value, unicode_form="NFKC")
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_audit_display_text(value: Any) -> str:
    """Normalize damaged CJK glyphs without folding source presentation characters."""

    source = unicodedata.normalize("NFC", str(value or ""))
    normalized = "".join(
        normalize_audit_text(character) if 0x2E80 <= ord(character) <= 0x2FFF else character
        for character in source
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_audit_text(value: Any, *, unicode_form: str) -> str:
    return unicodedata.normalize(unicode_form, str(value or "")).translate(_RADICAL_TRANSLATION)


def normalize_audit_value(value: Any) -> str:
    """Remove PDF layout whitespace from a normalized audit value."""

    return _normalize_audit_value(value, text_normalizer=normalize_audit_text)


def normalize_audit_display_value(value: Any) -> str:
    """Remove PDF layout whitespace while retaining source-compatible symbols."""

    return _normalize_audit_value(value, text_normalizer=normalize_audit_display_text)


def _normalize_audit_value(value: Any, *, text_normalizer: Callable[[Any], str]) -> str:
    protected = str(value or "").replace("（", "\ue000").replace("）", "\ue001").replace("：", "\ue002")
    normalized = text_normalizer(protected).replace("\ue000", "（").replace("\ue001", "）").replace("\ue002", "：")
    normalized = _CJK_LAYOUT_SPACE_RE.sub("", normalized)
    normalized = re.sub(r"(?<=[年月第])\s+(?=\d)", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=(?:年|月|日|号|个|%|％))", "", normalized)
    normalized = re.sub(r"(?<=[\u3400-\u9fff\d])\s+(?=[（(/，,。；;：:%％])", "", normalized)
    normalized = re.sub(r"(?<=[）)/，,。；;：:%％])\s+(?=[\u3400-\u9fff\d])", "", normalized)
    normalized = re.sub(r"(?<=[\u3400-\u9fff\d])\s+(?=--)", "", normalized)
    return normalized


def normalize_audit_label(value: Any) -> str:
    """Return a compact human label without changing the preserved source header."""

    label = normalize_audit_value(value)
    if _CJK_RE.search(label):
        label = re.sub(r"\s+", "", label)
    return re.sub(r"^(\d{1,2})(?!\d)(?=[\u3400-\u9fff])", r"\1、", label)


def _audit_column_key(label: str) -> str:
    normalized = normalize_audit_label(label)
    if _ASCII_FIELD_KEY_RE.fullmatch(normalized):
        return normalized
    lookup = normalized.replace("（", "(").replace("）", ")")
    direct = _AUDIT_COLUMN_ALIASES.get(lookup)
    if direct:
        return direct
    share_ratio = re.fullmatch(r"占(.+?)(?:期末余额)?合计数的?比例\(%\)", lookup)
    if share_ratio:
        subject = _SHARE_RATIO_SUBJECTS.get(share_ratio.group(1))
        if subject:
            return f"{subject}_share_ratio"
    components = [_AUDIT_COLUMN_COMPONENT_ALIASES.get(part) for part in lookup.split("/")]
    return "_".join(components) if components and all(components) else ""


def project_embedded_financial_statements(parse_result: Any) -> tuple[list[ProjectedSegment], list[str]]:
    """Project main statements using the existing financial plugin's public projector."""

    segments: list[ProjectedSegment] = []
    warnings: list[str] = []
    occurrences: dict[str, int] = {}
    financial_pages = embedded_financial_pages(parse_result)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        kind = _page_statement_kind(page, parse_result=parse_result, financial_pages=financial_pages)
        if kind is None:
            continue
        occurrence_start = occurrences.get(kind, 0)
        page_segments, page_warnings = _project_statement_page(
            parse_result,
            page,
            kind=kind,
            occurrence_start=occurrence_start,
        )
        if not page_segments:
            warnings.append(f"AUDIT_FINANCIAL_STATEMENT_UNRESOLVED:page={page_number}:kind={kind}")
            warnings.extend(page_warnings)
            continue
        segments.extend(page_segments)
        warnings.extend(page_warnings)
        occurrences[kind] = occurrence_start + len(page_segments)
    segments = _merge_statement_segments(segments)
    warnings.extend(_unresolved_landscape_pages(parse_result, segments))
    return segments, list(dict.fromkeys(warnings))


def recover_owner_equity_label_rows(
    segments: list[ProjectedSegment],
    parse_result: Any,
) -> list[str]:
    """Restore source-confirmed owner-equity label rows omitted by sparse table reconstruction."""

    segment = next((item for item in segments if item.kind == "owners_equity_changes"), None)
    if segment is None:
        return []
    page_index = {
        int(getattr(page, "page_number", 0) or 1): page for page in getattr(parse_result, "pages", None) or []
    }
    recovered = 0
    for page_number in sorted(dataset_pages(segment.records)):
        page_records = [
            record for record in segment.records if int((record.get("source") or {}).get("page") or 0) == page_number
        ]
        items = [normalize_audit_text((record.get("raw") or {}).get("item")) for record in page_records]
        compact_items = [item.replace(" ", "") for item in items]
        if not any("2.其他权益工具持有者投入资本" in item for item in compact_items):
            continue
        successor = next(
            (
                record
                for record in page_records
                if re.match(r"^4[.．、]其他$", normalize_audit_text((record.get("raw") or {}).get("item")))
            ),
            None,
        )
        if successor is None or any("股份支付计入所有者权益的金额" in item for item in compact_items):
            continue
        recovered_source = _owner_equity_label_source(
            page_index.get(page_number),
            parse_result=parse_result,
            page_number=page_number,
        )
        if recovered_source is None:
            continue
        insert_at = segment.records.index(successor)
        period_role = next(
            (
                str((record.get("normalized") or {}).get("period_role"))
                for record in page_records
                if (record.get("normalized") or {}).get("period_role")
            ),
            "",
        )
        keys = [column.key for column in segment.columns if column.key != "period_role"]
        raw = {key: "" for key in keys}
        raw["item"] = recovered_source.value
        reference: dict[str, Any] = {
            "page": page_number,
            "field_name": "item",
            "source": "canonical_text",
            "table_id": recovered_source.table_id,
            "row": recovered_source.source_row_index,
        }
        if recovered_source.bbox:
            reference["bbox"] = list(recovered_source.bbox)
        if recovered_source.evidence_ids:
            reference["evidence_ids"] = list(recovered_source.evidence_ids)
        record = {
            "record_id": f"{segment.dataset_id}:recovered:{page_number}",
            "raw": raw,
            "canonical_raw": dict(raw),
            "normalized": {**{key: None for key in keys}, "item": normalize_audit_text(recovered_source.value)},
            "source": {
                "page": page_number,
                "page_range": [page_number, page_number],
                "table_id": recovered_source.table_id,
                "source_row_index": recovered_source.source_row_index,
                "source_resolution": "canonical_text_label_row",
                "source_cell_refs": [reference],
                **({"evidence_ids": list(recovered_source.evidence_ids)} if recovered_source.evidence_ids else {}),
            },
            "source_cell_refs": [copy.deepcopy(reference)],
            "confidence": 0.95 if recovered_source.evidence_ids else 0.75,
        }
        if period_role:
            _set_statement_period_role(record, period_role)
        if recovered_source.evidence_ids:
            record["evidence_ids"] = list(recovered_source.evidence_ids)
        segment.records.insert(insert_at, record)
        segment.source_row_refs.insert(
            min(insert_at, len(segment.source_row_refs)),
            {
                "page": page_number,
                "table_id": recovered_source.table_id,
                "source_row_index": recovered_source.source_row_index,
            },
        )
        recovered += 1
    if recovered:
        _reindex_dataset_records(segment.records, segment.dataset_id)
    return [f"AUDIT_OWNER_EQUITY_LABEL_ROWS_RECOVERED:rows={recovered}"] if recovered else []


def _owner_equity_label_source(
    page: Any | None,
    *,
    parse_result: Any,
    page_number: int,
) -> _RecoveredCategory | None:
    if page is None:
        return None
    pattern = re.compile(r"3\s*[.．、]\s*股份支付计入所有者权益的金额")
    logical_tables = [
        table
        for table in getattr(parse_result, "logical_tables", None) or []
        if page_number in {int(value) for value in (getattr(table, "source_pages", None) or []) if int(value) > 0}
    ]
    for table in [*(getattr(page, "tables", None) or []), *logical_tables]:
        table_id = str(getattr(table, "table_id", "") or getattr(table, "logical_id", "") or "source_table")
        for row_index, row in enumerate(getattr(table, "rows", None) or []):
            source_page = int(getattr(row, "source_page", 0) or page_number)
            if source_page != page_number:
                continue
            cells = list(getattr(row, "cells", None) or [])
            joined = normalize_audit_text("".join(str(getattr(cell, "text", "") or "") for cell in cells))
            match = pattern.search(joined)
            if match is None:
                continue
            cell_boxes = [getattr(cell, "bbox", None) for cell in cells if getattr(cell, "bbox", None)]
            bbox = (
                (
                    min(float(value[0]) for value in cell_boxes),
                    min(float(value[1]) for value in cell_boxes),
                    max(float(value[2]) for value in cell_boxes),
                    max(float(value[3]) for value in cell_boxes),
                )
                if cell_boxes
                else ()
            )
            evidence_ids = tuple(
                dict.fromkeys(
                    str(value) for cell in cells for value in (getattr(cell, "evidence_ids", None) or []) if value
                )
            )
            return _RecoveredCategory(
                value=normalize_audit_text(match.group()),
                page=page_number,
                bbox=bbox,
                evidence_ids=evidence_ids,
                table_id=table_id,
                source_row_index=int(getattr(row, "source_row_index", row_index) or row_index),
            )
    for block_index, block in enumerate(getattr(page, "texts", None) or []):
        match = pattern.search(normalize_audit_text(getattr(block, "content", "")))
        if match is None:
            continue
        return _RecoveredCategory(
            value=normalize_audit_text(match.group()),
            page=int(getattr(page, "page_number", 0) or 1),
            bbox=tuple(getattr(block, "bbox", None) or ()),
            evidence_ids=tuple(str(value) for value in (getattr(block, "evidence_ids", None) or []) if value),
            source_row_index=block_index,
        )
    return None


def _page_statement_kind(page: Any, *, parse_result: Any, financial_pages: set[int]) -> str | None:
    kind = statement_kind(page)
    page_number = int(getattr(page, "page_number", 0) or 1)
    if kind is None and page_number in financial_pages and _landscape_table_candidate(page, parse_result=parse_result):
        return "owners_equity_changes"
    return kind


def _project_statement_page(
    parse_result: Any,
    page: Any,
    *,
    kind: str,
    occurrence_start: int,
) -> tuple[list[ProjectedSegment], list[str]]:
    page_number = int(getattr(page, "page_number", 0) or 1)
    continuation = _statement_continuation(page, kind)
    segments, warnings = _project_statement_sources(
        getattr(page, "tables", None) or [],
        page_number=page_number,
        kind=kind,
        occurrence_start=occurrence_start,
        continuation=continuation,
        logical=False,
    )
    if not segments:
        segments, warnings = _project_statement_sources(
            _logical_tables_for_page(parse_result, page_number),
            page_number=page_number,
            kind=kind,
            occurrence_start=occurrence_start,
            continuation=continuation,
            logical=True,
        )
        if len(segments) > 1:
            selected = max(segments, key=lambda item: _statement_segment_score(item, kind))
            segments = [selected]
    if kind in {"balance_sheet", "income_statement", "cash_flow_statement"}:
        segments, warnings = _prefer_statement_text_rows(
            page,
            segments,
            warnings,
            page_number=page_number,
            kind=kind,
            occurrence_start=occurrence_start,
            continuation=continuation,
        )
    if kind == "balance_sheet":
        for segment in segments:
            warnings.extend(_restore_balance_section_labels(segment, page))
    return segments, warnings


def _project_statement_sources(
    tables: Iterable[Any],
    *,
    page_number: int,
    kind: str,
    occurrence_start: int,
    continuation: bool,
    logical: bool,
) -> tuple[list[ProjectedSegment], list[str]]:
    segments: list[ProjectedSegment] = []
    warnings: list[str] = []
    for table in tables:
        occurrence = occurrence_start + len(segments) + 1
        segment, table_warnings = _project_audit_statement_table(
            table,
            page_number=page_number,
            dataset_id=_statement_dataset_id(kind, occurrence),
            kind=kind,
            continuation=continuation,
        )
        warnings.extend(table_warnings)
        if segment is None:
            continue
        if logical and kind == "owners_equity_changes" and not _reliable_logical_owner_equity(segment):
            warnings.append(
                f"AUDIT_OWNER_EQUITY_LOGICAL_TABLE_REJECTED:page={page_number}:"
                f"table={getattr(table, 'table_id', '')}:rows={len(segment.records)}"
            )
            continue
        segments.append(segment)
    return segments, warnings


def _prefer_statement_text_rows(
    page: Any,
    table_segments: list[ProjectedSegment],
    table_warnings: list[str],
    *,
    page_number: int,
    kind: str,
    occurrence_start: int,
    continuation: bool,
) -> tuple[list[ProjectedSegment], list[str]]:
    text_segment, text_warnings = _project_statement_text_rows(
        page,
        page_number=page_number,
        dataset_id=_statement_dataset_id(kind, occurrence_start + 1),
        kind=kind,
        continuation=continuation,
    )
    table_score = max((_statement_segment_score(item, kind) for item in table_segments), default=-10_000)
    text_score = _statement_segment_score(text_segment, kind)
    if text_segment is None or table_segments and text_score <= table_score:
        return table_segments, table_warnings
    warnings = list(text_warnings)
    if table_segments:
        table_rows = max(len(item.records) for item in table_segments)
        warnings.append(
            f"AUDIT_STATEMENT_TEXT_ROWS_SELECTED:page={page_number}:kind={kind}:"
            f"table_rows={table_rows}:text_rows={len(text_segment.records)}:"
            f"table_score={table_score}:text_score={text_score}"
        )
    return [text_segment], warnings


def _statement_dataset_id(kind: str, occurrence: int) -> str:
    return kind if occurrence == 1 else f"{kind}_{occurrence:02d}"


def _restore_balance_section_labels(segment: ProjectedSegment, page: Any) -> list[str]:
    existing = {normalize_audit_text(_record_item(record)).rstrip(":：") for record in segment.records}
    warnings: list[str] = []
    for block_index, block in enumerate(getattr(page, "texts", None) or []):
        for line_index, raw_line in enumerate(str(getattr(block, "content", "") or "").splitlines()):
            raw_label = raw_line.strip()
            canonical = normalize_audit_text(raw_label).replace("（", "(").replace("）", ")").rstrip(":：")
            anchors = _BALANCE_SECTION_ANCHORS.get(canonical)
            if anchors is None or canonical in existing:
                continue
            normalized_anchors = {
                normalize_audit_text(value).replace("（", "(").replace("）", ")") for value in anchors
            }
            insert_at = next(
                (
                    index
                    for index, record in enumerate(segment.records)
                    if normalize_audit_text(_record_item(record)).replace("（", "(").replace("）", ")")
                    in normalized_anchors
                ),
                None,
            )
            if insert_at is None:
                continue
            keys = [column.key for column in segment.columns]
            raw = {key: raw_label if key == "item" else "" for key in keys}
            table_id = f"text:p{segment.source_page:04d}:b{block_index:04d}"
            source_row_index = block_index * 1000 + line_index
            evidence_ids = [str(value) for value in (getattr(block, "evidence_ids", None) or []) if value]
            source_ref = {
                "page": segment.source_page,
                "table_id": table_id,
                "row": source_row_index,
                "col": 0,
                "field_name": "item",
                **({"bbox": list(block.bbox)} if getattr(block, "bbox", None) else {}),
                **({"evidence_ids": evidence_ids} if evidence_ids else {}),
            }
            source = {
                "page": segment.source_page,
                "page_range": [segment.source_page],
                "table_id": table_id,
                "physical_table_id": table_id,
                "source_row_index": source_row_index,
                "source_cell_refs": [source_ref],
                "recovery": "canonical_text_balance_section_label",
                **({"evidence_ids": evidence_ids} if evidence_ids else {}),
            }
            record = {
                "record_id": "",
                "raw": raw,
                "canonical_raw": dict(raw),
                "normalized": {
                    key: canonical if key == "item" else None if _decimal_field_key(key) else "" for key in keys
                },
                "source": source,
                "source_cell_refs": [source_ref],
                "evidence_ids": evidence_ids,
                "confidence": float(getattr(block, "confidence", 0.0) or 0.0),
            }
            segment.records.insert(insert_at, record)
            existing.add(canonical)
            warnings.append(
                f"AUDIT_BALANCE_SECTION_LABEL_RECOVERED:page={segment.source_page}:label={canonical}:source=canonical_text"
            )
    if warnings:
        _reindex_segment_records(segment)
    return warnings


def _statement_segment_score(segment: ProjectedSegment | None, kind: str) -> int:
    if segment is None or not segment.records:
        return -10_000
    keys = [column.key for column in segment.columns]
    generic_columns = sum(bool(re.fullmatch(r"(?:col|column)_\d+", key)) for key in keys)
    item_keys = [key for key in keys if "item" in key]
    note_keys = [key for key in keys if "note" in key]
    amount_keys = [
        key
        for key in keys
        if any(marker in key for marker in ("amount", "balance", "capital", "reserve", "profit", "total"))
    ]
    valid_items = 0
    paired_amounts = 0
    evidence_rows = 0
    reviewed_rows = 0
    amount_bleed_rows = 0
    missing_item_rows = 0
    invalid_amount_cells = 0
    for record in segment.records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        item_value = next(
            (normalize_audit_text(raw.get(key)) for key in item_keys if normalize_audit_text(raw.get(key))), ""
        )
        if item_value:
            valid_items += 1
        numeric_amounts = sum(_decimal(normalize_audit_text(normalized.get(key))) is not None for key in amount_keys)
        if numeric_amounts >= 2:
            paired_amounts += 1
        if _AMOUNT_RE.search(item_value):
            amount_bleed_rows += 1
        if not item_value and numeric_amounts:
            missing_item_rows += 1
        invalid_amount_cells += sum(
            bool(normalize_audit_text(raw.get(key))) and normalized.get(key) in (None, "") for key in amount_keys
        )
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        if source.get("evidence_ids") or source.get("source_cell_refs"):
            evidence_rows += 1
        review = record.get("review") if isinstance(record.get("review"), dict) else {}
        if review.get("required"):
            reviewed_rows += 1
    semantic_columns = int(bool(item_keys)) * 12 + int(bool(note_keys)) * 3 + min(2, len(amount_keys)) * 8
    if kind == "owners_equity_changes":
        semantic_columns += min(10, len(amount_keys))
    return (
        semantic_columns
        + min(20, len(segment.records))
        + min(20, valid_items)
        + min(24, paired_amounts * 2)
        + min(10, evidence_rows)
        - generic_columns * 12
        - reviewed_rows * 2
        - amount_bleed_rows * 16
        - missing_item_rows * 10
        - invalid_amount_cells * 4
    )


def _reliable_logical_owner_equity(segment: ProjectedSegment) -> bool:
    """Reject logical pseudo-tables whose rows are numerics or OCR fragments, not equity facts."""

    if len(segment.records) < 3:
        return False
    descriptive_rows = 0
    amount_rows = 0
    evidence_rows = 0
    for record in segment.records:
        raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else record.get("raw")
        if not isinstance(raw, dict):
            continue
        item = normalize_audit_text(raw.get("item"))
        if item and _OWNER_EQUITY_ROW_MARKERS.search(item):
            descriptive_rows += 1
        if any(
            _decimal(value) is not None
            for key, value in raw.items()
            if key not in {"item", "period_role", "note_reference"}
        ):
            amount_rows += 1
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        if source.get("evidence_ids") or source.get("source_cell_refs"):
            evidence_rows += 1
    return descriptive_rows >= 3 and amount_rows >= 1 and evidence_rows >= 1


def _project_statement_text_rows(
    page: Any,
    *,
    page_number: int,
    dataset_id: str,
    kind: str,
    continuation: bool,
) -> tuple[ProjectedSegment | None, list[str]]:
    """Recover a statement only when positioned text proves a four-column grid."""

    if kind not in {"balance_sheet", "income_statement", "cash_flow_statement"}:
        return None, []
    table = _statement_table_from_text(page, page_number=page_number, kind=kind)
    if table is None:
        return None, []
    segment, warnings = project_statement_table(
        table,
        page_number=page_number,
        dataset_id=dataset_id,
        kind=kind,
    )
    if segment is None:
        return None, warnings
    if kind == "balance_sheet":
        segment = _normalize_balance_segment(segment, continuation=continuation)
        warnings = _drop_note_line_warnings(warnings)
    elif kind in {"income_statement", "cash_flow_statement"}:
        segment = _map_comparative_columns(segment)
    for record in segment.records:
        record.setdefault("source", {})["recovery"] = "positioned_text_rows"
    warnings.append(f"AUDIT_STATEMENT_TEXT_ROWS_RECOVERED:page={page_number}:kind={kind}:rows={len(segment.records)}")
    return segment, warnings


def _statement_table_from_text(page: Any, *, page_number: int, kind: str) -> TableBlock | None:
    atoms = _positioned_text_atoms(page)
    header = _statement_text_header(atoms, kind=kind)
    if header is None:
        return None
    header_atoms, column_centers = header
    header_bottom = max(atom.bbox[3] for atom in header_atoms)
    footer_y = min(
        (
            atom.bbox[1]
            for atom in atoms
            if atom.center_y > header_bottom and re.search(r"法定代表人|会计机构负责人|主管会计工作负责人", atom.text)
        ),
        default=float("inf"),
    )
    excluded = set(header_atoms)
    body = [
        atom for atom in atoms if atom not in excluded and atom.center_y > header_bottom and atom.center_y < footer_y
    ]
    if sum(bool(_FULL_AMOUNT_RE.fullmatch(atom.text.replace(" ", ""))) for atom in body) < 4:
        return None

    heights = sorted(atom.bbox[3] - atom.bbox[1] for atom in body)
    median_height = heights[len(heights) // 2] if heights else 10.0
    row_tolerance = max(3.0, min(6.0, median_height * 0.45))
    row_atoms = _cluster_text_rows(body, tolerance=row_tolerance)
    rows: list[TableRow] = []
    table_id = f"audit_text_p{page_number}"
    boundaries = [(left + right) / 2 for left, right in zip(column_centers, column_centers[1:])]
    for source_index, atoms_in_row in enumerate(row_atoms):
        columns: list[list[_TextAtom]] = [[], [], [], []]
        for atom in atoms_in_row:
            column_index = sum(atom.center_x > boundary for boundary in boundaries)
            columns[min(column_index, 3)].append(atom)
        values = [_join_text_atoms(column) for column in columns]
        if not values[0] or _statement_text_footer(values[0]):
            continue
        cells = [
            _text_cell(column, text=value, row_index=len(rows), column_index=column_index)
            for column_index, (column, value) in enumerate(zip(columns, values, strict=True))
        ]
        rows.append(
            TableRow(
                cells=cells,
                confidence=min((cell.confidence for cell in cells if cell.text), default=0.0),
                source_page=page_number,
                source_physical_id=table_id,
                source_row_index=source_index,
                source_cell_refs=[{"evidence_id": evidence_id} for cell in cells for evidence_id in cell.evidence_ids],
            )
        )
    if len(rows) < 3 or not any(any(amount_like(cell.text) for cell in row.cells[2:]) for row in rows):
        return None

    headers = (
        ["项目", "附注", "期末余额", "期初余额"]
        if kind == "balance_sheet"
        else ["项目", "附注", "本期金额", "上期金额"]
    )
    all_atoms = [atom for row in row_atoms for atom in row]
    confidence = min((atom.confidence for atom in all_atoms), default=0.0)
    return TableBlock(
        table_id=table_id,
        headers=headers,
        rows=rows,
        page=page_number,
        bbox=list(_union_bbox(all_atoms)),
        confidence=confidence,
        extraction_layer="audit_positioned_text",
        extraction_confidence=confidence,
        evidence_ids=list(dict.fromkeys(evidence_id for atom in all_atoms for evidence_id in atom.evidence_ids)),
        metadata={"audit_text_row_recovery": True},
    )


def _positioned_text_atoms(page: Any) -> list[_TextAtom]:
    atoms: list[_TextAtom] = []
    for block in getattr(page, "texts", None) or []:
        lines = [line.strip() for line in str(getattr(block, "content", "") or "").splitlines() if line.strip()]
        bbox = getattr(block, "bbox", None)
        if len(lines) != 1 or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        atoms.append(
            _TextAtom(
                text=lines[0],
                bbox=tuple(float(value) for value in bbox),
                confidence=float(getattr(block, "confidence", 0.0) or 0.0),
                evidence_ids=tuple(str(value) for value in (getattr(block, "evidence_ids", None) or [])),
            )
        )
    return atoms


def _statement_text_header(
    atoms: list[_TextAtom],
    *,
    kind: str,
) -> tuple[tuple[_TextAtom, _TextAtom, _TextAtom, _TextAtom], tuple[float, float, float, float]] | None:
    item_atoms = [atom for atom in atoms if normalize_audit_text(atom.text).replace(" ", "") == "项目"]
    note_atoms = [atom for atom in atoms if normalize_audit_text(atom.text).replace(" ", "") in {"附注", "行次"}]
    period_marker = (
        re.compile(r"期末余额|期初余额|20\d{2}年\d{1,2}月\d{1,2}日")
        if kind == "balance_sheet"
        else re.compile(r"本期金额|上期金额|本年金额|上年金额|20\d{2}年度")
    )
    period_atoms = [atom for atom in atoms if period_marker.fullmatch(normalize_audit_text(atom.text).replace(" ", ""))]
    candidates: list[tuple[float, tuple[_TextAtom, _TextAtom, _TextAtom, _TextAtom]]] = []
    for item in item_atoms:
        for note in note_atoms:
            periods = sorted(
                (atom for atom in period_atoms if abs(atom.center_y - item.center_y) <= 10),
                key=lambda atom: atom.center_x,
            )
            if len(periods) < 2:
                continue
            selected = (item, note, periods[-2], periods[-1])
            centers = tuple(atom.center_x for atom in selected)
            vertical_spread = max(atom.center_y for atom in selected) - min(atom.center_y for atom in selected)
            if list(centers) != sorted(centers) or vertical_spread > 10:
                continue
            candidates.append((vertical_spread, selected))
    if not candidates:
        return None
    selected = min(candidates, key=lambda candidate: candidate[0])[1]
    return selected, tuple(atom.center_x for atom in selected)


def _cluster_text_rows(atoms: list[_TextAtom], *, tolerance: float) -> list[list[_TextAtom]]:
    rows: list[list[_TextAtom]] = []
    centers: list[float] = []
    for atom in sorted(atoms, key=lambda value: (value.center_y, value.center_x)):
        if not rows or abs(atom.center_y - centers[-1]) > tolerance:
            rows.append([atom])
            centers.append(atom.center_y)
            continue
        rows[-1].append(atom)
        centers[-1] = sum(value.center_y for value in rows[-1]) / len(rows[-1])
    return rows


def _join_text_atoms(atoms: list[_TextAtom]) -> str:
    return " ".join(atom.text for atom in sorted(atoms, key=lambda value: value.center_x)).strip()


def _text_cell(
    atoms: list[_TextAtom],
    *,
    text: str,
    row_index: int,
    column_index: int,
) -> CellValue:
    return CellValue(
        text=text,
        confidence=min((atom.confidence for atom in atoms), default=0.0),
        bbox=list(_union_bbox(atoms)) if atoms else None,
        row_index=row_index,
        col_index=column_index,
        geometry_status="exact" if atoms else "missing",
        geometry_source="positioned_text",
        geometry_confidence=min((atom.confidence for atom in atoms), default=None),
        evidence_ids=list(dict.fromkeys(evidence_id for atom in atoms for evidence_id in atom.evidence_ids)),
    )


def _union_bbox(atoms: list[_TextAtom]) -> tuple[float, float, float, float]:
    return (
        min(atom.bbox[0] for atom in atoms),
        min(atom.bbox[1] for atom in atoms),
        max(atom.bbox[2] for atom in atoms),
        max(atom.bbox[3] for atom in atoms),
    )


def _statement_text_footer(value: str) -> bool:
    return bool(re.search(r"法定代表人|会计机构负责人|主管会计工作负责人", value))


def _project_audit_statement_table(
    table: Any,
    *,
    page_number: int,
    dataset_id: str,
    kind: str,
    continuation: bool = False,
) -> tuple[ProjectedSegment | None, list[str]]:
    candidates: list[_StatementCandidate] = []
    failure_warnings: list[str] = []

    direct, direct_warnings = project_statement_table(
        table,
        page_number=page_number,
        dataset_id=dataset_id,
        kind=kind,
    )
    _append_statement_candidate(
        candidates,
        failure_warnings,
        mode="direct",
        segment=direct,
        warnings=direct_warnings,
        kind=kind,
        continuation=continuation,
        page_number=page_number,
    )

    period_table = _recover_period_headers(table, kind=kind)
    if period_table is not None:
        recovered, recovered_warnings = project_statement_table(
            period_table,
            page_number=page_number,
            dataset_id=dataset_id,
            kind=kind,
        )
        if recovered is not None:
            source_headers = list(getattr(table, "headers", None) or [])
            recovered.columns = [
                replace(spec, label=str(source_headers[spec.source_index]))
                if 0 <= spec.source_index < len(source_headers)
                else spec
                for spec in recovered.columns
            ]
            recovered_warnings.append(f"AUDIT_FINANCIAL_PERIOD_HEADERS_RECOVERED:page={page_number}:kind={kind}")
        _append_statement_candidate(
            candidates,
            failure_warnings,
            mode="period_headers",
            segment=recovered,
            warnings=recovered_warnings,
            kind=kind,
            continuation=continuation,
            page_number=page_number,
        )

    adapted, recovery_rows = _recover_statement_cells(table, kind=kind)
    if adapted is not None:
        recovered, recovered_warnings = project_statement_table(
            adapted,
            page_number=page_number,
            dataset_id=dataset_id,
            kind=kind,
        )
        if recovered is not None:
            _annotate_recovered_records(recovered, recovery_rows)
            recovered_warnings.extend(_statement_recovery_warnings(recovery_rows, page_number=page_number, kind=kind))
        _append_statement_candidate(
            candidates,
            failure_warnings,
            mode="cells",
            segment=recovered,
            warnings=recovered_warnings,
            kind=kind,
            continuation=continuation,
            page_number=page_number,
        )

    if kind == "owners_equity_changes":
        recovered, recovered_warnings = _recover_rotated_owner_equity(
            table,
            page_number=page_number,
            dataset_id=dataset_id,
        )
        _append_statement_candidate(
            candidates,
            failure_warnings,
            mode="rotated",
            segment=recovered,
            warnings=recovered_warnings,
            kind=kind,
            continuation=continuation,
            page_number=page_number,
        )

    if not candidates:
        return None, list(dict.fromkeys(failure_warnings))
    selected = max(candidates, key=lambda candidate: _statement_segment_score(candidate.segment, kind))
    warnings = list(selected.warnings)
    if selected.mode != "direct" and direct is not None:
        warnings.append(
            f"AUDIT_STATEMENT_RECOVERY_SELECTED:page={page_number}:kind={kind}:mode={selected.mode}:"
            f"direct_score={_statement_segment_score(direct, kind)}:"
            f"selected_score={_statement_segment_score(selected.segment, kind)}"
        )
    return selected.segment, list(dict.fromkeys(warnings))


def _append_statement_candidate(
    candidates: list[_StatementCandidate],
    failure_warnings: list[str],
    *,
    mode: str,
    segment: ProjectedSegment | None,
    warnings: list[str],
    kind: str,
    continuation: bool,
    page_number: int,
) -> None:
    if segment is None:
        failure_warnings.extend(warnings)
        return
    finalized, finalized_warnings = _finalize_statement_segment(
        segment,
        warnings,
        kind=kind,
        continuation=continuation,
        page_number=page_number,
    )
    candidates.append(_StatementCandidate(mode=mode, segment=finalized, warnings=tuple(finalized_warnings)))


def _finalize_statement_segment(
    segment: ProjectedSegment,
    warnings: list[str],
    *,
    kind: str,
    continuation: bool,
    page_number: int,
) -> tuple[ProjectedSegment, list[str]]:
    finalized_warnings = list(warnings)
    if kind == "balance_sheet":
        segment = _normalize_balance_segment(segment, continuation=continuation)
        segment, shifted, fragmented = _repair_balance_amount_shifts(segment)
        if shifted:
            finalized_warnings.append(
                f"AUDIT_BALANCE_AMOUNT_SHIFT_RECOVERED:page={page_number}:kind={kind}:rows={shifted}"
            )
        if fragmented:
            finalized_warnings.append(
                f"AUDIT_AMOUNT_FRAGMENT_INFERRED:page={page_number}:kind={kind}:rows={fragmented}"
            )
        segment, expanded = _expand_merged_balance_items(segment)
        if expanded:
            finalized_warnings.append(f"AUDIT_MERGED_ITEM_ROWS_INFERRED:page={page_number}:kind={kind}:rows={expanded}")
        finalized_warnings = _drop_note_line_warnings(finalized_warnings)
    elif kind in {"income_statement", "cash_flow_statement"}:
        segment = _map_comparative_columns(segment)
    elif kind == "owners_equity_changes":
        segment, removed = _prune_owner_equity_header_records(segment)
        if removed:
            finalized_warnings.append(f"AUDIT_OWNER_EQUITY_HEADER_ROWS_REMOVED:page={page_number}:rows={removed}")
        segment = _map_owner_equity_columns(segment)
    segment, removed = _prune_statement_header_records(segment)
    if removed:
        finalized_warnings.append(f"AUDIT_STATEMENT_HEADER_ROWS_REMOVED:page={page_number}:kind={kind}:rows={removed}")
    return segment, finalized_warnings


def _repair_balance_amount_shifts(segment: ProjectedSegment) -> tuple[ProjectedSegment, int, int]:
    shifted = 0
    fragmented = 0
    for record in segment.records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        item, note, embedded, exact = _split_item_note_amount(normalize_audit_text(raw.get("item")))
        if not exact or not embedded or not item:
            continue
        note = note or normalize_audit_text(raw.get("note_reference"))
        current = normalize_audit_text(raw.get("current_period_amount")).replace(" ", "")
        previous = normalize_audit_text(raw.get("previous_period_amount")).replace(" ", "")
        embedded_decimal = _decimal(embedded)
        current_decimal = _decimal(current)
        ratio = abs(current_decimal / embedded_decimal) if embedded_decimal and current_decimal is not None else None
        repaired = (
            _join_amount_fragment(embedded, f"{current}{previous}")
            if ratio is not None and not Decimal("0.01") <= ratio <= Decimal("100")
            else None
        )
        if repaired is not None:
            _set_recovered_balance_values(record, item, note, repaired[0], repaired[1], mode="amount_fragment_join")
            add_review_reason(record, "amount_fragment_join")
            fragmented += 1
            continue
        if current and not previous and _FULL_AMOUNT_RE.fullmatch(current):
            _set_recovered_balance_values(record, item, note, embedded, current, mode="embedded_amount_shift")
            shifted += 1
        elif not current:
            _set_recovered_balance_values(record, item, note, embedded, previous, mode="embedded_current_amount")
            shifted += 1
    return segment, shifted, fragmented


def _set_recovered_balance_values(
    record: dict[str, Any],
    item: str,
    note: str,
    current: str,
    previous: str,
    *,
    mode: str,
) -> None:
    raw_before = copy.deepcopy(record.get("raw") or {})
    for pool_name in ("raw", "canonical_raw"):
        pool = record.get(pool_name)
        if isinstance(pool, dict):
            pool.update(
                item=item,
                note_reference=note,
                current_period_amount=current,
                previous_period_amount=previous,
            )
    normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
    normalized.update(
        item=normalize_audit_text(item),
        note_reference=normalize_audit_text(note),
        current_period_amount=normalize_scalar(current, value_type="decimal"),
        previous_period_amount=normalize_scalar(previous, value_type="decimal"),
    )
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    source["recovery"] = mode
    source["recovery_raw"] = raw_before


def _expand_merged_balance_items(segment: ProjectedSegment) -> tuple[ProjectedSegment, int]:
    records: list[dict[str, Any]] = []
    cursor = 0
    expanded = 0
    while cursor < len(segment.records):
        record = segment.records[cursor]
        following = segment.records[cursor + 1] if cursor + 1 < len(segment.records) else None
        remainder = _record_item(record).replace(" ", "")
        labels: list[str] = []
        while remainder:
            label = next((candidate for candidate in _STANDARD_BALANCE_ITEMS if remainder.startswith(candidate)), "")
            if not label:
                labels = []
                break
            labels.append(label)
            remainder = remainder[len(label) :]
        following_raw = following.get("raw") if following and isinstance(following.get("raw"), dict) else {}
        if (
            len(labels) < 2
            or following is None
            or _record_item(following)
            or not any(normalize_audit_text(value) for key, value in following_raw.items() if key != "item")
        ):
            records.append(record)
            cursor += 1
            continue
        for label in labels[:-1]:
            split = copy.deepcopy(record)
            for pool_name in ("raw", "canonical_raw", "normalized"):
                pool = split.get(pool_name)
                if isinstance(pool, dict):
                    split[pool_name] = {key: label if key == "item" else "" for key in pool}
            split_source = split.get("source") if isinstance(split.get("source"), dict) else {}
            refs = [
                ref
                for ref in split_source.get("source_cell_refs") or []
                if isinstance(ref, dict) and ref.get("field_name") == "item"
            ]
            split_source.update(
                source_cell_refs=refs,
                recovery="merged_statement_item_sequence",
                recovery_raw=_record_item(record),
            )
            split["source_cell_refs"] = copy.deepcopy(refs)
            add_review_reason(split, "merged_item_sequence")
            records.append(split)

        combined = copy.deepcopy(following)
        for pool_name in ("raw", "canonical_raw", "normalized"):
            pool = combined.get(pool_name)
            if isinstance(pool, dict):
                pool["item"] = labels[-1]
        item_source = record.get("source") if isinstance(record.get("source"), dict) else {}
        combined_source = combined.get("source") if isinstance(combined.get("source"), dict) else {}
        item_refs = [
            copy.deepcopy(ref)
            for ref in item_source.get("source_cell_refs") or []
            if isinstance(ref, dict) and ref.get("field_name") == "item"
        ]
        refs = combined_source.setdefault("source_cell_refs", [])
        refs[:0] = item_refs
        evidence = [*(item_source.get("evidence_ids") or []), *(combined_source.get("evidence_ids") or [])]
        combined_source.update(
            evidence_ids=list(dict.fromkeys(str(value) for value in evidence if value)),
            recovery="merged_statement_item_sequence",
            recovery_raw=_record_item(record),
            recovery_sources=[_record_source_identity(record), _record_source_identity(following)],
        )
        combined["source_cell_refs"] = copy.deepcopy(refs)
        combined["evidence_ids"] = list(combined_source["evidence_ids"])
        add_review_reason(combined, "merged_item_sequence")
        records.append(combined)
        expanded += len(labels) - 1
        cursor += 2
    if not expanded:
        return segment, 0
    segment.records = records
    _reindex_segment_records(segment)
    return segment, expanded


def _record_item(record: dict[str, Any]) -> str:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    return normalize_audit_text(raw.get("item"))


def _record_source_identity(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    return {
        "page": int(source.get("page") or 0),
        "table_id": str(source.get("table_id") or ""),
        "source_row_index": int(source.get("source_row_index") or 0),
    }


def _reindex_segment_records(segment: ProjectedSegment) -> None:
    segment.source_row_refs = []
    for ordinal, record in enumerate(segment.records, start=1):
        record["record_id"] = f"{segment.dataset_id}:r{ordinal:06d}"
        segment.source_row_refs.append(_record_source_identity(record))


def _recover_period_headers(table: Any, *, kind: str) -> Any | None:
    headers = [normalize_audit_text(value).replace(" ", "") for value in (getattr(table, "headers", None) or [])]
    period_columns = [index for index, header in enumerate(headers) if _PERIOD_HEADER_RE.fullmatch(header)]
    if len(period_columns) < 2 or len(period_columns) % 2 or not any("项目" in header for header in headers[:2]):
        return None
    recovered_headers = list(headers)
    current, previous = ("期末余额", "期初余额") if kind == "balance_sheet" else ("本期金额", "上期金额")
    for ordinal, column in enumerate(period_columns):
        recovered_headers[column] = current if ordinal % 2 == 0 else previous
    return table.model_copy(update={"headers": recovered_headers})


def _statement_recovery_warnings(
    recovery_rows: dict[int, dict[str, str]],
    *,
    page_number: int,
    kind: str,
) -> list[str]:
    inferred = sum(info["mode"] == "implicit_amount_split" for info in recovery_rows.values())
    fragmented = sum(info["mode"] == "fragmented_amount_split" for info in recovery_rows.values())
    unresolved = sum(info["mode"] == "unresolved_combined_amount" for info in recovery_rows.values())
    warnings = [f"AUDIT_STATEMENT_CELLS_RECOVERED:page={page_number}:kind={kind}:rows={len(recovery_rows)}"]
    if inferred:
        warnings.append(f"AUDIT_AMOUNT_SPLIT_INFERRED:page={page_number}:kind={kind}:rows={inferred}")
    if fragmented:
        warnings.append(f"AUDIT_AMOUNT_FRAGMENT_INFERRED:page={page_number}:kind={kind}:rows={fragmented}")
    if unresolved:
        warnings.append(f"AUDIT_AMOUNT_SPLIT_UNRESOLVED:page={page_number}:kind={kind}:rows={unresolved}")
    return warnings


def _prune_owner_equity_header_records(segment: ProjectedSegment) -> tuple[ProjectedSegment, int]:
    header_markers = {
        "本期金额",
        "上期金额",
        "本年金额",
        "上年金额",
        "实收资本",
        "实收资本(或股本)",
        "其他权益工具",
        "优先股",
        "永续债",
        "其他",
        "资本公积",
        "减:库存股",
        "库存股",
        "其他综合收益",
        "专项储备",
        "盈余公积",
        "未分配利润",
        "所有者权益合计",
    }
    removed_indexes: set[int] = set()
    retained: list[dict[str, Any]] = []
    for record in segment.records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        item = normalize_audit_text(raw.get("item", ""))
        values = [normalize_audit_text(value).replace(" ", "") for value in raw.values() if value not in (None, "")]
        header_only = not item and bool(values) and all(value in header_markers for value in values)
        if header_only and not any(_FULL_AMOUNT_RE.fullmatch(value.replace(" ", "")) for value in values):
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            removed_indexes.add(int(source.get("source_row_index") or 0))
            continue
        retained.append(record)
    if not removed_indexes:
        return segment, 0
    segment.records = retained
    segment.source_row_refs = [
        ref for ref in segment.source_row_refs if int(ref.get("source_row_index") or 0) not in removed_indexes
    ]
    for ordinal, record in enumerate(segment.records, start=1):
        record["record_id"] = f"{segment.dataset_id}:r{ordinal:06d}"
    return segment, len(removed_indexes)


def _prune_statement_header_records(segment: ProjectedSegment) -> tuple[ProjectedSegment, int]:
    removed: set[tuple[int, str, int]] = set()
    retained: list[dict[str, Any]] = []
    amount_keys = {
        column.key
        for column in segment.columns
        if any(marker in column.key for marker in ("amount", "balance", "capital", "reserve", "profit", "total"))
    }
    item_keys = [column.key for column in segment.columns if "item" in column.key]
    for record in segment.records:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        item = "".join(normalize_audit_text(raw.get(key)).replace(" ", "") for key in item_keys)
        values = [normalize_audit_text(value).replace(" ", "") for value in raw.values() if value not in (None, "")]
        has_amount = any(amount_like(raw.get(key)) for key in amount_keys)
        header_only = (
            not has_amount
            and bool(values)
            and (
                ("项目" in item and "附注" in item)
                or all(bool(_TABLE_HEADER_MARKERS.search(value)) for value in values)
            )
        )
        if not header_only:
            retained.append(record)
            continue
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        removed.add(
            (
                int(source.get("page") or segment.source_page),
                str(source.get("table_id") or segment.table_id),
                int(source.get("source_row_index") or 0),
            )
        )
    if not removed:
        return segment, 0
    segment.records = retained
    segment.source_row_refs = [
        ref
        for ref in segment.source_row_refs
        if (
            int(ref.get("page") or segment.source_page),
            str(ref.get("table_id") or segment.table_id),
            int(ref.get("source_row_index") or 0),
        )
        not in removed
    ]
    for ordinal, record in enumerate(segment.records, start=1):
        record["record_id"] = f"{segment.dataset_id}:r{ordinal:06d}"
    return segment, len(removed)


def _recover_statement_cells(
    table: Any,
    *,
    kind: str,
) -> tuple[Any | None, dict[int, dict[str, str]]]:
    if kind not in {"balance_sheet", "income_statement", "cash_flow_statement"}:
        return None, {}
    rows = list(getattr(table, "rows", None) or [])
    width = table_width(table)
    if not rows or width < 2:
        return None, {}

    labels = (
        ["项目", "附注", "期末余额", "期初余额"]
        if kind == "balance_sheet"
        else ["项目", "附注", "本期金额", "上期金额"]
    )
    period_positions = _period_positions(table)
    typical_groups = _typical_comparative_groups(rows, width)
    adapted_rows: list[Any] = []
    recovery_rows: dict[int, dict[str, str]] = {}
    for row_index, row in enumerate(rows):
        cells = row_cells_by_column(row, width)
        values = [
            re.sub(r"\s+", " ", str(getattr(cell, "text", "") or "")).strip() if cell is not None else ""
            for cell in cells
        ]
        compact = "".join(values).replace(" ", "")
        if row_index == 0 and "项目" in compact and ("20" in compact or _TABLE_HEADER_MARKERS.search(compact)):
            projected_values = labels
            mode = "header_repartition"
        elif kind == "balance_sheet":
            projected_values, mode = _recover_balance_row(values, period_positions)
        else:
            projected_values, mode = _recover_comparative_row(values, typical_groups, period_positions)
        if projected_values is None:
            return None, {}
        templates = _field_templates(cells, values)
        adapted_rows.append(_projected_row(row, projected_values, templates, row_index))
        recovery_rows[int(getattr(row, "source_row_index", row_index) or row_index)] = {
            "mode": mode,
            "raw": " | ".join(value for value in values if value),
        }
    metadata = copy.deepcopy(getattr(table, "metadata", None) or {})
    metadata["audit_statement_recovery"] = True
    return table.model_copy(update={"headers": labels, "rows": adapted_rows, "metadata": metadata}), recovery_rows


def _period_positions(table: Any) -> tuple[int, int] | None:
    values = [normalize_audit_text(value).replace(" ", "") for value in (getattr(table, "headers", None) or [])]
    positions = [index for index, value in enumerate(values) if re.search(r"20\d{2}年", value)]
    if len(positions) >= 2:
        return positions[0], positions[-1]
    current = next((index for index, value in enumerate(values) if re.search(r"期末|本期|本年", value)), None)
    previous = next((index for index, value in enumerate(values) if re.search(r"期初|上期|上年", value)), None)
    if current is not None and previous is not None:
        return current, previous
    rows = list(getattr(table, "rows", None) or [])
    if rows:
        width = table_width(table)
        first = row_cells_by_column(rows[0], width)
        positions = [
            index
            for index, cell in enumerate(first)
            if re.search(r"20\d{2}年", normalize_audit_text(getattr(cell, "text", "")).replace(" ", ""))
        ]
    return (positions[0], positions[-1]) if len(positions) >= 2 else None


def _recover_balance_row(
    values: list[str],
    period_positions: tuple[int, int] | None,
) -> tuple[list[str] | None, str]:
    if (
        len(values) == 4
        and (not values[1] or _NOTE_RE.fullmatch(values[1]))
        and all(not value or _FULL_AMOUNT_RE.fullmatch(value.replace(" ", "")) for value in values[2:])
        and (values[2] or values[3] or not _AMOUNT_RE.search(values[0]))
    ):
        return [values[0], values[1].replace(" ", ""), values[2].replace(" ", ""), values[3].replace(" ", "")], (
            "source_cell_repartition"
        )
    previous_start = period_positions[1] if period_positions else len(values)
    prefix_values = [value for value in values[:previous_start] if value]
    prefix = " ".join(prefix_values).strip()
    previous_blob = " ".join(value for value in values[previous_start:] if value).strip()
    previous_matches = _AMOUNT_RE.findall(previous_blob)
    previous = previous_matches[-1] if previous_matches else ""
    standalone_note = next((value.replace(" ", "") for value in prefix_values if _NOTE_RE.fullmatch(value)), "")
    standalone_amounts = [value.replace(" ", "") for value in prefix_values if _FULL_AMOUNT_RE.fullmatch(value)]
    if standalone_amounts:
        current = standalone_amounts[-1]
        item_parts = [value for value in prefix_values if value.replace(" ", "") not in {standalone_note, current}]
        item_blob = " ".join(item_parts).strip()
        item, embedded_note = _split_trailing_note(item_blob)
        note = standalone_note or embedded_note
        return [item, note, current, previous], "source_cell_repartition"
    comparative = _split_combined_comparatives(prefix, typical_groups=2) if not previous else None
    if comparative is not None:
        item, note, current, previous, mode = comparative
        return [item, note, current, previous], mode
    item, note, current, exact = _split_item_note_amount(prefix)
    if not any((item, note, current, previous)):
        return ["", "", "", ""], "source_cell_repartition"
    mode = "source_cell_repartition" if exact else "unresolved_combined_amount"
    if not exact and prefix:
        item = item or prefix
    return [item, note, current, previous], mode


def _split_combined_comparatives(
    value: str,
    *,
    typical_groups: int,
) -> tuple[str, str, str, str, str] | None:
    compact = value.replace(" ", "")
    markers = list(re.finditer(r"[一二三四五六七八九十百]+[、.．]", compact))
    for marker in reversed(markers):
        note, remainder = _split_note_prefix(compact[marker.start() :])
        amounts = _explicit_amount_sequence(remainder)
        if note and len(amounts) == 2 and "".join(amounts) == remainder:
            return compact[: marker.start()], note, amounts[0], amounts[1], "source_cell_repartition"
        if note and len(amounts) == 1 and amounts[0] == remainder:
            implicit = _implicit_amount_pair(remainder, typical_groups=typical_groups)
            if implicit is not None:
                return compact[: marker.start()], note, implicit[0], implicit[1], "implicit_amount_split"
    amounts = list(_AMOUNT_RE.finditer(compact))
    if len(amounts) == 2 and amounts[0].end() == amounts[1].start() and amounts[1].end() == len(compact):
        return (
            compact[: amounts[0].start()],
            "",
            amounts[0].group(),
            amounts[1].group(),
            "source_cell_repartition",
        )
    if len(amounts) == 1 and amounts[0].end() == len(compact):
        implicit = _implicit_amount_pair(amounts[0].group(), typical_groups=typical_groups)
        if implicit is not None:
            return compact[: amounts[0].start()], "", implicit[0], implicit[1], "implicit_amount_split"
    return None


def _split_trailing_note(value: str) -> tuple[str, str]:
    match = re.search(r"(?P<note>[一二三四五六七八九十百]+[、.．]\s*\d{1,3})$", value)
    if not match:
        return value, ""
    return value[: match.start()].strip(), match.group("note").replace(" ", "")


def _split_item_note_amount(value: str) -> tuple[str, str, str, bool]:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    marker = re.search(r"[一二三四五六七八九十百]+[、.．]", text)
    if marker:
        item = text[: marker.start()].strip()
        chapter = marker.group().replace("．", ".")
        note_match = re.match(r"(?P<number>\d{1,3})(?:\s+)(?P<amount>.+)$", text[marker.end() :])
        if note_match and _FULL_AMOUNT_RE.fullmatch(note_match.group("amount").replace(" ", "")):
            return item, f"{chapter}{note_match.group('number')}", note_match.group("amount").replace(" ", ""), True
        tail = text[marker.end() :].replace(" ", "")
        leading = re.match(r"\d{1,6}", tail)
        if leading:
            digits = leading.group()
            suffix = tail[len(digits) :]
            candidates: list[tuple[int, str, str]] = []
            for note_length in range(1, min(2, len(digits)) + 1):
                note_number = digits[:note_length]
                amount = digits[note_length:] + suffix
                if _FULL_AMOUNT_RE.fullmatch(amount):
                    candidates.append((note_length, f"{chapter}{note_number}", amount))
            if candidates:
                _length, note, amount = max(candidates, key=lambda candidate: candidate[0])
                return item, note, amount, True
    matches = list(_AMOUNT_RE.finditer(text))
    if matches:
        match = matches[-1]
        prefix = text[: match.start()].strip()
        if not re.search(r"\d", prefix) or _NOTE_RE.search(prefix):
            note_match = _NOTE_RE.search(prefix)
            note = note_match.group().replace(" ", "") if note_match else ""
            item = (prefix[: note_match.start()] if note_match else prefix).strip()
            return item, note, match.group(), True
    return text, "", "", not bool(re.search(r"\d", text))


def _recover_comparative_row(
    values: list[str],
    typical_groups: int,
    period_positions: tuple[int, int] | None,
) -> tuple[list[str] | None, str]:
    nonempty = [(index, value) for index, value in enumerate(values) if value]
    if not nonempty:
        return ["", "", "", ""], "source_cell_repartition"
    item_index, item = nonempty[0]
    if item == "项目" or ("项目" in item and "20" in "".join(value for _, value in nonempty)):
        return ["项目", "附注", "本期金额", "上期金额"], "header_repartition"

    positioned = _positioned_comparatives(values, period_positions)
    if positioned is not None:
        return positioned, "source_cell_repartition"

    tail_values = [value for index, value in nonempty if index > item_index]
    note = next((value.replace(" ", "") for value in tail_values if _NOTE_RE.fullmatch(value)), "")
    standalone = [value.replace(" ", "") for value in tail_values if _FULL_AMOUNT_RE.fullmatch(value.replace(" ", ""))]
    direct = _standalone_comparatives(item, note, standalone, tail_values)
    if direct is not None:
        return direct

    combined = "".join(value for _, value in nonempty).replace(" ", "")
    comparative = _split_combined_comparatives(combined, typical_groups=typical_groups)
    if comparative is not None:
        split_item, split_note, current, previous, mode = comparative
        return [split_item, split_note, current, previous], mode

    return _tail_comparatives(item, note, tail_values, typical_groups=typical_groups)


def _positioned_comparatives(
    values: list[str],
    period_positions: tuple[int, int] | None,
) -> list[str] | None:
    if period_positions is None:
        return None
    current_index, previous_index = period_positions
    if max(current_index, previous_index) >= len(values):
        return None
    current = values[current_index].replace(" ", "")
    previous = values[previous_index].replace(" ", "")
    if not all(not value or _FULL_AMOUNT_RE.fullmatch(value) for value in (current, previous)):
        return None
    prefix_values = [value for value in values[: min(current_index, previous_index)] if value]
    note = next((value.replace(" ", "") for value in prefix_values if _NOTE_RE.fullmatch(value)), "")
    items = [value for value in prefix_values if value.replace(" ", "") != note]
    item = " ".join(items).strip()
    if not item or _AMOUNT_RE.search(item):
        return None
    return [item, note, current, previous]


def _standalone_comparatives(
    item: str,
    note: str,
    standalone: list[str],
    tail_values: list[str],
) -> tuple[list[str], str] | None:
    if len(standalone) >= 2:
        return [item, note, standalone[0], standalone[1]], "source_cell_repartition"
    if len(standalone) == 1 and len(tail_values) > 1:
        remainder = "".join(value for value in tail_values if value != note and value != standalone[0]).replace(" ", "")
        fragmented = _join_amount_fragment(standalone[0], remainder)
        if fragmented is not None:
            return [item, note, fragmented[0], fragmented[1]], "fragmented_amount_split"
        return [item, note, standalone[0], ""], "source_cell_repartition"
    return None


def _join_amount_fragment(prefix: str, remainder: str) -> tuple[str, str] | None:
    fragment = re.match(r"^(?P<digit>\d)[,.](?P<decimals>\d{2})(?P<previous>.*)$", remainder)
    if fragment is None or "." not in prefix:
        return None
    whole, prefix_decimals = prefix.rsplit(".", 1)
    sign = "-" if whole.startswith("-") else ""
    unsigned_whole = whole.removeprefix("-")
    group = f"{prefix_decimals}{fragment.group('digit')}"
    previous = fragment.group("previous")
    if len(group) != 3 or previous and not _FULL_AMOUNT_RE.fullmatch(previous):
        return None
    current = f"{sign}{unsigned_whole},{group}.{fragment.group('decimals')}"
    return (current, previous) if _FULL_AMOUNT_RE.fullmatch(current) else None


def _tail_comparatives(
    item: str,
    note: str,
    tail_values: list[str],
    *,
    typical_groups: int,
) -> tuple[list[str], str]:
    blob = "".join(value for value in tail_values if value != note).replace(" ", "")
    note, remainder = _split_note_prefix(blob)
    explicit = _explicit_amount_sequence(remainder)
    if len(explicit) >= 2:
        return [item, note, explicit[0], explicit[1]], "source_cell_repartition"
    if len(explicit) == 1 and explicit[0] == remainder:
        implicit = _implicit_amount_pair(remainder, typical_groups=typical_groups)
        if implicit is not None:
            return [item, note, implicit[0], implicit[1]], "implicit_amount_split"
        return [item, note, remainder, ""], "source_cell_repartition"
    if not blob:
        return [item, note, "", ""], "source_cell_repartition"
    return [item, note, remainder or blob, ""], "unresolved_combined_amount"


def _split_note_prefix(value: str) -> tuple[str, str]:
    marker = re.match(r"(?P<chapter>[一二三四五六七八九十百]+[、.．])(?P<body>.+)$", value)
    if not marker:
        return "", value
    chapter = marker.group("chapter").replace("．", ".")
    body = marker.group("body")
    leading = re.match(r"\d{1,6}", body)
    if not leading:
        return "", value
    digits = leading.group()
    suffix = body[len(digits) :]
    candidates: list[tuple[str, str]] = []
    for note_length in range(1, min(2, len(digits)) + 1):
        remainder = digits[note_length:] + suffix
        amounts = _explicit_amount_sequence(remainder)
        if amounts and "".join(amounts) == remainder:
            candidates.append((f"{chapter}{digits[:note_length]}", remainder))
    return max(candidates, key=lambda candidate: len(candidate[0])) if candidates else ("", value)


def _explicit_amount_sequence(value: str) -> list[str]:
    return [match.group() for match in _AMOUNT_RE.finditer(value)]


def _implicit_amount_pair(value: str, *, typical_groups: int) -> tuple[str, str] | None:
    if value.count(",") <= max(2, typical_groups):
        return None
    candidates: list[tuple[float, str, str]] = []
    for comma in (match.start() for match in re.finditer(",", value)):
        tail = value[comma + 1 :]
        if len(tail) < 4 or not tail[:2].isdigit():
            continue
        first = f"{value[:comma]}.{tail[:2]}"
        second = tail[2:]
        if not _FULL_AMOUNT_RE.fullmatch(first) or not _FULL_AMOUNT_RE.fullmatch(second):
            continue
        first_decimal = _decimal(first)
        second_decimal = _decimal(second)
        if first_decimal is None or second_decimal is None or not first_decimal or not second_decimal:
            continue
        ratio = float(max(abs(first_decimal), abs(second_decimal)) / min(abs(first_decimal), abs(second_decimal)))
        if ratio <= 100:
            candidates.append((abs(math.log10(ratio)), first, second))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.5:
        return None
    return candidates[0][1], candidates[0][2]


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _typical_comparative_groups(rows: list[Any], width: int) -> int:
    counts: list[int] = []
    for row in rows:
        blob = "".join(
            normalize_audit_text(getattr(cell, "text", "")).replace(" ", "")
            for cell in row_cells_by_column(row, width)
            if cell is not None
        )
        amounts = _explicit_amount_sequence(blob)
        if len(amounts) >= 2:
            counts.extend(amount.count(",") for amount in amounts[:2])
    return max(counts, default=2)


def _field_templates(cells: list[Any | None], values: list[str]) -> list[Any | None]:
    nonempty = [cell for cell, value in zip(cells, values, strict=True) if cell is not None and value]
    first = nonempty[0] if nonempty else next((cell for cell in cells if cell is not None), None)
    last = nonempty[-1] if nonempty else first
    combined = max(
        (pair for pair in zip(cells, values, strict=True) if pair[0] is not None),
        key=lambda pair: len(pair[1]),
        default=(first, ""),
    )[0]
    return [first, combined, combined, last]


def _projected_row(
    row: Any,
    values: list[str],
    templates: list[Any | None],
    row_index: int,
) -> Any:
    fallback = next((cell for cell in templates if cell is not None), None)
    if fallback is None:
        return row
    cells = []
    for column, (value, template) in enumerate(zip(values, templates, strict=True)):
        source = template or fallback
        cells.append(
            source.model_copy(
                deep=True,
                update={"text": value, "row_index": row_index, "col_index": column},
            )
        )
    return row.model_copy(deep=True, update={"cells": cells, "source_row_index": row_index})


def _annotate_recovered_records(
    segment: ProjectedSegment,
    recovery_rows: dict[int, dict[str, str]],
) -> None:
    for record in segment.records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        info = recovery_rows.get(int(source.get("source_row_index") or -1))
        if not info:
            continue
        source["recovery"] = info["mode"]
        source["recovery_raw"] = info["raw"]
        if info["mode"] in {"implicit_amount_split", "fragmented_amount_split", "unresolved_combined_amount"}:
            add_review_reason(record, info["mode"])


def _map_audit_balance_columns(segment: ProjectedSegment) -> ProjectedSegment:
    mapping = {
        "asset_item": "item",
        "liability_and_equity_item": "item",
        "liability_item": "item",
        "asset_line_no": "note_reference",
        "liability_line_no": "note_reference",
        "asset_note_ref": "note_reference",
        "liability_note_ref": "note_reference",
        "asset_ending_balance": "current_period_amount",
        "liability_ending_balance": "current_period_amount",
        "asset_opening_balance": "previous_period_amount",
        "liability_opening_balance": "previous_period_amount",
    }
    _remap_segment_keys(segment, mapping)
    return segment


def _normalize_balance_segment(segment: ProjectedSegment, *, continuation: bool) -> ProjectedSegment:
    if continuation:
        segment = _map_balance_continuation(segment)
    segment = _split_parallel_balance_records(segment)
    return _map_audit_balance_columns(segment)


def _split_parallel_balance_records(segment: ProjectedSegment) -> ProjectedSegment:
    """Split horizontal asset/liability pairs into one business fact per record."""

    asset_keys = [column.key for column in segment.columns if column.key.startswith("asset_")]
    liability_keys = [
        column.key
        for column in segment.columns
        if column.key.startswith("liability_") or column.key == "liability_and_equity_item"
    ]
    if not asset_keys or not liability_keys:
        return segment

    records: list[dict[str, Any]] = []
    source_refs: list[dict[str, Any]] = []
    for record in segment.records:
        for side, keys in (("assets", asset_keys), ("liabilities_and_equity", liability_keys)):
            split = _balance_side_record(record, keys, side=side)
            if split is None:
                continue
            records.append(split)
            source = split.get("source") if isinstance(split.get("source"), dict) else {}
            source_refs.append(
                {
                    "page": int(source.get("page") or segment.source_page),
                    "table_id": str(source.get("table_id") or segment.table_id),
                    "source_row_index": int(source.get("source_row_index") or 0),
                    "source_region": side,
                }
            )

    if not records:
        return segment
    retained_keys = set(asset_keys) | set(liability_keys)
    segment.columns = [column for column in segment.columns if column.key in retained_keys]
    segment.records = records
    segment.source_row_refs = source_refs
    for ordinal, record in enumerate(segment.records, start=1):
        record["record_id"] = f"{segment.dataset_id}:r{ordinal:06d}"
    return segment


def _balance_side_record(record: dict[str, Any], keys: list[str], *, side: str) -> dict[str, Any] | None:
    split = copy.deepcopy(record)
    has_value = False
    for pool_name in ("raw", "canonical_raw", "normalized"):
        pool = split.get(pool_name)
        if not isinstance(pool, dict):
            continue
        filtered = {key: pool.get(key, "") for key in keys}
        split[pool_name] = filtered
        if pool_name == "raw" and any(normalize_audit_text(value) for value in filtered.values()):
            has_value = True
    if not has_value:
        return None

    source = split.get("source") if isinstance(split.get("source"), dict) else {}
    refs = [
        ref
        for ref in source.get("source_cell_refs") or []
        if isinstance(ref, dict) and str(ref.get("field_name") or "") in keys
    ]
    source["source_cell_refs"] = refs
    source["source_region"] = side
    split["source_cell_refs"] = copy.deepcopy(refs)
    return split


def _map_comparative_columns(segment: ProjectedSegment) -> ProjectedSegment:
    mapping = {
        "column_02": "note_reference",
        "line_no": "note_reference",
    }
    _remap_segment_keys(segment, mapping)
    return segment


def _map_owner_equity_columns(segment: ProjectedSegment) -> ProjectedSegment:
    equity_keys = (
        "paid_in_capital",
        "preferred_shares",
        "perpetual_bonds",
        "other_equity_instruments",
        "capital_reserve",
        "treasury_shares",
        "other_comprehensive_income",
        "special_reserve",
        "surplus_reserve",
        "retained_earnings",
        "total_equity",
    )
    amount_columns = [column for column in segment.columns if column.key != "item"]
    if len(amount_columns) != len(equity_keys):
        return segment
    period_role = (
        "previous"
        if any("previous" in column.key or "上期" in normalize_audit_text(column.label) for column in amount_columns)
        else "current"
    )
    mapping = {column.key: key for column, key in zip(amount_columns, equity_keys, strict=True)}
    _remap_segment_keys(segment, mapping)
    segment.columns = [
        replace(column, label=_OWNER_EQUITY_COLUMN_LABELS.get(column.key, column.label)) for column in segment.columns
    ]
    if not any(column.key == "period_role" for column in segment.columns):
        insert_at = 1 if segment.columns and segment.columns[0].key == "item" else 0
        segment.columns.insert(
            insert_at, replace(segment.columns[0], source_index=-1, key="period_role", label="期间角色")
        )
    for record in segment.records:
        _set_statement_period_role(record, period_role)
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        item = normalize_audit_text(normalized.get("item"))
        normalized["item"] = re.sub(r"^-(?=\d+[.、])", "", item)
    return segment


def _set_statement_period_role(record: dict[str, Any], period_role: str) -> None:
    for pool_name in ("raw", "canonical_raw"):
        pool = record.get(pool_name)
        if isinstance(pool, dict):
            pool.pop("period_role", None)
    normalized = record.setdefault("normalized", {})
    normalized["period_role"] = period_role
    source = record.setdefault("source", {})
    field_sources = source.setdefault("field_sources", {})
    field_sources["period_role"] = {
        "source": "derived.statement_period_role",
        "page": source.get("page"),
        "derivation": "source_statement_period_header",
    }


def _remap_segment_keys(segment: ProjectedSegment, mapping: dict[str, str]) -> None:
    columns = [replace(column, key=mapping.get(column.key, column.key)) for column in segment.columns]
    segment.columns = _unique_columns(columns)
    for record in segment.records:
        for pool_name in ("raw", "canonical_raw", "normalized"):
            pool = record.get(pool_name)
            if isinstance(pool, dict):
                record[pool_name] = _remap_values(pool, mapping)
        _remap_record_sources(record, mapping)
        _drop_review_reason(record, "financial_line_no_invalid")


def _unique_columns(columns: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for column in columns:
        if column.key not in seen:
            seen.add(column.key)
            result.append(column)
    return result


def _remap_values(values: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in values.items():
        target = mapping.get(key, key)
        if target not in remapped or remapped[target] in (None, ""):
            remapped[target] = value
    return remapped


def _remap_record_sources(record: dict[str, Any], mapping: dict[str, str]) -> None:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    refs = source.get("source_cell_refs") or []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("field_name") in mapping:
            ref["field_name"] = mapping[ref["field_name"]]
    if isinstance(record.get("source_cell_refs"), list):
        record["source_cell_refs"] = copy.deepcopy(refs)
    field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else None
    if field_sources is not None:
        source["field_sources"] = {mapping.get(key, key): value for key, value in field_sources.items()}


def _drop_review_reason(record: dict[str, Any], reason: str) -> None:
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    retained = [value for value in review.get("reasons") or [] if value != reason]
    if retained:
        review["reasons"] = retained
    elif review:
        record.pop("review", None)


def _drop_note_line_warnings(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if not str(warning).startswith("FINANCIAL_LINE_NO_INVALID")]


def _map_balance_continuation(segment: ProjectedSegment) -> ProjectedSegment:
    mapping = {
        "asset_item": "liability_and_equity_item",
        "asset_line_no": "liability_line_no",
        "asset_note_ref": "liability_note_ref",
        "asset_ending_balance": "liability_ending_balance",
        "asset_opening_balance": "liability_opening_balance",
    }
    _remap_segment_keys(segment, mapping)
    return segment


def _recover_rotated_owner_equity(
    table: Any,
    *,
    page_number: int,
    dataset_id: str,
) -> tuple[ProjectedSegment | None, list[str]]:
    width = table_width(table)
    rows = list(getattr(table, "rows", None) or [])
    if width < 8 or len(rows) < 4:
        return None, []
    candidates: list[tuple[int, int, str, ProjectedSegment, list[str]]] = []
    for reverse_rows in (False, True):
        for reverse_columns in (False, True):
            orientation = f"transpose:r{int(reverse_rows)}:c{int(reverse_columns)}"
            adapted = _transpose_table(table, reverse_rows=reverse_rows, reverse_columns=reverse_columns)
            score = _owner_equity_table_score(adapted)
            if score < 4:
                continue
            segment, warnings = project_statement_table(
                adapted,
                page_number=page_number,
                dataset_id=dataset_id,
                kind="owners_equity_changes",
            )
            if segment is not None and len(segment.records) >= 3:
                candidates.append((score, len(segment.records), orientation, segment, warnings))
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        return None, [f"AUDIT_OWNER_EQUITY_ROTATION_AMBIGUOUS:page={page_number}"]
    _score, _rows, orientation, segment, warnings = candidates[0]
    for record in segment.records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        source["recovery"] = "rotated_table_transpose"
        source["orientation"] = orientation
    return segment, [*warnings, f"AUDIT_OWNER_EQUITY_ROTATION_RECOVERED:page={page_number}:orientation={orientation}"]


def _transpose_table(table: Any, *, reverse_rows: bool, reverse_columns: bool) -> Any:
    source_rows = list(getattr(table, "rows", None) or [])
    width = table_width(table)
    matrix = [row_cells_by_column(row, width) for row in source_rows]
    row_indexes = list(range(len(matrix)))
    column_indexes = list(range(width))
    if reverse_rows:
        row_indexes.reverse()
    if reverse_columns:
        column_indexes.reverse()
    projected_rows: list[Any] = []
    for new_row_index, source_column in enumerate(column_indexes):
        source_cells = [matrix[source_row][source_column] for source_row in row_indexes]
        fallback = next((cell for cell in source_cells if cell is not None), None)
        if fallback is None:
            continue
        cells = [
            (cell or fallback).model_copy(
                deep=True,
                update={
                    "text": str(getattr(cell, "text", "") or "") if cell is not None else "",
                    "row_index": new_row_index,
                    "col_index": new_column,
                },
            )
            for new_column, cell in enumerate(source_cells)
        ]
        template_row = source_rows[row_indexes[0]]
        projected_rows.append(
            template_row.model_copy(
                deep=True,
                update={"cells": cells, "source_row_index": source_column},
            )
        )
    metadata = copy.deepcopy(getattr(table, "metadata", None) or {})
    metadata["audit_rotation_recovery"] = True
    return table.model_copy(update={"headers": [], "rows": projected_rows, "metadata": metadata})


def _owner_equity_table_score(table: Any) -> int:
    rows = list(getattr(table, "rows", None) or [])[:8]
    values = [
        normalize_audit_text(getattr(cell, "text", ""))
        for row in rows
        for cell in getattr(row, "cells", None) or []
        if normalize_audit_text(getattr(cell, "text", ""))
    ]
    first_row = (
        [normalize_audit_text(getattr(cell, "text", "")) for cell in (getattr(rows[0], "cells", None) or [])]
        if rows
        else []
    )
    positional = 5 if first_row and "项目" in first_row[0] else 0
    positional += sum(bool(_TABLE_HEADER_MARKERS.search(value)) for value in first_row)
    marker_score = sum(bool(_OWNER_EQUITY_MARKERS.search(value)) for value in values)
    amount_score = min(3, sum(bool(_FULL_AMOUNT_RE.fullmatch(value.replace(" ", ""))) for value in values))
    return positional + marker_score * 2 + amount_score


def _merge_statement_segments(segments: list[ProjectedSegment]) -> list[ProjectedSegment]:
    merged: list[ProjectedSegment] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        previous_pages = dataset_pages(previous.records) if previous is not None else set()
        current_pages = dataset_pages(segment.records)
        adjacent = bool(previous_pages and current_pages and min(current_pages) == max(previous_pages) + 1)
        if previous is None or previous.kind != segment.kind or not adjacent:
            merged.append(segment)
            continue
        keys = {column.key for column in previous.columns}
        previous.columns.extend(column for column in segment.columns if column.key not in keys)
        previous.records.extend(segment.records)
        previous.source_row_refs.extend(segment.source_row_refs)
        previous.row_groups.extend(segment.row_groups)
        for ordinal, record in enumerate(previous.records, start=1):
            record["record_id"] = f"{previous.dataset_id}:r{ordinal:06d}"
    return merged


def embedded_financial_pages(parse_result: Any) -> set[int]:
    """Return pages reserved for the report's four primary statements."""

    pages = list(getattr(parse_result, "pages", None) or [])
    exact = {int(getattr(page, "page_number", 0) or 1) for page in pages if statement_kind(page) is not None}
    notes_start = next(
        (
            int(getattr(page, "page_number", 0) or 1)
            for page in pages
            if any("财务报表附注" in normalize_audit_text(line) for line in page_lines(page))
        ),
        0,
    )
    if not exact:
        return exact
    first = min(exact)
    last = notes_start if notes_start else max(exact) + 3
    for page in pages:
        page_number = int(getattr(page, "page_number", 0) or 1)
        if first <= page_number < last and _landscape_table_candidate(page, parse_result=parse_result):
            exact.add(page_number)
    return exact


def _landscape_table_candidate(page: Any, *, parse_result: Any | None = None) -> bool:
    page_number = int(getattr(page, "page_number", 0) or 1)
    tables = list(getattr(page, "tables", None) or [])
    if parse_result is not None:
        tables.extend(_logical_tables_for_page(parse_result, page_number))
    return max((table_width(table) for table in tables), default=0) >= 8


def _logical_tables_for_page(parse_result: Any, page_number: int) -> list[Any]:
    """Return single-page logical tables without assigning cross-page rows to one page."""

    candidates: list[Any] = []
    seen: set[str] = set()
    for table in getattr(parse_result, "logical_tables", None) or []:
        source_pages = sorted(
            {
                int(value)
                for value in (getattr(table, "source_pages", None) or [])
                if str(value).isdigit() and int(value) > 0
            }
        )
        if source_pages != [page_number]:
            continue
        table_id = str(getattr(table, "table_id", "") or getattr(table, "logical_id", "") or id(table))
        if table_id in seen:
            continue
        seen.add(table_id)
        candidates.append(table)
    return candidates


def _statement_continuation(page: Any, kind: str) -> bool:
    for line in page_lines(page):
        compact = normalize_audit_text(line).replace(" ", "")
        if any(candidate == kind and pattern.fullmatch(compact) for pattern, candidate in _STATEMENT_TITLE_PATTERNS):
            return "续" in compact
    return False


def resolve_note_table_candidates(
    datasets: dict[str, list[dict[str, Any]]],
    parse_result: Any,
) -> list[str]:
    """Resolve audit-note rows to physical tables, split mixed tables, and remove duplicate candidates."""

    warnings: list[str] = []
    table_index = _source_tables(parse_result)
    logical_pages = _logical_source_pages(parse_result)
    physical_tables = list(table_index.values())
    for name, rows in datasets.items():
        if not rows or re.sub(r"_\d+$", "", name) in _STATEMENT_KINDS or not _logical_dataset(rows):
            continue
        previous_table_id = ""
        for record in rows:
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            logical_id = str(source.get("table_id") or "").split(":segment_", 1)[0]
            pages = logical_pages.get(logical_id) or dataset_pages([record])
            candidates = [item for item in physical_tables if not pages or item.page in pages]
            match = _match_record_to_physical_row(record, candidates, previous_table_id=previous_table_id)
            if match is None:
                _discard_unverified_physical_source(record)
                continue
            _apply_physical_row_match(record, match)
            previous_table_id = _source_table_id(match.source_table)
        recovered_headers = _resolve_unmatched_header_records(rows, table_index)
        if recovered_headers:
            warnings.append(f"AUDIT_NOTE_HEADER_SOURCE_RECOVERED:dataset={name}:rows={recovered_headers}")
        for record in rows:
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            if source.get("source_resolution") != "unresolved_logical_row":
                continue
            logical_id = str(source.get("table_id") or "").split(":segment_", 1)[0]
            warnings.append(
                f"AUDIT_NOTE_PHYSICAL_SOURCE_UNRESOLVED:dataset={name}:"
                f"table={logical_id}:row={source.get('table_row_index', '')}"
            )

    warnings.extend(_split_mixed_logical_datasets(datasets, parse_result, table_index))
    warnings.extend(_remove_duplicate_note_datasets(datasets))
    return list(dict.fromkeys(warnings))


def _logical_source_pages(parse_result: Any) -> dict[str, set[int]]:
    pages: dict[str, set[int]] = {}
    for table in getattr(parse_result, "logical_tables", None) or []:
        table_id = str(getattr(table, "logical_id", "") or getattr(table, "table_id", ""))
        if not table_id:
            continue
        pages[table_id] = {
            int(page) for page in (getattr(table, "source_pages", None) or []) if str(page).isdigit() and int(page) > 0
        }
    return pages


def _logical_dataset(rows: list[dict[str, Any]]) -> bool:
    return any(
        str((record.get("source") or {}).get("table_id") or "").split(":segment_", 1)[0].startswith("lt_")
        for record in rows
        if isinstance(record.get("source"), dict)
    )


def _match_record_to_physical_row(
    record: dict[str, Any],
    source_tables: list[_SourceTable],
    *,
    previous_table_id: str,
) -> _PhysicalRowMatch | None:
    raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else record.get("raw")
    if not isinstance(raw, dict):
        return None
    record_values = [_match_token(value) for value in raw.values() if _match_token(value)]
    if not record_values:
        return None
    keys = [_match_token(key) for key in raw]
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    declared_page = int(source.get("page") or 0)
    declared_physical = str(source.get("physical_table_id") or "")
    matches: list[_PhysicalRowMatch] = []
    for source_table in source_tables:
        header_tokens = _source_table_header_tokens(source_table.table)
        header_text = " ".join(header_tokens)
        header_score = sum(
            (2 if key and key in header_text else 0) + sum(bool(key and header in key) for header in header_tokens)
            for key in keys
        )
        width = table_width(source_table.table)
        for row_index, row in enumerate(getattr(source_table.table, "rows", None) or []):
            cells = row_cells_by_column(row, width)
            physical_values = [_match_token(getattr(cell, "text", "")) for cell in cells]
            physical_values = [value for value in physical_values if value]
            overlap = _row_value_overlap(record_values, physical_values)
            coverage = overlap / max(1, len(record_values))
            if not _credible_physical_match(record_values, overlap, coverage):
                continue
            source_row_index = int(getattr(row, "source_row_index", row_index) or 0)
            matches.append(
                _PhysicalRowMatch(
                    source_table=source_table,
                    row=row,
                    row_index=row_index,
                    source_row_index=source_row_index,
                    matched_values=overlap,
                    coverage=coverage,
                    header_score=header_score,
                )
            )
    if not matches:
        return None

    def rank(match: _PhysicalRowMatch) -> tuple[int, float, int, int, int, int]:
        table_id = _source_table_id(match.source_table)
        return (
            match.matched_values,
            match.coverage,
            match.header_score,
            int(bool(previous_table_id and table_id == previous_table_id)),
            int(bool(declared_physical and table_id == declared_physical)),
            -abs(match.source_table.page - declared_page) if declared_page else 0,
        )

    matches.sort(key=rank, reverse=True)
    best = matches[0]
    best_rank = rank(best)
    tied = [match for match in matches if rank(match) == best_rank]
    table_ids = {_source_table_id(match.source_table) for match in tied}
    return best if len(table_ids) == 1 else None


def _credible_physical_match(values: list[str], overlap: int, coverage: float) -> bool:
    if len(values) >= 2:
        return overlap >= 2 and coverage >= 0.6
    value = values[0]
    return (
        overlap == 1
        and not _AMOUNT_HEADER_RE.fullmatch(value)
        and value not in {"合计", "总计", "小计"}
        and len(value) >= 4
    )


def _row_value_overlap(record_values: list[str], physical_values: list[str]) -> int:
    record_counts = Counter(record_values)
    physical_counts = Counter(physical_values)
    exact = record_counts & physical_counts
    overlap = sum(exact.values())
    remaining = list((record_counts - exact).elements())
    joined = "".join(physical_values)
    overlap += sum(
        bool(value and not _AMOUNT_HEADER_RE.fullmatch(value) and len(value) >= 4 and value in joined)
        for value in remaining
    )
    return overlap


def _match_token(value: Any) -> str:
    return normalize_audit_text(value).replace(" ", "").replace("（", "(").replace("）", ")")


def _source_table_header_tokens(table: Any) -> list[str]:
    values = [*(getattr(table, "headers", None) or [])]
    for row in list(getattr(table, "rows", None) or [])[:3]:
        values.extend(getattr(cell, "text", "") for cell in (getattr(row, "cells", None) or []))
    return [_match_token(value) for value in values if _match_token(value)]


def _source_table_id(source_table: _SourceTable) -> str:
    return str(getattr(source_table.table, "table_id", "") or f"pt_{source_table.page}_{source_table.index}")


def _apply_physical_row_match(record: dict[str, Any], match: _PhysicalRowMatch) -> None:
    source = record.setdefault("source", {})
    table_id = _source_table_id(match.source_table)
    page = int(getattr(match.row, "source_page", 0) or match.source_table.page)
    source.update(
        {
            "page": page,
            "page_range": [page, page],
            "physical_table_id": table_id,
            "source_row_index": match.source_row_index,
            "physical_table_row_index": match.row_index,
            "source_resolution": "matched_physical_row",
        }
    )
    raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else record.get("raw")
    raw = raw if isinstance(raw, dict) else {}
    cells = row_cells_by_column(match.row, table_width(match.source_table.table))
    positions = _record_cell_positions(raw, cells)
    refs: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for field_name, cell_index in zip(raw, positions, strict=True):
        if cell_index is None or cell_index >= len(cells):
            continue
        cell = cells[cell_index]
        evidence_ids.extend(str(value) for value in (getattr(cell, "evidence_ids", None) or []) if value)
        cell_refs = [dict(ref) for ref in (getattr(cell, "source_cell_refs", None) or []) if isinstance(ref, dict)]
        if not cell_refs:
            cell_refs = [{}]
        for ref in cell_refs:
            ref.update(
                {
                    "page": page,
                    "table_id": table_id,
                    "row": match.source_row_index,
                    "col": cell_index,
                    "field_name": field_name,
                }
            )
            bbox = getattr(cell, "bbox", None)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                ref.setdefault("bbox", [float(value) for value in bbox])
            refs.append(ref)
    evidence_ids.extend(
        str(value) for value in (getattr(match.source_table.table, "evidence_ids", None) or []) if value
    )
    if refs:
        source["source_cell_refs"] = refs
        record["source_cell_refs"] = copy.deepcopy(refs)
    if evidence_ids:
        unique_evidence = list(dict.fromkeys(evidence_ids))
        source["evidence_ids"] = unique_evidence
        record["evidence_ids"] = unique_evidence


def _record_cell_positions(raw: dict[str, Any], cells: list[Any]) -> list[int | None]:
    cell_values = [_match_token(getattr(cell, "text", "")) for cell in cells]
    used: set[int] = set()
    positions: list[int | None] = []
    positional = len(cells) == len(raw)
    for index, value in enumerate(raw.values()):
        token = _match_token(value)
        if positional and (not token or cell_values[index] == token):
            positions.append(index)
            used.add(index)
            continue
        match = next(
            (
                cell_index
                for cell_index, cell_value in enumerate(cell_values)
                if cell_index not in used and cell_value == token and token
            ),
            None,
        )
        positions.append(match)
        if match is not None:
            used.add(match)
    return positions


def _discard_unverified_physical_source(record: dict[str, Any]) -> None:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    source.pop("physical_table_id", None)
    source.pop("source_cell_refs", None)
    source.pop("evidence_ids", None)
    source["source_resolution"] = "unresolved_logical_row"
    record.pop("source_cell_refs", None)
    record.pop("evidence_ids", None)
    review = record.setdefault("review", {"required": True, "reasons": []})
    review["required"] = True
    reasons = review.setdefault("reasons", [])
    if "physical_source_unresolved" not in reasons:
        reasons.append("physical_source_unresolved")


def _resolve_unmatched_header_records(
    rows: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
) -> int:
    recovered = 0
    promoted_table_ids: set[str] = set()
    for index, record in enumerate(rows):
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        if source.get("source_resolution") != "unresolved_logical_row":
            continue
        neighbor_ids = [
            str((rows[position].get("source") or {}).get("physical_table_id") or "")
            for position in (index - 1, index + 1)
            if 0 <= position < len(rows)
        ]
        neighbor_ids = [table_id for table_id in neighbor_ids if table_id]
        if not neighbor_ids or len(set(neighbor_ids)) != 1:
            continue
        source_table = table_index.get(neighbor_ids[0])
        if source_table is None or not _record_matches_table_headers(record, source_table.table):
            continue
        _apply_table_header_match(record, source_table)
        promoted_table_ids.add(_source_table_id(source_table))
        recovered += 1
    for table_id in promoted_table_ids:
        _rebase_promoted_header_rows(rows, table_id)
    return recovered


def _rebase_promoted_header_rows(rows: list[dict[str, Any]], table_id: str) -> None:
    table_rows = [
        record for record in rows if str((record.get("source") or {}).get("physical_table_id") or "") == table_id
    ]
    for logical_row_index, record in enumerate(table_rows):
        source = record.setdefault("source", {})
        physical_row_index = source.get("source_row_index")
        if physical_row_index is not None:
            source["physical_source_row_index"] = physical_row_index
        source["source_row_index"] = logical_row_index


def _record_matches_table_headers(record: dict[str, Any], table: Any) -> bool:
    raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else record.get("raw")
    values = (
        [_match_token(value) for value in (raw or {}).values() if _match_token(value)] if isinstance(raw, dict) else []
    )
    headers = [_match_token(value) for value in (getattr(table, "headers", None) or []) if _match_token(value)]
    header_text = "".join(headers)
    return bool(values and headers and all(value in headers or value in header_text for value in values))


def _apply_table_header_match(record: dict[str, Any], source_table: _SourceTable) -> None:
    source = record.setdefault("source", {})
    table_id = _source_table_id(source_table)
    source.update(
        {
            "page": source_table.page,
            "page_range": [source_table.page, source_table.page],
            "physical_table_id": table_id,
            "source_row_index": -1,
            "physical_table_row_index": -1,
            "source_resolution": "matched_physical_table_header",
            "recovery": "source_table_header_was_data",
        }
    )
    raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else record.get("raw")
    refs = [
        {
            "page": source_table.page,
            "table_id": table_id,
            "row": -1,
            "col": index,
            "field_name": field_name,
            "recovery": "source_table_header_was_data",
        }
        for index, field_name in enumerate((raw or {}).keys())
    ]
    evidence_ids = [str(value) for value in (getattr(source_table.table, "evidence_ids", None) or []) if value]
    if refs:
        source["source_cell_refs"] = refs
        record["source_cell_refs"] = copy.deepcopy(refs)
    if evidence_ids:
        source["evidence_ids"] = evidence_ids
        record["evidence_ids"] = evidence_ids
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    reasons = [reason for reason in (review.get("reasons") or []) if reason != "physical_source_unresolved"]
    if evidence_ids:
        if reasons:
            review["reasons"] = reasons
            review["required"] = True
        else:
            record.pop("review", None)
    else:
        review.update(required=True, reasons=[*reasons, "table_header_evidence_missing"])
        record["review"] = review


def _split_mixed_logical_datasets(
    datasets: dict[str, list[dict[str, Any]]],
    parse_result: Any,
    table_index: dict[str, _SourceTable],
) -> list[str]:
    warnings: list[str] = []
    rewritten: dict[str, list[dict[str, Any]]] = {}
    for name, rows in datasets.items():
        groups = _consecutive_physical_groups(rows) if _logical_dataset(rows) else []
        subjects = [
            _subject_slug(_dataset_caption(group, parse_result, table_index))
            or _column_subject_slug(record_keys(group))
            for group in groups
        ]
        distinct = {subject for subject in subjects if subject}
        if len(groups) < 2 or len(distinct) < 2:
            rewritten[name] = rows
            continue
        for ordinal, group in enumerate(groups, start=1):
            split_name = f"{name}__split_{ordinal:02d}"
            rewritten[split_name] = group
            warnings.append(
                f"AUDIT_NOTE_MIXED_TABLE_SPLIT:from={name}:to={split_name}:"
                f"physical={((group[0].get('source') or {}).get('physical_table_id') or '')}"
            )
    datasets.clear()
    datasets.update(rewritten)
    return warnings


def _consecutive_physical_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    previous = ""
    for record in rows:
        physical = str((record.get("source") or {}).get("physical_table_id") or "")
        if not physical:
            return []
        if not groups or physical != previous:
            groups.append([])
        groups[-1].append(record)
        previous = physical
    return groups


def _remove_duplicate_note_datasets(datasets: dict[str, list[dict[str, Any]]]) -> list[str]:
    warnings: list[str] = []
    names = list(datasets)
    removed: set[str] = set()
    for index, left_name in enumerate(names):
        if left_name in removed or left_name not in datasets:
            continue
        for right_name in names[index + 1 :]:
            if right_name in removed or right_name not in datasets:
                continue
            left, right = datasets[left_name], datasets[right_name]
            if not _duplicate_dataset_pair(left, right):
                continue
            keep, drop = max(
                ((left_name, right_name), (right_name, left_name)),
                key=lambda pair: _dataset_candidate_rank(datasets[pair[0]]),
            )
            _retain_suppressed_source_ids(datasets[keep], datasets[drop])
            datasets.pop(drop, None)
            removed.add(drop)
            warnings.append(f"AUDIT_NOTE_DUPLICATE_DATASET_REMOVED:removed={drop}:kept={keep}")
            if drop == left_name:
                break
    return warnings


def _retain_suppressed_source_ids(kept: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> None:
    kept_physical = _dataset_physical_ids(kept)
    suppressed = {
        str((record.get("source") or {}).get("table_id") or "").split(":segment_", 1)[0]
        for record in dropped
        if (record.get("source") or {}).get("table_id")
    } - kept_physical
    if not suppressed:
        return
    for record in kept:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        existing = [str(value) for value in (source.get("suppressed_source_table_ids") or []) if value]
        source["suppressed_source_table_ids"] = list(dict.fromkeys([*existing, *sorted(suppressed)]))


def _duplicate_dataset_pair(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    left_physical = _dataset_physical_ids(left)
    right_physical = _dataset_physical_ids(right)
    if not left_physical.intersection(right_physical):
        return False
    left_rows = Counter(filter(None, (_record_value_fingerprint(record) for record in left)))
    right_rows = Counter(filter(None, (_record_value_fingerprint(record) for record in right)))
    if not left_rows or not right_rows:
        return False
    smaller, larger = (
        (left_rows, right_rows) if sum(left_rows.values()) <= sum(right_rows.values()) else (right_rows, left_rows)
    )
    return all(larger[fingerprint] >= count for fingerprint, count in smaller.items())


def _dataset_physical_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str((record.get("source") or {}).get("physical_table_id") or "")
        for record in rows
        if (record.get("source") or {}).get("physical_table_id")
    }


def _record_value_fingerprint(record: dict[str, Any]) -> tuple[str, ...]:
    raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else record.get("raw")
    return (
        tuple(value for value in (_match_token(item) for item in (raw or {}).values()) if value)
        if isinstance(raw, dict)
        else ()
    )


def _dataset_candidate_rank(rows: list[dict[str, Any]]) -> tuple[int, int, int, int, int]:
    keys = set(_record_pool_keys(rows))
    source_fields = {
        str(ref.get("field_name"))
        for record in rows
        for ref in ((record.get("source") or {}).get("source_cell_refs") or [])
        if isinstance(ref, dict) and ref.get("field_name")
    }
    field_coverage = round(1000 * len(keys.intersection(source_fields)) / len(source_fields)) if source_fields else 1000
    logical_rows = sum(
        str((record.get("source") or {}).get("table_id") or "").split(":segment_", 1)[0].startswith("lt_")
        for record in rows
    )
    evidence = sum(bool((record.get("source") or {}).get("evidence_ids")) for record in rows)
    return field_coverage, len(keys), len(rows), -logical_rows, evidence


def merge_horizontal_note_continuations(
    datasets: dict[str, list[dict[str, Any]]],
    parse_result: Any,
) -> list[str]:
    """Join adjacent source tables that continue the same business rows horizontally."""

    warnings: list[str] = []
    table_index = _source_tables(parse_result)
    names = list(datasets)
    cursor = 0
    while cursor + 1 < len(names):
        left_name, right_name = names[cursor], names[cursor + 1]
        left, right = datasets[left_name], datasets[right_name]
        if not _horizontal_continuation_pair(left, right, parse_result, table_index):
            cursor += 1
            continue
        datasets[left_name] = [
            _merge_horizontal_records(left_record, right_record)
            for left_record, right_record in zip(left, right, strict=True)
        ]
        datasets.pop(right_name)
        names.pop(cursor + 1)
        warnings.append(f"AUDIT_NOTE_HORIZONTAL_TABLE_MERGED:from={right_name}:to={left_name}:rows={len(left)}")
    return warnings


def _horizontal_continuation_pair(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    parse_result: Any,
    table_index: dict[str, _SourceTable],
) -> bool:
    if len(left) < 2 or len(left) != len(right):
        return False
    left_table = _dataset_source_table(left, table_index)
    right_table = _dataset_source_table(right, table_index)
    if (
        left_table is None
        or right_table is None
        or left_table.page != right_table.page
        or right_table.index != left_table.index + 1
    ):
        return False
    left_keys, right_keys = record_keys(left), record_keys(right)
    if len(left_keys) < 2 or len(right_keys) < 2 or left_keys == right_keys:
        return False
    left_anchors = [_match_token((record.get("raw") or {}).get(left_keys[0])) for record in left]
    right_anchors = [_match_token((record.get("raw") or {}).get(right_keys[0])) for record in right]
    if not all(left_anchors) or left_anchors != right_anchors:
        return False
    left_subject = _subject_slug(_dataset_caption(left, parse_result, table_index)) or _column_subject_slug(left_keys)
    right_subject = _subject_slug(_dataset_caption(right, parse_result, table_index)) or _column_subject_slug(
        right_keys
    )
    return not (left_subject and right_subject and left_subject != right_subject)


def _merge_horizontal_records(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(left)
    left_keys = list((left.get("raw") or {}).keys())
    right_keys = list((right.get("raw") or {}).keys())
    duplicate_anchor = right_keys[0] if left_keys and right_keys else ""
    for pool_name in ("normalized", "canonical_raw", "raw"):
        left_pool = merged.get(pool_name) if isinstance(merged.get(pool_name), dict) else {}
        right_pool = right.get(pool_name) if isinstance(right.get(pool_name), dict) else {}
        for key, value in right_pool.items():
            if key == duplicate_anchor and key in left_pool:
                continue
            target = key
            suffix = 2
            while target in left_pool:
                target = f"{key}_{suffix}"
                suffix += 1
            left_pool[target] = copy.deepcopy(value)
        merged[pool_name] = left_pool
    left_source = merged.get("source") if isinstance(merged.get("source"), dict) else {}
    right_source = right.get("source") if isinstance(right.get("source"), dict) else {}
    physical_ids = [
        str(value)
        for value in (
            left_source.get("physical_table_id"),
            right_source.get("physical_table_id"),
        )
        if value
    ]
    left_source["physical_table_ids"] = list(dict.fromkeys(physical_ids))
    if len(physical_ids) > 1:
        suppressed = [str(value) for value in (left_source.get("suppressed_source_table_ids") or []) if value]
        left_source["suppressed_source_table_ids"] = list(dict.fromkeys([*suppressed, *physical_ids[1:]]))
    left_source["horizontal_continuation"] = True
    refs = [
        copy.deepcopy(ref)
        for source in (left_source, right_source)
        for ref in (source.get("source_cell_refs") or [])
        if isinstance(ref, dict)
    ]
    evidence_ids = [
        str(value) for source in (left_source, right_source) for value in (source.get("evidence_ids") or []) if value
    ]
    if refs:
        left_source["source_cell_refs"] = refs
        merged["source_cell_refs"] = copy.deepcopy(refs)
    if evidence_ids:
        left_source["evidence_ids"] = list(dict.fromkeys(evidence_ids))
        merged["evidence_ids"] = list(dict.fromkeys(evidence_ids))
    merged["source"] = left_source
    return merged


def repair_stacked_note_headers(
    datasets: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Replace residual generic columns using stable audit-note stacked-header schemas."""

    warnings: list[str] = []
    for name, rows in datasets.items():
        source_keys = record_keys(rows)
        target_keys = _stacked_note_schema(source_keys)
        if not target_keys or source_keys == target_keys:
            continue
        datasets[name] = [_remap_record(record, source_keys, target_keys) for record in rows]
        warnings.append(f"AUDIT_NOTE_STACKED_HEADERS_RECOVERED:dataset={name}:columns={len(target_keys)}")
    return warnings


def _stacked_note_schema(keys: list[str]) -> list[str]:
    compact = [normalize_audit_text(key).replace(" ", "") for key in keys]
    generic = [bool(re.fullmatch(r"(?:col|column)_\d+", key)) for key in compact]
    if len(keys) == 4 and generic[:2] == [True, True] and "坏账准备" in compact[2] and "计提比例" in compact[3]:
        return ["名称", "期末余额/应收账款", "期末余额/坏账准备", "期末余额/计提比例(%)"]
    if len(keys) == 4 and generic[:2] == [True, True] and "占应收账款" in compact[2]:
        return ["单位名称", "期末余额/应收账款", keys[2], keys[3]]
    if len(keys) == 6 and generic[:2] == [True, True] and {"计提", "收回或转回", "核销"} <= set(compact):
        return ["项目", "期初余额", "本期变动金额/计提", "本期变动金额/收回或转回", "本期变动金额/核销", "期末余额"]
    period_match = re.fullmatch(r"(期初余额|期末余额)/账面余额/金额", compact[1]) if len(keys) == 6 else None
    period = period_match.group(1) if period_match else ""
    if (
        period
        and compact[0] == "类别"
        and compact[2] in {"比例(%)", f"{period}/账面余额/比例(%)"}
        and compact[3] == f"{period}/坏账准备/金额"
        and compact[4] in {"计提比例(%)", f"{period}/坏账准备/计提比例(%)"}
        and compact[5] == f"{period}/账面价值"
    ):
        return [
            "类别",
            f"{period}/账面余额/金额",
            f"{period}/账面余额/比例(%)",
            f"{period}/坏账准备/金额",
            f"{period}/坏账准备/计提比例(%)",
            f"{period}/账面价值",
        ]
    if (
        len(keys) == 6
        and generic[:2] == [True, True]
        and compact[2:6] == ["账面余额", "坏账准备", "账面余额_2", "坏账准备_2"]
    ):
        return [
            "项目",
            "关联方名称",
            "期末余额/账面余额",
            "期末余额/坏账准备",
            "期初余额/账面余额",
            "期初余额/坏账准备",
        ]
    return []


def merge_cross_page_continuations(
    datasets: dict[str, list[dict[str, Any]]],
    parse_result: Any,
) -> list[str]:
    """Merge only adjacent tables whose source headers prove a continuation."""

    table_index = _source_tables(parse_result)
    warnings = _merge_orphan_total_tables(datasets, table_index)
    names = list(datasets)
    cursor = 0
    while cursor + 1 < len(names):
        left_name, right_name = names[cursor], names[cursor + 1]
        if left_name not in datasets or right_name not in datasets:
            cursor += 1
            continue
        left, right = datasets[left_name], datasets[right_name]
        if not _adjacent_edge_tables(left, right, table_index):
            cursor += 1
            continue
        left_keys, right_keys = record_keys(left), record_keys(right)
        if _numeric_header_continuation(left_keys, right_keys):
            recovered = _header_record(right_name, left_keys, right_keys, right, table_index)
            merged = [*left, recovered, *[_remap_record(row, right_keys, left_keys) for row in right]]
            promoted_table_id = str((recovered.get("source") or {}).get("physical_table_id") or "")
            if promoted_table_id:
                _rebase_promoted_header_rows(merged, promoted_table_id)
            datasets[left_name] = merged
            datasets.pop(right_name)
            names.pop(cursor + 1)
            warnings.append(f"AUDIT_CROSS_PAGE_TABLE_REPAIRED:from={right_name}:to={left_name}:mode=data_header")
            continue
        if _repeated_header_continuation(left_keys, right_keys) and not _dataset_caption(
            right,
            parse_result,
            table_index,
        ):
            datasets[left_name] = [*left, *right]
            datasets.pop(right_name)
            names.pop(cursor + 1)
            warnings.append(f"AUDIT_CROSS_PAGE_TABLE_REPAIRED:from={right_name}:to={left_name}:mode=repeated_header")
            continue
        category_values = _category_continuation_values(parse_result, left, right, left_keys, right_keys)
        if category_values:
            target_keys = left_keys[1 : 1 + len(right_keys)]
            remapped = [
                _remap_record(
                    row,
                    right_keys,
                    target_keys,
                    prefix={left_keys[0]: category.value},
                    suffix={left_keys[-1]: ""},
                    injected={left_keys[0]: category},
                )
                for row, category in zip(right, category_values, strict=True)
            ]
            datasets[left_name] = [*left, *remapped]
            datasets.pop(right_name)
            names.pop(cursor + 1)
            warnings.append(f"AUDIT_CROSS_PAGE_TABLE_REPAIRED:from={right_name}:to={left_name}:mode=missing_outer")
            continue
        cursor += 1
    return warnings


def _merge_orphan_total_tables(
    datasets: dict[str, list[dict[str, Any]]],
    table_index: dict[str, _SourceTable],
) -> list[str]:
    """Attach a headerless total table at the next page boundary to its source dataset."""

    warnings: list[str] = []
    for candidate in _orphan_total_candidates(datasets, table_index):
        records = datasets[candidate.dataset_name]
        recovered = [
            _source_row_record(
                candidate.dataset_name,
                list(candidate.keys),
                candidate.source_table,
                row,
                row_index,
            )
            for row_index, row in enumerate(candidate.rows)
        ]
        records.extend(recovered)
        warnings.append(
            f"AUDIT_CROSS_PAGE_TABLE_REPAIRED:table={_source_table_id(candidate.source_table)}:"
            f"to={candidate.dataset_name}:mode=orphan_total:rows={len(recovered)}"
        )
    return warnings


def _orphan_total_candidates(
    datasets: dict[str, list[dict[str, Any]]],
    table_index: dict[str, _SourceTable],
) -> list[_OrphanTotalCandidate]:
    positions = {(source.page, source.index): source for source in table_index.values()}
    last_index_by_page: dict[int, int] = {}
    for source in table_index.values():
        last_index_by_page[source.page] = max(last_index_by_page.get(source.page, -1), source.index)
    represented = {table_id for rows in datasets.values() for table_id in _dataset_physical_ids(rows) if table_id}
    candidates: list[_OrphanTotalCandidate] = []
    for name, records in datasets.items():
        keys = record_keys(records)
        left = _dataset_source_table(records, table_index, last=True)
        if (
            len(keys) < 2
            or left is None
            or left.index != last_index_by_page.get(left.page)
            or table_width(left.table) != len(keys)
            or _record_starts_with_total(records[-1], keys[0])
        ):
            continue
        right = positions.get((left.page + 1, 0))
        if right is None:
            continue
        right_id = _source_table_id(right)
        headers = [normalize_audit_text(value) for value in (getattr(right.table, "headers", None) or [])]
        source_rows = list(getattr(right.table, "rows", None) or [])
        if (
            right_id in represented
            or any(headers)
            or not 1 <= len(source_rows) <= 3
            or table_width(right.table) != len(keys)
            or not all(_source_row_starts_with_total(row, len(keys)) for row in source_rows)
        ):
            continue
        represented.add(right_id)
        candidates.append(
            _OrphanTotalCandidate(
                dataset_name=name,
                keys=tuple(keys),
                source_table=right,
                rows=tuple(source_rows),
            )
        )
    return candidates


def _record_starts_with_total(record: dict[str, Any], first_key: str) -> bool:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    return _match_token(raw.get(first_key)) in {"合计", "总计", "小计"}


def _source_row_starts_with_total(row: Any, width: int) -> bool:
    cells = row_cells_by_column(row, width)
    return bool(cells) and _match_token(getattr(cells[0], "text", "")) in {"合计", "总计", "小计"}


def _source_row_record(
    dataset_name: str,
    keys: list[str],
    source_table: _SourceTable,
    row: Any,
    row_index: int,
) -> dict[str, Any]:
    table_id = _source_table_id(source_table)
    width = table_width(source_table.table)
    cells = row_cells_by_column(row, width)
    raw = {key: str(getattr(cell, "text", "") or "") for key, cell in zip(keys, cells, strict=True)}
    normalized = {
        key: normalize_scalar(value, value_type="decimal") if amount_like(value) else normalize_audit_text(value)
        for key, value in raw.items()
    }
    source_row_index = int(getattr(row, "source_row_index", row_index) or 0)
    refs: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for column, (field_name, cell) in enumerate(zip(keys, cells, strict=True)):
        evidence_ids.extend(str(value) for value in (getattr(cell, "evidence_ids", None) or []) if value)
        cell_refs = [dict(ref) for ref in (getattr(cell, "source_cell_refs", None) or []) if isinstance(ref, dict)]
        if not cell_refs:
            cell_refs = [{}]
        for ref in cell_refs:
            ref.update(
                {
                    "page": source_table.page,
                    "table_id": table_id,
                    "row": source_row_index,
                    "col": column,
                    "field_name": field_name,
                    "recovery": "cross_page_orphan_total",
                }
            )
            bbox = getattr(cell, "bbox", None)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                ref.setdefault("bbox", [float(value) for value in bbox])
            refs.append(ref)
    evidence_ids.extend(str(value) for value in (getattr(source_table.table, "evidence_ids", None) or []) if value)
    evidence_ids = list(dict.fromkeys(evidence_ids))
    source = {
        "page": source_table.page,
        "page_range": [source_table.page, source_table.page],
        "table_id": table_id,
        "physical_table_id": table_id,
        "table_row_index": row_index,
        "source_row_index": source_row_index,
        "source_resolution": "cross_page_orphan_total",
        "recovery": "cross_page_orphan_total",
        "source_cell_refs": refs,
        **({"evidence_ids": evidence_ids} if evidence_ids else {}),
    }
    return {
        "record_id": f"{dataset_name}:cross_page:{source_table.page}:{source_row_index}",
        "normalized": normalized,
        "canonical_raw": dict(raw),
        "raw": raw,
        "source": source,
        "source_cell_refs": copy.deepcopy(refs),
        **({"evidence_ids": list(evidence_ids)} if evidence_ids else {}),
    }


def repair_note_datasets(datasets: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Repair exact audit-note header and adjacent-cell failures without inventing values."""

    warnings: list[str] = []
    for name, rows in list(datasets.items()):
        if re.sub(r"_\d+$", "", name) in _STATEMENT_KINDS or not rows:
            continue
        rows, header_count = _promote_note_header_rows(name, rows)
        split_count = sum(_repair_adjacent_note_cells(record) for record in rows)
        datasets[name] = rows
        if header_count:
            warnings.append(f"AUDIT_NOTE_HEADER_ROWS_PROMOTED:dataset={name}:rows={header_count}")
        if split_count:
            warnings.append(f"AUDIT_NOTE_ADJACENT_CELLS_SPLIT:dataset={name}:rows={split_count}")
    return warnings


def _promote_note_header_rows(
    dataset_id: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    keys = record_keys(rows)
    generic_columns = sum(bool(re.fullmatch(r"(?:col|column)_\d+", key)) for key in keys)
    if len(keys) < 2 or generic_columns < max(1, (len(keys) - 1) // 2):
        return rows, 0
    candidates = rows[:2]
    scores = [_promoted_header_score(record) for record in candidates]
    anchor = next(
        (
            index
            for index, (score, record) in enumerate(zip(scores, candidates, strict=True))
            if score >= 3 or len(keys) == 2 and _semantic_two_column_header(record)
        ),
        None,
    )
    if anchor is None:
        return rows, 0
    start = 0 if anchor and scores[0] >= 1 else anchor
    header_rows = candidates[start : anchor + 1]
    labels = [
        next(
            (
                normalize_audit_text((record.get("raw") or {}).get(key))
                for record in reversed(header_rows)
                if normalize_audit_text((record.get("raw") or {}).get(key))
            ),
            key,
        )
        for key in keys
    ]
    for index in range(len(labels) - 1):
        if labels[index + 1] != keys[index + 1]:
            continue
        split = _split_concatenated_header(labels[index])
        if split is not None:
            labels[index], labels[index + 1] = split
    labels = _unique_field_names(labels)
    retained = [_remap_record(record, keys, labels) for record in rows[anchor + 1 :]]
    for ordinal, record in enumerate(retained, start=1):
        record["record_id"] = f"{dataset_id}:r{ordinal:06d}"
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        source["header_recovery"] = "promoted_source_rows"
    return retained, anchor + 1


def _promoted_header_score(record: dict[str, Any]) -> int:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    values = [normalize_audit_text(value) for value in raw.values() if normalize_audit_text(value)]
    if not values or any(amount_like(value) for value in values):
        return 0
    return sum(bool(_PROMOTED_HEADER_MARKERS.search(value)) for value in values)


def _semantic_two_column_header(record: dict[str, Any]) -> bool:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    values = [normalize_audit_text(value) for value in raw.values()]
    if len(values) != 2 or not all(values) or any(amount_like(value) for value in values):
        return False
    return bool(
        re.search(r"项目|名称|类别|组合", values[0])
        and re.search(r"依据|说明|金额|余额|比例|日期|状态|类别|名称", values[1])
    )


def _split_concatenated_header(value: str) -> tuple[str, str] | None:
    exact = {
        "担保起始日担保到期日": ("担保起始日", "担保到期日"),
        "开始日期结束日期": ("开始日期", "结束日期"),
    }
    compact = normalize_audit_text(value).replace(" ", "")
    if compact in exact:
        return exact[compact]
    match = re.fullmatch(r"(?P<left>.+?(?:起始|开始)日?期?)(?P<right>.+?(?:到期|结束)日?期?)", compact)
    return (match.group("left"), match.group("right")) if match else None


def _unique_field_names(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for index, label in enumerate(labels, start=1):
        base = normalize_audit_text(label) or f"column_{index:02d}"
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def _repair_adjacent_note_cells(record: dict[str, Any]) -> int:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    keys = list(raw)
    for index, key in enumerate(keys[:-1]):
        target = keys[index + 1]
        if raw.get(target) not in (None, ""):
            continue
        split = _split_adjacent_note_value(raw.get(key))
        if split is None:
            continue
        source_value, target_value = split
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        source.setdefault("recovery_raw", {})[key] = raw.get(key)
        source["recovery"] = "exact_adjacent_cell_split"
        for pool_name in ("raw", "canonical_raw"):
            pool = record.get(pool_name)
            if isinstance(pool, dict):
                pool[key], pool[target] = source_value, target_value
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        normalized[key] = normalize_scalar(source_value, value_type="decimal") or normalize_audit_text(source_value)
        normalized[target] = normalize_scalar(target_value, value_type="decimal") or normalize_audit_text(target_value)
        _duplicate_split_source_ref(record, source_key=key, target_key=target)
        return 1
    return 0


def _split_adjacent_note_value(value: Any) -> tuple[str, str] | None:
    compact = normalize_audit_text(value).replace(" ", "")
    amounts = list(_AMOUNT_RE.finditer(compact))
    if len(amounts) == 2 and amounts[0].start() == 0 and amounts[0].end() == amounts[1].start():
        if amounts[1].end() == len(compact):
            return amounts[0].group(), amounts[1].group()
    if len(amounts) == 1 and amounts[0].start() == 0:
        suffix = compact[amounts[0].end() :]
        if re.fullmatch(r"(?:\d+年以内|\d+[至到-]\d+年|\d+年以上)", suffix):
            return amounts[0].group(), suffix
    return None


def _duplicate_split_source_ref(record: dict[str, Any], *, source_key: str, target_key: str) -> None:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    refs = source.get("source_cell_refs") if isinstance(source.get("source_cell_refs"), list) else []
    copies = [copy.deepcopy(ref) for ref in refs if isinstance(ref, dict) and ref.get("field_name") == source_key]
    for ref in copies:
        ref["field_name"] = target_key
        ref["recovery"] = "split_from_same_source_cell"
        refs.append(ref)
    if refs:
        record["source_cell_refs"] = copy.deepcopy(refs)


def recover_note_text_continuations(
    datasets: dict[str, list[dict[str, Any]]],
    parse_result: Any,
) -> list[str]:
    """Recover exact audit-note continuation rows that survive only as positioned page text."""

    rows = datasets.get("accounts_payable")
    if not rows or any(_match_token((record.get("raw") or {}).get(record_keys(rows)[0])) == "合计" for record in rows):
        return []
    keys = record_keys(rows)
    if len(keys) != 3:
        return []
    next_page = max(dataset_pages(rows), default=0) + 1
    page = next(
        (
            item
            for item in getattr(parse_result, "pages", None) or []
            if int(getattr(item, "page_number", 0) or 1) == next_page
        ),
        None,
    )
    if page is None:
        return []
    amount = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
    pattern = re.compile(rf"合\s*计\s*(?P<current>{amount})\s+(?P<previous>{amount})")
    matches: list[tuple[re.Match[str], Any, int]] = []
    for block_index, block in enumerate(getattr(page, "texts", None) or []):
        match = pattern.search(normalize_audit_text(getattr(block, "content", "")))
        if match is not None:
            matches.append((match, block, block_index))
    if len(matches) != 1:
        return []
    match, block, block_index = matches[0]
    raw = {keys[0]: "合 计", keys[1]: match.group("current"), keys[2]: match.group("previous")}
    bbox = getattr(block, "bbox", None)
    evidence_ids = [str(value) for value in (getattr(block, "evidence_ids", None) or []) if value]
    refs = []
    for field_name in keys:
        ref: dict[str, Any] = {
            "page": next_page,
            "field_name": field_name,
            "source": "canonical_text",
            "recovery": "text_continuation_row",
        }
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            ref["bbox"] = [float(value) for value in bbox]
        if evidence_ids:
            ref["evidence_ids"] = list(evidence_ids)
        refs.append(ref)
    record = normalize_audit_record(
        {
            "record_id": "accounts_payable:recovered_total",
            "raw": raw,
            "canonical_raw": dict(raw),
            "normalized": {},
            "source": {
                "page": next_page,
                "page_range": [next_page, next_page],
                "table_id": "canonical_text",
                "source_row_index": block_index,
                "source_resolution": "canonical_text_continuation_row",
                "source_cell_refs": refs,
                **({"evidence_ids": evidence_ids} if evidence_ids else {}),
            },
            "source_cell_refs": copy.deepcopy(refs),
            **({"evidence_ids": evidence_ids} if evidence_ids else {}),
            "confidence": 0.95 if evidence_ids else 0.75,
        }
    )
    rows.append(record)
    _reindex_dataset_records(rows, "accounts_payable")
    return ["AUDIT_NOTE_TEXT_CONTINUATION_RECOVERED:dataset=accounts_payable:rows=1:page=40"]


def name_note_datasets(
    datasets: dict[str, list[dict[str, Any]]],
    sections: list[dict[str, Any]],
    parse_result: Any,
) -> tuple[dict[str, str], list[str]]:
    """Replace generic names with concise, source-ordered audit subject names."""

    table_index = _source_tables(parse_result)
    candidates = {
        name
        for name, rows in datasets.items()
        if _GENERIC_TABLE_DATASET_RE.fullmatch(name) and _dataset_source_table(rows, table_index) is not None
    }
    used = {name for name in datasets if name not in candidates}
    fixed_names = set(used)
    resolved: dict[str, tuple[str, str, tuple[int, int, int]]] = {}
    group_sizes: dict[str, int] = {}
    for position, (original_name, rows) in enumerate(datasets.items()):
        if original_name not in candidates:
            continue
        section = _select_dataset_section(rows, sections, table_index)
        section_title = normalize_audit_text((section or {}).get("title"))
        caption = _dataset_caption(rows, parse_result, table_index)
        base = _semantic_note_dataset_name(section_title, caption, rows)
        if not base:
            continue
        source_table = _dataset_source_table(rows, table_index)
        source_order = (
            source_table.page if source_table is not None else 10**9,
            source_table.index if source_table is not None else 10**9,
            position,
        )
        resolved[original_name] = (base, _note_dataset_label(base, caption, section_title, rows), source_order)
        group_sizes[base] = group_sizes.get(base, 0) + 1

    allocated: dict[str, str] = {}
    group_ordinals: dict[str, int] = {}
    for original_name, (base, _label, _source_order) in sorted(resolved.items(), key=lambda item: item[1][2]):
        ordinal = group_ordinals.get(base, 0) + 1
        numbered = group_sizes[base] > 1 or base in fixed_names
        semantic_name = f"{base}_{ordinal:02d}" if numbered else base
        while semantic_name in used:
            ordinal += 1
            semantic_name = f"{base}_{ordinal:02d}"
        group_ordinals[base] = ordinal
        allocated[original_name] = semantic_name
        used.add(semantic_name)

    renamed: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    warnings: list[str] = []
    for original_name, rows in datasets.items():
        if original_name not in candidates:
            renamed[original_name] = rows
            continue
        semantic_name = allocated.get(original_name)
        if semantic_name is None:
            renamed[original_name] = rows
            warnings.append(f"AUDIT_TABLE_TITLE_UNRESOLVED:dataset={original_name}")
            continue
        _reindex_dataset_records(rows, semantic_name)
        renamed[semantic_name] = rows
        labels[semantic_name] = resolved[original_name][1]
    datasets.clear()
    datasets.update(renamed)
    return labels, warnings


def canonicalize_audit_dataset_columns(
    datasets: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    """Replace audit-note source headers with stable English keys and retain clean labels."""

    schemas: dict[str, dict[str, dict[str, str]]] = {}
    warnings: list[str] = []
    for name, rows in datasets.items():
        if re.sub(r"_\d+$", "", name) in _STATEMENT_KINDS:
            continue
        source_keys = [
            key
            for key in record_keys(rows)
            if _audit_column_key(normalize_audit_label(key)) or _record_field_has_value(rows, key)
        ]
        labels = [normalize_audit_label(key) for key in source_keys]
        target_keys = [_audit_column_key(label) for label in labels]
        unresolved = [label for label, key in zip(labels, target_keys, strict=True) if not key]
        if unresolved:
            warnings.append(f"AUDIT_COLUMN_SCHEMA_UNRESOLVED:dataset={name}:labels={'|'.join(unresolved)}")
            continue
        duplicates = [key for key, count in Counter(target_keys).items() if count > 1]
        if duplicates:
            warnings.append(f"AUDIT_COLUMN_SCHEMA_COLLISION:dataset={name}:keys={'|'.join(duplicates)}")
            continue
        datasets[name] = [_remap_record(record, source_keys, target_keys) for record in rows]
        schemas[name] = {
            key: {"label": label, "type": _audit_column_type(datasets[name], key)}
            for key, label in zip(target_keys, labels, strict=True)
        }
    return schemas, warnings


def _audit_column_type(rows: list[dict[str, Any]], key: str) -> str:
    values = [
        pool[key]
        for record in rows
        for pool_name in ("canonical_raw", "raw", "normalized")
        if isinstance((pool := record.get(pool_name)), dict) and pool.get(key) not in (None, "")
    ]
    return "decimal" if values and all(_normalized_decimal(value) is not None for value in values) else "string"


def _record_field_has_value(rows: list[dict[str, Any]], key: str) -> bool:
    return any(
        pool.get(key) not in (None, "")
        for record in rows
        for pool_name in ("raw", "canonical_raw", "normalized")
        if isinstance((pool := record.get(pool_name)), dict)
    )


def _semantic_note_dataset_name(
    section_title: str,
    caption: str,
    rows: list[dict[str, Any]],
) -> str:
    keys = record_keys(rows)
    return _column_subject_slug(keys) or _subject_slug(caption) or _subject_slug(section_title)


def _note_dataset_label(
    base: str,
    caption: str,
    section_title: str,
    rows: list[dict[str, Any]],
) -> str:
    for candidate in (caption, section_title):
        candidate_base = _subject_slug(candidate)
        if candidate and candidate_base in {"", base}:
            return candidate
    keys = " ".join(normalize_audit_text(key) for key in record_keys(rows))
    if base == "deferred_tax":
        if "应纳税暂时性差异" in keys:
            return "递延所得税负债"
        if "可抵扣暂时性差异" in keys:
            return "递延所得税资产"
    return next((marker for marker, slug in _NOTE_SUBJECT_SLUGS if slug == base), section_title)


def _subject_slug(text: str) -> str:
    return next((slug for marker, slug in _NOTE_SUBJECT_SLUGS if marker in text), "")


def _column_subject_slug(keys: list[str]) -> str:
    text = " ".join(normalize_audit_text(key) for key in keys)
    rules = (
        (r"可抵扣暂时性差异|应纳税暂时性差异", "deferred_tax"),
        (r"被投资单位", "long_term_equity_investments"),
        (r"股东名称.*期初余额.*本期增加", "paid_in_capital"),
        (r"实际控制人", "ultimate_controller"),
        (r"子公司名称", "subsidiaries"),
        (r"其他关联方名称.*其他关联方与本企业关系", "other_related_parties"),
        (r"占其他应收.*合计数", "other_receivables"),
    )
    return next((slug for pattern, slug in rules if re.search(pattern, text)), "")


def _reindex_dataset_records(rows: list[dict[str, Any]], dataset_name: str) -> None:
    for ordinal, record in enumerate(rows, start=1):
        record["record_id"] = f"{dataset_name}:r{ordinal:06d}"


def _dataset_caption(
    rows: list[dict[str, Any]],
    parse_result: Any,
    table_index: dict[str, _SourceTable],
) -> str:
    source_table = _dataset_source_table(rows, table_index)
    if source_table is None:
        return ""
    page = next(
        (
            item
            for item in getattr(parse_result, "pages", None) or []
            if int(getattr(item, "page_number", 0) or 1) == source_table.page
        ),
        None,
    )
    table_bbox = getattr(source_table.table, "bbox", None)
    if page is None or not isinstance(table_bbox, (list, tuple)) or len(table_bbox) < 4:
        return ""
    table_top = float(table_bbox[1])
    candidates: list[tuple[float, str]] = []
    for block in getattr(page, "texts", None) or []:
        bbox = getattr(block, "bbox", None)
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or float(bbox[3]) > table_top + 1.0:
            continue
        for line in str(getattr(block, "content", "") or "").splitlines():
            title = _source_caption_text(line).rstrip(":：")
            if _valid_table_caption(title):
                candidates.append((float(bbox[3]), title))
    return max(candidates, default=(0.0, ""))[1]


def _source_caption_text(value: Any) -> str:
    source = str(value or "")
    normalized = normalize_audit_text(source)
    if "（" in source or "）" in source:
        normalized = normalized.replace("(", "（").replace(")", "）")
    return normalized


def _valid_table_caption(title: str) -> bool:
    compact = title.replace(" ", "")
    if not compact or compact in {"财务报表附注", "（续）", "(续)", "续"}:
        return False
    if re.fullmatch(r"(?:第)?\d{1,3}(?:页)?", compact):
        return False
    if _AUDIT_NOTE_HEADING_RE.match(title) or re.match(r"^[（(]\d{1,2}[）)]", compact):
        return True
    if re.search(r"[，,；;。！？!?]", title):
        return False
    if any(marker in title for marker, _slug in _NOTE_SUBJECT_SLUGS):
        return True
    return bool(len(title) <= 48 and _AUDIT_TABLE_CAPTION_RE.search(title))


def audit_semantic(
    base: dict[str, Any],
    datasets: dict[str, list[dict[str, Any]]],
    financial_segments: list[ProjectedSegment],
    section_ids: dict[str, str],
    dataset_labels: dict[str, str],
) -> dict[str, Any]:
    """Build dataset ordering and source-overlay presentation metadata."""

    semantic = copy.deepcopy(base)
    column_order = dict(semantic.get("dataset_column_order") or {})
    reading_columns = dict(semantic.get("dataset_reading_columns") or {})
    for name, rows in datasets.items():
        keys = record_keys(rows)
        column_order[name] = keys
        reading_columns[name] = keys
    row_groups: dict[str, Any] = {}
    for segment in financial_segments:
        if segment.dataset_id not in datasets:
            continue
        keys = [column.key for column in segment.columns]
        column_order[segment.dataset_id] = keys
        reading_columns[segment.dataset_id] = keys
        if segment.row_groups:
            row_groups[segment.dataset_id] = list(segment.row_groups)
    semantic.update(
        dataset_column_order={name: column_order[name] for name in datasets},
        dataset_reading_columns={name: reading_columns[name] for name in datasets},
        dataset_document_order=list(datasets),
        dataset_section_ids=section_ids,
        dataset_labels=dataset_labels,
        dataset_row_groups=row_groups,
        enhanced_markdown={
            **dict(semantic.get("enhanced_markdown") or {}),
            "strategy": "source_overlay",
        },
    )
    return semantic


def bind_datasets_to_sections(
    datasets: dict[str, list[dict[str, Any]]],
    sections: list[dict[str, Any]],
    parse_result: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """Bind datasets by exact statement identity or note subject before geometric fallback."""

    bindings: dict[str, str] = {}
    labels: dict[str, str] = {}
    table_index = _source_tables(parse_result)
    sections_by_id = {str(section.get("id") or ""): section for section in sections}
    for name, rows in datasets.items():
        financial_base = re.sub(r"_\d+$", "", name)
        selected = sections_by_id.get(f"section_{financial_base}") if financial_base in _STATEMENT_KINDS else None
        selected = (
            selected
            or _select_subject_section(name, rows, sections)
            or _select_dataset_section(
                rows,
                sections,
                table_index,
            )
        )
        selected = selected or sections[0]
        bindings[name] = str(selected["id"])
        labels[name] = (
            str(selected["title"])
            if not name.startswith(
                ("balance_sheet", "income_statement", "cash_flow_statement", "owners_equity_changes")
            )
            else _financial_label(name)
        )
    return bindings, labels


def _select_subject_section(
    name: str,
    rows: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    base = re.sub(r"_\d+$", "", name)
    if base not in {slug for _marker, slug in _NOTE_SUBJECT_SLUGS}:
        return None
    page = min(dataset_pages(rows), default=1)
    matches = [
        section
        for section in sections
        if int(section.get("page_start") or 1) <= page
        and _subject_slug(normalize_audit_text(section.get("title"))) == base
    ]
    return max(
        matches,
        key=lambda section: (
            int(section.get("page_start") or 1),
            float((section.get("bbox") or [0.0, 0.0])[1]),
        ),
        default=None,
    )


def _select_dataset_section(
    rows: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
) -> dict[str, Any] | None:
    page = min(dataset_pages(rows), default=1)
    candidates = [section for section in sections if int(section.get("page_start") or 1) <= page]
    if not candidates:
        return None
    table = _dataset_source_table(rows, table_index)
    if table is None:
        return max(candidates, key=lambda item: int(item.get("page_start") or 1))
    table_bbox = getattr(table.table, "bbox", None)
    if not isinstance(table_bbox, (list, tuple)) or len(table_bbox) < 4:
        same_page = [section for section in candidates if int(section.get("page_start") or 1) == page]
    else:
        top = float(table_bbox[1])
        same_page = [
            section
            for section in candidates
            if int(section.get("page_start") or 1) == page
            and isinstance(section.get("bbox"), (list, tuple))
            and len(section["bbox"]) >= 2
            and float(section["bbox"][1]) <= top
        ]
    if same_page:
        return max(same_page, key=lambda item: float((item.get("bbox") or [0.0, 0.0])[1]))
    previous = [section for section in candidates if int(section.get("page_start") or 1) < page]
    return max(previous, key=lambda item: int(item.get("page_start") or 1), default=candidates[0])


def quality_warnings(
    parse_result: Any,
    fields: dict[str, Any],
    sections: list[dict[str, Any]],
    datasets: dict[str, list[dict[str, Any]]],
    financial_segments: list[ProjectedSegment],
) -> list[str]:
    """Return audit-specific blockers without changing sealed quality facts."""

    warnings: list[str] = []
    section_types = {str(section.get("type") or "") for section in sections}
    for required in ("audit_opinion", "basis_for_opinion"):
        if required not in section_types:
            warnings.append(f"AUDIT_SECTION_MISSING:type={required}")
    if not fields.get("audit_document_number"):
        warnings.append("AUDIT_DOCUMENT_NUMBER_MISSING")
    table_index = _source_tables(parse_result)
    for name, rows in datasets.items():
        warnings.extend(_dataset_quality_warnings(name, rows))
        warnings.extend(_note_source_conservation_warnings(name, rows, table_index))
    for candidate in _orphan_total_candidates(datasets, table_index):
        warnings.append(
            f"AUDIT_CROSS_PAGE_ORPHAN_ROW_UNRESOLVED:dataset={candidate.dataset_name}:"
            f"table={_source_table_id(candidate.source_table)}:rows={len(candidate.rows)}"
        )
    if any(
        _contains_radical(value)
        for rows in datasets.values()
        for row in rows
        for value in (row.get("normalized") or {}).values()
    ):
        warnings.append("AUDIT_NORMALIZED_GLYPH_VARIANT_REMAINS")
    if not financial_segments and any(statement_kind(page) for page in getattr(parse_result, "pages", None) or []):
        warnings.append("AUDIT_FINANCIAL_STATEMENTS_UNRESOLVED")
    return list(dict.fromkeys(warnings))


def _note_source_conservation_warnings(
    name: str,
    rows: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
) -> list[str]:
    """Block verification when projected note fields omit physical source cells."""

    if not rows or re.sub(r"_\d+$", "", name) in _STATEMENT_KINDS:
        return []
    keys = _record_pool_keys(rows)
    key_set = set(keys)
    table_ids = _dataset_source_table_ids(rows, table_index)
    warnings: list[str] = []
    omitted_cells: set[tuple[str, int, int, str]] = set()
    omitted_fields: set[str] = set()
    for record in rows:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        for ref in source.get("source_cell_refs") or []:
            if not isinstance(ref, dict) or not ref.get("field_name"):
                continue
            field_name = str(ref["field_name"])
            table_id = str(ref.get("table_id") or source.get("physical_table_id") or "")
            row_index = int(ref.get("row", -1))
            col_index = int(ref.get("col", -1))
            cell = _physical_source_cell(table_index.get(table_id), row_index=row_index, col_index=col_index)
            source_value = _match_token(getattr(cell, "text", "")) if cell is not None else ""
            if not source_value:
                continue
            if field_name not in key_set or field_name not in raw or not _match_token(raw.get(field_name)):
                omitted_fields.add(field_name)
                omitted_cells.add((table_id, row_index, col_index, field_name))
    if omitted_fields:
        warnings.append(
            f"AUDIT_NOTE_SOURCE_COLUMN_OMITTED:dataset={name}:tables={'|'.join(table_ids)}:"
            f"expected={len(keys) + len(omitted_fields)}:actual={len(keys)}"
        )
    if omitted_cells:
        warnings.append(
            f"AUDIT_NOTE_SOURCE_CELL_OMITTED:dataset={name}:tables={'|'.join(table_ids)}:count={len(omitted_cells)}"
        )
    return warnings


def _dataset_source_table_ids(
    rows: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
) -> list[str]:
    table_ids: list[str] = []
    for record in rows:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        candidates = [
            *(str(value) for value in (source.get("physical_table_ids") or []) if value),
            str(source.get("physical_table_id") or ""),
            str(source.get("table_id") or "").split(":segment_", 1)[0],
        ]
        candidates.extend(
            str(ref.get("table_id") or "") for ref in source.get("source_cell_refs") or [] if isinstance(ref, dict)
        )
        for table_id in candidates:
            if table_id in table_index and table_id not in table_ids:
                table_ids.append(table_id)
    return table_ids


def _physical_source_cell(
    source_table: _SourceTable | None,
    *,
    row_index: int,
    col_index: int,
) -> Any | None:
    if source_table is None or row_index < 0 or col_index < 0:
        return None
    rows = list(getattr(source_table.table, "rows", None) or [])
    row = next(
        (item for item in rows if int(getattr(item, "source_row_index", -1)) == row_index),
        rows[row_index] if row_index < len(rows) else None,
    )
    if row is None:
        return None
    cells = row_cells_by_column(row, table_width(source_table.table))
    return cells[col_index] if col_index < len(cells) else None


def _dataset_quality_warnings(name: str, rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    keys = record_keys(rows)
    if sum(bool(re.fullmatch(r"(?:col|column)_\d+", key)) for key in keys) > max(1, len(keys) // 2):
        warnings.append(f"AUDIT_GENERIC_COLUMNS_EXCESSIVE:dataset={name}")
    if any(_AMOUNT_HEADER_RE.fullmatch(normalize_audit_text(key).replace(" ", "")) for key in keys):
        warnings.append(f"AUDIT_NUMERIC_COLUMN_NAME:dataset={name}")
    loss = sum(
        value not in (None, "")
        and not _empty_decimal_placeholder(value)
        and (record.get("normalized") or {}).get(key) in (None, "")
        for record in rows
        for key, value in (record.get("raw") or {}).items()
    )
    if loss:
        warnings.append(f"AUDIT_NORMALIZATION_LOSS:dataset={name}:count={loss}")
    missing_evidence = sum(
        not any((record.get("source") or {}).get(key) for key in ("evidence_ids", "source_cell_refs"))
        for record in rows
    )
    if missing_evidence:
        warnings.append(f"AUDIT_EVIDENCE_MISSING:dataset={name}:count={missing_evidence}")
    canonical_missing = 0
    field_source_missing = 0
    numeric_mismatches = 0
    for record in rows:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        canonical_raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        ref_fields = {
            str(ref.get("field_name"))
            for ref in (source.get("source_cell_refs") or [])
            if isinstance(ref, dict) and ref.get("field_name")
        }
        field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
        for key, value in normalized.items():
            raw_value = raw.get(key)
            canonical_value = canonical_raw.get(key)
            if raw_value not in (None, "") and canonical_value in (None, ""):
                canonical_missing += 1
            if value not in (None, "") and key not in ref_fields and key not in field_sources:
                field_source_missing += 1
            source_value = canonical_value if canonical_value not in (None, "") else raw_value
            if key in ref_fields or (source.get("evidence_ids") and source_value not in (None, "")):
                source_decimal = _normalized_decimal(source_value)
                normalized_decimal = _normalized_decimal(value)
                if source_decimal is not None and normalized_decimal is not None and source_decimal != normalized_decimal:
                    numeric_mismatches += 1
    if canonical_missing:
        warnings.append(f"AUDIT_CANONICAL_RAW_MISSING:dataset={name}:count={canonical_missing}")
    if field_source_missing:
        warnings.append(f"AUDIT_FIELD_SOURCE_MISSING:dataset={name}:count={field_source_missing}")
    if numeric_mismatches:
        warnings.append(f"AUDIT_NORMALIZED_NUMERIC_MISMATCH:dataset={name}:count={numeric_mismatches}")
    if re.sub(r"_\d+$", "", name) in _STATEMENT_KINDS:
        warnings.extend(_statement_dataset_quality_warnings(name, rows, keys))
    return warnings


def _statement_dataset_quality_warnings(
    name: str,
    rows: list[dict[str, Any]],
    keys: list[str],
) -> list[str]:
    amount_keys = [
        key
        for key in keys
        if any(marker in key for marker in ("amount", "balance", "capital", "reserve", "profit", "total"))
    ]
    amount_bleed = 0
    missing_items = 0
    for record in rows:
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
        item = normalize_audit_text(raw.get("item"))
        numeric_amounts = sum(_decimal(normalize_audit_text(normalized.get(key))) is not None for key in amount_keys)
        amount_bleed += bool(_AMOUNT_RE.search(item))
        missing_items += bool(not item and numeric_amounts)
    warnings: list[str] = []
    if amount_bleed:
        warnings.append(f"AUDIT_STATEMENT_AMOUNT_COLUMNS_EMPTY:dataset={name}:count={amount_bleed}")
    if missing_items:
        warnings.append(f"AUDIT_STATEMENT_ITEM_MISSING:dataset={name}:count={missing_items}")
    return warnings


def normalize_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize projected values while keeping raw and canonical_raw untouched."""

    normalized = copy.deepcopy(record)
    raw = normalized.get("raw") if isinstance(normalized.get("raw"), dict) else {}
    canonical_raw = normalized.get("canonical_raw") if isinstance(normalized.get("canonical_raw"), dict) else {}
    values = normalized.get("normalized") if isinstance(normalized.get("normalized"), dict) else None
    if not values:
        values = canonical_raw or raw
    source = normalized.get("source") if isinstance(normalized.get("source"), dict) else {}
    source_fields = {
        str(ref.get("field_name"))
        for ref in (source.get("source_cell_refs") or [])
        if isinstance(ref, dict) and ref.get("field_name")
    }
    row_evidence = bool(source.get("evidence_ids"))
    normalized_values: dict[str, Any] = {}
    keys = list(dict.fromkeys([*raw, *canonical_raw, *(values or {})]))
    for key in keys:
        value = (values or {}).get(key)
        source_value = canonical_raw.get(key) if canonical_raw.get(key) not in (None, "") else raw.get(key)
        if value in (None, "") and source_value not in (None, ""):
            value = source_value
        if source_value not in (None, "") and (key in source_fields or row_evidence):
            source_decimal = _normalized_decimal(source_value)
            current_decimal = _normalized_decimal(value)
            if source_decimal is not None and (_decimal_field_key(key) or current_decimal is not None):
                value = source_decimal
        normalized_value = normalize_audit_value(value) if isinstance(value, str) else value
        if _decimal_field_key(key) and isinstance(normalized_value, str):
            decimal_value = normalize_scalar(normalized_value, value_type="decimal")
            if decimal_value is not None or _empty_decimal_placeholder(normalized_value):
                normalized_value = decimal_value
        normalized_values[key] = normalized_value
    normalized["normalized"] = normalized_values
    if source:
        page = int(source.get("page") or 0)
        if page and not source.get("page_range"):
            source["page_range"] = [page, page]
        evidence_ids = [str(value) for value in (source.get("evidence_ids") or []) if value]
        evidence_ids.extend(str(value) for value in (normalized.get("evidence_ids") or []) if value)
        field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else {}
        for detail in field_sources.values():
            if isinstance(detail, dict):
                evidence_ids.extend(str(value) for value in (detail.get("evidence_ids") or []) if value)
        if evidence_ids:
            source["evidence_ids"] = list(dict.fromkeys(evidence_ids))
        for key in ("source_cell_refs", "evidence_ids"):
            if source.get(key):
                normalized[key] = copy.deepcopy(source[key])
    return normalized


def _normalized_decimal(value: Any) -> str | None:
    """Return an exact decimal spelling only when the source value is numeric."""

    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return _normalized_decimal(value.get("value"))
    cleaned = normalize_audit_value(value) if isinstance(value, str) else str(value)
    return normalize_scalar(cleaned, value_type="decimal")


def synchronize_audit_record_sources(datasets: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Backfill canonical raw values only from source-backed raw fields."""

    recovered = 0
    for records in datasets.values():
        for record in records:
            raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
            canonical_raw = record.get("canonical_raw") if isinstance(record.get("canonical_raw"), dict) else {}
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            backed_fields = {
                str(ref.get("field_name"))
                for ref in (source.get("source_cell_refs") or [])
                if isinstance(ref, dict) and ref.get("field_name")
            }
            row_evidence = bool(source.get("evidence_ids"))
            for key, value in raw.items():
                if (
                    value not in (None, "")
                    and canonical_raw.get(key) in (None, "")
                    and (key in backed_fields or row_evidence)
                ):
                    canonical_raw[key] = copy.deepcopy(value)
                    recovered += 1
            record["canonical_raw"] = canonical_raw
    return [f"AUDIT_CANONICAL_RAW_RECOVERED:fields={recovered}"] if recovered else []


def statement_kind(page: Any) -> str | None:
    """Return the exact main-statement kind named on a page."""

    for line in page_lines(page):
        compact = normalize_audit_text(line).replace(" ", "")
        for pattern, kind in _STATEMENT_TITLE_PATTERNS:
            if pattern.fullmatch(compact):
                return kind
    return None


def _unresolved_landscape_pages(parse_result: Any, segments: list[ProjectedSegment]) -> list[str]:
    resolved = {page for segment in segments for page in dataset_pages(segment.records)}
    candidate_pages = embedded_financial_pages(parse_result)
    warnings: list[str] = []
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        if (
            page_number not in candidate_pages
            or page_number in resolved
            or not _landscape_table_candidate(page, parse_result=parse_result)
        ):
            continue
        tables = [
            *(getattr(page, "tables", None) or []),
            *_logical_tables_for_page(parse_result, page_number),
        ]
        widest = max((table_width(table) for table in tables), default=0)
        warnings.append(f"AUDIT_OWNER_EQUITY_UNRESOLVED:page={page_number}:width={widest}")
    return warnings


def _source_tables(parse_result: Any) -> dict[str, _SourceTable]:
    index: dict[str, _SourceTable] = {}
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 1)
        tables = list(getattr(page, "tables", None) or [])
        for table_index, table in enumerate(tables):
            table_id = str(getattr(table, "table_id", "") or f"pt_{page_number}_{table_index}")
            index[table_id] = _SourceTable(table=table, page=page_number, index=table_index)
    return index


def _adjacent_edge_tables(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
) -> bool:
    left_table = _dataset_source_table(left, table_index, last=True)
    right_table = _dataset_source_table(right, table_index)
    if left_table is None or right_table is None or right_table.page != left_table.page + 1:
        return False
    left_page_tables = [item for item in table_index.values() if item.page == left_table.page]
    return left_table.index == max((item.index for item in left_page_tables), default=-1) and right_table.index == 0


def _numeric_header_continuation(left_keys: list[str], right_keys: list[str]) -> bool:
    if len(left_keys) < 2 or len(left_keys) != len(right_keys):
        return False
    semantic_left = sum(bool(_TABLE_HEADER_MARKERS.search(normalize_audit_text(key))) for key in left_keys)
    numeric_right = sum(
        bool(_AMOUNT_HEADER_RE.fullmatch(normalize_audit_text(key).replace(" ", ""))) for key in right_keys
    )
    return semantic_left >= 2 and numeric_right >= 2 and not _TABLE_HEADER_MARKERS.search(right_keys[0])


def _repeated_header_continuation(left_keys: list[str], right_keys: list[str]) -> bool:
    return len(left_keys) >= 2 and [normalize_audit_text(key) for key in left_keys] == [
        normalize_audit_text(key) for key in right_keys
    ]


def _category_continuation_values(
    parse_result: Any,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_keys: list[str],
    right_keys: list[str],
) -> list[_RecoveredCategory]:
    if len(left_keys) != len(right_keys) + 2 or len(right_keys) < 3:
        return []
    if [normalize_audit_text(key) for key in right_keys[1:]] != [normalize_audit_text(key) for key in left_keys[2:-1]]:
        return []
    categories = [
        normalize_audit_text((record.get("raw") or {}).get(left_keys[0], ""))
        for record in left
        if normalize_audit_text((record.get("raw") or {}).get(left_keys[0], ""))
    ]
    if not categories:
        return []
    page = min(dataset_pages(right), default=0)
    source_page = next(
        (
            item
            for item in getattr(parse_result, "pages", None) or []
            if int(getattr(item, "page_number", 0) or 1) == page
        ),
        None,
    )
    if source_page is None:
        return []
    allowed = set(categories)
    recovered: list[_RecoveredCategory] = []
    for block in getattr(source_page, "texts", None) or []:
        for line in str(getattr(block, "content", "") or "").splitlines():
            value = normalize_audit_text(line)
            if value not in allowed:
                continue
            recovered.append(
                _RecoveredCategory(
                    value=value,
                    page=page,
                    bbox=tuple(getattr(block, "bbox", None) or ()),
                    evidence_ids=tuple(str(item) for item in (getattr(block, "evidence_ids", None) or []) if item),
                )
            )
    return recovered[: len(right)] if len(recovered) >= len(right) else []


def _header_record(
    dataset_name: str,
    target_keys: list[str],
    source_keys: list[str],
    rows: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
) -> dict[str, Any]:
    source_table = _dataset_source_table(rows, table_index)
    page = source_table.page if source_table is not None else min(dataset_pages(rows), default=1)
    table_id = (
        str(getattr(source_table.table, "table_id", "") or "")
        if source_table is not None
        else str(((rows[0].get("source") or {}).get("physical_table_id") if rows else "") or "")
    )
    raw = dict(zip(target_keys, source_keys, strict=True))
    normalized = {
        key: normalize_scalar(value, value_type="decimal") if amount_like(value) else normalize_audit_text(value)
        for key, value in raw.items()
    }
    evidence_ids = list(getattr(source_table.table, "evidence_ids", None) or []) if source_table is not None else []
    refs = [
        {
            "page": page,
            "table_id": table_id,
            "row": -1,
            "col": column,
            "field_name": field_name,
            "recovery": "source_table_header_was_data",
        }
        for column, field_name in enumerate(target_keys)
    ]
    return {
        "record_id": f"{dataset_name}:recovered_header_row",
        "normalized": normalized,
        "canonical_raw": dict(raw),
        "raw": raw,
        "source": {
            "page": page,
            "page_range": [page, page],
            "table_id": table_id,
            "physical_table_id": table_id,
            "table_row_index": -1,
            "source_row_index": -1,
            "recovery": "source_table_header_was_data",
            "source_cell_refs": refs,
            **({"evidence_ids": evidence_ids} if evidence_ids else {}),
        },
        "source_cell_refs": copy.deepcopy(refs),
        **({"evidence_ids": list(evidence_ids)} if evidence_ids else {}),
        "review": {"required": not bool(evidence_ids), "reasons": ["recovered_source_table_header"]},
    }


def _remap_record(
    record: dict[str, Any],
    source_keys: list[str],
    target_keys: list[str],
    *,
    prefix: dict[str, Any] | None = None,
    suffix: dict[str, Any] | None = None,
    injected: dict[str, _RecoveredCategory] | None = None,
) -> dict[str, Any]:
    remapped = copy.deepcopy(record)
    for pool_name in ("normalized", "canonical_raw", "raw"):
        if not isinstance(remapped.get(pool_name), dict):
            continue
        pool = remapped[pool_name]
        mapped = dict(prefix or {})
        mapped.update({target: pool.get(source, "") for source, target in zip(source_keys, target_keys, strict=True)})
        mapped.update(suffix or {})
        remapped[pool_name] = mapped
    source = remapped.get("source") if isinstance(remapped.get("source"), dict) else {}
    field_map = dict(zip(source_keys, target_keys, strict=True))
    for ref in source.get("source_cell_refs") or []:
        if isinstance(ref, dict) and ref.get("field_name") in field_map:
            ref["field_name"] = field_map[ref["field_name"]]
    if isinstance(remapped.get("source_cell_refs"), list):
        remapped["source_cell_refs"] = copy.deepcopy(source.get("source_cell_refs") or [])
    field_sources = source.get("field_sources") if isinstance(source.get("field_sources"), dict) else None
    if field_sources is not None:
        source["field_sources"] = {field_map.get(key, key): value for key, value in field_sources.items()}
    for field_name, recovered in (injected or {}).items():
        ref: dict[str, Any] = {"page": recovered.page, "field_name": field_name, "source": "canonical_text"}
        if recovered.bbox:
            ref["bbox"] = list(recovered.bbox)
        if recovered.evidence_ids:
            ref["evidence_ids"] = list(recovered.evidence_ids)
            evidence_ids = list(source.get("evidence_ids") or [])
            source["evidence_ids"] = list(dict.fromkeys([*evidence_ids, *recovered.evidence_ids]))
        source.setdefault("source_cell_refs", []).insert(0, ref)
    return remapped


def _dataset_source_table(
    rows: list[dict[str, Any]],
    table_index: dict[str, _SourceTable],
    *,
    last: bool = False,
) -> _SourceTable | None:
    records = reversed(rows) if last else iter(rows)
    for record in records:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        for key in ("physical_table_id", "table_id"):
            table_id = str(source.get(key) or "").split(":segment_", 1)[0]
            if table_id in table_index:
                return table_index[table_id]
    return None


def record_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Return the complete projected record schema in source order."""

    return _record_pool_keys(rows)


def _record_pool_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for record in rows:
        for pool_name in ("raw", "canonical_raw", "normalized"):
            pool = record.get(pool_name)
            if not isinstance(pool, dict):
                continue
            for key in pool:
                value = str(key)
                if value not in seen:
                    seen.add(value)
                    keys.append(value)
    return keys


def dataset_pages(rows: list[dict[str, Any]]) -> set[int]:
    """Return exact source pages referenced by projected rows."""

    pages: set[int] = set()
    for record in rows:
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        page = int(source.get("page") or 0)
        if page:
            pages.add(page)
        pages.update(int(value) for value in source.get("pages") or [] if int(value) > 0)
    return pages


def audit_data_dictionary(
    segments: list[ProjectedSegment],
    note_schemas: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Add audit metadata to the reused financial data dictionary."""

    dictionary = data_dictionary(segments)
    fields = dictionary.setdefault("fields", {})
    fields.update(
        {
            "subject_name": {"label": "被审计单位", "type": "string"},
            "auditor_name": {"label": "会计师事务所", "type": "string"},
            "document_date": {"label": "审计报告日期", "type": "string"},
            "currency_unit": {"label": "金额单位", "type": "string"},
            "currency": {"label": "币种", "type": "string"},
            "audit_document_number": {"label": "审计报告文号", "type": "string"},
            "regulatory_report_id": {"label": "监管报告编号", "type": "string"},
            "audit_opinion_type": {"label": "审计意见类型", "type": "string"},
        }
    )
    datasets = dictionary.setdefault("datasets", {})
    for dataset_name, columns in (note_schemas or {}).items():
        dataset = datasets.setdefault(dataset_name, {})
        dataset["columns"] = copy.deepcopy(columns)
    for dataset in datasets.values():
        columns = dataset.get("columns") if isinstance(dataset, dict) else None
        if not isinstance(columns, dict):
            continue
        for column in columns.values():
            if not isinstance(column, dict):
                continue
            if column.get("label") not in (None, ""):
                column["label"] = normalize_audit_label(column["label"])
    return dictionary


def _financial_label(name: str) -> str:
    base = re.sub(r"_\d+$", "", name)
    labels = {
        "balance_sheet": "资产负债表",
        "income_statement": "利润表",
        "cash_flow_statement": "现金流量表",
        "owners_equity_changes": "所有者权益变动表",
    }
    return labels.get(base, name)


def page_lines(page: Any) -> list[str]:
    """Return non-empty source text lines in block order."""

    return [
        line
        for block in getattr(page, "texts", None) or []
        for line in str(getattr(block, "content", "") or "").splitlines()
        if line.strip()
    ]


def _contains_radical(value: Any) -> bool:
    return isinstance(value, str) and any(0x2E80 <= ord(char) <= 0x2FFF for char in value)


def _decimal_field_key(key: Any) -> bool:
    value = normalize_audit_text(key).lower()
    return bool(
        re.search(
            r"amount|balance|capital|reserve|profit|total|ratio|rate|share_count|current_period|previous_period|"
            r"increase|decrease|amortization|provision|reversal|write_off|revenue|income|expense|cost|years|"
            r"金额|发生额|余额|资本|公积|利润|合计|账面价值|比例|税率|折旧率|股份数|使用寿命",
            value,
        )
    )


def _empty_decimal_placeholder(value: Any) -> bool:
    return bool(re.fullmatch(r"[-‐‑‒–—―−－]{1,3}", normalize_audit_text(value).replace(" ", "")))


def dataset_blocking_warnings(warnings: Iterable[str]) -> list[str]:
    """Return warnings that must block statement-dataset verification."""

    return [str(warning) for warning in warnings if str(warning).startswith(_DATASET_BLOCKING_WARNING_PREFIXES)]


def blocking_warning(warnings: Iterable[str]) -> bool:
    """Return whether audit projection confidence must remain degraded."""

    return any(str(warning).startswith(_DOCUMENT_BLOCKING_WARNING_PREFIXES) for warning in warnings)


__all__ = [
    "audit_data_dictionary",
    "audit_semantic",
    "bind_datasets_to_sections",
    "blocking_warning",
    "canonicalize_audit_dataset_columns",
    "dataset_pages",
    "dataset_blocking_warnings",
    "embedded_financial_pages",
    "merge_cross_page_continuations",
    "name_note_datasets",
    "normalize_audit_display_text",
    "normalize_audit_display_value",
    "normalize_audit_record",
    "normalize_audit_label",
    "normalize_audit_text",
    "normalize_audit_value",
    "page_lines",
    "project_embedded_financial_statements",
    "quality_warnings",
    "repair_note_datasets",
    "record_keys",
    "statement_kind",
    "synchronize_audit_record_sources",
]
