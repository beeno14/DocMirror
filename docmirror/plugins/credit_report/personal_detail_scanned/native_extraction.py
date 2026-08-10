# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical table extraction for native personal detailed credit reports.

The personal-detailed layout is primarily a sequence of labelled tables.  A
native PDF must not be routed through the prose-oriented personal-brief
extractor: doing so drops the account cards entirely.  This module projects
the stable table grammar into the same canonical dataset collections used by
the scanned/OCR path and also emits a lossless source-row ledger.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Iterable, Mapping

from docmirror.plugins.credit_report.currency_codes import (
    CURRENCY_CODE_BY_ALIAS,
    ISO_4217_CURRENT_CODES,
)
from docmirror.plugins.credit_report.personal_detail_scanned.collapsed_clusters import (
    decode_employment_basic_cluster,
    decode_labeled_cluster,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
    is_explicit_source_absence,
    normalize_pboc_field,
    pboc_controlled_vocabulary,
    validate_pboc_field,
)
from docmirror.plugins.credit_report.personal_detail_scanned.quality import (
    cn_identity_number_valid,
    header_field_valid,
)
from docmirror.plugins.credit_report.personal_detail_scanned.summary_fallback import (
    decode_credit_business_overview_text_line,
    decode_credit_business_overview_text_lines,
    is_credit_business_overview_text_header,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id

_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[.年/-]\s*(\d{1,2})"
    r"(?:\s*[.月/-]\s*(\d{1,2})\s*日?)?(?!\d)"
)
_FULL_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[.年/-]\s*(\d{1,2})"
    r"\s*[.月/-]\s*(\d{1,2})\s*日?(?!\d)"
)
_CANONICAL_FULL_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[.,年/-]\s*(\d{1,2})"
    r"\s*[.,月/-]\s*(\d{1,2})\s*日?(?!\d)"
)
_MONTH_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[.年/-]\s*(\d{1,2})(?:\s*月)?"
    r"(?!\s*[.月/-]\s*\d)(?!\d)"
)
_AS_OF_RE = re.compile(r"截至\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_STATUS_CODES = frozenset(
    {"*", "/", "N", "1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "G", "M", "Z", "#"}
)
_INQUIRY_REASONS = (
    "本人查询",
    "贷后管理",
    "贷款审批",
    "信用卡审批",
    "担保资格审查",
    "融资审批",
    "保前审查",
    "保后管理",
    "客户准入资格审查",
    "资信审查",
    "法人代表、负责人、高管等资信审查",
    "特约商户实名审查",
    "异议处理",
    "司法调查",
)
_INQUIRY_REASON_REPAIRS = {
    "货后管理": "贷后管理",
    "贷后智理": "贷后管理",
    "货款审批": "贷款审批",
    "资款审批": "贷款审批",
    "信用卡审抛": "信用卡审批",
}

_CURRENCY_TOKEN_ALIASES = {
    **CURRENCY_CODE_BY_ALIAS,
    **{code: code for code in ISO_4217_CURRENT_CODES},
    "RMB": "CNY",
}

_ACCOUNT_LABELS = frozenset(
    {
        "管理机构",
        "发卡机构",
        "账户标识",
        "开立日期",
        "到期日期",
        "借款金额",
        "账户授信额度",
        "共享授信额度",
        "账户币种",
        "币种",
        "业务种类",
        "担保方式",
        "还款期数",
        "还款频率",
        "还款方式",
        "共同借款标志",
        "账户状态",
        "状态",
        "五级分类",
        "余额",
        "透支余额",
        "剩余还款期数",
        "剩余分期期数",
        "本月应还款",
        "应还款日",
        "账单日",
        "本月实还款",
        "最近一次还款日期",
        "当前逾期期数",
        "当前逾期总额",
        "逾期31—60天未还本金",
        "逾期31－60天未还本金",
        "逾期61—90天未还本金",
        "逾期61－90天未还本金",
        "逾期91—180天未还本金",
        "逾期91－180天未还本金",
        "逾期180天以上未还本金",
        "透支180天以上未付余额",
        "已用额度",
        "最近6个月平均使用额度",
        "最大使用额度",
        "最近6个月平均透支余额",
        "最大透支余额",
        "未出单的大额专项分期余额",
        "账户关闭日期",
        "销户日期",
        "转出月份",
        "大额专项分期额度",
        "分期额度生效日期",
        "分期额度到期日期",
        "已用分期金额",
        "还款日期",
        "还款金额",
        "当前还款状态",
        "授信协议标识",
        "生效日期",
        "授信额度用途",
        "授信额度",
        "授信限额",
        "授信限额编号",
        "责任人类型",
        "还款责任金额",
        "保证合同编号",
        "主业务借款人",
        "主业务借款人证件类型",
        "主业务借款人证件号码",
        "还款状态",
        "逾期月数",
        "机构名称",
        "业务类型",
        "业务开通日期",
        "当前缴费状态",
        "当前欠费金额",
        "记账年月",
        "债权接收日期",
        "原债权人",
        "原债务业务种类",
        "债权金额",
        "债权转移时的还款状态",
        "特殊交易类型",
        "发生日期",
        "变更月数",
        "发生金额",
        "明细记录",
    }
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _business_text(value: Any) -> str:
    """Remove PDF line-wrap whitespace from compact domain labels and names."""
    return _compact(value)


_ACCOUNT_SCALAR_CONTRACT_ROLES = {
    "business_type": "account_business_type",
    "guarantee_type": "guarantee_type",
    "repayment_frequency": "repayment_frequency",
    "repayment_method": "repayment_method",
}


def _account_institution(
    value: Any,
    *,
    independently_corroborated: bool = False,
) -> str | None:
    """Normalize one exact institution slot without completing a legal name.

    Whitespace removal and the plugin's bounded glyph/debris corrections are
    permitted.  A visibly broken legal-name shape (for example ``银行股份分行``)
    is not completed from a document-wide institution catalogue: that would
    silently choose an individualized business value which the source cell did
    not establish.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        institution_name_has_separated_leading_han,
        institution_slot_is_unambiguous,
        normalize_institution_name,
    )

    raw = _clean(value)
    if not institution_slot_is_unambiguous(raw):
        return None
    if institution_name_has_separated_leading_han(raw) and not independently_corroborated:
        return None
    normalized = normalize_institution_name(raw)
    compact = _compact(normalized)
    if not compact or compact == "本人":
        return compact or None
    if not re.search(r"[\u3400-\u9fffA-Za-z]", compact):
        return None
    if any(label in compact for label in _ACCOUNT_LABELS):
        return None
    if _DATE_RE.search(compact) or re.search(r"[A-Z][A-Z0-9-]{7,}\d", compact, re.IGNORECASE):
        return None
    # Legal-form fragments cannot jump directly into a branch suffix.  In
    # particular, ``家 广发银行股份 分行`` used to pass the broad ``分行``
    # suffix test and became silently wrong business data.
    if re.search(r"(?:银行)?股份(?:支行|分行|营业部|$)", compact):
        return None
    if re.search(r"有限(?:支行|分行|营业部|$)", compact):
        return None
    return normalized


def _identifier(value: Any) -> str | None:
    compact = re.sub(r"\s+", "", str(value or "")).strip()
    return compact if compact and compact != "--" else None


def _typed_identifier(value: Any) -> str | None:
    identifier = _identifier(value)
    if not identifier or not re.fullmatch(r"[A-Z0-9-]{8,80}", identifier, re.IGNORECASE):
        return None
    if not re.search(r"[A-Z]", identifier, re.IGNORECASE) or not re.search(r"\d", identifier):
        return None
    return identifier


_LIABILITY_CANONICAL_FIELDS = (
    "institution",
    "business_type",
    "open_date",
    "due_date",
    "responsibility_type",
    "responsibility_amount",
    "currency",
    "contract_number",
    "related_party_name",
    "related_party_id_type",
    "related_party_id_number",
    "snapshot_date",
    "balance",
    "five_tier_class",
    "overdue_months",
    "repayment_status_code",
)


def _liability_exact_replay_key(record: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return an exact business fingerprint, never a fuzzy record identity."""

    contract_number = _compact(record.get("contract_number"))
    if not contract_number:
        return None
    return (
        contract_number,
        *(
            _compact(record.get(field_name))
            for field_name in _LIABILITY_CANONICAL_FIELDS
            if field_name != "contract_number"
        ),
    )


def _dedupe_liability_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suppress only byte-equivalent canonical liability observations.

    A former implementation merged identifier prefixes or any observations
    sharing three business fields.  Different guarantee contracts commonly do
    share borrower, dates, and amounts, so that policy could silently erase a
    real record.  Candidate B now reconciles only exact contract/ordinal
    identities; this compatibility helper removes exact replays only.
    """

    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        replay_key = _liability_exact_replay_key(record)
        if replay_key is not None and replay_key in seen:
            continue
        if replay_key is not None:
            seen.add(replay_key)
        kept.append(record)
    for sequence, record in enumerate(kept, start=1):
        record["sequence"] = sequence
    return kept


def _number(value: Any) -> int | str | None:
    raw = _compact(value).replace(",", "")
    if raw in {"", "--", "-"}:
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def _valid_date_spans(value: Any) -> list[tuple[tuple[int, int], str]]:
    """Return every independently valid date/month span in one scalar slot."""

    raw = str(value or "").strip()
    dated_spans: list[tuple[tuple[int, int], str]] = []
    occupied: list[tuple[int, int]] = []
    for match in _CANONICAL_FULL_DATE_RE.finditer(raw):
        year, month, day = match.groups()
        normalized_year = int(year)
        normalized_month = int(month)
        if not 1 <= normalized_month <= 12:
            continue
        normalized_day = int(day)
        try:
            date(normalized_year, normalized_month, normalized_day)
        except ValueError:
            continue
        occupied.append(match.span())
        dated_spans.append(
            (match.span(), f"{normalized_year:04d}-{normalized_month:02d}-{normalized_day:02d}")
        )
    for match in _MONTH_RE.finditer(raw):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        year, month = match.groups()
        normalized_year = int(year)
        normalized_month = int(month)
        if 1 <= normalized_month <= 12:
            dated_spans.append(
                (match.span(), f"{normalized_year:04d}-{normalized_month:02d}")
            )
    return sorted(dated_spans)


def _short_ascii_date_residue(value: Any) -> str | None:
    """Return bounded non-business OCR debris, never a second field value.

    A recurring scan watermark contributes one isolated ASCII glyph beside a
    date cell.  Digits, non-ASCII text, and longer words can carry business
    meaning and therefore cannot be discarded by this correction.
    """

    residue = re.sub(r"\s+", "", str(value or ""))
    if not residue:
        return ""
    if not residue.isascii() or len(residue) > 3 or any(character.isdigit() for character in residue):
        return None
    if not all(character.isalpha() or character in r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~" for character in residue):
        return None
    # More than two letters is a short word/label, not an isolated watermark
    # glyph.  Punctuation may accompany either one or two glyphs.
    if sum(character.isalpha() for character in residue) > 2:
        return None
    return residue


def _canonical_date_slot(value: Any) -> tuple[str | None, str, str]:
    """Decode one exact date/month slot and classify any surviving residue."""

    raw = str(value or "").strip()
    if _compact(raw) in {"", "--", "长期"}:
        return None, "", "absent"
    spans = _valid_date_spans(raw)
    if len(spans) != 1:
        return None, raw, "multiple" if spans else "invalid"
    (start, end), normalized = spans[0]
    residue = raw[:start] + raw[end:]
    compact_residue = _short_ascii_date_residue(residue)
    if compact_residue is None:
        return None, residue, "business_residue"
    return normalized, compact_residue, "ascii_residue" if compact_residue else "exact"


def _date(value: Any) -> str | None:
    normalized, _residue, resolution = _canonical_date_slot(value)
    # Callers without issue/evidence context may consume only an exact scalar.
    # Bounded watermark residue is recoverable exclusively through the
    # canonical-slot merger below, which also emits the required audit row.
    return normalized if resolution == "exact" else None


def _currency_token(value: Any) -> tuple[str | None, str, str]:
    """Decode one finite currency token and preserve any surrounding residue."""

    raw = _compact(value).upper()
    if not raw or raw in {"-", "--"}:
        return None, "", "absent"
    exact = _CURRENCY_TOKEN_ALIASES.get(raw)
    if exact is not None:
        return exact, "", "exact"

    matches: list[tuple[int, int, str]] = []
    for token, code in sorted(_CURRENCY_TOKEN_ALIASES.items(), key=lambda item: -len(item[0])):
        start = 0
        while True:
            index = raw.find(token, start)
            if index < 0:
                break
            end = index + len(token)
            if token.isascii() and (
                (index > 0 and raw[index - 1].isascii() and raw[index - 1].isalnum())
                or (end < len(raw) and raw[end].isascii() and raw[end].isalnum())
            ):
                start = index + 1
                continue
            matches.append((index, end, code))
            start = index + 1

    maximal = [
        match
        for match in matches
        if not any(
            other[0] <= match[0]
            and match[1] <= other[1]
            and (other[0], other[1]) != (match[0], match[1])
            for other in matches
        )
    ]
    if len(maximal) != 1:
        return None, raw, "multiple" if maximal else "unknown"
    start, end, code = maximal[0]
    residue = raw[:start] + raw[end:]
    return (code, residue, "residue") if residue else (code, "", "exact")


def _currency(value: Any) -> str | None:
    return _currency_token(value)[0]


def _report_currency_residue(
    parse_result: Any,
    *,
    dataset: str,
    target_record_id: str,
    raw: Any,
    currency: str,
    residue: str,
    source_refs: Iterable[Mapping[str, Any]],
    parser_stage: str,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_currency_token_residue_corrected",
            message="A unique finite currency token was retained, but non-currency OCR or structure residue remained in its field.",
            severity="info",
            status="resolved",
            parser_stage=parser_stage,
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name="currency",
            observed_value={"raw": raw, "residue": residue},
            candidate_value={"currency": currency},
            source_refs=tuple(source_refs),
            reason_codes=(
                "finite_currency_vocabulary",
                "unique_currency_token",
                "non_currency_residue_reported",
                "professional_field_correction",
            ),
        ),
    )


def _table_rows(table: Any) -> list[list[str]]:
    metadata = getattr(table, "metadata", None) or {}
    raw_rows = metadata.get("raw_rows") if isinstance(metadata, dict) else None
    if isinstance(raw_rows, list) and raw_rows:
        return [[str(cell or "") for cell in row] for row in raw_rows if isinstance(row, list)]
    headers = [str(value or "") for value in getattr(table, "headers", None) or []]
    rows = [
        [str(getattr(cell, "text", "") or "") for cell in getattr(row, "cells", None) or []]
        for row in getattr(table, "rows", None) or []
    ]
    return ([headers] if headers else []) + rows


def _source_ref(
    page: Any,
    table: Any,
    *,
    row: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "source": "native_detail_table",
        "logical_page": int(getattr(page, "page_number", 0) or 0),
        "source_page": int(getattr(page, "source_page_number", 0) or getattr(page, "page_number", 0) or 0),
        "table_id": str(getattr(table, "table_id", "") or ""),
    }
    if row is not None:
        ref["row"] = row
    if column is not None:
        ref["column"] = column
    metadata = getattr(table, "metadata", None) or {}
    bbox = None
    if row is not None and column is not None and isinstance(metadata, dict):
        # Canonical projection keeps both registered-page coordinates and the
        # originating source-page coordinates.  A business-field repair must
        # point at the originating cell, never at the enclosing table.
        cell_bboxes = metadata.get("source_cell_bboxes") or metadata.get("cell_bboxes")
        if (
            isinstance(cell_bboxes, list)
            and 0 <= row < len(cell_bboxes)
            and isinstance(cell_bboxes[row], list)
            and 0 <= column < len(cell_bboxes[row])
        ):
            candidate = cell_bboxes[row][column]
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                bbox = candidate
                ref["source"] = "native_detail_table_cell"
                ref["geometry_scope"] = "cell"
        cell_evidence_ids = metadata.get("cell_evidence_ids")
        if (
            isinstance(cell_evidence_ids, list)
            and 0 <= row < len(cell_evidence_ids)
            and isinstance(cell_evidence_ids[row], list)
            and 0 <= column < len(cell_evidence_ids[row])
            and isinstance(cell_evidence_ids[row][column], list)
        ):
            ref["evidence_ids"] = [str(item) for item in cell_evidence_ids[row][column] if item]
        ref["binding_quality"] = "canonical_header_column"
        ref["binding"] = "canonical_field_slot"
        ref["canonical_row"] = row
        ref["canonical_column"] = column
        if bbox is None:
            # Row/column identity is still exact even when a synthetic fixture
            # has no pixel geometry.  Do not substitute a table-sized bbox:
            # that would falsely authorize a broad field-level correction.
            ref["geometry_scope"] = "canonical_field_slot"
    if bbox is None and (row is None or column is None):
        bbox = getattr(table, "bbox", None)
        if bbox and len(bbox) == 4:
            ref["geometry_scope"] = "table"
    if bbox and len(bbox) == 4:
        ref["bbox"] = list(bbox)
    return ref


def _nonempty(row: list[str]) -> list[str]:
    return [_clean(cell) for cell in row if _clean(cell)]


def _exact_label_observations(
    rows: list[list[str]],
) -> tuple[dict[str, list[tuple[str, int, int]]], set[tuple[str, int, int]]]:
    """Bind canonical labels only to the value in the same physical column.

    Missing cells are not allowed to shift later values left/right, and two
    equally plausible rows are retained as separate observations so the
    field-level reconciler can report a conflict instead of overwriting it.
    """

    observations: defaultdict[str, list[tuple[str, int, int]]] = defaultdict(list)
    unresolved: set[tuple[str, int, int]] = set()
    for index in range(len(rows) - 1):
        for column, cell in enumerate(rows[index]):
            label = _compact(cell)
            if label not in _ACCOUNT_LABELS:
                continue
            value = _clean(rows[index + 1][column]) if column < len(rows[index + 1]) else ""
            if not value or _compact(value) in _ACCOUNT_LABELS:
                unresolved.add((label, index, column))
                continue
            observations[label].append((value, index + 1, column))
    return dict(observations), unresolved


def _canonical_account_terminal_subtable(
    rows: list[list[str]],
) -> dict[str, Any] | None:
    """Decode one closed status/close-date row despite bounded watermark debris.

    This is deliberately narrower than the generic label decoder.  Both
    canonical labels must occupy unique columns in the same row, and the next
    row must contain one finite account status plus one independently valid
    calendar date in those exact columns.  A second candidate block or any
    business-shaped residue makes the observation ambiguous.
    """

    def header_column(row: list[str], label: str) -> list[int]:
        matches: list[int] = []
        for column, cell in enumerate(row):
            compact = _compact(cell)
            if compact.count(label) != 1:
                continue
            residue = compact.replace(label, "", 1)
            if len(residue) > 2 or any(character.isdigit() for character in residue):
                continue
            if any(
                account_label in residue
                for account_label in _ACCOUNT_LABELS
                if account_label != label
            ):
                continue
            matches.append(column)
        return matches

    candidates: list[dict[str, Any]] = []
    for header_index in range(len(rows) - 1):
        status_columns = header_column(rows[header_index], "账户状态")
        close_columns = header_column(rows[header_index], "账户关闭日期")
        if len(status_columns) != 1 or len(close_columns) != 1:
            continue
        status_column = status_columns[0]
        close_column = close_columns[0]
        if status_column == close_column:
            continue
        if (
            _compact(rows[header_index][status_column]) == "账户状态"
            and _compact(rows[header_index][close_column]) == "账户关闭日期"
        ):
            # The generic exact-column decoder already owns this clean case.
            continue
        value_row = rows[header_index + 1]
        if status_column >= len(value_row) or close_column >= len(value_row):
            continue
        status_raw = _clean(value_row[status_column])
        close_raw = _clean(value_row[close_column])
        status_values = _status_fields(status_raw)
        if status_values.get("account_status_resolution") not in {
            "resolved",
            "ocr_noise_normalized",
        }:
            continue
        normalized_close, _residue, close_resolution = _canonical_date_slot(close_raw)
        if normalized_close is None or close_resolution not in {"exact", "ascii_residue"}:
            continue
        candidates.append(
            {
                "status_raw": status_raw,
                "status_values": status_values,
                "status_row": header_index + 1,
                "status_column": status_column,
                "close_raw": close_raw,
                "close_date": normalized_close,
                "close_row": header_index + 1,
                "close_column": close_column,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def _pairs(rows: list[list[str]]) -> dict[str, str]:
    """Return only unambiguous exact-column label/value bindings."""

    observations, _unresolved = _exact_label_observations(rows)
    out: dict[str, str] = {}
    for label, values in observations.items():
        distinct = {_compact(value) for value, _row, _column in values}
        if len(distinct) == 1:
            out[label] = values[0][0]
    return out


def _label_row(row: list[str]) -> bool:
    return sum(_compact(cell) in _ACCOUNT_LABELS for cell in row) >= 2


def _column_centers(table: Any, width: int) -> list[float]:
    metadata = getattr(table, "metadata", None) or {}
    geometry = metadata.get("geometry") if isinstance(metadata, dict) else None
    bands = geometry.get("col_bands") if isinstance(geometry, dict) else None
    if isinstance(bands, list) and bands:
        centers: list[float] = []
        for band in bands:
            if isinstance(band, list) and len(band) >= 3:
                centers.append((float(band[0]) + float(band[2])) / 2.0)
            else:
                centers.append(float(len(centers)))
        return centers
    return [float(index) for index in range(width)]


def _month_centers(table: Any, rows: list[list[str]]) -> dict[int, float]:
    width = max((len(row) for row in rows), default=0)
    centers = _column_centers(table, width)
    for row in rows:
        candidates = {
            int(_compact(cell)): centers[index]
            for index, cell in enumerate(row)
            if index < len(centers) and re.fullmatch(r"(?:[1-9]|1[0-2])", _compact(cell))
        }
        if len(candidates) >= 5:
            return candidates
    return {}


def _nearest_month(column: int, centers: list[float], months: dict[int, float]) -> int | None:
    if not months or column >= len(centers):
        return None
    x = centers[column]
    return min(months, key=lambda month: abs(months[month] - x))


def _field(facts: dict[str, str], *labels: str) -> str | None:
    for label in labels:
        value = facts.get(_compact(label))
        if value not in (None, ""):
            return value
    return None


def _status_fields(raw_status: Any) -> dict[str, Any]:
    raw = _compact(raw_status)
    if not raw:
        return {}
    status_labels = {
        "正常": "active",
        "逾期": "active",
        "呆账": "active",
        "结清": "settled",
        "销户": "closed",
        "转出": "transferred_out",
        "结束": "closed",
        "未激活": "inactive",
        "冻结": "frozen",
        "止付": "suspended",
        "银行止付": "suspended",
        "司法追偿": "recovery",
        "担保物不足": "collateral_shortfall",
        "强制平仓": "forced_liquidation",
        "催收": "collection",
    }
    source_label = raw
    status = status_labels.get(raw)
    repaired_noise = False
    if status is None:
        matches = [label for label in status_labels if label in raw]
        if len(matches) == 1:
            remaining = raw.replace(matches[0], "", 1)
            if len(remaining) <= 2:
                source_label = matches[0]
                status = status_labels[source_label]
                repaired_noise = True
    if status is None:
        status = raw
    lifecycle = {
        "active": "open",
        "inactive": "open",
        "settled": "settled",
        "closed": "closed",
        "transferred_out": "transferred_out",
    }.get(status)
    result: dict[str, Any] = {
        "account_status": status,
        "account_status_raw": raw,
        "account_status_resolution": "ocr_noise_normalized" if repaired_noise else "resolved"
        if status in status_labels.values()
        else "unresolved",
    }
    if lifecycle:
        result["account_lifecycle_state"] = lifecycle
    if source_label == "未激活":
        result["card_activation_state"] = "not_activated"
    elif source_label == "呆账":
        result["credit_quality_status"] = "bad_debt"
    if source_label == "逾期":
        result["current_overdue"] = True
        result["current_overdue_status"] = "overdue"
    elif source_label in {"正常", "结清", "销户", "转出"}:
        result["current_overdue"] = False
        result["current_overdue_status"] = "not_overdue"
    return result


def _append_internal_field(record: dict[str, Any], key: str, field_name: str) -> None:
    values = record.setdefault(key, [])
    if field_name not in values:
        values.append(field_name)


def _mark_source_absent(record: dict[str, Any], field_name: str, raw: str = "--") -> None:
    _append_internal_field(record, "_source_absent_fields", field_name)
    canonical_raw = record.setdefault("canonical_raw", {})
    prior = canonical_raw.get(field_name)
    if prior in (None, ""):
        canonical_raw[field_name] = raw
        return
    prior_values = list(prior) if isinstance(prior, (list, tuple)) else [prior]
    if raw not in prior_values:
        prior_values.append(raw)
    canonical_raw[field_name] = (
        prior_values[0] if len(prior_values) == 1 else prior_values
    )


def _merge_exact_observation(
    parse_result: Any,
    record: dict[str, Any],
    *,
    dataset: str,
    target_record_id: str,
    field_name: str,
    value: Any,
    raw: str,
    source_ref: Mapping[str, Any],
    parser_stage: str,
) -> None:
    """Merge one exact-slot observation without last-write-wins behavior."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    ref = {**dict(source_ref), "field_name": field_name}
    refs = record.setdefault("source_refs_by_field", {}).setdefault(field_name, [])
    if ref not in refs:
        refs.append(ref)
    observations = record.setdefault("_field_observations", {}).setdefault(field_name, [])
    marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if not any(item.get("marker") == marker for item in observations):
        observations.append({"marker": marker, "value": value, "raw": raw, "source_refs": [ref]})
    else:
        prior = next(item for item in observations if item.get("marker") == marker)
        if ref not in prior.setdefault("source_refs", []):
            prior["source_refs"].append(ref)
    distinct = {str(item.get("marker") or "") for item in observations}
    if len(distinct) == 1 and field_name not in record.get("_invalid_observation_fields", []):
        record[field_name] = value
        record.setdefault("canonical_raw", {})[field_name] = raw
        return

    record.pop(field_name, None)
    _append_internal_field(record, "_unresolved_fields", field_name)
    prior_raw = record.setdefault("canonical_raw", {}).get(field_name)
    raw_values = prior_raw if isinstance(prior_raw, list) else ([prior_raw] if prior_raw not in (None, "") else [])
    for item in observations:
        candidate_raw = str(item.get("raw") or "")
        if candidate_raw not in raw_values:
            raw_values.append(candidate_raw)
    record.setdefault("canonical_raw", {})[field_name] = raw_values
    reported = record.setdefault("_reported_field_conflicts", [])
    if field_name in reported:
        return
    reported.append(field_name)
    conflict_refs = [
        candidate_ref
        for item in observations
        for candidate_ref in item.get("source_refs") or ()
        if isinstance(candidate_ref, Mapping)
    ]
    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_exact_slot_value_conflict",
            message="Canonical observations for one business field disagreed; the normalized value was withheld.",
            parser_stage=parser_stage,
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value=raw_values,
            source_refs=conflict_refs,
            reason_codes=("canonical_field_slot", "conflicting_exact_observations", "normalized_value_withheld"),
        ),
    )


def _reject_exact_observation(
    parse_result: Any,
    record: dict[str, Any],
    *,
    dataset: str,
    target_record_id: str,
    field_name: str,
    raw: str,
    source_ref: Mapping[str, Any],
    parser_stage: str,
) -> None:
    """Retain an invalid exact cell as issue evidence, never as business data."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    ref = {**dict(source_ref), "field_name": field_name}
    refs = record.setdefault("source_refs_by_field", {}).setdefault(field_name, [])
    if ref not in refs:
        refs.append(ref)
    record.pop(field_name, None)
    _append_internal_field(record, "_unresolved_fields", field_name)
    _append_internal_field(record, "_invalid_observation_fields", field_name)
    prior = record.setdefault("canonical_raw", {}).get(field_name)
    raw_values = prior if isinstance(prior, list) else ([prior] if prior not in (None, "") else [])
    if raw not in raw_values:
        raw_values.append(raw)
    record["canonical_raw"][field_name] = raw_values
    reported = record.setdefault("_reported_invalid_fields", [])
    if field_name in reported:
        return
    reported.append(field_name)
    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_exact_slot_value_invalid",
            message="A value was present in the canonical field slot but failed its field contract and was withheld.",
            parser_stage=parser_stage,
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value=raw_values,
            source_refs=(ref,),
            reason_codes=("canonical_field_slot", "field_contract_failed", "normalized_value_withheld"),
        ),
    )


def _merge_canonical_date_observation(
    parse_result: Any,
    record: dict[str, Any],
    *,
    dataset: str,
    target_record_id: str,
    field_name: str,
    raw: str,
    source_ref: Mapping[str, Any],
    parser_stage: str,
) -> str | None:
    """Validate, retain, and audit one canonical date-slot observation."""

    normalized, residue, resolution = _canonical_date_slot(raw)
    if normalized is None:
        _reject_exact_observation(
            parse_result,
            record,
            dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            raw=raw,
            source_ref=source_ref,
            parser_stage=parser_stage,
        )
        return None

    _merge_exact_observation(
        parse_result,
        record,
        dataset=dataset,
        target_record_id=target_record_id,
        field_name=field_name,
        value=normalized,
        raw=raw,
        source_ref=source_ref,
        parser_stage=parser_stage,
    )
    if resolution != "ascii_residue":
        return normalized

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_date_ascii_residue_corrected",
            message=(
                "One independently valid calendar date was retained from its canonical slot; "
                "bounded non-business ASCII OCR residue was preserved and reported."
            ),
            severity="info",
            status="resolved",
            parser_stage=parser_stage,
            target_dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value={"raw": raw, "residue": residue},
            candidate_value={"normalized_date": normalized},
            source_refs=(source_ref,),
            reason_codes=(
                "canonical_exact_date_slot",
                "single_independently_valid_calendar_date",
                "bounded_ascii_watermark_residue",
                "full_raw_preserved",
                "professional_field_correction",
            ),
        ),
    )
    return normalized


def _institution_refs_are_independent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Require two distinct source-bound field observations."""

    if dict(left) == dict(right):
        return False
    left_evidence = {str(value) for value in left.get("evidence_ids") or () if value}
    right_evidence = {str(value) for value in right.get("evidence_ids") or () if value}
    if left_evidence and right_evidence and left_evidence.isdisjoint(right_evidence):
        return True
    left_table = str(left.get("table_id") or "")
    right_table = str(right.get("table_id") or "")
    return bool(left_table and right_table and left_table != right_table)


def _merge_account_institution_observation(
    parse_result: Any,
    account: dict[str, Any],
    *,
    raw: str,
    source_ref: Mapping[str, Any],
    parser_stage: str,
) -> None:
    """Merge an institution without guessing across a leading Han boundary."""

    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        institution_name_has_separated_leading_han,
        normalize_institution_name,
    )

    target_record_id = str(account.get("account_id") or "")
    if not institution_name_has_separated_leading_han(raw):
        value = _account_institution(raw)
        if value is None:
            _reject_exact_observation(
                parse_result,
                account,
                dataset="credit_accounts",
                target_record_id=target_record_id,
                field_name="management_institution",
                raw=raw,
                source_ref=source_ref,
                parser_stage=parser_stage,
            )
            return
        _merge_exact_observation(
            parse_result,
            account,
            dataset="credit_accounts",
            target_record_id=target_record_id,
            field_name="management_institution",
            value=value,
            raw=raw,
            source_ref=source_ref,
            parser_stage=parser_stage,
        )
        pending = account.get("_pending_institution_observations")
        if not isinstance(pending, list):
            return
        matched = [
            item
            for item in pending
            if item.get("value") == value
            and _institution_refs_are_independent(item.get("source_ref") or {}, source_ref)
        ]
        if matched:
            for item in matched:
                _merge_exact_observation(
                    parse_result,
                    account,
                    dataset="credit_accounts",
                    target_record_id=target_record_id,
                    field_name="management_institution",
                    value=value,
                    raw=str(item.get("raw") or ""),
                    source_ref=item.get("source_ref") or {},
                    parser_stage=parser_stage,
                )
            account["_pending_institution_observations"] = [
                item for item in pending if item not in matched
            ]
        return

    value = _account_institution(raw, independently_corroborated=True)
    if value is None:
        _reject_exact_observation(
            parse_result,
            account,
            dataset="credit_accounts",
            target_record_id=target_record_id,
            field_name="management_institution",
            raw=raw,
            source_ref=source_ref,
            parser_stage=parser_stage,
        )
        return
    pending = account.setdefault("_pending_institution_observations", [])
    candidate = {
        "raw": raw,
        "value": normalize_institution_name(raw),
        "source_ref": dict(source_ref),
    }
    if candidate not in pending:
        pending.append(candidate)
    corroborating_refs = [
        item.get("source_ref") or {}
        for item in pending
        if item.get("value") == value
        and _institution_refs_are_independent(item.get("source_ref") or {}, source_ref)
    ]
    existing = account.get("_field_observations", {}).get("management_institution", [])
    for observation in existing:
        if observation.get("value") != value:
            continue
        corroborating_refs.extend(
            ref
            for ref in observation.get("source_refs") or ()
            if _institution_refs_are_independent(ref, source_ref)
        )
    if not corroborating_refs:
        return
    matched = [item for item in pending if item.get("value") == value]
    for item in matched:
        _merge_exact_observation(
            parse_result,
            account,
            dataset="credit_accounts",
            target_record_id=target_record_id,
            field_name="management_institution",
            value=value,
            raw=str(item.get("raw") or ""),
            source_ref=item.get("source_ref") or {},
            parser_stage=parser_stage,
        )
    account["_pending_institution_observations"] = [
        item for item in pending if item not in matched
    ]


def _flush_pending_account_institution_observations(
    parse_result: Any,
    account: dict[str, Any],
    *,
    boundary: str,
) -> None:
    pending = account.pop("_pending_institution_observations", [])
    if not pending:
        return
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    field_name = "management_institution"
    if account.get(field_name) in (None, ""):
        account.pop(field_name, None)
        _append_internal_field(account, "_unresolved_fields", field_name)
    refs = [
        dict(item.get("source_ref") or {})
        for item in pending
        if isinstance(item, Mapping) and item.get("source_ref")
    ]
    raw_values = [str(item.get("raw") or "") for item in pending if isinstance(item, Mapping)]
    account.setdefault("canonical_raw", {})[field_name] = raw_values
    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_institution_leading_boundary_ambiguous",
            message=(
                "A separated leading Han glyph could be either intra-name OCR whitespace or cross-cell debris; "
                "the joined institution was not silently selected without independent source evidence."
            ),
            parser_stage="candidate_b_account_canonical_slots",
            target_dataset="credit_accounts",
            target_record_id=str(account.get("account_id") or ""),
            field_name=field_name,
            observed_value=raw_values,
            candidate_value={
                "joined_values": sorted({str(item.get("value") or "") for item in pending}),
                "boundary": boundary,
            },
            source_refs=refs,
            reason_codes=(
                "separated_leading_han_boundary",
                "independent_source_corroboration_missing",
                "normalized_value_withheld",
            ),
        ),
    )


_ACCOUNT_CLUSTER_TOKEN_RE = re.compile(
    r"(?:19|20)\d{2}[.,/-]\d{1,2}[.,/-]\d{1,2}|"
    r"正常|逾期|呆账|结清|销户|转出|未激活|冻结|止付|"
    r"[-*#]+|(?<![A-Za-z0-9])\d[\d,.]*(?![A-Za-z0-9])"
)


def _single_business_cell(row: list[str]) -> tuple[int, str] | None:
    """Return one populated cell for event clusters that are canonically singular."""

    values = [(index, _clean(value)) for index, value in enumerate(row) if _clean(value)]
    return values[0] if len(values) == 1 else None


_ACCOUNT_CELL_FIRST_LOAN = frozenset({"管理机构", "账户标识", "开立日期"})
_ACCOUNT_CELL_FIRST_CARD = frozenset({"发卡机构", "账户标识", "开立日期"})
_ACCOUNT_CELL_LOAN_TERMS = frozenset({"到期日期", "借款金额", "账户币种"})
_ACCOUNT_CELL_CARD_TERMS = frozenset(
    {"账户授信额度", "币种", "业务种类", "担保方式"}
)
_ACCOUNT_CELL_LOAN_CLASSIFICATION = frozenset(
    {"业务种类", "担保方式", "还款期数"}
)
_ACCOUNT_CELL_REPAYMENT_TERMS = frozenset(
    {"还款频率", "还款方式", "共同借款标志"}
)
_ACCOUNT_CELL_CLUSTER_LABEL_SETS = frozenset(
    {
        _ACCOUNT_CELL_FIRST_LOAN,
        _ACCOUNT_CELL_FIRST_CARD,
        _ACCOUNT_CELL_LOAN_TERMS,
        _ACCOUNT_CELL_CARD_TERMS,
        _ACCOUNT_CELL_LOAN_CLASSIFICATION,
        _ACCOUNT_CELL_REPAYMENT_TERMS,
        _ACCOUNT_CELL_FIRST_LOAN | _ACCOUNT_CELL_LOAN_TERMS,
        _ACCOUNT_CELL_FIRST_CARD | _ACCOUNT_CELL_CARD_TERMS,
        _ACCOUNT_CELL_LOAN_CLASSIFICATION | _ACCOUNT_CELL_REPAYMENT_TERMS,
    }
)
_ACCOUNT_CELL_ORDERED_LABEL_GROUPS = (
    ("管理机构", "账户标识", "开立日期"),
    ("发卡机构", "账户标识", "开立日期"),
    ("到期日期", "借款金额", "账户币种"),
    ("账户授信额度", "币种", "业务种类", "担保方式"),
    ("业务种类", "担保方式", "还款期数"),
    ("还款频率", "还款方式", "共同借款标志"),
)
_ACCOUNT_CELL_ORDERED_COMBINED_LABEL_GROUPS = (
    _ACCOUNT_CELL_ORDERED_LABEL_GROUPS[0] + _ACCOUNT_CELL_ORDERED_LABEL_GROUPS[2],
    _ACCOUNT_CELL_ORDERED_LABEL_GROUPS[1] + _ACCOUNT_CELL_ORDERED_LABEL_GROUPS[3],
    _ACCOUNT_CELL_ORDERED_LABEL_GROUPS[4] + _ACCOUNT_CELL_ORDERED_LABEL_GROUPS[5],
)
_ACCOUNT_CELL_CLUSTER_LABELS = tuple(
    sorted(
        {label for labels in _ACCOUNT_CELL_CLUSTER_LABEL_SETS for label in labels},
        key=len,
        reverse=True,
    )
)
_ACCOUNT_CELL_LABEL_FIELDS = {
    "管理机构": "management_institution",
    "发卡机构": "management_institution",
    "账户标识": "account_identifier",
    "开立日期": "open_date",
    "到期日期": "due_date",
    "借款金额": "loan_amount",
    "账户授信额度": "credit_limit",
    "账户币种": "currency",
    "币种": "currency",
    "业务种类": "business_type",
    "担保方式": "guarantee_type",
    "还款期数": "repayment_periods",
    "还款频率": "repayment_frequency",
    "还款方式": "repayment_method",
    "共同借款标志": "co_borrower_flag",
}
_ACCOUNT_CELL_FINITE_ROLES = {
    "business_type": "account_business_type",
    "guarantee_type": "guarantee_type",
    "repayment_frequency": "repayment_frequency",
    "repayment_method": "repayment_method",
    "co_borrower_flag": "boolean_flag",
}


def _exact_account_cell_header_labels(value: Any) -> frozenset[str] | None:
    """Return one closed canonical header-label set, independent of order."""

    residue = _compact(value)
    observed: list[str] = []
    for label in _ACCOUNT_CELL_CLUSTER_LABELS:
        marker = _compact(label)
        count = residue.count(marker)
        if count > 1:
            return None
        if count == 1:
            observed.append(label)
            residue = residue.replace(marker, "", 1)
    labels = frozenset(observed)
    return labels if not residue and labels in _ACCOUNT_CELL_CLUSTER_LABEL_SETS else None


def _exact_account_cell_header_order(value: Any) -> tuple[str, ...] | None:
    """Return one canonical physical label order for a closed merged header."""

    marker = _compact(value)
    matches = [
        labels
        for labels in (
            *_ACCOUNT_CELL_ORDERED_LABEL_GROUPS,
            *_ACCOUNT_CELL_ORDERED_COMBINED_LABEL_GROUPS,
        )
        if marker == "".join(_compact(label) for label in labels)
    ]
    return matches[0] if len(matches) == 1 else None


def _exact_geometry_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    return (left, top, right, bottom) if right > left and bottom > top else None


def _account_merged_header_geometry_values(
    table: Any,
    rows: list[list[str]],
    *,
    header_row: int,
    header_column: int,
    labels: frozenset[str],
) -> tuple[str, list[dict[str, Any]]]:
    """Bind a closed merged header only through its exact physical grid.

    ``not_applicable`` preserves the existing whole-cell decoder for legacy
    tables without a physical geometry plane. Once an exact merged span is
    present, every ambiguity is ``rejected`` and cannot fall back to text
    order. ``resolved`` requires one unmerged exact value cell in each ordered
    physical header partition on the immediately following row.
    """

    if not (0 <= header_row < len(rows)) or not (
        0 <= header_column < len(rows[header_row])
    ):
        return "not_applicable", []
    ordered_labels = _exact_account_cell_header_order(
        rows[header_row][header_column]
    )
    if ordered_labels is None or frozenset(ordered_labels) != labels:
        return "not_applicable", []

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return "not_applicable", []
    geometry = metadata.get("geometry")
    if not isinstance(geometry, Mapping):
        geometry = metadata
    cell_bboxes = geometry.get("cell_bboxes")
    cell_status = geometry.get("cell_geometry_status")
    cell_evidence_ids = geometry.get("cell_evidence_ids")
    row_bands = geometry.get("row_bands")
    column_bands = geometry.get("col_bands")
    cell_spans = geometry.get("cell_spans")
    if not all(
        isinstance(value, list)
        for value in (
            cell_bboxes,
            cell_status,
            row_bands,
            column_bands,
            cell_spans,
        )
    ):
        return "not_applicable", []

    def exact_span_integer(
        span: Mapping[str, Any],
        key: str,
        *,
        minimum: int,
    ) -> int | None:
        value = span.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            return None
        return value

    header_spans: list[Mapping[str, Any]] = []
    for span in cell_spans:
        if not isinstance(span, Mapping):
            return "rejected", []
        span_row = exact_span_integer(span, "row", minimum=0)
        span_column = exact_span_integer(span, "col", minimum=0)
        if span_row is None or span_column is None:
            return "rejected", []
        if span_row == header_row and span_column == header_column:
            header_spans.append(span)
    if not header_spans:
        return "not_applicable", []
    if len(header_spans) != 1:
        return "rejected", []
    header_span = header_spans[0]
    row_span = exact_span_integer(header_span, "row_span", minimum=1)
    column_span = exact_span_integer(header_span, "col_span", minimum=1)
    if row_span is None or column_span is None:
        return "rejected", []
    if row_span != 1 or column_span <= 1 or column_span < len(ordered_labels):
        return "rejected", []
    if header_row + 1 >= len(rows):
        return "rejected", []

    def grid_value(grid: Any, row: int, column: int) -> Any:
        if not (
            isinstance(grid, list)
            and 0 <= row < len(grid)
            and isinstance(grid[row], list)
            and 0 <= column < len(grid[row])
        ):
            return None
        return grid[row][column]

    header_bbox = _exact_geometry_bbox(
        grid_value(cell_bboxes, header_row, header_column)
    )
    span_bbox = _exact_geometry_bbox(header_span.get("bbox"))
    header_status = str(
        grid_value(cell_status, header_row, header_column) or ""
    )
    if header_bbox is None or span_bbox != header_bbox or header_status != "exact":
        return "rejected", []
    if len(rows[header_row]) < header_column + column_span:
        return "rejected", []
    for covered_column in range(
        header_column + 1,
        header_column + column_span,
    ):
        if _clean(rows[header_row][covered_column]):
            return "rejected", []
        if grid_value(cell_bboxes, header_row, covered_column) is not None:
            return "rejected", []
        if str(grid_value(cell_status, header_row, covered_column) or "") != "derived":
            return "rejected", []
        covered_evidence = grid_value(
            cell_evidence_ids,
            header_row,
            covered_column,
        )
        if isinstance(covered_evidence, list) and covered_evidence:
            return "rejected", []
        if any(
            exact_span_integer(span, "row", minimum=0) == header_row
            and exact_span_integer(span, "col", minimum=0) == covered_column
            for span in cell_spans
        ):
            return "rejected", []

    def indexed_geometry_bands(
        bands: list[Any],
    ) -> dict[int, Mapping[str, Any]] | None:
        indexed: dict[int, Mapping[str, Any]] = {}
        for band in bands:
            if not isinstance(band, Mapping) or "index" not in band:
                return None
            raw_index = band.get("index")
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                return None
            index = raw_index
            if index < 0 or index in indexed:
                return None
            indexed[index] = band
        return indexed

    indexed_bands = indexed_geometry_bands(row_bands)
    indexed_column_bands = indexed_geometry_bands(column_bands)
    if indexed_bands is None or indexed_column_bands is None:
        return "rejected", []
    header_band = indexed_bands.get(header_row)
    value_band = indexed_bands.get(header_row + 1)
    first_column_band = indexed_column_bands.get(header_column)
    last_column_band = indexed_column_bands.get(
        header_column + column_span - 1
    )
    try:
        header_top = float((header_band or {}).get("y0"))
        header_bottom = float((header_band or {}).get("y1"))
        value_top = float((value_band or {}).get("y0"))
        value_bottom = float((value_band or {}).get("y1"))
        span_left = float((first_column_band or {}).get("x0"))
        span_right = float((last_column_band or {}).get("x1"))
    except (TypeError, ValueError):
        return "rejected", []
    if not all(
        math.isfinite(value)
        for value in (
            header_top,
            header_bottom,
            value_top,
            value_bottom,
            span_left,
            span_right,
        )
    ):
        return "rejected", []
    if abs(header_bottom - value_top) > 0.5:
        return "rejected", []
    declared_header_bbox = (
        span_left,
        header_top,
        span_right,
        header_bottom,
    )
    if any(
        abs(observed - declared) > 1e-6
        for observed, declared in zip(
            header_bbox,
            declared_header_bbox,
            strict=True,
        )
    ):
        return "rejected", []

    span_by_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for span in cell_spans:
        span_row = exact_span_integer(span, "row", minimum=0)
        span_column = exact_span_integer(span, "col", minimum=0)
        if span_row is None or span_column is None:
            return "rejected", []
        key = (span_row, span_column)
        span_by_cell[key].append(span)

    value_row = rows[header_row + 1]
    populated_columns_in_span = [
        column
        for column, raw_value in enumerate(value_row)
        if _clean(raw_value)
        and header_column <= column < header_column + column_span
    ]
    if populated_columns_in_span == [header_column]:
        # A single value cell aligned with the merged header is the existing
        # whole-cell cluster grammar. Its shape-unique typed values remain
        # governed by that decoder; this path is only for detached siblings.
        return "not_applicable", []
    contiguous_columns = list(
        range(header_column, header_column + len(ordered_labels))
    )
    interleaved_columns = [
        header_column + 1 + index * 2
        for index in range(len(ordered_labels))
    ]
    if populated_columns_in_span not in (
        contiguous_columns
        if column_span == len(ordered_labels)
        else [],
        interleaved_columns
        if column_span == len(ordered_labels) * 2 + 1
        else [],
    ):
        return "rejected", []
    candidates: list[dict[str, Any]] = []
    header_left, _header_top, header_right, _header_bottom = header_bbox
    for column, raw_value in enumerate(value_row):
        raw = _clean(raw_value)
        if not raw:
            continue
        bbox = _exact_geometry_bbox(
            grid_value(cell_bboxes, header_row + 1, column)
        )
        if bbox is None:
            continue
        overlap = max(0.0, min(header_right, bbox[2]) - max(header_left, bbox[0]))
        center = (bbox[0] + bbox[2]) / 2.0
        if overlap <= 0.0 and not header_left < center < header_right:
            continue
        if not (header_column <= column < header_column + column_span):
            return "rejected", []
        if str(grid_value(cell_status, header_row + 1, column) or "") != "exact":
            return "rejected", []
        primitive_column_band = indexed_column_bands.get(column)
        try:
            primitive_bbox = (
                float((primitive_column_band or {}).get("x0")),
                value_top,
                float((primitive_column_band or {}).get("x1")),
                value_bottom,
            )
        except (TypeError, ValueError):
            return "rejected", []
        if _exact_geometry_bbox(primitive_bbox) != primitive_bbox:
            return "rejected", []
        if any(
            abs(observed - declared) > 1e-6
            for observed, declared in zip(bbox, primitive_bbox, strict=True)
        ):
            return "rejected", []
        spans = span_by_cell.get((header_row + 1, column), [])
        if len(spans) > 1:
            return "rejected", []
        if spans:
            value_row_span = exact_span_integer(
                spans[0],
                "row_span",
                minimum=1,
            )
            value_column_span = exact_span_integer(
                spans[0],
                "col_span",
                minimum=1,
            )
            if value_row_span is None or value_column_span is None:
                return "rejected", []
            if value_row_span != 1 or value_column_span != 1:
                return "rejected", []
        if not header_left < center < header_right:
            return "rejected", []
        if bbox[1] < value_top - 0.5:
            return "rejected", []
        evidence_ids = grid_value(
            cell_evidence_ids, header_row + 1, column
        )
        candidates.append(
            {
                "raw": raw,
                "column": column,
                "bbox": bbox,
                "center": center,
                "evidence_ids": [
                    str(value) for value in evidence_ids or () if value
                ]
                if isinstance(evidence_ids, list)
                else [],
            }
        )

    if len(candidates) != len(ordered_labels):
        return "rejected", []
    candidates.sort(key=lambda item: (float(item["center"]), int(item["column"])))
    partition_width = (header_right - header_left) / len(ordered_labels)
    partitions: list[int] = []
    for item in candidates:
        offset = (float(item["center"]) - header_left) / partition_width
        nearest_boundary = round(offset)
        if 0 < nearest_boundary < len(ordered_labels) and abs(
            offset - nearest_boundary
        ) <= 0.03:
            return "rejected", []
        partitions.append(min(len(ordered_labels) - 1, int(offset)))
    if partitions != list(range(len(ordered_labels))):
        return "rejected", []

    return "resolved", [
        {
            **candidate,
            "label": label,
            "field_name": _ACCOUNT_CELL_LABEL_FIELDS[label],
        }
        for label, candidate in zip(ordered_labels, candidates, strict=True)
    ]


def _decode_account_geometry_field_value(
    field_name: str,
    raw: str,
) -> Any | None:
    """Apply one field contract to an already geometry-bound exact cell."""

    if not raw or is_explicit_source_absence(raw):
        return None
    if field_name == "management_institution":
        return _account_institution(raw)
    if field_name == "account_identifier":
        return _typed_identifier(raw)
    if field_name in {"open_date", "due_date"}:
        normalized, residue, resolution = _canonical_date_slot(raw)
        return normalized if normalized and not residue and resolution == "exact" else None
    if field_name in {"loan_amount", "credit_limit"}:
        compact = _clean(raw)
        if re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", compact) is None:
            return None
        value = _number(compact)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if field_name == "currency":
        currency, residue, resolution = _currency_token(raw)
        return currency if currency and not residue and resolution == "exact" else None
    if field_name == "repayment_periods":
        compact = _clean(raw)
        if re.fullmatch(r"\d{1,4}", compact) is None:
            return None
        value = _number(compact)
        return value if isinstance(value, int) and 0 <= value <= 1200 else None
    contract_role = _ACCOUNT_CELL_FINITE_ROLES.get(field_name)
    if contract_role:
        value = normalize_pboc_field(raw, contract_role)
        return value if validate_pboc_field(value, contract_role).valid else None
    return None


def _account_cluster_signature(value: Any) -> str:
    return re.sub(
        r"[\s,，.。;；:：|/()（）_\-]",
        "",
        str(value or ""),
    )


def _account_cluster_residue(value: Any, consumed_values: Iterable[Any]) -> str:
    residue = _account_cluster_signature(value)
    for consumed in consumed_values:
        signature = _account_cluster_signature(consumed)
        if signature and signature in residue:
            residue = residue.replace(signature, "", 1)
    return residue


def _unique_account_finite_span(
    value: Any,
    *,
    contract_role: str,
) -> tuple[str, str, int, int] | None:
    """Find one unique maximal exact finite-vocabulary span."""

    marker = _clean(value).translate(str.maketrans({"（": "(", "）": ")"}))
    matches: list[tuple[str, str, int, int]] = []
    for candidate in pboc_controlled_vocabulary(contract_role):
        candidate_marker = _clean(candidate).translate(
            str.maketrans({"（": "(", "）": ")"})
        )
        start = 0
        while candidate_marker and (index := marker.find(candidate_marker, start)) >= 0:
            end = index + len(candidate_marker)
            business_before = bool(
                index > 0 and re.match(r"[0-9A-Za-z\u3400-\u9fff]", marker[index - 1])
            )
            business_after = bool(
                end < len(marker)
                and re.match(r"[0-9A-Za-z\u3400-\u9fff]", marker[end])
            )
            if business_before or business_after:
                start = index + 1
                continue
            matches.append(
                (
                    normalize_pboc_field(candidate, contract_role),
                    candidate_marker,
                    index,
                    end,
                )
            )
            start = index + 1
    maximal = [
        match
        for match in matches
        if not any(
            match[2] >= other[2]
            and match[3] <= other[3]
            and len(match[1]) < len(other[1])
            for other in matches
        )
    ]
    if not maximal:
        return None
    longest = max(len(match[1]) for match in maximal)
    longest_matches = [match for match in maximal if len(match[1]) == longest]
    return longest_matches[0] if len(longest_matches) == 1 else None


def _unique_closed_account_finite_span(
    value: Any,
    *,
    contract_role: str,
) -> tuple[str, str, int, int] | None:
    """Resolve one finite value inside a topology-proven collapsed cell.

    Canonical table collapse may remove the whitespace between adjacent roles
    (for example ``个人消费贷款抵押36``).  The closed header label set proves the
    participating fields, so exact non-overlapping vocabulary spans remain
    safe even without word boundaries.  Multiple distinct spans for one role
    remain unresolved.
    """

    marker = _clean(value).translate(str.maketrans({"（": "(", "）": ")"}))
    matches: list[tuple[str, str, int, int]] = []
    for candidate in pboc_controlled_vocabulary(contract_role):
        candidate_marker = _clean(candidate).translate(
            str.maketrans({"（": "(", "）": ")"})
        )
        start = 0
        while candidate_marker and (index := marker.find(candidate_marker, start)) >= 0:
            matches.append(
                (
                    normalize_pboc_field(candidate, contract_role),
                    candidate_marker,
                    index,
                    index + len(candidate_marker),
                )
            )
            start = index + 1
    maximal = [
        match
        for match in matches
        if not any(
            match[2] >= other[2]
            and match[3] <= other[3]
            and len(match[1]) < len(other[1])
            for other in matches
        )
    ]
    unique = list(dict.fromkeys(maximal))
    return unique[0] if len(unique) == 1 else None


def _account_cluster_number_tokens(value: Any) -> list[str]:
    return [
        token
        for token in _ACCOUNT_CLUSTER_TOKEN_RE.findall(str(value or ""))
        if _date(token) is None
        and isinstance(_number(token), int)
        and not isinstance(_number(token), bool)
    ]


def _unique_account_money_token(value: Any) -> str | None:
    tokens = list(dict.fromkeys(_account_cluster_number_tokens(value)))
    if len(tokens) == 1:
        return tokens[0]
    comma_grouped = [
        token for token in tokens if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token)
    ]
    return comma_grouped[0] if len(comma_grouped) == 1 else None


def _account_currency_observation(value: Any) -> tuple[str, str] | None:
    """Return one exact currency token, excluding substrings of business prose."""

    raw = str(value or "")
    matches: list[tuple[int, int, str, str, int]] = []
    for alias, code in _CURRENCY_TOKEN_ALIASES.items():
        marker = _compact(alias)
        if not marker:
            continue
        pattern = re.compile(r"\s*".join(re.escape(char) for char in marker), re.IGNORECASE)
        for match in pattern.finditer(raw):
            before = raw[: match.start()].rstrip()[-1:]
            after = raw[match.end() :].lstrip()[:1]
            if marker.isascii() and (
                (before and before.isascii() and before.isalnum())
                or (after and after.isascii() and after.isalnum())
            ):
                continue
            if len(marker) == 1 and re.fullmatch(r"[\u3400-\u9fff]", marker) and (
                re.fullmatch(r"[\u3400-\u9fff]", before or "")
                or re.fullmatch(r"[\u3400-\u9fff]", after or "")
            ):
                continue
            matches.append(
                (match.start(), match.end(), code, match.group(), len(marker))
            )

    maximal = [
        match
        for match in matches
        if not any(
            other[0] <= match[0]
            and match[1] <= other[1]
            and other[4] > match[4]
            for other in matches
        )
    ]
    currencies = {match[2] for match in maximal}
    if len(currencies) != 1:
        return None
    currency = next(iter(currencies))
    longest = max(match[4] for match in maximal)
    source_tokens = list(
        dict.fromkeys(match[3] for match in maximal if match[4] == longest)
    )
    return (currency, source_tokens[0]) if len(source_tokens) == 1 else None


def _decode_account_cell_cluster(
    labels: frozenset[str],
    raw: str,
) -> tuple[dict[str, tuple[Any, str]], tuple[str, ...], str]:
    """Decode fields invariant under the OCR traversal of one closed cell."""

    expected_fields = {
        _ACCOUNT_CELL_LABEL_FIELDS[label]
        for label in labels
        if label in _ACCOUNT_CELL_LABEL_FIELDS
    }
    fields: dict[str, tuple[Any, str]] = {}
    consumed: list[str] = []

    date_fields = [field for field in ("open_date", "due_date") if field in expected_fields]
    full_dates = [
        (raw[start:end], normalized)
        for (start, end), normalized in _valid_date_spans(raw)
        if len(normalized) == 10
    ]
    if len(date_fields) == 1 and len(full_dates) == 1:
        date_raw, normalized = full_dates[0]
        fields[date_fields[0]] = (normalized, date_raw)
        consumed.append(date_raw)

    money_fields = [
        field for field in ("loan_amount", "credit_limit") if field in expected_fields
    ]
    if len(money_fields) == 1 and (money_raw := _unique_account_money_token(raw)):
        fields[money_fields[0]] = (int(_number(money_raw) or 0), money_raw)
        consumed.append(money_raw)

    if "currency" in expected_fields and (
        currency_observation := _account_currency_observation(raw)
    ):
        currency, currency_raw = currency_observation
        fields["currency"] = (currency, currency_raw)
        consumed.append(currency_raw)

    finite_candidates = {
        field_name: candidate
        for field_name, contract_role in _ACCOUNT_CELL_FINITE_ROLES.items()
        if field_name in expected_fields
        and (
            candidate := (
                _unique_closed_account_finite_span(
                    raw,
                    contract_role=contract_role,
                )
                if field_name in {"business_type", "guarantee_type"}
                else _unique_account_finite_span(
                    raw,
                    contract_role=contract_role,
                )
            )
        )
        is not None
    }
    shared_spans = {
        candidate[2:]
        for candidate in finite_candidates.values()
        if sum(other[2:] == candidate[2:] for other in finite_candidates.values()) > 1
    }
    for field_name, candidate in finite_candidates.items():
        normalized, source_token, start, end = candidate
        if (start, end) in shared_spans:
            continue
        fields[field_name] = (normalized, source_token)
        consumed.append(source_token)

    if "repayment_periods" in expected_fields:
        remaining_numbers = [
            token
            for token in _account_cluster_number_tokens(raw)
            if _account_cluster_signature(token)
            not in {_account_cluster_signature(value) for value in consumed}
        ]
        if len(remaining_numbers) == 1:
            periods = int(_number(remaining_numbers[0]) or 0)
            if 0 <= periods <= 1200:
                fields["repayment_periods"] = (periods, remaining_numbers[0])
                consumed.append(remaining_numbers[0])

    unresolved = tuple(
        field_name for field_name in sorted(expected_fields) if field_name not in fields
    )
    return fields, unresolved, _account_cluster_residue(raw, consumed)


def _account_header_suffix_institution(
    value: Any,
) -> tuple[str, str | None] | None:
    """Decode one exact institution suffix attached to its canonical label."""

    raw = _clean(value)
    for label in ("管理机构", "发卡机构"):
        match = re.fullmatch(rf"\s*{label}(?:\s+|\s*[:：]\s*)(.+)", raw)
        if match is None:
            continue
        suffix = _clean(match.group(1))
        if any(_compact(other) in _compact(suffix) for other in _ACCOUNT_LABELS):
            return None
        normalized = _account_institution(suffix)
        if normalized is None:
            return (label, None)
        source_signature = re.sub(r"\s+", "", suffix)
        normalized_signature = re.sub(r"\s+", "", normalized)
        independently_valid = bool(
            source_signature == normalized_signature
            and len(re.findall(r"[\u3400-\u9fff]", normalized_signature)) >= 3
            and re.search(
                r"(?:银行|公司|中心|信用社|信托|分行|支行|营业部)$",
                normalized_signature,
            )
        )
        return (label, normalized if independently_valid else None)
    return None


def _report_account_cluster_residue(
    parse_result: Any,
    account: dict[str, Any],
    *,
    target_record_id: str,
    field_names: Iterable[str],
    raw: str,
    residue: str,
    source_ref: dict[str, Any],
) -> None:
    """Keep typed values while exposing residue in their physical cell."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    account["extraction_status"] = "review"
    for field_name in dict.fromkeys(field_names):
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_account_cluster_residue_unresolved",
                message=(
                    "A closed canonical account cell retained a uniquely typed field "
                    "but also contained unassigned OCR residue."
                ),
                parser_stage="candidate_b_account_closed_cell_cluster",
                target_dataset="credit_accounts",
                target_record_id=target_record_id,
                field_name=field_name,
                observed_value={
                    "raw_cluster": raw,
                    "unconsumed_residue": residue,
                },
                source_refs=(source_ref,),
                reason_codes=(
                    "closed_canonical_header_label_set",
                    "uniquely_typed_value_retained",
                    "cell_residue_reported",
                ),
            ),
        )


def _report_account_cell_cluster_unresolved(
    parse_result: Any,
    account: dict[str, Any],
    *,
    target_record_id: str,
    field_names: Iterable[str],
    raw: str,
    source_ref: dict[str, Any],
) -> None:
    """Report fields ambiguous inside one cluster without poisoning later evidence."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    for field_name in dict.fromkeys(field_names):
        if account.get(field_name) not in (None, ""):
            continue
        _append_internal_field(account, "_unresolved_fields", field_name)
        account["extraction_status"] = "review"
        raw_values = account.setdefault("canonical_raw", {}).get(field_name)
        if not isinstance(raw_values, list):
            raw_values = [raw_values] if raw_values not in (None, "") else []
        if raw not in raw_values:
            raw_values.append(raw)
        account["canonical_raw"][field_name] = raw_values
        reported = account.setdefault("_reported_account_cluster_fields", [])
        if field_name in reported:
            continue
        reported.append(field_name)
        ref = {**source_ref, "field_name": field_name}
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_account_cluster_field_unresolved",
                message=(
                    "A closed canonical account cell did not uniquely assign one "
                    "of its declared business fields."
                ),
                parser_stage="candidate_b_account_closed_cell_cluster",
                target_dataset="credit_accounts",
                target_record_id=target_record_id,
                field_name=field_name,
                observed_value=raw_values,
                source_refs=(ref,),
                reason_codes=(
                    "closed_canonical_header_label_set",
                    "field_not_uniquely_typed",
                    "normalized_value_withheld",
                ),
            ),
        )


def _report_collapsed_cluster_fields(
    parse_result: Any,
    record: dict[str, Any],
    *,
    dataset: str,
    target_record_id: str,
    raw: str,
    source_ref: dict[str, Any],
    unresolved_fields: Iterable[str],
    parser_stage: str,
) -> None:
    """Retain every unresolved canonical role from one collapsed source cell."""

    for field_name in dict.fromkeys(str(value) for value in unresolved_fields if value):
        if record.get(field_name) not in (None, ""):
            continue
        _reject_exact_observation(
            parse_result,
            record,
            dataset=dataset,
            target_record_id=target_record_id,
            field_name=field_name,
            raw=raw,
            source_ref=source_ref,
            parser_stage=parser_stage,
        )


def _native_table_geometry(table: Any) -> Mapping[str, Any] | None:
    metadata = getattr(table, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    geometry = metadata.get("geometry")
    return geometry if isinstance(geometry, Mapping) else None


def _indexed_geometry_bands(
    geometry: Mapping[str, Any],
    key: str,
    *,
    lower_key: str,
    upper_key: str,
) -> dict[int, tuple[float, float]] | None:
    raw_bands = geometry.get(key)
    if not isinstance(raw_bands, list):
        return None
    bands: dict[int, tuple[float, float]] = {}
    for raw_band in raw_bands:
        if not isinstance(raw_band, Mapping):
            return None
        index = raw_band.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return None
        try:
            lower = float(raw_band.get(lower_key))
            upper = float(raw_band.get(upper_key))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            return None
        if index in bands:
            return None
        bands[index] = (lower, upper)
    return bands


def _exact_native_table_cell(
    table: Any,
    *,
    row: int,
    column: int,
) -> tuple[tuple[float, float, float, float], tuple[str, ...]] | None:
    """Return one exact, unspanned cell on an indexed native lattice."""

    geometry = _native_table_geometry(table)
    if geometry is None:
        return None
    row_bands = _indexed_geometry_bands(
        geometry,
        "row_bands",
        lower_key="y0",
        upper_key="y1",
    )
    column_bands = _indexed_geometry_bands(
        geometry,
        "col_bands",
        lower_key="x0",
        upper_key="x1",
    )
    if row_bands is None or column_bands is None:
        return None
    if row not in row_bands or column not in column_bands:
        return None
    cell_bboxes = geometry.get("cell_bboxes")
    cell_status = geometry.get("cell_geometry_status")
    cell_evidence_ids = geometry.get("cell_evidence_ids")
    if not all(
        isinstance(value, list)
        for value in (cell_bboxes, cell_status, cell_evidence_ids)
    ):
        return None
    if any(
        row >= len(grid)
        or not isinstance(grid[row], list)
        or column >= len(grid[row])
        for grid in (cell_bboxes, cell_status, cell_evidence_ids)
    ):
        return None
    if str(cell_status[row][column] or "") != "exact":
        return None
    bbox = _exact_geometry_bbox(cell_bboxes[row][column])
    evidence_ids = cell_evidence_ids[row][column]
    if bbox is None or not isinstance(evidence_ids, list):
        return None
    normalized_evidence = tuple(
        dict.fromkeys(str(value) for value in evidence_ids if str(value or ""))
    )
    if not normalized_evidence:
        return None
    expected = (
        column_bands[column][0],
        row_bands[row][0],
        column_bands[column][1],
        row_bands[row][1],
    )
    if any(abs(left - right) > 1.0 for left, right in zip(bbox, expected, strict=True)):
        return None
    spans = geometry.get("cell_spans")
    if spans is not None:
        if not isinstance(spans, list):
            return None
        for span in spans:
            if not isinstance(span, Mapping):
                return None
            try:
                span_row = int(span.get("row"))
                span_column = int(span.get("col"))
                row_span = int(span.get("row_span"))
                column_span = int(span.get("col_span"))
            except (TypeError, ValueError):
                return None
            if (
                span_row <= row < span_row + row_span
                and span_column <= column < span_column + column_span
            ):
                return None
    return bbox, normalized_evidence


_EXACT_TWO_CELL_CARD_HEADERS = (
    "发卡机构账户标识开立日期账户授信额度",
    "共享授信额度币种业务种类担保方式",
)


def _exact_two_cell_card_cluster_values(
    table: Any,
    rows: list[list[str]],
) -> dict[str, Any] | None:
    """Decode the one observed two-wide-cell card identity layout."""

    if (
        len(rows) < 2
        or len(rows[0]) != 2
        or len(rows[1]) != 2
        or tuple(_compact(value) for value in rows[0])
        != _EXACT_TWO_CELL_CARD_HEADERS
    ):
        return None
    geometry = _native_table_geometry(table)
    if geometry is None:
        return None
    column_bands = _indexed_geometry_bands(
        geometry,
        "col_bands",
        lower_key="x0",
        upper_key="x1",
    )
    if column_bands is None or set(column_bands) != {0, 1}:
        return None
    cell_geometry = {
        (row, column): _exact_native_table_cell(table, row=row, column=column)
        for row in (0, 1)
        for column in (0, 1)
    }
    if any(value is None for value in cell_geometry.values()):
        return None

    left_raw = _clean(rows[1][0])
    left = left_raw.split()
    if (
        len(left) != 7
        or re.fullmatch(r"[A-Z]\d{8}[A-Z]", left[0], re.I) is None
        or re.fullmatch(r"[\u3400-\u9fff]+", left[1]) is None
        or re.fullmatch(r"\d{4,20}", left[2]) is None
        or _date(left[3]) is None
        or re.fullmatch(r"\d{1,3}(?:,\d{3})+", left[4]) is None
        or re.fullmatch(r"[\u3400-\u9fff]+", left[5]) is None
        or re.fullmatch(r"\d{4,20}", left[6]) is None
    ):
        return None
    institution = _account_institution(f"{left[1]}{left[5]}")
    identifier = _canonical_pboc_account_identifier(f"{left[0]}{left[2]}{left[6]}")
    open_date = _date(left[3])
    credit_limit = _number(left[4])
    if (
        not institution
        or not identifier
        or open_date is None
        or not isinstance(credit_limit, int)
        or isinstance(credit_limit, bool)
        or credit_limit < 0
    ):
        return None

    right_raw = _clean(rows[1][1])
    right = right_raw.split()
    if len(right) not in {3, 4}:
        return None
    currency, currency_residue, currency_resolution = _currency_token(right[0])
    business_type = normalize_pboc_field(
        _business_text(right[1]),
        "account_business_type",
    )
    guarantee_type = normalize_pboc_field(
        _business_text(right[2]),
        "guarantee_type",
    )
    residue = "".join(right[3:])
    if (
        currency is None
        or currency_resolution != "exact"
        or currency_residue
        or not validate_pboc_field(business_type, "account_business_type").valid
        or not validate_pboc_field(guarantee_type, "guarantee_type").valid
        or (residue and re.fullmatch(r"[\u3400-\u9fff]", residue) is None)
    ):
        return None
    return {
        "values": {
            "management_institution": institution,
            "account_identifier": identifier,
            "open_date": open_date,
            "credit_limit": credit_limit,
            "currency": currency,
            "business_type": business_type,
            "guarantee_type": guarantee_type,
        },
        "raw_by_field": {
            "management_institution": left_raw,
            "account_identifier": left_raw,
            "open_date": left_raw,
            "credit_limit": left_raw,
            "currency": right_raw,
            "business_type": right_raw,
            "guarantee_type": right_raw,
        },
        "column_by_field": {
            "management_institution": 0,
            "account_identifier": 0,
            "open_date": 0,
            "credit_limit": 0,
            "currency": 1,
            "business_type": 1,
            "guarantee_type": 1,
        },
        "residue": residue,
    }


def _apply_collapsed_account_clusters(
    parse_result: Any,
    account: dict[str, Any],
    rows: list[list[str]],
    *,
    page: Any,
    table: Any,
    physical_row_indices: list[int | None] | None,
) -> None:
    """Decode only closed PBOC clusters whose role assignment is unique."""

    target_record_id = str(account.get("account_id") or "")

    def physical(index: int) -> int | None:
        if physical_row_indices is None:
            return index
        return physical_row_indices[index] if 0 <= index < len(physical_row_indices) else None

    def bind(
        field_name: str,
        value: Any,
        raw: str,
        row_index: int,
        column: int,
        *,
        binding: str = "closed_canonical_account_cluster",
    ) -> None:
        source_row = physical(row_index)
        if source_row is None:
            return
        ref = _source_ref(page, table, row=source_row, column=column)
        ref["binding"] = binding
        ref["binding_quality"] = binding
        _merge_exact_observation(
            parse_result,
            account,
            dataset="credit_accounts",
            target_record_id=target_record_id,
            field_name=field_name,
            value=value,
            raw=raw,
            source_ref=ref,
            parser_stage="candidate_b_account_closed_cluster",
        )

    def bind_status(raw: str, row_index: int, column: int) -> None:
        for field_name, value in _status_fields(raw).items():
            bind(field_name, value, raw, row_index, column)

    exact_two_cell_card = (
        _exact_two_cell_card_cluster_values(table, rows)
        if str(account.get("account_type") or "")
        in {"credit_card", "quasi_credit_card"}
        else None
    )
    if exact_two_cell_card is not None:
        for field_name, value in exact_two_cell_card["values"].items():
            column = int(exact_two_cell_card["column_by_field"][field_name])
            raw = str(exact_two_cell_card["raw_by_field"][field_name])
            bind(
                field_name,
                value,
                raw,
                1,
                column,
                binding="closed_exact_two_cell_card_cluster",
            )
            if field_name == "currency":
                bind(
                    "account_currency",
                    value,
                    raw,
                    1,
                    column,
                    binding="closed_exact_two_cell_card_cluster",
                )
        residue = str(exact_two_cell_card.get("residue") or "")
        if residue:
            residue_ref = _source_ref(page, table, row=physical(1), column=1)
            residue_ref["binding"] = "closed_exact_two_cell_card_cluster"
            residue_ref["binding_quality"] = "closed_exact_two_cell_card_cluster"
            _report_account_cluster_residue(
                parse_result,
                account,
                target_record_id=target_record_id,
                field_names=("currency", "business_type", "guarantee_type"),
                raw=str(rows[1][1]),
                residue=residue,
                source_ref=residue_ref,
            )

    for header_index in range(len(rows) - 1):
        for value_column, raw_header in enumerate(rows[header_index]):
            if exact_two_cell_card is not None and header_index == 0 and value_column in {0, 1}:
                continue
            header_raw = _clean(raw_header)
            if not header_raw:
                continue
            value_raw = _clean(
                rows[header_index + 1][value_column]
                if value_column < len(rows[header_index + 1])
                else ""
            )

            suffix_institution = _account_header_suffix_institution(header_raw)
            if suffix_institution is not None:
                _label, institution = suffix_institution
                if institution is not None:
                    bind(
                        "management_institution",
                        institution,
                        header_raw,
                        header_index,
                        value_column,
                        binding="closed_canonical_account_header_suffix",
                    )
                else:
                    cluster_ref = _source_ref(
                        page,
                        table,
                        row=physical(header_index),
                        column=value_column,
                    )
                    _report_collapsed_cluster_fields(
                        parse_result,
                        account,
                        dataset="credit_accounts",
                        target_record_id=target_record_id,
                        raw=header_raw,
                        source_ref=cluster_ref,
                        unresolved_fields=("management_institution",),
                        parser_stage="candidate_b_account_header_suffix",
                    )
                continue

            labels = _exact_account_cell_header_labels(header_raw)
            if labels is not None:
                geometry_status, geometry_values = (
                    _account_merged_header_geometry_values(
                        table,
                        rows,
                        header_row=header_index,
                        header_column=value_column,
                        labels=labels,
                    )
                )
                if geometry_status == "rejected":
                    cluster_ref = _source_ref(
                        page,
                        table,
                        row=physical(header_index),
                        column=value_column,
                    )
                    cluster_ref["binding"] = (
                        "closed_canonical_account_merged_header_geometry"
                    )
                    cluster_ref["binding_quality"] = (
                        "closed_canonical_account_merged_header_geometry_rejected"
                    )
                    _report_account_cell_cluster_unresolved(
                        parse_result,
                        account,
                        target_record_id=target_record_id,
                        raw=_clean(" ".join(rows[header_index + 1])),
                        source_ref=cluster_ref,
                        field_names=(
                            _ACCOUNT_CELL_LABEL_FIELDS[label] for label in labels
                        ),
                    )
                    continue
                if geometry_status == "resolved":
                    source_row = physical(header_index + 1)
                    if source_row is None:
                        continue
                    for item in geometry_values:
                        field_name = str(item["field_name"])
                        raw = str(item["raw"])
                        source_ref = _source_ref(
                            page,
                            table,
                            row=source_row,
                            column=int(item["column"]),
                        )
                        source_ref.update(
                            {
                                "source": "native_detail_table_cell",
                                "geometry_scope": "cell",
                                "bbox": list(item["bbox"]),
                                "evidence_ids": list(item["evidence_ids"]),
                                "binding": (
                                    "closed_canonical_account_merged_header_geometry"
                                ),
                                "binding_quality": (
                                    "closed_canonical_account_merged_header_geometry"
                                ),
                                "header_row": physical(header_index),
                                "header_column": value_column,
                            }
                        )
                        value = _decode_account_geometry_field_value(
                            field_name,
                            raw,
                        )
                        if value is None:
                            _reject_exact_observation(
                                parse_result,
                                account,
                                dataset="credit_accounts",
                                target_record_id=target_record_id,
                                field_name=field_name,
                                raw=raw,
                                source_ref=source_ref,
                                parser_stage=(
                                    "candidate_b_account_merged_header_geometry"
                                ),
                            )
                            continue
                        if field_name == "management_institution":
                            _merge_account_institution_observation(
                                parse_result,
                                account,
                                raw=raw,
                                source_ref=source_ref,
                                parser_stage=(
                                    "candidate_b_account_merged_header_geometry"
                                ),
                            )
                            continue
                        if field_name in {"open_date", "due_date"}:
                            _merge_canonical_date_observation(
                                parse_result,
                                account,
                                dataset="credit_accounts",
                                target_record_id=target_record_id,
                                field_name=field_name,
                                raw=raw,
                                source_ref=source_ref,
                                parser_stage=(
                                    "candidate_b_account_merged_header_geometry"
                                ),
                            )
                            continue
                        _merge_exact_observation(
                            parse_result,
                            account,
                            dataset="credit_accounts",
                            target_record_id=target_record_id,
                            field_name=field_name,
                            value=value,
                            raw=raw,
                            source_ref=source_ref,
                            parser_stage=(
                                "candidate_b_account_merged_header_geometry"
                            ),
                        )
                        if field_name == "currency":
                            _merge_exact_observation(
                                parse_result,
                                account,
                                dataset="credit_accounts",
                                target_record_id=target_record_id,
                                field_name="account_currency",
                                value=value,
                                raw=raw,
                                source_ref=source_ref,
                                parser_stage=(
                                    "candidate_b_account_merged_header_geometry"
                                ),
                            )
                    continue
                fields, unresolved_fields, residue = _decode_account_cell_cluster(
                    labels,
                    value_raw,
                )
                cluster_row_index = header_index + 1 if value_raw else header_index
                cluster_ref = _source_ref(
                    page,
                    table,
                    row=physical(cluster_row_index),
                    column=value_column,
                )
                cluster_ref["binding"] = "closed_canonical_account_cell_cluster"
                cluster_ref["binding_quality"] = (
                    "closed_canonical_account_cell_cluster"
                )
                for field_name, (value, _source_token) in fields.items():
                    bind(
                        field_name,
                        value,
                        value_raw,
                        header_index + 1,
                        value_column,
                        binding="closed_canonical_account_cell_cluster",
                    )
                    if field_name == "currency":
                        bind(
                            "account_currency",
                            value,
                            value_raw,
                            header_index + 1,
                            value_column,
                            binding="closed_canonical_account_cell_cluster",
                        )
                _report_account_cell_cluster_unresolved(
                    parse_result,
                    account,
                    target_record_id=target_record_id,
                    raw=value_raw,
                    source_ref=cluster_ref,
                    field_names=unresolved_fields,
                )
                if residue and fields:
                    _report_account_cluster_residue(
                        parse_result,
                        account,
                        target_record_id=target_record_id,
                        field_names=fields,
                        raw=value_raw,
                        residue=residue,
                        source_ref=cluster_ref,
                    )
                continue

            header = _compact(header_raw)
            tokens = _ACCOUNT_CLUSTER_TOKEN_RE.findall(value_raw)

            # Non-revolving status block.  Two identical finite statuses and
            # two shape-distinct numbers make every role unique even when OCR
            # merged the physical columns into one cell.
            if all(
                marker in header
                for marker in ("五级分类", "账户状态", "余额", "剩余还款期数")
            ):
                status_tokens = [
                    token
                    for token in tokens
                    if token in {"正常", "逾期", "呆账", "结清", "销户", "转出"}
                ]
                number_tokens = [
                    token
                    for token in tokens
                    if isinstance(_number(token), int)
                    and not isinstance(_number(token), bool)
                ]
                balance_tokens = [
                    token
                    for token in number_tokens
                    if "," in token or int(_number(token) or 0) >= 1000
                ]
                remaining_tokens = [
                    token
                    for token in number_tokens
                    if 0 <= int(_number(token) or 0) <= 600
                    and token not in balance_tokens
                ]
                if len(status_tokens) == 2 and len(set(status_tokens)) == 1:
                    bind(
                        "five_tier_class",
                        status_tokens[0],
                        status_tokens[0],
                        header_index + 1,
                        value_column,
                    )
                    bind_status(status_tokens[0], header_index + 1, value_column)
                if len(balance_tokens) == 1 and len(remaining_tokens) == 1:
                    bind(
                        "balance",
                        int(_number(balance_tokens[0]) or 0),
                        balance_tokens[0],
                        header_index + 1,
                        value_column,
                    )
                    bind(
                        "remaining_periods",
                        int(_number(remaining_tokens[0]) or 0),
                        remaining_tokens[0],
                        header_index + 1,
                        value_column,
                    )
                continue

            # Known merged card summary traversal.  The signature is the
            # canonical seven-column PBOC block; other token counts/orders are
            # withheld.
            card_signature = (
                "未出单的大额" in header
                and "最近6个月" in header
                and "剩余分期期数" in header
                and "最大使用额度" in header
                and "账户状态" in header
                and "余额" in header
                and "已用额度" in header
                and "平均使用额度" in header
                and "专项分期余额" in header
            )
            if card_signature and len(tokens) == 7:
                numeric = [_number(token) for token in tokens]
                if (
                    isinstance(numeric[0], int)
                    and 0 <= numeric[0] <= 600
                    and all(
                        isinstance(numeric[index], int) for index in (1, 2, 3, 5)
                    )
                    and tokens[4]
                    in {
                        "正常",
                        "逾期",
                        "呆账",
                        "结清",
                        "销户",
                        "转出",
                        "未激活",
                        "冻结",
                        "止付",
                    }
                    and numeric[3] >= max(int(numeric[1]), int(numeric[2]))
                    and re.fullmatch(r"[-*#]+", tokens[6]) is not None
                ):
                    for field_name, index in (
                        ("remaining_periods", 0),
                        ("used_amount", 1),
                        ("recent_6_month_average_used_amount", 2),
                        ("maximum_used_amount", 3),
                        ("balance", 5),
                    ):
                        bind(
                            field_name,
                            int(numeric[index]),
                            tokens[index],
                            header_index + 1,
                            value_column,
                        )
                    bind_status(tokens[4], header_index + 1, value_column)
                continue

            # Non-revolving payment row: value shapes alternate exactly with
            # the four canonical roles after the split label is restored.
            if all(
                marker in header
                for marker in (
                    "最近一次",
                    "本月应还款",
                    "应还款日",
                    "本月实还款",
                    "还款日期",
                )
            ):
                if len(tokens) == 4:
                    first_date, first_amount, second_date, second_amount = tokens
                    parsed = (
                        _date(first_date),
                        _number(first_amount),
                        _date(second_date),
                        _number(second_amount),
                    )
                    if (
                        isinstance(parsed[1], int)
                        and isinstance(parsed[3], int)
                        and parsed[0]
                        and parsed[2]
                    ):
                        for field_name, value, raw in (
                            ("last_repayment_date", parsed[0], first_date),
                            ("scheduled_payment", parsed[1], first_amount),
                            ("scheduled_payment_date", parsed[2], second_date),
                            ("actual_payment", parsed[3], second_amount),
                        ):
                            bind(
                                field_name,
                                value,
                                raw,
                                header_index + 1,
                                value_column,
                            )
                continue

            # Card payment rows can be merged in a different traversal.  The
            # dates are distinguished by the source-visible snapshot date.
            if all(
                marker in header
                for marker in (
                    "当前逾期期数",
                    "最近一次还款日期",
                    "当前逾期总额",
                    "账单日",
                    "本月应还款",
                    "本月实还款",
                )
            ):
                dates = [(token, _date(token)) for token in tokens if _date(token)]
                numbers = [
                    (token, _number(token))
                    for token in tokens
                    if isinstance(_number(token), int)
                ]
                snapshot = str(account.get("snapshot_date") or "")
                billing = [item for item in dates if item[1] == snapshot]
                prior_dates = [item for item in dates if item[1] != snapshot]
                nonzero = [item for item in numbers if int(item[1]) > 0]
                zeros = [item for item in numbers if int(item[1]) == 0]
                if (
                    len(billing) == 1
                    and len(prior_dates) == 1
                    and len(nonzero) == 2
                    and nonzero[0][1] == nonzero[1][1]
                    and len(zeros) == 2
                ):
                    for field_name, value, raw in (
                        ("billing_date", billing[0][1], billing[0][0]),
                        ("last_repayment_date", prior_dates[0][1], prior_dates[0][0]),
                        ("scheduled_payment", int(nonzero[0][1]), nonzero[0][0]),
                        ("actual_payment", int(nonzero[1][1]), nonzero[1][0]),
                        ("current_overdue_periods", 0, zeros[0][0]),
                        ("current_overdue_amount", 0, zeros[1][0]),
                    ):
                        bind(
                            field_name,
                            value,
                            raw,
                            header_index + 1,
                            value_column,
                        )


def _apply_account_facts(
    parse_result: Any,
    account: dict[str, Any],
    rows: list[list[str]],
    *,
    page: Any,
    table: Any,
    physical_row_indices: list[int | None] | None = None,
    defer_trailing_labels: bool = True,
) -> None:
    observations, unresolved = _exact_label_observations(rows)
    mappings: tuple[tuple[str, tuple[str, ...], Any], ...] = (
        ("management_institution", ("管理机构", "发卡机构"), _account_institution),
        ("account_identifier", ("账户标识",), _identifier),
        ("open_date", ("开立日期",), _date),
        ("due_date", ("到期日期",), _date),
        ("loan_amount", ("借款金额",), _number),
        ("credit_limit", ("账户授信额度", "账户授信额度"), _number),
        ("shared_credit_limit", ("共享授信额度",), _number),
        ("business_type", ("业务种类",), _business_text),
        ("guarantee_type", ("担保方式",), _business_text),
        ("repayment_periods", ("还款期数",), _number),
        ("repayment_frequency", ("还款频率",), _business_text),
        ("repayment_method", ("还款方式",), _business_text),
        ("co_borrower_flag", ("共同借款标志",), _business_text),
        ("balance", ("余额", "透支余额"), _number),
        ("remaining_periods", ("剩余还款期数", "剩余分期期数"), _number),
        ("scheduled_payment", ("本月应还款",), _number),
        ("actual_payment", ("本月实还款",), _number),
        ("scheduled_payment_date", ("应还款日",), _date),
        ("billing_date", ("账单日",), _date),
        ("last_repayment_date", ("最近一次还款日期",), _date),
        ("current_overdue_periods", ("当前逾期期数",), _number),
        ("current_overdue_amount", ("当前逾期总额",), _number),
        ("overdue_principal_31_60", ("逾期31—60天未还本金", "逾期31－60天未还本金"), _number),
        ("overdue_principal_61_90", ("逾期61—90天未还本金", "逾期61－90天未还本金"), _number),
        ("overdue_principal_91_180", ("逾期91—180天未还本金", "逾期91－180天未还本金"), _number),
        ("overdue_principal_over_180", ("逾期180天以上未还本金", "透支180天以上未付余额"), _number),
        ("used_amount", ("已用额度",), _number),
        ("recent_6_month_average_used_amount", ("最近6个月平均使用额度",), _number),
        ("maximum_used_amount", ("最大使用额度",), _number),
        ("recent_6_month_average_overdraft_balance", ("最近6个月平均透支余额",), _number),
        ("maximum_overdraft_balance", ("最大透支余额",), _number),
        ("unbilled_installment_balance", ("未出单的大额专项分期余额",), _number),
        ("five_tier_class", ("五级分类",), _clean),
        ("close_date", ("账户关闭日期", "销户日期"), _date),
        ("transfer_out_date", ("转出月份",), _date),
    )
    dataset = "credit_accounts"
    target_record_id = str(account.get("account_id") or "")
    parser_stage = "candidate_b_account_canonical_slots"

    def physical_row(row_index: int) -> int | None:
        if physical_row_indices is None:
            return row_index
        if 0 <= row_index < len(physical_row_indices):
            return physical_row_indices[row_index]
        return None

    for target, labels, converter in mappings:
        values = [item for label in labels for item in observations.get(_compact(label), ())]
        for raw, value_row_index, column in values:
            source_row = physical_row(value_row_index)
            if source_row is None:
                continue
            ref = _source_ref(page, table, row=source_row, column=column)
            if is_explicit_source_absence(raw):
                _mark_source_absent(account, target, raw)
                continue
            if target == "management_institution":
                _merge_account_institution_observation(
                    parse_result,
                    account,
                    raw=raw,
                    source_ref=ref,
                    parser_stage=parser_stage,
                )
                continue
            if converter is _date:
                _merge_canonical_date_observation(
                    parse_result,
                    account,
                    dataset=dataset,
                    target_record_id=target_record_id,
                    field_name=target,
                    raw=raw,
                    source_ref=ref,
                    parser_stage=parser_stage,
                )
                continue
            value = converter(raw)
            valid = value not in (None, "")
            if converter is _number:
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif converter in {_business_text, _account_institution}:
                compact_value = _compact(value)
                valid = bool(value) and not any(label in compact_value for label in _ACCOUNT_LABELS)
                if target == "management_institution":
                    valid = valid and not bool(
                        _DATE_RE.search(compact_value)
                        or re.search(r"[A-Z][A-Z0-9-]{7,}\d", compact_value, re.IGNORECASE)
                    )
                contract_role = _ACCOUNT_SCALAR_CONTRACT_ROLES.get(target)
                if contract_role:
                    value = normalize_pboc_field(str(value), contract_role)
                    valid = valid and validate_pboc_field(str(value), contract_role).valid
            elif target == "account_identifier":
                typed_value = _typed_identifier(value)
                valid = typed_value is not None
                value = typed_value
            if not valid:
                _reject_exact_observation(
                    parse_result,
                    account,
                    dataset=dataset,
                    target_record_id=target_record_id,
                    field_name=target,
                    raw=raw,
                    source_ref=ref,
                    parser_stage=parser_stage,
                )
                continue
            _merge_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=target,
                value=value,
                raw=raw,
                source_ref=ref,
                parser_stage=parser_stage,
            )

    for raw_currency, value_row_index, column in [
        item for label in ("账户币种", "币种") for item in observations.get(label, ())
    ]:
        source_row = physical_row(value_row_index)
        if source_row is None:
            continue
        ref = _source_ref(page, table, row=source_row, column=column)
        if _compact(raw_currency) in {"-", "--"}:
            _mark_source_absent(account, "currency", raw_currency)
            _mark_source_absent(account, "account_currency", raw_currency)
            continue
        currency, residue, resolution = _currency_token(raw_currency)
        if not currency:
            _reject_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name="currency",
                raw=raw_currency,
                source_ref=ref,
                parser_stage=parser_stage,
            )
            continue
        for field_name in ("currency", "account_currency"):
            _merge_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=field_name,
                value=currency,
                raw=raw_currency,
                source_ref=ref,
                parser_stage=parser_stage,
            )
        if resolution == "residue":
            _report_currency_residue(
                parse_result,
                dataset=dataset,
                target_record_id=target_record_id,
                raw=raw_currency,
                currency=currency,
                residue=residue,
                source_refs=(ref,),
                parser_stage=parser_stage,
            )

    for raw_status, value_row_index, column in [
        item for label in ("账户状态", "状态") for item in observations.get(label, ())
    ]:
        source_row = physical_row(value_row_index)
        if source_row is None:
            continue
        ref = _source_ref(page, table, row=source_row, column=column)
        if _compact(raw_status) in {"-", "--"}:
            _mark_source_absent(account, "account_status", raw_status)
            continue
        status_values = _status_fields(raw_status)
        if status_values.get("account_status_resolution") == "unresolved":
            _reject_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name="account_status",
                raw=raw_status,
                source_ref=ref,
                parser_stage=parser_stage,
            )
            continue
        for field_name, value in status_values.items():
            _merge_exact_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=field_name,
                value=value,
                raw=raw_status,
                source_ref=ref,
                parser_stage=parser_stage,
            )

    terminal_block = _canonical_account_terminal_subtable(rows)
    if terminal_block is not None:
        status_source_row = physical_row(int(terminal_block["status_row"]))
        close_source_row = physical_row(int(terminal_block["close_row"]))
        if status_source_row is not None and close_source_row is not None:
            status_ref = _source_ref(
                page,
                table,
                row=status_source_row,
                column=int(terminal_block["status_column"]),
            )
            close_ref = _source_ref(
                page,
                table,
                row=close_source_row,
                column=int(terminal_block["close_column"]),
            )
            for ref in (status_ref, close_ref):
                ref["binding"] = "canonical_account_terminal_subtable"
                ref["binding_quality"] = "canonical_account_terminal_subtable"
            for field_name, value in terminal_block["status_values"].items():
                _merge_exact_observation(
                    parse_result,
                    account,
                    dataset=dataset,
                    target_record_id=target_record_id,
                    field_name=field_name,
                    value=value,
                    raw=str(terminal_block["status_raw"]),
                    source_ref=status_ref,
                    parser_stage="candidate_b_account_terminal_subtable",
                )
            _merge_canonical_date_observation(
                parse_result,
                account,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name="close_date",
                raw=str(terminal_block["close_raw"]),
                source_ref=close_ref,
                parser_stage="candidate_b_account_terminal_subtable",
            )

    # Do not report the final label row yet: it may be a verified continuation
    # whose value row begins the next logical page.  Internal missing cells are
    # immediately repair-eligible.
    for label, label_row_index, column in unresolved:
        if defer_trailing_labels and label_row_index == len(rows) - 1:
            continue
        target = next((field for field, labels, _converter in mappings if label in labels), None)
        if target is None:
            target = {"账户币种": "currency", "币种": "currency", "账户状态": "account_status", "状态": "account_status"}.get(label)
        source_row = physical_row(label_row_index)
        if not target or source_row is None:
            continue
        _reject_exact_observation(
            parse_result,
            account,
            dataset=dataset,
            target_record_id=target_record_id,
            field_name=target,
            raw="",
            source_ref=_source_ref(page, table, row=source_row, column=column),
            parser_stage=parser_stage,
        )

    for row in rows:
        text = _clean(" ".join(row))
        as_of = _AS_OF_RE.search(text)
        if as_of:
            account["snapshot_date"] = f"{int(as_of.group(1)):04d}-{int(as_of.group(2)):02d}-{int(as_of.group(3)):02d}"
        inactive = re.search(r"账户状态为[“\"]([^”\"]+)", text)
        if inactive:
            account.update(_status_fields(inactive.group(1)))

    _apply_collapsed_account_clusters(
        parse_result,
        account,
        rows,
        page=page,
        table=table,
        physical_row_indices=physical_row_indices,
    )

    if account.get("due_date"):
        account["contract_maturity_date"] = account["due_date"]
    if account.get("currency"):
        account["reporting_amount_currency"] = account["currency"]
    account.setdefault("amount_unit", "yuan")
    account.setdefault("reporting_amount_unit", "yuan")
    account.setdefault("reporting_amount_precision", 0)


def _repayment_records(
    page: Any,
    table: Any,
    rows: list[list[str]],
    account: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = account.setdefault("_repayment_context", {})
    observed_months = _month_centers(table, rows)
    if observed_months:
        context["month_centers"] = observed_months
    months = dict(context.get("month_centers") or {})
    width = max((len(row) for row in rows), default=0)
    centers = _column_centers(table, width)

    def amount_values(row: list[str]) -> dict[int, str]:
        values: dict[int, str] = {}
        for column, cell in enumerate(row):
            raw = _compact(cell)
            if not raw or not re.fullmatch(r"(?:--|-?\d[\d,]*)", raw):
                continue
            month = _nearest_month(column, centers, months)
            if month is not None:
                values[month] = raw
        return values

    def make_record(
        *,
        year: int,
        month: int,
        status: str,
        amount_raw: str = "",
        row_index: int,
        source_refs: list[dict[str, Any]] | None = None,
        confidence: float = 1.0,
        inferred: bool = False,
    ) -> dict[str, Any]:
        account_id = str(account["account_id"])
        return {
            "repayment_id": stable_record_id("credit_repayment", account_id, year, month),
            "account_id": account_id,
            "_table_observation_instance_id": account.get(
                "_table_observation_instance_id"
            ),
            "account_identifier": account.get("account_identifier"),
            "grid_id": str(getattr(table, "table_id", "") or ""),
            "year": year,
            "month": month,
            "status": status,
            "overdue_amount": None if amount_raw in {"", "--"} else amount_raw.replace(",", ""),
            "source": "native_detail_repayment_grid_gap_recovery" if inferred else "native_detail_repayment_grid",
            "source_refs": source_refs or [_source_ref(page, table, row=row_index)],
            "confidence": confidence,
            **({"status_inferred_from_adjacent_months": True} if inferred else {}),
        }

    records: list[dict[str, Any]] = []
    pending = context.pop("pending_status", None)
    if isinstance(pending, dict) and rows and months:
        first_values = amount_values(rows[0])
        if first_values:
            for month, status in pending.get("statuses", {}).items():
                records.append(
                    make_record(
                        year=int(pending["year"]),
                        month=int(month),
                        status=str(status),
                        amount_raw=first_values.get(int(month), ""),
                        row_index=0,
                        source_refs=[
                            *list(pending.get("source_refs") or []),
                            _source_ref(page, table, row=0),
                        ],
                    )
                )

    for row_index, row in enumerate(rows):
        year_cell = next(
            ((index, _compact(cell)) for index, cell in enumerate(row) if re.fullmatch(r"20\d{2}", _compact(cell))),
            None,
        )
        if year_cell is None:
            continue
        year_col, year_raw = year_cell
        status_cells = [
            (index, _compact(cell))
            for index, cell in enumerate(row)
            if index != year_col and _compact(cell) in _STATUS_CODES
        ]
        if not status_cells:
            continue
        statuses: dict[int, str] = {}
        for column, status in status_cells:
            month = _nearest_month(column, centers, months)
            if month is not None:
                statuses[month] = status
        if not statuses:
            continue

        amount_row_index = row_index + 1
        amounts = (
            amount_values(rows[amount_row_index])
            if amount_row_index < len(rows)
            and not any(re.fullmatch(r"20\d{2}", _compact(cell)) for cell in rows[amount_row_index])
            else {}
        )
        # A native grid can occasionally omit one status glyph while keeping
        # the paired amount. Recover only a sandwiched value whose two adjacent
        # statuses agree; this is deterministic and intentionally conservative.
        inferred_months: set[int] = set()
        for month in sorted(amounts):
            if month in statuses or month <= 1 or month >= 12:
                continue
            previous = statuses.get(month - 1)
            following = statuses.get(month + 1)
            if previous == following and previous in {"1", "2", "3", "4", "5", "6", "7"}:
                statuses[month] = str(previous)
                inferred_months.add(month)

        if amounts or amount_row_index < len(rows):
            for month, status in sorted(statuses.items()):
                records.append(
                    make_record(
                        year=int(year_raw),
                        month=month,
                        status=status,
                        amount_raw=amounts.get(month, ""),
                        row_index=row_index,
                        confidence=0.8 if month in inferred_months else 1.0,
                        inferred=month in inferred_months,
                    )
                )
        else:
            context["pending_status"] = {
                "year": int(year_raw),
                "statuses": statuses,
                "source_refs": [_source_ref(page, table, row=row_index)],
            }
    return records, context


def _account_base(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows[:4] for cell in row))
    return "账户标识" in compact and ("管理机构" in compact or "发卡机构" in compact) and not _other_entity_table(rows)


def _other_entity_table(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows[:4] for cell in row))
    return any(
        marker in compact
        for marker in (
            "保证合同编号",
            "授信协议标识",
            "机构名称业务类型业务开通日期",
            "主管税务机关",
            "立案法院",
            "执行法院",
            "处罚机构",
            "查询日期查询机构查询原因",
        )
    )


def _account_heading_fields(value: Any) -> dict[str, str]:
    """Decode business identifiers printed in one canonical account anchor."""

    text = _compact(value)
    tail = re.search(r"卡片尾号[：:]?([0-9]{4})", text)
    agreement = re.search(r"授信协议标识[：:]?([A-Z0-9]{8,80})", text, re.IGNORECASE)
    return {
        **({"card_tail": tail.group(1)} if tail else {}),
        **({"credit_agreement_identifier": agreement.group(1).upper()} if agreement else {}),
    }


def _account_heading_for_table(page: Any, table: Any) -> dict[str, str]:
    bbox = getattr(table, "bbox", None) or []
    table_top = float(bbox[1]) if len(bbox) == 4 else float("inf")
    candidates: list[tuple[float, str]] = []
    for text_block in getattr(page, "texts", None) or []:
        text = _clean(getattr(text_block, "content", "") or getattr(text_block, "text", "") or "")
        compact_text = _compact(text)
        if not re.match(r"^(?:账户|业务)\s*\d{1,3}", compact_text):
            continue
        text_bbox = getattr(text_block, "bbox", None) or []
        bottom = float(text_bbox[3]) if len(text_bbox) == 4 else 0.0
        if bottom <= table_top + 1.0:
            candidates.append((bottom, text))
    if not candidates:
        return {}
    bottom, raw_text = max(candidates, key=lambda item: item[0])
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    maximum_gap = max(72.0, page_height * 0.10) if page_height else 72.0
    if table_top != float("inf") and table_top - bottom > maximum_gap:
        return {}
    return _account_heading_fields(raw_text)


_ACCOUNT_CONTINUATION_LABELS = frozenset(
    {
        "账户状态",
        "状态",
        "五级分类",
        "余额",
        "透支余额",
        "剩余还款期数",
        "剩余分期期数",
        "本月应还款",
        "本月实还款",
        "应还款日",
        "账单日",
        "最近一次还款日期",
        "当前逾期期数",
        "当前逾期总额",
        "逾期31—60天未还本金",
        "逾期31－60天未还本金",
        "逾期61—90天未还本金",
        "逾期61－90天未还本金",
        "逾期91—180天未还本金",
        "逾期91－180天未还本金",
        "逾期180天以上未还本金",
        "透支180天以上未付余额",
        "已用额度",
        "最近6个月平均使用额度",
        "最大使用额度",
        "未出单的大额专项分期余额",
        "大额专项分期额度",
        "分期额度生效日期",
        "分期额度到期日期",
        "已用分期金额",
        "还款日期",
        "还款金额",
        "当前还款状态",
        "特殊交易类型",
        "特殊事件说明",
    }
)


def _account_continuation_fragment(rows: list[list[str]], pending_labels: list[str] | None = None) -> bool:
    """Recognize a canonical account-detail fragment, not another account body."""

    compact = _compact(" ".join(cell for row in rows for cell in row))
    if not compact:
        return False
    if any(_compact(label) in compact for label in _ACCOUNT_CONTINUATION_LABELS):
        return True
    if "还款记录" in compact or "逾期金额" in compact:
        return True
    # A value-only table is eligible only when the immediately preceding
    # account fragment ended with an exact canonical label row.
    return bool(pending_labels and _label_row(pending_labels))


def _table_top_value(table: Any) -> float | None:
    bbox = getattr(table, "bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return float(bbox[1])
    except (TypeError, ValueError):
        return None


def _account_reading_order_resolution(
    parse_result: Any,
    pages: Iterable[Any],
) -> tuple[dict[int, int], list[int], list[int], list[int], bool]:
    """Validate the registered order for this exact account-page population."""

    materialized = list(pages)
    raw_order = getattr(parse_result, "reading_order_by_logical", None)
    registered_resolution = getattr(parse_result, "reading_order_resolution", None)
    legacy_sealed = not hasattr(
        parse_result, "reading_order_by_logical"
    ) and not hasattr(parse_result, "reading_order_resolution")
    provenance_resolved = legacy_sealed or not hasattr(
        parse_result, "reading_order_resolution"
    ) or bool(
        isinstance(registered_resolution, Mapping)
        and registered_resolution.get("resolved") is True
        and registered_resolution.get("authoritative") is True
    )
    normalized_order: dict[int, int] = {}
    if isinstance(raw_order, Mapping):
        for raw_logical, raw_position in raw_order.items():
            try:
                logical = int(raw_logical)
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            if logical > 0 and position > 0:
                normalized_order[logical] = position

    logical_pages: list[int] = []
    for page in materialized:
        raw_logical = (
            page.get("page")
            if isinstance(page, Mapping)
            else getattr(page, "page_number", 0)
        )
        try:
            logical_pages.append(int(raw_logical or 0))
        except (TypeError, ValueError):
            logical_pages.append(0)
    missing = sorted(
        {
            logical
            for logical in logical_pages
            if logical <= 0 or logical not in normalized_order
        }
    )
    observed_positions = [
        normalized_order[logical]
        for logical in logical_pages
        if logical > 0 and logical in normalized_order
    ]
    duplicate_positions = sorted(
        position
        for position, count in Counter(observed_positions).items()
        if count > 1
    )
    registered_pages = list(getattr(parse_result, "pages", None) or [])
    if registered_pages and registered_pages is not materialized:
        registered_logicals: list[int] = []
        for page in registered_pages:
            try:
                registered_logicals.append(int(getattr(page, "page_number", 0) or 0))
            except (TypeError, ValueError):
                registered_logicals.append(0)
        missing = sorted(
            set(missing)
            | {
                logical
                for logical in registered_logicals
                if logical <= 0 or logical not in normalized_order
            }
        )
        registered_positions = [
            normalized_order[logical]
            for logical in registered_logicals
            if logical > 0 and logical in normalized_order
        ]
        duplicate_positions = sorted(
            set(duplicate_positions)
            | {
                position
                for position, count in Counter(registered_positions).items()
                if count > 1
            }
        )
    # Contexts created before the reading-order plane existed have always
    # treated their sealed page list as authoritative.  Preserve that explicit
    # legacy contract.  Once a context exposes the plane, however, empty,
    # partial, or non-unique registration is unresolved and must fail closed.
    resolved = legacy_sealed or bool(
        isinstance(raw_order, Mapping)
        and normalized_order
        and not missing
        and not duplicate_positions
        and provenance_resolved
    )
    return normalized_order, logical_pages, missing, duplicate_positions, resolved


def _account_ordered_pages(parse_result: Any, pages: Iterable[Any]) -> list[Any]:
    """Return pages in the document's registered reading order.

    Logical page numbers remain the provenance coordinate.  Account-family
    state, however, must follow the already-registered printed reading order:
    some scanned PDFs store two physical halves in a different logical order.
    A partial or non-unique order is not repaired here; that case falls back to
    the sealed input order and is reported once for downstream review.
    """

    materialized = list(pages)
    if len(materialized) < 2:
        return materialized

    raw_order = getattr(parse_result, "reading_order_by_logical", None)
    registered_resolution = getattr(parse_result, "reading_order_resolution", None)
    # Small unit/legacy contexts created before the reading-order plane existed
    # retain their sealed order without manufacturing a diagnostic.  A real
    # context that exposes an empty/invalid plane is explicitly auditable.
    if raw_order is None and not hasattr(
        parse_result, "reading_order_by_logical"
    ) and not hasattr(parse_result, "reading_order_resolution"):
        return materialized
    (
        normalized_order,
        logical_pages,
        missing,
        duplicate_positions,
        resolved,
    ) = _account_reading_order_resolution(parse_result, materialized)
    if not resolved:
        if not getattr(parse_result, "_candidate_b_account_reading_order_issue", False):
            try:
                setattr(parse_result, "_candidate_b_account_reading_order_issue", True)
            except (AttributeError, TypeError):
                pass
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_account_reading_order_unresolved",
                    message=(
                        "The document-local account reading-order plane was missing or non-unique; "
                        "sealed page order was retained and no cross-page ownership was inferred from it."
                    ),
                    parser_stage="candidate_b_account_page_ownership",
                    target_dataset="credit_accounts",
                    observed_value={
                        "logical_pages": logical_pages,
                        "registered_reading_order": normalized_order,
                        "reading_order_resolution": (
                            dict(registered_resolution)
                            if isinstance(registered_resolution, Mapping)
                            else None
                        ),
                        "missing_logical_pages": missing,
                        "duplicate_reading_positions": duplicate_positions,
                    },
                    source_refs=tuple(
                        {
                            "source": "candidate_b_account_page_order",
                            "logical_page": logical,
                        }
                        for logical in logical_pages
                        if logical > 0
                    ),
                    reason_codes=(
                        "document_local_reading_order_unresolved",
                        "sealed_page_order_retained",
                        "cross_page_account_ownership_not_inferred",
                    ),
                ),
            )
        return materialized

    return [
        page
        for _index, page in sorted(
            enumerate(materialized),
            key=lambda item: (
                normalized_order[logical_pages[item[0]]],
                item[0],
            ),
        )
    ]


def _geometric_prior_account_continuation(
    *,
    parse_result: Any,
    page: Any,
    table: Any,
    page_tables: list[Any],
    current_logical_page: int,
    current_table_top: float | None,
    rows: list[list[str]],
    pending_labels: list[str] | None,
    cross_page_order_resolved: bool,
) -> bool:
    """Authorize the unique prior-account interval around a split table.

    On the same logical page, a secondary account table below the current body
    remains inside that body's interval.  At a logical-page boundary, a
    secondary table can belong to the prior account only when it is
    geometrically above the next canonical account body.  Page adjacency or a
    footer alone is never sufficient.
    """

    if not _account_continuation_fragment(rows, pending_labels):
        return False
    top = _table_top_value(table)
    logical_page = int(getattr(page, "page_number", 0) or 0)
    if top is None or not logical_page or not current_logical_page:
        return False
    if logical_page == current_logical_page:
        return current_table_top is not None and top + 1.0 >= current_table_top
    if not (
        cross_page_order_resolved
        and _registered_account_pages_are_adjacent(
            parse_result,
            current_logical_page,
            logical_page,
        )
    ):
        return False
    next_body_tops = [
        candidate_top
        for candidate in page_tables
        if candidate is not table
        and _account_base(_table_rows(candidate))
        and (candidate_top := _table_top_value(candidate)) is not None
        and candidate_top > top
    ]
    return bool(next_body_tops and top < min(next_body_tops))


_ACCOUNT_EVENT_DATASETS = {
    "special_transaction": "credit_account_special_transactions",
    "large_installment": "credit_card_large_installments",
    "latest_repayment": "credit_account_latest_repayments",
    "special_event_note": "credit_account_special_events",
}


def _account_event_type(row: list[str]) -> str | None:
    compact = _compact("".join(row))
    if "特殊交易类型" in compact:
        return "special_transaction"
    if "大额专项分期额度" in compact:
        return "large_installment"
    if "还款日期" in compact and "还款金额" in compact and "当前还款状态" in compact:
        return "latest_repayment"
    if "特殊事件说明" in compact:
        return "special_event_note"
    return None


def _large_installment_tail_header(row: list[str]) -> bool:
    compact = _compact("".join(row))
    return all(label in compact for label in ("分期额度到期日期", "已用分期金额"))


def _account_event_continuation_value_index(rows: list[list[str]]) -> int | None:
    """Return the first value row that can consume a pending typed event header."""

    for row_index, row in enumerate(rows):
        if not _nonempty(row):
            continue
        if _account_event_type(row) is not None or _label_row(row):
            return None
        return row_index
    return None


def _report_pending_account_event_unresolved(
    issue_owner: Any,
    pending: Mapping[str, Any],
    *,
    boundary: str,
    candidate_table: Any | None = None,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    event_type = str(pending.get("event_type") or "")
    account_id = str(pending.get("account_id") or "")
    page = pending.get("page")
    table = pending.get("table")
    row_index = int(pending.get("row_index") or 0)
    record_issue(
        issue_owner,
        make_issue(
            category="page_continuation",
            issue_code="candidate_b_account_event_continuation_unresolved",
            message=(
                "A canonical account-event header ended an account fragment, but no affirmatively owned "
                "continuation supplied its value row; the event was withheld."
            ),
            parser_stage="candidate_b_account_event_continuation",
            target_dataset=_ACCOUNT_EVENT_DATASETS.get(
                event_type, "personal_detail_account_events"
            ),
            target_record_id=str(
                (pending.get("target_event") or {}).get("account_event_id")
                or stable_record_id(
                    "personal_detail_account_event",
                    account_id,
                    event_type,
                    int(getattr(page, "page_number", 0) or 0),
                    row_index,
                )
            ),
            observed_value={
                "event_type": event_type,
                "header": list(pending.get("row") or ()),
                "boundary": boundary,
                **(
                    {"candidate_table_id": str(getattr(candidate_table, "table_id", "") or "")}
                    if candidate_table is not None
                    else {}
                ),
            },
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=(
                "canonical_event_header_at_fragment_end",
                "continuation_value_not_affirmatively_owned",
                "event_record_withheld",
                "dataset_incomplete",
            ),
        ),
    )


def _account_events(
    issue_owner: Any,
    account: dict[str, Any],
    page: Any,
    table: Any,
    rows: list[list[str]],
    *,
    leading_header: Mapping[str, Any] | None = None,
    leading_value_row_index: int = 0,
    defer_trailing_large_installment_tail: bool = False,
) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    scoped_rows: list[tuple[list[str], Any, Any, int]] = []
    if leading_header is not None:
        scoped_rows.append(
            (
                list(leading_header.get("row") or ()),
                leading_header.get("page"),
                leading_header.get("table"),
                int(leading_header.get("row_index") or 0),
            )
        )
    scoped_rows.extend(
        (row, page, table, row_index)
        for row_index, row in enumerate(rows)
        if leading_header is None or row_index >= leading_value_row_index
    )

    def scoped_ref(index: int, column: int | None = None) -> dict[str, Any]:
        _row, source_page, source_table, source_row = scoped_rows[index]
        return _source_ref(source_page, source_table, row=source_row, column=column)

    events: list[dict[str, Any]] = []
    for row_index, (row, header_page, header_table, header_row_index) in enumerate(scoped_rows):
        event_type = _account_event_type(row)
        if event_type is None or row_index + 1 >= len(scoped_rows):
            continue
        target_dataset = _ACCOUNT_EVENT_DATASETS[event_type]
        value_row = scoped_rows[row_index + 1][0]
        page_number = int(getattr(header_page, "page_number", 0) or 0)
        event_id = stable_record_id(
            "personal_detail_account_event",
            account.get("account_id"),
            event_type,
            page_number,
            header_row_index,
        )
        record: dict[str, Any] = {
            "record_id": event_id,
            "account_event_id": event_id,
            "account_id": account.get("account_id"),
            "_table_observation_instance_id": account.get(
                "_table_observation_instance_id"
            ),
            "event_type": event_type,
            "source": "native_personal_detail_account_event",
            "source_refs": [scoped_ref(row_index)],
            "source_refs_by_field": {},
            "canonical_raw": {},
            "confidence": min(
                float(getattr(header_table, "confidence", None) or 0.9),
                float(getattr(scoped_rows[row_index + 1][2], "confidence", None) or 0.9),
            ),
        }
        observations, unresolved_slots = _exact_label_observations([row, value_row])
        cluster_decoded = None
        cluster_raw = ""
        cluster_value_rows: dict[str, tuple[int, int | None]] = {}
        if event_type in {"special_transaction", "large_installment"}:
            kind = event_type
            fragments = [f"{_clean(' '.join(row))} {_clean(' '.join(value_row))}"]
            first_value_cell = _single_business_cell(value_row)
            for field_name in (
                ("transaction_type", "event_date", "changed_months", "amount", "details")
                if event_type == "special_transaction"
                else ("installment_limit", "effective_date")
            ):
                cluster_value_rows[field_name] = (
                    row_index + 1,
                    first_value_cell[0] if first_value_cell is not None else None,
                )
            if event_type == "large_installment" and row_index + 3 < len(scoped_rows):
                second_header = _compact("".join(scoped_rows[row_index + 2][0]))
                if all(label in second_header for label in ("分期额度到期日期", "已用分期金额")):
                    second_value = scoped_rows[row_index + 3][0]
                    fragments.append(
                        f"{_clean(' '.join(scoped_rows[row_index + 2][0]))} {_clean(' '.join(second_value))}"
                    )
                    second_value_cell = _single_business_cell(second_value)
                    for field_name in ("expiry_date", "used_installment_amount"):
                        cluster_value_rows[field_name] = (
                            row_index + 3,
                            second_value_cell[0] if second_value_cell is not None else None,
                        )
            cluster_decoded = decode_labeled_cluster(
                tuple(fragments),
                kind=kind,
            )
            cluster_raw = " | ".join(fragments)

        def bind_cluster(field_name: str) -> bool:
            if cluster_decoded is None or field_name not in cluster_decoded.fields:
                return False
            value = cluster_decoded.fields[field_name]
            source_row, source_column = cluster_value_rows.get(field_name, (row_index + 1, None))
            source_ref = scoped_ref(source_row, source_column)
            source_ref["binding"] = "closed_canonical_account_event_cluster"
            source_ref["binding_quality"] = "closed_canonical_account_event_cluster"
            record[field_name] = value
            record["canonical_raw"][field_name] = cluster_raw
            record["source_refs_by_field"].setdefault(field_name, []).append(
                {**source_ref, "field_name": field_name}
            )
            return True

        def bind_exact(field_name: str, label: str, converter: Any) -> None:
            candidates = observations.get(label) or []
            distinct = {_compact(raw) for raw, _physical_row, _column in candidates if _compact(raw)}
            if len(distinct) != 1:
                if bind_cluster(field_name):
                    return
                record_issue(
                    issue_owner,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_account_event_slot_unresolved",
                        message=(
                            "A canonical account-event field was missing or ambiguous; no adjacent value "
                            "was shifted into the slot."
                        ),
                        parser_stage="candidate_b_account_event_canonical_slots",
                        target_dataset=target_dataset,
                        target_record_id=event_id,
                        field_name=field_name,
                        observed_value={
                            "label": label,
                            "candidate_values": [raw for raw, _physical_row, _column in candidates],
                        },
                        source_refs=(scoped_ref(row_index),),
                        reason_codes=(
                            "canonical_event_label_observed",
                            "exact_value_slot_unresolved",
                            "positional_fallback_forbidden",
                            "normalized_value_withheld",
                        ),
                    ),
                )
                _append_internal_field(record, "_unresolved_fields", field_name)
                return
            raw, physical_row, column = candidates[0]
            source_ref = scoped_ref(row_index + physical_row, column)
            if _compact(raw) in {"-", "--"}:
                _mark_source_absent(record, field_name, raw)
                record["source_refs_by_field"].setdefault(field_name, []).append(
                    {**source_ref, "field_name": field_name}
                )
                return
            value = converter(raw)
            if value in (None, ""):
                if bind_cluster(field_name):
                    return
                _reject_exact_observation(
                    issue_owner,
                    record,
                    dataset=target_dataset,
                    target_record_id=event_id,
                    field_name=field_name,
                    raw=raw,
                    source_ref=source_ref,
                    parser_stage="candidate_b_account_event_canonical_slots",
                )
                return
            record[field_name] = value
            record["canonical_raw"][field_name] = raw
            record["source_refs_by_field"].setdefault(field_name, []).append(
                {**source_ref, "field_name": field_name}
            )

        field_contracts: dict[str, tuple[tuple[str, str, Any], ...]] = {
            "special_transaction": (
                ("transaction_type", "特殊交易类型", _clean),
                ("event_date", "发生日期", _date),
                ("changed_months", "变更月数", _number),
                ("amount", "发生金额", _number),
                ("details", "明细记录", _clean),
            ),
            "large_installment": (
                ("installment_limit", "大额专项分期额度", _number),
                ("effective_date", "分期额度生效日期", _date),
                ("expiry_date", "分期额度到期日期", _date),
                ("used_installment_amount", "已用分期金额", _number),
            ),
            "latest_repayment": (
                ("five_tier_class", "五级分类", _clean),
                ("balance", "余额", _number),
                ("repayment_date", "还款日期", _date),
                ("repayment_amount", "还款金额", _number),
                ("repayment_status", "当前还款状态", _clean),
            ),
            "special_event_note": (("details", "特殊事件说明", _clean),),
        }
        for field_name, label, converter in field_contracts[event_type]:
            if (
                defer_trailing_large_installment_tail
                and event_type == "large_installment"
                and field_name in {"expiry_date", "used_installment_amount"}
                and _large_installment_tail_header(scoped_rows[-1][0])
            ):
                continue
            bind_exact(field_name, label, converter)
        # Keep the observed row even when one slot failed: the normalized
        # omissions are linked to explicit issues and the source record count
        # remains conserved.
        if unresolved_slots:
            record.setdefault("_unresolved_source_slots", []).extend(
                sorted({label for label, _row, _column in unresolved_slots})
            )
        events.append(record)
    return events


def _consume_pending_large_installment_tail(
    issue_owner: Any,
    pending: Mapping[str, Any],
    *,
    page: Any,
    table: Any,
    rows: list[list[str]],
    value_row_index: int,
) -> None:
    """Bind the second large-installment pair to its already materialized event."""

    event = pending.get("target_event")
    if not isinstance(event, dict):
        _report_pending_account_event_unresolved(
            issue_owner,
            pending,
            boundary="large_installment_tail_without_primary_event",
            candidate_table=table,
        )
        return
    header = list(pending.get("row") or ())
    value_row = rows[value_row_index]
    observations, _unresolved = _exact_label_observations([header, value_row])
    event_id = str(event.get("account_event_id") or event.get("record_id") or "")
    target_dataset = _ACCOUNT_EVENT_DATASETS["large_installment"]
    for field_name, label, converter in (
        ("expiry_date", "分期额度到期日期", _date),
        ("used_installment_amount", "已用分期金额", _number),
    ):
        candidates = observations.get(label) or []
        distinct = {_compact(raw) for raw, _row, _column in candidates if _compact(raw)}
        if len(distinct) != 1:
            _reject_exact_observation(
                issue_owner,
                event,
                dataset=target_dataset,
                target_record_id=event_id,
                field_name=field_name,
                raw="",
                source_ref=_source_ref(
                    pending.get("page"),
                    pending.get("table"),
                    row=int(pending.get("row_index") or 0),
                ),
                parser_stage="candidate_b_account_event_continuation",
            )
            continue
        raw, _physical_row, column = candidates[0]
        value = converter(raw)
        valid = value not in (None, "")
        if converter is _number:
            valid = isinstance(value, int) and not isinstance(value, bool)
        if not valid:
            _reject_exact_observation(
                issue_owner,
                event,
                dataset=target_dataset,
                target_record_id=event_id,
                field_name=field_name,
                raw=raw,
                source_ref=_source_ref(page, table, row=value_row_index, column=column),
                parser_stage="candidate_b_account_event_continuation",
            )
            continue
        event[field_name] = value
        event.setdefault("canonical_raw", {})[field_name] = raw
        value_ref = _source_ref(page, table, row=value_row_index, column=column)
        event.setdefault("source_refs_by_field", {}).setdefault(field_name, []).append(
            {**value_ref, "field_name": field_name}
        )
    for ref in (
        _source_ref(
            pending.get("page"),
            pending.get("table"),
            row=int(pending.get("row_index") or 0),
        ),
        _source_ref(page, table, row=value_row_index),
    ):
        if ref not in event.setdefault("source_refs", []):
            event["source_refs"].append(ref)


def _registered_account_pages_are_adjacent(
    parse_result: Any,
    left_logical_page: int,
    right_logical_page: int,
) -> bool:
    """Require exact adjacency in the already-validated reading-order plane."""

    registered_resolution = getattr(parse_result, "reading_order_resolution", None)
    if hasattr(parse_result, "reading_order_resolution") and not (
        isinstance(registered_resolution, Mapping)
        and registered_resolution.get("resolved") is True
        and registered_resolution.get("authoritative") is True
    ):
        return False
    raw_order = getattr(parse_result, "reading_order_by_logical", None)
    if raw_order is None:
        # Sealed unit/legacy contexts predate the document order plane.  Their
        # only cross-page contract is the immediate logical edge; an explicit
        # plane below always takes the stricter complete/unique path.
        return bool(
            not hasattr(parse_result, "reading_order_by_logical")
            and not hasattr(parse_result, "reading_order_resolution")
            and int(left_logical_page or 0) > 0
            and int(right_logical_page or 0) == int(left_logical_page or 0) + 1
        )
    if not isinstance(raw_order, Mapping):
        return False
    normalized: dict[int, int] = {}
    for raw_logical, raw_position in raw_order.items():
        try:
            logical = int(raw_logical)
            position = int(raw_position)
        except (TypeError, ValueError):
            return False
        if logical <= 0 or position <= 0:
            return False
        normalized[logical] = position
    positions = list(normalized.values())
    if len(positions) != len(set(positions)):
        return False
    left = normalized.get(int(left_logical_page or 0))
    right = normalized.get(int(right_logical_page or 0))
    return bool(left is not None and right == left + 1)


def _bounded_card_credit_limit(value: Any) -> tuple[int, str] | None:
    """Return one money token with at most one trailing Han OCR glyph."""

    raw = _clean(value)
    tokens = list(
        dict.fromkeys(
            re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?![A-Za-z0-9])", raw)
        )
    )
    if len(tokens) != 1:
        return None
    parsed = _number(tokens[0])
    if (
        not isinstance(parsed, int)
        or isinstance(parsed, bool)
        or parsed < 0
    ):
        return None
    residue = _account_cluster_residue(raw, (tokens[0],))
    if residue and re.fullmatch(r"[\u3400-\u9fff]", residue) is None:
        return None
    return parsed, residue


def _bounded_headerless_card_values(
    pending_labels: list[str] | None,
    rows: list[list[str]],
) -> dict[str, Any] | None:
    """Decode the one complete card header followed by one exact value row.

    This is an ownership proof, not a general account decoder. All canonical
    card-header labels must be present once and in template order, and the six
    identity-bearing cells used below must independently satisfy their finite
    contracts. Weak, collapsed, or repayment-only rows therefore fail closed.
    """

    if not pending_labels or not rows:
        return None
    nonempty_columns: list[tuple[int, str]] = [
        (column, _compact(cell))
        for column, cell in enumerate(pending_labels)
        if _compact(cell)
    ]
    roles = [
        _ACCOUNT_BASIC_HEADER_ROLES.get(label)
        for _column, label in nonempty_columns
    ]
    if any(role is None for role in roles):
        return None
    if tuple(roles) != _ACCOUNT_BASIC_CARD_TEMPLATE:
        return None
    columns = {
        str(role): column
        for (column, _label), role in zip(nonempty_columns, roles, strict=True)
    }
    body = rows[0]
    if max(columns.values(), default=-1) >= len(body):
        return None

    raw_institution = _clean(body[columns["management_institution"]])
    raw_identifier = _clean(body[columns["account_identifier"]])
    raw_open_date = _clean(body[columns["open_date"]])
    raw_limit = _clean(body[columns["credit_limit"]])
    raw_currency = _clean(body[columns["currency"]])
    raw_business_type = _clean(body[columns["business_type"]])
    raw_guarantee_type = _clean(body[columns["guarantee_type"]])
    institution = _account_institution(raw_institution)
    identifier = _canonical_pboc_account_identifier(raw_identifier)
    open_date = _date(raw_open_date)
    bounded_limit = _bounded_card_credit_limit(raw_limit)
    credit_limit = bounded_limit[0] if bounded_limit is not None else None
    credit_limit_residue = bounded_limit[1] if bounded_limit is not None else ""
    currency, _currency_residue, currency_resolution = _currency_token(raw_currency)
    business_type = normalize_pboc_field(
        _business_text(raw_business_type),
        "account_business_type",
    )
    guarantee_type = normalize_pboc_field(
        _business_text(raw_guarantee_type),
        "guarantee_type",
    )
    if (
        not institution
        or not identifier
        or open_date is None
        or not isinstance(credit_limit, int)
        or isinstance(credit_limit, bool)
        or credit_limit < 0
        or currency is None
        or currency_resolution != "exact"
        or not business_type
        or not validate_pboc_field(business_type, "account_business_type").valid
        or not guarantee_type
        or not validate_pboc_field(guarantee_type, "guarantee_type").valid
    ):
        return None
    return {
        "management_institution": institution,
        "account_identifier": identifier,
        "open_date": open_date,
        "credit_limit": credit_limit,
        "currency": currency,
        "business_type": business_type,
        "guarantee_type": guarantee_type,
        "_credit_limit_residue": credit_limit_residue,
    }


def _exact_anchor_evidence_card_header(
    skeleton: Mapping[str, Any],
    *,
    table: Any,
    rows: list[list[str]],
    prior_logical_page: int,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """Project one exact post-anchor card header onto the next table lattice."""

    if not rows or not rows[0]:
        return None
    anchor_bbox = _exact_geometry_bbox(skeleton.get("bbox"))
    if anchor_bbox is None:
        return None
    header_lines: list[tuple[dict[str, Any], str, tuple[float, float, float, float]]] = []
    for raw_line in skeleton.get("raw_detail_lines") or ():
        if not isinstance(raw_line, Mapping):
            continue
        if int(raw_line.get("logical_page") or 0) != int(prior_logical_page or 0):
            continue
        role = _ACCOUNT_BASIC_HEADER_ROLES.get(_compact(raw_line.get("text") or ""))
        bbox = _exact_geometry_bbox(raw_line.get("bbox"))
        evidence_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in raw_line.get("evidence_ids") or ()
                if str(value or "")
            )
        )
        if (
            role not in _ACCOUNT_BASIC_CARD_TEMPLATE
            or bbox is None
            or not evidence_ids
            or bbox[1] + 1.0 < anchor_bbox[3]
            or bbox[1] > anchor_bbox[3] + 36.0
        ):
            continue
        header_lines.append((dict(raw_line), str(role), bbox))
    roles = [role for _line, role, _bbox in header_lines]
    if (
        len(header_lines) != len(_ACCOUNT_BASIC_CARD_TEMPLATE)
        or tuple(sorted(roles)) != tuple(sorted(_ACCOUNT_BASIC_CARD_TEMPLATE))
        or any(roles.count(role) != 1 for role in _ACCOUNT_BASIC_CARD_TEMPLATE)
    ):
        return None
    ordered = sorted(header_lines, key=lambda item: (item[2][0] + item[2][2]) / 2.0)
    if tuple(item[1] for item in ordered) != _ACCOUNT_BASIC_CARD_TEMPLATE:
        return None
    centers_y = [(item[2][1] + item[2][3]) / 2.0 for item in ordered]
    if max(centers_y) - min(centers_y) > 18.0:
        return None

    geometry = _native_table_geometry(table)
    if geometry is None:
        return None
    column_bands = _indexed_geometry_bands(
        geometry,
        "col_bands",
        lower_key="x0",
        upper_key="x1",
    )
    width = len(rows[0])
    if column_bands is None or set(column_bands) != set(range(width)):
        return None
    labels = ["" for _column in range(width)]
    refs: list[dict[str, Any]] = []
    for line, role, bbox in ordered:
        center = (bbox[0] + bbox[2]) / 2.0
        matching_columns = [
            column
            for column, (left, right) in column_bands.items()
            if left <= center <= right
        ]
        if len(matching_columns) != 1:
            return None
        column = matching_columns[0]
        if labels[column]:
            return None
        labels[column] = _compact(line.get("text") or "")
        raw_value = _clean(rows[0][column] if column < len(rows[0]) else "")
        if role != "shared_credit_limit":
            if not raw_value or _exact_native_table_cell(table, row=0, column=column) is None:
                return None
        refs.append(
            {
                "source": "candidate_b_account_anchor_header",
                "logical_page": int(prior_logical_page or 0),
                "source_page": int(line.get("source_page") or 0),
                "bbox": list(bbox),
                "evidence_ids": list(line.get("evidence_ids") or ()),
                "field_name": role,
                "binding": "printed_anchor_exact_card_header_lattice",
            }
        )
    return labels, refs


def _bounded_headerless_card_owner(
    parse_result: Any,
    *,
    current: Mapping[str, Any],
    prior_logical_page: int,
    prior_table_top: float | None,
    candidate_page: Any,
    candidate_table: Any,
    source_table_index: int,
    pending_labels: list[str] | None,
    rows: list[list[str]],
    prior_accounts: Iterable[Mapping[str, Any]],
    cross_page_order_resolved: bool,
    ) -> dict[str, Any] | None:
    """Resolve a headerless top-of-page card only from a pending exact anchor."""

    if not cross_page_order_resolved:
        return None
    if str(current.get("account_type") or "") not in {"credit_card", "quasi_credit_card"}:
        return None
    if source_table_index != 0 or prior_table_top is None:
        return None
    candidate_logical_page = int(getattr(candidate_page, "page_number", 0) or 0)
    if not _registered_account_pages_are_adjacent(
        parse_result,
        prior_logical_page,
        candidate_logical_page,
    ):
        return None
    prior_identifiers = {
        identifier
        for record in prior_accounts
        if (
            identifier := _canonical_pboc_account_identifier(
                record.get("account_identifier")
            )
        )
    }
    candidates: list[dict[str, Any]] = []
    for skeleton in _account_anchor_skeletons(parse_result):
        if str(skeleton.get("account_type") or "") not in {
            "credit_card",
            "quasi_credit_card",
        }:
            continue
        if skeleton.get("_printed_ordinal_status") != "printed_unique":
            continue
        if int(skeleton.get("page") or 0) != int(prior_logical_page or 0):
            continue
        bbox = skeleton.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            anchor_top = float(bbox[1])
        except (TypeError, ValueError):
            continue
        if anchor_top <= prior_table_top + 1.0:
            continue
        header_labels = list(pending_labels) if pending_labels is not None else None
        header_refs: list[dict[str, Any]] = []
        ownership_basis = "pending_table_header_row"
        strong_values = _bounded_headerless_card_values(header_labels, rows)
        if strong_values is None:
            projected = _exact_anchor_evidence_card_header(
                skeleton,
                table=candidate_table,
                rows=rows,
                prior_logical_page=prior_logical_page,
            )
            if projected is None:
                continue
            header_labels, header_refs = projected
            strong_values = _bounded_headerless_card_values(header_labels, rows)
            ownership_basis = "printed_anchor_header_lattice"
        if strong_values is None or header_labels is None:
            continue
        strong_identifier = _canonical_pboc_account_identifier(
            strong_values.get("account_identifier")
        )
        if strong_identifier in prior_identifiers:
            continue
        anchor_identifier = _account_card_identifier(skeleton)
        if anchor_identifier and anchor_identifier != strong_identifier:
            continue
        candidates.append(
            {
                "skeleton": skeleton,
                "strong_values": strong_values,
                "header_labels": header_labels,
                "header_refs": header_refs,
                "ownership_basis": ownership_basis,
            }
        )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _extract_table_accounts(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accounts: list[dict[str, Any]] = []
    repayments: list[dict[str, Any]] = []
    account_events: list[dict[str, Any]] = []
    phase = "non_revolving_loan"
    current: dict[str, Any] | None = None
    pending_labels: list[str] | None = None
    pending_label_source: tuple[Any, Any, int] | None = None
    pending_event: dict[str, Any] | None = None
    current_table_id = ""
    current_logical_page = 0
    current_table_top: float | None = None
    phase_logical_page = 0
    continuation_check = getattr(parse_result, "tables_continue", None)

    def flush_pending_fact_labels(boundary: str) -> None:
        nonlocal pending_labels, pending_label_source
        if current is None or pending_labels is None or pending_label_source is None:
            pending_labels = None
            pending_label_source = None
            return
        source_page, source_table, source_row_index = pending_label_source
        _apply_account_facts(
            parse_result,
            current,
            [pending_labels, []],
            page=source_page,
            table=source_table,
            physical_row_indices=[source_row_index, None],
            defer_trailing_labels=False,
        )
        current.setdefault("_terminal_unresolved_fact_boundaries", []).append(boundary)
        pending_labels = None
        pending_label_source = None

    def flush_pending_event(
        boundary: str,
        *,
        candidate_table: Any | None = None,
    ) -> None:
        nonlocal pending_event
        if pending_event is None:
            return
        _report_pending_account_event_unresolved(
            parse_result,
            pending_event,
            boundary=boundary,
            candidate_table=candidate_table,
        )
        pending_event = None

    def flush_pending_institution(boundary: str) -> None:
        if current is not None:
            _flush_pending_account_institution_observations(
                parse_result,
                current,
                boundary=boundary,
            )

    def set_trailing_pending_state(page: Any, table: Any, rows: list[list[str]]) -> None:
        nonlocal pending_labels, pending_label_source, pending_event
        trailing_event_type = _account_event_type(rows[-1]) if rows else None
        if trailing_event_type is not None:
            pending_event = {
                "event_type": trailing_event_type,
                "account_id": str(current.get("account_id") or "") if current is not None else "",
                "page": page,
                "table": table,
                "row_index": len(rows) - 1,
                "row": list(rows[-1]),
            }
            # Event headers are typed structures.  They must never also act as
            # generic account-fact headers (latest-repayment includes 余额).
            pending_labels = None
            pending_label_source = None
            return
        if rows and _large_installment_tail_header(rows[-1]):
            target_event = next(
                (
                    event
                    for event in reversed(account_events)
                    if event.get("account_id") == (current or {}).get("account_id")
                    and event.get("event_type") == "large_installment"
                    and not {
                        "expiry_date",
                        "used_installment_amount",
                    }.issubset(event)
                ),
                None,
            )
            pending_event = {
                "event_type": "large_installment",
                "event_part": "tail",
                "target_event": target_event,
                "account_id": str(current.get("account_id") or "") if current is not None else "",
                "page": page,
                "table": table,
                "row_index": len(rows) - 1,
                "row": list(rows[-1]),
            }
            pending_labels = None
            pending_label_source = None
            return
        pending_event = None
        pending_labels = rows[-1] if rows and _label_row(rows[-1]) else None
        pending_label_source = (
            (page, table, len(rows) - 1) if pending_labels is not None else None
        )

    account_pages = list(getattr(parse_result, "pages", None) or [])
    cross_page_order_resolved = _account_reading_order_resolution(
        parse_result, account_pages
    )[-1]
    for page_offset, page in enumerate(
        _account_ordered_pages(parse_result, account_pages)
    ):
        if page_offset > 0 and not cross_page_order_resolved:
            # The sealed input order remains usable inside each page, but it
            # cannot carry entity state across a page boundary without a
            # complete, unique registered reading-order plane.
            flush_pending_event("unresolved_reading_order_page_boundary")
            flush_pending_fact_labels("unresolved_reading_order_page_boundary")
            flush_pending_institution("unresolved_reading_order_page_boundary")
            current = None
            pending_labels = None
            pending_label_source = None
            current_table_id = ""
            current_logical_page = 0
            current_table_top = None
            phase = "non_revolving_loan"
            phase_logical_page = 0
        page_tables = list(getattr(page, "tables", None) or [])
        page_tables.sort(
            key=lambda candidate: (
                _table_top_value(candidate) is None,
                _table_top_value(candidate) or 0.0,
            )
        )
        for source_table_index, table in enumerate(page_tables):
            rows = _table_rows(table)
            if not rows:
                continue
            compact = _compact(" ".join(cell for row in rows[:6] for cell in row))
            if _account_base(rows):
                if current is not None:
                    flush_pending_event("next_account", candidate_table=table)
                    flush_pending_fact_labels("next_account")
                    flush_pending_institution("next_account")
                if "发卡机构" in compact:
                    account_type = "credit_card" if "业务种类" in compact else "quasi_credit_card"
                    account_family_basis = "card_table_signature"
                    phase = account_type
                    phase_logical_page = int(getattr(page, "page_number", 0) or 0)
                elif "账户授信额度" in compact:
                    account_type = "revolving_loan_subaccount"
                    # R1 and R2 share this canonical first-row label.  The
                    # table shape therefore proves a revolving-loan account,
                    # but it does not distinguish the printed family variant.
                    account_family_basis = "shared_revolving_credit_limit_signature"
                    phase = account_type
                    phase_logical_page = int(getattr(page, "page_number", 0) or 0)
                elif phase in {"revolving_loan_subaccount", "revolving_loan_account"} and (
                    int(getattr(page, "page_number", 0) or 0) == phase_logical_page
                    or (
                        cross_page_order_resolved
                        and _registered_account_pages_are_adjacent(
                            parse_result,
                            phase_logical_page,
                            int(getattr(page, "page_number", 0) or 0),
                        )
                    )
                ):
                    account_type = "revolving_loan_account"
                    account_family_basis = "revolving_table_phase_carry"
                    phase = account_type
                    phase_logical_page = int(getattr(page, "page_number", 0) or 0)
                else:
                    account_type = "non_revolving_loan"
                    account_family_basis = "non_revolving_table_signature"
                    phase = account_type
                    phase_logical_page = int(getattr(page, "page_number", 0) or 0)

                table_ref = _source_ref(page, table)
                table_observation_id = stable_record_id(
                    "credit_account_table_observation",
                    account_type,
                    table_ref.get("logical_page"),
                    table_ref.get("source_page"),
                    table_ref.get("table_id"),
                    table_ref.get("bbox"),
                    source_table_index,
                )
                current = {
                    # A table is an observation, not an account identity.  Only
                    # a printed anchor can contribute the family ordinal.  The
                    # source-position digest stays stable when the ordinal is
                    # unreadable and cannot silently masquerade as one.
                    "account_id": table_observation_id,
                    "_table_observation_id": table_observation_id,
                    "_table_observation_instance_id": stable_record_id(
                        "credit_account_table_observation_instance",
                        table_observation_id,
                        len(accounts),
                    ),
                    "sequence": len(accounts) + 1,
                    "account_type": account_type,
                    "_table_account_family_basis": account_family_basis,
                    "source": "native_detail_account_table",
                    "source_refs": [table_ref],
                    "confidence": 1.0,
                    "canonical_raw": {},
                }
                if account_type in {"credit_card", "quasi_credit_card"}:
                    current["credit_card_type"] = account_type
                _apply_account_facts(
                    parse_result,
                    current,
                    rows,
                    page=page,
                    table=table,
                    physical_row_indices=list(range(len(rows))),
                )
                accounts.append(current)
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                account_events.extend(
                    _account_events(
                        parse_result,
                        current,
                        page,
                        table,
                        rows,
                        defer_trailing_large_installment_tail=True,
                    )
                )
                set_trailing_pending_state(page, table, rows)
                current_table_id = str(getattr(table, "table_id", "") or "")
                current_logical_page = int(getattr(page, "page_number", 0) or 0)
                current_table_top = _table_top_value(table)
                continue

            logical_page = int(getattr(page, "page_number", 0) or 0)
            candidate_table_id = str(getattr(table, "table_id", "") or "")
            continuation = None
            if current is not None and callable(continuation_check) and current_table_id and candidate_table_id:
                continuation = continuation_check(current_table_id, candidate_table_id)

            headerless_owner = (
                _bounded_headerless_card_owner(
                    parse_result,
                    current=current,
                    prior_logical_page=current_logical_page,
                    prior_table_top=current_table_top,
                    candidate_page=page,
                    candidate_table=table,
                    source_table_index=source_table_index,
                    pending_labels=pending_labels,
                    rows=rows,
                    prior_accounts=accounts,
                    cross_page_order_resolved=cross_page_order_resolved,
                )
                if current is not None and not _other_entity_table(rows)
                else None
            )
            if headerless_owner is not None:
                owner_skeleton = dict(headerless_owner["skeleton"])
                strong_values = dict(headerless_owner["strong_values"])
                header_labels = list(headerless_owner["header_labels"])
                header_refs = [
                    dict(ref) for ref in headerless_owner.get("header_refs") or ()
                ]
                ownership_basis = str(
                    headerless_owner.get("ownership_basis") or ""
                )
                prior_account_id = str(current.get("account_id") or "")
                header_source = (
                    pending_label_source
                    if ownership_basis == "pending_table_header_row"
                    else None
                )
                flush_pending_event("next_account", candidate_table=table)
                flush_pending_institution("next_account")
                pending_labels = None
                pending_label_source = None

                account_type = str(owner_skeleton.get("account_type") or "credit_card")
                table_ref = _source_ref(page, table)
                table_observation_id = stable_record_id(
                    "credit_account_table_observation",
                    account_type,
                    table_ref.get("logical_page"),
                    table_ref.get("source_page"),
                    table_ref.get("table_id"),
                    table_ref.get("bbox"),
                    source_table_index,
                )
                current = {
                    "account_id": table_observation_id,
                    "_table_observation_id": table_observation_id,
                    "_table_observation_instance_id": stable_record_id(
                        "credit_account_table_observation_instance",
                        table_observation_id,
                        len(accounts),
                    ),
                    "_pending_anchor_account_id": str(
                        owner_skeleton.get("account_id") or ""
                    ),
                    "sequence": len(accounts) + 1,
                    "account_type": account_type,
                    "credit_card_type": account_type,
                    "_table_account_family_basis": (
                        "printed_card_anchor_exact_header_lattice"
                    ),
                    "source": "native_detail_account_table",
                    "source_refs": [
                        table_ref,
                        *(
                            dict(ref)
                            for ref in owner_skeleton.get("source_refs") or ()
                            if isinstance(ref, Mapping)
                        ),
                    ],
                    "confidence": 1.0,
                    "canonical_raw": {},
                }
                fact_rows = [list(row) for row in rows]
                credit_limit_residue = str(
                    strong_values.pop("_credit_limit_residue", "") or ""
                )
                credit_limit_column = next(
                    (
                        column
                        for column, label in enumerate(header_labels)
                        if _ACCOUNT_BASIC_HEADER_ROLES.get(_compact(label))
                        == "credit_limit"
                    ),
                    None,
                )
                raw_credit_limit = ""
                if (
                    credit_limit_residue
                    and credit_limit_column is not None
                    and fact_rows
                    and credit_limit_column < len(fact_rows[0])
                ):
                    raw_credit_limit = str(fact_rows[0][credit_limit_column])
                    fact_rows[0][credit_limit_column] = str(
                        strong_values["credit_limit"]
                    )
                _apply_account_facts(
                    parse_result,
                    current,
                    [header_labels, *fact_rows],
                    page=page,
                    table=table,
                    physical_row_indices=[None, *range(len(rows))],
                )
                if raw_credit_limit:
                    current.setdefault("canonical_raw", {})[
                        "credit_limit"
                    ] = raw_credit_limit
                    residue_ref = _source_ref(
                        page,
                        table,
                        row=0,
                        column=int(credit_limit_column),
                    )
                    residue_ref["binding"] = (
                        "printed_anchor_exact_card_header_lattice"
                    )
                    _report_account_cluster_residue(
                        parse_result,
                        current,
                        target_record_id=table_observation_id,
                        field_names=("credit_limit",),
                        raw=raw_credit_limit,
                        residue=credit_limit_residue,
                        source_ref=residue_ref,
                    )
                accounts.append(current)
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                account_events.extend(
                    _account_events(
                        parse_result,
                        current,
                        page,
                        table,
                        rows,
                        defer_trailing_large_installment_tail=True,
                    )
                )
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                issue_refs = [
                    *(
                        dict(ref)
                        for ref in owner_skeleton.get("source_refs") or ()
                        if isinstance(ref, Mapping)
                    ),
                    *(
                        (
                            _source_ref(
                                header_source[0],
                                header_source[1],
                                row=int(header_source[2]),
                            ),
                        )
                        if header_source is not None
                        else ()
                    ),
                    *header_refs,
                    table_ref,
                ]
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_headerless_account_owner_resolved",
                        message=(
                            "A complete exact card header and printed anchor assigned the "
                            "adjacent page's headerless value table to a new account."
                        ),
                        severity="info",
                        status="resolved",
                        parser_stage="candidate_b_account_table_ownership",
                        target_dataset="credit_accounts",
                        target_record_id=str(owner_skeleton.get("account_id") or "")
                        or table_observation_id,
                        observed_value={
                            "prior_account_observation_id": prior_account_id,
                            "pending_anchor_account_id": owner_skeleton.get("account_id"),
                            "candidate_table_id": candidate_table_id,
                            "strong_identity": strong_values,
                            "ownership_basis": ownership_basis,
                        },
                        candidate_value={
                            "account_type": account_type,
                            "category_sequence": owner_skeleton.get("category_sequence"),
                        },
                        source_refs=issue_refs,
                        reason_codes=(
                            "registered_printed_pages_adjacent",
                            "complete_card_header_in_template_order",
                            "finite_card_identity_contracts",
                            "distinct_account_identifier",
                            "unique_printed_anchor",
                            ownership_basis,
                        ),
                    ),
                )
                set_trailing_pending_state(page, table, rows)
                current_table_id = candidate_table_id
                current_logical_page = logical_page
                current_table_top = _table_top_value(table)
                continue

            geometric_prior_owner = bool(
                current is not None
                and not _other_entity_table(rows)
                and _geometric_prior_account_continuation(
                    parse_result=parse_result,
                    page=page,
                    table=table,
                    page_tables=page_tables,
                    current_logical_page=current_logical_page,
                    current_table_top=current_table_top,
                    rows=rows,
                    pending_labels=pending_labels,
                    cross_page_order_resolved=cross_page_order_resolved,
                )
            )

            # Ownership comes either from the entity graph or from the unique
            # static interval around a canonical secondary account table.  The
            # geometric path never relies on a footer or page adjacency alone.
            if current is not None and (continuation is True or geometric_prior_owner) and not _other_entity_table(rows):
                continuation_values = [value for row in rows for value in _nonempty(row)]
                if (
                    len(continuation_values) == 1
                    and continuation_values[0] == "期"
                    and account_events
                    and account_events[-1].get("account_id") == current.get("account_id")
                    and str(account_events[-1].get("transaction_type") or "").endswith("分")
                ):
                    account_events[-1]["transaction_type"] = (
                        str(account_events[-1]["transaction_type"]) + continuation_values[0]
                    )
                fact_rows = ([pending_labels] if pending_labels else []) + rows
                _apply_account_facts(
                    parse_result,
                    current,
                    fact_rows,
                    page=page,
                    table=table,
                    physical_row_indices=([None] if pending_labels else []) + list(range(len(rows))),
                )
                repayment_rows, _context = _repayment_records(page, table, rows, current)
                repayments.extend(repayment_rows)
                if pending_event is not None:
                    value_row_index = _account_event_continuation_value_index(rows)
                    if value_row_index is None:
                        flush_pending_event(
                            "affirmed_continuation_without_value_row",
                            candidate_table=table,
                        )
                        account_events.extend(
                            _account_events(
                                parse_result,
                                current,
                                page,
                                table,
                                rows,
                                defer_trailing_large_installment_tail=True,
                            )
                        )
                    else:
                        if pending_event.get("event_part") == "tail":
                            _consume_pending_large_installment_tail(
                                parse_result,
                                pending_event,
                                page=page,
                                table=table,
                                rows=rows,
                                value_row_index=value_row_index,
                            )
                            account_events.extend(
                                _account_events(
                                    parse_result,
                                    current,
                                    page,
                                    table,
                                    rows,
                                    defer_trailing_large_installment_tail=True,
                                )
                            )
                        else:
                            account_events.extend(
                                _account_events(
                                    parse_result,
                                    current,
                                    page,
                                    table,
                                    rows,
                                    leading_header=pending_event,
                                    leading_value_row_index=value_row_index,
                                    defer_trailing_large_installment_tail=True,
                                )
                            )
                        pending_event = None
                else:
                    account_events.extend(
                        _account_events(
                            parse_result,
                            current,
                            page,
                            table,
                            rows,
                            defer_trailing_large_installment_tail=True,
                        )
                    )
                continuation_ref = _source_ref(page, table)
                if continuation_ref not in current.setdefault("source_refs", []):
                    current["source_refs"].append(continuation_ref)
                set_trailing_pending_state(page, table, rows)
                current_table_id = str(getattr(table, "table_id", "") or current_table_id)
                current_logical_page = logical_page or current_logical_page
                current_table_top = _table_top_value(table) or current_table_top
                continue

            if current is not None:
                compact_values = _compact(" ".join(cell for row in rows for cell in row))
                looks_like_account_fragment = (
                    any(label in compact_values for label in _ACCOUNT_LABELS)
                    or "还款记录" in compact_values
                    or "逾期金额" in compact_values
                )
                if looks_like_account_fragment and not _other_entity_table(rows):
                    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                        make_issue,
                        record_issue,
                    )

                    record_issue(
                        parse_result,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_account_table_fragment_owner_unresolved",
                            message=(
                                "An account-like table fragment was not joined because canonical continuation "
                                "ownership was not affirmatively established."
                            ),
                            parser_stage="candidate_b_account_table_ownership",
                            target_dataset="credit_accounts",
                            target_record_id=str(current.get("account_id") or "") or None,
                            source_refs=(_source_ref(page, table),),
                            observed_value={
                                "left_table_id": current_table_id,
                                "candidate_table_id": candidate_table_id,
                                "left_logical_page": current_logical_page,
                                "candidate_logical_page": logical_page,
                                "continuation_decision": continuation,
                            },
                            reason_codes=(
                                "canonical_continuation_not_verified",
                                "table_fragment_withheld",
                                "neighbour_order_not_used",
                            ),
                        ),
                    )
                    flush_pending_event("unowned_account_fragment", candidate_table=table)
                    flush_pending_fact_labels("unowned_account_fragment")
                    flush_pending_institution("unowned_account_fragment")
                    current = None
                    current_table_id = ""
                    current_logical_page = 0
                    current_table_top = None
                elif _other_entity_table(rows):
                    flush_pending_event("next_section", candidate_table=table)
                    flush_pending_fact_labels("next_section")
                    flush_pending_institution("next_section")
                    current = None
                    current_table_id = ""
                    current_logical_page = 0
                    current_table_top = None

    flush_pending_event("end_of_document")
    flush_pending_fact_labels("end_of_document")
    flush_pending_institution("end_of_document")

    account_identifiers = {str(account["account_id"]): account.get("account_identifier") for account in accounts}
    deduped: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in repayments:
        if not record.get("account_identifier"):
            record["account_identifier"] = account_identifiers.get(str(record["account_id"]))
        key = (
            str(record.get("_table_observation_instance_id") or ""),
            str(record["account_id"]),
            int(record["year"]),
            int(record["month"]),
        )
        deduped[key] = record
    for account in accounts:
        account.pop("_repayment_context", None)
    return accounts, list(deduped.values()), account_events


_ACCOUNT_SECTION_END = (
    "授信协议信息",
    "相关还款责任信息",
    "非信贷交易信息明细",
    "公共信息明细",
    "查询记录",
    "报告说明",
)


def _account_family_from_heading(value: Any) -> tuple[str, str] | None:
    """Resolve the printed PBOC family without collapsing R1 and R2."""
    compact = re.sub(r"[\s（）()：:、，,。._-]+", "", str(value or ""))
    if "非循环贷账户" in compact:
        return "non_revolving_loan", "exact"
    if "循环贷账户一" in compact:
        return "revolving_loan_subaccount", "exact"
    if "循环贷账户二" in compact:
        return "revolving_loan_account", "exact"
    if "准贷记卡账户" in compact:
        return "quasi_credit_card", "exact"
    if "贷记卡账户" in compact:
        return "credit_card", "exact"
    if "循环贷账户" in compact:
        return "revolving_loan_subaccount", "ambiguous_missing_variant"
    return None
_ACCOUNT_ANCHOR_RE = re.compile(
    r"^[^\u3400-\u9fff0-9]*账户\s*(\d{1,3})?\s*(?:[：:(（]|$|\s)"
)
_ACCOUNT_IDENTIFIER_PREFIX_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z][0-9]{8}[A-Z][A-Z0-9]{0,20})(?![A-Z0-9])",
    re.I,
)
_ACCOUNT_IDENTIFIER_SUFFIX_RE = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9]{4,})(?=[A-Z0-9]*\d)[A-Z0-9]{4,}(?![A-Z0-9])",
    re.I,
)
_CANONICAL_PBOC_ACCOUNT_IDENTIFIER_RE = re.compile(
    r"[A-Z][0-9]{8}[A-Z][A-Z0-9]{14,70}",
    re.I,
)


def _canonical_pboc_account_identifier(value: Any) -> str:
    """Return one continuous canonical PBOC account identifier.

    Account-anchor geometry reconstructs the same letter + eight-digit +
    letter prefix and requires a 24--80 character continuous identifier.
    Headerless ownership must meet that stronger document identity contract;
    a merely mixed twelve-character token is not an account anchor.
    """

    marker = re.sub(r"\s+", "", str(value or "")).upper()
    if not 24 <= len(marker) <= 80:
        return ""
    return marker if _CANONICAL_PBOC_ACCOUNT_IDENTIFIER_RE.fullmatch(marker) else ""

_ACCOUNT_BASIC_HEADER_ROLES = {
    "管理机构": "management_institution",
    "发卡机构": "management_institution",
    "账户标识": "account_identifier",
    "开立日期": "open_date",
    "到期日期": "due_date",
    "借款金额": "loan_amount",
    "账户授信额度": "credit_limit",
    "共享授信额度": "shared_credit_limit",
    "账户币种": "currency",
    "币种": "currency",
    "业务种类": "business_type",
    "担保方式": "guarantee_type",
    "还款期数": "repayment_periods",
    "还款频率": "repayment_frequency",
    "还款方式": "repayment_method",
    "共同借款标志": "co_borrower_flag",
}
_ACCOUNT_BASIC_CARD_TEMPLATE = (
    "management_institution",
    "account_identifier",
    "open_date",
    "credit_limit",
    "shared_credit_limit",
    "currency",
    "business_type",
    "guarantee_type",
)
_ACCOUNT_BASIC_LOAN_TEMPLATE = (
    "management_institution",
    "account_identifier",
    "open_date",
    "due_date",
    "loan_amount",
    "currency",
)
_ACCOUNT_BASIC_REVOLVING_ACCOUNT_TEMPLATE = (
    "management_institution",
    "account_identifier",
    "open_date",
    "due_date",
    "credit_limit",
    "currency",
)
_ACCOUNT_LOAN_SECOND_ROW_TEMPLATE = (
    "business_type",
    "guarantee_type",
    "repayment_periods",
    "repayment_frequency",
    "repayment_method",
    "co_borrower_flag",
)
_ACCOUNT_BASIC_TEMPLATES = {
    "credit_card": _ACCOUNT_BASIC_CARD_TEMPLATE,
    "quasi_credit_card": _ACCOUNT_BASIC_CARD_TEMPLATE,
    "non_revolving_loan": _ACCOUNT_BASIC_LOAN_TEMPLATE,
    "revolving_loan_subaccount": _ACCOUNT_BASIC_LOAN_TEMPLATE,
    "revolving_loan_account": _ACCOUNT_BASIC_REVOLVING_ACCOUNT_TEMPLATE,
}
_ACCOUNT_BASIC_ALL_GEOMETRY_FIELDS = frozenset(
    {
        "management_institution",
        "account_identifier",
        "open_date",
        "due_date",
        "loan_amount",
        "credit_limit",
        "shared_credit_limit",
        "currency",
        "business_type",
        "guarantee_type",
        "repayment_periods",
        "repayment_frequency",
        "repayment_method",
        "co_borrower_flag",
    }
)
_ACCOUNT_BASIC_NON_BUSINESS_LABELS = frozenset({"附注", "备注", "说明"})
_ACCOUNT_BASIC_ROW_BOUNDARY_LABELS = frozenset(
    {
        "账户状态",
        "状态",
        "五级分类",
        "余额",
        "透支余额",
        "账单日",
        "本月应还款",
        "还款记录",
    }
)


def _account_evidence_bbox(line: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = line.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _account_evidence_source_ref(
    lines: Iterable[Mapping[str, Any]],
    *,
    field_name: str,
) -> dict[str, Any]:
    selected = [line for line in lines if _account_evidence_bbox(line) is not None]
    first = selected[0] if selected else {}
    logical_pages = list(
        dict.fromkeys(int(line.get("page") or line.get("logical_page") or 0) for line in selected)
    )
    source_pages = list(
        dict.fromkeys(int(line.get("source_page") or 0) for line in selected)
    )
    ref: dict[str, Any] = {
        "source": "candidate_b_account_anchor_interval",
        "field_name": field_name,
        "binding": "canonical_account_header_geometry",
        "binding_quality": "canonical_account_header_geometry",
        "logical_page": logical_pages[0] if logical_pages else int(first.get("page") or 0),
        "source_page": source_pages[0] if source_pages else int(first.get("source_page") or 0),
        "evidence_ids": list(
            dict.fromkeys(
                str(evidence_id)
                for line in selected
                for evidence_id in line.get("evidence_ids") or ()
                if evidence_id
            )
        ),
    }
    if len(logical_pages) > 1:
        ref["logical_pages"] = logical_pages
    if len(source_pages) > 1:
        ref["source_pages"] = source_pages
    if selected and len(logical_pages) <= 1:
        boxes = [_account_evidence_bbox(line) for line in selected]
        finite_boxes = [bbox for bbox in boxes if bbox is not None]
        if finite_boxes:
            ref["bbox"] = [
                min(bbox[0] for bbox in finite_boxes),
                min(bbox[1] for bbox in finite_boxes),
                max(bbox[2] for bbox in finite_boxes),
                max(bbox[3] for bbox in finite_boxes),
            ]
    return ref


def _reject_exact_source_absence_conflict(
    parse_result: Any,
    record: dict[str, Any],
    *,
    target_record_id: str,
    field_name: str,
    raw: str,
    source_ref: Mapping[str, Any],
) -> None:
    """Withhold a populated field contradicted by exact source absence."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    prior = record.pop(field_name, None)
    _append_internal_field(record, "_unresolved_fields", field_name)
    ref = {**dict(source_ref), "field_name": field_name}
    refs = record.setdefault("source_refs_by_field", {}).setdefault(field_name, [])
    if ref not in refs:
        refs.append(ref)
    record.setdefault("canonical_raw", {})[field_name] = [prior, raw]
    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_exact_slot_source_absence_conflict",
            message=(
                "A source-bound business value conflicted with explicit source absence "
                "in the same canonical field; the value was withheld."
            ),
            parser_stage="candidate_b_account_anchor_header_geometry",
            target_dataset="credit_accounts",
            target_record_id=target_record_id,
            field_name=field_name,
            observed_value=raw,
            candidate_value=prior,
            source_refs=(ref,),
            reason_codes=(
                "explicit_source_absence",
                "conflicting_exact_observation",
                "normalized_value_withheld",
            ),
        ),
    )


def _account_composite_header_parts(
    line: Mapping[str, Any],
    *,
    template: Iterable[str],
) -> list[dict[str, Any]] | None:
    """Split one exact adjacent-label header box into typed geometry parts."""

    bbox = _account_evidence_bbox(line)
    marker = _compact(line.get("text") or "")
    roles = tuple(template)
    if bbox is None or not marker:
        return None
    aliases_by_role = {
        role: tuple(
            sorted(
                {
                    _compact(label)
                    for label, candidate_role in _ACCOUNT_BASIC_HEADER_ROLES.items()
                    if candidate_role == role
                },
                key=len,
                reverse=True,
            )
        )
        for role in roles
    }
    for start in range(len(roles)):
        cursor = 0
        parts: list[tuple[str, str]] = []
        for role in roles[start:]:
            alias = next(
                (
                    candidate
                    for candidate in aliases_by_role[role]
                    if marker.startswith(candidate, cursor)
                ),
                None,
            )
            if alias is None:
                break
            parts.append((role, alias))
            cursor += len(alias)
            if cursor == len(marker) and len(parts) >= 2:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                # A composite label is usable only when its source box spans
                # the adjacent canonical columns it claims to represent.  A
                # narrow OCR box cannot be expanded into invented geometry:
                # doing so can shift the following value cells by a column.
                if width < height * 2.5 * len(parts):
                    return None
                left = bbox[0]
                expanded: list[dict[str, Any]] = []
                for index, (_part_role, alias_value) in enumerate(parts):
                    right = (
                        bbox[2]
                        if index + 1 == len(parts)
                        else left + width / len(parts)
                    )
                    expanded.append(
                        {
                            **dict(line),
                            "text": alias_value,
                            "bbox": [left, bbox[1], right, bbox[3]],
                            "_account_composite_header": True,
                        }
                    )
                    left = right
                return expanded
    return None


def _account_basic_header_cluster(
    detail: list[dict[str, Any]],
    *,
    expected_roles: Iterable[str],
) -> tuple[list[tuple[int, dict[str, Any], str, tuple[float, float, float, float]]], int] | None:
    """Find one complete canonical account header inside its anchor-owned interval."""

    labels: list[tuple[int, dict[str, Any], str, tuple[float, float, float, float]]] = []
    for index, line in enumerate(detail):
        bbox = _account_evidence_bbox(line)
        role = _ACCOUNT_BASIC_HEADER_ROLES.get(_compact(line.get("text") or ""))
        if bbox is None or role is None:
            continue
        labels.append((index, line, role, bbox))

    expected = frozenset(expected_roles)
    candidates: list[
        tuple[list[tuple[int, dict[str, Any], str, tuple[float, float, float, float]]], int]
    ] = []
    seen_clusters: set[tuple[int, ...]] = set()
    for seed_index, seed_line, _seed_role, seed_bbox in labels:
        seed_page = int(seed_line.get("page") or seed_line.get("logical_page") or 0)
        seed_center = (seed_bbox[1] + seed_bbox[3]) / 2.0
        seed_height = seed_bbox[3] - seed_bbox[1]
        cluster = [
            item
            for item in labels
            if int(item[1].get("page") or item[1].get("logical_page") or 0) == seed_page
            and abs(((item[3][1] + item[3][3]) / 2.0) - seed_center)
            <= max(12.0, seed_height * 1.75, (item[3][3] - item[3][1]) * 1.75)
        ]
        marker = tuple(sorted(item[0] for item in cluster))
        if marker in seen_clusters:
            continue
        seen_clusters.add(marker)
        roles = [item[2] for item in cluster]
        if frozenset(roles) != expected or any(
            roles.count(role) > 1 for role in set(roles)
        ):
            continue
        candidates.append((cluster, max(item[0] for item in cluster)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: min(label[0] for label in item[0]))
    earliest = candidates[0]
    earliest_marker = tuple(sorted(item[0] for item in earliest[0]))
    if any(
        tuple(sorted(item[0] for item in candidate[0])) != earliest_marker
        and min(item[0] for item in candidate[0]) == min(item[0] for item in earliest[0])
        for candidate in candidates[1:]
    ):
        return None
    return earliest


def _account_basic_header_clusters(
    detail: list[dict[str, Any]],
    *,
    expected_roles: Iterable[str],
) -> list[
    tuple[
        list[tuple[int, dict[str, Any], str, tuple[float, float, float, float]]],
        int,
    ]
]:
    """Return non-overlapping complete header rows in evidence order."""

    clusters: list[
        tuple[
            list[
                tuple[
                    int,
                    dict[str, Any],
                    str,
                    tuple[float, float, float, float],
                ]
            ],
            int,
        ]
    ] = []
    offset = 0
    while offset < len(detail):
        found = _account_basic_header_cluster(
            detail[offset:],
            expected_roles=expected_roles,
        )
        if found is None:
            break
        header, header_end = found
        adjusted = [
            (index + offset, line, role, bbox)
            for index, line, role, bbox in header
        ]
        clusters.append((adjusted, header_end + offset))
        offset += header_end + 1
    return clusters


def _account_header_value_population(
    detail: list[dict[str, Any]],
    header: list[
        tuple[int, dict[str, Any], str, tuple[float, float, float, float]]
    ],
    header_end: int,
) -> int:
    """Count physically populated canonical slots owned by one header row."""

    ordered = sorted(header, key=lambda item: (item[3][0] + item[3][2]) / 2.0)
    centers = [(item[3][0] + item[3][2]) / 2.0 for item in ordered]
    header_page = int(
        ordered[0][1].get("page") or ordered[0][1].get("logical_page") or 0
    )
    header_y = max((item[3][1] + item[3][3]) / 2.0 for item in ordered)
    populated: set[str] = set()
    for line in detail[header_end + 1 :]:
        compact = _compact(line.get("text") or "")
        if (
            compact.startswith("截至")
            or compact in _ACCOUNT_BASIC_HEADER_ROLES
            or compact in _ACCOUNT_BASIC_ROW_BOUNDARY_LABELS
        ):
            break
        bbox = _account_evidence_bbox(line)
        if bbox is None:
            continue
        line_page = int(line.get("page") or line.get("logical_page") or 0)
        if line_page < header_page:
            continue
        if line_page == header_page and (bbox[1] + bbox[3]) / 2.0 <= header_y:
            continue
        covered = {
            item[2]
            for item, center in zip(ordered, centers, strict=True)
            if bbox[0] <= center <= bbox[2]
        }
        if not covered:
            center = (bbox[0] + bbox[2]) / 2.0
            nearest = min(
                zip(ordered, centers, strict=True),
                key=lambda item: abs(item[1] - center),
            )
            covered.add(nearest[0][2])
        populated.update(covered)
    return len(populated)


def _account_basic_geometry_observations(
    detail: list[dict[str, Any]],
) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
    """Bind first-row account values by canonical header geometry.

    This consumes the corrected evidence plane already owned by one printed
    account anchor.  It does not synthesize a table or use document/person/page
    constants, and it remains valid when the value row starts on the next
    logical page.
    """

    account_type = str((detail[0] if detail else {}).get("account_type") or "")
    template = _ACCOUNT_BASIC_TEMPLATES.get(account_type)
    if template is None:
        return False, {}
    geometry_fields = frozenset(template)
    role_order = {role: index for index, role in enumerate(template)}
    expanded_detail: list[dict[str, Any]] = []
    for line in detail:
        composite_parts = _account_composite_header_parts(line, template=template)
        if composite_parts is None:
            expanded_detail.append(line)
        else:
            expanded_detail.extend(composite_parts)
    detail = expanded_detail

    composite_header_lines: list[dict[str, Any]] = []
    for line in detail:
        compact = _compact(line.get("text") or "")
        if compact in _ACCOUNT_BASIC_HEADER_ROLES:
            continue
        roles = {
            role
            for label, role in _ACCOUNT_BASIC_HEADER_ROLES.items()
            if _compact(label) in compact
        }.intersection(geometry_fields)
        if len(roles) > 1 and _account_evidence_bbox(line) is not None:
            composite_header_lines.append(line)
    if composite_header_lines:
        raw = " ".join(_clean(line.get("text") or "") for line in composite_header_lines)
        return True, {
            field_name: [
                {
                    "raw": raw,
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        composite_header_lines,
                        field_name=field_name,
                    ),
                }
            ]
            for field_name in geometry_fields
        }

    candidates = _account_basic_header_clusters(detail, expected_roles=template)
    if not candidates:
        header_like_lines: list[dict[str, Any]] = []
        labels = [
            (line, _ACCOUNT_BASIC_HEADER_ROLES.get(_compact(line.get("text") or "")))
            for line in detail
            if _account_evidence_bbox(line) is not None
        ]
        labels = [(line, role) for line, role in labels if role is not None]
        for seed_line, _seed_role in labels:
            seed_bbox = _account_evidence_bbox(seed_line)
            if seed_bbox is None:
                continue
            seed_page = int(seed_line.get("page") or seed_line.get("logical_page") or 0)
            seed_center = (seed_bbox[1] + seed_bbox[3]) / 2.0
            cluster = [
                line
                for line, _role in labels
                if int(line.get("page") or line.get("logical_page") or 0) == seed_page
                and (
                    (bbox := _account_evidence_bbox(line)) is not None
                    and abs(((bbox[1] + bbox[3]) / 2.0) - seed_center) <= 12.0
                )
            ]
            cluster_roles = {
                _ACCOUNT_BASIC_HEADER_ROLES.get(_compact(line.get("text") or ""))
                for line in cluster
            }
            if len(cluster_roles - {None}) >= 3:
                header_like_lines = cluster
                break
        if header_like_lines:
            raw = " ".join(_clean(line.get("text") or "") for line in header_like_lines)
            return True, {
                field_name: [
                    {
                        "raw": raw,
                        "value": None,
                        "source_ref": _account_evidence_source_ref(
                            header_like_lines,
                            field_name=field_name,
                        ),
                    }
                ]
                for field_name in geometry_fields
            }
        return False, {}
    anchor_line = detail[0] if detail else {}
    anchor_bbox = _account_evidence_bbox(anchor_line)
    anchor_page = int(anchor_line.get("page") or anchor_line.get("logical_page") or 0)
    eligible = [
        candidate
        for candidate in candidates
        if anchor_bbox is not None
        and anchor_page
        and (
            int(
                candidate[0][0][1].get("page")
                or candidate[0][0][1].get("logical_page")
                or 0
            )
            > anchor_page
            or (
                int(
                    candidate[0][0][1].get("page")
                    or candidate[0][0][1].get("logical_page")
                    or 0
                )
                == anchor_page
                and min(item[3][1] for item in candidate[0]) + 2.0
                >= anchor_bbox[3]
            )
        )
    ]
    if not eligible:
        invalid_header = candidates[0][0]
        raw = " ".join(
            _clean(item[1].get("text") or "") for item in invalid_header
        )
        return True, {
            field_name: [
                {
                    "raw": raw,
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        (item[1] for item in invalid_header),
                        field_name=field_name,
                    ),
                }
            ]
            for field_name in geometry_fields
        }
    if len(eligible) == 1:
        header, header_end = eligible[0]
    else:
        populated = [
            candidate
            for candidate in eligible
            if _account_header_value_population(detail, *candidate) >= 2
        ]
        if len(populated) == 1:
            header, header_end = populated[0]
        else:
            duplicate_header_lines = [
                item[1]
                for candidate, _end in eligible
                for item in candidate
            ]
            raw = " ".join(
                _clean(line.get("text") or "") for line in duplicate_header_lines
            )
            return True, {
                field_name: [
                    {
                        "raw": raw,
                        "value": None,
                        "source_ref": _account_evidence_source_ref(
                            duplicate_header_lines,
                            field_name=field_name,
                        ),
                    }
                ]
                for field_name in geometry_fields
            }
    header_page = int(
        header[0][1].get("page") or header[0][1].get("logical_page") or 0
    )
    ordered = sorted(header, key=lambda item: (item[3][0] + item[3][2]) / 2.0)
    ordered_contract_positions = [role_order[item[2]] for item in ordered]
    if ordered_contract_positions != sorted(ordered_contract_positions):
        # The template is canonical.  A transposed label row cannot redefine
        # its physical slots, so expose every affected target field instead of
        # decoding values under the wrong header.
        invalid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _index, line, role, _bbox in header:
            if role not in geometry_fields:
                continue
            invalid[role].append(
                {
                    "raw": "",
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        (line,),
                        field_name=role,
                    ),
                }
            )
        return True, dict(invalid)
    centers = [(item[3][0] + item[3][2]) / 2.0 for item in ordered]
    oversized_header_lines = [
        item[1]
        for item in ordered
        if sum(item[3][0] <= center <= item[3][2] for center in centers) > 1
    ]
    if oversized_header_lines:
        raw = " ".join(_clean(line.get("text") or "") for line in oversized_header_lines)
        return True, {
            field_name: [
                {
                    "raw": raw,
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        oversized_header_lines,
                        field_name=field_name,
                    ),
                }
            ]
            for field_name in geometry_fields
        }
    intervals: dict[str, tuple[float, float]] = {}
    for index, item in enumerate(ordered):
        left = float("-inf") if index == 0 else (centers[index - 1] + centers[index]) / 2.0
        right = (
            float("inf")
            if index + 1 == len(ordered)
            else (centers[index] + centers[index + 1]) / 2.0
        )
        intervals[item[2]] = (left, right)

    value_lines: list[dict[str, Any]] = []
    header_center_y = max((item[3][1] + item[3][3]) / 2.0 for item in header)
    for line in detail[header_end + 1 :]:
        compact = _compact(line.get("text") or "")
        if compact.startswith("截至"):
            break
        if compact in _ACCOUNT_BASIC_HEADER_ROLES:
            break
        if compact in _ACCOUNT_BASIC_ROW_BOUNDARY_LABELS:
            break
        bbox = _account_evidence_bbox(line)
        if bbox is None:
            continue
        line_page = int(line.get("page") or line.get("logical_page") or 0)
        if line_page < header_page:
            continue
        if line_page == header_page and (bbox[1] + bbox[3]) / 2.0 <= header_center_y:
            continue
        value_lines.append(line)

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in value_lines:
        bbox = _account_evidence_bbox(line)
        if bbox is None:
            continue
        # Corrected evidence normally preserves one word/fragment per box.  A
        # degraded OCR row can instead arrive as one box spanning several
        # canonical columns.  Route such a box to every header centre it
        # physically covers so only the field-specific finite contracts below
        # may retain a value; individualized fields cannot silently inherit a
        # centre-of-box guess.
        covered_roles = tuple(
            item[2]
            for item, header_center in zip(ordered, centers, strict=True)
            if bbox[0] <= header_center <= bbox[2]
        )
        if len(covered_roles) > 1:
            annotated = {
                **line,
                "_account_geometry_spanning_roles": covered_roles,
            }
            for role in covered_roles:
                by_role[role].append(annotated)
            continue
        center = (bbox[0] + bbox[2]) / 2.0
        for role, (left, right) in intervals.items():
            if left <= center < right:
                by_role[role].append(line)
                break

    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    header_lines_by_role = {
        role: line for _index, line, role, _bbox in header
    }
    source_absent_roles: set[str] = set()
    mixed_absence_roles: set[str] = set()
    for role in geometry_fields:
        lines = by_role.get(role, ())
        if not lines:
            continue
        raw = " ".join(_clean(line.get("text") or "") for line in lines)
        absent_lines = [
            line
            for line in lines
            if is_explicit_source_absence(_clean(line.get("text") or ""))
        ]
        if not absent_lines:
            continue
        if len(absent_lines) == len(lines):
            source_absent_roles.add(role)
            observations[role].append(
                {
                    "raw": raw,
                    "value": None,
                    "source_absent": True,
                    "source_ref": _account_evidence_source_ref(
                        lines,
                        field_name=role,
                    ),
                }
            )
            continue
        # A dash and a substantive token in one inferred slot contradict each
        # other.  Preserve the raw evidence but never silently discard either
        # half to publish the other as a business value.
        mixed_absence_roles.add(role)
        observations[role].append(
            {
                "raw": raw,
                "value": None,
                "source_ref": _account_evidence_source_ref(
                    lines,
                    field_name=role,
                ),
            }
        )

    institution_lines: list[dict[str, Any]] = []
    institution_chunks: list[str] = []
    institution_slot_lines = by_role.get("management_institution", ())
    institution_spans_columns = any(
        line.get("_account_geometry_spanning_roles") for line in institution_slot_lines
    )
    institution_residue_lines: list[dict[str, Any]] = []
    for line in institution_slot_lines:
        raw = _clean(line.get("text") or "")
        if _compact(raw) in _ACCOUNT_BASIC_NON_BUSINESS_LABELS or re.search(
            r"(?:其他|本栏|附?注|备注|说明|提示|注意|注释)",
            _compact(raw),
        ):
            institution_residue_lines.append(line)
            continue
        han_count = len(re.findall(r"[\u3400-\u9fff]", raw))
        if han_count < 2:
            continue
        # A multi-glyph legal-name fragment carrying alphanumeric debris is
        # ambiguous.  Single watermark/noise glyphs are simply not business
        # fragments and cannot contaminate the institution.
        if re.search(r"[A-Za-z0-9]", raw):
            institution_chunks = []
            institution_lines = []
            break
        institution_chunks.append(raw)
        institution_lines.append(line)
    if "management_institution" in source_absent_roles | mixed_absence_roles:
        pass
    elif institution_spans_columns:
        raw = " ".join(
            _clean(line.get("text") or "") for line in institution_slot_lines
        )
        observations["management_institution"].append(
            {
                "raw": raw,
                "value": None,
                "source_ref": _account_evidence_source_ref(
                    institution_slot_lines,
                    field_name="management_institution",
                ),
            }
        )
    elif institution_chunks:
        raw = " ".join(institution_chunks)
        value = _account_institution(raw)
        if value is not None and re.search(
            r"(?:银行|公司|中心|信用社|信托|分行|支行|营业部|合作社)$",
            _compact(value),
        ) is None:
            value = None
        observations["management_institution"].append(
            {
                "raw": raw,
                "value": value,
                "residue": _account_cluster_signature(
                    " ".join(
                        _clean(line.get("text") or "")
                        for line in institution_residue_lines
                    )
                ),
                "source_ref": _account_evidence_source_ref(
                    (*institution_residue_lines, *institution_lines),
                    field_name="management_institution",
                ),
            }
        )
    elif institution_residue_lines:
        raw = " ".join(
            _clean(line.get("text") or "") for line in institution_residue_lines
        )
        observations["management_institution"].append(
            {
                "raw": raw,
                "value": None,
                "source_ref": _account_evidence_source_ref(
                    institution_residue_lines,
                    field_name="management_institution",
                ),
            }
        )

    identifier_slot_lines = list(by_role.get("account_identifier", ()))
    identifier_lines: list[dict[str, Any]] = []
    identifier_parts: list[str] = []
    identifier_invalid = any(
        line.get("_account_geometry_spanning_roles") for line in identifier_slot_lines
    )
    for line in identifier_slot_lines:
        tokens = re.findall(r"[A-Z0-9]+", str(line.get("text") or "").upper())
        if not tokens:
            continue
        if len(tokens) != 1:
            identifier_invalid = True
        identifier_parts.extend(tokens)
        identifier_lines.append(line)

    if "account_identifier" in source_absent_roles | mixed_absence_roles:
        pass
    elif identifier_parts:
        # A printed identifier wraps down one narrow canonical column.  The
        # prefix and every non-final suffix fragment occupy the column width;
        # only the final fragment may be short.  This rejects watermark/noise
        # fragments such as a stray ``88`` instead of silently appending them.
        identifier_boxes = [
            _account_evidence_bbox(line) for line in identifier_lines
        ]
        finite_boxes = [bbox for bbox in identifier_boxes if bbox is not None]
        centers_x = [((bbox[0] + bbox[2]) / 2.0) for bbox in finite_boxes]
        widths = [(bbox[2] - bbox[0]) for bbox in finite_boxes]
        line_markers = [
            (
                str(line.get("text") or "").upper(),
                tuple(_account_evidence_bbox(line) or ()),
                tuple(str(value) for value in line.get("evidence_ids") or ()),
            )
            for line in identifier_lines
        ]
        geometrically_ordered_markers = [
            marker
            for _line, marker in sorted(
                zip(identifier_lines, line_markers, strict=True),
                key=lambda item: (
                    int(item[0].get("page") or item[0].get("logical_page") or 0),
                    (_account_evidence_bbox(item[0]) or (0.0, 0.0, 0.0, 0.0))[1],
                    (_account_evidence_bbox(item[0]) or (0.0, 0.0, 0.0, 0.0))[0],
                ),
            )
        ]
        evidence_ids = [
            str(evidence_id)
            for line in identifier_lines
            for evidence_id in line.get("evidence_ids") or ()
            if evidence_id
        ]
        if (
            line_markers != geometrically_ordered_markers
            or len(line_markers) != len(set(line_markers))
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            identifier_invalid = True
        ordered_boxes = [
            _account_evidence_bbox(line)
            for line in sorted(
                identifier_lines,
                key=lambda line: (
                    int(line.get("page") or line.get("logical_page") or 0),
                    (_account_evidence_bbox(line) or (0.0, 0.0, 0.0, 0.0))[1],
                    (_account_evidence_bbox(line) or (0.0, 0.0, 0.0, 0.0))[0],
                ),
            )
        ]
        ordered_lines = sorted(
            identifier_lines,
            key=lambda line: (
                int(line.get("page") or line.get("logical_page") or 0),
                (_account_evidence_bbox(line) or (0.0, 0.0, 0.0, 0.0))[1],
                (_account_evidence_bbox(line) or (0.0, 0.0, 0.0, 0.0))[0],
            ),
        )
        for previous_line, current_line, previous_box, current_box in zip(
            ordered_lines[:-1],
            ordered_lines[1:],
            ordered_boxes[:-1],
            ordered_boxes[1:],
            strict=True,
        ):
            if previous_box is None or current_box is None:
                identifier_invalid = True
                continue
            previous_page = int(
                previous_line.get("page") or previous_line.get("logical_page") or 0
            )
            current_page = int(
                current_line.get("page") or current_line.get("logical_page") or 0
            )
            if previous_page != current_page:
                continue
            previous_center = (previous_box[1] + previous_box[3]) / 2.0
            current_center = (current_box[1] + current_box[3]) / 2.0
            minimum_gap = 0.45 * max(
                previous_box[3] - previous_box[1],
                current_box[3] - current_box[1],
            )
            if current_center - previous_center < minimum_gap:
                identifier_invalid = True
        full_identifier = (
            identifier_parts[0]
            if len(identifier_parts) == 1
            and re.fullmatch(
                r"(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z]{1,3}[A-Z0-9]{9,77}",
                identifier_parts[0],
            )
            else None
        )
        if full_identifier is None:
            standard_prefix = bool(
                identifier_parts
                and _ACCOUNT_IDENTIFIER_PREFIX_RE.fullmatch(identifier_parts[0])
            )
            alternate_prefix = bool(
                identifier_parts
                and re.fullmatch(r"[A-Z]{2}\d{8,38}", identifier_parts[0])
                and len(identifier_parts) == 2
            )
            if (
                not identifier_parts
                or not (standard_prefix or alternate_prefix)
                or len(identifier_parts) < 2
                or (
                    account_type in {"credit_card", "quasi_credit_card"}
                    and standard_prefix
                    and len(identifier_parts) != 4
                )
                or (standard_prefix and len(identifier_parts) > 4)
                or any(
                    len(part) < max(4, len(identifier_parts[0]) - 2)
                    or not re.search(r"\d", part)
                    for part in identifier_parts[1:-1]
                )
                or len(identifier_parts[-1]) < 1
                or not re.search(r"\d", identifier_parts[-1])
                or len(finite_boxes) != len(identifier_lines)
                or any(
                    abs(center - centers_x[0]) > max(7.0, widths[0] * 0.2)
                    for center in centers_x[1:]
                )
                or any(width < widths[0] * 0.6 for width in widths[1:-1])
                or any(
                    not 0.65
                    <= (len(part) / width) / (len(identifier_parts[0]) / widths[0])
                    <= 1.55
                    for part, width in zip(
                        identifier_parts[1:-1],
                        widths[1:-1],
                        strict=True,
                    )
                )
            ):
                identifier_invalid = True
        raw = "".join(identifier_parts)
        value = (
            _typed_identifier(raw)
            if not identifier_invalid and 12 <= len(raw) <= 80
            else None
        )
        observations["account_identifier"].append(
            {
                "raw": raw,
                "value": value,
                "source_ref": _account_evidence_source_ref(
                    identifier_lines,
                    field_name="account_identifier",
                ),
            }
        )

    for field_name in ("open_date", "due_date"):
        if field_name not in geometry_fields or field_name in (
            source_absent_roles | mixed_absence_roles
        ):
            continue
        for line in by_role.get(field_name, ()):
            raw = _clean(line.get("text") or "")
            if field_name == "due_date" and _compact(raw) == "长期":
                observations[field_name].append(
                    {
                        "raw": raw,
                        "value": None,
                        "perpetual": True,
                        "source_ref": _account_evidence_source_ref(
                            (line,),
                            field_name=field_name,
                        ),
                    }
                )
                continue
            value = _date(raw)
            if value is None or len(value) != 10:
                continue
            observations[field_name].append(
                {
                    "raw": raw,
                    "value": value,
                    "source_ref": _account_evidence_source_ref(
                        (line,),
                        field_name=field_name,
                    ),
                }
            )

    for field_name in ("loan_amount", "credit_limit", "shared_credit_limit"):
        if field_name not in geometry_fields:
            continue
        if field_name in source_absent_roles | mixed_absence_roles:
            continue
        lines = by_role.get(field_name, ())
        if not lines:
            continue
        raw = " ".join(_clean(line.get("text") or "") for line in lines)
        money_raw = _unique_account_money_token(raw)
        value = _number(money_raw) if money_raw is not None else None
        residue = (
            _account_cluster_residue(raw, (money_raw,))
            if money_raw is not None
            else _account_cluster_residue(raw, ())
        )
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or residue
            or any(line.get("_account_geometry_spanning_roles") for line in lines)
        ):
            value = None
        observations[field_name].append(
            {
                "raw": raw,
                "value": value,
                "source_ref": _account_evidence_source_ref(
                    lines,
                    field_name=field_name,
                ),
            }
        )

    for line in by_role.get("currency", ()):
        if "currency" in source_absent_roles | mixed_absence_roles:
            break
        raw = _clean(line.get("text") or "")
        value, residue, resolution = _currency_token(raw)
        if value is None:
            continue
        observations["currency"].append(
            {
                "raw": raw,
                "value": value,
                "residue": residue,
                "resolution": resolution,
                "source_ref": _account_evidence_source_ref(
                    (line,),
                    field_name="currency",
                ),
            }
        )

    for field_name, contract_role in (
        ("business_type", "account_business_type"),
        ("guarantee_type", "guarantee_type"),
    ):
        if field_name not in geometry_fields:
            continue
        lines = by_role.get(field_name, ())
        if not lines or field_name in source_absent_roles | mixed_absence_roles:
            continue
        raw = " ".join(_clean(line.get("text") or "") for line in lines)
        candidate = _unique_account_finite_span(raw, contract_role=contract_role)
        value: str | None = None
        residue = ""
        if candidate is not None:
            value, source_token, start, end = candidate
            marker = _clean(raw).translate(str.maketrans({"（": "(", "）": ")"}))
            remaining = f"{marker[:start]} {marker[end:]}"
            # Two distinct values from the same finite role in one geometry
            # slot are a conflict, not a longest-string selection.
            if _unique_account_finite_span(
                remaining,
                contract_role=contract_role,
            ) is not None:
                value = None
            else:
                residue = _account_cluster_residue(raw, (source_token,))
        observations[field_name].append(
            {
                "raw": raw,
                "value": value,
                "residue": residue,
                "source_ref": _account_evidence_source_ref(
                    lines,
                    field_name=field_name,
                ),
            }
        )
    for field_name in geometry_fields:
        if field_name not in header_lines_by_role or observations.get(field_name):
            continue
        header_line = header_lines_by_role[field_name]
        observations[field_name].append(
            {
                "raw": "",
                "value": None,
                "source_ref": _account_evidence_source_ref(
                    (header_line,),
                    field_name=field_name,
                ),
            }
        )
    return True, dict(observations)


def _account_loan_second_row_geometry_observations(
    detail: list[dict[str, Any]],
) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
    """Decode the distinct canonical loan classification/repayment row."""

    account_type = str((detail[0] if detail else {}).get("account_type") or "")
    if account_type not in {
        "non_revolving_loan",
        "revolving_loan_subaccount",
        "revolving_loan_account",
    }:
        return False, {}
    template = _ACCOUNT_LOAN_SECOND_ROW_TEMPLATE
    geometry_fields = frozenset(template)
    candidates = _account_basic_header_clusters(detail, expected_roles=template)
    if not candidates:
        return False, {}

    anchor = detail[0]
    anchor_bbox = _account_evidence_bbox(anchor)
    anchor_page = int(anchor.get("page") or anchor.get("logical_page") or 0)
    eligible = [
        candidate
        for candidate in candidates
        if anchor_bbox is not None
        and anchor_page
        and (
            int(
                candidate[0][0][1].get("page")
                or candidate[0][0][1].get("logical_page")
                or 0
            )
            > anchor_page
            or (
                int(
                    candidate[0][0][1].get("page")
                    or candidate[0][0][1].get("logical_page")
                    or 0
                )
                == anchor_page
                and min(item[3][1] for item in candidate[0]) + 2.0
                >= anchor_bbox[3]
            )
        )
    ]
    populated = [
        candidate
        for candidate in eligible
        if _account_header_value_population(detail, *candidate) >= 2
    ]
    if len(eligible) == 1:
        header, header_end = eligible[0]
    elif len(populated) == 1:
        header, header_end = populated[0]
    else:
        invalid_lines = [
            item[1]
            for candidate, _end in (eligible or candidates)
            for item in candidate
        ]
        raw = " ".join(_clean(line.get("text") or "") for line in invalid_lines)
        return True, {
            field_name: [
                {
                    "raw": raw,
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        invalid_lines,
                        field_name=field_name,
                    ),
                }
            ]
            for field_name in geometry_fields
        }

    ordered = sorted(header, key=lambda item: (item[3][0] + item[3][2]) / 2.0)
    role_order = {role: index for index, role in enumerate(template)}
    if [role_order[item[2]] for item in ordered] != list(range(len(template))):
        return True, {
            field_name: [
                {
                    "raw": "",
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        (line,),
                        field_name=field_name,
                    ),
                }
            ]
            for _index, line, field_name, _bbox in header
        }
    centers = [(item[3][0] + item[3][2]) / 2.0 for item in ordered]
    intervals: dict[str, tuple[float, float]] = {}
    for index, item in enumerate(ordered):
        left = (
            float("-inf")
            if index == 0
            else (centers[index - 1] + centers[index]) / 2.0
        )
        right = (
            float("inf")
            if index + 1 == len(ordered)
            else (centers[index] + centers[index + 1]) / 2.0
        )
        intervals[item[2]] = (left, right)

    header_page = int(
        header[0][1].get("page") or header[0][1].get("logical_page") or 0
    )
    header_y = max((item[3][1] + item[3][3]) / 2.0 for item in header)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in detail[header_end + 1 :]:
        compact = _compact(line.get("text") or "")
        if (
            compact.startswith("截至")
            or compact in _ACCOUNT_BASIC_HEADER_ROLES
            or compact in _ACCOUNT_BASIC_ROW_BOUNDARY_LABELS
        ):
            break
        bbox = _account_evidence_bbox(line)
        if bbox is None:
            continue
        line_page = int(line.get("page") or line.get("logical_page") or 0)
        if line_page < header_page:
            continue
        if line_page == header_page and (bbox[1] + bbox[3]) / 2.0 <= header_y:
            continue
        covered_roles = tuple(
            item[2]
            for item, center in zip(ordered, centers, strict=True)
            if bbox[0] <= center <= bbox[2]
        )
        if len(covered_roles) > 1:
            annotated = {**line, "_account_geometry_spanning_roles": covered_roles}
            for role in covered_roles:
                by_role[role].append(annotated)
            continue
        center = (bbox[0] + bbox[2]) / 2.0
        for role, (left, right) in intervals.items():
            if left <= center < right:
                by_role[role].append(line)
                break

    header_by_role = {role: line for _index, line, role, _bbox in header}
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for field_name in template:
        lines = by_role.get(field_name, ())
        if not lines:
            observations[field_name].append(
                {
                    "raw": "",
                    "value": None,
                    "source_ref": _account_evidence_source_ref(
                        (header_by_role[field_name],),
                        field_name=field_name,
                    ),
                }
            )
            continue
        raw = " ".join(_clean(line.get("text") or "") for line in lines)
        absence = [
            line
            for line in lines
            if is_explicit_source_absence(_clean(line.get("text") or ""))
        ]
        source_ref = _account_evidence_source_ref(lines, field_name=field_name)
        if len(absence) == len(lines):
            observations[field_name].append(
                {
                    "raw": raw,
                    "value": None,
                    "source_absent": True,
                    "source_ref": source_ref,
                }
            )
            continue
        if absence or any(
            line.get("_account_geometry_spanning_roles") for line in lines
        ):
            observations[field_name].append(
                {"raw": raw, "value": None, "source_ref": source_ref}
            )
            continue

        value: Any = None
        residue = ""
        if field_name == "repayment_periods":
            tokens = list(dict.fromkeys(_account_cluster_number_tokens(raw)))
            if len(tokens) == 1:
                parsed = _number(tokens[0])
                if (
                    isinstance(parsed, int)
                    and not isinstance(parsed, bool)
                    and 0 <= parsed <= 1200
                ):
                    residue = _account_cluster_residue(raw, (tokens[0],))
                    if not residue:
                        value = parsed
        else:
            contract_role = _ACCOUNT_CELL_FINITE_ROLES[field_name]
            candidate = _unique_account_finite_span(
                raw,
                contract_role=contract_role,
            )
            if candidate is not None:
                value, source_token, _start, _end = candidate
                residue = _account_cluster_residue(raw, (source_token,))
                if residue:
                    value = None
        observations[field_name].append(
            {
                "raw": raw,
                "value": value,
                "residue": residue,
                "source_ref": source_ref,
            }
        )
    return True, dict(observations)


def _apply_account_anchor_basic_observations(
    parse_result: Any,
    account: dict[str, Any],
    observations: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Validate and merge geometry-bound anchor-interval observations."""

    target_record_id = str(account.get("account_id") or "")
    parser_stage = "candidate_b_account_anchor_header_geometry"
    for field_name, items in observations.items():
        for item in items:
            raw = str(item.get("raw") or "")
            value = item.get("value")
            source_ref = item.get("source_ref") if isinstance(item.get("source_ref"), Mapping) else {}
            if field_name == "due_date" and item.get("perpetual"):
                account.pop("due_date", None)
                account["validity_type"] = "perpetual"
                account.setdefault("canonical_raw", {})["due_date"] = raw
                ref = {**dict(source_ref), "field_name": "due_date"}
                refs = account.setdefault("source_refs_by_field", {}).setdefault(
                    "due_date",
                    [],
                )
                if ref not in refs:
                    refs.append(ref)
                continue
            if item.get("source_absent"):
                _mark_source_absent(account, field_name, raw)
                if field_name == "currency":
                    _mark_source_absent(account, "account_currency", raw)
                continue
            if value in (None, ""):
                _reject_exact_observation(
                    parse_result,
                    account,
                    dataset="credit_accounts",
                    target_record_id=target_record_id,
                    field_name=field_name,
                    raw=raw,
                    source_ref=source_ref,
                    parser_stage=parser_stage,
                )
                if field_name == "currency":
                    _reject_exact_observation(
                        parse_result,
                        account,
                        dataset="credit_accounts",
                        target_record_id=target_record_id,
                        field_name="account_currency",
                        raw=raw,
                        source_ref=source_ref,
                        parser_stage=parser_stage,
                    )
                continue
            _merge_exact_observation(
                parse_result,
                account,
                dataset="credit_accounts",
                target_record_id=target_record_id,
                field_name=field_name,
                value=value,
                raw=raw,
                source_ref=source_ref,
                parser_stage=parser_stage,
            )
            if field_name in {
                "management_institution",
                "business_type",
                "guarantee_type",
                "repayment_frequency",
                "repayment_method",
                "co_borrower_flag",
            } and item.get("residue"):
                _report_account_cluster_residue(
                    parse_result,
                    account,
                    target_record_id=target_record_id,
                    field_names=(field_name,),
                    raw=raw,
                    residue=str(item.get("residue") or ""),
                    source_ref=dict(source_ref),
                )
            if field_name != "currency":
                continue
            _merge_exact_observation(
                parse_result,
                account,
                dataset="credit_accounts",
                target_record_id=target_record_id,
                field_name="account_currency",
                value=value,
                raw=raw,
                source_ref=source_ref,
                parser_stage=parser_stage,
            )
            if item.get("resolution") == "residue":
                _report_currency_residue(
                    parse_result,
                    dataset="credit_accounts",
                    target_record_id=target_record_id,
                    raw=raw,
                    currency=str(value),
                    residue=str(item.get("residue") or ""),
                    source_refs=(source_ref,),
                    parser_stage=parser_stage,
                )


def _account_identifier_from_detail(detail: list[dict[str, Any]]) -> str | None:
    """Recover one identifier from the canonical account table row text.

    Whole-page OCR commonly emits the ten-character identifier prefix as its
    own line and the numeric suffix inside the following multi-cell row.  The
    printed open date is the canonical boundary that prevents credit limits or
    other numeric fields from being absorbed into the identifier.
    """
    header = next(
        (
            index
            for index, line in enumerate(detail)
            if "账户标识" in _compact(str(line.get("text") or ""))
        ),
        None,
    )
    if header is None:
        return None

    prefix = ""
    suffix_parts: list[str] = []
    saw_date_boundary = False
    for line in detail[header + 1 : header + 9]:
        text = str(line.get("text") or "").upper()
        if "授信协议标识" in _compact(text):
            continue
        date_match = re.search(r"(?:19|20)\d{2}\s*(?:[./,，-]|年)\s*\d{1,2}", text)
        before_date = text[: date_match.start()] if date_match else text
        matches = list(_ACCOUNT_IDENTIFIER_PREFIX_RE.finditer(before_date))
        if matches:
            if prefix or len(matches) != 1:
                return None
            prefix = matches[0].group(1).upper()
            before_date = before_date[matches[0].end() :]
        if prefix:
            suffix_parts.extend(match.group(0) for match in _ACCOUNT_IDENTIFIER_SUFFIX_RE.finditer(before_date))
        if date_match and prefix:
            saw_date_boundary = True
            break

    candidate = prefix + "".join(suffix_parts)
    if saw_date_boundary and 24 <= len(candidate) <= 80 and re.fullmatch(r"[A-Z0-9]+", candidate):
        return candidate
    return None


def _account_page_table_evidence(
    parse_result: Any,
    logical_page: int,
) -> tuple[tuple[str, str] | None, bool]:
    """Return one exact page-local family or a compatible shared-R1/R2 carry.

    Family headings may legitimately be followed by an account-bearing page
    whose OCR line plane contains only cell values.  A closed native account
    table is sufficient to start or refresh that family, while an empty page
    remains a hard boundary.  Shared R1/R2 signatures are deliberately not
    classified here because their subtype is not page-local.
    """

    target = int(logical_page or 0)
    if target <= 0:
        return None, False
    families: set[str] = set()
    account_base_count = 0
    shared_revolving_count = 0
    shared_revolving_has_identity = False
    for page in getattr(parse_result, "pages", None) or ():
        if int(getattr(page, "page_number", 0) or 0) != target:
            continue
        for table in getattr(page, "tables", None) or ():
            rows = _table_rows(table)
            if not rows or _other_entity_table(rows):
                continue
            if not _account_base(rows):
                continue
            account_base_count += 1
            compact = _compact(" ".join(cell for row in rows[:6] for cell in row))
            if "\u53d1\u5361\u673a\u6784" in compact:
                families.add(
                    "credit_card"
                    if "\u4e1a\u52a1\u79cd\u7c7b" in compact
                    else "quasi_credit_card"
                )
            elif "\u8d26\u6237\u6388\u4fe1\u989d\u5ea6" in compact:
                shared_revolving_count += 1
                shared_revolving_has_identity = (
                    shared_revolving_has_identity
                    or any(
                        bool(_canonical_pboc_account_identifier(cell))
                        for row in rows
                        for cell in row
                    )
                )
            else:
                families.add("non_revolving_loan")
    if len(families) == 1 and shared_revolving_count == 0:
        return (next(iter(families)), "exact"), False
    shared_revolving_carry = bool(
        not families
        and account_base_count > 0
        and shared_revolving_count == account_base_count
        and shared_revolving_has_identity
    )
    return None, shared_revolving_carry


def _bounded_revolving_family_carry_over_generic_table(
    parse_result: Any,
    *,
    page: Mapping[str, Any],
    active_family: str,
    active_family_quality: str,
    active_family_logical_page: int,
    active_family_last_ordinal: int,
    local_table_family: tuple[str, str] | None,
    cross_page_order_resolved: bool,
) -> bool:
    """Keep an exact printed R1/R2 family across one generic loan page.

    The six-column management-institution/identifier/date/amount morphology is
    shared by non-revolving loans and Ye's later R1 rows.  It therefore cannot
    override an already exact printed revolving family on its own.  The carry
    remains deliberately narrow: the registered page edge must be adjacent,
    the account anchors before the next family heading must be the dense next
    ordinals, and exactly the same number of generic account-base tables must
    occur before that heading.
    """

    logical_page = int(page.get("page") or 0)
    if (
        active_family not in {
            "revolving_loan_subaccount",
            "revolving_loan_account",
        }
        or active_family_quality != "exact"
        or active_family_last_ordinal <= 0
        or local_table_family != ("non_revolving_loan", "exact")
        or not cross_page_order_resolved
        or not _registered_account_pages_are_adjacent(
            parse_result,
            active_family_logical_page,
            logical_page,
        )
    ):
        return False

    prefix_ordinals: list[int] = []
    next_heading_top: float | None = None
    for line in page.get("lines") or ():
        if not isinstance(line, Mapping):
            continue
        raw_text = str(line.get("text") or line.get("content") or "")
        compact = _compact(raw_text)
        if any(marker in compact for marker in _ACCOUNT_SECTION_END):
            break
        if _account_family_from_heading(compact) is not None:
            bbox = _account_evidence_bbox(line)
            if bbox is None:
                return False
            next_heading_top = bbox[1]
            break
        anchor = _ACCOUNT_ANCHOR_RE.search(raw_text)
        if anchor is None:
            continue
        if not anchor.group(1):
            return False
        prefix_ordinals.append(int(anchor.group(1)))

    expected = list(
        range(
            active_family_last_ordinal + 1,
            active_family_last_ordinal + 1 + len(prefix_ordinals),
        )
    )
    if not prefix_ordinals or prefix_ordinals != expected:
        return False

    generic_table_count = 0
    for native_page in getattr(parse_result, "pages", None) or ():
        if int(getattr(native_page, "page_number", 0) or 0) != logical_page:
            continue
        for table in getattr(native_page, "tables", None) or ():
            rows = _table_rows(table)
            if not rows or _other_entity_table(rows) or not _account_base(rows):
                continue
            if next_heading_top is not None:
                table_top = _table_top_value(table)
                if table_top is None:
                    return False
                if table_top >= next_heading_top:
                    continue
            generic_table_count += 1
    return generic_table_count == len(prefix_ordinals)


def _account_segment_has_exact_two_cell_card_table(
    parse_result: Any,
    *,
    account_type: str,
    page_segments: Iterable[Mapping[str, Any]],
) -> bool:
    """Identify the unique table whose closed geometry owns a card segment."""

    if account_type not in {"credit_card", "quasi_credit_card"}:
        return False
    segments = {
        int(segment.get("logical_page") or 0): segment
        for segment in page_segments
        if int(segment.get("logical_page") or 0) > 0
    }
    if not segments:
        return False
    candidates: list[Any] = []
    for page in getattr(parse_result, "pages", None) or ():
        logical_page = int(getattr(page, "page_number", 0) or 0)
        segment = segments.get(logical_page)
        if segment is None:
            continue
        for table in getattr(page, "tables", None) or ():
            rows = _table_rows(table)
            if _exact_two_cell_card_cluster_values(table, rows) is None:
                continue
            table_top = _table_top_value(table)
            if table_top is None:
                continue
            try:
                minimum = float(segment.get("min_y") or 0.0)
                maximum = (
                    float(segment["max_y"])
                    if segment.get("max_y") is not None
                    else None
                )
            except (TypeError, ValueError):
                continue
            if table_top + 1.0 < minimum:
                continue
            if maximum is not None and table_top >= maximum:
                continue
            candidates.append(table)
    return len(candidates) == 1


def _account_anchor_skeletons(parse_result: Any) -> list[dict[str, Any]]:
    """Build the canonical account row skeleton from printed account anchors."""
    cached = getattr(parse_result, "_candidate_b_account_anchor_skeleton_cache", None)
    if isinstance(cached, list):
        return deepcopy(cached)
    evidence_loader = getattr(parse_result, "corrected_evidence_pages", None)
    if not callable(evidence_loader):
        return []
    flattened: list[dict[str, Any]] = []
    active_type = ""
    active_family_quality = ""
    active_family_logical_page = 0
    active_family_last_ordinal = 0
    evidence_pages = list(evidence_loader())
    cross_page_order_resolved = _account_reading_order_resolution(
        parse_result, evidence_pages
    )[-1]
    for page_offset, page in enumerate(
        _account_ordered_pages(parse_result, evidence_pages)
    ):
        logical_page = int(page.get("page") or 0)
        local_table_family, shared_revolving_carry = _account_page_table_evidence(
            parse_result,
            logical_page,
        )
        family_carry_allowed = bool(
            logical_page > 0
            and active_family_logical_page > 0
            and (
                logical_page == active_family_logical_page
                or (
                    cross_page_order_resolved
                    and _registered_account_pages_are_adjacent(
                        parse_result,
                        active_family_logical_page,
                        logical_page,
                    )
                )
            )
        )
        if page_offset > 0 and active_type and not family_carry_allowed:
            active_type = ""
            active_family_quality = ""
            active_family_logical_page = 0
            active_family_last_ordinal = 0
        preserve_revolving_family = _bounded_revolving_family_carry_over_generic_table(
            parse_result,
            page=page,
            active_family=active_type,
            active_family_quality=active_family_quality,
            active_family_logical_page=active_family_logical_page,
            active_family_last_ordinal=active_family_last_ordinal,
            local_table_family=local_table_family,
            cross_page_order_resolved=cross_page_order_resolved,
        )
        if local_table_family is not None and not preserve_revolving_family:
            active_type, active_family_quality = local_table_family
            active_family_logical_page = logical_page
            active_family_last_ordinal = 0
        elif preserve_revolving_family:
            active_family_logical_page = logical_page
        elif active_type in {
            "revolving_loan_subaccount",
            "revolving_loan_account",
        } and shared_revolving_carry:
            active_family_logical_page = logical_page
        lines = [line for line in page.get("lines") or () if isinstance(line, dict)]
        for index, line in enumerate(lines):
            text = str(line.get("text") or line.get("content") or "")
            compact = _compact(text)
            if active_type and any(marker in compact for marker in _ACCOUNT_SECTION_END):
                active_type = ""
                active_family_quality = ""
                active_family_logical_page = 0
                active_family_last_ordinal = 0
            family = _account_family_from_heading(compact)
            marker = family[0] if family else None
            if marker is not None:
                active_type = marker
                active_family_quality = family[1]
                active_family_logical_page = logical_page
                active_family_last_ordinal = 0
            anchor_match = _ACCOUNT_ANCHOR_RE.search(text)
            if active_type and anchor_match:
                active_family_logical_page = logical_page
                if anchor_match.group(1):
                    active_family_last_ordinal = int(anchor_match.group(1))
            flattened.append(
                {
                    **line,
                    "text": text,
                    "page": logical_page,
                    "source_page": int(page.get("source_page") or 0),
                    "account_type": active_type,
                    "account_family_quality": active_family_quality,
                    "line_index": index,
                }
            )

    starts = [
        index
        for index, line in enumerate(flattened)
        if line.get("account_type") and _ACCOUNT_ANCHOR_RE.search(str(line.get("text") or ""))
    ]
    printed_ordinals: list[int | None] = []
    ordinal_counts: Counter[tuple[str, int]] = Counter()
    for start in starts:
        anchor = flattened[start]
        match = _ACCOUNT_ANCHOR_RE.search(str(anchor.get("text") or ""))
        detected = int(match.group(1)) if match and match.group(1) else None
        printed_ordinals.append(detected)
        if detected is not None and detected > 0:
            ordinal_counts[(str(anchor.get("account_type") or ""), detected)] += 1

    transition_check = getattr(parse_result, "allows_scanned_line_transition", None)
    skeletons: list[dict[str, Any]] = []
    for position, start in enumerate(starts):
        anchor = flattened[start]
        account_type = str(anchor["account_type"])
        next_start = starts[position + 1] if position + 1 < len(starts) else len(flattened)
        end = next_start
        for index in range(start + 1, next_start):
            if flattened[index].get("account_type") != account_type:
                end = index
                break
        # A physical/logical page boundary belongs to the account segment only
        # when the canonical entity context affirms the evidence transition.
        # Adjacency by itself is deliberately insufficient.
        verified_end = end
        for index in range(start + 1, end):
            left = flattened[index - 1]
            right = flattened[index]
            left_page = int(left.get("page") or 0)
            right_page = int(right.get("page") or 0)
            if left_page == right_page:
                continue
            decision = None
            if cross_page_order_resolved and callable(transition_check):
                decision = transition_check(
                    left_page,
                    left,
                    int(left.get("line_index") or 0),
                    right_page,
                    right,
                    int(right.get("line_index") or 0),
                )
            if decision is not True:
                verified_end = index
                break
        detail = flattened[start:verified_end]
        boundary = flattened[verified_end] if verified_end < len(flattened) else None

        page_segments: list[dict[str, Any]] = []
        segment_pages: list[int] = []
        for line in detail:
            page_number = int(line.get("page") or 0)
            if page_number and page_number not in segment_pages:
                segment_pages.append(page_number)
        anchor_bbox = anchor.get("bbox")
        anchor_top = (
            float(anchor_bbox[1])
            if isinstance(anchor_bbox, (list, tuple)) and len(anchor_bbox) == 4
            else None
        )
        for page_index, page_number in enumerate(segment_pages):
            upper: float | None = None
            if boundary is not None and int(boundary.get("page") or 0) == page_number:
                boundary_bbox = boundary.get("bbox")
                if isinstance(boundary_bbox, (list, tuple)) and len(boundary_bbox) == 4:
                    upper = float(boundary_bbox[1])
            page_segments.append(
                {
                    "logical_page": page_number,
                    "min_y": anchor_top if page_index == 0 else 0.0,
                    "max_y": upper,
                    "continuation_verified": page_index > 0,
                }
            )

        detected = printed_ordinals[position]
        ordinal_is_unique = bool(
            detected is not None
            and detected > 0
            and ordinal_counts[(account_type, detected)] == 1
        )
        if ordinal_is_unique:
            account_id = f"credit_account:{account_type}:{detected}"
            ordinal_status = "printed_unique"
        else:
            ordinal_status = "printed_duplicate" if detected else "printed_unreadable"
            account_id = stable_record_id(
                "credit_account_provisional",
                account_type,
                anchor.get("source_page"),
                anchor.get("page"),
                anchor.get("bbox"),
                anchor.get("evidence_ids"),
                anchor.get("line_index"),
                anchor.get("text"),
            )
        skeleton = {
                "account_id": account_id,
                "sequence": len(skeletons) + 1,
                **({"category_sequence": detected} if ordinal_is_unique else {}),
                **_account_heading_fields(anchor.get("text")),
                "account_type": account_type,
                "account_family_quality": str(anchor.get("account_family_quality") or ""),
                "_printed_ordinal_status": ordinal_status,
                "_canonical_segment": {
                    "ownership_basis": "printed_anchor_to_next_anchor",
                    "anchor_logical_page": int(anchor.get("page") or 0),
                    "anchor_bbox": list(anchor.get("bbox") or ()),
                    "pages": page_segments,
                    "cross_page_continuation_verified": len(page_segments) > 1,
                },
                "account_state": "unknown",
                "source": "candidate_b_account_anchor",
                "page": int(anchor.get("page") or 0),
                "source_page": int(anchor.get("source_page") or 0),
                "bbox": list(anchor.get("bbox") or ()),
                "source_refs": [
                    {
                        "source": "candidate_b_account_anchor",
                        "logical_page": int(anchor.get("page") or 0),
                        "source_page": int(anchor.get("source_page") or 0),
                        "bbox": list(anchor.get("bbox") or ()),
                        "evidence_ids": list(anchor.get("evidence_ids") or ()),
                    }
                ],
                "raw_detail_lines": [
                    {
                        "logical_page": int(line.get("page") or 0),
                        "source_page": int(line.get("source_page") or 0),
                        "text": str(line.get("text") or ""),
                        "bbox": list(line.get("bbox") or ()),
                        "evidence_ids": list(line.get("evidence_ids") or ()),
                    }
                    for line in detail
                ],
                "confidence": float(anchor.get("confidence") or 0.0),
            }
        exact_two_cell_card_owner = _account_segment_has_exact_two_cell_card_table(
            parse_result,
            account_type=account_type,
            page_segments=page_segments,
        )
        if exact_two_cell_card_owner:
            # The table's exact two-cell lattice is the stronger source plane.
            # Do not let weak line fragments pre-empt it with invalid/XAU slots.
            header_found, basic_observations = True, {}
        else:
            header_found, basic_observations = _account_basic_geometry_observations(
                detail
            )
        _apply_account_anchor_basic_observations(
            parse_result,
            skeleton,
            basic_observations,
        )
        _second_row_found, second_row_observations = (
            _account_loan_second_row_geometry_observations(detail)
        )
        _apply_account_anchor_basic_observations(
            parse_result,
            skeleton,
            second_row_observations,
        )
        reconstructed_identifier = (
            None if header_found else _account_identifier_from_detail(detail)
        )
        if reconstructed_identifier and not skeleton.get("account_identifier"):
            skeleton["account_identifier"] = reconstructed_identifier
            skeleton["account_identifier_source"] = "canonical_anchor_table_row"
        elif skeleton.get("account_identifier"):
            skeleton["account_identifier_source"] = "canonical_anchor_header_geometry"
        skeletons.append(skeleton)
        if not ordinal_is_unique:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_account_printed_ordinal_unresolved",
                    message=(
                        "A printed account ordinal was missing or duplicated; encounter order was not emitted "
                        "as the account family sequence and a source-stable provisional identity was used."
                    ),
                    parser_stage="candidate_b_account_anchor_ownership",
                    target_dataset="credit_accounts",
                    target_record_id=account_id,
                    field_name="category_sequence",
                    observed_value={
                        "account_type": account_type,
                        "printed_ordinal": detected,
                        "ordinal_status": ordinal_status,
                    },
                    source_refs=skeleton.get("source_refs") or (),
                    reason_codes=(
                        "printed_ordinal_duplicate" if detected else "printed_ordinal_unreadable",
                        "encounter_order_not_used",
                        "stable_provisional_identity",
                        "normalized_ordinal_withheld",
                    ),
                ),
            )
    try:
        setattr(
            parse_result,
            "_candidate_b_account_anchor_skeleton_cache",
            deepcopy(skeletons),
        )
    except (AttributeError, TypeError):
        pass
    return skeletons


def _account_stream_position(record: dict[str, Any]) -> tuple[int, float] | None:
    """Return the canonical logical-page position of an account observation."""
    page = int(record.get("page") or 0)
    bbox = record.get("bbox")
    if not page:
        for ref in record.get("source_refs") or ():
            if not isinstance(ref, dict):
                continue
            page = int(ref.get("logical_page") or 0)
            if not bbox:
                bbox = ref.get("bbox")
            if page:
                break
    if not page or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        top = float(bbox[1])
    except (TypeError, ValueError):
        return None
    return page, top


def _match_account_table_observations(
    skeletons: list[dict[str, Any]],
    table_accounts: list[dict[str, Any]],
) -> dict[int, int]:
    """Match tables only inside an anchor-owned canonical stream segment.

    The exact ownership interval begins at the printed anchor and ends at the
    next printed anchor (or the registered section boundary).  Later logical
    pages are eligible only when ``_account_anchor_skeletons`` recorded an
    affirmative continuation transition.  Account-family compatibility is a
    mandatory type invariant; within one family, encounter order and a
    globally nearest predecessor are deliberately not identity keys.
    """
    matches: dict[int, int] = {}
    consumed_tables: set[int] = set()
    fallback_segments: dict[int, tuple[int, float, float | None]] = {}
    positioned_skeletons = [
        (index, position)
        for index, skeleton in enumerate(skeletons)
        if (position := _account_stream_position(skeleton)) is not None
    ]
    for offset, (skeleton_index, (page, top)) in enumerate(positioned_skeletons):
        upper: float | None = None
        for _next_index, (next_page, next_top) in positioned_skeletons[offset + 1 :]:
            if next_page == page:
                upper = next_top
                break
            if next_page > page:
                break
        fallback_segments[skeleton_index] = (page, top, upper)

    def owns(skeleton_index: int, page: int, top: float) -> bool:
        skeleton = skeletons[skeleton_index]
        segment = skeleton.get("_canonical_segment")
        pages = segment.get("pages") if isinstance(segment, Mapping) else None
        if isinstance(pages, list):
            for page_segment in pages:
                if not isinstance(page_segment, Mapping):
                    continue
                if int(page_segment.get("logical_page") or 0) != page:
                    continue
                minimum = page_segment.get("min_y")
                maximum = page_segment.get("max_y")
                if minimum is None:
                    return False
                try:
                    lower = float(minimum)
                    upper = float(maximum) if maximum is not None else None
                except (TypeError, ValueError):
                    return False
                return top + 8.0 >= lower and (upper is None or top < upper)
            return False
        fallback = fallback_segments.get(skeleton_index)
        if fallback is None:
            return False
        segment_page, lower, upper = fallback
        return page == segment_page and top + 8.0 >= lower and (upper is None or top < upper)

    # A headerless next-page table may sit outside the anchor segment when the
    # table graph did not preserve the cross-page edge.  The extractor records
    # this private linkage only after proving an exact pending printed anchor,
    # complete card header, adjacent registered pages, and a distinct strong
    # identifier.  Consume that proof before the general geometry matcher.
    for table_index, table in enumerate(table_accounts):
        pending_anchor_id = str(table.get("_pending_anchor_account_id") or "")
        if not pending_anchor_id:
            continue
        candidates = [
            skeleton_index
            for skeleton_index, skeleton in enumerate(skeletons)
            if str(skeleton.get("account_id") or "") == pending_anchor_id
            and _owned_account_table_family_is_compatible(skeleton, table)
            and (
                not _account_card_identifier(skeleton)
                or _account_card_identifier(skeleton) == _account_card_identifier(table)
            )
        ]
        if len(candidates) != 1:
            continue
        skeleton_index = candidates[0]
        if skeleton_index in matches:
            continue
        matches[skeleton_index] = table_index
        consumed_tables.add(table_index)

    positioned_tables = sorted(
        (
            (position, table_index)
            for table_index, table in enumerate(table_accounts)
            if (position := _account_stream_position(table)) is not None
        ),
        key=lambda item: item[0],
    )
    owned_family_candidates: dict[int, list[int]] = defaultdict(list)
    for (page, top), table_index in positioned_tables:
        table = table_accounts[table_index]
        for skeleton_index, skeleton in enumerate(skeletons):
            if not owns(skeleton_index, page, top):
                continue
            if _owned_account_table_family_is_compatible(
                skeleton,
                table,
            ) or _exact_printed_revolving_anchor_signature_candidate(
                skeleton,
                table,
            ):
                owned_family_candidates[skeleton_index].append(table_index)

    def compatible_in_owned_interval(
        skeleton_index: int,
        table_index: int,
    ) -> bool:
        skeleton = skeletons[skeleton_index]
        table = table_accounts[table_index]
        if _owned_account_table_family_is_compatible(skeleton, table):
            return True
        if not _owned_account_table_family_is_compatible(
            skeleton,
            table,
            exact_interval_owned=True,
        ):
            return False
        # The interval-only alias is deliberately unavailable when any table
        # that could directly bind this family, or any other named alias,
        # competes inside the same printed anchor's physical segment. Direct
        # exact-family/strong-identity matching above remains independent of
        # this positional uniqueness proof.
        return owned_family_candidates.get(skeleton_index) == [table_index]

    for (page, top), table_index in positioned_tables:
        if table_index in consumed_tables:
            continue
        table_type = str(table_accounts[table_index].get("account_type") or "")
        candidates = [
            skeleton_index
            for skeleton_index in range(len(skeletons))
            if owns(skeleton_index, page, top)
            and table_type
            and compatible_in_owned_interval(skeleton_index, table_index)
        ]
        if len(candidates) != 1:
            continue
        skeleton_index = candidates[0]
        if skeleton_index in matches or table_index in consumed_tables:
            continue
        matches[skeleton_index] = table_index
        consumed_tables.add(table_index)
    return matches


def _account_card_identifier(record: Mapping[str, Any]) -> str:
    """Return one strong account-card identity, never an observation ID."""

    marker = re.sub(
        r"[^0-9A-Z]",
        "",
        str(record.get("account_identifier") or "").upper(),
    )
    return marker if len(marker) >= 12 else ""


_REVOLVING_ACCOUNT_FAMILY_PAIR = frozenset(
    {"revolving_loan_subaccount", "revolving_loan_account"}
)


def _exact_printed_revolving_anchor_signature_candidate(
    skeleton: Mapping[str, Any],
    table: Mapping[str, Any],
) -> bool:
    """Recognize one bounded native signature eligible for interval resolution."""

    anchor_type = str(skeleton.get("account_type") or "")
    table_type = str(table.get("account_type") or "")
    if (
        anchor_type not in _REVOLVING_ACCOUNT_FAMILY_PAIR
        or skeleton.get("account_family_quality") != "exact"
        or skeleton.get("_printed_ordinal_status") != "printed_unique"
        or skeleton.get("source") != "candidate_b_account_anchor"
    ):
        return False
    segment = skeleton.get("_canonical_segment")
    if not (
        isinstance(segment, Mapping)
        and segment.get("ownership_basis") == "printed_anchor_to_next_anchor"
        and isinstance(segment.get("pages"), list)
        and segment.get("pages")
    ):
        return False
    if (
        table.get("source") != "native_detail_account_table"
        or not table.get("_table_observation_id")
        or table.get("_pending_anchor_account_id")
    ):
        return False
    anchor_identifier = _account_card_identifier(skeleton)
    table_identifier = _account_card_identifier(table)
    if (
        anchor_identifier
        and table_identifier
        and anchor_identifier != table_identifier
    ):
        return False
    basis = str(table.get("_table_account_family_basis") or "")
    return bool(
        (
            table_type == "non_revolving_loan"
            and basis == "non_revolving_table_signature"
        )
        or (
            table_type in _REVOLVING_ACCOUNT_FAMILY_PAIR
            and basis == "shared_revolving_credit_limit_signature"
        )
    )


def _owned_account_table_family_is_compatible(
    skeleton: Mapping[str, Any],
    table: Mapping[str, Any],
    *,
    exact_interval_owned: bool = False,
) -> bool:
    """Accept source-owned exact-family tables or one tightly proven alias.

    The canonical R1 and R2 first rows both use ``账户授信额度``.  Native table
    morphology can therefore identify the revolving-loan superfamily without
    distinguishing its printed variant. Strong matching remains available when
    both observations carry the same account identifier. Separately, the
    caller may prove a unique native table lies inside an exact printed R1/R2
    anchor interval; only the two named native signatures can use that path.
    """

    anchor_type = str(skeleton.get("account_type") or "")
    table_type = str(table.get("account_type") or "")
    if not anchor_type or not table_type:
        return False
    if anchor_type == table_type:
        return True
    if frozenset({anchor_type, table_type}) == _REVOLVING_ACCOUNT_FAMILY_PAIR:
        if (
            skeleton.get("account_family_quality") == "exact"
            and table.get("source") == "native_detail_account_table"
            and table.get("_table_observation_id")
            and table.get("_table_account_family_basis")
            == "shared_revolving_credit_limit_signature"
        ):
            anchor_identifier = _account_card_identifier(skeleton)
            table_identifier = _account_card_identifier(table)
            if anchor_identifier and anchor_identifier == table_identifier:
                return True
    return bool(
        exact_interval_owned
        and _exact_printed_revolving_anchor_signature_candidate(skeleton, table)
    )


def _resolve_owned_revolving_table_families(
    skeletons: list[dict[str, Any]],
    table_accounts: list[dict[str, Any]],
    table_matches: Mapping[int, int],
) -> None:
    """Let an exact printed R1/R2 anchor resolve one owned native signature."""

    for skeleton_index, table_index in table_matches.items():
        skeleton = skeletons[skeleton_index]
        table = table_accounts[table_index]
        anchor_type = str(skeleton.get("account_type") or "")
        table_type = str(table.get("account_type") or "")
        if anchor_type == table_type:
            continue
        strong_identity_resolution = _owned_account_table_family_is_compatible(
            skeleton,
            table,
        )
        if not strong_identity_resolution and not (
            _owned_account_table_family_is_compatible(
                skeleton,
                table,
                exact_interval_owned=True,
            )
        ):
            continue
        table["_table_account_type_candidate"] = table_type
        table["_table_account_family_resolution"] = (
            "exact_anchor_strong_identifier_in_owned_interval"
            if strong_identity_resolution
            else "exact_printed_anchor_unique_native_signature_interval"
        )
        table["account_type"] = anchor_type


def _account_card_source_boxes(
    record: Mapping[str, Any],
) -> list[tuple[int, tuple[float, float, float, float], str]]:
    """Collect table-scoped source boxes suitable for replay comparison."""

    boxes: list[tuple[int, tuple[float, float, float, float], str]] = []
    refs = list(record.get("source_refs") or ())
    if record.get("bbox"):
        refs.append(
            {
                "logical_page": record.get("page"),
                "source_page": record.get("source_page"),
                "bbox": record.get("bbox"),
            }
        )
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        bbox = ref.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            coordinates = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        page = int(ref.get("source_page") or ref.get("logical_page") or 0)
        if not page:
            continue
        boxes.append((page, coordinates, str(ref.get("table_id") or "")))
    return boxes


def _account_card_boxes_are_replays(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    """Require near-total physical overlap for identifier-free replay proof."""

    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    smaller_area = min(left_area, right_area)
    return bool(smaller_area and intersection / smaller_area >= 0.9)


def _same_native_account_card_replay(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Prove that two native table observations replay one source card."""

    if (
        left.get("source") != "native_detail_account_table"
        or right.get("source") != "native_detail_account_table"
        or not left.get("_table_observation_id")
        or not right.get("_table_observation_id")
        or str(left.get("account_type") or "")
        != str(right.get("account_type") or "")
    ):
        return False
    left_identifier = _account_card_identifier(left)
    right_identifier = _account_card_identifier(right)
    if left_identifier and right_identifier:
        return left_identifier == right_identifier

    for left_page, left_bbox, left_table_id in _account_card_source_boxes(left):
        for right_page, right_bbox, right_table_id in _account_card_source_boxes(right):
            if left_page != right_page:
                continue
            if left_table_id and left_table_id == right_table_id:
                return True
            if _account_card_boxes_are_replays(left_bbox, right_bbox):
                return True
    return False


def _native_account_card_groups(
    table_accounts: list[dict[str, Any]], table_indices: Iterable[int]
) -> list[tuple[int, ...]]:
    """Collapse only transitively proven replays of one native account card."""

    indices = sorted(set(int(index) for index in table_indices))
    remaining = set(indices)
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        frontier = [seed]
        remaining.remove(seed)
        while frontier:
            current = frontier.pop()
            connected = {
                candidate
                for candidate in remaining
                if _same_native_account_card_replay(
                    table_accounts[current], table_accounts[candidate]
                )
            }
            remaining.difference_update(connected)
            component.update(connected)
            frontier.extend(connected)
        groups.append(tuple(sorted(component)))
    return groups


def _canonical_singleton_account_matches(
    skeletons: list[dict[str, Any]],
    table_accounts: list[dict[str, Any]],
    table_matches: Mapping[int, int],
) -> dict[int, int]:
    """Prove the one safe exception to a missing printed family ordinal.

    A canonical PBOC family may omit ``1`` when it contains one account.  The
    ordinal is completed only when the independently decoded table population
    corroborates that exact singleton: one exact-family unnumbered anchor, one
    native same-family table observation, and the existing stream-ownership
    matcher binds those two observations one-to-one.  Encounter order and a
    family label alone are deliberately insufficient.
    """

    skeletons_by_family: defaultdict[str, list[int]] = defaultdict(list)
    tables_by_family: defaultdict[str, list[int]] = defaultdict(list)
    for index, skeleton in enumerate(skeletons):
        account_type = str(skeleton.get("account_type") or "")
        if account_type:
            skeletons_by_family[account_type].append(index)
    for index, table in enumerate(table_accounts):
        account_type = str(table.get("account_type") or "")
        if account_type:
            tables_by_family[account_type].append(index)

    completed: dict[int, int] = {}
    for account_type, skeleton_indices in skeletons_by_family.items():
        table_indices = tables_by_family.get(account_type, [])
        card_groups = _native_account_card_groups(table_accounts, table_indices)
        if len(skeleton_indices) != 1 or len(card_groups) != 1:
            continue
        skeleton_index = skeleton_indices[0]
        card_group = card_groups[0]
        table_index = table_matches.get(skeleton_index)
        if table_index not in card_group:
            continue
        skeleton = skeletons[skeleton_index]
        table = table_accounts[table_index]
        if skeleton.get("account_family_quality") != "exact":
            continue
        if skeleton.get("_printed_ordinal_status") != "printed_unreadable":
            continue
        if skeleton.get("category_sequence") not in (None, ""):
            continue
        if table.get("source") != "native_detail_account_table":
            continue
        if not table.get("_table_observation_id"):
            continue
        completed[skeleton_index] = table_index
    return completed


def _suppress_completed_singleton_ordinal_issue(
    parse_result: Any,
    *,
    account_type: str,
    provisional_account_id: str,
) -> None:
    """Remove only the diagnostic superseded by exact singleton proof."""

    issues = getattr(parse_result, "_personal_detail_extraction_issues", None)
    if not isinstance(issues, list):
        return
    issues[:] = [
        issue
        for issue in issues
        if not (
            isinstance(issue, Mapping)
            and issue.get("issue_code") == "candidate_b_account_printed_ordinal_unresolved"
            and issue.get("field_name") == "category_sequence"
            and str(issue.get("target_record_id") or "") == provisional_account_id
            and str((issue.get("observed_value") or {}).get("account_type") or "")
            == account_type
            and str((issue.get("observed_value") or {}).get("ordinal_status") or "")
            == "printed_unreadable"
        )
    ]


def _mark_account_observation_not_emitted(
    parse_result: Any,
    account_observation_id: str,
) -> None:
    """Give every diagnostic on a suppressed source row a final lifecycle."""

    if not account_observation_id:
        return
    issues = getattr(parse_result, "_personal_detail_extraction_issues", None)
    if not isinstance(issues, list):
        return
    marker = "record_not_emitted_due_to_unresolved_account_ownership"
    for index, issue in enumerate(issues):
        if (
            not isinstance(issue, Mapping)
            or issue.get("target_dataset") != "credit_accounts"
            or str(issue.get("target_record_id") or "")
            != account_observation_id
        ):
            continue
        updated = dict(issue)
        updated["reason_codes"] = list(
            dict.fromkeys(
                (
                    *(
                        str(value)
                        for value in updated.get("reason_codes") or ()
                        if value
                    ),
                    marker,
                )
            )
        )
        issues[index] = updated


def _extract_accounts(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize one account population from anchors enriched by tables."""
    table_accounts, repayments, events = _extract_table_accounts(parse_result)
    skeletons = _account_anchor_skeletons(parse_result)
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    if not skeletons:
        # Preserve otherwise useful canonical cells as explicitly partial
        # observations, but never promote table encounter order to an account
        # family identity when the printed anchors were not recovered.
        for table in table_accounts:
            table.pop("category_sequence", None)
            table["extraction_status"] = "review"
            table["_ownership_status"] = "printed_anchor_missing"
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_account_anchor_population_missing",
                    message=(
                        "Canonical account-table cells were retained as a partial observation, but no printed "
                        "account anchor was available to establish family identity or ordinal."
                    ),
                    parser_stage="candidate_b_account_anchor_ownership",
                    target_dataset="credit_accounts",
                    target_record_id=str(table.get("account_id") or "") or None,
                    field_name="category_sequence",
                    source_refs=table.get("source_refs") or (),
                    reason_codes=(
                        "printed_anchor_missing",
                        "stable_table_observation_identity",
                        "encounter_order_not_used",
                        "record_partial",
                    ),
                ),
            )
        return table_accounts, repayments, events

    table_matches = _match_account_table_observations(skeletons, table_accounts)
    _resolve_owned_revolving_table_families(
        skeletons,
        table_accounts,
        table_matches,
    )
    singleton_matches = _canonical_singleton_account_matches(
        skeletons,
        table_accounts,
        table_matches,
    )
    singleton_replays_by_skeleton: dict[int, tuple[int, ...]] = {}
    for skeleton_index, table_index in singleton_matches.items():
        account_type = str(skeletons[skeleton_index].get("account_type") or "")
        family_indices = [
            index
            for index, table in enumerate(table_accounts)
            if str(table.get("account_type") or "") == account_type
        ]
        replay_group = next(
            (
                group
                for group in _native_account_card_groups(table_accounts, family_indices)
                if table_index in group
            ),
            (table_index,),
        )
        singleton_replays_by_skeleton[skeleton_index] = tuple(
            index for index in replay_group if index != table_index
        )
    singleton_replay_indices = {
        index
        for replay_indices in singleton_replays_by_skeleton.values()
        for index in replay_indices
    }
    emitted: list[dict[str, Any]] = []
    consumed: set[int] = set()
    account_id_remap: dict[str, str] = {}
    for skeleton_index in singleton_matches:
        skeleton = skeletons[skeleton_index]
        account_type = str(skeleton.get("account_type") or "")
        provisional_account_id = str(skeleton.get("account_id") or "")
        canonical_account_id = f"credit_account:{account_type}:1"
        skeleton["account_id"] = canonical_account_id
        skeleton["category_sequence"] = 1
        skeleton["_printed_ordinal_status"] = "canonical_singleton_inferred"
        _suppress_completed_singleton_ordinal_issue(
            parse_result,
            account_type=account_type,
            provisional_account_id=provisional_account_id,
        )
        if provisional_account_id and provisional_account_id != canonical_account_id:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                register_issue_target_remap,
            )

            register_issue_target_remap(
                parse_result,
                provisional_account_id,
                canonical_account_id,
            )
    for skeleton_index, skeleton in enumerate(skeletons):
        table_index = table_matches.get(skeleton_index)
        table = table_accounts[table_index] if table_index is not None else None
        if table is not None:
            record = dict(table)
            # Printed anchors own population identity and canonical ordering;
            # the table contributes only decoded business cells.
            for field_name in (
                "account_id",
                "sequence",
                "category_sequence",
                "account_type",
                "account_family_quality",
                "_printed_ordinal_status",
                "_canonical_segment",
                "page",
                "source_page",
                "bbox",
                "credit_agreement_identifier",
                "card_tail",
            ):
                if field_name in skeleton:
                    record[field_name] = deepcopy(skeleton[field_name])
                else:
                    record.pop(field_name, None)
            anchor_refs = [
                ref for ref in skeleton.get("source_refs") or () if isinstance(ref, Mapping)
            ]
            for field_name in (
                "management_institution",
                "account_identifier",
                "open_date",
                "due_date",
                "loan_amount",
                "credit_limit",
                "shared_credit_limit",
                "currency",
                "account_currency",
                "business_type",
                "guarantee_type",
                "repayment_periods",
                "repayment_frequency",
                "repayment_method",
                "co_borrower_flag",
            ):
                value = skeleton.get(field_name)
                if value in (None, ""):
                    continue
                field_refs = [
                    ref
                    for ref in skeleton.get("source_refs_by_field", {}).get(field_name, ())
                    if isinstance(ref, Mapping)
                ]
                raw = skeleton.get("canonical_raw", {}).get(field_name, value)
                if field_refs:
                    for field_ref in field_refs:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="credit_accounts",
                            target_record_id=str(skeleton.get("account_id") or ""),
                            field_name=field_name,
                            value=value,
                            raw=str(raw),
                            source_ref=field_ref,
                            parser_stage="candidate_b_account_anchor_header_geometry",
                        )
                elif field_name == "account_identifier" and anchor_refs:
                    # Compatibility for the pre-geometry identifier recovery:
                    # the printed anchor is still account-owned, but never used
                    # as provenance for the new geometry-bound fields.
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="credit_accounts",
                        target_record_id=str(skeleton.get("account_id") or ""),
                        field_name=field_name,
                        value=value,
                        raw=str(raw),
                        source_ref={
                            **dict(anchor_refs[0]),
                            "binding": "canonical_account_anchor",
                            "binding_quality": "canonical_account_anchor",
                        },
                        parser_stage="candidate_b_account_canonical_slots",
                    )
                elif field_name == "account_identifier" and field_name not in record.get(
                    "_unresolved_fields", []
                ):
                    record[field_name] = value
                if record.get(field_name) not in (None, ""):
                    unresolved = record.get("_unresolved_fields")
                    if isinstance(unresolved, list) and field_name in unresolved:
                        unresolved.remove(field_name)
            if skeleton.get("validity_type") == "perpetual":
                due_refs = [
                    ref
                    for ref in skeleton.get("source_refs_by_field", {}).get("due_date", ())
                    if isinstance(ref, Mapping)
                ]
                if record.get("due_date") in (None, ""):
                    record["validity_type"] = "perpetual"
                    record.setdefault("canonical_raw", {})["due_date"] = "长期"
                    if due_refs:
                        record.setdefault("source_refs_by_field", {})["due_date"] = due_refs
                else:
                    _reject_exact_observation(
                        parse_result,
                        record,
                        dataset="credit_accounts",
                        target_record_id=str(skeleton.get("account_id") or ""),
                        field_name="due_date",
                        raw="长期",
                        source_ref=due_refs[0] if due_refs else anchor_refs[0],
                        parser_stage="candidate_b_account_anchor_header_geometry",
                    )
                    record["validity_type"] = "unknown"
            geometry_fields = {
                "management_institution",
                "account_identifier",
                "open_date",
                "due_date",
                "loan_amount",
                "credit_limit",
                "shared_credit_limit",
                "currency",
                "account_currency",
                "business_type",
                "guarantee_type",
                "repayment_periods",
                "repayment_frequency",
                "repayment_method",
                "co_borrower_flag",
            }
            for field_name in geometry_fields.intersection(
                skeleton.get("_source_absent_fields") or ()
            ):
                if record.get(field_name) not in (None, ""):
                    absence_refs = [
                        ref
                        for ref in skeleton.get("source_refs_by_field", {}).get(
                            field_name,
                            (),
                        )
                        if isinstance(ref, Mapping)
                    ]
                    _reject_exact_source_absence_conflict(
                        parse_result,
                        record,
                        target_record_id=str(skeleton.get("account_id") or ""),
                        field_name=field_name,
                        raw=str(
                            skeleton.get("canonical_raw", {}).get(field_name, "--")
                        ),
                        source_ref=absence_refs[0] if absence_refs else anchor_refs[0],
                    )
                    continue
                record.pop(field_name, None)
                _mark_source_absent(
                    record,
                    field_name,
                    str(skeleton.get("canonical_raw", {}).get(field_name, "--")),
                )
            for field_name in geometry_fields.intersection(
                skeleton.get("_invalid_observation_fields") or ()
            ):
                # Invalid alternate geometry is diagnostic evidence only.  It
                # cannot veto an independently validated, source-bound table
                # observation; only two different valid values may conflict.
                if record.get(field_name) not in (None, ""):
                    issues = getattr(parse_result, "_personal_detail_extraction_issues", None)
                    if isinstance(issues, list):
                        target_record_id = str(skeleton.get("account_id") or "")
                        issues[:] = [
                            issue
                            for issue in issues
                            if not (
                                issue.get("issue_code")
                                == "candidate_b_exact_slot_value_invalid"
                                and issue.get("parser_stage")
                                == "candidate_b_account_anchor_header_geometry"
                                and str(issue.get("target_record_id") or "")
                                == target_record_id
                                and issue.get("field_name") == field_name
                            )
                        ]
                    continue
                field_refs = [
                    ref
                    for ref in skeleton.get("source_refs_by_field", {}).get(field_name, ())
                    if isinstance(ref, Mapping)
                ]
                raw = skeleton.get("canonical_raw", {}).get(field_name, "")
                raw_values = raw if isinstance(raw, list) else [raw]
                _reject_exact_observation(
                    parse_result,
                    record,
                    dataset="credit_accounts",
                    target_record_id=str(skeleton.get("account_id") or ""),
                    field_name=field_name,
                    raw=" | ".join(str(value) for value in raw_values),
                    source_ref=field_refs[0] if field_refs else anchor_refs[0],
                    parser_stage="candidate_b_account_anchor_header_geometry",
                )
            if record.get("account_identifier"):
                record["account_identifier_source"] = skeleton.get("account_identifier_source")
            for field_name in ("credit_agreement_identifier", "card_tail"):
                if not skeleton.get(field_name):
                    continue
                record[field_name] = skeleton[field_name]
                record.setdefault("canonical_raw", {})[field_name] = skeleton[field_name]
                if anchor_refs:
                    record.setdefault("source_refs_by_field", {})[field_name] = [
                        {
                            **dict(anchor_refs[0]),
                            "field_name": field_name,
                            "binding": "canonical_account_anchor",
                            "binding_quality": "canonical_account_anchor",
                        }
                    ]
            record_refs = [
                *(table.get("source_refs") or ()),
                *(
                    ref
                    for replay_index in singleton_replays_by_skeleton.get(
                        skeleton_index, ()
                    )
                    for ref in table_accounts[replay_index].get("source_refs") or ()
                ),
                *(skeleton.get("source_refs") or ()),
            ]
            record["source_refs"] = []
            seen_record_refs: set[str] = set()
            for ref in record_refs:
                if not isinstance(ref, Mapping):
                    continue
                marker = json.dumps(
                    dict(ref), ensure_ascii=False, sort_keys=True, default=str
                )
                if marker in seen_record_refs:
                    continue
                seen_record_refs.add(marker)
                record["source_refs"].append(dict(ref))
            record["raw_detail_lines"] = list(skeleton.get("raw_detail_lines") or ())
            emitted.append(record)
            consumed.add(table_index)
            prior_account_id = str(table.get("account_id") or "")
            canonical_account_id = str(skeleton.get("account_id") or "")
            if prior_account_id and canonical_account_id:
                account_id_remap[prior_account_id] = canonical_account_id
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    register_issue_target_remap,
                )

                register_issue_target_remap(parse_result, prior_account_id, canonical_account_id)
                for replay_index in singleton_replays_by_skeleton.get(
                    skeleton_index, ()
                ):
                    replay_account_id = str(
                        table_accounts[replay_index].get("account_id") or ""
                    )
                    if not replay_account_id:
                        continue
                    account_id_remap[replay_account_id] = canonical_account_id
                    register_issue_target_remap(
                        parse_result, replay_account_id, canonical_account_id
                    )
            continue

        emitted.append(skeleton)
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_account_table_missing",
                message="A printed account anchor was present but its canonical account table was not recoverable.",
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                target_record_id=str(skeleton.get("account_id") or ""),
                observed_value={
                    "account_type": skeleton.get("account_type"),
                    "category_sequence": skeleton.get("category_sequence"),
                },
                source_refs=skeleton.get("source_refs") or (),
                reason_codes=("printed_account_anchor", "canonical_table_missing", "record_partial"),
            ),
        )

    anchored_types = {
        str(skeleton.get("account_type") or "")
        for skeleton in skeletons
        if skeleton.get("account_type")
    }
    consumed_observation_ids = {
        str(table_accounts[index].get("_table_observation_id") or "")
        for index in consumed
        if 0 <= index < len(table_accounts)
        and table_accounts[index].get("_table_observation_id")
    }
    for table_index, table in enumerate(table_accounts):
        if table_index in consumed or table_index in singleton_replay_indices:
            continue
        account_type = str(table.get("account_type") or "")
        structurally_missing_category = bool(account_type) and account_type not in anchored_types
        account_observation_id = str(
            table.get("_table_observation_id") or table.get("account_id") or ""
        )
        account_observation_instance_id = str(
            table.get("_table_observation_instance_id") or ""
        )
        suppressed_children: list[dict[str, Any]] = []
        if not structurally_missing_category:
            for child in repayments:
                child_instance_id = str(
                    child.get("_table_observation_instance_id") or ""
                )
                if account_observation_instance_id and child_instance_id:
                    if child_instance_id != account_observation_instance_id:
                        continue
                elif str(child.get("account_id") or "") != str(
                    table.get("account_id") or ""
                ):
                    continue
                suppressed_children.append(
                    {
                        "dataset": "credit_account_monthly_performance",
                        "child_observation_id": str(
                            child.get("repayment_id") or child.get("record_id") or ""
                        ),
                        "account_observation_id": account_observation_id,
                        "account_observation_instance_id": (
                            account_observation_instance_id or None
                        ),
                        "source_refs": [
                            dict(ref)
                            for ref in child.get("source_refs") or ()
                            if isinstance(ref, Mapping)
                        ],
                    }
                )
            for child in events:
                child_instance_id = str(
                    child.get("_table_observation_instance_id") or ""
                )
                if account_observation_instance_id and child_instance_id:
                    if child_instance_id != account_observation_instance_id:
                        continue
                elif str(child.get("account_id") or "") != str(
                    table.get("account_id") or ""
                ):
                    continue
                event_type = str(child.get("event_type") or "")
                suppressed_children.append(
                    {
                        "dataset": _ACCOUNT_EVENT_DATASETS.get(
                            event_type, "credit_account_special_events"
                        ),
                        "child_observation_id": str(
                            child.get("account_event_id") or child.get("record_id") or ""
                        ),
                        "account_observation_id": account_observation_id,
                        "account_observation_instance_id": (
                            account_observation_instance_id or None
                        ),
                        "event_type": event_type or None,
                        "source_refs": [
                            dict(ref)
                            for ref in child.get("source_refs") or ()
                            if isinstance(ref, Mapping)
                        ],
                    }
                )
        suppressed_child_counts = Counter(
            str(child["dataset"]) for child in suppressed_children
        )
        issue_refs: list[dict[str, Any]] = []
        seen_issue_refs: set[str] = set()
        for ref in (
            *(table.get("source_refs") or ()),
            *(
                child_ref
                for child in suppressed_children
                for child_ref in child.get("source_refs") or ()
            ),
        ):
            if not isinstance(ref, Mapping):
                continue
            normalized_ref = dict(ref)
            marker = json.dumps(normalized_ref, ensure_ascii=False, sort_keys=True, default=str)
            if marker in seen_issue_refs:
                continue
            seen_issue_refs.add(marker)
            issue_refs.append(normalized_ref)
        if structurally_missing_category:
            record = dict(table)
            record["sequence"] = len(emitted) + 1
            record.pop("category_sequence", None)
            record["extraction_status"] = "review"
            record["_ownership_status"] = "printed_category_anchor_missing"
            emitted.append(record)
        else:
            if account_observation_id not in consumed_observation_ids:
                _mark_account_observation_not_emitted(
                    parse_result,
                    account_observation_id,
                )
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code=(
                    "candidate_b_account_category_anchor_missing"
                    if structurally_missing_category
                    else "candidate_b_unmatched_account_table_suppressed"
                ),
                message=(
                    "A canonical account category had table evidence but no recoverable printed anchor; "
                    "the table row was retained with uncertainty."
                    if structurally_missing_category
                    else (
                        "An account-table observation in an already anchored category could not be assigned to "
                        "one printed account and was suppressed together with its exact related child observations."
                    )
                ),
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                target_record_id=(
                    (account_observation_instance_id or account_observation_id or None)
                    if not structurally_missing_category
                    else (account_observation_id or None)
                ),
                observed_value={
                    "table_observation_id": account_observation_id or None,
                    "account_observation_id": account_observation_id or None,
                    "account_observation_instance_id": (
                        account_observation_instance_id or None
                    ),
                    "account_type_candidate": account_type or None,
                    **(
                        {
                            "suppressed_child_count": len(suppressed_children),
                            "suppressed_child_counts_by_dataset": dict(
                                sorted(suppressed_child_counts.items())
                            ),
                            "affected_child_datasets": sorted(suppressed_child_counts),
                            "suppressed_child_observations": suppressed_children,
                        }
                        if not structurally_missing_category
                        else {}
                    ),
                },
                candidate_value=(
                    {
                        "same_category_emitted_account_ids": sorted(
                            {
                                str(account.get("account_id") or "")
                                for account in emitted
                                if str(account.get("account_type") or "") == account_type
                                and account.get("account_id")
                            }
                        )
                    }
                    if not structurally_missing_category
                    else None
                ),
                source_refs=issue_refs,
                reason_codes=(
                    "canonical_account_table",
                    (
                        "printed_anchor_missing"
                        if structurally_missing_category
                        else "printed_anchor_ownership_unresolved"
                    ),
                    "record_requires_review" if structurally_missing_category else "account_ownership_unresolved",
                    *(
                        ("related_child_observations_suppressed",)
                        if suppressed_children
                        else ()
                    ),
                    *(
                        (
                            "record_not_emitted_due_to_unresolved_account_ownership",
                        )
                        if not structurally_missing_category
                        else ()
                    ),
                ),
            ),
        )

    for account in emitted:
        if account.get("account_family_quality") != "ambiguous_missing_variant":
            continue
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_account_family_variant_unresolved",
                message=(
                    "The printed revolving-loan family heading did not preserve its one/two variant; "
                    "the conservative R1 family was retained and explicitly marked uncertain."
                ),
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                target_record_id=str(account.get("account_id") or ""),
                field_name="account_type",
                observed_value="循环贷账户",
                candidate_value=account.get("account_type"),
                source_refs=account.get("source_refs") or (),
                reason_codes=("account_family_variant_missing", "normalized_value_requires_review"),
            ),
        )

    emitted_by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for account in emitted:
        account_type = str(account.get("account_type") or "")
        if account_type:
            emitted_by_family[account_type].append(account)
    for account_type, family_accounts in emitted_by_family.items():
        observed: list[int] = []
        unresolved_accounts: list[dict[str, Any]] = []
        for account in family_accounts:
            try:
                ordinal = int(account.get("category_sequence") or 0)
            except (TypeError, ValueError):
                ordinal = 0
            if ordinal > 0:
                observed.append(ordinal)
            else:
                unresolved_accounts.append(account)
        observed = sorted(set(observed))
        # PBOC category ordinals start at one and are dense. Bound the gap
        # expansion so a single OCR-joined outlier cannot manufacture hundreds
        # of supposed missing records; the outlier is still reported below.
        credible_ceiling = max(12, len(observed) * 3)
        bounded_endpoint = min(max(observed), credible_ceiling) if observed else 0
        missing = sorted(set(range(1, bounded_endpoint + 1)) - set(observed))
        outliers = [ordinal for ordinal in observed if ordinal > credible_ceiling]
        if not missing and not outliers and not unresolved_accounts:
            continue
        refs = [
            dict(ref)
            for account in family_accounts
            for ref in account.get("source_refs") or ()
            if isinstance(ref, dict)
        ]
        record_issue(
            parse_result,
            make_issue(
                category="schema_incompleteness",
                issue_code="candidate_b_account_sequence_gap",
                message=(
                    "Printed account ordinals in one PBOC account family were not uniquely recoverable or "
                    "not dense; no ordinal or missing record was invented."
                ),
                parser_stage="candidate_b_account_schema",
                target_dataset="credit_accounts",
                observed_value={
                    "account_type": account_type,
                    "observed_category_sequences": observed,
                },
                candidate_value={
                    "missing_category_sequences": missing,
                    "outlier_category_sequences": outliers,
                    "unresolved_printed_ordinal_count": len(unresolved_accounts),
                    "provisional_account_ids": [
                        str(account.get("account_id") or "")
                        for account in unresolved_accounts
                        if account.get("account_id")
                    ],
                },
                source_refs=refs,
                reason_codes=(
                    (
                        "printed_category_ordinals_not_dense"
                        if missing or outliers
                        else "printed_category_ordinal_unresolved"
                    ),
                    *(
                        ("printed_category_ordinal_unresolved",)
                        if unresolved_accounts and (missing or outliers)
                        else ()
                    ),
                    "missing_records_not_invented",
                    "business_population_uncertain",
                ),
            ),
        )

    for account in emitted:
        if account.get("source") != "native_detail_account_table":
            continue
        source_absent = set(account.get("_source_absent_fields") or ())
        for field_name in ("management_institution", "account_identifier", "open_date", "currency"):
            if account.get(field_name) not in (None, "") or field_name in source_absent:
                continue
            _append_internal_field(account, "_unresolved_fields", field_name)
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_account_required_field_unresolved",
                    message="A required canonical account field was not safely decoded; the field remains withheld.",
                    parser_stage="candidate_b_account_canonical_slots",
                    target_dataset="credit_accounts",
                    target_record_id=str(account.get("account_id") or ""),
                    field_name=field_name,
                    source_refs=(
                        account.get("source_refs_by_field", {}).get(field_name)
                        or account.get("source_refs")
                        or ()
                    ),
                    reason_codes=(
                        "canonical_account_template",
                        "required_field_missing",
                        "normalized_value_withheld",
                    ),
                ),
            )

    account_identifiers = {
        str(account.get("account_id") or ""): account.get("account_identifier")
        for account in emitted
        if account.get("account_id")
    }
    accepted_table_account_ids = {
        str(table_accounts[index].get("account_id") or "")
        for index in consumed
        if 0 <= index < len(table_accounts)
    }
    accepted_table_account_ids.update(
        str(account.get("account_id") or "")
        for account in emitted
        if account.get("_ownership_status") == "printed_category_anchor_missing"
    )
    accepted_table_observation_instances = {
        str(table_accounts[index].get("_table_observation_instance_id") or "")
        for index in consumed
        if 0 <= index < len(table_accounts)
        and table_accounts[index].get("_table_observation_instance_id")
    }
    accepted_table_observation_instances.update(
        str(account.get("_table_observation_instance_id") or "")
        for account in emitted
        if account.get("_ownership_status") == "printed_category_anchor_missing"
        and account.get("_table_observation_instance_id")
    )
    filtered_repayments: list[dict[str, Any]] = []
    filtered_events: list[dict[str, Any]] = []
    for related_record, target in [
        *((record, filtered_repayments) for record in repayments),
        *((record, filtered_events) for record in events),
    ]:
        prior_account_id = str(related_record.get("account_id") or "")
        observation_instance_id = str(
            related_record.get("_table_observation_instance_id") or ""
        )
        if observation_instance_id:
            if observation_instance_id not in accepted_table_observation_instances:
                continue
        elif prior_account_id not in accepted_table_account_ids:
            # Compatibility for pre-instance synthetic observations.  Native
            # table children always carry the per-instance identity above.
            continue
        canonical_account_id = account_id_remap.get(prior_account_id)
        if canonical_account_id:
            related_record["account_id"] = canonical_account_id
            if not related_record.get("account_identifier"):
                related_record["account_identifier"] = account_identifiers.get(canonical_account_id)
        target.append(related_record)
    return emitted, filtered_repayments, filtered_events


def _extract_credit_lines(parse_result: Any) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    field_names = {
        "授信协议标识": "account_identifier",
        "管理机构": "institution",
        "授信额度用途": "facility_type",
        "生效日期": "effective_date",
        "到期日期": "due_date",
        "授信额度": "total_limit",
        "授信限额": "credit_limit",
        "已用额度": "used_limit",
        "授信限额编号": "limit_identifier",
        "币种": "currency",
    }
    records: list[dict[str, Any]] = []
    for candidate in PBOCPersonalDetailNativeParser(parse_result).records("credit_lines"):
        facts = candidate.fields
        identifier = _typed_identifier(_field(facts, "授信协议标识"))
        if not identifier:
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_identifier_unresolved",
                    message="A canonical credit-agreement card was observed but its agreement identifier was not safely extractable.",
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    field_name="account_identifier",
                    observed_value=_field(facts, "授信协议标识") or None,
                    source_refs=candidate.source_refs,
                    reason_codes=("canonical_credit_agreement_card", "typed_identifier_failed", "record_withheld"),
                ),
            )
            continue
        due_raw = _field(facts, "到期日期")
        raw_currency = _field(facts, "币种")
        if _agreement_source_absent(raw_currency):
            currency, currency_residue, currency_resolution = None, "", "source_absent"
        else:
            currency, currency_residue, currency_resolution = _currency_token(raw_currency)
        printed_sequence_raw = _field(facts, "__printed_sequence")
        printed_sequence = int(printed_sequence_raw) if str(printed_sequence_raw).isdigit() else None
        candidate_refs_by_field = getattr(candidate, "source_refs_by_field", {})
        candidate_bindings_by_field = getattr(candidate, "binding_quality_by_field", {})
        anchor_refs = [
            dict(ref)
            for ref in candidate_refs_by_field.get("__printed_sequence", ())
            if isinstance(ref, Mapping)
            and ref.get("binding") == "canonical_card_anchor"
        ] if isinstance(candidate_refs_by_field, Mapping) else []
        anchor_binding = (
            str(candidate_bindings_by_field.get("__printed_sequence") or "")
            if isinstance(candidate_bindings_by_field, Mapping)
            else ""
        )
        canonical_card_key = (
            f"credit_agreement:{printed_sequence}"
            if printed_sequence is not None
            and anchor_refs
            and anchor_binding == "canonical_card_anchor"
            else None
        )
        raw_limit_identifier = _field(facts, "授信限额编号")
        limit_identifier = _typed_identifier(raw_limit_identifier)
        if (
            raw_limit_identifier
            and not _agreement_source_absent(raw_limit_identifier)
            and not limit_identifier
        ):
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_credit_limit_identifier_unresolved",
                    message="The credit-limit identifier cell contained multiple or invalid identifier tokens and was withheld.",
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=stable_record_id("credit_line", identifier),
                    field_name="limit_identifier",
                    observed_value=raw_limit_identifier,
                    source_refs=candidate.source_refs,
                    reason_codes=(
                        "canonical_credit_agreement_card",
                        "typed_identifier_failed",
                        "normalized_value_withheld",
                    ),
                ),
            )
        raw_due_compact = _compact(due_raw)
        source_refs_by_field: dict[str, list[dict[str, Any]]] = {}
        for label, refs in candidate_refs_by_field.items():
            field_name = field_names.get(str(label))
            if not field_name:
                continue
            source_refs_by_field[field_name] = [
                {**dict(ref), "field_name": field_name}
                for ref in refs
                if isinstance(ref, Mapping)
            ]
        binding_quality_by_field = {
            field_names[label]: quality
            for label, quality in candidate_bindings_by_field.items()
            if label in field_names
        }
        source_absent_fields = {
            field_names[label]
            for label in field_names
            if _agreement_source_absent(_field(facts, label))
        }
        source_absent_raw = {
            field_names[label]: _field(facts, label)
            for label in field_names
            if _agreement_source_absent(_field(facts, label))
        }
        unresolved_fields = {
            field_names[label]
            for label in getattr(candidate, "unresolved_labels", frozenset())
            if label in field_names
        }
        credit_line_id = stable_record_id("credit_line", identifier)
        currency_refs = tuple(
            dict(ref)
            for ref in candidate_refs_by_field.get("币种", ())
            if isinstance(ref, Mapping)
        ) if isinstance(candidate_refs_by_field, Mapping) else tuple(candidate.source_refs)
        if currency_resolution == "residue" and currency is not None:
            _report_currency_residue(
                parse_result,
                dataset="credit_lines",
                target_record_id=credit_line_id,
                raw=raw_currency,
                currency=currency,
                residue=currency_residue,
                source_refs=currency_refs or tuple(candidate.source_refs),
                parser_stage="candidate_b_credit_agreement_schema",
            )
        elif currency_resolution in {"unknown", "multiple"}:
            unresolved_fields.add("currency")
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_credit_agreement_currency_unresolved",
                    message="The credit-agreement currency cell did not contain exactly one finite supported currency token and was withheld.",
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=credit_line_id,
                    field_name="currency",
                    observed_value=raw_currency,
                    source_refs=currency_refs or tuple(candidate.source_refs),
                    reason_codes=(
                        "finite_currency_vocabulary",
                        f"currency_token_{currency_resolution}",
                        "normalized_value_withheld",
                    ),
                ),
            )
        observed_fields = {
            field_names[label]
            for label in getattr(candidate, "observed_labels", frozenset())
            if label in field_names
        }
        raw_institution = _field(facts, "管理机构")
        raw_facility_type = _field(facts, "授信额度用途")
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            institution_name_has_separated_leading_han,
        )

        pending_institution = None
        if raw_institution and institution_name_has_separated_leading_han(raw_institution):
            pending_value = _account_institution(
                raw_institution,
                independently_corroborated=True,
            )
            if pending_value:
                pending_institution = {
                    "raw": raw_institution,
                    "value": pending_value,
                    "source_refs": list(source_refs_by_field.get("institution") or ()),
                }
        institution = _account_institution(raw_institution)
        raw_effective_date = _field(facts, "生效日期")
        effective_date = _date(raw_effective_date)
        due_date = _date(due_raw)
        record = {
                "credit_line_id": credit_line_id,
                "_printed_sequence": printed_sequence,
                "_canonical_card_key": canonical_card_key,
                "_canonical_card_anchor_refs": anchor_refs,
                "account_identifier": identifier,
                "institution": institution,
                "facility_type": (
                    None
                    if _agreement_source_absent(raw_facility_type)
                    else _clean(raw_facility_type)
                ),
                "effective_date": effective_date,
                "due_date": due_date,
                "validity_type": (
                    "perpetual"
                    if raw_due_compact == "长期"
                    else "fixed_term"
                    if due_date
                    else "unknown"
                ),
                "total_limit": (
                    None
                    if _agreement_source_absent(_field(facts, "授信额度"))
                    else _number(_field(facts, "授信额度"))
                ),
                "credit_limit": (
                    None
                    if _agreement_source_absent(_field(facts, "授信限额"))
                    else _number(_field(facts, "授信限额"))
                ),
                "used_limit": (
                    None
                    if _agreement_source_absent(_field(facts, "已用额度"))
                    else _number(_field(facts, "已用额度"))
                ),
                "limit_identifier": limit_identifier,
                "currency": currency,
                "account_currency": currency,
                "reporting_amount_currency": currency,
                "amount_unit": "yuan",
                "reporting_amount_unit": "yuan",
                "source": "candidate_b_credit_agreement_schema",
                "source_refs": list(candidate.source_refs),
                "confidence": candidate.confidence,
                "source_refs_by_field": source_refs_by_field,
                **(
                    {"canonical_raw": source_absent_raw}
                    if source_absent_raw
                    else {}
                ),
                "_field_binding_quality": binding_quality_by_field,
                "_source_absent_fields": sorted(source_absent_fields),
                "_unresolved_fields": sorted(unresolved_fields),
                "_observed_fields": sorted(observed_fields),
                **(
                    {"_pending_institution_observation": pending_institution}
                    if pending_institution is not None
                    else {}
                ),
            }
        for field_name, raw_value, converted in (
            ("institution", raw_institution, institution),
            ("effective_date", raw_effective_date, effective_date),
            ("due_date", due_raw, due_date),
        ):
            raw_compact = _compact(raw_value)
            if (
                converted not in (None, "")
                or not raw_compact
                or _agreement_source_absent(raw_value)
                or (field_name == "due_date" and raw_compact == "长期")
            ):
                continue
            if field_name == "institution" and pending_institution is not None:
                continue
            field_refs = source_refs_by_field.get(field_name) or []
            source_ref = (
                dict(field_refs[0])
                if field_refs
                else next(
                    (dict(ref) for ref in candidate.source_refs if isinstance(ref, Mapping)),
                    {},
                )
            )
            _reject_exact_observation(
                parse_result,
                record,
                dataset="credit_lines",
                target_record_id=credit_line_id,
                field_name=field_name,
                raw=raw_value,
                source_ref=source_ref,
                parser_stage="candidate_b_credit_agreement_schema",
            )
        records.append(record)
    return records


def _agreement_source_absent(value: Any) -> bool:
    """Return whether an agreement cell explicitly prints only dash glyphs."""

    return is_explicit_source_absence(value)


def _agreement_identifier_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _agreement_strong_field_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import normalize_institution_name

    matches = 0
    for field_name in (
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "currency",
    ):
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        if left_value in (None, "") or right_value in (None, ""):
            continue
        if field_name == "institution":
            equal = normalize_institution_name(str(left_value)) == normalize_institution_name(str(right_value))
        elif field_name in {"total_limit", "credit_limit", "used_limit"}:
            equal = str(left_value).replace(",", "") == str(right_value).replace(",", "")
        else:
            equal = _compact(left_value) == _compact(right_value)
        matches += int(equal)
    return matches


def _agreement_field_value_key(field_name: str, value: Any) -> str:
    if field_name == "institution":
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            normalize_institution_name,
        )
        return normalize_institution_name(str(value or ""))
    if field_name in {"total_limit", "credit_limit", "used_limit"}:
        parsed = _number(value)
        return str(parsed) if parsed is not None else ""
    return _compact(value)


def _agreement_printed_sequences_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_sequence = left.get("_printed_sequence")
    right_sequence = right.get("_printed_sequence")
    return bool(
        left_sequence not in (None, "")
        and right_sequence not in (None, "")
        and int(left_sequence) == int(right_sequence)
    )


def _agreement_single_leading_ocr_insertion(
    left_identifier: str,
    right_identifier: str,
) -> str | None:
    """Return the shorter exact identifier after one leading OCR insertion.

    This is deliberately not edit-distance matching: every character of the
    canonical candidate must be an exact suffix of the other observation.
    """

    longer, shorter = sorted(
        (left_identifier, right_identifier),
        key=lambda value: (len(value), value),
        reverse=True,
    )
    if (
        len(shorter) >= 20
        and len(longer) == len(shorter) + 1
        and longer[0].isalpha()
        and longer[1:] == shorter
    ):
        return shorter
    return None


def _agreement_same_source_page(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def pages(record: Mapping[str, Any]) -> set[int]:
        return {
            int(ref.get("source_page") or ref.get("logical_page") or 0)
            for ref in _agreement_observation_refs(record)
            if int(ref.get("source_page") or ref.get("logical_page") or 0) > 0
        }

    return bool(pages(left) & pages(right))


def _agreement_observation_refs(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs = [ref for ref in record.get("source_refs") or () if isinstance(ref, Mapping)]
    for field_refs in (record.get("source_refs_by_field") or {}).values():
        refs.extend(ref for ref in field_refs or () if isinstance(ref, Mapping))
    return refs


def _agreement_canonical_card_key(record: Mapping[str, Any]) -> str:
    key = str(record.get("_canonical_card_key") or "")
    sequence = record.get("_printed_sequence")
    if sequence in (None, "") or not str(sequence).isdigit():
        return ""
    expected = f"credit_agreement:{int(sequence)}"
    return key if key == expected else ""


def _agreement_anchor_planes(record: Mapping[str, Any]) -> set[str]:
    planes: set[str] = set()
    for ref in record.get("_canonical_card_anchor_refs") or ():
        if not isinstance(ref, Mapping) or ref.get("binding") != "canonical_card_anchor":
            continue
        source = str(ref.get("source") or "")
        if source.startswith("personal_detail_corrected_page"):
            planes.add("corrected_page")
        elif source.startswith("native_detail"):
            planes.add("native_table")
    return planes


def _agreement_anchor_geometry_overlaps(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_refs = [
        ref
        for ref in left.get("_canonical_card_anchor_refs") or ()
        if isinstance(ref, Mapping) and ref.get("binding") == "canonical_card_anchor"
    ]
    right_refs = [
        ref
        for ref in right.get("_canonical_card_anchor_refs") or ()
        if isinstance(ref, Mapping) and ref.get("binding") == "canonical_card_anchor"
    ]
    left_evidence = {
        str(evidence_id)
        for ref in left_refs
        for evidence_id in ref.get("evidence_ids") or ()
        if evidence_id
    }
    right_evidence = {
        str(evidence_id)
        for ref in right_refs
        for evidence_id in ref.get("evidence_ids") or ()
        if evidence_id
    }
    if left_evidence & right_evidence:
        return True
    for left_ref in left_refs:
        left_box = left_ref.get("bbox")
        left_page = int(left_ref.get("source_page") or left_ref.get("logical_page") or 0)
        if left_page <= 0 or not isinstance(left_box, (list, tuple)) or len(left_box) != 4:
            continue
        for right_ref in right_refs:
            right_box = right_ref.get("bbox")
            right_page = int(right_ref.get("source_page") or right_ref.get("logical_page") or 0)
            if right_page != left_page or not isinstance(right_box, (list, tuple)) or len(right_box) != 4:
                continue
            intersection = max(
                0.0,
                min(float(left_box[2]), float(right_box[2]))
                - max(float(left_box[0]), float(right_box[0])),
            ) * max(
                0.0,
                min(float(left_box[3]), float(right_box[3]))
                - max(float(left_box[1]), float(right_box[1])),
            )
            left_area = max(
                1.0,
                (float(left_box[2]) - float(left_box[0]))
                * (float(left_box[3]) - float(left_box[1])),
            )
            right_area = max(
                1.0,
                (float(right_box[2]) - float(right_box[0]))
                * (float(right_box[3]) - float(right_box[1])),
            )
            if intersection / min(left_area, right_area) >= 0.60:
                return True
    return False


def _agreement_authorized_cross_plane_anchors(
    records: list[dict[str, Any]],
) -> set[str]:
    """Return exact card anchors that are unique inside every evidence plane."""

    identifiers_by_key_and_plane: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        key = _agreement_canonical_card_key(record)
        identifier = _agreement_identifier_text(record.get("account_identifier"))
        if not key or not identifier:
            continue
        for plane in _agreement_anchor_planes(record):
            identifiers_by_key_and_plane[key][plane].add(identifier)
    return {
        key
        for key, identifiers_by_plane in identifiers_by_key_and_plane.items()
        if len(identifiers_by_plane) >= 2
        and all(len(identifiers) == 1 for identifiers in identifiers_by_plane.values())
    }


def _agreement_overlapping_business_conflicts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> set[str]:
    conflicts: set[str] = set()
    for field_name in (
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "currency",
    ):
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        if left_value in (None, "") or right_value in (None, ""):
            continue
        if _agreement_field_value_key(field_name, left_value) != _agreement_field_value_key(
            field_name,
            right_value,
        ):
            conflicts.add(field_name)
    return conflicts


def _agreement_has_strict_business_field_superset(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    field_names = {
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "currency",
    }
    left_fields = {field_name for field_name in field_names if left.get(field_name) not in (None, "")}
    right_fields = {field_name for field_name in field_names if right.get(field_name) not in (None, "")}
    return left_fields < right_fields or right_fields < left_fields


def _agreement_exact_anchor_authorizes_merge(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    authorized_anchors: set[str],
) -> bool:
    left_key = _agreement_canonical_card_key(left)
    right_key = _agreement_canonical_card_key(right)
    left_planes = _agreement_anchor_planes(left)
    right_planes = _agreement_anchor_planes(right)
    return bool(
        left_key
        and left_key == right_key
        and left_key in authorized_anchors
        and left_planes
        and right_planes
        and left_planes.isdisjoint(right_planes)
        and _agreement_strong_field_matches(left, right) >= 3
        and not _agreement_overlapping_business_conflicts(left, right)
        and _agreement_has_strict_business_field_superset(left, right)
    )


def _agreement_verified_observation_relation(
    issue_owner: Any,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Require shared canonical geometry or an explicit continuation edge.

    Page proximity and identifier resemblance are intentionally insufficient:
    several distinct agreements can share a page, institution, dates, and
    amount fields.  This relation only establishes that two observations came
    from the same canonical card region (or two tables explicitly joined by
    the plugin's continuation graph).
    """

    left_refs = _agreement_observation_refs(left)
    right_refs = _agreement_observation_refs(right)
    left_tables = {str(ref.get("table_id") or "") for ref in left_refs if ref.get("table_id")}
    right_tables = {str(ref.get("table_id") or "") for ref in right_refs if ref.get("table_id")}
    if left_tables & right_tables:
        return True

    continuation_check = getattr(issue_owner, "tables_continue", None)
    if callable(continuation_check) and any(
        continuation_check(left_table, right_table) is True
        for left_table in left_tables
        for right_table in right_tables
    ):
        return True

    for left_ref in left_refs:
        left_box = left_ref.get("bbox")
        if not isinstance(left_box, (list, tuple)) or len(left_box) != 4:
            continue
        left_page = int(left_ref.get("source_page") or left_ref.get("logical_page") or 0)
        for right_ref in right_refs:
            right_page = int(right_ref.get("source_page") or right_ref.get("logical_page") or 0)
            right_box = right_ref.get("bbox")
            if left_page <= 0 or left_page != right_page:
                continue
            if not isinstance(right_box, (list, tuple)) or len(right_box) != 4:
                continue
            intersection = max(0.0, min(float(left_box[2]), float(right_box[2])) - max(float(left_box[0]), float(right_box[0]))) * max(
                0.0, min(float(left_box[3]), float(right_box[3])) - max(float(left_box[1]), float(right_box[1]))
            )
            left_area = max(1.0, (float(left_box[2]) - float(left_box[0])) * (float(left_box[3]) - float(left_box[1])))
            right_area = max(1.0, (float(right_box[2]) - float(right_box[0])) * (float(right_box[3]) - float(right_box[1])))
            if intersection / min(left_area, right_area) >= 0.60:
                return True
    return False


def _agreement_field_provenance_quality(record: Mapping[str, Any], field_name: str) -> int:
    bindings = record.get("_field_binding_quality")
    binding = str(bindings.get(field_name) or "") if isinstance(bindings, Mapping) else ""
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    cell_scoped = any(
        isinstance(ref, Mapping)
        and ref.get("geometry_scope") == "cell"
        and ref.get("binding") in {"canonical_label_slot", "label_column"}
        for ref in refs or ()
    )
    if binding == "canonical_cell_slot" and cell_scoped:
        return 4
    if binding == "native_label_column" and cell_scoped:
        return 3
    if binding == "native_label_column":
        return 2
    # Backward-compatible synthetic observations remain comparable in unit
    # tests, but can never outrank a label-bound production observation.
    return 1 if record.get(field_name) not in (None, "") else 0


def _agreement_observation_quality(record: Mapping[str, Any]) -> tuple[int, int, int]:
    """Rank one agreement observation by schema validity before OCR confidence.

    Repeated observations of a card can contain a fully populated canonical row
    and a damaged continuation fragment.  Counting non-empty cells alone lets
    the fragment win when an OCR artifact happens to look numeric.  Candidate B
    therefore ranks observations by field contracts first and uses confidence
    only as a later tie-breaker.
    """
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        _is_valid_for_role,
    )

    role_fields = (
        ("account_identifier", "account_identifier"),
        ("institution", "institution_name"),
        ("facility_type", "facility_type"),
        ("effective_date", "date"),
        ("due_date", "date"),
        ("total_limit", "amount"),
        ("credit_limit", "amount"),
        ("used_limit", "amount"),
        ("currency", "currency"),
    )
    valid_fields = 0
    valid_identity_fields = 0
    provenance = 0
    for field_name, role in role_fields:
        value = record.get(field_name)
        if value in (None, ""):
            continue
        if _is_valid_for_role(str(value), role):
            valid_fields += 1
            provenance += _agreement_field_provenance_quality(record, field_name)
            if field_name in {"institution", "facility_type"}:
                valid_identity_fields += 1
    return provenance, valid_identity_fields, valid_fields


def reconcile_candidate_b_credit_lines(
    parse_result: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the one-row-per-agreement schema constraint after correction."""
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
        register_issue_target_remap,
    )

    authorized_cross_plane_anchors = _agreement_authorized_cross_plane_anchors(records)
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    prototypes: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(records):
        record = dict(source)
        if record.get("_printed_sequence") in (None, "") and record.get("sequence") not in (None, ""):
            record["_printed_sequence"] = record.get("sequence")
        issue_target_ids = {
            str(value)
            for value in (source.get("credit_line_id"), source.get("record_id"))
            if value
        }
        raw_identifier = _agreement_identifier_text(record.get("account_identifier"))
        identifier = _typed_identifier(record.get("account_identifier"))
        if identifier:
            record["account_identifier"] = identifier
            record["credit_line_id"] = stable_record_id("credit_line", identifier)
            issue_target_ids.add(str(record["credit_line_id"]))
        if issue_target_ids:
            record["_issue_target_ids"] = sorted(issue_target_ids)
        identity = str(record.get("credit_line_id") or f"candidate_b_credit_line_row:{index + 1}")
        if identity not in groups and raw_identifier:
            compatible: list[str] = []
            for candidate_identity, prototype in prototypes.items():
                prototype_identifier = _agreement_identifier_text(prototype.get("account_identifier"))
                strong_match_count = _agreement_strong_field_matches(record, prototype)
                sequence_match = _agreement_printed_sequences_match(record, prototype)
                verified_relation = _agreement_verified_observation_relation(
                    parse_result, prototype, record
                )
                leading_insertion_identifier = _agreement_single_leading_ocr_insertion(
                    raw_identifier,
                    prototype_identifier,
                )
                exact_leading_insertion_relation = bool(
                    leading_insertion_identifier
                    and sequence_match
                    and _agreement_same_source_page(prototype, record)
                    and strong_match_count >= 2
                )
                exact_anchor_relation = _agreement_exact_anchor_authorizes_merge(
                    prototype,
                    record,
                    authorized_cross_plane_anchors,
                )
                if (
                    raw_identifier != prototype_identifier
                    and (
                        (
                            sequence_match
                            and verified_relation
                            and strong_match_count >= 3
                        )
                        or exact_leading_insertion_relation
                        or exact_anchor_relation
                    )
                ):
                    if exact_leading_insertion_relation:
                        prototype["_identifier_leading_insertion_canonical"] = leading_insertion_identifier
                        record["_identifier_leading_insertion_canonical"] = leading_insertion_identifier
                    if exact_anchor_relation:
                        prototype["_canonical_card_anchor_verified"] = True
                        record["_canonical_card_anchor_verified"] = True
                    compatible.append(candidate_identity)
                    continue
                same_authorized_card_key = bool(
                    _agreement_canonical_card_key(record)
                    and _agreement_canonical_card_key(record)
                    == _agreement_canonical_card_key(prototype)
                )
                if raw_identifier != prototype_identifier and (
                    sequence_match
                    or same_authorized_card_key
                    or leading_insertion_identifier
                    or _agreement_anchor_geometry_overlaps(prototype, record)
                ):
                    record_issue(
                        parse_result,
                        make_issue(
                            category="schema_incompleteness",
                            issue_code="candidate_b_credit_agreement_identity_ambiguous",
                            message=(
                                "Two credit-agreement observations had different identifiers without the exact "
                                "ordinal-and-source relation required to prove one canonical card; both were retained."
                            ),
                            parser_stage="candidate_b_credit_agreement_schema",
                            target_dataset="credit_lines",
                            observed_value={
                                "left_identifier": prototype.get("account_identifier"),
                                "right_identifier": record.get("account_identifier"),
                                "left_sequence": prototype.get("_printed_sequence"),
                                "right_sequence": record.get("_printed_sequence"),
                            },
                            source_refs=[
                                *(prototype.get("source_refs") or ()),
                                *(record.get("source_refs") or ()),
                            ],
                            reason_codes=(
                                "different_identifiers",
                                "exact_card_identity_not_proven",
                                (
                                    "canonical_card_anchor_business_conflict"
                                    if same_authorized_card_key
                                    and _agreement_canonical_card_key(record)
                                    in authorized_cross_plane_anchors
                                    else "canonical_card_anchor_not_cross_plane_unique"
                                    if same_authorized_card_key
                                    else "canonical_card_anchor_unavailable"
                                ),
                                "fuzzy_identifier_merge_forbidden",
                                "records_conservatively_retained",
                            ),
                        ),
                    )
            if len(compatible) == 1:
                identity = compatible[0]
        if identity not in groups:
            groups[identity] = []
            order.append(identity)
            prototypes[identity] = record
        groups[identity].append(record)

    reconciled: list[dict[str, Any]] = []
    ignored_fields = {
        "canonical_raw",
        "raw",
        "source",
        "source_refs",
        "source_refs_by_field",
        "confidence",
        "credit_line_id",
        "sequence",
        "_printed_sequence",
        "_issue_target_ids",
        "_field_binding_quality",
        "_source_absent_fields",
        "_unresolved_fields",
        "_observed_fields",
        "_identifier_leading_insertion_canonical",
        "_canonical_card_key",
        "_canonical_card_anchor_refs",
        "_canonical_card_anchor_verified",
        "_pending_institution_observation",
    }
    provenance_fields = (
        "account_identifier",
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "currency",
    )
    for identity in order:
        observations = groups[identity]
        ranked = sorted(
            observations,
            key=lambda row: (
                _agreement_observation_quality(row),
                sum(value not in (None, "", [], {}) for key, value in row.items() if key not in ignored_fields),
                float(row.get("confidence") or 0.0),
                len(str(row.get("account_identifier") or "")),
            ),
            reverse=True,
        )
        selected = dict(ranked[0])
        printed_sequences = {
            int(observation["_printed_sequence"])
            for observation in observations
            if observation.get("_printed_sequence") not in (None, "")
        }
        if len(printed_sequences) == 1:
            selected["sequence"] = next(iter(printed_sequences))
        selected.pop("_printed_sequence", None)
        selected.pop("_issue_target_ids", None)
        selected.pop("_pending_institution_observation", None)
        conflicts: set[str] = set()
        conflict_values: dict[str, list[Any]] = {}
        merged_refs: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for observation in ranked:
            for ref in observation.get("source_refs") or ():
                if not isinstance(ref, dict):
                    continue
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    merged_refs.append(dict(ref))
            for key, value in observation.items():
                if key in ignored_fields or value in (None, "", [], {}):
                    continue
                if key in provenance_fields:
                    continue
                retained = selected.get(key)
                if retained in (None, "", [], {}):
                    selected[key] = deepcopy(value)
                elif retained != value:
                    conflicts.add(key)

        selected_field_refs: dict[str, list[dict[str, Any]]] = {}
        selected_binding_quality: dict[str, str] = {}
        source_absent_fields = {
            str(field_name)
            for observation in observations
            for field_name in observation.get("_source_absent_fields") or ()
        }
        unresolved_fields = {
            str(field_name)
            for observation in observations
            for field_name in observation.get("_unresolved_fields") or ()
        }
        observed_fields = {
            str(field_name)
            for observation in observations
            for field_name in observation.get("_observed_fields") or ()
        }
        unresolved_institution_boundaries: list[Mapping[str, Any]] = []
        for field_name in provenance_fields:
            if field_name == "account_identifier":
                insertion_candidates = {
                    str(observation.get("_identifier_leading_insertion_canonical") or "")
                    for observation in observations
                    if observation.get("_identifier_leading_insertion_canonical")
                }
                if len(insertion_candidates) == 1:
                    canonical_identifier = next(iter(insertion_candidates))
                    selected[field_name] = canonical_identifier
                    chosen_observations = [
                        observation
                        for observation in observations
                        if _agreement_identifier_text(observation.get(field_name)) == canonical_identifier
                    ]
                    selected_field_refs[field_name] = [
                        dict(ref)
                        for observation in chosen_observations
                        for ref in (
                            (observation.get("source_refs_by_field") or {}).get(field_name)
                            if isinstance(observation.get("source_refs_by_field"), Mapping)
                            else ()
                        )
                        if isinstance(ref, Mapping)
                    ]
                    continue
            candidates = [
                (
                    _agreement_field_provenance_quality(observation, field_name),
                    float(observation.get("confidence") or 0.0),
                    observation.get(field_name),
                    observation,
                )
                for observation in observations
                if observation.get(field_name) not in (None, "")
            ]
            if field_name == "institution":
                pending_boundaries = [
                    pending
                    for observation in observations
                    for pending in (observation.get("_pending_institution_observation"),)
                    if isinstance(pending, Mapping) and pending.get("value")
                ]
                corroborated_pending: list[Mapping[str, Any]] = []
                for pending in pending_boundaries:
                    pending_refs = [
                        ref for ref in pending.get("source_refs") or () if isinstance(ref, Mapping)
                    ]
                    same_pending_refs = [
                        ref
                        for other in pending_boundaries
                        if other is not pending and other.get("value") == pending.get("value")
                        for ref in other.get("source_refs") or ()
                        if isinstance(ref, Mapping)
                    ]
                    clean_refs = [
                        ref
                        for _quality, _confidence, value, observation in candidates
                        if _agreement_field_value_key("institution", value)
                        == _agreement_field_value_key("institution", pending.get("value"))
                        for ref in (
                            (observation.get("source_refs_by_field") or {}).get("institution")
                            if isinstance(observation.get("source_refs_by_field"), Mapping)
                            else ()
                        )
                        if isinstance(ref, Mapping)
                    ]
                    if any(
                        _institution_refs_are_independent(left, right)
                        for left in pending_refs
                        for right in [*same_pending_refs, *clean_refs]
                    ):
                        corroborated_pending.append(pending)
                corroborated_values = {
                    _agreement_field_value_key("institution", pending.get("value"))
                    for pending in corroborated_pending
                }
                clean_values = {
                    _agreement_field_value_key("institution", candidate[2])
                    for candidate in candidates
                }
                if corroborated_values and clean_values and len(corroborated_values | clean_values) > 1:
                    selected[field_name] = None
                    unresolved_fields.add(field_name)
                    conflicts.add(field_name)
                    conflict_values[field_name] = [
                        *[candidate[2] for candidate in candidates],
                        *[pending.get("value") for pending in corroborated_pending],
                    ]
                    unresolved_institution_boundaries.extend(pending_boundaries)
                    continue
                if len(corroborated_values) == 1 and not candidates:
                    selected[field_name] = next(
                        pending.get("value")
                        for pending in corroborated_pending
                        if _agreement_field_value_key("institution", pending.get("value"))
                        in corroborated_values
                    )
                    selected_field_refs[field_name] = [
                        dict(ref)
                        for pending in corroborated_pending
                        for ref in pending.get("source_refs") or ()
                        if isinstance(ref, Mapping)
                    ]
                    unresolved_fields.discard(field_name)
                    continue
                if len(corroborated_values) != 1:
                    corroborated_pending = []
                unresolved_institution_boundaries.extend(
                    pending
                    for pending in pending_boundaries
                    if pending not in corroborated_pending
                )
            if not candidates:
                selected[field_name] = None
                continue
            if field_name == "institution":
                source_bound = [candidate for candidate in candidates if candidate[0] >= 2]
                independent_values = {
                    _agreement_field_value_key(field_name, candidate[2])
                    for candidate in source_bound
                }
                if len(independent_values) > 1:
                    selected[field_name] = None
                    unresolved_fields.add(field_name)
                    conflicts.add(field_name)
                    conflict_values[field_name] = [
                        candidate[2]
                        for candidate in sorted(
                            source_bound,
                            key=lambda item: (item[0], item[1], str(item[2])),
                            reverse=True,
                        )
                    ]
                    continue
            best_quality = max(candidate[0] for candidate in candidates)
            best = [candidate for candidate in candidates if candidate[0] == best_quality]
            by_value: dict[str, list[tuple[int, float, Any, Mapping[str, Any]]]] = defaultdict(list)
            for candidate in best:
                by_value[_agreement_field_value_key(field_name, candidate[2])].append(candidate)
            if len(by_value) != 1:
                selected[field_name] = None
                unresolved_fields.add(field_name)
                conflicts.add(field_name)
                conflict_values[field_name] = [
                    candidate[2]
                    for candidate in sorted(best, key=lambda item: (item[1], str(item[2])), reverse=True)
                ]
                continue
            chosen = max(best, key=lambda item: (item[1], len(str(item[2]))))
            selected[field_name] = deepcopy(chosen[2])
            chosen_observations = by_value[next(iter(by_value))]
            field_refs: list[dict[str, Any]] = []
            field_ref_markers: set[str] = set()
            for _quality, _confidence, _value, observation in chosen_observations:
                refs_by_field = observation.get("source_refs_by_field")
                refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
                for ref in refs or ():
                    if not isinstance(ref, Mapping):
                        continue
                    marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                    if marker in field_ref_markers:
                        continue
                    field_ref_markers.add(marker)
                    field_refs.append(dict(ref))
            if field_refs:
                selected_field_refs[field_name] = field_refs
            binding_map = chosen[3].get("_field_binding_quality")
            if isinstance(binding_map, Mapping) and binding_map.get(field_name):
                selected_binding_quality[field_name] = str(binding_map[field_name])

        selected_identifier = _typed_identifier(selected.get("account_identifier"))
        if selected_identifier:
            selected["account_identifier"] = selected_identifier
            selected["credit_line_id"] = stable_record_id("credit_line", selected_identifier)
        else:
            selected["account_identifier"] = None
            selected["credit_line_id"] = stable_record_id(
                "credit_line_unresolved",
                next(iter(printed_sequences)) if len(printed_sequences) == 1 else identity,
                sorted(
                    {
                        _agreement_identifier_text(observation.get("account_identifier"))
                        for observation in observations
                        if _agreement_identifier_text(observation.get("account_identifier"))
                    }
                ),
            )
            unresolved_fields.add("account_identifier")
        final_target_id = str(selected["credit_line_id"])
        if unresolved_institution_boundaries:
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_institution_leading_boundary_ambiguous",
                    message=(
                        "A separated leading Han glyph could be either intra-name OCR whitespace or cross-cell "
                        "debris; the joined institution was not silently selected without independent source evidence."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=final_target_id,
                    field_name="institution",
                    observed_value=[
                        str(pending.get("raw") or "")
                        for pending in unresolved_institution_boundaries
                    ],
                    candidate_value={
                        "joined_values": sorted(
                            {
                                str(pending.get("value") or "")
                                for pending in unresolved_institution_boundaries
                            }
                        )
                    },
                    source_refs=[
                        ref
                        for pending in unresolved_institution_boundaries
                        for ref in pending.get("source_refs") or ()
                        if isinstance(ref, Mapping)
                    ],
                    reason_codes=(
                        "separated_leading_han_boundary",
                        "independent_source_corroboration_missing",
                        "normalized_value_withheld",
                    ),
                ),
            )
        for observation in observations:
            for prior_target_id in observation.get("_issue_target_ids") or ():
                register_issue_target_remap(parse_result, prior_target_id, final_target_id)

        selected["source_refs_by_field"] = selected_field_refs
        selected["_field_binding_quality"] = selected_binding_quality
        selected["_source_absent_fields"] = sorted(source_absent_fields)
        selected["_unresolved_fields"] = sorted(unresolved_fields)
        selected["_observed_fields"] = sorted(observed_fields)
        currency = selected.get("currency")
        selected["account_currency"] = currency
        selected["reporting_amount_currency"] = currency
        due_date = selected.get("due_date")
        if due_date not in (None, ""):
            selected["validity_type"] = "fixed_term"
        elif any(
            _compact(observation.get("canonical_raw", {}).get("due_date") if isinstance(observation.get("canonical_raw"), Mapping) else "") == "长期"
            for observation in observations
        ):
            selected["validity_type"] = "perpetual"
        elif selected.get("validity_type") != "perpetual":
            selected["validity_type"] = "unknown"
        if merged_refs:
            selected["source_refs"] = merged_refs
        anchor_verified = bool(selected.pop("_canonical_card_anchor_verified", False))
        selected.pop("_canonical_card_key", None)
        selected.pop("_canonical_card_anchor_refs", None)
        reconciled.append(selected)
        identifier_variants = {
            _agreement_identifier_text(observation.get("account_identifier"))
            for observation in observations
            if _agreement_identifier_text(observation.get("account_identifier"))
        }
        if len(identifier_variants) > 1:
            leading_insertion_canonical = selected.get("_identifier_leading_insertion_canonical")
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_credit_agreement_identifier_variant",
                    message=(
                        "Source-verified observations of one agreement contained different identifiers; "
                        "only a uniquely stronger canonical field observation may be retained."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=str(selected.get("credit_line_id") or identity),
                    field_name="account_identifier",
                    observed_value=sorted(identifier_variants),
                    candidate_value=selected.get("account_identifier"),
                    source_refs=merged_refs,
                    reason_codes=(
                        (
                            "exact_canonical_card_anchor_cross_plane"
                            if anchor_verified
                            else "same_printed_sequence"
                            if len(printed_sequences) == 1
                            else "verified_source_relation"
                        ),
                        "different_identifier_observations",
                        (
                            "exact_leading_ocr_insertion_corrected"
                            if leading_insertion_canonical
                            else "canonical_anchor_provenance_selection"
                            if anchor_verified
                            else "fuzzy_identifier_selection_forbidden"
                        ),
                        (
                            "normalized_value_withheld"
                            if selected.get("account_identifier") in (None, "")
                            else "higher_provenance_value_retained_for_review"
                        ),
                    ),
                ),
            )
        if len(observations) > 1 and conflicts:
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_observation_conflict",
                    message=(
                        "Multiple corrected observations resolved to one canonical credit-agreement identity; "
                        "equally source-bound values disagreed; the conflicting fields were withheld for review."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=str(selected.get("credit_line_id") or identity),
                    observed_value={
                        "candidate_count": len(observations),
                        "conflicting_fields": sorted(conflicts),
                        "field_candidates": conflict_values,
                    },
                    source_refs=merged_refs,
                    reason_codes=(
                        "canonical_identity_collision",
                        "field_provenance_tie",
                        "conflicting_values_withheld",
                    ),
                ),
            )
    sequence_counts = Counter(
        int(record["sequence"])
        for record in reconciled
        if record.get("sequence") not in (None, "")
    )
    for record in reconciled:
        sequence = record.get("sequence")
        reason_codes: tuple[str, ...]
        if sequence not in (None, "") and sequence_counts[int(sequence)] == 1:
            continue
        if sequence not in (None, ""):
            record.pop("sequence", None)
            reason_codes = (
                "printed_sequence_collision",
                "row_order_not_used",
                "normalized_value_withheld",
            )
        else:
            reason_codes = (
                "printed_sequence_unreadable",
                "row_order_not_used",
                "normalized_value_withheld",
            )
        record_issue(
            parse_result,
            make_issue(
                category="schema_incompleteness",
                issue_code="candidate_b_credit_agreement_sequence_unresolved",
                message="The printed credit-agreement ordinal was missing or non-unique; encounter order was not emitted as business data.",
                parser_stage="candidate_b_credit_agreement_schema",
                target_dataset="credit_lines",
                target_record_id=str(record.get("credit_line_id") or "") or None,
                field_name="sequence",
                observed_value=sequence,
                source_refs=record.get("source_refs") or (),
                reason_codes=reason_codes,
            ),
        )
    for record in reconciled:
        source_absent_fields = set(record.get("_source_absent_fields") or ())
        expected_fields = (
            "account_identifier",
            "institution",
            "facility_type",
            "effective_date",
            "due_date",
            "total_limit",
            "credit_limit",
            "used_limit",
            "limit_identifier",
            "currency",
        )
        for field_name in expected_fields:
            if record.get(field_name) not in (None, ""):
                continue
            if field_name in source_absent_fields:
                continue
            if field_name == "due_date" and record.get("validity_type") == "perpetual":
                continue
            refs_by_field = record.get("source_refs_by_field")
            field_refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_credit_agreement_required_field_unresolved",
                    message=(
                        "A required credit-agreement field remained unavailable after all canonical "
                        "observations were reconciled; no business value was invented."
                    ),
                    parser_stage="candidate_b_credit_agreement_schema",
                    target_dataset="credit_lines",
                    target_record_id=str(record.get("credit_line_id") or "") or None,
                    field_name=field_name,
                    source_refs=field_refs or record.get("source_refs") or (),
                    reason_codes=(
                        "required_field_missing",
                        "canonical_credit_agreement_field_unresolved",
                        "field_slot_not_safely_bound",
                        "preserved_unknown_value",
                    ),
                ),
            )
    return reconciled


_LIABILITY_LABEL_TO_FIELD = {
    "管理机构": "institution",
    "业务种类": "business_type",
    "开立日期": "open_date",
    "到期日期": "due_date",
    "责任人类型": "responsibility_type",
    "还款责任金额": "responsibility_amount",
    "币种": "currency",
    "保证合同编号": "contract_number",
    "主业务借款人": "related_party_name",
    "主业务借款人证件类型": "related_party_id_type",
    "主业务借款人证件号码": "related_party_id_number",
    "__snapshot_date": "snapshot_date",
    "余额": "balance",
    "五级分类": "five_tier_class",
    "逾期月数": "overdue_months",
    "还款状态": "repayment_status_code",
}

_LIABILITY_PLACEHOLDERS = frozenset({"", "-", "--", "---", "未报告", "不详"})
_LIABILITY_RESPONSIBILITY_TYPES = frozenset(
    {
        "本人",
        "保证",
        "保证人",
        "担保",
        "担保人",
        "共同借款人",
        "共同还款人",
        "抵押",
        "抵押人",
        "质押",
        "质押人",
        "其他",
        "未知",
    }
)
_LIABILITY_ID_TYPES = frozenset(
    {
        "身份证",
        "居民身份证",
        "统一社会信用代码",
        "中征码",
        "组织机构代码",
        "护照",
        "军官证",
        "士兵证",
        "其他证件",
        "未知",
    }
)
_LIABILITY_FIVE_TIER_CLASSES = frozenset({"正常", "关注", "次级", "可疑", "损失", "未分类", "未知"})


def _liability_source_absent(value: Any) -> bool:
    return _compact(value) in _LIABILITY_PLACEHOLDERS


def _liability_currency(value: Any) -> str | None:
    compact = _compact(value).upper()
    return _CURRENCY_TOKEN_ALIASES.get(compact)


def _liability_date(value: Any) -> str | None:
    raw = str(value or "").strip()
    snapshot = re.fullmatch(
        r"截至\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        raw,
    )
    if snapshot is not None:
        return _date("-".join(snapshot.groups()))
    return _date(raw)


def _liability_identifier(value: Any, *, contract: bool = False) -> str | None:
    compact = re.sub(r"\s+", "", str(value or "")).upper()
    minimum = 8 if contract else 6
    if not (minimum <= len(compact) <= 100):
        return None
    if not re.fullmatch(r"[A-Z0-9*\-]+", compact):
        return None
    if contract and not re.search(r"[A-Z0-9]", compact):
        return None
    return compact


def _liability_text(value: Any, *, minimum: int = 1, maximum: int = 120) -> str | None:
    text = _business_text(value)
    if not (minimum <= len(text) <= maximum):
        return None
    if not re.search(r"[\u3400-\u9fffA-Za-z]", text):
        return None
    if re.search(r"[\"'?？]{2,}", text):
        return None
    return text


def _liability_convert(
    field_name: str,
    raw_value: Any,
    *,
    related_party_id_type: Any = None,
) -> Any | None:
    compact = _compact(raw_value)
    if _liability_source_absent(raw_value):
        return None
    if field_name in {"responsibility_amount", "balance"}:
        parsed = _number(raw_value)
        return parsed if isinstance(parsed, int) and parsed >= 0 else None
    if field_name in {"open_date", "due_date", "snapshot_date"}:
        return _liability_date(raw_value)
    if field_name == "currency":
        return _liability_currency(raw_value)
    if field_name == "contract_number":
        return _liability_identifier(raw_value, contract=True)
    if field_name == "related_party_id_number":
        id_type = _compact(related_party_id_type)
        source = re.sub(r"\s+", "", str(raw_value or ""))
        if id_type == "统一社会信用代码":
            return source if re.fullmatch(r"[0-9A-Z]{18}", source) else None
        if id_type == "中征码":
            return source if re.fullmatch(r"[0-9A-Za-z]{16}", source) else None
        return _liability_identifier(raw_value)
    if field_name == "responsibility_type":
        return compact if compact in _LIABILITY_RESPONSIBILITY_TYPES else None
    if field_name == "related_party_id_type":
        return compact if compact in _LIABILITY_ID_TYPES else None
    if field_name == "five_tier_class":
        return compact if compact in _LIABILITY_FIVE_TIER_CLASSES else None
    if field_name == "overdue_months":
        return int(compact) if re.fullmatch(r"\d{1,2}", compact) else None
    if field_name == "repayment_status_code":
        upper = compact.upper()
        if upper in _STATUS_CODES:
            return upper
        return compact if compact in {"正常", "逾期", "结清", "未知"} else None
    if field_name == "related_party_name":
        return _liability_text(raw_value, minimum=2)
    if field_name == "institution":
        return _liability_text(raw_value, minimum=2)
    if field_name == "business_type":
        return _liability_text(raw_value, minimum=2, maximum=60)
    return None


def _liability_party_category(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("_party_category") or "")
    if explicit in {"person", "organization"}:
        return explicit
    id_type = _compact(record.get("related_party_id_type"))
    identifier = _compact(record.get("related_party_id_number")).upper()
    if "身份" in id_type or re.fullmatch(r"\d{17}[0-9X]", identifier):
        return "person"
    if any(marker in id_type for marker in ("统一社会信用", "中征码", "组织机构")):
        return "organization"
    return "unknown"


def _liability_field_provenance_quality(record: Mapping[str, Any], field_name: str) -> int:
    bindings = record.get("_field_binding_quality")
    binding = str(bindings.get(field_name) or "") if isinstance(bindings, Mapping) else ""
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    cell_scoped = any(
        isinstance(ref, Mapping) and ref.get("geometry_scope") == "cell" for ref in refs or ()
    )
    if binding in {"canonical_cell_slot", "canonical_snapshot_date_cell"} and cell_scoped:
        return 4
    if binding == "native_label_column" and cell_scoped:
        return 3
    if binding in {"native_label_column", "canonical_snapshot_date_cell"}:
        return 2
    return 1 if record.get(field_name) not in (None, "") else 0


def _liability_value_key(field_name: str, value: Any) -> str:
    if field_name in {"responsibility_amount", "balance", "overdue_months"}:
        return str(_number(value))
    return _compact(value).upper()


def _liability_records_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_category = _liability_party_category(left)
    right_category = _liability_party_category(right)
    if left_category != "unknown" and right_category != "unknown" and left_category != right_category:
        return False
    left_contract = _compact(left.get("contract_number")).upper()
    right_contract = _compact(right.get("contract_number")).upper()
    if left_contract and right_contract and left_contract == right_contract:
        return True
    left_sequence = left.get("_printed_sequence")
    right_sequence = right.get("_printed_sequence")
    same_canonical_ordinal = bool(
        str(left_sequence or "").isdigit()
        and str(right_sequence or "").isdigit()
        and int(left_sequence) == int(right_sequence)
        and left_category != "unknown"
        and left_category == right_category
    )
    if same_canonical_ordinal:
        return True
    if left_contract and right_contract:
        return False
    if (
        str(left_sequence or "").isdigit()
        and str(right_sequence or "").isdigit()
        and int(left_sequence) != int(right_sequence)
    ):
        return False

    def composite_base(record: Mapping[str, Any]) -> tuple[str, ...] | None:
        balance = record.get("balance")
        values = (
            _compact(record.get("institution")).upper(),
            _compact(record.get("open_date")),
            _compact(record.get("due_date")),
            str(_number(balance)) if balance not in (None, "") else "",
        )
        return values if all(values) else None

    left_composite = composite_base(left)
    right_composite = composite_base(right)
    if left_composite is None or left_composite != right_composite:
        return False
    left_party_id = _compact(left.get("related_party_id_number")).upper()
    right_party_id = _compact(right.get("related_party_id_number")).upper()
    if left_party_id and right_party_id:
        return left_party_id == right_party_id
    left_party_name = _compact(left.get("related_party_name")).upper()
    right_party_name = _compact(right.get("related_party_name")).upper()
    return bool(left_party_name and left_party_name == right_party_name)


def _liability_field_refs(record: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    refs_by_field = record.get("source_refs_by_field")
    refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
    return [dict(ref) for ref in refs or () if isinstance(ref, Mapping)]


def reconcile_candidate_b_liabilities(
    parse_result: Any,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconcile fixed-layout liability cards without fuzzy business matching."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    groups: list[list[dict[str, Any]]] = []
    for observation in observations:
        compatible = [
            group for group in groups if any(_liability_records_compatible(observation, prior) for prior in group)
        ]
        if len(compatible) == 1:
            compatible[0].append(observation)
        else:
            # Ambiguous bridges are never used to collapse business records.
            groups.append([observation])

    reconciled: list[dict[str, Any]] = []
    for output_sequence, group in enumerate(groups, start=1):
        merged_refs: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for observation in group:
            for ref in observation.get("source_refs") or ():
                if not isinstance(ref, Mapping):
                    continue
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    merged_refs.append(dict(ref))

        printed_sequences = {
            int(observation["_printed_sequence"])
            for observation in group
            if str(observation.get("_printed_sequence") or "").isdigit()
        }
        group_contracts = {
            str(observation.get("contract_number") or "")
            for observation in group
            if observation.get("contract_number") not in (None, "")
        }
        group_categories = {
            _liability_party_category(observation)
            for observation in group
            if _liability_party_category(observation) != "unknown"
        }
        identity_seed: Any = next(iter(group_contracts)) if len(group_contracts) == 1 else None
        if identity_seed is None and len(printed_sequences) == 1 and len(group_categories) == 1:
            identity_seed = f"{next(iter(group_categories))}:{next(iter(printed_sequences))}"
        if identity_seed is None:
            identity_seed = json.dumps(merged_refs, ensure_ascii=False, sort_keys=True, default=str)
        liability_id = stable_record_id("repayment_liability", identity_seed or output_sequence)
        selected: dict[str, Any] = {
            "liability_id": liability_id,
            "sequence": output_sequence,
            "source": "candidate_b_repayment_responsibility_schema",
            "source_refs": merged_refs,
            "confidence": max(float(observation.get("confidence") or 0.0) for observation in group),
            "amount_unit": "yuan",
            "reporting_amount_unit": "yuan",
        }
        if len(printed_sequences) == 1:
            selected["_printed_sequence"] = next(iter(printed_sequences))
        if len(group_categories) == 1:
            selected["related_party_category"] = next(iter(group_categories))
        selected_field_refs: dict[str, list[dict[str, Any]]] = {}
        selected_bindings: dict[str, str] = {}
        selected_raw: dict[str, Any] = {}
        source_absent_fields = {
            str(field_name)
            for observation in group
            for field_name in observation.get("_source_absent_fields") or ()
        }
        unresolved_fields = {
            str(field_name)
            for observation in group
            for field_name in observation.get("_unresolved_fields") or ()
        }
        observed_fields = {
            str(field_name)
            for observation in group
            for field_name in observation.get("_observed_fields") or ()
        }
        source_absent_quality = {
            field_name: max(
                (
                    _liability_field_provenance_quality(observation, field_name)
                    for observation in group
                    if field_name in set(observation.get("_source_absent_fields") or ())
                ),
                default=0,
            )
            for field_name in source_absent_fields
        }
        unresolved_quality = {
            field_name: max(
                (
                    _liability_field_provenance_quality(observation, field_name)
                    for observation in group
                    if field_name in set(observation.get("_unresolved_fields") or ())
                ),
                default=0,
            )
            for field_name in unresolved_fields
        }
        invalid_raw: dict[str, list[Any]] = defaultdict(list)
        for observation in group:
            raw_map = observation.get("_invalid_raw_by_field")
            if not isinstance(raw_map, Mapping):
                continue
            for field_name, values in raw_map.items():
                invalid_raw[str(field_name)].extend(values if isinstance(values, list) else [values])
        for field_name, values in invalid_raw.items():
            if values and field_name not in selected_raw:
                selected_raw[field_name] = deepcopy(values[0])

        for field_name in source_absent_fields:
            raw_absences = [
                observation.get("canonical_raw", {}).get(field_name)
                for observation in group
                if field_name in set(observation.get("_source_absent_fields") or ())
                and isinstance(observation.get("canonical_raw"), Mapping)
            ]
            raw_absences = [raw for raw in raw_absences if raw not in (None, "")]
            if raw_absences:
                selected_raw[field_name] = deepcopy(raw_absences[0])

        conflict_fields: set[str] = set()
        for field_name in _LIABILITY_CANONICAL_FIELDS:
            candidates = [
                (
                    _liability_field_provenance_quality(observation, field_name),
                    float(observation.get("confidence") or 0.0),
                    observation.get(field_name),
                    observation,
                )
                for observation in group
                if observation.get(field_name) not in (None, "")
            ]
            if not candidates:
                selected[field_name] = None
                continue
            best_quality = max(candidate[0] for candidate in candidates)
            best = [candidate for candidate in candidates if candidate[0] == best_quality]
            by_value: dict[str, list[tuple[int, float, Any, Mapping[str, Any]]]] = defaultdict(list)
            for candidate in best:
                by_value[_liability_value_key(field_name, candidate[2])].append(candidate)
            if len(by_value) != 1:
                selected[field_name] = None
                unresolved_fields.add(field_name)
                conflict_fields.add(field_name)
                conflict_refs = [
                    ref
                    for _quality, _confidence, _value, observation in best
                    for ref in _liability_field_refs(observation, field_name)
                ]
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_repayment_responsibility_field_conflict",
                        message=(
                            "Equally source-bound observations disagreed for one canonical repayment-"
                            "responsibility field; the value was withheld instead of guessed."
                        ),
                        parser_stage="candidate_b_repayment_responsibility_schema",
                        target_dataset="repayment_liability_records",
                        target_record_id=liability_id,
                        field_name=field_name,
                        observed_value=[candidate[2] for candidate in best],
                        source_refs=conflict_refs or merged_refs,
                        reason_codes=(
                            "closed_canonical_liability_slot",
                            "equal_provenance_conflict",
                            "conflicting_value_withheld",
                        ),
                    ),
                )
                continue
            chosen = max(best, key=lambda candidate: candidate[1])
            selected[field_name] = deepcopy(chosen[2])
            chosen_group = by_value[next(iter(by_value))]
            field_refs: list[dict[str, Any]] = []
            field_ref_markers: set[str] = set()
            for _quality, _confidence, _value, observation in chosen_group:
                for ref in _liability_field_refs(observation, field_name):
                    marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                    if marker not in field_ref_markers:
                        field_ref_markers.add(marker)
                        field_refs.append(ref)
            if field_refs:
                selected_field_refs[field_name] = field_refs
            bindings = chosen[3].get("_field_binding_quality")
            if isinstance(bindings, Mapping) and bindings.get(field_name):
                selected_bindings[field_name] = str(bindings[field_name])
            raw_values = chosen[3].get("canonical_raw")
            if isinstance(raw_values, Mapping) and field_name in raw_values:
                selected_raw[field_name] = deepcopy(raw_values[field_name])

        selected["responsibility_amount_reported"] = isinstance(selected.get("responsibility_amount"), int)
        selected["reporting_amount_currency"] = selected.get("currency")
        selected["source_refs_by_field"] = selected_field_refs
        selected["_field_binding_quality"] = selected_bindings
        selected["_source_absent_fields"] = sorted(source_absent_fields)
        selected["_unresolved_fields"] = sorted(unresolved_fields)
        if invalid_raw:
            selected["_invalid_raw_by_field"] = {
                field_name: list(dict.fromkeys(values))
                for field_name, values in invalid_raw.items()
            }
        if selected_raw:
            selected["canonical_raw"] = selected_raw

        for field_name in _LIABILITY_CANONICAL_FIELDS:
            if selected.get(field_name) not in (None, ""):
                continue
            if field_name in {"overdue_months", "repayment_status_code"} and field_name not in observed_fields:
                # Canonical liability cards print one of these mutually
                # exclusive labels.  The absent alternative is not a missing
                # business value and must not generate a false report.
                continue
            explicit_absence_is_best_evidence = bool(
                field_name in source_absent_fields
                and source_absent_quality.get(field_name, 0) >= unresolved_quality.get(field_name, 0)
            )
            if explicit_absence_is_best_evidence or field_name in conflict_fields:
                continue
            field_invalid_raw = list(dict.fromkeys(invalid_raw.get(field_name) or ()))
            if field_name == "related_party_id_number" and field_invalid_raw:
                # Exact related-party identity can only be repaired after every
                # distinct liability row is available for source-bound
                # corroboration.  The post-pass below reports success or
                # failure without publishing the contaminated observation.
                continue
            field_refs = selected_field_refs.get(field_name) or [
                ref
                for observation in group
                for ref in _liability_field_refs(observation, field_name)
            ]
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error" if field_invalid_raw else "schema_incompleteness",
                    issue_code=(
                        "candidate_b_repayment_responsibility_field_invalid"
                        if field_invalid_raw
                        else "candidate_b_repayment_responsibility_required_field_unresolved"
                    ),
                    message=(
                        "A canonical repayment-responsibility field failed its field contract and was withheld."
                        if field_invalid_raw
                        else "A fixed-layout repayment-responsibility field remained unresolved after correction."
                    ),
                    parser_stage="candidate_b_repayment_responsibility_schema",
                    target_dataset="repayment_liability_records",
                    target_record_id=liability_id,
                    field_name=field_name,
                    observed_value=field_invalid_raw or None,
                    source_refs=field_refs or merged_refs,
                    reason_codes=(
                        "closed_canonical_liability_slot",
                        "field_contract_failed" if field_invalid_raw else "required_field_missing",
                        "value_withheld_not_invented",
                    ),
                ),
            )
        reconciled.append(selected)
    return _corroborate_liability_party_identifiers(parse_result, reconciled)


def _corroborate_liability_party_identifiers(
    parse_result: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Repair a rejected party ID only from one other exact-party liability row."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            _compact(record.get("related_party_name")).upper(),
            str(record.get("related_party_category") or ""),
            _compact(record.get("related_party_id_type")),
        )
        if all(key):
            grouped[key].append(record)

    for group in grouped.values():
        for target in group:
            invalid_map = target.get("_invalid_raw_by_field")
            invalid_values = (
                invalid_map.get("related_party_id_number")
                if isinstance(invalid_map, Mapping)
                else None
            )
            invalid_values = list(dict.fromkeys(invalid_values or ()))
            if target.get("related_party_id_number") not in (None, "") or not invalid_values:
                continue

            id_type = target.get("related_party_id_type")
            corroborators: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for other in group:
                if other is target:
                    continue
                candidate = _liability_convert(
                    "related_party_id_number",
                    other.get("related_party_id_number"),
                    related_party_id_type=id_type,
                )
                if candidate is not None:
                    corroborators[str(candidate)].append(other)

            target_id = str(target.get("liability_id") or "")
            invalid_refs = _liability_field_refs(target, "related_party_id_number")
            if len(corroborators) == 1:
                value, supporting_records = next(iter(corroborators.items()))
                target["related_party_id_number"] = value
                unresolved = set(target.get("_unresolved_fields") or ())
                unresolved.discard("related_party_id_number")
                target["_unresolved_fields"] = sorted(unresolved)
                refs_by_field = target.setdefault("source_refs_by_field", {})
                supporting_refs = [
                    {**ref, "binding": "corroborated_exact_party_identity"}
                    for record in supporting_records
                    for ref in _liability_field_refs(record, "related_party_id_number")
                ]
                refs_by_field["related_party_id_number"] = [
                    *invalid_refs,
                    *supporting_refs,
                ]
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_liability_party_id_corroborated",
                        message=(
                            "A contaminated related-party identifier was replaced only after one "
                            "unique valid identifier was independently printed for the exact same "
                            "party, category, and identifier type."
                        ),
                        severity="info",
                        status="resolved",
                        parser_stage="candidate_b_repayment_responsibility_schema",
                        target_dataset="repayment_liability_records",
                        target_record_id=target_id,
                        field_name="related_party_id_number",
                        observed_value=invalid_values,
                        candidate_value=value,
                        source_refs=[*invalid_refs, *supporting_refs],
                        reason_codes=(
                            "identifier_field_contract_failed",
                            "independent_exact_party_identity_corroboration",
                            "unique_valid_identifier_published",
                        ),
                    ),
                )
                continue

            candidate_values = sorted(corroborators)
            record_issue(
                parse_result,
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_liability_party_id_corroboration_unresolved",
                    message=(
                        "A related-party identifier failed its exact type-specific contract and no "
                        "unique independently printed identifier could repair it; the value remains "
                        "withheld."
                    ),
                    parser_stage="candidate_b_repayment_responsibility_schema",
                    target_dataset="repayment_liability_records",
                    target_record_id=target_id,
                    field_name="related_party_id_number",
                    observed_value=invalid_values,
                    candidate_value=candidate_values or None,
                    source_refs=invalid_refs or target.get("source_refs") or (),
                    reason_codes=(
                        "identifier_field_contract_failed",
                        (
                            "ambiguous_independent_party_identifiers"
                            if candidate_values
                            else "independent_party_identifier_missing"
                        ),
                        "value_withheld_not_invented",
                    ),
                ),
            )
    return records


def _extract_liabilities(parse_result: Any) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    observations: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(
        PBOCPersonalDetailNativeParser(parse_result).records("repayment_liability_records"), start=1
    ):
        facts = candidate.fields
        observation: dict[str, Any] = {
            "_printed_sequence": facts.get("__printed_sequence"),
            "_party_category": facts.get("__party_category"),
            "source": "candidate_b_repayment_responsibility_observation",
            "source_refs": list(candidate.source_refs),
            "confidence": candidate.confidence,
            "source_refs_by_field": {},
            "_field_binding_quality": {},
            "_source_absent_fields": [],
            "_unresolved_fields": [],
            "_invalid_raw_by_field": {},
            "_observed_fields": [
                _LIABILITY_LABEL_TO_FIELD[label]
                for label in _LIABILITY_LABEL_TO_FIELD
                if label in candidate.observed_labels or facts.get(label) not in (None, "")
            ],
            "canonical_raw": {},
        }
        source_absent_fields: set[str] = set()
        unresolved_fields = {
            _LIABILITY_LABEL_TO_FIELD[label]
            for label in candidate.unresolved_labels
            if label in _LIABILITY_LABEL_TO_FIELD
        }
        for label, field_name in _LIABILITY_LABEL_TO_FIELD.items():
            raw_value = facts.get(label)
            if raw_value in (None, ""):
                continue
            observation["canonical_raw"][field_name] = raw_value
            if _liability_source_absent(raw_value):
                source_absent_fields.add(field_name)
                observation[field_name] = None
            else:
                converted = _liability_convert(
                    field_name,
                    raw_value,
                    related_party_id_type=observation.get("related_party_id_type"),
                )
                observation[field_name] = converted
                if converted is None:
                    unresolved_fields.add(field_name)
                    observation["_invalid_raw_by_field"].setdefault(field_name, []).append(raw_value)

            raw_refs = candidate.source_refs_by_field.get(label) or ()
            if raw_refs:
                observation["source_refs_by_field"].setdefault(field_name, []).extend(
                    dict(ref) for ref in raw_refs if isinstance(ref, Mapping)
                )
            binding = candidate.binding_quality_by_field.get(label)
            if binding:
                observation["_field_binding_quality"][field_name] = binding

        observation["_source_absent_fields"] = sorted(source_absent_fields)
        observation["_unresolved_fields"] = sorted(unresolved_fields)
        has_identity = observation.get("contract_number") not in (None, "")
        has_amount = isinstance(observation.get("responsibility_amount"), int)
        has_sequence = str(observation.get("_printed_sequence") or "").isdigit()
        if not (has_identity or has_amount or has_sequence):
            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_repayment_responsibility_identity_unresolved",
                    message=(
                        "A canonical repayment-responsibility card was observed but no contract, printed "
                        "ordinal, or valid responsibility amount could identify it."
                    ),
                    parser_stage="candidate_b_repayment_responsibility_schema",
                    target_dataset="repayment_liability_records",
                    observed_value=observation.get("_invalid_raw_by_field") or None,
                    source_refs=candidate.source_refs,
                    reason_codes=(
                        "canonical_responsibility_card",
                        "record_identity_unresolved",
                        "record_withheld",
                    ),
                ),
            )
            continue
        # Keep observations separate until exact contract/ordinal reconciliation.
        observation["_observation_index"] = candidate_index
        observations.append(observation)
    return reconcile_candidate_b_liabilities(parse_result, observations)


def _normalize_inquiry_reason(value: str) -> str:
    text = str(value or "")
    for observed, canonical in _INQUIRY_REASON_REPAIRS.items():
        text = text.replace(observed, canonical)
    return text


def _longest_inquiry_reason_suffix(value: str) -> str:
    """Return the longest closed-vocabulary inquiry reason printed as a suffix."""

    text = _normalize_inquiry_reason(value).strip()
    matches = [reason for reason in _INQUIRY_REASONS if text.endswith(reason)]
    return max(matches, key=len, default="")


def _repair_inquiry_reason_boundary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Move a finite long-reason prefix out of a contaminated institution cell."""

    repaired = dict(record)
    institution = _compact(record.get("institution"))
    reason = _compact(_normalize_inquiry_reason(str(record.get("reason") or "")))
    if not institution or not reason:
        return repaired
    candidates = [
        canonical
        for canonical in _INQUIRY_REASONS
        if len(canonical) > len(reason)
        and canonical.endswith(reason)
        and institution.endswith(canonical[: -len(reason)])
    ]
    if not candidates:
        return repaired
    canonical = max(candidates, key=len)
    prefix = canonical[: -len(reason)]
    repaired_institution = institution[: -len(prefix)]
    if repaired_institution:
        repaired["institution"] = repaired_institution
        repaired["reason"] = canonical
    return repaired


def _inquiry_geometry_groups(lines: Any) -> list[list[dict[str, Any]]]:
    """Join OCR tokens that occupy one canonical inquiry-table row."""
    positioned: list[tuple[float, float, dict[str, Any]]] = []
    for index, line in enumerate(lines or ()):
        if not isinstance(line, dict):
            continue
        bbox = line.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                top, left = float(bbox[1]), float(bbox[0])
            except (TypeError, ValueError):
                top, left = float(index * 20), 0.0
        else:
            top, left = float(index * 20), 0.0
        positioned.append((top, left, line))
    positioned.sort(key=lambda item: (item[0], item[1]))

    groups: list[list[tuple[float, float, dict[str, Any]]]] = []
    for top, left, line in positioned:
        if groups and abs(top - sum(item[0] for item in groups[-1]) / len(groups[-1])) <= 3.5:
            groups[-1].append((top, left, line))
        else:
            groups.append([(top, left, line)])
    return [
        [line for _top, _left, line in sorted(group, key=lambda item: item[1])]
        for group in groups
    ]


def _inquiry_sequence_token(value: Any) -> int | None:
    """Return one exact printed inquiry ordinal, without guessing its value."""

    match = re.fullmatch(
        r"[\s.,，。:：;；()（）\[\]【】]*(\d{1,4})"
        r"[\s.,，。:：;；()（）\[\]【】]*",
        _clean(value),
    )
    if match is None:
        return None
    sequence = int(match.group(1))
    return sequence if sequence > 0 else None


def _bounded_inquiry_sequence_noise_candidate(
    value: Any,
) -> tuple[int, str] | None:
    """Keep one edge glyph only as a neighbour-proven ordinal candidate."""

    marker = _clean(value)
    prefixed = re.fullmatch(r"[A-Za-z\u3400-\u9fff]\s*(\d{1,4})", marker)
    if prefixed is not None:
        sequence = int(prefixed.group(1))
        return (sequence, "prefixed_noise") if sequence > 0 else None
    suffixed = re.fullmatch(r"(\d{1,4})\s*[A-Za-z\u3400-\u9fff]", marker)
    if suffixed is not None:
        sequence = int(suffixed.group(1))
        return (sequence, "suffix_noise") if sequence > 0 else None
    return None


def _document_local_inquiry_ordinals(
    raw_sequences: Iterable[int | None],
    *,
    noisy_candidates: Iterable[tuple[int, str] | None] | None = None,
) -> list[tuple[int | None, str | None]]:
    """Normalize inquiry ordinals only from exact adjacent-row proof.

    A damaged middle row may be inferred when its two immediate neighbours
    prove the sole missing ordinal; a final complete row may continue its one
    immediate predecessor. Likewise, a high OCR value may shed a leading
    prefix only when both neighbours prove its complete suffix
    (``88, 789, 90`` -> ``88, 89, 90``). This deliberately preserves isolated
    high values, legitimate 100+ populations, and genuine ``788, 789, 790``.
    """

    observed = [
        int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None
        for value in raw_sequences
    ]
    bounded_noise = (
        list(noisy_candidates)
        if noisy_candidates is not None
        else [None for _value in observed]
    )
    if len(bounded_noise) != len(observed):
        return [(value, None) for value in observed]
    missing_indices = [
        index for index, value in enumerate(observed) if value is None
    ]
    normalized: list[tuple[int | None, str | None]] = [
        (
            value,
            "multiple_missing"
            if value is None and len(missing_indices) > 1
            else None,
        )
        for value in observed
    ]
    for index in range(1, len(observed) - 1):
        previous = observed[index - 1]
        current = observed[index]
        following = observed[index + 1]
        if previous is None or following is None or following != previous + 2:
            continue
        expected = previous + 1
        if current is None:
            noisy = bounded_noise[index]
            if (
                isinstance(noisy, tuple)
                and len(noisy) == 2
                and noisy[0] == expected
                and noisy[1] in {"prefixed_noise", "suffix_noise"}
            ):
                normalized[index] = (expected, str(noisy[1]))
                continue
            # More than one unreadable ordinal admits multiple equally
            # plausible assignments. Keep every missing value unresolved,
            # while still allowing independent nonmissing prefix-noise proof
            # elsewhere in the same canonical population.
            if len(missing_indices) == 1:
                normalized[index] = (expected, "missing")
            continue
        if current == expected or current < 300 or current <= expected:
            continue
        if str(current).endswith(str(expected)):
            normalized[index] = (expected, "prefixed_noise")
    if (
        len(missing_indices) == 1
        and len(observed) >= 2
        and observed[-1] is None
        and observed[-2] is not None
    ):
        normalized[-1] = (int(observed[-2]) + 1, "missing")
    return normalized


_BOUNDED_INQUIRY_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[./:-]?(\d{2})[./:-](\d{2})(?!\d)"
)


def _bounded_inquiry_date(value: Any) -> str | None:
    """Recover one valid date token with only tiny edge OCR residue."""

    exact = _date(value)
    if exact is not None:
        return exact
    text = str(value or "")
    matches = list(_BOUNDED_INQUIRY_DATE_RE.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    candidate = _date(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
    if candidate is None:
        return None
    residue = re.sub(
        r"[\W_]+",
        "",
        f"{text[:match.start()]}{text[match.end():]}",
        flags=re.UNICODE,
    )
    if len(residue) > 3 or len(re.findall(r"[\u3400-\u9fff]", residue)) > 1:
        return None
    return candidate


def _collapsed_inquiry_header_geometry_exact(table: Any) -> bool | None:
    """Validate the observed 4-column header with its middle two-cell span."""

    geometry = _native_table_geometry(table)
    if geometry is None:
        return None
    row_bands = _indexed_geometry_bands(
        geometry,
        "row_bands",
        lower_key="y0",
        upper_key="y1",
    )
    column_bands = _indexed_geometry_bands(
        geometry,
        "col_bands",
        lower_key="x0",
        upper_key="x1",
    )
    if (
        row_bands is None
        or 0 not in row_bands
        or column_bands is None
        or set(column_bands) != {0, 1, 2, 3}
    ):
        return False
    cell_bboxes = geometry.get("cell_bboxes")
    cell_status = geometry.get("cell_geometry_status")
    cell_evidence_ids = geometry.get("cell_evidence_ids")
    if not all(
        isinstance(grid, list)
        and grid
        and isinstance(grid[0], list)
        and len(grid[0]) == 4
        for grid in (cell_bboxes, cell_status, cell_evidence_ids)
    ):
        return False
    if tuple(str(value or "") for value in cell_status[0]) != (
        "exact",
        "exact",
        "derived",
        "exact",
    ):
        return False
    if not all(
        isinstance(cell_evidence_ids[0][column], list)
        and any(str(value or "") for value in cell_evidence_ids[0][column])
        for column in (0, 1, 3)
    ):
        return False
    if cell_evidence_ids[0][2] not in ([], None):
        return False
    expected_boxes = {
        0: (
            column_bands[0][0],
            row_bands[0][0],
            column_bands[0][1],
            row_bands[0][1],
        ),
        1: (
            column_bands[1][0],
            row_bands[0][0],
            column_bands[2][1],
            row_bands[0][1],
        ),
        3: (
            column_bands[3][0],
            row_bands[0][0],
            column_bands[3][1],
            row_bands[0][1],
        ),
    }
    for column, expected in expected_boxes.items():
        bbox = _exact_geometry_bbox(cell_bboxes[0][column])
        if bbox is None or any(
            abs(left - right) > 1.0
            for left, right in zip(bbox, expected, strict=True)
        ):
            return False
    if cell_bboxes[0][2] is not None:
        return False
    spans = geometry.get("cell_spans")
    if not isinstance(spans, list):
        return False
    header_spans = [
        span
        for span in spans
        if isinstance(span, Mapping) and span.get("row") == 0
    ]
    return bool(
        len(header_spans) == 1
        and header_spans[0].get("col") == 1
        and header_spans[0].get("row_span") == 1
        and header_spans[0].get("col_span") == 2
    )


def _bounded_collapsed_inquiry_header_slots(
    rows: list[list[str]],
    *,
    table: Any | None = None,
) -> dict[str, int] | None:
    """Recover the fixed four-column header only when its body proves it.

    The four complete labels must remain in canonical order, with only tiny
    non-business OCR residue.  Every non-empty body row must then satisfy the
    date, institution, and inquiry-reason contracts in the canonical columns,
    and its ordinals must be dense after at most one neighbour-proven missing
    value.  A malformed edge row, mixed population, or ambiguous column layout
    leaves the header unresolved.
    """

    if not rows or len(rows[0]) != 4:
        return None
    geometry_exact = (
        _collapsed_inquiry_header_geometry_exact(table)
        if table is not None
        else None
    )
    if geometry_exact is False:
        return None
    header_text = _compact("".join(str(value or "") for value in rows[0]))
    sequence_labels = [label for label in ("编号", "序号") if header_text.count(label) == 1]
    if len(sequence_labels) != 1:
        return None
    labels = (sequence_labels[0], "查询日期", "查询机构", "查询原因")
    if any(header_text.count(label) != 1 for label in labels):
        return None
    positions = [header_text.index(label) for label in labels]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        return None
    residue = header_text
    for label in labels:
        residue = residue.replace(label, "", 1)
    if len(residue) > 4 or re.search(r"[\u3400-\u9fff0-9]", residue):
        return None

    body = [row for row in rows[1:] if _nonempty(row)]
    if len(body) < 3 or any(len(row) != 4 for row in body):
        return None
    raw_sequences = [_inquiry_sequence_token(row[0]) for row in body]
    normalized_ordinals = _document_local_inquiry_ordinals(
        raw_sequences,
        noisy_candidates=(
            _bounded_inquiry_sequence_noise_candidate(row[0]) for row in body
        ),
    )
    sequences = [sequence for sequence, _repair in normalized_ordinals]
    observed_sequences = [
        int(sequence) for sequence in sequences if sequence is not None
    ]
    if len(observed_sequences) < 3:
        return None
    if any(
        right <= left
        for left, right in zip(
            observed_sequences[:-1],
            observed_sequences[1:],
            strict=True,
        )
    ):
        return None

    inquiry_types: set[str] = set()
    for row in body:
        if _bounded_inquiry_date(row[1]) is None:
            return None
        institution = _compact(_normalized_inquiry_field("institution", row[2]))
        reason = _normalized_inquiry_field("reason", row[3])
        if (
            not institution
            or len(institution) > 120
            or re.search(r"[\u3400-\u9fffA-Za-z]", institution) is None
            or reason not in _INQUIRY_REASONS
        ):
            return None
        inquiry_types.add(
            "personal"
            if institution == "本人" or reason.startswith("本人查询")
            else "institution"
        )
    if len(inquiry_types) != 1:
        return None
    return {
        "sequence": 0,
        "inquiry_date": 1,
        "institution": 2,
        "reason": 3,
    }


def _record_inquiry_ordinal_repair(
    parse_result: Any,
    *,
    inquiry_type: str,
    sequence: int,
    raw_sequence: int | None,
    inquiry_id: str,
    source_ref: Mapping[str, Any],
    repair_kind: str,
) -> None:
    """Publish the shared row-order repair without changing business fields."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    inferred = repair_kind == "missing"
    suffix_noise = repair_kind == "suffix_noise"
    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code=(
                "candidate_b_inquiry_sequence_inferred_from_row_order"
                if inferred
                else (
                    "candidate_b_inquiry_sequence_suffix_noise_corrected"
                    if suffix_noise
                    else "candidate_b_inquiry_sequence_prefix_noise_corrected"
                )
            ),
            message=(
                "An inquiry row number was unreadable and was inferred from canonical table order."
                if inferred
                else (
                    "A suffixed OCR glyph in an inquiry sequence was removed by exact adjacent-row proof."
                    if suffix_noise
                    else "A prefixed OCR glyph in an inquiry sequence was removed by exact adjacent-row proof."
                )
            ),
            severity="warning" if inferred else "info",
            status="requires_review" if inferred else "resolved",
            parser_stage="candidate_b_inquiry_schema",
            target_dataset="inquiry_records",
            target_record_id=inquiry_id,
            field_name="sequence",
            observed_value={
                "inquiry_type": inquiry_type,
                "missing_ocr_sequence": inferred,
                "raw_sequence": raw_sequence,
            },
            candidate_value={"normalized_sequence": sequence},
            source_refs=(dict(source_ref),),
            reason_codes=(
                "canonical_four_column_table",
                "exact_adjacent_row_sequence_proof",
                (
                    "sequence_missing_in_source_ocr"
                    if inferred
                    else (
                        "sequence_suffixed_ocr_noise"
                        if suffix_noise
                        else "sequence_prefixed_ocr_noise"
                    )
                ),
                (
                    "sequence_requires_review"
                    if inferred
                    else "deterministic_sequence_prefix_removed"
                ),
                "other_row_fields_verified_independently",
            ),
        ),
    )


def _record_inquiry_ordinal_unresolved(
    parse_result: Any,
    *,
    inquiry_type: str,
    source_ref: Mapping[str, Any],
    observed_row: Any,
) -> None:
    """Localize a row withheld because its population has two missing ordinals."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    target_record_id = stable_record_id(
        "credit_inquiry_unresolved_sequence",
        inquiry_type,
        source_ref.get("logical_page"),
        source_ref.get("source_page"),
        source_ref.get("table_id"),
        source_ref.get("row"),
        source_ref.get("bbox"),
    )
    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_inquiry_multiple_missing_sequences_unresolved",
            message=(
                "More than one ordinal was unreadable in one canonical inquiry population; "
                "the affected row was retained only as localized unresolved source evidence."
            ),
            severity="warning",
            status="requires_review",
            parser_stage="candidate_b_inquiry_schema",
            target_dataset="inquiry_records",
            target_record_id=target_record_id,
            field_name="sequence",
            observed_value={
                "inquiry_type": inquiry_type,
                "missing_ocr_sequence": True,
                "row": observed_row,
            },
            source_refs=(dict(source_ref),),
            reason_codes=(
                "canonical_four_column_table",
                "multiple_missing_ordinals_in_population",
                "ordinal_assignment_not_unique",
                "record_not_emitted",
            ),
        ),
    )


def _inquiry_order_plane_is_authoritative(parse_result: Any) -> bool:
    """Validate an explicitly registered inquiry reading-order plane.

    Tiny legacy/unit contexts that predate the plane retain their sealed input
    order. Once a context exposes resolution metadata, however, cross-page
    inquiry logic may use it only when both resolution and authority are
    affirmative and the registered positions are positive and unique.
    """

    if not hasattr(parse_result, "reading_order_resolution"):
        return True
    resolution = getattr(parse_result, "reading_order_resolution", None)
    if not (
        isinstance(resolution, Mapping)
        and resolution.get("resolved") is True
        and resolution.get("authoritative") is True
    ):
        return False
    raw_order = getattr(parse_result, "reading_order_by_logical", None)
    if not isinstance(raw_order, Mapping) or not raw_order:
        return False
    positions: list[int] = []
    try:
        for raw_logical, raw_position in raw_order.items():
            logical = int(raw_logical)
            position = int(raw_position)
            if logical <= 0 or position <= 0:
                return False
            positions.append(position)
    except (TypeError, ValueError):
        return False
    if len(positions) != len(set(positions)):
        return False
    registered_pages = list(getattr(parse_result, "pages", None) or [])
    return not registered_pages or _account_reading_order_resolution(
        parse_result,
        registered_pages,
    )[-1]


def _document_page_carry_allowed(
    parse_result: Any,
    left_logical_page: int | None,
    right_logical_page: int,
) -> bool:
    """Allow same-page reuse or one complete authoritative adjacent page edge."""

    left = int(left_logical_page or 0)
    right = int(right_logical_page or 0)
    if left <= 0 or right <= 0:
        return False
    if left == right:
        return True
    if not hasattr(parse_result, "reading_order_resolution"):
        # Compatibility for sealed contexts created before the order plane:
        # retain their former explicit-map/logical adjacency contract.  Every
        # real context publishing resolution metadata takes the strict path.
        raw_order = getattr(parse_result, "reading_order_by_logical", None)
        if isinstance(raw_order, Mapping) and raw_order:
            return _registered_account_pages_are_adjacent(parse_result, left, right)
        return right == left + 1
    if not _inquiry_order_plane_is_authoritative(parse_result):
        return False
    return _registered_account_pages_are_adjacent(parse_result, left, right)


def _inquiry_schema_carry_allowed(
    parse_result: Any,
    left_logical_page: int | None,
    right_logical_page: int,
) -> bool:
    """Apply the shared document-page ownership contract to inquiry schemas."""

    return _document_page_carry_allowed(
        parse_result,
        left_logical_page,
        right_logical_page,
    )


def _record_inquiry_schema_carry_unresolved(
    parse_result: Any,
    *,
    left_logical_page: int | None,
    page: Any,
    table: Any,
) -> None:
    """Localize one headerless inquiry row withheld at an unproven page edge."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    right_logical_page = int(getattr(page, "page_number", 0) or 0)
    resolution = getattr(parse_result, "reading_order_resolution", None)
    raw_order = getattr(parse_result, "reading_order_by_logical", None)
    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_inquiry_cross_page_schema_unresolved",
            message=(
                "A headerless inquiry row was withheld because its preceding "
                "four-column schema was not adjacent in an authoritative "
                "reading-order plane."
            ),
            parser_stage="candidate_b_inquiry_schema",
            target_dataset="inquiry_records",
            observed_value={
                "schema_logical_page": int(left_logical_page or 0),
                "candidate_logical_page": right_logical_page,
                "reading_order_resolution": (
                    dict(resolution) if isinstance(resolution, Mapping) else None
                ),
                "registered_reading_order": (
                    dict(raw_order) if isinstance(raw_order, Mapping) else None
                ),
            },
            source_refs=(_source_ref(page, table, row=0),),
            reason_codes=(
                "headerless_inquiry_schema_carry",
                "authoritative_adjacent_page_edge_required",
                "record_not_emitted",
            ),
        ),
    )


def _canonical_inquiry_line_rows(parse_result: Any) -> list[dict[str, Any]]:
    """Reconstruct known inquiry-template rows from canonical line geometry."""
    evidence_loader = getattr(parse_result, "corrected_evidence_pages", None)
    if not callable(evidence_loader):
        return []
    candidates: list[dict[str, Any]] = []
    for page in evidence_loader():
        if str(page.get("canonical_template_id") or "") != "annotations_and_inquiries":
            continue
        for group in _inquiry_geometry_groups(page.get("lines") or ()):
            text = _normalize_inquiry_reason(
                " ".join(str(line.get("text") or line.get("content") or "").strip() for line in group)
            )
            date_match = re.search(r"(?:19|20)\d{2}[.,/-]\d{1,2}[.,/-]\d{1,2}", text)
            if date_match is None:
                continue
            reason = _longest_inquiry_reason_suffix(text[date_match.end() :])
            if not reason:
                continue
            reason_index = text.find(reason, date_match.end())
            institution = re.sub(
                r"^[^\u3400-\u9fffA-Za-z]+",
                "",
                text[date_match.end() : reason_index],
            ).strip()
            inquiry_type = "personal" if reason.startswith("本人查询") or institution == "本人" else "institution"
            if inquiry_type == "personal" and not institution:
                institution = "本人"
            if not institution:
                continue
            raw_sequence_text = text[: date_match.start()].strip()
            detected = _inquiry_sequence_token(raw_sequence_text)
            noisy_sequence = _bounded_inquiry_sequence_noise_candidate(
                raw_sequence_text
            )
            inquiry_date = _date(date_match.group(0).replace(",", "."))
            if inquiry_date is None:
                continue
            boxes = [line.get("bbox") for line in group if isinstance(line.get("bbox"), list) and len(line["bbox"]) == 4]
            bbox = (
                [
                    min(float(box[0]) for box in boxes),
                    min(float(box[1]) for box in boxes),
                    max(float(box[2]) for box in boxes),
                    max(float(box[3]) for box in boxes),
                ]
                if boxes
                else []
            )
            source_ref = {
                "source": "candidate_b_canonical_inquiry_line",
                "logical_page": int(page.get("page") or 0),
                "source_page": int(page.get("source_page") or 0),
                "bbox": bbox,
                "geometry_scope": "row",
                "evidence_ids": list(
                    dict.fromkeys(
                        str(evidence_id)
                        for line in group
                        for evidence_id in (line.get("evidence_ids") or ())
                        if evidence_id
                    )
                ),
            }
            candidates.append(
                {
                    "_raw_sequence": detected,
                    "_noisy_sequence": noisy_sequence,
                    "_source_ref": source_ref,
                    "_confidence": min(
                        min(
                            (
                                float(line.get("confidence") or 0.0)
                                for line in group
                            ),
                            default=0.0,
                        ),
                        0.8,
                    ),
                    "inquiry_type": inquiry_type,
                    "inquiry_date": inquiry_date,
                    "institution": institution,
                    "reason": reason,
                    "source_reason": text[reason_index:].strip(),
                }
            )

    document_order_is_authoritative = _inquiry_order_plane_is_authoritative(
        parse_result
    )
    normalization_groups: defaultdict[tuple[str, int | None], list[int]] = (
        defaultdict(list)
    )
    for index, candidate in enumerate(candidates):
        inquiry_type = str(candidate["inquiry_type"])
        source_ref = candidate.get("_source_ref") or {}
        logical_page = int(source_ref.get("logical_page") or 0)
        normalization_groups[
            (inquiry_type, None if document_order_is_authoritative else logical_page)
        ].append(index)

    normalized_by_index: dict[int, tuple[int | None, str | None]] = {}
    for indices in normalization_groups.values():
        normalized = _document_local_inquiry_ordinals(
            (candidates[index].get("_raw_sequence") for index in indices),
            noisy_candidates=(
                candidates[index].get("_noisy_sequence") for index in indices
            ),
        )
        normalized_by_index.update(zip(indices, normalized, strict=True))

    rows: list[dict[str, Any]] = []
    last_sequence: defaultdict[tuple[str, int | None], int] = defaultdict(int)
    for index, candidate in enumerate(candidates):
        sequence, repair_kind = normalized_by_index.get(index, (None, None))
        inquiry_type = str(candidate["inquiry_type"])
        source_ref = dict(candidate["_source_ref"])
        logical_page = int(source_ref.get("logical_page") or 0)
        sequence_scope = (
            inquiry_type,
            None if document_order_is_authoritative else logical_page,
        )
        if sequence is None and repair_kind == "multiple_missing":
            _record_inquiry_ordinal_unresolved(
                parse_result,
                inquiry_type=inquiry_type,
                source_ref=source_ref,
                observed_row={
                    "raw_sequence": candidate.get("_raw_sequence"),
                    "inquiry_date": candidate.get("inquiry_date"),
                    "institution": candidate.get("institution"),
                    "reason": candidate.get("reason"),
                },
            )
        if sequence is None or sequence <= last_sequence[sequence_scope]:
            continue
        last_sequence[sequence_scope] = sequence
        inquiry_id = stable_record_id("credit_inquiry", inquiry_type, sequence)
        rows.append(
            {
                "inquiry_id": inquiry_id,
                "sequence": sequence,
                "inquiry_date": candidate["inquiry_date"],
                "institution": candidate["institution"],
                "reason": candidate["reason"],
                "source_reason": candidate["source_reason"],
                "query_channel": inquiry_type,
                "inquiry_type": inquiry_type,
                "source": "candidate_b_canonical_inquiry_line",
                "source_refs": [source_ref],
                "confidence": candidate["_confidence"],
            }
        )
        if repair_kind:
            _record_inquiry_ordinal_repair(
                parse_result,
                inquiry_type=inquiry_type,
                sequence=sequence,
                raw_sequence=candidate.get("_raw_sequence"),
                inquiry_id=inquiry_id,
                source_ref=source_ref,
                repair_kind=repair_kind,
            )
    return rows


def _normalized_inquiry_field(field_name: str, value: Any) -> str:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
        normalize_institution_name,
        normalize_role_candidate,
    )

    if field_name == "institution":
        normalized = normalize_institution_name(str(value or ""))
        compact = re.sub(r"\s+", "", normalized)
        # PBOC personal-query rows use 本人 as the institution.  Permit only a
        # tiny amount of surrounding OCR debris so bank names cannot collapse
        # into this special value.
        if "本人" in compact and len(compact.replace("本人", "")) <= 2:
            return "本人"
        return normalized

    if field_name == "reason":
        normalized = normalize_role_candidate(value, "inquiry_reason")
        compact = re.sub(r"\s+", "", normalized)
        for canonical in sorted(_INQUIRY_REASONS, key=len, reverse=True):
            if canonical in compact:
                return canonical
        return normalized

    role = {
        "inquiry_date": "date",
    }[field_name]
    return normalize_role_candidate(value, role)


def _inquiry_business_equivalent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Compare normalized business cells without using them as ordinal proof."""

    repaired_left = _repair_inquiry_reason_boundary(left)
    repaired_right = _repair_inquiry_reason_boundary(right)
    if any(
        _normalized_inquiry_field(field_name, repaired_left.get(field_name))
        != _normalized_inquiry_field(field_name, repaired_right.get(field_name))
        for field_name in ("inquiry_date", "reason")
    ):
        return False
    left_institution = re.sub(
        r"\s+",
        "",
        _normalized_inquiry_field("institution", repaired_left.get("institution")),
    )
    right_institution = re.sub(
        r"\s+",
        "",
        _normalized_inquiry_field("institution", repaired_right.get("institution")),
    )
    return bool(left_institution) and left_institution == right_institution


def _inquiry_observation_score(record: Mapping[str, Any]) -> tuple[int, int, float]:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import role_candidate_is_valid

    valid = sum(
        role_candidate_is_valid(record.get(field), role)
        for field, role in (
            ("inquiry_date", "date"),
            ("institution", "institution_name"),
            ("reason", "inquiry_reason"),
        )
    )
    populated = sum(record.get(field) not in (None, "") for field in ("inquiry_date", "institution", "reason"))
    refs_by_field = record.get("source_refs_by_field")
    exact_fields = sum(
        any(
            isinstance(ref, Mapping)
            and ref.get("geometry_scope") == "cell"
            and ref.get("binding") == "canonical_header_column"
            for ref in refs_by_field.get(field_name) or ()
        )
        for field_name in ("inquiry_date", "institution", "reason")
    ) if isinstance(refs_by_field, Mapping) else 0
    return exact_fields * 10 + valid, populated, float(record.get("confidence") or 0.0)


def _extract_inquiries(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    active_canonical_table = False
    active_slots: dict[str, int] = {}
    active_schema_page: int | None = None
    inquiry_aliases = {
        "sequence": ("编号", "序号"),
        "inquiry_date": ("查询日期",),
        "institution": ("查询机构",),
        "reason": ("查询原因",),
    }
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        page_is_canonical_inquiry = (
            str(getattr(page, "canonical_template_id", "") or "") == "annotations_and_inquiries"
        )
        page_had_inquiry_table = False
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            header = _nonempty(rows[0])
            compact_header = _compact("".join(header))
            has_header = all(marker in compact_header for marker in ("查询日期", "查询机构", "查询原因"))
            header_slots = _canonical_header_slots(tuple(str(value or "") for value in rows[0]), inquiry_aliases)
            has_exact_header = has_header and set(header_slots) == set(inquiry_aliases)
            repaired_header_slots = (
                _bounded_collapsed_inquiry_header_slots(rows, table=table)
                if has_header and not has_exact_header
                else None
            )
            has_repaired_header = repaired_header_slots is not None
            looks_like_continuation = (
                not has_header
                and page_is_canonical_inquiry
                and active_canonical_table
                and set(active_slots) == set(inquiry_aliases)
                and bool(_nonempty(rows[0]))
                and re.fullmatch(r"\d{1,4}", _nonempty(rows[0])[0]) is not None
            )
            carry_allowed = _inquiry_schema_carry_allowed(
                parse_result,
                active_schema_page,
                page_number,
            )
            is_continuation = looks_like_continuation and carry_allowed
            if looks_like_continuation and not carry_allowed:
                _record_inquiry_schema_carry_unresolved(
                    parse_result,
                    left_logical_page=active_schema_page,
                    page=page,
                    table=table,
                )
                active_canonical_table = False
                active_slots = {}
                active_schema_page = None
                continue
            if has_header and not has_exact_header and not has_repaired_header:
                from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                    make_issue,
                    record_issue,
                )

                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_inquiry_header_columns_unresolved",
                        message="The canonical inquiry header was visible but its four fields did not resolve to distinct columns.",
                        parser_stage="candidate_b_inquiry_schema",
                        target_dataset="inquiry_records",
                        observed_value={"header": rows[0], "resolved_roles": sorted(header_slots)},
                        source_refs=(_source_ref(page, table, row=0),),
                        reason_codes=(
                            "closed_canonical_inquiry_template",
                            "distinct_header_columns_required",
                            "rows_withheld_until_page_repair",
                        ),
                    ),
                )
                active_canonical_table = False
                active_slots = {}
                active_schema_page = None
                continue
            if not has_exact_header and not has_repaired_header and not is_continuation:
                continue
            page_had_inquiry_table = True
            active_canonical_table = True
            if has_exact_header:
                active_slots = header_slots
                active_schema_page = page_number
            elif has_repaired_header:
                active_slots = dict(repaired_header_slots)
                active_schema_page = page_number
            elif is_continuation:
                active_schema_page = page_number
            slots = dict(active_slots)
            start = 1 if has_exact_header or has_repaired_header else 0
            table_rows = [
                (row_index, tuple(str(value or "").strip() for value in row))
                for row_index, row in enumerate(rows[start:], start=start)
                if _nonempty(row)
            ]
            normalized_ordinals = _document_local_inquiry_ordinals(
                _inquiry_sequence_token(_slot_value(cells, slots, "sequence"))
                for _row_index, cells in table_rows
            )
            for (row_index, cells), (sequence, repair_kind) in zip(
                table_rows, normalized_ordinals, strict=True
            ):
                raw_sequence = _inquiry_sequence_token(
                    _slot_value(cells, slots, "sequence")
                )
                if sequence is None:
                    if repair_kind == "multiple_missing":
                        unresolved_institution = _slot_value(
                            cells, slots, "institution"
                        )
                        unresolved_reason = _slot_value(cells, slots, "reason")
                        unresolved_type = (
                            "personal"
                            if unresolved_institution == "本人"
                            or unresolved_reason.startswith("本人查询")
                            else "institution"
                        )
                        _record_inquiry_ordinal_unresolved(
                            parse_result,
                            inquiry_type=unresolved_type,
                            source_ref=_source_ref(page, table, row=row_index),
                            observed_row=list(cells),
                        )
                    continue
                date_cell = _slot_value(cells, slots, "inquiry_date")
                inquiry_date = _bounded_inquiry_date(date_cell)
                institution = _slot_value(cells, slots, "institution")
                source_reason = _slot_value(cells, slots, "reason")
                if not institution or not source_reason or inquiry_date is None:
                    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                        make_issue,
                        record_issue,
                    )

                    missing_fields = [
                        field_name
                        for field_name, value in (
                            ("inquiry_date", inquiry_date),
                            ("institution", institution),
                            ("reason", source_reason),
                        )
                        if value in (None, "")
                    ]
                    record_issue(
                        parse_result,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_inquiry_row_cells_unresolved",
                            message="A printed inquiry row could not be assigned completely to the canonical four-column schema.",
                            parser_stage="candidate_b_inquiry_schema",
                            target_dataset="inquiry_records",
                            field_name=missing_fields[0] if len(missing_fields) == 1 else None,
                            observed_value={"sequence": sequence, "row": list(cells)},
                            candidate_value={"missing_fields": missing_fields},
                            source_refs=(_source_ref(page, table, row=row_index),),
                            reason_codes=(
                                "printed_inquiry_sequence_observed",
                                "exact_header_column_binding_failed",
                                "record_not_invented",
                            ),
                        ),
                    )
                    continue
                inquiry_type = (
                    "personal" if institution == "本人" or source_reason.startswith("本人查询") else "institution"
                )
                inquiry_id = stable_record_id("credit_inquiry", inquiry_type, sequence)
                row_ref = _source_ref(page, table, row=row_index)
                refs_by_field = {
                    field_name: [
                        {
                            **_source_ref(page, table, row=row_index, column=slots[field_name]),
                            "field_name": field_name,
                            "binding": "canonical_header_column",
                        }
                    ]
                    for field_name in ("inquiry_date", "institution", "reason")
                }
                records.append(
                    {
                        "inquiry_id": inquiry_id,
                        "sequence": sequence,
                        "inquiry_date": inquiry_date,
                        "institution": institution,
                        "reason": source_reason,
                        "source_reason": source_reason,
                        "query_channel": "personal" if inquiry_type == "personal" else "institution",
                        "inquiry_type": inquiry_type,
                        "source": "native_detail_inquiry_table",
                        "source_refs": [row_ref],
                        "source_refs_by_field": refs_by_field,
                        "confidence": float(getattr(table, "confidence", None) or 0.9),
                    }
                )
                if repair_kind:
                    _record_inquiry_ordinal_repair(
                        parse_result,
                        inquiry_type=inquiry_type,
                        sequence=sequence,
                        raw_sequence=raw_sequence,
                        inquiry_id=inquiry_id,
                        source_ref=row_ref,
                        repair_kind=repair_kind,
                    )
        if not page_is_canonical_inquiry and not page_had_inquiry_table:
            active_canonical_table = False
            active_slots = {}
            active_schema_page = None
    records.extend(_canonical_inquiry_line_rows(parse_result))
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    for raw_record in records:
        record = _repair_inquiry_reason_boundary(raw_record)
        key = (str(record.get("inquiry_type") or ""), int(record.get("sequence") or 0))
        if key[0] and key[1] > 0:
            grouped[key].append(record)

    best: dict[tuple[str, int], dict[str, Any]] = {}
    for key, observations in grouped.items():
        selected = deepcopy(max(observations, key=_inquiry_observation_score))
        selected["inquiry_id"] = stable_record_id("credit_inquiry", key[0], key[1])
        merged_refs: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for observation in observations:
            for ref in observation.get("source_refs") or ():
                if not isinstance(ref, Mapping):
                    continue
                marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen_refs:
                    seen_refs.add(marker)
                    merged_refs.append(dict(ref))
        selected["source_refs"] = merged_refs
        selected_refs_by_field: dict[str, list[dict[str, Any]]] = {}
        for field_name in ("inquiry_date", "institution", "reason"):
            candidates: list[tuple[int, float, Any, Mapping[str, Any]]] = []
            for observation in observations:
                value = observation.get(field_name)
                if value in (None, ""):
                    continue
                refs_by_field = observation.get("source_refs_by_field")
                refs = refs_by_field.get(field_name) if isinstance(refs_by_field, Mapping) else ()
                exact = any(
                    isinstance(ref, Mapping)
                    and ref.get("geometry_scope") == "cell"
                    and ref.get("binding") == "canonical_header_column"
                    for ref in refs or ()
                )
                candidates.append((2 if exact else 1, float(observation.get("confidence") or 0.0), value, observation))
            if not candidates:
                selected[field_name] = None
                continue
            top_quality = max(candidate[0] for candidate in candidates)
            top = [candidate for candidate in candidates if candidate[0] == top_quality]
            normalized: defaultdict[str, list[tuple[int, float, Any, Mapping[str, Any]]]] = defaultdict(list)
            for candidate in top:
                normalized[_normalized_inquiry_field(field_name, candidate[2])].append(candidate)
            if len(normalized) > 1:
                selected[field_name] = None
                selected["extraction_status"] = "review"
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_inquiry_field_conflict",
                        message="Equally source-bound observations disagree for one canonical inquiry field; the value was withheld.",
                        parser_stage="candidate_b_inquiry_schema",
                        target_dataset="inquiry_records",
                        target_record_id=selected["inquiry_id"],
                        field_name=field_name,
                        observed_value=[candidate[2] for candidate in top],
                        source_refs=merged_refs,
                        reason_codes=(
                            "same_printed_inquiry_row",
                            "equal_field_provenance",
                            "conflicting_values_withheld",
                        ),
                    ),
                )
                continue
            chosen = max(top, key=lambda candidate: (candidate[1], len(str(candidate[2]))))
            selected[field_name] = chosen[2]
            for _quality, _confidence, _value, observation in normalized[next(iter(normalized))]:
                refs_by_field = observation.get("source_refs_by_field")
                for ref in refs_by_field.get(field_name) or () if isinstance(refs_by_field, Mapping) else ():
                    if isinstance(ref, Mapping):
                        selected_refs_by_field.setdefault(field_name, []).append(dict(ref))
        if selected_refs_by_field:
            selected["source_refs_by_field"] = selected_refs_by_field
        best[key] = selected
    ordered = sorted(best.values(), key=lambda row: (str(row.get("inquiry_type") or ""), int(row.get("sequence") or 0)))
    for inquiry_type in sorted({str(row.get("inquiry_type") or "unknown") for row in ordered}):
        sequences = {
            int(row.get("sequence") or 0)
            for row in ordered
            if str(row.get("inquiry_type") or "unknown") == inquiry_type and int(row.get("sequence") or 0) > 0
        }
        missing = sorted(set(range(1, max(sequences) + 1)) - sequences) if sequences else []
        if not missing:
            continue
        record_issue(
            parse_result,
            make_issue(
                category="page_continuation",
                issue_code="canonical_inquiry_sequence_gap",
                message="Canonical inquiry rows contain a printed sequence gap; no missing row was invented.",
                parser_stage="candidate_b_inquiry_schema",
                target_dataset="inquiry_records",
                observed_value={
                    "inquiry_type": inquiry_type,
                    "observed_row_count": len(sequences),
                    "observed_sequences": sorted(sequences),
                },
                candidate_value={"source_sequence_endpoint": max(sequences), "missing_sequences": missing},
                reason_codes=("canonical_sequence_not_contiguous", "missing_row_not_invented", "dataset_incomplete"),
            ),
        )
    _enforce_observed_header_terminal_invariant(
        parse_result,
        dataset="inquiry_records",
        aliases=inquiry_aliases,
        records=ordered,
    )
    return ordered


_PUBLIC_CANONICAL_LAYOUTS: tuple[dict[str, Any], ...] = (
    {
        "name": "tax_arrears",
        "record_type": "tax_arrears",
        "signature": ("tax_authority", "arrears_amount"),
        "aliases": {
            "sequence": ("编号",),
            "tax_authority": ("主管税务机关",),
            "arrears_amount": ("欠税总额",),
            "statistics_date": ("欠税统计日期",),
        },
        "fields": {
            "tax_authority": ("tax_authority", "text"),
            "arrears_amount": ("arrears_amount", "number"),
            "statistics_date": ("statistics_date", "date"),
        },
    },
    {
        "name": "civil_judgment_base",
        "record_type": "civil_judgment",
        "signature": ("filing_court", "closure_method"),
        "aliases": {
            "sequence": ("编号",),
            "filing_court": ("立案法院",),
            "cause": ("案由",),
            "filing_date": ("立案日期",),
            "closure_method": ("结案方式",),
        },
        "fields": {
            "filing_court": ("filing_court", "text"),
            "cause": ("cause", "text"),
            "filing_date": ("filing_date", "date"),
            "closure_method": ("closure_method", "text"),
        },
    },
    {
        "name": "civil_judgment_detail",
        "record_type": "civil_judgment",
        "signature": ("judgment_result", "claim_amount"),
        "aliases": {
            "sequence": ("编号",),
            "judgment_result": ("判决/调解结果", "判决／调解结果"),
            "judgment_effective_date": ("判决/调解生效日期", "判决／调解生效日期"),
            "claim_subject": ("诉讼标的",),
            "claim_amount": ("诉讼标的金额",),
        },
        "fields": {
            "judgment_result": ("judgment_result", "text"),
            "judgment_effective_date": ("judgment_effective_date", "date"),
            "claim_subject": ("claim_subject", "text"),
            "claim_amount": ("claim_amount", "number"),
        },
    },
    {
        "name": "enforcement_base",
        "record_type": "enforcement",
        "signature": ("court", "closure_method"),
        "aliases": {
            "sequence": ("编号",),
            "court": ("执行法院",),
            "cause": ("执行案由", "案由"),
            "filing_date": ("立案日期",),
            "closure_method": ("结案方式",),
        },
        "fields": {
            "court": ("court", "text"),
            "cause": ("cause", "text"),
            "filing_date": ("filing_date", "date"),
            "closure_method": ("closure_method", "text"),
        },
    },
    {
        "name": "enforcement_detail",
        "record_type": "enforcement",
        "signature": ("case_status", "requested_subject", "executed_subject"),
        "aliases": {
            "sequence": ("编号",),
            "case_status": ("案件状态",),
            "closure_date": ("结案日期",),
            "requested_subject": ("申请执行标的",),
            "requested_amount": ("申请执行标的金额", "申请执行标的的价值"),
            "executed_subject": ("已执行标的",),
            "executed_amount": ("已执行标的金额",),
        },
        "fields": {
            "case_status": ("case_status", "text"),
            "closure_date": ("closure_date", "date"),
            "requested_subject": ("requested_subject", "text"),
            "requested_amount": ("requested_amount", "number"),
            "executed_subject": ("executed_subject", "text"),
            "executed_amount": ("executed_amount", "number"),
        },
    },
    {
        "name": "administrative_penalty",
        "record_type": "administrative_penalty",
        "signature": ("authority", "penalty_content", "penalty_amount"),
        "aliases": {
            "sequence": ("编号",),
            "authority": ("处罚机构",),
            "penalty_content": ("处罚内容",),
            "penalty_amount": ("处罚金额",),
            "effective_date": ("生效日期",),
            "end_date": ("截止日期",),
            "administrative_review_result": ("行政复议结果",),
        },
        "fields": {
            "authority": ("authority", "text"),
            "penalty_content": ("penalty_content", "text"),
            "penalty_amount": ("penalty_amount", "number"),
            "effective_date": ("effective_date", "date"),
            "end_date": ("end_date", "date"),
            "administrative_review_result": ("administrative_review_result", "text"),
        },
    },
    {
        "name": "housing_fund_base",
        "record_type": "housing_fund",
        "signature": ("contribution_location", "monthly_contribution"),
        "sequence_optional": True,
        "record_boundary": "start",
        "aliases": {
            "contribution_location": ("参缴地",),
            "participation_date": ("参缴日期",),
            "first_contribution_month": ("初缴月份", "初缴日期"),
            "paid_through_month": ("缴至月份",),
            "payment_status": ("缴费状态",),
            "monthly_contribution": ("月缴存额",),
            "personal_contribution_ratio": ("个人缴存比例",),
            "employer_contribution_ratio": ("单位缴存比例",),
        },
        "fields": {
            "contribution_location": ("contribution_location", "text"),
            "participation_date": ("participation_date", "date"),
            "first_contribution_month": ("first_contribution_month", "date"),
            "paid_through_month": ("paid_through_month", "date"),
            "payment_status": ("payment_status", "text"),
            "monthly_contribution": ("monthly_contribution", "number"),
            "personal_contribution_ratio": ("personal_contribution_ratio", "text"),
            "employer_contribution_ratio": ("employer_contribution_ratio", "text"),
        },
    },
    {
        "name": "housing_fund_provider",
        "record_type": "housing_fund",
        "signature": ("employer", "information_updated_month"),
        "sequence_optional": True,
        "record_boundary": "continuation",
        "aliases": {
            "employer": ("缴费单位",),
            "information_updated_month": ("信息更新日期", "信息更新月份"),
        },
        "fields": {
            "employer": ("employer", "employer_name"),
            "information_updated_month": ("information_updated_month", "date"),
        },
    },
    {
        "name": "professional_qualification",
        "record_type": "professional_qualification",
        "signature": ("qualification_name", "issuing_authority"),
        "aliases": {
            "sequence": ("编号",),
            "qualification_name": ("执业资格名称",),
            "level": ("等级",),
            "obtained_date": ("获得日期",),
            "expiry_date": ("到期日期",),
            "revocation_date": ("吊销日期",),
            "issuing_authority": ("颁发机构",),
            "authority_location": ("机构所在地",),
        },
        "fields": {
            "qualification_name": ("qualification_name", "text"),
            "level": ("level", "text"),
            "obtained_date": ("obtained_date", "date"),
            "expiry_date": ("expiry_date", "date"),
            "revocation_date": ("revocation_date", "date"),
            "issuing_authority": ("issuing_authority", "text"),
            "authority_location": ("authority_location", "text"),
        },
    },
    {
        "name": "award",
        "record_type": "award",
        "signature": ("authority", "award_content"),
        "aliases": {
            "sequence": ("编号",),
            "authority": ("奖励机构",),
            "award_content": ("奖励内容",),
            "effective_date": ("生效日期",),
            "end_date": ("截止日期",),
        },
        "fields": {
            "authority": ("authority", "text"),
            "award_content": ("award_content", "text"),
            "effective_date": ("effective_date", "date"),
            "end_date": ("end_date", "date"),
        },
    },
)


_PUBLIC_LAYOUT_LABELS = frozenset(
    str(alias)
    for layout in _PUBLIC_CANONICAL_LAYOUTS
    for aliases in layout["aliases"].values()
    for alias in aliases
    if alias
)
_PUBLIC_TYPED_INSTITUTION_RE = re.compile(
    r"(?:人民法院|法院|税务局|管理局|监管局|管理中心|"
    r"住房公积金管理中心|银行(?:股份)?有限公司|有限公司)"
)
_PUBLIC_TYPED_AMOUNT_RE = re.compile(
    r"(?:[\uffe5¥] *-?\d[\d,]*(?:\.\d{1,2})?|"
    r"-?\d[\d,]*(?:\.\d{1,2})? *(?:元|%))"
)


def _public_text_slot_is_unambiguous(raw: str) -> bool:
    """Reject text cells that visibly contain another canonical slot."""

    compact = _compact(raw)
    if not compact:
        return False
    if any(label in compact for label in _PUBLIC_LAYOUT_LABELS):
        return False
    typed_markers = len(_valid_date_spans(compact))
    typed_markers += len(_PUBLIC_TYPED_INSTITUTION_RE.findall(compact))
    typed_markers += len(_PUBLIC_TYPED_AMOUNT_RE.findall(compact))
    return typed_markers < 2


def _public_value(raw: str, kind: str) -> Any:
    if kind == "number":
        value = _number(raw)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if kind == "date":
        return _date(raw)
    if kind == "employer_name":
        from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
            normalize_role_candidate,
            role_candidate_is_valid,
        )

        value = normalize_role_candidate(raw, "employer_name")
        return value if role_candidate_is_valid(value, "employer_name") else None
    return _clean(raw) if _public_text_slot_is_unambiguous(raw) else None


_PUBLIC_FINAL_DATASETS = {
    "tax_arrears": "tax_arrears_records",
    "civil_judgment": "civil_judgment_records",
    "enforcement": "enforcement_records",
    "administrative_penalty": "administrative_penalty_records",
    "housing_fund": "housing_fund_records",
    "professional_qualification": "professional_qualification_records",
    "award": "administrative_award_records",
}


def _consume_public_canonical_row(
    parse_result: Any,
    working: dict[tuple[str, int], dict[str, Any]],
    *,
    page: Any,
    table: Any,
    layout: Mapping[str, Any],
    slots: Mapping[str, int],
    data_row: tuple[str, ...],
    row_index: int,
    sequence: int,
) -> None:
    record_type = str(layout["record_type"])
    target_dataset = _PUBLIC_FINAL_DATASETS[record_type]
    target_id = stable_record_id("public_record", record_type, sequence)
    item = working.setdefault(
        (record_type, sequence),
        {
            "public_record_id": target_id,
            "sequence": sequence,
            "record_type": record_type,
            "source": "native_detail_public_table",
            "source_refs": [],
            "confidence": 1.0,
        },
    )
    row_ref = _source_ref(page, table, row=row_index)
    if row_ref not in item["source_refs"]:
        item["source_refs"].append(row_ref)
    for role, (field_name, kind) in layout["fields"].items():
        raw = _slot_value(data_row, slots, role)
        ref = _source_ref(page, table, row=row_index, column=slots[role])
        if not raw:
            _report_required_row_failure(
                parse_result,
                issue_code="candidate_b_public_record_cell_unresolved",
                dataset=target_dataset,
                sequence=sequence,
                field_name=field_name,
                row=data_row,
                page=page,
                table=table,
                row_index=row_index,
                target_record_id=target_id,
            )
            _append_internal_field(item, "_unresolved_fields", field_name)
            continue
        if _compact(raw) in {"-", "--", "---"}:
            _mark_source_absent(item, field_name, raw)
            continue
        value = _public_value(raw, kind)
        if value in (None, ""):
            _reject_exact_observation(
                parse_result,
                item,
                dataset=target_dataset,
                target_record_id=target_id,
                field_name=field_name,
                raw=raw,
                source_ref=ref,
                parser_stage="candidate_b_public_canonical_slots",
            )
            continue
        _merge_exact_observation(
            parse_result,
            item,
            dataset=target_dataset,
            target_record_id=target_id,
            field_name=field_name,
            value=value,
            raw=raw,
            source_ref=ref,
            parser_stage="candidate_b_public_canonical_slots",
        )


def _public_record_continuation_allowed(
    parse_result: Any,
    left_logical_page: int,
    right_logical_page: int,
) -> bool:
    """Bind a split public record only across one proven page edge."""

    return _document_page_carry_allowed(
        parse_result,
        left_logical_page,
        right_logical_page,
    )


def _extract_public_records(parse_result: Any) -> list[dict[str, Any]]:
    # All accepted layouts are enumerated from the canonical PBOC report.  A
    # missed physical cell therefore becomes uncertainty, never a left-shifted
    # value guessed from the remaining non-empty cells.
    working: dict[tuple[str, int], dict[str, Any]] = {}
    optional_sequence_counters: defaultdict[str, int] = defaultdict(int)
    pending_optional_sequences: dict[str, dict[str, int]] = {}

    def report_pending(record_type: str) -> None:
        pending = pending_optional_sequences.pop(record_type, None)
        if pending is None:
            return
        sequence = int(pending["sequence"])
        _report_optional_public_continuation_missing(
            parse_result,
            dataset=_PUBLIC_FINAL_DATASETS[record_type],
            target_record_id=stable_record_id("public_record", record_type, sequence),
            sequence=sequence,
            source_refs=working.get((record_type, sequence), {}).get("source_refs", []),
        )

    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = [tuple(row) for row in _table_rows(table)]
            for header_index, header in enumerate(rows):
                matched: dict[str, Any] | None = None
                slots: dict[str, int] = {}
                for layout in _PUBLIC_CANONICAL_LAYOUTS:
                    candidate = _canonical_header_slots(header, layout["aliases"])
                    if all(role in candidate for role in layout["signature"]):
                        matched = layout
                        slots = candidate
                        break
                if matched is None:
                    continue
                matched_record_type = str(matched["record_type"])
                matched_boundary = str(matched.get("record_boundary") or "standalone")
                for pending_record_type in tuple(pending_optional_sequences):
                    if not (
                        matched_record_type == pending_record_type
                        and matched_boundary == "continuation"
                    ):
                        report_pending(pending_record_type)
                expected = set(matched["aliases"])
                if matched.get("sequence_optional"):
                    expected.discard("sequence")
                if not expected.issubset(slots):
                    _report_header_graph_failure(
                        parse_result,
                        dataset=_PUBLIC_FINAL_DATASETS[str(matched["record_type"])],
                        page=page,
                        table=table,
                        row_index=header_index,
                        row=header,
                    )
                    continue
                if header_index + 1 >= len(rows):
                    _report_header_graph_failure(
                        parse_result,
                        dataset=_PUBLIC_FINAL_DATASETS[str(matched["record_type"])],
                        page=page,
                        table=table,
                        row_index=header_index,
                        row=header,
                    )
                    continue
                for data_index in range(header_index + 1, len(rows)):
                    data_row = rows[data_index]
                    if any(
                        all(
                            role in _canonical_header_slots(data_row, candidate_layout["aliases"])
                            for role in candidate_layout["signature"]
                        )
                        for candidate_layout in _PUBLIC_CANONICAL_LAYOUTS
                    ):
                        break
                    if matched.get("sequence_optional") and not any(_clean(cell) for cell in data_row):
                        continue
                    if matched.get("sequence_optional"):
                        record_type = matched_record_type
                        boundary = matched_boundary
                        if boundary == "start":
                            optional_sequence_counters[record_type] += 1
                            sequence = optional_sequence_counters[record_type]
                            logical_page = int(getattr(page, "page_number", 0) or 0)
                            pending_optional_sequences[record_type] = {
                                "sequence": sequence,
                                "logical_page": logical_page,
                            }
                        elif boundary == "continuation":
                            pending = pending_optional_sequences.get(record_type)
                            if pending is None:
                                _report_unowned_optional_public_fragment(
                                    parse_result,
                                    dataset=_PUBLIC_FINAL_DATASETS[record_type],
                                    row=data_row,
                                    page=page,
                                    table=table,
                                    row_index=data_index,
                                )
                                break
                            logical_page = int(getattr(page, "page_number", 0) or 0)
                            start_page = int(pending["logical_page"])
                            adjacent = _public_record_continuation_allowed(
                                parse_result,
                                start_page,
                                logical_page,
                            )
                            if not adjacent:
                                report_pending(record_type)
                                _report_unowned_optional_public_fragment(
                                    parse_result,
                                    dataset=_PUBLIC_FINAL_DATASETS[record_type],
                                    row=data_row,
                                    page=page,
                                    table=table,
                                    row_index=data_index,
                                )
                                break
                            sequence = int(pending["sequence"])
                        else:
                            optional_sequence_counters[record_type] += 1
                            sequence = optional_sequence_counters[record_type]
                    else:
                        sequence = _sequence_value(data_row, slots)
                    if sequence is None:
                        raw_sequence = _slot_value(data_row, slots, "sequence")
                        if raw_sequence or (
                            data_index == header_index + 1
                            and sum(bool(_clean(cell)) for cell in data_row) >= 2
                        ):
                            _report_unkeyed_business_row(
                                parse_result,
                                dataset=_PUBLIC_FINAL_DATASETS[str(matched["record_type"])],
                                row=data_row,
                                page=page,
                                table=table,
                                row_index=data_index,
                            )
                        continue
                    _consume_public_canonical_row(
                        parse_result,
                        working,
                        page=page,
                        table=table,
                        layout=matched,
                        slots=slots,
                        data_row=data_row,
                        row_index=data_index,
                        sequence=sequence,
                    )
                    if (
                        matched.get("sequence_optional")
                        and matched.get("record_boundary") == "continuation"
                    ):
                        pending_optional_sequences.pop(str(matched["record_type"]), None)
                    if matched.get("sequence_optional"):
                        break

    for record_type in tuple(pending_optional_sequences):
        report_pending(record_type)

    records: list[dict[str, Any]] = []
    authority_fields = {
        "tax_arrears": "tax_authority",
        "civil_judgment": "filing_court",
        "enforcement": "court",
        "administrative_penalty": "authority",
        "housing_fund": "employer",
        "professional_qualification": "issuing_authority",
        "award": "authority",
    }
    internal = {
        "public_record_id",
        "sequence",
        "record_type",
        "source",
        "source_refs",
        "source_refs_by_field",
        "confidence",
    }
    for (record_type, _sequence), item in sorted(working.items(), key=lambda pair: pair[0]):
        content = {
            key: value
            for key, value in item.items()
            if key not in internal and not key.startswith("_") and key != "canonical_raw"
        }
        for field_name in item.get("_source_absent_fields", []):
            content.setdefault(field_name, None)
        if record_type in {"tax_arrears", "civil_judgment", "enforcement", "administrative_penalty", "housing_fund"}:
            content["reporting_amount_currency"] = "CNY"
            content["reporting_amount_unit"] = "yuan"
        if "cause" in item:
            content["cause_status"] = "reported"
        elif "cause" in item.get("_source_absent_fields", []):
            content["cause_status"] = "not_reported"
        if "administrative_review_result" in item:
            content["administrative_review_result_status"] = "reported"
        elif "administrative_review_result" in item.get("_source_absent_fields", []):
            content["administrative_review_result_status"] = "not_reported"
        record = {
            **item,
            "authority": item.get(authority_fields[record_type]),
            "start_date": item.get("filing_date") or item.get("effective_date"),
            "end_date": item.get("closure_date") or item.get("end_date"),
            "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
        }
        records.append(record)
    return records


# Employment cells use finite PBOC code-list descriptions.  These are used
# only to disambiguate OCR cells whose printed column boundaries collapsed;
# arbitrary text is never coerced into one of these values.
_EMPLOYER_TYPES = (
    "机关事业单位",
    "国有企业",
    "集体企业",
    "外资企业",
    "私营企业",
    "民营企业",
    "个体工商户",
    "个体、私营企业",
    "其他（包括三资企业、民营企业、民间团体等）",
    "未知",
)
_EMPLOYMENT_INDUSTRIES = (
    "农、林、牧、渔业",
    "采矿业",
    "制造业",
    "电力、燃气及水的生产和供应业",
    "建筑业",
    "交通运输、仓储和邮政业",
    "信息传输、计算机服务和软件业",
    "批发和零售业",
    "住宿和餐饮业",
    "金融业",
    "房地产业",
    "租赁和商务服务业",
    "科学研究、技术服务和地质勘查业",
    "水利、环境和公共设施管理业",
    "居民服务和其他服务业",
    "教育",
    "卫生、社会保障和社会福利业",
    "文化、体育和娱乐业",
    "公共管理和社会组织",
    "国际组织",
    "未知",
)
_EMPLOYMENT_OCCUPATIONS = (
    "国家机关、党群组织、企业、事业单位负责人",
    "专业技术人员",
    "办事人员和有关人员",
    "商业、服务业人员",
    "农、林、牧、渔、水利业生产人员",
    "生产、运输设备操作人员及有关人员",
    "军人",
    "不便分类的其他从业人员",
    "未知",
)
_EMPLOYMENT_POSITIONS = ("高级领导", "中级领导", "一般员工", "其他", "未知")
_EMPLOYMENT_TITLES = ("高级", "中级", "初级", "无", "未知")
_EMPLOYMENT_DETAIL_VOCABULARIES = {
    "industry": _EMPLOYMENT_INDUSTRIES,
    "occupation": _EMPLOYMENT_OCCUPATIONS,
    "position": _EMPLOYMENT_POSITIONS,
    "professional_title": _EMPLOYMENT_TITLES,
}


def _canonical_header_slots(
    row: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]]
) -> dict[str, int]:
    """Map semantic roles to distinct physical columns in a printed header."""
    candidates: dict[str, int] = {}
    by_column: defaultdict[int, list[str]] = defaultdict(list)
    for role, labels in aliases.items():
        for column, cell in enumerate(row):
            text = _compact(cell)
            if text and any(_compact(label) in text for label in labels):
                candidates[role] = column
                by_column[column].append(role)
                break
    # A single OCR blob containing several labels is not a column graph.
    duplicate_columns = {column for column, roles in by_column.items() if len(roles) > 1}
    return {role: column for role, column in candidates.items() if column not in duplicate_columns}


def _header_role_columns(
    row: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]]
) -> dict[str, int]:
    """Return observed role columns without pretending merged columns are distinct."""

    columns: dict[str, int] = {}
    for role, labels in aliases.items():
        for column, cell in enumerate(row):
            text = _compact(cell)
            if text and any(_compact(label) in text for label in labels):
                columns[role] = column
                break
    return columns


def _employment_signature(value: Any) -> str:
    """Ignore layout punctuation when matching one finite employment value."""

    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]", "", str(value or ""))


def _finite_employment_value(value: Any, vocabulary: Iterable[str]) -> str | None:
    """Return a canonical finite value only for a unique exact glyph signature."""

    marker = _employment_signature(value)
    if not marker:
        return None
    matches = [candidate for candidate in vocabulary if _employment_signature(candidate) == marker]
    return matches[0] if len(matches) == 1 else None


def _employment_address_has_cross_field_contamination(record: Mapping[str, Any]) -> bool:
    """Detect exact business-field spans copied into an employer address.

    The address is withheld as a whole.  Removing the matching span would
    invent a cell boundary which the OCR evidence did not preserve.
    """

    address = str(record.get("employer_address") or "")
    address_marker = _employment_signature(address)
    if not address_marker:
        return False
    employer_marker = _employment_signature(record.get("employer"))
    department_suffixes = (
        "销售部",
        "营业部",
        "门市部",
        "办公室",
        "办事处",
        "项目部",
    )
    # A printed workplace address may legitimately end with the full employer
    # name and one exact department suffix.  Keep that source value only when
    # an address-shaped prefix precedes the employer; a bare organization name
    # is not promoted into an address by this exception.
    employer_is_address_suffix = bool(
        employer_marker
        and any(
            address_marker.endswith(employer_marker + suffix)
            and len(address_marker[: -len(employer_marker + suffix)]) >= 3
            and re.search(
                r"(?:省|市|区|县|镇|乡|街|路|道|巷|弄|号|楼|村|社区)",
                address_marker[: -len(employer_marker + suffix)],
            )
            for suffix in department_suffixes
        )
    )
    if (
        len(employer_marker) >= 3
        and employer_marker in address_marker
        and not employer_is_address_suffix
    ):
        return True

    # A merged/truncated cell can end part-way through the same record's legal
    # employer name.  Require a long prefix owned by this exact record; short
    # generic fragments such as a district or ``公司`` are never sufficient.
    minimum_prefix = max(6, (len(employer_marker) + 1) // 2)
    if len(employer_marker) >= 8 and len(address_marker) > minimum_prefix:
        if any(
            address_marker.endswith(employer_marker[:prefix_length])
            for prefix_length in range(len(employer_marker) - 1, minimum_prefix - 1, -1)
        ):
            return True

    # Position and title values are short, so require an exact suffix token;
    # an interior occurrence such as ``中级人民法院`` may be legitimate address
    # text.  Include both extracted values and the finite printed vocabulary
    # so contamination is still recognized when the corresponding detail cell
    # was itself unreadable.
    role_tokens = {
        str(record.get("position") or ""),
        str(record.get("professional_title") or ""),
        *_EMPLOYMENT_POSITIONS,
        *_EMPLOYMENT_TITLES,
    }
    return any(
        token_marker
        and address_marker != token_marker
        and address_marker.endswith(token_marker)
        for token in role_tokens
        if (token_marker := _employment_signature(token))
    )


def _enforce_employment_record_contracts(
    parse_result: Any,
    records: Iterable[dict[str, Any]],
) -> None:
    """Apply cross-field contracts after any employment record merge plane."""

    for record in records:
        if not _employment_address_has_cross_field_contamination(record):
            continue
        address = str(record.get("employer_address") or "")
        refs = record.get("source_refs_by_field", {}).get("employer_address") or ()
        source_ref = next(
            (dict(ref) for ref in refs if isinstance(ref, Mapping)),
            dict((record.get("source_refs") or [{}])[0]),
        )
        target_record_id = str(
            record.get("employment_record_id")
            or record.get("record_id")
            or "employment_record:unresolved"
        )
        _reject_exact_observation(
            parse_result,
            record,
            dataset="employment_records",
            target_record_id=target_record_id,
            field_name="employer_address",
            raw=address,
            source_ref=source_ref,
            parser_stage="candidate_b_employment_record_contract",
        )


def _finite_values_in_cluster(
    value: Any,
    vocabularies: Mapping[str, Iterable[str]],
) -> dict[str, str]:
    """Find uniquely owned finite values inside one OCR-merged business cell.

    A glyph sequence shared by multiple logical roles (for example ``未知``)
    is intentionally left unresolved.  For a role with nested candidates, the
    longest observed candidate wins only when it is unique.
    """

    marker = _employment_signature(value)
    if not marker:
        return {}
    candidates: dict[str, list[tuple[str, str]]] = {}
    for role, vocabulary in vocabularies.items():
        observed = [
            (candidate, _employment_signature(candidate))
            for candidate in vocabulary
            if _employment_signature(candidate) in marker
        ]
        if not observed:
            continue
        longest = max(len(signature) for _candidate, signature in observed)
        maximal = [item for item in observed if len(item[1]) == longest]
        if len(maximal) == 1:
            candidates[role] = maximal
    signatures_to_roles: defaultdict[str, list[str]] = defaultdict(list)
    for role, ((_, signature),) in candidates.items():
        signatures_to_roles[signature].append(role)
    return {
        role: candidate
        for role, ((candidate, signature),) in candidates.items()
        if len(signatures_to_roles[signature]) == 1
        and not any(
            signature != other_signature
            and signature in other_signature
            and marker.count(signature) <= marker.count(other_signature)
            for other_role, ((_, other_signature),) in candidates.items()
            if other_role != role
        )
    }


def _employment_cluster_residue(value: Any, owned_values: Iterable[Any]) -> str:
    """Return glyphs in one cluster not owned by selected exact values."""

    residue = _employment_signature(value)
    for owned_value in owned_values:
        signature = _employment_signature(owned_value)
        if signature and signature in residue:
            residue = residue.replace(signature, "", 1)
    return residue


def _recovered_employment_basic_header(
    row: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]]
) -> tuple[dict[str, int], dict[str, str]] | None:
    """Recover only the canonical five-column header with a damaged ordinal label."""

    columns = _header_role_columns(row, aliases)
    business_roles = ("employer", "employer_type", "employer_address", "employer_phone")
    if not all(role in columns for role in business_roles):
        return None
    if [columns[role] for role in business_roles] != [1, 2, 3, 4]:
        return None
    sequence_header = _compact(row[0] if row else "")
    if sequence_header and "编" not in sequence_header and "号" not in sequence_header:
        return None
    columns["sequence"] = 0
    overflow: dict[str, str] = {}
    for role in business_roles:
        cell = _clean(row[columns[role]])
        residue = cell
        for label in aliases[role]:
            residue = residue.replace(label, " ")
        residue = _clean(residue)
        if residue:
            overflow[role] = residue
    return columns, overflow


def _collapsed_employment_basic_header(row: tuple[str, ...]) -> tuple[int, int] | None:
    """Recognize the closed five-role header after OCR collapsed its columns.

    The four business labels may appear in the OCR traversal order rather than
    their visual left-to-right order.  Recognition therefore depends on the
    exact finite label set and rejects every extra glyph; values are assigned
    later by their field contracts, never by this traversal order.
    """

    labels = ("工作单位", "单位性质", "单位地址", "单位电话")
    populated = [(index, _compact(value)) for index, value in enumerate(row) if _compact(value)]
    if not populated:
        return None
    sequence_columns = [index for index, value in populated if value == "编号"]
    cluster_candidates: list[tuple[int, str]] = []
    for index, value in populated:
        residue = value
        if residue.startswith("编号"):
            residue = residue.removeprefix("编号")
        if all(residue.count(label) == 1 for label in labels):
            for label in labels:
                residue = residue.replace(label, "", 1)
            if not residue:
                cluster_candidates.append((index, value))
    if len(cluster_candidates) != 1:
        return None
    cluster_column, cluster_value = cluster_candidates[0]
    if cluster_value.startswith("编号"):
        if len(populated) != 1:
            return None
        return cluster_column, cluster_column
    if len(sequence_columns) != 1 or len(populated) != 2:
        return None
    return sequence_columns[0], cluster_column


def _clustered_employment_detail_header(
    row: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]]
) -> dict[str, int] | None:
    """Recognize the canonical detail layout after OCR merged its role columns."""

    columns = _header_role_columns(row, aliases)
    if not {"sequence", "industry", "occupation", "entry_year", "professional_title", "information_updated_date"} <= set(columns):
        return None
    if columns["sequence"] != 0:
        return None
    main_column = columns["industry"]
    secondary_column = columns["entry_year"]
    updated_column = columns["information_updated_date"]
    if columns["occupation"] != main_column or columns["professional_title"] != secondary_column:
        return None
    if not (0 < main_column < secondary_column < updated_column):
        return None
    # The position label is frequently the only damaged label in this exact
    # three-role cluster.  Infer its slot, never its value.
    position_column = columns.get("position")
    if position_column not in (None, secondary_column):
        return None
    return {
        "sequence": 0,
        "industry": main_column,
        "occupation": main_column,
        "entry_year": secondary_column,
        "professional_title": secondary_column,
        "position": secondary_column,
        "information_updated_date": updated_column,
    }


def _slot_value(row: tuple[str, ...], slots: Mapping[str, int], role: str) -> str:
    column = slots.get(role)
    return _clean(row[column]) if column is not None and column < len(row) else ""


def _sequence_value(row: tuple[str, ...], slots: Mapping[str, int]) -> int | None:
    value = _slot_value(row, slots, "sequence")
    match = re.fullmatch(r"\D*(\d{1,3})\D*", value)
    return int(match.group(1)) if match else None


def _employment_sequence_value(
    row: tuple[str, ...], slots: Mapping[str, int]
) -> int | None:
    """Accept an exact ordinal or a watermark-duplicated identical ordinal."""

    sequence = _sequence_value(row, slots)
    if sequence is not None:
        return sequence
    value = _slot_value(row, slots, "sequence")
    tokens = re.findall(r"\d{1,3}", value)
    if len(tokens) >= 2 and len(set(tokens)) == 1:
        return int(tokens[0])
    return None


def _canonical_employer_phone(value: Any) -> str | None:
    """Return digits from one exact phone slot with a bounded layout shape.

    The source may insert whitespace, a hyphen, or paired parentheses around
    an area code.  Letters, Han text, arbitrary punctuation, and layouts that
    look like multiple adjacent business values are rejected rather than
    concatenated.
    """

    raw = str(value or "").strip()
    if not raw or re.search(r"[A-Za-z\u3400-\u9fff]", raw):
        return None
    if not re.fullmatch(r"[0-9\s()（）\-－]+", raw):
        return None
    digits = re.sub(r"\D", "", raw)
    if not 5 <= len(digits) <= 16:
        return None
    runs = re.findall(r"\d+", raw)
    lengths = tuple(len(run) for run in runs)
    if len(runs) == 1 or all(length == 1 for length in lengths):
        return digits
    canonical_layouts = {
        (3, 7),
        (3, 8),
        (4, 7),
        (4, 8),
        (3, 4, 4),
        (4, 4, 4),
    }
    return digits if lengths in canonical_layouts or all(length <= 4 for length in lengths) else None


def _employment_run_end(rows: list[list[str]], start: int) -> int:
    """Return the exclusive end of one canonical employment data run."""

    header_labels = (
        "编号",
        "工作单位",
        "单位性质",
        "单位地址",
        "单位电话",
        "职业",
        "行业",
        "职务",
        "职称",
        "进入本单位年份",
        "信息更新日期",
        "数据发生机构名称",
    )
    for row_index in range(start, len(rows)):
        compact = _compact("".join(rows[row_index]))
        if sum(label in compact for label in header_labels) >= 2:
            return row_index
    return len(rows)


def _employment_row_sequence(
    row: tuple[str, ...],
    *,
    sequence_column: int,
    cluster_column: int | None = None,
) -> int | None:
    """Read a sequence from one row without using any business value."""

    sequence = _employment_sequence_value(row, {"sequence": sequence_column})
    if sequence is not None or cluster_column != sequence_column:
        return sequence
    cluster_raw = _clean(row[cluster_column] if cluster_column < len(row) else "")
    match = re.fullmatch(r"\s*(\d{1,3})\s+.+", cluster_raw)
    return int(match.group(1)) if match is not None else None


def _employment_sequence_repairs(
    rows: list[list[str]],
    *,
    start: int,
    sequence_column: int,
    mode: str,
    known_basic_population: Iterable[int] = (),
    cluster_column: int | None = None,
) -> dict[int, int]:
    """Recover unreadable ordinals only from a closed canonical row run.

    Basic-table repairs must sit between globally consistent printed anchors.
    Detail-table repairs additionally require a unique alignment to the dense
    basic-table population.  Printed dash sentinels are never replaced.
    """

    end = _employment_run_end(rows, start)
    run_indices = [index for index in range(start, end) if _nonempty(rows[index])]
    if not run_indices or mode not in {"basic", "detail"}:
        return {}
    observed = [
        _employment_row_sequence(
            tuple(rows[index]),
            sequence_column=sequence_column,
            cluster_column=cluster_column,
        )
        for index in run_indices
    ]
    inferred: dict[int, int] = {}
    if mode == "basic":
        anchors = [
            (position, sequence)
            for position, sequence in enumerate(observed)
            if sequence is not None
        ]
        if len(anchors) < 2:
            return {}
        offsets = {sequence - position for position, sequence in anchors}
        if len(offsets) != 1:
            return {}
        offset = offsets.pop()
        first_anchor = anchors[0][0]
        last_anchor = anchors[-1][0]
        for position in range(first_anchor + 1, last_anchor):
            if observed[position] is not None:
                continue
            row_index = run_indices[position]
            raw_sequence = (
                rows[row_index][sequence_column]
                if sequence_column < len(rows[row_index])
                else ""
            )
            if is_explicit_source_absence(raw_sequence):
                continue
            expected = position + offset
            if expected > 0:
                inferred[row_index] = expected
        return inferred

    population = sorted(set(int(value) for value in known_basic_population))
    if not population or population != list(range(1, population[-1] + 1)):
        return {}
    if len(run_indices) > len(population):
        return {}
    alignments: list[list[int]] = []
    for population_start in range(len(population) - len(run_indices) + 1):
        candidate = population[population_start : population_start + len(run_indices)]
        if all(
            sequence is None or sequence == candidate[position]
            for position, sequence in enumerate(observed)
        ):
            alignments.append(candidate)
    if len(alignments) != 1:
        return {}
    for position, expected in enumerate(alignments[0]):
        if observed[position] is not None:
            continue
        row_index = run_indices[position]
        raw_sequence = (
            rows[row_index][sequence_column]
            if sequence_column < len(rows[row_index])
            else ""
        )
        if not is_explicit_source_absence(raw_sequence):
            inferred[row_index] = expected
    return inferred


_EMPLOYMENT_PROVIDER_ANCHOR_RE = re.compile(
    r"(?:银行|信用社|信用合作联社|消费金融|汽车金融|财务|"
    r"信托|小额贷款|融资担保|征信|公积金管理)"
)
_EMPLOYMENT_PROVIDER_END_RE = re.compile(
    r"(?:有限公司|中心|分行|支行|营业部|银行|信托|信用社|联社)$"
)


def _strict_employment_provider_span(value: Any) -> str | None:
    """Return one complete institution span, excluding bounded OCR debris.

    Employment-provider rows are ordinal-keyed, but the provider may land in
    any physical column.  Isolated watermark glyphs are separated by OCR
    whitespace in this layout, so enumerate token spans and retain a unique
    institution-shaped span.  A one-glyph edge token is excluded only when the
    remaining span is independently complete; this preserves names split as
    ``有限公 司`` while dropping debris such as ``福 <name> 水``.
    """

    tokens = [token for token in re.split(r"\s+", _clean(value)) if token]
    if not tokens:
        return None

    def valid(candidate_tokens: list[str]) -> str | None:
        candidate = _compact(" ".join(candidate_tokens))
        if not 5 <= len(candidate) <= 96:
            return None
        if any(
            label in candidate
            for label in ("数据发生机构名称", "编号", "信息更新日期")
        ):
            return None
        if re.search(r"(?:19|20)\d{2}", candidate):
            return None
        if _EMPLOYMENT_PROVIDER_ANCHOR_RE.search(candidate) is None:
            return None
        if _EMPLOYMENT_PROVIDER_END_RE.search(candidate) is None:
            return None
        if len(re.findall(r"[\u3400-\u9fff]", candidate)) < 4:
            return None
        return candidate

    candidates: list[str] = []
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            candidate_tokens = tokens[start:end]
            candidate = valid(candidate_tokens)
            if candidate is None:
                continue
            if (
                len(_employment_signature(candidate_tokens[0])) == 1
                and len(candidate_tokens) > 1
                and valid(candidate_tokens[1:]) is not None
            ):
                continue
            if (
                len(_employment_signature(candidate_tokens[-1])) == 1
                and len(candidate_tokens) > 1
                and valid(candidate_tokens[:-1]) is not None
            ):
                continue
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _bounded_provider_header_cell(value: Any) -> bool:
    """Accept the canonical provider label with at most two stray glyphs."""

    signature = _employment_signature(value)
    label = "数据发生机构名称"
    if signature.count(label) != 1:
        return False
    residue = signature.replace(label, "", 1)
    return len(residue) <= 2 and not any(character.isdigit() for character in residue)


def _employment_provider_header(row: tuple[str, ...]) -> dict[str, int] | None:
    """Recognize the two-label provider header with bounded label debris."""

    populated = [
        (column, value)
        for column, value in enumerate(row)
        if _employment_signature(value)
    ]
    if len(populated) != 2:
        return None
    sequence_columns = [
        column
        for column, value in populated
        if _employment_signature(value) == "编号"
    ]
    provider_columns = [
        column for column, value in populated if _bounded_provider_header_cell(value)
    ]
    if len(sequence_columns) != 1 or len(provider_columns) != 1:
        return None
    sequence_column = sequence_columns[0]
    provider_column = provider_columns[0]
    if sequence_column == provider_column:
        return None
    return {"sequence": sequence_column, "data_provider": provider_column}


def _employment_provider_observation(
    row: tuple[str, ...], *, sequence_column: int
) -> tuple[str | None, int | None, str]:
    """Select the one institution-shaped non-sequence cell in a provider row."""

    candidates: list[tuple[str, int]] = []
    explicit_absence: list[tuple[str, int]] = []
    for column, raw in enumerate(row):
        if column == sequence_column:
            continue
        value = _clean(raw)
        if not value:
            continue
        if is_explicit_source_absence(value):
            explicit_absence.append((value, column))
            continue
        provider = _strict_employment_provider_span(value)
        if provider is not None:
            candidates.append((provider, column))
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0][0], unique[0][1], "exact"
    if not unique and len(explicit_absence) == 1:
        return explicit_absence[0][0], explicit_absence[0][1], "source_absent"
    return None, None, "ambiguous" if unique else "missing"


def _continuation_rows_are_provider_shaped(
    rows: list[list[str]],
    *,
    known_sequences: set[int],
) -> tuple[bool, int | None]:
    """Recognize a headerless provider tail without consuming business rows.

    A provider tail has exactly two populated columns, institution-shaped
    values, and either repeats already observed sequence numbers or contains
    at least two consistently unreadable sequence glyphs.  A new numbered
    residence/employment row therefore stays in its current canonical mode.
    """

    if not rows or not known_sequences:
        return False, None
    nonempty_columns = [
        [index for index, value in enumerate(row) if _clean(value)] for row in rows
    ]
    if not all(len(columns) == 2 and columns[0] == 0 for columns in nonempty_columns):
        return False, None
    provider_column = nonempty_columns[0][1]
    if any(columns[1] != provider_column for columns in nonempty_columns):
        return False, None
    provider_pattern = re.compile(
        r"(?:银行(?:股份)?有限公司|银行|信用社|消费金融(?:股份)?有限公司|财务有限公司|管理中心|征信中心)$"
    )
    if not all(provider_pattern.search(_clean(row[provider_column])) for row in rows):
        return False, None
    parsed_sequences: list[int] = []
    unreadable_sequences = 0
    for row in rows:
        first = _clean(row[0] if row else "")
        if first.isdigit():
            parsed_sequences.append(int(first))
        else:
            unreadable_sequences += 1
    if parsed_sequences:
        return (
            unreadable_sequences == 0 and set(parsed_sequences) <= known_sequences,
            provider_column,
        )
    return len(rows) >= 2 and unreadable_sequences == len(rows), provider_column


def _report_required_row_failure(
    parse_result: Any,
    *,
    issue_code: str,
    dataset: str,
    sequence: int,
    field_name: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
    target_record_id: str | None = None,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code=issue_code,
            message="A canonical table row was observed but a required business cell could not be assigned to its printed column.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            target_record_id=target_record_id or f"{dataset}:{sequence}",
            field_name=field_name,
            observed_value={"sequence": sequence, "physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("canonical_column_graph", "required_cell_unresolved", "normalized_value_withheld"),
        ),
    )


def _report_header_graph_failure(
    parse_result: Any,
    *,
    dataset: str,
    page: Any,
    table: Any,
    row_index: int,
    row: tuple[str, ...],
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_canonical_header_graph_unresolved",
            message="Canonical header labels were observed, but OCR did not preserve distinct physical columns.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            observed_value={"physical_header_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("merged_header_cells", "column_roles_unresolved", "page_ocr_eligible"),
        ),
    )


def _report_unkeyed_fragment(
    parse_result: Any,
    *,
    dataset: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="page_continuation",
            issue_code="candidate_b_continuation_sequence_unresolved",
            message="A continued provider fragment lacked a trustworthy printed sequence and was not attached by row order.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            field_name="sequence",
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("continuation_fragment", "printed_sequence_unreadable", "row_order_not_used"),
        ),
    )


_TERMINAL_HEADER_LABEL_GROUPS = (
    ("编号", "手机号码", "信息更新日期", "数据发生机构名称"),
    ("姓名", "证件类型", "证件号码", "工作单位", "联系电话"),
    ("编号", "居住地址", "住宅电话", "居住状况", "信息更新日期"),
    ("编号", "工作单位", "单位地址", "职业", "信息更新日期"),
    ("编号", "查询日期", "查询机构", "查询原因"),
)


def _enforce_observed_header_terminal_invariant(
    parse_result: Any,
    *,
    dataset: str,
    aliases: Mapping[str, tuple[str, ...]],
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Ensure every exact canonical header terminates in data, absence, or an issue."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    consumed_positions = {
        (int(ref.get("logical_page") or 0), str(ref.get("table_id") or ""), int(ref.get("row")))
        for record in records
        if isinstance(record, Mapping)
        for ref in record.get("source_refs") or ()
        if isinstance(ref, Mapping) and ref.get("row") is not None
    }
    issue_positions = {
        (int(ref.get("logical_page") or 0), str(ref.get("table_id") or ""), int(ref.get("row")))
        for issue in getattr(parse_result, "_personal_detail_extraction_issues", ())
        if isinstance(issue, Mapping) and issue.get("target_dataset") == dataset
        for ref in issue.get("source_refs") or ()
        if isinstance(ref, Mapping) and ref.get("row") is not None
    }
    active: dict[str, Any] | None = None
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)

    def finish() -> None:
        nonlocal active
        if active is None or active.get("terminal"):
            active = None
            return
        record_issue(
            parse_result,
            make_issue(
                category="schema_incompleteness",
                issue_code="candidate_b_observed_header_without_terminal_row",
                message=(
                    "A canonical section header was observed, but it produced neither a consumed business row, "
                    "an explicit source-absence row, nor a localized extraction issue."
                ),
                parser_stage="candidate_b_terminal_header_invariant",
                target_dataset=dataset,
                observed_value={"physical_header_cells": active["row"]},
                source_refs=(active["source_ref"],),
                reason_codes=(
                    "canonical_header_observed",
                    "no_consumed_business_row",
                    "no_explicit_source_absence",
                    "silent_drop_prevented",
                ),
            ),
        )
        active = None

    for page in getattr(parse_result, "pages", None) or ():
        page_number = int(getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", None) or ():
            table_id = str(getattr(table, "table_id", "") or "")
            continues = bool(
                active is not None
                and previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            if active is not None and not continues:
                finish()
            for row_index, row in enumerate(_table_rows(table)):
                slots = _canonical_header_slots(row, aliases)
                if set(aliases) <= set(slots):
                    finish()
                    active = {
                        "row": list(row),
                        "slots": slots,
                        "source_ref": _source_ref(page, table, row=row_index),
                        "terminal": False,
                    }
                    continue
                if active is None:
                    continue
                compact = _compact("".join(row))
                if any(sum(label in compact for label in labels) >= 2 for labels in _TERMINAL_HEADER_LABEL_GROUPS):
                    finish()
                    continue
                position = (page_number, table_id, row_index)
                if position in consumed_positions or position in issue_positions:
                    active["terminal"] = True
                    continue
                data_values = [
                    _slot_value(row, active["slots"], role)
                    for role in aliases
                    if role != "sequence"
                ]
                nonempty = [_compact(value) for value in data_values if _compact(value)]
                if nonempty and all(re.fullmatch(r"[-－‐‑‒–—―]+", value) for value in nonempty):
                    active["terminal"] = True
            previous_table_id = table_id or previous_table_id
    finish()


def _report_employment_cluster_field_unresolved(
    parse_result: Any,
    *,
    record: dict[str, Any],
    field_name: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    """Expose a logical employment field lost inside a merged OCR cell."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    _append_internal_field(record, "_unresolved_fields", field_name)
    record["extraction_status"] = "review"
    reported = record.setdefault("_reported_cluster_fields", [])
    if field_name in reported:
        return
    reported.append(field_name)
    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_employment_cluster_field_unresolved",
            message="A canonical employment field remained unassignable after finite-value decoding of an OCR-merged cell.",
            parser_stage="candidate_b_employment_cluster_decoder",
            target_dataset="employment_records",
            target_record_id=str(record["employment_record_id"]),
            field_name=field_name,
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=(
                "canonical_cluster_topology",
                "logical_field_not_uniquely_owned",
                "normalized_value_withheld",
            ),
        ),
    )


def _report_employment_provider_unresolved(
    parse_result: Any,
    *,
    record: dict[str, Any],
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
    resolution: str,
) -> None:
    """Expose a missing or ambiguous provider cell in an exact provider row."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    _append_internal_field(record, "_unresolved_fields", "data_provider")
    record["extraction_status"] = "review"
    if "data_provider" in record.setdefault("_reported_provider_fields", []):
        return
    record["_reported_provider_fields"].append("data_provider")
    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_employment_provider_cell_unresolved",
            message=(
                "An exact employment-provider row did not contain one uniquely "
                "institution-shaped non-sequence cell."
            ),
            parser_stage="candidate_b_employment_provider_row",
            target_dataset="employment_records",
            target_record_id=str(record["employment_record_id"]),
            field_name="data_provider",
            observed_value={
                "sequence": record.get("sequence"),
                "physical_cells": list(row),
                "resolution": resolution,
            },
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=(
                "exact_provider_header",
                f"provider_cell_{resolution}",
                "normalized_value_withheld",
            ),
        ),
    )


def _report_employment_cluster_residue(
    parse_result: Any,
    *,
    record: dict[str, Any],
    field_names: Iterable[str],
    raw: str,
    residue: str,
    page: Any,
    table: Any,
    row_index: int,
    column: int,
) -> None:
    """Report residue while retaining uniquely owned values and full raw evidence."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    record["extraction_status"] = "review"
    source_ref = _source_ref(page, table, row=row_index, column=column)
    for field_name in dict.fromkeys(field_names):
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_employment_cluster_residue_unresolved",
                message=(
                    "An OCR-merged employment cell contained residue beyond its uniquely "
                    "owned canonical values; the values were retained with scoped uncertainty."
                ),
                parser_stage="candidate_b_employment_cluster_decoder",
                target_dataset="employment_records",
                target_record_id=str(record["employment_record_id"]),
                field_name=field_name,
                observed_value={"raw_cluster": raw, "unconsumed_residue": residue},
                source_refs=(source_ref,),
                reason_codes=(
                    "canonical_cluster_topology",
                    "unconsumed_cluster_residue",
                    "candidate_value_retained_with_uncertainty",
                ),
            ),
        )


_STALE_EMPLOYMENT_FIELD_ISSUE_CODES = frozenset(
    {
        "candidate_b_employment_cluster_field_unresolved",
        "candidate_b_employment_canonical_cell_unresolved",
        "candidate_b_employment_recovered_header_cell_unresolved",
        "candidate_b_employment_required_cell_unresolved",
        "candidate_b_employment_provider_cell_unresolved",
    }
)


def _employment_field_is_independently_bound(
    record: Mapping[str, Any], field_name: str
) -> bool:
    """Require a final value and a row/cell-bound source observation."""

    if record.get(field_name) in (None, ""):
        return False
    refs = record.get("source_refs_by_field", {}).get(field_name) or ()
    for ref in refs:
        if not isinstance(ref, Mapping) or not isinstance(ref.get("row"), int):
            continue
        if isinstance(ref.get("column"), int):
            return True
        binding = str(ref.get("binding_quality") or ref.get("binding") or "")
        if binding.startswith("closed_canonical_employment_"):
            return True
    return False


def _prune_resolved_employment_field_issues(
    parse_result: Any, records: Iterable[dict[str, Any]]
) -> None:
    """Remove only stale missing-field diagnostics after exact reconciliation.

    Residue, invalid-value, and conflict issues describe real source evidence
    and deliberately survive.  Only a missing-field diagnostic is stale when
    that same final record/field now owns an independent canonical source
    binding.
    """

    records_by_id = {
        str(record.get("employment_record_id") or record.get("record_id") or ""): record
        for record in records
    }
    resolved_pairs = {
        (record_id, field_name)
        for record_id, record in records_by_id.items()
        if record_id
        for field_name in record
        if _employment_field_is_independently_bound(record, field_name)
    }
    if not resolved_pairs:
        return

    issues = getattr(parse_result, "_personal_detail_extraction_issues", None)
    if not isinstance(issues, list):
        return
    retained_issues = [
        issue
        for issue in issues
        if not (
            isinstance(issue, Mapping)
            and issue.get("issue_code") in _STALE_EMPLOYMENT_FIELD_ISSUE_CODES
            and (
                str(issue.get("target_record_id") or ""),
                str(issue.get("field_name") or ""),
            )
            in resolved_pairs
        )
    ]
    setattr(parse_result, "_personal_detail_extraction_issues", retained_issues)

    remaining_issue_targets = {
        str(issue.get("target_record_id") or "")
        for issue in retained_issues
        if isinstance(issue, Mapping)
        and issue.get("status") not in {
            "resolved",
            "suppressed_redundant",
            "informational",
        }
    }
    for record_id, record in records_by_id.items():
        unresolved = [
            field_name
            for field_name in record.get("_unresolved_fields", [])
            if (record_id, str(field_name)) not in resolved_pairs
        ]
        if unresolved:
            record["_unresolved_fields"] = unresolved
        else:
            record.pop("_unresolved_fields", None)
        if (
            record_id not in remaining_issue_targets
            and not unresolved
            and not record.get("_invalid_observation_fields")
            and not record.get("_reported_field_conflicts")
        ):
            record.pop("extraction_status", None)


def _report_employment_slot_unresolved(
    parse_result: Any,
    *,
    record: dict[str, Any],
    field_name: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
    column: int,
) -> None:
    """Report one blank value cell whose canonical header/row binding is exact."""

    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    if field_name in set(record.get("_source_absent_fields") or ()):
        return
    _append_internal_field(record, "_unresolved_fields", field_name)
    record["extraction_status"] = "review"
    reported = record.setdefault("_reported_employment_slot_fields", [])
    if field_name in reported:
        return
    reported.append(field_name)
    record_issue(
        parse_result,
        make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_employment_canonical_cell_unresolved",
            message=(
                "A canonical employment value cell was blank or unreadable even though "
                "its header column and keyed business row were established."
            ),
            parser_stage="candidate_b_employment_canonical_slots",
            target_dataset="employment_records",
            target_record_id=str(record["employment_record_id"]),
            field_name=field_name,
            observed_value={"sequence": record.get("sequence"), "physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index, column=column),),
            reason_codes=(
                "canonical_header_column",
                "keyed_employment_row",
                "value_cell_unreadable",
                "normalized_value_withheld",
            ),
        ),
    )


def _ordered_employment_detail_values(
    value: Any,
) -> tuple[dict[str, str], set[str]]:
    """Decode unique finite detail tokens in their canonical role order."""

    marker = _employment_signature(value)
    role_order = ("occupation", "industry", "position", "professional_title")
    role_candidates: dict[str, list[tuple[str, str, int, int]]] = {}
    for role in role_order:
        observed: list[tuple[str, str, int, int]] = []
        for candidate in _EMPLOYMENT_DETAIL_VOCABULARIES[role]:
            signature = _employment_signature(candidate)
            start = 0
            while signature and (index := marker.find(signature, start)) >= 0:
                observed.append((candidate, signature, index, index + len(signature)))
                start = index + 1
        maximal = [
            item
            for item in observed
            if not any(
                item[2] >= other[2]
                and item[3] <= other[3]
                and len(item[1]) < len(other[1])
                for other in observed
            )
        ]
        if len(maximal) == 1:
            role_candidates[role] = maximal

    unique_candidates = {
        role: candidates[0]
        for role, candidates in role_candidates.items()
        if not any(
            candidates[0][2] >= other_candidates[0][2]
            and candidates[0][3] <= other_candidates[0][3]
            and len(candidates[0][1]) < len(other_candidates[0][1])
            for other_role, other_candidates in role_candidates.items()
            if other_role != role
        )
        and not any(
            candidates[0][1] == other_candidates[0][1]
            and candidates[0][2:] == other_candidates[0][2:]
            for other_role, other_candidates in role_candidates.items()
            if other_role != role
        )
    }
    selected: dict[str, str] = {
        role: unique_candidates[role][0]
        for role in ("occupation", "industry")
        if role in unique_candidates
    }
    rejected_roles: set[str] = set(role_candidates) - set(unique_candidates)
    last_end = -1
    # OCR traversal can reverse the two wide description columns even when
    # their finite vocabularies own each token uniquely.  Position and title,
    # however, are adjacent short scalars: enforce their canonical order so a
    # pre-position ``无`` is not silently promoted to professional_title.
    for role in ("position", "professional_title"):
        candidate = unique_candidates.get(role)
        if candidate is None:
            continue
        canonical, _signature, start, end = candidate
        if start < last_end:
            rejected_roles.add(role)
            continue
        selected[role] = canonical
        last_end = end
    return selected, rejected_roles


def _bounded_glyph_edit_distance(left: str, right: str, *, limit: int = 1) -> int:
    """Return a small edit distance, stopping once ``limit`` is exceeded."""

    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _recover_unique_cluster_occupation(
    value: Any,
    *,
    owned_values: Iterable[Any],
) -> str | None:
    """Recover one long finite occupation after bounded OCR interleaving.

    The closed report schema supplies a finite occupation vocabulary.  This
    correction is deliberately unavailable to short or open-text fields: after
    removing independently decoded detail values and one isolated layout digit,
    a long occupation must be the unique candidate within one glyph edit and
    leave at most three unowned glyphs in the row cluster.
    """

    source_marker = _employment_signature(value)
    if not source_marker:
        return None
    matches: list[tuple[int, int, str]] = []
    for candidate in _EMPLOYMENT_OCCUPATIONS:
        candidate_marker = _employment_signature(candidate)
        if len(candidate_marker) < 8:
            continue
        marker = source_marker
        for owned_value in owned_values:
            owned_marker = _employment_signature(owned_value)
            if (
                owned_marker
                and owned_marker not in candidate_marker
                and owned_marker in marker
            ):
                marker = marker.replace(owned_marker, "", 1)
        marker = re.sub(r"(?<!\d)\d(?!\d)", "", marker)
        for window_length in range(
            max(1, len(candidate_marker) - 1), len(candidate_marker) + 2
        ):
            outside = len(marker) - window_length
            if outside < 0 or outside > 3:
                continue
            for start in range(0, len(marker) - window_length + 1):
                window = marker[start : start + window_length]
                distance = _bounded_glyph_edit_distance(
                    window, candidate_marker, limit=1
                )
                if distance <= 1:
                    matches.append((distance, outside, candidate))
    if not matches:
        return None
    best_score = min((distance, outside) for distance, outside, _candidate in matches)
    candidates = {
        candidate
        for distance, outside, candidate in matches
        if (distance, outside) == best_score
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _drop_detail_values_owned_by_longer_roles(
    values: dict[str, str], *, source_value: Any
) -> None:
    """Do not reuse glyphs inside occupation/industry as a short role."""

    source_marker = _employment_signature(source_value)
    for role in ("position", "professional_title"):
        value = values.get(role)
        value_marker = _employment_signature(value)
        if not value_marker:
            continue
        explained_occurrences = sum(
            _employment_signature(values.get(owner)).count(value_marker)
            for owner in ("occupation", "industry")
        )
        if (
            explained_occurrences
            and source_marker.count(value_marker) <= explained_occurrences
        ):
            values.pop(role, None)


def _employment_value_source(
    row: tuple[str, ...],
    *,
    sequence_column: int,
    value: Any,
) -> tuple[str, int | None]:
    """Locate one decoded token in a unique source cell when possible."""

    signature = _employment_signature(value)
    hits = [
        (raw, column)
        for column, raw in enumerate(row)
        if column != sequence_column and signature and signature in _employment_signature(raw)
    ]
    if len(hits) == 1:
        return _clean(hits[0][0]), hits[0][1]
    union_raw = " ".join(
        _clean(raw)
        for column, raw in enumerate(row)
        if column != sequence_column and _clean(raw)
    )
    return union_raw, None


def _decode_clustered_employment_detail(
    parse_result: Any,
    *,
    record: dict[str, Any],
    row: tuple[str, ...],
    slots: Mapping[str, int],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    """Decode one collapsed detail row from the union of its physical cells."""

    target_record_id = str(record["employment_record_id"])
    parser_stage = "candidate_b_employment_cluster_decoder"
    sequence_column = int(slots["sequence"])
    union_raw = " ".join(
        _clean(raw)
        for column, raw in enumerate(row)
        if column != sequence_column and _clean(raw)
    )

    full_dates = [
        (span, normalized)
        for span, normalized in _valid_date_spans(union_raw)
        if len(normalized) == 10
    ]
    working = union_raw
    updated_date: str | None = None
    updated_raw = ""
    if len(full_dates) == 1:
        (start, end), updated_date = full_dates[0]
        updated_raw = union_raw[start:end]
        working = union_raw[:start] + " " * (end - start) + union_raw[end:]

    year_matches = list(
        re.finditer(
            r"(?<!\d)((?:19|20)\d{2})(?!\s*[.,年/月\-]\s*\d)(?!\d)",
            working,
        )
    )
    entry_year: int | None = None
    entry_year_raw = ""
    if len(year_matches) == 1:
        match = year_matches[0]
        entry_year_raw = match.group(1)
        entry_year = int(entry_year_raw)
        working = working[: match.start()] + " " * len(match.group(0)) + working[match.end() :]

    cluster_values, _rejected_roles = _ordered_employment_detail_values(working)
    corrected_occupation = False
    if "occupation" not in cluster_values:
        occupation = _recover_unique_cluster_occupation(
            working,
            owned_values=cluster_values.values(),
        )
        if occupation is not None:
            cluster_values["occupation"] = occupation
            corrected_occupation = True
    _drop_detail_values_owned_by_longer_roles(cluster_values, source_value=working)
    for role in ("occupation", "industry", "position", "professional_title"):
        value = cluster_values.get(role)
        if value is None:
            _report_employment_cluster_field_unresolved(
                parse_result,
                record=record,
                field_name=role,
                row=row,
                page=page,
                table=table,
                row_index=row_index,
            )
            continue
        cluster_raw, column = _employment_value_source(
            row,
            sequence_column=sequence_column,
            value=value,
        )
        source_ref = _source_ref(page, table, row=row_index, column=column)
        binding = (
            "closed_canonical_employment_finite_occupation_correction"
            if role == "occupation" and corrected_occupation
            else "closed_canonical_employment_row_union"
        )
        source_ref["binding"] = binding
        source_ref["binding_quality"] = binding
        _merge_exact_observation(
            parse_result,
            record,
            dataset="employment_records",
            target_record_id=target_record_id,
            field_name=role,
            value=value,
            raw=cluster_raw,
            source_ref=source_ref,
            parser_stage=parser_stage,
        )

    if entry_year is None:
        _report_employment_cluster_field_unresolved(
            parse_result,
            record=record,
            field_name="entry_year",
            row=row,
            page=page,
            table=table,
            row_index=row_index,
        )
    else:
        year_raw, year_column = _employment_value_source(
            row,
            sequence_column=sequence_column,
            value=entry_year_raw,
        )
        _merge_exact_observation(
            parse_result,
            record,
            dataset="employment_records",
            target_record_id=target_record_id,
            field_name="entry_year",
            value=entry_year,
            raw=year_raw,
            source_ref=_source_ref(page, table, row=row_index, column=year_column),
            parser_stage=parser_stage,
        )

    if updated_date is None:
        date_slot_raw = _slot_value(row, slots, "information_updated_date")
        if is_explicit_source_absence(date_slot_raw):
            _mark_source_absent(record, "information_updated_date", date_slot_raw)
        else:
            _report_employment_cluster_field_unresolved(
                parse_result,
                record=record,
                field_name="information_updated_date",
                row=row,
                page=page,
                table=table,
                row_index=row_index,
            )
    else:
        _date_raw, date_column = _employment_value_source(
            row,
            sequence_column=sequence_column,
            value=updated_raw,
        )
        _merge_exact_observation(
            parse_result,
            record,
            dataset="employment_records",
            target_record_id=target_record_id,
            field_name="information_updated_date",
            value=updated_date,
            raw=updated_raw,
            source_ref=_source_ref(page, table, row=row_index, column=date_column),
            parser_stage=parser_stage,
        )

    # Residue is assessed per physical cell even though decoding used the row
    # union.  This keeps uncertainty attached only to values sharing the same
    # OCR cell rather than contaminating every recovered field in the row.
    for column, raw in enumerate(row):
        if column == sequence_column or not _clean(raw):
            continue
        scoped_fields = [
            role
            for role, value in cluster_values.items()
            if _employment_signature(value) in _employment_signature(raw)
        ]
        owned_values: list[Any] = [cluster_values[role] for role in scoped_fields]
        if updated_raw and _employment_signature(updated_raw) in _employment_signature(raw):
            owned_values.append(updated_raw)
            scoped_fields.append("information_updated_date")
        if entry_year_raw and entry_year_raw in _employment_signature(raw):
            owned_values.append(entry_year_raw)
            scoped_fields.append("entry_year")
        residue = _employment_cluster_residue(raw, owned_values)
        if residue and scoped_fields:
            _report_employment_cluster_residue(
                parse_result,
                record=record,
                field_names=scoped_fields,
                raw=_clean(raw),
                residue=residue,
                page=page,
                table=table,
                row_index=row_index,
                column=column,
            )


def _report_optional_public_continuation_missing(
    parse_result: Any,
    *,
    dataset: str,
    target_record_id: str,
    sequence: int,
    source_refs: Iterable[Mapping[str, Any]],
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="page_continuation",
            issue_code="candidate_b_public_record_continuation_missing",
            message="A canonical public-record start block was preserved, but its required continuation block was not observed.",
            parser_stage="candidate_b_public_record_boundaries",
            target_dataset=dataset,
            target_record_id=target_record_id,
            observed_value={"sequence": sequence, "start_block_observed": True},
            candidate_value={
                "missing_fields": ["employer", "information_updated_month"]
            },
            source_refs=tuple(source_refs),
            reason_codes=(
                "canonical_start_block_observed",
                "required_continuation_not_observed",
                "partial_record_preserved",
            ),
        ),
    )


def _report_unowned_optional_public_fragment(
    parse_result: Any,
    *,
    dataset: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="page_continuation",
            issue_code="candidate_b_public_record_continuation_unowned",
            message="A canonical public-record continuation block had no preceding start block and was not attached by proximity.",
            parser_stage="candidate_b_public_record_boundaries",
            target_dataset=dataset,
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=(
                "canonical_continuation_block_observed",
                "preceding_start_block_not_observed",
                "proximity_attachment_forbidden",
                "record_not_invented",
            ),
        ),
    )


def _report_unkeyed_business_row(
    parse_result: Any,
    *,
    dataset: str,
    row: tuple[str, ...],
    page: Any,
    table: Any,
    row_index: int,
) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import make_issue, record_issue

    record_issue(
        parse_result,
        make_issue(
            category="ocr_cell_level_error",
            issue_code="candidate_b_sequence_cell_unresolved",
            message="A canonical business row was present but its printed sequence cell was unreadable; no row identity was invented.",
            parser_stage="candidate_b_canonical_cell_graph",
            target_dataset=dataset,
            field_name="sequence",
            observed_value={"physical_cells": list(row)},
            source_refs=(_source_ref(page, table, row=row_index),),
            reason_codes=("canonical_business_row", "printed_sequence_unreadable", "record_not_silently_dropped"),
        ),
    )


def _extract_residence_records(parse_result: Any) -> list[dict[str, Any]]:
    aliases = {
        "sequence": ("编号",),
        "address": ("居住地址",),
        "residential_phone": ("住宅电话",),
        "residence_status": ("居住状况",),
        "information_updated_date": ("信息更新日期",),
    }
    records: dict[int, dict[str, Any]] = {}
    providers: defaultdict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    provider_source_absent: set[int] = set()
    unkeyed_providers: list[
        tuple[str, dict[str, Any], tuple[str, ...], Any, Any, int]
    ] = []
    active_slots: dict[str, int] = {}
    mode = ""
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            table_id = str(getattr(table, "table_id", "") or "")
            table_text = _compact(" ".join(cell for row in rows for cell in row))
            continues = bool(
                previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            residence_schema_markers = ("居住地址", "住宅电话", "居住状况")
            other_section_markers = ("学历", "学位", "电子邮箱", "通讯地址", "户籍地址", "工作单位", "职业")
            if (
                mode in {"residence", "provider"}
                and not any(marker in table_text for marker in residence_schema_markers)
                and any(marker in table_text for marker in other_section_markers)
            ):
                continues = False
            if not continues:
                active_slots = {}
                mode = ""
            if not continues and not any(marker in table_text for marker in residence_schema_markers):
                previous_table_id = table_id or previous_table_id
                continue
            if continues and mode == "residence" and rows:
                provider_shaped, provider_column = _continuation_rows_are_provider_shaped(
                    rows,
                    known_sequences=set(records),
                )
                if provider_shaped and provider_column is not None:
                    active_slots = {"sequence": 0, "data_provider": provider_column}
                    mode = "provider"
            for row_index, row in enumerate(rows):
                residence_slots = _canonical_header_slots(row, aliases)
                compact_header = _compact("".join(row))
                header_anchor = len(residence_slots) >= 2
                if (
                    all(marker in compact_header for marker in ("编号", "居住地址", "信息更新日期"))
                    and not set(aliases) <= set(residence_slots)
                ) or (header_anchor and not set(aliases) <= set(residence_slots)):
                    _report_header_graph_failure(
                        parse_result,
                        dataset="residence_records",
                        page=page,
                        table=table,
                        row_index=row_index,
                        row=row,
                    )
                    active_slots = {}
                    mode = ""
                    continue
                if set(aliases) <= set(residence_slots):
                    active_slots = residence_slots
                    mode = "residence"
                    continue
                if "数据发生机构名称" in _compact("".join(row)) and mode in {"residence", "provider"}:
                    provider_slots = _canonical_header_slots(
                        row, {"sequence": ("编号",), "data_provider": ("数据发生机构名称",)}
                    )
                    active_slots = provider_slots
                    mode = "provider"
                    continue
                if not active_slots or mode not in {"residence", "provider"}:
                    continue
                sequence = _sequence_value(row, active_slots)
                if sequence is None and mode == "provider":
                    first = _clean(row[0] if row else "")
                    sequence = int(first) if first.isdigit() else None
                if sequence is None:
                    nonempty = _nonempty(row)
                    if (
                        mode == "provider"
                        and continues
                        and 0 < len(nonempty) <= 2
                        and any(
                            re.search(r"(?:银行|信用社|有限公司|管理中心|征信中心)", value)
                            for value in nonempty
                        )
                    ):
                        institution = next(
                            (
                                value
                                for value in reversed(nonempty)
                                if re.search(
                                    r"(?:银行|信用社|有限公司|管理中心|征信中心)",
                                    value,
                                )
                            ),
                            "",
                        )
                        if institution:
                            unkeyed_providers.append(
                                (
                                    institution,
                                    _source_ref(page, table, row=row_index),
                                    tuple(row),
                                    page,
                                    table,
                                    row_index,
                                )
                            )
                        else:
                            _report_unkeyed_fragment(
                                parse_result,
                                dataset="residence_records",
                                row=row,
                                page=page,
                                table=table,
                                row_index=row_index,
                            )
                    elif mode == "provider" and nonempty:
                        _report_unkeyed_fragment(
                            parse_result,
                            dataset="residence_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    elif mode == "residence" and nonempty:
                        _report_unkeyed_business_row(
                            parse_result,
                            dataset="residence_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    continue
                ref = _source_ref(page, table, row=row_index)
                if mode == "provider":
                    provider = _slot_value(row, active_slots, "data_provider")
                    provider_column = active_slots.get("data_provider")
                    if is_explicit_source_absence(provider):
                        provider_source_absent.add(sequence)
                    elif provider and provider_column is not None:
                        providers[sequence].append(
                            (provider, _source_ref(page, table, row=row_index, column=provider_column))
                        )
                    continue
                residence_id = stable_record_id("credit_residence", sequence)
                address = _slot_value(row, active_slots, "address")
                updated_raw = _slot_value(row, active_slots, "information_updated_date")
                updated = _date(updated_raw)
                collapsed_updated: str | None = None
                if address and not updated and not updated_raw:
                    date_matches = list(_DATE_RE.finditer(address))
                    if len(date_matches) == 1:
                        candidate_date = _date(date_matches[0].group(0))
                        residual = (
                            address[: date_matches[0].start()]
                            + " "
                            + address[date_matches[0].end() :]
                        )
                        residual = re.sub(
                            r"^[\s\"“”'*#=—–-]+|[\s\"“”'*#=—–-]+$",
                            "",
                            residual,
                        )
                        if candidate_date and len(_compact(residual)) >= 4:
                            address = residual
                            collapsed_updated = candidate_date
                            updated = candidate_date
                record = records.setdefault(
                    sequence,
                    {
                        "sequence": sequence,
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_residence_table",
                        "source_refs": [],
                        "record_id": residence_id,
                        "residence_record_id": residence_id,
                        "confidence": 1.0,
                    },
                )
                record["source_refs"].append(ref)
                if address == "--":
                    _mark_source_absent(record, "address", address)
                elif address:
                    address_ref = _source_ref(
                        page, table, row=row_index, column=active_slots["address"]
                    )
                    suspicious_address = bool(
                        _DATE_RE.search(address)
                        or re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", address)
                        or any(
                            label in _compact(address)
                            for label_names in aliases.values()
                            for label in label_names
                        )
                    )
                    if suspicious_address:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="address",
                            raw=address,
                            source_ref=address_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="address",
                            value=address,
                            raw=address,
                            source_ref=address_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                else:
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_residence_required_cell_unresolved",
                        dataset="residence_records",
                        sequence=sequence,
                        field_name="address",
                        row=row,
                        page=page,
                        table=table,
                        row_index=row_index,
                        target_record_id=residence_id,
                    )
                phone = _slot_value(row, active_slots, "residential_phone")
                status = _slot_value(row, active_slots, "residence_status")
                if is_explicit_source_absence(phone):
                    _mark_source_absent(record, "residential_phone", phone)
                elif phone:
                    phone_digits = re.sub(r"\D", "", phone)
                    phone_ref = _source_ref(
                        page, table, row=row_index, column=active_slots["residential_phone"]
                    )
                    if any(character.isalpha() for character in phone) or not 5 <= len(phone_digits) <= 16:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="residential_phone",
                            raw=phone,
                            source_ref=phone_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="residence_records",
                            target_record_id=residence_id,
                            field_name="residential_phone",
                            value=phone,
                            raw=phone,
                            source_ref=phone_ref,
                            parser_stage="candidate_b_residence_canonical_slots",
                        )
                if status == "--":
                    _mark_source_absent(record, "residence_status", status)
                elif status:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="residence_records",
                        target_record_id=residence_id,
                        field_name="residence_status",
                        value=status,
                        raw=status,
                        source_ref=_source_ref(
                            page, table, row=row_index, column=active_slots["residence_status"]
                        ),
                        parser_stage="candidate_b_residence_canonical_slots",
                    )
                if updated_raw and not is_explicit_source_absence(updated_raw):
                    _merge_canonical_date_observation(
                        parse_result,
                        record,
                        dataset="residence_records",
                        target_record_id=residence_id,
                        field_name="information_updated_date",
                        raw=updated_raw,
                        source_ref=_source_ref(
                            page,
                            table,
                            row=row_index,
                            column=active_slots["information_updated_date"],
                        ),
                        parser_stage="candidate_b_residence_canonical_slots",
                    )
                elif collapsed_updated:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="residence_records",
                        target_record_id=residence_id,
                        field_name="information_updated_date",
                        value=collapsed_updated,
                        raw=collapsed_updated,
                        source_ref=_source_ref(
                            page,
                            table,
                            row=row_index,
                            column=active_slots["address"],
                        ),
                        parser_stage="candidate_b_residence_canonical_slots",
                    )
                elif is_explicit_source_absence(updated_raw):
                    _mark_source_absent(record, "information_updated_date", updated_raw)
                elif not updated_raw:
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_residence_required_cell_unresolved",
                        dataset="residence_records",
                        sequence=sequence,
                        field_name="information_updated_date",
                        row=row,
                        page=page,
                        table=table,
                        row_index=row_index,
                        target_record_id=residence_id,
                    )
            previous_table_id = table_id or previous_table_id

    keyed_sequences = {sequence for sequence, values in providers.items() if values}
    missing_provider_sequences = [
        sequence for sequence in sorted(records) if sequence not in keyed_sequences
    ]
    if (
        unkeyed_providers
        and len(unkeyed_providers) == len(missing_provider_sequences)
        and sorted(records) == list(range(1, max(records, default=0) + 1))
        and missing_provider_sequences
        == list(range(min(missing_provider_sequences), max(records) + 1))
    ):
        for sequence, (provider, provider_ref, _row, _page, _table, _row_index) in zip(
            missing_provider_sequences,
            unkeyed_providers,
            strict=True,
        ):
            providers[sequence].append((provider, provider_ref))
        unkeyed_providers = []
    for _provider, _ref, row, page, table, row_index in unkeyed_providers:
        _report_unkeyed_fragment(
            parse_result,
            dataset="residence_records",
            row=row,
            page=page,
            table=table,
            row_index=row_index,
        )
    for sequence, record in records.items():
        residence_id = str(record["residence_record_id"])
        if sequence in provider_source_absent:
            _mark_source_absent(record, "data_provider", "--")
        for provider, provider_ref in providers.get(sequence, ()):
            record["source_refs"].append(provider_ref)
            _merge_exact_observation(
                parse_result,
                record,
                dataset="residence_records",
                target_record_id=residence_id,
                field_name="data_provider",
                value=provider,
                raw=provider,
                source_ref=provider_ref,
                parser_stage="candidate_b_residence_canonical_slots",
            )
        if (
            not record.get("data_provider")
            and "data_provider" not in set(record.get("_source_absent_fields") or ())
        ):
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record["extraction_status"] = "review"
            record_issue(
                parse_result,
                make_issue(
                    category="page_continuation",
                    issue_code="candidate_b_residence_provider_missing",
                    message="A canonical residence row has no safely bound provider row.",
                    parser_stage="candidate_b_residence_canonical_slots",
                    target_dataset="residence_records",
                    target_record_id=residence_id,
                    field_name="data_provider",
                    observed_value={"sequence": sequence},
                    source_refs=tuple(record.get("source_refs") or ()),
                    reason_codes=(
                        "canonical_residence_row_observed",
                        "provider_component_not_bound",
                        "row_marked_for_review",
                    ),
                ),
            )
    output = [records[key] for key in sorted(records)]
    _enforce_observed_header_terminal_invariant(
        parse_result,
        dataset="residence_records",
        aliases=aliases,
        records=output,
    )
    return output


def _extract_employment_records(parse_result: Any) -> list[dict[str, Any]]:
    basic_aliases = {
        "sequence": ("编号",),
        "employer": ("工作单位",),
        "employer_type": ("单位性质",),
        "employer_address": ("单位地址",),
        "employer_phone": ("单位电话",),
    }
    detail_aliases = {
        "sequence": ("编号",),
        "occupation": ("职业",),
        "industry": ("行业",),
        "position": ("职务",),
        "professional_title": ("职称",),
        "entry_year": ("进入本单位年份",),
        "information_updated_date": ("信息更新日期",),
    }
    records: dict[int, dict[str, Any]] = {}
    active_slots: dict[str, int] = {}
    mode = ""
    basic_header_recovered = False
    basic_cluster_column: int | None = None
    detail_header_clustered = False
    sequence_repairs: dict[int, int] = {}
    pending_basic_overflow: dict[str, str] = {}
    basic_header_row_index: int | None = None
    employment_section_active = False
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            table_id = str(getattr(table, "table_id", "") or "")
            table_text = _compact(" ".join(cell for row in rows for cell in row))
            table_has_employment_schema = any(
                marker in table_text
                for marker in ("工作单位", "单位性质", "单位电话", "职业", "职务", "职称")
            )
            continues = bool(
                previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            other_section_markers = (
                "居住地址",
                "住宅电话",
                "居住状况",
                "手机号码",
                "配偶",
                "学历",
                "学位",
                "电子邮箱",
                "通讯地址",
                "户籍地址",
            )
            if (
                employment_section_active
                and not table_has_employment_schema
                and any(marker in table_text for marker in other_section_markers)
            ):
                continues = False
                employment_section_active = False
            if not continues:
                active_slots = {}
                mode = ""
                basic_header_recovered = False
                basic_cluster_column = None
                detail_header_clustered = False
                sequence_repairs = {}
                pending_basic_overflow = {}
                basic_header_row_index = None
                employment_section_active = table_has_employment_schema
            elif table_has_employment_schema:
                employment_section_active = True
            if not employment_section_active and not table_has_employment_schema:
                previous_table_id = table_id or previous_table_id
                continue
            if continues and mode in {"basic", "detail"} and rows:
                provider_shaped, provider_column = _continuation_rows_are_provider_shaped(
                    rows,
                    known_sequences=set(records),
                )
                if provider_shaped and provider_column is not None:
                    active_slots = {"sequence": 0, "data_provider": provider_column}
                    mode = "provider"
                    basic_header_recovered = False
                    basic_cluster_column = None
                    detail_header_clustered = False
                    sequence_repairs = {}
                    pending_basic_overflow = {}
                    basic_header_row_index = None
            for row_index, row in enumerate(rows):
                basic_slots = _canonical_header_slots(row, basic_aliases)
                detail_slots = _canonical_header_slots(row, detail_aliases)
                recovered_basic = _recovered_employment_basic_header(row, basic_aliases)
                collapsed_basic = _collapsed_employment_basic_header(row)
                clustered_detail = _clustered_employment_detail_header(row, detail_aliases)
                provider_header = _employment_provider_header(row)
                compact_header = _compact("".join(row))
                basic_anchor = len(basic_slots) >= 2
                detail_anchor = len(detail_slots) >= 2
                broken_basic = (
                    all(marker in compact_header for marker in ("编号", "工作单位", "单位性质", "单位电话"))
                    or basic_anchor
                ) and not set(basic_aliases) <= set(basic_slots) and recovered_basic is None and collapsed_basic is None
                broken_detail = (
                    all(marker in compact_header for marker in ("编号", "职业", "行业", "职务", "职称"))
                    or detail_anchor
                ) and not set(detail_aliases) <= set(detail_slots) and clustered_detail is None
                if broken_basic or broken_detail:
                    _report_header_graph_failure(
                        parse_result,
                        dataset="employment_records",
                        page=page,
                        table=table,
                        row_index=row_index,
                        row=row,
                    )
                    active_slots = {}
                    mode = ""
                    basic_header_recovered = False
                    basic_cluster_column = None
                    detail_header_clustered = False
                    sequence_repairs = {}
                    pending_basic_overflow = {}
                    basic_header_row_index = None
                    employment_section_active = True
                    continue
                if set(basic_aliases) <= set(basic_slots):
                    active_slots = basic_slots
                    mode = "basic"
                    basic_header_recovered = False
                    basic_cluster_column = None
                    detail_header_clustered = False
                    sequence_repairs = _employment_sequence_repairs(
                        rows,
                        start=row_index + 1,
                        sequence_column=active_slots["sequence"],
                        mode="basic",
                    )
                    pending_basic_overflow = {}
                    basic_header_row_index = None
                    continue
                if recovered_basic is not None:
                    active_slots, pending_basic_overflow = recovered_basic
                    mode = "basic"
                    basic_header_recovered = True
                    basic_cluster_column = None
                    detail_header_clustered = False
                    sequence_repairs = _employment_sequence_repairs(
                        rows,
                        start=row_index + 1,
                        sequence_column=active_slots["sequence"],
                        mode="basic",
                    )
                    basic_header_row_index = row_index
                    continue
                if collapsed_basic is not None:
                    sequence_column, basic_cluster_column = collapsed_basic
                    active_slots = {"sequence": sequence_column}
                    mode = "basic"
                    basic_header_recovered = False
                    detail_header_clustered = False
                    sequence_repairs = _employment_sequence_repairs(
                        rows,
                        start=row_index + 1,
                        sequence_column=active_slots["sequence"],
                        mode="basic",
                        cluster_column=basic_cluster_column,
                    )
                    pending_basic_overflow = {}
                    basic_header_row_index = row_index
                    continue
                if set(detail_aliases) <= set(detail_slots):
                    active_slots = detail_slots
                    mode = "detail"
                    basic_header_recovered = False
                    basic_cluster_column = None
                    detail_header_clustered = False
                    sequence_repairs = _employment_sequence_repairs(
                        rows,
                        start=row_index + 1,
                        sequence_column=active_slots["sequence"],
                        mode="detail",
                        known_basic_population=(
                            sequence
                            for sequence, record in records.items()
                            if "basic" in set(record.get("_observed_components") or ())
                        ),
                    )
                    pending_basic_overflow = {}
                    basic_header_row_index = None
                    continue
                if clustered_detail is not None:
                    active_slots = clustered_detail
                    mode = "detail"
                    basic_header_recovered = False
                    basic_cluster_column = None
                    detail_header_clustered = True
                    sequence_repairs = _employment_sequence_repairs(
                        rows,
                        start=row_index + 1,
                        sequence_column=active_slots["sequence"],
                        mode="detail",
                        known_basic_population=(
                            sequence
                            for sequence, record in records.items()
                            if "basic" in set(record.get("_observed_components") or ())
                        ),
                    )
                    pending_basic_overflow = {}
                    basic_header_row_index = None
                    continue
                if employment_section_active and provider_header is not None:
                    active_slots = provider_header
                    mode = "provider"
                    basic_header_recovered = False
                    basic_cluster_column = None
                    detail_header_clustered = False
                    sequence_repairs = {}
                    pending_basic_overflow = {}
                    basic_header_row_index = None
                    continue
                if not active_slots:
                    continue
                sequence = _employment_sequence_value(row, active_slots)
                sequence_recovered = False
                cluster_raw = ""
                if mode == "basic" and basic_cluster_column is not None:
                    cluster_raw = _slot_value(row, {"cluster": basic_cluster_column}, "cluster")
                    if active_slots.get("sequence") == basic_cluster_column:
                        match = re.fullmatch(r"\s*(\d{1,3})\s+(.+)", cluster_raw)
                        if match is not None:
                            sequence = int(match.group(1))
                            cluster_raw = match.group(2).strip()
                if sequence is None and mode == "provider":
                    first = _clean(row[0] if row else "")
                    sequence = int(first) if first.isdigit() else None
                if sequence is None and row_index in sequence_repairs:
                    sequence = sequence_repairs[row_index]
                    sequence_recovered = True
                if sequence is None:
                    if mode == "provider" and _nonempty(row):
                        _report_unkeyed_fragment(
                            parse_result,
                            dataset="employment_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    elif mode in {"basic", "detail"} and _nonempty(row):
                        _report_unkeyed_business_row(
                            parse_result,
                            dataset="employment_records",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                        )
                    continue
                record = records.setdefault(
                    sequence,
                    {
                        "record_id": stable_record_id("credit_employment", sequence),
                        "employment_record_id": stable_record_id("credit_employment", sequence),
                        "sequence": sequence,
                        "page": page_number,
                        "source_page": source_page,
                        "source": "native_personal_detail_employment_table",
                        "source_refs": [],
                        "confidence": 1.0,
                    },
                )
                row_ref = _source_ref(page, table, row=row_index)
                if sequence_recovered:
                    row_ref["binding"] = "closed_canonical_employment_sequence_run"
                    row_ref["binding_quality"] = "closed_canonical_employment_sequence_run"
                record["source_refs"].append(row_ref)
                observed_components = record.setdefault("_observed_components", [])
                if mode not in observed_components:
                    observed_components.append(mode)
                if mode == "provider":
                    provider, provider_column, provider_resolution = (
                        _employment_provider_observation(
                            row,
                            sequence_column=active_slots["sequence"],
                        )
                    )
                    if provider_resolution == "source_absent" and provider:
                        _mark_source_absent(record, "data_provider", provider)
                    elif provider_resolution == "exact" and provider and provider_column is not None:
                        provider_raw = _clean(
                            row[provider_column]
                            if provider_column < len(row)
                            else provider
                        )
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=str(record["employment_record_id"]),
                            field_name="data_provider",
                            value=provider,
                            raw=provider_raw,
                            source_ref=_source_ref(
                                page, table, row=row_index, column=provider_column
                            ),
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                    else:
                        _report_employment_provider_unresolved(
                            parse_result,
                            record=record,
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                            resolution=provider_resolution,
                        )
                    continue
                if mode == "basic" and basic_cluster_column is not None:
                    target_record_id = str(record["employment_record_id"])
                    source_ref = _source_ref(
                        page,
                        table,
                        row=row_index,
                        column=basic_cluster_column,
                    )
                    source_ref["binding"] = "closed_canonical_employment_cluster"
                    source_ref["binding_quality"] = "closed_canonical_employment_cluster"
                    decoded = decode_employment_basic_cluster(cluster_raw)
                    for field_name, value in decoded.fields.items():
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=field_name,
                            value=value,
                            raw=cluster_raw,
                            source_ref=source_ref,
                            parser_stage="candidate_b_employment_closed_cluster",
                        )
                    _report_collapsed_cluster_fields(
                        parse_result,
                        record,
                        dataset="employment_records",
                        target_record_id=target_record_id,
                        raw=decoded.unresolved_residue or cluster_raw,
                        source_ref=source_ref,
                        unresolved_fields=decoded.unresolved_fields,
                        parser_stage="candidate_b_employment_closed_cluster",
                    )
                    continue
                if mode == "detail" and detail_header_clustered:
                    _decode_clustered_employment_detail(
                        parse_result,
                        record=record,
                        row=row,
                        slots=active_slots,
                        page=page,
                        table=table,
                        row_index=row_index,
                    )
                    continue
                row_basic_overflow: dict[str, str] = {}
                if mode == "basic" and basic_header_recovered and pending_basic_overflow:
                    # Header-cell overflow can only belong to the immediately
                    # following keyed row.  Never carry it to a later record.
                    row_basic_overflow = pending_basic_overflow
                    pending_basic_overflow = {}
                roles = basic_aliases if mode == "basic" else detail_aliases
                for role in roles:
                    if role == "sequence":
                        continue
                    raw = _slot_value(row, active_slots, role)
                    ref = _source_ref(page, table, row=row_index, column=active_slots[role])
                    if (
                        mode == "basic"
                        and basic_header_recovered
                        and not raw
                        and role in row_basic_overflow
                    ):
                        raw = row_basic_overflow[role]
                        ref = _source_ref(
                            page,
                            table,
                            row=basic_header_row_index,
                            column=active_slots[role],
                        )
                    if is_explicit_source_absence(raw):
                        _mark_source_absent(record, role, raw)
                        continue
                    if not raw:
                        if not (mode == "basic" and basic_header_recovered):
                            _report_employment_slot_unresolved(
                                parse_result,
                                record=record,
                                field_name=role,
                                row=row,
                                page=page,
                                table=table,
                                row_index=row_index,
                                column=active_slots[role],
                            )
                        continue
                    target_record_id = str(record["employment_record_id"])
                    if role == "entry_year":
                        match = re.fullmatch(r"\D*((?:19|20)\d{2})\D*", raw)
                        if match:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=int(match.group(1)),
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role == "information_updated_date":
                        _merge_canonical_date_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=role,
                            raw=raw,
                            source_ref=ref,
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                    elif role == "employer_phone":
                        phone = _canonical_employer_phone(raw)
                        if phone is None:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=phone,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role == "employer_type":
                        employer_type = _finite_employment_value(raw, _EMPLOYER_TYPES)
                        if employer_type:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=employer_type,
                                raw=employer_type,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role in _EMPLOYMENT_DETAIL_VOCABULARIES:
                        canonical = _finite_employment_value(
                            raw, _EMPLOYMENT_DETAIL_VOCABULARIES[role]
                        )
                        if canonical:
                            _merge_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                value=canonical,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                record,
                                dataset="employment_records",
                                target_record_id=target_record_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_employment_canonical_slots",
                            )
                    elif role == "employer" and not validate_pboc_field(
                        raw, "employer_name"
                    ).valid:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=role,
                            raw=raw,
                            source_ref=ref,
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                    elif role == "employer_address" and not validate_pboc_field(
                        raw, "address"
                    ).valid:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=role,
                            raw=raw,
                            source_ref=ref,
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                    else:
                        _merge_exact_observation(
                            parse_result,
                            record,
                            dataset="employment_records",
                            target_record_id=target_record_id,
                            field_name=role,
                            value=raw,
                            raw=raw,
                            source_ref=ref,
                            parser_stage="candidate_b_employment_canonical_slots",
                        )
                if mode == "basic" and basic_header_recovered:
                    for unresolved_role in basic_aliases:
                        if unresolved_role == "sequence":
                            continue
                        if (
                            record.get(unresolved_role)
                            or unresolved_role in record.get("_unresolved_fields", [])
                            or unresolved_role in record.get("_source_absent_fields", [])
                        ):
                            continue
                        _append_internal_field(record, "_unresolved_fields", unresolved_role)
                        record["extraction_status"] = "review"
                        _report_required_row_failure(
                            parse_result,
                            issue_code="candidate_b_employment_recovered_header_cell_unresolved",
                            dataset="employment_records",
                            sequence=sequence,
                            field_name=unresolved_role,
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                            target_record_id=str(record["employment_record_id"]),
                        )
                required_roles = ("employer",) if mode == "basic" else (
                    "occupation",
                    "information_updated_date",
                )
                for required_role in required_roles:
                    if (
                        record.get(required_role)
                        or required_role in record.get("_unresolved_fields", [])
                        or required_role in record.get("_source_absent_fields", [])
                    ):
                        continue
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_employment_required_cell_unresolved",
                        dataset="employment_records",
                        sequence=sequence,
                        field_name=required_role,
                        row=row,
                        page=page,
                        table=table,
                        row_index=row_index,
                        target_record_id=str(record["employment_record_id"]),
                    )
            previous_table_id = table_id or previous_table_id
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    _enforce_employment_record_contracts(parse_result, records.values())
    _prune_resolved_employment_field_issues(parse_result, records.values())
    for sequence, record in records.items():
        observed_components = set(record.get("_observed_components") or ())
        missing_components = [
            component
            for component in ("basic", "detail", "provider")
            if component not in observed_components
        ]
        if not missing_components:
            continue
        record["extraction_status"] = "review"
        record_issue(
            parse_result,
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_employment_component_missing",
                message="A canonical employment row is missing one or more schema components.",
                parser_stage="candidate_b_employment_canonical_slots",
                target_dataset="employment_records",
                target_record_id=str(record["employment_record_id"]),
                field_name="employment_components",
                observed_value={
                    "sequence": sequence,
                    "observed_components": sorted(observed_components),
                },
                candidate_value={"missing_components": missing_components},
                source_refs=tuple(record.get("source_refs") or ()),
                reason_codes=(
                    "canonical_employment_row_observed",
                    "component_population_incomplete",
                    "row_marked_for_review",
                ),
            ),
        )
    output = [records[key] for key in sorted(records)]
    _enforce_observed_header_terminal_invariant(
        parse_result,
        dataset="employment_records",
        aliases=basic_aliases,
        records=output,
    )
    _enforce_observed_header_terminal_invariant(
        parse_result,
        dataset="employment_records",
        aliases=detail_aliases,
        records=output,
    )
    return output


def _credible_sequence_endpoint(values: set[int]) -> tuple[int | None, list[int]]:
    """Return a dense sequence endpoint and isolate implausible OCR outliers."""
    if not values:
        return None, []
    # Printed ordinals are dense within a PBOC account family. Permit a small
    # number of missing observations, but never let one account identifier or
    # OCR-joined number inflate the expected row count by dozens of records.
    ceiling = max(3, len(values) + max(2, len(values) // 4))
    credible = {value for value in values if 1 <= value <= ceiling}
    # Three exact consecutive ordinals at the observed high edge are an
    # independent printed tail, not one joined identifier.  Accept the whole
    # tail while retaining the conservative treatment of isolated highs such
    # as ``{1, 3, 115}``.
    high_values = sorted(
        value for value in values if value > ceiling and value > 0
    )
    consecutive_tail: list[int] = []
    if high_values:
        consecutive_tail = [high_values[-1]]
        for value in reversed(high_values[:-1]):
            if value != consecutive_tail[0] - 1:
                break
            consecutive_tail.insert(0, value)
    if len(consecutive_tail) >= 3:
        credible.update(consecutive_tail)
    outliers = sorted(values - credible)
    return (max(credible) if credible else None), outliers


def _inquiry_sequence_endpoint(
    values: set[int],
    rejected_values: Iterable[int] = (),
) -> tuple[int | None, list[int]]:
    """Return the endpoint after ordered row proof identifies any rejects.

    Unlike account-family discovery, a canonical inquiry sequence is already
    bound to the exact leftmost header column, so a sparse high ordinal is
    valid evidence of preceding rows.  Prefix removal is intentionally not
    attempted from this unordered set; callers must supply values rejected by
    :func:`_document_local_inquiry_ordinals` using exact adjacent-row proof.
    """

    retained = {value for value in values if 1 <= value <= 9999}
    rejected = sorted(
        {
            int(value)
            for value in rejected_values
            if isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= 9999
        }
    )
    retained.difference_update(rejected)
    return (max(retained) if retained else None), rejected


def _inquiry_source_coverage(parse_result: Any) -> dict[str, Any]:
    """Inventory printed inquiry ordinals before any business row is emitted.

    Inquiry rows are closed-world canonical records: institutional and personal
    inquiry ordinals each restart at one.  Completeness therefore comes from
    the printed ordinal populations, not from the subset of rows whose date,
    institution, and reason all happened to decode.  Headerless physical-page
    continuations remain attached to the last canonical four-column table.
    """

    aliases = {
        "sequence": ("编号", "序号"),
        "inquiry_date": ("查询日期",),
        "institution": ("查询机构",),
        "reason": ("查询原因",),
    }
    groups: list[dict[str, Any]] = []
    active_group: dict[str, Any] | None = None
    active_slots: dict[str, int] = {}
    active_group_page: int | None = None

    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        canonical_page = (
            str(getattr(page, "canonical_template_id", "") or "")
            == "annotations_and_inquiries"
        )
        page_had_inquiry_table = False
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            header_text = _compact("".join(_nonempty(rows[0])))
            has_header = all(
                marker in header_text
                for marker in ("查询日期", "查询机构", "查询原因")
            )
            header_slots = _canonical_header_slots(
                tuple(str(value or "") for value in rows[0]), aliases
            )
            has_exact_header = has_header and set(header_slots) == set(aliases)
            repaired_header_slots = (
                _bounded_collapsed_inquiry_header_slots(rows)
                if has_header and not has_exact_header
                else None
            )
            first_value = _clean(rows[0][0] if rows[0] else "")
            looks_like_continuation = bool(
                not has_header
                and canonical_page
                and active_group is not None
                and "sequence" in active_slots
                and re.fullmatch(r"\D*\d{1,4}\D*", first_value)
            )
            carry_allowed = _inquiry_schema_carry_allowed(
                parse_result,
                active_group_page,
                page_number,
            )
            is_continuation = looks_like_continuation and carry_allowed
            if looks_like_continuation and not carry_allowed:
                _record_inquiry_schema_carry_unresolved(
                    parse_result,
                    left_logical_page=active_group_page,
                    page=page,
                    table=table,
                )
                active_group = None
                active_slots = {}
                active_group_page = None
                continue
            if has_header:
                # Even a damaged/merged header is useful as a section boundary.
                # A body-proven collapsed header may recover all four fixed
                # slots; otherwise only exact partial bindings plus the printed
                # leftmost ordinal contribute to this independent inventory.
                active_slots = dict(repaired_header_slots or header_slots)
                active_slots.setdefault("sequence", 0)
                active_group = {
                    "logical_page": page_number,
                    "last_logical_page": page_number,
                    "observations": [],
                    "source_refs": [],
                }
                groups.append(active_group)
                active_group_page = page_number
                start = 1
            elif is_continuation:
                active_group_page = page_number
                active_group["last_logical_page"] = page_number
                start = 0
            else:
                continue

            page_had_inquiry_table = True
            table_ref = _source_ref(page, table)
            active_group["source_refs"].append(table_ref)
            for row_index, row in enumerate(rows[start:], start=start):
                cells = tuple(str(value or "").strip() for value in row)
                if not _nonempty(cells):
                    continue
                raw_sequence = _inquiry_sequence_token(
                    _slot_value(cells, active_slots, "sequence")
                )
                institution = _slot_value(cells, active_slots, "institution")
                reason = _slot_value(cells, active_slots, "reason")
                inquiry_date = _slot_value(cells, active_slots, "inquiry_date")
                if raw_sequence is None and not any(
                    value for value in (inquiry_date, institution, reason)
                ):
                    continue
                compact_institution = _compact(
                    _normalized_inquiry_field("institution", institution)
                )
                compact_reason = _compact(_normalized_inquiry_field("reason", reason))
                inquiry_type: str | None
                if compact_institution == "本人" or compact_reason.startswith("本人查询"):
                    inquiry_type = "personal"
                elif compact_institution or compact_reason:
                    inquiry_type = "institution"
                else:
                    inquiry_type = None
                active_group["observations"].append(
                    {
                        "raw_sequence": raw_sequence,
                        "sequence": raw_sequence,
                        "inquiry_type": inquiry_type,
                        "inquiry_date": inquiry_date,
                        "institution": institution,
                        "reason": reason,
                        "source_ref": _source_ref(page, table, row=row_index),
                    }
                )

        if not canonical_page and not page_had_inquiry_table:
            active_group = None
            active_slots = {}
            active_group_page = None

    for group in groups:
        observations = group["observations"]
        normalized_ordinals = _document_local_inquiry_ordinals(
            observation.get("raw_sequence") for observation in observations
        )
        for observation, (sequence, repair_kind) in zip(
            observations, normalized_ordinals, strict=True
        ):
            observation["sequence"] = sequence
            observation["sequence_repair_kind"] = repair_kind

    # A header can repeat at a physical-page boundary.  If all readable rows in
    # that group have one type, their unreadable neighbours share that canonical
    # population.  A headerless group beginning above one likewise continues
    # the nearest preceding typed population.  Mixed groups stay unclassified.
    group_types: list[str | None] = []
    for group in groups:
        types = {
            str(observation.get("inquiry_type"))
            for observation in group["observations"]
            if observation.get("inquiry_type")
        }
        group_types.append(next(iter(types)) if len(types) == 1 else None)
    for index, group in enumerate(groups):
        if group_types[index] is not None:
            continue
        sequences = {
            int(observation["sequence"])
            for observation in group["observations"]
            if int(observation.get("sequence") or 0) > 0
        }
        if sequences and min(sequences) > 1:
            preceding_index = next(
                (
                    cursor
                    for cursor in range(index - 1, -1, -1)
                    if group_types[cursor]
                ),
                None,
            )
            if preceding_index is not None and _inquiry_schema_carry_allowed(
                parse_result,
                int(
                    groups[preceding_index].get("last_logical_page")
                    or groups[preceding_index].get("logical_page")
                    or 0
                ),
                int(group.get("logical_page") or 0),
            ):
                group_types[index] = group_types[preceding_index]

    sequences_by_type: defaultdict[str, set[int]] = defaultdict(set)
    rejected_by_type: defaultdict[str, set[int]] = defaultdict(set)
    unclassified_endpoints: list[int] = []
    all_refs: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group_type = group_types[index]
        known_types = {
            str(observation.get("inquiry_type"))
            for observation in group["observations"]
            if observation.get("inquiry_type")
        }
        group_sequences: set[int] = set()
        for observation in group["observations"]:
            explicit_type = observation.get("inquiry_type")
            inquiry_type = group_type if len(known_types) <= 1 else explicit_type
            sequence = int(observation.get("sequence") or 0)
            if inquiry_type and sequence > 0:
                inquiry_type = str(inquiry_type)
                sequences_by_type[inquiry_type].add(sequence)
                if observation.get("sequence_repair_kind") == "prefixed_noise":
                    raw_sequence = observation.get("raw_sequence")
                    if isinstance(raw_sequence, int) and not isinstance(
                        raw_sequence, bool
                    ):
                        rejected_by_type[inquiry_type].add(raw_sequence)
            elif sequence > 0:
                group_sequences.add(sequence)
        if group_sequences:
            endpoint, _outliers = _inquiry_sequence_endpoint(group_sequences)
            if endpoint:
                unclassified_endpoints.append(endpoint)
        all_refs.extend(
            dict(ref) for ref in group.get("source_refs") or () if isinstance(ref, Mapping)
        )

    endpoints: dict[str, int] = {}
    outliers: dict[str, list[int]] = {}
    observed: dict[str, list[int]] = {}
    for inquiry_type, values in sequences_by_type.items():
        endpoint, rejected = _inquiry_sequence_endpoint(
            values, rejected_by_type.get(inquiry_type, set())
        )
        if endpoint:
            endpoints[inquiry_type] = endpoint
        observed[inquiry_type] = sorted(values)
        if rejected:
            outliers[inquiry_type] = rejected

    # Keep issue evidence compact and deterministic: one source table reference
    # is sufficient to select each affected page for the one-shot repair pass.
    unique_refs: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for ref in all_refs:
        marker = json.dumps(ref, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen_refs:
            seen_refs.add(marker)
            unique_refs.append(ref)
    expected = sum(endpoints.values()) + sum(unclassified_endpoints)
    return {
        **({"sequence_endpoints": endpoints} if endpoints else {}),
        **({"observed_sequences": observed} if observed else {}),
        **({"sequence_outliers": outliers} if outliers else {}),
        **({"unclassified_sequence_endpoints": unclassified_endpoints} if unclassified_endpoints else {}),
        **({"expected_row_count": expected} if expected else {}),
        **({"source_refs": unique_refs} if unique_refs else {}),
    }


def _source_completeness_ledger(parse_result: Any) -> dict[str, Any]:
    """Count printed sequence evidence independently of emitted records."""
    sequences: dict[str, set[int]] = {
        "residence_records": set(),
        "employment_records": set(),
    }
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows for cell in row))
            targets: list[str] = []
            if any(marker in compact for marker in ("居住地址", "住宅电话", "居住状况")):
                targets.append("residence_records")
            if any(marker in compact for marker in ("工作单位", "单位性质", "进入本单位年份")) or all(
                marker in compact for marker in ("职业", "行业", "职务", "职称")
            ):
                targets.append("employment_records")
            for row in rows:
                first = _clean(row[0] if row else "")
                numbers = [int(value) for value in re.findall(r"\d{1,3}", first)]
                sequence = numbers[0] if numbers and len(set(numbers)) == 1 else None
                if sequence is not None and 1 <= sequence <= 20:
                    for target in targets:
                        sequences[target].add(sequence)

    # Account numbers restart for each PBOC account family, so count the
    # maximum printed endpoint within each observed family instead of taking
    # one document-wide maximum.  Some exact families (notably R2) print an
    # account anchor without an ordinal.  Keep those anchors in a separate,
    # source-bound inventory: adding ``sum(endpoints)`` alone silently drops
    # the whole singleton family.
    family = ""
    family_quality = ""
    family_logical_page = 0
    family_last_ordinal = 0
    family_sequences: defaultdict[str, set[int]] = defaultdict(set)
    family_numbered_identifiers: defaultdict[str, set[str]] = defaultdict(set)
    family_unnumbered_identifiers: defaultdict[str, set[str]] = defaultdict(set)
    family_weak_unnumbered_anchors: defaultdict[str, set[str]] = defaultdict(set)
    loader = getattr(parse_result, "corrected_evidence_pages", None)
    evidence_pages = list(loader()) if callable(loader) else []
    cross_page_order_resolved = _account_reading_order_resolution(
        parse_result, evidence_pages
    )[-1]
    pages = _account_ordered_pages(parse_result, evidence_pages)
    for page_offset, page in enumerate(pages):
        logical_page = int(page.get("page") or 0)
        local_table_family, shared_revolving_carry = _account_page_table_evidence(
            parse_result,
            logical_page,
        )
        family_carry_allowed = bool(
            logical_page > 0
            and family_logical_page > 0
            and (
                logical_page == family_logical_page
                or (
                    cross_page_order_resolved
                    and _registered_account_pages_are_adjacent(
                        parse_result,
                        family_logical_page,
                        logical_page,
                    )
                )
            )
        )
        if page_offset > 0 and family and not family_carry_allowed:
            family = ""
            family_quality = ""
            family_logical_page = 0
            family_last_ordinal = 0
        preserve_revolving_family = _bounded_revolving_family_carry_over_generic_table(
            parse_result,
            page=page,
            active_family=family,
            active_family_quality=family_quality,
            active_family_logical_page=family_logical_page,
            active_family_last_ordinal=family_last_ordinal,
            local_table_family=local_table_family,
            cross_page_order_resolved=cross_page_order_resolved,
        )
        if local_table_family is not None and not preserve_revolving_family:
            family, family_quality = local_table_family
            family_logical_page = logical_page
            family_last_ordinal = 0
        elif preserve_revolving_family:
            family_logical_page = logical_page
        elif family in {
            "revolving_loan_subaccount",
            "revolving_loan_account",
        } and shared_revolving_carry:
            family_logical_page = logical_page
        source_page = int(page.get("source_page") or logical_page)
        for line_index, line in enumerate(page.get("lines") or []):
            if not isinstance(line, Mapping):
                continue
            raw_text = str(line.get("text") or line.get("content") or "")
            text = _compact(raw_text)
            if family and any(marker in text for marker in _ACCOUNT_SECTION_END):
                family = ""
                family_quality = ""
                family_logical_page = 0
                family_last_ordinal = 0
            heading = _account_family_from_heading(text)
            if heading is not None:
                family, family_quality = heading
                family_logical_page = logical_page
                family_last_ordinal = 0
            match = re.match(r"^(?:账户|业务)[（(]?(\d{1,3})(?:[）)]|\D|$)", text)
            if match and family:
                ordinal_value = int(match.group(1))
                family_sequences[family].add(ordinal_value)
                family_last_ordinal = ordinal_value
            anchor = _ACCOUNT_ANCHOR_RE.search(raw_text)
            if family and (match is not None or anchor is not None):
                family_logical_page = logical_page
                if anchor is not None and anchor.group(1):
                    family_last_ordinal = int(anchor.group(1))
            if anchor is None or not family:
                continue
            ordinal = int(anchor.group(1)) if anchor.group(1) else None
            heading_fields = _account_heading_fields(raw_text)
            agreement_identifier = re.sub(
                r"[^0-9A-Z]",
                "",
                str(heading_fields.get("credit_agreement_identifier") or "").upper(),
            )
            if ordinal is not None and ordinal > 0:
                if agreement_identifier:
                    family_numbered_identifiers[family].add(agreement_identifier)
                continue
            # An ambiguous ``循环贷账户`` heading cannot prove whether an
            # unnumbered anchor is R1 or R2.  It may still contribute numbered
            # population evidence above, but never a subtype-specific singleton.
            if family_quality != "exact":
                continue
            if agreement_identifier:
                family_unnumbered_identifiers[family].add(agreement_identifier)
                continue
            # A single exact source anchor is a bounded singleton witness.  Two
            # or more weak anchors are deliberately not counted: overlapping
            # corrected pages could otherwise manufacture extra accounts.
            evidence_ids = tuple(sorted(str(value) for value in line.get("evidence_ids") or ()))
            bbox = tuple(line.get("bbox") or ())
            weak_identity = json.dumps(
                {
                    "source_page": source_page,
                    "logical_page": logical_page,
                    "line_index": line_index,
                    "bbox": bbox,
                    "evidence_ids": evidence_ids,
                    "text": text,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            family_weak_unnumbered_anchors[family].add(weak_identity)

    # Agreement cards are repeated labelled records, and a parser that emits
    # one valid card cannot use that partial success as evidence that the
    # section is complete.  Count both printed ordinals and repeated primary
    # labels directly from canonical page evidence before business repair.
    agreement_sequences: set[int] = set()
    agreement_label_count = 0
    agreement_refs: list[dict[str, Any]] = []
    active_agreements = False
    previous_agreement_page: int | None = None
    agreement_heading = "授信协议信息"
    agreement_label = "授信协议标识"
    agreement_anchor = re.compile(r"授信协议\s*(\d{1,3})")
    agreement_end_markers = (
        "非信贷交易信息明细",
        "公共信息明细",
        "异议标注",
        "查询记录",
        "报告说明",
    )
    for page in pages or []:
        logical_page = int(page.get("page") or 0)
        source_page = int(page.get("source_page") or logical_page)
        agreement_page_carry = bool(
            previous_agreement_page == logical_page
            or (
                hasattr(parse_result, "reading_order_resolution")
                and _document_page_carry_allowed(
                    parse_result,
                    previous_agreement_page,
                    logical_page,
                )
            )
        )
        if (
            active_agreements
            and previous_agreement_page is not None
            and not agreement_page_carry
        ):
            active_agreements = False
        page_observed = False
        for line in page.get("lines") or []:
            if not isinstance(line, Mapping):
                continue
            text = _compact(line.get("text") or line.get("content") or "")
            if agreement_heading in text:
                active_agreements = True
                page_observed = True
            if not active_agreements and agreement_anchor.search(text):
                # A damaged/missing section heading must not hide an explicit
                # agreement ordinal.  The primary label alone is insufficient:
                # it also appears in ordinary account-detail headings.
                active_agreements = True
            if not active_agreements:
                continue
            matches = [int(match.group(1)) for match in agreement_anchor.finditer(text)]
            if matches:
                agreement_sequences.update(value for value in matches if 1 <= value <= 100)
                page_observed = True
            label_occurrences = text.count(agreement_label)
            if label_occurrences:
                agreement_label_count += label_occurrences
                page_observed = True
            if any(marker in text for marker in agreement_end_markers):
                active_agreements = False
        if page_observed and logical_page > 0:
            agreement_refs.append(
                {
                    "source": "candidate_b_source_coverage_ledger",
                    "logical_page": logical_page,
                    "source_page": source_page,
                    "geometry_scope": "logical_page",
                }
            )
        previous_agreement_page = logical_page

    endpoints: dict[str, int] = {}
    sequence_outliers: dict[str, list[int]] = {}
    for account_family, values in family_sequences.items():
        endpoint, outliers = _credible_sequence_endpoint(values)
        if endpoint is not None:
            endpoints[account_family] = endpoint
        if outliers:
            sequence_outliers[account_family] = outliers

    family_source_populations: dict[str, int] = {}
    unnumbered_anchor_counts: dict[str, int] = {}
    account_families = (
        set(endpoints)
        | set(family_unnumbered_identifiers)
        | set(family_weak_unnumbered_anchors)
    )
    for account_family in sorted(account_families):
        endpoint = int(endpoints.get(account_family) or 0)
        # A strong identifier seen on a numbered anchor proves that a repeated
        # unnumbered observation is the same account, not an additional row.
        strong_unnumbered = (
            family_unnumbered_identifiers.get(account_family, set())
            - family_numbered_identifiers.get(account_family, set())
        )
        unnumbered_count = len(strong_unnumbered)
        if endpoint == 0 and unnumbered_count == 0:
            weak_anchors = family_weak_unnumbered_anchors.get(account_family, set())
            if len(weak_anchors) == 1:
                unnumbered_count = 1
        # Mixing numbered and unnumbered observations within one family is not
        # sufficient proof that the latter are additional rows.  Use the dense
        # printed endpoint in that case; exact unnumbered populations are used
        # only for a family with no credible numbered sequence.
        population = endpoint or unnumbered_count
        if population > 0:
            family_source_populations[account_family] = population
        if endpoint == 0 and unnumbered_count > 0:
            unnumbered_anchor_counts[account_family] = unnumbered_count

    agreement_endpoint, agreement_outliers = _credible_sequence_endpoint(agreement_sequences)
    inquiry_coverage = _inquiry_source_coverage(parse_result)
    # Every canonical agreement card has a printed ordinal.  Once at least one
    # credible ordinal is available, its dense endpoint is a stronger
    # population witness than raw primary-label occurrences: corrected logical
    # pages can contain overlapping evidence fragments and therefore repeat the
    # same ``授信协议标识`` label.  Treating those duplicate labels as extra cards
    # creates a false partial-dataset report even when all printed ordinals were
    # recovered.  Label counting remains the fallback for a section whose
    # ordinal headings are completely unreadable.
    agreement_expected = agreement_endpoint or agreement_label_count
    source_refs: dict[str, list[dict[str, Any]]] = {}
    if agreement_refs:
        source_refs["credit_lines"] = agreement_refs
    if inquiry_coverage.get("source_refs"):
        source_refs["inquiry_records"] = list(inquiry_coverage["source_refs"])
    return {
        "sequence_endpoints": {
            name: max(values)
            for name, values in sequences.items()
            if values
        },
        **(
            {"credit_accounts": sum(family_source_populations.values())}
            if family_source_populations
            else {}
        ),
        **({"credit_agreements": agreement_expected} if agreement_expected else {}),
        **({"credit_agreement_sequence_endpoint": agreement_endpoint} if agreement_endpoint else {}),
        **({"credit_agreement_sequence_outliers": agreement_outliers} if agreement_outliers else {}),
        **(
            {"inquiry_records": int(inquiry_coverage["expected_row_count"])}
            if inquiry_coverage.get("expected_row_count")
            else {}
        ),
        **(
            {"inquiry_sequence_endpoints": dict(inquiry_coverage["sequence_endpoints"])}
            if inquiry_coverage.get("sequence_endpoints")
            else {}
        ),
        **(
            {"inquiry_observed_sequences": dict(inquiry_coverage["observed_sequences"])}
            if inquiry_coverage.get("observed_sequences")
            else {}
        ),
        **(
            {"inquiry_sequence_outliers": dict(inquiry_coverage["sequence_outliers"])}
            if inquiry_coverage.get("sequence_outliers")
            else {}
        ),
        **(
            {
                "inquiry_unclassified_sequence_endpoints": list(
                    inquiry_coverage["unclassified_sequence_endpoints"]
                )
            }
            if inquiry_coverage.get("unclassified_sequence_endpoints")
            else {}
        ),
        **({"account_family_endpoints": dict(endpoints)} if endpoints else {}),
        **(
            {"account_family_source_populations": family_source_populations}
            if family_source_populations
            else {}
        ),
        **(
            {"account_family_unnumbered_anchor_counts": unnumbered_anchor_counts}
            if unnumbered_anchor_counts
            else {}
        ),
        **({"account_family_sequence_outliers": sequence_outliers} if sequence_outliers else {}),
        **({"source_refs": source_refs} if source_refs else {}),
    }


def _record_pre_repair_source_gaps(
    parse_result: Any,
    datasets: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish source-population gaps early enough to select repair pages.

    The final completeness pass remains authoritative.  This early pass exists
    solely so a record that was never instantiated can still trigger the one
    allowed page repair instead of disappearing silently.
    """
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
    )

    ledger = _source_completeness_ledger(parse_result)
    source_refs = ledger.get("source_refs") if isinstance(ledger.get("source_refs"), Mapping) else {}
    checks = (
        ("credit_lines", "credit_agreements"),
        ("inquiry_records", "inquiry_records"),
    )
    for dataset_name, ledger_name in checks:
        expected = ledger.get(ledger_name)
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            continue
        observed = sum(isinstance(row, Mapping) for row in datasets.get(dataset_name) or ())
        if observed >= expected:
            continue
        record_issue(
            parse_result,
            make_issue(
                category="page_continuation",
                issue_code="source_sequence_or_count_gap",
                message=(
                    "Independent canonical section evidence contains more records than the first schema pass; "
                    "the affected pages must be repaired before completeness is decided."
                ),
                parser_stage="candidate_b_pre_repair_source_coverage",
                target_dataset=dataset_name,
                observed_value={"observed_row_count": observed},
                candidate_value={
                    "source_expected_row_count": expected,
                    **(
                        {
                            "source_sequence_endpoints": ledger.get(
                                "inquiry_sequence_endpoints", {}
                            ),
                            "unclassified_sequence_endpoints": ledger.get(
                                "inquiry_unclassified_sequence_endpoints", []
                            ),
                        }
                        if dataset_name == "inquiry_records"
                        else {}
                    ),
                },
                source_refs=(source_refs.get(dataset_name) or ()),
                reason_codes=(
                    "independent_source_ledger",
                    "missing_business_records",
                    "schema_triggered_page_repair_eligible",
                    "no_missing_row_invented",
                ),
            ),
        )
    return ledger


def _collapsed_mobile_observation(row: tuple[str, ...]) -> dict[str, Any] | None:
    """Decode a canonical mobile row whose labels and values share cells.

    Some OCR table reconstructions preserve the three printed physical cells
    but flatten the header and first (or only) data row into those cells.  This
    path is intentionally narrow: both the phone and its printed ordinal must
    be unique before a business row can be keyed.
    """

    compact_row = _compact("".join(row))
    if not all(
        marker in compact_row
        for marker in ("编号", "手机号码", "信息更新日期", "数据发生机构名称")
    ):
        return None
    sequence_columns = [index for index, cell in enumerate(row) if "编号" in _compact(cell)]
    phone_columns = [index for index, cell in enumerate(row) if "手机号码" in _compact(cell)]
    if (
        len(sequence_columns) != 1
        or len(phone_columns) != 1
        or sequence_columns[0] != phone_columns[0]
    ):
        return None
    shared_column = sequence_columns[0]
    shared_text = _clean(row[shared_column])
    phone_matches = re.findall(r"(?<!\d)1[3-9]\d{9}(?!\d)", shared_text)
    if len(set(phone_matches)) != 1:
        return None
    phone = phone_matches[0]
    sequence_residue = shared_text.replace(phone, " ")
    sequence_residue = re.sub(r"手机号码|编号", " ", sequence_residue)
    sequences = {
        int(match)
        for match in re.findall(r"(?<!\d)\d{1,3}(?!\d)", sequence_residue)
        if 1 <= int(match) <= 999
    }
    if len(sequences) != 1:
        return None

    updated_columns = [
        index for index, cell in enumerate(row) if "信息更新日期" in _compact(cell)
    ]
    provider_columns = [
        index for index, cell in enumerate(row) if "数据发生机构名称" in _compact(cell)
    ]
    updated_raw = ""
    updated_column = updated_columns[0] if len(updated_columns) == 1 else shared_column
    if len(updated_columns) == 1:
        matches = list(_DATE_RE.finditer(str(row[updated_column] or "")))
        if len(matches) == 1:
            updated_raw = matches[0].group(0)
    provider = ""
    provider_column = provider_columns[-1] if provider_columns else shared_column
    if provider_columns:
        provider_text = str(row[provider_column] or "")
        provider = _clean(provider_text.rsplit("数据发生机构名称", 1)[-1])
    return {
        "sequence": next(iter(sequences)),
        "mobile_phone": phone,
        "information_updated_date": updated_raw,
        "data_provider": provider,
        "columns": {
            "sequence": shared_column,
            "mobile_phone": shared_column,
            "information_updated_date": updated_column,
            "data_provider": provider_column,
        },
    }


def _materialize_mobile_record(
    parse_result: Any,
    records: dict[int, dict[str, Any]],
    *,
    page: Any,
    table: Any,
    row: tuple[str, ...],
    row_index: int,
    sequence: int,
    values: Mapping[str, str],
    columns: Mapping[str, int],
) -> None:
    mobile_id = stable_record_id("personal_mobile_phone", sequence)
    record = records.setdefault(
        sequence,
        {
            "record_id": mobile_id,
            "mobile_phone_record_id": mobile_id,
            "sequence": sequence,
            "source": "native_personal_detail_profile_table",
            "source_refs": [],
            "confidence": 1.0,
        },
    )
    record["source_refs"].append(_source_ref(page, table, row=row_index))
    raw_phone = _clean(values.get("mobile_phone"))
    phone = re.sub(r"\D", "", raw_phone)
    phone_column = int(columns.get("mobile_phone", 0) or 0)
    phone_ref = _source_ref(page, table, row=row_index, column=phone_column)
    if not raw_phone:
        _report_required_row_failure(
            parse_result,
            issue_code="candidate_b_mobile_row_unresolved",
            dataset="mobile_phone_records",
            sequence=sequence,
            field_name="mobile_phone",
            row=row,
            page=page,
            table=table,
            row_index=row_index,
            target_record_id=mobile_id,
        )
    elif raw_phone == "--":
        _mark_source_absent(record, "mobile_phone", raw_phone)
    elif any(character.isalpha() for character in raw_phone) or not re.fullmatch(
        r"1[3-9]\d{9}", phone
    ):
        _reject_exact_observation(
            parse_result,
            record,
            dataset="mobile_phone_records",
            target_record_id=mobile_id,
            field_name="mobile_phone",
            raw=raw_phone,
            source_ref=phone_ref,
            parser_stage="candidate_b_mobile_canonical_slots",
        )
    else:
        _merge_exact_observation(
            parse_result,
            record,
            dataset="mobile_phone_records",
            target_record_id=mobile_id,
            field_name="mobile_phone",
            value=phone,
            raw=raw_phone,
            source_ref=phone_ref,
            parser_stage="candidate_b_mobile_canonical_slots",
        )

    raw_updated = _clean(values.get("information_updated_date"))
    updated_column = int(columns.get("information_updated_date", 0) or 0)
    updated_ref = _source_ref(page, table, row=row_index, column=updated_column)
    if raw_updated and not is_explicit_source_absence(raw_updated):
        _merge_canonical_date_observation(
            parse_result,
            record,
            dataset="mobile_phone_records",
            target_record_id=mobile_id,
            field_name="information_updated_date",
            raw=raw_updated,
            source_ref=updated_ref,
            parser_stage="candidate_b_mobile_canonical_slots",
        )
    elif is_explicit_source_absence(raw_updated):
        _mark_source_absent(record, "information_updated_date", raw_updated)
    else:
        _report_required_row_failure(
            parse_result,
            issue_code="candidate_b_mobile_row_unresolved",
            dataset="mobile_phone_records",
            sequence=sequence,
            field_name="information_updated_date",
            row=row,
            page=page,
            table=table,
            row_index=row_index,
            target_record_id=mobile_id,
        )

    provider = _clean(values.get("data_provider"))
    provider_column = int(columns.get("data_provider", 0) or 0)
    if provider == "--":
        _mark_source_absent(record, "data_provider", provider)
    elif provider:
        _merge_exact_observation(
            parse_result,
            record,
            dataset="mobile_phone_records",
            target_record_id=mobile_id,
            field_name="data_provider",
            value=provider,
            raw=provider,
            source_ref=_source_ref(page, table, row=row_index, column=provider_column),
            parser_stage="candidate_b_mobile_canonical_slots",
        )
    else:
        _report_required_row_failure(
            parse_result,
            issue_code="candidate_b_mobile_row_unresolved",
            dataset="mobile_phone_records",
            sequence=sequence,
            field_name="data_provider",
            row=row,
            page=page,
            table=table,
            row_index=row_index,
            target_record_id=mobile_id,
        )


def _extract_profile_detail_records(parse_result: Any) -> dict[str, list[dict[str, Any]]]:
    mobile_aliases = {
        "sequence": ("编号",),
        "mobile_phone": ("手机号码",),
        "information_updated_date": ("信息更新日期",),
        "data_provider": ("数据发生机构名称",),
    }
    spouse_aliases = {
        "name": ("姓名",),
        "document_type": ("证件类型",),
        "document_number": ("证件号码",),
        "employer": ("工作单位",),
        "phone": ("联系电话",),
    }
    mobile_records: dict[int, dict[str, Any]] = {}
    spouse_record: dict[str, Any] | None = None
    active_slots: dict[str, int] = {}
    mode = ""
    spouse_observed = False
    previous_table_id = ""
    continuation_check = getattr(parse_result, "tables_continue", None)
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            table_id = str(getattr(table, "table_id", "") or "")
            continues = bool(
                previous_table_id
                and table_id
                and callable(continuation_check)
                and continuation_check(previous_table_id, table_id) is True
            )
            if not continues:
                active_slots = {}
                mode = ""
                spouse_observed = False
            for row_index, row in enumerate(rows):
                mobile_slots = _canonical_header_slots(row, mobile_aliases)
                spouse_slots = _canonical_header_slots(row, spouse_aliases)
                compact_header = _compact("".join(row))
                collapsed_mobile = (
                    _collapsed_mobile_observation(row)
                    if not set(mobile_aliases) <= set(mobile_slots)
                    else None
                )
                if collapsed_mobile is not None:
                    _materialize_mobile_record(
                        parse_result,
                        mobile_records,
                        page=page,
                        table=table,
                        row=row,
                        row_index=row_index,
                        sequence=int(collapsed_mobile["sequence"]),
                        values={
                            role: str(collapsed_mobile.get(role) or "")
                            for role in ("mobile_phone", "information_updated_date", "data_provider")
                        },
                        columns=collapsed_mobile["columns"],
                    )
                    active_slots = {}
                    mode = ""
                    continue
                broken_mobile = all(
                    marker in compact_header
                    for marker in ("编号", "手机号码", "信息更新日期", "数据发生机构名称")
                ) and not set(mobile_aliases) <= set(mobile_slots)
                broken_mobile = broken_mobile or (
                    len(mobile_slots) >= 2
                    and "mobile_phone" in mobile_slots
                    and not set(mobile_aliases) <= set(mobile_slots)
                )
                broken_spouse = all(
                    marker in compact_header for marker in ("姓名", "证件类型", "证件号码", "工作单位", "联系电话")
                ) and not set(spouse_aliases) <= set(spouse_slots)
                broken_spouse = broken_spouse or (
                    len(spouse_slots) >= 2
                    and bool({"employer", "phone"} & set(spouse_slots))
                    and not set(spouse_aliases) <= set(spouse_slots)
                )
                if broken_mobile or broken_spouse:
                    _report_header_graph_failure(
                        parse_result,
                        dataset="mobile_phone_records" if broken_mobile else "spouse_records",
                        page=page,
                        table=table,
                        row_index=row_index,
                        row=row,
                    )
                    active_slots = {}
                    mode = ""
                    continue
                if set(mobile_aliases) <= set(mobile_slots):
                    active_slots = mobile_slots
                    mode = "mobile"
                    continue
                if set(spouse_aliases) <= set(spouse_slots):
                    active_slots = spouse_slots
                    mode = "spouse"
                    continue
                if "数据发生机构名称" in _compact("".join(row)) and spouse_observed:
                    active_slots = _canonical_header_slots(
                        row, {"sequence": ("编号",), "data_provider": ("数据发生机构名称",)}
                    )
                    mode = "spouse_provider"
                    continue
                if not active_slots:
                    continue
                if mode == "mobile":
                    sequence = _sequence_value(row, active_slots)
                    if sequence is None:
                        if _nonempty(row):
                            _report_unkeyed_business_row(
                                parse_result,
                                dataset="mobile_phone_records",
                                row=row,
                                page=page,
                                table=table,
                                row_index=row_index,
                            )
                        continue
                    _materialize_mobile_record(
                        parse_result,
                        mobile_records,
                        page=page,
                        table=table,
                        row=row,
                        row_index=row_index,
                        sequence=sequence,
                        values={
                            role: _slot_value(row, active_slots, role)
                            for role in ("mobile_phone", "information_updated_date", "data_provider")
                        },
                        columns=active_slots,
                    )
                elif mode == "spouse":
                    values = {
                        role: _slot_value(row, active_slots, role)
                        for role in spouse_aliases
                    }
                    row_is_source_absent = bool(values) and all(
                        not value or is_explicit_source_absence(value)
                        for value in values.values()
                    )
                    if not any(value for value in values.values()) or row_is_source_absent:
                        spouse_observed = row_is_source_absent
                        continue
                    spouse_id = stable_record_id("personal_spouse", 1)
                    if spouse_record is None:
                        spouse_record = {
                            "record_id": spouse_id,
                            "spouse_record_id": spouse_id,
                            "source": "native_personal_detail_profile_table",
                            "source_refs": [],
                            "confidence": 1.0,
                        }
                    spouse_record["source_refs"].append(_source_ref(page, table, row=row_index))
                    spouse_observed = True
                    for role, raw in values.items():
                        if is_explicit_source_absence(raw):
                            _mark_source_absent(spouse_record, role, raw)
                            continue
                        if not raw:
                            continue
                        ref = _source_ref(page, table, row=row_index, column=active_slots[role])
                        valid = True
                        normalized = raw
                        if role == "phone":
                            digits = re.sub(r"\D", "", raw)
                            valid = not any(character.isalpha() for character in raw) and 5 <= len(digits) <= 16
                        elif role == "name":
                            valid = bool(re.search(r"[\u3400-\u9fffA-Za-z]", raw)) and _compact(raw) not in _ACCOUNT_LABELS
                        elif role == "document_number":
                            compact_number = re.sub(r"\s+", "", raw)
                            valid = bool(re.fullmatch(r"[A-Za-z0-9()（）-]{4,40}", compact_number))
                            normalized = compact_number
                        if valid:
                            _merge_exact_observation(
                                parse_result,
                                spouse_record,
                                dataset="spouse_records",
                                target_record_id=spouse_id,
                                field_name=role,
                                value=normalized,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_spouse_canonical_slots",
                            )
                        else:
                            _reject_exact_observation(
                                parse_result,
                                spouse_record,
                                dataset="spouse_records",
                                target_record_id=spouse_id,
                                field_name=role,
                                raw=raw,
                                source_ref=ref,
                                parser_stage="candidate_b_spouse_canonical_slots",
                            )
                    if (
                        not spouse_record.get("name")
                        and "name" not in set(spouse_record.get("_source_absent_fields") or ())
                    ):
                        _report_required_row_failure(
                            parse_result,
                            issue_code="candidate_b_spouse_row_unresolved",
                            dataset="spouse_records",
                            sequence=1,
                            field_name="name",
                            row=row,
                            page=page,
                            table=table,
                            row_index=row_index,
                            target_record_id=spouse_id,
                        )
                elif mode == "spouse_provider" and spouse_record is not None:
                    provider = _slot_value(row, active_slots, "data_provider")
                    if is_explicit_source_absence(provider):
                        _mark_source_absent(spouse_record, "data_provider", provider)
                    elif provider:
                        ref = _source_ref(
                            page, table, row=row_index, column=active_slots["data_provider"]
                        )
                        spouse_record["source_refs"].append(ref)
                        _merge_exact_observation(
                            parse_result,
                            spouse_record,
                            dataset="spouse_records",
                            target_record_id=str(spouse_record["spouse_record_id"]),
                            field_name="data_provider",
                            value=provider,
                            raw=provider,
                            source_ref=ref,
                            parser_stage="candidate_b_spouse_canonical_slots",
                        )
            previous_table_id = table_id or previous_table_id
    mobile_output = [mobile_records[key] for key in sorted(mobile_records)]
    spouse_output = [spouse_record] if spouse_record is not None else []
    _enforce_observed_header_terminal_invariant(
        parse_result,
        dataset="mobile_phone_records",
        aliases=mobile_aliases,
        records=mobile_output,
    )
    _enforce_observed_header_terminal_invariant(
        parse_result,
        dataset="spouse_records",
        aliases=spouse_aliases,
        records=spouse_output,
    )
    return {
        "mobile_phone_records": mobile_output,
        "spouse_records": spouse_output,
    }


def _extract_personal_notes(parse_result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations: list[dict[str, Any]] = []
    statements: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        source_page = int(getattr(page, "source_page_number", 0) or page_number)
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            for row_index, row in enumerate(rows):
                compact = _compact("".join(row))
                if all(marker in compact for marker in ("编号", "标注内容", "添加日期")):
                    slots = _canonical_header_slots(
                        tuple(row),
                        {"sequence": ("编号",), "text": ("标注内容",), "added_date": ("添加日期",)},
                    )
                    if set(slots) != {"sequence", "text", "added_date"}:
                        _report_header_graph_failure(
                            parse_result,
                            dataset="annotation_statements",
                            page=page,
                            table=table,
                            row_index=row_index,
                            row=tuple(row),
                        )
                        continue
                    for data_index, data_row in enumerate(rows[row_index + 1 :], start=row_index + 1):
                        physical = tuple(data_row)
                        sequence = _sequence_value(physical, slots)
                        if sequence is None:
                            if _slot_value(physical, slots, "sequence"):
                                _report_unkeyed_business_row(
                                    parse_result,
                                    dataset="annotation_statements",
                                    row=physical,
                                    page=page,
                                    table=table,
                                    row_index=data_index,
                                )
                            break
                        target_id = stable_record_id("personal_detail_annotation", sequence)
                        record: dict[str, Any] = {
                            "id": target_id,
                            "annotation_id": target_id,
                            "sequence": sequence,
                            "note_type": "report_annotation",
                            "logical_page": page_number,
                            "source_page": source_page,
                            "source": "native_personal_detail_note_table",
                            "source_refs": [_source_ref(page, table, row=data_index)],
                            "confidence": float(getattr(table, "confidence", None) or 0.9),
                        }
                        for role, field_name, kind in (
                            ("text", "text", "text"),
                            ("added_date", "added_date", "date"),
                        ):
                            raw = _slot_value(physical, slots, role)
                            ref = _source_ref(page, table, row=data_index, column=slots[role])
                            value = _date(raw) if kind == "date" else _clean(raw)
                            if value in (None, ""):
                                _reject_exact_observation(
                                    parse_result,
                                    record,
                                    dataset="annotation_statements",
                                    target_record_id=f"annotation_statement:{target_id}",
                                    field_name=field_name,
                                    raw=raw,
                                    source_ref=ref,
                                    parser_stage="candidate_b_note_canonical_slots",
                                )
                            else:
                                _merge_exact_observation(
                                    parse_result,
                                    record,
                                    dataset="annotation_statements",
                                    target_record_id=f"annotation_statement:{target_id}",
                                    field_name=field_name,
                                    value=value,
                                    raw=raw,
                                    source_ref=ref,
                                    parser_stage="candidate_b_note_canonical_slots",
                                )
                        annotations.append(record)
                marker = next(
                    (candidate for candidate in ("异议标注", "本人声明", "机构说明") if candidate in compact),
                    None,
                )
                if marker is None or row_index + 1 >= len(rows):
                    continue
                # A standalone canonical heading is followed by one narrative
                # cell (and, for annotation forms, optionally one date cell).
                # Multiple free-floating cells are not concatenated into an
                # unstructured text blob.
                if any(label in compact for label in ("编号", "标注内容", "添加日期")):
                    continue
                next_compact = _compact("".join(rows[row_index + 1]))
                if all(label in next_compact for label in ("编号", "标注内容", "添加日期")):
                    continue
                values = [
                    (column, _clean(value))
                    for column, value in enumerate(rows[row_index + 1])
                    if _clean(value)
                ]
                if not values:
                    continue
                target = annotations if marker == "异议标注" else statements
                target_id = stable_record_id("personal_detail_note", marker, page_number, row_index)
                record = {
                    "id": target_id,
                    (
                        "annotation_id"
                        if marker == "异议标注"
                        else "statement_id"
                    ): target_id,
                    "note_type": {
                        "异议标注": "dispute_annotation",
                        "本人声明": "subject_statement",
                        "机构说明": "institution_statement",
                    }[marker],
                    "logical_page": page_number,
                    "source_page": source_page,
                    "source": "native_personal_detail_note_table",
                    "source_refs": [_source_ref(page, table, row=row_index + 1)],
                    "confidence": float(getattr(table, "confidence", None) or 0.9),
                }
                text_column, text = values[0]
                if len(values) > 2 or (len(values) == 2 and _date(values[1][1]) is None):
                    _report_required_row_failure(
                        parse_result,
                        issue_code="candidate_b_note_narrative_cells_unresolved",
                        dataset="annotation_statements",
                        sequence=len(target) + 1,
                        field_name="text",
                        row=tuple(rows[row_index + 1]),
                        page=page,
                        table=table,
                        row_index=row_index + 1,
                        target_record_id=f"annotation_statement:{target_id}",
                    )
                    _append_internal_field(record, "_unresolved_fields", "text")
                else:
                    _merge_exact_observation(
                        parse_result,
                        record,
                        dataset="annotation_statements",
                        target_record_id=f"annotation_statement:{target_id}",
                        field_name="text",
                        value=text,
                        raw=text,
                        source_ref=_source_ref(page, table, row=row_index + 1, column=text_column),
                        parser_stage="candidate_b_note_narrative_slot",
                    )
                    if len(values) == 2:
                        date_column, date_raw = values[1]
                        record["added_date"] = _date(date_raw)
                        record.setdefault("source_refs_by_field", {})["added_date"] = [
                            _source_ref(page, table, row=row_index + 1, column=date_column)
                        ]
                target.append(record)
    return annotations, statements


def _extract_source_rows(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        page_number = int(getattr(page, "page_number", 0) or 0)
        for table in getattr(page, "tables", None) or []:
            table_id = str(getattr(table, "table_id", "") or "")
            for row_index, row in enumerate(_table_rows(table)):
                cells = [_clean(cell) for cell in row]
                records.append(
                    {
                        "source_table_row_id": stable_record_id(
                            "personal_detail_source_row", page_number, table_id, row_index
                        ),
                        "page": page_number,
                        "table_id": table_id,
                        "row_index": row_index,
                        "cells": cells,
                        "nonempty_cells": [cell for cell in cells if cell],
                        "source": "native_detail_source_table",
                        "source_refs": [_source_ref(page, table, row=row_index)],
                        "confidence": 1.0,
                    }
                )
    return records


def _decode_exact_label_card(
    parse_result: Any,
    *,
    page: Any,
    table: Any,
    dataset: str,
    target_record_id: str,
    fields: Mapping[str, tuple[tuple[str, ...], str]],
) -> dict[str, Any]:
    """Decode a canonical label/value card without compacting physical cells."""

    rows = _table_rows(table)
    observations, unresolved = _exact_label_observations(rows)
    record: dict[str, Any] = {"source_refs": [_source_ref(page, table)]}
    for field_name, (labels, kind) in fields.items():
        candidates = [item for label in labels for item in observations.get(_compact(label), ())]
        if not candidates:
            label_location = next(
                ((row, column) for label, row, column in unresolved if label in {_compact(item) for item in labels}),
                None,
            )
            row_index = label_location[0] if label_location is not None else 0
            _report_required_row_failure(
                parse_result,
                issue_code="candidate_b_canonical_card_field_unresolved",
                dataset=dataset,
                sequence=1,
                field_name=field_name,
                row=tuple(rows[row_index]) if row_index < len(rows) else (),
                page=page,
                table=table,
                row_index=row_index,
                target_record_id=target_record_id,
            )
            _append_internal_field(record, "_unresolved_fields", field_name)
            continue
        for raw, row_index, column in candidates:
            ref = _source_ref(page, table, row=row_index, column=column)
            if _compact(raw) in {"-", "--", "---"}:
                _mark_source_absent(record, field_name, raw)
                continue
            if kind == "date":
                value = _date(raw)
            elif kind == "number":
                candidate = _number(raw)
                value = candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
            else:
                value = _clean(raw)
            if value in (None, ""):
                _reject_exact_observation(
                    parse_result,
                    record,
                    dataset=dataset,
                    target_record_id=target_record_id,
                    field_name=field_name,
                    raw=raw,
                    source_ref=ref,
                    parser_stage="candidate_b_canonical_label_card",
                )
                continue
            _merge_exact_observation(
                parse_result,
                record,
                dataset=dataset,
                target_record_id=target_record_id,
                field_name=field_name,
                value=value,
                raw=raw,
                source_ref=ref,
                parser_stage="candidate_b_canonical_label_card",
            )
    return record


_POSTPAID_CARD_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "institution": (("机构名称",), "text"),
    "business_type": (("业务类型",), "text"),
    "service_start_date": (("业务开通日期",), "date"),
    "payment_status": (("当前缴费状态",), "text"),
    "current_arrears_amount": (("当前欠费金额", "当前欠款金额"), "number"),
    "billing_month": (("记账年月",), "date"),
}


def _postpaid_id(page: Any, table: Any, record: Mapping[str, Any]) -> str:
    identity = (record.get("institution"), record.get("business_type"), record.get("billing_month"))
    if all(value not in (None, "") for value in identity):
        return stable_record_id("postpaid", *identity)
    return stable_record_id(
        "postpaid_unresolved",
        int(getattr(page, "page_number", 0) or 0),
        str(getattr(table, "table_id", "") or ""),
    )


def _extract_postpaid_records(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if not rows:
                continue
            compact = _compact(" ".join(cell for row in rows[:2] for cell in row))
            if not all(
                marker in compact
                for marker in ("机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月")
            ):
                continue
            provisional_id = stable_record_id(
                "postpaid_unresolved",
                int(getattr(page, "page_number", 0) or 0),
                str(getattr(table, "table_id", "") or ""),
            )
            record = _decode_exact_label_card(
                parse_result,
                page=page,
                table=table,
                dataset="postpaid_records",
                target_record_id=provisional_id,
                fields=_POSTPAID_CARD_FIELDS,
            )
            postpaid_id = _postpaid_id(page, table, record)
            record.update(
                {
                    "postpaid_record_id": postpaid_id,
                    "sequence": len(records) + 1,
                    "reporting_amount_currency": "CNY",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_postpaid_table",
                    "confidence": float(getattr(table, "confidence", None) or 0.9),
                }
            )
            # Keep issue links valid if the canonical identity became readable.
            if postpaid_id != provisional_id:
                for issue in getattr(parse_result, "_personal_detail_extraction_issues", []) or []:
                    if isinstance(issue, dict) and issue.get("target_record_id") == provisional_id:
                        issue["target_record_id"] = postpaid_id
            records.append(record)
    return records


def _extract_postpaid_payment_history(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:3] for cell in row))
            if not all(marker in compact for marker in ("机构名称", "业务类型", "缴费记录")):
                continue
            observations, _unresolved = _exact_label_observations(rows)
            identity: dict[str, Any] = {}
            for field_name, label in (("institution", "机构名称"), ("business_type", "业务类型"), ("billing_month", "记账年月")):
                candidates = observations.get(label) or []
                distinct = {_compact(value) for value, _row, _column in candidates}
                if len(distinct) == 1:
                    raw = candidates[0][0]
                    identity[field_name] = _date(raw) if field_name == "billing_month" else _clean(raw)
            postpaid_id = _postpaid_id(page, table, identity)
            month_header = next(
                (
                    (row_index, {column: int(_compact(cell)) for column, cell in enumerate(row) if re.fullmatch(r"(?:[1-9]|1[0-2])", _compact(cell))})
                    for row_index, row in enumerate(rows)
                    if sum(bool(re.fullmatch(r"(?:[1-9]|1[0-2])", _compact(cell))) for cell in row) >= 5
                ),
                None,
            )
            if month_header is None:
                _report_header_graph_failure(
                    parse_result,
                    dataset="postpaid_payment_history",
                    page=page,
                    table=table,
                    row_index=0,
                    row=tuple(rows[0]) if rows else (),
                )
                continue
            month_header_index, months_by_column = month_header
            for row_index, row in enumerate(rows):
                if row_index <= month_header_index:
                    continue
                year_cell = next(
                    (
                        (column, _compact(cell))
                        for column, cell in enumerate(row)
                        if re.fullmatch(r"20\d{2}", _compact(cell))
                    ),
                    None,
                )
                if year_cell is None:
                    continue
                year_column, year_raw = year_cell
                for column, month in months_by_column.items():
                    status = _compact(row[column]) if column < len(row) else ""
                    history_id = stable_record_id("postpaid_payment_history", postpaid_id, year_raw, month)
                    record = {
                        "record_id": history_id,
                        "postpaid_payment_history_id": history_id,
                        "postpaid_record_id": postpaid_id,
                        "institution": identity.get("institution"),
                        "business_type": identity.get("business_type"),
                        "year": int(year_raw),
                        "month": month,
                        "source": "native_personal_detail_postpaid_history",
                        "source_refs": [_source_ref(page, table, row=row_index, column=column)],
                        "confidence": float(getattr(table, "confidence", None) or 0.9),
                    }
                    if status in _STATUS_CODES:
                        record["status"] = status
                    else:
                        _reject_exact_observation(
                            parse_result,
                            record,
                            dataset="postpaid_payment_history",
                            target_record_id=history_id,
                            field_name="status",
                            raw=status,
                            source_ref=_source_ref(page, table, row=row_index, column=column),
                            parser_stage="candidate_b_postpaid_month_slots",
                        )
                    records.append(record)
    return records


def _is_summary_anchor(rows: list[list[str]]) -> bool:
    compact = _compact(" ".join(cell for row in rows for cell in row))
    return bool(
        "汇总" in compact or ("账户数" in compact and "首笔业务发放月份" in compact) or "最近1个月内的查询" in compact
    )


_CREDIT_OVERVIEW_BUSINESS_TYPES = (
    "个人商用房贷款（包括商住两用房）",
    "个人商用房贷款(包括商住两用房)",
    "个人住房贷款",
    "其他类贷款",
    "准贷记卡",
    "贷记卡",
)


def _is_headerless_credit_overview_fragment(rows: list[list[str]]) -> bool:
    """Recognize only the four-column tail of the canonical credit overview."""

    if not rows or max((len(row) for row in rows), default=0) != 4:
        return False
    witnessed = 0
    for row in rows:
        category_cell = _compact(row[1] if len(row) > 1 else "")
        matches = [value for value in _CREDIT_OVERVIEW_BUSINESS_TYPES if value in category_cell]
        if "准贷记卡" in matches and "贷记卡" in matches:
            matches.remove("贷记卡")
        if len(matches) != 1:
            continue
        count_cell = _compact(row[2] if len(row) > 2 else "")
        month_cell = _compact(row[3] if len(row) > 3 else "")
        if re.search(r"\d", count_cell) or re.search(r"(?:19|20)\d{2}[.:/-]\d{1,2}", month_cell):
            witnessed += 1
    return witnessed >= 2


def _corrected_text_source_ref(
    page: Mapping[str, Any],
    line: Mapping[str, Any],
    line_index: int,
) -> dict[str, Any]:
    logical_page = int(page.get("page") or 0)
    source_page = int(page.get("source_page") or logical_page)
    ref: dict[str, Any] = {
        "source": "candidate_b_corrected_page_line",
        "logical_page": logical_page,
        "source_page": source_page,
        "line": line_index,
        "geometry_scope": "line",
        "binding": "closed_canonical_summary_text_row",
        "binding_quality": "closed_canonical_summary_text_row",
    }
    bbox = line.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        ref["bbox"] = list(bbox)
    evidence_ids = line.get("evidence_ids")
    if isinstance(evidence_ids, list) and evidence_ids:
        ref["evidence_ids"] = [str(value) for value in evidence_ids if value]
    elif line.get("evidence_id"):
        ref["evidence_ids"] = [str(line["evidence_id"])]
    return ref


def _credit_overview_text_evidence(parse_result: Any) -> dict[int, dict[str, Any]]:
    """Collect exact header-anchored overview rows from corrected page evidence."""

    loader = getattr(parse_result, "corrected_evidence_pages", None)
    pages = loader() if callable(loader) else []
    output: dict[int, dict[str, Any]] = {}
    for page in pages or ():
        if not isinstance(page, Mapping):
            continue
        lines = [line for line in page.get("lines") or () if isinstance(line, Mapping)]
        texts = [_clean(line.get("text") or line.get("content") or "") for line in lines]
        decoded_rows, unresolved_rows = decode_credit_business_overview_text_lines(texts)
        active = False
        header_refs: list[dict[str, Any]] = []
        decoded: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        unresolved_counts = Counter(unresolved_rows)
        unresolved: list[tuple[str, dict[str, Any]]] = []
        for line_index, (line, text) in enumerate(zip(lines, texts, strict=True)):
            if is_credit_business_overview_text_header(text):
                active = True
                header_refs.append(_corrected_text_source_ref(page, line, line_index))
                continue
            if not active:
                continue
            if "汇总" in _compact(text):
                break
            ref = _corrected_text_source_ref(page, line, line_index)
            row = decode_credit_business_overview_text_line(text)
            if row is not None:
                decoded.append((row, ref, text))
            elif unresolved_counts[text] > 0:
                unresolved.append((text, ref))
                unresolved_counts[text] -= 1
        if not header_refs:
            continue
        logical_page = int(page.get("page") or 0)
        if logical_page:
            output[logical_page] = {
                "decoded": decoded,
                "unresolved": unresolved,
                "header_refs": header_refs,
                "source_page": int(page.get("source_page") or logical_page),
            }
    return output


_CANONICAL_SUMMARY_LEAF_LABELS = frozenset(
    {
        "业务类型",
        "账户类型",
        "信息类型",
        "账户数",
        "记录数",
        "月份数",
        "首笔业务发放月份",
        "余额",
        "欠费金额",
        "涉及金额",
        "单月最高逾期/透支总额",
        "最长逾期/透支月数",
        "管理机构数",
        "发卡机构数",
        "授信总额",
        "单家机构最高授信额",
        "单家机构最低授信额",
        "已用额度",
        "透支余额",
        "最近6个月平均应还款",
        "最近6个月平均使用额度",
        "最近6个月平均透支余额",
        "担保金额",
        "还款责任金额",
        "贷款审批",
        "信用卡审批",
        "本人查询",
        "贷后管理",
        "担保资格审查",
        "特约商户实名审查",
    }
)
_CANONICAL_SUMMARY_GROUP_LABELS = frozenset(
    {
        "为个人",
        "为企业",
        "担保责任",
        "其他相关还款责任",
        "最近1个月内的查询机构数",
        "最近1个月内的查询次数",
        "最近2年内的查询次数",
    }
)
_CANONICAL_SUMMARY_HEADER_LABELS = frozenset(
    _compact(value)
    for value in (*_CANONICAL_SUMMARY_LEAF_LABELS, *_CANONICAL_SUMMARY_GROUP_LABELS)
)

_CANONICAL_SUMMARY_COLUMN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "信用业务概要": ("业务分组", "业务类型", "账户数", "首笔业务发放月份"),
    "逾期(透支)信息汇总": (
        "账户类型",
        "账户数",
        "月份数",
        "单月最高逾期/透支总额",
        "最长逾期/透支月数",
    ),
    "非循环贷账户信息汇总": (
        "管理机构数",
        "账户数",
        "授信总额",
        "余额",
        "最近6个月平均应还款",
    ),
    "循环贷账户一信息汇总": (
        "管理机构数",
        "账户数",
        "授信总额",
        "余额",
        "最近6个月平均应还款",
    ),
    "循环贷账户二信息汇总": (
        "管理机构数",
        "账户数",
        "授信总额",
        "余额",
        "最近6个月平均应还款",
    ),
    "贷记卡账户信息汇总": (
        "发卡机构数",
        "账户数",
        "授信总额",
        "单家机构最高授信额",
        "单家机构最低授信额",
        "已用额度",
        "最近6个月平均使用额度",
    ),
    "准贷记卡账户信息汇总": (
        "发卡机构数",
        "账户数",
        "授信总额",
        "单家机构最高授信额",
        "单家机构最低授信额",
        "透支余额",
        "最近6个月平均透支余额",
    ),
}


def _canonical_summary_columns(title: str, width: int) -> tuple[str, ...] | None:
    key = _compact(title).translate(str.maketrans({"（": "(", "）": ")"}))
    columns = _CANONICAL_SUMMARY_COLUMN_TEMPLATES.get(key)
    return columns if columns is not None and len(columns) == width else None


def _summary_title(rows: list[list[str]]) -> str:
    compact = _compact(" ".join(cell for row in rows for cell in row))
    return next(
        (_clean(cell) for row in rows for cell in row if "汇总" in _compact(cell)),
        "信用业务概要" if "首笔业务发放月份" in compact else "查询记录概要",
    )


def _summary_row_has_values(row: list[str]) -> bool:
    """Distinguish business rows from textual/group header rows."""
    for value in row:
        compact = _compact(value)
        if compact in {"--", "-"} or re.fullmatch(r"[-+]?\d[\d,./年月-]*", compact):
            return True
    return False


def _expanded_summary_headers(row: list[str], width: int) -> list[str]:
    values = [_clean(row[index] if index < len(row) else "") for index in range(width)]
    populated = [index for index, value in enumerate(values) if value]
    if not populated:
        return [""] * width
    if any(_compact(values[index]) not in _CANONICAL_SUMMARY_HEADER_LABELS for index in populated):
        return [""] * width
    expanded = [""] * width
    for position, start in enumerate(populated):
        end = populated[position + 1] if position + 1 < len(populated) else width
        for column in range(start, end):
            expanded[column] = values[start]
    return expanded


def _summary_business_rows(
    fragments: list[tuple[Any, Any, list[list[str]]]],
    *,
    title: str,
) -> tuple[
    list[tuple[Any, Any, int, list[str], list[str]]],
    list[tuple[Any, Any, int, list[str], str]],
]:
    width = max((len(row) for _page, _table, rows in fragments for row in rows), default=0)
    canonical_columns = _canonical_summary_columns(title, width)
    if canonical_columns is not None:
        output: list[tuple[Any, Any, int, list[str], list[str]]] = []
        rejected: list[tuple[Any, Any, int, list[str], str]] = []
        header_conflict = False
        for page, table, rows in fragments:
            for source_row_index, row in enumerate(rows):
                if _summary_row_has_values(row):
                    output.append(
                        (page, table, source_row_index, row, list(canonical_columns))
                    )
                    continue
                for column, value in enumerate(row[:width]):
                    compact = _compact(value)
                    if not compact or compact in _CANONICAL_SUMMARY_GROUP_LABELS or "汇总" in compact:
                        continue
                    if compact not in _CANONICAL_SUMMARY_HEADER_LABELS:
                        continue
                    if compact != _compact(canonical_columns[column]):
                        header_conflict = True
                        rejected.append(
                            (page, table, source_row_index, row, "canonical_template_header_conflict")
                        )
        if header_conflict:
            return [], rejected
        return output, rejected

    header_paths: list[list[str]] = [[] for _column in range(width)]
    output: list[tuple[Any, Any, int, list[str], list[str]]] = []
    rejected: list[tuple[Any, Any, int, list[str], str]] = []
    for page, table, rows in fragments:
        for source_row_index, row in enumerate(rows):
            nonempty = _nonempty(row)
            if len(nonempty) == 1 and "汇总" in _compact(nonempty[0]):
                continue
            if _summary_row_has_values(row):
                labels = ["/".join(path) for path in header_paths]
                missing_label_columns = [
                    column
                    for column, value in enumerate(row)
                    if _clean(value) and (column >= len(labels) or not labels[column])
                ]
                if missing_label_columns:
                    rejected.append(
                        (page, table, source_row_index, row, "value_without_exact_canonical_header")
                    )
                    continue
                output.append((page, table, source_row_index, row, labels))
                continue
            expanded = _expanded_summary_headers(row, width)
            if nonempty and not any(expanded):
                rejected.append((page, table, source_row_index, row, "unknown_summary_header"))
                continue
            distinct = {value for value in expanded if value}
            if len(distinct) == 1:
                label = next(iter(distinct))
                header_paths = [[label] for _column in range(width)]
                continue
            for column, label in enumerate(expanded):
                if label and (not header_paths[column] or header_paths[column][-1] != label):
                    header_paths[column].append(label)
    return output, rejected


def _extract_summary_datasets(
    parse_result: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project logical summary grids, including headerless cross-page fragments."""
    records: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    physical: list[tuple[Any, Any, list[list[str]]]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            if rows:
                physical.append((page, table, rows))

    continuation_check = getattr(parse_result, "tables_continue", None)
    reading_order = dict(getattr(parse_result, "reading_order_by_logical", {}) or {})
    text_evidence = _credit_overview_text_evidence(parse_result)
    consumed_text_pages: set[int] = set()
    consumed: set[int] = set()
    for index, (page, table, rows) in enumerate(physical):
        page_number = int(getattr(page, "page_number", 0) or 0)
        text_anchor = page_number in text_evidence and _is_headerless_credit_overview_fragment(rows)
        if index in consumed or (not _is_summary_anchor(rows) and not text_anchor):
            continue
        fragments = [(page, table, rows)]
        anchor_width = max((len(row) for row in rows), default=0)
        cursor = index + 1
        while cursor < len(physical):
            next_page, next_table, next_rows = physical[cursor]
            if _is_summary_anchor(next_rows):
                break
            previous_page, previous_table, _previous_rows = fragments[-1]
            previous_page_number = int(getattr(previous_page, "page_number", 0) or 0)
            next_page_number = int(getattr(next_page, "page_number", 0) or 0)
            next_width = max((len(row) for row in next_rows), default=0)
            previous_table_id = str(getattr(previous_table, "table_id", "") or "")
            next_table_id = str(getattr(next_table, "table_id", "") or "")
            previous_order = reading_order.get(previous_page_number, previous_page_number)
            next_order = reading_order.get(next_page_number, next_page_number)
            if (
                next_order != previous_order + 1
                or not callable(continuation_check)
                or continuation_check(previous_table_id, next_table_id) is not True
                or (anchor_width and next_width != anchor_width)
            ):
                break
            fragments.append((next_page, next_table, next_rows))
            consumed.add(cursor)
            cursor += 1

        title = "信用业务概要" if text_anchor else _summary_title(rows)
        table_id = str(getattr(table, "table_id", "") or "")
        summary_id = stable_record_id("personal_detail_summary", page_number, table_id, title)
        business_rows, rejected_rows = _summary_business_rows(fragments, title=title)
        text_rows: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        overview_evidence = text_evidence.get(page_number) if _compact(title) == "信用业务概要" else None
        if isinstance(overview_evidence, Mapping):
            remove_business_rows: set[int] = set()
            for decoded, ref, raw_text in overview_evidence.get("decoded") or ():
                if not isinstance(decoded, Mapping) or not isinstance(ref, Mapping):
                    continue
                count = decoded.get("account_count")
                month = str(decoded.get("first_business_issue_month") or "")
                business_type = str(decoded.get("business_type") or "")
                exact_matches: list[int] = []
                identity_matches: list[int] = []
                for row_index, (_row_page, _row_table, _source_index, source_row, _labels) in enumerate(
                    business_rows
                ):
                    if len(source_row) < 4:
                        continue
                    raw_count = _compact(source_row[2]).replace(",", "")
                    raw_month = _compact(source_row[3])
                    month_match = re.fullmatch(r"((?:19|20)\d{2})[.:/-](\d{1,2})", raw_month)
                    normalized_month = (
                        f"{int(month_match.group(1)):04d}-{int(month_match.group(2)):02d}"
                        if month_match and 1 <= int(month_match.group(2)) <= 12
                        else ""
                    )
                    if raw_count == str(count) and normalized_month == month:
                        identity_matches.append(row_index)
                        normalized_row = decode_credit_business_overview_text_line(
                            f"{source_row[1]} {source_row[2]} {source_row[3]}"
                        )
                        if normalized_row and str(normalized_row.get("business_type") or "") == business_type:
                            exact_matches.append(row_index)
                if exact_matches:
                    remove_business_rows.update(exact_matches[1:])
                    continue
                if len(identity_matches) == 1:
                    table_index = identity_matches[0]
                    source_row = business_rows[table_index][3]
                    normalized_table = decode_credit_business_overview_text_line(
                        f"{source_row[1]} {source_row[2]} {source_row[3]}"
                    )
                    table_category = (
                        str(normalized_table.get("business_type") or "")
                        if normalized_table
                        else ""
                    )
                    remove_business_rows.add(table_index)
                    if table_category and table_category != business_type:
                        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                            make_issue,
                            record_issue,
                        )

                        ambiguous = dict(decoded)
                        ambiguous["business_type"] = None
                        text_rows.append((ambiguous, dict(ref), str(raw_text)))
                        record_issue(
                            parse_result,
                            make_issue(
                                category="ocr_structure_correction",
                                issue_code="candidate_b_summary_category_collision_unresolved",
                                message=(
                                    "Text and table observations shared one count/month identity but disagreed "
                                    "on the business category; the category was withheld."
                                ),
                                parser_stage="candidate_b_summary_text_table_reconciliation",
                                target_dataset="personal_detail_summary_cells",
                                target_record_id=summary_id,
                                field_name="business_type",
                                observed_value={
                                    "table_category": table_category,
                                    "text_category": business_type,
                                    "account_count": count,
                                    "first_business_issue_month": month,
                                },
                                source_refs=(ref,),
                                reason_codes=(
                                    "one_to_one_count_month_collision",
                                    "conflicting_finite_categories",
                                    "category_withheld",
                                    "duplicate_row_suppressed",
                                ),
                            ),
                        )
                    else:
                        text_rows.append((dict(decoded), dict(ref), str(raw_text)))
                    continue
                if len(identity_matches) > 1:
                    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                        make_issue,
                        record_issue,
                    )

                    record_issue(
                        parse_result,
                        make_issue(
                            category="ocr_structure_correction",
                            issue_code="candidate_b_summary_category_collision_unresolved",
                            message=(
                                "A corrected text row matched multiple table rows by count and month but no "
                                "category matched exactly; the ambiguous text category was withheld."
                            ),
                            parser_stage="candidate_b_summary_text_table_reconciliation",
                            target_dataset="personal_detail_summary_cells",
                            target_record_id=summary_id,
                            field_name="business_type",
                            observed_value={
                                "text_category": business_type,
                                "account_count": count,
                                "first_business_issue_month": month,
                                "table_match_count": len(identity_matches),
                            },
                            source_refs=(ref,),
                            reason_codes=(
                                "one_to_many_count_month_collision",
                                "category_owner_ambiguous",
                                "text_row_withheld",
                                "duplicate_row_suppressed",
                            ),
                        ),
                    )
                    continue
                text_rows.append((dict(decoded), dict(ref), str(raw_text)))
            if remove_business_rows:
                business_rows = [
                    row for row_index, row in enumerate(business_rows) if row_index not in remove_business_rows
                ]
            consumed_text_pages.add(page_number)
        if rejected_rows:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            for rejected_page, rejected_table, rejected_row_index, rejected_row, reason in rejected_rows:
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code="candidate_b_summary_layout_unresolved",
                        message=(
                            "A canonical summary row could not be bound to the finite summary template; "
                            "its cells were withheld instead of inheriting positional labels."
                        ),
                        parser_stage="candidate_b_summary_canonical_slots",
                        target_dataset="personal_detail_summary_cells",
                        observed_value={"row": rejected_row, "reason": reason},
                        source_refs=(
                            _source_ref(
                                rejected_page,
                                rejected_table,
                                row=rejected_row_index,
                            ),
                        ),
                        reason_codes=(
                            "canonical_summary_table_observed",
                            reason,
                            "header_fill_inference_forbidden",
                            "normalized_cells_withheld",
                        ),
                    ),
                )
        if not business_rows and not text_rows:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_summary_anchor_without_usable_rows",
                    message="A canonical summary anchor was observed but yielded zero usable business rows.",
                    parser_stage="candidate_b_summary_canonical_slots",
                    target_dataset="personal_detail_summary_cells",
                    target_record_id=summary_id,
                    observed_value={"title": title, "usable_row_count": 0},
                    source_refs=(
                        _source_ref(fragment_page, fragment_table)
                        for fragment_page, fragment_table, _fragment_rows in fragments
                    ),
                    reason_codes=(
                        "canonical_summary_anchor_observed",
                        "zero_usable_rows",
                        "silent_drop_prevented",
                    ),
                ),
            )
        records.append(
            {
                "record_id": summary_id,
                "summary_record_id": summary_id,
                "summary_type": re.sub(r"信息?汇总$", "", title) or title,
                "title": title,
                "source_table_id": table_id,
                "source_column_count": max(
                    (len(row) for _fragment_page, _fragment_table, fragment_rows in fragments for row in fragment_rows),
                    default=0,
                ),
                "source_row_count": len(text_rows) + len(business_rows),
                "source": "native_personal_detail_summary_table",
                "source_refs": [
                    *[
                        _source_ref(fragment_page, fragment_table)
                        for fragment_page, fragment_table, _ in fragments
                    ],
                    *[ref for _decoded, ref, _raw_text in text_rows],
                ],
                "confidence": 1.0,
            }
        )
        for logical_row_index, (decoded, ref, raw_text) in enumerate(text_rows, start=1):
            for column_index, (header, value) in enumerate(
                (
                    ("业务类型", decoded.get("business_type")),
                    ("账户数", decoded.get("account_count")),
                    ("首笔业务发放月份", decoded.get("first_business_issue_month")),
                ),
                start=2,
            ):
                if value in (None, ""):
                    continue
                cell_id = stable_record_id(
                    "personal_detail_summary_cell",
                    summary_id,
                    logical_row_index,
                    column_index,
                    value,
                )
                cells.append(
                    {
                        "record_id": cell_id,
                        "summary_cell_id": cell_id,
                        "summary_record_id": summary_id,
                        "summary_type": "信用业务概要",
                        "title": "信用业务概要",
                        "row_index": logical_row_index,
                        "column_index": column_index,
                        "column_label": header,
                        "value": str(value),
                        "canonical_raw": {"value": raw_text},
                        "source": "candidate_b_corrected_summary_text_row",
                        "source_refs": [ref],
                        "confidence": 1.0,
                    }
                )
        for logical_row_index, (source_page, source_table, source_row_index, row, labels) in enumerate(
            business_rows, start=len(text_rows) + 1
        ):
            for column_index, value in enumerate(row, start=1):
                value = _clean(value)
                if not value:
                    continue
                header = labels[column_index - 1] if column_index <= len(labels) else ""
                if _compact(title) == "信用业务概要" and header == "业务分组":
                    # 贷款/信用卡/其他 is a merged visual group label, not the
                    # row's individualized business category.
                    continue
                cell_id = stable_record_id(
                    "personal_detail_summary_cell",
                    summary_id,
                    logical_row_index,
                    column_index,
                    value,
                )
                cells.append(
                    {
                        "record_id": cell_id,
                        "summary_cell_id": cell_id,
                        "summary_record_id": summary_id,
                        "summary_type": re.sub(r"信息?汇总$", "", title) or title,
                        "title": title,
                        "row_index": logical_row_index,
                        "column_index": column_index,
                        "column_label": header or None,
                        "value": value,
                        "source": "native_personal_detail_summary_cell",
                        "source_refs": [
                            _source_ref(
                                source_page,
                                source_table,
                                row=source_row_index,
                                column=column_index - 1,
                            )
                        ],
                        "confidence": 1.0,
                    }
                )
        if isinstance(overview_evidence, Mapping):
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            for raw_text, ref in overview_evidence.get("unresolved") or ():
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_summary_text_row_unresolved",
                        message=(
                            "A canonical credit-overview text row failed its finite category/count/month "
                            "contract and was withheld rather than guessed."
                        ),
                        parser_stage="candidate_b_summary_text_fallback",
                        target_dataset="personal_detail_summary_cells",
                        target_record_id=summary_id,
                        observed_value=raw_text,
                        source_refs=(ref,),
                        reason_codes=(
                            "canonical_credit_overview_text_header",
                            "typed_summary_row_contract_failed",
                            "normalized_cells_withheld",
                        ),
                    ),
                )
    for page_number, overview_evidence in text_evidence.items():
        if page_number in consumed_text_pages:
            continue
        decoded_rows = list(overview_evidence.get("decoded") or ())
        unresolved_rows = list(overview_evidence.get("unresolved") or ())
        summary_id = stable_record_id(
            "personal_detail_summary",
            page_number,
            "corrected_text",
            "信用业务概要",
        )
        source_refs = [
            dict(ref)
            for _decoded, ref, _raw_text in decoded_rows
            if isinstance(ref, Mapping)
        ]
        source_refs.extend(
            dict(ref)
            for ref in overview_evidence.get("header_refs") or ()
            if isinstance(ref, Mapping) and dict(ref) not in source_refs
        )
        if not decoded_rows:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            record_issue(
                parse_result,
                make_issue(
                    category="schema_incompleteness",
                    issue_code="candidate_b_summary_anchor_without_usable_rows",
                    message="A canonical summary text anchor was observed but yielded zero usable business rows.",
                    parser_stage="candidate_b_summary_text_fallback",
                    target_dataset="personal_detail_summary_cells",
                    target_record_id=summary_id,
                    observed_value={"title": "信用业务概要", "usable_row_count": 0},
                    source_refs=source_refs,
                    reason_codes=(
                        "canonical_summary_anchor_observed",
                        "zero_usable_rows",
                        "silent_drop_prevented",
                    ),
                ),
            )
        records.append(
            {
                "record_id": summary_id,
                "summary_record_id": summary_id,
                "summary_type": "信用业务概要",
                "title": "信用业务概要",
                "source_table_id": None,
                "source_column_count": 4,
                "source_row_count": len(decoded_rows),
                "source": "candidate_b_corrected_summary_text",
                "source_refs": source_refs,
                "confidence": 1.0,
            }
        )
        for logical_row_index, (decoded, ref, raw_text) in enumerate(decoded_rows, start=1):
            for column_index, (header, value) in enumerate(
                (
                    ("业务类型", decoded.get("business_type")),
                    ("账户数", decoded.get("account_count")),
                    ("首笔业务发放月份", decoded.get("first_business_issue_month")),
                ),
                start=2,
            ):
                if value in (None, ""):
                    continue
                cell_id = stable_record_id(
                    "personal_detail_summary_cell",
                    summary_id,
                    logical_row_index,
                    column_index,
                    value,
                )
                cells.append(
                    {
                        "record_id": cell_id,
                        "summary_cell_id": cell_id,
                        "summary_record_id": summary_id,
                        "summary_type": "信用业务概要",
                        "title": "信用业务概要",
                        "row_index": logical_row_index,
                        "column_index": column_index,
                        "column_label": header,
                        "value": str(value),
                        "canonical_raw": {"value": raw_text},
                        "source": "candidate_b_corrected_summary_text_row",
                        "source_refs": [dict(ref)],
                        "confidence": 1.0,
                    }
                )
        if unresolved_rows:
            from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
                make_issue,
                record_issue,
            )

            for raw_text, ref in unresolved_rows:
                record_issue(
                    parse_result,
                    make_issue(
                        category="ocr_cell_level_error",
                        issue_code="candidate_b_summary_text_row_unresolved",
                        message=(
                            "A canonical credit-overview text row failed its finite category/count/month "
                            "contract and was withheld rather than guessed."
                        ),
                        parser_stage="candidate_b_summary_text_fallback",
                        target_dataset="personal_detail_summary_cells",
                        target_record_id=summary_id,
                        observed_value=raw_text,
                        source_refs=(ref,),
                        reason_codes=(
                            "canonical_credit_overview_text_header",
                            "typed_summary_row_contract_failed",
                            "normalized_cells_withheld",
                        ),
                    ),
                )
    return records, cells


def _extract_recovery_records(parse_result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in getattr(parse_result, "pages", None) or []:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            compact = _compact(" ".join(cell for row in rows[:3] for cell in row))
            if "债权接收日期" not in compact or "原债权人" not in compact:
                continue
            sequence = len(records) + 1
            provisional_id = stable_record_id(
                "recovery_unresolved",
                int(getattr(page, "page_number", 0) or 0),
                str(getattr(table, "table_id", "") or ""),
            )
            record = _decode_exact_label_card(
                parse_result,
                page=page,
                table=table,
                dataset="recovery_records",
                target_record_id=provisional_id,
                fields={
                    "institution": (("管理机构",), "text"),
                    "business_type": (("业务种类",), "text"),
                    "debt_received_date": (("债权接收日期",), "date"),
                    "original_creditor": (("原债权人",), "text"),
                    "original_business_type": (("原债务业务种类",), "text"),
                    "debt_amount": (("债权金额",), "number"),
                    "transfer_repayment_status": (("债权转移时的还款状态",), "text"),
                    "account_status": (("账户状态",), "text"),
                    "balance": (("余额",), "number"),
                    "last_repayment_date": (("最近一次还款日期",), "date"),
                    "close_date": (("账户关闭日期",), "date"),
                },
            )
            received_date = record.get("debt_received_date")
            institution = record.get("institution")
            recovery_id = (
                stable_record_id("recovery", institution, received_date, sequence)
                if institution and received_date
                else provisional_id
            )
            record.update(
                {
                    "recovery_record_id": recovery_id,
                    "sequence": sequence,
                    "snapshot_date": next(
                        (
                            _date("-".join(match.groups()))
                            for row in rows
                            if (match := _AS_OF_RE.search(_clean(" ".join(row))))
                        ),
                        None,
                    ),
                    "reporting_amount_currency": "CNY",
                    "reporting_amount_unit": "yuan",
                    "source": "native_detail_recovery_table",
                    "confidence": float(getattr(table, "confidence", None) or 0.9),
                }
            )
            if recovery_id != provisional_id:
                for issue in getattr(parse_result, "_personal_detail_extraction_issues", []) or []:
                    if isinstance(issue, dict) and issue.get("target_record_id") == provisional_id:
                        issue["target_record_id"] = recovery_id
            records.append(record)
    return records


def _iso_report_time(parts: tuple[str, str, str, str, str, str]) -> str | None:
    try:
        value = datetime(*(int(part) for part in parts))
    except ValueError:
        return None
    return value.isoformat(timespec="seconds") + "+08:00"


def _strict_report_time(compact: str, report_number: str | None) -> str | None:
    standard = re.search(
        r"报告时间[:：]?(20\d{2})[.年/-](\d{1,2})[.月/-](\d{1,2})日?(\d{1,2}):(\d{2}):(\d{2})",
        compact,
    )
    if standard:
        return _iso_report_time(tuple(standard.groups()))

    # One recurring scan form deletes a single date digit, e.g.
    # ``2023.01314:45:37``.  Recover it only when the report-number timestamp
    # is valid and deleting exactly one of its digits reproduces the complete
    # malformed timestamp evidence.  This is cross-field source agreement,
    # not a date guess.
    damaged = re.search(r"报告时间[:：]?((?:19|20)\d{2}[.年/-]\d{3,6}:\d{2}:\d{2})", compact)
    prefix = str(report_number or "")[:14]
    if damaged is None or not re.fullmatch(r"\d{14}", prefix):
        return None
    source_digits = re.sub(r"\D", "", damaged.group(1))
    if len(source_digits) != 13 or not any(
        prefix[:index] + prefix[index + 1 :] == source_digits for index in range(len(prefix))
    ):
        return None
    return _iso_report_time(
        (prefix[0:4], prefix[4:6], prefix[6:8], prefix[8:10], prefix[10:12], prefix[12:14])
    )


def _extract_header_datasets(parse_result: Any, full_text: str) -> dict[str, list[dict[str, Any]]]:
    compact = _compact(full_text)
    report_number = next(iter(re.findall(r"报告编号[:：]?(\d{18,30})", compact)), None)
    report_time = _strict_report_time(compact, report_number)
    field_candidates: dict[str, list[str]] = defaultdict(list)
    for page in list(getattr(parse_result, "pages", None) or [])[:1]:
        for table in getattr(page, "tables", None) or []:
            rows = _table_rows(table)
            for row_index, row in enumerate(rows[:-1]):
                header_slots = _canonical_header_slots(
                    tuple(row),
                    {
                        "subject_name": ("被查询者姓名",),
                        "primary_id_type": ("被查询者证件类型",),
                        "primary_id_number": ("被查询者证件号码",),
                        "query_institution": ("查询机构",),
                        "query_reason": ("查询原因",),
                    },
                )
                if set(header_slots) != {
                    "subject_name",
                    "primary_id_type",
                    "primary_id_number",
                    "query_institution",
                    "query_reason",
                }:
                    continue
                value_row = tuple(rows[row_index + 1])
                for key, column in header_slots.items():
                    value = _clean(value_row[column] if column < len(value_row) else "")
                    if value:
                        field_candidates[key].append(value)
    from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
        make_issue,
        record_issue,
        retarget_issue_record,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import PBOCPersonalDetailNativeParser

    parser = PBOCPersonalDetailNativeParser(parse_result)
    label_to_key = {
        "被查询者姓名": "subject_name",
        "被查询者证件类型": "primary_id_type",
        "被查询者证件号码": "primary_id_number",
        "查询机构": "query_institution",
        "查询原因": "query_reason",
        "报告编号": "report_number",
        "报告时间": "report_time",
    }
    for candidate in parser.records("report_header"):
        for label, key in label_to_key.items():
            value = candidate.fields.get(label)
            if value not in (None, ""):
                field_candidates[key].append(str(value))
    if report_number:
        field_candidates["report_number"].append(report_number)
    if report_time:
        field_candidates["report_time"].append(report_time)

    def candidate_valid(key: str, value: str, selected_id_type: str | None = None) -> bool:
        if key == "query_reason":
            return validate_pboc_field(value, "inquiry_reason").valid
        return header_field_valid(key, value, id_type=selected_id_type)

    def select(key: str, selected_type: str | None = None) -> str | None:
        observed = tuple(dict.fromkeys(value.strip() for value in field_candidates.get(key, []) if value.strip()))
        valid = tuple(
            dict.fromkeys(
                normalize_pboc_field(value, "inquiry_reason") if key == "query_reason" else value
                for value in observed
                if candidate_valid(key, value, selected_type)
            )
        )
        if len(valid) == 1:
            return valid[0]
        target_dataset = (
            "report_query"
            if key in {"query_institution", "query_reason"}
            else "report_metadata"
        )
        record_issue(
            parse_result,
            make_issue(
                category="ocr_cell_level_error",
                issue_code="page_one_consensus_unresolved",
                message="Page-one header evidence was missing, invalid, or conflicting; the normalized value was withheld.",
                parser_stage="page_one_consensus",
                target_dataset=target_dataset,
                target_record_id="personal_report_metadata:primary",
                field_name=key,
                observed_value=list(observed),
                source_refs=(
                    {
                        "source": "candidate_b_page_one_business_fields",
                        "logical_page": 1,
                        "source_page": 1,
                        "geometry_scope": "logical_page",
                    },
                ),
                reason_codes=("page_one_consensus", "schema_field_validation", "normalized_value_withheld"),
            ),
        )
        return None

    subject_name = select("subject_name")
    id_type = select("primary_id_type")
    id_number = select("primary_id_number", id_type or "身份证")
    observed_primary_numbers = tuple(
        dict.fromkeys(
            value.strip()
            for value in field_candidates.get("primary_id_number", [])
            if value.strip()
        )
    )
    source_primary_id_number = (
        id_number
        or (observed_primary_numbers[0] if len(observed_primary_numbers) == 1 else None)
    )
    if id_type is None and cn_identity_number_valid(id_number):
        id_type = "身份证"
        for issue in getattr(parse_result, "_personal_detail_extraction_issues", []) or []:
            if (
                isinstance(issue, dict)
                and issue.get("issue_code") == "page_one_consensus_unresolved"
                and issue.get("field_name") == "primary_id_type"
            ):
                issue["status"] = "resolved"
                issue["severity"] = "info"
                issue["reason_codes"] = [
                    *issue.get("reason_codes", []),
                    "checksum_valid_resident_identity_implies_document_type",
                ]
    query_institution = select("query_institution")
    query_reason = select("query_reason")
    report_number = select("report_number")
    report_time = select("report_time")
    metadata_id = stable_record_id(
        "personal_report_metadata", report_number, report_time, subject_name
    )
    page_one_targets = {
        "report_metadata": metadata_id,
        "report_query": f"report_query:{metadata_id}",
    }
    issues = getattr(parse_result, "_personal_detail_extraction_issues", None)
    if isinstance(issues, list):
        for index, issue in enumerate(issues):
            if not isinstance(issue, Mapping):
                continue
            target_dataset = str(issue.get("target_dataset") or "")
            if (
                issue.get("issue_code") == "page_one_consensus_unresolved"
                and issue.get("target_record_id") == "personal_report_metadata:primary"
                and target_dataset in page_one_targets
            ):
                issues[index] = retarget_issue_record(
                    issue,
                    page_one_targets[target_dataset],
                )
    metadata = [
        {
            "personal_report_metadata_id": metadata_id,
            "report_number": report_number,
            "report_time": report_time,
            "subject_name": subject_name,
            "primary_id_type": id_type,
            "primary_id_number": id_number,
            "query_institution": query_institution,
            "query_reason": query_reason,
            "reporting_currency": "CNY",
            "reporting_amount_unit": "yuan",
            "reporting_amount_precision": 0,
            "amount_policy_source": "personal_detailed_report_notes",
            "source": "native_detail_header",
            "source_refs": [{"source": "native_detail_header", "logical_page": 1, "source_page": 1}],
            "confidence": 1.0,
        }
    ]
    identities: list[dict[str, Any]] = []
    if id_type and source_primary_id_number:
        identities.append(
            {
                "identity_document_id": stable_record_id(
                    "identity_document", "primary", id_type, source_primary_id_number
                ),
                "sequence": 1,
                "holder_name": subject_name,
                "document_type": id_type,
                # Preserve the source-visible row even when its displayed
                # number fails a checksum/type contract.  The v2 quality gate
                # withholds the normalized field and publishes the linked
                # uncertainty instead of silently deleting the identity row.
                "document_number": source_primary_id_number,
                "is_primary": True,
                "source": "native_detail_header",
                "source_refs": [{"source": "native_detail_header", "logical_page": 1, "source_page": 1}],
                "confidence": 1.0,
            }
        )
    return {"personal_report_metadata": metadata, "identity_documents": identities}


def extract_personal_detail_native_business(parse_result: Any, full_text: str) -> dict[str, Any]:
    """Return the business slice of the authoritative Candidate B result."""
    from copy import deepcopy

    from docmirror.plugins.credit_report.personal_detail_scanned.context import (
        build_personal_detail_extraction_context,
    )

    context = build_personal_detail_extraction_context(parse_result)
    return deepcopy(context.candidate_b_extraction(full_text).business)


def extract_personal_detail_section_content(parse_result: Any, full_text: str) -> dict[str, Any]:
    """Return the supplemental slice of the authoritative Candidate B result."""
    from copy import deepcopy

    from docmirror.plugins.credit_report.personal_detail_scanned.context import (
        build_personal_detail_extraction_context,
    )

    context = build_personal_detail_extraction_context(parse_result)
    return deepcopy(context.candidate_b_extraction(full_text).section_content)


def extract_personal_detail_common_datasets(
    parse_result: Any,
    full_text: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return report metadata datasets for any content mode."""
    return _extract_header_datasets(parse_result, full_text)


__all__ = [
    "extract_personal_detail_common_datasets",
    "extract_personal_detail_native_business",
    "extract_personal_detail_section_content",
]
