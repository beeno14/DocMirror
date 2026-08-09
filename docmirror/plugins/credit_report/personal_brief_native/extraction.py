# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variant-owned extraction entry points for native personal brief reports."""

from __future__ import annotations

import re
from typing import Any

_MARITAL_STATUS_CODES = {
    "未婚": "unmarried",
    "已婚": "married",
    "初婚": "married",
    "再婚": "married",
    "离婚": "divorced",
    "丧偶": "widowed",
    "其他": "other",
    "未说明": "not_reported",
    "未知": "not_reported",
}


def _personal_brief_blocks(parse_result: Any) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    entity_context = getattr(parse_result, "entity_context", None)
    ordered_text_blocks = getattr(entity_context, "ordered_text_blocks", None)
    if getattr(entity_context, "report_family", "") == "personal_brief" and callable(ordered_text_blocks):
        blocks.extend(
            (int(page), str(content).strip())
            for page, content in ordered_text_blocks()
            if str(content or "").strip()
        )
        return blocks
    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "page_number", 0) or page_index)
        for block in getattr(page, "texts", None) or []:
            content = str(getattr(block, "content", "") or "").strip()
            if content:
                blocks.append((page_number, content))
    return blocks


def _label_fields(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Read a compact label/value form without depending on table geometry."""
    flattened = re.sub(r"\s+", " ", str(text or "")).strip()
    found: list[tuple[int, int, str]] = []
    for label in labels:
        for match in re.finditer(rf"{re.escape(label)}\s*[:：]\s*", flattened):
            found.append((match.start(), match.end(), label))
    found.sort()
    fields: dict[str, str] = {}
    for index, (_start, value_start, label) in enumerate(found):
        value_end = found[index + 1][0] if index + 1 < len(found) else len(flattened)
        value = flattened[value_start:value_end].strip(" ，,；;。")
        fields.setdefault(label, value)
    return fields


def _table_chunks(parse_result: Any, required_label: str) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for page_index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
        page_number = int(getattr(page, "page_number", 0) or page_index)
        for table in getattr(page, "tables", None) or []:
            parts: list[str] = []
            raw_rows = (getattr(table, "metadata", None) or {}).get("raw_rows") or []
            if raw_rows:
                parts.extend(str(cell or "") for row in raw_rows for cell in row)
            else:
                parts.extend(str(header or "") for header in getattr(table, "headers", None) or [])
                for row in getattr(table, "rows", None) or []:
                    parts.extend(
                        str(getattr(cell, "text", "") or "")
                        for cell in getattr(row, "cells", None) or []
                    )
            text = "\n".join(part for part in parts if part.strip())
            if required_label in text:
                chunks.append((page_number, text))
    return chunks


def _pair_record_chunks(
    anchors: list[tuple[int, str]],
    chunks: list[tuple[int, str]],
    *,
    identity_labels: tuple[str, ...],
) -> list[tuple[int, str]]:
    """Pair public-record fragments by identity/page evidence, never ordinal alone."""
    paired: list[tuple[int, str]] = []
    unused = set(range(len(chunks)))
    for anchor_page, anchor_text in anchors:
        anchor_fields = _label_fields(anchor_text, identity_labels)

        def score(chunk_index: int) -> tuple[int, int, int, int]:
            chunk_page, chunk_text = chunks[chunk_index]
            chunk_fields = _label_fields(chunk_text, identity_labels)
            shared = sum(
                bool(anchor_fields.get(label) and anchor_fields.get(label) == chunk_fields.get(label))
                for label in identity_labels
            )
            page_affinity = 3 if chunk_page == anchor_page else 2 if chunk_page == anchor_page + 1 else 1
            return shared, page_affinity, -abs(chunk_page - anchor_page), -chunk_index

        if not unused:
            paired.append((anchor_page, ""))
            continue
        best = max(unused, key=score)
        # A fragment more than one page away with no shared identity is not a
        # defensible match.  Leave it unpaired instead of shifting every later
        # record by one physical table.
        best_page, best_text = chunks[best]
        best_score = score(best)
        if best_score[0] == 0 and abs(best_page - anchor_page) > 1:
            paired.append((anchor_page, ""))
            continue
        unused.remove(best)
        paired.append((best_page, best_text))
    return paired


def _public_record_refs(anchor_page: int, detail_page: int) -> list[dict[str, Any]]:
    refs = [{"source": "native_text_public_record_anchor", "page": anchor_page}]
    if detail_page != anchor_page:
        refs.append({"source": "canonical_public_record_table", "page": detail_page})
    else:
        refs[0]["source"] = "native_text_and_table"
    return refs


def _date_or_month(value: str) -> str | None:
    from docmirror.plugins.credit_report.business_records import _iso_date, _iso_month

    return _iso_date(value) or _iso_month(value) or None


def _reported_text(value: str | None) -> str | None:
    compact = re.sub(r"\s+", "", str(value or ""))
    return None if compact in {"", "--"} else str(value).strip()


def _summary_source_text(parse_result: Any, *fallbacks: str) -> str:
    """Include table units when the fast batch path does not materialize tables."""
    from docmirror.plugins.credit_report.business_records import _linear

    parts = [str(value or "") for value in fallbacks if str(value or "").strip()]
    entity_context = getattr(parse_result, "entity_context", None)
    ordered_page_flow = getattr(entity_context, "ordered_page_flow", None)
    if callable(ordered_page_flow):
        for _page, kind, payload in ordered_page_flow():
            if kind == "table" and isinstance(payload, tuple) and len(payload) == 2:
                _table_id, rows = payload
                parts.extend(
                    " ".join(str(cell or "") for cell in row)
                    for row in rows
                    if isinstance(row, list)
                )
            elif kind != "table":
                parts.append(str(payload or ""))
    return _linear("\n".join(parts))


def _personal_header_datasets(
    parse_result: Any,
    blocks: list[tuple[int, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from docmirror.plugins.credit_report.business_records import _source_refs, _stable_id

    header_parts: list[str] = []
    for _page, content in blocks:
        if re.sub(r"\s+", "", content) == "信贷记录":
            break
        header_parts.append(content)
    header = re.sub(r"\s+", " ", "\n".join(header_parts))
    compact = re.sub(r"\s+", "", header)
    report_number_match = re.search(r"报告编号[:：]([A-Za-z0-9-]+)", compact)
    report_time_match = re.search(
        r"报告时间[:：](20\d{2})[-年](\d{1,2})[-月](\d{1,2})日?(\d{1,2})[:时](\d{1,2})[:分](\d{1,2})秒?",
        compact,
    )
    name_match = re.search(r"姓名[:：]([^:：]+?)(?=证件类型[:：])", compact)
    id_type_match = re.search(r"证件类型[:：]([^:：]+?)(?=证件号码[:：])", compact)
    id_number_match = re.search(r"证件号码[:：]([A-Za-z0-9*]+)", compact)
    marital_status_match = re.search(
        r"婚姻状况[:：]?([^:：，,；;。]{1,8}?)(?=其他证件信息|信贷记录|报告|[:：，,；;。]|$)",
        compact,
    )
    if not marital_status_match:
        # Unlabelled marital status is printed next to the primary identity,
        # so constrain the fallback to that metadata tail.  Searching the
        # whole header incorrectly matched words such as ``其他用途`` in the
        # confidentiality notice before reaching the actual ``已婚`` value.
        identity_tail_start = id_number_match.end() if id_number_match else 0
        identity_tail_end = compact.find("其他证件信息", identity_tail_start)
        identity_tail = compact[
            identity_tail_start : identity_tail_end if identity_tail_end >= 0 else None
        ]
        marital_status_match = re.search(
            r"(未婚|已婚|初婚|再婚|离婚|丧偶|其他|未说明|未知)",
            identity_tail,
        )
    report_time = (
        f"{int(report_time_match.group(1)):04d}-{int(report_time_match.group(2)):02d}-"
        f"{int(report_time_match.group(3)):02d}T{int(report_time_match.group(4)):02d}:"
        f"{int(report_time_match.group(5)):02d}:{int(report_time_match.group(6)):02d}+08:00"
        if report_time_match
        else None
    )
    report_number = report_number_match.group(1) if report_number_match else None
    subject_name = name_match.group(1) if name_match else None
    primary_id_type = id_type_match.group(1) if id_type_match else None
    primary_id_number = id_number_match.group(1) if id_number_match else None
    marital_status_raw = marital_status_match.group(1) if marital_status_match else None
    marital_status = _MARITAL_STATUS_CODES.get(marital_status_raw or "")
    page = blocks[0][0] if blocks else 1
    identity_documents: list[dict[str, Any]] = []
    if primary_id_type and primary_id_number:
        identity_documents.append(
            {
                "identity_document_id": _stable_id(
                    "identity_document", "primary", primary_id_type, primary_id_number
                ),
                "sequence": 1,
                "holder_name": subject_name,
                "document_type": primary_id_type,
                "document_number": primary_id_number,
                "is_primary": True,
                "source": "personal_brief_header",
                "source_refs": _source_refs(page, "native_text_header"),
                "confidence": 0.99,
            }
        )
    other_match = re.search(r"其他证件信息[:：]([^信贷记录]+)", compact)
    if other_match:
        other_value = other_match.group(1).strip("，,。;；")
        for item in re.split(r"[，,；;]", other_value):
            item_match = re.match(r"([\u3400-\u9fff（）()]+?)([A-Za-z0-9][A-Za-z0-9*.-]+)$", item)
            if not item_match:
                continue
            identity_documents.append(
                {
                    "identity_document_id": _stable_id(
                        "identity_document", "other", item_match.group(1), item_match.group(2)
                    ),
                    "sequence": len(identity_documents) + 1,
                    "holder_name": subject_name,
                    "document_type": item_match.group(1),
                    "document_number": item_match.group(2),
                    "is_primary": False,
                    "source": "personal_brief_header",
                    "source_refs": _source_refs(page, "native_text_header"),
                    "confidence": 0.97,
                }
            )
    amount_policy = {
        "reporting_currency": "CNY",
        "reporting_amount_unit": "yuan",
        "reporting_amount_precision": 0,
        "amount_policy_source": (
            "source_statement"
            if "金额类数据均以人民币计算" in compact and "精确到元" in compact
            else "personal_brief_standard"
        ),
    }
    metadata = [
        {
            "personal_report_metadata_id": _stable_id(
                "personal_report_metadata", report_number, report_time, subject_name
            ),
            "report_number": report_number,
            "report_time": report_time,
            "subject_name": subject_name,
            "primary_id_type": primary_id_type,
            "primary_id_number": primary_id_number,
            "marital_status": marital_status,
            "marital_status_raw": marital_status_raw,
            **amount_policy,
            "source": "personal_brief_header",
            "source_refs": _source_refs(page, "native_text_header"),
            "confidence": 0.99,
        }
    ]
    return identity_documents, metadata, amount_policy


def _personal_summary_records(
    parse_result: Any,
    text: str = "",
    *,
    expected_account_count: int | None = None,
) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.business_records import (
        _linear,
        _personal_brief_summary_from_canonical_tables,
        _source_refs,
        _stable_id,
    )

    summary = _personal_brief_summary_from_canonical_tables(
        parse_result,
        text,
        expected_account_count=expected_account_count,
    )
    page = int(summary.get("source_summary_page") or 1)
    table_id = str(summary.get("source_summary_table_id") or "")
    summary_refs = _source_refs(page, "canonical_summary_table")
    if table_id:
        summary_refs[0]["table_id"] = table_id
        for node in getattr(getattr(parse_result, "document_flow", None), "nodes", None) or []:
            metadata = dict(getattr(node, "metadata", None) or {})
            if str(metadata.get("table_id") or "") != table_id:
                continue
            bbox = list(getattr(node, "bbox", None) or [])
            evidence_ids = [
                str(value)
                for value in (getattr(node, "evidence_refs", None) or [])
                if value
            ]
            if len(bbox) >= 4:
                summary_refs[0]["bbox"] = bbox[:4]
            if evidence_ids:
                summary_refs[0]["evidence_ids"] = evidence_ids
            break
    records: list[dict[str, Any]] = []
    sequence = 0
    grouped = {
        "account_count": summary.get("source_account_counts"),
        "unclosed_account_count": summary.get("source_unclosed_account_counts"),
        "ever_overdue_account_count": summary.get("source_overdue_account_counts"),
        "over_90_days_account_count": summary.get("source_over_90_days_account_counts"),
    }
    source_anchors = {
        "account_count": "账户数",
        "unclosed_account_count": "未结清/未销户账户数",
        "ever_overdue_account_count": "发生过逾期的账户数",
        "over_90_days_account_count": "发生过90天以上逾期的账户数",
        "asset_disposition_count": "资产处置信息",
        "guarantor_compensation_count": "垫款信息",
        "personal_repayment_liability_count": "相关还款责任账户数",
        "enterprise_repayment_liability_count": "相关还款责任账户数",
    }
    for metric, values in grouped.items():
        if not isinstance(values, dict):
            continue
        for category, value in values.items():
            sequence += 1
            records.append(
                {
                    "credit_summary_record_id": _stable_id(
                        "personal_credit_summary", metric, category
                    ),
                    "sequence": sequence,
                    "summary_scope": "source_reported",
                    "metric": metric,
                    "business_category": category,
                    "value": value,
                    "reporting_status": "reported" if value is not None else "not_reported",
                    "source_anchor": source_anchors[metric],
                    "source": "personal_brief_summary_table",
                    "source_refs": [dict(ref) for ref in summary_refs],
                    "evidence_ids": list(summary_refs[0].get("evidence_ids") or []),
                    "confidence": 1.0,
                }
            )
    singleton_metrics = {
        "asset_disposition_count": summary.get("source_asset_disposition_count"),
        "guarantor_compensation_count": summary.get("source_guarantor_compensation_count"),
        "personal_repayment_liability_count": summary.get("source_personal_liability_count"),
        "enterprise_repayment_liability_count": summary.get("source_enterprise_liability_count"),
    }
    present_singletons: set[str] = set()
    for source_page in getattr(parse_result, "pages", None) or []:
        for source_table in getattr(source_page, "tables", None) or []:
            headers = {
                re.sub(r"\s+", "", str(value or ""))
                for value in (getattr(source_table, "headers", None) or [])
            }
            raw_rows = (getattr(source_table, "metadata", None) or {}).get("raw_rows") or []
            for raw_row in raw_rows:
                cells = [re.sub(r"\s+", "", str(cell or "")) for cell in raw_row]
                if cells[:1] == ["账户数"] and len(cells) >= 3 and {
                    "资产处置信息",
                    "垫款信息",
                } <= headers:
                    present_singletons.update(
                        {"asset_disposition_count", "guarantor_compensation_count"}
                    )
                elif cells[:1] == ["相关还款责任账户数"] and len(cells) >= 3:
                    present_singletons.update(
                        {
                            "personal_repayment_liability_count",
                            "enterprise_repayment_liability_count",
                        }
                    )
    for metric, value in singleton_metrics.items():
        if value is None and metric not in present_singletons:
            continue
        sequence += 1
        records.append(
            {
                "credit_summary_record_id": _stable_id("personal_credit_summary", metric, "all"),
                "sequence": sequence,
                "summary_scope": "source_reported",
                "metric": metric,
                "business_category": "all",
                "value": value,
                "reporting_status": "reported" if value is not None else "not_reported",
                "source_anchor": source_anchors[metric],
                "source": "personal_brief_summary_table",
                "source_refs": [dict(ref) for ref in summary_refs],
                "evidence_ids": list(summary_refs[0].get("evidence_ids") or []),
                "confidence": 1.0,
            }
        )
    return records


def _derived_personal_summary_records(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provide an explicit derived view when a batch parser cannot expose the grid."""
    from docmirror.plugins.credit_report.business_records import _stable_id

    def category(account: dict[str, Any]) -> str:
        if account.get("account_type") == "credit_card":
            return "credit_card"
        business_type = str(account.get("business_type") or "")
        if any(marker in business_type for marker in ("住房", "商用房", "公积金")):
            return "housing_loan"
        return "other_loan"

    predicates = {
        "account_count": lambda account: True,
        "unclosed_account_count": lambda account: account.get("account_lifecycle_state") == "open",
        "ever_overdue_account_count": lambda account: account.get("ever_overdue") is True,
        "over_90_days_account_count": lambda account: account.get("over_90_days") is True,
    }
    categories = ("credit_card", "housing_loan", "other_loan", "other_business")
    records: list[dict[str, Any]] = []
    for metric, predicate in predicates.items():
        for business_category in categories:
            value = sum(
                predicate(account) and category(account) == business_category
                for account in accounts
            )
            records.append(
                {
                    "credit_summary_record_id": _stable_id(
                        "personal_credit_summary_derived", metric, business_category
                    ),
                    "sequence": len(records) + 1,
                    "summary_scope": "derived_from_account_records",
                    "metric": metric,
                    "business_category": business_category,
                    "value": value,
                    "reporting_status": "derived",
                    "source": "personal_brief_account_projection",
                    "source_refs": [],
                    "confidence": 1.0,
                }
            )
    return records


def _asset_and_compensation_records(
    parse_result: Any,
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from docmirror.plugins.credit_report.business_records import (
        _compact,
        _iso_date,
        _iso_month,
        _number,
        _page_texts,
        _source_page,
        _source_refs,
        _stable_id,
    )

    compact = _compact(text)
    pages = _page_texts(parse_result)
    assets: list[dict[str, Any]] = []
    asset_start = compact.find("资产处置信息")
    asset_end = compact.find("垫款信息", asset_start + 1) if asset_start >= 0 else -1
    asset_section = compact[asset_start:asset_end] if asset_start >= 0 and asset_end > asset_start else ""
    for sequence, match in enumerate(
        re.finditer(
            r"(\d+)\.(20\d{2}年\d{1,2}月\d{1,2}日)，?(.+?)接收债权，?"
            r"金额为?([\d,.]+)。截至(20\d{2}年\d{1,2}月\d{1,2}日)，?"
            r"余额为?([\d,.]+)，?最近一次还款日期为?(20\d{2}年\d{1,2}月\d{1,2}日)",
            asset_section,
        ),
        start=1,
    ):
        page = _source_page(pages, match.group(2) + match.group(3))
        assets.append(
            {
                "asset_disposition_id": _stable_id("asset_disposition", *match.groups()),
                "sequence": int(match.group(1) or sequence),
                "disposition_date": _iso_date(match.group(2)),
                "asset_management_company": match.group(3),
                "received_debt_amount": _number(match.group(4)),
                "snapshot_date": _iso_date(match.group(5)),
                "balance": _number(match.group(6)),
                "last_repayment_date": _iso_date(match.group(7)),
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_asset_disposition",
                "source_refs": _source_refs(page, "native_text_narrative"),
                "confidence": 0.97,
            }
        )
    compensations: list[dict[str, Any]] = []
    comp_start = compact.find("垫款信息")
    comp_end = compact.find("信用卡", comp_start + 1) if comp_start >= 0 else -1
    comp_section = compact[comp_start:comp_end] if comp_start >= 0 and comp_end > comp_start else ""
    for sequence, match in enumerate(
        re.finditer(
            r"(\d+)\.(20\d{2}年\d{1,2}月\d{1,2}日)以来(.+?)累计代偿金额([\d,.]+)。"
            r"(?:(20\d{2}年\d{1,2}月)(已结清))?",
            comp_section,
        ),
        start=1,
    ):
        page = _source_page(pages, match.group(2) + match.group(3))
        compensations.append(
            {
                "guarantor_compensation_id": _stable_id("guarantor_compensation", *match.groups()),
                "sequence": int(match.group(1) or sequence),
                "compensation_start_date": _iso_date(match.group(2)),
                "guarantor": match.group(3),
                "cumulative_compensation_amount": _number(match.group(4)),
                "settlement_date": _iso_month(match.group(5) or ""),
                "settlement_state": "settled" if match.group(6) else "not_reported",
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_guarantor_compensation",
                "source_refs": _source_refs(page, "native_text_narrative"),
                "confidence": 0.97,
            }
        )
    return assets, compensations


def _postpaid_records(blocks: list[tuple[int, str]]) -> list[dict[str, Any]]:
    from docmirror.plugins.credit_report.business_records import (
        _iso_date,
        _iso_month,
        _number,
        _source_refs,
        _stable_id,
    )

    records: list[dict[str, Any]] = []
    for index, (page, content) in enumerate(blocks):
        if "机构名称" not in content:
            continue
        preceding = "\n".join(value for _p, value in blocks[max(0, index - 4) : index])
        if "后付费记录" not in preceding and records == []:
            continue
        parts = [content]
        end_page = page
        for next_page, candidate in blocks[index + 1 :]:
            compact_candidate = re.sub(r"\s+", "", candidate)
            if (
                next_page > page + 1
                or "机构名称" in candidate
                or any(marker in compact_candidate for marker in ("公共记录", "查询记录", "欠税记录", "民事判决记录"))
            ):
                break
            if "第" in compact_candidate and "页" in compact_candidate and len(compact_candidate) < 20:
                continue
            parts.append(candidate)
            end_page = next_page
            if "当前欠费金额" in candidate:
                break
        fields = _label_fields(
            "\n".join(parts),
            ("机构名称", "业务类型", "记账年月", "业务开通日期", "当前缴费状态", "当前欠费金额"),
        )
        if not fields.get("机构名称") or not fields.get("业务类型"):
            continue
        sequence = len(records) + 1
        records.append(
            {
                "postpaid_record_id": _stable_id(
                    "postpaid", fields.get("机构名称"), fields.get("业务类型"), fields.get("记账年月")
                ),
                "sequence": sequence,
                "institution": fields.get("机构名称"),
                "business_type": fields.get("业务类型"),
                "billing_month": _iso_month(fields.get("记账年月", "")),
                "service_start_date": _iso_date(fields.get("业务开通日期", "")),
                "payment_status": _reported_text(fields.get("当前缴费状态")),
                "current_arrears_amount": _number(fields.get("当前欠费金额", "")),
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_postpaid_record",
                "source_refs": [
                    *_source_refs(page, "native_text_labeled_record"),
                    *(
                        _source_refs(end_page, "native_text_labeled_record_continuation")
                        if end_page != page
                        else []
                    ),
                ],
                "confidence": 0.98,
            }
        )
    return records


def _personal_public_records(
    parse_result: Any,
    blocks: list[tuple[int, str]],
) -> dict[str, list[dict[str, Any]]]:
    from docmirror.plugins.credit_report.business_records import (
        _number,
        _source_refs,
        _stable_id,
    )

    tax_tables = _table_chunks(parse_result, "欠税总额")
    judgment_tables = _table_chunks(parse_result, "诉讼标的")
    enforcement_tables = _table_chunks(parse_result, "申请执行标的")
    penalty_tables = _table_chunks(parse_result, "处罚内容")
    tax_anchors = [(page, value) for page, value in blocks if "主管税务机关" in value]
    judgment_anchors = [(page, value) for page, value in blocks if "立案法院" in value and "案号" in value]
    enforcement_positions = [
        index for index, (_page, value) in enumerate(blocks) if "执行法院" in value and "案号" in value
    ]
    penalty_anchors = [(page, value) for page, value in blocks if "处罚机构" in value and "文书编号" in value]
    paired_taxes = _pair_record_chunks(
        tax_anchors,
        tax_tables,
        identity_labels=("主管税务机关", "纳税人识别号"),
    )
    paired_judgments = _pair_record_chunks(
        judgment_anchors,
        judgment_tables,
        identity_labels=("立案法院", "案号"),
    )
    enforcement_anchors = [blocks[position] for position in enforcement_positions]
    paired_enforcements = _pair_record_chunks(
        enforcement_anchors,
        enforcement_tables,
        identity_labels=("执行法院", "案号"),
    )
    paired_penalties = _pair_record_chunks(
        penalty_anchors,
        penalty_tables,
        identity_labels=("处罚机构", "文书编号"),
    )

    taxes: list[dict[str, Any]] = []
    for index, (page, anchor) in enumerate(tax_anchors):
        detail_page, detail = paired_taxes[index]
        fields = _label_fields(
            anchor + "\n" + detail,
            ("主管税务机关", "欠税统计日期", "欠税总额", "纳税人识别号"),
        )
        taxes.append(
            {
                "tax_arrears_id": _stable_id("tax_arrears", fields.get("纳税人识别号"), fields.get("欠税统计日期")),
                "sequence": index + 1,
                "tax_authority": fields.get("主管税务机关"),
                "statistics_date": _date_or_month(fields.get("欠税统计日期", "")),
                "arrears_amount": _number(fields.get("欠税总额", "")),
                "taxpayer_identifier": _reported_text(fields.get("纳税人识别号")),
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_public_record",
                "source_refs": _public_record_refs(page, detail_page),
                "confidence": 0.98,
            }
        )

    judgments: list[dict[str, Any]] = []
    judgment_labels = (
        "立案法院", "案号", "案由", "结案方式", "立案日期", "判决/调解结果",
        "诉讼标的", "判决/调解生效日期", "诉讼标的金额",
    )
    for index, (page, anchor) in enumerate(judgment_anchors):
        detail_page, detail = paired_judgments[index]
        fields = _label_fields(anchor + "\n" + detail, judgment_labels)
        cause = _reported_text(fields.get("案由"))
        judgments.append(
            {
                "civil_judgment_id": _stable_id("civil_judgment", fields.get("立案法院"), fields.get("案号")),
                "sequence": index + 1,
                "filing_court": fields.get("立案法院"),
                "case_number": fields.get("案号"),
                "cause": cause,
                "cause_status": "reported" if cause else "not_reported",
                "filing_date": _date_or_month(fields.get("立案日期", "")),
                "closure_method": _reported_text(fields.get("结案方式")),
                "claim_subject": _reported_text(fields.get("诉讼标的")),
                "claim_amount": _number(fields.get("诉讼标的金额", "")),
                "judgment_result": _reported_text(fields.get("判决/调解结果")),
                "judgment_effective_date": _date_or_month(fields.get("判决/调解生效日期", "")),
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_public_record",
                "source_refs": _public_record_refs(page, detail_page),
                "confidence": 0.97,
            }
        )

    enforcements: list[dict[str, Any]] = []
    enforcement_labels = (
        "执行法院", "案号", "执行案由", "结案方式", "立案日期", "案件状态",
        "申请执行标的", "已执行标的", "申请执行标的金额", "已执行标的金额", "结案日期",
    )
    for index, position in enumerate(enforcement_positions):
        page, anchor = blocks[position]
        following: list[str] = []
        for next_page, candidate in blocks[position + 1 :]:
            if ("执行法院" in candidate and "案号" in candidate) or "行政处罚记录" in candidate:
                break
            if "第" in candidate and "页" in candidate and len(re.sub(r"\s+", "", candidate)) < 20:
                continue
            following.append(candidate)
        detail_page, table_detail = paired_enforcements[index]
        fields = _label_fields(
            "\n".join([anchor, table_detail, *following]),
            enforcement_labels,
        )
        cause = _reported_text(fields.get("执行案由"))
        enforcements.append(
            {
                "enforcement_record_id": _stable_id("enforcement", fields.get("执行法院"), fields.get("案号")),
                "sequence": index + 1,
                "court": fields.get("执行法院"),
                "case_number": fields.get("案号"),
                "cause": cause,
                "cause_status": "reported" if cause else "not_reported",
                "filing_date": _date_or_month(fields.get("立案日期", "")),
                "case_status": _reported_text(fields.get("案件状态")),
                "closure_method": _reported_text(fields.get("结案方式")),
                "closure_date": _date_or_month(fields.get("结案日期", "")),
                "requested_subject": _reported_text(fields.get("申请执行标的")),
                "requested_amount": _number(fields.get("申请执行标的金额", "")),
                "executed_subject": _reported_text(fields.get("已执行标的")),
                "executed_amount": _number(fields.get("已执行标的金额", "")),
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_public_record",
                "source_refs": _public_record_refs(page, detail_page),
                "confidence": 0.96,
            }
        )

    penalties: list[dict[str, Any]] = []
    penalty_labels = (
        "处罚机构", "文书编号", "处罚内容", "行政复议结果", "处罚金额", "生效日期", "截止日期",
    )
    for index, (page, anchor) in enumerate(penalty_anchors):
        detail_page, detail = paired_penalties[index]
        fields = _label_fields(anchor + "\n" + detail, penalty_labels)
        review_result = _reported_text(fields.get("行政复议结果"))
        penalties.append(
            {
                "administrative_penalty_id": _stable_id(
                    "administrative_penalty", fields.get("处罚机构"), fields.get("文书编号")
                ),
                "sequence": index + 1,
                "authority": fields.get("处罚机构"),
                "document_number": fields.get("文书编号"),
                "penalty_content": _reported_text(fields.get("处罚内容")),
                "penalty_amount": _number(fields.get("处罚金额", "")),
                "effective_date": _date_or_month(fields.get("生效日期", "")),
                "end_date": _date_or_month(fields.get("截止日期", "")),
                "administrative_review_result": review_result,
                "administrative_review_result_status": "reported" if review_result else "not_reported",
                "reporting_amount_currency": "CNY",
                "reporting_amount_unit": "yuan",
                "source": "personal_brief_public_record",
                "source_refs": _public_record_refs(page, detail_page),
                "confidence": 0.98,
            }
        )

    public_records: list[dict[str, Any]] = []
    typed = (
        ("tax_arrears", taxes, "tax_arrears_id", "tax_authority"),
        ("civil_judgment", judgments, "civil_judgment_id", "filing_court"),
        ("enforcement", enforcements, "enforcement_record_id", "court"),
        ("administrative_penalty", penalties, "administrative_penalty_id", "authority"),
    )
    for record_type, records, id_key, authority_key in typed:
        for record in records:
            public_records.append(
                {
                    "public_record_id": _stable_id("public_record", record_type, record.get(id_key)),
                    "sequence": len(public_records) + 1,
                    "record_type": record_type,
                    "authority": record.get(authority_key),
                    "content": {
                        key: value
                        for key, value in record.items()
                        if key not in {id_key, "source", "source_refs", "confidence"}
                    },
                    "source": record.get("source"),
                    "source_refs": record.get("source_refs"),
                    "confidence": record.get("confidence"),
                }
            )
    return {
        "tax_arrears_records": taxes,
        "civil_judgment_records": judgments,
        "enforcement_records": enforcements,
        "administrative_penalty_records": penalties,
        "public_records": public_records,
    }


def extract_personal_brief_native_business(
    parse_result: Any,
    full_text: str,
) -> dict[str, Any]:
    """Transform a ParseResult into personal-brief business candidates."""
    from docmirror.plugins.credit_report.business_records import (
        _linear,
        _overdue_from_personal_brief_accounts,
        _page_texts,
        _personal_brief_accounts,
        _personal_brief_credit_lines,
        _personal_brief_inquiries,
        _personal_brief_repayment_liabilities,
        _personal_brief_summary_from_canonical_tables,
        _personal_brief_text,
    )

    text = _personal_brief_text(parse_result, full_text)
    summary_text = _summary_source_text(parse_result, text, full_text)
    page_texts = _page_texts(parse_result)
    accounts = _personal_brief_accounts(text, page_texts)
    liabilities = _personal_brief_repayment_liabilities(text, page_texts)
    inquiries = _personal_brief_inquiries(parse_result, text, page_texts)
    overdue = _overdue_from_personal_brief_accounts(accounts)
    credit_lines = _personal_brief_credit_lines(accounts)
    source_summary = _personal_brief_summary_from_canonical_tables(
        parse_result,
        summary_text,
        expected_account_count=len(accounts) if accounts else None,
    )
    return {
        "credit_accounts": accounts,
        "credit_lines": credit_lines,
        "repayment_liability_records": liabilities,
        "overdue_records": overdue,
        "inquiry_records": inquiries,
        "credit_summary": {
            "source": "personal_brief_native_text",
            "account_count": len(accounts),
            "active_account_count": sum(account.get("account_status") == "active" for account in accounts),
            "active_account_count_basis": "legacy_compatibility_status_active",
            "unclosed_account_count": sum(
                account.get("account_lifecycle_state") == "open" for account in accounts
            ),
            "activated_credit_card_account_count": sum(
                account.get("card_activation_state") == "activated"
                for account in accounts
            ),
            "inactive_credit_card_account_count": sum(
                account.get("card_activation_state") == "not_activated"
                for account in accounts
            ),
            "settled_account_count": sum(
                account.get("termination_event_type") == "debt_settled" for account in accounts
            ),
            "closed_credit_card_account_count": sum(
                account.get("termination_event_type") == "account_closed" for account in accounts
            ),
            "transferred_out_account_count": sum(
                account.get("termination_event_type") == "transferred_out" for account in accounts
            ),
            "derived_ever_overdue_account_count": len(overdue),
            "repayment_liability_count": len(liabilities),
            "inquiry_count": len(inquiries),
            "institution_inquiry_count": sum(item.get("inquiry_type") == "institution" for item in inquiries),
            "personal_inquiry_count": sum(item.get("inquiry_type") == "personal" for item in inquiries),
            **source_summary,
        },
    }


def extract_personal_brief_section_content(
    parse_result: Any,
    full_text: str,
) -> dict[str, Any]:
    """Return personal-brief-only facts and supplemental records."""
    from docmirror.plugins.credit_report.business_records import (
        _compact,
        _linear,
        _page_texts,
        _personal_brief_accounts,
        _personal_brief_text,
        _source_page,
        _source_refs,
        _stable_id,
    )

    blocks = _personal_brief_blocks(parse_result)

    def statement_after(heading: str) -> tuple[int, str]:
        for index, (page, content) in enumerate(blocks):
            if _compact(content) != heading:
                continue
            for next_page, candidate in blocks[index + 1 :]:
                if next_page != page:
                    break
                compact = _compact(candidate)
                if compact and not re.fullmatch(r"第\d+页，共\d+页", compact):
                    return page, re.sub(r"\s+", " ", candidate).strip()
        text = _linear(full_text)
        marker = text.find(heading)
        if marker < 0:
            return 0, ""
        remainder = text[marker + len(heading) :]
        end = min(
            [
                position
                for value in ("非信贷交易记录", "公共记录", "查询记录", "说明")
                if (position := remainder.find(value)) >= 0
            ]
            or [len(remainder)]
        )
        return _source_page(_page_texts(parse_result), heading), remainder[:end].strip()

    non_credit_page, non_credit_statement = statement_after("非信贷交易记录")
    public_page, public_statement = statement_after("公共记录")
    notes: list[dict[str, Any]] = []
    for index, (page, content) in enumerate(blocks):
        if _compact(content) != "说明":
            continue
        note_text = "\n".join(
            candidate
            for next_page, candidate in blocks[index + 1 :]
            if next_page == page and not re.fullmatch(r"\s*第\s*\d+\s*页，共\s*\d+\s*页\s*", candidate)
        )
        for match in re.finditer(r"(?ms)(\d+)\.\s*(.*?)(?=^\d+\.|\Z)", note_text):
            sequence = int(match.group(1))
            content_value = re.sub(r"\s+", " ", match.group(2)).strip()
            if not content_value:
                continue
            notes.append(
                {
                    "note_id": _stable_id("credit_report_note", sequence, content_value),
                    "sequence": sequence,
                    "content": content_value,
                    "source": "personal_brief_notes",
                    "source_refs": _source_refs(page, "native_text_note"),
                    "confidence": 1.0,
                }
            )
        break
    text = _personal_brief_text(parse_result, full_text)
    summary_accounts = _personal_brief_accounts(text, _page_texts(parse_result))
    summary_account_count = len(summary_accounts)
    identity_documents, report_metadata, amount_policy = _personal_header_datasets(parse_result, blocks)
    asset_dispositions, guarantor_compensations = _asset_and_compensation_records(parse_result, text)
    public_datasets = _personal_public_records(parse_result, blocks)
    summary_records = _personal_summary_records(
        parse_result,
        _summary_source_text(parse_result, text, full_text),
        expected_account_count=summary_account_count or None,
    )
    if not summary_records and summary_accounts:
        summary_records = _derived_personal_summary_records(summary_accounts)
    datasets: dict[str, list[dict[str, Any]]] = {
        "identity_documents": identity_documents,
        "personal_report_metadata": report_metadata,
        "personal_credit_summary_records": summary_records,
        "asset_disposition_records": asset_dispositions,
        "guarantor_compensation_records": guarantor_compensations,
        "postpaid_records": _postpaid_records(blocks),
        **public_datasets,
    }
    return {
        "facts": {
            "reporting_context": amount_policy,
        },
        "non_credit_transaction_summary": {
            "record_status": "no_records" if "没有" in non_credit_statement else "reported",
            "lookback_years": 5 if "5年" in non_credit_statement else None,
            "source_statement": non_credit_statement,
            "source_page": non_credit_page or None,
        },
        "public_record_summary": {
            "record_status": "no_records" if "没有" in public_statement else "reported",
            "lookback_years": 5 if "5年" in public_statement else None,
            "source_statement": public_statement,
            "source_page": public_page or None,
        },
        "report_notes": notes,
        "datasets": datasets,
    }


__all__ = [
    "extract_personal_brief_native_business",
    "extract_personal_brief_section_content",
]
