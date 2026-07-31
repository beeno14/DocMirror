# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Community-semantic metadata and provenance enrichment for credit reports."""

from __future__ import annotations

import re
from typing import Any

from docmirror.plugins.credit_report.contracts import (
    CONTENT_MODE_MIXED,
    CONTENT_MODE_NATIVE,
    CONTENT_MODE_SCANNED,
)
from docmirror.plugins.credit_report.currency_codes import CURRENCY_CODE_BY_ALIAS
from docmirror.plugins.credit_report.value_utils import compact_text as _compact

_INSTITUTION_INQUIRY_SECTION = "机构查询记录明细"
_PERSONAL_INQUIRY_SECTION = "个人查询记录明细"
_INQUIRY_SECTION_END = "说明"
_INQUIRY_ROW_RE = re.compile(
    r"^(?P<sequence>\d{1,4})(?P<date>20\d{2}年\d{1,2}月\d{1,2}日)"
)
_PERSONAL_ACCOUNT_SOURCES = frozenset(
    {
        "personal_brief_narrative",
        "personal_brief_account_narrative",
    }
)
_PERSONAL_INQUIRY_SOURCE = "personal_brief_inquiry_ledger"


def _date_anchors(value: Any) -> list[str]:
    match = re.fullmatch(r"(20\d{2})-(\d{2})(?:-(\d{2}))?", str(value or ""))
    if not match:
        return []
    compact = f"{match.group(1)}年{int(match.group(2))}月"
    padded = f"{match.group(1)}年{match.group(2)}月"
    if match.group(3):
        compact += f"{int(match.group(3))}日"
        padded += f"{match.group(3)}日"
    return list(dict.fromkeys((compact, padded)))


def _record_anchors(record: dict[str, Any]) -> list[str]:
    raw_normalized = record.get("normalized")
    normalized: dict[str, Any] = raw_normalized if isinstance(raw_normalized, dict) else {}
    values = {**record, **normalized}
    anchors: list[str] = []
    date_value = next(
        (
            values.get(key)
            for key in ("inquiry_date", "liability_date", "open_date", "snapshot_date")
            if values.get(key)
        ),
        "",
    )
    date_anchors = _date_anchors(date_value)
    sequence = values.get("sequence")
    if sequence not in (None, ""):
        anchors.extend(f"{sequence}{date_anchor}" for date_anchor in date_anchors)
    for key in (
        "contract_number",
        "account_identifier",
        "related_party_id_number",
        "card_tail",
        "institution",
        "management_institution",
        "related_party_name",
        "source_anchor",
        "report_number",
        "document_number",
        "primary_id_number",
        "holder_name",
        "taxpayer_identifier",
        "case_number",
        "asset_management_company",
        "guarantor",
        "tax_authority",
        "filing_court",
        "court",
        "authority",
        "content",
    ):
        value = _compact(values.get(key))
        if len(value) >= 4:
            anchors.append(value[:80])
    anchors.extend(date_anchors)
    return list(dict.fromkeys(anchors))


def _node_payloads(parse_result: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    graph = getattr(parse_result, "document_flow", None)
    payloads: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for node in getattr(graph, "nodes", None) or []:
        node_id = str(getattr(node, "node_id", "") or "")
        if not node_id:
            continue
        payload = {
            "node_id": node_id,
            "page": int(getattr(node, "page", 0) or 0),
            "text": _compact(getattr(node, "text", "")),
            "bbox": list(getattr(node, "bbox", None) or []),
            "evidence_ids": [str(value) for value in (getattr(node, "evidence_refs", None) or []) if value],
        }
        payloads.append(payload)
        by_id[node_id] = payload
    return payloads, by_id


def _ordered_node_payloads(
    parse_result: Any,
    nodes: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    graph = getattr(parse_result, "document_flow", None)
    reading_flows = list(getattr(graph, "reading_flow", None) or [])
    if not reading_flows:
        return nodes
    ordered = [
        nodes_by_id[str(node_id)]
        for node_id in (getattr(reading_flows[0], "node_ids", None) or [])
        if str(node_id) in nodes_by_id
    ]
    return ordered or nodes


def _referenced_node_ids(refs: list[Any]) -> list[str]:
    ids: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        ids.extend(str(value) for value in [ref.get("node_id"), *(ref.get("node_ids") or [])] if value)
    return list(dict.fromkeys(ids))


def _record_values(record: dict[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    return {**record, **(normalized if isinstance(normalized, dict) else {})}


def _semantic_text(value: Any) -> str:
    return _compact(value).replace("(", "（").replace(")", "）")


def _subsequence_coverage(needle: str, haystack: str) -> int:
    if not needle:
        return 0
    position = 0
    for character in haystack:
        if position < len(needle) and character == needle[position]:
            position += 1
    return position


def _claims_covered(claims: list[str], nodes: list[dict[str, Any]]) -> bool:
    text = "".join(_semantic_text(node["text"]) for node in nodes)
    return all(_subsequence_coverage(claim, text) == len(claim) for claim in claims if claim)


def _minimal_claim_nodes(
    anchor: dict[str, Any],
    tail: list[dict[str, Any]],
    claims: list[str],
) -> list[dict[str, Any]]:
    selected = [anchor]
    current_text = _semantic_text(anchor["text"])
    current_score = sum(_subsequence_coverage(claim, current_text) for claim in claims)
    for node in tail:
        candidate_text = current_text + _semantic_text(node["text"])
        candidate_score = sum(_subsequence_coverage(claim, candidate_text) for claim in claims)
        if candidate_score > current_score:
            selected.append(node)
            current_text = candidate_text
            current_score = candidate_score
    return selected if _claims_covered(claims, selected) else []


def _inquiry_section_rows(
    nodes: list[dict[str, Any]],
    inquiry_type: str,
) -> list[list[dict[str, Any]]]:
    start_marker = _PERSONAL_INQUIRY_SECTION if inquiry_type == "personal" else _INSTITUTION_INQUIRY_SECTION
    end_markers = (
        (_INQUIRY_SECTION_END,)
        if inquiry_type == "personal"
        else (_PERSONAL_INQUIRY_SECTION, _INQUIRY_SECTION_END)
    )
    section_start = next(
        (index for index, node in enumerate(nodes) if start_marker in _semantic_text(node["text"])),
        -1,
    )
    if section_start < 0:
        return []
    section_end = next(
        (
            index
            for index, node in enumerate(nodes[section_start + 1 :], start=section_start + 1)
            if any(_semantic_text(node["text"]).startswith(marker) for marker in end_markers)
        ),
        len(nodes),
    )
    starts = [
        index
        for index in range(section_start + 1, section_end)
        if _INQUIRY_ROW_RE.match(_semantic_text(nodes[index]["text"]))
    ]
    return [
        nodes[start : starts[index + 1] if index + 1 < len(starts) else section_end]
        for index, start in enumerate(starts)
    ]


def _personal_inquiry_evidence(
    record: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    values = _record_values(record)
    inquiry_type = str(values.get("inquiry_type") or "")
    if inquiry_type not in {"institution", "personal"}:
        return []
    sequence = values.get("sequence")
    date_anchors = _date_anchors(values.get("inquiry_date"))
    institution = _semantic_text(values.get("institution"))
    reason = _semantic_text(values.get("source_reason") or values.get("reason"))
    if sequence in (None, "") or not date_anchors or not institution or not reason:
        return []

    for row in _inquiry_section_rows(nodes, inquiry_type):
        anchor_text = _semantic_text(row[0]["text"])
        if not any(anchor_text.startswith(f"{sequence}{date_anchor}") for date_anchor in date_anchors):
            continue
        chosen = _minimal_claim_nodes(row[0], row[1:], [institution, reason])
        if chosen:
            return chosen
    return []


def _number_anchor(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(",", "").replace("，", "")
    return text[:-2] if text.endswith(".0") else text


def _account_currency_anchors(code: Any) -> tuple[str, ...]:
    normalized = str(code or "").upper()
    aliases = [alias for alias, target in CURRENCY_CODE_BY_ALIAS.items() if target == normalized]
    return tuple(dict.fromkeys([*(f"{alias}账户" for alias in aliases), normalized]))


def _account_candidate_matches(record: dict[str, Any], nodes: list[dict[str, Any]]) -> bool:
    values = _record_values(record)
    text = _semantic_text("".join(node["text"] for node in nodes))
    numeric_text = text.replace(",", "").replace("，", "")
    date_anchors = _date_anchors(values.get("open_date"))
    institution = _semantic_text(values.get("institution") or values.get("management_institution"))
    if not date_anchors or not any(anchor in text for anchor in date_anchors):
        return False
    if institution and institution not in text:
        return False
    if "发放的" not in text and "贷款授信" not in text:
        return False

    currency = str(values.get("currency") or "").upper()
    account_type = str(values.get("account_type") or "")
    if currency and (currency != "CNY" or account_type == "credit_card"):
        currency_anchors = _account_currency_anchors(currency)
        if not currency_anchors or not any(anchor in text for anchor in currency_anchors):
            return False
    card_tail = _semantic_text(values.get("card_tail"))
    if card_tail and card_tail not in text:
        return False
    primary_amount = next(
        (
            _number_anchor(values.get(key))
            for key in ("credit_limit", "loan_amount")
            if values.get(key) not in (None, "")
        ),
        "",
    )
    if primary_amount and primary_amount not in numeric_text:
        return False
    return True


def _record_pages(refs: list[dict[str, Any]]) -> set[int]:
    return {
        int(ref.get("page") or ref.get("logical_page") or 0)
        for ref in refs
        if int(ref.get("page") or ref.get("logical_page") or 0) > 0
    }


def _personal_account_evidence(
    record: dict[str, Any],
    refs: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pages = _record_pages(refs)
    candidates = [node for node in nodes if not pages or node["page"] in pages]
    matches = [[node] for node in candidates if _account_candidate_matches(record, [node])]
    if matches:
        return matches[0]

    for width in range(2, 7):
        for index in range(0, len(candidates) - width + 1):
            window = candidates[index : index + width]
            if window[-1]["page"] - window[0]["page"] > 1:
                continue
            if _account_candidate_matches(record, window):
                return window
    if pages:
        return _personal_account_evidence(record, [], nodes)
    return []


def _is_authoritative_personal_record(collection: str, record: dict[str, Any]) -> bool:
    source = str(record.get("source") or "")
    return (
        collection in {"credit_accounts", "overdue_records"}
        and source in _PERSONAL_ACCOUNT_SOURCES
    ) or (collection == "inquiry_records" and source == _PERSONAL_INQUIRY_SOURCE)


def _attach_record_evidence(
    record: dict[str, Any],
    refs: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
) -> list[str]:
    if not chosen:
        record["source_refs"] = refs
        return []
    target = refs[0]
    target.pop("node_id", None)
    target["node_ids"] = [node["node_id"] for node in chosen]
    if len(chosen) == 1:
        target["node_id"] = chosen[0]["node_id"]
    if chosen[0]["bbox"]:
        target["bbox"] = chosen[0]["bbox"]
    evidence_ids = list(
        dict.fromkeys(
            [
                *(str(value) for value in (record.get("evidence_ids") or []) if value),
                *(value for node in chosen for value in node["evidence_ids"]),
            ]
        )
    )
    if evidence_ids:
        target["evidence_ids"] = evidence_ids
        record["evidence_ids"] = evidence_ids
    record["source_refs"] = refs
    return evidence_ids


def enrich_credit_report_record_evidence(
    parse_result: Any,
    collections: dict[str, list[dict[str, Any]]],
) -> tuple[str, ...]:
    """Attach canonical source nodes, bboxes, and atom IDs to plugin records."""
    nodes, nodes_by_id = _node_payloads(parse_result)
    ordered_nodes = _ordered_node_payloads(parse_result, nodes, nodes_by_id)
    all_evidence_ids: list[str] = []
    personal_accounts: dict[str, list[dict[str, Any]]] = {}
    for collection, records in collections.items():
        for record in records:
            refs = [dict(ref) for ref in (record.get("source_refs") or []) if isinstance(ref, dict)]
            if not refs:
                refs = [{"source": str(record.get("source") or "credit_business_projection")}]
                if record.get("page"):
                    refs[0]["page"] = record["page"]
            if _is_authoritative_personal_record(collection, record):
                if collection == "inquiry_records":
                    chosen = _personal_inquiry_evidence(record, ordered_nodes)
                elif collection == "overdue_records":
                    account_id = str(_record_values(record).get("account_id") or "")
                    chosen = personal_accounts.get(account_id, [])
                    if not chosen:
                        chosen = _personal_account_evidence(record, refs, ordered_nodes)
                    overdue_id = str(_record_values(record).get("overdue_id") or "")
                    if overdue_id:
                        record["record_id"] = overdue_id
                else:
                    chosen = _personal_account_evidence(record, refs, ordered_nodes)
                    account_id = str(_record_values(record).get("account_id") or "")
                    if account_id and chosen:
                        personal_accounts[account_id] = chosen
                all_evidence_ids.extend(_attach_record_evidence(record, refs, chosen))
                continue
            referenced_ids = _referenced_node_ids(refs)
            chosen = [nodes_by_id[node_id] for node_id in referenced_ids if node_id in nodes_by_id]
            if not chosen:
                pages = {
                    int(ref.get("page") or ref.get("logical_page") or 0)
                    for ref in refs
                    if int(ref.get("page") or ref.get("logical_page") or 0) > 0
                }
                candidates = [node for node in nodes if not pages or node["page"] in pages]
                anchors = _record_anchors(record)
                scored: list[tuple[int, int, dict[str, Any]]] = []
                for order, node in enumerate(candidates):
                    score = sum(len(anchor) for anchor in anchors if anchor and anchor in node["text"])
                    if anchors and anchors[0] in node["text"]:
                        score += 1000
                    if score:
                        scored.append((score, -order, node))
                if scored:
                    chosen = [max(scored, key=lambda item: (item[0], item[1]))[2]]
            if chosen:
                target = refs[0]
                target["node_ids"] = [node["node_id"] for node in chosen]
                if len(chosen) == 1:
                    target["node_id"] = chosen[0]["node_id"]
                if chosen[0]["bbox"]:
                    target["bbox"] = chosen[0]["bbox"]
                evidence_ids = list(
                    dict.fromkeys(
                        [
                            *(str(value) for value in (record.get("evidence_ids") or []) if value),
                            *(value for node in chosen for value in node["evidence_ids"]),
                        ]
                    )
                )
                if evidence_ids:
                    target["evidence_ids"] = evidence_ids
                    record["evidence_ids"] = evidence_ids
                    all_evidence_ids.extend(evidence_ids)
            record["source_refs"] = refs
    return tuple(dict.fromkeys(all_evidence_ids))


def credit_report_data_dictionary() -> dict[str, Any]:
    """Return public labels, null semantics, identifiers, and aggregation rules."""

    def descriptor(
        label: str,
        *,
        type_: str = "string",
        definition: str = "",
        unit: str = "",
        sensitive: bool = False,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {"label": label, "type": type_}
        if definition:
            value["definition"] = definition
        if unit:
            value["unit"] = unit
        if sensitive:
            value.update({"sensitive": True, "display": "masked"})
        return value

    fields = {
        "subject_name": descriptor("姓名"),
        "subject_id": descriptor("主体证件号码", type_="long_id", sensitive=True),
        "id_number": descriptor("证件号码", type_="long_id", sensitive=True),
        "id_type": descriptor("证件类型"),
        "marital_status": descriptor(
            "婚姻状况",
            type_="enum",
            definition="被查询人在报告生成时记录的婚姻状况。",
        ),
        "report_number": descriptor("报告编号", type_="long_id", sensitive=True),
        "report_time": descriptor("报告时间", type_="datetime"),
        "report_subtype": descriptor("报告类型"),
        "content_mode": descriptor("内容模式"),
        "source": descriptor("数据来源"),
        "account_count": descriptor("提取账户数", type_="integer"),
        "active_account_count": descriptor(
            "旧版派生有效状态账户数",
            type_="integer",
            definition="兼容字段；混合了卡片激活和贷款未结清语义，不用于业务展示。",
        ),
        "active_account_count_basis": descriptor("旧版派生有效状态口径"),
        "unclosed_account_count": descriptor("派生未关闭账户数", type_="integer"),
        "activated_credit_card_account_count": descriptor("已激活信用卡账户数", type_="integer"),
        "inactive_credit_card_account_count": descriptor("尚未激活信用卡账户数", type_="integer"),
        "settled_account_count": descriptor("派生已结清债务账户数", type_="integer"),
        "closed_credit_card_account_count": descriptor("派生已销户信用卡账户数", type_="integer"),
        "transferred_out_account_count": descriptor("派生已转出账户数", type_="integer"),
        "derived_ever_overdue_account_count": descriptor("派生曾逾期账户数", type_="integer"),
        "source_account_counts": descriptor("各类账户数", type_="object"),
        "source_account_count": descriptor("账户总数", type_="integer"),
        "source_unclosed_account_counts": descriptor("各类未结清/未销户账户数", type_="object"),
        "source_unclosed_account_count": descriptor("未结清/未销户账户数", type_="integer"),
        "source_overdue_account_counts": descriptor("各类发生过逾期账户数", type_="object"),
        "source_overdue_account_count": descriptor("源表发生过逾期账户合计", type_="integer"),
        "source_overdue_account_count_status": descriptor("发生过逾期账户数", type_="enum"),
        "source_over_90_days_account_counts": descriptor("各类发生过90天以上逾期账户数", type_="object"),
        "source_over_90_days_account_count": descriptor("源表90天以上逾期账户合计", type_="integer"),
        "source_over_90_days_account_count_status": descriptor("发生过90天以上逾期账户数", type_="enum"),
        "repayment_liability_count": descriptor("相关还款责任记录数", type_="integer"),
        "inquiry_count": descriptor("查询记录总数", type_="integer"),
        "institution_inquiry_count": descriptor("机构查询记录数", type_="integer"),
        "personal_inquiry_count": descriptor("个人查询记录数", type_="integer"),
        "projected_account_count": descriptor("投影账户数", type_="integer"),
        "source_summary_table_id": descriptor("源概要表标识"),
        "source_summary_page": descriptor("源概要表页码", type_="integer"),
        "source_asset_disposition_count": descriptor("资产处置信息账户数", type_="integer"),
        "source_guarantor_compensation_count": descriptor("垫款信息账户数", type_="integer"),
        "source_personal_liability_count": descriptor("为个人承担相关还款责任账户数", type_="integer"),
        "source_enterprise_liability_count": descriptor("为企业承担相关还款责任账户数", type_="integer"),
        "document_type": descriptor("文档类型"),
        "page_count": descriptor("页数", type_="integer"),
        "source_file_name": descriptor("源文件名"),
        "source_file_sha256": descriptor("源文件校验值"),
        "record_status": descriptor("记录状态", type_="enum"),
        "lookback_years": descriptor("查询期间（年）", type_="integer"),
        "source_statement": descriptor("源文说明"),
        "source_page": descriptor("源页码", type_="integer"),
        "credit_card": descriptor("信用卡"),
        "housing_loan": descriptor("购房贷款"),
        "other_loan": descriptor("其他贷款"),
        "other_business": descriptor("其他业务"),
        "reporting_context": descriptor("报告金额口径", type_="object"),
    }
    common_account_columns = {
        "sequence": descriptor("组内序号", type_="integer"),
        "account_id": descriptor("账户记录ID"),
        "account_identifier": descriptor("账户标识", type_="long_id", sensitive=True),
        "account_type": descriptor("账户类型", type_="enum"),
        "institution": descriptor("管理机构"),
        "business_type": descriptor("业务类型"),
        "credit_card_type": descriptor("信用卡产品类型", type_="enum"),
        "card_tail": descriptor("卡片尾号", type_="long_id"),
        "open_date": descriptor("开立日期", type_="date"),
        "snapshot_date": descriptor("信息截至日期", type_="date"),
        "due_date": descriptor(
            "到期日期",
            type_="date",
            definition="仅表示合同到期日；不再承载额度有效期。",
        ),
        "contract_maturity_date": descriptor("合同到期日期", type_="date"),
        "credit_line_expiry_date": descriptor("授信额度有效期至", type_="date"),
        "credit_line_validity_type": descriptor("授信额度有效期类型", type_="enum"),
        "close_date": descriptor(
            "兼容结清/销户日期",
            type_="date",
            definition="兼容字段；精确业务语义见 termination_event_type/date。",
        ),
        "transfer_out_date": descriptor("转出日期", type_="date"),
        "termination_event_date": descriptor("账户终止事件日期", type_="date"),
        "termination_event_type": descriptor("账户终止事件类型", type_="enum"),
        "currency": descriptor(
            "币种",
            type_="currency",
            definition="兼容字段；个人简版中表示账户计价币种，不表示报告金额币种。",
        ),
        "account_currency": descriptor("账户计价币种", type_="currency"),
        "reporting_amount_currency": descriptor(
            "报告金额币种",
            type_="currency",
            definition="个人简版金额按源报告声明统一折算为人民币。",
        ),
        "amount_unit": descriptor("金额单位"),
        "reporting_amount_unit": descriptor("报告金额单位"),
        "reporting_amount_precision": descriptor("报告金额小数位数", type_="integer"),
        "credit_limit": descriptor("信用额度", type_="money"),
        "credit_limit_status": descriptor("信用额度报告状态", type_="enum"),
        "used_amount": descriptor("已使用额度", type_="money"),
        "used_amount_status": descriptor("已使用额度报告状态", type_="enum"),
        "loan_amount": descriptor("贷款发放金额", type_="money"),
        "loan_amount_status": descriptor("贷款金额报告状态", type_="enum"),
        "balance": descriptor("余额", type_="money"),
        "balance_status": descriptor("余额报告状态", type_="enum"),
        "account_state": descriptor("账户开闭状态", type_="enum"),
        "account_lifecycle_state": descriptor("账户生命周期状态", type_="enum"),
        "card_activation_state": descriptor("卡片激活报告状态", type_="enum"),
        "activation_state": descriptor("兼容卡片激活状态", type_="enum"),
        "payoff_state": descriptor("债务结清状态", type_="enum"),
        "credit_quality_status": descriptor("信贷质量状态", type_="enum"),
        "current_overdue": descriptor("当前是否逾期", type_="boolean"),
        "ever_overdue": descriptor("是否曾逾期", type_="boolean"),
        "overdue_months": descriptor("逾期月数", type_="integer"),
        "over_90_days": descriptor("是否发生过90天以上逾期", type_="boolean"),
        "unbilled_installment_balance": descriptor("未出单大额专项分期余额", type_="money"),
        "status": descriptor("兼容状态（已弃用）", type_="enum"),
    }
    datasets = {
        "credit_accounts": {
            "definition": "一行对应一个信贷账户；是个人简版报告的唯一账户金额事实表。",
            "aggregation": "个人简版金额按 reporting_amount_currency 聚合；account_currency 仅表示账户计价币种；不得与 credit_lines 相加。",
            "columns": common_account_columns,
        },
        "credit_lines": {
            "definition": "授信额度视图；个人简版中由 credit_accounts 派生且不单独导出。",
            "non_additive_with": ["credit_accounts"],
            "columns": {
                key: value
                for key, value in common_account_columns.items()
                if key
                in {
                    "account_id",
                    "account_identifier",
                    "institution",
                    "currency",
                    "account_state",
                    "payoff_state",
                    "status",
                }
            }
            | {
                "credit_line_id": descriptor("授信记录ID"),
                "facility_type": descriptor("授信类型"),
                "total_limit": descriptor("总额度", type_="money"),
                "total_limit_status": descriptor("总额度报告状态", type_="enum"),
                "used_limit": descriptor("已用额度", type_="money"),
                "used_limit_status": descriptor("已用额度报告状态", type_="enum"),
                "available_limit": descriptor("剩余可用额度", type_="money"),
                "available_limit_status": descriptor("剩余可用额度报告状态", type_="enum"),
            },
        },
        "repayment_liability_records": {
            "columns": {
                "liability_id": descriptor("相关还款责任记录ID"),
                "sequence": descriptor("组内序号", type_="integer"),
                "liability_date": descriptor("责任发生日期", type_="date"),
                "related_party_name": descriptor("相关方名称"),
                "related_party_id_type": descriptor("相关方证件类型"),
                "related_party_id_number": descriptor("相关方证件号码", type_="long_id", sensitive=True),
                "institution": descriptor("管理机构"),
                "business_type": descriptor("业务类型"),
                "underlying_business_type": descriptor("责任对应业务类型"),
                "snapshot_balance_business_type": descriptor("余额快照业务类型"),
                "responsibility_type": descriptor("责任类型"),
                "responsibility_amount": descriptor("责任金额", type_="money"),
                "responsibility_amount_reported": descriptor("责任金额报告状态", type_="boolean"),
                "contract_number": descriptor("合同编号", type_="long_id", sensitive=True),
                "snapshot_date": descriptor("信息截至日期", type_="date"),
                "balance": descriptor("余额", type_="money"),
                "currency": descriptor("币种", type_="currency"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "amount_unit": descriptor("金额单位"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            }
        },
        "repayment_records": {
            "definition": "一行对应一个账户在一个自然月的还款状态。",
            "columns": {
                "repayment_id": descriptor("还款记录ID"),
                "account_id": descriptor("账户记录ID"),
                "grid_id": descriptor("还款网格ID"),
                "year": descriptor("年份", type_="integer"),
                "month": descriptor("月份", type_="integer"),
                "status": descriptor("还款状态", type_="enum"),
                "overdue_amount": descriptor("逾期金额", type_="money"),
            },
        },
        "overdue_records": {
            "definition": "一行对应一个曾发生逾期的信贷账户及其最近5年逾期事实。",
            "columns": {
                key: value
                for key, value in common_account_columns.items()
                if key
                in {
                    "sequence",
                    "account_id",
                    "account_type",
                    "institution",
                    "business_type",
                    "card_tail",
                    "open_date",
                    "currency",
                }
            }
            | {
                "overdue_id": descriptor("逾期记录ID"),
                "period_scope": descriptor("统计期间", type_="enum"),
                "overdue_months": descriptor("最近5年逾期月数", type_="integer"),
                "over_90_days_months": descriptor("其中超过90天月数", type_="integer"),
                "current_overdue": descriptor("当前是否逾期", type_="boolean"),
                "current_overdue_status": descriptor("当前逾期状态", type_="enum"),
                "over_90_days": descriptor("是否发生过90天以上逾期", type_="boolean"),
            },
        },
        "inquiry_records": {
            "definition": "机构查询和个人查询共享事实表；展示时按 inquiry_type 分组。",
            "columns": {
                "inquiry_id": descriptor("查询记录ID"),
                "sequence": descriptor("组内序号", type_="integer"),
                "inquiry_date": descriptor("查询日期", type_="date"),
                "institution": descriptor("查询机构"),
                "reason": descriptor("查询原因"),
                "source_reason": descriptor("源文查询原因"),
                "query_channel": descriptor("查询渠道"),
                "inquiry_type": descriptor("查询类型", type_="enum"),
            },
        },
        "public_records": {
            "definition": "一行对应一项源报告公共记录；content 保留其类型化业务字段副本。",
            "columns": {
                "public_record_id": descriptor("公共记录ID"),
                "sequence": descriptor("组内序号", type_="integer"),
                "record_type": descriptor("公共记录类型", type_="enum"),
                "authority": descriptor("记录机关"),
                "category": descriptor("记录类别"),
                "start_date": descriptor("开始日期", type_="date"),
                "end_date": descriptor("结束日期", type_="date"),
                "content": descriptor("记录内容"),
            },
        },
        "identity_documents": {
            "definition": "一行对应信息主体的一种证件；主证件及其他证件均保留。",
            "columns": {
                "identity_document_id": descriptor("证件记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "holder_name": descriptor("持有人姓名"),
                "document_type": descriptor("证件类型"),
                "document_number": descriptor("证件号码", type_="long_id", sensitive=True),
                "is_primary": descriptor("是否主证件", type_="boolean"),
            },
        },
        "personal_report_metadata": {
            "definition": "个人简版报告头信息及报告级金额口径的一行式副本。",
            "columns": {
                "personal_report_metadata_id": descriptor("报告元数据ID"),
                "report_number": descriptor("报告编号", type_="long_id", sensitive=True),
                "report_time": descriptor("报告时间", type_="datetime"),
                "subject_name": descriptor("姓名"),
                "primary_id_type": descriptor("主证件类型"),
                "primary_id_number": descriptor("主证件号码", type_="long_id", sensitive=True),
                "reporting_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
                "reporting_amount_precision": descriptor("报告金额小数位数", type_="integer"),
                "amount_policy_source": descriptor("金额口径来源", type_="enum"),
            },
        },
        "personal_credit_summary_records": {
            "definition": "一行对应源信息概要中的一个指标和业务类别，保留 -- 与数值零的区别。",
            "columns": {
                "credit_summary_record_id": descriptor("概要记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "summary_scope": descriptor("概要口径", type_="enum"),
                "metric": descriptor("概要指标", type_="enum"),
                "business_category": descriptor("业务类别", type_="enum"),
                "value": descriptor("指标值", type_="integer"),
                "reporting_status": descriptor("报告状态", type_="enum"),
            },
        },
        "asset_disposition_records": {
            "definition": "一行对应一项资产处置信息。",
            "columns": {
                "asset_disposition_id": descriptor("资产处置记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "disposition_date": descriptor("债权接收日期", type_="date"),
                "asset_management_company": descriptor("资产管理公司"),
                "received_debt_amount": descriptor("接收债权金额", type_="money"),
                "snapshot_date": descriptor("信息截至日期", type_="date"),
                "balance": descriptor("余额", type_="money"),
                "last_repayment_date": descriptor("最近一次还款日期", type_="date"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "guarantor_compensation_records": {
            "definition": "一行对应一项垫款/保证人代偿信息。",
            "columns": {
                "guarantor_compensation_id": descriptor("垫款记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "compensation_start_date": descriptor("累计代偿起始日期", type_="date"),
                "guarantor": descriptor("代偿机构"),
                "cumulative_compensation_amount": descriptor("累计代偿金额", type_="money"),
                "settlement_date": descriptor("结清日期", type_="date"),
                "settlement_state": descriptor("结清状态", type_="enum"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "postpaid_records": {
            "definition": "一行对应一项后付费非信贷交易记录。",
            "columns": {
                "postpaid_record_id": descriptor("后付费记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "institution": descriptor("机构名称"),
                "business_type": descriptor("业务类型"),
                "billing_month": descriptor("记账年月", type_="date"),
                "service_start_date": descriptor("业务开通日期", type_="date"),
                "payment_status": descriptor("当前缴费状态", type_="enum"),
                "current_arrears_amount": descriptor("当前欠费金额", type_="money"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "tax_arrears_records": {
            "definition": "一行对应一项欠税记录。",
            "columns": {
                "tax_arrears_id": descriptor("欠税记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "tax_authority": descriptor("主管税务机关"),
                "statistics_date": descriptor("欠税统计日期", type_="date"),
                "arrears_amount": descriptor("欠税总额", type_="money"),
                "taxpayer_identifier": descriptor("纳税人识别号", type_="long_id", sensitive=True),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "civil_judgment_records": {
            "definition": "一行对应一项民事判决记录。",
            "columns": {
                "civil_judgment_id": descriptor("民事判决记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "filing_court": descriptor("立案法院"),
                "case_number": descriptor("案号", type_="long_id", sensitive=True),
                "cause": descriptor("案由"),
                "cause_status": descriptor("案由报告状态", type_="enum"),
                "filing_date": descriptor("立案日期", type_="date"),
                "closure_method": descriptor("结案方式"),
                "claim_subject": descriptor("诉讼标的"),
                "claim_amount": descriptor("诉讼标的金额", type_="money"),
                "judgment_result": descriptor("判决/调解结果"),
                "judgment_effective_date": descriptor("判决/调解生效日期", type_="date"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "enforcement_records": {
            "definition": "一行对应一项强制执行记录。",
            "columns": {
                "enforcement_record_id": descriptor("强制执行记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "court": descriptor("执行法院"),
                "case_number": descriptor("案号", type_="long_id", sensitive=True),
                "cause": descriptor("执行案由"),
                "cause_status": descriptor("执行案由报告状态", type_="enum"),
                "filing_date": descriptor("立案日期", type_="date"),
                "case_status": descriptor("案件状态"),
                "closure_method": descriptor("结案方式"),
                "closure_date": descriptor("结案日期", type_="date"),
                "requested_subject": descriptor("申请执行标的"),
                "requested_amount": descriptor("申请执行标的金额", type_="money"),
                "executed_subject": descriptor("已执行标的"),
                "executed_amount": descriptor("已执行标的金额", type_="money"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "administrative_penalty_records": {
            "definition": "一行对应一项行政处罚记录。",
            "columns": {
                "administrative_penalty_id": descriptor("行政处罚记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "authority": descriptor("处罚机构"),
                "document_number": descriptor("文书编号", type_="long_id", sensitive=True),
                "penalty_content": descriptor("处罚内容"),
                "penalty_amount": descriptor("处罚金额", type_="money"),
                "effective_date": descriptor("生效日期", type_="date"),
                "end_date": descriptor("截止日期", type_="date"),
                "administrative_review_result": descriptor("行政复议结果"),
                "administrative_review_result_status": descriptor("行政复议结果报告状态", type_="enum"),
                "reporting_amount_currency": descriptor("报告金额币种", type_="currency"),
                "reporting_amount_unit": descriptor("报告金额单位"),
            },
        },
        "report_notes": {
            "definition": "源报告说明部分的逐条转录。",
            "columns": {
                "note_id": descriptor("说明记录ID"),
                "sequence": descriptor("序号", type_="integer"),
                "content": descriptor("说明内容"),
            },
        },
    }
    return {
        "schema_version": "credit_report.dictionary.v2",
        "fields": fields,
        "datasets": datasets,
        "null_semantics": {
            "null": "源文未报告或不适用；使用相邻的 *_status 字段区分。",
            "reported": "源文明确给出该值，包括数值零。",
            "not_reported": "源文使用 -- 或未给出数值。",
            "not_applicable": "该字段不适用于此账户类型。",
        },
        "enums": {
            "account_type": {
                "credit_card": "信用卡",
                "loan": "贷款",
                "credit_line": "贷款授信",
            },
            "account_state": {"open": "未关闭", "closed": "已关闭", "unknown": "未知"},
            "account_lifecycle_state": {
                "open": "未关闭",
                "settled": "已结清",
                "closed": "已销户",
                "transferred_out": "已转出",
                "unknown": "未知",
            },
            "card_activation_state": {
                "activated": "已激活",
                "not_activated": "尚未激活",
                "not_reported": "未报告",
                "not_applicable": "不适用",
            },
            "activation_state": {
                "active": "已激活",
                "inactive": "尚未激活",
                "not_reported": "未报告",
                "not_applicable": "不适用",
            },
            "credit_card_type": {
                "credit_card": "贷记卡",
                "quasi_credit_card": "准贷记卡",
            },
            "credit_line_validity_type": {
                "fixed_term": "固定期限",
                "perpetual": "长期有效",
                "not_reported": "未报告",
            },
            "termination_event_type": {
                "debt_settled": "债务结清",
                "account_closed": "信用卡销户",
                "transferred_out": "账户转出",
            },
            "credit_quality_status": {
                "bad_debt": "呆账",
                "not_reported": "未报告",
            },
            "payoff_state": {
                "outstanding": "未结清",
                "settled": "已结清",
                "not_applicable": "不适用",
                "unknown": "未知",
            },
            "inquiry_type": {"institution": "机构查询", "personal": "个人查询"},
            "period_scope": {"last_5_years": "最近5年", "account_snapshot": "账户快照", "month": "月份"},
            "current_overdue_status": {
                "overdue": "当前有逾期",
                "not_overdue": "当前无逾期",
                "not_reported": "未报告",
            },
            "record_status": {"no_records": "无记录", "reported": "已报告"},
            "marital_status": {
                "unmarried": "未婚",
                "married": "已婚",
                "divorced": "离婚",
            },
            "report_subtype": {
                "personal_brief": "个人信用报告（简版）",
                "personal_detail": "个人信用报告（详版）",
                "enterprise": "企业信用报告",
            },
            "content_mode": {
                CONTENT_MODE_NATIVE: "原生文本",
                CONTENT_MODE_MIXED: "混合文本与图像",
                CONTENT_MODE_SCANNED: "扫描图像",
            },
            "source": {
                "personal_brief_native_text": "个人简版信用报告原生文本",
                "enterprise_native_text": "企业信用报告原生文本",
            },
            "document_type": {
                "personal_credit_report_brief": "个人信用报告（简版）",
                "personal_credit_report_detailed": "个人信用报告（详版）",
                "enterprise_credit_report": "企业信用报告",
            },
            "reporting_status": {
                "reported": "已报告",
                "not_reported": "未报告",
                "not_applicable": "不适用",
                "derived": "由账户记录派生",
            },
        },
    }


def credit_report_semantic_extensions(*, report_subtype: str) -> dict[str, Any]:
    return {
        "rendering_contract": {
            "authoritative_business_records": "datasets",
            "presentation_order": "reading.document_flow",
            "source_provenance": "structure",
            "summary_facts": "domain.facts",
            "do_not_union_representations": True,
        },
        "presentation_policy": {
            "classification": "highly_sensitive_personal_financial_data",
            "default_display": "masked",
            "enhanced_markdown_display": "full" if report_subtype == "personal_brief" else "masked",
            "mask_fields": [
                "id_number",
                "subject_id",
                "report_number",
                "account_identifier",
                "related_party_id_number",
                "contract_number",
                "document_number",
                "primary_id_number",
                "taxpayer_identifier",
                "case_number",
            ],
            "source_structure_contains_verbatim_sensitive_text": True,
            "access_control_required": True,
        },
        "enhanced_markdown": {
            "privacy_mode": "full",
            "show_top_document_metadata": False,
            "section_layouts": {
                "credit_summary": {
                    "omit_unlisted": True,
                    "groups": [
                        {
                            "title": "个人信息",
                            "fields": ["subject_name", "id_type", "id_number", "marital_status"],
                        },
                        {
                            "title": "信用概览",
                            "fields": [
                                "source_account_count",
                                "source_unclosed_account_count",
                                "activated_credit_card_account_count",
                                "inactive_credit_card_account_count",
                                "repayment_liability_count",
                                "inquiry_count",
                                "institution_inquiry_count",
                                "personal_inquiry_count",
                                {
                                    "key": "source_overdue_account_count",
                                    "fallback": "source_overdue_account_count_status",
                                },
                                {
                                    "key": "source_over_90_days_account_count",
                                    "fallback": "source_over_90_days_account_count_status",
                                },
                            ],
                            "nested_groups": [
                                "source_account_counts",
                                "source_unclosed_account_counts",
                                "source_overdue_account_counts",
                                "source_over_90_days_account_counts",
                            ],
                        },
                        {
                            "title": "报告信息",
                            "fields": ["report_number", "report_time", "report_subtype"],
                            "document_fields": [
                                {"path": "page_count", "key": "page_count"},
                            ],
                        },
                    ],
                }
            },
            "dataset_layouts": {
                "credit_accounts": {
                    "mode": "partitioned_tables",
                    "partition_by": "account_type",
                    "partitions": [
                        {
                            "value": "credit_card",
                            "title": "信用卡账户",
                            "columns": [
                                "sequence",
                                "institution",
                                "business_type",
                                "credit_card_type",
                                "account_identifier",
                                "card_tail",
                                "open_date",
                                "snapshot_date",
                                "close_date",
                                "account_currency",
                                "reporting_amount_currency",
                                "reporting_amount_unit",
                                "credit_limit",
                                "credit_limit_status",
                                "used_amount",
                                "used_amount_status",
                                "balance",
                                "balance_status",
                                "account_state",
                                "account_lifecycle_state",
                                "card_activation_state",
                                "credit_quality_status",
                            ],
                        },
                        {
                            "value": "loan",
                            "title": "贷款账户",
                            "columns": [
                                "sequence",
                                "institution",
                                "business_type",
                                "account_identifier",
                                "open_date",
                                "snapshot_date",
                                "contract_maturity_date",
                                "close_date",
                                "account_currency",
                                "reporting_amount_currency",
                                "reporting_amount_unit",
                                "loan_amount",
                                "loan_amount_status",
                                "balance",
                                "balance_status",
                                "account_state",
                                "account_lifecycle_state",
                                "termination_event_type",
                                "payoff_state",
                            ],
                        },
                        {
                            "value": "credit_line",
                            "title": "贷款授信",
                            "columns": [
                                "sequence",
                                "institution",
                                "business_type",
                                "account_identifier",
                                "open_date",
                                "snapshot_date",
                                "credit_line_expiry_date",
                                "credit_line_validity_type",
                                "close_date",
                                "account_currency",
                                "reporting_amount_currency",
                                "reporting_amount_unit",
                                "credit_limit",
                                "credit_limit_status",
                                "balance",
                                "balance_status",
                                "account_state",
                                "account_lifecycle_state",
                                "termination_event_type",
                                "payoff_state",
                            ],
                        },
                    ],
                }
            },
            "appendix": {
                "title": "附录：文档来源与提取信息",
                "fields": [
                    "content_mode",
                    "source",
                    "source_summary_table_id",
                    "source_summary_page",
                    "account_count",
                    "projected_account_count",
                ],
                "document_fields": [
                    {"path": "type", "key": "document_type"},
                    {"path": "source_file.name", "key": "source_file_name"},
                    {"path": "source_file.sha256", "key": "source_file_sha256"},
                ],
            },
        }
        if report_subtype == "personal_brief"
        else {},
        "dataset_relationships": {
            "credit_lines": {
                "relationship": "derived_view_of_credit_accounts",
                "additive": False,
                "exported_for_personal_brief": report_subtype != "personal_brief",
            }
        },
    }


__all__ = [
    "credit_report_data_dictionary",
    "credit_report_semantic_extensions",
    "enrich_credit_report_record_evidence",
]
