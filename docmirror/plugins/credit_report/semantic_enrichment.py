# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Community-semantic metadata and provenance enrichment for credit reports."""

from __future__ import annotations

import re
from typing import Any


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


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
    normalized = record.get("normalized") if isinstance(record.get("normalized"), dict) else {}
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


def _referenced_node_ids(refs: list[Any]) -> list[str]:
    ids: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        ids.extend(str(value) for value in [ref.get("node_id"), *(ref.get("node_ids") or [])] if value)
    return list(dict.fromkeys(ids))


def enrich_credit_report_record_evidence(
    parse_result: Any,
    collections: dict[str, list[dict[str, Any]]],
) -> tuple[str, ...]:
    """Attach canonical source nodes, bboxes, and atom IDs to plugin records."""
    nodes, nodes_by_id = _node_payloads(parse_result)
    all_evidence_ids: list[str] = []
    for records in collections.values():
        for record in records:
            refs = [dict(ref) for ref in (record.get("source_refs") or []) if isinstance(ref, dict)]
            if not refs:
                refs = [{"source": str(record.get("source") or "credit_business_projection")}]
                if record.get("page"):
                    refs[0]["page"] = record["page"]
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
        "activated_credit_card_account_count": descriptor("已激活信用卡账户数", type_="integer"),
        "inactive_credit_card_account_count": descriptor("尚未激活信用卡账户数", type_="integer"),
        "settled_account_count": descriptor("派生已结清/关闭账户数", type_="integer"),
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
        "personal_inquiry_count": descriptor("本人查询记录数", type_="integer"),
        "projected_account_count": descriptor("投影账户数", type_="integer"),
        "source_summary_table_id": descriptor("源概要表标识"),
        "source_summary_page": descriptor("源概要表页码", type_="integer"),
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
    }
    common_account_columns = {
        "sequence": descriptor("序号", type_="integer"),
        "account_id": descriptor("账户记录ID"),
        "account_identifier": descriptor("账户标识", type_="long_id", sensitive=True),
        "account_type": descriptor("账户类型", type_="enum"),
        "institution": descriptor("管理机构"),
        "business_type": descriptor("业务类型"),
        "card_tail": descriptor("卡片尾号", type_="long_id"),
        "open_date": descriptor("开立日期", type_="date"),
        "snapshot_date": descriptor("信息截至日期", type_="date"),
        "due_date": descriptor("到期日期", type_="date"),
        "close_date": descriptor("结清/销户日期", type_="date"),
        "currency": descriptor(
            "币种",
            type_="currency",
            definition="所有金额只可在相同币种内聚合。",
        ),
        "credit_limit": descriptor("信用额度", type_="money"),
        "credit_limit_status": descriptor("信用额度报告状态", type_="enum"),
        "used_amount": descriptor("已使用额度", type_="money"),
        "used_amount_status": descriptor("已使用额度报告状态", type_="enum"),
        "loan_amount": descriptor("贷款发放金额", type_="money"),
        "loan_amount_status": descriptor("贷款金额报告状态", type_="enum"),
        "balance": descriptor("余额", type_="money"),
        "balance_status": descriptor("余额报告状态", type_="enum"),
        "account_state": descriptor("账户开闭状态", type_="enum"),
        "activation_state": descriptor("卡片激活状态", type_="enum"),
        "payoff_state": descriptor("债务结清状态", type_="enum"),
        "status": descriptor("兼容状态（已弃用）", type_="enum"),
    }
    datasets = {
        "credit_accounts": {
            "definition": "一行对应一个信贷账户；是个人简版报告的唯一账户金额事实表。",
            "aggregation": "金额必须按 currency 分组；不得与 credit_lines 相加。",
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
            },
        },
        "repayment_liability_records": {
            "columns": {
                "related_party_id_number": descriptor("相关方证件号码", type_="long_id", sensitive=True),
                "contract_number": descriptor("合同编号", type_="long_id", sensitive=True),
                "currency": descriptor("币种", type_="currency"),
            }
        },
        "inquiry_records": {
            "definition": "机构查询和本人查询共享事实表；展示时按 inquiry_type 分组。",
            "columns": {
                "sequence": descriptor("组内序号", type_="integer"),
                "inquiry_date": descriptor("查询日期", type_="date"),
                "institution": descriptor("查询机构"),
                "reason": descriptor("查询原因"),
                "inquiry_type": descriptor("查询类型", type_="enum"),
            },
        },
        "report_notes": {
            "definition": "源报告说明部分的逐条转录。",
            "columns": {
                "sequence": descriptor("序号", type_="integer"),
                "content": descriptor("说明内容"),
            },
        },
    }
    return {
        "schema_version": "credit_report.dictionary.v1",
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
            "activation_state": {"active": "已激活", "inactive": "尚未激活", "not_applicable": "不适用"},
            "payoff_state": {
                "outstanding": "未结清",
                "settled": "已结清",
                "not_applicable": "不适用",
                "unknown": "未知",
            },
            "inquiry_type": {"institution": "机构查询", "personal": "本人查询"},
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
                "native_text": "原生文本",
                "mixed": "混合文本与图像",
                "scanned": "扫描图像",
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
