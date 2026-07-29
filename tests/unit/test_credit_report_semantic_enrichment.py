# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from docmirror.models.entities.parse_result import DocumentEntities, ParseResult
from docmirror.models.mirror.document_flow import (
    DocumentFlowGraph,
    ReadingFlow,
    StructureNode,
)
from docmirror.plugins.credit_report.semantic_enrichment import (
    enrich_credit_report_record_evidence,
)


def _node(node_id: str, text: str, bbox: list[float]) -> StructureNode:
    return StructureNode(
        node_id=node_id,
        type="paragraph",
        page=1,
        bbox=bbox,
        text=text,
        evidence_refs=[f"ev:{node_id}"],
    )


def _result() -> ParseResult:
    nodes = [
        _node("title", "个人信用报告", [70, 40, 300, 50]),
        _node(
            "rmb_account",
            "2017年09月20日中国工商银行股份有限公司南平分行发放的贷记卡"
            "（人民币账户，卡片尾号：7723）。截至2026年06月，信用额度9,243，已使用额度0。",
            [50, 80, 550, 110],
        ),
        _node(
            "usd_account",
            "2017年09月20日中国工商银行股份有限公司南平分行发放的贷记卡"
            "（美元账户，卡片尾号：7723）。截至2026年06月，信用额度9,243，已使用额度0。",
            [50, 120, 550, 150],
        ),
        _node("institution_section", "机构查询记录明细", [70, 180, 300, 190]),
        _node(
            "institution_collision",
            "9\n2023年10月07日\n民生金融租赁股份有限公司\n融资审批",
            [75, 200, 510, 210],
        ),
        _node(
            "institution_wrapped",
            "56\n2023年11月01日\n深圳前海微众银行股份有限公司\n"
            "法人代表、负责人、高管等资信审",
            [75, 220, 540, 230],
        ),
        _node("institution_reason_tail", "查", [475, 230, 485, 240]),
        _node("personal_section", "个人查询记录明细", [70, 260, 300, 270]),
        _node(
            "personal_collision",
            "9\n2023年10月07日\n本人\n本人查询（商业银行网上银行）",
            [75, 280, 540, 290],
        ),
        _node(
            "personal_wrapped",
            "2\n2024年04月12日\n本人\n本人查询（互联网个人信用信息服",
            [75, 300, 540, 310],
        ),
        _node("personal_reason_tail", "务平台）", [460, 310, 520, 320]),
        _node("notes", "说明", [70, 350, 120, 360]),
    ]
    return ParseResult(
        document_flow=DocumentFlowGraph(
            nodes=nodes,
            reading_flow=[ReadingFlow(flow_id="main", node_ids=[node.node_id for node in nodes])],
        ),
        entities=DocumentEntities(document_type="credit_report"),
    )


def test_personal_semantic_evidence_is_section_and_currency_aware() -> None:
    account = {
        "source": "personal_brief_narrative",
        "source_refs": [{"source": "native_text_narrative", "page": 1, "node_id": "rmb_account"}],
        "normalized": {
            "account_id": "account:usd",
            "sequence": 6,
            "account_type": "credit_card",
            "institution": "中国工商银行股份有限公司南平分行",
            "open_date": "2017-09-20",
            "currency": "USD",
            "card_tail": "7723",
            "credit_limit": 9243,
        },
    }
    inquiries = [
        {
            "source": "personal_brief_inquiry_ledger",
            "source_refs": [{"source": "native_text_ledger", "page": 1, "node_id": "institution_collision"}],
            "normalized": {
                "sequence": 9,
                "inquiry_type": "personal",
                "inquiry_date": "2023-10-07",
                "institution": "本人",
                "reason": "本人查询",
                "source_reason": "本人查询（商业银行网上银行）",
            },
        },
        {
            "source": "personal_brief_inquiry_ledger",
            "source_refs": [{"source": "native_text_ledger", "page": 1}],
            "normalized": {
                "sequence": 56,
                "inquiry_type": "institution",
                "inquiry_date": "2023-11-01",
                "institution": "深圳前海微众银行股份有限公司",
                "reason": "法人代表、负责人、高管等资信审查",
            },
        },
        {
            "source": "personal_brief_inquiry_ledger",
            "source_refs": [{"source": "native_text_ledger", "page": 1}],
            "normalized": {
                "sequence": 2,
                "inquiry_type": "personal",
                "inquiry_date": "2024-04-12",
                "institution": "本人",
                "reason": "本人查询",
                "source_reason": "本人查询（互联网个人信用信息服务平台）",
            },
        },
    ]

    enrich_credit_report_record_evidence(
        _result(),
        {
            "credit_accounts": [account],
            "inquiry_records": inquiries,
        },
    )

    assert account["source_refs"][0]["node_ids"] == ["usd_account"]
    assert inquiries[0]["source_refs"][0]["node_ids"] == ["personal_collision"]
    assert inquiries[1]["source_refs"][0]["node_ids"] == [
        "institution_wrapped",
        "institution_reason_tail",
    ]
    assert inquiries[2]["source_refs"][0]["node_ids"] == [
        "personal_wrapped",
        "personal_reason_tail",
    ]


def test_personal_overdue_identity_and_evidence_follow_the_semantic_account() -> None:
    account = {
        "source": "personal_brief_narrative",
        "source_refs": [{"source": "native_text_narrative", "page": 1}],
        "normalized": {
            "account_id": "account:usd",
            "sequence": 6,
            "account_type": "credit_card",
            "institution": "中国工商银行股份有限公司南平分行",
            "open_date": "2017-09-20",
            "currency": "USD",
            "card_tail": "7723",
            "credit_limit": 9243,
        },
    }
    overdue = {
        "source": "personal_brief_account_narrative",
        "source_refs": [{"source": "native_text_narrative", "page": 1}],
        "normalized": {
            "overdue_id": "overdue:usd",
            "account_id": "account:usd",
            "sequence": 6,
            "account_type": "credit_card",
            "institution": "中国工商银行股份有限公司南平分行",
            "open_date": "2017-09-20",
            "currency": "USD",
            "card_tail": "7723",
        },
    }

    enrich_credit_report_record_evidence(
        _result(),
        {
            "credit_accounts": [account],
            "overdue_records": [overdue],
        },
    )

    assert overdue["record_id"] == "overdue:usd"
    assert overdue["source_refs"][0]["node_ids"] == ["usd_account"]
    assert overdue["source_refs"][0]["evidence_ids"] == ["ev:usd_account"]


def test_enterprise_records_keep_their_existing_exact_evidence() -> None:
    result = _result()
    enterprise = {
        "source": "canonical_enterprise_account_card",
        "source_refs": [
            {
                "source": "canonical_enterprise_table",
                "page": 1,
                "node_id": "rmb_account",
                "node_ids": ["rmb_account"],
            }
        ],
        "normalized": {
            "account_id": "enterprise:1",
            "account_type": "enterprise_credit",
        },
    }

    enrich_credit_report_record_evidence(result, {"credit_accounts": [enterprise]})

    assert enterprise["source_refs"][0]["node_id"] == "rmb_account"
    assert enterprise["source_refs"][0]["node_ids"] == ["rmb_account"]
