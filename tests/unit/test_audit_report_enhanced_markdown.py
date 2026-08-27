# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

from docmirror.models.entities.parse_result import (
    DocumentEntities,
    PageContent,
    ParseResult,
    ResultStatus,
    TextBlock,
)
from docmirror.models.mirror.document_flow import DocumentFlowGraph, ReadingFlow, StructureNode
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import render_community_reading_markdown
from docmirror.output.markdown_renderer import render_semantic_source_overlay_markdown
from docmirror.plugins.audit_report.community_plugin import AuditReportPlugin
from docmirror.plugins.audit_report.reading_projection import render_audit_reading_markdown
from docmirror.plugins.generic.community_plugin import plugin as generic_plugin


def test_audit_renderer_preserves_source_flow_and_does_not_mutate_semantic() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "2024年度审计报告", "page_count": 2},
        "structure": {
            "blocks": [
                {
                    "id": "heading:1",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "text": "⼀、审计意⻅",
                },
                {
                    "id": "body:1",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 2,
                    "page": 1,
                    "text": "审计意见正文。",
                },
                {
                    "id": "table:1",
                    "kind": "physical_table",
                    "role": "body",
                    "order": 3,
                    "page": 1,
                    "text": "",
                    "source_table_ref": "pt_1_0",
                },
                {
                    "id": "heading:2",
                    "kind": "heading",
                    "role": "heading",
                    "order": 4,
                    "page": 2,
                    "text": "二、财务报表附注",
                },
                {
                    "id": "body:2",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 5,
                    "page": 2,
                    "text": "附注正文。",
                },
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "headers": ["项目", "金额"],
                    "rows": [["货币资金", "1,234.OO"]],
                }
            ],
            "reading_flows": [{"node_ids": ["heading:1", "body:1", "table:1", "heading:2", "body:2"]}],
        },
        "datasets": [
            {
                "id": "ds_table_001",
                "name": "table_001",
                "columns": [
                    {"key": "item", "label": "项目"},
                    {"key": "amount", "label": "金额"},
                ],
                "rows": [
                    {
                        "record_id": "table_001:r000001",
                        "canonical_raw": {"item": "货币资金", "amount": "1,234.00"},
                        "normalized": {"item": "货币资金", "amount": 1234.0},
                    }
                ],
            }
        ],
        "bindings": [
            {
                "dataset_id": "ds_table_001",
                "record_id": "table_001:r000001",
                "source_table_refs": ["pt_1_0"],
            }
        ],
    }
    before = copy.deepcopy(semantic)

    enhanced = render_semantic_source_overlay_markdown(semantic)

    assert semantic == before
    assert enhanced.index("## ⼀、审计意⻅") < enhanced.index("审计意见正文。")
    assert enhanced.index("审计意见正文。") < enhanced.index("| 货币资金 | 1,234.00 |")
    assert enhanced.index("| 货币资金 | 1,234.00 |") < enhanced.index("## 二、财务报表附注")
    assert "附注正文。" in enhanced
    assert "1,234.OO" not in enhanced


def test_audit_renderer_keeps_source_table_when_dataset_row_count_is_lower() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "table:1",
                    "kind": "physical_table",
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                }
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "headers": ["项目", "金额"],
                    "rows": [["货币资金", "1,234.OO"], ["应收账款", "2,000.00"]],
                }
            ],
        },
        "datasets": [
            {
                "id": "ds_table_001",
                "columns": [{"key": "item", "label": "项目"}, {"key": "amount", "label": "金额"}],
                "rows": [
                    {
                        "record_id": "table_001:r000001",
                        "canonical_raw": {"item": "货币资金", "amount": "1,234.00"},
                    }
                ],
            }
        ],
        "bindings": [
            {
                "dataset_id": "ds_table_001",
                "record_id": "table_001:r000001",
                "source_table_refs": ["pt_1_0"],
            }
        ],
    }

    enhanced = render_semantic_source_overlay_markdown(semantic)

    assert "| 货币资金 | 1,234.OO |" in enhanced
    assert "| 应收账款 | 2,000.00 |" in enhanced


def test_audit_plugin_owns_audit_renderer_without_changing_generic_renderer() -> None:
    def result(document_type: str) -> ParseResult:
        node = StructureNode(node_id="body:1", type="paragraph", page=1, text="完整正文。")
        return ParseResult(
            status=ResultStatus.SUCCESS,
            pages=[PageContent(page_number=1, texts=[TextBlock(content=node.text)])],
            document_flow=DocumentFlowGraph(
                nodes=[node],
                reading_flow=[ReadingFlow(flow_id="main", node_ids=[node.node_id])],
            ),
            entities=DocumentEntities(document_type=document_type),
        )

    audit_bundle = AuditReportPlugin().project_bundle(seal_parse_result(result("audit_report")), file_path="audit.pdf")
    assert audit_bundle is not None
    audit_semantic = audit_bundle.semantic_payload()
    audit_json = audit_bundle.json_payload(audit_semantic)
    assert 'docmirror:audit-reading strategy="source-overlay"' in audit_bundle.render_enhanced_markdown(audit_semantic)
    assert audit_bundle.json_payload(audit_semantic) == audit_json

    financial_bundle = generic_plugin.project_bundle(
        seal_parse_result(result("financial_report")),
        file_path="finance.pdf",
    )
    assert financial_bundle is not None
    financial_semantic = financial_bundle.semantic_payload()
    assert financial_bundle.render_enhanced_markdown(financial_semantic) == render_community_reading_markdown(
        financial_semantic
    )


def test_audit_reading_projection_deduplicates_titles_and_preserves_blank_pages() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "2024年度审计报告", "page_count": 3},
        "structure": {
            "blocks": [
                {
                    "id": "title:1",
                    "kind": "heading",
                    "role": "title",
                    "order": 1,
                    "page": 1,
                    "text": "2024年度审计报告",
                    "bbox": [100.0, 80.0, 400.0, 120.0],
                },
                {
                    "id": "heading:1",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 2,
                    "page": 1,
                    "text": "⼀、审计意⻅",
                    "bbox": [50.0, 150.0, 300.0, 180.0],
                },
                {
                    "id": "page-number:2",
                    "kind": "paragraph",
                    "role": "list_marker",
                    "order": 3,
                    "page": 2,
                    "text": "1",
                    "bbox": [300.0, 790.0, 310.0, 805.0],
                },
                {
                    "id": "amount:3",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 4,
                    "page": 3,
                    "text": "123.45",
                    "bbox": [300.0, 300.0, 350.0, 320.0],
                },
            ],
            "source_tables": [],
            "reading_flows": [{"id": "main", "node_ids": ["title:1", "heading:1", "page-number:2", "amount:3"]}],
        },
        "datasets": [],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert enhanced.count("2024年度审计报告") == 1
    assert "## 一、审计意见" in enhanced
    assert '<!-- docmirror:page logical="2" source="2" -->' in enhanced
    assert "（空白页）" in enhanced
    assert "## 123.45" not in enhanced
    assert "123.45" in enhanced
