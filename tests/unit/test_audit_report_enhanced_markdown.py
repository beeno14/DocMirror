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


def test_audit_renderer_preserves_source_compatibility_markers() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": f"body:{index}",
                    "kind": "paragraph",
                    "role": "body",
                    "order": index,
                    "page": 1,
                    "text": text,
                }
                for index, text in enumerate(
                    (
                        "①以摊余成本计量的金融资产",
                        "②以公允价值计量且其变动计入其他综合收益的金融资产",
                        "③以公允价值计量且其变动计入当期损益的金融资产",
                        "第Ⅰ类及²项测试",
                    ),
                    start=1,
                )
            ],
            "source_tables": [],
        },
        "datasets": [],
    }
    before = copy.deepcopy(semantic)

    enhanced = render_audit_reading_markdown(semantic)

    assert semantic == before
    assert "①以摊余成本计量的金融资产" in enhanced
    assert "②以公允价值计量且其变动计入其他综合收益的金融资产" in enhanced
    assert "③以公允价值计量且其变动计入当期损益的金融资产" in enhanced
    assert "第Ⅰ类及²项测试" in enhanced
    assert "1以摊余成本计量的金融资产" not in enhanced


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
    assert "（空白页）" not in enhanced
    assert "**文档类型:**" not in enhanced
    assert "**页数:**" not in enhanced
    assert "## 123.45" not in enhanced
    assert "123.45" in enhanced


def test_audit_reading_projection_collapses_repeated_statement_title_and_landscape_footer() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "2024年度审计报告", "page_count": 1},
        "enhanced_markdown": {"page_dimensions": {"1": {"width": 842, "height": 595}}},
        "structure": {
            "blocks": [
                {
                    "id": "statement-title",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "text": "资产负债表 资产负债表 资产负债表（续）（续）（续）",
                    "bbox": [250.0, 30.0, 600.0, 60.0],
                },
                {
                    "id": "landscape-footer",
                    "kind": "paragraph",
                    "role": "list_marker",
                    "order": 4,
                    "page": 1,
                    "text": "8",
                    "bbox": [420.0, 570.0, 430.0, 585.0],
                },
                {
                    "id": "cash-flow-title-1",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 2,
                    "page": 1,
                    "text": "现金流量表",
                },
                {
                    "id": "cash-flow-title-2",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 3,
                    "page": 1,
                    "text": "现金流量表",
                },
            ],
            "source_tables": [],
            "reading_flows": [{"id": "main", "node_ids": ["statement-title", "landscape-footer"]}],
        },
        "datasets": [],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert enhanced.count("资产负债表") == 1
    assert "资产负债表（续）" in enhanced
    assert enhanced.count("现金流量表") == 1
    assert "\n8\n" not in enhanced


def test_audit_reading_projection_preserves_period_caption_above_table() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "period-caption",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "text": "2024年度",
                    "bbox": [250.0, 50.0, 340.0, 70.0],
                },
                {
                    "id": "table",
                    "kind": "physical_table",
                    "role": "body",
                    "order": 2,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                },
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "bbox": [80.0, 100.0, 520.0, 180.0],
                    "headers": ["项目", "2024年度"],
                    "rows": [["营业收入", "100.00"]],
                }
            ],
        },
        "datasets": [],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert enhanced.count("2024年度") == 2


def test_audit_reading_projection_reflows_physical_lines_and_restores_signature_columns() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "paragraph-line-1",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "text": "这是一个自然段，",
                    "bbox": [89.0, 100.0, 540.0, 116.8],
                },
                {
                    "id": "paragraph-line-2",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 2,
                    "page": 1,
                    "text": "被物理换行。",
                    "bbox": [65.0, 128.6, 180.0, 145.4],
                },
                {
                    "id": "paragraph-2",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 3,
                    "page": 1,
                    "text": "这是第二段。",
                    "bbox": [89.0, 151.9, 200.0, 168.7],
                },
                {
                    "id": "signature",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 4,
                    "page": 1,
                    "text": "上海示例会计师事务所有限公司\n中国注册会计师：",
                    "bbox": [65.0, 300.0, 440.0, 317.0],
                },
            ],
            "source_tables": [],
        },
        "datasets": [],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert "这是一个自然段，被物理换行。" in enhanced
    assert "这是一个自然段，\n\n被物理换行。" not in enhanced
    assert "这是第二段。" in enhanced
    assert "**上海示例会计师事务所有限公司:** 中国注册会计师：" in enhanced


def test_audit_reading_projection_keeps_circled_number_subheading_separate() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "subheading",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "text": "①以摊余成本计量的金融资产",
                    "bbox": [89.0, 100.0, 330.0, 116.8],
                },
                {
                    "id": "body",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 2,
                    "page": 1,
                    "text": "本公司管理此类金融资产。",
                    "bbox": [89.0, 123.3, 330.0, 140.1],
                },
            ],
            "source_tables": [],
        },
        "datasets": [],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert "①以摊余成本计量的金融资产\n\n本公司管理此类金融资产。" in enhanced
    assert "①以摊余成本计量的金融资产本公司" not in enhanced


def test_audit_reading_projection_normalizes_cjk_radicals_in_body_text() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "business-scope",
                    "kind": "paragraph",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "text": "江⻄企业销售⻋船配件及⻮轮，并提供⽹ 络技术和管理中⼼ （有限合伙）服务。",
                }
            ],
            "source_tables": [],
            "reading_flows": [{"id": "main", "node_ids": ["business-scope"]}],
        },
        "datasets": [],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert "江西企业销售车船配件及齿轮，并提供网络技术和管理中心（有限合伙）服务。" in enhanced
    assert not any(0x2E80 <= ord(character) <= 0x2FFF for character in enhanced)


def test_audit_reading_projection_normalizes_dataset_overlay_values_without_mutating_facts() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "table:1",
                    "kind": "physical_table",
                    "order": 1,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                }
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "headers": ["项 目", "期 末 余 额"],
                    "rows": [["合 计", "1,234.00"]],
                }
            ],
        },
        "datasets": [
            {
                "id": "ds_balance",
                "name": "balance",
                "columns": [
                    {"key": "item", "label": "项 目"},
                    {"key": "ending_balance", "label": "期 末 余 额"},
                ],
                "rows": [
                    {
                        "record_id": "balance:r000001",
                        "canonical_raw": {"item": "合 计", "ending_balance": "1,234.00"},
                        "normalized": {"item": "合计", "ending_balance": "1234.0"},
                        "source": {"physical_table_id": "pt_1_0", "page": 1},
                    }
                ],
            }
        ],
        "bindings": [
            {
                "dataset_id": "ds_balance",
                "record_id": "balance:r000001",
                "source_table_refs": ["pt_1_0"],
            }
        ],
    }
    before = copy.deepcopy(semantic)

    enhanced = render_audit_reading_markdown(semantic)

    assert semantic == before
    assert "| 项目 | 期末余额 |" in enhanced
    assert "| 合计 | 1,234.00 |" in enhanced
    assert "项 目" not in enhanced
    assert "合 计" not in enhanced


def test_audit_reading_projection_keeps_complete_source_table_when_dataset_is_unverified() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "table:1",
                    "kind": "physical_table",
                    "order": 1,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                }
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "headers": ["类别", "金额"],
                    "rows": [["按组合计提", "58,251,600.81"]],
                }
            ],
        },
        "datasets": [
            {
                "id": "ds_accounts_receivable",
                "name": "accounts_receivable",
                "status": "partial",
                "completeness": {"verified": False},
                "columns": [{"key": "category", "label": "类别"}],
                "rows": [
                    {
                        "record_id": "accounts_receivable:r000001",
                        "canonical_raw": {"category": "按组合计提"},
                        "normalized": {"category": "按组合计提"},
                        "source": {"physical_table_id": "pt_1_0", "page": 1},
                    }
                ],
            }
        ],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert "| 类别 | 金额 |" in enhanced
    assert "| 按组合计提 | 58,251,600.81 |" in enhanced


def test_audit_reading_projection_hides_period_role_and_uses_equity_component_labels() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "table:1",
                    "kind": "physical_table",
                    "role": "body",
                    "order": 1,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                }
            ],
            "source_tables": [{"id": "pt_1_0", "page": 1, "headers": ["项目", "本期金额"], "rows": []}],
            "reading_flows": [{"id": "main", "node_ids": ["table:1"]}],
        },
        "datasets": [
            {
                "id": "ds_owners_equity_changes",
                "name": "owners_equity_changes",
                "columns": [
                    {"key": "item", "label": "项目", "type": "string"},
                    {"key": "period_role", "label": "期间角色", "type": "string"},
                    {"key": "paid_in_capital", "label": "实收资本（或股本）", "type": "decimal"},
                    {"key": "capital_reserve", "label": "资本公积", "type": "decimal"},
                ],
                "rows": [
                    {
                        "record_id": "owners_equity_changes:r000001",
                        "canonical_raw": {
                            "item": "-2.对所有者（股东）的分配",
                            "paid_in_capital": "10.00",
                            "capital_reserve": "20.00",
                        },
                        "normalized": {
                            "item": "2.对所有者（股东）的分配",
                            "period_role": "current",
                            "paid_in_capital": "10.00",
                            "capital_reserve": "20.00",
                        },
                        "source": {"table_id": "pt_1_0", "page": 1, "recovery": "source_rows"},
                    }
                ],
            }
        ],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert "期间角色" not in enhanced
    assert "current" not in enhanced
    assert "实收资本（或股本）（本期金额）" in enhanced
    assert "资本公积（本期金额）" in enhanced
    assert "2.对所有者（股东）的分配" in enhanced
    assert "-2.对所有者" not in enhanced


def test_audit_reading_projection_uses_clean_note_dataset_and_suppresses_logical_duplicate() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "logical",
                    "kind": "physical_table",
                    "order": 1,
                    "page": 1,
                    "source_table_ref": "lt_1",
                },
                {
                    "id": "physical",
                    "kind": "physical_table",
                    "order": 2,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                },
            ],
            "source_tables": [
                {"id": "lt_1", "page": 1, "headers": ["错误表头"], "rows": [["重复错误行"]]},
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "headers": ["col_0", "col_1"],
                    "rows": [["项目", "期末余额"], ["运输服务款", "21,733,124.67"]],
                },
            ],
        },
        "datasets": [
            {
                "id": "ds_accounts_payable",
                "name": "accounts_payable",
                "columns": [{"key": "项目", "label": "项目"}, {"key": "期末余额", "label": "期末余额"}],
                "rows": [
                    {
                        "record_id": "accounts_payable:r000001",
                        "canonical_raw": {"项目": "运输服务款", "期末余额": "21,733,124.67"},
                        "normalized": {"项目": "运输服务款", "期末余额": "21733124.67"},
                        "source": {
                            "table_id": "lt_1:segment_1",
                            "physical_table_id": "pt_1_0",
                            "page": 1,
                        },
                    }
                ],
            }
        ],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert enhanced.count("| 运输服务款 | 21,733,124.67 |") == 1
    assert "重复错误行" not in enhanced
    assert "| 项目 | 期末余额 |" in enhanced
    assert "| col_0 | col_1 |" not in enhanced


def test_audit_reading_projection_places_text_recovered_row_on_its_source_page() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 2},
        "structure": {
            "blocks": [
                {
                    "id": "detail-table",
                    "kind": "physical_table",
                    "order": 1,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                },
                {
                    "id": "total-text",
                    "kind": "paragraph",
                    "order": 2,
                    "page": 2,
                    "text": "合 计 19,233,124.67 17,279,642.10",
                },
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "headers": ["项目", "期末余额", "期初余额"],
                    "rows": [["运输服务款", "21,733,124.67", "17,279,642.10"]],
                }
            ],
        },
        "datasets": [
            {
                "id": "ds_accounts_payable",
                "name": "accounts_payable",
                "columns": [
                    {"key": "项目", "label": "项目"},
                    {"key": "期末余额", "label": "期末余额"},
                    {"key": "期初余额", "label": "期初余额"},
                ],
                "rows": [
                    {
                        "record_id": "accounts_payable:r000001",
                        "canonical_raw": {
                            "项目": "运输服务款",
                            "期末余额": "21,733,124.67",
                            "期初余额": "17,279,642.10",
                        },
                        "source": {"table_id": "pt_1_0", "physical_table_id": "pt_1_0", "page": 1},
                    },
                    {
                        "record_id": "accounts_payable:r000002",
                        "canonical_raw": {
                            "项目": "合 计",
                            "期末余额": "19,233,124.67",
                            "期初余额": "17,279,642.10",
                        },
                        "source": {"table_id": "canonical_text", "page": 2},
                    },
                ],
            }
        ],
        "bindings": [],
    }

    enhanced = render_audit_reading_markdown(semantic)

    page_two = enhanced.index('docmirror:page logical="2" source="2"')
    recovered = enhanced.index("| 合计 | 19,233,124.67 | 17,279,642.10 |")
    assert recovered > page_two
    assert enhanced.count("19,233,124.67") == 1


def test_audit_reading_projection_suppresses_evidence_backed_table_fragments_outside_bbox() -> None:
    semantic = {
        "document": {"type": "audit_report", "title": "审计报告", "page_count": 1},
        "structure": {
            "blocks": [
                {
                    "id": "table",
                    "kind": "physical_table",
                    "order": 1,
                    "page": 1,
                    "source_table_ref": "pt_1_0",
                },
                {
                    "id": "missing-header-fragment",
                    "kind": "paragraph",
                    "order": 2,
                    "page": 1,
                    "text": "项目",
                    "bbox": [10.0, 10.0, 40.0, 20.0],
                    "evidence_refs": ["ev:0001:text:000001"],
                },
                {
                    "id": "missing-cell-fragment",
                    "kind": "paragraph",
                    "order": 3,
                    "page": 1,
                    "text": "其他应收款",
                    "bbox": [10.0, 30.0, 80.0, 40.0],
                    "evidence_refs": ["ev:0001:text:000002"],
                },
                {
                    "id": "border-header-fragment",
                    "kind": "paragraph",
                    "order": 4,
                    "page": 1,
                    "text": "期末余额",
                    "bbox": [505.0, 20.0, 550.0, 30.0],
                    "evidence_refs": [],
                },
                {
                    "id": "unpositioned-header-fragment",
                    "kind": "paragraph",
                    "order": 50,
                    "page": 1,
                    "text": "关联方名称",
                    "evidence_refs": [],
                },
                {
                    "id": "next-heading",
                    "kind": "heading",
                    "order": 60,
                    "page": 1,
                    "text": "（3）应付项目",
                    "evidence_refs": ["ev:0001:text:000003"],
                },
            ],
            "source_tables": [
                {
                    "id": "pt_1_0",
                    "page": 1,
                    "bbox": [100.0, 10.0, 500.0, 100.0],
                    "headers": ["关联方名称", "期末余额"],
                    "rows": [["上海示例公司", "100.00"]],
                }
            ],
        },
        "datasets": [
            {
                "id": "ds_related_party_transactions",
                "name": "related_party_transactions",
                "columns": [
                    {"key": "项目", "label": "项目"},
                    {"key": "关联方名称", "label": "关联方名称"},
                    {"key": "期末余额", "label": "期末余额"},
                    {"key": "期末余额/坏账准备", "label": "期末余额/坏账准备"},
                ],
                "rows": [
                    {
                        "record_id": "related_party_transactions:r000001",
                        "canonical_raw": {
                            "项目": "其他应收款",
                            "关联方名称": "上海示例公司",
                            "期末余额": "100.00",
                            "期末余额/坏账准备": "",
                        },
                        "source": {
                            "physical_table_id": "pt_1_0",
                            "page": 1,
                            "evidence_ids": ["ev:0001:text:000001", "ev:0001:text:000002"],
                            "source_cell_refs": [
                                {"bbox": [80.0, 30.0, 100.0, 40.0]},
                                {"bbox": [100.0, 30.0, 500.0, 40.0]},
                            ],
                        },
                    }
                ],
            }
        ],
    }

    enhanced = render_audit_reading_markdown(semantic)

    assert "| 项目 | 关联方名称 | 期末余额 |" in enhanced
    assert "| 其他应收款 | 上海示例公司 | 100.00 |" in enhanced
    assert enhanced.count("\n项目\n") == 0
    assert enhanced.count("\n其他应收款\n") == 0
    assert enhanced.count("\n期末余额\n") == 0
    assert enhanced.count("\n关联方名称\n") == 0
    assert enhanced.count("\n坏账准备\n") == 0
    assert "（3）应付项目" in enhanced
