# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from docmirror.models.entities.parse_result import (
    DocumentEntities,
    PageContent,
    ParseResult,
    TextBlock,
)
from docmirror.models.mirror.document_flow import (
    DocumentFlowGraph,
    ReadingFlow,
    StructureNode,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.output.reading_projection import FragmentJoin
from docmirror.plugins.credit_report.community_plugin import CreditReportPlugin
from docmirror.plugins.credit_report.inquiry_reading_order import (
    reconstruct_institution_inquiry_rows,
)
from docmirror.plugins.credit_report.projection import derive_credit_report_projection
from docmirror.server.edition_outputs import write_outputs


def _node(
    node_id: str,
    text: str,
    bbox: list[float],
    *,
    page: int = 1,
) -> StructureNode:
    return StructureNode(
        node_id=node_id,
        type="paragraph",
        page=page,
        bbox=bbox,
        text=text,
        evidence_refs=[f"ev:{node_id}"],
    )


def _result() -> ParseResult:
    nodes = [
        _node("title", "个人信用报告 信贷记录", [70, 40, 300, 50]),
        _node("section", "机构查询记录明细", [70, 100, 300, 110]),
        _node("headers", "编号\n查询日期\n查询机构\n查询原因", [70, 120, 510, 130]),
        _node(
            "row2",
            "2\n2025年01月22日\n平安融易（江苏）融资担保有限公",
            [75, 140, 402, 149],
        ),
        _node("row2_reason", "担保资格审查", [454, 140, 508, 149]),
        _node("row2_continuation", "司", [330, 149, 339, 158]),
        _node(
            "row3",
            "3\n2025年01月21日\n重庆度小满小额贷款有限公司",
            [75, 168, 402, 177],
        ),
        _node("row3_reason", "贷款审批", [454, 168, 508, 177]),
        _node(
            "row103",
            "103\n2023年04月28日\n福建漳州农村商业银行股份有限公",
            [75, 188, 402, 197],
        ),
        _node("row103_reason_prefix", "法人代表、负责人、高管等资信审", [454, 188, 508, 197]),
        _node("row103_institution_continuation", "司", [330, 197, 339, 206]),
        _node("row103_reason_continuation", "查", [454, 197, 463, 206]),
        _node("personal_section", "个人查询记录明细", [70, 230, 300, 240]),
        _node(
            "outside_row",
            "1\n2025年01月20日\n不应调整有限公",
            [75, 250, 402, 259],
        ),
        _node("outside_reason", "贷款审批", [454, 250, 508, 259]),
        _node("outside_continuation", "司", [330, 259, 339, 268]),
    ]
    return ParseResult(
        pages=[
            PageContent(
                page_number=1,
                page_mode="native_text",
                texts=[TextBlock(content=node.text, bbox=node.bbox) for node in nodes],
            )
        ],
        document_flow=DocumentFlowGraph(
            nodes=nodes,
            reading_flow=[ReadingFlow(flow_id="main", node_ids=[node.node_id for node in nodes])],
        ),
        entities=DocumentEntities(document_type="credit_report"),
    )


def test_reconstructs_rows_only_inside_institution_inquiry_section() -> None:
    result = _result()

    rows = reconstruct_institution_inquiry_rows(result)

    assert len(rows) == 3
    assert rows[0]["sequence"] == 2
    assert rows[0]["institution"] == "平安融易（江苏）融资担保有限公司"
    assert rows[0]["reason"] == "担保资格审查"
    assert rows[0]["source_node_ids"] == ["row2", "row2_reason", "row2_continuation"]
    assert rows[1]["institution"] == "重庆度小满小额贷款有限公司"
    assert rows[2]["institution"] == "福建漳州农村商业银行股份有限公司"
    assert rows[2]["reason"] == "法人代表、负责人、高管等资信审查"
    assert rows[2]["reason_node_ids"] == [
        "row103_reason_prefix",
        "row103_reason_continuation",
    ]
    assert all(row["institution"] != "不应调整有限公司" for row in rows)


def test_fragment_join_contract_rejects_nodes_outside_source_span() -> None:
    with pytest.raises(ValueError, match="fragment_node_ids"):
        FragmentJoin(
            scope="credit_report.test",
            source_node_ids=("anchor", "reason"),
            anchor_node_id="anchor",
            fragment_node_ids=("invented",),
            reason="test",
            confidence=1.0,
        )


def test_credit_projection_uses_reconstructed_order_for_dataset_and_csv() -> None:
    result = _result()
    plugin = CreditReportPlugin()
    projection = derive_credit_report_projection(plugin, result, result.full_text)
    inquiries = projection.datasets["inquiry_records"]

    assert inquiries[0]["institution"] == "平安融易（江苏）融资担保有限公司"
    assert inquiries[0]["reason"] == "担保资格审查"
    assert inquiries[0]["source_refs"][0]["source"] == "native_dfg_inquiry_ledger"
    assert inquiries[0]["source_refs"][0]["node_ids"] == [
        "row2",
        "row2_reason",
        "row2_continuation",
    ]

    bundle = project_community_bundle(
        seal_parse_result(result),
        projection_data=projection.model_dump(mode="python"),
    )
    rendered_csvs = bundle.render_dataset_csvs()
    inquiry_csv = next(value for key, value in rendered_csvs.items() if key.endswith("inquiry_records.csv"))
    assert "平安融易（江苏）融资担保有限公司" in inquiry_csv
    assert "平安融易（江苏）融资担保有限公," not in inquiry_csv


def test_community_enhanced_markdown_uses_projected_rows_without_changing_canonical_markdown() -> None:
    result = _result()
    sealed = seal_parse_result(result)
    fingerprint = sealed.integrity_fingerprint
    plugin = CreditReportPlugin()
    projection = derive_credit_report_projection(plugin, result, result.full_text)
    bundle = project_community_bundle(
        sealed,
        projection_data=projection.model_dump(mode="python"),
    )

    canonical = bundle.render_markdown()
    enhanced = bundle.render_enhanced_markdown()

    assert enhanced is not None
    assert 'docmirror:reading-profile version="2.0"' in enhanced
    assert 'source="community"' in enhanced
    assert "平安融易（江苏）融资担保有限公\n\n担保资格审查\n\n司" in canonical
    assert "| 2 | 2025-01-22 | 平安融易（江苏）融资担保有限公司 | 担保资格审查 | institution |" in enhanced
    assert (
        "| 103 | 2023-04-28 | 福建漳州农村商业银行股份有限公司 | "
        "法人代表、负责人、高管等资信审查 | institution |"
    ) in enhanced
    assert "不应调整有限公" not in enhanced
    assert sealed.integrity_fingerprint == fingerprint
    assert sealed.verify_integrity()


def test_write_outputs_persists_enhanced_reading_without_private_reading_projection(
    tmp_path,
    monkeypatch,
) -> None:
    result = _result()
    before = copy.deepcopy(result.model_dump(mode="python"))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("private reading projection must not render Community Markdown")

    monkeypatch.setattr(CreditReportPlugin, "reading_projection", fail_if_called)
    _task_id, written = write_outputs(
        result,
        tmp_path,
        file_id="001",
        task_id="enhanced_reading_test",
        include_mirror=False,
    )

    enhanced_path = written["enhanced_reading"]
    canonical = written["content"].read_text(encoding="utf-8")
    enhanced = enhanced_path.read_text(encoding="utf-8")
    assert enhanced_path.name == "001_enhanced_reading.md"
    assert "平安融易（江苏）融资担保有限公\n\n担保资格审查\n\n司" in canonical
    assert "| 2 | 2025-01-22 | 平安融易（江苏）融资担保有限公司 | 担保资格审查 | institution |" in enhanced
    assert 'source="community"' in enhanced
    assert result.model_dump(mode="python") == before


def test_reconstruction_fails_closed_without_geometry_and_preserves_fallback_row() -> None:
    result = _result()
    result.document_flow.nodes[3].bbox = None

    rows = reconstruct_institution_inquiry_rows(result)
    projection = derive_credit_report_projection(CreditReportPlugin(), result, result.full_text)
    inquiries = projection.datasets["inquiry_records"]

    assert [row["sequence"] for row in rows] == [3, 103]
    assert any(
        row["sequence"] == 2
        and row["inquiry_type"] == "institution"
        and row["source_refs"][0]["source"] == "native_text_ledger"
        for row in inquiries
    )
