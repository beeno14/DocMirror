# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from docmirror.models.entities.parse_result import (
    CellValue,
    KeyValuePair,
    LogicalTable,
    PageContent,
    ParseResult,
    RowProvenance,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.mirror.document_flow import (
    DocumentFlowGraph,
    ReadingFlow,
    StructureNode,
)
from docmirror.plugins.credit_report.enterprise_native.ir import (
    build_canonical_enterprise_document,
)


def _row(*values: str, page: int = 1) -> TableRow:
    return TableRow(cells=[CellValue(text=value) for value in values], source_page=page)


def test_duplicate_source_page_numbers_receive_unique_page_instance_ids() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=7,
                texts=[TextBlock(content="first")],
                tables=[TableBlock(table_id="same", metadata={"raw_rows": [["a", "1"]]})],
            ),
            PageContent(
                page_number=2,
                source_page_number=7,
                texts=[TextBlock(content="second")],
                tables=[TableBlock(table_id="same", metadata={"raw_rows": [["b", "2"]]})],
            ),
        ]
    )

    document = build_canonical_enterprise_document(result)
    unit_ids = [unit.unit_id for unit in document.entity_context.units]
    table_ids = [unit.table_id for unit in document.entity_context.units if unit.kind == "table"]

    assert len(unit_ids) == len(set(unit_ids)) == 4
    assert len(table_ids) == len(set(table_ids)) == 2
    assert document.source_page_count == 1
    assert document.pages[0].source_page_number == 7
    assert document.pages[0].page_instances == (1, 2)
    duplicate_flag = next(
        flag
        for flag in document.input_quality_flags
        if flag["code"] == "ENTERPRISE_INPUT_DUPLICATE_PAGE_NUMBERS"
    )
    assert duplicate_flag["severity"] == "warning"
    assert duplicate_flag["status"] == "logical_page_instances_reconstructed"
    assert document.content_conserved is True


def test_ir_consumes_key_values_captions_flow_order_and_flow_only_nodes() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[TextBlock(content="identity", reading_order=30)],
                key_values=[KeyValuePair(key="name", value="Example Ltd", reading_order=20)],
                tables=[
                    TableBlock(
                        table_id="profile",
                        caption="profile caption",
                        reading_order=10,
                        metadata={"raw_rows": [["field", "value"]]},
                    )
                ],
            )
        ],
        document_flow=DocumentFlowGraph(
            nodes=[
                StructureNode(
                    node_id="flow-identity",
                    type="paragraph",
                    page=1,
                    text="identity",
                    fact_refs=["text:p1:0"],
                ),
                StructureNode(
                    node_id="flow-note",
                    type="caption",
                    page=1,
                    text="business note retained only by document flow",
                ),
            ],
            reading_flow=[
                ReadingFlow(
                    flow_id="main",
                    type="main_reading_order",
                    node_ids=["flow-note", "flow-identity"],
                )
            ],
        ),
    )

    document = build_canonical_enterprise_document(result)
    views = [unit.source_view for unit in document.source_units]
    texts = [unit.text for unit in document.entity_context.units if unit.kind != "table"]

    assert "key_value" in views
    assert "table_caption" in views
    assert views.count("document_flow") == 2
    assert texts.count("identity") == 1
    assert "business note retained only by document flow" in texts
    assert any(
        unit.key == "name" and unit.value == "Example Ltd"
        for unit in document.source_units
    )
    assert all(unit.represented for unit in document.source_units)
    assert document.content_conserved is True


def test_page_free_table_view_prefers_parse_result_logical_tables() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="physical-1",
                        metadata={"raw_rows": [["field", "value-1"]]},
                    )
                ],
            ),
            PageContent(
                page_number=2,
                tables=[
                    TableBlock(
                        table_id="physical-2",
                        metadata={"raw_rows": [["field", "value-2"]]},
                    )
                ],
            ),
        ],
        logical_tables=[
            LogicalTable(
                logical_id="profile-logical",
                headers=["field", "value"],
                rows=[_row("name", "Example Ltd", page=1), _row("status", "normal", page=2)],
                source_physical_ids=["physical-1", "physical-2"],
                source_pages=[1, 2],
                provenance=[
                    RowProvenance(source_page=1, source_table_id="physical-1"),
                    RowProvenance(source_page=2, source_table_id="physical-2"),
                ],
                quality_passed=True,
            )
        ],
    )

    document = build_canonical_enterprise_document(result)

    assert document.page_free_table_rows == (
        (
            "profile-logical",
            (("field", "value"), ("name", "Example Ltd"), ("status", "normal")),
        ),
    )
    assert document.logical_tables[0].source_unit_ids
    logical_source = next(unit for unit in document.source_units if unit.source_view == "logical_table")
    assert logical_source.disposition == "logical_table_view"
    assert logical_source.represented is True
    assert document.content_conserved is True


def test_logical_table_only_parseresult_remains_extractable_and_is_not_bad_input() -> None:
    result = ParseResult(
        logical_tables=[
            LogicalTable(
                logical_id="identity",
                headers=["field", "value"],
                rows=[_row("company", "Example Ltd")],
                source_pages=[1],
                quality_passed=True,
            )
        ]
    )

    document = build_canonical_enterprise_document(result)

    assert any(unit.kind == "table" for unit in document.entity_context.units)
    assert document.table_rows[0][1] == "identity"
    assert document.page_free_table_rows[0][0] == "identity"
    assert document.content_conserved is True
    assert any(
        flag["code"] == "ENTERPRISE_INPUT_NO_PHYSICAL_PAGES"
        and flag["status"] == "alternate_source_views_available"
        for flag in document.input_quality_flags
    )
    assert not any(flag["status"] == "bad_input" for flag in document.input_quality_flags)
