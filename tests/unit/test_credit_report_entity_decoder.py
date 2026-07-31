from __future__ import annotations

import copy
from types import SimpleNamespace

from docmirror.plugins.credit_report.shared.entity_decoder import (
    decode_credit_report_entities,
)


def _table(table_id: str, rows: list[list[str]], bbox: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=bbox,
    )


def _text(content: str, bbox: list[float]) -> SimpleNamespace:
    return SimpleNamespace(content=content, bbox=bbox)


def _page(
    number: int,
    *,
    texts: list[SimpleNamespace] | None = None,
    tables: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=number,
        source_page_number=number,
        width=600,
        height=800,
        texts=texts or [],
        tables=tables or [],
    )


def test_header_only_table_remains_open_across_three_pages() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                texts=[_text("第 1 页，共 3 页", [260, 775, 340, 790])],
                tables=[
                    _table(
                        "header",
                        [["账户编号", "开立日期", "金额"]],
                        [20, 650, 580, 770],
                    )
                ],
            ),
            _page(
                2,
                texts=[_text("第 2 页，共 3 页", [260, 775, 340, 790])],
                tables=[
                    _table(
                        "body_2",
                        [["A0001", "2025-01-01", "100"]],
                        [20, 20, 580, 770],
                    )
                ],
            ),
            _page(
                3,
                texts=[
                    _text("说明", [270, 430, 330, 450]),
                    _text("本节为新的正文。", [20, 470, 580, 500]),
                    _text("第 3 页，共 3 页", [260, 775, 340, 790]),
                ],
                tables=[
                    _table(
                        "body_3",
                        [["A0002", "2025-01-02", "200"]],
                        [20, 20, 580, 390],
                    )
                ],
            ),
        ]
    )

    decoded = decode_credit_report_entities(result, report_family="enterprise")
    table_entity = decoded.entity_for_unit("table:header")

    assert decoded.content_conserved is True
    assert len(decoded.furniture_unit_ids) == 3
    assert table_entity is not None
    assert table_entity.kind == "table"
    assert table_entity.unit_ids == ("table:header", "table:body_2", "table:body_3")
    assert table_entity.pages == (1, 2, 3)
    assert decoded.same_table_entity("header", "body_2") is True
    assert decoded.same_table_entity("body_2", "body_3") is True
    notes_entity = decoded.entity_for_unit("text:p3:0")
    assert notes_entity is not None
    assert notes_entity.entity_id != table_entity.entity_id


def test_personal_borderless_ledger_continues_but_notes_start_new_entity() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                texts=[
                    _text("个人查询记录明细", [20, 100, 580, 125]),
                    _text("1 2025年01月03日 本人 本人查询", [20, 140, 580, 160]),
                    _text("2 2025年01月02日 本人 本人查询", [20, 740, 580, 765]),
                    _text("第1页，共3页", [260, 775, 340, 790]),
                ],
            ),
            _page(
                2,
                texts=[
                    _text("3 2025年01月01日 本人 本人查询", [20, 20, 580, 45]),
                    _text("第2页，共3页", [260, 775, 340, 790]),
                ],
            ),
            _page(
                3,
                texts=[
                    _text("说明", [270, 40, 330, 60]),
                    _text("1. 本报告说明。", [20, 90, 580, 120]),
                    _text("第3页，共3页", [260, 775, 340, 790]),
                ],
            ),
        ]
    )

    before = copy.deepcopy(result)
    decoded = decode_credit_report_entities(result, report_family="personal_brief")

    row_2_entity = decoded.entity_for_unit("text:p1:2")
    row_3_entity = decoded.entity_for_unit("text:p2:0")
    notes_entity = decoded.entity_for_unit("text:p3:0")
    assert decoded.content_conserved is True
    assert row_2_entity is not None and row_3_entity is not None
    assert row_2_entity.entity_id == row_3_entity.entity_id
    assert notes_entity is not None
    assert notes_entity.entity_id != row_3_entity.entity_id
    assert result == before


def test_table_text_transition_is_ranked_and_preserved() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                texts=[_text("信息来源机构：中国人民银行", [20, 360, 580, 390])],
                tables=[
                    _table(
                        "summary",
                        [
                            ["账户编号", "金额"],
                            ["A0001", "100"],
                        ],
                        [20, 100, 580, 350],
                    )
                ],
            )
        ]
    )

    decoded = decode_credit_report_entities(result, report_family="enterprise")
    decision = decoded.decision_between("table:summary", "text:p1:0")

    assert decision is not None
    assert {item.action for item in decision.hypotheses} == {
        "same_table",
        "different_table",
        "table_to_text_related",
        "table_to_text_unrelated",
        "text_to_table_related",
        "text_to_table_unrelated",
        "same_text_section",
        "different_text_section",
        "new_section",
    }
    assert decision.selected == "table_to_text_related"
    assert decision.continues_entity is True
    assert decoded.content_conserved is True


def test_text_order_is_preserved_inside_y_sorted_entities() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                texts=[
                    _text("第二个来源文本块。", [20, 140, 580, 160]),
                    _text("说明", [20, 100, 580, 120]),
                ],
            )
        ]
    )

    decoded = decode_credit_report_entities(result, report_family="personal_brief")

    assert [unit.text for unit in decoded.units] == ["说明", "第二个来源文本块。"]
    assert decoded.ordered_text_blocks() == (
        (1, "第二个来源文本块。"),
        (1, "说明"),
    )


def test_new_table_schema_closes_the_previous_table_entity() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                tables=[
                    _table(
                        "accounts",
                        [["账户编号", "金额"], ["A001", "100"]],
                        [20, 100, 580, 430],
                    )
                ],
            ),
            _page(
                2,
                tables=[
                    _table(
                        "queries",
                        [["查询日期", "查询机构", "查询原因"], ["2026-01-01", "某银行", "贷后管理"]],
                        [20, 30, 580, 280],
                    )
                ],
            ),
        ]
    )

    decoded = decode_credit_report_entities(result, report_family="enterprise")
    decision = decoded.decision_between("table:accounts", "table:queries")

    assert decision is not None
    assert decision.selected == "different_table"
    assert decision.continues_entity is False
    assert decoded.same_table_entity("accounts", "queries") is False


def test_open_body_text_section_continues_until_it_reaches_a_heading() -> None:
    result = SimpleNamespace(
        pages=[
            _page(
                1,
                texts=[_text("这是跨页正文的第一部分，没有结束标点", [20, 650, 580, 770])],
            ),
            _page(
                2,
                texts=[_text("这是跨页正文的第二部分，仍然没有结束标点", [20, 20, 580, 770])],
            ),
            _page(
                3,
                texts=[
                    _text("这是正文的最后一部分。", [20, 20, 580, 260]),
                    _text("查询记录", [20, 320, 580, 350]),
                ],
            ),
        ]
    )

    decoded = decode_credit_report_entities(result, report_family="enterprise")
    body_entity = decoded.entity_for_unit("text:p1:0")
    heading_entity = decoded.entity_for_unit("text:p3:1")

    assert body_entity is not None
    assert body_entity.pages == (1, 2, 3)
    assert body_entity.unit_ids == ("text:p1:0", "text:p2:0", "text:p3:0")
    assert heading_entity is not None
    assert heading_entity.entity_id != body_entity.entity_id
