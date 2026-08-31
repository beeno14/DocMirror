from __future__ import annotations

from docmirror.models.entities.parse_result import PageContent, ParseResult, TableBlock, TextBlock
from docmirror.plugins.bank_statement.community_plugin import plugin
from docmirror.plugins.bank_statement.context import build_digital_style_context
from docmirror.plugins.bank_statement.schema_guided_page_text import recover_schema_guided_page_text
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import _collect_table_candidates


def _block(content: str, *, page: int, index: int) -> TextBlock:
    y0 = 40.0 + index * 25.5
    return TextBlock(
        content=content,
        bbox=[36.0, y0, 540.0, y0 + (19.5 if "\n" in content else 9.75)],
        evidence_ids=[f"ev:{page:04d}:text:{index:06d}"],
    )


def _header(page: int) -> TextBlock:
    return _block("记账日期\n货币\n交易金额\n联机余额\n交易摘要\n对手信息", page=page, index=0)


def _row(page: int, index: int, *, date: str, amount: str, balance: str, party: str = "") -> TextBlock:
    suffix = f"\n{party}" if party else ""
    return _block(
        f"{date}\nCNY\n{amount}\n{balance}\n转账汇款{suffix}",
        page=page,
        index=index,
    )


def test_schema_is_carried_to_headerless_page_and_rows_are_atomic() -> None:
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    _header(1),
                    _row(1, 1, date="2024-01-01", amount="+10.00", balance="110.00", party="甲公司\n123"),
                ],
            ),
            PageContent(
                page_number=2,
                texts=[
                    _row(2, 0, date="2024-01-02", amount="-5.00", balance="105.00", party="乙公司\n456"),
                    _row(2, 1, date="2024-01-03", amount="+20.00", balance="125.00"),
                ],
            ),
        ]
    )

    recovered = recover_schema_guided_page_text(parse_result, source_route="digital")

    assert recovered.expected_rows == 3
    assert recovered.schema_pages == (1,)
    assert recovered.inherited_pages == (2,)
    assert recovered.records[1]["对手信息"] == "乙公司\n456"
    assert recovered.records[1]["_source"]["schema_inherited"] is True
    assert recovered.records[2]["对手信息"] == ""


def test_balance_break_does_not_suppress_source_rows() -> None:
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    _header(1),
                    _row(1, 1, date="2024-01-01", amount="+10.00", balance="110.00"),
                    _row(1, 2, date="2024-01-02", amount="-5.00", balance="999.00"),
                ],
            )
        ]
    )

    recovered = recover_schema_guided_page_text(parse_result, source_route="digital")

    assert recovered.expected_rows == 2
    full_text = "\n".join(text.content for page in parse_result.pages for text in page.texts)
    ctx = build_digital_style_context(parse_result, full_text)
    detection = BankStyleDetector().detect(ctx)
    candidates = _collect_table_candidates(detection, ctx, plugin)
    candidate = next(item for item in candidates if item.candidate_id == "schema_guided_page_text")

    assert len(candidate.records) == 2
    assert candidate.balance_chain_score < 1.0


def test_unproven_headerless_rows_and_prose_are_not_candidates() -> None:
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    _row(1, 0, date="2024-01-01", amount="+10.00", balance="110.00"),
                    _block("截至2024-01-02\n说明金额30.00\n参考余额140.00", page=1, index=1),
                ],
            )
        ]
    )

    assert recover_schema_guided_page_text(parse_result, source_route="digital").records == []
    assert recover_schema_guided_page_text(parse_result, source_route="scanned").records == []


def test_schema_guided_page_text_is_a_first_class_candidate() -> None:
    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                texts=[
                    _header(1),
                    _row(1, 1, date="2024-01-01", amount="+10.00", balance="110.00", party="甲公司"),
                ],
            ),
            PageContent(
                page_number=2,
                texts=[_row(2, 0, date="2024-01-02", amount="-5.00", balance="105.00", party="乙公司")],
            ),
        ]
    )
    full_text = "\n".join(text.content for page in parse_result.pages for text in page.texts)
    ctx = build_digital_style_context(parse_result, full_text)
    detection = BankStyleDetector().detect(ctx)

    candidates = _collect_table_candidates(detection, ctx, plugin)
    candidate = next(item for item in candidates if item.candidate_id == "schema_guided_page_text")

    assert len(candidate.records) == 2
    assert candidate.canonical_rows == 2
    assert candidate.source_page_coverage == 1.0
    assert candidate.balance_chain_score == 1.0


def test_plugin_recovers_tableless_continuation_when_core_would_skip() -> None:
    """Mixed table coverage is recovered entirely inside the bank plugin."""

    parse_result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        table_id="native-page-1",
                        headers=["记账日期", "货币", "交易金额", "联机余额", "交易摘要", "对手信息"],
                        rows=[],
                        page=1,
                    )
                ],
                texts=[
                    _row(1, 0, date="2024-01-01", amount="+10.00", balance="110.00", party="甲公司")
                ],
            ),
            PageContent(
                page_number=2,
                texts=[
                    _row(2, 0, date="2024-01-02", amount="-5.00", balance="105.00", party="乙公司")
                ],
            ),
        ]
    )
    full_text = "\n".join(text.content for page in parse_result.pages for text in page.texts)
    ctx = build_digital_style_context(parse_result, full_text)
    detection = BankStyleDetector().detect(ctx)

    recovered = recover_schema_guided_page_text(parse_result, source_route="digital")
    candidates = _collect_table_candidates(detection, ctx, plugin)
    candidate = next(item for item in candidates if item.candidate_id == "schema_guided_page_text")

    assert recovered.schema_pages == (1,)
    assert recovered.inherited_pages == (2,)
    assert [record["_source"]["source_page"] for record in candidate.records] == [1, 2]
    assert len(parse_result.pages[1].tables) == 0
