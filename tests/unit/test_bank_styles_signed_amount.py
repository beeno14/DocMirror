# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for signed_amount bank statement style."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    LogicalTable,
    PageContent,
    ParseResult,
    RowProvenance,
    TableRow,
)
from docmirror.plugins.bank_statement.canonical import ensure_canonical_normalized
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extract_pipeline import run_bank_statement_extract
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry
from docmirror.plugins.bank_statement.styles import grid_standard
from docmirror.plugins.bank_statement.styles.signed_amount import (
    extract_transactions,
    normalize_record,
    parse_signed_amount,
    table_has_signed_amount_cells,
)

SIGNED_TABLE = [
    [
        ["交易日期", "摘要", "交易金额", "余额"],
        ["2024-01-01", "工资入账", "+5000.00", "5000.00"],
        ["2024-01-02", "消费", "-200.00", "4800.00"],
        ["2024-01-03", "转账", "-50.00", "4750.00"],
    ]
]


def test_parse_signed_amount_income_and_expense():
    amount, direction = parse_signed_amount("+5000.00")
    assert amount == 5000.0
    assert direction == "income"

    amount, direction = parse_signed_amount("-200.00")
    assert amount == 200.0
    assert direction == "expense"


def test_table_has_signed_amount_cells():
    assert table_has_signed_amount_cells(SIGNED_TABLE) is True
    split_table = [
        [
            ["交易日期", "摘要", "收入", "支出", "余额"],
            ["2024-01-01", "工资", "5000.00", "0.00", "5000.00"],
        ]
    ]
    assert table_has_signed_amount_cells(split_table) is False


def test_signed_amount_style_accepts_unsigned_credits_when_debits_keep_minus_sign():
    table = [
        [
            ["交易日期", "摘要", "发生额", "余额"],
            ["2024-01-01\n10:30:00", "消费", "-200.00", "4800.00"],
            ["2024-01-02", "结息", "5.27", "4805.27"],
        ]
    ]

    assert table_has_signed_amount_cells(table) is True

    ctx = StyleContext(tables=table, full_text="某银行交易明细", institution=None, page_count=1)
    records, _ = BankStyleParserRegistry().run(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert [record["normalized"]["direction"] for record in records] == ["expense", "income"]
    assert records[0]["normalized"]["timestamp"] == "2024-01-01T10:30:00"


def test_canonical_normalization_does_not_invent_zero_amount() -> None:
    normalized = ensure_canonical_normalized({"date": "2024-01-01"}, ["date", "amount", "amount_cny"])

    assert normalized["amount"] is None
    assert normalized["amount_cny"] is None


@pytest.mark.parametrize(
    ("source_amount", "expected_direction"),
    [("+0.00", "income"), ("-0.00", "expense")],
)
def test_explicit_signed_zero_amount_is_preserved(
    source_amount: str,
    expected_direction: str,
) -> None:
    parsed_amount, direction = parse_signed_amount(source_amount)

    assert parsed_amount == 0.0
    assert direction == expected_direction


def test_detector_signed_amount_style():
    ctx = StyleContext(
        tables=SIGNED_TABLE,
        full_text="某银行交易明细",
        institution=None,
        page_count=1,
    )
    result = BankStyleDetector().detect(ctx)
    assert result.primary_style == "signed_amount"


def test_registry_signed_amount_records():
    ctx = StyleContext(
        tables=SIGNED_TABLE,
        full_text="某银行交易明细",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) == 3
    assert records[0]["normalized"]["direction"] == "income"
    assert records[0]["normalized"]["amount"] == pytest.approx(5000.0)
    assert records[1]["normalized"]["direction"] == "expense"
    assert records[1]["normalized"]["amount"] == pytest.approx(200.0)


def _promoted_signed_parse_result(*, metadata: dict | None = None, omit_marker: str = "") -> SimpleNamespace:
    promoted = [
        "20231002",
        "网银费用",
        "网上企业银行-网上企业银行服务费",
        "-25.00",
        "1,547.50",
        "对公中间业务收入-网上其他收入",
        "",
    ]
    rows = [
        [
            "20231007",
            "商户清算",
            "20231007C106320166",
            "招行20231001/3840.00/9.60",
            "3,830.40",
            "5,377.90",
            "收单类业务清算平台待与商户清算款项",
        ],
        [
            "20231031",
            "对公提回贷",
            "福州北辰星汽车服务有限公司",
            "（闽A",
            "2,743.00",
            "10,267.06",
            "中国平安财产保险股份有限公司",
        ],
    ]
    footer = ["特别提示Special", "Notice", "第1页/共1页打印时间", "16时30分", "招商银行股份有限公司", "", ""]
    header_texts = [
        "账务明细清单",
        "Statement Of Account",
        "日期",
        "Date",
        "业务类型\nBusiness Type",
        "票据号",
        "Bill No.",
        "摘要\nDescription",
        "借方/贷方金额\nDebit/Credit Amount",
        "余额\nBalance",
        "对手户名\nCounterparty Account Name",
        "示例银行股份有限公司",
    ]
    header_texts = [text for text in header_texts if not omit_marker or omit_marker not in text]

    def text_block(content: str, index: int) -> SimpleNamespace:
        return SimpleNamespace(
            content=content,
            bbox=[10.0, float(index * 10), 500.0, float(index * 10 + 8)],
            evidence_ids=[f"ev:{index}"],
        )

    table = SimpleNamespace(
        table_id="geo_table_0",
        headers=promoted,
        rows=[
            TableRow(
                cells=[CellValue(text=value) for value in row],
                source_page=1,
                source_physical_id="geo_table_0",
                source_row_index=row_index,
            )
            for row_index, row in enumerate([*rows, footer])
        ],
        confidence=1.0,
        row_count=len(rows) + 1,
        metadata=metadata
        or {"header_source": "data_row", "preserve_headers": False, "source": "geometric_reconstructor"},
    )
    texts = [text_block(text, index) for index, text in enumerate(header_texts)]
    texts.extend(
        text_block("\n".join(value for value in row if value), index + len(texts))
        for index, row in enumerate([promoted, *rows])
    )
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table], texts=texts)
    return SimpleNamespace(pages=[page], logical_tables=[])


def _two_page_promoted_signed_parse_result() -> SimpleNamespace:
    first_page = _promoted_signed_parse_result().pages[0]
    second_page = _promoted_signed_parse_result().pages[0]
    second_page.page_number = 2
    second_page.source_page_number = 2
    second_table = second_page.tables[0]
    second_table.table_id = "geo_table_1"
    for row in second_table.rows:
        row.source_page = 2
        row.source_physical_id = "geo_table_1"
    for text in second_page.texts:
        text.evidence_ids = [f"page:2:{evidence_id}" for evidence_id in text.evidence_ids]
    return SimpleNamespace(pages=[first_page, second_page], logical_tables=[])


def _promoted_transaction_texts(page: SimpleNamespace) -> list[SimpleNamespace]:
    return [
        text
        for text in page.texts
        if (first_line := str(text.content or "").splitlines()[0])
        and len(first_line) == 8
        and first_line.isdigit()
    ]


def _promoted_style_context(parse_result: object) -> StyleContext:
    return StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=2,
        parse_result=parse_result,
    )


def test_signed_strategy_recovers_promoted_rows_across_two_pages() -> None:
    parse_result = _two_page_promoted_signed_parse_result()

    recovered = extract_transactions(
        _promoted_style_context(parse_result),
        BankStatementCommunityPlugin(),
    )

    assert len(recovered) == 6
    assert [row["_source"]["source_page"] for row in recovered] == [1, 1, 1, 2, 2, 2]
    assert [row["_source"]["table_id"] for row in recovered] == [
        "geo_table_0",
        "geo_table_0",
        "geo_table_0",
        "geo_table_1",
        "geo_table_1",
        "geo_table_1",
    ]
    assert [row["_source"]["source_row_index"] for row in recovered] == [-1, 0, 1, -1, 0, 1]
    assert all(row["_source"]["evidence_ids"] for row in recovered)
    assert all(row["_source"]["source_refs"][0].get("bbox") for row in recovered)


def test_signed_strategy_rejects_unproved_candidate_page_without_row_text_plane() -> None:
    parse_result = _two_page_promoted_signed_parse_result()
    second_page = parse_result.pages[1]
    transaction_text_ids = {id(text) for text in _promoted_transaction_texts(second_page)}
    second_page.texts = [
        text
        for text in second_page.texts
        if id(text) not in transaction_text_ids and "Debit/Credit Amount" not in str(text.content or "")
    ]

    assert second_page.tables[0].metadata["header_source"] == "data_row"
    assert extract_transactions(
        _promoted_style_context(parse_result),
        BankStatementCommunityPlugin(),
    ) == []


def test_signed_strategy_recovers_two_page_promoted_rows_through_read_view_wrapper() -> None:
    parse_result = _two_page_promoted_signed_parse_result()
    wrapper = SimpleNamespace(to_read_view=lambda: parse_result)

    recovered = extract_transactions(
        _promoted_style_context(wrapper),
        BankStatementCommunityPlugin(),
    )

    assert len(recovered) == 6
    assert [row["_source"]["source_page"] for row in recovered] == [1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-source-row-index",
        "nonmonotonic-source-row-index",
        "absent-table-id",
        "missing-evidence",
        "missing-bbox",
        "invalid-bbox",
    ],
)
def test_signed_strategy_rejects_incomplete_or_inconsistent_promoted_row_provenance(
    mutation: str,
) -> None:
    parse_result = _promoted_signed_parse_result()
    page = parse_result.pages[0]
    table = page.tables[0]
    transaction_text = _promoted_transaction_texts(page)[0]

    if mutation == "duplicate-source-row-index":
        table.rows[1].source_row_index = table.rows[0].source_row_index
    elif mutation == "nonmonotonic-source-row-index":
        table.rows[0].source_row_index = 2
        table.rows[1].source_row_index = 1
    elif mutation == "absent-table-id":
        table.table_id = ""
    elif mutation == "missing-evidence":
        transaction_text.evidence_ids = []
    elif mutation == "missing-bbox":
        transaction_text.bbox = None
    elif mutation == "invalid-bbox":
        transaction_text.bbox = [10.0, 20.0, 5.0, 15.0]

    assert extract_transactions(
        _promoted_style_context(parse_result),
        BankStatementCommunityPlugin(),
    ) == []


def test_signed_strategy_recovers_issuer_neutral_first_transaction_promoted_to_header() -> None:
    parse_result = _promoted_signed_parse_result()
    plugin = BankStatementCommunityPlugin()
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )

    raw = extract_transactions(ctx, plugin)
    normalized = [normalize_record(row, plugin) for row in raw]

    assert len(raw) == 3
    assert [row["日期"] for row in raw] == ["20231002", "20231007", "20231031"]
    assert raw[0]["票据号"] == ""
    assert raw[1]["票据号"] == "20231007C106320166"
    assert raw[2]["票据号"] == ""
    assert raw[2]["摘要"] == "福州北辰星汽车服务有限公司\n（闽A"
    assert [row["direction"] for row in normalized] == ["expense", "income", "income"]
    assert [row["amount"] for row in normalized] == [25.0, 3830.4, 2743.0]
    assert [row["transaction_name"] for row in normalized] == ["网银费用", "商户清算", "对公提回贷"]
    assert [row["voucher_number"] for row in normalized] == ["", "20231007C106320166", ""]
    assert [row["counter_party"] for row in normalized] == [
        "对公中间业务收入-网上其他收入",
        "收单类业务清算平台待与商户清算款项",
        "中国平安财产保险股份有限公司",
    ]
    assert [row["_source"]["source_row_index"] for row in raw] == [-1, 0, 1]
    assert [row["_source"]["source_row_role"] for row in raw] == [
        "promoted_header",
        "table_body",
        "table_body",
    ]
    assert [row["_source"]["reconstructed_row_index"] for row in raw] == [0, 1, 2]
    assert raw[0]["_source"]["header_source"] == "data_row"
    assert all(row["_source"]["evidence_ids"] for row in raw)


def test_signed_strategy_replaces_nonempty_partial_grid_with_proven_source_census() -> None:
    parse_result = _promoted_signed_parse_result()
    source_table = parse_result.pages[0].tables[0]
    partial_table = [
        ["日期", "业务类型", "票据号", "摘要", "借方/贷方金额", "余额", "对手户名"],
        *[
            [str(cell.text or "") for cell in row.cells]
            for row in source_table.rows[:-1]
        ],
    ]
    plugin = BankStatementCommunityPlugin()
    ctx = StyleContext(
        tables=[partial_table],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )

    partial = grid_standard.extract_transactions(ctx, plugin)
    recovered = extract_transactions(ctx, plugin)

    assert len(partial) == 2
    assert len(recovered) == 3
    assert [row["日期"] for row in recovered] == ["20231002", "20231007", "20231031"]
    assert recovered[0]["_source"]["source_row_role"] == "promoted_header"


@pytest.mark.parametrize(
    ("parse_result"),
    [
        _promoted_signed_parse_result(omit_marker="Debit/Credit Amount"),
        _promoted_signed_parse_result(
            metadata={"header_source": "column_names", "preserve_headers": True, "source": "geometric_reconstructor"}
        ),
    ],
)
def test_signed_strategy_rejects_unproven_promoted_header_recovery(parse_result: SimpleNamespace) -> None:
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )

    assert extract_transactions(ctx, BankStatementCommunityPlugin()) == []


def test_signed_strategy_rejects_promoted_row_without_exact_page_text_match() -> None:
    parse_result = _promoted_signed_parse_result()
    parse_result.pages[0].texts[-1].content += "\nsource-only-tail"
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )

    assert extract_transactions(ctx, BankStatementCommunityPlugin()) == []


def test_signed_strategy_rejects_extra_source_transaction_without_table_row() -> None:
    parse_result = _promoted_signed_parse_result()
    parse_result.pages[0].texts.append(
        SimpleNamespace(
            content="20231101\n对公转账出\n备用金\n-100.00\n10,167.51\n测试收款人",
            bbox=[10.0, 500.0, 500.0, 508.0],
            evidence_ids=["ev:extra-transaction"],
        )
    )
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )

    assert extract_transactions(ctx, BankStatementCommunityPlugin()) == []


def test_signed_strategy_rejects_ambiguous_money_pair() -> None:
    parse_result = _promoted_signed_parse_result()
    table_row = parse_result.pages[0].tables[0].rows[0]
    table_row.cells[3].text = "1.00"
    matching_text = next(
        text for text in parse_result.pages[0].texts if "招行20231001/3840.00/9.60" in text.content
    )
    matching_text.content = matching_text.content.replace("招行20231001/3840.00/9.60", "1.00")
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )

    assert extract_transactions(ctx, BankStatementCommunityPlugin()) == []


def test_signed_strategy_rejects_partial_multi_page_recovery() -> None:
    parse_result = _promoted_signed_parse_result()
    parse_result.pages.append(
        SimpleNamespace(
            page_number=2,
            source_page_number=2,
            tables=[],
            texts=[
                SimpleNamespace(
                    content="20231101\n对公转账出\n备用金\n-100.00\n10,167.51\n测试收款人",
                    bbox=[10.0, 100.0, 500.0, 108.0],
                    evidence_ids=["ev:page-2-transaction"],
                )
            ],
        )
    )
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=2,
        parse_result=parse_result,
    )

    assert extract_transactions(ctx, BankStatementCommunityPlugin()) == []


def test_promoted_header_recovery_uses_existing_eager_signed_parser() -> None:
    parse_result = _promoted_signed_parse_result()
    ctx = StyleContext(
        tables=[],
        full_text="账务明细清单 Statement Of Account",
        institution="cmb",
        page_count=1,
        parse_result=parse_result,
    )
    detector = BankStyleDetector().detect(ctx)
    registry = BankStyleParserRegistry()

    records, _identity = registry.run(detector, ctx, BankStatementCommunityPlugin())

    assert len(records) == 3
    assert registry.last_selection_diagnostics["selected_candidate"] == "parser:signed_amount"
    assert registry.last_selection_diagnostics["deployment_mode"] == "eager_fallback"
    assert registry.last_selection_diagnostics["completion_reason"] == "primary_parser_returned_no_candidate"
    assert registry.last_selection_diagnostics["attempted_strategies"][:2] == [
        "parser:grid_standard",
        "parser:signed_amount",
    ]


@pytest.mark.parametrize("document_type", ["bank_statement", "bank_reconciliation"])
def test_compatibility_headers_share_multi_page_signed_ledger_rules(document_type: str) -> None:
    """Compatibility glyphs and ledger semantics are generic across both bank document types."""
    headers = ["交易⽇期", "交易⾦额", "账户余额", "对⼿信息", "摘要"]
    values = [
        ["20220808", "+100.00", "1100.00", "甲公司", "转入"],
        ["20220808", "-20.00", "1080.00", "乙公司", "转出"],
        ["20220808", "-20.00", "1080.00", "乙公司", "转出"],
        ["20220809", "5.00", "1085.00", "", "结息"],
    ]
    pages = [1, 1, 2, 2]
    rows = [
        TableRow(
            cells=[CellValue(text=value) for value in row],
            source_page=page,
            source_physical_id=f"pt_{page}_0",
            source_row_index=index,
        )
        for index, (row, page) in enumerate(zip(values, pages))
    ]
    provenance = [
        RowProvenance(
            source_page=page,
            source_table_id=f"pt_{page}_0",
            source_row_index=index,
        )
        for index, page in enumerate(pages)
    ]
    parse_result = ParseResult(
        pages=[PageContent(page_number=1), PageContent(page_number=2)],
        entities=DocumentEntities(document_type=document_type),
        logical_tables=[
            LogicalTable(
                table_id="lt_compat",
                headers=headers,
                rows=rows,
                row_count=len(rows),
                data_row_estimate=len(rows),
                source_pages=[1, 2],
                source_physical_ids=["pt_1_0", "pt_2_0"],
                page_span=(1, 2),
                provenance=provenance,
                quality_passed=True,
            )
        ],
    )

    result = run_bank_statement_extract(
        parse_result,
        "中国农业银行账户交易明细\n交易⽇期 交易⾦额 账户余额 对⼿信息 摘要",
        BankStatementCommunityPlugin(),
    )

    assert len(result.records) == 4
    assert [record["normalized"]["direction"] for record in result.records] == [
        "income",
        "expense",
        "expense",
        "income",
    ]
    assert [record["source"]["source_page"] for record in result.records] == [1, 1, 2, 2]
    assert all(record["source"]["page_range"][0] == record["source"]["page_range"][1] for record in result.records)
    # The logical table proves what this candidate emitted, not that the source
    # document has no omitted terminal rows. Completeness remains unknown until
    # an issuer-provided denominator is independently established.
    assert result.style_meta.expected_primary_rows == 0
    assert result.style_meta.canonical_extracted == 4
    assert result.style_meta.extract_status == "degraded"
