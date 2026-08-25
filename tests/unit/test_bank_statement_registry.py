# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.

"""Bank statement template registry and canonical style integration."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

import docmirror.plugins.bank_statement.style_registry as style_registry
from docmirror.models.sealed import seal_parse_result
from docmirror.output.mirror_projector import project_mirror
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.context import StyleContext
from docmirror.plugins.bank_statement.extraction_dispatch import (
    BankExtractionPolicy,
    BankExtractionRoute,
)
from docmirror.plugins.bank_statement.style_detector import BankStyleDetector, StyleDetectionResult
from docmirror.plugins.bank_statement.style_registry import (
    BankStyleParserRegistry,
    BankTableCandidate,
    _candidate_expected_rows,
    _candidate_sequence_continuity,
    _select_candidate,
)
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence, _recover_cross_page_wide_tables


def _candidate_for_authority_test(
    candidate_id: str,
    rows: int,
    evidence: RowCountEvidence,
) -> BankTableCandidate:
    records = [{"normalized": {"date": "2025-01-01", "amount": 1.0, "direction": "income"}}] * rows
    return BankTableCandidate(
        candidate_id=candidate_id,
        records=records,
        source="native_wide_table",
        canonical_rows=rows,
        directional_rows=rows,
        source_page_rows=rows,
        expected_rows=evidence,
        balance_chain_score=0.5,
        field_completeness=1.0,
        score=1.0,
        canonical_coverage=1.0,
        source_page_coverage=1.0,
        extraction_confidence=0.9,
        source_column_width=5.0,
        sequence_continuity=1.0,
    )


@pytest.mark.parametrize(
    "source",
    [
        "complete_page_local_sequences",
        "ccb_primary_source_sequence",
        "cmb_primary_source_rows",
        "native_page_datetime_census",
        "native_page_signed_ledger_census",
        "ocr_page_ordinal_census",
    ],
)
def test_short_row_plane_signal_cannot_beat_fuller_candidate(source: str) -> None:
    short = _candidate_for_authority_test("recovery:short", 4, RowCountEvidence(4, source, 0.99))
    fuller = _candidate_for_authority_test("parser:full", 5, RowCountEvidence(5, "candidate_rows", 0.55))

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is fuller


@pytest.mark.parametrize("fuller_route", ["ocr_implicit_table", "positioned_record_block"])
def test_fuller_candidate_guard_is_route_agnostic(fuller_route: str) -> None:
    short = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            2,
            RowCountEvidence(2, "positioned_date_anchors", 0.80),
        ),
        sequence_continuity=1.0,
    )
    fuller = replace(
        _candidate_for_authority_test(
            fuller_route,
            3,
            RowCountEvidence(3, "candidate_rows", 0.55),
        ),
        sequence_continuity=0.95,
    )

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is fuller


def test_fuller_candidate_guard_preserves_materially_richer_shorter_candidate() -> None:
    short = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            2,
            RowCountEvidence(2, "positioned_date_anchors", 0.80),
        ),
        sequence_continuity=1.0,
        source_column_width=8.0,
    )
    fuller = replace(
        _candidate_for_authority_test(
            "ocr_implicit_table",
            3,
            RowCountEvidence(3, "candidate_rows", 0.55),
        ),
        field_completeness=0.70,
        sequence_continuity=0.95,
        source_column_width=6.0,
    )

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is short


def _candidate_records_on_pages(rows: int, pages: list[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(rows):
        page = pages[index % len(pages)]
        records.append(
            {
                "normalized": {"date": "2025-01-01", "amount": 1.0, "direction": "income"},
                "_source": {
                    "source_page": page,
                    "page_range": [page, page],
                    "bbox": [10.0, float(index * 10), 100.0, float(index * 10 + 8)],
                },
            }
        )
    return records


def test_fuller_candidate_can_use_strict_source_page_superset_over_local_balance_score() -> None:
    short = replace(
        _candidate_for_authority_test(
            "native_wide_table:0",
            4,
            RowCountEvidence(4, "native_wide_rows", 0.70),
        ),
        records=_candidate_records_on_pages(4, [2, 3]),
        balance_chain_score=0.98,
        sequence_continuity=1.0,
        score=0.98,
    )
    fuller = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            10,
            RowCountEvidence(10, "positioned_date_anchors", 0.80),
        ),
        records=_candidate_records_on_pages(10, [1, 2, 3, 4, 5]),
        balance_chain_score=0.50,
        sequence_continuity=0.0,
        score=0.88,
    )

    selected, _diagnostics = _select_candidate([short, fuller])

    assert selected is fuller


def test_fuller_candidate_with_same_source_pages_still_needs_semantic_parity() -> None:
    short = replace(
        _candidate_for_authority_test(
            "native_wide_table:0",
            4,
            RowCountEvidence(4, "native_wide_rows", 0.70),
        ),
        records=_candidate_records_on_pages(4, [1, 2]),
        balance_chain_score=0.98,
        sequence_continuity=1.0,
        score=0.98,
    )
    noisier = replace(
        _candidate_for_authority_test(
            "evidence_atom",
            8,
            RowCountEvidence(8, "positioned_date_anchors", 0.80),
        ),
        records=_candidate_records_on_pages(8, [1, 2]),
        balance_chain_score=0.50,
        sequence_continuity=0.0,
        score=0.88,
    )

    selected, _diagnostics = _select_candidate([short, noisier])

    assert selected is short


def _sequenced_source_row(sequence: int, page: int, y: float) -> tuple[dict[str, str], dict[str, object]]:
    return (
        {"sequence_no": str(sequence)},
        {
            "_source": {
                "source_page": page,
                "page_range": [page, page],
                "bbox": [10.0, y, 100.0, y + 8.0],
            }
        },
    )


def test_sequence_continuity_accepts_only_forward_source_order_resets() -> None:
    rows = [
        _sequenced_source_row(1, 1, 100.0),
        _sequenced_source_row(2, 1, 120.0),
        _sequenced_source_row(1, 2, 80.0),
        _sequenced_source_row(2, 2, 100.0),
        _sequenced_source_row(3, 2, 120.0),
    ]

    assert _candidate_sequence_continuity(
        [normalized for normalized, _transaction in rows],
        [transaction for _normalized, transaction in rows],
    ) == 1.0


def test_sequence_continuity_rejects_a_duplicate_plane_page_rewind() -> None:
    rows = [
        _sequenced_source_row(1, 1, 100.0),
        _sequenced_source_row(2, 2, 100.0),
        _sequenced_source_row(1, 1, 101.0),
        _sequenced_source_row(2, 2, 101.0),
    ]

    score = _candidate_sequence_continuity(
        [normalized for normalized, _transaction in rows],
        [transaction for _normalized, transaction in rows],
    )

    assert score == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "source",
    [
        "complete_page_local_sequences",
        "ccb_primary_source_sequence",
        "cmb_primary_source_rows",
        "native_page_datetime_census",
        "native_page_signed_ledger_census",
        "ocr_page_ordinal_census",
    ],
)
def test_row_plane_signal_is_clamped_below_public_count_authority(source: str) -> None:
    evidence = _candidate_expected_rows(RowCountEvidence(4, source, 0.99))

    assert evidence == RowCountEvidence(4, source, 0.80)


def test_candidate_count_is_not_replaced_by_separate_row_plane_signal() -> None:
    evidence = _candidate_expected_rows(
        RowCountEvidence(4, "native_page_datetime_census", 0.99),
        count=5,
        source="native_wide_rows",
        confidence=0.70,
    )

    assert evidence == RowCountEvidence(5, "native_wide_rows", 0.70)


@pytest.mark.parametrize(
    "source",
    [
        "positioned_record_blocks",
        "physical_rows",
        "positioned_date_anchors",
        "page_transaction_anchors",
    ],
)
def test_candidate_fallback_row_plane_source_is_clamped_below_authority(source: str) -> None:
    evidence = _candidate_expected_rows(
        RowCountEvidence(0, "candidate_rows", 0.55),
        count=7,
        source=source,
        confidence=0.99,
    )

    assert evidence == RowCountEvidence(7, source, 0.80)


@pytest.mark.parametrize(
    "source,confidence",
    [
        ("split_footer", 0.98),
        ("header_total", 0.94),
        ("cumulative_footer_total", 0.99),
        ("page_footer", 0.90),
    ],
)
def test_issuer_count_remains_authoritative_for_candidate_selection(source: str, confidence: float) -> None:
    complete = _candidate_for_authority_test("recovery:complete", 4, RowCountEvidence(4, source, confidence))
    extra = _candidate_for_authority_test("parser:extra", 5, RowCountEvidence(4, source, confidence))

    selected, _diagnostics = _select_candidate([complete, extra])

    assert selected is complete


def test_builtin_templates_registered():
    registry_module = pytest.importorskip(
        "docmirror_enterprise.plugins.bank_statement.configs.registry",
        reason="enterprise bank-statement templates are not available in OSS CI",
    )

    registry_module.reset_registry()
    reg = registry_module.ensure_builtin_templates()
    ids = {t["template_id"] for t in reg.list_templates()}
    assert "generic" in ids
    assert "icbc_personal_v2022" in ids
    assert reg.template_count >= 3


def test_style_registry_extracts_three_transactions_from_clean_table():
    ctx = StyleContext(
        tables=[
            [
                ["交易日期", "摘要", "收入", "支出", "余额"],
                ["2024-01-01", "工资入账", "5000.00", "0.00", "8000.00"],
                ["2024-01-02", "转账支出", "0.00", "200.00", "7800.00"],
                ["2024-01-03", "消费", "0.00", "50.00", "7750.00"],
            ]
        ],
        full_text="中国工商银行\n个人客户交易明细\n户名：张三",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    assert detection.primary_style == "split_debit_credit"
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3


def test_preferred_context_tables_exclude_document_acquisition_planes(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = StyleContext(
        tables=[
            [
                ["交易日期", "摘要", "收入", "支出", "余额"],
                ["2024-01-02", "continued", "", "2.00", "7.00"],
                ["2024-01-03", "continued", "3.00", "", "10.00"],
            ]
        ],
        full_text="document-wide text must stay outside this local continuation batch",
        institution=None,
        page_count=14,
        prefer_context_tables=True,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("document-wide acquisition ran inside a context-tables-only batch")

    for name in (
        "recovered_native_datetime_row_evidence",
        "recover_evidence_atom_bank_tables",
        "_semantic_text_table_candidates",
        "collect_physical_tables_from_parse_result",
        "recover_positioned_record_block_bank_tables",
        "recover_wide_bank_tables",
        "recover_ocr_implicit_ledger_tables",
    ):
        monkeypatch.setattr(style_registry, name, forbidden)

    candidates = style_registry._collect_table_candidates(
        BankStyleDetector().detect(ctx),
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert candidates
    assert all(candidate.candidate_id.startswith("parser:") for candidate in candidates)
    assert max(len(candidate.records) for candidate in candidates) == 2


def _physical_candidate_parse_result(*, blank_body_geometry: bool = False) -> SimpleNamespace:
    headers = ["交易日期", "摘要", "收入", "支出", "余额"]
    values = ["2024-01-02", "工资", "100.00", "", "100.00"]
    header_boxes = [[float(index * 60), 1.0, float((index + 1) * 60), 10.0] for index in range(5)]
    body_boxes = (
        [None] * 5
        if blank_body_geometry
        else [
            [10.0, 20.0, 55.0, 32.0],
            [55.0, 20.0, 115.0, 32.0],
            [115.0, 20.0, 175.0, 32.0],
            [175.0, 20.0, 235.0, 32.0],
            [235.0, 20.0, 300.0, 32.0],
        ]
    )
    body_evidence = [[], [], [], [], []] if blank_body_geometry else [
        ["ev:body:date"],
        ["ev:body:summary"],
        ["ev:body:income"],
        [],
        ["ev:body:balance"],
    ]
    cells = [
        SimpleNamespace(
            text=value,
            cleaned=None,
            bbox=None,
            evidence_ids=[],
            source_cell_refs=[],
        )
        for value in values
    ]
    row = SimpleNamespace(
        cells=cells,
        source_page=3,
        source_physical_id="pt_3_7",
        source_row_index=4,
        source_cell_refs=[],
    )
    table = SimpleNamespace(
        headers=headers,
        rows=[row],
        table_id="pt_3_7",
        bbox=[0.0, 0.0, 999.0, 999.0],
        metadata={
            "raw_rows": [headers, values],
            "geometry": {
                "cell_bboxes": [header_boxes, body_boxes],
                "cell_evidence_ids": [
                    [[f"ev:header:{index}"] for index in range(5)],
                    body_evidence,
                ],
            },
        },
    )
    page = SimpleNamespace(page_number=3, source_page_number=3, tables=[table], texts=[])
    return SimpleNamespace(pages=[page], logical_tables=[], full_text="")


def _physical_candidate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    blank_body_geometry: bool = False,
) -> style_registry.BankTableCandidate:
    parse_result = _physical_candidate_parse_result(blank_body_geometry=blank_body_geometry)
    policy = BankExtractionPolicy(
        route=BankExtractionRoute.DIGITAL,
        allowed_parser_ids=frozenset(),
        allow_physical_tables=True,
    )
    ctx = StyleContext(
        tables=[],
        full_text="",
        institution=None,
        page_count=1,
        parse_result=parse_result,
        extraction_route=policy.route,
        extraction_policy=policy,
    )
    monkeypatch.setattr(
        style_registry,
        "recovered_native_datetime_row_evidence",
        lambda *_args, **_kwargs: (0, "", 0.0),
    )
    monkeypatch.setattr(style_registry, "physical_transaction_row_estimate", lambda _result: 1)

    candidates = style_registry._collect_table_candidates(
        StyleDetectionResult(primary_style="grid_standard", parser_chain=[]),
        ctx,
        BankStatementCommunityPlugin(),
    )

    return next(candidate for candidate in candidates if candidate.candidate_id == "physical_table")


def test_physical_candidate_attaches_exact_body_row_geometry_and_never_header_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _physical_candidate(monkeypatch)

    assert len(candidate.records) == 1
    source = candidate.records[0]["_source"]
    assert source["source"] == "canonical_physical_table"
    assert source["source_page"] == 3
    assert source["page_range"] == [3, 3]
    assert source["table_id"] == "pt_3_7"
    assert source["source_row_index"] == 4
    assert source["bbox"] == [10.0, 20.0, 300.0, 32.0]
    assert source["evidence_ids"] == [
        "ev:body:date",
        "ev:body:summary",
        "ev:body:income",
        "ev:body:balance",
    ]
    assert len(source["source_cell_refs"]) == 5
    assert {ref["raw_row"] for ref in source["source_cell_refs"]} == {1}
    assert {ref["row"] for ref in source["source_cell_refs"]} == {4}
    assert not any(value.startswith("ev:header:") for value in source["evidence_ids"])
    assert source["bbox"] != [0.0, 0.0, 999.0, 999.0]


def test_physical_candidate_blank_row_geometry_keeps_identity_without_borrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _physical_candidate(monkeypatch, blank_body_geometry=True)

    assert len(candidate.records) == 1
    source = candidate.records[0]["_source"]
    assert source["source"] == "canonical_physical_table"
    assert source["source_page"] == 3
    assert source["page_range"] == [3, 3]
    assert source["table_id"] == "pt_3_7"
    assert source["source_row_index"] == 4
    assert "bbox" not in source
    assert "evidence_ids" not in source
    assert "source_cell_refs" not in source


def test_style_registry_extracts_wide_debit_credit_table_and_skips_footer():
    ctx = StyleContext(
        tables=[
            [
                [
                    "序号",
                    "会计日期",
                    "交易日期",
                    "交易名称",
                    "借方发生额",
                    "贷方发生额",
                    "余额",
                    "对方账号",
                    "对方户名",
                    "摘要",
                ],
                [
                    "1",
                    "20251114",
                    "20251114",
                    "来账",
                    "",
                    "120,000.00",
                    "139,038.63",
                    "011101421000 9630",
                    "重庆正大华日软 件有限公司",
                    "往来款",
                ],
                [
                    "2",
                    "20251114",
                    "20251114",
                    "代付",
                    "97,462.92",
                    "",
                    "41,575.71",
                    "641106012890 900100012499",
                    "应付代收业务款 项",
                    "代发工资",
                ],
                ["当前账单借方发生数： 1", "当前账单贷方发生数：1", "", "", "", "", "", "", "", ""],
            ]
        ],
        full_text="交通银行\n当前账单借方发生数：1 当前账单贷方发生数：1 本月累计借方发生额：97,462.92 本月累计贷方发生额：120,000.00",
        institution=None,
        page_count=1,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)

    assert len(records) == 2
    assert [r["normalized"]["direction"] for r in records] == ["income", "expense"]
    assert records[0]["normalized"]["counter_account"] == "0111014210009630"
    assert records[0]["normalized"]["counter_party"] == "重庆正大华日软件有限公司"


def test_cross_page_native_income_expense_table_inherits_header():
    page_tables = [
        [
            ["序 号", "交易日期", "交易时 间", "支出金额", "收入金额", "余额", "对方账号", "对方户名"],
            [
                "1",
                "2023-12- 28",
                "15:28:5 3",
                "",
                "2,800.00",
                "2,932.04",
                "24020034091 00018033",
                "贵阳世钟 汽车配件 有限公司",
            ],
            ["2", "2023-12- 27", "14:06:0 2", "7.00", "", "132.04", "60220903", "网上银行 结算手续 费收入"],
        ],
        [
            [
                "3",
                "2023-12- 27",
                "14:06:0 2",
                "10,500.00",
                "",
                "139.04",
                "32050161716 000000050",
                "无锡市融 达汽车零 部件有限 公司",
            ],
            ["4", "2023-12- 27", "13:58:5 9", "", "10,500.00", "10,639.04", "62284810431 55907917", "张淑红"],
        ],
    ]
    tables = _recover_cross_page_wide_tables(page_tables)

    assert len(tables) == 1
    assert len(tables[0]) == 5
    assert tables[0][1][1] == "2023-12-28"
    assert tables[0][1][2] == "15:28:53"

    ctx = StyleContext(
        tables=tables,
        full_text="收入总金额：13300.00 收入总笔数：2 支出总金额：10507.00 支出总笔数：2",
        institution=None,
        page_count=2,
    )
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(BankStyleDetector().detect(ctx), ctx, plugin)
    norms = [record["normalized"] for record in records]

    assert len(records) == 4
    assert sum(1 for norm in norms if norm["direction"] == "income") == 2
    assert sum(1 for norm in norms if norm["direction"] == "expense") == 2
    assert norms[0]["counter_party"] == "贵阳世钟汽车配件有限公司"


def test_removed_detector_is_not_registered():
    pytest.importorskip("docmirror_enterprise", reason="enterprise package is not available in OSS CI")

    with pytest.raises(ImportError):
        from docmirror_enterprise.plugins.bank_statement.detectors.template_detector import (  # noqa: F401
            BankStatementDetector,
        )


@pytest.mark.skipif(
    not os.environ.get("DOCMIRROR_RUN_SYNTHETIC_TESTS"),
    reason="Synthetic PDF OCR test requires DOCMIRROR_RUN_SYNTHETIC_TESTS=1",
)
@pytest.mark.asyncio
async def test_bank_synthetic_extracts_transactions():
    from docmirror.plugins.bank_statement.context import collect_tables_from_parse_result
    from scripts.generate_synthetic_golden_pdfs import ensure_bank_synthetic
    from tests.golden.test_golden_matrix_benchmark import _parse_case

    pdf = ensure_bank_synthetic()
    pr = await _parse_case(pdf)
    assert pr.entities.document_type == "bank_statement"
    assert len(pr.extractor_full_text or pr.full_text) > 50

    ctx = StyleContext(
        tables=collect_tables_from_parse_result(pr),
        full_text=pr.full_text or "",
        institution=None,
        page_count=len(pr.pages or []),
        parse_result=pr,
    )
    detection = BankStyleDetector().detect(ctx)
    plugin = BankStatementCommunityPlugin()
    records, _ = BankStyleParserRegistry().run(detection, ctx, plugin)
    assert len(records) >= 3

    api = project_mirror(seal_parse_result(pr))
    doc = api
    assert len(doc.get("pages") or []) >= 1
