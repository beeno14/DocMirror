# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical evidence atom split debit/credit bank ledger recovery tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import CellValue, TableBlock, TableRow
from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.evidence_atom_table_recovery import (
    _column_aggregate_source_raw,
    _expand_composite_header_atoms,
    _geometry_header,
    _geometry_transaction_anchor_y,
    _infer_positioned_block_directions,
    _is_geometry_footer_text,
    _join_geometry_atoms,
    _positioned_block_counter_account,
    _repair_geometry_cell_spill,
    _repair_geometry_rows,
    _sort_positioned_block_records,
    _split_glued_geometry_data_atom,
    _strip_geometry_page_header_overlay,
    recover_evidence_atom_bank_tables,
    recover_positioned_record_block_bank_tables,
    recovered_evidence_atom_expected_row_count,
    recovered_evidence_atom_expected_row_evidence,
    recovered_evidence_atom_row_sources,
    recovered_native_datetime_row_evidence,
)
from docmirror.plugins.bank_statement.style_registry import (
    BankTableCandidate,
    _candidate_expected_rows,
    _candidate_from_batch,
    _candidate_reliable_count_coverage,
    _candidate_row_count_evidence,
    _collect_table_candidates,
    _continuous_source_sequence_evidence,
    _page_complete_sequence_evidence,
    _select_candidate,
)
from docmirror.plugins.bank_statement.wide_table_recovery import RowCountEvidence

pytestmark = pytest.mark.unit


def test_continuous_source_sequence_is_candidate_local_quality_evidence() -> None:
    rows = [{"序号": str(index), "交易日期": "2024-01-01"} for index in range(1, 550)]

    evidence = _continuous_source_sequence_evidence(rows)

    assert evidence == RowCountEvidence(549, "continuous_source_sequence", 0.80)
    assert _continuous_source_sequence_evidence([{"序号": "1"}, {"序号": "3"}]) is None


def test_continuous_source_prefix_is_not_independent_count_coverage() -> None:
    prefix = _candidate(
        "truncated_prefix",
        rows=5,
        expected_rows=RowCountEvidence(5, "continuous_source_sequence", 0.80),
    )

    assert _candidate_reliable_count_coverage(prefix) is None


def test_bounded_page_sequence_precedes_bare_global_continuity() -> None:
    rows = [{"序号": str(value), "_source": {"source_page": 1}} for value in range(1, 8)]
    source_rows = "\n".join(f"| {value} | 250101 | 250101 | business | detail |" for value in range(1, 8))
    page_text = f"|No. |Bk.D. |Val.D. | Type | Notes |\n{source_rows}\nDebit Total 1.00 Credit Total 2.00\nPage 1 of 1"

    evidence = _candidate_row_count_evidence(
        rows,
        None,
        page_count=1,
        page_texts=[(1, page_text)],
    )

    assert evidence == RowCountEvidence(7, "complete_page_local_sequences", 0.80)


def test_continuous_sequence_ignores_proven_aggregate_pseudo_row() -> None:
    aggregate = {"序号": "533\n534", "交易日期": "2024-12-01\n2024-12-02"}
    rows = [aggregate, *[{"序号": str(index), "交易日期": "2024-01-01"} for index in range(1, 550)]]

    assert _continuous_source_sequence_evidence(rows) == RowCountEvidence(
        549,
        "continuous_source_sequence",
        0.80,
    )


def _atom(atom_id: str, text: str, x0: float, y0: float, x1: float | None = None) -> dict:
    return {
        "id": atom_id,
        "page_id": "page:0001",
        "text": text,
        "bbox": [x0, y0, x1 if x1 is not None else x0 + 20.0, y0 + 8.0],
    }


def _result(atoms: list[dict], vector_atoms: list[dict] | None = None) -> SimpleNamespace:
    page_numbers = sorted({int(str(atom.get("page_id") or "page:0001").rsplit(":", 1)[-1]) for atom in atoms})
    return SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence={
                "text_atoms": atoms,
                "vector_atoms": list(vector_atoms or []),
            }
        ),
        pages=[SimpleNamespace(page_number=page, width=600, height=850, tables=[], texts=[]) for page in page_numbers],
        logical_tables=[],
        entities=SimpleNamespace(domain_specific={}),
    )


def _rotated_90(atom: dict, *, page_id: str) -> dict:
    x0, y0, x1, y1 = atom["bbox"]
    return {
        **atom,
        "page_id": page_id,
        "bbox": [600.0 - y1, x0, 600.0 - y0, x1],
    }


def _candidate(
    candidate_id: str,
    *,
    canonical_coverage: float = 1.0,
    source_page_coverage: float = 1.0,
    field_completeness: float = 1.0,
    balance_chain_score: float = 1.0,
    score: float = 0.9,
    rows: int = 3,
    expected_rows: RowCountEvidence | None = None,
    source_column_width: float = 0.0,
    extraction_confidence: float = 0.0,
    sequence_continuity: float = 0.0,
    native_cell_coverage: float = 0.0,
) -> BankTableCandidate:
    return BankTableCandidate(
        candidate_id=candidate_id,
        records=[{} for _ in range(rows)],
        source=candidate_id,
        canonical_rows=rows,
        directional_rows=rows,
        source_page_rows=round(rows * source_page_coverage),
        expected_rows=expected_rows,
        balance_chain_score=balance_chain_score,
        field_completeness=field_completeness,
        score=score,
        canonical_coverage=canonical_coverage,
        source_page_coverage=source_page_coverage,
        source_column_width=source_column_width,
        extraction_confidence=extraction_confidence,
        sequence_continuity=sequence_continuity,
        native_cell_coverage=native_cell_coverage,
    )


def test_candidate_selection_rejects_larger_ocr_result_without_page_provenance():
    selected, diagnostics = _select_candidate(
        [
            _candidate("physical_table", score=0.88, rows=4),
            _candidate("ocr_implicit_table", source_page_coverage=0.0, score=0.80, rows=6),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "physical_table"
    assert diagnostics["selected_candidate"] == "physical_table"


def test_candidate_selection_prefers_continuous_source_sequence_over_noisy_extra_rows():
    selected, diagnostics = _select_candidate(
        [
            _candidate("evidence_atom", rows=199, sequence_continuity=1.0, score=0.88),
            _candidate("ocr_implicit_table", rows=211, sequence_continuity=0.0, score=0.92),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "evidence_atom"
    assert diagnostics["selected_candidate"] == "evidence_atom"


def test_candidate_scoring_rejects_column_aggregated_dates_copied_as_rows() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "amount_cny", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
    )
    collapsed_date = "2025-09-2100:07:462025-10-2716:36:242025-10-2814:44:10"
    collapsed = [{"交易日期": collapsed_date, "金额": "0.04", "方向": "收入", "余额": "306.09"} for _ in range(3)]

    candidate = _candidate_from_batch(
        candidate_id="evidence_atom",
        transactions=collapsed,
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="canonical_evidence_table",
        expected_rows=RowCountEvidence(count=3, source="positioned_date_anchors", confidence=0.80),
        extraction_confidence=0.90,
    )

    assert candidate.canonical_rows == 0
    assert candidate.canonical_coverage == 0.0


def test_native_candidate_drops_page_aggregate_but_keeps_physical_rows() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "amount_cny", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
    )
    rows = [
        {
            "交易日期": "2025-01-012025-01-022025-01-03",
            "金额": "1.00",
            "方向": "收入",
            "余额": "3.00",
        },
        *[
            {
                "交易日期": f"2025-01-0{day}",
                "金额": "1.00",
                "方向": "收入",
                "余额": f"{day}.00",
            }
            for day in range(1, 4)
        ],
    ]

    candidate = _candidate_from_batch(
        candidate_id="native_wide_table",
        transactions=rows,
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="native_wide_table",
        expected_rows=RowCountEvidence(count=3, source="header_total", confidence=0.95),
        extraction_confidence=0.85,
    )

    assert len(candidate.records) == 3
    assert candidate.canonical_rows == 3
    assert candidate.semantic_anomaly_rows == 1
    assert candidate.rejection_reason == ""


def test_candidate_scoring_preserves_distinct_rows_with_same_business_values() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "amount_cny", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
    )
    repeated_business_rows = [
        {"交易日期": "2025-09-21", "金额": "0.04", "方向": "收入", "余额": "306.09"},
        {"交易日期": "2025-09-21", "金额": "0.04", "方向": "收入", "余额": "306.09"},
    ]

    candidate = _candidate_from_batch(
        candidate_id="native_wide_table",
        transactions=repeated_business_rows,
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="native_wide_table",
        expected_rows=RowCountEvidence(count=2, source="native_wide_rows", confidence=0.70),
        extraction_confidence=0.85,
    )

    assert candidate.canonical_rows == 2
    assert candidate.canonical_coverage == 1.0


def test_candidate_scoring_excludes_cross_role_echo_row() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance"],
        _normalize=lambda raw: dict(raw["normalized"]),
    )
    echoed = "2023-06-0161"
    transaction = {
        "交易日期": "2023-06-01",
        "交易金额": "2,356,210.84",
        "normalized": {
            "date": "2023-06-01",
            "amount": 2356210.84,
            "direction": "income",
            "balance": 2023.0,
            "timestamp": echoed,
            "summary": echoed,
            "purpose": echoed,
            "sequence_no": echoed,
            "channel": echoed,
        },
    }

    candidate = _candidate_from_batch(
        candidate_id="positioned_record_block",
        transactions=[transaction],
        normalize_fn=lambda raw: dict(raw["normalized"]),
        plugin=plugin,
        source="positioned_record_block",
        expected_rows=None,
        extraction_confidence=0.95,
    )

    assert candidate.canonical_rows == 0
    assert candidate.semantic_anomaly_rows == 1


def _native_datetime_atom(
    atom_id: str,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    *,
    page_id: str,
) -> dict:
    return {
        "id": atom_id,
        "page_id": page_id,
        "source_kind": "pdf_native",
        "text": text,
        "bbox": [x0, y0, x1, y0 + 10.0],
    }


def _native_datetime_page(
    page: int,
    page_count: int,
    rows: list[tuple[str, str, str, str, str]],
    *,
    account: str = "550933090800015",
) -> list[dict]:
    page_id = f"page:{page:04d}"
    atoms = [
        _native_datetime_atom("title", "台州银行交易明细", 235.0, 43.0, 360.0, page_id=page_id),
        _native_datetime_atom("account", f"账号:{account}", 67.0, 77.0, 170.0, page_id=page_id),
        _native_datetime_atom("page", f"第{page}", 355.0, 77.0, 375.0, page_id=page_id),
        _native_datetime_atom("slash", "/", 389.0, 77.0, 394.0, page_id=page_id),
        _native_datetime_atom("total", str(page_count), 404.0, 77.0, 409.0, page_id=page_id),
        _native_datetime_atom("page_label", "页", 414.0, 77.0, 424.0, page_id=page_id),
    ]
    for index, (label, x0, x1) in enumerate(
        zip(
            ("日期", "支出", "收入", "余额", "对方账户", "对方户名", "摘要/附言"),
            (38.0, 126.0, 199.0, 278.0, 339.0, 435.0, 514.0),
            (58.0, 146.0, 219.0, 298.0, 379.0, 475.0, 559.0),
        )
    ):
        atoms.append(_native_datetime_atom(f"header-{index}", label, x0, 156.0, x1, page_id=page_id))

    debit_total = 0.0
    credit_total = 0.0
    for index, (row_date, row_time, direction, amount, balance) in enumerate(rows):
        y = 174.0 + 35.0 * index
        atoms.extend(
            [
                _native_datetime_atom(f"date-{index}", row_date, 23.0, y, 73.0, page_id=page_id),
                _native_datetime_atom(f"time-{index}", row_time, 28.0, y + 10.0, 68.0, page_id=page_id),
                _native_datetime_atom(
                    f"amount-{index}",
                    amount,
                    112.0 if direction == "expense" else 185.0,
                    y,
                    147.0 if direction == "expense" else 220.0,
                    page_id=page_id,
                ),
                _native_datetime_atom(f"balance-{index}", balance, 263.0, y, 298.0, page_id=page_id),
            ]
        )
        if direction == "expense":
            debit_total += float(amount)
        else:
            credit_total += float(amount)

    atoms.extend(
        [
            _native_datetime_atom("footer", "合计:", 29.0, 773.0, 54.0, page_id=page_id),
            _native_datetime_atom("debit-total", f"{debit_total:.2f}", 104.0, 773.0, 149.0, page_id=page_id),
            _native_datetime_atom("credit-total", f"{credit_total:.2f}", 180.0, 773.0, 225.0, page_id=page_id),
            _native_datetime_atom("operator", "打印操作员：1", 105.0, 788.0, 220.0, page_id=page_id),
            _native_datetime_atom("print-date", "打印日期：2024-03-29", 265.0, 788.0, 365.0, page_id=page_id),
            _native_datetime_atom("print-time", "打印时间：09:31:18", 405.0, 788.0, 495.0, page_id=page_id),
        ]
    )
    return atoms


def _native_datetime_result(atoms: list[dict], page_count: int) -> SimpleNamespace:
    pages = [SimpleNamespace(page_number=page, width=595, height=842) for page in range(1, page_count + 1)]
    plane_pages = [
        SimpleNamespace(page_id=f"page:{page:04d}", page_number=page, content_mode="native_text")
        for page in range(1, page_count + 1)
    ]
    return SimpleNamespace(
        pages=pages,
        evidence_plane=SimpleNamespace(
            pages=plane_pages,
            evidence={"text_atoms": atoms},
        ),
        parser_info=SimpleNamespace(
            extraction_method="digital",
            options={
                "source_page_count": page_count,
                "selected_source_pages": list(range(1, page_count + 1)),
            },
        ),
    )


def _cib_native_page(
    page: int,
    page_count: int,
    rows: list[tuple[str, str, str]],
    *,
    account: str = "622908123094910928",
) -> list[dict]:
    page_id = f"page:{page:04d}"
    last_row_y = 212.0 + 34.0 * max(len(rows) - 1, 0)
    footer_y = last_row_y + 30.0
    atoms = [
        _native_datetime_atom("title", "兴业银行交易流水", 250.0, 52.0, 345.0, page_id=page_id),
        _native_datetime_atom("account", f"号：{account}", 333.0, 81.0, 480.0, page_id=page_id),
        _native_datetime_atom("header-date", "交易日期", 36.0, 184.0, 72.0, page_id=page_id),
        _native_datetime_atom("header-posting", "记账日期", 83.0, 184.0, 120.0, page_id=page_id),
        _native_datetime_atom("header-summary", "摘要", 130.0, 184.0, 149.0, page_id=page_id),
        _native_datetime_atom("header-money", "支/收交易金额", 196.0, 184.0, 255.0, page_id=page_id),
        _native_datetime_atom("header-balance", "账户余额", 265.0, 184.0, 301.0, page_id=page_id),
        _native_datetime_atom("header-location", "交易地点", 311.0, 184.0, 348.0, page_id=page_id),
        _native_datetime_atom("header-party", "对方户名", 388.0, 184.0, 425.0, page_id=page_id),
        _native_datetime_atom(
            "header-counter",
            "对方账户/对方银行",
            466.0,
            184.0,
            542.0,
            page_id=page_id,
        ),
    ]
    for index, (row_date, amount, balance) in enumerate(rows):
        y = 212.0 + 34.0 * index
        atoms.extend(
            [
                _native_datetime_atom(f"date-{index}", row_date, 36.0, y, 82.0, page_id=page_id),
                _native_datetime_atom(f"posting-{index}", row_date, 84.0, y, 130.0, page_id=page_id),
                _native_datetime_atom(f"amount-{index}", amount, 220.0, y, 263.0, page_id=page_id),
                _native_datetime_atom(f"balance-{index}", balance, 268.0, y, 309.0, page_id=page_id),
            ]
        )
    atoms.extend(
        [
            _native_datetime_atom(
                "privacy-footer",
                "说明：交易明细涉及您的个人隐私，请妥善处理。",
                36.0,
                footer_y,
                440.0,
                page_id=page_id,
            ),
            _native_datetime_atom(
                "page-marker",
                f"第{page}页/共{page_count}页",
                273.0,
                footer_y + 50.0,
                323.0,
                page_id=page_id,
            ),
        ]
    )
    return atoms


def test_native_datetime_census_counts_duplicate_timestamps_as_bbox_multiset() -> None:
    rows = [
        ("2024-03-26", "16:09:39", "expense", "1200.00", "395.67"),
        ("2024-03-26", "16:09:39", "income", "2800.00", "3195.67"),
    ]
    result = _native_datetime_result(_native_datetime_page(1, 1, rows), 1)

    assert recovered_native_datetime_row_evidence(result, source_route="digital") == (
        2,
        "native_page_datetime_census",
        0.80,
    )


def test_native_datetime_census_terminal_deletion_remains_low_confidence() -> None:
    rows = [
        ("2024-03-26", "16:09:39", "expense", "1200.00", "395.67"),
        ("2024-03-26", "16:08:39", "expense", "0.00", "395.67"),
        ("2024-03-26", "16:07:39", "income", "0.00", "395.67"),
    ]
    atoms = _native_datetime_page(1, 1, rows)
    terminal_ids = {"date-2", "time-2", "amount-2", "balance-2"}
    shortened_atoms = [atom for atom in atoms if atom["id"] not in terminal_ids]

    # The zero-value terminal row can disappear without changing page totals.
    # The surviving geometry is internally consistent, but cannot certify its
    # own terminal completeness.
    assert recovered_native_datetime_row_evidence(
        _native_datetime_result(shortened_atoms, 1), source_route="digital"
    ) == (2, "native_page_datetime_census", 0.80)


@pytest.mark.parametrize("missing_index", [1, 2])
def test_native_datetime_census_rejects_erased_zero_value_row_boundary(missing_index: int) -> None:
    rows = [
        ("2024-03-26", "16:09:39", "expense", "1200.00", "395.67"),
        ("2024-03-26", "16:08:39", "expense", "0.00", "395.67"),
        ("2024-03-26", "16:07:39", "income", "0.00", "395.67"),
    ]
    atoms = _native_datetime_page(1, 1, rows)
    row_pitch = 35.0
    footer_y = 174.0 + row_pitch * len(rows) + 20.0
    for atom in atoms:
        if atom["id"] in {"footer", "debit-total", "credit-total"}:
            atom["bbox"][1] = footer_y
            atom["bbox"][3] = footer_y + 10.0
        elif atom["id"] in {"operator", "print-date", "print-time"}:
            atom["bbox"][1] = footer_y + 15.0
            atom["bbox"][3] = footer_y + 25.0
    erased_ids = {
        f"date-{missing_index}",
        f"time-{missing_index}",
        f"amount-{missing_index}",
        f"balance-{missing_index}",
    }
    atoms = [atom for atom in atoms if atom["id"] not in erased_ids]

    assert recovered_native_datetime_row_evidence(_native_datetime_result(atoms, 1), source_route="digital") == (
        0,
        "",
        0.0,
    )


@pytest.mark.parametrize("orphan_id", ["date-1", "time-1", "amount-1", "balance-1"])
def test_native_datetime_census_rejects_orphan_row_atom(orphan_id: str) -> None:
    rows = [
        ("2024-03-26", "16:09:39", "expense", "1200.00", "395.67"),
        ("2024-03-26", "16:08:39", "income", "0.00", "395.67"),
    ]
    atoms = [atom for atom in _native_datetime_page(1, 1, rows) if atom["id"] != orphan_id]

    assert recovered_native_datetime_row_evidence(_native_datetime_result(atoms, 1), source_route="digital") == (
        0,
        "",
        0.0,
    )


def test_native_datetime_census_rejects_duplicate_row_atom() -> None:
    atoms = _native_datetime_page(
        1,
        1,
        [
            ("2024-03-26", "16:09:39", "expense", "1200.00", "395.67"),
            ("2024-03-26", "16:08:39", "income", "0.00", "395.67"),
        ],
    )
    duplicate = dict(next(atom for atom in atoms if atom["id"] == "date-1"), id="date-duplicate")

    assert recovered_native_datetime_row_evidence(
        _native_datetime_result([*atoms, duplicate], 1), source_route="digital"
    ) == (0, "", 0.0)


def test_native_datetime_census_does_not_override_candidate_rows_as_authority() -> None:
    evidence = RowCountEvidence(152, "native_page_datetime_census", 0.80)

    assert _candidate_expected_rows(
        evidence,
        count=149,
        source="candidate_rows",
        confidence=0.55,
    ) == RowCountEvidence(149, "candidate_rows", 0.55)
    assert _candidate_expected_rows(evidence) == evidence


def test_native_and_ocr_bounded_census_sources_stay_below_authority_threshold() -> None:
    for source in (
        "native_page_datetime_census",
        "native_page_signed_ledger_census",
        "ocr_page_ordinal_census",
    ):
        evidence = RowCountEvidence(152, source, 0.80)

        assert evidence.confidence < 0.85
        assert _candidate_expected_rows(
            evidence,
            count=149,
            source="candidate_rows",
            confidence=0.55,
        ) == RowCountEvidence(149, "candidate_rows", 0.55)


@pytest.mark.parametrize(
    "mutation",
    ["title", "header", "header_order", "footer", "account", "balance", "page"],
)
def test_native_datetime_census_fails_closed_when_page_proof_is_incomplete(mutation: str) -> None:
    rows = [("2024-03-26", "16:09:39", "expense", "1200.00", "395.67")]
    atoms = _native_datetime_page(1, 1, rows)
    if mutation == "title":
        next(atom for atom in atoms if atom["id"] == "title")["text"] = "鍏朵粬閾惰娴佹按"
    elif mutation == "header":
        atoms = [atom for atom in atoms if atom["id"] != "header-3"]
    elif mutation == "header_order":
        left = next(atom for atom in atoms if atom["id"] == "header-1")
        right = next(atom for atom in atoms if atom["id"] == "header-2")
        left["bbox"], right["bbox"] = right["bbox"], left["bbox"]
    elif mutation == "footer":
        next(atom for atom in atoms if atom["id"] == "debit-total")["text"] = "1199.99"
    elif mutation == "account":
        atoms.extend(_native_datetime_page(2, 2, rows, account="999999999999999"))
        result = _native_datetime_result(atoms, 2)
        assert recovered_native_datetime_row_evidence(result, source_route="digital") == (0, "", 0.0)
        return
    elif mutation == "balance":
        atoms = [atom for atom in atoms if atom["id"] != "balance-0"]
    else:
        result = _native_datetime_result(atoms, 2)
        assert recovered_native_datetime_row_evidence(result, source_route="digital") == (0, "", 0.0)
        return
    result = _native_datetime_result(atoms, 1)

    assert recovered_native_datetime_row_evidence(result, source_route="digital") == (0, "", 0.0)


def test_cib_native_census_counts_complete_signed_ledger_across_pages() -> None:
    atoms = [
        *_cib_native_page(
            1,
            2,
            [("2022-01-01", "100.00", "100.00"), ("2022-01-02", "-30.00", "70.00")],
        ),
        *_cib_native_page(2, 2, [("2022-01-03", "20.00", "90.00")]),
    ]

    assert recovered_native_datetime_row_evidence(_native_datetime_result(atoms, 2), source_route="digital") == (
        3,
        "native_page_signed_ledger_census",
        0.80,
    )


def test_cib_native_census_terminal_deletion_remains_low_confidence() -> None:
    atoms = _cib_native_page(
        1,
        1,
        [
            ("2022-01-01", "100.00", "100.00"),
            ("2022-01-02", "-30.00", "70.00"),
            ("2022-01-03", "20.00", "90.00"),
        ],
    )
    terminal_ids = {"date-2", "posting-2", "amount-2", "balance-2"}
    shortened_atoms = [atom for atom in atoms if atom["id"] not in terminal_ids]
    for atom in shortened_atoms:
        if atom["id"] in {"privacy-footer", "page-marker"}:
            atom["bbox"][1] -= 34.0
            atom["bbox"][3] -= 34.0

    # Moving the footer with a symmetrically shortened row plane leaves a valid
    # prefix and a closed balance chain.  It is useful coverage, not authority.
    assert recovered_native_datetime_row_evidence(
        _native_datetime_result(shortened_atoms, 1), source_route="digital"
    ) == (2, "native_page_signed_ledger_census", 0.80)


@pytest.mark.parametrize(
    "mutation",
    ["title", "header", "header_order", "footer", "page", "posting", "balance_chain", "missing_tail"],
)
def test_cib_native_census_fails_closed_when_source_proof_is_incomplete(mutation: str) -> None:
    atoms = _cib_native_page(
        1,
        1,
        [("2022-01-01", "100.00", "100.00"), ("2022-01-02", "-30.00", "70.00")],
    )
    if mutation == "title":
        next(atom for atom in atoms if atom["id"] == "title")["text"] = "其他银行交易流水"
    elif mutation == "header":
        atoms = [atom for atom in atoms if atom["id"] != "header-posting"]
    elif mutation == "header_order":
        date_header = next(atom for atom in atoms if atom["id"] == "header-date")
        posting_header = next(atom for atom in atoms if atom["id"] == "header-posting")
        date_header["bbox"], posting_header["bbox"] = posting_header["bbox"], date_header["bbox"]
    elif mutation == "footer":
        atoms = [atom for atom in atoms if atom["id"] != "privacy-footer"]
    elif mutation == "page":
        next(atom for atom in atoms if atom["id"] == "page-marker")["text"] = "第1页/共2页"
    elif mutation == "posting":
        atoms = [atom for atom in atoms if atom["id"] != "posting-1"]
    elif mutation == "balance_chain":
        next(atom for atom in atoms if atom["id"] == "balance-1")["text"] = "71.00"
    else:
        atoms = [
            atom for atom in atoms if not atom["id"].endswith("-1") or atom["id"] in {"privacy-footer", "page-marker"}
        ]

    assert recovered_native_datetime_row_evidence(_native_datetime_result(atoms, 1), source_route="digital") == (
        0,
        "",
        0.0,
    )


def test_candidate_scoring_excludes_explicit_header_furniture_row() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
    )
    transaction = {
        "交易日期": "2025-01-02",
        "金额": "25.00",
        "方向": "支出",
        "余额": "5000888.02",
        "摘要": "打印时间: 2026/02/24 16:19 记录数: 71",
        "对方户名": "用户所属公司: 重庆正大华日软件有限公司",
    }

    candidate = _candidate_from_batch(
        candidate_id="positioned_record_block",
        transactions=[transaction],
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="positioned_record_block",
        expected_rows=None,
        extraction_confidence=0.95,
    )

    assert candidate.canonical_rows == 0
    assert candidate.semantic_anomaly_rows == 1


def test_candidate_rejects_truncated_recognized_delimited_business_cell() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance", "summary"],
        _normalize=lambda raw: {
            **_normalized_candidate_row(raw),
            "summary": raw.get("摘要/附言", ""),
        },
    )
    shifted = {
        "序号": "1",
        "交易日期": "2025-01-02",
        "交易金额": "25.00",
        "交易类型": "支出",
        "币别": "CNY",
        "账户余额": "100.00",
        "对方账号": "",
        "对方户名": "",
        "金额": "25.00",
        "方向": "支出",
        "余额": "100.00",
        # A shifted cell can retain a complete-looking marker after a fragment
        # from the neighbouring row.  The marker's position makes the grammar
        # invalid even though the four delimiters are still present.
        "摘要/附言": "前行尾部0WL#12345S#WL协议#商户退款",
    }
    intact = {
        **shifted,
        "摘要/附言": "0WL#12345S#WL协议#商户退款",
    }

    rejected = _candidate_from_batch(
        candidate_id="geometry_shifted",
        transactions=[shifted],
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="canonical_evidence_table",
        expected_rows=None,
        extraction_confidence=0.9,
    )
    accepted = _candidate_from_batch(
        candidate_id="source_grid",
        transactions=[intact],
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="canonical_physical_tables",
        expected_rows=None,
        extraction_confidence=0.85,
    )

    selected, _diagnostics = _select_candidate([rejected, accepted])

    assert rejected.canonical_rows == 0
    assert rejected.rejection_reason == "source_role_corruption"
    assert selected is accepted


def test_candidate_does_not_apply_delimited_grammar_to_unknown_business_code() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance", "summary"],
        _normalize=lambda raw: {
            **_normalized_candidate_row(raw),
            "summary": raw.get("摘要/附言", ""),
        },
    )
    row = {
        "序号": "1",
        "交易日期": "2025-01-02",
        "交易金额": "25.00",
        "交易类型": "支出",
        "币别": "CNY",
        "账户余额": "100.00",
        "对方账号": "",
        "对方户名": "",
        "金额": "25.00",
        "方向": "支出",
        "余额": "100.00",
        "摘要/附言": "CUSTOM#free#form",
    }

    candidate = _candidate_from_batch(
        candidate_id="unknown_delimiter",
        transactions=[row],
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="canonical_physical_tables",
        expected_rows=None,
        extraction_confidence=0.85,
    )

    assert candidate.canonical_rows == 1
    assert candidate.semantic_anomaly_rows == 0


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("摘要/附言", "业务提示包含0WL但没有结构分隔符"),
        ("摘要/附言", "A0WL#12345S#WL协议#普通自由文本"),
        ("普通摘要", "前行尾部0WL#12345S#WL协议#普通自由文本"),
    ],
)
def test_delimited_business_guard_is_bounded_by_marker_and_source_role(
    header: str,
    value: str,
) -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance", "summary"],
        _normalize=lambda raw: {
            **_normalized_candidate_row(raw),
            "summary": raw.get(header, ""),
        },
    )
    row = {
        "序号": "1",
        "交易日期": "2025-01-02",
        "交易金额": "25.00",
        "交易类型": "支出",
        "币别": "CNY",
        "账户余额": "100.00",
        "对方账号": "",
        "对方户名": "",
        "金额": "25.00",
        "方向": "支出",
        "余额": "100.00",
        "摘要/附言": "" if header != "摘要/附言" else value,
        header: value,
    }

    candidate = _candidate_from_batch(
        candidate_id="bounded_delimiter_guard",
        transactions=[row],
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="canonical_physical_tables",
        expected_rows=None,
        extraction_confidence=0.85,
    )

    assert candidate.canonical_rows == 1
    assert candidate.semantic_anomaly_rows == 0


def test_delimited_business_guard_requires_complete_source_layout() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance", "summary"],
        _normalize=lambda raw: {
            **_normalized_candidate_row(raw),
            "summary": raw["摘要/附言"],
        },
    )
    row = {
        "序号": "1",
        "交易日期": "2025-01-02",
        "交易金额": "25.00",
        "交易类型": "支出",
        # Deliberately no 币别: the full source layout is not proven.
        "账户余额": "100.00",
        "对方账号": "",
        "对方户名": "",
        "金额": "25.00",
        "方向": "支出",
        "余额": "100.00",
        "摘要/附言": "前缀0WL#12345S#WL协议#文本",
    }

    candidate = _candidate_from_batch(
        candidate_id="incomplete_layout_delimiter_guard",
        transactions=[row],
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="canonical_physical_tables",
        expected_rows=None,
        extraction_confidence=0.85,
    )

    assert candidate.canonical_rows == 1
    assert candidate.semantic_anomaly_rows == 0


def test_candidate_selection_rejects_systemic_summary_date_role_swap() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance", "summary"],
        _normalize=lambda raw: {
            **_normalized_candidate_row(raw),
            "summary": raw.get("摘要", ""),
        },
    )
    bad_rows = [
        {
            "交易日期": f"2025-01-{day:02d}",
            "摘要": f"2025/01/{day:02d}",
            "金额": "1.00",
            "方向": "收入",
            "余额": f"{day}.00",
        }
        for day in range(1, 11)
    ]
    good_rows = [
        {
            "交易日期": f"2025-02-{day:02d}",
            "摘要": "转账",
            "金额": "1.00",
            "方向": "收入",
            "余额": f"{day}.00",
        }
        for day in range(1, 4)
    ]
    bad = _candidate_from_batch(
        candidate_id="positioned_record_block",
        transactions=bad_rows,
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="positioned_record_block",
        expected_rows=None,
        extraction_confidence=0.95,
    )
    good = _candidate_from_batch(
        candidate_id="native_wide_table",
        transactions=good_rows,
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="native_wide_table",
        expected_rows=None,
        extraction_confidence=0.85,
    )

    selected, _ = _select_candidate([bad, good])

    assert bad.canonical_rows == 0
    assert bad.source_role_swap_ratio == 1.0
    assert selected is good


def test_candidate_scoring_keeps_sparse_rows_without_optional_fields_or_provenance() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction"],
        _normalize=lambda raw: {
            "date": raw["交易日期"],
            "amount": abs(float(raw["交易金额"])),
            "direction": "expense" if raw["交易金额"].startswith("-") else "income",
        },
    )
    rows = [{"交易日期": f"2025-03-{day:02d}", "交易金额": "-1.00"} for day in range(1, 4)]
    candidate = _candidate_from_batch(
        candidate_id="semantic_text:0",
        transactions=rows,
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="semantic_text_table",
        expected_rows=None,
        extraction_confidence=0.65,
    )

    selected, _ = _select_candidate([candidate])

    assert candidate.canonical_rows == 3
    assert selected is candidate


def test_candidate_field_completeness_counts_explicit_zero_amount_and_balance() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance"],
        _normalize=lambda _raw: {
            "date": "2025-03-01",
            "amount": 0.0,
            "direction": "income",
            "balance": 0.0,
        },
    )
    candidate = _candidate_from_batch(
        candidate_id="native_wide_table",
        transactions=[{"交易日期": "2025-03-01", "交易金额": "0.00", "余额": "0.00"}],
        normalize_fn=plugin._normalize,
        plugin=plugin,
        source="native_wide_table",
        expected_rows=None,
        extraction_confidence=0.85,
    )

    assert candidate.field_completeness == 1.0


def test_candidate_with_shifted_header_furniture_rejects_whole_alignment() -> None:
    plugin = SimpleNamespace(
        standard_fields=["date", "amount", "direction", "balance"],
        _normalize=lambda raw: _normalized_candidate_row(raw),
        _canonical_raw_values=lambda raw, _normalized: dict(raw),
        _extract_identity=lambda _result: {},
        identity_fields=[],
    )
    valid_rows = [
        {"交易日期": "2025-01-02", "金额": "1.00", "方向": "收入", "余额": "1.00"},
        {"交易日期": "2025-01-03", "金额": "1.00", "方向": "收入", "余额": "2.00"},
    ]
    furniture = {
        "交易日期": "2025-01-01",
        "金额": "1.00",
        "方向": "收入",
        "余额": "0.00",
        "摘要": "打印时间: 2026/02/24 16:19 记录数: 71",
        "对方户名": "用户所属公司: 测试公司",
    }
    candidate = _candidate_from_batch(
        candidate_id="positioned_record_block",
        transactions=[furniture, *valid_rows],
        normalize_fn=_normalized_candidate_row,
        plugin=plugin,
        source="positioned_record_block",
        expected_rows=None,
        extraction_confidence=0.95,
    )

    assert candidate.canonical_rows == 0
    assert candidate.rejected_row_indexes == (0,)
    assert candidate.rejection_reason == "source_role_corruption"


def _normalized_candidate_row(raw: dict) -> dict:
    direction = "income" if raw["方向"] == "收入" else "expense"
    return {
        "date": raw["交易日期"][:10],
        "amount": float(raw["金额"]),
        "amount_cny": float(raw["金额"]),
        "direction": direction,
        "balance": float(raw["余额"]),
    }


def test_geometry_direction_repair_uses_summary_and_cross_page_balance() -> None:
    columns = {"direction": 0, "amount": 1, "balance": 2, "summary": 3}
    first_page = [["转账", "198.87", "144.74", "出账网联"]]
    second_page = [["转账", "3,000.00", "36,914.33", "微信转账"]]

    previous_balance = _repair_geometry_rows(first_page, columns)
    _repair_geometry_rows(second_page, columns, previous_balance=39_914.33)

    assert previous_balance == 144.74
    assert first_page[0][0] == "支出"
    assert second_page[0][0] == "支出"


def test_geometry_cell_spill_moves_bounded_summary_prefix_out_of_signed_amount() -> None:
    row = ["2023/04/08", "AppStore_Music", "Apple-19.00", "549.28"]

    _repair_geometry_cell_spill(
        row,
        {"date": 0, "summary": 1, "amount": 2, "balance": 3},
    )

    assert row == ["2023/04/08", "AppStore_MusicApple", "-19.00", "549.28"]


def test_geometry_cell_spill_does_not_treat_currency_or_unsigned_text_as_summary() -> None:
    currency = ["2023/04/08", "purchase", "USD-19.00", "549.28"]
    unsigned = ["2023/04/08", "purchase", "Apple19.00", "549.28"]
    columns = {"date": 0, "summary": 1, "amount": 2, "balance": 3}

    _repair_geometry_cell_spill(currency, columns)
    _repair_geometry_cell_spill(unsigned, columns)

    assert currency[1:3] == ["purchase", "USD-19.00"]
    assert unsigned[1:3] == ["purchase", "Apple19.00"]


def test_geometry_direction_repair_corrects_explicit_direction_when_balance_uniquely_disagrees() -> None:
    columns = {"direction": 0, "amount": 1, "balance": 2, "summary": 3}
    rows = [["收入", "2.00", "3,641.74", "短信收费"]]

    _repair_geometry_rows(rows, columns, previous_balance=3_643.74)

    assert rows[0][0] == "支出"


def test_geometry_atom_join_preserves_visual_account_order_across_font_baselines() -> None:
    atoms = [
        _atom("prefix", "00000000000000", 340.0, 160.2, 380.0),
        _atom("suffix", "864", 380.0, 160.0, 400.0),
    ]

    assert _join_geometry_atoms(atoms, line_tolerance=1.5) == "00000000000000864"


def test_geometry_footer_recognizes_issuer_important_notice() -> None:
    assert _is_geometry_footer_text("重要提示：请仔细核对账户余额，客服电话：95588") is True
    assert _is_geometry_footer_text("说明：交易明细涉及您的个人隐私，请妥善处理") is True
    assert _is_geometry_footer_text("收入交易笔数：391 收入金额合计：23,790,585.15") is True
    assert _is_geometry_footer_text("支出交易笔数：1861 支出金额合计：23,791,584.72") is True
    assert _is_geometry_footer_text("支出交易总额：14,146,954.73") is True
    assert _is_geometry_footer_text("收入交易总额：14,146,649.53") is True
    assert _is_geometry_footer_text("合计笔数：1821") is True
    assert _is_geometry_footer_text("https://secure.example/statement-verification") is True


def test_geometry_recovery_splits_source_fragmented_dates_and_glued_business_cells() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("hp", "记账日期", 70.0, 80.0, 110.0),
        _atom("hs", "摘要", 120.0, 80.0, 150.0),
        _atom("hdir1", "⽀/", 175.0, 73.0, 190.0),
        _atom("hdir2", "收", 175.0, 87.0, 190.0),
        _atom("hm", "交易⾦额账户余额交易地点", 200.0, 80.0, 360.0),
        _atom("hparty", "对方户名", 400.0, 80.0, 450.0),
        _atom("haccount", "对方账户/对方银行", 460.0, 80.0, 570.0),
        _atom("d1a", "2024-01-", 20.0, 110.0, 60.0),
        _atom("d1b", "02", 20.0, 122.0, 32.0),
        _atom("p1a", "2024-01-", 70.0, 110.0, 110.0),
        _atom("p1b", "02", 70.0, 122.0, 82.0),
        _atom("s1", "跨行代付", 120.0, 116.0, 165.0),
        _atom("dir1", "收", 175.0, 116.0, 185.0),
        _atom("money1", "10.0020.00测试支行", 205.0, 116.0, 365.0),
        _atom("party1", "测试商户", 400.0, 116.0, 445.0),
        _atom("account1", "123456789", 460.0, 116.0, 520.0),
        _atom("row2lead", "2024-01-032024-01-03转账支出", 20.0, 150.0, 165.0),
        _atom("dir2", "支", 175.0, 150.0, 185.0),
        _atom("money2", "-5.0015.00另一支行", 205.0, 150.0, 365.0),
        _atom("party2", "另一商户", 400.0, 150.0, 445.0),
        _atom("account2", "987654321", 460.0, 150.0, 520.0),
        _atom("footer", "打印完毕", 20.0, 180.0, 100.0),
    ]
    # Some PDF-native extractors give a horizontally glued row a tall bbox.
    # It remains one semantic row, not two vertically stacked date values.
    next(atom for atom in atoms if atom["id"] == "row2lead")["bbox"][3] = 174.0

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][0][:7] == ["交易日期", "记账日期", "摘要", "支/收", "交易金额", "账户余额", "交易地点"]
    assert tables[0][1][:7] == ["2024-01-02", "2024-01-02", "跨行代付", "收", "10.00", "20.00", "测试支行"]
    assert tables[0][2][:7] == [
        "2024-01-03",
        "2024-01-03",
        "转账支出",
        "支出",
        "-5.00",
        "15.00",
        "另一支行",
    ]


def test_coalesced_fragmented_date_anchor_uses_union_midpoint_only() -> None:
    coalesced = _atom("coalesced", "2024-01-02", 20.0, 100.0, 60.0)
    coalesced["bbox"][3] = 128.0
    coalesced["_coalesced_fragmented_date_anchor"] = True
    native_tall = {key: value for key, value in coalesced.items() if key != "_coalesced_fragmented_date_anchor"}

    assert _geometry_transaction_anchor_y(coalesced, 8.0) == 114.0
    assert _geometry_transaction_anchor_y(native_tall, 8.0) == 124.0


def test_geometry_exact_boundary_uses_target_account_evidence_without_stealing_prior_tail() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("ha", "交易金额", 120.0, 80.0, 170.0),
        _atom("hb", "账户余额", 200.0, 80.0, 250.0),
        _atom("hs", "对方账户/对方银行", 300.0, 80.0, 420.0),
        _atom("d1", "2024-01-02", 20.0, 116.0, 70.0),
        _atom("a1", "-1.00", 120.0, 116.0, 165.0),
        _atom("b1", "9.00", 200.0, 116.0, 245.0),
        # Native single-line date centers are 120 and 156, so the shared
        # boundary is exactly 138.  A CJK tail on that boundary completes the
        # prior bank name, while an account-shaped prefix completes the next
        # row's otherwise-too-short account fragment.
        _atom("prior-account", "111111中国测试有限公", 300.0, 116.0, 410.0),
        _atom("prior-tail", "司", 300.0, 134.0, 310.0),
        _atom("next-account-prefix", "AW8BAGYFokX0qiBJu7uua7e", 315.0, 134.0, 410.0),
        _atom("d2", "2024-01-03", 20.0, 152.0, 70.0),
        _atom("a2", "+2.00", 120.0, 152.0, 165.0),
        _atom("b2", "11.00", 200.0, 152.0, 245.0),
        _atom("next-account-body", "GZfG1财付通支付科技有限公司", 300.0, 152.0, 410.0),
        _atom("footer", "打印完毕", 20.0, 190.0, 100.0),
    ]
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert sources[0]["source_raw"]["对方账户/对方银行"] == "111111中国测试有限公司"
    assert sources[1]["source_raw"]["对方账户/对方银行"] == "AW8BAGYFokX0qiBJu7uua7eGZfG1财付通支付科技有限公司"
    assert "prior-tail" in sources[0]["evidence_ids"]
    assert "prior-tail" not in sources[1]["evidence_ids"]
    assert "next-account-prefix" not in sources[0]["evidence_ids"]
    assert "next-account-prefix" in sources[1]["evidence_ids"]


def test_noncoalesced_geometry_row_keeps_unproven_account_tail_at_exact_top() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("ha", "交易金额", 120.0, 80.0, 170.0),
        _atom("hb", "账户余额", 200.0, 80.0, 250.0),
        _atom("hs", "对方账户/对方银行", 300.0, 80.0, 420.0),
        _atom("d1", "2024-01-02", 20.0, 116.0, 70.0),
        _atom("a1", "-1.00", 120.0, 116.0, 165.0),
        _atom("b1", "9.00", 200.0, 116.0, 245.0),
        _atom("s1", "111111中国测试商户", 300.0, 116.0, 410.0),
        # Native date centers are 120 and 156, making the second row's exact
        # half-open top 138.  This CJK token does not complete an allowed prior
        # institution suffix, so it retains the existing next-row ownership.
        _atom("row2-exact-top", "备注", 300.0, 134.0, 325.0),
        _atom("d2", "2024-01-03", 20.0, 152.0, 70.0),
        _atom("a2", "+2.00", 120.0, 152.0, 165.0),
        _atom("b2", "11.00", 200.0, 152.0, 245.0),
        _atom("s2", "222222中国测试银行", 300.0, 152.0, 410.0),
        _atom("footer", "打印完毕", 20.0, 190.0, 100.0),
    ]
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert "row2-exact-top" not in sources[0]["evidence_ids"]
    assert "row2-exact-top" in sources[1]["evidence_ids"]


def test_fragmented_date_midpoint_keeps_leading_wrapped_roles_with_next_row() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("hp", "记账日期", 70.0, 80.0, 110.0),
        _atom("hs", "摘要", 120.0, 80.0, 150.0),
        _atom("hdir", "支/收", 175.0, 80.0, 195.0),
        _atom("ha", "交易金额", 210.0, 80.0, 255.0),
        _atom("hb", "账户余额", 270.0, 80.0, 315.0),
        _atom("hl", "交易地点", 325.0, 80.0, 375.0),
        _atom("hparty", "对方户名", 400.0, 80.0, 450.0),
        _atom("haccount", "对方账户/对方银行", 460.0, 80.0, 570.0),
        _atom("d1a", "2023-04-", 20.0, 110.0, 60.0),
        _atom("d1b", "02", 20.0, 122.0, 32.0),
        _atom("p1a", "2023-04-", 70.0, 110.0, 110.0),
        _atom("p1b", "02", 70.0, 122.0, 82.0),
        _atom("s1", "汇款汇入", 120.0, 116.0, 165.0),
        _atom("dir1", "收", 180.0, 116.0, 190.0),
        _atom("a1", "30,000.00", 210.0, 116.0, 260.0),
        _atom("b1", "30,059.12", 270.0, 116.0, 320.0),
        _atom("l1", "兴业银行漳州高新区支行", 325.0, 116.0, 390.0),
        _atom("party1", "曾燕雁", 400.0, 116.0, 430.0),
        _atom("account1", "6228480158325987077", 460.0, 110.0, 570.0),
        _atom("bank1", "中国农业银行", 460.0, 124.0, 525.0),
        # These are the first wrapped line of row two.  Their baseline is
        # above row two's fragmented date, but they must not leak into row one.
        _atom("party2-prefix", "支付宝(中国)", 400.0, 136.0, 460.0),
        _atom("account2", "215500690", 460.0, 136.0, 515.0),
        _atom("d2a", "2023-04-", 20.0, 146.0, 60.0),
        _atom("d2b", "02", 20.0, 158.0, 32.0),
        _atom("p2a", "2023-04-", 70.0, 146.0, 110.0),
        _atom("p2b", "02", 70.0, 158.0, 82.0),
        _atom("s2", "快捷支付", 120.0, 152.0, 165.0),
        _atom("dir2", "支", 180.0, 152.0, 190.0),
        _atom("a2", "-25.00", 210.0, 152.0, 250.0),
        _atom("b2", "30,034.12", 270.0, 152.0, 320.0),
        _atom("l2", "兴业银行漳州高新区支行", 325.0, 152.0, 390.0),
        _atom("party2-body", "网络技术有限公", 400.0, 152.0, 465.0),
        _atom("bank2-body", "支付宝(中国)网络技", 460.0, 152.0, 565.0),
        _atom("party2-tail", "司", 400.0, 164.0, 410.0),
        _atom("bank2-tail", "术有限公司", 460.0, 164.0, 510.0),
        _atom("footer", "打印完毕", 20.0, 190.0, 100.0),
    ]
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert sources[0]["source_raw"]["对方户名"] == "曾燕雁"
    assert sources[0]["source_raw"]["对方账户/对方银行"] == "6228480158325987077中国农业银行"
    assert sources[1]["source_raw"]["对方户名"] == "支付宝(中国)网络技术有限公司"
    assert sources[1]["source_raw"]["对方账户/对方银行"] == "215500690支付宝(中国)网络技术有限公司"
    assert {"party2-prefix", "account2"}.isdisjoint(sources[0]["evidence_ids"])
    assert {"party2-prefix", "account2"}.issubset(sources[1]["evidence_ids"])
    assert sources[0]["bbox"][3] < sources[1]["bbox"][1]

    plugin = BankStatementCommunityPlugin()
    first_raw = sources[0]["source_raw"]
    second_raw = sources[1]["source_raw"]
    first_canonical = plugin._canonical_raw_values(first_raw, plugin._normalize(first_raw))
    second_canonical = plugin._canonical_raw_values(second_raw, plugin._normalize(second_raw))
    assert {
        key: first_canonical[key] for key in ("counter_party", "counter_account", "counter_bank_name")
    } == {
        "counter_party": "曾燕雁",
        "counter_account": "6228480158325987077",
        "counter_bank_name": "中国农业银行",
    }
    assert {
        key: second_canonical[key] for key in ("counter_party", "counter_account", "counter_bank_name")
    } == {
        "counter_party": "支付宝(中国)网络技术有限公司",
        "counter_account": "215500690",
        "counter_bank_name": "支付宝(中国)网络技术有限公司",
    }


def test_geometry_recovery_only_splits_money_atoms_owned_by_balance_band() -> None:
    atoms = [
        _atom("hs", "序号", 15.0, 80.0, 45.0),
        _atom("hd", "交易日期", 75.0, 80.0, 125.0),
        _atom("ha", "交易金额", 175.0, 80.0, 225.0),
        _atom("hb", "账户余额", 255.0, 80.0, 305.0),
        _atom("hl", "交易地点", 355.0, 80.0, 405.0),
        _atom("hm", "摘要", 485.0, 80.0, 515.0),
        _atom("s1", "1", 25.0, 110.0, 35.0),
        _atom("d1", "2025-01-02", 75.0, 110.0, 125.0),
        _atom("a1", "-10.00", 180.0, 110.0, 220.0),
        _atom("b1", "90.00", 260.0, 110.0, 300.0),
        _atom("l1", "手机银行", 355.0, 110.0, 405.0),
        _atom("m1", "6.17两单6030", 480.0, 110.0, 550.0),
        _atom("s2", "2", 25.0, 150.0, 35.0),
        _atom("d2", "2025-01-03", 75.0, 150.0, 125.0),
        _atom("a2", "-10.00", 180.0, 150.0, 220.0),
        _atom("b2", "80.00", 260.0, 150.0, 300.0),
        _atom("l2", "手机银行", 355.0, 150.0, 405.0),
        _atom("m2a", "6.18订单号", 480.0, 136.0, 535.0),
        _atom("m2b", "274114825757金额", 480.0, 150.0, 555.0),
        _atom("m2c", "6030元", 480.0, 162.0, 515.0),
        _atom("s3", "3", 25.0, 190.0, 35.0),
        _atom("d3", "2025-01-04", 75.0, 190.0, 125.0),
        _atom("a3", "-10.00", 180.0, 190.0, 220.0),
        # The right-aligned balance atom spills eight points left of its
        # Voronoi band but still spans the immediately adjacent location.
        _atom("balance-location", "70.00测试支行", 232.0, 190.0, 410.0),
        _atom("s4", "4", 25.0, 230.0, 35.0),
        _atom("d4", "2025-01-05", 75.0, 230.0, 125.0),
        _atom("amount-balance-location", "-5.0065.00另一支行", 180.0, 230.0, 410.0),
        _atom("s5", "5", 25.0, 270.0, 35.0),
        _atom("d5", "2025-01-06", 75.0, 270.0, 125.0),
        _atom("a5", "-5.00", 180.0, 270.0, 220.0),
        _atom("b5", "60.00", 260.0, 270.0, 300.0),
        _atom("l5", "手机银行", 355.0, 270.0, 405.0),
        _atom("m5", "6.12", 480.0, 270.0, 505.0),
        _atom("footer", "打印完毕", 20.0, 310.0, 100.0),
    ]
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert tables == [
        [
            ["序号", "交易日期", "交易金额", "账户余额", "交易地点", "摘要"],
            ["1", "2025-01-02", "-10.00", "90.00", "手机银行", "6.17两单6030"],
            ["2", "2025-01-03", "-10.00", "80.00", "手机银行", "6.18订单号274114825757金额6030元"],
            ["3", "2025-01-04", "-10.00", "70.00", "测试支行", ""],
            ["4", "2025-01-05", "-5.00", "65.00", "另一支行", ""],
            ["5", "2025-01-06", "-5.00", "60.00", "手机银行", "6.12"],
        ]
    ]
    assert sources[0]["source_raw"] == {
        "序号": "1",
        "交易日期": "2025-01-02",
        "交易金额": "-10.00",
        "账户余额": "90.00",
        "交易地点": "手机银行",
        "摘要": "6.17两单6030",
    }
    assert sources[1]["source_raw"]["摘要"] == "6.18订单号274114825757金额6030元"
    assert {"m1", "m2a", "m2b", "m2c"}.issubset(
        {evidence_id for source in sources[:2] for evidence_id in source["evidence_ids"]}
    )
    assert "balance-location" in sources[2]["evidence_ids"]
    assert "amount-balance-location" in sources[3]["evidence_ids"]
    assert sources[2]["bbox"] == [25.0, 190.0, 410.0, 198.0]
    assert sources[3]["bbox"] == [25.0, 230.0, 410.0, 238.0]
    assert all("reconstruction_repairs" not in source for source in sources)


def test_geometry_balance_location_split_requires_adjacent_forward_roles() -> None:
    nonadjacent = _atom("nonadjacent", "90.00测试支行", 190.0, 110.0, 420.0)
    nonadjacent_centers = [100.0, 200.0, 300.0, 400.0]
    nonadjacent_bounds = [float("-inf"), 150.0, 250.0, 350.0, float("inf")]

    reversed_roles = _atom("reversed", "90.00测试支行", 280.0, 110.0, 420.0)
    reversed_centers = [100.0, 200.0, 300.0]
    reversed_bounds = [float("-inf"), 150.0, 250.0, float("inf")]

    assert (
        _split_glued_geometry_data_atom(
            nonadjacent,
            nonadjacent_bounds,
            nonadjacent_centers,
            {"amount": 0, "balance": 1, "counter_party": 2, "transaction_location": 3},
        )
        == []
    )
    assert (
        _split_glued_geometry_data_atom(
            reversed_roles,
            reversed_bounds,
            reversed_centers,
            {"transaction_location": 0, "amount": 1, "balance": 2},
        )
        == []
    )


def test_geometry_two_money_split_requires_adjacent_forward_roles() -> None:
    cases = [
        (
            _atom("nonadjacent-money", "10.0020.00", 90.0, 110.0, 320.0),
            [100.0, 200.0, 300.0],
            [float("-inf"), 150.0, 250.0, float("inf")],
            {"amount": 0, "counter_party": 1, "balance": 2},
        ),
        (
            _atom("reversed-money", "10.0020.00", 190.0, 110.0, 320.0),
            [100.0, 200.0, 300.0],
            [float("-inf"), 150.0, 250.0, float("inf")],
            {"balance": 0, "amount": 1, "transaction_location": 2},
        ),
        (
            _atom("nonadjacent-residue", "10.0020.00测试支行", 90.0, 110.0, 420.0),
            [100.0, 200.0, 300.0, 400.0],
            [float("-inf"), 150.0, 250.0, 350.0, float("inf")],
            {"amount": 0, "balance": 1, "counter_party": 2, "transaction_location": 3},
        ),
        (
            _atom("reversed-residue", "10.0020.00测试支行", 190.0, 110.0, 420.0),
            [100.0, 200.0, 300.0],
            [float("-inf"), 150.0, 250.0, float("inf")],
            {"transaction_location": 0, "amount": 1, "balance": 2},
        ),
    ]

    for atom, centers, bounds, col_map in cases:
        assert _split_glued_geometry_data_atom(atom, bounds, centers, col_map) == []


def test_geometry_recovery_splits_fused_counterparty_headers_by_source_position() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 70.0),
        _atom("hp", "记账日期", 80.0, 80.0, 130.0),
        _atom("hs", "摘要", 140.0, 80.0, 170.0),
        _atom("hdir", "支/收", 180.0, 80.0, 210.0),
        _atom("ha", "交易金额", 220.0, 80.0, 270.0),
        _atom("hb", "账户余额", 280.0, 80.0, 330.0),
        _atom("hl", "交易地点", 340.0, 80.0, 390.0),
        _atom("hcounter", "对方户名对方账户/对方银行", 410.0, 80.0, 570.0),
        _atom("d", "2025-04-01", 20.0, 110.0, 70.0),
        _atom("p", "2025-04-01", 80.0, 110.0, 130.0),
        _atom("s", "汇款汇入", 140.0, 110.0, 175.0),
        _atom("dir", "收", 190.0, 110.0, 200.0),
        _atom("a", "300.00", 225.0, 110.0, 265.0),
        _atom("b", "399.00", 285.0, 110.0, 325.0),
        _atom("l", "测试支行", 340.0, 110.0, 390.0),
        _atom("party", "测试商户", 410.0, 110.0, 445.0),
        _atom("account", "AW8BAGYF12345", 455.0, 105.0, 535.0),
        _atom("bank", "测试银行", 455.0, 118.0, 515.0),
        _atom("footer", "打印完毕", 20.0, 150.0, 100.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][0][-2:] == ["对方户名", "对方账户/对方银行"]
    assert tables[0][1][-2:] == ["测试商户", "AW8BAGYF12345测试银行"]


def test_geometry_header_rejects_tall_glued_money_row_atom() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("hp", "记账日期", 70.0, 80.0, 110.0),
        _atom("hs", "摘要", 120.0, 80.0, 150.0),
        _atom("hm", "支/收交易金额账户余额交易地点", 170.0, 80.0, 360.0),
        _atom("hparty", "对方户名", 400.0, 80.0, 450.0),
        _atom("haccount", "对方账户/对方银行", 460.0, 80.0, 570.0),
        _atom("first-row-money", "-1,000.0046,167.53测试支行", 205.0, 78.0, 365.0),
    ]
    next(atom for atom in atoms if atom["id"] == "first-row-money")["bbox"][3] = 118.0

    header = _geometry_header(_expand_composite_header_atoms(atoms))

    assert header is not None
    assert all(not str(atom.get("text") or "").startswith("-1,000.00") for atom in header[0])


def test_geometry_recovery_splits_source_atom_across_summary_direction_boundary() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("hp", "记账日期", 70.0, 80.0, 110.0),
        _atom("hs", "摘要", 120.0, 80.0, 150.0),
        _atom("hm", "支/收交易金额账户余额交易地点", 170.0, 80.0, 360.0),
        _atom("d1", "2024-02-01", 20.0, 110.0, 60.0),
        _atom("p1", "2024-02-01", 70.0, 110.0, 110.0),
        _atom("summary-direction", "快捷支付支", 120.0, 110.0, 185.0),
        _atom("amount", "-3.00", 205.0, 110.0, 240.0),
        _atom("balance", "15.00", 265.0, 110.0, 300.0),
        _atom("location", "测试支行", 320.0, 110.0, 365.0),
        _atom("footer", "打印完毕", 20.0, 140.0, 100.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    header, row = tables[0]
    assert row[header.index("摘要")] == "快捷支付"
    assert row[header.index("支/收")] == "支"


def test_geometry_overlay_splitter_keeps_only_row_local_business_values() -> None:
    row = [
        "2024-06-19",
        "户名：测试客户币种：⼈⺠币打印⽇期：2025-02-20记账⽇期2024-06-19",
        "某银行交易流水2024-01-01-2024-12-31账摘要快捷支付",
        "16:40:13⽀/收⽀",
        "交易⾦额账户余额-38.00",
        "某银行交易流水2024-01-01-2024-12-31账账户余额4,210.68",
        "账户类型：活期交易地点测试支行",
        "账号：123456789对⽅户名测试商户",
        "2025年02月20日对⽅账户/对⽅银⾏99887766测试银行",
    ]
    col_map = {
        "date": 0,
        "posting_date": 1,
        "summary": 2,
        "direction": 3,
        "amount": 4,
        "balance": 5,
        "transaction_location": 6,
        "counter_party": 7,
        "counter_account": 8,
    }

    repairs = _strip_geometry_page_header_overlay(row, col_map)

    assert row == [
        "2024-06-19",
        "2024-06-19",
        "快捷支付",
        "⽀",
        "-38.00",
        "4,210.68",
        "测试支行",
        "测试商户",
        "99887766测试银行",
    ]
    assert {repair["field"] for repair in repairs} == {
        "posting_date",
        "summary",
        "direction",
        "amount",
        "balance",
        "transaction_location",
        "counter_party",
        "counter_account",
    }


def test_geometry_overlay_splitter_does_not_rewrite_clean_dedicated_values() -> None:
    row = ["2024-06-19", "借Dr", "退款", "-8.00", "92.00"]
    original = list(row)

    repairs = _strip_geometry_page_header_overlay(
        row,
        {"date": 0, "direction": 1, "summary": 2, "amount": 3, "balance": 4},
    )

    assert row == original
    assert repairs == []


def test_geometry_header_and_rows_ignore_tall_overlapping_transaction_bboxes() -> None:
    atoms = [
        _atom("hd", "交易日期", 20.0, 80.0, 60.0),
        _atom("hp", "记账日期", 70.0, 80.0, 110.0),
        _atom("hs", "摘要", 120.0, 80.0, 150.0),
        _atom("hm", "支/收交易金额账户余额交易地点", 170.0, 80.0, 360.0),
        _atom("hparty", "对方户名", 400.0, 80.0, 450.0),
        _atom("haccount", "对方账户/对方银行", 460.0, 80.0, 570.0),
        _atom("d1", "2024-02-01", 20.0, 78.0, 60.0),
        _atom("p1", "2024-02-01", 70.0, 78.0, 110.0),
        _atom("s1", "入账", 120.0, 108.0, 150.0),
        _atom("dir1", "收", 180.0, 108.0, 190.0),
        _atom("a1", "8.00", 220.0, 108.0, 250.0),
        _atom("b1", "18.00", 270.0, 108.0, 305.0),
        _atom("l1", "测试支行", 320.0, 108.0, 370.0),
        _atom("party1", "甲公司", 400.0, 108.0, 445.0),
        _atom("account1", "10001", 460.0, 108.0, 510.0),
        _atom("d2", "2024-02-02", 20.0, 108.0, 60.0),
        _atom("p2", "2024-02-02", 70.0, 108.0, 110.0),
        _atom("s2", "支出", 120.0, 138.0, 150.0),
        _atom("dir2", "支", 180.0, 138.0, 190.0),
        _atom("a2", "-3.00", 220.0, 138.0, 250.0),
        _atom("b2", "15.00", 270.0, 138.0, 305.0),
        _atom("l2", "另一支行", 320.0, 138.0, 370.0),
        _atom("party2", "乙公司", 400.0, 138.0, 445.0),
        _atom("account2", "10002", 460.0, 138.0, 510.0),
        _atom("footer", "打印完毕", 20.0, 170.0, 100.0),
    ]
    for atom_id in ("d1", "p1", "d2", "p2"):
        atom = next(item for item in atoms if item["id"] == atom_id)
        atom["bbox"][3] = atom["bbox"][1] + 40.0

    header = _geometry_header(_expand_composite_header_atoms(atoms))
    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert header is not None
    assert all("2024-02" not in str(atom.get("text") or "") for atom in header[0])
    assert len(tables) == 1
    assert [row[0] for row in tables[0][1:]] == ["2024-02-01", "2024-02-02"]


def test_geometry_recovery_prefers_sequence_spine_and_repairs_glued_cells() -> None:
    atoms = [
        _atom("hs", "序号", 20.0, 80.0, 40.0),
        _atom("hd", "记账日期", 55.0, 80.0, 100.0),
        _atom("ha", "交易金额", 145.0, 80.0, 200.0),
        _atom("hb", "账户余额", 225.0, 80.0, 280.0),
        _atom("hm", "摘要描述", 300.0, 80.0, 370.0),
        _atom("hp", "对方户名", 430.0, 80.0, 500.0),
        _atom("s1", "1", 25.0, 110.0, 32.0),
        _atom("d1", "2025-01-02", 55.0, 110.0, 100.0),
        _atom("a1", "-10.00", 160.0, 110.0, 195.0),
        _atom("b1", "90.00税费社保", 230.0, 110.0, 330.0),
        _atom("p1", "待报解预算收入", 430.0, 110.0, 500.0),
        _atom("s2", "2", 25.0, 130.0, 32.0),
        _atom("d2", "2025-01--03", 55.0, 130.0, 105.0),
        _atom("a2", "5.00", 165.0, 130.0, 195.0),
        _atom("b2", "85.00电子银行转账", 230.0, 130.0, 335.0),
        _atom("p2", "测试公司", 430.0, 130.0, 480.0),
        _atom("s3", "3", 25.0, 150.0, 32.0),
        _atom("d3", "2025-01-04", 55.0, 150.0, 100.0),
        _atom("a3", "10.00", 165.0, 150.0, 195.0),
        _atom("b3", "75.00", 230.0, 150.0, 275.0),
        _atom("m3", "0网银路行互联", 300.0, 150.0, 370.0),
        _atom("p3", "另一公司", 430.0, 150.0, 480.0),
        _atom("footer-in", "收入总额:0.00", 25.0, 175.0, 120.0),
        _atom("footer-out", "支出总额:25.00", 230.0, 175.0, 330.0),
        _atom("print-date", "打印日期:2025-02-01", 55.0, 190.0, 180.0),
    ]
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert recovered_evidence_atom_expected_row_count(parse_result) == 3
    assert recovered_evidence_atom_expected_row_evidence(parse_result) == (3, "positioned_date_anchors", 0.80)
    assert len(tables) == 1
    assert len(tables[0]) == 4
    assert tables[0][1] == ["1", "2025-01-02", "-10.00", "90.00", "税费社保", "待报解预算收入"]
    assert tables[0][2] == ["2", "2025-01-03", "-5.00", "85.00", "电子银行转账", "测试公司"]
    assert tables[0][3] == ["3", "2025-01-04", "-10.00", "75.00", "网银跨行互联", "另一公司"]
    assert all(source["row_anchor_type"] == "sequence" for source in sources)
    assert any(source.get("reconstruction_repairs") for source in sources)


def test_geometry_recovery_carries_proven_schema_to_headerless_continuation_page() -> None:
    def on_page(atom: dict, page: int) -> dict:
        return {**atom, "page_id": f"page:{page:04d}"}

    page_one = [
        _atom("hd", "交易时间", 55.0, 80.0, 105.0),
        _atom("ha", "交易金额", 145.0, 80.0, 200.0),
        _atom("hb", "账户余额", 225.0, 80.0, 280.0),
        _atom("ht", "交易类型", 300.0, 80.0, 360.0),
        _atom("hn", "交易备注", 390.0, 80.0, 470.0),
        _atom("d1", "2025-01-02 08:10:00", 55.0, 110.0, 120.0),
        _atom("a1", "+10.00", 160.0, 110.0, 195.0),
        _atom("b1", "110.00", 235.0, 110.0, 275.0),
        _atom("t1", "汇入汇款", 310.0, 110.0, 355.0),
        _atom("n1", "首笔", 400.0, 110.0, 430.0),
    ]
    page_two = [
        on_page(_atom("page", "Page 2 of 2", 390.0, 5.0, 470.0), 2),
        on_page(_atom("d2", "2025-01-03 09:20:00", 55.0, 58.0, 120.0), 2),
        on_page(_atom("a2", "-5.00", 160.0, 58.0, 195.0), 2),
        on_page(_atom("b2", "105.00", 235.0, 58.0, 275.0), 2),
        on_page(_atom("t2", "协议支付", 310.0, 58.0, 355.0), 2),
        on_page(_atom("n2", "续页首笔", 400.0, 58.0, 455.0), 2),
        on_page(_atom("wrap", "补充附言", 400.0, 83.0, 455.0), 2),
        on_page(_atom("d3", "2025-01-04 10:30:00", 55.0, 92.0, 120.0), 2),
        on_page(_atom("a3", "+20.00", 160.0, 92.0, 195.0), 2),
        on_page(_atom("b3", "125.00", 235.0, 92.0, 275.0), 2),
        on_page(_atom("t3", "转账", 310.0, 92.0, 355.0), 2),
        on_page(_atom("n3", "续页次笔", 400.0, 92.0, 455.0), 2),
    ]
    # These tokens have the right x positions but not one transaction baseline;
    # carrying the schema must not convert them into a third-page ledger.
    page_three = [
        on_page(_atom("prose-date", "截至2025-01-05", 55.0, 70.0, 120.0), 3),
        on_page(_atom("prose-amount", "说明金额30.00", 160.0, 90.0, 210.0), 3),
        on_page(_atom("prose-balance", "参考余额155.00", 235.0, 110.0, 300.0), 3),
    ]
    parse_result = _result([*page_one, *page_two, *page_three])

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert [len(table) - 1 for table in tables] == [1, 2]
    assert sum(len(table) - 1 for table in tables) == 3
    assert [source["source_page"] for source in sources] == [1, 2, 2]
    assert tables[1][1][-1] == "续页首笔"
    assert tables[1][2][-1] == "补充附言续页次笔"
    assert tables[1][2][:3] == ["2025-01-0410:30:00", "+20.00", "125.00"]


def test_geometry_frame_rules_do_not_merge_multiple_rows() -> None:
    atoms = [
        _atom("hd", "交易日期", 55.0, 80.0, 105.0),
        _atom("ha", "交易金额", 145.0, 80.0, 200.0),
        _atom("hb", "账户余额", 225.0, 80.0, 280.0),
        _atom("d1", "2025-02-01", 55.0, 110.0, 105.0),
        _atom("a1", "+1.00", 160.0, 110.0, 195.0),
        _atom("b1", "11.00", 235.0, 110.0, 275.0),
        _atom("d2", "2025-02-02", 55.0, 130.0, 105.0),
        _atom("a2", "+2.00", 160.0, 130.0, 195.0),
        _atom("b2", "13.00", 235.0, 130.0, 275.0),
    ]
    frame_rules = [
        {"id": "top", "page_id": "page:0001", "bbox": [0.0, 100.0, 600.0, 100.0]},
        {"id": "bottom", "page_id": "page:0001", "bbox": [0.0, 145.0, 600.0, 145.0]},
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms, frame_rules))

    assert tables == [
        [
            ["交易日期", "交易金额", "账户余额"],
            ["2025-02-01", "+1.00", "11.00"],
            ["2025-02-02", "+2.00", "13.00"],
        ]
    ]


def test_compound_datetime_and_money_header_expands_to_semantic_roles() -> None:
    atoms = _expand_composite_header_atoms(
        [
            _atom("datetime", "日期时间", 20.0, 80.0, 90.0),
            _atom("money", "支/收交易金额账户余额交易地点", 120.0, 80.0, 400.0),
            _atom("party", "对方户名", 430.0, 80.0, 500.0),
        ]
    )

    header = _geometry_header(atoms)

    assert header is not None
    assert set(header[1]).issuperset({"timestamp", "direction", "amount", "balance"})


def test_compound_direction_amount_header_preserves_dedicated_direction() -> None:
    atoms = _expand_composite_header_atoms(
        [
            _atom("date", "交易日期", 20.0, 80.0, 90.0),
            _atom("summary", "摘要", 100.0, 80.0, 150.0),
            _atom("money", "支/收交易金额", 160.0, 80.0, 300.0),
            _atom("balance", "账户余额", 320.0, 80.0, 390.0),
        ]
    )

    header = _geometry_header(atoms)
    normalized = BankStatementCommunityPlugin()._normalize(
        {"交易日期": "2025-02-01", "收/支": "收", "交易金额": "12.34", "账户余额": "56.78"}
    )

    assert header is not None
    assert set(header[1]).issuperset({"date", "direction", "amount", "balance"})
    assert normalized["direction"] == "income"


def test_geometry_recovery_preserves_second_tier_business_column_per_row() -> None:
    atoms = [
        _atom("hd", "日期时间", 20.0, 80.0, 90.0),
        _atom("hs", "日志号", 130.0, 80.0, 180.0),
        _atom("hm", "短摘要", 200.0, 80.0, 250.0),
        _atom("ha", "交易金额", 270.0, 80.0, 320.0),
        _atom("hb", "本次余额", 360.0, 80.0, 410.0),
        _atom("aux-header", "对方账号户名/附言", 130.0, 95.0, 250.0),
        _atom("d1", "20250102081000", 20.0, 120.0, 100.0),
        _atom("s1", "1001", 140.0, 120.0, 175.0),
        _atom("m1", "转账", 210.0, 120.0, 240.0),
        _atom("a1", "+10.00", 280.0, 120.0, 315.0),
        _atom("b1", "110.00", 370.0, 120.0, 405.0),
        _atom("aux1a", "622200001", 130.0, 133.0, 180.0),
        _atom("aux1b", "第一笔附言", 185.0, 133.0, 250.0),
        _atom("d2", "20250103092000", 20.0, 150.0, 100.0),
        _atom("s2", "1002", 140.0, 150.0, 175.0),
        _atom("m2", "支付", 210.0, 150.0, 240.0),
        _atom("a2", "-5.00", 280.0, 150.0, 315.0),
        _atom("b2", "105.00", 370.0, 150.0, 405.0),
        _atom("aux2a", "622200002", 130.0, 163.0, 180.0),
        _atom("aux2b", "第二笔附言", 185.0, 163.0, 250.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert tables == [
        [
            ["日期时间", "日志号", "短摘要", "交易金额", "本次余额", "对方账号户名/附言"],
            ["20250102081000", "1001", "转账", "+10.00", "110.00", "622200001第一笔附言"],
            ["20250103092000", "1002", "支付", "-5.00", "105.00", "622200002第二笔附言"],
        ]
    ]


def test_icbc_geometry_recovery_preserves_exact_raw_source_headers() -> None:
    headers = [
        "交易日期",
        "账号",
        "储种",
        "序号",
        "币种",
        "钞汇",
        "摘要",
        "地区",
        "收入/支出金额",
        "余额",
        "渠道",
    ]
    row = [
        "2022-09-0410:01:14",
        "1104060001031076947",
        "活期",
        "00000",
        "人民币",
        "钞",
        "消费",
        "1104",
        "-23.00",
        "268.08",
        "快捷支付",
    ]
    centers = [35.0, 105.0, 185.0, 235.0, 285.0, 330.0, 375.0, 420.0, 475.0, 535.0, 580.0]
    atoms = [
        _atom(f"h{index}", header, center - 18.0, 80.0, center + 18.0)
        for index, (header, center) in enumerate(zip(headers, centers))
    ]
    atoms.extend(
        _atom(f"r{index}", value, center - 18.0, 110.0, center + 18.0)
        for index, (value, center) in enumerate(zip(row, centers))
    )
    parse_result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert tables == [[headers, row]]
    assert sources[0]["source_raw"] == dict(zip(headers, row))


def test_candidate_selection_rejects_balance_chain_weaker_near_tie():
    selected, _diagnostics = _select_candidate(
        [
            _candidate("evidence_atom", score=0.91, balance_chain_score=1.0),
            _candidate("fallback", score=0.87, balance_chain_score=0.0),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "evidence_atom"


def test_candidate_selection_preserves_richer_equal_quality_source_columns():
    selected, diagnostics = _select_candidate(
        [
            _candidate("parser:grid_standard", source_column_width=8.0),
            _candidate(
                "evidence_atom",
                expected_rows=RowCountEvidence(count=3, source="positioned_date_anchors", confidence=0.95),
                extraction_confidence=0.95,
                source_column_width=7.0,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "parser:grid_standard"
    assert diagnostics["selected_candidate"] == "parser:grid_standard"


def test_candidate_selection_prefers_equal_quality_native_physical_cells() -> None:
    evidence = RowCountEvidence(count=497, source="header_total", confidence=0.94)
    selected, diagnostics = _select_candidate(
        [
            _candidate(
                "evidence_atom",
                rows=497,
                expected_rows=evidence,
                extraction_confidence=0.90,
                source_column_width=8.0,
            ),
            _candidate(
                "native_wide_table",
                rows=497,
                expected_rows=evidence,
                extraction_confidence=0.85,
                source_column_width=8.0,
                native_cell_coverage=1.0,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "native_wide_table"
    assert diagnostics["selected_candidate"] == "native_wide_table"


def test_candidate_derived_count_cannot_replace_full_native_candidate():
    selected, _diagnostics = _select_candidate(
        [
            _candidate("parser:grid_standard", rows=4),
            _candidate(
                "evidence_atom",
                rows=3,
                expected_rows=RowCountEvidence(count=3, source="positioned_date_anchors", confidence=0.95),
                extraction_confidence=0.95,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "parser:grid_standard"


def test_candidate_selection_penalizes_rows_above_independent_total():
    evidence = RowCountEvidence(count=10, source="header_total", confidence=0.94)
    selected, _diagnostics = _select_candidate(
        [
            _candidate("exact", rows=10, expected_rows=evidence),
            _candidate("over_extracted", rows=12, expected_rows=evidence),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "exact"


def test_source_width_tiebreaker_cannot_override_exact_independent_count() -> None:
    evidence = RowCountEvidence(count=10, source="header_total", confidence=0.94)
    selected, _diagnostics = _select_candidate(
        [
            _candidate("exact", rows=10, expected_rows=evidence, source_column_width=7.0),
            _candidate("wider_over", rows=12, expected_rows=evidence, source_column_width=8.0),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "exact"


def test_partial_sequence_cannot_override_conflicting_issuer_total() -> None:
    issuer_total = RowCountEvidence(count=63, source="split_footer", confidence=0.98)
    partial_sequence = RowCountEvidence(count=27, source="continuous_source_sequence", confidence=0.99)

    selected, _diagnostics = _select_candidate(
        [
            _candidate("complete", rows=63, expected_rows=issuer_total),
            _candidate("truncated_sequence", rows=27, expected_rows=partial_sequence),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "complete"


def test_partial_sequence_evidence_is_not_used_against_conflicting_issuer_total() -> None:
    rows = [{"序号": str(index), "交易日期": "2024-01-01"} for index in range(1, 13)]
    sequence = _continuous_source_sequence_evidence(rows)
    issuer_total = RowCountEvidence(count=475, source="split_footer", confidence=0.98)

    if sequence is not None and issuer_total.count != sequence.count:
        sequence = None

    assert sequence is None


def test_page_anchor_conflict_does_not_promote_bare_candidate_sequence() -> None:
    anchors = RowCountEvidence(count=475, source="page_transaction_anchors", confidence=0.93)
    rows = [{"序号": str(index), "交易日期": "2024-01-01"} for index in range(1, 550)]

    assert _candidate_row_count_evidence(rows, anchors) == anchors


def test_non_one_sequence_span_does_not_self_certify_from_amount_totals() -> None:
    rows = [{"序号": str(value), "_source": {"source_page": 1 + (value - 571) // 19}} for value in range(571, 656)]
    page_texts = [(page, "序号 记账日 借方发生额 贷方发生额 余额\n借方合计 1.00 贷方合计 2.00") for page in range(1, 6)]
    evidence = _page_complete_sequence_evidence(rows, page_count=5, page_texts=page_texts)

    assert evidence is None


def test_complete_page_local_sequence_resets_prove_sum() -> None:
    rows = [
        {"序号": str(value), "_source": {"source_page": page}}
        for page, count in enumerate((20, 21, 19, 15), start=1)
        for value in range(1, count + 1)
    ]

    counts = (20, 21, 19, 15)
    page_texts = []
    for page, count in enumerate(counts, start=1):
        rows_text = "\n".join(
            f"| {value} |220401|220401|交易| |REF{value}| | 1.00| 2.00|SERIAL|NOTE|" for value in range(1, count + 1)
        )
        page_texts.append(
            (
                page,
                "\n".join(
                    [
                        f"Page {page} of 4",
                        "|No. |Bk.D. |Val.D.| Type |Vou.| Details | Debit Amount | Credit Amount | Balance | Reference No. | Notes |",
                        rows_text,
                        "Debit Total Credit Total Current Page Balance",
                    ]
                ),
            )
        )
    evidence = _page_complete_sequence_evidence(rows, page_count=4, page_texts=page_texts)

    assert evidence == RowCountEvidence(75, "complete_page_local_sequences", 0.80)


def test_page_local_sequence_source_census_rejects_missing_candidate_tail() -> None:
    candidate_rows = [{"序号": str(value), "_source": {"source_page": 1}} for value in range(1, 6)]
    source_rows = "\n".join(f"| {value} |220401|220401|交易| |REF| | 1.00| 2.00|SERIAL|NOTE|" for value in range(1, 7))
    page_text = "\n".join(
        [
            "Page 1 of 1",
            "|No. |Bk.D. |Val.D.| Type |Vou.| Details | Debit Amount | Credit Amount | Balance | Reference No. | Notes |",
            source_rows,
            "Debit Total Credit Total Current Page Balance",
        ]
    )

    assert (
        _page_complete_sequence_evidence(
            candidate_rows,
            page_count=1,
            page_texts=[(1, page_text)],
        )
        is None
    )


def test_page_complete_sequence_evidence_rejects_gap_reset_and_missing_page() -> None:
    assert (
        _page_complete_sequence_evidence(
            [
                {"序号": "1", "_source": {"source_page": 1}},
                {"序号": "3", "_source": {"source_page": 1}},
            ],
            page_count=1,
            page_texts=[(1, "序号 记账日\n借方合计 1.00 贷方合计 2.00")],
        )
        is None
    )
    assert (
        _page_complete_sequence_evidence(
            [
                {"序号": "1", "_source": {"source_page": 1}},
                {"序号": "2", "_source": {"source_page": 1}},
            ],
            page_count=2,
            page_texts=[
                (1, "序号 记账日\n借方合计 1.00 贷方合计 2.00"),
                (2, "序号 记账日\n借方合计 1.00 贷方合计 2.00"),
            ],
        )
        is None
    )


def test_page_sequence_prefixes_do_not_prove_count_without_source_boundaries() -> None:
    rows = [{"序号": str(value), "_source": {"source_page": page}} for page in (1, 2) for value in range(1, 6)]

    assert (
        _page_complete_sequence_evidence(
            rows,
            page_count=2,
            page_texts=[(1, "序号 记账日"), (2, "序号 记账日")],
        )
        is None
    )


def test_non_one_span_does_not_prove_omitted_edge_rows_without_source_boundaries() -> None:
    rows = [{"序号": str(value), "_source": {"source_page": 1 if value <= 20 else 2}} for value in range(11, 31)]

    assert (
        _page_complete_sequence_evidence(
            rows,
            page_count=2,
            page_texts=[(1, "序号 记账日"), (2, "序号 记账日")],
        )
        is None
    )


def test_fuller_candidate_outranks_shorter_heuristic_page_anchors() -> None:
    selected, _diagnostics = _select_candidate(
        [
            _candidate(
                "evidence_atom",
                rows=475,
                expected_rows=RowCountEvidence(
                    count=475,
                    source="page_transaction_anchors",
                    confidence=0.93,
                ),
                field_completeness=1.0,
                sequence_continuity=0.50,
            ),
            _candidate(
                "native_wide_table",
                rows=549,
                expected_rows=RowCountEvidence(
                    count=549,
                    source="continuous_source_sequence",
                    confidence=0.80,
                ),
                field_completeness=0.989,
                native_cell_coverage=1.0,
                sequence_continuity=1.0,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "native_wide_table"


def test_fuller_candidate_tolerates_tiny_relative_optional_column_sparsity() -> None:
    selected, _diagnostics = _select_candidate(
        [
            _candidate(
                "native_wide_table:page_prefix",
                rows=14,
                score=0.9925,
                extraction_confidence=0.99,
                native_cell_coverage=1.0,
                source_column_width=14.5714,
            ),
            _candidate(
                "parser:grid_standard",
                rows=70,
                score=0.9900,
                extraction_confidence=0.95,
                source_column_width=14.3143,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "parser:grid_standard"


def test_full_exact_sequence_outranks_shorter_exact_prefix() -> None:
    selected, _diagnostics = _select_candidate(
        [
            _candidate(
                "parser:grid_standard",
                rows=475,
                expected_rows=RowCountEvidence(
                    count=475,
                    source="continuous_source_sequence",
                    confidence=0.99,
                ),
                source_page_coverage=0.95,
                balance_chain_score=0.97,
            ),
            _candidate(
                "native_wide_table",
                rows=12,
                expected_rows=RowCountEvidence(
                    count=12,
                    source="continuous_source_sequence",
                    confidence=0.99,
                ),
                source_page_coverage=1.0,
                balance_chain_score=1.0,
                native_cell_coverage=1.0,
            ),
        ]
    )

    assert selected is not None
    assert selected.candidate_id == "parser:grid_standard"


def test_registry_does_not_reemit_semantically_rejected_candidate_through_fallback(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.style_registry as registry_module
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult
    from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry

    rejected = BankTableCandidate(
        candidate_id="parser:grid_standard",
        records=[{"交易日期": "2025-01-02", "交易金额": "25.00", "收/支": "支出"}],
        source="parser:grid_standard",
        canonical_rows=0,
        directional_rows=1,
        source_page_rows=0,
        expected_rows=None,
        balance_chain_score=0.0,
        field_completeness=1.0,
        score=0.0,
        rejection_reason="source_role_corruption",
    )
    monkeypatch.setattr(registry_module, "_collect_table_candidates", lambda *_args, **_kwargs: [rejected])

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("a semantically rejected candidate must not bypass selection through a fallback parser")

    monkeypatch.setattr(registry_module, "_run_parser", forbidden_fallback)
    ctx = StyleContext(tables=[], full_text="", institution=None, page_count=1, parse_result=None)
    detection = StyleDetectionResult(primary_style="grid_standard", parser_chain=["grid_standard"])

    records, _identity = BankStyleParserRegistry(adaptive=False).run(
        detection,
        ctx,
        BankStatementCommunityPlugin(),
    )

    assert records == []


def test_native_recovery_streams_compete_as_whole_table_alternatives(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.style_registry as registry_module
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult

    headers = ["序号", "交易日期", "交易金额", "余额"]
    first = [headers, ["1", "2025-01-01", "+1.00", "1.00"], ["2", "2025-01-02", "+1.00", "2.00"]]
    second = [headers, ["1", "2025-01-01", "+1.00", "1.00"], ["2", "2025-01-02", "+1.00", "2.00"]]
    monkeypatch.setattr(registry_module, "recover_wide_bank_tables", lambda *_args: [first, second])
    monkeypatch.setattr(registry_module, "recover_evidence_atom_bank_tables", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(registry_module, "recover_positioned_record_block_bank_tables", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(registry_module, "_semantic_text_table_candidates", lambda *_args: [])
    monkeypatch.setattr(
        registry_module,
        "_run_parser",
        lambda _parser, ctx, _plugin: (
            [
                {
                    "序号": row[0],
                    "交易日期": row[1],
                    "交易金额": row[2],
                    "余额": row[3],
                }
                for table in ctx.tables
                for row in table[1:]
            ],
            lambda raw: {
                "sequence_no": raw["序号"],
                "date": raw["交易日期"],
                "amount": abs(float(raw["交易金额"])),
                "direction": "income" if raw["交易金额"].startswith("+") else "expense",
                "balance": float(raw["余额"]),
            },
        ),
    )
    plugin = SimpleNamespace(
        standard_fields=["sequence_no", "date", "amount", "direction", "balance"],
        _normalize=lambda raw: raw,
    )
    ctx = StyleContext(tables=[], full_text="", institution=None, page_count=1, parse_result=None)
    detection = StyleDetectionResult(primary_style="grid_standard", parser_chain=[])

    candidates = _collect_table_candidates(detection, ctx, plugin)
    native = [candidate for candidate in candidates if candidate.candidate_id.startswith("native_wide_table")]

    assert [(candidate.candidate_id, len(candidate.records)) for candidate in native] == [
        ("native_wide_table:0", 2),
        ("native_wide_table:1", 2),
    ]
    selected, _diagnostics = _select_candidate(native)
    assert selected is not None
    assert len(selected.records) == 2


def test_disjoint_native_streams_combine_when_independent_total_proves_union(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.style_registry as registry_module
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult

    headers = ["序号", "交易日期", "交易金额", "余额"]
    first = [headers, ["1", "2025-01-01", "+1.00", "1.00"], ["2", "2025-01-02", "+1.00", "2.00"]]
    second = [headers, ["3", "2025-01-03", "+1.00", "3.00"], ["4", "2025-01-04", "+1.00", "4.00"]]
    monkeypatch.setattr(registry_module, "recover_wide_bank_tables", lambda *_args: [first, second])
    monkeypatch.setattr(registry_module, "recover_evidence_atom_bank_tables", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(registry_module, "recover_positioned_record_block_bank_tables", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(registry_module, "_semantic_text_table_candidates", lambda *_args: [])
    monkeypatch.setattr(
        registry_module,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence(4, "split_footer", 0.98),
    )
    monkeypatch.setattr(
        registry_module,
        "_run_parser",
        lambda _parser, ctx, _plugin: (
            [
                {"序号": row[0], "交易日期": row[1], "交易金额": row[2], "余额": row[3]}
                for table in ctx.tables
                for row in table[1:]
            ],
            lambda raw: {
                "sequence_no": raw["序号"],
                "date": raw["交易日期"],
                "amount": abs(float(raw["交易金额"])),
                "direction": "income",
                "balance": float(raw["余额"]),
            },
        ),
    )
    plugin = SimpleNamespace(
        standard_fields=["sequence_no", "date", "amount", "direction", "balance"],
        _normalize=lambda raw: raw,
    )
    ctx = StyleContext(tables=[], full_text="", institution=None, page_count=1, parse_result=None)
    detection = StyleDetectionResult(primary_style="grid_standard", parser_chain=[])

    native = [
        candidate
        for candidate in _collect_table_candidates(detection, ctx, plugin)
        if candidate.candidate_id.startswith("native_wide_table")
    ]
    selected, _diagnostics = _select_candidate(native)

    assert selected is not None
    assert selected.candidate_id == "native_wide_table:combined"
    assert [record["序号"] for record in selected.records] == ["1", "2", "3", "4"]


def test_overlapping_native_streams_do_not_combine_even_when_counts_sum_to_issuer_total(monkeypatch) -> None:
    import docmirror.plugins.bank_statement.style_registry as registry_module
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import StyleDetectionResult

    headers = ["序号", "交易日期", "交易金额", "余额"]
    first = [headers, ["1", "2025-01-01", "+1.00", "1.00"], ["2", "2025-01-02", "+1.00", "2.00"]]
    second = [headers, ["1", "2025-01-01", "+1.00", "1.00"], ["2", "2025-01-02", "+1.00", "2.00"]]
    monkeypatch.setattr(registry_module, "recover_wide_bank_tables", lambda *_args: [first, second])
    monkeypatch.setattr(registry_module, "recover_evidence_atom_bank_tables", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(registry_module, "recover_positioned_record_block_bank_tables", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(registry_module, "_semantic_text_table_candidates", lambda *_args: [])
    monkeypatch.setattr(
        registry_module,
        "resolve_row_count_evidence",
        lambda *_args, **_kwargs: RowCountEvidence(4, "split_footer", 0.98),
    )

    def run_parser(_parser, ctx, _plugin):
        rows = []
        for table_index, table in enumerate(ctx.tables):
            for row_index, row in enumerate(table[1:], start=1):
                rows.append(
                    {
                        "序号": row[0],
                        "交易日期": row[1],
                        "交易金额": row[2],
                        "余额": row[3],
                        "_source": {
                            "source_page": 1,
                            "bbox": [10.0, row_index * 20.0 + table_index, 200.0, row_index * 20.0 + 10.0],
                        },
                    }
                )
        return rows, lambda raw: {
            "sequence_no": raw["序号"],
            "date": raw["交易日期"],
            "amount": abs(float(raw["交易金额"])),
            "direction": "income",
            "balance": float(raw["余额"]),
        }

    monkeypatch.setattr(registry_module, "_run_parser", run_parser)
    plugin = SimpleNamespace(
        standard_fields=["sequence_no", "date", "amount", "direction", "balance"],
        _normalize=lambda raw: raw,
    )
    ctx = StyleContext(tables=[], full_text="", institution=None, page_count=1, parse_result=None)
    detection = StyleDetectionResult(primary_style="grid_standard", parser_chain=[])

    native = [
        candidate
        for candidate in _collect_table_candidates(detection, ctx, plugin)
        if candidate.candidate_id.startswith("native_wide_table")
    ]

    assert [candidate.candidate_id for candidate in native] == ["native_wide_table:0", "native_wide_table:1"]


def test_recovers_complete_split_debit_credit_rows():
    atoms = [
        _atom("hs", "序号", 25.0, 80.0, 41.0),
        _atom("hd", "交易日期", 52.0, 80.0, 85.0),
        _atom("hr", "交易流水号", 108.0, 80.0, 149.0),
        _atom("hm", "支出（元）收入（元）账户余额（元）", 165.0, 80.0, 312.0),
        _atom("ha", "对方账号", 339.0, 80.0, 372.0),
        _atom("hp", "对方户名", 428.0, 80.0, 461.0),
        _atom("hn", "对方行号", 505.0, 80.0, 538.0),
        _atom("hbn", "对方行名", 563.0, 80.0, 596.0),
        _atom("hc", "交易渠道", 613.0, 80.0, 646.0),
        _atom("hpu", "用途", 698.0, 80.0, 715.0),
        _atom("hsu", "摘要", 782.0, 80.0, 799.0),
        _atom("s1", "1", 30.0, 110.0, 34.0),
        _atom("d1", "20260102", 49.0, 110.0, 88.0),
        _atom("r1", "REF001", 108.0, 110.0, 145.0),
        _atom("e1", "12.34", 181.0, 110.0, 205.2),
        _atom("b1", "100.00", 281.0, 110.0, 313.1),
        _atom("a1", "1234567890", 340.0, 110.0, 390.0),
        _atom("p1", "甲公司", 428.0, 110.0, 460.0),
        _atom("bn1", "BANK001", 505.0, 110.0, 535.0),
        _atom("bname1", "测试银行", 563.0, 110.0, 600.0),
        _atom("channel1", "网银", 613.0, 110.0, 635.0),
        _atom("purpose1", "货款", 698.0, 110.0, 720.0),
        _atom("summary1", "转账", 782.0, 110.0, 804.0),
        _atom("s2", "2", 30.0, 140.0, 34.0),
        _atom("d2", "20260103", 49.0, 140.0, 88.0),
        _atom("r2", "REF002", 108.0, 140.0, 145.0),
        _atom("i2", "20.00", 224.0, 140.0, 248.3),
        _atom("b2", "120.00", 281.0, 140.0, 313.1),
        _atom("s3", "3", 30.0, 170.0, 34.0),
        _atom("d3", "20260104", 49.0, 170.0, 88.0),
        _atom("r3", "REF003", 108.0, 170.0, 145.0),
        _atom("e3", "1.00", 181.0, 170.0, 205.2),
        _atom("b3", "119.00", 281.0, 170.0, 313.1),
        _atom("s4", "4", 30.0, 200.0, 34.0),
        _atom("d4", "20260105", 49.0, 200.0, 88.0),
        _atom("r4", "REF004", 108.0, 200.0, 145.0),
        _atom("i4", "1.00", 224.0, 200.0, 248.3),
        _atom("b4", "120.00", 281.0, 200.0, 313.1),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][0] == [
        "序号",
        "交易日期",
        "交易流水号",
        "支出金额",
        "收入金额",
        "余额",
        "对方账号",
        "对方户名",
        "对方行号",
        "对方行名",
        "交易渠道",
        "用途",
        "摘要",
    ]
    assert tables[0][1] == [
        "1",
        "20260102",
        "REF001",
        "12.34",
        "",
        "100.00",
        "1234567890",
        "甲公司",
        "BANK001",
        "测试银行",
        "网银",
        "货款",
        "转账",
    ]
    assert tables[0][2][:6] == ["2", "20260103", "REF002", "", "20.00", "120.00"]
    assert len(tables[0]) == 5


def test_rejects_layout_without_complete_issuer_headers():
    atoms = [
        _atom("hd", "交易日期", 52.0, 80.0),
        _atom("d1", "20260102", 49.0, 110.0),
        _atom("a1", "12.34", 181.0, 110.0, 205.2),
    ]

    assert recover_evidence_atom_bank_tables(_result(atoms)) == []


def test_recovers_rotated_column_major_record_blocks_with_single_page_sources():
    atoms = [
        _atom(
            "header",
            "\u5e8f\u53f7\n\u6458\u8981\n\u5e01\u522b\n\u949e\u6c47\n\u4ea4\u6613\u65e5\u671f\n"
            "\u4ea4\u6613\u91d1\u989d\n\u8d26\u6237\u4f59\u989d\n\u4ea4\u6613\u5730\u70b9/\u9644\u8a00\n"
            "\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
            20.0,
            20.0,
            32.0,
        ),
        _atom(
            "row-1",
            "1\n\u94f6\u8054\u5165\u8d26\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n100.00\n100.00\n"
            "\u5546\u6237\n1234567890/\u7532\u516c\u53f8",
            50.0,
            20.0,
            62.0,
        ),
        _atom(
            "row-2",
            "2\n\u8f6c\u8d26\u652f\u53d6\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n-20.00\n80.00\n"
            "\u5546\u6237\n1234567891/\u4e59\u516c\u53f8",
            70.0,
            20.0,
            82.0,
        ),
        _atom(
            "row-3",
            "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n10.00\n90.00\n"
            "\u5546\u6237\n6217002120017593862/\u4e19\u516c\u53f8",
            90.0,
            20.0,
            102.0,
        ),
    ]

    recovery = recover_positioned_record_block_bank_tables(_result(atoms))

    assert recovery.expected_rows == 3
    assert len(recovery.tables) == 1
    assert len(recovery.tables[0]) == 4
    assert recovery.tables[0][1][3:6] == ["", "100.00", "100.00"]
    assert recovery.tables[0][2][3:6] == ["20.00", "", "80.00"]
    assert recovery.tables[0][3][3:6] == ["", "10.00", "90.00"]
    assert [source["source_page"] for source in recovery.row_sources] == [1, 1, 1]
    assert all(source["page_range"] == [1, 1] for source in recovery.row_sources)
    assert recovery.row_sources[0]["source_raw"] == {
        "序号": "1",
        "摘要": "银联入账",
        "币别": "人民币元",
        "钞汇": "钞",
        "交易日期": "20211025",
        "交易金额": "100.00",
        "账户余额": "100.00",
        "交易地点/附言": "商户",
        "对方账号与户名": "1234567890/甲公司",
    }


def test_positioned_recovery_keeps_short_continuation_page():
    atoms: list[dict] = []
    for sequence, page_number in [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2)]:
        atom = _atom(
            f"row-{sequence}",
            f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
            f"6222020202020{sequence:03d}/甲公司",
            20.0,
            20.0 + (sequence - 1) * 20.0,
            60.0,
        )
        atom["page_id"] = f"page:{page_number:04d}"
        atoms.append(atom)

    recovery = recover_positioned_record_block_bank_tables(_result(atoms))

    assert recovery.expected_rows == 5
    assert sum(len(table) - 1 for table in recovery.tables) == 5
    assert [source["source_page"] for source in recovery.row_sources] == [1, 1, 1, 2, 2]


def test_positioned_page_text_uses_actual_source_page_number():
    blocks = [
        SimpleNamespace(
            content=(
                f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
                f"6222020202020{sequence:03d}/甲公司"
            ),
            bbox=[20.0, 20.0 + sequence * 20.0, 60.0, 30.0 + sequence * 20.0],
            evidence_ids=[],
        )
        for sequence in range(1, 4)
    ]
    result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={}),
        pages=[
            SimpleNamespace(
                page_number=1,
                source_page_number=5,
                width=600,
                height=850,
                tables=[],
                texts=blocks,
            )
        ],
        logical_tables=[],
        entities=SimpleNamespace(domain_specific={}),
    )

    recovery = recover_positioned_record_block_bank_tables(result)

    assert [source["source_page"] for source in recovery.row_sources] == [5, 5, 5]
    assert all(source["page_range"] == [5, 5] for source in recovery.row_sources)


def test_geometry_orientation_uses_transaction_columns_instead_of_header_dates():
    atoms = [
        _atom("h-seq", "序号", 20.0, 100.0, 40.0),
        _atom("h-date", "交易日期", 60.0, 100.0, 92.0),
        _atom("h-time", "交易时间", 110.0, 100.0, 142.0),
        _atom("h-type", "交易类型", 160.0, 100.0, 192.0),
        _atom("h-direction", "借贷", 210.0, 100.0, 226.0),
        _atom("h-amount", "交易金额", 250.0, 100.0, 282.0),
        _atom("h-balance", "余额", 320.0, 100.0, 336.0),
        _atom("h-account", "对方账号", 370.0, 100.0, 402.0),
        _atom("h-party", "对方户名", 430.0, 100.0, 462.0),
        _atom("h-channel", "交易地点", 490.0, 100.0, 522.0),
        _atom("h-summary", "摘要", 550.0, 100.0, 566.0),
    ]
    for index in range(6):
        atoms.append(_atom(f"query-date-{index}", f"2024-01-{index + 1:02d}", 300.0, 20.0 + index * 10.0, 340.0))
    atoms.extend(
        [
            _atom("r1-seq", "1", 25.0, 130.0, 29.0),
            _atom("r1-date", "2022-08-05", 60.0, 130.0, 100.0),
            _atom("r1-time", "14:05:18", 110.0, 130.0, 142.0),
            _atom("r1-type", "跨行汇款", 160.0, 130.0, 192.0),
            _atom("r1-direction", "贷 Cr", 210.0, 130.0, 230.0),
            _atom("r1-amount", "40.00", 250.0, 130.0, 272.0),
            _atom("r1-balance", "41.06", 320.0, 130.0, 342.0),
            _atom("r1-account", "6214857212810271", 370.0, 130.0, 426.0),
            _atom("r1-party", "周深", 440.0, 130.0, 456.0),
            _atom("r1-channel", "网上银行", 490.0, 130.0, 522.0),
            _atom("r1-summary", "转账", 550.0, 130.0, 566.0),
            _atom("r2-seq", "2", 25.0, 150.0, 29.0),
            _atom("r2-date", "2022-08-06", 60.0, 150.0, 100.0),
            _atom("r2-time", "16:14:05", 110.0, 150.0, 142.0),
            _atom("r2-type", "网上支付", 160.0, 150.0, 192.0),
            _atom("r2-direction", "借 Dr", 210.0, 150.0, 230.0),
            _atom("r2-amount", "37.98", 250.0, 150.0, 272.0),
            _atom("r2-balance", "3.08", 320.0, 150.0, 338.0),
            _atom("r2-account", "301440373999502", 370.0, 150.0, 426.0),
            _atom("r2-party", "江苏欧飞电子商务有限公司", 430.0, 150.0, 482.0),
            _atom("r2-channel", "网上支付", 490.0, 150.0, 522.0),
            _atom("r2-summary", "消费", 550.0, 150.0, 566.0),
        ]
    )
    result = _result(atoms)

    tables = recover_evidence_atom_bank_tables(result)

    assert recovered_evidence_atom_expected_row_count(result) == 2
    assert len(tables) == 1
    assert tables[0][1][0:7] == ["1", "2022-08-05", "14:05:18", "跨行汇款", "贷 Cr", "40.00", "41.06"]
    assert tables[0][2][0:7] == ["2", "2022-08-06", "16:14:05", "网上支付", "借 Dr", "37.98", "3.08"]
    assert [source["source_page"] for source in recovered_evidence_atom_row_sources(result)] == [1, 1]


def test_positioned_recovery_uses_evidence_atoms_for_pages_without_text_blocks():
    page_one_blocks = [
        SimpleNamespace(
            content=(
                f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
                f"6222020202020{sequence:03d}/甲公司"
            ),
            bbox=[20.0, 20.0 + sequence * 20.0, 60.0, 30.0 + sequence * 20.0],
            evidence_ids=[],
        )
        for sequence in range(1, 4)
    ]
    page_two_atoms = []
    for sequence in range(4, 6):
        atom = _atom(
            f"row-{sequence}",
            f"{sequence}\n转账支取\n2025010{sequence}\n-10.00\n{100 - sequence * 10:.2f}\n"
            f"6222020202020{sequence:03d}/甲公司",
            20.0,
            20.0 + sequence * 20.0,
            60.0,
        )
        atom["page_id"] = "page:0002"
        page_two_atoms.append(atom)
    result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={"text_atoms": page_two_atoms}),
        pages=[
            SimpleNamespace(page_number=1, width=600, height=850, tables=[], texts=page_one_blocks),
            SimpleNamespace(page_number=2, width=600, height=850, tables=[], texts=[]),
        ],
        logical_tables=[],
        entities=SimpleNamespace(domain_specific={}),
    )

    recovery = recover_positioned_record_block_bank_tables(result)

    assert recovery.expected_rows == 5
    assert [source["source_page"] for source in recovery.row_sources] == [1, 1, 1, 2, 2]


def test_recovers_column_aggregate_table_from_positioned_record_spines():
    atoms = [
        _atom(
            "header",
            "\u5e8f\u53f7\n\u6458\u8981\n\u5e01\u522b\n\u949e\u6c47\n\u4ea4\u6613\u65e5\u671f",
            20.0,
            20.0,
            32.0,
        ),
        _atom("row-1", "1\n\u94f6\u8054\u5165\u8d26\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025", 50.0, 20.0, 62.0),
        _atom("row-2", "2\n\u8f6c\u8d26\u652f\u53d6\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025", 70.0, 20.0, 82.0),
        _atom(
            "row-3",
            "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025",
            90.0,
            20.0,
            102.0,
        ),
    ]
    result = _result(atoms)
    result.pages[0].tables = [
        SimpleNamespace(
            table_id="aggregate",
            confidence=1.0,
            row_count=1,
            headers=[
                "\u5e8f\u53f7",
                "\u6458\u8981",
                "\u4ea4\u6613\u65e5\u671f",
                "\u4ea4\u6613\u91d1\u989d",
                "\u8d26\u6237\u4f59\u989d",
                "\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
            ],
            rows=[
                SimpleNamespace(
                    source_page=1,
                    cells=[
                        SimpleNamespace(text=value)
                        for value in [
                            "1\n2\n3",
                            "\u94f6\u8054\u5165\u8d26\n\u8f6c\u8d26\u652f\u53d6\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0",
                            "20211025\n20211025\n20211025",
                            "100.00\n-20.00\n10.00",
                            "100.00\n80.00\n90.00",
                            "1234567890/\u7532\u516c\u53f8\n1234567891/\u4e59\u516c\u53f8\n6217002120017593862/\u4e19\u516c\u53f8",
                        ]
                    ],
                )
            ],
        )
    ]

    recovery = recover_positioned_record_block_bank_tables(result)

    assert recovery.expected_rows == 3
    assert recovery.tables[0][1][3:6] == ["", "100.00", "100.00"]
    assert recovery.tables[0][2][3:6] == ["20.00", "", "80.00"]
    assert recovery.tables[0][3][3:6] == ["", "10.00", "90.00"]
    assert all(source["page_range"] == [1, 1] for source in recovery.row_sources)
    assert recovery.row_sources[0]["source_raw"] == {
        "序号": "1",
        "摘要": "银联入账",
        "交易日期": "20211025",
        "交易金额": "100.00",
        "账户余额": "100.00",
        "对方账号与户名": "1234567890/甲公司",
    }


def test_column_aggregate_source_raw_recovers_counterparty_after_blank_cell():
    row_atoms = [
        _atom("date", "20220128", 0.0, 10.0, 40.0),
        _atom("amount", "-4,515.48", 50.0, 10.0, 85.0),
        _atom("balance", "1,172.52", 95.0, 10.0, 130.0),
        _atom("counterparty", "6214921500056813/黄说英", 160.0, 10.0, 270.0),
    ]
    raw = _column_aggregate_source_raw(
        None,
        "page:0001",
        [{"atom": _atom("spine", "1\n摘要\n20220128", 0.0, 10.0, 20.0)}],
        0,
        ["交易日期", "交易金额", "账户余额", "对方账号与户名"],
        {"date": ["20220128"], "amount": ["-4,515.48"], "balance": ["1,172.52"]},
        row_atoms=row_atoms,
        column_axis=0,
    )

    assert raw["对方账号与户名"] == "6214921500056813/黄说英"


def test_positioned_record_direction_uses_continuous_previous_page_balance():
    previous = {"sequence_no": 285, "amount": "3021.00", "balance": "25733.42"}
    records = [
        {
            "sequence_no": 286,
            "amount": "3826.07",
            "balance": "29559.49",
            "direction": "",
        }
    ]

    _infer_positioned_block_directions(records, preceding_record=previous)

    assert records[0]["direction"] == "income"


def test_positioned_record_direction_rejects_non_contiguous_balance_pair():
    records = [
        {"sequence_no": 1, "amount": "10.00", "balance": "100.00", "direction": ""},
        {"sequence_no": 3, "amount": "10.00", "balance": "110.00", "direction": ""},
    ]

    _infer_positioned_block_directions(records)

    assert records[1]["direction"] == ""


def test_positioned_counter_account_prefers_account_joined_to_party_name():
    text = "\n".join(
        [
            "1",
            "转账",
            "20250101",
            "100.00",
            "900.00",
            "123456789012",
            "6222020202020202/甲公司",
        ]
    )

    assert _positioned_block_counter_account(text) == "6222020202020202"


def test_positioned_counter_account_preserves_masked_source_value():
    text = "\n".join(["1", "转账", "20250101", "-10.00", "90.00", "6230****6516/甲公司"])

    assert _positioned_block_counter_account(text) == "6230****6516"


def test_positioned_record_sort_preserves_descending_source_order():
    records = [
        {"sequence_no": 3, "atom": _atom("r3", "3", 10.0, 10.0)},
        {"sequence_no": 2, "atom": _atom("r2", "2", 10.0, 30.0)},
        {"sequence_no": 1, "atom": _atom("r1", "1", 10.0, 50.0)},
    ]

    _sort_positioned_block_records(records)

    assert [record["sequence_no"] for record in records] == [3, 2, 1]


def test_registry_selects_column_aggregate_recovery_when_physical_table_is_collapsed():
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
    from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry

    atoms = [
        _atom("header", "\u5e8f\u53f7\n\u6458\u8981\n\u4ea4\u6613\u65e5\u671f", 20.0, 20.0, 32.0),
        _atom("row-1", "1\n\u94f6\u8054\u5165\u8d26\n20211025", 50.0, 20.0, 62.0),
        _atom("row-2", "2\n\u8f6c\u8d26\u652f\u53d6\n20211025", 70.0, 20.0, 82.0),
        _atom("row-3", "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n20211025", 90.0, 20.0, 102.0),
    ]
    result = _result(atoms)
    result.pages[0].tables = [
        TableBlock(
            table_id="aggregate-registry",
            confidence=1.0,
            headers=[
                "\u5e8f\u53f7",
                "\u6458\u8981",
                "\u4ea4\u6613\u65e5\u671f",
                "\u4ea4\u6613\u91d1\u989d",
                "\u8d26\u6237\u4f59\u989d",
            ],
            rows=[
                TableRow(
                    source_page=1,
                    cells=[
                        CellValue(text=value)
                        for value in [
                            "1\n2\n3",
                            "\u94f6\u8054\u5165\u8d26\n\u8f6c\u8d26\u652f\u53d6\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0",
                            "20211025\n20211025\n20211025",
                            "100.00\n-20.00\n10.00",
                            "100.00\n80.00\n90.00",
                        ]
                    ],
                )
            ],
        )
    ]
    ctx = StyleContext(
        tables=[],
        full_text="\u4e2a\u4eba\u8d26\u6237\u4ea4\u6613\u660e\u7ec6",
        institution=None,
        page_count=1,
        parse_result=result,
    )
    registry = BankStyleParserRegistry()

    records, _identity = registry.run(BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin())

    assert len(records) == 3
    assert registry.last_selection_diagnostics["selected_candidate"] == "positioned_record_block"
    assert all(record["source"]["page_range"] == [1, 1] for record in records)


def test_registry_prefers_positioned_record_blocks_over_collapsed_physical_row():
    from docmirror.plugins.bank_statement.context import StyleContext
    from docmirror.plugins.bank_statement.style_detector import BankStyleDetector
    from docmirror.plugins.bank_statement.style_registry import BankStyleParserRegistry

    atoms = [
        _atom(
            "header",
            "\u5e8f\u53f7\n\u6458\u8981\n\u5e01\u522b\n\u949e\u6c47\n\u4ea4\u6613\u65e5\u671f\n"
            "\u4ea4\u6613\u91d1\u989d\n\u8d26\u6237\u4f59\u989d\n\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
            20.0,
            20.0,
            32.0,
        ),
        _atom(
            "row-1",
            "1\n\u94f6\u8054\u5165\u8d26\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n100.00\n100.00\n"
            "\u5546\u6237\n1234567890/\u7532\u516c\u53f8",
            50.0,
            20.0,
            62.0,
        ),
        _atom(
            "row-2",
            "2\n\u8f6c\u8d26\u652f\u53d6\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n-20.00\n80.00\n"
            "\u5546\u6237\n1234567891/\u4e59\u516c\u53f8",
            70.0,
            20.0,
            82.0,
        ),
        _atom(
            "row-3",
            "3\n\u652f\u4ed8\u673a\u6784\u63d0\u73b0\n\u4eba\u6c11\u5e01\u5143\n\u949e\n20211025\n10.00\n90.00\n"
            "\u5546\u6237\n1234567892/\u4e19\u516c\u53f8",
            90.0,
            20.0,
            102.0,
        ),
    ]
    parse_result = _result(atoms)
    collapsed = [
        [
            "\u5e8f\u53f7",
            "\u6458\u8981",
            "\u5e01\u522b",
            "\u949e\u6c47",
            "\u4ea4\u6613\u65e5\u671f",
            "\u4ea4\u6613\u91d1\u989d",
            "\u8d26\u6237\u4f59\u989d",
            "\u5bf9\u65b9\u8d26\u53f7\u4e0e\u6237\u540d",
        ],
        [
            "123",
            "\u94f6\u8054\u5165\u8d26\u8f6c\u8d26\u652f\u53d6\u652f\u4ed8\u673a\u6784\u63d0\u73b0",
            "\u4eba\u6c11\u5e01\u5143" * 3,
            "\u949e" * 3,
            "20211025" * 3,
            "100.00-20.0010.00",
            "100.0080.0090.00",
            "1234567890/\u7532\u516c\u53f8",
        ],
    ]
    ctx = StyleContext(
        tables=[collapsed],
        full_text="\u4e2a\u4eba\u8d26\u6237\u4ea4\u6613\u660e\u7ec6",
        institution=None,
        page_count=1,
        parse_result=parse_result,
    )
    registry = BankStyleParserRegistry()
    records, _ = registry.run(BankStyleDetector().detect(ctx), ctx, BankStatementCommunityPlugin())

    assert len(records) == 3
    assert registry.last_selection_diagnostics["selected_candidate"] == "positioned_record_block"
    assert all(record["source"]["page_range"] == [1, 1] for record in records)


def test_recovers_borderless_date_anchored_split_columns():
    atoms = [
        _atom("ht", "交易时间", 22.0, 80.0, 50.0),
        _atom("hs", "摘要", 61.0, 80.0, 75.0),
        _atom("hd", "借方发生额", 311.0, 80.0, 346.0),
        _atom("hc", "贷方发生额", 385.0, 80.0, 420.0),
        _atom("hb", "账户余额流水号", 466.0, 80.0, 519.0),
        _atom("d1", "2025/01/03", 22.0, 110.0, 57.0),
        _atom("t1", "16:18:35", 22.0, 119.0, 50.0),
        _atom("s1", "个人所得税", 61.0, 110.0, 100.0),
        _atom("e1", "15.00", 329.0, 110.0, 346.0),
        _atom("c1", "0.00", 403.0, 110.0, 420.0),
        _atom("b1", "363,693.02", 459.0, 110.0, 494.0),
        _atom("r1", "5542025010300824", 466.0, 119.0, 530.0),
        _atom("d2", "2025/01/07", 22.0, 140.0, 57.0),
        _atom("s2", "社保费", 61.0, 140.0, 90.0),
        _atom("e2", "113.16", 329.0, 140.0, 346.0),
        _atom("c2", "0.00", 403.0, 140.0, 420.0),
        _atom("b2", "363,579.86", 459.0, 140.0, 494.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][0] == ["交易时间", "摘要", "借方发生额", "贷方发生额", "账户余额流水号"]
    assert len(tables[0]) == 3
    assert tables[0][1][0] == "2025/01/0316:18:35"
    assert tables[0][1][4].startswith("363,693.02")


def test_recovers_parenthetical_split_columns_and_glued_balance_summary_header():
    atoms = [
        _atom("hd", "交易日期", 17.0, 106.0, 53.0),
        _atom("hdebit", "借方(出账)", 107.0, 106.0, 152.0),
        _atom("hcredit", "贷方(入账)", 193.0, 106.0, 238.0),
        _atom("hbalance_summary", "余额摘要", 305.0, 106.0, 344.0),
        _atom("hparty", "收(付)方名称", 394.0, 106.0, 449.0),
        _atom("haccount", "收(付)方账号", 463.0, 106.0, 517.0),
        _atom("htype", "交易类型", 531.0, 106.0, 568.0),
        _atom("d1", "2025-01-02", 17.0, 124.0, 62.0),
        _atom("debit1", "25.00", 129.0, 129.0, 152.0),
        _atom("balance1", "5,000,888.02", 269.0, 129.0, 324.0),
        _atom("summary1", "服务费", 326.0, 124.0, 370.0),
        _atom("party1", "测试有限公司", 394.0, 124.0, 449.0),
        _atom("account1", "123917394110001", 463.0, 124.0, 527.0),
        _atom("type1", "对公转账", 531.0, 124.0, 568.0),
        _atom("d2", "2025-01-03", 17.0, 156.0, 62.0),
        _atom("credit2", "200.00", 210.0, 161.0, 238.0),
        _atom("balance2", "5,001,088.02", 269.0, 161.0, 324.0),
        _atom("summary2", "往来款", 326.0, 156.0, 370.0),
        _atom("party2", "第二有限公司", 394.0, 156.0, 449.0),
        _atom("account2", "123917394110002", 463.0, 156.0, 527.0),
        _atom("type2", "提回收款", 531.0, 156.0, 568.0),
    ]
    vector_atoms = [
        {"id": f"rule-{index}", "page_id": "page:0001", "bbox": [17.0, y, 568.0, y]}
        for index, y in enumerate((116.0, 149.0, 182.0), start=1)
    ]
    parse_result = _result(atoms, vector_atoms)

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert tables[0][0] == [
        "交易日期",
        "借方(出账)",
        "贷方(入账)",
        "余额",
        "摘要",
        "收(付)方名称",
        "收(付)方账号",
        "交易类型",
    ]
    assert tables[0][1] == [
        "2025-01-02",
        "25.00",
        "",
        "5,000,888.02",
        "服务费",
        "测试有限公司",
        "123917394110001",
        "对公转账",
    ]
    assert tables[0][2][2:6] == ["200.00", "5,001,088.02", "往来款", "第二有限公司"]
    assert recovered_evidence_atom_expected_row_count(parse_result) == 2
    assert [source["source_page"] for source in sources] == [1, 1]


def test_recovers_borderless_embedded_direction_amount_rows():
    atoms = [
        _atom("hd", "交易日期", 36.0, 80.0, 72.0),
        _atom("hbd", "记账日期", 87.0, 80.0, 123.0),
        _atom("hs", "摘要", 138.0, 80.0, 156.0),
        _atom("ha", "支/收交易金额", 178.0, 80.0, 238.0),
        _atom("hb", "账户余额", 246.0, 80.0, 282.0),
        _atom("d1", "2023-10-02", 36.0, 110.0, 81.0),
        _atom("bd1", "2023-10-02", 87.0, 110.0, 132.0),
        _atom("s1", "跨行代付收", 138.0, 110.0, 187.0),
        _atom("a1", "23,903.69", 202.0, 110.0, 241.0),
        _atom("b1", "23,903.69", 246.0, 110.0, 286.0),
        _atom("d2", "2023-10-07", 36.0, 140.0, 81.0),
        _atom("bd2", "2023-10-07", 87.0, 140.0, 132.0),
        _atom("s2", "跨行代付收", 138.0, 140.0, 187.0),
        _atom("a2", "13,610.09", 202.0, 140.0, 241.0),
        _atom("b2", "13,610.09", 246.0, 140.0, 286.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][1][2] == "跨行代付"
    assert tables[0][1][3] == "收"
    assert tables[0][1][4] == "23,903.69"


def test_recovers_staggered_multilingual_borderless_header():
    atoms = [
        _atom("hd", "交易日期", 75.0, 145.7, 107.0),
        _atom("ht", "交易时间", 120.0, 145.7, 152.0),
        _atom("hy", "交易类型", 165.0, 145.7, 197.0),
        _atom("hf", "借贷", 220.0, 145.7, 236.0),
        _atom("ha", "交易金额", 258.0, 145.7, 290.0),
        _atom("hb", "余额", 337.0, 145.7, 353.0),
        _atom("hp", "交易地点", 583.0, 145.7, 615.0),
        _atom("hm", "摘要", 667.0, 145.7, 683.0),
        _atom("hs", "序号", 51.0, 150.2, 67.0),
        _atom("hc", "对方账号", 411.0, 150.2, 443.0),
        _atom("hn", "对方户名", 496.0, 150.2, 528.0),
        _atom("s1", "1", 56.5, 164.6, 60.5),
        _atom("d1", "2022-08-05", 75.0, 164.6, 115.0),
        _atom("t1", "14:05:18", 120.0, 164.6, 152.0),
        _atom("y1", "跨行汇款", 165.0, 164.6, 197.0),
        _atom("f1", "贷", 220.0, 164.6, 228.0),
        _atom("a1", "40.00", 253.0, 164.6, 273.0),
        _atom("b1", "41.06", 332.0, 164.6, 352.0),
        _atom("c1", "6214857212810271", 412.0, 164.6, 476.0),
        _atom("n1", "周深", 497.0, 164.6, 513.0),
        _atom("p1", "网上银行", 581.0, 164.6, 613.0),
        _atom("m1", "转账", 667.0, 164.6, 683.0),
        _atom("s2", "2", 56.5, 181.1, 60.5),
        _atom("d2", "2022-08-06", 75.0, 181.1, 115.0),
        _atom("t2", "16:14:05", 120.0, 181.1, 152.0),
        _atom("y2", "网上支付", 165.0, 181.1, 197.0),
        _atom("f2", "借", 220.0, 181.1, 228.0),
        _atom("a2", "37.98", 253.0, 181.1, 273.0),
        _atom("b2", "3.08", 332.0, 181.1, 348.0),
        _atom("c2", "301440373999502", 412.0, 181.1, 472.0),
        _atom("n2", "测试公司", 497.0, 181.1, 529.0),
        _atom("done", "打印完毕", 412.0, 195.6, 444.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][1][:7] == [
        "1",
        "2022-08-05",
        "14:05:18",
        "跨行汇款",
        "贷",
        "40.00",
        "41.06",
    ]
    assert tables[0][2][0:2] == ["2", "2022-08-06"]


def test_recovers_mixed_page_orientations_and_preserves_source_geometry():
    def page_atoms(page_id: str, sequence: str, date: str, amount: str, balance: str, y: float) -> list[dict]:
        values = [
            ("序号", 20.0, 50.0),
            ("交易日期", 60.0, 110.0),
            ("收入/支出", 120.0, 180.0),
            ("交易金额", 190.0, 250.0),
            ("账户余额", 260.0, 320.0),
            ("对方账号", 330.0, 410.0),
            ("对方户名", 420.0, 500.0),
            ("摘要", 510.0, 560.0),
        ]
        atoms = [
            {
                "id": f"{page_id}:h:{index}",
                "page_id": page_id,
                "text": text,
                "bbox": [x0, 80.0, x1, 90.0],
            }
            for index, (text, x0, x1) in enumerate(values)
        ]
        row = [
            (sequence, 20.0, 50.0),
            (date, 60.0, 110.0),
            ("支出", 120.0, 180.0),
            (amount, 190.0, 250.0),
            (balance, 260.0, 320.0),
            ("1000050001", 330.0, 410.0),
            ("测试对手", 420.0, 500.0),
            ("转账", 510.0, 560.0),
        ]
        atoms.extend(
            {
                "id": f"{page_id}:r:{index}",
                "page_id": page_id,
                "text": text,
                "bbox": [x0, y, x1, y + 10.0],
            }
            for index, (text, x0, x1) in enumerate(row)
        )
        atoms.append(
            {
                "id": f"{page_id}:footer",
                "page_id": page_id,
                "text": "本页合计及打印信息",
                "bbox": [20.0, y + 25.0, 300.0, y + 35.0],
            }
        )
        return atoms

    page_one = page_atoms("page:0001", "1", "2023-01-01", "10.00", "90.00", 110.0)
    page_two_horizontal = page_atoms("page:0002", "2", "2023-01-02", "20.00", "70.00", 110.0)
    page_two = [_rotated_90(atom, page_id="page:0002") for atom in page_two_horizontal]
    parse_result = _result([*page_one, *page_two])

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert sum(len(table) - 1 for table in tables) == 2
    assert recovered_evidence_atom_expected_row_count(parse_result) == 2
    assert [source["source_page"] for source in sources] == [1, 2]
    assert all(source.get("bbox") and source.get("evidence_ids") for source in sources)
    assert sources[1]["bbox"][0] > 400.0
    assert all("本页合计" not in "".join(row) for table in tables for row in table[1:])


def test_uses_positioned_page_text_when_evidence_atoms_are_not_promoted():
    atoms = [
        _atom("h1", "序号", 20.0, 80.0, 50.0),
        _atom("h2", "交易日期", 60.0, 80.0, 110.0),
        _atom("h3", "收入/支出", 120.0, 80.0, 180.0),
        _atom("h4", "交易金额", 190.0, 80.0, 250.0),
        _atom("h5", "账户余额", 260.0, 80.0, 320.0),
        _atom("r1", "1", 20.0, 110.0, 50.0),
        _atom("r2", "2023-01-01", 60.0, 110.0, 110.0),
        _atom("r3", "支出", 120.0, 110.0, 180.0),
        _atom("r4", "10.00", 190.0, 110.0, 250.0),
        _atom("r5", "90.00", 260.0, 110.0, 320.0),
    ]
    page = SimpleNamespace(
        page_number=1,
        width=600,
        height=850,
        texts=[
            SimpleNamespace(
                content=atom["text"],
                bbox=atom["bbox"],
                evidence_ids=[atom["id"]],
            )
            for atom in atoms
        ],
    )
    parse_result = SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={}),
        pages=[page],
        entities=SimpleNamespace(domain_specific={}),
    )

    tables = recover_evidence_atom_bank_tables(parse_result)
    sources = recovered_evidence_atom_row_sources(parse_result)

    assert sum(len(table) - 1 for table in tables) == 1
    assert recovered_evidence_atom_expected_row_count(parse_result) == 1
    assert sources[0]["source_page"] == 1
    assert "r2" in sources[0]["evidence_ids"]


def test_dedupe_does_not_treat_unproven_repeated_reference_as_row_identity():
    base = {"normalized": {"date": "2026-01-02", "amount": 100.0, "balance": 200.0, "counter_party": "甲"}}
    records = [
        {**base, "raw": {"交易流水号": "REF001"}},
        {**base, "raw": {"交易流水号": "REF002"}},
        {**base, "raw": {"交易流水号": "REF002"}},
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 3


def test_dedupe_preserves_distinct_rows_that_share_a_bank_reference():
    records = [
        {
            "normalized": {
                "date": "2026-01-02",
                "direction": "expense",
                "amount": 100.0,
                "balance": 900.0,
                "counter_party": "甲",
                "summary": "付款",
            },
            "raw": {"交易流水号": "REF001"},
        },
        {
            "normalized": {
                "date": "2026-01-02",
                "direction": "expense",
                "amount": 0.9,
                "balance": 899.1,
                "counter_party": "",
                "summary": "手续费",
            },
            "raw": {"交易流水号": "REF001"},
        },
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 2


def test_dedupe_does_not_collapse_unproven_repeated_sequence_rows():
    base = {"date": "2026-01-02", "amount": 100.0, "balance": 200.0, "counter_party": "same"}
    records = [
        {"normalized": {**base, "sequence_no": "491"}, "raw": {}},
        {"normalized": {**base, "sequence_no": "638"}, "raw": {}},
        {"normalized": {**base, "sequence_no": "638"}, "raw": {}},
    ]

    deduped = dedupe_transaction_rows(records)

    assert [record["normalized"]["sequence_no"] for record in deduped] == ["491", "638", "638"]


def test_recovers_bank_header_title_and_total_row_count_from_evidence_atoms():
    atoms = [
        _atom("title", "测试银行账户交易明细表", 200.0, 10.0, 400.0),
        _atom("bank_label", "开户行", 10.0, 20.0, 50.0),
        _atom("bank_value", "浦发银行重庆分行营业部", 80.0, 18.0, 220.0),
        _atom("print", "打印日期：2026-07-18", 10.0, 30.0, 150.0),
        _atom("period", "交易时段：2026-01-01 至 2026-06-30", 10.0, 45.0, 260.0),
        _atom("holder", "户名：测试用户", 10.0, 60.0, 100.0),
        _atom("account", "账号：1234567890", 110.0, 60.0, 230.0),
        _atom("currency", "币种：人民币", 240.0, 60.0, 320.0),
        _atom("total_label", "汇总交易笔数", 10.0, 220.0, 80.0),
        _atom("total_value", "38笔", 110.0, 225.0, 140.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["statement_title"]["normalized_value"] == "测试银行账户交易明细表"
    assert fields["print_date"]["normalized_value"] == "2026-07-18"
    assert fields["query_period"]["normalized_value"] == "2026-01-01 至 2026-06-30"
    assert fields["total_transactions"]["normalized_value"] == "38"
    assert fields["account_number"]["normalized_value"] == "1234567890"
    assert fields["branch_name"]["normalized_value"] == "浦发银行重庆分行营业部"
    assert "bank_name" not in fields


def test_evidence_identity_requires_an_explicit_issuer_label_for_bank_name():
    atoms = [
        _atom("bank_label", "银行名称", 10.0, 20.0, 70.0),
        _atom("bank_value", "测试银行", 80.0, 18.0, 150.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["bank_name"]["normalized_value"] == "测试银行"
    assert fields["bank_name"]["raw_name"] == "银行名称"
    assert fields["bank_name"]["source_refs"][0]["source"] == "canonical_evidence_atoms"


def test_evidence_identity_ignores_native_atoms_rejected_by_ocr_fallback():
    atoms = [
        _atom("holder", "户名：上上上上上上", 10.0, 60.0, 120.0),
        _atom("account", "账号：1234567890", 130.0, 60.0, 240.0),
    ]
    for atom in atoms:
        atom["source_kind"] = "pdf_native"
    result = _result(atoms)
    result.parser_info = SimpleNamespace(options={"native_text_ocr_fallback_pages": [1]})

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(result)

    assert fields == {}


def test_evidence_identity_stays_within_selected_source_pages():
    atoms = [
        _atom("selected_holder", "户名：测试科技有限公司 账号：1234567890", 10.0, 60.0, 240.0),
        {
            **_atom("unselected_holder", "户名：错误公司 验证码：", 10.0, 60.0, 240.0),
            "page_id": "page:0002",
        },
    ]
    result = _result(atoms)
    result.evidence_plane.pages = [
        SimpleNamespace(page_id="page:0001", page_number=1),
        SimpleNamespace(page_id="page:0002", page_number=2),
    ]
    result.parser_info = SimpleNamespace(options={"selected_source_pages": [1]})

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(result)

    assert fields["account_holder"]["normalized_value"] == "测试科技有限公司"


def test_evidence_identity_stops_holder_before_account_card_label():
    atoms = [
        _atom("holder", "户名：吴文坤", 10.0, 60.0, 90.0),
        _atom("account_label", "账号/卡号：", 100.0, 60.0, 165.0),
        _atom("account", "6230361108033553943", 175.0, 60.0, 310.0),
        _atom("currency", "币种：人民币", 320.0, 60.0, 400.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_holder"]["normalized_value"] == "吴文坤"
    assert fields["account_number"]["normalized_value"] == "6230361108033553943"


def test_evidence_identity_ignores_account_reference_below_transaction_header() -> None:
    atoms = [
        _atom("hd", "交易日期", 17.0, 106.0, 53.0),
        _atom("hdebit", "借方(出账)", 107.0, 106.0, 152.0),
        _atom("hcredit", "贷方(入账)", 193.0, 106.0, 238.0),
        _atom("hbalance_summary", "余额摘要", 305.0, 106.0, 344.0),
        _atom("hparty", "收(付)方名称", 394.0, 106.0, 449.0),
        _atom("haccount", "收(付)方账号", 463.0, 106.0, 517.0),
        _atom("htype", "交易类型", 531.0, 106.0, 568.0),
        _atom("date", "2025-03-21", 17.0, 130.0, 62.0),
        _atom("summary", "收息，结息账号:999019305110001", 326.0, 130.0, 450.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert "account_number" not in fields


def test_evidence_identity_recovers_split_header_values_and_directional_totals():
    atoms = [
        _atom("title", "交通银行某分行明细对账单", 180.0, 20.0, 390.0),
        _atom("account_label", "账号：", 20.0, 50.0, 55.0),
        _atom("account", "641301106013000859983", 60.0, 50.0, 180.0),
        _atom("holder_label", "户名：", 220.0, 50.0, 255.0),
        _atom("holder", "测试软件有限公司银川分公司", 260.0, 50.0, 430.0),
        _atom("year_label", "年份：", 20.0, 65.0, 55.0),
        _atom("year", "2025", 60.0, 65.0, 90.0),
        _atom("month_label", "月份：", 120.0, 65.0, 155.0),
        _atom("month", "07", 160.0, 65.0, 180.0),
        _atom("currency_label", "币种：", 220.0, 65.0, 255.0),
        _atom("currency", "人民币", 260.0, 65.0, 300.0),
        _atom("carry", "承前", 20.0, 100.0, 50.0),
        _atom("debit_label", "当前账单借方发生数：", 20.0, 220.0, 145.0),
        _atom("debit", "10", 150.0, 220.0, 170.0),
        _atom("credit", "当前账单贷方发生数：1", 300.0, 220.0, 450.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_holder"]["normalized_value"] == "测试软件有限公司银川分公司"
    assert fields["account_number"]["normalized_value"] == "641301106013000859983"
    assert fields["currency"]["raw_value"] == "人民币"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["query_period"]["normalized_value"] == "2025-07-01 至 2025-07-31"
    assert "total_transactions" not in fields


def test_evidence_identity_pairs_parallel_label_value_columns_by_geometry():
    atoms = [
        _atom("account_label", "银行账号：", 20.0, 50.0, 80.0),
        _atom("account", "120023710020000001988", 90.0, 50.0, 230.0),
        _atom("currency_label", "币种：", 430.0, 50.0, 470.0),
        _atom("currency", "人民币", 480.0, 50.0, 530.0),
        _atom("holder_label", "账户名称：", 20.0, 70.0, 80.0),
        _atom("holder", "测试信用管理有限公司", 90.0, 70.0, 250.0),
        _atom("deposit_label", "存款种类：", 430.0, 70.0, 490.0),
        _atom("deposit", "单位活期存款", 500.0, 70.0, 580.0),
        _atom("print_bank_label", "打印机构：", 350.0, 780.0, 420.0),
        _atom("print_bank", "富滇银行", 430.0, 780.0, 490.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_number"]["normalized_value"] == "120023710020000001988"
    assert fields["account_holder"]["normalized_value"] == "测试信用管理有限公司"
    assert fields["currency"]["raw_value"] == "人民币"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["branch_name"]["normalized_value"] == "富滇银行"
    assert "bank_name" not in fields


def test_evidence_identity_supports_hyphenated_account_and_chinese_date_range():
    atoms = [
        _atom("title", "账户明细", 250.0, 20.0, 340.0),
        _atom("account", "账号:31-080201040015288", 20.0, 45.0, 170.0),
        _atom("identity", "户名:测试农业科技有限公司币种:人民币", 190.0, 45.0, 430.0),
        _atom("period_label", "起止日期:", 450.0, 45.0, 510.0),
        _atom("period_start", "2025年11月01日", 520.0, 45.0, 610.0),
        _atom("period_sep", "-", 615.0, 45.0, 620.0),
        _atom("period_end", "2025年12月31日", 625.0, 45.0, 715.0),
        _atom("income_count", "总收入笔数", 20.0, 220.0, 90.0),
        _atom("income_value", "2", 95.0, 220.0, 105.0),
        _atom("expense_count", "总支出笔数", 220.0, 220.0, 290.0),
        _atom("expense_value", "2", 295.0, 220.0, 305.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_number"]["normalized_value"] == "31-080201040015288"
    assert fields["account_holder"]["normalized_value"] == "测试农业科技有限公司"
    assert fields["currency"]["normalized_value"] == "CNY"
    assert fields["query_period"]["normalized_value"] == "2025-11-01 至 2025-12-31"
    assert "total_transactions" not in fields


def test_evidence_identity_normalizes_compatibility_currency_without_changing_raw_value():
    atoms = [_atom("currency", "币种：⼈⺠币", 20.0, 45.0, 120.0)]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["currency"]["raw_value"] == "⼈⺠币"
    assert fields["currency"]["normalized_value"] == "CNY"


def test_geometry_recovery_keeps_wrapped_cells_with_preceding_date_and_stops_at_footer():
    atoms = [
        _atom("h1", "交易时间", 20.0, 80.0, 80.0),
        _atom("h2", "收入金额", 100.0, 80.0, 160.0),
        _atom("h3", "支出金额", 180.0, 80.0, 240.0),
        _atom("h4", "账户余额", 260.0, 80.0, 320.0),
        _atom("h5", "对方账号", 340.0, 80.0, 400.0),
        _atom("h6", "对方户名", 420.0, 80.0, 480.0),
        _atom("h7", "对方开户行", 500.0, 80.0, 575.0),
        _atom("h8", "摘要", 600.0, 80.0, 640.0),
        _atom("d1", "2025-11-24", 20.0, 110.0, 80.0),
        _atom("t1", "00:46:17", 20.0, 120.0, 70.0),
        _atom("e1", "100.00", 180.0, 110.0, 235.0),
        _atom("b1", "110.97", 260.0, 110.0, 315.0),
        _atom("a1", "31080243CNYFC0445", 340.0, 110.0, 400.0),
        _atom("p1", "测试银行股份有限公司", 420.0, 110.0, 480.0),
        _atom("bn1a", "测试银行股份有限公司", 500.0, 110.0, 575.0),
        _atom("bn1b", "重庆九龙坡二", 500.0, 125.0, 560.0),
        _atom("bn1c", "郎支行", 500.0, 145.0, 535.0),
        _atom("s1", "批量扣费", 600.0, 110.0, 640.0),
        _atom("d2", "2025-12-21", 20.0, 160.0, 80.0),
        _atom("t2", "01:10:43", 20.0, 170.0, 70.0),
        _atom("i2", "0.03", 100.0, 160.0, 150.0),
        _atom("b2", "111.00", 260.0, 160.0, 315.0),
        _atom("bn2", "第二银行", 500.0, 152.0, 550.0),
        _atom("s2", "批量结息", 600.0, 160.0, 640.0),
        _atom("income_count", "总收入笔数", 20.0, 205.0, 90.0),
        _atom("income_value", "1", 100.0, 205.0, 110.0),
        _atom("income_total", "总收入金额", 180.0, 205.0, 250.0),
        _atom("income_amount", "0.03", 260.0, 205.0, 300.0),
        _atom("print_date", "2026/02/24", 20.0, 390.0, 90.0),
        _atom("page", "第1页/共1页", 280.0, 400.0, 350.0),
    ]

    vector_atoms = [
        {
            "id": f"rule-{index}",
            "page_id": "page:0001",
            "bbox": [0.0, y, 650.0, y],
        }
        for index, y in enumerate((95.0, 150.0, 200.0), start=1)
    ]
    parse_result = _result(atoms, vector_atoms)
    tables = recover_evidence_atom_bank_tables(parse_result)

    assert len(tables) == 1
    assert len(tables[0]) == 3
    assert tables[0][1][6] == "测试银行股份有限公司重庆九龙坡二郎支行"
    assert tables[0][2][5:7] == ["", "第二银行"]
    assert all("总收入" not in "".join(row) for row in tables[0][1:])
    assert all("第1页" not in "".join(row) for row in tables[0][1:])
    assert recovered_evidence_atom_expected_row_count(parse_result) == 2


def test_geometry_recovery_assigns_leading_wrapped_cells_to_nearest_date_anchor():
    atoms = [
        _atom("h0", "序号", 10.0, 80.0, 30.0),
        _atom("h1", "交易日期", 40.0, 80.0, 90.0),
        _atom("h2", "交易时间", 100.0, 80.0, 150.0),
        _atom("h3", "交易类型", 160.0, 80.0, 210.0),
        _atom("h4", "借贷", 220.0, 80.0, 250.0),
        _atom("h5", "交易金额", 260.0, 80.0, 320.0),
        _atom("h6", "余额", 330.0, 80.0, 370.0),
        _atom("h7", "对方账号", 380.0, 80.0, 440.0),
        _atom("h8", "对方户名", 450.0, 80.0, 510.0),
        _atom("h9", "摘要", 520.0, 80.0, 560.0),
        _atom("s1", "1", 10.0, 110.0, 20.0),
        _atom("d1", "2022-08-05", 40.0, 110.0, 90.0),
        _atom("t1", "14:05:18", 100.0, 110.0, 150.0),
        _atom("type1", "跨行汇款", 160.0, 110.0, 210.0),
        _atom("dir1", "贷 Cr", 220.0, 110.0, 250.0),
        _atom("a1", "40.00", 260.0, 110.0, 320.0),
        _atom("b1", "41.06", 330.0, 110.0, 370.0),
        _atom("cp1", "周深", 450.0, 110.0, 510.0),
        _atom("s2", "2", 10.0, 140.0, 20.0),
        _atom("d2", "2022-08-06", 40.0, 140.0, 90.0),
        _atom("t2", "16:14:05", 100.0, 140.0, 150.0),
        _atom("type2", "网上支付", 160.0, 140.0, 210.0),
        _atom("dir2", "借 Dr", 220.0, 140.0, 250.0),
        _atom("a2", "37.98", 260.0, 140.0, 320.0),
        _atom("b2", "3.08", 330.0, 140.0, 370.0),
        _atom("cp2a", "江苏欧飞电子商务有限", 450.0, 126.0, 510.0),
        _atom("cp2b", "公司", 450.0, 145.0, 480.0),
        _atom("summary2", "有限公司", 520.0, 145.0, 560.0),
        _atom("footer", "打印完毕", 380.0, 165.0, 440.0),
    ]

    tables = recover_evidence_atom_bank_tables(_result(atoms))

    assert len(tables) == 1
    assert tables[0][1][4] == "贷 Cr"
    assert tables[0][1][8] == "周深"
    assert tables[0][2][4] == "借 Dr"
    assert tables[0][2][8] == "江苏欧飞电子商务有限公司"


def test_evidence_identity_stops_at_branch_and_ignores_transaction_loan_account():
    atoms = [
        _atom("title", "对公客户账户明细", 200.0, 10.0, 400.0),
        _atom("holder", "客户名称：重庆正大华日软件有限公司", 10.0, 40.0, 220.0),
        _atom("branch", "开户机构：510601", 230.0, 40.0, 340.0),
        _atom("account", "账    号：5106010120010001125", 10.0, 60.0, 230.0),
        _atom("loan", "贷款账号：5101010179730017689", 10.0, 160.0, 230.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["account_holder"]["normalized_value"] == "重庆正大华日软件有限公司"
    assert fields["account_number"]["normalized_value"] == "5106010120010001125"
