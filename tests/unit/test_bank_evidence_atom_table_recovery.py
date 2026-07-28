# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical evidence atom split debit/credit bank ledger recovery tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.bank_statement.canonical import dedupe_transaction_rows
from docmirror.plugins.bank_statement.community_plugin import BankStatementCommunityPlugin
from docmirror.plugins.bank_statement.evidence_atom_table_recovery import (
    recover_evidence_atom_bank_tables,
    recovered_evidence_atom_expected_row_count,
    recovered_evidence_atom_row_sources,
)

pytestmark = pytest.mark.unit


def _atom(atom_id: str, text: str, x0: float, y0: float, x1: float | None = None) -> dict:
    return {
        "id": atom_id,
        "page_id": "page:0001",
        "text": text,
        "bbox": [x0, y0, x1 if x1 is not None else x0 + 20.0, y0 + 8.0],
    }


def _result(atoms: list[dict]) -> SimpleNamespace:
    page_numbers = sorted({int(str(atom.get("page_id") or "page:0001").rsplit(":", 1)[-1]) for atom in atoms})
    return SimpleNamespace(
        evidence_plane=SimpleNamespace(evidence={"text_atoms": atoms}),
        pages=[SimpleNamespace(page_number=page, width=600, height=850) for page in page_numbers],
        entities=SimpleNamespace(domain_specific={}),
    )


def _rotated_90(atom: dict, *, page_id: str) -> dict:
    x0, y0, x1, y1 = atom["bbox"]
    return {
        **atom,
        "page_id": page_id,
        "bbox": [600.0 - y1, x0, 600.0 - y0, x1],
    }


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
    assert tables[0][1][2] == "跨行代付收"
    assert tables[0][1][3] == "23,903.69"


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


def test_dedupe_uses_bank_reference_before_lossy_business_fields():
    base = {"normalized": {"date": "2026-01-02", "amount": 100.0, "balance": 200.0, "counter_party": "甲"}}
    records = [
        {**base, "raw": {"交易流水号": "REF001"}},
        {**base, "raw": {"交易流水号": "REF002"}},
        {**base, "raw": {"交易流水号": "REF002"}},
    ]

    deduped = dedupe_transaction_rows(records)

    assert len(deduped) == 2


def test_dedupe_keeps_same_business_fields_when_sequence_differs():
    base = {"date": "2026-01-02", "amount": 100.0, "balance": 200.0, "counter_party": "same"}
    records = [
        {"normalized": {**base, "sequence_no": "491"}, "raw": {}},
        {"normalized": {**base, "sequence_no": "638"}, "raw": {}},
        {"normalized": {**base, "sequence_no": "638"}, "raw": {}},
    ]

    deduped = dedupe_transaction_rows(records)

    assert [record["normalized"]["sequence_no"] for record in deduped] == ["491", "638"]


def test_recovers_bank_header_title_and_total_row_count_from_evidence_atoms():
    atoms = [
        _atom("title", "测试银行账户交易明细表", 200.0, 10.0, 400.0),
        _atom("print", "打印日期：2026-07-18", 10.0, 30.0, 150.0),
        _atom("period", "交易时段：2026-01-01 至 2026-06-30", 10.0, 45.0, 260.0),
        _atom("holder", "户名：测试用户", 10.0, 60.0, 100.0),
        _atom("account", "账号：1234567890", 110.0, 60.0, 230.0),
        _atom("currency", "币种：人民币", 240.0, 60.0, 320.0),
        _atom("total", "总条数：38", 10.0, 220.0, 80.0),
    ]

    fields = BankStatementCommunityPlugin()._recover_identity_from_evidence(_result(atoms))

    assert fields["statement_title"]["normalized_value"] == "测试银行账户交易明细表"
    assert fields["print_date"]["normalized_value"] == "2026-07-18"
    assert fields["query_period"]["normalized_value"] == "2026-01-01 至 2026-06-30"
    assert fields["total_transactions"]["normalized_value"] == "38"
    assert fields["account_number"]["normalized_value"] == "1234567890"
