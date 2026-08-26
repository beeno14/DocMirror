from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.bank_statement import statement_context


def _atom(page: int, text: str, x0: float, y0: float, x1: float, y1: float, index: int) -> dict:
    return {
        "id": f"ev:{page:04d}:text:{index:06d}",
        "page_id": f"page:{page:04d}",
        "text": text,
        "bbox": [x0, y0, x1, y1],
        "source_kind": "pdf_native",
        "confidence": 1.0,
    }


def _current_account_atoms(*, total: str = "90", end: str = "2023-05-22") -> list[dict]:
    rows = [
        [("活期账户明细查询", 100, 20, 390, 50)],
        [("2023/08/24", 770, 70, 825, 84)],
        [
            ("开始日期:", 0, 100, 60, 112),
            ("2023-02-23", 70, 100, 135, 112),
            ("结束日期:", 200, 100, 260, 112),
            (end, 270, 100, 335, 112),
        ],
        [
            ("账", 0, 120, 12, 132),
            ("号:", 14, 120, 36, 132),
            ("3211020801201000170968", 45, 120, 190, 132),
            ("币", 210, 120, 222, 132),
            ("种:", 224, 120, 246, 132),
            ("人民币", 255, 120, 290, 132),
            ("户", 320, 120, 332, 132),
            ("名:", 334, 120, 356, 132),
            ("测试企业有限公司", 365, 120, 455, 132),
        ],
        [
            ("总笔数:", 0, 140, 48, 152),
            (total, 55, 140, 75, 152),
            ("借方总金额:", 100, 140, 175, 152),
            ("3,019,670.00", 185, 140, 260, 152),
            ("借方总笔数:", 290, 140, 365, 152),
            ("43", 375, 140, 390, 152),
        ],
        [
            ("总金额:", 0, 160, 48, 172),
            ("5,992,890.32", 55, 160, 130, 172),
            ("贷方总金额:", 160, 160, 235, 172),
            ("2,973,220.32", 245, 160, 320, 172),
            ("贷方总笔数:", 350, 160, 425, 172),
            ("47", 435, 160, 450, 172),
        ],
        [
            ("序号", 0, 200, 25, 212),
            ("交易日期", 40, 200, 90, 212),
            ("交易时间", 105, 200, 155, 212),
            ("收入", 170, 200, 195, 212),
            ("支出", 210, 200, 235, 212),
            ("余额", 250, 200, 275, 212),
            ("摘要", 290, 200, 315, 212),
        ],
        [
            ("1", 0, 225, 8, 237),
            ("2023-02-24", 40, 225, 105, 237),
            ("16:50", 115, 225, 150, 237),
            ("100.00", 170, 225, 210, 237),
            ("100.00", 250, 225, 290, 237),
        ],
    ]
    atoms: list[dict] = []
    for row in rows:
        for values in row:
            atoms.append(_atom(1, *values, len(atoms) + 1))
    return atoms


@pytest.fixture
def parse_result() -> SimpleNamespace:
    return SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[], texts=[])],
        parser_info=SimpleNamespace(options={}),
    )


def test_split_geometry_header_preserves_every_business_fact(monkeypatch, parse_result):
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: _current_account_atoms())
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    records = statement_context.build_statement_header_records(parse_result, {})

    assert len(records) == 1
    row = records[0]
    assert row["normalized"] == {
        "period_start": "2023-02-23",
        "period_end": "2023-05-22",
        "account_number": "3211020801201000170968",
        "currency": "CNY",
        "account_holder": "测试企业有限公司",
        "total_transactions": 90,
        "debit_total": "3019670.00",
        "debit_count": 43,
        "total_amount": "5992890.32",
        "credit_total": "2973220.32",
        "credit_count": 47,
        "statement_title": "活期账户明细查询",
        "document_date": "2023-08-24",
        "query_period": "2023-02-23 ~ 2023-05-22",
    }
    assert row["canonical_raw"]["currency"] == "人民币"
    assert row["raw"]["总笔数:"] == "90"
    assert row["raw"]["借方总金额:"] == "3,019,670.00"
    assert "bank_name" not in row["normalized"]
    assert row["source"]["page_range"] == [1, 1]
    assert row["source"]["field_sources"]["account_number"]["evidence_ids"]


def test_same_layout_values_are_not_fixture_constants(monkeypatch, parse_result):
    monkeypatch.setattr(
        statement_context,
        "text_atoms",
        lambda _result: _current_account_atoms(total="71", end="2023-08-22"),
    )
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["total_transactions"] == 71
    assert record["normalized"]["period_end"] == "2023-08-22"


def test_terminal_transaction_count_and_amount_footers_are_business_header_facts(monkeypatch):
    first_page_atoms = [
        atom for atom in _current_account_atoms() if not 140 <= float(atom["bbox"][1]) < 180
    ]
    second_page_atoms = [
        _atom(2, "交易时间", 20, 100, 70, 114, 1),
        _atom(2, "交易金额", 100, 100, 150, 114, 2),
        _atom(2, "余额", 180, 100, 210, 114, 3),
        _atom(2, "摘要", 240, 100, 275, 114, 15),
        _atom(2, "2023-05-22", 20, 130, 85, 144, 4),
        _atom(2, "10.00", 100, 130, 135, 144, 5),
        _atom(2, "110.00", 180, 130, 220, 144, 6),
        _atom(2, "测试交易", 240, 130, 300, 144, 16),
        _atom(2, "收入交易笔数:", 20, 500, 110, 514, 7),
        _atom(2, "31", 120, 500, 138, 514, 8),
        _atom(2, "收入金额合计:", 200, 500, 290, 514, 9),
        _atom(2, "12,345.67", 300, 500, 365, 514, 10),
        _atom(2, "支出交易笔数:", 20, 525, 110, 539, 11),
        _atom(2, "79", 120, 525, 138, 539, 12),
        _atom(2, "支出金额合计:", 200, 525, 290, 539, 13),
        _atom(2, "8,765.43", 300, 525, 365, 539, 14),
    ]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[], texts=[]),
        ],
        parser_info=SimpleNamespace(options={}),
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: [*first_page_atoms, *second_page_atoms])
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: "", 2: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["credit_count"] == 31
    assert record["normalized"]["credit_total"] == "12345.67"
    assert record["normalized"]["debit_count"] == 79
    assert record["normalized"]["debit_total"] == "8765.43"
    assert record["raw"]["收入交易笔数:"] == "31"
    assert record["source"]["field_sources"]["debit_total"]["source_refs"][0]["source_page"] == 2


def test_rotated_multi_field_header_band_is_not_mistaken_for_transaction_data(monkeypatch, parse_result):
    atoms = [
        _atom(1, "某银行账户明细清单", 80, 10, 260, 26, 1),
        _atom(1, "账号:", -195, 40, -165, 52, 2),
        _atom(1, "4402254019022147099", -160, 40, -70, 52, 3),
        _atom(1, "币种:", 10, 40, 40, 52, 4),
        _atom(1, "人民币", 45, 40, 85, 52, 5),
        _atom(1, "本方账号户名:", -195, 55, -145, 67, 6),
        _atom(1, "测试机电配件厂", -140, 55, -75, 67, 7),
        _atom(1, "本方账号开户行:", 80, 55, 135, 67, 8),
        _atom(1, "某银行测试支行", 140, 55, 200, 67, 9),
        _atom(1, "时间范围:", 295, 55, 330, 67, 10),
        _atom(1, "20230701", 335, 55, 365, 67, 11),
        _atom(1, "-", 370, 55, 374, 67, 12),
        _atom(1, "20230731", 380, 55, 410, 67, 13),
        _atom(1, "1", 0, 80, 8, 92, 14),
        _atom(1, "20230702", 40, 80, 95, 92, 15),
        _atom(1, "100.00", 120, 80, 165, 92, 16),
        _atom(1, "200.00", 190, 80, 235, 92, 17),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_holder"] == "测试机电配件厂"
    assert record["normalized"]["branch_name"] == "某银行测试支行"
    assert record["normalized"]["period_start"] == "2023-07-01"
    assert record["normalized"]["period_end"] == "2023-07-31"
    assert record["raw"]["本方账号户名:"] == "测试机电配件厂"


def test_counterparty_qualified_holder_label_never_becomes_own_identity(monkeypatch, parse_result):
    atoms = [
        _atom(1, "账户交易明细", 80, 10, 220, 26, 1),
        _atom(1, "账号:", 0, 40, 35, 52, 2),
        _atom(1, "4402254019022147099", 40, 40, 170, 52, 3),
        _atom(1, "对方账号户名:", 190, 40, 260, 52, 4),
        _atom(1, "交易对手企业", 265, 40, 330, 52, 5),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert "account_holder" not in record["normalized"]
    assert record["raw"]["对方账号户名"] == "交易对手企业"


@pytest.mark.parametrize("account_label", ["账⼾", "賬戶"])
def test_compatibility_account_label_is_an_explicit_own_account_number(
    monkeypatch,
    parse_result,
    account_label,
):
    atoms = [
        _atom(1, "账户交易明细", 80, 10, 220, 26, 1),
        _atom(1, f"{account_label}：4402254019022147099", 0, 40, 170, 52, 2),
        _atom(1, "1", 0, 80, 8, 92, 3),
        _atom(1, "2023-07-02", 40, 80, 105, 92, 4),
        _atom(1, "100.00", 120, 80, 165, 92, 5),
        _atom(1, "200.00", 190, 80, 235, 92, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_number"] == "4402254019022147099"
    assert record["canonical_raw"]["account_number"] == "4402254019022147099"
    assert record["source"]["field_sources"]["account_number"]["evidence_ids"] == [
        "ev:0001:text:000002"
    ]


def test_exact_card_account_label_is_a_source_backed_account_number(monkeypatch, parse_result):
    atoms = [
        _atom(1, "账户交易明细", 80, 10, 220, 26, 1),
        _atom(1, "卡/账号:", 0, 40, 55, 52, 2),
        _atom(1, "6230580000166957147", 60, 40, 185, 52, 3),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_number"] == "6230580000166957147"
    assert record["canonical_raw"]["account_number"] == "6230580000166957147"
    assert record["raw"]["卡/账号:"] == "6230580000166957147"
    assert record["source"]["field_sources"]["account_number"] == {
        "raw_name": "卡/账号:",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": [0.0, 40.0, 185.0, 52.0],
            }
        ],
        "evidence_ids": ["ev:0001:text:000002", "ev:0001:text:000003"],
    }


@pytest.mark.parametrize("value", ["12", "活期", "全部"])
def test_exact_card_account_label_retains_strict_account_validation(value):
    row = [_atom(1, f"卡/账号:{value}", 0, 40, 170, 52, 1)]

    facts = statement_context._facts_from_row(row, 1)

    assert all(fact.field_key != "account_number" for fact in facts)


@pytest.mark.parametrize("value", ["CURRENT", "活期", "全部"])
def test_bare_account_label_requires_an_account_like_value(value):
    row = [_atom(1, f"账户:{value}", 0, 40, 170, 52, 1)]

    facts = statement_context._facts_from_row(row, 1)

    assert all(fact.field_key != "account_number" for fact in facts)


def test_fragmented_bilingual_teller_label_does_not_contaminate_department(monkeypatch, parse_result):
    atoms = [
        _atom(1, "个人客户交易清单", 80, 10, 220, 26, 1),
        _atom(1, "部门Department:", 0, 40, 95, 52, 2),
        _atom(1, "01381200999", 100, 40, 170, 52, 3),
        _atom(1, "柜员Search", 190, 40, 240, 52, 4),
        _atom(1, "Teller:", 245, 40, 285, 52, 5),
        _atom(1, "ECT0001", 290, 40, 340, 52, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["department"] == "01381200999"
    assert record["normalized"]["query_teller"] == "ECT0001"


def test_exact_bilingual_query_teller_is_source_backed_and_in_community_schema(
    monkeypatch,
    parse_result,
):
    atoms = [
        _atom(1, "个人客户交易清单", 80, 10, 220, 26, 1),
        _atom(1, "查询柜员SearchTeller:", 0, 40, 125, 52, 2),
        _atom(1, "ECT0001", 130, 40, 180, 52, 3),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["query_teller"] == "ECT0001"
    assert record["canonical_raw"]["query_teller"] == "ECT0001"
    assert record["raw"]["查询柜员SearchTeller:"] == "ECT0001"
    assert record["source"]["field_sources"]["query_teller"] == {
        "raw_name": "查询柜员SearchTeller:",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": [0.0, 40.0, 180.0, 52.0],
            }
        ],
        "evidence_ids": ["ev:0001:text:000002", "ev:0001:text:000003"],
    }
    from docmirror.plugins.bank_statement.community_plugin import BANK_DATA_DICTIONARY

    assert BANK_DATA_DICTIONARY["fields"]["query_teller"] == {
        "label": "查询柜员",
        "type": "string",
    }
    assert BANK_DATA_DICTIONARY["datasets"]["statement_header"]["columns"]["query_teller"] == {
        "label": "查询柜员",
        "type": "string",
    }


def test_exact_bilingual_query_teller_below_ledger_header_is_not_context(
    monkeypatch,
    parse_result,
):
    atoms = [
        _atom(1, "个人客户交易清单", 80, 10, 220, 26, 1),
        _atom(1, "交易日期", 0, 80, 55, 92, 2),
        _atom(1, "交易金额", 70, 80, 125, 92, 3),
        _atom(1, "余额", 140, 80, 165, 92, 4),
        _atom(1, "查询柜员SearchTeller: ECT0001", 180, 105, 350, 117, 5),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert "query_teller" not in record["normalized"]
    assert "query_teller" not in record["canonical_raw"]


def test_table_only_source_does_not_invent_statement_identity(monkeypatch, parse_result):
    atoms = _current_account_atoms()
    atoms = [atom for atom in atoms if atom["bbox"][1] >= 200]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    assert statement_context.build_statement_header_records(parse_result, {}) == []


def test_wrapped_transaction_cells_below_ledger_header_cannot_enter_header_raw(monkeypatch, parse_result):
    atoms = [
        _atom(1, "账户交易明细", 80, 20, 220, 36, 1),
        _atom(1, "交易日期", 0, 100, 55, 112, 2),
        _atom(1, "交易金额", 70, 100, 125, 112, 3),
        _atom(1, "余额", 140, 100, 165, 112, 4),
        _atom(1, "摘要", 180, 100, 215, 112, 5),
        _atom(1, "目:对公", 240, 115, 285, 127, 6),
        _atom(1, "1", 0, 145, 8, 157, 7),
        _atom(1, "2023-07-02", 20, 145, 85, 157, 8),
        _atom(1, "100.00", 100, 145, 145, 157, 9),
        _atom(1, "200.00", 160, 145, 205, 157, 10),
        _atom(1, "收费项目:对公人民币转账、汇款", 220, 145, 390, 157, 11),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["statement_title"] == "账户交易明细"
    assert all("对公" not in str(value) for value in record["raw"].values())


def test_one_character_generic_label_fragment_fails_closed():
    row = [_atom(1, "目:对公", 240, 50, 285, 62, 1)]

    assert statement_context._facts_from_row(row, 1) == []


def test_page_reset_creates_distinct_statement_scopes(monkeypatch):
    first = _current_account_atoms(total="2", end="2023-02-28")
    second = [
        {
            **atom,
            "id": str(atom["id"]).replace("0001", "0002", 1),
            "page_id": "page:0002",
            "bbox": [atom["bbox"][0], atom["bbox"][1], atom["bbox"][2], atom["bbox"][3]],
            "text": (
                "2023-03-01"
                if atom["text"] == "2023-02-23"
                else "2023-03-31"
                if atom["text"] == "2023-02-28"
                else "3"
                if atom["text"] == "2"
                else atom["text"]
            ),
        }
        for atom in first
    ]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[], texts=[]),
        ],
        parser_info=SimpleNamespace(options={}),
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: [*first, *second])
    monkeypatch.setattr(
        statement_context,
        "_source_page_texts",
        lambda _result: {1: "第1页 共1页", 2: "第1页 共1页"},
    )

    records = statement_context.build_statement_header_records(parse_result, {})

    assert [record["source"]["page_range"] for record in records] == [[1, 1], [2, 2]]
    assert [record["normalized"]["period_start"] for record in records] == ["2023-02-23", "2023-03-01"]
    assert [record["normalized"]["total_transactions"] for record in records] == [2, 3]


def test_english_page_reset_keeps_source_headers_and_ledger_body_scope_local(monkeypatch):
    def statement_atoms(page: int, print_date: str) -> list[dict]:
        base = page * 100
        return [
            _atom(page, "Example Bank Account Statement", 120, 20, 360, 38, base + 1),
            _atom(page, "Page:", 20, 50, 55, 62, base + 2),
            _atom(page, "1/1", 60, 50, 82, 62, base + 3),
            _atom(page, "Account Number:", 120, 50, 220, 62, base + 4),
            _atom(page, "123456789012", 225, 50, 305, 62, base + 5),
            _atom(page, "Account Name:", 20, 70, 105, 82, base + 6),
            _atom(page, "Example Trading Ltd", 110, 70, 225, 82, base + 7),
            _atom(page, "Currency:", 250, 70, 310, 82, base + 8),
            _atom(page, "USD", 315, 70, 340, 82, base + 9),
            _atom(page, "Print Date:", 365, 70, 430, 82, base + 10),
            _atom(page, print_date, 435, 70, 500, 82, base + 11),
            _atom(page, "Statement Covered Period:", 20, 90, 175, 102, base + 12),
            _atom(page, "2024-01-01", 180, 90, 245, 102, base + 13),
            _atom(page, "-", 250, 90, 256, 102, base + 14),
            _atom(page, "2024-01-31", 261, 90, 326, 102, base + 15),
            _atom(page, "Filter:", 350, 90, 395, 102, base + 29),
            _atom(page, "Posted electronic transfers", 400, 90, 540, 102, base + 30),
            _atom(page, "Date", 20, 120, 50, 132, base + 16),
            _atom(page, "Description", 65, 120, 130, 132, base + 17),
            _atom(page, "Account Name", 145, 120, 230, 132, base + 18),
            _atom(page, "Account Number", 245, 120, 345, 132, base + 19),
            _atom(page, "Debit Amount", 360, 120, 440, 132, base + 20),
            _atom(page, "Credit Amount", 455, 120, 540, 132, base + 21),
            _atom(page, "Balance", 555, 120, 605, 132, base + 22),
            _atom(page, "2024-01-15", 20, 145, 85, 157, base + 23),
            _atom(page, "Invoice payment", 95, 145, 180, 157, base + 24),
            _atom(page, "Body Counterparty", 190, 145, 290, 157, base + 25),
            _atom(page, "CP-998877", 300, 145, 365, 157, base + 26),
            _atom(page, "25.00", 380, 145, 415, 157, base + 27),
            _atom(page, "975.00", 555, 145, 600, 157, base + 28),
        ]

    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[], texts=[]),
        ],
        parser_info=SimpleNamespace(options={}),
    )
    first_atoms = statement_atoms(1, "2024-02-01")
    second_atoms = statement_atoms(2, "2024-02-02")
    ledger_header = [atom for atom in first_atoms if float(atom["bbox"][1]) == 120.0]
    assert statement_context._is_transaction_header_row(ledger_header) is True
    assert statement_context._facts_from_row(ledger_header, 1) == []
    monkeypatch.setattr(
        statement_context,
        "text_atoms",
        lambda _result: [*first_atoms, *second_atoms],
    )
    monkeypatch.setattr(
        statement_context,
        "_source_page_texts",
        lambda _result: {1: "Page 1 of 1", 2: "Page 1 of 1"},
    )

    records = statement_context.build_statement_header_records(parse_result, {})

    assert statement_context._is_local_first_page("Page 1 of 1") is True
    assert [record["source"]["page_range"] for record in records] == [[1, 1], [2, 2]]
    assert [record["normalized"]["print_date"] for record in records] == ["2024-02-01", "2024-02-02"]
    for page, record in enumerate(records, start=1):
        assert record["normalized"]["statement_title"] == "Example Bank Account Statement"
        assert record["normalized"]["account_holder"] == "Example Trading Ltd"
        assert record["normalized"]["account_number"] == "123456789012"
        assert record["normalized"]["currency"] == "USD"
        assert record["normalized"]["query_period"] == "2024-01-01 ~ 2024-01-31"
        assert record["normalized"]["filter_condition"] == "Posted electronic transfers"
        assert record["canonical_raw"]["query_period"] == "2024-01-01-2024-01-31"
        assert record["canonical_raw"]["filter_condition"] == "Posted electronic transfers"
        assert record["raw"]["Page:"] == "1/1"
        assert record["raw"]["Statement Covered Period:"] == "2024-01-01-2024-01-31"
        assert record["raw"]["Filter:"] == "Posted electronic transfers"
        assert record["source"]["field_sources"]["query_period"]["source_refs"] == [
            {
                "source": "canonical_evidence_atoms",
                "source_page": page,
                "bbox": [20.0, 90.0, 326.0, 102.0],
            }
        ]
        assert record["source"]["field_sources"]["filter_condition"]["source_refs"] == [
            {
                "source": "canonical_evidence_atoms",
                "source_page": page,
                "bbox": [350.0, 90.0, 540.0, 102.0],
            }
        ]
        assert "source_header_page_label" not in record["normalized"]
        assert "source_header_page_label" not in record["canonical_raw"]
        assert "Body Counterparty" not in str(record)
        assert not {"date", "description", "amount", "balance"}.intersection(record["normalized"])


@pytest.mark.parametrize("failure", ["distant", "ambiguous"])
def test_english_page_label_requires_one_tightly_bounded_source_value(failure):
    values = [_atom(1, "1/2", 200, 50, 222, 62, 2)]
    if failure == "ambiguous":
        values = [
            _atom(1, "1/2", 60, 50, 82, 62, 2),
            _atom(1, "2/2", 86, 50, 108, 62, 3),
        ]
    row = [_atom(1, "Page:", 20, 50, 55, 62, 1), *values]

    facts = statement_context._facts_from_row(row, 1)

    assert all(fact.field_key != "source_header_page_label" for fact in facts)


def test_geometry_header_text_is_page_bounded(monkeypatch, parse_result):
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: _current_account_atoms())

    [(page, text)] = statement_context.page_texts_with_business_headers(parse_result, [(1, "BASE")])

    assert page == 1
    assert "户名：测试企业有限公司" in text
    assert "总笔数：90" in text
    assert text.endswith("BASE")


def test_distinct_account_card_and_extended_business_labels_are_preserved(monkeypatch, parse_result):
    atoms = _current_account_atoms()
    atoms.extend(
        [
            _atom(1, "卡号:", 470, 120, 505, 132, 1001),
            _atom(1, "6230580000166957147", 510, 120, 640, 132, 1002),
            _atom(1, "申请时间:", 470, 140, 530, 152, 1003),
            _atom(1, "2023-09-04 13:24:09", 535, 140, 665, 152, 1004),
            _atom(1, "验证码:", 470, 160, 520, 172, 1005),
            _atom(1, "HE79A9HT", 525, 160, 585, 172, 1006),
            _atom(1, "自定义业务字段:来源值", 600, 160, 735, 172, 1007),
        ]
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_number"] == "3211020801201000170968"
    assert record["normalized"]["card_number"] == "6230580000166957147"
    assert record["normalized"]["application_time"] == "2023-09-04 13:24:09"
    assert record["normalized"]["verification_code"] == "HE79A9HT"
    assert record["raw"]["自定义业务字段"] == "来源值"


def test_verification_identifier_alias_is_source_bound_and_not_a_ledger_reference(monkeypatch, parse_result):
    header_atoms = [
        _atom(1, "账户交易明细", 80, 20, 220, 36, 1),
        _atom(1, "核验编号:2402203CZ1DJ", 40, 50, 220, 64, 2),
        _atom(1, "1", 0, 100, 8, 112, 3),
        _atom(1, "2024-02-20", 20, 100, 85, 112, 4),
        _atom(1, "3.15", 100, 100, 130, 112, 5),
        _atom(1, "96.85", 150, 100, 185, 112, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: header_atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["verification_code"] == "2402203CZ1DJ"
    assert record["source"]["field_sources"]["verification_code"]["evidence_ids"] == [
        "ev:0001:text:000002"
    ]

    ledger_atoms = [
        _atom(1, "账户交易明细", 80, 20, 220, 36, 1),
        _atom(1, "交易日期", 0, 70, 55, 82, 2),
        _atom(1, "交易金额", 70, 70, 125, 82, 3),
        _atom(1, "余额", 140, 70, 165, 82, 4),
        _atom(1, "摘要", 180, 70, 215, 82, 5),
        _atom(1, "核验编号:2402203CZ1DJ", 220, 90, 370, 102, 6),
        _atom(1, "1", 0, 110, 8, 122, 7),
        _atom(1, "2024-02-20", 20, 110, 85, 122, 8),
        _atom(1, "3.15", 100, 110, 130, 122, 9),
        _atom(1, "96.85", 150, 110, 185, 122, 10),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: ledger_atoms)

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert "verification_code" not in record["normalized"]


def test_outlet_number_is_preserved_without_claiming_a_branch_name(monkeypatch, parse_result):
    atoms = [
        _atom(1, "账户交易明细", 80, 20, 220, 36, 1),
        _atom(1, "网点号:0000", 40, 50, 120, 64, 2),
        _atom(1, "1", 0, 100, 8, 112, 3),
        _atom(1, "2024-02-20", 20, 100, 85, 112, 4),
        _atom(1, "3.15", 100, 100, 130, 112, 5),
        _atom(1, "96.85", 150, 100, 185, 112, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["branch_number"] == "0000"
    assert record["canonical_raw"]["branch_number"] == "0000"
    assert "branch_name" not in record["normalized"]
    assert record["source"]["field_sources"]["branch_number"]["evidence_ids"] == [
        "ev:0001:text:000002"
    ]


@pytest.mark.parametrize("direction_filter", ["全部", "收入"])
def test_query_filter_block_preserves_absence_semantics_without_claiming_business_values(
    monkeypatch,
    parse_result,
    direction_filter,
):
    atoms = [
        _atom(1, "账户交易明细", 80, 20, 220, 36, 1),
        _atom(1, f"收支类别:{direction_filter}", 40, 45, 140, 59, 2),
        _atom(1, "交易类型:全部", 160, 45, 260, 59, 3),
        _atom(1, "转账金额区间:无", 40, 65, 160, 79, 4),
        _atom(1, "对方户名:无", 180, 65, 260, 79, 5),
        _atom(1, "对方账号:无", 280, 65, 360, 79, 6),
        _atom(1, "用途/备注:无", 380, 65, 470, 79, 7),
        _atom(1, "1", 0, 120, 8, 132, 8),
        _atom(1, "2024-02-20", 20, 120, 85, 132, 9),
        _atom(1, "3.15", 100, 120, 130, 132, 10),
        _atom(1, "96.85", 150, 120, 185, 132, 11),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["direction_filter"] == direction_filter
    assert record["normalized"]["transaction_type_filter"] == "全部"
    assert record["normalized"]["transfer_amount_filter"] == "无"
    assert record["normalized"]["counterparty_name_filter"] == "无"
    assert record["normalized"]["counterparty_account_filter"] == "无"
    assert record["normalized"]["purpose_note_filter"] == "无"
    expected_sources = {
        "direction_filter": (direction_filter, "ev:0001:text:000002", [40.0, 45.0, 140.0, 59.0]),
        "transaction_type_filter": ("全部", "ev:0001:text:000003", [160.0, 45.0, 260.0, 59.0]),
        "transfer_amount_filter": ("无", "ev:0001:text:000004", [40.0, 65.0, 160.0, 79.0]),
        "counterparty_name_filter": ("无", "ev:0001:text:000005", [180.0, 65.0, 260.0, 79.0]),
        "counterparty_account_filter": ("无", "ev:0001:text:000006", [280.0, 65.0, 360.0, 79.0]),
        "purpose_note_filter": ("无", "ev:0001:text:000007", [380.0, 65.0, 470.0, 79.0]),
    }
    field_sources = record["source"]["field_sources"]
    for field_key, (raw_value, evidence_id, bbox) in expected_sources.items():
        assert record["canonical_raw"][field_key] == raw_value
        assert field_sources[field_key]["evidence_ids"] == [evidence_id]
        assert field_sources[field_key]["source_refs"] == [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": bbox,
            }
        ]
    assert "amount_lower_limit" not in record["normalized"]
    assert "amount_upper_limit" not in record["normalized"]
    assert "amount_lower_limit" not in record["canonical_raw"]
    assert "amount_upper_limit" not in record["canonical_raw"]
    assert "amount_lower_limit" not in field_sources
    assert "amount_upper_limit" not in field_sources
    assert "account_holder" not in record["normalized"]
    assert "account_number" not in record["normalized"]


def test_query_filter_block_preserves_substantive_source_business_values(monkeypatch, parse_result):
    expected = {
        "direction_filter": "支出",
        "transaction_type_filter": "转账",
        "transfer_amount_filter": "100.00-5000.00",
        "counterparty_name_filter": "上海示例供应链有限公司",
        "counterparty_account_filter": "6222020202020202",
        "purpose_note_filter": "货款",
    }
    atoms = [
        _atom(1, "账户交易明细", 80, 20, 220, 36, 1),
        _atom(1, "收支类别:支出", 20, 45, 120, 59, 2),
        _atom(1, "交易类型:转账", 130, 45, 230, 59, 3),
        _atom(1, "转账金额区间:100.00-5000.00", 240, 45, 430, 59, 4),
        _atom(1, "对方户名:上海示例供应链有限公司", 20, 65, 220, 79, 5),
        _atom(1, "对方账号:6222020202020202", 230, 65, 420, 79, 6),
        _atom(1, "用途/备注:货款", 430, 65, 530, 79, 7),
        _atom(1, "1", 0, 120, 8, 132, 8),
        _atom(1, "2024-02-20", 20, 120, 85, 132, 9),
        _atom(1, "3.15", 100, 120, 130, 132, 10),
        _atom(1, "96.85", 150, 120, 185, 132, 11),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    for field_key, value in expected.items():
        assert record["normalized"][field_key] == value
        assert record["canonical_raw"][field_key] == value
        source = record["source"]["field_sources"][field_key]
        assert source["source"] == "canonical_evidence_atoms"
        assert source["evidence_ids"]
        assert source["source_refs"][0]["source_page"] == 1
    assert "account_holder" not in record["normalized"]
    assert "account_number" not in record["normalized"]


@pytest.mark.parametrize(
    ("field_key", "value"),
    [
        ("transaction_type_filter", "转账 对方账号:6222020202020202"),
        ("transfer_amount_filter", "5000.00-100.00"),
        ("transfer_amount_filter", "2024-02-20"),
        ("counterparty_name_filter", "对方账号:6222020202020202"),
        ("counterparty_account_filter", "上海示例供应链有限公司"),
        ("purpose_note_filter", "交易日期"),
    ],
)
def test_substantive_query_filter_values_remain_bounded_and_fail_closed(field_key, value):
    assert statement_context._fact_value_is_plausible(field_key, value) is False


def test_payment_certificate_promotes_embedded_identity_currency_and_unit(monkeypatch, parse_result):
    atoms = [
        _atom(1, "微信支付交易明细证明", 100, 20, 260, 40, 1),
        _atom(
            1,
            "兹证明:蔡子亮(身份证:340111199002288516),在其微信号:cailiang1215中的交易明细信息如下:",
            40,
            50,
            500,
            64,
            2,
        ),
        _atom(1, "币种:人民币/单位:元", 350, 70, 500, 84, 3),
        _atom(1, "交易明细对应时间段", 40, 90, 170, 104, 4),
        _atom(1, "2023-01-02", 180, 90, 245, 104, 5),
        _atom(1, "00:00:00至2024-01-01", 250, 90, 390, 104, 6),
        _atom(1, "23:59:59", 395, 90, 450, 104, 7),
        _atom(1, "序号", 0, 115, 25, 127, 8),
        _atom(1, "交易日期", 40, 115, 90, 127, 9),
        _atom(1, "交易金额", 110, 115, 165, 127, 10),
        _atom(1, "余额", 180, 115, 210, 127, 11),
        _atom(1, "1", 0, 145, 8, 157, 12),
        _atom(1, "2024-01-01", 40, 145, 105, 157, 13),
        _atom(1, "3.15", 110, 145, 140, 157, 14),
        _atom(1, "96.85", 180, 145, 215, 157, 15),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_holder"] == "蔡子亮"
    assert record["normalized"]["id_number"] == "340111199002288516"
    assert record["normalized"]["wechat_id"] == "cailiang1215"
    assert record["normalized"]["currency"] == "CNY"
    assert record["normalized"]["amount_unit"] == "元"
    assert record["normalized"]["period_start"] == "2023-01-02"
    assert record["normalized"]["period_end"] == "2024-01-01"
    assert record["canonical_raw"]["currency"] == "人民币"
    assert record["canonical_raw"]["amount_unit"] == "元"
    field_sources = record["source"]["field_sources"]
    assert field_sources["account_holder"]["evidence_ids"] == ["ev:0001:text:000002"]
    assert field_sources["id_number"]["evidence_ids"] == ["ev:0001:text:000002"]
    assert field_sources["wechat_id"]["evidence_ids"] == ["ev:0001:text:000002"]
    assert field_sources["currency"]["evidence_ids"] == ["ev:0001:text:000003"]
    assert field_sources["amount_unit"]["evidence_ids"] == ["ev:0001:text:000003"]


@pytest.mark.parametrize(
    ("packed_header", "expected"),
    [
        ("户名:张三/币种:人民币", {"account_holder": "张三", "currency": "CNY"}),
        ("开户行:测试支行/单位:元", {"branch_name": "测试支行", "amount_unit": "元"}),
    ],
)
def test_packed_header_supersedes_contaminated_whole_atom_fact(
    monkeypatch,
    parse_result,
    packed_header,
    expected,
):
    atoms = [
        _atom(1, "账户交易明细", 100, 20, 260, 40, 1),
        _atom(1, packed_header, 40, 50, 300, 64, 2),
        _atom(1, "1", 0, 130, 8, 142, 3),
        _atom(1, "2024-01-01", 40, 130, 105, 142, 4),
        _atom(1, "3.15", 110, 130, 140, 142, 5),
        _atom(1, "96.85", 180, 130, 215, 142, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert {key: record["normalized"][key] for key in expected} == expected


def test_packed_traditional_labels_use_length_preserving_matching_with_source_provenance(
    monkeypatch,
    parse_result,
):
    atoms = [
        _atom(1, "账户交易明细", 100, 20, 260, 40, 1),
        _atom(1, "戶名:張三/幣種:人民幣", 40, 50, 300, 64, 2),
        _atom(1, "1", 0, 130, 8, 142, 3),
        _atom(1, "2024-01-01", 40, 130, 105, 142, 4),
        _atom(1, "3.15", 110, 130, 140, 142, 5),
        _atom(1, "96.85", 180, 130, 215, 142, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_holder"] == "張三"
    assert record["normalized"]["currency"] == "CNY"
    assert record["canonical_raw"]["account_holder"] == "張三"
    assert record["canonical_raw"]["currency"] == "人民幣"
    assert record["raw"]["戶名:"] == "張三"
    assert record["raw"]["幣種:"] == "人民幣"
    for field_key in ("account_holder", "currency"):
        assert record["source"]["field_sources"][field_key]["evidence_ids"] == [
            "ev:0001:text:000002"
        ]
        assert record["source"]["field_sources"][field_key]["source_refs"] == [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": [40.0, 50.0, 300.0, 64.0],
            }
        ]


def test_packed_header_rejects_the_entire_compound_when_amount_unit_is_prose():
    row = [_atom(1, "币种:CNY/单位:仅供参考", 40, 50, 260, 64, 1)]

    facts = statement_context._facts_from_row(row, 1)

    assert not {"currency", "amount_unit"}.intersection(fact.field_key for fact in facts)


@pytest.mark.parametrize(
    "packed_prose",
    [
        "备注文本 账号:A001/币种:CNY",
        "对方账号:6222020000000000/户名:李四/币种:CNY",
    ],
)
def test_packed_kv_scanner_cannot_start_inside_prose_or_counterparty_label(
    monkeypatch,
    parse_result,
    packed_prose,
):
    atoms = [
        _atom(1, "账户交易明细", 100, 20, 260, 40, 1),
        _atom(1, packed_prose, 40, 50, 360, 64, 2),
        _atom(1, "1", 0, 130, 8, 142, 3),
        _atom(1, "2024-01-01", 40, 130, 105, 142, 4),
        _atom(1, "3.15", 110, 130, 140, 142, 5),
        _atom(1, "96.85", 180, 130, 215, 142, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert "account_number" not in record["normalized"]
    assert "account_holder" not in record["normalized"]
    assert "currency" not in record["normalized"]


def test_certificate_identity_requires_a_plausible_chinese_id(monkeypatch, parse_result):
    atoms = [
        _atom(1, "微信支付交易明细证明", 100, 20, 260, 40, 1),
        _atom(
            1,
            "兹证明:测试人员(身份证:UNKNOWN),在其微信号:valid_id中的交易明细信息如下:",
            40,
            50,
            500,
            64,
            2,
        ),
        _atom(1, "1", 0, 130, 8, 142, 3),
        _atom(1, "2024-01-01", 40, 130, 105, 142, 4),
        _atom(1, "3.15", 110, 130, 140, 142, 5),
        _atom(1, "96.85", 180, 130, 215, 142, 6),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert "account_holder" not in record["normalized"]
    assert "id_number" not in record["normalized"]
    assert "wechat_id" not in record["normalized"]


def test_embedded_identity_like_prose_and_counterparty_labels_fail_closed(monkeypatch, parse_result):
    atoms = [
        _atom(1, "微信支付交易明细证明", 100, 20, 260, 40, 1),
        _atom(
            1,
            "备注:兹证明:交易对手(身份证:340111199002288516),在其微信号:other_party中的交易明细信息如下:",
            40,
            50,
            520,
            64,
            2,
        ),
        _atom(1, "对方账号:6222020000000000/单位:元", 40, 70, 300, 84, 3),
        _atom(1, "1", 0, 130, 8, 142, 4),
        _atom(1, "2024-01-01", 40, 130, 105, 142, 5),
        _atom(1, "3.15", 110, 130, 140, 142, 6),
        _atom(1, "96.85", 180, 130, 215, 142, 7),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert "account_holder" not in record["normalized"]
    assert "id_number" not in record["normalized"]
    assert "wechat_id" not in record["normalized"]
    assert "account_number" not in record["normalized"]
    assert "amount_unit" not in record["normalized"]


def test_unlabelled_period_and_month_are_source_backed_segment_boundaries(monkeypatch):
    first = [
        _atom(1, "客户存款月结单", 100, 20, 260, 40, 1),
        _atom(1, "20230101-20230131", 100, 60, 220, 72, 2),
        _atom(1, "2023年01月", 100, 80, 180, 92, 3),
        _atom(1, "交易日期", 0, 120, 55, 132, 4),
        _atom(1, "交易金额", 70, 120, 125, 132, 5),
        _atom(1, "余额", 140, 120, 165, 132, 6),
        _atom(1, "1", 0, 145, 8, 157, 7),
        _atom(1, "2023-01-02", 20, 145, 85, 157, 8),
        _atom(1, "1.00", 100, 145, 130, 157, 9),
        _atom(1, "1.00", 140, 145, 170, 157, 10),
    ]
    second = [
        {
            **atom,
            "id": str(atom["id"]).replace("0001", "0002", 1),
            "page_id": "page:0002",
            "text": (
                "20230201-20230228"
                if atom["text"] == "20230101-20230131"
                else "2023年02月"
                if atom["text"] == "2023年01月"
                else atom["text"]
            ),
        }
        for atom in first
    ]
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[], texts=[]),
        ],
        parser_info=SimpleNamespace(options={}),
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: [*first, *second])
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: "", 2: ""})

    records = statement_context.build_statement_header_records(parse_result, {})

    assert len(records) == 2
    assert [record["normalized"]["statement_month"] for record in records] == ["2023-01", "2023-02"]
    assert [record["normalized"]["period_start"] for record in records] == ["2023-01-01", "2023-02-01"]
    assert records[0]["canonical_raw"]["query_period"] == "20230101-20230131"


def test_ocr_fallback_uses_positioned_text_blocks_and_preserves_month_scope_business_data(monkeypatch):
    def block(page: int, index: int, text: str, x0: float, y0: float, x1: float, y1: float):
        return SimpleNamespace(
            content=text,
            bbox=[x0, y0, x1, y1],
            confidence=0.99,
            evidence_ids=[f"ocr:p{page:04d}:{index:04d}"],
        )

    def page(page_number: int, month: str, opening: str, *, footer: bool = False):
        values = [
            block(page_number, 1, "平安银行", 20, 20, 85, 38),
            block(page_number, 2, "客户存款月结单", 120, 42, 230, 58),
            block(page_number, 3, "结单号:23090821289990000809", 270, 42, 460, 58),
            block(page_number, 4, month, 500, 42, 570, 58),
            block(page_number, 5, "户名:测试企业有限公司", 20, 70, 170, 84),
            block(page_number, 6, "账号:11005350836201", 190, 70, 330, 84),
            block(page_number, 7, "币种:RMB", 350, 70, 420, 84),
            block(page_number, 8, f"承前余额:{opening}", 440, 70, 560, 84),
            block(page_number, 9, "开户行:测试开户支行", 20, 90, 160, 104),
        ]
        if footer:
            values.extend(
                [
                    block(page_number, 20, "已打印次数:2", 20, 700, 100, 714),
                    block(page_number, 21, "打印方式:系统PDF生成", 120, 700, 250, 714),
                    block(page_number, 22, "设备编号:0000", 270, 700, 350, 714),
                    block(page_number, 23, "柜员号:3100525", 370, 700, 460, 714),
                ]
            )
        return SimpleNamespace(
            page_number=page_number,
            source_page_number=page_number,
            tables=[],
            texts=values,
        )

    parse_result = SimpleNamespace(
        pages=[
            page(1, "2023年3月", "405,962.31"),
            page(2, "2023年4月", "1,076,520.92"),
            page(3, "2023年4月", "900,000.00", footer=True),
        ],
        parser_info=SimpleNamespace(options={"native_text_ocr_fallback_pages": [1, 2, 3]}),
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: [])
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: "", 2: "", 3: ""})

    records = statement_context.build_statement_header_records(parse_result, {})

    assert [record["source"]["page_range"] for record in records] == [[1, 1], [2, 3]]
    assert [record["normalized"]["statement_month"] for record in records] == ["2023-03", "2023-04"]
    assert records[0]["normalized"]["bank_name"] == "平安银行"
    assert records[0]["normalized"]["branch_name"] == "测试开户支行"
    assert records[0]["normalized"]["statement_number"] == "23090821289990000809"
    assert records[1]["normalized"]["brought_forward_balance"] == "1076520.92"
    assert records[1]["raw"]["承前余额:"] == [
        {"page": 2, "value": "1,076,520.92"},
        {"page": 3, "value": "900,000.00"},
    ]
    assert records[1]["normalized"]["print_count"] == 2
    assert records[1]["normalized"]["print_method"] == "系统PDF生成"
    assert records[1]["normalized"]["device_number"] == "0000"
    assert records[1]["normalized"]["print_teller"] == "3100525"
    assert records[1]["source"]["field_sources"]["statement_number"]["source"] == "parse_result_ocr_text"


def test_hyphenated_local_page_reset_and_unlabelled_month_next_to_year_form_scopes(monkeypatch):
    atoms: list[dict] = []
    for page, month in ((1, "02"), (2, "03")):
        base = len(atoms) + 1
        atoms.extend(
            [
                _atom(page, "交通银行明细对账单", 20, 20, 180, 36, base),
                _atom(page, "年份:", 20, 55, 60, 68, base + 1),
                _atom(page, "2022", 65, 55, 95, 68, base + 2),
                _atom(page, month, 105, 55, 122, 68, base + 3),
                _atom(page, "账号:", 150, 55, 190, 68, base + 4),
                _atom(page, "A001", 195, 55, 230, 68, base + 5),
                _atom(page, "测试橡塑有限公司", 240, 55, 340, 68, base + 10),
                _atom(page, "承前", 20, 78, 50, 91, base + 11),
                _atom(page, "1,000.00", 58, 78, 110, 91, base + 12),
                _atom(page, "交易日期", 20, 100, 70, 114, base + 6),
                _atom(page, "交易金额", 90, 100, 140, 114, base + 7),
                _atom(page, "余额", 160, 100, 185, 114, base + 8),
                _atom(page, "摘要", 205, 100, 230, 114, base + 9),
                _atom(page, "本月累计借方发生数:", 20, 500, 140, 514, base + 13),
                _atom(page, "12", 150, 500, 166, 514, base + 14),
                _atom(page, "本月累计贷方", 300, 500, 390, 514, base + 15),
                _atom(page, "8", 400, 500, 410, 514, base + 16),
                _atom(page, "本月累计借方发生额:", 20, 520, 140, 534, base + 17),
                _atom(page, "1,234.56", 150, 520, 200, 534, base + 18),
                _atom(page, "本月累计贷方", 300, 520, 390, 534, base + 19),
                _atom(page, "789.01", 400, 520, 445, 534, base + 20),
                _atom(page, f"出单截至日期:2022-{month}-28", 20, 550, 190, 564, base + 21),
            ]
        )
    parse_result = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1, tables=[], texts=[]),
            SimpleNamespace(page_number=2, source_page_number=2, tables=[], texts=[]),
        ],
        parser_info=SimpleNamespace(options={}),
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(
        statement_context,
        "_source_page_texts",
        lambda _result: {1: "页码:1-1", 2: "页码:1-1"},
    )

    records = statement_context.build_statement_header_records(parse_result, {})

    assert statement_context._is_local_first_page("页码:1-1") is True
    assert statement_context._normalize_field_value("statement_month", "2023年3月") == "2023-03"
    assert [record["normalized"]["statement_month"] for record in records] == ["2022-02", "2022-03"]
    assert [record["source"]["page_range"] for record in records] == [[1, 1], [2, 2]]
    assert records[0]["normalized"]["account_number"] == "A001"
    assert records[0]["normalized"]["account_holder"] == "测试橡塑有限公司"
    assert records[0]["normalized"]["brought_forward_balance"] == "1000.00"
    assert records[0]["normalized"]["debit_count"] == 12
    assert records[0]["normalized"]["credit_count"] == 8
    assert records[0]["normalized"]["debit_total"] == "1234.56"
    assert records[0]["normalized"]["credit_total"] == "789.01"
    assert records[0]["normalized"]["statement_cutoff_date"] == "2022-02-28"


def test_transaction_column_headings_do_not_become_statement_facts(monkeypatch, parse_result):
    atoms = [atom for atom in _current_account_atoms() if atom["bbox"][1] >= 200]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    assert statement_context.build_statement_header_records(parse_result, {}) == []


def test_transaction_column_headings_cannot_contaminate_a_valid_statement_header(monkeypatch, parse_result):
    atoms = [atom for atom in _current_account_atoms() if atom["bbox"][1] < 200]
    atoms.extend(
        [
            _atom(1, "序号", 0, 200, 25, 212, 3001),
            _atom(1, "摘要", 30, 200, 55, 212, 3002),
            _atom(1, "币别", 60, 200, 85, 212, 3003),
            _atom(1, "钞汇", 90, 200, 115, 212, 3004),
            _atom(1, "交易日期", 120, 200, 170, 212, 3005),
            _atom(1, "交易金额", 175, 200, 225, 212, 3006),
            _atom(1, "账户余额", 230, 200, 280, 212, 3007),
            _atom(1, "交易地点/附言", 285, 200, 365, 212, 3008),
            _atom(1, "对方账号与户名", 370, 200, 465, 212, 3009),
            _atom(1, "1", 0, 225, 8, 237, 3010),
            _atom(1, "2023-02-24", 120, 225, 185, 237, 3011),
            _atom(1, "100.00", 175, 225, 215, 237, 3012),
            _atom(1, "100.00", 230, 225, 270, 237, 3013),
        ]
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["currency"] == "CNY"
    assert "cash_remittance" not in record["normalized"]
    assert "钞汇" not in record["raw"]


def test_mixed_compatibility_glyphs_match_header_aliases_without_changing_raw_labels(monkeypatch, parse_result):
    replacements = {"户": "戶", "名:": "名:", "账": "賬", "人民币": "人⺠币"}
    atoms = [{**atom, "text": replacements.get(atom["text"], atom["text"])} for atom in _current_account_atoms()]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_holder"] == "测试企业有限公司"
    assert record["normalized"]["account_number"] == "3211020801201000170968"
    assert record["normalized"]["currency"] == "CNY"
    assert any("戶名" in key for key in record["raw"])


@pytest.mark.parametrize("separator", ["-", "—", "~", "至"])
def test_inline_period_consumes_adjacent_separator_and_end_date(monkeypatch, parse_result, separator):
    atoms = [
        _atom(1, "账户交易明细", 100, 20, 260, 40, 1),
        _atom(1, "起止日期:2022-09-04", 100, 60, 235, 72, 2),
        _atom(1, separator, 240, 60, 248, 72, 3),
        _atom(1, "2023-09-04", 253, 60, 318, 72, 4),
        _atom(1, "交易日期", 0, 100, 55, 112, 5),
        _atom(1, "交易金额", 70, 100, 125, 112, 6),
        _atom(1, "余额", 140, 100, 165, 112, 7),
        _atom(1, "1", 0, 125, 8, 137, 8),
        _atom(1, "2022-09-05", 20, 125, 85, 137, 9),
        _atom(1, "1.00", 100, 125, 130, 137, 10),
        _atom(1, "1.00", 140, 125, 170, 137, 11),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["query_period"] == "2022-09-04 ~ 2023-09-04"
    assert record["normalized"]["period_start"] == "2022-09-04"
    assert record["normalized"]["period_end"] == "2023-09-04"


def test_header_transaction_time_range_is_a_source_backed_query_period(monkeypatch, parse_result):
    atoms = [
        _atom(1, "账户交易明细", 100, 20, 260, 40, 1),
        _atom(1, "交易时间:2022-09-04", 100, 60, 230, 72, 2),
        _atom(1, "至", 235, 60, 247, 72, 3),
        _atom(1, "2023-09-04", 252, 60, 317, 72, 4),
        _atom(1, "交易日期", 0, 100, 55, 112, 5),
        _atom(1, "交易金额", 70, 100, 125, 112, 6),
        _atom(1, "余额", 140, 100, 165, 112, 7),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["query_period"] == "2022-09-04 ~ 2023-09-04"
    assert record["normalized"]["period_start"] == "2022-09-04"
    assert record["normalized"]["period_end"] == "2023-09-04"
    assert record["canonical_raw"]["query_period"] == "2022-09-04 至 2023-09-04"
    assert record["canonical_raw"]["period_start"] == "2022-09-04"
    assert record["canonical_raw"]["period_end"] == "2023-09-04"
    assert record["raw"]["交易时间"] == "2022-09-04 至 2023-09-04"
    query_source = record["source"]["field_sources"]["query_period"]
    assert query_source == {
        "raw_name": "交易时间",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 1,
                "bbox": [100.0, 60.0, 317.0, 72.0],
            }
        ],
        "evidence_ids": [
            "ev:0001:text:000002",
            "ev:0001:text:000003",
            "ev:0001:text:000004",
        ],
    }
    assert record["source"]["field_sources"]["period_start"] == query_source
    assert record["source"]["field_sources"]["period_end"] == query_source


@pytest.mark.parametrize(
    "row",
    [
        [
            _atom(1, "交易时间", 100, 60, 150, 72, 1),
            _atom(1, "2022-09-04", 155, 60, 220, 72, 2),
            _atom(1, "至", 225, 60, 237, 72, 3),
            _atom(1, "2023-09-04", 242, 60, 307, 72, 4),
        ],
        [
            _atom(1, "交易时间:2022-09-04", 100, 60, 230, 72, 1),
            _atom(1, "至2023-09-04", 235, 60, 312, 72, 2),
        ],
        [
            _atom(1, "交易时间:2023-09-04", 100, 60, 230, 72, 1),
            _atom(1, "至", 235, 60, 247, 72, 2),
            _atom(1, "2022-09-04", 252, 60, 317, 72, 3),
        ],
    ],
    ids=["start-not-inline", "separator-not-standalone", "reversed"],
)
def test_header_transaction_time_range_fails_closed_without_exact_structure(row):
    facts = statement_context._facts_from_row(row, 1)

    assert all(fact.field_key != "query_period" for fact in facts)


def test_ledger_transaction_time_header_remains_excluded():
    row = [
        _atom(1, "交易日期", 0, 100, 55, 112, 1),
        _atom(1, "交易时间", 70, 100, 125, 112, 2),
        _atom(1, "交易金额", 140, 100, 195, 112, 3),
        _atom(1, "余额", 210, 100, 235, 112, 4),
    ]

    assert statement_context._facts_from_row(row, 1) == []


@pytest.mark.parametrize(
    "value",
    [
        "2024-02-18-2010-13-98",
        "2024-03-24-2052-00-90",
        "2024-03-24-2023-03-01",
        "reference 20240108 text 20240301",
    ],
)
def test_unlabelled_period_rejects_invalid_reversed_or_unbounded_transaction_text(value):
    row = [_atom(1, value, 100, 30, 350, 42, 1)]

    assert statement_context._unlabelled_period_fact([row], [], 1) is None


def test_top_document_title_beats_lower_transaction_value():
    rows = [
        [_atom(1, "交通银行个人客户交易清单", 100, 20, 300, 40, 1)],
        [_atom(1, "交易流水", 100, 180, 170, 192, 2)],
    ]

    title = statement_context._title_fact(rows, 1)

    assert title is not None
    assert title.normalized_value == "交通银行个人客户交易清单"


def test_transaction_value_below_ledger_header_is_not_a_document_title():
    rows = [
        [
            _atom(1, "交易日期", 0, 50, 50, 62, 1),
            _atom(1, "交易金额", 60, 50, 110, 62, 2),
            _atom(1, "账户余额", 120, 50, 170, 62, 3),
            _atom(1, "摘要", 180, 50, 210, 62, 4),
        ],
        [_atom(1, "交易流水", 100, 75, 170, 87, 5)],
    ]

    assert statement_context._title_fact(rows, 1) is None


def test_disclaimer_is_preserved_but_never_claimed_as_statement_title():
    disclaimer_text = "该交易明细因不可预测的非人控技术原因可能导致数据缺失，明细内容仅供参考。"
    rows = [[_atom(2, disclaimer_text, 40, 20, 500, 36, 1)]]

    assert statement_context._title_fact(rows, 2) is None
    disclaimer = statement_context._statement_disclaimer_fact(rows, 2)
    assert disclaimer is not None
    assert disclaimer.field_key == "statement_disclaimer"
    assert disclaimer.normalized_value == statement_context._nfkc(disclaimer_text)
    assert disclaimer.evidence_ids == ("ev:0002:text:000001",)

    ranked_rows = [
        [_atom(1, disclaimer_text, 40, 10, 500, 26, 1)],
        [_atom(1, "中国农业银行账户活期交易明细清单", 100, 40, 360, 58, 2)],
    ]
    title = statement_context._title_fact(ranked_rows, 1)
    assert title is not None
    assert title.normalized_value == "中国农业银行账户活期交易明细清单"
    assert title.evidence_ids == ("ev:0001:text:000002",)


def test_english_disclaimer_is_source_business_text_but_never_a_statement_title():
    disclaimer_text = (
        "Disclaimer: This account statement is provided for reference only and is not legal proof."
    )
    rows = [[_atom(2, disclaimer_text, 40, 20, 540, 36, 1)]]

    assert statement_context._title_fact(rows, 2) is None
    disclaimer = statement_context._statement_disclaimer_fact(rows, 2)
    assert disclaimer is not None

    record = statement_context._record_from_facts([disclaimer], [2], 1)

    assert record["normalized"] == {"statement_disclaimer": disclaimer_text}
    assert record["canonical_raw"] == {"statement_disclaimer": disclaimer_text}
    assert record["raw"] == {"document_disclaimer": disclaimer_text}
    assert record["source"]["field_sources"]["statement_disclaimer"] == {
        "raw_name": "document_disclaimer",
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source": "canonical_evidence_atoms",
                "source_page": 2,
                "bbox": [40.0, 20.0, 540.0, 36.0],
            }
        ],
        "evidence_ids": ["ev:0002:text:000001"],
    }
    assert "statement_title" not in record["normalized"]


@pytest.mark.parametrize(
    "disclaimer_text",
    [
        "Disclaimer: Values shown below are unaudited.",
        "This account statement is provided for reference only and is not legal proof.",
    ],
    ids=["explicit-prefix", "unprefixed-legal-reference-markers"],
)
def test_generic_english_disclaimer_prefix_and_markers_block_title_promotion(
    disclaimer_text: str,
) -> None:
    rows = [[_atom(1, disclaimer_text, 40, 20, 540, 36, 1)]]

    disclaimer = statement_context._statement_disclaimer_fact(rows, 1)

    assert disclaimer is not None
    assert disclaimer.raw_value == disclaimer_text
    assert disclaimer.normalized_value == disclaimer_text
    assert statement_context._title_fact(rows, 1) is None


def test_first_page_title_wins_while_later_disclaimer_remains_raw():
    title = statement_context._HeaderFact(
        "statement_title",
        "document_title",
        "ABC银行卡交易明细",
        "ABC银行卡交易明细",
        1,
        "page:0001",
        (100.0, 20.0, 300.0, 40.0),
        ("ev:title",),
    )
    disclaimer = statement_context._HeaderFact(
        "statement_title",
        "document_title",
        "本交易明细仅供参考",
        "本交易明细仅供参考",
        2,
        "page:0002",
        (40.0, 700.0, 260.0, 714.0),
        ("ev:disclaimer",),
    )

    record = statement_context._record_from_facts([title, disclaimer], [1, 2], 1)

    assert record["normalized"]["statement_title"] == "ABC银行卡交易明细"
    assert record["canonical_raw"]["statement_title"] == "ABC银行卡交易明细"
    assert record["raw"]["document_title"] == [
        {"page": 1, "value": "ABC银行卡交易明细"},
        {"page": 2, "value": "本交易明细仅供参考"},
    ]


def test_complete_page_direction_quartets_are_summed_with_component_provenance():
    labels = {
        "debit_count": "本页支出笔数:",
        "debit_total": "本页支出算数合计:",
        "credit_count": "本页收入笔数:",
        "credit_total": "本页收入算数合计:",
    }

    def fact(page, field_key, raw_value, index):
        return statement_context._HeaderFact(
            field_key,
            labels[field_key],
            raw_value,
            statement_context._normalize_field_value(field_key, raw_value),
            page,
            f"page:{page:04d}",
            (10.0, 500.0, 100.0, 514.0),
            (f"ev:{page:04d}:text:{index:06d}",),
        )

    facts = [
        fact(1, "debit_count", "2", 1),
        fact(1, "debit_total", "-10.00", 2),
        fact(1, "credit_count", "1", 3),
        fact(1, "credit_total", "20.00", 4),
        fact(2, "debit_count", "1", 1),
        fact(2, "debit_total", "-5.00", 2),
        fact(2, "credit_count", "0", 3),
        fact(2, "credit_total", "0.00", 4),
    ]

    record = statement_context._record_from_facts(
        facts,
        [1, 2],
        1,
        allow_page_direction_aggregates=True,
    )

    assert record["normalized"]["debit_count"] == 3
    assert record["normalized"]["debit_total"] == "15.00"
    assert record["normalized"]["credit_count"] == 1
    assert record["normalized"]["credit_total"] == "20.00"
    assert not {"debit_count", "debit_total", "credit_count", "credit_total"}.intersection(
        record["canonical_raw"]
    )
    source = record["source"]["field_sources"]["debit_total"]
    assert source["source"] == "derived_explicit_page_aggregate"
    assert source["derivation"] == "sum_explicit_page_totals"
    assert source["sign_normalization"] == "magnitude_from_nonpositive_expense_page_totals"
    assert [component["page"] for component in source["components"]] == [1, 2]
    assert [component["raw_value"] for component in source["components"]] == ["-10.00", "-5.00"]


@pytest.mark.parametrize(
    ("debit_values", "expected_sign_normalization"),
    [
        (("10.00", "5.00"), "magnitude_from_nonnegative_expense_page_totals"),
        (("-10.00", "-5.00"), "magnitude_from_nonpositive_expense_page_totals"),
    ],
    ids=["positive-magnitudes", "negative-signed"],
)
def test_complete_page_total_pairs_are_summed_without_fabricating_counts(
    debit_values,
    expected_sign_normalization,
):
    labels = {
        "debit_total": "本页支出算术合计:",
        "credit_total": "本页收入算术合计:",
    }

    def fact(page, field_key, raw_value, index):
        return statement_context._HeaderFact(
            field_key,
            labels[field_key],
            raw_value,
            statement_context._normalize_field_value(field_key, raw_value),
            page,
            f"page:{page:04d}",
            (10.0, 500.0 + index * 20.0, 160.0, 514.0 + index * 20.0),
            (f"ev:{page:04d}:text:{index:06d}",),
        )

    facts = [
        fact(1, "debit_total", debit_values[0], 1),
        fact(1, "credit_total", "20.00", 2),
        fact(2, "debit_total", debit_values[1], 1),
        fact(2, "credit_total", "3.00", 2),
    ]

    record = statement_context._record_from_facts(
        facts,
        [1, 2],
        1,
        allow_page_direction_aggregates=True,
    )

    assert record["normalized"] == {"debit_total": "15.00", "credit_total": "23.00"}
    assert not {"debit_count", "credit_count"}.intersection(record["normalized"])
    assert not {"debit_total", "credit_total", "debit_count", "credit_count"}.intersection(
        record["canonical_raw"]
    )
    assert record["raw"]["本页支出算术合计:"] == [
        {"page": 1, "value": debit_values[0]},
        {"page": 2, "value": debit_values[1]},
    ]
    assert record["raw"]["本页收入算术合计:"] == [
        {"page": 1, "value": "20.00"},
        {"page": 2, "value": "3.00"},
    ]
    debit_source = record["source"]["field_sources"]["debit_total"]
    assert debit_source["sign_normalization"] == expected_sign_normalization
    assert debit_source["evidence_ids"] == ["ev:0001:text:000001", "ev:0002:text:000001"]
    assert debit_source["source_refs"] == [
        {
            "source": "canonical_evidence_atoms",
            "source_page": 1,
            "bbox": [10.0, 520.0, 160.0, 534.0],
        },
        {
            "source": "canonical_evidence_atoms",
            "source_page": 2,
            "bbox": [10.0, 520.0, 160.0, 534.0],
        },
    ]
    assert debit_source["components"] == [
        {
            "page": 1,
            "raw_name": "本页支出算术合计:",
            "raw_value": debit_values[0],
            "normalized_value": debit_values[0],
            "bbox": [10.0, 520.0, 160.0, 534.0],
            "evidence_ids": ["ev:0001:text:000001"],
            "source": "canonical_evidence_atoms",
        },
        {
            "page": 2,
            "raw_name": "本页支出算术合计:",
            "raw_value": debit_values[1],
            "normalized_value": debit_values[1],
            "bbox": [10.0, 520.0, 160.0, 534.0],
            "evidence_ids": ["ev:0002:text:000001"],
            "source": "canonical_evidence_atoms",
        },
    ]


def test_repeated_identical_page_totals_remain_page_scoped_raw_arrays():
    facts = [
        statement_context._HeaderFact(
            "debit_total",
            "本页支出算术合计:",
            "10.00",
            "10.00",
            page,
            f"page:{page:04d}",
            None,
            (f"ev:{page:04d}:text:000001",),
        )
        for page in (1, 2)
    ]

    assert statement_context._raw_header_map(facts) == {
        "本页支出算术合计:": [
            {"page": 1, "value": "10.00"},
            {"page": 2, "value": "10.00"},
        ]
    }


@pytest.mark.parametrize(
    "failure",
    ["missing", "mixed_sign", "negative_credit", "competing", "partial_counts", "noncontiguous"],
)
def test_page_total_pair_fails_closed(failure):
    labels = {
        "debit_total": "本页支出算术合计:",
        "credit_total": "本页收入算术合计:",
    }

    def fact(page, field_key, raw_value, index, raw_name=None):
        return statement_context._HeaderFact(
            field_key,
            raw_name or labels.get(field_key, "本页支出笔数:"),
            raw_value,
            statement_context._normalize_field_value(field_key, raw_value),
            page,
            f"page:{page:04d}",
            None,
            (f"ev:{page:04d}:text:{index:06d}",),
        )

    facts = [
        fact(1, "debit_total", "10.00", 1),
        fact(1, "credit_total", "20.00", 2),
        fact(2, "debit_total", "5.00", 1),
        fact(2, "credit_total", "3.00", 2),
    ]
    pages = [1, 2]
    if failure == "missing":
        facts = [item for item in facts if not (item.page == 2 and item.field_key == "credit_total")]
    elif failure == "mixed_sign":
        facts = [
            fact(2, "debit_total", "-5.00", 1)
            if item.page == 2 and item.field_key == "debit_total"
            else item
            for item in facts
        ]
    elif failure == "negative_credit":
        facts = [
            fact(2, "credit_total", "-3.00", 2)
            if item.page == 2 and item.field_key == "credit_total"
            else item
            for item in facts
        ]
    elif failure == "competing":
        facts.append(fact(2, "debit_total", "15.00", 99, raw_name="借方总金额:"))
    elif failure == "partial_counts":
        facts.append(fact(1, "debit_count", "2", 3))
    else:
        pages = [1, 3]
        facts = [
            statement_context._HeaderFact(
                item.field_key,
                item.raw_name,
                item.raw_value,
                item.normalized_value,
                3 if item.page == 2 else item.page,
                "page:0003" if item.page == 2 else item.page_id,
                item.bbox,
                item.evidence_ids,
            )
            for item in facts
        ]

    record = statement_context._record_from_facts(
        facts,
        pages,
        1,
        allow_page_direction_aggregates=True,
    )

    assert not {"debit_count", "debit_total", "credit_count", "credit_total"}.intersection(
        record["normalized"]
    )


def test_page_total_pair_requires_known_complete_source_page_count(monkeypatch):
    def fact(page, field_key, raw_value, index):
        return statement_context._HeaderFact(
            field_key,
            "本页支出算术合计:" if field_key == "debit_total" else "本页收入算术合计:",
            raw_value,
            statement_context._normalize_field_value(field_key, raw_value),
            page,
            f"page:{page:04d}",
            None,
            (f"ev:{page:04d}:text:{index:06d}",),
        )

    facts_by_page = {
        1: [fact(1, "debit_total", "10.00", 1), fact(1, "credit_total", "20.00", 2)],
        2: [fact(2, "debit_total", "5.00", 1), fact(2, "credit_total", "3.00", 2)],
    }
    complete = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1),
            SimpleNamespace(page_number=2, source_page_number=2),
        ],
        parser_info=SimpleNamespace(options={"source_page_count": 2, "selected_source_pages": [1, 2]}),
    )
    unknown = deepcopy(complete)
    unknown.parser_info.options = {}
    monkeypatch.setattr(statement_context, "_page_header_facts", lambda _result: (facts_by_page, {}))
    monkeypatch.setattr(statement_context, "_context_page_groups", lambda _result, _facts: [[1, 2]])

    [complete_record] = statement_context.build_statement_header_records(complete, {})
    [unknown_record] = statement_context.build_statement_header_records(unknown, {})

    assert complete_record["normalized"] == {"debit_total": "15.00", "credit_total": "23.00"}
    assert not {"debit_total", "credit_total"}.intersection(unknown_record["normalized"])


@pytest.mark.parametrize("failure", ["missing", "mixed_sign", "competing", "noncontiguous"])
def test_page_direction_aggregate_fails_closed_as_an_atomic_quartet(failure):
    labels = {
        "debit_count": "本页支出笔数:",
        "debit_total": "本页支出算数合计:",
        "credit_count": "本页收入笔数:",
        "credit_total": "本页收入算数合计:",
    }

    def fact(page, field_key, raw_value, index, raw_name=None):
        return statement_context._HeaderFact(
            field_key,
            raw_name or labels[field_key],
            raw_value,
            statement_context._normalize_field_value(field_key, raw_value),
            page,
            f"page:{page:04d}",
            None,
            (f"ev:{page:04d}:text:{index:06d}",),
        )

    facts = [
        fact(page, field_key, raw_value, index)
        for page, values in (
            (1, ("2", "-10.00", "1", "20.00")),
            (2, ("1", "-5.00", "0", "0.00")),
        )
        for index, (field_key, raw_value) in enumerate(zip(labels, values, strict=True), start=1)
    ]
    pages = [1, 2]
    if failure == "missing":
        facts = [fact for fact in facts if not (fact.page == 2 and fact.field_key == "credit_total")]
    elif failure == "mixed_sign":
        facts = [
            fact(2, "debit_total", "5.00", 2) if item.page == 2 and item.field_key == "debit_total" else item
            for item in facts
        ]
    elif failure == "competing":
        facts.append(fact(2, "debit_total", "15.00", 99, raw_name="借方总金额:"))
    else:
        pages = [1, 3]
        facts = [
            statement_context._HeaderFact(
                item.field_key,
                item.raw_name,
                item.raw_value,
                item.normalized_value,
                3 if item.page == 2 else item.page,
                "page:0003" if item.page == 2 else item.page_id,
                item.bbox,
                item.evidence_ids,
            )
            for item in facts
        ]

    record = statement_context._record_from_facts(
        facts,
        pages,
        1,
        allow_page_direction_aggregates=True,
    )

    assert not {"debit_count", "debit_total", "credit_count", "credit_total"}.intersection(
        record["normalized"]
    )


def test_page_aggregate_requires_a_complete_source_page_selection():
    partial = SimpleNamespace(
        pages=[
            SimpleNamespace(page_number=1, source_page_number=1),
            SimpleNamespace(page_number=2, source_page_number=2),
        ],
        parser_info=SimpleNamespace(options={"source_page_count": 32, "selected_source_pages": [1, 2]}),
    )
    complete = SimpleNamespace(
        pages=[SimpleNamespace(page_number=page, source_page_number=page) for page in range(1, 3)],
        parser_info=SimpleNamespace(options={"source_page_count": 2, "selected_source_pages": [1, 2]}),
    )
    unknown = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1)],
        parser_info=SimpleNamespace(options={}),
    )

    assert not statement_context._has_complete_source_page_selection(partial)
    assert statement_context._has_complete_source_page_selection(complete)
    assert not statement_context._has_complete_source_page_selection(unknown)


def test_source_bound_identity_is_retained_in_its_multi_scope_header(monkeypatch):
    title_one = statement_context._HeaderFact(
        "statement_title", "document_title", "一月对账单", "一月对账单", 1, "page:0001", None, ("title:1",)
    )
    title_two = statement_context._HeaderFact(
        "statement_title", "document_title", "二月对账单", "二月对账单", 2, "page:0002", None, ("title:2",)
    )
    parse_result = SimpleNamespace(pages=[SimpleNamespace(page_number=1), SimpleNamespace(page_number=2)])
    monkeypatch.setattr(statement_context, "_page_header_facts", lambda _result: ({1: [title_one], 2: [title_two]}, {}))
    monkeypatch.setattr(statement_context, "_context_page_groups", lambda _result, _facts: [[1], [2]])
    identity = {
        "verification_code": {
            "raw_name": "验证编号",
            "raw_value": "VERIFY-ABC",
            "source": "canonical_evidence_atoms",
            "source_refs": [{"source_page": 2}],
            "evidence_ids": ["verify:2"],
        }
    }

    records = statement_context.build_statement_header_records(parse_result, identity)

    assert "verification_code" not in records[0]["normalized"]
    assert records[1]["normalized"]["verification_code"] == "VERIFY-ABC"


def test_canonical_identity_placeholder_and_branch_label_cannot_claim_bank_name():
    assert not statement_context._identity_detail_is_source_bound(
        "bank_name", "bank_name", "错误银行", {"source": "header.kv"}
    )
    assert not statement_context._identity_detail_is_source_bound(
        "bank_name", "开户行", "测试银行支行", {"source": "header.kv"}
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("卡号/账号Account/Card No", "account_number"),
        ("户名Account Name", "account_holder"),
        ("查询起日Query Starting Date", "period_start"),
        ("查询止日Query Ending Date", "period_end"),
        ("证件号码ID Number", "id_number"),
    ],
)
def test_bilingual_source_label_requires_two_exact_same_role_aliases(label, expected):
    assert statement_context._field_for_label(label) == expected


def test_bilingual_label_does_not_match_cross_role_concatenation():
    assert statement_context._field_for_label("账号Account Name") == ""


def test_statement_date_signature_drops_adjacent_pagination_furniture():
    raw = "2022年08月31日第7页,共8页"

    cleaned = statement_context._clean_header_value("statement_period", raw)

    assert cleaned == "2022年08月31日"
    assert statement_context._normalize_field_value("statement_period", cleaned) == "2022-08-31"
    assert statement_context._field_for_label("账单统计日期") == "statement_period"


def test_inferred_identity_cannot_enter_source_header_dataset(monkeypatch, parse_result):
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: [])
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})
    identities = {
        "bank_name": {
            "raw_name": "bank_name",
            "raw_value": "错误银行",
            "normalized_value": "错误银行",
            "source": "plugin.institutions",
        },
        "account_holder": {
            "raw_name": "客户名称",
            "raw_value": "无 2023年12月22日 用途/备注：无 对方",
            "normalized_value": "无 2023年12月22日 用途/备注：无 对方",
            "source": "canonical_evidence_atoms",
        },
    }

    assert statement_context.build_statement_header_records(parse_result, identities) == []


def test_exact_source_identity_label_remains_an_admissible_fallback(monkeypatch, parse_result):
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: [])
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})
    identities = {
        "account_holder": {
            "raw_name": "客户名称",
            "raw_value": "测试企业有限公司",
            "normalized_value": "错误的派生值",
            "source": "header.kv",
        }
    }

    [record] = statement_context.build_statement_header_records(parse_result, identities)

    assert record["normalized"]["account_holder"] == "测试企业有限公司"
    assert record["canonical_raw"]["account_holder"] == "测试企业有限公司"


def test_visible_issuer_is_derived_only_from_statement_title(monkeypatch, parse_result):
    atoms = _current_account_atoms()
    atoms[0] = {**atoms[0], "text": "上海浦东发展银行电子明细对账单"}
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["bank_name"] == "上海浦东发展银行"
    assert record["canonical_raw"]["bank_name"] == "上海浦东发展银行"
    assert record["source"]["field_sources"]["bank_name"]["raw_name"] == "statement_title_issuer"


def test_cross_scope_context_is_carried_only_after_independent_agreement():
    fact = statement_context._HeaderFact(
        "customer_number",
        "客户号:",
        "CUST-001",
        "CUST-001",
        1,
        "page:0001",
        (10.0, 10.0, 100.0, 20.0),
        ("ev:customer:1",),
    )
    repeat = statement_context._HeaderFact(
        **{**fact.__dict__, "page": 2, "page_id": "page:0002", "evidence_ids": ("ev:customer:2",)}
    )

    stable = statement_context._stable_cross_scope_facts(
        {1: [fact], 2: [repeat], 3: []},
        [[1], [2], [3]],
    )

    assert stable["customer_number"] == [fact, repeat]


def test_cross_scope_context_does_not_carry_single_or_conflicting_value():
    first = statement_context._HeaderFact(
        "account_number",
        "账号:",
        "A-001",
        "A-001",
        1,
        "page:0001",
        None,
        (),
    )
    conflict = statement_context._HeaderFact(
        **{**first.__dict__, "raw_value": "A-002", "normalized_value": "A-002", "page": 2, "page_id": "page:0002"}
    )

    assert statement_context._stable_cross_scope_facts({1: [first], 2: []}, [[1], [2]]) == {}
    assert statement_context._stable_cross_scope_facts({1: [first], 2: [conflict]}, [[1], [2]]) == {}


def test_explicit_issuer_mark_and_issue_timestamp_are_source_business_facts(monkeypatch, parse_result):
    atoms = _current_account_atoms()
    atoms.extend(
        [
            _atom(1, "平安银行（银行签章）", 500, 20, 650, 40, 2001),
            _atom(1, "开立日期:", 500, 70, 560, 84, 2002),
            _atom(1, "2024-04-24", 570, 70, 635, 84, 2003),
            _atom(1, "17:02:39", 640, 70, 690, 84, 2006),
            _atom(1, "List", 500, 80, 525, 88, 2007),
            _atom(1, "Number:", 528, 80, 565, 88, 2008),
            _atom(1, "Issuing", 570, 80, 607, 88, 2009),
            _atom(1, "Date:", 610, 80, 640, 88, 2010),
            _atom(1, "清单编号:", 500, 90, 560, 104, 2004),
            _atom(1, "JYLS240424059483", 570, 90, 680, 104, 2005),
        ]
    )
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["bank_name"] == "平安银行"
    assert record["normalized"]["issue_timestamp"] == "2024-04-24 17:02:39"
    assert record["normalized"]["list_number"] == "JYLS240424059483"
    assert "Issuing" not in str(record["raw"])


def test_unlabelled_bank_location_is_not_an_issuer_mark(monkeypatch, parse_result):
    atoms = [
        _atom(1, "平安银行", 500, 20, 560, 40, 1),
        *[atom for atom in _current_account_atoms() if atom["bbox"][1] >= 200],
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    assert statement_context.build_statement_header_records(parse_result, {}) == []


def test_ledger_field_labels_are_not_promoted_to_title_or_header_values():
    rows = [
        [_atom(1, "账单类型", 0, 20, 50, 32, 1)],
        [
            _atom(1, "柜员流水号", 0, 50, 60, 62, 2),
            _atom(1, "账户余额", 70, 50, 120, 62, 3),
            _atom(1, "摘要代码", 130, 50, 180, 62, 4),
            _atom(1, "备注", 190, 50, 220, 62, 5),
        ],
    ]

    assert statement_context._title_fact(rows, 1) is None
    assert statement_context._facts_from_row(rows[1], 1) == []


@pytest.mark.parametrize(
    ("title", "bank"),
    [
        ("交通银行上海市分行明细对账单", "交通银行"),
        ("江苏银行交易明细", "江苏银行"),
        ("中国邮政储蓄银行账户交易明细", "中国邮政储蓄银行"),
    ],
)
def test_visible_title_issuer_supports_short_and_long_bank_names(title, bank):
    fact = statement_context._HeaderFact(
        "statement_title",
        "document_title",
        title,
        title,
        1,
        "page:0001",
        None,
        ("ev:title",),
    )

    issuer = statement_context._bank_name_from_title_fact(fact)

    assert issuer is not None
    assert issuer.normalized_value == bank


def _context_header(record_id: str, page_range: list[int], **context_fields: str) -> dict:
    normalized = {
        "account_holder": f"holder:{record_id}",
        "statement_title": "source statement",
        **context_fields,
    }
    canonical_raw = {key: value.replace("-", "") for key, value in context_fields.items()}
    field_sources = {
        key: {
            "source": "canonical_evidence_atoms",
            "source_refs": [
                {
                    "source_page": page_range[0],
                    "bbox": [10.0, 20.0, 100.0, 30.0],
                }
            ],
            "evidence_ids": [f"header:{record_id}:{key}"],
        }
        for key in context_fields
    }
    field_sources["account_holder"] = {
        "source": "canonical_evidence_atoms",
        "source_refs": [
            {
                "source_page": page_range[0],
                "bbox": [10.0, 40.0, 100.0, 50.0],
            }
        ],
        "evidence_ids": [f"header:{record_id}:holder"],
    }
    return {
        "record_id": record_id,
        "normalized": normalized,
        "canonical_raw": canonical_raw,
        "raw": {"source_context": dict(context_fields)},
        "source": {
            "page_range": page_range,
            "field_sources": field_sources,
        },
    }


def _context_row(index: int, page: int, date: str) -> dict:
    return {
        "record_id": f"records:r{index:06d}",
        "normalized": {"date": date},
        "canonical_raw": {"date": date.replace("-", "")},
        "raw": {"交易日期": date.replace("-", "")},
        "source": {
            "source_page": page,
            "page_range": [page, page],
            "field_sources": {"date": {"source": "transaction_row"}},
        },
    }


def test_incoherent_scope_does_not_inherit_either_period_bound():
    header = _context_header(
        "statement_header:r000001",
        [1, 2],
        period_start="2023-03-10",
        period_end="2023-09-10",
    )
    rows = [
        _context_row(1, 1, "2023-04-01"),
        _context_row(2, 2, "2022-09-11"),
    ]

    attached = statement_context.attach_statement_context(rows, [header])

    assert header["normalized"]["period_start"] == "2023-03-10"
    assert header["normalized"]["period_end"] == "2023-09-10"
    for row in attached:
        assert row["normalized"]["statement_header_id"] == header["record_id"]
        assert row["normalized"]["account_holder"] == "holder:statement_header:r000001"
        assert "period_start" not in row["normalized"]
        assert "period_end" not in row["normalized"]
        assert "period_start" not in row["canonical_raw"]
        assert "period_end" not in row["canonical_raw"]
        assert "period_start" not in row["source"]["field_sources"]
        assert "period_end" not in row["source"]["field_sources"]


def test_timestamp_fallback_participates_in_scope_coherence():
    header = _context_header(
        "statement_header:r000001",
        [1, 1],
        period_start="2023-03-10",
        period_end="2023-09-10",
    )
    row = _context_row(1, 1, "2023-04-01")
    row["normalized"] = {"date": "not-a-date", "timestamp": "2022-09-11 11:27:48"}

    [attached] = statement_context.attach_statement_context([row], [header])

    assert attached["normalized"]["statement_header_id"] == header["record_id"]
    assert attached["normalized"]["date"] == "not-a-date"
    assert "period_start" not in attached["normalized"]
    assert "period_end" not in attached["normalized"]


@pytest.mark.parametrize(
    ("period_fields", "row_date"),
    [
        ({"period_start": "not-a-date", "period_end": "2023-09-10"}, "2023-04-01"),
        ({"period_start": "2023-09-10", "period_end": "2023-03-10"}, "2023-04-01"),
        ({"period_start": "2023-03-10"}, "2023-03-09"),
        ({"period_end": "2023-09-10"}, "2023-09-11"),
    ],
)
def test_invalid_or_violated_period_scope_does_not_inherit_bounds(period_fields, row_date):
    header = _context_header("statement_header:r000001", [1, 1], **period_fields)

    [attached] = statement_context.attach_statement_context([_context_row(1, 1, row_date)], [header])

    assert attached["normalized"]["statement_header_id"] == header["record_id"]
    assert "period_start" not in attached["normalized"]
    assert "period_end" not in attached["normalized"]


def test_coherent_scope_still_inherits_period_values_and_provenance():
    header = _context_header(
        "statement_header:r000001",
        [1, 2],
        period_start="2023-03-10",
        period_end="2023-09-10",
    )
    rows = [
        _context_row(1, 1, "2023-03-10"),
        _context_row(2, 2, "2023-09-10"),
    ]

    attached = statement_context.attach_statement_context(rows, [header])

    for row in attached:
        assert row["normalized"]["period_start"] == "2023-03-10"
        assert row["normalized"]["period_end"] == "2023-09-10"
        assert row["canonical_raw"]["period_start"] == "20230310"
        assert row["canonical_raw"]["period_end"] == "20230910"
        assert row["source"]["field_sources"]["period_start"] == header["source"]["field_sources"][
            "period_start"
        ]
        assert row["source"]["field_sources"]["period_end"] == header["source"]["field_sources"][
            "period_end"
        ]


def test_segmented_open_period_bounds_are_checked_independently():
    start_header = _context_header(
        "statement_header:r000001",
        [1, 1],
        period_start="2023-01-01",
    )
    end_header = _context_header(
        "statement_header:r000002",
        [2, 2],
        period_end="2023-06-30",
    )

    attached = statement_context.attach_statement_context(
        [_context_row(1, 1, "2023-02-01"), _context_row(2, 2, "2023-05-31")],
        [start_header, end_header],
    )

    assert attached[0]["normalized"]["statement_header_id"] == start_header["record_id"]
    assert attached[0]["normalized"]["period_start"] == "2023-01-01"
    assert "period_end" not in attached[0]["normalized"]
    assert attached[1]["normalized"]["statement_header_id"] == end_header["record_id"]
    assert "period_start" not in attached[1]["normalized"]
    assert attached[1]["normalized"]["period_end"] == "2023-06-30"


def test_cutoff_only_header_does_not_manufacture_transaction_period():
    header = _context_header(
        "statement_header:r000001",
        [1, 1],
        statement_cutoff_date="2023-06-30",
    )

    [attached] = statement_context.attach_statement_context([_context_row(1, 1, "2023-05-31")], [header])

    assert attached["normalized"]["statement_header_id"] == header["record_id"]
    assert attached["normalized"]["account_holder"] == "holder:statement_header:r000001"
    assert "period_start" not in attached["normalized"]
    assert "period_end" not in attached["normalized"]
    assert header["normalized"]["statement_cutoff_date"] == "2023-06-30"


def _residual_header(
    *,
    debit_count: int = 3,
    debit_total: str = "31.00",
    credit_count: int = 1,
    credit_total: str = "5.00",
) -> dict:
    terminal_fields = {
        "debit_count": debit_count,
        "debit_total": debit_total,
        "credit_count": credit_count,
        "credit_total": credit_total,
    }
    return {
        "record_id": "statement_header:r000001",
        "normalized": dict(terminal_fields),
        "canonical_raw": {key: str(value) for key, value in terminal_fields.items()},
        "raw": {"本月累计": dict(terminal_fields)},
        "source": {
            "source": "statement_header_scope",
            "page_range": [1, 3],
            "field_sources": {
                key: {
                    "raw_name": f"issuer_{key}",
                    "source": "canonical_evidence_atoms",
                    "source_refs": [
                        {
                            "source": "canonical_evidence_atoms",
                            "source_page": 3,
                            "bbox": [10.0, 500.0, 100.0, 515.0],
                        }
                    ],
                    "evidence_ids": [f"aggregate:{key}"],
                }
                for key in terminal_fields
            },
        },
        "confidence": 1.0,
    }


def _residual_records(*, first_carry: str = "89.00", second_carry: str | None = None) -> list[dict]:
    page_two_balance = statement_context._as_decimal(first_carry) + statement_context._as_decimal("5.00")
    page_three_carry = second_carry or str(page_two_balance)
    page_three_balance = statement_context._as_decimal(page_three_carry) - statement_context._as_decimal("20.00")
    values = [
        (1, "expense", "10.00", "90.00"),
        (2, "income", "5.00", str(page_two_balance)),
        (3, "expense", "20.00", str(page_three_balance)),
    ]
    return [
        {
            "record_id": f"bank:r{index:06d}",
            "normalized": {
                "statement_header_id": "statement_header:r000001",
                "direction": direction,
                "amount": amount,
                "balance": balance,
            },
            "canonical_raw": {"direction": direction, "amount": amount, "balance": balance},
            "raw": {},
            "source": {
                "source": "physical_table",
                "source_page": page,
                "page_range": [page, page],
                "bbox": [10.0, 100.0 + index * 20.0, 400.0, 115.0 + index * 20.0],
                "evidence_ids": [f"row:{index}"],
            },
        }
        for index, (page, direction, amount, balance) in enumerate(values, start=1)
    ]


def _physical_census_parse_result(records: list[dict]) -> tuple[SimpleNamespace, list[dict]]:
    headers = ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"]
    carry_values = ["100.00", "89.00", "94.00"]
    transaction_values = [
        ["2023-03-01", "支出", "10.00", "", "90.00"],
        ["2023-03-02", "收入", "", "5.00", "94.00"],
        ["2023-03-03", "支出", "20.00", "", "74.00"],
    ]
    physical_records = deepcopy(records)
    pages = []
    for page_number, (carry_value, transaction) in enumerate(
        zip(carry_values, transaction_values, strict=True),
        start=1,
    ):
        table_id = f"physical:p{page_number:04d}:t0001"

        def physical_row(values: list[str], source_row_index: int, y0: float) -> SimpleNamespace:
            cells = []
            for column, value in enumerate(values):
                source_ref = {
                    "source": "canonical_physical_table",
                    "page": page_number,
                    "table_id": table_id,
                    "row": source_row_index,
                    "raw_row": source_row_index + 1,
                    "col": column,
                }
                cells.append(
                    SimpleNamespace(
                        text=value,
                        cleaned=None,
                        bbox=[column * 100.0, y0, column * 100.0 + 80.0, y0 + 14.0],
                        evidence_ids=(
                            [f"ev:{page_number:04d}:r{source_row_index:04d}:c{column:04d}"]
                            if value
                            else []
                        ),
                        source_cell_refs=[source_ref],
                    )
                )
            return SimpleNamespace(
                cells=cells,
                row_type="data",
                source_page=page_number,
                source_physical_id=table_id,
                source_row_index=source_row_index,
                source_cell_refs=[],
            )

        carry_row = physical_row(["", "承前", "", "", carry_value], 0, 80.0)
        transaction_row = physical_row(transaction, 1, 120.0)
        table = SimpleNamespace(
            table_id=table_id,
            headers=headers,
            rows=[carry_row, transaction_row],
            metadata={"preserve_headers": True},
        )
        pages.append(
            SimpleNamespace(
                page_number=page_number,
                source_page_number=page_number,
                tables=[table],
                texts=[],
            )
        )
        transaction_cells = transaction_row.cells
        physical_records[page_number - 1]["source"] = {
            "source": "canonical_physical_table",
            "source_page": page_number,
            "page_range": [page_number, page_number],
            "table_id": table_id,
            "source_row_index": 1,
            "bbox": [0.0, 120.0, 480.0, 134.0],
            "evidence_ids": [
                evidence_id
                for cell in transaction_cells
                for evidence_id in cell.evidence_ids
            ],
            "source_cell_refs": [
                deepcopy(ref)
                for cell in transaction_cells
                for ref in cell.source_cell_refs
            ],
        }
        source_date = transaction[0]
        physical_records[page_number - 1]["normalized"]["date"] = source_date
        physical_records[page_number - 1]["canonical_raw"]["date"] = source_date
    parse_result = SimpleNamespace(
        pages=pages,
        parser_info=SimpleNamespace(
            page_count=3,
            options={"source_page_count": 3, "selected_source_pages": [1, 2, 3]},
            structure={
                "table_extraction": "full",
                "physical_table_count": 3,
                "table_reconstruction_gate": {
                    "applicable": True,
                    "passed": True,
                    "candidate_count": 3,
                    "physical_table_count": 3,
                },
            },
        ),
    )
    return parse_result, physical_records


def _patch_residual_evidence(
    monkeypatch,
    *,
    first_carry: str = "89.00",
    second_carry: str = "94.00",
    anchor_pages: list[int] | None = None,
) -> None:
    def carry_fact(page: int, value: str) -> statement_context._HeaderFact:
        return statement_context._HeaderFact(
            "brought_forward_balance",
            "承前余额",
            value,
            value,
            page,
            f"page:{page:04d}",
            (20.0, 60.0, 120.0, 75.0),
            (f"carry:{page}",),
        )

    monkeypatch.setattr(
        statement_context,
        "_page_header_facts",
        lambda _result: ({1: [], 2: [carry_fact(2, first_carry)], 3: [carry_fact(3, second_carry)]}, {}),
    )
    pages = anchor_pages or [1, 2, 3]
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {
            "expected_rows": len(pages),
            "source": "positioned_date_anchors",
            "confidence": 0.80,
            "row_sources": [{"source_page": page} for page in pages],
            "pages": pages,
        },
    )


def _reconcile_residual(monkeypatch, header: dict, records: list[dict], **evidence) -> dict:
    _patch_residual_evidence(monkeypatch, **evidence)
    [result] = statement_context.reconcile_source_unitemized_residuals(
        SimpleNamespace(),
        records,
        [header],
        source_route="digital",
        selected_source="canonical_table",
    )
    return result


def test_source_unitemized_debit_residual_is_header_only_and_source_bound(monkeypatch):
    header = _residual_header()
    records = _residual_records()
    original_header = deepcopy(header)
    original_records = deepcopy(records)

    result = _reconcile_residual(monkeypatch, header, records)

    assert result["normalized"]["source_unitemized_debit_count"] == 1
    assert result["normalized"]["source_unitemized_debit_amount"] == "1.00"
    assert "source_unitemized_debit_count" not in result["canonical_raw"]
    assert "source_unitemized_debit_count" not in result["raw"]
    assert header == original_header
    assert records == original_records
    provenance = result["source"]["field_sources"]["source_unitemized_debit_amount"]
    assert provenance["derivation"] == "source_unitemized_reconciliation"
    assert {ref["source_page"] for ref in provenance["source_refs"]} == {1, 2, 3}
    assert provenance["independent_row_anchors"]["page_counts"] == {"1": 1, "2": 1, "3": 1}
    assert not {"date", "counter_party", "summary"} & result["normalized"].keys()


def test_source_unitemized_residuals_are_isolated_per_statement_scope(monkeypatch):
    first_header = _residual_header()
    second_header = _residual_header(debit_count=2, debit_total="30.00")
    second_header["record_id"] = "statement_header:r000002"
    second_header["source"]["page_range"] = [4, 6]
    for detail in second_header["source"]["field_sources"].values():
        detail["source_refs"][0]["source_page"] = 6

    first_records = _residual_records()
    second_records = _residual_records(first_carry="90.00", second_carry="95.00")
    for index, record in enumerate(second_records, start=4):
        record["record_id"] = f"bank:r{index:06d}"
        record["normalized"]["statement_header_id"] = second_header["record_id"]
        source_page = int(record["source"]["source_page"]) + 3
        record["source"]["source_page"] = source_page
        record["source"]["page_range"] = [source_page, source_page]

    def carry_fact(page: int, value: str) -> statement_context._HeaderFact:
        return statement_context._HeaderFact(
            "brought_forward_balance",
            "承前余额",
            value,
            value,
            page,
            f"page:{page:04d}",
            (20.0, 60.0, 120.0, 75.0),
            (f"carry:{page}",),
        )

    monkeypatch.setattr(
        statement_context,
        "_page_header_facts",
        lambda _result: (
            {
                1: [],
                2: [carry_fact(2, "89.00")],
                3: [carry_fact(3, "94.00")],
                4: [],
                5: [carry_fact(5, "90.00")],
                6: [carry_fact(6, "95.00")],
            },
            {},
        ),
    )
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {
            "expected_rows": 6,
            "source": "positioned_date_anchors",
            "confidence": 0.80,
            "row_sources": [{"source_page": page} for page in range(1, 7)],
            "pages": list(range(1, 7)),
        },
    )

    first_result, second_result = statement_context.reconcile_source_unitemized_residuals(
        SimpleNamespace(),
        [*first_records, *second_records],
        [first_header, second_header],
        source_route="digital",
        selected_source="canonical_table",
    )

    assert first_result["normalized"]["source_unitemized_debit_count"] == 1
    assert first_result["normalized"]["source_unitemized_debit_amount"] == "1.00"
    first_provenance = first_result["source"]["field_sources"]["source_unitemized_debit_amount"]
    assert first_provenance["independent_row_anchors"]["page_counts"] == {"1": 1, "2": 1, "3": 1}
    assert not any(key.startswith("source_unitemized_") for key in second_result["normalized"])


@pytest.mark.parametrize(
    "invalid_contract",
    [
        "canonical_count_mismatch",
        "canonical_total_mismatch",
        "missing_count_evidence",
        "missing_total_bbox",
        "indirect_count_source",
        "normalized_only_total",
        "nonterminal_count_ref",
        "mismatched_count_ref_source",
    ],
)
def test_source_unitemized_requires_exact_direct_terminal_aggregates(monkeypatch, invalid_contract):
    header = _residual_header()
    if invalid_contract == "canonical_count_mismatch":
        header["canonical_raw"]["debit_count"] = "4"
    elif invalid_contract == "canonical_total_mismatch":
        header["canonical_raw"]["debit_total"] = "32.00"
    elif invalid_contract == "missing_count_evidence":
        header["source"]["field_sources"]["debit_count"]["evidence_ids"] = []
    elif invalid_contract == "missing_total_bbox":
        header["source"]["field_sources"]["debit_total"]["source_refs"][0].pop("bbox")
    elif invalid_contract == "indirect_count_source":
        header["source"]["field_sources"]["debit_count"]["source"] = "derived_summary"
    elif invalid_contract == "normalized_only_total":
        header["source"]["field_sources"]["debit_total"]["normalized_only"] = True
    elif invalid_contract == "nonterminal_count_ref":
        header["source"]["field_sources"]["debit_count"]["source_refs"][0]["source_page"] = 2
    else:
        header["source"]["field_sources"]["debit_count"]["source_refs"][0]["source"] = "page_headers"

    result = _reconcile_residual(monkeypatch, header, _residual_records())

    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


def test_source_unitemized_credit_residual_is_symmetric(monkeypatch):
    header = _residual_header(debit_count=2, debit_total="30.00", credit_count=2, credit_total="6.00")
    records = _residual_records(first_carry="91.00")

    result = _reconcile_residual(
        monkeypatch,
        header,
        records,
        first_carry="91.00",
        second_carry="96.00",
    )

    assert result["normalized"]["source_unitemized_credit_count"] == 1
    assert result["normalized"]["source_unitemized_credit_amount"] == "1.00"
    assert "source_unitemized_debit_count" not in result["normalized"]


def test_source_unitemized_multiple_carry_gaps_are_summed(monkeypatch):
    header = _residual_header(debit_count=4, debit_total="33.00")
    records = _residual_records(first_carry="89.00", second_carry="92.00")

    result = _reconcile_residual(
        monkeypatch,
        header,
        records,
        first_carry="89.00",
        second_carry="92.00",
    )

    assert result["normalized"]["source_unitemized_debit_count"] == 2
    assert result["normalized"]["source_unitemized_debit_amount"] == "3.00"
    boundaries = result["source"]["field_sources"]["source_unitemized_debit_amount"]["carry_boundaries"]
    assert [boundary["amount"] for boundary in boundaries] == ["1.00", "2.00"]


@pytest.mark.parametrize(
    ("header", "first_carry", "second_carry"),
    [
        (_residual_header(debit_count=2, debit_total="30.00"), "90.00", "95.00"),
        (_residual_header(debit_total="32.00"), "89.00", "94.00"),
        (_residual_header(debit_count=2), "89.00", "94.00"),
        (_residual_header(), "91.00", "96.00"),
    ],
    ids=["zero_residual", "amount_contradiction", "count_contradiction", "direction_contradiction"],
)
def test_source_unitemized_residual_contradictions_fail_closed(
    monkeypatch,
    header,
    first_carry,
    second_carry,
):
    records = _residual_records(first_carry=first_carry, second_carry=second_carry)

    result = _reconcile_residual(
        monkeypatch,
        header,
        records,
        first_carry=first_carry,
        second_carry=second_carry,
    )

    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


@pytest.mark.parametrize(
    ("first_carry", "expected"),
    [("88.999999", True), ("88.99", False)],
    ids=["subcent_decimal_noise", "material_cent_difference"],
)
def test_source_unitemized_residual_uses_strict_subcent_tolerance(monkeypatch, first_carry, expected):
    records = _residual_records(first_carry=first_carry)
    second_carry = str(statement_context._as_decimal(first_carry) + statement_context._as_decimal("5.00"))

    result = _reconcile_residual(
        monkeypatch,
        _residual_header(),
        records,
        first_carry=first_carry,
        second_carry=second_carry,
    )

    assert ("source_unitemized_debit_count" in result["normalized"]) is expected


def test_source_unitemized_requires_row_local_pages(monkeypatch):
    records = _residual_records()
    records[0]["source"].pop("source_page")
    records[0]["source"]["page_range"] = [1, 3]

    result = _reconcile_residual(monkeypatch, _residual_header(), records)

    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


def test_source_unitemized_rejects_extra_independent_row_anchor(monkeypatch):
    result = _reconcile_residual(
        monkeypatch,
        _residual_header(),
        _residual_records(),
        anchor_pages=[1, 1, 2, 3],
    )

    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


def test_source_unitemized_rejects_anchor_dependent_selected_candidate(monkeypatch):
    _patch_residual_evidence(monkeypatch)

    [result] = statement_context.reconcile_source_unitemized_residuals(
        SimpleNamespace(),
        _residual_records(),
        [_residual_header()],
        source_route="digital",
        selected_source="canonical_evidence_table",
    )

    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


def test_source_unitemized_uses_exact_canonical_physical_row_census(monkeypatch):
    parse_result, records = _physical_census_parse_result(_residual_records())
    original_records = deepcopy(records)
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {},
    )
    monkeypatch.setattr(statement_context, "_group_atoms", lambda _result: {1: [], 2: [], 3: []})

    census = statement_context._canonical_physical_row_census_evidence(
        parse_result,
        records,
        source_route="digital",
        selected_source="canonical_table",
    )
    [result] = statement_context.reconcile_source_unitemized_residuals(
        parse_result,
        records,
        [_residual_header()],
        source_route="digital",
        selected_source="canonical_table",
    )

    assert census["expected_rows"] == 3
    assert census["census"]["exact_lineage_match"] is True
    assert census["census"]["exact_semantic_match"] is True
    assert census["census"]["consistent_physical_order"] is True
    assert census["census"]["page_counts"] == {"1": 1, "2": 1, "3": 1}
    first_row_source = census["row_sources"][0]
    assert first_row_source["bbox"] == [0.0, 120.0, 480.0, 134.0]
    assert first_row_source["source_headers"] == ["交易日期", "摘要", "借方发生额", "贷方发生额", "余额"]
    assert first_row_source["active_amount_role"] == "debit_amount"
    assert first_row_source["header_roles"] == {
        "date": {"column": 0, "header": "交易日期"},
        "debit_amount": {"column": 2, "header": "借方发生额"},
        "credit_amount": {"column": 3, "header": "贷方发生额"},
        "balance": {"column": 4, "header": "余额"},
    }
    assert first_row_source["source_roles"] == {
        "date": {"column": 0, "header": "交易日期"},
        "amount": {"column": 2, "header": "借方发生额", "direction": "expense"},
        "balance": {"column": 4, "header": "余额"},
    }
    assert set(first_row_source["required_evidence_ids"]) == {
        "ev:0001:r0001:c0000",
        "ev:0001:r0001:c0002",
        "ev:0001:r0001:c0004",
    }
    assert {ref["col"] for ref in first_row_source["required_source_cell_refs"]} == {0, 2, 4}
    assert first_row_source["source_cell_ref_owner"] == "canonical_physical_table"
    assert result["normalized"]["source_unitemized_debit_count"] == 1
    assert result["normalized"]["source_unitemized_debit_amount"] == "1.00"
    assert "source_unitemized_debit_count" not in result["canonical_raw"]
    assert "source_unitemized_debit_count" not in result["raw"]
    provenance = result["source"]["field_sources"]["source_unitemized_debit_amount"]
    assert provenance["independent_row_anchors"]["source"] == "canonical_physical_table_row_census"
    assert provenance["independent_row_anchors"]["page_counts"] == {"1": 1, "2": 1, "3": 1}
    assert provenance["independent_row_anchors"]["exact_semantic_match"] is True
    scoped_first_row = provenance["independent_row_anchors"]["row_sources"][0]
    assert scoped_first_row["required_source_cell_refs"] == first_row_source["required_source_cell_refs"]
    assert scoped_first_row["required_evidence_ids"] == first_row_source["required_evidence_ids"]
    assert scoped_first_row["header_roles"] == first_row_source["header_roles"]
    assert scoped_first_row["source_roles"] == first_row_source["source_roles"]
    carry_ref = next(
        ref
        for ref in provenance["source_refs"]
        if ref["role"] == "carry_boundary.brought_forward_balance"
    )
    assert carry_ref["table_id"] == "physical:p0002:t0001"
    assert carry_ref["source_row_index"] == 0
    assert {ref["col"] for ref in carry_ref["source_cell_refs"]} == {1, 4}
    assert records == original_records


def test_canonical_physical_contract_accepts_container_inherited_cell_ref_owner():
    parse_result, records = _physical_census_parse_result(_residual_records())
    for page in parse_result.pages:
        for table in page.tables:
            for row in table.rows:
                for cell in row.cells:
                    for ref in cell.source_cell_refs:
                        ref.pop("source", None)
    for record in records:
        for ref in record["source"]["source_cell_refs"]:
            ref.pop("source", None)

    census = statement_context._canonical_physical_row_census_evidence(
        parse_result,
        records,
        source_route="digital",
        selected_source="canonical_table",
    )
    carry_facts = statement_context._physical_brought_forward_facts(parse_result)

    assert census["expected_rows"] == 3
    assert census["census"]["exact_lineage_match"] is True
    assert all(row["source_cell_ref_owner"] == "canonical_physical_table" for row in census["row_sources"])
    assert all(
        "source" not in ref
        for row in census["row_sources"]
        for ref in row["required_source_cell_refs"]
    )
    assert set(carry_facts) == {1, 2, 3}


def test_exact_physical_carry_provenance_wins_duplicate_atom_fact(monkeypatch):
    parse_result, _records = _physical_census_parse_result(_residual_records())
    duplicate_atoms = [
        _atom(2, "承前", 100.0, 80.0, 150.0, 94.0, 1),
        _atom(2, "89.00", 160.0, 80.0, 210.0, 94.0, 2),
    ]
    assert any(
        fact.field_key == "brought_forward_balance"
        for fact in statement_context._facts_from_row(duplicate_atoms, 2)
    )
    monkeypatch.setattr(statement_context, "_group_atoms", lambda _result: {2: duplicate_atoms})

    facts_by_page, _lines = statement_context._page_header_facts(parse_result)
    carry_facts = [
        fact for fact in facts_by_page[2] if fact.field_key == "brought_forward_balance"
    ]

    assert len(carry_facts) == 1
    [carry_fact] = carry_facts
    assert carry_fact.source_kind == "canonical_physical_table"
    assert carry_fact.derivation == "physical_brought_forward_row"
    source = statement_context._field_source(carry_facts)
    assert source["source_refs"][0]["table_id"] == "physical:p0002:t0001"
    assert {ref["col"] for ref in source["source_refs"][0]["source_cell_refs"]} == {1, 4}


@pytest.mark.parametrize(
    "invalid_contract",
    [
        "wrong_route",
        "wrong_selected_source",
        "incomplete_pages",
        "failed_gate",
        "missing_lineage",
        "altered_direction",
        "altered_amount",
        "physical_raw_row_mismatch",
        "physical_row_type",
        "physical_cell_ref_source",
        "emitted_cell_ref_source",
    ],
)
def test_canonical_physical_row_census_fails_closed(monkeypatch, invalid_contract):
    parse_result, records = _physical_census_parse_result(_residual_records())
    source_route = "digital"
    selected_source = "canonical_table"
    if invalid_contract == "wrong_route":
        source_route = "scanned"
    elif invalid_contract == "wrong_selected_source":
        selected_source = "pipe_text"
    elif invalid_contract == "incomplete_pages":
        parse_result.parser_info.options["selected_source_pages"] = [1, 2]
    elif invalid_contract == "failed_gate":
        parse_result.parser_info.structure["table_reconstruction_gate"]["passed"] = False
    elif invalid_contract == "altered_direction":
        records[0]["normalized"]["direction"] = "income"
    elif invalid_contract == "altered_amount":
        records[0]["normalized"]["amount"] = "11.00"
    elif invalid_contract == "physical_raw_row_mismatch":
        transaction_row = parse_result.pages[0].tables[0].rows[1]
        transaction_row.cells[2].source_cell_refs[0]["raw_row"] = 99
    elif invalid_contract == "physical_row_type":
        parse_result.pages[0].tables[0].rows[1].row_type = "header"
    elif invalid_contract == "physical_cell_ref_source":
        parse_result.pages[0].tables[0].rows[1].cells[0].source_cell_refs[0]["source"] = "page_headers"
    elif invalid_contract == "emitted_cell_ref_source":
        records[0]["source"]["source_cell_refs"][0]["source"] = "page_headers"
    else:
        records[0]["source"].pop("source_cell_refs")
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {},
    )
    monkeypatch.setattr(statement_context, "_group_atoms", lambda _result: {1: [], 2: [], 3: []})

    evidence = statement_context._canonical_physical_row_census_evidence(
        parse_result,
        records,
        source_route=source_route,
        selected_source=selected_source,
    )
    [result] = statement_context.reconcile_source_unitemized_residuals(
        parse_result,
        records,
        [_residual_header()],
        source_route=source_route,
        selected_source=selected_source,
    )

    assert evidence == {}
    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


def test_source_unitemized_rejects_carry_after_first_physical_transaction(monkeypatch):
    parse_result, records = _physical_census_parse_result(_residual_records())
    carry_row = parse_result.pages[1].tables[0].rows[0]
    carry_row.source_row_index = 2
    for cell in carry_row.cells:
        cell.source_cell_refs[0]["row"] = 2
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {},
    )
    monkeypatch.setattr(statement_context, "_group_atoms", lambda _result: {1: [], 2: [], 3: []})

    census = statement_context._canonical_physical_row_census_evidence(
        parse_result,
        records,
        source_route="digital",
        selected_source="canonical_table",
    )
    [result] = statement_context.reconcile_source_unitemized_residuals(
        parse_result,
        records,
        [_residual_header()],
        source_route="digital",
        selected_source="canonical_table",
    )

    assert census["expected_rows"] == 3
    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


@pytest.mark.parametrize("carry_raw_row", [0, 3], ids=["inconsistent_offset", "after_transaction"])
def test_source_unitemized_rejects_inconsistent_physical_carry_raw_order(monkeypatch, carry_raw_row):
    parse_result, records = _physical_census_parse_result(_residual_records())
    carry_row = parse_result.pages[1].tables[0].rows[0]
    for cell in carry_row.cells:
        cell.source_cell_refs[0]["raw_row"] = carry_raw_row
    monkeypatch.setattr(
        statement_context,
        "_cached_independent_row_anchor_evidence",
        lambda _result, source_route: {},
    )
    monkeypatch.setattr(statement_context, "_group_atoms", lambda _result: {1: [], 2: [], 3: []})

    [result] = statement_context.reconcile_source_unitemized_residuals(
        parse_result,
        records,
        [_residual_header()],
        source_route="digital",
        selected_source="canonical_table",
    )

    assert not any(key.startswith("source_unitemized_") for key in result["normalized"])


@pytest.mark.parametrize(
    "facts",
    [
        [
            {"source_page": 1, "table_id": "pt", "source_row_index": 0, "raw_row": 1, "bbox": [0, 20, 10, 30]},
            {"source_page": 1, "table_id": "pt", "source_row_index": 1, "raw_row": 3, "bbox": [0, 40, 10, 50]},
        ],
        [
            {"source_page": 1, "table_id": "pt", "source_row_index": 0, "raw_row": 2, "bbox": [0, 20, 10, 30]},
            {"source_page": 1, "table_id": "pt", "source_row_index": 1, "raw_row": 1, "bbox": [0, 40, 10, 50]},
        ],
        [
            {"source_page": 1, "table_id": "pt", "source_row_index": 0, "raw_row": 1, "bbox": [0, 40, 10, 50]},
            {"source_page": 1, "table_id": "pt", "source_row_index": 1, "raw_row": 2, "bbox": [0, 20, 10, 30]},
        ],
    ],
    ids=["inconsistent_offset", "reversed_raw_rows", "reversed_geometry"],
)
def test_physical_row_order_certificate_fails_closed(facts):
    assert statement_context._physical_row_order_is_consistent(facts) is False


def test_physical_row_order_certificate_accepts_jointly_ordered_rows():
    facts = [
        {"source_page": 1, "table_id": "pt", "source_row_index": 0, "raw_row": 1, "bbox": [0, 20, 10, 30]},
        {"source_page": 1, "table_id": "pt", "source_row_index": 1, "raw_row": 2, "bbox": [0, 40, 10, 50]},
    ]

    assert statement_context._physical_row_order_is_consistent(facts) is True


def test_physical_row_order_certificate_accepts_blank_row_spanning_geometry():
    facts = [
        {
            "source_page": 2,
            "table_id": "pt_2_0",
            "source_row_index": 1,
            "raw_row": 2,
            "bbox": [13.98, 119.0031, 766.92, 527.64],
        },
        {
            "source_page": 2,
            "table_id": "pt_2_0",
            "source_row_index": 2,
            "raw_row": 3,
            "bbox": [13.98, 134.28, 718.656, 150.0],
        },
    ]

    assert statement_context._physical_row_order_is_consistent(facts) is True


def test_terminal_visible_record_uses_physical_order_not_emitted_order():
    early = {
        "record_id": "early",
        "source": {
            "source_page": 1,
            "table_id": "pt_1_0",
            "source_row_index": 1,
            "bbox": [0.0, 100.0, 400.0, 114.0],
        },
    }
    late = {
        "record_id": "late",
        "source": {
            "source_page": 1,
            "table_id": "pt_1_0",
            "source_row_index": 9,
            "bbox": [0.0, 300.0, 400.0, 314.0],
        },
    }

    assert statement_context._terminal_visible_record([late, early]) is late


@pytest.mark.parametrize("invalid_contract", ["raw_row_order", "bbox_order"])
def test_terminal_visible_record_rejects_disagreeing_canonical_physical_orders(invalid_contract):
    def physical_record(record_id: str, source_row_index: int, raw_row: int, y0: float) -> dict:
        table_id = "physical:p0001:t0001"
        return {
            "record_id": record_id,
            "source": {
                "source": "canonical_physical_table",
                "source_page": 1,
                "page_range": [1, 1],
                "table_id": table_id,
                "source_row_index": source_row_index,
                "bbox": [0.0, y0, 400.0, y0 + 14.0],
                "evidence_ids": [f"evidence:{record_id}"],
                "source_cell_refs": [
                    {
                        "source": "canonical_physical_table",
                        "page": 1,
                        "table_id": table_id,
                        "row": source_row_index,
                        "raw_row": raw_row,
                        "col": column,
                    }
                    for column in (0, 2, 4)
                ],
            },
        }

    early = physical_record("early", 1, 2, 100.0)
    late = physical_record("late", 2, 3, 200.0)
    assert statement_context._terminal_visible_record([late, early]) is late
    if invalid_contract == "raw_row_order":
        for ref in late["source"]["source_cell_refs"]:
            ref["raw_row"] = 2
    else:
        late["source"]["bbox"] = [0.0, 50.0, 400.0, 64.0]

    assert statement_context._terminal_visible_record([late, early]) is None


def test_detached_currency_in_opening_branch_metadata_row_is_source_bound(monkeypatch, parse_result):
    atoms = [
        _atom(1, "交通银行明细对账单", 200, 20, 380, 38, 1),
        _atom(1, "开户机构：", 15.95, 55, 65.65, 68, 2),
        _atom(1, "交通银行厦门金尚支行", 87.25, 55, 186.75, 68, 3),
        _atom(1, "人民币", 255.85, 55, 285.7, 68, 4),
        _atom(1, "年份：", 324.6, 55, 354.35, 68, 5),
        _atom(1, "2022", 393.35, 55, 413.47, 68, 6),
        _atom(1, "02", 463.55, 55, 473.57, 68, 7),
        _atom(1, "页码：", 527.75, 55, 557.5, 68, 8),
        _atom(1, "1-1", 591.95, 55, 607.02, 68, 9),
        _atom(1, "交易日期", 20, 100, 70, 114, 10),
        _atom(1, "交易金额", 90, 100, 140, 114, 11),
        _atom(1, "余额", 160, 100, 185, 114, 12),
        _atom(1, "摘要", 205, 100, 230, 114, 13),
        _atom(1, "2022-02-01", 20, 130, 85, 144, 14),
        _atom(1, "10.00", 90, 130, 125, 144, 15),
        _atom(1, "1010.00", 160, 130, 210, 144, 16),
        _atom(1, "测试交易", 220, 130, 280, 144, 17),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["branch_name"] == "交通银行厦门金尚支行"
    assert record["canonical_raw"]["branch_name"] == "交通银行厦门金尚支行"
    assert record["normalized"]["currency"] == "CNY"
    assert record["canonical_raw"]["currency"] == "人民币"
    assert record["raw"]["unlabelled_currency"] == "人民币"
    assert record["raw"]["页码："] == "1-1"
    assert "source_header_page_label" not in record["normalized"]
    assert "source_header_page_label" not in record["canonical_raw"]
    assert record["source"]["field_sources"]["branch_name"]["evidence_ids"] == [
        "ev:0001:text:000002",
        "ev:0001:text:000003",
    ]
    currency_source = record["source"]["field_sources"]["currency"]
    assert currency_source["derivation"] == "isolated_currency_in_opening_branch_metadata_row"
    assert currency_source["normalized_only"] is False
    assert currency_source["evidence_ids"] == [
        "ev:0001:text:000004",
        "ev:0001:text:000002",
        "ev:0001:text:000003",
        "ev:0001:text:000005",
    ]
    assert currency_source["source_refs"] == [
        {
            "source": "canonical_evidence_atoms",
            "source_page": 1,
            "bbox": [15.95, 55.0, 354.35, 68.0],
        }
    ]
    assert currency_source["components"][0]["value_evidence_id"] == "ev:0001:text:000004"
    assert currency_source["components"][0]["branch_value_texts"] == ["交通银行厦门金尚支行"]
    assert currency_source["components"][0]["branch_value_evidence_ids"] == ["ev:0001:text:000003"]


@pytest.mark.parametrize(
    ("row", "expected_branch"),
    [
        (
            [
                _atom(1, "开户机构：", 20, 55, 70, 68, 1),
                _atom(1, "交通银行厦门", 90, 55, 170, 68, 2),
                _atom(1, "人民币", 174, 55, 204, 68, 3),
                _atom(1, "年份：", 230, 55, 270, 68, 4),
                _atom(1, "2022", 280, 55, 310, 68, 5),
            ],
            "交通银行厦门人民币",
        ),
        (
            [
                _atom(1, "开户机构：", 20, 55, 70, 68, 1),
                _atom(1, "交通银行厦门金尚支行", 90, 55, 190, 68, 2),
                _atom(1, "人民币", 255, 55, 285, 68, 3),
            ],
            "交通银行厦门金尚支行人民币",
        ),
        (
            [
                _atom(1, "开户机构：", 20, 55, 70, 68, 1),
                _atom(1, "交通银行厦门金尚支行", 90, 55, 190, 68, 2),
                _atom(1, "人民币", 400, 55, 430, 68, 3),
                _atom(1, "年份：", 450, 55, 490, 68, 4),
                _atom(1, "2022", 500, 55, 530, 68, 5),
            ],
            "交通银行厦门金尚支行人民币",
        ),
        (
            [
                _atom(1, "开户机构：", 20, 55, 70, 68, 1),
                _atom(1, "交通银行厦门金尚支行", 90, 55, 190, 68, 2),
                _atom(1, "人民币", 255, 55, 285, 68, 3),
                _atom(1, "年份：", 500, 55, 540, 68, 4),
                _atom(1, "2022", 550, 55, 580, 68, 5),
            ],
            "交通银行厦门金尚支行人民币",
        ),
    ],
    ids=[
        "tight_legitimate_branch_text",
        "no_supporting_metadata_label",
        "currency_too_far_from_branch",
        "metadata_too_far_from_currency",
    ],
)
def test_detached_currency_role_fails_closed_without_the_full_row_contract(row, expected_branch):
    facts = statement_context._facts_from_row(row, 1)

    assert not any(fact.field_key == "currency" for fact in facts)
    assert next(fact.raw_value for fact in facts if fact.field_key == "branch_name") == expected_branch


@pytest.mark.parametrize(
    ("suffix", "left_value", "right_value", "credit_field", "normalized", "right_x1"),
    [
        ("数", "12", "8", "credit_count", 8, 410.0),
        ("额", "1,234.56", "789.01", "credit_total", "789.01", 445.0),
    ],
)
def test_paired_cumulative_credit_suffix_has_structural_source_provenance(
    suffix,
    left_value,
    right_value,
    credit_field,
    normalized,
    right_x1,
):
    row = [
        _atom(1, f"本月累计借方发生{suffix}:", 20, 500, 140, 514, 1),
        _atom(1, left_value, 150, 500, 200, 514, 2),
        _atom(1, "本月累计贷方", 300, 500, 390, 514, 3),
        _atom(1, right_value, 400, 500, right_x1, 514, 4),
    ]

    facts = statement_context._paired_cumulative_direction_facts(row, 1)
    record = statement_context._record_from_facts(facts, [1], 1)
    credit_fact = next(fact for fact in facts if fact.field_key == credit_field)

    derived_label = f"本月累计贷方发生{suffix}"
    assert credit_fact.raw_name == derived_label
    assert credit_fact.raw_value == right_value
    assert credit_fact.normalized_value == normalized
    assert credit_fact.evidence_ids == (
        "ev:0001:text:000003",
        "ev:0001:text:000001",
        "ev:0001:text:000004",
    )
    assert credit_fact.bbox == (20.0, 500.0, right_x1, 514.0)
    assert record["raw"][derived_label] == right_value
    assert record["canonical_raw"][credit_field] == right_value
    source = record["source"]["field_sources"][credit_field]
    assert source["derivation"] == "structural_pair_suffix_from_explicit_left_label"
    assert source["normalized_only"] is False
    assert source["evidence_ids"] == list(credit_fact.evidence_ids)
    assert source["source_refs"][0]["bbox"] == [20.0, 500.0, right_x1, 514.0]
    assert source["components"] == [
        {
            "page": 1,
            "derived_raw_name": derived_label,
            "left_label": f"本月累计借方发生{suffix}:",
            "right_label": "本月累计贷方",
            "raw_value": right_value,
            "value_evidence_id": "ev:0001:text:000004",
            "evidence_ids": list(credit_fact.evidence_ids),
            "bbox": [20.0, 500.0, right_x1, 514.0],
            "source": "canonical_evidence_atoms",
            "normalized_only": False,
        }
    ]


def test_paired_cumulative_count_accepts_tightly_bounded_wide_issuer_layout():
    row = [
        _atom(1, "本月累计借方发生数:", 15.95, 500.0, 115.45, 514.0, 1),
        _atom(1, "70", 139.45, 500.0, 149.475, 514.0, 2),
        _atom(1, "本月累计贷方", 393.35, 500.0, 453.05, 514.0, 3),
        _atom(1, "22", 463.55, 500.0, 473.575, 514.0, 4),
    ]

    facts = statement_context._paired_cumulative_direction_facts(row, 1)

    assert {fact.field_key: fact.normalized_value for fact in facts} == {
        "debit_count": 70,
        "credit_count": 22,
    }


def test_terminal_wide_cumulative_count_supersedes_stale_prior_page_value():
    atoms = [
        _atom(81, "发生数", 393.35, 407.894, 423.20, 417.844, 247),
        _atom(81, "本月累计借方发生数:", 15.95, 411.494, 115.45, 421.444, 249),
        _atom(81, "70", 139.45, 411.494, 149.475, 421.444, 250),
        _atom(81, "本月累计贷方", 393.35, 410.294, 453.05, 420.244, 251),
        _atom(81, "发生数", 393.35, 422.644, 423.20, 432.594, 252),
        _atom(81, "22", 463.55, 411.494, 473.575, 421.444, 253),
    ]
    footer_row = next(
        row
        for row in statement_context._baseline_rows(atoms)
        if any(atom["text"] == "本月累计借方发生数:" for atom in row)
    )
    terminal_facts = statement_context._paired_cumulative_direction_facts(footer_row, 81)
    stale_fact = statement_context._HeaderFact(
        "credit_count",
        "本月累计贷方发生数:",
        "4",
        4,
        77,
        "page:0077",
        (393.35, 410.294, 473.575, 421.444),
        ("ev:0077:text:000100",),
    )

    record = statement_context._record_from_facts([stale_fact, *terminal_facts], [77, 78, 79, 80, 81], 1)

    assert record["normalized"]["credit_count"] == 22
    assert record["canonical_raw"]["credit_count"] == "22"
    source = record["source"]["field_sources"]["credit_count"]
    assert source["source_refs"][0]["source_page"] == 81
    assert "ev:0081:text:000253" in source["evidence_ids"]


def test_paired_cumulative_suffix_recovery_rejects_ambiguous_left_labels():
    row = [
        _atom(1, "本月累计借方发生数:", 20, 500, 140, 514, 1),
        _atom(1, "12", 150, 500, 170, 514, 2),
        _atom(1, "本月累计借方发生额:", 180, 500, 290, 514, 3),
        _atom(1, "1,234.56", 300, 500, 350, 514, 4),
        _atom(1, "本月累计贷方", 370, 500, 460, 514, 5),
        _atom(1, "8", 470, 500, 480, 514, 6),
    ]

    assert statement_context._paired_cumulative_direction_facts(row, 1) == []


@pytest.mark.parametrize(
    "row",
    [
        [
            _atom(1, "本月累计贷方", 20, 500, 110, 514, 1),
            _atom(1, "8", 120, 500, 130, 514, 2),
            _atom(1, "本月累计借方发生数:", 200, 500, 320, 514, 3),
            _atom(1, "12", 330, 500, 350, 514, 4),
        ],
        [
            _atom(1, "本月累计借方发生数:", 20, 500, 140, 514, 1),
            _atom(1, "12", 150, 500, 170, 514, 2),
            _atom(1, "本月累计贷方", 300, 500, 390, 514, 3),
            _atom(1, "8", 700, 500, 710, 514, 4),
        ],
        [
            _atom(1, "本月累计借方发生数:", 20, 500, 140, 514, 1),
            _atom(1, "12", 150, 530, 170, 544, 2),
            _atom(1, "本月累计贷方", 300, 500, 390, 514, 3),
            _atom(1, "8", 400, 500, 410, 514, 4),
        ],
        [
            _atom(1, "本月累计借方发生数:", 20, 500, 140, 514, 1),
            _atom(1, "12", 150, 500, 170, 514, 2),
            _atom(1, "本月累计贷方", 421, 500, 511, 514, 3),
            _atom(1, "8", 521, 500, 531, 514, 4),
        ],
    ],
    ids=["reversed", "distant_credit_value", "off_baseline_debit_value", "distant_middle_bridge"],
)
def test_paired_cumulative_suffix_recovery_requires_ordered_bounded_geometry(row):
    assert statement_context._paired_cumulative_direction_facts(row, 1) == []


def test_page_layout_fact_and_nearby_seal_code_remain_separate(monkeypatch, parse_result):
    atoms = [
        _atom(1, "交通银行上海市分行明细对账单", 180, 10, 400, 25, 1),
        _atom(1, "开户机构：交通银行上海松江支行", 20, 53, 210, 65, 2),
        _atom(1, "币种：", 220, 53, 260, 65, 3),
        _atom(1, "人民币", 270, 53, 305, 65, 4),
        _atom(1, "页码：", 324, 53, 354, 65, 5),
        _atom(1, "1-1", 356, 53, 371, 65, 6),
        _atom(1, "9A5C698C", 483, 53, 539, 65, 7),
        _atom(1, "对", 455, 30, 475, 50, 8),
        _atom(1, "账", 477, 30, 497, 50, 9),
        _atom(1, "专用", 501, 30, 545, 50, 10),
        _atom(1, "章", 547, 30, 567, 50, 11),
        _atom(1, "交易日期", 20, 100, 70, 114, 12),
        _atom(1, "交易金额", 90, 100, 140, 114, 13),
        _atom(1, "余额", 160, 100, 185, 114, 14),
        _atom(1, "摘要", 205, 100, 230, 114, 15),
        _atom(1, "2023-02-01", 20, 130, 85, 144, 16),
        _atom(1, "10.00", 90, 130, 125, 144, 17),
        _atom(1, "1010.00", 160, 130, 210, 144, 18),
        _atom(1, "测试交易", 220, 130, 280, 144, 19),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["raw"]["页码："] == "1-1"
    assert "source_header_page_label" not in record["normalized"]
    assert "source_header_page_label" not in record["canonical_raw"]
    assert record["normalized"]["seal_code"] == "9A5C698C"
    assert record["canonical_raw"]["seal_code"] == "9A5C698C"
    assert record["raw"]["对账专用章"] == "9A5C698C"
    source = record["source"]["field_sources"]["seal_code"]
    assert source["derivation"] == "seal_code_adjacent_to_explicit_reconciliation_stamp"
    assert source["normalized_only"] is False
    assert source["evidence_ids"] == [
        "ev:0001:text:000007",
        "ev:0001:text:000008",
        "ev:0001:text:000009",
        "ev:0001:text:000010",
        "ev:0001:text:000011",
    ]
    assert source["source_refs"] == [
        {
            "source": "canonical_evidence_atoms",
            "source_page": 1,
            "bbox": [455.0, 30.0, 567.0, 65.0],
        }
    ]
    assert source["components"][0]["stamp_texts"] == ["对", "账", "专用", "章"]
    assert source["components"][0]["value_evidence_id"] == "ev:0001:text:000007"


def test_body_carry_below_ledger_header_cannot_become_statement_balance(monkeypatch, parse_result):
    atoms = [
        _atom(1, "交通银行上海市分行明细对账单", 180, 10, 400, 25, 1),
        _atom(1, "开户机构：交通银行上海松江支行", 20, 53, 210, 65, 2),
        _atom(1, "币种：", 220, 53, 260, 65, 3),
        _atom(1, "人民币", 270, 53, 305, 65, 4),
        _atom(1, "页码：", 324, 53, 354, 65, 5),
        _atom(1, "1-1", 356, 53, 371, 65, 6),
        _atom(1, "9A5C698C", 483, 53, 539, 65, 7),
        _atom(1, "对", 455, 30, 475, 50, 8),
        _atom(1, "账", 477, 30, 497, 50, 9),
        _atom(1, "专用", 501, 30, 545, 50, 10),
        _atom(1, "章", 547, 30, 567, 50, 11),
        _atom(1, "账号：", 20, 72, 55, 84, 12),
        _atom(1, "310069053013003854138", 60, 72, 185, 84, 13),
        _atom(1, "户名：", 220, 72, 255, 84, 14),
        _atom(1, "上海帝芝杰物资有限公司", 260, 72, 400, 84, 15),
        _atom(1, "起始日期：", 20, 88, 75, 100, 16),
        _atom(1, "2023-02-01", 80, 88, 145, 100, 17),
        _atom(1, "终止日期：", 220, 88, 275, 100, 18),
        _atom(1, "2023-02-28", 280, 88, 345, 100, 19),
        _atom(1, "交易日期", 20, 110, 70, 124, 20),
        _atom(1, "交易金额", 90, 110, 140, 124, 21),
        _atom(1, "余额", 160, 110, 185, 124, 22),
        _atom(1, "摘要", 205, 110, 230, 124, 23),
        _atom(1, "承前", 90, 135, 125, 149, 24),
        _atom(1, "6,170.91", 160, 135, 215, 149, 25),
    ]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    [record] = statement_context.build_statement_header_records(parse_result, {})

    assert record["normalized"]["account_number"] == "310069053013003854138"
    assert record["normalized"]["account_holder"] == "上海帝芝杰物资有限公司"
    assert record["normalized"]["query_period"] == "2023-02-01 ~ 2023-02-28"
    assert record["normalized"]["period_start"] == "2023-02-01"
    assert record["normalized"]["period_end"] == "2023-02-28"
    assert record["normalized"]["seal_code"] == "9A5C698C"
    assert "brought_forward_balance" not in record["normalized"]
    assert "brought_forward_balance" not in record["canonical_raw"]
    assert "承前" not in record["raw"]


@pytest.mark.parametrize(
    ("code", "code_y0", "include_stamp"),
    [
        ("9A5C698c", 53.0, True),
        ("９Ａ５Ｃ６９８Ｃ", 53.0, True),
        ("12345678", 53.0, True),
        ("ABCDEFGH", 53.0, True),
        ("9A5C698C", 80.0, True),
        ("9A5C698C", 53.0, False),
    ],
    ids=["lowercase", "fullwidth", "digits_only", "letters_only", "too_far", "no_stamp"],
)
def test_seal_code_requires_strict_vocabulary_and_tight_stamp_proximity(code, code_y0, include_stamp):
    stamp_atoms = [
        _atom(1, "对", 455, 30, 475, 50, 1),
        _atom(1, "账", 477, 30, 497, 50, 2),
        _atom(1, "专用", 501, 30, 545, 50, 3),
        _atom(1, "章", 547, 30, 567, 50, 4),
    ]
    code_atom = _atom(1, code, 483, code_y0, 539, code_y0 + 12, 5)
    atoms = [*stamp_atoms, code_atom] if include_stamp else [code_atom]

    assert statement_context._seal_code_fact(atoms, 1) is None


def test_seal_code_like_label_without_stamp_does_not_bypass_spatial_contract():
    row = [
        _atom(1, "印章编码:", 20, 50, 85, 64, 1),
        _atom(1, "9A5C698C", 95, 50, 151, 64, 2),
    ]

    facts = statement_context._facts_from_row(row, 1)

    assert not any(fact.field_key == "seal_code" for fact in facts)
    assert any(fact.field_key.startswith("source_header_") for fact in facts)


def test_multiple_nearby_seal_codes_fail_closed():
    atoms = [
        _atom(1, "对", 455, 30, 475, 50, 1),
        _atom(1, "账", 477, 30, 497, 50, 2),
        _atom(1, "专用", 501, 30, 545, 50, 3),
        _atom(1, "章", 547, 30, 567, 50, 4),
        _atom(1, "9A5C698C", 483, 53, 539, 65, 5),
        _atom(1, "8B6D709D", 483, 66, 539, 78, 6),
    ]

    assert statement_context._seal_code_fact(atoms, 1) is None


def test_community_dictionary_declares_seal_code_as_statement_header_field():
    from docmirror.plugins.bank_statement.community_plugin import BANK_DATA_DICTIONARY

    expected = {"label": "印章编码", "type": "string"}
    assert BANK_DATA_DICTIONARY["fields"]["seal_code"] == expected
    assert BANK_DATA_DICTIONARY["datasets"]["statement_header"]["columns"]["seal_code"] == expected
