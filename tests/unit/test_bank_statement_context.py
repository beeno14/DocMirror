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


def test_table_only_source_does_not_invent_statement_identity(monkeypatch, parse_result):
    atoms = _current_account_atoms()
    atoms = [atom for atom in atoms if atom["bbox"][1] >= 200]
    monkeypatch.setattr(statement_context, "text_atoms", lambda _result: atoms)
    monkeypatch.setattr(statement_context, "_source_page_texts", lambda _result: {1: ""})

    assert statement_context.build_statement_header_records(parse_result, {}) == []


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
