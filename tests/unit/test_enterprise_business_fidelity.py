# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-led regressions for enterprise business fidelity, independent of PDFs."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.enterprise_native.business_values import opaque_identifier
from docmirror.plugins.credit_report.enterprise_native.extraction import (
    extract_enterprise_attachment_datasets,
    extract_enterprise_continuation_audit,
    extract_enterprise_repayment_liability_records,
    extract_enterprise_report_qualifications,
)
from docmirror.plugins.credit_report.enterprise_native.ir import build_canonical_enterprise_document
from docmirror.plugins.credit_report.enterprise_native.pipeline import (
    _apply_business_qualifications,
    _dataset_completeness,
    run_enterprise_pipeline,
)
from docmirror.plugins.credit_report.enterprise_native.projector import (
    enterprise_public_dataset_policy,
    project_enterprise_community_json,
)
from docmirror.plugins.credit_report.enterprise_native.variant import variant

HEADER = ["交易类型", "交易日期", "交易金额", "到期日期变更月数", "交易明细信息"]
TRANSACTIONS = [
    ["无还本续贷", "2023-11-15", "500", "12", "--"],
    ["提前还款", "2024-11-13", "500", "0", "--"],
    ["展期", "2024-11-14", "0", "3", "调整合同期限"],
    ["合同调整", "2024-11-15", "20", "0", "源报告的交易明细"],
]


def _text(content, top=10):
    return SimpleNamespace(content=content, bbox=[0, top, 500, top + 12])


def _table(name, rows, top=600):
    return SimpleNamespace(
        table_id=name, headers=[], rows=[], metadata={"raw_rows": rows}, bbox=[0, top, 500, top + 80]
    )


def _page(number, texts=(), tables=()):
    return SimpleNamespace(page_number=number, width=595, height=842, texts=list(texts), tables=list(tables))


def _headings():
    return [
        _text("企业信用报告（自主查询版）"),
        _text("附件1：信用记录补充信息", 30),
        _text("短期借款的历史表现", 50),
        _text("1.已结清账户编号：ACCOUNT000001 授信机构：甲银行 业务种类：流动资金贷款", 70),
    ]


def _split_result(sizes, repeat_header=False):
    pages = []
    offset = 0
    for index, size in enumerate(sizes):
        rows = deepcopy(TRANSACTIONS[offset:offset + size])
        if index == 0 or repeat_header:
            rows.insert(0, list(HEADER))
        pages.append(_page(index + 1, _headings() if index == 0 else [], [_table(f"table-{index}", rows, 600 if index == 0 else 20)]))
        offset += size
    assert offset == len(TRANSACTIONS)
    return SimpleNamespace(pages=pages, confidence=1.0)


@pytest.mark.parametrize("sizes", [(4,), (1, 3), (2, 2), (3, 1), (1, 1, 2), (1, 1, 1, 1), (0, 4)])
@pytest.mark.parametrize("repeat_header", [False, True])
def test_special_transactions_are_independent_of_page_split(sizes, repeat_header):
    source = _split_result(sizes, repeat_header)
    before = deepcopy(source)
    document = build_canonical_enterprise_document(source)
    datasets = extract_enterprise_attachment_datasets(document)
    rows = datasets["enterprise_special_transactions"]
    assert [(row["transaction_type"], row["transaction_date"], row["transaction_amount"], row["due_date_change_months"])
            for row in rows] == [(row[0], row[1], int(row[2]), int(row[3])) for row in TRANSACTIONS]
    assert {row["account_identifier"] for row in rows} == {"ACCOUNT000001"}
    assert {row["institution"] for row in rows} == {"甲银行"}
    assert len({row["special_transaction_id"] for row in rows}) == 4
    assert all(row["source_refs"] for row in rows)
    assert source == before
    audit = next(row for row in extract_enterprise_continuation_audit(document, datasets=datasets)
                 if row["continuation_family"] == "special_transaction")
    assert audit["expected_record_count"] == audit["extracted_record_count"] == 4


def test_headerless_transaction_keeps_header_and_body_provenance():
    source = _split_result((1, 3))
    rows = extract_enterprise_attachment_datasets(build_canonical_enterprise_document(source))["enterprise_special_transactions"]
    assert {ref["page"] for ref in rows[1]["source_refs"]} == {1, 2}
    assert rows[1]["source_page"] == 2


@pytest.mark.parametrize("split", [0, 1, 2, 3])
@pytest.mark.parametrize("blank_amount", [False, True])
def test_transaction_continuation_maps_expanded_grid_to_logical_columns(split, blank_amount):
    source = _split_result((split, 4 - split))
    slots = (0, 2, 4, 7, 9)

    def expand(row):
        grid = [""] * 11
        for index, value in zip(slots, row, strict=True):
            grid[index] = value
        return grid

    if blank_amount:
        source.pages[1].tables[0].metadata["raw_rows"][0][2] = ""
    source.pages[0].tables[0].metadata["raw_rows"] = [
        expand(row) for row in source.pages[0].tables[0].metadata["raw_rows"]
    ]
    document = build_canonical_enterprise_document(source)
    datasets = extract_enterprise_attachment_datasets(document)
    rows = datasets["enterprise_special_transactions"]
    assert len(rows) == 4
    for index, row in enumerate(rows):
        assert row["transaction_date"] == TRANSACTIONS[index][1]
        assert row["due_date_change_months"] == int(TRANSACTIONS[index][3])
        assert row["transaction_amount"] == (None if blank_amount and index == split else int(TRANSACTIONS[index][2]))
        assert row["transaction_detail"] == ("" if TRANSACTIONS[index][4] == "--" else TRANSACTIONS[index][4])
    datasets["enterprise_special_transactions"].pop()
    audit = next(row for row in extract_enterprise_continuation_audit(document, datasets=datasets)
                 if row["continuation_family"] == "special_transaction")
    assert audit["expected_record_count"] == 4
    assert audit["extracted_record_count"] == 3


def test_unknown_nonempty_grid_slot_is_not_silently_discarded():
    from docmirror.plugins.credit_report.enterprise_native.continuation import special_transaction_cells

    assert special_transaction_cells(HEADER, [*TRANSACTIONS[0], "extra business value"]) is None
    assert special_transaction_cells(HEADER, TRANSACTIONS[0][:2]) is None


@pytest.mark.parametrize("boundary", ["account", "category", "unrelated_table", "page_gap"])
def test_transaction_header_is_not_inherited_across_a_business_boundary(boundary):
    source = _split_result((1, 3))
    if boundary == "account":
        source.pages[1].texts = [_text("2.已结清账户编号：ACCOUNT000002 授信机构：乙银行", 1)]
    elif boundary == "category":
        source.pages[1].texts = [_text("中长期借款的历史表现", 1)]
    elif boundary == "unrelated_table":
        source.pages[1].tables.insert(0, _table("unrelated", [["其他字段", "其他值"]], 1))
    else:
        source.pages[1].page_number = 3
    rows = extract_enterprise_attachment_datasets(build_canonical_enterprise_document(source))["enterprise_special_transactions"]
    assert len(rows) == 1
    assert rows[0]["transaction_date"] == "2023-11-15"


def test_new_account_with_its_own_header_starts_new_transaction_context():
    source = _split_result((1, 3), repeat_header=True)
    source.pages[1].texts = [_text("2.已结清账户编号：ACCOUNT000002 授信机构：乙银行", 1)]
    rows = extract_enterprise_attachment_datasets(build_canonical_enterprise_document(source))["enterprise_special_transactions"]
    assert [row["account_identifier"] for row in rows] == ["ACCOUNT000001", "ACCOUNT000002", "ACCOUNT000002", "ACCOUNT000002"]


@pytest.mark.parametrize("prefix", [
    ["其他业务类型", "日期", "金额", "期限", "备注"],
    ["其他记录", "2024-11-01"],
])
def test_transaction_continuation_cannot_skip_an_unrelated_header_or_short_row(prefix):
    source = _split_result((1, 3))
    source.pages[1].tables[0].metadata["raw_rows"].insert(0, prefix)
    rows = extract_enterprise_attachment_datasets(build_canonical_enterprise_document(source))["enterprise_special_transactions"]
    assert len(rows) == 1


def test_transaction_continuation_accepts_blank_furniture_before_the_body():
    source = _split_result((1, 3))
    source.pages[1].tables[0].metadata["raw_rows"].insert(0, ["", "", "", "", ""])
    rows = extract_enterprise_attachment_datasets(build_canonical_enterprise_document(source))["enterprise_special_transactions"]
    assert len(rows) == 4


def test_duplicate_printed_transactions_are_not_deduplicated():
    source = _split_result((4,))
    source.pages[0].tables[0].metadata["raw_rows"].append(list(TRANSACTIONS[0]))
    rows = extract_enterprise_attachment_datasets(build_canonical_enterprise_document(source))["enterprise_special_transactions"]
    assert len(rows) == len({row["special_transaction_id"] for row in rows}) == 5


def test_source_count_detects_a_dropped_transaction_independently_of_output():
    document = build_canonical_enterprise_document(_split_result((1, 3)))
    datasets = extract_enterprise_attachment_datasets(document)
    datasets["enterprise_special_transactions"].pop()
    audit = extract_enterprise_continuation_audit(document, datasets=datasets)
    transaction_audit = next(row for row in audit if row["continuation_family"] == "special_transaction")
    assert transaction_audit["expected_record_count"] == 4
    assert transaction_audit["extracted_record_count"] == 3
    assert transaction_audit["unresolved_record_count"] == 1
    completeness = _dataset_completeness(datasets, audit, ())["enterprise_special_transactions"]
    assert completeness["verified"] is False
    assert completeness["omitted_row_count"] == 1


@pytest.mark.parametrize("value,expected", [
    ("B10911000H0001053d0f821da4084c118dd76aebb26c0617", "B10911000H0001053d0f821da4084c118dd76aebb26c0617"),
    ("Ab-09/XY.\n001", "Ab-09/XY.001"),
    ("--", ""),
    (None, ""),
])
def test_opaque_contract_identifiers_preserve_source_characters(value, expected):
    assert opaque_identifier(value) == expected


def test_responsibility_decoder_publishes_precise_fields_without_aliases():
    contract = "B10911000H0001Mixed-Case/0001"
    rows = [
        ["账户编号", "责任类型", "保证合同编号", "币种", "还款责任金额", "授信机构", "业务种类", "开立日期/接收日期", "到期日", "币种"],
        ["ACCOUNT000001", "保证责任", contract, "人民币", "100", "甲银行", "流动资金贷款", "2024-01-01", "2026-01-01", "人民币"],
        ["", "100", "30", "正常", "0", "0", "0", "3", "2025-01-01"],
    ]
    source = SimpleNamespace(pages=[_page(1, [_text("相关还款责任信息明细")], [_table("liability", rows)])])
    record = extract_enterprise_repayment_liability_records(build_canonical_enterprise_document(source))[0]
    assert record["guarantee_contract_identifier"] == contract
    assert record["open_or_receive_date"] == "2024-01-01"
    assert "contract_number" not in record
    assert "open_date" not in record
    fields = enterprise_public_dataset_policy()["enterprise_repayment_responsibility_accounts"]
    assert "contract_number" not in fields
    assert "open_date" not in fields


@pytest.mark.parametrize("value,kind", [("长期", "indefinite"), ("长期有效", "indefinite"), ("2028-12-31", "dated"), ("--", None)])
def test_certificate_validity_is_lossless_and_honestly_typed(value, kind):
    source = SimpleNamespace(pages=[_page(1, [_text("企业信用报告（自主查询版）\n基本信息")], [
        _table("profile", [["登记证书有效截止日期", value, "信息来源机构", "甲银行"]])
    ])], confidence=1.0)
    semantic = run_enterprise_pipeline(source).semantic_document
    profile = semantic.datasets["enterprise_profile"][0]["normalized"]
    assert profile["registration_certificate_valid_through"] == (None if value == "--" else value)
    assert profile.get("registration_certificate_validity_kind") == kind
    columns = variant.data_dictionary()["datasets"]["enterprise_profile"]["columns"]
    assert columns["registration_certificate_valid_through"]["type"] == "string"
    assert columns["registration_certificate_validity_kind"]["enum_ref"] == "registration_certificate_validity_kind"


def test_source_qualifications_are_business_data_and_do_not_change_amounts():
    source = SimpleNamespace(pages=[_page(1, [
        _text("受篇幅所限，本报告只展示部分信贷记录。"),
        _text("说明：由于存在授信限额的控制，剩余可用额度无法准确计算，需要结合授信明细信息进行估算。", 30),
    ])])
    qualifications = extract_enterprise_report_qualifications(build_canonical_enterprise_document(source))
    assert {item["kind"] for item in qualifications} == {"display_scope", "available_limit_estimation"}
    datasets = {
        "enterprise_credit_overview": [{"credit_balance": "300"}],
        "enterprise_facility_summary": [{"available_limit": "300", "facility_type": "non_revolving"}],
        "enterprise_credit_facilities": [{"available_limit": "200"}],
    }
    _apply_business_qualifications(datasets, qualifications)
    overview = datasets["enterprise_credit_overview"][0]
    facility = datasets["enterprise_facility_summary"][0]
    assert overview["source_display_limited"] is True
    assert overview["source_display_scopes"] == ["credit_records"]
    assert facility["available_limit_requires_estimation"] is True
    assert facility["available_limit"] == "300"
    assert datasets["enterprise_credit_facilities"] == [{"available_limit": "200"}]
    _apply_business_qualifications(datasets, qualifications)
    assert overview["source_display_scopes"] == ["credit_records"]
    assert len(overview["source_refs"]) == 1


def test_absent_qualification_does_not_assert_full_source_disclosure():
    datasets = {"enterprise_credit_overview": [{"credit_balance": "300"}]}
    before = deepcopy(datasets)
    _apply_business_qualifications(datasets, [])
    assert datasets == before


def test_non_credit_scope_does_not_accidentally_include_credit_scope():
    source = SimpleNamespace(pages=[_page(1, [_text("本报告仅展示部分非信贷记录。")])])
    qualifications = extract_enterprise_report_qualifications(build_canonical_enterprise_document(source))
    assert [item["scope"] for item in qualifications] == ["non_credit_records"]


def test_public_projection_preserves_business_qualifiers_and_validity_kind():
    from tests.unit.test_enterprise_public_json_projection import _dataset, _payload, _row

    payload = _payload()
    for name, values in (
        ("enterprise_credit_overview", {"source_display_limited": True, "source_display_scopes": ["credit_records"]}),
        ("enterprise_facility_summary", {"available_limit": "300", "available_limit_requires_estimation": True, "currency": "CNY", "amount_unit": "CNY_10K"}),
        ("enterprise_profile", {"registration_certificate_valid_through": "长期", "registration_certificate_validity_kind": "indefinite"}),
    ):
        payload["datasets"] = [dataset for dataset in payload["datasets"] if dataset["name"] != name]
        payload["datasets"].append(_dataset(name, [_row(name + ":1", values)]))
    public = project_enterprise_community_json(payload)
    rows = {dataset["name"]: dataset["rows"][0]["normalized"] for dataset in public["datasets"]}
    assert rows["enterprise_credit_overview"]["source_display_limited"] is True
    assert rows["enterprise_credit_overview"]["source_display_scopes"] == ["credit_records"]
    assert rows["enterprise_facility_summary"]["available_limit_requires_estimation"] is True
    assert rows["enterprise_profile"]["registration_certificate_validity_kind"] == "indefinite"


def test_enterprise_array_survives_the_actual_generic_public_record_exporter():
    from docmirror.output.community_bundle import _dataset_columns, _public_record
    from tests.unit.test_enterprise_public_json_projection import _dataset, _payload, _row

    name = "enterprise_credit_overview"
    scopes = ["credit_records", "public_records"]
    row = _row("overview:1", {"source_display_limited": True, "source_display_scopes": scopes})
    columns = _dataset_columns([row], variant.data_dictionary(), name)
    exported = _public_record(row, dataset_id=name, row_index=1, columns=columns, fallback_page_range=[1, 1])
    dataset = _dataset(name, [exported])
    dataset["columns"] = columns
    payload = _payload()
    payload["datasets"].append(dataset)
    before = deepcopy(payload)
    public = project_enterprise_community_json(payload)
    overview = next(dataset for dataset in public["datasets"] if dataset["name"] == name)
    assert overview["rows"][0]["normalized"]["source_display_scopes"] == scopes
    assert next(column for column in overview["columns"] if column["key"] == "source_display_scopes")["type"] == "array"
    assert project_enterprise_community_json(public) == public
    assert payload == before


@pytest.mark.parametrize("value", ['{"scope":"credit_records"}', '"credit_records"', "not json"])
def test_invalid_structured_business_value_is_not_exported_as_a_string(value):
    from tests.unit.test_enterprise_public_json_projection import _dataset, _payload, _row

    payload = _payload()
    payload["datasets"].append(_dataset("enterprise_credit_overview", [
        _row("overview:1", {"source_display_scopes": value})
    ]))
    with pytest.raises(ValueError):
        project_enterprise_community_json(payload)


@pytest.mark.parametrize("source_value,key,value,value_type,label", [
    ("0", "overdue_months", 0, "integer", "逾期月数"),
    ("12", "overdue_months", 12, "integer", "逾期月数"),
    ("N", "repayment_status", "N", "string", "还款状态"),
])
def test_split_business_field_uses_its_own_type_and_label(source_value, key, value, value_type, label):
    from tests.unit.test_enterprise_public_json_projection import _dataset, _payload, _row

    name = "enterprise_repayment_responsibility_accounts"
    dataset = _dataset(name, [_row("liability:1", {"overdue_months_or_repayment_status": source_value})])
    dataset["columns"][0].update({"label": "逾期月数/还款状态", "type": "string"})
    payload = _payload()
    payload["datasets"] = [item for item in payload["datasets"] if item["name"] != name] + [dataset]
    result = next(item for item in project_enterprise_community_json(payload)["datasets"] if item["name"] == name)
    assert result["rows"][0]["normalized"] == {key: value}
    column = next(column for column in result["columns"] if column["key"] == key)
    assert column["type"] == value_type
    assert column["label"] == label


def test_emitted_only_counts_do_not_claim_independent_completeness():
    result = _dataset_completeness({"enterprise_profile": [{"source_page": 1}]}, (), ())["enterprise_profile"]
    assert result["verified"] is False
    assert result["basis"] == "emitted_records_only"
    assert result["status"] == "unverified"
