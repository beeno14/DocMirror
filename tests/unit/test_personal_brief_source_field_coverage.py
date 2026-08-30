# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import PageContent, TextBlock
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report import business_records
from docmirror.plugins.credit_report.personal_brief_native.account_rules import account_narratives
from docmirror.plugins.credit_report.personal_brief_native.audit import append_personal_brief_observational_warnings
from docmirror.plugins.credit_report.personal_brief_native.contracts import (
    PersonalBriefContractError,
    validate_personal_brief_public_record,
)
from docmirror.plugins.credit_report.personal_brief_native.pipeline import run_personal_brief_pipeline
from docmirror.plugins.credit_report.personal_brief_native.projector import (
    derive_personal_brief_projection,
    project_personal_brief_community_json,
)
from tests.unit.test_personal_brief_canonical_pipeline import _result
from tests.unit.test_personal_brief_observational_audit import _valid_payload
from tests.unit.test_personal_brief_public_json_projection import _payload

HEADER = (
    "个人信用报告 报告编号：2026071900012345678901 "
    "报告时间：2026-07-19 09:08:07 姓名：张三 "
    "证件类型：身份证 证件号码：11010519491231002X"
)
FIRST = (
    "2021年11月08日天津银行股份有限公司为其他个人消费贷款授信，"
    "额度有效期至2027年03月24日，可循环使用。截至2025年02月，"
    "信用额度1,000元（人民币），余额为0，当前无逾期。"
)
NEXT = (
    "2021年11月08日重庆度小满小额贷款有限公司为其他个人消费贷款授信，"
    "额度有效期至2025年02月28日，可循环使用。截至2025年02月，"
    "信用额度200,000元（人民币），余额为0，当前无逾期。"
)


@pytest.mark.parametrize(
    ("first", "following", "expiry", "snapshot", "limit", "balance"),
    [
        (FIRST, NEXT, "2027-03-24", "2025-02", 1000, 0),
        (
            "2021年12月23日浙江泰隆商业银行股份有限公司上海普陀支行为个人经营性贷款授信，"
            "额度有效期至2026年02月28日，可循环使用。截至2025年01月，"
            "信用额度300,000元（人民币），余额为0，当前无逾期。",
            "2024年10月08日浙江网商银行股份有限公司为个人经营性贷款授信，额度长期有效，"
            "可循环使用。截至2025年02月，信用额度1,420,010元（人民币），余额为333,333，当前无逾期。",
            "2026-02-28", "2025-01", 300000, 0,
        ),
        (
            "2025年03月20日重庆京东盛际小额贷款有限公司为其他个人消费贷款授信，"
            "额度有效期至2025年08月22日，可循环使用。截至2025年06月，"
            "信用额度140,300元（人民币），余额为375，当前无逾期。",
            "2025年03月20日昆仑银行股份有限公司为其他个人消费贷款授信，"
            "额度有效期至2026年05月21日，可循环使用。截至2025年06月，"
            "信用额度30,000元（人民币），余额为0，当前无逾期。",
            "2025-08-22", "2025-06", 140300, 375,
        ),
    ],
)
def test_expiry_date_never_uses_the_next_accounts_grant_clause(
    first: str, following: str, expiry: str, snapshot: str, limit: int, balance: int
) -> None:
    # All three former real-file failures happen without any page break.
    assert [row.text for row in account_narratives(first + following)] == [first, following]
    semantic = run_personal_brief_pipeline(_result(HEADER, "信贷记录", "贷款", first + following)).semantic_document
    accounts = semantic.datasets["credit_accounts"]
    assert len(accounts) == 2
    assert {
        field: accounts[0][field]
        for field in ("credit_line_expiry_date", "information_as_of", "credit_limit", "balance", "current_overdue", "is_revolving")
    } == {
        "credit_line_expiry_date": expiry,
        "information_as_of": snapshot,
        "credit_limit": limit,
        "balance": balance,
        "current_overdue": False,
        "is_revolving": True,
    }
    assert semantic.dataset_completeness["credit_accounts"]["verified"] is True


@pytest.mark.parametrize("cut", ["额度有效期至", "可循环", "截至", "信用额度", "余额为"])
def test_account_fields_survive_page_and_unit_splits(cut: str) -> None:
    split = FIRST.index(cut) + len(cut)
    result = _result(HEADER, "信贷记录", "贷款", FIRST[:split])
    result.pages.append(PageContent(
        page_number=2, source_page_number=2, width=600, height=800,
        texts=[TextBlock(content=FIRST[split:] + NEXT, bbox=[20, 25, 580, 150])],
    ))
    semantic = run_personal_brief_pipeline(result).semantic_document
    first, second = semantic.datasets["credit_accounts"]
    assert first["credit_limit"] == 1000
    assert first["balance"] == 0
    assert first["current_overdue"] is False
    assert first["is_revolving"] is True
    assert second["credit_limit"] == 200000
    assert semantic.dataset_completeness["credit_accounts"]["verified"] is True


@pytest.mark.parametrize("opening", ["为其他个人消费贷\n款授\n信", "为 其他个人消费贷款 授信"])
def test_native_word_wrapping_does_not_change_grant_boundaries(opening: str) -> None:
    text = FIRST.replace("为其他个人消费贷款授信", opening)
    semantic = run_personal_brief_pipeline(_result(HEADER, "贷款", text + NEXT)).semantic_document
    assert len(semantic.datasets["credit_accounts"]) == 2
    assert semantic.datasets["credit_accounts"][0]["credit_limit"] == 1000


def test_candidate_without_a_valid_issuer_does_not_cut_the_previous_record() -> None:
    narratives = account_narratives(FIRST + "2028年01月01日，提供其他贷款资料。" + NEXT)
    assert len(narratives) == 2
    assert narratives[0].text.startswith(FIRST)


def test_source_present_value_loss_is_incomplete_not_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    original = business_records._personal_brief_account_from_chunk

    def lose_balance(*args: object, **kwargs: object) -> dict[str, object]:
        row = original(*args, **kwargs)
        assert row is not None
        row.pop("balance")
        return row

    monkeypatch.setattr(business_records, "_personal_brief_account_from_chunk", lose_balance)
    result = _result(HEADER, "信贷记录", "贷款", FIRST)
    semantic = run_personal_brief_pipeline(result).semantic_document
    details = semantic.dataset_completeness["credit_accounts"]
    assert details["expected_row_count"] == details["emitted_row_count"] == 1
    assert details["verified"] is False
    assert details["missing_source_fields"][0]["fields"] == ["balance"]
    assert semantic.extraction_report["status"] == "incomplete"

    projection = derive_personal_brief_projection(
        SimpleNamespace(projector_id="test", domain_name="credit_report"), result
    )
    values = projection.datasets["credit_accounts"][0]["normalized"]
    assert "balance" not in values
    assert "balance_status" not in values
    assert values["is_revolving"] is True
    assert any("INCOMPLETE" in warning for warning in projection.warnings)


def test_unprinted_currency_is_not_invented_for_a_settled_credit_line() -> None:
    text = "2022年02月07日浙江网商银行股份有限公司为个人经营性贷款授信，可循环使用。2023年04月已结清。"
    projection = derive_personal_brief_projection(
        SimpleNamespace(projector_id="test", domain_name="credit_report"), _result(HEADER, "贷款", text)
    )
    values = projection.datasets["credit_accounts"][0]["normalized"]
    assert "account_currency" not in values
    assert "currency" not in values
    assert values["reporting_amount_currency"] == "CNY"
    assert values["reporting_amount_unit"] == "CNY_1"
    assert values["is_revolving"] is True


def test_currency_label_is_retained_when_it_is_not_in_the_iso_dictionary() -> None:
    row = business_records._personal_brief_account_from_chunk(FIRST.replace("人民币", "测试币"), [])
    assert row["account_currency"] == "测试币"
    assert row["reporting_amount_currency"] == "CNY"


@pytest.mark.parametrize(("clause", "expected"), [("可循环使用", True), ("不可循环使用", False), ("", None)])
def test_revolving_permission_is_explicit_and_independent_of_account_type(clause: str, expected: bool | None) -> None:
    row = business_records._personal_brief_account_from_chunk(FIRST.replace("可循环使用", clause), [])
    assert row["account_type"] == "credit_line"
    assert row.get("is_revolving") is expected


@pytest.mark.parametrize(
    "business_type",
    [
        "个人商用房贷款（包括商住两用房）",
        "个人商用房（含商住两用）贷款",
        "个人住房公积金贷款",
    ],
)
def test_complete_loan_business_label_is_preserved(business_type: str) -> None:
    text = (
        "2024年01月20日样例银行某分行发放的300,000元（人民币）"
        f"{business_type}，2034年01月19日到期。"
    )

    row = business_records._personal_brief_account_from_chunk(text, [])

    assert row is not None
    assert row["business_type"] == business_type


def test_noncredit_and_public_lookbacks_ignore_pdf_spacing() -> None:
    semantic = run_personal_brief_pipeline(
        _result(
            HEADER,
            "非信贷交易记录",
            "这部分包含您最近5 年内的非信贷交易记录。金额类数据均以人民币计算，精确到元。",
            "公共记录",
            "这部分包含您最近 5 年内的公共信息。金额类数据均以人民币计算，精确到元。",
        )
    ).semantic_document

    assert semantic.facts["non_credit_transaction_summary"]["lookback_years"] == 5
    assert semantic.facts["public_record_summary"]["lookback_years"] == 5
    assert semantic.extraction_report["source_field_coverage"]["sections"] == [
        {
            "section_type": "non_credit_transactions",
            "fields": {"lookback_years": 5},
            "source_pages": [1],
        },
        {
            "section_type": "public_records",
            "fields": {"lookback_years": 5},
            "source_pages": [1],
        },
    ]


@pytest.mark.parametrize("value", [0, 1, "true", "false", "可循环使用"])
def test_revolving_contract_rejects_non_boolean_values(value: object) -> None:
    with pytest.raises(PersonalBriefContractError, match="BOOLEAN_CONTRACT"):
        validate_personal_brief_public_record("credit_accounts", "account:1", {"is_revolving": value})


def test_inquiry_scope_is_reconstructed_across_pages() -> None:
    result = _result(HEADER, "查询记录", "这部分包含您的信用报告最近")
    result.pages.append(PageContent(
        page_number=2, source_page_number=2, width=600, height=800,
        texts=[
            TextBlock(content="2年内被查询的记录。", bbox=[20, 20, 580, 40]),
            TextBlock(content="机构查询记录明细", bbox=[20, 60, 580, 80]),
        ],
    ))
    semantic = run_personal_brief_pipeline(result).semantic_document
    assert semantic.facts["inquiry_record_summary"] == {
        "lookback_years": 2,
        "source_statement": "这部分包含您的信用报告最近2年内被查询的记录。",
        "source_pages": [1, 2],
    }
    assert semantic.extraction_report["source_field_coverage"]["sections"][0]["fields"] == {"lookback_years": 2}


def test_absent_inquiry_scope_is_not_invented() -> None:
    semantic = run_personal_brief_pipeline(_result(HEADER)).semantic_document
    assert "inquiry_record_summary" not in semantic.facts
    assert semantic.extraction_report["source_field_coverage"]["sections"] == []


def test_source_coverage_warns_even_when_semantic_and_public_both_lost_the_field() -> None:
    semantic = _valid_payload()
    account = next(d for d in semantic["datasets"] if d["name"] == "credit_accounts")["rows"][0]
    account["normalized"].update(source_section="credit_cards", source_sequence=1)
    account["normalized"].pop("balance")
    account["normalized"].pop("balance_status")
    semantic["domain"] = {"facts": {"personal_brief_extraction_report": {"source_field_coverage": {
        "accounts": [{
            "source_section": "credit_cards", "source_sequence": 1,
            "source_pages": [1, 2], "fields": {"credit_accounts": {"balance": "balance"}},
        }],
        "sections": [],
    }}}}
    public = project_personal_brief_community_json(semantic)
    before = deepcopy(public)
    audited = append_personal_brief_observational_warnings(semantic, public)
    warnings = [w for w in audited["warnings"] if w["code"] == "PERSONAL_BRIEF_AUDIT_SOURCE_FIELD_MISSING"]
    assert len(warnings) == 1
    assert "balance" in warnings[0]["message"]
    assert "credit_account:1" in warnings[0]["message"]
    assert warnings[0]["dataset_id"] == "ds_credit_accounts"
    assert warnings[0]["page_range"] == [1, 2]
    assert public == before
    assert audited["datasets"] == before["datasets"]


def test_public_projection_retains_false_revolving_flag_and_boolean_descriptor() -> None:
    payload = _payload()
    dataset = next(d for d in payload["datasets"] if d["name"] == "credit_accounts")
    dataset["rows"][0]["normalized"]["is_revolving"] = False
    public = project_personal_brief_community_json(payload)
    dataset = next(d for d in public["datasets"] if d["name"] == "credit_accounts")
    assert dataset["rows"][0]["normalized"]["is_revolving"] is False
    assert next(c for c in dataset["columns"] if c["key"] == "is_revolving")["type"] == "boolean"


@pytest.mark.parametrize("document_type", ["enterprise_credit_report", "personal_credit_report_detailed", "invoice"])
def test_source_envelope_relaxation_is_personal_brief_only(document_type: str) -> None:
    public = project_personal_brief_community_json(_payload())
    assert validate_projection_payload("community", public).valid
    public["document"]["type"] = document_type
    assert not validate_projection_payload("community", public).valid
    for dataset in public["datasets"]:
        for row in dataset["rows"]:
            row["source"] = {}
    assert validate_projection_payload("community", public).valid
