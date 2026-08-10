from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)


def _table(table_id: str, top: float, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        bbox=[10.0, top, 590.0, top + 80.0],
        metadata={"raw_rows": rows},
        confidence=0.95,
        rows=[],
    )


def _page(number: int, tables: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=number,
        source_page_number=number,
        height=800.0,
        tables=tables,
        texts=[],
    )


@pytest.mark.parametrize(
    ("base_rows", "continuation_rows", "expected"),
    [
        (
            [
                ["管理机构", "账户标识", "开立日期", "到期日期", "借款金额", "账户币种"],
                ["五矿国际信托有限公司", "J10158510H000110000000640557", "2021.01.12", "2024.01.12", "140,000", "人民币元"],
            ],
            [
                ["五级分类", "账户状态", "余额", "剩余还款期数"],
                ["正常", "正常", "55,046", "13"],
            ],
            {"balance": 55046, "remaining_periods": 13, "account_status": "active"},
        ),
        (
            [
                ["发卡机构", "账户标识", "开立日期", "账户授信额度", "币种", "业务种类", "担保方式"],
                ["中国建设银行股份有限公司福建省分行", "B10411000H000115602800002159651255803723", "2013.08.14", "20,000", "人民币元", "贷记卡", "信用/免担保"],
            ],
            [
                ["账户状态", "余额", "已用额度", "剩余分期期数", "最近6个月平均使用额度", "最大使用额度"],
                ["正常", "18,193", "17,891", "0", "18,656", "26,765"],
            ],
            {
                "balance": 18193,
                "used_amount": 17891,
                "remaining_periods": 0,
                "recent_6_month_average_used_amount": 18656,
                "maximum_used_amount": 26765,
                "account_status": "active",
            },
        ),
    ],
)
def test_geometric_table_above_next_account_body_belongs_to_prior_account(
    base_rows: list[list[str]],
    continuation_rows: list[list[str]],
    expected: dict[str, object],
) -> None:
    prior = _table("prior", 100.0, base_rows)
    continuation = _table("continued-detail", 20.0, continuation_rows)
    next_body = _table("next-body", 250.0, base_rows)
    context = SimpleNamespace(
        pages=[_page(1, [prior]), _page(2, [continuation, next_body])],
        tables_continue=lambda _left, _right: None,
    )

    accounts, _repayments, _events = native_extraction._extract_table_accounts(context)

    assert len(accounts) == 2
    for field_name, value in expected.items():
        assert accounts[0][field_name] == value
    assert len(accounts[0]["source_refs"]) == 2


def test_account_table_issues_remap_to_final_printed_account_identity(monkeypatch) -> None:
    base = _table(
        "account-base",
        100.0,
        [
            ["管理机构", "账户标识", "开立日期", "到期日期", "借款金额", "账户币种"],
            ["五矿国际信托有限公司", "J10158510H000110000000640557", "2021.01.12", "2024.01.72", "140,000", "人民币元"],
        ],
    )
    context = SimpleNamespace(pages=[_page(1, [base])])
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [
            {
                "account_id": "credit_account:non_revolving_loan:2",
                "sequence": 1,
                "category_sequence": 2,
                "account_type": "non_revolving_loan",
                "page": 1,
                "bbox": [10.0, 80.0, 100.0, 90.0],
                "_canonical_segment": {
                    "pages": [{"logical_page": 1, "min_y": 80.0, "max_y": None}]
                },
                "source_refs": [{"logical_page": 1, "bbox": [10.0, 80.0, 100.0, 90.0]}],
            }
        ],
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    issues = collect_extraction_issues(context)

    assert accounts[0]["account_id"] == "credit_account:non_revolving_loan:2"
    due_date_issue = next(issue for issue in issues if issue.get("field_name") == "due_date")
    assert due_date_issue["target_record_id"] == accounts[0]["account_id"]
    assert "table_observation" not in due_date_issue["target_record_id"]


def test_r2_exact_anchor_resolves_shared_native_revolving_table_signature(monkeypatch) -> None:
    identifier = "D10053310H00012022052901021012089466554314"
    base = _table(
        "r2-account-base",
        370.0,
        [
            ["管理机构", "账户标识", "开立日期", "到期日期", "账户授信额度", "账户币种"],
            ["浙江网商银行股份有限公司", identifier, "2022.05.29", "长期", "10,000", "人民币元"],
            ["业务种类", "担保方式", "还款期数", "还款频率", "还款方式", "共同借款标志"],
            ["个人经营性贷款", "信用/免担保", "--", "月", "不区分还款方式", "无"],
        ],
    )
    context = SimpleNamespace(pages=[_page(12, [base])])
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [
            {
                "account_id": "credit_account_provisional:r2",
                "account_type": "revolving_loan_account",
                "account_family_quality": "exact",
                "_printed_ordinal_status": "printed_unreadable",
                "account_identifier": identifier,
                "page": 12,
                "source_page": 12,
                "bbox": [10.0, 360.0, 300.0, 369.0],
                "_canonical_segment": {
                    "pages": [{"logical_page": 12, "min_y": 360.0, "max_y": None}]
                },
                "source_refs": [
                    {
                        "source": "candidate_b_account_anchor",
                        "logical_page": 12,
                        "source_page": 12,
                        "bbox": [10.0, 360.0, 300.0, 369.0],
                    }
                ],
            }
        ],
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["account_id"] == "credit_account:revolving_loan_account:1"
    assert accounts[0]["account_type"] == "revolving_loan_account"
    assert accounts[0]["account_identifier"] == identifier
    assert {ref.get("source") for ref in accounts[0]["source_refs"]} == {
        "candidate_b_account_anchor",
        "native_detail_table",
    }


def test_closed_collapsed_account_clusters_recover_only_shape_unique_fields() -> None:
    page = _page(16, [])
    loan_table = _table(
        "loan-cluster",
        100.0,
        [
            ["剩余还款期数 五级分类 余额 账户状态"],
            ["正常 正常 13 55,046"],
            ["最近一次 本月应还款 应还款日 本月实还款 还款日期"],
            ["2022.11.12 0 2022.12.12 0"],
        ],
    )
    loan = {"account_id": "loan:2", "canonical_raw": {}}
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    native_extraction._apply_account_facts(
        context, loan, loan_table.metadata["raw_rows"], page=page, table=loan_table
    )

    assert loan["balance"] == 55046
    assert loan["remaining_periods"] == 13
    assert loan["account_status"] == "active"
    assert loan["five_tier_class"] == "正常"
    assert loan["last_repayment_date"] == "2022-11-12"
    assert loan["scheduled_payment_date"] == "2022-12-12"
    assert loan["scheduled_payment"] == loan["actual_payment"] == 0

    card_table = _table(
        "card-cluster",
        300.0,
        [
            ["截至2022年12月13日"],
            ["未出单的大额 最近6个月 剩余分期期数 最大使用额度 账户状态 余额 已用额度 平均使用额度 专项分期余额"],
            ["0 17,891 18,656 26,765 正常 18,193 -*"],
            ["当前逾期期数 最近一次还款日期 当前逾期总额 账单日 本月应还款 本月实还款"],
            ["2,186 0 2022.12.13 2022.12.03 0 2,186"],
        ],
    )
    card = {"account_id": "card:6", "canonical_raw": {}}

    native_extraction._apply_account_facts(
        context, card, card_table.metadata["raw_rows"], page=page, table=card_table
    )

    assert card["snapshot_date"] == "2022-12-13"
    assert card["balance"] == 18193
    assert card["used_amount"] == 17891
    assert card["remaining_periods"] == 0
    assert card["recent_6_month_average_used_amount"] == 18656
    assert card["maximum_used_amount"] == 26765
    assert card["billing_date"] == "2022-12-13"
    assert card["last_repayment_date"] == "2022-12-03"
    assert card["scheduled_payment"] == card["actual_payment"] == 2186
    assert card["current_overdue_periods"] == card["current_overdue_amount"] == 0


def test_watermarked_canonical_terminal_subtable_recovers_settled_state_and_close_date() -> None:
    page = _page(8, [])
    table = _table(
        "pt_8_2",
        300.0,
        [
            ["物 账户状态 爱", "? 账户关闭日期"],
            ["家 结清", "Q 2020.02.21"],
        ],
    )
    account = {
        "account_id": "credit_account:non_revolving_loan:12",
        "canonical_raw": {},
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    native_extraction._apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["account_status"] == "settled"
    assert account["account_lifecycle_state"] == "settled"
    assert account["current_overdue"] is False
    assert account["close_date"] == "2020-02-21"
    assert account["source_refs_by_field"]["account_lifecycle_state"][0][
        "binding_quality"
    ] == "canonical_account_terminal_subtable"
    assert account["source_refs_by_field"]["close_date"][0]["binding_quality"] == (
        "canonical_account_terminal_subtable"
    )
    assert not any(
        issue.get("status") not in {"resolved", "informational"}
        and issue.get("field_name") in {"account_lifecycle_state", "close_date"}
        for issue in context._personal_detail_extraction_issues
    )


def test_terminal_subtable_rejects_multiple_candidate_blocks() -> None:
    page = _page(8, [])
    table = _table(
        "ambiguous-terminal-block",
        300.0,
        [
            ["物 账户状态 爱", "? 账户关闭日期"],
            ["家 结清", "Q 2020.02.21"],
            ["账户状态 X", "账户关闭日期 ?"],
            ["结清 Y", "2020.02.22 Q"],
        ],
    )
    account = {"account_id": "account:ambiguous", "canonical_raw": {}}
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    native_extraction._apply_account_facts(
        context,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert "account_lifecycle_state" not in account
    assert "close_date" not in account


def test_printed_heading_business_identifier_overrides_nearby_table_guess(monkeypatch) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    table = {
        "account_id": "credit_account_table_observation:account10",
        "account_type": "credit_card",
        "credit_agreement_identifier": "WRONGNEXTHEADING123",
        "source_refs": [{"logical_page": 18, "bbox": [10, 200, 590, 300]}],
    }
    anchor = {
        "account_id": "credit_account:credit_card:10",
        "sequence": 1,
        "category_sequence": 10,
        "account_type": "credit_card",
        "credit_agreement_identifier": "B11512900H0001002044607320210926",
        "page": 18,
        "bbox": [10, 100, 500, 120],
        "_canonical_segment": {
            "pages": [{"logical_page": 18, "min_y": 100.0, "max_y": 350.0}]
        },
        "source_refs": [{"logical_page": 18, "bbox": [10, 100, 500, 120]}],
    }
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [], []))
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [anchor])

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert accounts[0]["credit_agreement_identifier"] == "B11512900H0001002044607320210926"
    assert accounts[0]["source_refs_by_field"]["credit_agreement_identifier"][0]["binding"] == (
        "canonical_account_anchor"
    )


def test_malformed_account_institution_is_withheld_and_reported() -> None:
    page = _page(18, [])
    table = _table(
        "bad-institution",
        100.0,
        [
            ["发卡机构", "账户标识", "开立日期", "账户授信额度", "币种", "业务种类", "担保方式"],
            ["家 广发银行股份 分行", "B11215800H0001100991849200001156040", "2014.06.19", "11,500", "人民币元", "贷记卡", "信用/免担保"],
        ],
    )
    account = {"account_id": "credit_account_table_observation:bad", "canonical_raw": {}}
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    native_extraction._apply_account_facts(context, account, table.metadata["raw_rows"], page=page, table=table)

    assert "management_institution" not in account
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") == "management_institution"
    )
    assert issue["issue_code"] == "candidate_b_exact_slot_value_invalid"
    assert "normalized_value_withheld" in issue["reason_codes"]


def test_header_recovers_one_deleted_timestamp_digit_and_exact_visible_query_reason() -> None:
    header = _table(
        "report-header",
        20.0,
        [
            ["被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"],
            ["林岚挺", "身份证", "350102198311011933", "本人", "本人查询（自助查询机）"],
        ],
    )
    context = SimpleNamespace(pages=[_page(1, [header])])
    text = "报告编号:2023011314453720187289 报告时间:2023.01314:45:37"

    result = native_extraction._extract_header_datasets(context, text)
    metadata = result["personal_report_metadata"][0]

    assert metadata["report_time"] == "2023-01-13T14:45:37+08:00"
    assert metadata["query_reason"] == "本人查询(自助查询机)"


def test_spouse_document_columns_are_not_secondary_subject_identity_documents(
    monkeypatch,
) -> None:
    header = _table(
        "report-header",
        20.0,
        [
            ["被查询者姓名", "被查询者证件类型", "被查询者证件号码", "查询机构", "查询原因"],
            ["林某", "身份证", "350102198311011933", "本人", "本人查询(自助查询机)"],
        ],
    )
    spouse = _table(
        "spouse",
        120.0,
        [
            ["姓名", "证件类型", "证件号码", "工作单位", "联系电话"],
            ["--", '"', "** Te", "--", "-."],
        ],
    )
    context = SimpleNamespace(
        pages=[_page(1, [header, spouse])],
        _personal_detail_extraction_issues=[],
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, _name: [],
    )

    result = native_extraction._extract_header_datasets(context, "")

    assert len(result["identity_documents"]) == 1
    assert result["identity_documents"][0]["is_primary"] is True
    assert result["identity_documents"][0]["document_number"] == "350102198311011933"


def test_inferred_inquiry_sequence_uncertainty_is_field_scoped() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 8,
                "source_page": 4,
                "canonical_template_id": "annotations_and_inquiries",
                "lines": [
                    {"text": "1 2024.01.02 银行甲 贷款审批", "bbox": [50, 10, 390, 18]},
                    {"text": "2024.01.01", "bbox": [110, 30, 170, 38]},
                    {"text": "银行乙", "bbox": [200, 31, 280, 39]},
                    {"text": "贷后管理", "bbox": [345, 29, 390, 37]},
                ],
            }
        ]
    )

    rows = native_extraction._canonical_inquiry_line_rows(context)

    inferred = rows[1]
    assert inferred["sequence"] == 2
    assert inferred["institution"] == "银行乙"
    assert inferred["reason"] == "贷后管理"
    assert "extraction_status" not in inferred
    assert "audit" not in inferred
    issue = context._personal_detail_extraction_issues[0]
    assert issue["target_record_id"] == inferred["inquiry_id"]
    assert issue["field_name"] == "sequence"


def test_monthly_status_contract_accepts_slash_and_hash() -> None:
    assert {"/", "#"} <= native_extraction._STATUS_CODES
