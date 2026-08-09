# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration contracts for closed whole-cell Candidate-B decoders."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _account_events,
    _apply_account_facts,
    _extract_employment_records,
    _extract_table_accounts,
)


def _table(table_id: str, *rows: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": list(rows)},
        headers=[],
        rows=[],
        bbox=[20.0, 20.0, 580.0, 300.0],
        confidence=0.96,
    )


def _page(*tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=2,
        source_page_number=1,
        tables=list(tables),
        texts=[],
        height=800.0,
    )


def _result(*tables: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        pages=[_page(*tables)],
        tables_continue=lambda _left, _right: False,
        _personal_detail_extraction_issues=[],
    )


def _account_base_rows(*trailing_rows: list[str]) -> list[list[str]]:
    return [
        ["管理机构", "账户标识", "开立日期", "到期日期", "借款金额", "账户币种"],
        [
            "五矿国际信托有限公司",
            "J10158510H000110000000640557",
            "2021.01.12",
            "2024.01.12",
            "140,000",
            "人民币元",
        ],
        *trailing_rows,
    ]


def _split_account_result(
    first_rows: list[list[str]],
    second_rows: list[list[str]],
) -> SimpleNamespace:
    first = _table("account-base", *first_rows)
    second = _table("account-continuation", *second_rows)
    first.bbox = [20.0, 20.0, 580.0, 300.0]
    second.bbox = [20.0, 20.0, 580.0, 300.0]
    first_page = _page(first)
    second_page = _page(second)
    second_page.page_number = 3
    second_page.source_page_number = 2
    return SimpleNamespace(
        pages=[first_page, second_page],
        tables_continue=lambda left, right: (
            True if (left, right) == ("account-base", "account-continuation") else False
        ),
        _personal_detail_extraction_issues=[],
    )


def test_collapsed_employment_rows_bind_only_unique_business_fields() -> None:
    table = _table(
        "employment-collapsed",
        ["编号工作单位单位性质单位地址单位电话"],
        ["1 福建海峡粮油购销有限公司 国有企业 鼓楼区鼓屏路60号13层 059100000000"],
        ["2 北方星河科技有限公司 国有企业 私营企业 海州市新城路8号 01012345678"],
    )
    result = _result(table)

    records = _extract_employment_records(result)

    assert [record["sequence"] for record in records] == [1, 2]
    assert {
        field: records[0][field]
        for field in ("employer", "employer_type", "employer_address", "employer_phone")
    } == {
        "employer": "福建海峡粮油购销有限公司",
        "employer_type": "国有企业",
        "employer_address": "鼓楼区鼓屏路60号13层",
        "employer_phone": "059100000000",
    }

    ambiguous = records[1]
    assert ambiguous["employer_phone"] == "01012345678"
    assert all(
        field not in ambiguous
        for field in ("employer", "employer_type", "employer_address")
    )
    unresolved = {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue.get("target_record_id") == ambiguous["employment_record_id"]
        and issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
    }
    assert {"employer", "employer_type", "employer_address"} <= unresolved


def test_collapsed_account_terms_bind_typed_fields_without_column_order_guessing() -> None:
    table = _table(
        "account-terms-collapsed",
        ["到期日期 借款金额 账户币种"],
        ["2024.06.14 6,100,000 人民币元"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:non_revolving_loan:1", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["due_date"] == "2024-06-14"
    assert account["contract_maturity_date"] == "2024-06-14"
    assert account["loan_amount"] == 6_100_000
    assert account["currency"] == "CNY"
    assert account["account_currency"] == "CNY"
    assert result._personal_detail_extraction_issues == []


def test_collapsed_account_terms_withhold_only_ambiguous_date() -> None:
    table = _table(
        "account-terms-ambiguous",
        ["到期日期 借款金额 账户币种"],
        ["2024.06.14 2025.06.14 6,100,000 人民币元"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:non_revolving_loan:2", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert "due_date" not in account
    assert account["loan_amount"] == 6_100_000
    assert account["currency"] == "CNY"
    assert any(
        issue.get("target_record_id") == account["account_id"]
        and issue.get("field_name") == "due_date"
        and issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


def test_two_cell_account_header_recovers_only_invariant_typed_values() -> None:
    first_raw = (
        "B10211000H0001 样例银行股份 2019.06.13 "
        "350220190204838 有限公司示例分行"
    )
    terms_raw = "7 2043.06.12 2,950,000 人民币元 B"
    table = _table(
        "account-two-cell-clusters",
        ["管理机构 开立日期 账户标识", "借款金额 账户币种 到期日期"],
        [first_raw, terms_raw],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:two-cell", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["open_date"] == "2019-06-13"
    assert account["due_date"] == "2043-06-12"
    assert account["loan_amount"] == 2_950_000
    assert account["currency"] == account["account_currency"] == "CNY"
    assert "management_institution" not in account
    assert "account_identifier" not in account
    assert account["canonical_raw"]["open_date"] == first_raw
    assert account["canonical_raw"]["loan_amount"] == terms_raw

    invalid_fields = {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
    }
    assert {"management_institution", "account_identifier"} <= invalid_fields
    residue_issues = [
        issue
        for issue in result._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_account_cluster_residue_unresolved"
    ]
    assert {issue["field_name"] for issue in residue_issues} == {
        "open_date",
        "due_date",
        "loan_amount",
        "currency",
    }
    assert any(
        issue["observed_value"]
        == {"raw_cluster": terms_raw, "unconsumed_residue": "7B"}
        for issue in residue_issues
    )


def test_header_suffix_institution_binds_without_a_value_row_cell() -> None:
    header_raw = "管理机构 福特汽车金融(中 国)有限公司"
    table = _table(
        "account-header-suffix-institution",
        [header_raw, "账户标识", "开立日期", "到期日期", "借款金额", "账户币种"],
        ["", "38053363", "2013.07.12", "--", "120,000", "人民币元"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:suffix", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["management_institution"] == "福特汽车金融(中国)有限公司"
    assert account["canonical_raw"]["management_institution"] == header_raw
    assert account["open_date"] == "2013-07-12"
    assert account["loan_amount"] == 120_000
    assert account["currency"] == "CNY"
    assert not any(
        issue.get("field_name") == "management_institution"
        for issue in result._personal_detail_extraction_issues
    )


def test_header_suffix_with_two_institution_spans_is_withheld() -> None:
    header_raw = "管理机构 甲银行股份有限公司 乙银行股份有限公司"
    table = _table(
        "account-header-suffix-ambiguous",
        [header_raw, "账户标识", "开立日期"],
        ["", "A123456789", "2020.01.02"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:suffix-ambiguous", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert "management_institution" not in account
    assert any(
        issue.get("field_name") == "management_institution"
        and issue.get("observed_value") == [header_raw]
        for issue in result._personal_detail_extraction_issues
    )


def test_standard_account_slots_recover_independently_of_noisy_neighbors() -> None:
    table = _table(
        "account-standard-independent-slots",
        ["管理机构", "账户标识", "开立日期", "到期日期", "借款金额", "账户币种"],
        [
            "样例银行股份有限公司",
            "两个 标识 值",
            "2013.07.12",
            '"',
            "120,000",
            "人民币元",
        ],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:standard", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["management_institution"] == "样例银行股份有限公司"
    assert account["open_date"] == "2013-07-12"
    assert account["loan_amount"] == 120_000
    assert account["currency"] == "CNY"
    assert "account_identifier" not in account
    assert "due_date" not in account


def test_collapsed_first_group_never_assigns_one_date_to_two_roles() -> None:
    raw = (
        "样例银行股份有限公司 A123456789 "
        "2020.01.02 2020.01.02 120,000 人民币元"
    )
    table = _table(
        "account-first-group-duplicate-date",
        ["账户币种 借款金额 到期日期 开立日期 账户标识 管理机构"],
        [raw],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:duplicate-date", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert "open_date" not in account
    assert "due_date" not in account
    assert account["loan_amount"] == 120_000
    assert account["currency"] == account["account_currency"] == "CNY"
    assert {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
    } >= {"open_date", "due_date", "management_institution", "account_identifier"}


def test_permuted_finite_account_clusters_recover_only_unique_roles() -> None:
    table = _table(
        "account-finite-cell-clusters",
        ["担保方式 还款期数 业务种类", "还款频率 共同借款标志 还款方式"],
        ["个人住房商业贷款 抵押 288", "穿期等额本息 无 月"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:finite", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["business_type"] == "个人住房商业贷款"
    assert account["guarantee_type"] == "抵押"
    assert account["repayment_periods"] == 288
    assert account["repayment_frequency"] == "月"
    assert "repayment_method" not in account
    assert "co_borrower_flag" not in account
    assert {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_cluster_field_unresolved"
    } >= {"repayment_method", "co_borrower_flag"}


def test_clean_finite_account_cluster_recovers_all_roles() -> None:
    table = _table(
        "account-finite-cell-clean",
        ["还款方式 共同借款标志 还款频率"],
        ["月 分期等额本息 无"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:finite-clean", "canonical_raw": {}}

    _apply_account_facts(
        result,
        account,
        table.metadata["raw_rows"],
        page=page,
        table=table,
    )

    assert account["repayment_frequency"] == "月"
    assert account["repayment_method"] == "分期等额本息"
    assert account["co_borrower_flag"] == "无"
    assert result._personal_detail_extraction_issues == []


def test_special_transaction_whole_cell_cluster_materializes_one_typed_event() -> None:
    table = _table(
        "special-transaction-collapsed",
        ["变更月数 发生日期 发生金额 明细记录 特殊交易类型"],
        ["提前还款(全部),变更月数-55个月 2020.05.25 4,200,000 55 提前结清"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:non_revolving_loan:1"}

    events = _account_events(result, account, page, table, table.metadata["raw_rows"])

    assert len(events) == 1
    assert {
        field: events[0][field]
        for field in ("transaction_type", "event_date", "changed_months", "amount", "details")
    } == {
        "transaction_type": "提前还款(全部),变更月数-55个月",
        "event_date": "2020-05-25",
        "changed_months": 55,
        "amount": 4_200_000,
        "details": "提前结清",
    }
    assert result._personal_detail_extraction_issues == []


def test_two_collapsed_large_installment_pairs_materialize_one_typed_event() -> None:
    table = _table(
        "large-installment-collapsed",
        ["分期额度生效日期 大额专项分期额度"],
        ["30,000 2023.10.18"],
        ["分期额度到期日期 已用分期金额"],
        ["2024.02.14 0"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:credit_card:1"}

    events = _account_events(result, account, page, table, table.metadata["raw_rows"])

    assert len(events) == 1
    assert {
        field: events[0][field]
        for field in (
            "installment_limit",
            "effective_date",
            "expiry_date",
            "used_installment_amount",
        )
    } == {
        "installment_limit": 30_000,
        "effective_date": "2023-10-18",
        "expiry_date": "2024-02-14",
        "used_installment_amount": 0,
    }
    assert result._personal_detail_extraction_issues == []


def test_ambiguous_special_transaction_reports_each_unresolved_slot() -> None:
    table = _table(
        "special-transaction-ambiguous",
        ["特殊交易类型 发生日期 变更月数 发生金额 明细记录"],
        ["提前结清 2020.05.25 55 1 已结清"],
    )
    page = _page(table)
    result = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:non_revolving_loan:2"}

    events = _account_events(result, account, page, table, table.metadata["raw_rows"])

    assert len(events) == 1
    event = events[0]
    assert event["event_date"] == "2020-05-25"
    assert all(
        field not in event
        for field in ("transaction_type", "changed_months", "amount", "details")
    )
    unresolved = {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue.get("target_record_id") == event["account_event_id"]
        and issue.get("issue_code") == "candidate_b_account_event_slot_unresolved"
    }
    assert {"transaction_type", "changed_months", "amount", "details"} <= unresolved


@pytest.mark.parametrize(
    ("header", "continuation_rows", "event_type", "expected"),
    [
        (
            ["特殊交易类型", "发生日期", "变更月数", "发生金额", "明细记录"],
            [["提前结清", "2020.05.25", "55", "4,200,000", "提前还款"]],
            "special_transaction",
            {
                "transaction_type": "提前结清",
                "event_date": "2020-05-25",
                "changed_months": 55,
                "amount": 4_200_000,
                "details": "提前还款",
            },
        ),
        (
            ["五级分类", "余额", "还款日期", "还款金额", "当前还款状态"],
            [["正常", "1,200", "2024.01.02", "300", "正常"]],
            "latest_repayment",
            {
                "five_tier_class": "正常",
                "balance": 1_200,
                "repayment_date": "2024-01-02",
                "repayment_amount": 300,
                "repayment_status": "正常",
            },
        ),
        (
            ["大额专项分期额度", "分期额度生效日期"],
            [
                ["30,000", "2023.10.18"],
                ["分期额度到期日期", "已用分期金额"],
                ["2024.02.14", "0"],
            ],
            "large_installment",
            {
                "installment_limit": 30_000,
                "effective_date": "2023-10-18",
                "expiry_date": "2024-02-14",
                "used_installment_amount": 0,
            },
        ),
    ],
)
def test_account_event_header_is_consumed_only_by_affirmed_continuation(
    header: list[str],
    continuation_rows: list[list[str]],
    event_type: str,
    expected: dict[str, object],
) -> None:
    result = _split_account_result(
        _account_base_rows(header),
        continuation_rows,
    )

    accounts, _repayments, events = _extract_table_accounts(result)

    assert len(accounts) == 1
    assert len(events) == 1
    assert events[0]["event_type"] == event_type
    assert {field: events[0][field] for field in expected} == expected
    assert not any(
        issue.get("issue_code") == "candidate_b_account_event_continuation_unresolved"
        for issue in result._personal_detail_extraction_issues
    )
    if event_type == "latest_repayment":
        # The event header also contains 余额; it must not be interpreted as an
        # account-fact header while the typed continuation is pending.
        assert "balance" not in accounts[0]


def test_account_event_header_at_eof_is_reported_instead_of_silently_lost() -> None:
    header = ["特殊交易类型", "发生日期", "变更月数", "发生金额", "明细记录"]
    table = _table("account-base", *_account_base_rows(header))
    result = _result(table)

    _accounts, _repayments, events = _extract_table_accounts(result)

    assert events == []
    unresolved = [
        issue
        for issue in result._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_account_event_continuation_unresolved"
    ]
    assert len(unresolved) == 1
    assert unresolved[0]["target_dataset"] == "credit_account_special_transactions"
    assert unresolved[0]["observed_value"]["boundary"] == "end_of_document"
    assert not any(
        issue.get("field_name") in {"balance", "repayment_date"}
        for issue in result._personal_detail_extraction_issues
    )


def test_large_installment_second_header_is_typed_across_affirmed_continuation() -> None:
    result = _split_account_result(
        _account_base_rows(
            ["大额专项分期额度", "分期额度生效日期"],
            ["30,000", "2023.10.18"],
            ["分期额度到期日期", "已用分期金额"],
        ),
        [["2024.02.14", "0"]],
    )

    _accounts, _repayments, events = _extract_table_accounts(result)

    assert len(events) == 1
    assert {
        field: events[0][field]
        for field in (
            "installment_limit",
            "effective_date",
            "expiry_date",
            "used_installment_amount",
        )
    } == {
        "installment_limit": 30_000,
        "effective_date": "2023-10-18",
        "expiry_date": "2024-02-14",
        "used_installment_amount": 0,
    }
    assert not any(
        issue.get("field_name") in {"expiry_date", "used_installment_amount"}
        or issue.get("issue_code") == "candidate_b_account_event_continuation_unresolved"
        for issue in result._personal_detail_extraction_issues
    )


@pytest.mark.parametrize("boundary", ["next_account", "next_section", "end_of_document"])
def test_trailing_optional_account_fact_labels_are_consumed_or_reported(boundary: str) -> None:
    trailing = ["还款频率", "还款方式"]
    first = _table("account-base", *_account_base_rows(trailing))
    tables = [first]
    if boundary == "next_account":
        tables.append(_table("next-account", *_account_base_rows()))
    elif boundary == "next_section":
        tables.append(_table("credit-agreement", ["授信协议标识", "授信额度用途"]))
    result = _result(*tables)

    accounts, _repayments, _events = _extract_table_accounts(result)

    unresolved_fields = {
        issue.get("field_name")
        for issue in result._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
        and issue.get("target_record_id") == accounts[0]["account_id"]
    }
    assert {"repayment_frequency", "repayment_method"} <= unresolved_fields
    assert accounts[0]["_terminal_unresolved_fact_boundaries"] == [boundary]


def test_trailing_optional_account_fact_labels_bind_across_affirmed_continuation() -> None:
    result = _split_account_result(
        _account_base_rows(["还款频率", "还款方式"]),
        [["月", "等额本息"]],
    )

    accounts, _repayments, _events = _extract_table_accounts(result)

    assert accounts[0]["repayment_frequency"] == "月"
    assert accounts[0]["repayment_method"] == "等额本息"
    assert "_terminal_unresolved_fact_boundaries" not in accounts[0]
    assert not any(
        issue.get("field_name") in {"repayment_frequency", "repayment_method"}
        for issue in result._personal_detail_extraction_issues
    )
