# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite whole-cell decoders for canonical scanned personal-detail fields."""

from __future__ import annotations

from docmirror.plugins.credit_report.personal_detail_scanned.collapsed_clusters import (
    decode_employment_basic_cluster,
    decode_labeled_cluster,
)


def test_employment_cluster_decodes_exact_ye_company_row() -> None:
    result = decode_employment_basic_cluster(
        "福建海峡粮油购销有限公司 国有企业 鼓楼区鼓屏路60号13层 059100000000"
    )

    assert result.fields == {
        "employer": "福建海峡粮油购销有限公司",
        "employer_type": "国有企业",
        "employer_address": "鼓楼区鼓屏路60号13层",
        "employer_phone": "059100000000",
    }
    assert result.unresolved_residue == ""
    assert result.unresolved_fields == ()


def test_employment_cluster_with_repeated_school_values_withholds_ambiguous_roles() -> None:
    result = decode_employment_basic_cluster(
        "福建省福州市晋安区连江中路139号仁育特 殊儿童学校"
        "福州晋安区仁育特殊儿童学校 福州晋安区仁育特殊儿童学校 "
        "059183636123 行政部"
    )

    assert result.fields == {"employer_phone": "059183636123"}
    assert result.unresolved_fields == (
        "employer",
        "employer_type",
        "employer_address",
    )
    assert "仁育特殊儿童学校" in result.unresolved_residue
    assert "行政部" in result.unresolved_residue
    assert "059183636123" not in result.unresolved_residue


def test_employment_decoder_uses_field_contracts_not_known_business_values() -> None:
    result = decode_employment_basic_cluster(
        "北方星河科技有限公司 私营企业 海州市新城路8号 01012345678"
    )

    assert result.fields == {
        "employer": "北方星河科技有限公司",
        "employer_type": "私营企业",
        "employer_address": "海州市新城路8号",
        "employer_phone": "01012345678",
    }


def test_employment_decoder_never_splits_type_term_nested_in_legal_name() -> None:
    result = decode_employment_basic_cluster(
        "甲私营企业有限公司 福建省福州市星河路8号 01012345678"
    )

    assert result.fields == {"employer_phone": "01012345678"}
    assert result.unresolved_fields == (
        "employer",
        "employer_type",
        "employer_address",
    )
    assert "甲私营企业有限公司" in result.unresolved_residue
    assert "福建省福州市星河路8号" in result.unresolved_residue


def test_employment_decoder_does_not_choose_between_two_employer_types() -> None:
    result = decode_employment_basic_cluster(
        "北方星河科技有限公司 国有企业 私营企业 海州市新城路8号 01012345678"
    )

    assert result.fields == {"employer_phone": "01012345678"}
    assert "employer_type" in result.unresolved_fields
    assert "国有企业私营企业" in result.unresolved_residue


def test_employment_decoder_does_not_choose_between_two_phone_observations() -> None:
    result = decode_employment_basic_cluster(
        "北方星河科技有限公司 私营企业 海州市新城路8号 01012345678 01087654321"
    )

    assert "employer_phone" not in result.fields
    assert "employer_phone" in result.unresolved_fields
    assert "01012345678" in result.unresolved_residue
    assert "01087654321" in result.unresolved_residue


def test_account_term_cluster_uses_date_money_and_currency_contracts() -> None:
    result = decode_labeled_cluster(
        "到期日期 借款金额 账户币种 2024.06.14 6,100,000 人民币元",
        kind="account_terms",
    )

    assert result.fields == {
        "due_date": "2024-06-14",
        "loan_amount": 6_100_000,
        "currency": "CNY",
    }
    assert result.unresolved_residue == ""
    assert result.unresolved_fields == ()


def test_account_term_cluster_accepts_extended_exact_currency_alias() -> None:
    result = decode_labeled_cluster(
        "到期日期 借款金额 账户币种 2024.06.14 6,100,000 澳元",
        kind="account_terms",
    )

    assert result.fields == {
        "due_date": "2024-06-14",
        "loan_amount": 6_100_000,
        "currency": "AUD",
    }
    assert result.unresolved_residue == ""
    assert result.unresolved_fields == ()


def test_account_term_cluster_with_multiple_currencies_withholds_currency() -> None:
    result = decode_labeled_cluster(
        "到期日期 借款金额 账户币种 2024.06.14 6,100,000 澳元美元",
        kind="account_terms",
    )

    assert result.fields == {"due_date": "2024-06-14", "loan_amount": 6_100_000}
    assert result.unresolved_fields == ("currency",)
    assert "澳元美元" in result.unresolved_residue


def test_account_term_cluster_rejects_arbitrary_three_letter_code() -> None:
    result = decode_labeled_cluster(
        "到期日期 借款金额 账户币种 2024.06.14 6,100,000 ZZZ",
        kind="account_terms",
    )

    assert result.fields == {"due_date": "2024-06-14", "loan_amount": 6_100_000}
    assert result.unresolved_fields == ("currency",)
    assert "ZZZ" in result.unresolved_residue


def test_account_term_cluster_with_two_dates_withholds_only_ambiguous_date() -> None:
    result = decode_labeled_cluster(
        "到期日期 借款金额 账户币种 2024.06.14 2025.06.14 6,100,000 人民币元",
        kind="account_terms",
    )

    assert result.fields == {"loan_amount": 6_100_000, "currency": "CNY"}
    assert result.unresolved_fields == ("due_date",)
    assert "2024.06.14" in result.unresolved_residue
    assert "2025.06.14" in result.unresolved_residue


def test_special_transaction_cluster_decodes_unique_typed_fields() -> None:
    result = decode_labeled_cluster(
        "变更月数 发生日期 发生金额 明细记录 特殊交易类型 "
        "提前还款(全部),变更月数-55个月 2020.05.25 4,200,000 55 提前结清",
        kind="special_transaction",
    )

    assert result.fields == {
        "transaction_type": "提前还款(全部),变更月数-55个月",
        "event_date": "2020-05-25",
        "changed_months": 55,
        "amount": 4_200_000,
        "details": "提前结清",
    }
    assert result.unresolved_residue == ""
    assert result.unresolved_fields == ()


def test_special_transaction_cluster_does_not_guess_two_plain_numbers() -> None:
    result = decode_labeled_cluster(
        "特殊交易类型 发生日期 变更月数 发生金额 明细记录 "
        "提前结清 2020.05.25 55 1 已结清",
        kind="special_transaction",
    )

    assert result.fields == {"event_date": "2020-05-25"}
    assert set(result.unresolved_fields) == {
        "transaction_type",
        "changed_months",
        "amount",
        "details",
    }
    assert "55 1" in result.unresolved_residue


def test_large_installment_pair_clusters_decode_without_positional_swapping() -> None:
    result = decode_labeled_cluster(
        (
            "分期额度生效日期 大额专项分期额度 30,000 2023.10.18",
            "分期额度到期日期 已用分期金额 2024.02.14 0",
        ),
        kind="large_installment",
    )

    assert result.fields == {
        "installment_limit": 30_000,
        "effective_date": "2023-10-18",
        "expiry_date": "2024-02-14",
        "used_installment_amount": 0,
    }
    assert result.unresolved_residue == ""
    assert result.unresolved_fields == ()


def test_unknown_label_order_is_not_reinterpreted() -> None:
    raw = "借款金额 到期日期 账户币种 6,100,000 2024.06.14 人民币元"
    result = decode_labeled_cluster(raw, kind="account_terms")

    assert result.fields == {}
    assert result.unresolved_residue == raw
    assert result.unresolved_fields == ("due_date", "loan_amount", "currency")
