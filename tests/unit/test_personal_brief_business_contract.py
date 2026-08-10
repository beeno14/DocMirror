# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.plugins.credit_report.personal_brief_native.contracts import (
    PERSONAL_BRIEF_ENUM_CONTRACT,
    PERSONAL_BRIEF_MONEY_FIELDS,
    PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION,
    PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
    PERSONAL_BRIEF_REPORTING_CURRENCY,
    PersonalBriefContractError,
    canonical_personal_brief_amount_unit,
    canonicalize_personal_brief_reporting_units,
    validate_personal_brief_public_record,
)
from docmirror.plugins.credit_report.personal_brief_native.projector import (
    _PUBLIC_BUSINESS_FIELDS,
    _project_business_dataset,
    personal_brief_public_dataset_policy,
)
from docmirror.plugins.credit_report.personal_brief_native.schema import (
    personal_brief_data_dictionary,
)


def _valid_contract_values(dataset_name: str) -> dict[str, object]:
    values: dict[str, object] = {}
    if dataset_name == "personal_report_metadata":
        values.update(
            {
                "reporting_currency": PERSONAL_BRIEF_REPORTING_CURRENCY,
                "reporting_amount_unit": PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
                "reporting_amount_precision": PERSONAL_BRIEF_REPORTING_AMOUNT_PRECISION,
            }
        )
    if dataset_name in PERSONAL_BRIEF_MONEY_FIELDS:
        values.update(
            {
                "reporting_amount_currency": PERSONAL_BRIEF_REPORTING_CURRENCY,
                "reporting_amount_unit": PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT,
            }
        )
    return values


def test_dataset_enum_metadata_exactly_matches_runtime_contract() -> None:
    dictionary = personal_brief_data_dictionary()["datasets"]
    policy = personal_brief_public_dataset_policy()

    published_enum_fields = {
        (dataset_name, field_name)
        for dataset_name, field_names in policy.items()
        if field_names is not None
        for field_name in field_names
        if dictionary[dataset_name]["columns"][field_name].get("type") == "enum"
    }

    assert published_enum_fields == set(PERSONAL_BRIEF_ENUM_CONTRACT)
    for contract_key, allowed_values in PERSONAL_BRIEF_ENUM_CONTRACT.items():
        dataset_name, field_name = contract_key
        assert dictionary[dataset_name]["columns"][field_name]["enum"] == allowed_values


def test_money_metadata_exactly_matches_runtime_contract() -> None:
    dictionary = personal_brief_data_dictionary()["datasets"]
    policy = personal_brief_public_dataset_policy()
    published_money_fields = {
        (dataset_name, field_name)
        for dataset_name, field_names in policy.items()
        if field_names is not None
        for field_name in field_names
        if dictionary[dataset_name]["columns"][field_name].get("type")
        in {"amount", "money"}
    }
    contract_money_fields = {
        (dataset_name, field_name)
        for dataset_name, field_names in PERSONAL_BRIEF_MONEY_FIELDS.items()
        for field_name in field_names
    }

    assert published_money_fields == contract_money_fields
    for dataset_name, field_name in sorted(contract_money_fields):
        assert (
            dictionary[dataset_name]["columns"][field_name]["unit"]
            == PERSONAL_BRIEF_REPORTING_AMOUNT_UNIT
        )


def test_every_declared_enum_value_is_accepted_by_runtime_contract() -> None:
    for (dataset_name, field_name), allowed_values in PERSONAL_BRIEF_ENUM_CONTRACT.items():
        for value in allowed_values:
            values = _valid_contract_values(dataset_name)
            values[field_name] = value
            if (
                dataset_name == "credit_accounts"
                and field_name == "payoff_state"
                and value == "unknown"
            ):
                values["account_lifecycle_state"] = "transferred_out"
            validate_personal_brief_public_record(
                dataset_name,
                f"{dataset_name}:1",
                values,
            )


@pytest.mark.parametrize(
    ("dataset_name", "field_name"),
    sorted(PERSONAL_BRIEF_ENUM_CONTRACT),
)
def test_each_closed_enum_rejects_an_undeclared_value(
    dataset_name: str,
    field_name: str,
) -> None:
    values = _valid_contract_values(dataset_name)
    values[field_name] = "__not_a_pboc_value__"

    with pytest.raises(
        PersonalBriefContractError,
        match=(
            "PERSONAL_BRIEF_ENUM_CONTRACT_VIOLATION.*"
            f"dataset='{dataset_name}'.*field='{field_name}'"
        ),
    ):
        validate_personal_brief_public_record(dataset_name, "record:invalid", values)


@pytest.mark.parametrize(
    ("field_name", "sentinel"),
    [
        ("credit_quality_status", "unresolved"),
        ("account_lifecycle_state", "unknown"),
    ],
)
def test_extraction_uncertainty_is_not_published_as_business_data(
    field_name: str,
    sentinel: str,
) -> None:
    values = {
        **_valid_contract_values("credit_accounts"),
        field_name: sentinel,
    }

    with pytest.raises(
        PersonalBriefContractError,
        match=f"PERSONAL_BRIEF_ENUM_CONTRACT_VIOLATION.*{field_name}",
    ):
        validate_personal_brief_public_record(
            "credit_accounts",
            "credit_account:unresolved",
            values,
        )


def test_unknown_payoff_is_limited_to_transferred_accounts() -> None:
    values = {
        **_valid_contract_values("credit_accounts"),
        "account_lifecycle_state": "open",
        "payoff_state": "unknown",
    }

    with pytest.raises(
        PersonalBriefContractError,
        match="PERSONAL_BRIEF_ENUM_RELATION_CONTRACT_VIOLATION",
    ):
        validate_personal_brief_public_record(
            "credit_accounts",
            "credit_account:invalid-payoff",
            values,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("reporting_currency", "USD"),
        ("reporting_amount_unit", "CNY_10K"),
        ("reporting_amount_precision", 2),
    ],
)
def test_metadata_rejects_invalid_amount_policy(
    field_name: str,
    invalid_value: object,
) -> None:
    values = _valid_contract_values("personal_report_metadata")
    values[field_name] = invalid_value

    with pytest.raises(
        PersonalBriefContractError,
        match="PERSONAL_BRIEF_AMOUNT_POLICY_CONTRACT_VIOLATION",
    ):
        validate_personal_brief_public_record(
            "personal_report_metadata",
            "metadata:invalid",
            values,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("reporting_amount_currency", "USD"),
        ("reporting_amount_unit", "CNY_10K"),
    ],
)
def test_money_dataset_rejects_invalid_reporting_currency_or_unit(
    field_name: str,
    invalid_value: object,
) -> None:
    values = _valid_contract_values("credit_accounts")
    values[field_name] = invalid_value

    with pytest.raises(
        PersonalBriefContractError,
        match="PERSONAL_BRIEF_MONEY_UNIT_CONTRACT_VIOLATION",
    ):
        validate_personal_brief_public_record(
            "credit_accounts",
            "credit_account:invalid",
            values,
        )


def test_projector_rejects_nonzero_account_amount_precision() -> None:
    values = {
        **_valid_contract_values("credit_accounts"),
        "account_type": "credit_card",
        "business_category": "credit_cards",
        "credit_card_type": "credit_card",
        "credit_limit_status": "reported",
        "used_amount_status": "not_reported",
        "balance_status": "reported",
        "account_lifecycle_state": "open",
        "card_activation_state": "activated",
        "payoff_state": "not_applicable",
        "credit_quality_status": "not_reported",
        "reporting_amount_precision": 2,
    }
    dataset = {
        "name": "credit_accounts",
        "rows": [
            {
                "record_id": "credit_account:invalid",
                "normalized": values,
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="unexpected personal-brief account amount precision: 2",
    ):
        _project_business_dataset(
            dataset,
            _PUBLIC_BUSINESS_FIELDS["credit_accounts"],
        )


def test_legacy_yuan_is_canonicalized_everywhere_before_publication() -> None:
    amount_policy = {"reporting_amount_unit": "yuan"}
    datasets = {
        "credit_accounts": [
            {
                "reporting_amount_unit": "yuan",
                "amount_unit": "yuan",
            }
        ],
        "tax_arrears_records": [{"reporting_amount_unit": "CNY_1"}],
    }

    canonicalize_personal_brief_reporting_units(
        datasets,
        amount_policy=amount_policy,
    )

    assert canonical_personal_brief_amount_unit("yuan") == "CNY_1"
    assert amount_policy["reporting_amount_unit"] == "CNY_1"
    assert datasets["credit_accounts"][0] == {
        "reporting_amount_unit": "CNY_1",
        "amount_unit": "CNY_1",
    }
    assert datasets["tax_arrears_records"][0]["reporting_amount_unit"] == "CNY_1"


def test_unknown_amount_unit_is_not_silently_canonicalized() -> None:
    datasets = {
        "credit_accounts": [
            {"reporting_amount_unit": "yuan"},
            {"reporting_amount_unit": "CNY_10K"},
        ]
    }
    before = deepcopy(datasets)

    with pytest.raises(
        PersonalBriefContractError,
        match="PERSONAL_BRIEF_AMOUNT_UNIT_CONTRACT_VIOLATION",
    ):
        canonicalize_personal_brief_reporting_units(datasets)

    assert datasets == before
