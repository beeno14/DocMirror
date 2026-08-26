# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.business_repair_policy import (
    bounded_inquiry_sequence_noise_candidate,
    deterministic_agreement_institution_candidate,
    deterministic_inquiry_date_candidate,
    deterministic_liability_business_type_candidate,
    liability_business_type_is_valid,
    separated_leading_han_company_boundary,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("2022.06.16 广", "2022-06-16"),
        ("字 2021/11/26", "2021-11-26"),
        ("2023.01.03 20", None),
        ("2021.11.26 22", None),
        ("2022.06.16", None),
        ("2022.06.16 广 2022.06.17", None),
        ("2022.13.16 广", None),
        ("2022.06.16 广文字", None),
        ("广 2022.06.16 字", None),
        ("Noo 2022.04.14", "2022-04-14"),
    ),
)
def test_inquiry_date_policy_only_deletes_short_nonnumeric_edge_residue(
    raw: str,
    expected: str | None,
) -> None:
    assert deterministic_inquiry_date_candidate(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("8 游", (8, "suffix_noise")),
        ("敬9", (9, "prefixed_noise")),
        ("8", None),
        ("游 敬 8", None),
        ("8 9", None),
        ("0 游", None),
        ("游 10000", None),
    ),
)
def test_inquiry_sequence_policy_only_identifies_one_bounded_edge_glyph(
    raw: str,
    expected: tuple[int, str] | None,
) -> None:
    assert bounded_inquiry_sequence_noise_candidate(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (
            "上海浦东发展银行股份 有限公司信用卡中心 授信额度",
            "上海浦东发展银行股份有限公司信用卡中心",
        ),
        ("中国农业银行股份有限公司", None),
        ("中国农业银行股份有限公司未知文字", None),
        ("授信额度中国农业银行股份有限公司", None),
        ("中国农业银行授信额度", None),
    ),
)
def test_agreement_policy_requires_one_complete_trailing_label_and_legal_name(
    raw: str,
    expected: str | None,
) -> None:
    assert deterministic_agreement_institution_candidate(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("爱 贷款", "贷款"),
        ("贷款 爱", "贷款"),
        ("贷款", None),
        ("爱 喜 贷款", None),
        ("爱 未知业务", None),
    ),
)
def test_liability_business_type_policy_requires_one_edge_glyph_and_closed_value(
    raw: str,
    expected: str | None,
) -> None:
    assert deterministic_liability_business_type_candidate(raw) == expected


def test_liability_repair_policy_keeps_ambiguous_company_boundary_for_reocr() -> None:
    assert liability_business_type_is_valid("贷款") is True
    assert liability_business_type_is_valid("爱贷款") is False
    assert separated_leading_han_company_boundary(
        "密 厦门雯明轩商贸有限公司"
    ) is True
    assert separated_leading_han_company_boundary(
        "厦门雯玥轩商贸有限公司"
    ) is False
