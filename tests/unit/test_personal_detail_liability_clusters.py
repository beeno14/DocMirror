from __future__ import annotations

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.liability_clusters import (
    CANONICAL_LIABILITY_LABELS,
    decode_packed_liability_row,
    liability_status_projection_field,
    normalize_packed_liability_header,
)

MERCEDES_HEADER = "管理机构 业务种类 开立日期 到期日期 贵任人类型 还款贵任金额 币种 保证合同编号"
MERCEDES_RAW_VALUE = (
    "梅赛德斯-奔 驰汽车金融有 限公司 贷款 2021.08.03 2024.08.03 "
    "保证 629,860 多 民币元 Y10061000H 0001EIP1967 714G01"
)
WEIZHONG_HEADER = "管理机构 业务种类 成立日期 到期日期 责任人类型 还款贵任金额 币种 保证合同编号"
WEIZHONG_RAW_VALUE = (
    "深圳前海微众 银行股份有限 公司 爱 贷款 2022.02.28 囍 2023.02.28 "
    "保证人 341,000 人民币元 D10055840H 0001DB2022 0228XS0000 00109"
)
HUANENG_HEADER = "管理机构 业务种类 成立日期 到期日期 责任人类型 还款责任金额 币种 保证合同编号"
HUANENG_RAW_VALUE = (
    "华能贵诚信托 有限公司 贷款 2022.09.02 2024.09.07 保证人 福 成 56,000 "
    "人民币元 70105501018 BZYQ202209 02XS0M000 00460"
)


def test_exact_known_header_aliases_normalize_to_complete_canonical_sequence() -> None:
    assert normalize_packed_liability_header(MERCEDES_HEADER) == CANONICAL_LIABILITY_LABELS
    assert normalize_packed_liability_header(WEIZHONG_HEADER) == CANONICAL_LIABILITY_LABELS
    assert normalize_packed_liability_header(HUANENG_HEADER) == CANONICAL_LIABILITY_LABELS


def test_clean_mercedes_witness_decodes_to_eight_typed_fields() -> None:
    decoded = decode_packed_liability_row(
        MERCEDES_HEADER,
        (
            "梅赛德斯-奔 驰汽车金融有 限公司 贷款 2021.08.03 2024.08.03 "
            "保证 629,860 人民币元 Y10061000H 0001EIP1967 714G01"
        ),
    )

    assert decoded.resolved is True
    assert decoded.unresolved_reason is None
    assert decoded.fields == {
        "institution": "梅赛德斯-奔驰汽车金融有限公司",
        "business_type": "贷款",
        "open_date": "2021-08-03",
        "due_date": "2024-08-03",
        "responsibility_type": "保证",
        "responsibility_amount": 629860,
        "currency": "CNY",
        "contract_number": "Y10061000H0001EIP1967714G01",
    }


def test_clean_packed_liability_accepts_extended_exact_currency_alias() -> None:
    decoded = decode_packed_liability_row(
        " ".join(CANONICAL_LIABILITY_LABELS),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 "
            "保证人 1,000 加拿大元 HT0000001"
        ),
    )

    assert decoded.resolved is True
    assert decoded.fields["currency"] == "CAD"


def test_clean_packed_liability_accepts_extended_exact_iso_code() -> None:
    decoded = decode_packed_liability_row(
        " ".join(CANONICAL_LIABILITY_LABELS),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 "
            "保证人 1,000 CHF HT0000001"
        ),
    )

    assert decoded.resolved is True
    assert decoded.fields["currency"] == "CHF"


def test_packed_liability_with_multiple_currencies_withholds_currency() -> None:
    decoded = decode_packed_liability_row(
        " ".join(CANONICAL_LIABILITY_LABELS),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 "
            "保证人 1,000 澳元美元 HT0000001"
        ),
    )

    assert decoded.resolved is False
    assert "currency" not in decoded.fields
    assert decoded.unresolved_reason == "typed_spans_with_ocr_residue"


def test_packed_liability_rejects_arbitrary_three_letter_currency_code() -> None:
    decoded = decode_packed_liability_row(
        " ".join(CANONICAL_LIABILITY_LABELS),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 "
            "保证人 1,000 ZZZ HT0000001"
        ),
    )

    assert decoded.resolved is False
    assert "currency" not in decoded.fields
    assert decoded.unresolved_reason == "typed_spans_with_ocr_residue"


@pytest.mark.parametrize(
    ("header", "value", "expected_fields", "missing_fields"),
    [
        (
            MERCEDES_HEADER,
            MERCEDES_RAW_VALUE,
            {
                "institution": "梅赛德斯-奔驰汽车金融有限公司",
                "business_type": "贷款",
                "open_date": "2021-08-03",
                "due_date": "2024-08-03",
                "responsibility_type": "保证",
                "responsibility_amount": 629860,
                "contract_number": "Y10061000H0001EIP1967714G01",
            },
            {"currency"},
        ),
        (
            WEIZHONG_HEADER,
            WEIZHONG_RAW_VALUE,
            {
                "institution": "深圳前海微众银行股份有限公司",
                "business_type": "贷款",
                "open_date": "2022-02-28",
                "due_date": "2023-02-28",
                "responsibility_type": "保证人",
                "responsibility_amount": 341000,
                "currency": "CNY",
                "contract_number": "D10055840H0001DB20220228XS000000109",
            },
            set(),
        ),
        (
            HUANENG_HEADER,
            HUANENG_RAW_VALUE,
            {
                "institution": "华能贵诚信托有限公司",
                "business_type": "贷款",
                "open_date": "2022-09-02",
                "due_date": "2024-09-07",
                "responsibility_type": "保证人",
                "responsibility_amount": 56000,
                "currency": "CNY",
                "contract_number": "70105501018BZYQ20220902XS0M00000460",
            },
            set(),
        ),
    ],
)
def test_exact_observed_witnesses_retain_safe_fields_and_report_ocr_residue(
    header: str,
    value: str,
    expected_fields: dict[str, str | int],
    missing_fields: set[str],
) -> None:
    decoded = decode_packed_liability_row(header, value)

    assert decoded.resolved is False
    assert decoded.fields == expected_fields
    assert missing_fields.isdisjoint(decoded.fields)
    assert decoded.unresolved_reason == "typed_spans_with_ocr_residue"


@pytest.mark.parametrize(
    ("header", "value", "expected"),
    [
        (
            WEIZHONG_HEADER,
            (
                "深圳前海微众 银行股份有限 公司 贷款 2022.02.28 2023.02.28 保证人 "
                "341,000 人民币元 D10055840H 0001DB2022 0228XS0000 00109"
            ),
            {
                "institution": "深圳前海微众银行股份有限公司",
                "open_date": "2022-02-28",
                "due_date": "2023-02-28",
                "responsibility_amount": 341000,
                "contract_number": "D10055840H0001DB20220228XS000000109",
            },
        ),
        (
            HUANENG_HEADER,
            (
                "华能贵诚信托 有限公司 贷款 2022.09.02 2024.09.07 保证人 56,000 "
                "人民币元 70105501018 BZYQ202209 02XS0M000 00460"
            ),
            {
                "institution": "华能贵诚信托有限公司",
                "open_date": "2022-09-02",
                "due_date": "2024-09-07",
                "responsibility_amount": 56000,
                "contract_number": "70105501018BZYQ20220902XS0M00000460",
            },
        ),
    ],
)
def test_later_examples_decode_after_ocr_residue_is_absent(
    header: str,
    value: str,
    expected: dict[str, str | int],
) -> None:
    decoded = decode_packed_liability_row(header, value)

    assert decoded.resolved is True
    assert decoded.fields["business_type"] == "贷款"
    assert decoded.fields["responsibility_type"] == "保证人"
    assert decoded.fields["currency"] == "CNY"
    for field_name, expected_value in expected.items():
        assert decoded.fields[field_name] == expected_value


@pytest.mark.parametrize(
    "header",
    [
        "管理机构 业务种类 开立日期 到期日期 责仼人类型 还款责任金额 币种 保证合同编号",
        "管理机构 业务种类 开立日期 到期日期 责任人类型 责任人类型 还款责任金额 币种 保证合同编号",
        "管理机构 业务种类 开立日期 责任人类型 到期日期 还款责任金额 币种 保证合同编号",
        "管理机构 业务种类 开立日期 到期日期 责任人类型 还款责任金额 币种",
    ],
)
def test_unknown_duplicate_out_of_order_or_incomplete_headers_are_rejected(header: str) -> None:
    decoded = decode_packed_liability_row(
        header,
        "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 保证人 1,000 人民币元 HT0000001",
    )

    assert decoded.resolved is False
    assert decoded.fields == {}
    assert decoded.unresolved_reason == "header_not_canonical"


@pytest.mark.parametrize(
    ("value", "expected_absent"),
    [
        (
            "示例银行股份有限公司 未知业务 2021.01.01 2022.01.01 保证人 1,000 人民币元 HT0000001",
            {"institution", "business_type"},
        ),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 保证人 -1,000 人民币元 HT0000001",
            {"responsibility_amount"},
        ),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 保证人 1,000 人民币币 HT0000001",
            {"currency"},
        ),
        (
            "示例银行股份有限公司 贷款 2021.01.01 2022.01.01 保证人 1,000 人民币元 HT0000001 附注",
            {"contract_number"},
        ),
        (
            "示例银行股份有限公司 贷款 2021.01.01 噪声 2022.01.01 保证人 1,000 人民币元 HT0000001",
            set(),
        ),
    ],
)
def test_ambiguous_or_untyped_value_residue_keeps_only_independently_typed_fields(
    value: str,
    expected_absent: set[str],
) -> None:
    decoded = decode_packed_liability_row(" ".join(CANONICAL_LIABILITY_LABELS), value)

    assert decoded.resolved is False
    assert decoded.fields["open_date"] == "2021-01-01"
    assert decoded.fields["due_date"] == "2022-01-01"
    assert expected_absent.isdisjoint(decoded.fields)
    assert decoded.unresolved_reason is not None


def test_source_status_label_keeps_overdue_months_distinct_from_repayment_status() -> None:
    assert liability_status_projection_field("逾期月数") == "overdue_months"
    assert liability_status_projection_field("还 款 状 态") == "repayment_status_code"
    assert liability_status_projection_field("逾期状态") is None
