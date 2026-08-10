# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
    normalize_pboc_field,
    validate_pboc_field,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _apply_account_facts,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)


def _table(rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id="account-scalars",
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[20.0, 20.0, 580.0, 300.0],
        confidence=0.96,
    )


def _page(table: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=2,
        source_page_number=1,
        tables=[table],
        texts=[],
        height=800.0,
    )


@pytest.mark.parametrize(
    ("role", "raw"),
    (
        ("repayment_frequency", "签月**"),
        ("repayment_frequency", "月38"),
        ("repayment_method", "4人"),
        ("repayment_method", "梦-"),
        ("guarantee_type", "品信用/无担保"),
        ("account_business_type", "个人经营性贷款 其他"),
        ("facility_type", "循环贷款额度 %6"),
        ("facility_type", "信用求共享额度"),
    ),
)
def test_account_scalar_contracts_do_not_guess_or_delete_residue(role: str, raw: str) -> None:
    assert normalize_pboc_field(raw, role) == raw
    assert validate_pboc_field(raw, role).valid is False


def test_account_scalar_contracts_preserve_exact_report_categories() -> None:
    expected = {
        ("repayment_frequency", "月"): "月",
        ("repayment_frequency", "不定期"): "不定期",
        ("repayment_method", "分期等额本息"): "分期等额本息",
        ("repayment_method", "按期结息,到期还本"): "按期结息，到期还本",
        ("guarantee_type", "信用/无担保"): "信用/无担保",
        ("guarantee_type", "组合(含保证)"): "组合（含保证）",
        ("account_business_type", "个人汽车消费贷款"): "个人汽车消费贷款",
        ("account_business_type", "融资租赁业务"): "融资租赁业务",
        ("facility_type", "信用卡共享额度"): "信用卡共享额度",
    }

    for (role, raw), canonical in expected.items():
        assert normalize_pboc_field(raw, role) == canonical
        assert validate_pboc_field(raw, role).valid is True


def test_native_account_slots_withhold_invalid_scalars_with_raw_witnesses() -> None:
    rows = [
        ["业务种类", "担保方式", "还款频率", "还款方式"],
        ["个人经营性贷款 其他", "品信用/无担保", "签 月 **", "4 人"],
    ]
    table = _table(rows)
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:bad-scalars", "canonical_raw": {}}

    _apply_account_facts(context, account, rows, page=_page(table), table=table)

    for field_name, raw in {
        "business_type": "个人经营性贷款 其他",
        "guarantee_type": "品信用/无担保",
        "repayment_frequency": "签 月 **",
        "repayment_method": "4 人",
    }.items():
        assert field_name not in account
        assert account["canonical_raw"][field_name] == [raw]
        assert any(
            issue.get("issue_code") == "candidate_b_exact_slot_value_invalid"
            and issue.get("field_name") == field_name
            and issue.get("observed_value") == [raw]
            for issue in context._personal_detail_extraction_issues
        )


def test_native_account_slots_publish_only_exact_canonical_scalars() -> None:
    rows = [
        ["业务种类", "担保方式", "还款频率", "还款方式"],
        ["个人汽车消费贷款", "组合(含保证)", "月", "按期结息,到期还本"],
    ]
    table = _table(rows)
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {"account_id": "credit_account:loan:good-scalars", "canonical_raw": {}}

    _apply_account_facts(context, account, rows, page=_page(table), table=table)

    assert account["business_type"] == "个人汽车消费贷款"
    assert account["guarantee_type"] == "组合（含保证）"
    assert account["repayment_frequency"] == "月"
    assert account["repayment_method"] == "按期结息，到期还本"
    assert context._personal_detail_extraction_issues == []


def test_final_validation_withholds_unknown_account_scalars_and_keeps_raw() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "credit_accounts": [
                {
                    "account_id": "credit_account:loan:final-scalars",
                    "account_identifier": "ABCD12345678",
                "business_type": "个人经营性贷款 其他",
                "guarantee_type": "品信用/无担保",
                "repayment_frequency": "月38",
                "repayment_method": "梦-",
                "source_refs": [{"logical_page": 2, "geometry_scope": "table"}],
            }
        ],
        "credit_lines": [
            {
                "credit_line_id": "credit_line:final-scalars",
                "facility_type": "循环贷款额度 %6",
                "source_refs": [{"logical_page": 2, "geometry_scope": "table"}],
            }
        ],
    }

    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")

    account = corrected["credit_accounts"][0]
    for field_name, raw in {
        "business_type": "个人经营性贷款 其他",
        "guarantee_type": "品信用/无担保",
        "repayment_frequency": "月38",
        "repayment_method": "梦-",
    }.items():
        assert account[field_name] is None
        assert account["canonical_raw"][field_name] == raw
    credit_line = corrected["credit_lines"][0]
    assert credit_line["facility_type"] is None
    assert credit_line["canonical_raw"]["facility_type"] == "循环贷款额度 %6"
    anomalies = overlay.audit()["cell_anomalies"]
    assert {item["field_name"] for item in anomalies} >= {
        "business_type",
        "guarantee_type",
        "repayment_frequency",
        "repayment_method",
        "facility_type",
    }
    assert all(item["normalized_value_withheld"] for item in anomalies)


def test_final_validation_does_not_fuzzy_change_known_business_type() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "credit_accounts": [
                {
                    "account_id": "credit_account:loan:car",
                    "account_identifier": "ABCD12345678",
                "business_type": "个人汽车消费贷款",
                "guarantee_type": "信用/免担保",
                "repayment_frequency": "一次性",
                "repayment_method": "一次性还本付息",
            }
        ]
    }

    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")

    assert corrected["credit_accounts"][0]["business_type"] == "个人汽车消费贷款"
    assert overlay.audit()["cell_anomalies"] == []
