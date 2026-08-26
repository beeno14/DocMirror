# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.evidence.repair import RepairCandidate
from docmirror.plugins.credit_report.personal_detail_scanned import relations
from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessFieldRepair,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
    _final_account_field_is_valid,
    _reconcile_final_account_field_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
    make_issue,
    record_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import (
    validate_pboc_field,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _account_institution,
    _apply_account_facts,
    _date,
    _flush_pending_account_institution_observations,
    _liability_date,
    _number,
    _source_ref,
    reconcile_candidate_b_credit_lines,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
    institution_slot_is_unambiguous,
    normalize_institution_name,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.scanned_business import (
    _extract_inquiries,
    link_repayment_records_to_accounts,
)


def _overlay(**kwargs) -> PersonalDetailOCRCorrectionOverlay:
    return PersonalDetailOCRCorrectionOverlay(SimpleNamespace(), **kwargs)


def _one_shot_page_line(
    text: str,
    *,
    bbox: list[float],
    confidence: float = 0.99,
    page_key: str = "test-page",
    word_index: int = 0,
    content: str | None = None,
) -> dict[str, object]:
    line: dict[str, object] = {
        "text": text,
        "confidence": confidence,
        "bbox": bbox,
        "evidence_ids": [f"personal_detail_page_reocr:{page_key}:w{word_index}"],
        "source": "personal_detail_page_reocr_once",
    }
    if content is not None:
        line["content"] = content
    return line


def _planned_repair_ref(
    *,
    field_name: str = "inquiry_date",
    row: int = 1,
    column: int = 1,
    bbox: list[float] | None = None,
) -> dict[str, object]:
    return {
        "source": "native_detail_table_cell",
        "logical_page": 1,
        "source_page": 1,
        "table_id": "pt_1_0",
        "row": row,
        "column": column,
        "bbox": bbox or [10, 10, 40, 30],
        "evidence_ids": [f"native:{field_name}:{row}:{column}"],
        "geometry_scope": "cell",
        "field_name": field_name,
        "binding": "canonical_header_column",
    }


def _planned_repair(
    *,
    mode: str,
    observed: str,
    candidate: str | None,
    source_ref: dict[str, object],
) -> BusinessFieldRepair:
    return BusinessFieldRepair(
        repair_id="repair:1",
        uncertainty_id="uncertainty:1",
        mode=mode,
        dataset_name="inquiry_records",
        record_id="credit_inquiry:institution:3",
        field_name="inquiry_date",
        role="date",
        observed_value=observed,
        candidate_value=candidate,
        source_refs=(source_ref,),
        reason_codes=("focused_policy_fixture",),
    )


def test_planned_deterministic_repair_mutates_only_its_exact_owned_field() -> None:
    overlay = _overlay()
    target = _planned_repair_ref()
    repair = _planned_repair(
        mode="deterministic",
        observed="2022.06.16 广",
        candidate="2022-06-16",
        source_ref=target,
    )

    corrected, decision = overlay.repair_planned_text(
        "2022.06.16 广",
        repair=repair,
        source_refs=(target,),
    )
    wrong_owner, wrong_decision = overlay.repair_planned_text(
        "2022.06.16 广",
        repair=repair,
        source_refs=(_planned_repair_ref(column=2),),
    )

    assert corrected == "2022-06-16"
    assert decision is not None
    assert decision.method == "schema_bound_deterministic_field_repair"
    assert "no_ocr_acquisition" in decision.reason_codes
    assert wrong_owner == "2022.06.16 广"
    assert wrong_decision is None


def test_planned_context_rich_reocr_uses_page_context_but_only_target_cell() -> None:
    overlay = _overlay()
    target = _planned_repair_ref()
    repair = _planned_repair(
        mode="context_rich_reocr",
        observed="2023.01.03 20",
        candidate=None,
        source_ref=target,
    )
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "logical_page": 1,
                "source_page": 1,
                "page_key": "inquiry-date-context",
                "lines": [
                    _one_shot_page_line(
                        "2023.01.03",
                        bbox=[10, 10, 40, 30],
                        page_key="inquiry-date-context",
                    ),
                    _one_shot_page_line(
                        "unrelated page context",
                        bbox=[80, 80, 160, 95],
                        page_key="inquiry-date-context",
                        word_index=1,
                    ),
                ],
            }
        ],
        affected_pages={1},
    )

    corrected, decision = overlay.repair_planned_text(
        "2023.01.03 20",
        repair=repair,
        source_refs=(target,),
    )

    assert corrected == "2023-01-03"
    assert decision is not None
    assert decision.method == "schema_bound_page_evidence_reparse"
    assert decision.selected_acquisition == (
        "personal_detail_page_reocr_once:inquiry-date-context"
    )
    assert len(decision.source_refs) == 2
    assert decision.source_refs[0]["geometry_scope"] == "cell"
    assert decision.source_refs[1]["geometry_scope"] == "token_band"


def test_installed_reocr_evidence_is_sealed_to_planned_target_cells() -> None:
    overlay = _overlay()
    allowed_target = _planned_repair_ref()
    unrelated_target = _planned_repair_ref(
        row=2,
        bbox=[10, 40, 40, 60],
    )
    same_cell_other_field = {
        **allowed_target,
        "field_name": "birth_date",
    }
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "logical_page": 1,
                "source_page": 1,
                "page_key": "target-seal",
                "lines": [
                    _one_shot_page_line(
                        "2023.01.03",
                        bbox=[10, 10, 40, 30],
                        page_key="target-seal",
                    ),
                    _one_shot_page_line(
                        "2023.01.04",
                        bbox=[10, 40, 40, 60],
                        page_key="target-seal",
                        word_index=1,
                    ),
                ],
            }
        ],
        affected_pages={1},
        allowed_target_refs=(allowed_target,),
    )

    allowed, allowed_decision = overlay.correct_text(
        "2023.01.03 20",
        role="date",
        field_name="inquiry_date",
        source_refs=(allowed_target,),
    )
    unrelated, unrelated_decision = overlay.correct_text(
        "2023.01.04 21",
        role="date",
        field_name="inquiry_date",
        source_refs=(unrelated_target,),
    )
    same_cell, same_cell_decision = overlay.correct_text(
        "2023.01.03 20",
        role="date",
        field_name="birth_date",
        source_refs=(same_cell_other_field,),
    )

    assert allowed == "2023-01-03"
    assert allowed_decision is not None
    assert unrelated == "2023.01.04 21"
    assert unrelated_decision is None
    assert same_cell == "2023.01.03 20"
    assert same_cell_decision is None
    assert overlay.audit()["repair_evidence_target_count"] == 1


def test_context_rich_reocr_does_not_treat_compacted_ambiguous_name_as_repair() -> None:
    overlay = _overlay()
    target = _planned_repair_ref(
        field_name="related_party_name",
        row=3,
        column=0,
        bbox=[10, 40, 90, 60],
    )
    repair = BusinessFieldRepair(
        repair_id="repair:liability-name",
        uncertainty_id="uncertainty:liability-name",
        mode="context_rich_reocr",
        dataset_name="repayment_liability_records",
        record_id="repayment_liability:2",
        field_name="related_party_name",
        role="liability_related_party_name",
        observed_value="密 厦门雯明轩商贸有限公司",
        candidate_value=None,
        source_refs=(target,),
        reason_codes=("separated_leading_han_company_boundary",),
    )
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "logical_page": 1,
                "source_page": 1,
                "page_key": "unchanged-liability-name",
                "lines": [
                    _one_shot_page_line(
                        "密厦门雯明轩商贸有限公司",
                        bbox=[10, 40, 90, 60],
                        page_key="unchanged-liability-name",
                    )
                ],
            }
        ],
        affected_pages={1},
        allowed_target_refs=(target,),
    )

    corrected, decision = overlay.repair_planned_text(
        "密 厦门雯明轩商贸有限公司",
        repair=repair,
        source_refs=(target,),
    )

    assert corrected == "密 厦门雯明轩商贸有限公司"
    assert decision is None


@pytest.mark.parametrize("raw", ("1,2", "1,,2", "1,23,456", "12,34"))
def test_integer_parser_never_concatenates_malformed_grouping(raw: str) -> None:
    assert _number(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("0", 0), ("-12", -12), ("1,234", 1234), ("12,345,678", 12_345_678)),
)
def test_integer_parser_accepts_only_plain_or_registered_thousands_groups(
    raw: str,
    expected: int,
) -> None:
    assert _number(raw) == expected


@pytest.mark.parametrize("value", ("1,2", "1,,000", "1234,567", "$1,200", True))
def test_relations_number_rejects_malformed_presentation(value: object) -> None:
    assert relations._number(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    (("1,200", "1200"), ("12,345.67", "12345.67"), ("1234.5", "1234.5")),
)
def test_relations_number_accepts_registered_presentation(
    value: object,
    expected: str,
) -> None:
    assert relations._number(value) == expected


@pytest.mark.parametrize(
    ("raw", "role"),
    (
        ("1,2", "amount"),
        ("1,,2", "amount"),
        ("1,23,456", "nonnegative_integer"),
        ("$13800138007", "mobile_phone"),
        ("010-12345678?", "phone"),
        ("010--12345678", "phone"),
        ("010-12-345-678", "phone"),
        ("010))12345678", "phone"),
        ("1-2-3-4-5", "phone"),
        ("12345", "phone"),
        ("010-12345678 ext 9", "phone"),
        ("０１０－１２３４５６７８", "phone"),
        ("010-\u200b12345678", "phone"),
        ("１３８００１３８００７", "mobile_phone"),
        ("138\u200b0013\u200b8007", "mobile_phone"),
        ("138001380\u0660\u0667", "mobile_phone"),
        ("０１０１２３４５６７８", "phone"),
        ("\u0660\u0661\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668", "phone"),
        ("1O0", "amount"),
        ("B", "amount"),
        ("I2", "nonnegative_integer"),
        ("ABCD,1234", "account_identifier"),
        ("X2024-01-02Y", "date"),
        ("X20240102123456Y", "report_datetime"),
        ("正常,", "account_state"),
        ("正常；", "five_tier_class"),
        ("个人住房贷款,", "summary_business_category"),
        ("11O10519491231002X", "identity_document_number"),
        ("110105-19491231-002X", "identity_document_number"),
        ("110105 19491231 002X", "identity_document_number"),
    ),
)
def test_typed_overlay_never_deletes_unregistered_punctuation(
    raw: str,
    role: str,
) -> None:
    overlay = _overlay()

    normalized, decision = overlay.correct_text(raw, role=role)

    assert normalized == raw
    assert decision is None


@pytest.mark.parametrize(
    ("field_name", "raw"),
    (
        ("mobile_phone", "138\u200b0013\u200b8007"),
        ("mobile_phone", "１３８００１３８００７"),
        ("mobile_phone", "138001380\u0660\u0667"),
        ("phone", "０１０１２３４５６７８"),
        ("phone", "\u0660\u0661\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668"),
        ("phone", "010--12345678"),
    ),
)
def test_final_overlay_withholds_non_ascii_or_unregistered_phone_grammar(
    field_name: str,
    raw: str,
) -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())

    corrected = overlay.correct_business_candidates(
        {"phone_records": [{"record_id": "phone:1", field_name: raw}]},
        stage="candidate_b_final_validation",
    )

    record = corrected["phone_records"][0]
    assert record[field_name] is None
    assert record["canonical_raw"][field_name] == raw
    assert any(
        anomaly["field_name"] == field_name
        and anomaly["normalized_value_withheld"] is True
        for anomaly in overlay.audit()["cell_anomalies"]
    )


@pytest.mark.parametrize(
    ("raw", "role", "expected"),
    (
        ("1,234.50", "amount", "1234.50"),
        ("12,345", "nonnegative_integer", "12345"),
        ("+86 138-0013-8007", "mobile_phone", "13800138007"),
        ("138 0013 8007", "mobile_phone", "13800138007"),
        ("138 0013 8007", "phone", "13800138007"),
        ("(010) 12345678", "phone", "01012345678"),
        ("010-12345678", "phone", "01012345678"),
        ("010 12345678", "phone", "01012345678"),
    ),
)
def test_typed_overlay_accepts_only_registered_presentation_grammars(
    raw: str,
    role: str,
    expected: str,
) -> None:
    overlay = _overlay()

    normalized, decision = overlay.correct_text(raw, role=role)

    assert normalized == expected
    assert decision is not None


def test_typed_correction_is_role_scoped_and_preserves_invalid_values() -> None:
    overlay = _overlay()

    report_time, time_decision = overlay.correct_text(
        "2024.031311:05:32",
        role="report_datetime",
    )
    identity, identity_decision = overlay.correct_text(
        "11010519491231002 X",
        role="identity_document_number",
    )
    amount, amount_decision = overlay.correct_text("1O0", role="amount")
    signed_amount, signed_amount_decision = overlay.correct_text("+500", role="amount")
    birth_date, birth_date_decision = overlay.correct_text("1987.09.05", role="date")
    event_month, event_month_decision = overlay.correct_text("2018.10", role="date_or_month")
    invalid, invalid_decision = overlay.correct_text("amount S", role="amount")

    assert report_time == "2024-03-13T11:05:32+08:00"
    assert identity == "11010519491231002 X"
    assert amount == "1O0"
    assert signed_amount == "500"
    assert birth_date == "1987-09-05"
    assert event_month == "2018-10"
    assert invalid == "amount S"
    assert (
        time_decision
        and signed_amount_decision
        and birth_date_decision
        and event_month_decision
    )
    assert identity_decision is None
    assert amount_decision is None
    assert invalid_decision is None
    assert all(decision.action == "applied" for decision in overlay.decisions)


def test_final_overlay_uses_closed_employment_vocabularies() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    corrected = overlay.correct_business_candidates(
        {
            "employment_records": [
                {
                    "record_id": "employment:canonical",
                    "employment_record_id": "employment:canonical",
                    "industry": "批发和零售业",
                    "occupation": "商业、服务业人员",
                    "position": "一般员工",
                    "professional_title": "中级",
                },
                {
                    "record_id": "employment:polluted",
                    "employment_record_id": "employment:polluted",
                    "occupation": "商业、服务业人员X",
                },
            ]
        },
        stage="candidate_b_final_validation",
    )

    canonical, polluted = corrected["employment_records"]
    assert canonical["industry"] == "批发和零售业"
    assert canonical["occupation"] == "商业、服务业人员"
    assert canonical["position"] == "一般员工"
    assert canonical["professional_title"] == "中级"
    assert polluted["occupation"] is None
    anomalies = overlay.audit()["cell_anomalies"]
    assert not any(
        item["record_id"] == "employment:canonical"
        and item["field_name"]
        in {"industry", "occupation", "position", "professional_title"}
        for item in anomalies
    )
    assert any(
        item["record_id"] == "employment:polluted"
        and item["field_name"] == "occupation"
        and item["normalized_value_withheld"] is True
        for item in anomalies
    )


def test_final_overlay_validates_identity_number_against_exact_document_type() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    corrected = overlay.correct_business_candidates(
        {
            "identity_documents": [
                {
                    "record_id": "identity:passport",
                    "document_type": "护照",
                    "document_number": "E12345678",
                },
                {
                    "record_id": "identity:bad-passport",
                    "document_type": "护照",
                    "document_number": "E12?345",
                },
                {
                    "record_id": "identity:cn",
                    "document_type": "身份证",
                    "document_number": "11010519491231002X",
                },
                {
                    "record_id": "identity:bad-cn",
                    "document_type": "身份证",
                    "document_number": "110105194912310021",
                },
                {
                    "record_id": "identity:passport-with-cn",
                    "document_type": "护照",
                    "document_number": "11010519491231002X",
                },
                {
                    "record_id": "identity:cn-with-passport",
                    "document_type": "身份证",
                    "document_number": "E12345678",
                },
            ]
        },
        stage="candidate_b_final_validation",
    )

    rows = {
        row["record_id"]: row for row in corrected["identity_documents"]
    }
    assert rows["identity:passport"]["document_number"] == "E12345678"
    assert rows["identity:cn"]["document_number"] == "11010519491231002X"
    for record_id in (
        "identity:bad-passport",
        "identity:bad-cn",
        "identity:passport-with-cn",
        "identity:cn-with-passport",
    ):
        assert rows[record_id]["document_number"] is None
        assert rows[record_id]["canonical_raw"]["document_number"]

    anomalies = overlay.audit()["cell_anomalies"]
    assert not any(
        item["record_id"] in {"identity:passport", "identity:cn"}
        and item["field_name"] == "document_number"
        for item in anomalies
    )
    assert {
        item["record_id"]
        for item in anomalies
        if item["field_name"] == "document_number"
        and item["normalized_value_withheld"] is True
    } == {
        "identity:bad-passport",
        "identity:bad-cn",
        "identity:passport-with-cn",
        "identity:cn-with-passport",
    }


def test_institution_correction_removes_debris_without_general_fuzzy_rewrite() -> None:
    assert normalize_institution_name("重庆蚂蚊消费金融有限公司 Ss") == "重庆蚂蚊消费金融有限公司"
    assert normalize_institution_name("福 中信银行股份有限公司个人信贷部") == (
        "福中信银行股份有限公司个人信贷部"
    )
    assert _account_institution("福 中信银行股份有限公司个人信贷部") is None
    assert normalize_institution_name("未知机构名称") == "未知机构名称"
    assert normalize_institution_name("新疆样例银行股份有限公司乌鲁木齐分行") == (
        "新疆样例银行股份有限公司乌鲁木齐分行"
    )
    assert normalize_institution_name("云南省农村信用社联合社") == "云南省农村信用社联合社"
    contaminated = "开立日期账户授信额度共享授信额度币种业务种类担保方式中国建设银行股份有限公司"
    assert normalize_institution_name(contaminated) == contaminated
    assert not institution_slot_is_unambiguous(contaminated)


def test_pboc_institution_contract_covers_official_roles_without_near_match_repair() -> None:
    for value in (
        "中国人民银行营业管理部",
        "某市住房公积金管理中心",
        "某农村信用合作联社",
        "本人",
    ):
        contract = validate_pboc_field(value, "institution_name")
        assert contract.assessed and contract.valid

    for value in (
        "中国人民银行营业管理",
        "营业管理部备注",
        "任意机构名称残片",
    ):
        contract = validate_pboc_field(value, "institution_name")
        assert contract.assessed and not contract.valid


def test_institution_normalization_has_no_unconditional_business_name_aliases() -> None:
    assert normalize_institution_name("重庆蚂蚊消费金融有限公司") == "重庆蚂蚊消费金融有限公司"
    assert normalize_institution_name("某银行偏用卡中心") == "某银行偏用卡中心"
    assert normalize_institution_name("某银行个大信贷部") == "某银行个大信贷部"


def test_institution_leading_han_is_never_deleted_by_scalar_normalization() -> None:
    assert normalize_institution_name("中 国银行股份有限公司") == "中国银行股份有限公司"
    assert normalize_institution_name("中 信银行股份有限公司") == "中信银行股份有限公司"
    assert _account_institution("中 国银行股份有限公司") is None
    assert _account_institution("中 信银行股份有限公司") is None
    assert _account_institution("中国银行股份有限公司") == "中国银行股份有限公司"


@pytest.mark.parametrize(
    "raw",
    (
        "导中国银行股份有限公司",
        "S 中国银行股份有限公司",
        "中国银行股份有限公司 Ss",
    ),
)
def test_final_account_gate_cannot_validate_a_glyph_deleting_institution(
    raw: str,
) -> None:
    assert not _final_account_field_is_valid("management_institution", raw)


def test_final_account_gate_accepts_a_lossless_whole_institution() -> None:
    assert _final_account_field_is_valid(
        "management_institution",
        "中国银行股份有限公司",
    )


def test_final_overlay_withholds_uncorroborated_leading_han_business_names() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "inquiry_records": [
            {
                "inquiry_id": "inquiry:1",
                "institution": "福 中国银行股份有限公司",
            }
        ],
        "employment_records": [
            {
                "employment_record_id": "employment:1",
                "employer": "中 信银行股份有限公司",
                "data_provider": "福 中国银行股份有限公司",
            }
        ],
        "residence_records": [
            {
                "residence_record_id": "residence:1",
                "data_provider": "福 中国银行股份有限公司",
            }
        ],
        "personal_profile": [
            {
                "personal_profile_id": "profile:1",
                "query_institution": "福 中国银行股份有限公司",
            }
        ],
        "credit_lines": [
            {
                "credit_line_id": "line:clean",
                "institution": "中国银行股份有限公司",
            }
        ],
    }

    corrected = overlay.correct_business_candidates(
        payload,
        stage="candidate_b_final_validation",
    )

    assert corrected["inquiry_records"][0]["institution"] is None
    assert corrected["employment_records"][0]["employer"] is None
    assert corrected["employment_records"][0]["data_provider"] is None
    assert corrected["residence_records"][0]["data_provider"] is None
    assert corrected["personal_profile"][0]["query_institution"] is None
    assert corrected["credit_lines"][0]["institution"] == "中国银行股份有限公司"
    assert corrected["inquiry_records"][0]["canonical_raw"]["institution"] == (
        "福 中国银行股份有限公司"
    )
    anomalies = overlay.audit()["cell_anomalies"]
    assert len(anomalies) == 5
    assert all(
        anomaly["reason_codes"]
        == (
            "separated_leading_han_boundary",
            "independent_source_corroboration_missing",
            "normalized_value_withheld",
        )
        for anomaly in anomalies
    )


def test_institution_legal_name_preserves_sanctioned_internal_dashes() -> None:
    clean = "梅赛德斯-奔驰汽车金融有限公司"
    split = "梅赛德斯 - 奔驰汽车金融 有限公司"

    assert normalize_institution_name(clean) == clean
    assert normalize_institution_name(split) == clean
    assert _account_institution(clean) == clean
    assert _account_institution(split) == clean
    assert _account_institution("中国银行股份有限公司") == "中国银行股份有限公司"

    multiple = f"{clean} 乙银行股份有限公司"
    assert not institution_slot_is_unambiguous(multiple)
    assert normalize_institution_name(multiple) == multiple
    assert _account_institution(multiple) is None

    debris = "账户标识 梅赛德斯-奔驰汽车金融有限公司"
    assert not institution_slot_is_unambiguous(debris)
    assert normalize_institution_name(debris) == debris
    assert _account_institution(debris) is None


def test_institution_legal_root_is_ranked_after_attaching_its_branch_tail() -> None:
    expected = "中国建设银行股份有限公司福建自贸试验区福州片区分行"
    for raw in (
        expected,
        "中国建设银行股份有限 公司福建自贸试验区福 州片区分行",
        "中国建设银行 股份有限公司 福建自贸试验 区福州片区分 行",
    ):
        assert normalize_institution_name(raw) == expected
        assert _account_institution(raw) == expected

    branch_only = "福建自贸试验区福州片区分行"
    assert normalize_institution_name(branch_only) == branch_only


def test_account_institution_requires_independent_source_bound_corroboration() -> None:
    context = SimpleNamespace()
    page = SimpleNamespace(page_number=1, source_page_number=1)
    account = {"account_id": "account:1", "canonical_raw": {}}
    rows = [["管理机构", "账户标识"], ["中 国银行股份有限公司", "ABC12345678"]]
    for table_id in ("native:institution", "corrected:institution"):
        table = SimpleNamespace(table_id=table_id, metadata={})
        _apply_account_facts(context, account, rows, page=page, table=table)

    assert account["management_institution"] == "中国银行股份有限公司"
    assert "_pending_institution_observations" not in account or not account[
        "_pending_institution_observations"
    ]
    assert not any(
        issue["issue_code"] == "candidate_b_institution_leading_boundary_ambiguous"
        for issue in collect_extraction_issues(context)
    )


def test_account_institution_reports_uncorroborated_cross_cell_debris() -> None:
    context = SimpleNamespace()
    page = SimpleNamespace(page_number=1, source_page_number=1)
    table = SimpleNamespace(table_id="native:institution", metadata={})
    account = {"account_id": "account:1", "canonical_raw": {}}
    rows = [["管理机构", "账户标识"], ["福 中信银行股份有限公司个人信贷部", "ABC12345678"]]

    _apply_account_facts(context, account, rows, page=page, table=table)
    _flush_pending_account_institution_observations(context, account, boundary="unit_test")

    assert "management_institution" not in account
    issues = collect_extraction_issues(context)
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_institution_leading_boundary_ambiguous"
    assert issues[0]["field_name"] == "management_institution"


def test_credit_agreement_reconciler_accepts_only_corroborated_leading_boundary() -> None:
    context = SimpleNamespace()
    identifier = "ABC123456789"
    records = []
    for table_id in ("native:agreement", "corrected:agreement"):
        ref = {"table_id": table_id, "source_page": 1, "field_name": "institution"}
        records.append(
            {
                "credit_line_id": "credit-line:1",
                "account_identifier": identifier,
                "institution": None,
                "source_refs": [ref],
                "source_refs_by_field": {"institution": [ref]},
                "_pending_institution_observation": {
                    "raw": "中 信银行股份有限公司",
                    "value": "中信银行股份有限公司",
                    "source_refs": [ref],
                },
            }
        )

    reconciled = reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 1
    assert reconciled[0]["institution"] == "中信银行股份有限公司"
    assert "_pending_institution_observation" not in reconciled[0]
    assert not any(
        issue["issue_code"] == "candidate_b_institution_leading_boundary_ambiguous"
        for issue in collect_extraction_issues(context)
    )


def test_institution_slot_rejects_multiple_legal_names_and_preserves_one_branch() -> None:
    multiple = "甲银行股份有限公司 乙银行股份有限公司"
    clean = "中国银行股份有限公司厦门分行"

    assert not institution_slot_is_unambiguous(multiple)
    assert normalize_institution_name(multiple) == multiple
    assert _account_institution(multiple) is None
    assert institution_slot_is_unambiguous(clean)
    assert normalize_institution_name(clean) == clean
    assert _account_institution(clean) == clean


def test_typed_date_correction_rejects_multiple_valid_spans() -> None:
    overlay = _overlay()

    clean, clean_decision = overlay.correct_text("2024.02.29", role="date")
    ambiguous, ambiguous_decision = overlay.correct_text(
        "2024.02.29 2025.03.01",
        role="date",
    )

    assert clean == "2024-02-29"
    assert clean_decision is not None
    assert ambiguous == "2024.02.29 2025.03.01"
    assert ambiguous_decision is None


def test_native_scalar_dates_accept_exact_19xx_and_reject_ambiguity_or_invalid_days() -> None:
    assert _date("1999.01.02") == "1999-01-02"
    assert _liability_date("1999年1月2日") == "1999-01-02"
    assert _date("1999.01.02 2000.03.04") is None
    assert _liability_date("1999.02.29") is None
    assert _date("1999.01.02 信息更新日期") is None
    assert _date("1999.01.02 A") is None

    context = SimpleNamespace()
    page = SimpleNamespace(page_number=1, source_page_number=1)
    table = SimpleNamespace(table_id="account-date", metadata={})
    account = {"account_id": "account:1999", "canonical_raw": {}}
    _apply_account_facts(
        context,
        account,
        [["开立日期"], ["1999.01.02"]],
        page=page,
        table=table,
    )
    assert account["open_date"] == "1999-01-02"
    assert collect_extraction_issues(context) == []

    ambiguous_context = SimpleNamespace()
    ambiguous_account = {"account_id": "account:ambiguous", "canonical_raw": {}}
    raw = "1999.01.02 2000.03.04"
    _apply_account_facts(
        ambiguous_context,
        ambiguous_account,
        [["开立日期"], [raw]],
        page=page,
        table=table,
    )
    assert "open_date" not in ambiguous_account
    issue = collect_extraction_issues(ambiguous_context)[0]
    assert issue["field_name"] == "open_date"
    assert issue["observed_value"] == [raw]


def test_typed_date_correction_requires_the_complete_field_value() -> None:
    overlay = _overlay()

    corrected, decision = overlay.correct_text("2022,09.15 A", role="date")
    labeled, labeled_decision = overlay.correct_text(
        "2022.09.15 信息更新日期",
        role="date",
    )
    ambiguous, ambiguous_decision = overlay.correct_text(
        "2022.09.15 2023.01.02",
        role="date",
    )

    assert corrected == "2022,09.15 A"
    assert decision is None
    assert labeled == "2022.09.15 信息更新日期"
    assert labeled_decision is None
    assert ambiguous == "2022.09.15 2023.01.02"
    assert ambiguous_decision is None


def test_institution_debris_diagnostic_does_not_authorize_business_value_deletion() -> None:
    assert normalize_institution_name("S 福建海峡粮油购销有限公司") == "福建海峡粮油购销有限公司"
    assert normalize_institution_name("限 福建省国资粮食发展有限公司") == "福建省国资粮食发展有限公司"
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    corrected = overlay.correct_business_candidates(
        {
            "public_records": [
                {
                    "public_record_id": "housing:1",
                    "employer": "限 福建省国资粮食发展有限公司",
                    "source_refs": [
                        {
                            "logical_page": 13,
                            "bbox": [10, 20, 100, 40],
                            "geometry_scope": "cell",
                            "field_name": "employer",
                        }
                    ],
                }
            ]
        },
        stage="candidate_b_final_validation",
    )
    record = corrected["public_records"][0]
    assert record["employer"] is None
    assert record["canonical_raw"]["employer"] == "限 福建省国资粮食发展有限公司"
    assert any(
        anomaly["field_name"] == "employer"
        and anomaly["normalized_value_withheld"] is True
        for anomaly in overlay.audit()["cell_anomalies"]
    )


def test_institution_correction_preserves_legitimate_leading_legal_name_glyphs() -> None:
    assert normalize_institution_name("苏银凯基消费金融 有限公司") == (
        "苏银凯基消费金融有限公司"
    )
    assert normalize_institution_name("中信消费金融有限公司") == "中信消费金融有限公司"
    assert normalize_institution_name("福州奇富网络小额贷款有限公司") == (
        "福州奇富网络小额贷款有限公司"
    )
    assert normalize_institution_name("德州银行股份有限公司") == "德州银行股份有限公司"


def test_summary_cells_use_role_scoped_correction_and_audit_unresolved_values() -> None:
    payload = {
        "personal_detail_summary_cells": [
            {
                "record_id": "cell:type",
                "column_label": "账户类型",
                "row_index": 1,
                "column_index": 1,
                "value": "教 循环贷账户一",
                "source_refs": [
                    {
                        "logical_page": 3,
                        "bbox": [10, 20, 30, 40],
                        "geometry_scope": "cell",
                    }
                ],
            },
            {
                "record_id": "cell:count",
                "column_label": "月份数",
                "row_index": 1,
                "column_index": 2,
                "value": "二",
                "source_refs": [{"logical_page": 3, "bbox": [30, 20, 40, 40]}],
            },
            {
                "record_id": "cell:amount",
                "column_label": "单月最高逾期/透支总额",
                "row_index": 1,
                "column_index": 3,
                "value": "*-",
                "source_refs": [
                    {
                        "logical_page": 3,
                        "bbox": [40, 20, 50, 40],
                        "geometry_scope": "cell",
                    }
                ],
            },
        ]
    }
    overlay = _overlay()

    corrected = overlay.correct_business_candidates(payload, stage="native_business")
    audit = overlay.audit()

    cells = corrected["personal_detail_summary_cells"]
    assert cells[0]["value"] is None
    assert cells[0]["canonical_raw"]["value"] == "教 循环贷账户一"
    assert cells[1]["value"] == "2"
    assert cells[2]["value"] is None
    assert audit["audited_cell_count"] >= 3
    assert audit["abnormal_cell_count"] == 2
    assert {item["role"] for item in audit["cell_anomalies"]} == {
        "account_type_label",
        "amount_or_placeholder",
    }
    assert all(item["normalized_value_withheld"] is True for item in audit["cell_anomalies"])


def test_summary_business_category_pollution_is_withheld_and_reported() -> None:
    overlay = _overlay()
    corrected = overlay.correct_business_candidates(
        {
            "personal_detail_summary_cells": [
                {"record_id": "summary:1", "column_label": "业务类型", "value": "其馆类贷款"},
                {"record_id": "summary:2", "column_label": "业务类型", "value": "贯记卡"},
                {"record_id": "summary:3", "column_label": "业务类型", "value": "2 贷记卡 n"},
            ],
            "other_rows": [{"record_id": "other:1", "value": "贯记卡"}],
        },
        stage="candidate_b_final_validation",
    )

    cells = corrected["personal_detail_summary_cells"]
    assert [row["value"] for row in cells] == [None, None, None]
    assert [row["canonical_raw"]["value"] for row in cells] == [
        "其馆类贷款",
        "贯记卡",
        "2 贷记卡 n",
    ]
    assert corrected["other_rows"][0]["value"] == "贯记卡"
    anomalies = overlay.audit()["cell_anomalies"]
    assert len(anomalies) == 3
    assert all(item["field_name"] == "value" for item in anomalies)
    assert all(item["normalized_value_withheld"] is True for item in anomalies)


def test_exact_summary_business_category_and_account_type_stay_silent() -> None:
    overlay = _overlay()
    corrected = overlay.correct_business_candidates(
        {
            "personal_detail_summary_cells": [
                {"record_id": "summary:1", "column_label": "业务类型", "value": "其他类贷款"},
                {"record_id": "summary:2", "column_label": "业务类型", "value": "贷记卡"},
                {"record_id": "summary:3", "column_label": "账户类型", "value": "非循环贷账户"},
            ]
        },
        stage="candidate_b_final_validation",
    )

    assert [row["value"] for row in corrected["personal_detail_summary_cells"]] == [
        "其他类贷款",
        "贷记卡",
        "非循环贷账户",
    ]
    assert overlay.audit()["abnormal_cell_count"] == 0
    assert overlay.audit()["decision_count"] == 0


def test_generic_enum_near_match_is_not_silently_coerced() -> None:
    overlay = _overlay()
    corrected = overlay.correct_business_candidates(
        {
            "personal_detail_summary_cells": [
                {
                    "record_id": "summary:account-type",
                    "column_label": "账户类型",
                    "value": "非循环贷账",
                    "source_refs": [
                        {
                            "logical_page": 2,
                            "geometry_scope": "cell",
                            "field_name": "value",
                        }
                    ],
                }
            ]
        },
        stage="candidate_b_final_validation",
    )

    cell = corrected["personal_detail_summary_cells"][0]
    assert cell["value"] is None
    assert cell["canonical_raw"]["value"] == "非循环贷账"
    anomaly = overlay.audit()["cell_anomalies"][0]
    assert anomaly["role"] == "account_type_label"
    assert anomaly["value"] == "非循环贷账"
    assert anomaly["normalized_value_withheld"] is True


def test_polluted_summary_value_reaches_public_issue_observation() -> None:
    payload = {
        "personal_detail_summary_records": [
            {
                "summary_record_id": "summary:1",
                "summary_type": "信用业务概要",
                "title": "信贷交易信息提示",
            }
        ],
        "personal_detail_summary_cells": [
            {
                "record_id": "summary:polluted",
                "summary_cell_id": "summary:polluted",
                "summary_record_id": "summary:1",
                "summary_type": "信用业务概要",
                "title": "信贷交易信息提示",
                "row_index": 1,
                "column_index": 1,
                "column_label": "业务类型",
                "value": "2 贷记卡 n",
                "source_refs": [
                    {
                        "logical_page": 1,
                        "geometry_scope": "cell",
                        "field_name": "value",
                    }
                ],
            }
        ],
    }
    overlay = _overlay()
    corrected = overlay.correct_business_candidates(
        payload,
        stage="candidate_b_final_validation",
    )
    issues = collect_extraction_issues(SimpleNamespace(ocr_correction_audit=overlay.audit))
    corrected["personal_detail_extraction_issues"] = issues
    prepared = prepare_personal_detail_source_collections({"facts": {}, "datasets": corrected})

    assert corrected["personal_detail_summary_cells"][0]["value"] is None
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "pboc_cell_contract_unresolved"
    assert issues[0]["target_dataset"] == "personal_detail_summary_cells"
    assert issues[0]["field_name"] == "value"
    assert issues[0]["observed_value"] == "2 贷记卡 n"
    observations = prepared["datasets"]["personal_detail_field_observations"]
    assert any(
        row["dataset_name"] == "personal_detail_summary_cells"
        and row["field_name"] == "value"
        and row["raw_value"] == "2 贷记卡 n"
        for row in observations
    )


def test_exact_liability_identity_and_responsibility_types_stay_silent() -> None:
    identity_types = ("统一社会信用代码", "中征码")
    responsibility_types = ("保证", "担保", "抵押", "质押", "保证人", "担保人", "抵押人", "质押人")

    for value in identity_types:
        overlay = _overlay()
        corrected = overlay.correct_business_candidates(
            {
                "repayment_liability_records": [
                    {"liability_id": f"liability:id:{value}", "related_party_id_type": value}
                ]
            },
            stage="candidate_b_final_validation",
        )
        assert corrected["repayment_liability_records"][0]["related_party_id_type"] == value
        assert overlay.audit()["abnormal_cell_count"] == 0
        assert overlay.audit()["decision_count"] == 0

    for value in responsibility_types:
        overlay = _overlay()
        corrected = overlay.correct_business_candidates(
            {
                "repayment_liability_records": [
                    {"liability_id": f"liability:responsibility:{value}", "responsibility_type": value}
                ]
            },
            stage="candidate_b_final_validation",
        )
        assert corrected["repayment_liability_records"][0]["responsibility_type"] == value
        assert overlay.audit()["abnormal_cell_count"] == 0
        assert overlay.audit()["decision_count"] == 0


def test_credit_line_contract_does_not_invent_used_not_above_agreement_limit_rule() -> None:
    overlay = _overlay()

    corrected = overlay.correct_business_candidates(
        {
            "credit_lines": [
                {
                    "credit_line_id": "credit_line:1",
                    "total_limit": 100,
                    "used_limit": 108000,
                    "normalized": {
                        "total_limit": "100",
                        "used_limit": "108000",
                        "used_limit_status": "reported",
                    },
                    "source_refs": [{"logical_page": 20, "geometry_scope": "table"}],
                }
            ]
        },
        stage="native_business",
    )

    assert corrected["credit_lines"][0]["used_limit"] == 108000
    assert corrected["credit_lines"][0]["normalized"]["used_limit"] == "108000"
    assert corrected["credit_lines"][0]["normalized"]["used_limit_status"] == "reported"
    assert not any(
        item["field_name"] == "used_limit"
        and "used_limit_exceeds_total_limit" in item["reason_codes"]
        for item in overlay.audit()["cell_anomalies"]
    )


def test_unknown_monthly_status_is_repair_eligible_and_reported_if_unresolved() -> None:
    overlay = _overlay()

    corrected = overlay.correct_business_candidates(
        {
            "repayment_records": [
                {
                    "repayment_id": "repayment:1",
                    "year": 2024,
                    "month": 1,
                    "status": "unknown",
                    "source_refs_by_field": {
                        "status": [
                            {
                                "logical_page": 3,
                                "bbox": [10, 10, 20, 20],
                                "geometry_scope": "cell",
                                "binding": "canonical_field_slot",
                            }
                        ]
                    },
                }
            ]
        },
        stage="candidate_b_final_validation",
    )

    assert corrected["repayment_records"][0]["status"] is None
    anomaly = next(item for item in overlay.audit()["cell_anomalies"] if item["field_name"] == "status")
    assert anomaly["value"] == "unknown"
    assert anomaly["normalized_value_withheld"] is True


def test_missing_account_identifier_is_an_explicit_required_field_anomaly() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "credit_accounts": [
            {
                "account_id": "credit_account:loan:6",
                "account_identifier": None,
                "source_cell_refs": [
                    {
                        "logical_page": 4,
                        "bbox": [10, 20, 100, 40],
                        "geometry_scope": "cell",
                        "field_name": "account_identifier",
                    }
                ],
            }
        ]
    }

    overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")

    anomaly = overlay.audit()["cell_anomalies"][0]
    assert anomaly["dataset_name"] == "credit_accounts"
    assert anomaly["record_id"] == "credit_account:loan:6"
    assert anomaly["field_name"] == "account_identifier"
    assert anomaly["reason_codes"] == (
        "required_field_missing",
        "canonical_account_identifier_unresolved",
        "preserved_unknown_value",
    )
    assert anomaly["source_refs"][0]["geometry_scope"] == "cell"


def test_anonymous_correction_observation_does_not_invent_none_record_id() -> None:
    overlay = _overlay()

    overlay._walk(
        {"total_limit": "not-an-amount"},
        parent="credit_lines",
        refs=(),
        stage="candidate_b_final_validation",
    )

    anomaly = overlay.audit()["cell_anomalies"][0]
    assert anomaly["record_id"] == ""


def test_count_roles_remove_only_canonical_display_units() -> None:
    overlay = _overlay()

    months, months_decision = overlay.correct_text("12月", role="nonnegative_integer")
    contaminated, contaminated_decision = overlay.correct_text("12月还款", role="nonnegative_integer")

    assert months == "12"
    assert months_decision is not None
    assert contaminated == "12月还款"
    assert contaminated_decision is None


def test_account_open_date_accepts_pboc_month_precision() -> None:
    overlay = _overlay()
    payload = {
        "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_identifier": "ACCOUNT0001",
                    "open_date": "2018.10",
                "source_refs": [{"logical_page": 2, "bbox": [1, 2, 3, 4]}],
            }
        ]
    }

    corrected = overlay.correct_business_candidates(payload, stage="native_business")

    assert corrected["credit_accounts"][0]["open_date"] == "2018-10"
    assert overlay.audit()["abnormal_cell_count"] == 0


def test_r2_billing_adjustment_is_a_valid_personal_detail_repayment_status() -> None:
    overlay = _overlay()

    value, decision = overlay.correct_text("a", role="repayment_status")

    assert value == "A"
    assert decision is not None


def test_printed_hash_is_a_valid_personal_detail_repayment_status() -> None:
    overlay = _overlay()

    value, decision = overlay.correct_text("#", role="repayment_status")

    assert value == "#"
    assert decision is None


def test_summary_source_ref_prefers_exact_cell_geometry() -> None:
    page = SimpleNamespace(page_number=3, source_page_number=2)
    table = SimpleNamespace(
        table_id="summary:1",
        bbox=[0, 0, 100, 100],
        metadata={
            "cell_bboxes": [[[1, 2, 3, 4], [5, 6, 7, 8]]],
            "cell_evidence_ids": [[["ocr:1"], ["ocr:2"]]],
        },
    )

    ref = _source_ref(page, table, row=0, column=1)

    assert ref["bbox"] == [5, 6, 7, 8]
    assert ref["geometry_scope"] == "cell"
    assert ref["source"] == "native_detail_table_cell"
    assert ref["evidence_ids"] == ["ocr:2"]


def test_source_ref_reads_cell_provenance_nested_under_geometry() -> None:
    page = SimpleNamespace(page_number=23, source_page_number=12)
    table = SimpleNamespace(
        table_id="pt_23_1",
        bbox=[0, 0, 400, 200],
        metadata={
            "geometry": {
                "coordinate_system": "pdf_points",
                "cell_bboxes": [
                    [[0, 0, 40, 10]],
                    [[10, 20, 50, 40]],
                ],
                "cell_evidence_ids": [
                    [["native:header"]],
                    [["native:value"]],
                ],
            }
        },
    )

    ref = _source_ref(page, table, row=1, column=0)

    assert ref["source"] == "native_detail_table_cell"
    assert ref["bbox"] == [10, 20, 50, 40]
    assert ref["evidence_ids"] == ["native:value"]
    assert ref["coordinate_system"] == "pdf_points"


def _blank_account_currency_observation(
    context: SimpleNamespace,
    *,
    with_geometry: bool = True,
) -> dict[str, object]:
    metadata: dict[str, object] = {"raw_rows": [["币种"], [""]]}
    if with_geometry:
        metadata["geometry"] = {
            "coordinate_system": "pdf_points",
            "cell_bboxes": [
                [[0, 0, 40, 10]],
                [[10, 20, 50, 40]],
            ],
            "cell_evidence_ids": [
                [["native:currency-header"]],
                [["native:currency-blank"]],
            ],
        }
    table = SimpleNamespace(
        table_id="pt_23_1",
        bbox=[0, 0, 400, 200],
        metadata=metadata,
        headers=[],
        rows=[],
    )
    account: dict[str, object] = {
        "account_id": "credit_account:credit_card:20",
    }
    _apply_account_facts(
        context,
        account,
        metadata["raw_rows"],
        page=SimpleNamespace(page_number=23, source_page_number=12),
        table=table,
    )
    return account


def test_blank_account_currency_reports_unreadable_value_cell_not_invalid_value() -> None:
    context = SimpleNamespace()

    account = _blank_account_currency_observation(context)

    assert "currency" not in account
    issues = collect_extraction_issues(context)
    assert len(issues) == 1
    issue = issues[0]
    assert issue["issue_code"] == "candidate_b_exact_slot_value_unreadable"
    assert issue["observed_value"] == {
        "raw": "",
        "slot_state": "blank_or_unreadable",
    }
    assert "blank or unreadable" in issue["message"]
    assert "field_contract_failed" not in issue["reason_codes"]
    ref = issue["source_refs"][0]
    assert ref["row"] == 1
    assert ref["canonical_label_row"] == 0
    assert ref["canonical_value_row"] == 1
    assert ref["field_slot_role"] == "value"
    assert ref["bbox"] == [10, 20, 50, 40]
    assert ref["evidence_ids"] == ["native:currency-blank"]


def test_final_overlay_recovers_one_exact_blank_account_currency_and_closes_issue() -> None:
    context = SimpleNamespace()
    account = _blank_account_currency_observation(context)
    native_currency_ref = account["source_refs_by_field"]["currency"][0]
    record_issue(
        context,
        make_issue(
            category="schema_incompleteness",
            issue_code="candidate_b_account_required_field_unresolved",
            message="A required canonical account field remains withheld.",
            parser_stage="candidate_b_account_canonical_slots",
            target_dataset="credit_accounts",
            target_record_id="credit_account:credit_card:20",
            field_name="currency",
            source_refs=(native_currency_ref,),
            reason_codes=(
                "canonical_account_template",
                "required_field_missing",
                "normalized_value_withheld",
            ),
        ),
    )
    wrapped_account = {
        "record_id": "credit_account:credit_card:20",
        "normalized": {
            "account_id": "credit_account:credit_card:20",
            "currency": None,
            "account_currency": None,
            "reporting_amount_currency": None,
        },
        # Model the live compatibility envelope: stale flat aliases must not
        # override the repaired normalized record downstream.
        "currency": None,
        "account_currency": "STALE",
        "reporting_amount_currency": None,
        "canonical_raw": account["canonical_raw"],
        "source_refs_by_field": account["source_refs_by_field"],
        "_unresolved_fields": account["_unresolved_fields"],
        "_invalid_observation_fields": account["_invalid_observation_fields"],
        "_reported_invalid_fields": account["_reported_invalid_fields"],
    }
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    raw_currency = "\u4eba\u6c11\u5e01\u5143"
    overlay.install_business_repair_evidence(
        [
            {
                "page": 23,
                "source_page": 12,
                "lines": [
                    {
                        **_one_shot_page_line(
                            raw_currency,
                            confidence=0.96,
                            bbox=[12, 22, 48, 38],
                            page_key="currency-card-20",
                        ),
                    }
                ],
            }
        ],
        affected_pages={23},
    )

    corrected = overlay.correct_business_candidates(
        {"credit_accounts": [wrapped_account]},
        stage="candidate_b_final_validation",
    )
    corrected_account = corrected["credit_accounts"][0]
    normalized = corrected_account["normalized"]

    assert normalized["currency"] == "CNY"
    assert normalized["account_currency"] == "CNY"
    assert normalized["reporting_amount_currency"] == "CNY"
    assert corrected_account["currency"] == "CNY"
    assert corrected_account["account_currency"] == "CNY"
    assert corrected_account["reporting_amount_currency"] == "CNY"
    assert corrected_account["canonical_raw"]["currency"] == ["", raw_currency]
    assert corrected_account["canonical_raw"]["account_currency"] == raw_currency
    assert (
        corrected_account["canonical_raw"]["reporting_amount_currency"]
        == raw_currency
    )
    corrected_refs = [
        ref
        for ref in corrected_account["source_refs_by_field"]["account_currency"]
        if ref.get("source") == "personal_detail_corrected_page_cell"
    ]
    assert len(corrected_refs) == 1
    assert corrected_refs[0]["bbox"] == [12.0, 22.0, 48.0, 38.0]
    assert corrected_refs[0]["evidence_ids"] == [
        "personal_detail_page_reocr:currency-card-20:w0"
    ]
    assert corrected_refs[0]["raw_text"] == raw_currency
    assert corrected_refs[0]["producer_source"] == "personal_detail_page_reocr_once"
    assert (
        corrected_refs[0]["acquisition_id"]
        == "personal_detail_page_reocr_once:currency-card-20"
    )
    assert "table_id" not in corrected_refs[0]
    assert "row" not in corrected_refs[0]
    decisions = overlay.audit()["decisions"]
    assert any(
        decision["role"] == "currency"
        and decision["corrected"] == "CNY"
        and decision["selected_raw"] == raw_currency
        and decision["selected_acquisition"]
        == "personal_detail_page_reocr_once:currency-card-20"
        and decision["method"]
        == "schema_bound_missing_account_currency_reparse"
        for decision in decisions
    )

    _reconcile_final_account_field_issues(context, [corrected_account])

    assert collect_extraction_issues(context) == []
    for marker in (
        "_unresolved_fields",
        "_invalid_observation_fields",
        "_reported_invalid_fields",
    ):
        assert marker not in corrected_account


def test_missing_currency_repair_resolves_target_ids_against_available_sealed_store() -> None:
    context = SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(
                text_atoms=[
                    {
                        "id": "native:unrelated",
                        "text": "unrelated",
                        "bbox": [100, 100, 120, 110],
                    }
                ]
            )
        )
    )
    account = _blank_account_currency_observation(context)
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    overlay.install_business_repair_evidence(
        [
            {
                "page": 23,
                "source_page": 12,
                "page_key": "currency-sealed-target",
                "lines": [
                    _one_shot_page_line(
                        "人民币元",
                        bbox=[12, 22, 48, 38],
                        page_key="currency-sealed-target",
                    )
                ],
            }
        ],
        affected_pages={23},
    )

    corrected = overlay.correct_business_candidates(
        {"credit_accounts": [account]},
        stage="candidate_b_final_validation",
    )["credit_accounts"][0]

    assert "currency" not in corrected
    assert corrected["canonical_raw"]["currency"] == [""]
    assert overlay.audit()["applied_count"] == 0


@pytest.mark.parametrize("guard", ("conflict", "nonblank_raw"))
def test_missing_currency_repair_never_overwrites_withheld_reporting_currency_history(
    guard: str,
) -> None:
    context = SimpleNamespace()
    account = _blank_account_currency_observation(context)
    account["canonical_raw"]["reporting_amount_currency"] = ["USD", "CNY"]
    if guard == "conflict":
        account["_reported_field_conflicts"] = ["reporting_amount_currency"]
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    overlay.install_business_repair_evidence(
        [
            {
                "page": 23,
                "source_page": 12,
                "page_key": "currency-reporting-conflict",
                "lines": [
                    _one_shot_page_line(
                        "人民币元",
                        bbox=[12, 22, 48, 38],
                        page_key="currency-reporting-conflict",
                    )
                ],
            }
        ],
        affected_pages={23},
    )

    corrected = overlay.correct_business_candidates(
        {"credit_accounts": [account]},
        stage="candidate_b_final_validation",
    )["credit_accounts"][0]

    assert "currency" not in corrected
    assert corrected["canonical_raw"]["currency"] == [""]
    assert corrected["canonical_raw"]["reporting_amount_currency"] == ["USD", "CNY"]
    assert overlay.audit()["applied_count"] == 0


@pytest.mark.parametrize(
    ("case", "lines"),
    (
        (
            "missing_bbox",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [12, 22, 48, 38],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "header_ref",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [12, 22, 48, 38],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "low_confidence",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.71,
                    "bbox": [12, 22, 48, 38],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "multiple_currencies",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.99,
                    "bbox": [12, 22, 30, 38],
                    "evidence_ids": ["repair:cny"],
                },
                {
                    "text": "USD",
                    "confidence": 0.80,
                    "bbox": [30, 22, 48, 38],
                    "evidence_ids": ["repair:usd"],
                },
            ],
        ),
        (
            "near_tie",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [12, 22, 30, 38],
                    "evidence_ids": ["repair:cny"],
                },
                {
                    "text": "USD",
                    "confidence": 0.92,
                    "bbox": [30, 22, 48, 38],
                    "evidence_ids": ["repair:usd"],
                },
            ],
        ),
        (
            "outside_value_cell",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [200, 200, 240, 220],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "row_spanning_bbox",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [0, 20, 400, 40],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "mostly_outside_bbox",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [-200, 20, 20, 40],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "neighboring_halo_bbox",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [51, 22, 65, 38],
                    "evidence_ids": ["repair:cny"],
                }
            ],
        ),
        (
            "missing_repair_evidence_id",
            [
                {
                    "text": "\u4eba\u6c11\u5e01\u5143",
                    "confidence": 0.96,
                    "bbox": [12, 22, 48, 38],
                }
            ],
        ),
    ),
)
def test_missing_account_currency_repair_remains_fail_closed(
    case: str,
    lines: list[dict[str, object]],
) -> None:
    for word_index, line in enumerate(lines):
        line["source"] = "personal_detail_page_reocr_once"
        if line.get("evidence_ids"):
            line["evidence_ids"] = [
                f"personal_detail_page_reocr:currency-fail-{case}:w{word_index}"
            ]
    context = SimpleNamespace()
    account = _blank_account_currency_observation(
        context,
        with_geometry=case != "missing_bbox",
    )
    if case == "header_ref":
        ref = account["source_refs_by_field"]["currency"][0]
        ref["row"] = 0
        ref["canonical_value_row"] = 0
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    overlay.install_business_repair_evidence(
        [{"page": 23, "source_page": 12, "lines": lines}],
        affected_pages={23},
    )

    corrected = overlay.correct_business_candidates(
        {"credit_accounts": [account]},
        stage="candidate_b_final_validation",
    )
    corrected_account = corrected["credit_accounts"][0]
    _reconcile_final_account_field_issues(context, [corrected_account])

    assert corrected_account.get("currency") is None
    assert corrected_account.get("account_currency") is None
    assert corrected_account.get("reporting_amount_currency") is None
    issues = collect_extraction_issues(context)
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_exact_slot_value_unreadable"
    assert issues[0]["status"] == "requires_review"


def test_missing_account_currency_repair_never_overwrites_existing_currency() -> None:
    context = SimpleNamespace()
    account = _blank_account_currency_observation(context)
    account.update(
        {
            "currency": "USD",
            "account_currency": "USD",
            "reporting_amount_currency": "USD",
        }
    )
    overlay = PersonalDetailOCRCorrectionOverlay(context)
    overlay.install_business_repair_evidence(
        [
            {
                "page": 23,
                "source_page": 12,
                "lines": [
                    {
                        "text": "\u4eba\u6c11\u5e01\u5143",
                        "confidence": 0.99,
                        "bbox": [12, 22, 48, 38],
                        "evidence_ids": ["repair:cny"],
                    }
                ],
            }
        ],
        affected_pages={23},
    )

    corrected = overlay.correct_business_candidates(
        {"credit_accounts": [account]},
        stage="candidate_b_final_validation",
    )

    assert corrected["credit_accounts"][0]["currency"] == "USD"
    assert corrected["credit_accounts"][0]["account_currency"] == "USD"
    assert corrected["credit_accounts"][0]["reporting_amount_currency"] == "USD"
    assert not any(
        decision["method"] == "schema_bound_missing_account_currency_reparse"
        for decision in overlay.audit()["decisions"]
    )


def test_evidence_overlay_corrects_inquiry_lines_without_mutating_raw_evidence() -> None:
    raw_pages = [
        {
            "page": 27,
            "source_page": 14,
            "lines": [
                {
                    "text": "2 2024,02.16 中邮消费金融有限公司 费后管理",
                    "confidence": 0.91,
                    "bbox": [10, 20, 300, 35],
                    "evidence_ids": ["ocr:2"],
                }
            ],
        }
    ]
    overlay = _overlay()

    corrected = overlay.corrected_evidence_pages(raw_pages)

    assert raw_pages[0]["lines"][0]["text"] == "2 2024,02.16 中邮消费金融有限公司 费后管理"
    line = corrected[0]["lines"][0]
    assert line["text"] == raw_pages[0]["lines"][0]["text"]
    assert "ocr_original_text" not in line
    assert "ocr_correction" not in line


def test_inquiry_row_correction_handles_typed_date_and_reason_confusions() -> None:
    overlay = _overlay()

    corrected, decision = overlay.correct_text(
        "12 2025:03.20 福鼎市农村信用合作联社 信用卡审批",
        role="inquiry_row",
    )
    reason_corrected, reason_decision = overlay.correct_text(
        "19 2020.12.12 深圳市中融小额贷款有限公司 资款审批",
        role="inquiry_row",
    )
    compact_date, compact_date_decision = overlay.correct_text(
        "4 2022.1214 平安普惠融资担保有限公司 担保资格审查",
        role="inquiry_row",
    )
    financing, financing_decision = overlay.correct_text(
        "101 2023.01.06 平安国际融资租赁（天津）有限公司 磁资审批",
        role="inquiry_row",
    )

    assert corrected == "12 2025.03.20 福鼎市农村信用合作联社 信用卡审批"
    assert reason_corrected == "19 2020.12.12 深圳市中融小额贷款有限公司 资款审批"
    assert compact_date == "4 2022.12.14 平安普惠融资担保有限公司 担保资格审查"
    assert financing == "101 2023.01.06 平安国际融资租赁（天津）有限公司 磁资审批"
    assert decision is not None
    assert reason_decision is None
    assert compact_date_decision is not None
    assert financing_decision is None


def test_inquiry_reason_ocr_alias_is_withheld_without_independent_authority() -> None:
    overlay = _overlay()

    reason, reason_decision = overlay.correct_text("资款审批", role="inquiry_reason")
    institution_line, line_decision = overlay.correct_text(
        "19 2020.12.12 某货后管理服务有限公司 贷款审批",
        role="inquiry_row",
    )

    assert reason == "资款审批"
    assert reason_decision is None
    assert institution_line == "19 2020.12.12 某货后管理服务有限公司 贷款审批"
    assert line_decision is None


def test_corrected_inquiry_extraction_recovers_rows_and_keeps_section_sequences() -> None:
    pages = [
        {
            "page": 27,
            "source_page": 14,
            "lines": [
                {"text": "1 2024.03.08 泉州银行股份有限公司 贷后管理", "confidence": 0.99},
                {"text": "2024,02.16 中邮消费金融有限公司 费后管理", "confidence": 0.95},
                {"text": "3 2024.02.13 重庆蚂蚊消费金融有限公司 Ss 贷后管理", "confidence": 0.90},
                {"text": "4 2024.01.13 中国平安财产保险股份有限公司 保后管理", "confidence": 0.92},
                {"text": "1 2024.01.15 本人 本人查询(自助查询机)", "confidence": 0.98},
            ],
        }
    ]
    overlay = _overlay()
    parse_result = SimpleNamespace(
        corrected_evidence_pages=lambda: overlay.corrected_evidence_pages(pages),
        pages=[],
    )

    records = _extract_inquiries(parse_result)

    assert len(records) == 4
    assert [record["sequence"] for record in records if record["inquiry_type"] == "institution"] == [1, 3, 4]
    assert [record["sequence"] for record in records if record["inquiry_type"] == "personal"] == [1]
    corrected = overlay.correct_business_candidates({"inquiry_records": records}, stage="test")
    assert corrected["inquiry_records"][1]["institution"] == "重庆蚂蚊消费金融有限公司 Ss"


def test_inquiry_extraction_reconstructs_full_page_ocr_cells_by_date_row() -> None:
    pages = [
        {
            "page": 35,
            "source_page": 18,
            "lines": [
                {"text": "1", "confidence": 0.99, "bbox": [10, 100, 20, 120]},
                {"text": "2024.08.12", "confidence": 0.99, "bbox": [80, 100, 150, 120]},
                {"text": "样例银行股份有限公司", "confidence": 0.98, "bbox": [210, 100, 350, 120]},
                {"text": "担保资格审查", "confidence": 0.97, "bbox": [410, 100, 500, 120]},
                {"text": "2", "confidence": 0.99, "bbox": [10, 140, 20, 160]},
                {"text": "2024.08.02", "confidence": 0.99, "bbox": [80, 140, 150, 160]},
                {"text": "样例保险股份有限公司", "confidence": 0.98, "bbox": [210, 140, 350, 160]},
                {"text": "保后管理", "confidence": 0.97, "bbox": [410, 140, 480, 160]},
            ],
        }
    ]
    parse_result = SimpleNamespace(corrected_evidence_pages=lambda: pages, pages=[])

    records = _extract_inquiries(parse_result)

    assert [(record["sequence"], record["inquiry_date"], record["reason"]) for record in records] == [
        (1, "2024-08-12", "担保资格审查"),
        (2, "2024-08-02", "保后管理"),
    ]
    assert records[0]["institution"] == "样例银行股份有限公司"


def test_inquiry_extraction_excludes_report_header_query_request() -> None:
    pages = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {
                    "text": "2024.08.17 被查询者姓名 曾耀 查询机构 征信中心 查询原因 本人查询",
                    "confidence": 0.99,
                }
            ],
        },
        {
            "page": 35,
            "source_page": 18,
            "lines": [
                {"text": "四 查询记录 机构查询记录明细", "confidence": 0.99},
                {"text": "1 2024.08.12 样例银行股份有限公司 担保资格审查", "confidence": 0.99},
            ],
        },
    ]
    parse_result = SimpleNamespace(corrected_evidence_pages=lambda: pages, pages=[])

    records = _extract_inquiries(parse_result)

    assert len(records) == 1
    assert records[0]["inquiry_type"] == "institution"
    assert records[0]["inquiry_date"] == "2024-08-12"


def test_inquiry_sequences_preserve_gaps_but_repair_duplicate_ocr_ordinals() -> None:
    pages = [
        {
            "page": 27,
            "source_page": 14,
            "lines": [
                {"text": "1 2024.03.08 泉州银行股份有限公司 贷后管理", "confidence": 0.99},
                {"text": "3 2024.02.16 中邮消费金融有限公司 贷后管理", "confidence": 0.95},
                {"text": "3 2024.02.13 重庆蚂蚁消费金融有限公司 贷款审批", "confidence": 0.90},
            ],
        }
    ]
    overlay = _overlay()
    parse_result = SimpleNamespace(
        corrected_evidence_pages=lambda: overlay.corrected_evidence_pages(pages),
        pages=[],
    )

    records = _extract_inquiries(parse_result)

    assert [record["sequence"] for record in records] == [1, 3, 4]
    assert [record["sequence_source"] for record in records] == [
        "ocr_row_number",
        "ocr_row_number",
        "inferred_from_section_order",
    ]


def test_business_overlay_does_not_promote_unowned_identifier_candidates() -> None:
    payload = {
        "credit_accounts": [
            {
                "account_id": "credit_account:loan:1",
                "management_institution": "重庆蚂蚊消费金融有限公司 Ss",
                "account_identifier_candidates": ["ABCD 1234 5678"],
                "raw_detail_text": "重庆蚂蚊消费金融有限公司 Ss ABCD 1234 5678",
                "source_refs": [{"logical_page": 3, "bbox": [1, 2, 3, 4]}],
                "confidence": 0.9,
            }
        ]
    }
    overlay = _overlay()

    corrected = overlay.correct_business_candidates(payload, stage="test")

    assert "account_identifier" not in payload["credit_accounts"][0]
    account = corrected["credit_accounts"][0]
    assert account["management_institution"] == "重庆蚂蚊消费金融有限公司 Ss"
    assert "account_identifier" not in account
    assert account["account_identifier_candidates"] == ["ABCD 1234 5678"]
    assert account["raw_detail_text"] == payload["credit_accounts"][0]["raw_detail_text"]
    assert overlay.audit()["applied_count"] == 0


class _FakeResolver:
    def __call__(self, logical_page: int):
        assert logical_page == 1
        return {
            "image": SimpleNamespace(shape=(100, 100, 3)),
            "page_width": 100,
            "page_height": 100,
        }


class _FakeRepairEngine:
    def repair_from_image(self, request, image, **_kwargs):
        assert request.kind == "identity_document_number"
        assert image.shape == (100, 100, 3)
        return [
            RepairCandidate(
                candidate_id="candidate:1",
                request_id=request.request_id,
                text="11010519491231002X",
                confidence=0.95,
                source="test_ocr",
            )
        ]


class _FakeInquiryRepairEngine:
    def repair_from_image(self, request, image, **_kwargs):
        assert request.kind == "inquiry_row"
        assert image.shape == (100, 100, 3)
        return [
            RepairCandidate(
                candidate_id="candidate:inquiry:1",
                request_id=request.request_id,
                text="39 2024.11.21 深圳市华融融资担保有限公司 担保资格审查",
                confidence=0.94,
                source="test_ocr",
            )
        ]


class _FakeCellRepairEngine:
    def repair_from_image(self, request, image, **_kwargs):
        assert request.kind == "integer_or_placeholder"
        assert request.bbox == (30.0, 20.0, 40.0, 40.0)
        return [
            RepairCandidate(
                candidate_id="candidate:cell:1",
                request_id=request.request_id,
                text="2",
                confidence=0.93,
                source="test_cell_ocr",
            )
        ]


def _exact_identity_repair_target_ref() -> dict[str, object]:
    return {
        "source": "native_detail_table_cell",
        "logical_page": 1,
        "source_page": 9,
        "bbox": [1, 1, 20, 10],
        "evidence_ids": ["native:identity:1"],
        "geometry_scope": "cell",
        "geometry_status": "exact",
        "binding": "canonical_field_slot",
    }


def test_schema_role_repair_does_not_fall_back_to_crop_ocr() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None
    assert overlay.audit()["repair_evidence_reparse_attempt_count"] == 0
    assert overlay.audit()["ocr_started_by_correction_overlay"] is False


def test_schema_assigned_field_uses_only_coordinator_installed_page_evidence() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    _one_shot_page_line(
                        "11010519491231002X",
                        confidence=0.96,
                        bbox=[1, 1, 20, 10],
                        page_key="identity-page-1",
                    )
                ],
            }
        ],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "11010519491231002X"
    assert decision is not None
    assert decision.method == "schema_bound_page_evidence_reparse"
    assert decision.reason_codes[-2:] == (
        "exact_repair_candidate_source_ref",
        "unique_source_bound_candidate",
    )
    assert len(decision.source_refs) == 2
    assert decision.source_refs[0] == _exact_identity_repair_target_ref()
    candidate_ref = decision.source_refs[1]
    assert candidate_ref == {
        "source": "personal_detail_installed_page_evidence",
        "logical_page": 1,
        "source_page": 9,
        "bbox": [1.0, 1.0, 20.0, 10.0],
        "evidence_ids": ["personal_detail_page_reocr:identity-page-1:w0"],
        "geometry_scope": "token_band",
        "geometry_status": "exact",
        "binding": "canonical_field_slot_repair_candidate",
        "binding_quality": "exact_source_bound_candidate",
        "raw_text": "11010519491231002X",
        "producer_source": "personal_detail_page_reocr_once",
        "acquisition_id": "personal_detail_page_reocr_once:identity-page-1",
    }
    assert overlay.audit()["decisions"][0]["source_refs"][1] == candidate_ref
    assert overlay.audit()["decisions"][0]["selected_raw"] == "11010519491231002X"
    assert (
        overlay.audit()["decisions"][0]["selected_acquisition"]
        == "personal_detail_page_reocr_once:identity-page-1"
    )
    assert overlay.audit()["repair_evidence_reparse_attempt_count"] == 1
    assert overlay.audit()["ocr_started_by_correction_overlay"] is False


def test_page_repair_rejects_destructive_normalization_of_candidate_text() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    _one_shot_page_line(
                        "导 示例银行股份有限公司",
                        bbox=[1, 1, 20, 10],
                        page_key="institution-page-1",
                    )
                ],
            }
        ],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "坏值",
        role="institution_name",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "坏值"
    assert decision is None
    assert overlay.audit()["decisions"] == []


def test_page_repair_resolves_target_ids_when_sealed_store_is_available() -> None:
    target_ref = _exact_identity_repair_target_ref()
    owner = SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(
                text_atoms=[
                    {
                        "id": "native:identity:1",
                        "text": "1101051949123100?X",
                        "bbox": [1, 1, 20, 10],
                    }
                ]
            )
        )
    )
    overlay = PersonalDetailOCRCorrectionOverlay(owner)
    repair_page = {
        "page": 1,
        "source_page": 9,
        "lines": [
            _one_shot_page_line(
                "11010519491231002X",
                bbox=[1, 1, 20, 10],
                page_key="identity-sealed-page-1",
            )
        ],
    }
    overlay.install_business_repair_evidence(
        [repair_page],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(target_ref,),
    )
    assert corrected == "11010519491231002X"
    assert decision is not None

    owner.evidence_plane.evidence.text_atoms.clear()
    second_overlay = PersonalDetailOCRCorrectionOverlay(owner)
    second_overlay.install_business_repair_evidence(
        [repair_page],
        affected_pages={1},
    )
    rejected, rejected_decision = second_overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(target_ref,),
    )
    assert rejected == "1101051949123100?X"
    assert rejected_decision is None


def test_page_repair_rejects_target_ref_bound_to_another_field() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    _one_shot_page_line(
                        "11010519491231002X",
                        bbox=[1, 1, 20, 10],
                        page_key="identity-field-page-1",
                    )
                ],
            }
        ],
        affected_pages={1},
    )
    target_ref = {**_exact_identity_repair_target_ref(), "field_name": "credit_limit"}

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        field_name="document_number",
        source_refs=(target_ref,),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


def test_page_repair_rejects_conflicting_text_and_content() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    _one_shot_page_line(
                        "11010519491231002X",
                        content="110105194912310011",
                        bbox=[1, 1, 20, 10],
                        page_key="identity-conflict-page-1",
                    )
                ],
            }
        ],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


def test_page_repair_rejects_distinct_ids_from_same_one_shot_acquisition() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    _one_shot_page_line(
                        "11010519491231002X",
                        bbox=[1, 1, 20, 10],
                        page_key="same-acquisition",
                        word_index=1,
                    )
                ],
            }
        ],
        affected_pages={1},
    )
    target_ref = {
        **_exact_identity_repair_target_ref(),
        "evidence_ids": ["personal_detail_page_reocr:same-acquisition:w0"],
    }

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(target_ref,),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


@pytest.mark.parametrize(
    "case",
    (
        "caller_role_disagrees_with_field",
        "conflicting_role_tags",
    ),
)
def test_page_repair_requires_consistent_target_field_and_role_bindings(
    case: str,
) -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "page_key": "binding-page-1",
                "lines": [
                    _one_shot_page_line(
                        "11010519491231002X",
                        bbox=[1, 1, 20, 10],
                        page_key="binding-page-1",
                    )
                ],
            }
        ],
        affected_pages={1},
    )
    target_ref = _exact_identity_repair_target_ref()
    if case == "caller_role_disagrees_with_field":
        target_ref["field_name"] = "credit_limit"
        field_name = "credit_limit"
    else:
        target_ref.update(
            {
                "field_name": "document_number",
                "field_role": "identity_document_number",
                "semantic_role": "amount",
            }
        )
        field_name = "document_number"

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        field_name=field_name,
        source_refs=(target_ref,),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


@pytest.mark.parametrize(
    "case",
    (
        "page_key_mismatch",
        "logical_page_mismatch",
        "candidate_acquisition_mismatch",
        "mixed_target_ids",
        "target_acquisition_mismatch",
        "mixed_candidate_page_acquisitions",
        "malformed_hidden_replay",
    ),
)
def test_page_repair_rejects_contradictory_or_replayed_acquisition_planes(
    case: str,
) -> None:
    target_ref = _exact_identity_repair_target_ref()
    page_key = "candidate-page-1"
    page: dict[str, object] = {
        "page": 1,
        "source_page": 9,
        "page_key": page_key,
        "lines": [
            _one_shot_page_line(
                "11010519491231002X",
                bbox=[1, 1, 20, 10],
                page_key=page_key,
            )
        ],
    }
    if case == "page_key_mismatch":
        page["page_key"] = "different-page"
    elif case == "logical_page_mismatch":
        page["logical_page"] = 2
    elif case == "candidate_acquisition_mismatch":
        page["lines"][0]["acquisition_id"] = "personal_detail_page_reocr_once:different-page"
    elif case == "mixed_target_ids":
        target_ref.update(
            {
                "source": "personal_detail_page_reocr_once",
                "evidence_ids": [
                    "personal_detail_page_reocr:target-page:w0",
                    "native:identity:mixed",
                ],
            }
        )
    elif case == "target_acquisition_mismatch":
        target_ref.update(
            {
                "source": "personal_detail_page_reocr_once",
                "evidence_ids": ["personal_detail_page_reocr:target-page:w0"],
                "acquisition_id": "personal_detail_page_reocr_once:different-target-page",
            }
        )
    elif case == "mixed_candidate_page_acquisitions":
        page["lines"].append(
            _one_shot_page_line(
                "unrelated",
                bbox=[100, 100, 120, 110],
                page_key="different-page",
            )
        )
    elif case == "malformed_hidden_replay":
        replayed_id = "personal_detail_page_reocr:candidate-page-1:w0"
        page["lines"].append(
            {
                "text": "unrelated",
                "confidence": 0.9,
                "bbox": [100, 100, 120, 110],
                "evidence_ids": [replayed_id, replayed_id],
                "source": "personal_detail_page_reocr_once",
            }
        )
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence([page], affected_pages={1})

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(target_ref,),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


def test_sealed_repair_plane_target_without_acquisition_cannot_authorize_repair() -> None:
    owner = SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(
                text_atoms=[
                    {
                        "id": "sealed:repair-target",
                        "text": "1101051949123100?X",
                        "bbox": [1, 1, 20, 10],
                    }
                ]
            )
        )
    )
    overlay = PersonalDetailOCRCorrectionOverlay(owner)
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "page_key": "candidate-page-1",
                "lines": [
                    _one_shot_page_line(
                        "11010519491231002X",
                        bbox=[1, 1, 20, 10],
                        page_key="candidate-page-1",
                    )
                ],
            }
        ],
        affected_pages={1},
    )
    target_ref = {
        **_exact_identity_repair_target_ref(),
        "source": "personal_detail_installed_page_evidence",
        "evidence_ids": ["sealed:repair-target"],
    }

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(target_ref,),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


def test_final_walk_preserves_displaced_flat_scalar_for_applied_normalization() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())

    corrected = overlay.correct_business_candidates(
        {
            "credit_lines": [
                {
                    "record_id": "credit_line:date",
                    "effective_date": "2024.01.02",
                }
            ]
        },
        stage="candidate_b_final_validation",
    )

    record = corrected["credit_lines"][0]
    assert record["effective_date"] == "2024-01-02"
    assert record["canonical_raw"]["effective_date"] == "2024.01.02"


def test_final_walk_preserves_displaced_nested_scalar_when_prior_raw_exists() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())

    corrected = overlay.correct_business_candidates(
        {
            "credit_lines": [
                {
                    "record_id": "credit_line:nested-date",
                    "effective_date": {
                        "value": "2024.01.02",
                        "raw": "OLDER_OBSERVATION",
                    },
                }
            ]
        },
        stage="candidate_b_final_validation",
    )

    value = corrected["credit_lines"][0]["effective_date"]
    assert value["value"] == "2024-01-02"
    assert value["raw"] == "OLDER_OBSERVATION"
    assert value["canonical_raw"]["value"] == "2024.01.02"


def test_nested_correction_preserves_raw_when_existing_audit_container_is_malformed() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())

    corrected = overlay.correct_business_candidates(
        {
            "credit_lines": [
                {
                    "record_id": "credit_line:nested-malformed-audit",
                    "effective_date": {
                        "value": "2024.01.02",
                        "raw": "OLDER_OBSERVATION",
                        "canonical_raw": "MALFORMED_BUT_PRESERVED",
                    },
                }
            ]
        },
        stage="candidate_b_final_validation",
    )

    value = corrected["credit_lines"][0]["effective_date"]
    assert value["value"] == "2024-01-02"
    assert value["raw"] == "OLDER_OBSERVATION"
    assert value["canonical_raw"] == "MALFORMED_BUT_PRESERVED"
    assert value["_ocr_raw_history"] == ["2024.01.02"]


def test_material_inquiry_reason_alias_is_withheld_and_raw_is_preserved() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())

    corrected = overlay.correct_business_candidates(
        {
            "inquiry_records": [
                {
                    "record_id": "inquiry:1",
                    "inquiry_date": "2024-01-02",
                    "reason": "资款审批",
                }
            ]
        },
        stage="candidate_b_final_validation",
    )

    record = corrected["inquiry_records"][0]
    assert record["reason"] is None
    assert record["canonical_raw"]["reason"] == "资款审批"
    assert any(
        anomaly["record_id"] == "inquiry:1"
        and anomaly["field_name"] == "reason"
        and anomaly["normalized_value_withheld"] is True
        for anomaly in overlay.audit()["cell_anomalies"]
    )


def test_account_line_label_alias_does_not_rewrite_business_value_substrings() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    source = "账户 1 管理机构 营理机构有限公司"

    corrected, decision = overlay.correct_text(source, role="account_line")

    assert corrected == source
    assert decision is None


@pytest.mark.parametrize(
    "evidence_ids",
    (
        None,
        [],
        [""],
        [" repair:identity:1"],
        ["repair:identity:1", "repair:identity:1"],
        [1],
    ),
)
def test_schema_assigned_field_rejects_missing_or_malformed_candidate_evidence(
    evidence_ids: object,
) -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    line: dict[str, object] = {
        "text": "11010519491231002X",
        "confidence": 0.99,
        "bbox": [1, 1, 20, 10],
    }
    if evidence_ids is not None:
        line["evidence_ids"] = evidence_ids
    overlay.install_business_repair_evidence(
        [{"page": 1, "source_page": 9, "lines": [line]}],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None
    assert overlay.audit()["decisions"] == []


@pytest.mark.parametrize("target_evidence_ids", (None, [], ["native:identity:1", "native:identity:1"]))
def test_schema_assigned_field_rejects_missing_or_replayed_target_evidence(
    target_evidence_ids: object,
) -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    {
                        "text": "11010519491231002X",
                        "confidence": 0.99,
                        "bbox": [1, 1, 20, 10],
                        "evidence_ids": ["repair:identity:1"],
                    }
                ],
            }
        ],
        affected_pages={1},
    )
    target_ref = _exact_identity_repair_target_ref()
    if target_evidence_ids is None:
        target_ref.pop("evidence_ids")
    else:
        target_ref["evidence_ids"] = target_evidence_ids

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(target_ref,),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


@pytest.mark.parametrize("replay_scope", ("target", "page"))
def test_schema_assigned_field_rejects_replayed_candidate_evidence(
    replay_scope: str,
) -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    candidate_id = (
        "native:identity:1" if replay_scope == "target" else "repair:identity:replayed"
    )
    lines: list[dict[str, object]] = [
        {
            "text": "11010519491231002X",
            "confidence": 0.99,
            "bbox": [1, 1, 20, 10],
            "evidence_ids": [candidate_id],
        }
    ]
    if replay_scope == "page":
        lines.append(
            {
                "text": "unrelated",
                "confidence": 0.8,
                "bbox": [100, 100, 120, 110],
                "evidence_ids": [candidate_id],
            }
        )
    overlay.install_business_repair_evidence(
        [{"page": 1, "source_page": 9, "lines": lines}],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


def test_schema_assigned_field_rejects_competing_exact_candidates_regardless_of_margin() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(
        [
            {
                "page": 1,
                "source_page": 9,
                "lines": [
                    {
                        "text": "11010519491231002X",
                        "confidence": 0.99,
                        "bbox": [1, 1, 20, 4],
                        "evidence_ids": ["repair:identity:a"],
                    },
                    {
                        "text": "110105194912310011",
                        "confidence": 0.75,
                        "bbox": [1, 6, 20, 10],
                        "evidence_ids": ["repair:identity:b"],
                    },
                ],
            }
        ],
        affected_pages={1},
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=(_exact_identity_repair_target_ref(),),
    )

    assert corrected == "1101051949123100?X"
    assert decision is None


def test_damaged_inquiry_date_is_not_repaired_from_a_crop() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    pages = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {
                    "text": "39 2024.月:21 深圳市华融融资担保有限公司 担保资格审查",
                    "confidence": 0.7,
                    "bbox": [1, 1, 20, 10],
                    "evidence_ids": ["ocr:damaged-date"],
                }
            ],
        }
    ]

    corrected = overlay.corrected_evidence_pages(pages)

    line = corrected[0]["lines"][0]
    assert line["text"] == pages[0]["lines"][0]["text"]
    assert "ocr_correction" not in line
    assert overlay.audit()["repair_evidence_reparse_attempt_count"] == 0


def test_invalid_summary_value_is_withheld_without_whole_page_replay() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "personal_detail_summary_cells": [
            {
                "record_id": "cell:count",
                "column_label": "月份数",
                "value": "达",
                "source_refs": [
                    {
                        "logical_page": 1,
                        "bbox": [30, 20, 40, 40],
                        "geometry_scope": "cell",
                    }
                ],
            }
        ]
    }

    corrected = overlay.correct_business_candidates(payload, stage="native_business")
    audit = overlay.audit()

    assert corrected["personal_detail_summary_cells"][0]["value"] is None
    assert audit["repair_evidence_reparse_attempt_count"] == 0
    assert audit["ocr_started_by_correction_overlay"] is False
    assert audit["abnormal_cell_count"] == 1
    assert audit["decisions"] == []


def test_repayment_linking_uses_global_predecessor_and_collapses_duplicate_months() -> None:
    accounts = [
        {
            "account_id": "credit_account:loan:1",
            "account_identifier": "ACCOUNT0001",
            "page": 4,
            "bbox": [10, 10, 100, 40],
            "sequence": 1,
        }
    ]
    grids = [{"grid_id": "grid:1", "page": 6, "bbox": [10, 100, 200, 200]}]
    records = [
        {
            "year": 2024,
            "month": 1,
            "status": "unknown",
            "confidence": 0.4,
            "source_cell_refs": [{"grid_id": "grid:1", "page": 6}],
        },
        {
            "year": 2024,
            "month": 1,
            "status": "N",
            "confidence": 0.9,
            "recognition_source": "cell_crop_consensus",
            "source_cell_refs": [{"grid_id": "grid:1", "page": 6, "cell": "1"}],
        },
    ]

    linked = link_repayment_records_to_accounts(records, accounts, grids)

    assert len(linked) == 1
    assert linked[0]["account_id"] == "credit_account:loan:1"
    assert linked[0]["account_identifier"] == "ACCOUNT0001"
    assert linked[0]["status"] == "N"
    assert linked[0]["audit"]["duplicate_month_candidates"] == 2


def test_final_validation_withholds_cross_cell_employer_text_and_preserves_raw_value() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "employment_records": [
            {
                "employment_id": "employment:1",
                "employer": "2024-01-01 示例公司 13800138000",
                "source_refs": [
                    {
                        "logical_page": 2,
                        "bbox": [10, 20, 100, 40],
                        "geometry_scope": "cell",
                        "field_name": "employer",
                    }
                ],
            }
        ]
    }

    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")
    record = corrected["employment_records"][0]

    assert record["employer"] is None
    assert record["canonical_raw"]["employer"] == "2024-01-01 示例公司 13800138000"
    assert overlay.audit()["cell_anomalies"][0]["field_name"] == "employer"


def test_document_consensus_never_rewrites_valid_individual_institution_names() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "credit_lines": [
            {"credit_line_id": "line:1", "institution": "中国银行股份有限公司"},
            {"credit_line_id": "line:2", "institution": "中国银行股份有限公司上海分行"},
            {"credit_line_id": "line:3", "institution": "中国银行股份有限公司上海分行"},
        ]
    }

    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")

    assert corrected["credit_lines"][0]["institution"] == "中国银行股份有限公司"


def test_explicit_unknown_monthly_status_is_withheld_and_reported() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "repayment_records": [
            {
                "repayment_id": "grid:1:2024-01",
                "year": 2024,
                "month": 1,
                "status": "unknown",
                "overdue_amount": None,
            }
        ]
    }

    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")

    assert corrected["repayment_records"][0]["status"] is None
    anomaly = next(
        item for item in overlay.audit()["cell_anomalies"]
        if item["field_name"] == "status"
    )
    assert anomaly["value"] == "unknown"
    assert anomaly["normalized_value_withheld"] is True


def test_final_validation_skips_internal_binding_metadata_but_checks_business_scalar() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "credit_lines": [
            {
                "record_id": "credit_line:1",
                "effective_date": "not-a-date",
                "_field_binding_quality": {
                    "institution": "native_label_column",
                    "effective_date": "canonical_cell_slot",
                },
                "_private_probe": {"institution": "native_label_column"},
                "source_refs_by_field": {
                    "effective_date": [{"binding": "canonical_cell_slot"}]
                },
            }
        ]
    }

    corrected = overlay.correct_business_candidates(
        payload, stage="candidate_b_final_validation"
    )
    anomalies = overlay.audit()["cell_anomalies"]

    assert corrected["credit_lines"][0]["effective_date"] is None
    assert corrected["credit_lines"][0]["canonical_raw"]["effective_date"] == "not-a-date"
    assert len(anomalies) == 1
    assert anomalies[0]["record_id"] == "credit_line:1"
    assert anomalies[0]["field_name"] == "effective_date"
    assert anomalies[0]["value"] == "not-a-date"
    assert not any(
        anomaly["value"]
        in {"native_label_column", "canonical_cell_slot", "canonical_packed_liability_row"}
        for anomaly in anomalies
    )
