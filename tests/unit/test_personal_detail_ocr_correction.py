# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from docmirror.evidence.repair import RepairCandidate
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import _source_ref
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
    normalize_institution_name,
)
from docmirror.plugins.credit_report.scanned_business import (
    _extract_inquiries,
    link_repayment_records_to_accounts,
)


def _overlay(**kwargs) -> PersonalDetailOCRCorrectionOverlay:
    return PersonalDetailOCRCorrectionOverlay(SimpleNamespace(), enable_targeted_ocr=False, **kwargs)


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
    assert identity == "11010519491231002X"
    assert amount == "100"
    assert signed_amount == "500"
    assert birth_date == "1987-09-05"
    assert event_month == "2018-10"
    assert invalid == "amount S"
    assert (
        time_decision
        and identity_decision
        and amount_decision
        and signed_amount_decision
        and birth_date_decision
        and event_month_decision
    )
    assert invalid_decision is None
    assert all(decision.action == "applied" for decision in overlay.decisions)


def test_institution_correction_removes_debris_without_general_fuzzy_rewrite() -> None:
    assert normalize_institution_name("重庆蚂蚊消费金融有限公司 Ss") == "重庆蚂蚁消费金融有限公司"
    assert normalize_institution_name("福 中信银行股份有限公司个人信贷部") == (
        "中信银行股份有限公司个人信贷部"
    )
    assert normalize_institution_name("未知机构名称") == "未知机构名称"
    assert normalize_institution_name("新疆样例银行股份有限公司乌鲁木齐分行") == (
        "新疆样例银行股份有限公司乌鲁木齐分行"
    )
    assert normalize_institution_name("云南省农村信用社联合社") == "云南省农村信用社联合社"


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
                "source_refs": [{"logical_page": 3, "bbox": [40, 20, 50, 40]}],
            },
        ]
    }
    overlay = _overlay()

    corrected = overlay.correct_business_candidates(payload, stage="native_business")
    audit = overlay.audit()

    cells = corrected["personal_detail_summary_cells"]
    assert cells[0]["value"] == "循环贷账户一"
    assert cells[1]["value"] == "2"
    assert cells[2]["value"] == "*-"
    assert audit["audited_cell_count"] >= 3
    assert audit["abnormal_cell_count"] == 1
    assert audit["cell_anomalies"][0]["role"] == "amount_or_placeholder"
    assert audit["cell_anomalies"][0]["value"] == "*-"


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


def test_summary_source_ref_prefers_exact_cell_geometry() -> None:
    page = SimpleNamespace(page_number=3, source_page_number=2)
    table = SimpleNamespace(
        table_id="summary:1",
        bbox=[0, 0, 100, 100],
        metadata={
            "cell_bboxes": [[[1, 2, 3, 4], [5, 6, 7, 8]]],
            "cell_evidence_ids": [[['ocr:1'], ['ocr:2']]],
        },
    )

    ref = _source_ref(page, table, row=0, column=1)

    assert ref["bbox"] == [5, 6, 7, 8]
    assert ref["geometry_scope"] == "cell"
    assert ref["source"] == "native_detail_table_cell"
    assert ref["evidence_ids"] == ["ocr:2"]


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
    assert line["text"] == "2 2024.02.16 中邮消费金融有限公司 贷后管理"
    assert line["ocr_original_text"] == raw_pages[0]["lines"][0]["text"]
    assert line["ocr_correction"]["role"] == "inquiry_row"


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

    assert corrected == "12 2025.03.20 福鼎市农村信用合作联社 信用卡审批"
    assert reason_corrected == "19 2020.12.12 深圳市中融小额贷款有限公司 贷款审批"
    assert compact_date == "4 2022.12.14 平安普惠融资担保有限公司 担保资格审查"
    assert decision is not None
    assert reason_decision is not None
    assert compact_date_decision is not None


def test_corrected_inquiry_extraction_recovers_rows_and_keeps_section_sequences() -> None:
    pages = [
        {
            "page": 27,
            "source_page": 14,
            "lines": [
                {"text": "1 2024.03.08 泉州银行股份有限公司 贷后管理", "confidence": 0.99},
                {"text": "2024,02.16 中邮消费金融有限公司 费后管理", "confidence": 0.95},
                {"text": "3 2024.02.13 重庆蚂蚊消费金融有限公司 Ss 贷后管理", "confidence": 0.90},
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
    assert [record["sequence"] for record in records if record["inquiry_type"] == "institution"] == [1, 2, 3]
    assert [record["sequence"] for record in records if record["inquiry_type"] == "personal"] == [1]
    corrected = overlay.correct_business_candidates({"inquiry_records": records}, stage="test")
    assert corrected["inquiry_records"][2]["institution"] == "重庆蚂蚁消费金融有限公司"


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


def test_business_overlay_promotes_one_valid_account_identifier_and_preserves_raw_payload() -> None:
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
    assert account["management_institution"] == "重庆蚂蚁消费金融有限公司"
    assert account["account_identifier"] == "ABCD12345678"
    assert account["raw_detail_text"] == payload["credit_accounts"][0]["raw_detail_text"]
    assert overlay.audit()["applied_count"] >= 2


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


def test_targeted_crop_ocr_only_adopts_a_typed_valid_candidate() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(
        SimpleNamespace(),
        page_image_resolver=_FakeResolver(),
        repair_engine=_FakeRepairEngine(),
        enable_targeted_ocr=True,
    )

    corrected, decision = overlay.correct_text(
        "1101051949123100?X",
        role="identity_document_number",
        source_refs=({"logical_page": 1, "bbox": [1, 1, 20, 10]},),
        allow_targeted_ocr=True,
    )

    assert corrected == "11010519491231002X"
    assert decision is not None
    assert decision.method == "targeted_crop_ocr_consensus"
    assert overlay.audit()["targeted_ocr_requests"] == 1


def test_damaged_inquiry_date_triggers_bounded_typed_crop_ocr() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(
        SimpleNamespace(),
        page_image_resolver=_FakeResolver(),
        repair_engine=_FakeInquiryRepairEngine(),
        enable_targeted_ocr=True,
    )
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
    assert line["text"] == "39 2024.11.21 深圳市华融融资担保有限公司 担保资格审查"
    assert line["ocr_original_text"] == pages[0]["lines"][0]["text"]
    assert line["ocr_correction"]["method"] == "targeted_crop_ocr_consensus"
    assert overlay.audit()["targeted_ocr_requests"] == 1


def test_invalid_summary_value_uses_only_cell_scoped_targeted_ocr() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(
        SimpleNamespace(),
        page_image_resolver=_FakeResolver(),
        repair_engine=_FakeCellRepairEngine(),
        enable_targeted_ocr=True,
    )
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

    assert corrected["personal_detail_summary_cells"][0]["value"] == "2"
    assert audit["targeted_ocr_requests"] == 1
    assert audit["abnormal_cell_count"] == 0
    assert any(decision["method"] == "targeted_crop_ocr_consensus" for decision in audit["decisions"])


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
