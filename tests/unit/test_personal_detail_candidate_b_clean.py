from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    extract_candidate_b_profile,
)
from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
    link_candidate_b_repayments,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    PersonalDetailScannedVariant,
)


def test_candidate_b_branch_has_no_shared_extraction_or_assembly_imports() -> None:
    root = Path("docmirror/plugins/credit_report/personal_detail_scanned")
    active = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("candidate_b.py", "context.py", "variant.py")
    )

    assert "extract_scanned_credit_business" not in active
    assert "link_repayment_records_to_accounts" not in active
    assert "assemble_credit_report_business" not in active


def test_only_business_repair_coordinator_can_request_plugin_page_ocr() -> None:
    root = Path("docmirror/plugins/credit_report/personal_detail_scanned")
    forbidden = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "canonical_layout.py",
            "native_extraction.py",
            "native_parser.py",
            "ocr_correction.py",
            "page_topology.py",
        )
    )
    coordinator = (root / "business_repair.py").read_text(encoding="utf-8")

    assert "full_page_ocr_evidence" not in forbidden
    assert "page_ocr_loader" not in forbidden
    assert "page_ocr_loader" in coordinator


def test_variant_discards_conflicting_projector_candidates() -> None:
    authoritative = {
        "credit_accounts": [{"account_id": "candidate-b"}],
        "repayment_records": [],
    }
    context = SimpleNamespace(
        candidate_b_extraction=lambda _text: SimpleNamespace(business=authoritative)
    )

    result = PersonalDetailScannedVariant().assemble_business(
        SimpleNamespace(),
        "",
        content_mode="scanned_ocr",
        existing_collections={"credit_accounts": [{"account_id": "legacy"}]},
        existing_summary={"projected_account_count": 999},
        variant_input=context,
    )

    assert result == authoritative
    assert result is not authoritative


def test_context_builds_candidate_b_once(monkeypatch) -> None:
    marker = SimpleNamespace(business={}, section_content={}, audit={})
    calls: list[str] = []

    class Pipeline:
        def __init__(self, _context, full_text: str) -> None:
            calls.append(full_text)

        def run(self):
            return marker

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.candidate_b.CandidateBPipeline",
        Pipeline,
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}

    assert context.candidate_b_extraction("source") is marker
    assert context.candidate_b_extraction("source") is marker
    assert calls == ["source"]


def test_profile_schema_withholds_concatenated_multi_region_address() -> None:
    table = SimpleNamespace(
        table_id="profile",
        metadata={
            "raw_rows": [
                ["性别", "出生日期", "通讯地址"],
                ["男", "1990.01.02", "福建省福州市某路1号 江西省上饶市某村2号"],
            ]
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["gender"]["normalized_value"] == "男"
    assert profile["birth_date"]["normalized_value"] == "1990-01-02"
    assert profile["mailing_address"]["normalized_value"] is None
    assert profile["mailing_address"]["observation_status"] == "unreadable"
    assert context._personal_detail_extraction_issues[0]["issue_code"] == "candidate_b_profile_contract_unresolved"


def test_account_schema_recovers_anchor_with_unreadable_ordinal() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "贷记卡账户"},
                    {"text": "账户 1：发卡机构甲"},
                    {"text": "账户（发卡机构乙）"},
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert [row["account_type"] for row in rows] == ["credit_card", "credit_card"]
    assert [row["category_sequence"] for row in rows] == [1, 2]


def test_account_schema_reconstructs_split_identifier_at_date_boundary() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 4,
                "source_page": 2,
                "lines": [
                    {"text": "循环贷账户（二）"},
                    {"text": "账户 1"},
                    {"text": "管理机构 账户标识 开立日期 信用额度"},
                    {"text": "D10053310H0001"},
                    {"text": "某银行 2022052901021012089466554314 2022.05.29 10000"},
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert rows[0]["account_identifier"] == "D10053310H00012022052901021012089466554314"
    assert rows[0]["account_identifier_source"] == "canonical_anchor_table_row"


def test_account_schema_suppresses_unmatched_table_in_anchored_category(monkeypatch) -> None:
    table = {
        "account_id": "credit_account:credit_card:2",
        "account_type": "credit_card",
        "category_sequence": 2,
        "source_refs": [],
    }
    anchor = {
        "account_id": "credit_account:credit_card:1",
        "account_type": "credit_card",
        "category_sequence": 1,
        "source_refs": [],
    }
    context = SimpleNamespace()
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [], []))
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [anchor])

    accounts, _repayments, _events = native_extraction._extract_accounts(context)

    assert accounts == [anchor]
    assert context._personal_detail_extraction_issues[1]["issue_code"] == (
        "candidate_b_unmatched_account_table_suppressed"
    )


def test_account_schema_joins_misclassified_table_by_canonical_stream_position(monkeypatch) -> None:
    table = {
        "account_id": "credit_account:revolving_loan_subaccount:1",
        "account_type": "revolving_loan_subaccount",
        "category_sequence": 1,
        "account_identifier": "D0206000CA202506XZ20011136047",
        "source_refs": [{"logical_page": 4, "bbox": [10, 220, 500, 280]}],
    }
    anchor = {
        "account_id": "credit_account:revolving_loan_account:2",
        "account_type": "revolving_loan_account",
        "category_sequence": 2,
        "page": 4,
        "bbox": [10, 200, 300, 210],
        "source_refs": [{"logical_page": 4, "bbox": [10, 200, 300, 210]}],
    }
    repayment = {
        "account_id": "credit_account:revolving_loan_subaccount:1",
        "year": 2025,
        "month": 1,
        "status": "N",
    }
    context = SimpleNamespace()
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [repayment], []))
    monkeypatch.setattr(native_extraction, "_account_anchor_skeletons", lambda _context: [anchor])

    accounts, repayments, _events = native_extraction._extract_accounts(context)

    assert len(accounts) == 1
    assert accounts[0]["account_id"] == "credit_account:revolving_loan_account:2"
    assert accounts[0]["account_type"] == "revolving_loan_account"
    assert accounts[0]["account_identifier"] == "D0206000CA202506XZ20011136047"
    assert repayments[0]["account_id"] == "credit_account:revolving_loan_account:2"


def test_account_stream_match_never_spills_replay_into_older_unmatched_anchor() -> None:
    anchors = [
        {"account_id": "account:1", "page": 1, "bbox": [0, 100, 100, 110]},
        {"account_id": "account:2", "page": 2, "bbox": [0, 100, 100, 110]},
    ]
    tables = [
        {"account_id": "table:1", "source_refs": [{"logical_page": 2, "bbox": [0, 120, 100, 180]}]},
        {"account_id": "table:replay", "source_refs": [{"logical_page": 3, "bbox": [0, 20, 100, 80]}]},
    ]

    matches = native_extraction._match_account_table_observations(anchors, tables)

    assert matches == {1: 0}


def test_credit_agreement_schema_reports_identity_conflict_and_emits_one_row() -> None:
    context = SimpleNamespace()
    records = [
        {
            "credit_line_id": "obsolete:1",
            "account_identifier": "T10151210H0001ABC12345",
            "institution": "机构甲",
            "source_refs": [{"logical_page": 8}],
            "confidence": 0.9,
        },
        {
            "credit_line_id": "obsolete:2",
            "account_identifier": "T10151210H0001ABC12345",
            "institution": "机构乙",
            "source_refs": [{"logical_page": 9}],
            "confidence": 0.8,
        },
    ]

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 1
    assert reconciled[0]["institution"] == "机构甲"
    assert len(reconciled[0]["source_refs"]) == 2
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_credit_agreement_observation_conflict"
    )


def test_credit_agreement_reconciles_one_damaged_identifier_only_with_strong_business_match() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "total_limit": "100000",
    }
    records = [
        {
            "account_identifier": "T10151210H0001ABC12345",
            "_printed_sequence": 1,
            **shared,
            "confidence": 0.9,
        },
        {
            "account_identifier": "T10151210H0001ABC1234?",
            "_printed_sequence": 1,
            **shared,
            "confidence": 0.8,
        },
    ]

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 1
    assert reconciled[0]["account_identifier"] == "T10151210H0001ABC12345"
    assert not context._personal_detail_extraction_issues


def test_credit_agreement_reconciles_valid_ocr_variants_only_on_same_source_page() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "total_limit": "100000",
    }
    first = {
        "account_identifier": "T10151210H0001ABC123456789",
        "_printed_sequence": 4,
        **shared,
        "source_refs": [{"logical_page": 7}],
    }
    same_page_variant = {
        "account_identifier": "T10151210H0001ABC1234567",
        "_printed_sequence": 4,
        **shared,
        "source_refs": [{"logical_page": 7}],
    }
    other_page_agreement = {
        "account_identifier": "T10151210H0001ABC1234568",
        "_printed_sequence": 5,
        **shared,
        "source_refs": [{"logical_page": 8}],
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [first, same_page_variant, other_page_agreement],
    )

    assert len(reconciled) == 2
    assert {row["account_identifier"] for row in reconciled} == {
        "T10151210H0001ABC123456789",
        "T10151210H0001ABC1234568",
    }
    assert {row["sequence"] for row in reconciled} == {4, 5}
    assert all("_printed_sequence" not in row for row in reconciled)


def test_credit_agreement_withholds_non_unique_printed_sequences() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "account_identifier": "T10151210H0001ABC12345",
                "_printed_sequence": 3,
                "source_refs": [{"logical_page": 7}],
            },
            {
                "account_identifier": "B10151210H0001XYZ12345",
                "_printed_sequence": 3,
                "source_refs": [{"logical_page": 8}],
            },
        ],
    )

    assert all("sequence" not in row for row in rows)
    assert sum(
        issue["issue_code"] == "candidate_b_credit_agreement_sequence_unresolved"
        for issue in context._personal_detail_extraction_issues
    ) == 2


def test_credit_agreement_withholds_concatenated_limit_identifiers(monkeypatch) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "授信限额编号": "B10411000H0001799190000103302585" * 3,
        },
        source_refs=[],
        confidence=0.9,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    rows = native_extraction._extract_credit_lines(context)

    assert rows[0]["limit_identifier"] is None
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_credit_limit_identifier_unresolved"
    )


def test_inquiry_boundary_and_normalization_differences_are_semantically_equivalent() -> None:
    assert native_extraction._inquiry_business_equivalent(
        {"inquiry_date": "2024.01.02", "institution": " 示例银行股份有限公司 ", "reason": "货款审批"},
        {"inquiry_date": "2024-01-02", "institution": "示例银行股份有限公司", "reason": "贷款审批"},
    )
    assert native_extraction._inquiry_business_equivalent(
        {"inquiry_date": "2024-01-02", "institution": "安 本人", "reason": "本人查询"},
        {
            "inquiry_date": "2024-01-02",
            "institution": "本人 安",
            "reason": "本人查询(自助查询机)",
        },
    )
    assert native_extraction._inquiry_business_equivalent(
        {"inquiry_date": "2024-01-02", "institution": "美 兴业银行股份有限公司 你", "reason": "贷后管理"},
        {"inquiry_date": "2024-01-02", "institution": "兴业银行股份有限公司 你 美", "reason": "贷后管理"},
    )
    assert native_extraction._inquiry_business_equivalent(
        {
            "inquiry_date": "2024-01-02",
            "institution": "深圳前海微众银行股份有限公司 法人代表、负责人、高管等",
            "reason": "资信审查",
        },
        {
            "inquiry_date": "2024-01-02",
            "institution": "深圳前海微众银行股份有限公司",
            "reason": "法人代表、负责人、高管等资信审查",
        },
    )


def test_projection_keeps_schema_values_separate_from_raw_evidence() -> None:
    rows = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "account_id": "credit_account:credit_card:1",
                    "account_type": "credit_card",
                    "canonical_raw": {"management_institution": "OCR evidence"},
                }
            ]
        }
    )["credit_accounts"]

    assert rows[0]["normalized"]["account_id"] == "credit_account:credit_card:1"
    assert rows[0]["normalized"]["pboc_account_type_code"] == "R3"
    assert rows[0]["canonical_raw"] == {"management_institution": "OCR evidence"}


def test_monthly_link_recovers_grid_geometry_from_cells() -> None:
    accounts = [
        {"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1},
        {"account_id": "account:2", "page": 4, "bbox": [10, 400, 100, 450], "sequence": 2},
    ]
    repayments = [{"grid_id": "grid:1", "year": 2024, "month": 1, "status": "N"}]
    grids = [
        {
            "grid_id": "grid:1",
            "page": 4,
            "cells": [[{"bbox": [10, 200, 100, 220], "text": "N"}]],
        }
    ]

    linked = link_candidate_b_repayments(repayments, accounts, grids)

    assert linked[0]["account_id"] == "account:1"


def test_monthly_link_reports_page_order_fallback_when_geometry_is_missing() -> None:
    context = SimpleNamespace()
    accounts = [
        {"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1},
        {"account_id": "account:2", "page": 4, "bbox": [10, 400, 100, 450], "sequence": 2},
    ]

    linked = link_candidate_b_repayments(
        [{"grid_id": "grid:1", "year": 2024, "month": 1, "status": "unknown"}],
        accounts,
        [{"grid_id": "grid:1", "page": 4, "cells": []}],
        issue_context=context,
    )

    assert linked[0]["account_id"] == "account:1"
    assert linked[0]["audit"]["account_linkage"] == "inferred_page_order"
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_monthly_link_inferred_from_page_order"
    )


def test_inquiry_schema_joins_geometry_tokens_and_reports_inferred_sequence() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 8,
                "source_page": 4,
                "canonical_template_id": "annotations_and_inquiries",
                "lines": [
                    {"text": "1 2024.01.02 银行甲 贷款审批", "bbox": [50, 10, 390, 18]},
                    {"text": "2024,01.01", "bbox": [110, 30, 170, 38]},
                    {"text": "银行乙", "bbox": [200, 31, 280, 39]},
                    {"text": "贷后管理", "bbox": [345, 29, 390, 37]},
                ],
            }
        ]
    )

    rows = native_extraction._canonical_inquiry_line_rows(context)

    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[1]["inquiry_date"] == "2024-01-01"
    assert rows[1]["institution"] == "银行乙"
    assert rows[1]["extraction_status"] == "review"
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_inquiry_sequence_inferred_from_row_order"
    )


def test_account_schema_reports_non_dense_family_ordinals_without_inventing_rows(monkeypatch) -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([], [], []))
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: [
            {
                "account_id": "credit_account:credit_card:1",
                "sequence": 1,
                "category_sequence": 1,
                "account_type": "credit_card",
                "source_refs": [{"logical_page": 8, "bbox": [10, 10, 40, 20]}],
            },
            {
                "account_id": "credit_account:credit_card:3",
                "sequence": 2,
                "category_sequence": 3,
                "account_type": "credit_card",
                "source_refs": [{"logical_page": 9, "bbox": [10, 10, 40, 20]}],
            },
        ],
    )

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert len(accounts) == 2
    assert repayments == []
    assert events == []
    gap = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_sequence_gap"
    )
    assert gap["candidate_value"]["missing_category_sequences"] == [2]


def test_account_family_state_preserves_parenthesized_revolving_variant() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 3,
                "source_page": 2,
                "lines": [
                    {"text": "循环贷账户（二）", "bbox": [10, 10, 200, 30]},
                    {"text": "账户 1：", "bbox": [10, 40, 100, 60]},
                    {"text": "账户标识", "bbox": [10, 70, 100, 90]},
                    {"text": "查询记录", "bbox": [10, 200, 100, 220]},
                ],
            }
        ]
    )

    skeletons = native_extraction._account_anchor_skeletons(context)

    assert len(skeletons) == 1
    assert skeletons[0]["account_type"] == "revolving_loan_account"
    assert skeletons[0]["account_family_quality"] == "exact"


def test_monthly_link_reports_population_loss_when_account_ordinals_are_missing() -> None:
    gap = {
        "issue_code": "candidate_b_account_sequence_gap",
        "status": "requires_review",
        "observed_value": {"account_type": "credit_card"},
        "candidate_value": {"missing_category_sequences": [2]},
        "source_refs": [{"logical_page": 1, "bbox": [10, 10, 50, 20]}],
    }
    context = SimpleNamespace(_personal_detail_extraction_issues=[gap])
    accounts = [
        {
            "account_id": "credit_account:credit_card:1",
            "sequence": 1,
            "source_refs": [{"logical_page": 1, "bbox": [10, 10, 50, 20]}],
        }
    ]
    repayments = [
        {
            "grid_id": f"grid:{index}",
            "year": 2024,
            "month": 1,
            "status": status,
            "source_cell_refs": [{"logical_page": 1, "bbox": [10, 30, 20, 40]}],
        }
        for index, status in enumerate(("N", "1"), start=1)
    ]

    linked = link_candidate_b_repayments(
        repayments,
        accounts,
        [],
        reading_order_by_logical={1: 1},
        issue_context=context,
    )

    assert len(linked) == 1
    linkage_gap = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "monthly_linkage_collision_from_account_gap"
    )
    assert linkage_gap["observed_value"]["final_linked_row_count"] == 1
    assert linkage_gap["candidate_value"]["pre_deduplication_row_count"] == 2


def test_monthly_grid_uses_one_owner_and_matches_printed_date_range() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": "account:1",
            "account_identifier": "蚂蚁借呗合并BAD12345",
            "page": 4,
            "bbox": [10, 20, 100, 100],
            "sequence": 1,
        }
    ]
    grid = {
        "grid_id": "grid:range",
        "page": 4,
        "bbox": [10, 120, 100, 220],
        "audit": {
            "date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 2}
        },
    }
    rows = [
        {"grid_id": "grid:range", "year": 2024, "month": month, "status": "N"}
        for month in (1, 2)
    ]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert {row["account_id"] for row in linked} == {"account:1"}
    assert all("account_identifier" not in row for row in linked)
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_grid_contract_unresolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_grid_reports_missing_printed_month_without_inventing_it() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {
        "grid_id": "grid:range",
        "page": 4,
        "bbox": [10, 120, 100, 220],
        "audit": {
            "date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 2}
        },
    }

    linked = link_candidate_b_repayments(
        [{"grid_id": "grid:range", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [grid],
        issue_context=context,
    )

    assert len(linked) == 1
    issue = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_monthly_grid_contract_unresolved"
    )
    assert issue["observed_value"]["month_count"] == 1
    assert issue["candidate_value"]["printed_month_count"] == 2
