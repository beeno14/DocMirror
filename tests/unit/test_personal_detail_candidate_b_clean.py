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
            "canonical_template_id": "report_header_and_identity",
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
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["gender"]["normalized_value"] == "男"
    assert profile["birth_date"]["normalized_value"] == "1990-01-02"
    assert profile["mailing_address"]["normalized_value"] is None
    assert profile["mailing_address"]["observation_status"] == "unreadable"
    assert context._personal_detail_extraction_issues[0]["issue_code"] == "candidate_b_profile_contract_unresolved"


def test_profile_schema_ignores_residence_and_employment_contact_tables() -> None:
    profile_table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
                "raw_rows": [
                [
                    "性别", "出生日期", "婚姻状况", "就业状况", "学历", "学位", "国籍",
                    "手机号码", "住宅电话", "单位电话", "电子邮箱", "通讯地址", "户籍地址",
                ],
                [
                    "男", "--", "--", "--", "--", "--", "--",
                    "13800138000", "010-12345678", "021-87654321", "--",
                    "北京市朝阳区示例路1号", "--",
                ],
            ],
        },
        rows=[],
    )
    residence_table = SimpleNamespace(
        table_id="residence-history",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号", "居住地址", "住宅电话", "通讯地址", "居住状况"],
                ["1", "上海市浦东新区示例路2号", "010-99999999", "上海市浦东新区示例路2号", "租房"],
            ],
        },
        rows=[],
    )
    employment_table = SimpleNamespace(
        table_id="employment-history",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["编号", "工作单位", "单位地址", "单位电话", "通讯地址"],
                ["1", "示例单位", "深圳市南山区示例路3号", "0755-99999999", "深圳市南山区示例路3号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[profile_table, residence_table, employment_table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["mobile_phone"]["normalized_value"] == "13800138000"
    assert profile["residence_phone"]["normalized_value"] == "010-12345678"
    assert profile["work_phone"]["normalized_value"] == "021-87654321"
    assert profile["mailing_address"]["normalized_value"] == "北京市朝阳区示例路1号"
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_ignores_noncanonical_profile_like_tables() -> None:
    table = SimpleNamespace(
        table_id="unrelated-contact-table",
        metadata={
            "canonical_template_id": "credit_account_detail",
            "raw_rows": [
                ["性别", "手机号码", "通讯地址"],
                ["女", "13900139000", "广东省深圳市南山区示例路1号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=5,
        source_page_number=3,
        canonical_template_id="credit_account_detail",
        tables=[table],
    )

    assert extract_candidate_b_profile(SimpleNamespace(pages=[page])) == {}


def test_profile_schema_rejects_alphanumeric_phone_contamination() -> None:
    table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["性别", "手机号码", "住宅电话", "单位电话"],
                ["男", "138O0138000", "010A12345678", "021B87654321"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    for field in ("mobile_phone", "residence_phone", "work_phone"):
        assert profile[field]["normalized_value"] is None
        assert profile[field]["observation_status"] == "unreadable"
    assert {issue["field_name"] for issue in context._personal_detail_extraction_issues} >= {
        "mobile_phone",
        "residence_phone",
        "work_phone",
    }


def test_profile_schema_does_not_report_roles_absent_from_printed_template() -> None:
    table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [
                ["性别", "手机号码", "通讯地址"],
                ["女", "13800138000", "北京市朝阳区示例路1号"],
            ],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert "work_phone" not in profile
    assert "residence_phone" not in profile
    assert not hasattr(context, "_personal_detail_extraction_issues")


def test_profile_schema_reports_visible_label_with_unreadable_value() -> None:
    table = SimpleNamespace(
        table_id="identity-profile",
        metadata={
            "canonical_template_id": "report_header_and_identity",
            "raw_rows": [["性别", "国籍"], ["女", ""]],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="report_header_and_identity",
        tables=[table],
    )
    context = SimpleNamespace(pages=[page])

    profile = extract_candidate_b_profile(context)

    assert profile["nationality"]["observation_status"] == "unreadable"
    assert context._personal_detail_extraction_issues[0]["field_name"] == "nationality"


def test_account_schema_withholds_unreadable_ordinal_and_uses_provisional_id() -> None:
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
    assert rows[0]["category_sequence"] == 1
    assert "category_sequence" not in rows[1]
    assert rows[1]["account_id"].startswith("credit_account_provisional:")
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_account_printed_ordinal_unresolved"
    assert issue["target_record_id"] == rows[1]["account_id"]
    assert "encounter_order_not_used" in issue["reason_codes"]


def test_account_schema_withholds_every_duplicate_printed_ordinal() -> None:
    context = SimpleNamespace(
        corrected_evidence_pages=lambda: [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "贷记卡账户", "bbox": [10, 10, 100, 20]},
                    {"text": "账户 1：发卡机构甲", "bbox": [10, 30, 200, 40]},
                    {"text": "账户 1：发卡机构乙", "bbox": [10, 130, 200, 140]},
                ],
            }
        ]
    )

    rows = native_extraction._account_anchor_skeletons(context)

    assert len(rows) == 2
    assert all("category_sequence" not in row for row in rows)
    assert len({row["account_id"] for row in rows}) == 2
    assert all(row["account_id"].startswith("credit_account_provisional:") for row in rows)
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "candidate_b_account_printed_ordinal_unresolved"
    ]
    assert len(issues) == 2
    assert all(issue["observed_value"]["ordinal_status"] == "printed_duplicate" for issue in issues)


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


def test_account_schema_keeps_unanchored_table_only_as_reported_partial(monkeypatch) -> None:
    table = {
        "account_id": "credit_account_table_observation:abc",
        "account_type": "credit_card",
        "sequence": 1,
        "category_sequence": 1,
        "management_institution": "样例银行",
        "source_refs": [{"logical_page": 2, "bbox": [10, 100, 500, 200]}],
    }
    context = SimpleNamespace()
    monkeypatch.setattr(native_extraction, "_extract_table_accounts", lambda _context: ([table], [], []))

    accounts, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    assert len(accounts) == 1
    assert accounts[0]["management_institution"] == "样例银行"
    assert "category_sequence" not in accounts[0]
    assert accounts[0]["extraction_status"] == "review"
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_account_anchor_population_missing"
    assert issue["target_record_id"] == accounts[0]["account_id"]
    assert "encounter_order_not_used" in issue["reason_codes"]


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


def test_account_table_match_does_not_use_category_or_encounter_order_without_geometry() -> None:
    anchors = [
        {
            "account_id": "credit_account:credit_card:1",
            "account_type": "credit_card",
            "category_sequence": 1,
            "source_refs": [],
        }
    ]
    tables = [
        {
            "account_id": "credit_account_table_observation:abc",
            "account_type": "credit_card",
            "category_sequence": 1,
            "source_refs": [],
        }
    ]

    assert native_extraction._match_account_table_observations(anchors, tables) == {}


def test_account_table_match_requires_verified_segment_for_later_page() -> None:
    anchor = {
        "account_id": "account:1",
        "page": 1,
        "bbox": [0, 100, 100, 110],
        "_canonical_segment": {
            "pages": [{"logical_page": 1, "min_y": 100.0, "max_y": None}]
        },
    }
    table = {
        "account_id": "table:1",
        "source_refs": [{"logical_page": 2, "bbox": [0, 20, 100, 80]}],
    }

    assert native_extraction._match_account_table_observations([anchor], [table]) == {}

    anchor["_canonical_segment"]["pages"].append(
        {
            "logical_page": 2,
            "min_y": 0.0,
            "max_y": 90.0,
            "continuation_verified": True,
        }
    )
    assert native_extraction._match_account_table_observations([anchor], [table]) == {0: 0}


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
    # Equal-provenance conflicts are not resolved by confidence.  Confidence
    # describes OCR certainty, not which of two incompatible business values
    # belongs in the canonical slot.
    assert reconciled[0]["institution"] is None
    assert len(reconciled[0]["source_refs"]) == 2
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_credit_agreement_observation_conflict"
    )


def test_credit_agreement_does_not_merge_damaged_identifier_from_business_similarity() -> None:
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

    assert len(reconciled) == 2
    assert {row.get("account_identifier") for row in reconciled} == {
        "T10151210H0001ABC12345",
        None,
    }
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "fuzzy_identifier_merge_forbidden" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_same_page_without_shared_card_geometry_is_not_merge_authority() -> None:
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
        "_canonical_card_key": "credit_agreement:4",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
        "source_refs": [{"logical_page": 7}],
    }
    same_page_variant = {
        "account_identifier": "T10151210H0001ABC1234567",
        "_printed_sequence": 4,
        "_canonical_card_key": "credit_agreement:4",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
        "source_refs": [{"logical_page": 7}],
    }
    other_page_agreement = {
        "account_identifier": "T10151210H0001ABC1234568",
        "_printed_sequence": 5,
        "_canonical_card_key": "credit_agreement:5",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
        "source_refs": [{"logical_page": 8}],
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [first, same_page_variant, other_page_agreement],
    )

    assert len(reconciled) == 3
    assert {row["account_identifier"] for row in reconciled} == {
        "T10151210H0001ABC123456789",
        "T10151210H0001ABC1234567",
        "T10151210H0001ABC1234568",
    }
    assert [row.get("sequence") for row in reconciled].count(5) == 1
    assert [row.get("sequence") for row in reconciled].count(None) == 2
    assert all("_printed_sequence" not in row for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "canonical_card_anchor_not_cross_plane_unique" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_prefixes_without_shared_sequence_are_never_collapsed() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环贷款额度",
        "effective_date": "2024-01-01",
        "total_limit": "100000",
        "source_refs": [{"logical_page": 7}],
    }
    records = [
        {"account_identifier": "T10151210H0001ABC1234", "_printed_sequence": 1, **shared},
        {"account_identifier": "T10151210H0001ABC12345", **shared},
    ]

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, records)

    assert len(reconciled) == 2
    assert {row["account_identifier"] for row in reconciled} == {
        "T10151210H0001ABC1234",
        "T10151210H0001ABC12345",
    }
    # Shared business fields and an identifier prefix are not structural
    # evidence that two canonical agreements are one record.  Retain both,
    # without falsely flagging otherwise valid agreements as an identity error.
    assert not any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_same_physical_page_and_containment_do_not_prove_identity() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "effective_date": "2019-05-21",
        "used_limit": "36393",
    }
    damaged = {
        "account_identifier": "RB10711000H0001100000111111111498898000000",
        "institution": None,
        "facility_type": "",
        "total_limit": "364002",
        "_printed_sequence": 7,
        "source_refs": [{"logical_page": 25, "source_page": 13}],
        **shared,
    }
    complete = {
        "account_identifier": "B10711000H000110000011111111149889800",
        "institution": "中国光大银行股份有限公司",
        "facility_type": "信用卡共享额度",
        "total_limit": "36400",
        "credit_limit": "--",
        "limit_identifier": "--",
        "currency": "CNY",
        "due_date": "2029-05-21",
        "_printed_sequence": 7,
        "source_refs": [{"logical_page": 26, "source_page": 13}],
        "source_refs_by_field": {
            field_name: [
                {
                    "logical_page": 26,
                    "source_page": 13,
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                }
            ]
            for field_name in (
                "institution",
                "facility_type",
                "effective_date",
                "due_date",
                "total_limit",
                "credit_limit",
                "used_limit",
                "limit_identifier",
                "currency",
            )
        },
        "_field_binding_quality": {
            field_name: "canonical_cell_slot"
            for field_name in (
                "institution",
                "facility_type",
                "effective_date",
                "due_date",
                "total_limit",
                "credit_limit",
                "used_limit",
                "limit_identifier",
                "currency",
            )
        },
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context, [damaged, complete]
    )

    assert len(reconciled) == 2
    assert {row.get("account_identifier") for row in reconciled} == {
        damaged["account_identifier"],
        complete["account_identifier"],
    }
    assert all(row.get("sequence") is None for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "exact_card_identity_not_proven" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_exact_leading_insertion_can_reconcile_same_card() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "_printed_sequence": 7,
        "effective_date": "2019-05-21",
        "used_limit": "36393",
        "source_refs": [{"logical_page": 25, "source_page": 13}],
    }
    damaged = {
        "account_identifier": "RB10711000H0001100000111111111498898000000",
        **shared,
    }
    canonical = {
        "account_identifier": "B10711000H0001100000111111111498898000000",
        "institution": "中国光大银行股份有限公司",
        "facility_type": "信用卡共享额度",
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [damaged, canonical],
    )

    assert len(reconciled) == 1
    assert reconciled[0]["account_identifier"] == canonical["account_identifier"]
    assert reconciled[0]["sequence"] == 7
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identifier_variant"
        and "exact_leading_ocr_insertion_corrected" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_exact_ordinal_and_verified_continuation_can_reconcile_observations() -> None:
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        tables_continue=lambda left, right: (left, right) == ("agreement:left", "agreement:right"),
    )
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "循环额度",
        "effective_date": "2024-01-01",
        "used_limit": "50000",
    }
    weak = {
        "account_identifier": "T10151210H0001ABC12340",
        "_printed_sequence": 4,
        "source_refs": [{"logical_page": 7, "table_id": "agreement:left"}],
        **shared,
    }
    strong = {
        "account_identifier": "T10151210H0001ABC12345",
        "_printed_sequence": 4,
        "source_refs": [{"logical_page": 8, "table_id": "agreement:right"}],
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "logical_page": 8,
                    "table_id": "agreement:right",
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                }
            ]
        },
        "_field_binding_quality": {"account_identifier": "canonical_cell_slot"},
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(context, [weak, strong])

    assert len(reconciled) == 1
    assert reconciled[0]["account_identifier"] == strong["account_identifier"]
    assert reconciled[0]["sequence"] == 4
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identifier_variant"
        and "higher_provenance_value_retained_for_review" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_exact_cross_plane_card_anchor_reconciles_ocr_observations() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "_printed_sequence": 2,
        "_canonical_card_key": "credit_agreement:2",
        "institution": "中国光大银行股份有限公司",
        "effective_date": "2019-12-01",
        "used_limit": "0",
    }
    native = {
        "account_identifier": "B10711000H0001100001111112446567900000",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
                "logical_page": 15,
            }
        ],
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "source": "native_detail_tolerant_table_cell",
                    "geometry_scope": "cell",
                    "binding": "label_column",
                }
            ]
        },
        "_field_binding_quality": {"account_identifier": "native_label_column"},
        **shared,
    }
    corrected = {
        "account_identifier": "B10711000H0001100000111111112446567900000",
        "facility_type": "信用卡共享额度",
        "total_limit": "0",
        "currency": "CNY",
        "validity_type": "perpetual",
        "_canonical_card_anchor_refs": [
            {
                "source": "personal_detail_corrected_page_cell",
                "binding": "canonical_card_anchor",
                "logical_page": 15,
            }
        ],
        "source_refs_by_field": {
            "account_identifier": [
                {
                    "source": "personal_detail_corrected_page_cell",
                    "geometry_scope": "cell",
                    "binding": "canonical_label_slot",
                }
            ]
        },
        "_field_binding_quality": {"account_identifier": "canonical_cell_slot"},
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [native, corrected],
    )

    assert len(reconciled) == 1
    assert reconciled[0]["sequence"] == 2
    assert reconciled[0]["account_identifier"] == corrected["account_identifier"]
    assert "_canonical_card_key" not in reconciled[0]
    assert "_canonical_card_anchor_refs" not in reconciled[0]
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identifier_variant"
        and "exact_canonical_card_anchor_cross_plane" in issue.get("reason_codes", ())
        and "canonical_anchor_provenance_selection" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )
    assert not any(
        issue.get("issue_code")
        in {
            "candidate_b_credit_agreement_identity_ambiguous",
            "candidate_b_credit_agreement_sequence_unresolved",
        }
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_cross_plane_anchor_does_not_override_business_conflict() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "_printed_sequence": 2,
        "_canonical_card_key": "credit_agreement:2",
        "institution": "示例银行股份有限公司",
        "facility_type": "信用卡共享额度",
        "effective_date": "2019-12-01",
    }
    native = {
        "account_identifier": "B10711000H0001100001111112446567900000",
        "used_limit": "0",
        "_canonical_card_anchor_refs": [
            {
                "source": "native_detail_canonical_anchor_text",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
    }
    corrected = {
        "account_identifier": "B10711000H0001100000111111112446567900000",
        "used_limit": "999",
        "_canonical_card_anchor_refs": [
            {
                "source": "personal_detail_corrected_page_cell",
                "binding": "canonical_card_anchor",
            }
        ],
        **shared,
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [native, corrected],
    )

    assert len(reconciled) == 2
    assert all(row.get("sequence") is None for row in reconciled)
    assert any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        and "canonical_card_anchor_business_conflict" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_business_similarity_alone_does_not_raise_identity_issue() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    shared = {
        "institution": "示例银行股份有限公司",
        "facility_type": "信用卡共享额度",
        "effective_date": "2019-12-01",
        "used_limit": "0",
    }

    reconciled = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "account_identifier": "B10711000H0001100001111112446567900000",
                "_printed_sequence": 1,
                "source_refs": [{"table_id": "agreement-list"}],
                **shared,
            },
            {
                "account_identifier": "B10711000H0001100000111111112446567900000",
                "_printed_sequence": 2,
                "source_refs": [{"table_id": "agreement-list"}],
                **shared,
            },
        ],
    )

    assert len(reconciled) == 2
    assert [row.get("sequence") for row in reconciled] == [1, 2]
    assert not any(
        issue.get("issue_code") == "candidate_b_credit_agreement_identity_ambiguous"
        for issue in context._personal_detail_extraction_issues
    )


def test_credit_agreement_reports_required_fields_missing_after_final_merge() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [
            {
                "credit_line_id": "credit-line:missing-fields",
                "account_identifier": "B10512900H00010010011135264974289",
                "institution": "",
                "facility_type": None,
                "effective_date": "2020-03-29",
                "_printed_sequence": 1,
                "source_refs": [{"logical_page": 8, "bbox": [1, 2, 30, 40]}],
            }
        ],
    )

    assert len(rows) == 1
    final_id = rows[0]["credit_line_id"]
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_credit_agreement_required_field_unresolved"
    ]
    assert {issue["field_name"] for issue in issues} == {
        "institution",
        "facility_type",
        "due_date",
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "currency",
    }
    assert {issue["target_record_id"] for issue in issues} == {final_id}
    assert all(issue["source_refs"][0]["logical_page"] == 8 for issue in issues)
    assert all(
        issue["reason_codes"]
        == [
            "required_field_missing",
            "canonical_credit_agreement_field_unresolved",
            "field_slot_not_safely_bound",
            "preserved_unknown_value",
        ]
        for issue in issues
    )


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


def test_credit_agreement_dash_glyphs_are_explicit_absence_not_failures(monkeypatch) -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
        PBOCPersonalDetailNativeParser,
    )

    candidate = SimpleNamespace(
        fields={
            "授信协议标识": "T10151210H0001ABC12345",
            "授信额度": "－",
            "授信限额": "——",
            "已用额度": "--",
            "授信限额编号": "--",
            "到期日期": "—",
        },
        source_refs=[],
        source_refs_by_field={},
        binding_quality_by_field={},
        unresolved_labels=frozenset(),
        observed_labels=frozenset(),
        confidence=0.9,
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _self, dataset_name: [candidate] if dataset_name == "credit_lines" else [],
    )
    context = SimpleNamespace(_personal_detail_extraction_issues=[])

    extracted = native_extraction._extract_credit_lines(context)
    rows = native_extraction.reconcile_candidate_b_credit_lines(context, extracted)

    assert len(rows) == 1
    assert rows[0]["total_limit"] is None
    assert rows[0]["credit_limit"] is None
    assert rows[0]["used_limit"] is None
    assert rows[0]["limit_identifier"] is None
    assert rows[0]["validity_type"] == "unknown"
    assert {
        "total_limit",
        "credit_limit",
        "used_limit",
        "limit_identifier",
        "due_date",
    } <= set(rows[0]["_source_absent_fields"])
    assert not any(
        issue.get("field_name")
        in {"total_limit", "credit_limit", "used_limit", "limit_identifier", "due_date"}
        for issue in context._personal_detail_extraction_issues
    )


def test_inquiry_boundary_and_normalization_differences_require_exact_institution_identity() -> None:
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
    assert not native_extraction._inquiry_business_equivalent(
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


def test_public_record_projection_keeps_canonical_authorities_and_typed_fields() -> None:
    def table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return SimpleNamespace(
            table_id=table_id,
            bbox=[0, 0, 600, 200],
            metadata={"raw_rows": rows},
        )

    page = SimpleNamespace(
        page_number=13,
        source_page_number=13,
        tables=[
            table(
                "tax",
                [
                    ["编号", "主管税务机关", "欠税总额", "欠税统计日期"],
                    ["1", "某市税务局", "500", "2023.03.17"],
                ],
            ),
            table(
                "penalty",
                [
                    ["编号", "处罚机构", "处罚内容", "处罚金额", "生效日期", "截止日期", "行政复议结果"],
                    ["1", "某市监管局", "警告", "400", "2021.08", "2024.07", "--"],
                ],
            ),
            table(
                "award",
                [
                    ["编号", "奖励机构", "奖励内容", "生效日期", "截止日期"],
                    ["1", "某市总工会", "先进工作者", "2023.02", "2033.06"],
                ],
            ),
        ],
    )

    public_rows = native_extraction._extract_public_records(SimpleNamespace(pages=[page]))
    projected = project_personal_detail_datasets({"public_records": public_rows})

    tax = projected["tax_arrears_records"][0]
    assert tax["tax_authority"] == "某市税务局"
    assert tax["arrears_amount"] == 500
    assert tax["reporting_amount_currency"] == "CNY"
    penalty = projected["administrative_penalty_records"][0]
    assert penalty["authority"] == "某市监管局"
    assert penalty["penalty_content"] == "警告"
    assert penalty["administrative_review_result"] is None
    award = projected["administrative_award_records"][0]
    assert award["authority"] == "某市总工会"
    assert award["award_content"] == "先进工作者"


def test_housing_fund_blocks_use_canonical_boundaries_across_pages() -> None:
    layouts = {layout["name"]: layout for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS}
    base = layouts["housing_fund_base"]
    provider = layouts["housing_fund_provider"]

    def header(layout: dict[str, object]) -> list[str]:
        aliases = layout["aliases"]
        fields = layout["fields"]
        assert isinstance(aliases, dict)
        assert isinstance(fields, dict)
        return [aliases[role][0] for role in fields]

    def table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return SimpleNamespace(
            table_id=table_id,
            bbox=[0, 0, 600, 200],
            metadata={"raw_rows": rows},
        )

    page_1 = SimpleNamespace(
        page_number=13,
        source_page_number=13,
        tables=[
            table(
                "housing-base-1",
                [
                    header(base),
                    ["", "", "", "", "", "", "", ""],
                    ["Fuzhou", "2018.09.03", "2018.09", "2023.08", "active", "906", "6%", "6%"],
                ],
            )
        ],
    )
    page_2 = SimpleNamespace(
        page_number=14,
        source_page_number=14,
        tables=[
            table(
                "housing-continuation-and-second-record",
                [
                    header(provider),
                    ["", ""],
                    ["S 示例科技有限公司", "2023.08"],
                    header(base),
                    ["Xiamen", "2015.06.25", "2015.06", "2018.08", "closed", "1023", "8%", "8%"],
                    header(provider),
                    ["限 示例服务有限公司", "2023.08"],
                ],
            )
        ],
    )
    context = SimpleNamespace(
        pages=[page_1, page_2],
        _personal_detail_extraction_issues=[],
    )

    public_rows = native_extraction._extract_public_records(context)
    projected = project_personal_detail_datasets({"public_records": public_rows})
    housing = projected["housing_fund_records"]

    assert [row["sequence"] for row in housing] == [1, 2]
    assert [row["employer"] for row in housing] == ["示例科技有限公司", "示例服务有限公司"]
    assert [row["contribution_location"] for row in housing] == ["Fuzhou", "Xiamen"]
    assert len({row["public_record_id"] for row in housing}) == 2
    assert not context._personal_detail_extraction_issues


def test_housing_fund_continuation_without_start_is_reported_and_not_invented() -> None:
    provider = next(
        layout
        for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS
        if layout["name"] == "housing_fund_provider"
    )
    header = [provider["aliases"][role][0] for role in provider["fields"]]
    page = SimpleNamespace(
        page_number=14,
        source_page_number=14,
        tables=[
            SimpleNamespace(
                table_id="orphan-housing-provider",
                bbox=[0, 0, 600, 100],
                metadata={"raw_rows": [header, ["Employer A", "2023.08"]]},
            )
        ],
    )
    context = SimpleNamespace(pages=[page], _personal_detail_extraction_issues=[])

    assert native_extraction._extract_public_records(context) == []
    assert [issue["issue_code"] for issue in context._personal_detail_extraction_issues] == [
        "candidate_b_public_record_continuation_unowned"
    ]


def test_housing_fund_nonadjacent_continuation_is_not_attached() -> None:
    layouts = {layout["name"]: layout for layout in native_extraction._PUBLIC_CANONICAL_LAYOUTS}
    base = layouts["housing_fund_base"]
    provider = layouts["housing_fund_provider"]

    def header(layout: dict[str, object]) -> list[str]:
        aliases = layout["aliases"]
        fields = layout["fields"]
        assert isinstance(aliases, dict)
        assert isinstance(fields, dict)
        return [aliases[role][0] for role in fields]

    def page(number: int, table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=number,
            source_page_number=number,
            tables=[
                SimpleNamespace(
                    table_id=table_id,
                    bbox=[0, 0, 600, 100],
                    metadata={"raw_rows": rows},
                )
            ],
        )

    context = SimpleNamespace(
        pages=[
            page(
                13,
                "housing-base",
                [
                    header(base),
                    ["Fuzhou", "2018.09.03", "2018.09", "2023.08", "active", "906", "6%", "6%"],
                ],
            ),
            page(15, "late-provider", [header(provider), ["Employer A", "2023.08"]]),
        ],
        reading_order_by_logical={13: 13, 15: 15},
        _personal_detail_extraction_issues=[],
    )

    rows = native_extraction._extract_public_records(context)
    housing = [row for row in rows if row["record_type"] == "housing_fund"]
    issues = context._personal_detail_extraction_issues

    assert len(housing) == 1
    assert "employer" not in housing[0]
    assert [issue["issue_code"] for issue in issues] == [
        "candidate_b_public_record_continuation_missing",
        "candidate_b_public_record_continuation_unowned",
    ]
    assert issues[0]["candidate_value"]["missing_fields"] == [
        "employer",
        "information_updated_month",
    ]


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


def test_monthly_link_withholds_owner_when_grid_geometry_is_missing() -> None:
    context = SimpleNamespace()
    accounts = [
        {"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1},
        {"account_id": "account:2", "page": 4, "bbox": [10, 400, 100, 450], "sequence": 2},
    ]

    linked = link_candidate_b_repayments(
        [{"grid_id": "grid:1", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [{"grid_id": "grid:1", "page": 4, "cells": []}],
        issue_context=context,
    )

    assert linked == []
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_monthly_grid_owner_unresolved"
    )
    assert "target_record_id" not in context._personal_detail_extraction_issues[0]
    assert context._personal_detail_extraction_issues[0]["observed_value"]["observed_candidate_count"] == 1


def test_monthly_cross_page_predecessor_is_not_used_as_an_owner() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    repayment_id = "grid:continued:2024-01"
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": repayment_id,
                "grid_id": "grid:continued",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
                "source_cell_refs": [{"grid_id": "grid:continued", "logical_page": 6}],
            }
        ],
        [
            {
                "account_id": "account:1",
                "page": 4,
                "bbox": [10, 20, 100, 100],
                "sequence": 1,
            }
        ],
        [{"grid_id": "grid:continued", "page": 6, "bbox": [10, 100, 200, 200]}],
        reading_order_by_logical={4: 4, 6: 6},
        issue_context=context,
    )

    assert linked == []
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    assert "target_record_id" not in issue
    assert issue["observed_value"]["grid_id"] == "grid:continued"
    assert issue["observed_value"]["observed_candidate_months"] == ["2024-01"]


def test_monthly_verified_cross_page_account_segment_is_silent() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": "grid:continued:2024-01",
                "grid_id": "grid:continued",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
            }
        ],
        [
            {
                "account_id": "account:1",
                "page": 4,
                "bbox": [10, 20, 100, 100],
                "sequence": 1,
                "_canonical_segment": {
                    "pages": [
                        {"logical_page": 4, "min_y": 20.0, "max_y": None},
                        {
                            "logical_page": 6,
                            "min_y": 0.0,
                            "max_y": 300.0,
                            "continuation_verified": True,
                        },
                    ]
                },
            }
        ],
        [{"grid_id": "grid:continued", "page": 6, "bbox": [10, 100, 200, 200]}],
        issue_context=context,
    )

    assert linked[0]["account_id"] == "account:1"
    assert "account_linkage" not in linked[0].get("audit", {})
    assert context._personal_detail_extraction_issues == []


def test_monthly_ambiguous_account_segments_withhold_orphan_rows() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [
        {
            "account_id": account_id,
            "_canonical_segment": {
                "pages": [{"logical_page": 4, "min_y": 20.0, "max_y": 300.0}]
            },
        }
        for account_id in ("account:1", "account:2")
    ]

    linked = link_candidate_b_repayments(
        [{"repayment_id": "grid:1:2024-01", "grid_id": "grid:1", "year": 2024, "month": 1, "status": "N"}],
        accounts,
        [{"grid_id": "grid:1", "page": 4, "bbox": [10, 100, 200, 200]}],
        issue_context=context,
    )

    assert linked == []
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_monthly_grid_owner_unresolved"
    assert issue["observed_value"]["linkage_basis"] == "ambiguous_account_segments"
    assert issue["observed_value"]["candidate_account_ids"] == ["account:1", "account:2"]


def test_monthly_explicit_cross_page_owner_requires_segment_proof() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    linked = link_candidate_b_repayments(
        [
            {
                "repayment_id": "grid:continued:2024-01",
                "grid_id": "grid:continued",
                "account_id": "account:1",
                "year": 2024,
                "month": 1,
                "status": "N",
                "overdue_amount": "0",
            }
        ],
        [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}],
        [{"grid_id": "grid:continued", "page": 6, "bbox": [10, 100, 200, 200]}],
        issue_context=context,
    )

    assert linked == []
    assert context._personal_detail_extraction_issues[0]["issue_code"] == (
        "candidate_b_monthly_grid_owner_unresolved"
    )


def test_monthly_equivalent_duplicate_replays_merge_without_false_review() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {"grid_id": "grid:1", "page": 4, "bbox": [10, 120, 100, 220]}
    rows = [
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0.00",
            "confidence": 0.8,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "a"}],
        },
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "confidence": 0.9,
            "source_cell_refs": [{"grid_id": "grid:1", "cell": "b"}],
        },
    ]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert len(linked) == 1
    assert linked[0]["confidence"] == 0.9
    assert linked[0]["source_cell_refs"] == [
        {"grid_id": "grid:1", "cell": "b"},
        {"grid_id": "grid:1", "cell": "a"},
    ]
    assert "duplicate_month_candidates" not in linked[0].get("audit", {})
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_duplicate_conflict"
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_conflicting_duplicate_is_selected_and_reported() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {"grid_id": "grid:1", "page": 4, "bbox": [10, 120, 100, 220]}
    rows = [
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "unknown",
            "overdue_amount": None,
            "confidence": 0.4,
        },
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "year": 2024,
            "month": 1,
            "status": "N",
            "overdue_amount": "0",
            "confidence": 0.9,
        },
    ]

    linked = link_candidate_b_repayments(rows, accounts, [grid], issue_context=context)

    assert len(linked) == 1
    assert linked[0]["status"] == "N"
    assert "extraction_status" not in linked[0]
    assert not any(
        issue["issue_code"] in {
            "candidate_b_monthly_duplicate_conflict",
            "candidate_b_monthly_status_grid_unresolved",
        }
        for issue in context._personal_detail_extraction_issues
    )


def test_monthly_unresolved_status_survives_linking_for_final_correction() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    accounts = [{"account_id": "account:1", "page": 4, "bbox": [10, 20, 100, 100], "sequence": 1}]
    grid = {
        "grid_id": "grid:1",
        "page": 4,
        "bbox": [10, 120, 100, 220],
        "audit": {"date_range": {"start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 1}},
    }

    linked = link_candidate_b_repayments(
        [{"repayment_id": "grid:1:2024-01", "grid_id": "grid:1", "year": 2024, "month": 1, "status": "unknown"}],
        accounts,
        [grid],
        issue_context=context,
    )

    assert len(linked) == 1
    assert linked[0]["account_id"] == "account:1"
    assert linked[0]["status"] == "unknown"
    assert not any(
        issue["issue_code"] == "candidate_b_monthly_status_grid_unresolved"
        for issue in context._personal_detail_extraction_issues
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


def test_account_event_does_not_shift_first_nonempty_cell_into_missing_note_slot() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    page = SimpleNamespace(page_number=8, source_page_number=4, height=800)
    table = SimpleNamespace(
        table_id="account:event",
        metadata={},
        confidence=0.95,
        bbox=[10, 10, 590, 120],
    )
    account = {"account_id": "credit_account:1"}
    rows = [
        ["其他字段", "特殊事件说明"],
        ["不属于说明字段的值", ""],
    ]

    events = native_extraction._account_events(context, account, page, table, rows)

    assert len(events) == 1
    assert events[0]["event_type"] == "special_event_note"
    assert "details" not in events[0]
    assert any(
        issue.get("issue_code") == "candidate_b_account_event_slot_unresolved"
        and issue.get("field_name") == "details"
        and "positional_fallback_forbidden" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


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


def test_account_anchor_segment_crosses_page_only_after_verified_transition() -> None:
    pages = [
        {
            "page": 1,
            "source_page": 1,
            "lines": [
                {"text": "贷记卡账户", "bbox": [10, 10, 200, 20]},
                {"text": "账户 1：", "bbox": [10, 30, 100, 40]},
                {"text": "账户标识", "bbox": [10, 700, 100, 720]},
            ],
        },
        {
            "page": 2,
            "source_page": 2,
            "lines": [
                {"text": "还款记录", "bbox": [10, 20, 100, 40]},
                {"text": "查询记录", "bbox": [10, 300, 100, 320]},
            ],
        },
    ]
    denied = SimpleNamespace(
        corrected_evidence_pages=lambda: pages,
        allows_scanned_line_transition=lambda *_args: None,
    )
    verified = SimpleNamespace(
        corrected_evidence_pages=lambda: pages,
        allows_scanned_line_transition=lambda *_args: True,
    )

    denied_row = native_extraction._account_anchor_skeletons(denied)[0]
    verified_row = native_extraction._account_anchor_skeletons(verified)[0]

    assert [page["logical_page"] for page in denied_row["_canonical_segment"]["pages"]] == [1]
    assert [page["logical_page"] for page in verified_row["_canonical_segment"]["pages"]] == [1, 2]
    assert verified_row["_canonical_segment"]["pages"][1]["continuation_verified"] is True


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
    assert issue["observed_value"]["linked_month_count"] == 1
    assert issue["observed_value"]["linked_months"] == ["2024-01"]
    assert issue["observed_value"]["grid_id"] == "grid:range"
    assert issue["candidate_value"]["printed_month_count"] == 2
    assert issue["candidate_value"]["printed_months"] == ["2024-01", "2024-02"]
    assert "target_record_id" not in issue
