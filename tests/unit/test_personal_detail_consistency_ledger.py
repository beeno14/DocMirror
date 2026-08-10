from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.community_plugin import _CreditReportCommunityBundle
from docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger import (
    apply_document_consistency_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)


def _ref(row: int, column: int = 0) -> dict[str, object]:
    return {
        "source": "native_detail_table",
        "logical_page": 5,
        "source_page": 3,
        "table_id": "pt_5_1",
        "row": row,
        "column": column,
        "geometry_scope": "canonical_field_slot",
    }


def _context(*issues: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(_personal_detail_extraction_issues=list(issues))


def _profile_observation(value: str, row: int = 1) -> dict[str, object]:
    return {
        "value": value,
        "normalized_value": value,
        "raw": value,
        "source_refs": [_ref(row)],
        "observation_status": "observed",
        "confidence": 0.96,
    }


def _account(
    account_id: str,
    institution: str,
    row: int,
    *,
    confidence: float = 0.95,
) -> dict[str, object]:
    return {
        "record_id": account_id,
        "account_id": account_id,
        "account_type": "non_revolving_loan",
        "management_institution": institution,
        "source_refs_by_field": {"management_institution": [_ref(row)]},
        "confidence": confidence,
    }


def test_candidate_b_wires_consistency_after_final_correction_before_dataset_snapshot() -> None:
    source = Path(
        "docmirror/plugins/credit_report/personal_detail_scanned/candidate_b.py"
    ).read_text(encoding="utf-8")

    correction = source.index("corrected_payload = self.context.correct_candidate_b_datasets")
    consistency = source.index("consistency_audit = apply_document_consistency_ledger")
    dataset_snapshot = source.index("all_datasets: dict[str, list[dict[str, Any]]]", consistency)

    assert correction < consistency < dataset_snapshot
    assert '"personal_detail_document_consistency_ledger": consistency_audit' in source
    assert '"document_consistency": consistency_audit' in source
    assert 'consistency_input["personal_profile"] = [source_profile]' in source


def test_document_local_institution_conflict_reports_singleton_without_majority_correction() -> None:
    correct = "重庆市蚂蚁商诚小额贷款有限公司"
    singleton = "重庆市蚂蚊商诚小额贷款有限公司"
    datasets = {
        "credit_accounts": [
            _account("account:1", correct, 1),
            _account("account:2", correct, 2),
            _account("account:3", singleton, 3),
        ]
    }
    context = _context()

    audit = apply_document_consistency_ledger(context, datasets)

    assert datasets["credit_accounts"][2]["management_institution"] == singleton
    assert datasets["credit_accounts"][0]["management_institution"] == correct
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_document_local_institution_glyph_conflict"
    assert issue["target_record_id"] == "account:3"
    assert issue["field_name"] == "management_institution"
    assert "normalized_value_withheld" not in issue["reason_codes"]
    assert audit["institution_glyph_conflict_retained_with_issue"] == 1


def test_document_local_institution_conflict_withholds_only_materially_weaker_singleton() -> None:
    correct = "重庆市蚂蚁商诚小额贷款有限公司"
    singleton = "重庆市蚂蚊商诚小额贷款有限公司"
    datasets = {
        "credit_accounts": [
            _account("account:1", correct, 1, confidence=0.98),
            _account("account:2", correct, 2, confidence=0.96),
            _account("account:3", singleton, 3, confidence=0.60),
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    outlier = datasets["credit_accounts"][2]
    assert "management_institution" not in outlier
    assert outlier["canonical_raw"]["management_institution"] == singleton
    assert "management_institution" in outlier["_unresolved_fields"]
    assert "normalized_value_withheld" in context._personal_detail_extraction_issues[0]["reason_codes"]


def test_document_local_address_conflict_is_localized_but_directional_addresses_are_distinct() -> None:
    correct = "福建省福州市仓山区卢滨路中庚城19号楼704"
    singleton = "福建省福州市仓山区泸滨路中庚城19号楼704"
    east = "北京市朝阳区建国东路1号"
    west = "北京市朝阳区建国西路1号"
    datasets = {
        "residence_records": [
            {
                "record_id": "residence:1",
                "residence_record_id": "residence:1",
                "address": correct,
                "source_refs_by_field": {"address": [_ref(1)]},
            },
            {
                "record_id": "residence:2",
                "residence_record_id": "residence:2",
                "address": correct,
                "source_refs_by_field": {"address": [_ref(2)]},
            },
            {
                "record_id": "residence:3",
                "residence_record_id": "residence:3",
                "address": singleton,
                "source_refs_by_field": {"address": [_ref(3)]},
            },
            {
                "record_id": "residence:4",
                "residence_record_id": "residence:4",
                "address": east,
                "source_refs_by_field": {"address": [_ref(4)]},
            },
            {
                "record_id": "residence:5",
                "residence_record_id": "residence:5",
                "address": east,
                "source_refs_by_field": {"address": [_ref(5)]},
            },
            {
                "record_id": "residence:6",
                "residence_record_id": "residence:6",
                "address": west,
                "source_refs_by_field": {"address": [_ref(6)]},
            },
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    issues = context._personal_detail_extraction_issues
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "candidate_b_document_local_address_glyph_conflict"
    assert issues[0]["target_record_id"] == "residence:3"
    assert datasets["residence_records"][5]["address"] == west


def test_cross_format_profile_and_residence_street_glyph_conflict_is_not_silent() -> None:
    profile_address = "福建省福州市仓山区仓山镇仓山村委会卢滨路中庚城19座704"
    residence_address = "福建省福州市仓山区泸滨路中庚城19号楼704"
    datasets = {
        "personal_profile": [
            {
                "record_id": "personal_profile:1",
                "subject_profile_id": "personal_profile:1",
                "mailing_address": profile_address,
                "source_refs_by_field": {"mailing_address": [_ref(1)]},
            }
        ],
        "residence_records": [
            {
                "record_id": "residence:5",
                "residence_record_id": "residence:5",
                "address": residence_address,
                "source_refs_by_field": {"address": [_ref(5)]},
            }
        ],
    }
    context = _context()

    audit = apply_document_consistency_ledger(context, datasets)

    assert datasets["personal_profile"][0]["mailing_address"] == profile_address
    assert datasets["residence_records"][0]["address"] == residence_address
    assert datasets["personal_profile"][0]["_reported_field_conflicts"] == [
        "mailing_address"
    ]
    assert datasets["residence_records"][0]["_reported_field_conflicts"] == ["address"]
    issues = context._personal_detail_extraction_issues
    assert len(issues) == 2
    assert {issue["target_record_id"] for issue in issues} == {
        "personal_profile:1",
        "residence:5",
    }
    assert {
        (issue["target_record_id"], issue["field_name"])
        for issue in issues
    } == {
        ("personal_profile:1", "mailing_address"),
        ("residence:5", "address"),
    }
    assert all(
        issue["issue_code"] == "candidate_b_document_local_address_glyph_conflict"
        for issue in issues
    )
    assert audit["address_cross_format_glyph_conflict_retained_with_issue"] == 2


def test_candidate_shaped_nested_profile_and_residence_conflict_is_not_silent() -> None:
    profile_address = "福建省福州市仓山区仓山镇仓山村委会卢滨路中庚城19座704"
    residence_address = "福建省福州市仓山区泸滨路中庚城19号楼704"
    profile = {"mailing_address": _profile_observation(profile_address)}
    residence = {
        "record_id": "residence:5",
        "residence_record_id": "residence:5",
        "address": residence_address,
        "source_refs_by_field": {"address": [_ref(5)]},
    }
    context = _context()

    audit = apply_document_consistency_ledger(
        context,
        {"personal_profile": [profile], "residence_records": [residence]},
    )

    assert profile["mailing_address"]["normalized_value"] == profile_address
    assert residence["address"] == residence_address
    assert profile["_reported_field_conflicts"] == ["mailing_address"]
    assert residence["_reported_field_conflicts"] == ["address"]
    assert {
        (issue["target_dataset"], issue["target_record_id"], issue["field_name"])
        for issue in context._personal_detail_extraction_issues
    } == {
        ("personal_profile", "personal_profile:1", "mailing_address"),
        ("residence_records", "residence:5", "address"),
    }
    assert audit["address_cross_format_glyph_conflict_retained_with_issue"] == 2


def test_nested_profile_address_conflict_requires_independent_matching_location() -> None:
    profile_address = "福建省福州市仓山区仓山镇仓山村委会卢滨路中庚城19座704"
    residence_address = "福建省福州市仓山区泸滨路中庚城19号楼704"

    same_source_profile = {"mailing_address": _profile_observation(profile_address, row=1)}
    same_source_residence = {
        "record_id": "residence:same-source",
        "residence_record_id": "residence:same-source",
        "address": residence_address,
        "source_refs_by_field": {"address": [_ref(1)]},
    }
    same_source_context = _context()
    apply_document_consistency_ledger(
        same_source_context,
        {
            "personal_profile": [same_source_profile],
            "residence_records": [same_source_residence],
        },
    )

    different_unit_profile = {
        "mailing_address": _profile_observation(profile_address, row=1)
    }
    different_unit_residence = {
        "record_id": "residence:different-unit",
        "residence_record_id": "residence:different-unit",
        "address": "福建省福州市仓山区泸滨路中庚城19号楼705",
        "source_refs_by_field": {"address": [_ref(5)]},
    }
    different_unit_context = _context()
    apply_document_consistency_ledger(
        different_unit_context,
        {
            "personal_profile": [different_unit_profile],
            "residence_records": [different_unit_residence],
        },
    )

    assert not same_source_context._personal_detail_extraction_issues
    assert not different_unit_context._personal_detail_extraction_issues
    assert "_reported_field_conflicts" not in same_source_profile
    assert "_reported_field_conflicts" not in different_unit_profile


def test_nested_profile_conflict_targets_survive_v2_and_community_projection() -> None:
    profile_address = "福建省福州市仓山区仓山镇仓山村委会卢滨路中庚城19座704"
    residence_address = "福建省福州市仓山区泸滨路中庚城19号楼704"
    profile = {"mailing_address": _profile_observation(profile_address)}
    residence = {
        "record_id": "residence:5",
        "residence_record_id": "residence:5",
        "address": residence_address,
        "information_updated_date": "2024-01-02",
        "canonical_raw": {
            "address": residence_address,
            "information_updated_date": "2024-01-02",
        },
        "source_refs_by_field": {
            "address": [_ref(5)],
            "information_updated_date": [_ref(6)],
        },
    }
    context = _context()
    apply_document_consistency_ledger(
        context,
        {"personal_profile": [profile], "residence_records": [residence]},
    )
    content = prepare_personal_detail_source_collections(
        {
            "facts": {"subject_profile": profile},
            "datasets": {
                "residence_records": [residence],
                "personal_detail_extraction_issues": list(
                    context._personal_detail_extraction_issues
                ),
            },
        },
        final_dataset_counts={"residence_records": 1},
    )
    v2_datasets = project_personal_detail_datasets(content["datasets"])
    v2_issues = [row.get("normalized", row) for row in v2_datasets["extraction_issues"]]

    assert v2_datasets["subject_profile"][0]["record_id"] == "personal_profile:1"
    assert v2_datasets["subject_residences"][0]["record_id"] == "residence:5"
    assert {
        (issue["target_dataset"], issue["target_record_id"], issue["field_name"])
        for issue in v2_issues
    } == {
        ("subject_profile", "personal_profile:1", "mailing_address"),
        ("subject_residences", "residence:5", "address"),
    }
    assert all(
        issue["target_dataset"] not in {"personal_profile", "residence_records"}
        for issue in v2_issues
    )

    semantic = personal_detail_semantic_extensions()
    facts = {
        **content["facts"],
        **{
            f"personal_detail_v2_expected_{name}_count": len(rows)
            for name, rows in v2_datasets.items()
        },
    }
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            "report_subtype": "personal_detail",
            "content_mode": "scanned_ocr",
            "data_dictionary": personal_detail_data_dictionary(),
            **facts,
        },
        "semantic": semantic,
        "datasets": v2_datasets,
        "sections": [],
    }
    parse_result = ParseResult(
        entities=DocumentEntities(document_type="personal_credit_report_detailed"),
        pages=[PageContent(page_number=1)],
    )
    projected = project_community_bundle(
        seal_parse_result(parse_result),
        file_path=str(Path(__file__)),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    )
    payload = _CreditReportCommunityBundle(
        schema=projected.schema,
        document=projected.document,
        sections=projected.sections,
        datasets=projected.datasets,
        files=projected.files,
        warnings=projected.warnings,
        result=projected.result,
        source_fingerprint=projected.source_fingerprint,
        parse_result_schema=projected.parse_result_schema,
        classification=projected.classification,
        domain=projected.domain,
        diagnostics=projected.diagnostics,
        content_markdown_override=projected.content_markdown_override,
    ).json_payload()
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    profile_row = datasets["subject_profile"]["rows"][0]
    residence_row = datasets["subject_residences"]["rows"][0]
    issue_rows = datasets["extraction_issues"]["rows"]

    assert {
        (
            row["normalized"]["target_dataset"],
            row["normalized"]["target_record_id"],
            row["normalized"]["field_name"],
        )
        for row in issue_rows
    } == {
        ("subject_profile", "personal_profile:1", "mailing_address"),
        ("subject_residences", "residence:5", "address"),
    }
    assert set(profile_row["canonical_raw"]) == {"mailing_address"}
    assert set(profile_row["raw"]) == {"mailing_address"}
    assert set(residence_row["canonical_raw"]) == {"address"}
    assert set(residence_row["raw"]) == {"address"}
    assert residence_row["normalized"]["information_updated_date"] == "2024-01-02"
    for name, expected in (
        ("subject_profile", 1),
        ("subject_residences", 1),
        ("extraction_issues", 2),
    ):
        dataset = datasets[name]
        assert dataset["row_count"] == expected
        assert dataset["completeness"]["expected_row_count"] == expected
        assert dataset["completeness"]["emitted_row_count"] == expected
        assert dataset["completeness"]["omitted_row_count"] == 0
    assert not {
        warning["code"]
        for warning in payload["warnings"]
        if warning["code"]
        in {"DATASET_COMPLETENESS_UNVERIFIED", "DATASET_ROW_COUNT_MISMATCH"}
    }


def test_cross_format_address_similarity_with_different_unit_numbers_is_not_a_conflict() -> None:
    datasets = {
        "personal_profile": [
            {
                "record_id": "personal_profile:1",
                "subject_profile_id": "personal_profile:1",
                "mailing_address": "福建省福州市仓山区卢滨路中庚城19座704",
                "source_refs_by_field": {"mailing_address": [_ref(1)]},
            }
        ],
        "residence_records": [
            {
                "record_id": "residence:5",
                "residence_record_id": "residence:5",
                "address": "福建省福州市仓山区泸滨路中庚城19号楼705",
                "source_refs_by_field": {"address": [_ref(5)]},
            }
        ],
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    assert not context._personal_detail_extraction_issues


def test_legitimate_one_glyph_distinct_organizations_are_not_conflated() -> None:
    datasets = {
        "credit_accounts": [
            _account("account:1", "上海银行股份有限公司", 1),
            _account("account:2", "上海银行股份有限公司", 2),
            _account("account:3", "上饶银行股份有限公司", 3),
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    assert not context._personal_detail_extraction_issues
    assert datasets["credit_accounts"][2]["management_institution"] == "上饶银行股份有限公司"


def test_rootless_branch_fragment_is_withheld_but_complete_branch_and_legal_center_survive() -> None:
    datasets = {
        "credit_lines": [
            {
                "record_id": "line:1",
                "credit_line_id": "line:1",
                "institution": "福建自贸试验区福州片区分行",
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "line:2",
                "credit_line_id": "line:2",
                "institution": "中国建设银行股份有限公司福建自贸试验区福州片区分行",
                "source_refs_by_field": {"institution": [_ref(2)]},
            },
            {
                "record_id": "line:3",
                "credit_line_id": "line:3",
                "institution": "福州市住房公积金管理中心",
                "source_refs_by_field": {"institution": [_ref(3)]},
            },
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    assert "institution" not in datasets["credit_lines"][0]
    assert datasets["credit_lines"][1]["institution"].startswith("中国建设银行")
    assert datasets["credit_lines"][2]["institution"] == "福州市住房公积金管理中心"
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_institution_branch_without_legal_root"
    assert issue["target_record_id"] == "line:1"


def test_generic_standalone_center_is_not_treated_as_a_branch_fragment() -> None:
    datasets = {
        "credit_lines": [
            {
                "record_id": "line:center",
                "credit_line_id": "line:center",
                "institution": "福州市不动产登记中心",
                "source_refs_by_field": {"institution": [_ref(1)]},
            }
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    assert datasets["credit_lines"][0]["institution"] == "福州市不动产登记中心"
    assert not context._personal_detail_extraction_issues


def test_separated_inquiry_prefix_is_resolved_only_with_two_document_local_root_witnesses() -> None:
    target_id = "inquiry:1"
    stale = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="institution unreadable",
        target_dataset="inquiry_records",
        target_record_id=target_id,
        field_name="institution",
        observed_value="福 中国建设银行股份有限公司北京市分行",
        reason_codes=("normalized_value_withheld",),
    )
    datasets = {
        "inquiry_records": [
            {
                "record_id": target_id,
                "inquiry_id": target_id,
                "institution": None,
                "canonical_raw": {"institution": "福 中国建设银行股份有限公司北京市分行"},
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "inquiry:2",
                "inquiry_id": "inquiry:2",
                "institution": "中国建设银行股份有限公司福州城东支行",
                "source_refs_by_field": {"institution": [_ref(2)]},
            },
        ],
        "credit_accounts": [
            {
                "record_id": "account:1",
                "account_id": "account:1",
                "account_type": "credit_card",
                "management_institution": (
                    "中国建设银行股份有限公司福建自贸试验区福州片区分行"
                ),
                "canonical_raw": {
                    "management_institution": "中国建设银行 股份有限公司 福建自贸试验 区福州片区分 行"
                },
                "source_refs_by_field": {"management_institution": [_ref(3)]},
            }
        ],
    }
    context = _context(stale)

    audit = apply_document_consistency_ledger(context, datasets)

    target = datasets["inquiry_records"][0]
    assert target["institution"] == "中国建设银行股份有限公司北京市分行"
    assert all(issue["issue_code"] != "pboc_cell_contract_unresolved" for issue in context._personal_detail_extraction_issues)
    resolved = context._personal_detail_extraction_issues[0]
    assert resolved["issue_code"] == "candidate_b_document_local_institution_prefix_resolved"
    assert resolved["status"] == "resolved"
    assert resolved["severity"] == "info"
    assert "normalized_value_withheld" not in resolved.get("reason_codes", ())
    assert audit["institution_prefix_resolved"] == 1


def test_separated_inquiry_prefix_without_two_witnesses_gets_localized_extraction_issue() -> None:
    datasets = {
        "inquiry_records": [
            {
                "record_id": "inquiry:1",
                "inquiry_id": "inquiry:1",
                "institution": None,
                "canonical_raw": {"institution": "福 中国建设银行股份有限公司北京市分行"},
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "inquiry:2",
                "inquiry_id": "inquiry:2",
                "institution": "中国建设银行股份有限公司福州城东支行",
                "source_refs_by_field": {"institution": [_ref(2)]},
            },
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    assert datasets["inquiry_records"][0].get("institution") is None
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_document_local_institution_prefix_unresolved"
    assert issue["target_record_id"] == "inquiry:1"
    assert "normalized_value_withheld" in issue["reason_codes"]


def test_separated_prefix_does_not_count_cross_plane_source_cell_aliases_as_two_witnesses() -> None:
    target_id = "inquiry:target"
    duplicated_ref = _ref(2)
    alias_ref = dict(duplicated_ref)
    alias_ref.pop("source_page")
    alias_ref["source"] = "candidate_b_corrected_table"
    datasets = {
        "inquiry_records": [
            {
                "record_id": target_id,
                "inquiry_id": target_id,
                "institution": None,
                "canonical_raw": {
                    "institution": "福 中国建设银行股份有限公司北京市分行"
                },
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "inquiry:witness",
                "inquiry_id": "inquiry:witness",
                "institution": "中国建设银行股份有限公司福州城东支行",
                "source_refs_by_field": {"institution": [duplicated_ref]},
            },
        ],
        "credit_accounts": [
            {
                "record_id": "account:alias-of-witness",
                "account_id": "account:alias-of-witness",
                "account_type": "credit_card",
                "management_institution": "中国建设银行股份有限公司福州城东支行",
                "source_refs_by_field": {"management_institution": [alias_ref]},
            }
        ],
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    target = datasets["inquiry_records"][0]
    assert target.get("institution") is None
    assert [issue["issue_code"] for issue in context._personal_detail_extraction_issues] == [
        "candidate_b_document_local_institution_prefix_unresolved"
    ]
    assert context._personal_detail_extraction_issues[0]["target_record_id"] == target_id


def test_separated_prefix_does_not_count_unresolved_raw_fields_as_clean_witnesses() -> None:
    target_id = "inquiry:target"
    datasets = {
        "inquiry_records": [
            {
                "record_id": target_id,
                "inquiry_id": target_id,
                "institution": None,
                "canonical_raw": {
                    "institution": "福 中国建设银行股份有限公司北京市分行"
                },
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "inquiry:unresolved-1",
                "inquiry_id": "inquiry:unresolved-1",
                "institution": None,
                "canonical_raw": {
                    "institution": "中国建设银行股份有限公司福州城东支行"
                },
                "_unresolved_fields": ["institution"],
                "extraction_status": "review",
                "source_refs_by_field": {"institution": [_ref(2)]},
            },
            {
                "record_id": "inquiry:unresolved-2",
                "inquiry_id": "inquiry:unresolved-2",
                "institution": None,
                "canonical_raw": {
                    "institution": "中国建设银行股份有限公司上海市分行"
                },
                "_unresolved_fields": ["institution"],
                "extraction_status": "review",
                "source_refs_by_field": {"institution": [_ref(3)]},
            },
        ]
    }
    context = _context()

    apply_document_consistency_ledger(context, datasets)

    target = datasets["inquiry_records"][0]
    assert target.get("institution") is None
    assert [issue["issue_code"] for issue in context._personal_detail_extraction_issues] == [
        "candidate_b_document_local_institution_prefix_unresolved"
    ]
    assert context._personal_detail_extraction_issues[0]["target_record_id"] == target_id


def test_summary_account_count_over_document_total_is_withheld_not_corrected() -> None:
    accounts = [
        {
            "record_id": f"account:{index}",
            "account_id": f"account:{index}",
            "account_type": "non_revolving_loan" if index <= 22 else "credit_card",
        }
        for index in range(1, 46)
    ]
    bad_cell = {
        "record_id": "summary-cell:count",
        "summary_cell_id": "summary-cell:count",
        "summary_type": "非循环贷账户",
        "title": "非循环贷账户信息汇总",
        "column_label": "账户数",
        "value": "50",
        "source_refs_by_field": {"value": [_ref(1, 1)]},
    }
    family_only_conflict = {
        "record_id": "summary-cell:family",
        "summary_cell_id": "summary-cell:family",
        "summary_type": "非循环贷账户",
        "title": "非循环贷账户信息汇总",
        "column_label": "账户数",
        "value": "23",
        "source_refs_by_field": {"value": [_ref(2, 1)]},
    }
    datasets = {
        "credit_accounts": accounts,
        "personal_detail_summary_cells": [bad_cell, family_only_conflict],
    }
    context = _context()

    audit = apply_document_consistency_ledger(context, datasets)

    assert bad_cell["value"] == "50"
    assert bad_cell["value_status"] == "unreadable"
    assert family_only_conflict["value"] == "23"
    assert family_only_conflict.get("value_status") is None
    issues = {issue["target_record_id"]: issue for issue in context._personal_detail_extraction_issues}
    assert issues["summary-cell:count"]["issue_code"] == (
        "candidate_b_summary_account_count_exceeds_document_population"
    )
    assert "normalized_value_withheld" in issues["summary-cell:count"]["reason_codes"]
    assert issues["summary-cell:family"]["issue_code"] == (
        "candidate_b_summary_account_count_exceeds_family_population"
    )
    assert "normalized_value_withheld" not in issues["summary-cell:family"]["reason_codes"]
    assert audit["summary_count_withheld"] == 1
    assert audit["summary_count_retained_with_issue"] == 1
