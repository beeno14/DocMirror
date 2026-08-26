from __future__ import annotations

from copy import deepcopy
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
from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
    link_candidate_b_repayments,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


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


def test_separated_inquiry_whitespace_is_resolved_only_with_two_exact_glyph_witnesses() -> None:
    target_id = "inquiry:1"
    stale = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="institution unreadable",
        target_dataset="inquiry_records",
        target_record_id=target_id,
        field_name="institution",
        observed_value="中 国建设银行股份有限公司北京市分行",
        reason_codes=("normalized_value_withheld",),
    )
    datasets = {
        "inquiry_records": [
            {
                "record_id": target_id,
                "inquiry_id": target_id,
                "institution": None,
                "canonical_raw": {"institution": "中 国建设银行股份有限公司北京市分行"},
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "inquiry:2",
                "inquiry_id": "inquiry:2",
                "institution": "中国建设银行股份有限公司北京市分行",
                "source_refs_by_field": {"institution": [_ref(2)]},
            },
        ],
        "credit_accounts": [
            {
                "record_id": "account:1",
                "account_id": "account:1",
                "account_type": "credit_card",
                "management_institution": "中国建设银行股份有限公司北京市分行",
                "canonical_raw": {
                    "management_institution": "中国建设银行 股份有限公司 北京市分 行"
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
    assert "complete_non_whitespace_glyph_sequence_preserved" in resolved["reason_codes"]
    assert resolved["candidate_value"]["normalized_institution"] == (
        "中国建设银行股份有限公司北京市分行"
    )
    assert audit["institution_prefix_resolved"] == 1


def test_repeated_legal_root_never_deletes_an_individualized_leading_han_glyph() -> None:
    raw = "福 中信银行股份有限公司"
    witness = "中信银行股份有限公司"
    datasets = {
        "inquiry_records": [
            {
                "record_id": "inquiry:target",
                "inquiry_id": "inquiry:target",
                "institution": None,
                "canonical_raw": {"institution": raw},
                "source_refs_by_field": {"institution": [_ref(1)]},
            },
            {
                "record_id": "inquiry:witness-1",
                "inquiry_id": "inquiry:witness-1",
                "institution": witness,
                "source_refs_by_field": {"institution": [_ref(2)]},
            },
        ],
        "credit_accounts": [
            _account("account:witness-2", witness, 3),
        ],
    }
    context = _context()

    audit = apply_document_consistency_ledger(context, datasets)

    target = datasets["inquiry_records"][0]
    assert target.get("institution") is None
    assert target["canonical_raw"]["institution"] == raw
    assert "institution" in target["_unresolved_fields"]
    assert audit.get("institution_prefix_resolved", 0) == 0
    assert audit["institution_prefix_unresolved"] == 1
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "candidate_b_document_local_institution_prefix_unresolved"
    assert issue["candidate_value"]["withheld_complete_value"] == "福中信银行股份有限公司"
    assert issue["candidate_value"]["independent_witness_count"] == 0
    assert "legal_root_repetition_not_correction_proof" in issue["reason_codes"]


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


def test_source_projection_account_month_ledger_is_set_based_and_fail_closed() -> None:
    def proof(grid_id: str, month: str) -> dict[str, object]:
        return {
            "account_id": "account:1",
            "performance_month": month,
            "grid_id": grid_id,
            "owner_basis": "canonical_account_segment",
            "account_anchor_exact": True,
            "printed_month_range_exact": True,
            "grid_geometry_exact": True,
            "unique_owner": True,
        }

    owner_hash = stable_record_id(
        "source_account_month_owner", "account:1"
    ).split(":", 1)[-1]
    missing = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_account_range_missing_month",
        message="Exact printed account range month was withheld.",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id=f"source_account_month:{owner_hash}:2024-02",
        field_name="performance_month",
        observed_value={
            "account_id": "account:1",
            "performance_month": "2024-02",
            "source_identity_type": "account_month_from_printed_repayment_range",
        },
        source_refs=(
            {
                "logical_page": 3,
                "bbox": [20, 40, 280, 52],
                "performance_month": "2024-02",
                "account_id": "account:1",
                "geometry_scope": "line",
                "binding": "source_account_month_range",
            },
        ),
    )
    unresolved = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_grid_owner_unresolved_field",
        message="Printed grid month has no unique account owner.",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="grid:unowned:2024-03",
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:unowned",
            "performance_month": "2024-03",
        },
        source_refs=(
            {
                "logical_page": 4,
                "grid_id": "grid:unowned",
                "performance_month": "2024-03",
                "bbox": [40, 80, 180, 120],
                "geometry_scope": "grid",
            },
        ),
    )
    reconciled_alias = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="A second exact source position maps to the represented identity.",
        severity="info",
        status="informational",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="source_account_month:any:2024-01",
        field_name="performance_month",
        observed_value={
            "account_id": "account:1",
            "grid_id": "grid:alias-c",
            "performance_month": "2024-01",
            "source_position_state": "owner_bound_alias",
        },
        source_refs=(
            {
                "logical_page": 5,
                "grid_id": "grid:alias-c",
                "performance_month": "2024-01",
                "bbox": [50, 90, 190, 130],
                "geometry_scope": "grid",
            },
        ),
    )
    alias_pending_owner_reconciliation = make_issue(
        category="ocr_structure_correction",
        issue_code="canonical_monthly_source_structure_missing_field",
        message="The detached source position is awaiting owner reconciliation.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        target_record_id="grid:alias-c:2024-01",
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:alias-c",
            "performance_month": "2024-01",
        },
        source_refs=reconciled_alias["source_refs"],
    )
    content = {
        "facts": {},
        "datasets": {
            "repayment_records": [
                {
                    "record_id": f"{grid_id}:2024-01",
                    "repayment_id": f"{grid_id}:2024-01",
                    "grid_id": grid_id,
                    "account_id": "account:1",
                    "performance_month": "2024-01",
                    "status": "N",
                    "_account_month_identity_proof": proof(grid_id, "2024-01"),
                }
                for grid_id in ("grid:alias-a", "grid:alias-b")
            ],
            "personal_detail_extraction_issues": [
                missing,
                dict(missing),
                unresolved,
                alias_pending_owner_reconciliation,
                reconciled_alias,
            ],
        },
    }

    prepared = prepare_personal_detail_source_collections(content)

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["identity_fields"] == ["account_id", "performance_month"]
    assert closure["candidate_identity_count"] == 1
    assert closure["localized_missing_identity_count"] == 1
    assert closure["expected_identity_count"] == 2
    assert closure["raw_source_month_positions"] == 4
    assert closure["owner_bound_account_months"] == 3
    assert closure["owner_unresolved_positions"] == 1
    assert closure["alias_source_month_positions"] == 2
    assert closure["source_localized_owner_unresolved_positions"] == 1
    assert closure["unlocalized_owner_unresolved_positions"] == 0
    assert closure["source_position_balance_valid"] is True
    assert closure["unresolved_source_position_count"] == 1
    assert closure["status"] == "partial_owner_unresolved"
    assert len(closure["identity_sha256"]) == 64
    state = prepared["facts"]["personal_detail_dataset_states"][
        "repayment_records"
    ]
    assert state["observed_row_count"] == 1
    assert state["expected_row_count"] == 2


def test_source_projection_deduplicates_evidence_alias_across_geometry_variants() -> None:
    performance_month = "2024-01"
    physical_evidence_id = "native:monthly:7:4:1"

    def physical_ref(
        grid_id: str,
        *,
        bbox: list[float],
        extra_evidence_id: str,
    ) -> dict[str, object]:
        return {
            "logical_page": 7,
            "grid_id": grid_id,
            "performance_month": performance_month,
            "row": 4,
            "column": 1,
            "bbox": bbox,
            "geometry_scope": "cell",
            "evidence_ids": [physical_evidence_id, extra_evidence_id],
        }

    proof = {
        "account_id": "account:1",
        "performance_month": performance_month,
        "grid_id": "grid:canonical",
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    alias = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="The physical month is an exact alias of the canonical identity.",
        severity="info",
        status="informational",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="source_account_month:any:2024-01",
        field_name="performance_month",
        observed_value={
            "account_id": "account:1",
            "grid_id": "grid:alias",
            "performance_month": performance_month,
            "source_position_state": "owner_bound_alias",
        },
        source_refs=(
            physical_ref(
                "grid:alias",
                bbox=[121.0, 300.0, 146.0, 314.0],
                extra_evidence_id="native:monthly:alias-refinement",
            ),
        ),
    )
    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    {
                        "record_id": "grid:canonical:2024-01",
                        "repayment_id": "grid:canonical:2024-01",
                        "grid_id": "grid:canonical",
                        "account_id": "account:1",
                        "performance_month": performance_month,
                        "status": "N",
                        "_account_month_identity_proof": proof,
                        "_account_month_identity_proof_status": "exact",
                        "source_cell_refs": [
                            physical_ref(
                                "grid:canonical",
                                bbox=[120.0, 300.0, 145.0, 314.0],
                                extra_evidence_id=(
                                    "native:monthly:canonical-refinement"
                                ),
                            )
                        ],
                    }
                ],
                "personal_detail_extraction_issues": [alias],
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["expected_identity_count"] == 1
    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 0
    assert closure["alias_source_month_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 1
    assert closure["source_position_balance_valid"] is True


def test_source_projection_deduplicates_exact_geometry_with_distinct_evidence() -> None:
    performance_month = "2024-01"

    def physical_ref(grid_id: str, evidence_id: str) -> dict[str, object]:
        return {
            "logical_page": 7,
            "grid_id": grid_id,
            "performance_month": performance_month,
            "row": 4,
            "column": 1,
            "bbox": [120.0, 300.0, 145.0, 314.0],
            "geometry_scope": "cell",
            "evidence_ids": [evidence_id],
        }

    proof = {
        "account_id": "account:1",
        "performance_month": performance_month,
        "grid_id": "grid:canonical",
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    alias = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="The physical month is an exact alias of the canonical identity.",
        severity="info",
        status="informational",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="source_account_month:any:2024-01",
        field_name="performance_month",
        observed_value={
            "account_id": "account:1",
            "grid_id": "grid:alias",
            "performance_month": performance_month,
            "source_position_state": "owner_bound_alias",
        },
        source_refs=(physical_ref("grid:alias", "native:monthly:alias"),),
    )
    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    {
                        "record_id": "grid:canonical:2024-01",
                        "repayment_id": "grid:canonical:2024-01",
                        "grid_id": "grid:canonical",
                        "account_id": "account:1",
                        "performance_month": performance_month,
                        "status": "N",
                        "_account_month_identity_proof": proof,
                        "_account_month_identity_proof_status": "exact",
                        "source_cell_refs": [
                            physical_ref(
                                "grid:canonical", "native:monthly:canonical"
                            )
                        ],
                    }
                ],
                "personal_detail_extraction_issues": [alias],
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["source_month_position_observations"] == 2
    assert closure["unique_physical_source_month_positions"] == 1
    assert closure["canonical_account_month_identity_count"] == 1
    assert closure["physical_alias_source_month_observations"] == 1


def test_source_projection_reconciles_early_grid_alias_after_final_cell_calibration() -> None:
    performance_month = "2024-01"
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    account = {
        "account_id": "account:1",
        "_canonical_segment": {
            "ownership_basis": "printed_anchor_to_next_anchor",
            "anchor_logical_page": 4,
            "anchor_bbox": [10.0, 20.0, 90.0, 30.0],
            "pages": [
                {"logical_page": 4, "min_y": 20.0, "max_y": 300.0}
            ],
        },
    }
    source_table_geometry = {
        "source": "source_table_geometry",
        "coordinate_system": "pdf_points_top_left",
        "logical_page": 4,
        "table_id": "pt_4_1",
        "calibrated_from_source_table_geometry": True,
        "active_cell_geometry_exact": True,
        "value_inputs_used": False,
        "status_row_index": 6,
        "amount_row_index": 7,
        "year_anchor_row_index": 6,
    }
    date_range = {
        "start_year": 2024,
        "start_month": 1,
        "end_year": 2024,
        "end_month": 1,
    }
    grid_contract = {
        "page": 4,
        "coordinate_system": "pdf_points_top_left",
        "audit": {
            "date_range": date_range,
            "visual_month_geometry_by_page": {"4": source_table_geometry},
        },
        "col_bands": [
            {"index": 1, "header": "1", "bbox": [100.0, 140.0, 125.0, 220.0]}
        ],
    }
    grids = [
        {
            **grid_contract,
            "grid_id": "grid:canonical",
            "bbox": [90.0, 130.0, 400.0, 220.0],
        },
        {
            **grid_contract,
            "grid_id": "grid:early-alias",
            "bbox": [80.0, 135.0, 410.0, 225.0],
        },
    ]
    linked = link_candidate_b_repayments(
        [
            {"grid_id": grid["grid_id"], "year": 2024, "month": 1, "status": "N"}
            for grid in grids
        ],
        [account],
        grids,
        issue_context=context,
    )
    assert len(linked) == 1
    alias_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_monthly_source_position_alias_reconciled"
    ]
    assert len(alias_issues) == 1
    assert alias_issues[0]["observed_value"]["performance_month"] == performance_month
    alias_ref = alias_issues[0]["source_refs"][0]
    assert alias_ref["source"] == "candidate_b_monthly_grid_omission"
    assert alias_ref["geometry_scope"] == "grid"
    assert alias_ref["table_id"] == "pt_4_1"
    assert not alias_ref.get("evidence_ids")

    # Final correction calibrates the retained record after the relationship
    # stage has already frozen its informational alias issue.
    linked[0]["source_cell_refs"] = [
        {
            "page": 4,
            "logical_page": 4,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": linked[0]["grid_id"],
            "row": 4,
            "col": 1,
            "field_name": "status",
            "geometry_provenance": source_table_geometry,
            "bbox": [100.0, 160.0, 125.0, 175.0],
        }
    ]
    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": linked,
                "personal_detail_extraction_issues": (
                    context._personal_detail_extraction_issues
                ),
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 1
    assert closure["unique_physical_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 1
    assert closure["alias_source_month_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 1


def test_source_projection_calibrated_grid_alias_requires_consistent_source_page() -> None:
    account_id = "account:1"
    performance_month = "2024-01"

    def prepared_for(
        *,
        alias_source_page: int,
        cell_source_page: int,
    ) -> dict[str, object]:
        proof = {
            "account_id": account_id,
            "performance_month": performance_month,
            "grid_id": "grid:canonical",
            "owner_basis": "canonical_account_segment",
            "account_anchor_exact": True,
            "printed_month_range_exact": True,
            "grid_geometry_exact": True,
            "unique_owner": True,
        }
        calibrated_ref = {
            "page": 7,
            "logical_page": 7,
            "source_page": cell_source_page,
            "grid_id": "grid:canonical",
            "performance_month": performance_month,
            "row": 4,
            "col": 1,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "bbox": [100.0, 160.0, 125.0, 175.0],
            "evidence_ids": ["evidence:canonical"],
            "geometry_provenance": {
                "source": "source_table_geometry",
                "coordinate_system": "pdf_points_top_left",
                "logical_page": 7,
                "source_page": cell_source_page,
                "table_id": "table:shared",
                "calibrated_from_source_table_geometry": True,
                "active_cell_geometry_exact": True,
                "value_inputs_used": False,
            },
        }
        alias_ref = {
            "source": "candidate_b_monthly_grid_omission",
            "binding": "source_account_month_alias",
            "binding_quality": "source_account_month_alias",
            "page": 7,
            "logical_page": 7,
            "source_page": alias_source_page,
            "grid_id": "grid:alias",
            "performance_month": performance_month,
            "geometry_scope": "grid",
            "coordinate_system": "pdf_points_top_left",
            "table_id": "table:shared",
            "bbox": [80.0, 135.0, 410.0, 225.0],
            "evidence_ids": ["evidence:alias"],
        }
        alias_issue = make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_monthly_source_position_alias_reconciled",
            message="The grid is an exact alias only on one physical source page.",
            severity="info",
            status="informational",
            parser_stage="candidate_b_relationship_schema",
            target_dataset="repayment_records",
            target_record_id="source_account_month:any:2024-01",
            field_name="performance_month",
            observed_value={
                "account_id": account_id,
                "grid_id": "grid:alias",
                "performance_month": performance_month,
                "source_position_state": "owner_bound_alias",
            },
            source_refs=(alias_ref,),
        )
        return prepare_personal_detail_source_collections(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": [
                        {
                            "record_id": "grid:canonical:2024-01",
                            "repayment_id": "grid:canonical:2024-01",
                            "grid_id": "grid:canonical",
                            "account_id": account_id,
                            "performance_month": performance_month,
                            "status": "N",
                            "_account_month_identity_proof": proof,
                            "_account_month_identity_proof_status": "exact",
                            "source_cell_refs": [calibrated_ref],
                        }
                    ],
                    "personal_detail_extraction_issues": [alias_issue],
                },
            }
        )

    same_source = prepared_for(alias_source_page=4, cell_source_page=4)["facts"][
        "personal_detail_account_month_closure"
    ]
    conflicting_source = prepared_for(
        alias_source_page=4,
        cell_source_page=5,
    )["facts"]["personal_detail_account_month_closure"]

    assert same_source["source_month_position_observations"] == 2
    assert same_source["raw_source_month_positions"] == 1
    assert same_source["owner_bound_account_months"] == 1
    assert same_source["physical_alias_source_month_observations"] == 1
    assert conflicting_source["source_month_position_observations"] == 2
    assert conflicting_source["raw_source_month_positions"] == 2
    assert conflicting_source["owner_bound_account_months"] == 2
    assert conflicting_source["physical_alias_source_month_observations"] == 0


def test_source_projection_rejects_cross_account_physical_alias_reuse() -> None:
    performance_month = "2024-01"

    def record(account_id: str, grid_id: str) -> dict[str, object]:
        return {
            "record_id": f"{grid_id}:{performance_month}",
            "repayment_id": f"{grid_id}:{performance_month}",
            "grid_id": grid_id,
            "account_id": account_id,
            "performance_month": performance_month,
            "status": "N",
            "_account_month_identity_proof_status": "exact",
            "_account_month_identity_proof": {
                "account_id": account_id,
                "performance_month": performance_month,
                "grid_id": grid_id,
                "owner_basis": "canonical_account_segment",
                "account_anchor_exact": True,
                "printed_month_range_exact": True,
                "grid_geometry_exact": True,
                "unique_owner": True,
            },
            "source_cell_refs": [
                {
                    "page": 7,
                    "logical_page": 7,
                    "source_page": 4,
                    "grid_id": grid_id,
                    "performance_month": performance_month,
                    "row": 4,
                    "col": 1,
                    "geometry_scope": "cell",
                    "coordinate_system": "pdf_points_top_left",
                    "bbox": [100.0, 160.0, 125.0, 175.0],
                    "evidence_ids": ["shared-physical-month-cell"],
                    "geometry_provenance": {
                        "source": "source_table_geometry",
                        "coordinate_system": "pdf_points_top_left",
                        "logical_page": 7,
                        "source_page": 4,
                        "table_id": "table:shared",
                        "calibrated_from_source_table_geometry": True,
                        "active_cell_geometry_exact": True,
                        "value_inputs_used": False,
                    },
                }
            ],
        }

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    record("account:A", "grid:A"),
                    record("account:B", "grid:B"),
                ]
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["candidate_identity_count"] == 2
    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 2
    assert closure["physical_alias_source_month_observations"] == 0
    assert closure["cross_owner_physical_conflict_count"] == 1
    assert closure["physical_owner_conflict_free"] is False
    assert closure["status"] == "physical_owner_conflict"
    assert "_personal_detail_account_month_closure_proof" not in prepared["datasets"]
    assert prepared["facts"]["personal_detail_dataset_states"]["repayment_records"] == {
        "presence_status": "partial",
        "reason": "account_month_physical_owner_conflict",
        "observed_row_count": 2,
        "expected_row_count": 2,
    }
    conflicts = [
        issue
        for issue in prepared["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "account_month_physical_owner_conflict"
    ]
    assert len(conflicts) == 1
    assert conflicts[0]["field_name"] == "account_id"
    assert len(conflicts[0]["source_refs"]) == 2


def test_source_projection_rejects_malformed_calibrated_lattice_identities() -> None:
    account_id = "account:1"
    performance_month = "2024-01"

    def record(
        grid_id: str,
        x: float,
        *,
        ref_table_id: object,
        provenance_table_id: object,
        page: object = 7,
        logical_page: object = 7,
        provenance_logical_page: object = 7,
        source_page: object = 4,
        provenance_source_page: object | None = None,
        provenance_page: object | None = None,
        include_bbox: bool = True,
    ) -> dict[str, object]:
        ref: dict[str, object] = {
            "page": page,
            "logical_page": logical_page,
            "source_page": source_page,
            "grid_id": grid_id,
            "performance_month": performance_month,
            "row": 4,
            "col": 1,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "evidence_ids": [f"evidence:{grid_id}"],
            "geometry_provenance": {
                "source": "source_table_geometry",
                "coordinate_system": "pdf_points_top_left",
                "logical_page": provenance_logical_page,
                "table_id": provenance_table_id,
                "calibrated_from_source_table_geometry": True,
                "active_cell_geometry_exact": True,
                "value_inputs_used": False,
                **(
                    {"source_page": provenance_source_page}
                    if provenance_source_page is not None
                    else {}
                ),
                **(
                    {"page": provenance_page}
                    if provenance_page is not None
                    else {}
                ),
            },
        }
        if ref_table_id is not None:
            ref["table_id"] = ref_table_id
        if include_bbox:
            ref["bbox"] = [x, 10.0, x + 10.0, 20.0]
        return {
            "record_id": f"{grid_id}:{performance_month}",
            "repayment_id": f"{grid_id}:{performance_month}",
            "grid_id": grid_id,
            "account_id": account_id,
            "performance_month": performance_month,
            "status": "N",
            "_account_month_identity_proof_status": "exact",
            "_account_month_identity_proof": {
                "account_id": account_id,
                "performance_month": performance_month,
                "grid_id": grid_id,
                "owner_basis": "canonical_account_segment",
                "account_anchor_exact": True,
                "printed_month_range_exact": True,
                "grid_geometry_exact": True,
                "unique_owner": True,
            },
            "source_cell_refs": [ref],
        }

    cases = (
        (
            record(
                "grid:table-conflict",
                10.0,
                ref_table_id="table:A",
                provenance_table_id="table:B",
            ),
            record(
                "grid:table-clean",
                110.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
            ),
        ),
        (
            record(
                "grid:boolean-table",
                10.0,
                ref_table_id=None,
                provenance_table_id=True,
            ),
            record(
                "grid:string-table",
                110.0,
                ref_table_id="True",
                provenance_table_id="True",
            ),
        ),
        (
            record(
                "grid:page-conflict",
                10.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                page=8,
            ),
            record(
                "grid:page-clean",
                110.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
            ),
        ),
        (
            record(
                "grid:source-page-conflict",
                10.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                provenance_source_page=5,
            ),
            record(
                "grid:source-page-clean",
                110.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                provenance_source_page=4,
            ),
        ),
        (
            record(
                "grid:cross-source-A",
                10.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                source_page=4,
                provenance_source_page=4,
            ),
            record(
                "grid:cross-source-B",
                110.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                source_page=5,
                provenance_source_page=5,
            ),
        ),
        (
            record(
                "grid:provenance-page-conflict",
                10.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                provenance_page=8,
            ),
            record(
                "grid:provenance-page-clean",
                110.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                provenance_page=7,
            ),
        ),
        (
            record(
                "grid:bbox-missing",
                10.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
                include_bbox=False,
            ),
            record(
                "grid:bbox-clean",
                110.0,
                ref_table_id="table:B",
                provenance_table_id="table:B",
            ),
        ),
    )

    for records in cases:
        prepared = prepare_personal_detail_source_collections(
            {
                "facts": {},
                "datasets": {"repayment_records": list(records)},
            }
        )
        closure = prepared["facts"]["personal_detail_account_month_closure"]
        assert closure["source_month_position_observations"] == 2
        assert closure["raw_source_month_positions"] == 2
        assert closure["owner_bound_account_months"] == 2
        assert closure["physical_alias_source_month_observations"] == 0


def test_source_projection_rejects_non_exact_evidence_id_containers() -> None:
    performance_month = "2024-01"

    def record(
        account_id: str,
        grid_id: str,
        x: float,
        evidence_ids: object,
    ) -> dict[str, object]:
        return {
            "record_id": f"{grid_id}:{performance_month}",
            "repayment_id": f"{grid_id}:{performance_month}",
            "grid_id": grid_id,
            "account_id": account_id,
            "performance_month": performance_month,
            "status": "N",
            "_account_month_identity_proof_status": "exact",
            "_account_month_identity_proof": {
                "account_id": account_id,
                "performance_month": performance_month,
                "grid_id": grid_id,
                "owner_basis": "canonical_account_segment",
                "account_anchor_exact": True,
                "printed_month_range_exact": True,
                "grid_geometry_exact": True,
                "unique_owner": True,
            },
            "source_cell_refs": [
                {
                    "logical_page": 7,
                    "grid_id": grid_id,
                    "performance_month": performance_month,
                    "row": 4,
                    "col": 1,
                    "bbox": [x, 10.0, x + 10.0, 20.0],
                    "geometry_scope": "cell",
                    "evidence_ids": evidence_ids,
                }
            ],
        }

    evidence_cases = (
        ("alpha", "azure"),
        (["shared", "shared"], ["shared"]),
        (["shared", 7], ["shared"]),
        (["shared", " "], ["shared"]),
    )
    for first_ids, second_ids in evidence_cases:
        prepared = prepare_personal_detail_source_collections(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": [
                        record("account:A", "grid:A", 10.0, first_ids),
                        record("account:B", "grid:B", 110.0, second_ids),
                    ]
                },
            }
        )
        closure = prepared["facts"]["personal_detail_account_month_closure"]
        assert closure["candidate_identity_count"] == 2
        assert closure["source_month_position_observations"] == 2
        assert closure["raw_source_month_positions"] == 2
        assert closure["owner_bound_account_months"] == 2
        assert closure["physical_alias_source_month_observations"] == 0


def test_source_projection_conserves_observations_physical_positions_and_identities() -> None:
    candidate_records: list[dict[str, object]] = []
    aliases: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []

    for index in range(384):
        month_number = index % 12 + 1
        performance_month = f"2024-{month_number:02d}"
        account_id = f"account:{index // 12 + 1}"
        grid_id = f"grid:canonical:{index}"
        page = index // 12 + 1
        evidence_id = f"native:monthly:{index}"
        source_ref = {
            "logical_page": page,
            "grid_id": grid_id,
            "performance_month": performance_month,
            "bbox": [float(index), 100.0, float(index + 1), 110.0],
            "geometry_scope": "cell",
            "evidence_ids": [evidence_id],
        }
        candidate_records.append(
            {
                "record_id": f"{grid_id}:{performance_month}",
                "repayment_id": f"{grid_id}:{performance_month}",
                "grid_id": grid_id,
                "account_id": account_id,
                "performance_month": performance_month,
                "status": "N",
                "_account_month_identity_proof": {
                    "account_id": account_id,
                    "performance_month": performance_month,
                    "grid_id": grid_id,
                    "owner_basis": "canonical_account_segment",
                    "account_anchor_exact": True,
                    "printed_month_range_exact": True,
                    "grid_geometry_exact": True,
                    "unique_owner": True,
                },
                "_account_month_identity_proof_status": "exact",
                "source_cell_refs": [source_ref],
            }
        )
        if index < 5:
            alias_grid_id = f"grid:alias:{index}"
            aliases.append(
                make_issue(
                    category="ocr_structure_correction",
                    issue_code=(
                        "candidate_b_monthly_source_position_alias_reconciled"
                    ),
                    message="A detector alias observed the same physical month.",
                    severity="info",
                    status="informational",
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id=f"alias:{index}:{performance_month}",
                    field_name="performance_month",
                    observed_value={
                        "account_id": account_id,
                        "grid_id": alias_grid_id,
                        "performance_month": performance_month,
                        "source_position_state": "owner_bound_alias",
                    },
                    source_refs=(
                        {
                            **source_ref,
                            "grid_id": alias_grid_id,
                            "bbox": [
                                float(index) + 0.25,
                                100.0,
                                float(index) + 1.25,
                                110.0,
                            ],
                        },
                    ),
                )
            )

    for index in range(231):
        month_number = index % 12 + 1
        performance_month = f"2023-{month_number:02d}"
        grid_id = f"grid:unresolved:{index}"
        unresolved.append(
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_monthly_grid_owner_unresolved_field",
                message="The physical month has no exact account owner.",
                parser_stage="candidate_b_relationship_schema",
                target_dataset="repayment_records",
                target_record_id=f"{grid_id}:{performance_month}",
                field_name="performance_month",
                observed_value={
                    "grid_id": grid_id,
                    "performance_month": performance_month,
                },
                source_refs=(
                    {
                        "logical_page": index // 12 + 40,
                        "grid_id": grid_id,
                        "performance_month": performance_month,
                        "bbox": [
                            float(index),
                            200.0,
                            float(index + 1),
                            210.0,
                        ],
                        "geometry_scope": "cell",
                        "evidence_ids": [f"native:unresolved-monthly:{index}"],
                    },
                ),
            )
        )

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": candidate_records,
                "personal_detail_extraction_issues": [*aliases, *unresolved],
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["source_month_position_observations"] == 620
    assert closure["unique_physical_source_month_positions"] == 615
    assert closure["canonical_account_month_identity_count"] == 384
    assert closure["owner_bound_account_months"] == 384
    assert closure["owner_unresolved_positions"] == 231
    assert closure["owner_unresolved_position_observations"] == 231
    assert closure["physical_alias_source_month_observations"] == 5
    assert closure["alias_source_month_positions"] == 5
    assert closure["unlocalized_owner_unresolved_positions"] == 0
    assert closure["source_position_balance_valid"] is True


def test_source_projection_reconciles_detached_month_diagnostic_by_physical_position() -> None:
    performance_month = "2024-01"

    def physical_ref(grid_id: str) -> dict[str, object]:
        return {
            "logical_page": 7,
            "grid_id": grid_id,
            "performance_month": performance_month,
            "row": 4,
            "column": 1,
            "bbox": [120.0, 300.0, 145.0, 314.0],
            "geometry_scope": "cell",
            "evidence_ids": ["native:monthly:7:4:1"],
        }

    proof = {
        "account_id": "account:1",
        "performance_month": performance_month,
        "grid_id": "grid:canonical",
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    alias = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="The physical month is an exact alias of the canonical identity.",
        severity="info",
        status="informational",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="source_account_month:any:2024-01",
        field_name="performance_month",
        observed_value={
            "account_id": "account:1",
            "grid_id": "grid:alias",
            "performance_month": performance_month,
            "source_position_state": "owner_bound_alias",
        },
        source_refs=(physical_ref("grid:alias"),),
    )
    detached = make_issue(
        category="ocr_structure_correction",
        issue_code="canonical_monthly_source_structure_missing_field",
        message="A detached detector replayed the already inventoried cell.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        target_record_id="grid:detached:2024-01",
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:detached",
            "performance_month": performance_month,
        },
        source_refs=(physical_ref("grid:detached"),),
    )

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    {
                        "record_id": "grid:canonical:2024-01",
                        "repayment_id": "grid:canonical:2024-01",
                        "grid_id": "grid:canonical",
                        "account_id": "account:1",
                        "performance_month": performance_month,
                        "status": "N",
                        "_account_month_identity_proof": proof,
                        "_account_month_identity_proof_status": "exact",
                        "source_cell_refs": [physical_ref("grid:canonical")],
                    }
                ],
                "personal_detail_extraction_issues": [alias, detached],
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["source_month_position_observations"] == 3
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 0
    assert closure["alias_source_month_positions"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 1
    assert closure["source_position_balance_valid"] is True


def _detached_source_page_replay_content(
    *,
    bound_source_pages: tuple[int, ...] = (5,),
    geometry_state: str = "exact",
) -> dict[str, object]:
    account_id = "account:detached-source-page"
    performance_month = "2020-10"
    detached_grid_id = "grid:detached"
    status_bbox = [316.375, 374.5, 341.4166666666667, 396.0]
    amount_bbox = [316.375, 389.5, 341.4166666666667, 412.5]
    if geometry_state == "ambiguous":
        amount_bbox = [317.375, 389.5, 342.4166666666667, 412.5]

    def cell_refs(
        grid_id: str,
        *,
        source_page: int | None,
    ) -> list[dict[str, object]]:
        refs = [
            {
                "page": 10,
                "logical_page": 10,
                "geometry_scope": "cell",
                "coordinate_system": "pdf_points_top_left",
                "grid_id": grid_id,
                "row": row,
                "col": 10,
                "field_name": "performance_month",
                "source_field_name": source_field_name,
                "performance_month": performance_month,
                "bbox": bbox,
            }
            for row, source_field_name, bbox in (
                (2, "status", status_bbox),
                (3, "overdue_amount", amount_bbox),
            )
        ]
        if source_page is not None:
            for ref in refs:
                ref["source_page"] = source_page
        return refs

    records: list[dict[str, object]] = []
    for index, source_page in enumerate(bound_source_pages):
        grid_id = f"grid:bound:{index}"
        records.append(
            {
                "record_id": f"{grid_id}:{performance_month}",
                "repayment_id": f"{grid_id}:{performance_month}",
                "grid_id": grid_id,
                "account_id": account_id,
                "performance_month": performance_month,
                "status": "N",
                "_account_month_identity_proof": {
                    "account_id": account_id,
                    "performance_month": performance_month,
                    "grid_id": grid_id,
                    "owner_basis": "canonical_account_segment",
                    "account_anchor_exact": True,
                    "printed_month_range_exact": True,
                    "grid_geometry_exact": True,
                    "unique_owner": True,
                },
                "_account_month_identity_proof_status": "exact",
                "source_cell_refs": cell_refs(
                    grid_id,
                    source_page=source_page,
                ),
            }
        )

    detached_refs = cell_refs(detached_grid_id, source_page=None)
    performance_refs = detached_refs
    if geometry_state == "incomplete":
        performance_refs = detached_refs[:1]
    common_issue = {
        "category": "ocr_structure_correction",
        "issue_code": "canonical_monthly_source_structure_missing_field",
        "message": "A detached source structure replayed an inventoried month cell.",
        "parser_stage": "canonical_monthly_grid_materialization",
        "target_dataset": "repayment_records",
        "target_record_id": f"{detached_grid_id}:{performance_month}",
        "candidate_value": {"resolution": "withheld_pending_review"},
        "reason_codes": (
            "detached_source_structure_exact_key",
            "canonical_deduplicated_key_missing",
            "source_structure_is_audit_only",
            "account_month_owner_reconciliation_pending",
            "dataset_incomplete",
            "exact_grid_month_source_position",
            "normalized_value_withheld",
            "owner_or_status_value_not_invented",
        ),
    }

    def detached_issue(
        field_name: str,
        refs: list[dict[str, object]],
        source_observations: list[object],
    ) -> dict[str, object]:
        return make_issue(
            **common_issue,
            field_name=field_name,
            observed_value={
                "grid_id": detached_grid_id,
                "performance_month": performance_month,
                "field_state": "source_position_withheld",
                "source_observations": source_observations,
                "source_structure_key_count": 1,
            },
            source_refs=refs,
        )

    status_ref = deepcopy(detached_refs[0])
    status_ref["field_name"] = "status_code"
    amount_ref = deepcopy(detached_refs[1])
    amount_ref["field_name"] = "status_amount"
    issues = [
        detached_issue(
            "performance_month",
            deepcopy(performance_refs),
            [performance_month],
        ),
        detached_issue("status_code", [status_ref], ["N"]),
        detached_issue("status_amount", [amount_ref], [0]),
    ]
    return {
        "facts": {},
        "datasets": {
            "repayment_records": records,
            "personal_detail_extraction_issues": issues,
        },
    }


def test_source_projection_reconciles_complete_detached_pair_when_only_source_page_is_missing() -> None:
    closure = _month_closure(_detached_source_page_replay_content())

    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 0
    assert closure["reconciled_detached_diagnostic_positions"] == 1
    assert closure["source_position_balance_valid"] is True


def test_source_projection_does_not_bridge_detached_pair_across_conflicting_source_pages() -> None:
    closure = _month_closure(
        _detached_source_page_replay_content(bound_source_pages=(5, 6))
    )

    assert closure["source_month_position_observations"] == 3
    assert closure["raw_source_month_positions"] == 3
    assert closure["owner_bound_account_months"] == 2
    assert closure["owner_unresolved_positions"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 0
    assert closure["source_position_balance_valid"] is True


def test_source_projection_does_not_bridge_incomplete_or_ambiguous_detached_geometry() -> None:
    for geometry_state in ("incomplete", "ambiguous"):
        closure = _month_closure(
            _detached_source_page_replay_content(geometry_state=geometry_state)
        )

        assert closure["source_month_position_observations"] == 2, geometry_state
        assert closure["raw_source_month_positions"] == 2, geometry_state
        assert closure["owner_bound_account_months"] == 1, geometry_state
        assert closure["owner_unresolved_positions"] == 1, geometry_state
        assert closure["reconciled_detached_diagnostic_positions"] == 0, geometry_state
        assert closure["source_position_balance_valid"] is True, geometry_state


def test_source_projection_does_not_trust_unrelated_issue_geometry_for_bridge() -> None:
    content = _detached_source_page_replay_content()
    datasets = content["datasets"]
    bound_record = datasets["repayment_records"][0]
    for ref in bound_record["source_cell_refs"]:
        left, top, right, bottom = ref["bbox"]
        ref["bbox"] = [left + 100.0, top, right + 100.0, bottom]

    detached_month_issue = next(
        issue
        for issue in datasets["personal_detail_extraction_issues"]
        if issue["field_name"] == "performance_month"
    )
    probe_refs = deepcopy(detached_month_issue["source_refs"])
    for ref in probe_refs:
        ref["grid_id"] = "grid:bound:0"
        ref["source_page"] = 5
    datasets["personal_detail_extraction_issues"].append(
        make_issue(
            category="ocr_structure_correction",
            issue_code="unrelated_probe",
            message="An unrelated diagnostic must not supply owner inventory geometry.",
            parser_stage="unrelated_probe",
            target_dataset="repayment_records",
            target_record_id="unrelated:probe",
            field_name="performance_month",
            observed_value={
                "account_id": "account:detached-source-page",
                "grid_id": "grid:bound:0",
                "performance_month": "2020-10",
            },
            source_refs=probe_refs,
        )
    )

    closure = _month_closure(content)

    assert closure["reconciled_detached_diagnostic_positions"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["status"] == "partial_owner_unresolved"


def test_source_projection_does_not_build_bridge_pair_from_single_ref_claims() -> None:
    content = _detached_source_page_replay_content()
    datasets = content["datasets"]
    bound_record = datasets["repayment_records"][0]
    for ref in bound_record["source_cell_refs"]:
        left, top, right, bottom = ref["bbox"]
        ref["bbox"] = [left + 100.0, top, right + 100.0, bottom]

    account_id = "account:detached-source-page"
    performance_month = "2020-10"
    owner_key = stable_record_id(
        "source_account_month_owner",
        account_id,
    ).split(":", 1)[-1]
    detached_month_issue = next(
        issue
        for issue in datasets["personal_detail_extraction_issues"]
        if issue["field_name"] == "performance_month"
    )
    for source_ref in detached_month_issue["source_refs"]:
        claim_ref = deepcopy(source_ref)
        claim_ref.update(
            {
                "account_id": account_id,
                "binding": "source_account_month_range",
                "grid_id": "grid:bound:0",
                "source_page": 5,
            }
        )
        datasets["personal_detail_extraction_issues"].append(
            make_issue(
                category="ocr_structure_correction",
                issue_code="candidate_b_monthly_account_range_missing_month",
                message="One range claim cannot supply a complete bridge inventory.",
                parser_stage="candidate_b_relationship_schema",
                target_dataset="repayment_records",
                target_record_id=(
                    f"source_account_month:{owner_key}:{performance_month}"
                ),
                field_name="performance_month",
                observed_value={
                    "account_id": account_id,
                    "grid_id": "grid:bound:0",
                    "performance_month": performance_month,
                    "source_identity_type": (
                        "account_month_from_printed_repayment_range"
                    ),
                },
                source_refs=[claim_ref],
            )
        )

    closure = _month_closure(content)

    assert closure["reconciled_detached_diagnostic_positions"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["status"] == "partial_owner_unresolved"


def test_source_projection_requires_source_page_on_each_inventoried_cell() -> None:
    content = _detached_source_page_replay_content()
    bound_record = content["datasets"]["repayment_records"][0]
    bound_record["source_cell_refs"][1].pop("source_page")

    closure = _month_closure(content)

    assert closure["reconciled_detached_diagnostic_positions"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["status"] == "partial_owner_unresolved"


def test_source_projection_requires_every_detached_sibling_to_be_exact() -> None:
    for field_name in ("status_code", "status_amount"):
        content = _detached_source_page_replay_content()
        issues = content["datasets"]["personal_detail_extraction_issues"]
        sibling = next(issue for issue in issues if issue["field_name"] == field_name)
        sibling["category"] = "unrelated_category"
        sibling["status"] = "resolved"
        sibling["parser_stage"] = "unrelated_stage"
        sibling["candidate_value"] = {"resolution": "invented"}
        sibling["reason_codes"] = ["unrelated_reason"]
        sibling["source_refs"] = []

        closure = _month_closure(content)

        assert closure["reconciled_detached_diagnostic_positions"] == 0, field_name
        assert closure["owner_unresolved_positions"] == 1, field_name
        assert closure["status"] == "source_localization_invalid", field_name


def test_source_projection_keeps_distinct_physical_cells_for_one_identity() -> None:
    performance_month = "2024-01"

    def record(grid_id: str, bbox: list[float], evidence_id: str) -> dict[str, object]:
        return {
            "record_id": f"{grid_id}:{performance_month}",
            "repayment_id": f"{grid_id}:{performance_month}",
            "grid_id": grid_id,
            "account_id": "account:1",
            "performance_month": performance_month,
            "status": "N",
            "_account_month_identity_proof": {
                "account_id": "account:1",
                "performance_month": performance_month,
                "grid_id": grid_id,
                "owner_basis": "canonical_account_segment",
                "account_anchor_exact": True,
                "printed_month_range_exact": True,
                "grid_geometry_exact": True,
                "unique_owner": True,
            },
            "_account_month_identity_proof_status": "exact",
            "source_cell_refs": [
                {
                    "logical_page": 7,
                    "grid_id": grid_id,
                    "performance_month": performance_month,
                    "row": 4,
                    "column": 1,
                    "bbox": bbox,
                    "geometry_scope": "cell",
                    "coordinate_system": "pdf_points_top_left",
                    "geometry_provenance": {
                        "source": "source_table_geometry",
                        "coordinate_system": "pdf_points_top_left",
                        "logical_page": 7,
                        "table_id": f"table:{grid_id}",
                        "calibrated_from_source_table_geometry": True,
                        "active_cell_geometry_exact": True,
                        "value_inputs_used": False,
                    },
                    "evidence_ids": [evidence_id],
                }
            ],
        }

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    record(
                        "grid:first",
                        [120.0, 300.0, 145.0, 314.0],
                        "native:monthly:7:4:1",
                    ),
                    record(
                        "grid:second",
                        [220.0, 300.0, 245.0, 314.0],
                        "native:monthly:7:4:2",
                    ),
                ]
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["expected_identity_count"] == 1
    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 2
    assert closure["unique_physical_source_month_positions"] == 2
    assert closure["canonical_account_month_identity_count"] == 1
    assert closure["physical_alias_source_month_observations"] == 0
    assert closure["owner_bound_account_months"] == 2
    assert closure["owner_unresolved_positions"] == 0
    assert closure["alias_source_month_positions"] == 1
    assert closure["source_position_balance_valid"] is True


def test_source_projection_owned_grid_issue_preserves_raw_position_and_closed_source() -> None:
    account_id = "account:owned"
    performance_month = "2024-04"
    owner_hash = stable_record_id(
        "source_account_month_owner", account_id
    ).split(":", 1)[-1]

    def prepared_for(source: str) -> dict[str, object]:
        issue = make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_monthly_owned_grid_missing_field",
            message="Exact owned grid month was withheld.",
            parser_stage="candidate_b_final_monthly_gate",
            target_dataset="repayment_records",
            target_record_id=(
                f"source_account_month:{owner_hash}:{performance_month}"
            ),
            field_name="performance_month",
            observed_value={
                "account_id": account_id,
                "performance_month": performance_month,
            },
            source_refs=(
                {
                    "source": source,
                    "logical_page": 6,
                    "source_page": 3,
                    "table_id": "monthly-table:owned",
                    "row": 4,
                    "column": 4,
                    "bbox": [20.0, 40.0, 30.0, 50.0],
                    "geometry_scope": "cell",
                    "account_id": account_id,
                    "grid_id": "grid:owned",
                    "performance_month": performance_month,
                    "field_name": "performance_month",
                    "evidence_ids": ["native:monthly-table:owned:4:4"],
                    "binding": "source_account_month_identity",
                    "binding_quality": "source_account_month_identity",
                },
            ),
        )
        return prepare_personal_detail_source_collections(
            {
                "facts": {},
                "datasets": {
                    "personal_detail_extraction_issues": [issue],
                },
            }
        )

    accepted = prepared_for("candidate_b_monthly_owned_grid_cell")
    closure = accepted["facts"]["personal_detail_account_month_closure"]
    assert closure["expected_identity_count"] == 1
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 0

    rejected = prepared_for("unrelated_cell")
    rejected_closure = rejected["facts"]["personal_detail_account_month_closure"]
    assert rejected_closure["expected_identity_count"] == 0
    assert rejected_closure["raw_source_month_positions"] == 1
    assert rejected_closure["owner_bound_account_months"] == 0
    assert rejected_closure["owner_unresolved_positions"] == 1


def test_source_projection_keeps_unlocalized_month_position_visible_and_invalid() -> None:
    unresolved = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_grid_owner_unresolved_field",
        message="The source position has no unique owner or surviving ref.",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="grid:unlocalized:2024-04",
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:unlocalized",
            "performance_month": "2024-04",
        },
    )

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [],
                "personal_detail_extraction_issues": [unresolved],
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["owner_unresolved_position_observations"] == 1
    assert closure["unlocalized_owner_unresolved_positions"] == 1
    assert closure["source_position_balance_valid"] is True
    assert closure["status"] == "source_localization_invalid"


def test_source_projection_does_not_reconcile_alias_without_exact_owner() -> None:
    ownerless_alias = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="A source position was tentatively labelled as an alias.",
        severity="info",
        status="informational",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="grid:ownerless:2024-05",
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:ownerless",
            "performance_month": "2024-05",
        },
        source_refs=(
            {
                "logical_page": 6,
                "grid_id": "grid:ownerless",
                "performance_month": "2024-05",
                "bbox": [40, 80, 180, 120],
                "geometry_scope": "grid",
            },
        ),
    )

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [],
                "personal_detail_extraction_issues": [ownerless_alias],
            },
        }
    )

    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["expected_identity_count"] == 0
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["alias_source_month_positions"] == 0
    assert closure["source_localized_owner_unresolved_positions"] == 1
    assert closure["source_position_balance_valid"] is True
    assert closure["status"] == "partial_owner_unresolved"


def _asymmetric_detached_month_alias_content(
    *,
    alias_status: str = "informational",
) -> dict[str, object]:
    """Model Yang page 8 without depending on private OCR fixtures."""

    account_id = "credit_account:non_revolving_loan:15"
    performance_month = "2020-01"
    primary_grid_id = "mg_p8_repayment_1"
    alias_grid_id = "mg_p8_repayment_0"
    owner_key = stable_record_id(
        "source_account_month_owner", account_id
    ).split(":", 1)[-1]
    target_record_id = f"source_account_month:{owner_key}:{performance_month}"
    proof = {
        "account_id": account_id,
        "performance_month": performance_month,
        "grid_id": primary_grid_id,
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    primary = {
        "record_id": f"{primary_grid_id}:{performance_month}",
        "repayment_id": f"{primary_grid_id}:{performance_month}",
        "grid_id": primary_grid_id,
        "account_id": account_id,
        "performance_month": performance_month,
        "status": "N",
        "_account_month_identity_proof": proof,
        "_account_month_identity_proof_status": "exact",
        # The retained grid has an exact owner/month proof, but its source-table
        # cell geometry was rejected.  It intentionally contributes no physical
        # fingerprint, matching the frozen Yang artifact.
        "source_cell_refs": [
            {
                "page": 8,
                "logical_page": 8,
                "geometry_scope": "logical_page",
                "geometry_status": "unresolved",
                "coordinate_system": "pdf_points_top_left",
                "grid_id": primary_grid_id,
                "row": 2,
                "col": 1,
                "field_name": "status",
                "geometry_rejection": {
                    "source": "rejected_month_geometry",
                    "reason": "source_table_month_ownership_required",
                    "logical_page": 8,
                    "value_inputs_used": False,
                },
            }
        ],
    }
    alias_refs = [
        {
            "page": 8,
            "logical_page": 8,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": alias_grid_id,
            "row": row,
            "col": 1,
            "field_name": "performance_month",
            "source_field_name": source_field_name,
            "performance_month": performance_month,
            "bbox": bbox,
            "account_id": account_id,
            "binding": "source_account_month_alias",
            "binding_quality": "source_account_month_alias",
        }
        for row, source_field_name, bbox in (
            (2, "status", [91.5, 287.5, 116.58333333333333, 304.5]),
            (3, "overdue_amount", [91.5, 301.0, 116.58333333333333, 317.0]),
        )
    ]
    sibling_refs = [
        {
            key: value
            for key, value in ref.items()
            if key not in {"account_id", "binding", "binding_quality"}
        }
        for ref in alias_refs
    ]
    sibling = make_issue(
        category="ocr_structure_correction",
        issue_code="canonical_monthly_source_structure_missing_field",
        message="The detached source grid/month was absent from the corrected grid plane.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        target_record_id=f"{alias_grid_id}:{performance_month}",
        field_name="performance_month",
        observed_value={
            "grid_id": alias_grid_id,
            "performance_month": performance_month,
            "field_state": "source_position_withheld",
            "source_observations": [performance_month],
            "source_structure_key_count": 5,
        },
        candidate_value={"resolution": "withheld_pending_review"},
        source_refs=sibling_refs,
        reason_codes=(
            "detached_source_structure_exact_key",
            "canonical_deduplicated_key_missing",
            "source_structure_is_audit_only",
            "account_month_owner_reconciliation_pending",
            "dataset_incomplete",
            "exact_grid_month_source_position",
            "normalized_value_withheld",
            "owner_or_status_value_not_invented",
        ),
    )
    alias = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="The detached position aliases an exact owner-bound account month.",
        severity="info",
        status=alias_status,
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id=target_record_id,
        field_name="performance_month",
        observed_value={
            "account_id": account_id,
            "grid_id": alias_grid_id,
            "performance_month": performance_month,
            "source_position_state": "owner_bound_alias",
            "account_month_owner_basis": "canonical_account_segment",
        },
        candidate_value={
            "resolution": "reconciled_to_existing_account_month_identity"
        },
        source_refs=alias_refs,
        reason_codes=(
            "exact_account_month_identity",
            "distinct_source_position_alias",
            "canonical_identity_not_double_counted",
            "source_position_audit_preserved",
        ),
    )
    return {
        "facts": {},
        "datasets": {
            "repayment_records": [primary],
            "personal_detail_extraction_issues": [sibling, alias],
        },
    }


def _month_closure(content: dict[str, object]) -> dict[str, object]:
    prepared = prepare_personal_detail_source_collections(content)
    return prepared["facts"]["personal_detail_account_month_closure"]


def test_source_projection_reconciles_strict_asymmetric_detached_alias_only_in_raw_denominator() -> None:
    for status in ("informational", "resolved"):
        closure = _month_closure(
            _asymmetric_detached_month_alias_content(alias_status=status)
        )
        assert closure["candidate_identity_count"] == 1
        assert closure["source_month_position_observations"] == 2
        assert closure["raw_source_month_positions"] == 1
        assert closure["unique_physical_source_month_positions"] == 1
        assert closure["owner_bound_account_months"] == 1
        assert closure["owner_unresolved_positions"] == 0
        assert closure["alias_source_month_positions"] == 1
        assert closure["physical_alias_source_month_observations"] == 1
        assert closure["source_position_balance_valid"] is True

    for retained_optional_keys in (
        frozenset(),
        frozenset({"source_observations"}),
        frozenset({"source_structure_key_count"}),
    ):
        content = _asymmetric_detached_month_alias_content()
        sibling = content["datasets"]["personal_detail_extraction_issues"][0]  # type: ignore[index]
        observed = sibling["observed_value"]
        for key in {"source_observations", "source_structure_key_count"}:
            if key not in retained_optional_keys:
                observed.pop(key)
        assert _month_closure(content)["raw_source_month_positions"] == 1

    order_insensitive = _asymmetric_detached_month_alias_content()
    issues = order_insensitive["datasets"]["personal_detail_extraction_issues"]  # type: ignore[index]
    issues[1]["source_refs"].reverse()
    assert _month_closure(order_insensitive)["raw_source_month_positions"] == 1

    wrapped = _asymmetric_detached_month_alias_content()
    wrapped_issues = wrapped["datasets"]["personal_detail_extraction_issues"]  # type: ignore[index]
    for index, issue in enumerate(wrapped_issues):
        refs = issue.pop("source_refs")
        wrapped_issues[index] = {
            "normalized": issue,
            "source": {"source_refs": refs},
        }
    assert _month_closure(wrapped)["raw_source_month_positions"] == 1

    production_family = _asymmetric_detached_month_alias_content()
    production_issues = production_family["datasets"][  # type: ignore[index]
        "personal_detail_extraction_issues"
    ]
    performance_sibling = production_issues[0]
    status_sibling = deepcopy(performance_sibling)
    status_sibling["field_name"] = "status_code"
    status_sibling["observed_value"]["source_observations"] = ["N"]
    status_sibling["source_refs"] = [status_sibling["source_refs"][0]]
    status_sibling["source_refs"][0]["field_name"] = "status_code"
    amount_sibling = deepcopy(performance_sibling)
    amount_sibling["field_name"] = "status_amount"
    amount_sibling["observed_value"]["source_observations"] = [100]
    amount_sibling["source_refs"] = [amount_sibling["source_refs"][1]]
    amount_sibling["source_refs"][0]["field_name"] = "status_amount"
    production_issues[1:1] = [status_sibling, amount_sibling]
    assert _month_closure(production_family)["raw_source_month_positions"] == 1


def test_source_projection_detached_alias_contract_mutations_fail_closed() -> None:
    def alias(content: dict[str, object]) -> dict[str, object]:
        return content["datasets"]["personal_detail_extraction_issues"][1]  # type: ignore[index]

    def sibling(content: dict[str, object]) -> dict[str, object]:
        return content["datasets"]["personal_detail_extraction_issues"][0]  # type: ignore[index]

    def alias_ref(content: dict[str, object], index: int = 0) -> dict[str, object]:
        return alias(content)["source_refs"][index]  # type: ignore[index]

    def mutate_alias_field(key: str, value: object):
        return lambda content: alias(content).__setitem__(key, value)

    def mutate_alias_observed(key: str, value: object):
        return lambda content: alias(content)["observed_value"].__setitem__(  # type: ignore[union-attr]
            key, value
        )

    def mutate_alias_ref(key: str, value: object, index: int = 0):
        return lambda content: alias_ref(content, index).__setitem__(key, value)

    def mutate_sibling_field(key: str, value: object):
        return lambda content: sibling(content).__setitem__(key, value)

    def mutate_sibling_observed(key: str, value: object):
        return lambda content: sibling(content)["observed_value"].__setitem__(  # type: ignore[union-attr]
            key, value
        )

    def remove_sibling_localization(content: dict[str, object]) -> None:
        for ref in sibling(content)["source_refs"]:  # type: ignore[union-attr]
            ref.pop("performance_month", None)
            ref["col"] = 2

    def shift_alias_pair(content: dict[str, object]) -> None:
        for ref in alias(content)["source_refs"]:  # type: ignore[union-attr]
            ref["row"] += 10
            x0, y0, x1, y1 = ref["bbox"]
            ref["bbox"] = [x0, y0 + 100.0, x1, y1 + 100.0]

    def append_invalid_alias_replay(content: dict[str, object]) -> None:
        replay = deepcopy(alias(content))
        replay["observed_value"]["account_month_owner_basis"] = "nearest_account"
        content["datasets"]["personal_detail_extraction_issues"].append(replay)  # type: ignore[index]

    def append_invalid_sibling_replay(content: dict[str, object]) -> None:
        replay = deepcopy(sibling(content))
        replay["parser_stage"] = "manual_stage"
        content["datasets"]["personal_detail_extraction_issues"].insert(1, replay)  # type: ignore[index]

    def append_blank_alias_replay(
        content: dict[str, object],
        field_name: str,
    ) -> None:
        replay = deepcopy(alias(content))
        replay["observed_value"][field_name] = ""
        content["datasets"]["personal_detail_extraction_issues"].append(replay)  # type: ignore[index]

    def append_blank_sibling_replay(
        content: dict[str, object],
        field_name: str,
    ) -> None:
        replay = deepcopy(sibling(content))
        replay["observed_value"][field_name] = ""
        content["datasets"]["personal_detail_extraction_issues"].insert(1, replay)  # type: ignore[index]

    def append_unaddressable_target_replay(
        content: dict[str, object],
        *,
        issue_kind: str,
        ref_damage: str,
    ) -> None:
        source_issue = alias(content) if issue_kind == "alias" else sibling(content)
        replay = deepcopy(source_issue)
        for key in ("account_id", "grid_id", "performance_month"):
            if key in replay["observed_value"]:
                replay["observed_value"][key] = ""
        if ref_damage == "nonmapping":
            replay["source_refs"].append("not-a-ref")
        else:
            replay["source_refs"][0]["bbox"] = [1.0, 2.0, 1.0, 3.0]
        content["datasets"]["personal_detail_extraction_issues"].append(replay)  # type: ignore[index]

    def append_sibling_fields(
        content: dict[str, object],
        *field_names: str,
    ) -> None:
        for field_name in field_names:
            replay = deepcopy(sibling(content))
            replay["field_name"] = field_name
            content["datasets"]["personal_detail_extraction_issues"].insert(1, replay)  # type: ignore[index]

    mutations = (
        ("alias category", mutate_alias_field("category", "ocr_cell_level_error")),
        ("alias code", mutate_alias_field("issue_code", "manual_alias")),
        ("alias severity", mutate_alias_field("severity", "warning")),
        ("alias status", mutate_alias_field("status", "requires_review")),
        ("alias stage", mutate_alias_field("parser_stage", "manual_stage")),
        ("alias dataset", mutate_alias_field("target_dataset", "other_dataset")),
        ("alias field", mutate_alias_field("field_name", "status_code")),
        ("alias target", mutate_alias_field("target_record_id", "source_account_month:any:2020-01")),
        ("alias source state", mutate_alias_observed("source_position_state", "tentative_alias")),
        ("alias owner basis empty", mutate_alias_observed("account_month_owner_basis", "")),
        ("alias owner basis unknown", mutate_alias_observed("account_month_owner_basis", "nearest_account")),
        ("alias observed surplus", mutate_alias_observed("untrusted", "value")),
        (
            "alias candidate resolution",
            lambda content: alias(content)["candidate_value"].__setitem__(  # type: ignore[union-attr]
                "resolution", "manual_reconciliation"
            ),
        ),
        (
            "alias reason codes",
            lambda content: alias(content)["reason_codes"].append("forged_reason"),  # type: ignore[union-attr]
        ),
        ("alias ref binding", mutate_alias_ref("binding", "source_account_month_identity")),
        ("alias ref binding quality", mutate_alias_ref("binding_quality", "tentative")),
        ("alias ref account", mutate_alias_ref("account_id", "account:other")),
        ("alias ref grid", mutate_alias_ref("grid_id", "grid:other")),
        ("alias ref month", mutate_alias_ref("performance_month", "2020-02")),
        ("alias ref field", mutate_alias_ref("field_name", "status_code")),
        ("alias ref scope", mutate_alias_ref("geometry_scope", "logical_page")),
        ("alias ref coordinates", mutate_alias_ref("coordinate_system", "image_pixels")),
        ("alias ref row", mutate_alias_ref("row", True)),
        ("alias ref column", mutate_alias_ref("col", 2)),
        ("alias ref bbox", mutate_alias_ref("bbox", [1.0, 2.0, 1.0, 3.0])),
        ("alias ref page", mutate_alias_ref("logical_page", 9)),
        ("alias ref foreign source", mutate_alias_ref("source", "foreign")),
        ("alias ref foreign role", mutate_alias_ref("role", "foreign")),
        ("mixed locator page", mutate_alias_ref("page", 9, 1)),
        ("mixed source-page state", mutate_alias_ref("source_page", 4, 1)),
        ("shifted alias role pair", shift_alias_pair),
        (
            "alias nonmapping source ref",
            lambda content: alias(content)["source_refs"].append("not-a-ref"),  # type: ignore[union-attr]
        ),
        (
            "surplus alias ref",
            lambda content: alias(content)["source_refs"].append(  # type: ignore[union-attr]
                {
                    **deepcopy(alias_ref(content)),
                    "row": 999,
                    "source_field_name": "foreign",
                    "bbox": [500.0, 600.0, 525.0, 614.0],
                }
            ),
        ),
        ("sibling category", mutate_sibling_field("category", "ocr_cell_level_error")),
        ("sibling code", mutate_sibling_field("issue_code", "manual_detached_position")),
        ("sibling status", mutate_sibling_field("status", "resolved")),
        ("sibling stage", mutate_sibling_field("parser_stage", "manual_stage")),
        ("sibling dataset", mutate_sibling_field("target_dataset", "other_dataset")),
        ("sibling field", mutate_sibling_field("field_name", "status_code")),
        ("sibling target", mutate_sibling_field("target_record_id", "grid:other:2020-01")),
        ("sibling grid", mutate_sibling_observed("grid_id", "grid:other")),
        ("sibling month", mutate_sibling_observed("performance_month", "2020-02")),
        ("sibling observed surplus", mutate_sibling_observed("untrusted", "value")),
        ("sibling observations", mutate_sibling_observed("source_observations", ["2020-02"])),
        ("sibling key count zero", mutate_sibling_observed("source_structure_key_count", 0)),
        ("sibling key count bool", mutate_sibling_observed("source_structure_key_count", True)),
        (
            "sibling candidate",
            lambda content: sibling(content)["candidate_value"].__setitem__(  # type: ignore[union-attr]
                "resolution", "forged_resolution"
            ),
        ),
        (
            "sibling reason codes",
            lambda content: sibling(content)["reason_codes"].append("forged_reason"),  # type: ignore[union-attr]
        ),
        (
            "sibling degenerate bbox",
            lambda content: sibling(content)["source_refs"][0].__setitem__(  # type: ignore[index]
                "bbox", [1.0, 2.0, 1.0, 3.0]
            ),
        ),
        (
            "sibling foreign source",
            lambda content: sibling(content)["source_refs"][0].__setitem__(  # type: ignore[index]
                "source", "foreign"
            ),
        ),
        (
            "sibling mixed locator",
            lambda content: sibling(content)["source_refs"][1].__setitem__(  # type: ignore[index]
                "page", 9
            ),
        ),
        (
            "sibling nonmapping source ref",
            lambda content: sibling(content)["source_refs"].append("not-a-ref"),  # type: ignore[union-attr]
        ),
        ("sibling localization", remove_sibling_localization),
        (
            "duplicate sibling",
            lambda content: content["datasets"][  # type: ignore[index]
                "personal_detail_extraction_issues"
            ].insert(1, deepcopy(sibling(content))),
        ),
        (
            "duplicate alias replay",
            lambda content: content["datasets"][  # type: ignore[index]
                "personal_detail_extraction_issues"
            ].append(deepcopy(alias(content))),
        ),
        ("invalid alias replay", append_invalid_alias_replay),
        ("invalid sibling replay", append_invalid_sibling_replay),
        (
            "blank-account alias replay",
            lambda content: append_blank_alias_replay(content, "account_id"),
        ),
        (
            "blank-month alias replay",
            lambda content: append_blank_alias_replay(content, "performance_month"),
        ),
        (
            "blank-grid sibling replay",
            lambda content: append_blank_sibling_replay(content, "grid_id"),
        ),
        (
            "blank-month sibling replay",
            lambda content: append_blank_sibling_replay(
                content,
                "performance_month",
            ),
        ),
        (
            "alias target replay with blank observation and nonmapping ref",
            lambda content: append_unaddressable_target_replay(
                content,
                issue_kind="alias",
                ref_damage="nonmapping",
            ),
        ),
        (
            "alias target replay with blank observation and invalid bbox",
            lambda content: append_unaddressable_target_replay(
                content,
                issue_kind="alias",
                ref_damage="bbox",
            ),
        ),
        (
            "sibling target replay with blank observation and nonmapping ref",
            lambda content: append_unaddressable_target_replay(
                content,
                issue_kind="sibling",
                ref_damage="nonmapping",
            ),
        ),
        (
            "sibling target replay with blank observation and invalid bbox",
            lambda content: append_unaddressable_target_replay(
                content,
                issue_kind="sibling",
                ref_damage="bbox",
            ),
        ),
        (
            "duplicate optional sibling field",
            lambda content: append_sibling_fields(
                content,
                "status_code",
                "status_code",
            ),
        ),
        (
            "blank sibling field",
            lambda content: append_sibling_fields(content, ""),
        ),
        (
            "unknown sibling field",
            lambda content: append_sibling_fields(content, "foreign_field"),
        ),
    )

    for label, mutate in mutations:
        content = _asymmetric_detached_month_alias_content()
        mutate(content)
        closure = _month_closure(content)
        assert int(closure["raw_source_month_positions"]) >= 2, (
            label,
            closure,
        )
        assert closure["source_position_balance_valid"] is True, (label, closure)


def test_source_projection_does_not_hide_alias_only_or_cross_owner_positions() -> None:
    alias_only = _asymmetric_detached_month_alias_content()
    alias_only["datasets"]["repayment_records"] = []  # type: ignore[index]
    alias_only_closure = _month_closure(alias_only)
    assert alias_only_closure["source_month_position_observations"] == 1
    assert alias_only_closure["raw_source_month_positions"] == 1
    assert alias_only_closure["owner_bound_account_months"] == 1
    assert alias_only_closure["physical_alias_source_month_observations"] == 0

    cross_owner = _asymmetric_detached_month_alias_content()
    second_account = "credit_account:non_revolving_loan:16"
    alias_row = cross_owner["datasets"]["personal_detail_extraction_issues"][1]  # type: ignore[index]
    shared_ref = deepcopy(alias_row["source_refs"][0])
    shared_ref["account_id"] = second_account
    second_proof = {
        "account_id": second_account,
        "performance_month": "2020-01",
        "grid_id": "mg_p8_repayment_0",
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    cross_owner["datasets"]["repayment_records"].append(  # type: ignore[index]
        {
            "record_id": "mg_p8_repayment_0:2020-01:other-owner",
            "repayment_id": "mg_p8_repayment_0:2020-01:other-owner",
            "grid_id": "mg_p8_repayment_0",
            "account_id": second_account,
            "performance_month": "2020-01",
            "status": "N",
            "_account_month_identity_proof": second_proof,
            "_account_month_identity_proof_status": "exact",
            "source_cell_refs": [shared_ref],
        }
    )
    cross_owner_closure = _month_closure(cross_owner)
    assert cross_owner_closure["source_month_position_observations"] == 2
    assert cross_owner_closure["raw_source_month_positions"] == 2
    assert cross_owner_closure["owner_bound_account_months"] == 2
    assert cross_owner_closure["cross_owner_physical_conflict_count"] == 1
    assert cross_owner_closure["physical_owner_conflict_free"] is False
