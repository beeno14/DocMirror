from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.community_plugin import (
    _CreditReportCommunityBundle,
    _apply_personal_detail_dataset_status,
)
from docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger import (
    apply_document_consistency_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
    _PRINTED_GRID_CENSUS_ISSUE_CODE,
    _localized_monthly_source_refs,
    _printed_anchor_identity_key,
    _printed_anchor_inventory_source_refs,
    _printed_grid_census_source_refs,
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


def _physical_month_refs(
    grid_id: str,
    performance_month: str = "2024-01",
    *,
    page: int = 7,
    source_page: int = 4,
    x: float = 120.0,
    y: float = 300.0,
    evidence_prefix: str = "native:monthly",
) -> list[dict[str, object]]:
    """One coherent, explicitly located status/amount observation."""
    month = int(performance_month[5:7])
    return [
        {
            "source": "native_detail_table",
            "page": page,
            "logical_page": page,
            "source_page": source_page,
            "source_logical_page": page,
            "table_id": "table:monthly",
            "grid_id": grid_id,
            "performance_month": performance_month,
            "row": row,
            "col": month,
            "field_name": "performance_month",
            "source_field_name": role,
            "bbox": [x, top, x + 25.0, top + 14.0],
            "geometry_scope": "cell",
            "geometry_status": "accepted",
            "coordinate_system": "pdf_points_top_left",
            "evidence_ids": [f"{evidence_prefix}:{role}"],
            "geometry_provenance": {
                "source": "source_table_geometry",
                "coordinate_system": "pdf_points_top_left",
                "logical_page": page,
                "source_page": source_page,
                "source_logical_page": page,
                "table_id": "table:monthly",
                "active_cell_geometry_exact": True,
                "value_inputs_used": False,
            },
        }
        for row, role, top in (
            (4, "status", y),
            (5, "overdue_amount", y + 14.0),
        )
    ]


def _physical_month_record(
    grid_id: str,
    *,
    account_id: str = "account:1",
    performance_month: str = "2024-01",
    refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "record_id": f"{grid_id}:{performance_month}:{account_id}",
        "repayment_id": f"{grid_id}:{performance_month}:{account_id}",
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
        "source_cell_refs": deepcopy(
            refs if refs is not None else _physical_month_refs(grid_id, performance_month)
        ),
    }


def _physical_month_structural_issue(
    grid_id: str,
    *,
    performance_month: str = "2024-01",
    refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return make_issue(
        category="ocr_structure_correction",
        issue_code="canonical_monthly_source_structure_missing_field",
        message="The exact source position is withheld pending owner reconciliation.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        target_record_id=f"{grid_id}:{performance_month}",
        field_name="performance_month",
        observed_value={
            "grid_id": grid_id,
            "performance_month": performance_month,
        },
        candidate_value={"resolution": "withheld_pending_review"},
        source_refs=deepcopy(
            refs if refs is not None else _physical_month_refs(grid_id, performance_month)
        ),
    )


def _physical_month_alias_issue(
    grid_id: str,
    *,
    account_id: str = "account:1",
    performance_month: str = "2024-01",
    refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    alias_refs = deepcopy(
        refs if refs is not None else _physical_month_refs(grid_id, performance_month)
    )
    for ref in alias_refs:
        ref.update(
            account_id=account_id,
            binding="source_account_month_alias",
            binding_quality="source_account_month_alias",
        )
    owner_key = stable_record_id(
        "source_account_month_owner", account_id
    ).split(":", 1)[-1]
    return make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_source_position_alias_reconciled",
        message="The source position has an exact account/month owner.",
        severity="info",
        status="informational",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id=f"source_account_month:{owner_key}:{performance_month}",
        field_name="performance_month",
        observed_value={
            "account_id": account_id,
            "grid_id": grid_id,
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


def test_monthly_source_localization_keeps_year_and_detector_identity() -> None:
    """A multi-year detached grid must not pool same-column role pairs."""
    grid_id = "mg_p7_repayment_1"
    records = []
    for source_grid_id, performance_month, top in (
        (grid_id, "2019-09", 379.0),
        (grid_id, "2020-09", 351.0),
        ("mg_p7_repayment_0", "2019-09", 160.0),
    ):
        refs = _physical_month_refs(source_grid_id, performance_month, y=top)
        # Native detached rows historically carry year/month on the row and
        # only a month column on each source ref (the saved Ye shape).
        for ref in refs:
            ref.pop("performance_month")
        year, month = map(int, performance_month.split("-"))
        records.append({
            "grid_id": source_grid_id,
            "year": year,
            "month": month,
            "source_cell_refs": refs,
        })
    conflicting = deepcopy(records[0])
    conflicting["source_cell_refs"][0]["grid_id"] = "grid:contradictory"
    records.append(conflicting)
    before = deepcopy(records)

    refs = _localized_monthly_source_refs(
        records, None, grid_id=grid_id, year=2019, month=9,
        field_name="performance_month",
    )

    assert len(refs) == 2
    assert {ref["source_field_name"] for ref in refs} == {
        "status", "overdue_amount",
    }
    assert {ref["bbox"][1] for ref in refs} == {379.0, 393.0}
    assert all(ref["performance_month"] == "2019-09" for ref in refs)
    assert records == before


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
    assert closure["expected_source_position_count"] == 4
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
    assert state["expected_row_count"] == 4
    projected = project_personal_detail_datasets(prepared["datasets"])
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 4


def test_source_projection_keeps_evidence_only_alias_across_geometry_variants() -> None:
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
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 2
    assert closure["owner_unresolved_positions"] == 0
    assert closure["alias_source_month_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 0
    assert closure["source_position_balance_valid"] is True


def test_source_projection_deduplicates_exact_geometry_with_distinct_evidence() -> None:
    performance_month = "2024-01"

    def physical_refs(grid_id: str, evidence_id: str) -> list[dict[str, object]]:
        return _physical_month_refs(
            grid_id, performance_month, evidence_prefix=evidence_id
        )

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
    alias = _physical_month_alias_issue(
        "grid:alias",
        refs=physical_refs("grid:alias", "native:monthly:alias"),
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
                        "source_cell_refs": physical_refs(
                            "grid:canonical", "native:monthly:canonical"
                        ),
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


def test_source_projection_keeps_grid_alias_physical_count_after_final_cell_calibration() -> None:
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
    assert closure["raw_source_month_positions"] == 2
    assert closure["unique_physical_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 2
    assert closure["alias_source_month_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 0


def test_source_projection_does_not_treat_a_source_table_as_one_month_cell() -> None:
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
    # A source-table id proves which table calibrated the grid. It is not a
    # physical month-cell identity: these observations have different exact
    # geometry and evidence, even when their source page and table agree.
    assert same_source["raw_source_month_positions"] == 2
    assert same_source["owner_bound_account_months"] == 2
    assert same_source["physical_alias_source_month_observations"] == 0
    assert conflicting_source["source_month_position_observations"] == 2
    assert conflicting_source["raw_source_month_positions"] == 2
    assert conflicting_source["owner_bound_account_months"] == 2
    assert conflicting_source["physical_alias_source_month_observations"] == 0


def test_source_projection_rejects_cross_account_physical_alias_reuse() -> None:
    performance_month = "2024-01"

    def record(account_id: str, grid_id: str) -> dict[str, object]:
        return _physical_month_record(
            grid_id, account_id=account_id, performance_month=performance_month
        )

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
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 0
    assert closure["owner_unresolved_positions"] == 0
    assert closure["owner_conflict_positions"] == 1
    assert closure["owner_conflict_position_observations"] == 2
    assert closure["physical_alias_source_month_observations"] == 1
    assert closure["source_position_balance_valid"] is True
    assert closure["cross_owner_physical_conflict_count"] == 1
    assert closure["physical_owner_conflict_free"] is False
    assert closure["status"] == "physical_owner_conflict"
    assert "_personal_detail_account_month_closure_proof" not in prepared["datasets"]
    assert prepared["facts"]["personal_detail_dataset_states"]["repayment_records"] == {
        "presence_status": "partial",
        "reason": "account_month_physical_owner_conflict",
        "observed_row_count": 2,
        "expected_row_count": 1,
    }
    conflicts = [
        issue
        for issue in prepared["datasets"]["personal_detail_extraction_issues"]
        if issue.get("issue_code") == "account_month_physical_owner_conflict"
    ]
    assert len(conflicts) == 1
    assert conflicts[0]["field_name"] == "account_id"
    assert len(conflicts[0]["source_refs"]) == 4


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
            "source": "native_detail_table",
            "page": page,
            "logical_page": logical_page,
            "source_page": source_page,
            "grid_id": grid_id,
            "performance_month": performance_month,
            "row": 4,
            "col": 1,
            "source_field_name": "status",
            "geometry_scope": "cell",
            "geometry_status": "accepted",
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
        amount_ref = deepcopy(ref)
        amount_ref["row"] = 5
        amount_ref["source_field_name"] = "overdue_amount"
        if include_bbox:
            amount_ref["bbox"] = [x, 20.0, x + 10.0, 30.0]
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
            "source_cell_refs": [ref, amount_ref],
        }

    control = _month_closure(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    record(
                        grid_id,
                        10.0,
                        ref_table_id="table:B",
                        provenance_table_id="table:B",
                    )
                    for grid_id in ("grid:control-a", "grid:control-b")
                ]
            },
        }
    )
    assert control["raw_source_month_positions"] == 1

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
                10.0,
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
                10.0,
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
                10.0,
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
                10.0,
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
                10.0,
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
                10.0,
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
                10.0,
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
        source_refs = _physical_month_refs(
            grid_id,
            performance_month,
            page=page,
            x=float(index),
            y=100.0,
            evidence_prefix=evidence_id,
        )
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
                "source_cell_refs": source_refs,
            }
        )
        if index < 5:
            alias_grid_id = f"grid:alias:{index}"
            aliases.append(
                _physical_month_alias_issue(
                    alias_grid_id,
                    account_id=account_id,
                    performance_month=performance_month,
                    refs=[{**ref, "grid_id": alias_grid_id} for ref in source_refs],
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

    def physical_refs(grid_id: str) -> list[dict[str, object]]:
        return _physical_month_refs(grid_id, performance_month)

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
    alias = _physical_month_alias_issue("grid:alias")
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
        source_refs=physical_refs("grid:detached"),
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
                        "source_cell_refs": physical_refs("grid:canonical"),
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
    detached_source_page: int | None = None,
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
                "source": "native_detail_table",
                "page": 10,
                "logical_page": 10,
                "table_id": "table:detached-replay",
                "geometry_scope": "cell",
                "geometry_status": "accepted",
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

    detached_refs = cell_refs(detached_grid_id, source_page=detached_source_page)
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


def test_source_projection_keeps_detached_pair_when_source_page_is_missing() -> None:
    closure = _month_closure(_detached_source_page_replay_content())

    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 0
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
            _detached_source_page_replay_content(
                geometry_state=geometry_state, detached_source_page=5
            )
        )

        assert closure["source_month_position_observations"] == 2, geometry_state
        assert closure["raw_source_month_positions"] == 2, geometry_state
        assert closure["owner_bound_account_months"] == 1, geometry_state
        assert closure["owner_unresolved_positions"] == 1, geometry_state
        assert closure["reconciled_detached_diagnostic_positions"] == 0, geometry_state
        assert closure["source_position_balance_valid"] is True, geometry_state


def test_source_projection_does_not_trust_unrelated_issue_geometry_for_bridge() -> None:
    content = _detached_source_page_replay_content(detached_source_page=5)
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
    content = _detached_source_page_replay_content(detached_source_page=5)
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

    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["owner_conflict_positions"] == 0
    assert closure["physical_alias_source_month_observations"] == 0
    assert closure["alias_source_month_positions"] == 0
    assert closure["unresolved_detector_source_position_collisions"] == 0
    assert closure["source_position_balance_valid"] is True
    assert closure["status"] == "partial_owner_unresolved"


def test_source_projection_deduplicates_only_independently_bound_identity_only_claims() -> None:
    for scenario in ("same_owner", "same_owner_alias", "new_owner", "range_only"):
        for reverse_issues in (False, True):
            account_id = "account:new" if scenario == "new_owner" else "account:1"
            owner_key = stable_record_id(
                "source_account_month_owner", account_id
            ).split(":", 1)[-1]
            issues = []
            for ref in _physical_month_refs("grid:source"):
                claim_ref = {
                    **ref,
                    "account_id": account_id,
                    "binding": "source_account_month_range",
                }
                issues.append(make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_account_range_missing_month",
                    message="An exact range proves identity, not another physical pair.",
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id=f"source_account_month:{owner_key}:2024-01",
                    field_name="performance_month",
                    observed_value={
                        "account_id": account_id,
                        "grid_id": "grid:source",
                        "performance_month": "2024-01",
                        "source_identity_type": "account_month_from_printed_repayment_range",
                    },
                    source_refs=[claim_ref],
                ))
            if scenario == "range_only":
                # An anonymous complete pair does not independently establish
                # the claimed owner.  Even matching range singletons must not
                # supply that missing ownership or collapse the opaque claim.
                records = []
                issues.append(_physical_month_structural_issue("grid:source"))
            elif scenario == "same_owner_alias":
                records = []
                issues.append(_physical_month_alias_issue("grid:source"))
            else:
                records = [_physical_month_record("grid:source")]
            if reverse_issues:
                issues.reverse()
            content = {
                "facts": {},
                "datasets": {
                    "repayment_records": records,
                    "personal_detail_extraction_issues": issues,
                },
            }
            closure = _month_closure(content)
            label = (scenario, reverse_issues)
            independently_bound = scenario in {"same_owner", "same_owner_alias"}
            expected_positions = 1 if independently_bound else 2
            assert closure["source_month_position_observations"] == expected_positions, label
            assert closure["raw_source_month_positions"] == expected_positions, label
            assert closure["owner_bound_account_months"] == (2 if scenario == "new_owner" else 1), label
            assert closure["owner_unresolved_positions"] == int(scenario == "range_only"), label
            assert closure["owner_conflict_positions"] == 0, label
            assert closure["physical_alias_source_month_observations"] == 0, label
            assert closure["alias_source_month_positions"] == int(scenario == "same_owner_alias"), label
            assert closure["source_position_balance_valid"] is True, label
            assert closure["unresolved_detector_source_position_collisions"] == int(not independently_bound), label
            if independently_bound:
                assert closure["status"] == "identity_closed", label
                assert closure["expected_identity_count"] == 1, label
            else:
                assert closure["status"] == "source_localization_invalid", label
                assert "_personal_detail_account_month_closure_proof" not in content["datasets"], label


def test_source_projection_requires_source_page_on_each_inventoried_cell() -> None:
    content = _detached_source_page_replay_content(detached_source_page=5)
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
    assert "_personal_detail_account_month_closure_proof" not in prepared["datasets"]


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
    performance_month: str = "2020-01",
    primary_exact_geometry: bool = False,
) -> dict[str, object]:
    """Model Yang page 8 without depending on private OCR fixtures."""

    account_id = "credit_account:non_revolving_loan:15"
    month_number = int(performance_month[-2:])
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
    primary_cell_refs: list[dict[str, object]] = [
        {
            "page": 8,
            "logical_page": 8,
            "geometry_scope": "logical_page",
            "geometry_status": "unresolved",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": primary_grid_id,
            "row": 2,
            "col": month_number,
            "field_name": "status",
            "geometry_rejection": {
                "source": "rejected_month_geometry",
                "reason": "source_table_month_ownership_required",
                "logical_page": 8,
                "value_inputs_used": False,
            },
        }
    ]
    if primary_exact_geometry:
        left = 66.41666666666667 + 25.08333333333333 * month_number
        right = 91.5 + 25.08333333333333 * month_number
        primary_cell_refs = [
            {
                "page": 8,
                "logical_page": 8,
                "geometry_scope": "cell",
                "coordinate_system": "pdf_points_top_left",
                "grid_id": primary_grid_id,
                "row": row,
                "col": month_number,
                "field_name": field_name,
                "performance_month": performance_month,
                "bbox": [left, top, right, bottom],
            }
            for row, field_name, top, bottom in (
                (2, "status", 197.5, 210.5),
                (3, "overdue_amount", 210.5, 223.5),
            )
        ]
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
        "source_cell_refs": primary_cell_refs,
    }
    alias_refs = [
        {
            "page": 8,
            "logical_page": 8,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": alias_grid_id,
            "row": row,
            "col": month_number,
            "field_name": "performance_month",
            "source_field_name": source_field_name,
            "performance_month": performance_month,
            "bbox": bbox,
            "account_id": account_id,
            "binding": "source_account_month_alias",
            "binding_quality": "source_account_month_alias",
        }
        for row, source_field_name, bbox in (
            (
                2,
                "status",
                [
                    66.41666666666667 + 25.08333333333333 * month_number,
                    287.5,
                    91.5 + 25.08333333333333 * month_number,
                    304.5,
                ],
            ),
            (
                3,
                "overdue_amount",
                [
                    66.41666666666667 + 25.08333333333333 * month_number,
                    301.0,
                    91.5 + 25.08333333333333 * month_number,
                    317.0,
                ],
            ),
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


def test_source_projection_binds_asymmetric_alias_without_inventing_physical_identity() -> None:
    for status in ("informational", "resolved"):
        closure = _month_closure(
            _asymmetric_detached_month_alias_content(alias_status=status)
        )
        assert closure["candidate_identity_count"] == 1
        assert closure["source_month_position_observations"] == 2
        assert closure["raw_source_month_positions"] == 2
        assert closure["unique_physical_source_month_positions"] == 2
        assert closure["owner_bound_account_months"] == 2
        assert closure["owner_unresolved_positions"] == 0
        assert closure["alias_source_month_positions"] == 1
        assert closure["physical_alias_source_month_observations"] == 0
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
        assert _month_closure(content)["raw_source_month_positions"] == 2

    order_insensitive = _asymmetric_detached_month_alias_content()
    issues = order_insensitive["datasets"]["personal_detail_extraction_issues"]  # type: ignore[index]
    issues[1]["source_refs"].reverse()
    assert _month_closure(order_insensitive)["raw_source_month_positions"] == 2

    wrapped = _asymmetric_detached_month_alias_content()
    wrapped_issues = wrapped["datasets"]["personal_detail_extraction_issues"]  # type: ignore[index]
    for index, issue in enumerate(wrapped_issues):
        refs = issue.pop("source_refs")
        wrapped_issues[index] = {
            "normalized": issue,
            "source": {"source_refs": refs},
        }
    assert _month_closure(wrapped)["raw_source_month_positions"] == 2

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
    assert _month_closure(production_family)["raw_source_month_positions"] == 2


def test_source_projection_counts_ye_shaped_nine_month_detached_block() -> None:
    months = [
        *(f"2023-{month:02d}" for month in range(7, 13)),
        *(f"2024-{month:02d}" for month in range(1, 4)),
    ]
    contents = [
        _asymmetric_detached_month_alias_content(
            performance_month=month,
            primary_exact_geometry=True,
        )
        for month in months
    ]
    closure = _month_closure(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    row
                    for content in contents
                    for row in content["datasets"]["repayment_records"]  # type: ignore[index]
                ],
                "personal_detail_extraction_issues": [
                    row
                    for content in contents
                    for row in content["datasets"]["personal_detail_extraction_issues"]  # type: ignore[index]
                ],
            },
        }
    )

    assert closure["candidate_identity_count"] == 9
    assert closure["canonical_account_month_identity_count"] == 9
    assert closure["source_month_position_observations"] == 18
    assert closure["raw_source_month_positions"] == 18
    assert closure["owner_bound_account_months"] == 18
    assert closure["alias_source_month_positions"] == 9
    assert closure["physical_alias_source_month_observations"] == 0


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
    assert cross_owner_closure["owner_bound_account_months"] == 1
    assert cross_owner_closure["owner_unresolved_positions"] == 0
    assert cross_owner_closure["owner_conflict_positions"] == 1
    assert cross_owner_closure["source_position_balance_valid"] is True
    assert cross_owner_closure["cross_owner_physical_conflict_count"] == 1
    assert cross_owner_closure["physical_owner_conflict_free"] is False


def test_source_projection_complete_pair_alias_is_order_and_duplicate_invariant() -> None:
    for variant in ("original", "reversed", "duplicate", "provenance_only"):
        refs = _physical_month_refs("grid:alias", evidence_prefix="other-evidence")
        if variant == "reversed":
            refs.reverse()
        elif variant == "duplicate":
            refs.extend(deepcopy(refs))
        elif variant == "provenance_only":
            for ref in refs:
                ref.pop("geometry_status")
        closure = _month_closure(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": [
                        _physical_month_record("grid:primary"),
                        _physical_month_record("grid:alias", refs=refs),
                    ]
                },
            }
        )
        assert closure["source_month_position_observations"] == 2, variant
        assert closure["raw_source_month_positions"] == 1, variant
        assert closure["owner_bound_account_months"] == 1, variant
        assert closure["owner_unresolved_positions"] == 0, variant
        assert closure["owner_conflict_positions"] == 0, variant
        assert closure["physical_alias_source_month_observations"] == 1, variant
        assert closure["source_position_balance_valid"] is True, variant


def test_source_projection_enriched_alias_metadata_preserves_exact_owner_declaration() -> None:
    for variant in ("direct", "reversed", "wrapped"):
        alias = _physical_month_alias_issue("grid:alias")
        if variant == "reversed":
            alias["source_refs"].reverse()
        elif variant == "wrapped":
            refs = alias.pop("source_refs")
            alias = {"normalized": alias, "source": {"source_refs": refs}}
        content = {
            "facts": {},
            "datasets": {
                # The declaration must bind ownership independently of physical
                # deduplication; this primary has no usable pair geometry.
                "repayment_records": [_physical_month_record("grid:primary", refs=[])],
                "personal_detail_extraction_issues": [alias],
            },
        }
        closure = _month_closure(content)
        assert closure["source_month_position_observations"] == 2, variant
        assert closure["raw_source_month_positions"] == 2, variant
        assert closure["owner_bound_account_months"] == 2, variant
        assert closure["owner_unresolved_positions"] == 0, variant
        assert closure["alias_source_month_positions"] == 1, variant
        assert closure["physical_alias_source_month_observations"] == 0, variant
        assert closure["source_position_balance_valid"] is True, variant


def test_source_projection_enriched_alias_metadata_is_not_an_open_grammar() -> None:
    for variant in (
        "unknown_key",
        "rejected_geometry",
        "conflicting_origin",
        "source_page_conflict",
        "table_conflict",
        "duplicate_evidence",
        "nonstring_evidence",
        "wrong_binding",
    ):
        alias = _physical_month_alias_issue("grid:alias")
        first = alias["source_refs"][0]
        if variant == "unknown_key":
            first["untrusted_owner_override"] = "account:1"
        elif variant == "rejected_geometry":
            first["geometry_status"] = "rejected"
        elif variant == "conflicting_origin":
            first["source_origin"] = "scan:A"
            first["geometry_provenance"]["source_origin"] = "scan:B"
        elif variant == "source_page_conflict":
            first["geometry_provenance"]["source_page"] = 5
        elif variant == "table_conflict":
            first["geometry_provenance"]["table_id"] = "table:other"
        elif variant == "duplicate_evidence":
            first["evidence_ids"] *= 2
        elif variant == "nonstring_evidence":
            first["evidence_ids"] = [True]
        elif variant == "wrong_binding":
            first["binding"] = "unconfirmed_account_month_alias"
        closure = _month_closure(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": [_physical_month_record("grid:primary", refs=[])],
                    "personal_detail_extraction_issues": [alias],
                },
            }
        )
        assert closure["raw_source_month_positions"] == 2, variant
        assert closure["owner_bound_account_months"] == 1, variant
        assert closure["owner_unresolved_positions"] == 1, variant
        assert closure["alias_source_month_positions"] == 0, variant
        assert closure["source_position_balance_valid"] is True, variant


def test_source_projection_does_not_merge_transitive_single_cell_chains() -> None:
    records = []
    # (S1, A1), (S1, A2), and (S2, A2) share individual cells but are three
    # physical pairs.  Only the fourth observation is an exact alias of pair 2.
    for grid_id, status_top, amount_top in (
        ("grid:one", 300.0, 314.0),
        ("grid:two", 300.0, 330.0),
        ("grid:three", 310.0, 330.0),
        ("grid:two-alias", 300.0, 330.0),
    ):
        refs = _physical_month_refs(grid_id)
        refs[0]["bbox"] = [120.0, status_top, 145.0, status_top + 14.0]
        refs[1]["bbox"] = [120.0, amount_top, 145.0, amount_top + 14.0]
        records.append(_physical_month_record(grid_id, refs=refs))
    for ordered_records in (records, list(reversed(records))):
        closure = _month_closure(
            {"facts": {}, "datasets": {"repayment_records": deepcopy(ordered_records)}}
        )
        assert closure["source_month_position_observations"] == 4
        assert closure["raw_source_month_positions"] == 3
        assert closure["owner_bound_account_months"] == 3
        assert closure["owner_unresolved_positions"] == 0
        assert closure["owner_conflict_positions"] == 0
        assert closure["physical_alias_source_month_observations"] == 1
        assert closure["source_position_balance_valid"] is True


def test_source_projection_conserves_distinct_pairs_under_one_reused_detector_id() -> None:
    records = [
        _physical_month_record("grid:ambiguous"),
        _physical_month_record(
            "grid:ambiguous", refs=_physical_month_refs("grid:ambiguous", x=220.0)
        ),
        _physical_month_record("grid:alias"),
    ]
    for ordered_records in (records, list(reversed(records))):
        closure = _month_closure(
            {"facts": {}, "datasets": {"repayment_records": deepcopy(ordered_records)}}
        )
        # Two proven pairs under one detector key are two observations.  The
        # third record repeats pair 1, not an ambiguous choice between 1 and 2.
        assert closure["source_month_position_observations"] == 3
        assert closure["raw_source_month_positions"] == 2
        assert closure["owner_bound_account_months"] == 2
        assert closure["physical_alias_source_month_observations"] == 1
        assert closure["owner_conflict_positions"] == 0
        assert closure["source_position_balance_valid"] is True


def test_source_projection_reused_detector_id_does_not_mix_proven_owners() -> None:
    for third_owner in (None, "account:A", "account:C"):
        records = [
            _physical_month_record("grid:reused", account_id="account:A"),
            _physical_month_record(
                "grid:reused", account_id="account:B",
                refs=_physical_month_refs("grid:reused", x=220.0),
            ),
        ]
        if third_owner is not None:
            records.append(_physical_month_record("grid:alias", account_id=third_owner))
        for ordered_records in (records, list(reversed(records))):
            content = {"facts": {}, "datasets": {"repayment_records": deepcopy(ordered_records)}}
            closure = _month_closure(content)
            conflict_expected = third_owner == "account:C"
            assert closure["source_month_position_observations"] == len(records)
            assert closure["raw_source_month_positions"] == 2
            assert closure["owner_bound_account_months"] == (1 if conflict_expected else 2)
            assert closure["owner_unresolved_positions"] == 0
            assert closure["owner_conflict_positions"] == int(conflict_expected)
            assert closure["physical_alias_source_month_observations"] == len(records) - 2
            assert closure["unresolved_detector_source_position_collisions"] == 0
            assert closure["source_position_balance_valid"] is True
            if conflict_expected:
                conflicts = [
                    issue for issue in content["datasets"]["personal_detail_extraction_issues"]
                    if issue.get("issue_code") == "account_month_physical_owner_conflict"
                ]
                assert len(conflicts) == 1
                assert conflicts[0]["observed_value"]["claimed_account_month_identities"] == [
                    {"account_id": "account:A", "performance_month": "2024-01"},
                    {"account_id": "account:C", "performance_month": "2024-01"},
                ]
                assert conflicts[0]["observed_value"]["shared_complete_status_amount_pair_count"] == 1
                assert "_personal_detail_account_month_closure_proof" not in content["datasets"]
            else:
                assert closure["status"] == "identity_closed"
                assert content["facts"]["personal_detail_dataset_states"]["repayment_records"]["presence_status"] == "present"


def test_source_projection_keeps_deficient_reused_id_claims_explicit() -> None:
    for owner in ("account:A", "account:B"):
        content = {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    _physical_month_record("grid:reused", account_id="account:A"),
                    _physical_month_record("grid:reused", account_id=owner, refs=[]),
                ],
            },
        }
        closure = _month_closure(content)
        assert closure["source_month_position_observations"] == 2
        assert closure["raw_source_month_positions"] == 2
        assert closure["owner_bound_account_months"] == 2
        assert closure["owner_conflict_positions"] == 0
        assert closure["physical_alias_source_month_observations"] == 0
        assert closure["unresolved_detector_source_position_collisions"] == 1
        assert closure["physical_source_position_localization_valid"] is False
        assert closure["status"] == "source_localization_invalid"
        assert closure["source_position_balance_valid"] is True
        assert "_personal_detail_account_month_closure_proof" not in content["datasets"]
        assert content["facts"]["personal_detail_dataset_states"]["repayment_records"]["presence_status"] == "partial"


def test_source_projection_field_fragments_cannot_create_pairs_or_claim_owners() -> None:
    for detached_grid in ("grid:primary", "grid:detached"):
        refs = _physical_month_refs(detached_grid)
        content = {
            "facts": {},
            "datasets": {
                "repayment_records": [_physical_month_record("grid:primary")],
                "personal_detail_extraction_issues": [
                    _physical_month_structural_issue(detached_grid, refs=[ref])
                    for ref in refs
                ],
            },
        }
        closure = _month_closure(content)
        same_detector = detached_grid == "grid:primary"
        # Same-detector field diagnostics can be placed inside the independently
        # complete canonical pair.  Two different-detector singleton issues
        # cannot synthesize a pair to become a physical alias of that canonical.
        assert closure["raw_source_month_positions"] == (1 if same_detector else 2)
        assert closure["owner_bound_account_months"] == 1
        assert closure["owner_unresolved_positions"] == (0 if same_detector else 1)
        assert closure["physical_alias_source_month_observations"] == 0
        assert closure["source_position_balance_valid"] is True


def test_source_projection_invalid_pair_cannot_reenter_as_field_fragments() -> None:
    for variant in ("nonadjacent", "upstream_coalesced_pairs"):
        invalid_pair = _physical_month_refs("grid:primary")
        if variant == "nonadjacent":
            invalid_pair[1]["row"] += 2
        else:
            # An upstream issue-ID deduplication may merge two complete ref
            # containers.  This consumer must not pick a convenient subset or
            # certify that the original coherent observations were preserved.
            invalid_pair.extend(_physical_month_refs("grid:primary", x=220.0))
        content = {
            "facts": {},
            "datasets": {
                "repayment_records": [_physical_month_record("grid:primary")],
                "personal_detail_extraction_issues": [
                    _physical_month_structural_issue("grid:primary", refs=invalid_pair),
                ],
            },
        }
        closure = _month_closure(content)
        assert closure["raw_source_month_positions"] == 2, variant
        assert closure["owner_bound_account_months"] == 1, variant
        assert closure["owner_unresolved_positions"] == 1, variant
        assert closure["unresolved_detector_source_position_collisions"] == 1, variant
        assert closure["physical_alias_source_month_observations"] == 0, variant
        assert closure["source_position_balance_valid"] is True, variant
        assert "_personal_detail_account_month_closure_proof" not in content["datasets"], variant


def test_source_projection_reconciles_every_structural_pair_under_reused_detector_id() -> None:
    first_refs = _physical_month_refs("grid:reused")
    second_refs = _physical_month_refs("grid:reused", x=220.0)
    aggregate = make_issue(
        category="schema_incompleteness",
        issue_code="canonical_monthly_reconstruction_incomplete",
        message="Two independently observed source-table positions need reconciliation.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        field_name="performance_month",
        candidate_value={"unreconciled_source_position_count": 2},
    )
    details = [
        _physical_month_structural_issue("grid:reused", refs=first_refs),
        _physical_month_structural_issue("grid:reused", refs=second_refs),
        # Field siblings are not additional source positions once their own
        # detail plane already contains the complete pair above.
        *[
            _physical_month_structural_issue("grid:reused", refs=[ref])
            for ref in second_refs
        ],
    ]
    content = {
        "facts": {},
        "datasets": {
            "repayment_records": [_physical_month_record("grid:reused", refs=first_refs)],
            "personal_detail_extraction_issues": [aggregate, *details],
        },
    }
    closure = _month_closure(content)
    assert closure["source_month_position_observations"] == 2
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 0
    assert closure["source_position_balance_valid"] is True
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 1
    assert aggregate["status"] == "requires_review"
    basis = deepcopy(aggregate["candidate_value"]["source_position_reconciliation"])
    assert basis["source_position_observation_count"] == 2

    content["datasets"]["repayment_records"].append(
        _physical_month_record("grid:reused", account_id="account:B", refs=second_refs)
    )
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 2
    assert closure["owner_unresolved_positions"] == 0
    assert closure["owner_conflict_positions"] == 0
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 0
    assert aggregate["candidate_value"]["source_position_reconciliation"] == basis
    assert aggregate["status"] == "resolved"

    conflicting_refs = _physical_month_refs("grid:alias", x=220.0)
    content["datasets"]["repayment_records"].append(
        _physical_month_record("grid:alias", account_id="account:C", refs=conflicting_refs)
    )
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_conflict_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 1
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 1
    assert aggregate["candidate_value"]["source_position_reconciliation"] == basis
    assert aggregate["status"] == "requires_review"
    assert "_personal_detail_account_month_closure_proof" not in content["datasets"]


def test_source_projection_reused_id_does_not_authenticate_collapsed_aggregate() -> None:
    aggregate = make_issue(
        category="schema_incompleteness",
        issue_code="canonical_monthly_reconstruction_incomplete",
        message="The earlier detector-key count incorrectly collapsed two tables.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        candidate_value={"unreconciled_source_position_count": 1},
    )
    second_refs = _physical_month_refs("grid:reused", x=220.0)
    content = {
        "facts": {},
        "datasets": {
            "repayment_records": [
                _physical_month_record("grid:reused"),
                _physical_month_record("grid:reused", account_id="account:B", refs=second_refs),
            ],
            "personal_detail_extraction_issues": [
                aggregate,
                _physical_month_structural_issue("grid:reused"),
                _physical_month_structural_issue("grid:reused", refs=second_refs),
            ],
        },
    }
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 2
    assert closure["source_position_balance_valid"] is True
    assert closure["source_position_reconciliation_valid"] is False
    assert closure["status"] == "source_localization_invalid"
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 1
    assert aggregate["candidate_value"]["account_month_expected_row_count"] is None
    assert aggregate["status"] == "requires_review"
    assert "_personal_detail_account_month_closure_proof" not in content["datasets"]


def test_source_projection_origin_is_not_a_producer_method_tag() -> None:
    for variant in ("different_methods", "absent_methods", "legacy_same_page", "origin_alias", "registered"):
        primary = _physical_month_refs("grid:primary")
        alias = _physical_month_refs("grid:alias")
        for ref in alias:
            provenance = ref["geometry_provenance"]
            if variant == "different_methods":
                ref["source"] = "canonical_monthly_source_structure_cell"
                provenance["source"] = "accepted_source_table_copy"
            elif variant == "absent_methods":
                ref.pop("source")
                provenance.pop("source")
            elif variant == "legacy_same_page":
                ref.pop("source_logical_page")
                provenance.pop("source_logical_page")
            elif variant == "origin_alias":
                ref["source_origin_logical_page"] = ref.pop("source_logical_page")
                provenance["source_origin_logical_page"] = provenance.pop("source_logical_page")
        if variant == "registered":
            for ref in [*primary, *alias]:
                ref["source_logical_page"] = 5
                ref["geometry_provenance"]["source_logical_page"] = 5
                ref["geometry_provenance"]["visual_source_to_coordinate_affine"] = {
                    "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 14.0,
                }
        closure = _month_closure({
            "facts": {},
            "datasets": {"repayment_records": [
                _physical_month_record("grid:primary", refs=primary),
                _physical_month_record("grid:alias", refs=alias),
            ]},
        })
        assert closure["raw_source_month_positions"] == 1, variant
        assert closure["owner_bound_account_months"] == 1, variant
        assert closure["physical_alias_source_month_observations"] == 1, variant
        assert closure["source_position_balance_valid"] is True, variant


def test_source_projection_preserves_distinct_or_contradictory_actual_origins() -> None:
    for variant in ("different_logical_origin", "different_origin_label", "contradictory_logical", "contradictory_label", "boolean_origin"):
        primary = _physical_month_refs("grid:primary")
        alias = _physical_month_refs("grid:alias")
        if variant == "different_origin_label":
            for ref in primary:
                ref["source_origin"] = "scan:A"
        for ref in alias:
            provenance = ref["geometry_provenance"]
            if variant == "different_logical_origin":
                ref["source_logical_page"] = 8
                provenance["source_logical_page"] = 8
            elif variant == "different_origin_label":
                ref["source_origin"] = "scan:B"
            elif variant == "contradictory_logical":
                ref["source_origin_logical_page"] = 8
            elif variant == "contradictory_label":
                ref["source_origin"] = "scan:A"
                provenance["source_origin"] = "scan:B"
            else:
                ref["source_origin_logical_page"] = True
        closure = _month_closure({
            "facts": {},
            "datasets": {"repayment_records": [
                _physical_month_record("grid:primary", refs=primary),
                _physical_month_record("grid:alias", refs=alias),
            ]},
        })
        assert closure["raw_source_month_positions"] == 2, variant
        assert closure["physical_alias_source_month_observations"] == 0, variant
        assert closure["source_position_balance_valid"] is True, variant


def test_source_projection_partitions_one_inventory_into_bound_unresolved_and_conflict() -> None:
    content = {
        "facts": {},
        "datasets": {
            "repayment_records": [
                _physical_month_record("grid:bound"),
                _physical_month_record(
                    "grid:conflict-a",
                    account_id="account:A",
                    refs=_physical_month_refs("grid:conflict-a", x=300.0),
                ),
                _physical_month_record(
                    "grid:conflict-b",
                    account_id="account:B",
                    refs=_physical_month_refs("grid:conflict-b", x=300.0),
                ),
            ],
            "personal_detail_extraction_issues": [
                _physical_month_structural_issue("grid:detached-bound"),
                _physical_month_structural_issue(
                    "grid:unresolved",
                    refs=_physical_month_refs("grid:unresolved", x=200.0),
                ),
            ],
        },
    }
    closure = _month_closure(content)
    assert closure["source_month_position_observations"] == 5
    assert closure["raw_source_month_positions"] == 3
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 1
    assert closure["owner_conflict_positions"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 2
    assert closure["owner_conflict_position_observations"] == 2
    assert closure["source_position_balance_valid"] is True
    assert closure["raw_source_month_positions"] == sum(
        closure[key]
        for key in (
            "owner_bound_account_months",
            "owner_unresolved_positions",
            "owner_conflict_positions",
        )
    )
    assert "_personal_detail_account_month_closure_proof" not in content["datasets"]


def test_source_projection_explicit_geometry_rejection_vetoes_stale_exact_provenance() -> None:
    for status in ("rejected", "unresolved", "ambiguous", "", None, False, 1):
        for duplicate_accepted_role in (False, True):
            refs = _physical_month_refs("grid:rejected")
            rejected = deepcopy(refs[0])
            rejected["geometry_status"] = status
            if duplicate_accepted_role:
                refs.append(rejected)
            else:
                refs[0] = rejected
            closure = _month_closure(
                {
                    "facts": {},
                    "datasets": {
                        "repayment_records": [
                            _physical_month_record("grid:primary"),
                            _physical_month_record("grid:rejected", refs=refs),
                        ]
                    },
                }
            )
            label = (status, duplicate_accepted_role)
            assert closure["raw_source_month_positions"] == 2, label
            assert closure["owner_bound_account_months"] == 2, label
            assert closure["physical_alias_source_month_observations"] == 0, label
            assert closure["source_position_balance_valid"] is True, label


def test_source_projection_requires_complete_explicit_physical_pair_metadata() -> None:
    variants = (
        "registered_origin_missing",
        "missing_source_page",
        "conflicting_source_page",
        "missing_coordinates",
        "wrong_coordinates",
        "coordinate_page_mismatch",
        "boolean_page",
        "table_mismatch",
        "missing_role",
        "wrong_role_order",
        "nonadjacent_rows",
        "misaligned_amount",
        "nearly_equal_pair",
        "false_month",
        "zero_month",
        "list_month",
        "extra_locator",
        "contradictory_role_duplicate",
    )
    for variant in variants:
        refs = _physical_month_refs("grid:probe")
        first = refs[0]
        provenance = first["geometry_provenance"]
        if variant == "registered_origin_missing":
            first.pop("source_logical_page")
            provenance.pop("source_logical_page")
            provenance["visual_source_to_coordinate_affine"] = {
                "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0,
            }
        elif variant == "missing_source_page":
            first.pop("source_page")
            provenance.pop("source_page")
        elif variant == "conflicting_source_page":
            provenance["source_page"] = 5
        elif variant == "missing_coordinates":
            first.pop("coordinate_system")
            provenance.pop("coordinate_system")
        elif variant == "wrong_coordinates":
            first["coordinate_system"] = "image_pixels_top_left"
        elif variant == "coordinate_page_mismatch":
            first["logical_page"] = 8
        elif variant == "boolean_page":
            first["page"] = True
        elif variant == "table_mismatch":
            provenance["table_id"] = "other-table"
        elif variant == "missing_role":
            first.pop("source_field_name")
        elif variant == "wrong_role_order":
            first["source_field_name"] = "overdue_amount"
            refs[1]["source_field_name"] = "status"
        elif variant == "nonadjacent_rows":
            refs[1]["row"] = 6
        elif variant == "misaligned_amount":
            refs[1]["bbox"] = [121.0, 314.0, 146.0, 328.0]
        elif variant == "nearly_equal_pair":
            for ref in refs:
                left, top, right, bottom = ref["bbox"]
                ref["bbox"] = [left + 0.0000001, top, right + 0.0000001, bottom]
        elif variant in {"false_month", "zero_month", "list_month"}:
            first["performance_month"] = {
                "false_month": False,
                "zero_month": 0,
                "list_month": ["2024-01"],
            }[variant]
        elif variant == "extra_locator":
            extra = deepcopy(first)
            extra["source_page"] = 5
            extra["geometry_provenance"]["source_page"] = 5
            refs.append(extra)
        elif variant == "contradictory_role_duplicate":
            extra = deepcopy(first)
            extra["field_name"] = "status_amount"
            refs.append(extra)
        closure = _month_closure(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": [
                        _physical_month_record("grid:primary"),
                        _physical_month_record("grid:probe", refs=refs),
                    ]
                },
            }
        )
        assert closure["raw_source_month_positions"] == 2, variant
        assert closure["physical_alias_source_month_observations"] == 0, variant
        assert closure["source_position_balance_valid"] is True, variant


def test_source_projection_does_not_stitch_singletons_across_observations() -> None:
    for variant in ("record_fields", "wrapped_record", "record_and_issue", "issue_rows"):
        refs = _physical_month_refs("grid:split")
        split_record = _physical_month_record("grid:split", refs=[])
        issues = []
        if variant == "record_fields":
            split_record["source_cell_refs"] = refs[:1]
            split_record["source_refs"] = refs[1:]
        elif variant == "wrapped_record":
            split_record["source_cell_refs"] = refs[:1]
            split_record = {
                "normalized": split_record,
                "source": {"source_refs": refs[1:]},
            }
        else:
            if variant == "record_and_issue":
                split_record["source_cell_refs"] = refs[:1]
                issue_refs = refs[1:]
            else:
                issue_refs = refs
            for ref in issue_refs:
                issues.append(
                    make_issue(
                        category="ocr_structure_correction",
                        issue_code=f"unrelated_probe_{ref['source_field_name']}",
                        message="A singleton diagnostic is not a pair observation.",
                        parser_stage="unrelated_probe",
                        target_dataset="repayment_records",
                        target_record_id="grid:split:2024-01",
                        field_name="performance_month",
                        observed_value={
                            "account_id": "account:1",
                            "grid_id": "grid:split",
                            "performance_month": "2024-01",
                        },
                        source_refs=[ref],
                    )
                )
        closure = _month_closure(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": [
                        _physical_month_record("grid:primary"),
                        split_record,
                    ],
                    "personal_detail_extraction_issues": issues,
                },
            }
        )
        assert closure["source_month_position_observations"] == 2, variant
        assert closure["raw_source_month_positions"] == 2, variant
        assert closure["owner_bound_account_months"] == 2, variant
        assert closure["physical_alias_source_month_observations"] == 0, variant

    refs = _physical_month_refs("grid:split-issue")
    issue = _physical_month_structural_issue("grid:split-issue", refs=refs[:1])
    closure = _month_closure(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [_physical_month_record("grid:primary")],
                "personal_detail_extraction_issues": [
                    {"normalized": issue, "source": {"source_refs": refs[1:]}}
                ],
            },
        }
    )
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 0


def test_source_projection_uses_coherent_wrapped_observations_without_unrelated_geometry() -> None:
    for wrapped_owner in ("record", "issue"):
        refs = _physical_month_refs("grid:wrapped")
        records = [_physical_month_record("grid:primary")]
        issues = []
        if wrapped_owner == "record":
            records.append(
                {
                    "normalized": _physical_month_record("grid:wrapped", refs=[]),
                    "source": {"source_refs": refs},
                }
            )
        else:
            issue = _physical_month_structural_issue("grid:wrapped", refs=refs)
            issue.pop("source_refs")
            issues.append({"normalized": issue, "source": {"source_refs": refs}})
        closure = _month_closure(
            {
                "facts": {},
                "datasets": {
                    "repayment_records": records,
                    "personal_detail_extraction_issues": issues,
                },
            }
        )
        assert closure["raw_source_month_positions"] == 1, wrapped_owner
        assert closure["owner_bound_account_months"] == 1, wrapped_owner
        assert closure["owner_unresolved_positions"] == 0, wrapped_owner

    unrelated = make_issue(
        category="ocr_structure_correction",
        issue_code="unrelated_probe",
        message="An unrelated diagnostic cannot supply canonical owner geometry.",
        parser_stage="unrelated_probe",
        target_dataset="repayment_records",
        target_record_id="grid:primary:2024-01",
        field_name="performance_month",
        observed_value={
            "account_id": "account:1",
            "grid_id": "grid:primary",
            "performance_month": "2024-01",
        },
        source_refs=_physical_month_refs("grid:primary"),
    )
    closure = _month_closure(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [_physical_month_record("grid:primary", refs=[])],
                "personal_detail_extraction_issues": [
                    unrelated,
                    _physical_month_structural_issue("grid:detached"),
                ],
            },
        }
    )
    assert closure["raw_source_month_positions"] == 2
    assert closure["owner_bound_account_months"] == 1
    assert closure["owner_unresolved_positions"] == 1
    assert closure["reconciled_detached_diagnostic_positions"] == 0


def _structural_reconciliation_content(
    *, reported_count: object = 2
) -> tuple[dict[str, object], dict[str, object]]:
    aggregate = make_issue(
        category="schema_incompleteness",
        issue_code="canonical_monthly_reconstruction_incomplete",
        message="Two source positions require owner reconciliation.",
        parser_stage="canonical_monthly_grid_materialization",
        target_dataset="repayment_records",
        candidate_value={
            "unreconciled_source_position_count": reported_count,
            "account_month_expected_row_count": None,
            "localization_status": "account_month_owner_reconciliation_pending",
        },
    )
    content = {
        "facts": {},
        "datasets": {
            "repayment_records": [_physical_month_record("grid:primary")],
            "personal_detail_extraction_issues": [
                _physical_month_structural_issue("grid:known"),
                _physical_month_structural_issue(
                    "grid:unknown", refs=_physical_month_refs("grid:unknown", x=220.0)
                ),
                _physical_month_alias_issue("grid:known"),
                aggregate,
            ],
        },
    }
    return content, aggregate


def test_source_projection_refreshes_reconciliation_and_reopens_conflicts() -> None:
    content, aggregate = _structural_reconciliation_content()
    issue_id = aggregate["extraction_issue_id"]
    _month_closure(content)
    candidate = aggregate["candidate_value"]
    assert candidate["unreconciled_source_position_count"] == 1
    assert candidate["localization_status"] == "pending_unique_account_owner_reconciliation"
    assert aggregate["status"] == "requires_review"
    basis = deepcopy(candidate["source_position_reconciliation"])
    assert basis["source_position_observation_count"] == 2
    assert len(basis["source_position_identity_sha256"]) == 64

    # An unchanged second preparation must not compare the reduced count 1
    # against the original two-position detail plane and abandon reconciliation.
    _month_closure(content)
    assert aggregate["candidate_value"]["source_position_reconciliation"] == basis
    records = content["datasets"]["repayment_records"]
    records.append(
        _physical_month_record(
            "grid:unknown", refs=_physical_month_refs("grid:unknown", x=220.0)
        )
    )
    _month_closure(content)
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 0
    assert aggregate["candidate_value"]["account_month_expected_row_count"] == 1
    assert aggregate["candidate_value"]["localization_status"] == "identity_closed"
    assert aggregate["status"] == "resolved"

    records.append(
        _physical_month_record(
            "grid:unknown",
            account_id="account:conflicting",
            refs=_physical_month_refs("grid:unknown", x=220.0),
        )
    )
    closure = _month_closure(content)
    assert closure["owner_conflict_positions"] == 1
    assert closure["source_position_balance_valid"] is True
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 1
    assert aggregate["candidate_value"]["account_month_expected_row_count"] is None
    assert aggregate["candidate_value"]["localization_status"] == "physical_owner_conflict"
    assert aggregate["candidate_value"]["source_position_reconciliation"] == basis
    assert aggregate["status"] == "requires_review"
    assert aggregate["extraction_issue_id"] == issue_id
    assert "_personal_detail_account_month_closure_proof" not in content["datasets"]


def test_source_projection_does_not_reconcile_structural_gap_by_owner_identity() -> None:
    for primary_refs in ([], _physical_month_refs("grid:primary", x=320.0)):
        content, aggregate = _structural_reconciliation_content()
        content["datasets"]["repayment_records"][0]["source_cell_refs"] = deepcopy(primary_refs)
        closure = _month_closure(content)

        # One detached source position has the same proven account/month as
        # the canonical row.  That does not establish that the row covers its
        # printed cells: its geometry is either absent or in a different table.
        assert closure["candidate_identity_count"] == 1
        assert closure["owner_bound_account_months"] == 2
        assert closure["raw_source_month_positions"] == 3
        assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 2
        assert aggregate["candidate_value"]["localization_status"] == (
            "pending_unique_account_owner_reconciliation"
        )
        assert aggregate["status"] == "requires_review"
        assert closure["source_position_balance_valid"] is True


def test_managed_monthly_population_cannot_fall_back_to_known_identities() -> None:
    for failure in ("physical_conflict", "unlocalized_position", "aggregate_mismatch"):
        if failure == "aggregate_mismatch":
            content, _aggregate = _structural_reconciliation_content(reported_count=1)
        else:
            records = [_physical_month_record("grid:A", account_id="account:A")]
            if failure == "physical_conflict":
                records.append(_physical_month_record("grid:B", account_id="account:B"))
                refs = _physical_month_refs("grid:unknown", x=220.0)
            else:
                refs = []
            content = {
                "facts": {},
                "datasets": {
                    "repayment_records": records,
                    "personal_detail_extraction_issues": [
                        _physical_month_structural_issue("grid:unknown", refs=refs)
                    ],
                },
            }
        for _refresh in range(2):
            _month_closure(content)
            source = content["datasets"]
            assert "_personal_detail_account_month_population_guard" in source, failure
            assert "_personal_detail_account_month_closure_proof" not in source, failure
            projected = project_personal_detail_datasets(source)
            status = next(
                row for row in projected["dataset_status"]
                if row["dataset_name"] == "credit_account_monthly_performance"
            )
            assert status["presence_status"] == "partial", failure
            assert status.get("expected_row_count") is None, failure
            assert status["reason"] == "account_month_source_position_population_unverified", failure
            assert not any(name.startswith("_personal_detail") for name in projected), failure


def test_managed_monthly_population_uses_physical_proof_with_unknown_owners() -> None:
    content = {
        "facts": {},
        "datasets": {
            "repayment_records": [_physical_month_record("grid:known")],
            "personal_detail_extraction_issues": [
                _physical_month_structural_issue(
                    "grid:unknown", refs=_physical_month_refs("grid:unknown", x=220.0)
                )
            ],
        },
    }
    _month_closure(content)
    projected = project_personal_detail_datasets(content["datasets"])
    status = next(
        row for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert status["presence_status"] == "partial"
    assert status["expected_row_count"] == 2
    assert status["reason"] == "account_month_source_position_partial_owner_unresolved"


def test_unverified_monthly_population_public_count_is_explicitly_observed_only() -> None:
    payload = {"datasets": [
        {"name": "credit_account_monthly_performance", "row_count": 2, "rows": [{}, {}],
         "completeness": {"expected_row_count": 2, "verified": True}},
        {"name": "dataset_status", "rows": [{"normalized": {
            "dataset_name": "credit_account_monthly_performance",
            "presence_status": "partial",
            "reason": "account_month_source_position_population_unverified",
        }}]},
    ]}
    _apply_personal_detail_dataset_status(payload)
    monthly = payload["datasets"][0]
    assert monthly["status"] == "partial"
    assert monthly["completeness"] == {
        "expected_row_count": 2, "emitted_row_count": 2, "omitted_row_count": 0,
        "verified": False,
        "basis": "personal_detail_dataset_status:partial:observed_only:population_unverified",
    }


def _low_confidence_monthly_projection_case(amount: str | None = "0") -> tuple[dict, dict]:
    """Lin-like legal M/weak OCR with one sealed month and independent amount."""
    grid_id, month = "mg_p19_repayment_0", "2019-07"
    record = _physical_month_record(grid_id, performance_month=month)
    record["record_id"] = record["repayment_id"] = f"{grid_id}:{month}"
    record["status"] = "unknown"
    record["overdue_amount"] = amount
    record["canonical_raw"] = {"status": ["M"]}
    record["_unresolved_fields"] = ["status"]
    registered = deepcopy(record["source_cell_refs"][0])
    registered["field_name"] = "status"
    registered["geometry_status"] = "exact"
    slot = {
        "source": "sealed_native_monthly_field_slot",
        "evidence_plane": "sealed_native_source_table",
        "logical_page": 7, "source_page": 4, "table_id": "table:monthly",
        "row": 4, "column": 7, "field_name": "status_code",
        "geometry_scope": "cell", "geometry_status": "exact",
        "coordinate_system": "pdf_points_top_left", "bbox": registered["bbox"],
        "evidence_ids": ["native:monthly:status"],
        "monthly_slot_proof": {
            "schema": "sealed_monthly_field_slot_v1", "account_id": "account:1",
            "year": 2019, "month": 7, "year_row": 3, "status_row": 4,
            "amount_row": 5, "header_row": 1,
            "year_evidence_ids": ["native:monthly:2019"],
            "header_evidence_ids": ["native:monthly:month-header"],
            "parent_evidence_ids": ["native:monthly:status"],
        },
        "registered_logical_page": 7, "registered_bbox": registered["bbox"],
        "registered_source_ref": registered,
        "source_ocr_confidence": 0.0767, "observed_raw": "M",
    }
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_monthly_source_ocr_confidence_unresolved",
        message="The weak source status remains unknown pending independent page evidence.",
        parser_stage="candidate_b_final_native_source_cell_guard",
        target_dataset="repayment_records", target_record_id=record["record_id"],
        field_name="status_code",
        observed_value={"observed_status": "M", "source_ocr_confidence": 0.0767,
                        "minimum_source_ocr_confidence": 0.72},
        candidate_value={"resolution": "withheld_pending_independent_page_evidence"},
        source_refs=[slot, registered],
        reason_codes=("low_source_ocr_confidence", "exact_monthly_source_slot",
                      "independent_page_confirmation_missing", "normalized_value_withheld"),
    )
    return record, issue


def test_low_confidence_monthly_status_retains_exact_month_and_independent_amount() -> None:
    for amount in ("0", None):
        for wrapped in (False, True):
            record, issue = _low_confidence_monthly_projection_case(amount)
            if wrapped:
                issue_id = issue.pop("record_id")
                refs = issue.pop("source_refs")
                issue = {"record_id": issue_id, "normalized": issue, "source_refs": refs}
            source = {"repayment_records": [record], "personal_detail_extraction_issues": [issue]}
            before = deepcopy(source)
            projected = project_personal_detail_datasets(source)
            rows = projected["credit_account_monthly_performance"]
            assert len(rows) == 1, (amount, wrapped)
            row = rows[0]
            assert row["monthly_performance_id"] == "mg_p19_repayment_0:2019-07"
            assert row["status_code"] is None
            assert row.get("status_amount") == amount
            assert row["extraction_status"] == "review"
            issues = projected["extraction_issues"]
            status_issues = [value for value in issues if (value.get("normalized") or value).get("issue_code") == (
                "candidate_b_monthly_source_ocr_confidence_unresolved"
            )]
            assert len(status_issues) == 1
            status_values = status_issues[0].get("normalized") or status_issues[0]
            assert status_values["target_record_id"] == row["monthly_performance_id"]
            assert status_values["field_name"] == "status_code"
            amount_issues = [value for value in issues if (value.get("normalized") or value).get("issue_code") == (
                "candidate_b_monthly_status_amount_unresolved"
            )]
            assert len(amount_issues) == int(amount is None)
            assert source == before


def test_low_confidence_monthly_status_cannot_retain_an_unproved_or_wrong_month() -> None:
    for defect in (
        "field_alias", "owner", "month", "grid", "target", "missing_source_ref",
        "missing_value_evidence", "boolean_score", "nan_score", "adequate_score",
        "missing_header_evidence", "wrong_stage", "resolved", "unproven_identity",
    ):
        record, issue = _low_confidence_monthly_projection_case()
        slot = issue["source_refs"][0]
        if defect == "field_alias":
            issue["field_name"] = "status"
        elif defect == "owner":
            slot["monthly_slot_proof"]["account_id"] = "account:other"
        elif defect == "month":
            slot["monthly_slot_proof"]["month"] = 8
        elif defect == "grid":
            slot["registered_source_ref"]["grid_id"] = "grid:other"
        elif defect == "target":
            issue["target_record_id"] = "grid:other:2019-07"
        elif defect == "missing_source_ref":
            issue["source_refs"].pop()
        elif defect == "missing_value_evidence":
            slot["evidence_ids"] = []
        elif defect in {"boolean_score", "nan_score", "adequate_score"}:
            score = {"boolean_score": False, "nan_score": float("nan"), "adequate_score": 0.99}[defect]
            issue["observed_value"]["source_ocr_confidence"] = score
            slot["source_ocr_confidence"] = score
        elif defect == "missing_header_evidence":
            slot["monthly_slot_proof"]["header_evidence_ids"] = []
        elif defect == "wrong_stage":
            issue["parser_stage"] = "independent_comment"
        elif defect == "resolved":
            issue["status"] = "resolved"
        elif defect == "unproven_identity":
            record["_account_month_identity_proof_status"] = "unproven_owner"
        projected = project_personal_detail_datasets({
            "repayment_records": [record], "personal_detail_extraction_issues": [issue],
        })
        assert projected.get("credit_account_monthly_performance", []) == [], defect


def _printed_anchor_case(
    name: str, *, page: int = 7, source_page: int = 4, top: float = 280.0,
    date_range: tuple[int, int, int, int] = (2024, 1, 2024, 1),
) -> dict:
    return {
        "coordinate_system": "pdf_points_top_left", "coordinate_plane": "raw_logical_page",
        "source_logical_page": page, "source_page": source_page,
        "evidence_ids": [f"sealed:{name}:date-range"],
        "bbox": [65.0, top, 315.0, top + 14.0], "date_range": list(date_range),
    }


def _printed_census_entry(plane: str, grid_id: str, anchor: dict, *, account_id: str | None = "account:1") -> dict:
    start_year, start_month, end_year, end_month = anchor["date_range"]
    months = [
        f"{index // 12:04d}-{index % 12 + 1:02d}"
        for index in range(start_year * 12 + start_month - 1, end_year * 12 + end_month)
    ]
    return {
        "plane": plane, "grid_id": grid_id, "printed_anchor_provenance": deepcopy(anchor),
        "observed_months": months, "printed_months": list(months),
        "invalid_record_count": 0, "duplicate_grid_id": False,
        "account_id": account_id,
        "account_month_owner_basis": "canonical_account_segment" if account_id else None,
    }


def _printed_census_issue(entries: list[dict], *, anchors: list[dict] | None = None, complete: bool = True) -> dict:
    if anchors is None:
        anchors = list({
            _printed_anchor_identity_key(entry["printed_anchor_provenance"]): entry["printed_anchor_provenance"]
            for entry in entries if entry.get("printed_anchor_provenance") is not None
        }.values())
    return make_issue(
        category="schema_incompleteness", issue_code=_PRINTED_GRID_CENSUS_ISSUE_CODE,
        message="Complete original printed ranges are independent of extracted business values.",
        severity="info", status="informational", parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records", target_record_id="printed_month_grid_inventory:1",
        observed_value={"anchors": anchors, "anchor_inventory_complete": complete, "grids": entries},
        candidate_value={"resolution": "source_population_audit_only"},
        source_refs=[*_printed_anchor_inventory_source_refs(anchors), *_printed_grid_census_source_refs(entries)],
        reason_codes=("sealed_printed_anchor_identity", "separate_canonical_and_detached_detector_namespaces", "business_cell_values_not_used"),
    )


def _single_structural_gap() -> dict:
    return make_issue(
        category="schema_incompleteness", issue_code="canonical_monthly_reconstruction_incomplete",
        message="One source position has no confirmed canonical reconstruction.",
        parser_stage="canonical_monthly_grid_materialization", target_dataset="repayment_records",
        candidate_value={"unreconciled_source_position_count": 1, "account_month_expected_row_count": None,
                         "localization_status": "pending_unique_account_owner_reconciliation"},
    )


def test_printed_census_separates_ye_page7_reused_detector_ids_and_owners() -> None:
    """Saved Ye p7 ranges: Shanghai21, Ant9 and rural-bank7, not grid-ID aliases."""
    shanghai = _printed_anchor_case("ye-p7-shanghai", top=137.0, date_range=(2019, 4, 2020, 12))
    ant = _printed_anchor_case("ye-p7-ant", top=333.0, date_range=(2019, 4, 2019, 12))
    rural = _printed_anchor_case("ye-p7-rural", top=542.0, date_range=(2019, 7, 2020, 1))
    entries = [
        _printed_census_entry("canonical", "mg_p7_repayment_0", ant, account_id="D1:8"),
        _printed_census_entry("canonical", "mg_p7_repayment_1", rural, account_id="D1:9"),
        _printed_census_entry("detached", "mg_p7_repayment_0", shanghai, account_id=None),
        _printed_census_entry("detached", "mg_p7_repayment_1", ant, account_id="D1:8"),
        _printed_census_entry("detached", "mg_p7_repayment_2", rural, account_id="D1:9"),
    ]
    month = "2019-09"
    rows = [
        _physical_month_record("mg_p7_repayment_0", account_id="D1:8", performance_month=month,
                               refs=_physical_month_refs("mg_p7_repayment_0", month, y=379.0)),
        _physical_month_record("mg_p7_repayment_1", account_id="D1:9", performance_month=month,
                               refs=_physical_month_refs("mg_p7_repayment_1", month, page=8, y=69.0)),
    ]
    alias = _physical_month_alias_issue("mg_p7_repayment_1", account_id="D1:8", performance_month=month)
    alias["candidate_value"]["source_grid_plane"] = "detached"
    alias["source_refs"] = [{
        "source": "candidate_b_monthly_grid_omission", "binding": "source_account_month_alias",
        "binding_quality": "source_account_month_alias", "account_id": "D1:8",
        "grid_id": "mg_p7_repayment_1", "performance_month": month, "field_name": "performance_month",
        "geometry_scope": "grid", "coordinate_system": "pdf_points_top_left",
        "page": 7, "logical_page": 7, "source_page": 4, "table_id": "pt_7_1",
        "bbox": [44.5, 241.5, 394.5, 411.5],
    }]
    content = {"facts": {}, "datasets": {
        "repayment_records": rows,
        "personal_detail_extraction_issues": [alias, _printed_census_issue(entries)],
    }}
    for _refresh in range(2):
        closure = _month_closure(content)
        assert closure["raw_source_month_positions"] == 37
        assert closure["owner_bound_account_months"] == 16
        assert closure["owner_unresolved_positions"] == 21
        assert closure["owner_conflict_positions"] == 0
        assert closure["printed_grid_census_valid"] is True
        assert closure["printed_grid_census_mapping_valid"] is True
        assert closure["source_position_balance_valid"] is True
        assert closure["physical_alias_source_month_observations"] >= 0
        projected = project_personal_detail_datasets(content["datasets"])
        status = next(row for row in projected["dataset_status"] if row["dataset_name"] == "credit_account_monthly_performance")
        assert status["expected_row_count"] == 37


def test_printed_census_population_deduplication_is_not_canonical_reconstruction() -> None:
    for canonical_pair in (False, True):
        for conflicting_owner in (False, True):
            anchor = _printed_anchor_case("lin-p10-october", page=10, source_page=5, date_range=(2020, 10, 2020, 10))
            month = "2020-10"
            primary = _physical_month_record("mg_p10_repayment_1", performance_month=month,
                refs=_physical_month_refs("mg_p10_repayment_1", month, page=10, source_page=5) if canonical_pair else [])
            if not canonical_pair:
                primary["status"] = "unknown"
            entries = [
                _printed_census_entry("canonical", "mg_p10_repayment_1", anchor),
                _printed_census_entry("detached", "mg_p10_repayment_0", anchor,
                                      account_id="account:other" if conflicting_owner else "account:1"),
            ]
            aggregate = _single_structural_gap()
            content = {"facts": {}, "datasets": {"repayment_records": [primary],
                "personal_detail_extraction_issues": [
                    _physical_month_structural_issue("mg_p10_repayment_0", performance_month=month,
                        refs=_physical_month_refs("mg_p10_repayment_0", month, page=10, source_page=5)),
                    aggregate, _printed_census_issue(entries),
                ]}}
            closure = _month_closure(content)
            assert closure["raw_source_month_positions"] == 1
            assert closure["canonical_present_source_month_positions"] == 1
            assert closure["canonical_covered_source_month_positions"] == int(canonical_pair and not conflicting_owner)
            assert closure["owner_conflict_positions"] == int(conflicting_owner)
            assert aggregate["candidate_value"]["unreconciled_source_position_count"] == int(not canonical_pair or conflicting_owner)
            assert aggregate["status"] == ("resolved" if canonical_pair and not conflicting_owner else "requires_review")
            assert "_personal_detail_printed_month_population_proof" in content["datasets"]
            assert ("_personal_detail_account_month_closure_proof" in content["datasets"]) is (not conflicting_owner)


def test_printed_census_preserves_separate_printed_blocks_with_identical_account_months() -> None:
    first = _printed_anchor_case("ye-nine-original", top=90.0, date_range=(2023, 7, 2024, 3))
    second = _printed_anchor_case("ye-nine-separate", top=420.0, date_range=(2023, 7, 2024, 3))
    entries = [_printed_census_entry("canonical", "grid:first", first),
               _printed_census_entry("detached", "grid:second", second)]
    content = {"facts": {}, "datasets": {"personal_detail_extraction_issues": [_printed_census_issue(entries)]}}
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 18
    assert closure["owner_bound_account_months"] == 18
    assert closure["owner_conflict_positions"] == 0
    assert closure["canonical_covered_source_month_positions"] == 0
    projected = project_personal_detail_datasets(content["datasets"])
    status = next(row for row in projected["dataset_status"] if row["dataset_name"] == "credit_account_monthly_performance")
    assert status["expected_row_count"] == 18
    assert status["observed_row_count"] == 0


def test_printed_census_mixed_new_ocr_anchor_requires_independent_complete_pair_mapping() -> None:
    for complete_pair in (False, True):
        for reverse in (False, True):
            anchor = _printed_anchor_case("original-raw-range")
            canonical = _printed_census_entry("canonical", "grid:new-ocr", anchor)
            canonical["printed_anchor_provenance"] = None
            detached = _printed_census_entry("detached", "grid:raw", anchor)
            refs = _physical_month_refs("grid:new-ocr")
            if not complete_pair:
                refs = refs[:1]
            record = _physical_month_record("grid:new-ocr", refs=refs)
            aggregate = _single_structural_gap()
            issues = [_physical_month_structural_issue("grid:raw"), aggregate,
                      _printed_census_issue([canonical, detached], anchors=[anchor])]
            if reverse:
                issues.reverse()
            content = {"facts": {}, "datasets": {"repayment_records": [record], "personal_detail_extraction_issues": issues}}
            closure = _month_closure(content)
            assert closure["raw_source_month_positions"] == 1
            assert closure["printed_grid_census_valid"] is True
            assert closure["printed_grid_census_mapping_valid"] is complete_pair
            assert closure["printed_grid_census_unmapped_observation_count"] == int(not complete_pair)
            assert closure["canonical_covered_source_month_positions"] == int(complete_pair)
            assert aggregate["candidate_value"]["unreconciled_source_position_count"] == int(not complete_pair)
            assert ("_personal_detail_account_month_closure_proof" in content["datasets"]) is complete_pair
            projected = project_personal_detail_datasets(content["datasets"])
            status = next(row for row in projected["dataset_status"] if row["dataset_name"] == "credit_account_monthly_performance")
            assert status["expected_row_count"] == 1
            assert status["presence_status"] == "partial"


def test_printed_census_counts_original_anchor_undiscovered_by_either_detector() -> None:
    discovered = _printed_anchor_case("detected", date_range=(2024, 1, 2024, 2))
    undiscovered = _printed_anchor_case("not-detected", top=500.0, date_range=(2024, 4, 2024, 6))
    manifest = _printed_census_issue(
        [_printed_census_entry("canonical", "grid:known", discovered)],
        anchors=[discovered, undiscovered],
    )
    content = {"facts": {}, "datasets": {"personal_detail_extraction_issues": [manifest]}}
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 5
    assert closure["owner_bound_account_months"] == 2
    assert closure["owner_unresolved_positions"] == 3
    assert closure["source_position_balance_valid"] is True
    status = next(row for row in project_personal_detail_datasets(content["datasets"])["dataset_status"]
                  if row["dataset_name"] == "credit_account_monthly_performance")
    assert status["expected_row_count"] == 5


def test_printed_census_rejects_mutated_or_incomplete_source_receipts() -> None:
    for defect in ("missing_ref", "changed_ref", "changed_observed", "outer_id", "false_complete", "boolean_count"):
        anchor = _printed_anchor_case("guard")
        issue = _printed_census_issue([_printed_census_entry("canonical", "grid:known", anchor)], complete=defect != "false_complete")
        if defect == "missing_ref":
            issue["source_refs"].pop()
        elif defect == "changed_ref":
            issue["source_refs"][0]["bbox"][1] += 1.0
        elif defect == "changed_observed":
            issue["observed_value"]["grids"][0]["account_id"] = "account:other"
        elif defect == "outer_id":
            issue = {"record_id": "contradictory", "normalized": issue}
        content = {"facts": {}, "datasets": {"repayment_records": [_physical_month_record("grid:known")],
                                             "personal_detail_extraction_issues": [issue]}}
        _month_closure(content)
        if defect == "boolean_count":
            content["datasets"]["_personal_detail_printed_month_population_proof"][0]["expected_source_position_count"] = True
        else:
            assert "_personal_detail_printed_month_population_proof" not in content["datasets"], defect
        projected = project_personal_detail_datasets(content["datasets"])
        status = next(row for row in projected["dataset_status"] if row["dataset_name"] == "credit_account_monthly_performance")
        assert status.get("expected_row_count") is None, defect
        assert status["reason"] == "account_month_source_position_population_unverified", defect


def test_printed_census_rejects_partial_anchor_overlap_but_scopes_opaque_ids_by_page() -> None:
    first = _printed_anchor_case("same-opaque")
    for same_page in (False, True):
        second = deepcopy(first)
        second["bbox"][1] += 100.0
        second["bbox"][3] += 100.0
        if not same_page:
            second["source_logical_page"] = 8
            second["source_page"] = 5
        manifest = _printed_census_issue([], anchors=[first, second])
        content = {"facts": {}, "datasets": {"personal_detail_extraction_issues": [manifest]}}
        closure = _month_closure(content)
        assert closure["printed_grid_census_valid"] is (not same_page)
        if not same_page:
            assert closure["raw_source_month_positions"] == 2
            assert closure["owner_unresolved_positions"] == 2
        assert ("_personal_detail_printed_month_population_proof" in content["datasets"]) is (not same_page)


def test_printed_population_receipt_refresh_drops_stale_public_denominator() -> None:
    anchor = _printed_anchor_case("refresh")
    issue = _printed_census_issue([_printed_census_entry("canonical", "grid:known", anchor)])
    content = {"facts": {}, "datasets": {"repayment_records": [_physical_month_record("grid:known")],
                                         "personal_detail_extraction_issues": [issue]}}
    _month_closure(content)
    assert "_personal_detail_printed_month_population_proof" in content["datasets"]
    issue["source_refs"].pop()
    _month_closure(content)
    assert "_personal_detail_printed_month_population_proof" not in content["datasets"]
    assert "_personal_detail_account_month_closure_proof" not in content["datasets"]
    status = next(row for row in project_personal_detail_datasets(content["datasets"])["dataset_status"]
                  if row["dataset_name"] == "credit_account_monthly_performance")
    assert status.get("expected_row_count") is None


def test_printed_census_refresh_cannot_resurrect_removed_canonical_alias_coverage() -> None:
    anchor = _printed_anchor_case("current-vs-historical")
    canonical = _printed_census_entry("canonical", "grid:former-canonical", anchor)
    detached = _printed_census_entry("detached", "grid:raw", anchor)
    alias = _physical_month_alias_issue("grid:former-canonical")
    alias["candidate_value"]["source_grid_plane"] = "canonical"
    aggregate = _single_structural_gap()
    source_issue = _physical_month_structural_issue("grid:raw")
    content = {"facts": {}, "datasets": {"personal_detail_extraction_issues": [
        alias, source_issue, aggregate, _printed_census_issue([canonical, detached]),
    ]}}
    before = _month_closure(content)
    assert before["raw_source_month_positions"] == 1
    assert before["canonical_covered_source_month_positions"] == 1
    assert aggregate["status"] == "resolved"
    content["datasets"]["personal_detail_extraction_issues"] = [
        alias, source_issue, aggregate, _printed_census_issue([detached]),
    ]
    for _refresh in range(2):
        after = _month_closure(content)
        assert after["raw_source_month_positions"] == 1
        assert after["canonical_present_source_month_positions"] == 0
        assert after["canonical_covered_source_month_positions"] == 0
        assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 1
        assert aggregate["status"] == "requires_review"
        assert "_personal_detail_printed_month_population_proof" in content["datasets"]


def test_printed_census_removal_cannot_reenter_legacy_identity_denominator() -> None:
    for remove in (False, True):
        anchor = _printed_anchor_case("managed-refresh", date_range=(2024, 1, 2024, 2))
        issue = _printed_census_issue([_printed_census_entry("canonical", "grid:known", anchor)])
        content = {"facts": {}, "datasets": {"repayment_records": [_physical_month_record("grid:known")],
                                             "personal_detail_extraction_issues": [issue]}}
        _month_closure(content)
        assert content["datasets"]["_personal_detail_printed_month_population_proof"][0]["expected_source_position_count"] == 2
        if remove:
            content["datasets"]["personal_detail_extraction_issues"] = []
        else:
            issue["status"] = "resolved"
        for _refresh in range(2):
            closure = _month_closure(content)
            assert closure["printed_grid_census_required"] is True
            assert closure["printed_grid_census_valid"] is False
            assert "_personal_detail_printed_month_population_proof" not in content["datasets"]
            assert "_personal_detail_account_month_closure_proof" not in content["datasets"]
            status = next(row for row in project_personal_detail_datasets(content["datasets"])["dataset_status"]
                          if row["dataset_name"] == "credit_account_monthly_performance")
            assert status.get("expected_row_count") is None
            assert status["reason"] == "account_month_source_position_population_unverified"


def test_printed_census_invalid_canonical_alias_cannot_certify_reconstruction() -> None:
    anchor = _printed_anchor_case("invalid-alias")
    canonical = _printed_census_entry("canonical", "grid:canonical", anchor)
    detached = _printed_census_entry("detached", "grid:raw", anchor)
    alias = _physical_month_alias_issue("grid:canonical")
    alias["candidate_value"]["source_grid_plane"] = "canonical"
    alias["reason_codes"].append("independent_unvalidated_claim")
    aggregate = _single_structural_gap()
    content = {"facts": {}, "datasets": {"personal_detail_extraction_issues": [
        alias, _physical_month_structural_issue("grid:raw"), aggregate,
        _printed_census_issue([canonical, detached]),
    ]}}
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 1
    assert closure["canonical_present_source_month_positions"] == 1
    assert closure["canonical_covered_source_month_positions"] == 0
    assert aggregate["candidate_value"]["unreconciled_source_position_count"] == 1
    assert aggregate["status"] == "requires_review"


def test_relationship_printed_census_reaches_source_and_public_schema_without_value_changes() -> None:
    anchor = _printed_anchor_case("producer")
    grid = {"grid_id": "grid:canonical", "page": 7, "bbox": [60.0, 280.0, 400.0, 350.0],
            "coordinate_system": "pdf_points_top_left",
            "audit": {"printed_anchor_provenance": anchor, "date_range": {
                "start_year": 2024, "start_month": 1, "end_year": 2024, "end_month": 1}}}
    account = {"account_id": "account:1", "_canonical_segment": {
        "ownership_basis": "printed_anchor_to_next_anchor", "anchor_logical_page": 7,
        "anchor_bbox": [45.0, 200.0, 350.0, 215.0],
        "pages": [{"logical_page": 7, "min_y": 200.0, "max_y": 400.0}],
    }}
    context = SimpleNamespace(
        _candidate_b_printed_grid_census_required=True,
        _candidate_b_printed_anchor_inventory=[deepcopy(anchor)],
        _candidate_b_printed_anchor_inventory_complete=True,
        _personal_detail_extraction_issues=[],
    )
    # Both real monthly materializers supply this stable native ID.  An exact
    # account/month proof does not authorize schema projection to invent it.
    source_record = {"repayment_id": "grid:canonical:2024-01",
                     "grid_id": "grid:canonical", "year": 2024, "month": 1,
                     "status": "N", "overdue_amount": "0",
                     "source_cell_refs": _physical_month_refs("grid:canonical")}
    before = deepcopy(source_record)
    linked = link_candidate_b_repayments([source_record], [account], [grid], issue_context=context)
    assert len(linked) == 1
    assert linked[0]["repayment_id"] == source_record["repayment_id"]
    assert linked[0]["status"] == "N" and str(linked[0]["overdue_amount"]) == "0"
    manifests = [issue for issue in context._personal_detail_extraction_issues if issue.get("issue_code") == _PRINTED_GRID_CENSUS_ISSUE_CODE]
    assert len(manifests) == 1
    assert manifests[0]["observed_value"]["anchor_inventory_complete"] is True
    content = {"facts": {}, "datasets": {"repayment_records": linked,
        "personal_detail_extraction_issues": context._personal_detail_extraction_issues}}
    closure = _month_closure(content)
    assert closure["raw_source_month_positions"] == 1
    assert closure["canonical_covered_source_month_positions"] == 1
    assert closure["printed_grid_census_valid"] is True
    projected = project_personal_detail_datasets(content["datasets"])
    assert len(projected["credit_account_monthly_performance"]) == 1
    row = projected["credit_account_monthly_performance"][0]
    assert row["monthly_performance_id"] == "grid:canonical:2024-01"
    assert row["status_code"] == "N" and row["status_amount"] == "0"
    status = next(row for row in projected["dataset_status"] if row["dataset_name"] == "credit_account_monthly_performance")
    assert status["expected_row_count"] == 1
    assert source_record["status"] == before["status"]
    assert source_record["overdue_amount"] == before["overdue_amount"]
    for invalid_id in (None, "grid:other:2024-01"):
        invalid_source = deepcopy(content["datasets"])
        invalid_record = invalid_source["repayment_records"][0]
        invalid_record.pop("repayment_id", None)
        invalid_record.pop("record_id", None)
        if invalid_id is not None:
            invalid_record["repayment_id"] = invalid_id
        invalid_projection = project_personal_detail_datasets(invalid_source)
        assert invalid_projection["credit_account_monthly_performance"] == [], invalid_id
        invalid_status = next(
            row for row in invalid_projection["dataset_status"]
            if row["dataset_name"] == "credit_account_monthly_performance"
        )
        # Physical population remains independently known even when a malformed
        # business row cannot pass the stable-identity gate.
        assert invalid_status["expected_row_count"] == 1, invalid_id
        assert invalid_status["observed_row_count"] == 0, invalid_id


def test_source_projection_requires_original_aggregate_detail_count_agreement() -> None:
    for reported_count in (0, 1, 3, True, "2", None):
        for detail_owner_bound in (False, True):
            content, aggregate = _structural_reconciliation_content(
                reported_count=reported_count
            )
            if detail_owner_bound:
                content["datasets"]["repayment_records"].append(
                    _physical_month_record(
                        "grid:unknown",
                        refs=_physical_month_refs("grid:unknown", x=220.0),
                    )
                )
            for _refresh in range(2):
                closure = _month_closure(content)
                label = (reported_count, detail_owner_bound, _refresh)
                candidate = aggregate["candidate_value"]
                assert candidate["unreconciled_source_position_count"] == reported_count, label
                assert candidate["account_month_expected_row_count"] is None, label
                assert candidate["localization_status"] == "source_position_reconciliation_invalid", label
                assert "source_position_reconciliation" not in candidate, label
                assert aggregate["status"] == "requires_review", label
                assert closure["source_position_reconciliation_valid"] is False, label
                assert closure["status"] == "source_localization_invalid", label
                assert "_personal_detail_account_month_closure_proof" not in content["datasets"], label
                state = content["facts"]["personal_detail_dataset_states"]["repayment_records"]
                assert state["presence_status"] == "partial", label
                assert state["reason"] == "account_month_source_localization_invalid", label


def test_source_projection_rejects_aggregate_only_or_malformed_initial_basis() -> None:
    for variant in ("aggregate_only", "missing_candidate", "missing_count"):
        content, aggregate = _structural_reconciliation_content()
        if variant == "aggregate_only":
            content["datasets"]["repayment_records"] = []
            content["datasets"]["personal_detail_extraction_issues"] = [aggregate]
        elif variant == "missing_candidate":
            aggregate.pop("candidate_value")
        elif variant == "missing_count":
            aggregate["candidate_value"].pop("unreconciled_source_position_count")
        closure = _month_closure(content)
        assert closure["source_position_reconciliation_valid"] is False, variant
        assert closure["status"] == "source_localization_invalid", variant
        assert aggregate["status"] == "requires_review", variant
        assert "_personal_detail_account_month_closure_proof" not in content["datasets"], variant


def test_source_projection_empty_refresh_clears_only_function_owned_public_state() -> None:
    for external_repayment_state in (False, True):
        external_state = {
            "presence_status": "partial",
            "reason": "manual_source_review",
            "expected_row_count": 3,
        }
        content = {
            "facts": {
                "personal_detail_dataset_states": {"statements": deepcopy(external_state)}
            },
            "datasets": {"repayment_records": [_physical_month_record("grid:primary")]},
        }
        prepare_personal_detail_source_collections(content)
        assert content["facts"]["personal_detail_account_month_closure"]["status"] == "identity_closed"
        assert "_personal_detail_account_month_closure_proof" in content["datasets"]
        content["datasets"]["repayment_records"] = []
        content["datasets"]["personal_detail_extraction_issues"] = []
        if external_repayment_state:
            content["facts"]["personal_detail_dataset_states"]["repayment_records"] = deepcopy(external_state)
        for _refresh in range(2):
            prepare_personal_detail_source_collections(content)
            assert "personal_detail_account_month_closure" not in content["facts"]
            assert "_personal_detail_account_month_closure_proof" not in content["datasets"]
            states = content["facts"]["personal_detail_dataset_states"]
            assert states["statements"] == external_state
            if external_repayment_state:
                assert states["repayment_records"] == external_state
            else:
                assert "repayment_records" not in states
            published = next(
                row for row in content["datasets"]["personal_detail_dataset_status"]
                if row["dataset_name"] == "repayment_records"
            )
            assert published["observed_row_count"] == 0
            assert published["presence_status"] == (
                "partial" if external_repayment_state else "unknown"
            )


def test_source_projection_physical_pair_identity_ignores_detector_local_row_offsets() -> None:
    for status_rows in ((0, 2), (2, 4), (4, 20)):
        for same_owner in (False, True):
            records = []
            for index, status_row in enumerate(status_rows):
                grid_id = f"grid:local-offset:{index}"
                refs = _physical_month_refs(grid_id)
                refs[0]["row"] = status_row
                refs[1]["row"] = status_row + 1
                records.append(
                    _physical_month_record(
                        grid_id,
                        account_id="account:A" if same_owner or index == 0 else "account:B",
                        refs=refs,
                    )
                )
            content = {"facts": {}, "datasets": {"repayment_records": records}}
            closure = _month_closure(content)
            label = (status_rows, same_owner)
            assert closure["source_month_position_observations"] == 2, label
            assert closure["raw_source_month_positions"] == 1, label
            assert closure["physical_alias_source_month_observations"] == 1, label
            assert closure["owner_unresolved_positions"] == 0, label
            assert closure["owner_bound_account_months"] == int(same_owner), label
            assert closure["owner_conflict_positions"] == int(not same_owner), label
            assert closure["source_position_balance_valid"] is True, label
            assert ("_personal_detail_account_month_closure_proof" in content["datasets"]) is same_owner, label
            assert [record["source_cell_refs"][0]["row"] for record in records] == list(status_rows)


def test_source_projection_retires_and_reactivates_only_owned_conflict_history() -> None:
    for wrapped in (False, True):
        first = _physical_month_record("grid:A", account_id="account:A")
        second = _physical_month_record("grid:B", account_id="account:B")
        content = {"facts": {}, "datasets": {"repayment_records": [first, second]}}
        closure = _month_closure(content)
        assert closure["owner_conflict_positions"] == 1
        issues = content["datasets"]["personal_detail_extraction_issues"]
        owned = next(issue for issue in issues if issue.get("issue_code") == "account_month_physical_owner_conflict")
        issue_id = owned["extraction_issue_id"]
        if wrapped:
            record_id = owned.pop("record_id")
            index = issues.index(owned)
            issues[index] = {"record_id": record_id, "normalized": owned}
        independent = make_issue(
            category="schema_incompleteness",
            issue_code="account_month_physical_owner_conflict",
            message="An independent review remains open until its reviewer resolves it.",
            parser_stage="independent_manual_review",
            target_dataset="repayment_records",
            target_record_id="manual:physical-conflict",
            field_name="account_id",
            observed_value=deepcopy(owned["observed_value"]),
            reason_codes=owned["reason_codes"],
        )
        issues.append(independent)
        content["datasets"]["repayment_records"] = [first]
        for _refresh in range(2):
            closure = _month_closure(content)
            assert closure["status"] == "identity_closed", wrapped
            assert "_personal_detail_account_month_closure_proof" in content["datasets"], wrapped
            assert owned["status"] == "resolved", wrapped
            assert owned["extraction_issue_id"] == issue_id, wrapped
            assert independent["status"] == "requires_review", wrapped
        content["datasets"]["repayment_records"].append(second)
        closure = _month_closure(content)
        assert closure["status"] == "physical_owner_conflict", wrapped
        assert "_personal_detail_account_month_closure_proof" not in content["datasets"], wrapped
        assert owned["status"] == "requires_review", wrapped
        assert independent["status"] == "requires_review", wrapped
        assert sum(
            (row.get("normalized") or row).get("extraction_issue_id") == issue_id
            for row in issues
        ) == 1, wrapped
        content["datasets"]["repayment_records"] = []
        prepare_personal_detail_source_collections(content)
        assert "personal_detail_account_month_closure" not in content["facts"], wrapped
        assert "_personal_detail_account_month_closure_proof" not in content["datasets"], wrapped
        assert owned["status"] == "resolved", wrapped
        assert independent["status"] == "requires_review", wrapped


def test_source_projection_preserves_unowned_or_contradictory_conflict_rows() -> None:
    for variant in ("stage", "reasons", "outer_record_id", "inner_record_id", "outer_issue_id"):
        first = _physical_month_record("grid:A", account_id="account:A")
        content = {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    first, _physical_month_record("grid:B", account_id="account:B")
                ]
            },
        }
        _month_closure(content)
        probe = next(
            issue for issue in content["datasets"]["personal_detail_extraction_issues"]
            if issue.get("issue_code") == "account_month_physical_owner_conflict"
        )
        if variant == "stage":
            probe["parser_stage"] = "independent_manual_review"
            row = probe
        elif variant == "reasons":
            probe["reason_codes"].append("independent_manual_review")
            row = probe
        else:
            row = {"record_id": probe["record_id"], "normalized": probe}
            if variant == "outer_record_id":
                row["record_id"] = "manual:contradictory-record"
            elif variant == "inner_record_id":
                probe["record_id"] = "manual:contradictory-record"
            elif variant == "outer_issue_id":
                row["extraction_issue_id"] = "manual:contradictory-issue"
        content["datasets"]["repayment_records"] = [first]
        content["datasets"]["personal_detail_extraction_issues"] = [row]
        closure = _month_closure(content)
        assert closure["status"] == "identity_closed", variant
        assert probe["status"] == "requires_review", variant


def test_source_projection_invalidates_changed_reconciliation_basis_or_detail_plane() -> None:
    for variant in (
        "count",
        "boolean_count",
        "digest",
        "schema",
        "missing_detail",
        "changed_detail",
        "empty_source_plane",
    ):
        content, aggregate = _structural_reconciliation_content()
        records = content["datasets"]["repayment_records"]
        records.append(
            _physical_month_record(
                "grid:unknown", refs=_physical_month_refs("grid:unknown", x=220.0)
            )
        )
        _month_closure(content)
        assert aggregate["status"] == "resolved", variant
        original_count = aggregate["candidate_value"]["unreconciled_source_position_count"]
        basis = aggregate["candidate_value"]["source_position_reconciliation"]
        issues = content["datasets"]["personal_detail_extraction_issues"]
        if variant == "count":
            basis["source_position_observation_count"] = 1
        elif variant == "boolean_count":
            basis["source_position_observation_count"] = True
        elif variant == "digest":
            basis["source_position_identity_sha256"] = "0" * 64
        elif variant == "schema":
            basis["schema"] = "unknown"
        elif variant == "missing_detail":
            issues.remove(issues[0])
        elif variant == "changed_detail":
            issues[0]["observed_value"]["grid_id"] = "grid:changed"
        elif variant == "empty_source_plane":
            content["datasets"]["repayment_records"] = []
            content["datasets"]["personal_detail_extraction_issues"] = [aggregate]
        closure = _month_closure(content)
        candidate = aggregate["candidate_value"]
        assert candidate["unreconciled_source_position_count"] == original_count, variant
        assert candidate["account_month_expected_row_count"] is None, variant
        assert candidate["localization_status"] == "source_position_reconciliation_invalid", variant
        assert aggregate["status"] == "requires_review", variant
        assert closure["source_position_reconciliation_valid"] is False, variant
        assert closure["status"] == "source_localization_invalid", variant
        assert "_personal_detail_account_month_closure_proof" not in content["datasets"], variant
