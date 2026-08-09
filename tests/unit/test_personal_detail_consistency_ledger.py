from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger import (
    apply_document_consistency_ledger,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
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
                "management_institution": None,
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
