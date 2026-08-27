# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.models.entities.parse_result import ParseResult
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.enterprise_native.projector import (
    enterprise_public_dataset_policy,
    project_enterprise_artifact_semantic,
    project_enterprise_community_json,
)
from docmirror.plugins.credit_report.enterprise_native.variant import variant


def _column(key: str, value_type: str = "string") -> dict[str, object]:
    return {
        "key": key,
        "label": key,
        "type": value_type,
        "nullable": False,
        "raw_available": True,
        "evidence_available": True,
    }


def _row(record_id: str, normalized: dict[str, object]) -> dict[str, object]:
    return {
        "record_id": record_id,
        "normalized": normalized,
        "canonical_raw": deepcopy(normalized),
        "raw": deepcopy(normalized),
        "source": {
            "page_range": [2, 3],
            "source_refs": [{"page": 2, "table_id": "source-table"}],
        },
        "confidence": 1.0,
    }


def _dataset(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    keys = list(dict.fromkeys(key for row in rows for key in row["normalized"]))
    return {
        "id": f"ds_{name}",
        "name": name,
        "label": name,
        "type": name,
        "section_id": "sec_enterprise",
        "csv": f"001_datasets/{name}.csv",
        "row_count": len(rows),
        "grain": f"one row per {name}",
        "primary_key": "record_id",
        "schema_version": "1.0",
        "status": "complete",
        "columns": [_column(key) for key in keys],
        "completeness": {
            "expected_row_count": len(rows),
            "emitted_row_count": len(rows),
            "omitted_row_count": 0,
            "verified": True,
            "basis": "test",
        },
        "reading_columns": keys,
        "storage_role": "canonical",
        "record_path": "rows",
        "rows": rows,
    }


def _payload() -> dict[str, object]:
    datasets = [
        _dataset(
            "report_notes",
            [_row("note:1", {"note_id": "note:1", "content": "boilerplate"})],
        ),
        _dataset(
            "enterprise_section_presence",
            [
                _row(
                    "presence:1",
                    {
                        "section_presence_id": "presence:1",
                        "section_key": "credit",
                        "presence_status": "present_with_records",
                    },
                )
            ],
        ),
        _dataset(
            "enterprise_displayed_credit_summary",
            [
                _row(
                    "displayed:1",
                    {
                        "displayed_summary_id": "displayed:1",
                        "transaction_group": "借贷交易",
                        "settlement_status": "active",
                        "business_category": "贴现",
                        "institution": "甲银行",
                        "business_type": "票据贴现",
                        "source_account_count": 2,
                        "source_reported_amount": "45.00",
                        "source_reported_amount_status": "reported",
                        "amount_kind": "balance",
                        "currency": "CNY",
                        "amount_unit": "CNY_10K",
                        "sequence": 1,
                        "source_page": 2,
                    },
                )
            ],
        ),
        _dataset(
            "enterprise_credit_detail_groups",
            [
                _row(
                    "group:1",
                    {
                        "credit_detail_group_id": "group:1",
                        "group_phase": "active",
                        "group_kind": "account_card",
                        "business_category": "短期借款",
                        "reported_record_count": 2,
                        "represented_dataset": "enterprise_credit_accounts",
                    },
                )
            ],
        ),
        _dataset(
            "enterprise_attachment_credit_details",
            [
                _row(
                    "detail:1",
                    {
                        "attachment_detail_id": "detail:1",
                        "attachment_account_id": "attachment:1",
                        "account_identifier": "ACCOUNT-1",
                        "business_category": "贴现",
                        "institution": "乙银行",
                        "amount": "99.50",
                        "amount_kind": "discount_amount",
                        "discount_amount": "99.50",
                        "discount_amount_status": "reported",
                        "currency": "CNY",
                        "amount_unit": "CNY_10K",
                    },
                )
            ],
        ),
        _dataset(
            "enterprise_repayment_responsibility_accounts",
            [
                _row(
                    "liability:1",
                    {
                        "liability_id": "liability:1",
                        "responsibility_type": "保证人",
                        "institution": "丙银行",
                        "overdue_months_or_repayment_status": "0",
                        "currency": "CNY",
                        "amount_unit": "CNY_10K",
                        "continuation_complete": True,
                    },
                ),
                _row(
                    "liability:2",
                    {
                        "liability_id": "liability:2",
                        "responsibility_type": "共同债务人",
                        "institution": "丁银行",
                        "overdue_months_or_repayment_status": "N",
                        "currency": "CNY",
                        "amount_unit": "CNY_10K",
                    },
                ),
            ],
        ),
        _dataset(
            "enterprise_attachment_accounts",
            [
                _row(
                    "attachment:1",
                    {
                        "attachment_account_id": "attachment:1",
                        "attachment_record_type": "account",
                        "account_id": "account:1",
                        "account_identifier": "ACCOUNT-1",
                        "account_status": "active",
                        "business_category": "短期借款",
                    },
                ),
                _row(
                    "attachment:2",
                    {
                        "attachment_account_id": "attachment:2",
                        "attachment_record_type": "business",
                        "business_category": "银行保函及其他业务",
                        "business_type": "银行保函",
                    },
                ),
            ],
        ),
        _dataset(
            "enterprise_profile",
            [
                _row(
                    "profile:1",
                    {
                        "enterprise_profile_id": "profile:1",
                        "operating_status": "存续",
                        "operating_status_status": "reported",
                        "operating_status_source_institution": "市场监管总局",
                        "operating_status_source_institution_status": "reported",
                    },
                )
            ],
        ),
    ]
    dataset_ids = [dataset["id"] for dataset in datasets]
    return {
        "schema": {
            "name": "docmirror.community",
            "version": "3.0.0",
            "edition": "community",
            "domain": "enterprise_credit_report",
            "support_level": "ga",
        },
        "document": {
            "id": "doc:1",
            "type": "enterprise_credit_report",
            "title": "企业信用报告",
            "page_count": 3,
            "language": ["zh-CN"],
            "source_file": {
                "name": "enterprise.pdf",
                "mime_type": "application/pdf",
                "sha256": "",
            },
            "units": {},
        },
        "sections": [
            {
                "id": "sec_enterprise",
                "title": "企业报告",
                "type": "enterprise",
                "page_range": [1, 3],
                "items": [{"key": "source_account_summary_page", "value": 2}],
                "groups": [
                    {"key": "extracted_public_record_type_counts", "items": []}
                ],
                "dataset_refs": dataset_ids,
            }
        ],
        "datasets": datasets,
        "reading": {
            "version": "1.0",
            "profile": "community",
            "document_flow": [
                {"order": 1, "kind": "document", "ref_id": "doc:1"},
                {"order": 2, "kind": "section", "ref_id": "sec_enterprise"},
                *[
                    {"order": index + 3, "kind": "dataset", "ref_id": dataset_id}
                    for index, dataset_id in enumerate(dataset_ids)
                ],
            ],
            "tables": [
                {
                    "id": f"reading:{dataset['id']}",
                    "dataset_id": dataset["id"],
                    "section_id": "sec_enterprise",
                    "title": dataset["label"],
                    "column_keys": [column["key"] for column in dataset["columns"]],
                    "row_count": dataset["row_count"],
                }
                for dataset in datasets
            ],
        },
        "files": {
            "content_md": "001_content.md",
            "enhanced_reading_md": "001_enhanced_reading.md",
            "datasets_dir": "001_datasets",
            "dataset_audit_csv": "001_datasets/_audit_cells.csv",
        },
        "warnings": [],
    }


def test_enterprise_public_policy_covers_the_closed_world_schema() -> None:
    policy = enterprise_public_dataset_policy()

    assert tuple(policy) == variant.dataset_names()
    assert policy["report_notes"] is None
    assert policy["enterprise_section_presence"] is None
    assert policy["enterprise_credit_detail_groups"] == (
        "group_phase",
        "group_kind",
        "business_category",
        "reported_record_count",
    )
    assert "attachment_record_type" in policy["enterprise_attachment_accounts"]


def test_enterprise_public_projection_is_business_only_and_non_mutating() -> None:
    payload = _payload()
    payload["sections"].append(
        {
            "id": "sec_empty",
            "title": "Empty source section",
            "type": "empty",
            "page_range": [3, 3],
            "items": [],
            "groups": [],
            "dataset_refs": [],
        }
    )
    payload["reading"]["document_flow"].append(
        {"order": 99, "kind": "section", "ref_id": "sec_empty"}
    )
    before = deepcopy(payload)

    projected = project_enterprise_community_json(payload)

    assert payload == before
    assert projected["schema"]["version"] == "4.0.0"
    datasets = {dataset["name"]: dataset for dataset in projected["datasets"]}
    assert "report_notes" not in datasets
    assert "enterprise_section_presence" not in datasets

    displayed = datasets["enterprise_displayed_credit_summary"]
    expected_displayed_values = {
        "transaction_group": "借贷交易",
        "settlement_status": "active",
        "business_category": "贴现",
        "institution": "甲银行",
        "business_type": "票据贴现",
        "account_count": 2,
        "balance": "45.00",
        "currency": "CNY",
        "amount_unit": "CNY_10K",
    }
    assert displayed["rows"][0] == {
        "record_id": "displayed:1",
        "normalized": expected_displayed_values,
        "source": {"page_range": [2, 3]},
    }
    assert [column["key"] for column in displayed["columns"]] == list(
        displayed["rows"][0]["normalized"]
    )
    assert all(
        not column["raw_available"] and not column["evidence_available"]
        for column in displayed["columns"]
    )

    detail = datasets["enterprise_attachment_credit_details"]["rows"][0]["normalized"]
    assert detail["discount_amount"] == "99.50"
    assert "amount" not in detail
    assert "amount_kind" not in detail
    assert "discount_amount_status" not in detail

    group = datasets["enterprise_credit_detail_groups"]["rows"][0]["normalized"]
    assert group == {
        "group_phase": "active",
        "group_kind": "account_card",
        "business_category": "短期借款",
        "reported_record_count": 2,
    }

    liability_rows = datasets["enterprise_repayment_responsibility_accounts"]["rows"]
    assert liability_rows[0]["normalized"]["overdue_months"] == 0
    assert "repayment_status" not in liability_rows[0]["normalized"]
    assert liability_rows[1]["normalized"]["repayment_status"] == "N"
    assert "overdue_months" not in liability_rows[1]["normalized"]
    assert all(
        "overdue_months_or_repayment_status" not in row["normalized"]
        for row in liability_rows
    )
    assert all(
        set(row) == {"record_id", "normalized", "source"}
        for dataset in datasets.values()
        for row in dataset["rows"]
    )

    profile = datasets["enterprise_profile"]["rows"][0]["normalized"]
    assert profile == {
        "operating_status": "存续",
        "operating_status_source_institution": "市场监管总局",
    }

    attachment_rows = datasets["enterprise_attachment_accounts"]["rows"]
    assert [row["normalized"]["attachment_record_type"] for row in attachment_rows] == [
        "account",
        "business",
    ]

    section = projected["sections"][0]
    assert [item["id"] for item in projected["sections"]] == ["sec_enterprise"]
    assert section["items"] == []
    assert section["groups"] == []
    assert set(section["dataset_refs"]) == {dataset["id"] for dataset in datasets.values()}
    reading_dataset_ids = {
        table["dataset_id"] for table in projected["reading"]["tables"]
    }
    assert reading_dataset_ids == set(section["dataset_refs"])
    assert [item["order"] for item in projected["reading"]["document_flow"]] == list(
        range(1, len(projected["reading"]["document_flow"]) + 1)
    )


@pytest.mark.parametrize("pool", ["raw", "canonical_raw"])
def test_enterprise_v4_forbids_raw_pools_without_weakening_v3(pool: str) -> None:
    projected = project_enterprise_community_json(_payload())
    validation = validate_projection_payload("community", projected)
    assert validation.valid, validation.errors

    invalid_v4 = deepcopy(projected)
    invalid_v4["datasets"][0]["rows"][0][pool] = {}
    assert not validate_projection_payload("community", invalid_v4).valid

    legacy = deepcopy(projected)
    legacy["schema"]["version"] = "3.0.0"
    for dataset in legacy["datasets"]:
        for row in dataset["rows"]:
            row["raw"] = deepcopy(row["normalized"])
            row["canonical_raw"] = deepcopy(row["normalized"])
    assert validate_projection_payload("community", legacy).valid
    legacy["datasets"][0]["rows"][0].pop(pool)
    assert not validate_projection_payload("community", legacy).valid


def test_enterprise_json_replay_keeps_normalized_rows_without_raw_projections() -> None:
    from docmirror.server.output_builder import materialize_community_bundle

    payload = _payload()
    payload["datasets"] = [
        dataset for dataset in payload["datasets"] if dataset["name"] == "enterprise_profile"
    ]
    projected = project_enterprise_community_json(payload)
    restored = materialize_community_bundle(projected, ParseResult())
    replayed = restored.json_payload()

    assert replayed["schema"]["version"] == "4.0.0"
    assert replayed["datasets"] == projected["datasets"]
    assert validate_projection_payload("community", replayed).valid


def test_enterprise_artifact_projection_uses_clean_columns_with_rich_evidence() -> None:
    payload = _payload()
    semantic = {
        "datasets": deepcopy(payload["datasets"]),
        "structure": {
            "sections": deepcopy(payload["sections"]),
            "source_tables": [{"id": "source-table", "page": 2}],
        },
        "reading": deepcopy(payload["reading"]),
        "domain": {"facts": {"report_subtype": "enterprise"}},
    }
    before = deepcopy(semantic)
    public = project_enterprise_community_json(payload)
    public_before = deepcopy(public)

    artifact = project_enterprise_artifact_semantic(semantic, public)

    assert semantic == before
    assert public == public_before
    datasets = {dataset["name"]: dataset for dataset in artifact["datasets"]}
    assert "report_notes" not in datasets
    displayed = datasets["enterprise_displayed_credit_summary"]["rows"][0]
    assert displayed["normalized"]["balance"] == "45.00"
    assert displayed["canonical_raw"]["balance"] == "45.00"
    assert displayed["raw"]["account_count"] == 2
    assert displayed["source"]["source_refs"] == [
        {"page": 2, "table_id": "source-table"}
    ]
    assert displayed["confidence"] == 1.0
    liability = datasets["enterprise_repayment_responsibility_accounts"]["rows"]
    assert liability[0]["normalized"]["overdue_months"] == 0
    assert liability[0]["canonical_raw"]["overdue_months"] == "0"
    assert liability[1]["raw"]["repayment_status"] == "N"


def test_enterprise_public_projection_rejects_unclassified_business_fields() -> None:
    payload = _payload()
    profile = next(
        dataset
        for dataset in payload["datasets"]
        if dataset["name"] == "enterprise_profile"
    )
    profile["rows"][0]["normalized"]["future_business_field"] = "value"

    with pytest.raises(ValueError, match="future_business_field"):
        project_enterprise_community_json(payload)


def test_enterprise_public_projection_rejects_unknown_amount_kind() -> None:
    payload = _payload()
    displayed = next(
        dataset
        for dataset in payload["datasets"]
        if dataset["name"] == "enterprise_displayed_credit_summary"
    )
    displayed["rows"][0]["normalized"]["amount_kind"] = "future_amount"

    with pytest.raises(ValueError, match="future_amount"):
        project_enterprise_community_json(payload)


def test_enterprise_public_projection_rejects_unmaterialized_section_business_value() -> None:
    payload = _payload()
    payload["sections"][0]["items"].append(
        {
            "key": "reported_account_count",
            "label": "报告账户数",
            "value": 3,
            "raw": "3",
            "type": "integer",
        }
    )

    with pytest.raises(ValueError, match="reported account totals"):
        project_enterprise_community_json(payload)


def test_enterprise_public_projection_proves_section_totals_before_pruning() -> None:
    payload = _payload()
    summary = _dataset(
        "enterprise_current_credit_summary",
        [
            _row(
                "summary:1",
                {
                    "current_summary_id": "summary:1",
                    "transaction_group": "借贷交易",
                    "business_category": "合计",
                    "is_total": True,
                    "total_account_count": 3,
                    "total_balance": "65.41",
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                },
            ),
            _row(
                "summary:2",
                {
                    "current_summary_id": "summary:2",
                    "transaction_group": "担保交易",
                    "business_category": "合计",
                    "is_total": True,
                    "total_account_count": 3,
                    "total_balance": "65.41",
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                },
            ),
            _row(
                "summary:3",
                {
                    "current_summary_id": "summary:3",
                    "transaction_group": "借贷交易",
                    "business_category": "短期借款",
                    "is_total": False,
                    "total_account_count": 3,
                    "total_balance": "65.41",
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                },
            ),
            _row(
                "summary:4",
                {
                    "current_summary_id": "summary:4",
                    "transaction_group": "担保交易",
                    "business_category": "短期借款",
                    "is_total": False,
                    "total_account_count": 3,
                    "total_balance": "65.41",
                    "currency": "CNY",
                    "amount_unit": "CNY_10K",
                },
            ),
        ],
    )
    payload["datasets"].append(summary)
    payload["sections"][0]["dataset_refs"].append(summary["id"])
    payload["reading"]["document_flow"].append(
        {"order": 99, "kind": "dataset", "ref_id": summary["id"]}
    )
    payload["reading"]["tables"].append(
        {
            "id": f"reading:{summary['id']}",
            "dataset_id": summary["id"],
            "section_id": "sec_enterprise",
            "title": summary["label"],
            "column_keys": [column["key"] for column in summary["columns"]],
            "row_count": summary["row_count"],
        }
    )
    payload["sections"][0]["items"].extend(
        [
            {"key": "reported_account_count", "value": 3},
            {"key": "reported_account_balance", "value": "65.41"},
        ]
    )
    payload["sections"][0]["groups"].extend(
        [
            {
                "key": "reported_account_counts",
                "items": [{"key": "短期借款", "value": 3}],
            },
            {
                "key": "reported_account_balances",
                "items": [{"key": "短期借款", "value": "65.41"}],
            },
        ]
    )

    projected = project_enterprise_community_json(payload)

    assert projected["sections"][0]["items"] == []


def test_enterprise_public_projection_retains_only_conflicting_identity_assertions() -> None:
    payload = _payload()
    identity = _dataset(
        "enterprise_report_identity",
        [
            _row(
                "identity:1",
                {
                    "enterprise_identity_id": "identity:1",
                    "subject_name": "身份名称",
                    "identity_subject_name": "身份名称",
                    "cover_subject_name": "封面名称",
                    "subject_name_assertion_status": "conflict",
                    "zhongzheng_code": "CODE-1",
                    "cover_zhongzheng_code": "CODE-1",
                },
            )
        ],
    )
    payload["datasets"].append(identity)
    payload["sections"][0]["dataset_refs"].append(identity["id"])
    payload["sections"][0]["items"].extend(
        [
            {"key": "subject_name", "value": "封面名称"},
            {"key": "identity_subject_name", "value": "身份名称"},
            {"key": "cover_subject_name", "value": "封面名称"},
            {"key": "subject_name_assertion_status", "value": "conflict"},
            {"key": "zhongzheng_code", "value": "CODE-1"},
            {"key": "cover_zhongzheng_code", "value": "CODE-1"},
        ]
    )

    projected = project_enterprise_community_json(payload)
    projected_identity = next(
        dataset
        for dataset in projected["datasets"]
        if dataset["name"] == "enterprise_report_identity"
    )["rows"][0]["normalized"]

    assert projected_identity == {
        "subject_name": "身份名称",
        "cover_subject_name": "封面名称",
        "zhongzheng_code": "CODE-1",
    }


def test_enterprise_section_total_cannot_match_a_guarantee_total() -> None:
    payload = _payload()
    payload["datasets"].append(
        _dataset(
            "enterprise_current_credit_summary",
            [
                _row(
                    "summary:guarantee",
                    {
                        "current_summary_id": "summary:guarantee",
                        "transaction_group": "担保交易",
                        "business_category": "合计",
                        "is_total": True,
                        "total_account_count": 3,
                        "total_balance": "65.41",
                    },
                )
            ],
        )
    )
    payload["sections"][0]["items"].extend(
        [
            {"key": "reported_account_count", "value": 3},
            {"key": "reported_account_balance", "value": "65.41"},
        ]
    )

    with pytest.raises(ValueError, match="reported account totals"):
        project_enterprise_community_json(payload)


@pytest.mark.parametrize("duplicate_kind", ["item", "group"])
def test_enterprise_section_projection_rejects_duplicate_keys(
    duplicate_kind: str,
) -> None:
    payload = _payload()
    if duplicate_kind == "item":
        payload["sections"][0]["items"].append(
            {"key": "source_account_summary_page", "value": 3}
        )
    else:
        payload["sections"][0]["groups"].append(
            {"key": "extracted_public_record_type_counts", "items": []}
        )

    with pytest.raises(ValueError, match="duplicate enterprise Community section"):
        project_enterprise_community_json(payload)
