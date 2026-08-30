# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.personal_brief_native.projector import (
    personal_brief_public_dataset_policy,
    project_personal_brief_artifact_semantic,
    project_personal_brief_community_json,
)
from docmirror.plugins.credit_report.personal_brief_native.schema import (
    personal_brief_semantic_extensions,
)


def _keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_keys(item))
    return keys


def _column(key: str) -> dict[str, object]:
    return {
        "key": key,
        "label": key,
        "type": "string",
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
            "page_range": [1, 2],
            "source_refs": [{"page": 1, "node_id": "node:1"}],
        },
        "confidence": 1.0,
        "review": {"required": False},
    }


def _dataset(name: str, section_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    keys = list(dict.fromkeys(key for row in rows for key in row["normalized"]))
    return {
        "id": f"ds_{name}",
        "name": name,
        "label": name,
        "type": "records",
        "section_id": section_id,
        "csv": f"001_datasets/{name}.csv",
        "row_count": len(rows),
        "grain": f"one row per {name}",
        "primary_key": "record_id",
        "schema_version": "1.0",
        "status": "complete" if rows else "empty",
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


def _section(
    section_id: str,
    section_type: str,
    dataset_names: list[str],
    *,
    items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": section_id,
        "title": section_type,
        "type": section_type,
        "page_range": [1, 2],
        "items": list(items or []),
        "groups": [],
        "dataset_refs": [f"ds_{name}" for name in dataset_names],
    }


def _payload() -> dict[str, object]:
    datasets = [
        _dataset(
            "personal_report_metadata",
            "sec_header",
            [
                _row(
                    "metadata:1",
                    {
                        "personal_report_metadata_id": "metadata:1",
                        "report_number": "R-1",
                        "report_time": "2026-08-08T10:00:00+08:00",
                        "subject_name": "Alice",
                        "primary_id_type": "身份证",
                        "primary_id_number": "110101199001010000",
                        "marital_status": "married",
                        "marital_status_raw": "已婚",
                        "reporting_currency": "CNY",
                        "reporting_amount_unit": "CNY_1",
                        "reporting_amount_precision": 0,
                        "amount_policy_source": "personal_brief_standard",
                    },
                )
            ],
        ),
        _dataset(
            "credit_accounts",
            "sec_credit",
            [
                _row(
                    "credit_account:1",
                    {
                        "sequence": 1,
                        "account_id": "credit_account:1",
                        "account_identifier": None,
                        "account_type": "credit_card",
                        "business_category": "credit_cards",
                        "institution": "Bank A",
                        "business_type": "贷记卡",
                        "credit_card_type": "credit_card",
                        "card_tail": "1234",
                        "open_date": "2020-01-01",
                        "snapshot_date": "2026-07",
                        "due_date": None,
                        "account_currency": "CNY",
                        "currency": "CNY",
                        "reporting_amount_currency": "CNY",
                        "amount_unit": "CNY_1",
                        "reporting_amount_unit": "CNY_1",
                        "reporting_amount_precision": 0,
                        "credit_limit": "10000",
                        "credit_limit_status": "reported",
                        "balance": "100",
                        "balance_status": "reported",
                        "account_state": "open",
                        "account_lifecycle_state": "open",
                        "activation_state": "active",
                        "card_activation_state": "activated",
                        "payoff_state": "not_applicable",
                        "credit_quality_status": "not_reported",
                        "ever_overdue": True,
                        "current_overdue": False,
                        "overdue_months": 1,
                        "over_90_days": False,
                        "status": "active",
                        "source_section": "credit_cards",
                        "source_sequence": 1,
                    },
                )
            ],
        ),
        _dataset(
            "overdue_records",
            "sec_credit",
            [
                _row(
                    "overdue:1",
                    {
                        "overdue_id": "overdue:1",
                        "sequence": 1,
                        "account_id": "credit_account:1",
                        "account_type": "credit_card",
                        "institution": "Bank A",
                        "business_type": "贷记卡",
                        "card_tail": "1234",
                        "open_date": "2020-01-01",
                        "currency": "CNY",
                        "period_scope": "last_5_years",
                        "overdue_months": 1,
                        "over_90_days_months": 0,
                        "current_overdue": False,
                        "current_overdue_status": "not_overdue",
                        "over_90_days": False,
                    },
                )
            ],
        ),
        _dataset(
            "tax_arrears_records",
            "sec_public",
            [
                _row(
                    "tax:1",
                    {
                        "tax_arrears_id": "tax:1",
                        "sequence": 1,
                        "tax_authority": "Tax office",
                        "statistics_date": "2026-06",
                        "arrears_amount": "100",
                        "taxpayer_identifier": "TAX-1",
                        "reporting_amount_currency": "CNY",
                        "reporting_amount_unit": "CNY_1",
                    },
                )
            ],
        ),
        _dataset(
            "public_records",
            "sec_public",
            [
                _row(
                    "public:1",
                    {
                        "public_record_id": "public:1",
                        "sequence": 1,
                        "record_type": "tax_arrears",
                        "authority": "Tax office",
                        "content": "{}",
                    },
                )
            ],
        ),
        _dataset("repayment_records", "sec_credit", []),
        _dataset(
            "report_notes",
            "sec_notes",
            [_row("note:1", {"note_id": "note:1", "sequence": 1, "content": "legal"})],
        ),
    ]
    sections = [
        _section(
            "sec_header",
            "report_header",
            ["personal_report_metadata"],
            items=[
                {
                    "key": "canonical_ir_schema_version",
                    "label": "schema",
                    "value": "1.0",
                    "raw": "1.0",
                    "type": "string",
                }
            ],
        ),
        _section(
            "sec_credit",
            "credit_details",
            ["credit_accounts", "overdue_records", "repayment_records"],
        ),
        _section(
            "sec_non_credit",
            "non_credit_transactions",
            [],
            items=[
                {
                    "key": "record_status",
                    "label": "status",
                    "value": "not_reported",
                    "raw": "not_reported",
                    "type": "string",
                },
                {
                    "key": "source_page",
                    "label": "page",
                    "value": 2,
                    "raw": "2",
                    "type": "integer",
                },
            ],
        ),
        _section(
            "sec_public",
            "public_records",
            ["tax_arrears_records", "public_records"],
        ),
        _section("sec_notes", "notes", ["report_notes"]),
    ]
    dataset_ids = [dataset["id"] for dataset in datasets]
    return {
        "schema": {
            "name": "docmirror.community",
            "version": "3.0.0",
            "edition": "community",
            "domain": "credit_report",
            "support_level": "beta",
        },
        "document": {
            "id": "doc:1",
            "type": "personal_credit_report_brief",
            "title": "Personal credit report",
            "page_count": 2,
            "language": ["zh-CN"],
            "source_file": {
                "name": "brief.pdf",
                "mime_type": "application/pdf",
                "sha256": "sha256:test",
            },
            "units": {},
        },
        "sections": sections,
        "datasets": datasets,
        "reading": {
            "version": "1.0",
            "profile": "community",
            "document_flow": [
                {"order": 1, "kind": "document", "ref_id": "doc:1"},
                *[
                    {"order": index + 2, "kind": "section", "ref_id": section["id"]}
                    for index, section in enumerate(sections)
                ],
                *[
                    {
                        "order": index + len(sections) + 2,
                        "kind": "dataset",
                        "ref_id": dataset_id,
                    }
                    for index, dataset_id in enumerate(dataset_ids)
                ],
            ],
            "tables": [
                {
                    "id": f"reading:{dataset['id']}",
                    "dataset_id": dataset["id"],
                    "section_id": dataset["section_id"],
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


def test_personal_brief_public_policy_covers_the_closed_world_schema() -> None:
    policy = personal_brief_public_dataset_policy()

    assert tuple(policy) == tuple(personal_brief_semantic_extensions()["dataset_document_order"])
    assert policy["repayment_records"] is None
    assert policy["public_records"] is None
    assert policy["report_notes"] is None
    assert "account_id" in policy["credit_accounts"]
    assert "status" not in policy["credit_accounts"]
    assert "credit_limit_status" in policy["credit_accounts"]
    assert "responsibility_amount_reported" in policy["repayment_liability_records"]


def test_personal_brief_public_projection_is_lean_and_non_mutating() -> None:
    payload = _payload()
    before = deepcopy(payload)

    projected = project_personal_brief_community_json(payload)

    assert payload == before
    assert projected["schema"]["version"] == "4.0.0"
    assert not {"raw", "canonical_raw"} & _keys(projected)
    validation = validate_projection_payload("community", projected)
    assert validation.valid, validation.errors

    datasets = {dataset["name"]: dataset for dataset in projected["datasets"]}
    assert not {"repayment_records", "public_records", "report_notes"} & set(datasets)
    metadata = datasets["personal_report_metadata"]
    assert metadata["rows"][0]["normalized"] == {
        "report_number": "R-1",
        "report_time": "2026-08-08T10:00:00+08:00",
        "subject_name": "Alice",
        "primary_id_type": "身份证",
        "primary_id_number": "110101199001010000",
        "marital_status": "married",
        "marital_status_raw": "\u5df2\u5a5a",
        "reporting_currency": "CNY",
        "reporting_amount_unit": "CNY_1",
        "reporting_amount_precision": 0,
    }
    account = datasets["credit_accounts"]["rows"][0]
    assert account["normalized"]["account_id"] == "credit_account:1"
    assert account["normalized"]["account_lifecycle_state"] == "open"
    assert not {
        "sequence",
        "status",
        "source_section",
        "source_sequence",
        "currency",
        "amount_unit",
        "account_state",
        "activation_state",
    } & set(account["normalized"])
    assert account["normalized"]["credit_limit_status"] == "reported"
    assert account["normalized"]["balance_status"] == "reported"
    assert [column["key"] for column in datasets["credit_accounts"]["columns"]] == list(
        personal_brief_public_dataset_policy()["credit_accounts"]
    )
    account_columns = {
        column["key"]: column for column in datasets["credit_accounts"]["columns"]
    }
    assert account_columns["account_type"]["enum"] == {
        "credit_card": "信用卡",
        "loan": "贷款",
        "credit_line": "贷款授信",
        "other_business": "其他业务",
    }
    assert account_columns["credit_limit"]["unit"] == "CNY_1"
    assert account_columns["reporting_amount_unit"]["enum"] == {
        "CNY_1": "元（人民币）"
    }
    assert "account_identifier" not in account["normalized"]
    assert "account_identifier" in {
        column["key"] for column in datasets["credit_accounts"]["columns"]
    }
    for dataset in datasets.values():
        assert not {"reading_columns", "storage_role", "record_path"} & set(dataset)
        for row in dataset["rows"]:
            assert set(row) == {"record_id", "normalized"}
        assert all(not column["raw_available"] and not column["evidence_available"] for column in dataset["columns"])

    sections = {section["id"]: section for section in projected["sections"]}
    assert sections["sec_header"]["items"] == []
    assert [item["key"] for item in sections["sec_non_credit"]["items"]] == ["record_status"]
    assert "additional_values" not in sections["sec_non_credit"]["items"][0]
    assert "sec_notes" not in sections
    assert "ds_public_records" not in sections["sec_public"]["dataset_refs"]
    public_ids = {dataset["id"] for dataset in datasets.values()}
    assert {table["dataset_id"] for table in projected["reading"]["tables"]} == public_ids
    for table in projected["reading"]["tables"]:
        dataset = next(item for item in datasets.values() if item["id"] == table["dataset_id"])
        assert table["column_keys"] == [column["key"] for column in dataset["columns"]]
    assert [item["order"] for item in projected["reading"]["document_flow"]] == list(
        range(1, len(projected["reading"]["document_flow"]) + 1)
    )
    assert {item["ref_id"] for item in projected["reading"]["document_flow"] if item["kind"] == "dataset"} == public_ids


def test_personal_brief_public_projection_preserves_distinct_section_business_text() -> None:
    payload = _payload()
    section = next(section for section in payload["sections"] if section["id"] == "sec_non_credit")
    section["items"][0]["raw"] = "未记录非信贷交易信息"
    section["items"].append(
        {"key": "lookback_years", "label": "统计年限", "value": 5, "raw": "5", "type": "integer"}
    )
    before = deepcopy(payload)

    projected = project_personal_brief_community_json(payload)

    assert payload == before
    items = next(section for section in projected["sections"] if section["id"] == "sec_non_credit")["items"]
    assert items[0]["value"] == "not_reported"
    assert items[0]["additional_values"] == ["未记录非信贷交易信息"]
    assert items[1]["value"] == 5
    assert "additional_values" not in items[1]
    assert not {"raw", "canonical_raw"} & _keys(projected)
    assert validate_projection_payload("community", projected).valid
    assert project_personal_brief_community_json(projected) == projected


@pytest.mark.parametrize("empty", [False, True], ids=["populated", "empty"])
def test_personal_brief_v4_projection_is_idempotent(empty: bool) -> None:
    payload = _payload()
    if empty:
        payload["datasets"] = []
        payload["sections"] = []

    projected = project_personal_brief_community_json(payload)

    assert projected["schema"]["version"] == "4.0.0"
    assert not {"raw", "canonical_raw"} & _keys(projected)
    assert validate_projection_payload("community", projected).valid
    assert project_personal_brief_community_json(projected) == projected
    if empty:
        assert projected["datasets"] == []
        assert projected["sections"] == []
        assert projected["reading"]["tables"] == []


@pytest.mark.parametrize("location", ["record_raw", "record_canonical_raw", "section_raw", "group_raw"])
def test_personal_brief_v4_schema_rejects_reintroduced_source_value_pools(location: str) -> None:
    projected = project_personal_brief_community_json(_payload())
    if location.startswith("record_"):
        target = projected["datasets"][0]["rows"][0]
        key = location.removeprefix("record_")
        value = {}
    else:
        section = next(section for section in projected["sections"] if section["id"] == "sec_non_credit")
        target = section["items"][0]
        if location == "group_raw":
            target = deepcopy(target)
            section["groups"] = [{"key": "status", "label": "记录状态", "items": [target]}]
        key = "raw"
        value = "not_reported"

    assert validate_projection_payload("community", projected).valid
    target[key] = value

    validation = validate_projection_payload("community", projected)
    assert not validation.valid
    assert validation.errors


def test_personal_brief_public_projection_rejects_unclassified_content() -> None:
    payload = _payload()
    account = next(dataset for dataset in payload["datasets"] if dataset["name"] == "credit_accounts")
    account["rows"][0]["normalized"]["future_business_field"] = "new"

    with pytest.raises(ValueError, match="future_business_field"):
        project_personal_brief_community_json(payload)

    payload = _payload()
    repayment = next(dataset for dataset in payload["datasets"] if dataset["name"] == "repayment_records")
    repayment["rows"] = [_row("repayment:1", {"repayment_id": "repayment:1"})]
    with pytest.raises(ValueError, match="monthly repayment records"):
        project_personal_brief_community_json(payload)

    payload = _payload()
    payload["datasets"].append(_dataset("unknown_dataset", "sec_credit", []))
    with pytest.raises(ValueError, match="unknown_dataset"):
        project_personal_brief_community_json(payload)

    payload = _payload()
    account = next(dataset for dataset in payload["datasets"] if dataset["name"] == "credit_accounts")
    account["rows"][0]["normalized"]["currency"] = "USD"
    with pytest.raises(ValueError, match="currency is not conserved by account_currency"):
        project_personal_brief_community_json(payload)


def test_personal_brief_artifact_projection_leaves_rich_semantic_untouched() -> None:
    payload = _payload()
    rich_account = next(dataset for dataset in payload["datasets"] if dataset["name"] == "credit_accounts")["rows"][0]
    rich_account["canonical_raw"]["used_amount"] = "not-a-number"
    rich_account["raw"]["used_amount"] = "not-a-number"
    projected = project_personal_brief_community_json(payload)
    semantic = {
        "datasets": deepcopy(payload["datasets"]),
        "structure": {"sections": deepcopy(payload["sections"]), "blocks": [{"id": "b1"}]},
        "reading": deepcopy(payload["reading"]),
        "bindings": [
            {
                "dataset_id": "ds_personal_report_metadata",
                "record_id": "metadata:1",
                "source_block_refs": ["b1"],
            },
            {
                "dataset_id": "ds_report_notes",
                "record_id": "note:1",
                "source_block_refs": ["b1"],
            },
        ],
    }
    before = deepcopy(semantic)
    public_before = deepcopy(projected)

    artifact = project_personal_brief_artifact_semantic(semantic, projected)

    assert semantic == before
    assert projected == public_before
    assert not {"raw", "canonical_raw"} & _keys(projected)
    assert [dataset["name"] for dataset in artifact["datasets"]] == [
        dataset["name"] for dataset in projected["datasets"]
    ]
    assert artifact["datasets"][0]["rows"][0]["source"]["source_refs"]
    assert artifact["datasets"][0]["rows"][0]["confidence"] == 1.0
    assert set(artifact["datasets"][0]["rows"][0]["canonical_raw"]) == set(
        artifact["datasets"][0]["rows"][0]["normalized"]
    )
    assert artifact["datasets"][0]["rows"][0]["raw"] == artifact["datasets"][0]["rows"][0]["canonical_raw"]
    assert semantic["datasets"][0]["rows"][0]["source"]["source_refs"]
    assert artifact["structure"]["blocks"] == [{"id": "b1"}]
    assert artifact["structure"]["sections"] == projected["sections"]
    artifact_account = next(dataset for dataset in artifact["datasets"] if dataset["name"] == "credit_accounts")
    assert "used_amount" not in artifact_account["rows"][0]["normalized"]
    assert artifact_account["rows"][0]["canonical_raw"]["used_amount"] == "not-a-number"
    assert artifact_account["rows"][0]["raw"]["used_amount"] == "not-a-number"
    assert {binding["dataset_id"] for binding in artifact["bindings"]} == {
        "ds_personal_report_metadata"
    }
