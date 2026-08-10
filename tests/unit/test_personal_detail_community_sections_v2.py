from __future__ import annotations

import importlib
from types import SimpleNamespace

from docmirror.plugins.credit_report.community_plugin import (
    _apply_personal_detail_dataset_status,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    PersonalDetailScannedVariant,
)


def _row(record_id: str, **normalized):
    return {
        "record_id": record_id,
        "normalized": dict(normalized),
        "canonical_raw": dict(normalized),
    }


def test_final_v2_rows_replace_generic_section_scalars_and_withhold_nulls() -> None:
    variant = PersonalDetailScannedVariant()
    domain_facts = {
        "subject_name": "被查询者证件类型",
        "id_number": "generic-id",
        "id_type": "generic-type",
        "marital_status": "generic-status",
        "query_institution": "generic-query",
        "report_time": "2023.01314:45:37",
        "report_number": "generic-number",
    }
    field_details = {key: {"source": "generic"} for key in domain_facts}
    datasets = {
        "report_metadata": [
            _row(
                "metadata:1",
                subject_name="林岚挺",
                report_number="2023011314453720187289",
                report_time=None,
            )
        ],
        "report_query": [_row("query:1", query_institution="本人")],
        "subject_profile": [_row("profile:1", marital_status="已婚")],
        "subject_identity_documents": [
            _row(
                "identity:1",
                holder_name="林岚挺",
                document_type="身份证",
                document_number="350102198311011 933",
            )
        ],
    }

    resolved = variant.reconcile_final_v2_section_fields(domain_facts, field_details, datasets)

    assert resolved == {
        "subject_name": "林岚挺",
        "id_number": "350102198311011 933",
        "id_type": "身份证",
        "marital_status": "已婚",
        "query_institution": "本人",
        "report_time": None,
        "report_number": "2023011314453720187289",
        "subject_id": "350102198311011 933",
    }
    assert domain_facts["report_time"] is None
    assert "report_time" not in field_details
    assert domain_facts["subject_name"] == datasets["report_metadata"][0]["normalized"]["subject_name"]
    assert field_details["subject_name"]["source"] == "personal_detail_final_v2"
    assert field_details["subject_id"]["source"] == "personal_detail_final_v2_alias"


def test_final_v2_nulls_clear_stale_generic_identity() -> None:
    variant = PersonalDetailScannedVariant()
    domain_facts = {"subject_name": "stale", "id_number": "stale", "subject_id": "stale"}
    field_details = {key: {"source": "generic"} for key in domain_facts}

    resolved = variant.reconcile_final_v2_section_fields(domain_facts, field_details, {})

    assert resolved["subject_name"] is None
    assert resolved["id_number"] is None
    assert resolved["subject_id"] is None
    assert domain_facts["subject_name"] is None
    assert domain_facts["subject_id"] is None
    assert not {"subject_name", "id_number", "subject_id"} & field_details.keys()


def test_build_sections_does_not_synthesize_missing_basic_section(monkeypatch) -> None:
    variant_module = importlib.import_module(
        "docmirror.plugins.credit_report.personal_detail_scanned.variant"
    )
    monkeypatch.setattr(
        variant_module,
        "_classified_sections",
        lambda _parse_result: (
            {
                "id": "sec_credit_summary",
                "title": "信息概要",
                "type": "credit_summary",
                "page_start": 2,
                "page_end": 2,
            },
        ),
    )
    pages = [SimpleNamespace(page_number=page) for page in range(1, 16)]

    sections = PersonalDetailScannedVariant().build_sections(SimpleNamespace(pages=pages), "")

    assert sections == (
        {
            "id": "sec_credit_summary",
            "title": "信息概要",
            "type": "credit_summary",
            "page_start": 2,
            "page_end": 2,
        },
    )


def test_build_sections_does_not_infer_ranges_from_page_count() -> None:
    pages = [SimpleNamespace(page_number=page) for page in range(1, 16)]

    sections = PersonalDetailScannedVariant().build_sections(SimpleNamespace(pages=pages), "")

    assert sections == ()


def test_build_sections_uses_only_registered_canonical_audit_entries() -> None:
    parse_result = SimpleNamespace(
        pages=[],
        canonical_layout_audit=lambda: {
            "registrations": [
                {
                    "status": "registered",
                    "template_id": "report_header_and_identity",
                    "logical_page": 2,
                },
                {
                    "status": "registered",
                    "template_id": "information_summary",
                    "logical_page": 4,
                },
                {
                    "status": "unresolved",
                    "template_id": "public_information",
                    "logical_page": 8,
                },
            ]
        },
    )

    sections = PersonalDetailScannedVariant().build_sections(parse_result, "")

    assert [(section["id"], section["page_start"]) for section in sections] == [
        ("sec_personal_basic", 2),
        ("sec_credit_summary", 4),
    ]


def test_public_dataset_envelope_obeys_personal_detail_source_status() -> None:
    payload = {
        "datasets": [
            {
                "name": "credit_agreements",
                "row_count": 4,
                "rows": [{}, {}, {}, {}],
                "status": "complete",
                "completeness": {
                    "expected_row_count": 4,
                    "emitted_row_count": 4,
                    "omitted_row_count": 0,
                    "verified": True,
                    "basis": "source_report_summary",
                },
            },
            {
                "name": "dataset_status",
                "rows": [
                    {
                        "normalized": {
                            "dataset_name": "credit_agreements",
                            "presence_status": "partial",
                            "expected_row_count": 7,
                            "observed_row_count": 4,
                        }
                    }
                ],
            },
        ]
    }

    _apply_personal_detail_dataset_status(payload)

    dataset = payload["datasets"][0]
    assert dataset["status"] == "partial"
    assert dataset["completeness"] == {
        "expected_row_count": 7,
        "emitted_row_count": 4,
        "omitted_row_count": 3,
        "verified": False,
        "basis": "personal_detail_dataset_status:partial",
    }


def test_public_dataset_envelope_accepts_explicit_source_completeness() -> None:
    payload = {
        "datasets": [
            {"name": "inquiries", "row_count": 2, "rows": [{}, {}]},
            {
                "name": "dataset_status",
                "rows": [
                    {
                        "normalized": {
                            "dataset_name": "inquiries",
                            "presence_status": "observed_nonempty",
                            "observed_row_count": 2,
                        }
                    }
                ],
            },
        ]
    }

    _apply_personal_detail_dataset_status(payload)

    dataset = payload["datasets"][0]
    assert dataset["status"] == "complete"
    assert dataset["completeness"]["verified"] is True
    assert dataset["completeness"]["basis"] == (
        "personal_detail_dataset_status:observed_nonempty"
    )
