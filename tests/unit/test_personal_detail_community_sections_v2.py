from __future__ import annotations

import importlib
from types import SimpleNamespace

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


def test_personal_basic_section_exists_when_ocr_classification_misses_heading(monkeypatch) -> None:
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

    assert sections[0] == {
        "id": "sec_personal_basic",
        "title": "个人基本信息",
        "type": "basic_information",
        "page_start": 1,
        "page_end": 2,
    }
    assert sum(section.get("id") == "sec_personal_basic" for section in sections) == 1
