# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.mirror.domain_access import micro_grid_structures_from_domain_specific
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins._base.kv_community_enrich import _canonicalize_credit_accounts
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    build_personal_detail_extraction_context,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema_v2 import (
    PBOC_DATASET_ORDER,
    PBOC_SCHEMA_VERSION_ENV,
)
from docmirror.plugins.credit_report.scanned_business import link_repayment_records_to_accounts
from docmirror.server.output_builder import build_community_bundle

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]

_FIXTURE_DIR = Path("tests/fixtures-private/credit_report/Scanned Personal Detailed")
_FIXTURES = sorted(_FIXTURE_DIR.glob("*.pdf"))
_EXPECTED_SCHEMA_INPUT_COUNTS = {
    "余泽熙7.15征信.pdf": (27, 641),
    "叶永燕征信.pdf": (42, 884),
    "征信.pdf": (43, 408),
    "林岚挺征信.pdf": (48, 801),
    "洪晓鑫征信报告2025.11.05.pdf": (8, 176),
    "王根镇征信.pdf": (61, 757),
}


def _perceive(fixture: Path):
    return asyncio.run(
        perceive_document(
            fixture,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda path: path.name)
def test_personal_detail_ocr_correction_invariants(
    fixture: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise source correction and the schema boundary on every private report."""
    sealed = _perceive(fixture)
    result = sealed.to_read_view()
    context = build_personal_detail_extraction_context(result)
    domain_specific = result.entities.domain_specific
    raw_bundles = domain_specific.get("_page_evidence_bundles") or []
    raw_snapshot = deepcopy(raw_bundles)
    topology_audit = context.page_topology_audit()

    corrected_pages = context.corrected_evidence_pages()
    business = context.scanned_business(result.full_text or "")
    native_business = context.native_business(result.full_text or "")
    repayment_records = context.corrected_repayment_records()
    audit = context.ocr_correction_audit()
    ocr_bundles = [
        bundle
        for bundle in raw_bundles
        if isinstance(bundle, dict)
        and isinstance(bundle.get("local_structure_evidence"), dict)
        and bundle["local_structure_evidence"].get("lines")
    ]

    assert raw_bundles == raw_snapshot
    assert topology_audit["valid"] is True
    assert topology_audit["logical_page_count"] == len(result.pages)
    assert all(1 <= len(logical_pages) <= 2 for logical_pages in topology_audit["logical_pages_by_source"].values())
    assert sorted(context.reading_order_by_logical.values()) == list(
        range(1, len(context.reading_order_by_logical) + 1)
    )
    assert context.entity_context.content_conserved is True
    replayed_pages = [page for page in corrected_pages if page.get("plugin_replayed_subpage")]
    replayed_sources = {int(page.get("source_page") or 0) for page in replayed_pages}
    if replayed_pages:
        raw_counts = Counter(
            int(
                bundle.get("source_page_number")
                or (bundle.get("local_structure_evidence") or {}).get("source_page")
                or 0
            )
            for bundle in ocr_bundles
        )
        corrected_counts = Counter(int(page.get("source_page") or 0) for page in corrected_pages)
        assert all(
            sorted(
                int(page.get("segment_index") or 0)
                for page in replayed_pages
                if int(page.get("source_page") or 0) == source
            )
            == [0, 1]
            for source in replayed_sources
        )
        assert all(corrected_counts[source] == 2 and raw_counts[source] == 1 for source in replayed_sources)
        assert all(
            corrected_counts[source] == count for source, count in raw_counts.items() if source not in replayed_sources
        )
    else:
        assert len(corrected_pages) == len(ocr_bundles)
    assert audit["targeted_ocr_requests"] <= 8
    assert all(
        decision["original"] != decision["corrected"]
        and decision["action"] in {"applied", "suggested"}
        and decision["role"]
        for decision in audit["decisions"]
    )

    accounts = business.get("credit_accounts") or []
    expected_accounts, expected_repayments = _EXPECTED_SCHEMA_INPUT_COUNTS[fixture.name]
    assert len(accounts) == expected_accounts
    linked_repayments = link_repayment_records_to_accounts(
        repayment_records,
        _canonicalize_credit_accounts(accounts),
        micro_grid_structures_from_domain_specific(domain_specific),
        reading_order_by_logical=dict(context.reading_order_by_logical),
        force_relink=True,
    )
    assert len(linked_repayments) == expected_repayments
    assert all(record.get("account_id") for record in linked_repayments)
    account_ids = [account.get("account_id") for account in accounts]
    assert len(account_ids) == len(set(account_ids))
    logical_pages = set(context.source_page_by_logical)
    assert all(
        int(ref.get("logical_page") or ref.get("page") or 0) in logical_pages
        for account in accounts
        for ref in account.get("source_refs") or []
        if isinstance(ref, dict)
    )
    assert all(
        int(ref.get("page") or ref.get("logical_page") or 0) in logical_pages
        for record in repayment_records
        for ref in record.get("source_cell_refs") or []
        if isinstance(ref, dict) and (ref.get("page") or ref.get("logical_page"))
    )
    assert isinstance(native_business.get("credit_accounts") or [], list)

    inquiries = business.get("inquiry_records") or []
    for inquiry_type in {row.get("inquiry_type") for row in inquiries}:
        sequences = [row.get("sequence") for row in inquiries if row.get("inquiry_type") == inquiry_type]
        # OCR-visible gaps remain explicit evidence of a missed source row, but
        # duplicate/backward row numbers must not leak into reconstruction.
        assert sequences == sorted(set(sequences))

    monkeypatch.setenv(PBOC_SCHEMA_VERSION_ENV, "2.0.0")
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    assert validate_projection_payload("community", payload).valid
    v2_validation = validate_projection_payload("personal_credit_report_detailed_v2", payload)
    assert v2_validation.valid, v2_validation.errors
    assert payload["document"]["domain_schema"]["version"] == "2.0.0"
    v2_datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    assert v2_datasets["credit_accounts"]["row_count"] == expected_accounts
    assert v2_datasets["credit_account_monthly_performance"]["row_count"] == expected_repayments
    v2_statuses = {
        row["normalized"]["dataset_name"]: row["normalized"] for row in v2_datasets["dataset_status"]["rows"]
    }
    assert set(v2_statuses) == set(PBOC_DATASET_ORDER) - {"dataset_status"}
    assert all(
        status["observed_row_count"] == v2_datasets.get(name, {}).get("row_count", 0)
        for name, status in v2_statuses.items()
    )

    if fixture.name.startswith("叶永燕"):
        institutional = [row for row in inquiries if row.get("inquiry_type") == "institution"]
        personal = [row for row in inquiries if row.get("inquiry_type") == "personal"]

        # Source-grounded totals pin reconstruction quality, not JSON shape.
        assert len(accounts) == 42
        assert sum(bool(account.get("account_identifier")) for account in accounts) >= 28
        assert len(institutional) == 96
        assert len(personal) == 16
        assert [row["sequence"] for row in institutional] == list(range(1, 97))
        assert [row["sequence"] for row in personal] == list(range(1, 17))
        assert audit["applied_count"] >= 70
