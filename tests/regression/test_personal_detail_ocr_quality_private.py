# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    build_personal_detail_extraction_context,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]

_FIXTURE_DIR = Path("tests/fixtures-private/credit_report/Scanned Personal Detailed")
_FIXTURES = sorted(_FIXTURE_DIR.glob("*.pdf"))


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
    ).to_read_view()


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda path: path.name)
def test_personal_detail_ocr_correction_invariants(fixture: Path) -> None:
    """Exercise correction quality without coupling to the output schema."""
    result = _perceive(fixture)
    context = build_personal_detail_extraction_context(result)
    domain_specific = result.entities.domain_specific
    raw_bundles = domain_specific.get("_page_evidence_bundles") or []
    raw_snapshot = deepcopy(raw_bundles)

    corrected_pages = context.corrected_evidence_pages()
    business = context.scanned_business(result.full_text or "")
    audit = context.ocr_correction_audit()
    ocr_bundles = [
        bundle
        for bundle in raw_bundles
        if isinstance(bundle, dict)
        and isinstance(bundle.get("local_structure_evidence"), dict)
        and bundle["local_structure_evidence"].get("lines")
    ]

    assert raw_bundles == raw_snapshot
    assert len(corrected_pages) == len(ocr_bundles)
    assert audit["targeted_ocr_requests"] <= 8
    assert all(
        decision["original"] != decision["corrected"]
        and decision["action"] in {"applied", "suggested"}
        and decision["role"]
        for decision in audit["decisions"]
    )

    accounts = business.get("credit_accounts") or []
    account_ids = [account.get("account_id") for account in accounts]
    assert len(account_ids) == len(set(account_ids))

    inquiries = business.get("inquiry_records") or []
    for inquiry_type in {row.get("inquiry_type") for row in inquiries}:
        sequences = [row.get("sequence") for row in inquiries if row.get("inquiry_type") == inquiry_type]
        # OCR-visible gaps remain explicit evidence of a missed source row, but
        # duplicate/backward row numbers must not leak into reconstruction.
        assert sequences == sorted(set(sequences))

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
