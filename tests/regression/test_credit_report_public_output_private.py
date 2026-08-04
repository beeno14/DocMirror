# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.output.community_bundle import CommunityBundle
from docmirror.server.edition_outputs import write_outputs
from docmirror.server.output_builder import build_community_bundle
from scripts.validate.validate_community_artifacts import validate_community_artifacts

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]

_CASES = (
    pytest.param(
        Path("tests/fixtures-private/credit_report/Digital Enterprise/雯玥轩企业征信(1).pdf"),
        "enterprise",
        id="digital-enterprise",
    ),
    pytest.param(
        Path(
            "tests/fixtures-private/credit_report/Digital Personal Brief/"
            "征信报告_平安银行_20090811_1.pdf"
        ),
        "personal_brief",
        id="digital-personal-brief",
    ),
    pytest.param(
        Path(
            "tests/fixtures-private/credit_report/Scanned Personal Detailed/"
            "洪晓鑫征信报告2025.11.05.pdf"
        ),
        "personal_detail",
        id="scanned-personal-detailed",
    ),
)
_PRIVATE_SOURCE_KEYS = frozenset(
    {
        "bbox",
        "evidence_ids",
        "node_id",
        "node_ids",
        "source_anchor",
        "source_cell_refs",
        "source_fact_ids",
        "source_refs",
    }
)


def _keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_keys(item))
    return keys


@pytest.mark.parametrize(("fixture", "subtype"), _CASES)
def test_credit_report_public_json_compacts_private_source_provenance(
    fixture: Path,
    subtype: str,
    tmp_path: Path,
) -> None:
    if not fixture.exists():
        pytest.skip(f"private credit-report fixture is unavailable: {fixture}")

    sealed = asyncio.run(
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
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    rich_payload = CommunityBundle.json_payload(bundle, semantic)
    public_payload = bundle.json_payload(semantic)

    assert semantic["domain"]["facts"]["report_subtype"] == subtype
    assert _keys(rich_payload["datasets"]) & _PRIVATE_SOURCE_KEYS
    assert not (_keys(public_payload) & _PRIVATE_SOURCE_KEYS)
    assert any(dataset["row_count"] for dataset in public_payload["datasets"])
    assert all(
        set(row["source"]) <= {"page_range"}
        for dataset in public_payload["datasets"]
        for row in dataset["rows"]
    )
    assert validate_projection_payload("community", public_payload).valid

    _task_id, written = write_outputs(
        sealed,
        tmp_path,
        file_path=str(fixture),
        task_id=f"public-output-{subtype}",
        include_mirror=False,
        include_manifest=False,
    )
    persisted = json.loads(written["community"].read_text(encoding="utf-8"))

    assert "community_semantic" not in written
    assert "semantic_json" not in persisted["files"]
    assert not (written["community"].parent / "001_community_semantic.json").exists()
    assert not (_keys(persisted) & _PRIVATE_SOURCE_KEYS)
    assert validate_community_artifacts(written["community"]) == []
