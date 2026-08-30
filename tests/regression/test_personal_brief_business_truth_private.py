"""Persisted public JSON versus per-PDF, independently authored source truth."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path, PureWindowsPath

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.server.edition_outputs import write_outputs
from scripts.validate.validate_community_artifacts import validate_community_artifacts
from tests._personal_brief_business_truth import audit_business_truth

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures-private/credit_report"
_STANDARDS = _FIXTURES / "personal_brief_business_truth_v2"
_CANONICAL_STANDARD = Path(__file__).resolve().parent / "goldens/personal_brief_canonical_sample.json"
_CASES = (
    "main_01", "main_02", "main_03", "main_04",
    "external_01", "external_02", "external_03", "canonical_sample",
)


@pytest.mark.parametrize("case", _CASES)
def test_personal_brief_every_source_business_fact(case, tmp_path):
    path = _CANONICAL_STANDARD if case == "canonical_sample" else _STANDARDS / f"{case}.json"
    if not path.exists():
        pytest.skip("Independent private source standards are not distributed")
    standard = json.loads(path.read_text(encoding="utf-8"))
    directory = (
        _FIXTURES
        if case == "canonical_sample"
        else _FIXTURES / (
            "External/Digital Personal Brief"
            if case.startswith("external")
            else "Digital Personal Brief"
        )
    )
    pdf = directory / PureWindowsPath(standard["source_pdf"]).name
    if not pdf.exists():
        pytest.skip(f"Private PDF is unavailable: {pdf.name}")
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == standard["source_sha256"], "Re-audit changed source PDFs; never regenerate truth from output"
    sealed = asyncio.run(perceive_document(pdf, PerceiveOptions(policy=normalize_parse_policy(
        enhance_mode="standard", doc_type_hint="credit_report:force"))))
    _, written = write_outputs(sealed, tmp_path, file_path=str(pdf), task_id="parsed", include_mirror=False, include_manifest=False)
    payload = json.loads(written["community"].read_text(encoding="utf-8"))
    assert validate_projection_payload("community", payload).valid
    assert validate_community_artifacts(written["community"]) == []
    audit = audit_business_truth(standard, payload)
    assert audit["correct"] == audit["source_facts"] == standard["fact_count"], [
        value for value in audit["field_audit"] if value["verdict"] != "correct"
    ]
    assert not audit["value_failures"], audit["value_failures"]
    assert not audit["structure_issues"], audit["structure_issues"]
    assert audit["passed"]
