from __future__ import annotations

import hashlib
import json

import pytest

from docmirror.models.entities.parse_result import DocumentEntities, ParseResult
from docmirror.models.sealed import seal_parse_result
from scripts.personal_detail_audit_input import write_sealed_audit_input


def test_private_audit_input_preserves_sealed_raw_evidence_and_source_digest(tmp_path):
    fixture = tmp_path / "simulated-business-report.pdf"
    fixture.write_bytes(b"synthetic source identity; not parsed as a PDF")
    sealed = seal_parse_result(
        ParseResult(
            entities=DocumentEntities(
                document_type="credit_report",
                domain_specific={
                    "raw_excerpt": {
                        "evidence_ids": ["sealed-original-1"],
                        "text": "2021年还款记录 N 0",
                        "bbox": [0.0, 12.0, 25.0, 26.0],
                        "confidence": 0.2325,
                    }
                },
            )
        )
    )
    original = sealed.model_dump(mode="json", exclude_none=False)

    target = write_sealed_audit_input(sealed, fixture, tmp_path / "audit")
    payload = json.loads(target.read_text(encoding="utf-8"))
    recovered = seal_parse_result(ParseResult.model_validate(payload["parse_result"]))

    assert payload["source_pdf_name"] == fixture.name
    assert payload["source_pdf_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert payload["sealed_integrity_fingerprint"] == sealed.integrity_fingerprint
    assert recovered.integrity_fingerprint == sealed.integrity_fingerprint
    assert recovered.model_dump(mode="json", exclude_none=False) == original
    assert sealed.model_dump(mode="json", exclude_none=False) == original


def test_private_audit_input_refuses_to_replace_prior_evidence(tmp_path):
    fixture = tmp_path / "simulated-business-report.pdf"
    fixture.write_bytes(b"first simulated source")
    sealed = seal_parse_result(ParseResult())
    target = write_sealed_audit_input(sealed, fixture, tmp_path / "audit")
    original = target.read_bytes()

    with pytest.raises(FileExistsError):
        write_sealed_audit_input(sealed, fixture, tmp_path / "audit")

    assert target.read_bytes() == original


def test_private_audit_input_requires_sealed_input_and_real_source_path(tmp_path):
    with pytest.raises(ValueError, match="intact sealed"):
        write_sealed_audit_input(ParseResult(), tmp_path / "absent.pdf", tmp_path / "audit")
    with pytest.raises(FileNotFoundError):
        write_sealed_audit_input(
            seal_parse_result(ParseResult()), tmp_path / "absent.pdf", tmp_path / "audit"
        )
    assert not (tmp_path / "audit").exists()
