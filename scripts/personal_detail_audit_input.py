# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in local evidence capture for private personal-detail regression runs.

This writes an audit input, not an extraction cache: live Primary tests always
perceive the actual PDF. The snapshot preserves raw tokens and geometry that
public Community/Semantic outputs intentionally omit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from docmirror.models.sealed import SealedParseResult


def write_sealed_audit_input(
    sealed: SealedParseResult, fixture: Path, destination: Path
) -> Path:
    """Persist a source-linked, integrity-checked input without overwriting one."""

    if not isinstance(sealed, SealedParseResult) or not sealed.verify_integrity():
        raise ValueError("private audit input requires an intact sealed ParseResult")
    fixture = fixture.resolve(strict=True)
    if not fixture.is_file():
        raise ValueError("private audit input source must be a file")
    parse_result = sealed.model_dump(mode="json", exclude_none=False)
    canonical_bytes = json.dumps(
        parse_result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical_bytes).hexdigest() != sealed.integrity_fingerprint:
        raise ValueError("private audit input serialization changed the sealed snapshot")
    source_hasher = hashlib.sha256()
    with fixture.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_hasher.update(chunk)
    source_digest = source_hasher.hexdigest()
    payload = {
        "schema": "docmirror.personal_detail.private_audit_input.v1",
        "source_pdf_name": fixture.name,
        "source_pdf_sha256": source_digest,
        "sealed_schema_version": sealed.schema_version,
        "sealed_integrity_fingerprint": sealed.integrity_fingerprint,
        "parse_result": parse_result,
    }
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{fixture.stem}.sealed-input.json"
    # A fresh audit directory is mandatory for each live run. Accidentally
    # reusing one must not destroy the evidence of the earlier run.
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    return target
