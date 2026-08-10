# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliberate freeze gate for the scanned-personal-detail v2 public contract."""

from __future__ import annotations

import hashlib
import json

from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
)

# This digest covers every public dataset and column descriptor plus the
# Community ordering, reading, role, and foreign-key policies consumed by the
# Markdown renderer.  A schema change must therefore be an explicit contract
# decision (and normally a version bump), never an incidental extraction edit.
_FROZEN_V2_PROJECTION_SHA256 = "001f03a7c3a699ab43b41dc42eb0be08acc0bb010405924ab6e117b2acdbef7b"


def _projection_contract_bytes() -> bytes:
    dictionary = personal_detail_data_dictionary()
    semantic = personal_detail_semantic_extensions()
    policy = semantic["community_projection_overrides"]
    contract = {
        "schema_id": dictionary["schema_id"],
        "version": dictionary["version"],
        "datasets": dictionary["datasets"],
        "dataset_document_order": semantic["dataset_document_order"],
        "dataset_reading_columns": semantic["dataset_reading_columns"],
        "dataset_representation_roles": policy["dataset_representation_roles"],
        "dataset_foreign_keys": policy["dataset_foreign_keys"],
        "personal_detail_contract": semantic["personal_detail_contract"],
    }
    return json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_scanned_personal_detail_v2_projection_contract_is_frozen() -> None:
    assert hashlib.sha256(_projection_contract_bytes()).hexdigest() == (
        _FROZEN_V2_PROJECTION_SHA256
    )
