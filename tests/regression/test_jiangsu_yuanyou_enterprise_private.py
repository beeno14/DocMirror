# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused regression coverage for the Jiangsu Yuanyou enterprise report."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.server.output_builder import build_community_bundle

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]

_FIXTURE = (
    Path("tests/fixtures-private/credit_report")
    / "Digital Enterprise"
    / "江苏缘油征信报告20260526.pdf"
)
_ACTIVE_LOAN_ACCOUNT = "B10611000H00011811138147245001"


def test_jiangsu_yuanyou_preserves_reported_zeroes_and_display_scope() -> None:
    if not _FIXTURE.exists():
        pytest.skip("Jiangsu Yuanyou enterprise regression fixture is unavailable")

    sealed = asyncio.run(
        perceive_document(
            _FIXTURE,
            PerceiveOptions(
                policy=normalize_parse_policy(
                    enhance_mode="standard",
                    doc_type_hint="credit_report:force",
                )
            ),
        )
    )
    bundle = build_community_bundle(sealed, file_path=str(_FIXTURE))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}

    account = next(
        row["normalized"]
        for row in datasets["enterprise_credit_accounts"]["rows"]
        if row["normalized"]["account_identifier"] == _ACTIVE_LOAN_ACCOUNT
    )
    assert account["current_overdue_amount"] == "0"
    assert account["overdue_principal"] == "0"
    assert account["current_overdue_periods"] == 0
    assert account["current_overdue"] is False
    assert account["current_overdue_status"] == "not_overdue"

    summary = semantic["domain"]["facts"]["credit_summary"]
    assert summary["source_display_limited"] is True
    assert "信息展示范围受限" in summary["account_dataset_scope_note"]
