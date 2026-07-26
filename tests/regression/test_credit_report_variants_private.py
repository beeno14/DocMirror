# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private credit-report subtype coverage for canonical facts and Bundle v3."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import normalize_parse_policy
from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.server.edition_outputs import write_outputs
from docmirror.server.output_builder import build_community_bundle
from scripts.validate.validate_community_artifacts import validate_community_artifacts

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.tier_slow]
_FIXTURE_DIR = Path("tests/fixtures-private/credit_report")
_DIGITAL_PERSONAL_BRIEF_DIR = _FIXTURE_DIR / "Digital Personal Brief"

_DIGITAL_PERSONAL_BRIEF_EXPECTED = {
    "人行征信报告-2025-04-14.pdf": (18, 0, 79, 10, 1),
    "人行征信报告-2026-06-24 08-52-53(1).pdf": (39, 0, 95, 13, 0),
    "征信报告_平安银行_20090811_1.pdf": (29, 4, 97, 1, 2),
    "汪婧妍征信.pdf": (39, 6, 161, 13, 0),
    "沈俊艺个人征信.pdf": (83, 3, 86, 6, 1),
    "赵思雯个人征信.pdf": (45, 4, 124, 9, 1),
    "陈是兴_征信报告_中国建设银行_20101012.pdf": (87, 5, 108, 4, 0),
    "陈是兴_征信报告_中国建设银行_20101012_1.pdf": (87, 5, 108, 4, 0),
}


def _cases(pattern: str, subtype: str, public_type: str) -> list[pytest.ParameterSet]:
    fixtures = sorted(_FIXTURE_DIR.glob(pattern))
    if not fixtures:
        return [pytest.param(Path("__missing__"), subtype, public_type, marks=pytest.mark.skip)]
    return [pytest.param(path, subtype, public_type, id=f"{subtype}-{index}") for index, path in enumerate(fixtures, 1)]


CASES = [
    *_cases("*_个人简版征信报告.pdf", "personal_brief", "personal_credit_report_brief"),
    *[
        pytest.param(path, "personal_brief", "personal_credit_report_brief", id=f"digital-personal-brief-{index}")
        for index, path in enumerate(sorted(_DIGITAL_PERSONAL_BRIEF_DIR.glob("*.pdf")), 1)
    ],
    *_cases("*_个人详版征信报告.pdf", "personal_detail", "personal_credit_report_detailed"),
    *_cases("*_企业征信*.pdf", "enterprise", "enterprise_credit_report"),
]


@pytest.mark.parametrize("fixture,subtype,public_type", CASES)
def test_credit_report_subtype_projects_complete_v3(
    fixture: Path,
    subtype: str,
    public_type: str,
    tmp_path: Path,
) -> None:
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
    result = sealed.to_read_view()
    bundle = build_community_bundle(sealed, file_path=str(fixture))
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    assert "report_subtype" not in result.entities.domain_specific
    assert payload is not None
    assert payload["document"]["type"] == public_type
    assert semantic["classification"]["document_type"] == public_type
    assert semantic["schema"]["document_type"] == public_type
    assert semantic["source"]["fingerprint"] == sealed.integrity_fingerprint
    assert semantic["domain"]["facts"]["report_subtype"] == subtype
    assert semantic["structure"]["blocks"]
    assert len(semantic["bindings"]) == sum(dataset["row_count"] for dataset in semantic["datasets"])
    assert validate_projection_payload("community_semantic", semantic).valid
    assert validate_projection_payload("community", payload).valid
    assert payload["sections"]
    assert any(dataset["row_count"] > 0 for dataset in payload["datasets"])
    if fixture.name in _DIGITAL_PERSONAL_BRIEF_EXPECTED:
        expected_accounts, expected_liabilities, expected_inquiries, expected_personal, expected_inactive = (
            _DIGITAL_PERSONAL_BRIEF_EXPECTED[fixture.name]
        )
        datasets = {dataset["name"]: dataset["rows"] for dataset in payload["datasets"]}
        accounts = datasets["credit_accounts"]
        inquiries = datasets["inquiry_records"]
        liabilities = datasets.get("repayment_liability_records", [])
        assert len(accounts) == expected_accounts
        assert len(liabilities) == expected_liabilities
        assert len(inquiries) == expected_inquiries
        assert sum(row["normalized"]["inquiry_type"] == "personal" for row in inquiries) == expected_personal
        assert sum(row["normalized"]["status"] == "inactive" for row in accounts) == expected_inactive
        if fixture.name == "赵思雯个人征信.pdf":
            assert any(block["text"].strip() == "说明" for block in semantic["structure"]["blocks"])
            assert any(
                row and row[0] == "账户数"
                for table in semantic["structure"]["source_tables"]
                for row in table["rows"]
            )
            institution_sequences = {
                int(row["normalized"]["sequence"])
                for row in inquiries
                if row["normalized"]["inquiry_type"] == "institution"
            }
            assert institution_sequences == set(range(1, 116))
            _task_id, written = write_outputs(
                sealed,
                tmp_path,
                file_path=str(fixture),
                file_id="001",
                task_id="private-reading-parity",
                include_mirror=False,
                include_manifest=False,
            )
            persisted = json.loads(written["community"].read_text(encoding="utf-8"))
            persisted_inquiries = next(
                dataset for dataset in persisted["datasets"] if dataset["name"] == "inquiry_records"
            )
            with (written["community"].parent / persisted_inquiries["csv"]).open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                csv_rows = list(csv.DictReader(stream))
            reading_table = next(
                table
                for table in persisted["reading"]["tables"]
                if table["dataset_id"] == persisted_inquiries["id"]
            )
            enhanced = written["enhanced_reading"].read_text(encoding="utf-8")
            table_lines = enhanced.split(f"### {reading_table['title']}", maxsplit=1)[1].split(
                "### ",
                maxsplit=1,
            )[0]
            markdown_rows = sum(line.startswith("| ") for line in table_lines.splitlines()) - 2
            assert validate_community_artifacts(written["community"]) == []
            assert (
                len(persisted_inquiries["rows"])
                == len(csv_rows)
                == reading_table["row_count"]
                == markdown_rows
                == 124
            )
            assert [row["record_id"] for row in persisted_inquiries["rows"]] == [
                row["record_id"] for row in csv_rows
            ]
