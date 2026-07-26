# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private credit-report subtype coverage for canonical facts and Bundle v3."""

from __future__ import annotations

import asyncio
import csv
import json
import re
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
    "人行征信报告-2025-04-14.pdf": (18, 0, 82, 10, 1),
    "人行征信报告-2026-06-24 08-52-53(1).pdf": (39, 0, 100, 13, 0),
    "征信报告_平安银行_20090811_1.pdf": (29, 4, 97, 1, 2),
    "汪婧妍征信.pdf": (39, 6, 170, 13, 0),
    "沈俊艺个人征信.pdf": (83, 3, 90, 6, 1),
    "赵思雯个人征信.pdf": (45, 4, 124, 9, 1),
    "陈是兴_征信报告_中国建设银行_20101012.pdf": (87, 5, 108, 4, 0),
    "陈是兴_征信报告_中国建设银行_20101012_1.pdf": (87, 5, 108, 4, 0),
}
_DIGITAL_PERSONAL_BRIEF_MARITAL_STATUS = {
    "人行征信报告-2025-04-14.pdf": ("divorced", "离婚"),
    "人行征信报告-2026-06-24 08-52-53(1).pdf": ("married", "已婚"),
    "征信报告_平安银行_20090811_1.pdf": ("married", "已婚"),
    "汪婧妍征信.pdf": ("married", "已婚"),
    "沈俊艺个人征信.pdf": ("divorced", "离婚"),
    "赵思雯个人征信.pdf": ("married", "已婚"),
    "陈是兴_征信报告_中国建设银行_20101012.pdf": ("married", "已婚"),
    "陈是兴_征信报告_中国建设银行_20101012_1.pdf": ("married", "已婚"),
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
        inquiry_dataset = next(dataset for dataset in payload["datasets"] if dataset["name"] == "inquiry_records")
        accounts = datasets["credit_accounts"]
        inquiries = datasets["inquiry_records"]
        liabilities = datasets.get("repayment_liability_records", [])
        assert len(accounts) == expected_accounts
        assert len(liabilities) == expected_liabilities
        assert len(inquiries) == expected_inquiries
        assert semantic["document"]["title"] == "个人信用报告"
        assert inquiry_dataset["completeness"]["verified"] is True
        assert sum(row["normalized"]["inquiry_type"] == "personal" for row in inquiries) == expected_personal
        personal_inquiries = [
            row["normalized"] for row in inquiries if row["normalized"]["inquiry_type"] == "personal"
        ]
        assert all(row["reason"] == "本人查询" for row in personal_inquiries)
        assert all(row["source_reason"].startswith("本人查询") for row in personal_inquiries)
        assert sum(row["normalized"]["status"] == "inactive" for row in accounts) == expected_inactive
        enhanced_preview = bundle.render_enhanced_markdown(semantic)
        information_summary_preview = enhanced_preview.split("## 信息概要", maxsplit=1)[1].split(
            "\n## 信贷记录",
            maxsplit=1,
        )[0]
        assert "### 个人信息" in information_summary_preview
        assert "### 信用概览" in information_summary_preview
        assert "### 报告信息" in information_summary_preview
        assert "## 附录：文档来源与提取信息" in enhanced_preview
        assert semantic["domain"]["facts"]["id_number"] in information_summary_preview
        assert semantic["domain"]["facts"]["report_number"] in information_summary_preview
        expected_marital_status, expected_marital_label = _DIGITAL_PERSONAL_BRIEF_MARITAL_STATUS[fixture.name]
        assert semantic["domain"]["facts"]["marital_status"] == expected_marital_status
        assert semantic["domain"]["entity_fields"]["marital_status"] == expected_marital_status
        assert f"**婚姻状况:** {expected_marital_label}" in information_summary_preview
        assert not any(
            re.search(r"[A-Za-z_]", label)
            for label in re.findall(r"\*\*([^*]+):\*\*", enhanced_preview)
        )
        if personal_inquiries:
            assert "#### 个人查询" in enhanced_preview
            assert "#### 本人查询" not in enhanced_preview
            assert "| 本人 | 本人查询 | 个人查询 |" in enhanced_preview
        if fixture.name == "人行征信报告-2026-06-24 08-52-53(1).pdf":
            overdue_rows = datasets["overdue_records"]
            normalized_overdue = [row["normalized"] for row in overdue_rows]
            assert len(normalized_overdue) == 9
            assert [row["over_90_days_months"] for row in normalized_overdue] == [1, 3, 2, 2, 2, 1, 2, 1, 0]
            assert all(row["current_overdue_status"] == "overdue" for row in normalized_overdue)
            assert sum(row["over_90_days"] is True for row in normalized_overdue) == 8
            overdue_markdown = enhanced_preview.split("### 逾期记录", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
            assert (
                "| 组内序号 | 账户类型 | 管理机构 | 业务类型 | 卡片尾号 | 开立日期 | "
                "最近5年逾期月数 | 其中超过90天月数 | 当前逾期状态 |"
            ) in overdue_markdown
            assert "account id" not in overdue_markdown
            assert "overdue id" not in overdue_markdown
            assert "last_5_years" not in overdue_markdown
            assert "当前有逾期" in overdue_markdown
        if (expected_accounts, expected_liabilities, expected_inquiries) == (45, 4, 124):
            semantic_datasets = {dataset["name"]: dataset for dataset in semantic["datasets"]}
            summary = semantic["domain"]["facts"]["credit_summary"]
            audit = semantic["domain"]["facts"]["credit_extraction_audit"]
            sections = semantic["structure"]["sections"]
            owned_blocks = [block_ref for section in sections for block_ref in section["block_refs"]]
            inquiry_bindings = [
                binding
                for binding in semantic["bindings"]
                if binding["dataset_id"] == semantic_datasets["inquiry_records"]["id"]
            ]
            logical_inquiry_table = next(
                table
                for table in semantic["structure"]["source_tables"]
                if table["id"] == "logical:ds_inquiry_records"
            )

            assert "credit_lines" not in semantic_datasets
            assert semantic_datasets["report_notes"]["row_count"] == 5
            assert semantic["domain"]["data_dictionary"]["fields"]
            assert semantic["domain"]["extensions"]["rendering_contract"]["do_not_union_representations"] is True
            assert summary["source_unclosed_account_count"] == 29
            assert summary["source_account_count"] == 45
            assert summary["source_overdue_account_count"] is None
            assert summary["source_overdue_account_count_status"] == "not_reported"
            assert audit["source_page_complete"] is True
            assert audit["evidence_complete"] is True
            assert len(sections) == 6
            assert len(owned_blocks) == len(set(owned_blocks))
            assert len(semantic["structure"]["outline"]) == 6
            assert len(semantic["structure"]["cross_page_flows"]) == 1
            assert len(logical_inquiry_table["rows"]) == 124
            assert len(logical_inquiry_table["segments"]) == 5
            assert len(inquiry_bindings) == 124
            assert all(binding["source_block_refs"] for binding in inquiry_bindings)
            assert all(binding["source_table_refs"] == ["logical:ds_inquiry_records"] for binding in inquiry_bindings)
            assert all(binding["evidence_refs"] for binding in inquiry_bindings)
            assert semantic["diagnostics"]["evidence_ids"]
        if fixture.name == "赵思雯个人征信.pdf":
            normalized_accounts = [row["normalized"] for row in accounts]
            credit_cards = [row for row in normalized_accounts if row["account_type"] == "credit_card"]
            loans = [row for row in normalized_accounts if row["account_type"] != "credit_card"]
            assert [row["sequence"] for row in credit_cards] == list(range(1, 22))
            assert [row["sequence"] for row in loans] == list(range(1, 25))
            assert next(row for row in credit_cards if row["sequence"] == 13)["currency"] == "HKD"
            assert next(row for row in credit_cards if row["sequence"] == 14)["currency"] == "CHF"
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
            assert persisted["document"]["title"] == "个人信用报告"
            persisted_inquiries = next(
                dataset for dataset in persisted["datasets"] if dataset["name"] == "inquiry_records"
            )
            persisted_accounts = next(
                dataset for dataset in persisted["datasets"] if dataset["name"] == "credit_accounts"
            )
            with (written["community"].parent / persisted_accounts["csv"]).open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                account_csv_rows = list(csv.DictReader(stream))
            with (written["community"].parent / persisted_inquiries["csv"]).open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                csv_rows = list(csv.DictReader(stream))
            with (written["community"].parent / "001_datasets" / "_audit_cells.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                audit_rows = list(csv.DictReader(stream))
            reading_table = next(
                table
                for table in persisted["reading"]["tables"]
                if table["dataset_id"] == persisted_inquiries["id"]
            )
            enhanced = written["enhanced_reading"].read_text(encoding="utf-8")
            assert next(line for line in enhanced.splitlines() if line.startswith("# ")) == "# 个人信用报告"
            assert next(
                row
                for row in account_csv_rows
                if row["account_type"] == "credit_card" and row["sequence"] == "13"
            )["currency"] == "HKD"
            assert next(
                row
                for row in account_csv_rows
                if row["account_type"] == "credit_card" and row["sequence"] == "14"
            )["currency"] == "CHF"
            table_lines = enhanced.split(f"### {reading_table['title']}", maxsplit=1)[1].split(
                "\n## ",
                maxsplit=1,
            )[0]
            table_count = sum(line.startswith("| ---") for line in table_lines.splitlines())
            markdown_rows = sum(line.startswith("| ") for line in table_lines.splitlines()) - (2 * table_count)
            assert validate_community_artifacts(written["community"]) == []
            assert (
                len(persisted_inquiries["rows"])
                == len(csv_rows)
                == reading_table["row_count"]
                == markdown_rows
                == 124
            )
            assert "#### 机构查询" in table_lines
            assert "#### 个人查询" in table_lines
            assert [row["record_id"] for row in persisted_inquiries["rows"]] == [
                row["record_id"] for row in csv_rows
            ]
            assert all(row["bbox"] for row in audit_rows)
            assert all(row["confidence"] for row in audit_rows)
            assert all(row["evidence_ref"] for row in audit_rows)
            assert semantic["domain"]["facts"]["id_number"] in enhanced
            assert semantic["domain"]["facts"]["report_number"] in enhanced
            information_summary = enhanced.split("## 信息概要", maxsplit=1)[1].split("\n## 信贷记录", maxsplit=1)[0]
            appendix = enhanced.split("## 附录：文档来源与提取信息", maxsplit=1)[1]
            assert information_summary.index("### 个人信息") < information_summary.index("### 信用概览")
            assert information_summary.index("### 信用概览") < information_summary.index("### 报告信息")
            assert "**内容模式:**" not in information_summary
            assert "**数据来源:**" not in information_summary
            assert "**旧版派生有效状态口径:**" not in information_summary
            assert "**源概要表标识:**" not in information_summary
            assert "**内容模式:**" in appendix
            assert "**数据来源:**" in appendix
            assert "**源概要表标识:**" in appendix
            assert "**源概要表页码:**" in appendix
            bold_labels = re.findall(r"\*\*([^*]+):\*\*", enhanced)
            assert bold_labels
            assert not any(re.search(r"[A-Za-z_]", label) for label in bold_labels)
