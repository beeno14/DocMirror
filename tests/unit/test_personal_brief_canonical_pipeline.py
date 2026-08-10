# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.models.entities.parse_result import (
    CanonicalEvidencePlane,
    CellValue,
    PageContent,
    ParseResult,
    TableBlock,
    TableRow,
    TextBlock,
)
from docmirror.models.mirror.vnext import EvidenceAtom, EvidenceStore
from docmirror.plugins.credit_report.personal_brief_native.extraction import (
    _personal_header_datasets,
)
from docmirror.plugins.credit_report.personal_brief_native.ir import (
    build_canonical_personal_brief_document,
)
from docmirror.plugins.credit_report.personal_brief_native.pipeline import (
    _component_refs,
    _dataset_completeness,
    run_personal_brief_pipeline,
)


def _result(*texts: str) -> ParseResult:
    return ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    TextBlock(content=text, bbox=[20, 20 + index * 60, 580, 42 + index * 60])
                    for index, text in enumerate(texts)
                ],
            )
        ]
    )


@pytest.mark.parametrize(
    "parse_result",
    [
        ParseResult(),
        _result("Quarterly engineering status report"),
    ],
)
def test_pipeline_rejects_empty_or_non_pboc_input(parse_result: ParseResult) -> None:
    semantic = run_personal_brief_pipeline(parse_result).semantic_document

    assert semantic.extraction_report["status"] == "incomplete"
    assert semantic.extraction_report["business_record_count"] == 0
    assert semantic.extraction_report["failures"][0]["code"] == (
        "PERSONAL_BRIEF_DOCUMENT_NOT_RECOGNIZED"
    )
    assert not any(semantic.datasets.values())


def test_header_only_canonical_subset_is_recognized() -> None:
    result = _result(
        "个人信用报告 报告编号：2026071900012345678901 "
        "报告时间：2026-07-19 09:08:07 姓名：张三 "
        "证件类型：身份证 证件号码：11010519491231002X 已婚"
    )

    semantic = run_personal_brief_pipeline(result).semantic_document

    assert semantic.extraction_report["status"] == "complete"
    assert semantic.datasets["personal_report_metadata"][0]["marital_status"] == "married"


def test_unlabelled_marital_status_is_scoped_to_identity_metadata() -> None:
    blocks = [
        (1, "本报告不用于信贷申请及其他用途。"),
        (
            1,
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X 已婚 "
            "其他证件信息：护照CHN1234567",
        ),
        (1, "信贷记录"),
    ]

    _identities, metadata, _policy = _personal_header_datasets(ParseResult(), blocks)

    assert metadata[0]["marital_status"] == "married"
    assert metadata[0]["marital_status_raw"] == "已婚"


def test_account_completeness_uses_independent_canonical_boundaries() -> None:
    document = build_canonical_personal_brief_document(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "信贷记录",
            "贷款",
            "1.2024年1月1日甲银行发放的个人消费贷款1,000元。",
            "2.2024年2月2日乙银行发放的个人消费贷款2,000元。",
        )
    )
    first = next(
        component
        for component in document.components_for("loans")
        if "甲银行" in component.text
    )
    account = {
        "account_id": "account:1",
        "sequence": 1,
        "source_sequence": 1,
        "source_section": "loans",
        "account_type": "loan",
        "source_refs": _component_refs([first], "test_account"),
    }

    details = _dataset_completeness(
        document,
        {"credit_accounts": [account]},
        {},
    )["credit_accounts"]

    assert details["expected_row_count"] == 2
    assert details["emitted_row_count"] == 1
    assert details["omitted_row_count"] == 1
    assert details["missing_required_field_record_count"] == 1
    assert details["uncovered_boundary_ids"] == ["loans:2"]
    assert details["verified"] is False


def test_completeness_rejects_unresolvable_provenance() -> None:
    document = build_canonical_personal_brief_document(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "信贷记录",
            "贷款",
            "1.2024年1月1日甲银行发放的个人消费贷款1,000元。",
        )
    )
    account = {
        "account_id": "account:1",
        "sequence": 1,
        "source_sequence": 1,
        "source_section": "loans",
        "account_type": "loan",
        "source_refs": [
            {
                "source": "fake",
                "page": 999,
                "unit_id": "missing:unit",
                "evidence_ids": ["ev:ghost"],
            }
        ],
    }

    details = _dataset_completeness(
        document,
        {"credit_accounts": [account]},
        {},
    )["credit_accounts"]

    assert details["invalid_provenance_record_count"] == 1
    assert details["invalid_provenance_record_ids"] == ["account:1"]
    assert details["verified"] is False


def test_trailing_heading_transitions_the_following_component_only() -> None:
    document = build_canonical_personal_brief_document(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "信贷记录",
            "贷款",
            "1.2024年1月1日甲银行发放的个人消费贷款1,000元。\n相关还款责任信息",
            "1.",
            "2024年2月2日，为李四（证件类型：身份证，证件号码：110105194912310011）"
            "在乙银行办理的个人经营性贷款承担相关还款责任，责任人类型为保证人，"
            "相关还款责任金额2,000。截至2025年1月1日，个人经营性贷款余额1,000。",
        )
    )

    loan_component = next(component for component in document.components if "甲银行" in component.text)
    liability_component = next(component for component in document.components if "乙银行" in component.text)
    semantic = run_personal_brief_pipeline(document).semantic_document
    liability_completeness = semantic.extraction_report["dataset_completeness"][
        "repayment_liability_records"
    ]

    assert loan_component.section_key == "loans"
    assert liability_component.section_key == "repayment_liability"
    assert len(semantic.datasets["repayment_liability_records"]) == 1
    assert liability_completeness["expected_row_count"] == 1
    assert liability_completeness["verified"] is True


def test_liability_reconstruction_uses_evidence_lines_across_page_break() -> None:
    def atom(
        evidence_id: str,
        text: str,
        page: int,
        bbox: list[float],
    ) -> EvidenceAtom:
        return EvidenceAtom(
            id=evidence_id,
            page_id=f"page:{page:04d}",
            text=text,
            bbox=bbox,
        )

    record_1 = (
        "2024年1月1日，为甲（证件类型：身份证，证件号码：110105194912310011）"
        "在甲银行办理的贷款承担相关还款责任，责任人类型为保证人，"
        "相关还款责任金额1,000（保证合同编号：A1）。截至2025年1月，贷款余额500。"
    )
    record_2 = (
        "2024年2月2日，为乙（证件类型：身份证，证件号码：110105194912310022）"
        "在乙银行办理的贷款承担相关还款责任，责任人类型为保证人，"
        "相关还款责任金额2,000（保证合同编号：B2）。截至2025年1月，贷款余额1,000。"
    )
    record_3_head = (
        "2024年3月3日，为丙（证件类型：身份证，证件号码：110105194912310033）"
        "在丙银行办理的贷款承担相关还款责任，责任人类型为保证人，"
        "相关还款责任金额3,000（保证合同编号："
    )
    record_3_tail = "C3）。截至2025年1月，贷款余额1,500。"
    record_4 = (
        "2024年4月4日，为丁（证件类型：身份证，证件号码：110105194912310044）"
        "在丁银行办理的贷款承担相关还款责任，责任人类型为保证人，"
        "相关还款责任金额4,000（保证合同编号：D4）。截至2025年1月，贷款余额2,000。"
    )
    page_1_atoms = [
        atom("ev:p1:o1", "1.", 1, [20, 100, 30, 109]),
        atom("ev:p1:r1", record_1, 1, [40, 100, 580, 109]),
        atom("ev:p1:o2", "2.", 1, [20, 130, 30, 139]),
        atom("ev:p1:r2", record_2, 1, [40, 130, 580, 139]),
        atom("ev:p1:o3", "3.", 1, [20, 160, 30, 169]),
        atom("ev:p1:r3", record_3_head, 1, [40, 160, 580, 169]),
    ]
    page_2_atoms = [
        atom("ev:p2:tail", record_3_tail, 2, [40, 20, 580, 29]),
        atom("ev:p2:o4", "4.", 2, [20, 50, 30, 59]),
        atom("ev:p2:r4", record_4, 2, [40, 50, 580, 59]),
    ]
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    TextBlock(
                        content=(
                            "个人信用报告 报告编号：2026071900012345678901 "
                            "报告时间：2026-07-19 09:08:07 姓名：张三 "
                            "证件类型：身份证 证件号码：11010519491231002X"
                        ),
                        bbox=[20, 20, 580, 40],
                    ),
                    TextBlock(content="信贷记录", bbox=[20, 50, 100, 60]),
                    TextBlock(content="相关还款责任信息", bbox=[20, 70, 160, 80]),
                    TextBlock(
                        content="1.",
                        bbox=[20, 100, 30, 109],
                        evidence_ids=["ev:p1:o1"],
                    ),
                    TextBlock(
                        content="\n".join((record_1, record_2, record_3_head)),
                        bbox=[40, 100, 580, 169],
                        evidence_ids=["ev:p1:r1", "ev:p1:r2", "ev:p1:r3"],
                    ),
                    TextBlock(
                        content="2.",
                        bbox=[20, 130, 30, 139],
                        evidence_ids=["ev:p1:o2"],
                    ),
                    TextBlock(
                        content="3.",
                        bbox=[20, 160, 30, 169],
                        evidence_ids=["ev:p1:o3"],
                    ),
                ],
            ),
            PageContent(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                texts=[
                    TextBlock(
                        content=f"{record_3_tail}\n{record_4}",
                        bbox=[40, 20, 580, 59],
                        evidence_ids=["ev:p2:tail", "ev:p2:r4"],
                    ),
                    TextBlock(
                        content="4.",
                        bbox=[20, 50, 30, 59],
                        evidence_ids=["ev:p2:o4"],
                    ),
                    TextBlock(content="非信贷交易记录", bbox=[20, 90, 160, 100]),
                ],
            ),
        ],
        evidence_plane=CanonicalEvidencePlane(
            evidence=EvidenceStore(text_atoms=[*page_1_atoms, *page_2_atoms])
        ),
    )

    document = build_canonical_personal_brief_document(result)
    typed = [
        component
        for component in document.components_for("repayment_liability")
        if component.semantic_role == "repayment_liability_record"
    ]
    semantic = run_personal_brief_pipeline(document).semantic_document
    liabilities = semantic.datasets["repayment_liability_records"]

    assert document.content_conserved is True
    assert len(typed) == 4
    assert typed[2].source_pages == (1, 2)
    assert "2.\n3.\nC3" not in document.full_text
    assert [row["contract_number"] for row in liabilities] == ["A1", "B2", "C3", "D4"]
    assert {ref["page"] for ref in liabilities[2]["source_refs"]} == {1, 2}
    assert not {
        "ev:p1:o1",
        "ev:p1:o2",
        "ev:p1:o3",
        "ev:p2:o4",
    }.intersection(
        evidence_id
        for ref in liabilities[2]["source_refs"]
        for evidence_id in ref.get("evidence_ids", [])
    )
    decision = next(
        item
        for item in document.continuation_decisions
        if item.selected == "continue_numbered_record"
    )
    assert (decision.from_page, decision.to_page) == (1, 2)
    assert decision.best_score >= 0.70


def _liability_narrative(
    *,
    date: str,
    name: str,
    document_number: str,
    institution: str,
    amount: str,
    contract_number: str,
    snapshot_tail: str = "",
) -> str:
    return (
        f"{date}，为{name}（证件类型：身份证，证件号码：{document_number}）"
        f"在{institution}办理的贷款承担相关还款责任，责任人类型为保证人，"
        f"相关还款责任金额{amount}（保证合同编号：{contract_number}）。"
        f"{snapshot_tail}"
    )


def _two_page_liability_result(page_1_record: str, page_2_text: str) -> ParseResult:
    return ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    TextBlock(
                        content=(
                            "个人信用报告 报告编号：2026071900012345678901 "
                            "报告时间：2026-07-19 09:08:07 姓名：张三 "
                            "证件类型：身份证 证件号码：11010519491231002X"
                        ),
                        bbox=[20, 20, 580, 40],
                    ),
                    TextBlock(content="信贷记录", bbox=[20, 50, 100, 60]),
                    TextBlock(content="相关还款责任信息", bbox=[20, 70, 160, 80]),
                    TextBlock(content="1.", bbox=[20, 760, 30, 769]),
                    TextBlock(content=page_1_record, bbox=[40, 740, 580, 779]),
                ],
            ),
            PageContent(
                page_number=2,
                source_page_number=2,
                width=600,
                height=800,
                texts=[TextBlock(content=page_2_text, bbox=[40, 20, 580, 100])],
            ),
        ]
    )


def test_liability_reconstruction_groups_token_atoms_on_one_visual_baseline() -> None:
    record = _liability_narrative(
        date="2024年1月1日",
        name="甲",
        document_number="110105194912310011",
        institution="甲银行",
        amount="1,000",
        contract_number="A1",
        snapshot_tail="截至2025年1月，贷款余额500。",
    )
    text = f"1.{record}"
    fragments = ("1.2024", "年1", "月1", f"日，{record.split('日，', 1)[1]}")
    evidence_ids = [f"ev:token:{index}" for index in range(len(fragments))]
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    TextBlock(
                        content=(
                            "个人信用报告 报告编号：2026071900012345678901 "
                            "报告时间：2026-07-19 09:08:07 姓名：张三 "
                            "证件类型：身份证 证件号码：11010519491231002X"
                        ),
                        bbox=[20, 20, 580, 40],
                    ),
                    TextBlock(content="信贷记录", bbox=[20, 50, 100, 60]),
                    TextBlock(content="相关还款责任信息", bbox=[20, 70, 160, 80]),
                    TextBlock(
                        content=text,
                        bbox=[20, 100, 580, 112],
                        evidence_ids=evidence_ids,
                    ),
                ],
            )
        ],
        evidence_plane=CanonicalEvidencePlane(
            evidence=EvidenceStore(
                text_atoms=[
                    EvidenceAtom(
                        id=evidence_id,
                        page_id="page:0001",
                        text=fragment,
                        bbox=[20 + index * 100, 100, 115 + index * 100, 112],
                    )
                    for index, (evidence_id, fragment) in enumerate(
                        zip(evidence_ids, fragments, strict=True)
                    )
                ]
            )
        ),
    )
    document = build_canonical_personal_brief_document(result)
    semantic = run_personal_brief_pipeline(document).semantic_document

    assert document.content_conserved is True
    assert len(semantic.datasets["repayment_liability_records"]) == 1
    assert semantic.datasets["repayment_liability_records"][0]["contract_number"] == "A1"
    assert set(evidence_ids).issubset(
        {
            evidence_id
            for ref in semantic.datasets["repayment_liability_records"][0]["source_refs"]
            for evidence_id in ref["evidence_ids"]
        }
    )


def test_liability_reconstruction_joins_normal_snapshot_tail_before_next_record() -> None:
    first = _liability_narrative(
        date="2024年1月1日",
        name="甲",
        document_number="110105194912310011",
        institution="甲银行",
        amount="1,000",
        contract_number="A1",
    )
    second = _liability_narrative(
        date="2024年2月2日",
        name="乙",
        document_number="110105194912310022",
        institution="乙银行",
        amount="2,000",
        contract_number="B2",
        snapshot_tail="截至2025年1月，贷款余额1,000。",
    )
    document = build_canonical_personal_brief_document(
        _two_page_liability_result(
            first,
            f"截至2025年1月，贷款余额500。\n2. {second}",
        )
    )
    semantic = run_personal_brief_pipeline(document).semantic_document
    records = semantic.datasets["repayment_liability_records"]

    assert document.content_conserved is True
    assert records[0]["snapshot_date"] == "2025-01"
    assert records[0]["balance"] == 500.0
    assert {ref["page"] for ref in records[0]["source_refs"]} == {1, 2}
    assert records[0]["extraction_status"] == "reported"
    assert document.full_text.count("截至2025年1月，贷款余额500。") == 1


def test_liability_reconstruction_preserves_trailing_next_section_heading() -> None:
    record = _liability_narrative(
        date="2024年1月1日",
        name="甲",
        document_number="110105194912310011",
        institution="甲银行",
        amount="1,000",
        contract_number="A1",
        snapshot_tail="截至2025年1月，贷款余额500。",
    )
    result = _result(
        "个人信用报告 报告编号：2026071900012345678901 "
        "报告时间：2026-07-19 09:08:07 姓名：张三 "
        "证件类型：身份证 证件号码：11010519491231002X",
        "信贷记录",
        "相关还款责任信息",
        f"1.\n{record}\n非信贷交易记录\n本报告中没有非信贷交易记录。",
    )
    document = build_canonical_personal_brief_document(result)
    liability = next(
        component
        for component in document.components
        if component.semantic_role == "repayment_liability_record"
    )
    non_credit = document.components_for("non_credit_transactions")

    assert document.content_conserved is True
    assert "非信贷交易记录" not in liability.text
    assert document.section_presence["non_credit_transactions"] == "present"
    assert any(component.kind == "heading" for component in non_credit)
    assert sum(
        line.strip() == "非信贷交易记录" for line in document.full_text.splitlines()
    ) == 1
    assert document.full_text.count("本报告中没有非信贷交易记录。") == 1


def test_liability_reconstruction_preserves_ambiguous_page_prefix_as_unresolved() -> None:
    first = _liability_narrative(
        date="2024年1月1日",
        name="甲",
        document_number="110105194912310011",
        institution="甲银行",
        amount="1,000",
        contract_number="A1",
    )
    second = _liability_narrative(
        date="2024年2月2日",
        name="乙",
        document_number="110105194912310022",
        institution="乙银行",
        amount="2,000",
        contract_number="B2",
        snapshot_tail="截至2025年1月，贷款余额1,000。",
    )
    document = build_canonical_personal_brief_document(
        _two_page_liability_result(first, f"无法归属的页首文本\n2. {second}")
    )
    semantic = run_personal_brief_pipeline(document).semantic_document
    records = semantic.datasets["repayment_liability_records"]
    first_component = next(
        component
        for component in document.components
        if component.semantic_role == "repayment_liability_record"
        and component.rows[0].values[0] == "1"
    )

    assert document.content_conserved is True
    assert "无法归属" not in first_component.text
    assert document.full_text.count("无法归属的页首文本") == 1
    assert records[0]["extraction_status"] == "unresolved"
    assert semantic.extraction_report["dataset_completeness"][
        "repayment_liability_records"
    ]["verified"] is False


def test_ambiguous_inline_prefix_keeps_canonical_source_order() -> None:
    first = _liability_narrative(
        date="2024年1月1日",
        name="甲",
        document_number="110105194912310011",
        institution="甲银行",
        amount="1,000",
        contract_number="A1",
    )
    second = _liability_narrative(
        date="2024年2月2日",
        name="乙",
        document_number="110105194912310022",
        institution="乙银行",
        amount="2,000",
        contract_number="B2",
        snapshot_tail="截至2025年1月，贷款余额1,000。",
    )
    document = build_canonical_personal_brief_document(
        _two_page_liability_result(first, f"无法归属的页首文本 2. {second}")
    )
    ordered = [
        component
        for component in document.components_for("repayment_liability")
        if component.semantic_role
        in {"repayment_liability_record", "unresolved_liability_fragment"}
    ]

    assert [component.semantic_role for component in ordered] == [
        "repayment_liability_record",
        "unresolved_liability_fragment",
        "repayment_liability_record",
    ]
    assert document.content_conserved is True


def test_structural_note_ordinal_remains_available_to_note_extraction() -> None:
    semantic = run_personal_brief_pipeline(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "说明",
            "1.",
            "本报告仅用于约定用途。",
        )
    ).semantic_document

    assert len(semantic.datasets["report_notes"]) == 1
    assert semantic.datasets["report_notes"][0]["content"] == "本报告仅用于约定用途。"


def test_blank_canonical_summary_cells_are_explicit_not_reported_fields() -> None:
    def rows(values: list[list[str]]) -> list[TableRow]:
        return [
            TableRow(cells=[CellValue(text=value) for value in row])
            for row in values
        ]

    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    TextBlock(
                        content=(
                            "个人信用报告 报告编号：2026071900012345678901 "
                            "报告时间：2026-07-19 09:08:07 姓名：张三 "
                            "证件类型：身份证 证件号码：11010519491231002X"
                        ),
                        bbox=[20, 20, 580, 50],
                    ),
                    TextBlock(content="信贷记录\n信息概要", bbox=[20, 60, 580, 90]),
                ],
                tables=[
                    TableBlock(
                        table_id="summary-grid",
                        headers=["", "信用卡", "购房贷款", "其他贷款", "其他业务"],
                        rows=rows(
                            [
                                ["账户数", "1", "2", "3", ""],
                                ["未结清/未销户账户数", "", "1", "2", ""],
                                ["发生过逾期的账户数", "1", "", "", ""],
                                ["发生过90天以上逾期的账户数", "", "", "", ""],
                            ]
                        ),
                        page=1,
                        bbox=[20, 100, 580, 260],
                    ),
                    TableBlock(
                        table_id="liability-summary",
                        headers=["", "为个人", "为企业"],
                        rows=rows([["相关还款责任账户数", "1", ""]]),
                        page=1,
                        bbox=[20, 270, 580, 330],
                    ),
                ],
            )
        ]
    )

    semantic = run_personal_brief_pipeline(result).semantic_document
    records = semantic.datasets["personal_credit_summary_records"]
    completeness = semantic.extraction_report["dataset_completeness"][
        "personal_credit_summary_records"
    ]
    enterprise_liability = next(
        record
        for record in records
        if record["metric"] == "enterprise_repayment_liability_count"
    )

    assert len(records) == 18
    assert enterprise_liability["value"] is None
    assert enterprise_liability["reporting_status"] == "not_reported"
    assert completeness["expected_row_count"] == 18
    assert completeness["verified"] is True


def test_optional_other_business_and_institution_statement_form_a_valid_subset() -> None:
    semantic = run_personal_brief_pipeline(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "信贷记录",
            "其他业务",
            "1.2024年1月1日上海融资租赁有限公司发放的100,000元融资租赁，"
            "2025年1月到期。截至2024年12月，余额80,000，当前无逾期，"
            "从未发生过逾期。",
            "机构说明",
            "1.说明内容：本人声明该账户已有异议。添加日期：2024年1月2日",
        )
    ).semantic_document

    account = semantic.datasets["credit_accounts"][0]
    statement = semantic.datasets["institution_statement_records"][0]
    summary_completeness = semantic.extraction_report["dataset_completeness"][
        "personal_credit_summary_records"
    ]

    assert semantic.extraction_report["status"] == "complete"
    assert semantic.datasets["personal_credit_summary_records"] == []
    assert summary_completeness["status"] == "absent_from_report"
    assert account["account_type"] == "other_business"
    assert account["business_category"] == "other_business"
    assert account["business_type"] == "融资租赁"
    assert account["loan_amount"] == 100000
    assert statement["statement_content"] == "本人声明该账户已有异议"
    assert statement["added_date"] == "2024-01-02"


def test_inquiry_completeness_uses_printed_ordinal_maximum() -> None:
    semantic = run_personal_brief_pipeline(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "查询记录",
            "机构查询记录明细",
            "1 2024年1月1日 示例商业银行 贷款审批",
            "3 2024年1月3日 示例商业银行 贷后管理",
            "个人查询记录明细",
        )
    ).semantic_document
    completeness = semantic.extraction_report["dataset_completeness"]["inquiry_records"]

    assert [row["sequence"] for row in semantic.datasets["inquiry_records"]] == [1, 3]
    assert completeness["expected_row_count"] == 3
    assert completeness["emitted_row_count"] == 2
    assert completeness["omitted_row_count"] == 1
    assert completeness["verified"] is False


def test_inquiry_with_lost_ordinal_is_retained_and_forces_review() -> None:
    semantic = run_personal_brief_pipeline(
        _result(
            "个人信用报告 报告编号：2026071900012345678901 "
            "报告时间：2026-07-19 09:08:07 姓名：张三 "
            "证件类型：身份证 证件号码：11010519491231002X",
            "查询记录",
            "机构查询记录明细",
            "1 2024年1月1日 示例商业银行 贷款审批",
            "2024年1月2日 另一商业银行 贷后管理",
            "个人查询记录明细",
        )
    ).semantic_document
    records = semantic.datasets["inquiry_records"]
    completeness = semantic.extraction_report["dataset_completeness"]["inquiry_records"]

    assert [row["sequence"] for row in records] == [1, 2]
    assert records[1]["institution"] == "另一商业银行"
    assert records[1]["reason"] == "贷后管理"
    assert records[1]["extraction_status"] == "unresolved"
    assert completeness["expected_row_count"] == 2
    assert completeness["emitted_row_count"] == 2
    assert completeness["unresolved_row_count"] == 1
    assert completeness["verified"] is False


def test_table_only_institution_statement_is_counted_independently() -> None:
    result = ParseResult(
        pages=[
            PageContent(
                page_number=1,
                source_page_number=1,
                width=600,
                height=800,
                texts=[
                    TextBlock(
                        content=(
                            "个人信用报告 报告编号：2026071900012345678901 "
                            "报告时间：2026-07-19 09:08:07 姓名：张三 "
                            "证件类型：身份证 证件号码：11010519491231002X"
                        ),
                        bbox=[20, 20, 580, 50],
                    ),
                    TextBlock(content="机构说明", bbox=[20, 100, 580, 130]),
                ],
                tables=[
                    TableBlock(
                        table_id="institution-statement",
                        headers=["说明内容", "添加日期"],
                        rows=[
                            TableRow(
                                cells=[
                                    CellValue(text="本人声明该账户已有异议"),
                                    CellValue(text="2024年1月2日"),
                                ]
                            )
                        ],
                        page=1,
                        bbox=[20, 150, 580, 220],
                    )
                ],
            )
        ]
    )

    semantic = run_personal_brief_pipeline(result).semantic_document
    completeness = semantic.extraction_report["dataset_completeness"][
        "institution_statement_records"
    ]

    assert len(semantic.datasets["institution_statement_records"]) == 1
    assert completeness["expected_row_count"] == 1
    assert completeness["emitted_row_count"] == 1
    assert completeness["verified"] is True


def test_cross_page_table_entity_becomes_one_logical_ir_component() -> None:
    def table(table_id: str, rows: list[list[str]], bbox: list[float]) -> SimpleNamespace:
        return SimpleNamespace(
            table_id=table_id,
            metadata={"raw_rows": rows},
            headers=[],
            rows=[],
            bbox=bbox,
            evidence_ids=[],
        )

    def page(
        number: int,
        *,
        texts: list[SimpleNamespace] | None = None,
        tables: list[SimpleNamespace] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=number,
            source_page_number=number,
            width=600,
            height=800,
            texts=texts or [],
            tables=tables or [],
        )

    result = SimpleNamespace(
        pages=[
            page(
                1,
                texts=[SimpleNamespace(content="个人信用报告", bbox=[20, 20, 580, 50])],
                tables=[
                    table("statement-header", [["说明内容", "添加日期"]], [20, 650, 580, 770])
                ],
            ),
            page(
                2,
                tables=[
                    table("statement-body-1", [["第一项", "2024年1月1日"]], [20, 20, 580, 770])
                ],
            ),
            page(
                3,
                texts=[SimpleNamespace(content="说明", bbox=[20, 400, 100, 430])],
                tables=[
                    table("statement-body-2", [["第二项", "2024年1月2日"]], [20, 20, 580, 390])
                ],
            ),
        ]
    )

    document = build_canonical_personal_brief_document(result)
    logical = next(
        component
        for component in document.components
        if component.kind == "logical_table" and len(component.source_unit_ids) == 3
    )

    assert logical.source_unit_ids == (
        "table:statement-header",
        "table:statement-body-1",
        "table:statement-body-2",
    )
    assert [row.values for row in logical.rows] == [
        ("说明内容", "添加日期"),
        ("第一项", "2024年1月1日"),
        ("第二项", "2024年1月2日"),
    ]
    assert document.content_conserved is True
