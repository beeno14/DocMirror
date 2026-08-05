from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    _printed_reading_order,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
    dataset_states_from_issues,
    liability_record_is_substantive,
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.field_contracts import validate_pboc_field
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _dedupe_liability_records,
    _extract_liabilities,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.personal_detail_scanned.variant import (
    PersonalDetailScannedVariant,
)


def _table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        table_id=table_id,
        metadata={"raw_rows": rows},
        headers=[],
        rows=[],
        bbox=[20, 20, 580, 300],
    )


def test_uncertain_cell_is_preserved_and_published_for_review() -> None:
    context = SimpleNamespace(
        ocr_correction_audit=lambda: {
            "cell_anomalies": [
                {
                    "stage": "native_business",
                    "path": "credit_accounts[0].balance",
                    "role": "amount",
                    "value": "12O0",
                    "reason_codes": ["role_validation_failed", "preserved_unresolved_value"],
                    "source_refs": [{"logical_page": 3, "source_page": 2}],
                }
            ]
        },
        page_topology_audit=lambda: {"issues": []},
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["observed_value"] == "12O0"
    assert issues[0]["status"] == "requires_review"
    assert dataset_states_from_issues(issues) == {}


def test_pboc_controlled_vocabulary_is_diagnostic_and_broad() -> None:
    assert validate_pboc_field("贷后管理", "inquiry_reason").valid is True
    assert validate_pboc_field("司法调查", "inquiry_reason").valid is True
    invalid = validate_pboc_field("贷后管埋", "inquiry_reason")
    assert invalid.assessed is True
    assert invalid.valid is False
    assert invalid.reason_code == "controlled_vocabulary_mismatch"


def test_only_row_blocking_structure_failures_degrade_dataset_presence() -> None:
    issue = make_issue(
        category="ocr_structure_correction",
        issue_code="recognized_native_section_missing_required_value",
        message="required value missing",
        target_dataset="credit_lines",
    )

    assert dataset_states_from_issues([issue])["credit_lines"]["presence_status"] == "extraction_failed"


def test_default_liability_currency_is_not_substantive_source_evidence() -> None:
    assert liability_record_is_substantive({"liability_id": "synthetic", "currency": "CNY"}) is False
    assert liability_record_is_substantive({"currency": "CNY", "responsibility_amount": 4000000}) is True


def test_native_liability_extraction_drops_header_only_tables_before_counting() -> None:
    header_only = _table(
        "header-only",
        [["责任人类型", "保证合同编号"], ["", ""], ["报告日期", "2024.07.22"]],
    )
    substantive = _table(
        "liability-1",
        [
            ["责任人类型", "保证合同编号", "还款责任金额"],
            ["保证人", "B10512900H0001C240320GR3596590", "4,000,000"],
        ],
    )
    result = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[header_only, substantive])]
    )

    rows = _extract_liabilities(result)

    assert len(rows) == 1
    assert rows[0]["sequence"] == 1
    assert rows[0]["responsibility_amount"] == 4000000


def test_liability_dedupe_tolerates_transposed_ocr_contract_identifier() -> None:
    native = {
        "contract_number": "J10257010H00012016DB20221130NS000003202",
        "related_party_id_number": "5309020000053763",
        "responsibility_amount": 700210,
        "balance": 116702,
        "due_date": "2024-11-20",
        "sequence": 4,
    }
    replay = {
        "contract_number": "00012016DBJ10257010H20221130NS000003202",
        "related_party_id_number": "5309020000053763",
        "responsibility_amount": 700210,
        "balance": 116702,
        "due_date": "2024-11-20",
        "sequence": 14,
    }
    distinct = {
        "contract_number": None,
        "related_party_id_number": "5329010002043257",
        "responsibility_amount": 700210,
        "balance": 3500000,
        "due_date": "2024-09-13",
        "sequence": 10,
    }

    rows = _dedupe_liability_records([native, replay, distinct])

    assert rows == [native, distinct]
    assert [row["sequence"] for row in rows] == [1, 2]


def test_credit_line_limits_are_schema_typed_amounts() -> None:
    from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import _mapping_role

    assert _mapping_role({}, "total_limit") == "amount"
    assert _mapping_role({}, "used_limit") == "amount"


def test_final_grid_count_repairs_only_zero_expected_count() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_expected_repayment_records_count": 0,
                "personal_detail_expected_credit_lines_count": 5,
            },
            "datasets": {},
        },
        final_dataset_counts={"repayment_records": 583, "credit_lines": 4},
    )

    assert content["facts"]["personal_detail_expected_repayment_records_count"] == 583
    assert content["facts"]["personal_detail_expected_credit_lines_count"] == 5


def test_withheld_typed_value_is_partial_and_has_field_observation() -> None:
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="withheld",
        target_dataset="credit_accounts",
        target_record_id="account:1",
        field_name="balance",
        observed_value="12O0",
        reason_codes=("role_validation_failed", "normalized_value_withheld"),
    )

    assert dataset_states_from_issues([issue])["credit_accounts"]["presence_status"] == "partial"
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_dataset_states": dataset_states_from_issues([issue]),
            },
            "datasets": {
                "credit_accounts": [{"record_id": "account:1"}],
                "personal_detail_extraction_issues": [issue],
            },
        }
    )
    observations = content["datasets"]["personal_detail_field_observations"]
    balance = next(row for row in observations if row["field_name"] == "balance")
    assert balance["business_record_id"] == "account:1"
    assert balance["raw_value"] == "12O0"
    assert balance["observation_status"] == "unreadable"
    statuses = {row["dataset_name"]: row for row in content["datasets"]["personal_detail_dataset_status"]}
    assert statuses["credit_accounts"]["presence_status"] == "partial"


def test_tolerant_native_parser_accepts_unique_high_margin_label_damage() -> None:
    table = _table(
        "credit-line",
        [
            ["授信协议标", "授信额度用途", "管理机构"],
            ["AGREEMENT0001", "循环额度", "示例银行股份有限公司"],
        ],
    )
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0001"


def test_native_parser_uses_whole_page_ocr_when_cell_structure_is_incomplete() -> None:
    table = _table(
        "credit-line-incomplete",
        [
            ["授信协议标识", "授信额度用途"],
            ["", "循环额度"],
        ],
    )
    page = SimpleNamespace(page_number=1, source_page_number=1, tables=[table])
    recorded: list[dict[str, object]] = []

    def full_page_ocr(_pages: set[int], *, reason: str) -> list[dict[str, object]]:
        assert "missing_required_value" in reason
        return [
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {"text": "授信协议标识", "bbox": [20, 100, 180, 125]},
                    {"text": "授信额度用途", "bbox": [300, 100, 450, 125]},
                    {"text": "AGREEMENT0002", "bbox": [20, 145, 180, 170]},
                    {"text": "循环额度", "bbox": [300, 145, 450, 170]},
                ],
            }
        ]

    context = SimpleNamespace(
        pages=[page],
        reading_order_by_logical={1: 1},
        tables_continue=lambda _left, _right: None,
        full_page_ocr_evidence=full_page_ocr,
        _personal_detail_extraction_issues=recorded,
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0002"
    assert context._personal_detail_extraction_issues[0]["status"] == "resolved"


def test_native_parser_segments_repeated_liabilities_from_corrected_page_rows() -> None:
    def line(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "bbox": [x, y, x + 120, y + 18], "confidence": 0.98}

    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 7,
                "source_page": 4,
                "lines": [
                    line("(六)相关还款责任信息", 20, 10),
                    line("账户1", 20, 40),
                    line("责任人类型", 20, 70),
                    line("还款责任金额", 180, 70),
                    line("保证合同编号", 340, 70),
                    line("保证人", 20, 100),
                    line("4,000,000", 180, 100),
                    line("G-001", 340, 100),
                    line("账户2", 20, 140),
                    line("责任人类型", 20, 170),
                    line("还款责任金额", 180, 170),
                    line("保证合同编号", 340, 170),
                    line("保证人", 20, 200),
                    line("2,400,000", 180, 200),
                    line("G-002", 340, 200),
                    line("(七)授信协议信息", 20, 240),
                ],
            }
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("repayment_liability_records")

    assert [record.fields["保证合同编号"] for record in records] == ["G-001", "G-002"]
    assert [record.fields["还款责任金额"] for record in records] == ["4,000,000", "2,400,000"]


def test_native_parser_carries_credit_agreement_card_across_corrected_pages() -> None:
    def line(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "bbox": [x, y, x + 150, y + 18], "confidence": 0.98}

    context = SimpleNamespace(
        pages=[],
        corrected_evidence_pages=lambda: [
            {
                "page": 8,
                "source_page": 4,
                "lines": [
                    line("(七)授信协议信息", 20, 10),
                    line("授信协议1", 20, 40),
                    line("授信协议标识", 20, 70),
                    line("授信额度用途", 220, 70),
                ],
            },
            {
                "page": 9,
                "source_page": 5,
                "lines": [
                    line("AGREEMENT0001", 20, 20),
                    line("循环贷款额度", 220, 20),
                    line("非信贷交易信息", 20, 60),
                ],
            },
        ],
    )

    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")

    assert len(records) == 1
    assert records[0].fields["授信协议标识"] == "AGREEMENT0001"
    assert len(records[0].source_refs) == 2


def test_scanned_auxiliary_includes_schema_parsed_native_datasets() -> None:
    context = SimpleNamespace(
        scanned_business=lambda _text: {"credit_accounts": [{"account_id": "account:1"}]},
        native_business=lambda _text: {
            "credit_lines": [{"credit_line_id": "line:1"}],
            "repayment_liability_records": [{"liability_id": "liability:1"}],
        },
    )

    result = PersonalDetailScannedVariant().extract_auxiliary_business(
        context,
        "",
        content_mode="scanned_ocr",
    )

    assert result["credit_accounts"] == [{"account_id": "account:1"}]
    assert result["credit_lines"] == [{"credit_line_id": "line:1"}]
    assert result["repayment_liability_records"] == [{"liability_id": "liability:1"}]


def test_printed_reading_order_tolerates_large_observed_page_gaps() -> None:
    pages = [
        SimpleNamespace(page_number=30, source_page_number=3, texts=[SimpleNamespace(content="第 8 页，共 8 页")]),
        SimpleNamespace(page_number=10, source_page_number=1, texts=[SimpleNamespace(content="第 1 页，共 8 页")]),
        SimpleNamespace(page_number=20, source_page_number=2, texts=[SimpleNamespace(content="第 4 页，共 8 页")]),
    ]
    result = SimpleNamespace(pages=pages, entities=SimpleNamespace(domain_specific={}))

    assert _printed_reading_order(result) == {10: 1, 20: 2, 30: 3}


def test_sections_are_derived_from_observed_page_anchors() -> None:
    def page(number: int, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            page_number=number,
            texts=[SimpleNamespace(content=text)],
            tables=[],
        )

    result = SimpleNamespace(
        pages=[
            page(1, "个人基本信息"),
            page(3, "信息概要"),
            page(8, "信贷交易信息明细"),
            page(45, "公共信息明细"),
            page(50, "查询记录"),
            page(55, "报告说明"),
        ]
    )

    sections = PersonalDetailScannedVariant().build_sections(result, "")
    by_type = {section["type"]: section for section in sections}

    assert by_type["credit_details"]["page_start"] == 8
    assert by_type["public_records"]["page_start"] == 45
    assert by_type["inquiries"]["page_start"] == 50
    assert by_type["report_explanation"]["page_start"] == 55


def test_section_roots_do_not_promote_summary_labels_or_report_explanations() -> None:
    def page(number: int, text: str) -> SimpleNamespace:
        return SimpleNamespace(page_number=number, texts=[SimpleNamespace(content=text)], tables=[])

    result = SimpleNamespace(
        pages=[
            page(1, "一 个人基本信息"),
            page(2, "二 信息概要 非循环贷账户 （六）查询记录概要"),
            page(4, "三 信贷交易信息明细"),
            page(6, "机构说明 添加日期"),
            page(12, "四 非信贷交易信息明细 五 公共信息明细"),
            page(13, "异议标注 六 查询记录 机构查询记录明细"),
            page(14, "报告说明 本人声明与异议标注的含义"),
            page(15, "编制说明"),
        ]
    )

    sections = PersonalDetailScannedVariant().build_sections(result, "")
    by_type = {section["type"]: section for section in sections}

    assert by_type["credit_details"]["page_start"] == 4
    assert by_type["inquiries"]["page_start"] == 13
    assert by_type["public_records"]["page_start"] == 12
    assert by_type["statements"]["page_end"] == 6
    assert by_type["annotations"]["page_end"] == 13
    assert by_type["report_explanation"]["page_end"] == 15


def test_split_replay_replaces_unsplit_evidence_and_inserts_dense_order() -> None:
    context = object.__new__(PersonalDetailExtractionContext)
    unsplit_geometry = SimpleNamespace(split_kind="none", segment_index=0)
    context.page_topology = SimpleNamespace(
        audit=lambda: {"logical_pages_by_source": {"1": [1], "2": [2]}},
        logicals_for_source=lambda source: (source,),
        geometry=lambda logical: unsplit_geometry if logical in {1, 2} else None,
    )
    context.source_page_by_logical = {1: 1, 2: 2}
    context.reading_order_by_logical = {1: 1, 2: 2}
    context.supplemental_page_ocr_evidence = lambda _sources, **_kwargs: [
        {"source_page": 1, "segment_index": 0, "lines": [{"text": "left"}]},
        {"source_page": 1, "segment_index": 1, "lines": [{"text": "right"}]},
    ]
    context._ocr_correction_overlay = SimpleNamespace(
        corrected_evidence_pages=lambda pages: pages,
    )

    merged = context._merge_split_replay_pages(
        [
            {"page": 1, "source_page": 1, "lines": [{"text": "unsplit"}]},
            {"page": 2, "source_page": 2, "lines": [{"text": "next"}]},
        ]
    )

    assert [page["lines"][0]["text"] for page in merged] == ["left", "right", "next"]
    assert [page["page"] for page in merged] == [1, 3, 2]
    assert context.source_page_by_logical == {1: 1, 2: 2, 3: 1}
    assert context.reading_order_by_logical == {1: 1, 3: 2, 2: 3}


def test_identifier_only_liability_rows_are_redundant_not_business_records() -> None:
    assert liability_record_is_substantive({"liability_id": "row:1", "source_refs": []}) is False
    assert liability_record_is_substantive({"liability_id": "row:2", "responsibility_type": "保证人"}) is True


def test_v2_projection_keeps_extraction_issues_as_typed_control_rows() -> None:
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="review",
        target_dataset="credit_accounts",
        field_name="balance",
        observed_value="12O0",
    )

    projected = project_personal_detail_datasets({"personal_detail_extraction_issues": [issue]})
    issue_row = projected["extraction_issues"][0]
    values = issue_row.get("normalized", issue_row)

    assert values["target_dataset"] == "credit_accounts"
    assert values["field_name"] == "balance"
    assert values["observed_value"] == "12O0"
