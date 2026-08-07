from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    PBOCCanonicalTemplateAssembler,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    _page_ocr_score,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _canonical_inquiry_line_rows,
    _extract_inquiries,
)


def _line(text: str, bbox: list[float]) -> dict[str, object]:
    return {"text": text, "bbox": bbox, "confidence": 0.98}


def _page(logical: int, *, source: int, width: float = 600, height: float = 800, tables=()):
    return SimpleNamespace(
        page_number=logical,
        source_page_number=source,
        width=width,
        height=height,
        tables=list(tables),
        texts=[],
    )


def _assembler(result, evidence, _retries=None, topology=None, owner=None):
    return PBOCCanonicalTemplateAssembler(
        result,
        topology=topology or SimpleNamespace(geometry=lambda _logical: None),
        reading_order_by_logical={page.page_number: index for index, page in enumerate(result.pages, start=1)},
        source_evidence_loader=lambda: evidence,
        issue_owner=owner or SimpleNamespace(),
    )


def test_unknown_layout_is_reported_without_registration_ocr_or_table_rewrite() -> None:
    table = SimpleNamespace(
        table_id="account",
        bbox=[10, 100, 590, 220],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": [["字段一", "字段二"], ["A00000000", "样例银仃"]],
            "cell_bboxes": [
                [[10, 100, 300, 150], [300, 100, 590, 150]],
                [[10, 150, 300, 220], [300, 150, 590, 220]],
            ],
        },
    )
    result = SimpleNamespace(pages=[_page(1, source=1, tables=[table])])
    initial = [{"page": 1, "source_page": 1, "page_width": 600, "page_height": 800, "lines": []}]
    replay = {
        "page": 1,
        "source_page": 1,
        "page_width": 600,
        "page_height": 800,
        "lines": [
            _line("信贷交易信息明细", [20, 20, 250, 50]),
            _line("账户标识", [20, 110, 150, 140]),
            _line("管理机构", [320, 110, 450, 140]),
            _line("B12345678", [20, 170, 180, 205]),
            _line("样例银行", [320, 170, 450, 205]),
        ],
    }

    projection = _assembler(result, initial, {1: replay}).build()

    assert projection.unresolved_pages == (1,)
    assert projection.pages == ()
    assert projection.registrations[0]["basis"] == "canonical_registration_exhausted"
    assert projection.audit()["template_registration_ocr_used"] is False


def test_fragments_with_same_printed_page_join_one_canonical_canvas() -> None:
    result = SimpleNamespace(
        pages=[
            _page(10, source=1, width=300),
            _page(20, source=1, width=300),
        ]
    )
    evidence = [
        {
            "page": 10,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "lines": [
                _line("个人信用报告 报告编号", [10, 20, 280, 60]),
                _line("第1页，共1页", [100, 760, 250, 790]),
            ],
        },
        {
            "page": 20,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "source_crop_bbox": [300, 0, 600, 800],
                    "plugin_static_subpage": True,
            "lines": [
                _line("个人基本信息", [10, 20, 200, 60]),
            ],
        },
    ]
    geometry = {
        10: SimpleNamespace(source_crop_bbox=(0.0, 0.0, 300.0, 800.0)),
        20: SimpleNamespace(source_crop_bbox=(300.0, 0.0, 600.0, 800.0)),
    }
    topology = SimpleNamespace(geometry=lambda logical: geometry[logical])

    projection = _assembler(result, evidence, {}, topology=topology).build()

    assert len(projection.pages) == 1
    assert projection.pages[0].canonical_fragment_logical_pages == (10, 20)
    assert projection.pages[0].width == 600
    assert projection.fragment_groups[0]["joined_fragment_count"] == 2
    assert projection.fragment_groups[0]["coverage_status"] == "full"


def test_explicit_summary_heading_outranks_repeated_account_category_names() -> None:
    result = SimpleNamespace(pages=[_page(1, source=1)])
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("二 信息概要", [10, 10, 200, 40]),
                _line("非循环贷账户 循环贷账户一 贷记卡账户 账户数 余额 授信总额", [10, 80, 580, 130]),
            ],
        }
    ]

    projection = _assembler(result, evidence, {}).build()

    assert projection.registrations[0]["template_id"] == "information_summary"
    assert projection.registrations[0]["confidence"] >= 0.92


def test_unregistered_page_is_reported_and_not_generically_extracted() -> None:
    owner = SimpleNamespace()
    result = SimpleNamespace(pages=[_page(1, source=1)])
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("无法识别的页面内容但是并非空白页面", [20, 20, 500, 80])],
        }
    ]
    retry = {1: {**evidence[0], "lines": [_line("仍然无法识别的页面内容", [20, 20, 500, 80])]}}

    projection = _assembler(result, evidence, retry, owner=owner).build()

    assert projection.pages == ()
    assert projection.unresolved_pages == (1,)
    issues = owner._personal_detail_extraction_issues
    assert issues[0]["issue_code"] == "canonical_page_registration_failed"
    assert "no_generic_layout_fallback" in issues[0]["reason_codes"]


def test_monthly_materialization_uses_canonical_page_lines_without_cell_ocr(monkeypatch) -> None:
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {1: 1}
    context.parse_result = SimpleNamespace(
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    {
                        "page": 1,
                        "local_structure_evidence": {"page": 1, "lines": [_line("sealed", [1, 1, 5, 5])]},
                        "micro_grid_evidence": {"page": 1, "lines": [_line("sealed", [1, 1, 5, 5])]},
                    }
                ]
            }
        )
    )
    canonical_line = _line("2024年01月—2024年12月的还款记录", [10, 10, 500, 40])
    context.corrected_evidence_pages = lambda: [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [canonical_line],
        }
    ]
    observed: list[dict[str, object]] = []

    def materialize(detached, **kwargs):
        bundle = detached["_page_evidence_bundles"][0]
        observed.append({**kwargs, "lines": bundle["micro_grid_evidence"]["lines"]})
        return []

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize.materialize_credit_repayment_micro_grids_from_bundles",
        materialize,
    )

    assert context.corrected_repayment_records() == []
    assert len(observed) == 2
    assert all(call["enable_cell_ocr"] is False for call in observed)
    assert all(call["page_image_resolver"] is None for call in observed)
    assert observed[0]["lines"] == [canonical_line]


def test_monthly_sequence_holes_are_reported_from_schema_not_legacy_ocr(monkeypatch) -> None:
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {1: 1}
    context.parse_result = SimpleNamespace(
        entities=SimpleNamespace(domain_specific={"_page_evidence_bundles": [{"page": 1}]})
    )
    context.corrected_evidence_pages = lambda: [
        {"page": 1, "source_page": 1, "page_width": 600, "page_height": 800, "lines": []}
    ]
    records = [
        {
            "record_id": "monthly:1",
            "year": 2024,
            "month": 1,
            "status_code": "N",
            "source_cell_refs": [{"grid_id": "grid:1"}],
        },
        {
            "record_id": "monthly:2",
            "year": 2024,
            "month": 3,
            "status_code": "N",
            "source_cell_refs": [{"grid_id": "grid:1"}],
        },
    ]

    monkeypatch.setattr(
        "docmirror.models.mirror.domain_access.micro_grid_structures_from_domain_specific",
        lambda _domain: [{}],
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.repayment_grid.records_from_micro_grid_dict",
        lambda _grid: list(records),
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.repayment_grid.dedupe_repayment_records",
        lambda values: list(values),
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize.materialize_credit_repayment_micro_grids_from_bundles",
        lambda *_args, **_kwargs: None,
    )

    assert len(context.corrected_repayment_records()) == 2
    issue = context._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "canonical_monthly_reconstruction_incomplete"
    assert issue["observed_value"] == {"canonical_row_count": 2}
    assert issue["candidate_value"]["structural_expected_row_count"] == 3
    assert issue["candidate_value"]["schema_implied_row_count"] == 3
    assert issue["candidate_value"]["missing_month_count"] == 1


def test_page_orientation_score_prefers_horizontal_rows_on_portrait_page() -> None:
    horizontal = [{"text": "89 2022.05.22 示例机构 贷款审批", "bbox": [10, 10, 400, 28], "confidence": 0.99}]
    vertical = [{"text": "89 2022.05.22 示例机构 贷款审批", "bbox": [10, 10, 28, 400], "confidence": 0.99}]

    assert _page_ocr_score(horizontal, image_shape=(800, 600, 3)) > _page_ocr_score(
        vertical,
        image_shape=(800, 600, 3),
    )


def test_inquiry_schema_joins_headerless_pages_and_splits_merged_date_cell() -> None:
    def table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return SimpleNamespace(table_id=table_id, metadata={"raw_rows": rows}, bbox=[10, 10, 590, 700])

    pages = [
        _page(
            27,
            source=14,
            tables=[table("q27", [["编号", "查询日期", "查询机构", "查询原因"], ["1", "2024.01.01", "机构甲", "贷款审批"]])],
        ),
        _page(28, source=14, tables=[table("q28", [["2", "2023.12.01", "机构乙", "贷后管理"]])]),
        _page(
            29,
            source=15,
            tables=[
                table("q29a", [["3", "2023.11.01 机构丙", "", "信用卡审批"]]),
                table("q29b", [["编号", "查询日期", "查询机构", "查询原因"], ["1", "2023.10.01", "本人", "本人查询"]]),
            ],
        ),
    ]
    for page in pages:
        page.canonical_template_id = "annotations_and_inquiries"

    rows = _extract_inquiries(SimpleNamespace(pages=pages))

    institutional = [row for row in rows if row["inquiry_type"] == "institution"]
    personal = [row for row in rows if row["inquiry_type"] == "personal"]
    assert [row["sequence"] for row in institutional] == [1, 2, 3]
    assert institutional[-1]["institution"] == "机构丙"
    assert [row["sequence"] for row in personal] == [1]


def test_canonical_inquiry_lines_correct_prefixed_sequence_noise_without_value_lexicon() -> None:
    page = {
        "page": 28,
        "source_page": 14,
        "canonical_template_id": "annotations_and_inquiries",
        "lines": [
            _line("88 2022.05.31 机构甲 贷款审批", [10, 10, 500, 25]),
            _line("789 2022.05.22 机构乙 贷后管理", [10, 30, 500, 45]),
            _line("90 2022.05.20 机构丙 信用卡审批", [10, 50, 500, 65]),
        ],
    }

    rows = _canonical_inquiry_line_rows(SimpleNamespace(corrected_evidence_pages=lambda: [page]))

    assert [row["sequence"] for row in rows] == [88, 89, 90]
    assert rows[1]["institution"] == "机构乙"
    assert rows[1]["extraction_status"] == "review"
    assert rows[1]["audit"]["raw_sequence"] == 789
