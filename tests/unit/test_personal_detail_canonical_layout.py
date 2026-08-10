from __future__ import annotations

from copy import deepcopy
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


def test_body_position_printed_page_duplicate_does_not_join_fragments() -> None:
    result = SimpleNamespace(
        pages=[_page(1, source=1), _page(2, source=2)]
    )
    evidence = [
        {
            "page": logical,
            "source_page": logical,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("\u4e2a\u4eba\u4fe1\u7528\u62a5\u544a \u62a5\u544a\u7f16\u53f7", [10, 20, 300, 60]),
                _line("\u7b2c 1 \u9875\uff0c\u5171 2 \u9875", [100, 100, 250, 125]),
            ],
        }
        for logical in (1, 2)
    ]

    projection = _assembler(result, evidence).build()

    assert len(projection.pages) == 2
    assert all("printed_page" not in registration for registration in projection.registrations)
    assert all(page.canonical_fragment_logical_pages in {(1,), (2,)} for page in projection.pages)


def test_conflicting_templates_with_one_proved_printed_identity_fail_closed() -> None:
    owner = SimpleNamespace()
    result = SimpleNamespace(
        pages=[_page(1, source=1), _page(2, source=2)]
    )
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("\u4e2a\u4eba\u4fe1\u7528\u62a5\u544a \u62a5\u544a\u7f16\u53f7", [10, 20, 300, 60]),
                _line("\u7b2c 1 \u9875\uff0c\u5171 2 \u9875", [220, 760, 380, 785]),
            ],
        },
        {
            "page": 2,
            "source_page": 2,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("\u4fe1\u606f\u6982\u8981", [10, 20, 300, 60]),
                _line("\u7b2c 1 \u9875\uff0c\u5171 2 \u9875", [220, 760, 380, 785]),
            ],
        },
    ]

    projection = _assembler(result, evidence, owner=owner).build()

    assert projection.pages == ()
    assert projection.unresolved_pages == (1, 2)
    assert {
        registration["template_id"] for registration in projection.registrations
    } == {"report_header_and_identity", "information_summary"}
    assert owner._personal_detail_extraction_issues[0]["issue_code"] == (
        "canonical_fragment_template_conflict"
    )


def test_static_split_does_not_inherit_missing_or_non_authoritative_identity() -> None:
    result = SimpleNamespace(
        pages=[_page(10, source=1, width=300), _page(20, source=1, width=300)]
    )
    evidence = [
        {
            "page": 10,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "lines": [
                _line("\u4e2a\u4eba\u4fe1\u7528\u62a5\u544a \u62a5\u544a\u7f16\u53f7", [10, 20, 280, 60]),
                _line("\u7b2c 1 \u9875\uff0c\u5171 4 \u9875", [100, 100, 250, 125]),
            ],
        },
        {
            "page": 20,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "source_crop_bbox": [300, 0, 600, 800],
            "plugin_static_subpage": True,
            "lines": [_line("\u4e2a\u4eba\u57fa\u672c\u4fe1\u606f", [10, 20, 200, 60])],
        },
    ]
    resolutions = (
        None,
        {
            "resolved": False,
            "authoritative": False,
            "identity_fallback": True,
            "printed_total": 4,
            "printed_page_by_logical": {10: 1},
        },
        {
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "printed_total": 4,
            "printed_page_by_logical": {},
        },
    )

    for resolution in resolutions:
        owner = SimpleNamespace()
        case_evidence = deepcopy(evidence)
        if resolution is not None:
            owner.reading_order_resolution = resolution
            # Even an exact footer cannot override an available context
            # resolution that rejected or omitted the logical page.
            case_evidence[0]["lines"][1]["bbox"] = [100, 760, 250, 785]
        projection = _assembler(result, case_evidence, owner=owner).build()

        assert len(projection.pages) == 2
        assert all(len(page.canonical_fragment_logical_pages) == 1 for page in projection.pages)


def test_authoritative_context_identity_allows_valid_same_template_static_split() -> None:
    owner = SimpleNamespace(
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "basis": "complete_unique_printed_page_permutation",
            "printed_total": 4,
            "printed_page_by_logical": {10: 1},
            "unresolved_logical_pages": [],
        }
    )
    result = SimpleNamespace(
        pages=[_page(10, source=1, width=300), _page(20, source=1, width=300)]
    )
    evidence = [
        {
            "page": 10,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "source_crop_bbox": [0, 0, 300, 800],
            "lines": [
                _line("\u4e2a\u4eba\u4fe1\u7528\u62a5\u544a \u62a5\u544a\u7f16\u53f7", [10, 20, 280, 60]),
                _line("\u7b2c 1 \u9875\uff0c\u5171 4 \u9875", [100, 100, 250, 125]),
            ],
        },
        {
            "page": 20,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "source_crop_bbox": [300, 0, 600, 800],
            "plugin_static_subpage": True,
            "lines": [_line("\u4e2a\u4eba\u57fa\u672c\u4fe1\u606f", [10, 20, 200, 60])],
        },
    ]
    topology = SimpleNamespace(
        geometry=lambda logical: SimpleNamespace(
            source_crop_bbox=(0.0, 0.0, 300.0, 800.0)
            if logical == 10
            else (300.0, 0.0, 600.0, 800.0)
        )
    )

    projection = _assembler(result, evidence, topology=topology, owner=owner).build()

    assert len(projection.pages) == 1
    assert projection.pages[0].canonical_fragment_logical_pages == (10, 20)
    assert projection.registrations[0]["printed_identity_basis"] == (
        "context_authoritative_printed_order"
    )


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


def test_unanchored_table_page_does_not_inherit_previous_template_by_shape_alone() -> None:
    summary_table = SimpleNamespace(table_id="summary:1", metadata={"raw_rows": [["A", "B"]]}, bbox=[10, 80, 590, 300])
    unrelated_table = SimpleNamespace(table_id="unknown:2", metadata={"raw_rows": [["X", "Y"]]}, bbox=[10, 80, 590, 300])
    result = SimpleNamespace(
        pages=[
            _page(1, source=1, tables=[summary_table]),
            _page(2, source=2, tables=[unrelated_table]),
        ]
    )
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("\u4fe1\u606f\u6982\u8981", [20, 20, 220, 50])],
        },
        {
            "page": 2,
            "source_page": 2,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("\u672a\u6ce8\u518c\u7684\u4e1a\u52a1\u8868\u683c\u5185\u5bb9", [20, 20, 400, 50])],
        },
    ]

    projection = _assembler(result, evidence).build()

    assert projection.registrations[0]["template_id"] == "information_summary"
    assert projection.registrations[1]["template_id"] == "unresolved"
    assert projection.registrations[1]["basis"] == "canonical_registration_exhausted"
    assert projection.unresolved_pages == (2,)


def test_explicit_table_continuation_can_inherit_previous_canonical_template() -> None:
    summary_table = SimpleNamespace(table_id="summary:left", metadata={"raw_rows": [["A", "B"]]}, bbox=[10, 80, 590, 300])
    continuation_table = SimpleNamespace(table_id="summary:right", metadata={"raw_rows": [["C", "D"]]}, bbox=[10, 80, 590, 300])
    result = SimpleNamespace(
        pages=[
            _page(1, source=1, tables=[summary_table]),
            _page(2, source=2, tables=[continuation_table]),
        ]
    )
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("\u4fe1\u606f\u6982\u8981", [20, 20, 220, 50])],
        },
        {
            "page": 2,
            "source_page": 2,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("\u8868\u683c\u7684\u65e0\u6807\u9898\u7eed\u9875", [20, 20, 400, 50])],
        },
    ]
    owner = SimpleNamespace(tables_continue=lambda left, right: (left, right) == ("summary:left", "summary:right"))

    projection = _assembler(result, evidence, owner=owner).build()

    assert projection.registrations[1]["template_id"] == "information_summary"
    assert projection.registrations[1]["basis"] == "canonical_flow_continuation"
    assert "explicit_table_continuation" in projection.registrations[1]["signals"]
    assert projection.unresolved_pages == ()


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
    page_image_resolver = object()
    context._page_image_resolver = page_image_resolver
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
    assert observed[0]["page_image_resolver"] is page_image_resolver
    assert observed[0]["enable_static_status_validation"] is True
    assert observed[0]["enable_candidate_b_amount_pairing"] is True
    assert all(call["extra_status_chars"] == {"A", "#"} for call in observed)
    assert observed[1]["page_image_resolver"] is None
    assert observed[1].get("enable_static_status_validation", False) is False
    assert observed[1].get("enable_candidate_b_amount_pairing", False) is False
    assert observed[0]["lines"] == [canonical_line]


def test_monthly_cross_page_augmentation_requires_authoritative_order(
    monkeypatch,
) -> None:
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {1: 1, 2: 2}
    context.reading_order_resolution = {
        "resolved": False,
        "authoritative": False,
        "basis": "unresolved_identity_fallback",
    }
    context._page_image_resolver = None
    anchor = _line("2024年01月-2024年12月的还款记录", [10, 700, 500, 730])
    continuation = _line("2024 N N N N N N N N N N N N", [10, 20, 500, 40])
    stale_shifted = {
        **continuation,
        "bbox": [10, 820, 500, 840],
        "source_logical_page": 2,
        "coordinate_status": "cross_page_y_shift",
    }
    bundles = [
        {
            "page": 1,
            "local_structure_evidence": {
                "page": 1,
                "page_height": 800,
                "lines": [anchor],
            },
            "micro_grid_evidence": {
                "page": 1,
                "page_height": 800,
                "lines": [anchor, stale_shifted],
                "credit_cross_page_augmented": True,
                "continuation_logical_pages": [2],
                "continuation_source_table_geometry_by_page": {
                    "2": [{"table_id": "pt_2_0"}]
                },
            },
        },
        {
            "page": 2,
            "local_structure_evidence": {
                "page": 2,
                "page_height": 800,
                "lines": [continuation],
            },
            "micro_grid_evidence": {
                "page": 2,
                "page_height": 800,
                "lines": [continuation],
                "source_table_geometry": [{"table_id": "pt_2_0"}],
            },
        },
    ]
    context.parse_result = SimpleNamespace(
        pages=[],
        entities=SimpleNamespace(
            domain_specific={"_page_evidence_bundles": bundles}
        ),
    )
    context.corrected_evidence_pages = lambda: [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [anchor],
        },
        {
            "page": 2,
            "source_page": 2,
            "page_width": 600,
            "page_height": 800,
            "lines": [continuation],
        },
    ]
    augmentation_calls: list[object] = []
    materialized_inputs: list[dict[str, object]] = []

    from docmirror.plugins.credit_report import micro_grid_materialize

    real_augment = micro_grid_materialize.augment_credit_repayment_evidence_bundles

    def augment(*args, **kwargs):
        augmentation_calls.append(object())
        return real_augment(*args, **kwargs)

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize.augment_credit_repayment_evidence_bundles",
        augment,
    )

    def materialize(detached, **_kwargs):
        materialized_inputs.append(deepcopy(detached))
        return []

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize.materialize_credit_repayment_micro_grids_from_bundles",
        materialize,
    )

    assert context.corrected_repayment_records() == []
    assert augmentation_calls == []
    assert len(materialized_inputs) == 2
    for detached in materialized_inputs:
        first = detached["_page_evidence_bundles"][0]["micro_grid_evidence"]
        assert "credit_cross_page_augmented" not in first
        assert "continuation_logical_pages" not in first
        assert "continuation_source_table_geometry_by_page" not in first
        assert not any(
            line.get("source_logical_page") == 2
            for line in first.get("lines") or []
            if isinstance(line, dict)
        )

    context._cache = {}
    context.reading_order_resolution = {
        "resolved": True,
        "authoritative": True,
        "basis": "complete_unique_printed_page_permutation",
    }
    augmentation_calls.clear()
    materialized_inputs.clear()

    assert context.corrected_repayment_records() == []
    assert len(augmentation_calls) == 2
    assert len(materialized_inputs) == 2
    for detached in materialized_inputs:
        first = detached["_page_evidence_bundles"][0]["micro_grid_evidence"]
        assert first["credit_cross_page_augmented"] is True
        assert first["continuation_logical_pages"] == [2]
        assert "2" in first["continuation_source_table_geometry_by_page"]
        assert any(
            line.get("source_logical_page") == 2
            for line in first.get("lines") or []
            if isinstance(line, dict)
        )


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
        lambda _grid, **_kwargs: list(records),
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


def test_inquiry_schema_joins_headerless_pages_but_withholds_merged_date_cell() -> None:
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

    context = SimpleNamespace(pages=pages)
    rows = _extract_inquiries(context)

    institutional = [row for row in rows if row["inquiry_type"] == "institution"]
    personal = [row for row in rows if row["inquiry_type"] == "personal"]
    assert [row["sequence"] for row in institutional] == [1, 2]
    assert [row["sequence"] for row in personal] == [1]
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_row_cells_unresolved"
        and issue.get("observed_value", {}).get("sequence") == 3
        for issue in context._personal_detail_extraction_issues
    )


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

    context = SimpleNamespace(corrected_evidence_pages=lambda: [page])
    rows = _canonical_inquiry_line_rows(context)

    assert [row["sequence"] for row in rows] == [88, 89, 90]
    assert rows[1]["institution"] == "机构乙"
    assert "extraction_status" not in rows[1]
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_sequence_prefix_noise_corrected"
        and issue.get("target_record_id") == rows[1]["inquiry_id"]
        and issue.get("field_name") == "sequence"
        and issue.get("observed_value", {}).get("raw_sequence") == 789
        and issue.get("candidate_value", {}).get("normalized_sequence") == 89
        and issue.get("severity") == "info"
        and issue.get("status") == "resolved"
        for issue in context._personal_detail_extraction_issues
    )


def test_canonical_inquiry_line_uses_longest_reason_suffix_and_final_boundary() -> None:
    line = _line(
        "1 2024.01.02 深圳前海微众银行股份有限公司 法人代表、负责人、高管等资信审查",
        [10, 10, 580, 28],
    )
    evidence = {
        "page": 28,
        "source_page": 14,
        "canonical_template_id": "annotations_and_inquiries",
        "lines": [line],
    }
    table = SimpleNamespace(
        table_id="inquiries",
        bbox=[10, 5, 590, 40],
        metadata={
            "raw_rows": [
                ["编号", "查询日期", "查询机构", "查询原因"],
                [
                    "1",
                    "2024.01.02",
                    "深圳前海微众银行股份有限公司 法人代表、负责人、高管等",
                    "资信审查",
                ],
            ]
        },
        headers=[],
        rows=[],
        confidence=0.99,
    )
    page = _page(28, source=14, tables=[table])
    page.canonical_template_id = "annotations_and_inquiries"
    context = SimpleNamespace(
        pages=[page],
        corrected_evidence_pages=lambda: [evidence],
    )

    line_rows = _canonical_inquiry_line_rows(context)
    final_rows = _extract_inquiries(context)

    assert line_rows[0]["institution"] == "深圳前海微众银行股份有限公司"
    assert line_rows[0]["reason"] == "法人代表、负责人、高管等资信审查"
    assert final_rows[0]["institution"] == "深圳前海微众银行股份有限公司"
    assert final_rows[0]["reason"] == "法人代表、负责人、高管等资信审查"


def test_inquiry_sequence_prefix_suppression_preserves_valid_101() -> None:
    table = SimpleNamespace(
        table_id="inquiries",
        bbox=[10, 5, 590, 80],
        metadata={
            "raw_rows": [
                ["编号", "查询日期", "查询机构", "查询原因"],
                ["1", "2024.01.01", "机构甲银行股份有限公司", "贷款审批"],
                ["101", "2024.01.01", "机构乙银行股份有限公司", "融资审批"],
            ]
        },
        confidence=0.99,
    )
    page = _page(28, source=14, tables=[table])
    page.canonical_template_id = "annotations_and_inquiries"
    context = SimpleNamespace(pages=[page])

    rows = _extract_inquiries(context)

    assert [row["sequence"] for row in rows] == [1, 101]
    assert not any(
        issue.get("issue_code") == "candidate_b_inquiry_sequence_prefix_suppressed"
        for issue in context._personal_detail_extraction_issues
    )


def test_inquiry_sequence_prefix_repair_requires_exact_row_neighbors() -> None:
    def extract(sequences: list[int]) -> tuple[list[dict[str, object]], SimpleNamespace]:
        table = SimpleNamespace(
            table_id="inquiries",
            bbox=[10, 5, 590, 100],
            metadata={
                "raw_rows": [
                    ["编号", "查询日期", "查询机构", "查询原因"],
                    *[
                        [str(sequence), "2024.01.01", "机构甲银行股份有限公司", "贷款审批"]
                        for sequence in sequences
                    ],
                ]
            },
            confidence=0.99,
        )
        page = _page(28, source=14, tables=[table])
        page.canonical_template_id = "annotations_and_inquiries"
        context = SimpleNamespace(pages=[page])
        return _extract_inquiries(context), context

    repaired_rows, repaired_context = extract([88, 789, 90])
    isolated_rows, isolated_context = extract([89, 789])
    genuine_rows, genuine_context = extract([788, 789, 790])

    assert [row["sequence"] for row in repaired_rows] == [88, 89, 90]
    assert any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_prefix_noise_corrected"
        and issue.get("observed_value", {}).get("raw_sequence") == 789
        and issue.get("candidate_value") == {"normalized_sequence": 89}
        for issue in repaired_context._personal_detail_extraction_issues
    )
    assert [row["sequence"] for row in isolated_rows] == [89, 789]
    assert [row["sequence"] for row in genuine_rows] == [788, 789, 790]
    assert not any(
        issue.get("issue_code")
        == "candidate_b_inquiry_sequence_prefix_noise_corrected"
        for context in (isolated_context, genuine_context)
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )
