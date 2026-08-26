from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.canonical_layout import (
    PBOCCanonicalTemplateAssembler,
    _project_table,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
    _page_ocr_score,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _canonical_inquiry_line_rows,
    _extract_inquiries,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)
from docmirror.plugins.credit_report.personal_detail_scanned.table_ownership import (
    canonical_table_role,
)


def _line(text: str, bbox: list[float]) -> dict[str, object]:
    return {
        "text": text,
        "bbox": bbox,
        "confidence": 0.98,
        "evidence_ids": [f"test-line:{bbox[0]}:{bbox[1]}:{text}"],
    }


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
                _line("个人基本信息", [10, 20, 280, 60]),
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
        10: SimpleNamespace(
            source_page=1,
            source_crop_bbox=(0.0, 0.0, 300.0, 800.0),
            transform_usable=True,
            selected_rotation=0,
        ),
        20: SimpleNamespace(
            source_page=1,
            source_crop_bbox=(300.0, 0.0, 600.0, 800.0),
            transform_usable=True,
            selected_rotation=0,
        ),
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
                _line("\u4e2a\u4eba\u57fa\u672c\u4fe1\u606f", [10, 20, 300, 60]),
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
                _line("\u4e2a\u4eba\u57fa\u672c\u4fe1\u606f", [10, 20, 300, 60]),
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
                _line("\u4e2a\u4eba\u57fa\u672c\u4fe1\u606f", [10, 20, 280, 60]),
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
                _line("\u4e2a\u4eba\u57fa\u672c\u4fe1\u606f", [10, 20, 280, 60]),
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
            source_page=1,
            source_crop_bbox=(0.0, 0.0, 300.0, 800.0)
            if logical == 10
            else (300.0, 0.0, 600.0, 800.0),
            transform_usable=True,
            selected_rotation=0,
        )
    )

    projection = _assembler(result, evidence, topology=topology, owner=owner).build()

    assert len(projection.pages) == 1
    assert projection.pages[0].canonical_fragment_logical_pages == (10, 20)
    assert projection.registrations[0]["printed_identity_basis"] == (
        "context_authoritative_printed_order"
    )


def test_vector_text_fallback_preserves_sealed_evidence_ownership() -> None:
    sealed_heading = SimpleNamespace(
        content="一个人基本信息",
        bbox=[200.0, 80.0, 400.0, 110.0],
        evidence_ids=["ev:page-3:heading"],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=3,
                source_page_number=3,
                width=600.0,
                height=800.0,
                tables=[],
                texts=[sealed_heading],
            )
        ]
    )

    projection = _assembler(result, []).build()

    assert len(projection.pages) == 1
    assert projection.pages[0].canonical_template_id == "report_header_and_identity"
    assert projection.evidence_pages[0]["lines"][0]["evidence_ids"] == [
        "ev:page-3:heading"
    ]
    assert projection.pages[0].texts[0].evidence_ids == ["ev:page-3:heading"]
    assert projection.pages[0].texts[0].source_bbox == [200.0, 80.0, 400.0, 110.0]


def test_vector_text_fallback_without_sealed_owner_stays_unregistered() -> None:
    unsealed_heading = SimpleNamespace(
        content="一个人基本信息",
        bbox=[200.0, 80.0, 400.0, 110.0],
        evidence_ids=[],
    )
    result = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=3,
                source_page_number=3,
                width=600.0,
                height=800.0,
                tables=[],
                texts=[unsealed_heading],
            )
        ]
    )

    projection = _assembler(result, []).build()

    assert projection.pages == ()
    assert projection.unresolved_pages == (3,)


def test_crop_less_multi_fragment_canonical_page_is_withheld_and_reported() -> None:
    owner = SimpleNamespace(
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "identity_fallback": False,
            "printed_total": 4,
            "printed_page_by_logical": {10: 1},
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
            "lines": [
                _line("个人基本信息", [10, 20, 280, 60]),
                _line("第 1 页，共 4 页", [100, 760, 250, 785]),
            ],
        },
        {
            "page": 20,
            "source_page": 1,
            "page_width": 300,
            "page_height": 800,
            "plugin_static_subpage": True,
            "lines": [_line("个人基本信息", [10, 20, 200, 60])],
        },
    ]

    projection = _assembler(result, evidence, owner=owner).build()

    assert projection.pages == ()
    assert projection.unresolved_pages == (10, 20)
    assert all(registration["status"] == "unresolved" for registration in projection.registrations)
    issue = owner._personal_detail_extraction_issues[0]
    assert issue["issue_code"] == "canonical_fragment_geometry_unresolved"
    assert issue["observed_value"]["geometry_failure"] == (
        "source_crop_bbox_missing_or_invalid"
    )


def test_overlapping_or_cross_source_fragment_crops_are_withheld() -> None:
    for second_source, second_crop, failure in (
        (1, (250.0, 0.0, 550.0, 800.0), "source_crop_bboxes_overlap"),
        (2, (300.0, 0.0, 600.0, 800.0), "fragments_do_not_share_source_surface"),
    ):
        owner = SimpleNamespace(
            reading_order_resolution={
                "resolved": True,
                "authoritative": True,
                "identity_fallback": False,
                "printed_total": 4,
                "printed_page_by_logical": {10: 1},
            }
        )
        result = SimpleNamespace(
            pages=[_page(10, source=1, width=300), _page(20, source=second_source, width=300)]
        )
        evidence = [
            {
                "page": 10,
                "source_page": 1,
                "page_width": 300,
                "page_height": 800,
                "source_crop_bbox": [0, 0, 300, 800],
                "lines": [
                    _line("个人基本信息", [10, 20, 280, 60]),
                    _line("第 1 页，共 4 页", [100, 760, 250, 785]),
                ],
            },
            {
                "page": 20,
                "source_page": second_source,
                "page_width": 300,
                "page_height": 800,
                "source_crop_bbox": list(second_crop),
                "plugin_static_subpage": True,
                "lines": [_line("个人基本信息", [10, 20, 200, 60])],
            },
        ]
        geometries = {
            10: SimpleNamespace(
                source_page=1,
                source_crop_bbox=(0.0, 0.0, 300.0, 800.0),
                transform_usable=True,
                selected_rotation=0,
            ),
            20: SimpleNamespace(
                source_page=second_source,
                source_crop_bbox=second_crop,
                transform_usable=True,
                selected_rotation=0,
            ),
        }

        projection = _assembler(
            result,
            evidence,
            topology=SimpleNamespace(geometry=lambda logical: geometries[logical]),
            owner=owner,
        ).build()

        assert projection.pages == ()
        assert projection.unresolved_pages == (10, 20)
        assert owner._personal_detail_extraction_issues[0]["observed_value"][
            "geometry_failure"
        ] == failure


def test_detached_fragment_provenance_cannot_override_immutable_topology() -> None:
    cases = (
        (
            2,
            (300.0, 0.0, 600.0, 800.0),
            "source_page_provenance_mismatch",
        ),
        (
            1,
            (320.0, 0.0, 620.0, 800.0),
            "source_crop_provenance_mismatch",
        ),
    )
    for topology_source, topology_crop, failure in cases:
        owner = SimpleNamespace(
            reading_order_resolution={
                "resolved": True,
                "authoritative": True,
                "identity_fallback": False,
                "printed_total": 4,
                "printed_page_by_logical": {10: 1},
            }
        )
        result = SimpleNamespace(
            pages=[_page(10, source=1, width=300), _page(20, source=topology_source, width=300)]
        )
        evidence = [
            {
                "page": 10,
                "source_page": 1,
                "page_width": 300,
                "page_height": 800,
                "source_crop_bbox": [0, 0, 300, 800],
                "lines": [
                    _line("个人基本信息", [10, 20, 280, 60]),
                    _line("第 1 页，共 4 页", [100, 760, 250, 785]),
                ],
            },
            {
                "page": 20,
                "source_page": 1,
                "page_width": 300,
                "page_height": 800,
                "source_crop_bbox": [300, 0, 600, 800],
                "plugin_static_subpage": True,
                "lines": [_line("个人基本信息", [10, 20, 200, 60])],
            },
        ]
        geometries = {
            10: SimpleNamespace(
                source_page=1,
                source_crop_bbox=(0.0, 0.0, 300.0, 800.0),
                transform_usable=True,
                selected_rotation=0,
            ),
            20: SimpleNamespace(
                source_page=topology_source,
                source_crop_bbox=topology_crop,
                transform_usable=True,
                selected_rotation=0,
            ),
        }

        projection = _assembler(
            result,
            evidence,
            topology=SimpleNamespace(geometry=lambda logical: geometries[logical]),
            owner=owner,
        ).build()

        assert projection.pages == ()
        assert projection.unresolved_pages == (10, 20)
        assert owner._personal_detail_extraction_issues[0]["observed_value"][
            "geometry_failure"
        ] == failure


def test_explicit_summary_heading_outranks_repeated_account_category_names() -> None:
    result = SimpleNamespace(pages=[_page(1, source=1)])
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("（二）信息概要", [10, 10, 200, 40]),
                _line("非循环贷账户 循环贷账户一 贷记卡账户 账户数 余额 授信总额", [10, 80, 580, 130]),
            ],
        }
    ]

    projection = _assembler(result, evidence, {}).build()

    assert projection.registrations[0]["template_id"] == "information_summary"
    assert projection.registrations[0]["confidence"] >= 0.92


def _information_summary_continuation_case(
    defect: str = "",
) -> tuple[SimpleNamespace, list[dict[str, object]]]:
    first_rows = [[f"概要指标{index}", str(index)] for index in range(1, 25)]
    second_rows = [[f"概要指标{index}", str(index)] for index in range(25, 48)]
    first_rows[0][0] = "逾期(透支)信息汇总"
    second_rows[0][0] = "最近1个月内的查询机构数和查询次数"
    first_table = SimpleNamespace(
        table_id="summary:违约",
        bbox=[30.0, 100.0, 570.0, 285.0],
        headers=[],
        rows=[],
        metadata={"raw_rows": first_rows},
    )
    second_table = SimpleNamespace(
        table_id="summary:查询",
        bbox=[30.0, 360.0, 570.0, 570.0],
        headers=[],
        rows=[],
        metadata={"raw_rows": second_rows},
    )
    tables = [first_table, second_table]
    if defect == "extra_unowned_table":
        tables.append(
            SimpleNamespace(
                table_id="unowned:third",
                bbox=[30.0, 610.0, 570.0, 680.0],
                headers=[],
                rows=[],
                metadata={"raw_rows": [["任意字段", "任意值"]]},
            )
        )
    elif defect == "extra_unheaded_summary_table":
        tables.append(
            SimpleNamespace(
                table_id="unheaded:summary-shaped",
                bbox=[30.0, 610.0, 570.0, 680.0],
                headers=[],
                rows=[],
                metadata={
                    "raw_rows": [
                        ["最近1个月内的查询机构数", "3"],
                        ["最近1个月内的查询次数", "8"],
                    ]
                },
            )
        )
    elif defect == "independently_owned_table":
        tables.append(
            _exact_mixed_schema_table(
                "owned:inquiry",
                top=620.0,
                header=("查询原因", "查询机构", "编号", "查询日期"),
                values=("贷后管理", "机构乙", "1", "2025.04.03"),
            )
        )
    pages = [
        _page(2, source=1),
        _page(3, source=2, tables=tables),
    ]
    evidence: list[dict[str, object]] = [
        {
            "page": 2,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("二信息概要", [210.0, 620.0, 330.0, 640.0]),
                _line("第2页,共31页", [220.0, 760.0, 380.0, 785.0]),
            ],
        },
        {
            "page": 3,
            "source_page": 2,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line(
                    "(二)信贷交易违约信息概要",
                    [170.0, 80.0, 430.0, 100.0],
                ),
                # This exact account-family atom is legitimate summary-table
                # content and reproduces the live false whole-page role.
                _line("贷记卡账户", [80.0, 130.0, 190.0, 150.0]),
                _line(
                    "(四)查询记录概要",
                    [190.0, 340.0, 410.0, 360.0],
                ),
                _line("第3页,共31页", [220.0, 760.0, 380.0, 785.0]),
            ],
        },
    ]
    if defect == "one_current_heading":
        evidence[1]["lines"] = [line for line in evidence[1]["lines"] if line["text"] != "(四)查询记录概要"]
    elif defect == "nonconsecutive_footer":
        evidence[1]["lines"][-1]["text"] = "第4页,共31页"
    elif defect == "conflicting_top_level":
        evidence[1]["lines"].append(_line("公共信息明细", [180.0, 590.0, 420.0, 610.0]))
    elif defect == "numbered_account_card":
        evidence[1]["lines"].append(_line("账户12", [80.0, 180.0, 160.0, 200.0]))
    elif defect == "account_family_outside_table":
        evidence[1]["lines"][1]["bbox"] = [80.0, 305.0, 190.0, 325.0]
    elif defect == "independently_owned_table":
        evidence[1]["lines"].insert(
            -1,
            _line("(十一)查询记录", [180.0, 595.0, 360.0, 620.0]),
        )
    return SimpleNamespace(pages=pages), evidence


def test_sealed_summary_continuation_preserves_all_47_metric_shaped_rows() -> None:
    result, evidence = _information_summary_continuation_case()

    projection = _assembler(result, evidence).build()

    registration = next(row for row in projection.registrations if row["logical_page"] == 3)
    assert registration["template_id"] == "information_summary"
    assert "multiple_exact_summary_subsection_headings" in registration["signals"]
    current = next(page for page in projection.pages if page.page_number == 3)
    assert current.canonical_template_id == "information_summary"
    assert sum(len(table.metadata["raw_rows"]) for table in current.tables) == 47
    assert all(table.metadata["canonical_template_id"] == "information_summary" for table in current.tables)


def test_summary_continuation_fails_closed_without_every_local_proof() -> None:
    for defect in (
        "one_current_heading",
        "nonconsecutive_footer",
        "conflicting_top_level",
        "account_family_outside_table",
        "extra_unowned_table",
        "extra_unheaded_summary_table",
        "independently_owned_table",
        "non_authoritative_identity",
        "numbered_account_card",
    ):
        result, evidence = _information_summary_continuation_case(defect)
        owner = (
            SimpleNamespace(
                reading_order_resolution={
                    "resolved": False,
                    "authoritative": False,
                    "identity_fallback": True,
                    "printed_total": 31,
                    "printed_page_by_logical": {2: 2, 3: 3},
                }
            )
            if defect == "non_authoritative_identity"
            else None
        )

        projection = _assembler(result, evidence, owner=owner).build()

        registration = next(row for row in projection.registrations if row["logical_page"] == 3)
        assert registration["template_id"] != "information_summary", defect


def test_summary_page_with_independently_owned_table_stays_mixed() -> None:
    result, evidence = _information_summary_continuation_case(
        "independently_owned_table"
    )

    projection = _assembler(result, evidence).build()

    registration = next(row for row in projection.registrations if row["logical_page"] == 3)
    assert registration["template_id"] == "mixed_pboc_sections"
    current = next(page for page in projection.pages if page.page_number == 3)
    assert [table.table_id for table in current.tables] == ["owned:inquiry"]


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


def test_explicit_table_continuation_cannot_override_sealed_current_section_headings() -> None:
    summary_table = SimpleNamespace(
        table_id="summary:left",
        metadata={"raw_rows": [["A", "B"]]},
        bbox=[10, 80, 590, 300],
    )
    foreign_table = SimpleNamespace(
        table_id="mixed:right",
        metadata={"raw_rows": [["C", "D"]]},
        bbox=[10, 160, 590, 300],
    )
    later_table = SimpleNamespace(
        table_id="summary:later",
        metadata={"raw_rows": [["E", "F"]]},
        bbox=[10, 80, 590, 300],
    )
    result = SimpleNamespace(
        pages=[
            _page(1, source=1, tables=[summary_table]),
            _page(2, source=2, tables=[foreign_table]),
            _page(3, source=3, tables=[later_table]),
        ]
    )
    evidence = [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("信息概要", [20, 20, 220, 50])],
        },
        {
            "page": 2,
            "source_page": 2,
            "page_width": 600,
            "page_height": 800,
            "lines": [
                _line("公共信息明细", [20, 20, 220, 50]),
                _line("查询记录明细", [20, 80, 220, 110]),
            ],
        },
        {
            "page": 3,
            "source_page": 3,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("无标题的后续表格", [20, 20, 400, 50])],
        },
    ]
    owner = SimpleNamespace(
        tables_continue=lambda left, right: (left, right)
        in {
            ("summary:left", "mixed:right"),
            ("summary:left", "summary:later"),
        }
    )

    projection = _assembler(result, evidence, owner=owner).build()

    assert projection.registrations[1]["template_id"] == "unresolved"
    assert projection.registrations[1]["basis"] == "canonical_registration_exhausted"
    assert projection.registrations[2]["template_id"] == "unresolved"
    assert projection.unresolved_pages == (2, 3)


def _authoritative_account_flow_owner(
    *,
    authoritative: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        reading_order_resolution={
            "resolved": authoritative,
            "authoritative": authoritative,
            "identity_fallback": not authoritative,
            "printed_total": 40,
            "printed_page_by_logical": {41: 8, 77: 11},
        }
    )


def _dense_pboc_card_table() -> SimpleNamespace:
    columns = 8
    left = 10.0
    width = (590.0 - left) / columns
    cell_bboxes = [
        [
            [left + column * width, top, left + (column + 1) * width, bottom]
            for column in range(columns)
        ]
        for top, bottom in ((100.0, 130.0), (130.0, 170.0))
    ]
    cell_evidence_ids = [
        [[f"dense-card:r{row}:c{column}"] for column in range(columns)]
        for row in range(2)
    ]
    return SimpleNamespace(
        table_id="source-card-table",
        metadata={
            "raw_rows": [
                [
                    "发卡机构",
                    "账户标识",
                    "开立日期",
                    "账户授信额度",
                    "共享授信额度",
                    "币种",
                    "业务种类",
                    "担保方式",
                ],
                ["机构甲", "B123456789", "2024.01.02", "1", "1", "人民币", "贷记卡", "信用"],
            ],
            "cell_bboxes": cell_bboxes,
            "cell_evidence_ids": cell_evidence_ids,
            "cell_geometry_status": [["exact"] * columns for _row in range(2)],
        },
        bbox=[10, 100, 590, 240],
    )


def _account_flow_evidence(*, anchor: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "page": 41,
            "source_page": 41,
            "page_width": 600,
            "page_height": 800,
            "lines": [_line("（九）贷记卡账户", [20, 20, 220, 45])],
        },
        {
            "page": 77,
            "source_page": 77,
            "page_width": 600,
            "page_height": 800,
            "lines": [anchor],
        },
    ]


def test_authoritative_account_flow_adopts_geometry_bound_dense_card_fragment() -> None:
    result = SimpleNamespace(
        pages=[
            _page(41, source=41),
            _page(77, source=77, tables=[_dense_pboc_card_table()]),
        ]
    )
    evidence = _account_flow_evidence(anchor=_line("账户37", [20, 70, 180, 90]))

    projection = _assembler(
        result,
        evidence,
        owner=_authoritative_account_flow_owner(),
    ).build()

    assert [row["template_id"] for row in projection.registrations] == [
        "credit_account_detail",
        "credit_account_detail",
    ]
    assert projection.registrations[1]["basis"] == "canonical_flow_continuation"
    assert "authoritative_printed_account_order" in projection.registrations[1]["signals"]
    assert "anchor_to_table_geometry" in projection.registrations[1]["signals"]


def test_account_flow_requires_sealed_anchor_dense_table_and_authoritative_order() -> None:
    for defect in ("unsealed_anchor", "weak_table", "non_authoritative"):
        anchor = _line("账户37", [20, 70, 180, 90])
        if defect == "unsealed_anchor":
            anchor["evidence_ids"] = []
        table = _dense_pboc_card_table()
        if defect == "weak_table":
            table.metadata["raw_rows"][0] = ["发卡机构", "账户标识", "开立日期"]
            table.metadata["raw_rows"][1] = ["机构甲", "B123456789", "2024.01.02"]
        result = SimpleNamespace(
            pages=[
                _page(41, source=41),
                _page(77, source=77, tables=[table]),
            ]
        )

        projection = _assembler(
            result,
            _account_flow_evidence(anchor=anchor),
            owner=_authoritative_account_flow_owner(
                authoritative=defect != "non_authoritative"
            ),
        ).build()

        assert projection.registrations[1]["template_id"] == "unresolved"
        assert projection.registrations[1]["basis"] == "canonical_registration_exhausted"


def test_mixed_heading_vetoes_account_flow_inheritance() -> None:
    result = SimpleNamespace(
        pages=[
            _page(41, source=41),
            _page(77, source=77, tables=[_dense_pboc_card_table()]),
        ]
    )
    evidence = _account_flow_evidence(anchor=_line("账户37", [20, 70, 180, 90]))
    evidence[1]["lines"].append(_line("（八）授信协议信息", [20, 300, 260, 325]))

    projection = _assembler(
        result,
        evidence,
        owner=_authoritative_account_flow_owner(),
    ).build()

    assert projection.registrations[1]["template_id"] == "credit_agreement"
    assert "authoritative_printed_account_order" not in projection.registrations[1]["signals"]


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


def _exact_mixed_schema_table(
    table_id: str,
    *,
    top: float,
    header: tuple[str, ...],
    values: tuple[str, ...],
    widths: tuple[float, ...] | None = None,
) -> SimpleNamespace:
    assert len(header) == len(values)
    widths = widths or tuple(float(index + 2) for index in range(len(header)))
    assert len(widths) == len(header)
    left = 25.0
    scale = 500.0 / sum(widths)
    bands: list[tuple[float, float]] = []
    for width in widths:
        right = left + width * scale
        bands.append((left, right))
        left = right
    cell_bboxes = [
        [[x0, top, x1, top + 18.0] for x0, x1 in bands],
        [[x0, top + 18.0, x1, top + 38.0] for x0, x1 in bands],
    ]
    cell_evidence_ids = [
        [[f"{table_id}:header:{column}"] for column in range(len(header))],
        [[f"{table_id}:value:{column}"] for column in range(len(header))],
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[25.0, top, 525.0, top + 38.0],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": [list(header), list(values)],
            "geometry": {
                "cell_bboxes": cell_bboxes,
                "cell_evidence_ids": cell_evidence_ids,
                "cell_geometry_status": [
                    ["exact"] * len(header),
                    ["exact"] * len(header),
                ],
                "coordinate_system": "pdf_points_top_left",
            },
        },
    )


def _mixed_pboc_page_case() -> tuple[SimpleNamespace, list[dict[str, object]]]:
    postpaid_header = (
        "记账年月",
        "机构名称",
        "当前欠费金额",
        "业务开通日期",
        "业务类型",
        "当前缴费状态",
    )
    public_header = ("欠税统计日期", "编号", "欠税总额", "主管税务机关")
    inquiry_header = ("查询原因", "查询机构", "编号", "查询日期")
    tables = [
        _exact_mixed_schema_table(
            "mixed-postpaid",
            top=80.0,
            header=postpaid_header,
            values=("2026.04", "机构甲", "0", "2024.01.02", "固定电话后付费", "正常"),
            widths=(3.0, 8.0, 4.0, 6.0, 7.0, 5.0),
        ),
        _exact_mixed_schema_table(
            "mixed-public",
            top=240.0,
            header=public_header,
            values=("2025.03.01", "1", "100", "主管机关"),
            widths=(8.0, 2.0, 4.0, 9.0),
        ),
        _exact_mixed_schema_table(
            "mixed-inquiry",
            top=400.0,
            header=inquiry_header,
            values=("贷后管理", "机构乙", "1", "2025.04.03"),
            widths=(7.0, 9.0, 2.0, 6.0),
        ),
    ]
    result = SimpleNamespace(pages=[_page(9, source=4, width=550.0, height=700.0, tables=tables)])
    evidence: list[dict[str, object]] = [
        {
            "page": 9,
            "source_page": 4,
            "page_width": 550.0,
            "page_height": 700.0,
            "lines": [
                _line("（九）后付费记录", [180.0, 30.0, 360.0, 55.0]),
                _line("（三）欠税记录", [190.0, 190.0, 350.0, 215.0]),
                _line("（十一）查询记录", [180.0, 350.0, 360.0, 375.0]),
            ],
        }
    ]
    return result, evidence


def test_mixed_pboc_page_projects_only_exact_table_local_section_owners() -> None:
    result, evidence = _mixed_pboc_page_case()

    projection = _assembler(result, evidence).build()

    assert projection.unresolved_pages == ()
    assert len(projection.pages) == 1
    page = projection.pages[0]
    assert page.canonical_template_id == "mixed_pboc_sections"
    assert {
        table.table_id: table.metadata["canonical_template_id"]
        for table in page.tables
    } == {
        "mixed-postpaid": "postpaid_detail",
        "mixed-public": "public_information",
        "mixed-inquiry": "annotations_and_inquiries",
    }
    assert all(
        table.metadata["canonical_section_owner"]["binding"]
        == "exact_pboc_section_heading_and_table_schema"
        for table in page.tables
    )
    assert projection.registrations[0]["basis"] == (
        "exact_table_local_pboc_section_ownership"
    )


def test_mixed_pboc_owned_reordered_inquiry_header_extracts_by_semantic_roles() -> None:
    result, evidence = _mixed_pboc_page_case()
    inquiry = result.pages[0].tables[2]
    rows = inquiry.metadata["raw_rows"]
    geometry = inquiry.metadata["geometry"]
    original_boxes = geometry["cell_bboxes"]
    inquiry.metadata["raw_rows"] = [["", "", "", ""], rows[0], rows[1]]
    geometry.update(
        {
            "row_bands": [
                {"index": 0, "y0": 390.0, "y1": 400.0},
                {"index": 1, "y0": 400.0, "y1": 418.0},
                {"index": 2, "y0": 418.0, "y1": 438.0},
            ],
            "col_bands": [
                {"index": column, "x0": box[0], "x1": box[2]}
                for column, box in enumerate(original_boxes[0])
            ],
            "cell_bboxes": [
                [[box[0], 390.0, box[2], 400.0] for box in original_boxes[0]],
                original_boxes[0],
                original_boxes[1],
            ],
            "cell_geometry_status": [
                ["exact"] * 4,
                geometry["cell_geometry_status"][0],
                geometry["cell_geometry_status"][1],
            ],
            "cell_evidence_ids": [
                [[], [], [], []],
                geometry["cell_evidence_ids"][0],
                geometry["cell_evidence_ids"][1],
            ],
        }
    )
    evidence[0]["lines"].append(
        _line(
            "\u673a\u6784\u67e5\u8be2\u8bb0\u5f55\u660e\u7ec6",
            [180.0, 385.0, 360.0, 398.0],
        )
    )

    projection = _assembler(result, evidence).build()
    context = SimpleNamespace(
        pages=list(projection.pages),
        _personal_detail_extraction_issues=[],
    )

    assert projection.unresolved_pages == ()
    inquiry_rows = _extract_inquiries(context)
    assert [
        (
            row["sequence"],
            row["inquiry_date"],
            row["institution"],
            row["reason"],
        )
        for row in inquiry_rows
    ] == [(1, rows[1][3].replace(".", "-"), rows[1][1], rows[1][0])]


def _collapsed_inquiry_page(*, residue: str = ""):
    table = _exact_mixed_schema_table(
        "collapsed-inquiry",
        top=120.0,
        header=(f"编号查询日期{residue}", "", "查询机构", "查询原因"),
        values=("1", "2025.04.03", "机构乙", "贷后管理"),
        widths=(2.0, 6.0, 9.0, 7.0),
    )
    geometry = table.metadata["geometry"]
    first, covered = geometry["cell_bboxes"][0][:2]
    geometry["cell_bboxes"][0][0] = [first[0], first[1], covered[2], first[3]]
    geometry["cell_bboxes"][0][1] = None
    geometry["cell_geometry_status"][0][1] = "derived"
    geometry["cell_evidence_ids"][0][1] = []
    geometry["cell_spans"] = [
        {"row": 0, "col": 0, "row_span": 1, "col_span": 2}
    ]
    result = SimpleNamespace(
        pages=[_page(9, source=9, width=550.0, height=700.0, tables=[table])]
    )
    evidence = [
        {
            "page": 9,
            "source_page": 9,
            "page_width": 550.0,
            "page_height": 700.0,
            "lines": [
                _line("（十一）查询记录", [180.0, 70.0, 360.0, 95.0])
            ],
        }
    ]
    return result, evidence


def test_clean_collapsed_inquiry_header_gets_one_table_local_owner() -> None:
    result, evidence = _collapsed_inquiry_page()

    projection = _assembler(result, evidence).build()

    assert projection.unresolved_pages == ()
    projected = projection.pages[0].tables[0]
    assert projected.metadata["canonical_template_id"] == (
        "annotations_and_inquiries"
    )
    assert projected.metadata["canonical_section_owner"]["header_binding"] == (
        "exact_collapsed_colspan_header_lattice"
    )


def test_collapsed_inquiry_header_with_residual_business_text_fails_closed() -> None:
    result, evidence = _collapsed_inquiry_page(residue="X")

    projection = _assembler(result, evidence).build()

    assert projection.pages == ()
    assert projection.unresolved_pages == (9,)


def test_mixed_pboc_page_withholds_heading_schema_mismatch_and_unsealed_owner() -> None:
    result, evidence = _mixed_pboc_page_case()
    # The nearest heading above the postpaid table names another semantic role.
    evidence[0]["lines"][0] = _line("（九）查询记录", [180.0, 30.0, 360.0, 55.0])
    # The public heading is semantically right but lacks immutable ownership.
    evidence[0]["lines"][1]["evidence_ids"] = []

    projection = _assembler(result, evidence).build()

    # Only the final inquiry table remains locally proved.  Preserve that
    # bounded owner without allowing it to swallow the two unowned tables.
    assert projection.unresolved_pages == ()
    assert [table.table_id for table in projection.pages[0].tables] == [
        "mixed-inquiry"
    ]
    assert projection.pages[0].tables[0].metadata["canonical_template_id"] == (
        "annotations_and_inquiries"
    )


def test_mixed_pboc_page_rejects_cross_table_replayed_cell_owner() -> None:
    result, evidence = _mixed_pboc_page_case()
    postpaid, public, _inquiry = result.pages[0].tables
    public.metadata["geometry"]["cell_evidence_ids"][0][0] = list(
        postpaid.metadata["geometry"]["cell_evidence_ids"][0][0]
    )

    projection = _assembler(result, evidence).build()

    assert {
        table.table_id for table in projection.pages[0].tables
    } == {"mixed-postpaid", "mixed-inquiry"}


def test_inquiry_owner_cannot_skip_a_partial_first_population_row() -> None:
    result, evidence = _mixed_pboc_page_case()
    inquiry = result.pages[0].tables[2]
    rows = inquiry.metadata["raw_rows"]
    geometry = inquiry.metadata["geometry"]
    rows[1][2] = ""
    geometry["cell_evidence_ids"][1][2] = []
    second_boxes = [
        [box[0], box[1] + 20.0, box[2], box[3] + 20.0]
        for box in geometry["cell_bboxes"][1]
    ]
    rows.append(["贷后管理", "机构丙", "2", "2025.04.02"])
    geometry["cell_bboxes"].append(second_boxes)
    geometry["cell_geometry_status"].append(["exact"] * 4)
    geometry["cell_evidence_ids"].append(
        [[f"mixed-inquiry:second:{column}"] for column in range(4)]
    )
    inquiry.bbox[3] += 20.0

    projection = _assembler(result, evidence).build()

    assert {
        table.table_id for table in projection.pages[0].tables
    } == {"mixed-postpaid", "mixed-public"}


def test_mixed_pboc_contents_headings_without_exact_tables_do_not_own_page() -> None:
    result = SimpleNamespace(pages=[_page(5, source=5, tables=[])])
    evidence = [
        {
            "page": 5,
            "source_page": 5,
            "page_width": 600.0,
            "page_height": 800.0,
            "lines": [
                _line("非信贷交易信息明细", [30.0, 60.0, 260.0, 85.0]),
                _line("公共信息明细", [30.0, 100.0, 260.0, 125.0]),
                _line("查询记录明细", [30.0, 140.0, 260.0, 165.0]),
            ],
        }
    ]

    projection = _assembler(result, evidence).build()

    assert projection.pages == ()
    assert projection.unresolved_pages == (5,)


def _agreement_card_table(table_id: str, *, top: float) -> SimpleNamespace:
    return _exact_mixed_schema_table(
        table_id,
        top=top,
        header=("管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"),
        values=("机构甲", "A123456789", "2024.01.01", "长期", "贷款"),
        widths=(7.0, 9.0, 5.0, 5.0, 6.0),
    )


def _cross_page_agreement_case() -> tuple[SimpleNamespace, list[dict[str, object]]]:
    first_tables = [
        _agreement_card_table("agreement-1", top=110.0),
        _agreement_card_table("agreement-2", top=260.0),
    ]
    second_tables = [
        _agreement_card_table("agreement-3", top=35.0),
        _exact_mixed_schema_table(
            "postpaid-after-card",
            top=230.0,
            header=("机构名称", "业务类型", "业务开通日期", "当前缴费状态", "当前欠费金额", "记账年月"),
            values=("机构乙", "固定电话后付费", "2024.01.02", "正常", "0", "2025.01"),
        ),
        _exact_mixed_schema_table(
            "public-after-card",
            top=430.0,
            header=("编号", "主管税务机关", "欠税总额", "欠税统计日期"),
            values=("1", "主管机关", "100", "2025.02.01"),
        ),
    ]
    result = SimpleNamespace(
        pages=[
            _page(21, source=21, width=600.0, height=800.0, tables=first_tables),
            _page(37, source=37, width=600.0, height=800.0, tables=second_tables),
        ]
    )
    evidence: list[dict[str, object]] = [
        {
            "page": 21,
            "source_page": 21,
            "page_width": 600.0,
            "page_height": 800.0,
            "lines": [
                _line("（八）授信协议信息", [180.0, 30.0, 360.0, 55.0]),
                _line("授信协议1", [30.0, 80.0, 120.0, 100.0]),
                _line("授信协议2", [30.0, 230.0, 120.0, 250.0]),
                _line("授信协议3", [30.0, 710.0, 120.0, 730.0]),
                _line("第8页，共20页", [240.0, 760.0, 360.0, 790.0]),
            ],
        },
        {
            "page": 37,
            "source_page": 37,
            "page_width": 600.0,
            "page_height": 800.0,
            "lines": [
                _line("（九）非信贷交易信息明细", [170.0, 180.0, 380.0, 205.0]),
                _line("后付费记录", [200.0, 210.0, 340.0, 225.0]),
                _line("（十）公共信息明细", [180.0, 380.0, 360.0, 405.0]),
                _line("欠税记录", [210.0, 410.0, 330.0, 425.0]),
                _line("第9页，共20页", [240.0, 760.0, 360.0, 790.0]),
            ],
        },
    ]
    return result, evidence


def test_cross_page_agreement_card_requires_reciprocal_terminal_anchor() -> None:
    result, evidence = _cross_page_agreement_case()

    projection = _assembler(result, evidence).build()

    second = next(page for page in projection.pages if page.page_number == 37)
    assert [(table.table_id, table.metadata["canonical_template_id"]) for table in second.tables] == [
        ("agreement-3", "credit_agreement"),
        ("postpaid-after-card", "postpaid_detail"),
        ("public-after-card", "public_information"),
    ]
    agreement_owner = second.tables[0].metadata["canonical_section_owner"]
    assert agreement_owner["printed_sequence"] == 3
    assert agreement_owner["binding"] == (
        "terminal_prior_page_agreement_anchor_and_leading_exact_table"
    )


def test_mixed_table_roles_are_local_and_cross_page_agreement_is_consumable() -> None:
    result, evidence = _cross_page_agreement_case()
    projection = _assembler(result, evidence).build()
    context = SimpleNamespace(
        pages=list(projection.pages),
        reading_order_by_logical={21: 1, 37: 2},
        tables_continue=lambda _left, _right: None,
        _personal_detail_extraction_issues=[],
    )
    mixed_page = next(page for page in context.pages if page.page_number == 37)

    assert [canonical_table_role(context, mixed_page, table) for table in mixed_page.tables] == [
        "credit_agreement",
        "postpaid_detail",
        "public_information",
    ]
    records = PBOCPersonalDetailNativeParser(context).records("credit_lines")
    assert [record.fields["__printed_sequence"] for record in records] == ["1", "2", "3"]


def test_mixed_table_role_rejects_missing_forged_and_duplicate_owner_metadata() -> None:
    mutations = (
        lambda table: table.metadata.pop("canonical_section_owner"),
        lambda table: table.metadata["canonical_section_owner"].update(
            {"table_id": "another-table"}
        ),
        lambda table: table.metadata["canonical_section_owner"].update(
            {
                "heading_evidence_ids": [
                    *table.metadata["canonical_section_owner"]["heading_evidence_ids"],
                    *table.metadata["canonical_section_owner"]["heading_evidence_ids"],
                ]
            }
        ),
    )
    for mutate in mutations:
        result, evidence = _cross_page_agreement_case()
        projection = _assembler(result, evidence).build()
        context = SimpleNamespace(
            pages=list(projection.pages),
            reading_order_by_logical={21: 1, 37: 2},
            tables_continue=lambda _left, _right: None,
            _personal_detail_extraction_issues=[],
        )
        mixed_page = next(page for page in context.pages if page.page_number == 37)
        agreement = mixed_page.tables[0]
        mutate(agreement)

        assert canonical_table_role(context, mixed_page, agreement) is None
        assert len(PBOCPersonalDetailNativeParser(context).records("credit_lines")) == 2


def test_cross_page_agreement_plus_one_local_role_establishes_mixed_page() -> None:
    result, evidence = _cross_page_agreement_case()
    result.pages[1].tables = result.pages[1].tables[:2]
    evidence[1]["lines"] = [
        line
        for line in evidence[1]["lines"]
        if "鍏叡淇℃伅" not in str(line.get("text") or "")
        and "娆犵◣璁板綍" not in str(line.get("text") or "")
    ]

    projection = _assembler(result, evidence).build()

    second = next(page for page in projection.pages if page.page_number == 37)
    assert [(table.table_id, table.metadata["canonical_template_id"]) for table in second.tables] == [
        ("agreement-3", "credit_agreement"),
        ("postpaid-after-card", "postpaid_detail"),
    ]


def test_cross_page_agreement_card_fails_closed_without_dense_or_consecutive_proof() -> None:
    for defect in (
        "ordinal_gap",
        "printed_page_gap",
        "intervening_business_line",
        "second_leading_table",
        "replayed_anchor_evidence",
        "replayed_table_evidence",
    ):
        result, evidence = _cross_page_agreement_case()
        if defect == "ordinal_gap":
            evidence[0]["lines"][2]["text"] = "授信协议8"
        elif defect == "printed_page_gap":
            evidence[1]["lines"][-1]["text"] = "第11页，共20页"
        elif defect == "intervening_business_line":
            evidence[0]["lines"].insert(-1, _line("另一业务段", [30.0, 735.0, 150.0, 750.0]))
        elif defect == "second_leading_table":
            result.pages[1].tables.insert(
                1,
                _agreement_card_table("competing-leading-card", top=90.0),
            )
        elif defect == "replayed_anchor_evidence":
            evidence[0]["lines"][3]["evidence_ids"] = list(
                evidence[0]["lines"][2]["evidence_ids"]
            )
        else:
            result.pages[0].tables[1].metadata["geometry"]["cell_evidence_ids"][0][0] = list(
                result.pages[0].tables[0].metadata["geometry"]["cell_evidence_ids"][0][0]
            )

        projection = _assembler(result, evidence).build()

        second = next(page for page in projection.pages if page.page_number == 37)
        assert all(table.table_id != "agreement-3" for table in second.tables)


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


def test_monthly_gaps_between_printed_positions_do_not_imply_missing_months(
    monkeypatch,
) -> None:
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
    assert not any(
        issue.get("issue_code") == "canonical_monthly_reconstruction_incomplete"
        for issue in getattr(context, "_personal_detail_extraction_issues", [])
    )


def test_monthly_detached_source_diff_reports_exact_missing_grid_month_fields(
    monkeypatch,
) -> None:
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {1: 1}
    context.reading_order_resolution = {
        "resolved": False,
        "authoritative": False,
        "basis": "single_page_test",
    }
    context._page_image_resolver = None
    context.parse_result = SimpleNamespace(
        pages=[],
        entities=SimpleNamespace(
            domain_specific={"_page_evidence_bundles": [{"page": 1}]}
        ),
    )


    context.corrected_evidence_pages = lambda: [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [],
        }
    ]
    canonical_records = [
        {
            "repayment_id": "grid:1:2024-01",
            "grid_id": "grid:1",
            "performance_month": "2024-01",
            "year": 2024,
            "month": 1,
            "status": "N",
            "source_cell_refs": [
                {
                    "logical_page": 1,
                    "grid_id": "grid:1",
                    "col": 1,
                    "field_name": "status",
                    "bbox": [10, 10, 20, 20],
                    "geometry_scope": "cell",
                }
            ],
        }
    ]
    source_records = [
        *canonical_records,
        {
            "repayment_id": "grid:1:2024-02",
            "grid_id": "grid:1",
            "performance_month": "2024-02",
            "year": 2024,
            "month": 2,
            "status": "unknown",
            "source_cell_refs": [
                {
                    "logical_page": 1,
                    "grid_id": "grid:1",
                    "col": 2,
                    "field_name": "status",
                    "bbox": [20, 10, 30, 20],
                    "geometry_scope": "cell",
                }
            ],
        },
    ]
    grid = {
        "grid_id": "grid:1",
        "page": 1,
        "bbox": [10, 10, 130, 40],
        "audit": {
            "date_range": {
                "start_year": 2024,
                "start_month": 1,
                "end_year": 2024,
                "end_month": 2,
            }
        },
    }
    calls = 0

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize."
        "materialize_credit_repayment_micro_grids_from_bundles",
        lambda *_args, **_kwargs: None,
    )

    def grids_for_domain(_domain):
        nonlocal calls
        calls += 1
        return [{**grid, "_records": canonical_records if calls == 1 else source_records}]

    monkeypatch.setattr(
        "docmirror.models.mirror.domain_access."
        "micro_grid_structures_from_domain_specific",
        grids_for_domain,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.repayment_grid.records_from_micro_grid_dict",
        lambda current_grid, **_kwargs: list(current_grid["_records"]),
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.repayment_grid.dedupe_repayment_records",
        lambda values: list(values),
    )

    assert context.corrected_repayment_records() == canonical_records
    aggregate = next(
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"] == "canonical_monthly_reconstruction_incomplete"
    )
    assert aggregate["observed_value"] == {"canonical_row_count": 1}
    assert aggregate["candidate_value"] == {
        "source_structure_row_count": 2,
        "unreconciled_source_position_count": 1,
        "account_month_expected_row_count": None,
        "localization_status": "pending_unique_account_owner_reconciliation",
    }
    assert {
        "source_structure_is_audit_only",
        "raw_grid_positions_not_a_population_denominator",
        "printed_ranges_do_not_imply_intervening_months",
    }.issubset(set(aggregate["reason_codes"]))
    assert "structural_expected_row_count" not in aggregate["candidate_value"]
    assert "missing_month_count" not in aggregate["candidate_value"]
    field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue["issue_code"]
        == "canonical_monthly_source_structure_missing_field"
    ]
    assert {
        (issue["target_record_id"], issue["field_name"]) for issue in field_issues
    } == {
        ("grid:1:2024-02", "performance_month"),
        ("grid:1:2024-02", "status_code"),
    }
    assert not any(issue["field_name"] == "status_amount" for issue in field_issues)
    assert all(issue["source_refs"] for issue in field_issues)
    assert all(
        any(ref.get("geometry_scope") == "cell" for ref in issue["source_refs"])
        for issue in field_issues
    )
    assert all(
        "owner_or_status_value_not_invented" in issue["reason_codes"]
        for issue in field_issues
    )
    assert len(
        {
            (issue["target_record_id"], issue["field_name"])
            for issue in field_issues
        }
    ) == len(field_issues)


def test_monthly_detached_source_diff_localizes_every_grid_only_position(
    monkeypatch,
) -> None:
    context = object.__new__(PersonalDetailExtractionContext)
    context._cache = {}
    context.reading_order_by_logical = {1: 1}
    context.reading_order_resolution = {
        "resolved": False,
        "authoritative": False,
        "basis": "single_page_test",
    }
    context._page_image_resolver = None
    context.parse_result = SimpleNamespace(
        pages=[],
        entities=SimpleNamespace(
            domain_specific={"_page_evidence_bundles": [{"page": 1}]}
        ),
    )
    context.corrected_evidence_pages = lambda: [
        {
            "page": 1,
            "source_page": 1,
            "page_width": 600,
            "page_height": 800,
            "lines": [],
        }
    ]
    canonical_record = {
        "repayment_id": "grid:1:2024-01",
        "grid_id": "grid:1",
        "performance_month": "2024-01",
        "year": 2024,
        "month": 1,
        "status": "N",
    }
    grid_only_source_record = {
        "repayment_id": "grid:1:2024-02",
        "grid_id": "grid:1",
        "performance_month": "2024-02",
        "year": 2024,
        "month": 2,
        "status": "unknown",
        "source_cell_refs": [
            {
                "logical_page": 1,
                "grid_id": "grid:1",
                "field_name": "status",
                "bbox": [10, 10, 130, 40],
                "geometry_scope": "grid",
            }
        ],
    }
    grid = {"grid_id": "grid:1", "page": 1, "bbox": [10, 10, 130, 40]}
    calls = 0

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.micro_grid_materialize."
        "materialize_credit_repayment_micro_grids_from_bundles",
        lambda *_args, **_kwargs: None,
    )

    def grids_for_domain(_domain):
        nonlocal calls
        calls += 1
        records = (
            [canonical_record]
            if calls == 1
            else [canonical_record, grid_only_source_record]
        )
        return [{**grid, "_records": records}]

    monkeypatch.setattr(
        "docmirror.models.mirror.domain_access."
        "micro_grid_structures_from_domain_specific",
        grids_for_domain,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.repayment_grid.records_from_micro_grid_dict",
        lambda current_grid, **_kwargs: list(current_grid["_records"]),
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.repayment_grid.dedupe_repayment_records",
        lambda values: list(values),
    )

    assert context.corrected_repayment_records() == [canonical_record]
    issues = context._personal_detail_extraction_issues
    assert any(
        issue["issue_code"] == "canonical_monthly_reconstruction_incomplete"
        for issue in issues
    )
    localized = [
        issue
        for issue in issues
        if issue["issue_code"] == "canonical_monthly_source_structure_missing_field"
    ]
    assert {issue["field_name"] for issue in localized} == {
        "performance_month",
        "status_code",
    }
    assert all(
        issue["target_record_id"] == "grid:1:2024-02"
        and issue["observed_value"]["grid_id"] == "grid:1"
        and issue["observed_value"]["performance_month"] == "2024-02"
        and "account_id" not in issue["observed_value"]
        and issue["source_refs"]
        and issue["source_refs"][0]["grid_id"] == "grid:1"
        and issue["source_refs"][0]["performance_month"] == "2024-02"
        and "exact_grid_month_source_position" in issue["reason_codes"]
        for issue in localized
    )


def test_page_orientation_score_prefers_horizontal_rows_on_portrait_page() -> None:
    horizontal = [{"text": "89 2022.05.22 示例机构 贷款审批", "bbox": [10, 10, 400, 28], "confidence": 0.99}]
    vertical = [{"text": "89 2022.05.22 示例机构 贷款审批", "bbox": [10, 10, 28, 400], "confidence": 0.99}]

    assert _page_ocr_score(horizontal, image_shape=(800, 600, 3)) > _page_ocr_score(
        vertical,
        image_shape=(800, 600, 3),
    )


def test_inquiry_schema_withholds_geometry_free_headerless_page_and_merged_date_cell() -> None:
    def table(table_id: str, rows: list[list[str]]) -> SimpleNamespace:
        return SimpleNamespace(
            table_id=table_id,
            metadata={
                "raw_rows": rows,
                "canonical_template_id": "annotations_and_inquiries",
            },
            bbox=[10, 10, 590, 700],
        )

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
    assert [row["sequence"] for row in institutional] == [1]
    assert all(row["sequence"] != 2 for row in rows)
    assert [row["sequence"] for row in personal] == [1]
    # The malformed row has no exact table owner, so it cannot authorize a
    # business row or a field-local issue.  Local reporting starts only after
    # source ownership has been established.
    assert all(row["sequence"] != 3 for row in rows)


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
            "canonical_template_id": "annotations_and_inquiries",
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
    assert final_rows[0]["institution"] is None
    assert final_rows[0]["reason"] is None
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_field_conflict"
        and issue.get("field_name") == "institution"
        and "conflicting_values_withheld" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )
    assert any(
        issue.get("issue_code") == "candidate_b_inquiry_field_conflict"
        and issue.get("field_name") == "reason"
        and "conflicting_values_withheld" in issue.get("reason_codes", ())
        for issue in context._personal_detail_extraction_issues
    )


def test_inquiry_sequence_prefix_suppression_preserves_valid_101() -> None:
    table = SimpleNamespace(
        table_id="inquiries",
        bbox=[10, 5, 590, 80],
        metadata={
            "canonical_template_id": "annotations_and_inquiries",
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
                "canonical_template_id": "annotations_and_inquiries",
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


def test_projected_canonical_table_preserves_exact_atomic_cell_provenance() -> None:
    cell = SimpleNamespace(
        text="中国(含港澳台)",
        bbox=[10.0, 20.0, 60.0, 35.0],
        evidence_ids=["ocr:sp0001:lp0001:0051"],
        token_ids=["ocr:sp0001:lp0001:0051"],
        geometry_status="exact",
        geometry_confidence=0.97,
    )
    table = SimpleNamespace(
        table_id="profile-source",
        headers=[],
        rows=[SimpleNamespace(cells=[cell])],
        metadata={},
        bbox=[10.0, 20.0, 60.0, 35.0],
        confidence=0.9,
    )

    projected = _project_table(
        table,
        template_id="report_header_and_identity",
        transform=lambda box: [float(value) + 100.0 for value in box],
    )

    assert projected.metadata["raw_rows"] == [["中国(含港澳台)"]]
    assert projected.metadata["source_cell_bboxes"] == [[[10.0, 20.0, 60.0, 35.0]]]
    assert projected.metadata["cell_bboxes"] == [[[110.0, 120.0, 160.0, 135.0]]]
    assert projected.metadata["cell_evidence_ids"] == [[[
        "ocr:sp0001:lp0001:0051"
    ]]]
    assert projected.metadata["cell_geometry_status"] == [["exact"]]
    assert projected.metadata["cell_geometry_confidences"] == [[0.97]]
    assert "source_cell_objects" not in projected.metadata
    assert projected.source_cell_objects == [[cell]]
