"""Fast, production-shaped replay for the Lin personal-detail discovery run.

Provenance
----------
The immutable source excerpts in this module were transcribed from
``artifacts/personal_detail_six_live_iteration_20260826_linfix4/林岚挺征信.semantic.json``
(source PDF sha256
``a44515a83ae226d19008437ac6a757fa58dabc14d3f1fb5ac9a01c4441cdfdd2``).
They intentionally preserve the physical page/table ids, printed headings,
OCR residue, geometry, and evidence-id namespaces that exposed the production
failures.  No OCR or PDF access is needed to replay the contracts below.

This is deliberately an integration corpus, not a new extraction strategy.
It enters through the current materializer/assemblers and freezes the desired
population, ownership, field, unit, and fail-closed behavior.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    CANDIDATE_B_STAGE_REGISTRY,
    stage_names_for_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id

LIN_ARTIFACT = (
    "artifacts/personal_detail_six_live_iteration_20260826_linfix4/"
    "林岚挺征信.semantic.json"
)
LIN_SOURCE_SHA256 = "a44515a83ae226d19008437ac6a757fa58dabc14d3f1fb5ac9a01c4441cdfdd2"


def _text(
    content: str,
    bbox: tuple[float, float, float, float],
    evidence_id: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        text=content,
        bbox=list(bbox),
        evidence_ids=[evidence_id],
        confidence=1.0,
    )


def _table(
    table_id: str,
    rows: list[list[str]],
    *,
    logical_page: int,
    source_page: int,
    bbox: tuple[float, float, float, float],
    evidence_start: int,
) -> SimpleNamespace:
    """Recreate the native table contract, including exact cell provenance."""

    left, top, right, bottom = bbox
    row_height = (bottom - top) / max(len(rows), 1)
    cell_bboxes: list[list[list[float]]] = []
    cell_evidence_ids: list[list[list[str]]] = []
    evidence_number = evidence_start
    for row_index, row in enumerate(rows):
        column_width = (right - left) / max(len(row), 1)
        row_boxes: list[list[float]] = []
        row_evidence: list[list[str]] = []
        for column_index, _value in enumerate(row):
            row_boxes.append(
                [
                    left + column_index * column_width,
                    top + row_index * row_height,
                    left + (column_index + 1) * column_width,
                    top + (row_index + 1) * row_height,
                ]
            )
            row_evidence.append(
                [
                    f"ocr:sp{source_page:04d}:lp{logical_page:04d}:"
                    f"{evidence_number:04d}"
                ]
            )
            evidence_number += 1
        cell_bboxes.append(row_boxes)
        cell_evidence_ids.append(row_evidence)
    metadata = {
        "raw_rows": deepcopy(rows),
        "cell_bboxes": deepcopy(cell_bboxes),
        "source_cell_bboxes": deepcopy(cell_bboxes),
        "cell_evidence_ids": deepcopy(cell_evidence_ids),
        "source_cell_evidence_ids": deepcopy(cell_evidence_ids),
        "cell_geometry_status": [
            ["exact" for _value in row]
            for row in rows
        ],
        "source_logical_page": logical_page,
        "source_page": source_page,
    }
    return SimpleNamespace(
        table_id=table_id,
        id=table_id,
        bbox=list(bbox),
        headers=[],
        rows=[],
        page_number=logical_page,
        logical_page=logical_page,
        source_page_number=source_page,
        source_page=source_page,
        metadata=metadata,
        geometry={
            "cell_bboxes": deepcopy(cell_bboxes),
            "cell_evidence_ids": deepcopy(cell_evidence_ids),
            "cell_geometry_status": deepcopy(metadata["cell_geometry_status"]),
        },
    )


def _page(
    logical_page: int,
    source_page: int,
    *,
    texts: Iterable[SimpleNamespace] = (),
    tables: Iterable[SimpleNamespace] = (),
    canonical_template_id: str = "mixed_pboc_sections",
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=logical_page,
        logical_page=logical_page,
        source_page_number=source_page,
        source_page=source_page,
        texts=list(texts),
        tables=list(tables),
        canonical_template_id=canonical_template_id,
        metadata={"printed_page_number": logical_page, "printed_page_total": 30},
    )


def _context(
    pages: Iterable[SimpleNamespace],
    *,
    corrected_pages: Iterable[Mapping[str, Any]] = (),
) -> SimpleNamespace:
    frozen = list(pages)
    order = {
        int(page.page_number): position
        for position, page in enumerate(frozen, start=1)
    }
    return SimpleNamespace(
        pages=frozen,
        _frozen_logical_pages={int(page.page_number): page for page in frozen},
        reading_order_by_logical=order,
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
            "basis": "printed_page_identity",
        },
        corrected_evidence_pages=lambda: deepcopy(list(corrected_pages)),
        allows_scanned_line_transition=lambda *_args, **_kwargs: True,
        tables_continue=lambda *_args, **_kwargs: False,
        _personal_detail_extraction_issues=[],
    )


def _activate_field_repair(context: Any, plan: Any) -> None:
    context._business_repair_plan = plan
    context._business_repair_active = True
    context._personal_detail_extraction_issues = []
    context._ocr_correction_overlay = PersonalDetailOCRCorrectionOverlay(context)
    context.candidate_b_planned_field_repair = (
        PersonalDetailExtractionContext.candidate_b_planned_field_repair.__get__(
            context,
            type(context),
        )
    )
    context.candidate_b_field_repair = (
        PersonalDetailExtractionContext.candidate_b_field_repair.__get__(
            context,
            type(context),
        )
    )
    context._ocr_correction_overlay.install_business_repair_evidence(
        plan.page_evidence.values(),
        affected_pages=plan.affected_pages,
        allowed_target_refs=(
            {**dict(ref), "field_name": repair.field_name}
            for repair in plan.field_repairs
            for ref in repair.source_refs
        ),
    )


def _resolve_with_simulated_page_ocr(
    coordinator: BusinessUncertaintyRepairCoordinator,
    plan: Any,
    *,
    source_pages: Iterable[Any],
    candidate_for: Callable[[Any], str | None],
) -> list[tuple[set[int], str]]:
    calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(pages: set[int], *, reason: str) -> list[dict[str, Any]]:
        calls.append((set(pages), reason))
        acquired: list[dict[str, Any]] = []
        for logical_page in sorted(pages):
            repairs = [
                repair
                for repair in plan.field_repairs
                if repair.mode == "context_rich_reocr"
                and any(
                    int(ref.get("logical_page") or 0) == logical_page
                    for ref in repair.source_refs
                )
            ]
            assert repairs
            page_key = f"lin-business-{logical_page}"
            source_page = int(repairs[0].source_refs[0]["source_page"])
            lines: list[dict[str, Any]] = [
                {
                    "text": "征信业务明细",
                    "content": "征信业务明细",
                    "confidence": 0.99,
                    "bbox": [1.0, 1.0, 20.0, 10.0],
                    "evidence_ids": [
                        f"personal_detail_page_reocr:{page_key}:w0"
                    ],
                    "source": "personal_detail_page_reocr_once",
                }
            ]
            for repair in repairs:
                candidate = candidate_for(repair)
                if candidate is None:
                    continue
                ref = repair.source_refs[0]
                word_index = len(lines)
                lines.append(
                    {
                        "text": candidate,
                        "content": candidate,
                        "confidence": 0.99,
                        "bbox": list(ref["bbox"]),
                        "evidence_ids": [
                            f"personal_detail_page_reocr:{page_key}:w{word_index}"
                        ],
                        "source": "personal_detail_page_reocr_once",
                    }
                )
            page: dict[str, Any] = {
                "page": logical_page,
                "logical_page": logical_page,
                "source_page": source_page,
                "page_key": page_key,
                "lines": lines,
            }
            coordinate_system = str(
                repairs[0].source_refs[0].get("coordinate_system") or ""
            )
            if coordinate_system:
                page["coordinate_system"] = coordinate_system
            acquired.append(page)
        return acquired

    coordinator.resolve_page_evidence(
        plan,
        source_pages=source_pages,
        page_ocr_loader=page_ocr_loader,
    )
    return calls


# Exact canonical population and exact physical table ownership observed in Lin.
# A tuple denotes a base table plus a physically continued table.  Card 4 has
# only its page-15 continuation table in this artifact; that physical evidence
# still has to remain attached to the printed account owner.
LIN_ACCOUNT_TABLES: dict[str, tuple[str, ...]] = {
    **{
        f"credit_account:non_revolving_loan:{ordinal}": table_ids
        for ordinal, table_ids in enumerate(
            (
                ("pt_4_1",),
                ("pt_4_2", "pt_5_0"),
                ("pt_5_1",),
                ("pt_5_2", "pt_6_0"),
                ("pt_6_1",),
                ("pt_6_2",),
                ("pt_6_3",),
                ("pt_7_0",),
                ("pt_7_1",),
                ("pt_7_2", "pt_8_0"),
                ("pt_8_1",),
                ("pt_8_2",),
                ("pt_9_0",),
                ("pt_9_1",),
                ("pt_9_2",),
                ("pt_10_0",),
                ("pt_10_1",),
                ("pt_10_2", "pt_11_0"),
                ("pt_11_1",),
                ("pt_11_2",),
                ("pt_11_3", "pt_12_0"),
                ("pt_12_1",),
            ),
            start=1,
        )
    },
    "credit_account:revolving_loan_account:1": ("pt_12_2",),
    **{
        f"credit_account:credit_card:{ordinal}": table_ids
        for ordinal, table_ids in enumerate(
            (
                ("pt_13_0",),
                ("pt_13_1", "pt_14_0"),
                ("pt_14_1",),
                ("pt_15_0",),
                ("pt_15_1",),
                ("pt_15_2", "pt_16_0"),
                ("pt_16_1", "pt_17_0"),
                ("pt_17_1",),
                ("pt_17_2", "pt_18_0"),
                ("pt_18_1",),
                ("pt_19_0",),
                ("pt_19_1", "pt_20_0"),
                ("pt_20_1",),
                ("pt_20_2", "pt_21_0"),
                ("pt_21_1",),
                ("pt_21_2",),
                ("pt_22_0",),
                ("pt_22_1",),
                ("pt_22_2", "pt_23_0"),
                ("pt_23_1",),
                ("pt_23_2",),
                ("pt_23_3",),
            ),
            start=1,
        )
    },
}

LIN_ACCOUNT_IDS = tuple(LIN_ACCOUNT_TABLES)

LIN_CARD_CURRENCIES: dict[int, str | None] = {
    1: "CNY",
    2: "CNY",
    3: "USD",
    4: None,
    5: None,
    6: "CNY",
    7: "CNY",
    8: "CNY",
    9: "CNY",
    10: "CNY",
    11: "CNY",
    12: "CNY",
    13: "CNY",
    14: "MOP",
    15: "USD",
    16: "EUR",
    17: "HKD",
    18: "USD",
    19: "CNY",
    20: "CNY",
    21: "CNY",
    22: "CNY",
}


# Family, ordinal, logical page, source page, top, immutable evidence id.
_ACCOUNT_ANCHORS: tuple[tuple[str, int, int, int, float, str], ...] = (
    *(  # Non-revolving loan headings on printed pages 4-12.
        ("non_revolving_loan", ordinal, page, source, top, eid)
        for ordinal, page, source, top, eid in (
            (1, 4, 2, 146.0, "ocr:sp0002:lp0004:0041"),
            (2, 4, 2, 434.5, "ocr:sp0002:lp0004:0211"),
            (3, 5, 3, 164.5, "ocr:sp0003:lp0005:0087"),
            (4, 5, 3, 383.5, "ocr:sp0003:lp0005:0192"),
            (5, 6, 3, 84.0, "ocr:sp0003:lp0006:0020"),
            (6, 6, 3, 303.5, "ocr:sp0003:lp0006:0112"),
            (7, 6, 3, 425.5, "ocr:sp0003:lp0006:0151"),
            (8, 7, 4, 40.5, "ocr:sp0004:lp0007:0000"),
            (9, 7, 4, 239.0, "ocr:sp0004:lp0007:0084"),
            (10, 7, 4, 412.0, "ocr:sp0004:lp0007:0147"),
            (11, 8, 4, 88.5, "ocr:sp0004:lp0008:0015"),
            (12, 8, 4, 347.0, "ocr:sp0004:lp0008:0156"),
            (13, 9, 5, 41.0, "ocr:sp0005:lp0009:0000"),
            (14, 9, 5, 222.0, "ocr:sp0005:lp0009:0064"),
            (15, 9, 5, 402.0, "ocr:sp0005:lp0009:0124"),
            (16, 10, 5, 71.5, "ocr:sp0005:lp0010:0000"),
            (17, 10, 5, 251.5, "ocr:sp0005:lp0010:0060"),
            (18, 10, 5, 432.5, "ocr:sp0005:lp0010:0113"),
            (19, 11, 6, 91.5, "ocr:sp0006:lp0011:0017"),
            (20, 11, 6, 296.0, "ocr:sp0006:lp0011:0088"),
            (21, 11, 6, 476.0, "ocr:sp0006:lp0011:0165"),
            (22, 12, 6, 167.5, "ocr:sp0006:lp0012:0041"),
        )
    ),
    (
        "revolving_loan_account",
        1,
        12,
        6,
        360.0,
        "ocr:sp0006:lp0012:0098",
    ),
    *(  # Credit-card headings on printed pages 13-23.
        ("credit_card", ordinal, page, source, top, eid)
        for ordinal, page, source, top, eid in (
            (1, 13, 7, 85.0, "ocr:sp0007:lp0013:0010"),
            (2, 13, 7, 303.0, "ocr:sp0007:lp0013:0127"),
            (3, 14, 7, 84.0, "ocr:sp0007:lp0014:0024"),
            (4, 14, 7, 380.0, "ocr:sp0007:lp0014:0215"),
            (5, 15, 8, 166.5, "ocr:sp0008:lp0015:0114"),
            (6, 15, 8, 475.5, "ocr:sp0008:lp0015:0311"),
            (7, 16, 8, 263.5, "ocr:sp0008:lp0016:0167"),
            (8, 17, 9, 104.0, "ocr:sp0009:lp0017:0052"),
            (9, 17, 9, 422.0, "ocr:sp0009:lp0017:0254"),
            (10, 18, 9, 239.0, "ocr:sp0009:lp0018:0144"),
            (11, 18, 9, 535.5, "ocr:sp0009:lp0018:0339"),
            (12, 19, 10, 288.5, "ocr:sp0010:lp0019:0160"),
            (13, 20, 10, 75.5, "ocr:sp0010:lp0020:0022"),
            (14, 20, 10, 385.0, "ocr:sp0010:lp0020:0169"),
            (15, 21, 11, 93.5, "ocr:sp0011:lp0021:0029"),
            (16, 21, 11, 312.5, "ocr:sp0011:lp0021:0142"),
            (17, 21, 11, 531.5, "ocr:sp0011:lp0021:0257"),
            (18, 22, 11, 239.0, "ocr:sp0011:lp0022:0101"),
            (19, 22, 11, 425.5, "ocr:sp0011:lp0022:0182"),
            (20, 23, 12, 90.0, "ocr:sp0012:lp0023:0029"),
            (21, 23, 12, 198.0, "ocr:sp0012:lp0023:0067"),
            (22, 23, 12, 305.0, "ocr:sp0012:lp0023:0096"),
        )
    ),
)


def _account_id(family: str, ordinal: int) -> str:
    return f"credit_account:{family}:{ordinal}"


_EXACT_CRITICAL_ACCOUNT_BBOXES: dict[tuple[str, int], list[float]] = {
    ("credit_card", 3): [55.0, 84.0, 264.0, 97.0],
    ("credit_card", 4): [52.5, 380.0, 324.5, 394.5],
    ("credit_card", 16): [34.5, 312.5, 216.5, 324.5],
    ("credit_card", 17): [33.0, 531.5, 214.5, 543.0],
    ("credit_card", 20): [46.5, 90.0, 70.0, 100.5],
    ("credit_card", 21): [46.5, 198.0, 69.0, 208.5],
    ("credit_card", 22): [45.5, 305.0, 244.0, 317.5],
}


def _account_anchor_bbox(family: str, ordinal: int, top: float) -> list[float]:
    return deepcopy(
        _EXACT_CRITICAL_ACCOUNT_BBOXES.get(
            (family, ordinal),
            [46.0, top, 330.0, top + 12.0],
        )
    )


def _anchor_ref(
    family: str,
    ordinal: int,
    logical_page: int,
    source_page: int,
    top: float,
    evidence_id: str,
) -> dict[str, Any]:
    return {
        "source": "candidate_b_account_anchor",
        "logical_page": logical_page,
        "source_page": source_page,
        "geometry_scope": "line",
        "binding": "printed_account_ordinal",
        "binding_quality": "printed_account_ordinal",
        "account_type": family,
        "category_sequence": ordinal,
        "bbox": _account_anchor_bbox(family, ordinal, top),
        "evidence_ids": [evidence_id],
    }


def _account_skeletons_through_card_19() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, ordinal, logical_page, source_page, top, evidence_id in _ACCOUNT_ANCHORS:
        if family == "credit_card" and ordinal > 19:
            continue
        account_id = _account_id(family, ordinal)
        rows.append(
            {
                "account_id": account_id,
                "account_type": family,
                "category_sequence": ordinal,
                "account_family_quality": "exact",
                "_printed_ordinal_status": "printed_unique",
                "_canonical_segment": {
                    "ownership_basis": "printed_anchor_to_next_anchor",
                    "pages": [logical_page],
                },
                "source_refs": [
                    _anchor_ref(
                        family,
                        ordinal,
                        logical_page,
                        source_page,
                        top,
                        evidence_id,
                    )
                ],
                "confidence": 1.0,
            }
        )
    return rows


def _account_frozen_pages() -> list[SimpleNamespace]:
    pages = {
        logical: _page(logical, source)
        for logical, source in (
            (4, 2),
            (5, 3),
            (6, 3),
            (7, 4),
            (8, 4),
            (9, 5),
            (10, 5),
            (11, 6),
            (12, 6),
            (13, 7),
            (14, 7),
            (15, 8),
            (16, 8),
            (17, 9),
            (18, 9),
            (19, 10),
            (20, 10),
            (21, 11),
            (22, 11),
            (23, 12),
            (24, 12),
        )
    }
    pages[4].texts.extend(
        (
            _text(
                "三信贷交易信息明细",
                (187.5, 109.5, 265.0, 119.5),
                "ocr:sp0002:lp0004:0026",
            ),
            _text(
                "(一)非循环贷账户",
                (197.0, 132.0, 260.0, 144.0),
                "ocr:sp0002:lp0004:0037",
            ),
        )
    )
    pages[12].texts.append(
        _text(
            "(二)循环贷账户二",
            (198.0, 347.0, 260.0, 357.0),
            "ocr:sp0006:lp0012:0096",
        )
    )
    pages[13].texts.append(
        _text(
            "(三)贷记卡账户",
            (176.0, 68.5, 244.5, 84.5),
            "ocr:sp0007:lp0013:0009",
        )
    )
    for family, ordinal, logical, _source, top, evidence_id in _ACCOUNT_ANCHORS:
        if family == "revolving_loan_account":
            heading = (
                "账户(授信协议标识:"
                "D10053310H00011022661000153960931220220529)"
            )
        elif family == "credit_card" and ordinal == 22:
            heading = (
                "账户22(授信协议标识:"
                "B10211000H00011310540169000143100001)"
            )
        else:
            heading = f"账户{ordinal}"
        pages[logical].texts.append(
            _text(
                heading,
                tuple(_account_anchor_bbox(family, ordinal, top)),
                evidence_id,
            )
        )
    pages[23].texts.extend(
        (
            _text(
                "(四)相关还款责任信息",
                (184.5, 384.0, 261.0, 392.5),
                "ocr:sp0012:lp0023:0123",
            ),
            _text(
                "账户1",
                (47.0, 420.5, 66.0, 429.5),
                "ocr:sp0012:lp0023:0131",
            ),
        )
    )
    pages[24].texts.extend(
        (
            _text(
                "账户2",
                (55.5, 38.5, 74.5, 48.5),
                "ocr:sp0012:lp0024:0000",
            ),
            _text(
                "账户3",
                (54.0, 180.0, 85.0, 195.5),
                "ocr:sp0012:lp0024:0046",
            ),
            _text(
                "(五)授信协议信息",
                (197.5, 326.5, 261.0, 338.0),
                "ocr:sp0012:lp0024:0086",
            ),
        )
    )

    # The raw census requires a physical PBOC account table on every page that
    # owns an account heading.  The exact trailing tables are kept verbatim;
    # earlier pages need only the same canonical base signature because this
    # population plane is forbidden from carrying business values.
    canonical_base = [
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
        ["机构", "B10000000H0001", "2020.01.01", "100", "100", "人民币元", "贷记卡", "信用/免担保"],
    ]
    for logical, page in pages.items():
        if any(str(text.content).startswith("账户") for text in page.texts):
            page.tables.append(
                _table(
                    f"population-proof-{logical}",
                    canonical_base,
                    logical_page=logical,
                    source_page=int(page.source_page_number),
                    bbox=(43.0, 40.0, 402.0, 90.0),
                    evidence_start=900,
                )
            )
    return list(pages.values())


def _account_context() -> SimpleNamespace:
    pages = _account_frozen_pages()
    # Repaired evidence has the exact production defect: the corrected account
    # registration stops at card 19 even though the immutable source plane has
    # a dense 1..22 card census.
    corrected: list[dict[str, Any]] = []
    for page in pages:
        lines = []
        for block in page.texts:
            content = str(block.content)
            if content in {"账户20", "账户21"} or content.startswith("账户22("):
                continue
            lines.append(
                {
                    "text": content,
                    "content": content,
                    "bbox": list(block.bbox),
                    "page": int(page.page_number),
                    "source_page": int(page.source_page_number),
                    "evidence_ids": list(block.evidence_ids),
                }
            )
        corrected.append(
            {
                "page": int(page.page_number),
                "source_page": int(page.source_page_number),
                "lines": lines,
            }
        )
    return _context(pages, corrected_pages=corrected)


def _table_page(table_id: str) -> int:
    return int(table_id.split("_", 2)[1])


def _account_table_observation(account_id: str) -> dict[str, Any]:
    family, ordinal_text = account_id.rsplit(":", 2)[-2:]
    ordinal = int(ordinal_text)
    table_ids = LIN_ACCOUNT_TABLES[account_id]
    source_refs = [
        {
            "source": "native_detail_table",
            "logical_page": _table_page(table_id),
            "source_page": (_table_page(table_id) + 1) // 2,
            "table_id": table_id,
            "geometry_scope": "table",
            "bbox": [43.0, 100.0, 402.0, 180.0],
        }
        for table_id in table_ids
    ]
    if family == "credit_card":
        currency = LIN_CARD_CURRENCIES[ordinal]
    else:
        currency = "CNY"
    row: dict[str, Any] = {
        "account_id": f"credit_account_table_observation:{account_id}",
        "_table_observation_id": f"credit_account_table_observation:{account_id}",
        "_table_observation_instance_id": f"lin:{table_ids[0]}",
        "_expected_account_id": account_id,
        "account_type": family,
        "source": "native_detail_account_table",
        "source_refs": source_refs,
        "source_refs_by_field": {},
        "canonical_raw": {},
        "confidence": 1.0,
    }
    if currency is not None:
        row.update(
            {
                "currency": currency,
                "account_currency": currency,
                "reporting_amount_currency": currency,
                "amount_unit": "yuan" if currency == "CNY" else None,
                "reporting_amount_unit": "yuan" if currency == "CNY" else None,
            }
        )
        field_ref = {
            **source_refs[0],
            "source": "native_detail_table_cell",
            "geometry_scope": "cell",
            "row": 1,
            "column": 5,
            "binding": "canonical_label_slot",
            "binding_quality": "native_label_column",
            "evidence_ids": [f"lin:{table_ids[0]}:currency"],
        }
        row["source_refs_by_field"] = {
            "currency": [field_ref],
            "account_currency": [field_ref],
        }
        row["canonical_raw"] = {
            "currency": currency,
            "account_currency": currency,
        }
    return row


def _assert_lin_account_contract(rows: list[dict[str, Any]]) -> None:
    by_id = {str(row.get("account_id") or ""): row for row in rows}
    assert len(rows) == len(by_id) == 45
    assert tuple(by_id) == LIN_ACCOUNT_IDS
    for account_id, expected_table_ids in LIN_ACCOUNT_TABLES.items():
        row = by_id[account_id]
        anchors = [
            ref
            for ref in row.get("source_refs") or ()
            if ref.get("source") == "candidate_b_account_anchor"
        ]
        assert len(anchors) == 1, account_id
        assert anchors[0]["geometry_scope"] == "line"
        assert anchors[0]["binding"] == "printed_account_ordinal"
        assert anchors[0]["binding_quality"] == "printed_account_ordinal"
        assert anchors[0]["evidence_ids"]
        actual_tables = tuple(
            ref["table_id"]
            for ref in row.get("source_refs") or ()
            if ref.get("source") == "native_detail_table"
        )
        assert actual_tables == expected_table_ids, account_id

    for ordinal, expected_currency in LIN_CARD_CURRENCIES.items():
        row = by_id[f"credit_account:credit_card:{ordinal}"]
        assert row.get("account_currency") == expected_currency
        assert row.get("reporting_amount_currency") == expected_currency
        expected_unit = "yuan" if expected_currency == "CNY" else None
        assert row.get("amount_unit") == expected_unit
        assert row.get("reporting_amount_unit") == expected_unit


def test_lin_raw_account_census_proves_the_dense_45_identity_population() -> None:
    context = _account_context()

    census = native_extraction._sealed_raw_account_population_census(context)

    assert census is not None
    assert census["sequences"]["non_revolving_loan"] == list(range(1, 23))
    assert census["sequences"]["credit_card"] == list(range(1, 23))
    assert census["endpoints"] == {
        "non_revolving_loan": 22,
        "credit_card": 22,
    }
    assert {
        _account_id(family, ordinal)
        for family, observations in census["ordinal_observations"].items()
        for ordinal in observations
    } == set(LIN_ACCOUNT_IDS) - {"credit_account:revolving_loan_account:1"}


def test_lin_materializer_replays_raw_cards_20_to_22_after_corrected_registration_loss() -> None:
    context = _account_context()

    rows = native_extraction._materialize_registered_account_population_skeletons(
        context,
        _account_skeletons_through_card_19(),
    )

    assert len(rows) == 45
    assert {row["account_id"] for row in rows} == set(LIN_ACCOUNT_IDS)
    for ordinal in (20, 21, 22):
        row = next(
            item
            for item in rows
            if item["account_id"] == f"credit_account:credit_card:{ordinal}"
        )
        expected = next(
            item for item in _ACCOUNT_ANCHORS if item[0:2] == ("credit_card", ordinal)
        )
        assert row["source_refs"] == [
            _anchor_ref(*expected)
        ]


def test_lin_account_assembly_publishes_exact_ids_tables_currencies_and_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _account_context()
    table_observations = [
        _account_table_observation(account_id) for account_id in LIN_ACCOUNT_IDS
    ]
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: _account_skeletons_through_card_19(),
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: list(skeletons),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: (deepcopy(table_observations), [], []),
    )

    # This test isolates the population materializer and account assembler: the
    # observations are already exact owner-tagged outputs of the unchanged
    # table strategy.  The next test exercises the real matcher on the five Lin
    # owners that failed in production, so this deterministic join cannot hide
    # the card-4/card-17 ownership seam.
    def exact_owner_matches(
        skeletons: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> dict[int, int]:
        observation_index = {
            row["_expected_account_id"]: index
            for index, row in enumerate(observations)
        }
        return {
            index: observation_index[row["account_id"]]
            for index, row in enumerate(skeletons)
            if row["account_id"] in observation_index
        }

    monkeypatch.setattr(
        native_extraction,
        "_match_account_table_observations",
        exact_owner_matches,
    )
    monkeypatch.setattr(
        native_extraction,
        "_resolve_owned_revolving_table_families",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_canonical_singleton_account_matches",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_provisional_physical_account_records",
        lambda *_args, **_kwargs: {},
    )

    rows, repayments, events = native_extraction._extract_accounts(context)

    assert repayments == []
    assert events == []
    _assert_lin_account_contract(rows)


def test_lin_account_population_fails_closed_on_anchor_owner_ambiguity() -> None:
    context = _account_context()
    page23 = context._frozen_logical_pages[23]
    duplicate = _text(
        "账户20",
        (46.0, 112.0, 70.0, 122.5),
        "ocr:sp0012:lp0023:adversarial-duplicate-20",
    )
    page23.texts.append(duplicate)

    census = native_extraction._sealed_raw_account_population_census(context)

    assert census is not None
    assert "credit_card" not in census["sequences"]


def test_lin_account_population_fails_closed_on_non_authoritative_page_topology() -> None:
    context = _account_context()
    context.reading_order_by_logical[23] = context.reading_order_by_logical[22]

    assert native_extraction._sealed_raw_account_population_census(context) is None


def test_lin_account_table_matcher_withholds_two_equal_physical_candidates() -> None:
    skeleton = next(
        row
        for row in _account_skeletons_through_card_19()
        if row["account_id"] == "credit_account:credit_card:17"
    )
    observation = _account_table_observation("credit_account:credit_card:17")
    competing = deepcopy(observation)
    competing["_table_observation_id"] += ":competing"
    competing["_table_observation_instance_id"] += ":competing"

    matches = native_extraction._match_account_table_observations(
        [skeleton],
        [observation, competing],
    )

    assert matches == {}


def test_lin_actual_matcher_binds_currency_cross_page_and_trailing_failure_cards() -> None:
    """Replay the exact geometry seam without replacing the production matcher."""

    critical = (
        (3, "pt_14_1", (50.5, 94.5, 402.0, 370.5), "USD"),
        (4, "pt_15_0", (45.0, 35.5, 398.5, 154.0), None),
        (16, "pt_21_2", (30.5, 322.5, 386.5, 520.0), "EUR"),
        (17, "pt_22_0", (52.5, 43.5, 402.5, 228.5), "HKD"),
        (20, "pt_23_1", (43.5, 99.0, 397.0, 185.0), "CNY"),
        (21, "pt_23_2", (43.5, 207.0, 397.0, 293.0), "CNY"),
        (22, "pt_23_3", (43.5, 315.5, 397.0, 369.0), "CNY"),
    )
    next_top = {
        3: 380.0,
        16: 531.5,
        20: 198.0,
        21: 305.0,
        22: 384.0,
    }
    skeletons: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for ordinal, table_id, table_bbox, currency in critical:
        anchor = next(
            item
            for item in _ACCOUNT_ANCHORS
            if item[0:2] == ("credit_card", ordinal)
        )
        family, _ordinal, logical_page, source_page, top, evidence_id = anchor
        account_id = _account_id(family, ordinal)
        page_segments = [
            {
                "logical_page": logical_page,
                "min_y": top,
                "max_y": next_top.get(ordinal),
            }
        ]
        if ordinal in {4, 17}:
            page_segments.append(
                {
                    "logical_page": _table_page(table_id),
                    "min_y": 0.0,
                    "max_y": 166.5 if ordinal == 4 else 239.0,
                }
            )
        skeletons.append(
            {
                "account_id": account_id,
                "account_type": "credit_card",
                "category_sequence": ordinal,
                "account_family_quality": "exact",
                "_printed_ordinal_status": "printed_unique",
                "_canonical_segment": {
                    "ownership_basis": "printed_anchor_to_next_anchor",
                    "anchor_logical_page": logical_page,
                    "anchor_bbox": _account_anchor_bbox(family, ordinal, top),
                    "pages": page_segments,
                },
                "source_refs": [
                    _anchor_ref(
                        family,
                        ordinal,
                        logical_page,
                        source_page,
                        top,
                        evidence_id,
                    )
                ],
            }
        )
        observation = _account_table_observation(account_id)
        observation["source_refs"] = [
            {
                "source": "native_detail_table",
                "logical_page": _table_page(table_id),
                "source_page": (_table_page(table_id) + 1) // 2,
                "table_id": table_id,
                "geometry_scope": "table",
                "bbox": list(table_bbox),
            }
        ]
        if ordinal in {4, 17}:
            observation["_pending_anchor_account_id"] = account_id
            observation["_table_account_family_resolution"] = (
                "exact_trailing_anchor_adjacent_page_prefix_base"
            )
        if currency is not None:
            currency_ref = {
                **observation["source_refs"][0],
                "source": "native_detail_table_cell",
                "geometry_scope": "cell",
                "row": 1,
                "column": 5,
                "binding": "canonical_label_slot",
                "binding_quality": "native_label_column",
                "evidence_ids": [f"lin:{table_id}:currency"],
            }
            observation["account_currency"] = currency
            observation["reporting_amount_currency"] = currency
            observation["amount_unit"] = "yuan" if currency == "CNY" else None
            observation["reporting_amount_unit"] = (
                "yuan" if currency == "CNY" else None
            )
            observation["source_refs_by_field"] = {
                "currency": [currency_ref],
                "account_currency": [currency_ref],
            }
        observations.append(observation)

    matches = native_extraction._match_account_table_observations(
        skeletons,
        observations,
        parse_result=_account_context(),
    )

    assert matches == {index: index for index in range(len(critical))}
    for index, (ordinal, table_id, _bbox, currency) in enumerate(critical):
        observation = observations[matches[index]]
        assert [
            ref["table_id"]
            for ref in observation["source_refs"]
            if ref.get("geometry_scope") == "table"
        ] == [table_id]
        assert observation.get("account_currency") == currency
        expected_unit = "yuan" if currency == "CNY" else None
        assert observation.get("amount_unit") == expected_unit
        if currency is not None:
            refs = observation["source_refs_by_field"]["account_currency"]
            assert len(refs) == 1
            assert refs[0]["table_id"] == table_id
            assert refs[0]["geometry_scope"] == "cell"
            assert refs[0]["evidence_ids"]


# Exact agreement source tables from printed pages 24-26.  Ordinals 3 and 9
# start with terminal headings/upper labels and continue as headerless tables on
# the next page.  Ordinal 5 is an OCR-damaged same-page fixed-layout table.
LIN_AGREEMENT_ROWS: dict[str, list[list[str]]] = {
    "pt_24_2": [
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"],
        [
            "浙江网商银行股份有限 公司 准",
            "D10053310H00011022 6610001539609312202 20529",
            "2022.11.30",
            "长期 2",
            "循环贷款额度",
        ],
        ["授信额度总", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["10,00%", "**", "", "8,667", "人民市元"],
    ],
    "pt_24_3": [
        ["银 管理机构", "授信协议标识", "生效日期", "S 到期日期", "授信额度用途"],
        [
            "平安消费金融有限公司",
            "X3101010000603001 WXNEWZ0000146075 72",
            "2022.09.14",
            "福",
            "循环贷款额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["15,000", "-", "--", "浙", "人民币元"],
        ["", "", "", "", ""],
    ],
    "pt_25_0": [
        [
            "上海浦东发展银行股份 有限公司信用卡中心 授信额度",
            "B11512900H00010020 44607320210926",
            "2022.06.26",
            "长期",
            "信用卡共享额度",
        ],
        ["", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["40,000", "", "--", "39,403", "人民币元"],
        ["", "", "", "", ""],
    ],
    "pt_25_1": [
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"],
        [
            "公司度门市分行 中国工商银行股份有限",
            "0000530059989 B10111000H00014100",
            "2021.05.31",
            "长期",
            "信用卡共享额度",
        ],
        ["授信额度 及", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["50,000", "", "", "49,226", "人民币元"],
        ["", "", "", "", ""],
    ],
    "pt_25_2": [
        [
            "管理机构 张",
            "授信协议标识",
            "生效日期",
            "繁 到期日期",
            "授信额度用途 信用卡共享额度 币种",
        ],
        [
            "中国农业银行股份有限 公司",
            "B10211000H00011310 540169000143100001",
            "2021.04.10",
            "签",
            "",
        ],
        ["授信额度", "授信限额 \"", "授信限额编号", "已用额度", ""],
        ["45,000", "", "**", "梦 44,767", "人民币元"],
        ["", "", "", "", ""],
    ],
    "pt_25_3": [
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途 信用卡共享额度"],
        [
            "中信银行股份有限公司 信用卡中心",
            "B10611000H00016226 880016197709607",
            "2019.10.28",
            "长期",
            "",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种 人民币元"],
        ["30,000", "", "", "27,377", ""],
    ],
    "pt_25_4": [
        ["管理机构", "授信协议标识", "生效日期", "到期日期 授信额度用途", ""],
        [
            "公司 中国光大银行股份有限",
            "R B10711000H00011000 0011111111149889800 0000",
            "2019.05.21",
            "长期",
            "信用卡共享额度",
        ],
        ["授信额度 福", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["36,400 2", "", "", "36,393", "人民市元"],
    ],
    "pt_25_5": [
        ["管理机构", "授信协议标识", "生效日期", "到期日期 16", "授信额度用途"],
        [
            "福州分行 广发银行股份有限公司",
            "B11215800H00011009 918492156",
            "2019.05.03",
            "长期 A 已用额度",
            "信用卡共享额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "", "币种"],
        ["11,500", "**", "", "1,008", "人民币元"],
    ],
    "pt_26_0": [
        [
            "招商银行股份有限公司 信用卡中心",
            "B11115840H00010000 0000000000000000002 0211001012824255300 1001",
            "2016.03.28 a",
            "长期",
            "信用卡共享额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["50,000", "", "*.", "50,998", "人民币元"],
    ],
    "pt_26_1": [
        [
            "管理机构 平安银行股份有限公司 信用卡中心",
            "授信协议标识",
            "生效日期",
            "到期日期",
            "授信额度用途",
        ],
        ["", "B11415840H00012422 998009899727458", "2014.01.16", "长期", "信用卡独立额度"],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["19,000", "", "", "18,889", "瓷 人民币元"],
    ],
    "pt_26_2": [
        ["管理机构", "授信协议标识", "生效日期", "4 到期日期", "授信额度用途"],
        [
            "中国建设银行股份有限 公司福建自贸试验区福 州片区分行",
            "B10411000H00018461 90000103535597",
            "2013.08.14",
            "密 3长期 多",
            "信用卡共享额度",
        ],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["20,000", "", "*-", "17,891", "人民币元"],
    ],
}

LIN_AGREEMENT_EXPECTED: dict[int, tuple[str, str, str]] = {
    1: (
        "D10053310H00011022661000153960931220220529",
        "浙江网商银行股份有限公司",
        "pt_24_2",
    ),
    2: (
        "X3101010000603001WXNEWZ000014607572",
        "平安消费金融有限公司",
        "pt_24_3",
    ),
    3: (
        "B11512900H0001002044607320210926",
        "上海浦东发展银行股份有限公司信用卡中心",
        "pt_25_0",
    ),
    4: (
        "0000530059989B10111000H00014100",
        "中国工商银行股份有限公司厦门市分行",
        "pt_25_1",
    ),
    5: (
        "B10211000H00011310540169000143100001",
        "中国农业银行股份有限公司",
        "pt_25_2",
    ),
    6: (
        "B10611000H00016226880016197709607",
        "中信银行股份有限公司信用卡中心",
        "pt_25_3",
    ),
    7: (
        "RB10711000H0001100000111111111498898000000",
        "中国光大银行股份有限公司",
        "pt_25_4",
    ),
    8: (
        "B11215800H00011009918492156",
        "广发银行股份有限公司福州分行",
        "pt_25_5",
    ),
    9: (
        "B11115840H00010000000000000000000000202110010128242553001001",
        "招商银行股份有限公司信用卡中心",
        "pt_26_0",
    ),
    10: (
        "B11415840H00012422998009899727458",
        "平安银行股份有限公司信用卡中心",
        "pt_26_1",
    ),
    11: (
        "B10411000H0001846190000103535597",
        "中国建设银行股份有限公司福建自贸试验区福州片区分行",
        "pt_26_2",
    ),
}

# These seven institution cells remain discovery observations, not business
# values. The discovery strategy withholds them and preserves exact OCR text
# plus cell-local evidence; the separately tested repair policy acts later.
LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW: dict[int, str] = {
    1: "浙江网商银行股份有限 公司 准",
    2: "平安消费金融有限公司",
    3: "上海浦东发展银行股份 有限公司信用卡中心 授信额度",
    4: "公司度门市分行 中国工商银行股份有限",
    5: "中国农业银行股份有限 公司",
    8: "福州分行 广发银行股份有限公司",
    9: "招商银行股份有限公司 信用卡中心",
}
LIN_AGREEMENT_FUTURE_INSTITUTION_REPAIR_TARGETS: dict[int, str] = {
    sequence: LIN_AGREEMENT_EXPECTED[sequence][1]
    for sequence in LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW
}
LIN_AGREEMENT_SAFE_PUBLISHED_INSTITUTIONS: dict[int, str] = {
    sequence: LIN_AGREEMENT_EXPECTED[sequence][1]
    for sequence in (6, 7, 10, 11)
}


def _agreement_pages() -> list[SimpleNamespace]:
    page24 = _page(
        24,
        12,
        canonical_template_id="mixed_pboc_sections",
        texts=(
            _text("(五)授信协议信息", (197.5, 326.5, 261.0, 338.0), "ocr:sp0012:lp0024:0086"),
            _text("授信协议1", (56.0, 349.5, 87.0, 359.5), "ocr:sp0012:lp0024:0090"),
            _text("授信协议2", (55.0, 435.5, 88.5, 450.0), "ocr:sp0012:lp0024:0118"),
            _text("授信协议3", (56.0, 528.0, 87.5, 538.0), "ocr:sp0012:lp0024:0145"),
            _text("生效日期", (213.5, 540.0, 242.5, 550.5), "ocr:sp0012:lp0024:0146"),
            _text("授信额度用途", (346.5, 540.0, 387.0, 552.0), "ocr:sp0012:lp0024:0147"),
            _text("授信协议标识", (138.0, 540.5, 178.5, 550.5), "ocr:sp0012:lp0024:0148"),
            _text("到期日期", (283.0, 540.5, 311.5, 551.0), "ocr:sp0012:lp0024:0149"),
            _text("管理机构", (76.0, 541.0, 102.5, 550.0), "ocr:sp0012:lp0024:0150"),
        ),
        tables=(
            _table("pt_24_0", LIN_LIABILITY_ROWS["pt_24_0"], logical_page=24, source_page=12, bbox=(52.5, 48.0, 402.0, 158.5), evidence_start=1),
            _table("pt_24_1", LIN_LIABILITY_ROWS["pt_24_1"], logical_page=24, source_page=12, bbox=(52.5, 191.0, 402.0, 303.0), evidence_start=47),
            _table("pt_24_2", LIN_AGREEMENT_ROWS["pt_24_2"], logical_page=24, source_page=12, bbox=(52.5, 359.0, 402.0, 425.5), evidence_start=91),
            _table("pt_24_3", LIN_AGREEMENT_ROWS["pt_24_3"], logical_page=24, source_page=12, bbox=(52.5, 448.5, 402.5, 517.0), evidence_start=119),
        ),
    )
    page25_texts = [
        _text(f"授信协议{sequence}", bbox, evidence_id)
        for sequence, bbox, evidence_id in (
            (4, (37.0, 100.5, 68.5, 110.5), "ocr:sp0013:lp0025:0018"),
            (5, (36.0, 183.0, 69.5, 195.5), "ocr:sp0013:lp0025:0045"),
            (6, (35.0, 267.0, 67.5, 279.0), "ocr:sp0013:lp0025:0072"),
            (7, (34.0, 350.0, 67.5, 361.5), "ocr:sp0013:lp0025:0095"),
            (8, (33.5, 440.5, 67.0, 452.5), "ocr:sp0013:lp0025:0123"),
            (9, (32.5, 523.5, 65.0, 535.5), "ocr:sp0013:lp0025:0150"),
        )
    ]
    page25_texts.extend(
        (
            _text("管理机构", (53.0, 537.5, 80.5, 548.0), "ocr:sp0013:lp0025:0151"),
            _text("生效日期", (194.0, 537.5, 222.5, 550.0), "ocr:sp0013:lp0025:0152"),
            _text("授信协议标识", (117.5, 538.0, 158.5, 548.5), "ocr:sp0013:lp0025:0153"),
            _text("授信额度用途", (329.5, 539.5, 369.5, 549.5), "ocr:sp0013:lp0025:0154"),
            _text("到期日期", (251.0, 538.0, 281.0, 549.0), "ocr:sp0013:lp0025:0155"),
        )
    )
    page25 = _page(
        25,
        13,
        canonical_template_id="unresolved",
        texts=page25_texts,
        tables=(
            _table("pt_25_0", LIN_AGREEMENT_ROWS["pt_25_0"], logical_page=25, source_page=13, bbox=(34.0, 40.5, 389.5, 89.5), evidence_start=1),
            _table("pt_25_1", LIN_AGREEMENT_ROWS["pt_25_1"], logical_page=25, source_page=13, bbox=(33.5, 110.0, 389.0, 173.0), evidence_start=19),
            _table("pt_25_2", LIN_AGREEMENT_ROWS["pt_25_2"], logical_page=25, source_page=13, bbox=(32.5, 193.5, 388.5, 257.0), evidence_start=46),
            _table("pt_25_3", LIN_AGREEMENT_ROWS["pt_25_3"], logical_page=25, source_page=13, bbox=(32.0, 277.5, 387.5, 340.0), evidence_start=73),
            _table("pt_25_4", LIN_AGREEMENT_ROWS["pt_25_4"], logical_page=25, source_page=13, bbox=(31.0, 360.5, 387.0, 429.5), evidence_start=96),
            _table("pt_25_5", LIN_AGREEMENT_ROWS["pt_25_5"], logical_page=25, source_page=13, bbox=(30.5, 450.5, 386.5, 513.0), evidence_start=124),
        ),
    )
    page26 = _page(
        26,
        13,
        canonical_template_id="mixed_pboc_sections",
        texts=(
            _text("授信协议10", (56.5, 115.5, 92.0, 126.0), "ocr:sp0013:lp0026:0020"),
            _text("授信协议11", (56.0, 200.0, 90.5, 210.0), "ocr:sp0013:lp0026:0045"),
            _text("四查询记录", (205.0, 292.0, 250.0, 303.0), "ocr:sp0013:lp0026:0071"),
        ),
        tables=(
            _table("pt_26_0", LIN_AGREEMENT_ROWS["pt_26_0"], logical_page=26, source_page=13, bbox=(53.5, 44.0, 403.0, 104.5), evidence_start=1),
            _table("pt_26_1", LIN_AGREEMENT_ROWS["pt_26_1"], logical_page=26, source_page=13, bbox=(52.5, 125.5, 402.5, 188.5), evidence_start=21),
            _table("pt_26_2", LIN_AGREEMENT_ROWS["pt_26_2"], logical_page=26, source_page=13, bbox=(52.0, 209.5, 402.0, 278.0), evidence_start=46),
        ),
    )
    return [page24, page25, page26]


def _agreement_context() -> SimpleNamespace:
    return _context(_agreement_pages())


def _one_preserved_raw_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    assert isinstance(value, (list, tuple))
    assert len(value) == 1
    assert isinstance(value[0], str)
    return value[0]


def _assert_lin_agreement_contract(
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    by_sequence = {int(row["sequence"]): row for row in rows}
    assert len(rows) == len(by_sequence) == 11
    assert set(by_sequence) == set(range(1, 12))
    for sequence, (identifier, institution, table_id) in LIN_AGREEMENT_EXPECTED.items():
        row = by_sequence[sequence]
        assert row["account_identifier"] == identifier
        assert {
            ref.get("table_id")
            for ref in row.get("source_refs") or ()
            if ref.get("geometry_scope") == "table"
        } == {table_id}
        anchor_refs = [
            ref
            for ref in row.get("source_refs") or ()
            if ref.get("binding") == "canonical_card_anchor"
        ]
        assert len(anchor_refs) == 1
        assert anchor_refs[0]["sequence"] == sequence
        assert anchor_refs[0]["field_name"] == "sequence"
        assert anchor_refs[0]["geometry_scope"] == "text"
        assert (
            anchor_refs[0]["binding_quality"]
            == "printed_credit_agreement_ordinal"
        )
        assert anchor_refs[0]["evidence_ids"]

        if sequence in LIN_AGREEMENT_SAFE_PUBLISHED_INSTITUTIONS:
            assert row["institution"] == institution
            continue

        raw_institution = LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW[sequence]
        assert row["institution"] is None
        assert "institution" in set(row.get("_unresolved_fields") or ())
        assert "institution" in set(row.get("_observed_fields") or ())
        assert _one_preserved_raw_text(
            row.get("canonical_raw", {}).get("institution")
        ) == raw_institution
        field_refs = row.get("source_refs_by_field", {}).get("institution") or ()
        assert len(field_refs) == 1
        assert field_refs[0]["table_id"] == table_id
        assert field_refs[0]["geometry_scope"] == "cell"
        assert field_refs[0]["evidence_ids"]

        field_issues = [
            issue
            for issue in issues
            if issue.get("target_record_id") == row["credit_line_id"]
            and issue.get("target_dataset") == "credit_lines"
            and issue.get("field_name") == "institution"
        ]
        assert field_issues, sequence
        assert not any(issue.get("status") == "resolved" for issue in field_issues)
        assert any(
            _one_preserved_raw_text(issue.get("observed_value"))
            == raw_institution
            for issue in field_issues
        )
        for issue in field_issues:
            refs = issue.get("source_refs") or ()
            assert refs
            assert {ref.get("table_id") for ref in refs} == {table_id}
            assert all(ref.get("geometry_scope") == "cell" for ref in refs)


def test_lin_agreement_census_conserves_all_eleven_printed_ordinals() -> None:
    census = native_extraction._sealed_agreement_population_census(
        _agreement_context()
    )

    assert census is not None
    assert census["sequences"] == list(range(1, 12))
    assert {
        sequence: (
            refs[0]["logical_page"],
            refs[0]["source_page"],
            tuple(refs[0]["evidence_ids"]),
        )
        for sequence, observation in census["ordinal_observations"].items()
        if (refs := observation["source_refs"])
    }[3] == (24, 12, ("ocr:sp0012:lp0024:0145",))
    assert census["ordinal_observations"][9]["source_refs"][0][
        "evidence_ids"
    ] == ["ocr:sp0013:lp0025:0150"]


def test_lin_credit_line_assembly_conserves_all_owners_and_defers_institution_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _agreement_context()
    # The live failure reached the sealed exact/identity materializers after the
    # ordinary parser produced no usable rows for these owners.  Emptying only
    # that upstream observation list reproduces the production fallback while
    # leaving both agreement materializers and the assembler untouched.
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset: [] if dataset == "credit_lines" else [],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    _assert_lin_agreement_contract(
        rows,
        context._personal_detail_extraction_issues,
    )


def test_lin_agreement_policy_repairs_only_deferred_institution_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _agreement_context()
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset: [] if dataset == "credit_lines" else [],
    )
    discovery_rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )
    deferred_rows = [
        row
        for row in discovery_rows
        if int(row["sequence"]) in LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW
    ]
    repair_payload = {
        "credit_lines": [
            {
                "credit_line_id": row["credit_line_id"],
                "institution": row.get("institution"),
                "canonical_raw": {
                    "institution": row["canonical_raw"]["institution"]
                },
                "source_refs_by_field": {
                    "institution": deepcopy(
                        row["source_refs_by_field"]["institution"]
                    )
                },
            }
            for row in deferred_rows
        ]
    }
    repair_issues = [
        deepcopy(issue)
        for issue in context._personal_detail_extraction_issues
        if issue.get("status") != "resolved"
        and issue.get("target_dataset") == "credit_lines"
        and issue.get("field_name") == "institution"
    ]
    coordinator = BusinessUncertaintyRepairCoordinator(context)
    plan = coordinator.plan(
        repair_payload,
        canonical_audit={"unresolved_pages": []},
        extraction_issues=repair_issues,
    )
    sequence_by_record_id = {
        str(row["credit_line_id"]): int(row["sequence"])
        for row in deferred_rows
    }
    institution_repairs = [
        repair
        for repair in plan.field_repairs
        if repair.dataset_name == "credit_lines"
        and repair.field_name == "institution"
    ]
    by_sequence = {
        sequence_by_record_id[repair.record_id]: repair
        for repair in institution_repairs
    }

    assert set(by_sequence) == set(LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW)
    assert by_sequence[3].mode == "deterministic"
    assert by_sequence[3].candidate_value == (
        LIN_AGREEMENT_FUTURE_INSTITUTION_REPAIR_TARGETS[3]
    )
    assert {
        sequence
        for sequence, repair in by_sequence.items()
        if repair.mode == "context_rich_reocr"
    } == {1, 2, 4, 5, 8, 9}

    calls = _resolve_with_simulated_page_ocr(
        coordinator,
        plan,
        source_pages=[
            {
                "page": int(page.page_number),
                "source_page": int(page.source_page_number),
                "lines": [],
            }
            for page in context.pages
        ],
        candidate_for=lambda repair: (
            LIN_AGREEMENT_FUTURE_INSTITUTION_REPAIR_TARGETS[
                sequence_by_record_id[repair.record_id]
            ]
            if repair.field_name == "institution"
            and repair.mode == "context_rich_reocr"
            else None
        ),
    )
    _activate_field_repair(context, plan)
    repaired_rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )
    repaired_by_sequence = {
        int(row["sequence"]): row for row in repaired_rows
    }

    assert len(repaired_rows) == 11
    assert plan.reconstruction_evidence == {}
    assert all(
        reason == "business_field_context_rich_reocr_required"
        for _pages, reason in calls
    )
    assert {
        sequence: repaired_by_sequence[sequence]["institution"]
        for sequence in LIN_AGREEMENT_EXPECTED
    } == {
        sequence: expected_row[1]
        for sequence, expected_row in LIN_AGREEMENT_EXPECTED.items()
    }
    assert all(
        _one_preserved_raw_text(
            repaired_by_sequence[sequence]["canonical_raw"]["institution"]
        )
        == raw
        for sequence, raw in LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW.items()
    )


def test_lin_cross_page_agreement_owner_requires_adjacent_source_topology() -> None:
    context = _agreement_context()
    context._frozen_logical_pages[25].source_page_number = 14
    context._frozen_logical_pages[25].source_page = 14

    candidates = native_extraction._sealed_agreement_identity_table_candidates(
        context
    )

    assert not any(
        candidate.fields.get("__printed_sequence") in {"3", "9"}
        for candidate in candidates
    )


def test_lin_cross_page_agreement_owner_fails_closed_on_invalid_marker_geometry() -> None:
    context = _agreement_context()
    invalid_marker = _text(
        "授信协议4",
        (56.0, 560.0, 88.0, 571.0),
        "ocr:sp0012:lp0024:0999",
    )
    invalid_marker.bbox = None
    context._frozen_logical_pages[24].texts.append(invalid_marker)

    candidates = native_extraction._sealed_agreement_identity_table_candidates(
        context
    )

    assert not any(
        candidate.fields.get("__printed_sequence") == "3"
        for candidate in candidates
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("page_number", True),
        ("source_page_number", 13.5),
    ),
)
def test_lin_cross_page_agreement_owner_requires_native_integer_page_identity(
    field_name: str,
    invalid_value: Any,
) -> None:
    context = _agreement_context()
    setattr(context._frozen_logical_pages[25], field_name, invalid_value)

    candidates = native_extraction._sealed_agreement_identity_table_candidates(
        context
    )

    assert not any(
        candidate.fields.get("__printed_sequence") in {"3", "9"}
        for candidate in candidates
    )


@pytest.mark.parametrize(
    "invalid_order",
    (
        {24: 1, 25: True, 26: 3},
        {24: 1, 25: 2.0, 26: 3},
        {24: 1, "25": 2, 26: 3},
    ),
)
def test_lin_cross_page_agreement_owner_requires_native_integer_reading_order(
    invalid_order: Mapping[Any, Any],
) -> None:
    context = _agreement_context()
    context.reading_order_by_logical = dict(invalid_order)

    candidates = native_extraction._sealed_agreement_identity_table_candidates(
        context
    )

    assert not any(
        candidate.fields.get("__printed_sequence") in {"3", "9"}
        for candidate in candidates
    )


def test_lin_agreement_population_fails_closed_on_non_dense_page_order() -> None:
    context = _agreement_context()
    context.reading_order_by_logical = {24: 1, 25: 3, 26: 4}

    assert native_extraction._sealed_agreement_population_census(context) is None


def test_lin_headerless_agreement_identity_fails_closed_on_two_table_owners() -> None:
    context = _agreement_context()
    page25 = context._frozen_logical_pages[25]
    duplicate = _table(
        "pt_25_0_competing",
        LIN_AGREEMENT_ROWS["pt_25_0"],
        logical_page=25,
        source_page=13,
        bbox=(34.0, 41.0, 389.5, 90.0),
        evidence_start=801,
    )
    page25.tables.insert(1, duplicate)

    candidates = native_extraction._sealed_agreement_identity_table_candidates(
        context
    )

    assert not any(
        candidate.fields.get("授信协议标识")
        == LIN_AGREEMENT_EXPECTED[3][0]
        for candidate in candidates
    )


def test_lin_malformed_primary_candidate_cannot_monopolize_agreement_5_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _agreement_context()
    malformed = SimpleNamespace(
        fields={"管理机构": "中国农业银行股份有限公司"},
        source_refs=(
            {
                "source": "native_detail_table",
                "logical_page": 25,
                "source_page": 13,
                "table_id": "pt_25_2",
                "geometry_scope": "table",
                "bbox": [32.5, 193.5, 388.5, 257.0],
            },
        ),
        confidence=1.0,
        source_refs_by_field={},
        binding_quality_by_field={},
        observed_labels=frozenset({"管理机构"}),
        unresolved_labels=frozenset({"授信协议标识"}),
        agreement_raw_observations={},
        agreement_corrections=(),
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset: [malformed]
        if dataset == "credit_lines"
        else [],
    )

    rows = native_extraction._extract_credit_lines(context)

    agreement5 = [row for row in rows if row.get("_printed_sequence") == 5]
    assert len(agreement5) == 1
    assert agreement5[0]["account_identifier"] == LIN_AGREEMENT_EXPECTED[5][0]


LIN_LIABILITY_ROWS: dict[str, list[list[str]]] = {
    "pt_23_4": [
        ["管理机构", "业务种类", "开立日期", "", "到期日期", "贵任人类型", "还款贵任金额", "", "币种", "保证合同编号"],
        ["梅赛德斯-奔 驰汽车金融有 限公司", "贷款", "2021.08.03", "", "2024.08.03", "保证", "629,860", "", "多 民币元", "Y10061000H 0001EIP1967 714G01"],
        ["主业务借款人", "", "", "主业务借款人证件类型", "", "", "", "主业务借款人证件号码", "", ""],
        ["厦门雯玥轩商贸有限公司", "", "", "中征码", "", "", "", "3502030011973425 2", "", ""],
        ["# 截至2023年01月03日", "", "", "", "", "", "", "", "", ""],
        ["余额", "", "", "7 五级分类 S", "", "", "", "逾期月数", "", ""],
        ["348,791", "", "", "正常 发", "", "", "", "0", "", ""],
    ],
    "pt_24_0": [
        ["管理机构", "业务种类", "成立日期", "", "到期日期", "责任人类型", "还款贵任金额", "", "币种", "保证合同编号"],
        ["深圳前海微众 银行股份有限 公司", "爱 贷款", "2022.02.28", "", "囍 2023.02.28", "保证人", "341,000", "", "人民币元", "D10055840H 0001DB2022 0228XS0000 00109"],
        ["主业务借款人", "", "", "主业务借款人证件类型", "", "", "", "主业务借款人证件号码", "", ""],
        ["密 厦门雯明轩商贸有限公司", "", "", "统一社会信用代码", "", "", "", "91350203MA33H1DP8L", "", ""],
        ["? 截至2022年12月31日 司", "", "", "", "", "", "", "", "", ""],
        ["余额", "", "", "五级分类", "", "", "", "逾期月数", "", ""],
        ["258,666", "", "", "正常", "", "", "", "0", "", ""],
    ],
    "pt_24_1": [
        ["管理机构", "业务种类", "成立日期", "", "到期日期", "责任人类型", "还款责任金额", "", "币种", "保证合同编号"],
        ["华能贵诚信托 有限公司", "贷款", "2022.09.02", "", "2024.09.07", "保证人", "福 成 56,000", "", "人民币元", "70105501018 BZYQ202209 02XS0M000 00460"],
        ["主业务借款人", "", "", "主业务借款人证件类型", "", "", "", "主业务借款人证件号码", "", ""],
        ["厦门雯玥轩商贸有限公司", "", "", "饭 中征码", "", "", "", "3502030011973425", "", ""],
        ["截至2023年01月07日", "", "", "", "", "", "", "", "", ""],
        ["余额", "", "", "五级分类", "", "", "", "逾期月数", "", ""],
        ["46,667", "", "", "正常", "", "", "", "\"", "", ""],
    ],
}

LIN_LIABILITY_EXPECTED = {
    "Y10061000H0001EIP1967714G01": (1, "pt_23_4"),
    "D10055840H0001DB20220228XS000000109": (2, "pt_24_0"),
    "70105501018BZYQ20220902XS0M00000460": (3, "pt_24_1"),
}
LIN_LIABILITY_ROW_2_CONTRACT = "D10055840H0001DB20220228XS000000109"
LIN_LIABILITY_ROW_2_STRICT_DISCOVERY_VALUES = {
    "business_type": "爱贷款",
    "related_party_name": "密厦门雯明轩商贸有限公司",
}
LIN_LIABILITY_ROW_2_CANONICAL_RAW = {
    "business_type": "爱 贷款",
    "related_party_name": "密 厦门雯明轩商贸有限公司",
}
LIN_LIABILITY_ROW_2_INVALID_RAW = {
    "due_date": "囍 2023.02.28",
    "snapshot_date": "? 截至2022年12月31日 司",
}
# Approved field-repair targets. The discovery fixture above deliberately keeps
# the damaged source text so both deterministic and independent-OCR paths run.
LIN_LIABILITY_ROW_2_FUTURE_REPAIR_TARGETS = {
    "business_type": "贷款",
    "related_party_name": "厦门雯玥轩商贸有限公司",
}


def _liability_context() -> SimpleNamespace:
    page23 = _page(
        23,
        12,
        tables=(
            _table("pt_23_4", LIN_LIABILITY_ROWS["pt_23_4"], logical_page=23, source_page=12, bbox=(43.5, 428.5, 397.0, 534.0), evidence_start=131),
        ),
    )
    page24 = _page(
        24,
        12,
        tables=(
            _table("pt_24_0", LIN_LIABILITY_ROWS["pt_24_0"], logical_page=24, source_page=12, bbox=(52.5, 48.0, 402.0, 158.5), evidence_start=1),
            _table("pt_24_1", LIN_LIABILITY_ROWS["pt_24_1"], logical_page=24, source_page=12, bbox=(52.5, 191.0, 402.0, 303.0), evidence_start=47),
        ),
    )
    return _context((page23, page24))


def test_lin_liability_assembly_conserves_three_rows_and_defers_row_2_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _liability_context()
    # As with the saved discovery artifact, this replay makes the immutable
    # fixed-layout tables the sole candidate source; the real sealed candidate
    # builder, liability assembler, reconciliation, and issue ledger all run.
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset: []
        if dataset == "repayment_liability_records"
        else [],
    )

    rows = native_extraction._extract_liabilities(context)

    by_contract = {row["contract_number"]: row for row in rows}
    assert len(rows) == len(by_contract) == 3
    assert set(by_contract) == set(LIN_LIABILITY_EXPECTED)
    for contract, (sequence, table_id) in LIN_LIABILITY_EXPECTED.items():
        row = by_contract[contract]
        assert row["sequence"] == sequence
        assert {
            ref.get("table_id")
            for ref in row.get("source_refs") or ()
        } == {table_id}

    row2 = by_contract[LIN_LIABILITY_ROW_2_CONTRACT]
    for field_name, value in LIN_LIABILITY_ROW_2_STRICT_DISCOVERY_VALUES.items():
        assert row2[field_name] == value
    for field_name, value in LIN_LIABILITY_ROW_2_CANONICAL_RAW.items():
        assert row2["canonical_raw"][field_name] == value
    for field_name, value in LIN_LIABILITY_ROW_2_INVALID_RAW.items():
        assert row2["canonical_raw"][field_name] == value
    assert row2["due_date"] is None
    assert row2["snapshot_date"] is None
    assert set(LIN_LIABILITY_ROW_2_INVALID_RAW).issubset(
        row2["_unresolved_fields"]
    )

    deferred_field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("field_name") in LIN_LIABILITY_ROW_2_FUTURE_REPAIR_TARGETS
        and issue.get("target_dataset") == "repayment_liability_records"
        and issue.get("target_record_id") == row2["liability_id"]
    ]
    assert not any(
        issue.get("status") == "resolved" for issue in deferred_field_issues
    )

    unresolved = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("status") != "resolved"
        and issue.get("field_name") in LIN_LIABILITY_ROW_2_INVALID_RAW
        and issue.get("target_dataset") == "repayment_liability_records"
        and issue.get("target_record_id") == row2["liability_id"]
    ]
    assert {issue["field_name"] for issue in unresolved} == set(
        LIN_LIABILITY_ROW_2_INVALID_RAW
    )
    for issue in unresolved:
        assert issue["issue_code"] == "candidate_b_repayment_responsibility_field_invalid"
        assert _one_preserved_raw_text(issue["observed_value"]) == (
            LIN_LIABILITY_ROW_2_INVALID_RAW[issue["field_name"]]
        )
        assert {
            ref.get("table_id") for ref in issue["source_refs"]
        } == {"pt_24_0"}
        assert all(ref.get("geometry_scope") == "cell" for ref in issue["source_refs"])


def test_lin_liability_policy_uses_deterministic_type_and_context_rich_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _liability_context()
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset: []
        if dataset == "repayment_liability_records"
        else [],
    )
    discovery_rows = native_extraction._extract_liabilities(context)
    discovery_by_contract = {
        row["contract_number"]: row for row in discovery_rows
    }
    row2 = discovery_by_contract[LIN_LIABILITY_ROW_2_CONTRACT]
    repair_payload = {
        "repayment_liability_records": [
            {
                "liability_id": row2["liability_id"],
                "business_type": row2["business_type"],
                "related_party_name": row2["related_party_name"],
                "canonical_raw": {
                    field_name: row2["canonical_raw"][field_name]
                    for field_name in LIN_LIABILITY_ROW_2_CANONICAL_RAW
                },
                "source_refs_by_field": {
                    field_name: deepcopy(
                        row2["source_refs_by_field"][field_name]
                    )
                    for field_name in LIN_LIABILITY_ROW_2_CANONICAL_RAW
                },
            }
        ]
    }
    coordinator = BusinessUncertaintyRepairCoordinator(context)
    plan = coordinator.plan(
        repair_payload,
        canonical_audit={"unresolved_pages": []},
    )
    field_repairs = {
        repair.field_name: repair
        for repair in plan.field_repairs
        if repair.dataset_name == "repayment_liability_records"
    }

    assert set(field_repairs) == set(LIN_LIABILITY_ROW_2_FUTURE_REPAIR_TARGETS)
    assert field_repairs["business_type"].mode == "deterministic"
    assert field_repairs["business_type"].candidate_value == "贷款"
    assert field_repairs["related_party_name"].mode == "context_rich_reocr"
    assert field_repairs["related_party_name"].candidate_value is None

    calls = _resolve_with_simulated_page_ocr(
        coordinator,
        plan,
        source_pages=[
            {
                "page": int(page.page_number),
                "source_page": int(page.source_page_number),
                "lines": [],
            }
            for page in context.pages
        ],
        candidate_for=lambda repair: (
            LIN_LIABILITY_ROW_2_FUTURE_REPAIR_TARGETS["related_party_name"]
            if repair.field_name == "related_party_name"
            else None
        ),
    )
    _activate_field_repair(context, plan)
    repaired_rows = native_extraction._extract_liabilities(context)
    repaired_by_contract = {
        row["contract_number"]: row for row in repaired_rows
    }
    repaired_row2 = repaired_by_contract[LIN_LIABILITY_ROW_2_CONTRACT]

    assert len(repaired_rows) == 3
    assert repaired_row2["business_type"] == "贷款"
    assert repaired_row2["related_party_name"] == "厦门雯玥轩商贸有限公司"
    assert {
        field_name: repaired_row2["canonical_raw"][field_name]
        for field_name in LIN_LIABILITY_ROW_2_CANONICAL_RAW
    } == LIN_LIABILITY_ROW_2_CANONICAL_RAW
    assert plan.reconstruction_evidence == {}
    assert calls == [
        ({24}, "business_field_context_rich_reocr_required")
    ]
    for contract in set(LIN_LIABILITY_EXPECTED).difference(
        {LIN_LIABILITY_ROW_2_CONTRACT}
    ):
        assert {
            field_name: repaired_by_contract[contract][field_name]
            for field_name in ("business_type", "related_party_name")
        } == {
            field_name: discovery_by_contract[contract][field_name]
            for field_name in ("business_type", "related_party_name")
        }
def _public_values(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    return dict(normalized) if isinstance(normalized, Mapping) else dict(record)


def _public_source_refs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = record.get("source_refs")
    if not isinstance(refs, (list, tuple)):
        source = record.get("source")
        refs = source.get("source_refs") if isinstance(source, Mapping) else ()
    return [dict(ref) for ref in refs or () if isinstance(ref, Mapping)]


def _public_issue_observed_strings(
    values: Mapping[str, Any],
    issue_evidence: list[dict[str, Any]],
) -> set[str]:
    observed = values.get("observed_value")
    strings: set[str] = set()
    if isinstance(observed, str):
        strings.add(observed)
    elif isinstance(observed, (list, tuple)):
        strings.update(value for value in observed if isinstance(value, str))
    issue_id = values.get("extraction_issue_id")
    strings.update(
        str(evidence["string_value"])
        for evidence in issue_evidence
        if evidence.get("extraction_issue_id") == issue_id
        and evidence.get("evidence_kind") == "observed"
        and evidence.get("string_value") not in (None, "")
    )
    return strings


def _project_lin_collections(
    datasets: dict[str, list[dict[str, Any]]],
    *,
    source_ledger: Mapping[str, Any] | None = None,
    final_dataset_counts: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Cross both production projection boundaries used by the plugin."""

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": (
                {"personal_detail_source_completeness_ledger": deepcopy(source_ledger)}
                if source_ledger is not None
                else {}
            ),
            "datasets": deepcopy(datasets),
        },
        final_dataset_counts=final_dataset_counts,
    )
    return project_personal_detail_datasets(prepared["datasets"])


def test_lin_account_producer_contract_survives_public_canonical_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exact Lin accounts at the terminal public-schema boundary.

    Deliberate-red target: the saved discovery path emitted 42/45 accounts.
    Until materialization is repaired, every omitted exact owner must still be
    public as a source-localized issue rather than a false complete dataset.
    """

    context = _account_context()
    census = native_extraction._sealed_raw_account_population_census(context)
    assert census is not None
    source_ledger = {
        "credit_accounts": 45,
        "account_family_endpoints": {
            **dict(census["endpoints"]),
            "revolving_loan_account": 1,
        },
        "account_family_ordinal_observations": {
            family: {
                str(ordinal): deepcopy(observation)
                for ordinal, observation in observations.items()
            }
            for family, observations in census["ordinal_observations"].items()
        },
    }
    table_observations = [
        _account_table_observation(account_id) for account_id in LIN_ACCOUNT_IDS
    ]
    monkeypatch.setattr(
        native_extraction,
        "_account_anchor_skeletons",
        lambda _context: _account_skeletons_through_card_19(),
    )
    monkeypatch.setattr(
        native_extraction,
        "_repair_complete_account_anchor_skeletons",
        lambda _context, skeletons: list(skeletons),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_table_accounts",
        lambda _context: (deepcopy(table_observations), [], []),
    )

    def exact_owner_matches(
        skeletons: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> dict[int, int]:
        observation_index = {
            row["_expected_account_id"]: index
            for index, row in enumerate(observations)
        }
        return {
            index: observation_index[row["account_id"]]
            for index, row in enumerate(skeletons)
            if row["account_id"] in observation_index
        }

    monkeypatch.setattr(
        native_extraction,
        "_match_account_table_observations",
        exact_owner_matches,
    )
    monkeypatch.setattr(
        native_extraction,
        "_resolve_owned_revolving_table_families",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        native_extraction,
        "_canonical_singleton_account_matches",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_provisional_physical_account_records",
        lambda *_args, **_kwargs: {},
    )

    accounts, _repayments, _events = native_extraction._extract_accounts(context)
    projected = _project_lin_collections(
        {
            "credit_accounts": accounts,
            "personal_detail_extraction_issues": list(
                context._personal_detail_extraction_issues
            ),
        },
        source_ledger=source_ledger,
        final_dataset_counts={"credit_accounts": len(accounts)},
    )
    public_rows = projected.get("credit_accounts") or []
    by_id = {
        str(_public_values(row).get("account_id") or ""): row
        for row in public_rows
    }
    missing_ids = set(LIN_ACCOUNT_IDS) - set(by_id)
    public_issues = [
        (_public_values(row), row)
        for row in projected.get("extraction_issues") or ()
        if isinstance(row, Mapping)
    ]
    for missing_id in missing_ids:
        omission = next(
            (
                (values, record)
                for values, record in public_issues
                if values.get("issue_code") == "source_account_record_omitted"
                and values.get("target_dataset") == "credit_accounts"
                and values.get("target_record_id") == missing_id
                and values.get("field_name") == "account_id"
            ),
            None,
        )
        assert omission is not None, missing_id
        assert len(_public_source_refs(omission[1])) == 1
        assert _public_source_refs(omission[1])[0]["geometry_scope"] == "line"

    # DELIBERATE RED on the frozen 42-row producer; this is the terminal target.
    assert not missing_ids, sorted(missing_ids)
    assert len(public_rows) == len(by_id) == 45
    assert tuple(by_id) == LIN_ACCOUNT_IDS
    for account_id, expected_table_ids in LIN_ACCOUNT_TABLES.items():
        table_ids = tuple(
            ref["table_id"]
            for ref in _public_source_refs(by_id[account_id])
            if ref.get("geometry_scope") == "table" and ref.get("table_id")
        )
        assert table_ids == expected_table_ids, account_id

    for ordinal, expected_currency in LIN_CARD_CURRENCIES.items():
        values = _public_values(
            by_id[f"credit_account:credit_card:{ordinal}"]
        )
        assert values.get("account_currency") == expected_currency
        assert values.get("reporting_amount_currency") == expected_currency
        expected_unit = "yuan" if expected_currency == "CNY" else None
        assert values.get("amount_unit") == expected_unit
        assert values.get("reporting_amount_unit") == expected_unit


def test_lin_agreement_producer_contract_survives_public_canonical_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require all 11 source-owned agreements after public ID projection."""

    context = _agreement_context()
    census = native_extraction._sealed_agreement_population_census(context)
    assert census is not None
    source_ledger = {
        "credit_agreements": 11,
        "credit_agreement_sequence_endpoint": 11,
        "credit_agreement_observed_sequences": list(census["sequences"]),
        "credit_agreement_ordinal_observations": {
            str(sequence): deepcopy(observation)
            for sequence, observation in census["ordinal_observations"].items()
        },
    }
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    agreements = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )
    projected = _project_lin_collections(
        {
            "credit_lines": agreements,
            "personal_detail_extraction_issues": list(
                context._personal_detail_extraction_issues
            ),
        },
        source_ledger=source_ledger,
        final_dataset_counts={"credit_lines": len(agreements)},
    )
    public_rows = projected.get("credit_agreements") or []
    by_sequence = {
        int(_public_values(row)["sequence"]): row
        for row in public_rows
        if _public_values(row).get("sequence") not in (None, "")
    }
    missing_sequences = set(range(1, 12)) - set(by_sequence)
    public_issues = [
        (_public_values(row), row)
        for row in projected.get("extraction_issues") or ()
        if isinstance(row, Mapping)
    ]
    public_issue_evidence = [
        _public_values(row)
        for row in projected.get("extraction_issue_evidence") or ()
        if isinstance(row, Mapping)
    ]
    for sequence in missing_sequences:
        omission = next(
            (
                (values, record)
                for values, record in public_issues
                if values.get("issue_code")
                == "source_credit_agreement_record_omitted"
                and values.get("target_dataset") == "credit_agreements"
                and values.get("target_record_id")
                == f"credit_agreement:{sequence}"
                and values.get("field_name") == "credit_agreement_id"
            ),
            None,
        )
        assert omission is not None, sequence
        refs = _public_source_refs(omission[1])
        assert refs
        assert all(ref.get("geometry_scope") == "line" for ref in refs)

    # DELIBERATE RED on the frozen 8-row producer (missing 3, 5, and 9).
    assert not missing_sequences, sorted(missing_sequences)
    assert len(public_rows) == len(by_sequence) == 11
    for sequence, (identifier, institution, table_id) in LIN_AGREEMENT_EXPECTED.items():
        record = by_sequence[sequence]
        values = _public_values(record)
        assert values["credit_agreement_id"] == stable_record_id(
            "credit_line", identifier
        )
        assert values["account_identifier"] == identifier
        assert {
            ref.get("table_id")
            for ref in _public_source_refs(record)
            if ref.get("geometry_scope") == "table"
        } == {table_id}
        anchor_refs = [
            ref
            for ref in _public_source_refs(record)
            if ref.get("binding") == "canonical_card_anchor"
        ]
        assert len(anchor_refs) == 1
        assert anchor_refs[0]["sequence"] == sequence
        assert anchor_refs[0]["field_name"] == "sequence"
        assert anchor_refs[0]["geometry_scope"] == "text"
        assert (
            anchor_refs[0]["binding_quality"]
            == "printed_credit_agreement_ordinal"
        )
        assert anchor_refs[0]["evidence_ids"]

        if sequence in LIN_AGREEMENT_SAFE_PUBLISHED_INSTITUTIONS:
            assert values["institution"] == institution
            continue

        raw_institution = LIN_AGREEMENT_DEFERRED_INSTITUTION_RAW[sequence]
        assert values["institution"] is None
        field_issues = [
            (issue_values, issue_record)
            for issue_values, issue_record in public_issues
            if issue_values.get("target_record_id")
            == values["credit_agreement_id"]
            and issue_values.get("target_dataset") == "credit_agreements"
            and issue_values.get("field_name") == "institution"
        ]
        assert field_issues, sequence
        assert not any(
            issue_values.get("status") == "resolved"
            for issue_values, _record in field_issues
        )
        assert any(
            raw_institution
            in _public_issue_observed_strings(
                issue_values,
                public_issue_evidence,
            )
            for issue_values, _record in field_issues
        )
        for _issue_values, issue_record in field_issues:
            refs = _public_source_refs(issue_record)
            assert refs
            assert {ref.get("table_id") for ref in refs} == {table_id}
            assert all(ref.get("geometry_scope") == "cell" for ref in refs)


def test_lin_liability_producer_contract_survives_public_canonical_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve strict discovery values and field-local unresolved omissions."""

    context = _liability_context()
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, _dataset: [],
    )

    liabilities = native_extraction._extract_liabilities(context)
    native_deferred_field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("target_dataset") == "repayment_liability_records"
        and issue.get("field_name") in LIN_LIABILITY_ROW_2_FUTURE_REPAIR_TARGETS
    ]
    projected = _project_lin_collections(
        {
            "repayment_liability_records": liabilities,
            "personal_detail_extraction_issues": list(
                context._personal_detail_extraction_issues
            ),
        },
        final_dataset_counts={
            "repayment_liability_records": len(liabilities)
        },
    )
    assert not any(
        issue.get("status") == "resolved"
        for issue in native_deferred_field_issues
    )

    public_rows = projected.get("repayment_responsibilities") or []
    by_contract = {
        str(_public_values(row).get("contract_number") or ""): row
        for row in public_rows
    }
    assert len(public_rows) == len(by_contract) == 3
    assert set(by_contract) == set(LIN_LIABILITY_EXPECTED)
    for contract, (sequence, table_id) in LIN_LIABILITY_EXPECTED.items():
        record = by_contract[contract]
        values = _public_values(record)
        assert values["sequence"] == sequence
        assert values["repayment_responsibility_id"] == stable_record_id(
            "repayment_liability", contract
        )
        assert {
            ref.get("table_id") for ref in _public_source_refs(record)
        } == {table_id}

    row2_record = by_contract[LIN_LIABILITY_ROW_2_CONTRACT]
    row2 = _public_values(row2_record)
    for field_name, value in LIN_LIABILITY_ROW_2_STRICT_DISCOVERY_VALUES.items():
        assert row2[field_name] == value
    assert row2["due_date"] is None
    assert row2["snapshot_date"] is None
    for field_name, value in LIN_LIABILITY_ROW_2_CANONICAL_RAW.items():
        assert row2_record["canonical_raw"][field_name] == value
    for field_name, value in LIN_LIABILITY_ROW_2_INVALID_RAW.items():
        assert row2_record["canonical_raw"][field_name] == value

    issue_rows = [
        (_public_values(record), record)
        for record in projected.get("extraction_issues") or ()
        if isinstance(record, Mapping)
    ]
    row2_id = row2["repayment_responsibility_id"]
    unresolved = [
        (values, record)
        for values, record in issue_rows
        if values.get("issue_code")
        == "candidate_b_repayment_responsibility_field_invalid"
        and values.get("target_dataset") == "repayment_responsibilities"
        and values.get("target_record_id") == row2_id
        and values.get("field_name") in LIN_LIABILITY_ROW_2_INVALID_RAW
    ]
    assert {values["field_name"] for values, _record in unresolved} == set(
        LIN_LIABILITY_ROW_2_INVALID_RAW
    )
    issue_evidence = [
        _public_values(record)
        for record in projected.get("extraction_issue_evidence") or ()
        if isinstance(record, Mapping)
    ]
    for values, record in unresolved:
        assert (
            LIN_LIABILITY_ROW_2_INVALID_RAW[values["field_name"]]
            in _public_issue_observed_strings(values, issue_evidence)
        )
        refs = _public_source_refs(record)
        assert {ref.get("table_id") for ref in refs} == {"pt_24_0"}
        assert all(ref.get("geometry_scope") == "cell" for ref in refs)

    # Repair targets remain metadata during discovery. The extraction strategy
    # must not report either field as repaired before the repair pass runs.
    assert not any(
        values.get("status") == "resolved"
        and values.get("target_dataset") == "repayment_responsibilities"
        and values.get("field_name") in LIN_LIABILITY_ROW_2_FUTURE_REPAIR_TARGETS
        for values, _record in issue_rows
    )


def test_lin_liability_population_fails_closed_on_duplicate_contract_owner() -> None:
    context = _liability_context()
    duplicate = _table(
        "pt_24_0_duplicate",
        LIN_LIABILITY_ROWS["pt_24_0"],
        logical_page=24,
        source_page=12,
        bbox=(52.5, 320.0, 402.0, 430.5),
        evidence_start=801,
    )
    context._frozen_logical_pages[24].tables.append(duplicate)

    assert native_extraction._sealed_liability_table_candidates(context) == []


def test_lin_liability_unknown_header_alias_is_not_promoted() -> None:
    context = _liability_context()
    table = context._frozen_logical_pages[24].tables[0]
    rows = deepcopy(table.metadata["raw_rows"])
    rows[0][1] = "融资品种"
    table.metadata["raw_rows"] = rows

    candidates = native_extraction._sealed_liability_table_candidates(context)

    assert not any(
        candidate.fields.get("保证合同编号")
        == "D10055840H 0001DB2022 0228XS0000 00109"
        for candidate in candidates
    )


def test_lin_strategy_composition_uses_current_extractors_without_strategy_changes() -> None:
    assert stage_names_for_datasets(
        ["credit_accounts", "credit_lines", "repayment_liability_records"]
    ) == ("account_inventory", "credit_agreements", "liabilities")

    account_stage = CANDIDATE_B_STAGE_REGISTRY.stage("account_inventory")
    agreement_stage = CANDIDATE_B_STAGE_REGISTRY.stage("credit_agreements")
    liability_stage = CANDIDATE_B_STAGE_REGISTRY.stage("liabilities")
    assert account_stage.output_names == (
        "credit_accounts",
        "personal_detail_account_events",
    )
    assert agreement_stage.output_names == ("credit_lines",)
    assert agreement_stage.dependencies == frozenset({"account_inventory"})
    assert liability_stage.output_names == ("repayment_liability_records",)
    assert liability_stage.dependencies == frozenset()

    # Freeze the actual production entry points exercised above.  This guards
    # against replacing the extractors while fixing only population/ownership
    # registration and materialization.
    assert native_extraction._extract_accounts.__name__ == "_extract_accounts"
    assert native_extraction._extract_credit_lines.__name__ == "_extract_credit_lines"
    assert native_extraction._extract_liabilities.__name__ == "_extract_liabilities"
