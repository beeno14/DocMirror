# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-owned canonical layout registration for detailed PBOC reports.

The sealed :class:`ParseResult` is source evidence, not the document model used
by the detailed-report extractors.  This module projects its arbitrary logical
fragments onto evidence-registered PBOC semantic roles using static source
evidence and exposes detached pages and tables that all downstream extractors
share.  OCR
acquisition is forbidden here; schema-triggered page repair happens later.

Templates describe semantic page roles and dynamic tables.  They deliberately
do not encode subject names, institution names, account identifiers, or any
other report-specific business value.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from docmirror.plugins.credit_report.personal_detail_scanned.agreement_ocr import (
    CREDIT_AGREEMENT_AMOUNT_LABELS,
    CREDIT_AGREEMENT_CARD_HEADING_RE,
    CREDIT_AGREEMENT_PRIMARY_LABELS,
    CREDIT_AGREEMENT_PURPOSE_LABELS,
)
from docmirror.plugins.credit_report.personal_detail_scanned.layout_profile import (
    exact_inquiry_header_owner,
)
from docmirror.plugins.credit_report.personal_detail_scanned.section_headings import (
    REGISTERED_SECTION_TEMPLATE_BY_TITLE,
    canonical_account_family_heading,
    canonical_registered_section_heading,
    canonical_registered_subsection_heading,
)

_PRINTED_PAGE_RE = re.compile(r"第\s*(?P<page>\d+)\s*页\s*[,，]?\s*共\s*(?P<total>\d+)\s*页")
_MONTHLY_GRID_RE = re.compile(
    r"20\d{2}\s*年\s*\d{1,2}\s*月\s*[-—一至到~～]\s*20\d{2}\s*年\s*\d{1,2}\s*月.*(?:还款|缴费)记录"
)
_ACCOUNT_CARD_HEADING_RE = re.compile(r"^账户\s*(?P<sequence>[1-9]\d{0,2})$")
_INFORMATION_SUMMARY_SUBSECTION_RE = re.compile(
    r"^(?:(?:[（(][\u3007零一二三四五六七八九十百]{1,5}[）)])|"
    r"[\u3007零一二三四五六七八九十百]{1,5})?"
    r"(?P<title>"
    r"信贷交易信息提示|"
    r"信贷交易违约信息概要|"
    r"信贷交易授信及负债信息概要(?:[（(]未结清/未销户[）)])?|"
    r"查询记录概要"
    r")$"
)
_INQUIRY_SEED_SUBSECTION_TITLES = frozenset(
    {
        "查询记录机构查询记录明细",
        "机构查询记录明细",
        "本人查询记录明细",
    }
)
_BOUNDED_INQUIRY_HEADER_RESIDUE_RE = re.compile(
    r"^(?:\?查询日期查询机构X|\?查询日期查询机构|查询日期查询机构X)$"
)
_PRINTED_IDENTITY_UNSPECIFIED = object()
_INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF = (
    "exact_printed_footer_schema_carry_bridge"
)
_LIABILITY_HEADER_ROLE_BY_LABEL = {
    "管理机构": "管理机构",
    "业务种类": "业务种类",
    "开立日期": "开立日期",
    "成立日期": "开立日期",
    "到期日期": "到期日期",
    "责任人类型": "责任人类型",
    "还款责任金额": "还款责任金额",
    "币种": "币种",
    "保证合同编号": "保证合同编号",
}
_LIABILITY_HEADER_ROLES = frozenset(_LIABILITY_HEADER_ROLE_BY_LABEL.values())
_LIABILITY_REQUIRED_VALUE_ROLES = frozenset(
    {
        "管理机构",
        "业务种类",
        "开立日期",
        "到期日期",
        "责任人类型",
        "币种",
    }
)
_LIABILITY_PARTY_ROLES = frozenset(
    {
        "主业务借款人",
        "主业务借款人证件类型",
        "主业务借款人证件号码",
    }
)
_LIABILITY_PARTY_ROLE_SETS_BY_LABEL: Mapping[str, frozenset[str]] = {
    "主业务借款人": frozenset({"主业务借款人"}),
    "民主业务借款人": frozenset({"主业务借款人"}),
    "2主业务借款人": frozenset({"主业务借款人"}),
    "主业务借款人证件类型": frozenset({"主业务借款人证件类型"}),
    "多主业务借款人证件类型": frozenset({"主业务借款人证件类型"}),
    "主业务借款人证件号码": frozenset({"主业务借款人证件号码"}),
    (
        "主业务借款人"
        "主业务借款人证件类型"
        "主业务借款人证件号码"
    ): _LIABILITY_PARTY_ROLES,
    (
        "主业务借款人证件类型"
        "主业务借款人证件号码"
        "主业务借款人"
    ): _LIABILITY_PARTY_ROLES,
    # One compact source card can collapse the three merged header cells into
    # a single exact cell.  This is the observed closed ordering, not a fuzzy
    # substring rule; unknown packed text remains unowned.
    (
        "0主业务借款人证件类型"
        "主业务借款人证件号码"
        "主业务借款人"
    ): _LIABILITY_PARTY_ROLES,
}
_LIABILITY_STATUS_ROLE_BY_LABEL = {
    "余额": "余额",
    "五级分类": "五级分类",
    "五级分类囍": "五级分类",
    "逾期月数": "履约状态",
    "还款状态": "履约状态",
}
_LIABILITY_STATUS_ROLES = frozenset({"余额", "五级分类", "履约状态"})
_LIABILITY_SNAPSHOT_RE = re.compile(
    r"^(?:[0-9人])?截至(?:19|20)\d{2}年\d{1,2}月\d{1,2}日$"
)
_HEADERLESS_SEQUENCE_DATE_CELL_RE = re.compile(
    r"^\s*(?P<sequence>[1-9]\d{0,3})\s+"
    r"(?P<date>(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})\s*$"
)
_ACCOUNT_TABLE_LABELS = frozenset(
    {
        "管理机构",
        "发卡机构",
        "账户标识",
        "开立日期",
        "借款金额",
        "账户授信额度",
        "共享授信额度",
        "币种",
        "账户币种",
        "业务种类",
        "担保方式",
        "到期日期",
    }
)

# A mixed physical page is owned table-by-table.  Each alternative below is a
# semantic PBOC header-role set, not a physical column order, width, row count,
# page number, or report-fixture fingerprint.  Exact source cells and a sealed
# section/subsection heading are additionally required before the role can own
# a table.
_MIXED_PAGE_TABLE_SCHEMAS: Mapping[str, tuple[frozenset[str], ...]] = {
    "postpaid_detail": (
        frozenset(
            {
                "机构名称",
                "业务类型",
                "业务开通日期",
                "当前缴费状态",
                "当前欠费金额",
                "记账年月",
            }
        ),
    ),
    "public_information": (
        frozenset({"编号", "主管税务机关", "欠税总额", "欠税统计日期"}),
        frozenset({"编号", "立案法院", "案由", "立案日期", "结案方式"}),
        frozenset({"编号", "执行法院", "执行案由", "立案日期", "结案方式"}),
        frozenset(
            {
                "编号",
                "处罚机构",
                "处罚内容",
                "处罚金额",
                "生效日期",
                "截止日期",
                "行政复议结果",
            }
        ),
        frozenset(
            {
                "参缴地",
                "参缴日期",
                "初缴月份",
                "缴至月份",
                "缴费状态",
                "月缴存额",
                "个人缴存比例",
                "单位缴存比例",
            }
        ),
        frozenset(
            {
                "编号",
                "执业资格名称",
                "等级",
                "获得日期",
                "到期日期",
                "吊销日期",
                "颁发机构",
                "机构所在地",
            }
        ),
        frozenset({"编号", "奖励机构", "奖励内容", "生效日期", "截止日期"}),
    ),
    "annotations_and_inquiries": (
        frozenset({"编号", "标注内容", "添加日期"}),
    ),
    "credit_agreement": (
        frozenset(
            {
                "管理机构",
                "授信协议标识",
                "生效日期",
                "到期日期",
                "授信额度用途",
            }
        ),
    ),
}


@dataclass(frozen=True)
class CanonicalTemplateSpec:
    template_id: str
    anchor_groups: tuple[tuple[str, ...], ...]
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalLayoutProjection:
    pages: tuple[Any, ...]
    evidence_pages: tuple[dict[str, Any], ...]
    registrations: tuple[dict[str, Any], ...]
    fragment_groups: tuple[dict[str, Any], ...]
    unresolved_pages: tuple[int, ...]

    def audit(self) -> dict[str, Any]:
        return {
            "architecture": "canonical_template_registration_v3_static",
            "parse_result_mutated": False,
            "page_count": len(self.pages),
            "registrations": deepcopy(list(self.registrations)),
            "fragment_groups": deepcopy(list(self.fragment_groups)),
            "unresolved_pages": list(self.unresolved_pages),
            "all_extractors_share_canonical_evidence": True,
            "cell_level_ocr_enabled": False,
            "topology_ocr_free": True,
            "template_registration_ocr_used": False,
            "business_repair_after_schema": True,
        }


_TEMPLATES: tuple[CanonicalTemplateSpec, ...] = (
    CanonicalTemplateSpec(
        "report_header_and_identity",
        (
            ("个人信用报告", "报告编号"),
            ("个人基本信息",),
            ("身份信息", "手机号码"),
            ("居住信息", "职业信息"),
            ("配偶信息",),
            ("工作单位", "单位性质"),
            ("居住地址", "居住状况"),
            ("手机号码", "信息更新日期"),
        ),
        (
            "personal_report_metadata",
            "personal_profile",
            "identity_document_records",
            "mobile_phone_records",
            "spouse_records",
            "residence_records",
            "employment_records",
        ),
    ),
    CanonicalTemplateSpec(
        "information_summary",
        (
            ("信息概要",),
            ("信贷交易信息提示",),
            ("信贷交易授信及负债信息概要",),
            ("查询记录概要",),
            ("账户数", "余额", "授信总额"),
        ),
        (
            "personal_detail_summary_records",
            "personal_detail_summary_cells",
        ),
    ),
    CanonicalTemplateSpec(
        "credit_account_detail",
        (
            ("信贷交易信息明细",),
            ("管理机构", "账户标识"),
            ("发卡机构", "账户标识"),
            ("账户状态", "余额"),
            ("当前逾期期数", "当前逾期总额"),
            ("还款记录",),
            ("被追偿信息",),
            ("非循环贷账户",),
            ("循环贷账户一",),
            ("循环贷账户二",),
            ("贷记卡账户",),
            ("准贷记卡账户",),
        ),
        (
            "credit_accounts",
            "repayment_records",
            "personal_detail_account_events",
            "recovery_account_details",
        ),
    ),
    CanonicalTemplateSpec(
        "repayment_responsibility",
        (
            ("相关还款责任信息",),
            ("责任人类型", "还款责任金额"),
            ("主业务借款人", "保证合同编号"),
        ),
        ("repayment_liability_records",),
    ),
    CanonicalTemplateSpec(
        "credit_agreement",
        (
            ("授信协议信息",),
            ("管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度用途"),
            ("授信额度", "授信限额", "授信限额编号", "已用额度", "币种"),
        ),
        ("credit_lines",),
    ),
    CanonicalTemplateSpec(
        "postpaid_detail",
        (
            ("非信贷交易信息明细",),
            ("后付费记录",),
            ("机构名称", "业务类型", "当前缴费状态"),
            ("缴费记录",),
        ),
        ("postpaid_accounts", "postpaid_payment_history"),
    ),
    CanonicalTemplateSpec(
        "public_information",
        (
            ("公共信息明细",),
            ("欠税记录",),
            ("民事判决记录",),
            ("强制执行记录",),
            ("行政处罚记录",),
            ("住房公积金参缴记录",),
            ("执业资格记录",),
            ("行政奖励记录",),
        ),
        ("public_records",),
    ),
    CanonicalTemplateSpec(
        "annotations_and_inquiries",
        (
            ("异议标注",),
            ("查询记录", "查询日期", "查询机构", "查询原因"),
            ("机构查询记录明细",),
            ("本人查询记录明细",),
        ),
        ("statements", "annotations", "inquiry_records"),
    ),
    CanonicalTemplateSpec(
        "report_explanation",
        (
            ("报告说明",),
            ("还款状态说明",),
            ("贷记卡账户", "还款状态说明"),
            ("后付费业务", "还款状态说明"),
        ),
        ("report_notes",),
    ),
)


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox") if isinstance(value, Mapping) else getattr(value, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    box = tuple(_finite(item) for item in raw[:4])
    return box if box[2] > box[0] and box[3] > box[1] else None


def _affine_pair_valid(forward: Any, inverse: Any) -> bool:
    """Validate an explicit finite, invertible source/logical transform pair."""

    def matrix3(value: Any) -> tuple[tuple[float, float, float], ...] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            matrix = tuple(tuple(float(item) for item in row) for row in value)
        except (TypeError, ValueError):
            return None
        if any(len(row) != 3 for row in matrix):
            return None
        if any(not math.isfinite(item) for row in matrix for item in row):
            return None
        return matrix

    left = matrix3(forward)
    right = matrix3(inverse)
    if left is None or right is None:
        return False
    determinant = left[0][0] * left[1][1] - left[0][1] * left[1][0]
    if abs(determinant) <= 1e-12:
        return False
    products = tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3))
        for row in range(3)
    )
    return all(
        abs(products[row][column] - (1.0 if row == column else 0.0)) <= 1e-5
        for row in range(3)
        for column in range(3)
    )


def _affine_axis_signature(value: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the dominant source-axis mapping, ignoring scale and translation."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        linear = (
            (float(value[0][0]), float(value[0][1])),
            (float(value[1][0]), float(value[1][1])),
        )
    except (IndexError, TypeError, ValueError):
        return None
    if any(not math.isfinite(item) for row in linear for item in row):
        return None
    signature: list[tuple[int, int]] = []
    for source_axis in range(2):
        candidates = (linear[0][source_axis], linear[1][source_axis])
        output_axis = 0 if abs(candidates[0]) >= abs(candidates[1]) else 1
        dominant = candidates[output_axis]
        if abs(dominant) <= 1e-12:
            return None
        signature.append((output_axis, 1 if dominant > 0 else -1))
    return signature[0], signature[1]


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _raw_rows(table: Any) -> list[list[str]]:
    metadata = dict(getattr(table, "metadata", None) or {})
    raw = metadata.get("raw_rows")
    if isinstance(raw, list) and raw:
        return [[str(cell or "").replace("\n", "").strip() for cell in row] for row in raw if isinstance(row, list)]
    rows: list[list[str]] = []
    headers = [str(value or "") for value in getattr(table, "headers", None) or []]
    if headers:
        rows.append(headers)
    for row in getattr(table, "rows", None) or []:
        rows.append([str(getattr(cell, "text", cell) or "") for cell in getattr(row, "cells", None) or []])
    return rows


def _page_text(page: Any, evidence: Mapping[str, Any]) -> str:
    values = [str(line.get("text") or line.get("content") or "") for line in evidence.get("lines") or []]
    for table in getattr(page, "tables", None) or []:
        values.extend(cell for row in _raw_rows(table) for cell in row)
    values.extend(str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or [])
    return "\n".join(value for value in values if value)


def _template_score(text: str, spec: CanonicalTemplateSpec) -> tuple[int, int]:
    compact = _compact(text)
    matched = [group for group in spec.anchor_groups if all(marker in compact for marker in group)]
    strength = sum(len(group) for group in matched)
    return len(matched), strength


def _credit_agreement_card_signature(compact: str) -> bool:
    """Recognize only a numbered agreement card with its printed field schema.

    Some scans confuse the first two glyphs of ``授信`` while the rest of each
    label remains stable.  A finite confusion set plus four
    independent field roles is intentionally stronger than a fuzzy section
    heading or an aggregate page that merely mentions an agreement number.
    """

    if CREDIT_AGREEMENT_CARD_HEADING_RE.search(compact) is None:
        return False
    if not any(label in compact for label in CREDIT_AGREEMENT_PRIMARY_LABELS):
        return False
    supporting_roles = (
        any(label in compact for label in ("管理机构", "营理机构")),
        "生效日期" in compact,
        "到期日期" in compact,
        any(label in compact for label in CREDIT_AGREEMENT_PURPOSE_LABELS),
    )
    return sum(supporting_roles) >= 3


def _classify(text: str) -> tuple[str, float, tuple[str, ...]] | None:
    compact = _compact(text)
    # A canonical top-level heading is stronger evidence than repeated
    # category names.  The latter also occur in information-summary tables
    # and in the report explanation, so allowing them to win by frequency
    # registers otherwise valid pages to the wrong canonical role.
    explicit_roles = (
        ("report_explanation", ("报告说明", "还款状态说明")),
        ("annotations_and_inquiries", ("查询记录明细", "异议标注")),
        ("public_information", ("公共信息明细",)),
        ("postpaid_detail", ("非信贷交易信息明细",)),
        ("repayment_responsibility", ("相关还款责任信息",)),
        ("credit_agreement", ("授信协议信息",)),
        ("credit_account_detail", ("信贷交易信息明细",)),
        ("information_summary", ("信息概要",)),
        ("report_header_and_identity", ("个人基本信息", "个人信用报告")),
    )
    for template_id, headings in explicit_roles:
        matched = tuple(heading for heading in headings if heading in compact)
        if matched:
            return template_id, min(0.99, 0.92 + 0.03 * len(matched)), matched

    # A complete numbered-card schema is independent of how many unrelated OCR
    # characters precede it. Mixed-page ownership is a geometry question and
    # is handled by ``_classify_page``; raw string offsets are not page layout.
    first_agreement_heading = CREDIT_AGREEMENT_CARD_HEADING_RE.search(compact)
    if (
        first_agreement_heading is not None
        and _credit_agreement_card_signature(compact)
    ):
        return "credit_agreement", 0.94, ("numbered_agreement_card_schema",)

    # A dense run of distinct printed card ordinals is a source-local
    # agreement population witness.  Require at least three consecutive cards
    # plus independent primary/limit schemas so a summary mention cannot be
    # promoted into an agreement page.
    agreement_ordinals = {
        int(match.group("sequence"))
        for match in CREDIT_AGREEMENT_CARD_HEADING_RE.finditer(compact)
    }
    if (
        len(agreement_ordinals) >= 3
        and max(agreement_ordinals) - min(agreement_ordinals) + 1 == len(agreement_ordinals)
        and any(label in compact for label in CREDIT_AGREEMENT_PRIMARY_LABELS)
        and any(label in compact for label in CREDIT_AGREEMENT_AMOUNT_LABELS)
    ):
        return "credit_agreement", 0.94, ("dense_numbered_agreement_cards",)

    # A physical page may begin between account cards and therefore omit the
    # enclosing section heading.  The numbered card heading plus its printed
    # agreement-id label is a canonical account-detail signature, not a
    # footer-based or generic previous-page guess.
    if re.search(
        r"\u8d26\u6237\s*\d{1,3}\s*[\uff08(][^\uff09)]{0,40}\u6388\u4fe1\u534f\u8bae\u6807\u8bc6",
        compact,
    ):
        return "credit_account_detail", 0.94, ("canonical_numbered_account_heading",)

    if len(compact) < 8:
        return None

    ranked = sorted(
        (
            (*_template_score(text, spec), spec)
            for spec in _TEMPLATES
        ),
        key=lambda item: (-item[0], -item[1], item[2].template_id),
    )
    count, strength, spec = ranked[0]
    if count <= 0:
        if _MONTHLY_GRID_RE.search(compact):
            return "credit_account_detail", 0.90, ("monthly_grid_signature",)
        return None
    confidence = min(0.99, 0.70 + 0.08 * count + 0.02 * strength)
    signals = tuple("+".join(group) for group in spec.anchor_groups if all(marker in compact for marker in group))
    return spec.template_id, confidence, signals


def _exact_evidence_line_bbox(line: Any) -> tuple[float, float, float, float] | None:
    """Return a finite source-line box only when its evidence atom is sealed."""

    if not isinstance(line, Mapping) or not any(str(value or "") for value in line.get("evidence_ids") or ()):
        return None
    return _bbox(line)


def _sealed_registered_heading_roles(
    page: Any,
    evidence: Mapping[str, Any],
) -> frozenset[str]:
    """Return semantic roles named by sealed, page-contained PBOC headings.

    This is a continuation veto, not an ownership grant.  A registered title
    on the current page is enough to disprove inheritance from a different
    preceding section, but it does not by itself authorize any table.  Table
    ownership still requires the independent geometry/evidence contracts used
    by the canonical assembler.
    """

    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    roles: set[str] = set()
    for line in evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None:
            continue
        if page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width):
            continue
        if page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        title = canonical_registered_section_heading(text)
        if title is not None:
            roles.add(REGISTERED_SECTION_TEMPLATE_BY_TITLE[title])
            continue
        subsection = canonical_registered_subsection_heading(text)
        if subsection is not None:
            roles.add(subsection[0])
            continue
        if canonical_account_family_heading(text) is not None:
            roles.add("credit_account_detail")
    return frozenset(roles)


def _exact_table_cell_owner(
    table: Any,
    *,
    row: int,
    column: int,
) -> tuple[tuple[float, float, float, float], tuple[str, ...]] | None:
    """Resolve one immutable exact native-table cell owner.

    The canonical assembler may receive either a raw native table (geometry
    nested under ``metadata.geometry``) or a detached canonical table (the same
    matrices promoted to metadata).  Conflicting copies are rejected rather
    than resolved by preference.
    """

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    geometry = metadata.get("geometry")
    grids = [owner for owner in (geometry, metadata) if isinstance(owner, Mapping)]
    candidates: set[tuple[tuple[float, float, float, float], tuple[str, ...]]] = set()
    for owner in grids:
        statuses = owner.get("cell_geometry_status")
        bboxes = owner.get("cell_bboxes")
        evidence_ids = owner.get("cell_evidence_ids")
        if not all(isinstance(grid, list) for grid in (statuses, bboxes, evidence_ids)):
            continue
        if any(
            row < 0
            or row >= len(grid)
            or not isinstance(grid[row], list)
            or column < 0
            or column >= len(grid[row])
            for grid in (statuses, bboxes, evidence_ids)
        ):
            continue
        if str(statuses[row][column] or "") != "exact":
            continue
        raw_bbox = bboxes[row][column]
        bbox = (
            tuple(_finite(value) for value in raw_bbox[:4])
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4
            else None
        )
        if bbox is not None and not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
            bbox = None
        raw_ids = evidence_ids[row][column]
        if bbox is None or not isinstance(raw_ids, list) or not raw_ids:
            continue
        if any(not isinstance(value, str) or not value.strip() for value in raw_ids):
            continue
        sealed = tuple(value.strip() for value in raw_ids)
        if len(sealed) != len(set(sealed)):
            continue
        candidates.add((bbox, sealed))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _boxes_have_disjoint_interiors(
    boxes: Iterable[tuple[float, float, float, float]],
) -> bool:
    ordered = tuple(boxes)
    return all(
        max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        <= 1e-8
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )


def _exact_inquiry_table_schema(table: Any) -> dict[str, Any] | None:
    """Bind one PBOC inquiry table through the shared exact header owner."""

    table_id = str(getattr(table, "table_id", "") or "")
    table_bbox = _bbox(table)
    rows = _raw_rows(table)
    header_owner = exact_inquiry_header_owner(table)
    if not table_id or table_bbox is None or header_owner is None:
        return None
    columns = header_owner.columns()
    if set(columns) != {"sequence", "inquiry_date", "institution", "reason"}:
        return None

    # A schema-shaped header is not enough to own business records.  The first
    # non-empty body row must itself be a fully populated exact witness; an
    # unrelated or partial row cannot be skipped in search of a later match.
    # Later incomplete OCR observations are conserved downstream and do not
    # decide canonical ownership or table cardinality.
    populated_body_rows = [
        row_index
        for row_index in range(header_owner.body_start, len(rows))
        if any(_compact(cell) for cell in rows[row_index])
    ]
    if not populated_body_rows:
        return None
    witness_row = populated_body_rows[0]
    row = rows[witness_row]
    if any(
        column >= len(row) or not _compact(row[column])
        for column in columns.values()
    ):
        return None
    header_ids = set(header_owner.evidence_ids)
    value_owners = [
        _exact_table_cell_owner(table, row=witness_row, column=column)
        for column in columns.values()
    ]
    if any(owner is None for owner in value_owners):
        return None
    exact_values = [owner for owner in value_owners if owner is not None]
    witness_ids = [
        evidence_id
        for _bbox_value, evidence_ids in exact_values
        for evidence_id in evidence_ids
    ]
    witness_boxes = [bbox for bbox, _evidence_ids in exact_values]
    if (
        len(witness_ids) != len(set(witness_ids))
        or header_ids.intersection(witness_ids)
        or not _boxes_have_disjoint_interiors(witness_boxes)
        or any(
            box[0] < table_bbox[0] - 1e-6
            or box[1] < table_bbox[1] - 1e-6
            or box[2] > table_bbox[2] + 1e-6
            or box[3] > table_bbox[3] + 1e-6
            for box in witness_boxes
        )
    ):
        return None
    return {
        "template_id": "annotations_and_inquiries",
        "table_id": table_id,
        "header_row": header_owner.header_rows[0],
        "header_rows": list(header_owner.header_rows),
        "population_witness_row": witness_row,
        "header_labels": list(header_owner.header_labels),
        "inquiry_role_columns": dict(header_owner.inquiry_role_columns),
        "evidence_ids": sorted({*header_owner.evidence_ids, *witness_ids}),
        "table_bbox": list(table_bbox),
        "header_binding": header_owner.binding,
    }


def _bounded_inquiry_seed_table_schema(table: Any) -> dict[str, Any] | None:
    """Own one first inquiry table with a narrowly damaged colspan header.

    Some scanned reports preserve the four immutable header roles and their
    exact two-column span while adding one or two ASCII OCR atoms around the
    merged ``查询日期/查询机构`` label.  The shared exact-header
    contract correctly rejects that text.  Canonical registration may still
    seed the inquiry chain when the residue is from the observed finite set,
    the colspan geometry is exact, and every physical body row has exact
    four-cell ownership.  First/last sequence endpoints close the physical row
    count; interior sequence OCR anomalies stay raw and field-local instead of
    being normalized from row order.  Institution/reason values never select
    the owner.
    """

    table_id = str(getattr(table, "table_id", "") or "")
    table_bbox = _bbox(table)
    rows = _raw_rows(table)
    if not table_id or table_bbox is None or len(rows) < 4 or any(len(row) != 4 for row in rows):
        return None
    header = tuple(_compact(value) for value in rows[0])
    if (
        header[0] != "编号"
        or header[2]
        or header[3] != "查询原因"
        or _BOUNDED_INQUIRY_HEADER_RESIDUE_RE.fullmatch(header[1]) is None
    ):
        return None

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    geometry_views = [
        owner
        for owner in (metadata.get("geometry"), metadata)
        if isinstance(owner, Mapping)
        and all(
            isinstance(owner.get(key), list)
            for key in (
                "cell_geometry_status",
                "cell_bboxes",
                "cell_evidence_ids",
            )
        )
    ]
    if not geometry_views:
        return None
    for geometry in geometry_views:
        statuses = geometry["cell_geometry_status"]
        bboxes = geometry["cell_bboxes"]
        evidence_ids = geometry["cell_evidence_ids"]
        if any(
            not grid or not isinstance(grid[0], list) or len(grid[0]) != 4 for grid in (statuses, bboxes, evidence_ids)
        ):
            return None
        if tuple(str(value or "") for value in statuses[0]) != (
            "exact",
            "exact",
            "derived",
            "exact",
        ):
            return None
        if bboxes[0][2] is not None or evidence_ids[0][2] not in ([], None):
            return None
        spans = geometry.get("cell_spans")
        matching_spans = [
            span
            for span in spans or ()
            if isinstance(span, Mapping)
            and span.get("row") == 0
            and span.get("col") == 1
            and span.get("row_span") == 1
            and span.get("col_span") == 2
        ]
        if len(matching_spans) != 1:
            return None
        merged_ids = tuple(str(value) for value in evidence_ids[0][1] or () if str(value or ""))
        span_ids = tuple(str(value) for value in matching_spans[0].get("evidence_ids") or () if str(value or ""))
        if (
            not merged_ids
            or len(merged_ids) != len(set(merged_ids))
            or span_ids != merged_ids
            or _bbox({"bbox": matching_spans[0].get("bbox")}) != _bbox({"bbox": bboxes[0][1]})
        ):
            return None

    header_owners = [_exact_table_cell_owner(table, row=0, column=column) for column in (0, 1, 3)]
    if any(owner is None for owner in header_owners):
        return None
    exact_header_owners = [owner for owner in header_owners if owner is not None]

    populated_rows = [
        row_index for row_index in range(1, len(rows)) if any(_compact(value) for value in rows[row_index])
    ]
    if populated_rows != list(range(1, len(rows))):
        return None
    if (
        _compact(rows[populated_rows[0]][0]) != "1"
        or _compact(rows[populated_rows[-1]][0]) != str(len(populated_rows))
    ):
        return None
    date_re = re.compile(r"(?:19|20)\d{2}[.,/-]\d{1,2}[.,/-]\d{1,2}")
    for row_index in populated_rows:
        row = rows[row_index]
        raw_date = _compact(row[1])
        date_matches = tuple(date_re.finditer(raw_date))
        date_residue = (
            raw_date[: date_matches[0].start()] + raw_date[date_matches[0].end() :]
            if len(date_matches) == 1
            else ""
        )
        if (
            len(date_matches) != 1
            or len(date_residue) > 2
            or any(character.isdigit() or character in ".,/-" for character in date_residue)
            or not _compact(row[2])
            or not _compact(row[3])
        ):
            return None

    def exact_empty_sequence_slot(row_index: int) -> tuple[float, float, float, float] | None:
        candidates: set[tuple[float, float, float, float]] = set()
        for geometry in geometry_views:
            statuses = geometry["cell_geometry_status"]
            bboxes = geometry["cell_bboxes"]
            evidence_ids = geometry["cell_evidence_ids"]
            if any(
                row_index >= len(grid)
                or not isinstance(grid[row_index], list)
                or not grid[row_index]
                for grid in (statuses, bboxes, evidence_ids)
            ):
                return None
            bbox = _bbox({"bbox": bboxes[row_index][0]})
            if (
                str(statuses[row_index][0] or "") != "exact"
                or bbox is None
                or evidence_ids[row_index][0] not in ([], None)
            ):
                return None
            token_ids = geometry.get("cell_token_ids")
            if (
                isinstance(token_ids, list)
                and (
                    row_index >= len(token_ids)
                    or not isinstance(token_ids[row_index], list)
                    or not token_ids[row_index]
                    or token_ids[row_index][0] not in ([], None)
                )
            ):
                return None
            if any(
                isinstance(span, Mapping)
                and isinstance(span.get("row"), int)
                and isinstance(span.get("col"), int)
                and isinstance(span.get("row_span"), int)
                and isinstance(span.get("col_span"), int)
                and int(span["row"]) <= row_index < int(span["row"]) + int(span["row_span"])
                and int(span["col"]) <= 0 < int(span["col"]) + int(span["col_span"])
                for span in geometry.get("cell_spans") or ()
            ):
                return None
            candidates.add(bbox)
        return next(iter(candidates)) if len(candidates) == 1 else None

    non_sequence_owners = [
        _exact_table_cell_owner(table, row=row_index, column=column)
        for row_index in populated_rows
        for column in (1, 2, 3)
    ]
    if any(owner is None for owner in non_sequence_owners):
        return None
    exact_body_owners = [owner for owner in non_sequence_owners if owner is not None]
    empty_sequence_boxes: list[tuple[float, float, float, float]] = []
    sequence_anomalies: list[dict[str, Any]] = []
    physical_field_omission_rows: list[int] = []
    for expected_sequence, row_index in enumerate(populated_rows, start=1):
        raw_sequence = _compact(rows[row_index][0])
        if raw_sequence:
            owner = _exact_table_cell_owner(table, row=row_index, column=0)
            if owner is None:
                return None
            exact_body_owners.append(owner)
        else:
            empty_box = exact_empty_sequence_slot(row_index)
            if empty_box is None:
                return None
            empty_sequence_boxes.append(empty_box)
            physical_field_omission_rows.append(row_index)
        if raw_sequence != str(expected_sequence):
            sequence_anomalies.append(
                {
                    "row": row_index,
                    "expected_sequence": expected_sequence,
                    "raw_sequence": raw_sequence,
                    "status": (
                        "physical_field_omission"
                        if not raw_sequence
                        else "unparsed_raw_sequence"
                    ),
                }
            )
    all_owners = [*exact_header_owners, *exact_body_owners]
    owner_ids = [evidence_id for _owner_bbox, evidence_ids in all_owners for evidence_id in evidence_ids]
    owner_boxes = [
        *(owner_bbox for owner_bbox, _evidence_ids in all_owners),
        *empty_sequence_boxes,
    ]
    if (
        len(owner_ids) != len(set(owner_ids))
        or not _boxes_have_disjoint_interiors(owner_boxes)
        or any(
            box[0] < table_bbox[0] - 1e-6
            or box[1] < table_bbox[1] - 1e-6
            or box[2] > table_bbox[2] + 1e-6
            or box[3] > table_bbox[3] + 1e-6
            for box in owner_boxes
        )
    ):
        return None
    return {
        "template_id": "annotations_and_inquiries",
        "table_id": table_id,
        "header_row": 0,
        "header_rows": [0],
        "population_witness_row": populated_rows[0],
        "population_endpoint_row": populated_rows[-1],
        "population_start": 1,
        "population_endpoint": len(populated_rows),
        "header_labels": ["编号", "查询日期", "查询机构", "查询原因"],
        "inquiry_role_columns": {
            "sequence": 0,
            "inquiry_date": 1,
            "institution": 2,
            "reason": 3,
        },
        "evidence_ids": sorted(owner_ids),
        "table_bbox": list(table_bbox),
        "header_binding": "exact_bounded_residue_collapsed_header_lattice",
        "physical_field_omission_rows": physical_field_omission_rows,
        "sequence_field_anomalies": sequence_anomalies,
    }


def _sealed_inquiry_seed_table_owner(
    page: Any,
    evidence: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Bind one damaged first inquiry table to an exact local subsection."""

    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    headings: list[
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[str, ...],
        ]
    ] = []
    for line in evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None:
            continue
        if page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width):
            continue
        if page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height):
            continue
        subsection = canonical_registered_subsection_heading(_compact(line.get("text") or line.get("content") or ""))
        if (
            subsection is None
            or subsection[0] != "annotations_and_inquiries"
            or subsection[1] not in _INQUIRY_SEED_SUBSECTION_TITLES
        ):
            continue
        heading_ids = tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
        if not heading_ids or len(heading_ids) != len(set(heading_ids)):
            return None
        headings.append((subsection[1], bbox, heading_ids))
    if len(headings) != 1:
        return None
    heading_title, heading_bbox, heading_ids = headings[0]
    candidates: list[tuple[str, dict[str, Any]]] = []
    for table in getattr(page, "tables", None) or ():
        table_bbox = _bbox(table)
        schema = _bounded_inquiry_seed_table_schema(table)
        if table_bbox is None or schema is None:
            continue
        horizontal_overlap = max(
            0.0,
            min(heading_bbox[2], table_bbox[2]) - max(heading_bbox[0], table_bbox[0]),
        )
        if horizontal_overlap <= 0.0 or not _heading_attaches_to_table(
            heading_bbox,
            table_bbox,
        ):
            continue
        if set(heading_ids).intersection(schema["evidence_ids"]):
            return None
        candidates.append(
            (
                str(schema["table_id"]),
                {
                    **schema,
                    "heading_title": heading_title,
                    "heading_bbox": list(heading_bbox),
                    "heading_evidence_ids": list(heading_ids),
                    "binding": ("exact_inquiry_subsection_and_bounded_header_residue"),
                },
            )
        )
    return candidates[0] if len(candidates) == 1 else None


def _exact_mixed_page_table_schema(
    table: Any,
) -> dict[str, Any] | None:
    """Identify one table role from exact PBOC header and populated cells."""

    table_id = str(getattr(table, "table_id", "") or "")
    table_bbox = _bbox(table)
    rows = _raw_rows(table)
    if not table_id or table_bbox is None or len(rows) < 2:
        return None
    candidates: list[dict[str, Any]] = []
    inquiry_candidate = _exact_inquiry_table_schema(table)
    if inquiry_candidate is not None:
        candidates.append(inquiry_candidate)
    for row_index, row in enumerate(rows[:-1]):
        populated = [(column, _compact(value)) for column, value in enumerate(row) if _compact(value)]
        labels = tuple(value for _column, value in populated)
        if not labels or len(labels) != len(set(labels)):
            continue
        label_set = frozenset(labels)
        matching_roles = {
            template_id
            for template_id, schemas in _MIXED_PAGE_TABLE_SCHEMAS.items()
            if label_set in schemas
        }
        if len(matching_roles) != 1:
            continue
        header_owners = [
            _exact_table_cell_owner(table, row=row_index, column=column)
            for column, _label in populated
        ]
        if any(owner is None for owner in header_owners):
            continue
        exact_headers = [owner for owner in header_owners if owner is not None]
        header_ids = [value for _bbox_value, ids in exact_headers for value in ids]
        header_boxes = [bbox for bbox, _ids in exact_headers]
        if (
            len(header_ids) != len(set(header_ids))
            or not _boxes_have_disjoint_interiors(header_boxes)
            or any(
                bbox[0] < table_bbox[0] - 1e-6
                or bbox[1] < table_bbox[1] - 1e-6
                or bbox[2] > table_bbox[2] + 1e-6
                or bbox[3] > table_bbox[3] + 1e-6
                for bbox in header_boxes
            )
        ):
            continue

        # A header lattice is a schema, not proof that a source record exists.
        # Require one later row with exact, non-empty owners in every semantic
        # header column.  The values themselves never influence the role.
        value_proofs: list[tuple[int, list[tuple[tuple[float, float, float, float], tuple[str, ...]]]]] = []
        for value_row_index in range(row_index + 1, len(rows)):
            value_row = rows[value_row_index]
            if any(column >= len(value_row) or not _compact(value_row[column]) for column, _label in populated):
                continue
            value_owners = [
                _exact_table_cell_owner(table, row=value_row_index, column=column)
                for column, _label in populated
            ]
            if any(owner is None for owner in value_owners):
                continue
            exact_values = [owner for owner in value_owners if owner is not None]
            value_ids = [value for _bbox_value, ids in exact_values for value in ids]
            value_boxes = [bbox for bbox, _ids in exact_values]
            if (
                len(value_ids) != len(set(value_ids))
                or set(value_ids).intersection(header_ids)
                or not _boxes_have_disjoint_interiors(value_boxes)
            ):
                continue
            value_proofs.append((value_row_index, exact_values))
        if len(value_proofs) != 1:
            # Multiple populated rows are legitimate for inquiry/public lists;
            # choose the first physical row only after proving that every such
            # row has independent exact ownership.  Ambiguous ownership, not
            # record cardinality, is the veto.
            if not value_proofs:
                continue
            value_proofs.sort(key=lambda item: item[0])
        value_row_index, exact_values = value_proofs[0]
        value_ids = [value for _bbox_value, ids in exact_values for value in ids]
        candidates.append(
            {
                "template_id": next(iter(matching_roles)),
                "table_id": table_id,
                "header_row": row_index,
                "population_witness_row": value_row_index,
                "header_labels": list(labels),
                "evidence_ids": sorted({*header_ids, *value_ids}),
                "table_bbox": list(table_bbox),
            }
        )

    # Some official PBOC inquiry tables split one semantic header across
    # complementary ruled rows (for example, a vertically spanning sequence
    # label beside the remaining three labels).  Derive the role map from the
    # exact lattice instead of requiring one physical header row.  Every
    # semantic role must occur exactly once, no foreign text may remain, and a
    # later complete exact population row is still mandatory.
    for start_row in range(len(rows) - 2):
        for end_row in range(start_row + 1, len(rows) - 1):
            width = max(len(row) for row in rows[start_row : end_row + 1])
            if any(
                not any(_compact(value) for value in rows[row_index])
                for row_index in range(start_row, end_row + 1)
            ):
                continue
            populated: list[tuple[int, int, str]] = []
            labels_by_column: dict[int, list[str]] = {}
            for row_index in range(start_row, end_row + 1):
                for column, value in enumerate(rows[row_index]):
                    label = _compact(value)
                    if not label:
                        continue
                    populated.append((row_index, column, label))
                    labels_by_column.setdefault(column, []).append(label)
            if (
                not populated
                or len(labels_by_column) != width
                or any(len(values) != 1 for values in labels_by_column.values())
            ):
                continue
            labels = tuple(labels_by_column[column][0] for column in range(width))
            if len(labels) != len(set(labels)):
                continue
            label_set = frozenset(labels)
            matching_roles = {
                template_id
                for template_id, schemas in _MIXED_PAGE_TABLE_SCHEMAS.items()
                if label_set in schemas
            }
            if len(matching_roles) != 1:
                continue
            header_owners = [
                _exact_table_cell_owner(table, row=row_index, column=column)
                for row_index, column, _label in populated
            ]
            if any(owner is None for owner in header_owners):
                continue
            exact_headers = [owner for owner in header_owners if owner is not None]
            header_ids = [
                value
                for _bbox_value, ids in exact_headers
                for value in ids
            ]
            header_boxes = [bbox for bbox, _ids in exact_headers]
            if (
                len(header_ids) != len(set(header_ids))
                or not _boxes_have_disjoint_interiors(header_boxes)
                or any(
                    bbox[0] < table_bbox[0] - 1e-6
                    or bbox[1] < table_bbox[1] - 1e-6
                    or bbox[2] > table_bbox[2] + 1e-6
                    or bbox[3] > table_bbox[3] + 1e-6
                    for bbox in header_boxes
                )
            ):
                continue
            value_proofs: list[
                tuple[
                    int,
                    list[
                        tuple[
                            tuple[float, float, float, float],
                            tuple[str, ...],
                        ]
                    ],
                ]
            ] = []
            for value_row_index in range(end_row + 1, len(rows)):
                value_row = rows[value_row_index]
                if len(value_row) != width or any(
                    not _compact(value_row[column]) for column in range(width)
                ):
                    continue
                value_owners = [
                    _exact_table_cell_owner(
                        table,
                        row=value_row_index,
                        column=column,
                    )
                    for column in range(width)
                ]
                if any(owner is None for owner in value_owners):
                    continue
                exact_values = [
                    owner for owner in value_owners if owner is not None
                ]
                value_ids = [
                    value
                    for _bbox_value, ids in exact_values
                    for value in ids
                ]
                value_boxes = [bbox for bbox, _ids in exact_values]
                if (
                    len(value_ids) != len(set(value_ids))
                    or set(value_ids).intersection(header_ids)
                    or not _boxes_have_disjoint_interiors(value_boxes)
                ):
                    continue
                value_proofs.append((value_row_index, exact_values))
            if not value_proofs:
                continue
            value_proofs.sort(key=lambda item: item[0])
            value_row_index, exact_values = value_proofs[0]
            value_ids = [
                value for _bbox_value, ids in exact_values for value in ids
            ]
            candidates.append(
                {
                    "template_id": next(iter(matching_roles)),
                    "table_id": table_id,
                    "header_row": start_row,
                    "header_rows": list(range(start_row, end_row + 1)),
                    "population_witness_row": value_row_index,
                    "header_labels": list(labels),
                    "evidence_ids": sorted({*header_ids, *value_ids}),
                    "table_bbox": list(table_bbox),
                    "header_binding": "exact_complementary_header_lattice",
                }
            )
    return candidates[0] if len(candidates) == 1 else None


def _mixed_page_section_table_owners(
    page: Any,
    evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return independently proved table-local PBOC section owners.

    Cardinality is deliberately not decided here.  A single local owner may
    combine with a reciprocally proved cross-page card continuation to
    establish a mixed page; without that independent second role, the caller
    still withholds the otherwise unknown page.
    """

    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    table_boxes = tuple(
        box
        for table in getattr(page, "tables", None) or ()
        if (box := _bbox(table)) is not None
    )
    headings: list[dict[str, Any]] = []
    heading_evidence_ids: set[str] = set()
    heading_owner_conflict = False
    for line in evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None or any(_line_in_box(line, table_box) for table_box in table_boxes):
            continue
        if page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width):
            continue
        if page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        title = canonical_registered_section_heading(text)
        if title is not None:
            template_id = REGISTERED_SECTION_TEMPLATE_BY_TITLE[title]
        else:
            subsection = canonical_registered_subsection_heading(text)
            if subsection is not None:
                template_id, title = subsection
            elif canonical_account_family_heading(text) is not None:
                template_id, title = "credit_account_detail", text
            else:
                continue
        owners = tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
        if not owners or len(owners) != len(set(owners)) or heading_evidence_ids.intersection(owners):
            heading_owner_conflict = True
            continue
        heading_evidence_ids.update(owners)
        headings.append(
            {
                "template_id": template_id,
                "title": title,
                "bbox": bbox,
                "evidence_ids": owners,
            }
        )
    if heading_owner_conflict or not headings:
        return {}

    tables = tuple(getattr(page, "tables", None) or ())
    table_ids = [str(getattr(table, "table_id", "") or "") for table in tables]
    if not all(table_ids) or len(table_ids) != len(set(table_ids)):
        return {}
    schemas = {
        str(getattr(table, "table_id", "") or ""): _exact_mixed_page_table_schema(table)
        for table in tables
    }
    candidates: dict[str, dict[str, Any]] = {}
    consumed_evidence_ids: set[str] = set(heading_evidence_ids)
    for table in sorted(tables, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0])):
        table_id = str(getattr(table, "table_id", "") or "")
        schema = schemas.get(table_id)
        table_bbox = _bbox(table)
        if schema is None or table_bbox is None:
            continue
        preceding = [
            heading
            for heading in headings
            if _heading_attaches_to_table(heading["bbox"], table_bbox)
        ]
        if not preceding:
            continue
        nearest_bottom = max(heading["bbox"][3] for heading in preceding)
        nearest = [
            heading
            for heading in preceding
            if math.isclose(heading["bbox"][3], nearest_bottom, rel_tol=1e-7, abs_tol=1e-6)
        ]
        if len(nearest) != 1 or nearest[0]["template_id"] != schema["template_id"]:
            continue
        evidence_ids = tuple(schema["evidence_ids"])
        if consumed_evidence_ids.intersection(evidence_ids):
            continue
        consumed_evidence_ids.update(evidence_ids)
        heading = nearest[0]
        candidates[table_id] = {
            **schema,
            "heading_title": heading["title"],
            "heading_bbox": list(heading["bbox"]),
            "heading_evidence_ids": list(heading["evidence_ids"]),
            "binding": "exact_pboc_section_heading_and_table_schema",
        }

    return candidates


def _cross_page_agreement_table_owner(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Bind one leading agreement table to a terminal prior-page card anchor.

    PBOC cards may split exactly between their numbered heading and table.  The
    proof is reciprocal: one terminal sealed ``授信协议 N`` heading must be the
    only unmatched card anchor on the prior page, and one leading exact
    agreement table must be the only table before the first current-page
    registered section.  No page number, sample ordinal, or adjacency alone is
    authority.
    """

    prior_tables = tuple(getattr(previous_page, "tables", None) or ())
    current_tables = tuple(getattr(current_page, "tables", None) or ())
    if not prior_tables or not current_tables:
        return None
    prior_printed = _printed_identity(previous_evidence)
    current_printed = _printed_identity(current_evidence)
    if (
        prior_printed is None
        or current_printed is None
        or prior_printed[1] != current_printed[1]
        or current_printed[0] != prior_printed[0] + 1
    ):
        return None
    agreement_section_owners = [
        tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
        for line in previous_evidence.get("lines") or ()
        if _exact_evidence_line_bbox(line) is not None
        and canonical_registered_section_heading(
            _compact(line.get("text") or line.get("content") or "")
        )
        == "授信协议信息"
    ]
    if (
        len(agreement_section_owners) != 1
        or not agreement_section_owners[0]
        or len(agreement_section_owners[0]) != len(set(agreement_section_owners[0]))
    ):
        return None
    prior_agreement_tables = [
        (table, schema, box)
        for table in prior_tables
        if (box := _bbox(table)) is not None
        and (schema := _exact_mixed_page_table_schema(table)) is not None
        and schema["template_id"] == "credit_agreement"
    ]
    prior_table_boxes = [box for _table, _schema, box in prior_agreement_tables]
    prior_table_ids = [str(getattr(table, "table_id", "") or "") for table, _schema, _box in prior_agreement_tables]
    prior_table_evidence = [
        evidence_id
        for _table, schema, _box in prior_agreement_tables
        for evidence_id in schema["evidence_ids"]
    ]
    if (
        not all(prior_table_ids)
        or len(prior_table_ids) != len(set(prior_table_ids))
        or len(prior_table_evidence) != len(set(prior_table_evidence))
    ):
        return None
    current_table_boxes = [box for table in current_tables if (box := _bbox(table)) is not None]
    if len(current_table_boxes) != len(current_tables):
        return None

    anchors: list[dict[str, Any]] = []
    for line in previous_evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        text = _compact(line.get("text") or line.get("content") or "") if isinstance(line, Mapping) else ""
        match = CREDIT_AGREEMENT_CARD_HEADING_RE.fullmatch(text)
        if bbox is None or match is None:
            continue
        owners = tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
        if not owners or len(owners) != len(set(owners)):
            return None
        anchors.append(
            {
                "sequence": int(match.group("sequence")),
                "bbox": bbox,
                "evidence_ids": owners,
            }
        )
    if len(anchors) < 2 or len({item["sequence"] for item in anchors}) != len(anchors):
        return None
    anchors.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    anchor_evidence = [
        evidence_id for anchor in anchors for evidence_id in anchor["evidence_ids"]
    ]
    if (
        len(anchor_evidence) != len(set(anchor_evidence))
        or set(anchor_evidence).intersection(prior_table_evidence)
        or set(agreement_section_owners[0]).intersection(
            {*anchor_evidence, *prior_table_evidence}
        )
    ):
        return None
    sequences = [int(item["sequence"]) for item in anchors]
    if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
        return None
    matched_anchor_indexes: set[int] = set()
    for index, anchor in enumerate(anchors):
        next_top = anchors[index + 1]["bbox"][1] if index + 1 < len(anchors) else math.inf
        matches = [
            box
            for box in prior_table_boxes
            if _heading_attaches_to_table(anchor["bbox"], box)
            and box[3] <= next_top
        ]
        if len(matches) == 1:
            matched_anchor_indexes.add(index)
    unmatched = [
        anchor for index, anchor in enumerate(anchors) if index not in matched_anchor_indexes
    ]
    if (
        len(unmatched) != 1
        or unmatched[0] is not anchors[-1]
        or len(matched_anchor_indexes) != len(anchors) - 1
    ):
        return None
    # The unmatched card must be the terminal business heading. Footer furniture
    # is ignored only through the same narrow bottom-footer geometry contract
    # used by printed identity resolution.
    prior_height = previous_evidence.get("page_height") or getattr(previous_page, "height", 0)
    following_business_lines = [
        line
        for line in previous_evidence.get("lines") or ()
        if (bbox := _exact_evidence_line_bbox(line)) is not None
        and bbox[1] >= unmatched[0]["bbox"][3]
        and not (
            _bottom_footer_geometry(line, page_height=prior_height)
            and _PRINTED_PAGE_RE.search(
                str(line.get("text") or line.get("content") or "")
            )
            is not None
        )
    ]
    if following_business_lines:
        return None

    section_heading_tops = [
        bbox[1]
        for line in current_evidence.get("lines") or ()
        if (bbox := _exact_evidence_line_bbox(line)) is not None
        and canonical_registered_section_heading(
            _compact(line.get("text") or line.get("content") or "")
        )
        is not None
    ]
    if not section_heading_tops:
        return None
    first_section_top = min(section_heading_tops)
    leading_tables = [
        table
        for table in current_tables
        if (box := _bbox(table)) is not None and box[3] <= first_section_top
    ]
    if len(leading_tables) != 1:
        return None
    leading = leading_tables[0]
    schema = _exact_mixed_page_table_schema(leading)
    if schema is None or schema["template_id"] != "credit_agreement":
        return None
    if set(schema["evidence_ids"]).intersection(
        {
            *agreement_section_owners[0],
            *anchor_evidence,
            *prior_table_evidence,
        }
    ):
        return None
    table_id = str(getattr(leading, "table_id", "") or "")
    return table_id, {
        **schema,
        "printed_sequence": unmatched[0]["sequence"],
        "heading_bbox": list(unmatched[0]["bbox"]),
        "heading_evidence_ids": list(unmatched[0]["evidence_ids"]),
        "heading_source_logical_page": int(
            previous_evidence.get("page") or getattr(previous_page, "page_number", 0) or 0
        ),
        "binding": "terminal_prior_page_agreement_anchor_and_leading_exact_table",
    }


def _exact_agreement_continuation_schema(
    table: Any,
    *,
    expected_labels: tuple[str, ...],
) -> dict[str, Any] | None:
    """Prove an agreement card against a preceding exact semantic role map.

    A repeated header may contain one OCR-contaminated cell.  Continuation is
    still source-provable when every physical header/value cell is exact, all
    registered labels remain in their preceding-page columns, and at most one
    header label is unknown.  No business value participates in the role map.
    """

    exact = _exact_mixed_page_table_schema(table)
    if (
        exact is not None
        and exact.get("template_id") == "credit_agreement"
        and tuple(exact.get("header_labels") or ()) == expected_labels
    ):
        return exact

    table_id = str(getattr(table, "table_id", "") or "")
    table_bbox = _bbox(table)
    rows = _raw_rows(table)
    width = len(expected_labels)
    if not table_id or table_bbox is None or width < 2 or len(rows) < 2:
        return None
    candidates: list[dict[str, Any]] = []
    expected_set = set(expected_labels)
    for row_index, row in enumerate(rows[:-1]):
        observed = tuple(_compact(value) for value in row[:width])
        if len(row) != width or any(not value for value in observed):
            continue
        exact_positions = sum(
            value == expected_labels[column]
            for column, value in enumerate(observed)
        )
        misplaced_registered = any(
            value in expected_set and value != expected_labels[column]
            for column, value in enumerate(observed)
        )
        if misplaced_registered or exact_positions < width - 1:
            continue
        values = rows[row_index + 1]
        if len(values) != width or any(not _compact(value) for value in values):
            continue
        header_owners = [
            _exact_table_cell_owner(table, row=row_index, column=column)
            for column in range(width)
        ]
        value_owners = [
            _exact_table_cell_owner(table, row=row_index + 1, column=column)
            for column in range(width)
        ]
        if any(owner is None for owner in (*header_owners, *value_owners)):
            continue
        exact_owners = [
            owner
            for owner in (*header_owners, *value_owners)
            if owner is not None
        ]
        evidence_ids = [
            evidence_id
            for _box, owners in exact_owners
            for evidence_id in owners
        ]
        boxes = [box for box, _owners in exact_owners]
        if (
            len(evidence_ids) != len(set(evidence_ids))
            or not _boxes_have_disjoint_interiors(boxes)
            or any(
                box[0] < table_bbox[0] - 1e-6
                or box[1] < table_bbox[1] - 1e-6
                or box[2] > table_bbox[2] + 1e-6
                or box[3] > table_bbox[3] + 1e-6
                for box in boxes
            )
        ):
            continue
        candidates.append(
            {
                "template_id": "credit_agreement",
                "table_id": table_id,
                "header_row": row_index,
                "population_witness_row": row_index + 1,
                "header_labels": list(expected_labels),
                "observed_header_labels": list(observed),
                "evidence_ids": sorted(evidence_ids),
                "table_bbox": list(table_bbox),
                "header_binding": "preceding_exact_agreement_role_map",
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def _exact_card_anchors(
    evidence: Mapping[str, Any],
    *,
    before_top: float = math.inf,
) -> tuple[dict[str, Any], ...]:
    anchors: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for line in evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        text = (
            _compact(line.get("text") or line.get("content") or "")
            if isinstance(line, Mapping)
            else ""
        )
        match = CREDIT_AGREEMENT_CARD_HEADING_RE.fullmatch(text)
        if bbox is None or bbox[1] >= before_top or match is None:
            continue
        owners = tuple(
            str(value)
            for value in line.get("evidence_ids") or ()
            if str(value or "")
        )
        if (
            not owners
            or len(owners) != len(set(owners))
            or consumed.intersection(owners)
        ):
            return ()
        consumed.update(owners)
        anchors.append(
            {
                "sequence": int(match.group("sequence")),
                "bbox": bbox,
                "evidence_ids": owners,
            }
        )
    anchors.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    sequences = [int(item["sequence"]) for item in anchors]
    if (
        not anchors
        or len(sequences) != len(set(sequences))
        or sequences != list(range(sequences[0], sequences[0] + len(sequences)))
    ):
        return ()
    return tuple(anchors)


def _card_table_pairs(
    anchors: tuple[dict[str, Any], ...],
    tables: tuple[tuple[Any, dict[str, Any], tuple[float, float, float, float]], ...],
    *,
    boundary_top: float,
) -> tuple[tuple[dict[str, Any], Any, dict[str, Any]], ...]:
    pairs: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    consumed_tables: set[str] = set()
    for index, anchor in enumerate(anchors):
        next_top = (
            anchors[index + 1]["bbox"][1]
            if index + 1 < len(anchors)
            else boundary_top
        )
        matches = [
            (table, schema)
            for table, schema, table_box in tables
            if _heading_attaches_to_table(anchor["bbox"], table_box)
            and table_box[3] <= next_top
        ]
        if len(matches) != 1:
            return ()
        table, schema = matches[0]
        table_id = str(getattr(table, "table_id", "") or "")
        if not table_id or table_id in consumed_tables:
            return ()
        consumed_tables.add(table_id)
        pairs.append((anchor, table, schema))
    if len(consumed_tables) != len(tables):
        return ()
    return tuple(pairs)


def _same_page_agreement_continuation_owners(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    previous_registration: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Own repeated agreement cards on the next printed page.

    The proof is independent of page numbers and fixture cardinalities: an
    exact registered agreement section, consecutive printed pages, dense card
    ordinals, a preceding exact semantic header map, and one-to-one local
    anchor/table geometry are all mandatory.
    """

    if str(previous_registration.get("template_id") or "") != "credit_agreement":
        return {}
    previous_printed = _printed_identity(previous_evidence)
    current_printed = _printed_identity(current_evidence)
    if (
        previous_printed is None
        or current_printed is None
        or previous_printed[1] != current_printed[1]
        or current_printed[0] != previous_printed[0] + 1
    ):
        return {}
    section_owners = [
        tuple(
            str(value)
            for value in line.get("evidence_ids") or ()
            if str(value or "")
        )
        for line in previous_evidence.get("lines") or ()
        if _exact_evidence_line_bbox(line) is not None
        and canonical_registered_section_heading(
            _compact(line.get("text") or line.get("content") or "")
        )
        == "授信协议信息"
    ]
    if (
        len(section_owners) != 1
        or not section_owners[0]
        or len(section_owners[0]) != len(set(section_owners[0]))
    ):
        return {}

    previous_schemas = [
        schema
        for table in getattr(previous_page, "tables", None) or ()
        if (schema := _exact_mixed_page_table_schema(table)) is not None
        and schema.get("template_id") == "credit_agreement"
    ]
    header_orders = {
        tuple(schema.get("header_labels") or ()) for schema in previous_schemas
    }
    if not previous_schemas or len(header_orders) != 1:
        return {}
    expected_labels = next(iter(header_orders))
    if not expected_labels:
        return {}
    previous_anchors = _exact_card_anchors(previous_evidence)
    if not previous_anchors:
        return {}

    boundary_tops: list[float] = []
    for line in current_evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None:
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        title = canonical_registered_section_heading(text)
        subsection = canonical_registered_subsection_heading(text)
        role = (
            REGISTERED_SECTION_TEMPLATE_BY_TITLE[title]
            if title is not None
            else subsection[0]
            if subsection is not None
            else None
        )
        if role is not None and role != "credit_agreement":
            boundary_tops.append(bbox[1])
    boundary_top = min(boundary_tops, default=math.inf)
    current_anchors = _exact_card_anchors(
        current_evidence,
        before_top=boundary_top,
    )
    if (
        not current_anchors
        or current_anchors[0]["sequence"]
        != previous_anchors[-1]["sequence"] + 1
    ):
        return {}

    current_tables: list[
        tuple[Any, dict[str, Any], tuple[float, float, float, float]]
    ] = []
    for table in getattr(current_page, "tables", None) or ():
        table_box = _bbox(table)
        if table_box is None or table_box[3] > boundary_top:
            continue
        schema = _exact_agreement_continuation_schema(
            table,
            expected_labels=expected_labels,
        )
        if schema is not None:
            current_tables.append((table, schema, table_box))
    pairs = _card_table_pairs(
        current_anchors,
        tuple(current_tables),
        boundary_top=boundary_top,
    )
    if len(pairs) != len(current_anchors):
        return {}

    current_ids = [
        evidence_id
        for anchor, _table, schema in pairs
        for evidence_id in (*anchor["evidence_ids"], *schema["evidence_ids"])
    ]
    if (
        len(current_ids) != len(set(current_ids))
        or set(current_ids).intersection(section_owners[0])
    ):
        return {}
    return {
        str(getattr(table, "table_id", "") or ""): {
            **schema,
            "printed_sequence": anchor["sequence"],
            "heading_bbox": list(anchor["bbox"]),
            "heading_evidence_ids": list(anchor["evidence_ids"]),
            "heading_source_logical_page": int(
                current_evidence.get("page")
                or getattr(current_page, "page_number", 0)
                or 0
            ),
            "binding": "consecutive_printed_page_agreement_card_and_exact_table",
        }
        for anchor, table, schema in pairs
    }


def _table_geometry_lattice(table: Any) -> Mapping[str, Any] | None:
    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return None
    nested = metadata.get("geometry")
    candidates = [
        owner
        for owner in (nested, metadata)
        if isinstance(owner, Mapping)
        and isinstance(owner.get("cell_geometry_status"), list)
        and isinstance(owner.get("cell_bboxes"), list)
        and isinstance(owner.get("cell_evidence_ids"), list)
    ]
    if not candidates:
        return None
    if len(candidates) == 2 and any(
        candidates[0].get(key) != candidates[1].get(key)
        for key in (
            "cell_geometry_status",
            "cell_bboxes",
            "cell_evidence_ids",
            "cell_spans",
        )
        if key in candidates[0] and key in candidates[1]
    ):
        return None
    return candidates[0]


def _exact_empty_role_slot(
    geometry: Mapping[str, Any],
    *,
    row: int,
    column: int,
) -> tuple[float, float, float, float] | None:
    statuses = geometry.get("cell_geometry_status")
    bboxes = geometry.get("cell_bboxes")
    evidence_ids = geometry.get("cell_evidence_ids")
    token_ids = geometry.get("cell_token_ids")
    if not all(isinstance(grid, list) for grid in (statuses, bboxes, evidence_ids)):
        return None
    if any(
        row >= len(grid)
        or not isinstance(grid[row], list)
        or column >= len(grid[row])
        for grid in (statuses, bboxes, evidence_ids)
    ):
        return None
    bbox = _bbox({"bbox": bboxes[row][column]})
    raw_ids = evidence_ids[row][column]
    if (
        str(statuses[row][column] or "") != "exact"
        or bbox is None
        or not all(math.isfinite(value) for value in bbox)
        or not isinstance(raw_ids, list)
        or raw_ids
    ):
        return None
    if isinstance(token_ids, list) and (
        row >= len(token_ids)
        or not isinstance(token_ids[row], list)
        or column >= len(token_ids[row])
        or not isinstance(token_ids[row][column], list)
        or token_ids[row][column]
    ):
        return None
    return bbox


def _headerless_foreign_role_label(value: Any) -> bool:
    label = _compact(value)
    registered = {
        "编号",
        "序号",
        "查询日期",
        "查询机构",
        "查询原因",
        *_ACCOUNT_TABLE_LABELS,
        *(
            schema_label
            for schemas in _MIXED_PAGE_TABLE_SCHEMAS.values()
            for schema in schemas
            for schema_label in schema
        ),
    }
    return label in registered


def _exact_headerless_row_spans(
    table: Any,
    geometry: Mapping[str, Any],
    rows: Sequence[Sequence[Any]],
    *,
    row: int,
    width: int,
) -> tuple[
    dict[int, tuple[int, tuple[float, float, float, float], tuple[str, ...]]],
    set[int],
] | None:
    raw_spans = geometry.get("cell_spans")
    if raw_spans is None:
        raw_spans = []
    if not isinstance(raw_spans, list):
        return None
    spans = [
        span
        for span in raw_spans
        if isinstance(span, Mapping) and span.get("row") == row
    ]
    if any(
        not isinstance(span.get("col"), int)
        or isinstance(span.get("col"), bool)
        or not isinstance(span.get("col_span"), int)
        or isinstance(span.get("col_span"), bool)
        or span.get("row_span") != 1
        or int(span["col_span"]) < 2
        or int(span["col"]) < 0
        or int(span["col"]) + int(span["col_span"]) > width
        for span in spans
    ):
        return None
    owners: dict[
        int,
        tuple[int, tuple[float, float, float, float], tuple[str, ...]],
    ] = {}
    covered: set[int] = set()
    for span in spans:
        owner_column = int(span["col"])
        end_column = owner_column + int(span["col_span"])
        if any(column in covered for column in range(owner_column, end_column)):
            return None
        owner = _exact_table_cell_owner(table, row=row, column=owner_column)
        span_bbox = _bbox({"bbox": span.get("bbox")})
        raw_span_ids = span.get("evidence_ids")
        span_ids = (
            tuple(str(value) for value in raw_span_ids if str(value or ""))
            if isinstance(raw_span_ids, list)
            else ()
        )
        if (
            owner is None
            or not _compact(rows[row][owner_column])
            or span_bbox is None
            or not all(math.isfinite(value) for value in (*owner[0], *span_bbox))
            or any(
                not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-6)
                for left, right in zip(owner[0], span_bbox, strict=True)
            )
            or owner[1] != span_ids
        ):
            return None
        for column in range(owner_column + 1, end_column):
            statuses = geometry.get("cell_geometry_status")
            bboxes = geometry.get("cell_bboxes")
            evidence_ids = geometry.get("cell_evidence_ids")
            if (
                _compact(rows[row][column])
                or not all(isinstance(grid, list) for grid in (statuses, bboxes, evidence_ids))
                or row >= len(statuses)
                or row >= len(bboxes)
                or row >= len(evidence_ids)
                or column >= len(statuses[row])
                or column >= len(bboxes[row])
                or column >= len(evidence_ids[row])
                or str(statuses[row][column] or "") != "derived"
                or bboxes[row][column] is not None
                or evidence_ids[row][column] not in ([], None)
            ):
                return None
        owners[owner_column] = (end_column, owner[0], owner[1])
        covered.update(range(owner_column, end_column))
    return owners, covered


def _exact_headerless_four_role_lattice(
    table: Any,
    rows: Sequence[Sequence[Any]],
    *,
    role_columns: Mapping[str, int],
    table_bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    geometry = _table_geometry_lattice(table)
    if geometry is None or set(role_columns.values()) != set(range(4)):
        return None
    sequence_column = int(role_columns["sequence"])
    date_column = int(role_columns["inquiry_date"])
    evidence_ids: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    omission_rows: list[int] = []
    witness_rows: list[int] = []
    for row_index, row in enumerate(rows):
        if len(row) != 4 or not any(_compact(value) for value in row):
            return None
        if any(_headerless_foreign_role_label(value) for value in row):
            return None
        resolved_spans = _exact_headerless_row_spans(
            table,
            geometry,
            rows,
            row=row_index,
            width=4,
        )
        if resolved_spans is None:
            return None
        span_owners, covered = resolved_spans
        row_has_omission = bool(span_owners)
        for column in range(4):
            if column in covered and column not in span_owners:
                continue
            if column in span_owners:
                _end_column, bbox, ids = span_owners[column]
                boxes.append(bbox)
                evidence_ids.extend(ids)
                continue
            if _compact(row[column]):
                owner = _exact_table_cell_owner(
                    table,
                    row=row_index,
                    column=column,
                )
                if owner is None:
                    return None
                boxes.append(owner[0])
                evidence_ids.extend(owner[1])
                continue
            empty_bbox = _exact_empty_role_slot(
                geometry,
                row=row_index,
                column=column,
            )
            if empty_bbox is None:
                return None
            boxes.append(empty_bbox)
            row_has_omission = True
        sequence = _compact(row[sequence_column])
        inquiry_date = _compact(row[date_column])
        if (
            not row_has_omission
            and re.fullmatch(r"[1-9]\d{0,3}", sequence) is not None
            and re.fullmatch(
                r"(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}",
                inquiry_date,
            )
            is not None
        ):
            witness_rows.append(row_index)
        else:
            row_text = " ".join(_compact(value) for value in row if _compact(value))
            if (
                re.search(r"(?<!\d)[1-9]\d{0,3}(?!\d)", row_text) is None
                and re.search(
                    r"(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}",
                    row_text,
                )
                is None
            ):
                return None
            omission_rows.append(row_index)
    if (
        not witness_rows
        or len(evidence_ids) != len(set(evidence_ids))
        or any(not all(math.isfinite(value) for value in box) for box in boxes)
        or not _boxes_have_disjoint_interiors(boxes)
        or any(
            box[0] < table_bbox[0] - 1e-6
            or box[1] < table_bbox[1] - 1e-6
            or box[2] > table_bbox[2] + 1e-6
            or box[3] > table_bbox[3] + 1e-6
            for box in boxes
        )
    ):
        return None
    return {
        "evidence_ids": evidence_ids,
        "population_witness_row": witness_rows[0],
        "physical_field_omission_rows": omission_rows,
        "header_binding": "inherited_exact_four_role_lattice",
    }


def _unique_sequence_date_cell(
    value: Any,
    *,
    sequence_precedes_date: bool,
) -> tuple[str, str] | None:
    compact = " ".join(str(value or "").split())
    sequence = r"(?P<sequence>[1-9]\d{0,3})"
    inquiry_date = r"(?P<date>(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})"
    pattern = (
        rf"^{sequence}\s+{inquiry_date}$"
        if sequence_precedes_date
        else rf"^{inquiry_date}\s+{sequence}$"
    )
    match = re.fullmatch(pattern, compact)
    return (
        (match.group("sequence"), match.group("date"))
        if match is not None
        else None
    )


def _exact_headerless_collapsed_sequence_date_lattice(
    table: Any,
    rows: Sequence[Sequence[Any]],
    *,
    role_columns: Mapping[str, int],
    table_bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    geometry = _table_geometry_lattice(table)
    sequence_column = role_columns.get("sequence")
    date_column = role_columns.get("inquiry_date")
    if (
        geometry is None
        or not isinstance(sequence_column, int)
        or not isinstance(date_column, int)
        or abs(sequence_column - date_column) != 1
        or set(role_columns.values()) != set(range(4))
    ):
        return None
    merged_column = min(sequence_column, date_column)
    physical_roles: list[tuple[str, ...]] = []
    for column in range(4):
        roles = tuple(
            role for role, role_column in role_columns.items() if role_column == column
        )
        if column == merged_column:
            physical_roles.append(
                tuple(
                    role
                    for role, role_column in sorted(
                        role_columns.items(), key=lambda item: item[1]
                    )
                    if role_column in {sequence_column, date_column}
                )
            )
        elif column == merged_column + 1:
            continue
        else:
            physical_roles.append(roles)
    if len(physical_roles) != 3 or len(physical_roles[merged_column]) != 2:
        return None

    evidence_ids: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    omission_rows: list[int] = []
    witness_rows: list[int] = []
    for row_index, row in enumerate(rows):
        if len(row) != 3 or not any(_compact(value) for value in row):
            return None
        if any(_headerless_foreign_role_label(value) for value in row):
            return None
        resolved_spans = _exact_headerless_row_spans(
            table,
            geometry,
            rows,
            row=row_index,
            width=3,
        )
        if resolved_spans is None:
            return None
        span_owners, covered = resolved_spans
        if span_owners:
            if set(span_owners) != {0} or span_owners[0][0] != 3:
                return None
            _end_column, bbox, ids = span_owners[0]
            boxes.append(bbox)
            evidence_ids.extend(ids)
            omission_rows.append(row_index)
            continue
        if covered or any(not _compact(value) for value in row):
            return None
        row_owners = [
            _exact_table_cell_owner(table, row=row_index, column=column)
            for column in range(3)
        ]
        if any(owner is None for owner in row_owners):
            return None
        exact_owners = [owner for owner in row_owners if owner is not None]
        boxes.extend(owner[0] for owner in exact_owners)
        evidence_ids.extend(
            evidence_id
            for _bbox_value, ids in exact_owners
            for evidence_id in ids
        )
        merged_value = row[merged_column]
        parsed = _unique_sequence_date_cell(
            merged_value,
            sequence_precedes_date=sequence_column < date_column,
        )
        merged_ids = exact_owners[merged_column][1]
        if parsed is not None and len(merged_ids) >= 2:
            witness_rows.append(row_index)
        else:
            omission_rows.append(row_index)
    if (
        not witness_rows
        or len(evidence_ids) != len(set(evidence_ids))
        or any(not all(math.isfinite(value) for value in box) for box in boxes)
        or not _boxes_have_disjoint_interiors(boxes)
        or any(
            box[0] < table_bbox[0] - 1e-6
            or box[1] < table_bbox[1] - 1e-6
            or box[2] > table_bbox[2] + 1e-6
            or box[3] > table_bbox[3] + 1e-6
            for box in boxes
        )
    ):
        return None
    return {
        "evidence_ids": evidence_ids,
        "population_witness_row": witness_rows[0],
        "physical_field_omission_rows": omission_rows,
        "physical_role_columns": [list(roles) for roles in physical_roles],
        "header_binding": "inherited_exact_sequence_date_collapsed_lattice",
    }


def _strict_inquiry_sequence_from_row(
    row: Sequence[Any],
    *,
    role_columns: Mapping[str, int],
    physical_role_columns: Any = None,
) -> int | None:
    if len(row) == 4:
        sequence_column = role_columns.get("sequence")
        if not isinstance(sequence_column, int) or not 0 <= sequence_column < 4:
            return None
        raw_sequence = _compact(row[sequence_column])
        return int(raw_sequence) if re.fullmatch(r"[1-9]\d{0,3}", raw_sequence) else None
    if len(row) != 3 or not isinstance(physical_role_columns, list) or len(physical_role_columns) != 3:
        return None
    for column, raw_roles in enumerate(physical_role_columns):
        roles = tuple(str(role) for role in raw_roles or ()) if isinstance(raw_roles, list) else ()
        if "sequence" not in roles:
            continue
        if roles == ("sequence",):
            raw_sequence = _compact(row[column])
            return int(raw_sequence) if re.fullmatch(r"[1-9]\d{0,3}", raw_sequence) else None
        if set(roles) == {"sequence", "inquiry_date"}:
            parsed = _unique_sequence_date_cell(
                row[column],
                sequence_precedes_date=roles.index("sequence") < roles.index("inquiry_date"),
            )
            return int(parsed[0]) if parsed is not None else None
    return None


def _raw_inquiry_sequence_from_row(
    row: Sequence[Any],
    *,
    role_columns: Mapping[str, int],
    physical_role_columns: Any = None,
) -> str | None:
    if len(row) == 4:
        sequence_column = role_columns.get("sequence")
        return (
            _compact(row[sequence_column])
            if isinstance(sequence_column, int) and 0 <= sequence_column < 4
            else None
        )
    if len(row) != 3 or not isinstance(physical_role_columns, list) or len(physical_role_columns) != 3:
        return None
    candidates = [
        _compact(row[column])
        for column, raw_roles in enumerate(physical_role_columns)
        if isinstance(raw_roles, list)
        and "sequence" in {str(role) for role in raw_roles}
    ]
    return candidates[0] if len(candidates) == 1 else None


def _owned_inquiry_population_endpoint(
    table: Any,
    owner: Mapping[str, Any],
) -> int | None:
    declared = _positive_int(owner.get("population_endpoint"))
    if declared is not None:
        return declared
    role_columns = owner.get("inquiry_role_columns")
    if not isinstance(role_columns, Mapping):
        return None
    rows = _raw_rows(table)
    header_rows = owner.get("header_rows") or ()
    if not isinstance(header_rows, (list, tuple)) or any(
        not isinstance(row, int) or isinstance(row, bool) or row < 0 for row in header_rows
    ):
        return None
    body_start = max(header_rows, default=-1) + 1
    body_rows = [row for row in rows[body_start:] if any(_compact(value) for value in row)]
    if not body_rows:
        return None
    return _strict_inquiry_sequence_from_row(
        body_rows[-1],
        role_columns=role_columns,
        physical_role_columns=owner.get("physical_role_columns"),
    )


def _sealed_inquiry_population_bounds(
    rows: Sequence[Sequence[Any]],
    *,
    role_columns: Mapping[str, int],
    physical_role_columns: Any = None,
) -> tuple[int, int] | None:
    """Close physical endpoints without deciding damaged interior fields."""

    if not rows:
        return None
    start = _strict_inquiry_sequence_from_row(
        rows[0],
        role_columns=role_columns,
        physical_role_columns=physical_role_columns,
    )
    endpoint = _strict_inquiry_sequence_from_row(
        rows[-1],
        role_columns=role_columns,
        physical_role_columns=physical_role_columns,
    )
    if start is None and endpoint is None:
        return None
    if start is None:
        start = endpoint - len(rows) + 1 if endpoint is not None else None
    if endpoint is None:
        endpoint = start + len(rows) - 1 if start is not None else None
    if (
        start is None
        or endpoint is None
        or start <= 0
        or endpoint != start + len(rows) - 1
    ):
        return None
    return start, endpoint


def _reading_order_seals_exact_printed_pair(
    reading_order_resolution: Any,
    previous_page: Any,
    current_page: Any,
    *,
    previous_printed: tuple[int, int],
    current_printed: tuple[int, int],
) -> bool:
    if not isinstance(reading_order_resolution, Mapping):
        return True
    previous_logical = _positive_int(getattr(previous_page, "page_number", None))
    current_logical = _positive_int(getattr(current_page, "page_number", None))
    printed_total = _positive_int(reading_order_resolution.get("printed_total"))
    printed_by_logical = _logical_int_mapping(
        reading_order_resolution.get("printed_page_by_logical")
    )
    return bool(
        reading_order_resolution.get("resolved") is True
        and reading_order_resolution.get("authoritative") is True
        and reading_order_resolution.get("identity_fallback") is not True
        and previous_logical is not None
        and current_logical is not None
        and printed_total == previous_printed[1] == current_printed[1]
        and printed_by_logical is not None
        and printed_by_logical.get(previous_logical) == previous_printed[0]
        and printed_by_logical.get(current_logical) == current_printed[0]
    )


def _inquiry_continuation_adjacency_proof(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
    *,
    previous_printed: tuple[int, int],
    current_printed: tuple[int, int],
    kind: str,
    identity_kind: str,
    topology: Any = None,
) -> dict[str, Any] | None:
    """Seal the exact page identities used by one continuation edge."""

    if kind not in {
        "exact_printed_footer_table_edge",
        _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF,
        "local_paired_topology_entity_edge",
    } or identity_kind not in {
        "exact_footer_pair",
        "paired_inferred_current_footer",
    }:
        return None

    def page_identity(
        page: Any,
        evidence: Mapping[str, Any],
        printed: tuple[int, int],
    ) -> dict[str, Any] | None:
        logical = _positive_int(
            getattr(page, "page_number", None) or evidence.get("page")
        )
        source = _positive_int(
            getattr(page, "source_page_number", None)
            or evidence.get("source_page")
        )
        evidence_source = _positive_int(evidence.get("source_page"))
        width = _finite(getattr(page, "width", 0))
        height = _finite(getattr(page, "height", 0))
        evidence_width = _finite(evidence.get("page_width"))
        evidence_height = _finite(evidence.get("page_height"))
        topology_geometry = None
        if topology is not None and logical is not None:
            try:
                topology_geometry = topology.geometry(logical)
            except (AttributeError, TypeError, ValueError):
                return None
        if (
            logical is None
            or source is None
            or evidence_source != source
            or printed[0] <= 0
            or printed[1] <= 0
            or printed[0] > printed[1]
            or width <= 0.0
            or height <= 0.0
            or evidence_width <= 0.0
            or evidence_height <= 0.0
            or not math.isclose(width, evidence_width, rel_tol=1e-7, abs_tol=1e-6)
            or not math.isclose(height, evidence_height, rel_tol=1e-7, abs_tol=1e-6)
        ):
            return None
        geometry = {
            "kind": "page",
            "width": width,
            "height": height,
        }
        if topology_geometry is not None:
            topology_source = _positive_int(
                getattr(topology_geometry, "source_page", None)
            )
            topology_width = _finite(getattr(topology_geometry, "width", 0))
            topology_height = _finite(getattr(topology_geometry, "height", 0))
            crop = getattr(topology_geometry, "source_crop_bbox", None)
            if (
                topology_source != source
                or topology_width <= 0.0
                or topology_height <= 0.0
                or (
                    crop is not None
                    and (
                        not isinstance(crop, tuple)
                        or len(crop) != 4
                        or any(not math.isfinite(float(value)) for value in crop)
                    )
                )
            ):
                return None
            geometry = {
                "kind": "topology",
                "width": topology_width,
                "height": topology_height,
                "split_kind": str(
                    getattr(topology_geometry, "split_kind", "") or ""
                ),
                "segment_index": getattr(topology_geometry, "segment_index", None),
                "selected_rotation": int(
                    getattr(topology_geometry, "selected_rotation", 0) or 0
                ),
                "transform_usable": getattr(
                    topology_geometry,
                    "transform_usable",
                    None,
                )
                is True,
                "source_crop_bbox": list(crop) if crop is not None else None,
            }
        return {
            "logical_page": logical,
            "source_page": source,
            "printed_page": printed[0],
            "printed_total": printed[1],
            "geometry": geometry,
        }

    previous = page_identity(previous_page, previous_evidence, previous_printed)
    current = page_identity(current_page, current_evidence, current_printed)
    if previous is None or current is None:
        return None
    return {
        "kind": kind,
        "identity_kind": identity_kind,
        "previous": previous,
        "current": current,
    }


def _headerless_inquiry_continuation_owner(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    previous_registration: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
    local_owners: Mapping[str, Mapping[str, Any]],
    *,
    tables_continue: Any,
    reading_order_resolution: Any = None,
    topology: Any = None,
    frozen_topology_audit_loader: Any = None,
    entity_context: Any = None,
) -> tuple[str, dict[str, Any]] | None:
    """Bind one exact headerless inquiry table to its preceding owner.

    This is a table-local continuation proof, not page adjacency inference.
    It requires consecutive authoritative printed identities, an explicit
    entity/table continuation edge, a previously owned exact inquiry schema,
    and an exact current lattice in the inherited semantic column order.  A
    missing current footer may be replaced only by the context's sealed
    paired-spread inference after that inference is independently replayed.
    For an ordinary exact inquiry seed, a false entity-table edge still permits
    schema-only carry across distinct physical tables when authoritative exact
    consecutive footers and consecutive population endpoints agree.  This
    proof never grants ordinal closure.  A closed-ordinal seed with a false
    document-wide edge still requires the same frozen repeated-spread topology
    and an independently sealed local entity transition.
    Exact empty slots and sealed colspans preserve physical omissions without
    manufacturing values.  A three-column continuation is accepted only when
    the inherited sequence/date roles are adjacent and at least one merged
    cell has a unique, mutually typed sequence/date decomposition.
    """

    if not callable(tables_continue):
        return None
    previous_printed = _printed_identity(previous_evidence)
    current_printed = _printed_identity(current_evidence)
    printed_adjacency = bool(
        previous_printed is not None
        and current_printed is not None
        and previous_printed[1] == current_printed[1]
        and current_printed[0] == previous_printed[0] + 1
    )
    if printed_adjacency and not _reading_order_seals_exact_printed_pair(
        reading_order_resolution,
        previous_page,
        current_page,
        previous_printed=previous_printed,
        current_printed=current_printed,
    ):
        return None
    local_topology_adjacency = False
    if printed_adjacency:
        local_topology_adjacency = _local_exact_spread_printed_adjacency(
            previous_page,
            previous_evidence,
            current_page,
            current_evidence,
            previous_printed=previous_printed,
            current_printed=current_printed,
            reading_order_resolution=reading_order_resolution,
            topology=topology,
            frozen_topology_audit_loader=frozen_topology_audit_loader,
        )
    else:
        local_paired_adjacency = _local_paired_printed_adjacency(
            previous_page,
            previous_evidence,
            current_page,
            current_evidence,
            previous_printed=previous_printed,
            current_printed=current_printed,
            reading_order_resolution=reading_order_resolution,
            topology=topology,
            frozen_topology_audit_loader=frozen_topology_audit_loader,
        )
        local_topology_adjacency = local_paired_adjacency
        printed_adjacency = local_topology_adjacency
    if (
        not printed_adjacency
        or _sealed_registered_heading_roles(current_page, current_evidence).difference(
            {"annotations_and_inquiries"}
        )
    ):
        return None

    previous_owned = previous_registration.get("section_table_owners") or {}
    previous_candidates = [
        (table, owner, _bbox(table))
        for table in getattr(previous_page, "tables", None) or ()
        for owner in (
            previous_owned.get(str(getattr(table, "table_id", "") or "")),
        )
        if isinstance(owner, Mapping)
        and owner.get("template_id") == "annotations_and_inquiries"
        and _bbox(table) is not None
    ]
    if not previous_candidates:
        return None
    max_bottom = max(box[3] for _table, _owner, box in previous_candidates)
    terminal = [
        (table, owner)
        for table, owner, box in previous_candidates
        if math.isclose(box[3], max_bottom, rel_tol=1e-7, abs_tol=1e-6)
    ]
    if len(terminal) != 1:
        return None
    previous_table, previous_owner = terminal[0]
    header_labels = tuple(previous_owner.get("header_labels") or ())
    prior_role_columns = previous_owner.get("inquiry_role_columns") or {}
    if frozenset(header_labels) != frozenset(
        {"编号", "查询日期", "查询机构", "查询原因"}
    ) or (
        not isinstance(prior_role_columns, Mapping)
        or set(prior_role_columns)
        != {"sequence", "inquiry_date", "institution", "reason"}
        or any(
            not isinstance(column, int) or isinstance(column, bool)
            for column in prior_role_columns.values()
        )
        or set(prior_role_columns.values()) != set(range(4))
    ):
        return None

    tables = tuple(getattr(current_page, "tables", None) or ())
    unowned = [
        table
        for table in tables
        if str(getattr(table, "table_id", "") or "") not in local_owners
        and _bbox(table) is not None
    ]
    local_inquiry_tops = [
        box[1]
        for table in tables
        if isinstance(
            local_owners.get(str(getattr(table, "table_id", "") or "")),
            Mapping,
        )
        and local_owners[str(getattr(table, "table_id", "") or "")].get(
            "template_id"
        )
        == "annotations_and_inquiries"
        and (box := _bbox(table)) is not None
    ]
    if local_inquiry_tops:
        first_local_top = min(local_inquiry_tops)
        unowned = [table for table in unowned if (_bbox(table) or (0, 0, 0, math.inf))[3] <= first_local_top]
    if len(unowned) != 1:
        return None
    current_table = unowned[0]
    previous_table_id = str(getattr(previous_table, "table_id", "") or "")
    current_table_id = str(getattr(current_table, "table_id", "") or "")
    if (
        not previous_table_id
        or not current_table_id
    ):
        return None
    continuation_decision = tables_continue(previous_table_id, current_table_id)
    if continuation_decision is not True and continuation_decision is not False:
        return None

    rows = _raw_rows(current_table)
    table_bbox = _bbox(current_table)
    physical_widths = {len(row) for row in rows}
    if (
        table_bbox is None
        or not all(math.isfinite(value) for value in table_bbox)
        or not rows
        or len(header_labels) != 4
        or len(physical_widths) != 1
    ):
        return None
    physical_width = next(iter(physical_widths))
    if physical_width not in {3, 4}:
        return None
    populated_indexes = [
        row_index
        for row_index, row in enumerate(rows)
        if any(_compact(value) for value in row)
    ]
    if not populated_indexes:
        return None
    last_populated = max(populated_indexes)
    if any(
        not _exact_empty_table_border_row(
            current_table,
            row=row_index,
            width=physical_width,
        )
        for row_index in range(last_populated + 1, len(rows))
    ):
        return None
    rows = rows[: last_populated + 1]
    lattice = (
        _exact_headerless_four_role_lattice(
            current_table,
            rows,
            role_columns=prior_role_columns,
            table_bbox=table_bbox,
        )
        if physical_width == 4
        else _exact_headerless_collapsed_sequence_date_lattice(
            current_table,
            rows,
            role_columns=prior_role_columns,
            table_bbox=table_bbox,
        )
    )
    if lattice is None:
        return None
    previous_endpoint = _owned_inquiry_population_endpoint(
        previous_table,
        previous_owner,
    )
    current_bounds = _sealed_inquiry_population_bounds(
        rows,
        role_columns=prior_role_columns,
        physical_role_columns=lattice.get("physical_role_columns"),
    )
    if (
        previous_endpoint is None
        or current_bounds is None
        or current_bounds[0] != previous_endpoint + 1
    ):
        return None
    current_start, current_endpoint = current_bounds
    sequence_anomalies = [
        {
            "row": row_index,
            "expected_sequence": current_start + row_index,
            "raw_sequence": raw_sequence,
            "status": (
                "physical_field_omission"
                if not raw_sequence
                else "unparsed_raw_sequence"
            ),
        }
        for row_index, row in enumerate(rows)
        for raw_sequence in (
            _raw_inquiry_sequence_from_row(
                row,
                role_columns=prior_role_columns,
                physical_role_columns=lattice.get("physical_role_columns"),
            ),
        )
        if raw_sequence != str(current_start + row_index)
    ]
    local_entity_adjacency = bool(
        local_topology_adjacency
        and _local_paired_inquiry_entity_continuation_proved(
            entity_context,
            previous_page,
            previous_table,
            previous_owner,
            current_page,
            current_table,
            lattice,
        )
    )
    previous_proof = previous_owner.get("adjacency_proof")
    ordinary_schema_carry_chain = bool(
        previous_owner.get("binding")
        == "exact_pboc_section_heading_and_table_schema"
        or (
            isinstance(previous_proof, Mapping)
            and previous_proof.get("kind")
            == _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF
        )
    )
    exact_footer_schema_carry_bridge = bool(
        continuation_decision is False
        and current_printed is not None
        and isinstance(reading_order_resolution, Mapping)
        and ordinary_schema_carry_chain
    )
    requires_local_paired_proof = bool(
        (
            continuation_decision is not True
            and not exact_footer_schema_carry_bridge
        )
        or current_printed is None
    )
    if requires_local_paired_proof and not local_entity_adjacency:
        return None
    sealed_current_printed = current_printed
    if sealed_current_printed is None:
        printed_total = (
            _positive_int(reading_order_resolution.get("printed_total"))
            if isinstance(reading_order_resolution, Mapping)
            else None
        )
        printed_by_logical = (
            _logical_int_mapping(
                reading_order_resolution.get("printed_page_by_logical")
            )
            if isinstance(reading_order_resolution, Mapping)
            else None
        )
        current_logical = _positive_int(getattr(current_page, "page_number", None))
        inferred_current = (
            printed_by_logical.get(current_logical)
            if printed_by_logical is not None and current_logical is not None
            else None
        )
        if printed_total is None or inferred_current is None:
            return None
        sealed_current_printed = (inferred_current, printed_total)
    if previous_printed is None:
        return None
    adjacency_proof = _inquiry_continuation_adjacency_proof(
        previous_page,
        previous_evidence,
        current_page,
        current_evidence,
        previous_printed=previous_printed,
        current_printed=sealed_current_printed,
        kind=(
            "local_paired_topology_entity_edge"
            if requires_local_paired_proof
            else (
                _INQUIRY_EXACT_FOOTER_SCHEMA_CARRY_PROOF
                if exact_footer_schema_carry_bridge
                else "exact_printed_footer_table_edge"
            )
        ),
        identity_kind=(
            "paired_inferred_current_footer"
            if current_printed is None
            else "exact_footer_pair"
        ),
        topology=topology,
    )
    if adjacency_proof is None:
        return None
    evidence_ids = tuple(lattice["evidence_ids"])
    prior_evidence_ids = {
        str(value)
        for value in previous_owner.get("evidence_ids") or ()
        if str(value or "")
    }
    if prior_evidence_ids.intersection(evidence_ids):
        return None
    return current_table_id, {
        "template_id": "annotations_and_inquiries",
        "table_id": current_table_id,
        "header_row": None,
        "population_witness_row": lattice["population_witness_row"],
        "population_endpoint_row": len(rows) - 1,
        "header_labels": list(header_labels),
        "inquiry_role_columns": dict(prior_role_columns),
        "evidence_ids": sorted(evidence_ids),
        "table_bbox": list(table_bbox),
        "prior_table_id": previous_table_id,
        "header_binding": lattice["header_binding"],
        "physical_field_omission_rows": list(
            lattice["physical_field_omission_rows"]
        ),
        "population_start": current_start,
        "population_endpoint": current_endpoint,
        "sequence_field_anomalies": sequence_anomalies,
        "adjacency_proof": adjacency_proof,
        **(
            {"physical_role_columns": lattice["physical_role_columns"]}
            if "physical_role_columns" in lattice
            else {}
        ),
        "binding": "authoritative_prior_inquiry_table_continuation",
    }


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _logical_int_mapping(value: Any) -> dict[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[int, int] = {}
    for raw_logical, raw_value in value.items():
        if isinstance(raw_logical, bool):
            return None
        try:
            logical = int(raw_logical)
        except (TypeError, ValueError):
            return None
        mapped = _positive_int(raw_value)
        if logical <= 0 or mapped is None or logical in result:
            return None
        result[logical] = mapped
    return result


def _positive_int_set(value: Any) -> set[int] | None:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    result: set[int] = set()
    for raw in value:
        item = _positive_int(raw)
        if item is None or item in result:
            return None
        result.add(item)
    return result


def _topology_sources(value: Any) -> dict[int, tuple[int, ...]] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[int, tuple[int, ...]] = {}
    seen_logicals: set[int] = set()
    for raw_source, raw_logicals in value.items():
        if isinstance(raw_source, bool):
            return None
        try:
            source = int(raw_source)
        except (TypeError, ValueError):
            return None
        logicals = _positive_int_set(raw_logicals)
        if source <= 0 or source in result or logicals is None or not logicals:
            return None
        if seen_logicals.intersection(logicals):
            return None
        seen_logicals.update(logicals)
        result[source] = tuple(sorted(logicals))
    return result


def _local_exact_spread_printed_adjacency(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
    *,
    previous_printed: tuple[int, int] | None,
    current_printed: tuple[int, int] | None,
    reading_order_resolution: Any,
    topology: Any,
    frozen_topology_audit_loader: Any,
) -> bool:
    """Prove one exact-footer edge inside or across a repeated two-up spread.

    With no document-wide resolution, exact local footers may prove the
    printed identities.  A present resolution is an authority boundary and
    must itself seal the same pair.  The frozen/runtime topology audits must
    also agree, and the edge must be either the validated pair on one source
    surface or the terminal-to-leading edge across two consecutive source
    surfaces with the same complete two-up geometry profile.
    """

    if (
        previous_printed is None
        or current_printed is None
        or previous_printed[1] != current_printed[1]
        or current_printed[0] != previous_printed[0] + 1
        or topology is None
        or not callable(frozen_topology_audit_loader)
    ):
        return False
    previous_logical = _positive_int(
        getattr(previous_page, "page_number", None) or previous_evidence.get("page")
    )
    current_logical = _positive_int(
        getattr(current_page, "page_number", None) or current_evidence.get("page")
    )
    if previous_logical is None or current_logical is None or previous_logical == current_logical:
        return False
    if isinstance(reading_order_resolution, Mapping):
        printed_total = _positive_int(reading_order_resolution.get("printed_total"))
        printed_by_logical = _logical_int_mapping(
            reading_order_resolution.get("printed_page_by_logical")
        )
        if (
            reading_order_resolution.get("resolved") is not True
            or reading_order_resolution.get("authoritative") is not True
            or reading_order_resolution.get("identity_fallback") is True
            or printed_total is None
            or printed_by_logical is None
            or printed_total != previous_printed[1]
            or printed_total != current_printed[1]
            or printed_by_logical.get(previous_logical) != previous_printed[0]
            or printed_by_logical.get(current_logical) != current_printed[0]
        ):
            return False
    try:
        frozen_audit = frozen_topology_audit_loader()
        runtime_audit = topology.audit()
        previous_geometry = topology.geometry(previous_logical)
        current_geometry = topology.geometry(current_logical)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        not isinstance(frozen_audit, Mapping)
        or not isinstance(runtime_audit, Mapping)
        or previous_geometry is None
        or current_geometry is None
    ):
        return False
    frozen_sources = _topology_sources(frozen_audit.get("logical_pages_by_source"))
    runtime_sources = _topology_sources(runtime_audit.get("logical_pages_by_source"))
    if (
        frozen_audit.get("topology_frozen_before_reocr") is not True
        or frozen_audit.get("valid") is not True
        or runtime_audit.get("valid") is not True
        or frozen_sources is None
        or runtime_sources is None
        or frozen_sources != runtime_sources
    ):
        return False
    previous_source = _positive_int(getattr(previous_geometry, "source_page", None))
    current_source = _positive_int(getattr(current_geometry, "source_page", None))
    if (
        previous_source is None
        or current_source is None
        or _positive_int(previous_evidence.get("source_page")) != previous_source
        or _positive_int(current_evidence.get("source_page")) != current_source
        or _positive_int(getattr(previous_page, "source_page_number", None)) != previous_source
        or _positive_int(getattr(current_page, "source_page_number", None)) != current_source
        or getattr(previous_geometry, "split_kind", None) != "two_page_spread"
        or getattr(current_geometry, "split_kind", None) != "two_page_spread"
        or getattr(previous_geometry, "transform_usable", None) is not True
        or getattr(current_geometry, "transform_usable", None) is not True
    ):
        return False
    if previous_source == current_source:
        return bool(
            getattr(previous_geometry, "segment_index", None) == 0
            and getattr(current_geometry, "segment_index", None) == 1
            and topology.ordered_pair((previous_logical, current_logical))
            == (previous_logical, current_logical)
        )
    if (
        current_source != previous_source + 1
        or getattr(previous_geometry, "segment_index", None) != 1
        or getattr(current_geometry, "segment_index", None) != 0
    ):
        return False

    try:
        previous_pair = tuple(topology.logicals_for_source(previous_source))
        current_pair = tuple(topology.logicals_for_source(current_source))
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        len(previous_pair) != 2
        or len(current_pair) != 2
        or previous_pair[-1] != previous_logical
        or current_pair[0] != current_logical
        or set(previous_pair) != set(runtime_sources.get(previous_source, ()))
        or set(current_pair) != set(runtime_sources.get(current_source, ()))
    ):
        return False

    profiles: list[tuple[Any, Any]] = []
    for left_logical, right_logical in zip(previous_pair, current_pair, strict=True):
        try:
            left = topology.geometry(left_logical)
            right = topology.geometry(right_logical)
        except (AttributeError, TypeError, ValueError):
            return False
        if left is None or right is None:
            return False
        profiles.append((left, right))
    for segment, (left, right) in enumerate(profiles):
        left_crop = getattr(left, "source_crop_bbox", None)
        right_crop = getattr(right, "source_crop_bbox", None)
        if (
            getattr(left, "split_kind", None) != "two_page_spread"
            or getattr(right, "split_kind", None) != "two_page_spread"
            or getattr(left, "segment_index", None) != segment
            or getattr(right, "segment_index", None) != segment
            or getattr(left, "transform_usable", None) is not True
            or getattr(right, "transform_usable", None) is not True
            or getattr(left, "selected_rotation", None) != getattr(right, "selected_rotation", None)
            or not isinstance(left_crop, tuple)
            or not isinstance(right_crop, tuple)
            or len(left_crop) != 4
            or len(right_crop) != 4
            or any(not math.isfinite(float(value)) for value in (*left_crop, *right_crop))
            or any(
                not math.isclose(
                    float(left_value),
                    float(right_value),
                    rel_tol=1e-7,
                    abs_tol=1e-6,
                )
                for left_value, right_value in zip(left_crop, right_crop, strict=True)
            )
            or not math.isclose(float(getattr(left, "width", 0)), float(getattr(right, "width", 0)))
            or not math.isclose(float(getattr(left, "height", 0)), float(getattr(right, "height", 0)))
        ):
            return False
    return True


def _local_paired_printed_adjacency(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
    *,
    previous_printed: tuple[int, int] | None,
    current_printed: tuple[int, int] | None,
    reading_order_resolution: Any,
    topology: Any,
    frozen_topology_audit_loader: Any,
) -> bool:
    """Replay one paired-inferred footer edge against its frozen topology.

    This deliberately does not promote an unresolved document-wide reading
    order.  It proves only the ordered source pair needed by the adjacent
    table continuation.  The context inference is replayed from full footers
    after removing every inferred value, so a stale or hand-authored
    ``paired_inferred_logical_pages`` label cannot authorize the edge.
    """

    if (
        previous_printed is None
        or current_printed is not None
        or _printed_identity_candidates(current_evidence)
        or not isinstance(reading_order_resolution, Mapping)
        or topology is None
        or not callable(frozen_topology_audit_loader)
    ):
        # Any exact current footer owns its own identity.  A conflicting one
        # cannot be bypassed through paired inference.
        return False
    previous_logical = _positive_int(
        getattr(previous_page, "page_number", None)
        or previous_evidence.get("page")
    )
    current_logical = _positive_int(
        getattr(current_page, "page_number", None)
        or current_evidence.get("page")
    )
    printed_total = _positive_int(reading_order_resolution.get("printed_total"))
    printed_by_logical = _logical_int_mapping(
        reading_order_resolution.get("printed_page_by_logical")
    )
    full_footer_pages = _positive_int_set(
        reading_order_resolution.get("full_footer_logical_pages")
    )
    paired_inferred_pages = _positive_int_set(
        reading_order_resolution.get("paired_inferred_logical_pages")
    )
    unresolved_pages = _positive_int_set(
        reading_order_resolution.get("unresolved_logical_pages")
    )
    page_only_footer_pages = _positive_int_set(
        reading_order_resolution.get("page_only_footer_logical_pages")
    )
    blank_pages = _positive_int_set(
        reading_order_resolution.get("blank_logical_pages")
    )
    if (
        previous_logical is None
        or current_logical is None
        or previous_logical == current_logical
        or printed_total is None
        or printed_by_logical is None
        or full_footer_pages is None
        or paired_inferred_pages is None
        or unresolved_pages is None
        or page_only_footer_pages is None
        or blank_pages is None
        or previous_logical not in full_footer_pages
        or current_logical not in paired_inferred_pages
        or current_logical in full_footer_pages
        or current_logical in page_only_footer_pages
        or current_logical in blank_pages
        or current_logical in unresolved_pages
        or previous_printed[1] != printed_total
        or printed_by_logical.get(previous_logical) != previous_printed[0]
        or printed_by_logical.get(current_logical) != previous_printed[0] + 1
        or previous_printed[0] >= printed_total
        or any(printed > printed_total for printed in printed_by_logical.values())
        or sum(
            printed == printed_by_logical[previous_logical]
            for printed in printed_by_logical.values()
        )
        != 1
        or sum(
            printed == printed_by_logical[current_logical]
            for printed in printed_by_logical.values()
        )
        != 1
    ):
        return False

    try:
        frozen_audit = frozen_topology_audit_loader()
        runtime_audit = topology.audit()
    except (AttributeError, TypeError, ValueError):
        return False
    if not isinstance(frozen_audit, Mapping) or not isinstance(runtime_audit, Mapping):
        return False
    frozen_sources = _topology_sources(frozen_audit.get("logical_pages_by_source"))
    runtime_sources = _topology_sources(runtime_audit.get("logical_pages_by_source"))
    if (
        frozen_audit.get("topology_frozen_before_reocr") is not True
        or frozen_audit.get("valid") is not True
        or runtime_audit.get("valid") is not True
        or frozen_sources is None
        or runtime_sources is None
        or frozen_sources != runtime_sources
    ):
        return False

    source_by_logical = {
        logical: source
        for source, logicals in runtime_sources.items()
        for logical in logicals
    }
    if (
        previous_logical not in source_by_logical
        or current_logical not in source_by_logical
        or source_by_logical[previous_logical] != source_by_logical[current_logical]
        or topology.ordered_pair((previous_logical, current_logical))
        != (previous_logical, current_logical)
    ):
        return False

    # Reuse the context's single authoritative repeated-spread proof instead
    # of maintaining a second crop/affine tolerance contract here.
    from docmirror.plugins.credit_report.personal_detail_scanned.context import (
        _infer_paired_printed_pages,
    )

    replayed_printed = {
        logical: printed
        for logical, printed in printed_by_logical.items()
        if logical not in paired_inferred_pages
    }
    try:
        replayed_inferred = _infer_paired_printed_pages(
            replayed_printed,
            source_by_logical,
            topology=topology,
            expected_total=printed_total,
            full_footer_logical_pages=full_footer_pages,
        )
    except (AttributeError, TypeError, ValueError, ArithmeticError):
        return False
    return bool(
        current_logical in replayed_inferred
        and replayed_printed.get(previous_logical) == previous_printed[0]
        and replayed_printed.get(current_logical) == previous_printed[0] + 1
    )


def _exact_same_box(left: Any, right: Any) -> bool:
    left_box = _bbox(left)
    right_box = _bbox(right)
    return bool(
        left_box is not None
        and right_box is not None
        and all(math.isfinite(value) for value in (*left_box, *right_box))
        and all(
            math.isclose(left_value, right_value, rel_tol=1e-7, abs_tol=1e-6)
            for left_value, right_value in zip(
                left_box,
                right_box,
                strict=True,
            )
        )
    )


def _local_paired_inquiry_entity_continuation_proved(
    entity_context: Any,
    previous_page: Any,
    previous_table: Any,
    previous_owner: Mapping[str, Any],
    current_page: Any,
    current_table: Any,
    current_lattice: Mapping[str, Any],
) -> bool:
    """Replace only the global-adjacency half of ``tables_continue``.

    The caller has already proved the frozen local paired-page edge.  This
    predicate independently preserves the other half of ``tables_continue``:
    two uniquely bound source table units must own one immutable entity and a
    unique direct ``same_table`` transition.  Exact terminal-to-leading table
    geometry prevents an entity decoder's broader section grouping from being
    mistaken for this particular physical continuation.
    """

    previous_table_id = str(getattr(previous_table, "table_id", "") or "")
    current_table_id = str(getattr(current_table, "table_id", "") or "")
    if (
        entity_context is None
        or not previous_table_id
        or not current_table_id
        or previous_table_id == current_table_id
        or previous_owner.get("template_id") != "annotations_and_inquiries"
        or previous_owner.get("table_id") not in (None, previous_table_id)
        or frozenset(previous_owner.get("header_labels") or ())
        != frozenset({"编号", "查询日期", "查询机构", "查询原因"})
        or current_lattice.get("header_binding")
        not in {
            "inherited_exact_four_role_lattice",
            "inherited_exact_sequence_date_collapsed_lattice",
        }
    ):
        return False

    previous_tables = [
        table
        for table in getattr(previous_page, "tables", None) or ()
        if _bbox(table) is not None
    ]
    current_tables = [
        table
        for table in getattr(current_page, "tables", None) or ()
        if _bbox(table) is not None
    ]
    if not previous_tables or not current_tables:
        return False
    previous_bottom = max((_bbox(table) or (0, 0, 0, 0))[3] for table in previous_tables)
    current_top = min((_bbox(table) or (0, 0, 0, 0))[1] for table in current_tables)
    previous_terminal = [
        table
        for table in previous_tables
        if math.isclose(
            (_bbox(table) or (0, 0, 0, 0))[3],
            previous_bottom,
            rel_tol=1e-7,
            abs_tol=1e-6,
        )
    ]
    current_leading = [
        table
        for table in current_tables
        if math.isclose(
            (_bbox(table) or (0, 0, 0, 0))[1],
            current_top,
            rel_tol=1e-7,
            abs_tol=1e-6,
        )
    ]
    if (
        len(previous_terminal) != 1
        or previous_terminal[0] is not previous_table
        or len(current_leading) != 1
        or current_leading[0] is not current_table
    ):
        return False

    units = getattr(entity_context, "units", None)
    entities = getattr(entity_context, "entities", None)
    decisions = getattr(entity_context, "decisions", None)
    if not all(isinstance(value, tuple) for value in (units, entities, decisions)):
        return False
    unit_ids = tuple(str(getattr(unit, "unit_id", "") or "") for unit in units)
    if not unit_ids or any(not unit_id for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
        return False
    previous_units = [
        unit
        for unit in units
        if str(getattr(unit, "table_id", "") or "") == previous_table_id
    ]
    current_units = [
        unit
        for unit in units
        if str(getattr(unit, "table_id", "") or "") == current_table_id
    ]
    if len(previous_units) != 1 or len(current_units) != 1:
        return False
    previous_unit = previous_units[0]
    current_unit = current_units[0]
    if (
        getattr(previous_unit, "kind", None) != "table"
        or getattr(current_unit, "kind", None) != "table"
        or tuple(getattr(previous_unit, "rows", ()) or ())
        != tuple(tuple(row) for row in _raw_rows(previous_table))
        or tuple(getattr(current_unit, "rows", ()) or ())
        != tuple(tuple(row) for row in _raw_rows(current_table))
        or not _exact_same_box(previous_unit, previous_table)
        or not _exact_same_box(current_unit, current_table)
        or getattr(previous_unit, "page", None) == getattr(current_unit, "page", None)
    ):
        return False
    previous_unit_id = str(getattr(previous_unit, "unit_id", "") or "")
    current_unit_id = str(getattr(current_unit, "unit_id", "") or "")
    table_unit_ids = [
        str(getattr(unit, "unit_id", "") or "")
        for unit in units
        if getattr(unit, "kind", None) == "table"
    ]
    try:
        previous_table_index = table_unit_ids.index(previous_unit_id)
        current_table_index = table_unit_ids.index(current_unit_id)
    except ValueError:
        return False
    if current_table_index != previous_table_index + 1:
        return False

    previous_entities = [
        entity
        for entity in entities
        if previous_unit_id in tuple(getattr(entity, "unit_ids", ()) or ())
    ]
    current_entities = [
        entity
        for entity in entities
        if current_unit_id in tuple(getattr(entity, "unit_ids", ()) or ())
    ]
    if len(previous_entities) != 1 or len(current_entities) != 1:
        return False
    previous_entity = previous_entities[0]
    current_entity = current_entities[0]
    entity_id = str(getattr(previous_entity, "entity_id", "") or "")
    entity_unit_ids = getattr(previous_entity, "unit_ids", None)
    if (
        not entity_id
        or previous_entity is not current_entity
        or str(getattr(current_entity, "entity_id", "") or "") != entity_id
        or getattr(previous_entity, "kind", None) not in {"table", "mixed"}
        or not isinstance(entity_unit_ids, tuple)
        or len(entity_unit_ids) != len(set(entity_unit_ids))
        or previous_unit_id not in entity_unit_ids
        or current_unit_id not in entity_unit_ids
        or entity_unit_ids.index(current_unit_id)
        != entity_unit_ids.index(previous_unit_id) + 1
    ):
        return False

    pair_decisions = [
        decision
        for decision in decisions
        if str(getattr(decision, "left_unit_id", "") or "") == previous_unit_id
        and str(getattr(decision, "right_unit_id", "") or "") == current_unit_id
    ]
    if len(pair_decisions) != 1:
        return False
    pair_decision = pair_decisions[0]
    hypotheses = getattr(pair_decision, "hypotheses", None)
    confidence = getattr(pair_decision, "confidence", None)
    if (
        getattr(pair_decision, "selected", None) != "same_table"
        or getattr(pair_decision, "from_page", None)
        == getattr(pair_decision, "to_page", None)
        or not isinstance(hypotheses, tuple)
        or not hypotheses
        or getattr(hypotheses[0], "action", None) != "same_table"
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or float(confidence) <= 0.0
    ):
        return False
    continuation_actions = {
        "same_table",
        "table_to_text_related",
        "text_to_table_related",
        "same_text_section",
    }
    competing = [
        decision
        for decision in decisions
        if decision is not pair_decision
        and getattr(decision, "selected", None) in continuation_actions
        and (
            str(getattr(decision, "left_unit_id", "") or "")
            == previous_unit_id
            or str(getattr(decision, "right_unit_id", "") or "")
            == current_unit_id
        )
    ]
    return not competing


def _exact_empty_table_border_row(
    table: Any,
    *,
    row: int,
    width: int,
) -> bool:
    """Prove one terminal ruled band contains no source-owned content.

    Native bordered-table reconstruction may retain a narrow final band below
    the last populated PBOC row.  It is furniture only when every physical cell
    has exact finite geometry and both evidence/token inventories are empty.
    The caller applies this proof only after the final populated row, so an
    internal blank row can never be skipped as a continuation heuristic.
    """

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return False
    geometry = metadata.get("geometry")
    grids = [owner for owner in (geometry, metadata) if isinstance(owner, Mapping)]
    table_bbox = _bbox(table)
    if (
        table_bbox is None
        or not all(math.isfinite(value) for value in table_bbox)
        or width <= 0
    ):
        return False
    boxes: list[tuple[float, float, float, float]] = []
    for column in range(width):
        candidates: set[tuple[float, float, float, float]] = set()
        for owner in grids:
            statuses = owner.get("cell_geometry_status")
            bboxes = owner.get("cell_bboxes")
            evidence_ids = owner.get("cell_evidence_ids")
            token_ids = owner.get("cell_token_ids")
            if not all(isinstance(grid, list) for grid in (statuses, bboxes, evidence_ids)):
                continue
            if any(
                row < 0
                or row >= len(grid)
                or not isinstance(grid[row], list)
                or column < 0
                or column >= len(grid[row])
                for grid in (statuses, bboxes, evidence_ids)
            ):
                continue
            if str(statuses[row][column] or "") != "exact":
                continue
            raw_bbox = bboxes[row][column]
            bbox = (
                tuple(_finite(value) for value in raw_bbox[:4])
                if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4
                else None
            )
            raw_evidence_ids = evidence_ids[row][column]
            if (
                bbox is None
                or not all(math.isfinite(value) for value in bbox)
                or not (bbox[2] > bbox[0] and bbox[3] > bbox[1])
                or not isinstance(raw_evidence_ids, list)
                or raw_evidence_ids
            ):
                continue
            if isinstance(token_ids, list):
                if (
                    row >= len(token_ids)
                    or not isinstance(token_ids[row], list)
                    or column >= len(token_ids[row])
                    or not isinstance(token_ids[row][column], list)
                    or token_ids[row][column]
                ):
                    continue
            if (
                bbox[0] < table_bbox[0] - 1e-6
                or bbox[1] < table_bbox[1] - 1e-6
                or bbox[2] > table_bbox[2] + 1e-6
                or bbox[3] > table_bbox[3] + 1e-6
            ):
                continue
            candidates.add(bbox)
        if len(candidates) != 1:
            return False
        boxes.append(next(iter(candidates)))
    return _boxes_have_disjoint_interiors(boxes)


def _classify_page(
    page: Any,
    evidence: Mapping[str, Any],
    *,
    allow_legacy_text_only: bool = False,
) -> tuple[str, float, tuple[str, ...]] | None:
    """Classify a whole page from one unique sealed PBOC section heading.

    Registered titles repeat in contents pages, summaries, legends, and prose.
    A flattened substring therefore cannot own a page. Runtime classification
    requires exactly one complete registered heading line with sealed evidence
    and a finite box contained by the page. Multiple semantic roles fail
    closed, including mixed pages whose lower section must be extracted by its
    own geometry-bound handler. The deliberately named text-only seam exists
    only for legacy unit contracts and is never selected by the assembler.
    """

    flattened_result = _classify(_page_text(page, evidence))
    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    runtime_lines = [
        line
        for line in evidence.get("lines") or ()
        if isinstance(line, Mapping)
        and _compact(line.get("text") or line.get("content") or "")
    ]
    if not runtime_lines:
        return flattened_result if allow_legacy_text_only else None

    exact_lines: list[tuple[float, str]] = []
    for line in runtime_lines:
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None:
            continue
        # Containment is the only page-relative coordinate contract here; no
        # absolute character or pixel cutoff decides the role.
        if page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width):
            continue
        if page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height):
            continue
        exact_lines.append((bbox[1], _compact(line.get("text") or line.get("content") or "")))

    heading_lines: list[tuple[str, str, str]] = []
    subsection_roles: set[str] = set()
    for _top, text in exact_lines:
        title = canonical_registered_section_heading(text)
        if title is not None:
            template_id = REGISTERED_SECTION_TEMPLATE_BY_TITLE[title]
            heading_lines.append((template_id, title, "registered_section"))
            continue
        account_family = canonical_account_family_heading(text)
        if account_family is not None:
            heading_lines.append(
                ("credit_account_detail", account_family, "account_family")
            )
            continue
        subsection = canonical_registered_subsection_heading(text)
        if subsection is not None:
            # A subsection never owns the whole page.  It is nevertheless a
            # decisive veto when a different top-level role occurs earlier on
            # the same logical page; that page must be resolved table-locally.
            subsection_roles.add(subsection[0])
    if not heading_lines:
        return None
    roles = {
        *(template_id for template_id, _title, _kind in heading_lines),
        *subsection_roles,
    }
    physical_titles = {(kind, title) for _role, title, kind in heading_lines}
    registered = [item for item in heading_lines if item[2] == "registered_section"]
    # One top-level PBOC section may contain several distinct subordinate
    # account-family headings on the same physical page.  Their agreement on
    # the semantic page role is positive evidence; section encounter order and
    # printed numerals are not.  Duplicated atoms, two top-level owners, or any
    # cross-role mixture remain unresolved for section-local handling.
    if (
        len(roles) != 1
        or len(physical_titles) != len(heading_lines)
        or len(registered) > 1
    ):
        return None
    template_id = next(iter(roles))
    titles = tuple(title for _role, title, _kind in heading_lines)
    return template_id, 0.99, titles


@dataclass(frozen=True)
class _DenseAccountTableOwner:
    table_id: str
    bbox: tuple[float, float, float, float]
    evidence_ids: tuple[str, ...]
    evidence_cells: tuple[
        tuple[str, str, int, int, tuple[float, float, float, float]],
        ...,
    ]


@dataclass(frozen=True)
class _ExactLiabilityTableOwner:
    """One complete, exact PBOC repayment-responsibility record table."""

    table_id: str
    bbox: tuple[float, float, float, float]
    evidence_ids: tuple[str, ...]
    evidence_cells: tuple[
        tuple[str, str, int, int, tuple[float, float, float, float]],
        ...,
    ]


def _page_evidence_owners(
    page: Any,
    evidence: Mapping[str, Any],
) -> tuple[
    dict[str, set[tuple[str, int, int]]],
    dict[str, list[tuple[int, tuple[float, float, float, float] | None]]],
]:
    """Inventory every page-local cell/line owner of each evidence atom.

    Geometry and promoted metadata may repeat the same cell matrix.  Those
    copies describe one physical owner and therefore collapse on
    ``(table_id, row, column)``.  A different cell or line remains a distinct
    owner and can veto an authority-bearing evidence replay.
    """

    cell_owners: defaultdict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for table_index, table in enumerate(getattr(page, "tables", None) or ()):
        table_id = str(getattr(table, "table_id", "") or f"table-index:{table_index}")
        metadata = getattr(table, "metadata", None) or {}
        if not isinstance(metadata, Mapping):
            continue
        geometry = metadata.get("geometry")
        for owner in (geometry, metadata):
            if not isinstance(owner, Mapping):
                continue
            grid = owner.get("cell_evidence_ids")
            if not isinstance(grid, list):
                continue
            for row, cells in enumerate(grid):
                if not isinstance(cells, list):
                    continue
                for column, raw_ids in enumerate(cells):
                    if not isinstance(raw_ids, list):
                        continue
                    for raw_id in raw_ids:
                        if isinstance(raw_id, str) and raw_id.strip():
                            cell_owners[raw_id.strip()].add((table_id, row, column))

    line_owners: defaultdict[
        str,
        list[tuple[int, tuple[float, float, float, float] | None]],
    ] = defaultdict(list)
    for line_index, line in enumerate(evidence.get("lines") or ()):
        if not isinstance(line, Mapping):
            continue
        raw_ids = line.get("evidence_ids")
        if not isinstance(raw_ids, list):
            continue
        line_box = _bbox(line)
        for raw_id in raw_ids:
            if isinstance(raw_id, str) and raw_id.strip():
                line_owners[raw_id.strip()].append((line_index, line_box))
    return dict(cell_owners), dict(line_owners)


def _box_is_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    return bool(
        inner[0] >= outer[0] - 1e-6
        and inner[1] >= outer[1] - 1e-6
        and inner[2] <= outer[2] + 1e-6
        and inner[3] <= outer[3] + 1e-6
    )


def _continuation_deciding_evidence_is_page_unique(
    page: Any,
    evidence: Mapping[str, Any],
    *,
    table_owners: Sequence[_DenseAccountTableOwner | _ExactLiabilityTableOwner],
    anchor_owners: Mapping[str, tuple[int, tuple[float, float, float, float]]],
) -> bool:
    """Require each authority atom to have exactly one physical page owner."""

    cells, lines = _page_evidence_owners(page, evidence)
    deciding_cells = {
        evidence_id: (table_id, row, column, cell_box)
        for owner in table_owners
        for evidence_id, table_id, row, column, cell_box in owner.evidence_cells
    }
    if len(deciding_cells) != sum(len(owner.evidence_cells) for owner in table_owners):
        return False
    for evidence_id, (table_id, row, column, cell_box) in deciding_cells.items():
        if cells.get(evidence_id, set()) != {(table_id, row, column)}:
            return False
        line_aliases = lines.get(evidence_id, [])
        if len(line_aliases) > 1 or any(
            line_box is None or not _box_is_inside(line_box, cell_box)
            for _line_index, line_box in line_aliases
        ):
            return False
    for evidence_id, expected_line_owner in anchor_owners.items():
        if cells.get(evidence_id):
            return False
        line_aliases = lines.get(evidence_id, [])
        if len(line_aliases) != 1 or line_aliases[0] != expected_line_owner:
            return False
    return True


def _dense_account_table_owners(page: Any) -> tuple[_DenseAccountTableOwner, ...]:
    """Return source tables with an exact, sealed PBOC account schema.

    This is deliberately a closed label-cell contract, not a substring score.
    A header must expose the institution, account identifier, and opening-date
    roles as distinct cells, at least three further registered account roles,
    and an immediately following populated value row.  Every deciding header
    and identity value cell must have one immutable exact geometry/evidence
    owner.  Business values never influence the decision.
    """

    result: list[_DenseAccountTableOwner] = []
    observed_table_ids: set[str] = set()
    observed_evidence_ids: set[str] = set()
    for table in getattr(page, "tables", None) or ():
        table_box = _bbox(table)
        table_id = str(getattr(table, "table_id", "") or "")
        if table_box is None or not table_id or table_id in observed_table_ids:
            return ()
        observed_table_ids.add(table_id)
        rows = _raw_rows(table)
        candidates: list[_DenseAccountTableOwner] = []
        for row_index, row in enumerate(rows[:-1]):
            labels = tuple(_compact(cell) for cell in row)
            recognized_columns = tuple(
                (column, label)
                for column, label in enumerate(labels)
                if label in _ACCOUNT_TABLE_LABELS
            )
            recognized = tuple(label for _column, label in recognized_columns)
            if len(recognized) < 6 or len(set(recognized)) != len(recognized):
                continue
            if (
                len({"管理机构", "发卡机构"}.intersection(recognized)) != 1
                or "账户标识" not in recognized
                or "开立日期" not in recognized
            ):
                continue
            values = rows[row_index + 1]
            required_columns = tuple(
                index
                for index, label in enumerate(labels)
                if label in {"管理机构", "发卡机构", "账户标识", "开立日期"}
            )
            if (
                len(values) < len(labels)
                or any(not _compact(values[index]) for index in required_columns)
            ):
                continue
            header_owners = [
                _exact_table_cell_owner(table, row=row_index, column=column)
                for column, _label in recognized_columns
            ]
            value_owners = [
                _exact_table_cell_owner(table, row=row_index + 1, column=column)
                for column in required_columns
            ]
            if any(owner is None for owner in (*header_owners, *value_owners)):
                continue
            exact_owner_cells = [
                (row_index, column, owner)
                for (column, _label), owner in zip(
                    recognized_columns,
                    header_owners,
                    strict=True,
                )
                if owner is not None
            ]
            exact_owner_cells.extend(
                (row_index + 1, column, owner)
                for column, owner in zip(required_columns, value_owners, strict=True)
                if owner is not None
            )
            exact_owners = [owner for _row, _column, owner in exact_owner_cells]
            cell_boxes = [bbox for bbox, _ids in exact_owners]
            cell_evidence_ids = [
                evidence_id
                for _bbox_value, evidence_ids in exact_owners
                for evidence_id in evidence_ids
            ]
            if (
                len(cell_evidence_ids) != len(set(cell_evidence_ids))
                or not _boxes_have_disjoint_interiors(cell_boxes)
                or any(
                    bbox[0] < table_box[0] - 1e-6
                    or bbox[1] < table_box[1] - 1e-6
                    or bbox[2] > table_box[2] + 1e-6
                    or bbox[3] > table_box[3] + 1e-6
                    for bbox in cell_boxes
                )
            ):
                continue
            candidates.append(
                _DenseAccountTableOwner(
                    table_id=table_id,
                    bbox=table_box,
                    evidence_ids=tuple(cell_evidence_ids),
                    evidence_cells=tuple(
                        (
                            evidence_id,
                            table_id,
                            cell_row,
                            cell_column,
                            cell_box,
                        )
                        for cell_row, cell_column, (cell_box, evidence_ids) in exact_owner_cells
                        for evidence_id in evidence_ids
                    ),
                )
            )
        if len(candidates) > 1:
            return ()
        if candidates:
            candidate = candidates[0]
            if observed_evidence_ids.intersection(candidate.evidence_ids):
                return ()
            observed_evidence_ids.update(candidate.evidence_ids)
            result.append(candidate)
    return tuple(result)


def _dense_account_table_boxes(page: Any) -> tuple[tuple[float, float, float, float], ...]:
    """Compatibility view of the sealed account-table owners."""

    return tuple(owner.bbox for owner in _dense_account_table_owners(page))


def _liability_exact_role_columns(
    row: Sequence[Any],
    *,
    role_sets_by_label: Mapping[str, frozenset[str]],
    expected_roles: frozenset[str],
) -> dict[str, int] | None:
    """Bind a header row only through a finite, non-overlapping role grammar."""

    role_columns: dict[str, int] = {}
    for column, value in enumerate(row):
        label = _compact(value)
        if not label:
            continue
        roles = role_sets_by_label.get(label)
        if roles is None or roles.intersection(role_columns):
            return None
        for role in roles:
            role_columns[role] = column
    return role_columns if set(role_columns) == expected_roles else None


def _liability_exact_value_columns(
    row: Sequence[Any],
    *,
    role_columns: Mapping[str, int],
    required_roles: frozenset[str],
) -> tuple[int, ...] | None:
    """Consume one value row without admitting values in inactive columns."""

    active_columns = frozenset(role_columns.values())
    populated_columns = tuple(
        column for column, value in enumerate(row) if _compact(value)
    )
    if any(column not in active_columns for column in populated_columns):
        return None
    if any(
        column >= len(row) or not _compact(row[column])
        for role, column in role_columns.items()
        if role in required_roles
    ):
        return None
    return populated_columns


def _liability_cell_lattice_is_closed(
    table: Any,
    rows: Sequence[Sequence[Any]],
) -> bool:
    """Reject textual or evidence residue outside the visible cell lattice.

    Exact non-empty cells must have immutable evidence.  Empty merged,
    nullable, and border cells may be exact or derived, but cannot hide an
    evidence token.  Multiple serialized geometry copies must agree.
    """

    metadata = getattr(table, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return False
    geometry = metadata.get("geometry")
    grids = tuple(
        owner for owner in (geometry, metadata) if isinstance(owner, Mapping)
    )
    candidates: set[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    ] = set()
    for owner in grids:
        statuses = owner.get("cell_geometry_status")
        evidence_ids = owner.get("cell_evidence_ids")
        if not isinstance(statuses, list) or not isinstance(evidence_ids, list):
            continue
        if len(statuses) != len(rows) or len(evidence_ids) != len(rows):
            return False
        signature: list[tuple[tuple[str, tuple[str, ...]], ...]] = []
        valid = True
        for row_index, row in enumerate(rows):
            status_row = statuses[row_index]
            id_row = evidence_ids[row_index]
            if (
                not isinstance(status_row, list)
                or not isinstance(id_row, list)
                or len(status_row) != len(row)
                or len(id_row) != len(row)
            ):
                valid = False
                break
            sealed_row: list[tuple[str, tuple[str, ...]]] = []
            for column, value in enumerate(row):
                status = str(status_row[column] or "")
                raw_ids = id_row[column]
                if (
                    status not in {"exact", "derived"}
                    or not isinstance(raw_ids, list)
                    or any(
                        not isinstance(evidence_id, str) or not evidence_id.strip()
                        for evidence_id in raw_ids
                    )
                ):
                    valid = False
                    break
                sealed_ids = tuple(evidence_id.strip() for evidence_id in raw_ids)
                if (
                    len(sealed_ids) != len(set(sealed_ids))
                    or (_compact(value) and (status != "exact" or not sealed_ids))
                    or (not _compact(value) and sealed_ids)
                ):
                    valid = False
                    break
                sealed_row.append((status, sealed_ids))
            if not valid:
                break
            signature.append(tuple(sealed_row))
        if valid:
            candidates.add(tuple(signature))
    return len(candidates) == 1


def _exact_liability_table_owner(table: Any) -> _ExactLiabilityTableOwner | None:
    """Prove and completely consume one finite PBOC liability-card graph.

    A full card is the exact seven-row graph: base header/value, related-party
    header/value, snapshot, and status header/value.  The only other admitted
    graph is the five-row page-boundary form ending at the snapshot.  Empty
    physical rows are permitted only when their cell lattice is sealed and has
    no evidence residue.  Thus a valid base pair cannot lend ownership to an
    unrelated trailing row, cross-section fragment, or inactive-column value.
    """

    table_id = str(getattr(table, "table_id", "") or "")
    table_bbox = _bbox(table)
    rows = _raw_rows(table)
    if (
        not table_id
        or table_bbox is None
        or not rows
        or not _liability_cell_lattice_is_closed(table, rows)
    ):
        return None

    nonempty_rows = tuple(
        row_index
        for row_index, row in enumerate(rows)
        if any(_compact(value) for value in row)
    )
    if len(nonempty_rows) not in {5, 7}:
        return None

    base_header_row, base_value_row, party_header_row, party_value_row, snapshot_row = (
        nonempty_rows[:5]
    )
    base_role_columns = _liability_exact_role_columns(
        rows[base_header_row],
        role_sets_by_label={
            label: frozenset({role})
            for label, role in _LIABILITY_HEADER_ROLE_BY_LABEL.items()
        },
        expected_roles=_LIABILITY_HEADER_ROLES,
    )
    if base_role_columns is None:
        return None
    base_value_columns = _liability_exact_value_columns(
        rows[base_value_row],
        role_columns=base_role_columns,
        required_roles=_LIABILITY_REQUIRED_VALUE_ROLES,
    )
    if base_value_columns is None:
        return None

    party_role_columns = _liability_exact_role_columns(
        rows[party_header_row],
        role_sets_by_label=_LIABILITY_PARTY_ROLE_SETS_BY_LABEL,
        expected_roles=_LIABILITY_PARTY_ROLES,
    )
    if party_role_columns is None:
        return None
    party_value_columns = _liability_exact_value_columns(
        rows[party_value_row],
        role_columns=party_role_columns,
        required_roles=_LIABILITY_PARTY_ROLES,
    )
    if party_value_columns is None:
        return None

    snapshot_columns = tuple(
        column
        for column, value in enumerate(rows[snapshot_row])
        if _compact(value)
    )
    if (
        len(snapshot_columns) != 1
        or _LIABILITY_SNAPSHOT_RE.fullmatch(
            _compact(rows[snapshot_row][snapshot_columns[0]])
        )
        is None
    ):
        return None

    selected_cells = [
        (row_index, column)
        for row_index in nonempty_rows[:5]
        for column, value in enumerate(rows[row_index])
        if _compact(value)
    ]
    if len(nonempty_rows) == 7:
        status_header_row, status_value_row = nonempty_rows[5:]
        status_role_columns = _liability_exact_role_columns(
            rows[status_header_row],
            role_sets_by_label={
                label: frozenset({role})
                for label, role in _LIABILITY_STATUS_ROLE_BY_LABEL.items()
            },
            expected_roles=_LIABILITY_STATUS_ROLES,
        )
        if status_role_columns is None:
            return None
        status_value_columns = _liability_exact_value_columns(
            rows[status_value_row],
            role_columns=status_role_columns,
            required_roles=_LIABILITY_STATUS_ROLES,
        )
        if status_value_columns is None:
            return None
        selected_cells.extend(
            (row_index, column)
            for row_index in (status_header_row, status_value_row)
            for column, value in enumerate(rows[row_index])
            if _compact(value)
        )

    # These columns were independently consumed above; the equality makes the
    # closed-world invariant explicit and guards later grammar edits.
    expected_selected_cells = {
        (row_index, column)
        for row_index, row in enumerate(rows)
        for column, value in enumerate(row)
        if _compact(value)
    }
    if set(selected_cells) != expected_selected_cells:
        return None

    exact_cells = [
        (
            row_index,
            column,
            _exact_table_cell_owner(table, row=row_index, column=column),
        )
        for row_index, column in selected_cells
    ]
    if any(owner is None for _row, _column, owner in exact_cells):
        return None
    resolved_cells = [
        (row_index, column, owner)
        for row_index, column, owner in exact_cells
        if owner is not None
    ]
    boxes = [owner[0] for _row, _column, owner in resolved_cells]
    evidence_ids = [
        evidence_id
        for _row, _column, (_box_value, owners) in resolved_cells
        for evidence_id in owners
    ]
    if (
        len(evidence_ids) != len(set(evidence_ids))
        or not _boxes_have_disjoint_interiors(boxes)
        or any(not _box_is_inside(box, table_bbox) for box in boxes)
    ):
        return None
    return _ExactLiabilityTableOwner(
        table_id=table_id,
        bbox=table_bbox,
        evidence_ids=tuple(evidence_ids),
        evidence_cells=tuple(
            (
                evidence_id,
                table_id,
                row_index,
                column,
                cell_box,
            )
            for row_index, column, (cell_box, owners) in resolved_cells
            for evidence_id in owners
        ),
    )


def _exact_liability_page_table_owners(
    page: Any,
) -> tuple[_ExactLiabilityTableOwner, ...]:
    """Return owners only when every physical table is one liability record."""

    tables = tuple(getattr(page, "tables", None) or ())
    if not tables:
        return ()
    owners: list[_ExactLiabilityTableOwner] = []
    table_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for table in tables:
        owner = _exact_liability_table_owner(table)
        if (
            owner is None
            or owner.table_id in table_ids
            or evidence_ids.intersection(owner.evidence_ids)
        ):
            return ()
        table_ids.add(owner.table_id)
        evidence_ids.update(owner.evidence_ids)
        owners.append(owner)
    return tuple(owners)


def _exact_liability_card_anchors(
    page: Any,
    evidence: Mapping[str, Any],
    *,
    after_top: float = -math.inf,
) -> tuple[dict[str, Any], ...]:
    """Return a dense run of exact, sealed ``账户 N`` line owners."""

    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    anchors: list[dict[str, Any]] = []
    consumed_ids: set[str] = set()
    for line_index, line in enumerate(evidence.get("lines") or ()):
        if not isinstance(line, Mapping):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        match = _ACCOUNT_CARD_HEADING_RE.fullmatch(text)
        if match is None:
            continue
        loose_bbox = _bbox(line)
        if loose_bbox is not None and loose_bbox[1] < after_top:
            continue
        bbox = _exact_evidence_line_bbox(line)
        raw_ids = line.get("evidence_ids")
        if (
            bbox is None
            or bbox[1] < after_top
            or (page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width))
            or (page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height))
            or not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(value, str) or not value.strip() for value in raw_ids)
        ):
            return ()
        sealed_ids = tuple(value.strip() for value in raw_ids)
        if (
            len(sealed_ids) != len(set(sealed_ids))
            or consumed_ids.intersection(sealed_ids)
        ):
            return ()
        consumed_ids.update(sealed_ids)
        anchors.append(
            {
                "sequence": int(match.group("sequence")),
                "bbox": bbox,
                "evidence_ids": sealed_ids,
                "line_index": line_index,
            }
        )
    anchors.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    sequences = [int(item["sequence"]) for item in anchors]
    if (
        not anchors
        or len(sequences) != len(set(sequences))
        or sequences != list(range(sequences[0], sequences[0] + len(sequences)))
        or not _boxes_have_disjoint_interiors(item["bbox"] for item in anchors)
    ):
        return ()
    return tuple(anchors)


def _exact_liability_section_heading_owner(
    page: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the one exact heading that independently registered the section."""

    if _sealed_registered_heading_roles(page, evidence) != frozenset(
        {"repayment_responsibility"}
    ):
        return None
    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    candidates: list[dict[str, Any]] = []
    for line_index, line in enumerate(evidence.get("lines") or ()):
        if not isinstance(line, Mapping):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        if canonical_registered_section_heading(text) != "相关还款责任信息":
            continue
        bbox = _exact_evidence_line_bbox(line)
        raw_ids = line.get("evidence_ids")
        if (
            bbox is None
            or (page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width))
            or (page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height))
            or not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(value, str) or not value.strip() for value in raw_ids)
        ):
            return None
        sealed_ids = tuple(value.strip() for value in raw_ids)
        if len(sealed_ids) != len(set(sealed_ids)):
            return None
        candidates.append(
            {
                "bbox": bbox,
                "evidence_ids": sealed_ids,
                "line_index": line_index,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def _exact_printed_footer_owner(
    page: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return one uniquely sealed bottom footer and its printed identity."""

    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    candidates: list[dict[str, Any]] = []
    for line_index, line in enumerate(evidence.get("lines") or ()):
        if not isinstance(line, Mapping):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        match = _PRINTED_PAGE_RE.fullmatch(text)
        if match is None:
            continue
        bbox = _exact_evidence_line_bbox(line)
        raw_ids = line.get("evidence_ids")
        if (
            bbox is None
            or not _bottom_footer_geometry(line, page_height=page_height)
            or (page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width))
            or not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(value, str) or not value.strip() for value in raw_ids)
        ):
            return None
        sealed_ids = tuple(value.strip() for value in raw_ids)
        printed_page = int(match.group("page"))
        printed_total = int(match.group("total"))
        if (
            len(sealed_ids) != len(set(sealed_ids))
            or not 1 <= printed_page <= printed_total
        ):
            return None
        candidates.append(
            {
                "printed_page": printed_page,
                "printed_total": printed_total,
                "bbox": bbox,
                "evidence_ids": sealed_ids,
                "line_index": line_index,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def _liability_anchor_table_bijection(
    anchors: Sequence[Mapping[str, Any]],
    table_owners: Sequence[_ExactLiabilityTableOwner],
    *,
    boundary_top: float,
) -> bool:
    """Prove one geometrically attached table for every printed card anchor."""

    candidate_tables_by_anchor: list[set[str]] = []
    candidate_anchors_by_table = {owner.table_id: set() for owner in table_owners}
    for index, anchor in enumerate(anchors):
        anchor_box = anchor.get("bbox")
        if not isinstance(anchor_box, tuple):
            return False
        next_top = (
            float(anchors[index + 1]["bbox"][1])
            if index + 1 < len(anchors)
            else boundary_top
        )
        candidates: set[str] = set()
        for owner in table_owners:
            horizontal_overlap = max(
                0.0,
                min(anchor_box[2], owner.bbox[2])
                - max(anchor_box[0], owner.bbox[0]),
            )
            if (
                horizontal_overlap > 0.0
                and _heading_attaches_to_table(anchor_box, owner.bbox)
                and owner.bbox[3] <= next_top + 1e-6
            ):
                candidates.add(owner.table_id)
                candidate_anchors_by_table[owner.table_id].add(index)
        candidate_tables_by_anchor.append(candidates)
    return bool(
        len(anchors) == len(table_owners)
        and all(len(candidates) == 1 for candidates in candidate_tables_by_anchor)
        and all(len(candidates) == 1 for candidates in candidate_anchors_by_table.values())
    )


def _sealed_liability_page_continuation_proved(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    previous_registration: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
) -> bool:
    """Prove a complete headingless liability-card continuation page.

    No single signal owns the page.  The grant requires an independently
    heading-registered preceding section, consecutive exact printed footers,
    the terminal prior card/table pair, a dense next ordinal run, a complete
    liability table population, and unique immutable evidence for every
    deciding line and cell.
    """

    if (
        previous_registration.get("status") != "registered"
        or previous_registration.get("template_id") != "repayment_responsibility"
        or previous_registration.get("basis") != "source_page_evidence"
        or _sealed_registered_heading_roles(current_page, current_evidence)
    ):
        return False
    section = _exact_liability_section_heading_owner(
        previous_page,
        previous_evidence,
    )
    previous_footer = _exact_printed_footer_owner(previous_page, previous_evidence)
    current_footer = _exact_printed_footer_owner(current_page, current_evidence)
    if section is None or previous_footer is None or current_footer is None:
        return False
    if (
        previous_footer["printed_total"] != current_footer["printed_total"]
        or current_footer["printed_page"] != previous_footer["printed_page"] + 1
    ):
        return False

    previous_anchors = _exact_liability_card_anchors(
        previous_page,
        previous_evidence,
        after_top=float(section["bbox"][3]),
    )
    current_anchors = _exact_liability_card_anchors(current_page, current_evidence)
    if (
        not previous_anchors
        or not current_anchors
        or current_anchors[0]["sequence"] != previous_anchors[-1]["sequence"] + 1
    ):
        return False

    previous_tables = tuple(getattr(previous_page, "tables", None) or ())
    previous_table_boxes = [
        (table, _bbox(table))
        for table in previous_tables
    ]
    if not previous_table_boxes or any(box is None for _table, box in previous_table_boxes):
        return False
    terminal_top = max(
        box[1]
        for _table, box in previous_table_boxes
        if box is not None
    )
    terminal_tables = [
        table
        for table, box in previous_table_boxes
        if box is not None and math.isclose(box[1], terminal_top, abs_tol=1e-6)
    ]
    if len(terminal_tables) != 1:
        return False
    terminal_owner = _exact_liability_table_owner(terminal_tables[0])
    terminal_anchor = previous_anchors[-1]
    if (
        terminal_owner is None
        or not _liability_anchor_table_bijection(
            (terminal_anchor,),
            (terminal_owner,),
            boundary_top=float(previous_footer["bbox"][1]),
        )
    ):
        return False

    current_table_owners = _exact_liability_page_table_owners(current_page)
    if not current_table_owners or not _liability_anchor_table_bijection(
        current_anchors,
        current_table_owners,
        boundary_top=float(current_footer["bbox"][1]),
    ):
        return False

    previous_line_owners = {
        evidence_id: (int(owner["line_index"]), owner["bbox"])
        for owner in (section, *previous_anchors, previous_footer)
        for evidence_id in owner["evidence_ids"]
    }
    current_line_owners = {
        evidence_id: (int(owner["line_index"]), owner["bbox"])
        for owner in (*current_anchors, current_footer)
        for evidence_id in owner["evidence_ids"]
    }
    previous_ids = {
        *previous_line_owners,
        *terminal_owner.evidence_ids,
    }
    current_ids = {
        *current_line_owners,
        *(
            evidence_id
            for owner in current_table_owners
            for evidence_id in owner.evidence_ids
        ),
    }
    previous_id_count = sum(
        len(owner["evidence_ids"])
        for owner in (section, *previous_anchors, previous_footer)
    ) + len(terminal_owner.evidence_ids)
    current_id_count = sum(
        len(owner["evidence_ids"])
        for owner in (*current_anchors, current_footer)
    ) + sum(len(owner.evidence_ids) for owner in current_table_owners)
    if (
        len(previous_ids) != previous_id_count
        or len(current_ids) != current_id_count
        or previous_ids.intersection(current_ids)
        or not _continuation_deciding_evidence_is_page_unique(
            previous_page,
            previous_evidence,
            table_owners=(terminal_owner,),
            anchor_owners=previous_line_owners,
        )
        or not _continuation_deciding_evidence_is_page_unique(
            current_page,
            current_evidence,
            table_owners=current_table_owners,
            anchor_owners=current_line_owners,
        )
    ):
        return False
    return True


def _sealed_account_card_continuation_proved(
    page: Any,
    evidence: Mapping[str, Any],
) -> bool:
    """Prove one headingless account fragment from local semantic geometry.

    The enclosing account section must already be active in authoritative
    printed order.  Locally, this predicate additionally requires a unique
    sealed ``账户 N`` anchor geometry-bound to a dense source account table.
    Any complete registered section heading makes the whole-page role mixed or
    independently classifiable and therefore vetoes continuation adoption.
    """

    page_width = _finite(evidence.get("page_width") or getattr(page, "width", 0))
    page_height = _finite(evidence.get("page_height") or getattr(page, "height", 0))
    anchors: list[tuple[int, tuple[float, float, float, float], tuple[str, ...]]] = []
    anchor_evidence_ids: set[str] = set()
    anchor_evidence_owners: dict[
        str,
        tuple[int, tuple[float, float, float, float]],
    ] = {}
    for line_index, line in enumerate(evidence.get("lines") or ()):
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None:
            continue
        if page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width):
            continue
        if page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        if (
            canonical_registered_section_heading(text) is not None
            or canonical_account_family_heading(text) is not None
        ):
            return False
        match = _ACCOUNT_CARD_HEADING_RE.fullmatch(text)
        if match is not None:
            raw_ids = line.get("evidence_ids") if isinstance(line, Mapping) else None
            if not isinstance(raw_ids, list) or not raw_ids:
                return False
            if any(not isinstance(value, str) or not value.strip() for value in raw_ids):
                return False
            sealed_ids = tuple(value.strip() for value in raw_ids)
            if (
                len(sealed_ids) != len(set(sealed_ids))
                or anchor_evidence_ids.intersection(sealed_ids)
            ):
                return False
            anchor_evidence_ids.update(sealed_ids)
            anchor_evidence_owners.update(
                {evidence_id: (line_index, bbox) for evidence_id in sealed_ids}
            )
            anchors.append((int(match.group("sequence")), bbox, sealed_ids))
    if not anchors or len({sequence for sequence, _bbox_value, _ids in anchors}) != len(anchors):
        return False

    table_owners = _dense_account_table_owners(page)
    if len(table_owners) != len(anchors) or not table_owners:
        return False
    table_evidence_ids = {
        evidence_id
        for owner in table_owners
        for evidence_id in owner.evidence_ids
    }
    if anchor_evidence_ids.intersection(table_evidence_ids):
        return False
    if not _continuation_deciding_evidence_is_page_unique(
        page,
        evidence,
        table_owners=table_owners,
        anchor_owners=anchor_evidence_owners,
    ):
        return False

    ordered_anchors = sorted(anchors, key=lambda item: (item[1][1], item[1][0]))
    if any(
        right_sequence <= left_sequence or left_box[1] >= right_box[1]
        for (left_sequence, left_box, _left_ids), (right_sequence, right_box, _right_ids) in zip(
            ordered_anchors,
            ordered_anchors[1:],
        )
    ):
        return False
    candidate_tables_by_anchor: list[set[str]] = []
    candidate_anchors_by_table: dict[str, set[int]] = {
        owner.table_id: set() for owner in table_owners
    }
    for index, (_sequence, anchor_box, _anchor_ids) in enumerate(ordered_anchors):
        next_top = (
            ordered_anchors[index + 1][1][1]
            if index + 1 < len(ordered_anchors)
            else page_height if page_height > 0 else math.inf
        )
        candidates: set[str] = set()
        for owner in table_owners:
            table_box = owner.bbox
            horizontal_overlap = max(
                0.0,
                min(anchor_box[2], table_box[2]) - max(anchor_box[0], table_box[0]),
            )
            if (
                horizontal_overlap > 0
                and _heading_attaches_to_table(anchor_box, table_box)
                and table_box[3] <= next_top
            ):
                candidates.add(owner.table_id)
                candidate_anchors_by_table[owner.table_id].add(index)
        candidate_tables_by_anchor.append(candidates)
    return bool(
        all(len(candidates) == 1 for candidates in candidate_tables_by_anchor)
        and all(len(candidates) == 1 for candidates in candidate_anchors_by_table.values())
    )


def _heading_attaches_to_table(
    heading_box: tuple[float, float, float, float],
    table_box: tuple[float, float, float, float],
) -> bool:
    """Accept a source heading immediately above a ruled table border.

    OCR glyph boxes commonly overlap the top rule by a small fraction of the
    heading height.  Requiring a strict gap loses otherwise exact PBOC card
    anchors, while a fixed point tolerance overfits one renderer.  This
    relative contract permits only a shallow border overlap and still requires
    the heading centre to remain above the table.
    """

    heading_height = heading_box[3] - heading_box[1]
    if heading_height <= 0.0 or table_box[3] <= table_box[1]:
        return False
    return bool(
        heading_box[1] < table_box[1]
        and (heading_box[1] + heading_box[3]) / 2.0 < table_box[1]
        and heading_box[3] <= table_box[1] + heading_height * 0.25
    )


def _information_summary_table_witness(table: Any) -> bool:
    """Require a finite official summary metric family in every owned table."""

    rows = _raw_rows(table)
    cells = tuple(
        _compact(value)
        for row in rows
        for value in row
        if _compact(value)
    )
    compact = "".join(cells)
    return bool(
        cells
        and (
            any("信息汇总" in cell for cell in cells)
            or (
                "个人住房贷款" in compact
                and "个人商用房贷款" in compact
                and "其他类贷款" in compact
            )
            or ("查询机构数" in compact and "查询次数" in compact)
        )
    )


def _information_summary_table_explicit_continuation_witness(table: Any) -> bool:
    exact_internal_titles = {
        "逾期(透支)信息汇总",
        "非循环贷账户信息汇总",
        "循环贷账户一信息汇总",
        "循环贷账户二信息汇总",
        "贷记卡账户信息汇总",
        "准贷记卡账户信息汇总",
    }
    return any(
        _compact(value) in exact_internal_titles
        for row in _raw_rows(table)
        for value in row
    )


def _sealed_information_summary_continuation_proved(
    previous_page: Any,
    previous_evidence: Mapping[str, Any],
    previous_registration: Mapping[str, Any],
    current_page: Any,
    current_evidence: Mapping[str, Any],
    *,
    previous_authoritative_printed: Any = _PRINTED_IDENTITY_UNSPECIFIED,
    current_authoritative_printed: Any = _PRINTED_IDENTITY_UNSPECIFIED,
    current_table_owners: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Prove one headerless summary continuation from sealed local structure.

    Summary tables repeat account-family labels that are top-level headings on
    detail pages.  Those labels must not win when the immediately preceding
    printed page owns ``信息概要`` and the current page independently carries
    multiple exact summary-subsection headings attached to distinct tables.
    Printed-page order, heading ownership, and local geometry are all
    mandatory; page adjacency or table shape alone grants nothing.
    """

    if (
        previous_registration.get("status") != "registered"
        or previous_registration.get("template_id") != "information_summary"
        or current_table_owners
    ):
        return False

    def resolved_printed_identity(
        value: Any,
        evidence: Mapping[str, Any],
    ) -> tuple[int, int] | None:
        if value is _PRINTED_IDENTITY_UNSPECIFIED:
            return _printed_identity(evidence)
        if not isinstance(value, tuple) or len(value) != 2:
            return None
        page = _positive_int(value[0])
        total = _positive_int(value[1])
        return (page, total) if page is not None and total is not None and page <= total else None

    previous_printed = resolved_printed_identity(
        previous_authoritative_printed,
        previous_evidence,
    )
    current_printed = resolved_printed_identity(
        current_authoritative_printed,
        current_evidence,
    )
    if (
        previous_printed is None
        or current_printed is None
        or previous_printed[1] != current_printed[1]
        or current_printed[0] != previous_printed[0] + 1
    ):
        return False

    previous_summary_headings = [
        tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
        for line in previous_evidence.get("lines") or ()
        if _exact_evidence_line_bbox(line) is not None
        and canonical_registered_section_heading(_compact(line.get("text") or line.get("content") or "")) == "信息概要"
    ]
    if (
        len(previous_summary_headings) != 1
        or not previous_summary_headings[0]
        or len(previous_summary_headings[0]) != len(set(previous_summary_headings[0]))
    ):
        return False

    page_width = _finite(current_evidence.get("page_width") or getattr(current_page, "width", 0))
    page_height = _finite(current_evidence.get("page_height") or getattr(current_page, "height", 0))
    headings: list[
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[str, ...],
        ]
    ] = []
    account_family_lines: list[Mapping[str, Any]] = []
    conflicting_top_level_role = False
    for line in current_evidence.get("lines") or ():
        bbox = _exact_evidence_line_bbox(line)
        if bbox is None:
            continue
        if page_width > 0 and (bbox[0] < 0 or bbox[2] > page_width):
            continue
        if page_height > 0 and (bbox[1] < 0 or bbox[3] > page_height):
            continue
        text = _compact(line.get("text") or line.get("content") or "")
        title = canonical_registered_section_heading(text)
        if title is not None and REGISTERED_SECTION_TEMPLATE_BY_TITLE[title] != "information_summary":
            conflicting_top_level_role = True
        if _ACCOUNT_CARD_HEADING_RE.fullmatch(text) is not None or _MONTHLY_GRID_RE.search(text):
            return False
        if text in {
            "非循环贷账户",
            "循环贷账户一",
            "循环贷账户二",
            "贷记卡账户",
            "准贷记卡账户",
        }:
            account_family_lines.append(line)
        match = _INFORMATION_SUMMARY_SUBSECTION_RE.fullmatch(text)
        if match is None:
            continue
        evidence_ids = tuple(str(value) for value in line.get("evidence_ids") or () if str(value or ""))
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            return False
        headings.append((match.group("title"), bbox, evidence_ids))
    if conflicting_top_level_role or len({title for title, _box, _ids in headings}) < 2:
        return False
    heading_ids = [evidence_id for _title, _box, evidence_ids in headings for evidence_id in evidence_ids]
    if (
        len(heading_ids) != len(set(heading_ids))
        or set(previous_summary_headings[0]).intersection(heading_ids)
    ):
        return False

    tables = tuple(getattr(current_page, "tables", None) or ())
    table_boxes = [
        (str(getattr(table, "table_id", "") or ""), table_bbox)
        for table in tables
        if (table_bbox := _bbox(table)) is not None
    ]
    table_ids = [table_id for table_id, _box in table_boxes]
    if (
        not table_boxes
        or len(table_boxes) != len(tables)
        or any(not table_id for table_id in table_ids)
        or len(table_ids) != len(set(table_ids))
        or any(
            not any(_line_in_box(line, table_bbox) for _table_id, table_bbox in table_boxes)
            for line in account_family_lines
        )
    ):
        return False
    if any(
        not _information_summary_table_witness(table)
        for table in tables
    ):
        return False
    attached_table_ids: set[str] = set()
    for _title, heading_bbox, _evidence_ids in headings:
        candidates = [
            (table_id, table_bbox)
            for table_id, table_bbox in table_boxes
            if max(
                0.0,
                min(heading_bbox[2], table_bbox[2]) - max(heading_bbox[0], table_bbox[0]),
            )
            > 0.0
            and _heading_attaches_to_table(heading_bbox, table_bbox)
        ]
        if not candidates:
            return False
        nearest_top = min(table_bbox[1] for _table_id, table_bbox in candidates)
        nearest = [
            table_id
            for table_id, table_bbox in candidates
            if math.isclose(
                table_bbox[1],
                nearest_top,
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
        ]
        if len(nearest) != 1 or nearest[0] in attached_table_ids:
            return False
        attached_table_ids.add(nearest[0])
    leading_table_id = min(table_boxes, key=lambda item: (item[1][1], item[1][0]))[0]
    first_heading_top = min(heading_bbox[1] for _title, heading_bbox, _ids in headings)
    explicit_continuations = {
        str(getattr(table, "table_id", "") or "")
        for table in tables
        if _information_summary_table_explicit_continuation_witness(table)
    }
    if next(box for table_id, box in table_boxes if table_id == leading_table_id)[1] >= first_heading_top:
        leading_table_id = ""
    return bool(
        len(attached_table_ids) >= 2
        and set(table_ids)
        <= attached_table_ids | explicit_continuations | {leading_table_id}
    )


def _template_spec(template_id: str) -> CanonicalTemplateSpec | None:
    return next((spec for spec in _TEMPLATES if spec.template_id == template_id), None)


def _bottom_footer_geometry(value: Any, *, page_height: Any) -> bool:
    """Mirror the context's exact narrow bottom-furniture proof."""

    box = _bbox(value)
    height = _finite(page_height)
    if box is None or height <= 0.0:
        return False
    tolerance = max(2.0, height * 0.01)
    return bool(
        box[1] >= height * 0.85
        and box[3] >= height * 0.90
        and box[3] <= height + tolerance
        and box[3] - box[1] <= height * 0.08
    )


def _printed_identity_candidates(
    evidence: Mapping[str, Any],
) -> frozenset[tuple[int, int]]:
    page_height = evidence.get("page_height") or evidence.get("height")
    matches = {
        (int(match.group("page")), int(match.group("total")))
        for line in evidence.get("lines") or []
        if isinstance(line, Mapping)
        if _bottom_footer_geometry(line, page_height=page_height)
        for match in _PRINTED_PAGE_RE.finditer(str(line.get("text") or line.get("content") or ""))
    }
    return frozenset(
        (page, total) for page, total in matches if 1 <= page <= total
    )


def _printed_identity(evidence: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return one full printed-page footer proved by local geometry."""

    valid = _printed_identity_candidates(evidence)
    return next(iter(valid)) if len(valid) == 1 else None


def _overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    return intersection / area


def _line_in_box(line: Mapping[str, Any], box: Sequence[float]) -> bool:
    candidate = _bbox(line)
    if candidate is None:
        return False
    center = ((candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0)
    return (
        box[0] <= center[0] <= box[2]
        and box[1] <= center[1] <= box[3]
    ) or _overlap_ratio(candidate, box) >= 0.35


def _project_table(
    table: Any,
    *,
    template_id: str,
    transform: Callable[[Sequence[float]], list[float]],
) -> Any:
    metadata = deepcopy(dict(getattr(table, "metadata", None) or {}))
    rows = _raw_rows(table)
    cell_boxes = metadata.get("cell_bboxes")
    table_rows = list(getattr(table, "rows", None) or ())
    row_offset = 1 if getattr(table, "headers", None) and len(rows) == len(table_rows) + 1 else 0
    source_cell_objects: list[list[Any]] | None = None

    # Canonical pages intentionally project immutable source tables into one
    # registered coordinate plane.  Preserve the atomic cell provenance while
    # doing so: profile and other field extractors must be able to distinguish
    # an exact, evidence-sealed source cell from a merely canonical slot.  Older
    # reconstructed tables persisted only ``raw_rows``/``cell_bboxes`` in
    # metadata even though the CellValue objects still owned richer geometry.
    if table_rows and len(rows) == len(table_rows) + row_offset:
        source_boxes: list[list[Any]] = [[] for _ in rows]
        evidence_matrix: list[list[list[str]]] = [[] for _ in rows]
        token_matrix: list[list[list[str]]] = [[] for _ in rows]
        geometry_status_matrix: list[list[str]] = [[] for _ in rows]
        geometry_confidence_matrix: list[list[float | None]] = [[] for _ in rows]
        if row_offset:
            header_count = len(rows[0])
            source_boxes[0] = [None] * header_count
            evidence_matrix[0] = [[] for _ in range(header_count)]
            token_matrix[0] = [[] for _ in range(header_count)]
            geometry_status_matrix[0] = ["missing"] * header_count
            geometry_confidence_matrix[0] = [None] * header_count
        for source_row, table_row in enumerate(table_rows):
            target_row = source_row + row_offset
            for cell in getattr(table_row, "cells", None) or ():
                bbox = getattr(cell, "bbox", None)
                exact_bbox = list(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None
                source_boxes[target_row].append(exact_bbox)
                evidence_matrix[target_row].append(
                    [str(value) for value in getattr(cell, "evidence_ids", None) or () if value]
                )
                token_matrix[target_row].append(
                    [str(value) for value in getattr(cell, "token_ids", None) or () if value]
                )
                geometry_status_matrix[target_row].append(str(getattr(cell, "geometry_status", "") or "missing"))
                confidence = getattr(cell, "geometry_confidence", None)
                geometry_confidence_matrix[target_row].append(
                    float(confidence)
                    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                    else None
                )
        if all(len(source_boxes[index]) == len(rows[index]) for index in range(len(rows))):
            # Feed the native boxes into the existing single transform below.
            metadata.setdefault("cell_bboxes", source_boxes)
            metadata.setdefault("cell_evidence_ids", evidence_matrix)
            metadata.setdefault("cell_token_ids", token_matrix)
            metadata.setdefault("cell_geometry_status", geometry_status_matrix)
            metadata.setdefault("cell_geometry_confidences", geometry_confidence_matrix)
            # Detached canonical pages are in-memory only.  Retain immutable
            # CellValue owners on the projected table object rather than in its
            # serializable metadata so visual verifiers can inspect a slot
            # without changing the Semantic/Community contract.
            source_cell_objects = [
                [None] * len(rows[0])
                if row_offset and index == 0
                else list(getattr(table_rows[index - row_offset], "cells", None) or ())
                for index in range(len(rows))
            ]
            cell_boxes = metadata.get("cell_bboxes")
    if rows:
        metadata["raw_rows"] = rows
    if isinstance(cell_boxes, list):
        metadata["source_cell_bboxes"] = deepcopy(cell_boxes)
        metadata["cell_bboxes"] = [
            [transform(box) if isinstance(box, (list, tuple)) and len(box) == 4 else box for box in row]
            if isinstance(row, list)
            else row
            for row in cell_boxes
        ]
    metadata["canonical_template_id"] = template_id
    table_box = _bbox(table)
    return SimpleNamespace(
        table_id=str(getattr(table, "table_id", "") or ""),
        metadata=metadata,
        headers=[],
        rows=[],
        bbox=transform(table_box) if table_box is not None else None,
        confidence=getattr(table, "confidence", None),
        source_cell_objects=source_cell_objects,
    )


class PBOCCanonicalTemplateAssembler:
    """Register arbitrary ParseResult fragments against canonical PBOC pages."""

    def __init__(
        self,
        parse_result: Any,
        *,
        topology: Any,
        reading_order_by_logical: Mapping[int, int],
        source_evidence_loader: Callable[[], list[dict[str, Any]]],
        issue_owner: Any,
        source_page_loader: Callable[[], Iterable[Any]] | None = None,
    ) -> None:
        self.parse_result = parse_result
        self.topology = topology
        self.reading_order = {int(key): int(value) for key, value in reading_order_by_logical.items()}
        self.source_evidence_loader = source_evidence_loader
        self.issue_owner = issue_owner
        self.source_page_loader = source_page_loader

    def build(self) -> CanonicalLayoutProjection:
        source_evidence = self.source_evidence_loader()
        source_pages = (
            list(self.source_page_loader())
            if callable(self.source_page_loader)
            else list(getattr(self.parse_result, "pages", None) or [])
        )
        raw_pages = {
            int(getattr(page, "page_number", 0) or index): page
            for index, page in enumerate(source_pages, start=1)
        }
        evidence = {
            int(page.get("page") or 0): deepcopy(page)
            for page in source_evidence
            if isinstance(page, Mapping) and int(page.get("page") or 0) > 0
        }
        for logical, page in raw_pages.items():
            if logical in evidence:
                continue
            evidence[logical] = {
                "page": logical,
                "source_page": int(getattr(page, "source_page_number", 0) or logical),
                "page_width": _finite(getattr(page, "width", 0)),
                "page_height": _finite(getattr(page, "height", 0)),
                "lines": [
                    {
                        "text": str(getattr(block, "content", "") or ""),
                        "bbox": list(_bbox(block) or ()),
                        # Vector-native TextBlocks already point at immutable
                        # EvidenceStore atoms.  Preserve that ownership when no
                        # OCR/local-structure bundle exists; otherwise the
                        # strict runtime classifier mistakes sealed PDF text
                        # for an untrusted text-only fallback and drops every
                        # canonical page.
                        "evidence_ids": [
                            str(value)
                            for value in (getattr(block, "evidence_ids", None) or ())
                            if str(value or "")
                        ],
                        "source": "sealed_page_text_fallback",
                    }
                    for block in getattr(page, "texts", None) or []
                    if str(getattr(block, "content", "") or "").strip() and _bbox(block) is not None
                ],
            }
        registrations: dict[int, dict[str, Any]] = {}
        ordered_logicals = sorted(evidence, key=lambda value: self.reading_order.get(value, value))
        for logical_index, logical in enumerate(ordered_logicals):
            page_evidence = evidence[logical]
            page = raw_pages.get(logical, SimpleNamespace(tables=[], texts=[]))
            result = _classify_page(page, page_evidence)
            table_owners = _mixed_page_section_table_owners(page, page_evidence)
            inquiry_seed = _sealed_inquiry_seed_table_owner(page, page_evidence)
            if inquiry_seed is not None and inquiry_seed[0] not in table_owners:
                table_owners[inquiry_seed[0]] = inquiry_seed[1]
            if logical_index > 0:
                previous_logical = ordered_logicals[logical_index - 1]
                previous_page = raw_pages.get(previous_logical)
                if previous_page is not None:
                    cross_page_owner = _cross_page_agreement_table_owner(
                        previous_page,
                        evidence[previous_logical],
                        page,
                        page_evidence,
                    )
                    if cross_page_owner is not None:
                        table_id, owner = cross_page_owner
                        if table_id not in table_owners:
                            table_owners[table_id] = owner
                    previous_registration = registrations.get(previous_logical)
                    if isinstance(previous_registration, Mapping):
                        if _sealed_information_summary_continuation_proved(
                            previous_page,
                            evidence[previous_logical],
                            previous_registration,
                            page,
                            page_evidence,
                            previous_authoritative_printed=(
                                (
                                    int(previous_registration["printed_page"]),
                                    int(previous_registration["printed_total"]),
                                )
                                if previous_registration.get("printed_identity_authoritative") is True
                                and _positive_int(previous_registration.get("printed_page")) is not None
                                and _positive_int(previous_registration.get("printed_total")) is not None
                                else None
                            ),
                            current_authoritative_printed=self._authoritative_printed_identity(
                                logical,
                                page_evidence,
                            )[0],
                            current_table_owners=table_owners,
                        ):
                            result = (
                                "information_summary",
                                0.99,
                                (
                                    "exact_prior_information_summary_owner",
                                    "consecutive_sealed_printed_footers",
                                    "multiple_exact_summary_subsection_headings",
                                    "distinct_heading_to_table_geometry",
                                    "no_conflicting_top_level_section",
                                ),
                            )
                        if _sealed_liability_page_continuation_proved(
                            previous_page,
                            evidence[previous_logical],
                            previous_registration,
                            page,
                            page_evidence,
                        ):
                            result = (
                                "repayment_responsibility",
                                0.99,
                                (
                                    "exact_cross_page_liability_continuation",
                                    "consecutive_sealed_printed_footers",
                                    "terminal_prior_liability_card_owner",
                                    "dense_exact_liability_card_ordinals",
                                    "complete_exact_liability_header_graphs",
                                    "anchor_table_geometry_bijection",
                                    "unique_nonreplayed_deciding_evidence",
                                ),
                            )
                        agreement_owners = _same_page_agreement_continuation_owners(
                            previous_page,
                            evidence[previous_logical],
                            previous_registration,
                            page,
                            page_evidence,
                        )
                        for table_id, owner in agreement_owners.items():
                            if table_id not in table_owners:
                                table_owners[table_id] = owner
                        inquiry_owner = _headerless_inquiry_continuation_owner(
                            previous_page,
                            evidence[previous_logical],
                            previous_registration,
                            page,
                            page_evidence,
                            table_owners,
                            tables_continue=getattr(
                                self.issue_owner,
                                "tables_continue",
                                None,
                            ),
                            reading_order_resolution=getattr(
                                self.issue_owner,
                                "reading_order_resolution",
                                None,
                            ),
                            topology=self.topology,
                            frozen_topology_audit_loader=(
                                getattr(
                                    self.issue_owner,
                                    "page_topology_audit",
                                    None,
                                )
                                if getattr(
                                    self.issue_owner,
                                    "page_topology",
                                    None,
                                )
                                is self.topology
                                else None
                            ),
                            entity_context=getattr(
                                self.issue_owner,
                                "entity_context",
                                None,
                            ),
                        )
                        if (
                            inquiry_owner is not None
                            and inquiry_owner[0] not in table_owners
                        ):
                            table_owners[inquiry_owner[0]] = inquiry_owner[1]
            roles = tuple(
                sorted({owner["template_id"] for owner in table_owners.values()})
            )
            has_exact_cross_page_continuation = any(
                owner.get("binding")
                == "authoritative_prior_inquiry_table_continuation"
                for owner in table_owners.values()
            )
            source_table_ids = {
                str(getattr(table, "table_id", "") or "")
                for table in getattr(page, "tables", None) or ()
                if str(getattr(table, "table_id", "") or "")
            }
            whole_page_role = result[0] if result is not None else ""
            whole_page_ownership_is_complete = (
                bool(table_owners)
                and whole_page_role != "annotations_and_inquiries"
                and set(roles) == {whole_page_role}
                and set(table_owners) == source_table_ids
            )
            if result is not None and whole_page_ownership_is_complete:
                template_id, confidence, signals = result
                registrations[logical] = self._registration(
                    logical,
                    template_id,
                    confidence,
                    "source_page_evidence",
                    signals,
                    page_evidence,
                )
                continue
            if table_owners:
                registration = self._registration(
                    logical,
                    "mixed_pboc_sections",
                    0.99,
                    "exact_table_local_pboc_section_ownership",
                    (
                        "independently_proven_table_local_pboc_roles",
                        "sealed_section_or_subsection_headings",
                        "exact_table_header_and_population_lattices",
                        "ambiguous_and_unowned_tables_withheld",
                        *(
                            ("exact_cross_page_table_continuation",)
                            if has_exact_cross_page_continuation
                            else ()
                        ),
                        *roles,
                    ),
                    page_evidence,
                )
                registration["section_table_owners"] = deepcopy(table_owners)
                registration["affected_source_datasets"] = sorted(
                    {
                        dataset
                        for role in roles
                        for spec in (_template_spec(role),)
                        if spec is not None
                        for dataset in spec.datasets
                    }
                )
                registrations[logical] = registration
                continue
            if result is not None:
                template_id, confidence, signals = result
                # Inquiry business tables are always table-local.  A sealed
                # section heading cannot by itself turn a malformed or loose
                # header, unrelated table, or OCR residue into query-record
                # ownership.  Clean inquiry tables reached the projection
                # envelope above; all remaining inquiry-shaped pages stay
                # unresolved for explicit reporting.
                if template_id != "annotations_and_inquiries":
                    registrations[logical] = self._registration(
                        logical,
                        template_id,
                        confidence,
                        "source_page_evidence",
                        signals,
                        page_evidence,
                    )
                    continue
            if len(_compact(_page_text(page, page_evidence))) < 8:
                registrations[logical] = self._registration(
                    logical,
                    "blank_fragment",
                    1.0,
                    "explicitly_blank_fragment",
                    ("no_business_content",),
                    page_evidence,
                    status="blank",
                )

        # Canonical pages often start in the middle of an account or repeated
        # table. A remaining fragment may inherit only when an explicit
        # canonical continuation relation is present; document proximity or
        # page shape alone is not sufficient.
        ordered = ordered_logicals
        active_template = ""
        active_logical = 0
        for logical in ordered:
            registration = registrations.get(logical)
            if registration and registration["status"] == "registered":
                active_template = str(registration["template_id"])
                active_logical = logical
                continue
            if registration:
                continue
            if active_template == "mixed_pboc_sections":
                # ``mixed_pboc_sections`` is a projection envelope, not a
                # semantic section role.  It may never flow to another page
                # without fresh table-local owners; otherwise an explicit edge
                # can create an apparently registered page with no owned data.
                active_template = ""
                active_logical = 0
                continue
            text = _compact(_page_text(raw_pages.get(logical, SimpleNamespace(tables=[], texts=[])), evidence[logical]))
            continuation_signals: list[str] = []
            continuation_check = getattr(self.issue_owner, "tables_continue", None)
            previous_page = raw_pages.get(active_logical)
            current_page = raw_pages.get(logical)
            current_heading_roles = (
                _sealed_registered_heading_roles(current_page, evidence[logical])
                if current_page is not None
                else frozenset()
            )
            # Reading order and an explicit table edge cannot override a
            # sealed semantic heading on the current page.  Same-role
            # subsection headings remain compatible; any other registered
            # role leaves the page unresolved for table-local classification.
            if current_heading_roles.difference({active_template}):
                active_template = ""
                active_logical = 0
                continue
            if (
                active_template
                and active_template != "repayment_responsibility"
                and callable(continuation_check)
                and previous_page is not None
                and current_page is not None
            ):
                previous_ids = [
                    str(getattr(table, "table_id", "") or "")
                    for table in getattr(previous_page, "tables", None) or ()
                ]
                current_ids = [
                    str(getattr(table, "table_id", "") or "")
                    for table in getattr(current_page, "tables", None) or ()
                ]
                if any(
                    left and right and continuation_check(left, right) is True
                    for left in previous_ids
                    for right in current_ids
                ):
                    continuation_signals.append("explicit_table_continuation")
            if active_template == "credit_account_detail" and _MONTHLY_GRID_RE.search(text):
                continuation_signals.append("canonical_monthly_grid_continuation")
            if active_template == "credit_account_detail" and current_page is not None:
                active_registration = registrations.get(active_logical, {})
                current_printed, _current_printed_basis = self._authoritative_printed_identity(
                    logical,
                    evidence[logical],
                )
                active_printed = (
                    int(active_registration.get("printed_page") or 0),
                    int(active_registration.get("printed_total") or 0),
                )
                if (
                    active_registration.get("printed_identity_authoritative") is True
                    and current_printed is not None
                    and active_printed[0] > 0
                    and active_printed[1] == current_printed[1]
                    and current_printed[0] > active_printed[0]
                    and _sealed_account_card_continuation_proved(
                        current_page,
                        evidence[logical],
                    )
                ):
                    continuation_signals.extend(
                        (
                            "authoritative_printed_account_order",
                            "sealed_account_card_anchor",
                            "dense_exact_account_table_schema",
                            "anchor_to_table_geometry",
                        )
                    )
            if active_template == "annotations_and_inquiries" and re.search(
                r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}.*(?:贷后管理|贷款审批|信用卡审批|本人查询)",
                text,
            ):
                continuation_signals.append("canonical_inquiry_row_continuation")
            if active_template == "postpaid_detail" and (
                "缴费记录" in text
                or bool(re.search(r"20\d{2}[N1-7*]{4,}", text))
            ):
                continuation_signals.append("canonical_postpaid_history_continuation")
            if active_template == "report_explanation" and re.search(
                r"(?:N-正常|C-结清|G-结束|#-账户|1-逾期)",
                text,
            ):
                continuation_signals.append("canonical_explanation_continuation")
            # A table merely existing, or a page merely containing enough
            # text, is not proof that it continues the preceding section.
            # Under the closed canonical catalog an unproven page stays
            # unresolved and becomes eligible for the one business-repair
            # OCR pass instead of being silently registered to the wrong role.
            if active_template and continuation_signals:
                registrations[logical] = self._registration(
                    logical,
                    active_template,
                    0.78,
                    "canonical_flow_continuation",
                    (
                        "preceding_template_role",
                        *continuation_signals,
                        "no_conflicting_section_anchor",
                    ),
                    evidence[logical],
                )
                active_logical = logical

        unresolved = tuple(
            logical
            for logical in ordered
            if logical not in registrations or registrations[logical].get("status") == "unresolved"
        )
        for logical in unresolved:
            registrations[logical] = self._registration(
                logical,
                "unresolved",
                0.0,
                "canonical_registration_exhausted",
                ("no_generic_layout_fallback",),
                evidence[logical],
                status="unresolved",
            )
            self._record_registration_failure(logical, registrations[logical])

        groups = self._fragment_groups(evidence, registrations)
        withheld_group_pages = {
            int(logical)
            for group in groups
            if group.get("status") == "unresolved"
            and group.get("reason") in {
                "conflicting_registered_template_ids",
                "ambiguous_fragment_geometry",
            }
            for logical in group.get("logical_pages") or ()
        }
        if withheld_group_pages:
            unresolved_pages = set(unresolved) | withheld_group_pages
            unresolved = tuple(
                logical
                for logical in ordered
                if logical in unresolved_pages
            )
            for group in groups:
                if group.get("reason") == "conflicting_registered_template_ids":
                    self._record_fragment_group_conflict(group, registrations)
                elif group.get("reason") == "ambiguous_fragment_geometry":
                    for logical in group.get("logical_pages") or ():
                        registration = registrations[int(logical)]
                        registration["status"] = "unresolved"
                        registration["basis"] = "canonical_fragment_geometry_unresolved"
                        registration["signals"] = [
                            *registration.get("signals", ()),
                            "ambiguous_fragment_geometry",
                        ]
                    self._record_fragment_geometry_failure(group, registrations)
        canonical_pages: list[Any] = []
        canonical_evidence: list[dict[str, Any]] = []
        group_audits: list[dict[str, Any]] = []
        for group in groups:
            if group["status"] == "blank":
                continue
            if group["status"] == "unresolved":
                # No generic extraction is allowed from an unregistered page.
                continue
            page, page_evidence, audit = self._assemble_group(group, raw_pages, evidence, registrations)
            canonical_pages.append(page)
            canonical_evidence.append(page_evidence)
            group_audits.append(audit)

        return CanonicalLayoutProjection(
            pages=tuple(canonical_pages),
            evidence_pages=tuple(canonical_evidence),
            registrations=tuple(registrations[key] for key in sorted(registrations, key=lambda v: self.reading_order.get(v, v))),
            fragment_groups=tuple(group_audits),
            unresolved_pages=unresolved,
        )

    def _registration(
        self,
        logical: int,
        template_id: str,
        confidence: float,
        basis: str,
        signals: Iterable[str],
        evidence: Mapping[str, Any],
        *,
        status: str = "registered",
    ) -> dict[str, Any]:
        printed, printed_basis = self._authoritative_printed_identity(logical, evidence)
        spec = _template_spec(template_id)
        return {
            "logical_page": logical,
            "source_page": int(evidence.get("source_page") or logical),
            "template_id": template_id,
            "status": status,
            "confidence": round(float(confidence), 4),
            "basis": basis,
            "signals": list(signals),
            **(
                {
                    "printed_page": printed[0],
                    "printed_total": printed[1],
                    "printed_identity_authoritative": True,
                    "printed_identity_basis": printed_basis,
                }
                if printed
                else {}
            ),
            **({"affected_source_datasets": list(spec.datasets)} if spec else {}),
        }

    def _authoritative_printed_identity(
        self,
        logical: int,
        evidence: Mapping[str, Any],
    ) -> tuple[tuple[int, int] | None, str]:
        """Use context provenance when present, otherwise prove footer geometry."""

        resolution = getattr(self.issue_owner, "reading_order_resolution", None)
        if isinstance(resolution, Mapping):
            if (
                resolution.get("resolved") is True
                and resolution.get("authoritative") is True
                and resolution.get("identity_fallback") is not True
            ):
                printed_by_logical = resolution.get("printed_page_by_logical")
                printed_total = resolution.get("printed_total")
                if isinstance(printed_by_logical, Mapping):
                    printed_raw = printed_by_logical.get(logical)
                    if printed_raw is None:
                        printed_raw = printed_by_logical.get(str(logical))
                    if (
                        isinstance(printed_raw, int)
                        and not isinstance(printed_raw, bool)
                        and isinstance(printed_total, int)
                        and not isinstance(printed_total, bool)
                        and 1 <= printed_raw <= printed_total
                    ):
                        return (
                            (printed_raw, printed_total),
                            "context_authoritative_printed_order",
                        )
            # A present context resolution is the authority boundary.  Its
            # explicit rejection or omission cannot be bypassed by rescanning
            # the same local evidence inside canonical assembly.
            return None, ""

        printed = _printed_identity(evidence)
        if printed is not None:
            return printed, "bottom_footer_geometry"
        return None, ""

    def _fragment_groups(
        self,
        evidence: Mapping[int, Mapping[str, Any]],
        registrations: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        printed_by_source: dict[int, list[tuple[int, int, int, str]]] = {}
        for logical, registration in registrations.items():
            printed_page = int(registration.get("printed_page") or 0)
            printed_total = int(registration.get("printed_total") or 0)
            source_page = int(registration.get("source_page") or 0)
            if (
                printed_page
                and printed_total
                and source_page
                and registration.get("printed_identity_authoritative") is True
            ):
                printed_by_source.setdefault(source_page, []).append(
                    (
                        printed_page,
                        printed_total,
                        logical,
                        str(registration.get("template_id") or ""),
                    )
                )
        grouped: dict[tuple[str, int, int], list[int]] = {}
        printed_candidates = [
            (page, total, source_page, seed_logical, seed_template)
            for source_page, candidates in printed_by_source.items()
            for page, total, seed_logical, seed_template in candidates
        ]
        for logical in sorted(evidence, key=lambda value: self.reading_order.get(value, value)):
            registration = registrations[logical]
            printed_page = (
                int(registration.get("printed_page") or 0)
                if registration.get("printed_identity_authoritative") is True
                else 0
            )
            printed_total = (
                int(registration.get("printed_total") or 0)
                if registration.get("printed_identity_authoritative") is True
                else 0
            )
            if not printed_page and evidence[logical].get("plugin_static_subpage"):
                source_page = int(registration.get("source_page") or evidence[logical].get("source_page") or 0)
                template_id = str(registration.get("template_id") or "")
                compatible = {
                    (page, total)
                    for page, total, _seed_logical, seed_template in printed_by_source.get(source_page, ())
                    if seed_template == template_id
                }
                if not compatible:
                    # A plugin-declared static subpage is not an independent
                    # canonical page.  When exactly one authoritative seed in
                    # the document has the same semantic role, tentatively
                    # associate it only so the fragment-geometry validator can
                    # prove or reject the composition.  Cross-source surfaces,
                    # overlapping crops, and incompatible transforms then
                    # quarantine the whole attempted join.  Multiple seeds
                    # remain ambiguous and are never selected by proximity or
                    # encounter order.
                    global_compatible = {
                        (page, total)
                        for page, total, _source, _seed_logical, seed_template in printed_candidates
                        if seed_template == template_id
                    }
                    if len(global_compatible) == 1:
                        compatible = global_compatible
                if len(compatible) == 1:
                    # This is another crop of the same canonical printed page,
                    # not a new template variant.  Its source crop determines
                    # where it lands on the virtual page canvas.
                    printed_page, printed_total = next(iter(compatible))
            key = (
                ("printed", printed_page, printed_total)
                if printed_page and printed_total
                else ("logical", logical, 0)
            )
            grouped.setdefault(key, []).append(logical)
        result: list[dict[str, Any]] = []
        for key, logicals in grouped.items():
            statuses = {str(registrations[logical].get("status") or "unresolved") for logical in logicals}
            status = "unresolved" if "unresolved" in statuses else "blank" if statuses == {"blank"} else "registered"
            template_ids = [
                str(registrations[logical].get("template_id") or "")
                for logical in logicals
                if registrations[logical].get("status") == "registered"
            ]
            unique_template_ids = sorted(set(template_ids))
            template_conflict = len(unique_template_ids) > 1
            if template_conflict:
                status = "unresolved"
            template_id = unique_template_ids[0] if len(unique_template_ids) == 1 else "unresolved"
            geometry_failure = (
                self._fragment_geometry_failure(logicals, evidence)
                if len(logicals) > 1 and not template_conflict
                else None
            )
            if geometry_failure is not None:
                status = "unresolved"
            result.append(
                {
                    "group_key": (
                        f"{key[0]}:{key[1]}/{key[2]}"
                        if key[0] == "printed"
                        else f"{key[0]}:{key[1]}"
                    ),
                    "canonical_page": key[1] if key[0] == "printed" else self.reading_order.get(logicals[0], logicals[0]),
                    "logical_pages": logicals,
                    "template_id": template_id,
                    "status": status,
                    **(
                        {
                            "reason": "conflicting_registered_template_ids",
                            "conflicting_template_ids": unique_template_ids,
                        }
                        if template_conflict
                        else {}
                    ),
                    **(
                        {
                            "reason": "ambiguous_fragment_geometry",
                            "geometry_failure": geometry_failure,
                        }
                        if geometry_failure is not None
                        else {}
                    ),
                }
            )
        return result

    def _fragment_geometry_failure(
        self,
        logicals: Sequence[int],
        evidence: Mapping[int, Mapping[str, Any]],
    ) -> str | None:
        """Validate that fragment placements share one explicit source plane."""

        source_pages: set[int] = set()
        crops: list[tuple[float, float, float, float]] = []
        rotations: set[int] = set()
        transform_signatures: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for logical in logicals:
            local = evidence[logical]
            geometry = self.topology.geometry(logical) if self.topology is not None else None
            evidence_crop = local.get("source_crop_bbox")
            detached_crop = (
                tuple(_finite(value) for value in evidence_crop)
                if isinstance(evidence_crop, (list, tuple)) and len(evidence_crop) == 4
                else None
            )
            topology_crop = getattr(geometry, "source_crop_bbox", None)
            if geometry is not None:
                topology_source_page = int(getattr(geometry, "source_page", 0) or 0)
                detached_source_page = int(local.get("source_page") or 0)
                if topology_source_page <= 0:
                    return "source_page_missing"
                if detached_source_page and detached_source_page != topology_source_page:
                    return "source_page_provenance_mismatch"
                if topology_crop is None:
                    return "source_crop_bbox_missing_or_invalid"
                if detached_crop is not None and not all(
                    math.isclose(
                        float(detached_value),
                        float(topology_value),
                        rel_tol=1e-4,
                        abs_tol=0.05,
                    )
                    for detached_value, topology_value in zip(
                        detached_crop,
                        topology_crop,
                        strict=True,
                    )
                ):
                    return "source_crop_provenance_mismatch"
                source_page = topology_source_page
                crop = tuple(float(value) for value in topology_crop)
            else:
                source_page = int(local.get("source_page") or 0)
                crop = detached_crop
            if crop is None or crop[2] <= crop[0] or crop[3] <= crop[1]:
                return "source_crop_bbox_missing_or_invalid"
            if source_page <= 0:
                return "source_page_missing"
            source_pages.add(source_page)

            transform_proved = bool(
                geometry is not None
                and getattr(geometry, "transform_usable", False) is True
            )
            if geometry is not None:
                rotations.add(int(getattr(geometry, "selected_rotation", 0)) % 360)
                rotation = int(getattr(geometry, "selected_rotation", 0)) % 360
                transform_signatures.add(
                    {
                        0: ((0, 1), (1, 1)),
                        90: ((1, 1), (0, -1)),
                        180: ((0, -1), (1, -1)),
                        270: ((1, -1), (0, 1)),
                    }.get(rotation, ((0, 1), (1, 1)))
                )
            else:
                transform = local.get("coordinate_transform")
                transform = transform if isinstance(transform, Mapping) else {}
                forward = local.get("source_to_logical") or transform.get("matrix")
                inverse = local.get("logical_to_source") or transform.get("inverse_matrix")
                transform_proved = _affine_pair_valid(forward, inverse)
                decomposition = transform.get("decomposition")
                decomposition = decomposition if isinstance(decomposition, Mapping) else {}
                rotations.add(
                    int(
                        local.get("selected_rotation")
                        or decomposition.get("selected_rotation")
                        or 0
                    )
                    % 360
                )
                signature = _affine_axis_signature(forward)
                if signature is None:
                    return "source_transform_missing_or_invalid"
                transform_signatures.add(signature)
            if not transform_proved:
                return "source_transform_missing_or_invalid"
            crops.append(tuple(float(value) for value in crop))

        if len(source_pages) != 1:
            return "fragments_do_not_share_source_surface"
        if len(rotations) != 1:
            return "fragment_orientation_mismatch"
        if len(transform_signatures) != 1:
            return "fragment_transform_orientation_mismatch"
        for index, left in enumerate(crops):
            for right in crops[index + 1 :]:
                intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
                    0.0,
                    min(left[3], right[3]) - max(left[1], right[1]),
                )
                smaller_area = min(
                    (left[2] - left[0]) * (left[3] - left[1]),
                    (right[2] - right[0]) * (right[3] - right[1]),
                )
                if intersection > max(1e-6, smaller_area * 1e-6):
                    return "source_crop_bboxes_overlap"
        return None

    def _assemble_group(
        self,
        group: Mapping[str, Any],
        raw_pages: Mapping[int, Any],
        evidence: Mapping[int, Mapping[str, Any]],
        registrations: Mapping[int, Mapping[str, Any]],
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        logicals = [int(value) for value in group["logical_pages"]]
        representative = logicals[0]
        template_id = str(group["template_id"])
        transforms, width, height, coverage = self._fragment_transforms(logicals, evidence)
        output_lines: list[dict[str, Any]] = []
        tables: list[Any] = []
        table_boxes: list[tuple[float, float, float, float]] = []

        for logical in logicals:
            local_evidence = evidence[logical]
            transform = transforms[logical]
            local_lines = [line for line in local_evidence.get("lines") or [] if isinstance(line, Mapping)]
            raw_page = raw_pages.get(logical)
            if raw_page is not None:
                local_registration = registrations.get(logical, {})
                section_table_owners = (
                    local_registration.get("section_table_owners")
                    if isinstance(local_registration, Mapping)
                    else None
                )
                section_table_owners = (
                    section_table_owners
                    if isinstance(section_table_owners, Mapping)
                    else {}
                )
                for table in getattr(raw_page, "tables", None) or []:
                    table_id = str(getattr(table, "table_id", "") or "")
                    section_owner = section_table_owners.get(table_id)
                    if template_id == "mixed_pboc_sections" and not isinstance(
                        section_owner,
                        Mapping,
                    ):
                        continue
                    table_template_id = (
                        str(section_owner.get("template_id") or "")
                        if isinstance(section_owner, Mapping)
                        else template_id
                    )
                    if not table_template_id:
                        continue
                    projected = _project_table(
                        table,
                        template_id=table_template_id,
                        transform=transform,
                    )
                    projected.metadata["source_logical_page"] = logical
                    projected.metadata["source_page"] = int(local_evidence.get("source_page") or logical)
                    if isinstance(section_owner, Mapping):
                        projected.metadata["canonical_page_template_id"] = template_id
                        projected.metadata["canonical_section_owner"] = deepcopy(
                            dict(section_owner)
                        )
                    tables.append(projected)
                    if (box := _bbox(projected)) is not None:
                        table_boxes.append(box)
            for line in local_lines:
                box = _bbox(line)
                if box is None:
                    continue
                output_lines.append(
                    {
                        **deepcopy(dict(line)),
                        "source_bbox": list(box),
                        "bbox": transform(box),
                        "page": representative,
                        "source_logical_page": logical,
                        "canonical_page": int(group["canonical_page"]),
                        "canonical_template_id": template_id,
                    }
                )

        output_lines.sort(key=lambda line: ((_bbox(line) or (0, 0, 0, 0))[1], (_bbox(line) or (0, 0, 0, 0))[0]))
        texts = [
            SimpleNamespace(
                content=str(line.get("text") or ""),
                bbox=list(line.get("bbox") or []),
                evidence_ids=[
                    str(value)
                    for value in line.get("evidence_ids") or ()
                    if str(value or "")
                ],
                source_bbox=list(line.get("source_bbox") or line.get("bbox") or []),
                source_logical_page=int(line.get("source_logical_page") or representative),
            )
            for line in output_lines
            if not any(_line_in_box(line, box) for box in table_boxes)
        ]
        source_pages = sorted({int(evidence[logical].get("source_page") or logical) for logical in logicals})
        page = SimpleNamespace(
            page_number=representative,
            source_page_number=source_pages[0] if source_pages else representative,
            width=width,
            height=height,
            tables=tables,
            texts=texts,
            coordinate_transform={
                "kind": "plugin_canonical_template",
                "canonical_page": int(group["canonical_page"]),
                "fragment_logical_pages": logicals,
                "source_page_numbers": source_pages,
            },
            canonical_template_id=template_id,
            canonical_fragment_logical_pages=tuple(logicals),
        )
        page_evidence = {
            "page": representative,
            "canonical_page": int(group["canonical_page"]),
            "source_page": source_pages[0] if source_pages else representative,
            "source_pages": source_pages,
            "fragment_logical_pages": logicals,
            "page_width": width,
            "page_height": height,
            "canonical_template_id": template_id,
            "canonical_coverage_status": "full" if coverage >= 0.985 else "partial",
            "canonical_coverage_ratio": round(coverage, 4),
            "lines": output_lines,
        }
        audit = {
            "canonical_page": int(group["canonical_page"]),
            "template_id": template_id,
            "fragment_logical_pages": logicals,
            "source_pages": source_pages,
            "coverage_ratio": round(coverage, 4),
            "coverage_status": "full" if coverage >= 0.985 else "partial",
            "joined_fragment_count": len(logicals),
            **(
                {
                    "section_table_owners": {
                        table_id: deepcopy(dict(owner))
                        for logical in logicals
                        for table_id, owner in (
                            registrations.get(logical, {}).get(
                                "section_table_owners",
                                {},
                            )
                            or {}
                        ).items()
                        if isinstance(owner, Mapping)
                    }
                }
                if template_id == "mixed_pboc_sections"
                else {}
            ),
        }
        return page, page_evidence, audit

    def _fragment_transforms(
        self,
        logicals: Sequence[int],
        evidence: Mapping[int, Mapping[str, Any]],
    ) -> tuple[dict[int, Callable[[Sequence[float]], list[float]]], float, float, float]:
        placements: dict[int, tuple[float, float, float, float, float, float]] = {}
        source_boxes: list[tuple[float, float, float, float]] = []
        for logical in logicals:
            geometry = self.topology.geometry(logical) if self.topology is not None else None
            evidence_crop = evidence[logical].get("source_crop_bbox")
            crop = (
                tuple(float(value) for value in evidence_crop)
                if isinstance(evidence_crop, (list, tuple)) and len(evidence_crop) == 4
                else getattr(geometry, "source_crop_bbox", None)
            )
            local_width = max(1.0, _finite(evidence[logical].get("page_width")))
            local_height = max(1.0, _finite(evidence[logical].get("page_height")))
            if crop is not None:
                source_box = tuple(float(value) for value in crop)
            elif len(logicals) == 1:
                source_box = (0.0, 0.0, local_width, local_height)
            else:
                # Multi-fragment groups are admitted only by the explicit
                # geometry contract above.  Never manufacture a vertical
                # placement for fragments whose source crop is unknown.
                raise ValueError("multi-fragment canonical placement requires source crops")
            source_boxes.append(source_box)
            placements[logical] = (*source_box, local_width, local_height)

        union = (
            min(box[0] for box in source_boxes),
            min(box[1] for box in source_boxes),
            max(box[2] for box in source_boxes),
            max(box[3] for box in source_boxes),
        )
        transforms: dict[int, Callable[[Sequence[float]], list[float]]] = {}
        for logical, placement in placements.items():
            sx0, sy0, sx1, sy1, local_width, local_height = placement
            scale_x = (sx1 - sx0) / local_width
            scale_y = (sy1 - sy0) / local_height

            def transform(
                box: Sequence[float],
                *,
                x0: float = sx0,
                y0: float = sy0,
                fx: float = scale_x,
                fy: float = scale_y,
                ux0: float = union[0],
                uy0: float = union[1],
            ) -> list[float]:
                return [
                    x0 - ux0 + float(box[0]) * fx,
                    y0 - uy0 + float(box[1]) * fy,
                    x0 - ux0 + float(box[2]) * fx,
                    y0 - uy0 + float(box[3]) * fy,
                ]

            transforms[logical] = transform

        union_area = max(1.0, (union[2] - union[0]) * (union[3] - union[1]))
        covered_area = sum(max(0.0, (box[2] - box[0]) * (box[3] - box[1])) for box in source_boxes)
        # Overlaps can make the simple sum exceed the union.  They represent
        # duplicated evidence rather than additional coverage.
        coverage = min(1.0, covered_area / union_area)
        return transforms, union[2] - union[0], union[3] - union[1], coverage

    def _record_registration_failure(self, logical: int, registration: Mapping[str, Any]) -> None:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            make_issue,
            record_issue,
        )

        record_issue(
            self.issue_owner,
            make_issue(
                category="ocr_structure_correction",
                issue_code="canonical_page_registration_failed",
                message=(
                    "The logical page could not be registered to a canonical PBOC layout from static source "
                    "evidence; generic reconstruction was not used and business repair may retry the page."
                ),
                parser_stage="canonical_template_registration",
                observed_value={
                    "logical_page": logical,
                    "source_page": registration.get("source_page"),
                },
                confidence=0.0,
                source_refs=(
                    {
                        "source": "canonical_template_registration",
                        "logical_page": logical,
                        "source_page": int(registration.get("source_page") or logical),
                        "geometry_scope": "logical_page",
                    },
                ),
                reason_codes=(
                    "canonical_layout_unresolved_from_source_evidence",
                    "schema_triggered_page_repair_eligible",
                    "no_generic_layout_fallback",
                    "normalized_values_withheld_for_page",
                ),
            ),
        )

    def _record_fragment_group_conflict(
        self,
        group: Mapping[str, Any],
        registrations: Mapping[int, Mapping[str, Any]],
    ) -> None:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            make_issue,
            record_issue,
        )

        logicals = tuple(int(value) for value in group.get("logical_pages") or ())
        source_refs = tuple(
            {
                "source": "canonical_template_registration",
                "logical_page": logical,
                "source_page": int(registrations[logical].get("source_page") or logical),
                "geometry_scope": "logical_page",
            }
            for logical in logicals
        )
        record_issue(
            self.issue_owner,
            make_issue(
                category="ocr_structure_correction",
                issue_code="canonical_fragment_template_conflict",
                message=(
                    "Fragments with one authoritative printed-page identity registered to conflicting "
                    "canonical templates; the complete group was withheld."
                ),
                parser_stage="canonical_template_registration",
                observed_value={
                    "group_key": group.get("group_key"),
                    "logical_pages": list(logicals),
                    "template_ids": list(group.get("conflicting_template_ids") or ()),
                },
                confidence=0.0,
                source_refs=source_refs,
                reason_codes=(
                    "authoritative_printed_identity_template_conflict",
                    "canonical_group_withheld",
                    "no_first_template_fallback",
                    "normalized_values_withheld_for_page",
                ),
            ),
        )

    def _record_fragment_geometry_failure(
        self,
        group: Mapping[str, Any],
        registrations: Mapping[int, Mapping[str, Any]],
    ) -> None:
        from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
            make_issue,
            record_issue,
        )

        logicals = tuple(int(value) for value in group.get("logical_pages") or ())
        record_issue(
            self.issue_owner,
            make_issue(
                category="ocr_structure_correction",
                issue_code="canonical_fragment_geometry_unresolved",
                message=(
                    "Fragments assigned to one canonical PBOC page lacked one explicit, "
                    "non-overlapping source-plane geometry; the complete group was withheld."
                ),
                parser_stage="canonical_template_registration",
                observed_value={
                    "group_key": group.get("group_key"),
                    "logical_pages": list(logicals),
                    "geometry_failure": group.get("geometry_failure"),
                },
                confidence=0.0,
                source_refs=tuple(
                    {
                        "source": "canonical_template_registration",
                        "logical_page": logical,
                        "source_page": int(registrations[logical].get("source_page") or logical),
                        "geometry_scope": "logical_page",
                    }
                    for logical in logicals
                ),
                reason_codes=(
                    "explicit_source_crop_and_transform_required",
                    "non_overlapping_source_plane_required",
                    "canonical_group_withheld",
                    "no_synthetic_vertical_fragment_stack",
                    "normalized_values_withheld_for_page",
                ),
            ),
        )


__all__ = [
    "CanonicalLayoutProjection",
    "CanonicalTemplateSpec",
    "PBOCCanonicalTemplateAssembler",
]
