from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    extract_candidate_b_profile,
)

_PROFILE_TEMPLATE = "report_header_and_identity"


def _plain_profile_context(headers: list[str], values: list[str]) -> SimpleNamespace:
    table = SimpleNamespace(
        table_id="profile-scalar-whole-cell",
        metadata={
            "canonical_template_id": _PROFILE_TEMPLATE,
            "raw_rows": [headers, values],
        },
        rows=[],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id=_PROFILE_TEMPLATE,
        tables=[table],
    )
    return SimpleNamespace(pages=[page])


def _token(
    token_id: str,
    text: str,
    bbox: list[float],
) -> SimpleNamespace:
    return SimpleNamespace(id=token_id, text=text, bbox=bbox)


def _merged_scalar_context(
    *,
    header_parts: tuple[str, str] = ("出生日期", "性别"),
    value_parts: tuple[str, str] = ("2002.08.03", "女"),
    scale: float = 1.0,
    raw_header: str | None = None,
    raw_value: str | None = None,
    overlap_values: bool = False,
    duplicate_value_id: bool = False,
    replay_value_id: bool = False,
    extra_value_token: bool = False,
    reverse_value_geometry: bool = False,
) -> SimpleNamespace:
    header_raw = raw_header if raw_header is not None else " ".join(header_parts)
    value_raw = raw_value if raw_value is not None else " ".join(value_parts)
    header_widths = [max(24.0, len(text) * 11.0) * scale for text in header_parts]
    value_widths = [max(24.0, len(text) * 9.0) * scale for text in value_parts]
    gap = 8.0 * scale

    header_left = 4.0 * scale
    header_boxes: list[list[float]] = []
    for width in header_widths:
        header_boxes.append([header_left, 2.0 * scale, header_left + width, 18.0 * scale])
        header_left += width + gap
    value_left = 4.0 * scale
    value_boxes: list[list[float]] = []
    for width in value_widths:
        value_boxes.append([value_left, 32.0 * scale, value_left + width, 48.0 * scale])
        value_left += width + gap
    if overlap_values:
        value_boxes[1][0] = value_boxes[0][2] - 4.0 * scale
    if reverse_value_geometry:
        value_boxes.reverse()

    header_tokens = [_token(f"header-{index}", text, header_boxes[index]) for index, text in enumerate(header_parts)]
    value_tokens = [_token(f"value-{index}", text, value_boxes[index]) for index, text in enumerate(value_parts)]
    value_ids = [token.id for token in value_tokens]
    if duplicate_value_id:
        value_ids.append(value_ids[-1])
    if extra_value_token:
        note_box = [
            value_boxes[-1][2] + gap,
            32.0 * scale,
            value_boxes[-1][2] + gap + 36.0 * scale,
            48.0 * scale,
        ]
        value_tokens.append(_token("value-note", "备注", note_box))
        value_ids.append("value-note")

    cell_right = max(
        *(box[2] for box in header_boxes),
        *(token.bbox[2] for token in value_tokens),
    ) + 4.0 * scale
    header_cell = SimpleNamespace(
        text=header_raw,
        geometry_status="exact",
        evidence_ids=[token.id for token in header_tokens],
        token_ids=[token.id for token in header_tokens],
        bbox=[0.0, 0.0, cell_right, 20.0 * scale],
    )
    value_cell = SimpleNamespace(
        text=value_raw,
        geometry_status="exact",
        evidence_ids=list(value_ids),
        token_ids=list(value_ids),
        bbox=[0.0, 30.0 * scale, cell_right, 50.0 * scale],
    )
    table = SimpleNamespace(
        table_id="profile-scalar-merged",
        metadata={
            "canonical_template_id": _PROFILE_TEMPLATE,
            "source_logical_page": 1,
            "source_page": 1,
            "raw_rows": [[header_raw], [value_raw]],
        },
        rows=[],
        source_cell_objects=[[header_cell], [value_cell]],
    )
    tables = [table]
    if replay_value_id:
        replay_cell = SimpleNamespace(
            text=value_parts[-1],
            geometry_status="exact",
            evidence_ids=["value-1"],
            token_ids=["value-1"],
            bbox=[300.0 * scale, 30.0 * scale, 340.0 * scale, 50.0 * scale],
        )
        tables.append(
            SimpleNamespace(
                table_id="competing-token-owner",
                metadata={"raw_rows": [[value_parts[-1]]]},
                rows=[],
                source_cell_objects=[[replay_cell]],
            )
        )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id=_PROFILE_TEMPLATE,
        tables=tables,
    )
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(evidence=SimpleNamespace(text_atoms=[*header_tokens, *value_tokens])),
    )


def test_profile_strict_scalars_accept_valid_whole_cells_and_real_leap_day() -> None:
    profile = extract_candidate_b_profile(
        _plain_profile_context(
            ["性别", "出生日期", "国籍"],
            ["女", "2000.02.29", "中国（含港澳台）"],
        )
    )

    assert profile["gender"]["normalized_value"] == "女"
    assert profile["birth_date"]["normalized_value"] == "2000-02-29"
    assert profile["nationality"]["normalized_value"] == "中国（含港澳台）"


@pytest.mark.parametrize(
    ("field", "label", "value", "expected"),
    [
        ("mobile_phone", "手机号码", "+86 138-0013-8007", "13800138007"),
        ("work_phone", "单位电话", "010-12345678", "010-12345678"),
        ("residence_phone", "住宅电话", "(010) 12345678", "(010) 12345678"),
        ("email", "电子邮箱", "person@example.com", "person@example.com"),
    ],
)
def test_profile_contact_fields_accept_only_complete_registered_presentations(
    field: str,
    label: str,
    value: str,
    expected: str,
) -> None:
    profile = extract_candidate_b_profile(
        _plain_profile_context(["性别", label], ["女", value])
    )

    assert profile[field]["normalized_value"] == expected


@pytest.mark.parametrize(
    ("field", "label", "value"),
    [
        ("mobile_phone", "手机号码", "$13800138007"),
        ("mobile_phone", "手机号码", "手机13800138007"),
        ("mobile_phone", "手机号码", "138,0013,8007"),
        ("work_phone", "单位电话", "电话010-12345678"),
        ("work_phone", "单位电话", "010--12345678"),
        ("work_phone", "单位电话", "010- -12345678"),
        ("work_phone", "单位电话", "010-12 345 678"),
        ("residence_phone", "住宅电话", "010-12345678?"),
        ("residence_phone", "住宅电话", "123 45 67"),
        ("email", "电子邮箱", "备注 person@example.com"),
        ("email", "电子邮箱", "person@example.com 附注"),
        ("email", "电子邮箱", "person@example.com,"),
        ("email", "电子邮箱", "person..name@example.com"),
        ("email", "电子邮箱", "person@example..com"),
        ("email", "电子邮箱", "person@-example.com"),
        ("email", "电子邮箱", "person@example.com."),
        ("email", "电子邮箱", f"{'a' * 65}@example.com"),
    ],
)
def test_profile_contact_fields_never_substring_mine_contaminated_cells(
    field: str,
    label: str,
    value: str,
) -> None:
    context = _plain_profile_context(["性别", label], ["女", value])

    profile = extract_candidate_b_profile(context)

    assert profile[field]["normalized_value"] is None
    raw = profile[field]["raw"]
    assert value in raw if isinstance(raw, list) else raw == value
    assert any(
        issue.get("issue_code") == "candidate_b_profile_contract_unresolved"
        and issue.get("field_name") == field
        for issue in context._personal_detail_extraction_issues
    )


def test_profile_labels_require_residue_free_whole_cell_aliases() -> None:
    context = _plain_profile_context(
        ["非性别说明", "电子邮箱"],
        ["女", "person@example.com"],
    )

    profile = extract_candidate_b_profile(context)

    assert "gender" not in profile
    assert profile["email"]["normalized_value"] == "person@example.com"
    assert not any(
        issue.get("field_name") == "gender"
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )


def test_profile_does_not_invent_nationality_from_education_email_adjacency() -> None:
    context = _plain_profile_context(
        ["学历学位", "备用", "电子邮箱"],
        ["本科", "中国", "person@example.com"],
    )

    profile = extract_candidate_b_profile(context)

    assert "nationality" not in profile
    assert profile["email"]["normalized_value"] == "person@example.com"
    assert not any(
        issue.get("field_name") == "nationality"
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )


def test_profile_clipped_nationality_requires_exact_header_and_value_owners() -> None:
    context = _plain_profile_context(
        ["学历", "学位", "国", "电子邮箱"],
        ["本科", "学士", "中国", "person@example.com"],
    )

    profile = extract_candidate_b_profile(context)

    assert "nationality" not in profile


def test_profile_merged_contact_fields_require_exact_residue_free_token_owners() -> None:
    profile = extract_candidate_b_profile(
        _merged_scalar_context(
            header_parts=("手机号码", "电子邮箱"),
            value_parts=("+86 138-0013-8007", "person@example.com"),
            scale=1.7,
        )
    )

    assert profile["mobile_phone"]["normalized_value"] == "13800138007"
    assert profile["email"]["normalized_value"] == "person@example.com"
    assert profile["mobile_phone"]["source_refs"][0]["geometry_scope"] == "token"
    assert profile["email"]["source_refs"][0]["geometry_scope"] == "token"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("birth_date", "1990.02.31"),
        ("birth_date", "1990.02.03附注"),
        ("gender", "女数据发生机构名称示例银行"),
        ("nationality", "中国工商银行股份有限公司"),
    ],
)
def test_profile_strict_scalar_whole_cell_contract_rejects_contamination(
    field: str,
    value: str,
) -> None:
    labels = {
        "birth_date": "出生日期",
        "gender": "性别",
        "nationality": "国籍",
    }
    context = _plain_profile_context([labels[field]], [value])

    profile = extract_candidate_b_profile(context)

    assert profile[field]["normalized_value"] is None
    assert profile[field]["observation_status"] == "unreadable"
    assert any(
        issue.get("issue_code") == "candidate_b_profile_contract_unresolved" and issue.get("field_name") == field
        for issue in context._personal_detail_extraction_issues
    )


@pytest.mark.parametrize(
    ("header_parts", "value_parts", "scale"),
    [
        (("出生日期", "性别"), ("2002.08.03", "女"), 0.65),
        (("性别", "出生日期"), ("女", "2002.08.03"), 2.4),
    ],
)
def test_profile_merged_scalars_use_semantic_token_owners_not_order_or_scale(
    header_parts: tuple[str, str],
    value_parts: tuple[str, str],
    scale: float,
) -> None:
    profile = extract_candidate_b_profile(
        _merged_scalar_context(
            header_parts=header_parts,
            value_parts=value_parts,
            scale=scale,
        )
    )

    assert profile["gender"]["normalized_value"] == "女"
    assert profile["birth_date"]["normalized_value"] == "2002-08-03"
    assert profile["gender"]["source_refs"][0]["geometry_scope"] == "token"
    assert profile["birth_date"]["source_refs"][0]["geometry_scope"] == "token"
    assert profile["gender"]["source_refs"][0]["evidence_ids"] == [f"value-{value_parts.index('女')}"]


def test_profile_merged_scalars_reject_type_based_cross_column_reordering() -> None:
    context = _merged_scalar_context(
        reverse_value_geometry=True,
        raw_value="女 2002.08.03",
    )

    profile = extract_candidate_b_profile(context)

    assert profile["gender"]["normalized_value"] is None
    assert profile["birth_date"]["normalized_value"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_residue",
        "single_role_value",
        "extra_token",
        "duplicate_id",
        "replayed_owner",
        "overlap",
        "invalid_date",
        "malformed_label",
        "topology_reversal",
        "ambiguous_vocabulary",
    ],
)
def test_profile_merged_scalar_partition_fails_closed_on_adversarial_mutations(
    mutation: str,
) -> None:
    kwargs: dict[str, object] = {}
    fields = ("birth_date", "gender")
    if mutation == "raw_residue":
        kwargs["raw_value"] = "2002.08.03 女 备注"
    elif mutation == "single_role_value":
        kwargs["raw_value"] = "女"
    elif mutation == "extra_token":
        kwargs.update(raw_value="2002.08.03 女 备注", extra_value_token=True)
    elif mutation == "duplicate_id":
        kwargs["duplicate_value_id"] = True
    elif mutation == "replayed_owner":
        kwargs["replay_value_id"] = True
    elif mutation == "overlap":
        kwargs["overlap_values"] = True
    elif mutation == "invalid_date":
        kwargs["value_parts"] = ("2002.02.30", "女")
    elif mutation == "malformed_label":
        kwargs.update(header_parts=("出生日", "性别"), raw_header="出生日 性别")
    elif mutation == "topology_reversal":
        kwargs["reverse_value_geometry"] = True
    elif mutation == "ambiguous_vocabulary":
        kwargs.update(
            header_parts=("学位", "学历"),
            value_parts=("未知", "大专"),
        )
        fields = ("degree", "education_level")

    context = _merged_scalar_context(**kwargs)
    profile = extract_candidate_b_profile(context)

    assert all(field not in profile or profile[field]["normalized_value"] is None for field in fields)
