from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    extract_candidate_b_profile,
)

_PROFILE_TEMPLATE = "report_header_and_identity"
_GENDER = "\u6027\u522b"
_FEMALE = "\u5973"
_MAILING = "\u901a\u8baf\u5730\u5740"
_HOUSEHOLD = "\u6237\u7c4d\u5730\u5740"
_NOTE = "\u5907\u6ce8"
_HOUSEHOLD_VALUE = "\u798f\u5efa\u7701\u798f\u5dde\u5e02\u9f13\u697c\u533a\u793a\u4f8b\u8def1\u53f7"
_MAILING_VALUE = "\u798f\u5efa\u7701\u53a6\u95e8\u5e02\u601d\u660e\u533a\u793a\u4f8b\u8def2\u53f7"


def _plain_context(headers: list[str], values: list[str]) -> SimpleNamespace:
    table = SimpleNamespace(
        table_id="profile-household-owner",
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


def _merged_context(*, duplicate_owner: bool = False) -> SimpleNamespace:
    headers = [_GENDER, f"{_MAILING} {_HOUSEHOLD}"]
    values = [_FEMALE, _HOUSEHOLD_VALUE]
    header_tokens = [
        SimpleNamespace(
            text=_MAILING,
            bbox=[110.0, 2.0, 165.0, 18.0],
            id="mailing-header",
        ),
        SimpleNamespace(
            text=_HOUSEHOLD,
            bbox=[170.0, 2.0, 225.0, 18.0],
            id="household-header",
        ),
    ]
    cells = [
        SimpleNamespace(
            text=_GENDER,
            geometry_status="exact",
            evidence_ids=["gender-header"],
            token_ids=["gender-header"],
            bbox=[0.0, 0.0, 100.0, 20.0],
        ),
        SimpleNamespace(
            text=headers[1],
            geometry_status="exact",
            evidence_ids=[token.id for token in header_tokens],
            token_ids=[token.id for token in header_tokens],
            bbox=[100.0, 0.0, 240.0, 20.0],
        ),
    ]
    if duplicate_owner:
        headers.append(f"{_HOUSEHOLD} {_MAILING}")
        values.append(_MAILING_VALUE)
        duplicate_tokens = [
            SimpleNamespace(
                text=_HOUSEHOLD,
                bbox=[250.0, 2.0, 305.0, 18.0],
                id="household-header-duplicate",
            ),
            SimpleNamespace(
                text=_MAILING,
                bbox=[310.0, 2.0, 365.0, 18.0],
                id="mailing-header-duplicate",
            ),
        ]
        header_tokens.extend(duplicate_tokens)
        cells.append(
            SimpleNamespace(
                text=headers[2],
                geometry_status="exact",
                evidence_ids=[token.id for token in duplicate_tokens],
                token_ids=[token.id for token in duplicate_tokens],
                bbox=[240.0, 0.0, 380.0, 20.0],
            )
        )
    table = SimpleNamespace(
        table_id="profile-merged-household-owner",
        metadata={
            "canonical_template_id": _PROFILE_TEMPLATE,
            "source_logical_page": 1,
            "source_page": 1,
            "raw_rows": [headers, values],
        },
        rows=[],
        source_cell_objects=[cells, [None] * len(headers)],
    )
    page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id=_PROFILE_TEMPLATE,
        tables=[table],
    )
    atoms = [
        SimpleNamespace(id="gender-header", text=_GENDER, bbox=[10.0, 2.0, 50.0, 18.0]),
        *header_tokens,
    ]
    return SimpleNamespace(
        pages=[page],
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(text_atoms=atoms)
        ),
    )


def test_exact_household_label_extracts_when_columns_are_reordered() -> None:
    context = _plain_context(
        [_HOUSEHOLD, _GENDER, _MAILING],
        [_HOUSEHOLD_VALUE, _FEMALE, _MAILING_VALUE],
    )

    profile = extract_candidate_b_profile(context)

    assert profile["household_address"]["normalized_value"] == _HOUSEHOLD_VALUE


def test_unknown_column_after_mailing_address_never_becomes_household() -> None:
    context = _plain_context(
        [_MAILING, _NOTE, _GENDER],
        [_HOUSEHOLD_VALUE, _MAILING_VALUE, _FEMALE],
    )

    profile = extract_candidate_b_profile(context)

    assert "household_address" not in profile


def test_arbitrary_unknown_column_never_becomes_household() -> None:
    context = _plain_context(
        [_MAILING, "\u4efb\u610f\u672a\u6ce8\u518c\u5217", _GENDER],
        [_HOUSEHOLD_VALUE, _MAILING_VALUE, _FEMALE],
    )

    profile = extract_candidate_b_profile(context)

    assert "household_address" not in profile


def test_duplicate_exact_household_labels_are_withheld() -> None:
    context = _plain_context(
        [_HOUSEHOLD, _GENDER, _HOUSEHOLD],
        [_HOUSEHOLD_VALUE, _FEMALE, _MAILING_VALUE],
    )

    profile = extract_candidate_b_profile(context)

    assert "household_address" not in profile


def test_source_owned_merged_household_header_trait_extracts() -> None:
    profile = extract_candidate_b_profile(_merged_context())

    assert profile["household_address"]["normalized_value"] == _HOUSEHOLD_VALUE


def test_merged_household_header_requires_unique_source_owner() -> None:
    profile = extract_candidate_b_profile(_merged_context(duplicate_owner=True))

    assert "household_address" not in profile
