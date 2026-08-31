# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for lazy embedded bank-seal metadata OCR."""

from __future__ import annotations

from docmirror.plugins.bank_statement.embedded_metadata import (
    _bank_matches,
    _branch_matches,
    _candidate_image_atoms,
    _code_matches,
    _facts_from_ocr_words,
    _forced_multiline_code_observations,
    _inline_image_bytes,
    _Observation,
    _select_observation,
)


def _word(text: str, confidence: float = 0.99) -> tuple:
    return (0.0, 0.0, 100.0, 20.0, text, 0, 0, 0, confidence)


def test_embedded_seal_ocr_atomizes_bank_branch_and_code() -> None:
    facts = _facts_from_ocr_words(
        [
            _word("中国工商银行股份有限公司"),
            _word("镇江京口支行"),
            _word("业务专用章"),
            _word("8DD4EA031026"),
        ],
        page=1,
        page_id="page:0001",
        bbox=(1.0, 2.0, 3.0, 4.0),
        evidence_id="ev:0001:image:000001",
    )

    assert {fact.field_key: fact.value for fact in facts} == {
        "seal_type": "业务专用章",
        "issuing_branch": "镇江京口支行",
        "seal_code": "8DD4EA031026",
        "issuing_bank": "中国工商银行股份有限公司",
    }


def test_embedded_seal_ocr_rejects_irrelevant_or_low_confidence_identifiers() -> None:
    irrelevant = _facts_from_ocr_words(
        [_word("付款二维码"), _word("8DD4EA031026")],
        page=1,
        page_id="page:0001",
        bbox=None,
        evidence_id="image:1",
    )
    low_confidence = _facts_from_ocr_words(
        [_word("业务专用章"), _word("8DD4EA031026", 0.69)],
        page=1,
        page_id="page:0001",
        bbox=None,
        evidence_id="image:1",
    )

    assert irrelevant == []
    assert not any(fact.field_key == "seal_code" for fact in low_confidence)


def test_embedded_seal_code_consensus_supports_split_alphanumeric_and_numeric_codes() -> None:
    assert _code_matches(["FBF3QCBY", "M3QKX1TE"]) == ["FBF3QCBY", "M3QKX1TE", "FBF3QCBYM3QKX1TE"]
    assert _code_matches(["030305026"]) == ["030305026"]

    selected = _select_observation(
        "seal_code",
        [
            _Observation("seal_code", "CQ15AP3I9116", 0.91, "fixed_scale_1"),
            _Observation("seal_code", "CQ15AP3I9116", 0.94, "fixed_scale_1.5"),
            _Observation("seal_code", "CQ15AP319116", 0.96, "fixed_scale_2"),
        ],
    )

    assert selected is not None
    assert selected[0] == "CQ15AP3I9116"


def test_forced_multiline_code_recognition_preserves_one_logical_identifier() -> None:
    class Image:
        shape = (198, 318, 3)

    class Engine:
        @staticmethod
        def detect_image_words(_image, *, multi_scale):
            assert multi_scale is False
            return [
                (119, 140, 198, 153, "FBF3QCBY", 0, 0, 0, 0.99),
                (119, 158, 198, 171, "H3QKX1TE", 0, 0, 0, 0.95),
            ]

        @staticmethod
        def force_recognize_regions(_image, regions):
            assert regions == [(104, 127, 213, 159), (104, 145, 213, 177)]
            return [
                (*regions[0], "FBF3QCBY", 0, 0, 0, 0.99),
                (*regions[1], "M3QKX1TE", 0, 0, 0, 0.96),
            ]

    observations = _forced_multiline_code_observations(Image(), Engine())

    assert observations == [
        _Observation(
            "seal_code",
            "FBF3QCBYM3QKX1TE",
            0.96,
            "forced_multiline_code_1",
        )
    ]
    selected = _select_observation(
        "seal_code",
        [
            _Observation("seal_code", "FBF3QCBY", 0.99, "fixed_scale_1"),
            *observations,
        ],
    )
    assert selected is not None
    assert selected[0] == "FBF3QCBYM3QKX1TE"


def test_numeric_account_inside_image_is_not_promoted_to_seal_code() -> None:
    selected = _select_observation(
        "seal_code",
        [
            _Observation(
                "seal_code",
                "6217000730029163942",
                1.0,
                "native_text_in_image_bbox",
            )
        ],
    )

    assert selected is None


def test_numeric_bottom_crop_noise_requires_a_strict_consensus_majority() -> None:
    observations = [
        _Observation("seal_code", "2993409370301940", 0.95, f"page_render_bottom_{index}")
        for index in range(1, 4)
    ]
    observations.extend(
        _Observation("seal_code", str(10_000_000 + index), 0.95, f"page_render_bottom_{index}")
        for index in range(4, 13)
    )

    assert _select_observation("seal_code", observations) is None


def test_curved_legal_bank_name_can_stitch_across_numeric_ocr_noise() -> None:
    assert "平安银行股份有限公司" in _bank_matches(
        ["平安银行股份有限公", "90000000", "司"]
    )


def test_curved_legal_bank_suffix_survives_lower_confidence_terminal_character() -> None:
    facts = _facts_from_ocr_words(
        [
            _word("零售业务电子凭证专用章", 0.99),
            _word("平安银行股份有限公", 0.99),
            _word("9809080C01", 0.37),
            _word("司", 0.63),
        ],
        page=1,
        page_id="page:0001",
        bbox=(1.0, 2.0, 3.0, 4.0),
        evidence_id="ev:0001:image:000001",
    )

    assert {fact.field_key: fact.value for fact in facts}["issuing_bank"] == "平安银行股份有限公司"


def test_expanded_retail_seal_marker_is_preserved_atomically() -> None:
    facts = _facts_from_ocr_words(
        [_word("零售业务电子凭证专用章")],
        page=1,
        page_id="page:0001",
        bbox=(1.0, 2.0, 3.0, 4.0),
        evidence_id="ev:0001:image:000001",
    )

    assert {fact.field_key: fact.value for fact in facts} == {
        "seal_type": "零售业务电子凭证专用章"
    }


def test_branch_matching_preserves_an_exact_source_token_before_joined_fallback() -> None:
    assert _branch_matches(["中国邮政储蓄银行股份有限公司", "上海普陀区澳门路支行"]) == [
        "上海普陀区澳门路支行"
    ]
    assert _branch_matches(["中国邮政储蓄银行股份有限公司上海普陀区澳门路支行"]) == [
        "上海普陀区澳门路支行"
    ]


def test_inline_xref_zero_images_remain_lazy_ocr_candidates_and_match_source_bbox() -> None:
    atom = {
        "id": "ev:0001:image:000001",
        "kind": "embedded_image",
        "page_id": "page:0001",
        "bbox": [10.0, 20.0, 110.0, 120.0],
        "metadata": {"width": 171, "height": 171, "xref": 0},
    }
    parse_result = {
        "evidence_plane": {"evidence": {"image_atoms": [atom]}},
    }

    class Page:
        @staticmethod
        def get_text(_kind):
            return {
                "blocks": [
                    {
                        "type": 1,
                        "bbox": (10.0, 20.0, 110.0, 120.0),
                        "width": 171,
                        "height": 171,
                        "image": b"inline-image",
                    }
                ]
            }

    assert _candidate_image_atoms(parse_result) == [atom]
    assert _inline_image_bytes(Page(), atom) == b"inline-image"


def test_page_render_code_requires_repeated_high_confidence_observations() -> None:
    observations = [
        _Observation("seal_code", "030305026", 0.99, f"page_render_bottom_{index}")
        for index in range(1, 7)
    ]
    observations.append(_Observation("seal_code", "4403030502843", 0.99, "page_render_bottom_7"))

    selected = _select_observation("seal_code", observations)

    assert selected is not None
    assert selected[0] == "030305026"
