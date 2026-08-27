from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.models.mirror.vnext import EvidenceAtom
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    _exact_source_table_repair_tokens_by_page,
)
from docmirror.plugins.credit_report.personal_detail_scanned.exact_evidence import (
    resolve_exact_page_token_atoms,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _exact_native_table_cell_tokens,
)
from docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction import (
    _exact_profile_cell_tokens,
    _row_label_fields,
)


def _owner(*atoms: Any) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_plane=SimpleNamespace(
            evidence=SimpleNamespace(text_atoms=list(atoms))
        )
    )


def _raw_atom(
    *,
    atom_id: str,
    token_id: str,
    page: int = 1,
    text: str = "N",
    source_refs: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=atom_id,
        source_kind="metadata_ocr_token",
        page_id=f"page:{page:04d}",
        text=text,
        bbox=[10.0, 20.0, 20.0, 30.0],
        source_refs=[token_id] if source_refs is None else source_refs,
        metadata={},
    )


@pytest.mark.parametrize("require_raw_tokens", (False, True))
def test_opaque_raw_atom_resolves_by_one_source_reference(require_raw_tokens: bool) -> None:
    owner = _owner(
        _raw_atom(atom_id="ev:0001:text:opaque", token_id="ocr-token")
    )

    assert resolve_exact_page_token_atoms(
        owner,
        ["ocr-token"],
        logical_page=1,
        require_raw_tokens=require_raw_tokens,
    ) == (("N", (10.0, 20.0, 20.0, 30.0), "ocr-token"),)


@pytest.mark.parametrize("require_raw_tokens", (False, True))
def test_direct_legacy_raw_atom_may_omit_redundant_source_reference(
    require_raw_tokens: bool,
) -> None:
    owner = _owner(
        _raw_atom(
            atom_id="ocr-token",
            token_id="ocr-token",
            source_refs=[],
        )
    )

    assert resolve_exact_page_token_atoms(
        owner,
        ["ocr-token"],
        logical_page=1,
        require_raw_tokens=require_raw_tokens,
    ) == (("N", (10.0, 20.0, 20.0, 30.0), "ocr-token"),)


@pytest.mark.parametrize("require_raw_tokens", (False, True))
def test_table_cell_projection_cannot_satisfy_raw_token_ownership(
    require_raw_tokens: bool,
) -> None:
    owner = _owner(
        SimpleNamespace(
            id="ev:0001:text:cell",
            source_kind="parse_result_table_cell",
            page_id="page:0001",
            text="M",
            bbox=[10.0, 20.0, 20.0, 30.0],
            source_refs=["ocr-token"],
            metadata={"granularity": "token"},
        ),
        _raw_atom(atom_id="ev:0001:text:raw", token_id="ocr-token"),
    )

    assert resolve_exact_page_token_atoms(
        owner,
        ["ocr-token"],
        logical_page=1,
        require_raw_tokens=require_raw_tokens,
    ) == (("N", (10.0, 20.0, 20.0, 30.0), "ocr-token"),)


@pytest.mark.parametrize(
    "atoms,requested,page",
    [
        (
            [_raw_atom(atom_id="ev:wrong-page", token_id="ocr-token", page=2)],
            ["ocr-token"],
            1,
        ),
        (
            [
                _raw_atom(atom_id="ev:one", token_id="ocr-token"),
                _raw_atom(atom_id="ev:two", token_id="ocr-token"),
            ],
            ["ocr-token"],
            1,
        ),
        (
            [_raw_atom(atom_id="ev:partial", token_id="ocr-token")],
            ["ocr-token", "missing-token"],
            1,
        ),
        (
            [
                SimpleNamespace(
                    id="ocr-token",
                    source_kind="parse_result_table_cell",
                    page_id="page:0001",
                    text="N",
                    bbox=[10.0, 20.0, 20.0, 30.0],
                    source_refs=["ocr-token"],
                    metadata={"granularity": "token"},
                )
            ],
            ["ocr-token"],
            1,
        ),
    ],
    ids=("wrong-page", "duplicate", "partial", "direct-non-raw"),
)
def test_exact_token_resolution_fails_closed(
    atoms: list[SimpleNamespace],
    requested: list[str],
    page: int,
) -> None:
    assert resolve_exact_page_token_atoms(
        _owner(*atoms),
        requested,
        logical_page=page,
        require_raw_tokens=True,
    ) is None


@pytest.mark.parametrize(
    "atom_factory", (dict, EvidenceAtom, SimpleNamespace), ids=("dict", "sealed-model", "attribute-record")
)
@pytest.mark.parametrize("logical_page", (None, 0, 1))
def test_generic_direct_ids_preserve_sealed_atoms_without_raw_metadata(
    atom_factory: Any,
    logical_page: int | None,
) -> None:
    # These are the actual direct-ID shapes used by the inquiry/profile/header
    # readers, not raw-token fixtures with manufactured provenance metadata.
    first = atom_factory(id="header:open", text="开立日期", bbox=[10, 20, 50, 30])
    second = atom_factory(id="header:limit", text="账户授信额度", bbox=[60, 20, 120, 30])
    if isinstance(first, EvidenceAtom):
        assert first.source_kind == "parse_result"
        assert first.page_id == ""
    owner = _owner(second, first)
    before = deepcopy(owner.evidence_plane.evidence.text_atoms)

    assert resolve_exact_page_token_atoms(
        owner, ["header:open", "header:limit"], logical_page=logical_page
    ) == (
        ("开立日期", (10.0, 20.0, 50.0, 30.0), "header:open"),
        ("账户授信额度", (60.0, 20.0, 120.0, 30.0), "header:limit"),
    )
    assert resolve_exact_page_token_atoms(
        owner,
        ["header:open", "header:limit"],
        logical_page=1,
        require_raw_tokens=True,
    ) is None
    assert owner.evidence_plane.evidence.text_atoms == before


@pytest.mark.parametrize("logical_page", (False, -1, "0", "1"))
def test_generic_direct_ids_do_not_treat_invalid_page_hints_as_unspecified(logical_page: Any) -> None:
    atom = SimpleNamespace(id="header:open", text="开立日期", bbox=[10, 20, 50, 30])
    assert resolve_exact_page_token_atoms(
        _owner(atom), [atom.id], logical_page=logical_page
    ) is None


def test_strict_direct_raw_token_does_not_accept_generic_zero_page_sentinel() -> None:
    atom = _raw_atom(atom_id="ocr-token", token_id="ocr-token")
    assert resolve_exact_page_token_atoms(
        _owner(atom), ["ocr-token"], logical_page=0, require_raw_tokens=True
    ) is None


@pytest.mark.parametrize("source_page_hint", (None, 0, 1))
def test_clipped_nationality_signature_keeps_generic_exact_cell_page_contract(
    source_page_hint: int | None,
) -> None:
    # This profile helper supplies integer 0 when the table has no source-page
    # metadata. The closed four-column signature still has independent exact
    # header/value atoms; the value must not be lost at the generic ID boundary.
    raw_rows = [
        ["学历", "学位", "国", "电子邮箱"],
        ["大专", "--", "中国(含港澳台)", "user@example.com"],
    ]
    header = SimpleNamespace(id="h2", text="国", bbox=[55.0, 1.0, 65.0, 9.0])
    value = SimpleNamespace(id="v2", text="中国(含港澳台)", bbox=[52.0, 11.0, 73.0, 19.0])
    metadata: dict[str, Any] = {
        "canonical_template_id": "report_header_and_identity",
        "raw_rows": raw_rows,
    }
    if source_page_hint is not None:
        metadata["source_logical_page"] = source_page_hint
    source_cells = []
    for atom, box in ((header, [50, 0, 75, 10]), (value, [50, 10, 75, 20])):
        source_cells.append(
            [
                None,
                None,
                SimpleNamespace(
                    text=atom.text,
                    bbox=box,
                    geometry_status="exact",
                    evidence_ids=[atom.id],
                    token_ids=[atom.id],
                ),
                None,
            ]
        )
    table = SimpleNamespace(
        table_id="identity-profile-clipped-nationality",
        metadata=metadata,
        rows=[],
        source_cell_objects=source_cells,
    )
    context = _owner(header, value)
    context.pages = [SimpleNamespace(page_number=1, tables=[table])]

    assert (2, "nationality") in _row_label_fields(
        raw_rows[0], context=context, table=table, row_index=0
    )
    assert _exact_profile_cell_tokens(
        context, table, 1, 2, logical_page=source_page_hint or 0
    ) == (("中国(含港澳台)", (52.0, 11.0, 73.0, 19.0), "v2"),)


@pytest.mark.parametrize(
    "source_kind",
    ("parse_result_table_cell", "parse_result_text", "semantic_projection", "metadata_ocr_token"),
)
def test_generic_direct_atom_source_refs_are_provenance_not_its_canonical_id(
    source_kind: str,
) -> None:
    atom = EvidenceAtom(
        id="ev:0001:text:canonical",
        text="2021.05.31",
        bbox=[10, 20, 80, 30],
        source_kind=source_kind,
        page_id="page:0001",
        source_refs=["underlying-token:1", "underlying-token:2"],
    )
    owner = _owner(atom)

    assert resolve_exact_page_token_atoms(
        owner, [atom.id], logical_page=1
    ) == (("2021.05.31", (10.0, 20.0, 80.0, 30.0), atom.id),)
    assert resolve_exact_page_token_atoms(
        owner, [atom.id], logical_page=1, require_raw_tokens=True
    ) is None


@pytest.mark.parametrize("page_id", ("page:0002", "page:1", 1, False))
def test_generic_direct_atoms_reject_explicit_wrong_or_malformed_page(page_id: Any) -> None:
    atom = {"id": "sealed-token", "text": "N", "bbox": [10, 20, 20, 30], "page_id": page_id}
    assert resolve_exact_page_token_atoms(
        _owner(atom), ["sealed-token"], logical_page=1
    ) is None


def _with_bundle(owner: SimpleNamespace, *token_ids: str) -> SimpleNamespace:
    owner.parse_result = SimpleNamespace(
        entities=SimpleNamespace(
            domain_specific={
                "_page_evidence_bundles": [
                    {
                        "page": 1,
                        "tokens": [
                            {
                                "token_id": token_id,
                                "page": 1,
                                "evidence_ids": [token_id],
                                "text": "N",
                                "bbox": [10, 20, 20, 30],
                            }
                            for token_id in token_ids
                        ],
                    }
                ]
            }
        )
    )
    return owner


@pytest.mark.parametrize("require_raw_tokens", (False, True))
@pytest.mark.parametrize("defect", ("duplicate", "partial", "wrong-page", "geometry", "text"))
def test_plane_ownership_defects_cannot_fall_back_to_a_good_bundle(
    require_raw_tokens: bool,
    defect: str,
) -> None:
    first: Any = (
        _raw_atom(atom_id="first", token_id="first")
        if require_raw_tokens
        else {"id": "first", "text": "N", "bbox": [10, 20, 20, 30]}
    )
    second: Any = (
        _raw_atom(atom_id="second", token_id="second")
        if require_raw_tokens
        else {"id": "second", "text": "N", "bbox": [10, 20, 20, 30]}
    )
    atoms = [first, second]
    if defect == "duplicate":
        atoms.append(deepcopy(first))
    elif defect == "partial":
        atoms.pop()
    else:
        key, value = {
            "wrong-page": ("page_id", "page:0002"),
            "geometry": ("bbox", [10, 20, 10, 30]),
            "text": ("content", "M"),
        }[defect]
        if isinstance(first, dict):
            first[key] = value
        else:
            setattr(first, key, value)
    owner = _with_bundle(_owner(*atoms), "first", "second")

    assert resolve_exact_page_token_atoms(
        owner, ["first", "second"], logical_page=1, require_raw_tokens=require_raw_tokens
    ) is None


@pytest.mark.parametrize("require_raw_tokens", (False, True))
@pytest.mark.parametrize(
    "source_refs",
    (
        "ocr-token",
        {"ocr-token": "not-a-reference-list"},
        ["ocr-token", None],
        ["ocr-token", ""],
        ["ocr-token", "ocr-token"],
        ["ocr-token", "another-token"],
    ),
)
def test_opaque_raw_alias_ownership_must_be_one_well_formed_reference(
    require_raw_tokens: bool,
    source_refs: Any,
) -> None:
    atom = _raw_atom(atom_id="ev:opaque", token_id="ocr-token")
    atom.source_refs = source_refs
    owner = _with_bundle(_owner(atom), "ocr-token")

    assert resolve_exact_page_token_atoms(
        owner, ["ocr-token"], logical_page=1, require_raw_tokens=require_raw_tokens
    ) is None


@pytest.mark.parametrize("require_raw_tokens", (False, True))
@pytest.mark.parametrize("page_id", (None, "", "page:0002"))
def test_opaque_raw_alias_requires_explicit_matching_page(
    require_raw_tokens: bool,
    page_id: str | None,
) -> None:
    atom = _raw_atom(atom_id="ev:opaque", token_id="ocr-token")
    atom.page_id = page_id
    assert resolve_exact_page_token_atoms(
        _with_bundle(_owner(atom), "ocr-token"),
        ["ocr-token"],
        logical_page=1,
        require_raw_tokens=require_raw_tokens,
    ) is None


@pytest.mark.parametrize("require_raw_tokens", (False, True))
@pytest.mark.parametrize("logical_page", (None, 0, -1, True, "1"))
def test_raw_alias_lookup_never_infers_or_coerces_the_requested_page(
    require_raw_tokens: bool,
    logical_page: Any,
) -> None:
    atom = _raw_atom(atom_id="ev:opaque", token_id="ocr-token")
    assert resolve_exact_page_token_atoms(
        _owner(atom), ["ocr-token"], logical_page=logical_page, require_raw_tokens=require_raw_tokens
    ) is None


@pytest.mark.parametrize("require_raw_tokens", (False, True))
def test_exact_bundle_fallback_retains_its_singleton_token_contract(require_raw_tokens: bool) -> None:
    owner = _with_bundle(_owner(), "N")
    assert resolve_exact_page_token_atoms(
        owner, ["N"], logical_page=1, require_raw_tokens=require_raw_tokens
    ) == (("N", (10.0, 20.0, 20.0, 30.0), "N"),)
    token = owner.parse_result.entities.domain_specific["_page_evidence_bundles"][0]["tokens"][0]
    token["evidence_ids"] = "N"
    assert resolve_exact_page_token_atoms(
        owner, ["N"], logical_page=1, require_raw_tokens=require_raw_tokens
    ) is None


@pytest.mark.parametrize("source_kind", ("parse_result_table_cell", "ocr_text_line", "semantic_projection"))
def test_projection_with_a_direct_id_cannot_authorize_monthly_raw_bundle_fallback(source_kind: str) -> None:
    atom = _raw_atom(atom_id="ocr-token", token_id="ocr-token")
    atom.source_kind = source_kind
    atom.metadata = {"granularity": "token"}
    owner = _with_bundle(_owner(atom), "ocr-token")

    assert resolve_exact_page_token_atoms(
        owner, ["ocr-token"], logical_page=1, require_raw_tokens=True
    ) is None


@pytest.mark.parametrize("shape", ("dict", "sealed-model", "typed-projection", "raw-direct", "raw-opaque"))
def test_monthly_context_opts_into_raw_only_without_narrowing_shared_native_cell_readers(shape: str) -> None:
    # One printed status/amount excerpt. The shared cell reader may use sealed
    # canonical IDs, but monthly repair must independently demand the raw plane.
    boxes = [[10.0, 20.0, 20.0, 30.0], [10.0, 30.0, 20.0, 40.0]]
    ids = ["monthly-status", "monthly-amount"]
    atoms: list[Any] = []
    for token_id, text, box in zip(ids, ("N", "0"), boxes, strict=True):
        if shape == "dict":
            atom: Any = {"id": token_id, "text": text, "bbox": box}
        elif shape == "sealed-model":
            atom = EvidenceAtom(id=token_id, text=text, bbox=box)
        else:
            atom = _raw_atom(
                atom_id=f"ev:{token_id}" if shape == "raw-opaque" else token_id,
                token_id=token_id,
                text=text,
            )
            atom.bbox = box
            atom.confidence = 0.97
            if shape == "typed-projection":
                atom.source_kind = "parse_result_table_cell"
                atom.metadata = {"granularity": "token"}
        atoms.append(atom)
    table = SimpleNamespace(
        table_id="monthly-pair-excerpt",
        rows=[["?"], ["?"]],
        metadata={
            "geometry": {
                "coordinate_system": "pdf_points_top_left",
                "cell_bboxes": [[box] for box in boxes],
                "cell_geometry_status": [["exact"], ["exact"]],
                "cell_evidence_ids": [[[token_id]] for token_id in ids],
                "cell_token_ids": [[[token_id]] for token_id in ids],
                "cell_spans": [],
                "row_bands": [
                    {"index": index, "y0": box[1], "y1": box[3]}
                    for index, box in enumerate(boxes)
                ],
                "col_bands": [{"index": 0, "x0": 10.0, "x1": 20.0}],
            }
        },
    )
    owner = _owner(*atoms)
    page = SimpleNamespace(page_number=1, tables=[table])
    assert _exact_native_table_cell_tokens(
        owner, table, row=0, column=0, logical_page=1
    ) == (("N", tuple(boxes[0]), ids[0]),)

    repaired = _exact_source_table_repair_tokens_by_page(owner, [page], {1})
    if shape in {"raw-direct", "raw-opaque"}:
        assert [(token["token_id"], token["content"]) for token in repaired[1]] == list(zip(ids, ("N", "0")))
    else:
        assert repaired == {}
