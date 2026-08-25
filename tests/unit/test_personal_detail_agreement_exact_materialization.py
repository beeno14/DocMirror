from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import native_extraction
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)

_IDENTIFIER = "B1234567890123456789012345678901234567"
_CORROBORATED_IDENTIFIER = "C9876543210987654321098765432109876543"
_YU_CORROBORATED_IDENTIFIER = "B10711000H0001100000111111112446567900000"


def _rows(
    *,
    identifier: str = _IDENTIFIER,
    institution: str = "中国光大银行股份有限公司",
) -> list[list[str]]:
    return [
        [
            "管理机构",
            "授信协议标识",
            "生效日期",
            "到期日期",
            "授信额度用途 信用卡共享额度",
        ],
        [
            f"{institution} 授信额度",
            identifier,
            "2019.12.01",
            "长期 杂",
            "",
        ],
        ["", "授信限额", "授信限额编号", "已用额度", "币种 人民币元"],
        ["0", "--", "--", "0", ""],
    ]


def _table(
    *,
    table_id: str = "agreement-exact-owner",
    rows: list[list[str]] | None = None,
    missing_evidence: tuple[int, int] | None = None,
    top: float = 100.0,
) -> SimpleNamespace:
    materialized = deepcopy(rows if rows is not None else _rows())
    cell_bboxes = [
        [
            [20.0 + column * 112.0, top + row * 20.0, 132.0 + column * 112.0, top + 20.0 + row * 20.0]
            for column in range(len(values))
        ]
        for row, values in enumerate(materialized)
    ]
    cell_evidence_ids = [
        [
            (
                []
                if missing_evidence == (row, column)
                else [f"ocr:agreement:{table_id}:{row}:{column}"]
            )
            for column in range(len(values))
        ]
        for row, values in enumerate(materialized)
    ]
    return SimpleNamespace(
        table_id=table_id,
        bbox=[20.0, top, 580.0, top + len(materialized) * 20.0],
        headers=[],
        rows=[],
        metadata={
            "raw_rows": materialized,
            "cell_bboxes": cell_bboxes,
            "cell_evidence_ids": cell_evidence_ids,
        },
    )


def _context(
    *tables: SimpleNamespace,
    frozen_tables: list[SimpleNamespace] | None = None,
    accounts: list[dict] | None = None,
    evidence_pages: list[dict] | None = None,
) -> SimpleNamespace:
    page = SimpleNamespace(
        page_number=12,
        source_page_number=6,
        tables=list(tables),
        texts=[],
    )
    frozen_page = SimpleNamespace(
        page_number=12,
        source_page_number=6,
        tables=list(tables) if frozen_tables is None else frozen_tables,
        texts=[],
    )
    context = SimpleNamespace(
        pages=[page],
        _frozen_logical_pages={12: frozen_page},
        reading_order_by_logical={12: 12},
        tables_continue=lambda _left, _right: None,
        _personal_detail_extraction_issues=[],
    )
    if accounts is not None:
        context.account_collections = lambda: (deepcopy(accounts), [], [])
    if evidence_pages is not None:
        context.corrected_evidence_pages = lambda: deepcopy(evidence_pages)
    return context


def _field_ref(
    field_name: str,
    *,
    ordinal: int,
    offset: float,
    logical_page: int = 20,
    source_page: int = 10,
) -> dict:
    return {
        "source": "candidate_b_account_anchor_interval",
        "field_name": field_name,
        "binding": "canonical_account_header_geometry",
        "binding_quality": "canonical_account_header_geometry",
        "logical_page": logical_page,
        "source_page": source_page,
        "bbox": [20.0 + offset, 100.0 + ordinal * 30.0, 35.0 + offset, 110.0 + ordinal * 30.0],
        "evidence_ids": [f"ocr:account:{ordinal}:{field_name}"],
    }


def _account_witness(
    ordinal: int,
    *,
    identifier: str = _CORROBORATED_IDENTIFIER,
    institution: str = "中国光大银行股份有限公司",
    currency: str = "CNY",
    limit: int = 0,
    anchor_evidence: str | None = None,
    anchor_bbox: list[float] | None = None,
    logical_page: int = 20,
    source_page: int = 10,
) -> tuple[dict, dict]:
    evidence_id = anchor_evidence or f"ocr:account:{ordinal}:anchor"
    bbox = anchor_bbox or [20.0, 100.0 + ordinal * 30.0, 520.0, 112.0 + ordinal * 30.0]
    anchor_ref = {
        "source": "candidate_b_account_anchor",
        "logical_page": logical_page,
        "source_page": source_page,
        "bbox": bbox,
        "evidence_ids": [evidence_id],
    }
    heading_ref = {
        **anchor_ref,
        "field_name": "credit_agreement_identifier",
        "binding": "canonical_account_anchor",
        "binding_quality": "canonical_account_anchor",
    }
    account = {
        "account_id": f"credit_account:credit_card:{ordinal}",
        "account_type": "credit_card",
        "category_sequence": ordinal,
        "account_family_quality": "exact",
        "_printed_ordinal_status": "printed_unique",
        "credit_agreement_identifier": identifier,
        "management_institution": institution,
        "business_type": "贷记卡",
        "account_currency": currency,
        "credit_limit": limit,
        "source_refs": [anchor_ref],
        "source_refs_by_field": {
            "credit_agreement_identifier": [heading_ref],
            "management_institution": [
                _field_ref(
                    "management_institution",
                    ordinal=ordinal,
                    offset=0.0,
                    logical_page=logical_page,
                    source_page=source_page,
                )
            ],
            "business_type": [
                _field_ref(
                    "business_type",
                    ordinal=ordinal,
                    offset=40.0,
                    logical_page=logical_page,
                    source_page=source_page,
                )
            ],
            "account_currency": [
                _field_ref(
                    "account_currency",
                    ordinal=ordinal,
                    offset=80.0,
                    logical_page=logical_page,
                    source_page=source_page,
                )
            ],
            "credit_limit": [
                _field_ref(
                    "credit_limit",
                    ordinal=ordinal,
                    offset=120.0,
                    logical_page=logical_page,
                    source_page=source_page,
                )
            ],
        },
    }
    corrected = f"账户{ordinal}（授信协议标识：{identifier}）（卡片尾号：3906）"
    line = {
        "text": corrected,
        "ocr_original_text": f"账户{ordinal}(授信协议标识:{identifier})(卡片尾号:3906)",
        "bbox": bbox,
        "evidence_ids": [evidence_id],
    }
    return account, line


def _corroboration_context(
    accounts: list[dict],
    lines: list[dict],
) -> SimpleNamespace:
    return _context(
        _table(rows=_rows()),
        accounts=accounts,
        evidence_pages=[{"page": 20, "source_page": 10, "lines": lines}],
    )


def _sealed_census() -> dict:
    return {
        "sequences": [1],
        "ordinal_observations": {
            1: {
                "sequence": 1,
                "source_refs": [
                    {
                        "source": "candidate_b_source_coverage_ledger",
                        "logical_page": 12,
                        "source_page": 6,
                        "geometry_scope": "line",
                        "binding": "printed_credit_agreement_ordinal",
                        "binding_quality": "printed_credit_agreement_ordinal",
                        "sequence": 1,
                        "bbox": [20.0, 78.0, 120.0, 98.0],
                        "evidence_ids": ["ocr:agreement:heading:1"],
                    }
                ],
            }
        },
        "source_refs": [],
    }


def _install_census(monkeypatch: pytest.MonkeyPatch, census: dict | None) -> None:
    monkeypatch.setattr(
        native_extraction,
        "_sealed_agreement_population_census",
        lambda _context: census,
    )


def test_sealed_exact_card_materializes_field_local_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_table())
    _install_census(monkeypatch, _sealed_census())

    extracted = native_extraction._extract_credit_lines(context)
    rows = native_extraction.reconcile_candidate_b_credit_lines(context, extracted)

    assert len(rows) == 1
    row = rows[0]
    assert row["sequence"] == 1
    assert row["account_identifier"] == _IDENTIFIER
    assert row["institution"] == "中国光大银行股份有限公司"
    assert row["facility_type"] == "信用卡共享额度"
    assert row["effective_date"] == "2019-12-01"
    assert row["validity_type"] == "perpetual"
    assert row["total_limit"] == 0
    assert row["used_limit"] == 0
    assert row["currency"] == "CNY"
    assert row["canonical_raw"]["institution"] == "中国光大银行股份有限公司 授信额度"
    assert row["canonical_raw"]["due_date"] == "长期 杂"
    assert row["canonical_raw"]["total_limit"] == "0"
    for field_name in (
        "account_identifier",
        "institution",
        "facility_type",
        "effective_date",
        "due_date",
        "total_limit",
        "used_limit",
        "currency",
    ):
        refs = row["source_refs_by_field"][field_name]
        assert len(refs) == 1
        assert refs[0]["geometry_scope"] == "cell"
        assert refs[0]["evidence_ids"]

    resolved = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("status") == "resolved"
        and issue.get("target_record_id") == row["credit_line_id"]
    ]
    assert {
        (issue["issue_code"], issue.get("field_name")) for issue in resolved
    } >= {
        ("candidate_b_credit_agreement_lower_label_drift_recovered", "institution"),
        ("candidate_b_credit_agreement_lower_label_drift_recovered", "total_limit"),
        ("candidate_b_credit_agreement_perpetual_residue", "due_date"),
    }
    assert all(
        ref.get("geometry_scope") == "cell"
        for issue in resolved
        for ref in issue.get("source_refs") or ()
    )


def test_filtered_canonical_plane_uses_exact_frozen_table_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _table()
    table.metadata["geometry"] = {
        "cell_bboxes": table.metadata.pop("cell_bboxes"),
        "cell_evidence_ids": table.metadata.pop("cell_evidence_ids"),
    }
    context = _context(frozen_tables=[table])
    _install_census(monkeypatch, _sealed_census())

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["sequence"] == 1
    assert rows[0]["account_identifier"] == _IDENTIFIER
    assert rows[0]["institution"] == "中国光大银行股份有限公司"
    assert rows[0]["total_limit"] == 0


def test_partial_parser_owner_does_not_veto_exact_card_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_table())
    _install_census(monkeypatch, _sealed_census())
    complete = native_extraction._sealed_exact_agreement_table_candidates(context)[0]
    partial_fields = dict(complete.fields)
    partial_fields.pop("管理机构")
    partial_fields.pop("授信额度用途")
    partial_fields.pop("授信额度")
    partial = SimpleNamespace(
        **{
            **vars(complete),
            "fields": partial_fields,
        }
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset_name: [partial] if dataset_name == "credit_lines" else [],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["sequence"] == 1
    assert rows[0]["institution"] == "中国光大银行股份有限公司"
    assert rows[0]["facility_type"] == "信用卡共享额度"
    assert rows[0]["total_limit"] == 0


def test_partial_parser_owner_cannot_fill_sealed_damaged_required_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    damaged_value = "信用卡共享额度 庭"
    damaged_rows = [
        ["管理机构", "授信协议标识", "生效日期", "到期日期", "授信额度庭用途"],
        ["中国光大银行股份有限公司", _IDENTIFIER, "2019.12.01", "长期", damaged_value],
        ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
        ["0", "--", "--", "0", "人民币元"],
    ]
    context = _context(_table(rows=damaged_rows))
    _install_census(monkeypatch, _sealed_census())
    sealed = native_extraction._sealed_exact_agreement_table_candidates(context)[0]
    partial_fields = dict(sealed.fields)
    partial_fields["授信额度用途"] = "信用卡共享额度"
    partial_bindings = dict(sealed.binding_quality_by_field)
    partial_bindings["授信额度用途"] = "native_label_column"
    partial = SimpleNamespace(
        **{
            **vars(sealed),
            "fields": partial_fields,
            "binding_quality_by_field": partial_bindings,
            "unresolved_labels": frozenset(),
        }
    )
    monkeypatch.setattr(
        PBOCPersonalDetailNativeParser,
        "records",
        lambda _parser, dataset_name: [partial] if dataset_name == "credit_lines" else [],
    )

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["facility_type"] is None
    assert row["canonical_raw"]["facility_type"] == damaged_value
    assert row["source_refs_by_field"]["facility_type"][0]["row"] == 1
    issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("status") != "resolved"
        and issue.get("issue_code")
        == "candidate_b_credit_agreement_required_field_unresolved"
        and issue.get("target_record_id") == row["credit_line_id"]
        and issue.get("field_name") == "facility_type"
    ]
    assert len(issues) == 1
    assert issues[0]["source_refs"] == row["source_refs_by_field"]["facility_type"]


def test_candidate_b_two_pass_keeps_frozen_exact_agreements_filtered_from_canonical_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production Candidate-B repair boundary and source projection."""

    from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b import (
        CandidateBPipeline,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        project_personal_detail_datasets,
    )
    from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
        prepare_personal_detail_source_collections as real_prepare_source_collections,
    )

    institutions = (
        "交通银行股份有限公司",
        "中国光大银行股份有限公司",
        "中国工商银行股份有限公司",
        "中国农业银行股份有限公司",
        "中国建设银行股份有限公司",
        "招商银行股份有限公司",
        "平安银行股份有限公司",
        "兴业银行股份有限公司",
    )
    identifiers = [
        f"B{sequence:02d}12345678901234567890123456789012345"
        for sequence in range(1, 9)
    ]
    identifiers[1] = _IDENTIFIER
    damaged_facility_header = "授信额度庭用途"
    damaged_facility_value = "信用卡共享额度 庭"
    frozen_tables: list[SimpleNamespace] = []
    frozen_texts: list[SimpleNamespace] = [
        SimpleNamespace(
            content="授信协议信息",
            bbox=[20.0, 10.0, 160.0, 30.0],
            evidence_ids=["ocr:agreement:section"],
        )
    ]
    for sequence, (institution, identifier) in enumerate(
        zip(institutions, identifiers, strict=True),
        start=1,
    ):
        top = 100.0 + (sequence - 1) * 100.0
        agreement_rows = _rows(identifier=identifier, institution=institution)
        if sequence == 3:
            agreement_rows = [
                [
                    "管理机构",
                    "授信协议标识",
                    "生效日期",
                    "到期日期",
                    damaged_facility_header,
                ],
                [
                    institution,
                    identifier,
                    "2017.07.31",
                    "长期",
                    damaged_facility_value,
                ],
                ["授信额度", "授信限额", "授信限额编号", "已用额度", "币种"],
                ["0", "--", "--", "0", "人民币元"],
            ]
        frozen_tables.append(
            _table(
                table_id=f"agreement-source-{sequence}",
                rows=agreement_rows,
                top=top,
            )
        )
        frozen_texts.append(
            SimpleNamespace(
                content=f"授信协议{sequence}",
                bbox=[20.0, top - 22.0, 160.0, top - 2.0],
                evidence_ids=[f"ocr:agreement:heading:{sequence}"],
            )
        )
    frozen_texts.append(
        SimpleNamespace(
            content="公共信息明细",
            bbox=[20.0, 900.0, 160.0, 920.0],
            evidence_ids=["ocr:agreement:boundary"],
        )
    )
    frozen_page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        tables=frozen_tables,
        texts=frozen_texts,
    )

    # This is the actual failure shape: mixed-page canonical ownership retained
    # only two ordinary parser tables even though the immutable source plane
    # contains every numbered agreement card.
    canonical_tables = [deepcopy(frozen_tables[index]) for index in (0, 4)]
    for table in canonical_tables:
        table.metadata["canonical_template_id"] = "credit_agreement"
    canonical_page = SimpleNamespace(
        page_number=1,
        source_page_number=1,
        canonical_template_id="credit_agreement",
        tables=canonical_tables,
        texts=deepcopy(frozen_texts),
    )

    account8, line8 = _account_witness(
        8,
        identifier=_YU_CORROBORATED_IDENTIFIER,
        anchor_bbox=[20.0, 940.0, 520.0, 952.0],
        logical_page=1,
        source_page=1,
    )
    account9, line9 = _account_witness(
        9,
        identifier=_YU_CORROBORATED_IDENTIFIER,
        currency="USD",
        anchor_bbox=[20.0, 970.0, 520.0, 982.0],
        logical_page=1,
        source_page=1,
    )

    class StaticTwoPassContext:
        def __init__(self) -> None:
            self._frozen_logical_pages = {1: frozen_page}
            self.reading_order_by_logical = {1: 1}
            self.reading_order_resolution = {
                "resolved": True,
                "authoritative": True,
            }
            self._personal_detail_extraction_issues: list[dict] = []
            self._business_repair_active = False
            self.tables_continue = lambda _left, _right: None

        @property
        def pages(self) -> list[SimpleNamespace]:
            return [canonical_page]

        def account_collections(self):
            return deepcopy([account8, account9]), [], []

        def corrected_repayment_records(self):
            return []

        def corrected_repayment_micro_grids(self):
            return []

        def corrected_evidence_pages(self):
            return [{"page": 1, "source_page": 1, "lines": deepcopy([line8, line9])}]

        def candidate_b_status_glyph_observations(self):
            return []

        def prepare_candidate_b_business_repair(self, _payload):
            self._business_repair_active = True
            self._personal_detail_extraction_issues = []
            return True

        def correct_candidate_b_datasets(self, payload):
            corrected = deepcopy(payload)
            sequence_three = next(
                row
                for row in corrected.get("credit_lines") or ()
                if row.get("sequence") == 3
            )
            # A later partial plane must not make the unresolved source slot
            # disappear merely by populating a plausible controlled value.
            sequence_three["facility_type"] = "信用卡共享额度"
            return corrected

        def ocr_correction_audit(self):
            table = frozen_tables[2]
            return {
                "business_repair": {"second_schema_pass_required": True},
                "cell_anomalies": [
                    {
                        "normalized_value_withheld": True,
                        "stage": "candidate_b_final_validation",
                        "dataset_name": "credit_lines",
                        "record_id": "credit_agreement:3",
                        "field_name": "facility_type",
                        "role": "facility_type",
                        "value": damaged_facility_value,
                        "source_refs": [
                            {
                                "source": "native_detail_tolerant_table_cell",
                                "logical_page": 1,
                                "source_page": 1,
                                "table_id": table.table_id,
                                "row": 1,
                                "column": 4,
                                "geometry_scope": "cell",
                                "field_name": "facility_type",
                                "binding": "label_column",
                                "bbox": table.metadata["cell_bboxes"][1][4],
                                "evidence_ids": table.metadata["cell_evidence_ids"][1][4],
                            }
                        ],
                        "reason_codes": [
                            "registered_facility_type_contract_failed",
                            "normalized_value_withheld",
                        ],
                    }
                ],
            }

        def canonical_layout_audit(self):
            return {}

        def page_topology_audit(self):
            return {}

    context = StaticTwoPassContext()
    census = native_extraction._sealed_agreement_population_census(context)
    assert (census or {}).get("sequences") == list(range(1, 9))
    assert [table.table_id for table in context.pages[0].tables] == [
        "agreement-source-1",
        "agreement-source-5",
    ]

    def empty(_context):
        return []

    for name in (
        "_extract_employment_records",
        "_extract_inquiries",
        "_extract_liabilities",
        "_extract_postpaid_payment_history",
        "_extract_postpaid_records",
        "_extract_public_records",
        "_extract_recovery_records",
        "_extract_residence_records",
        "_extract_source_rows",
    ):
        monkeypatch.setattr(native_extraction, name, empty)
    monkeypatch.setattr(
        native_extraction,
        "_extract_header_datasets",
        lambda _context, _text: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_personal_notes",
        lambda _context: ([], []),
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_profile_detail_records",
        lambda _context: {},
    )
    monkeypatch.setattr(
        native_extraction,
        "_extract_summary_datasets",
        lambda _context: ([], []),
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.profile_extraction.extract_candidate_b_profile",
        lambda _context: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.candidate_b_repayment_anchor_ledger",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.link_candidate_b_repayments",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.relations.derive_candidate_b_overdue_records",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.consistency_ledger.apply_document_consistency_ledger",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.document_glyph_bank.apply_document_local_status_glyph_bank",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict.apply_candidate_b_native_status_conflict_guard",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues.register_final_liability_issue_records",
        lambda *_args, **_kwargs: None,
    )

    projected: dict = {}

    def capture_projection(content, business, **kwargs):
        result = real_prepare_source_collections(content, business, **kwargs)
        projected.update(deepcopy(result))
        return result

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.source_projection.prepare_personal_detail_source_collections",
        capture_projection,
    )

    result = CandidateBPipeline(context, "").run()

    rows = result.business["credit_lines"]
    assert [row["sequence"] for row in rows] == list(range(1, 9))
    assert rows[1]["account_identifier"] == _YU_CORROBORATED_IDENTIFIER
    assert rows[1]["canonical_raw"]["account_identifier"] == _IDENTIFIER
    sequence_three = rows[2]
    assert sequence_three["facility_type"] is None
    assert sequence_three["canonical_raw"]["facility_type"] == damaged_facility_value
    assert "facility_type" in sequence_three["_unresolved_fields"]
    facility_refs = sequence_three["source_refs_by_field"]["facility_type"]
    assert len(facility_refs) == 1
    assert facility_refs[0]["table_id"] == frozen_tables[2].table_id
    assert facility_refs[0]["row"] == 1
    assert facility_refs[0]["column"] == 4
    assert facility_refs[0]["evidence_ids"]
    assert [row["sequence"] for row in projected["datasets"]["credit_lines"]] == list(
        range(1, 9)
    )
    active_facility_issues = [
        issue
        for issue in projected["datasets"].get(
            "personal_detail_extraction_issues",
            (),
        )
        if issue.get("status") != "resolved"
        and issue.get("target_dataset") == "credit_lines"
        and issue.get("target_record_id") == sequence_three["credit_line_id"]
        and issue.get("field_name") == "facility_type"
    ]
    assert len(active_facility_issues) == 1
    assert active_facility_issues[0]["issue_code"] == "pboc_cell_contract_unresolved"
    assert active_facility_issues[0]["observed_value"] == damaged_facility_value
    assert active_facility_issues[0]["source_refs"]
    canonical_projection = project_personal_detail_datasets(projected["datasets"])
    canonical_sequence_three = next(
        row
        for row in canonical_projection["credit_agreements"]
        if row.get("sequence") == 3
    )
    assert canonical_sequence_three["credit_agreement_id"] == sequence_three["credit_line_id"]
    assert canonical_sequence_three["facility_type"] is None
    canonical_facility_issues = [
        issue
        for issue in canonical_projection["extraction_issues"]
        if issue.get("status") != "resolved"
        and issue.get("target_dataset") == "credit_agreements"
        and issue.get("target_record_id") == sequence_three["credit_line_id"]
        and issue.get("field_name") == "facility_type"
    ]
    assert len(canonical_facility_issues) == 1
    active_omissions = [
        issue
        for issue in projected["datasets"].get(
            "personal_detail_extraction_issues",
            (),
        )
        if issue.get("issue_code") == "source_credit_agreement_record_omitted"
        and issue.get("status") != "resolved"
    ]
    assert active_omissions == []


def test_two_exact_account_headings_correct_one_batch_unique_agreement_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account8, line8 = _account_witness(8)
    account9, line9 = _account_witness(9, currency="USD")
    context = _corroboration_context([account8, account9], [line8, line9])
    _install_census(monkeypatch, _sealed_census())

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["account_identifier"] == _CORROBORATED_IDENTIFIER
    assert row["canonical_raw"]["account_identifier"] == _IDENTIFIER
    assert {
        ref["account_id"]
        for ref in row["source_refs_by_field"]["account_identifier"]
    } == {
        "credit_account:credit_card:8",
        "credit_account:credit_card:9",
    }
    assert all(
        ref["binding"] == "printed_account_anchor_heading"
        for ref in row["source_refs_by_field"]["account_identifier"]
    )
    local_refs = [
        ref
        for ref in row["source_refs"]
        if ref.get("field_name") == "account_identifier"
        and ref.get("geometry_scope") == "cell"
    ]
    assert len(local_refs) == 1
    resolved = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_credit_agreement_identifier_exact_account_corroborated"
    ]
    assert len(resolved) == 1
    assert resolved[0]["status"] == "resolved"
    assert resolved[0]["target_record_id"] == row["credit_line_id"]
    assert resolved[0]["observed_value"] == {
        "agreement_cell_identifier": _IDENTIFIER,
        "account_heading_identifier": _CORROBORATED_IDENTIFIER,
        "account_ids": [
            "credit_account:credit_card:8",
            "credit_account:credit_card:9",
        ],
    }

    from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
        _append_exact_agreement_population_issues,
        _emitted_agreement_ordinals,
    )

    ledger = {
        "credit_agreement_sequence_endpoint": 1,
        "credit_agreement_observed_sequences": [1],
        "credit_agreement_sequence_outliers": [],
        "credit_agreement_ordinal_observations": {
            "1": _sealed_census()["ordinal_observations"][1]
        },
    }
    datasets = {"credit_lines": rows}
    assert _emitted_agreement_ordinals(ledger, datasets) == {1}
    projection_issues: list[dict] = []
    _append_exact_agreement_population_issues(
        ledger,
        datasets,
        projection_issues,
        set(),
    )
    assert not any(
        issue.get("issue_code") == "source_credit_agreement_record_omitted"
        for issue in projection_issues
    )


def test_one_account_heading_cannot_materially_correct_agreement_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account8, line8 = _account_witness(8)
    context = _corroboration_context([account8], [line8])
    _install_census(monkeypatch, _sealed_census())

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["account_identifier"] == _IDENTIFIER


def test_replayed_account_heading_does_not_count_as_second_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_bbox = [20.0, 340.0, 520.0, 352.0]
    account8, line8 = _account_witness(
        8,
        anchor_evidence="ocr:account:replayed-anchor",
        anchor_bbox=shared_bbox,
    )
    account9, line9 = _account_witness(
        9,
        anchor_evidence="ocr:account:replayed-anchor",
        anchor_bbox=shared_bbox,
    )
    context = _corroboration_context([account8, account9], [line8, line9])
    _install_census(monkeypatch, _sealed_census())

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["account_identifier"] == _IDENTIFIER


def test_mismatched_exact_business_tuple_blocks_cross_section_id_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account8, line8 = _account_witness(
        8,
        institution="其他银行股份有限公司",
    )
    account9, line9 = _account_witness(
        9,
        institution="其他银行股份有限公司",
    )
    context = _corroboration_context([account8, account9], [line8, line9])
    _install_census(monkeypatch, _sealed_census())

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["account_identifier"] == _IDENTIFIER


def test_one_witness_competing_id_makes_account_join_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account8, line8 = _account_witness(8)
    account9, line9 = _account_witness(9, currency="USD")
    competing_id = "D9876543210987654321098765432109876543"
    account10, line10 = _account_witness(10, identifier=competing_id)
    context = _corroboration_context(
        [account8, account9, account10],
        [line8, line9, line10],
    )
    _install_census(monkeypatch, _sealed_census())

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        native_extraction._extract_credit_lines(context),
    )

    assert len(rows) == 1
    assert rows[0]["account_identifier"] == _IDENTIFIER


def test_two_agreement_cards_with_same_exact_tuple_block_batch_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account8, line8 = _account_witness(8)
    account9, line9 = _account_witness(9, currency="USD")
    context = _corroboration_context([account8, account9], [line8, line9])
    _install_census(monkeypatch, _sealed_census())
    base = native_extraction._extract_credit_lines(context)[0]
    second = deepcopy(base)
    second_identifier = "E1234567890123456789012345678901234567"
    second["account_identifier"] = second_identifier
    second["credit_line_id"] = native_extraction.stable_record_id(
        "credit_line",
        second_identifier,
    )
    second["_printed_sequence"] = 2
    second["_canonical_card_key"] = "credit_agreement:2"
    delta = 200.0
    for ref in second["source_refs"]:
        ref["table_id"] = "agreement-second-owner"
        ref["bbox"] = [
            ref["bbox"][0],
            ref["bbox"][1] + delta,
            ref["bbox"][2],
            ref["bbox"][3] + delta,
        ]
    for ref in second["_canonical_card_anchor_refs"]:
        ref["sequence"] = 2
        ref["bbox"] = [
            ref["bbox"][0],
            ref["bbox"][1] + delta,
            ref["bbox"][2],
            ref["bbox"][3] + delta,
        ]
    for refs in second["source_refs_by_field"].values():
        for ref in refs:
            ref["table_id"] = "agreement-second-owner"
            ref["bbox"] = [
                ref["bbox"][0],
                ref["bbox"][1] + delta,
                ref["bbox"][2],
                ref["bbox"][3] + delta,
            ]
    two_card_census = deepcopy(_sealed_census())
    two_card_census["sequences"] = [1, 2]
    two_card_census["ordinal_observations"][2] = deepcopy(
        two_card_census["ordinal_observations"][1]
    )
    two_card_census["ordinal_observations"][2]["sequence"] = 2
    two_card_census["ordinal_observations"][2]["source_refs"][0]["sequence"] = 2
    _install_census(monkeypatch, two_card_census)

    rows = native_extraction.reconcile_candidate_b_credit_lines(
        context,
        [base, second],
    )

    assert len(rows) == 2
    assert _CORROBORATED_IDENTIFIER not in {
        row["account_identifier"] for row in rows
    }


def test_unsealed_agreement_heading_never_owns_table_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_table())
    _install_census(monkeypatch, None)

    assert native_extraction._extract_credit_lines(context) == []


def test_one_heading_cannot_own_two_candidate_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        _table(table_id="agreement-left"),
        _table(table_id="agreement-right"),
    )
    _install_census(monkeypatch, _sealed_census())

    assert native_extraction._extract_credit_lines(context) == []


def test_nonempty_residue_row_vetoes_exact_card_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    rows.append(["cross-section", "", "", "", ""])
    context = _context(_table(rows=rows))
    _install_census(monkeypatch, _sealed_census())

    assert native_extraction._extract_credit_lines(context) == []


def test_identifier_without_exact_cell_evidence_withholds_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_table(missing_evidence=(1, 1)))
    _install_census(monkeypatch, _sealed_census())

    assert native_extraction._extract_credit_lines(context) == []


def test_field_cell_outside_owner_table_withholds_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _table()
    table.metadata["cell_bboxes"][1][1] = [620.0, 120.0, 700.0, 140.0]
    context = _context(table)
    _install_census(monkeypatch, _sealed_census())

    assert native_extraction._extract_credit_lines(context) == []


@pytest.mark.parametrize(
    "value",
    ["长期", "长期不详", "长期 2025", "2025.01.01 长期"],
)
def test_perpetual_residue_requires_one_finite_token_and_one_residue_glyph(
    value: str,
) -> None:
    assert native_extraction._agreement_unique_perpetual_token(value) is None
