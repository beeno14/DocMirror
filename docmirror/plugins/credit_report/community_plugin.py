# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Credit report community domain plugin.

Premium community plugin for personal brief, personal detail, and enterprise
credit reports. Extracts identity fields, report subtype/content mode, optional
lightweight section hints, and table records via shared KV extract helpers.

Pipeline role: post-seal domain derivation and Community JSON projection.

Key exports: ``CreditReportPlugin``, ``plugin``.

Dependencies: ``ProjectionData`` and the credit-report projection orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from docmirror.output.community_bundle import CommunityBundle
from docmirror.plugins._base.projector import CommunityProjector, ProjectionData

_PERSONAL_DETAIL_SOURCE_COMPLETE = frozenset(
    {"observed_nonempty", "explicitly_empty", "not_applicable"}
)
_PERSONAL_DETAIL_CONTROL_DATASETS = frozenset(
    {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
)
_PERSONAL_DETAIL_SPARSE_STATUS_SEMANTICS = {
    "mode": "potentially_flawed_only",
    "present_dataset_without_status": "silently_trusted_complete",
    "absent_dataset_without_status": "silently_trusted_empty_or_not_applicable",
    "status_row_present": "partial_unknown_or_failed_extraction",
}


def _merge_warning_page_range(
    warning: dict[str, Any],
    pages: Sequence[int],
) -> None:
    """Conserve all cited pages when compact findings share one warning row."""

    positive_pages = [
        int(page)
        for page in [*(warning.get("page_range") or []), *pages]
        if isinstance(page, int) and page > 0
    ]
    if positive_pages:
        warning["page_range"] = [min(positive_pages), max(positive_pages)]


def _personal_detail_source_rows(
    source_datasets: Sequence[Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for dataset in source_datasets:
        public = getattr(dataset, "public", None)
        if not isinstance(public, Mapping):
            continue
        name = str(public.get("name") or "")
        if not name:
            continue
        rows_by_name[name] = [
            row for row in (getattr(dataset, "rows", None) or ()) if isinstance(row, dict)
        ]
    return rows_by_name


def _personal_detail_review_fields(
    datasets: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    affected: dict[tuple[str, str], set[str]] = {}
    by_name = {
        str(dataset.get("name") or ""): dataset
        for dataset in datasets
        if isinstance(dataset, dict)
    }
    issues = by_name.get("extraction_issues", {}).get("rows") or ()
    for wrapper in issues:
        if not isinstance(wrapper, Mapping):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), Mapping) else wrapper
        status = str(values.get("status") or "")
        dataset_name = str(values.get("target_dataset") or "")
        record_id = str(values.get("target_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if (
            status not in {"resolved", "dismissed"}
            and dataset_name
            and record_id
            and field_name
        ):
            affected.setdefault((dataset_name, record_id), set()).add(field_name)
    observations = by_name.get("field_observations", {}).get("rows") or ()
    for wrapper in observations:
        if not isinstance(wrapper, Mapping):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), Mapping) else wrapper
        observation_status = str(values.get("observation_status") or "")
        dataset_name = str(values.get("dataset_name") or "")
        record_id = str(values.get("business_record_id") or "")
        field_name = str(values.get("field_name") or "")
        if (
            observation_status != "ocr_corrected"
            and dataset_name
            and record_id
            and field_name
        ):
            affected.setdefault((dataset_name, record_id), set()).add(field_name)
    return affected


def _review_metadata_requires_attention(review: Any) -> bool:
    """Accept only an explicit open-review contract, not any truthy mapping."""

    if not isinstance(review, Mapping):
        return False
    status = str(review.get("status") or review.get("extraction_status") or "").lower()
    return bool(
        review.get("required") is True
        or status
        in {
            "requires_review",
            "review",
            "unresolved",
            "failed",
            "partial",
            "uncertain",
            "unknown",
        }
        or review.get("reason")
        or review.get("reason_codes")
    )


def _compact_personal_detail_public_projection(
    payload: dict[str, Any],
    *,
    source_datasets: Sequence[Any],
) -> None:
    """Close scanned-detail Community rows over the declared v2 contract.

    The rich semantic result keeps correction/provenance state.  The final
    Community JSON exposes only declared normalized fields and raw evidence for
    fields which still require review; successful normalization stays silent.
    """

    from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
        personal_detail_data_dictionary,
    )

    datasets = [item for item in payload.get("datasets") or () if isinstance(item, dict)]
    dictionary = personal_detail_data_dictionary().get("datasets") or {}
    source_rows_by_name = _personal_detail_source_rows(source_datasets)
    review_fields = _personal_detail_review_fields(datasets)

    for dataset in datasets:
        name = str(dataset.get("name") or "")
        definition = dictionary.get(name) if isinstance(dictionary.get(name), Mapping) else {}
        declared = definition.get("columns") if isinstance(definition.get("columns"), Mapping) else {}
        declared_keys = tuple(str(key) for key in declared)
        allowed = frozenset(declared_keys)
        public_rows = [row for row in dataset.get("rows") or () if isinstance(row, dict)]
        source_rows = source_rows_by_name.get(name, [])
        source_by_id = {
            str(row.get("record_id") or ""): row
            for row in source_rows
            if str(row.get("record_id") or "")
        }
        raw_available: set[str] = set()

        for index, row in enumerate(public_rows):
            normalized = row.get("normalized") if isinstance(row.get("normalized"), Mapping) else {}
            row["normalized"] = {
                key: normalized.get(key)
                for key in declared_keys
            }
            record_id = str(row.get("record_id") or "")
            source_row = source_by_id.get(record_id)
            if source_row is None and not record_id and index < len(source_rows):
                positional_source = source_rows[index]
                if not str(positional_source.get("record_id") or ""):
                    source_row = positional_source
            source_row = source_row if isinstance(source_row, Mapping) else {}
            canonical_source = (
                source_row.get("canonical_raw")
                if isinstance(source_row.get("canonical_raw"), Mapping)
                else {}
            )
            raw_source = (
                source_row.get("raw")
                if isinstance(source_row.get("raw"), Mapping)
                else {}
            )
            affected = set(review_fields.get((name, record_id), ()))
            affected &= allowed
            if name in _PERSONAL_DETAIL_CONTROL_DATASETS:
                affected.clear()
            existing_review = row.get("review")
            keep_review_metadata = bool(
                name in _PERSONAL_DETAIL_CONTROL_DATASETS
                or affected
                or _review_metadata_requires_attention(existing_review)
            )

            public_canonical = row.get("canonical_raw") if isinstance(row.get("canonical_raw"), Mapping) else {}
            public_raw = row.get("raw") if isinstance(row.get("raw"), Mapping) else {}
            canonical_evidence: dict[str, Any] = {}
            raw_evidence: dict[str, Any] = {}
            for key in declared_keys:
                if key not in affected:
                    continue
                if key in canonical_source:
                    canonical_evidence[key] = canonical_source[key]
                elif key in raw_source:
                    canonical_evidence[key] = raw_source[key]
                elif key in public_canonical:
                    canonical_evidence[key] = public_canonical[key]
                if key in raw_source:
                    raw_evidence[key] = raw_source[key]
                elif key in canonical_source:
                    raw_evidence[key] = canonical_source[key]
                elif key in public_raw:
                    raw_evidence[key] = public_raw[key]
            row["canonical_raw"] = canonical_evidence
            row["raw"] = raw_evidence
            if not keep_review_metadata:
                # Community's record envelope requires a source object, but a
                # successfully trusted business row does not need to publish
                # page/provenance diagnostics.  The rich semantic payload is
                # untouched and still carries that evidence for internal use.
                row["source"] = {}
                row.pop("confidence", None)
                row.pop("review", None)
            raw_available.update(
                key
                for key in affected
                if canonical_evidence.get(key) not in (None, "")
                or raw_evidence.get(key) not in (None, "")
            )

        columns = [
            column
            for column in dataset.get("columns") or ()
            if isinstance(column, dict) and str(column.get("key") or "") in allowed
        ]
        for column in columns:
            column["raw_available"] = str(column.get("key") or "") in raw_available
        dataset["columns"] = columns
        if isinstance(dataset.get("reading_columns"), list):
            dataset["reading_columns"] = [
                key for key in dataset["reading_columns"] if str(key) in allowed
            ]

    status_dataset = next(
        (dataset for dataset in datasets if str(dataset.get("name") or "") == "dataset_status"),
        None,
    )
    if status_dataset is not None:
        status_dataset["sparse_status_semantics"] = dict(
            _PERSONAL_DETAIL_SPARSE_STATUS_SEMANTICS
        )


def _apply_personal_detail_dataset_status(payload: dict[str, Any]) -> None:
    """Make public dataset envelopes agree with the v2 source-status ledger.

    Community's ordinary completeness calculation proves only that projected
    rows were conserved during serialization.  It must not turn a source-
    partial scanned dataset into ``complete`` merely because all rows that
    reached the projector were written successfully.
    """

    datasets = [item for item in payload.get("datasets") or () if isinstance(item, dict)]
    status_dataset = next(
        (item for item in datasets if str(item.get("name") or "") == "dataset_status"),
        None,
    )
    if status_dataset is None:
        return
    status_by_name: dict[str, dict[str, Any]] = {}
    for wrapper in status_dataset.get("rows") or ():
        if not isinstance(wrapper, dict):
            continue
        values = wrapper.get("normalized") if isinstance(wrapper.get("normalized"), dict) else wrapper
        name = str(values.get("dataset_name") or "")
        if name:
            status_by_name[name] = values

    for dataset in datasets:
        name = str(dataset.get("name") or "")
        control = status_by_name.get(name)
        if control is None:
            continue
        presence = str(control.get("presence_status") or "unknown")
        source_complete = presence in _PERSONAL_DETAIL_SOURCE_COMPLETE
        dataset["status"] = "complete" if source_complete else "partial"
        emitted = int(dataset.get("row_count") or len(dataset.get("rows") or ()))
        expected_raw = control.get("expected_row_count")
        expected = (
            int(expected_raw)
            if isinstance(expected_raw, (int, float)) and not isinstance(expected_raw, bool)
            else None
        )
        completeness = dict(dataset.get("completeness") or {})
        count_conflict = expected is not None and expected < emitted
        if expected is None:
            # The sparse status ledger intentionally omits a source expected
            # count when it only knows that extraction was partial/failed.
            # Keep Community's integer row-conservation count in that case;
            # ``verified`` and ``basis`` carry the source-completeness truth.
            projected_expected = completeness.get("expected_row_count")
            expected = (
                max(int(projected_expected), emitted)
                if isinstance(projected_expected, (int, float))
                and not isinstance(projected_expected, bool)
                else emitted
            )
        elif count_conflict:
            # The source ledger remains available in dataset_status, while the
            # public envelope must obey expected >= emitted.  More emitted
            # rows than the source expected is itself an unresolved population
            # conflict, never a verified complete dataset.
            expected = emitted
            source_complete = False
            dataset["status"] = "partial"
        completeness.update(
            {
                "expected_row_count": expected,
                "emitted_row_count": emitted,
                "omitted_row_count": max(expected - emitted, 0),
                "verified": bool(source_complete),
                "basis": (
                    f"personal_detail_dataset_status:{presence}:expected_less_than_emitted"
                    if count_conflict
                    else f"personal_detail_dataset_status:{presence}"
                ),
            }
        )
        dataset["completeness"] = completeness


class _CreditReportCommunityBundle(CommunityBundle):
    """Publish compact Community JSON without weakening rich semantic bindings."""

    @staticmethod
    def _uses_scanned_personal_detail_public_projection(facts: Mapping[str, Any]) -> bool:
        """Match the stable facts emitted by the personal-detail variant router."""

        return bool(
            facts.get("report_subtype") == "personal_detail"
            and facts.get("content_mode") in {"scanned_ocr", "mixed"}
        )

    @staticmethod
    def _is_enterprise_semantic(payload: dict[str, Any]) -> bool:
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return facts.get("report_subtype") == "enterprise"

    @staticmethod
    def _is_personal_brief_semantic(payload: dict[str, Any]) -> bool:
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return facts.get("report_subtype") == "personal_brief"

    def semantic_payload(self) -> dict[str, Any]:
        payload = super().semantic_payload()
        domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        if facts.get("report_subtype") != "enterprise":
            return payload
        extraction = facts.pop("extraction_report", None)
        if isinstance(extraction, dict):
            payload["extraction"] = extraction
        audit = facts.pop("audit_report", None)
        if isinstance(audit, dict):
            payload["audit"] = audit
        for key in tuple(facts):
            if key.startswith("enterprise_expected_"):
                facts.pop(key, None)
        extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
        extensions.pop("enterprise_dataset_completeness", None)
        overrides = (
            extensions.get("community_projection_overrides")
            if isinstance(extensions.get("community_projection_overrides"), dict)
            else {}
        )
        for key in ("internal_fields", "internal_facts"):
            values = overrides.get(key)
            if not isinstance(values, list):
                continue
            overrides[key] = [
                value for value in values if not str(value).startswith("enterprise_expected_")
            ]
        return payload

    def json_payload(self, semantic: dict[str, Any] | None = None) -> dict[str, Any]:
        from docmirror.plugins.credit_report.projection import _compact_public_datasets

        semantic_payload = semantic or self.semantic_payload()
        payload = super().json_payload(semantic_payload)
        domain = (
            semantic_payload.get("domain")
            if isinstance(semantic_payload.get("domain"), dict)
            else {}
        )
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        enterprise = facts.get("report_subtype") == "enterprise"
        personal_brief = facts.get("report_subtype") == "personal_brief"
        scanned_personal_detail = self._uses_scanned_personal_detail_public_projection(facts)
        extensions = domain.get("extensions") if isinstance(domain.get("extensions"), dict) else {}
        enterprise_completeness = (
            extensions.get("enterprise_dataset_completeness", {}) if enterprise else {}
        )
        for dataset in payload.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            dataset_id = str(dataset.get("id") or dataset.get("name") or "dataset")
            records = [record for record in (dataset.get("rows") or []) if isinstance(record, dict)]
            dataset["rows"] = _compact_public_datasets({dataset_id: records})[dataset_id]
            if not enterprise:
                continue
            details = enterprise_completeness.get(str(dataset.get("name") or ""))
            if not isinstance(details, dict):
                continue
            dataset["completeness"] = {
                key: details[key]
                for key in (
                    "expected_row_count",
                    "emitted_row_count",
                    "omitted_row_count",
                    "verified",
                    "basis",
                )
                if key in details
            }
        if enterprise:
            from docmirror.plugins.credit_report.enterprise_native.projector import (
                project_enterprise_community_json,
            )

            payload = project_enterprise_community_json(payload)
            audit = (
                semantic_payload.get("audit")
                if isinstance(semantic_payload.get("audit"), dict)
                else {}
            )
            datasets_by_name = {
                str(dataset.get("name") or ""): str(dataset.get("id") or "")
                for dataset in payload.get("datasets") or []
                if isinstance(dataset, dict)
            }
            warning_rows = [
                warning
                for warning in (payload.get("warnings") or [])
                if isinstance(warning, dict)
            ]
            for finding in audit.get("findings") or []:
                if not isinstance(finding, dict) or finding.get("severity") not in {
                    "warning",
                    "error",
                }:
                    continue
                code = str(finding.get("code") or "ENTERPRISE_AUDIT_REVIEW")
                path = str(finding.get("path") or "")
                message = (
                    f"{code}: {str(finding.get('message') or '').strip()}"
                    f"{' Review ' + path + '.' if path else ''}"
                )
                warning = next(
                    (
                        row
                        for row in warning_rows
                        if row.get("code") == code and row.get("message") == message
                    ),
                    None,
                )
                if warning is None:
                    warning = {
                        "code": code,
                        "level": (
                            "error" if finding.get("severity") == "error" else "warning"
                        ),
                        "message": message,
                    }
                    warning_rows.append(warning)
                dataset_id = datasets_by_name.get(str(finding.get("dataset") or ""))
                if dataset_id:
                    warning["dataset_id"] = dataset_id
                pages = sorted(
                    {
                        int(page)
                        for page in (finding.get("source_pages") or [])
                        if str(page).isdigit() and int(page) > 0
                    }
                )
                if pages:
                    _merge_warning_page_range(warning, pages)
            payload["warnings"] = warning_rows
        elif personal_brief:
            from docmirror.plugins.credit_report.personal_brief_native.audit import (
                append_personal_brief_observational_warnings,
            )
            from docmirror.plugins.credit_report.personal_brief_native.projector import (
                project_personal_brief_community_json,
            )

            payload = project_personal_brief_community_json(payload)
            payload = append_personal_brief_observational_warnings(
                semantic_payload,
                payload,
            )
        elif scanned_personal_detail:
            _compact_personal_detail_public_projection(
                payload,
                source_datasets=self.datasets,
            )
            _apply_personal_detail_dataset_status(payload)
        return payload

    @staticmethod
    def _is_scanned_personal_detail_semantic(semantic: dict[str, Any]) -> bool:
        domain = semantic.get("domain") if isinstance(semantic.get("domain"), dict) else {}
        facts = domain.get("facts") if isinstance(domain.get("facts"), dict) else {}
        return bool(
            facts.get("report_subtype") == "personal_detail"
            and facts.get("content_mode") in {"scanned_ocr", "mixed"}
        )

    def render_markdown(self) -> str:
        semantic = self.semantic_payload()
        if not self._is_scanned_personal_detail_semantic(semantic):
            return super().render_markdown()
        from docmirror.plugins.credit_report.personal_detail_scanned.markdown import (
            render_personal_detail_business_markdown,
        )

        return render_personal_detail_business_markdown(semantic)

    def render_enhanced_markdown(self, semantic: dict[str, Any] | None = None) -> str:
        semantic_payload = semantic or self.semantic_payload()
        if not self._is_scanned_personal_detail_semantic(semantic_payload):
            return super().render_enhanced_markdown(semantic_payload)
        from docmirror.plugins.credit_report.personal_detail_scanned.markdown import (
            render_personal_detail_business_markdown,
        )

        return render_personal_detail_business_markdown(semantic_payload)

    def _enterprise_artifact_semantic(
        self,
        semantic: dict[str, Any],
    ) -> dict[str, Any]:
        from docmirror.plugins.credit_report.enterprise_native.projector import (
            project_enterprise_artifact_semantic,
        )

        return project_enterprise_artifact_semantic(
            semantic,
            self.json_payload(semantic),
        )

    def _personal_brief_artifact_semantic(
        self,
        semantic: dict[str, Any],
    ) -> dict[str, Any]:
        from docmirror.plugins.credit_report.personal_brief_native.projector import (
            project_personal_brief_artifact_semantic,
        )

        return project_personal_brief_artifact_semantic(
            semantic,
            self.json_payload(semantic),
        )

    def render_dataset_csvs(self, semantic: dict[str, Any] | None = None) -> dict[str, str]:
        semantic_payload = semantic or self.semantic_payload()
        if self._is_enterprise_semantic(semantic_payload):
            return super().render_dataset_csvs(
                self._enterprise_artifact_semantic(semantic_payload)
            )
        if self._is_personal_brief_semantic(semantic_payload):
            return super().render_dataset_csvs(
                self._personal_brief_artifact_semantic(semantic_payload)
            )
        return super().render_dataset_csvs(semantic_payload)

    def render_audit_csv(self, semantic: dict[str, Any] | None = None) -> str:
        semantic_payload = semantic or self.semantic_payload()
        if self._is_enterprise_semantic(semantic_payload):
            return super().render_audit_csv(
                self._enterprise_artifact_semantic(semantic_payload)
            )
        if self._is_personal_brief_semantic(semantic_payload):
            return super().render_audit_csv(
                self._personal_brief_artifact_semantic(semantic_payload)
            )
        return super().render_audit_csv(semantic_payload)


class CreditReportPlugin(CommunityProjector):
    """Community edition plugin for credit report document processing."""

    @property
    def domain_name(self) -> str:
        return "credit_report"

    @property
    def display_name(self) -> str:
        return "Credit Report (Community)"

    @property
    def projector_id(self) -> str:
        return self.domain_name

    @property
    def identity_fields(self) -> Sequence[tuple[str, Sequence[str]]]:
        return (
            ("subject_name", ("被查询者姓名", "企业名称", "姓名", "Name", "报告主体")),
            ("id_number", ("被查询者证件号码", "身份证号", "证件号码", "ID Number")),
            ("id_type", ("被查询者证件类型", "证件类型", "ID Type")),
            ("marital_status", ("婚姻状况",)),
            ("unified_social_credit_code", ("统一社会信用代码",)),
            ("zhongzheng_code", ("中征码", "贷款卡编码", "贷款卡号")),
            ("query_institution", ("查询机构",)),
            ("report_time", ("报告时间", "查询时间", "Report Time")),
            ("report_number", ("报告编号", "Report No", "NO.")),
        )

    def derive(self, parse_result, text: str = "") -> ProjectionData:
        from docmirror.plugins.credit_report.report_profile import (
            detect_credit_report_subtype,
        )

        report_subtype = detect_credit_report_subtype(parse_result, text)
        if report_subtype == "enterprise":
            from docmirror.plugins.credit_report.enterprise_native.projector import (
                derive_enterprise_projection,
            )

            return derive_enterprise_projection(self, parse_result, text)
        if report_subtype == "personal_brief":
            from docmirror.plugins.credit_report.personal_brief_native.projector import (
                derive_personal_brief_projection,
            )

            return derive_personal_brief_projection(self, parse_result, text)
        from docmirror.plugins.credit_report.projection import derive_credit_report_projection

        return derive_credit_report_projection(self, parse_result, text)

    def project_bundle(
        self,
        sealed,
        *,
        file_path: str = "",
        file_id: str = "001",
        document_id: str = "",
    ):
        """Apply document-variant presentation overrides inside this plugin."""
        from docmirror.models.sealed import SealedParseResult
        from docmirror.output.community_bundle import project_community_bundle
        from docmirror.plugins._base.projector import load_projection_policy

        if not isinstance(sealed, SealedParseResult):
            raise TypeError(f"{type(self).__name__}.project expects SealedParseResult")
        if not self.supports(sealed):
            return None
        before = sealed.integrity_fingerprint
        read_view = sealed.to_read_view()
        derived = self.derive(
            read_view,
            str(read_view.full_text or read_view.raw_text or ""),
        )
        policy = load_projection_policy(type(self).__module__.rsplit(".", 1)[0])
        overrides = derived.semantic.get("community_projection_overrides")
        if isinstance(overrides, dict):
            for key, values in overrides.items():
                if isinstance(values, dict):
                    policy[key] = {**dict(policy.get(key) or {}), **values}
                elif key in {
                    "internal_fields",
                    "internal_facts",
                    "publish_empty_datasets",
                } and isinstance(
                    values, (list, tuple)
                ):
                    policy[key] = list(
                        dict.fromkeys([*(policy.get(key) or ()), *map(str, values)])
                    )
        projected = project_community_bundle(
            sealed,
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            projection_data=derived.model_dump(mode="python"),
            projection_policy=policy,
        )
        bundle = _CreditReportCommunityBundle(
            schema=projected.schema,
            document=projected.document,
            sections=projected.sections,
            datasets=projected.datasets,
            files=projected.files,
            warnings=projected.warnings,
            result=projected.result,
            source_fingerprint=projected.source_fingerprint,
            parse_result_schema=projected.parse_result_schema,
            classification=projected.classification,
            domain=projected.domain,
            diagnostics=projected.diagnostics,
            content_markdown_override=projected.content_markdown_override,
        )
        bundle.render_markdown()
        if sealed.integrity_fingerprint != before or not sealed.verify_integrity():
            raise RuntimeError("Post-seal projector changed the sealed snapshot")
        return bundle

    def reading_projection(self, parse_result):
        from docmirror.plugins.credit_report.report_profile import (
            detect_credit_report_content_mode,
            detect_credit_report_subtype,
        )
        from docmirror.plugins.credit_report.variant_router import (
            resolve_credit_report_variant,
        )

        text = str(parse_result.full_text or parse_result.raw_text or "")
        report_subtype = detect_credit_report_subtype(parse_result, text)
        content_mode = detect_credit_report_content_mode(parse_result)
        variant = resolve_credit_report_variant(report_subtype, content_mode)
        return variant.build_reading_projection(
            parse_result,
            content_mode=content_mode,
        )


plugin = CreditReportPlugin()
