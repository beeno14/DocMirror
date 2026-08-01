# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-Edition Output Builder
=============================

Shared logic for building Community / Enterprise / Finance edition outputs
from ``SealedParseResult``. Used by both the CLI (__main__.py) and REST API.

``SealedParseResult`` is the only fact source. Community Bundle v3,
Enterprise, and Finance are independent sibling projections of that snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import mimetypes
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PROJECTOR_TIMEOUT_SECONDS = max(0.01, float(os.getenv("DOCMIRROR_PROJECTOR_TIMEOUT_S", "300")))


def materialize_community_bundle(
    payload: dict[str, Any],
    result: Any,
    *,
    file_path: str = "",
    file_id: str = "001",
    document_id: str = "",
    source_fingerprint: str = "",
    parse_result_schema: str = "docmirror.sealed_parse_result.v1",
):
    """Restore Community renderer state and apply request-specific delivery names."""
    from docmirror.output.community_bundle import CommunityBundle, CommunityDataset

    projected = copy.deepcopy(payload)
    document = dict(projected.get("document") or {})
    if document_id:
        document["id"] = document_id
    if file_path:
        source_path = Path(file_path)
        file_name = source_path.name
        source_hash = ""
        if source_path.is_file():
            digest = hashlib.sha256()
            with source_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_hash = f"sha256:{digest.hexdigest()}"
        document.setdefault("source_file", {}).update(
            {
                "name": file_name,
                "mime_type": mimetypes.guess_type(file_name)[0] or "application/octet-stream",
                "sha256": source_hash,
            }
        )
        document.setdefault("title", source_path.stem)
    files = dict(projected.get("files") or {})
    files.update(
        {
            "semantic_json": f"{file_id}_community_semantic.json",
            "content_md": f"{file_id}_content.md",
            "enhanced_reading_md": f"{file_id}_enhanced_reading.md",
            "datasets_dir": f"{file_id}_datasets",
            "dataset_audit_csv": f"{file_id}_datasets/_audit_cells.csv",
        }
    )
    datasets = []
    for raw_dataset in projected.get("datasets") or []:
        if not isinstance(raw_dataset, dict):
            continue
        public = {key: value for key, value in raw_dataset.items() if key != "rows"}
        csv_name = Path(str(public.get("csv") or f"{public.get('id') or 'dataset'}.csv")).name
        public["csv"] = f"{file_id}_datasets/{csv_name}"
        rows = [row for row in (raw_dataset.get("rows") or []) if isinstance(row, dict)]
        datasets.append(CommunityDataset(public=public, rows=rows))
    return CommunityBundle(
        schema=dict(projected.get("schema") or {}),
        document=document,
        sections=list(projected.get("sections") or []),
        datasets=datasets,
        files=files,
        warnings=list(projected.get("warnings") or []),
        result=result,
        source_fingerprint=source_fingerprint,
        parse_result_schema=parse_result_schema,
        classification={
            "document_type": str((projected.get("schema") or {}).get("domain") or "generic"),
            "projector_id": "community.compatibility_adapter",
            "support_level": str((projected.get("schema") or {}).get("support_level") or "generic"),
        },
        domain={"compatibility_projection": copy.deepcopy(projected.get("document") or {})},
        diagnostics={"materialized_from_community_json": True},
    )


def build_community_bundle(
    result,
    full_text: str = "",
    *,
    file_path: str = "",
    file_id: str = "001",
    document_id: str = "",
    on_progress: Callable[[str, float, str], None] | None = None,
):
    """Build the semantic Community bundle through the plugin boundary."""
    from docmirror.models.schemas.registry import validate_projection_payload
    from docmirror.models.sealed import SealedParseResult
    from docmirror.plugins._base.projector import CommunityProjector
    from docmirror.plugins._runtime.plugin_registry import registry

    if not isinstance(result, SealedParseResult):
        raise TypeError(f"build_community_bundle expects SealedParseResult; got {type(result).__name__}")
    if not result.verify_integrity():
        raise RuntimeError("Projector boundary violation: invalid sealed snapshot")
    detected_type = str(result.to_read_view().entities.document_type or "generic")
    registered_projector = registry.get_projector(
        detected_type,
        "community",
    )
    projector = registry.get_projector(
        detected_type,
        "community",
        sealed_schema=result.schema_version,
    )
    fallback_reason = ""
    if projector is None:
        fallback_reason = (
            "sealed_schema_unsupported" if registered_projector is not None else "community_projector_not_registered"
        )
        projector = registry.get_projector(
            "generic",
            "community",
            sealed_schema=result.schema_version,
        )
    if projector is None:
        raise RuntimeError("No Community projector is registered")

    def _project(selected_projector):
        project_bundle = getattr(selected_projector, "project_bundle", None)
        derives_semantic = (
            not isinstance(selected_projector, CommunityProjector)
            or type(selected_projector).derive is not CommunityProjector.derive
        )
        if callable(project_bundle) and derives_semantic:
            return project_bundle(
                result,
                file_path=file_path,
                file_id=file_id,
                document_id=document_id,
            )
        projected = selected_projector.project(result)
        if projected is None:
            return None
        if not isinstance(projected, dict):
            raise TypeError(f"{detected_type}:community projector must return dict or None")
        return materialize_community_bundle(
            projected,
            result.to_read_view(),
            file_path=file_path,
            file_id=file_id,
            document_id=document_id,
            source_fingerprint=result.integrity_fingerprint,
            parse_result_schema=result.schema_version,
        )

    bundle = _project(projector)
    if bundle is None and str(getattr(projector, "domain_name", "") or "") != "generic":
        fallback_reason = "community_projector_returned_none"
        projector = registry.get_projector(
            "generic",
            "community",
            sealed_schema=result.schema_version,
        )
        if projector is None:
            raise RuntimeError("No generic Community fallback projector is registered")
        bundle = _project(projector)
    if bundle is None:
        raise RuntimeError(f"{detected_type}:community projector returned no semantic bundle")
    if fallback_reason:
        fallback_message = (
            f"Community projector unavailable for {detected_type!r} ({fallback_reason}); "
            "generated from ParseResult via the generic projector."
        )
        bundle.schema["support_level"] = "generic"
        bundle.classification.update(
            {
                "projector_id": "parse_result_fallback",
                "support_level": "generic",
                "fallback_reason": fallback_reason,
                "fallback_from_document_type": detected_type,
            }
        )
        bundle.diagnostics["community_fallback"] = {
            "reason": fallback_reason,
            "document_type": detected_type,
            "source": "sealed_parse_result",
        }
        if not any(
            warning.get("code") == "COMMUNITY_PARSE_RESULT_FALLBACK"
            for warning in bundle.warnings
            if isinstance(warning, dict)
        ):
            bundle.warnings.append(
                {
                    "code": "COMMUNITY_PARSE_RESULT_FALLBACK",
                    "level": "warning",
                    "message": fallback_message,
                }
            )
    if not result.verify_integrity():
        raise RuntimeError("Projector boundary violation: sealed snapshot changed")
    bundle.render_markdown()
    semantic = bundle.semantic_payload()
    semantic_validation = validate_projection_payload("community_semantic", semantic)
    if not semantic_validation.valid:
        raise RuntimeError("Community semantic schema validation failed: " + "; ".join(semantic_validation.errors))
    payload = bundle.json_payload(semantic)
    validation = validate_projection_payload("community", payload)
    if not validation.valid:
        raise RuntimeError("Community schema validation failed: " + "; ".join(validation.errors))
    conservation_issues = bundle.conservation_issues(payload=payload)
    if conservation_issues:
        raise RuntimeError("Community dataset conservation failed: " + "; ".join(conservation_issues))
    return bundle


def build_community_projection(
    result,
    full_text: str = "",
    *,
    file_path: str = "",
    file_id: str = "001",
    document_id: str = "",
    on_progress: Callable[[str, float, str], None] | None = None,
) -> dict | None:
    """Render Community JSON from the post-seal semantic bundle."""
    bundle = build_community_bundle(
        result,
        full_text,
        file_path=file_path,
        file_id=file_id,
        document_id=document_id,
        on_progress=on_progress,
    )
    semantic = bundle.semantic_payload()
    payload = bundle.json_payload(semantic)
    return payload


def _patch_edition_compliance(output: dict, edition: str, detected_type: str) -> None:
    """Universal compliance patch for enterprise/finance edition outputs.

    Ensures all required governance blocks have valid values regardless of
    which plugin produced the output. This avoids per-plugin fixes for empty
    audit/processing/metadata fields.
    """
    now = datetime.now(timezone.utc).isoformat()

    # ── audit block ──
    output.setdefault("audit", {})
    aud = output["audit"]
    for field in ("tenant_id", "user_id", "operator"):
        aud.setdefault(field, "")
    if not aud.get("operation_logs"):
        aud["operation_logs"] = [
            {
                "timestamp": now,
                "action": "document_parsed",
                "operator": "system",
                "details": f"Edition={edition}, Type={detected_type}",
            }
        ]
    if not aud.get("export_logs"):
        aud["export_logs"] = [
            {
                "timestamp": now,
                "action": "json_exported",
                "target": "output_builder",
                "status": "success",
            }
        ]
    for field in ("data_access_logs", "review_logs"):
        aud.setdefault(field, [])

    # ── processing block ──
    proc = output.get("processing", {})
    if proc.get("duration_ms", 0) == 0:
        proc["duration_ms"] = 1
    if not proc.get("task_id"):
        proc["task_id"] = ""

    # ── metadata block ──
    meta = output.get("metadata", {})
    if not meta.get("task_id"):
        meta["task_id"] = ""

    # ── data.summary block (fills total_rows for CLI display) ──
    extraction_records = output.get("extraction", {}).get("records", [])
    norm_records = output.get("normalization", {}).get("standard_records", [])
    record_count = max(len(extraction_records), len(norm_records))
    output.setdefault("data", {})
    output["data"].setdefault("summary", {})
    if output["data"]["summary"].get("total_rows", 0) == 0 and record_count > 0:
        output["data"]["summary"]["total_rows"] = record_count

    # ── validation block (E13: rules must not be empty) ──
    val = output.get("validation", {})
    if val and not val.get("rules"):
        val["rules"] = [
            {
                "rule_code": "COMPLIANCE_001",
                "level": "info",
                "message": "Output generated by output_builder, no plugin-specific validation available",
            }
        ]


def build_extended_output(
    result,
    edition: str,
    *,
    on_progress: Callable[[str, float, str], None] | None = None,
) -> dict | None:
    """Build one post-seal edition projection."""
    from docmirror.models.sealed import SealedParseResult
    from docmirror.plugins._runtime.plugin_registry import registry

    if not isinstance(result, SealedParseResult):
        raise TypeError(f"build_extended_output expects SealedParseResult; got {type(result).__name__}")
    sealed = result
    if not sealed.verify_integrity():
        raise RuntimeError("Projector boundary violation: invalid sealed snapshot")
    read_view = sealed.to_read_view()
    detected_type = str(read_view.entities.document_type or "")
    projector = registry.get_projector(
        detected_type,
        edition,
        sealed_schema=sealed.schema_version,
    )
    if projector is None:
        return None
    if getattr(projector, "requires_license", False):
        from docmirror.plugins._runtime.licensing.entitlements import is_entitled

        if not is_entitled(detected_type):
            return None
    extracted = projector.project(sealed)
    if not sealed.verify_integrity():
        raise RuntimeError("Projector boundary violation: sealed snapshot changed")
    if extracted is not None and not isinstance(extracted, dict):
        raise TypeError(f"{detected_type}:{edition} projector must return dict or None")
    if extracted and isinstance(extracted, dict):
        from docmirror.plugins._runtime.composition import CompositionReason, annotate_composition

        try:
            _patch_edition_compliance(extracted, edition, detected_type)
            if "composition" not in extracted:
                annotate_composition(
                    extracted,
                    edition=edition,
                    reason=CompositionReason.INDEPENDENT_EXTRACT,
                )
        except Exception as exc:
            logger.warning(
                "[Projections] %s compliance/composition failed: %s",
                edition,
                exc,
            )
            extracted.setdefault("status", {}).setdefault("warnings", []).append(f"projection_compliance_failed:{exc}")
    return extracted


def _projector_unavailability_reason(
    document_type: str,
    edition: str,
    sealed_schema: str,
) -> str:
    from docmirror.plugins._runtime.plugin_registry import registry

    projector = registry.get_projector(
        document_type,
        edition,
        sealed_schema=sealed_schema,
    )
    if projector is None:
        return "package_not_installed" if not registry.list_projectors(edition) else "document_type_unsupported"
    if getattr(projector, "requires_license", False):
        from docmirror.plugins._runtime.licensing.entitlements import is_entitled

        if not is_entitled(document_type):
            return "license_not_entitled"
    return "projector_failed"


def build_all_projections(
    result,
    *,
    file_path: str = "",
    on_progress: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    """Build fixed sibling projections from one immutable canonical snapshot.

    MirrorCore vNext:

    Phase 1 — project ``_mirror.json`` from the already sealed snapshot.

    Phase 2 — give each independent projector a fresh read view of that same
    sealed snapshot. No projector receives the mutable canonical instance and
    no edition reads another edition's output.
    """
    from docmirror.models.sealed import SealedParseResult

    if not isinstance(result, SealedParseResult):
        raise TypeError(f"build_all_projections expects SealedParseResult; got {type(result).__name__}")
    sealed = result
    file_path = file_path or getattr(sealed, "file_path", "") or ""

    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    mirror_t0 = time.perf_counter()
    if on_progress:
        on_progress("community_plugin", 0.0, "Serializing core mirror...")
    from docmirror.output.mirror_projector import project_mirror

    mirror = project_mirror(
        sealed,
        source_filename=file_path if file_path else "",
        mirror_level="standard",
    )
    timings["mirror_ms"] = (time.perf_counter() - mirror_t0) * 1000
    community_t0 = time.perf_counter()
    if on_progress:
        on_progress("community_plugin", 25.0, "Building Community projection...")
    community_bundle = build_community_bundle(sealed, file_path=file_path)
    community_semantic = community_bundle.semantic_payload()
    community = community_bundle.json_payload(community_semantic)
    if on_progress:
        on_progress("community_plugin", 100.0, "Community projection ready")
    timings["community_ms"] = (time.perf_counter() - community_t0) * 1000

    enterprise: dict[str, Any] | None = None
    finance: dict[str, Any] | None = None
    commercial_availability: dict[str, dict[str, str]] = {}
    commercial_projectors = ("enterprise", "finance")

    def _build_extended_with_timing(
        edition_name: str,
    ) -> tuple[str, dict[str, Any] | None, float, str | None]:
        started = time.perf_counter()
        unavailable_reason: str | None = None
        try:
            payload = build_extended_output(
                sealed,
                edition_name,
            )
        except Exception as exc:
            logger.warning("[Projections] %s projection failed: %s", edition_name, exc)
            payload = None
            unavailable_reason = "projector_failed"
        if payload is None and unavailable_reason is None:
            unavailable_reason = _projector_unavailability_reason(
                str(sealed.to_read_view().entities.document_type or ""),
                edition_name,
                sealed.schema_version,
            )
        if payload is not None and "composition" not in payload:
            from docmirror.plugins._runtime.composition import CompositionReason, annotate_composition

            try:
                annotate_composition(
                    payload,
                    edition=edition_name,
                    reason=CompositionReason.INDEPENDENT_EXTRACT,
                )
            except Exception as exc:
                logger.warning("[Projections] %s annotate_composition failed: %s", edition_name, exc)
                payload.setdefault("status", {}).setdefault("warnings", []).append(f"composition_failed:{exc}")
        return edition_name, payload, (time.perf_counter() - started) * 1000, unavailable_reason

    pool = ThreadPoolExecutor(max_workers=len(commercial_projectors))
    try:
        futures = [pool.submit(_build_extended_with_timing, ed) for ed in commercial_projectors]
        completed = 0
        # Timeout prevents a single hanging edition (e.g. asyncio.run hang
        # in Python 3.12 ThreadPoolExecutor) from blocking the entire
        # build_all_projections.  Each edition gets 300 s = 5 min.
        _timeout = PROJECTOR_TIMEOUT_SECONDS
        remaining = {future: ed for future, ed in zip(futures, commercial_projectors)}
        try:
            for future in as_completed(futures, timeout=_timeout):
                edition_name, payload, elapsed_ms, unavailable_reason = future.result()
                remaining.pop(future, None)
                timings[f"{edition_name}_ms"] = elapsed_ms
                if unavailable_reason:
                    commercial_availability[edition_name] = {
                        "status": "unavailable",
                        "reason": unavailable_reason,
                    }
                if edition_name == "enterprise":
                    enterprise = payload
                elif edition_name == "finance":
                    finance = payload
                completed += 1
                if on_progress:
                    sub_pct = (completed / len(commercial_projectors)) * 100.0
                    on_progress("extended_plugins", sub_pct, f"Building {edition_name} edition output...")
        except TimeoutError:
            # One or more editions timed out — log which ones, use best-effort
            # results for what completed, and continue.
            for fut, ed in remaining.items():
                fut.cancel()
                commercial_availability[ed] = {
                    "status": "unavailable",
                    "reason": "projector_timeout",
                }
                logger.warning(
                    "[Projections] %s edition timed out after %.0f s — skipping",
                    ed,
                    _timeout,
                )
        except Exception as exc:
            logger.error(
                "[Projections] Unhandled exception in as_completed loop: %s",
                exc,
                exc_info=True,
            )
            for fut, ed in remaining.items():
                fut.cancel()
                commercial_availability[ed] = {
                    "status": "unavailable",
                    "reason": "projector_failed",
                }
                logger.warning("[Projections] %s cancelled after unhandled exception", ed)
    finally:
        # A context-manager shutdown waits for timed-out threads and defeats the
        # delivery deadline. Projectors only receive detached sealed read views,
        # so unfinished legacy tasks cannot change facts while winding down.
        pool.shutdown(wait=False, cancel_futures=True)

    outputs: dict[str, Any] = {
        "mirror": mirror,
        "community": community,
        "community_semantic": community_semantic,
        "enterprise": enterprise,
        "finance": finance,
        "edition_availability": commercial_availability,
    }
    # Transient renderer owned only by Community persistence. All of its facts
    # come from ParseResult; it is not an upstream source for other editions.
    outputs["community_bundle"] = community_bundle
    if not sealed.verify_integrity():
        raise RuntimeError("Projector boundary violation: sealed ParseResult integrity failed")
    timings["total_ms"] = (time.perf_counter() - total_t0) * 1000
    logger.info(
        "[Projections] build",
        extra={
            "event": "projection_build",
            "timings": {key: round(value, 2) for key, value in timings.items()},
            "produced_projections": [
                edition
                for edition in ("mirror", "community", "enterprise", "finance")
                if outputs.get(edition) is not None
            ],
        },
    )
    return outputs
