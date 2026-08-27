"""Validate actual bank export files against dense and frozen output baselines."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from docmirror.runtime.serialization import dumps_json
from docmirror.server.edition_outputs import _write_community_bundle_files


def _first_difference(expected: Any, actual: Any, path: str = "$") -> str:
    """Report locations, not private business values, in regression failures."""
    if type(expected) is not type(actual):
        return path + ":type"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return path + ":keys"
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return path + ":length"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    elif expected != actual:
        return path
    return ""


def _source_value_equal(source: Any, value: Any) -> bool:
    if not _first_difference(source, value):
        return True
    # Legacy public source pools serialized structured cells as JSON strings.
    return isinstance(source, str) and isinstance(value, (dict, list)) and source == json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _audit_additional_fields(
    row: dict, delivered: dict, columns: list, aliases: dict, source: dict, *, serialized_sources: bool = False
) -> None:
    from docmirror.output.normalized_records import source_fields, value_is_represented

    normalized = delivered["normalized"]
    additional = normalized.get("additional_fields") or []
    original_additional = row["normalized"].get("additional_fields") or []
    if _first_difference(original_additional, additional[:len(original_additional)]):
        raise AssertionError("existing supplemental normalized values changed")
    descriptors = {column["key"]: column for column in columns}

    def equal(left: Any, right: Any) -> bool:
        return _source_value_equal(left, right) if serialized_sources else not _first_difference(left, right)

    for item in additional:
        if any(not _first_difference(item, original) for original in original_additional):
            continue
        if not isinstance(item, dict) or set(item) - {"name", "field", "value"}:
            raise AssertionError("invalid supplemental normalized field")
        name, value, field = item.get("name"), item.get("value"), item.get("field")
        raw = source.get("raw") or {}
        canonical = source.get("canonical_raw") or {}
        raw_backed = name in raw and equal(raw[name], value)
        canonical_backed = field in canonical and equal(canonical[field], value)
        if not (raw_backed or (name == field and canonical_backed)) or (field and not canonical_backed):
            raise AssertionError("supplemental normalized field is not source-backed")
    for name, value in (source.get("raw") or {}).items():
        fields = source_fields(name, row, columns, aliases)
        represented = any(
            key in normalized and value_is_represented(value, normalized[key], descriptors[key])
            for key in fields
        )
        retained = any(item["name"] == name and equal(value, item["value"]) for item in additional)
        if not represented and not retained:
            raise AssertionError(f"unrepresented original source field: {name}")
    for key, value in (source.get("canonical_raw") or {}).items():
        if value in (None, ""):
            continue
        represented = key in normalized and value_is_represented(value, normalized[key], descriptors.get(key, {}))
        retained = any(item.get("field") == key and equal(value, item["value"]) for item in additional)
        if not represented and not retained:
            raise AssertionError(f"unrepresented canonical source field: {key}")


def bind_internal_evidence(payload: dict, evidence: dict) -> dict:
    """Join actual evidence by stable identity, never invent raw from normalized."""
    if payload.get("schema", {}).get("version") == "5.0.0":
        from scripts.validate.bank_business_exports import assert_business_value_conservation, evidence_delivery

        original = evidence_delivery(evidence)
        assert_business_value_conservation(original, payload)
        return bind_internal_evidence(original, evidence)
    result = copy.deepcopy(payload)
    expected_datasets = evidence.get("datasets") or []
    datasets = result.get("datasets") or []
    if len(datasets) != len(expected_datasets):
        raise AssertionError("internal evidence dataset count differs")
    for dataset, expected in zip(datasets, expected_datasets, strict=True):
        for key in ("id", "name", "row_count", "completeness"):
            if _first_difference(dataset.get(key), expected.get(key)):
                raise AssertionError(f"internal evidence dataset {key} differs")
        rows, expected_rows = dataset.get("rows") or [], expected.get("rows") or []
        if len(rows) != len(expected_rows):
            raise AssertionError("internal evidence record count differs")
        for row, source_row in zip(rows, expected_rows, strict=True):
            for key in ("record_id", "normalized", "source", "confidence", "review"):
                if _first_difference(row.get(key), source_row.get(key)):
                    raise AssertionError(f"internal evidence record {key} differs")
            for key in ("canonical_raw", "raw"):
                if not isinstance(source_row.get(key), dict):
                    raise AssertionError(f"internal evidence missing {key}")
                row[key] = copy.deepcopy(source_row[key])
    result["schema"]["version"] = "3.0.0"
    return result


def assert_value_preserving_compaction(
    dense: dict,
    compact: dict,
    *,
    source_aliases: dict | None = None,
    source_datasets: dict | None = None,
) -> None:
    """Allow only declared empty normalized keys and reading columns to vanish."""
    expected = copy.deepcopy(dense)
    normalized_only = compact.get("schema", {}).get("version") == "4.0.0"
    if normalized_only:
        from docmirror.output.normalized_records import ADDITIONAL_FIELDS_COLUMN, strip_source_value_pools

        strip_source_value_pools(expected)
        if compact.get("reading", {}).get("privacy_mode") == "full":
            expected["reading"]["privacy_mode"] = "full"
    datasets = expected.get("datasets") or []
    compact_datasets = compact.get("datasets") or []
    if len(datasets) != len(compact_datasets):
        raise AssertionError("compact export changed dataset count")
    omissions: dict[str, set[str]] = {}
    for dataset, compact_dataset in zip(datasets, compact_datasets, strict=True):
        if normalized_only:
            original_dataset = next(item for item in dense["datasets"] if item["id"] == dataset["id"])
            source_rows = (source_datasets or {}).get(dataset["name"], original_dataset["rows"])
            if len(dataset["rows"]) != len(compact_dataset["rows"]):
                raise AssertionError("compact export changed record count")
            for row, original, delivered, source_row in zip(
                dataset["rows"], original_dataset["rows"], compact_dataset["rows"], source_rows, strict=True
            ):
                _audit_additional_fields(
                    original, delivered, dataset["columns"],
                    (source_aliases or {}).get(dataset["name"], {}), source_row,
                    serialized_sources=dataset["name"] not in (source_datasets or {}),
                )
                if "additional_fields" in delivered["normalized"]:
                    row["normalized"]["additional_fields"] = copy.deepcopy(delivered["normalized"]["additional_fields"])
            added = [column for column in compact_dataset["columns"] if column["key"] == "additional_fields"]
            if added and not any(column["key"] == "additional_fields" for column in dataset["columns"]):
                if added != [ADDITIONAL_FIELDS_COLUMN]:
                    raise AssertionError("supplemental normalized column contract changed")
                dataset["columns"].extend(copy.deepcopy(added))
                for table in expected.get("reading", {}).get("tables", []):
                    if table["dataset_id"] == dataset["id"]:
                        table["column_keys"].append("additional_fields")
        omitted = compact_dataset.get("omitted_normalized_fields") or []
        if omitted:
            if len(omitted) != len(set(omitted)):
                raise AssertionError("duplicate omitted field declaration")
            declared = {column["key"] for column in dataset["columns"]}
            if not set(omitted).issubset(declared):
                raise AssertionError("omission refers to an undeclared field")
            dataset["omitted_normalized_fields"] = omitted
            omissions[dataset["id"]] = set(omitted)
        for row in dataset.get("rows") or []:
            for key in omitted:
                if row["normalized"].get(key) not in (None, ""):
                    raise AssertionError(f"compact export discarded nonempty normalized field: {key}")
                row["normalized"].pop(key, None)
    for table in expected.get("reading", {}).get("tables", []):
        omitted = omissions.get(table["dataset_id"], set())
        table["column_keys"] = [key for key in table["column_keys"] if key not in omitted]
    difference = _first_difference(expected, compact)
    if difference:
        raise AssertionError(f"compact export changed data outside the allowed omissions: {difference}")


def validate_and_write_bank_exports(
    bundle: Any,
    output_dir: Path,
    *,
    baseline: dict | None = None,
) -> tuple[dict, dict]:
    """Exercise the production writer, without a second extraction or PDF run."""
    if bundle.compact_output.get("omit_absent_fields") is not True:
        raise AssertionError("digital bank compact export was not enabled")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_community_bundle_files(bundle, output_dir, file_id="001", document_id=bundle.document["id"])
    compact_text = paths["community"].read_text(encoding="utf-8")
    compact = json.loads(compact_text)
    semantic = bundle.semantic_payload()
    dense_domain = copy.deepcopy(bundle.domain)
    dense_domain["extensions"].pop("compact_output", None)
    dense_bundle = replace(bundle, domain=dense_domain)
    dense_semantic = dense_bundle.semantic_payload()
    dense = dense_bundle.json_payload(dense_semantic)
    business_only = compact.get("schema", {}).get("version") == "5.0.0"
    comparison_payload = compact
    if business_only:
        from scripts.validate.bank_business_exports import (
            assert_business_csv_conservation,
            assert_business_markdown_values,
            assert_business_value_conservation,
            evidence_delivery,
        )
        from scripts.validate.validate_community_artifacts import validate_community_artifacts

        artifact_issues = validate_community_artifacts(paths["community"])
        if artifact_issues:
            raise AssertionError(f"business artifact contract failed: {'; '.join(artifact_issues[:10])}")
        comparison_payload = evidence_delivery(semantic)
        assert_business_value_conservation(comparison_payload, compact)
        assert_business_markdown_values(compact, paths["enhanced_reading"].read_text(encoding="utf-8"))
    assert_value_preserving_compaction(
        dense, comparison_payload,
        source_aliases=bundle.compact_output.get("source_aliases"),
        source_datasets={dataset.public["name"]: dataset.rows for dataset in bundle.datasets},
    )
    normalized_only = bundle.compact_output.get("normalized_only") is True
    for actual_dataset, original_dataset in zip(semantic["datasets"], dense_semantic["datasets"], strict=True):
        for actual_row, original_row in zip(actual_dataset["rows"], original_dataset["rows"], strict=True):
            for key in ("canonical_raw", "raw", "source"):
                if _first_difference(actual_row[key], original_row[key]):
                    raise AssertionError(f"internal source evidence changed: {key}")
    if baseline is not None:
        difference = _first_difference(baseline, dense)
        if difference:
            raise AssertionError(f"dense extraction differs from frozen baseline: {difference}")
    if compact_text != dumps_json(compact, ensure_ascii=False, separators=(",", ":")):
        raise AssertionError("Community JSON was not losslessly minified")

    for relative_path, content in dense_bundle.render_dataset_csvs(dense_semantic).items():
        actual = (output_dir / relative_path).read_text(encoding="utf-8")
        if business_only:
            dataset = next(item for item in compact["datasets"] if item["csv"] == relative_path)
            assert_business_csv_conservation(content, actual, dataset)
            continue
        original_rows = list(csv.DictReader(io.StringIO(content.lstrip("\ufeff"))))
        actual_rows = list(csv.DictReader(io.StringIO(actual.lstrip("\ufeff"))))
        if normalized_only:
            for row in actual_rows:
                row.pop("additional_fields", None)
        if _first_difference(original_rows, actual_rows):
            raise AssertionError(f"CSV changed: {Path(relative_path).name}")
    original_audit = list(csv.DictReader(io.StringIO(dense_bundle.render_audit_csv(dense_semantic))))
    actual_audit = list(csv.DictReader(io.StringIO((output_dir / "001_datasets/_audit_cells.csv").read_text(encoding="utf-8"))))
    if normalized_only:
        actual_audit = [row for row in actual_audit if row["field_key"] != "additional_fields"]
    if _first_difference(original_audit, actual_audit):
        raise AssertionError("audit CSV changed")
    if paths["content"].read_bytes() != dense_bundle.render_markdown().encode("utf-8"):
        raise AssertionError("source-faithful Markdown changed")

    dense_markdown = dense_bundle.render_enhanced_markdown(dense_semantic)
    return compact, {
        "status": "pass",
        "baseline_checked": baseline is not None,
        "business_values_unchanged": True,
        "raw_and_provenance_unchanged": True,
        "csv_unchanged": not normalized_only,
        "audit_csv_unchanged": not normalized_only,
        "existing_csv_fields_unchanged": not business_only,
        "existing_csv_business_fields_unchanged": True,
        "existing_audit_cells_unchanged": True,
        "source_markdown_unchanged": True,
        "normalized_only": normalized_only,
        "business_view": business_only,
        "artifact_contract_checked": business_only,
        "raw_only_business_fields_accounted_for": normalized_only,
        "additional_field_count": sum(
            len(row["normalized"].get("additional_fields") or [])
            for dataset in semantic["datasets"] for row in dataset["rows"]
        ),
        "dense_json_bytes": len(dumps_json(dense, ensure_ascii=False, indent=2).encode("utf-8")),
        "compact_json_bytes": paths["community"].stat().st_size,
        "dense_enhanced_markdown_bytes": len(dense_markdown.encode("utf-8")),
        "compact_enhanced_markdown_bytes": paths["enhanced_reading"].stat().st_size,
        "omitted_fields_by_dataset": {
            dataset["name"]: dataset.get("omitted_normalized_fields", []) for dataset in compact["datasets"]
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
    }


def replay_export_report(report_path: Path) -> dict:
    """Revalidate saved exports after replay-only fixes, without extraction."""
    from docmirror.models.entities.parse_result import ParseResult
    from docmirror.models.schemas.registry import validate_projection_payload
    from docmirror.output.community_bundle import render_community_reading_markdown
    from docmirror.server.output_builder import materialize_community_bundle
    from scripts.validate.validate_community_artifacts import validate_community_artifacts

    repo_root = Path(__file__).resolve().parents[2]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = []
    started = time.perf_counter()
    for result in report["results"]:
        outcome = {"filename": result["filename"], "source_sha256": result["source_sha256"]}
        try:
            validation = result["export_validation"]
            if validation.get("status") != "pass" or validation.get("baseline_checked") is not True:
                raise AssertionError("prior export did not pass frozen-baseline validation")
            artifact_path = Path(validation["artifacts"]["community"])
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            cached_payload = json.loads((repo_root / result["community"]).read_text(encoding="utf-8"))
            if _first_difference(cached_payload, payload):
                raise AssertionError("saved delivery artifact differs from validated projection cache")
            if payload.get("schema", {}).get("version") == "5.0.0":
                artifact_issues = validate_community_artifacts(artifact_path)
                if artifact_issues:
                    raise AssertionError(f"business artifact contract failed: {'; '.join(artifact_issues[:10])}")
                outcome["artifact_contract_checked"] = True
            restored = materialize_community_bundle(payload, ParseResult())
            replayed = restored.json_payload()
            difference = _first_difference(payload, replayed)
            if difference:
                raise AssertionError(f"replay changed Community data: {difference}")
            if not validate_projection_payload("community", replayed).valid:
                raise AssertionError("replayed Community JSON failed schema validation")
            if render_community_reading_markdown(payload) != render_community_reading_markdown(replayed):
                raise AssertionError("replay changed public enhanced Markdown")
            if payload["schema"]["version"] in {"4.0.0", "5.0.0"}:
                from docmirror.output.community_bundle import _community_view_from_semantic

                cache_path = repo_root / result["community"]
                evidence_path = cache_path.with_suffix(".evidence.json")
                meta_path = cache_path.with_name(cache_path.name.replace(".community.json", ".meta.json"))
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("evidence_sha256") != hashlib.sha256(evidence_path.read_bytes()).hexdigest():
                    raise AssertionError("retained internal evidence checksum differs")
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                bind_internal_evidence(payload, evidence)
                if _first_difference(payload, _community_view_from_semantic(evidence)):
                    raise AssertionError("current-code export from retained evidence differs")
                if render_community_reading_markdown(evidence).encode("utf-8") != Path(
                    validation["artifacts"]["enhanced_reading"]
                ).read_bytes():
                    raise AssertionError("current-code enhanced Markdown differs from delivery")
                aliases = evidence["domain"]["extensions"]["compact_output"].get("source_aliases") or {}
                for dataset in evidence["datasets"]:
                    for row in dataset["rows"]:
                        original = {**row, "normalized": {k: v for k, v in row["normalized"].items() if k != "additional_fields"}}
                        _audit_additional_fields(
                            original, row, dataset["columns"], aliases.get(dataset["name"], {}), row,
                            serialized_sources=True,
                        )
                evidence_bundle = replace(restored, domain=evidence["domain"], diagnostics={})
                for relative, content in evidence_bundle.render_dataset_csvs(evidence).items():
                    if (artifact_path.parent / relative).read_bytes() != content.encode("utf-8"):
                        raise AssertionError("current-code dataset CSV differs from delivery")
                if (artifact_path.parent / "001_datasets/_audit_cells.csv").read_bytes() != evidence_bundle.render_audit_csv(
                    evidence
                ).encode("utf-8"):
                    raise AssertionError("current-code audit CSV differs from delivery")
                outcome["internal_evidence_checked"] = True
                outcome["current_exporters_checked"] = True
                outcome["source_business_fields_accounted_for"] = True
                if payload["schema"]["version"] == "5.0.0":
                    from scripts.validate.bank_business_exports import assert_business_markdown_values

                    assert_business_markdown_values(payload, render_community_reading_markdown(payload))
                    outcome["business_view_checked"] = True
                if validation.get("markdown_unmasked") is True:
                    if payload["reading"].get("privacy_mode") != "full":
                        raise AssertionError("unmasked Markdown policy was lost on replay")
                    outcome["unmasked_markdown_preserved"] = True
            if any(dataset.get("omitted_normalized_fields") for dataset in payload["datasets"]):
                if restored.compact_output.get("minify_json") is not True:
                    raise AssertionError("sparse replay lost compact JSON formatting")
            outcome.update(
                status="pass",
                business_values_unchanged=True,
                source_evidence_unchanged=True,
                sparse_json_preserved=True,
                enhanced_markdown_preserved=True,
            )
        except Exception as exc:
            outcome.update(status="error", error=f"{type(exc).__name__}: {exc}")
        results.append(outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
    return {
        "operation": "compact_export_replay",
        "source_report": str(report_path.resolve()),
        "extraction_executed": False,
        "file_count": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "failed": sum(result["status"] != "pass" for result in results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "code_sha256": {
            relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
            for relative in (
                "docmirror/output/community_bundle.py",
                "docmirror/output/normalized_records.py",
                "docmirror/output/bank_business_view.py",
                "docmirror/server/edition_outputs.py",
                "docmirror/server/output_builder.py",
                "docmirror/plugins/bank_statement/community_plugin.py",
                "docmirror/configs/schemas/community_bundle.schema.json",
                "scripts/validate/bank_compact_exports.py",
                "scripts/validate/bank_business_exports.py",
                "scripts/validate/validate_community_artifacts.py",
            )
        },
        "results": results,
    }


def refresh_bank_markdown_report(report_path: Path, output_dir: Path, *, business_only: bool = False) -> dict:
    """Publish an unmasked presentation from retained facts, never re-extract.

    Historical artifacts remain immutable. Only presentation policies and the
    public layout change; every normalized value, source cell and CSV is checked.
    """
    from datetime import datetime, timezone

    from docmirror.models.entities.parse_result import ParseResult
    from docmirror.models.schemas.registry import validate_projection_payload
    from docmirror.output.community_bundle import _community_view_from_semantic, render_community_reading_markdown
    from docmirror.server.artifact_writer import ArtifactWriter
    from docmirror.server.output_builder import materialize_community_bundle
    from scripts.validate.bank_business_exports import (
        assert_business_csv_conservation,
        assert_business_markdown_values,
        assert_business_value_conservation,
        evidence_delivery,
    )
    from scripts.validate.validate_community_artifacts import validate_community_artifacts

    repo_root = Path(__file__).resolve().parents[2]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    outcomes = []
    started = time.perf_counter()
    for index, original_result in enumerate(report["results"], start=1):
        outcome = copy.deepcopy(original_result)
        try:
            validation = outcome["export_validation"]
            if validation.get("status") != "pass" or validation.get("baseline_checked") is not True:
                raise AssertionError("only previously validated exports can be refreshed")
            if outcome.get("audit_status") != "pass" or outcome.get("error_count") != 0:
                raise AssertionError("prior business audit did not pass")
            cache_path = (repo_root / outcome["community"]).resolve()
            evidence_path = cache_path.with_suffix(".evidence.json")
            meta_path = cache_path.with_name(cache_path.name.replace(".community.json", ".meta.json"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source_evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if meta.get("evidence_sha256") != source_evidence_sha:
                raise AssertionError("retained internal evidence checksum differs")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            previous = json.loads(cache_path.read_text(encoding="utf-8"))
            artifact_path = Path(validation["artifacts"]["community"]).resolve()
            if _first_difference(previous, json.loads(artifact_path.read_text(encoding="utf-8"))):
                raise AssertionError("prior artifact differs from validated projection cache")
            if previous["schema"]["domain"] != "bank_statement" or previous["schema"]["version"] != "4.0.0":
                raise AssertionError("presentation refresh is limited to normalized digital-bank exports")
            bind_internal_evidence(previous, evidence)
            if _first_difference(previous, _community_view_from_semantic(evidence)):
                raise AssertionError("prior export differs from retained facts")

            # Only presentation policies change in the saved semantic source.
            evidence["domain"]["extensions"].setdefault("enhanced_markdown", {})["privacy_mode"] = "full"
            if business_only:
                evidence["domain"]["extensions"]["compact_output"]["business_view"] = True
            payload = _community_view_from_semantic(evidence)
            if business_only:
                original_business = evidence_delivery(evidence)
                assert_business_value_conservation(original_business, payload)
                assert_business_markdown_values(payload, render_community_reading_markdown(evidence))
            else:
                expected = copy.deepcopy(previous)
                expected["reading"]["privacy_mode"] = "full"
                if _first_difference(expected, payload):
                    raise AssertionError("unmasking changed data beyond the reading policy")
            for kind, value in (("community", payload), ("community_semantic", evidence)):
                if not validate_projection_payload(kind, value).valid:
                    raise AssertionError(f"refreshed {kind} failed schema validation")
            restored = materialize_community_bundle(payload, ParseResult())
            if _first_difference(payload, restored.json_payload()):
                raise AssertionError("unmasked JSON replay changed data")
            restored = replace(restored, domain=evidence["domain"], diagnostics={})
            csvs = restored.render_dataset_csvs(evidence)
            csvs["001_datasets/_audit_cells.csv"] = restored.render_audit_csv(evidence)
            if restored.conservation_issues(payload=payload, dataset_csvs=csvs):
                raise AssertionError("refreshed dataset conservation failed")
            case_dir = output_dir / f"{index:03d}"
            for relative, content in csvs.items():
                if not (case_dir / relative).resolve().is_relative_to(case_dir):
                    raise AssertionError("CSV path escapes the new artifact directory")
                if not (artifact_path.parent / relative).resolve().is_relative_to(artifact_path.parent):
                    raise AssertionError("CSV path escapes the original artifact directory")
                previous_bytes = (artifact_path.parent / relative).read_bytes()
                if business_only and relative != "001_datasets/_audit_cells.csv":
                    dataset = next(item for item in payload["datasets"] if item["csv"] == relative)
                    assert_business_csv_conservation(previous_bytes.decode("utf-8"), content, dataset)
                elif previous_bytes != content.encode("utf-8"):
                    raise AssertionError("presentation refresh changed CSV business values or evidence")
            writer = ArtifactWriter(case_dir)
            paths = {
                "community": writer.write_text("001_community.json", dumps_json(payload, ensure_ascii=False, separators=(",", ":"))),
                "enhanced_reading": writer.write_text("001_enhanced_reading.md", render_community_reading_markdown(evidence)),
                "content": writer.write_text("001_content.md", Path(validation["artifacts"]["content"]).read_bytes().decode("utf-8")),
                "datasets": case_dir / "001_datasets",
            }
            for relative, content in csvs.items():
                writer.write_text(relative, content)
            if business_only:
                artifact_issues = validate_community_artifacts(paths["community"])
                if artifact_issues:
                    raise AssertionError(f"business artifact contract failed: {'; '.join(artifact_issues[:10])}")
            refreshed_cache = writer.write_json("projection.community.json", payload)
            refreshed_evidence = writer.write_json("projection.community.evidence.json", evidence)
            writer.write_json("projection.meta.json", {
                "community_sha256": hashlib.sha256(refreshed_cache.read_bytes()).hexdigest(),
                "evidence_sha256": hashlib.sha256(refreshed_evidence.read_bytes()).hexdigest(),
                "source_community": str(cache_path),
                "source_evidence_sha256": source_evidence_sha,
                "presentation_only_refresh": True,
                "business_view": business_only,
            })
            validation.update(
                artifacts={key: str(path) for key, path in paths.items()},
                compact_json_bytes=paths["community"].stat().st_size,
                compact_enhanced_markdown_bytes=paths["enhanced_reading"].stat().st_size,
                markdown_unmasked=True,
                presentation_only_refresh=True,
                business_view=business_only,
                artifact_contract_checked=business_only,
                existing_csv_fields_unchanged=not business_only and validation.get("existing_csv_fields_unchanged", False),
                existing_csv_business_fields_unchanged=True,
            )
            outcome.update(status="presentation_refreshed", community=str(refreshed_cache))
        except Exception as exc:
            outcome.update(status="error", error=f"{type(exc).__name__}: {exc}")
        outcomes.append(outcome)
        print(json.dumps({"filename": outcome["filename"], "status": outcome["status"], "error": outcome.get("error")}), flush=True)
    return {
        **{key: value for key, value in report.items() if key not in {"results", "status_counts"}},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation": "business_output_refresh" if business_only else "unmasked_markdown_refresh",
        "source_report": str(report_path.resolve()),
        "extraction_executed": False,
        "file_count": len(outcomes),
        "passed": sum(result["status"] == "presentation_refreshed" for result in outcomes),
        "failed": sum(result["status"] != "presentation_refreshed" for result in outcomes),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": outcomes,
    }


def main() -> int:
    import argparse

    from docmirror.server.artifact_writer import ArtifactWriter

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay-report", type=Path)
    source.add_argument("--unmask-report", type=Path)
    source.add_argument("--business-report", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_report = args.replay_report or args.unmask_report or args.business_report
    if args.output.resolve() == source_report.resolve():
        parser.error("replay output must not overwrite the source report")
    if (args.unmask_report or args.business_report) and not args.artifact_dir:
        parser.error("presentation refresh requires a new --artifact-dir")
    report = (
        refresh_bank_markdown_report(source_report, args.artifact_dir, business_only=bool(args.business_report))
        if (args.unmask_report or args.business_report) else replay_export_report(source_report)
    )
    ArtifactWriter(args.output.resolve().parent).write_json(args.output.name, report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
