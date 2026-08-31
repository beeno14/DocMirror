# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for removed pre-refactor import paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts.code_hygiene.clean_manifest import load_clean_manifest


def _find_spec(module: str):
    """Return no spec when any parent in an already-removed path is absent."""
    try:
        spec = importlib.util.find_spec(module)
    except (AttributeError, ModuleNotFoundError):
        return None
    if spec is not None and spec.loader is None and spec.submodule_search_locations:
        locations = (Path(location) for location in spec.submodule_search_locations)
        if not any(location.is_dir() and any(location.rglob("*.py")) for location in locations):
            return None
    return spec


def test_removed_pre_refactor_import_paths_do_not_resolve():
    removed = sorted(load_clean_manifest().removed_modules)

    for module in removed:
        assert _find_spec(module) is None, module


def test_canonical_replacements_resolve():
    replacements = [
        "docmirror.layout",
        "docmirror.input.adapters",
        "docmirror.framework.middlewares",
        "docmirror.output.exporters",
        "docmirror.sdk.integration",
        "docmirror.framework.di",
        "docmirror.plugins._runtime.licensing",
        "docmirror.plugin_api",
    ]

    for module in replacements:
        assert _find_spec(module) is not None, module
