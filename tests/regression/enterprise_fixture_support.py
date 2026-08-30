# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover enterprise PDFs in both private corpora, independent of cwd/case."""

import os
from fnmatch import fnmatchcase
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures-private" / "credit_report"
ENTERPRISE_FIXTURE_ROOTS = (
    _SOURCE_ROOT / "Digital Enterprise",
    _SOURCE_ROOT / "External" / "Digital Enterprise",
)


def enterprise_fixtures(pattern: str = "*.pdf") -> list[Path]:
    strict = os.environ.get("DOCMIRROR_REQUIRE_ENTERPRISE_FIXTURES") == "1"
    paths: list[Path] = []
    for root in ENTERPRISE_FIXTURE_ROOTS:
        pdfs = sorted(
            (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: path.name.casefold(),
        ) if root.is_dir() else []
        if strict and not pdfs:
            raise FileNotFoundError(f"Required enterprise PDF corpus is missing or empty: {root}")
        paths.extend(path for path in pdfs if fnmatchcase(path.name.casefold(), pattern.casefold()))
    if strict and not paths:
        raise FileNotFoundError(f"No required enterprise PDF matches {pattern!r}")
    return paths


def enterprise_fixture(name: str) -> Path:
    matches = enterprise_fixtures(name)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous enterprise fixture {name!r}: {matches}")
    return matches[0] if matches else ENTERPRISE_FIXTURE_ROOTS[0] / name
