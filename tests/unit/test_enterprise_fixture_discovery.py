from pathlib import Path

import pytest

from tests.regression import enterprise_fixture_support as support


def test_both_corpora_and_uppercase_pdf_extensions_are_discovered(tmp_path: Path, monkeypatch):
    internal = tmp_path / "Digital Enterprise"
    external = tmp_path / "External" / "Digital Enterprise"
    internal.mkdir()
    external.mkdir(parents=True)
    (internal / "internal.PDF").touch()
    (external / "external.pdf").touch()
    monkeypatch.setattr(support, "ENTERPRISE_FIXTURE_ROOTS", (internal, external))
    monkeypatch.setenv("DOCMIRROR_REQUIRE_ENTERPRISE_FIXTURES", "1")
    assert support.enterprise_fixtures() == [internal / "internal.PDF", external / "external.pdf"]
    assert support.enterprise_fixture("EXTERNAL.PDF") == external / "external.pdf"


def test_strict_acceptance_never_silently_skips_a_missing_corpus(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(support, "ENTERPRISE_FIXTURE_ROOTS", (tmp_path / "missing",))
    monkeypatch.setenv("DOCMIRROR_REQUIRE_ENTERPRISE_FIXTURES", "1")
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        support.enterprise_fixtures()


def test_strict_acceptance_rejects_missing_named_fixture(tmp_path: Path, monkeypatch):
    (tmp_path / "other.pdf").touch()
    monkeypatch.setattr(support, "ENTERPRISE_FIXTURE_ROOTS", (tmp_path,))
    monkeypatch.setenv("DOCMIRROR_REQUIRE_ENTERPRISE_FIXTURES", "1")
    with pytest.raises(FileNotFoundError, match="No required"):
        support.enterprise_fixture("required.pdf")
