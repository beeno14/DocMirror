# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io

from rich.console import Console

from docmirror.models.entities.parse_result import ParseResult, ResultStatus
from docmirror.models.sealed import seal_parse_result


def test_cli_safe_string_respects_legacy_console_encoding() -> None:
    from docmirror.__main__ import _safe_str

    rendered = _safe_str("❌ failed", encoding="gbk")

    assert "❌" not in rendered
    assert rendered.endswith(" failed")
    rendered.encode("gbk")


def test_single_file_cli_displays_sealed_parse_result(monkeypatch, tmp_path):
    import docmirror.__main__ as cli
    import docmirror.input.entry.factory as factory

    source = tmp_path / "id-card.jpg"
    source.write_bytes(b"fixture")
    sealed = seal_parse_result(ParseResult(status=ResultStatus.SUCCESS, raw_text="姓名 示例"))

    async def fake_perceive_document(*_args, **_kwargs):
        return sealed

    output = io.StringIO()
    monkeypatch.setattr(factory, "perceive_document", fake_perceive_document)
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))

    asyncio.run(cli.parse_document(str(source), tmp_path / "output", no_save=True))

    rendered = output.getvalue()
    assert "Parsing Complete!" in rendered
    assert "Critical Error" not in rendered
