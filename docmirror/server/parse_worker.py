# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated process entry point for one DocMirror parse task."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

from docmirror.sdk.integration.request import InputRef, ParseRequest
from docmirror.server.task_executor import execute_parse_task

logger = logging.getLogger(__name__)


def parse_request_from_dict(payload: dict[str, Any]) -> ParseRequest:
    """Rebuild a canonical request serialized by the API process."""
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("serialized ParseRequest must contain at least one input")

    inputs: list[InputRef] = []
    input_fields = {item.name for item in fields(InputRef)}
    for raw_input in raw_inputs:
        if not isinstance(raw_input, dict):
            raise TypeError("serialized ParseRequest inputs must be objects")
        if raw_input.get("data") is not None:
            raise ValueError("isolated REST parsing requires file-backed inputs")
        inputs.append(InputRef(**{key: value for key, value in raw_input.items() if key in input_fields}))

    request_fields = {item.name for item in fields(ParseRequest)} - {"inputs"}
    request_values = {key: value for key, value in payload.items() if key in request_fields}
    return ParseRequest(inputs=inputs, **request_values)


async def run_worker(*, task_id: str, request_path: Path, output_root: Path) -> int:
    """Execute one serialized request and return a process exit code."""
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    request = parse_request_from_dict(payload)
    logger.info("event=parse_worker_started task_id=%s input_count=%s", task_id, len(request.inputs))
    result = await execute_parse_task(request, output_root=output_root, task_id=task_id)
    logger.info("event=parse_worker_finished task_id=%s status=%s", task_id, result.status)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated DocMirror parse task")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by :class:`ParseProcessManager`."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return asyncio.run(
            run_worker(
                task_id=args.task_id,
                request_path=args.request.resolve(),
                output_root=args.output_root.resolve(),
            )
        )
    except Exception:
        logger.exception("event=parse_worker_crashed task_id=%s", args.task_id)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
