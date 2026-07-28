# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ping An Bank monthly statement parser for special embedded-font PDFs."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from docmirror.plugins.bank_statement.context import StyleContext

PARSER_ID = "pingan_monthly_statement"
STYLE_ID = "pingan_monthly_statement"

_CACHE_KEY = "_bank_pingan_monthly_ocr"
_DATE_RE = re.compile(r"^20\d{6}$")
_MONEY_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^[+-]?\d+(?:\.\d{2})$")
_ACCOUNT_RE = re.compile(r"^\d{7,24}$")
_ROW_TOLERANCE = 11.0

_HEADER_LABELS = {
    "seq": "序号",
    "date": "日期",
    "amount": "借/贷方发生额",
    "balance": "余额",
    "counter_party": "对方户名",
    "counter_account": "对方账户",
    "voucher": "传票号",
    "summary": "摘要",
}

logger = logging.getLogger(__name__)


def looks_like_pingan_monthly_statement(ctx: StyleContext) -> bool:
    """Return whether the sealed result looks like a Ping An monthly statement."""
    text = ctx.full_text or ""
    headers = " ".join(cell for table in ctx.tables for row in table[:12] for cell in row)
    joined = f"{text}\n{headers}"
    required = ("客户存款月结单", "借/贷方发生额", "承前余额")
    institution = str(ctx.institution or "")
    has_pingan = "平安银行" in joined or "PINGAN" in joined.upper() or "平安银行" in institution
    return all(token in joined for token in required) and has_pingan


def extract_transactions(ctx: StyleContext, _plugin: Any) -> list[dict[str, Any]]:
    """Extract Ping An monthly statement rows from OCR tokens."""
    payload = _recover(ctx)
    rows = list(payload.get("transactions") or [])
    if rows:
        expected = len(rows)
        if ctx.reconstruction is not None:
            ctx.reconstruction = replace(
                ctx.reconstruction,
                source="ocr_implicit_table",
                expected_primary_rows=expected,
                pipe_parse_failed=False,
            )
    return rows


def normalize_record(raw_txn: dict[str, Any], _plugin: Any) -> dict[str, Any]:
    """Normalize one already column-aligned Ping An monthly statement row."""
    amount_text = str(raw_txn.get("交易金额") or "").replace(",", "")
    direction = "income" if str(raw_txn.get("收/支") or "") == "收入" else "expense"
    try:
        amount = abs(float(amount_text))
    except ValueError:
        amount = 0.0
    try:
        balance = float(str(raw_txn.get("余额") or "").replace(",", ""))
    except ValueError:
        balance = None
    return {
        "sequence_no": str(raw_txn.get("序号") or ""),
        "date": _normalize_date(str(raw_txn.get("交易日期") or "")),
        "timestamp": "",
        "direction": direction,
        "summary": str(raw_txn.get("摘要") or ""),
        "amount": amount,
        "amount_cny": amount,
        "balance": balance,
        "counter_party": str(raw_txn.get("对方户名") or ""),
        "counter_account": str(raw_txn.get("对方账号") or ""),
        "counter_bank_code": "",
        "counter_bank_name": "",
        "channel": "",
        "purpose": "",
        "counterparty_status": "present",
    }


def extract_identity(ctx: StyleContext) -> dict[str, dict[str, Any]]:
    """Return OCR-backed identity fields from the cached Ping An page header."""
    payload = _recover(ctx)
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    labels = {
        "account_holder": "户名",
        "account_number": "账号",
        "bank_name": "银行",
        "bank_branch": "客户行",
        "currency": "币种",
    }
    for field_name, value in identity.items():
        if field_name not in labels or value in (None, ""):
            continue
        out[field_name] = {
            "raw_name": labels[field_name],
            "raw_value": str(value),
            "normalized_value": str(value),
            "data_type": "string",
            "source": "pingan_monthly_ocr",
            "source_refs": [{"source": "pingan_monthly_ocr", "page_id": "page:0001"}],
            "evidence_ids": list(payload.get("identity_evidence_ids") or []),
        }
    return out


def recovery_warnings(ctx: StyleContext) -> list[str]:
    """Return explicit recovery warnings for quality gates."""
    payload = _recover(ctx)
    return [str(item) for item in (payload.get("warnings") or []) if str(item)]


def _recover(ctx: StyleContext) -> dict[str, Any]:
    cached = _cached_payload(ctx.parse_result)
    if cached is not None:
        return cached
    payload = _recover_from_source_pdf(ctx)
    _store_payload(ctx.parse_result, payload)
    return payload


def _cached_payload(parse_result: Any) -> dict[str, Any] | None:
    ds = _domain_specific(parse_result)
    if ds is None:
        return None
    cached = ds.get(_CACHE_KEY)
    return cached if isinstance(cached, dict) else None


def _store_payload(parse_result: Any, payload: dict[str, Any]) -> None:
    ds = _domain_specific(parse_result)
    if ds is not None:
        ds[_CACHE_KEY] = payload


def _domain_specific(parse_result: Any) -> dict[str, Any] | None:
    entities = getattr(parse_result, "entities", None) if parse_result is not None else None
    ds = getattr(entities, "domain_specific", None) if entities is not None else None
    return ds if isinstance(ds, dict) else None


def _recover_from_source_pdf(ctx: StyleContext) -> dict[str, Any]:
    path = Path(str(getattr(ctx.parse_result, "file_path", "") or ""))
    if not path.is_file():
        return _failed_payload("pingan_monthly_ocr:source_pdf_unavailable")
    try:
        pages = _ocr_pdf_pages(path)
    except Exception as exc:
        logger.warning("[PingAnMonthly] OCR failed: %s", exc)
        return _failed_payload("pingan_monthly_ocr:ocr_unavailable_or_failed")

    identity: dict[str, str] = {}
    identity_evidence_ids: list[str] = []
    transactions: list[dict[str, Any]] = []
    page_counts: dict[int, int] = {}
    for page_no, tokens in pages:
        if page_no == 1:
            identity, identity_evidence_ids = _extract_identity_from_tokens(tokens)
        rows = _extract_page_rows(tokens, page_no=page_no)
        page_counts[page_no] = len(rows)
        transactions.extend(rows)

    warnings: list[str] = []
    if not transactions:
        warnings.append("pingan_monthly_ocr:no_rows_recovered")
    if not identity.get("account_holder"):
        warnings.append("pingan_monthly_ocr:account_holder_missing")
    if not identity.get("account_number"):
        warnings.append("pingan_monthly_ocr:account_number_missing")
    if _has_balance_breaks(transactions):
        warnings.append("pingan_monthly_ocr:balance_chain_failed")
    if any(not (row.get("_source") or {}).get("bbox") for row in transactions):
        warnings.append("pingan_monthly_ocr:row_bbox_missing")

    return {
        "status": "ready" if transactions else "failed",
        "source": "pingan_monthly_ocr",
        "identity": identity,
        "identity_evidence_ids": identity_evidence_ids,
        "transactions": transactions,
        "page_counts": page_counts,
        "warnings": warnings,
    }


def _failed_payload(reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "source": "pingan_monthly_ocr",
        "identity": {},
        "identity_evidence_ids": [],
        "transactions": [],
        "page_counts": {},
        "warnings": [reason],
    }


def _ocr_pdf_pages(path: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    import fitz

    from docmirror.ocr.vision.rapidocr_engine import get_ocr_engine

    engine = get_ocr_engine()
    pages: list[tuple[int, list[dict[str, Any]]]] = []
    with fitz.open(path) as doc:
        for page_index in range(len(doc)):
            page_no = page_index + 1
            image = _render_page_bgr(doc[page_index])
            words = engine.detect_image_words(image, multi_scale=False)
            tokens = [_word_to_token(word, page_no=page_no, index=index) for index, word in enumerate(words)]
            pages.append((page_no, [token for token in tokens if token["text"]]))
    return pages


def _render_page_bgr(page: Any) -> Any:
    import cv2
    import numpy as np

    pix = page.get_pixmap(matrix=fitz_matrix(1.5), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def fitz_matrix(scale: float) -> Any:
    import fitz

    return fitz.Matrix(scale, scale)


def _word_to_token(word: tuple[Any, ...], *, page_no: int, index: int) -> dict[str, Any]:
    confidence = 1.0
    for pos in (-1, 5):
        try:
            candidate = float(word[pos])
        except (IndexError, TypeError, ValueError):
            continue
        if 0.0 <= candidate <= 1.0:
            confidence = candidate
            break
    bbox = [float(word[0]), float(word[1]), float(word[2]), float(word[3])]
    return {
        "id": f"pingan_ocr:p{page_no}:t{index}",
        "text": str(word[4] or "").strip(),
        "bbox": bbox,
        "confidence": confidence,
        "page": page_no,
    }


def _extract_identity_from_tokens(tokens: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    header_tokens = [token for token in tokens if _center(token)[1] < 155]
    text = " ".join(str(token["text"]) for token in sorted(header_tokens, key=lambda item: (_y0(item), _x0(item))))
    identity = {
        "bank_name": "平安银行" if "平安银行" in text else "",
        "bank_branch": _match_value(text, r"客户行[:：]\s*([^\s]+?支行)"),
        "account_holder": _match_value(text, r"户名[:：]\s*([^\s]+?有限公司)"),
        "account_number": _match_value(text, r"账号[:：]\s*(\d{8,24})"),
        "currency": _match_value(text, r"币种[:：]\s*([A-Z ]?R\s*M\s*B|RMB)").replace(" ", ""),
    }
    evidence_ids = [str(token["id"]) for token in header_tokens if any(label in str(token["text"]) for label in ("客户行", "户名", "账号", "币种", "平安银行"))]
    return identity, evidence_ids


def _match_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_page_rows(tokens: list[dict[str, Any]], *, page_no: int) -> list[dict[str, Any]]:
    header = _header_positions(tokens)
    if not header:
        return []
    boundaries = _column_boundaries(header)
    date_tokens = [
        token
        for token in tokens
        if _DATE_RE.match(str(token["text"]))
        and boundaries["date"][0] <= _center(token)[0] < boundaries["date"][1]
        and _center(token)[1] > header["bottom"]
    ]
    date_tokens.sort(key=lambda token: (_center(token)[1], _center(token)[0]))
    rows: list[dict[str, Any]] = []
    for fallback_seq, date_token in enumerate(date_tokens, start=1):
        row_tokens = _tokens_on_row(tokens, _center(date_token)[1], boundaries)
        row = _build_row(row_tokens, boundaries, page_no=page_no, fallback_seq=fallback_seq)
        if row:
            rows.append(row)
    return rows


def _header_positions(tokens: list[dict[str, Any]]) -> dict[str, float]:
    positions: dict[str, float] = {}
    bottom = 0.0
    for field, label in _HEADER_LABELS.items():
        token = _find_header_token(tokens, label)
        if token is None:
            continue
        positions[field] = _center(token)[0]
        bottom = max(bottom, _y1(token))
    required = {"date", "amount", "balance", "counter_party", "counter_account", "summary"}
    if not required.issubset(positions):
        return {}
    positions["bottom"] = bottom
    return positions


def _find_header_token(tokens: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    exact = [
        token
        for token in tokens
        if str(token.get("text") or "").strip() == label and _center(token)[1] > 120
    ]
    if exact:
        return min(exact, key=lambda token: (_center(token)[1], _center(token)[0]))
    contained = [
        token
        for token in tokens
        if label in str(token.get("text") or "") and _center(token)[1] > 120
    ]
    if contained:
        return min(contained, key=lambda token: (_center(token)[1], _center(token)[0]))
    return None


def _column_boundaries(header: dict[str, float]) -> dict[str, tuple[float, float]]:
    ordered = ["seq", "date", "amount", "balance", "counter_party", "counter_account", "voucher", "summary"]
    centers = {key: float(header.get(key, 0.0)) for key in ordered}
    fallback = {
        "seq": 45.0,
        "date": 140.0,
        "amount": 260.0,
        "balance": 390.0,
        "counter_party": 565.0,
        "counter_account": 840.0,
        "voucher": 1025.0,
        "summary": 1100.0,
    }
    for key, value in fallback.items():
        centers[key] = centers[key] or value
    bounds: dict[str, tuple[float, float]] = {}
    for idx, key in enumerate(ordered):
        left = 0.0 if idx == 0 else (centers[ordered[idx - 1]] + centers[key]) / 2.0
        right = 100000.0 if idx == len(ordered) - 1 else (centers[key] + centers[ordered[idx + 1]]) / 2.0
        bounds[key] = (left, right)
    return bounds


def _tokens_on_row(
    tokens: list[dict[str, Any]],
    y_center: float,
    boundaries: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    left = boundaries["seq"][0]
    right = boundaries["summary"][1]
    row = [
        token
        for token in tokens
        if left <= _center(token)[0] <= right and abs(_center(token)[1] - y_center) <= _ROW_TOLERANCE
    ]
    return sorted(row, key=lambda token: (_center(token)[0], _center(token)[1]))


def _build_row(
    tokens: list[dict[str, Any]],
    boundaries: dict[str, tuple[float, float]],
    *,
    page_no: int,
    fallback_seq: int,
) -> dict[str, Any] | None:
    columns = {name: _tokens_in_band(tokens, bounds) for name, bounds in boundaries.items()}
    date = _first_text(columns["date"], _DATE_RE)
    amount = _first_text(columns["amount"], _MONEY_RE)
    balance = _first_text(columns["balance"], _MONEY_RE)
    if not date or not amount or not balance:
        return None
    direction = "收入" if amount.startswith("+") else "支出"
    amount_abs = amount.lstrip("+-").replace(",", "")
    sequence = _first_text(columns["seq"], re.compile(r"^\d{1,4}$")) or str(fallback_seq)
    counter_party = _join_text(columns["counter_party"])
    counter_account = _first_text(columns["counter_account"], _ACCOUNT_RE) or _join_text(columns["counter_account"])
    summary = _join_text(columns["summary"])
    voucher = _join_text(columns["voucher"])
    row_bbox = _union_bbox([token["bbox"] for token in tokens if token.get("bbox")])
    evidence_ids = [str(token["id"]) for token in tokens if token.get("id")]
    confidence = round(mean(float(token.get("confidence", 1.0)) for token in tokens), 4) if tokens else 1.0
    return {
        "序号": sequence,
        "交易日期": date,
        "收/支": direction,
        "交易金额": amount_abs,
        "余额": balance.replace(",", ""),
        "对方户名": counter_party,
        "对方账号": counter_account,
        "传票号": voucher,
        "摘要": summary,
        "_source": {
            "source": "pingan_monthly_ocr",
            "source_page": page_no,
            "page_id": f"page:{page_no:04d}",
            "table_id": f"pingan_monthly_ocr_p{page_no}",
            "source_row_index": max(int(sequence or fallback_seq) - 1, 0),
            "page_range": [page_no, page_no],
            "bbox": row_bbox,
            "confidence": confidence,
            "evidence_ids": evidence_ids,
            "source_cell_refs": [
                {
                    "page": page_no,
                    "bbox": row_bbox,
                    "confidence": confidence,
                    "evidence_ref": evidence_ids[0] if evidence_ids else "",
                }
            ],
        },
    }


def _tokens_in_band(tokens: list[dict[str, Any]], bounds: tuple[float, float]) -> list[dict[str, Any]]:
    left, right = bounds
    return [token for token in tokens if left <= _center(token)[0] < right]


def _first_text(tokens: list[dict[str, Any]], pattern: re.Pattern[str]) -> str:
    for token in tokens:
        text = str(token["text"]).strip()
        if pattern.match(text):
            return text
    return ""


def _join_text(tokens: list[dict[str, Any]]) -> str:
    return "".join(str(token["text"]).strip() for token in tokens if str(token["text"]).strip())


def _normalize_date(value: str) -> str:
    if _DATE_RE.match(value):
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def _has_balance_breaks(rows: list[dict[str, Any]]) -> bool:
    previous_balance: float | None = None
    for row in rows:
        try:
            amount = float(str(row.get("交易金额") or "0").replace(",", ""))
            balance = float(str(row.get("余额") or "0").replace(",", ""))
        except ValueError:
            return True
        direction = str(row.get("收/支") or "")
        if previous_balance is not None:
            expected = previous_balance + amount if direction == "收入" else previous_balance - amount
            if abs(round(expected - balance, 2)) > 0.01:
                return True
        previous_balance = balance
    return False


def _center(token: dict[str, Any]) -> tuple[float, float]:
    bbox = token.get("bbox") or [0, 0, 0, 0]
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _x0(token: dict[str, Any]) -> float:
    return float((token.get("bbox") or [0])[0])


def _y0(token: dict[str, Any]) -> float:
    return float((token.get("bbox") or [0, 0])[1])


def _y1(token: dict[str, Any]) -> float:
    return float((token.get("bbox") or [0, 0, 0, 0])[3])


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return []
    return [
        round(min(float(box[0]) for box in boxes), 3),
        round(min(float(box[1]) for box in boxes), 3),
        round(max(float(box[2]) for box in boxes), 3),
        round(max(float(box[3]) for box in boxes), 3),
    ]


__all__ = [
    "PARSER_ID",
    "STYLE_ID",
    "extract_identity",
    "extract_transactions",
    "looks_like_pingan_monthly_statement",
    "normalize_record",
    "recovery_warnings",
]
