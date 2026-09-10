"""
core/extractor.py — Azure Edition
Supports: Mistral Azure OCR, OpenAI Azure, Phi Azure, Llama Azure.
Adding a new Azure provider: implement _extract_<name> and add to PROVIDERS.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _retry(fn, retries: int, timeout_codes: tuple = (408, 503)):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = fn()
            raw, status = result[0], result[1]
            if status == 429:
                wait = 30 * attempt
                print(f"  [429] Rate limited, attempt {attempt}/{retries} — waiting {wait}s")
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Rate limited after {retries} attempts.")
            if status in timeout_codes:
                wait = 10 * attempt
                print(f"  [{status}] attempt {attempt}/{retries} — retrying in {wait}s")
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API returned {status} after {retries} attempts.")
            if status != 200:
                raise RuntimeError(f"API call failed [{status}]: {raw}")
            return raw, status
        except requests.exceptions.Timeout as e:
            last_exc = e
            wait = 10 * attempt
            print(f"  Timeout on attempt {attempt}/{retries} — retrying in {wait}s")
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"All {retries} attempts timed out.") from last_exc


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        clean = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return json.loads(clean)


def _loose(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", k.lower())


def _unwrap(data: dict) -> dict:
    for _ in range(3):
        if not isinstance(data, dict):
            break
        for v in data.values():
            if isinstance(v, dict) and len(v) >= 3:
                data = v
                break
        else:
            break
    return data


def _normalize_response(data: dict, positions: dict, estimate_confidence: bool = False) -> dict:
    if not positions:
        return data

    flat           = _unwrap(data)
    response_loose = {_loose(k): v for k, v in flat.items()}
    normalized     = {}

    for key, label, *_ in positions["header_fields"]:
        val = (
            flat.get(key)
            or response_loose.get(_loose(key))
            or response_loose.get(_loose(label))
        )
        if isinstance(val, dict) and "value" in val:
            normalized[key] = val
        else:
            conf = _estimate_confidence(val) if estimate_confidence else None
            normalized[key] = {"value": val, "confidence": conf}

    normalized["line_items"] = data.get("line_items", [])

    # If the same value is explicitly extracted for multiple header fields,
    # treat the duplicate occurrence as high-confidence as well.
    header_values = {}

    for field, result in normalized.items():
        if field == "line_items":
            continue

        if isinstance(result, dict):
            value = result.get("value")

            if value is not None and str(value).strip():
                normalized_value = re.sub(r"[^a-zA-Z0-9]", "", str(value)).lower()

                if normalized_value:
                    header_values.setdefault(normalized_value, []).append(field)

    # Duplicate values across different header fields are valid.
    # Example: Supplier Phone == Ship To Phone.
    for normalized_value, fields in header_values.items():
        if len(fields) > 1:
            for field in fields:
                if isinstance(normalized[field], dict):
                    normalized[field]["confidence"] = 1.0

    return normalized


def _estimate_confidence(val) -> float | None:
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return 0.0
    if len(s) < 2:
        return 0.4
    return 0.85


def _is_missing_header_value(value) -> bool:
    """Return True when an extracted header value is effectively empty."""
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or text in {"null", "none", "n/a", "na", "-", "—"}


def _value_search_text(value: Any) -> str:
    """Normalize a value for duplicate-value matching in OCR/PDF text."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _duplicate_value_present_in_document(value: Any, document_text: str) -> bool:
    """
    Check whether a non-empty extracted value occurs at least twice in the
    document. Two occurrences are important: one can belong to the source
    field and the second can be the duplicated value for the missing field.
    """
    if _is_missing_header_value(value):
        return False

    value_text = _value_search_text(value)
    document_text = _value_search_text(document_text)
    if not value_text or not document_text:
        return False

    # Exact text match first (preserves formatted values such as 877-624-5757).
    if document_text.count(value_text) >= 2:
        return True

    # Phone numbers are often formatted differently by OCR, e.g.
    # 877-624-5757 vs 8776245757. Match their digits instead.
    digits = re.sub(r"\D", "", value_text)
    if len(digits) >= 7:
        digit_text = re.sub(r"\D", "", document_text)
        return digit_text.count(digits) >= 2

    return False


def _field_family(key: str, label: str) -> str:
    """
    Classify header fields so a duplicated value is only copied between
    semantically compatible fields. Line items are deliberately excluded.
    """
    text = f"{key} {label}".lower()
    if re.search(r"\b(phone|telephone|tel)\b", text):
        return "phone"
    if re.search(r"\bfax\b", text):
        return "fax"
    if re.search(r"\b(email|e-mail)\b", text):
        return "email"
    if re.search(r"\b(address|addr)\b", text):
        return "address"
    if re.search(r"\b(date|day)\b", text):
        return "date"
    if re.search(r"\b(id|identifier|number|no\.?|#)\b", text):
        return "id_or_number"
    if re.search(r"\bcurrency\b", text):
        return "currency"
    if re.search(r"\b(terms|payment)\b", text):
        return "terms"
    return "other"


def _apply_duplicate_header_fallback(
    result: dict,
    pdf_path: str,
    positions: dict = None,
) -> dict:
    """
    Fill missing HEADER fields when the same explicitly extracted value
    appears more than once in the source document.

    This is intentionally a post-processing safeguard rather than another
    prompt instruction. It does NOT inspect or modify ``line_items``.

    Example:
        Supplier Phone = 8776245757
        Ship To Phone = null

    If 8776245757 occurs twice in the PDF, Ship To Phone is populated with
    the same value and a high confidence score.

    Values are only copied between compatible field families (phone->phone,
    date->date, address->address, etc.) to avoid blindly copying unrelated
    duplicate numbers.
    """
    if not isinstance(result, dict) or not positions:
        return result

    try:
        document_text = _extract_pdf_text(pdf_path)
    except Exception as exc:
        print(f"  [duplicate-header-fallback] Could not read PDF text: {exc}")
        return result

    header_fields = positions.get("header_fields", [])
    if not header_fields or not document_text:
        return result

    # Collect existing non-empty header values.
    existing = []
    for key, label, *_ in header_fields:
        field = result.get(key)
        value = field.get("value") if isinstance(field, dict) else field
        if _is_missing_header_value(value):
            continue
        existing.append((key, label, value, _field_family(key, label)))

    changes = []

    for target_key, target_label, *_ in header_fields:
        target = result.get(target_key)
        target_value = target.get("value") if isinstance(target, dict) else target

        # Never overwrite a value that the model already extracted.
        if not _is_missing_header_value(target_value):
            continue

        target_family = _field_family(target_key, target_label)
        candidates = []

        for source_key, source_label, source_value, source_family in existing:
            if source_key == target_key or source_family != target_family:
                continue
            if _duplicate_value_present_in_document(source_value, document_text):
                candidates.append((source_key, source_label, source_value))

        # Only fill when exactly one compatible duplicate candidate exists.
        # This avoids making a guess when multiple different values are present.
        if len(candidates) == 1:
            source_key, source_label, source_value = candidates[0]
            result[target_key] = {
                "value": source_value,
                "confidence": 1.0,
            }
            changes.append(
                f"{target_key} <- {source_key} ({source_value!r})"
            )

    if changes:
        print("  [duplicate-header-fallback] Filled: " + ", ".join(changes))

    return result


def _extract_pdf_text(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is required. Install: pip install pdfplumber")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def _build_schema_instruction(positions: dict) -> str:
    if not positions:
        return ""

    field_list = "\n".join(
        f'  "{k}": {{"value": "<extracted text or null>", "confidence": <0.0-1.0>}}'
        for k, l, *_ in positions["header_fields"]
    )
    li_fields = "\n".join(
        f'    "{k}": "<{l}>"'
        for k, l, _ in positions["li_fields"]
        if k != "confidence"
    )

    return (
        "Return a JSON object with this EXACT structure:\n"
        "{\n"
        f"{field_list},\n"
        '  "line_items": [\n'
        "    {\n"
        f"{li_fields},\n"
        '      "confidence": <0.0-1.0 score for this row>\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Confidence scoring rules:\n"
        "  1.0 = the value is clearly present anywhere in the document and can be confidently assigned to the requested field.\n"
        "  0.7 = the value is present but its association with the requested field is somewhat ambiguous.\n"
        "  0.4 = the value is only inferred or uncertain.\n"
        "  0.0 = the value cannot be found anywhere in the document; in this case set the value to null.\n"
        "IMPORTANT: If a requested field has an exact matching value elsewhere in the document and the value is clearly applicable to that field, assign confidence 1.0 even if the value is duplicated under another field.\n"
        "IMPORTANT: Duplicate values across fields must NOT reduce confidence. If Supplier Phone and Ship To Phone contain the same phone number, and the number is clearly applicable to both fields, both fields must have confidence 1.0.\n"
        "COLUMN SEPARATION RULES:\n"
        "  - The terms table has 4 columns: Terms | Expected Date | Ship Via | FOB\n"
        "  - Extract each column value independently — never merge adjacent column values\n"
        "  - FOB contains only the delivery point (e.g. 'Your Shop') — not the shipping method\n"
        "  - Ship Via contains only the shipping method (e.g. 'best way') — not FOB\n"
        "  - Ship Via may contain multiple words including 'pp & c' — this is valid and complete\n"
        "  - Do not penalise confidence for Ship Via containing 'pp & c' — it is part of the shipping method\n"
        "IMPORTANT — FIELD EXTRACTION RULES:\n"
        "  1. Treat every header field as an independent field.\n"
        "  2. Search the entire purchase order for each requested field before deciding it is missing.\n"
        "  3. If the same value appears multiple times and is explicitly associated with multiple fields, extract it into EVERY applicable field.\n"
        "  4. Never leave a field null simply because the same value was already extracted for another field.\n"
        "  5. Do not deduplicate values across different fields. Duplicate values are valid and expected.\n"
        "  6. For example, if Supplier Phone and Ship To Phone contain the same phone number, populate BOTH fields with that phone number.\n"
        "  7. The same rule applies to repeated addresses, dates, IDs, currencies, names, or other values when they are explicitly associated with different fields.\n"
        "  8. Use null ONLY when the requested field cannot be found anywhere in the document.\n"
        "DUPLICATE FIELD EXAMPLE:\n"
        "If Supplier Phone is 877-624-5757 and Ship To Phone is also 877-624-5757, return 877-624-5757 for BOTH fields. Do not set Ship To Phone to null because the value is duplicated."
    )


# ── Company detection ──────────────────────────────────────────────────────────

def detect_company_from_pdf(pdf_path: str, model_cfg: dict = None) -> str:
    """
    Extract the buying/issuing company name from a PO PDF using text parsing.
    No model call needed — reads first meaningful line or Ship To section.
    """
    import pdfplumber

    skip_words = {
        "purchase order", "invoice", "quotation", "quote", "sales order",
        "date", "page", "cad funds", "canada", "phone", "fax", "attn",
        "purchase order number", "purchase order date page", "ship to",
        "vendor address", "vendor", "supplier", "bill to", "sold to",
        "reference", "terms", "ship via", "currency", "buyer name",
    }

    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text() or ""

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    def is_company_line(line):
        if len(line) < 4:
            return False
        if re.match(r'^[\d\s\/\-\.\,]+$', line):
            return False
        if line.lower().rstrip(":") in skip_words:
            return False
        if any(line.lower().startswith(w) for w in
               ["phone", "fax", "attn", "http", "www", "po box", "p.o. box", "gst", "hst", "pst"]):
            return False
        if re.search(r'\d{3,}', line) and any(
            w in line.lower() for w in ["st.", "ave", "blvd", "rd", "drive", "way"]
        ):
            return False
        return True

    # Strategy 1 — company name near top of document
    for line in lines[:10]:
        if is_company_line(line):
            if (re.search(r'\b(inc|ltd|llc|corp|limited|company|consultants|associates|brooks)\b',
                          line, re.IGNORECASE)
                    or (line.isupper() and len(line.split()) >= 2)):
                cleaned = re.sub(
                    r'\s+(purchase\s+order|invoice|quotation|order|po|page|date).*$',
                    '', line, flags=re.IGNORECASE
                ).strip()
                return cleaned

    # Strategy 2 — find Ship To label
    for i, line in enumerate(lines):
        if re.search(r'ship\s*to', line, re.IGNORECASE):
            parts = re.split(r'ship\s*to\s*:?\s*', line, flags=re.IGNORECASE)
            if len(parts) > 1 and is_company_line(parts[-1].strip()):
                return parts[-1].strip()
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                parts = re.split(r'\s{3,}|\t', candidate)
                right = parts[-1].strip() if len(parts) >= 2 else candidate
                if is_company_line(right):
                    return right

    # Strategy 3 — first meaningful line
    for line in lines:
        if is_company_line(line):
            return line

    return ""


# ── Mistral Azure — Document AI with chat completions fallback ─────────────────

def _extract_mistral_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Mistral on Azure — tries Document AI annotation first, falls back to chat completions."""
    pdf_b64 = base64.standard_b64encode(Path(pdf_path).read_bytes()).decode("utf-8")
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}",
    }

    # ── Attempt 1: Document AI annotation API ─────────────────────────────────
    schema  = _build_mistral_schema(positions)
    payload = {
        "model": model_cfg["model_name"],
        "document": {
            "type":         "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        "document_annotation_format": schema,
        "document_annotation_prompt": prompt,   # may be rejected on older Azure deployments
        "include_image_base64":       False,
    }

    ocr_text = None

    def call_annotation():
        resp = requests.post(
            model_cfg["endpoint"].rstrip("/"),
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 300),
        )
        return resp.text, resp.status_code

    try:
        raw, status = call_annotation()

        if status == 422 and "extra_forbidden" in raw:
            # Older Azure deployment — strip the prompt field and retry
            print("  [mistral_azure] document_annotation_prompt rejected — retrying without it")
            payload.pop("document_annotation_prompt")
            raw, status = call_annotation()

        if status != 200:
            raise RuntimeError(f"API call failed [{status}]: {raw}")

        data = json.loads(raw)
        ann  = data.get("document_annotation")

        if ann:
            result = json.loads(ann) if isinstance(ann, str) else ann
            result = _normalize_response(result, positions, estimate_confidence=False)
            return _apply_duplicate_header_fallback(result, pdf_path, positions)

        # Annotation empty — extract OCR text from pages for fallback
        print("  [mistral_azure] No document_annotation in response — falling back to OCR text + chat")
        ocr_text = "\n\n".join(p.get("markdown", "") for p in data.get("pages", []))

    except Exception as e:
        print(f"  [mistral_azure] Annotation API failed ({e}) — falling back to chat completions")
        ocr_text = None

    # ── Attempt 2: Chat completions fallback ──────────────────────────────────
    if not ocr_text:
        ocr_text = _extract_pdf_text(pdf_path)

    schema_instruction = _build_schema_instruction(positions)
    chat_payload = {
        "model": model_cfg["model_name"],
        "messages": [
            {
                "role":    "system",
                "content": "You are a purchase order data extraction assistant. Return valid JSON only.",
            },
            {
                "role":    "user",
                "content": (
                    f"{prompt}\n\n{schema_instruction}\n\n"
                    "Return ONLY valid JSON — no explanation, no markdown.\n\n"
                    f"PURCHASE ORDER TEXT:\n{ocr_text}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature":     0,
    }

    # Derive chat endpoint from OCR endpoint
    chat_endpoint = model_cfg["endpoint"].rstrip("/")
    if "/ocr" in chat_endpoint:
        chat_endpoint = chat_endpoint.replace("/ocr", "/chat/completions")
    elif not chat_endpoint.endswith("/chat/completions"):
        chat_endpoint = chat_endpoint + "/chat/completions"

    def call_chat():
        resp = requests.post(
            chat_endpoint,
            headers=headers,
            json=chat_payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, _  = _retry(call_chat, model_cfg.get("retries", 3))
    data    = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    result  = _parse_json(content)
    result = _normalize_response(result, positions, estimate_confidence=False)
    return _apply_duplicate_header_fallback(result, pdf_path, positions)


def _build_mistral_schema(positions: dict) -> dict:
    """Build JSON schema for Mistral Azure annotation API."""
    if not positions:
        return {}

    props = {}
    for key, label, *_ in positions["header_fields"]:
        props[key] = {
            "type": "object",
            "properties": {
                "value":      {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "description": label,
        }

    li_props = {}
    for key, label, _ in positions["li_fields"]:
        if key == "confidence":
            li_props["confidence"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        else:
            li_props[key] = {"type": ["string", "null"], "description": label}
    if "confidence" not in li_props:
        li_props["confidence"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}

    props["line_items"] = {
        "type":  "array",
        "items": {"type": "object", "properties": li_props},
    }

    required = [f[0] for f in positions["header_fields"]] + ["line_items"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "purchase_order_template",
            "schema": {
                "type":       "object",
                "properties": props,
                "required":   required,
            },
        },
    }


# ── OpenAI Azure — chat completions ───────────────────────────────────────────

def _extract_openai_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """OpenAI GPT-4o on Azure — extracts text then uses chat completions."""
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    headers = {
        "api-key":      model_cfg["api_key"],
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {
                "role":    "system",
                "content": "You are a purchase order data extraction assistant. Always return valid JSON only.",
            },
            {
                "role":    "user",
                "content": (
                    f"{prompt}\n\n{schema_instruction}\n\n"
                    "Return ONLY valid JSON — no explanation, no markdown.\n\n"
                    f"PURCHASE ORDER TEXT:\n{text}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature":     0,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, _  = _retry(call, model_cfg.get("retries", 3))
    data    = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    result  = _parse_json(content)
    result = _normalize_response(result, positions, estimate_confidence=False)
    return _apply_duplicate_header_fallback(result, pdf_path, positions)


# ── Phi Azure — Azure AI chat completions ─────────────────────────────────────

def _extract_phi_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Microsoft Phi on Azure AI — uses Azure AI inference chat completions."""
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    headers = {
        "Authorization": f"Bearer {model_cfg['api_key']}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model_cfg["model_name"],
        "messages": [
            {
                "role":    "system",
                "content": "You are a purchase order data extraction assistant. Always return valid JSON only.",
            },
            {
                "role":    "user",
                "content": (
                    f"{prompt}\n\n{schema_instruction}\n\n"
                    "Return ONLY valid JSON — no explanation, no markdown.\n\n"
                    f"PURCHASE ORDER TEXT:\n{text}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature":     0,
        "max_tokens":      4096,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, _  = _retry(call, model_cfg.get("retries", 3))
    data    = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    result  = _parse_json(content)
    result = _normalize_response(result, positions, estimate_confidence=False)
    return _apply_duplicate_header_fallback(result, pdf_path, positions)


# ── Llama Azure — Azure AI chat completions ───────────────────────────────────

def _extract_llama_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    li_list = ""
    if positions:
        li_list = "\n".join(
            f'- "{k}": {l}' for k, l, _ in positions["li_fields"]
            if k != "confidence"
        )

    headers = {
        "Authorization": f"Bearer {model_cfg['api_key']}",
        "Content-Type":  "application/json",
    }

    full_prompt = (
        f"{prompt}\n\n{schema_instruction}\n\n"
        "CRITICAL — LINE ITEMS:\n"
        "You MUST extract every single product row from the table.\n"
        "Even if 'Supplier Phone' and 'Ship To Phone' is same, extract both.\n"
        "Count the rows in the table and make sure your line_items array has the same count.\n"
        f"Each line item must have these fields:\n{li_list}\n\n"
        "Return ONLY valid JSON — no explanation, no markdown.\n\n"
        f"PURCHASE ORDER TEXT:\n{text}"
    )

    payload = {
        "model":    model_cfg["model_name"],
        "messages": [
            {
                "role":    "system",
                "content": "You are a precise purchase order data extraction assistant. Always extract ALL line items — never skip any rows. Return valid JSON only.",
            },
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0,
        "max_tokens":  8192,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, _  = _retry(call, model_cfg.get("retries", 3))
    data    = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    result  = _parse_json(content)
    result = _normalize_response(result, positions, estimate_confidence=True)
    return _apply_duplicate_header_fallback(result, pdf_path, positions)

# ── OpenAI Azure — Responses API (/v1/responses endpoint) ────────────────────

def _extract_openai_azure_responses(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """GPT-5.x on Azure using the new /v1/responses API (not chat completions)."""
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    headers = {
        "api-key":      model_cfg["api_key"],
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_cfg["model_name"],
        "input": (
            f"{prompt}\n\n{schema_instruction}\n\n"
            "Return ONLY valid JSON — no explanation, no markdown.\n\n"
            f"PURCHASE ORDER TEXT:\n{text}"
        ),
        "instructions": "You are a purchase order data extraction assistant. Always return valid JSON only.",
        "text": {
            "format": {
                "type": "json_object",
            }
        },
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"],
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 120),
        )
        return resp.text, resp.status_code

    raw, _  = _retry(call, model_cfg.get("retries", 3))
    data    = json.loads(raw)

    # Responses API returns output as a list of content blocks
    try:
        content = data["output"][0]["content"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Responses API response structure: {raw[:300]}")

    result = _parse_json(content)
    result = _normalize_response(result, positions, estimate_confidence=False)
    return _apply_duplicate_header_fallback(result, pdf_path, positions)

def download_from_sharepoint(url: str, output_dir: str = None) -> str:
    import tempfile, re, time, urllib.parse
    import requests as req

    download_url = url + ("&download=1" if "?" in url else "?download=1")

    # Ensure output dir exists
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_dir = output_dir or tempfile.gettempdir()

    print(f"  Downloading from SharePoint: {url[:50]}...")
    r = req.get(download_url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download from SharePoint [{r.status_code}]")

    content_type = r.headers.get("Content-Type", "")
    print(f"  [sharepoint] Content-Type: {content_type!r}")

    # ── Step 1: Try Content-Disposition header ────────────────────────────────
    cd    = r.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.IGNORECASE)
    filename = match.group(1).strip() if match else ""

    # ── Step 2: Try extracting filename from URL path ─────────────────────────
    if not filename or filename in ("download", ""):
        parsed     = urllib.parse.urlparse(url)
        path_parts = parsed.path.split("/")
        for part in reversed(path_parts):
            decoded = urllib.parse.unquote(part)
            if decoded.endswith((".xlsx", ".pdf", ".xlsm")):
                filename = decoded
                break

    # ── Step 3: Final fallback — use content type ─────────────────────────────
    if not filename or filename in ("download", ""):
        if "spreadsheet" in content_type or "excel" in content_type:
            ext = ".xlsx"
        elif "pdf" in content_type:
            ext = ".pdf"
        else:
            ext = ".xlsx" if ":x:" in url else ".pdf"
        filename = f"sp_download_{int(time.time())}{ext}"

    # ── Step 4: Force correct extension based on Content-Type ─────────────────
    if "spreadsheet" in content_type or "excel" in content_type:
        if not filename.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            filename = Path(filename).stem + ".xlsx"

    # ── Step 5: Force .xlsx if saving to templates folder ────────────────────
    if output_dir and "template" in output_dir.lower():
        if not filename.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            filename = Path(filename).stem + ".xlsx"

    save_path = str(Path(out_dir) / filename)
    with open(save_path, "wb") as f:
        f.write(r.content)

    print(f"  [sharepoint] Saved as: {save_path}")
    return save_path
# ── Provider registry ──────────────────────────────────────────────────────────

def _extract_cohere_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Cohere Command R+ on Azure — same interface as Llama Azure."""
    return _extract_llama_azure(pdf_path, model_cfg, prompt, positions)


PROVIDERS: dict[str, Any] = {
    "mistral_azure": _extract_mistral_azure,
    "openai_azure":  _extract_openai_azure,
    "openai_azure_responses":  _extract_openai_azure_responses,
    "phi_azure":     _extract_phi_azure,
    "llama_azure":   _extract_llama_azure,
    "cohere_azure":  _extract_cohere_azure,
}


def extract(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None, schema: dict = None) -> dict:
    """Route extraction to the correct Azure provider."""
    provider = model_cfg.get("provider")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(PROVIDERS.keys())}"
        )
    print(f"  [{provider}] Extracting with model: {model_cfg.get('model_name')}")
    return PROVIDERS[provider](pdf_path, model_cfg, prompt, positions)