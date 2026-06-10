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
        "  1.0 = clearly and explicitly stated\n"
        "  0.7 = present but partially ambiguous\n"
        "  0.4 = inferred or uncertain\n"
        "  0.0 = not found (set value to null)\n"
        "Use null for any field value not found."
    )


# ── Company detection ─────────────────────────────────────────────────────────

def detect_company_from_pdf(pdf_path: str, model_cfg: dict) -> str:
    """
    Extract the buying company name from a PO PDF using text parsing.
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
               ["phone", "fax", "attn", "http", "www", "po box",
                "p.o. box", "gst", "hst", "pst"]):
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
                return line

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


# ── Mistral Azure — native PDF OCR ────────────────────────────────────────────

def _extract_mistral_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Mistral Document AI on Azure — uses native OCR annotation API."""
    pdf_b64 = base64.standard_b64encode(Path(pdf_path).read_bytes()).decode("utf-8")

    schema = _build_mistral_schema(positions)

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}",
    }
    payload = {
        "model": model_cfg["model_name"],
        "document": {
            "type":         "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        "document_annotation_prompt": prompt,
        "document_annotation_format": schema,
        "include_image_base64":       False,
    }

    def call():
        resp = requests.post(
            model_cfg["endpoint"].rstrip("/"),
            headers=headers,
            json=payload,
            timeout=model_cfg.get("timeout", 300),
        )
        return resp.text, resp.status_code

    raw, _ = _retry(call, model_cfg.get("retries", 3))
    data   = json.loads(raw)
    ann    = data.get("document_annotation")
    if not ann:
        raise ValueError("No document_annotation in Azure response.")
    result = json.loads(ann) if isinstance(ann, str) else ann
    return _normalize_response(result, positions, estimate_confidence=False)


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
    text              = _extract_pdf_text(pdf_path)
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
    return _normalize_response(result, positions, estimate_confidence=False)


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
    return _normalize_response(result, positions, estimate_confidence=False)


# ── Llama Azure — Azure AI chat completions ───────────────────────────────────

def _extract_llama_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    text               = _extract_pdf_text(pdf_path)
    schema_instruction = _build_schema_instruction(positions)

    # Build focused line items field list
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
    return _normalize_response(result, positions, estimate_confidence=True)


# ── Provider registry ─────────────────────────────────────────────────────────
# Add new Azure providers here — key must match `provider` in config.yaml
def _extract_cohere_azure(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Cohere Command R+ on Azure — same interface as Llama Azure."""
    return _extract_llama_azure(pdf_path, model_cfg, prompt, positions)

PROVIDERS: dict[str, Any] = {
    "mistral_azure": _extract_mistral_azure,
    "openai_azure":  _extract_openai_azure,
    "phi_azure":     _extract_phi_azure,
    "llama_azure":   _extract_llama_azure,
    "cohere_azure":  _extract_cohere_azure,   # ← add this
}


def extract(pdf_path: str, model_cfg: dict, prompt: str, positions: dict = None) -> dict:
    """Route extraction to the correct Azure provider."""
    provider = model_cfg.get("provider")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(PROVIDERS.keys())}"
        )
    print(f"  [{provider}] Extracting with model: {model_cfg.get('model_name')}")
    return PROVIDERS[provider](pdf_path, model_cfg, prompt, positions)