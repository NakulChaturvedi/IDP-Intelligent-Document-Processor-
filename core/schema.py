"""
core/schema.py
Builds the Mistral JSON schema and extraction prompt from scanned template positions.
"""

from __future__ import annotations


def build_schema(positions: dict) -> dict:
    """Auto-generate the Mistral JSON schema from discovered template fields."""
    props: dict = {}

    for key, label, *_ in positions["header_fields"]:
        props[key] = {
            "type": "object",
            "properties": {
                "value":      {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "description": label,
        }

    li_props: dict = {}
    for key, label, _ in positions["li_fields"]:
        if key == "confidence":
            li_props["confidence"] = {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score for this row (0.0–1.0)",
            }
        else:
            li_props[key] = {"type": ["string", "null"], "description": label}

    # Always ensure confidence exists on line items
    if "confidence" not in li_props:
        li_props["confidence"] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score for this row (0.0–1.0)",
        }

    props["line_items"] = {
        "type": "array",
        "description": "All product/order rows from the document.",
        "items": {"type": "object", "properties": li_props},
    }

    required_keys = [f[0] for f in positions["header_fields"]] + ["line_items"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "purchase_order_template",
            "schema": {
                "type": "object",
                "properties": props,
                "required": required_keys,
            },
        },
    }


def build_prompt(positions: dict) -> str:
    """Auto-generate the extraction prompt from discovered template fields."""
    lines = [
        "You are a purchase order data extraction assistant.",
        "Read the document and fill in ONLY the following fields.",
        "Do not invent or add extra fields.",
        "",
        "HEADER FIELDS (return each as {value, confidence}):",
    ]
    for key, label, *_ in positions["header_fields"]:
        lines.append(f"- {key}: {label}")

    lines += [
    "",
    "LINE ITEMS — extract EVERY product row with ALL sub-lines:",
    "  - Each product has 2-3 lines:",
    "    Line 1: Item code (e.g. '15SH10K6') — goes in item field",
    "    Line 2: Description (e.g. 'CENT. PUMP 3500RPM') — goes in description field",
    "    Line 3: Pricing detail (e.g. '$ 1,923 x 0.54 (F)') — goes in pricing_detail field",
    "  - The '$ X x Y (Z)' line shows: list price x multiplier (tax code)",
    "  - Include this pricing detail line — do NOT skip it",
    "  - For Item ID Item Description field: combine the item code AND its full description",
    "    e.g. '10K55 Shaft Seal Car/Sil Car/Vi' (item code first, then description)",
    "  - If description appears twice (repeated line), include it only once",
    "  - Part No. is explicitly labelled 'Part No.:' in the document — extract that value",
    "  - Skip ONLY: 'HST On Purchase' tax summary rows and blank rows",
    "  - 'MISC. GOODS...' in Item column means miscellaneous — keep as-is",
    "",
    "For each product row extract:",
]
    for key, label, _ in positions["li_fields"]:
        if key != "confidence":
            lines.append(f"    * {key}: {label}")

    lines += ["", "Return null for any header field value not found in the document."]
    return "\n".join(lines)
