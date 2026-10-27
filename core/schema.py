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

    field_keys    = {f[0] for f in positions["header_fields"]}
    li_field_keys = {f[0] for f in positions["li_fields"]}

    lines = [
        "You are a purchase order data extraction assistant.",
        "Read the document and fill in ONLY the following fields.",
        "Do not invent or add extra fields.",
        "",
        "HEADER FIELDS (return each as {value, confidence}):",
    ]
    for key, label, *_ in positions["header_fields"]:
        lines.append(f"- {key}: {label}")

    # ── Dynamic layout hints (only added when the template has these fields) ──
    layout_hints = []

    if {"expected_arrival", "tax_id"} <= field_keys:
        layout_hints.append(
            "- The terms row has columns: Reference | Tax ID# | Expected Arrival | Ship Via | FOB | Terms\n"
            "  Reference, Tax ID# and Expected Arrival may be blank. Return null for blank ones.\n"
            "  Do not shift values left to fill blank columns."
        )

    if "part_number_description" in li_field_keys:
        layout_hints.append(
            "- Each line item prints the part number on one line and the description on the next.\n"
            "  Put both in part_number_description, part number first, then the description."
        )

    if "ship_via" in field_keys:
        layout_hints.append(
            "- ship_via is the complete shipping method text (e.g. 'Ship Best Weight Method'). "
            "Do not truncate it."
        )

    if "vendor_number" in field_keys and "reference" in field_keys:
        layout_hints.append(
            "- This PO has a 4-column reference row: Reference | Vendor Number | Terms | Ship Via\n"
            "  Extract each column independently — never merge adjacent values\n"
            "  Vendor Number is ONLY the vendor/supplier code (e.g. 'ITTWAT') — not the reference\n"
            "  Reference is ONLY the reference code (e.g. 'Q26-1002 Hunter') — not the vendor number"
        )

    if "fob" in field_keys and "ship_via" in field_keys:
        layout_hints.append(
            "- This PO has a 4-column terms row: Terms | Expected Date | Ship Via | FOB\n"
            "  Ship Via is ONLY the shipping method (e.g. 'best way') — not FOB\n"
            "  FOB is ONLY the delivery point (e.g. 'Your Shop') — not the shipping method\n"
            "  'pp & c' is part of Ship Via, not FOB — keep it in Ship Via"
        )

    if any(k in ("no", "no_", "item_no", "line_no") for k in li_field_keys):
        layout_hints.append(
            "- 'No.' in line items means the LINE ITEM NUMBER or ITEM CODE "
            "(e.g. 'P57000') — NOT the Purchase Order number\n"
            "  The Purchase Order number appears in the header section only\n"
            "  Line item 'No.' values are short codes like 'P57000', '5K320', '1L93'"
        )

    if "gst_amount" in field_keys or "pst_amount" in field_keys:
        layout_hints.append(
            "- GST Amount and PST Amount are tax totals in the Tax Summary section\n"
            "  Extract numeric values only (e.g. '411.05', '0.00')\n"
            "  Do not confuse with line item tax codes (H, GST, PST)"
        )

    if "subtotal" in field_keys:
        layout_hints.append(
            "- Subtotal is the sum before taxes — distinct from Total which includes taxes"
        )

    if layout_hints:
        lines += ["", "COLUMN SEPARATION RULES:"]
        lines += layout_hints

    # ── Blank field rules ─────────────────────────────────────────────────────
    lines += [
        "",
        "BLANK FIELD RULES:",
        "  - A field's value must be printed next to, or directly under, its own label.",
        "  - If a label has no value printed for it (the next thing is another label, or empty space),",
        "    set value to null.",
        "  - Never fill a blank field with the PO number, a page number, or any value that belongs",
        "    to a different label.",
        "  - IDs, order numbers, reference numbers, page numbers and dates must never be copied",
        "    from one field to another.",
        "  - The same value may appear in two fields only for phone, fax, email and address,",
        "    and only when it is printed for both.",
    ]

    # ── Confidence scoring rules ──────────────────────────────────────────────
    lines += [
        "",
        "CONFIDENCE SCORING RULES:",
        "  1.0 = value clearly and explicitly printed for that field",
        "  1.0 = field is blank or absent in this PO (value null) — you are sure there is nothing there",
        "  0.7 = value present but partially ambiguous",
        "  0.4 = value inferred or uncertain",
        "  0.0 = a value is clearly printed but is illegible or cut off, so you cannot read it",
        "  Never lower the confidence of a null field just because the label is present.",
    ]

    # ── Line items ────────────────────────────────────────────────────────────
    lines += [
        "",
        "LINE ITEMS — extract EVERY product row from the line items table:",
        "  - The line items table starts AFTER the header fields",
        "  - 'No.' or 'Ln' in line items = item/line number or item code (e.g. 'P57000') — NOT the PO number",
        "  - Each row in the table is one line item — extract all rows",
        "  - Quantity and Unit may appear as one combined cell (e.g. '6 Each') — split them if needed",
        "  - Unit Price and Line Amount are separate columns — do not merge them",
        "  - If item codes are missing for some rows, still extract quantity, price and description",
        "  - Skip only: Subtotal, Invoice Discount, Total Tax, Total USD summary rows",
        "  - Do not skip rows just because item code is blank",
        "",
        "For each product row extract these fields exactly as named:",
    ]
    for key, label, _ in positions["li_fields"]:
        if key != "confidence":
            lines.append(f"    * {key}: {label}")

    lines += ["", "Return null for any header field value not found in the document."]
    return "\n".join(lines)
