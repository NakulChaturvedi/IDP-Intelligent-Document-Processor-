"""
core/filler.py
Fills a customer Excel template with extracted PO data and confidence scores.
Uses a temp file to avoid Windows file-lock errors.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import PatternFill


# ── Confidence helpers ────────────────────────────────────────────────────────

def conf_label(score: float | None) -> str:
    if score is None:
        return "N/A"
    pct = int(round(score * 100))
    if score >= 0.85:
        return f"{pct}%  High"
    if score >= 0.55:
        return f"{pct}%  Medium"
    return f"{pct}%  Low"


def conf_fill(score: float | None) -> PatternFill | None:
    if score is None:
        return None
    if score >= 0.85:
        return PatternFill("solid", start_color="C6EFCE")   # green
    if score >= 0.55:
        return PatternFill("solid", start_color="FFEB9C")   # yellow
    return PatternFill("solid", start_color="FFC7CE")        # red


# ── Merged-cell safe writer ───────────────────────────────────────────────────

def _resolve_cell(ws, row: int, col: int):
    """Return the writable top-left cell for any merged region covering (row, col)."""
    for merge in ws.merged_cells.ranges:
        if (merge.min_row <= row <= merge.max_row
                and merge.min_col <= col <= merge.max_col):
            return ws.cell(merge.min_row, merge.min_col)
    return ws.cell(row, col)


def _write(ws, row: int, col: int, value, fill: PatternFill | None = None) -> None:
    cell = _resolve_cell(ws, row, col)
    cell.value = value
    if fill:
        cell.fill = fill


# ── Main filler ───────────────────────────────────────────────────────────────

def fill_template(
    po:            dict,
    positions:     dict,
    template_path: str,
    output_path:   str,
    sheet_name:    str,
) -> None:
    """
    Copy the blank template and fill in extracted PO values + confidence.
    Writes to a temp file first, then atomically replaces output_path to
    avoid Windows file-lock errors when overwriting an existing file.
    """
    out_dir = Path(output_path).parent
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx", dir=str(out_dir))
    os.close(tmp_fd)

    try:
        shutil.copy2(template_path, tmp_path)
        wb = load_workbook(tmp_path)
        ws = wb[sheet_name]

        # ── Header fields ─────────────────────────────────────────────────────
        for key, label, row, val_col, conf_col in positions["header_fields"]:
            field = po.get(key, {})
            val   = field.get("value")      if isinstance(field, dict) else None
            conf  = field.get("confidence") if isinstance(field, dict) else None
            _write(ws, row, val_col,  str(val) if val is not None else "")
            _write(ws, row, conf_col, conf_label(conf), fill=conf_fill(conf))

        # ── Line items ────────────────────────────────────────────────────────
        li_row = positions["li_header_row"]
        if li_row is None:
            print("  WARNING: Line items table not found in template — skipping.")
        else:
            for i, item in enumerate(po.get("line_items", [])):
                r    = li_row + 1 + i
                conf = item.get("confidence")
                fill = conf_fill(conf)
                for key, label, col in positions["li_fields"]:
                    if key == "confidence":
                        _write(ws, r, col, conf_label(conf), fill=fill)
                    else:
                        val = item.get(key)
                        _write(ws, r, col, str(val) if val is not None else "", fill=fill)

        wb.save(tmp_path)
        wb.close()

        # Atomically replace the output file
        if Path(output_path).exists():
            os.remove(output_path)
        shutil.move(tmp_path, output_path)
        print(f"  Saved → {output_path}")

    except Exception:
        try:
            if Path(tmp_path).exists():
                os.remove(tmp_path)
        except OSError:
            pass
        raise
