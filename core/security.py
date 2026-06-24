"""
core/security.py
API authentication and usage tracking.

- Validates Bearer tokens on every request
- Logs every API call to a running Excel file
- Tracks: timestamp, user, endpoint, model, PO file, tokens, duration, status
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ── Thread lock for Excel writes ──────────────────────────────────────────────
_excel_lock = threading.Lock()

# ── Auth scheme ───────────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)


# ── Load tokens from config ───────────────────────────────────────────────────

def _load_tokens(cfg: dict) -> dict[str, str]:
    """
    Returns {token: user_label} from config.yaml security section.
    Tokens are stored as SHA256 hashes in config for safety.
    """
    return cfg.get("security", {}).get("tokens", {})


def verify_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """
    FastAPI dependency — validates Bearer token on every request.
    Returns the user label associated with the token.
    Raises 401 if token is missing or invalid.
    """
    cfg = request.app.state.cfg

    # Skip auth if security is disabled in config
    if not cfg.get("security", {}).get("enabled", True):
        return "anonymous"

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token. Include 'Authorization: Bearer <token>' header.",
        )

    token    = credentials.credentials
    tokens   = _load_tokens(cfg)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # Check plain token match first, then hash match
    if token in tokens:
        return tokens[token]
    if token_hash in tokens:
        return tokens[token_hash]

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication token.",
    )

def get_usage_summary(cfg: dict) -> dict:
    """Returns aggregated usage stats per user from the log Excel."""
    from openpyxl import load_workbook

    log_path = Path(cfg.get("security", {}).get("usage_log", "usage_log.xlsx"))
    if not log_path.exists():
        return {"total_calls": 0, "by_user": {}}

    wb = load_workbook(log_path, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(min_row=2, values_only=True))
    headers = [c.value for c in ws[1]]

    by_user = {}
    total   = 0
    for row in rows:
        record = dict(zip(headers, row))
        user   = record.get("User", "unknown")
        if user not in by_user:
            by_user[user] = {"calls": 0, "success": 0, "errors": 0, "po_files": 0, "line_items": 0}
        by_user[user]["calls"] += 1
        if record.get("Status") == "success":
            by_user[user]["success"] += 1
            by_user[user]["po_files"] += 1
            by_user[user]["line_items"] += record.get("Line Items", 0) or 0
        else:
            by_user[user]["errors"] += 1
        total += 1

    return {"total_calls": total, "by_user": by_user}
# ── Usage logger ──────────────────────────────────────────────────────────────

def log_usage(
    cfg:        dict,
    user:       str,
    endpoint:   str,
    model:      str,
    po_file:    str,
    status:     str,
    duration:   float,
    fields:     int   = 0,
    line_items: int   = 0,
    error:      str   = "",
) -> None:
    """
    Append one row to the usage Excel log.
    Thread-safe — multiple requests can log simultaneously.
    """
    log_path = cfg.get("security", {}).get("usage_log", "usage_log.xlsx")
    log_path = Path(log_path)

    row = {
        "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User":        user,
        "Endpoint":    endpoint,
        "Model":       model,
        "PO File":     Path(po_file).name if po_file else "",
        "Status":      status,
        "Duration (s)": round(duration, 2),
        "Fields Filled": fields,
        "Line Items":  line_items,
        "Error":       error[:200] if error else "",
    }

    with _excel_lock:
        _append_to_excel(log_path, row)


def _append_to_excel(log_path: Path, row: dict) -> None:
    """Create or append to the usage Excel log."""
    try:
        from openpyxl import load_workbook, Workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        headers = list(row.keys())

        if log_path.exists():
            wb = load_workbook(log_path)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "Usage Log"

            # Write headers
            for col, header in enumerate(headers, 1):
                cell = ws.cell(1, col, header)
                cell.font      = Font(bold=True, color="FFFFFF")
                cell.fill      = PatternFill("solid", start_color="1F3864")
                cell.alignment = Alignment(horizontal="center")

            # Set column widths
            widths = [20, 15, 25, 25, 35, 10, 12, 14, 12, 40]
            for col, width in enumerate(widths, 1):
                ws.column_dimensions[ws.cell(1, col).column_letter].width = width

        # Append data row
        next_row = ws.max_row + 1
        for col, key in enumerate(headers, 1):
            ws.cell(next_row, col, row[key])

        # Color status cell
        status_col = headers.index("Status") + 1
        status_cell = ws.cell(next_row, status_col)
        if row["Status"] == "success":
            status_cell.fill = PatternFill("solid", start_color="C6EFCE")
            status_cell.font = Font(color="276221")
        else:
            status_cell.fill = PatternFill("solid", start_color="FFC7CE")
            status_cell.font = Font(color="9C0006")

        wb.save(log_path)

    except Exception as e:
        # Never let logging errors break the main request
        print(f"  WARNING: Failed to write usage log: {e}")


# ── Token management helpers ──────────────────────────────────────────────────

def generate_token(label: str) -> tuple[str, str]:
    """
    Generate a new API token for a user.
    Returns (plain_token, hashed_token) — store the hash in config, give plain to user.
    """
    import secrets
    token      = f"po-{secrets.token_urlsafe(32)}"
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    print(f"\n  New token for '{label}':")
    print(f"  Plain token (give to user): {token}")
    print(f"  Hash (store in config.yaml): {token_hash}\n")
    return token, token_hash
