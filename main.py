"""
main.py — PO Processor API (Azure)
Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from core.filler    import fill_template
from core.template  import find_template_by_company, find_template_by_po, scan_template
from core.security import log_usage, verify_token
from core.extractor import detect_company_from_pdf, extract

# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PO Processor API — Azure",
    description = "Extract purchase order data using Mistral Azure / Mistral API models.",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    app.state.cfg = CFG


# ── Request / Response models ─────────────────────────────────────────────────

class DetectCompanyRequest(BaseModel):
    customer_po_path: str          # Full path to the PO PDF


class DetectCompanyResponse(BaseModel):
    customer_po_path: str
    company_name:     Optional[str]
    template_found:   Optional[str]


class ProcessRequest(BaseModel):
    customer_po_path:          str
    customer_po_template_path: Optional[str] = None   # override auto-matched template
    model:                     Optional[str] = None   # override active model
    output_dir:                Optional[str] = None


class BatchRequest(BaseModel):
    po_folder:                 Optional[str] = None
    customer_po_template_path: Optional[str] = None
    model:                     Optional[str] = None
    output_dir:                Optional[str] = None


class POResult(BaseModel):
    customer_po_path:          str
    customer_po_template_path: str
    model_used:                str
    output_file:               str
    company_name:              Optional[str]
    fields_filled:             int
    line_items:                int
    duration_sec:              float
    status:                    str
    error:                     Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model(model_key: Optional[str]) -> tuple[str, dict]:
    key = (model_key or "").strip() or list(CFG["models"].keys())[0]
    if key not in CFG["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{key}' not found. Available: {list(CFG['models'].keys())}",
        )
    return key, CFG["models"][key]


def _process_single(
    customer_po_path:          str,
    customer_po_template_path: Optional[str],
    model_key:                 Optional[str],
    output_dir:                Optional[str],
    user:                      str = "unknown",
) -> POResult:
    t0 = time.time()

    po_file = Path(customer_po_path.strip('"').strip("'"))
    if not po_file.is_absolute():
        po_file = Path(CFG["folders"]["po_input"]) / customer_po_path
    if not po_file.exists():
        raise HTTPException(status_code=404, detail=f"PO file not found: {po_file}")

    out_dir = Path(output_dir or CFG["folders"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{po_file.stem}_extracted.xlsx")

    model_key, model_cfg = _resolve_model(model_key)

    # ── Step 1: Detect company name ───────────────────────────────────────────
    company_name = None
    tmpl         = customer_po_template_path

    if not tmpl:
        print(f"  Detecting company name from PO...")
        company_name = detect_company_from_pdf(str(po_file))
        print(f"  Company detected: {company_name}")

        if company_name:
            # find_template_by_company returns {"matched": bool, "template_path": str|None}
            match = find_template_by_company(company_name, CFG["folders"]["templates"])
            tmpl  = match["template_path"] if match and match.get("matched") else None
            print(f"  Template match: {tmpl or 'none'}")

        if not tmpl:
            tmpl = find_template_by_po(po_file.name, CFG["folders"]["templates"])
            print(f"  Template (PO fallback): {tmpl or 'none'}")

    if not tmpl:
        log_usage(
            cfg=CFG, user=user, endpoint="/process", model=model_key,
            po_file=str(po_file), status="error",
            duration=round(time.time() - t0, 2),
            error=f"No template found for '{company_name or po_file.name}'",
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"No template found for company '{company_name or po_file.name}'. "
                f"Add a template to '{CFG['folders']['templates']}' "
                f"in a subfolder named after the company."
            ),
        )

    tmpl = str(tmpl).strip('"').strip("'")

    print(f"\n── Processing: {po_file.name}")
    print(f"   Company  : {company_name or 'unknown'}")
    print(f"   Template : {tmpl}")
    print(f"   Model    : {model_key} ({model_cfg['display_name']})")

    # ── Step 2: Scan template & extract ──────────────────────────────────────
    positions = scan_template(tmpl, CFG)

    from core.schema import build_schema, build_prompt
    schema = build_schema(positions)
    prompt = build_prompt(positions)

    # FIX: correct arg order — extract(pdf_path, model_cfg, prompt, positions)
    po_data = extract(str(po_file), model_cfg, prompt, positions)

    # ── Step 3: Fill template ─────────────────────────────────────────────────
    fill_template(
        po            = po_data,
        positions     = positions,
        template_path = tmpl,
        output_path   = output_path,
        sheet_name    = CFG["template"]["sheet_name"],
    )

    filled   = sum(1 for k, v in po_data.items() if k != "line_items" and v)
    li_count = len(po_data.get("line_items", []))
    duration = round(time.time() - t0, 2)

    # ── Log success ───────────────────────────────────────────────────────────
    log_usage(
        cfg=CFG, user=user, endpoint="/process", model=model_key,
        po_file=str(po_file), status="success", duration=duration,
        fields=filled, line_items=li_count,
    )

    return POResult(
        customer_po_path          = str(po_file),
        customer_po_template_path = tmpl,
        model_used                = model_key,
        output_file               = output_path,
        company_name              = company_name,
        fields_filled             = filled,
        line_items                = li_count,
        duration_sec              = duration,
        status                    = "success",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def frontend():
    html_path = Path(__file__).parent / "frontend.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", summary="List available models")
def list_models():
    """Returns all models defined in config.yaml."""
    CFG.update(load_config())
    return {
        "models": {
            key: {
                "provider":     m["provider"],
                "display_name": m["display_name"],
                "model_name":   m.get("model_name", ""),
            }
            for key, m in CFG["models"].items()
        }
    }


@app.post("/detect-company", response_model=DetectCompanyResponse, summary="Detect company name from PO")
def detect_company(req: DetectCompanyRequest, user: str = Depends(verify_token)):
    t0      = time.time()
    po_path = req.customer_po_path.strip('"').strip("'")
    if not Path(po_path).exists():
        raise HTTPException(status_code=404, detail=f"PO file not found: {po_path}")

    try:
        company_name = detect_company_from_pdf(po_path)
        match        = find_template_by_company(company_name, CFG["folders"]["templates"])
        template_found = match["template_path"] if match and match.get("matched") else None

        log_usage(
            cfg=CFG, user=user, endpoint="/detect-company", model="none",
            po_file=po_path, status="success", duration=round(time.time() - t0, 2),
        )
        return DetectCompanyResponse(
            customer_po_path = po_path,
            company_name     = company_name,
            template_found   = template_found,
        )
    except Exception as e:
        log_usage(
            cfg=CFG, user=user, endpoint="/detect-company", model="none",
            po_file=po_path, status="error", duration=round(time.time() - t0, 2),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=POResult, summary="Process a single PO")
def process_po(req: ProcessRequest, user: str = Depends(verify_token)):
    """
    Full pipeline: detect company → match template → extract → fill Excel.

    - **customer_po_path**: full path to the PO PDF
    - **customer_po_template_path**: (optional) override auto-matched template
    - **model**: (optional) model key from config
    - **output_dir**: (optional) where to save the output Excel
    """
    try:
        return _process_single(
            customer_po_path          = req.customer_po_path,
            customer_po_template_path = req.customer_po_template_path,
            model_key                 = req.model,
            output_dir                = req.output_dir,
            user                      = user,
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ /process EXCEPTION:\n{tb}")
        log_usage(
            cfg=CFG, user=user, endpoint="/process", model=req.model or "auto",
            po_file=req.customer_po_path, status="error", duration=0, error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n\n{tb}")


@app.post("/batch", summary="Process all POs in a folder")
def batch_process(req: BatchRequest, user: str = Depends(verify_token)):
    """
    Process every PDF in a folder.
    Company name is detected per PO and used to match templates automatically.
    """
    po_folder = Path(req.po_folder or CFG["folders"]["po_input"])
    if not po_folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {po_folder}")

    pdfs = sorted(po_folder.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail=f"No PDFs found in: {po_folder}")

    results   = []
    successes = 0
    failures  = 0

    for pdf in pdfs:
        try:
            result = _process_single(
                customer_po_path          = str(pdf),
                customer_po_template_path = req.customer_po_template_path,
                model_key                 = req.model,
                output_dir                = req.output_dir,
                user                      = user,
            )
            results.append(result)
            successes += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n❌ /batch EXCEPTION on {pdf.name}:\n{tb}")
            log_usage(
                cfg=CFG, user=user, endpoint="/batch", model=req.model or "auto",
                po_file=str(pdf), status="error", duration=0, error=str(e),
            )
            results.append(POResult(
                customer_po_path          = str(pdf),
                customer_po_template_path = req.customer_po_template_path or "auto",
                model_used                = req.model or "auto",
                output_file               = "",
                company_name              = None,
                fields_filled             = 0,
                line_items                = 0,
                duration_sec              = 0,
                status                    = "error",
                error                     = str(e),
            ))
            failures += 1

    return {
        "total":     len(pdfs),
        "succeeded": successes,
        "failed":    failures,
        "results":   results,
    }


@app.get("/download", summary="Download a processed Excel file")
def download_file(path: str = Query(..., description="Full path to output Excel")):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(
        path       = str(p),
        filename   = p.name,
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/config", summary="View config (keys redacted)")
def get_config():
    CFG.update(load_config())
    safe = {"folders": CFG["folders"], "models": {}}
    for k, m in CFG["models"].items():
        safe["models"][k] = {
            "provider":     m["provider"],
            "display_name": m["display_name"],
            "model_name":   m.get("model_name", ""),
            "api_key":      "***" if "api_key" in m else None,
        }
    return safe


@app.post("/reload-config", summary="Reload config.yaml without restarting")
def reload_config():
    CFG.update(load_config())
    return {"status": "reloaded"}