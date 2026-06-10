"""
main.py — PO Processor API (Azure Edition)
Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
Frontend:  http://localhost:8000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from core.extractor import detect_company_from_pdf, extract
from core.filler    import fill_template
from core.schema    import build_prompt
from core.template  import find_template_by_company, scan_template


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PO Processor API — Azure Edition",
    description = "Extract PO data using Azure-hosted models (Mistral, GPT-4o, Phi, Llama).",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class DetectCompanyRequest(BaseModel):
    customer_po_path: str
    model:            Optional[str] = None


class DetectCompanyResponse(BaseModel):
    company_name:     str
    template_matched: bool
    template_path:    Optional[str]
    template_name:    Optional[str]
    match_score:      int
    all_templates:    list[str]
    model_used:       str


class ProcessRequest(BaseModel):
    customer_po_path:          str
    customer_po_template_path: Optional[str] = None
    model:                     Optional[str] = None
    output_dir:                Optional[str] = None


class BatchRequest(BaseModel):
    po_folder:                 Optional[str] = None
    customer_po_template_path: Optional[str] = None
    model:                     Optional[str] = None
    output_dir:                Optional[str] = None


class POResult(BaseModel):
    po_file:       str
    company_name:  Optional[str]
    template_used: str
    model_used:    str
    output_file:   str
    fields_filled: int
    line_items:    int
    duration_sec:  float
    status:        str
    error:         Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model(model_key: Optional[str]) -> tuple[str, dict]:
    key = (model_key or "").strip() or list(CFG["models"].keys())[0]
    if key not in CFG["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{key}' not found. Available: {list(CFG['models'].keys())}",
        )
    return key, CFG["models"][key]


def _clean_path(path: str) -> str:
    return path.strip().strip('"').strip("'")


def _process_single(
    customer_po_path:          str,
    customer_po_template_path: Optional[str],
    model_key:                 Optional[str],
    output_dir:                Optional[str],
) -> POResult:
    t0 = time.time()

    po_file = Path(_clean_path(customer_po_path))
    if not po_file.is_absolute():
        po_file = Path(CFG["folders"]["po_input"]) / customer_po_path
    if not po_file.exists():
        raise HTTPException(status_code=404, detail=f"PO file not found: {po_file}")

    out_dir = Path(_clean_path(output_dir) if output_dir else CFG["folders"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{po_file.stem}_extracted.xlsx")

    model_key, model_cfg = _resolve_model(model_key)

    print(f"\n── Processing: {po_file.name}")
    print(f"   Model    : {model_key} ({model_cfg['display_name']})")

    # ── Step 1: Detect company ────────────────────────────────────────────────
    tmpl = _clean_path(customer_po_template_path) if customer_po_template_path else None

    print("   Detecting company name...")
    company_name = detect_company_from_pdf(str(po_file), model_cfg)
    print(f"   Company  : {company_name}")

    if not tmpl and company_name:
        match = find_template_by_company(company_name, CFG["folders"]["templates"])
        if match["matched"]:
            tmpl = match["template_path"]

    if not tmpl:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No template found for '{company_name or po_file.name}'. "
                "Pass customer_po_template_path to override."
            ),
        )

    print(f"   Template : {tmpl}")

    # ── Step 2: Scan template & build prompt ──────────────────────────────────
    positions = scan_template(tmpl, CFG)
    prompt    = build_prompt(positions)

    # ── Step 3: Extract ───────────────────────────────────────────────────────
    po_data = extract(str(po_file), model_cfg, prompt, positions)

    # ── Step 4: Fill Excel ────────────────────────────────────────────────────
    fill_template(
        po            = po_data,
        positions     = positions,
        template_path = tmpl,
        output_path   = output_path,
        sheet_name    = CFG["template"]["sheet_name"],
    )

    filled   = sum(1 for k, v in po_data.items() if k != "line_items" and v)
    li_count = len(po_data.get("line_items", []))

    return POResult(
        po_file       = str(po_file),
        company_name  = company_name,
        template_used = tmpl,
        model_used    = model_key,
        output_file   = output_path,
        fields_filled = filled,
        line_items    = li_count,
        duration_sec  = round(time.time() - t0, 2),
        status        = "success",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def frontend():
    return (Path(__file__).parent / "frontend.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", summary="List available Azure models")
def list_models():
    CFG.update(load_config())
    return {
        "models": {
            key: {
                "provider":     m["provider"],
                "display_name": m["display_name"],
                "model_name":   m.get("model_name", ""),
                "endpoint":     m.get("endpoint", "")[:50] + "...",
            }
            for key, m in CFG["models"].items()
        }
    }


@app.post("/detect-company", response_model=DetectCompanyResponse)
def detect_company(req: DetectCompanyRequest):
    """Step 1 — detect company name from PO and find matching template."""
    try:
        po_path = _clean_path(req.customer_po_path)
        if not Path(po_path).exists():
            raise HTTPException(status_code=404, detail=f"PO not found: {po_path}")

        model_key, model_cfg = _resolve_model(req.model)
        company_name         = detect_company_from_pdf(po_path, model_cfg)
        match                = find_template_by_company(company_name, CFG["folders"]["templates"])

        return DetectCompanyResponse(
            company_name     = company_name,
            template_matched = match["matched"],
            template_path    = match["template_path"],
            template_name    = match["template_name"],
            match_score      = match["score"],
            all_templates    = match["all_templates"],
            model_used       = model_key,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-customer-po", response_model=POResult)
def process_customer_po(req: ProcessRequest):
    """Step 2 — full extraction pipeline."""
    try:
        return _process_single(
            customer_po_path          = req.customer_po_path,
            customer_po_template_path = req.customer_po_template_path,
            model_key                 = req.model,
            output_dir                = req.output_dir,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-customer-po", summary="Process all POs in a folder")
def batch_customer_po(req: BatchRequest):
    po_folder = Path(_clean_path(req.po_folder) if req.po_folder else CFG["folders"]["po_input"])
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
            )
            results.append(result)
            successes += 1
        except Exception as e:
            results.append(POResult(
                po_file       = str(pdf),
                company_name  = None,
                template_used = req.customer_po_template_path or "auto",
                model_used    = req.model or "auto",
                output_file   = "",
                fields_filled = 0,
                line_items    = 0,
                duration_sec  = 0,
                status        = "error",
                error         = str(e),
            ))
            failures += 1

    return {"total": len(pdfs), "succeeded": successes, "failed": failures, "results": results}


@app.get("/download")
def download_file(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(
        path       = str(p),
        filename   = p.name,
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/config")
def get_config():
    CFG.update(load_config())
    safe = {"folders": CFG["folders"], "models": {}}
    for k, m in CFG["models"].items():
        safe["models"][k] = {
            "provider":     m["provider"],
            "display_name": m["display_name"],
            "model_name":   m.get("model_name", ""),
            "api_key":      "***",
            "endpoint":     m.get("endpoint", "")[:60] + "...",
        }
    return safe


@app.post("/reload-config")
def reload_config():
    CFG.update(load_config())
    return {"status": "reloaded"}
