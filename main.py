"""
main.py — PO Processor API (Azure)
Run with:  uvicorn main:app --reload --port 8001
Docs at:   http://localhost:8001/docs
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from core.extractor import detect_company_from_pdf, extract
from core.filler    import fill_template
from core.template  import find_template_by_company, find_template_by_po, scan_template
from core.security  import generate_token, log_usage, verify_token, USAGE_LOG


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PO Processor API — Azure",
    description = "Extract purchase order data using Azure models.",
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

class GenerateTokenRequest(BaseModel):
    label: str

class DetectCompanyRequest(BaseModel):
    customer_po_path: str
    model:            Optional[str] = None

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


def _match_template(company_name: str | None, po_filename: str) -> tuple[str | None, dict | None]:
    """
    Returns (template_path_str, match_dict) where match_dict is the full
    result from find_template_by_company (for returning to the frontend).
    """
    match = None
    tmpl  = None

    if company_name:
        match = find_template_by_company(company_name, CFG["folders"]["templates"])
        # match = {"matched": bool, "template_path": str|None, "template_name": str|None,
        #          "score": int, "all_templates": [str]}
        if match.get("matched"):
            tmpl = match["template_path"]

    if not tmpl:
        tmpl = find_template_by_po(po_filename, CFG["folders"]["templates"])

    return tmpl, match


def _process_single(
    customer_po_path:          str,
    customer_po_template_path: Optional[str],
    model_key:                 Optional[str],
    output_dir:                Optional[str],
    auth:                      dict,
) -> POResult:
    t0 = time.time()

    # ── Handle SharePoint or HTTP URLs for PO ─────────────────────────────────
    print(f"  [debug] customer_po_path = {customer_po_path!r}")
    print(f"  [debug] starts with http = {customer_po_path.startswith('http')}")
    if customer_po_path.startswith("http"):
        from core.extractor import download_from_sharepoint
        customer_po_path = download_from_sharepoint(
            customer_po_path,
            output_dir=str(Path(CFG["folders"]["output"]))
        )

    # ── Handle SharePoint or HTTP URLs for template ───────────────────────────
    if customer_po_template_path and customer_po_template_path.startswith("http"):
        from core.extractor import download_from_sharepoint
        tmpl_dir = Path(CFG["folders"]["templates"])
        tmpl_dir.mkdir(parents=True, exist_ok=True)
        customer_po_template_path = download_from_sharepoint(
            customer_po_template_path,
            output_dir=str(tmpl_dir)
        )

    po_file = Path(customer_po_path.strip('"').strip("'"))
    if not po_file.is_absolute():
        po_file = Path(CFG["folders"]["po_input"]) / customer_po_path
    if not po_file.exists():
        raise HTTPException(status_code=404, detail=f"PO file not found: {po_file}")
    
    # ... rest of function unchanged
    out_dir = Path(output_dir or CFG["folders"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{po_file.stem}_extracted.xlsx")

    model_key, model_cfg = _resolve_model(model_key)

    # ── Step 1: Detect company & match template ───────────────────────────────
    company_name = None
    tmpl         = customer_po_template_path   # use override if provided

    if not tmpl:
        print(f"  Detecting company name from PO...")
        # detect_company_from_pdf only needs pdf_path — model_cfg is optional/unused
        company_name = detect_company_from_pdf(str(po_file))
        print(f"  Company detected: {company_name!r}")

        tmpl, match = _match_template(company_name, po_file.name)
        print(f"  Template: {tmpl or 'none found'}")

    if not tmpl:
        err = f"No template found for company '{company_name or po_file.name}'."
        log_usage(
            token=auth["token"], user=auth["label"],
            endpoint="/process-customer-po", model=model_key,
            po_file=str(po_file), status="error",
            duration=round(time.time() - t0, 2), error=err,
        )
        raise HTTPException(status_code=404, detail=(
            f"{err} Add a template to '{CFG['folders']['templates']}' "
            f"in a subfolder named after the company."
        ))

    tmpl = str(tmpl).strip('"').strip("'")

    print(f"\n── Processing: {po_file.name}")
    print(f"   Company  : {company_name or 'unknown'}")
    print(f"   Template : {tmpl}")
    print(f"   Model    : {model_key} ({model_cfg['display_name']})")

    # ── Step 2: Scan template & extract ──────────────────────────────────────
    positions = scan_template(tmpl, CFG)

    from core.schema import build_schema, build_prompt
    prompt = build_prompt(positions)

    # extract(pdf_path, model_cfg, prompt, positions) — no schema positional arg
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

    log_usage(
        token=auth["token"], user=auth["label"],
        endpoint="/process-customer-po", model=model_key,
        po_file=str(po_file), status="success",
        duration=duration, fields=filled, line_items=li_count,
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
    return (Path(__file__).parent / "frontend.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Token generation (public) ─────────────────────────────────────────────────

@app.post("/generate-token", summary="Generate a new API token")
def generate_token_endpoint(body: GenerateTokenRequest):
    if not body.label.strip():
        raise HTTPException(status_code=400, detail="Label cannot be empty.")
    token = generate_token(body.label.strip())
    return {"token": token, "label": body.label.strip()}

@app.post("/auth/generate-token")
def create_token(label: str):
    token = generate_token(label)
    return {
        "token": token,
        "label": label
    }

# ── Models ────────────────────────────────────────────────────────────────────

@app.get("/models", summary="List available models")
def list_models(auth: dict = Depends(verify_token)):
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


# ── Detect company ────────────────────────────────────────────────────────────

@app.post("/detect-company", summary="Detect company name from PO")
def detect_company(req: DetectCompanyRequest, auth: dict = Depends(verify_token)):
    t0      = time.time()
    po_path = req.customer_po_path.strip('"').strip("'")

    po_path = req.customer_po_path.strip('"').strip("'").strip()

    if po_path.startswith("http"):
        from core.extractor import download_from_sharepoint
        try:
            po_path = download_from_sharepoint(
                po_path,
                output_dir=str(Path(CFG["folders"]["output"]))
            )
            print(f"  [detect-company] downloaded to: {po_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download from SharePoint: {e}")

    if not Path(po_path).exists():
        raise HTTPException(status_code=404, detail=f"PO file not found: {po_path}")

    try:
        # detect_company_from_pdf uses pdfplumber text parsing — no model needed
        company_name = detect_company_from_pdf(po_path)
        print(f"  [detect-company] Detected: {company_name!r}")

        tmpl, match = _match_template(company_name, Path(po_path).name)

        # Collect all available template subfolders for frontend display
        tmpl_root     = Path(CFG["folders"]["templates"])
        all_templates = (
            [p.name for p in tmpl_root.iterdir() if p.is_dir()]
            if tmpl_root.exists() else []
        )
        # Fall back to match's all_templates if no subdirs
        if not all_templates and match:
            all_templates = match.get("all_templates", [])

        log_usage(
            token=auth["token"], user=auth["label"],
            endpoint="/detect-company", model="none",
            po_file=po_path, status="success",
            duration=round(time.time() - t0, 2),
        )

        return {
            "customer_po_path": po_path,
            "company_name":     company_name,
            "template_matched": bool(tmpl and match and match.get("matched")),
            "template_path":    tmpl,
            "template_name":    match.get("template_name") if match else None,
            "match_score":      match.get("score") if match else None,
            "all_templates":    all_templates,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_usage(
            token=auth["token"], user=auth["label"],
            endpoint="/detect-company", model="none",
            po_file=po_path, status="error",
            duration=round(time.time() - t0, 2), error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


# ── Process single PO ─────────────────────────────────────────────────────────

@app.post("/process-customer-po", response_model=POResult, summary="Process a single PO")
def process_po(req: ProcessRequest, auth: dict = Depends(verify_token)):
    try:
        return _process_single(
            customer_po_path          = req.customer_po_path,
            customer_po_template_path = req.customer_po_template_path,
            model_key                 = req.model,
            output_dir                = req.output_dir,
            auth                      = auth,
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ /process-customer-po EXCEPTION:\n{tb}")
        log_usage(
            token=auth["token"], user=auth["label"],
            endpoint="/process-customer-po", model=req.model or "auto",
            po_file=req.customer_po_path, status="error", duration=0, error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n\n{tb}")


# ── Batch process ─────────────────────────────────────────────────────────────

@app.post("/batch-customer-po", summary="Process all POs in a folder")
def batch_process(req: BatchRequest, auth: dict = Depends(verify_token)):
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
                auth                      = auth,
            )
            results.append(result)
            successes += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n❌ /batch-customer-po on {pdf.name}:\n{tb}")
            log_usage(
                token=auth["token"], user=auth["label"],
                endpoint="/batch-customer-po", model=req.model or "auto",
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


# ── Download endpoints ────────────────────────────────────────────────────────

@app.get("/download", summary="Download a processed Excel file")
def download_file(path: str = Query(...)):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(
        path=str(p), filename=p.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/download-usage-log", summary="Download the usage log Excel")
def download_usage_log(auth: dict = Depends(verify_token)):
    if not USAGE_LOG.exists():
        raise HTTPException(status_code=404, detail="No usage log yet. Process a PO first.")
    return FileResponse(
        path=str(USAGE_LOG), filename="usage_log.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Config endpoints ──────────────────────────────────────────────────────────

@app.get("/config", summary="View config (keys redacted)")
def get_config(auth: dict = Depends(verify_token)):
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
def reload_config(auth: dict = Depends(verify_token)):
    CFG.update(load_config())
    return {"status": "reloaded"}

@app.post("/decrypt-token", summary="Decrypt a token (admin only)")
def decrypt_token_endpoint(body: dict, user: str = Depends(verify_token)):
    from core.security import decrypt_token
    enc_key = CFG.get("security", {}).get("encryption_key", "")
    if not enc_key:
        raise HTTPException(status_code=500, detail="No encryption key configured.")
    try:
        decrypted = decrypt_token(body.get("token", ""), enc_key)
        return {"decrypted": decrypted}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")