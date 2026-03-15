"""
PageIndex API server: submit PDF indexing jobs, poll status, and fetch results.
"""
import logging
import traceback
import uuid
import threading
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from pageindex.page_index import page_index_main
from pageindex.utils import ConfigLoader
from types import SimpleNamespace as config

app = FastAPI(
    title="PageIndex API",
    description="Submit PDF indexing jobs and poll for results.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Log full traceback on server; return actual error message to client."""
    err_msg = str(exc)
    logger.exception("Unhandled error: %s", err_msg)
    return JSONResponse(
        status_code=500,
        content={
            "detail": err_msg,
            "error_type": type(exc).__name__,
        },
    )

# In-memory job store: job_id -> { status, result?, error? }
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# Config options matching run_pageindex / config.yaml
class JobConfig(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use (e.g. gpt-4o-2024-11-20)")
    toc_check_pages: Optional[int] = Field(default=None, description="Pages to check for TOC (PDF)")
    max_pages_per_node: Optional[int] = Field(default=None, description="Max pages per node (PDF)")
    max_tokens_per_node: Optional[int] = Field(default=None, description="Max tokens per node (PDF)")
    if_add_node_id: Optional[str] = Field(default=None, description="Add node id (yes/no)")
    if_add_node_summary: Optional[str] = Field(default=None, description="Add node summary (yes/no)")
    if_add_doc_description: Optional[str] = Field(default=None, description="Add doc description (yes/no)")
    if_add_node_text: Optional[str] = Field(default=None, description="Add text to node (yes/no)")


def _config_from_body(c: Optional[JobConfig]) -> config:
    """Build config SimpleNamespace from optional request body."""
    loader = ConfigLoader()
    if c is None:
        return loader.load(None)
    user_opt = {}
    if c.model is not None:
        user_opt["model"] = c.model
    if c.toc_check_pages is not None:
        user_opt["toc_check_page_num"] = c.toc_check_pages
    if c.max_pages_per_node is not None:
        user_opt["max_page_num_each_node"] = c.max_pages_per_node
    if c.max_tokens_per_node is not None:
        user_opt["max_token_num_each_node"] = c.max_tokens_per_node
    if c.if_add_node_id is not None:
        user_opt["if_add_node_id"] = c.if_add_node_id
    if c.if_add_node_summary is not None:
        user_opt["if_add_node_summary"] = c.if_add_node_summary
    if c.if_add_doc_description is not None:
        user_opt["if_add_doc_description"] = c.if_add_doc_description
    if c.if_add_node_text is not None:
        user_opt["if_add_node_text"] = c.if_add_node_text
    return loader.load(user_opt)


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "service": "PageIndex API",
        "docs": "/docs",
        "endpoints": {
            "POST /jobs": "Submit a PDF (multipart form: file + optional config). Returns job_id.",
            "GET /jobs/{job_id}/status": "Poll job status (pending|running|completed|failed).",
            "GET /jobs/{job_id}/result": "Get result JSON when completed; 202 if not ready.",
        },
    }


def _run_job(job_id: str, file_bytes: bytes, opt: config) -> None:
    """Run page_index_main in background and store result/error in _jobs."""
    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        buffer = BytesIO(file_bytes)
        result = page_index_main(buffer, opt)
        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
    except Exception as e:
        err_msg = str(e)
        logger.error("Job %s failed: %s", job_id, err_msg)
        logger.error("Traceback:\n%s", traceback.format_exc())
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = err_msg


def _parse_int(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(..., description="PDF file to index"),
    model: Optional[str] = Form(None),
    toc_check_pages: Optional[str] = Form(None),
    max_pages_per_node: Optional[str] = Form(None),
    max_tokens_per_node: Optional[str] = Form(None),
    if_add_node_id: Optional[str] = Form(None),
    if_add_node_summary: Optional[str] = Form(None),
    if_add_doc_description: Optional[str] = Form(None),
    if_add_node_text: Optional[str] = Form(None),
):
    """
    Submit a PDF for indexing. Returns a job_id to poll for status and fetch result.
    Config options can be sent as form fields (all optional).
    """
    if not file.filename:
        detail = "Missing or invalid file: use multipart form with field name 'file' and a PDF file."
        logger.warning("POST /jobs 400: %s", detail)
        raise HTTPException(status_code=400, detail=detail)
    if not file.filename.lower().endswith(".pdf"):
        detail = f"File must be a PDF (got: {file.filename}). Use form field name 'file'."
        logger.warning("POST /jobs 400: %s", detail)
        raise HTTPException(status_code=400, detail=detail)

    content = await file.read()
    if not content:
        detail = "Uploaded file is empty."
        logger.warning("POST /jobs 400: %s", detail)
        raise HTTPException(status_code=400, detail=detail)

    try:
        config_body = JobConfig(
        model=model or None,
        toc_check_pages=_parse_int(toc_check_pages),
        max_pages_per_node=_parse_int(max_pages_per_node),
        max_tokens_per_node=_parse_int(max_tokens_per_node),
        if_add_node_id=if_add_node_id or None,
        if_add_node_summary=if_add_node_summary or None,
        if_add_doc_description=if_add_doc_description or None,
        if_add_node_text=if_add_node_text or None,
        )
        opt = _config_from_body(config_body)
    except Exception as e:
        detail = f"Invalid config or request: {e!s}"
        logger.exception("POST /jobs config/validation error: %s", e)
        raise HTTPException(status_code=400, detail=detail)

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "pending", "result": None, "error": None}

    thread = threading.Thread(target=_run_job, args=(job_id, content, opt))
    thread.daemon = True
    thread.start()

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "pending", "message": "Job submitted. Poll /jobs/{job_id}/status and /jobs/{job_id}/result."},
    )


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Poll job status: pending | running | completed | failed."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        logger.warning("GET /jobs/%s/status: job not found", job_id)
        raise HTTPException(status_code=404, detail="Job not found.")
    out = {"job_id": job_id, "status": job["status"]}
    if job.get("error"):
        out["error"] = job["error"]
    return out


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the indexing result. Returns 202 if still pending/running, or the result JSON when completed.
    """
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        logger.warning("GET /jobs/%s/result: job not found", job_id)
        raise HTTPException(status_code=404, detail="Job not found.")
    status = job["status"]
    if status == "pending" or status == "running":
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": status, "message": "Job not ready. Poll again later."},
        )
    if status == "failed":
        err = job.get("error", "Unknown error")
        logger.warning("GET /jobs/%s/result: job failed - %s", job_id, err)
        raise HTTPException(
            status_code=422,
            detail={"job_id": job_id, "status": "failed", "error": err},
        )
    return job["result"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
