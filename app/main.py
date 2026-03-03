from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    ClarifyRequest,
    ClarifyResponse,
    DirectAskRequest,
    DirectAskResponse,
    ExecuteRequest,
    ExecuteResponse,
)
from app.services.clarifier import ClarifierService
from app.services.executor import ExecutorService

load_dotenv()

app = FastAPI(title="Prompt Clarifier / Prompt Coach MVP", version="0.1.0")

clarifier_service = ClarifierService()
executor_service = ExecutorService(clarifier_service)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/v1/clarify", response_model=ClarifyResponse)
def clarify(request: ClarifyRequest) -> ClarifyResponse:
    try:
        return clarifier_service.generate(
            message=request.message,
            locale=request.locale,
            context=request.context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    try:
        return executor_service.execute(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/direct", response_model=DirectAskResponse)
def direct(request: DirectAskRequest) -> DirectAskResponse:
    try:
        return executor_service.ask_direct(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
