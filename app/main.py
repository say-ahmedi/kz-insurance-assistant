"""
Web interface for the KZ Insurance Law Assistant.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import rag
from .config import settings
from .ingest import build_index

app = FastAPI(title="KZ Insurance Law Assistant")


class AskIn(BaseModel):
    query: str
    k: int = 5


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).parent.joinpath("ui.html").read_text(encoding="utf-8")


@app.post("/ask")
def ask(payload: AskIn):
    if not payload.query.strip():
        raise HTTPException(400, "Empty query")
    return rag.answer(payload.query, k=payload.k)


@app.post("/admin/upload")
async def upload_law(file: UploadFile = File(...)):
    """Drop a new .docx into the laws/ directory. Re-run /admin/reindex after."""
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(400, "Only .docx files are accepted")
    target = Path(settings.LAWS_DIR) / file.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())
    return {"saved": str(target), "next": "Call POST /admin/reindex to update the index"}


@app.post("/admin/reindex")
def reindex():
    build_index(Path(settings.LAWS_DIR), Path(settings.PERSIST_DIR))
    # Force the cached collection to refresh on next query
    rag._collection = None
    return {"status": "reindexed"}


@app.get("/health")
def health():
    try:
        coll = rag._get_collection()
        return {"status": "ok", "documents_indexed": coll.count()}
    except Exception as e:
        return {"status": "no-index", "error": str(e),
                "hint": "Run `python -m app.ingest` first"}
