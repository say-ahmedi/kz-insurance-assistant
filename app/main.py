from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import adilet, rag
from .config import settings
from .ingest import build_index

app = FastAPI(title="ИИ-ассистент: страховое право РК")

UI_PATH = Path(__file__).parent / "ui.html"


class AskIn(BaseModel):
    query: str
    k: int = 5


class AdiletIn(BaseModel):
    url: str
    save_as: str | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return UI_PATH.read_text(encoding="utf-8")


@app.post("/ask")
def ask(payload: AskIn):
    q = payload.query.strip()
    if not q:
        raise HTTPException(400, "Пустой запрос")
    return rag.answer(q, k=payload.k)


@app.get("/laws")
def list_laws():
    laws_dir = Path(settings.laws_dir)
    if not laws_dir.exists():
        return {"laws": []}
    return {"laws": sorted(p.name for p in laws_dir.glob("*.docx"))}


@app.post("/admin/upload")
async def upload_law(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(400, "Принимаются только .docx файлы")

    target = Path(settings.laws_dir) / file.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())

    return {
        "saved": str(target),
        "next": "Вызовите POST /admin/reindex для обновления индекса",
    }


@app.post("/admin/fetch_adilet")
def fetch_from_adilet(payload: AdiletIn):
    try:
        path = adilet.fetch_law(payload.url, save_as=payload.save_as)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Не удалось загрузить документ: {e}")
    return {
        "saved": str(path),
        "next": "Вызовите POST /admin/reindex для обновления индекса",
    }


@app.post("/admin/reindex")
def reindex():
    total = build_index(settings.laws_dir, settings.persist_dir)
    rag.reset_cache()
    return {"status": "reindexed", "documents": total}


@app.get("/health")
def health():
    try:
        coll = rag._get_collection()
        return {"status": "ok", "documents_indexed": coll.count()}
    except Exception as e:
        return {
            "status": "no-index",
            "error": str(e),
            "hint": "Запустите `python -m app.ingest` или POST /admin/reindex",
        }
