import logging
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from docx import Document

from .config import settings

log = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Header patterns. Some docx exports leave stray ** around numbers, so allow them.
ARTICLE_RE = re.compile(r"^\s*\**\s*Статья\s*\**\s*\d+(-\d+)?\.?", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*\**\s*Глава\s+\d+", re.IGNORECASE)

MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 2000
BATCH_SIZE = 64


def read_docx(path):
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines)


def split_by_article(text):
    lines = text.split("\n")
    chunks = []
    title = "Преамбула"
    body = []

    for line in lines:
        if ARTICLE_RE.match(line):
            if body:
                chunks.append({"article": title, "content": "\n".join(body).strip()})
            title = line.strip()
            body = [line]
        else:
            body.append(line)

    if body:
        chunks.append({"article": title, "content": "\n".join(body).strip()})

    return [c for c in chunks if len(c["content"]) > MIN_CHUNK_CHARS]


def chunk_long_articles(chunks, max_chars=MAX_CHUNK_CHARS):
    out = []
    for c in chunks:
        if len(c["content"]) <= max_chars:
            out.append(c)
            continue

        paragraphs = c["content"].split("\n")
        buf, size, part = [], 0, 1
        for p in paragraphs:
            if size + len(p) > max_chars and buf:
                out.append({
                    "article": f"{c['article']} (часть {part})",
                    "content": "\n".join(buf),
                })
                part += 1
                buf, size = [], 0
            buf.append(p)
            size += len(p)
        if buf:
            out.append({
                "article": f"{c['article']} (часть {part})",
                "content": "\n".join(buf),
            })
    return out


def iter_chunks(laws_dir):
    for path in sorted(Path(laws_dir).glob("*.docx")):
        log.info("Parsing %s", path.name)
        text = read_docx(path)
        if not text:
            log.warning("  empty document, skipped")
            continue

        articles = chunk_long_articles(split_by_article(text))
        log.info("  -> %d chunks", len(articles))

        first_line = text.split("\n", 1)[0][:200]
        for i, art in enumerate(articles):
            yield {
                "id": f"{path.stem}::{i:04d}",
                "text": art["content"],
                "metadata": {
                    "source_file": path.name,
                    "law_title": first_line or path.stem,
                    "article": art["article"][:200],
                },
            }


def build_index(laws_dir, persist_dir):
    laws_dir = Path(laws_dir)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if not laws_dir.exists() or not any(laws_dir.glob("*.docx")):
        log.warning("No .docx files in %s — nothing to index.", laws_dir)
        return 0

    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass

    coll = client.create_collection(
        name=settings.collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    total = 0
    for chunk in iter_chunks(laws_dir):
        ids.append(chunk["id"])
        docs.append(chunk["text"])
        metas.append(chunk["metadata"])

        if len(ids) >= BATCH_SIZE:
            coll.add(ids=ids, documents=docs, metadatas=metas)
            total += len(ids)
            ids, docs, metas = [], [], []

    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas)
        total += len(ids)

    log.info("Index built. Total documents: %d", coll.count())
    return total


if __name__ == "__main__":
    build_index(settings.laws_dir, settings.persist_dir)
