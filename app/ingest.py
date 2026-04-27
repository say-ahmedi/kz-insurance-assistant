"""
Ingestion pipeline:

  .docx files in laws/  -->  text  -->  chunks (by article/section)
                                          -->  embeddings  -->  vector store

Run once after dropping new files into the laws/ directory:
    python -m app.ingest
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions
from docx import Document

from .config import settings

log = logging.getLogger("ingest")
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)

# Russian patterns: "Статья 5.", "Статья 12-1." etc. — the natural unit of KZ legal text.
# Allow optional Markdown bold markers (`**Статья 5.**`) and arbitrary whitespace
# inside the header (some docs render as `Статья ****5.`).
ARTICLE_RE = re.compile(r"^\s*\**\s*Статья\s*\**\s*\d+(-\d+)?\.?", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*\**\s*Глава\s+\d+", re.IGNORECASE)


def read_docx(path: Path) -> str:
    """Return the document body as plain text, one paragraph per line."""
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


def split_by_article(text: str) -> list[dict]:
    """
    Split a law's full text into article-level chunks.
    Returns list of {article: str, content: str}.
    Falls back to a single chunk when no 'Статья' headers are detected
    (some documents only use 'Глава' / chapter level).
    """
    lines = text.split("\n")
    chunks: list[dict] = []
    current_title = "Преамбула"
    current_body: list[str] = []

    for line in lines:
        if ARTICLE_RE.match(line):
            if current_body:
                chunks.append({
                    "article": current_title,
                    "content": "\n".join(current_body).strip(),
                })
            current_title = line.strip()
            current_body = [line]
        else:
            current_body.append(line)

    if current_body:
        chunks.append({
            "article": current_title,
            "content": "\n".join(current_body).strip(),
        })

    # Drop empties + tiny noise chunks
    chunks = [c for c in chunks if len(c["content"]) > 50]
    return chunks


def chunk_long_articles(chunks: list[dict], max_chars: int = 2000) -> list[dict]:
    """
    Some articles are very long (>5000 chars). Split those further on paragraph
    boundaries so embeddings stay semantically focused.
    """
    out = []
    for c in chunks:
        if len(c["content"]) <= max_chars:
            out.append(c)
            continue
        paragraphs = c["content"].split("\n")
        buf, size = [], 0
        part_n = 1
        for p in paragraphs:
            if size + len(p) > max_chars and buf:
                out.append({
                    "article": f"{c['article']} (часть {part_n})",
                    "content": "\n".join(buf),
                })
                part_n += 1
                buf, size = [], 0
            buf.append(p)
            size += len(p)
        if buf:
            out.append({
                "article": f"{c['article']} (часть {part_n})",
                "content": "\n".join(buf),
            })
    return out


def iter_chunks(laws_dir: Path) -> Iterable[dict]:
    for path in sorted(laws_dir.glob("*.docx")):
        log.info("Parsing %s", path.name)
        text = read_docx(path)
        articles = split_by_article(text)
        articles = chunk_long_articles(articles)
        log.info("  -> %d chunks", len(articles))
        for i, art in enumerate(articles):
            yield {
                "id": f"{path.stem}::{i:04d}",
                "text": art["content"],
                "metadata": {
                    "source_file": path.name,
                    "law_title": text.split("\n")[0][:200] if text else path.stem,
                    "article": art["article"][:200],
                },
            }


def build_index(laws_dir: Path, persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Multilingual model — works well on Russian legal text
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )

    # Recreate from scratch so re-ingest is deterministic
    try:
        client.delete_collection(settings.COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    for chunk in iter_chunks(laws_dir):
        ids.append(chunk["id"])
        docs.append(chunk["text"])
        metas.append(chunk["metadata"])

        # Flush in batches — embedding all at once can OOM on big corpora
        if len(ids) >= 64:
            coll.add(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []
    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas)

    log.info("Index built. Total documents: %d", coll.count())


if __name__ == "__main__":
    build_index(Path(settings.LAWS_DIR), Path(settings.PERSIST_DIR))
