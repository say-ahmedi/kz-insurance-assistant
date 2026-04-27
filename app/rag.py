"""
RAG core: query the vector store, build a grounded prompt, call the LLM.

The LLM provider is pluggable. Two backends are wired here:
  * 'openai'  -> uses OPENAI_API_KEY (default: gpt-4o-mini)
  * 'ollama'  -> calls a local Ollama server (default: llama3.1)

Pick via settings.LLM_PROVIDER. To swap to GigaChat / YandexGPT / Claude /
anything else, add another branch in `_call_llm`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import chromadb
import httpx
from chromadb.utils import embedding_functions

from .config import settings

log = logging.getLogger("rag")

SYSTEM_PROMPT = """Ты — ИИ-ассистент юридического отдела казахстанской страховой компании.
Твоя задача — консультировать сотрудников по вопросам страхового
законодательства Республики Казахстан, опираясь ИСКЛЮЧИТЕЛЬНО на
выдержки из официальных нормативных документов, приведённые ниже.

Правила:
1. Отвечай только на основе предоставленных фрагментов. Если ответа в
   них нет — честно скажи, что в загруженной базе нет соответствующей
   нормы, и предложи проверить на adilet.zan.kz.
2. Всегда указывай источник: название закона и номер статьи.
3. Отвечай по-русски, чётко, формально, профессиональным юридическим
   языком, но без воды.
4. Если вопрос неоднозначный — попроси уточнение, прежде чем отвечать.
5. Никогда не придумывай статьи, номера или формулировки.
"""


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    law_title: str
    article: str
    distance: float


# --------------------------------------------------------------------------
# Vector store handle (lazy)
# --------------------------------------------------------------------------
_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = chromadb.PersistentClient(path=settings.PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )
    _collection = client.get_collection(
        name=settings.COLLECTION_NAME, embedding_function=ef
    )
    return _collection


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    coll = _get_collection()
    res = coll.query(query_texts=[query], n_results=k)
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        out.append(RetrievedChunk(
            text=res["documents"][0][i],
            source_file=meta.get("source_file", "?"),
            law_title=meta.get("law_title", "?"),
            article=meta.get("article", "?"),
            distance=res["distances"][0][i],
        ))
    return out


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------
def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    sources_block = []
    for i, c in enumerate(chunks, 1):
        sources_block.append(
            f"--- Фрагмент {i} ---\n"
            f"Закон: {c.law_title}\n"
            f"Файл: {c.source_file}\n"
            f"{c.article}\n\n"
            f"{c.text}\n"
        )
    sources_text = "\n".join(sources_block)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== Релевантные выдержки из законов РК ===\n"
        f"{sources_text}\n\n"
        f"=== Вопрос сотрудника ===\n{query}\n\n"
        f"=== Твой ответ (со ссылками на статьи) ==="
    )


# --------------------------------------------------------------------------
# LLM dispatch
# --------------------------------------------------------------------------
def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": settings.OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str) -> str:
    r = httpx.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def _call_llm(prompt: str) -> str:
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        return _call_openai(prompt)
    if provider == "ollama":
        return _call_ollama(prompt)
    raise ValueError(f"Unknown LLM provider: {provider}")


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------
def answer(query: str, k: int = 5) -> dict:
    chunks = retrieve(query, k=k)
    if not chunks:
        return {
            "answer": "В загруженной базе законов не найдено релевантных норм. "
                      "Рекомендую проверить на https://adilet.zan.kz/rus.",
            "sources": [],
        }
    prompt = build_prompt(query, chunks)
    try:
        text = _call_llm(prompt)
    except Exception as e:
        log.exception("LLM call failed")
        text = (f"⚠️ Не удалось получить ответ от LLM ({e}). "
                f"Ниже — найденные релевантные фрагменты, на которые "
                f"следует опираться:")
    return {
        "answer": text,
        "sources": [
            {
                "law": c.law_title,
                "article": c.article,
                "file": c.source_file,
                "snippet": c.text[:500],
                "score": round(1 - c.distance, 3),
            }
            for c in chunks
        ],
    }
