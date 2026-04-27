import os
from dataclasses import dataclass


def _env(name, default):
    val = os.getenv(name)
    return val if val else default


@dataclass
class Settings:
    laws_dir: str = _env("LAWS_DIR", "laws")
    persist_dir: str = _env("PERSIST_DIR", "vector_store")
    collection_name: str = _env("COLLECTION_NAME", "kz_insurance_laws")

    embedding_model: str = _env(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    llm_provider: str = _env("LLM_PROVIDER", "openai")
    openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.1")

    top_k_default: int = int(_env("TOP_K", "5"))
    request_timeout: int = int(_env("LLM_TIMEOUT", "60"))


settings = Settings()
