import os
from dataclasses import dataclass


@dataclass
class Settings:
    LAWS_DIR: str = os.getenv("LAWS_DIR", "laws")
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "vector_store")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "kz_insurance_laws")

    # Multilingual MiniLM works well on Russian legal text and is small/fast.
    # Alternatives: "intfloat/multilingual-e5-base" (better quality, bigger)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # 'openai' | 'ollama'
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")


settings = Settings()
