# KZ Insurance Law Assistant

[Русская версия](README.ru.md)

An internal RAG assistant for an insurance company. It answers staff questions
about the insurance legislation of the Republic of Kazakhstan, grounded in the
text of the laws themselves and citing the article it pulled the answer from.

## What it does

- Parses Word documents from `laws/` and splits them by `Статья N`
- Embeds the chunks (multilingual MiniLM) into a local Chroma store
- On each query: retrieves the top-k articles and passes them to an LLM
  (OpenAI by default, Ollama as an offline alternative) with a strict
  "answer only from sources, cite the article" prompt
- Provides a small web chat and a REST API
- Lets you upload more `.docx` laws or pull a page from `adilet.zan.kz`,
  then re-index

## Project layout

```
insurance_assistant/
├── app/
│   ├── ingest.py     # docx → chunks → embeddings → Chroma
│   ├── rag.py        # retrieve + prompt + LLM call
│   ├── adilet.py     # fetch a law from adilet.zan.kz and save as .docx
│   ├── main.py       # FastAPI: /ask, /admin/upload, /admin/fetch_adilet, /admin/reindex
│   ├── ui.html       # web chat
│   └── config.py
├── laws/             # drop .docx files here
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Running

### With Docker (OpenAI)

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
```

Then open http://localhost:8000.

### With Docker (Ollama, fully offline)

1. Uncomment the `ollama` service in `docker-compose.yml`.
2. Set `LLM_PROVIDER=ollama` in the environment.
3. Run:
   ```bash
   docker compose up --build
   docker compose exec ollama ollama pull llama3.1
   ```

### Without Docker

```bash
pip install -r requirements.txt
python -m app.ingest                    # build the index (one-time)
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload
```

## Using it

### Web UI

`http://localhost:8000` — text box, example chips for common questions, and an
expandable list of source articles under every answer (law name, article,
relevance score, snippet).

### REST API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Какова страховая сумма по ОГПОВТС?"}'
```

### Adding a new law

Three options:

1. **Upload a .docx**:
   ```bash
   curl -F "file=@new_law.docx" http://localhost:8000/admin/upload
   curl -X POST http://localhost:8000/admin/reindex
   ```
2. **Pull from adilet.zan.kz**:
   ```bash
   curl -X POST http://localhost:8000/admin/fetch_adilet \
     -H "Content-Type: application/json" \
     -d '{"url": "https://adilet.zan.kz/rus/docs/...", "save_as": "new_law"}'
   curl -X POST http://localhost:8000/admin/reindex
   ```
3. Drop the file into `laws/` and rebuild the image (or re-run `python -m app.ingest`).

## Design notes

| Step | Choice | Why |
|------|--------|-----|
| Parsing | `python-docx` + regex on `Статья N` | Article is the natural unit of a KZ legal act |
| Chunking | Paragraph split, ≤2000 chars | Keeps embeddings semantically focused |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Small, fast, handles Russian legal text |
| Vector DB | Chroma (persistent, on disk) | No external service, ships in the container |
| LLM | Pluggable: OpenAI / Ollama | Cloud or fully offline |
| Prompt | Strict system prompt + cited articles | Forbids fabrication; always cites source |
| Temperature | 0.1 | Maximum determinism for legal text |
