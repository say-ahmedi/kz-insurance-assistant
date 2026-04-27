# ИИ-ассистент: страховое право РК

[English version](README.md)

Внутренний RAG-ассистент для страховой компании. Отвечает на вопросы
сотрудников по страховому законодательству Республики Казахстан, опираясь
на тексты загруженных нормативных актов и указывая, из какой именно статьи
взят ответ.

## Что умеет

- Разбирает Word-документы из `laws/` и режет их по заголовку `Статья N`
- Считает эмбеддинги (multilingual MiniLM) и кладёт их в локальный Chroma
- На каждый запрос достаёт top-k фрагментов и передаёт LLM
  (по умолчанию — OpenAI, оффлайн-вариант — Ollama) с жёстким
  промптом «отвечай только по источникам, ссылайся на статью»
- Предоставляет веб-чат и REST API
- Позволяет загружать новые `.docx` или подтягивать страницу с
  `adilet.zan.kz`, после чего переиндексировать базу

## Структура

```
insurance_assistant/
├── app/
│   ├── ingest.py     # docx → чанки → эмбеддинги → Chroma
│   ├── rag.py        # retrieve + промпт + вызов LLM
│   ├── adilet.py     # загрузка закона с adilet.zan.kz и сохранение в .docx
│   ├── main.py       # FastAPI: /ask, /admin/upload, /admin/fetch_adilet, /admin/reindex
│   ├── ui.html       # веб-чат
│   └── config.py
├── laws/             # сюда кладутся .docx с законами
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Запуск

### Через Docker (OpenAI)

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
```

Откройте http://localhost:8000.

### Через Docker (Ollama, полностью оффлайн)

1. Раскомментируйте сервис `ollama` в `docker-compose.yml`.
2. Установите `LLM_PROVIDER=ollama` в окружении.
3. Запустите:
   ```bash
   docker compose up --build
   docker compose exec ollama ollama pull llama3.1
   ```

### Без Docker

```bash
pip install -r requirements.txt
python -m app.ingest                    # одноразовая сборка индекса
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload
```

## Использование

### Веб-интерфейс

`http://localhost:8000` — поле ввода, чипсы с примерами популярных
вопросов, под каждым ответом — раскрывающийся список источников
(название закона, номер статьи, оценка релевантности, фрагмент текста).

### REST API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Какова страховая сумма по ОГПОВТС?"}'
```

### Добавление нового закона

Три способа:

1. **Загрузить .docx**:
   ```bash
   curl -F "file=@new_law.docx" http://localhost:8000/admin/upload
   curl -X POST http://localhost:8000/admin/reindex
   ```
2. **Подтянуть с adilet.zan.kz**:
   ```bash
   curl -X POST http://localhost:8000/admin/fetch_adilet \
     -H "Content-Type: application/json" \
     -d '{"url": "https://adilet.zan.kz/rus/docs/...", "save_as": "new_law"}'
   curl -X POST http://localhost:8000/admin/reindex
   ```
3. Положить файл в `laws/` и пересобрать образ (или заново
   выполнить `python -m app.ingest`).

## Архитектурные решения

| Этап | Решение | Почему |
|------|---------|--------|
| Парсинг | `python-docx` + regex по `Статья N` | Статья — естественная единица казахстанского НПА |
| Разбивка | По абзацам, ≤2000 символов | Эмбеддинги остаются семантически фокусными |
| Эмбеддинги | `paraphrase-multilingual-MiniLM-L12-v2` | Малый, быстрый, понимает русский юридический язык |
| Векторная БД | Chroma (persistent, на диске) | Не нужен внешний сервис — всё в одном контейнере |
| LLM | Pluggable: OpenAI / Ollama | Облако или полный оффлайн |
| Промпт | Жёсткий system prompt + цитаты | Запрещает выдумывать, всегда указывает источник |
| Температура | 0.1 | Максимум детерминизма для юридического текста |

## Что можно улучшить

- Гибридный поиск (BM25 + векторный) — для запросов вида «Статья 7 пункт 2»
- Reranker (`bge-reranker-v2-m3`) поверх top-20 чанков
- История диалога (сейчас каждый запрос независим)
- Сбор обратной связи (👍 / 👎) для последующего анализа
- Авторизация сотрудников (JWT)
