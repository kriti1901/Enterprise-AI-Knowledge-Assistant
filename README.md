# Enterprise AI Knowledge Assistant

Employees ask questions in plain language and get answers grounded in internal documents, with citations. Each user only sees documents from their own departments.

## Features

1. Upload PDF, DOCX and TXT documents (admin).
2. Extract and clean text; remove repeated headers/footers; re-join hyphenated words.
3. Chunk documents (page-aware) and generate embeddings with `google/embeddinggemma-2` via Sentence Transformers.
4. Store vectors in Chroma with department metadata.
5. Hybrid retrieval: semantic (Chroma) + keyword (BM25) fused with reciprocal rank fusion. Semantic-only mode is also available.
6. Answers from OpenAI or a local Ollama model, with numbered citations `[n]` that map to document, department and page.
7. Department-based access control, enforced in the vector query, the keyword candidate set, and the agent tools.
8. LangGraph agent that decides between `search`, `list_documents` and `answer`, with a step limit.
9. Chat history saved per user; the sidebar lists past conversations and reopens them with their sources. Conversation context is passed to the model on follow-up questions.
10. Evaluation: retrieval Hit@k and MRR, and answer accuracy against a golden set, via an admin page, API, or CLI.

Stack: Python 3.11, FastAPI, SQLAlchemy (PostgreSQL), Redis (answer cache), LangChain text splitters, LangGraph, Chroma, Sentence Transformers, Streamlit, Docker Compose, GitHub Actions, CloudWatch (custom metrics, optional).

## Architecture

```
Streamlit ──HTTP + JWT──> FastAPI
                            ├─ auth / users            → PostgreSQL
                            ├─ documents (ingestion)   → parse → clean → chunk → embed → Chroma + PostgreSQL
                            ├─ chat → LangGraph agent  → tools: search (hybrid, ACL), list_documents → LLM (OpenAI | Ollama)
                            ├─ conversations/messages  → PostgreSQL
                            ├─ answer cache            → Redis (optional)
                            └─ metrics / logs          → CloudWatch (optional) / stdout JSON
```

## Access model

- Every user has a list of departments. Users can search and read documents in those departments only.
- Only admins upload, delete and manage users, and run evaluation.
- Admins have no implicit access to every department. Give an admin departments too if they need to chat against them.
- The department filter is applied inside Chroma (`$in` filter), to the BM25 candidate set, and in the agent's `list_documents` tool. The LLM never receives chunks outside the caller's departments.
- Department membership is read from the database on each request, so revoking access takes effect immediately.

## Quick start (Docker, local, free)

```bash
cp .env.example .env          # set JWT_SECRET, ADMIN_PASSWORD; set OPENAI_API_KEY or use Ollama
docker compose up --build
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

Sign in with `ADMIN_USERNAME` / `ADMIN_PASSWORD`. Create users in **Documents → Create a user**, then upload documents with a department name.

### Free, fully offline option

Set `LLM_PROVIDER=ollama` and run a local model (`ollama pull llama3.2`). Embeddings are local. There are no API costs. Answer quality depends on the local model.

### Running without Docker

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=sqlite:///./data/app.db ADMIN_USERNAME=admin ADMIN_PASSWORD=change-me-123
uvicorn app.main:app --reload
# in another terminal
cd frontend && pip install -r requirements.txt && API_URL=http://localhost:8000 streamlit run app.py
```

## Evaluation

```bash
cd backend
python -m app.evaluation --seed --departments HR,Finance,Engineering
python -m app.evaluation --seed --no-answers   # retrieval metrics only
```

The golden set is `eval/golden_set.jsonl`; the sample corpus is `eval/sample_docs/<Department>/`. Metrics: `hit_at_k`, `mrr`, `answer_accuracy` (keyword match), `avg_latency_ms`. The admin endpoint is `POST /eval/run?departments=HR&departments=Finance`.

Tip: a meaningful answer-accuracy number needs `LLM_PROVIDER=openai` or `ollama`. The `fake` provider is only for offline tests.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
ruff check app tests
```

Tests run offline: `EMBEDDING_BACKEND=hash` and `LLM_PROVIDER=fake` are set in `tests/conftest.py`. They cover parsing and cleaning, the department access controls (including a Finance user asking about HR content), citations, conversation history privacy, deletion, and evaluation.

## Configuration

See `.env.example`. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `ollama`, or `fake` |
| `EMBEDDING_MODEL` | `google/embeddinggemma-2` | Sentence Transformers model. Changing it requires re-uploading or re-indexing documents. |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `hash` for offline tests only |
| `TOP_K` / `CANDIDATE_K` | `5` / `20` | Chunks passed to the LLM / candidates per retriever |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Characters per chunk |
| `REDIS_URL` | unset | Enables the answer cache (invalidated on any document change) |
| `CLOUDWATCH_ENABLED` | `false` | Publish custom metrics to CloudWatch |

## AWS deployment

This repository does not include Terraform. Suggested mapping:

- **ECS Fargate** for `api` and `frontend` images (built by CI).
- **RDS PostgreSQL** for `DATABASE_URL`, **ElastiCache Redis** for `REDIS_URL`.
- **EFS** (or S3 with a change to the Chroma persistence) for `CHROMA_PATH` and `UPLOAD_DIR`. Chroma is single-writer, so run one API task or use a shared volume carefully.
- **CloudWatch**: see [`deploy/cloudwatch/README.md`](deploy/cloudwatch/README.md) for metrics, example alarm and log filter.
- Secrets (`OPENAI_API_KEY`, `JWT_SECRET`, `ADMIN_PASSWORD`) in AWS Secrets Manager.

Costs: AWS services are billed (see their pricing pages). A local Docker run and Ollama have no service costs.

## Known limitations

- Chroma is a single-node store, so horizontal scaling of the API needs a different vector backend.
- The BM25 index is rebuilt from PostgreSQL on each query. This is fine for thousands of chunks, but it should move to a persistent index for larger corpora.
- Documents are assigned to one department each. Cross-department documents need to be uploaded once per department.
- Answer quality depends on the LLM. Use the evaluation endpoint to measure it on your own golden set before rollout.
- Uploads are processed synchronously. For very large files, move ingestion to a background worker.
