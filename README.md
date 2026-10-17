# AI Codebase Onboarding Assistant

An AI-powered codebase onboarding assistant that indexes Git repositories, retrieves relevant code chunks, and generates answers to questions about the codebase with source references.

## Features

- **Git repository ingestion:** Clones a Git repository into a temporary directory for indexing.
- **Python code parsing:** Extracts structured information from Python source files.
- **Code chunking:** Splits source code into structured chunks for retrieval.
- **Embeddings:** Generates vector embeddings using Gemini's `gemini-embedding-001` model.
- **Persistent storage:** Stores code chunks and embeddings in PostgreSQL with pgvector.
- **Repository indexing:** Tracks new, changed, unchanged, and removed chunks.
- **Semantic retrieval:** Retrieves relevant code chunks for a question.
- **Answer synthesis:** Uses Gemini `gemini-flash-lite-latest` by default to generate answers grounded in retrieved code.
- **Source references:** Returns file paths, chunk identifiers, relevance scores, and ranks with answers.
- **HTTP API:** Exposes repository indexing and question-answering endpoints through FastAPI.

## Architecture

The application is organized into separate components:

| Component | Responsibility |
| --- | --- |
| `app/ingestion/` | Cloning repositories and discovering files |
| `app/parsing/` | Parsing Python source code |
| `app/chunking/` | Creating structured code chunks |
| `app/embedding/` | Embedding generation and chunk identity |
| `app/persistence/` | Database configuration and chunk storage |
| `app/retrieval/` | Vector search and retrieval |
| `app/pipeline/` | Repository indexing and query orchestration |
| `app/synthesis/` | Answer generation and LLM integration |
| `app/api/` | FastAPI routes, request models, and responses |
| `app/composition.py` | Construction of application services |

### Indexing workflow

1. Clone a Git repository.
2. Discover supported source files.
3. Parse and split source code into chunks.
4. Generate embeddings for chunks.
5. Persist chunk metadata and embeddings in PostgreSQL.
6. Track indexing changes between repository versions.

### Question-answering workflow

1. Receive a question and repository identifier.
2. Retrieve relevant code chunks using vector search.
3. Provide the retrieved context to the answer-generation service.
4. Return an answer with references to the retrieved source chunks.

## Technology Stack

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- pgvector
- Tree-sitter
- Google Gemini API (default embedding and answer-generation provider)
- OpenAI API (optional alternative providers)
- pytest

## Prerequisites

Before running the application, install:

- Python 3.13 (the version the project is developed and tested with)
- Git
- PostgreSQL
- The pgvector PostgreSQL extension
- A Google Gemini API key

The project dependencies are pinned in `requirements.txt`.

## Setup

### 1. Clone the project

```bash
git clone https://github.com/Asyayla/ai-codebase-onboarding.git
cd ai-codebase-onboarding
```

Replace `<YOUR_PROJECT_URL>` with the URL of this project.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a dedicated PostgreSQL database for the application:

```sql
CREATE DATABASE ai_codebase;
```

Connect to that database and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Set the database connection URL in the environment:

```bash
export DATABASE_URL="postgresql+psycopg://USER:PASSWORD@localhost:5432/ai_codebase"
```

Replace `USER` and `PASSWORD` with your local PostgreSQL credentials.

Initialize the application's SQLAlchemy tables:

```bash
python -c "from app.persistence.database import Base, get_engine; import app.persistence.orm_models; Base.metadata.create_all(get_engine())"
```

Run this command after setting `DATABASE_URL` and enabling pgvector.

### 5. Configure the Gemini API key

```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

The default embedding provider uses Google's `gemini-embedding-001` model with 1536-dimensional vectors. Repository code chunks use the `RETRIEVAL_DOCUMENT` task type, while search queries use `CODE_RETRIEVAL_QUERY`. Answer generation uses `gemini-flash-lite-latest` by default. OpenAI providers remain available as optional alternatives.

Keep API keys and database credentials private. Do not commit `.env` files or secrets to version control.

**Embedding migration note:** Gemini and OpenAI embeddings use different vector spaces, even when their dimensions match. If you switch embedding providers for a repository that has already been indexed, re-index its code chunks before relying on semantic search.

## Running the API

Start the FastAPI application from the project root:

```bash
uvicorn app.api.app:create_app --factory --reload --reload-dir app
```

`--reload-dir app` limits auto-reload to the application code. Without it, changes inside `.venv` can restart the server in the middle of a long request such as repository indexing.

The API will normally be available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`

## API Endpoints

### Index a repository

`POST /repositories/index`

Request:

```json
{
  "source": "https://github.com/OWNER/REPOSITORY"
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/repositories/index" \
  -H "Content-Type: application/json" \
  -d '{"source":"https://github.com/OWNER/REPOSITORY"}'
```

The response includes the repository identifier, number of files considered, counts of new, changed, unchanged, and removed chunks, and embedding failures.

Use the returned `repository_id` when asking questions.

### Ask a question about a repository

`POST /repositories/{repository_id}/ask`

The `repository_id` is the repository URL returned by the index endpoint (for example `https://github.com/OWNER/REPOSITORY`). In `curl`, URL-encode it. In Swagger UI, enter it as-is, without quotes.

Request:

```json
{
  "question": "How does repository indexing work?",
  "top_k": 5
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/repositories/https%3A%2F%2Fgithub.com%2FOWNER%2FREPOSITORY/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"How does repository indexing work?","top_k":5}'
```

`top_k` is optional, defaults to `5`, and must be between `1` and `50`.

The response contains the generated answer and a list of source references. Each source includes its file path, chunk ID, score, and rank.

### Error responses

If the language model service is temporarily unavailable (for example, Gemini returns 503 or rate-limits the request), the API responds with `503` and a clear message instead of an unhandled server error. If the model service rejects a request, the API responds with `502`.

## Example: asking the assistant about its own codebase

The project was indexed with itself as the input repository. The answers below are abridged from real responses; each response also includes ranked source references.

**Question:** `How does the repository indexer decide whether a chunk is new, changed or unchanged?`

> The indexer computes a `chunk_id` from the repository, file path, symbol type, symbol name and line range, and a `content_hash` from the chunk content. A chunk whose `chunk_id` is not stored is **new**; one whose stored hash equals the current hash is **unchanged**; one whose stored hash differs is **changed**. Only new and changed chunks are embedded.

Top source: `app/pipeline/repository_indexer.py`

**Question:** `What does AnswerService return when retrieval finds no chunks, and does it call the LLM?`

> It returns `AnswerResult(answer=NO_CONTEXT_ANSWER, sources=[])` and does not call the LLM; it returns early before `generate_answer` is invoked.

Top source: `app/synthesis/answer_service.py`

When the retrieved code does not contain the answer, the assistant says the context is insufficient instead of guessing.

## Testing

Run the complete test suite:

```bash
pytest -q
```

The PostgreSQL integration tests use a dedicated test database. They require `AI_CODEBASE_TEST_DATABASE_URL` and a PostgreSQL instance with pgvector installed.

Create a separate test database before running these integration tests:

```sql
CREATE DATABASE ai_codebase_test;
```

Enable pgvector in the test database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Set the test database URL:

```bash
export AI_CODEBASE_TEST_DATABASE_URL="postgresql+psycopg://USER:PASSWORD@localhost:5432/ai_codebase_test"
```

Run the PostgreSQL integration tests:

```bash
pytest -q tests/test_postgres_chunk_store.py tests/retrieval/test_postgres_vector_store.py
```

Then run the full suite:

```bash
pytest -q
```

**Important:** Use a dedicated test database. The integration-test fixtures create and drop the `chunks` table and clear its data.

## Current Status

The project includes repository ingestion, Python parsing and chunking, embedding generation, PostgreSQL persistence, vector retrieval, answer synthesis, and a FastAPI interface.

The automated test suite currently passes **290 tests**, including PostgreSQL integration tests, in the development environment.

## Limitations and Future Work

- Only Python source files are parsed and indexed.
- Retrieval quality depends on how a question is phrased; hybrid (keyword + vector) search and reranking are natural next steps.
- The API has no authentication or rate limiting and is intended for local development.

## Security Considerations

- Only index repositories you are authorized to access.
- Treat repository contents as potentially untrusted input.
- Keep API credentials and database connection strings out of source control.
- Do not expose the development API publicly without appropriate authentication, authorization, and deployment hardening.
