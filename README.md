# Fieldnotes — Local PDF RAG Demo

Fieldnotes is a full-stack Retrieval-Augmented Generation (RAG) demo. Upload text-based PDFs, ask questions about them, and receive answers grounded in retrieved passages with document and page citations.

By default, the app uses **Ollama** to run embedding and chat models locally, so it does not need a Mistral API key or incur Mistral API rate limits. Mistral remains available as an optional cloud provider. The RAG pipeline is implemented directly in Python; it does not use LangChain.

## Features

- Upload PDFs with the browser file picker or drag and drop.
- Extract selectable text from PDF pages and split it into overlapping chunks.
- Generate embeddings with a local Ollama model or Mistral's embedding API.
- Persist vectors and source metadata in a local Chroma database.
- Retrieve the five nearest chunks for a question and use them as answer context.
- Display source document and page references with answers.
- Keep a bounded conversation buffer per browser session.
- List and delete indexed documents.

## RAG pipeline

1. **Upload and extract:** FastAPI accepts a PDF up to the configured size limit. PyMuPDF extracts text page by page.
2. **Chunk:** Page text is whitespace-normalized and split into chunks of up to 1,200 characters with a 180-character overlap. Page numbers are retained for citations.
3. **Embed:** The configured provider embeds chunks in batches of 32. The same embedding model is used for question vectors.
4. **Index:** Chroma stores vectors, text, and metadata (`document_id`, source filename, page, and chunk number) in its persistent `pdf_chunks` collection using cosine distance.
5. **Retrieve:** The query vector retrieves up to five nearest chunks across indexed PDFs.
6. **Generate:** The configured chat model receives the retrieved passages and recent conversation. Its system prompt asks it to answer from the passages, say when evidence is insufficient, and cite pages.
7. **Remember:** A browser session ID is kept in local storage. The server buffers up to 20 messages (10 user/assistant exchanges) in memory per session.

## Requirements

- Python 3.10 or later.
- [Ollama for Windows](https://ollama.com/download) for the default local setup.
- Enough memory and disk space for the selected Ollama models. Model speed depends on your computer.

No separate frontend build or Node.js installation is needed; the UI is plain HTML, CSS, and JavaScript.

### Python dependencies

The packages are listed in [`requirements.txt`](requirements.txt).

| Package | Purpose |
| --- | --- |
| `fastapi` | Web application and JSON API |
| `uvicorn[standard]` | ASGI server |
| `python-multipart` | PDF upload form handling |
| `python-dotenv` | Load settings from `.env` |
| `httpx` | Async HTTP requests to Ollama or Mistral |
| `pymupdf` | Extract PDF text |
| `chromadb` | Persistent local vector database and similarity search |

## Run locally on Windows with Ollama

### 1. Install and prepare Ollama

Install Ollama, then open PowerShell and download the default models:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Keep the Ollama app/server running while using Fieldnotes. Verify that the local API responds:

```powershell
ollama list
```

The default server address is `http://localhost:11434`.

### 2. Set up Python

Open PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `py` launcher is optional. If PowerShell does not recognize `python`, install Python and enable **Add Python to PATH**, or invoke the full path to your Python executable.

If PowerShell prevents environment activation, run commands using the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure and run Fieldnotes

Create `.env` from the example if it does not already exist, then open it:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

The default provider is Ollama; no API key is required. Start the app from the project directory:

```powershell
uvicorn app.main:app --reload
```

If the virtual environment is not activated:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a PDF, wait for indexing, and ask a question. Interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Configuration

The app loads these values from `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | Inference provider: `ollama` or `mistral` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server URL |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Ollama model used to generate answers |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama model used for document/query embeddings |
| `MISTRAL_API_KEY` | *(not needed for Ollama)* | API key required only when using Mistral |
| `MISTRAL_CHAT_MODEL` | `mistral-small-latest` | Mistral chat model |
| `MISTRAL_EMBED_MODEL` | `mistral-embed` | Mistral embedding model |
| `CHROMA_PATH` | `./data/chroma` | Local persistent vector database directory |
| `MAX_UPLOAD_MB` | `25` | Maximum PDF upload size in megabytes |

### Optional Mistral provider

To use Mistral instead of local Ollama, set the following in `.env`, add your own API key, then restart Fieldnotes:

```dotenv
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key
```

Mistral usage is subject to your account's model access, quota, and rate limits. Ollama mode uses local inference and does not send prompts or PDF text to Mistral.

Use the same embedding model for indexing and querying. If changing providers or embedding models after indexing, remove and re-upload the indexed documents; vector dimensions and embedding spaces may differ.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Report app status, configured provider, and indexed document count |
| `GET` | `/api/documents` | List indexed documents and chunk counts |
| `POST` | `/api/documents` | Upload and index a PDF as multipart field `file` |
| `DELETE` | `/api/documents/{document_id}` | Remove a document's chunks from Chroma |
| `POST` | `/api/chat` | Retrieve context and generate a response |
| `DELETE` | `/api/chat/{session_id}` | Clear a session's in-memory conversation |

Example chat request:

```json
{
  "session_id": "your-browser-session-id",
  "message": "What are the main findings?"
}
```

The response contains an `answer` and a list of cited `sources`, each with a document name and page number.

## Data, privacy, and limitations

- Chroma persists extracted chunk text and embeddings under `data/chroma`. The original uploaded PDF is read in memory and is not saved by the app.
- With Ollama, embedding and chat requests go to the configured local Ollama server. With Mistral, text and prompts are sent to Mistral's cloud API.
- Conversation history is held in server memory, cleared when the server restarts, and not shared between server processes.
- Scanned/image-only PDFs are not OCR-processed. Apply OCR before uploading.
- Retrieval is a simple top-five cosine similarity search; there is no document filter, relevance threshold, or reranker.
- Generated answers may be incorrect. Review their cited passages.

## Troubleshooting

- **Cannot connect to Ollama:** Start the Ollama app and check `ollama list`. Confirm `OLLAMA_BASE_URL` points to the running server.
- **Model not found:** Pull the configured model, for example `ollama pull llama3.2` or `ollama pull nomic-embed-text`.
- **Mistral 429:** This applies only when `LLM_PROVIDER=mistral`; wait and check your Mistral account quota/rate limits, or switch back to `LLM_PROVIDER=ollama`.
- **Embedding dimension error after changing providers:** Remove the old indexed documents and re-upload them using the selected embedding model. Do not mix vectors from different embedding models.
- **No readable PDF text:** The PDF may be scanned or image-only; run OCR and upload a searchable PDF.
- **Upload rejected:** Confirm it is a valid PDF within the `MAX_UPLOAD_MB` limit.
- **Port already in use:** Start on a different port, such as `uvicorn app.main:app --reload --port 8001`.

