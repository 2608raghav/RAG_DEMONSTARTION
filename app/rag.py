import os
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

import chromadb
import httpx
import pymupdf as fitz


MISTRAL_API_URL = "https://api.mistral.ai/v1"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 32
RETRIEVAL_COUNT = 5


class ConfigurationError(RuntimeError):
    pass


class MistralAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str, retry_after: str | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class EmbeddingModelMismatchError(RuntimeError):
    pass


class RAGProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def configured(self) -> bool: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def answer(self, messages: list[dict[str, str]]) -> str: ...


class MistralClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.chat_model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
        self.embed_model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
        self.timeout = httpx.Timeout(90.0, connect=15.0)

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError(
                "MISTRAL_API_KEY is missing. Add it to your .env file and restart the app."
            )
        return self.api_key

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.require_api_key()}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{MISTRAL_API_URL}/{path}", headers=headers, json=payload
            )
        if response.is_error:
            try:
                details = response.json().get("message") or response.text
            except ValueError:
                details = response.text
            raise MistralAPIError(
                response.status_code,
                f"Mistral API returned {response.status_code}: {details}",
                response.headers.get("Retry-After"),
            )
        return response.json()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[offset : offset + EMBED_BATCH_SIZE]
            result = await self._post(
                "embeddings", {"model": self.embed_model, "input": batch}
            )
            vectors.extend(
                item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])
            )
        return vectors

    async def answer(self, messages: list[dict[str, str]]) -> str:
        result = await self._post(
            "chat/completions",
            {
                "model": self.chat_model,
                "messages": messages,
                "temperature": 0.2,
            },
        )
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Mistral returned an empty answer.")
        return content.strip()


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
        self.embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.timeout = httpx.Timeout(180.0, connect=5.0)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/{path}", json=payload)
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Start Ollama and try again."
            ) from exc
        if response.is_error:
            try:
                details = response.json().get("error") or response.text
            except ValueError:
                details = response.text
            raise RuntimeError(f"Ollama API returned {response.status_code}: {details}")
        return response.json()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[offset : offset + EMBED_BATCH_SIZE]
            result = await self._post(
                "api/embed",
                {"model": self.embed_model, "input": batch},
            )
            batch_vectors = result.get("embeddings")
            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
                raise RuntimeError("Ollama returned an unexpected embedding response.")
            vectors.extend(batch_vectors)
        return vectors

    async def answer(self, messages: list[dict[str, str]]) -> str:
        result = await self._post(
            "api/chat",
            {
                "model": self.chat_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        content = result.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty answer.")
        return content.strip()


def create_llm_client() -> RAGProvider:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "mistral":
        return MistralClient()
    raise ConfigurationError(
        f"Unsupported LLM_PROVIDER '{provider}'. Choose 'ollama' or 'mistral'."
    )


class VectorStore:
    def __init__(self) -> None:
        path = Path(os.getenv("CHROMA_PATH", "./data/chroma")).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def _check_embedding_dimension(self, embeddings: list[list[float]]) -> None:
        if not embeddings or self.collection.count() == 0:
            return
        stored = self.collection.get(limit=1, include=["embeddings"])["embeddings"]
        if stored is None or len(stored) == 0:
            return
        stored_dimension = len(stored[0])
        incoming_dimension = len(embeddings[0])
        if stored_dimension != incoming_dimension:
            raise EmbeddingModelMismatchError(
                f"The current vector index uses {stored_dimension}-dimension embeddings, "
                f"but the selected model produces {incoming_dimension}. Remove all "
                "indexed documents from the sidebar, then upload the original PDFs again "
                "to rebuild the index with the selected model."
            )

    async def add_document(
        self, filename: str, pages: list[tuple[int, str]], provider: RAGProvider
    ) -> tuple[str, int]:
        records: list[tuple[str, int, int, str]] = []
        for page_number, text in pages:
            for chunk_number, chunk in enumerate(split_text(text)):
                records.append((chunk, page_number, chunk_number, filename))
        if not records:
            raise ValueError("No readable text was found in the PDF.")

        if self.collection.count() == 0:
            self.client.delete_collection(name="pdf_chunks")
            self.collection = self.client.get_or_create_collection(
                name="pdf_chunks",
                metadata={"hnsw:space": "cosine"},
            )

        document_id = str(uuid.uuid4())
        embeddings = await provider.embed([record[0] for record in records])
        self._check_embedding_dimension(embeddings)
        ids = [f"{document_id}:{index}" for index in range(len(records))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[record[0] for record in records],
            metadatas=[
                {
                    "document_id": document_id,
                    "source": filename,
                    "page": page_number,
                    "chunk": chunk_number,
                }
                for _, page_number, chunk_number, filename in records
            ],
        )
        return document_id, len(records)

    async def search(self, query: str, provider: RAGProvider) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        query_embedding = (await provider.embed([query]))[0]
        self._check_embedding_dimension([query_embedding])
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(RETRIEVAL_COUNT, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {"text": text, "metadata": metadata, "distance": distance}
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]

    def remove_document(self, document_id: str) -> int:
        matches = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        ids = matches["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def documents(self) -> list[dict[str, Any]]:
        matches = self.collection.get(include=["metadatas"])
        by_id: dict[str, dict[str, Any]] = {}
        for metadata in matches["metadatas"] or []:
            document_id = metadata["document_id"]
            if document_id not in by_id:
                by_id[document_id] = {
                    "id": document_id,
                    "name": metadata["source"],
                    "chunks": 0,
                }
            by_id[document_id]["chunks"] += 1
        return list(by_id.values())


def split_text(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def extract_pdf_pages(content: bytes) -> list[tuple[int, str]]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("This file is not a valid PDF.") from exc
    if document.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported.")
    pages = [(index + 1, page.get_text("text")) for index, page in enumerate(document)]
    document.close()
    return [(number, text) for number, text in pages if text.strip()]
