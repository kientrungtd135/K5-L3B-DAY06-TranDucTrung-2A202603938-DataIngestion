"""Create embeddings through the configured OpenAI-compatible API."""
import math
from pathlib import Path
from typing import Protocol

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Call the configured embedding API without a provider fallback."""

    _MAX_BATCH_SIZE = 128

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        env_path = Path(__file__).resolve().parents[1] / ".env"
        config = dotenv_values(env_path)
        api_key = config.get("OPENAI_API_KEY")
        base_url = config.get("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from starter/.env")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is missing from starter/.env")

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._dimensions: int | None = None
        self.api_call_count = 0
        self.usage_by_request: list[dict[str, int] | None] = []

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._embed_batch(["embedding dimension check"])
        assert self._dimensions is not None
        return self._dimensions

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.api_call_count += 1
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
        except OpenAIError as exc:
            status_code = getattr(exc, "status_code", None)
            status = f", HTTP {status_code}" if status_code is not None else ""
            raise RuntimeError(
                f"Embedding API request failed ({type(exc).__name__}{status})"
            ) from None

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and isinstance(total_tokens, int)
            and not isinstance(total_tokens, bool)
        ):
            self.usage_by_request.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                }
            )
        else:
            self.usage_by_request.append(None)

        if len(response.data) != len(texts):
            raise ValueError(
                "Embedding API returned an unexpected number of vectors"
            )

        ordered: list[list[float] | None] = [None] * len(texts)
        dimensions = self._dimensions
        for item in response.data:
            index = item.index
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(texts)
            ):
                raise ValueError("Embedding API returned an invalid vector index")
            if ordered[index] is not None:
                raise ValueError("Embedding API returned a duplicate vector index")

            vector = [float(value) for value in item.embedding]
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError(
                    "Embedding API returned an empty or non-finite vector"
                )
            if not any(value != 0.0 for value in vector):
                raise ValueError("Embedding API returned a zero vector")
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError(
                    "Embedding API returned inconsistent vector dimensions"
                )
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise ValueError("Embedding API omitted one or more vector indexes")

        result = [vector for vector in ordered if vector is not None]
        self._dimensions = dimensions
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = []
        for start in range(0, len(texts), self._MAX_BATCH_SIZE):
            batch = texts[start:start + self._MAX_BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._embed_batch([question])[0]
