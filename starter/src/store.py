"""Persist and search embeddings with PostgreSQL and pgvector."""
import json
import math
from typing import Any

import psycopg

from .models import Chunk


_TABLE = "public.day6_chunks"
_FILTER_COLUMNS = {"category": "metadata ->> 'category'"}


def _validated_vector(vector: list[float], dimensions: int) -> list[float]:
    if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
        raise ValueError(f"Vector must contain exactly {dimensions} values")

    values = []
    try:
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Vector values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Vector values must be finite")
            values.append(number)
    except OverflowError as exc:
        raise ValueError("Vector values must be finite numbers") from exc

    if not any(value != 0.0 for value in values):
        raise ValueError("Vector must not be the zero vector")
    return values


def _vector_literal(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"), allow_nan=False)


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings and validate the vector dimension."""
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must not be empty")
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or not 1 <= dimensions <= 16000
        ):
            raise ValueError("dimensions must be an integer from 1 to 16000")

        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the chunk table and reject an incompatible vector schema."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_TABLE} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding VECTOR({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT a.attname, format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute AS a
                    WHERE a.attrelid = %s::regclass
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    """,
                    (_TABLE,),
                )
                columns = dict(cursor.fetchall())

                embedding_type = columns.get("embedding")
                expected_type = f"vector({self.dimensions})"
                if embedding_type != expected_type:
                    raise RuntimeError(
                        f"{_TABLE}.embedding has type {embedding_type!r}; "
                        f"expected {expected_type}. Existing data was not changed."
                    )

                required_columns = {
                    "strategy",
                    "chunk_id",
                    "document_id",
                    "source",
                    "page",
                    "text",
                    "metadata",
                    "embedding",
                }
                missing = required_columns.difference(columns)
                if missing:
                    raise RuntimeError(
                        f"{_TABLE} is missing required columns: "
                        f"{', '.join(sorted(missing))}"
                    )

                cursor.execute(
                    """
                    SELECT array_agg(a.attname)
                    FROM pg_constraint AS c
                    CROSS JOIN LATERAL
                        unnest(c.conkey) WITH ORDINALITY AS key(attnum, position)
                    JOIN pg_attribute AS a
                      ON a.attrelid = c.conrelid
                     AND a.attnum = key.attnum
                    WHERE c.conrelid = %s::regclass
                      AND c.contype = 'p'
                    GROUP BY c.oid
                    """,
                    (_TABLE,),
                )
                primary_key = cursor.fetchone()
                if (
                    primary_key is None
                    or set(primary_key[0]) != {"strategy", "chunk_id"}
                ):
                    raise RuntimeError(
                        f"{_TABLE} must have a primary key on "
                        "(strategy, chunk_id)"
                    )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Replace submitted documents atomically within one strategy."""
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("strategy must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        prepared_rows = []
        document_ids = set()
        chunk_ids = set()
        for chunk, vector in zip(chunks, vectors):
            if not chunk.chunk_id or not chunk.document_id:
                raise ValueError("chunk_id and document_id must not be empty")
            if chunk.chunk_id in chunk_ids:
                raise ValueError("chunk_ids must be unique within an upsert")
            chunk_ids.add(chunk.chunk_id)
            document_ids.add(chunk.document_id)

            if not isinstance(chunk.metadata, dict):
                raise ValueError("chunk metadata must be a dictionary")
            try:
                metadata = json.dumps(
                    chunk.metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("chunk metadata must be valid JSON data") from exc

            checked_vector = _validated_vector(vector, self.dimensions)
            prepared_rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    metadata,
                    _vector_literal(checked_vector),
                )
            )

        insert_sql = f"""
            INSERT INTO {_TABLE} (
                strategy, chunk_id, document_id, source, page, text,
                metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"DELETE FROM {_TABLE} "
                    "WHERE strategy = %s AND document_id = %s",
                    [(strategy, document_id) for document_id in document_ids],
                )
                cursor.executemany(insert_sql, prepared_rows)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact cosine-nearest chunks after applying metadata filters."""
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("strategy must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        checked_vector = _validated_vector(query_vector, self.dimensions)
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")

        conditions = ["strategy = %s"]
        parameters: list[Any] = [
            _vector_literal(checked_vector),
            strategy,
        ]
        for field, value in filters.items():
            column = _FILTER_COLUMNS.get(field)
            if column is None:
                raise ValueError(f"Unsupported metadata filter: {field}")
            if not isinstance(value, str):
                raise ValueError(f"Filter {field!r} must have a string value")
            conditions.append(f"{column} = %s")
            parameters.append(value)

        parameters.extend((_vector_literal(checked_vector), top_k))
        query = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {_TABLE}
            WHERE {" AND ".join(conditions)}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
